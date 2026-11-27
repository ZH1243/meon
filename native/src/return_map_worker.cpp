#include "return_map_worker.hpp"

#include "cuda_buffers.hpp"
#include "logging.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif

namespace rdma_proxy {
namespace {

std::size_t checked_table_words(std::uint64_t rows, std::uint64_t stride) {
    if (!rows || rows > std::numeric_limits<std::size_t>::max() / sizeof(int32_t) / stride)
        throw std::runtime_error("return map table exceeds addressable storage");
    return static_cast<std::size_t>(rows * stride);
}

#if RDMA_PROXY_HAVE_CUDA
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#endif

}  // namespace

std::size_t intra_return_map_words(std::size_t nodes, std::uint64_t capacity, int peers) {
    if (!nodes || peers <= 0 || peers > 8 || capacity > std::numeric_limits<int32_t>::max() / nodes ||
        nodes > std::numeric_limits<std::size_t>::max() / static_cast<unsigned>(peers))
        throw std::runtime_error("invalid intra-node combine dimensions");
    return checked_table_words(nodes * peers, nodes * capacity + 2);
}

std::size_t inter_return_map_words(std::size_t nodes, std::uint64_t capacity) {
    if (capacity > std::numeric_limits<int32_t>::max())
        throw std::runtime_error("invalid inter-node combine dimensions");
    return checked_table_words(nodes, capacity + 2);
}

void fill_intra_return_rows(const std::vector<TokenMaskView>& domains, std::size_t capacity,
                            int peers, int local_gpu, int32_t* rows) {
    const auto stride = domains.size() * capacity + 2;
    for (std::size_t d = 0; d < domains.size(); ++d) {
        const auto masks = domains[d];
        for (int peer = 0; peer < peers; ++peer) {
            auto* row = rows + (d * peers + peer) * stride;
            auto* live = row + d * capacity;
            const unsigned bit = 1U << ((local_gpu - peer + peers) % peers);
            std::fill(row, live, -1);
            int32_t selected = 0;
            for (std::size_t i = 0; i < masks.size(); ++i) {
                const bool hit = (masks[i] & bit) != 0;
                live[i] = hit ? selected : -1;
                selected += hit;
            }
            std::fill(live + masks.size(), row + stride - 2, -1);
            row[stride - 2] = selected;
            row[stride - 1] = static_cast<int32_t>(masks.size());
        }
    }
}

void fill_inter_return_rows(const std::vector<TokenIndexView>& domains, std::size_t capacity, int32_t* rows) {
    const auto stride = capacity + 2;
    for (std::size_t d = 0; d < domains.size(); ++d) {
        const auto tokens = domains[d];
        auto* row = rows + d * stride;
        std::fill_n(row, capacity, -1);
        for (std::size_t ordinal = 0; ordinal < tokens.size(); ++ordinal) {
            const auto token = tokens[ordinal];
            if (token >= capacity || row[token] != -1)
                throw std::runtime_error("invalid original token in inter-node combine route");
            row[token] = static_cast<int32_t>(ordinal);
        }
        row[capacity] = static_cast<int32_t>(tokens.size());
        row[capacity + 1] = 0;
    }
}

ReturnMapWorker::ReturnMapWorker(std::size_t nodes, std::size_t capacity, int peers, int local_gpu,
                                 bool include_inter, int32_t* pinned, uintptr_t device_slab,
                                 std::size_t words, int device, bool mock, void* publication_stream,
                                 std::atomic<bool>& down_metadata_failed)
    : nodes_(nodes), capacity_(capacity), intra_words_(intra_return_map_words(nodes, capacity, peers)),
      words_(words), peers_(peers), local_gpu_(local_gpu), device_(device), include_inter_(include_inter),
      mock_(mock), pinned_(pinned), destination_(reinterpret_cast<void*>(device_slab)), stream_(publication_stream),
      down_metadata_failed_(down_metadata_failed) {
    const auto inter_words = include_inter ? inter_return_map_words(nodes, capacity) : 0;
    if (local_gpu < 0 || local_gpu >= peers || !pinned || !device_slab || device_slab % alignof(int32_t) ||
        words > std::numeric_limits<std::size_t>::max() / sizeof(int32_t) ||
        words < intra_words_ || words - intra_words_ != inter_words || (!mock && !stream_))
        throw std::runtime_error("invalid packed return map reservation");
    select_cuda_device_for_thread(device_, mock_);
#if RDMA_PROXY_HAVE_CUDA
    if (!mock_) {
        cudaEvent_t event{};
        check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "create return map event");
        event_ = event;
    }
#endif
    try {
        thread_ = std::thread(&ReturnMapWorker::run, this);
    } catch (...) {
#if RDMA_PROXY_HAVE_CUDA
        if (event_) cudaEventDestroy(static_cast<cudaEvent_t>(event_));
#endif
        throw;
    }
}

ReturnMapWorker::~ReturnMapWorker() {
    try { stop(); }
    catch (const std::exception& e) { RDMA_PROXY_LOG_WARN("return map worker shutdown: ", e.what()); }
#if RDMA_PROXY_HAVE_CUDA
    if (event_) cudaEventDestroy(static_cast<cudaEvent_t>(event_));
#endif
}

void ReturnMapWorker::start(uint64_t iteration, ReturnMapRoute route, bool final_gather_submitted) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_) std::rethrow_exception(error_);
    if (stopping_ || busy_ || phase_ != Phase::idle || dma_pending_ || (iteration_ && *iteration_ >= iteration) ||
        route.masks.size() != nodes_ || route.tokens.size() != (include_inter_ ? nodes_ : 0))
        throw std::runtime_error("invalid return map job or iteration");
    for (const auto masks : route.masks) {
        if (masks.size() > capacity_ || (masks.size() && !masks.data()))
            throw std::runtime_error("invalid return map mask domain");
    }
    for (const auto tokens : route.tokens) {
        if (tokens.size() > capacity_) throw std::runtime_error("invalid return map token domain");
    }
    route_ = std::move(route);
    iteration_ = iteration;
    final_ = final_gather_submitted;
    phase_ = Phase::build;
    changed_.notify_all();
}

void ReturnMapWorker::final_gather(uint64_t iteration) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (phase_ != Phase::idle && iteration_ == iteration) {
        final_ = true;
        changed_.notify_all();
    }
}

void ReturnMapWorker::upload() {
    if (mock_) {
        std::memcpy(destination_, pinned_, words_ * sizeof(int32_t));
        return;
    }
#if RDMA_PROXY_HAVE_CUDA
    auto stream = static_cast<cudaStream_t>(stream_);
    check(cudaMemcpyAsync(destination_, pinned_, words_ * sizeof(int32_t), cudaMemcpyHostToDevice, stream),
          "upload packed return maps");
    check(cudaEventRecord(static_cast<cudaEvent_t>(event_), stream), "record return map upload");
#endif
}

void ReturnMapWorker::run() noexcept {
    try {
#if defined(__linux__)
        pthread_setname_np(pthread_self(), "rdma-return-map");
#elif defined(__APPLE__)
        pthread_setname_np("rdma-return-map");
#endif
        select_cuda_device_for_thread(device_, mock_);
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            changed_.wait(lock, [&] { return stopping_ || phase_ == Phase::build || (phase_ == Phase::packed && final_); });
            if (stopping_) return;
            const auto build = phase_ == Phase::build;
            busy_ = true;
            // Even if event recording fails following a successful copy, retirement must fence its DMA.
            if (!build) dma_pending_ = true;
            lock.unlock();
            if (build) {
                fill_intra_return_rows(route_.masks, capacity_, peers_, local_gpu_, pinned_);
                if (include_inter_) fill_inter_return_rows(route_.tokens, capacity_, pinned_ + intra_words_);
            } else {
                upload();
            }
            lock.lock();
            busy_ = false;
            if (phase_ != Phase::idle) phase_ = build ? Phase::packed : Phase::submitted;
            changed_.notify_all();
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        busy_ = false;
        error_ = std::current_exception();
        down_metadata_failed_.store(true, std::memory_order_release);
        changed_.notify_all();
    }
}

void ReturnMapWorker::wait(uint64_t iteration, uintptr_t consumer_stream, uint64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto active = [&] { return !stopping_ && phase_ != Phase::idle && iteration_ == iteration; };
    const auto max_wait = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::time_point::max() - std::chrono::steady_clock::now()).count();
    if (!active() || !consumer_stream || !timeout_ms || timeout_ms > static_cast<uint64_t>(max_wait))
        throw std::runtime_error("invalid return map wait epoch, stream or timeout");
    if (!changed_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                          [&] { return error_ || !active() || phase_ == Phase::submitted; }))
        throw std::runtime_error("timed out waiting for return maps for iteration " + std::to_string(iteration));
    if (error_) std::rethrow_exception(error_);
    if (!active()) throw std::runtime_error("return map job retired during wait");
#if RDMA_PROXY_HAVE_CUDA
    if (!mock_) {
        select_cuda_device_for_thread(device_, false);
        check(cudaStreamWaitEvent(reinterpret_cast<cudaStream_t>(consumer_stream), static_cast<cudaEvent_t>(event_), 0),
              "wait for packed return maps");
    }
#endif
}

void ReturnMapWorker::rethrow_error() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (error_) std::rethrow_exception(error_);
}

void ReturnMapWorker::retire() {
    std::unique_lock<std::mutex> lock(mutex_);
    phase_ = Phase::idle;
    changed_.notify_all();
    changed_.wait(lock, [&] { return !busy_; });
    route_ = {};
    if (dma_pending_) {
#if RDMA_PROXY_HAVE_CUDA
        if (!mock_) {
            select_cuda_device_for_thread(device_, false);
            // A failed event record can follow a successful DMA enqueue.
            if (error_) synchronize_cuda_stream(stream_, false);
            else check(cudaEventSynchronize(static_cast<cudaEvent_t>(event_)), "retire packed return maps");
        }
#endif
        dma_pending_ = false;
    }
    if (error_ && !stopping_) std::rethrow_exception(error_);
}

void ReturnMapWorker::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        phase_ = Phase::idle;
        changed_.notify_all();
    }
    if (thread_.joinable()) thread_.join();
    retire();
}

}  // namespace rdma_proxy
