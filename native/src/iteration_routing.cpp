#include "iteration_routing_internal.hpp"
#include "cuda_buffers.hpp"

#include <condition_variable>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

namespace rdma_proxy {
namespace {
std::unique_ptr<RouterRouting> generate(const ProxyConfig& c, int node, int gpu, uint64_t iteration,
                                        std::shared_ptr<RouterRouting::Workspace> workspace = {}) {
    auto cfg = c;
    cfg.node_rank = node;
    cfg.local_gpu_index = gpu;
    const uint64_t world = uint64_t(c.num_nodes) * c.num_gpus_per_node;
    // generate_r hashes (seed ^ token). Adjacent seeds would only permute
    // rows within power-of-two token batches and keep expert loads identical.
    const uint64_t ordinal = iteration * world + uint64_t(node) * c.num_gpus_per_node + gpu;
    cfg.router_seed += 0x9e3779b97f4a7c15ULL * ordinal;
    auto source = std::make_unique<RouterRouting>(std::move(cfg), std::move(workspace));
    source->initialize();
    return source;
}
void mock_exchange(const ProxyConfig& c, uint64_t iteration, IterationRouteData& result) {
    for (int node = 0; node < c.num_nodes; ++node) {
        for (int gpu = 0; gpu < c.num_gpus_per_node; ++gpu) {
            std::unique_ptr<RouterRouting> peer;
            if (node != c.node_rank || gpu != c.local_gpu_index) peer = generate(c, node, gpu, iteration);
            const auto& source = peer ? *peer : *result.source;
            result.incoming_experts.push_back(source.expert_metadata_for_gpu(c.node_rank, c.local_gpu_index));
            if (node != c.node_rank && gpu == c.local_gpu_index) {
                auto record = iteration_node_record(c, node, c.node_rank);
                record.token_indices = source.token_indices_for_node(c.node_rank);
                record.token_masks = source.token_masks_for_node(c.node_rank);
                result.incoming_nodes.emplace(node, std::move(record));
            }
        }
    }
}
}

struct IterationRouting::Impl {
    explicit Impl(const ProxyConfig& config) : c(config) {}
    ProxyConfig c;
    std::mutex mutex;
    std::condition_variable changed;
    std::thread worker;
    std::atomic<bool> stopping{false};
    bool ready = false, pending = false, busy = false, finished = false;
    uint64_t next = 0;
    std::function<void(const RouterRouting&, void*)> stage;
    std::exception_ptr failure;
    std::optional<IterationRouteData> output;
    std::unique_ptr<IterationMetadataTransport> transport;

    void run() {
        void* staging_stream = nullptr;
        std::shared_ptr<RouterRouting::Workspace> workspace;
        try {
            workspace = RouterRouting::make_workspace(c);
            if (!c.mock_mode) {
                transport = make_iteration_metadata_transport(c, stopping);
                transport->initialize();
            }
            staging_stream = create_cuda_stream(c.cuda_device_id, true, c.mock_mode);
            std::unique_lock<std::mutex> lock(mutex);
            ready = true;
            changed.notify_all();
            while (true) {
                changed.wait(lock, [&] { return pending || stopping.load(); });
                if (stopping.load()) break;
                const auto iteration = next;
                auto gather = std::move(stage);
                pending = false;
                lock.unlock();
                IterationRouteData result;
                result.source = generate(c, c.node_rank, c.local_gpu_index, iteration, workspace);
                // Generation has completed before the gather reads x3. The
                // worker can drive metadata exchange while this stream packs rows.
                if (gather) gather(*result.source, staging_stream);
                if (transport) transport->exchange(iteration + 1, result);
                else mock_exchange(c, iteration, result);
                if (gather) synchronize_cuda_stream(staging_stream, c.mock_mode);
                if (transport) transport->finish(iteration + 1);
                lock.lock();
                output.emplace(std::move(result));
                ++next;
                changed.notify_all();
            }
            lock.unlock();
            if (transport) transport->close(true);
        } catch (...) {
            auto error = std::current_exception();
            if (staging_stream) {
                try { synchronize_cuda_stream(staging_stream, false); } catch (...) {}
            }
            if (transport) {
                try { transport->close(false); } catch (...) {}
            }
            std::lock_guard<std::mutex> lock(mutex);
            failure = error;
        }
        destroy_cuda_stream(staging_stream, c.mock_mode);
        std::lock_guard<std::mutex> lock(mutex);
        finished = true;
        changed.notify_all();
    }
};

IterationRouting::IterationRouting(const ProxyConfig& c) : impl_(std::make_unique<Impl>(c)) {}
IterationRouting::~IterationRouting() { try { shutdown(); } catch (...) {} }
void IterationRouting::initialize() {
    auto& p = *impl_;
    std::unique_lock<std::mutex> lock(p.mutex);
    if (p.worker.joinable() || p.stopping.load()) throw std::logic_error("iteration routing already initialized");
    p.worker = std::thread([&p] { p.run(); });
    p.changed.wait(lock, [&] { return p.ready || p.finished; });
    if (p.failure) std::rethrow_exception(p.failure);
    if (p.stopping) throw std::runtime_error("iteration routing stopped during initialization");
}
IterationRouteData IterationRouting::prepare(uint64_t iteration,
        const std::function<void(const RouterRouting&, void*)>& stage) {
    auto& p = *impl_;
    std::unique_lock<std::mutex> lock(p.mutex);
    if (p.failure) std::rethrow_exception(p.failure);
    if (!p.ready || p.stopping || p.busy || iteration != p.next || iteration >= UINT64_MAX / 8 - 1)
        throw std::logic_error("iteration routing requires one consecutive active generation");
    p.stage = stage;
    p.pending = p.busy = true;
    p.changed.notify_all();
    p.changed.wait(lock, [&] { return p.output || p.finished; });
    if (p.failure) std::rethrow_exception(p.failure);
    if (!p.output) throw std::runtime_error("iteration routing stopped");
    auto result = std::move(*p.output);
    p.output.reset();
    p.busy = false;
    return result;
}
void IterationRouting::shutdown() {
    auto& p = *impl_;
    {
        std::lock_guard<std::mutex> lock(p.mutex);
        p.stopping = true;
        p.changed.notify_all();
    }
    if (p.worker.joinable()) p.worker.join();
    if (p.failure) std::rethrow_exception(p.failure);
}
}
