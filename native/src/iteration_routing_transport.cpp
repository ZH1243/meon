#include "iteration_routing_internal.hpp"

#include "iteration_metadata_exchange.hpp"

#include <algorithm>
#include <stdexcept>

namespace rdma_proxy::iteration_metadata {
Layout make_layout(const ProxyConfig& c) {
    if (c.num_nodes < 1 || c.num_nodes > 2 || c.num_gpus_per_node < 2 || c.num_gpus_per_node > 8 ||
        c.router_top_k <= 0 || c.router_num_experts <= 0 ||
        c.router_top_k > c.router_num_experts || c.router_num_experts % (c.num_nodes * c.num_gpus_per_node) ||
        c.num_tokens == 0 || c.num_tokens > INT32_MAX / static_cast<std::size_t>(c.router_top_k))
        throw std::invalid_argument("unsupported iteration metadata geometry");
    Layout l{c.num_nodes, c.num_gpus_per_node, c.router_num_experts /
        (c.num_nodes * c.num_gpus_per_node), static_cast<int>(c.num_tokens), c.router_top_k,
        c.router_weighted_contributions_enabled, 0, 0, 0, 0, 0};
    const auto slots = static_cast<std::size_t>(l.nodes * l.gpus);
    l.directory = (slots + 1) * sizeof(Header);
    const auto record = expert_bytes(l, c.num_tokens * std::min(l.experts, l.top_k));
    if (record > UINT32_MAX || x3_bytes(c.num_tokens) > UINT32_MAX)
        throw std::invalid_argument("per-iteration metadata record exceeds one RDMA write");
    l.stride = (record + 7) & ~std::size_t{7};
    l.x3_offset = l.directory + slots * l.stride;
    l.control_offset = (l.x3_offset + x3_bytes(c.num_tokens) + 7) & ~std::size_t{7};
    l.bytes = l.control_offset + sizeof(uint64_t);
    return l;
}
}

#if RDMA_PROXY_HAVE_CUDA && defined(__linux__)
#include "rdma_connection.hpp"
#include "rdma_metadata_qp.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace rdma_proxy {
namespace {
namespace wire = iteration_metadata;
using Clock = std::chrono::steady_clock;

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

struct alignas(64) Producer {
    cudaIpcMemHandle_t handle;
    uint64_t phase, aborted, closed;
};

uint64_t load(const uint64_t& value) { return __atomic_load_n(&value, __ATOMIC_ACQUIRE); }
void store(uint64_t& value, uint64_t next) { __atomic_store_n(&value, next, __ATOMIC_RELEASE); }

class Relay final : public IterationMetadataTransport {
public:
    Relay(ProxyConfig config, const std::atomic<bool>& stopping)
        : c_(std::move(config)), stopping_(stopping), l_(wire::make_layout(c_)), control_(l_.gpus, nullptr) {}
    void initialize() override;
    void exchange(uint64_t generation, IterationRouteData&) override;
    void finish(uint64_t generation) override;
    void close(bool clean) override;

private:
    ProxyConfig c_;
    const std::atomic<bool>& stopping_;
    const wire::Layout l_;
    std::vector<Producer*> control_;
    std::string own_name_;
    wire::PeerArenas arenas_{};
    char *gpu_ = nullptr, *host_ = nullptr;
    float* weights_ = nullptr;
    std::vector<float> packed_weights_; // Source of the async H2D, lives until stream drain.
    cudaStream_t stream_ = nullptr;
    cudaEvent_t done_ = nullptr;
    std::unique_ptr<RdmaContext> context_;
    std::unique_ptr<MetadataQp> qp_;
    std::vector<Completion> completions_;
    MemoryRegionInfo send_mr_{}, recv_mr_{}, remote_{};
    bool exposed_ = false, initialized_ = false, closed_ = false;
    uint64_t phase_ = 0;
    Producer* own() const { return control_[c_.local_gpu_index]; }
    std::size_t offset(int slot) const { return l_.directory + slot * l_.stride; }
    void progress(Clock::time_point start, bool retiring = false) const;
    void barrier();
    void wait_stream();
    std::string signature() const;
    void open_control(int rank, const std::string& name, bool create);
    void discover(const std::string& rdma_descriptor);
    void remote_barrier(uint64_t generation);
    void transfer(uint64_t generation);
    void install(uint64_t generation, IterationRouteData& result);
    void validate(const wire::Header& h, uint64_t sequence, int source, int destination, bool x3) const;
};

void Relay::progress(Clock::time_point start, bool retiring) const {
    if (!retiring) {
        if (stopping_.load()) throw std::runtime_error("iteration metadata cancelled");
        for (const auto* slot : control_) if (slot && load(slot->aborted))
            throw std::runtime_error("iteration metadata aborted by local peer");
    }
    if (Clock::now() - start >= std::chrono::milliseconds(c_.completion_timeout_ms))
        throw std::runtime_error("iteration metadata completion timed out");
    std::this_thread::yield();
}

void Relay::barrier() {
    store(own()->phase, ++phase_);
    const auto start = Clock::now();
    for (const auto slot : control_) while (load(slot->phase) < phase_) progress(start);
}

void Relay::wait_stream() {
    check(cudaEventRecord(done_, stream_), "record metadata publication");
    const auto start = Clock::now();
    for (;;) {
        const auto status = cudaEventQuery(done_);
        if (status == cudaSuccess) return;
        if (status != cudaErrorNotReady) check(status, "poll metadata stream");
        progress(start);
    }
}

void Relay::open_control(int rank, const std::string& name, bool create) {
    const int fd = shm_open(name.c_str(), O_RDWR | (create ? O_CREAT | O_EXCL : 0), 0600);
    if (fd < 0) throw std::runtime_error("cannot open metadata IPC control " + name);
    if (create) own_name_ = name;
    if (create && ftruncate(fd, sizeof(Producer)) != 0) {
        ::close(fd);
        throw std::runtime_error("cannot size metadata IPC control");
    }
    struct stat status{};
    if (fstat(fd, &status) || status.st_size != static_cast<off_t>(sizeof(Producer))) {
        ::close(fd);
        throw std::runtime_error("metadata IPC control has an invalid size");
    }
    void* p = mmap(nullptr, sizeof(Producer), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) throw std::runtime_error("cannot map metadata IPC control");
    control_[rank] = static_cast<Producer*>(p);
    if (create) std::memset(p, 0, sizeof(Producer));
}

std::string Relay::signature() const {
    std::ostringstream out;
    out << "meon_iter_v1/" << l_.nodes << '/' << l_.gpus << '/' << l_.tokens << '/'
        << l_.experts << '/' << l_.top_k << '/' << l_.weighted << '/' << c_.router_seed
        << '/' << c_.token_dimension << '/' << dtype_size(c_.dtype) << '/' << c_.tokens_per_chunk;
    return out.str();
}

void Relay::discover(const std::string& rdma_descriptor) {
    ConnectionManager control(c_);
    const int g = c_.local_gpu_index;
    const auto port = [&](int gpu) { return static_cast<uint16_t>(c_.router_metadata_port_base + gpu); };
    const auto timeout = c_.completion_timeout_ms;
    const auto peers = static_cast<std::size_t>(l_.gpus - 1 + (l_.nodes == 2));
    auto incoming = std::async(std::launch::async, [&] {
        return control.receive_global_control_messages(port(g), peers, 65536, timeout);
    });
    const auto prefix = signature() + " " + std::to_string(c_.node_rank) + " " + std::to_string(g) + " ";
    exposed_ = true;
    for (int i = 0; i < l_.gpus; ++i) if (g != i) {
        control.send_global_control_message({c_.node_rank, "::1", port(i)}, prefix + own_name_, timeout);
    }
    if (l_.nodes == 2) {
        auto peer = c_.peers.at(0);
        peer.port = port(g);
        control.send_global_control_message(peer, prefix + rdma_descriptor, timeout);
    }
    for (const auto& payload : incoming.get()) {
        std::istringstream input(payload);
        int node, gpu;
        std::string shape;
        if (!(input >> shape >> node >> gpu) || shape != signature() || gpu < 0 || gpu >= l_.gpus)
            throw std::runtime_error("iteration metadata peer configuration mismatch");
        std::string value;
        std::getline(input >> std::ws, value, '\0');
        if (node == c_.node_rank && gpu != g && !control_[gpu]) {
            if (value.rfind("/meon_iter_", 0) != 0 || value.find('/', 1) != std::string::npos)
                throw std::runtime_error("invalid metadata IPC control name");
            open_control(gpu, value, false);
            check(cudaIpcOpenMemHandle(reinterpret_cast<void**>(&arenas_.address[gpu]),
                control_[gpu]->handle, cudaIpcMemLazyEnablePeerAccess), "import metadata GPU arena");
        } else if (qp_ && node == 1 - c_.node_rank && gpu == g && !remote_.addr) {
            const auto info = deserialize_peer_info(value);
            if (info.node_rank != node || info.gpu_index != g || info.qps.size() != 1 ||
                !info.recv_buffer.addr || info.recv_buffer.length != l_.bytes)
                throw std::runtime_error("metadata QP peer arena mismatch");
            remote_ = info.recv_buffer;
            qp_->connect(info.qps.front());
        } else throw std::runtime_error("duplicate or unexpected metadata setup peer");
    }
}

void Relay::initialize() {
    check(cudaSetDevice(c_.cuda_device_id), "select metadata worker GPU");
    check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create metadata stream");
    check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming), "create metadata completion event");
    check(cudaMalloc(&gpu_, l_.bytes), "allocate metadata GPU relay");
    check(cudaMallocHost(&host_, l_.bytes), "allocate metadata pinned receive records");
    check(cudaMemsetAsync(gpu_ + l_.control_offset, 0, sizeof(uint64_t), stream_), "clear metadata marker");
    if (l_.weighted)
        check(cudaMalloc(&weights_, c_.num_tokens * l_.top_k * sizeof(float)), "allocate metadata weights");
    check(cudaStreamSynchronize(stream_), "initialize metadata relay");
    const auto g = c_.local_gpu_index;
    const auto name = "/meon_iter_" + std::to_string(getpid()) + "_" +
        std::to_string(Clock::now().time_since_epoch().count()) + "_" + std::to_string(g);
    open_control(g, name, true);
    check(cudaIpcGetMemHandle(&own()->handle, gpu_), "export metadata relay");
    arenas_.address[g] = gpu_;
    std::string descriptor;
    if (l_.nodes == 2) {
        unsigned sync = 1;
        if (cuPointerSetAttribute(&sync, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                reinterpret_cast<CUdeviceptr>(gpu_)) != CUDA_SUCCESS)
            throw std::runtime_error("metadata relay requires GPUDirect SYNC_MEMOPS");
        context_ = std::make_unique<RdmaContext>(c_);
        context_->initialize();
        send_mr_ = context_->register_memory(gpu_, l_.bytes, "iteration metadata relay HBM");
        recv_mr_ = context_->register_memory(host_, l_.bytes, "iteration metadata receive DRAM");
        qp_ = std::make_unique<MetadataQp>(*context_, c_);
        completions_.reserve(16);
        PeerConnectionInfo info;
        info.node_rank = c_.node_rank;
        info.gpu_index = g;
        info.recv_buffer = recv_mr_;
        info.qps.push_back(qp_->local_info());
        descriptor = serialize_peer_info(info);
    }
    discover(descriptor);
    barrier();
    shm_unlink(own_name_.c_str());
    own_name_.clear();
    initialized_ = true;
}

void Relay::validate(const wire::Header& h, uint64_t sequence, int source, int destination, bool x3) const {
    const auto bound = x3 ? c_.num_tokens : c_.num_tokens * std::min(l_.experts, l_.top_k);
    if (h.version != 1 || h.sequence != sequence || h.source != source || h.destination != destination ||
        h.count < 0 || static_cast<std::size_t>(h.count) > bound ||
        h.experts != (x3 ? 0 : l_.experts) || h.weighted != (x3 ? 0 : l_.weighted))
        throw std::runtime_error("iteration routing record identity, epoch or length mismatch");
}

void Relay::remote_barrier(uint64_t sequence) {
    if (!qp_) return;
    // No outstanding receive WRs from the payload phase. Its final CQEs have
    // been consumed before we arm this one marker, so a fast peer cannot
    // overwrite pinned records that the application is still copying.
    constexpr uint32_t marker = UINT32_MAX;
    qp_->post_receive(sequence);
    qp_->post(sequence, send_mr_.addr + l_.control_offset, send_mr_.lkey,
        remote_.addr + l_.control_offset, remote_.rkey, sizeof(uint64_t), marker);
    bool sent = false, received = false;
    const auto start = Clock::now();
    while (!sent || !received) {
        progress(start);
        completions_.clear();
        qp_->poll(completions_);
        for (const auto& c : completions_) {
            if (c.wr_id != sequence) throw std::runtime_error("stale metadata barrier CQE");
            if (c.kind == CompletionKind::kSend && !sent) sent = true;
            else if (c.kind == CompletionKind::kRecvWithImmediate && c.imm_data == marker &&
                     c.byte_len == sizeof(uint64_t) && !received) received = true;
            else throw std::runtime_error("unexpected metadata barrier completion");
        }
    }
}

void Relay::transfer(uint64_t sequence) {
    if (!qp_) return;
    const int g = c_.local_gpu_index, node = c_.node_rank, other = 1 - node;
    const auto* directory = reinterpret_cast<const wire::Header*>(host_);
    // Dedicated QP, one receive WQE per original source plus same-index x3/x4.
    // Credits are posted only after all parties finish the preceding generation.
    for (int i = 0; i <= l_.gpus; ++i) qp_->post_receive(sequence);
    for (int i = 0; i <= l_.gpus; ++i) {
        const bool x3 = i == l_.gpus;
        const auto from = x3 ? l_.x3_offset : offset(other * l_.gpus + i);
        const auto to = x3 ? l_.x3_offset : offset(node * l_.gpus + i);
        const auto& h = directory[x3 ? l_.nodes * l_.gpus : other * l_.gpus + i];
        validate(h, sequence, node * l_.gpus + (x3 ? g : i), other * l_.gpus + g, x3);
        const auto bytes = x3 ? wire::x3_bytes(h.count) : wire::expert_bytes(l_, h.count);
        qp_->post(i, send_mr_.addr + from, send_mr_.lkey, remote_.addr + to, remote_.rkey,
                  static_cast<uint32_t>(bytes), i);
    }
    unsigned sent = 0, received = 0;
    const unsigned all = (1u << (l_.gpus + 1)) - 1;
    const auto start = Clock::now();
    while (sent != all || received != all) {
        progress(start);
        completions_.clear();
        qp_->poll(completions_);
        for (const auto& completion : completions_) {
            if (completion.kind == CompletionKind::kSend) {
                if (completion.wr_id > unsigned(l_.gpus) || (sent & (1u << completion.wr_id)))
                    throw std::runtime_error("duplicate or invalid metadata write completion");
                sent |= 1u << completion.wr_id;
            } else {
                const auto i = completion.imm_data;
                if (completion.wr_id != sequence || i > unsigned(l_.gpus) || (received & (1u << i)))
                    throw std::runtime_error("duplicate or stale metadata receive");
                const bool x3 = i == unsigned(l_.gpus);
                const auto address = x3 ? l_.x3_offset : offset(other * l_.gpus + i);
                const auto& h = *reinterpret_cast<const wire::Header*>(host_ + address);
                validate(h, sequence, other * l_.gpus + (x3 ? g : static_cast<int>(i)), node * l_.gpus + g, x3);
                if (completion.byte_len != (x3 ? wire::x3_bytes(h.count) : wire::expert_bytes(l_, h.count)))
                    throw std::runtime_error("truncated or oversized RDMA metadata record");
                received |= 1u << i;
            }
        }
    }
}

void Relay::install(uint64_t sequence, IterationRouteData& result) {
    const int g = c_.local_gpu_index, node = c_.node_rank, dest = node * l_.gpus + g;
    for (int source = 0; source < l_.nodes * l_.gpus; ++source) {
        const auto* h = reinterpret_cast<const wire::Header*>(host_ + offset(source));
        validate(*h, sequence, source, dest, false);
        auto& record = result.incoming_experts.emplace_back();
        record.source_node_rank = source / l_.gpus;
        record.source_gpu_index = source % l_.gpus;
        record.destination_node_rank = node;
        record.destination_gpu_index = g;
        record.num_nodes = l_.nodes;
        record.num_gpus_per_node = l_.gpus;
        record.num_experts = c_.router_num_experts;
        record.experts_per_gpu = l_.experts;
        record.first_global_expert = dest * l_.experts;
        record.num_tokens = c_.num_tokens;
        record.top_k = l_.top_k;
        record.weighted_contributions = l_.weighted;
        const auto* offsets = reinterpret_cast<const int32_t*>(h + 1);
        const auto* indices = offsets + l_.experts + 1;
        if (offsets[0] || offsets[l_.experts] != h->count || !std::is_sorted(offsets, indices))
            throw std::runtime_error("invalid normalized expert offsets");
        record.expert_offsets.assign(offsets, indices);
        record.expert_token_indices.assign(indices, indices + h->count);
        if (l_.weighted) {
            const auto* weights = reinterpret_cast<const float*>(indices + h->count);
            record.expert_route_weights.assign(weights, weights + h->count);
        }
    }
    if (l_.nodes == 2) {
        const int other = 1 - node;
        const auto* h = reinterpret_cast<const wire::Header*>(host_ + l_.x3_offset);
        validate(*h, sequence, other * l_.gpus + g, dest, true);
        auto record = iteration_node_record(c_, other, node);
        const auto* ids = reinterpret_cast<const int32_t*>(h + 1);
        const auto* masks = reinterpret_cast<const uint8_t*>(ids + h->count);
        for (int i = 0; i < h->count; ++i) {
            if (ids[i] < 0 || ids[i] >= l_.tokens || masks[i] == 0 || (masks[i] >> l_.gpus) ||
                (i && (masks[i] > masks[i-1] || (masks[i] == masks[i-1] && ids[i] <= ids[i-1]))))
                throw std::runtime_error("invalid same-index x3/x4 record");
        }
        record.token_indices.assign(ids, ids + h->count);
        record.token_masks.assign(masks, masks + h->count);
        result.incoming_nodes.emplace(other, std::move(record));
    }
}

void Relay::exchange(uint64_t sequence, IterationRouteData& result) {
    const auto tables = result.source->device_tables();
    // Weight lookup is by list position, with a separate entry for each
    // expert assignment even when several experts share the same token.
    if (l_.weighted) {
        packed_weights_.clear();
        packed_weights_.reserve(c_.num_tokens * l_.top_k);
        for (int node = 0; node < l_.nodes; ++node) {
            for (int g = 0; g < l_.gpus; ++g) {
                const auto& weights = result.source->expert_metadata_for_gpu(node, g).expert_route_weights;
                packed_weights_.insert(packed_weights_.end(), weights.begin(), weights.end());
            }
        }
        if (packed_weights_.size() != c_.num_tokens * l_.top_k)
            throw std::runtime_error("iteration weight count differs from number of routes");
        check(cudaMemcpyAsync(weights_, packed_weights_.data(), packed_weights_.size() * sizeof(float),
            cudaMemcpyHostToDevice, stream_), "upload packed iteration weights");
    }
    wire::publish(l_, arenas_, c_.node_rank, c_.local_gpu_index, sequence,
        tables.expert_offsets, tables.expert_indices, weights_, tables.node_offsets,
        tables.x3, tables.x4, stream_);
    wait_stream();
    barrier(); // Every source drained its writes to all peer GPU arenas.
    check(cudaMemcpyAsync(host_, gpu_, l_.directory, cudaMemcpyDeviceToHost, stream_), "read metadata directory");
    wait_stream();
    const auto* directory = reinterpret_cast<const wire::Header*>(host_);
    const int node = c_.node_rank;
    for (int gpu = 0; gpu < l_.gpus; ++gpu) {
        const int slot = node * l_.gpus + gpu;
        const auto& h = directory[slot];
        validate(h, sequence, slot, node * l_.gpus + c_.local_gpu_index, false);
        const auto start = offset(slot);
        check(cudaMemcpyAsync(host_ + start, gpu_ + start, wire::expert_bytes(l_, h.count),
            cudaMemcpyDeviceToHost, stream_), "read local expert record");
    }
    // Disjoint local/remote banks allow local D2H to run during the writes.
    transfer(sequence);
    wait_stream();
    install(sequence, result);
}

void Relay::finish(uint64_t generation) {
    barrier();
    remote_barrier(generation);
    barrier();
}

void Relay::close(bool clean) {
    if (closed_) return;
    if (!clean && own()) store(own()->aborted, 1);
    // The owning worker has stopped submissions. Retire own DMA before closing
    // any IPC import, then release the exported HBM only if all importers closed.
    if (stream_) check(cudaStreamSynchronize(stream_), "drain metadata stream");
    qp_.reset();
    context_.reset();
    for (int gpu = 0; gpu < l_.gpus; ++gpu) {
        if (gpu == c_.local_gpu_index || !arenas_.address[gpu]) continue;
        check(cudaIpcCloseMemHandle(arenas_.address[gpu]), "close metadata IPC import");
        arenas_.address[gpu] = nullptr;
    }
    if (own()) store(own()->closed, 1);
    bool safe = !exposed_;
    if (initialized_ && clean) {
        const auto start = Clock::now();
        for (const auto slot : control_) while (!load(slot->closed)) progress(start, true);
        safe = true;
    }
    if (done_) cudaEventDestroy(done_);
    if (stream_) cudaStreamDestroy(stream_);
    if (weights_) cudaFree(weights_);
    if (safe && gpu_) cudaFree(gpu_);
    if (host_) cudaFreeHost(host_);
    if (!own_name_.empty()) shm_unlink(own_name_.c_str());
    for (auto* slot : control_) if (slot) munmap(slot, sizeof(Producer));
    closed_ = true;
}

}
std::unique_ptr<IterationMetadataTransport> make_iteration_metadata_transport(
        const ProxyConfig& c, const std::atomic<bool>& stopping) {
    return std::make_unique<Relay>(c, stopping);
}
}
#else
namespace rdma_proxy {
std::unique_ptr<IterationMetadataTransport> make_iteration_metadata_transport(
        const ProxyConfig&, const std::atomic<bool>&) {
    throw std::runtime_error("GPU iteration metadata exchange requires Linux and CUDA");
}
}
#endif
