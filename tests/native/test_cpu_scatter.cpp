#include "cuda_buffers.hpp"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <chrono>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rdma_proxy;
static void require(bool value) {
    if (!value) throw std::runtime_error("CPU scatter check failed");
}

template<class F> static void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected);
}
struct alignas(16) SlabBlock { unsigned char bytes[16]; };

// Test-only access: production exposes no callbacks or thread-control API.
namespace rdma_proxy {
struct CpuScatterTestAccess {
    static void wait_wave_packed(CudaBuffers& buffers) {
        auto& wave = buffers.down_wave_;
        std::unique_lock<std::mutex> lock(wave.mutex);
        require(wave.changed.wait_for(lock, std::chrono::seconds(5), [&] {
            return wave.error || wave.phase == CudaBuffers::WavePhase::packed;
        }));
        if (wave.error) std::rethrow_exception(wave.error);
    }
    static void fail_parked_wave_build(CudaBuffers& buffers) {
        // Feed invalid counts to the worker itself to exercise its catch path.
        // It is parked with no outstanding DMA and no final gather for this epoch.
        wait_wave_packed(buffers);
        auto& wave = buffers.down_wave_;
        std::unique_lock<std::mutex> lock(wave.mutex);
        require(!wave.final_gather && !wave.busy);
        wave.counts.front() = std::numeric_limits<std::size_t>::max();
        wave.phase = CudaBuffers::WavePhase::build;
        wave.changed.notify_all();
        require(wave.changed.wait_for(lock, std::chrono::seconds(5), [&] { return bool(wave.error); }));
    }
    static void require_no_batch_staging(const CudaBuffers& buffers) {
        require(!buffers.cpu_scatter_staging_.ptr && !buffers.cpu_route_weights_staging_.ptr);
    }
    static const CpuPinnedBuffer& slab(const CudaBuffers& buffers) {
        return buffers.cpu_scatter_host_slab_;
    }
    static void set_hook(CudaBuffers& buffers, std::function<void()> hook) {
        buffers.cpu_scatter_pack_test_hook_ = std::move(hook);
    }
    static void set_lookup_hook(CudaBuffers& buffers, std::function<void()> hook) {
        buffers.cpu_scatter_lookup_test_hook_ = std::move(hook);
    }
    static void require_full_bundle(CudaBuffers& buffers) {
        std::lock_guard<std::mutex> lock(buffers.cpu_scatter_worker_mutex_);
        const auto groups = buffers.router_notification_publication_buffers_.n_groups_per_m_cluster;
        require(groups != 0);
        require(buffers.cpu_scatter_worker_consumed_rows_ % groups == 0);
        require(buffers.cpu_scatter_worker_committed_rows_ >=
                buffers.cpu_scatter_worker_consumed_rows_ + groups);
    }
    static void drain(CudaBuffers& buffers, std::size_t prefix) {
        std::unique_lock<std::mutex> lock(buffers.cpu_scatter_worker_mutex_);
        const auto groups = buffers.router_notification_publication_buffers_.n_groups_per_m_cluster;
        prefix = prefix / groups * groups;
        require(buffers.cpu_scatter_worker_cv_.wait_for(lock, std::chrono::seconds(5), [&] {
            return buffers.cpu_scatter_worker_error_ || buffers.cpu_scatter_worker_stop_ ||
                (!buffers.cpu_scatter_worker_busy_ && buffers.cpu_scatter_worker_consumed_rows_ >= prefix);
        }));
        if (buffers.cpu_scatter_worker_error_) std::rethrow_exception(buffers.cpu_scatter_worker_error_);
        require(!buffers.cpu_scatter_worker_stop_);
    }
    static void wait_cancelled(CudaBuffers& buffers) {
        std::unique_lock<std::mutex> lock(buffers.cpu_scatter_worker_mutex_);
        require(buffers.cpu_scatter_worker_cv_.wait_for(lock, std::chrono::seconds(5), [&] {
            return buffers.cpu_scatter_worker_cancelled_ || buffers.cpu_scatter_worker_stop_;
        }));
    }
    static void require_empty_epoch(CudaBuffers& buffers, uint64_t epoch) {
        std::lock_guard<std::mutex> lock(buffers.cpu_scatter_worker_mutex_);
        require(buffers.cpu_scatter_worker_epoch_ == epoch);
        require(!buffers.cpu_scatter_worker_busy_ && buffers.cpu_scatter_worker_consumed_rows_ == 0 &&
                buffers.cpu_scatter_worker_staged_rows_ == 0 && buffers.cpu_scatter_worker_committed_rows_ == 0);
    }
};
}

struct PackingGate {
    std::mutex mutex;
    std::condition_variable cv;
    bool pause{false}, entered{false}, released{true}, fail{false};
    void arm(bool inject_error = false) {
        std::lock_guard<std::mutex> lock(mutex);
        pause = !inject_error;
        entered = false;
        released = false;
        fail = inject_error;
    }
    void operator()() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        cv.notify_all();
        if (fail) {
            fail = false;
            throw std::runtime_error("injected CPU scatter worker failure");
        }
        if (pause) {
            cv.wait(lock, [&] { return released; });
            pause = false;
        }
    }
    void wait_entered() {
        std::unique_lock<std::mutex> lock(mutex);
        require(cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }));
    }
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        cv.notify_all();
    }
    ~PackingGate() { release(); }
};
struct ReleasePackingGate {
    PackingGate& gate;
    ~ReleasePackingGate() { gate.release(); }
};
struct StopWorkerBeforeStorageRelease {
    std::unique_ptr<CudaBuffers>& buffers;
    PackingGate& gate;
    ~StopWorkerBeforeStorageRelease() {
        gate.release();
        if (buffers) {
            try { buffers->stop_cpu_scatter_worker(); }
            catch (...) { /* Preserve an existing test failure during cleanup. */ }
        }
    }
};

// publication_mode: legacy per-batch=0, deferred=1, dedicated CPU thread=2.
static void run(bool per_entry, uint64_t tile, bool cuda, int weight_mode,
                int publication_mode, bool memcpy_flag, int lifecycle = 0, bool pooled = false) {
    const bool deferred = publication_mode == 1;
    const bool threaded = publication_mode == 2;
    const bool single_upload = deferred || threaded;
    ProxyConfig config;
    config.mock_mode = !cuda;
    config.router_routing_enabled = true;
    config.router_weighted_contributions_enabled = weight_mode != 0;
    config.router_pooled_up_gemm_enabled = pooled;
    config.nvlink_forwarding_enabled = true;
    config.nvlink_forward_completion_notifications_enabled = true;
    config.nvlink_forward_notification_flush_per_entry_enabled = per_entry;
    config.nvlink_forward_notification_flag_update_mode = memcpy_flag ?
        NvlinkForwardNotificationFlagUpdateMode::kMemcpy : NvlinkForwardNotificationFlagUpdateMode::kStreamWrite;
    config.num_nodes = 1;
    config.num_gpus_per_node = 3;
    config.router_num_experts = 6;
    config.router_top_k = 2;
    config.num_tokens = 10;
    config.token_dimension = 256;
    config.dtype = DataType::kBF16;
    config.expert_gemm_m_tile = 2;
    config.expert_gemm_cluster_m = 1;
    config.expert_gemm_n_tile = 128;
    config.expert_gemm_dimension = 512;
    config.expert_gemm_max_swizzle_size = 2; // repeated N groups
    PackingGate packing_gate; // Must outlive every callback and worker join.
    auto buffers_owner = std::make_unique<CudaBuffers>(config);
    ReleasePackingGate release_before_buffers_cleanup{packing_gate};
    auto& buffers = *buffers_owner;
    buffers.initialize();
    const std::vector<std::vector<int32_t>> indices{{2,5,9,5,9}, {1,7,1,7}, {}};
    const std::vector<std::vector<int32_t>> offsets{{0,3,5}, {0,2,4}, {0,0,0}};
    std::vector<std::vector<float>> weights{{0.125f, 0.25f, 0.0f, 0.75f, 1.0f},
                                             {0.375f, 0.625f, 0.5f, 0.875f}, {}};
    if (weight_mode == 2) for (auto& source : weights) std::fill(source.begin(), source.end(), 1.0f);
    for (int source = 0; source < 3; ++source) {
        RouterExpertMetadata m;
        m.destination_node_rank = 0;
        m.destination_gpu_index = 0;
        m.source_node_rank = 0;
        m.source_gpu_index = source;
        m.num_nodes = 1;
        m.num_gpus_per_node = 3;
        m.num_experts = 6;
        m.experts_per_gpu = 2;
        m.first_global_expert = 0;
        m.num_tokens = 10;
        m.expert_token_indices = indices[source];
        m.expert_offsets = offsets[source];
        m.top_k = config.router_top_k;
        m.weighted_contributions = weight_mode != 0;
        if (weight_mode) m.expert_route_weights = weights[source];
        buffers.install_expert_metadata(m);
    }
    for (int source = 0; source < 3; ++source) {
        const auto& buffer = buffers.nvlink_receive_buffer_for_source(0, source);
        if (!weight_mode) require(buffer.expert_route_weights_device.ptr == nullptr);
        else {
            require(buffer.expert_route_weights_device.bytes == buffer.expert_token_indices_device.bytes);
            std::vector<float> copied(buffer.expert_route_weights_device.bytes / sizeof(float));
            if (!cuda) std::copy_n(static_cast<const float*>(buffer.expert_route_weights_device.ptr),
                                  copied.size(), copied.begin());
#if RDMA_PROXY_HAVE_CUDA
            else require(cudaMemcpy(copied.data(), buffer.expert_route_weights_device.ptr,
                                    buffer.expert_route_weights_device.bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
#endif
            require(std::equal(weights[source].begin(), weights[source].end(), copied.begin()));
            require(std::all_of(copied.begin() + weights[source].size(), copied.end(),
                                [](float value) { return value == 0; }));
        }
    }
    const std::size_t columns = tile ? 2 : 1;
    constexpr std::size_t routes = 9;
    std::vector<int64_t> scatter(routes*columns, -1), counts(3), inverse(30);
    std::vector<float> packed_weights(routes, -1.0f);
    const auto scatter_bytes = scatter.size() * sizeof(int64_t);
    const auto weight_offset = (scatter_bytes + 15) & ~std::size_t{15};
    const auto slab_bytes = weight_mode ? weight_offset + routes * sizeof(float) : scatter_bytes;
    // A deliberate 16-byte gap in legacy mode tests rejection of a misplaced
    // weight pointer without depending on allocator adjacency.
    const auto weight_gap = weight_mode && !single_upload ? std::size_t{16} : 0;
    const auto allocation_bytes = slab_bytes + weight_gap;
    std::vector<SlabBlock> slab((allocation_bytes + 15) / 16);
    void* compute_stream = create_cuda_stream(config.cuda_device_id, true, config.mock_mode);
    const auto compute_stream_address = cuda ? reinterpret_cast<uintptr_t>(compute_stream) : uintptr_t{1};
    void* device_weights = packed_weights.data();
    void* device_scatter = slab.data();
#if RDMA_PROXY_HAVE_CUDA
    if (cuda) {
        require(cudaMalloc(&device_scatter, allocation_bytes) == cudaSuccess);
    }
#endif
    if (weight_mode) device_weights = static_cast<unsigned char*>(device_scatter) + weight_offset + weight_gap;
    int32_t pairs[]{0,0,0,1,0,2};
    uint64_t bases[]{0x10000, 0x20000, 0x30000};
    RdmaProxyCpuScatterConfig setup{sizeof(setup), 3, routes, 10, 256, tile, 2,
        reinterpret_cast<uintptr_t>(device_scatter), pairs, bases, counts.data(), inverse.data()};
    auto bad = setup;
    bad.routes++;
    bool rejected = false;
    try { buffers.configure_cpu_scatter(bad); } catch (const std::exception&) { rejected = true; }
    require(rejected);
    buffers.configure_cpu_scatter(setup);
    if (weight_mode) {
        bool bad_count = false;
        try { buffers.configure_cpu_route_weights(reinterpret_cast<uintptr_t>(device_weights), routes + 1); }
        catch (const std::exception&) { bad_count = true; }
        require(bad_count);
        buffers.configure_cpu_route_weights(reinterpret_cast<uintptr_t>(device_weights), routes);
    }
    if (single_upload) {
        CpuScatterTestAccess::require_no_batch_staging(buffers);
        const auto configure = [&](std::size_t bytes) {
            if (threaded) buffers.configure_cpu_scatter_threaded(bytes);
            else buffers.configure_cpu_scatter_deferred(bytes);
        };
        rejects([&] { configure(slab_bytes - 1); });
        rejects([&] { configure(slab_bytes + 1); });
        configure(slab_bytes);
        rejects([&] { configure(slab_bytes); });
    } else if (weight_mode) {
        // A noncontiguous weight view cannot be passed as a single slab.
        rejects([&] { buffers.configure_cpu_scatter_deferred(slab_bytes); });
    }
    rejects([&] { buffers.wait_cpu_scatter_ready(0, compute_stream_address, 1); });
    require(counts == std::vector<int64_t>({3,2,0}));
    require(inverse == std::vector<int64_t>({2,5,9,-1,-1,-1,-1,-1,-1,-1,
                                          1,7,-1,-1,-1,-1,-1,-1,-1,-1,
                                          -1,-1,-1,-1,-1,-1,-1,-1,-1,-1}));
    const auto& publication = buffers.router_notification_publication_buffers();
    if (threaded) CpuScatterTestAccess::set_hook(buffers, [&] {
        CpuScatterTestAccess::require_full_bundle(buffers);
        packing_gate();
    });
    // Declared after borrowed slab storage so assertion unwinding joins the
    // packer before any of that storage is destroyed.
    StopWorkerBeforeStorageRelease stop_before_slab_cleanup{buffers_owner, packing_gate};
    std::vector<int64_t> first_iteration;
    for (uint64_t iteration = 0; iteration < 2; ++iteration) {
        const auto packed_before = publication.cpu_scatter_packed_rows.load();
        const auto pack_ready_before = publication.cpu_scatter_pack_ready_publication_count.load();
        buffers.begin_router_notification_iteration(iteration);
        require(publication.cpu_scatter_packed_rows == packed_before);
        require(publication.cpu_scatter_pack_ready_publication_count == pack_ready_before);
        const auto copies_before = publication.cpu_scatter_copy_count.load();
        const auto bytes_before = publication.cpu_scatter_copy_bytes.load();
        std::vector<int32_t> partial_ready_prefixes;
        bool final_publication_seen = false;
        uint64_t final_pack_ready_count = 0;
        std::fill(scatter.begin(), scatter.end(), -1);
        if (!cuda) std::memset(device_scatter, 0xff, allocation_bytes);
#if RDMA_PROXY_HAVE_CUDA
        if (cuda) {
            require(cudaMemset(device_scatter, 0xff, allocation_bytes) == cudaSuccess);
            require(cudaDeviceSynchronize() == cudaSuccess);
        }
#endif
        std::future<void> waiter;
        std::vector<int64_t> consumer_scatter(scatter.size(), -1);
        if (single_upload) {
            rejects([&] { buffers.wait_cpu_scatter_ready(iteration, 0, 1); });
            rejects([&] { buffers.wait_cpu_scatter_ready(iteration, compute_stream_address, 0); });
            rejects([&] { buffers.wait_cpu_scatter_ready(iteration + 1, compute_stream_address, 1); });
            if (iteration) rejects([&] { buffers.wait_cpu_scatter_ready(iteration - 1, compute_stream_address, 1); });
            rejects([&] { buffers.wait_cpu_scatter_ready(iteration, compute_stream_address, 1); });
            waiter = std::async(std::launch::async, [&] {
                buffers.wait_cpu_scatter_ready(iteration, compute_stream_address, 5000);
#if RDMA_PROXY_HAVE_CUDA
                if (cuda) {
                    require(cudaMemcpyAsync(consumer_scatter.data(), device_scatter, scatter_bytes,
                        cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(compute_stream)) == cudaSuccess);
                    require(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(compute_stream)) == cudaSuccess);
                } else
#endif
                    std::memcpy(consumer_scatter.data(), device_scatter, scatter_bytes);
            });
            // No event has been recorded yet: a device-only wait here would incorrectly finish.
            require(waiter.wait_for(std::chrono::milliseconds(2)) == std::future_status::timeout);
        }
        if (threaded) packing_gate.arm();
        const auto check_published = [&]() {
            int32_t ready = 0;
            auto* table = static_cast<const int32_t*>(publication.device_table.ptr);
            auto* flat = static_cast<const int32_t*>(publication.device_pooled_a_idx.ptr);
#if RDMA_PROXY_HAVE_CUDA
            std::vector<int32_t> table_copy, flat_copy;
#endif
            if (!cuda) {
                ready = *static_cast<const int32_t*>(publication.device_ready_rows.ptr);
                std::memcpy(scatter.data(), device_scatter, scatter_bytes);
                if (weight_mode) std::memcpy(packed_weights.data(), device_weights, routes * sizeof(float));
            }
#if RDMA_PROXY_HAVE_CUDA
            else {
                buffers.synchronize_router_notification_publication();
                table_copy.resize(publication.device_table.bytes / sizeof(int32_t));
                require(cudaMemcpy(&ready, publication.device_ready_rows.ptr, sizeof(ready),
                                   cudaMemcpyDeviceToHost) == cudaSuccess);
                require(cudaMemcpy(table_copy.data(), table, publication.device_table.bytes,
                                   cudaMemcpyDeviceToHost) == cudaSuccess);
                require(cudaMemcpy(scatter.data(), device_scatter, scatter.size()*sizeof(int64_t),
                                   cudaMemcpyDeviceToHost) == cudaSuccess);
                if (weight_mode)
                    require(cudaMemcpy(packed_weights.data(), device_weights, packed_weights.size()*sizeof(float),
                                       cudaMemcpyDeviceToHost) == cudaSuccess);
                table = table_copy.data();
                if (pooled) {
                    flat_copy.resize(routes);
                    require(cudaMemcpy(flat_copy.data(), flat, routes * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost) == cudaSuccess);
                    flat = flat_copy.data();
                }
            }
#endif
            require(ready == static_cast<int32_t>(
                buffers.router_computation_scheduler_state().published_rows));
            if (ready > 0 && static_cast<std::size_t>(ready) < publication.table_rows &&
                (partial_ready_prefixes.empty() || partial_ready_prefixes.back() != ready))
                partial_ready_prefixes.push_back(ready);
            if (single_upload && static_cast<std::size_t>(ready) != publication.table_rows) {
                if (deferred) {
                    // Deferred CPU work itself starts only after final gather.
                    require(publication.cpu_scatter_packed_rows == packed_before);
                    require(publication.cpu_scatter_pack_ready_publication_count == pack_ready_before);
                } else {
                    // The threaded packer has drained this partial prefix, but
                    // no upload is allowed until the final gather publication.
                    std::size_t covered = 0;
                    for (int r = 0; r < ready; r += publication.n_groups_per_m_cluster)
                        for (int source = 0; source < 3; ++source)
                            covered += table[r * publication.table_width + 3 + 2 * source] -
                                       table[r * publication.table_width + 2 + 2 * source];
                    require(covered > 0 && covered < routes);
                    require(publication.cpu_scatter_packed_rows == packed_before + covered);
                }
                require(publication.cpu_scatter_copy_count == copies_before);
                require(publication.cpu_scatter_copy_bytes == bytes_before);
                require(std::all_of(scatter.begin(), scatter.end(), [](int64_t x) { return x == -1; }));
                require(waiter.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout);
                return;
            }
            std::vector<int> seen(routes);
            std::size_t expected_packed = 0;
            for (int r = 0; r < ready; r += publication.n_groups_per_m_cluster) {
                auto* row = table + r * publication.table_width;
                std::size_t packed = 0;
                for (int s = 0; s < 3; ++s) packed += row[2+2*s];
                for (int s = 0; s < 3; ++s) {
                    for (int i = row[2+2*s]; i < row[3+2*s]; ++i, ++packed) {
                        require(packed < routes && !seen[packed]++);
                        ++expected_packed;
                        if (weight_mode) require(packed_weights[packed] == weights[s][i]);
                        const auto input = indices[s][i];
                        if (pooled) require(flat[packed] == s * 10 + input);
                        const auto begin = inverse.begin() + s*10;
                        const auto compact = std::find(begin, begin + counts[s], input) - begin;
                        require(compact < counts[s]);
                        require(scatter[packed*columns] == static_cast<int64_t>(
                            bases[s] + compact*(tile ? tile : 256)*2));
                        if (tile) require(scatter[packed*columns+1] == counts[s]*tile*2);
                    }
                }
            }
            // Repeated N groups share each M batch's packing; no routed row is
            // repacked either in legacy mode or during the final deferred walk.
            require(publication.cpu_scatter_packed_rows == packed_before + expected_packed);
            if (static_cast<std::size_t>(ready) == publication.table_rows)
                require(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }));
        };
        // Chunked arrivals cause several partial gather publications before the
        // final one. Reversing source priority changes packed row positions.
        // Unused tail rows and the empty source still notify after finalization.
        struct Arrival { int source; std::size_t begin; std::size_t count; };
        const auto arrivals = iteration == 0 ?
            std::vector<Arrival>{{0,0,6}, {1,0,2}, {1,2,6}, {0,6,4}, {1,8,2}, {2,0,10}} :
            std::vector<Arrival>{{1,0,8}, {0,0,3}, {0,3,3}, {0,6,4}, {1,8,2}, {2,0,10}};
        for (std::size_t arrival_index = 0; arrival_index < arrivals.size(); ++arrival_index) {
            const auto& arrival = arrivals[arrival_index];
            buffers.process_router_notification_completion(
                0, arrival.source, iteration, arrival.begin, arrival.count);
            if (threaded && arrival_index == 0) {
                packing_gate.wait_entered();
                const auto blocked_prefix = buffers.router_computation_scheduler_state().published_rows;
                require(blocked_prefix > 0 && blocked_prefix < publication.table_rows);
                partial_ready_prefixes.push_back(static_cast<int32_t>(blocked_prefix));
                require(publication.cpu_scatter_packed_rows == packed_before);
                require(publication.cpu_scatter_copy_count == copies_before);
                // Notification processing must progress while the CPU packer is
                // deliberately blocked. A shared scheduler lock would deadlock.
                do {
                    require(++arrival_index < arrivals.size());
                    const auto next_arrival = arrivals[arrival_index];
                    auto notification = std::async(std::launch::async, [&buffers, iteration, next_arrival] {
                        buffers.process_router_notification_completion(
                            0, next_arrival.source, iteration, next_arrival.begin, next_arrival.count);
                    });
                    const bool advanced = notification.wait_for(std::chrono::seconds(5)) ==
                                          std::future_status::ready;
                    if (!advanced) packing_gate.release();
                    require(advanced);
                    notification.get();
                } while (buffers.router_computation_scheduler_state().published_rows == blocked_prefix);
                require(buffers.router_computation_scheduler_state().published_rows < publication.table_rows);
                require(publication.cpu_scatter_packed_rows == packed_before);
                require(publication.cpu_scatter_copy_count == copies_before);
                packing_gate.release();
            }
            if (threaded)
                CpuScatterTestAccess::drain(buffers, buffers.router_computation_scheduler_state().published_rows);
            if (deferred && !final_publication_seen &&
                buffers.router_computation_scheduler_state().published_rows == publication.table_rows) {
                // Snapshot is taken at the start of CPU packing. Matching the
                // FINAL ready-publication counter proves packing starts after
                // that submission, not immediately before it in the same call.
                final_pack_ready_count = publication.ready_rows_publication_count;
                require(publication.cpu_scatter_pack_ready_publication_count == final_pack_ready_count);
                require(final_pack_ready_count > pack_ready_before);
                final_publication_seen = true;
            }
            if (single_upload && waiter.valid() &&
                buffers.router_computation_scheduler_state().published_rows == publication.table_rows) {
                // Join the event-dependent consumer BEFORE any publication-stream sync below.
                waiter.get();
                require(std::all_of(consumer_scatter.begin(), consumer_scatter.end(), [](int64_t x) { return x != -1; }));
            }
            check_published();
            if (final_publication_seen)
                require(publication.cpu_scatter_pack_ready_publication_count == final_pack_ready_count);
            if (buffers.router_computation_scheduler_state().published_rows < publication.table_rows) {
                const auto packed_before_flush = publication.cpu_scatter_packed_rows.load();
                buffers.flush_router_notification_publication();
                require(publication.cpu_scatter_packed_rows == packed_before_flush);
                check_published();
            }
        }
        require(buffers.router_computation_scheduler_state().published_rows == publication.table_rows);
        require(partial_ready_prefixes.size() >= 2);
        require(publication.cpu_scatter_packed_rows == packed_before + routes);
        if (single_upload) {
            if (deferred) require(final_publication_seen);
            require(publication.cpu_scatter_copy_count == copies_before + 1);
            require(publication.cpu_scatter_copy_bytes == bytes_before + slab_bytes);
            require(consumer_scatter == scatter);
            buffers.wait_cpu_scatter_ready(iteration, compute_stream_address, 1);
            buffers.flush_router_notification_publication();
            require(publication.cpu_scatter_copy_count == copies_before + 1);
            if (deferred)
                require(publication.cpu_scatter_pack_ready_publication_count == final_pack_ready_count);
        } else {
            require(publication.cpu_scatter_copy_count > copies_before + 1);
            require(publication.cpu_scatter_copy_bytes == bytes_before + scatter_bytes +
                (weight_mode ? routes * sizeof(float) : 0));
        }
        // Repeated final flushes must neither upload nor pack a second time.
        const auto final_copies = publication.cpu_scatter_copy_count.load();
        for (int repeat = 0; repeat < 2; ++repeat) {
            buffers.flush_router_notification_publication();
            require(publication.cpu_scatter_packed_rows == packed_before + routes);
            require(publication.cpu_scatter_copy_count == final_copies);
            if (deferred)
                require(publication.cpu_scatter_pack_ready_publication_count == final_pack_ready_count);
        }
        if (!iteration) first_iteration = scatter;
        else require(scatter != first_iteration);
    }
    if (single_upload) {
        buffers.begin_router_notification_iteration(2);
        auto changed_generation = std::async(std::launch::async, [&] {
            try { buffers.wait_cpu_scatter_ready(2, compute_stream_address, 5000); }
            catch (const std::runtime_error&) { return true; }
            return false;
        });
        require(changed_generation.wait_for(std::chrono::milliseconds(2)) == std::future_status::timeout);
        buffers.begin_router_notification_iteration(3);
        require(changed_generation.get());
    }
    if (threaded && lifecycle == 1) {
        require(!cuda);
        buffers.begin_router_notification_iteration(4);
        packing_gate.arm();
        buffers.process_router_notification_completion(0, 0, 4, 0, 6);
        packing_gate.wait_entered();
        const auto copies_before_reset = publication.cpu_scatter_copy_count.load();
        auto resetting = std::async(std::launch::async, [&] {
            buffers.begin_router_notification_iteration(5);
        });
        ReleasePackingGate release_before_reset_future_cleanup{packing_gate};
        // Reset must wait for the blocked worker to finish using its descriptor and slab.
        CpuScatterTestAccess::wait_cancelled(buffers);
        const bool reset_blocked = resetting.wait_for(std::chrono::milliseconds(0)) ==
                                   std::future_status::timeout;
        packing_gate.release();
        require(reset_blocked);
        require(resetting.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        resetting.get();
        CpuScatterTestAccess::require_empty_epoch(buffers, 5);
        require(publication.cpu_scatter_copy_count == copies_before_reset);
        rejects([&] { buffers.wait_cpu_scatter_ready(4, compute_stream_address, 1); });
        // A fresh epoch remains usable after cancelling an in-flight partial pack.
        for (int source : {1, 0, 2})
            buffers.process_router_notification_completion(0, source, 5, 0, 10);
        buffers.wait_cpu_scatter_ready(5, compute_stream_address, 5000);
        require(publication.cpu_scatter_copy_count == copies_before_reset + 1);

        buffers.begin_router_notification_iteration(6);
        packing_gate.arm(true);
        auto failed_waiter = std::async(std::launch::async, [&] {
            try { buffers.wait_cpu_scatter_ready(6, compute_stream_address, 5000); }
            catch (const std::runtime_error& error) {
                return std::string(error.what()).find("injected CPU scatter worker failure") != std::string::npos;
            }
            return false;
        });
        require(failed_waiter.wait_for(std::chrono::milliseconds(2)) == std::future_status::timeout);
        buffers.process_router_notification_completion(0, 0, 6, 0, 6);
        require(failed_waiter.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        require(failed_waiter.get());
        require(publication.cpu_scatter_copy_count == copies_before_reset + 1);
        // Failed worker teardown still joins and never publishes stale metadata.
        buffers.stop_cpu_scatter_worker();
    }
    if (threaded && lifecycle == 2) {
        require(!cuda);
        buffers.begin_router_notification_iteration(4);
        packing_gate.arm();
        buffers.process_router_notification_completion(0, 0, 4, 0, 6);
        packing_gate.wait_entered();
        std::promise<void> destructor_started;
        auto started = destructor_started.get_future();
        auto destroying = std::async(std::launch::async, [&] {
            destructor_started.set_value();
            buffers_owner.reset();
        });
        ReleasePackingGate release_before_destructor_future_cleanup{packing_gate};
        started.get();
        const bool destructor_blocked = destroying.wait_for(std::chrono::milliseconds(20)) ==
                                        std::future_status::timeout;
        packing_gate.release();
        require(destructor_blocked);
        require(destroying.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
        destroying.get();
        require(!buffers_owner);
        // The callback's captured gate is still alive here; successful destructor
        // completion establishes that no worker can use it after this scope.
    }
#if RDMA_PROXY_HAVE_CUDA
    if (cuda) {
        buffers.synchronize_router_notification_publication();
        buffers.stop_cpu_scatter_worker();
        require(cudaFree(device_scatter) == cudaSuccess);
    }
#endif
    destroy_cuda_stream(compute_stream, config.mock_mode);
}

static void test_threaded_dynamic_reuse(bool reserve_at_startup, bool cuda, bool async_lookup = false) {
    ProxyConfig c;
    c.mock_mode = !cuda;
    c.router_routing_enabled = c.router_dynamic_routing_enabled = true;
    c.router_weighted_contributions_enabled = c.nvlink_forwarding_enabled = true;
    c.nvlink_forward_completion_notifications_enabled = true;
    c.num_nodes = 1;
    c.num_gpus_per_node = c.router_num_experts = 2;
    c.router_top_k = 1;
    c.num_tokens = 4;
    c.token_dimension = c.expert_gemm_dimension = c.expert_gemm_n_tile = 1;
    const std::size_t tile = async_lookup ? 128 : 0;
    const std::size_t columns = tile ? 2 : 1;
    if (tile) c.token_dimension = 2 * tile;
    c.expert_gemm_m_tile = 4;  // The largest generation covers two M batches.
    c.dtype = DataType::kBF16;

    const auto destroy_stream = [&](void* stream) { destroy_cuda_stream(stream, c.mock_mode); };
    std::unique_ptr<void, decltype(destroy_stream)> consumer_stream(
        create_cuda_stream(c.cuda_device_id, true, c.mock_mode), destroy_stream);
    const auto stream_address = cuda ? reinterpret_cast<uintptr_t>(consumer_stream.get()) : uintptr_t{1};
    alignas(16) unsigned char mock_destination[256];
    auto* destination = mock_destination;
#if RDMA_PROXY_HAVE_CUDA
    std::unique_ptr<void, decltype(&cudaFree)> managed_destination(nullptr, cudaFree);
    if (cuda) {
        void* allocation = nullptr;
        require(cudaMallocManaged(&allocation, sizeof(mock_destination)) == cudaSuccess);
        managed_destination.reset(allocation);
        destination = static_cast<unsigned char*>(allocation);
    }
#endif
    // Join the publisher and finish all copies before releasing its destination.
    PackingGate packing_gate;
    PackingGate lookup_gate;
    int64_t counts[2]{}, inverse[8]{};
    std::atomic<unsigned> packing_threads{0};
    CudaBuffers buffers(c);
    ReleasePackingGate release_before_buffers_cleanup{packing_gate};
    ReleasePackingGate release_lookup{lookup_gate};
    CpuScatterTestAccess::set_lookup_hook(buffers, [&] { lookup_gate(); });
    CpuScatterTestAccess::set_hook(buffers, [&] {
        // A newly created std::thread can have the same OS ID as its predecessor.
        thread_local bool first_pack = true;
        if (first_pack) {
            ++packing_threads;
            first_pack = false;
        }
        packing_gate();
    });
    buffers.initialize();
    const std::size_t reserved_bytes = 8 * (columns * sizeof(int64_t) + sizeof(float));
    void* pinned = nullptr;
    if (reserve_at_startup) {
        buffers.reserve_threaded_cpu_scatter(columns);
        const auto& slab = CpuScatterTestAccess::slab(buffers);
        pinned = slab.ptr;
        require(pinned && slab.bytes == reserved_bytes);
    }
    const std::vector<std::vector<std::vector<int32_t>>> epochs{
        {{}, {}}, {{0}, {}}, {{0,1,2,3}, {0,1,2,3}}, {{}, {}}, {{1,2,3}, {}}, {{}, {1}}, {{}, {}},
    };
    // On the mock path also replace a route while the packer holds its old descriptors.
    const bool cancel_busy_pack = !cuda && !reserve_at_startup;
    constexpr std::size_t busy_epoch = 2;
    for (std::size_t epoch = 0; epoch < epochs.size(); ++epoch) {
        const bool per_batch = epoch == 4;  // Resume threaded publication afterwards.
        const auto& lists = epochs[epoch];
        std::vector<RouterExpertMetadata> metadata;
        std::vector<float> expected_weights;
        for (int source = 0; source < 2; ++source) {
            auto& m = metadata.emplace_back();
            m.destination_node_rank = m.destination_gpu_index = m.source_node_rank = 0;
            m.source_gpu_index = source;
            m.num_nodes = m.experts_per_gpu = m.top_k = 1;
            m.num_gpus_per_node = m.num_experts = 2;
            m.num_tokens = c.num_tokens;
            m.expert_token_indices = lists[source];
            m.expert_offsets = {0, static_cast<int32_t>(lists[source].size())};
            m.weighted_contributions = true;
            for (const auto row : lists[source]) {
                const float weight = (1 + epoch + source + row) * 0.125f;
                m.expert_route_weights.push_back(weight);
                expected_weights.push_back(weight);
            }
        }
        if (cancel_busy_pack && epoch == busy_epoch + 1) {
            auto installing = std::async(std::launch::async, [&] {
                buffers.install_iteration_expert_metadata(std::move(metadata), epoch);
            });
            ReleasePackingGate release_before_future_cleanup{packing_gate};
            CpuScatterTestAccess::wait_cancelled(buffers);
            require(installing.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout);
            packing_gate.release();
            installing.get();
            require(std::all_of(destination, destination + sizeof(mock_destination),
                                [](unsigned char value) { return value == 0xa5; }));
        } else {
            buffers.install_iteration_expert_metadata(std::move(metadata), epoch);
        }
        const auto& slab = CpuScatterTestAccess::slab(buffers);
        if (pinned) require(slab.ptr == pinned);  // Still owned between route bindings.
        auto& publication = buffers.router_notification_publication_buffers();
        const auto previous_bytes = publication.cpu_scatter_copy_bytes.load();
        const auto previous_copies = publication.cpu_scatter_copy_count.load();
        const auto routes = expected_weights.size();
        if (!routes) {
            buffers.begin_router_notification_iteration(epoch);
            buffers.flush_router_notification_publication();
            require(publication.cpu_scatter_copy_count == previous_copies);
            continue;
        }
        const auto scatter_bytes = routes * columns * sizeof(int64_t);
        const auto weight_offset = (scatter_bytes + 15) & ~std::size_t{15};
        const auto active_bytes = weight_offset + routes * sizeof(float);
        int32_t pairs[]{0, 0, 0, 1};
        uint64_t bases[]{0x1000, 0x2000};
        RdmaProxyCpuScatterConfig setup{sizeof(setup), 2, routes, c.num_tokens, c.token_dimension, tile, 2,
            reinterpret_cast<uintptr_t>(destination), pairs, bases, counts, inverse};
        const auto weights = reinterpret_cast<uintptr_t>(destination + weight_offset);
        const bool pause_lookup = async_lookup && epoch == busy_epoch;
        if (async_lookup && !per_batch) {
            if (epoch == 1) setup.output_to_input = nullptr;  // Counts suffice for unchecked returns.
            if (pause_lookup) lookup_gate.arm();
            rejects([&] { buffers.configure_cpu_scatter_async(setup, weights, active_bytes - 1); });
            buffers.configure_cpu_scatter_async(setup, weights, active_bytes);
            if (pause_lookup) {
                lookup_gate.wait_entered();
                // Pair IDs and numerical bases belong to the caller only through configure.
                std::fill_n(pairs, 4, -1);
                std::fill_n(bases, 2, 0);
            }
        } else {
            buffers.configure_cpu_scatter(setup);
            buffers.configure_cpu_route_weights(weights, routes);
            if (!per_batch) buffers.configure_cpu_scatter_threaded(active_bytes);
        }
        CpuScatterTestAccess::require_no_batch_staging(buffers);
        if (!per_batch && !pinned) pinned = slab.ptr;
        require(pinned && slab.ptr == pinned && slab.bytes == reserved_bytes);

        // Leave old packed bytes in the pinned slab, and detect oversized uploads
        // into the device destination even when the new route is smaller.
#if RDMA_PROXY_HAVE_CUDA
        if (cuda) require(cudaDeviceSynchronize() == cudaSuccess);
#endif
        std::memset(destination, 0xa5, sizeof(mock_destination));
        auto publish = [&] {
            buffers.begin_router_notification_iteration(epoch);
            buffers.synchronize_router_notification_publication(); // Pre-up ready=0 fence.
            if (cancel_busy_pack && epoch == busy_epoch) packing_gate.arm();
            for (int source = 0; source < 2; ++source)
                buffers.process_router_notification_completion(0, source, epoch, 0, c.num_tokens);
        };
        if (pause_lookup) {
            // A blocked lookup must permit pre-up reset and ALL gather commits.
            auto publisher = std::async(std::launch::async, publish);
            ReleasePackingGate release_before_future{lookup_gate};
            require(publisher.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
            publisher.get();
            require(publication.cpu_scatter_copy_count == previous_copies);
            rejects([&] { buffers.wait_cpu_scatter_ready(epoch, stream_address, 1); });
            lookup_gate.release();
        } else {
            publish();
        }
        if (cancel_busy_pack && epoch == busy_epoch) {
            packing_gate.wait_entered();
            continue;
        }
        if (per_batch) {
            // A reservation alone must not select the one-copy publication path.
            rejects([&] { buffers.wait_cpu_scatter_ready(epoch, stream_address, 1); });
            buffers.synchronize_router_notification_publication();
            require(publication.cpu_scatter_copy_count == previous_copies + 2);
            require(publication.cpu_scatter_copy_bytes == previous_bytes + scatter_bytes + routes * sizeof(float));
        } else {
            buffers.wait_cpu_scatter_ready(epoch, stream_address, 5000);
            synchronize_cuda_stream(consumer_stream.get(), c.mock_mode);
            require(publication.cpu_scatter_copy_count == previous_copies + 1);
            require(publication.cpu_scatter_copy_bytes == previous_bytes + active_bytes);
        }
        std::size_t packed = 0;
        for (int source = 0; source < 2; ++source) {
            require(counts[source] == static_cast<int64_t>(lists[source].size()));
            if (setup.output_to_input) {
                auto begin = inverse + source*c.num_tokens;
                require(std::equal(lists[source].begin(), lists[source].end(), begin));
                require(std::all_of(begin + lists[source].size(), begin + c.num_tokens,
                                    [](int64_t i) { return i == -1; }));
            }
            for (std::size_t row = 0; row < lists[source].size(); ++row, ++packed) {
                const auto row_bytes = (tile ? tile : c.token_dimension) * sizeof(uint16_t);
                const uint64_t address = (source + 1) * 0x1000 + row * row_bytes;
                const auto* entry = destination + packed * columns * sizeof(address);
                require(std::memcmp(entry, &address, sizeof(address)) == 0);
                if (tile) {
                    const uint64_t stride = lists[source].size() * row_bytes;
                    require(std::memcmp(entry + sizeof(address), &stride, sizeof(stride)) == 0);
                }
            }
        }
        require(std::memcmp(destination + weight_offset, expected_weights.data(), routes * sizeof(float)) == 0);
        const unsigned char expected_padding = per_batch ? 0xa5 : 0;
        require(std::all_of(destination + scatter_bytes, destination + weight_offset,
                            [&](unsigned char value) { return value == expected_padding; }));
        require(std::all_of(destination + active_bytes, destination + sizeof(mock_destination),
                            [](unsigned char value) { return value == 0xa5; }));
    }
    if (packing_threads != 1)
        throw std::runtime_error("expected one scatter worker to pack all dynamic generations");
}

static void test_async_scatter_lookup_retirement(bool fail) {
    ProxyConfig c;
    c.mock_mode = c.router_routing_enabled = c.router_dynamic_routing_enabled = true;
    c.nvlink_forwarding_enabled = c.nvlink_forward_completion_notifications_enabled = true;
    c.num_nodes = c.router_top_k = 1;
    c.num_gpus_per_node = c.router_num_experts = 2;
    c.num_tokens = 4;
    c.expert_gemm_m_tile = 1;
    c.token_dimension = c.expert_gemm_n_tile = c.expert_gemm_dimension = 1;
    c.dtype = DataType::kBF16;
    int32_t offsets[]{0, 1}, indices[]{3}, pairs[]{0, 0, 0, 1};
    uint64_t bases[]{0x1000, 0x2000};
    alignas(16) int64_t destination[]{-1, -1};
    int64_t counts[]{-1, -1};
    PackingGate gate;
    CudaBuffers buffers(c);
    ReleasePackingGate release_before_cleanup{gate};
    buffers.initialize();
    std::vector<RouterExpertMetadata> metadata(2);
    for (int source = 0; source < 2; ++source) {
        auto& m = metadata[source];
        m.destination_node_rank = m.destination_gpu_index = m.source_node_rank = 0;
        m.source_gpu_index = source;
        m.num_nodes = m.experts_per_gpu = m.top_k = 1;
        m.num_gpus_per_node = m.num_experts = 2;
        m.num_tokens = c.num_tokens;
        m.borrow_expert_lists(offsets, indices, 1);
    }
    buffers.install_iteration_expert_metadata(std::move(metadata), 0);
    gate.arm();
    CpuScatterTestAccess::set_lookup_hook(buffers, [&] {
        gate();
        if (fail) throw std::runtime_error("injected asynchronous lookup failure");
    });
    RdmaProxyCpuScatterConfig setup{sizeof(setup), 2, 2, c.num_tokens, 1, 0, 2,
        reinterpret_cast<uintptr_t>(destination), pairs, bases, counts, nullptr};
    buffers.configure_cpu_scatter_async(setup, 0, sizeof(destination));
    gate.wait_entered();
    if (fail) {
        buffers.begin_router_notification_iteration(0);
        buffers.synchronize_router_notification_publication();
        gate.release();
        try {
            buffers.wait_cpu_scatter_ready(0, 1, 5000);
            require(false);
        } catch (const std::exception& e) {
            require(std::string(e.what()) == "injected asynchronous lookup failure");
        }
        // The lookup failed with up already launched and no gather rows yet.
        // A scatter error must leave both partial and final up publishes live.
        auto& publication = buffers.router_notification_publication_buffers();
        auto* ready = static_cast<int32_t*>(publication.device_ready_rows.ptr);
        require(*ready == 0 && publication.table_rows == 2);
        // Preparation joins the ready=0 GPU write even if the CPU lookup failed
        // in the meantime. The transport must still submit its forward traffic.
        buffers.synchronize_router_notification_stream();
        buffers.process_router_notification_completion(0, 0, 0, 0, c.num_tokens);
        require(*ready == 1);
        buffers.process_router_notification_completion(0, 1, 0, 0, c.num_tokens);
        require(*ready == 2 && buffers.router_computation_scheduler_state().final_ready_submitted);
        rejects([&] { buffers.check_down_metadata_error(); });
        require(publication.cpu_scatter_copy_count == 0);
        buffers.stop_cpu_scatter_worker();
        buffers.retire_borrowed_expert_metadata();
    } else {
        // Abort before iteration preparation. The exchange cannot recycle those
        // borrowed source indices or the host result arrays until the lookup exits.
        auto retire = std::async(std::launch::async, [&] { buffers.retire_borrowed_expert_metadata(); });
        ReleasePackingGate release_before_future{gate};
        require(retire.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
        gate.release();
        retire.get();
        require(counts[0] == 1 && counts[1] == 1);
    }
    require(destination[0] == -1 && destination[1] == -1);
}

static void test_down_wave_publication(bool cuda, bool inject_failure = false) {
    ProxyConfig c;
    c.mock_mode = !cuda;
    c.router_routing_enabled = c.router_dynamic_routing_enabled = c.nvlink_forwarding_enabled = true;
    c.nvlink_forward_completion_notifications_enabled = true;
    c.num_nodes = 1;
    c.num_gpus_per_node = c.router_num_experts = 2;
    c.router_top_k = 1;
    c.num_tokens = c.expert_gemm_m_tile = 4;
    c.token_dimension = c.expert_gemm_dimension = c.expert_gemm_n_tile = 1;
    c.dtype = DataType::kBF16;
    const auto destroy = [&](void* stream) { destroy_cuda_stream(stream, c.mock_mode); };
    std::unique_ptr<void, decltype(destroy)> consumer(create_cuda_stream(0, true, c.mock_mode), destroy);
    const auto stream = cuda ? reinterpret_cast<uintptr_t>(consumer.get()) : uintptr_t{1};
    constexpr std::size_t words = 64;
    int32_t mock[words];
    auto* destination = mock;
#if RDMA_PROXY_HAVE_CUDA
    std::unique_ptr<void, decltype(&cudaFree)> managed(nullptr, cudaFree);
    if (cuda) {
        void* allocation = nullptr;
        require(cudaMallocManaged(&allocation, sizeof(mock)) == cudaSuccess);
        managed.reset(allocation);
        destination = static_cast<int32_t*>(allocation);
    }
#endif
    const DownWaveGeometry geometry{2, 2, 1, 3};
    CudaBuffers buffers(c);
    buffers.initialize();
    const auto install = [&](uint64_t epoch, std::size_t per_source) {
        std::vector<RouterExpertMetadata> metadata;
        for (int source = 0; source < 2; ++source) {
            auto& m = metadata.emplace_back();
            m.destination_node_rank = m.destination_gpu_index = m.source_node_rank = 0;
            m.source_gpu_index = source;
            m.num_nodes = m.experts_per_gpu = m.top_k = 1;
            m.num_gpus_per_node = m.num_experts = 2;
            m.num_tokens = c.num_tokens;
            for (std::size_t i = 0; i < per_source; ++i) m.expert_token_indices.push_back(i);
            m.expert_offsets = {0, static_cast<int32_t>(per_source)};
        }
        buffers.install_iteration_expert_metadata(std::move(metadata), epoch);
    };
    const auto verify = [&](std::size_t total) {
        std::vector<int32_t> expected(words, -1);
        build_down_wave_schedule(geometry, {total}, expected.data(), sizeof(mock));
        synchronize_cuda_stream(consumer.get(), c.mock_mode);
        require(std::equal(expected.begin(), expected.end(), destination));
    };
    auto& stats = buffers.router_notification_publication_buffers();
    install(0, 4);
    const auto reserve = [&] {
        buffers.configure_down_wave_schedule(reinterpret_cast<uintptr_t>(destination), words, geometry);
    };
    reserve();
    rejects(reserve);
    std::fill_n(destination, words, -1);
    rejects([&] { buffers.start_down_wave_schedule(0, {7}); });
    const auto first = buffers.start_down_wave_schedule(0, {8});
    rejects([&] { buffers.start_down_wave_schedule(0, {8}); });
    CpuScatterTestAccess::wait_wave_packed(buffers);
    buffers.begin_router_notification_iteration(0);
    buffers.process_router_notification_completion(0, 0, 0, 0, 4);
    require(buffers.router_computation_scheduler_state().published_rows == 1);
    rejects([&] { buffers.wait_down_wave_schedule(0, stream, 1); });
    require(stats.down_wave_copy_count == 0);
    buffers.process_router_notification_completion(0, 1, 0, 0, 4);
    buffers.wait_down_wave_schedule(0, stream, 5000);
    verify(8);
    buffers.flush_router_notification_publication();  // Repeating the final prefix cannot rearm DMA.
    require(stats.down_wave_copy_count == 1 && stats.down_wave_copy_bytes == first.bytes);

    install(1, 0);
    rejects([&] { buffers.wait_down_wave_schedule(0, stream, 10); });
    require(buffers.start_down_wave_schedule(1, {0}).bytes == 0);
    buffers.begin_router_notification_iteration(1);
    buffers.wait_down_wave_schedule(1, stream, 5000);
    require(stats.down_wave_copy_count == 1);

    install(2, 1);
    buffers.start_down_wave_schedule(2, {2});
    CpuScatterTestAccess::wait_wave_packed(buffers);
    auto waiter = std::async(std::launch::async, [&] {
        rejects([&] { buffers.wait_down_wave_schedule(2, stream, 5000); });
    });
    install(3, 2);  // Cancel a packed schedule before final gather submission.
    waiter.get();
    std::fill_n(destination, words, -1);
    buffers.begin_router_notification_iteration(3);
    for (int source = 0; source < 2; ++source)
        buffers.process_router_notification_completion(0, source, 3, 0, 4);
    const auto late = buffers.start_down_wave_schedule(3, {4});
    buffers.wait_down_wave_schedule(3, stream, 5000);
    verify(4);
    require(stats.down_wave_copy_count == 2 && stats.down_wave_copy_bytes == first.bytes + late.bytes);

    install(4, 1);
    buffers.start_down_wave_schedule(4, {2});
    if (inject_failure) {
        CpuScatterTestAccess::fail_parked_wave_build(buffers);
        rejects([&] { buffers.wait_down_wave_schedule(4, stream, 10); });
        rejects([&] { buffers.start_down_wave_schedule(4, {2}); });
        rejects([&] { buffers.retire_borrowed_expert_metadata(); });
    }
    buffers.stop_down_wave_worker();
    buffers.retire_borrowed_expert_metadata();  // Shutdown must retire borrowed metadata after a failure.
    rejects([&] { buffers.wait_down_wave_schedule(4, stream, 10); });
    require(stats.down_wave_copy_count == 2);
}

int main(int argc, char** argv) {
    const bool cuda = argc == 2 && std::string(argv[1]) == "--cuda";
    if (cuda) {
#if RDMA_PROXY_HAVE_CUDA
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
#else
        return 77;
#endif
    }
    for (bool per_entry : {false, true}) for (uint64_t tile : {0, 128})
        for (int weights : {0, 1, 2}) for (int publication_mode : {0, 1, 2})
            for (bool memcpy_flag : {false, true})
                run(per_entry, tile, cuda, weights, publication_mode, memcpy_flag);
    // Check single-buffer gathers against the compact down addresses and
    // position-specific weights in the SAME packed slots in both iterations.
    run(true, 128, cuda, 1, 0, true, 0, true);
    run(false, 128, cuda, 1, 2, false, 0, true);
    test_threaded_dynamic_reuse(true, cuda);
    test_threaded_dynamic_reuse(false, cuda);
    test_threaded_dynamic_reuse(true, cuda, true);
    test_down_wave_publication(cuda);
    if (!cuda) {
        test_async_scatter_lookup_retirement(false);
        test_async_scatter_lookup_retirement(true);
        test_down_wave_publication(false, true);
        run(false, 128, false, 1, 2, false, 1); // busy reset and worker failure
        run(false, 128, false, 1, 2, false, 2); // destructor joins a busy worker
    }
    std::cout << "test_cpu_scatter passed\n";
}
