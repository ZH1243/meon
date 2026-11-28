#include "cuda_buffers.hpp"
#include "benchmark_index_padding.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace rdma_proxy;

static RouterExpertMetadata table(int source, const std::vector<int32_t>& rows,
                                 int nodes = 1, int gpus_per_node = 2) {
    RouterExpertMetadata m;
    m.destination_node_rank = m.destination_gpu_index = 0;
    m.source_node_rank = source / gpus_per_node;
    m.source_gpu_index = source % gpus_per_node;
    m.num_nodes = nodes;
    m.num_gpus_per_node = gpus_per_node;
    m.num_experts = nodes * gpus_per_node;
    m.experts_per_gpu = 1;
    m.first_global_expert = 0;
    m.num_tokens = 7;
    m.top_k = 1;
    m.expert_offsets = {0, static_cast<int32_t>(rows.size())};
    m.expert_token_indices = rows;
    m.weighted_contributions = true;
    for (const auto row : rows) m.expert_route_weights.push_back((row + 1) * 0.125f);
    return m;
}

static ProxyConfig gather_config() {
    ProxyConfig c;
    c.mock_mode = c.router_routing_enabled = c.router_dynamic_routing_enabled = true;
    c.router_local_input_staging_enabled = c.nvlink_forwarding_enabled = true;
    c.nvlink_forward_completion_notifications_enabled = c.router_pooled_up_gemm_enabled = true;
    c.router_weighted_contributions_enabled = true;
    c.nvlink_forward_notification_flag_update_mode = NvlinkForwardNotificationFlagUpdateMode::kMemcpy;
    c.num_nodes = 1;
    c.num_gpus_per_node = c.router_num_experts = 2;
    c.router_top_k = 1;
    c.num_tokens = 7;
    c.token_dimension = c.expert_gemm_dimension = c.expert_gemm_n_tile = 1;
    c.expert_gemm_m_tile = 2;
    return c;
}

static void check_mock_gather(ProxyConfig c) {
    // The benchmark opt-out must not relax ordinary dynamic routing.
    c.router_benchmark_metadata_validation_enabled = false;
    CudaBuffers buffers(c);
    buffers.initialize();
    bool rejected = false;
    try { buffers.install_iteration_expert_metadata({table(0, {2, 1}), table(1, {})}, 0); }
    catch (const std::runtime_error& e) {
        rejected = std::string(e.what()).find("not monotonic") != std::string::npos;
    }
    assert(rejected);
    // Include an initially empty GPU, growth, same-size rerouting, then shrink.
    const std::vector<std::vector<int32_t>> lists{{}, {0,2,4,6}, {1,2,3,5}, {4}};
    void* indices = nullptr;
    void* descriptors = nullptr;
    void* pooled_indices = nullptr;
    for (std::size_t i = 0; i < lists.size(); ++i) {
        const auto& rows = lists[i];
        if (i == 2) {
            bool rejected = false;
            try { buffers.install_iteration_expert_metadata({table(0, rows), table(0, {})}, i); }
            catch (const std::runtime_error&) { rejected = true; }
            assert(rejected && buffers.router_computation_num_tokens() == lists[i-1].size());
        }
        buffers.install_iteration_expert_metadata({table(1, {}), table(0, rows)}, i);
        const auto& pub = buffers.router_notification_publication_buffers();
        const auto& source = buffers.nvlink_receive_buffer_for_source(0, 0);
        if (i == 0) {
            indices = source.expert_token_indices_device.ptr;
            descriptors = pub.device_table.ptr;
            pooled_indices = pub.device_pooled_a_idx.ptr;
        }
        assert(indices && descriptors && pooled_indices);
        assert(source.expert_token_indices_device.ptr == indices && pub.device_table.ptr == descriptors);
        assert(pub.device_pooled_a_idx.ptr == pooled_indices);
        assert(pub.a_idx_capacity == std::max<std::size_t>(1, rows.size()));
        assert(pub.table_rows == (rows.size() + 1) / 2);
        assert(buffers.router_computation_num_tokens() == rows.size());
        assert(std::equal(rows.begin(), rows.end(), static_cast<const int32_t*>(indices)));
        const auto* weights = static_cast<const float*>(source.expert_route_weights_device.ptr);
        for (std::size_t j = 0; j < rows.size(); ++j) assert(weights[j] == (rows[j] + 1) * 0.125f);
        buffers.begin_router_notification_iteration(i);
        const auto* ready = static_cast<const int32_t*>(pub.device_ready_rows.ptr);
        assert(*ready == 0);
        buffers.process_router_notification_completion(0, 0, i, 0, c.num_tokens);
        assert(*ready == static_cast<int32_t>(pub.table_rows));
        assert(std::equal(rows.begin(), rows.end(), static_cast<const int32_t*>(pooled_indices)));
    }
}

static void check_gather_table_reuse(ProxyConfig config) {
    constexpr int sources = 16;
    constexpr int width = 2 + 2 * sources;
    constexpr std::size_t n_groups = 2;
    config.num_nodes = 2;
    config.num_gpus_per_node = 8;
    config.router_num_experts = sources;
    config.peers = {{1, "peer-1", 1}};
    config.router_pooled_up_gemm_enabled = false;
    config.nvlink_forward_notification_scheduling_mode = "complete_round";
    config.expert_gemm_dimension = n_groups;
    config.expert_gemm_max_swizzle_size = 1;
    CudaBuffers buffers(config);
    buffers.initialize();

    // Each arriving source supplies one two-token M batch. Shrink the route,
    // run an empty iteration, then grow it with a different arrival order.
    const std::vector<std::vector<int>> arrivals{{0, 15}, {9}, {}, {15, 0, 8}};
    const std::vector<int32_t> batch{0, 1}, empty;
    std::vector<int32_t> expected_host, expected_device;
    for (std::size_t epoch = 0; epoch < arrivals.size(); ++epoch) {
        const auto& order = arrivals[epoch];
        std::vector<RouterExpertMetadata> metadata;
        for (int source = 0; source < sources; ++source) {
            const bool active = std::find(order.begin(), order.end(), source) != order.end();
            metadata.push_back(table(source, active ? batch : empty, config.num_nodes, config.num_gpus_per_node));
        }
        buffers.install_iteration_expert_metadata(std::move(metadata), epoch);
        const auto& pub = buffers.router_notification_publication_buffers();
        assert(pub.table_rows == order.size() * n_groups && pub.table_width == width);
        auto* host = static_cast<int32_t*>(pub.host_table.ptr);
        auto* device = static_cast<int32_t*>(pub.device_table.ptr);
        const auto words = pub.host_table.bytes / sizeof(*host);
        if (epoch == 0) {
            expected_host.assign(words, -17);
            expected_device.assign(words, -31);
            std::copy(expected_host.begin(), expected_host.end(), host);
            std::copy(expected_device.begin(), expected_device.end(), device);
        }
        assert(words == expected_host.size());
        const auto check_prefix = [&](std::size_t ready) {
            assert(*static_cast<const int32_t*>(pub.device_ready_rows.ptr) == static_cast<int32_t>(ready));
            assert(std::equal(expected_host.begin(), expected_host.end(), host));
            assert(std::equal(expected_device.begin(), expected_device.end(), device));
        };
        const bool armed = buffers.prepare_early_up(epoch, 1);
        assert(armed == !order.empty());
        buffers.begin_router_notification_iteration(epoch, armed);
        check_prefix(0);

        std::array<int32_t, width> row{};
        for (std::size_t i = 0; i < order.size(); ++i) {
            const int source = order[i];
            buffers.ingest_router_notification_completion(source / config.num_gpus_per_node,
                                                          source % config.num_gpus_per_node, epoch, 0, 2);
            check_prefix(i * n_groups);
            buffers.schedule_router_notification_completions(epoch);
            row[3 + 2 * source] = 2;
            for (std::size_t group = 0; group < n_groups; ++group) {
                row[1] = static_cast<int32_t>(group);
                const auto offset = (i * n_groups + group) * width;
                std::copy(row.begin(), row.end(), expected_host.begin() + offset);
                std::copy(row.begin(), row.end(), expected_device.begin() + offset);
            }
            check_prefix((i + 1) * n_groups);
            row[2 + 2 * source] = 2;  // This source's range is [2, 2) in subsequent rows.
        }
    }
}

static void check_cancelled_staged_completion(ProxyConfig config, bool per_entry) {
    config.router_pooled_up_gemm_enabled = false;
    config.nvlink_forward_notification_flush_per_entry_enabled = per_entry;
    CudaBuffers buffers(config);
    buffers.initialize();
    buffers.install_iteration_expert_metadata({table(0, {0, 1}), table(1, {0, 1, 2, 3})}, 0);
    const auto& pub = buffers.router_notification_publication_buffers();
    const auto words = pub.table_rows * pub.table_width;
    std::fill_n(static_cast<int32_t*>(pub.host_table.ptr), words, -17);
    std::fill_n(static_cast<int32_t*>(pub.device_table.ptr), words, -31);
    if (!buffers.prepare_early_up(0, 1)) throw std::runtime_error("expected an armed gather");
    buffers.begin_router_notification_iteration(0, true);
    buffers.process_router_notification_completion(0, 0, 0, 0, 2);
    const auto* table_words = static_cast<const int32_t*>(pub.device_table.ptr);
    const auto* ready = static_cast<const int32_t*>(pub.device_ready_rows.ptr);
    assert(*ready == 1 && pub.table_rows == 3);
    // A complete round can be between ingest and schedule when the launch fails.
    buffers.ingest_router_notification_completion(0, 1, 0, 0, 2);
    buffers.cancel_early_up(0);
    std::vector<int32_t> cancelled(table_words, table_words + words);
    const std::array<int32_t, 6> local_row{0, 0, 0, 2, 0, 0};
    assert(std::equal(local_row.begin(), local_row.end(), cancelled.begin()));
    assert(std::all_of(cancelled.begin() + pub.table_width, cancelled.end(), [](int32_t v) { return v == 0; }));
    buffers.schedule_router_notification_completions(0);
    buffers.process_router_notification_completion(0, 1, 0, 2, 2);
    assert(*ready == 3 && buffers.router_computation_scheduler_state().published_rows == 1);
    assert(std::equal(cancelled.begin(), cancelled.end(), table_words));
    assert(std::equal(cancelled.begin(), cancelled.end(), static_cast<const int32_t*>(pub.host_table.ptr)));
}

struct TestRecords {
    std::array<int32_t, 4> rows{1, 2, 4, 6};
    std::array<int32_t, 2> full_offsets{0, 4}, shrinking_offsets{0, 0};
    std::array<int32_t, 4> readback{};
};

#if RDMA_PROXY_HAVE_CUDA
static void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

static void check_early_reset_cuda(ProxyConfig config, TestRecords& pinned) {
    config.router_pooled_up_gemm_enabled = false;
    CudaBuffers buffers(config);
    buffers.initialize();
    const auto release = [](void* stream) { destroy_cuda_stream(stream, false); };
    std::unique_ptr<void, decltype(release)> consumer(create_cuda_stream(config.cuda_device_id, true, false), release);
    const auto stream = static_cast<cudaStream_t>(consumer.get());
    for (unsigned epoch = 0; epoch < 3; ++epoch) {
        const int source = epoch % 2;
        buffers.install_iteration_expert_metadata({table(source, {1, 2}), table(1 - source, {})}, epoch);
        const auto& pub = buffers.router_notification_publication_buffers();
        if (epoch) {
            if (!buffers.prepare_early_up(epoch, reinterpret_cast<uintptr_t>(stream)))
                throw std::runtime_error("expected early up reset");
            // The nonblocking consumer's first read is ordered only by the
            // current reset event. The preceding generation published a positive row.
            check_cuda(cudaMemcpyAsync(pinned.readback.data(), pub.device_ready_rows.ptr, sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, stream));
            check_cuda(cudaStreamSynchronize(stream));
            if (pinned.readback[0] != 0) throw std::runtime_error("consumer read a stale ready prefix");
        }
        buffers.begin_router_notification_iteration(epoch);
        buffers.process_router_notification_completion(0, source, epoch, 0, config.num_tokens);
        buffers.synchronize_router_notification_stream();
        check_cuda(cudaMemcpy(pinned.readback.data(), pub.device_ready_rows.ptr, sizeof(int32_t), cudaMemcpyDeviceToHost));
        if (pinned.readback[0] != 1) throw std::runtime_error("missing current gather publication");
        std::array<int32_t, 6> descriptor{}, expected{};
        expected[3 + 2 * source] = 2;
        check_cuda(cudaMemcpy(descriptor.data(), pub.device_table.ptr, sizeof(descriptor), cudaMemcpyDeviceToHost));
        if (descriptor != expected) throw std::runtime_error("published row contains stale gather fields");
    }
}

static void check_direct_index_padding_cuda() {
    // Include a long tail (multiple grid strides), an empty list, a full list,
    // and one guard word on either side of every view, as in the IPC arena.
    constexpr uint32_t extent = 18'003;
    constexpr uint32_t stride = extent + 2;
    BenchmarkIndexPadding padding;
    padding.common_count = extent;
    std::vector<int32_t> host(padding.sources * stride, 17);
    int32_t* device = nullptr;
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device), host.size() * sizeof(int32_t)));
    std::unique_ptr<void, decltype(&cudaFree)> owner(device, cudaFree);
    check_cuda(cudaMemcpy(device, host.data(), host.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    for (uint32_t s = 0; s < padding.sources; ++s) {
        padding.indices[s] = device + s * stride + 1;
        padding.counts[s] = s == 0 ? 0 : extent - s + 1;
    }
    clear_benchmark_index_padding(padding, extent);
    check_cuda(cudaStreamSynchronize(cudaStreamLegacy));
    check_cuda(cudaMemcpy(host.data(), device, host.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    for (uint32_t s = 0; s < padding.sources; ++s) {
        for (uint32_t i = 0; i < stride; ++i) {
            const auto expected = (i > padding.counts[s] && i <= extent) ? 0 : 17;
            if (host[s * stride + i] != expected)
                throw std::runtime_error("incorrect direct benchmark index padding or clobbered guard");
        }
    }
}
#endif

// The source slot contains old rows beyond its new count. A second logical
// source keeps the padded device extent at four throughout the test.
static void check_borrowed_upload(ProxyConfig c, TestRecords& records) {
    c.router_weighted_contributions_enabled = false;
    CudaBuffers buffers(c);
    buffers.initialize();
    const auto destroy_stream = [&](void* stream) { destroy_cuda_stream(stream, c.mock_mode); };
    std::unique_ptr<void, decltype(destroy_stream)> consumer(
        create_cuda_stream(c.cuda_device_id, true, c.mock_mode), destroy_stream);
    const int32_t shrinking_counts[] = {4, 2, 0};
    for (std::size_t iteration = 0; iteration < 3; ++iteration) {
        const auto count = static_cast<std::size_t>(shrinking_counts[iteration]);
        records.shrinking_offsets[1] = shrinking_counts[iteration];
        auto full = table(0, {});
        full.weighted_contributions = false;
        auto shrinking = full;
        shrinking.source_gpu_index = 1;
        full.borrow_expert_lists(records.full_offsets.data(), records.rows.data(), records.rows.size());
        shrinking.borrow_expert_lists(records.shrinking_offsets.data(), count ? records.rows.data() : nullptr, count);
        buffers.install_iteration_expert_metadata({full, shrinking}, iteration);

        const auto& source = buffers.nvlink_receive_buffer_for_source(0, 1);
        const auto* device = static_cast<const int32_t*>(source.expert_token_indices_device.ptr);
        if (c.mock_mode) {
            std::copy_n(device, records.readback.size(), records.readback.begin());
        } else {
#if RDMA_PROXY_HAVE_CUDA
            // The real consumer can use the indices immediately after install.
            // A nonblocking stream has no implicit dependency on the uploader.
            check_cuda(cudaMemcpyAsync(records.readback.data(), device, sizeof(records.readback),
                                       cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(consumer.get())));
            synchronize_cuda_stream(consumer.get(), false);
#endif
        }
        buffers.retire_borrowed_expert_metadata();
        if (!std::equal(records.rows.begin(), records.rows.begin() + count, records.readback.begin()) ||
            !std::all_of(records.readback.begin() + count, records.readback.end(),
                         [](int32_t row) { return row == 0; }))
            throw std::runtime_error("borrowed A_idx has an incorrect prefix or stale padding");
    }
}

int main(int argc, char** argv) try {
    auto config = gather_config();
    if (argc == 2 && std::string(argv[1]) == "--cuda") {
#if RDMA_PROXY_HAVE_CUDA
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
        config.mock_mode = false;
        check_cuda(cudaSetDevice(config.cuda_device_id));
        check_direct_index_padding_cuda();
        void* memory = nullptr;
        check_cuda(cudaMallocHost(&memory, sizeof(TestRecords)));
        std::unique_ptr<void, decltype(&cudaFreeHost)> owner(memory, cudaFreeHost);
        // TestRecords is trivially destructible. Its pinned backing outlives
        // CudaBuffers, including cleanup after a partially completed upload.
        auto* records = new (memory) TestRecords{};
        check_borrowed_upload(config, *records);
        check_early_reset_cuda(config, *records);
#else
        return 77;
#endif
    } else {
        check_mock_gather(config);
        check_gather_table_reuse(config);
        for (bool per_entry : {false, true}) check_cancelled_staged_completion(config, per_entry);
        TestRecords records;
        check_borrowed_upload(config, records);
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
