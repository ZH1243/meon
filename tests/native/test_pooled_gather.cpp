#include "cuda_buffers.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

using namespace rdma_proxy;

// Simulate the four-column kernel at each visible prefix. Expected routes
// below are computed from source arrivals, independently of either work table.
static void check_stream(int nodes, bool per_entry) {
    ProxyConfig config;
    config.mock_mode = true;
    config.num_nodes = nodes;
    config.num_gpus_per_node = 8;
    const int sources = 8 * nodes;
    if (nodes > 1) config.peers.push_back({1, "localhost", 0});
    config.router_routing_enabled = true;
    config.nvlink_forwarding_enabled = true;
    config.nvlink_forward_completion_notifications_enabled = true;
    config.nvlink_forward_notification_flush_per_entry_enabled = per_entry;
    config.router_pooled_up_gemm_enabled = true;
    config.router_num_experts = 2 * sources;
    config.router_top_k = 2;
    config.num_tokens = 11;
    config.token_dimension = 1;
    config.dtype = DataType::kFP32;
    config.expert_gemm_m_tile = 3;
    config.expert_gemm_n_tile = 128;
    config.expert_gemm_dimension = 512;
    config.expert_gemm_max_swizzle_size = 2;
    CudaBuffers buffers(config);
    buffers.initialize();
    const int a = 1, b = sources / 2, c = sources - 1;
    for (int source = 0; source < sources; ++source) {
        RouterExpertMetadata m;
        m.destination_node_rank = m.destination_gpu_index = 0;
        m.source_node_rank = source / 8;
        m.source_gpu_index = source % 8;
        m.num_nodes = nodes;
        m.num_gpus_per_node = 8;
        m.num_experts = config.router_num_experts;
        m.experts_per_gpu = 2;
        m.first_global_expert = 0;
        m.num_tokens = config.num_tokens;
        m.expert_offsets = {0, 0, 0};
        if (source == a || source == b) {
            m.expert_offsets = {0, 2, 3};
            m.expert_token_indices = source == a ? std::vector<int32_t>{1, 4, 1}
                                                 : std::vector<int32_t>{0, 6, 6};
        } else if (source == c) {
            m.expert_offsets = {0, 1, 2};
            m.expert_token_indices = {3, 3};
        }
        buffers.install_expert_metadata(std::move(m));
    }
    const auto& pub = buffers.router_notification_publication_buffers();
    assert(pub.device_pooled_table.bytes == 6 * 4 * sizeof(int32_t));
    assert(pub.device_pooled_a_idx.bytes == sources * 3 * sizeof(int32_t));
    const auto* table = static_cast<const int32_t*>(pub.device_pooled_table.ptr);
    const auto* flat = static_cast<const int32_t*>(pub.device_pooled_a_idx.ptr);
    auto* pool = static_cast<float*>(buffers.input_pool().ptr);
    const auto* ready = static_cast<const int32_t*>(pub.device_ready_rows.ptr);
    const auto id = [&](int source, int row) { return source * 11 + row; };
    for (int iteration = 0; iteration < 2; ++iteration) {
        buffers.begin_router_notification_iteration(iteration);
        assert(*ready == 0);
        std::fill_n(pool, sources * 11, -1.0f);
        const auto notify = [&](int source, int start, int end) {
            const auto& slice = buffers.nvlink_receive_buffer_for_source(source / 8, source % 8).recv;
            auto* rows = static_cast<float*>(slice.ptr);
            for (int i = start; i < end; ++i) rows[i] = 1000 * iteration + id(source, i);
            buffers.process_router_notification_completion(source / 8, source % 8, iteration, start, end - start);
        };
        const std::vector<int32_t> first = iteration == 0
            ? std::vector<int32_t>{id(a, 1), id(a, 4), id(b, 0)}
            : std::vector<int32_t>{id(b, 0), id(b, 6), id(c, 3)};
        if (iteration == 0) {
            notify(a, 0, 11);
            notify(b, 0, 1);
        } else {
            notify(b, 0, 11);
            notify(c, 0, 11);
        }
        const auto check = [&](const std::vector<int32_t>& expected, int rows) {
            assert(*ready == rows);
            const std::array<std::array<int32_t, 4>, 6> descriptors{{
                {{0, 0, 3, 0}}, {{0, 0, 3, 2}}, {{0, 3, 5, 0}},
                {{0, 3, 5, 2}}, {{1, 5, 8, 0}}, {{1, 5, 8, 2}}
            }};
            for (int r = 0; r < rows; ++r) {
                const auto* row = table + r * 4;
                assert(std::equal(descriptors[r].begin(), descriptors[r].end(), row));
                for (int p = row[1]; p < row[2]; ++p) {
                    assert(flat[p] == expected[p]);
                    const auto weight = 1 + row[0] + row[3];
                    assert(pool[flat[p]] * weight == (1000 * iteration + expected[p]) * weight);
                }
            }
        };
        check(first, 2); // Remaining sources have not supplied their payloads.
        std::vector<int32_t> expected = first;
        if (iteration == 0) {
            notify(c, 0, 11);
            notify(b, 1, 11);
            expected.insert(expected.end(), {id(b, 6), id(c, 3)});
        } else {
            notify(a, 0, 11);
            expected.insert(expected.end(), {id(a, 1), id(a, 4)});
        }
        expected.insert(expected.end(), {id(a, 1), id(b, 6), id(c, 3)});
        check(expected, 6);
    }
}

int main() {
    for (int nodes : {1, 2})
        for (bool per_entry : {false, true}) check_stream(nodes, per_entry);
}
