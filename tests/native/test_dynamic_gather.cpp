#include "cuda_buffers.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace rdma_proxy;

static RouterExpertMetadata table(int source, const std::vector<int32_t>& rows) {
    RouterExpertMetadata m;
    m.source_node_rank = m.destination_node_rank = m.destination_gpu_index = 0;
    m.source_gpu_index = source;
    m.num_nodes = 1;
    m.num_gpus_per_node = 2;
    m.num_experts = 2;
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

int main() {
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
    CudaBuffers buffers(c);
    buffers.initialize();
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
