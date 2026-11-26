#pragma once

#include "config.hpp"
#include "protocol.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rdma_proxy {

class RouterRouting {
public:
    // Device tables and pinned readback storage can be reused after the previous
    // generation's device readers retire. Each instance owns its host snapshot.
    struct Workspace;
    static std::shared_ptr<Workspace> make_workspace(const ProxyConfig& config);
    explicit RouterRouting(ProxyConfig config, std::shared_ptr<Workspace> workspace = {});
    ~RouterRouting();

    RouterRouting(const RouterRouting&) = delete;
    RouterRouting& operator=(const RouterRouting&) = delete;

    void initialize();
    // Snapshot the completed benchmark's canonical source outputs. No GPU
    // pointer is retained after this call.
    void initialize_benchmark(uintptr_t device_r, uintptr_t device_node_offsets,
                              uintptr_t device_x3, uintptr_t device_x4);
    void initialize_benchmark_host(const int32_t* r, const int32_t* node_offsets,
                                   const int32_t* x3, const uint8_t* x4, uint64_t packed_count);
    void copy_benchmark_node_slice(const int32_t* x3, const uint8_t* x4, std::size_t count,
                                   std::vector<std::size_t>& indices,
                                   std::vector<uint8_t>& masks) const;
    const std::vector<std::size_t>& token_indices_for_node(int node_rank) const;
    const std::vector<uint8_t>& token_masks_for_node(int node_rank) const;
    const RouterExpertMetadata& expert_metadata_for_gpu(
        int node_rank,
        int local_gpu_index) const;
    const int32_t* device_token_indices_for_node(int node_rank) const;
    int experts_per_gpu() const;
    const std::vector<int32_t>& original_experts() const;
    const std::vector<float>& original_weights() const;
    struct DeviceTables {
        const int32_t* expert_offsets{};
        const int32_t* expert_indices{};
        const int32_t* node_offsets{};
        const int32_t* x3{};
        const uint8_t* x4{};
    };
    DeviceTables device_tables() const { return device_tables_; }

private:
    void require_fresh_benchmark_route() const;
    std::size_t benchmark_packed_rows(const int32_t* offsets) const;
    void copy_benchmark_nodes(const int32_t* offsets, const int32_t* x3, const uint8_t* x4);
    void finish_benchmark_route();
    void initialize_mock();
    void generate_source_weights();
    void align_expert_weights();
    void initialize_cuda();

    ProxyConfig config_;
    std::vector<std::vector<std::size_t>> token_indices_by_node_;
    std::vector<std::vector<uint8_t>> token_masks_by_node_;
    std::vector<RouterExpertMetadata> expert_metadata_by_gpu_;
    // Original [token, top_k] assignments retained independently of row packing.
    std::vector<int32_t> original_experts_;
    std::vector<float> original_weights_;
    std::shared_ptr<Workspace> workspace_;
    DeviceTables device_tables_{};
    bool initialized_{false};
};

}  // namespace rdma_proxy
