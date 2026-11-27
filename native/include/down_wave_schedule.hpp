#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rdma_proxy {

struct DownWaveGeometry {
    std::uint64_t cluster_rows;
    std::uint64_t num_groups;
    std::uint64_t group_size;
    std::uint64_t max_wave_work_ids;
};

struct DownWaveLayout {
    std::size_t table_rows;
    std::size_t prefix_entries;
    std::size_t wave_work_ids;
    std::size_t bytes;
};

// Reserve for any distribution of up to max_routes among num_experts.
DownWaveLayout down_wave_capacity(const DownWaveGeometry& geometry,
                                  std::uint64_t max_routes, std::size_t num_experts);
DownWaveLayout down_wave_layout(const DownWaveGeometry& geometry,
                                const std::vector<std::size_t>& counts);

// Write QuACK's [expert, M start, M end, N base] descriptors, then its
// wave-ready prefixes, contiguously. An empty route has no payload.
DownWaveLayout build_down_wave_schedule(const DownWaveGeometry& geometry,
                                        const std::vector<std::size_t>& counts,
                                        std::int32_t* packed, std::size_t capacity_bytes);

}  // namespace rdma_proxy
