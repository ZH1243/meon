#pragma once

#include <cstdint>

namespace rdma_proxy {

// Passed by value as CUDA kernel arguments. Each pointer starts at the first
// expert index, after any exchange record header.
struct BenchmarkIndexPadding {
    static constexpr std::uint32_t sources = 16;
    std::int32_t* indices[sources]{};
    std::uint32_t counts[sources]{};
    std::uint32_t common_count{};
};

void clear_benchmark_index_padding(const BenchmarkIndexPadding& padding, std::uint32_t max_padding);

}  // namespace rdma_proxy
