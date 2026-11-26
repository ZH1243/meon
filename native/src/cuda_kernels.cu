#include <cuda_runtime.h>

#include "benchmark_index_padding.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace rdma_proxy {

namespace {

__global__ void copy_bytes_kernel(std::uint8_t* dst, const std::uint8_t* src, std::size_t bytes) {
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t stride = blockDim.x * gridDim.x;
    for (std::size_t i = tid; i < bytes; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void clear_index_tails_kernel(BenchmarkIndexPadding padding) {
    const auto source = blockIdx.y;
    auto* indices = padding.indices[source];
    if (!indices) return;
    const auto start = padding.counts[source] + blockIdx.x * blockDim.x + threadIdx.x;
    for (auto row = start; row < padding.common_count; row += blockDim.x * gridDim.x)
        indices[row] = 0;
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

void clear_benchmark_index_padding(const BenchmarkIndexPadding& padding, std::uint32_t max_padding) {
    if (!max_padding) return;
    constexpr unsigned threads = 256;
    const auto blocks = std::min(32u, 1u + (max_padding - 1u) / threads);
    clear_index_tails_kernel<<<dim3(blocks, BenchmarkIndexPadding::sources), threads, 0, cudaStreamLegacy>>>(padding);
    check_cuda(cudaGetLastError(), "clear benchmark A_idx padding");
}

void launch_copy_tokens_kernel(void* dst, const void* src, std::size_t bytes) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>((bytes + threads - 1) / threads);
    copy_bytes_kernel<<<std::max(1, blocks), threads>>>(
        static_cast<std::uint8_t*>(dst),
        static_cast<const std::uint8_t*>(src),
        bytes);
    check_cuda(cudaGetLastError(), "copy_bytes_kernel launch");
}

}  // namespace rdma_proxy
