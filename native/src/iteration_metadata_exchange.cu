#include "iteration_metadata_exchange.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace rdma_proxy::iteration_metadata {
namespace {
__global__ void publish_records(Layout l, PeerArenas peers, int node, int gpu,
                                uint64_t sequence, const int32_t* offsets,
                                const int32_t* indices, const float* weights,
                                const int32_t* node_offsets, const int32_t* x3, const uint8_t* x4) {
    const int dest = blockIdx.x;
    const int source = node * l.gpus + gpu;
    if (dest == l.nodes * l.gpus) {
        const int other = 1 - node;
        const int first = node_offsets[other];
        const int count = node_offsets[other + 1] - first;
        char* arena = peers.address[gpu];
        auto* h = reinterpret_cast<Header*>(arena + l.x3_offset);
        if (!threadIdx.x) {
            *h = {sequence, source, other * l.gpus + gpu, count, 0, 0, 1};
            reinterpret_cast<Header*>(arena)[dest] = *h;
        }
        auto* out = reinterpret_cast<int32_t*>(h + 1);
        auto* masks = reinterpret_cast<uint8_t*>(out + count);
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            out[i] = x3[first + i];
            masks[i] = x4[first + i];
        }
    } else {
        const int slot = (dest / l.gpus) * l.gpus + gpu;
        const int first_exp = dest * l.experts;
        const int first = offsets[first_exp];
        const int count = offsets[first_exp + l.experts] - first;
        char* arena = peers.address[dest % l.gpus];
        auto* h = reinterpret_cast<Header*>(arena + l.directory + size_t(slot) * l.stride);
        if (!threadIdx.x) {
            *h = {sequence, source, dest, count, l.experts, l.weighted, 1};
            reinterpret_cast<Header*>(arena)[slot] = *h;
        }
        auto* off = reinterpret_cast<int32_t*>(h + 1);
        auto* output = off + l.experts + 1;
        auto* out_w = reinterpret_cast<float*>(output + count);
        for (int i = threadIdx.x; i <= l.experts; i += blockDim.x)
            off[i] = offsets[first_exp + i] - first;
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            output[i] = indices[first + i];
            if (l.weighted) out_w[i] = weights[first + i];
        }
    }
    // All peer writes become visible before the owning stream completes;
    // the CPU producer barrier then releases the relays to read their arenas.
    __threadfence_system();
}
}
void publish(const Layout& l, const PeerArenas& peers, int node, int gpu, uint64_t sequence,
             const int32_t* offsets, const int32_t* indices, const float* weights,
             const int32_t* node_offsets, const int32_t* x3, const uint8_t* x4,
             void* stream) {
    publish_records<<<l.nodes * l.gpus + (l.nodes == 2), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        l, peers, node, gpu, sequence, offsets, indices, weights, node_offsets, x3, x4);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string("publish routing records: ") + cudaGetErrorString(error));
}
}
