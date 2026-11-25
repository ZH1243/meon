#pragma once

#include "config.hpp"

#include <cstddef>
#include <cstdint>

namespace rdma_proxy::iteration_metadata {

// A local destination GPU relays its corresponding GPU on the other node.
// Slot destination_node * gpus + source_gpu identifies the producer on the GPU
// arena. On the receiver's pinned arena the slot uses the original source node.
struct Header {
    uint64_t sequence;
    int32_t source, destination, count, experts, weighted, version;
};
static_assert(sizeof(Header) == 32);
struct Layout {
    int nodes, gpus, experts, tokens, top_k, weighted;
    std::size_t directory, stride, x3_offset, control_offset, bytes;
};
Layout make_layout(const ProxyConfig&);
struct PeerArenas { char* address[8]{}; };
inline std::size_t expert_bytes(const Layout& l, std::size_t count) {
    return sizeof(Header) + (l.experts + 1 + count * (l.weighted ? 2u : 1u)) * sizeof(int32_t);
}
inline std::size_t x3_bytes(std::size_t count) {
    return sizeof(Header) + count * (sizeof(int32_t) + sizeof(uint8_t));
}
void publish(const Layout&, const PeerArenas&, int node, int gpu, uint64_t sequence,
             const int32_t* offsets, const int32_t* indices, const float* weights,
             const int32_t* node_offsets, const int32_t* x3, const uint8_t* x4,
             void* stream);
}
