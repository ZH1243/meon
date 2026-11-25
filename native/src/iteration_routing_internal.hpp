#pragma once

#include "iteration_routing.hpp"
#include <atomic>

namespace rdma_proxy {

inline RouterX3Metadata iteration_node_record(const ProxyConfig& c, int source, int destination) {
    RouterX3Metadata result;
    result.source_node_rank = source;
    result.destination_node_rank = destination;
    result.local_gpu_index = c.local_gpu_index;
    result.num_nodes = c.num_nodes;
    result.num_gpus_per_node = c.num_gpus_per_node;
    result.num_experts = c.router_num_experts;
    result.top_k = c.router_top_k;
    result.num_tokens = c.num_tokens;
    result.token_dimension = c.token_dimension;
    result.element_bytes = dtype_size(c.dtype);
    result.tokens_per_chunk = c.tokens_per_chunk;
    return result;
}

class IterationMetadataTransport {
public:
    virtual ~IterationMetadataTransport() = default;
    virtual void initialize() = 0;
    virtual void exchange(uint64_t generation, IterationRouteData&) = 0;
    virtual void finish(uint64_t generation) = 0;
    virtual void close(bool clean) = 0;
};
std::unique_ptr<IterationMetadataTransport> make_iteration_metadata_transport(
    const ProxyConfig&, const std::atomic<bool>& stopping);
}
