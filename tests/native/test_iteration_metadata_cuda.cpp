#include "iteration_metadata_exchange.hpp"
#include "router_routing.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>

#if RDMA_PROXY_HAVE_CUDA
#include <cuda_runtime.h>

namespace {
using namespace rdma_proxy;
namespace wire = iteration_metadata;
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

// Exercise both banks, weighted records and unused tails on one physical GPU.
// IPC opens, peer stores and GPUDirect completions need an additional fabric run.
void check_records(int nodes, bool weighted) {
    ProxyConfig c;
    c.router_routing_enabled = true;
    c.num_gpus_per_node = 2;
    c.num_nodes = nodes;
    c.node_rank = nodes - 1;
    c.local_gpu_index = 1;
    c.router_num_experts = 16;
    c.router_top_k = 2;
    c.router_weighted_contributions_enabled = weighted;
    const uint64_t epoch = UINT64_C(0x100000003);
    for (const int tokens : {1, 257}) {
        c.num_tokens = tokens;
        RouterRouting source(c);
        source.initialize();
        auto cpu_config = c;
        cpu_config.mock_mode = true;
        RouterRouting reference(cpu_config);
        reference.initialize();
        const auto l = wire::make_layout(c);
        wire::PeerArenas peers;
        for (int i = 0; i < l.gpus; ++i) {
            check(cudaMalloc(&peers.address[i], l.bytes));
            check(cudaMemset(peers.address[i], 0x5a, l.bytes));
        }
        float* device_weights = nullptr;
        std::vector<float> weights;
        if (weighted) {
            for (int i = 0; i < nodes * l.gpus; ++i) {
                const auto& m = source.expert_metadata_for_gpu(i / l.gpus, i % l.gpus);
                weights.insert(weights.end(), m.expert_route_weights.begin(), m.expert_route_weights.end());
            }
            check(cudaMalloc(&device_weights, weights.size() * sizeof(float)));
            check(cudaMemcpy(device_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
        const auto tables = source.device_tables();
        wire::publish(l, peers, c.node_rank, c.local_gpu_index, epoch, tables.expert_offsets, tables.expert_indices,
                      device_weights, tables.node_offsets, tables.x3, tables.x4, nullptr);
        check(cudaDeviceSynchronize());
        for (int gpu = 0; gpu < l.gpus; ++gpu) {
            std::vector<uint64_t> arena(l.bytes / sizeof(uint64_t));
            check(cudaMemcpy(arena.data(), peers.address[gpu], l.bytes, cudaMemcpyDeviceToHost));
            const auto* raw = reinterpret_cast<const char*>(arena.data());
            const auto* directory = reinterpret_cast<const wire::Header*>(raw);
            for (int node = 0; node < l.nodes; ++node) {
                const auto slot = node * l.gpus + c.local_gpu_index;
                const char* record = raw + l.directory + slot * l.stride;
                const auto* h = reinterpret_cast<const wire::Header*>(record);
                const auto& m = reference.expert_metadata_for_gpu(node, gpu);
                assert(std::memcmp(h, directory + slot, sizeof(*h)) == 0);
                assert(h->sequence == epoch && h->source == c.node_rank * l.gpus + c.local_gpu_index);
                assert(h->destination == node * l.gpus + gpu && h->experts == l.experts);
                assert(h->version == 1 && h->weighted == weighted && size_t(h->count) == m.expert_token_indices.size());
                const auto* offsets = reinterpret_cast<const int32_t*>(h + 1);
                const auto* indices = offsets + l.experts + 1;
                assert(std::equal(m.expert_offsets.begin(), m.expert_offsets.end(), offsets));
                assert(std::equal(m.expert_token_indices.begin(), m.expert_token_indices.end(), indices));
                assert(std::equal(m.expert_route_weights.begin(), m.expert_route_weights.end(),
                                  reinterpret_cast<const float*>(indices + h->count)));
                assert(std::all_of(record + wire::expert_bytes(l, h->count), record + l.stride,
                                   [](char v) { return v == 0x5a; }));
            }
            if (nodes == 2 && gpu == c.local_gpu_index) {
                const auto* h = reinterpret_cast<const wire::Header*>(raw + l.x3_offset);
                const auto* ids = reinterpret_cast<const int32_t*>(h + 1);
                const auto* masks = reinterpret_cast<const uint8_t*>(ids + h->count);
                const auto& expected = reference.token_indices_for_node(1 - c.node_rank);
                assert(h->sequence == epoch && size_t(h->count) == expected.size());
                assert(h->source == c.node_rank * l.gpus + gpu && h->destination == (1-c.node_rank) * l.gpus + gpu);
                assert(std::equal(expected.begin(), expected.end(), ids));
                const auto& x4 = reference.token_masks_for_node(1 - c.node_rank);
                assert(std::equal(x4.begin(), x4.end(), masks));
            }
            check(cudaFree(peers.address[gpu]));
        }
        if (device_weights) check(cudaFree(device_weights));
    }
}
}
#endif

int main() {
#if RDMA_PROXY_HAVE_CUDA
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    check_records(1, false);
    check_records(2, true);
    std::cout << "validated iteration metadata exchange CUDA records\n";
    return 0;
#else
    return 77;
#endif
}
