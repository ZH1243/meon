#include "iteration_routing.hpp"

#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace rdma_proxy;

void check_generations(int nodes, bool weighted) {
    ProxyConfig c;
    c.mock_mode = true;
    c.router_routing_enabled = true;
    c.num_nodes = nodes;
    c.node_rank = nodes - 1;
    c.local_gpu_index = 1;
    c.num_gpus_per_node = 2;
    c.router_num_experts = 32;
    c.router_top_k = 3;
    c.router_weighted_contributions_enabled = weighted;
    c.num_tokens = 256;
    c.router_seed = UINT64_MAX - 1; // Include wrap across source and iteration.
    IterationRouting worker(c);
    worker.initialize();
    const auto caller = std::this_thread::get_id();
    std::thread::id owner;
    int staged = 0;
    std::vector<int32_t> previous;
    bool changed_counts = false;
    std::vector<std::size_t> lengths;
    for (uint64_t generation = 0; generation < 4; ++generation) {
        auto data = worker.prepare(generation, [&](const RouterRouting& source, void*) {
            assert(!source.token_indices_for_node(c.node_rank).empty());
            assert(std::this_thread::get_id() != caller);
            if (staged++) assert(std::this_thread::get_id() == owner);
            owner = std::this_thread::get_id();
        });
        assert(data.source->original_experts() != previous);
        previous = data.source->original_experts();
        std::vector<std::size_t> actual_lengths;
        const int world = nodes * c.num_gpus_per_node;
        assert(int(data.incoming_experts.size()) == world);
        assert(int(data.incoming_nodes.size()) == nodes - 1);
        for (int i = 0; i < world; ++i) {
            auto cfg = c;
            cfg.node_rank = i / c.num_gpus_per_node;
            cfg.local_gpu_index = i % c.num_gpus_per_node;
            cfg.router_seed += (generation * world + i) * 0x9e3779b97f4a7c15ULL;
            RouterRouting ref(cfg);
            ref.initialize();
            const auto& actual = data.incoming_experts[i];
            const auto& expected = ref.expert_metadata_for_gpu(c.node_rank, c.local_gpu_index);
            assert(actual.source_node_rank == cfg.node_rank && actual.source_gpu_index == cfg.local_gpu_index);
            assert(actual.destination_node_rank == c.node_rank && actual.destination_gpu_index == c.local_gpu_index);
            assert(actual.expert_offsets == expected.expert_offsets);
            assert(actual.expert_token_indices == expected.expert_token_indices);
            assert(actual.expert_route_weights == expected.expert_route_weights);
            actual_lengths.push_back(actual.expert_token_indices.size());
            if (cfg.node_rank != c.node_rank && cfg.local_gpu_index == c.local_gpu_index) {
                const auto& route = data.incoming_nodes.at(cfg.node_rank);
                assert(route.token_indices == ref.token_indices_for_node(c.node_rank));
                assert(route.token_masks == ref.token_masks_for_node(c.node_rank));
            }
        }
        if (generation && lengths != actual_lengths) changed_counts = true;
        lengths = actual_lengths;
    }
    assert(changed_counts && staged == 4);
    bool rejected = false;
    try { worker.prepare(3); } catch (const std::logic_error&) { rejected = true; }
    assert(rejected);
    worker.shutdown();
}

// A failed epoch cannot be retried against half-published buffers. A concurrent
// shutdown must keep a borrowed staging callback alive until its work retires.
void check_worker_lifetime() {
    ProxyConfig c;
    c.mock_mode = c.router_routing_enabled = true;
    c.num_nodes = 1;
    c.num_gpus_per_node = c.router_num_experts = 2;
    c.router_top_k = 1;
    c.num_tokens = 4;
    IterationRouting worker(c);
    worker.initialize();
    std::promise<void> entered, resume;
    auto release = resume.get_future();
    auto preparing = std::async(std::launch::async, [&] {
        try {
            worker.prepare(0, [&](const RouterRouting&, void*) {
                entered.set_value();
                release.wait();
                throw std::runtime_error("injected staging failure");
            });
        } catch (const std::runtime_error& e) {
            return std::string(e.what()) == "injected staging failure";
        }
        return false;
    });
    entered.get_future().wait();
    auto closing = std::async(std::launch::async, [&] {
        try { worker.shutdown(); }
        catch (const std::runtime_error& e) { return std::string(e.what()) == "injected staging failure"; }
        return false;
    });
    assert(preparing.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout);
    resume.set_value();
    assert(preparing.get() && closing.get());
    bool rejected = false;
    try { worker.prepare(0); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}

int main() {
    check_worker_lifetime();
    check_generations(1, false);
    check_generations(2, true);
    std::cout << "checked source-specific dynamic routing, weights, staging and generation order\n";
}
