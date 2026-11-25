#include "logging.hpp"
#include "proxy.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <stdexcept>

using namespace rdma_proxy;

namespace {
constexpr int gpus = 8, nodes = 2, experts_per_gpu = 8, tokens = 2;
constexpr int row_bytes = 16;
constexpr int combine_row_stride = nodes * tokens + 2;

struct Rank {
    std::vector<std::unique_ptr<uint8_t[]>> allocations;
    uint8_t* send{};
    uint8_t* rdma_receive{};
    std::array<uint8_t*, nodes*gpus> inputs{};
    std::array<int32_t, nodes * gpus * combine_row_stride> combine_rows{};
    std::array<int32_t, nodes * (tokens + 2)> inter_rows{};
    std::unique_ptr<Proxy> proxy;
};

template<class F> void on_all_gpus(F work) {
    std::vector<std::future<void>> jobs;
    for (int gpu = 0; gpu < gpus; ++gpu)
        jobs.emplace_back(std::async(std::launch::async, work, gpu));
    for (auto& job : jobs) job.get();
}

template<class F> void expect_error(F operation) {
    try { operation(); } catch (const std::runtime_error&) { return; }
    throw std::runtime_error("accepted invalid benchmark routing operation");
}

// Each token chooses all eight experts on one GPU. The mock NIC reflects
// payloads, so incoming and outgoing remote compact buffers have equal lengths.
int destination_node(int epoch, int token) {
    if (epoch == 0) return 0;
    if (epoch == 1) return 1;
    return token == 0 ? 1 : 0;
}
int destination_gpu(int epoch, int source_gpu, int source_node) {
    if (epoch != 2) return source_gpu;
    // Local and remote original sources use different neighbors, so reusing
    // an outgoing mask for the same-index incoming record is observable.
    return (source_gpu + (source_node == 0 ? 1 : gpus - 1)) % gpus;
}
uint8_t payload_value(int epoch, int gpu, int token) {
    return static_cast<uint8_t>(epoch*40 + gpu*3 + token + 1);
}

void install(Rank& rank, int gpu, int epoch) {
    std::array<std::vector<int32_t>, nodes> rows;
    std::vector<int32_t> r;
    const auto dest = destination_gpu(epoch, gpu, 0);
    for (int token = 0; token < tokens; ++token) {
        const auto node = destination_node(epoch, token);
        const auto first_expert = (node*gpus + dest) * experts_per_gpu;
        for (int k = 0; k < experts_per_gpu; ++k) r.push_back(first_expert + k);
        const auto value = payload_value(epoch, gpu, token);
        if (node == 0) {
            // Only the completed external gather has the correct bytes. A
            // second native gather would overwrite them with this poison.
            std::memset(rank.inputs[gpu] + rows[0].size()*row_bytes, value, row_bytes);
            std::memset(rank.send + token*row_bytes, 0xff, row_bytes);
        } else {
            std::memset(rank.send + token*row_bytes, value, row_bytes);
        }
        rows[node].push_back(token);
    }
    std::vector<int32_t> x3 = rows[0];
    x3.insert(x3.end(), rows[1].begin(), rows[1].end());
    const auto mask = static_cast<uint8_t>(1u << ((gpu - dest + gpus) % gpus));
    std::vector<uint8_t> x4(x3.size(), mask);
    const int remote_dest = destination_gpu(epoch, gpu, 1);
    auto remote_rows = rows[1];
    if (epoch == 2) remote_rows = {1};  // This GPU sends token 0 and receives the remote source's token 1.
    std::vector<uint8_t> remote_masks(remote_rows.size(), 1u << ((gpu - remote_dest + gpus) % gpus));
    const int32_t node_offsets[] = {0, static_cast<int32_t>(rows[0].size()), tokens};
    std::array<std::array<int32_t, experts_per_gpu+1>, nodes*gpus> offsets{};
    std::array<std::vector<int32_t>, nodes*gpus> indices;
    std::array<RdmaProxyBenchmarkSource, nodes*gpus> sources;
    for (int s = 0; s < nodes*gpus; ++s) {
        const auto count = destination_gpu(epoch, s % gpus, s / gpus) == gpu ? rows[s/gpus].size() : 0;
        for (int expert = 0; expert < experts_per_gpu; ++expert) {
            for (std::size_t i = 0; i < count; ++i) indices[s].push_back(static_cast<int32_t>(i));
            offsets[s][expert+1] = static_cast<int32_t>(indices[s].size());
        }
        sources[s] = {offsets[s].data(), indices[s].data(), indices[s].size()};
    }
    RdmaProxyBenchmarkRouting data{sizeof(data), nodes*gpus, sources.data(), remote_rows.data(),
        remote_masks.data(), remote_rows.size(), reinterpret_cast<uintptr_t>(r.data()),
        reinterpret_cast<uintptr_t>(node_offsets), reinterpret_cast<uintptr_t>(x3.data()),
        reinterpret_cast<uintptr_t>(x4.data())};
    expect_error([&] { rank.proxy->prepare_routing(epoch); });
    if (!epoch) {
        expect_error([&] { rank.proxy->install_benchmark_routing(1, data); });
        auto corrupt = data;
        --corrupt.num_sources;
        expect_error([&] { rank.proxy->install_benchmark_routing(epoch, corrupt); });
        corrupt = data;
        --corrupt.struct_size;
        expect_error([&] { rank.proxy->install_benchmark_routing(epoch, corrupt); });
        corrupt = data;
        corrupt.remote_count = tokens + 1;
        expect_error([&] { rank.proxy->install_benchmark_routing(epoch, corrupt); });
        const auto saved = offsets[gpu][1];
        offsets[gpu][1] = -1;
        expect_error([&] { rank.proxy->install_benchmark_routing(epoch, data); });
        offsets[gpu][1] = saved;
    }
    struct Extended { RdmaProxyBenchmarkRouting current; uint64_t extension; };
    Extended extended{data, 0};
    extended.current.struct_size = sizeof(extended);
    rank.proxy->install_benchmark_routing(epoch, extended.current);
    expect_error([&] { rank.proxy->install_benchmark_routing(epoch, data); });
    rank.proxy->prepare_routing(epoch);
    // The external buffers may now be reused, including the remote pinned views.
    std::fill(x3.begin(), x3.end(), -1);
    std::fill(x4.begin(), x4.end(), 0);
    std::fill(remote_masks.begin(), remote_masks.end(), 0);
    const auto expected_r = r;
    std::fill(r.begin(), r.end(), -1);
    for (auto& source_indices : indices) std::fill(source_indices.begin(), source_indices.end(), -1);
    for (auto& source_offsets : offsets) source_offsets.fill(-1);
    const auto& weights = rank.proxy->original_router_weights();
    if (rank.proxy->original_router_experts() != expected_r || weights != std::vector<float>(r.size(), 1))
        throw std::runtime_error("incorrect source routing snapshot");
    for (int n = 0; n < nodes; ++n) {
        const auto& outgoing = rank.proxy->outgoing_token_indices(n);
        if (!std::equal(outgoing.begin(), outgoing.end(), rows[n].begin(), rows[n].end()) ||
            rank.proxy->intra_node_source_masks(n) !=
                std::vector<uint8_t>(rows[n].size(), 1u << destination_gpu(epoch, gpu, n)))
            throw std::runtime_error("incorrect outgoing rows or incoming source masks");
    }
    // Use remote-first order for both combine kernels, including when one
    // domain is empty or shrinks after being full in the previous generation.
    const std::array<int32_t, nodes> domain_nodes{1, 0};
    if (!epoch) {
        rank.inter_rows.fill(37);
        for (int32_t bad_node : {1, -1, nodes}) {
            const std::array<int32_t, nodes> invalid{1, bad_node};
            expect_error([&] {
                rank.proxy->build_inter_node_combine_rows(invalid.data(), nodes, tokens,
                    rank.inter_rows.data(), rank.inter_rows.size());
            });
            if (!std::all_of(rank.inter_rows.begin(), rank.inter_rows.end(), [](int32_t x) { return x == 37; }))
                throw std::runtime_error("partial inter-node combine table after invalid destinations");
        }
    }
    rank.proxy->build_inter_node_combine_rows(domain_nodes.data(), nodes, tokens,
        rank.inter_rows.data(), rank.inter_rows.size());
    // Expected inverse of outgoing x3. Include missing tokens, the packed stride
    // and the zero trailer in every comparison so stale storage is observable.
    constexpr int inter_stride = tokens + 2;
    for (int domain = 0; domain < nodes; ++domain) {
        const auto& outgoing = rows[domain_nodes[domain]];
        std::array<int32_t, inter_stride> expected;
        expected.fill(-1);
        for (std::size_t slot = 0; slot < outgoing.size(); ++slot)
            expected[outgoing[slot]] = static_cast<int32_t>(slot);
        expected[tokens] = static_cast<int32_t>(outgoing.size());
        expected[tokens + 1] = 0;
        if (!std::equal(expected.begin(), expected.end(), rank.inter_rows.begin() + domain * inter_stride))
            throw std::runtime_error("incorrect original-token to outgoing x3 lookup");
    }
    std::array<int32_t, inter_stride> remote_only;
    rank.proxy->build_inter_node_combine_rows(domain_nodes.data(), 1, tokens,
        remote_only.data(), remote_only.size());
    if (!std::equal(remote_only.begin(), remote_only.end(), rank.inter_rows.begin()))
        throw std::runtime_error("incorrect remote-only inter-node combine lookup");
    std::array<uint64_t, nodes> domain_counts{};
    if (!epoch) {
        rank.combine_rows.fill(37);
        domain_counts.fill(99);
        const std::array<int32_t, nodes> duplicate{1, 1};
        expect_error([&] {
            rank.proxy->build_intra_node_combine_rows(duplicate.data(), nodes, tokens,
                rank.combine_rows.data(), rank.combine_rows.size(), domain_counts.data());
        });
        if (!std::all_of(rank.combine_rows.begin(), rank.combine_rows.end(), [](int32_t x) { return x == 37; }) ||
            domain_counts != std::array<uint64_t, nodes>{99, 99})
            throw std::runtime_error("partial combine table after invalid source nodes");
    }
    rank.proxy->build_intra_node_combine_rows(domain_nodes.data(), nodes, tokens,
        rank.combine_rows.data(), rank.combine_rows.size(), domain_counts.data());
    for (int domain = 0; domain < nodes; ++domain) {
        const int node = domain_nodes[domain];
        const int valid = static_cast<int>(rows[node].size());
        if (domain_counts[domain] != static_cast<uint64_t>(valid))
            throw std::runtime_error("incorrect combine domain length");
        for (int peer = 0; peer < gpus; ++peer) {
            const bool selected = peer == destination_gpu(epoch, gpu, node);
            const auto* row = rank.combine_rows.data() + (domain * gpus + peer) * combine_row_stride;
            for (int column = 0; column < nodes * tokens; ++column) {
                const int local = column - domain * tokens;
                const int expected = selected && local >= 0 && local < valid ? local : -1;
                if (row[column] != expected) throw std::runtime_error("incorrect combine lookup row");
            }
            if (row[combine_row_stride - 2] != (selected ? valid : 0) || row[combine_row_stride - 1] != valid)
                throw std::runtime_error("incorrect combine input stride or domain count");
        }
    }
}
}

int main() {
    Logger::instance().set_level(LogLevel::kError);
    ProxyConfig config;
    config.num_nodes = nodes;
    config.num_gpus_per_node = gpus;
    config.num_tokens = tokens;
    config.token_dimension = row_bytes / dtype_size(config.dtype);
    config.num_iterations = 3;
    config.mock_mode = true;
    config.fill_test_data = config.validate_data = false;
    config.router_routing_enabled = config.router_dynamic_routing_enabled = true;
    config.router_benchmark_metadata_enabled = config.router_local_input_staging_enabled = true;
    config.router_num_experts = nodes*gpus*experts_per_gpu;
    config.router_top_k = experts_per_gpu;
    config.nvlink_forwarding_enabled = config.nvlink_forward_synchronize_batches = true;
    config.nvlink_forward_completion_notifications_enabled = true;
    config.nvlink_forward_threshold_tokens = config.nvlink_forward_chunk_tokens = 1;
    config.local_iteration_sync_enabled = true;
    config.completion_timeout_ms = 10000;
    config.peers.push_back(PeerAddress{1, "mock-peer", 18515});
    const auto id = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    config.local_iteration_sync_run_id = "bench_handoff_" + id;
    config.nvlink_forward_exchange_dir = "/tmp/meon_bench_handoff_" + id;
    std::array<Rank, gpus> ranks;
    for (int gpu = 0; gpu < gpus; ++gpu) {
        auto& rank = ranks[gpu];
        config.cuda_device_id = config.local_gpu_index = gpu;
        rank.proxy = std::make_unique<Proxy>(config);
        rank.proxy->set_external_device_buffer_allocator([&rank, gpu](const DeviceBufferAllocationRequest& r) {
            const auto source = r.source_node_rank*gpus + r.source_gpu_index;
            if (r.kind == DeviceBufferKind::kNvlinkReceive && source == gpus + gpu)
                return rank.inputs[source] = rank.rdma_receive;
            rank.allocations.emplace_back(std::make_unique<uint8_t[]>(r.bytes));
            auto* ptr = rank.allocations.back().get();
            if (r.kind == DeviceBufferKind::kRdmaSend) rank.send = ptr;
            if (r.kind == DeviceBufferKind::kRdmaReceive) rank.rdma_receive = ptr;
            if (r.kind == DeviceBufferKind::kNvlinkReceive) rank.inputs.at(source) = ptr;
            return ptr;
        });
    }
    on_all_gpus([&](int gpu) { ranks[gpu].proxy->initialize(); });
    for (int epoch = 0; epoch < 3; ++epoch) {
        on_all_gpus([&](int gpu) {
            install(ranks[gpu], gpu, epoch);
            ranks[gpu].proxy->run_iteration_step(epoch);
        });
        for (int source = 0; source < gpus; ++source) {
            std::array<std::size_t, nodes> row{};
            for (int token = 0; token < tokens; ++token) {
                const int node = destination_node(epoch, token);
                const auto& rank = ranks[destination_gpu(epoch, source, node)];
                auto* input = rank.inputs[node*gpus + source] + row[node]++ * row_bytes;
                if (!std::all_of(input, input + row_bytes, [&](uint8_t v) {
                        return v == payload_value(epoch, source, token); }))
                    throw std::runtime_error("benchmark handoff forwarded incorrect payload");
            }
            for (int node = 0; node < nodes; ++node) {
                const auto& rank = ranks[destination_gpu(epoch, source, node)];
                const auto& table = rank.proxy->router_expert_metadata_for_source(node, source);
                const auto state = rank.proxy->router_expert_token_head_state_for_source(node, source);
                if (table.expert_token_indices.size() != row[node]*experts_per_gpu ||
                    state.received_token_frontier != row[node])
                    throw std::runtime_error("incorrect benchmark source table or completion frontier");
            }
        }
    }
    on_all_gpus([&](int gpu) { ranks[gpu].proxy->shutdown(); });
}
