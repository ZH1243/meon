#include "logging.hpp"
#include "proxy.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <future>
#include <iterator>
#include <poll.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>

using namespace rdma_proxy;

namespace rdma_proxy {
struct BenchmarkRoutingTestAccess {
    static uint64_t completed_local_iterations(const Proxy& proxy) {
        return proxy.local_router_forwarding_completed_iterations_.load();
    }

    // Use the real start/done framing with mock QPs. The test controls exactly
    // when the opposite same-index GPU replies, without a listening TCP port.
    static std::unique_ptr<IterationControlChannel> attach_peer(Proxy& proxy, int& peer_fd) {
        int fds[2];
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds)) throw std::runtime_error("socketpair failed");
        auto client = std::unique_ptr<IterationControlChannel>(new IterationControlChannel(fds[0], true, 1));
        auto server = std::unique_ptr<IterationControlChannel>(new IterationControlChannel(fds[1], false, 0));
        for (int fd : fds)
            if (::fcntl(fd, F_SETFL, O_NONBLOCK) < 0) throw std::runtime_error("nonblocking control socket failed");
        proxy.peers_.at(0).iteration_control = std::move(client);
        peer_fd = fds[1];
        return server;
    }
};
}

namespace {
constexpr int gpus = 8, nodes = 2, experts_per_gpu = 8, tokens = 2;
constexpr int row_bytes = 16;
constexpr int combine_row_stride = nodes * tokens + 2;

struct Rank {
    std::vector<std::unique_ptr<uint8_t[]>> allocations;
    uint8_t* send{};
    uint8_t* rdma_receive{};
    std::array<uint8_t*, nodes*gpus> inputs{};
    std::array<int32_t*, nodes*gpus> device_indices{};
    std::array<std::array<int32_t, experts_per_gpu+1>, nodes*gpus> offsets{};
    std::array<std::vector<int32_t>, nodes*gpus> indices;
    // Receiver-owned storage. Keep its addresses alive through retirement or
    // shutdown, including when a new install fails after accepting a view.
    std::array<int32_t, nodes + 1> source_offsets{};
    std::array<int32_t, tokens> source_x3{};
    std::array<uint8_t, tokens> source_x4{};
    std::array<int32_t, tokens * experts_per_gpu> source_r{};
    std::array<int32_t, nodes * gpus * combine_row_stride> combine_rows{};
    std::array<int32_t, nodes * (tokens + 2)> inter_rows{};
    std::array<int32_t, nodes * gpus * combine_row_stride + nodes * (tokens + 2)> packed_maps{};
    alignas(16) std::array<int64_t, 2 * nodes*gpus*tokens*experts_per_gpu> scatter_slab{};
    std::array<int64_t, nodes*gpus> scatter_counts{};
    std::unique_ptr<Proxy> proxy;

    void bind_scatter(std::size_t bytes) {
        std::array<int32_t, 2 * nodes*gpus> pairs;
        std::array<uint64_t, nodes*gpus> bases;
        for (int i = 0; i < nodes*gpus; ++i) {
            pairs[2*i] = i / gpus;
            pairs[2*i + 1] = i % gpus;
            bases[i] = reinterpret_cast<uintptr_t>(inputs[i]);
        }
        RdmaProxyCpuScatterConfig scatter{sizeof(scatter), nodes*gpus, 0, tokens, row_bytes / 2, 4, 2,
            reinterpret_cast<uintptr_t>(scatter_slab.data()), pairs.data(), bases.data(), scatter_counts.data(), nullptr};
        proxy->bind_native_local_publication(1, &scatter, bytes);
    }

    std::array<uintptr_t, nodes*gpus> index_addresses() const {
        std::array<uintptr_t, nodes*gpus> addresses{};
        for (int source = 0; source < nodes*gpus; ++source)
            addresses[source] = reinterpret_cast<uintptr_t>(device_indices[source]);
        return addresses;
    }

    void poison_expert_records() {
        for (auto& source : indices) std::fill(source.begin(), source.end(), -1);
        for (auto& source : offsets) source.fill(-1);
    }

    void poison_source_route() {
        source_offsets.fill(-1);
        source_x3.fill(-1);
        source_x4.fill(0);
        source_r.fill(-1);
    }
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

bool borrow_source(int epoch, int gpu) {
    return (epoch == 1 && gpu % 3 == 0) || (epoch == 2 && gpu % 2 == 0);
}

void check_borrowed_source_views(const ProxyConfig& config) {
    // Equal token IDs across destination nodes are valid. The two slices have
    // different lengths, so incorrect offsets or a lost mask are observable.
    std::array<int32_t, nodes + 1> offsets{0, 2, 3};
    std::array<int32_t, 3> x3{1, 0, 1};
    std::array<uint8_t, 3> x4{2, 1, 4};
    std::array<int32_t, tokens * experts_per_gpu> r{};
    for (std::size_t i = 0; i < r.size(); ++i) r[i] = static_cast<int32_t>(i);
    auto check_install = [&](bool should_reject) {
        auto borrow = [&] {
            RouterRouting route(config);
            route.borrow_benchmark_host(r.data(), offsets.data(), x3.data(), x4.data(), x3.size());
        };
        auto copy = [&] {
            RouterRouting route(config);
            route.initialize_benchmark_host(r.data(), offsets.data(), x3.data(), x4.data(), x3.size());
        };
        if (should_reject) {
            expect_error(borrow);
            expect_error(copy);
        } else {
            borrow();
            copy();
        }
    };
    const bool validate = config.router_benchmark_metadata_validation_enabled;
    offsets[1] = 3;
    check_install(true);
    offsets[1] = 2;
    x3[0] = tokens;
    check_install(true);
    x3[0] = -1;
    check_install(true);
    x3[0] = 0;
    check_install(validate); // Duplicate original token in the first node.
    x3[0] = 1;
    x4[0] = 0;
    check_install(validate);
    x4[0] = 1;
    check_install(validate); // Equal masks require increasing original token IDs.
    x4[0] = 2;
    r[0] = config.router_num_experts;
    check_install(validate);
    r[0] = 0;

    RouterRouting route(config);
    route.borrow_benchmark_host(r.data(), offsets.data(), x3.data(), x4.data(), x3.size());
    if (!route.has_borrowed_source_route() || route.original_experts_view().data() != r.data())
        throw std::runtime_error("borrowed expert IDs do not reference the receiver");
    for (int node = 0; node < nodes; ++node) {
        const auto indices = route.token_indices_view(node);
        const auto masks = route.token_masks_view(node);
        const auto first = offsets[node];
        const auto size = static_cast<std::size_t>(offsets[node + 1] - first);
        if (indices.compact_data() != x3.data() + first || masks.data() != x4.data() + first ||
            indices.size() != size || masks.size() != size)
            throw std::runtime_error("borrowed forwarding slice does not reference the receiver");
        for (std::size_t i = 0; i < size; ++i)
            if (indices[i] != static_cast<std::size_t>(x3[first + i]) || masks[i] != x4[first + i])
                throw std::runtime_error("incorrect borrowed row coordinates or mask");
    }
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
    // Rank zero has enough work from its own staged source to publish before
    // peer traffic. The other ranks still exercise distinct remote/local masks.
    if (source_node == 0 && source_gpu == 0) return 0;
    if (source_node == 0 && source_gpu == gpus - 1) return 1;
    if (source_node == 1 && source_gpu == 1) return 1;
    if (source_node == 1 && source_gpu == 3) return 3;  // GPU 2 receives only staged rows from GPU 1.
    return (source_gpu + (source_node == 0 ? 1 : gpus - 1)) % gpus;
}
uint8_t payload_value(int epoch, int gpu, int token) {
    return static_cast<uint8_t>(epoch*40 + gpu*3 + token + 1);
}

void wait_for_staged_descriptor(const Rank& rank, int epoch, int source) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    for (;;) {
        const auto state = rank.proxy->router_computation_scheduler_state();
        if (state.iteration_initialized && state.iteration == static_cast<uint64_t>(epoch) && state.published_rows) {
            const auto& pub = rank.proxy->router_notification_publication_buffers();
            const auto* table = static_cast<const int32_t*>(pub.device_table.ptr);
            if (*static_cast<const int32_t*>(pub.device_ready_rows.ptr) <= 0)
                throw std::runtime_error("missing device gather prefix");
            for (std::size_t row = 0; row < state.published_rows; ++row) {
                const auto* ranges = table + row * pub.table_width + 2;
                for (int slot = 0; slot < nodes*gpus; ++slot)
                    if (ranges[2*slot+1] - ranges[2*slot] != (slot == source ? 1 : 0))
                        throw std::runtime_error("early gather descriptor selected the wrong staged source");
            }
            if (!std::all_of(rank.inputs[source], rank.inputs[source] + row_bytes, [&](uint8_t byte) {
                    return byte == payload_value(epoch, source, 1); }))
                throw std::runtime_error("gather prefix preceded the staged payload copy");
            return;
        }
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("timed out waiting for staged descriptor from GPU " + std::to_string(source));
        std::this_thread::yield();
    }
}

void run_benchmark_iteration(Rank& rank, int gpu, int epoch, bool split_transport, bool prebound,
                             std::atomic<int>& finished_consumers, const std::function<void()>& after_install) {
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
    std::array<int32_t, nodes + 1> node_offsets{0, static_cast<int32_t>(rows[0].size()), tokens};
    auto& offsets = rank.offsets;
    auto& indices = rank.indices;
    std::array<RdmaProxyBenchmarkSource, nodes*gpus> sources;
    for (int s = 0; s < nodes*gpus; ++s) {
        offsets[s][0] = 0;
        indices[s].clear();
        const auto count = destination_gpu(epoch, s % gpus, s / gpus) == gpu ? rows[s/gpus].size() : 0;
        for (int expert = 0; expert < experts_per_gpu; ++expert) {
            for (std::size_t i = 0; i < count; ++i) indices[s].push_back(static_cast<int32_t>(i));
            offsets[s][expert+1] = static_cast<int32_t>(indices[s].size());
        }
        sources[s] = {offsets[s].data(), indices[s].empty() ? nullptr : indices[s].data(), indices[s].size()};
    }
    RdmaProxyBenchmarkRouting data{sizeof(data), nodes*gpus, sources.data(), remote_rows.data(),
        remote_masks.data(), remote_rows.size(), reinterpret_cast<uintptr_t>(r.data()),
        reinterpret_cast<uintptr_t>(node_offsets.data()), reinterpret_cast<uintptr_t>(x3.data()),
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
    const bool direct_on_all_gpus = prebound && epoch == 2;
    const bool borrowed_source = borrow_source(epoch, gpu) || direct_on_all_gpus;
    // Keep the original call-scoped contract on epoch zero. The next two
    // generations use the same producer storage with an explicit retirement.
    if (epoch == 0) {
        rank.proxy->install_benchmark_routing(epoch, extended.current);
        expect_error([&] { rank.proxy->retire_borrowed_benchmark_routing(epoch); });
    } else {
        if (borrowed_source || epoch == 2 || gpu % 2 == 0) {
            // The new installer consumes only the outgoing host snapshot. The
            // device fields are deliberately unusable, including in mock mode.
            extended.current.device_r = extended.current.device_node_offsets = 0;
            extended.current.device_x3 = extended.current.device_x4 = 0;
            rank.source_offsets = node_offsets;
            std::copy(x3.begin(), x3.end(), rank.source_x3.begin());
            std::copy(x4.begin(), x4.end(), rank.source_x4.begin());
            std::copy(r.begin(), r.end(), rank.source_r.begin());
            RdmaProxyBenchmarkHostRoute host{sizeof(host), nodes + 1, x3.size(), r.size(),
                rank.source_offsets.data(), rank.source_x3.data(), rank.source_x4.data(), rank.source_r.data()};
            auto install = [&](const RdmaProxyBenchmarkHostRoute& source) {
                if (borrowed_source)
                    rank.proxy->install_borrowed_benchmark_route_views(epoch, extended.current, source);
                else
                    rank.proxy->install_borrowed_benchmark_routing(epoch, extended.current, &source);
            };
            auto reject = [&](const RdmaProxyBenchmarkHostRoute& bad) {
                expect_error([&] { install(bad); });
            };
            auto bad = host;
            --bad.struct_size;
            reject(bad);
            bad = host;
            --bad.node_offset_count;
            reject(bad);
            bad = host;
            --bad.packed_count;
            reject(bad);
            bad = host;
            --bad.assignment_count;
            reject(bad);
            bad = host;
            bad.r = nullptr;
            reject(bad);
            const int32_t saved = rank.source_x3[0];
            rank.source_x3[0] = tokens;
            reject(host);
            rank.source_x3[0] = saved;
            if (gpu == 0 || direct_on_all_gpus) {
                // Simulate the exchange delivering live indices directly into the
                // allocator's device slots. Poison tails to expose stale rows.
                for (int s = 0; s < nodes*gpus; ++s) {
                    std::fill_n(rank.device_indices[s], tokens * experts_per_gpu, -7);
                    std::copy(indices[s].begin(), indices[s].end(), rank.device_indices[s]);
                }
                auto addresses = rank.index_addresses();
                RdmaProxyBenchmarkDeviceIndices views{sizeof(views), nodes*gpus, tokens*experts_per_gpu, addresses.data()};
                const auto direct_install = [&](const RdmaProxyBenchmarkDeviceIndices& views) {
                    rank.proxy->install_borrowed_benchmark_route_views(epoch, extended.current, host, &views,
                                                                      split_transport);
                };
                auto invalid = views;
                --invalid.struct_size;
                expect_error([&] { direct_install(invalid); });
                invalid = views;
                --invalid.num_sources;
                expect_error([&] { direct_install(invalid); });
                invalid = views;
                --invalid.capacity_per_source;
                expect_error([&] { direct_install(invalid); });
                const auto first = addresses[0];
                addresses[0] = addresses[1];
                expect_error([&] { direct_install(views); });
                addresses[0] = first;
                if (split_transport && !remote_masks.empty()) {
                    const auto mask = remote_masks.front();
                    remote_masks.front() = 0;
                    expect_error([&] { direct_install(views); });
                    remote_masks.front() = mask;
                }
                // A canary in one live device prefix distinguishes direct consumption
                // from a redundant host copy. Restore it before normal consumers run.
                const auto live = std::find_if(indices.begin(), indices.end(), [](const auto& list) {
                    return !list.empty();
                }) - indices.begin();
                rank.device_indices.at(live)[0] = -11;
                const bool automatic = split_transport && gpu != 5 && (epoch == 2 || prebound);
                if (automatic && !prebound) rank.bind_scatter(sizeof(rank.scatter_slab));
                direct_install(views);
                const auto& pub = rank.proxy->router_notification_publication_buffers();
                if (automatic) {
                    if (rank.proxy->native_local_publication_state(epoch) != RDMA_PROXY_LOCAL_REQUESTED)
                        throw std::runtime_error("native install did not request local publication");
                    expect_error([&] { rank.bind_scatter(sizeof(rank.scatter_slab)); });
                } else {
                    expect_error([&] { rank.proxy->prepare_local_publication(epoch); });
                    if (rank.proxy->prepare_early_up(epoch, 1) != split_transport)
                        throw std::runtime_error("incorrect early up eligibility");
                }
                if (split_transport && !automatic) {
                    if (*static_cast<const int32_t*>(pub.device_ready_rows.ptr) != 0 ||
                        rank.proxy->router_computation_scheduler_state().iteration_initialized)
                        throw std::runtime_error("early up must reset the device prefix before scheduler setup");
                    expect_error([&] { rank.proxy->prepare_early_up(epoch, 1); });
                    expect_error([&] { rank.proxy->cancel_early_up(epoch - 1); });
                }
                if (rank.device_indices[live][0] != -11)
                    throw std::runtime_error("direct A_idx installation overwrote a live device index");
                rank.device_indices[live][0] = indices[live][0];
            } else {
                install(host);
            }
            if (!borrowed_source) rank.poison_source_route();
        } else {
            rank.proxy->install_borrowed_benchmark_routing(epoch, extended.current);
        }
        expect_error([&] { rank.proxy->retire_borrowed_benchmark_routing(epoch); });
        expect_error([&] { rank.proxy->install_borrowed_benchmark_routing(epoch, data); });
    }
    if (after_install) after_install();
    // Even the direct-device installer only borrows remote_x3/x4 for the
    // call. Neither transport setup nor return maps may read these later.
    std::fill(remote_rows.begin(), remote_rows.end(), -1);
    std::fill(remote_masks.begin(), remote_masks.end(), 0);
    expect_error([&] { rank.proxy->install_benchmark_routing(epoch, data); });
    rank.proxy->prepare_routing(epoch);
    const auto expected_r = r;
    // Retain epoch one's descriptor for the attempted unretired reinstall below.
    // These arrays are distinct from the leased outgoing source storage above.
    if (epoch != 1) {
        std::fill(std::begin(node_offsets), std::end(node_offsets), -1);
        std::fill(x3.begin(), x3.end(), -1);
        std::fill(x4.begin(), x4.end(), 0);
        std::fill(r.begin(), r.end(), -1);
    }
    const auto common_indices = std::max(rows[0].size(), rows[1].size()) * experts_per_gpu;
    for (int s = 0; s < nodes*gpus; ++s) {
        const auto& installed = rank.proxy->router_expert_metadata_for_source(s / gpus, s % gpus);
        const bool borrowed = epoch > 0;
        const bool shares_offsets = installed.offsets_view().data() == sources[s].offsets;
        const bool shares_indices = installed.indices_view().data() == sources[s].indices;
        if (shares_offsets != borrowed || (!indices[s].empty() && shares_indices != borrowed))
            throw std::runtime_error("incorrect ownership of installed expert records");
        const auto* device = rank.device_indices[s];
        if (!device || !std::equal(indices[s].begin(), indices[s].end(), device) ||
            !std::all_of(device + indices[s].size(), device + common_indices, [](int32_t n) { return n == 0; }))
            throw std::runtime_error("incorrect valid indices or stale padding in device A_idx");
        if (gpu == 0 && epoch && !std::all_of(device + common_indices, device + tokens*experts_per_gpu,
                                               [](int32_t n) { return n == -7; }))
            throw std::runtime_error("direct A_idx padding wrote beyond the logical extent");
        if (borrowed) {
            const auto decoded = deserialize_router_expert_metadata(
                serialize_router_expert_metadata(installed), tokens, tokens * experts_per_gpu);
            if (!std::equal(offsets[s].begin(), offsets[s].end(),
                            decoded.expert_offsets.begin(), decoded.expert_offsets.end()) ||
                decoded.expert_token_indices != indices[s])
                throw std::runtime_error("borrowed expert records changed on the wire");
        }
    }
    if (epoch == 0) rank.poison_expert_records();
    uint64_t count{}, num_tokens{}, top_k{};
    auto query = [&](int32_t* ids, float* weights, uint64_t capacity) {
        rank.proxy->copy_original_router_assignments(ids, weights, capacity, count, num_tokens, top_k);
    };
    query(nullptr, nullptr, 0);
    if (count != expected_r.size() || num_tokens != tokens || top_k != experts_per_gpu)
        throw std::runtime_error("incorrect source routing assignment dimensions");
    std::vector<int32_t> actual_r(count, -1);
    std::vector<float> weights(count, -1);
    expect_error([&] { query(actual_r.data(), weights.data(), count - 1); });
    if (count != expected_r.size() || weights.front() != -1 || actual_r.front() != -1)
        throw std::runtime_error("undersized source assignment query wrote partial results");
    query(actual_r.data(), weights.data(), count);
    if (actual_r != expected_r || weights != std::vector<float>(r.size(), 1))
        throw std::runtime_error("incorrect source routing snapshot");
    for (int n = 0; n < nodes; ++n) {
        std::array<uint64_t, tokens> outgoing{};
        uint64_t row_count{}, total_tokens{};
        rank.proxy->copy_outgoing_token_indices(n, outgoing.data(), outgoing.size(), row_count, total_tokens);
        if (row_count != rows[n].size() || total_tokens != tokens ||
            !std::equal(rows[n].begin(), rows[n].end(), outgoing.begin()) ||
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
    if (!epoch) {
        rank.proxy->configure_return_maps(domain_nodes.data(), nodes, tokens,
            reinterpret_cast<uintptr_t>(rank.packed_maps.data()), rank.packed_maps.size(), true);
    }
    std::array<uint64_t, nodes> mask_counts{}, token_counts{};
    if (split_transport && gpu == 0 && epoch == 2) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!rank.proxy->router_computation_scheduler_state().published_rows) {
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("local publication waited for return maps or full preparation");
            std::this_thread::yield();
        }
        rank.proxy->prepare_local_publication(epoch);
    }
    rank.proxy->start_return_maps(epoch, mask_counts.data(), token_counts.data(), nodes);
    for (int domain = 0; domain < nodes; ++domain) {
        if (mask_counts[domain] != domain_counts[domain] || token_counts[domain] != rows[domain_nodes[domain]].size())
            throw std::runtime_error("incorrect async return map extents");
    }
    if (split_transport && gpu == 0 && epoch != 0) {
        const auto& publication = rank.proxy->router_notification_publication_buffers();
        const auto before = publication.ready_rows_publication_count;
        rank.proxy->prepare_iteration_step(epoch);
        if (publication.ready_rows_publication_count != before)
            throw std::runtime_error("full preparation republished the current prefix");
        const auto state = rank.proxy->router_computation_scheduler_state();
        rank.proxy->prepare_iteration_step(epoch);
        if (rank.proxy->router_computation_scheduler_state().published_rows != state.published_rows ||
            (epoch == 1 && state.published_rows != 0))
            throw std::runtime_error("repeated preparation changed the local prefix");
    }
    if (split_transport) {
        expect_error([&] { rank.proxy->complete_iteration_step(epoch); });
        rank.proxy->run_iteration_payload(epoch);
        expect_error([&] { rank.proxy->run_iteration_step(epoch); });
        expect_error([&] { rank.proxy->complete_iteration_step(epoch + 1); });
        expect_error([&] { rank.proxy->prepare_routing(epoch + 1); });
        expect_error([&] { rank.proxy->finish_run(); });
        // Hold one simulated GPU consumer until every other rank can enter done.
        if (gpu == gpus - 1) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (finished_consumers.load() < gpus - 1) {
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("payload completion waited for a peer's final consumer");
                std::this_thread::yield();
            }
        }
        ++finished_consumers;
        rank.proxy->complete_iteration_step(epoch);
        if (finished_consumers.load() != gpus)
            throw std::runtime_error("iteration_done released before every consumer finished");
        expect_error([&] { rank.proxy->complete_iteration_step(epoch); });
    } else {
        rank.proxy->run_iteration_step(epoch);
    }
    if (split_transport && gpu == 0 && epoch == 2) {
        rank.proxy->wait_cpu_scatter_ready(epoch, 1, 5000);
        for (int source = 0; source < nodes*gpus; ++source) {
            const auto& indices = rank.indices[source];
            const auto expected = indices.empty() ? 0 : indices.back() + 1;
            if (rank.scatter_counts[source] != expected)
                throw std::runtime_error("incorrect scatter count from automatic publication");
        }
        if (rank.scatter_slab.front() != reinterpret_cast<int64_t>(rank.inputs[0]))
            throw std::runtime_error("first local scatter row has the wrong output address");
    }
    rank.proxy->wait_return_maps(epoch, 1, 5000);
    if (!std::equal(rank.combine_rows.begin(), rank.combine_rows.end(), rank.packed_maps.begin()) ||
        !std::equal(rank.inter_rows.begin(), rank.inter_rows.end(), rank.packed_maps.begin() + rank.combine_rows.size()))
        throw std::runtime_error("async return map disagrees with original-token/compact-row lookup");
    if (epoch == 1) {
        // A completed payload alone cannot authorize either install path to
        // replace expert records while the producer still has an active lease.
        expect_error([&] { rank.proxy->install_benchmark_routing(epoch + 1, data); });
        expect_error([&] { rank.proxy->install_borrowed_benchmark_routing(epoch + 1, data); });
    }
}

void check_final_borrowed_install(Rank& rank, uint64_t epoch, std::vector<int32_t> rows,
                                  bool swap_device_sources, const char* expected_error) {
    std::array<RdmaProxyBenchmarkSource, nodes * gpus> sources{};
    for (std::size_t i = 0; i < sources.size(); ++i) {
        rank.offsets[i].fill(0);
        rank.indices[i].clear();
        sources[i] = {rank.offsets[i].data(), nullptr, 0};
    }
    // Keep all arrays alive through shutdown, including after a partial installation.
    rank.indices[0] = std::move(rows);
    rank.offsets[0].back() = static_cast<int32_t>(rank.indices[0].size());
    sources[0] = {rank.offsets[0].data(), rank.indices[0].data(), rank.indices[0].size()};
    rank.source_offsets.fill(0);
    rank.source_r.fill(0);
    RdmaProxyBenchmarkRouting routing{sizeof(routing), nodes * gpus, sources.data()};
    RdmaProxyBenchmarkHostRoute source{sizeof(source), nodes + 1, 0, rank.source_r.size(),
        rank.source_offsets.data(), nullptr, nullptr, rank.source_r.data()};
    auto addresses = rank.index_addresses();
    std::swap(addresses[0], addresses[1]);
    RdmaProxyBenchmarkDeviceIndices views{sizeof(views), nodes*gpus, tokens*experts_per_gpu, addresses.data()};
    if (swap_device_sources)
        for (auto* slot : rank.device_indices) std::fill_n(slot, tokens*experts_per_gpu, -9);
    auto install = [&] {
        rank.proxy->install_borrowed_benchmark_route_views(epoch, routing, source, swap_device_sources ? &views : nullptr);
    };
    if (!expected_error) {
        // Observe that trusted mode skips this scan. Never dispatch the deliberately bad list.
        install();
        return;
    }
    try {
        install();
        throw std::logic_error("partial borrowed installation unexpectedly succeeded");
    } catch (const std::runtime_error& error) {
        if (std::string(error.what()).find(expected_error) == std::string::npos) throw;
    }
    if (swap_device_sources)
        for (const auto* slot : rank.device_indices)
            if (!std::all_of(slot, slot + tokens*experts_per_gpu, [](int32_t n) { return n == -9; }))
                throw std::runtime_error("failed installation corrupted a device slot");
    uint64_t count{}, num_tokens{}, top_k{};
    expect_error([&] {
        rank.proxy->copy_original_router_assignments(nullptr, nullptr, 0, count, num_tokens, top_k);
    });
    expect_error([&] { rank.proxy->retire_borrowed_benchmark_routing(epoch); });
    expect_error([&] { rank.proxy->install_borrowed_benchmark_route_views(epoch, routing, source); });
}

void check_empty_device_indices(Rank& rank, uint64_t epoch, bool split_transport) {
    std::array<RdmaProxyBenchmarkSource, nodes*gpus> sources{};
    for (int s = 0; s < nodes*gpus; ++s) {
        rank.offsets[s].fill(0);
        sources[s] = {rank.offsets[s].data(), nullptr, 0};
        std::fill_n(rank.device_indices[s], tokens*experts_per_gpu, -5);
    }
    rank.source_offsets.fill(0);
    rank.source_r.fill(0);
    RdmaProxyBenchmarkRouting routing{sizeof(routing), nodes*gpus, sources.data()};
    RdmaProxyBenchmarkHostRoute host{sizeof(host), nodes + 1, 0, rank.source_r.size(),
        rank.source_offsets.data(), nullptr, nullptr, rank.source_r.data()};
    const auto addresses = rank.index_addresses();
    RdmaProxyBenchmarkDeviceIndices views{sizeof(views), nodes*gpus, tokens*experts_per_gpu, addresses.data()};
    if (split_transport) rank.proxy->bind_native_local_publication(1, nullptr, 0);
    rank.proxy->install_borrowed_benchmark_route_views(epoch, routing, host, &views, split_transport);
    if (rank.proxy->native_local_publication_state(epoch) != RDMA_PROXY_LOCAL_INACTIVE)
        throw std::runtime_error("empty route requested publication");
    for (const auto* slot : rank.device_indices)
        if (slot[0] != 0 || !std::all_of(slot + 1, slot + tokens*experts_per_gpu, [](int32_t n) { return n == -5; }))
            throw std::runtime_error("empty source lists must expose exactly one zero index");
    if (rank.proxy->prepare_early_up(epoch, 1)) throw std::runtime_error("empty up route was armed");
    if (split_transport)
        rank.proxy->prepare_iteration_step(epoch);  // Shutdown before peer transport starts.
}

void check_cancelled_publication(Rank& rank, int gpu, bool publish_local, bool grow = false) {
    constexpr uint64_t epoch = 3;
    std::array<RdmaProxyBenchmarkSource, nodes*gpus> sources{};
    for (int s = 0; s < nodes*gpus; ++s) {
        rank.offsets[s].fill(0);
        rank.indices[s].clear();
        if (s == gpu || s == gpus + gpu) {
            // Distinct experts, so this rank can publish its first complete
            // batch from its local row while the remote descriptor stays pending.
            std::fill(rank.offsets[s].begin() + (s == gpu ? 1 : 2), rank.offsets[s].end(), 1);
            rank.indices[s].push_back(0);
        }
        sources[s] = {rank.offsets[s].data(), rank.indices[s].data(), rank.indices[s].size()};
    }
    rank.source_offsets = {0, 1, 1};
    rank.source_x3[0] = 0;
    rank.source_x4[0] = 1;
    rank.source_r.fill(0);
    int32_t remote_row = 0;
    uint8_t remote_mask = 1;
    RdmaProxyBenchmarkRouting route{sizeof(route), nodes*gpus, sources.data(), &remote_row, &remote_mask, 1};
    RdmaProxyBenchmarkHostRoute host{sizeof(host), nodes+1, 1, rank.source_r.size(),
        rank.source_offsets.data(), rank.source_x3.data(), rank.source_x4.data(), rank.source_r.data()};
    auto slots = rank.index_addresses();
    RdmaProxyBenchmarkDeviceIndices views{sizeof(views), nodes*gpus, tokens*experts_per_gpu, slots.data()};
    if (publish_local) {
        if (grow) rank.bind_scatter(16);  // One row. The route needs two.
        else rank.proxy->bind_native_local_publication(1, nullptr, 0);
    }
    rank.proxy->install_borrowed_benchmark_route_views(epoch, route, host, &views, true);
    if (grow) {
        if (rank.proxy->native_local_publication_state(epoch) != RDMA_PROXY_LOCAL_NEEDS_STORAGE ||
            rank.proxy->router_computation_scheduler_state().iteration_initialized)
            throw std::runtime_error("undersized scatter slab did not defer publication");
        rank.bind_scatter(32);
    }
    const auto& pub = rank.proxy->router_notification_publication_buffers();
    auto* table = static_cast<int32_t*>(pub.device_table.ptr);
    const auto words = pub.table_width * pub.table_rows;
    if (!publish_local) {
        if (!rank.proxy->prepare_early_up(epoch, 1)) throw std::runtime_error("expected early up launch");
        std::fill_n(table, words, -9);
    } else {
        if (rank.proxy->native_local_publication_state(epoch) != RDMA_PROXY_LOCAL_REQUESTED)
            throw std::runtime_error("missing native local request before cancellation");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!rank.proxy->router_computation_scheduler_state().published_rows) {
            if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("missing local publication");
            std::this_thread::yield();
        }
    }
    const auto ready = *static_cast<const int32_t*>(pub.device_ready_rows.ptr);
    if (ready < 0 || static_cast<std::size_t>(ready) >= pub.table_rows || (ready != 0) != publish_local)
        throw std::runtime_error("incorrect prefix before cancellation");
    const std::vector<int32_t> prefix(table, table + ready*pub.table_width);
    rank.proxy->cancel_early_up(epoch);
    rank.proxy->cancel_early_up(epoch);
    expect_error([&] { rank.proxy->native_local_publication_state(epoch); });
    expect_error([&] { rank.proxy->prepare_iteration_step(epoch); });
    expect_error([&] { rank.proxy->run_iteration_step(epoch); });
    if (static_cast<std::size_t>(*static_cast<const int32_t*>(pub.device_ready_rows.ptr)) != pub.table_rows ||
        !std::equal(prefix.begin(), prefix.end(), table) ||
        !std::all_of(table + prefix.size(), table + words, [](auto v) { return v == 0; }))
        throw std::runtime_error("failed to release only the unpublished suffix");
}

void run(bool prepared, bool validate, bool split_transport, bool atomic_ring = false, bool batch_sync = false) {
    ProxyConfig config;
    config.num_nodes = nodes;
    config.num_gpus_per_node = gpus;
    config.num_tokens = tokens;
    config.token_dimension = row_bytes / dtype_size(config.dtype);
    config.num_iterations = 4;
    config.mock_mode = true;
    config.fill_test_data = config.validate_data = false;
    config.router_routing_enabled = config.router_dynamic_routing_enabled = true;
    config.router_benchmark_metadata_enabled = config.router_local_input_staging_enabled = true;
    config.router_benchmark_metadata_validation_enabled = validate;
    config.router_local_input_early_ready_enabled = split_transport;
    if (prepared && split_transport) config.nvlink_forward_notification_scheduling_mode = "complete_round";
    config.router_num_experts = nodes*gpus*experts_per_gpu;
    config.router_top_k = experts_per_gpu;
    config.nvlink_forwarding_enabled = config.nvlink_forward_synchronize_batches = true;
    config.nvlink_forward_completion_notifications_enabled = true;
    config.nvlink_forward_preparation_enabled = prepared;
    config.nvlink_forward_local_staging_prepared_enabled = prepared;
    config.nvlink_forward_submit_epilogue_enabled = prepared;
    config.nvlink_forward_prepared_atomic_ring_enabled = atomic_ring;
    config.nvlink_forward_local_batch_sync_enabled = batch_sync;
    config.local_forwarding_rdma_overlap_enabled = config.sequential_peer_transfers = prepared && split_transport;
    config.nvlink_forward_prepared_batch_chunks = 1;
    config.nvlink_forward_threshold_tokens = config.nvlink_forward_chunk_tokens = 1;
    config.local_iteration_sync_enabled = true;
    config.completion_timeout_ms = 10000;
    config.peers.push_back(PeerAddress{1, "mock-peer", 18515});
    const auto id = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    config.local_iteration_sync_run_id = "bench_handoff_" + id;
    config.nvlink_forward_exchange_dir = "/tmp/meon_bench_handoff_" + id;
    check_borrowed_source_views(config);
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
            if (r.kind == DeviceBufferKind::kRouterAIdx)
                rank.device_indices.at(source) = reinterpret_cast<int32_t*>(ptr);
            return ptr;
        });
    }
    on_all_gpus([&](int gpu) { ranks[gpu].proxy->initialize(); });
    for (int epoch = 0; epoch < 3; ++epoch) {
        std::atomic<int> finished_consumers{0};
        const bool hold_remote = prepared && split_transport && epoch == 2;
        std::atomic<int> installed{0};
        int peer_fd = -1;
        std::unique_ptr<IterationControlChannel> peer;
        std::future<void> done_reply;
        if (hold_remote) {
            peer = BenchmarkRoutingTestAccess::attach_peer(*ranks[0].proxy, peer_fd);
            std::memset(ranks[2].inputs[1], 0xee, row_bytes);
            // Every mock payload WR copies directly into the registered receive slice.
            for (auto& rank : ranks) std::memset(rank.rdma_receive, 0xee, tokens * row_bytes);
        }
        on_all_gpus([&](int gpu) {
            std::function<void()> after_install;
            if (hold_remote && gpu != gpus - 1) after_install = [&] { ++installed; };
            if (split_transport && epoch == 2 && gpu == gpus - 1) {
                wait_for_staged_descriptor(ranks[0], epoch, 0);
                if (hold_remote) {
                    pollfd waiting{peer_fd, POLLIN, 0};
                    if (::poll(&waiting, 1, 5000) != 1 || !(waiting.revents & POLLIN))
                        throw std::runtime_error("rank zero did not reach the same-index start exchange");
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                    while (installed.load() < gpus - 1) {
                        if (std::chrono::steady_clock::now() >= deadline)
                            throw std::runtime_error("local route installation waited for another GPU");
                        std::this_thread::yield();
                    }
                    const auto state = ranks[2].proxy->router_computation_scheduler_state();
                    if (state.iteration != static_cast<uint64_t>(epoch) || state.published_rows ||
                        !std::all_of(ranks[2].inputs[1], ranks[2].inputs[1] + row_bytes,
                                     [](uint8_t byte) { return byte == 0xee; }))
                        throw std::runtime_error("peer copy or gather prefix preceded the last local receiver");
                    after_install = [&] {
                        // GPU 7 has installed its native binding but has not entered payload preparation.
                        // GPU 1 must now reach GPU 2 while GPU 0's remote start reply is still withheld.
                        wait_for_staged_descriptor(ranks[2], epoch, 1);
                        for (const auto& rank : ranks)
                            if (!std::all_of(rank.rdma_receive, rank.rdma_receive + tokens * row_bytes,
                                             [](uint8_t byte) { return byte == 0xee; }))
                                throw std::runtime_error("payload RDMA write preceded the remote start reply");
                        const auto exchange = [&](const std::string& phase) {
                            const auto suffix = " gpu=0 iteration=" + std::to_string(epoch);
                            if (peer->exchange(phase + " rank=1" + suffix, 10000) != phase + " rank=0" + suffix)
                                throw std::runtime_error("wrong same-index iteration marker");
                        };
                        exchange("iteration_start");
                        done_reply = std::async(std::launch::async, [exchange] { exchange("iteration_done"); });
                    };
                }
            }
            run_benchmark_iteration(ranks[gpu], gpu, epoch, split_transport, prepared && split_transport,
                                    finished_consumers, after_install);
        });
        if (done_reply.valid()) done_reply.get();
        if (!epoch) {
            const auto state = ranks[0].proxy->native_local_publication_state(epoch);
            if (state != (split_transport ? RDMA_PROXY_LOCAL_UNBOUND : RDMA_PROXY_LOCAL_INACTIVE))
                throw std::runtime_error("incorrect publication eligibility at startup");
            if (prepared && split_transport) {
                ranks[0].bind_scatter(sizeof(ranks[0].scatter_slab));
                // GPU 5 exercises the explicit prepare call and cancellation without a binding.
                for (int gpu = 1; gpu < gpus; ++gpu)
                    if (gpu != 5) ranks[gpu].proxy->bind_native_local_publication(1, nullptr, 0);
            }
        }
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
                if (table.indices_view().size() != row[node]*experts_per_gpu ||
                    state.received_token_frontier != row[node])
                    throw std::runtime_error("incorrect benchmark source table or completion frontier");
            }
        }
        if (epoch > 0) {
            for (int gpu = 0; gpu < gpus; ++gpu) {
                auto& rank = ranks[gpu];
                // The worker may shut down directly after the last iteration.
                if (epoch == 2 && gpu == 0) continue;
                expect_error([&] { rank.proxy->retire_borrowed_benchmark_routing(epoch - 1); });
                rank.proxy->retire_borrowed_benchmark_routing(epoch);
                expect_error([&] { rank.proxy->cancel_early_up(epoch); });
                expect_error([&] { rank.proxy->retire_borrowed_benchmark_routing(epoch); });
                expect_error([&] { (void)rank.proxy->router_expert_metadata_for_source(0, 0); });
                rank.poison_expert_records();
                rank.poison_source_route();
                uint64_t count{}, num_tokens{}, top_k{};
                const auto query_size = [&] {
                    rank.proxy->copy_original_router_assignments(nullptr, nullptr, 0, count, num_tokens, top_k);
                };
                if (borrow_source(epoch, gpu) || (prepared && split_transport && epoch == 2)) {
                    expect_error(query_size);
                    expect_error([&] { (void)rank.proxy->outgoing_token_indices(0); });
                } else {
                    query_size();
                    if (count != rank.source_r.size()) throw std::runtime_error("retirement lost an owned route");
                }
            }
        }
    }
    if (batch_sync) {
        // Release a staging round from ordinary preparation on eight empty destinations,
        // then shut down with the remote phase still closed.
        ranks[0].proxy->retire_borrowed_benchmark_routing(2);
        for (auto& rank : ranks) check_empty_device_indices(rank, 3, true);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        for (const auto& rank : ranks) {
            while (BenchmarkRoutingTestAccess::completed_local_iterations(*rank.proxy) != 4) {
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("empty staging phase waited for payload startup");
                std::this_thread::yield();
            }
        }
    } else {
        check_final_borrowed_install(ranks[1], 3, {tokens}, false, validate ? "invalid token index" : nullptr);
        check_final_borrowed_install(ranks[2], 3, {0}, true, "device index slot disagrees");
        check_empty_device_indices(ranks[3], 3, split_transport);
        if (split_transport) {
            check_cancelled_publication(ranks[5], 5, false);
            check_cancelled_publication(ranks[6], 6, true, prepared);
        }
        check_final_borrowed_install(ranks[4], 3, {1, 0}, false, validate ? "not monotonic" : nullptr);
    }
    on_all_gpus([&](int gpu) { ranks[gpu].proxy->shutdown(); });
    expect_error([&] { (void)ranks[0].proxy->router_expert_metadata_for_source(0, 0); });
    for (int gpu : {0, 1}) {
        ranks[gpu].poison_source_route();
        uint64_t count{}, num_tokens{}, top_k{};
        expect_error([&] {
            ranks[gpu].proxy->copy_original_router_assignments(nullptr, nullptr, 0, count, num_tokens, top_k);
        });
    }
    ranks[0].poison_expert_records();
}
}

int main() {
    Logger::instance().set_level(LogLevel::kError);
    for (bool prepared : {false, true}) run(prepared, true, false);
    run(false, true, true);
    run(true, false, true);
    run(true, false, true, true);
    run(true, false, true, true, true);
}
