#include "proxy.hpp"

#include <future>
#include <stdexcept>
#include <string>

namespace rdma_proxy {

// Exercise peer setup and the actual start/done entry points without GPU memory or QPs.
struct IterationBarrierTestAccess {
    static void ready(Proxy& proxy) {
        const auto& address = proxy.config_.peers.at(0);
        auto& peer = proxy.peers_.emplace_back();
        peer.peer_rank = address.node_rank;
        peer.remote_gpu_index = proxy.config_.local_gpu_index;
        proxy.synchronize_peer_ready(address, peer);
    }
    static IterationControlChannel* channel(Proxy& proxy) {
        return proxy.peers_.at(0).iteration_control.get();
    }
    static void phase(Proxy& proxy, bool start, uint64_t iteration) {
        if (start) proxy.synchronize_iteration_start(iteration);
        else proxy.synchronize_iteration(iteration);
    }
};

} // namespace rdma_proxy

namespace {
using namespace rdma_proxy;
using Access = IterationBarrierTestAccess;
constexpr uint64_t timeout_ms = 2000;

ProxyConfig endpoint(int rank, uint16_t port) {
    ProxyConfig config;
    config.node_rank = rank;
    config.num_nodes = 2;
    config.num_gpus_per_node = 1;
    config.mock_mode = false;
    config.listen_port = port;
    config.completion_timeout_ms = timeout_ms;
    config.peers.push_back({1 - rank, "::1", port});
    return config;
}

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void negotiate(Proxy& client, Proxy& server) {
    auto ready = std::async(std::launch::async, [&] { Access::ready(server); });
    Access::ready(client);
    ready.get();
    require(Access::channel(client) && Access::channel(server), "peers did not negotiate a persistent channel");
}

std::string marker(bool start, int rank, uint64_t iteration) {
    return std::string(start ? "iteration_start" : "iteration_done") +
           " rank=" + std::to_string(rank) + " gpu=0 iteration=" + std::to_string(iteration);
}

void check_repeated_barriers(uint16_t port) {
    auto a = endpoint(0, port), b = endpoint(1, port);
    Proxy client(a), server(b);
    negotiate(client, server);
    ConnectionManager left(a), right(b);
    for (uint64_t iteration = 0; iteration < 3; ++iteration) {
        for (bool start : {true, false}) {
            auto waiting = std::async(std::launch::async, [&] { Access::phase(server, start, iteration); });
            Access::phase(client, start, iteration);
            waiting.get();
        }
        // Optional RDMA return checks still perform a one-shot exchange on this port.
        auto checksum = std::async(std::launch::async, [&] {
            return right.exchange_control_message(b.peers.front(), "rdma_return_check right", timeout_ms);
        });
        require(left.exchange_control_message(a.peers.front(), "rdma_return_check left", timeout_ms) ==
                "rdma_return_check right", "server control port was unavailable after the barrier");
        require(checksum.get() == "rdma_return_check left", "one-shot exchange was consumed by a barrier");
    }
}

void check_legacy_peer(int updated_rank, uint16_t port) {
    const int old_rank = 1 - updated_rank;
    Proxy updated(endpoint(updated_rank, port));
    const auto old = endpoint(old_rank, port);
    ConnectionManager legacy(old);
    auto legacy_peer = std::async(std::launch::async, [&] {
        const auto exchange = [&](const std::string& message) {
            return legacy.exchange_control_message(old.peers.front(), message, timeout_ms);
        };
        const auto ready = exchange("peer_ready rank=" + std::to_string(old_rank) +
                                    " gpu=0 peer_rank=" + std::to_string(updated_rank) + " remote_gpu=0");
        require(ready.find(" iteration_control_v1") != std::string::npos, "updated peer omitted its capability");
        for (uint64_t iteration = 0; iteration < 2; ++iteration)
            for (bool start : {true, false})
                require(exchange(marker(start, old_rank, iteration)) == marker(start, updated_rank, iteration),
                        "iteration barrier was incompatible with the legacy peer");
    });
    Access::ready(updated);
    require(!Access::channel(updated), "upgraded a legacy peer");
    for (uint64_t iteration = 0; iteration < 2; ++iteration)
        for (bool start : {true, false}) Access::phase(updated, start, iteration);
    legacy_peer.get();
}

void check_invalid_marker(uint16_t port, const std::string& wrong) {
    Proxy client(endpoint(0, port)), server(endpoint(1, port));
    negotiate(client, server);
    auto response = std::async(std::launch::async, [&] {
        return Access::channel(server)->exchange(wrong, timeout_ms);
    });
    bool rejected = false;
    try { Access::phase(client, true, 1); }
    catch (const std::runtime_error&) { rejected = true; }
    require(response.get() == marker(true, 0, 1), "peer did not receive the expected barrier request");
    require(rejected, "mismatched barrier marker allowed the proxy to continue");

    // A malformed marker is terminal. Even a valid next response cannot release this epoch.
    auto retry = std::async(std::launch::async, [&] {
        try { Access::phase(server, true, 1); }
        catch (const std::runtime_error&) { return true; }
        return false;
    });
    rejected = false;
    try { Access::phase(client, true, 1); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected && retry.get(), "a failed stream was reused after a protocol mismatch");
}
} // namespace

void test_proxy_iteration_barrier(uint16_t port) {
    check_repeated_barriers(port);
    check_legacy_peer(0, port);
    check_legacy_peer(1, port);
    check_invalid_marker(port, "iteration_done rank=1 gpu=0 iteration=1");
    check_invalid_marker(port, "iteration_start rank=1 gpu=0 iteration=10");
    check_invalid_marker(port, "iteration_start rank=0 gpu=0 iteration=1");
    check_invalid_marker(port, "iteration_start rank=1 gpu=1 iteration=1");
}
