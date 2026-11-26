#include "rdma_connection.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <future>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

using namespace rdma_proxy;
using namespace std::chrono_literals;

void test_proxy_iteration_barrier(uint16_t port);

namespace {
constexpr uint64_t timeout_ms = 2000;

void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}

template<class F>
void require_error(F operation) {
    try { operation(); }
    catch (const std::runtime_error&) { return; }
    throw std::runtime_error("expected the channel operation to fail");
}

// Keep the socket bound while handing its port to the raw-peer tests.
int bind_loopback(uint16_t& port) {
    const int fd = ::socket(AF_INET6, SOCK_STREAM, 0);
    require(fd >= 0, "failed to create loopback socket");
    sockaddr_in6 address{};
    address.sin6_family = AF_INET6;
    address.sin6_addr = in6addr_loopback;
    socklen_t size = sizeof(address);
    require(::bind(fd, reinterpret_cast<sockaddr*>(&address), size) == 0, "loopback bind failed");
    require(::getsockname(fd, reinterpret_cast<sockaddr*>(&address), &size) == 0, "getsockname failed");
    port = ntohs(address.sin6_port);
    return fd;
}

ProxyConfig rank_config(int rank, uint16_t port) {
    ProxyConfig config;
    config.node_rank = rank;
    config.mock_mode = false;
    config.listen_port = port;
    return config;
}

void check_paired_channels() {
    uint16_t port;
    ::close(bind_loopback(port));
    ConnectionManager node0(rank_config(0, port)), node1(rank_config(1, port));
    auto opening = std::async(std::launch::async, [&] {
        return node1.open_iteration_control({0, "::1", port}, timeout_ms);
    });
    auto client = node0.open_iteration_control({1, "::1", port}, timeout_ms);
    auto server = opening.get();
    for (unsigned epoch = 0; epoch < 32; ++epoch) {
        for (bool start : {true, false}) {
            const std::string phase = start ? "iteration_start" : "iteration_done";
            const auto suffix = " gpu=0 iteration=" + std::to_string(epoch);
            // Each side sometimes arrives early. No exchange can complete until
            // the matching peer sends the marker for this epoch and phase.
            const bool client_first = (epoch % 2 == 0) == start;
            auto& early = client_first ? client : server;
            auto& late = client_first ? server : client;
            const auto first = phase + (client_first ? " rank=0" : " rank=1") + suffix;
            const auto second = phase + (client_first ? " rank=1" : " rank=0") + suffix;
            std::promise<void> entering;
            auto entered = entering.get_future();
            auto pending = std::async(std::launch::async, [&] {
                entering.set_value();
                return early->exchange(first, timeout_ms);
            });
            entered.get();
            require(pending.wait_for(1ms) == std::future_status::timeout, "passed before the peer arrived");
            require(late->exchange(second, timeout_ms) == first, "late peer received a stale marker");
            require(pending.get() == second, "early peer received a stale marker");
        }
    }
    // Keep the peer open but silent. A timeout invalidates this stream:
    // replaying a marker on it could consume a delayed response.
    require_error([&] { (void)server->exchange("iteration_done", 30); });
    require_error([&] { (void)server->exchange("iteration_done", timeout_ms); });
    require_error([&] { (void)client->exchange("iteration_done", timeout_ms); });
}

// These test records fit in a single byte of the protocol's big-endian
// 64-bit length prefix. Read the bytes independently of the native encoder.
std::string wire_record(const std::string& text) {
    require(text.size() < 256, "test record too large");
    return std::string(7, '\0') + static_cast<char>(text.size()) + text;
}

void check_legacy_wire_and_close(bool malformed_response) {
    uint16_t port;
    const int listener = bind_loopback(port);
    require(::listen(listener, 1) == 0, "listen failed");
    ConnectionManager manager(rank_config(0, port));
    auto channel = manager.open_iteration_control({1, "::1", port}, timeout_ms);
    const int peer = ::accept(listener, nullptr, nullptr);
    require(peer >= 0, "accept failed");
    ::close(listener); // All further exchanges must use the original socket.
    const timeval receive_timeout{2, 0};
    require(::setsockopt(peer, SOL_SOCKET, SO_RCVTIMEO, &receive_timeout, sizeof(receive_timeout)) == 0,
            "failed to set receive timeout");

    for (const std::string phase : {"peer_ready", "iteration_start", "iteration_done"}) {
        const auto sent = phase + " rank=0 gpu=0";
        const auto reply = phase + " rank=1 gpu=0";
        auto pending = std::async(std::launch::async, [&] { return channel->exchange(sent, timeout_ms); });
        const auto expected = wire_record(sent);
        std::string received(expected.size(), '\0');
        require(::recv(peer, received.data(), received.size(), MSG_WAITALL) ==
                    static_cast<ssize_t>(received.size()) && received == expected, "changed control wire format");
        const auto frame = wire_record(reply);
        require(::send(peer, frame.data(), 3, 0) == 3, "failed to send split header");
        require(::send(peer, frame.data() + 3, frame.size() - 3, 0) ==
                    static_cast<ssize_t>(frame.size() - 3), "failed to send remainder");
        require(pending.get() == reply, "persistent socket returned the wrong message");
    }
    if (malformed_response) {
        // Force rejection before allocation or any read of the advertised body.
        const std::string bad_length(8, static_cast<char>(0xff));
        require(::send(peer, bad_length.data(), bad_length.size(), 0) == 8, "failed to send malformed header");
        require_error([&] { (void)channel->exchange("iteration_start", timeout_ms); });
        require_error([&] { (void)channel->exchange("iteration_start", timeout_ms); });
    }
    channel.reset();
    // Drain a possible final request and confirm that destruction closes the
    // connected socket even while the manager remains alive.
    char bytes[256];
    ssize_t n;
    do { n = ::recv(peer, bytes, sizeof(bytes), 0); } while (n > 0);
    require(n == 0 || errno == ECONNRESET, "missing close on the persistent connection");
    ::close(peer);
}
}  // namespace

int main() {
    check_paired_channels();
    check_legacy_wire_and_close(false);
    check_legacy_wire_and_close(true);
    uint16_t port;
    ::close(bind_loopback(port));
    test_proxy_iteration_barrier(port);
}
