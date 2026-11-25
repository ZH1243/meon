#include "logging.hpp"
#include "proxy.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <future>
#include <stdexcept>

using namespace rdma_proxy;

namespace {

// Two independent proxy owners exchange real host-backed mock NVLink copies.
// Join both ranks between checks so the next round cannot overwrite a reader.
template<class F>
void on_both_gpus(F function) {
    auto first = std::async(std::launch::async, [&] { function(0); });
    auto second = std::async(std::launch::async, [&] { function(1); });
    first.get();
    second.get();
}

struct Rank {
    std::vector<std::unique_ptr<uint8_t[]>> allocations;
    uint8_t* input{};
    std::array<uint8_t*, 4> receives{};
    uint8_t* rdma_receive{};
    std::unique_ptr<Proxy> proxy;
};

void run(bool prepared, bool with_remote) {
    ProxyConfig config;
    config.mock_mode = true;
    config.num_gpus_per_node = 2;
    config.num_tokens = with_remote ? 1 : 43;
    config.token_dimension = 8;
    config.num_iterations = 4;
    config.tokens_per_chunk = with_remote ? 1 : 2;
    config.router_num_experts = 4;
    config.router_top_k = 1;
    config.fill_test_data = false;
    config.validate_data = false;
    config.router_routing_enabled = true;
    config.router_dynamic_routing_enabled = true;
    config.router_local_input_staging_enabled = true;
    config.nvlink_forwarding_enabled = true;
    config.nvlink_forward_preparation_enabled = prepared;
    config.nvlink_forward_local_staging_prepared_enabled = prepared;
    config.nvlink_forward_submit_epilogue_enabled = prepared;
    config.nvlink_forward_prepared_atomic_ring_enabled = prepared;
    config.nvlink_forward_prepared_batch_chunks = 2;
    config.nvlink_forward_chunk_tokens = 1;
    config.nvlink_forward_threshold_tokens = with_remote ? 1 : 4;
    config.nvlink_forward_synchronize_batches = true;
    config.nvlink_forward_completion_notifications_enabled = true;
    config.local_iteration_sync_enabled = true;
    config.nvlink_forward_local_batch_sync_enabled = true;
    config.sequential_peer_transfers = true;
    if (with_remote) {
        // Mock verbs echoes outgoing WRs. For this seed, the independently
        // generated incoming and outgoing lengths match at each same-index
        // proxy. They cycle through (0,0), (0,1), (1,0), (0,0) on the two
        // GPUs, exercising a newly appearing chunk ID and a shrinking plan.
        config.router_seed = 128653;
        config.num_nodes = 2;
        config.nvlink_forward_prepared_batch_chunks = 1;
        config.peers.push_back(PeerAddress{1, "mock-peer", 18515});
    }
    config.completion_timeout_ms = 5000;
    const auto nonce = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    config.local_iteration_sync_run_id = "dynamic_forward_" + nonce;
    config.nvlink_forward_exchange_dir = "/tmp/meon_dynamic_forward_" + nonce;
    std::array<Rank, 2> ranks;
    for (int gpu = 0; gpu < 2; ++gpu) {
        config.cuda_device_id = config.local_gpu_index = gpu;
        validate_config(config);
        auto& rank = ranks[gpu];
        rank.proxy = std::make_unique<Proxy>(config);
        rank.proxy->set_external_device_buffer_allocator([&rank, gpu](const DeviceBufferAllocationRequest& request) {
            if (request.kind == DeviceBufferKind::kNvlinkReceive &&
                request.source_node_rank == 1 && request.source_gpu_index == gpu) {
                rank.receives[2 + gpu] = rank.rdma_receive;
                return rank.rdma_receive;
            }
            rank.allocations.push_back(std::make_unique<uint8_t[]>(request.bytes));
            auto* ptr = rank.allocations.back().get();
            if (request.kind == DeviceBufferKind::kRdmaSend) rank.input = ptr;
            if (request.kind == DeviceBufferKind::kRdmaReceive) rank.rdma_receive = ptr;
            if (request.kind == DeviceBufferKind::kNvlinkReceive)
                rank.receives.at(request.source_node_rank * 2 + request.source_gpu_index) = ptr;
            return ptr;
        });
    }

    on_both_gpus([&](int gpu) { ranks[gpu].proxy->initialize(); });
    const auto row_bytes = config.token_dimension * dtype_size(config.dtype);
    std::array<std::vector<int32_t>, 2> previous;
    for (uint64_t epoch = 0; epoch < config.num_iterations; ++epoch) {
        on_both_gpus([&](int gpu) {
            auto& rank = ranks[gpu];
            for (std::size_t byte = 0; byte < config.num_tokens * row_bytes; ++byte)
                rank.input[byte] = (byte + epoch * 29 + gpu * 67) % 251;
            rank.proxy->prepare_routing(epoch);
            // An explicit preparation followed by an implicit preparation
            // must neither regenerate routes nor consume the epoch twice.
            rank.proxy->prepare_routing(epoch);
            bool rejected = false;
            try { rank.proxy->prepare_routing(epoch + 1); }
            catch (const std::runtime_error&) { rejected = true; }
            if (!rejected) throw std::runtime_error("replaced routing before payload completion");
            rank.proxy->run_iteration_step(epoch);
        });
        for (int source = 0; source < 2; ++source) {
            const auto& proxy = *ranks[source].proxy;
            const auto& experts = proxy.original_router_experts();
            if (!with_remote && experts == previous[source]) throw std::runtime_error("routing did not change");
            previous[source] = experts;
            const auto& x3 = proxy.outgoing_token_indices(0);
            for (int destination = 0; destination < 2; ++destination) {
                const auto& dst = ranks[destination];
                std::size_t rows = 0;
                for (auto token : x3) {
                    const int experts_per_gpu = config.router_num_experts / (2 * config.num_nodes);
                    if (destination != source && experts.at(token) / experts_per_gpu != destination) continue;
                    if (std::memcmp(dst.receives[source] + rows * row_bytes,
                                    ranks[source].input + token * row_bytes, row_bytes))
                        throw std::runtime_error("stale or incorrect staged/forwarded row");
                    ++rows;
                }
                const auto state = dst.proxy->router_expert_token_head_state_for_source(0, source);
                if (state.received_token_frontier != rows || (rows && state.iteration != epoch))
                    throw std::runtime_error("incorrect dynamic completion frontier");
                if (with_remote) {
                    const auto masks = proxy.intra_node_source_masks(1);
                    const std::size_t remote_rows = (epoch == 1 && source == 1) || (epoch == 2 && source == 0);
                    if (masks.size() != remote_rows)
                        throw std::runtime_error("unexpected remote fixture routing");
                    const bool delivered = remote_rows && (source == destination || (masks[0] & (1 << destination)));
                    const auto head = dst.proxy->router_expert_token_head_state_for_source(1, source);
                    if (head.received_token_frontier != static_cast<std::size_t>(delivered) ||
                        (delivered && std::memcmp(dst.receives[2 + source], ranks[source].input, row_bytes)))
                        throw std::runtime_error("incorrect dynamic remote completion or payload");
                }
            }
        }
    }
    on_both_gpus([&](int gpu) { ranks[gpu].proxy->shutdown(); });
}

} // namespace

int main() {
    Logger::instance().set_level(LogLevel::kError);
    for (bool prepared : {true, false})
        for (bool with_remote : {false, true}) run(prepared, with_remote);
}
