#include "rdma_proxy_c_api.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

struct ExternalAllocations {
    std::vector<void*> pointers;
    std::vector<RdmaProxyDeviceBufferRequest> requests;
};

uintptr_t allocate_external_buffer(
    void* context,
    const RdmaProxyDeviceBufferRequest* request) {
    if (!context || !request || request->struct_size != sizeof(*request) ||
        request->version != 1 || request->bytes == 0) {
        return 0;
    }
    auto* allocations = static_cast<ExternalAllocations*>(context);
    void* pointer = std::malloc(static_cast<std::size_t>(request->bytes));
    if (!pointer) return 0;
    std::memset(pointer, 0, static_cast<std::size_t>(request->bytes));
    allocations->pointers.push_back(pointer);
    allocations->requests.push_back(*request);
    return reinterpret_cast<uintptr_t>(pointer);
}

// Exercise the same public sequence used by the Python worker, including
// prepare_routing followed by implicit preparation inside prepare_iteration.
void check_dynamic_rank(int gpu, const std::string& run_id, const std::string& profile) {
    const std::string rank = "--local_gpu_index=" + std::to_string(gpu);
    const std::string run = "--local_iteration_sync_run_id=" + run_id;
    const std::string dir = "--nvlink_forward_exchange_dir=/tmp/" + run_id;
    const char* args[] = {"proxy", "--config", profile.c_str(), "--mock_mode=true", "--cpu_affinity=none",
        "--num_gpus_per_node=2", "--num_tokens=32", "--token_dimension=8", "--num_experts=4",
        "--top_k=2", "--num_iterations=3", "--completion_timeout_ms=5000",
        "--router_computation_kernel_enabled=false", "--router_dynamic_routing_enabled=true",
        "--log_level=error", rank.c_str(), run.c_str(), dir.c_str()};
    auto require = [](bool success) {
        if (!success) throw std::runtime_error(std::string("dynamic routing ABI: ") + rdma_proxy_last_error());
    };
    std::unique_ptr<RdmaProxyHandle, decltype(&rdma_proxy_destroy)> owner(
        rdma_proxy_create(sizeof(args) / sizeof(args[0]), args), rdma_proxy_destroy);
    auto* handle = owner.get();
    int32_t enabled = 0;
    require(handle && rdma_proxy_dynamic_routing_enabled(handle, &enabled) == 0 && enabled == 1);
    require(rdma_proxy_initialize(handle) == 0);
    const int32_t address_columns = gpu + 1;  // Exercise both supported layouts.
    require(rdma_proxy_reserve_threaded_cpu_scatter(handle, 0) != 0 && *rdma_proxy_last_error());
    require(rdma_proxy_reserve_threaded_cpu_scatter(handle, 3) != 0 && *rdma_proxy_last_error());
    require(rdma_proxy_reserve_threaded_cpu_scatter(handle, address_columns) == 0 && !*rdma_proxy_last_error());
    uint64_t indices = 0, table_rows = 0, count = 0, tokens = 0, top_k = 0;
    require(rdma_proxy_get_routing_extents(handle, &indices, &table_rows) != 0);
    constexpr uint64_t capacity = 32, world = 2, stride = capacity + 2;
    const int32_t source_node = 0;
    std::array<int32_t, world * stride> combine_rows;
    combine_rows.fill(-2);
    uint64_t domain_count = UINT64_MAX;
    auto build_rows = [&](RdmaProxyHandle* h) {
        return rdma_proxy_build_intra_node_combine_rows(h, &source_node, 1, capacity,
            combine_rows.data(), combine_rows.size(), &domain_count);
    };
    std::array<int32_t, stride> inter_rows;
    auto build_inter = [&](RdmaProxyHandle* h) {
        return rdma_proxy_build_inter_node_combine_rows(h, &source_node, 1, capacity,
            inter_rows.data(), inter_rows.size());
    };
    require(build_rows(nullptr) != 0 && build_rows(handle) != 0);
    require(build_inter(nullptr) != 0 && build_inter(handle) != 0);
    std::vector<int32_t> previous;
    for (uint64_t i = 0; i < 3; ++i) {
        require(rdma_proxy_prepare_routing(handle, i + 1) != 0);
        require(rdma_proxy_prepare_routing(handle, i) == 0 && rdma_proxy_prepare_routing(handle, i) == 0);
        require(rdma_proxy_reserve_threaded_cpu_scatter(handle, address_columns) != 0 && *rdma_proxy_last_error());
        require(rdma_proxy_get_routing_extents(handle, &indices, &table_rows) == 0 && indices && table_rows);
        std::vector<int32_t> experts(2 * capacity);
        std::vector<float> weights(experts.size());
        require(rdma_proxy_get_local_router_assignments(handle, experts.data(), weights.data(), experts.size(),
                &count, &tokens, &top_k) == 0 && count == experts.size() && top_k == 2 && experts != previous);
        previous = std::move(experts);
        if (i == 0) {
            auto invalid_inter = [&](const int32_t* nodes, uint64_t n, uint64_t num_tokens,
                                     int32_t* out, uint64_t elements) {
                require(rdma_proxy_build_inter_node_combine_rows(handle, nodes, n, num_tokens,
                    out, elements) != 0 && *rdma_proxy_last_error());
            };
            invalid_inter(nullptr, 1, capacity, inter_rows.data(), stride);
            invalid_inter(&source_node, 1, capacity, nullptr, stride);
            invalid_inter(&source_node, 0, capacity, inter_rows.data(), stride);
            invalid_inter(&source_node, UINT64_MAX, capacity, inter_rows.data(), stride);
            invalid_inter(&source_node, 1, UINT64_MAX, inter_rows.data(), stride);
            invalid_inter(&source_node, 1, capacity, inter_rows.data(), stride - 1);
            require(rdma_proxy_build_intra_node_combine_rows(handle, nullptr, 1, capacity,
                    combine_rows.data(), combine_rows.size(), &domain_count) != 0);
            require(rdma_proxy_build_intra_node_combine_rows(handle, &source_node, 1, capacity,
                    nullptr, combine_rows.size(), &domain_count) != 0);
            require(rdma_proxy_build_intra_node_combine_rows(handle, &source_node, 1, capacity,
                    combine_rows.data(), combine_rows.size(), nullptr) != 0);
            require(rdma_proxy_build_intra_node_combine_rows(handle, &source_node, 1, capacity,
                    combine_rows.data(), combine_rows.size() - 1, &domain_count) != 0);
            require(rdma_proxy_build_intra_node_combine_rows(handle, &source_node, 1, UINT64_MAX,
                    combine_rows.data(), combine_rows.size(), &domain_count) != 0);
        }
        require(build_inter(handle) == 0 && !*rdma_proxy_last_error());
        std::array<uint64_t, capacity> outgoing;
        uint64_t outgoing_rows = 0, original_tokens = 0;
        require(rdma_proxy_get_outgoing_token_indices(handle, source_node, outgoing.data(), capacity,
                    &outgoing_rows, &original_tokens) == 0 && original_tokens == capacity);
        std::array<int32_t, stride> expected_inter;
        expected_inter.fill(-1);
        for (uint64_t slot = 0; slot < outgoing_rows; ++slot)
            expected_inter.at(outgoing[slot]) = static_cast<int32_t>(slot);
        expected_inter[capacity] = static_cast<int32_t>(outgoing_rows);
        expected_inter[capacity + 1] = 0;
        require(inter_rows == expected_inter);
        require(build_rows(handle) == 0 && !*rdma_proxy_last_error());
        std::array<uint8_t, capacity> masks{};
        uint64_t mask_count = 0;
        require(rdma_proxy_get_intra_node_source_masks(handle, source_node, masks.data(),
                masks.size(), &mask_count) == 0 && domain_count == mask_count && mask_count == capacity);
        require(std::find(masks.begin(), masks.end(), 3) != masks.end());  // One row contributes to both GPUs.
        for (unsigned peer = 0; peer < world; ++peer) {
            int32_t packed = 0;
            for (uint64_t row = 0; row < capacity; ++row) {
                const auto expected = (masks[row] & (1U << peer)) ? packed++ : -1;
                require(combine_rows[peer * stride + row] == expected);
            }
            require(combine_rows[peer * stride + capacity] == packed &&
                    combine_rows[peer * stride + capacity + 1] == static_cast<int32_t>(mask_count));
        }
        require(rdma_proxy_prepare_iteration(handle, i) == 0);
        require(build_rows(handle) != 0 && build_inter(handle) != 0);
        require(rdma_proxy_run_iteration(handle, i) == 0);
        require(build_rows(handle) != 0 && build_inter(handle) != 0);  // Completed generation.
    }
    require(rdma_proxy_finish(handle) == 0 && rdma_proxy_shutdown(handle) == 0);
    require(build_rows(handle) != 0 && build_inter(handle) != 0);
}

}  // namespace

int main(int argc, char** argv) {
    if (rdma_proxy_abi_version() != 1) {
        std::cerr << "unexpected RDMA proxy C ABI version\n";
        return 1;
    }
    if (argc != 2) {
        std::cerr << "usage: test_c_api <config>\n";
        return 1;
    }

    const char* proxy_argv[] = {
        "rdma_cpu_proxy",
        "--config",
        argv[1],
        "--cpu_affinity=none",
        "--num_tokens=32",
        "--token_dimension=8",
        "--tokens_per_chunk=8",
        "--num_qps_per_peer=2",
        "--num_iterations=1",
        "--completion_timeout_ms=5000",
        "--fill_test_data=false",
        "--validate_data=false",
        "--log_level=error",
    };
    constexpr int proxy_argc = static_cast<int>(sizeof(proxy_argv) / sizeof(proxy_argv[0]));
    if (rdma_proxy_run_argv(proxy_argc, proxy_argv) != 0) {
        std::cerr << "shared-library proxy run failed: " << rdma_proxy_last_error() << '\n';
        return 1;
    }

    RdmaProxyHandle* handle = rdma_proxy_create(proxy_argc, proxy_argv);
    if (!handle) {
        std::cerr << "per-iteration proxy creation failed: " << rdma_proxy_last_error() << '\n';
        return 1;
    }
    int32_t benchmark_enabled = -1;
    RdmaProxyBenchmarkConfig benchmark_config{};
    benchmark_config.struct_size = sizeof(benchmark_config);
    if (rdma_proxy_benchmark_metadata_enabled(handle, &benchmark_enabled) || benchmark_enabled ||
        rdma_proxy_benchmark_metadata_enabled(nullptr, &benchmark_enabled) == 0 ||
        rdma_proxy_get_benchmark_routing_config(handle, &benchmark_config) == 0 ||
        rdma_proxy_install_benchmark_routing(handle, 0, nullptr) == 0 ||
        rdma_proxy_install_borrowed_benchmark_routing(handle, 0, nullptr) == 0 ||
        rdma_proxy_retire_borrowed_benchmark_routing(nullptr, 0) == 0 ||
        rdma_proxy_retire_borrowed_benchmark_routing(handle, 0) == 0) {
        std::cerr << "invalid benchmark metadata query, install or retirement accepted\n";
        return 1;
    }
    {
        const auto profile = (std::filesystem::path(argv[1]).parent_path() / "node0_config.json").string();
        const char* args[] = {"proxy", "--config", profile.c_str(), "--num_gpus_per_node=8",
            "--num_experts=128", "--router_routing_enabled=true", "--router_dynamic_routing_enabled=true",
            "--router_benchmark_metadata_enabled=true", "--router_local_input_staging_enabled=true",
            "--nvlink_forwarding_enabled=true", "--nvlink_forward_completion_notifications_enabled=true",
            "--nvlink_forward_synchronize_batches=true", "--fill_test_data=false", "--rdma_device_name=test_nic",
            "--num_tokens=10240", "--cuda_device_id=3", "--rdma_port=2", "--gid_index=5", "--cpu_affinity=none"};
        std::unique_ptr<RdmaProxyHandle, decltype(&rdma_proxy_destroy)> provider(
            rdma_proxy_create(sizeof(args) / sizeof(*args), args), rdma_proxy_destroy);
        if (!provider || rdma_proxy_benchmark_metadata_enabled(provider.get(), &benchmark_enabled) ||
            benchmark_enabled != 1 || rdma_proxy_get_benchmark_routing_config(provider.get(), &benchmark_config) ||
            benchmark_config.num_tokens != 10240 || benchmark_config.router_seed != 1234 ||
            benchmark_config.cuda_device_id != 3 || benchmark_config.num_gpus_per_node != 8 ||
            benchmark_config.num_nodes != 2 || benchmark_config.rdma_port != 2 || benchmark_config.gid_index != 5 ||
            std::string(benchmark_config.rdma_device_name) != "test_nic") {
            std::cerr << "incorrect effective benchmark configuration: " << rdma_proxy_last_error() << '\n';
            return 1;
        }
        benchmark_config.struct_size = sizeof(benchmark_config) - 1;
        if (rdma_proxy_get_benchmark_routing_config(provider.get(), &benchmark_config) == 0) return 1;
    }
    if (rdma_proxy_reserve_threaded_cpu_scatter(nullptr, 1) == 0 ||
        rdma_proxy_reserve_threaded_cpu_scatter(handle, 1) == 0 ||
        rdma_proxy_configure_cpu_scatter_threaded(nullptr, 16) == 0 ||
        rdma_proxy_configure_cpu_scatter_threaded(handle, 16) == 0) {
        std::cerr << "threaded scatter accepted a null or uninitialized proxy\n";
        rdma_proxy_destroy(handle);
        return 1;
    }
    ExternalAllocations external_allocations;
    if (rdma_proxy_set_device_buffer_allocator(
            handle, allocate_external_buffer, &external_allocations) != 0) {
        std::cerr << "failed to install external allocator: "
                  << rdma_proxy_last_error() << '\n';
        rdma_proxy_destroy(handle);
        return 1;
    }
    if (rdma_proxy_initialize(handle) != 0) {
        std::cerr << "per-iteration proxy initialization failed: " << rdma_proxy_last_error() << '\n';
        rdma_proxy_destroy(handle);
        return 1;
    }
    if (rdma_proxy_reserve_threaded_cpu_scatter(handle, 1) == 0) {
        std::cerr << "threaded scatter reservation accepted a static route\n";
        rdma_proxy_destroy(handle);
        return 1;
    }
    if (rdma_proxy_configure_cpu_scatter_threaded(handle, 16) == 0) {
        std::cerr << "threaded scatter accepted missing scatter configuration\n";
        rdma_proxy_destroy(handle);
        return 1;
    }
    int32_t return_prefix = -1;
    uintptr_t return_host = 0, return_device = 0;
    if (rdma_proxy_get_return_ready_publication(handle, &return_host, &return_device) == 0 ||
        rdma_proxy_get_return_ready_publication(nullptr, &return_host, &return_device) == 0 ||
        rdma_proxy_get_return_ready_publication(handle, nullptr, &return_device) == 0 ||
        rdma_proxy_get_return_ready_publication(handle, &return_host, nullptr) == 0) {
        std::cerr << "return publication accepted disabled feature or invalid arguments\n";
        rdma_proxy_destroy(handle);
        return 1;
    }
    if (rdma_proxy_get_return_ready_prefix(handle, 0, &return_prefix) == 0 ||
        rdma_proxy_get_return_ready_prefix(nullptr, 0, &return_prefix) == 0 ||
        rdma_proxy_get_return_ready_prefix(handle, 0, nullptr) == 0) {
        std::cerr << "return prefix accepted disabled feature or invalid arguments\n";
        rdma_proxy_destroy(handle);
        return 1;
    }
    std::size_t send_request_count = 0;
    std::size_t receive_request_count = 0;
    bool invalid_request = false;
    for (const auto& request : external_allocations.requests) {
        send_request_count +=
            request.kind == RDMA_PROXY_DEVICE_BUFFER_RDMA_SEND ? 1 : 0;
        receive_request_count +=
            request.kind == RDMA_PROXY_DEVICE_BUFFER_RDMA_RECEIVE ? 1 : 0;
        invalid_request = invalid_request || request.dimension_count != 2 ||
            request.dimensions[0] != 32 || request.dimensions[1] != 8 ||
            request.bytes != 32 * 8 * 2;
    }
    if (external_allocations.requests.size() != 6 ||
        send_request_count != 3 || receive_request_count != 3 ||
        invalid_request) {
        std::cerr << "external allocator received unexpected buffer requests\n";
        rdma_proxy_shutdown(handle);
        rdma_proxy_destroy(handle);
        for (void* pointer : external_allocations.pointers) std::free(pointer);
        return 1;
    }
    if (rdma_proxy_set_device_buffer_allocator(
            handle, allocate_external_buffer, &external_allocations) == 0) {
        std::cerr << "external allocator was accepted after initialization\n";
        rdma_proxy_shutdown(handle);
        rdma_proxy_destroy(handle);
        for (void* pointer : external_allocations.pointers) std::free(pointer);
        return 1;
    }
    uint64_t num_iterations = 0;
    if (rdma_proxy_get_num_iterations(handle, &num_iterations) != 0 || num_iterations != 1) {
        std::cerr << "per-iteration proxy reported unexpected iteration count: "
                  << rdma_proxy_last_error() << '\n';
        rdma_proxy_shutdown(handle);
        rdma_proxy_destroy(handle);
        return 1;
    }
    if (rdma_proxy_prepare_iteration(handle, 0) != 0) {
        std::cerr << "per-iteration proxy prepare failed: "
                  << rdma_proxy_last_error() << '\n';
        rdma_proxy_shutdown(handle);
        rdma_proxy_destroy(handle);
        return 1;
    }
    int iteration_status = -1;
    std::string iteration_error;
    std::thread iteration_thread([&] {
        iteration_status = rdma_proxy_run_iteration(handle, 0);
        if (iteration_status != 0) iteration_error = rdma_proxy_last_error();
    });
    iteration_thread.join();
    if (iteration_status != 0) {
        std::cerr << "per-iteration proxy run failed: " << iteration_error << '\n';
        rdma_proxy_shutdown(handle);
        rdma_proxy_destroy(handle);
        return 1;
    }
    if (rdma_proxy_finish(handle) != 0) {
        std::cerr << "per-iteration proxy finish failed: " << rdma_proxy_last_error() << '\n';
        rdma_proxy_shutdown(handle);
        rdma_proxy_destroy(handle);
        return 1;
    }
    if (rdma_proxy_shutdown(handle) != 0) {
        std::cerr << "per-iteration proxy shutdown failed: " << rdma_proxy_last_error() << '\n';
        rdma_proxy_destroy(handle);
        return 1;
    }
    rdma_proxy_destroy(handle);
    // The proxy borrows callback allocations and must leave freeing them to
    // the embedding runtime.
    for (void* pointer : external_allocations.pointers) std::free(pointer);

    // Two local mock proxies are needed for CUDA-IPC metadata/notification setup.
    const auto nonce = std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    const std::string exchange = "--nvlink_forward_exchange_dir=/tmp/test_c_api_" + nonce;
    const std::string run_id = "--local_iteration_sync_run_id=test_c_api_" + nonce;
    bool counts_ok[2] = {false, false};
    std::thread query_threads[2];
    for (int gpu = 0; gpu < 2; ++gpu) {
        query_threads[gpu] = std::thread([&, gpu] {
            std::vector<const char*> router_argv(proxy_argv, proxy_argv + proxy_argc);
            const std::string gpu_arg = "--local_gpu_index=" + std::to_string(gpu);
            const std::string device_arg = "--cuda_device_id=" + std::to_string(gpu);
            for (const char* option : {
                     "--num_gpus_per_node=2", "--router_routing_enabled=true",
                     "--router_local_input_staging_enabled=true",
                     "--router_weighted_contributions_enabled=true",
                     "--nvlink_forwarding_enabled=true", "--num_experts=64",
                     "--nvlink_forward_synchronize_batches=true",
                     "--nvlink_forward_completion_notifications_enabled=true",
                     "--nvlink_forward_threshold_chunks=1", "--nvlink_forward_min_threshold_chunks=1",
                     "--nvlink_forward_max_threshold_chunks=2",
                     "--nvlink_forward_synchronize_iteration=false",
                     exchange.c_str(), run_id.c_str(), gpu_arg.c_str(), device_arg.c_str()}) {
                router_argv.push_back(option);
            }
            auto* router = rdma_proxy_create(static_cast<int>(router_argv.size()), router_argv.data());
            uint64_t n = 0;
            if (!router || rdma_proxy_get_computation_expert_counts(router, nullptr, 0, &n) == 0 ||
                rdma_proxy_initialize(router) != 0) {
                std::cerr << "router query setup failed: " << rdma_proxy_last_error() << '\n';
                rdma_proxy_destroy(router);
                return;
            }
            uint64_t assignment_count = 0, source_tokens = 0, top_k = 0;
            bool assignments_ok = rdma_proxy_get_local_router_assignments(
                router, nullptr, nullptr, 0, &assignment_count, &source_tokens, &top_k) == 0 &&
                source_tokens == 32 && assignment_count == source_tokens * top_k && top_k > 1;
            std::vector<int32_t> original_experts(assignment_count);
            std::vector<float> original_weights(assignment_count);
            assignments_ok = assignments_ok && rdma_proxy_get_local_router_assignments(
                router, original_experts.data(), original_weights.data(), assignment_count,
                &assignment_count, &source_tokens, &top_k) == 0;
            for (uint64_t token = 0; token < source_tokens; ++token) {
                double sum_weights = 0;
                for (uint64_t k = 0; k < top_k; ++k) {
                    const auto i = token * top_k + k;
                    assignments_ok = assignments_ok && original_experts[i] >= 0 && original_experts[i] < 64 &&
                        original_weights[i] > 0 && original_weights[i] <= 1;
                    sum_weights += original_weights[i];
                }
                assignments_ok = assignments_ok && std::abs(sum_weights - 1.0) < 1e-6;
            }
            assignments_ok = assignments_ok && rdma_proxy_get_local_router_assignments(
                router, original_experts.data(), original_weights.data(), 1,
                &assignment_count, &source_tokens, &top_k) != 0 &&
                rdma_proxy_get_local_router_assignments(router, nullptr, original_weights.data(), 0,
                &assignment_count, &source_tokens, &top_k) != 0;
            bool ok = assignments_ok && rdma_proxy_get_computation_expert_counts(router, nullptr, 0, &n) == 0 && n == 8;
            std::vector<uint64_t> counts(n);
            uint64_t total = 0;
            ok = ok && rdma_proxy_get_computation_expert_counts(router, counts.data(), n, &n) == 0 &&
                rdma_proxy_get_computation_num_tokens(router, &total) == 0;
            uint64_t sum = 0;
            for (auto count : counts) sum += count;
            counts_ok[gpu] = ok && sum == total && total > 0 &&
                rdma_proxy_get_computation_expert_counts(router, counts.data(), 1, &n) != 0 &&
                rdma_proxy_get_computation_expert_counts(router, nullptr, 1, &n) != 0;
            uint64_t mask_count = 0;
            bool masks_ok = rdma_proxy_get_intra_node_source_masks(
                router, 0, nullptr, 0, &mask_count) == 0 && mask_count > 0 && mask_count <= 32;
            std::vector<uint8_t> masks(mask_count);
            masks_ok = masks_ok && rdma_proxy_get_intra_node_source_masks(
                router, 0, masks.data(), masks.size(), &mask_count) == 0;
            for (auto mask : masks) masks_ok = masks_ok && mask > 0 && mask < 4;
            masks_ok = masks_ok && rdma_proxy_get_intra_node_source_masks(
                router, -1, nullptr, 0, &mask_count) != 0 &&
                rdma_proxy_get_intra_node_source_masks(router, 0, nullptr, 1, &mask_count) != 0 &&
                rdma_proxy_get_intra_node_source_masks(router, 0, masks.data(), 0, &mask_count) != 0;
            counts_ok[gpu] = counts_ok[gpu] && masks_ok;
            uint64_t token_count = 0, original_tokens = 0;
            const int32_t destination = 0;
            std::array<int32_t, 32 + 2> inter_rows;
            bool tokens_ok = rdma_proxy_build_inter_node_combine_rows(router, &destination, 1, 32,
                inter_rows.data(), inter_rows.size()) != 0 && *rdma_proxy_last_error();
            for (int node = 0; node < 4; ++node) {
                tokens_ok = tokens_ok && rdma_proxy_get_outgoing_token_indices(
                    router, node, nullptr, 0, &token_count, &original_tokens) == 0 && original_tokens == 32;
                std::vector<uint64_t> ids(token_count);
                tokens_ok = tokens_ok && rdma_proxy_get_outgoing_token_indices(
                    router, node, ids.data(), ids.size(), &token_count, &original_tokens) == 0;
                bool seen[32]{};
                for (auto id : ids) {
                    if (id >= 32 || seen[id]) { tokens_ok = false; break; }
                    seen[id] = true;
                }
                if (!ids.empty()) tokens_ok = tokens_ok && rdma_proxy_get_outgoing_token_indices(
                    router, node, ids.data(), ids.size() - 1, &token_count, &original_tokens) != 0;
            }
            tokens_ok = tokens_ok && rdma_proxy_get_outgoing_token_indices(
                router, -1, nullptr, 0, &token_count, &original_tokens) != 0 &&
                rdma_proxy_get_outgoing_token_indices(router, 0, nullptr, 1, &token_count, &original_tokens) != 0 &&
                rdma_proxy_get_outgoing_token_indices(router, 0, nullptr, 0, &token_count, nullptr) != 0;
            counts_ok[gpu] = counts_ok[gpu] && tokens_ok;
            uintptr_t address = 0;
            int32_t groups[3]{};
            double times[3]{};
            uint64_t observed = 0;
            bool trace_ok = rdma_proxy_configure_down_ready_trace(router, 3, &address) == 0 &&
                address != 0 && rdma_proxy_prepare_iteration(router, 0) == 0;
            if (trace_ok) {
                // Mock mapped storage; the real notification receiver observes
                // this publication, exercising the ABI and receiver integration.
                __atomic_store_n(reinterpret_cast<int32_t*>(address), 3, __ATOMIC_RELEASE);
                trace_ok = rdma_proxy_read_down_ready_trace(
                    router, 0, groups, times, 3, &observed) == 0 &&
                    observed == 1 && groups[0] == 3 && times[0] >= 0 &&
                    rdma_proxy_read_down_ready_trace(router, 0, groups, times, 0, &observed) != 0 &&
                    rdma_proxy_read_down_ready_trace(router, 1, groups, times, 3, &observed) != 0;
            }
            counts_ok[gpu] = counts_ok[gpu] && trace_ok;
            rdma_proxy_shutdown(router);
            rdma_proxy_destroy(router);
        });
    }
    for (auto& thread : query_threads) thread.join();
    if (!counts_ok[0] || !counts_ok[1]) {
        std::cerr << "per-expert C ABI counts or capacity validation failed\n";
        return 1;
    }

    const auto dynamic_run = "dynamic_routing_abi_" + nonce;
    const auto profile = (std::filesystem::path(argv[1]).parent_path() / "single_node_config.json").string();
    auto rank0 = std::async(std::launch::async, check_dynamic_rank, 0, dynamic_run, profile);
    auto rank1 = std::async(std::launch::async, check_dynamic_rank, 1, dynamic_run, profile);
    rank0.get();
    rank1.get();

    const char* invalid_argv[] = {"rdma_cpu_proxy"};
    if (rdma_proxy_create(1, invalid_argv) != nullptr) {
        std::cerr << "invalid proxy arguments unexpectedly created a handle\n";
        return 1;
    }
    if (std::string(rdma_proxy_last_error()).empty()) {
        std::cerr << "invalid proxy arguments did not report an error\n";
        return 1;
    }
    return 0;
}
