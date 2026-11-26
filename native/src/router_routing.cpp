#include "router_routing.hpp"

#include "logging.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#if RDMA_PROXY_HAVE_CUDA
#include <cuda_runtime.h>
#include "expert_token_idx_standalone.hpp"
#endif

namespace rdma_proxy {
namespace {

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

int gcd(int a, int b) {
    while (b != 0) {
        const int remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

#if RDMA_PROXY_HAVE_CUDA
constexpr std::size_t kKernelChunkTokens = 256;

std::size_t checked_multiply(std::size_t a, std::size_t b, const char* what) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::runtime_error(std::string(what) + " overflows size_t");
    }
    return a * b;
}

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}
#endif

}  // namespace

struct RouterRouting::Workspace {
    explicit Workspace(const ProxyConfig& c)
        : device(c.cuda_device_id), tokens(c.num_tokens), nodes(c.num_nodes),
          gpus(c.num_gpus_per_node), experts(c.router_num_experts), top_k(c.router_top_k) {}
    bool allocated = false;
    int device;
    std::size_t tokens;
    int nodes, gpus, experts, top_k;
    int32_t *r{}, *expert_offsets{}, *expert_indices{}, *x3{}, *node_offsets{};
    uint8_t* x4{};
    int32_t *input_counts{}, *input_prefixes{}, *mask_offsets{}, *reordered_counts{}, *reordered_prefixes{};
    int32_t *host_node_offsets{}, *host_x3{};
    uint8_t* host_x4{};
#if RDMA_PROXY_HAVE_CUDA
    template<class T>
    void allocate(T*& ptr, std::size_t count, const char* name, bool pinned = false) {
        const auto bytes = checked_multiply(count, sizeof(T), name);
        check_cuda(pinned ? cudaMallocHost(reinterpret_cast<void**>(&ptr), bytes)
                          : cudaMalloc(reinterpret_cast<void**>(&ptr), bytes), name);
    }
    ~Workspace() {
        if (!r) return;
        int previous = -1;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        for (auto p : {r, expert_offsets, expert_indices, x3, node_offsets, input_counts,
                       input_prefixes, mask_offsets, reordered_counts, reordered_prefixes})
            if (p) cudaFree(p);
        if (x4) cudaFree(x4);
        if (host_node_offsets) cudaFreeHost(host_node_offsets);
        if (host_x3) cudaFreeHost(host_x3);
        if (host_x4) cudaFreeHost(host_x4);
        if (previous >= 0 && previous != device) cudaSetDevice(previous);
    }
#endif
};

std::shared_ptr<RouterRouting::Workspace> RouterRouting::make_workspace(const ProxyConfig& c) {
    return std::make_shared<Workspace>(c);
}

RouterRouting::RouterRouting(ProxyConfig config, std::shared_ptr<Workspace> workspace)
    : config_(std::move(config)), workspace_(std::move(workspace)) {}

RouterRouting::~RouterRouting() = default;

int RouterRouting::experts_per_gpu() const {
    return config_.router_num_experts / (config_.num_nodes * config_.num_gpus_per_node);
}

void RouterRouting::copy_benchmark_node_slice(
    const int32_t* x3, const uint8_t* x4, std::size_t count,
    std::vector<std::size_t>& indices, std::vector<uint8_t>& masks) const {
    validate_benchmark_node_slice(x3, x4, count);
    if (count) {
        indices.assign(x3, x3 + count);
        masks.assign(x4, x4 + count);
    } else {
        indices.clear();
        masks.clear();
    }
}

void RouterRouting::validate_benchmark_node_slice(
    const int32_t* x3, const uint8_t* x4, std::size_t count) const {
    if (count > config_.num_tokens || (count && (!x3 || !x4)))
        throw std::runtime_error("invalid benchmark x3/x4 slice length or address");
    if (!count) return;
    std::vector<bool> seen(config_.num_tokens, false);
    for (std::size_t i = 0; i < count; ++i) {
        const auto token = x3[i];
        if (token < 0 || static_cast<std::size_t>(token) >= config_.num_tokens ||
            seen[token] || !x4[i] || (x4[i] >> config_.num_gpus_per_node) ||
            (i && (x4[i] > x4[i - 1] || (x4[i] == x4[i - 1] && token <= x3[i - 1]))))
            throw std::runtime_error("invalid benchmark x3 token, x4 mask or row order");
        seen[token] = true;
    }
}

void RouterRouting::require_fresh_benchmark_route() const {
    if (initialized_ || !config_.router_benchmark_metadata_enabled ||
        config_.router_weighted_contributions_enabled)
        throw std::runtime_error("benchmark source snapshot requires fresh unweighted routing");
}

void RouterRouting::initialize_benchmark(uintptr_t device_r, uintptr_t device_node_offsets,
                                         uintptr_t device_x3, uintptr_t device_x4) {
    require_fresh_benchmark_route();
    if (!device_r || !device_node_offsets || !device_x3 || !device_x4 ||
        device_r % alignof(int32_t) || device_node_offsets % alignof(int32_t) ||
        device_x3 % alignof(int32_t))
        throw std::runtime_error("invalid benchmark source buffer address");
#if RDMA_PROXY_HAVE_CUDA
    if (!config_.mock_mode)
        check_cuda(cudaSetDevice(config_.cuda_device_id), "select benchmark source GPU");
#endif
    const auto read = [&](auto& output, uintptr_t address) {
        if (output.empty()) return;
        const auto bytes = output.size() * sizeof(output[0]);
        auto* source = reinterpret_cast<const void*>(address);
        if (config_.mock_mode) {
            std::memcpy(output.data(), source, bytes);
            return;
        }
#if RDMA_PROXY_HAVE_CUDA
        check_cuda(cudaMemcpy(output.data(), source, bytes, cudaMemcpyDeviceToHost),
                   "read benchmark source snapshot");
#else
        throw std::runtime_error("benchmark source snapshot requires CUDA");
#endif
    };
    std::vector<int32_t> offsets(config_.num_nodes + 1);
    read(offsets, device_node_offsets);
    std::vector<int32_t> x3(benchmark_packed_rows(offsets.data()));
    std::vector<uint8_t> x4(x3.size());
    read(x3, device_x3);
    read(x4, device_x4);
    copy_benchmark_nodes(offsets.data(), x3.data(), x4.data());
    original_experts_.resize(config_.num_tokens * config_.router_top_k);
    read(original_experts_, device_r);
    finish_benchmark_route();
}

void RouterRouting::initialize_benchmark_host(const int32_t* r, const int32_t* offsets,
                                              const int32_t* x3, const uint8_t* x4, uint64_t packed_count) {
    validate_benchmark_host(r, offsets, x3, x4, packed_count);
    copy_benchmark_nodes(offsets, x3, x4);
    original_experts_.assign(r, r + config_.num_tokens * config_.router_top_k);
    finish_benchmark_route();
}

void RouterRouting::validate_benchmark_host(const int32_t* r, const int32_t* offsets,
                                           const int32_t* x3, const uint8_t* x4,
                                           uint64_t packed_count) const {
    require_fresh_benchmark_route();
    if (!r || !offsets || (packed_count && (!x3 || !x4)) ||
        reinterpret_cast<uintptr_t>(r) % alignof(int32_t) ||
        reinterpret_cast<uintptr_t>(offsets) % alignof(int32_t) ||
        reinterpret_cast<uintptr_t>(x3) % alignof(int32_t))
        throw std::runtime_error("invalid benchmark outgoing host buffer address");
    if (packed_count != benchmark_packed_rows(offsets))
        throw std::runtime_error("benchmark outgoing host count disagrees with node offsets");
}

void RouterRouting::borrow_benchmark_host(const int32_t* r, const int32_t* offsets,
                                         const int32_t* x3, const uint8_t* x4, uint64_t packed_count) {
    validate_benchmark_host(r, offsets, x3, x4, packed_count);
    for (int node = 0; node < config_.num_nodes; ++node) {
        const auto start = offsets[node];
        validate_benchmark_node_slice(x3 ? x3 + start : nullptr, x4 ? x4 + start : nullptr,
                                      offsets[node + 1] - start);
    }
    validate_benchmark_experts({r, config_.num_tokens * config_.router_top_k});
    borrowed_ = {r, offsets, x3, x4};
    initialized_ = true;
}

std::size_t RouterRouting::benchmark_packed_rows(const int32_t* offsets) const {
    if (offsets[0] != 0)
        throw std::runtime_error("benchmark node offsets must start at zero");
    for (int node = 0; node < config_.num_nodes; ++node) {
        if (offsets[node + 1] < offsets[node] ||
            static_cast<std::size_t>(offsets[node + 1] - offsets[node]) > config_.num_tokens)
            throw std::runtime_error("invalid benchmark node offsets");
    }
    return static_cast<std::size_t>(offsets[config_.num_nodes]);
}

void RouterRouting::copy_benchmark_nodes(const int32_t* offsets, const int32_t* x3, const uint8_t* x4) {
    token_indices_by_node_.resize(config_.num_nodes);
    token_masks_by_node_.resize(config_.num_nodes);
    for (int node = 0; node < config_.num_nodes; ++node) {
        const auto count = offsets[node + 1] - offsets[node];
        copy_benchmark_node_slice(count ? x3 + offsets[node] : nullptr,
                                  count ? x4 + offsets[node] : nullptr, count,
                                  token_indices_by_node_[node], token_masks_by_node_[node]);
    }
}

void RouterRouting::validate_benchmark_experts(ExpertIndexView experts) const {
    for (auto expert : experts)
        if (expert < 0 || expert >= config_.router_num_experts)
            throw std::runtime_error("benchmark R contains an out-of-range expert");
}

void RouterRouting::finish_benchmark_route() {
    validate_benchmark_experts({original_experts_.data(), original_experts_.size()});
    generate_source_weights();
    initialized_ = true;
}

void RouterRouting::initialize() {
    if (initialized_ || !config_.router_routing_enabled) return;
    token_indices_by_node_.assign(static_cast<std::size_t>(config_.num_nodes), {});
    token_masks_by_node_.assign(static_cast<std::size_t>(config_.num_nodes), {});
    expert_metadata_by_gpu_.assign(
        static_cast<std::size_t>(config_.num_nodes * config_.num_gpus_per_node), {});
    if (config_.mock_mode) {
        initialize_mock();
    } else {
        initialize_cuda();
    }
    align_expert_weights();
    initialized_ = true;
    for (int node = 0; node < config_.num_nodes; ++node) {
        RDMA_PROXY_LOG_INFO(
            "router x3/x4 ready node=", node,
            " tokens=", token_indices_by_node_[static_cast<std::size_t>(node)].size(),
            node == config_.node_rank ? " (local node ignored for RDMA)" : "");
    }
}

const std::vector<std::size_t>& RouterRouting::token_indices_for_node(int node_rank) const {
    if (!initialized_) throw std::runtime_error("router routing is not initialized");
    if (has_borrowed_source_route()) throw std::runtime_error("borrowed routing requires a token index view");
    if (node_rank < 0 || node_rank >= config_.num_nodes) {
        throw std::runtime_error("router node rank out of range");
    }
    return token_indices_by_node_[static_cast<std::size_t>(node_rank)];
}

const std::vector<uint8_t>& RouterRouting::token_masks_for_node(int node_rank) const {
    if (!initialized_) throw std::runtime_error("router routing is not initialized");
    if (has_borrowed_source_route()) throw std::runtime_error("borrowed routing requires a token mask view");
    if (node_rank < 0 || node_rank >= config_.num_nodes) {
        throw std::runtime_error("router node rank out of range");
    }
    return token_masks_by_node_[static_cast<std::size_t>(node_rank)];
}

TokenIndexView RouterRouting::token_indices_view(int node_rank) const {
    if (!has_borrowed_source_route()) return token_indices_for_node(node_rank);
    if (node_rank < 0 || node_rank >= config_.num_nodes) throw std::runtime_error("router node rank out of range");
    const auto start = borrowed_.offsets[node_rank];
    return {borrowed_.x3 ? borrowed_.x3 + start : nullptr,
            static_cast<std::size_t>(borrowed_.offsets[node_rank + 1] - start)};
}

TokenMaskView RouterRouting::token_masks_view(int node_rank) const {
    if (!has_borrowed_source_route()) {
        const auto& masks = token_masks_for_node(node_rank);
        return {masks.data(), masks.size()};
    }
    if (node_rank < 0 || node_rank >= config_.num_nodes) throw std::runtime_error("router node rank out of range");
    const auto start = borrowed_.offsets[node_rank];
    return {borrowed_.x4 ? borrowed_.x4 + start : nullptr,
            static_cast<std::size_t>(borrowed_.offsets[node_rank + 1] - start)};
}

ExpertIndexView RouterRouting::original_experts_view() const {
    if (has_borrowed_source_route()) return {borrowed_.r, config_.num_tokens * config_.router_top_k};
    const auto& experts = original_experts();
    return {experts.data(), experts.size()};
}

const int32_t* RouterRouting::device_token_indices_for_node(int node_rank) const {
    if (!initialized_) throw std::runtime_error("router routing is not initialized");
    if (node_rank < 0 || node_rank >= config_.num_nodes) {
        throw std::runtime_error("router node rank out of range");
    }
    if (config_.mock_mode) return nullptr;
#if RDMA_PROXY_HAVE_CUDA
    if (!workspace_ || !workspace_->x3 || !workspace_->host_node_offsets) {
        throw std::runtime_error("router device x3 is not initialized");
    }
    return workspace_->x3 + workspace_->host_node_offsets[node_rank];
#else
    throw std::runtime_error("router device x3 requested without CUDA support");
#endif
}

const RouterExpertMetadata& RouterRouting::expert_metadata_for_gpu(
    int node_rank,
    int local_gpu_index) const {
    if (!initialized_) throw std::runtime_error("router routing is not initialized");
    if (node_rank < 0 || node_rank >= config_.num_nodes ||
        local_gpu_index < 0 || local_gpu_index >= config_.num_gpus_per_node) {
        throw std::runtime_error("router expert metadata destination is out of range");
    }
    const auto global_gpu = static_cast<std::size_t>(
        node_rank * config_.num_gpus_per_node + local_gpu_index);
    return expert_metadata_by_gpu_.at(global_gpu);
}

const std::vector<int32_t>& RouterRouting::original_experts() const {
    if (!initialized_) throw std::runtime_error("router routing is not initialized");
    if (has_borrowed_source_route()) throw std::runtime_error("borrowed routing requires an expert view");
    return original_experts_;
}

const std::vector<float>& RouterRouting::original_weights() const {
    if (!initialized_) throw std::runtime_error("router routing is not initialized");
    if (has_borrowed_source_route()) throw std::runtime_error("borrowed benchmark route has implicit unit weights");
    return original_weights_;
}

void RouterRouting::generate_source_weights() {
    const auto top_k = static_cast<std::size_t>(config_.router_top_k);
    if (original_experts_.size() != config_.num_tokens * top_k)
        throw std::runtime_error("original routing assignment length mismatch");
    original_weights_.assign(original_experts_.size(), 1.0f);
    if (!config_.router_weighted_contributions_enabled) return;
    // Fixed integer PRNG; exactly representable positive 24-bit FP32 draws.
    // Normalize over the COMPLETE selected top-k before destination partitioning.
    const auto source_seed = splitmix64(config_.router_seed ^
        (static_cast<uint64_t>(config_.node_rank) << 32) ^
        static_cast<uint64_t>(config_.local_gpu_index) ^ 0xd1b54a32d192ed03ULL);
    for (std::size_t token = 0; token < config_.num_tokens; ++token) {
        double sum = 0.0;
        for (std::size_t k = 0; k < top_k; ++k) {
            const auto i = token * top_k + k;
            const uint64_t bits = splitmix64(source_seed ^ splitmix64(i));
            const float value = static_cast<float>(1 + (bits >> 40));
            original_weights_[i] = value;
            sum += value;
        }
        for (std::size_t k = 0; k < top_k; ++k)
            original_weights_[token * top_k + k] = static_cast<float>(
                static_cast<double>(original_weights_[token * top_k + k]) / sum);
    }
}

void RouterRouting::align_expert_weights() {
    for (auto& metadata : expert_metadata_by_gpu_) {
        metadata.top_k = config_.router_top_k;
        metadata.weighted_contributions = config_.router_weighted_contributions_enabled;
        if (!metadata.weighted_contributions) continue;
        const auto node = static_cast<std::size_t>(metadata.destination_node_rank);
        const auto& x3 = token_indices_by_node_[node];
        const auto& x4 = token_masks_by_node_[node];
        std::vector<std::size_t> original_tokens;
        if (metadata.destination_gpu_index == config_.local_gpu_index) {
            original_tokens = x3; // same-index inputs retain the full node x3 domain
        } else {
            int bit = config_.local_gpu_index - metadata.destination_gpu_index;
            if (bit < 0) bit += config_.num_gpus_per_node;
            for (std::size_t i = 0; i < x3.size(); ++i)
                if (x4[i] & (1U << bit)) original_tokens.push_back(x3[i]);
        }
        metadata.expert_route_weights.resize(metadata.expert_token_indices.size());
        for (int expert = 0; expert < metadata.experts_per_gpu; ++expert) {
            for (int32_t i = metadata.expert_offsets[expert];
                 i < metadata.expert_offsets[expert + 1]; ++i) {
                const auto row = metadata.expert_token_indices[i];
                if (row < 0 || static_cast<std::size_t>(row) >= original_tokens.size())
                    throw std::runtime_error("weighted route input row is out of bounds");
                const auto token = original_tokens[row];
                const auto begin = token * static_cast<std::size_t>(config_.router_top_k);
                bool found = false;
                for (int k = 0; k < config_.router_top_k; ++k) {
                    if (original_experts_[begin + k] == metadata.first_global_expert + expert) {
                        metadata.expert_route_weights[i] = original_weights_[begin + k];
                        found = true;
                        break;
                    }
                }
                if (!found) throw std::runtime_error("expert row does not match original routing assignment");
            }
        }
    }
}

void RouterRouting::initialize_mock() {
    const int experts_per_node = experts_per_gpu() * config_.num_gpus_per_node;
    std::vector<std::vector<std::pair<int, std::size_t>>> entries(
        static_cast<std::size_t>(config_.num_nodes));
    std::vector<std::vector<int>> routes(config_.num_tokens);

    for (std::size_t token = 0; token < config_.num_tokens; ++token) {
        uint64_t random0 = splitmix64(config_.router_seed ^ static_cast<uint64_t>(token));
        const uint64_t random1 = splitmix64(random0);
        int expert = static_cast<int>(random0 % static_cast<uint64_t>(config_.router_num_experts));
        int stride = 1;
        if (config_.router_num_experts > 1) {
            stride = 1 + static_cast<int>(
                random1 % static_cast<uint64_t>(config_.router_num_experts - 1));
            while (gcd(stride, config_.router_num_experts) != 1) {
                ++stride;
                if (stride == config_.router_num_experts) stride = 1;
            }
        }

        std::map<int, int> masks;
        for (int route = 0; route < config_.router_top_k; ++route) {
            routes[token].push_back(expert);
            original_experts_.push_back(expert);
            const int node = expert / experts_per_node;
            const int local_gpu = (expert / experts_per_gpu()) % config_.num_gpus_per_node;
            int bit = config_.local_gpu_index - local_gpu;
            if (bit < 0) bit += config_.num_gpus_per_node;
            masks[node] |= 1 << bit;
            expert += stride;
            if (expert >= config_.router_num_experts) expert -= config_.router_num_experts;
        }
        for (const auto& [node, mask] : masks) {
            entries[static_cast<std::size_t>(node)].emplace_back(mask, token);
        }
    }

    generate_source_weights();
    std::vector<std::vector<int32_t>> expert_lists(
        static_cast<std::size_t>(config_.router_num_experts));

    for (int node = 0; node < config_.num_nodes; ++node) {
        auto& node_entries = entries[static_cast<std::size_t>(node)];
        std::stable_sort(node_entries.begin(), node_entries.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first > rhs.first;
        });
        auto& output = token_indices_by_node_[static_cast<std::size_t>(node)];
        auto& masks = token_masks_by_node_[static_cast<std::size_t>(node)];
        output.reserve(node_entries.size());
        masks.reserve(node_entries.size());
        for (const auto& entry : node_entries) {
            masks.push_back(static_cast<uint8_t>(entry.first));
            output.push_back(entry.second);
        }

        std::vector<int32_t> gpu_positions(
            static_cast<std::size_t>(config_.num_gpus_per_node), 0);
        for (std::size_t node_position = 0;
             node_position < node_entries.size();
             ++node_position) {
            const auto& [mask, token] = node_entries[node_position];
            for (const int expert : routes[token]) {
                if (expert / experts_per_node != node) continue;
                const int local_gpu =
                    (expert / experts_per_gpu()) % config_.num_gpus_per_node;
                expert_lists[static_cast<std::size_t>(expert)].push_back(
                    local_gpu == config_.local_gpu_index
                        ? static_cast<int32_t>(node_position)
                        : gpu_positions[static_cast<std::size_t>(local_gpu)]);
            }
            for (int gpu = 0; gpu < config_.num_gpus_per_node; ++gpu) {
                int bit = config_.local_gpu_index - gpu;
                if (bit < 0) bit += config_.num_gpus_per_node;
                if ((mask & (1 << bit)) != 0) {
                    ++gpu_positions[static_cast<std::size_t>(gpu)];
                }
            }
        }
    }

    for (int destination_node = 0; destination_node < config_.num_nodes; ++destination_node) {
        for (int destination_gpu = 0;
             destination_gpu < config_.num_gpus_per_node;
             ++destination_gpu) {
            const int global_gpu =
                destination_node * config_.num_gpus_per_node + destination_gpu;
            const int first_expert = global_gpu * experts_per_gpu();
            auto& metadata = expert_metadata_by_gpu_[static_cast<std::size_t>(global_gpu)];
            metadata.source_node_rank = config_.node_rank;
            metadata.source_gpu_index = config_.local_gpu_index;
            metadata.destination_node_rank = destination_node;
            metadata.destination_gpu_index = destination_gpu;
            metadata.num_nodes = config_.num_nodes;
            metadata.num_gpus_per_node = config_.num_gpus_per_node;
            metadata.num_experts = config_.router_num_experts;
            metadata.experts_per_gpu = experts_per_gpu();
            metadata.first_global_expert = first_expert;
            metadata.num_tokens = config_.num_tokens;
            metadata.expert_offsets.push_back(0);
            for (int local_expert = 0; local_expert < experts_per_gpu(); ++local_expert) {
                const auto& list = expert_lists[static_cast<std::size_t>(
                    first_expert + local_expert)];
                metadata.expert_token_indices.insert(
                    metadata.expert_token_indices.end(), list.begin(), list.end());
                metadata.expert_offsets.push_back(static_cast<int32_t>(
                    metadata.expert_token_indices.size()));
            }
        }
    }
}

void RouterRouting::initialize_cuda() {
#if RDMA_PROXY_HAVE_CUDA
    check_cuda(cudaSetDevice(config_.cuda_device_id), "cudaSetDevice for router routing");
    const std::size_t num_tokens = config_.num_tokens;
    const std::size_t top_k = static_cast<std::size_t>(config_.router_top_k);
    const std::size_t num_experts = static_cast<std::size_t>(config_.router_num_experts);
    const std::size_t num_nodes = static_cast<std::size_t>(config_.num_nodes);
    const std::size_t num_masks = std::size_t{1} << config_.num_gpus_per_node;
    const std::size_t num_input_chunks =
        (num_tokens + kKernelChunkTokens - 1) / kKernelChunkTokens;
    const std::size_t input_num_bins = num_experts + num_nodes * num_masks;
    const std::size_t max_node_tokens =
        checked_multiply(num_tokens, std::min(top_k, num_nodes), "router x3 capacity");

    if (!workspace_) workspace_ = make_workspace(config_);
    auto& w = *workspace_;
    if (w.device != config_.cuda_device_id || w.tokens != num_tokens ||
        w.experts != config_.router_num_experts || w.top_k != config_.router_top_k ||
        w.nodes != config_.num_nodes || w.gpus != config_.num_gpus_per_node)
        throw std::logic_error("incompatible router workspace");
    if (!w.allocated) {
        if (w.r) throw std::runtime_error("previous router workspace setup failed");
        const auto assignments = checked_multiply(num_tokens, top_k, "router assignments");
        const auto input_scratch = checked_multiply(num_input_chunks, input_num_bins, "router input scratch");
        const auto reordered_bins = static_cast<std::size_t>(
            (experts_per_gpu() + 1) * config_.num_gpus_per_node);
        const auto reordered_scratch = checked_multiply(
            checked_multiply(num_nodes, num_input_chunks, "router reordered chunks"), reordered_bins,
            "router reordered scratch");
        w.allocate(w.r, assignments, "cudaMalloc R");
        w.allocate(w.expert_offsets, num_experts + 1, "cudaMalloc expert offsets");
        w.allocate(w.expert_indices, assignments, "cudaMalloc expert indices");
        w.allocate(w.x3, max_node_tokens, "cudaMalloc x3");
        w.allocate(w.x4, max_node_tokens, "cudaMalloc x4");
        w.allocate(w.node_offsets, num_nodes + 1, "cudaMalloc node offsets");
        w.allocate(w.input_counts, input_scratch, "cudaMalloc input counts");
        w.allocate(w.input_prefixes, input_scratch, "cudaMalloc input prefixes");
        w.allocate(w.mask_offsets, num_nodes * num_masks, "cudaMalloc mask offsets");
        w.allocate(w.reordered_counts, reordered_scratch, "cudaMalloc reordered counts");
        w.allocate(w.reordered_prefixes, reordered_scratch, "cudaMalloc reordered prefixes");
        w.allocate(w.host_node_offsets, num_nodes + 1, "cudaMallocHost node offsets", true);
        w.allocate(w.host_x3, max_node_tokens, "cudaMallocHost x3", true);
        w.allocate(w.host_x4, max_node_tokens, "cudaMallocHost x4", true);
        w.allocated = true;
    }
    device_tables_ = {w.expert_offsets, w.expert_indices, w.node_offsets, w.x3, w.x4};
    generate_r_cuda_raw(
        w.r, static_cast<int>(num_tokens), config_.router_top_k,
        config_.router_num_experts, config_.router_seed, nullptr);
    get_expert_token_idx_node_mask_cuda_raw(
        w.r, static_cast<int>(num_tokens), config_.router_top_k,
        config_.router_num_experts, experts_per_gpu(), config_.num_gpus_per_node,
        config_.local_gpu_index, w.expert_indices, w.expert_offsets, w.x3, w.x4,
        w.node_offsets, w.input_counts, w.input_prefixes, w.mask_offsets,
        w.reordered_counts, w.reordered_prefixes, nullptr);

    check_cuda(cudaMemcpy(
        w.host_node_offsets, w.node_offsets, (num_nodes + 1) * sizeof(int32_t),
        cudaMemcpyDeviceToHost), "cudaMemcpy D2H router node offsets");
    const int32_t total_x3 = w.host_node_offsets[num_nodes];
    if (total_x3 < 0 || static_cast<std::size_t>(total_x3) > max_node_tokens) {
        throw std::runtime_error("router kernel returned an invalid x3 length");
    }
    check_cuda(cudaMemcpy(
        w.host_x3, w.x3, static_cast<std::size_t>(total_x3) * sizeof(int32_t),
        cudaMemcpyDeviceToHost), "cudaMemcpy D2H router x3 to pinned memory");
    check_cuda(cudaMemcpy(
        w.host_x4, w.x4, static_cast<std::size_t>(total_x3) * sizeof(uint8_t),
        cudaMemcpyDeviceToHost), "cudaMemcpy D2H router x4 to pinned memory");

    original_experts_.resize(num_tokens * top_k);
    check_cuda(cudaMemcpy(original_experts_.data(), w.r, original_experts_.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost), "cudaMemcpy D2H original routing assignments");
    generate_source_weights();
    std::vector<int32_t> host_expert_offsets(num_experts + 1);
    std::vector<int32_t> host_expert_token_indices(num_tokens * top_k);
    check_cuda(cudaMemcpy(
        host_expert_offsets.data(), w.expert_offsets,
        host_expert_offsets.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H router expert offsets");
    check_cuda(cudaMemcpy(
        host_expert_token_indices.data(), w.expert_indices,
        host_expert_token_indices.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H router expert token indices");

    for (std::size_t global_gpu = 0;
         global_gpu < num_nodes * static_cast<std::size_t>(config_.num_gpus_per_node);
         ++global_gpu) {
        const int first_expert = static_cast<int>(global_gpu) * experts_per_gpu();
        const int32_t begin = host_expert_offsets[static_cast<std::size_t>(first_expert)];
        const int32_t end = host_expert_offsets[
            static_cast<std::size_t>(first_expert + experts_per_gpu())];
        if (begin < 0 || end < begin ||
            static_cast<std::size_t>(end) > host_expert_token_indices.size()) {
            throw std::runtime_error("router kernel returned invalid expert metadata offsets");
        }
        auto& metadata = expert_metadata_by_gpu_[global_gpu];
        metadata.source_node_rank = config_.node_rank;
        metadata.source_gpu_index = config_.local_gpu_index;
        metadata.destination_node_rank =
            static_cast<int>(global_gpu) / config_.num_gpus_per_node;
        metadata.destination_gpu_index =
            static_cast<int>(global_gpu) % config_.num_gpus_per_node;
        metadata.num_nodes = config_.num_nodes;
        metadata.num_gpus_per_node = config_.num_gpus_per_node;
        metadata.num_experts = config_.router_num_experts;
        metadata.experts_per_gpu = experts_per_gpu();
        metadata.first_global_expert = first_expert;
        metadata.num_tokens = num_tokens;
        metadata.expert_token_indices.assign(
            host_expert_token_indices.begin() + begin,
            host_expert_token_indices.begin() + end);
        metadata.expert_offsets.reserve(
            static_cast<std::size_t>(experts_per_gpu()) + 1);
        for (int local_expert = 0; local_expert <= experts_per_gpu(); ++local_expert) {
            metadata.expert_offsets.push_back(
                host_expert_offsets[static_cast<std::size_t>(
                    first_expert + local_expert)] - begin);
        }
    }

    for (std::size_t node = 0; node < num_nodes; ++node) {
        const int32_t begin = w.host_node_offsets[node];
        const int32_t end = w.host_node_offsets[node + 1];
        if (begin < 0 || end < begin || end > total_x3) {
            throw std::runtime_error("router kernel returned invalid per-node x3 offsets");
        }
        auto& output = token_indices_by_node_[node];
        auto& masks = token_masks_by_node_[node];
        output.reserve(static_cast<std::size_t>(end - begin));
        masks.reserve(static_cast<std::size_t>(end - begin));
        for (int32_t position = begin; position < end; ++position) {
            const int32_t token = w.host_x3[position];
            if (token < 0 || static_cast<std::size_t>(token) >= num_tokens) {
                throw std::runtime_error("router kernel returned an out-of-range token index");
            }
            output.push_back(static_cast<std::size_t>(token));
            const auto mask = w.host_x4[position];
            const unsigned meaningful_mask = config_.num_gpus_per_node == 8 ?
                0xffU : ((1U << config_.num_gpus_per_node) - 1U);
            if (mask == 0 || (static_cast<unsigned>(mask) & ~meaningful_mask) != 0) {
                throw std::runtime_error("router kernel returned an invalid x4 GPU mask");
            }
            masks.push_back(mask);
        }
    }
#else
    throw std::runtime_error("router routing requires a CUDA build unless mock_mode=true");
#endif
}

}  // namespace rdma_proxy
