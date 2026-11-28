#pragma once
#include "../../include/rdma_proxy_c_api.h"
#include "receiver_layout.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace rme2 {
// The caller holds the proxy mutex and has observed a successful wait for iteration.
inline void collect_benchmark_route_views(uint64_t iteration, int tokens,
        const std::array<Installed, two::sources>& installed, const two::InstalledX3X4& remote,
        const receiver::SourceSnapshot& local, RdmaProxyBenchmarkSource* sources, int source_capacity,
        RdmaProxyBenchmarkRouting* routing, RdmaProxyBenchmarkHostRoute* host_route) {
    if (!sources || source_capacity < two::sources || !routing || routing->struct_size < sizeof(*routing) ||
        !host_route || host_route->struct_size < sizeof(*host_route))
        throw std::runtime_error("invalid benchmark route view outputs");
    if (remote.iteration != iteration || remote.count < 0 || (remote.count && (!remote.x3 || !remote.x4)))
        throw std::runtime_error("invalid remote x3/x4 view for this iteration");
    const auto* offsets = local.node_offsets;
    if (!offsets || !local.r || !receiver::valid_source_offsets(offsets[0], offsets[1], offsets[2], tokens) ||
        (offsets[2] && (!local.x3 || !local.x4)))
        throw std::runtime_error("invalid local source snapshot for this iteration");

    // Finish checking every source before exposing any descriptor to the caller.
    std::array<RdmaProxyBenchmarkSource, two::sources> views;
    for (int rank = 0; rank < two::sources; ++rank) {
        const auto& view = installed[rank];
        if (view.iteration != iteration || !view.offsets || view.count < 0 || (view.count && !view.indices))
            throw std::runtime_error("invalid expert list view for this iteration");
        views[rank] = {view.offsets, view.indices, static_cast<uint64_t>(view.count)};
    }
    std::copy(views.begin(), views.end(), sources);
    *routing = {sizeof(*routing), two::sources, sources, remote.x3, remote.x4,
                static_cast<uint64_t>(remote.count), 0, 0, 0, 0};
    *host_route = {sizeof(*host_route), two::nodes + 1, static_cast<uint64_t>(offsets[two::nodes]),
                   static_cast<uint64_t>(tokens) * 8, offsets, local.x3, local.x4, local.r};
}
} // namespace rme2
