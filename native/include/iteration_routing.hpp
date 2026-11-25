#pragma once

#include "router_routing.hpp"

#include <functional>
#include <map>
#include <memory>
#include <vector>

namespace rdma_proxy {

struct IterationRouteData {
    std::unique_ptr<RouterRouting> source;
    std::vector<RouterExpertMetadata> incoming_experts;
    std::map<int, RouterX3Metadata> incoming_nodes;
};

// Exactly one worker owns generation, exchange and staging. prepare() returns
// vectors with independent ownership after this generation's transfers drain.
class IterationRouting {
public:
    explicit IterationRouting(const ProxyConfig&);
    ~IterationRouting();
    void initialize();
    IterationRouteData prepare(uint64_t iteration,
        const std::function<void(const RouterRouting&, void*)>& stage = {});
    void shutdown();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rdma_proxy
