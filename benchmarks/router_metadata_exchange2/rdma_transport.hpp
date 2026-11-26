#pragma once
#include "metadata_qp.hpp"
#include "two_node_layout.hpp"
#include <memory>
#include <sstream>

namespace rme2 {
struct MetadataEndpoint {
    rdma_proxy::PeerConnectionInfo host;
    rdma_proxy::MemoryRegionInfo indices{};

    std::string serialize() const {
        const auto legacy = rdma_proxy::serialize_peer_info(host);
        if (!indices.addr) return legacy;
        std::ostringstream out;
        out << "rme2_device_indices_v1 " << indices.addr << ' ' << indices.rkey
            << ' ' << indices.length << '\n' << legacy;
        return out.str();
    }

    static MetadataEndpoint parse(const std::string& descriptor, size_t expected_index_bytes) {
        MetadataEndpoint endpoint;
        if (!expected_index_bytes) {
            // The legacy parser rejects the non-numeric direct-mode prefix.
            endpoint.host = rdma_proxy::deserialize_peer_info(descriptor);
            return endpoint;
        }
        const auto newline = descriptor.find('\n');
        const auto fields = descriptor.substr(0, newline);
        std::istringstream header(fields);
        auto& mr = endpoint.indices;
        std::string version, extra;
        if (newline == std::string::npos || fields.find('-') != std::string::npos
            || !(header >> version >> mr.addr >> mr.rkey >> mr.length)
            || version != "rme2_device_indices_v1" || (header >> extra) || !mr.addr
            || mr.addr % alignof(int32_t) || mr.length != expected_index_bytes
            || mr.addr > std::numeric_limits<uint64_t>::max() - mr.length)
            throw std::runtime_error("invalid or incompatible RDMA device index descriptor");
        endpoint.host = rdma_proxy::deserialize_peer_info(descriptor.substr(newline + 1));
        const auto& host_mr = endpoint.host.recv_buffer;
        if (host_mr.addr > std::numeric_limits<uint64_t>::max() - host_mr.length ||
            (mr.addr < host_mr.addr + host_mr.length && host_mr.addr < mr.addr + mr.length))
            throw std::runtime_error("overlapping or invalid RDMA receive regions");
        return endpoint;
    }
};

inline MetadataQp::WritePrefix device_index_write(uint64_t send_arena,
        const rdma_proxy::MemoryRegionInfo& receive_indices, int node, int source, int capacity, int count) {
    if (!receive_indices.addr || !count) return {};
    const auto record = two::record_offset((1 - node) * gpus + source, capacity);
    const auto values = (record + two::header_words) * sizeof(int32_t);
    const auto destination = two::remote_indices_offset(source, capacity) * sizeof(int32_t);
    return {send_arena + values, receive_indices.addr + destination,
            receive_indices.rkey, uint32_t(size_t(count) * sizeof(int32_t))};
}

// Control descriptors travel through Gloo during setup only. No TCP metadata
// payload or host-staged send buffer is used by this transport.
struct MetadataRdma {
    std::unique_ptr<rdma_proxy::RdmaContext> context;
    std::unique_ptr<MetadataQp> qp;
    rdma_proxy::MemoryRegionInfo send_mr{}, receive_mr{}, remote{};
    rdma_proxy::MemoryRegionInfo device_indices_mr{}, remote_indices{};
    std::vector<rdma_proxy::Completion> completions;
    int node, gpu, capacity;
    bool connected = false;

    MetadataRdma(int n, int g, int cap, const char* device, int port, int gid,
                 void* hbm, void* dram, void* device_indices) : node(n), gpu(g), capacity(cap) {
        rdma_proxy::ProxyConfig config;
        config.node_rank = node; config.num_nodes = 2; config.local_gpu_index = gpu;
        config.rdma_device_name = device; config.rdma_port = port; config.gid_index = gid;
        config.send_queue_depth = 32; config.recv_queue_depth = 32; config.cq_depth = 64;
        context = std::make_unique<rdma_proxy::RdmaContext>(config);
        context->initialize();
        const auto bytes = two::words(capacity) * sizeof(int32_t);
        send_mr = context->register_memory(hbm, bytes, "metadata relay HBM");
        receive_mr = context->register_memory(dram, bytes, "metadata receive DRAM");
        if (device_indices)
            device_indices_mr = context->register_memory(device_indices,
                two::remote_indices_bytes(capacity), "metadata receive index HBM");
        qp = std::make_unique<MetadataQp>(*context, config);
        completions.reserve(16);
    }
    std::string descriptor() const {
        rdma_proxy::PeerConnectionInfo info;
        info.node_rank = node; info.gpu_index = gpu; info.recv_buffer = receive_mr;
        info.qps.push_back(qp->local_info());
        return MetadataEndpoint{info, device_indices_mr}.serialize();
    }
    void connect(const std::string& descriptor) {
        if (connected) throw std::runtime_error("RDMA already connected");
        const auto endpoint = MetadataEndpoint::parse(descriptor, device_indices_mr.length);
        const auto& info = endpoint.host;
        if (info.node_rank != 1 - node || info.gpu_index != gpu || info.qps.size() != 1 ||
            !info.recv_buffer.addr || info.recv_buffer.length != receive_mr.length)
            throw std::runtime_error("RDMA peer topology/arena mismatch");
        remote = info.recv_buffer;
        remote_indices = endpoint.indices;
        qp->connect(info.qps.front());
        for (int s = 0; s < two::messages; ++s) qp->post_receive(1);
        connected = true;
    }
    void post(uint64_t iteration, int source, int count) {
        const size_t from = two::record_offset((1 - node) * 8 + source, capacity) * 4;
        const size_t to = two::record_offset(node * 8 + source, capacity) * 4;
        const auto first = device_index_write(send_mr.addr, remote_indices, node, source, capacity, count);
        qp->post(two::send_id(iteration, source), send_mr.addr + from, send_mr.lkey,
                 remote.addr + to, remote.rkey, uint32_t(two::record_bytes(count)), uint32_t(source), first);
    }
    void post_x3_x4(uint64_t iteration, int count) {
        const size_t offset = two::x3_x4_offset(capacity) * sizeof(int32_t);
        qp->post(two::send_id(iteration, two::x3_x4_tag), send_mr.addr + offset,
            send_mr.lkey, remote.addr + offset, remote.rkey,
            uint32_t(two::x3_x4_bytes(count)), two::x3_x4_tag);
    }
    // Destroy QP before deregistering its memory; caller destroys transport
    // before freeing either CUDA allocation, including on failure.
    ~MetadataRdma() { qp.reset(); context.reset(); }
};
}
