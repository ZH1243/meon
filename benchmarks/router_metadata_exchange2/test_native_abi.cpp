#include "native_proxy.h"
#include "../../include/rdma_proxy_c_api.h"
#include <cstddef>
#include <iostream>

// Compare the actual public C layout with ctypes without a CUDA runtime.
int main() {
    std::cout << "size " << sizeof(Rme2Stats) << '\n';
#define FIELD(name) std::cout << #name << ' ' << offsetof(Rme2Stats, name) << '\n'
    FIELD(rdma_work_complete_ms);
    FIELD(local_staging_gpu_ms);
    FIELD(local_staging_wait_ms);
    FIELD(staging_before_d2h_wait_ms);
    FIELD(receiver_gpu_ms);
    FIELD(receiver_wait_ms);
    FIELD(received_index_bytes);
    FIELD(x3_x4_received_bytes);
    FIELD(receiver_copy_bytes);
    FIELD(payload_copy_bytes);
#undef FIELD

#define SIZE(type) std::cout << #type ".size " << sizeof(type) << '\n'
#define OFFSET(type, name) std::cout << #type "." #name " " << offsetof(type, name) << '\n'
    SIZE(RdmaProxyBenchmarkSource);
    OFFSET(RdmaProxyBenchmarkSource, offsets);
    OFFSET(RdmaProxyBenchmarkSource, indices);
    OFFSET(RdmaProxyBenchmarkSource, count);
    SIZE(RdmaProxyBenchmarkRouting);
    OFFSET(RdmaProxyBenchmarkRouting, struct_size);
    OFFSET(RdmaProxyBenchmarkRouting, num_sources);
    OFFSET(RdmaProxyBenchmarkRouting, sources);
    OFFSET(RdmaProxyBenchmarkRouting, remote_x3);
    OFFSET(RdmaProxyBenchmarkRouting, remote_x4);
    OFFSET(RdmaProxyBenchmarkRouting, remote_count);
    OFFSET(RdmaProxyBenchmarkRouting, device_r);
    OFFSET(RdmaProxyBenchmarkRouting, device_node_offsets);
    OFFSET(RdmaProxyBenchmarkRouting, device_x3);
    OFFSET(RdmaProxyBenchmarkRouting, device_x4);
    SIZE(RdmaProxyBenchmarkHostRoute);
    OFFSET(RdmaProxyBenchmarkHostRoute, struct_size);
    OFFSET(RdmaProxyBenchmarkHostRoute, node_offset_count);
    OFFSET(RdmaProxyBenchmarkHostRoute, packed_count);
    OFFSET(RdmaProxyBenchmarkHostRoute, assignment_count);
    OFFSET(RdmaProxyBenchmarkHostRoute, node_offsets);
    OFFSET(RdmaProxyBenchmarkHostRoute, x3);
    OFFSET(RdmaProxyBenchmarkHostRoute, x4);
    OFFSET(RdmaProxyBenchmarkHostRoute, r);
#undef OFFSET
#undef SIZE
}
