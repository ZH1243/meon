#pragma once
#include "two_node_layout.hpp"

namespace rme2::receiver {
// Outside the registered wire arena. Each directory entry owns one cache-line-separated
// readiness flag on each local destination GPU; epochs are never reset/reused.
struct alignas(64) Flag { uint64_t epoch; };
struct alignas(64) Result { uint64_t epoch; int error; };
constexpr int records = two::sources + 1; // 16 expert headers plus x3/x4 header
constexpr int timeout_error = 1, record_error = 2;
inline size_t flags_offset(int capacity) {
    const size_t bytes = two::words(capacity) * sizeof(int32_t);
    return (bytes + 63) & ~size_t(63);
}
inline size_t allocation_bytes(int capacity) {
    return flags_offset(capacity) + records * sizeof(Flag);
}

// Private mapped host memory, separate from the two-node RDMA wire arena.
// Leave a 32-byte header so R and x3 start on coalesced-store boundaries.
struct SourceSnapshot {
    int32_t* node_offsets;
    int32_t* x3;
    uint8_t* x4;
    int32_t* r;
};
constexpr int snapshot_header_words = 8;
inline size_t source_snapshot_bytes(int tokens) {
    return (snapshot_header_words + size_t(tokens) * (8 + two::nodes)) * sizeof(int32_t)
        + size_t(tokens) * two::nodes * sizeof(uint8_t);
}
inline SourceSnapshot source_snapshot(int32_t* base, int tokens) {
    auto* r = base + snapshot_header_words;
    auto* x3 = r + size_t(tokens) * 8;
    return {base, x3, reinterpret_cast<uint8_t*>(x3 + size_t(tokens) * two::nodes), r};
}
#ifdef __CUDACC__
__host__ __device__
#endif
inline bool valid_source_offsets(int start, int split, int end, int tokens) {
    return start == 0 && split >= 0 && split <= tokens && end >= split && end - split <= tokens;
}
} // namespace rme2::receiver
