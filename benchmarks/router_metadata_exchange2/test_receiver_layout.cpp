#include "receiver_layout.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <vector>

static void check_source_layout(int tokens) {
    using namespace rme2::receiver;
    const auto bytes = source_snapshot_bytes(tokens);
    std::vector<int32_t> memory((bytes + 3) / 4);
    const auto out = source_snapshot(memory.data(), tokens);
    const auto start = reinterpret_cast<uintptr_t>(memory.data());
    const auto limit = start + bytes;
    std::array<std::pair<uintptr_t, size_t>, 4> regions{{
        {reinterpret_cast<uintptr_t>(out.node_offsets), 3 * sizeof(int32_t)},
        {reinterpret_cast<uintptr_t>(out.x3), size_t(2 * tokens) * sizeof(int32_t)},
        {reinterpret_cast<uintptr_t>(out.x4), size_t(2 * tokens)},
        {reinterpret_cast<uintptr_t>(out.r), size_t(8 * tokens) * sizeof(int32_t)},
    }};
    std::sort(regions.begin(), regions.end());
    auto previous_end = start;
    for (const auto& [address, length] : regions) {
        if (address < previous_end || address > limit || length > limit - address)
            throw std::runtime_error("source snapshot fields overlap or exceed their allocation");
        previous_end = address + length;
    }
    if (reinterpret_cast<uintptr_t>(out.r) % alignof(int32_t))
        throw std::runtime_error("source snapshot expert IDs are not word aligned");
}

int main() {
    using namespace rme2;
    static_assert(sizeof(receiver::Flag) == 64 && alignof(receiver::Flag) == 64);
    static_assert(sizeof(receiver::Result) == 64);
    for (int tokens : {1, 33, 257, 20480}) check_source_layout(tokens);
    if (!receiver::valid_source_offsets(0, 0, 0, 7) ||
        !receiver::valid_source_offsets(0, 0, 7, 7) ||
        !receiver::valid_source_offsets(0, 7, 7, 7) ||
        !receiver::valid_source_offsets(0, 7, 14, 7) ||
        receiver::valid_source_offsets(1, 2, 3, 7) ||
        receiver::valid_source_offsets(0, -1, 6, 7) ||
        receiver::valid_source_offsets(0, 6, 5, 7) ||
        receiver::valid_source_offsets(0, 8, 8, 7) ||
        receiver::valid_source_offsets(0, 6, 14, 7) ||
        receiver::valid_source_offsets(0, 0, std::numeric_limits<int>::max(), 7))
        throw std::runtime_error("invalid source snapshot slice bounds");
    for (int tokens : {1, 33, 257, 20480, 10000000}) {
        const int capacity = tokens * 8;
        const auto wire_bytes = two::words(capacity) * 4;
        const auto start = receiver::flags_offset(capacity);
        if (start < wire_bytes || start % 64 || start - wire_bytes >= 64 ||
            receiver::allocation_bytes(capacity) != start + receiver::records * sizeof(receiver::Flag))
            throw std::runtime_error("flags overlap wire records or are misaligned");
        for (int node = 0; node < 2; ++node)
            for (int source = 0; source < 8; ++source) {
                const auto local = two::record_offset(node * 8 + source, capacity) * 4;
                const auto remote = two::record_offset((1 - node) * 8 + source, capacity) * 4;
                const auto bytes = two::record_bytes(capacity);
                if (!(local + bytes <= remote || remote + bytes <= local) ||
                    local + bytes > two::x3_x4_offset(capacity) * 4)
                    throw std::runtime_error("receiver record overlaps incoming RDMA or x3/x4");
            }
    }
    std::cout << "receiver layout passed\n";
}
