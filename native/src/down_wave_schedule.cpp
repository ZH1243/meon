#include "down_wave_schedule.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace rdma_proxy {
namespace {

constexpr std::uint64_t int32_max = std::numeric_limits<std::int32_t>::max();

std::uint64_t ceil_div(std::uint64_t value, std::uint64_t divisor) {
    return value / divisor + (value % divisor != 0);
}

void check_geometry(const DownWaveGeometry& g, std::size_t num_experts) {
    if (!g.cluster_rows || !g.num_groups || !g.group_size || !g.max_wave_work_ids) {
        throw std::runtime_error("down wave geometry must be positive");
    }
    if (num_experts > int32_max + 1 || g.num_groups > int32_max ||
        g.num_groups - 1 > int32_max / g.group_size) {
        throw std::runtime_error("down wave expert and column coordinates must fit int32");
    }
}

DownWaveLayout layout_for_blocks(const DownWaveGeometry& g, std::uint64_t blocks) {
    if (!blocks) return {};
    // QuACK addresses cluster work IDs with signed int32.
    if (blocks > int32_max / g.num_groups / g.group_size) {
        throw std::runtime_error("down wave work ID count exceeds int32");
    }
    const auto rows = blocks * g.num_groups;
    const auto wave = std::min(rows * g.group_size, g.max_wave_work_ids);
    if (wave % g.group_size) {
        throw std::runtime_error("down wave must contain whole column groups");
    }
    const auto prefixes = ceil_div(rows, wave / g.group_size);
    const auto bytes = (4 * rows + prefixes) * sizeof(std::int32_t);
    if (bytes > std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("down wave payload exceeds addressable size");
    }
    return {static_cast<std::size_t>(rows), static_cast<std::size_t>(prefixes),
            static_cast<std::size_t>(wave), static_cast<std::size_t>(bytes)};
}

struct ExpertStream {
    std::uint64_t offset;
    std::uint64_t blocks;
    std::uint64_t cursor = 0;
};

}  // namespace

DownWaveLayout down_wave_capacity(const DownWaveGeometry& g, std::uint64_t max_routes,
                                  std::size_t num_experts) {
    check_geometry(g, num_experts);
    if (max_routes > int32_max || (max_routes && !num_experts)) {
        throw std::runtime_error("invalid down wave route capacity");
    }
    // Each nonempty expert gets its first M block for one row. Every further
    // block costs cluster_rows additional rows, regardless of the expert.
    const auto active = std::min<std::uint64_t>(max_routes, num_experts);
    const auto blocks = active + (max_routes - active) / g.cluster_rows;
    if (g.max_wave_work_ids % g.group_size) {
        // QuACK can still launch short iterations: their entire work fits in
        // a single aligned wave. Bound the table normally and allow one prefix.
        auto single_wave = g;
        single_wave.max_wave_work_ids = std::numeric_limits<std::uint64_t>::max();
        return layout_for_blocks(single_wave, blocks);
    }
    return layout_for_blocks(g, blocks);
}

DownWaveLayout down_wave_layout(const DownWaveGeometry& g,
                                const std::vector<std::size_t>& counts) {
    check_geometry(g, counts.size());
    std::uint64_t routes = 0;
    std::uint64_t blocks = 0;
    for (const auto count : counts) {
        if (count > int32_max - routes) {
            throw std::runtime_error("down wave M coordinates exceed int32");
        }
        routes += count;
        blocks += ceil_div(count, g.cluster_rows);
    }
    return layout_for_blocks(g, blocks);
}

DownWaveLayout build_down_wave_schedule(const DownWaveGeometry& g,
                                        const std::vector<std::size_t>& counts,
                                        std::int32_t* packed, std::size_t capacity_bytes) {
    const auto layout = down_wave_layout(g, counts);
    if (!layout.bytes) return layout;
    if (!packed || reinterpret_cast<std::uintptr_t>(packed) % alignof(std::int32_t) ||
        capacity_bytes < layout.bytes) {
        throw std::runtime_error("down wave destination is missing, unaligned, or too small");
    }
    std::vector<ExpertStream> streams;
    streams.reserve(counts.size());
    std::uint64_t offset = 0;
    for (const auto count : counts) {
        streams.push_back({offset, ceil_div(count, g.cluster_rows)});
        offset += count;
    }
    std::vector<std::uint64_t> last_wave(g.num_groups);
    const auto rows_per_wave = layout.wave_work_ids / g.group_size;
    std::uint64_t written = 0;
    auto* out = packed;
    for (std::uint64_t group = 0; group < g.num_groups; ++group) {
        for (std::size_t i = 0; i < counts.size(); ++i) {
            const auto expert = group % 2 ? counts.size() - 1 - i : i;
            auto& stream = streams[expert];
            const auto stop = (group + 1) * stream.blocks;
            if (stream.cursor >= stop) continue;

            // Finish this expert's current group and borrow its subsequent
            // rows up to the next wave boundary or the end of its stream.
            const auto needed = stop - stream.cursor;
            const auto remainder = (written + needed) % rows_per_wave;
            const auto borrow = remainder ? rows_per_wave - remainder : 0;
            const auto end = std::min(stop + borrow, g.num_groups * stream.blocks);
            auto column = stream.cursor / stream.blocks;
            while (stream.cursor < end) {
                const auto column_end = std::min(end, (column + 1) * stream.blocks);
                for (; stream.cursor < column_end; ++stream.cursor) {
                    const auto index = stream.cursor - column * stream.blocks;
                    const auto block = column % 2 ? stream.blocks - 1 - index : index;
                    const auto local_begin = block * g.cluster_rows;
                    const auto length = std::min(g.cluster_rows, counts[expert] - local_begin);
                    *out++ = static_cast<std::int32_t>(expert);
                    *out++ = static_cast<std::int32_t>(stream.offset + local_begin);
                    *out++ = static_cast<std::int32_t>(stream.offset + local_begin + length);
                    *out++ = static_cast<std::int32_t>(column * g.group_size);
                    ++written;
                }
                last_wave[column++] = (written - 1) / rows_per_wave;
            }
        }
    }
    std::size_t ready_groups = 0;
    for (std::size_t wave = 0; wave < layout.prefix_entries; ++wave) {
        while (ready_groups < last_wave.size() && last_wave[ready_groups] <= wave) {
            ++ready_groups;
        }
        *out++ = static_cast<std::int32_t>(ready_groups);
    }
    return layout;
}

}  // namespace rdma_proxy
