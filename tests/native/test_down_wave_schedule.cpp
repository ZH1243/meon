#include "down_wave_schedule.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace rdma_proxy;

namespace {
constexpr std::int32_t canary = -773;

template <class F>
void expect_error(F operation) {
    try {
        operation();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("invalid wave input was accepted");
}

void compare_reference_table() {
    // Generated with QuACK run.wave_ready.build_expert_wave_schedule.
    // Expert 1 is empty, several experts borrow across groups, and the final
    // wave holds one descriptor. The odd groups reverse partial blocks too.
    const DownWaveGeometry g{2, 4, 2, 6};
    const std::vector<std::size_t> counts{3, 0, 1, 2};
    const std::vector<std::int32_t> expected{
        0, 0, 2, 0,    0, 2, 3, 0,    0, 2, 3, 2,    2, 3, 4, 0,
        2, 3, 4, 2,    2, 3, 4, 4,    3, 4, 6, 0,    3, 4, 6, 2,
        3, 4, 6, 4,    0, 0, 2, 2,    0, 0, 2, 4,    0, 2, 3, 4,
        3, 4, 6, 6,    2, 3, 4, 6,    0, 2, 3, 6,    0, 0, 2, 6,
        0, 0, 1, 3, 3, 4,
    };
    std::vector<std::int32_t> slab(expected.size() + 2, canary);
    const auto layout = build_down_wave_schedule(g, counts, slab.data() + 1,
                                                 expected.size() * sizeof(std::int32_t));
    assert(layout.table_rows == 16 && layout.prefix_entries == 6 && layout.wave_work_ids == 6);
    assert(layout.bytes == expected.size() * sizeof(std::int32_t));
    assert(slab.front() == canary && slab.back() == canary);
    assert(std::equal(expected.begin(), expected.end(), slab.begin() + 1));

    // Reuse the slab with fewer routes. The prefix now follows the shorter
    // table and no payload bytes beyond the valid prefix are overwritten.
    std::fill(slab.begin(), slab.end(), canary);
    const auto short_layout = build_down_wave_schedule({2, 2, 3, 99}, {2, 1},
        slab.data(), slab.size() * sizeof(std::int32_t));
    const std::vector<std::int32_t> short_expected{
        0, 0, 2, 0,    0, 0, 2, 3,    1, 2, 3, 0,    1, 2, 3, 3,    2,
    };
    assert(short_layout.wave_work_ids == 12 && short_layout.prefix_entries == 1);
    assert(std::equal(short_expected.begin(), short_expected.end(), slab.begin()));
    assert(slab[short_expected.size()] == canary);
}

void check_capacity_and_prefix() {
    const DownWaveGeometry g{3, 5, 2, 8};
    for (std::size_t max_routes = 0; max_routes <= 24; ++max_routes) {
        const auto cap = down_wave_capacity(g, max_routes, 3);
        std::vector<std::int32_t> slab(cap.bytes / sizeof(std::int32_t));
        std::size_t largest = 0;
        for (std::size_t a = 0; a <= max_routes; ++a) {
            for (std::size_t b = 0; b <= max_routes - a; ++b) {
                const std::vector<std::size_t> counts{a, b, max_routes - a - b};
                const auto layout = build_down_wave_schedule(g, counts, slab.data(), cap.bytes);
                largest = std::max(largest, layout.bytes);
                if (!layout.bytes) continue;
                // Every expert block occurs exactly once per column group.
                // Recompute each prefix from coverage of the actual output.
                using Row = std::array<std::int32_t, 4>;
                std::vector<Row> expected;
                std::size_t offset = 0;
                for (std::size_t expert = 0; expert < counts.size(); ++expert) {
                    for (std::size_t row = 0; row < counts[expert]; row += g.cluster_rows) {
                        for (std::size_t group = 0; group < g.num_groups; ++group) {
                            expected.push_back({static_cast<std::int32_t>(expert),
                                static_cast<std::int32_t>(offset + row),
                                static_cast<std::int32_t>(offset + std::min<std::size_t>(
                                    counts[expert], row + g.cluster_rows)),
                                static_cast<std::int32_t>(group * g.group_size)});
                        }
                    }
                    offset += counts[expert];
                }
                std::vector<Row> actual(layout.table_rows);
                std::vector<std::size_t> last_wave(g.num_groups);
                for (std::size_t row = 0; row < layout.table_rows; ++row) {
                    std::copy_n(&slab[4 * row], 4, actual[row].begin());
                    const auto group = actual[row][3] / g.group_size;
                    assert(group < g.num_groups);
                    last_wave[group] = row / (layout.wave_work_ids / g.group_size);
                }
                std::sort(expected.begin(), expected.end());
                std::sort(actual.begin(), actual.end());
                assert(actual == expected);
                for (std::size_t wave = 0; wave < layout.prefix_entries; ++wave) {
                    const auto stop = std::find_if(last_wave.begin(), last_wave.end(),
                        [wave](std::size_t last) { return last > wave; });
                    assert(slab[layout.table_rows * 4 + wave] == stop - last_wave.begin());
                }
            }
        }
        assert(largest == cap.bytes);
    }
}

void check_geometry_and_bounds() {
    const DownWaveGeometry h800{192, 28, 2, 124};
    const auto layout = down_wave_layout(h800,
        {20394, 20430, 20592, 20392, 20444, 20405, 20549, 20491});
    assert(layout.table_rows == 24024 && layout.prefix_entries == 388 && layout.bytes == 385936);
    assert(down_wave_capacity(h800, 16 * 20480 * 8, 8).bytes >= layout.bytes);
    assert(build_down_wave_schedule(h800, {0, 0, 0}, nullptr, 0).bytes == 0);
    assert(down_wave_capacity(h800, 0, 8).bytes == 0);
    const DownWaveGeometry uneven_residency{128, 1, 4, 10};
    const auto cap = down_wave_capacity(uneven_residency, 1024, 2);
    const auto tiny = down_wave_layout(uneven_residency, {1, 1});
    assert(cap.bytes >= tiny.bytes && tiny.wave_work_ids == 8 && tiny.prefix_entries == 1);
    expect_error([&] { down_wave_layout(uneven_residency, {129, 1}); });

    std::array<std::int32_t, 64> slab;
    slab.fill(canary);
    expect_error([&] { build_down_wave_schedule(h800, {1}, slab.data(), 2); });
    expect_error([&] { build_down_wave_schedule(h800, {1}, nullptr, layout.bytes); });
    expect_error([] { down_wave_layout({0, 1, 1, 1}, {1}); });
    expect_error([] { down_wave_layout({1, 0, 1, 1}, {1}); });
    expect_error([] { down_wave_layout({1, 1, 0, 1}, {1}); });
    expect_error([] { down_wave_layout({1, 1, 1, 0}, {1}); });
    expect_error([] { down_wave_layout({1, 3, 2, 3}, {1}); });
    expect_error([] { down_wave_capacity({1, 1, 1, 2}, 1, 0); });
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    expect_error([&] { down_wave_layout(h800, {limit, 1}); });
    expect_error([&] { down_wave_capacity(h800, limit + 1, 1); });
    expect_error([&] { down_wave_layout({1, 1, 2, 124}, {limit}); });
    expect_error([&] { down_wave_layout({1, 3, limit, 124}, {1}); });
    assert(std::all_of(slab.begin(), slab.end(), [](std::int32_t v) { return v == canary; }));
    // A very large cluster_rows value still clips M end correctly.
    build_down_wave_schedule({std::numeric_limits<std::uint64_t>::max(), 1, 1, 1},
        {0, 3, 1}, slab.data(), sizeof(slab));
    assert(slab[0] == 1 && slab[1] == 0 && slab[2] == 3);
    assert(slab[4] == 2 && slab[5] == 3 && slab[6] == 4);
}
}  // namespace

int main() {
    compare_reference_table();
    check_capacity_and_prefix();
    check_geometry_and_bounds();
    std::cout << "test_down_wave_schedule passed\n";
}
