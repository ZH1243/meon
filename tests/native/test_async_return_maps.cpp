#include "return_map_worker.hpp"
#include "cuda_buffers.hpp"

#include <algorithm>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace rdma_proxy;

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <class F>
void rejects(F&& operation) {
    try { operation(); } catch (const std::runtime_error&) { return; }
    throw std::runtime_error("expected invalid return map operation to fail");
}

// Rows 0/1 are the first domain, rows 2/3 the second. The dense columns
// index domain*capacity+input; the last two values retain compressed sizes.
const std::vector<int32_t> expected = {
     0, 1,-1,-1,-1,-1,-1,-1, 2, 3,
    -1, 0, 1,-1,-1,-1,-1,-1, 2, 3,
    -1,-1,-1,-1,-1,-1,-1,-1, 0, 1,
    -1,-1,-1,-1, 0,-1,-1,-1, 1, 1,
     1,-1, 0,-1, 2, 0,
    -1, 0,-1,-1, 1, 0,
};

void exercise(bool mock, int32_t* pinned, int32_t* destination, void* publication, uintptr_t consumer) {
    const auto words = intra_return_map_words(2, 4, 2) + inter_return_map_words(2, 4);
    require(expected.size() == words, "wrong packed map size");
    const std::vector<uint8_t> a{1,3,2}, b{2};
    const std::vector<std::size_t> a_tokens{2,0};
    const int32_t b_token = 1;
    const ReturnMapRoute live{{{a.data(), a.size()}, {b.data(), b.size()}},
                              {TokenIndexView(a_tokens), TokenIndexView(&b_token, 1)}};
    const ReturnMapRoute empty{{{nullptr, 0}, {nullptr, 0}}, {{nullptr, 0}, {nullptr, 0}}};
    const auto device = reinterpret_cast<uintptr_t>(destination);
    std::atomic<bool> failed{false};
    ReturnMapWorker worker(2, 4, 2, 0, true, pinned, device, words, 0, mock, publication, failed);
    std::fill_n(destination, words + 1, -91);
    worker.start(1, live, false);
    worker.final_gather(0);  // A stale prefix must never release the packed maps.
    rejects([&] { worker.wait(1, consumer, 2); });
    require(std::all_of(destination, destination + words + 1, [](int32_t x) { return x == -91; }),
            "return map uploaded before final gather");
    worker.final_gather(1);
    worker.wait(1, consumer, 5000);
    synchronize_cuda_stream(reinterpret_cast<void*>(consumer), mock);
    require(std::equal(expected.begin(), expected.end(), destination), "incorrect packed return map");
    require(destination[words] == -91, "return map crossed the destination boundary");
    rejects([&] { worker.start(2, empty, true); });  // Current generation still owns the slab.
    worker.retire();
    rejects([&] { worker.wait(1, consumer, 1); });
    rejects([&] { worker.start(1, live, true); });

    worker.start(2, empty, true);
    worker.wait(2, consumer, 5000);
    synchronize_cuda_stream(reinterpret_cast<void*>(consumer), mock);
    for (std::size_t i = 0; i < 4; ++i) {
        const auto* row = destination + 10*i;
        require(row[8] == 0 && row[9] == 0, "stale intra-node counts on an empty route");
        require(std::all_of(row, row + 8, [](int32_t x) { return x == -1; }),
                "stale intra-node rows on an empty route");
    }
    require(destination[words - 1] == 0 && destination[words - 2] == 0 && destination[words] == -91,
            "incorrect inter-node counts or guard on empty route");
    worker.retire();
    worker.start(3, live, false);
    auto waiter = std::async(std::launch::async, [&] { rejects([&] { worker.wait(3, consumer, 5000); }); });
    worker.retire();  // Wake a waiter even when the final gather never arrives.
    waiter.get();
    worker.start(4, live, false);
    worker.stop();
    rejects([&] { worker.wait(4, consumer, 1); });
    rejects([&] { worker.start(5, live, true); });
}

void errors_and_optional_inter() {
    int32_t pinned[4]{}, device[4]{};
    const uint8_t mask = 1;
    std::atomic<bool> failed{false};
    ReturnMapWorker intra(1, 2, 1, 0, false, pinned, reinterpret_cast<uintptr_t>(device), 4, 0, true, nullptr, failed);
    intra.start(0, {{{&mask, 1}}, {}}, true);
    intra.wait(0, 1, 1000);
    require(std::vector<int32_t>(device, device+4) == std::vector<int32_t>{0,-1,1,1},
            "incorrect intra-only return map");
    intra.retire();
    rejects([&] { intra.start(1, {{{&mask, 3}}, {}}, false); });
    rejects([&] { intra_return_map_words(2, std::numeric_limits<int32_t>::max(), 8); });
    rejects([&] { inter_return_map_words(std::numeric_limits<std::size_t>::max(), 42); });

    int32_t bad_staging[8]{}, bad_device[8]{};
    const int32_t repeated[2]{0, 0};
    ReturnMapWorker bad(1, 2, 1, 0, true, bad_staging, reinterpret_cast<uintptr_t>(bad_device), 8, 0, true, nullptr, failed);
    bad.start(0, {{{&mask, 1}}, {{repeated, 2}}}, false);
    rejects([&] { bad.wait(0, 1, 1000); });
    require(failed.load(), "map failure was not reported to downstream consumers");
    bad.final_gather(0);
    rejects([&] { bad.rethrow_error(); });
    rejects([&] { bad.retire(); });  // Error from the worker is sticky for this proxy.
    bad.stop();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--cuda") {
#if RDMA_PROXY_HAVE_CUDA
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
        int32_t *host{}, *device{};
        if (cudaMallocHost(reinterpret_cast<void**>(&host), expected.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaMallocManaged(reinterpret_cast<void**>(&device), (expected.size() + 1) * sizeof(int32_t)) != cudaSuccess)
            throw std::runtime_error("could not reserve return-map test storage");
        void* publication = create_cuda_stream(0, true, false);
        void* consumer = create_cuda_stream(0, true, false);
        exercise(false, host, device, publication, reinterpret_cast<uintptr_t>(consumer));
        destroy_cuda_stream(consumer, false);
        destroy_cuda_stream(publication, false);
        cudaFree(device);
        cudaFreeHost(host);
        return 0;
#else
        return 77;
#endif
    }
    std::vector<int32_t> pinned(expected.size()), device(expected.size() + 1);
    exercise(true, pinned.data(), device.data(), nullptr, 1);
    errors_and_optional_inter();
}
