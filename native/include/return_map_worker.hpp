#pragma once

#include "protocol.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace rdma_proxy {

std::size_t intra_return_map_words(std::size_t nodes, std::uint64_t capacity, int peers);
std::size_t inter_return_map_words(std::size_t nodes, std::uint64_t capacity);
void fill_intra_return_rows(const std::vector<TokenMaskView>& domains, std::size_t capacity,
                            int peers, int local_gpu, int32_t* rows);
void fill_inter_return_rows(const std::vector<TokenIndexView>& domains, std::size_t capacity, int32_t* rows);

// The proxy retains the masks/token IDs until retire() or stop() has joined
// their CPU reader, and the embedder joins GPU map readers before route reuse.
struct ReturnMapRoute {
    std::vector<TokenMaskView> masks;
    std::vector<TokenIndexView> tokens;
};

class ReturnMapWorker {
public:
    ReturnMapWorker(std::size_t nodes, std::size_t capacity, int peers, int local_gpu, bool include_inter,
                    int32_t* pinned, uintptr_t device_slab, std::size_t words,
                    int device, bool mock, void* publication_stream, std::atomic<bool>& down_metadata_failed);
    ~ReturnMapWorker();
    void start(uint64_t iteration, ReturnMapRoute route, bool final_gather_submitted);
    void final_gather(uint64_t iteration);
    void wait(uint64_t iteration, uintptr_t consumer_stream, uint64_t timeout_ms);
    void rethrow_error();
    void retire();
    void stop();

private:
    void run() noexcept;
    void upload();

    const std::size_t nodes_, capacity_, intra_words_, words_;
    const int peers_, local_gpu_, device_;
    const bool include_inter_, mock_;
    int32_t* const pinned_;
    void* const destination_;
    void* const stream_;
    std::atomic<bool>& down_metadata_failed_;
    void* event_{nullptr};
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable changed_;
    enum class Phase { idle, build, packed, submitted };
    Phase phase_{Phase::idle};
    bool final_{false}, busy_{false}, stopping_{false}, dma_pending_{false};
    std::optional<uint64_t> iteration_;
    ReturnMapRoute route_;
    std::exception_ptr error_;
};

}  // namespace rdma_proxy
