#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define RDMA_PROXY_C_API __declspec(dllexport)
#elif defined(__GNUC__)
#define RDMA_PROXY_C_API __attribute__((visibility("default")))
#else
#define RDMA_PROXY_C_API
#endif

/* Optional CPU scatter setup. Fixed routing binds this once before iteration zero.
 * Dynamic routing binds this after prepare_routing and before prepare_iteration
 * for each nonempty generation. Previous device reads must already have retired.
 * source_pairs is host int32[num_sources,2] in gather input order; output_bases
 * is host uint64[num_sources]. Host outputs are counts[num_sources] and
 * output_to_input[num_sources,capacity] (-1 in unused tails). All host arrays
 * are used synchronously. scatter_addresses is borrowed CUDA int64[routes,2]
 * for tile_n > 0, or int64[routes] for row-major; retain it and output buffers
 * through their last native use (shutdown for static routing, next successful
 * routing preparation for dynamic routing).
 */
typedef struct RdmaProxyCpuScatterConfig {
    uint64_t struct_size;
    uint64_t num_sources, routes, capacity, width, tile_n, element_bytes;
    uintptr_t scatter_addresses;
    const int32_t* source_pairs;
    const uint64_t* output_bases;
    int64_t* counts;
    int64_t* output_to_input;
} RdmaProxyCpuScatterConfig;

/*
 * Stable C ABI for embedding the C++ proxy in another process (for example,
 * a Python worker using ctypes). The handle is intentionally opaque so C++
 * types and standard-library objects never cross the shared-library boundary.
 */
typedef struct RdmaProxyHandle RdmaProxyHandle;

/* Optional expert-wave schedule for dynamic routing with gather-ready publication.
 * Geometry matches the down kernel: M rows per cluster, N column groups, work
 * IDs per group, and maximum resident cluster count (including the SM budget).
 * During startup, query the capacity in int32 words, then configure once on an
 * idle handle with generation zero installed. The caller owns the contiguous
 * device allocation through shutdown. The proxy owns the pinned staging slab. */
typedef struct RdmaProxyDownWaveGeometry {
    uint64_t struct_size, cluster_rows, num_groups, group_size, max_wave_work_ids;
} RdmaProxyDownWaveGeometry;
RDMA_PROXY_C_API int rdma_proxy_get_down_wave_capacity(
    RdmaProxyHandle* handle, const RdmaProxyDownWaveGeometry* geometry, uint64_t* capacity_words);
RDMA_PROXY_C_API int rdma_proxy_configure_down_wave_schedule(
    RdmaProxyHandle* handle, const RdmaProxyDownWaveGeometry* geometry, uintptr_t device_slab,
    uint64_t capacity_words);
/* Start after installing this route and before prepare_iteration. The call
 * copies the counts synchronously. The allocation contains int32[table_rows,4] followed
 * immediately by int32[prefix_entries]. Empty routes return two zero lengths.
 * After the final up-gather publication is enqueued, the native worker submits
 * one H2D transfer of the live payload and records this generation's event. */
RDMA_PROXY_C_API int rdma_proxy_start_down_wave_schedule(
    RdmaProxyHandle* handle, uint64_t iteration, const uint64_t* counts, uint64_t num_experts,
    uint64_t* table_rows, uint64_t* prefix_entries);
/* Wait on the host for this iteration's event record (timeout_ms > 0), then
 * enqueue a wait on the nonzero consumer stream. Keep the publisher running.
 * Join all device readers before replacing the route or releasing the slab. */
RDMA_PROXY_C_API int rdma_proxy_wait_down_wave_schedule(
    RdmaProxyHandle* handle, uint64_t iteration, uintptr_t cuda_stream, uint64_t timeout_ms);

/* Optional startup reservation for threaded CPU scatter with dynamic routing.
 * Call on an initialized handle before preparing or installing generation zero.
 * address_columns is 1 for row-major output and 2 for N-tile-major output.
 * The proxy owns the pinned storage through shutdown. Each nonempty generation
 * still binds its current scatter destinations, weights and publication policy. */
RDMA_PROXY_C_API int rdma_proxy_reserve_threaded_cpu_scatter(
    RdmaProxyHandle* handle, int32_t address_columns);

RDMA_PROXY_C_API int rdma_proxy_configure_cpu_scatter(
    RdmaProxyHandle* handle, const RdmaProxyCpuScatterConfig* config);

/* Optional weighted CPU scatter setup: call after configure_cpu_scatter and
 * before prepare_iteration. Borrow FP32[routes] for the same lifetime as scatter_addresses.
 * Each selected range is copied once before its gather readiness publication,
 * unless deferred publication is configured below. */
RDMA_PROXY_C_API int rdma_proxy_configure_cpu_route_weights(
    RdmaProxyHandle* handle, uintptr_t packed_weights, uint64_t routes);

/* Opt-in one-copy CPU metadata publication. Call after scatter and optional
 * weights setup, before prepare_iteration. scatter_addresses begins a 16-byte
 * aligned borrowed allocation. Its exact size is routes*columns*8 unweighted;
 * weighted storage places FP32[routes] at round_up(scatter_bytes,16), and bytes
 * includes that weight array. The whole allocation shares the scatter_addresses lifetime.
 * After final gather-ready publication is enqueued, CPU staging is reconstructed
 * from the completed host gather table, once per M batch. One slab H2D copy and
 * event record then follow on the publication stream.
 * The existing per-batch behavior remains default unless this API is called. */
RDMA_PROXY_C_API int rdma_proxy_configure_cpu_scatter_deferred(
    RdmaProxyHandle* handle, uint64_t bytes);
/* Alternative opt-in policy with the same contiguous slab layout. One persistent
 * native CPU worker consumes committed host gather prefixes while notification
 * publication continues. It packs each complete M-batch descriptor once, outside
 * notification locks; one slab H2D/event follows final gather submission and all
 * packing. Configure exactly one of deferred/threaded before prepare_iteration.
 * Native shutdown joins the worker and retires its DMA before returning. */
RDMA_PROXY_C_API int rdma_proxy_configure_cpu_scatter_threaded(
    RdmaProxyHandle* handle, uint64_t bytes);
/* Combined threaded configuration for a freshly installed nonempty dynamic route.
 * The worker builds the compact address lookup and then packs published gather rows.
 * The call consumes source_pairs and output_bases synchronously. counts and the
 * optional output_to_input are writable HOST buffers borrowed until a successful
 * wait_cpu_scatter_ready for this epoch, route retirement, or shutdown. Read them
 * only after that synchronization. Device destinations follow the threaded slab
 * layout above. Pass packed_weights=0 for an unweighted route. */
RDMA_PROXY_C_API int rdma_proxy_configure_cpu_scatter_async(
    RdmaProxyHandle* handle, const RdmaProxyCpuScatterConfig* config,
    uintptr_t packed_weights, uint64_t slab_bytes);
/* Host-wait, bounded by timeout_ms > 0, until this prepared iteration's event
 * RECORD is submitted, then enqueue cudaStreamWaitEvent on a nonzero caller
 * stream. Supports deferred and threaded modes; worker errors are reported.
 * This does not synchronize the GPU. Call before down GEMM submission;
 * another host thread must continue driving notification/communication work.
 * Join compute consumers before preparing the next iteration or shutting down. */
RDMA_PROXY_C_API int rdma_proxy_wait_cpu_scatter_ready(
    RdmaProxyHandle* handle, uint64_t iteration, uintptr_t cuda_stream, uint64_t timeout_ms);

/* Reserve once during dynamic route setup. nodes lists N source/destination
 * nodes in combine order, G is the number of local GPUs, C is num_tokens.
 * device_slab is borrowed until shutdown. Its int32 layout is the intra-node
 * rows [N*G, N*C+2], then inter-node rows [N, C+2] if include_inter is 1.
 * words is the exact combined element count. The worker owns pinned staging. */
RDMA_PROXY_C_API int rdma_proxy_configure_return_maps(
    RdmaProxyHandle* handle, const int32_t* nodes, uint64_t num_nodes,
    uint64_t capacity, uintptr_t device_slab, uint64_t words, int32_t include_inter);
/* Call once for a refreshed route before prepare_iteration, even when this GPU
 * has zero expert work. The two host outputs (num_nodes each, in nodes order)
 * are filled synchronously from installed route extents. The worker builds
 * the maps in parallel and uploads them after final up-gather submission. */
RDMA_PROXY_C_API int rdma_proxy_start_return_maps(
    RdmaProxyHandle* handle, uint64_t iteration, uint64_t* intra_counts,
    uint64_t* outgoing_counts, uint64_t num_nodes);
/* Wait for the current generation's event record, then order the nonzero
 * consumer stream after its upload. Host wait is bounded by timeout_ms > 0.
 * Preserve the combine's data readiness waits. Join GPU readers before the
 * next route installation or shutdown. */
RDMA_PROXY_C_API int rdma_proxy_wait_return_maps(
    RdmaProxyHandle* handle, uint64_t iteration, uintptr_t cuda_stream, uint64_t timeout_ms);

/* Original source [num_tokens,top_k] assignments, before destination row mapping.
 * Query with both arrays NULL and capacity=0; otherwise copy both host arrays.
 * Weights are normalized FP32 when enabled, exactly one when disabled.
 * These assignments refer to the installed routing generation and allow
 * references independent of gather/scatter packing. */
RDMA_PROXY_C_API int rdma_proxy_get_local_router_assignments(
    RdmaProxyHandle* handle, int32_t* experts, float* weights, uint64_t capacity,
    uint64_t* count, uint64_t* num_tokens, uint64_t* top_k);

/* Device-buffer kinds requested through RdmaProxyDeviceBufferAllocator. */
typedef enum RdmaProxyDeviceBufferKind {
    RDMA_PROXY_DEVICE_BUFFER_RDMA_SEND = 1,
    RDMA_PROXY_DEVICE_BUFFER_RDMA_RECEIVE = 2,
    RDMA_PROXY_DEVICE_BUFFER_NVLINK_RECEIVE = 3,
    RDMA_PROXY_DEVICE_BUFFER_ROUTER_A_IDX = 4,
    RDMA_PROXY_DEVICE_BUFFER_GATHER_TABLE = 5,
    RDMA_PROXY_DEVICE_BUFFER_GATHER_READY_ROWS = 6,
    RDMA_PROXY_DEVICE_BUFFER_ROUTER_ROUTE_WEIGHTS = 7,
    RDMA_PROXY_DEVICE_BUFFER_INPUT_POOL = 8,
    RDMA_PROXY_DEVICE_BUFFER_POOLED_A_IDX = 9,
    RDMA_PROXY_DEVICE_BUFFER_POOLED_GATHER_TABLE = 10,
} RdmaProxyDeviceBufferKind;

typedef enum RdmaProxyDeviceBufferElementType {
    RDMA_PROXY_DEVICE_BUFFER_BF16 = 1,
    RDMA_PROXY_DEVICE_BUFFER_FP16 = 2,
    RDMA_PROXY_DEVICE_BUFFER_FP32 = 3,
    RDMA_PROXY_DEVICE_BUFFER_INT32 = 4,
} RdmaProxyDeviceBufferElementType;

/*
 * Allocation request passed to an embedding runtime. dimensions contains up
 * to two logical tensor dimensions; bytes is always the required allocation
 * size. Negative peer/source identifiers mean "not applicable". A router
 * NVLINK_RECEIVE request whose remote source GPU equals this proxy's local GPU
 * is a logical alias request: the allocator must return the pointer previously
 * supplied for that source node's RDMA_RECEIVE request.
 * In pooled up-GEMM mode the first INPUT_POOL request is [global GPUs *
 * num_tokens, token_dimension]. All NVLINK_RECEIVE requests are views into
 * this pool, starting at row (source_node_rank * num_gpus_per_node +
 * source_gpu_index) * num_tokens. RDMA_RECEIVE requests use peer_rank and
 * the local GPU index for these source coordinates. Keep the pool alive
 * through transport teardown. If shutdown cannot confirm peer IPC closure,
 * retain the exported pool for the process lifetime. POOLED_A_IDX and
 * POOLED_GATHER_TABLE supply the packed indices and four-column up-GEMM
 * descriptors, respectively.
 */
typedef struct RdmaProxyDeviceBufferRequest {
    uint32_t struct_size;
    uint32_t version;
    int32_t kind;
    int32_t element_type;
    int32_t peer_rank;
    int32_t source_node_rank;
    int32_t source_gpu_index;
    uint32_t dimension_count;
    uint64_t bytes;
    uint64_t dimensions[2];
} RdmaProxyDeviceBufferRequest;

/* Return a device pointer as uintptr_t, or zero to reject the request. */
typedef uintptr_t (*RdmaProxyDeviceBufferAllocator)(
    void* context,
    const RdmaProxyDeviceBufferRequest* request);

/* Increment this value only when the exported ABI changes incompatibly. */
RDMA_PROXY_C_API int rdma_proxy_abi_version(void);

/*
 * Create a proxy from the same argv accepted by the rdma_cpu_proxy executable.
 * The argv strings are copied before this function returns.
 * Returns NULL on failure; retrieve the diagnostic with rdma_proxy_last_error.
 */
RDMA_PROXY_C_API RdmaProxyHandle* rdma_proxy_create(
    int argc,
    const char* const* argv);

/*
 * Install an allocator before rdma_proxy_initialize(). The proxy borrows all
 * returned buffers and never frees them; the embedding runtime must keep them
 * alive until after rdma_proxy_shutdown()/rdma_proxy_destroy(). Passing NULL
 * restores the proxy's normal internal CUDA allocation behavior.
 */
RDMA_PROXY_C_API int rdma_proxy_set_device_buffer_allocator(
    RdmaProxyHandle* handle,
    RdmaProxyDeviceBufferAllocator allocator,
    void* context);

RDMA_PROXY_C_API int rdma_proxy_initialize(RdmaProxyHandle* handle);
RDMA_PROXY_C_API int rdma_proxy_run(RdmaProxyHandle* handle);

/* Optional per-iteration routing. After every reader of the previous iteration
 * has finished, prepare_routing commits the next routing tables and local
 * input staging. Repeated preparation of the same pending iteration is a no-op.
 * prepare_iteration also does this when the caller does not need to bind compute
 * views first. Use the current logical extents for QuACK launches; the device
 * allocator supplies stable, capacity-sized backing allocations. */
RDMA_PROXY_C_API int rdma_proxy_dynamic_routing_enabled(RdmaProxyHandle* handle, int32_t* enabled);
RDMA_PROXY_C_API int rdma_proxy_prepare_routing(RdmaProxyHandle* handle, uint64_t iteration);

/* Optional benchmark exchange provider. Set struct_size before either call.
 * Source views are in global source-rank order, offsets has local_experts + 1
 * entries, and count is the compact index count. Each index must address a
 * valid input row, with nondecreasing indices within each expert. With
 * router_benchmark_metadata_validation_enabled=false the caller guarantees
 * this ordering and range, valid expert IDs, and valid, ordered x3/x4 slices.
 * remote_x3/x4 describes the same-local-GPU peer's incoming payload (one
 * destination mask per token).
 * The four device arrays belong to this rank's completed metadata graph.
 * Complete exchange and local input staging before install. Install once per
 * retired epoch's successor, including epoch zero, before preparing or
 * launching compute. The install functions below specify pointer lifetimes. */
typedef struct RdmaProxyBenchmarkSource {
    const int32_t* offsets;
    const int32_t* indices;
    uint64_t count;
} RdmaProxyBenchmarkSource;

typedef struct RdmaProxyBenchmarkRouting {
    uint64_t struct_size;
    int32_t num_sources;
    const RdmaProxyBenchmarkSource* sources;
    const int32_t* remote_x3;
    const uint8_t* remote_x4;
    uint64_t remote_count;
    uintptr_t device_r, device_node_offsets, device_x3, device_x4;
} RdmaProxyBenchmarkRouting;

/* Current source GPU's outgoing route, in host memory. Counts are in elements.
 * node_offsets has num_nodes + 1 entries and ends at packed_count. x3 and x4
 * each contain packed_count entries. r contains num_tokens * top_k expert IDs. */
typedef struct RdmaProxyBenchmarkHostRoute {
    uint64_t struct_size, node_offset_count, packed_count, assignment_count;
    const int32_t* node_offsets;
    const int32_t* x3;
    const uint8_t* x4;
    const int32_t* r;
} RdmaProxyBenchmarkHostRoute;

/* One destination GPU's writable exchange slots, in global source-rank order.
 * indices points to num_sources device addresses (each starts at the first
 * int32 index, after any record header). Capacity is in int32 elements. */
typedef struct RdmaProxyBenchmarkDeviceIndices {
    uint32_t struct_size, num_sources;
    uint64_t capacity_per_source;
    const uintptr_t* indices;
} RdmaProxyBenchmarkDeviceIndices;

typedef struct RdmaProxyBenchmarkConfig {
    uint64_t struct_size, num_tokens, router_seed, completion_timeout_ms;
    int32_t cuda_device_id, node_rank, local_gpu_index, num_gpus_per_node, num_nodes;
    int32_t rdma_port, gid_index;
    char rdma_device_name[128];
} RdmaProxyBenchmarkConfig;

RDMA_PROXY_C_API int rdma_proxy_benchmark_metadata_enabled(RdmaProxyHandle* handle, int32_t* enabled);
RDMA_PROXY_C_API int rdma_proxy_get_benchmark_routing_config(
    RdmaProxyHandle* handle, RdmaProxyBenchmarkConfig* config);
/* All input pointers are borrowed only for this synchronous call. */
RDMA_PROXY_C_API int rdma_proxy_install_benchmark_routing(
    RdmaProxyHandle* handle, uint64_t iteration, const RdmaProxyBenchmarkRouting* routing);
/* Supply pinned host indices for direct GPU uploads on CUDA backends.
 * Keep the source offsets and indices readable and immutable until successful
 * retirement or successful payload proxy shutdown, including after a partial
 * installation failure. All other inputs, including the sources descriptor
 * array, are borrowed only during this call.
 * Serialize benchmark routing control calls on each handle. */
RDMA_PROXY_C_API int rdma_proxy_install_borrowed_benchmark_routing(
    RdmaProxyHandle* handle, uint64_t iteration, const RdmaProxyBenchmarkRouting* routing);
/* Same borrow and retirement contract for the incoming expert lists. The
 * outgoing host route is read only during this synchronous call. Set both
 * descriptors' struct_size fields. routing's four device addresses are ignored. */
RDMA_PROXY_C_API int rdma_proxy_install_borrowed_benchmark_host_routing(
    RdmaProxyHandle* handle, uint64_t iteration, const RdmaProxyBenchmarkRouting* routing,
    const RdmaProxyBenchmarkHostRoute* host_route);
/* Borrow both the incoming expert lists and this rank's outgoing node_offsets,
 * x3, x4 and r. Keep those arrays readable and immutable until successful
 * retirement or successful shutdown, including after a partial installation
 * failure. The descriptors and remote_x3/x4 need only survive this call.
 * Serialize queries and control calls with retirement, installation and shutdown.
 * Retiring this installation ends access to the outgoing route and assignments
 * until the next successful install. The host-route installer above retains
 * its call-duration lifetime for all four outgoing arrays. */
RDMA_PROXY_C_API int rdma_proxy_install_borrowed_benchmark_route_views(
    RdmaProxyHandle* handle, uint64_t iteration, const RdmaProxyBenchmarkRouting* routing,
    const RdmaProxyBenchmarkHostRoute* host_route);
/* Same host-view lifetime as install_borrowed_benchmark_route_views. For each
 * source, already publish routing->sources[i].count int32 indices (identical
 * to the host list) in the corresponding device slot. Complete incoming DMA
 * and establish owner-GPU visibility before calling. The external allocator
 * must return these exact disjoint slots for ROUTER_A_IDX and keep their
 * addresses stable until successful shutdown. The installer zeros the tails
 * up to the current common length. Retire all CPU/GPU readers on every rank
 * before reusing any slot. The device descriptor/address array is call-scoped. */
RDMA_PROXY_C_API int rdma_proxy_install_borrowed_benchmark_device_indices(
    RdmaProxyHandle* handle, uint64_t iteration, const RdmaProxyBenchmarkRouting* routing,
    const RdmaProxyBenchmarkHostRoute* host_route, const RdmaProxyBenchmarkDeviceIndices* device_indices);
/* Concurrent-compute version with the same buffer ownership and visibility
 * requirements. The local staging gather must also have completed. The live
 * compute extents and route views are available when the call returns. On a
 * refreshed route with early local readiness, prepare_iteration can release
 * this GPU's own gather rows. Run run_iteration concurrently with the GEMM
 * to finish transport preparation and publish the remaining rows. */
RDMA_PROXY_C_API int rdma_proxy_install_benchmark_device_indices_for_compute(
    RdmaProxyHandle* handle, uint64_t iteration, const RdmaProxyBenchmarkRouting* routing,
    const RdmaProxyBenchmarkHostRoute* host_route, const RdmaProxyBenchmarkDeviceIndices* device_indices);
/* Call after run_iteration and any external consumers of this iteration have
 * completed. Drains the remaining native readers. Retire on every rank before
 * any rank starts the next exchange or otherwise reuses the records. */
RDMA_PROXY_C_API int rdma_proxy_retire_borrowed_benchmark_routing(RdmaProxyHandle* handle, uint64_t iteration);
RDMA_PROXY_C_API int rdma_proxy_get_routing_extents(
    RdmaProxyHandle* handle, uint64_t* indices_per_source, uint64_t* table_rows);

/* Finite-run iteration control for embedding runtimes that own the main loop. */
RDMA_PROXY_C_API int rdma_proxy_get_num_iterations(
    RdmaProxyHandle* handle,
    uint64_t* num_iterations);
/*
 * Reset the gather-table publication state for one iteration and wait until
 * ready_rows=0 is visible on the configured GPU. This lets an embedding
 * runtime safely launch a readiness-gated persistent kernel before calling
 * rdma_proxy_run_iteration, possibly from another host thread.
 */
RDMA_PROXY_C_API int rdma_proxy_prepare_iteration(
    RdmaProxyHandle* handle,
    uint64_t iteration);
RDMA_PROXY_C_API int rdma_proxy_run_iteration(
    RdmaProxyHandle* handle,
    uint64_t iteration);

/* GPU-event timing for readiness-gated computation launched by the embedder. */
RDMA_PROXY_C_API int rdma_proxy_record_computation_end(
    RdmaProxyHandle* handle,
    uint64_t iteration,
    uintptr_t cuda_stream);
RDMA_PROXY_C_API int rdma_proxy_get_computation_elapsed_ms(
    RdmaProxyHandle* handle,
    uint64_t iteration,
    float* elapsed_ms);
RDMA_PROXY_C_API int rdma_proxy_get_computation_num_tokens(
    RdmaProxyHandle* handle,
    uint64_t* num_tokens);
/* Query length with counts=NULL, capacity=0; otherwise copy all local expert
 * counts in scheduler expert order. Counts exclude A_idx capacity padding. */
RDMA_PROXY_C_API int rdma_proxy_get_computation_expert_counts(
    RdmaProxyHandle* handle, uint64_t* counts, uint64_t capacity,
    uint64_t* num_experts);
/* Installed routing masks in this GPU's full staging (local source_node) or RDMA
 * receive (remote source_node) row order. Bit g means computing GPU g uses row.
 * Query length with masks=NULL, capacity=0; populated rows exclude capacity tails.
 * For dynamic routing, query after installing the desired generation. */
RDMA_PROXY_C_API int rdma_proxy_get_intra_node_source_masks(
    RdmaProxyHandle* handle, int32_t source_node, uint8_t* masks,
    uint64_t capacity, uint64_t* count);
/* Build the combine kernel's row table on the calling CPU thread after dynamic
 * routing installation and before preparing that payload iteration. source_nodes
 * lists distinct source-node IDs in kernel domain order; capacity is num_tokens.
 * rows is contiguous host int32[num_nodes * gpus_per_node][num_nodes * capacity + 2].
 * row_elements must equal the number of int32 elements in that shape.
 * Row (domain, gpu) maps input slots to compact row ordinals, with -1 for absent
 * contributions, unused capacity and other domains. Its final two elements give
 * this GPU's compact row count and this domain's live row count, respectively.
 * domain_counts receives num_nodes live row counts in source_nodes order.
 * Arrays must not overlap. All pointers are borrowed only for this synchronous call. */
RDMA_PROXY_C_API int rdma_proxy_build_intra_node_combine_rows(
    RdmaProxyHandle* handle, const int32_t* source_nodes, uint64_t num_nodes,
    uint64_t capacity, int32_t* rows, uint64_t row_elements, uint64_t* domain_counts);
RDMA_PROXY_C_API int rdma_proxy_finish(RdmaProxyHandle* handle);

/* This source GPU's original token IDs in outgoing x3 order for a destination
 * node, including the local staging domain. Query with indices=NULL, capacity=0.
 * num_tokens is the full original input size, including tokens absent from x3.
 * Routing is immutable after initialization in this benchmark. */
RDMA_PROXY_C_API int rdma_proxy_get_outgoing_token_indices(
    RdmaProxyHandle* handle, int32_t destination_node, uint64_t* indices,
    uint64_t capacity, uint64_t* count, uint64_t* num_tokens);
/* Build the final combine kernel's original-token row table after installing a
 * dynamic route and before preparing its payload iteration. destination_nodes
 * lists distinct node IDs in kernel input order. num_tokens is the configured
 * source token count. rows is contiguous host int32[num_nodes][num_tokens + 2],
 * with row_elements equal to the total number of int32 elements. Row d maps
 * each original token to its ordinal in outgoing x3 for destination_nodes[d],
 * or -1 if absent. Its final two elements are the x3 length and zero.
 * Arrays must not overlap. All pointers are borrowed only for this synchronous call. */
RDMA_PROXY_C_API int rdma_proxy_build_inter_node_combine_rows(
    RdmaProxyHandle* handle, const int32_t* destination_nodes, uint64_t num_nodes,
    uint64_t num_tokens, int32_t* rows, uint64_t row_elements);

/* Reverse RDMA: caller owns the stable tiled source/destination GPU storage.
 * Empty buffers still supply a non-null one-element allocation for registration.
 * Setup is collective among same-index GPUs across nodes, before iterations.
 * Source rows are incoming remote x3 rows; destination rows are outgoing x3 rows.
 * Layout is [width/tile_n, rows, tile_n], packed independently per peer.
 * The returned device address aliases a native-owned mapped pinned int32 flag.
 * After all CTA stores for groups [0,N), GPU must system-release store N there.
 * prepare_iteration resets it; run_iteration waits for all outgoing and incoming
 * return group markers, including GPUDirect receive visibility, before returning.
 * Keep GPU storage alive until shutdown. Join all GPU writers before shutdown.
 * Existing QP workers send returns after their current forward task finishes.
 */
typedef struct RdmaProxyNetworkReturnBuffer {
    int32_t peer;
    uintptr_t source;
    uintptr_t destination;
    uint64_t source_rows;
    uint64_t destination_rows;
} RdmaProxyNetworkReturnBuffer;
RDMA_PROXY_C_API int rdma_proxy_get_rdma_return_receive_rows(
    RdmaProxyHandle* handle, int32_t peer, uint64_t* rows);
/* In dynamic mode both descriptor row fields are the registered capacity
 * (num_tokens); buffer addresses stay fixed. Current routing supplies the
 * valid send/receive lengths and the packed tile strides per iteration. */
RDMA_PROXY_C_API int rdma_proxy_configure_rdma_returns(
    RdmaProxyHandle* handle, const RdmaProxyNetworkReturnBuffer* buffers, uint64_t count,
    uint64_t width, uint64_t tile_n, uint64_t group_tiles, uint64_t element_bytes,
    uint64_t chunk_bytes, uint64_t timeout_ms, uintptr_t* mapped_device_prefix);
/* Optional post-iteration SHA256 transport check; same-index peers call in rank order. */
RDMA_PROXY_C_API int rdma_proxy_check_rdma_return(
    RdmaProxyHandle* handle, int32_t peer, const char* sent_sha256, const char* received_sha256);

/* Configure once after initialization, before the first prepare_iteration.
 * The proxy owns the mapped flag. Finish GPU writes before shutdown/destroy.
 * prepare_iteration resets/arms it; the notification receiver polls it.
 * Times are CPU detection microseconds since iteration preparation, including
 * up computation and host launch/compilation delays, not GPU timestamps.
 * read waits for the final prefix; capacity must hold up to final_groups entries.
 */
RDMA_PROXY_C_API int rdma_proxy_configure_down_ready_trace(
    RdmaProxyHandle* handle, int32_t final_groups, uintptr_t* device_address);
RDMA_PROXY_C_API int rdma_proxy_read_down_ready_trace(
    RdmaProxyHandle* handle, uint64_t iteration, int32_t* groups,
    double* times_us, uint64_t capacity, uint64_t* count);

typedef struct RdmaProxyReturnCopy {
    uintptr_t source;
    uintptr_t destination;
    uint64_t capacity_bytes; /* Minimum of source and destination capacities. */
    uint64_t row_index;      /* Index into host_valid_rows. */
    int32_t peer;            /* Same-node destination GPU index; batches group by peer. */
} RdmaProxyReturnCopy;

/* Configure after down-ready trace, before the first iteration. Pointers/events
 * are borrowed until shutdown. host_valid_rows must be pinned host int64 storage,
 * updated on the compute stream BEFORE down GEMM. tile_n=0 means row-major.
 * All destination buffers use the same packed layout/count as their sources.
 * read_down_ready_trace additionally waits for outgoing returns when configured;
 * the embedder must also coordinate incoming completion across local ranks.
 */
RDMA_PROXY_C_API int rdma_proxy_configure_return_copies(
    RdmaProxyHandle* handle, const RdmaProxyReturnCopy* copies, uint64_t count,
    const int64_t* host_valid_rows, uint64_t row_count, uint64_t width,
    uint64_t tile_n, uint64_t group_tiles, uint64_t element_bytes,
    int32_t per_group, uintptr_t down_done_event);
/* Call after recording this iteration's down_done_event, never before. */
RDMA_PROXY_C_API int rdma_proxy_arm_return_copies(RdmaProxyHandle* handle, uint64_t iteration);

/* CPU DRAM readiness snapshot, enabled by nvlink_return_completion_notifications_enabled.
 * prefix=N means groups [0,N) arrived from every other local GPU and this
 * GPU's retained outputs are ready for those groups. No final
 * reduction or reverse RDMA is implied. Requires the matching prepared iteration.
 * With the feature enabled, read_down_ready_trace also waits for incoming
 * notifications to be consumed and outgoing notifications to be dispatched
 * and consumed by their destinations.
 */
RDMA_PROXY_C_API int rdma_proxy_get_return_ready_prefix(
    RdmaProxyHandle* handle, uint64_t iteration, int32_t* prefix);

/* Available after configure_down_ready_trace with
 * nvlink_return_ready_gpu_publication_enabled=true. Borrowed int32 flags:
 * host_address is pinned CPU memory, device_address is separate GPU HBM.
 * Host reads must be atomic/acquire; GPU consumers must use a suitable
 * acquire polling protocol. Neither address may be modified by the embedder.
 * The host value denotes enqueued publication; the device update is async.
 * prepare_iteration resets both flags to zero and completes the GPU reset
 * before returning; read_down_ready_trace waits for final GPU publication.
 * Finish all GPU consumers before the next preparation or shutdown. Both
 * addresses remain stable until shutdown, when their allocations are freed.
 */
RDMA_PROXY_C_API int rdma_proxy_get_return_ready_publication(
    RdmaProxyHandle* handle, uintptr_t* host_address, uintptr_t* device_address);

RDMA_PROXY_C_API int rdma_proxy_shutdown(RdmaProxyHandle* handle);
RDMA_PROXY_C_API void rdma_proxy_destroy(RdmaProxyHandle* handle);

/* Convenience entry point equivalent to create/initialize/run/shutdown/destroy. */
RDMA_PROXY_C_API int rdma_proxy_run_argv(
    int argc,
    const char* const* argv);

/*
 * Returns the most recent error for the calling thread. The pointer remains
 * valid until the next C API call on that thread.
 */
RDMA_PROXY_C_API const char* rdma_proxy_last_error(void);

#ifdef __cplusplus
}
#endif
