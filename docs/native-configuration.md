# Native proxy configuration

The executable and the embedded C ABI call the same parser. Start with a JSON file and apply scalar overrides in order as `--key=value`. The equivalent short form for `--config FILE` is `-c FILE`. Booleans on the command line recognize `1`, `true` and `yes` as true. JSON booleans use JSON syntax. Validation runs on the effective configuration. The Python worker and launchers supply additional options and override several native defaults, including geometry, NIC assignments and rank identity. See the [JSON profiles](../config), [native defaults](../native/include/config.hpp) and [parser and validation](../native/src/config.cpp).

```sh
./build/rdma_cpu_proxy --config config/example_config.json --cpu_affinity=none --num_iterations=2
```

This example uses the native mock backend. The node profiles contain the source deployment's addresses and devices. For two-node runs use the matching node launcher on each host and set the topology overrides for that deployment. The standalone example, single-node production profile and torchrun launchers select different workloads.

The shipped JSON profiles retain three legacy keys named `router_computation_kernel_enabled`, `router_computation_timeout_ms` and `nvlink_forward_computation_enabled`. The current native parser ignores these keys and other unknown scalar overrides. The worker launches its concurrent GEMM path with `--concurrent-kernel`. For custom native overrides, check the spelling against the table or inspect the effective configuration in the startup log.

## Identity, workload and network

Defaults in the tables are the C++ member initializers, before JSON or launcher overrides. A slash between values in one cell corresponds to the keys in their listed order.

| JSON or CLI key | Default | Meaning |
| --- | --- | --- |
| `node_rank`, `num_nodes` | 0 / 1 | Zero-based node identity and size. |
| `local_gpu_index`, `num_gpus_per_node`, `cuda_device_id` | 0 / 8 / 0 | Logical GPU, local group size and CUDA device ordinal. |
| `num_tokens`, `token_dimension`, `tokens_per_chunk` | 0 / 0 / 1 | Source rows, elements per row and rows per receive chunk. All must be positive in the effective config. |
| `dtype` | `bf16` | Also accepts `bfloat16`, `fp16`/`half`, `fp32`/`float` in the native transport. GEMMs have their own dtype constraints. |
| `num_iterations`, `completion_timeout_ms` | 1 / 30000 | Iteration count and positive progress timeout. The native run loop accepts zero iterations as unbounded. Stepped embedding and return notifications need a finite count. |
| `mock_mode`, `fill_test_data`, `validate_data` | false / true / true | Enable host simulation, initialize deterministic payloads, and check native transferred payloads. Projection checks are independent. |
| `cpu_affinity`, `log_level` | empty / `info` | On Linux use a CPU list or `auto` for the `nvidia-smi topo -m` mask. Empty or `none` disables binding. Logging levels are debug, info, warn and error. |
| `peers` | [] | JSON array of `node_rank`, `host` and optional `port`. Supply one entry per other node. |
| `listen_port`, `connection_manager_port` | 18515 / 18515 | Local TCP bootstrap listener and default for a missing JSON peer port. CLI `--listen_port` also sets every peer port. `--peer_port` overrides all peer ports, and `--peer_host` overrides the host for an exactly one-peer config. |
| `rdma_device_name`, `rdma_port`, `gid_index` | empty / 1 / -1 | Empty device selects the first verbs device. Use a GID index for global addressing or -1 for LID routing. |
| `rdma_device_name2`, `rdma_port2`, `gid_index2` | empty / 0 / -2 | Optional second, distinct explicitly named NIC. Port 0 and index -2 inherit the primary values. Second index -1 selects LID routing. |
| `num_qps_per_peer` | 1 | Total per peer. A dual-NIC entry needs an even total ≥2. Each NIC owns one half and its own PD, MRs and keys. Both directions use this topology. |
| `send_queue_depth`, `recv_queue_depth`, `cq_depth` | 128 / 128 / 256 | Positive capacities. Receive WQEs provide the receiver-side credit for immediate notifications. |
| `completion_poll_batch_size`, `max_in_flight_chunks_per_qp` | 16 / 1 | Positive CQ poll batch and per-QP work window. The window fits in the send queue. |
| `data_signal_interval` | 1 | Retained compatibility setting. Dynamic chunk distribution signals its data WRs for window management. |
| `rdma_chunk_per_token_sge_enabled`, `rdma_discontinuous_token_payload_enabled` | false / false | Per-row SGEs and discontinuous source payload test. The latter needs the former and disables NVLink forwarding in this explicit test mode. Router mode enables both *effective* behaviors for routed data. |
| `sequential_peer_transfers`, `local_forwarding_rdma_overlap_enabled` | false / false | Visit peers sequentially and optionally prime the first network transfer before draining local staging. Overlap requires sequential transfers. |
| `rdma_qp_calibration_enabled`, `min_bandwidth_gbps_needed` | false / 340.0 | Optional collective QP calibration before measurement, for a two-node RDMA-only workload with forwarding disabled. |
| `rdma_qp_calibration_warmup_iterations`, `rdma_qp_calibration_sample_iterations`, `rdma_qp_calibration_max_attempts`, `rdma_qp_calibration_timeout_ms` | 2 / 5 / 10 / 300000 | Bound warmups, samples, candidate QP attempts and calibration time. Ranks agree on success before accepting a candidate. |

Coverage includes `test_config`, `test_chunking`, `test_dual_rdma`, `test_qp_calibration`, `test_qp_calibration_transport` and `test_measured_run`.

## Routing and gather scheduling

| Key | Default | Meaning |
| --- | --- | --- |
| `router_routing_enabled` | false | Use original token-to-expert assignments and node/GPU masks to route a shared send buffer. Supports 2–8 GPUs per node. |
| `router_dynamic_routing_enabled` | false | Generate routes, exchange per-source expert tables and same-index remote x3/x4, and stage local input rows before every payload iteration. Applies to finite one- or two-node runs with router routing, NVLink forwarding and local input staging. All peers must select the same mode. [Launch and validation](development.md#per-iteration-routing) |
| `router_benchmark_metadata_enabled` | false | Use an external per-epoch handoff from the embedded GPU exchange benchmark. Requires dynamic routing, two nodes of eight GPUs, 128 experts, top-8, unweighted routes and at most 10,000,000 tokens per GPU. The torchrun switch `--benchmark-metadata-exchange` enables both native flags. After exchange and staging, `rdma_proxy_install_borrowed_benchmark_host_routing` takes the outgoing route in host memory and borrows the pinned expert lists. With outgoing arrays on the device, an embedding can borrow the lists with `rdma_proxy_install_borrowed_benchmark_routing` or snapshot them with `rdma_proxy_install_benchmark_routing`. Retire borrowed lists on every rank before submitting the next exchange. [Backend setup](development.md#benchmark-metadata-backend) |
| `num_experts` (alias `router_num_experts`), `top_k` (alias `router_top_k`) | 0 / 0 | Routed runs need 1–1024 experts divisible by total GPUs, 1 ≤ top-k ≤ experts, and token count × top-k within int32. |
| `router_seed`, `router_metadata_port_base` | 1234 / 19515 | Routing seed and all-to-all expert metadata listener base, with one valid port per local GPU. |
| `router_weighted_contributions_enabled` | false | Propagate source FP32 normalized weights aligned to packed routes. Disabled routing weights are unit weights. In the embedded pipeline the worker sets this from `--weighted-contributions`. |
| `router_pooled_up_gemm_enabled` | false | Group the receive slices into one destination allocation, and publish packed pool-relative indices and four-column descriptors for SM90 up GEMM. The torchrun switch is `--pooled-up-gemm`. |
| `router_computation_forwarded_inputs_only` | false | Restrict the QuACK input set to routed sources. Remote same-index inputs alias RDMA receives. Local sources enter this restricted set when local staging is enabled. |
| `router_local_input_staging_enabled` | false | Gather local-node x3 rows once from the shared send buffer and reuse the staging. Requires routed forwarding, automatic IPC destinations, `fill_test_data=false` and notifications or preparation. |
| `router_local_input_early_ready_enabled` | false | Publish the staged own-GPU input at iteration activation. Requires staging and router completion notifications. Incompatible with flush-only measurement. |
| `expert_gemm_device_capacity`, `expert_gemm_m_tile`, `expert_gemm_n_tile` | 9 / 128 / 256 | Architecture major 9 (Hopper) or 10 (Blackwell), MMA M tile and N tile. Inputs must match the external GEMM geometry. |
| `expert_gemm_dimension`, `expert_gemm_cluster_m`, `expert_gemm_max_swizzle_size` | 8192 / 1 / 8 | Output N dimension, cluster M and maximum N swizzle. N-cluster count must be divisible by the effective group size. |
| `nvlink_forward_completion_notifications_enabled`, `nvlink_forward_notification_queue_depth` | false / 1024 | Receiver-owned bounded queues. Requires forwarding and synchronized batches. Depth also bounds pending dispatch work in prepared/ping-pong modes. |
| `nvlink_forward_notification_scheduling_mode` | `per_notification` | Advance eligible experts on each arrival, or use `complete_round` to collect one completed ordinal from each non-exhausted source in deterministic phase and source order. Both use the same gather-table format. |
| `nvlink_forward_notification_scheduling_diagnostics` | false | Scheduling/flush/publication counters and timings. Needs routed notifications. |
| `nvlink_forward_notification_flush_per_entry_enabled`, `nvlink_forward_notification_flush_only_enabled` | false / false | Normally coalesce newly emitted rows. Per-entry publishes each row separately. Flush-only is a diagnostic first-row copy that bypasses scheduling. These modes are mutually exclusive. Complete rounds exclude flush-only. |
| `nvlink_forward_notification_flag_update_mode` | `stream-write` | Publish the ready prefix after the table copy with a stream write or a `memcpy` from stable pinned values. |

SM100 non-blockscaled two-CTA MMA with an even cluster M and MMA M of 128 or 256 uses half that MMA M per CTA. Descriptor capacity is CTA rows × cluster M. Every int32 table row is `[expert, cid_n_base, (start, end) per input]` with half-open ranges into each input's expert-index list. Each gather row must reach device memory before its ready prefix is visible. CPU scatter metadata has a separate event in deferred/threaded modes. A phase-end record proves even a source with zero batches is exhausted. The scheduler must keep future-round batches private while handling early-local readiness. See [scheduler storage and publication](../native/src/cuda_buffers.cpp) and [round collector](../native/include/forwarding_rounds.hpp).

Coverage includes `test_router_routing`, `test_route_weights`, `test_gemm_geometry`, `test_global_metadata_exchange`, `test_forward_round_scheduler`, `test_forward_round_protocol`, `test_forwarding_rounds`, `test_early_local_ready` and `test_forward_round_integration`.

## Forwarding and return policies

| Key | Default | Meaning |
| --- | --- | --- |
| `nvlink_forwarding_enabled` | false | Enable intra-node forwarding across 2–8 local GPUs. The CLI also accepts `nvlink_forwarding_enable`. |
| `nvlink_forward_threshold_tokens`, `nvlink_forward_threshold_chunks` | 0 / 0 | Fixed threshold in tokens or receive chunks. If both are set, they must agree in fixed-threshold mode. |
| `nvlink_forward_min_threshold_chunks`, `nvlink_forward_max_threshold_chunks` | 0 / 0 | Set both positive to use dynamic batching with minimum ≤ maximum. |
| `nvlink_forward_out_of_order_chunks_enabled`, `nvlink_forward_chunk_tokens` | false / 0 | Optionally admit out-of-order chunks in dynamic finite non-router mode. `chunk_tokens` sets the legacy forwarding chunk size. |
| `nvlink_forward_use_round_robin`, `nvlink_routing_probability`, `nvlink_routing_seed` | false / 0.5 / 1 | Non-router forwarding patterns. Round robin uses fixed threshold `chunk_tokens × (GPUs - 1)`. Router forwarding always follows x4 in ordered chunks. |
| `nvlink_forward_use_batch_api`, `nvlink_forward_stream_nonblocking`, `nvlink_forward_synchronize_batches`, `nvlink_forward_synchronize_iteration` | true / true / false / true | CUDA batch copies, nonblocking forwarding stream, per-batch completion and end-of-iteration synchronization. |
| `nvlink_forward_local_batch_sync_enabled` | false | Coordinate forward batch starts among local processes. Needs synchronized forwarding batches and a local synchronization directory. |
| `nvlink_forward_ping_pong_enabled`, `nvlink_forward_ping_pong_handoff_copy` | false / 1 | Alternate submitters on the same stream. One-based handoff API-call index. Ordered router batches, notifications and local batch synchronization are required. |
| `nvlink_forward_preparation_enabled`, `nvlink_forward_prepared_batch_chunks`, `nvlink_forward_prepared_queue_depth` | false / 0 / 2 | Prepare CPU descriptors ahead of copying. Choose positive batch chunks and queue depth ≥2. Requires ordered router batches, batch API and synchronized batches. Without notifications enable local iteration sync for safe destination reuse. Prepared chunks take precedence over threshold settings. |
| `nvlink_forward_ping_pong2_enabled` | false | Two prepared forwarding lanes. Requires preparation and excludes submit/epilogue mode. |
| `nvlink_forward_submit_epilogue_enabled`, `nvlink_forward_prepared_atomic_ring_enabled` | false / false | Split preparation, ordered submission and completion. Submit/epilogue needs preparation and both ping-pong modes off. Its optional atomic ring replaces the conventional prepared queue. |
| `nvlink_forward_completion_mode` | `stream_sync` | Optional `stream_query` completion requires submit/epilogue mode. |
| `nvlink_forward_local_staging_prepared_enabled` | false | Schedule local staging through the prepared submit/epilogue pipeline before remote batches. Requires local staging and submit/epilogue. |
| `nvlink_forward_destinations` | [] | Empty selects automatic CUDA IPC. For manual JSON entry, supply one non-source `gpu_index`, `cuda_device_id`, `buffer_addr` and sufficient `buffer_bytes` per other GPU. |
| `nvlink_return_completion_notifications_enabled` | false | Track local retained and incoming per-group returns. Requires submit/epilogue, forward notifications, routed execution, ≥2 GPUs and finite iterations. |
| `nvlink_return_local_batch_sync_enabled`, `nvlink_return_ready_gpu_publication_enabled` | false / false | Optional return batch-start rendezvous per group and HBM-ready prefix for GPU acquire polling. Both require return notifications. |

Coverage includes `test_prepared_forwarding`, `test_forward_events`, `test_submit_epilogue`, `test_stream_query`, `test_unified_local`, `test_ping_pong`, `test_ping_pong2`, `test_dual_device_forwarding`, `test_return_copies`, `test_return_notifications`, `test_return_batch_sync`, `test_return_ready_publication` and `test_rdma_returns`.

## Coordination and diagnostics

| Key | Default | Meaning |
| --- | --- | --- |
| `nvlink_forward_exchange_dir` | `/tmp/rdma_cpu_proxy_nvlink` | Node-local CUDA IPC descriptor exchange. |
| `local_iteration_sync_enabled`, `local_iteration_sync_dir`, `local_iteration_sync_run_id` | false / `/tmp/rdma_cpu_proxy_local_iteration_sync` / empty | Optional node-local iteration barrier and synchronization namespace. Empty uses a topology-derived run ID. Processes in the same launch must agree. |
| `rdma_bandwidth_summary_dir` | `/tmp/rdma_cpu_proxy_results` | Per-rank/GPU bandwidth summary destination. |
| `nvlink_forward_notification_log_enabled`, `nvlink_forward_notification_log_dir` | false / `/tmp/rdma_cpu_proxy_nvlink_notifications` | Write notification diagnostics. Needs active completion notifications. |
| `nvlink_forward_prepared_nvtx_enabled`, `nvlink_forward_log_batches`, `log_qp_reports`, `log_marker_wait_reports` | false / false / false / false | Prepared pipeline NVTX ranges, batch reports, per-QP reports and marker waits. Collect separately when comparing uninstrumented latency. |

## Native lifecycle and compatibility

The C ABI version and allocator request version are both 1. Install the allocator before initialization. Its device pointers are borrowed until after shutdown. The same-index remote NVLink receive alias must return the previously supplied RDMA receive pointer. In the default mode, routing and compact output layouts stay fixed. Dynamic embedders call `rdma_proxy_prepare_routing` after the previous generation's consumers finish, obtain valid index and table lengths with `rdma_proxy_get_routing_extents`, and bind those views before `rdma_proxy_prepare_iteration`. Transport allocations and their peer registrations survive generation changes. Exact signatures, capacity conventions, element types, and lifetime rules are in the [C header](../include/rdma_proxy_c_api.h).

Configure CPU scatter and optional packed FP32 weights after initialization and before iteration zero. Calling only the base C ABI setup publishes scatter metadata per batch before its gather readiness. The Python worker selects deferred publication by default when CPU scatter is enabled. Deferred and threaded modes require a single contiguous, 16-byte-aligned slab. The optional weight subarray starts at the next 16-byte boundary after the addresses. Deferred reconstruction follows final gather submission. Threaded packing consumes committed whole M batches outside notification locks. Both submit one H2D slab copy and event record. Before down GEMM the embedder waits for *submission* of that iteration's record and queues a stream wait while another host thread drives communication. Tests exercise address layout, weights, publication order and failures. [Scatter contracts](../include/rdma_proxy_c_api.h) · [Native tests](../tests/native/test_cpu_scatter.cpp)

Each prepare waits for pending early-local publication, checks outstanding rounds, resets gather readiness, synchronizes that reset and arms return/down progress. On return completion, the host prefix counts all local participants. Its optional HBM flag follows asynchronously. Reverse RDMA registers tiled GPU storage for incoming and outgoing peer-specific x3 rows. A system-release store advances the mapped source prefix. The native iteration waits for outgoing completion and incoming markers, including GPU visibility. Configure all return metadata collectively before iteration zero. The embedder records the current down completion event before arming return copies, joins GPU work before reset and shutdown, and retains storage through worker join, DMA retirement, and deregistration. [Proxy control](../native/src/proxy.cpp) · [Reverse RDMA](../native/src/rdma_return.cpp)

Shutdown cancels return batch synchronization and down-ready polling, stops/joins forwarding producers and the scatter worker, drains publication and returns, stops QP workers, deregisters memory on each NIC, then releases peer and iteration state. Operation errors cross the C ABI through a status and a thread-local diagnostic. `rdma_proxy_finish` in stepped mode requires the final configured iteration to have completed. [C ABI implementation](../native/src/c_api.cpp)

| Interface | Required format |
| --- | --- |
| Shared library | Output `rdma_cpu_proxy`, CMake VERSION 1.0.0, SOVERSION 1. |
| Local coordination | Shared memory local iteration sync v7, NVLink notification v6. Live incompatible layouts are rejected. |
| Router peer control | `router_x3_x4_v2` and `router_expert_metadata_v2`. The latter carries route weights with their exact IEEE FP32 bits when enabled. |
| QP setup and returns | Single-device legacy peer-info frame or `dual_rdma_v1`. Reverse channels use `rdma_return_v1` for a single NIC and `rdma_return_v2` for two. |
| Calibration agreement | Fingerprint of the `qp_calibration_v3` settings. |

Run matching configurations and builds on every participating process. Source definitions for the frame layouts and validation checks are in the [serializer](../native/src/protocol.cpp) and [connection code](../native/src/rdma_connection.cpp).
