# Development and validation

Run these blocks in Bash from the root of the meon checkout unless a block specifies another directory. CPU tests exercise routing, protocols and lifecycle logic. Establish GPU numerical correctness on each target architecture before comparing latency.

## Blackwell CUDA 13.0 quick start

This example uses an existing CUDA 13.0 Conda environment, eight visible GPUs per node, and two NICs per GPU. Activate that environment on each participating host, then run the shared setup below in each shell. Adjust the checkout paths and NIC names if they differ between hosts. For a new environment, follow the [CUDA 13 installation recipe](runtime.md#cuda-13-alternative).

```bash
export PROXY_ROOT=/sjtu-aisys-workspace/meon
export QUACK_ROOT=/sjtu-aisys-workspace/quack_for_communication_overlap
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export RDMA_PROXY_BUILD_DIR="$PROXY_ROOT/build-blackwell-cu130"
export QUACK_CACHE_DIR="$CONDA_PREFIX/var/cache/quack"  # optional

export RDMA_PROXY_GEMM_ARCH=blackwell
export CUDA_DEVICE_MAP="0,1,2,3,4,5,6,7"
export RDMA_DEVICE_MAP="mlx5_000+mlx5_001,mlx5_002+mlx5_003,mlx5_004+mlx5_005,mlx5_006+mlx5_007,mlx5_008+mlx5_009,mlx5_010+mlx5_011,mlx5_012+mlx5_013,mlx5_014+mlx5_015"
export RDMA_GID_INDEX=3
export RDMA_PORT=1
cd "$PROXY_ROOT"
```

Build once per build directory. CMake uses the CUDA toolkit and architecture 100 to compile the native library. The launchers use `RDMA_PROXY_BUILD_DIR` to find that library. The GPU map, NIC map, GID and peer addresses take effect at launch.

```bash
cmake -S "$PROXY_ROOT" -B "$RDMA_PROXY_BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=100 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DRDMA_PROXY_REQUIRE_CUDA=ON -DRDMA_PROXY_REQUIRE_VERBS=ON
cmake --build "$RDMA_PROXY_BUILD_DIR" -j4
```

**Single-node test.** Run on either host. Rendezvous uses loopback and the launcher initializes the local NICs. This checks projections and the enabled return/combine stages over three iterations.

```bash
MASTER_ADDR=127.0.0.1 bash scripts/run_single_node_torchrun.sh \
  --num_tokens=1024 --num_iterations=3 --check-projections
```

### Weighted single-node test on Blackwell

After the CUDA 13.0 setup and build above, run this on one eight-GPU Blackwell host. Each GPU supplies 20,480 tokens of width 7,168. The up and down GEMMs use 256 × 512 tiles, and the four `--down-bulk-*` options tune the additive `bulk_rows_reduce` store. The worker selects the Blackwell down kernel with ping-pong disabled.

```bash
RDMA_PROXY_GEMM_ARCH=blackwell MASTER_ADDR=127.0.0.1 \
  bash scripts/run_single_node_torchrun.sh \
    --num_iterations=3 --num_tokens=20480 --token_dimension=7168 \
    --nvlink_forward_prepared_batch_chunks=64 \
    --grouped-gemm-m-tile=256 --grouped-gemm-n-tile=512 \
    --grouped-gemm-output-dimension=1536 --grouped-gemm-max-swizzle-size=1 \
    --down-tile-m=256 --down-tile-n=512 --down-cluster-m=2 --down-max-swizzle-size=2 \
    --down-bulk-smem-padding=16 --down-bulk-smem-cols=128 \
    --down-bulk-smem-buffers=1 --down-bulk-store-warps=4 \
    --down-sm-count=140 --intra-node-combine-ctas=6 --intra-node-combine-relax-sm-guard \
    --weighted-contributions --cpu-scatter-preparation --cpu-scatter-publication=threaded \
    --router_local_input_early_ready_enabled=true \
    --nvlink_forward_notification_scheduling_mode=complete_round \
    --down-output-reset=end-of-iteration --nvlink_return_local_batch_sync_enabled=true \
    --completion_timeout_ms=600000 --return-token-timeout=600 --check-projections
```

The relaxed guard requires at least **146 visible SMs per GPU** for the down budget of 140 and six combine CTAs. The launcher checks returns and combine results. The added `--check-projections` compares the up and down results with numerical references on all three iterations. [Parameter descriptions](features.md#reading-a-single-node-command)

### Two-node test on Blackwell

Start the following commands concurrently, one in each host's prepared shell. This example assumes node 0 uses `10.51.1.3` for both torchrun rendezvous and proxy traffic. Both ranks use the same `MASTER_ADDR` and `MASTER_PORT`. Each rank sets `PEER_HOST` to the other host's reachable proxy address.

On **node 0**, replace `<node-1-proxy-IP>` with node 1's address.

```bash
export MASTER_ADDR=10.51.1.3 MASTER_PORT=29500 PEER_HOST="<node-1-proxy-IP>"
bash scripts/run_node0_torchrun.sh --num_tokens=1024 --num_iterations=3 --check-projections
```

On **node 1**, the peer is node 0.

```bash
export MASTER_ADDR=10.51.1.3 MASTER_PORT=29500 PEER_HOST=10.51.1.3
bash scripts/run_node1_torchrun.sh --num_tokens=1024 --num_iterations=3 --check-projections
```

Check that each launcher exits successfully and every worker reports `Projection numerical checks passed: iteration=2`. The two-node profile also prints `RDMA return transport check: PASSED`.

## Hopper H100 quick start

Activate the Hopper Conda environment on each participating host and start in the meon checkout. This example uses CUDA 12.9 with matching CUDA-enabled PyTorch and the external QuACK checkout alongside meon. The map assigns one bonded NIC to each of eight GPUs in the order used by the shipped Hopper launchers. Check the local names with `nvidia-smi topo -m` and `ibv_devinfo`, and adjust the map, port and GID for that host if needed.

```bash
export PROXY_ROOT="$PWD"
export QUACK_ROOT="$(dirname "$PROXY_ROOT")/quack_for_communication_overlap"
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export QUACK_CACHE_DIR="$CONDA_PREFIX/var/cache/quack"  # optional
export RDMA_PROXY_BUILD_DIR="$PROXY_ROOT/build-hopper"
export RDMA_PROXY_GEMM_ARCH=hopper
export CUDA_DEVICE_MAP="0,1,2,3,4,5,6,7"
export RDMA_DEVICE_MAP="mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_7,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5"
export RDMA_GID_INDEX=3
export RDMA_PORT=1
```

**Build for SM90.** Build once per build directory and keep `RDMA_PROXY_BUILD_DIR` exported for the launchers. If your Hopper environment already uses CUDA 13.0 and cu130 PyTorch, set `CUDA_HOME=/usr/local/cuda-13.0` before exporting `PATH` above and choose a fresh directory such as `$PROXY_ROOT/build-hopper-cu130`. The H100 build target stays `90` with either toolkit.

```bash
cmake -S "$PROXY_ROOT" -B "$RDMA_PROXY_BUILD_DIR" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared \
  -DRDMA_PROXY_REQUIRE_CUDA=ON -DRDMA_PROXY_REQUIRE_VERBS=ON
cmake --build "$RDMA_PROXY_BUILD_DIR" -j4
```

**Single node.** Run on either host. The launcher uses the local NIC map and loopback rendezvous. It ignores `PEER_HOST`.

```bash
MASTER_ADDR=127.0.0.1 bash scripts/run_single_node_torchrun.sh \
  --num_tokens=1024 --num_iterations=3 --check-projections
```

### Weighted single-node test on Hopper

After the SM90 build and environment setup above, run this from the meon root on one eight-GPU Hopper host. It uses 20,480 tokens per GPU, 7,168 features per token and a SwiGLU output width of 1,536. The command makes the Hopper geometry and ping-pong choice explicit and checks projection results on each of three iterations.

```bash
RDMA_PROXY_GEMM_ARCH=hopper MASTER_ADDR=127.0.0.1 \
  bash scripts/run_single_node_torchrun.sh \
    --num_iterations=3 --num_tokens=20480 --token_dimension=7168 \
    --nvlink_forward_prepared_batch_chunks=64 \
    --grouped-gemm-m-tile=128 --grouped-gemm-n-tile=128 \
    --grouped-gemm-output-dimension=1536 --grouped-gemm-max-swizzle-size=1 \
    --down-tile-m=128 --down-tile-n=128 --down-cluster-m=1 \
    --down-max-swizzle-size=2 --down-sm-count=74 --down-pingpong \
    --intra-node-combine-ctas=6 --weighted-contributions \
    --cpu-scatter-preparation --cpu-scatter-publication=threaded \
    --router_local_input_early_ready_enabled=true \
    --nvlink_forward_notification_scheduling_mode=complete_round \
    --down-output-reset=end-of-iteration --nvlink_return_local_batch_sync_enabled=true \
    --completion_timeout_ms=600000 --return-token-timeout=600 --check-projections
```

The launcher enables expert waves, the wave barrier, readiness publication and the additive `bulk_rows_reduce` epilogue. Expert waves alone determine the order of work. Enabling the wave barrier on Hopper requires ping-pong, and the worker selects it automatically if you omit `--down-pingpong`. QuACK rejects the four Blackwell shared-memory and store-warp tuning flags on Hopper. With 74 down SMs, down cluster M=1 and six combine CTAs, the standard guard requires at least 80 visible SMs. [Parameter descriptions](features.md#reading-a-single-node-command)

### Per-iteration routing

Build the native library from the same checkout on both hosts and use the matching Python worker. In the two-node profile, one persistent C++ metadata worker per GPU generates a fresh top-k table, exchanges per-source records and stages the input into the runtime's local receive slice. Each same-index GPU pair has one additional bidirectional RC connection for metadata, separate from the configured payload QPs. The single-node path uses local CUDA IPC. The feature is off by default.

Run these commands concurrently from the repository root on the corresponding Hopper hosts, with the environment and network settings in the two-node quick start. A small workload and projection checks exercise changed routes through both projections, return and combine.

```bash
# Node 0
bash scripts/run_node0_torchrun.sh --router_dynamic_routing_enabled=true --num_iterations=5 --num_tokens=1024 --check-projections

# Node 1
bash scripts/run_node1_torchrun.sh --router_dynamic_routing_enabled=true --num_iterations=5 --num_tokens=1024 --check-projections
```

Single-node coverage uses `bash scripts/run_single_node_torchrun.sh --router_dynamic_routing_enabled=true --num_iterations=5 --num_tokens=1024 --check-projections`. Add `--weighted-contributions` on all participating ranks for FP32 route-weight coverage. For a workload comparison use the same shape and configuration on both runs, then toggle the routing flag. Changing just an iteration count or seed in a fixed-routing run does not exercise metadata replacement.

The source seed for zero-based iteration `i` is `router_seed + 0x9e3779b97f4a7c15 * (i * world_size + global_source_rank)`, modulo 2^64. Runtime setup owns the metadata arenas and peer connections. The metadata worker launches the local gather on a separate CUDA stream so it can overlap exchange. It waits for both before releasing the payload phase. The current source lists, inbound masks and valid lengths feed that iteration’s consumers. Forwarding publishes GEMM descriptors as payloads become visible. The standalone exchange benchmark ends before installing the runtime's forward plan, GEMM views and return mappings, so its end-to-end time has a different endpoint.

The torchrun worker also accepts `--dynamic-routing`. At the end of a concurrent run, each rank prints `routing node=… gpu=… iteration=… elapsed_us=…`. This is host wall time from entry to return of `rdma_proxy_prepare_routing`, including generation, exchange, staging and native installation. Python then binds the new compute views. Iteration zero includes first-use allocation of route and scheduler storage. The separate `pipeline iteration … elapsed_ms=…` line includes payload and compute. With `--dynamic-routing` alone, the runtime uses the shared raw CUDA routing kernels and its native metadata worker.

The CPU tests for up-output capacity exercise growth, live views and retrying
an allocation under memory pressure. Host verification used a Debug build on
an Apple M1 Max with CPU PyTorch 1.13.1.

```bash
OMP_NUM_THREADS=1 /Users/zhuhao/opt/anaconda3/bin/python3 -B -m unittest discover -s tests/python -v
ctest --test-dir build/mock -R '^test_(dynamic_gather|pooled_gather|gemm_geometry)$' --output-on-failure
```

For the H800 timing check, rerun the same two-node dynamic routing workload.
On each GPU, compare the steady-state interval from the final routing-index
upload to the kernel that zeros the combine row counts. Look for a
`cudaMalloc` when the required up-output rows exceed the reserved capacity.

To check threaded CPU scatter staging on the host, run these commands from the meon root.

```bash
cmake -S . -B build/mock -DCMAKE_BUILD_TYPE=Debug
cmake --build build/mock --target test_cpu_scatter -j4
ctest --test-dir build/mock -R '^test_cpu_scatter$' --output-on-failure
python -m unittest discover -s tests/python -p test_cpu_scatter_publication.py -v
```

The dynamic scatter test counts distinct worker threads that pack a batch across routing changes. It covers a growing route, an empty route, switching to per-batch publication and back, and replacing a route while its worker is still packing. On an Apple M1 Max with Apple Clang 14, the Debug host test passed ten consecutive runs under ThreadSanitizer. To repeat it, use a separate build directory.

```bash
cmake -S . -B build/mock-tsan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=thread -fno-omit-frame-pointer" \
  -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=thread"
cmake --build build/mock-tsan --target test_cpu_scatter -j4
TSAN_OPTIONS=halt_on_error=1 ctest --test-dir build/mock-tsan \
  -R '^test_cpu_scatter$' --repeat until-fail:10 --output-on-failure
```

On Hopper, run `ctest --test-dir "$RDMA_PROXY_BUILD_DIR" -R '^test_cpu_scatter_cuda$' --output-on-failure` after building the CUDA tests. Rebuild the shared library on both hosts and repeat the dynamic numerical check above with `--cpu-scatter-preparation --cpu-scatter-publication=threaded`. In Nsight Systems, follow the scatter metadata H2D back to its CUDA API call. Each rank should show one `rdma-scatter` thread issuing that upload across all nonempty iterations. Look for the pinned scatter allocation during setup and one upload per nonempty iteration, sized by that iteration's route count. To measure the effect on latency in the benchmark metadata run, compare steady-state `install_us` and the interval from the end of the local staging gather to the start of up-GEMM at the same workload. [Scatter staging and upload sizes](runtime.md#optional-cpu-scatter-preparation)

To verify direct binding of threaded scatter addresses, use the updated
`prepare_grouped_inputs` from `QUACK_ROOT` on both nodes and compare dynamic
iterations after warmup. In the before/after profiles, check that the
device-to-device copy of the provisional address table disappears during
down-input setup and that the helper still uploads the final table before
down-GEMM. For N-tile-major output, the provisional copy was 16 bytes per
expert assignment. Compare the down-input setup interval at the same token
count, and run the weighted projection check above to verify the packed weights.

The host checks for this change used an Apple M1 Max, macOS 15.6.1, CPU
PyTorch 1.13.1 and the Debug native build. From the meon checkout:

```bash
cmake --build build/mock --target test_cpu_scatter -j4
ctest --test-dir build/mock -R '^test_cpu_scatter$' --output-on-failure
OMP_NUM_THREADS=1 /Users/zhuhao/opt/anaconda3/bin/python3 -B -m unittest discover -s tests/python -v
(cd "$QUACK_ROOT" && PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /Users/zhuhao/opt/anaconda3/bin/python3 -B -m pytest -q -p no:cacheprovider --noconftest tests/test_wave_ready_prefix.py)
```

The QuACK numerical and CUDA graph replay check on Hopper is
`python -m pytest -q tests/test_gemm_gather_table.py -k wave_ready_publication_tail_and_replay`
from `QUACK_ROOT`. It and the two-node timing comparison require a CUDA host.

For the dynamic intra-node combine table, `test_c_api` compares the native output
with the mask query, and `test_benchmark_routing` covers incoming remote masks
and shrinking and empty domains. The Python regression in `test_dynamic_routing.py`
checks table and buffer reuse across generations. On the two Hopper hosts, use the
matching updated worker and Release shared library. Run the dynamic numerical check
above with `--check-intra-node-combine --benchmark-metadata-exchange` on both nodes.
For the timing run, use `--no-check-intra-node-combine`. With 20,480 tokens and
eight GPUs per node, look for one 2,621,568-byte host-to-device copy per rank and
iteration for this table. [Table layout](runtime.md#intra-node-combine-concurrent-with-down-gemm)

For the dynamic **inter-node** combine table, the same native tests check each
original token's packed row, including a route where the outgoing and incoming
remote token IDs differ. `test_inter_node_combine.py` checks host and device
buffer reuse, a single copy per generation, and the independent numerical
reference. Run it on Hopper with
`RDMA_PROXY_TEST_CUDA=1 python -m unittest discover -s tests/python -p test_inter_node_combine.py -v`.
On both H800 hosts, rebuild the shared library in the launcher's build directory.
Use `--dynamic-routing --benchmark-metadata-exchange --check-inter-node-combine`
for a numerical run. The setup log should show `row_builder=native`. With
`--no-check-inter-node-combine` in the timing run, look for one 163,856-byte
host-to-device upload per rank per iteration at 20,480 tokens and two nodes.
The earlier path followed that upload with a device-to-device copy of the same size.
[Table layout](runtime.md#inter-node-combine-in-original-token-order)

On an Apple M1 Max with CPU PyTorch 1.13.1 set to one thread, the median
`InterNodeCombine.refresh_routes` time fell from 6.20 ms at `f4103bd` to
0.062 ms with the native builder. The paired replay used 120 fresh mock routes
after three warmups, two nodes, eight GPUs per node, 20,480 tokens per GPU,
128 experts and top-8 routing. It alternated the execution order of the two
implementations on each installed route and compared every table entry. The
timings include the Python call and the copy into a CPU destination tensor.
GPU upload time and the gather-to-up-GEMM gap require a two-node hardware run.

**Validation.** On macOS 15.6.1 arm64 with Apple Clang 14, CPU-only PyTorch 1.13.1 and a Debug build, the runtime native suite passed 44 tests with four CUDA tests skipped. The Python suite passed 86 tests with eight skipped. The shared metadata checks passed all four tests. The CPU exchange benchmark passed its native codec test and all 38 Python tests. The GPU exchange benchmark passed its ten host tests. The exact host commands are in [Mac or CPU-only development](#mac-or-cpu-only-development). CUDA compilation, the numerical checks with QuACK, and two-node GPU/RDMA transport and timing still require the Hopper hosts. The production record-layout test runs with `ctest --test-dir "$RDMA_PROXY_BUILD_DIR" -R test_iteration_metadata_cuda --output-on-failure` on a CUDA host. Run the dynamic combine tests there with `RDMA_PROXY_TEST_CUDA=1 python -m unittest discover -s tests/python -p test_dynamic_routing.py -v`.

### Benchmark metadata backend

From the meon root, build on both hosts using the Hopper environment above. The CUDA and verbs build produces `librouter_metadata_exchange2.so` alongside `librdma_cpu_proxy.so`. The worker imports CuTe DSL and CUDA Python from the same environment used for the GPU exchange benchmark. An alternate build of the matching benchmark library can be selected with `MEON_BENCHMARK_METADATA_LIBRARY`.

```bash
cmake -S . -B build-hopper -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DRDMA_PROXY_REQUIRE_CUDA=ON -DRDMA_PROXY_REQUIRE_VERBS=ON
cmake --build build-hopper --target rdma_cpu_proxy_shared -j4
```

Use `--benchmark-metadata-exchange` on all ranks. The receiver requires eight visible, peer-accessible GPUs with native P2P atomics per host and CUDA ordinal equal to local rank. The meon launcher supplies 128 experts, top-8, unweighted routes and a NIC for each GPU. Each same-index pair across hosts opens one metadata RC QP on its primary NIC during setup. A persistent C++ worker per GPU drives the benchmark exchange. Gloo coordinates setup, arrival and buffer reuse. The expert records and remote x3/x4 travel over the benchmark's GPU IPC and RDMA paths.

First run a small numerical check, starting the two commands concurrently in the corresponding host shells.

```bash
# Node 0
bash scripts/run_node0_torchrun.sh --benchmark-metadata-exchange --num_tokens=1024 --num_iterations=3 --check-projections
# Node 1
bash scripts/run_node1_torchrun.sh --benchmark-metadata-exchange --num_tokens=1024 --num_iterations=3 --check-projections
```

For the 10,240-token experiment, use the following arguments on both hosts. The payload, up/down GEMMs and return/combine stages still execute. The four switches disable the launcher's downstream numerical checks for the timing run.

```bash
args=(--benchmark-metadata-exchange --num_tokens=10240 --num_iterations=220
      --no-check-return-token-results --no-check-intra-node-combine
      --no-check-rdma-return-token-results --no-check-inter-node-combine)
# Node 0
bash scripts/run_node0_torchrun.sh "${args[@]}"
# Node 1, with the same args array
bash scripts/run_node1_torchrun.sh "${args[@]}"
```

Each iteration generates fresh affine R using meon's per-rank seed sequence. The backend runs the benchmark's captured metadata graph with its native launch barrier, direct x3/x4 output, overlapping receiver and local staging after the receiver. The gather writes directly into meon's own input slice. With matching exchange and payload libraries, each local `A_idx` points into its source's record in the destination GPU's exchange buffer. The NIC writes the eight remote index prefixes to registered GPU slots in addition to delivering their host records. Each slot keeps the original global source identity. The CPU borrows the 16 pinned lists for validation, scheduling and scatter preparation. After exchange completion, meon clears any padding through the common logical length in one GPU kernel and rebuilds the forwarding plan. Python binds the current compute views before up-GEMM submission. Older library combinations use the per-source H2D upload path.

Before the next exchange, each rank retires the previous route after its payload and assignment-query readers finish, then enters the Gloo reuse barrier. Once all ranks leave that barrier, the receiver may reuse its outgoing buffer and peers may write the next incoming lists. At the end of the run the worker shuts down the payload proxy before freeing the benchmark's storage.

For an A/B timing run with the benchmark's generated routes, build the updated `rdma_cpu_proxy_shared` on both hosts. Run the same seed and launch arguments twice, adding `--no-benchmark-metadata-validation` to both node commands for the second run. The default checks every outgoing expert ID, checks each x3/x4 slice for duplicates, valid masks and row ordering, and checks every incoming `A_idx` for bounds and ordering within its expert. The switch skips those installation scans. The benchmark producer must supply valid values and sorted per-expert lists because QuACK reads the indices and the CPU uses their order to decide which rows are ready.

Both modes check x3 token bounds before those IDs can become RDMA addresses. They also check source identities, pointers, offsets and storage capacities, observe exchange completion, and construct receive chunks and forwarding tables. The four `--no-check-*` result switches are independent. For a numerical comparison, run the small two-node `--check-projections` example above in both modes. For a profile comparison, examine steady `install_us`, local gather end to the first index operation, and local gather end to up-GEMM start.

Look for `a_idx_storage=exchange_device` on the per-rank timing line to confirm that the incoming indices use the direct GPU path. If any rank's libraries lack the required interfaces, the worker selects `a_idx_storage=host_upload` across the group. The separate fields `source_route=receiver source_route_storage=borrowed` describe this GPU's outgoing route. The receiver writes its outgoing node offsets, x3, x4 and top-8 R into pinned memory before the local staging gather. Meon borrows x3, x4 and R for the iteration and, by default, checks the routing values there. Older payload builds that support only the call-duration host handoff report `source_route=receiver source_route_storage=copied` with the updated worker. If the receiver handoff is unavailable, the worker reports `source_route=device_readback` and reads the arrays from the GPU during installation.

| Per-rank field | Host wall interval |
| --- | --- |
| `routing … elapsed_us` | Entry to benchmark preparation through completion of local route installation, including generation, retirement, alignment and exchange. |
| `exchange_staging_us` | Aligned submission through benchmark wait completion, including the seven kernels, GPU exchange, RDMA and local gather. With `source_route=receiver`, this includes writing the outgoing snapshot. With direct GPU indices, it also includes the remote index mirrors and their GPU visibility. Pass `--device-indices` to the standalone two-node benchmark to include those writes there. |
| `install_us` | Benchmark wait return through completion of route installation on the calling thread. On the concurrent early-local path, peer transport preparation runs later on the coordinator. |
| `retire_us` | Local time to release the previous iteration's borrowed route data before entering the reuse barrier. Zero for the first iteration and for snapshot installs. |
| `r_ready_to_up_submit_us` | Fresh R completed through host return from up-GEMM submission, including rank alignment before exchange, installation and compute setup. Available for compute iterations that submit work. Use a GPU trace to measure the actual kernel start. |
| `generation_us`, `reuse_barrier_us` | Route generation and the Gloo barrier before exchange submission, respectively. |
| `release_barrier_us` | Reports zero for compatibility with existing log readers. Older workers measured the post-install Gloo barrier here. |

To compare with the standalone timing table, discard meon iterations 0–19. For each of the remaining 200 iterations take the largest `exchange_staging_us` across all 16 ranks, then compute median and p95 over those 200 samples. Shape, token width, GPU/NIC placement, and rank skew affect the measured values. Inspect `install_us` and `r_ready_to_up_submit_us` when investigating the generator-to-GEMM gap. The first iteration also includes compute setup.

Use the same Python worker and benchmark backend on all 16 ranks so their Gloo collectives occur in the same order. For comparison with H800 profile 13, update both hosts and repeat the same launch with `--no-benchmark-metadata-validation`. Compare the steady-state intervals from the end of `clear_index_tails_kernel` to the first combine-row fill and from local gather completion to up-GEMM start. Also compare the time to the first gather-table ready-row publication. Ranks arriving early can spend some of their time waiting at native payload startup.

For a comparison with H800 profile 14, rebuild `rdma_cpu_proxy_shared` on both hosts and repeat that launch with the same arguments. Confirm `iteration_control=persistent_tcp` for all eight peer pairs. Use the `phase=iteration_start` logs from both nodes to compare `peer_wait_us` and `local_wait_us` per GPU and iteration. In Nsight Systems, measure up-GEMM start to the first gather-table ready-row publication on the same GPU, excluding warmup. That interval also includes work outside the native barrier.

The Debug native test `test_iteration_control` runs over IPv6 loopback without GPUs or verbs. It exercises repeated start/done markers, arrival skew in both directions, mixed-version peers, interrupted exchanges, mismatched markers and one-shot control traffic on the same port.

Local verification used an Apple M1 Max running macOS 15.6.1 with CUDA and verbs disabled.

```bash
cmake -S . -B build/mock -DCMAKE_BUILD_TYPE=Debug
cmake --build build/mock -j4
ctest --test-dir build/mock --output-on-failure
ctest --test-dir build/mock -R '^test_iteration_control$' --repeat until-fail:10 --output-on-failure
```

The two-node launchers enable the QuACK down gather table, expert waves and ready publication. Rebuild the meon shared library and set `QUACK_ROOT` to the updated checkout on both hosts. For the profile 15 arguments, look for `Native down-wave schedule reserved` during initialization. With an older native library or QuACK helper, the worker uses the previous schedule path.

For iterations 1–4, identify the final up-table ready-row publication on the native publication stream and the subsequent packed down-schedule H2D. The CPU scatter upload has a separate event and can also appear on that stream. For the example counts below the schedule transfer is 385,936 bytes. Compare local staging gather completion to up-GEMM start (profile 15 median **6.68 ms**), gather completion to the first ready-row publication (**7.30 ms**), and full iteration time (**61.26 ms**). Also check whether any remaining setup extends past up completion. A two-node Hopper run is needed to measure the change.

For comparison with report 17, rebuild the shared library and install the updated meon worker on both hosts. Use the same threaded scatter and dynamic routing flags. Check for `Native return maps reserved` on each participating rank. For iterations 1–4 of the 20,480-token two-node configuration, the publication stream should show one **2,785,424-byte** return-map H2D after the last up-gather ready write. Identify it separately from the down-wave and scatter uploads on that stream. Report 17’s median local gather completion to up-GEMM start was **5.24 ms**, its up-to-down handoff **21.1 µs**, and its complete iteration **59.60 ms**. Also compare when the first combine actually starts. Run a smaller two-node comparison with projection and combine checks enabled before collecting the timing run.

The Python worker resolves fixed QuACK down-GEMM options once per run. Each refreshed route gets a private copy with its current row count, and each nonempty route is validated before use. To measure the effect, repeat the report 18 launch with the updated worker on both Hopper nodes. Across the eight Node 0 GPUs in iterations 1–4, report 18’s median local-gather-end to up-GEMM-start interval was **2.15 ms**. The **~1.04 ms** host interval from route installation to entry into `cudaDeviceSynchronize` included runner construction, tensor binding and worker handoffs. [Down projection runner](../python/rdma_proxy_worker.py)

The CPU regression covers empty routes, changing row counts and isolation of QuACK’s per-iteration arguments. On an Apple M1 Max with macOS 15.6.1 and CPU PyTorch 1.13.1, `OMP_NUM_THREADS=1 PYTHONPATH=tests/python python -B -m unittest -v test_cpu_scatter_publication` passed all 12 tests.

To compare with report 19, rebuild `rdma_cpu_proxy_shared` and update the worker on both Hopper nodes. Add `--router_local_input_early_ready_enabled=true` to both commands in the small two-node numerical run above, with projection and return/combine checks enabled. Then repeat the original timing command and confirm `a_idx_storage=exchange_device`. For iterations 1–4 compare local staging gather end to the first positive ready-row publication, up-GEMM start, and full iteration time. Report 19's median gather-to-up-GEMM interval on Node 0 was 1.56 ms. For an attribution run, enable `--nvlink_forward_prepared_nvtx_enabled=true` and inspect `router/local_input_early_ready` on the notification receiver and `router/prepare_peer_transport` on the coordinator. An expert with too few ready local rows waits for another source.

For comparison with H800 report 20, rebuild the shared library and update the Python
worker on both nodes. Use the same launch arguments. Confirm `early_up_launch=1`
in each nonempty steady iteration's `benchmark_metadata` line. This path requires
Hopper multi-buffer up-GEMM, direct benchmark indices and early local readiness.
When running down and combine, it also requires deferred QuACK down setup,
native threaded scatter and wave scheduling, and asynchronous combine maps.
Empty routes, iteration zero, older libraries and synchronous CUDA launches (`CUDA_LAUNCH_BLOCKING=1`) use the ordinary launch path.

On node 0, match each GPU's local staging graph end to its up-GEMM start and
first positive ready-prefix publication. Report 20 medians across 32 observations
(iterations 1–4, eight GPUs) were **1.379 ms** to kernel start and **0.924 ms**
to first publication. Also compare full iteration time. The timing line measures
host submission and cannot establish the first useful GPU tile time.
Before collecting timings, run the same path on both hosts with 1,024 tokens
and at least three iterations, enable `--check-projections` and retain the
launcher's return and combine checks.

Host verification used an Apple M1 Max, macOS 15.6.1, CPU PyTorch 1.13.1 and
Python 3.9.12. The two-node H800 numerical run and performance comparison are
outstanding. The CUDA reset-event regression runs in
`test_borrowed_routing_upload_cuda`. The Hopper composition suite exercises
the empty-range cancellation descriptors with QuACK.

```bash
cmake -S . -B build/mock -DCMAKE_BUILD_TYPE=Debug
cmake --build build/mock -j6
ctest --test-dir build/mock --output-on-failure
ctest --test-dir build/exchange-host --output-on-failure
OMP_NUM_THREADS=1 /Users/zhuhao/opt/anaconda3/bin/python3 -B -m unittest discover -s tests/python -q
```

For comparison with H800 report 21, rebuild the shared library and update the
Python worker on both nodes. The same launch flags select concurrent local
publication. Check for `concurrent_local_publication=1` on each nonempty steady
iteration. Run the small numerical check above, then repeat the 20,480-token
profile. Across node 0's eight GPUs in iterations 1–4, report 21's median
gather-end to first positive ready-prefix update was **1.212 ms**, gather-end
to up-kernel start was **0.742 ms**, and full iteration time was **56.29 ms**.
Measure all three intervals to distinguish earlier usable work from earlier
kernel residency. For an attribution trace, enable
`--nvlink_forward_prepared_nvtx_enabled=true` and inspect
`router/local_input_early_ready` on the notification receiver.
An expert with insufficient local rows will still need peer
input before its first batch can publish.

Host checks for concurrent publication used the Debug mock build, Python
3.9.12 and CPU PyTorch 1.13.1 on an Apple M1 Max running macOS 15.6.1.
The native regression waits for a positive local prefix before starting return
maps or preparing peer payloads. The Python regression holds the QuACK submit
call open while the coordinator requests publication and exercises cancellation
on both threads. Two-node H800 numerical results and timings remain outstanding.

```bash
cmake --build build/mock -j6
ctest --test-dir build/mock --output-on-failure
ctest --test-dir build/mock -R '^test_benchmark_routing$' --repeat until-fail:10 --output-on-failure
ctest --test-dir build/exchange-host --output-on-failure
OMP_NUM_THREADS=1 PYTHONPATH=tests/python /Users/zhuhao/opt/anaconda3/bin/python3 -B -m unittest -q test_benchmark_routing_worker
OMP_NUM_THREADS=1 /Users/zhuhao/opt/anaconda3/bin/python3 -B -m unittest discover -s tests/python -q
ROUTER_METADATA_BENCH_LIBRARY="$PWD/build/exchange-cpu/librouter_metadata_bench.dylib" PYTHONDONTWRITEBYTECODE=1 /Users/zhuhao/opt/anaconda3/bin/python3 -m unittest discover -s benchmarks/router_metadata_exchange -p test_benchmark.py -v
```

Local verification of this change used an Apple M1 Max, macOS 15.6.1 and CPU PyTorch 1.13.1. The Debug native regressions cover a delayed rank, an empty local source, multiple epochs and shutdown after preparation. The two-node Hopper numerical and timing checks above require CUDA, NVLink and RDMA.

```bash
cmake --build build/mock -j4
ctest --test-dir build/mock --output-on-failure
ctest --test-dir build/mock -R '^test_benchmark_routing$' --repeat until-fail:10 --output-on-failure
cmake --build build/exchange-host -j4
ctest --test-dir build/exchange-host --output-on-failure
OMP_NUM_THREADS=1 /Users/zhuhao/opt/anaconda3/bin/python3 -B -m unittest discover -s tests/python -q
```

The async scatter and return-map change passed 47 native tests and 102 Python tests on an Apple M1 Max with macOS 15.6.1 and CPU PyTorch 1.13.1. Six native tests and eight Python tests were skipped. The two-node Hopper numerical and timing runs remain outstanding. Reproduce the local checks from the meon checkout.

```bash
OMP_NUM_THREADS=1 /Users/zhuhao/opt/anaconda3/bin/python3 -B -m unittest discover -s tests/python
cmake -S . -B build/mock -DCMAKE_BUILD_TYPE=Debug
cmake --build build/mock -j4
ctest --test-dir build/mock --output-on-failure
ctest --test-dir build/mock -R '^(test_down_wave_schedule|test_cpu_scatter)$' --repeat until-fail:10 --output-on-failure
ctest --test-dir build/mock -R '^(test_async_return_maps|test_cpu_scatter|test_rdma_returns|test_c_api|test_benchmark_routing)$' --repeat until-fail:5 --output-on-failure
ctest --test-dir build/exchange-host --output-on-failure
ctest --test-dir build/exchange-cpu --output-on-failure
ROUTER_METADATA_BENCH_LIBRARY="$PWD/build/exchange-cpu/librouter_metadata_bench.dylib" PYTHONDONTWRITEBYTECODE=1 /Users/zhuhao/opt/anaconda3/bin/python3 -m unittest discover -s benchmarks/router_metadata_exchange -p test_benchmark.py -v
(cd ../quack_for_communication_overlap && PYTHONDONTWRITEBYTECODE=1 /Users/zhuhao/opt/anaconda3/bin/python3 -m pytest --noconftest -q tests/test_wave_ready_prefix.py -p no:cacheprovider)
```

The native lifecycle regression also passed with ThreadSanitizer on this host. The builder regression passed with AddressSanitizer and UndefinedBehaviorSanitizer.

```bash
cmake -S . -B /private/tmp/meon-down-wave-thread-sanitizer -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-fsanitize=thread
cmake --build /private/tmp/meon-down-wave-thread-sanitizer --target test_cpu_scatter -j6
ctest --test-dir /private/tmp/meon-down-wave-thread-sanitizer -R '^test_cpu_scatter$' --output-on-failure
cmake -S . -B build/return-maps-tsan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-fsanitize=thread
cmake --build build/return-maps-tsan --target test_async_return_maps test_cpu_scatter test_benchmark_routing test_c_api test_rdma_returns -j4
ctest --test-dir build/return-maps-tsan -R '^(test_async_return_maps|test_cpu_scatter|test_rdma_returns|test_c_api|test_benchmark_routing)$' --output-on-failure
c++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -g -O1 -fsanitize=undefined,address -fno-omit-frame-pointer -I native/include native/src/down_wave_schedule.cpp tests/native/test_down_wave_schedule.cpp -o /private/tmp/meon_down_wave_asan
/private/tmp/meon_down_wave_asan
```

On a single Hopper GPU with `QUACK_ROOT` set and the CUDA Debug build available, these regressions exercise packed-slab reuse, GPU event ordering, weighted down output and final combine.

```bash
ctest --test-dir "${RDMA_PROXY_BUILD_DIR:-build-hopper}" -R '^(test_async_return_maps|test_cpu_scatter)_cuda$' --output-on-failure
PYTHONPATH=tests/python python -m unittest -v \
  test_worker_composition.CompositionCudaTests.test_hopper_reuses_packed_down_wave_schedule
```

These commands check the schedule on CPU and time construction for node 0, GPU 0 of the 20,480-token H800 run.

```bash
cd "$QUACK_ROOT"
python -m pytest --noconftest -q tests/test_wave_ready_prefix.py
python -m timeit -n 100 -r 5 \
  -s 'import torch; torch.set_num_threads(1); from run.wave_ready import build_expert_wave_schedule' \
  -s 'counts = [20394, 20430, 20592, 20392, 20444, 20405, 20549, 20491]' \
  'build_expert_wave_schedule(counts, cluster_rows=192, num_groups=28, group_size=2, wave_work_ids=124)'
```

On macOS 15.6.1 arm64 with Python 3.9.12 and CPU PyTorch 1.13.1, a one-thread replay alternated the original QuACK builder at `05e4efc` with the packed builder for 100 measured calls each. Mean host construction time was 20.03 ms and 0.96 ms, respectively. All 24,024 descriptors and 388 ready-prefix values matched. A new two-node H800 profile is needed to measure the resulting change in gather-to-up-GEMM latency.

Host coverage includes a three-epoch, eight-rank mock handoff that checks empty expert lists, distinct local and remote masks, payload destinations, and reuse of staging and metadata buffers. Both launchers passed a 10,240-token worker dry run. The two-node numerical check above, CUDA graph replay, GPUDirect registration and the 300–400 µs timing target still need a hardware run.

The shared expert-list handoff was checked on an arm64 Mac running macOS 15.6.1 with Debug native builds and CPU PyTorch. The mock regression checks pointer identity, rejection of early reuse, final shutdown, and zero padding when the valid list shrinks from four rows to two and then zero. The worker tests cover retirement before the reuse barrier and interrupted shutdown. The host checks used these commands from the meon root.

```bash
cmake --build build/mock -j4 && ctest --test-dir build/mock --output-on-failure
OMP_NUM_THREADS=1 /Users/zhuhao/opt/anaconda3/bin/python3 -B -m unittest discover -s tests/python
cmake --build build/exchange-host -j4 && ctest --test-dir build/exchange-host --output-on-failure
cmake --build build/exchange-cpu -j4 && ctest --test-dir build/exchange-cpu --output-on-failure
ROUTER_METADATA_BENCH_LIBRARY="$PWD/build/exchange-cpu/librouter_metadata_bench.dylib" PYTHONDONTWRITEBYTECODE=1 \
  /Users/zhuhao/opt/anaconda3/bin/python3 -m unittest discover -s benchmarks/router_metadata_exchange -p test_benchmark.py -v
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s metadata -p test_metadata_cute.py
```

The single-GPU check below exercises pinned-memory uploads and reads the result on an independent CUDA stream. The two-node H800 run above is needed to measure the change in `install_us`, `retire_us`, and the interval from local gather completion to up-GEMM.

```bash
cmake --build "$RDMA_PROXY_BUILD_DIR" --target rdma_cpu_proxy_shared test_dynamic_gather -j4
ctest --test-dir "$RDMA_PROXY_BUILD_DIR" -R '^test_borrowed_routing_upload_cuda$' --output-on-failure
```

The outgoing route handoff was checked on macOS 15.6.1 arm64 with the Debug native and CPU Python suites above. The eight-rank mock checks that the borrowed arrays alias the receiver's storage, forwarding and return rows follow the current iteration, and the assignment query returns the original top-8 choices and unit weights. It exercises reuse after retirement, rejected queries for a retired route, and shutdown after a partial install. The older host installer still permits immediate reuse after its call. The two lifetime-sensitive native tests also ran under UBSan.

```bash
cmake -S . -B /private/tmp/meon-iteration-route-views-ubsan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=undefined -fno-sanitize-recover=all'
cmake --build /private/tmp/meon-iteration-route-views-ubsan --target test_benchmark_routing test_c_api -j4
ctest --test-dir /private/tmp/meon-iteration-route-views-ubsan -R '^test_(benchmark_routing|c_api)$' --output-on-failure
```

The following CUDA regression runs on one host with eight peer-accessible GPUs. It compares the receiver's mapped output against all four source arrays as soon as the receiver event completes.

```bash
cmake -S "$PROXY_ROOT/benchmarks/router_metadata_exchange2" -B "$RDMA_PROXY_BUILD_DIR/receiver-tests" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build "$RDMA_PROXY_BUILD_DIR/receiver-tests" --target test_receiver_kernel -j4
ctest --test-dir "$RDMA_PROXY_BUILD_DIR/receiver-tests" -R '^receiver_kernel_cuda$' --output-on-failure
```

For the two-node H800 profile, rebuild `rdma_cpu_proxy_shared` on both hosts and run the same launch arguments as profile 11. First run the small two-node projection check above, then compare steady-state `exchange_staging_us`, `install_us` and the interval from local gather completion to up-GEMM. In the direct path the GPU trace shows at most one `clear_index_tails_kernel` in place of the 16 H2Ds and their padding memsets. Each rank receives one extra RDMA write per nonempty remote list, carrying its live indices. With the default settings, route validation precedes padding and compute.

To check the device and host lists independently against generated routes, run this GPU exchange suite concurrently on the two nodes with `NODE_RANK=0` or `1` and `RME2_LIBRARY` pointing at the rebuilt exchange library. Its cycling routes exercise empty lists, skew and changing lengths.

```bash
bash benchmarks/router_metadata_exchange2/run_two_node_gpu_checks.sh "$NODE_RANK" \
  --native-metadata-launch --receiver-kernel --receiver-overlap-exchange --device-indices
```

The direct-index change was checked on macOS 15.6.1 arm64 with Debug native builds, Python 3.9.12 and CPU PyTorch 1.13.1. The host suites above passed 44 native tests, 96 Python tests and all 10 exchange host tests. Five native and eight Python tests skipped. They check both relay directions, source identity, live-prefix preservation, zero padding on empty and shrinking routes, and retirement after a failed install. The CPU exchange suite and CuTe reference tests also passed. The following commands also checked the two native lifecycle tests under UBSan.

```bash
cmake -S . -B /private/tmp/meon-direct-indices-ubsan -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=undefined -fno-sanitize-recover=all'
cmake --build /private/tmp/meon-direct-indices-ubsan --target test_benchmark_routing test_c_api -j4
ctest --test-dir /private/tmp/meon-direct-indices-ubsan -R '^test_(benchmark_routing|c_api)$' --output-on-failure
```

The validation switch was tested on an Apple M1 Max with macOS 15.6.1, Python 3.9.12 and CPU PyTorch 1.13.1. The eight-rank mock exercises both validation settings across routing generations and checks forwarding, return rows, and acceptance or rejection of individual malformed fields. The ordinary dynamic-routing test checks that an unsorted expert list is rejected even when benchmark validation is disabled.

```bash
cmake -S . -B build/mock -DCMAKE_BUILD_TYPE=Debug
cmake --build build/mock -j4
ctest --test-dir build/mock --output-on-failure
OMP_NUM_THREADS=1 /Users/zhuhao/opt/anaconda3/bin/python3 -B -m unittest discover -s tests/python
ctest --test-dir build/exchange-host --output-on-failure
```

The post-install barrier removal passed the same host suites on the M1 Max. The eight-rank mock delayed one install until another rank reached native preparation, then checked the iteration's forwarding and returned rows. It also passed five consecutive runs with `ctest --test-dir build/mock -R '^test_benchmark_routing$' --repeat until-fail:5 --output-on-failure`.

The CUDA build, two-node numerical and transport checks, and Nsight timing comparison require the Hopper hosts.

#### Diagnosing the host delay in H800 report 22

Update `python/rdma_proxy_worker.py` on both nodes and repeat the report 22
command. For each successful steady iteration with `concurrent_local_publication=1`,
the worker prints one `local_publication_timing` line in the end-of-run summary.
Each value is host wall time in microseconds from the same rank's monotonic clock.

| Field | Interval |
| --- | --- |
| `r_ready_to_submit_us` | Benchmark R readiness through main-thread submission of the coordinator task. Shares the origin of `r_ready_to_up_submit_us`. |
| `handoff_us` | Immediately before executor submission through entry into the preparation callable on the coordinator. Includes executor dispatch and time until Python can run there. |
| `setup_context_us` | Coordinator entry through selection of its CUDA device and setup stream. |
| `expert_counts_us` | Reading current expert counts and computing the routed token total. |
| `down_wave_start_us` | Starting the asynchronous down-wave job and binding its returned views. |
| `down_runner_us` | Constructing the refreshed down projection runner. |
| `scatter_register_us` | Configuring the current scatter destinations and starting native lookup. |
| `submit_to_request_us` | Total from executor submission to entry into the native local-publication call. Includes all preceding phases starting at handoff. |
| `request_call_us` | Wall time inside that native call, which prepares gather inputs and posts the local-ready request. The receiver publishes eligible descriptors asynchronously. |

For iterations 1–4, compare `handoff_us` with the individual setup phases on
the same GPU. Optional phases appear only when executed. The timings include
time a thread spends descheduled within each interval. Use the GPU timeline
to measure gather-end to first positive ready prefix (report 22 median
**1.323 ms** across node 0's 32 samples). The worker collects host timestamps
without adding CUDA synchronization or printing during submission.

Host verification used an Apple M1 Max on macOS 15.6.1 with Python 3.9.12 and
CPU PyTorch 1.13.1. The instrumented two-node H800 run is outstanding.

```bash
OMP_NUM_THREADS=1 PYTHONPATH=tests/python /Users/zhuhao/opt/anaconda3/bin/python3 -B -m unittest -q test_benchmark_routing_worker
OMP_NUM_THREADS=1 /Users/zhuhao/opt/anaconda3/bin/python3 -B -m unittest discover -s tests/python -q
```

### Two-node test on Hopper

Start these commands concurrently. The addresses below are the checked-in Hopper profile. If your hosts have different addresses, use node 0's rendezvous address for `MASTER_ADDR` in both shells and the opposite host's proxy address for each `PEER_HOST`.

Run on **node 0**.

```bash
export MASTER_ADDR=28.59.80.49 MASTER_PORT=29500 PEER_HOST=29.162.224.226
bash scripts/run_node0_torchrun.sh --num_tokens=1024 --num_iterations=3 --check-projections
```

Run on **node 1**.

```bash
export MASTER_ADDR=28.59.80.49 MASTER_PORT=29500 PEER_HOST=28.59.80.49
bash scripts/run_node1_torchrun.sh --num_tokens=1024 --num_iterations=3 --check-projections
```

Each launcher should exit successfully, and every worker should print `Projection numerical checks passed: iteration=2`. The two-node run also checks the reverse RDMA path and prints `RDMA return transport check: PASSED`.

## Pooled up GEMM on H200

Use eight H200s per host. Start in the Meon checkout with the GPU/NIC maps and CUDA 12.9 environment from the [Hopper quick start](#hopper-h100-quick-start). The paired runs use one build and QuACK checkout. The pooled reference was inspected at QuACK revision `05e4efc4bb70fed084b3e2f8488f3ed316b1838b`. Run this setup on each host.

```bash
set -eo pipefail
export RDMA_PROXY_GEMM_ARCH=hopper NPROC_PER_NODE=8
export RDMA_PROXY_BUILD_DIR="$PWD/build/h200-release"
native_flags=(-DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME"
  -DCMAKE_CUDA_ARCHITECTURES=90 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
  -DRDMA_PROXY_REQUIRE_CUDA=ON -DRDMA_PROXY_REQUIRE_VERBS=ON)
cmake -S . -B build/h200-debug -DCMAKE_BUILD_TYPE=Debug "${native_flags[@]}"
cmake --build build/h200-debug --target test_cpu_scatter -j4
CUDA_VISIBLE_DEVICES=0 build/h200-debug/test_cpu_scatter --cuda
cmake -S . -B "$RDMA_PROXY_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release "${native_flags[@]}"
cmake --build "$RDMA_PROXY_BUILD_DIR" --target rdma_cpu_proxy_shared -j4
git -C "$QUACK_ROOT" rev-parse HEAD
CUDA_VISIBLE_DEVICES=0 RDMA_PROXY_TEST_CUDA=1 python -m unittest discover -s tests/python -v
```

Define this function in Bash on each participating host. It checks weighted SwiGLU with the launcher's GPU scatter, repeats with threaded CPU scatter, then measures the default multi-buffer and pooled modes in alternating order. The first pooled timing job also replays its last completed batch sequence after all consumers join. Each replay uses the exact same pool allocation, payload, weights, packed output, ready counter and descriptor order observed on that GPU. Both layouts must reproduce the valid live rows. Eighteen CUDA event samples per layout measure the two kernels in a graph with the order multi, pooled, pooled, multi.

```bash
h200_pooled_runs() {
  local launcher="$1" tag="$2" output_dir="$3"
  local common=(--num_experts=128 --top_k=8 --router_seed=1234 --token_dimension=4096
    --grouped-gemm-activation=swiglu --grouped-gemm-output-dimension=2048
    --grouped-gemm-m-tile=128 --grouped-gemm-n-tile=128 --grouped-gemm-cluster-m=2
    --grouped-gemm-max-swizzle-size=4 --up-sm-count=96
    --down-tile-m=128 --down-tile-n=128 --down-cluster-m=1 --down-max-swizzle-size=1
    --down-pingpong --down-sm-count=74 --intra-node-combine-ctas=6
    --weighted-contributions --nvlink_forward_notification_scheduling_mode=per_notification
    --router_local_input_early_ready_enabled=true --nvlink_forward_prepared_batch_chunks=8
    --completion_timeout_ms=600000 --return-token-timeout=600)
  local checks_off=(--no-check-projections --no-check-return-token-results
    --no-check-rdma-return-token-results --no-check-intra-node-combine --no-check-inter-node-combine)
  local replay=() layout sample=0
  mkdir -p "$output_dir"
  timeout 3600 bash "$launcher" "${common[@]}" --pooled-up-gemm --num_tokens=1024 \
    --num_iterations=3 --check-projections --local_iteration_sync_run_id="${tag}_gpu" \
    2>&1 | tee "$output_dir/check-gpu.log" || return
  timeout 3600 bash "$launcher" "${common[@]}" --pooled-up-gemm --num_tokens=1024 \
    --num_iterations=3 --check-projections --cpu-scatter-preparation \
    --cpu-scatter-publication=threaded --local_iteration_sync_run_id="${tag}_cpu" \
    2>&1 | tee "$output_dir/check-cpu.log" || return
  for layout in --no-pooled-up-gemm --pooled-up-gemm --pooled-up-gemm --no-pooled-up-gemm; do
    sample=$((sample + 1))
    replay=()
    if [[ $sample == 2 ]]; then replay+=(--benchmark-ready-up-gemm); fi
    timeout 7200 bash "$launcher" "${common[@]}" "${checks_off[@]}" "$layout" "${replay[@]}" \
      --num_tokens=20480 --num_iterations=30 --local_iteration_sync_run_id="${tag}_perf_$sample" \
      2>&1 | tee "$output_dir/perf-$sample.log" || return
  done
}
```

**One node, eight logical sources.** Run in the prepared shell.

```bash
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 h200_pooled_runs scripts/run_single_node_torchrun.sh \
  h200_single_01 results/h200-single/node0
```

**Two nodes, sixteen logical sources.** Start both commands concurrently in the prepared shells. The examples use the shipped Hopper IPs. Set the rendezvous to node 0's reachable address and each proxy peer to the opposite host. Use a fresh shared tag when retrying a failed job.

```bash
# Node 0
MASTER_ADDR=28.59.80.49 MASTER_PORT=29500 PEER_HOST=29.162.224.226 \
  h200_pooled_runs scripts/run_node0_torchrun.sh h200_pair_01 results/h200-pair/node0
```

```bash
# Node 1
MASTER_ADDR=28.59.80.49 MASTER_PORT=29500 PEER_HOST=28.59.80.49 \
  h200_pooled_runs scripts/run_node1_torchrun.sh h200_pair_01 results/h200-pair/node1
```

Check every rank for three projection and combine passes. The two-node profile also checks reverse RDMA. In `perf-2.log`, each `Ready up-GEMM` line reports the per-GPU median time with all descriptors ready. For live runs, the up line (`iteration I: elapsed_ms=… effective_throughput=…`) measures the first publication through up completion, including stalls on later arrivals. The `pipeline iteration I` line measures the complete host interval through returns, combine and reuse barriers. For each job and metric take the largest rank time at each iteration across **both** node logs for a two-node job. Compare medians over iterations 3–29 and retain the spread across the two runs per layout. Live jobs have matching routing and resource budgets. Arrival order can change between jobs.

Collect the per-node log directories under the paths below and summarize each topology.

```bash
python - <<'PY'
from collections import defaultdict
from pathlib import Path
import re, statistics
for path, nodes in (("results/h200-single", 1), ("results/h200-pair", 2)):
    medians = defaultdict(list)
    for run in range(1, 5):
        logs = [Path(path, f"node{n}", f"perf-{run}.log").read_text() for n in range(nodes)]
        layout = "multi" if run in (1, 4) else "pooled"
        for stage, pattern in (
            ("up", r"\biteration (\d+): elapsed_ms=([\d.]+) effective_throughput="),
            ("pipeline", r"\bpipeline iteration (\d+): elapsed_ms=([\d.]+)"),
        ):
            ranks = defaultdict(list)
            for log in logs:
                for i, ms in re.findall(pattern, log):
                    ranks[int(i)].append(float(ms))
            assert all(len(ranks[i]) == 8 * nodes for i in range(30)), (path, run, stage)
            medians[stage, layout].append(statistics.median(max(ranks[i]) for i in range(3, 30)))
    for stage in ("up", "pipeline"):
        multi, pooled = medians[stage, "multi"], medians[stage, "pooled"]
        print(path, stage, "multi ms", multi, "pooled ms", pooled,
              "speedup", statistics.median(multi) / statistics.median(pooled))
PY
```

The reported standalone H200 result exceeded 700 TFLOP/s with eight logical sources and reached about 670–680 TFLOP/s with sixteen or thirty-two. Meon's streaming speedup still needs measurement on H200.

Local validation used an Apple M1 Max MacBook Pro running macOS 15.6.1, with Python 3.9.19 and CPU PyTorch 2.0.0. Debug CTest passed 40 tests and skipped three CUDA tests. The Python suite reported 70 tests, five CUDA skips and zero failures. These runs covered mock transport, scheduler publication, allocator aliases, weighted scatter and CPU numerical references. The H200 commands above cover the outstanding CUDA IPC, GPUDirect registration, live QuACK and timing checks.


```bash
cmake -S . -B build/mock -DCMAKE_BUILD_TYPE=Debug
cmake --build build/mock -j4
ctest --test-dir build/mock --output-on-failure
./build/mock/rdma_cpu_proxy --config config/example_config.json --cpu_affinity=none
/Users/zhuhao/opt/anaconda3/envs/d2l/bin/python -m unittest discover -s tests/python -v
```

## Mac or CPU-only development

CMake 3.20+, a C++17 compiler and Python 3.9+ suffice for the commands below. The project also builds with CMake 3.18/3.19. With those versions run `(cd build/mock && ctest --output-on-failure)` and use the same pattern in each benchmark build directory. A Python environment containing CPU PyTorch additionally exercises tensor arithmetic and corruption detection. Keep assertions enabled. Socket tests need permission to bind loopback ports.

```bash
export MEON_ROOT="$PWD"
cmake -S . -B build/mock -DCMAKE_BUILD_TYPE=Debug
cmake --build build/mock -j4
ctest --test-dir build/mock --output-on-failure
./build/mock/rdma_cpu_proxy --config config/example_config.json --cpu_affinity=none
python3 -m unittest discover -s tests/python -v
# Validation on the arm64 Mac used the CPU-PyTorch Python environment.
# OMP_NUM_THREADS=1 /Users/zhuhao/opt/anaconda3/bin/python3 -B -m unittest discover -s tests/python -v
python3 -m unittest discover -s metadata -p test_metadata_cute.py -v
for f in scripts/*.sh metadata/*.sh benchmarks/*/*.sh; do bash -n "$f" || exit 1; done
TORCHRUN_BIN=echo bash scripts/run_single_node_torchrun.sh
LOCAL_RANK=0 LOCAL_WORLD_SIZE=2 RANK=0 WORLD_SIZE=2 GROUP_RANK=0 \
  python3 python/rdma_proxy_worker.py --library=/unused.so --cuda-device-map=0,1 \
  --rdma-device-map=nic0,nic1 --config config/single_node_config.json --communication-only --dry-run
```

The two benchmark projects have independent build files. On macOS the CPU exchange library has the `.dylib` suffix. Its Python suite runs multi-process loopback payload and reference checks.

```bash
cmake -S benchmarks/router_metadata_exchange -B build/exchange-cpu -DCMAKE_BUILD_TYPE=Debug
cmake --build build/exchange-cpu -j4
ctest --test-dir build/exchange-cpu --output-on-failure
suffix=so; if [[ "$(uname -s)" = Darwin ]]; then suffix=dylib; fi
ROUTER_METADATA_BENCH_LIBRARY="$PWD/build/exchange-cpu/librouter_metadata_bench.$suffix" \
  python3 -m unittest discover -s benchmarks/router_metadata_exchange -p test_benchmark.py -v
cmake -S benchmarks/router_metadata_exchange2 -B build/exchange-host \
  -DRME2_HOST_TESTS_ONLY=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/exchange-host -j4
ctest --test-dir build/exchange-host --output-on-failure
```

Inspect skips. CUDA scatter copies, CUDA events, return publication, actual CuTe/QuACK kernels, GPUDirect registration and NVLink/IPC need NVIDIA hardware. The host-only exchange build has no GPU transport.

## Linux NVIDIA hosts — build and preflight

Run on **each** host with the same meon revision and external QuACK checkout. Use a fresh Python 3.12 environment for the CUDA 12.9 installation below. An existing matching installation can start at `pip check`. The documented QuACK reference is revision `ccc12c585a453a4d96c333677aa58714d117c496` with CUTLASS DSL 4.7.0. A full toolkit with nvcc, a compatible driver/compiler, rdma-core/libibverbs development files, Ninja and working GPU-memory registration are required. The checkout's `run/` helpers must be present. The separate CUDA 13.0 environment uses the cu130 PyTorch wheel, CUDA bindings 13.0.3 and the QuACK `cu13` extra. Its installation commands are in the [CUDA 13 recipe](runtime.md#cuda-13-alternative).

```bash
set -eo pipefail
export MEON_ROOT="$PWD"
export QUACK_ROOT="${QUACK_ROOT:-$MEON_ROOT/../quack_for_communication_overlap}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export PATH="$CUDA_HOME/bin:$PATH"
git -C "$QUACK_ROOT" rev-parse HEAD
"$CUDA_HOME/bin/nvcc" --version
nvidia-smi
python -m pip install 'torch==2.13.0' --index-url https://download.pytorch.org/whl/cu129
python -m pip install 'cuda-python==12.9.4' 'cuda-bindings==12.9.4' ninja
python -m pip install -e "$QUACK_ROOT[dev]"
python -m pip check
python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import os, quack, torch
assert torch.cuda.is_available()
assert Path(quack.__file__).resolve().is_relative_to(Path(os.environ["QUACK_ROOT"]).resolve())
print("GPU", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("torch", torch.__version__, "CUDA", torch.version.cuda, "CuTe", version("nvidia-cutlass-dsl"))
PY
```

Select Hopper SM90 or Blackwell SM100-family geometry based on the actual device. B300 reporting 10.3 uses generic SM100 for native transport. Set the PyTorch JIT target separately. Use fresh build and cache locations when switching SDK/architecture or Python environments.

```bash
case "$(python -c 'import torch; print(torch.cuda.get_device_capability(0)[0])')" in
  9)  export RDMA_PROXY_GEMM_ARCH=hopper TORCH_CUDA_ARCH_LIST=9.0; arch=90 ;;
  10) export RDMA_PROXY_GEMM_ARCH=blackwell TORCH_CUDA_ARCH_LIST=10.0; arch=100 ;;
  *) echo "This integrated GEMM profile requires Hopper or Blackwell" >&2; exit 1 ;;
esac
cuda_tag="$(python -c 'import torch; print(torch.version.cuda)')"
export TORCH_EXTENSIONS_DIR="$MEON_ROOT/build/torch-extensions-$arch-cu$cuda_tag"
export RDMA_PROXY_BUILD_DIR="$MEON_ROOT/build/$arch-cu$cuda_tag-debug"
native_cmake=(-DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_HOME"
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared -DCMAKE_CUDA_ARCHITECTURES="$arch"
  -DRDMA_PROXY_REQUIRE_CUDA=ON -DRDMA_PROXY_REQUIRE_VERBS=ON)
cmake -S "$MEON_ROOT" -B "$RDMA_PROXY_BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug "${native_cmake[@]}"
cmake --build "$RDMA_PROXY_BUILD_DIR" -j4
ctest --test-dir "$RDMA_PROXY_BUILD_DIR" --output-on-failure
readelf -d "$RDMA_PROXY_BUILD_DIR/librdma_cpu_proxy.so" | grep libcudart
PYTHONPATH="$MEON_ROOT/python" python - <<'PY'
import os
from pathlib import Path
from rdma_proxy_worker import import_torch_for_proxy, load_proxy_library
torch = import_torch_for_proxy(0)
load_proxy_library(Path(os.environ["RDMA_PROXY_BUILD_DIR"]) / "librdma_cpu_proxy.so",
                   torch_cuda_version=torch.version.cuda)
print("Embedded runtime preflight passed for CUDA", torch.version.cuda)
PY
```

Check CTest output for unexpected device skips. The worker preflight loads PyTorch first and checks the mapped CUDA runtime before native initialization. A mixed major or duplicate runtime must be resolved by matching the toolkit, PyTorch and shared-library build.

## One GPU — kernel loading and numerical composition

The root CMake build compiles the raw routing kernel into the proxy. The PyTorch driver JIT-loads its wrapper and the same CUDA source from `metadata/`. CuTe compiles a separate implementation. Check all three entry points. The following raw benchmark/JIT/CuTe commands can run on one GPU while modeling a larger logical routing topology. Use cache directories specific to the active toolkit, architecture and environment for an initial cold load.

```bash
cd "$MEON_ROOT"
export NVCC="$CUDA_HOME/bin/nvcc" CUDA_ARCH="sm_$arch"
export RDMA_METADATA_BUILD_DIR="$MEON_ROOT/build/raw-metadata-$arch-cu$cuda_tag"
bash metadata/build_rdma_metadata_benchmark.sh
"$RDMA_METADATA_BUILD_DIR/benchmark_rdma_metadata" --device 0 --tokens 1024 --warmup 1 --iters 3
CUDA_VISIBLE_DEVICES=0 python metadata/run.py --tokens 1024 --top-k 8 --experts 128 \
  --experts-per-gpu 8 --node-mask-sort --gpus-per-node 8 --local-gpu-id 0 --check --warmup 1 --iters 3
CUDA_VISIBLE_DEVICES=0 timeout 3600 python metadata/benchmark_metadata_cute.py \
  --device 0 --check-suite --compare-cuda --warmup 1 --iters 3
CUDA_VISIBLE_DEVICES=0 RDMA_PROXY_TEST_CUDA=1 timeout 7200 \
  python -m unittest discover -s tests/python -v
(cd "$QUACK_ROOT" && CUDA_VISIBLE_DEVICES=0 timeout 1800 \
  python -m pytest -q tests/test_gemm_route_weights.py -x)
if [[ "$arch" = 100 ]]; then
  (cd "$QUACK_ROOT" && CUDA_VISIBLE_DEVICES=0 timeout 3600 \
    python -m pytest -q run/test_blackwell_gemm_communication.py run/test_proxy_gemm_integration.py -x)
fi
```

Look for execution of `CompositionCudaTests` and the opt-in combine tests. These check poisoned and reused buffers, partial multi-source clusters, weight alignment and application, the actual up/down/compact scatter plus consumer composition, and output reset. The SM-cap test checks the requested clustered launch under different up budgets. The actual native metadata publication and IPC/RDMA timeline require the distributed runs below. [Projection checks](../python/projection_validation.py) · [Composition tests](../tests/python/test_worker_composition.py)

## One node, then two nodes — transfer and complete pipeline

Use eight peer-accessible GPUs for the shipped profiles. Set site overrides in the remote shell. Inspect `nvidia-smi topo -m` and `ibv_devinfo` to select the NIC entry for each CUDA index, the port and the GID. `RDMA_DEVICE_MAP` accepts one device or `nicA+nicB` at each position. The Hopper launcher can supply the original single-NIC map. On Blackwell provide the actual NIC entries explicitly. Each pair of peer NICs must have fabric connectivity. Set site-specific SM counts/geometry through `site_options` below when the profile exceeds the device budget.

```bash
nvidia-smi topo -m
ibv_devinfo
export CUDA_DEVICE_MAP="${CUDA_DEVICE_MAP:-0,1,2,3,4,5,6,7}"
if [[ "$arch" = 100 ]]; then
  : "${RDMA_DEVICE_MAP:?supply one NIC entry per local GPU in CUDA map order}"
fi
export RDMA_DEVICE_MAP="${RDMA_DEVICE_MAP:-mlx5_bond_1,mlx5_bond_4,mlx5_bond_3,mlx5_bond_2,mlx5_bond_7,mlx5_bond_6,mlx5_bond_8,mlx5_bond_5}"
export RDMA_PORT="${RDMA_PORT:-1}"
export RDMA_GID_INDEX="${RDMA_GID_INDEX:?set the valid local GID index, or -1 for no global addressing}"
site_options=()
```

Run the function below in Bash on every participating host. It first exercises the communication-only path, then checks an unweighted integrated baseline and two repeated weighted/reset variants. Each integrated job retains the launcher's same-node byte checks, combine references and (on two nodes) reverse-RDMA digest checks. Apply the same stage options on both nodes.

```bash
check_pipeline() {
  local launcher="$1" tag="$2"
  local common=(--num_tokens=1024 --num_iterations=8 --completion_timeout_ms=600000 --return-token-timeout=600)
  local checked=(--check-projections --weighted-contributions --cpu-scatter-preparation)
  timeout 1800 bash "$launcher" "${site_options[@]}" "${common[@]}" --communication-only \
    --local_iteration_sync_run_id="${tag}_comm" || return
  timeout 3600 bash "$launcher" "${site_options[@]}" "${common[@]}" --check-projections \
    --local_iteration_sync_run_id="${tag}_gpu" || return
  local policy
  for policy in deferred threaded; do
    timeout 3600 bash "$launcher" "${site_options[@]}" "${common[@]}" "${checked[@]}" \
      --cpu-scatter-publication="$policy" --down-output-reset=end-of-iteration \
      --nvlink_forward_notification_scheduling_mode=complete_round \
      --router_local_input_early_ready_enabled=true --nvlink_return_local_batch_sync_enabled=true \
      --local_iteration_sync_run_id="${tag}_${policy}" || return
  done
}
cd "$MEON_ROOT"
MASTER_ADDR=127.0.0.1 check_pipeline scripts/run_single_node_torchrun.sh "single_$(date +%s)"
```

For two nodes, set `NODE_RANK=0` on the rendezvous host and `NODE_RANK=1` on its peer. Export the **same** reachable `MASTER_ADDR`, free `MASTER_PORT` and fresh `RUN_TAG` on both. Export each host's `PEER_HOST` as the **other** host's proxy address. Both nodes use equal process counts, options and compatible libraries. Start the following block concurrently in both shells, with the function above already defined. Bootstrap TCP listens at 18515 + local rank, with router metadata at 19515 + local rank by default.

```bash
: "${NODE_RANK:?set to 0 or 1}" "${MASTER_ADDR:?node 0 address}" "${PEER_HOST:?other node address}"
: "${MASTER_PORT:?free rendezvous port}" "${RUN_TAG:?fresh shared launch identifier}"
export MASTER_ADDR MASTER_PORT PEER_HOST
launcher="scripts/run_node${NODE_RANK}_torchrun.sh"
TORCHRUN_BIN=echo bash "$launcher"
check_pipeline "$launcher" "$RUN_TAG"
```

A successful checked iteration should report projection and intra/inter-node numerical success and valid return comparisons. Examine all ranks for missing progress, residency errors, prefix mismatches or failures at shutdown. Repeat on Hopper and Blackwell. For per-entry publication, per-batch CPU preparation, empty/unequal rounds and separate instrumented profiles, use the [extended validation matrix](runtime.md#remote-validation-and-performance-sequence). After a failed launch stop both jobs and use a fresh rendezvous port and run identifier on retry.

## Performance regression against the previous implementation

Use the same hardware, operator-confirmed maps, toolkit, compiler flags, co-tenancy, QuACK revision/cache settings, batch policy and SM budgets on both builds. Supply absolute paths to a retained working baseline checkout/bundle containing its own `scripts/` and its matching **Release** library directory on each remote host. Use the same environment and launch defaults that passed the correctness comparison. Keep timing uninstrumented and run the same paired sequence simultaneously on both nodes. Enable `set -eo pipefail` so a worker failure terminates the comparison.

```bash
: "${BASELINE_ROOT:?prior project root with scripts and config}" "${BASELINE_BUILD:?prior matching Release build}"
export PERF_BUILD="$MEON_ROOT/build/$arch-cu$cuda_tag-release"
cmake -S "$MEON_ROOT" -B "$PERF_BUILD" -DCMAKE_BUILD_TYPE=Release "${native_cmake[@]}"
cmake --build "$PERF_BUILD" --target rdma_cpu_proxy_shared -j4
perf_dir="$MEON_ROOT/results/perf-$RUN_TAG/node$NODE_RANK"
mkdir -p "$perf_dir"
checks_off=(--no-check-projections --no-check-return-token-results --no-check-rdma-return-token-results
  --no-check-intra-node-combine --no-check-inter-node-combine)
sample=0
for variant in baseline migrated migrated baseline; do
  if [[ "$variant" = baseline ]]; then root="$BASELINE_ROOT"; build="$BASELINE_BUILD"
  else root="$MEON_ROOT"; build="$PERF_BUILD"; fi
  sample=$((sample + 1))
  (cd "$perf_dir" && RDMA_PROXY_BUILD_DIR="$build" timeout 3600 \
    bash "$root/$launcher" "${site_options[@]}" "${checks_off[@]}" \
    --num_tokens=20480 --num_iterations=30 --completion_timeout_ms=600000 --return-token-timeout=600 \
    --nvlink_forward_notification_log_enabled=false --nvlink_forward_log_batches=false \
    --nvlink_forward_prepared_nvtx_enabled=false --nvlink_forward_notification_scheduling_diagnostics=false \
    --local_iteration_sync_run_id="${RUN_TAG}_perf_${sample}") >"$perf_dir/$sample-$variant.log" 2>&1
done
```

Run this summary on each node. It reports the median of the **slowest local worker** per warm iteration. Preserve both nodes' results and raw logs. Use the maximum over all participating workers for each iteration when combining logs across nodes. Compare both baseline/migrated pairs and run-to-run spread against your existing regression threshold. The host pipeline interval includes preparation, completion joins and reset. Down-GEMM event timing and outgoing RDMA bandwidth can help attribute a difference.

```bash
python - "$perf_dir" "${NPROC_PER_NODE:-8}" <<'PY'
from collections import defaultdict
from pathlib import Path
import re, statistics, sys
for log in sorted(Path(sys.argv[1]).glob("*.log")):
    times = defaultdict(list)
    for i, ms in re.findall(r"pipeline iteration (\d+): elapsed_ms=([\d.]+)", log.read_text()):
        times[int(i)].append(float(ms))
    assert set(times) == set(range(30)), f"incomplete run: {log}"
    assert all(len(rows) == int(sys.argv[2]) for rows in times.values()), f"incomplete ranks: {log}"
    print(log.name, "warm pipeline ms", statistics.median(max(times[i]) for i in range(3, 30)))
PY
```

For an all-ready combine measurement independent of NIC traffic, use `python python/benchmark_intra_node_combine.py --quack-root "$QUACK_ROOT"`. To measure exchange separately, follow its own build and test procedure. The GPU exchange benchmark has dedicated single-node and paired two-node correctness launchers. [CPU exchange](../benchmarks/router_metadata_exchange/README.md) · [GPU exchange](../benchmarks/router_metadata_exchange2/README.md) · [Raw/CuTe metadata benchmarks](../metadata/README.md)
