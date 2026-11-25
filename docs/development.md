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

The torchrun worker also accepts `--dynamic-routing`. At the end of a concurrent run, each rank prints `routing node=… gpu=… iteration=… elapsed_us=…`. This is host wall time from entry to return of `rdma_proxy_prepare_routing`, including generation, exchange, staging and native installation. Python then binds the new compute views. Iteration zero includes first-use allocation of route and scheduler storage. The separate `pipeline iteration … elapsed_ms=…` line includes payload and compute. The runtime uses the shared raw CUDA routing kernels and metadata QP. The CUDA graph and receiver-kernel options remain specific to the exchange benchmark.

**Validation.** On macOS 15.6.1 arm64 with Apple Clang 14, CPU-only PyTorch 1.13.1 and a Debug build, the runtime native suite passed 43 tests with four CUDA tests skipped. The Python suite passed 69 tests with eight skipped. The shared metadata checks passed all four tests. The CPU exchange benchmark passed its native codec test and 38 Python tests. The GPU exchange benchmark passed its ten host tests. The exact host commands are in [Mac or CPU-only development](#mac-or-cpu-only-development). CUDA compilation, the numerical checks with QuACK, and two-node GPU/RDMA transport and timing still require the Hopper hosts. The production record-layout test runs with `ctest --test-dir "$RDMA_PROXY_BUILD_DIR" -R test_iteration_metadata_cuda --output-on-failure` on a CUDA host. Run the dynamic combine tests there with `RDMA_PROXY_TEST_CUDA=1 python -m unittest discover -s tests/python -p test_dynamic_routing.py -v`.

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
# PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 /Users/zhuhao/opt/anaconda3/bin/python3 -m unittest discover -s tests/python -p 'test_*.py'
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
