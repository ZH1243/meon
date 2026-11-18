# Development and validation

Run these blocks in Bash from the root of the meon checkout unless a block specifies another directory. CPU tests exercise routing, protocols and lifecycle logic. Establish GPU numerical correctness on each target architecture before comparing latency.

## Mac or CPU-only development

CMake 3.20+, a C++17 compiler and Python 3.9+ suffice for the commands below. The project also builds with CMake 3.18/3.19. With those versions run `(cd build/mock && ctest --output-on-failure)` and use the same pattern in each benchmark build directory. A Python environment containing CPU PyTorch additionally exercises tensor arithmetic and corruption detection. Keep assertions enabled. Socket tests need permission to bind loopback ports.

```bash
export MEON_ROOT="$PWD"
cmake -S . -B build/mock -DCMAKE_BUILD_TYPE=Debug
cmake --build build/mock -j4
ctest --test-dir build/mock --output-on-failure
./build/mock/rdma_cpu_proxy --config config/example_config.json --cpu_affinity=none
python3 -m unittest discover -s tests/python -v
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
