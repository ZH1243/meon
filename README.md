# meon

meon is a CPU-driven RDMA and NVLink runtime for routed mixture-of-experts workloads. One process owns each GPU. CPU threads submit transfers, track completions and publish the gather-table rows that an already-running GPU kernel can consume. A worker can run communication alone or compose it with QuACK up/down projections, result returns and a reduction back into the original token order.

The executable and the Python worker use the same native implementation. Use the executable for transport and mock tests. Use the torchrun launchers for the Python-owned buffers and concurrent GEMM pipeline.

## First run on a Mac or Linux CPU

From this checkout, with CMake 3.18+ and a C++17 compiler:

```bash
cmake -S . -B build/mock -DCMAKE_BUILD_TYPE=Debug
cmake --build build/mock -j4
(cd build/mock && ctest --output-on-failure)
./build/mock/rdma_cpu_proxy --config config/example_config.json --cpu_affinity=none
```

The example explicitly selects host-backed mock transport. The affinity override runs it on macOS, where Linux CPU pinning is unavailable. Several tests use real loopback sockets between mock peers. A launcher preview works without torchrun, a GPU, or a built library:

```bash
TORCHRUN_BIN=echo bash scripts/run_single_node_torchrun.sh --communication-only
```

## Running on NVIDIA hardware

For eight-GPU nodes, choose the quick start for your hardware. Each includes the environment exports, native build, single-node test and commands for both hosts in a two-node test. [Hopper H100 / CUDA 12.9](docs/development.md#hopper-h100-quick-start) · [Blackwell / CUDA 13.0](docs/development.md#blackwell-cuda-130-quick-start)

- **Transport** uses NVIDIA GPUs, NVLink/peer-accessible CUDA IPC allocations and an NVIDIA/Mellanox NIC with libibverbs. Inter-node payloads require working GPU-memory registration and GPUDirect RDMA. The local-only launcher also initializes its configured NIC. The integrated profiles target eight GPUs per node. GPU/NIC maps, peer addresses, ports and GIDs are site-specific.
- **Build** the Linux shared library with a full CUDA 12.9+ toolkit, a compatible host compiler, rdma-core development headers/libraries and POSIX threads. Use CUDA architecture 90 for Hopper H100 or 100 for Blackwell B200/B300. Both paths support BF16/FP16 compute. Native transport, CuTe metadata and QuACK GEMM each have their own compilation path.
- **Compute** requires CUDA-enabled PyTorch and the separate communication-overlap QuACK checkout, including its `run/` helpers. Set `QUACK_ROOT` to that checkout or pass `--quack-root` to the worker. The inspected revision is `ccc12c585a453a4d96c333677aa58714d117c496`. Its documented B300 baseline uses Python 3.12, PyTorch 2.13.0+cu129, CUDA Python/bindings 12.9.4 and CUTLASS DSL 4.7.0. Match the shared CUDA runtime major used by PyTorch and the native library. [QuACK revision](https://github.com/ZH1243/quack_for_communication_overlap/tree/ccc12c585a453a4d96c333677aa58714d117c496)
- **Communication-only** still uses CUDA PyTorch in the embedded worker. The standalone CUDA metadata executable builds without PyTorch, QuACK or a NIC. The optional PyTorch metadata extension also needs the extension build toolchain including Ninja.

After setting the build and fabric environment in the development guide, a minimal single-node numerical run is:

```bash
bash scripts/run_single_node_torchrun.sh --num_tokens=1024 --num_iterations=3 --check-projections
```

For a larger single-node run, the [annotated parameter example](docs/features.md#reading-a-single-node-command) explains the workload, GEMM tiles, SM budget, forwarding and return policies, and timeouts, with Hopper and Blackwell constraints.

On Hopper, add `--pooled-up-gemm` to stream up GEMM from a contiguous receive pool. The default gathers from multiple buffers. The [H200 validation and timing commands](docs/development.md#pooled-up-gemm-on-h200) cover one and two nodes, including a same-pool kernel comparison.

Add `--router_dynamic_routing_enabled=true` to the matching launchers to regenerate routing, exchange metadata and stage inputs before each payload iteration. The [per-iteration routing guide](docs/development.md#per-iteration-routing) covers the extra metadata QP, compute views and numerical checks.

## Directory map

| Path | Responsibility |
| --- | --- |
| [CMakeLists.txt](CMakeLists.txt), [include/](include/), [native/](native/) | Build targets, public embedding ABI, internal C++ transport/scheduler and CUDA helpers |
| [metadata/](metadata/README.md) | Shared routing CUDA kernel, raw API, PyTorch JIT extension, CuTe kernels, references and benchmarks |
| [python/](python/) | torchrun worker, allocator, QuACK integration, local and network returns, combine, numerical references |
| [scripts/](scripts/), [config/](config/) | Single-node and two-node launch profiles, JSON configurations and site overrides |
| [tests/native/](tests/native/), [tests/python/](tests/python/) | CTest protocol/runtime tests and Python worker, projection and composition checks |
| [CPU exchange](benchmarks/router_metadata_exchange/README.md), [GPU exchange](benchmarks/router_metadata_exchange2/README.md) | Independent experiments, each with its own tests, build and launchers |
| [docs/](docs/) | Architecture, settings, validation and migration record |

Read the [iteration walkthrough](docs/architecture.md) to follow buffer and readiness ownership. Use the [feature reference](docs/features.md), [native configuration reference](docs/native-configuration.md) and [runtime and tuning reference](docs/runtime.md) to select a policy or interpret a trace. The [development guide](docs/development.md) contains Mac checks, remote setup, numerical and performance commands. The [migration guide and manifest](docs/migration.md) maps previous entry points and records provenance. [Agent instructions](AGENTS.md) and [copyable task prompts](docs/agent-prompts.md) cover future changes.
