# Migration to meon

The migration uses the working-tree contents of `GPU_Data_Transfer_Profile` at `532313556f368a1d1cccb5440564dcde82e9fee5` on branch `RDMA_CPU_Proxy_Add_SonicMoE_IntegrateRouter`. Both requested directories were clean against that revision. The destination started with an initialized Git directory, an unborn `main` tracking `origin/main` (reported gone), and no other files. The migration left Git remotes, the index, branches and history as found.

All **144 tracked inputs** from the two directories are accounted for. Of these, 119 are byte-identical at their new paths. The inventory includes test code, benchmark-local ignore files and documentation. The file manifest gives each original blob, destination, and byte-equality result. The production C/C++ headers, runtime sources, routing CUDA kernel, CUDA extension binding, CuTe metadata kernels (apart from a provenance path in one docstring), C++ tests and four JSON profiles retain their implementation. [Per-file manifest](migration-manifest.md)

## Find the replacement

| Original in the source checkout | New path in this checkout | Reason |
| --- | --- | --- |
| `RDMA_CPU_Proxy/CMakeLists.txt` | `CMakeLists.txt` | Build from the repository root. |
| `RDMA_CPU_Proxy/src/`, `RDMA_CPU_Proxy/include/` | `native/src/`, `native/include/` | Runtime internals share an owner. The embedding header alone lives at `include/rdma_proxy_c_api.h`. Its include spelling stays the same. |
| `RDMA_CPU_Proxy/scripts/*.py` | `python/` | Worker, return and combine modules retain sibling imports. The standalone combine benchmark uses the same modules. |
| `RDMA_CPU_Proxy/scripts/*.sh`, `RDMA_CPU_Proxy/config/` | `scripts/`, `config/` | Existing per-node launchers remain the operational entry points. |
| `get_expert_token_idx_list_gpu/` | `metadata/` | Native CMake, the PyTorch JIT extension and the raw CUDA benchmark share one CUDA implementation. The two-node GPU benchmark loads the CuTe implementation here. |
| `RDMA_CPU_Proxy/tests/` | `tests/native/`, `tests/python/` | CTest and unittest have separate discovery roots. The independent metadata/benchmark tests stay with their respective builds. |
| `RDMA_CPU_Proxy/benchmarks/*` | `benchmarks/*` | Each experiment retains its own CMake project, library, launchers and checks. Source paths into the shared native runtime and metadata directory were updated. |
| `RDMA_CPU_Proxy/README.md`, `RDMA_CPU_Proxy/docs/` | `docs/runtime.md`, `docs/system_arch_graph_01.md` | Preserve the detailed hardware/tuning material alongside a shorter entry point and iteration guide. |

## Update a job or development environment

Each right-hand command starts in the **meon root**. On the left, the paths are relative to the old outer checkout.

| Previous command or setting | Replacement |
| --- | --- |
| `cmake -S RDMA_CPU_Proxy -B RDMA_CPU_Proxy/build-hopper` | `cmake -S . -B build-hopper` with the same CUDA/toolkit/verbs options. Configure a fresh directory on each node. |
| `cmake --build RDMA_CPU_Proxy/build-hopper` | `cmake --build build-hopper` |
| `RDMA_CPU_Proxy/build-hopper/rdma_cpu_proxy --config RDMA_CPU_Proxy/config/node0_config.json` | `./build-hopper/rdma_cpu_proxy --config config/node0_config.json` with the same topology/transport overrides |
| `bash RDMA_CPU_Proxy/scripts/run_single_node_torchrun.sh` | `bash scripts/run_single_node_torchrun.sh` |
| `bash RDMA_CPU_Proxy/scripts/run_node0_torchrun.sh` | `bash scripts/run_node0_torchrun.sh` on node 0 and the node 1 counterpart on node 1 |
| `torchrun … RDMA_CPU_Proxy/scripts/rdma_proxy_worker.py …` | `torchrun … python/rdma_proxy_worker.py …`, updating `--library` and `--config` as needed |
| `python -m unittest discover -s RDMA_CPU_Proxy/tests` | `python -m unittest discover -s tests/python -v` |
| `python RDMA_CPU_Proxy/scripts/benchmark_intra_node_combine.py` | `python python/benchmark_intra_node_combine.py` |
| `python get_expert_token_idx_list_gpu/run.py --node-mask-sort --check` | `python metadata/run.py --node-mask-sort --check` |
| `bash get_expert_token_idx_list_gpu/build_rdma_metadata_benchmark.sh` | `bash metadata/build_rdma_metadata_benchmark.sh` |
| `cd RDMA_CPU_Proxy/benchmarks/router_metadata_exchange2` | `cd benchmarks/router_metadata_exchange2`, then use the benchmark's local commands |

**External dependencies.** The default lookup for QuACK now starts at the parent of this standalone checkout. Set `QUACK_ROOT` or the worker's `--quack-root` when it lives elsewhere. Keep the full fork, including its `run/` modules, alongside its installed Python dependencies. The integration uses architecture-specific CTA geometry, streamed gather/activation, weighted scatter, bulk-store options and IPC helpers. The inspected QuACK revision is `ccc12c585a453a4d96c333677aa58714d117c496`, with package version 0.6.4. It is the compatibility reference for this migration. The development guide gives the build flags, pinned baseline environment and device checks. [QuACK fork at the inspected revision](https://github.com/ZH1243/quack_for_communication_overlap/tree/ccc12c585a453a4d96c333677aa58714d117c496) · [Setup and tests](development.md)

**Compatibility.** Native targets, the Linux library basename and SONAME, C ABI version 1, allocator request version 1, core structure layouts, QP/metadata encodings, notification v6 and local iteration synchronization v7 are preserved. The CPU exchange experiment retains ABI 7 and timing schema 9. The GPU exchange experiment retains ABI 15 and result schema 16. The launchers still forward caller arguments after the profile and site arguments. Site-specific JSON, device maps, build-directory and launcher overrides remain available. Existing installations should point jobs at the new checkout and rebuild binaries/extensions there. Installations with custom directory layouts can continue to set their existing location overrides. [Native contracts](native-configuration.md) · [Effective profiles](features.md)

**Test fixture correction.** The CPU scatter publication unittest constructed a partial down runner using `__new__`. The original fixture lacked the output-reset policy and reset-end event used by the real runner and failed before testing its metadata gate. The migrated fixture supplies both, and checks the reset-end ordering on success and failure. Production initialization, error handling and scheduling were preserved. [Test](../tests/python/test_cpu_scatter_publication.py)

## Scope and exclusions

- **Inside the two input directories:** inspected tracked, untracked and ignored files. There was no additional untracked project source. Excluded native/benchmark `build/` directories (CMake metadata, generated test JSON, objects, binaries and dylib links), Python bytecode and Finder metadata. An obsolete built test binary also existed with no current source/target. All builds used for checks were fresh.
- **Outside them:** inspected source includes, CMake inputs, JIT paths, Python imports, relative file opens and launch scripts. All implementation dependencies on neighboring project source resolve within the migrated directories. The outer repository's untracked `some_commands_used.sh` is a personal command/profiler scratchpad. `Megatron-LM-6-11/` is a separate nested checkout. Neither is called by the runtime, tests or build. The surrounding profiling/DeepEP/transfer experiments are independent.
- **QuACK:** read its instructions, source, dependency pins and examples. Its untracked editor settings and Nsight captures were excluded. QuACK code, Git metadata and compiled caches stay in the external checkout. The only file copied directly from QuACK was its license text for an attribution already present in the migrated code.
- **Provenance:** the source root and migrated components supplied no project license. The register-prefetch combine kernel identifies its adaptation from QuACK's `run/final_combine.py`. That attribution is retained and QuACK's Apache 2.0 text is included with this component-specific record. The single-node CuTe exchange specialization also credits the project's own metadata implementation. [Adapted kernel](../python/intra_node_combine_kernel.py) · [QuACK license text](licenses/QuACK-Apache-2.0.txt)

## Parity checks

Mapped every file and checked original executable permissions. The root and benchmark CMake builds retain their targets, tests, options and compile definitions. Compared worker argparse flags, defaults, aliases and choices, all JSON profiles and the effective launch arguments across all three launchers and all three architecture settings. Comparisons also covered communication-only, weighted and threaded scatter, complete rounds, return batch synchronization and end-of-iteration reset. The native transport, public ABI, projection reference and composition assertions remain available for direct review. [Inventory](features.md) · [Manifest](migration-manifest.md)

Host builds and mock runs exercise wiring, protocol checks, host-side ordering, reset/cleanup and loopback exchange. Remote acceptance still requires fresh CUDA builds, extension and CuTe loading, actual QuACK execution, device memory registration and peer visibility, real asynchronous publications, repeated checked iterations and matched warm performance measurements. [Remote validation sequence](development.md)

### Mac validation record, 2026-11-18

| Check actually run | Result |
| --- | --- |
| Fresh root Debug build on AppleClang 14, including a second build from an isolated copy with neither source checkout beside it | Built the executable, static and shared libraries and all test targets. The compile database uses the standalone sources. The public header compiled as C11 and Python loaded C ABI 1 from the new dylib. |
| Native CTest | 38 passed, 3 CUDA tests skipped, zero failures. The sandbox denied shared-memory creation and loopback binding on the isolated run. The same suite passed with those permissions. |
| Standalone CLI, example JSON, affinity disabled, two iterations | Each mock iteration transferred 3,840,000 bytes across three emulated peers with zero immediate mismatches and zero validation errors. |
| Python worker/projection/return tests | With CPU PyTorch 1.13.1, unittest reported 66 tests, OK with 5 skips. With the separate Python interpreter without PyTorch, it reported 59 tests, OK with 9 skips. The CPU PyTorch suite also passed from the isolated copy. |
| Metadata and exchange experiments | 4 host metadata tests passed. CPU exchange passed its native codec test and all 38 Python loopback tests. GPU exchange passed all 10 host-only CTests. Both benchmark projects built in Debug and Release host configurations. |
| Configuration parity | 66 worker parser actions matched. All 9 single/node-0/node-1 by auto/Hopper/Blackwell launcher comparisons and 27 derived native-argument comparisons matched after path normalization. Runtime-guide examples were checked against native configuration validation. |
| Source and documentation | Parsed all 34 Python files and four JSON profiles, syntax-checked 14 shell files and 126 shell example blocks, and resolved all 260 local Markdown links. No active path back to the source checkout or symlink is required. |
