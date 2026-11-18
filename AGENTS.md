# Working in meon

Start with [README.md](README.md) and the [iteration walkthrough](docs/architecture.md). Trace worker arguments from `scripts/` through `python/rdma_proxy_worker.py`, then through `native/src/config.cpp`. Compiled defaults, JSON and launch profiles are different layers. [Feature reference](docs/features.md) · [Native settings](docs/native-configuration.md)

- The public embedding contract lives in `include/rdma_proxy_c_api.h`. Preserve ctypes layouts, version checks, aliases, return/error semantics, shared-memory and wire formats across peers. Reuse the routing CUDA source in `metadata/` for the native and JIT paths.
- Follow each asynchronous pointer to its owner and last consumer. Preserve the order of payload visibility, descriptor upload, ready-prefix publication, consumption and reuse. Record the current iteration's event before another stream waits on it. Include cancellation, repeated iterations and partial setup in changes to this path.
- Keep source identity and row coordinates intact through x3/x4, expert lists, packed GEMM rows, weights and compact return buffers. Use the independent composition and projection references for numerical changes.
- QuACK is external. Select its compatible checkout with `QUACK_ROOT`. Review both sides of a proposed interface change before editing either.
- Keep the relevant native tests, Python regressions and documentation with each feature. Build Debug for assertion-based host tests. Run the affected benchmark's own suite when changing shared metadata. Use the [test guide](docs/development.md) to select hardware checks. Record the exact commands and hardware actually used, plus outstanding remote checks.

[Runtime details](docs/runtime.md) · [Migration and provenance](docs/migration.md) · [Task prompts](docs/agent-prompts.md)
