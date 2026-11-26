"""Bridge the two-node CuTe/rme2 metadata graphs into the running meon proxy."""

import ctypes as C
from datetime import timedelta
import importlib.util
import os
from pathlib import Path
import socket
import sys
import time
import uuid


_ROOT = Path(__file__).resolve().parents[1]
_BENCH = _ROOT / "benchmarks" / "router_metadata_exchange2"
_I32P = C.POINTER(C.c_int32)
_U8P = C.POINTER(C.c_uint8)
# An interrupted job can leave another GPU with a live CUDA IPC mapping. Keep
# its allocation, graph operands and allocator owner alive until process exit.
_RETAINED = []


class BenchmarkSource(C.Structure):
    _fields_ = [("offsets", _I32P), ("indices", _I32P), ("count", C.c_uint64)]


class BenchmarkRouting(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint64), ("num_sources", C.c_int32),
        ("sources", C.POINTER(BenchmarkSource)), ("remote_x3", _I32P),
        ("remote_x4", _U8P), ("remote_count", C.c_uint64),
        ("device_r", C.c_size_t), ("device_node_offsets", C.c_size_t),
        ("device_x3", C.c_size_t), ("device_x4", C.c_size_t),
    ]


class BenchmarkHostRoute(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint64), ("node_offset_count", C.c_uint64),
        ("packed_count", C.c_uint64), ("assignment_count", C.c_uint64),
        ("node_offsets", _I32P), ("x3", _I32P), ("x4", _U8P), ("r", _I32P),
    ]


class BenchmarkDeviceIndices(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint32), ("num_sources", C.c_uint32),
        ("capacity_per_source", C.c_uint64), ("indices", C.POINTER(C.c_size_t)),
    ]


class BenchmarkConfig(C.Structure):
    _fields_ = [
        ("struct_size", C.c_uint64), ("num_tokens", C.c_uint64),
        ("router_seed", C.c_uint64), ("completion_timeout_ms", C.c_uint64),
        ("cuda_device_id", C.c_int32), ("node_rank", C.c_int32),
        ("local_gpu_index", C.c_int32), ("num_gpus_per_node", C.c_int32),
        ("num_nodes", C.c_int32), ("rdma_port", C.c_int32), ("gid_index", C.c_int32),
        ("rdma_device_name", C.c_char * 128),
    ]


def _load_module(name, path):
    """Give the CuTe compiler a stable Python module/source for introspection."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _generation_seed(base, global_rank, iteration):
    # Same uint64 Weyl sequence as IterationRouting::generate for this topology.
    return (base + 0x9E3779B97F4A7C15 * (iteration * 16 + global_rank)) & ((1 << 64) - 1)


class BenchmarkMetadataBackend:
    """One benchmark metadata worker, IPC arena and same-index QP per meon rank."""

    def __init__(self, library, handle, torch, device_buffers):
        self.library = library
        self.handle = handle
        self.torch = torch
        self.device_buffers = device_buffers
        self.session = None
        self.dist = None
        self.group = None
        self.owns_group = False
        self.exposed = False
        self.connected = False
        self.closed = False
        self.failed = False
        self.next_iteration = 0
        self.source_snapshot_view = None
        self.device_indices = None
        self.last_stats = {}
        self.config = BenchmarkConfig(struct_size=C.sizeof(BenchmarkConfig))
        self._bind()
        self._check(library.rdma_proxy_get_benchmark_routing_config(handle, C.byref(self.config)),
                    "querying benchmark metadata configuration")
        try:
            self._setup()
        except BaseException:
            try:
                self.close(clean=False)
            except Exception as cleanup_error:
                print(f"Benchmark metadata setup cleanup failed: {cleanup_error}", file=sys.stderr, flush=True)
            raise

    def _bind(self):
        install = getattr(self.library, "rdma_proxy_install_borrowed_benchmark_routing", None)
        retire = getattr(self.library, "rdma_proxy_retire_borrowed_benchmark_routing", None)
        if install is None or retire is None:
            install = self.library.rdma_proxy_install_benchmark_routing
            retire = None
        self.install_routing = install
        self.retire_routing = retire
        self.install_host_routing = None
        self.install_device_indices = None
        self.borrow_source_route = False
        if retire is not None:
            retire.argtypes, retire.restype = [C.c_void_p, C.c_uint64], C.c_int
            self.install_host_routing = getattr(self.library, "rdma_proxy_install_borrowed_benchmark_route_views", None)
            self.borrow_source_route = self.install_host_routing is not None
            if not self.borrow_source_route:
                self.install_host_routing = getattr(
                    self.library, "rdma_proxy_install_borrowed_benchmark_host_routing", None)
            if self.install_host_routing is not None:
                self.install_host_routing.argtypes = [
                    C.c_void_p, C.c_uint64, C.POINTER(BenchmarkRouting), C.POINTER(BenchmarkHostRoute)]
                self.install_host_routing.restype = C.c_int
            if self.borrow_source_route:
                self.install_device_indices = getattr(
                    self.library, "rdma_proxy_install_borrowed_benchmark_device_indices", None)
                if self.install_device_indices is not None:
                    self.install_device_indices.argtypes = [
                        C.c_void_p, C.c_uint64, C.POINTER(BenchmarkRouting),
                        C.POINTER(BenchmarkHostRoute), C.POINTER(BenchmarkDeviceIndices)]
                    self.install_device_indices.restype = C.c_int
        calls = (
            (self.library.rdma_proxy_get_benchmark_routing_config, [C.c_void_p, C.POINTER(BenchmarkConfig)]),
            (install, [C.c_void_p, C.c_uint64, C.POINTER(BenchmarkRouting)]),
            (self.library.rdma_proxy_get_routing_extents, [C.c_void_p, C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]),
        )
        for fn, args in calls:
            fn.argtypes, fn.restype = args, C.c_int

    def _check(self, status, operation):
        if status:
            message = self.library.rdma_proxy_last_error()
            raise RuntimeError(f"{operation} failed: {message.decode(errors='replace') if message else 'unknown error'}")

    def _all(self, value):
        out = [None] * self.dist.get_world_size(group=self.group)
        self.dist.all_gather_object(out, value, group=self.group)
        return out

    def _setup_step(self, operation, work):
        """Allow all participants to unwind when one rank rejects setup."""
        try:
            result, error = work(), None
        except Exception as exc:
            result, error = None, f"rank {self.rank}: {type(exc).__name__}: {exc}"
        failures = [error for error in self._all(error) if error]
        if failures:
            raise RuntimeError(f"{operation} failed: " + " | ".join(failures))
        return result

    def _barrier(self):
        self.dist.barrier(group=self.group)

    def _open_group(self, dist):
        self.dist = dist
        timeout = timedelta(milliseconds=self.config.completion_timeout_ms)
        if dist.is_initialized():
            # A caller's (possibly NCCL) default group has its own lifetime
            # and its own collective sequence. Use a dedicated CPU subgroup.
            self.group = dist.new_group(backend="gloo", timeout=timeout)
        else:
            dist.init_process_group("gloo", timeout=timeout)
            self.group = dist.group.WORLD
        self.owns_group = True
        self.rank = dist.get_rank(group=self.group)
        self.node, self.local = divmod(self.rank, 8)
        if dist.get_world_size(group=self.group) != 16:
            raise ValueError("benchmark metadata requires two torchrun nodes with eight ranks each")

    def _setup(self):
        import torch.distributed as dist

        self._open_group(dist)
        c = self.config
        def local_setup():
            if (c.num_nodes, c.num_gpus_per_node, c.node_rank, c.local_gpu_index, c.cuda_device_id) != (
                    2, 8, self.node, self.local, self.local):
                raise ValueError("benchmark metadata requires contiguous node ranks and CUDA device ID equal to local rank")
            if not c.rdma_device_name or not 0 < c.num_tokens <= 10_000_000:
                raise ValueError("benchmark metadata requires an explicit NIC and 1 to 10,000,000 tokens")
            torch = self.torch
            torch.cuda.set_device(self.local)
            if torch.cuda.device_count() < 8 or torch.cuda.get_device_capability(self.local)[0] < 9:
                raise ValueError("benchmark metadata requires eight visible Hopper or newer GPUs per node")
            if not all(torch.cuda.can_device_access_peer(self.local, gpu) for gpu in range(8) if gpu != self.local):
                raise ValueError("benchmark metadata requires peer access to every local GPU")
            source, destination = self.device_buffers.benchmark_staging_buffers()
            if source.shape != destination.shape or source.dtype != destination.dtype or source.shape[0] != c.num_tokens:
                raise ValueError("local staging buffers disagree with the routed workload")
            return socket.gethostname(), (c.num_tokens, c.router_seed, tuple(source.shape), str(source.dtype))

        shapes = self._all(self._setup_step("checking benchmark topology", local_setup))
        if len({shape for _, shape in shapes}) != 1 or any(
                len({host for host, _ in shapes[n:n + 8]}) != 1 for n in (0, 8)):
            raise ValueError("benchmark metadata token shape, seed, dtype or node rank grouping differs across ranks")

        names = ["/meon_rme2_" + uuid.uuid4().hex if self.rank == 0 else None]
        dist.broadcast_object_list(names, src=0, group=self.group)
        shm_name = f"{names[0]}_n{self.node}"
        library_path = os.environ.get("MEON_BENCHMARK_METADATA_LIBRARY",
                                      Path(self.library._name).resolve().with_name("librouter_metadata_exchange2.so"))

        def create():
            proxy_cls = _load_module("_meon_rme2_native", _BENCH / "native.py").Proxy
            self.session = proxy_cls(library_path, self.local, self.local, c.num_tokens, shm_name,
                                     c.completion_timeout_ms / 1000, False, -1, nodes=2, node=self.node,
                                     receiver_kernel=True, receiver_overlap_exchange=True,
                                     local_staging_after_d2h=True)

        self._setup_step("creating node metadata control", lambda: create() if self.local == 0 else None)
        self._setup_step("joining node metadata control", lambda: create() if self.local != 0 else None)
        direct_indices = self._enable_device_indices()
        descriptor = self._setup_step("registering metadata arenas", lambda: self.session.rdma_init(
            os.fsdecode(c.rdma_device_name), c.rdma_port, c.gid_index))
        peers = self._all(descriptor)
        self._setup_step("connecting the same-index metadata QP",
                         lambda: self.session.rdma_connect(peers[self.rank ^ 8]))
        exported = self._setup_step("exporting metadata IPC arena", self.session.export)
        self.exposed = True
        handles = self._all(exported)
        self._setup_step("connecting local metadata IPC arenas",
                         lambda: self.session.connect(handles[self.node * 8:self.node * 8 + 8]))
        self.connected = True
        self._setup_step("retiring metadata control name", lambda: self.session.unlink() if self.local == 0 else None)
        if direct_indices:
            self._setup_step("binding exchange device indices", self._bind_device_indices)
        self._setup_step("capturing CuTe metadata and meon local staging graphs", self._capture)

    def _enable_device_indices(self):
        # Peers must agree on their RDMA destinations before registering the arenas.
        capable = (self.install_device_indices is not None and self.session.supports_device_indices()
                   and hasattr(self.session.lib, "rme2_enable_source_snapshot")
                   and hasattr(self.session.lib, "rme2_source_snapshot_view"))
        if not all(self._all(capable)):
            return False
        self._setup_step("enabling exchange device indices", self.session.enable_device_indices)
        return True

    def _bind_device_indices(self):
        gpus = self.config.num_gpus_per_node
        sources = self.config.num_nodes * gpus
        arrays = [self.session.device_index_array(source) for source in range(sources)]
        capacity = self.config.num_tokens * 8
        self.device_buffers.bind_benchmark_index_arrays(arrays, gpus, capacity)
        addresses = (C.c_size_t * sources)(*(array.address for array in arrays))
        self.device_indices = BenchmarkDeviceIndices(C.sizeof(BenchmarkDeviceIndices), sources, capacity, addresses)

    def _enable_source_snapshot(self):
        if self.install_host_routing is None:
            return
        enable = getattr(self.session.lib, "rme2_enable_source_snapshot", None)
        view = getattr(self.session.lib, "rme2_source_snapshot_view", None)
        if enable is None or view is None:
            return  # Older exchange libraries leave the outgoing route on the device.
        enable.argtypes, enable.restype = [C.c_void_p, C.c_void_p], C.c_int
        view.argtypes = [C.c_void_p, C.c_uint64, C.POINTER(_I32P), C.POINTER(_I32P),
                         C.POINTER(_U8P), C.POINTER(_I32P)]
        view.restype = C.c_int
        # self.r is updated in place and stays alive through benchmark shutdown.
        self.session.check(enable(self.session.handle, self.r.data_ptr()))
        self.source_snapshot_view = view

    def _capture(self):
        import cuda.bindings.driver as cuda
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        torch = self.torch
        module = _load_module("_meon_metadata_cute", _ROOT / "metadata" / "metadata_cute.py")
        device = torch.device("cuda", self.local)
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream):
            self.r = torch.arange(8, device=device, dtype=torch.int32).expand(self.config.num_tokens, 8).clone()
            self.plan = module.MetadataPlan(self.r, self.local,
                                            rdma_x3_x4_payload=self.session.enable_direct_x3_x4(),
                                            remote_output_node=1 - self.node)
            self._enable_source_snapshot()
            self.staging = module.LocalStagingPlan(self.plan, *self.device_buffers.benchmark_staging_buffers(),
                                                  node_rank=self.node)
            self.r_argument = from_dlpack(self.r)
            self.cu_stream = cuda.CUstream(self.stream.cuda_stream)
            self.seed_type = cutlass.Uint64
            self.generator = cute.compile(module.RouteGenerator(), self.r_argument,
                                          self.seed_type(self.config.router_seed), self.cu_stream)
            # Prime the same operations as the benchmark, against the actual
            # payload allocation. The first measured gather overwrites its prefix.
            self._generate(0)
            self.plan.launch(self.stream)
            self.staging.launch(self.stream)
            self.stream.synchronize()
            self.session.capture_metadata(self.plan, self.stream, launch_barrier=True)
            self.session.capture_staging(self.staging, self.stream)

    def _generate(self, iteration):
        seed = _generation_seed(self.config.router_seed, self.rank, iteration)
        self.generator(self.r_argument, self.seed_type(seed), self.cu_stream)

    def _install(self, iteration):
        session = self.session
        sources = (BenchmarkSource * 16)()
        for rank in range(16):
            offsets, indices, count = _I32P(), _I32P(), C.c_int()
            session.check(session.lib.rme2_view(session.handle, rank, C.byref(offsets), C.byref(indices), C.byref(count)))
            if not offsets or count.value < 0 or (count.value and not indices):
                raise RuntimeError(f"invalid borrowed benchmark record for source {rank}")
            sources[rank] = BenchmarkSource(offsets, indices, count.value)
        x3, x4, count = _I32P(), _U8P(), C.c_int()
        session.check(session.lib.rme2_x3_x4_view(session.handle, C.byref(x3), C.byref(x4), C.byref(count)))
        if count.value < 0 or (count.value and (not x3 or not x4)):
            raise RuntimeError("invalid borrowed remote x3/x4 record")
        record = BenchmarkRouting(C.sizeof(BenchmarkRouting), 16, sources, x3, x4, count.value)
        if self.source_snapshot_view is not None:
            offsets, source_x3, source_x4, r = _I32P(), _I32P(), _U8P(), _I32P()
            # Benchmark sequences start at one; payload iterations start at zero.
            session.check(self.source_snapshot_view(session.handle, iteration + 1, C.byref(offsets),
                                                    C.byref(source_x3), C.byref(source_x4), C.byref(r)))
            if not offsets:
                raise RuntimeError("benchmark source snapshot is missing its node offsets")
            nodes = self.config.num_nodes
            host = BenchmarkHostRoute(C.sizeof(BenchmarkHostRoute), nodes + 1, offsets[nodes],
                                      8 * self.config.num_tokens, offsets, source_x3, source_x4, r)
            if self.device_indices is not None:
                status = self.install_device_indices(self.handle, iteration, C.byref(record), C.byref(host),
                                                     C.byref(self.device_indices))
            else:
                status = self.install_host_routing(self.handle, iteration, C.byref(record), C.byref(host))
        else:
            record.device_r, record.device_node_offsets = self.r.data_ptr(), self.plan.node_offsets.data_ptr()
            record.device_x3, record.device_x4 = self.plan.x3.data_ptr(), self.plan.x4.data_ptr()
            status = self.install_routing(self.handle, iteration, C.byref(record))
        self._check(status, "installing benchmark routing in the payload proxy")

    def prepare(self, iteration, *, compute_extents=True):
        if self.closed or self.failed or iteration != self.next_iteration:
            raise RuntimeError("benchmark metadata requires an active session and consecutive iterations")
        try:
            started = time.perf_counter_ns()
            self._generate(iteration)
            self.stream.synchronize()
            r_ready = time.perf_counter_ns()
            # Drain readers of both the peer records and our outgoing snapshot
            # before this rank or its peers can overwrite either for the next epoch.
            reuse_start = r_ready
            if iteration and self.retire_routing is not None:
                self._check(self.retire_routing(self.handle, iteration - 1),
                            "retiring the previous benchmark routing")
                reuse_start = time.perf_counter_ns()
            self._barrier()
            exchange_start = time.perf_counter_ns()
            self.session.submit(iteration + 1, self.plan, self.stream)
            native_stats = self.session.wait()
            exchanged = time.perf_counter_ns()
            self._install(iteration)
            installed = time.perf_counter_ns()
            # Local setup can use this route now. Native iteration-start synchronization
            # precedes payload dispatch and publication of ready rows on both nodes.
            extents = None
            if compute_extents:
                indices, rows = C.c_uint64(), C.c_uint64()
                self._check(self.library.rdma_proxy_get_routing_extents(
                    self.handle, C.byref(indices), C.byref(rows)), "querying benchmark routing extents")
                extents = (indices.value, rows.value)
            self.last_stats = {
                "r_ready_ns": r_ready, "generation_elapsed_ns": r_ready - started,
                "exchange_staging_elapsed_ns": exchanged - exchange_start, "install_elapsed_ns": installed - exchanged,
                "r_ready_to_install_elapsed_ns": installed - r_ready,
                "retire_elapsed_ns": reuse_start - r_ready, "reuse_barrier_ns": exchange_start - reuse_start,
                "release_barrier_ns": 0,  # Retain the timing field for older log readers.
                "source_route": "receiver" if self.source_snapshot_view is not None else "device_readback",
                "source_route_storage": ("borrowed" if self.source_snapshot_view is not None
                                         and self.borrow_source_route else "copied"),
                "a_idx_storage": "exchange_device" if self.device_indices is not None else "host_upload",
                "rme2_stats": native_stats,
            }
            self.next_iteration += 1
            return extents, installed - started
        except BaseException:
            self.failed = True
            raise

    def close(self, clean=True):
        """A clean close requires confirmed shutdown of the payload and its readers of pinned routing data."""
        if self.closed:
            return
        self.closed = True
        if self.exposed and (not clean or self.failed or not self.connected):
            self._abort_and_retain()
            return
        try:
            if clean and not self.failed and self.connected:
                self._barrier()
                self.session.disconnect()
                self._barrier()  # Every IPC import is closed before any owner frees its arena.
                self.exposed = False
            if self.session is not None:
                self.session.close()
            if self.owns_group:
                self.dist.destroy_process_group(self.group)
                self.owns_group = False
        except BaseException:
            if self.exposed:
                self._abort_and_retain()
            raise

    def _abort_and_retain(self):
        if self not in _RETAINED:
            _RETAINED.append(self)
        # ABI 15 libraries built before rme2_abort still bound waits using the
        # configured timeout. Neither path frees memory used by unconfirmed peers.
        if self.session is not None:
            abort = getattr(self.session.lib, "rme2_abort", None)
            if abort is not None:
                abort.argtypes = [C.c_void_p]
                abort.restype = C.c_int
                self.session.check(abort(self.session.handle))
