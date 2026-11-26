"""CPU regressions for the benchmark session's rank, epoch, ABI and owner handoff."""
import ctypes as C
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import benchmark_metadata_backend as backend


class ProviderTests(unittest.TestCase):
    def provider(self):
        p = backend.BenchmarkMetadataBackend.__new__(backend.BenchmarkMetadataBackend)
        p.handle = 3
        p.closed = p.failed = p.owns_group = p.exposed = p.connected = False
        p.next_iteration = 0
        p.retire_routing = None
        p.install_host_routing = p.source_snapshot_view = None
        p.install_device_indices = p.device_indices = None
        p.borrow_source_route = False
        p.last_stats = {}
        p.group = None
        p.session = None
        p.config = backend.BenchmarkConfig(struct_size=C.sizeof(backend.BenchmarkConfig),
                                           num_tokens=20, num_nodes=2, num_gpus_per_node=8,
                                           completion_timeout_ms=20_000)
        return p

    def test_uint64_generation_seed_and_epoch(self):
        self.assertEqual(backend._generation_seed(21, 0, 0), 21)
        self.assertEqual(backend._generation_seed(21, 15, 0), (21 + 15 * 0x9e3779b97f4a7c15) % 2**64)
        self.assertEqual(backend._generation_seed(2**64-1, 0, 1), (16 * 0x9e3779b97f4a7c15 - 1) % 2**64)

    def test_capture_uses_registered_remote_slice_and_real_local_payload(self):
        p = self.provider()
        p.node, p.local, p.rank = 1, 3, 11
        p.torch, p.session, p.device_buffers = MagicMock(), MagicMock(), MagicMock()
        p.install_host_routing = MagicMock()
        p.session.lib.rme2_enable_source_snapshot.return_value = 0
        source, destination = object(), object()
        p.device_buffers.benchmark_staging_buffers.return_value = source, destination
        p.session.enable_direct_x3_x4.return_value = 2**40
        cute_metadata, cuda, cutlass = MagicMock(), MagicMock(), MagicMock()
        imports = {"cuda": cuda, "cuda.bindings": cuda.bindings, "cuda.bindings.driver": cuda.bindings.driver,
                   "cutlass": cutlass, "cutlass.cute": cutlass.cute, "cutlass.cute.runtime": cutlass.cute.runtime}
        cutlass.Uint64.side_effect = int
        with patch.dict(sys.modules, imports), patch.object(backend, "_load_module", return_value=cute_metadata):
            p._capture()
        cute_metadata.MetadataPlan.assert_called_once_with(
            p.r, 3, rdma_x3_x4_payload=2**40, remote_output_node=0)
        cute_metadata.LocalStagingPlan.assert_called_once_with(
            p.plan, source, destination, node_rank=1)
        p.session.capture_metadata.assert_called_once_with(p.plan, p.stream, launch_barrier=True)
        p.session.lib.rme2_enable_source_snapshot.assert_called_once_with(p.session.handle, p.r.data_ptr())
        p.session.capture_staging.assert_called_once_with(p.staging, p.stream)
        p.generator.assert_called_once_with(p.r_argument, backend._generation_seed(0, 11, 0), p.cu_stream)

    def test_source_snapshot_requires_support_in_both_libraries(self):
        enable, view = MagicMock(return_value=0), MagicMock()
        for use_payload, use_enable, use_view in ((True, True, True), (True, True, False),
                                                  (True, False, True), (False, True, True)):
            with self.subTest(payload=use_payload, enable=use_enable, view=use_view):
                enable.reset_mock()
                p = self.provider()
                p.install_host_routing = MagicMock() if use_payload else None
                p.r = SimpleNamespace(data_ptr=lambda: 2**40)
                lib = SimpleNamespace()
                p.session = SimpleNamespace(handle=9, lib=lib, check=lambda s: self.assertEqual(s, 0))
                if use_enable:
                    lib.rme2_enable_source_snapshot = enable
                if use_view:
                    lib.rme2_source_snapshot_view = view
                p._enable_source_snapshot()
                if use_payload and use_enable and use_view:
                    enable.assert_called_once_with(9, 2**40)
                    self.assertIs(p.source_snapshot_view, view)
                else:
                    enable.assert_not_called()
                    self.assertIsNone(p.source_snapshot_view)

    def test_borrowed_views_keep_all_source_identities_without_python_row_lists(self):
        p = self.provider()
        offsets = [(C.c_int32 * 9)(*([0] * 8), rank % 3) for rank in range(16)]
        values = [(C.c_int32 * 2)(rank, rank + 1) for rank in range(16)]
        remote_x3 = (C.c_int32 * 2)(2, 7)
        remote_x4 = (C.c_uint8 * 2)(0x80, 0x03)
        def view(h, rank, o, v, n):
            C.cast(o, C.POINTER(backend._I32P))[0] = C.cast(offsets[rank], backend._I32P)
            C.cast(v, C.POINTER(backend._I32P))[0] = C.cast(values[rank], backend._I32P)
            C.cast(n, C.POINTER(C.c_int))[0] = rank % 3
            return 0
        def x3_view(h, x, m, n):
            C.cast(x, C.POINTER(backend._I32P))[0] = C.cast(remote_x3, backend._I32P)
            C.cast(m, C.POINTER(backend._U8P))[0] = C.cast(remote_x4, backend._U8P)
            C.cast(n, C.POINTER(C.c_int))[0] = 2
            return 0
        def install(h, epoch, ptr):
            record = C.cast(ptr, C.POINTER(backend.BenchmarkRouting)).contents
            self.assertEqual((h, epoch), (3, 7))
            self.assertEqual((record.struct_size, record.num_sources),
                             (C.sizeof(backend.BenchmarkRouting), 16))
            for rank in range(16):
                source = record.sources[rank]
                self.assertEqual(source.count, rank % 3)
                self.assertEqual(C.addressof(source.offsets.contents), C.addressof(offsets[rank]))
                self.assertEqual(C.addressof(source.indices.contents), C.addressof(values[rank]))
            self.assertEqual(record.remote_x3[:record.remote_count], [2, 7])
            self.assertEqual(record.remote_x4[:record.remote_count], [0x80, 3])
            self.assertEqual([record.device_r, record.device_node_offsets, record.device_x3, record.device_x4],
                             [2**40 + i for i in range(4)])
            return 0
        p.session = SimpleNamespace(handle=4, check=lambda s: self.assertEqual(s, 0),
                                    lib=SimpleNamespace(rme2_view=view, rme2_x3_x4_view=x3_view))
        p.install_routing = install
        def device(i):
            return SimpleNamespace(data_ptr=lambda: 2**40 + i)
        p.r = device(0)
        p.plan = SimpleNamespace(node_offsets=device(1), x3=device(2), x4=device(3))
        p._install(7)

        local_offsets = (C.c_int32 * 3)(0, 3, 4)
        local_x3 = (C.c_int32 * 4)(1, 3, 7, 12)
        local_x4 = (C.c_uint8 * 4)(1, 2, 4, 8)
        local_r = (C.c_int32 * (8 * p.config.num_tokens))()
        def source_view(handle, sequence, o, x, m, r):
            self.assertEqual((handle, sequence), (p.session.handle, 8))
            for output, value, typ in ((o, local_offsets, backend._I32P), (x, local_x3, backend._I32P),
                                       (m, local_x4, backend._U8P), (r, local_r, backend._I32P)):
                C.cast(output, C.POINTER(typ))[0] = C.cast(value, typ)
            return 0
        def install_host(handle, epoch, ptr, local):
            record = C.cast(ptr, C.POINTER(backend.BenchmarkRouting)).contents
            self.assertEqual((handle, epoch, record.num_sources), (3, 7, 16))
            self.assertEqual((record.device_r, record.device_node_offsets, record.device_x3, record.device_x4),
                             (0, 0, 0, 0))
            host = C.cast(local, C.POINTER(backend.BenchmarkHostRoute)).contents
            self.assertEqual((host.struct_size, host.node_offset_count, host.packed_count, host.assignment_count),
                             (C.sizeof(backend.BenchmarkHostRoute), 3, 4, 8 * p.config.num_tokens))
            for pointer, storage in ((host.node_offsets, local_offsets), (host.x3, local_x3),
                                     (host.x4, local_x4), (host.r, local_r), (record.remote_x3, remote_x3),
                                     (record.remote_x4, remote_x4)):
                self.assertEqual(C.addressof(pointer.contents), C.addressof(storage))
            return 0
        p.install_routing = MagicMock(side_effect=AssertionError("unexpected device readback"))
        p.source_snapshot_view, p.install_host_routing = source_view, MagicMock(side_effect=install_host)
        p.r, p.plan = None, None  # A host install gets every pointer from the receiver.
        p._install(7)
        p.install_host_routing.assert_called_once()
        arrays = [SimpleNamespace(address=2**40 + source * 4096, capacity=160) for source in range(16)]
        p.device_buffers = MagicMock()
        p.session.device_index_array = arrays.__getitem__
        p._bind_device_indices()
        p.device_buffers.bind_benchmark_index_arrays.assert_called_once_with(arrays, 8, 160)
        def install_direct(handle, epoch, ptr, local, device):
            install_host(handle, epoch, ptr, local)
            slots = C.cast(device, C.POINTER(backend.BenchmarkDeviceIndices)).contents
            self.assertEqual(slots.struct_size, C.sizeof(backend.BenchmarkDeviceIndices))
            self.assertEqual((slots.num_sources, slots.capacity_per_source), (16, 160))
            self.assertEqual(slots.indices[:16], [array.address for array in arrays])
            return 0
        p.install_device_indices = MagicMock(side_effect=install_direct)
        p._install(7)
        p.install_device_indices.assert_called_once()
        p.install_host_routing.assert_called_once()
        p.source_snapshot_view = MagicMock(return_value=-1)
        def check(status):
            if status:
                raise RuntimeError("failed receiver view")
        p.session.check = check
        with self.assertRaisesRegex(RuntimeError, "failed receiver view"):
            p._install(8)
        p.install_host_routing.assert_called_once()
        p.install_device_indices.assert_called_once()

    def test_index_destination_agrees_before_enabling_the_exchange(self):
        for local_payload, local_exchange, peer, expected in (
                (False, True, True, False), (True, False, True, False),
                (True, True, False, False), (True, True, True, True)):
            with self.subTest(local_payload=local_payload, local_exchange=local_exchange, peer=peer):
                p = self.provider()
                p.install_device_indices = MagicMock() if local_payload else None
                p.session = MagicMock()
                p.session.supports_device_indices.return_value = local_exchange
                p._all = lambda capable: [capable] + [peer] * 15
                p._setup_step = lambda operation, work: work()
                self.assertIs(p._enable_device_indices(), expected)
                self.assertEqual(p.session.enable_device_indices.call_count, int(expected))

    def test_borrowed_install_requires_both_native_entry_points(self):
        for has_install in (False, True):
            for has_retire in (False, True):
                with self.subTest(has_install=has_install, has_retire=has_retire):
                    p = self.provider()
                    lib = p.library = SimpleNamespace(rdma_proxy_get_benchmark_routing_config=MagicMock(),
                                                      rdma_proxy_get_routing_extents=MagicMock(),
                                                      rdma_proxy_install_borrowed_benchmark_host_routing=MagicMock(),
                                                      rdma_proxy_install_benchmark_routing=MagicMock())
                    if has_install:
                        lib.rdma_proxy_install_borrowed_benchmark_routing = MagicMock()
                    if has_retire:
                        lib.rdma_proxy_retire_borrowed_benchmark_routing = MagicMock()
                    p._bind()
                    if has_install and has_retire:
                        self.assertIs(p.install_routing, lib.rdma_proxy_install_borrowed_benchmark_routing)
                        self.assertIs(p.retire_routing, lib.rdma_proxy_retire_borrowed_benchmark_routing)
                        self.assertIs(p.install_host_routing, lib.rdma_proxy_install_borrowed_benchmark_host_routing)
                    else:
                        self.assertIs(p.install_routing, lib.rdma_proxy_install_benchmark_routing)
                        self.assertIsNone(p.retire_routing)
                        self.assertIsNone(p.install_host_routing)

    def test_outgoing_route_views_require_retirement_and_take_precedence_over_host_copy(self):
        p = self.provider()
        lib = p.library = SimpleNamespace(
            rdma_proxy_get_benchmark_routing_config=MagicMock(), rdma_proxy_get_routing_extents=MagicMock(),
            rdma_proxy_install_benchmark_routing=MagicMock(), rdma_proxy_install_borrowed_benchmark_routing=MagicMock(),
            rdma_proxy_install_borrowed_benchmark_route_views=MagicMock(),
            rdma_proxy_install_borrowed_benchmark_host_routing=MagicMock())
        p._bind()
        self.assertIsNone(p.install_host_routing)
        self.assertFalse(p.borrow_source_route)
        lib.rdma_proxy_retire_borrowed_benchmark_routing = MagicMock()
        p._bind()
        self.assertIs(p.install_host_routing, lib.rdma_proxy_install_borrowed_benchmark_route_views)
        self.assertTrue(p.borrow_source_route)
        self.assertEqual(p.install_host_routing.argtypes, [
            C.c_void_p, C.c_uint64, C.POINTER(backend.BenchmarkRouting), C.POINTER(backend.BenchmarkHostRoute)])
        del lib.rdma_proxy_install_borrowed_benchmark_route_views
        p._bind()
        self.assertIs(p.install_host_routing, lib.rdma_proxy_install_borrowed_benchmark_host_routing)
        self.assertFalse(p.borrow_source_route)

    def test_timed_windows_fork_after_r_ready_and_join_before_release(self):
        p = self.provider()
        events = []
        p.plan = object()
        p._generate = lambda i: events.append(("R", i))
        p._barrier = lambda: events.append("world")
        p._install = lambda i: events.append(("install", i))
        p.stream = SimpleNamespace(synchronize=lambda: events.append("R complete"))
        def wait():
            events.append("rme2 complete")
            return {"metadata_gpu_ms": 0.125}
        p.session = SimpleNamespace(submit=lambda i, plan, stream: events.append(("rme2", i)), wait=wait)
        def extents(handle, n, r):
            C.cast(n, C.POINTER(C.c_uint64))[0] = 18
            C.cast(r, C.POINTER(C.c_uint64))[0] = 4
            return 0
        p.library = SimpleNamespace(rdma_proxy_get_routing_extents=extents)
        def retire(handle, iteration):
            events.append(("retire", handle, iteration))
            return 0
        p.retire_routing = retire
        with patch.object(backend.time, "perf_counter_ns", side_effect=[100, 130, 145, 190, 215, 219]):
            self.assertEqual(p.prepare(0), ((18, 4), 119))
        self.assertEqual(events, [("R", 0), "R complete", "world", ("rme2", 1),
                                  "rme2 complete", ("install", 0), "world"])
        self.assertEqual(p.last_stats["r_ready_ns"], 130)
        self.assertEqual(p.last_stats["exchange_staging_elapsed_ns"], 45)
        self.assertEqual(p.last_stats["install_elapsed_ns"], 25)
        self.assertEqual(p.last_stats["r_ready_to_install_elapsed_ns"], 85)
        self.assertEqual(p.last_stats["reuse_barrier_ns"], 15)
        self.assertEqual(p.last_stats["retire_elapsed_ns"], 0)
        self.assertEqual(p.last_stats["release_barrier_ns"], 4)
        self.assertEqual(p.last_stats["rme2_stats"]["metadata_gpu_ms"], 0.125)
        self.assertEqual(p.last_stats["source_route_storage"], "copied")
        self.assertEqual(p.last_stats["a_idx_storage"], "host_upload")
        with self.assertRaisesRegex(RuntimeError, "consecutive"):
            p.prepare(0)
        events.clear()
        p.borrow_source_route = True
        p.source_snapshot_view = MagicMock()
        p.device_indices = backend.BenchmarkDeviceIndices()
        with patch.object(backend.time, "perf_counter_ns", side_effect=[220, 235, 240, 250, 272, 292, 300]):
            self.assertEqual(p.prepare(1), ((18, 4), 80))
        self.assertEqual(events, [("R", 1), "R complete", ("retire", 3, 0), "world", ("rme2", 2),
                                  "rme2 complete", ("install", 1), "world"])
        self.assertEqual(p.last_stats["retire_elapsed_ns"], 5)
        self.assertEqual(p.last_stats["reuse_barrier_ns"], 10)
        self.assertEqual(p.last_stats["source_route_storage"], "borrowed")
        self.assertEqual(p.last_stats["a_idx_storage"], "exchange_device")

    def test_failed_retirement_prevents_reuse(self):
        p = self.provider()
        p.next_iteration = 1
        p._generate, p.stream, p._barrier, p.session = MagicMock(), MagicMock(), MagicMock(), MagicMock()
        p.retire_routing = MagicMock(return_value=-1)
        p.library = SimpleNamespace(rdma_proxy_last_error=lambda: b"payload still active")
        with self.assertRaisesRegex(RuntimeError, "payload still active"):
            p.prepare(1)
        p.retire_routing.assert_called_once_with(p.handle, 0)
        p._barrier.assert_not_called()
        p.session.submit.assert_not_called()
        self.assertTrue(p.failed)

    def test_borrowed_default_group_survives_clean_disconnect(self):
        p = self.provider()
        dist = MagicMock()
        dist.is_initialized.return_value = True
        dist.get_world_size.return_value = 16
        dist.get_rank.return_value = 9
        p._open_group(dist)
        self.assertEqual((p.node, p.local), (1, 1))
        dist.init_process_group.assert_not_called()
        self.assertIs(p.group, dist.new_group.return_value)
        events = []
        p._barrier = lambda: events.append("world")
        p.session = SimpleNamespace(disconnect=lambda: events.append("disconnect"), close=lambda: events.append("free"))
        p.connected = p.exposed = True
        p.close()
        p.close()
        self.assertEqual(events, ["world", "disconnect", "world", "free"])
        dist.destroy_process_group.assert_called_once_with(dist.new_group.return_value)

    def test_communication_only_installs_epoch_without_querying_compute_extents(self):
        p = self.provider()
        p.plan, p.stream, p.session, p.library = object(), MagicMock(), MagicMock(), MagicMock()
        p._generate, p._barrier, p._install = MagicMock(), MagicMock(), MagicMock()
        extents, elapsed = p.prepare(0, compute_extents=False)
        self.assertIsNone(extents)
        self.assertGreaterEqual(elapsed, 0)
        p._install.assert_called_once_with(0)
        p.library.rdma_proxy_get_routing_extents.assert_not_called()

    def test_failed_peer_disconnect_never_frees_export(self):
        p = self.provider()
        p.exposed = p.connected = True
        p.session = MagicMock()
        p.session.lib.rme2_abort.return_value = 0
        p._barrier = MagicMock(side_effect=[None, RuntimeError("lost peer")])
        with patch.object(backend, "_RETAINED", []):
            with self.assertRaisesRegex(RuntimeError, "lost peer"):
                p.close()
            self.assertIn(p, backend._RETAINED)
        p.session.close.assert_not_called()
        p.session.lib.rme2_abort.assert_called_once_with(p.session.handle)


if __name__ == "__main__":
    unittest.main()
