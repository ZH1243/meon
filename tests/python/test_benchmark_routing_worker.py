"""Worker handoff from the benchmark's per-epoch metadata to the production proxy."""
import contextlib
import ctypes as C
import io
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import rdma_proxy_worker as worker


class BenchmarkRoutingWorkerTests(unittest.TestCase):
    def test_index_allocator_returns_the_exchange_slot_for_each_global_source(self):
        capacity = 80
        arrays = [SimpleNamespace(address=2**40 + rank * 4096, capacity=capacity) for rank in range(16)]
        for array in arrays:
            array.tensor = SimpleNamespace(
                data_ptr=lambda array=array: array.address, dtype="int32", device="cuda:3",
                shape=(capacity,), stride=lambda: (1,), numel=lambda: capacity, element_size=lambda: 4)
        torch = SimpleNamespace(as_tensor=lambda array: array.tensor, device=lambda typ, idx: f"{typ}:{idx}",
                                bfloat16="bf16", float16="fp16", float32="fp32", int32="int32")
        buffers = worker.TorchDeviceBuffers(torch, 3, node_rank=1, local_gpu_index=3, num_gpus_per_node=8)
        arrays[-1].tensor.device = "cuda:0"
        with self.assertRaisesRegex(RuntimeError, "source 15"):
            buffers.bind_benchmark_index_arrays(arrays, 8, capacity)
        self.assertFalse(buffers.benchmark_index_views)
        arrays[-1].tensor.device = "cuda:3"
        buffers.bind_benchmark_index_arrays(arrays, 8, capacity)
        for rank in reversed(range(16)):
            node, gpu = divmod(rank, 8)
            key = worker.DEVICE_BUFFER_ROUTER_A_IDX, -1, node, gpu
            request = worker.DeviceBufferRequest(
                C.sizeof(worker.DeviceBufferRequest), 1, worker.DEVICE_BUFFER_ROUTER_A_IDX,
                worker.DEVICE_BUFFER_INT32, -1, node, gpu, 1, capacity * 4, (C.c_uint64 * 2)(capacity, 0))
            self.assertEqual(buffers.callback(None, C.pointer(request)), arrays[rank].address)
            self.assertIs(buffers.tensors[key], arrays[rank].tensor)
        with self.assertRaisesRegex(RuntimeError, "before the first"):
            buffers.bind_benchmark_index_arrays(arrays, 8, capacity)

    def test_shorthand_and_native_cli(self):
        environment = dict(LOCAL_RANK="0", LOCAL_WORLD_SIZE="1", RANK="0", WORLD_SIZE="1", GROUP_RANK="0")
        base = ["worker", "--dry-run", "--library=unused", "--cuda-device-map=0", "--rdma-device-map=unused"]
        for flags, expected in (
            (["--benchmark-metadata-exchange"], ["--router_dynamic_routing_enabled=true",
                                                 "--router_benchmark_metadata_enabled=true"]),
            (["--no-benchmark-metadata-exchange", "--dynamic-routing"],
             ["--router_dynamic_routing_enabled=true", "--router_benchmark_metadata_enabled=false"]),
            (["--communication-only", "--router_benchmark_metadata_enabled=true", "--dynamic-routing"],
             ["--router_benchmark_metadata_enabled=true", "--router_dynamic_routing_enabled=true"]),
            (["--benchmark-metadata-exchange", "--no-benchmark-metadata-validation"],
             ["--router_dynamic_routing_enabled=true", "--router_benchmark_metadata_enabled=true",
              "--router_benchmark_metadata_validation_enabled=false"]),
            (["--no-benchmark-metadata-validation", "--benchmark-metadata-validation"],
             ["--router_benchmark_metadata_validation_enabled=true"]),
            (["--router_benchmark_metadata_validation_enabled=false"],
             ["--router_benchmark_metadata_validation_enabled=false"]),
            (["--no-benchmark-metadata-validation", "--router_benchmark_metadata_validation_enabled=true"],
             ["--router_benchmark_metadata_validation_enabled=true",
              "--router_benchmark_metadata_validation_enabled=false"]),
            ([], []),
        ):
            output = io.StringIO()
            with self.subTest(flags=flags), patch.dict(os.environ, environment), patch.object(sys, "argv", base + flags):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(worker.main(), 0)
                derived = output.getvalue().splitlines()[-1].split()
                observed = [arg for arg in derived if arg.startswith(("--router_dynamic_routing_enabled=",
                                                                       "--router_benchmark_metadata_enabled=",
                                                                       "--router_benchmark_metadata_validation_enabled="))]
                self.assertEqual(observed, expected)

    def test_legacy_library_rejects_enabled_config_and_respects_override(self):
        lib = SimpleNamespace()
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "test.json"
            config.write_text('{"router_benchmark_metadata_enabled": true}')
            argv = ["worker", "--config", str(config)]
            with self.assertRaisesRegex(RuntimeError, "benchmark metadata exchange requires rebuilding"):
                worker.open_benchmark_metadata(lib, 1, argv, None, None)
            self.assertIsNone(worker.open_benchmark_metadata(
                lib, 1, [*argv, "--router_benchmark_metadata_enabled=false"], None, None))

    def test_staging_uses_the_local_source_including_nonzero_rank(self):
        buffers = worker.TorchDeviceBuffers.__new__(worker.TorchDeviceBuffers)
        buffers.node_rank, buffers.local_gpu_index = 1, 5
        source, destination = object(), object()
        buffers.tensors = {(worker.DEVICE_BUFFER_RDMA_SEND, -1, -1, -1): source,
                           (worker.DEVICE_BUFFER_NVLINK_RECEIVE, -1, 1, 5): destination,
                           (worker.DEVICE_BUFFER_NVLINK_RECEIVE, -1, 0, 5): object(),
                           (worker.DEVICE_BUFFER_NVLINK_RECEIVE, -1, 1, 0): object()}
        self.assertEqual(buffers.benchmark_staging_buffers(), (source, destination))

    def run_worker(self, *, compute=False, fail_epoch=None, scatter_publication=None,
                   tile_major=True, has_reservation_api=True, rows_per_epoch=(1, 0), down_factory=None,
                   native_returns=False, early=False, fail_prepare=False):
        events, output = [], io.StringIO()

        def status(name):
            def status_call(*args):
                events.append((name, *args[1:]))
                return 0
            return status_call

        def write(value):
            def write_pointer(*args):
                args[-1]._obj.value = value
                return 0
            return write_pointer

        library = SimpleNamespace(
            rdma_proxy_create=lambda *args: 99, rdma_proxy_initialize=status("init"),
            rdma_proxy_benchmark_metadata_enabled=write(True),
            rdma_proxy_get_num_iterations=write(len(rows_per_epoch)),
            rdma_proxy_get_computation_num_tokens=write(rows_per_epoch[0]),
            rdma_proxy_prepare_iteration=status("prepare"), rdma_proxy_run_iteration=status("payload"),
            rdma_proxy_record_computation_end=status("up_end"), rdma_proxy_get_computation_elapsed_ms=write(1.0),
            rdma_proxy_finish=status("finish"), rdma_proxy_shutdown=status("shutdown"), rdma_proxy_destroy=status("destroy"),
        )
        if early:
            def arm(handle, epoch, stream, armed):
                events.append(("early-reset", epoch, stream))
                armed._obj.value = 1
                return 0
            library.rdma_proxy_prepare_early_up = arm
            library.rdma_proxy_cancel_early_up = status("cancel")
        if fail_prepare:
            def prepare(handle, epoch):
                events.append(("prepare", epoch))
                if epoch:
                    raise RuntimeError("injected preparation failure")
                return 0
            library.rdma_proxy_prepare_iteration = prepare
        if has_reservation_api:
            library.rdma_proxy_reserve_threaded_cpu_scatter = status("reserve_scatter")
        buffers = SimpleNamespace(finish_initialization=lambda: events.append(("buffers",)), tensors_of_kind=lambda _: [])
        test = self

        class Provider:
            def __init__(self, lib, handle, torch, device_buffers, *, concurrent_compute):
                test.assertIs(device_buffers, buffers)
                test.assertEqual(concurrent_compute, compute)
                events.append(("provider",))

            def prepare(self, epoch, *, compute_extents=True):
                test.assertEqual(compute_extents, compute)
                events.append(("install", epoch))
                if epoch == fail_epoch:
                    raise RuntimeError("injected exchange failure")
                self.last_stats = dict(r_ready_ns=time.perf_counter_ns(), generation_elapsed_ns=1000,
                                       exchange_staging_elapsed_ns=2000, install_elapsed_ns=3000,
                                       r_ready_to_install_elapsed_ns=5000, retire_elapsed_ns=100,
                                       reuse_barrier_ns=0, release_barrier_ns=0, source_route="receiver",
                                       source_route_storage="borrowed", a_idx_storage="exchange_device")
                return ((1, rows_per_epoch[epoch]) if compute_extents else None), 6000

            def close(self, *, clean):
                events.append(("close", clean))

        class Up:
            x_buffers = (SimpleNamespace(shape=(1, 8)),)
            gemm_dimension = 4

            def __init__(self, *args, routing_extents, **kwargs):
                self.bind_routing(*routing_extents)

            def bind_routing(self, source_rows, rows):
                self.empty_epoch = not rows
                events.append(("bind", rows))

            def launch(self):
                if not self.empty_epoch:
                    events.append(("up",))

        stream = SimpleNamespace(cuda_stream=42, synchronize=lambda: events.append(("join-stream",)),
                                 wait_stream=lambda other: events.append(("wait-setup",)))
        torch = SimpleNamespace(cuda=SimpleNamespace(
            get_device_capability=lambda _: (9, 0), Stream=lambda **_: stream,
            Event=lambda **_: SimpleNamespace(record=lambda _: None, synchronize=lambda: None),
            stream=lambda _: contextlib.nullcontext(), current_stream=lambda _: stream,
            synchronize=lambda _: events.append(("device-sync",))))
        options = SimpleNamespace(pooled_up_gemm=False, down_projection2=scatter_publication is not None,
                                  cpu_scatter_preparation=scatter_publication is not None,
                                  cpu_scatter_publication=scatter_publication, down_n_tile_major_output=tile_major,
                                  down_pingpong=None, down_gather_table_wave_barrier=True,
                                  down_gather_table=True, down_gather_table_expert_waves=True,
                                  down_gather_table_publish_ready=True, return_token_results=native_returns,
                                  intra_node_combine=native_returns, inter_node_combine=False,
                                  return_token_timeout=2,
                                  benchmark_ready_up_gemm=False, check_projections=False, trace_down_ready=False)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, benchmark_metadata_backend=SimpleNamespace(
                BenchmarkMetadataBackend=Provider)))
            stack.enter_context(patch.object(worker, "install_python_device_buffers", return_value=buffers))
            stack.enter_context(patch.object(worker, "import_torch_for_proxy", return_value=torch))
            stack.enter_context(patch.object(worker, "GroupedGemmRunner", Up))
            stack.enter_context(patch.object(worker, "get_expert_counts",
                                             side_effect=([rows] for rows in rows_per_epoch)))
            if down_factory is not None:
                stack.enter_context(patch.object(worker, "DownProjectionRunner", down_factory(events)))
            if native_returns:
                maps, exchange, intra = Mock(), Mock(), Mock(inter_node_combine=None)
                maps.start.side_effect = lambda epoch: events.append(("start-maps", epoch))
                exchange.arm_and_bind_maps.side_effect = lambda epoch: events.append(("arm-maps", epoch))
                exchange.arm.side_effect = lambda epoch: events.append(("arm", epoch))
                exchange.bind_down.side_effect = lambda down: setattr(down, "intra_node_combine", intra)
                stack.enter_context(patch.object(worker, "DownReadyTrace"))
                stack.enter_context(patch.object(worker.AsyncScatterLookup, "reserve_if_supported", return_value=Mock()))
                stack.enter_context(patch("return_maps.ReturnMaps.reserve_if_supported", return_value=maps))
                stack.enter_context(patch("rdma_return_exchange.ReturnExchange", return_value=exchange))
                stack.enter_context(patch("intra_node_combine.IntraNodeCombine", return_value=intra))
            stack.enter_context(contextlib.redirect_stdout(output))
            if fail_prepare:
                stack.enter_context(self.assertRaisesRegex(RuntimeError, "injected preparation failure"))
            if fail_epoch is not None:
                stack.enter_context(self.assertRaisesRegex(RuntimeError, "injected exchange failure"))
            if compute:
                worker.run_proxy_with_concurrent_kernel(
                    library, ["worker"], 0, node_rank=0, local_gpu_index=0, world_size=16, num_experts=128,
                    expert_gemm_dimension=4, activation=None, expert_gemm_m_tile=128, expert_gemm_n_tile=128,
                    expert_gemm_k_tile=None, expert_gemm_cluster_m=2, expert_gemm_max_swizzle_size=1,
                    quack_root=Path("."), down_options=options)
            else:
                worker.run_proxy(library, ["worker"], 0, node_rank=0, local_gpu_index=0)
        return events, output.getvalue()

    def test_communication_only_prepares_every_epoch(self):
        events, result = self.run_worker()
        self.assertEqual(events, [("init",), ("buffers",), ("provider",),
                                  ("install", 0), ("prepare", 0), ("payload", 0),
                                  ("install", 1), ("prepare", 1), ("payload", 1), ("finish",),
                                  ("shutdown",), ("close", True), ("destroy",)])
        self.assertIn("phase=steady generation_us=1.000 exchange_staging_us=2.000 install_us=3.000", result)
        self.assertIn("retire_us=0.100", result)
        self.assertEqual(result.count("release_barrier_us=0.000"), 2)
        self.assertIn("source_route=receiver source_route_storage=borrowed", result)
        self.assertIn("a_idx_storage=exchange_device", result)
        self.assertNotIn("r_ready_to_up_submit_us", result)

    def test_up_uses_current_route_and_timing_is_only_for_actual_launches(self):
        events, result = self.run_worker(compute=True)
        for epoch in (0, 1):
            install = events.index(("install", epoch))
            bind = events.index(("bind", 1 - epoch))
            prepare = events.index(("prepare", epoch))
            self.assertLess(install, bind)
            self.assertLess(bind, prepare)
            self.assertLess(prepare, events.index(("payload", epoch)))
        self.assertLess(events.index(("prepare", 0)), events.index(("up",)))
        self.assertEqual(events[-3:], [("shutdown",), ("close", True), ("destroy",)])
        records = [line for line in result.splitlines() if line.strip().startswith("benchmark_metadata ")]
        self.assertIn("r_ready_to_up_submit_us=", records[0])
        self.assertNotIn("r_ready_to_up_submit_us=", records[1])

    def test_steady_down_schedule_orders_construction_publication_and_consumption(self):
        def down_factory(events):
            prepared_options = SimpleNamespace(external_scatter_addresses=True)

            def create(torch, up, counts, options, *, defer_wave_schedule=False, external_wave_schedule=None,
                       previous=None, kernel_options=None, **kwargs):
                self.assertIs(kernel_options, prepared_options if previous is not None else None)
                rows, = counts
                clock = Mock()
                clock.elapsed_time.return_value = 1.0
                down = SimpleNamespace(
                    cpu_scatter=bool(rows), cpu_scatter_publication="threaded", combine=True, combine_rows=4,
                    route_weights=None, kernel_options=prepared_options,
                    output_reset_policy="before-down", intra_node_combine=None, flops=2 * rows,
                    reset_start=clock, reset_end=clock, mapping_start=clock, gemm_start=clock, end=clock,
                    warmup=lambda: None,
                    configure_cpu_scatter=lambda *_: events.append(("scatter", rows)),
                    wait_cpu_scatter_ready=lambda epoch: events.append(("wait-scatter", epoch)),
                )
                if not defer_wave_schedule:
                    events.append(("schedule", rows))

                def launch(wait_for_metadata, wait_for_wave_schedule, before_combine):
                    if wait_for_metadata:
                        wait_for_metadata()
                    if defer_wave_schedule and rows and external_wave_schedule is None:
                        events.append(("schedule", rows))
                    if wait_for_wave_schedule:
                        self.assertEqual(external_wave_schedule, ("table", rows))
                        wait_for_wave_schedule()
                    events.append(("down", rows))
                    if before_combine is not None:
                        before_combine()
                        events.append(("combine", rows))

                down.launch = launch
                return down
            return create

        rows_per_epoch = (1, 3, 0, 2)
        for native, native_returns, early in ((False, False, False), (True, False, False),
                                              (True, True, False), (True, True, True)):
            with self.subTest(native=native, native_returns=native_returns, early=early):
                def factory(events):
                    def submit(epoch, counts):
                        events.append(("schedule", counts[0]))
                        return ("table", counts[0])

                    fake.start = submit
                    fake.wait = lambda epoch, stream: events.append(("wait-schedule", epoch, stream.cuda_stream))
                    return down_factory(events)

                fake = SimpleNamespace()
                with patch("down_wave_schedule.DownWaveSchedule.reserve_if_supported",
                           return_value=fake if native else None) as reserve:
                    events, _ = self.run_worker(compute=True, scatter_publication="threaded",
                                                rows_per_epoch=rows_per_epoch, down_factory=factory,
                                                native_returns=native_returns, early=early)
                reserve.assert_called_once()
                for epoch, rows in enumerate(rows_per_epoch):
                    start = events.index(("install", epoch))
                    if native_returns and epoch:
                        map_start = events.index(("start-maps", epoch))
                        down = events.index(("down", rows))
                        map_wait = events.index(("arm-maps", epoch))
                        self.assertLess(start, map_start)
                        self.assertLess(map_start, events.index(("prepare", epoch)))
                        self.assertLess(down, map_wait)
                        self.assertLess(map_wait, events.index(("combine", rows)))
                        self.assertNotIn(("arm", epoch), events)
                    elif native_returns:
                        self.assertNotIn(("start-maps", epoch), events)
                        self.assertIn(("arm", epoch), events)
                    if not rows:
                        self.assertNotIn(("schedule", 0), events)
                        self.assertNotIn(("wait-schedule", epoch, 42), events)
                        continue
                    up = events.index(("up",), start)
                    schedule = events.index(("schedule", rows))
                    prepared = events.index(("prepare", epoch))
                    self.assertLess(events.index(("scatter", rows)), prepared)
                    if early and epoch:
                        reset = events.index(("early-reset", epoch, 42))
                        self.assertLess(reset, up)
                        self.assertLess(up, map_start)
                        self.assertLess(up, events.index(("scatter", rows)))
                        self.assertLess(up, prepared)
                        self.assertNotIn(("device-sync",), events[up:prepared])
                    else:
                        self.assertLess(prepared, up)
                    wait = events.index(("wait-scatter", epoch))
                    self.assertEqual(events.count(("wait-scatter", epoch)), 1)
                    self.assertLess(up, wait)
                    consumed = events.index(("down", rows))
                    if native and epoch:
                        self.assertLess(start, schedule)
                        if early:
                            self.assertLess(up, schedule)
                        else:
                            self.assertLess(schedule, events.index(("bind", rows)))
                        self.assertLess(schedule, events.index(("scatter", rows)))
                        join = events.index(("wait-schedule", epoch, 42))
                        self.assertLess(wait, join)
                        self.assertLess(join, consumed)
                    elif epoch:
                        self.assertLess(wait, schedule)
                    else:
                        self.assertLess(schedule, up)
                    self.assertLess(schedule, consumed)

    def test_blocking_cuda_launch_releases_the_coordinator_first(self):
        with patch.dict(os.environ, CUDA_LAUNCH_BLOCKING="1"):
            events, _ = self.run_worker(compute=True, early=True, rows_per_epoch=(1, 2))
        self.assertNotIn(("early-reset", 1, 42), events)
        second = events.index(("bind", 2))
        self.assertLess(events.index(("prepare", 1)), events.index(("up",), second))

    def test_preparation_failure_drains_early_up_before_join(self):
        events, _ = self.run_worker(compute=True, early=True, rows_per_epoch=(1, 2), fail_prepare=True)
        cancel = events.index(("cancel", 1))
        self.assertLess(events.index(("up",), events.index(("early-reset", 1, 42))), cancel)
        self.assertLess(cancel, events.index(("join-stream",)))
        self.assertNotIn(("payload", 1), events)

    def test_partial_exchange_failure_stops_before_payload_and_closes_provider(self):
        events, _ = self.run_worker(fail_epoch=1)
        self.assertEqual(events[-4:], [("install", 1), ("shutdown",), ("close", False), ("destroy",)])
        self.assertNotIn(("prepare", 1), events)

    def test_interrupted_payload_shutdown_keeps_borrowed_storage_alive(self):
        library, provider = Mock(), Mock()
        library.rdma_proxy_shutdown.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            worker.shutdown_embedded_proxy(library, 1, None, True, True, benchmark_metadata=provider)
        provider.close.assert_called_once_with(clean=False)
        library.rdma_proxy_destroy.assert_called_once_with(1)

    def test_threaded_scatter_reserves_before_first_route_counts_are_available(self):
        for mode, tiled, has_api, expected in (
                ("threaded", True, True, [("reserve_scatter", 2)]),
                ("threaded", False, True, [("reserve_scatter", 1)]),
                ("threaded", True, False, []),
                ("deferred", True, True, []),
                ("per-batch", True, True, [])):
            with self.subTest(mode=mode, tiled=tiled, has_api=has_api):
                events, _ = self.run_worker(compute=True, fail_epoch=0, scatter_publication=mode,
                                            tile_major=tiled, has_reservation_api=has_api)
                before_route = events[:events.index(("install", 0))]
                self.assertEqual(before_route, [("init",), ("buffers",), ("provider",), *expected])


if __name__ == "__main__":
    unittest.main()
