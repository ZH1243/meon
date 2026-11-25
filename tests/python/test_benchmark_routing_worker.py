"""Worker handoff from the benchmark's per-epoch metadata to the production proxy."""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import rdma_proxy_worker as worker


class BenchmarkRoutingWorkerTests(unittest.TestCase):
    def test_shorthand_and_native_cli(self):
        environment = dict(LOCAL_RANK="0", LOCAL_WORLD_SIZE="1", RANK="0", WORLD_SIZE="1", GROUP_RANK="0")
        base = ["worker", "--dry-run", "--library=unused", "--cuda-device-map=0", "--rdma-device-map=unused"]
        for flags, expected in (
            (["--benchmark-metadata-exchange"], ["--router_dynamic_routing_enabled=true",
                                                 "--router_benchmark_metadata_enabled=true"]),
            (["--no-benchmark-metadata-exchange", "--dynamic-routing"],
             ["--router_dynamic_routing_enabled=true", "--router_benchmark_metadata_enabled=false"]),
            (["--communication-only", "--router_benchmark_metadata_enabled=true", "--dynamic-routing"],
             ["--router_dynamic_routing_enabled=true", "--router_benchmark_metadata_enabled=true"]),
            ([], []),
        ):
            output = io.StringIO()
            with self.subTest(flags=flags), patch.dict(os.environ, environment), patch.object(sys, "argv", base + flags):
                with contextlib.redirect_stdout(output):
                    self.assertEqual(worker.main(), 0)
                derived = output.getvalue().splitlines()[-1].split()
                observed = [arg for arg in derived if arg.startswith(("--router_dynamic_routing_enabled=",
                                                                       "--router_benchmark_metadata_enabled="))]
                self.assertCountEqual(observed, expected)

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

    def run_worker(self, *, compute=False, fail_epoch=None):
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
            rdma_proxy_get_num_iterations=write(2), rdma_proxy_get_computation_num_tokens=write(1),
            rdma_proxy_prepare_iteration=status("prepare"), rdma_proxy_run_iteration=status("payload"),
            rdma_proxy_record_computation_end=status("up_end"), rdma_proxy_get_computation_elapsed_ms=write(1.0),
            rdma_proxy_finish=status("finish"), rdma_proxy_shutdown=status("shutdown"), rdma_proxy_destroy=status("destroy"),
        )
        buffers = SimpleNamespace(finish_initialization=lambda: events.append(("buffers",)), tensors_of_kind=lambda _: [])
        test = self

        class Provider:
            def __init__(self, lib, handle, torch, device_buffers):
                test.assertIs(device_buffers, buffers)
                events.append(("provider",))

            def prepare(self, epoch, *, compute_extents=True):
                test.assertEqual(compute_extents, compute)
                events.append(("install", epoch))
                if epoch == fail_epoch:
                    raise RuntimeError("injected exchange failure")
                self.last_stats = dict(r_ready_ns=time.perf_counter_ns(), generation_elapsed_ns=1000,
                                       exchange_staging_elapsed_ns=2000, install_elapsed_ns=3000,
                                       r_ready_to_install_elapsed_ns=5000, reuse_barrier_ns=0, release_barrier_ns=0)
                return ((1, 1 - epoch) if compute_extents else None), 6000

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

        stream = SimpleNamespace(cuda_stream=42, synchronize=lambda: None)
        torch = SimpleNamespace(cuda=SimpleNamespace(
            get_device_capability=lambda _: (9, 0), Stream=lambda **_: stream,
            Event=lambda **_: SimpleNamespace(record=lambda _: None, synchronize=lambda: None),
            stream=lambda _: contextlib.nullcontext(), synchronize=lambda _: None))
        options = SimpleNamespace(pooled_up_gemm=False, down_projection2=False, benchmark_ready_up_gemm=False,
                                  check_projections=False, trace_down_ready=False)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, benchmark_metadata_backend=SimpleNamespace(
                BenchmarkMetadataBackend=Provider)))
            stack.enter_context(patch.object(worker, "install_python_device_buffers", return_value=buffers))
            stack.enter_context(patch.object(worker, "import_torch_for_proxy", return_value=torch))
            stack.enter_context(patch.object(worker, "GroupedGemmRunner", Up))
            stack.enter_context(patch.object(worker, "get_expert_counts", return_value=[0]))
            stack.enter_context(contextlib.redirect_stdout(output))
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

    def test_partial_exchange_failure_stops_before_payload_and_closes_provider(self):
        events, _ = self.run_worker(fail_epoch=1)
        self.assertEqual(events[-4:], [("install", 1), ("shutdown",), ("close", False), ("destroy",)])
        self.assertNotIn(("prepare", 1), events)


if __name__ == "__main__":
    unittest.main()
