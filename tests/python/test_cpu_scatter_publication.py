"""CPU coverage for deferred/threaded metadata layout and option validation.

GPU publication ordering is exercised by test_worker_composition.py; the actual
native event generations and H2D submission are covered by test_cpu_scatter.
"""

import argparse
from contextlib import contextmanager, nullcontext
import ctypes
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import rdma_proxy_worker as worker
from down_wave_schedule import DownWaveGeometry, DownWaveSchedule


class PublicationOptionsTests(unittest.TestCase):
    def parse(self, *flags):
        parser = argparse.ArgumentParser()
        worker.add_down_arguments(parser)
        args = parser.parse_args(flags)
        args.concurrent_kernel = True
        args.grouped_gemm_activation = "swiglu"
        return args

    def test_defaults_and_explicit_modes(self):
        common = ("--cpu-scatter-preparation", "--local-gpu-combine", "--down-projection2")
        defaults = self.parse(*common)
        self.assertIsNone(defaults.cpu_scatter_publication)
        worker.validate_down_arguments(defaults)
        for mode in ("deferred", "per-batch", "threaded"):
            with self.subTest(mode=mode):
                args = self.parse(*common, f"--cpu-scatter-publication={mode}")
                worker.validate_down_arguments(args)
                self.assertEqual(args.cpu_scatter_publication, mode)

    def test_explicit_mode_requires_cpu_preparation(self):
        for mode in ("deferred", "per-batch", "threaded"):
            args = self.parse("--local-gpu-combine", "--down-projection2",
                              f"--cpu-scatter-publication={mode}")
            with self.assertRaisesRegex(ValueError, "cpu-scatter-preparation"):
                worker.validate_down_arguments(args)

    def test_submission_wait_has_finite_positive_timeout(self):
        common = ("--cpu-scatter-preparation", "--local-gpu-combine", "--down-projection2")
        for timeout in ("0", "-1", "nan", "inf", "1e20"):
            with self.subTest(timeout=timeout):
                args = self.parse(*common, f"--return-token-timeout={timeout}")
                with self.assertRaisesRegex(ValueError, "finite.*positive"):
                    worker.validate_down_arguments(args)
        args = self.parse(*common, "--return-token-timeout=0.125")
        worker.validate_down_arguments(args)


class MetadataSlabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        cls.torch = torch

    def test_async_lookup_publishes_host_counts_to_return_owner(self):
        torch = self.torch
        for checked in (False, True):
            with self.subTest(checked=checked):
                opt = SimpleNamespace(cpu_scatter_preparation=True, cpu_scatter_publication="threaded",
                                      check_intra_node_combine=checked, check_projections=False)
                up = SimpleNamespace(torch=torch, source_pairs=[(0, 0), (0, 1)], output=torch.empty(0),
                                     x_buffers=[torch.empty(3, 4)], cuda_device_id=0)
                outputs = [torch.empty(3, 4) for _ in up.source_pairs]
                down = worker.DownProjectionRunner.__new__(worker.DownProjectionRunner)
                down.torch, down.up, down.combine_rows = torch, up, 3
                down.combine_tile_n, down.routes = None, 2
                down.inputs, down.registered_outputs = None, outputs
                down.cpu_scatter_metadata = torch.empty(2 * 8, dtype=torch.uint8)
                down.route_weights, down.cpu_scatter_publication = None, "threaded"
                exchange = SimpleNamespace(rows_host=torch.full((2,), -1, dtype=torch.int64))
                lib = SimpleNamespace(rdma_proxy_configure_cpu_scatter=mock.Mock(),
                                      rdma_proxy_configure_cpu_scatter_threaded=mock.Mock(),
                                      rdma_proxy_wait_cpu_scatter_ready=mock.Mock(return_value=0))
                self.assertIsNone(worker.AsyncScatterLookup.reserve_if_supported(lib, down, opt, exchange))

                def submit(handle, config, weights, slab_bytes):
                    c = ctypes.cast(config, ctypes.POINTER(worker.CpuScatterConfig)).contents
                    self.assertEqual((slab_bytes, weights, c.routes), (16, 0, 2))
                    self.assertEqual(c.scatter_addresses, down.cpu_scatter_metadata.data_ptr())
                    self.assertEqual([c.output_bases[i] for i in range(2)], [t.data_ptr() for t in outputs])
                    self.assertEqual(ctypes.cast(c.counts, ctypes.c_void_p).value, exchange.rows_host.data_ptr())
                    self.assertEqual(bool(c.output_to_input), checked)
                    if checked:
                        self.assertEqual(ctypes.cast(c.output_to_input, ctypes.c_void_p).value,
                                         down.async_scatter.inverse.data_ptr())
                    return 0

                lib.rdma_proxy_configure_cpu_scatter_async = mock.Mock(side_effect=submit)
                down.async_scatter = worker.AsyncScatterLookup.reserve_if_supported(lib, down, opt, exchange)
                with mock.patch.object(torch.cuda, "synchronize", side_effect=AssertionError("unexpected CUDA wait")):
                    down.configure_cpu_scatter(lib, 3)
                self.assertIs(down.cpu_scatter_counts, exchange.rows_host)
                self.assertEqual(exchange.rows_host.tolist(), [-1, -1])
                lib.rdma_proxy_configure_cpu_scatter_async.assert_called_once()
                lib.rdma_proxy_configure_cpu_scatter.assert_not_called()
                lib.rdma_proxy_configure_cpu_scatter_threaded.assert_not_called()
                lib.rdma_proxy_wait_cpu_scatter_ready.assert_not_called()

    def test_native_wave_slab_reuses_storage_with_exact_table_and_prefix_views(self):
        torch = self.torch
        device = torch.device("cpu")
        up = SimpleNamespace(output=torch.empty(1), x_buffers=(torch.empty(0, 7168),),
                             device_capacity=9, cuda_device_id=3)
        options = SimpleNamespace(cpu_scatter_preparation=True, cpu_scatter_publication="threaded",
                                  down_gather_table_publish_ready=True, down_cluster_m=1, down_tile_m=192,
                                  down_tile_n=128, down_max_swizzle_size=2, down_sm_count=124,
                                  return_token_timeout=0.125)
        lib = SimpleNamespace()
        check = lambda status, operation: self.assertEqual(status, 0, operation)
        reserve = lambda: DownWaveSchedule.reserve_if_supported(torch, lib, 1, up, options, check)
        self.assertIsNone(reserve())  # ABI-v1 library predating the schedule functions.

        def capacity(handle, ptr, words):
            geom = ctypes.cast(ptr, ctypes.POINTER(DownWaveGeometry)).contents
            self.assertEqual((geom.struct_size, geom.cluster_rows, geom.num_groups, geom.group_size,
                              geom.max_wave_work_ids), (ctypes.sizeof(DownWaveGeometry), 192, 28, 2, 124))
            words._obj.value = 100
            return 0

        def start(handle, epoch, counts, experts, rows, prefixes):
            self.assertEqual(tuple(counts[:experts]), (0, epoch))
            rows._obj.value = epoch + 1
            prefixes._obj.value = epoch
            return 0

        lib.rdma_proxy_get_down_wave_capacity = mock.Mock(side_effect=capacity)
        lib.rdma_proxy_configure_down_wave_schedule = mock.Mock(return_value=0)
        lib.rdma_proxy_start_down_wave_schedule = mock.Mock(side_effect=start)
        lib.rdma_proxy_wait_down_wave_schedule = mock.Mock(return_value=0)
        pregather = SimpleNamespace(prepare_grouped_inputs=lambda *, scatter_row_addresses: None)
        max_clusters = mock.Mock(return_value=124)
        with mock.patch.dict(sys.modules, {
            "run": SimpleNamespace(hopper_pregather_gemm=pregather),
            "quack.cute_dsl_utils": SimpleNamespace(get_device_capacity=lambda _: (9, 0),
                                                    get_max_active_clusters=max_clusters),
            "quack.gemm_config": SimpleNamespace(cta_tile_shape_m=lambda tile, cluster, arch: tile),
        }):
            self.assertIsNone(reserve())  # QuACK with only external scatter support.
            pregather.prepare_grouped_inputs = lambda *, scatter_row_addresses, external_wave_schedule: None
            schedule = reserve()
        max_clusters.assert_called_once_with(1, device_capacity=(9, 0), device_id=3, sm_count=124)
        self.assertEqual(schedule.slab.device, device)
        self.assertEqual(schedule.slab.dtype, torch.int32)
        self.assertEqual(lib.rdma_proxy_configure_down_wave_schedule.call_args[0][2:],
                         (schedule.slab.data_ptr(), 100))
        stream = SimpleNamespace(cuda_stream=2**40)
        for epoch in (1, 4):
            table, prefix = schedule.start(epoch, [0, epoch])
            self.assertEqual(table.shape, (epoch + 1, 4))
            self.assertEqual(prefix.shape, (epoch,))
            self.assertEqual(table.data_ptr(), schedule.slab.data_ptr())
            self.assertEqual(prefix.data_ptr(), table.data_ptr() + table.numel() * 4)
            schedule.wait(epoch, stream)
            lib.rdma_proxy_wait_down_wave_schedule.assert_called_with(1, epoch, 2**40, 125)
        self.assertEqual(lib.rdma_proxy_configure_down_wave_schedule.call_count, 1)

    def test_byte_layout_alignment_and_single_copy_round_trip(self):
        torch = self.torch
        for routes in (1, 3, 4, 257):
            for columns in (1, 2):
                for weighted in (False, True):
                    with self.subTest(routes=routes, columns=columns, weighted=weighted):
                        shape = (routes,) if columns == 1 else (routes, columns)
                        addresses = torch.arange(routes * columns, dtype=torch.int64).reshape(shape)
                        addresses += 0x123456700000000
                        weights = (torch.arange(routes, dtype=torch.float32) / 8 if weighted else None)
                        slab, address_view, weight_view = worker.make_cpu_scatter_metadata_storage(
                            torch, addresses, weights)
                        self.assertEqual(slab.dtype, torch.uint8)
                        self.assertTrue(slab.is_contiguous())
                        self.assertEqual(address_view.dtype, torch.int64)
                        self.assertEqual(address_view.shape, addresses.shape)
                        self.assertEqual(address_view.data_ptr(), slab.data_ptr())
                        torch.testing.assert_close(address_view, addresses, rtol=0, atol=0)
                        address_bytes = addresses.numel() * addresses.element_size()
                        if weighted:
                            weight_offset = weight_view.data_ptr() - slab.data_ptr()
                            self.assertEqual(weight_offset % 16, 0)
                            self.assertGreaterEqual(weight_offset, address_bytes)
                            self.assertLess(weight_offset - address_bytes, 16)
                            self.assertEqual(weight_view.dtype, torch.float32)
                            self.assertEqual(weight_view.shape, weights.shape)
                            self.assertEqual(slab.numel(), weight_offset + routes * 4)
                            self.assertTrue(torch.equal(slab[address_bytes:weight_offset],
                                                        torch.zeros(weight_offset - address_bytes,
                                                                    dtype=torch.uint8)))
                            torch.testing.assert_close(weight_view, weights, rtol=0, atol=0)
                        else:
                            self.assertIsNone(weight_view)
                            self.assertEqual(slab.numel(), address_bytes)
                        # Distinct destination storage models one complete H2D
                        # upload. Consumers retain aliased views across iterations.
                        destination, copied_addresses, copied_weights = worker.make_cpu_scatter_metadata_storage(
                            torch, torch.zeros_like(addresses),
                            torch.zeros_like(weights) if weighted else None)
                        for iteration in range(3):
                            address_view.copy_(addresses + iteration * 4096)
                            if weighted:
                                weight_view.copy_(weights if iteration % 2 else torch.zeros_like(weights))
                            destination.fill_(255)
                            destination.copy_(slab)
                            torch.testing.assert_close(copied_addresses, address_view, rtol=0, atol=0)
                            if weighted:
                                torch.testing.assert_close(copied_weights, weight_view, rtol=0, atol=0)

    def test_invalid_storage_is_rejected_before_native_borrow(self):
        torch = self.torch
        for addresses in (torch.zeros(0, dtype=torch.int64), torch.zeros(3, dtype=torch.int32),
                          torch.zeros(3, 1, dtype=torch.int64), torch.zeros(3, 3, dtype=torch.int64),
                          torch.zeros(6, dtype=torch.int64)[::2]):
            with self.subTest(shape=addresses.shape, dtype=addresses.dtype):
                with self.assertRaisesRegex(ValueError, "contiguous int64"):
                    worker.make_cpu_scatter_metadata_storage(torch, addresses)
        addresses = torch.zeros(3, 2, dtype=torch.int64)
        for weights in (torch.ones(2), torch.ones(3, dtype=torch.float64), torch.ones(6)[::2]):
            with self.subTest(shape=weights.shape, dtype=weights.dtype):
                with self.assertRaisesRegex(ValueError, "contiguous FP32"):
                    worker.make_cpu_scatter_metadata_storage(torch, addresses, weights)

    def test_refresh_binds_quack_to_the_native_publication_slab(self):
        torch = self.torch
        parser = argparse.ArgumentParser()
        worker.add_down_arguments(parser)
        options = parser.parse_args(["--local-gpu-combine", "--cpu-scatter-preparation"])
        up = SimpleNamespace(
            output=torch.empty(8, 4), weights=torch.empty(2, 4, 4), device_capacity=9,
            x_buffers=(torch.empty(8, 4), torch.empty(8, 4)), gemm_dimension=4,
            tile_n=2, max_swizzle_size=1,
        )
        previous = mock.Mock()
        previous.inputs.output_buffers.return_value = up.x_buffers
        seen = []

        def prepare(args, a, w, counts, *, output_buffers, scatter_mapping=None,
                    scatter_row_addresses=None, route_weights=None):
            seen.append((scatter_mapping, scatter_row_addresses, route_weights))
            if scatter_row_addresses is None:
                self.assertEqual(scatter_mapping.shape, (sum(counts), 2))
                shape = (sum(counts), 2) if args.n_tile_major_output else (sum(counts),)
                scatter_row_addresses = torch.full(shape, 123, dtype=torch.int64)
            return SimpleNamespace(scatter_row_addresses=scatter_row_addresses,
                                   output_buffers=lambda: output_buffers)

        def legacy(args, a, w, counts, *, output_buffers, scatter_mapping, route_weights=None):
            return prepare(args, a, w, counts, output_buffers=output_buffers,
                           scatter_mapping=scatter_mapping, route_weights=route_weights)

        pregather = mock.Mock()
        with mock.patch.dict(sys.modules, {"run": SimpleNamespace(hopper_pregather_gemm=pregather),
                                           "quack.gemm": mock.Mock()}):
            cases = [
                ("threaded", prepare, True, False), ("threaded", prepare, True, True),
                ("threaded", prepare, False, True), ("threaded", legacy, True, False),
                ("deferred", prepare, True, True), ("per-batch", prepare, False, False),
            ]
            for mode, builder, tiled, weighted in cases:
                with self.subTest(mode=mode, builder=builder.__name__, tiled=tiled, weighted=weighted):
                    options.cpu_scatter_publication = mode
                    options.down_n_tile_major_output = tiled
                    options.weighted_contributions = weighted
                    options.down_gather_table = True
                    options.down_gather_table_expert_waves = True
                    options.down_gather_table_publish_ready = builder is legacy or mode != "threaded"
                    up.route_weights = (torch.empty(3),) if weighted else None
                    pregather.prepare_grouped_inputs = builder
                    seen.clear()
                    empty = worker.DownProjectionRunner(torch, up, [0, 0], options, previous=previous)
                    self.assertEqual(seen, [])
                    self.assertIsNone(empty.cpu_scatter_metadata)
                    # Missing wave publication or the legacy QuACK signature keeps setup eager.
                    runner = worker.DownProjectionRunner(torch, up, [2, 1], options, previous=empty,
                                                         defer_wave_schedule=True)
                    self.assertEqual(len(seen), 1)
                    mapping, supplied, weights = seen[0]
                    if mode == "threaded" and builder is prepare:
                        self.assertIsNone(mapping)
                        self.assertIs(supplied, runner.inputs.scatter_row_addresses)
                        self.assertEqual(supplied.shape, (3, 2) if tiled else (3,))
                        self.assertEqual(supplied.data_ptr(), runner.cpu_scatter_metadata.data_ptr())
                        self.assertIs(weights, runner.route_weights)
                        if weighted:
                            offset = weights.data_ptr() - supplied.data_ptr()
                            self.assertEqual(offset, (supplied.numel() * 8 + 15) // 16 * 16)
                            self.assertEqual(runner.cpu_scatter_metadata.numel(), offset + 3 * 4)
                    else:
                        self.assertIsNotNone(mapping)
                        self.assertIsNone(supplied)
                        self.assertTrue(torch.all(runner.inputs.scatter_row_addresses == 123).item())

    def test_wave_schedule_uses_current_counts_and_the_early_scatter_destination(self):
        torch = self.torch
        parser = argparse.ArgumentParser()
        worker.add_down_arguments(parser)
        options = parser.parse_args([
            "--local-gpu-combine", "--cpu-scatter-preparation", "--cpu-scatter-publication=threaded",
            "--down-gather-table", "--down-gather-table-expert-waves", "--down-gather-table-publish-ready",
            "--down-n-tile-major-output",
        ])
        options.weighted_contributions = True
        options.return_token_results = True
        sources = (torch.empty(8, 4), torch.empty(8, 4))
        up = SimpleNamespace(torch=torch, output=torch.empty(8, 4), weights=torch.empty(2, 4, 4),
                             device_capacity=9, x_buffers=sources, source_pairs=((0, 0), (0, 1)),
                             gemm_dimension=4, tile_n=2,
                             max_swizzle_size=1, route_weights=(torch.empty(8),), cuda_device_id=0)
        ready_address = torch.tensor([0x1234], dtype=torch.int64)
        submitted = []
        setup_stream = object()
        compute_stream = mock.Mock()
        active_stream = setup_stream

        @contextmanager
        def use_stream(stream):
            nonlocal active_stream
            previous_stream = active_stream
            active_stream = stream
            try:
                yield
            finally:
                active_stream = previous_stream

        def wait_for_metadata():
            self.assertIs(active_stream, compute_stream)
            self.assertTrue(all(torch.count_nonzero(output) == 0 for output in outputs))
            runner.mapping_start.record.assert_called_once()
            runner.wait_cpu_scatter_ready(epoch)

        external_views = None

        def prepare(args, a, w, counts, *, output_buffers, scatter_row_addresses, route_weights,
                    external_wave_schedule=None):
            runner.wait_cpu_scatter_ready.assert_called_once_with(epoch)
            self.assertIs(external_wave_schedule, external_views)
            self.assertIs(active_stream, setup_stream)
            self.assertEqual(a.shape[0], sum(counts))
            self.assertEqual((args.tokens, args.routes, args.experts), (sum(counts), sum(counts), len(counts)))
            self.assertEqual(args.tile_k, 32)
            self.assertTrue(args.multi_buffer_scatter)
            self.assertIs(output_buffers[0], outputs[0])
            self.assertEqual(scatter_row_addresses.data_ptr(), runner.cpu_scatter_metadata.data_ptr())
            self.assertIs(route_weights, runner.route_weights)
            submitted.append(tuple(counts))
            return SimpleNamespace(output_buffers=lambda: output_buffers, scatter_row_addresses=scatter_row_addresses,
                                   wave_ready_prefix=torch.tensor([1], dtype=torch.int32))

        pregather = mock.Mock(prepare_grouped_inputs=prepare)
        # Model a QuACK default that the meon CLI does not expose.
        pregather.add_kernel_arguments.side_effect = lambda parser: parser.add_argument("--tile-k", type=int, default=32)
        empty_publisher = SimpleNamespace(prepare_empty_down_publisher=mock.Mock())
        with mock.patch.dict(sys.modules, {"run": SimpleNamespace(hopper_pregather_gemm=pregather),
                                           "quack.gemm": mock.Mock(), "empty_down": empty_publisher}), \
                mock.patch.object(torch.cuda, "Event", side_effect=lambda **kwargs: mock.Mock()), \
                mock.patch.object(torch.cuda, "synchronize"), \
                mock.patch.object(torch.cuda, "current_stream", side_effect=lambda _: active_stream), \
                mock.patch.object(torch.cuda, "stream", side_effect=use_stream):
            previous = worker.DownProjectionRunner(torch, up, [0, 0], options,
                                                   trace_configure=lambda groups: ready_address.item())
            outputs = previous.output_buffers()
            empty_publisher.prepare_empty_down_publisher.reset_mock()
            for epoch, counts in enumerate(([1, 4], [4, 1], [0, 0], [2, 1]), start=1):
                with self.subTest(counts=counts):
                    count_before = len(submitted)
                    address = previous.inputs.wave_ready_host_address
                    external_views = (torch.empty(2, 4), torch.empty(1)) if epoch in (2, 4) else None
                    runner = worker.DownProjectionRunner(torch, up, counts, options, previous=previous,
                                                         trace_configure=lambda groups: ready_address.item(),
                                                         kernel_options=previous.kernel_options,
                                                         defer_wave_schedule=True, external_wave_schedule=external_views)
                    runner.wait_cpu_scatter_ready = mock.Mock()
                    self.assertEqual(len(submitted), count_before)
                    compute_stream.wait_stream.reset_mock()
                    runner.mapping_start.record.reset_mock()
                    if not sum(counts):
                        self.assertIsInstance(runner.inputs, worker.EmptyDownInputs)
                        runner.finish_wave_schedule(None)
                        runner.wait_cpu_scatter_ready.assert_not_called()
                        compute_stream.wait_stream.assert_not_called()
                        empty_publisher.prepare_empty_down_publisher.assert_called_once()
                        previous = runner
                        continue
                    # Two int64 address words plus one float32 weight per route.
                    self.assertEqual(runner.cpu_scatter_metadata.numel(), sum(counts) * 20)
                    self.assertIs(runner.output_buffers()[0], outputs[0])
                    for output in outputs:
                        output.fill_(123)

                    def wait_for_schedule():
                        self.assertIs(active_stream, compute_stream)
                        self.assertIsNotNone(runner.inputs)
                        runner.launch_gemm.assert_not_called()
                        compute_stream.wait_stream.assert_called_once_with(setup_stream)

                    pregather.make_launch.return_value.reset_mock()
                    with use_stream(compute_stream):
                        runner.launch(wait_for_metadata=wait_for_metadata,
                                      wait_for_wave_schedule=wait_for_schedule if external_views is not None else None)
                    runner.launch_gemm.assert_called_once()
                    self.assertEqual(submitted[-1], tuple(counts))
                    # QuACK may retain the argument namespace captured by a launch.
                    # Its changes must not affect the next route's argument defaults.
                    pregather.make_launch.call_args.args[0].tile_k = 16
                    compute_stream.wait_stream.assert_called_once_with(setup_stream)
                    self.assertIs(runner.inputs.wave_ready_host_address, address)
                    self.assertIs(runner.output_buffers()[0], outputs[0])
                    previous = runner

            pregather.add_kernel_arguments.assert_called_once()
            checked = [call.args[0] for call in pregather.validate_args.call_args_list]
            self.assertEqual([(args.tokens, args.routes) for args in checked], [(5, 5), (5, 5), (3, 3)])
            self.assertTrue(all(not args.multi_buffer_scatter for args in checked))
            external_views = None
            runner = worker.DownProjectionRunner(torch, up, [3, 1], options, previous=previous,
                                                 trace_configure=mock.Mock(), defer_wave_schedule=True,
                                                 kernel_options=previous.kernel_options)
            epoch += 1
            runner.wait_cpu_scatter_ready = mock.Mock()
            with self.assertRaisesRegex(RuntimeError, "current scatter publication"):
                runner.finish_wave_schedule(None)
            pregather.make_launch.side_effect = RuntimeError("cannot create down launch")
            compute_stream.wait_stream.reset_mock()
            runner.mapping_start.record.reset_mock()
            with use_stream(compute_stream), self.assertRaisesRegex(RuntimeError, "cannot create down launch"):
                runner.launch(wait_for_metadata=wait_for_metadata)
            # QuACK may already have submitted copies on the setup stream when make_launch fails.
            compute_stream.wait_stream.assert_called_once_with(setup_stream)

    def test_native_configuration_selects_policy_and_retains_slab(self):
        torch = self.torch
        for mode in ("deferred", "threaded", "per-batch"):
            for weighted in (False, True):
                with self.subTest(mode=mode, weighted=weighted):
                    calls = []
                    runner = worker.DownProjectionRunner.__new__(worker.DownProjectionRunner)
                    runner.up = SimpleNamespace(torch=torch, cuda_device_id=0, source_pairs=((0, 0), (0, 1)),
                                                x_buffers=(torch.zeros(3, 4), torch.zeros(3, 4)),
                                                output=torch.zeros(3, 4, dtype=torch.bfloat16))
                    runner.routes, runner.combine_rows, runner.combine_tile_n = 3, 3, None
                    storage, addresses, weights = worker.make_cpu_scatter_metadata_storage(
                        torch, torch.zeros(3, dtype=torch.int64), torch.ones(3) if weighted else None)
                    runner.cpu_scatter_metadata = storage if mode != "per-batch" else None
                    runner.route_weights = weights
                    runner.inputs = SimpleNamespace(scatter_row_addresses=addresses,
                                                     output_buffers=lambda: runner.up.x_buffers)
                    if mode == "threaded":
                        runner.registered_outputs = runner.up.x_buffers
                        runner.inputs = None
                    runner.combine_valid_rows = torch.zeros(2, dtype=torch.int64)
                    runner.combine_output_to_input = torch.full((2, 3), -1, dtype=torch.int64)
                    runner.cpu_scatter_publication = mode
                    runner._cpu_scatter_wait = None

                    def configure(handle, pointer):
                        config = pointer._obj
                        calls.append(("scatter", handle, config.scatter_addresses))
                        self.assertEqual(list(config.output_bases[:2]),
                                         [buffer.data_ptr() for buffer in runner.up.x_buffers])
                        config.counts[0], config.counts[1] = 2, 1
                        for index, value in enumerate((0, 2, -1, 1, -1, -1)):
                            config.output_to_input[index] = value
                        return 0

                    def configure_weights(handle, pointer, routes):
                        calls.append(("weights", handle, pointer, routes))
                        return 0

                    def configure_deferred(handle, size):
                        calls.append(("deferred", handle, size))
                        return 0

                    def configure_threaded(handle, size):
                        calls.append(("threaded", handle, size))
                        return 0

                    def wait(*arguments):
                        return 0

                    library = SimpleNamespace(rdma_proxy_configure_cpu_scatter=configure,
                                              rdma_proxy_configure_cpu_route_weights=configure_weights,
                                              rdma_proxy_configure_cpu_scatter_deferred=configure_deferred,
                                              rdma_proxy_configure_cpu_scatter_threaded=configure_threaded,
                                              rdma_proxy_wait_cpu_scatter_ready=wait)
                    with mock.patch.object(torch.cuda, "synchronize"):
                        runner.configure_cpu_scatter(library, 123)
                    expected = [("scatter", 123, addresses.data_ptr())]
                    if weighted:
                        expected.append(("weights", 123, weights.data_ptr(), 3))
                    if mode != "per-batch":
                        expected.append((mode, 123, storage.numel()))
                        self.assertIs(runner.cpu_scatter_metadata, storage)
                        self.assertIs(runner._cpu_scatter_wait, wait)
                        self.assertIs(runner._cpu_scatter_library, library)
                        self.assertEqual(runner._cpu_scatter_handle, 123)
                    else:
                        self.assertIsNone(runner._cpu_scatter_wait)
                    self.assertEqual(calls, expected)
                    self.assertEqual(runner.cpu_scatter_counts.tolist(), [2, 1])
                    self.assertEqual(runner.combine_output_to_input.tolist(), [[0, 2, -1], [1, -1, -1]])


class MetadataDependencyTests(unittest.TestCase):
    def test_native_wait_receives_iteration_current_stream_and_timeout(self):
        calls = []
        runner = worker.DownProjectionRunner.__new__(worker.DownProjectionRunner)
        stream = SimpleNamespace(cuda_stream=0x123456789)
        runner.up = SimpleNamespace(cuda_device_id=7, torch=SimpleNamespace(
            cuda=SimpleNamespace(current_stream=lambda device: (
                self.assertEqual(device, 7), stream)[1])))
        runner._cpu_scatter_handle = object()
        runner._cpu_scatter_library = SimpleNamespace(rdma_proxy_last_error=lambda: b"publication timed out")
        runner.cpu_scatter_timeout_ms = 125
        runner._cpu_scatter_wait = lambda *args: (calls.append(args), 0)[1]
        for iteration in (0, 1, 11):
            runner.wait_cpu_scatter_ready(iteration)
        self.assertEqual(calls, [(runner._cpu_scatter_handle, iteration, stream.cuda_stream, 125)
                                 for iteration in (0, 1, 11)])
        runner._cpu_scatter_wait = lambda *args: -1
        with self.assertRaisesRegex(RuntimeError, "iteration 12.*publication timed out"):
            runner.wait_cpu_scatter_ready(12)
        runner._cpu_scatter_wait = None
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            runner.wait_cpu_scatter_ready(13)

    def test_down_and_combine_are_ordered_after_metadata_gate(self):
        def make_runner(events):
            event = lambda label: SimpleNamespace(record=lambda: events.append(label))
            runner = worker.DownProjectionRunner.__new__(worker.DownProjectionRunner)
            runner.up = SimpleNamespace(torch=SimpleNamespace(inference_mode=nullcontext))
            runner._log_combine_startup = lambda message: None
            runner.output_reset_policy = "before-down"
            runner.reset_start, runner.reset_end = event("reset"), event("reset-end")
            runner.mapping_start = event("mapping")
            runner.gemm_start, runner.end = event("gemm"), event("done")
            runner.inputs = SimpleNamespace(zero_up_output=lambda: events.append("zero"))
            runner.combine, runner.cpu_scatter = True, True
            runner.route_weights = object()  # Deferred weighted metadata is retained.
            runner.return_exchange = SimpleNamespace(publish_counts=lambda: events.append("counts"))
            runner.intra_node_combine = SimpleNamespace(mapping_ready=event("combine-ready"),
                                                       launch=lambda: events.append("combine"))
            runner.launch_gemm = lambda: events.append("down")
            return runner

        events = []
        runner = make_runner(events)
        runner.launch(wait_for_metadata=lambda: events.append("metadata-wait"),
                      before_combine=lambda: events.append("return-maps"))
        self.assertEqual(events, ["reset", "zero", "reset-end", "mapping", "metadata-wait", "counts",
                                  "gemm", "combine-ready", "down", "done", "return-maps", "combine"])
        events.clear()

        def fail_map_upload():
            events.append("return-maps-failed")
            raise RuntimeError("return-map upload failed")

        with self.assertRaisesRegex(RuntimeError, "return-map upload failed"):
            make_runner(events).launch(wait_for_metadata=lambda: events.append("metadata-wait"),
                                       before_combine=fail_map_upload)
        self.assertEqual(events[-3:], ["down", "done", "return-maps-failed"])
        events.clear()

        def fail():
            events.append("metadata-failed")
            raise RuntimeError("native submission gate failed")

        with self.assertRaisesRegex(RuntimeError, "submission gate failed"):
            make_runner(events).launch(wait_for_metadata=fail)
        self.assertEqual(events, ["reset", "zero", "reset-end", "mapping", "metadata-failed"])


if __name__ == "__main__":
    unittest.main()
