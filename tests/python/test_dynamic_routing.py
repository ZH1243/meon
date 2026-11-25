"""Changing logical lengths, registered storage and row mappings across generations."""
import ctypes
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import rdma_proxy_worker as worker
from intra_node_combine import IntraNodeCombine
from rdma_return_exchange import ReturnExchange


class DynamicRoutingTests(unittest.TestCase):
    def test_prepare_then_query_active_extents(self):
        events = []

        def prepare(handle, epoch):
            events.append(epoch)
            return 0

        def extents(handle, indices, rows):
            ctypes.cast(indices, ctypes.POINTER(ctypes.c_uint64))[0] = 1 + events[-1]
            ctypes.cast(rows, ctypes.POINTER(ctypes.c_uint64))[0] = 4 * events[-1]
            return 0

        lib = SimpleNamespace(rdma_proxy_prepare_routing=prepare, rdma_proxy_get_routing_extents=extents)
        for epoch, expected in enumerate(((1, 0), (2, 4))):
            extents, elapsed_ns = worker.prepare_routing(lib, None, epoch)
            self.assertEqual(extents, expected)
            self.assertGreaterEqual(elapsed_ns, 0)
        self.assertEqual(events, [0, 1])

    def setUp(self):
        try:
            import torch
            self.torch = torch
        except ImportError:
            self.torch = None

    def test_up_live_views_and_weight_copies_follow_epoch(self):
        torch = self.torch
        if torch is None:
            self.skipTest("requires PyTorch")
        up = worker.GroupedGemmRunner.__new__(worker.GroupedGemmRunner)
        up.torch = torch
        up.index_storage = [torch.arange(8, dtype=torch.int32) for _ in range(2)]
        up.weight_storage = [torch.arange(8, dtype=torch.float32) for _ in range(2)]
        up.output_storage = torch.empty((2, 16))
        up.table_storage = torch.empty((8, 6), dtype=torch.int32)
        up.x_pool = object()
        up.pooled_index_storage = torch.empty(16, dtype=torch.int32)
        up.pooled_table_storage = torch.empty((8, 4), dtype=torch.int32)
        for epoch, (indices, rows) in enumerate(((2, 4), (1, 0), (7, 8), (2, 2))):
            up.weight_storage[0].fill_(epoch)
            up.bind_routing(indices, rows)
            self.assertEqual(tuple(up.output.shape), (2 * indices, 16))
            self.assertEqual(up.work_table.shape[0], rows)
            self.assertEqual(up.pooled_work_table.shape[0], rows)
            self.assertEqual(up.pooled_a_idx.numel(), 2 * indices)
            self.assertEqual(up.a_idx[0].numel(), indices)
            self.assertEqual(up.route_weights_stacked[0].tolist(), [epoch] * indices)
            self.assertEqual(up.empty_epoch, not rows)
            if up.empty_epoch:
                up.launch()
        with self.assertRaises(ValueError):
            up.bind_routing(9, 1)

    def test_bind_down_retains_ipc_and_updates_pinned_counts(self):
        torch = self.torch
        if torch is None:
            self.skipTest("requires PyTorch")
        exchange = ReturnExchange.__new__(ReturnExchange)
        exchange.rows_host = torch.zeros(2, dtype=torch.int64)
        exchange.outputs = (torch.empty(2), torch.empty(2))
        seen = []
        exchange.down = SimpleNamespace(intra_node_combine=SimpleNamespace(
            refresh_routes=lambda: seen.append(exchange.down.cpu_scatter_counts.tolist())))
        inputs = SimpleNamespace(output_buffers=lambda: exchange.outputs)
        for counts in ([2, 0], [0, 1]):
            down = SimpleNamespace(inputs=inputs, cpu_scatter_counts=torch.tensor(counts))
            exchange.bind_down(down)
            self.assertIs(exchange.down, down)
            self.assertEqual(exchange.rows_host.tolist(), counts)
        self.assertEqual(seen, [[2, 0], [0, 1]])

    def test_compact_tile_strides_follow_each_source_and_domain(self):
        torch = self.torch
        if torch is None:
            self.skipTest("requires PyTorch")
        combine = IntraNodeCombine.__new__(IntraNodeCombine)
        combine.dynamic_capacity = 5
        combine.nodes = [0, 1]
        combine.torch, combine.device = torch, torch.device("cpu")
        combine.rows = torch.empty((4, 12), dtype=torch.int32)
        combine.output = torch.empty(80)
        combine.width, combine.tile, combine.output_tile_n = 8, 4, 4
        for a, b in ((3, 5), (0, 2), (5, 0)):
            combine.domains = [(0, 0, a), (1, 5, b)]
            entries = [(0, 0, 0, torch.arange(a), None), (0, 1, 0, torch.arange(1, a + 1, 2) - 1, None),
                       (1, 0, 5, torch.arange(b), None), (1, 1, 5, torch.arange(b), None)]
            combine._update_views(entries)
            self.assertEqual(combine.rows[:, -1].tolist(), [a, a, b, b])
            self.assertEqual(combine.rows[:, -2].tolist(), [a, (a + 1) // 2, b, b])
            self.assertEqual(combine.rows[2, 5:10].tolist(), list(range(b)) + [-1] * (5 - b))
            self.assertEqual(combine.outputs[1].shape, (2, b, 4))
            self.assertEqual(combine.storage_for_node(1).data_ptr(), combine.output[40:].data_ptr())

    def test_direct_combine_rows_follow_dynamic_masks_with_one_upload(self):
        torch = self.torch
        if torch is None:
            self.skipTest("requires PyTorch")
        capacity, world = 4, 2
        device = torch.device("cuda" if os.environ.get("RDMA_PROXY_TEST_CUDA") == "1" else "cpu")
        generations = (([], []), ([2, 3, 1, 3], [3]), ([1], []), ([], [1, 2, 1, 3]))
        mask_queries = []
        builds = []

        def read_masks(handle, node, destination, size, count):
            mask_queries.append(node)
            ctypes.cast(count, ctypes.POINTER(ctypes.c_uint64))[0] = len(masks[node])
            if destination is not None:
                self.assertGreaterEqual(size, len(masks[node]))
                for i, mask in enumerate(masks[node]):
                    destination[i] = mask
            return 0

        def build(handle, nodes, num_nodes, limit, rows, elements, counts):
            self.assertEqual(list(nodes[:num_nodes]), [0, 1])
            self.assertEqual(limit, capacity)
            self.assertEqual(elements, expected_rows.numel())
            builds.append(ctypes.cast(rows, ctypes.c_void_p).value)
            ctypes.memmove(rows, expected_rows.data_ptr(), elements * expected_rows.element_size())
            for domain, source_masks in enumerate(masks):
                counts[domain] = len(source_masks)
            return 0

        def make_combine(has_builder, dynamic):
            combine = IntraNodeCombine.__new__(IntraNodeCombine)
            lib = SimpleNamespace(rdma_proxy_get_intra_node_source_masks=read_masks)
            if has_builder:
                lib.rdma_proxy_build_intra_node_combine_rows = build
            source_pairs = [(node, peer) for node in range(2) for peer in range(world)]
            outputs = [torch.zeros(capacity * 2, device=device) for _ in source_pairs]
            combine.exchange = SimpleNamespace(
                library=lib, require=lambda status: self.assertEqual(status, 0), handle=None,
                down=SimpleNamespace(up=SimpleNamespace(source_pairs=source_pairs)),
                outputs=outputs, received={(0, node): torch.zeros_like(outputs[0]) for node in range(2)},
                rank=1, world=world)
            combine.nodes = [0, 1]
            combine.dynamic_capacity = capacity if dynamic else None
            combine.torch, combine.device = torch, device
            combine.width = combine.tile = combine.output_tile_n = 2
            combine.rdma_return = combine.inter_node_combine = None
            return combine

        direct, legacy = make_combine(True, True), make_combine(False, True)
        uploads = []
        copy = torch.Tensor.copy_

        def track_copy(dst, src, *args, **kwargs):
            uploads.append((dst.data_ptr(), src.data_ptr(), src.numel()))
            return copy(dst, src, *args, **kwargs)

        for generation, masks in enumerate(generations):
            mask_queries.clear()
            if generation == 0:
                legacy._init_routes()
            else:
                legacy.refresh_routes()
            self.assertEqual(mask_queries, [0, 0, 1, 1])
            expected_rows = legacy.rows.cpu()
            mask_queries.clear()
            with patch.object(torch.Tensor, "copy_", track_copy):
                if generation == 0:
                    direct._init_routes()
                else:
                    direct.refresh_routes()
            self.assertEqual(mask_queries, [])
            self.assertEqual(len(uploads), generation + 1)
            self.assertEqual(uploads[-1], (direct.rows.data_ptr(), builds[-1], direct.rows.numel()))
            self.assertEqual(uploads[-1], uploads[0])
            self.assertEqual(direct.rows_host.is_pinned(), device.type == "cuda")
            self.assertEqual(direct.domains, legacy.domains)
            for node in direct.nodes:
                self.assertEqual(direct.outputs[node].shape, legacy.outputs[node].shape)
            torch.testing.assert_close(direct.rows, legacy.rows)

        # A corrupt domain length must be detected independently of the built table.
        node, offset, length = direct.domains[-1]
        direct.domains[-1] = (node, offset, length - 1)
        mask_queries.clear()
        with self.assertRaisesRegex(RuntimeError, "domain lengths disagree"):
            direct.check()
        self.assertEqual(mask_queries, [0, 0, 1, 1])

        # A library exporting both APIs must still use the fixed-routing layout.
        static = make_combine(True, False)
        mask_queries.clear()
        active = static._init_routes()
        self.assertEqual(mask_queries, [0, 0, 1, 1])
        self.assertEqual(len(builds), len(generations))
        self.assertEqual(static.rows.shape, (len(active), sum(map(len, masks))))

    @unittest.skipUnless(os.environ.get("RDMA_PROXY_TEST_CUDA") == "1", "requires CUDA opt-in")
    def test_empty_down_releases_all_groups_after_prior_stream_work(self):
        torch = self.torch
        import cuda.bindings.driver as cuda
        from empty_down import prepare_empty_down_publisher
        host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        status, device_pointer = cuda.cuMemHostGetDevicePointer(host.data_ptr(), 0)
        self.assertEqual(status, cuda.CUresult.CUDA_SUCCESS)
        address = torch.tensor([int(device_pointer)], dtype=torch.int64, device="cuda")
        stream = torch.cuda.current_stream()
        for groups in (1, 9):
            publish = prepare_empty_down_publisher(address, groups, stream)
            host.zero_()
            publish(stream)
            stream.synchronize()
            self.assertEqual(host.item(), groups)

    @unittest.skipUnless(os.environ.get("RDMA_PROXY_TEST_CUDA") == "1", "requires CUDA opt-in")
    def test_dynamic_intra_node_reduction_keeps_domain_bases_and_uses_live_tile_strides(self):
        torch = self.torch
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        sys.path.insert(0, os.environ.get("QUACK_ROOT", str(
            Path(__file__).resolve().parents[3] / "quack_for_communication_overlap")))
        from intra_node_combine import contribution_indices
        from intra_node_combine_kernel import RegisterIntraNodeKernel
        width, tile, capacity = 64, 16, 4
        rows = torch.full((4, 2 * capacity + 2), -1, dtype=torch.int32, device="cuda")
        inputs = [torch.empty(capacity * width, dtype=torch.bfloat16, device="cuda") for _ in range(4)]
        output = torch.empty(2 * capacity * width, dtype=torch.bfloat16, device="cuda")
        args = (tuple(from_dlpack(x, assumed_align=16) for x in inputs), from_dlpack(rows),
                from_dlpack(output, assumed_align=16), None)
        kernel = RegisterIntraNodeKernel((capacity,) * 4, width, tile, 2, 8, 128, True, tile)
        kernel.source_tokens, kernel.dynamic_domain_capacity = 2 * capacity, capacity
        kernel.input_domains = ((0, capacity, 0, 2), (capacity, capacity, 2, 2))
        kernel.output_domains = ((0, capacity), (capacity, capacity))
        kernel.domain_peers, kernel.dynamic_strides, kernel.wait_for_inputs = 2, True, False
        kernel.lanes_per_group, kernel.prefetch_peers = 2, 2
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(kernel, *args, stream)
        for domains in (([1, 3, 1], [3]), ([], [2, 3, 1, 2]), ([2], [])):
            rows.fill_(-1)
            output.fill_(-9)
            expected = torch.full_like(output, -9)
            for node, masks in enumerate(domains):
                base = node * capacity
                valid = len(masks)
                acc = torch.zeros(valid, width, device="cuda")
                for peer, indices in enumerate(contribution_indices(masks, 2)):
                    source = 2 * node + peer
                    rows[source, -2:] = torch.tensor([len(indices), valid], device="cuda")
                    for j, idx in enumerate(indices):
                        rows[source, base + idx] = j
                        value = torch.arange(width, device="cuda") + (1 + node + peer + idx)
                        inputs[source][:len(indices) * width].view(-1, len(indices), tile)[:, j, :].copy_(
                            value.view(-1, tile))
                        acc[idx] += value
                expected[base * width:(base + valid) * width].copy_(
                    acc.view(valid, width // tile, tile).permute(1, 0, 2).reshape(-1))
            compiled(*args, stream)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get("RDMA_PROXY_TEST_CUDA") == "1", "requires CUDA opt-in")
    def test_one_compiled_original_order_combine_across_empty_and_growing_nodes(self):
        torch = self.torch
        from inter_node_combine import TokenOrderCombine
        sys.path.insert(0, os.environ.get("QUACK_ROOT", str(
            Path(__file__).resolve().parents[3] / "quack_for_communication_overlap")))
        width, tile, tokens = 64, 16, 5
        storage = [torch.empty(tokens * width, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        combine = None
        for domains in (([4, 1], []), ([], [2, 3, 4]), ([4], [4, 1])):
            views = [buf[:len(ids) * width].view(width // tile, len(ids), tile)
                     for buf, ids in zip(storage, domains)]
            expected = torch.zeros((tokens, width), device="cuda", dtype=torch.float32)
            for node, (view, ids) in enumerate(zip(views, domains)):
                for row, token in enumerate(ids):
                    value = torch.arange(width, device="cuda") + token + node
                    view[:, row, :].copy_(value.view(-1, tile))
                    expected[token] += value
            if combine is None:
                combine = TokenOrderCombine(views, domains, tokens, check=True, ctas=2, capacity_buffers=storage)
            else:
                combine.bind_routes(views, domains)
            torch.cuda.synchronize()
            combine.run()
            torch.testing.assert_close(combine.output, expected.to(combine.output.dtype), rtol=0, atol=0)
