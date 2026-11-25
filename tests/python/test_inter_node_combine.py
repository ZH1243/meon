"""CPU map tests and opt-in Hopper/Blackwell post-RDMA reduction tests."""
import ctypes
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'python'))
from inter_node_combine import InterNodeCombine, inverse_token_rows


class MappingTests(unittest.TestCase):
    def test_independent_unsorted_domains(self):
        domains = [[5, 0, 2], [2, 4], [], [0, 5, 3]]
        inverse = inverse_token_rows(domains, 7)
        self.assertEqual(inverse[0], [1, -1, 2, -1, -1, 0, -1])
        # Independent scatter reference versus gather in original token order.
        expected = [0] * 7
        packed = [[100 * (node + 1) + token for token in ids]
                  for node, ids in enumerate(domains)]
        for ids, values in zip(domains, packed):
            for token, value in zip(ids, values):
                expected[token] += value
        actual = [sum(values[rows[token]] for rows, values in zip(inverse, packed)
                      if rows[token] >= 0) for token in range(7)]
        self.assertEqual(actual, expected)
        self.assertEqual(actual[6], 0)
        self.assertEqual(inverse_token_rows([[], []], 0), [[], []])

    def test_reject_invalid_routes(self):
        for ids in ([1, 1], [-1], [3]):
            with self.assertRaises(ValueError):
                inverse_token_rows([ids], 3)
        with self.assertRaises(ValueError):
            inverse_token_rows([], -1)


class DynamicMappingTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("requires PyTorch")
        self.torch = torch
        self.device = torch.device("cuda" if os.environ.get("RDMA_PROXY_TEST_CUDA") == "1" else "cpu")
        self.capacity, self.width, self.tile = 6, 32, 16
        if self.device.type == "cuda":
            sys.path.insert(0, os.environ.get("QUACK_ROOT", str(
                Path(__file__).resolve().parents[3] / "quack_for_communication_overlap")))

    def make_source(self, num_nodes, dynamic=True, direct=True):
        nodes = list(range(num_nodes))
        storage = [self.torch.empty(self.capacity * self.width, dtype=self.torch.bfloat16,
                                    device=self.device) for _ in nodes]
        self.queries, self.builds = [], []

        def query(handle, node, indices, size, count, total):
            self.queries.append(node)
            tokens = self.domains[node]
            ctypes.cast(count, ctypes.POINTER(ctypes.c_uint64))[0] = len(tokens)
            ctypes.cast(total, ctypes.POINTER(ctypes.c_uint64))[0] = self.capacity
            if indices is not None:
                self.assertEqual(size, len(tokens))
                indices[:] = tokens
            return 0

        def build(handle, destination_nodes, count, tokens, rows, elements):
            self.assertEqual(list(destination_nodes[:count]), nodes)
            self.assertEqual(tokens, self.capacity)
            self.assertEqual(elements, num_nodes * (tokens + 2))
            self.builds.append(ctypes.cast(rows, ctypes.c_void_p).value)
            for i, values in enumerate(self.reference_rows()):
                for j, value in enumerate(values):
                    rows[i * (tokens + 2) + j] = value
            return 0

        lib = SimpleNamespace(rdma_proxy_get_outgoing_token_indices=query)
        if direct:
            lib.rdma_proxy_build_inter_node_combine_rows = build
        # The two-node case places the local node second to exercise input ordering.
        return SimpleNamespace(
            nodes=nodes,
            dynamic_capacity=self.capacity if dynamic else None, outputs={},
            storage_for_node=storage.__getitem__,
            rdma_return=SimpleNamespace(receive_storage=storage, received={}),
            exchange=SimpleNamespace(node=nodes[-1], library=lib, handle=None, torch=self.torch,
                                     require=lambda status: self.assertEqual(status, 0)))

    def reference_rows(self):
        return [row + [len(tokens), 0] for row, tokens in
                zip(inverse_token_rows(self.domains, self.capacity), self.domains)]

    def install(self, source, routes):
        self.domains = [list(ids) for ids in routes[:len(source.nodes)]]
        expected = self.torch.zeros((self.capacity, self.width), dtype=self.torch.bfloat16,
                                    device=self.device)
        for node, ids in zip(source.nodes, self.domains):
            storage = source.storage_for_node(node)
            view = storage[:len(ids) * self.width].view(self.width // self.tile, len(ids), self.tile)
            outputs = source.outputs if node == source.exchange.node else source.rdma_return.received
            outputs[node] = view
            for row, token in enumerate(ids):
                value = 10 * (node + 1) + row
                view[:, row].fill_(value)
                expected[token] += value
        return expected

    def test_direct_refresh_and_independent_check(self):
        torch = self.torch
        generations = (([], []), ([4, 1], [1, 2]), ([], [2]), ([3, 0], [0, 4]))
        for num_nodes in (1, 2):
            with self.subTest(num_nodes=num_nodes):
                source = self.make_source(num_nodes)
                uploads = []
                copy = torch.Tensor.copy_

                def track_copy(dst, src, *args, **kwargs):
                    uploads.append((dst.data_ptr(), src.data_ptr(), src.numel()))
                    return copy(dst, src, *args, **kwargs)

                for epoch, routes in enumerate(generations):
                    expected = self.install(source, routes)
                    self.queries.clear()
                    with patch.object(torch.Tensor, "copy_", track_copy):
                        if epoch == 0:
                            if self.device.type == "cuda":
                                options = SimpleNamespace(check_inter_node_combine=False, inter_node_combine_ctas=2)
                                combine = InterNodeCombine(source, options)
                            else:
                                combine = InterNodeCombine.__new__(InterNodeCombine)
                                combine.torch, combine.width = torch, self.width
                                combine.inputs, combine.token_lists, combine.num_tokens, combine.rows = (
                                    combine._init_route_data(source))
                        else:
                            combine.refresh_routes(source)
                    self.assertEqual(self.queries, [])
                    self.assertEqual(len(self.builds), epoch + 1)
                    self.assertEqual(len(uploads), epoch + 1)
                    self.assertEqual(uploads[-1], (combine.rows.data_ptr(), self.builds[-1],
                                                   num_nodes * (self.capacity + 2)))
                    self.assertEqual(uploads[-1], uploads[0])
                    self.assertEqual(combine.rows_host.is_pinned(), self.device.type == "cuda")
                    self.assertEqual(combine.rows.tolist(), self.reference_rows())
                    if self.device.type == "cuda":
                        combine.run()
                        torch.testing.assert_close(combine.output, expected)
                    else:
                        combine.output = expected
                    if epoch == len(generations) - 1:
                        # A same-length permutation in the independent x3 source must fail the check.
                        self.domains[0].reverse()
                        with self.assertRaises(AssertionError):
                            combine.check()
                    else:
                        combine.check()
                    self.assertEqual(self.queries, [n for n in source.nodes for _ in range(2)])

                source.outputs[source.exchange.node] = source.outputs[source.exchange.node][:, :1]
                with self.assertRaisesRegex(ValueError, "lengths disagree"):
                    combine.refresh_routes(source)

    def test_static_and_older_dynamic_library_use_x3_queries(self):
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                source = self.make_source(2, dynamic=dynamic, direct=not dynamic)
                self.install(source, ([5, 1], []))
                combine = InterNodeCombine.__new__(InterNodeCombine)
                combine.torch = self.torch
                _, lists, capacity, prepared = combine._init_route_data(source)
                self.assertEqual(lists, self.domains)
                self.assertEqual(capacity, self.capacity)
                self.assertIsNone(prepared)
                self.assertEqual(self.builds, [])
                self.assertEqual(self.queries, [0, 0, 1, 1])
                if dynamic:
                    combine.num_tokens = capacity
                    combine.rows = self.torch.empty((2, capacity + 2), dtype=self.torch.int32, device=self.device)
                    self.install(source, ([], [5, 0]))
                    combine.refresh_routes(source)
                    self.assertEqual(combine.rows.tolist(), self.reference_rows())
                    self.assertEqual(combine.token_lists, self.domains)


@unittest.skipUnless(os.environ.get('RDMA_PROXY_TEST_CUDA') == '1', 'requires CUDA opt-in')
class KernelTests(unittest.TestCase):
    def test_token_order_empty_domains_and_iteration_reuse(self):
        import torch
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10):
            self.skipTest('requires SM90 or SM100/SM103')
        sys.path.insert(0, os.environ.get('QUACK_ROOT', str(
            Path(__file__).resolve().parents[3] / 'quack_for_communication_overlap')))
        from inter_node_combine import TokenOrderCombine
        for dtype in (torch.bfloat16, torch.float16):
            for domains, tokens in (([[5, 0, 2]], 7), ([[]], 7), ([[]], 0),
                                    ([[5, 0, 2], [2, 4]], 7),
                                    ([[5, 0, 2], [], [4, 2], [0, 5, 3]], 7),
                                    ([[], []], 7), ([[], []], 0)):
                for tile in (8, 128):
                    with self.subTest(dtype=dtype, domains=domains, tokens=tokens, tile=tile):
                        width = tile * 3
                        inputs = [torch.empty((3, len(ids), tile), dtype=dtype, device='cuda')
                                  for ids in domains]
                        combine = TokenOrderCombine(inputs, domains, tokens, check=True,
                                                    ctas=1 if tile == 8 else 4)
                        for iteration in range(2):
                            expected = torch.zeros((tokens, width), dtype=torch.float32, device='cuda')
                            for tensor, ids in zip(inputs, domains):
                                values = torch.randn((len(ids), width), dtype=dtype, device='cuda') + iteration
                                tensor.copy_(values.view(len(ids), 3, tile).permute(1, 0, 2))
                                for row, token in enumerate(ids):
                                    expected[token] += values[row].float()
                            combine.output.fill_(float('nan'))
                            torch.cuda.synchronize()
                            combine.run()
                            torch.testing.assert_close(combine.output, expected.to(dtype), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
