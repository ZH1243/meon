"""Exercise the Python/native return-map contract using actual tensor storage on CPU."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from inter_node_combine import InterNodeCombine
from intra_node_combine import IntraNodeCombine
from rdma_network_return import NetworkReturn
from rdma_return_exchange import ReturnExchange
from return_maps import ReturnMaps


class ReturnMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("PyTorch is not installed")
        cls.torch = torch

    def test_stable_views_counts_and_independent_consumer_waits(self):
        torch = self.torch
        nodes, world, capacity = [0, 1], 2, 4
        shape = (len(nodes) * world, len(nodes) * capacity + 2)
        torch_dtype = torch.float16
        for include_inter in (False, True):
            with self.subTest(include_inter=include_inter):
                calls = []
                lib = SimpleNamespace()
                source_pairs = [(node, rank) for node in nodes for rank in range(world)]
                up = SimpleNamespace(output=torch.empty(0), source_pairs=source_pairs)
                ex = ReturnExchange.__new__(ReturnExchange)
                ex.torch, ex.library, ex.handle = torch, lib, 1
                ex.require = lambda result: self.assertEqual(result, 0)
                ex.rank, ex.world, ex.node = 0, world, 0
                ex.outputs = [torch.empty(capacity * 2, dtype=torch_dtype) for _ in source_pairs]
                ex.received = {(1, n): torch.empty_like(ex.outputs[0]) for n in nodes}
                ex.down = SimpleNamespace(up=up)
                ex.rows_host = torch.empty(len(source_pairs), dtype=torch.int64)
                self.assertIsNone(ReturnMaps.reserve_if_supported(ex, capacity, include_inter, 1))

                def configure(handle, node_ids, n, cap, slab, words, inter):
                    self.assertEqual((list(node_ids), cap, inter), (nodes, capacity, include_inter))
                    self.assertEqual(words, shape[0] * shape[1] + (n * (capacity + 2) if inter else 0))
                    calls.append(("configure", slab))
                    return 0

                def build_intra(handle, ids, n, cap, rows, size, counts):
                    # One initial CPU table is copied to the eventual native destination.
                    for i in range(n):
                        counts[i] = 0
                    for i in range(size):
                        rows[i] = -1
                    return 0

                def build_inter(handle, ids, n, cap, rows, size):
                    for i in range(size):
                        rows[i] = 0
                    return 0

                def start(handle, epoch, intra, outgoing, n):
                    intra[:], outgoing[:] = ([0, 2], [0, 3]) if epoch == 1 else ([1, 0], [1, 0])
                    calls.append(("start", epoch))
                    return 0

                def wait(handle, epoch, stream, timeout):
                    self.assertEqual(timeout, 125)
                    calls.append(("wait", epoch, stream))
                    return 0

                lib.rdma_proxy_build_intra_node_combine_rows = Mock(side_effect=build_intra)
                lib.rdma_proxy_build_inter_node_combine_rows = Mock(side_effect=build_inter)
                lib.rdma_proxy_configure_return_maps = Mock(side_effect=configure)
                lib.rdma_proxy_start_return_maps = Mock(side_effect=start)
                lib.rdma_proxy_wait_return_maps = Mock(side_effect=wait)
                maps = ex.return_maps = ReturnMaps.reserve_if_supported(ex, capacity, include_inter, 0.125)
                intra = IntraNodeCombine.__new__(IntraNodeCombine)
                intra.exchange, intra.dynamic_capacity, intra.device = ex, capacity, torch.device("cpu")
                intra.nodes, intra.width, intra.tile, intra.output_tile_n = nodes, 2, 1, 1
                intra.torch, intra.stream = torch, SimpleNamespace(cuda_stream=11)
                intra._init_routes()
                self.assertEqual(intra.rows.data_ptr(), calls[0][1])
                self.assertEqual(tuple(intra.rows.shape), shape)
                net = intra.rdma_return = NetworkReturn.__new__(NetworkReturn)
                net.receive_storage = {1: torch.empty(capacity * intra.width, dtype=torch_dtype)}
                net.sources, net.received = {}, {1: net.receive_storage[1][:0].view(2, 0, 1)}
                inter = intra.inter_node_combine = None
                if include_inter:
                    inter = intra.inter_node_combine = InterNodeCombine.__new__(InterNodeCombine)
                    inter.stream = SimpleNamespace(cuda_stream=13)
                    inter.inputs, inter.token_lists, inter.num_tokens, inter.rows = inter._init_route_data(intra)
                    inter.torch, inter.width = torch, intra.width
                    inter.output = torch.empty(capacity, intra.width, dtype=torch_dtype)
                    inter._read_token_lists = Mock(side_effect=[
                        ([[], [3, 0, 2]], capacity), ([[1], []], capacity)])
                    self.assertEqual(inter.rows.data_ptr(), intra.rows.data_ptr() + intra.rows.numel() * 4)
                ex.down.intra_node_combine = intra
                ex.down_done = SimpleNamespace(record=lambda: calls.append(("down_done",)))
                ex.arm_fn = lambda handle, epoch: calls.append(("arm", epoch)) or 0
                for epoch in (1, 2):
                    initial_calls = len(calls)
                    maps.start(epoch)
                    next_down = SimpleNamespace(output_buffers=lambda: ex.outputs, cpu_scatter_counts=ex.rows_host)
                    ex.bind_down(next_down)
                    # No host count consumption/map wait before launching up.
                    self.assertEqual(calls[initial_calls:], [("start", epoch)])
                    ex.arm_and_bind_maps(epoch)
                    expected = [("start", epoch), ("down_done",), ("arm", epoch), ("wait", epoch, 11)]
                    if include_inter:
                        expected.append(("wait", epoch, 13))
                        self.assertEqual([t.shape[1] for t in inter.inputs], list(maps.outgoing_counts))
                        # Exercise both independent reference checks on successively different x3.
                        for tensor in inter.inputs:
                            tensor.fill_(epoch)
                        inter.output.zero_()
                        inter.output[[3, 0, 2] if epoch == 1 else [1]] = epoch
                        inter.check()
                        self.assertEqual(inter._read_token_lists.call_count, epoch)
                    self.assertEqual(calls[initial_calls:], expected)
                    self.assertEqual([out.shape[1] for out in intra.outputs.values()], list(maps.intra_counts))
                    self.assertEqual(net.received[1].shape[1], maps.outgoing_counts[1])
                self.assertEqual(lib.rdma_proxy_configure_return_maps.call_count, 1)
                self.assertEqual(lib.rdma_proxy_build_intra_node_combine_rows.call_count, 1)
                self.assertEqual(lib.rdma_proxy_build_inter_node_combine_rows.call_count, int(include_inter))
