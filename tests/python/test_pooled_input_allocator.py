"""Exercise the embedding allocator's receive aliases with CPU tensor storage."""

import ctypes
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
import rdma_proxy_worker as worker


class PooledInputAllocatorTests(unittest.TestCase):
    def test_failed_pool_teardown_keeps_export_alive(self):
        pool = object()
        buffers = mock.Mock()
        buffers.tensors_of_kind.return_value = [pool]
        for initialized, success, shutdown_status in (
            (False, False, -1), (True, True, -1), (True, True, 0),
        ):
            with self.subTest(initialized=initialized, success=success, shutdown_status=shutdown_status):
                library, exchange = mock.Mock(), mock.Mock()
                library.rdma_proxy_shutdown.return_value = shutdown_status
                library.rdma_proxy_last_error.return_value = b"peer still has a mapping"
                with mock.patch.object(worker, "_unclosed_input_pools", []) as retained, \
                        mock.patch.object(sys, "stderr"):
                    if success and shutdown_status:
                        with self.assertRaisesRegex(RuntimeError, "peer still has a mapping"):
                            worker.shutdown_embedded_proxy(library, 1, buffers, initialized, success, exchange)
                    else:
                        worker.shutdown_embedded_proxy(library, 1, buffers, initialized, success, exchange)
                    self.assertEqual(retained, [pool] if shutdown_status else [])
                    library.rdma_proxy_shutdown.assert_called_once_with(1)
                    library.rdma_proxy_destroy.assert_called_once_with(1)
                    exchange.close.assert_called_once_with(success and shutdown_status == 0)

    def test_one_and_two_node_receive_slices(self):
        try:
            import torch
        except ImportError:
            self.skipTest("CPU PyTorch required")
        # Keep the production allocator and its ctypes callback. Only torch.empty
        # changes device so the tests can inspect the actual aliased storage.
        cpu = SimpleNamespace(empty=lambda shape, dtype, device: torch.empty(shape, dtype=dtype),
                              bfloat16=torch.bfloat16, float16=torch.float16,
                              float32=torch.float32, int32=torch.int32)
        for nodes in (1, 2):
            with self.subTest(nodes=nodes):
                gpus, rows, width = 8, 3, 4
                local_gpu, local_node = 3, nodes - 1
                buffers = worker.TorchDeviceBuffers(cpu, 0, node_rank=local_node,
                                                    local_gpu_index=local_gpu, num_gpus_per_node=gpus)

                def request(kind, shape, peer=-1, node=-1, gpu=-1):
                    desc = worker.DeviceBufferRequest(
                        ctypes.sizeof(worker.DeviceBufferRequest), 1, kind,
                        worker.DEVICE_BUFFER_BF16, peer, node, gpu, 2,
                        math.prod(shape) * 2, (ctypes.c_uint64 * 2)(*shape))
                    pointer = buffers.callback(None, ctypes.pointer(desc))
                    self.assertTrue(pointer, str(buffers.allocation_error))
                    return buffers.tensors[kind, peer, node, gpu]

                pool = request(worker.DEVICE_BUFFER_INPUT_POOL, (nodes * gpus * rows, width))
                receives = {node: request(worker.DEVICE_BUFFER_RDMA_RECEIVE, (rows, width), peer=node)
                            for node in range(nodes) if node != local_node}
                expected = torch.empty_like(pool)
                for node in range(nodes):
                    for gpu in range(gpus):
                        slot = node * gpus + gpu
                        target = request(worker.DEVICE_BUFFER_NVLINK_RECEIVE, (rows, width),
                                         node=node, gpu=gpu)
                        self.assertEqual(target.data_ptr(),
                                         pool.data_ptr() + slot * rows * width * pool.element_size())
                        if node != local_node and gpu == local_gpu:
                            self.assertIs(target, receives[node])
                        # Model independent RDMA/NVLink writes into adjacent destinations.
                        values = torch.arange(rows * width).reshape(rows, width) + slot * 16
                        target.copy_(values)
                        expected[slot * rows:(slot + 1) * rows] = values
                torch.testing.assert_close(pool, expected)
