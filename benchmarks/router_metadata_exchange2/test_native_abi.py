import ctypes
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from native import Proxy, Stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from benchmark_metadata_backend import BenchmarkHostRoute, BenchmarkRouting, BenchmarkSource

class NativeAbiTests(unittest.TestCase):
    def test_receiver_overlap_is_configured_after_receiver_before_connect(self):
        for overlap in (False, True):
            lib = MagicMock()
            lib.rme2_abi_version.return_value = 15
            def create(*args):
                ctypes.cast(args[-1], ctypes.POINTER(ctypes.c_void_p))[0] = 123
                return 0
            lib.rme2_create.side_effect = create
            for name in ('batch_payload_copy', 'd2h_after_local_staging', 'local_staging_after_d2h',
                         'receiver_kernel', 'receiver_overlap_exchange'):
                getattr(lib, 'rme2_set_' + name).return_value = 0
            with patch('native.C.CDLL', return_value=lib):
                proxy = Proxy('unused.so', 0, 0, 33, '/test', 120, True, -1,
                              nodes=2, receiver_kernel=True, receiver_overlap_exchange=overlap)
            try:
                calls = [call[0] for call in lib.method_calls]
                self.assertLess(calls.index('rme2_set_receiver_kernel'),
                                calls.index('rme2_set_receiver_overlap_exchange'))
                lib.rme2_set_receiver_overlap_exchange.assert_called_once_with(proxy.handle, int(overlap))
                lib.rme2_connect.assert_not_called()
                self.assertEqual(lib.rme2_set_receiver_overlap_exchange.argtypes,
                                 [ctypes.c_void_p, ctypes.c_int])
            finally:
                proxy.close()

    def test_c_layouts_match_ctypes(self):
        result = subprocess.run([os.environ['RME2_ABI_BIN']], check=True, text=True, capture_output=True)
        layout = dict(line.split() for line in result.stdout.splitlines())
        for cls in (BenchmarkSource, BenchmarkRouting, BenchmarkHostRoute):
            prefix = f'RdmaProxy{cls.__name__}.'
            self.assertEqual(int(layout.pop(prefix + 'size')), ctypes.sizeof(cls))
            for field, _ in cls._fields_:
                self.assertEqual(int(layout.pop(prefix + field)), getattr(cls, field).offset, prefix + field)
        self.assertEqual(int(layout.pop('size')), ctypes.sizeof(Stats))
        for field, offset in layout.items():
            self.assertEqual(int(offset), getattr(Stats, field).offset, field)

    def test_direct_output_borrows_pointer_and_checks_capacity(self):
        proxy = Proxy.__new__(Proxy)
        proxy.handle, proxy.tokens = 123, 33
        for address, size, valid in ((4096, 165, True), (0, 165, False),
                                     (4097, 165, False), (4096, 164, False)):
            def enable(handle, payload, capacity):
                self.assertEqual(handle, 123)
                ctypes.cast(payload, ctypes.POINTER(ctypes.c_void_p))[0] = address
                ctypes.cast(capacity, ctypes.POINTER(ctypes.c_uint64))[0] = size
                return 0
            proxy.lib = SimpleNamespace(rme2_enable_direct_x3_x4=enable)
            if valid:
                self.assertEqual(proxy.enable_direct_x3_x4(), address)
            else:
                with self.assertRaisesRegex(RuntimeError, 'invalid native'):
                    proxy.enable_direct_x3_x4()

    def test_device_indices_preserve_address_capacity_and_session(self):
        proxy = Proxy.__new__(Proxy)
        proxy.handle, proxy.tokens = 123, 33
        proxy.lib = SimpleNamespace()
        self.assertFalse(proxy.supports_device_indices())
        for pointer, elements, valid in ((4096, 264, True), (0, 264, False),
                                         (4097, 264, False), (4096, 263, False)):
            def lookup(handle, source, address, capacity):
                self.assertEqual((handle, source), (123, 15))
                ctypes.cast(address, ctypes.POINTER(ctypes.c_void_p))[0] = pointer
                ctypes.cast(capacity, ctypes.POINTER(ctypes.c_int))[0] = elements
                return 0
            proxy.lib = SimpleNamespace(rme2_enable_device_indices=lambda handle: 0,
                                        rme2_device_indices=lookup)
            self.assertTrue(proxy.supports_device_indices())
            if valid:
                view = proxy.device_index_array(15)
                self.assertIs(view.owner, proxy)
                self.assertEqual(view.__cuda_array_interface__['data'], (pointer, False))
                self.assertEqual(view.__cuda_array_interface__['shape'], (elements,))
                self.assertEqual(view.__cuda_array_interface__['typestr'], '<i4')
            else:
                with self.assertRaisesRegex(RuntimeError, 'invalid native GPU index'):
                    proxy.device_index_array(15)

    def test_staging_capture_retains_tensor_owners_and_uses_separate_api(self):
        calls = []
        proxy = Proxy.__new__(Proxy)
        proxy.handle = 123
        proxy.lib = SimpleNamespace(
            rme2_staging_capture_begin=lambda handle, stream: calls.append(('begin', handle, stream)),
            rme2_staging_capture_end=lambda handle: calls.append(('end', handle)),
        )
        stream = SimpleNamespace(cuda_stream=456)
        plan = SimpleNamespace(launch=lambda s: calls.append(('gather', s.cuda_stream)))
        proxy.capture_staging(plan, stream)
        self.assertEqual(calls, [('begin', 123, 456), ('gather', 456), ('end', 123)])
        self.assertIs(proxy._staging_plan, plan)
        self.assertIs(proxy._staging_stream, stream)


if __name__ == '__main__':
    unittest.main()
