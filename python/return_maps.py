"""Persistent CUDA storage for both dynamic combine kernels' row maps."""

import ctypes as C
import math


class ReturnMaps:
    @classmethod
    def reserve_if_supported(cls, exchange, capacity, include_inter, timeout):
        lib = exchange.library
        try:
            configure = lib.rdma_proxy_configure_return_maps
            start = lib.rdma_proxy_start_return_maps
            wait = lib.rdma_proxy_wait_return_maps
            # Eager initialization fills the same GPU views used during replay.
            lib.rdma_proxy_build_intra_node_combine_rows
            if include_inter:
                lib.rdma_proxy_build_inter_node_combine_rows
        except AttributeError:
            return None
        int_ptr, count_ptr = C.POINTER(C.c_int32), C.POINTER(C.c_uint64)
        configure.argtypes = [C.c_void_p, int_ptr, C.c_uint64, C.c_uint64,
                              C.c_size_t, C.c_uint64, C.c_int32]
        start.argtypes = [C.c_void_p, C.c_uint64, count_ptr, count_ptr, C.c_uint64]
        wait.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t, C.c_uint64]
        for fn in (configure, start, wait):
            fn.restype = C.c_int
        torch = exchange.torch
        nodes = sorted({node for node, _ in exchange.down.up.source_pairs})
        n = len(nodes)
        intra_shape = (n * exchange.world, n * capacity + 2)
        inter_shape = (n, capacity + 2)
        intra_words = math.prod(intra_shape)
        words = intra_words + (math.prod(inter_shape) if include_inter else 0)
        slab = torch.empty(words, dtype=torch.int32, device=exchange.down.up.output.device)
        obj = cls()
        obj.intra_rows = slab[:intra_words].view(intra_shape)
        obj.inter_rows = slab[intra_words:].view(inter_shape) if include_inter else None
        obj.intra_counts = (C.c_uint64 * n)()
        obj.outgoing_counts = (C.c_uint64 * n)()
        obj.handle, obj.check = exchange.handle, exchange.require
        obj._start, obj._wait = start, wait
        obj.timeout_ms = max(1, math.ceil(timeout * 1000))
        obj.check(configure(obj.handle, (C.c_int32 * n)(*nodes), n, capacity,
                            slab.data_ptr(), words, int(include_inter)))
        print(f"Native return maps reserved: device_bytes={words * 4} domains={n}", flush=True)
        return obj

    def start(self, iteration):
        """Arm this epoch and read its live view lengths before starting communication."""
        self.check(self._start(self.handle, iteration, self.intra_counts, self.outgoing_counts,
                               len(self.intra_counts)))

    def wait(self, iteration, stream):
        self.check(self._wait(self.handle, iteration, int(stream.cuda_stream), self.timeout_ms))
