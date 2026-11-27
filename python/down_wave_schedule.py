"""Bind the per-proxy CPU wave builder to a reusable CUDA int32 allocation."""

import ctypes as C
import inspect
import math


class DownWaveGeometry(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "struct_size", "cluster_rows", "num_groups", "group_size", "max_wave_work_ids")]


class DownWaveSchedule:
    @classmethod
    def reserve_if_supported(cls, torch, library, handle, up, options, check):
        """Use the eager/deferred QuACK path with older ABI-v1 libraries or checkouts."""
        if not (options.cpu_scatter_preparation and options.cpu_scatter_publication == "threaded"
                and options.down_gather_table_publish_ready):
            return None
        try:
            capacity = library.rdma_proxy_get_down_wave_capacity
            configure = library.rdma_proxy_configure_down_wave_schedule
            start = library.rdma_proxy_start_down_wave_schedule
            wait = library.rdma_proxy_wait_down_wave_schedule
        except AttributeError:
            return None
        from run import hopper_pregather_gemm as pregather

        parameters = inspect.signature(pregather.prepare_grouped_inputs).parameters
        if not {"scatter_row_addresses", "external_wave_schedule"} <= parameters.keys():
            return None

        from quack.cute_dsl_utils import get_device_capacity, get_max_active_clusters
        from quack.gemm_config import cta_tile_shape_m

        device = up.output.device
        cluster_m = options.down_cluster_m
        cluster_rows = cta_tile_shape_m(options.down_tile_m, cluster_m, up.device_capacity) * cluster_m
        clusters_n = math.ceil(up.x_buffers[0].shape[1] / options.down_tile_n)
        group_size = min(options.down_max_swizzle_size, clusters_n)
        if clusters_n % group_size:
            raise ValueError("down wave schedule requires whole column groups")
        max_ids = get_max_active_clusters(cluster_m, device_capacity=get_device_capacity(device),
                                          device_id=up.cuda_device_id, sm_count=options.down_sm_count)
        geometry = DownWaveGeometry(C.sizeof(DownWaveGeometry), cluster_rows,
                                    clusters_n // group_size, group_size, max_ids)
        geo_ptr, count_ptr = C.POINTER(DownWaveGeometry), C.POINTER(C.c_uint64)
        capacity.argtypes = [C.c_void_p, geo_ptr, count_ptr]
        configure.argtypes = [C.c_void_p, geo_ptr, C.c_size_t, C.c_uint64]
        start.argtypes = [C.c_void_p, C.c_uint64, count_ptr, C.c_uint64, count_ptr, count_ptr]
        wait.argtypes = [C.c_void_p, C.c_uint64, C.c_size_t, C.c_uint64]
        for fn in (capacity, configure, start, wait):
            fn.restype = C.c_int
        words = C.c_uint64()
        check(capacity(handle, C.byref(geometry), C.byref(words)), "querying down-wave capacity")
        slab = torch.empty(words.value, dtype=torch.int32, device=device)
        check(configure(handle, C.byref(geometry), slab.data_ptr(), words.value), "reserving down-wave storage")
        print(f"Native down-wave schedule reserved: device_bytes={words.value * 4} "
              f"cluster_rows={cluster_rows} column_groups={geometry.num_groups}", flush=True)
        return cls(slab, handle, start, wait, check, math.ceil(options.return_token_timeout * 1000))

    def __init__(self, slab, handle, start, wait, check, timeout_ms):
        self.slab = slab  # Native writes here. Keep the owner live until after proxy shutdown.
        self.handle = handle
        self._start = start
        self._wait = wait
        self.check = check
        self.timeout_ms = timeout_ms

    def start(self, iteration, counts):
        """Start CPU construction and return shape-only views into this iteration's upload."""
        host_counts = (C.c_uint64 * len(counts))(*counts)
        rows, prefixes = C.c_uint64(), C.c_uint64()
        self.check(self._start(self.handle, iteration, host_counts, len(counts),
                               C.byref(rows), C.byref(prefixes)), f"starting down-wave schedule for {iteration}")
        table_words = rows.value * 4
        table = self.slab.narrow(0, 0, table_words).view(rows.value, 4)
        prefix = self.slab.narrow(0, table_words, prefixes.value)
        return table, prefix

    def wait(self, iteration, stream):
        self.check(self._wait(self.handle, iteration, int(stream.cuda_stream), self.timeout_ms),
                   f"waiting for down-wave schedule for {iteration}")
