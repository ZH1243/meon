#!/usr/bin/env python3
"""Load one RDMA CPU proxy into each torchrun worker process."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import ctypes
from functools import partial
import inspect
import math
import os
from pathlib import Path
import re
import shlex
import sys
import threading
import time


EXPECTED_ABI_VERSION = 1

# Keep CLI parsing (including --dry-run) independent of CUDA/QuACK imports.
ORDINARY_ACTIVATIONS = ("silu", "silu-tanh", "relu", "relu_sq", "gelu_tanh_approx", "tanh")
GATED_ACTIVATIONS = (
    "swiglu", "swiglu-tanh", "swiglu_oai", "swiglu_oai-tanh", "reglu", "geglu", "glu",
)


def resolve_gemm_architecture(requested, capability):
    """Use the device's major architecture, including B300's SM103 as SM100."""
    major, minor = capability
    if major not in (9, 10):
        raise ValueError(f"proxy GEMMs require Hopper SM90 or Blackwell SM100, got SM{major}{minor}")
    expected = {"auto": major, "hopper": 9, "blackwell": 10}[requested]
    if expected != major:
        raise ValueError(f"--gemm-arch={requested} disagrees with device SM{major}{minor}")
    return major


def resolve_down_pingpong(requested, device_capacity, wave_barrier):
    if requested is None:
        return device_capacity == 9 and wave_barrier
    if device_capacity == 10 and requested:
        raise ValueError("Blackwell requires --no-down-pingpong (or omit the pingpong option)")
    if device_capacity == 9 and wave_barrier and not requested:
        raise ValueError("Hopper down wave barriers require --down-pingpong")
    return requested


def validate_up_sm_count(sm_count: int | None, cluster_m: int) -> None:
    """Match QuACK's requested cluster budget; hardware supplies its own cap."""
    if sm_count is None:
        return
    if cluster_m <= 0:
        raise ValueError("--grouped-gemm-cluster-m must be positive with --up-sm-count")
    if sm_count <= 0 or sm_count % cluster_m:
        raise ValueError("--up-sm-count must be positive and divisible by --grouped-gemm-cluster-m")


def projection_dimension(output_dimension: int, activation: str | None) -> int:
    """The table schedules projection columns, before gated activation halves them."""
    return output_dimension * (2 if activation in GATED_ACTIVATIONS else 1)

DEVICE_BUFFER_RDMA_SEND = 1
DEVICE_BUFFER_RDMA_RECEIVE = 2
DEVICE_BUFFER_NVLINK_RECEIVE = 3
DEVICE_BUFFER_ROUTER_A_IDX = 4
DEVICE_BUFFER_GATHER_TABLE = 5
DEVICE_BUFFER_GATHER_READY_ROWS = 6
DEVICE_BUFFER_ROUTER_ROUTE_WEIGHTS = 7
DEVICE_BUFFER_INPUT_POOL = 8
DEVICE_BUFFER_POOLED_A_IDX = 9
DEVICE_BUFFER_POOLED_GATHER_TABLE = 10

DEVICE_BUFFER_BF16 = 1
DEVICE_BUFFER_FP16 = 2
DEVICE_BUFFER_FP32 = 3
DEVICE_BUFFER_INT32 = 4


class DeviceBufferRequest(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("version", ctypes.c_uint32),
        ("kind", ctypes.c_int32),
        ("element_type", ctypes.c_int32),
        ("peer_rank", ctypes.c_int32),
        ("source_node_rank", ctypes.c_int32),
        ("source_gpu_index", ctypes.c_int32),
        ("dimension_count", ctypes.c_uint32),
        ("bytes", ctypes.c_uint64),
        ("dimensions", ctypes.c_uint64 * 2),
    ]


DeviceBufferAllocatorCallback = ctypes.CFUNCTYPE(
    ctypes.c_uint64,
    ctypes.c_void_p,
    ctypes.POINTER(DeviceBufferRequest),
)


class TorchDeviceBuffers:
    """Own proxy-visible CUDA tensors and service C++ allocation requests."""

    _KIND_NAMES = {
        DEVICE_BUFFER_RDMA_SEND: "rdma_send",
        DEVICE_BUFFER_RDMA_RECEIVE: "rdma_receive",
        DEVICE_BUFFER_NVLINK_RECEIVE: "nvlink_receive",
        DEVICE_BUFFER_ROUTER_A_IDX: "router_a_idx",
        DEVICE_BUFFER_GATHER_TABLE: "gather_table",
        DEVICE_BUFFER_GATHER_READY_ROWS: "gather_ready_rows",
        DEVICE_BUFFER_ROUTER_ROUTE_WEIGHTS: "router_route_weights",
        DEVICE_BUFFER_INPUT_POOL: "input_pool",
        DEVICE_BUFFER_POOLED_A_IDX: "pooled_a_idx",
        DEVICE_BUFFER_POOLED_GATHER_TABLE: "pooled_gather_table",
    }

    def __init__(
        self,
        torch_module,
        cuda_device_id: int,
        *,
        node_rank: int,
        local_gpu_index: int,
        num_gpus_per_node: int | None = None,
    ) -> None:
        self.torch = torch_module
        self.cuda_device_id = cuda_device_id
        self.node_rank = node_rank
        self.local_gpu_index = local_gpu_index
        self.num_gpus_per_node = num_gpus_per_node
        self.tensors: dict[tuple[int, int, int, int], object] = {}
        self.benchmark_index_views = {}
        self.allocation_error: BaseException | None = None
        # ctypes callbacks must themselves remain strongly referenced for as
        # long as C++ can invoke them.
        self.callback = DeviceBufferAllocatorCallback(self._allocate)

    def bind_benchmark_index_arrays(self, arrays, gpus_per_node, capacity):
        """Expose the exchange's per-source slots through the native allocation callback."""
        if self.benchmark_index_views or self.tensors_of_kind(DEVICE_BUFFER_ROUTER_A_IDX):
            raise RuntimeError("benchmark device indices must be bound before the first index allocation")
        views = {}
        device = self.torch.device("cuda", self.cuda_device_id)
        for source, array in enumerate(arrays):
            # Infer the device and dtype from the provider; coercing either could copy the slot.
            tensor = self.torch.as_tensor(array)
            if (array.capacity != capacity or tensor.data_ptr() != array.address
                    or tensor.device != device or tensor.dtype != self.torch.int32
                    or tuple(tensor.shape) != (capacity,) or tensor.stride() != (1,)):
                raise RuntimeError(f"source {source} has an incompatible benchmark device index view")
            # Keep both the non-owning tensor and its provider (which owns the session).
            views[divmod(source, gpus_per_node)] = (tensor, array)
        self.benchmark_index_views = views

    def _allocate(
        self,
        _context: int | None,
        request_pointer: ctypes.POINTER(DeviceBufferRequest),
    ) -> int:
        try:
            if not request_pointer:
                raise RuntimeError("C++ requested a device buffer without a descriptor")
            request = request_pointer.contents
            if request.struct_size != ctypes.sizeof(DeviceBufferRequest):
                raise RuntimeError(
                    "device-buffer request structure size mismatch: "
                    f"C++={request.struct_size}, Python={ctypes.sizeof(DeviceBufferRequest)}"
                )
            if request.version != 1:
                raise RuntimeError(
                    f"unsupported device-buffer request version {request.version}"
                )
            if request.dimension_count < 1 or request.dimension_count > 2:
                raise RuntimeError(
                    f"invalid device-buffer dimension count {request.dimension_count}"
                )

            dtype_by_element_type = {
                DEVICE_BUFFER_BF16: self.torch.bfloat16,
                DEVICE_BUFFER_FP16: self.torch.float16,
                DEVICE_BUFFER_FP32: self.torch.float32,
                DEVICE_BUFFER_INT32: self.torch.int32,
            }
            try:
                dtype = dtype_by_element_type[request.element_type]
                kind_name = self._KIND_NAMES[request.kind]
            except KeyError as error:
                raise RuntimeError(
                    "C++ requested an unknown device-buffer kind or element type: "
                    f"kind={request.kind}, element_type={request.element_type}"
                ) from error

            shape = tuple(
                int(request.dimensions[index])
                for index in range(request.dimension_count)
            )
            key = (
                int(request.kind),
                int(request.peer_rank),
                int(request.source_node_rank),
                int(request.source_gpu_index),
            )
            pool = self.tensors.get((DEVICE_BUFFER_INPUT_POOL, -1, -1, -1))
            if (
                request.kind == DEVICE_BUFFER_NVLINK_RECEIVE
                and request.source_node_rank != self.node_rank
                and request.source_gpu_index == self.local_gpu_index
            ):
                rdma_key = (
                    DEVICE_BUFFER_RDMA_RECEIVE,
                    int(request.source_node_rank),
                    -1,
                    -1,
                )
                try:
                    tensor = self.tensors[rdma_key]
                except KeyError as error:
                    raise RuntimeError(
                        "direct-RDMA input alias was requested before its peer "
                        f"receive tensor: source_node={request.source_node_rank}"
                    ) from error
            elif pool is not None and request.kind in (DEVICE_BUFFER_RDMA_RECEIVE,
                                                        DEVICE_BUFFER_NVLINK_RECEIVE):
                if self.num_gpus_per_node is None:
                    raise RuntimeError("pooled receive allocation requires num_gpus_per_node")
                if request.kind == DEVICE_BUFFER_RDMA_RECEIVE:
                    node, gpu = request.peer_rank, self.local_gpu_index
                else:
                    node, gpu = request.source_node_rank, request.source_gpu_index
                row_base = (node * self.num_gpus_per_node + gpu) * shape[0]
                tensor = pool.narrow(0, row_base, shape[0])
            elif request.kind == DEVICE_BUFFER_ROUTER_A_IDX and self.benchmark_index_views:
                tensor, _ = self.benchmark_index_views[request.source_node_rank, request.source_gpu_index]
            else:
                tensor = self.torch.empty(
                    shape,
                    dtype=dtype,
                    device=f"cuda:{self.cuda_device_id}",
                )
            tensor_bytes = tensor.numel() * tensor.element_size()
            if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor_bytes != request.bytes:
                raise RuntimeError(
                    f"{kind_name} tensor {tuple(tensor.shape)} {tensor.dtype} ({tensor_bytes} bytes) "
                    f"disagrees with C++ request {shape} {dtype} ({request.bytes} bytes)"
                )

            # The launch scripts disable C++ test-pattern filling. Give the
            # shared RDMA send tensor finite floating-point payloads now so it
            # is also suitable as the future communication/GEMM input source.
            if request.kind == DEVICE_BUFFER_RDMA_SEND:
                tensor.normal_(mean=0.0, std=0.02)

            if key in self.tensors:
                raise RuntimeError(f"duplicate C++ device-buffer request for key {key}")
            self.tensors[key] = tensor
            return int(tensor.data_ptr())
        except BaseException as error:
            self.allocation_error = error
            return 0

    def finish_initialization(self) -> None:
        if self.allocation_error is not None:
            raise RuntimeError(
                f"Python device-buffer allocation failed: {self.allocation_error}"
            ) from self.allocation_error
        # In particular, finish the random send-buffer initialization before
        # the first C++ iteration is allowed to post RDMA work.
        self.torch.cuda.synchronize(self.cuda_device_id)
        # Slice views have distinct data_ptrs but share the pool's storage.
        pooled = (DEVICE_BUFFER_INPUT_POOL, -1, -1, -1) in self.tensors
        unique_tensors = {
            int(tensor.data_ptr()): tensor for (kind, _, _, _), tensor in self.tensors.items()
            if not pooled or kind not in (DEVICE_BUFFER_RDMA_RECEIVE, DEVICE_BUFFER_NVLINK_RECEIVE)
        }
        total_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in unique_tensors.values()
        )
        counts: dict[str, int] = {}
        for kind, _, _, _ in self.tensors:
            name = self._KIND_NAMES[kind]
            counts[name] = counts.get(name, 0) + 1
        print(
            "Python owns proxy CUDA buffers: "
            f"logical_count={len(self.tensors)} "
            f"physical_count={len(unique_tensors)} bytes={total_bytes} "
            f"by_kind={counts}",
            flush=True,
        )

    def tensors_of_kind(self, kind: int) -> list[object]:
        """Return tensors in the same allocation/source order used by C++."""
        return [
            tensor
            for (tensor_kind, _, _, _), tensor in self.tensors.items()
            if tensor_kind == kind
        ]

    def only_tensor_of_kind(self, kind: int):
        tensors = self.tensors_of_kind(kind)
        if len(tensors) != 1:
            raise RuntimeError(
                f"expected one {self._KIND_NAMES[kind]} tensor, got {len(tensors)}"
            )
        return tensors[0]

    def benchmark_staging_buffers(self):
        local_source = (DEVICE_BUFFER_NVLINK_RECEIVE, -1, self.node_rank, self.local_gpu_index)
        return self.only_tensor_of_kind(DEVICE_BUFFER_RDMA_SEND), self.tensors[local_source]


class GroupedGemmRunner:
    """Launch QuACK's readiness-gated up GEMM on the proxy's receive storage."""

    empty_epoch = False

    def __init__(
        self,
        torch_module,
        device_buffers: TorchDeviceBuffers,
        *,
        cuda_device_id: int,
        world_size: int,
        num_experts: int,
        output_dimension: int,
        activation: str | None,
        tile_m: int,
        tile_n: int,
        tile_k: int | None,
        cluster_m: int,
        max_swizzle_size: int,
        quack_root: Path,
        sm_count: int | None = None,
        routing_extents: tuple[int, int] | None = None,
    ) -> None:
        validate_up_sm_count(sm_count, cluster_m)
        self.torch = torch_module
        self.cuda_device_id = cuda_device_id
        self.sm_count = sm_count
        if not device_buffers.tensors_of_kind(DEVICE_BUFFER_GATHER_READY_ROWS):
            raise RuntimeError(
                "--concurrent-kernel requires the notification-driven router scheduler; "
                "set --nvlink_forward_completion_notifications_enabled=true"
            )
        capability = torch_module.cuda.get_device_capability(cuda_device_id)
        self.device_capacity = resolve_gemm_architecture("auto", capability)
        if num_experts <= 0 or num_experts % world_size != 0:
            raise RuntimeError(
                f"--num-experts={num_experts} must be divisible by WORLD_SIZE={world_size}"
            )
        quack_root = quack_root.resolve()
        if not (quack_root / "quack" / "gemm.py").is_file():
            raise RuntimeError(f"QuACK repository not found at {quack_root}")
        quack_root_text = str(quack_root)
        if quack_root_text not in sys.path:
            sys.path.insert(0, quack_root_text)
        try:
            from quack.gemm import gemm as quack_gemm
        except Exception as error:
            raise RuntimeError(f"failed to import QuACK from {quack_root}: {error}") from error
        self.quack_gemm = quack_gemm
        self.activation = activation
        self.gated = activation in GATED_ACTIVATIONS
        self.gemm_dimension = projection_dimension(output_dimension, activation)
        if activation is not None:
            from quack.gemm_config import GemmConfig
            from quack.gemm_interface import gemm_act

            self.quack_gemm_act = gemm_act
            self.activation_config = GemmConfig(
                tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
                cluster_m=cluster_m, cluster_n=1, cluster_k=1,
                pingpong=False, is_dynamic_persistent=False,
                max_swizzle_size=max_swizzle_size,
                device_capacity=self.device_capacity, use_tma_gather=False,
            )

        a_idx_entries = [
            (key, tensor)
            for key, tensor in device_buffers.tensors.items()
            if key[0] == DEVICE_BUFFER_ROUTER_A_IDX
        ]
        if not a_idx_entries:
            raise RuntimeError(
                "C++ exposed no active A_idx tensors; forwarded-input gather mode is not initialized"
            )
        x_buffers = []
        a_idx = []
        source_pairs = []
        for (_, _, source_node, source_gpu), index_tensor in a_idx_entries:
            x_key = (
                DEVICE_BUFFER_NVLINK_RECEIVE,
                -1,
                source_node,
                source_gpu,
            )
            try:
                x_tensor = device_buffers.tensors[x_key]
            except KeyError as error:
                raise RuntimeError(
                    "missing NVLink tensor matching active A_idx source "
                    f"node={source_node}, gpu={source_gpu}"
                ) from error
            x_buffers.append(x_tensor)
            a_idx.append(index_tensor)
            source_pairs.append((source_node, source_gpu))

        first_x = x_buffers[0]
        if first_x.dtype not in (torch_module.bfloat16, torch_module.float16):
            raise RuntimeError(
                f"QuACK runner supports bf16/fp16 inputs, got {first_x.dtype}"
            )
        if any(tensor.shape != first_x.shape or tensor.dtype != first_x.dtype
               for tensor in x_buffers):
            raise RuntimeError("active NVLink input tensors do not share shape and dtype")
        first_a_idx = a_idx[0]
        if any(tensor.shape != first_a_idx.shape or tensor.dtype != torch_module.int32
               for tensor in a_idx):
            raise RuntimeError("active A_idx tensors do not share int32 shape")

        work_table = device_buffers.only_tensor_of_kind(
            DEVICE_BUFFER_GATHER_TABLE
        )
        ready_rows = device_buffers.only_tensor_of_kind(
            DEVICE_BUFFER_GATHER_READY_ROWS
        )
        expected_width = 2 + 2 * len(x_buffers)
        if work_table.ndim != 2 or work_table.shape[1] != expected_width:
            raise RuntimeError(
                f"gather table shape {tuple(work_table.shape)} does not match "
                f"{len(x_buffers)} active inputs (expected width {expected_width})"
            )
        if work_table.shape[0] == 0 and routing_extents is None:
            raise RuntimeError("QuACK gather table has zero rows")

        clusters_n = math.ceil(self.gemm_dimension / tile_n)
        group_size = min(max_swizzle_size, clusters_n)
        if group_size == 0 or clusters_n % group_size != 0:
            raise RuntimeError(
                "GroupedGEMM N clusters must be divisible by the effective swizzle group"
            )

        local_experts = num_experts // world_size
        hidden_dimension = int(first_x.shape[1])
        # Gated epilogues use QuACK's concat-B layout: contiguous [gate; up]
        # rows in storage, exposed logically as [E, K, 2H].
        weight_shape = (
            (local_experts, self.gemm_dimension, hidden_dimension)
            if self.gated else (local_experts, hidden_dimension, self.gemm_dimension)
        )
        self.weights = torch_module.randn(
            weight_shape,
            dtype=first_x.dtype,
            device=first_x.device,
        )
        if self.gated:
            self.weights = self.weights.transpose(1, 2)
        self.weights.mul_(1.0 / math.sqrt(hidden_dimension))
        # Dynamic routing sizes the first allocation in bind_routing, just like later growth.
        initial_output_rows = len(x_buffers) * first_a_idx.numel() if routing_extents is None else 0
        self.output_storage = torch_module.empty(
            (initial_output_rows, output_dimension),
            dtype=first_x.dtype,
            device=first_x.device,
        )
        self.output = self.output_storage
        self.x_buffers = tuple(x_buffers)
        self.source_pairs = tuple(source_pairs)
        self.a_idx = tuple(a_idx)
        # Keep the logical table and per-source list positions for down scatter,
        # weights and validation. Only the operands passed to the up kernel vary.
        self.x_pool = None
        if device_buffers.tensors_of_kind(DEVICE_BUFFER_INPUT_POOL):
            self.x_pool = device_buffers.only_tensor_of_kind(DEVICE_BUFFER_INPUT_POOL)
            self.pooled_a_idx = device_buffers.only_tensor_of_kind(DEVICE_BUFFER_POOLED_A_IDX)
            self.pooled_work_table = device_buffers.only_tensor_of_kind(DEVICE_BUFFER_POOLED_GATHER_TABLE)
            if self.device_capacity != 9:
                raise ValueError("pooled up GEMM requires Hopper SM90")
            if (self.pooled_a_idx.shape != (len(x_buffers) * first_a_idx.numel(),)
                    or self.pooled_work_table.shape != (work_table.shape[0], 4)):
                raise RuntimeError("pooled A_idx or four-column table has incompatible capacity")
        weight_entries = device_buffers.tensors_of_kind(DEVICE_BUFFER_ROUTER_ROUTE_WEIGHTS)
        self.route_weights = None
        if weight_entries:
            self.route_weights = tuple(device_buffers.tensors[
                (DEVICE_BUFFER_ROUTER_ROUTE_WEIGHTS, -1, node, gpu)
            ] for node, gpu in source_pairs)
            if any(weight.shape != index.shape or weight.dtype != torch_module.float32
                   for weight, index in zip(self.route_weights, self.a_idx)):
                raise RuntimeError("routing weights must be FP32 and aligned with each source A_idx")
        # Static routing keeps one stacked lookup. Dynamic routing rebuilds it from each live prefix.
        self.route_weights_stacked = (torch_module.stack(self.route_weights)
                                      if self.route_weights is not None and routing_extents is None else None)
        torch_module.cuda.synchronize(cuda_device_id)
        self.work_table = work_table
        self.ready_rows = ready_rows
        self.index_storage = self.a_idx
        self.table_storage = work_table
        self.weight_storage = self.route_weights
        if self.x_pool is not None:
            self.pooled_index_storage = self.pooled_a_idx
            self.pooled_table_storage = self.pooled_work_table
        if routing_extents is not None:
            self.bind_routing(*routing_extents)
        self.transposed_weights = self.weights.transpose(1, 2)
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.cluster_m = cluster_m
        self.max_swizzle_size = max_swizzle_size
        from quack.gemm_config import cta_tile_shape_m
        self.cta_tile_m = cta_tile_shape_m(tile_m, cluster_m, self.device_capacity)
        pooled = self.x_pool is not None
        input_shape = self.x_pool.shape if pooled else first_x.shape
        index_shape = self.pooled_a_idx.shape if pooled else first_a_idx.shape
        table_shape = self.pooled_work_table.shape if pooled else work_table.shape
        print(
            "GroupedGEMM configured: "
            f"device_capacity={self.device_capacity} mma_tile_m={tile_m} "
            f"cta_tile_m={self.cta_tile_m} cluster_rows={self.cta_tile_m * cluster_m} "
            f"sm_count={self.sm_count if self.sm_count is not None else 'hardware-default'} "
            f"activation={activation or 'none'} projection_N={self.gemm_dimension} "
            f"active_sources={source_pairs} X={tuple(input_shape)} "
            f"A_idx={tuple(index_shape)} W={tuple(self.weights.shape)} "
            f"output={tuple(self.output.shape)} table={tuple(table_shape)} pooled_up_gemm={pooled}",
            flush=True,
        )

    def bind_routing(self, source_index_rows, table_rows):
        """Bind QuACK to live prefixes. Padding is storage, never scheduler work."""
        max_source_rows = self.index_storage[0].numel()
        if (not 1 <= source_index_rows <= max_source_rows
                or not 0 <= table_rows <= self.table_storage.shape[0]):
            raise ValueError("native routing extents exceed their allocated storage")
        weights = stacked_weights = None
        if self.weight_storage is not None:
            weights = tuple(t[:source_index_rows] for t in self.weight_storage)
            stacked_weights = self.torch.stack(weights)
        source_count = len(self.index_storage)
        output_rows = source_index_rows * source_count
        storage = self.output_storage
        if output_rows > storage.shape[0]:
            # One extra row per 64 live rows per source absorbs small routing fluctuations.
            spare_per_source = (source_index_rows + 63) // 64
            capacity_rows = min(source_index_rows + spare_per_source, max_source_rows) * source_count
            try:
                storage = storage.new_empty((capacity_rows, storage.shape[1]))
            except self.torch.cuda.OutOfMemoryError:
                if capacity_rows == output_rows:
                    raise
                storage = storage.new_empty((output_rows, storage.shape[1]))

        self.route_weights = weights
        self.route_weights_stacked = stacked_weights
        self.output_storage = storage
        self.output = storage[:output_rows]
        self.a_idx = tuple(t[:source_index_rows] for t in self.index_storage)
        self.work_table = self.table_storage[:table_rows]
        self.empty_epoch = table_rows == 0
        if self.x_pool is not None:
            self.pooled_a_idx = self.pooled_index_storage[:output_rows]
            self.pooled_work_table = self.pooled_table_storage[:table_rows]

    def launch(self, *, pooled: bool | None = None) -> None:
        if pooled is None:
            pooled = self.x_pool is not None
        if pooled and self.x_pool is None:
            raise ValueError("pooled launch requires a pooled receive allocation")
        x = self.x_pool if pooled else self.x_buffers
        indices = self.pooled_a_idx if pooled else self.a_idx
        table = self.pooled_work_table if pooled else self.work_table
        if self.empty_epoch:
            return
        if self.activation is not None:
            self.quack_gemm_act(
                x,
                self.weights,
                activation=self.activation,
                postact_out=self.output,
                A_idx=indices,
                gather_work_table=table,
                gather_work_table_ready=self.ready_rows,
                multi_buffer_gather=not pooled,
                store_preact=False,
                dynamic_scheduler=False,
                tuned=False,
                config=self.activation_config,
                concat_layout=("B",) if self.gated else None,
                # Preserve default launches with older QuACK checkouts. An
                # explicit cap requires the updated fused API and is never lost.
                **({"sm_count": self.sm_count} if self.sm_count is not None else {}),
            )
            return
        self.quack_gemm(
            x,
            self.transposed_weights,
            self.output,
            C=None,
            tile_count_semaphore=None,
            tile_M=self.tile_m,
            tile_N=self.tile_n,
            tile_K=self.tile_k,
            cluster_M=self.cluster_m,
            cluster_N=1,
            cluster_K=1,
            pingpong=False,
            persistent=True,
            is_dynamic_persistent=False,
            max_swizzle_size=self.max_swizzle_size,
            cu_seqlens_m=None,
            A_idx=indices,
            gather_work_table=table,
            gather_work_table_ready=self.ready_rows,
            multi_buffer_gather=not pooled,
            use_tma_gather=False,
            sm_count=self.sm_count,
        )


def add_down_arguments(parser):
    parser.add_argument("--cpu-scatter-preparation", action="store_true",
                        help="prepare down scatter addresses on the CPU; requires local GPU combine")
    parser.add_argument("--cpu-scatter-publication", choices=("deferred", "threaded", "per-batch"), default=None,
                        help="CPU metadata policy: threaded packs on a dedicated CPU thread with one final upload; "
                             "deferred (default) packs and uploads after all up gather entries; "
                             "per-batch uploads each M batch before gather readiness")
    parser.add_argument("--inter-node-combine", action="store_true",
                        help="combine node sums into original token order after local combine and any RDMA returns")
    parser.add_argument("--inter-node-combine-ctas", type=int, default=None,
                        help="positive CTA count for inter-node combine; defaults to GPU SM count")
    parser.add_argument("--check-inter-node-combine", action=argparse.BooleanOptionalAction, default=False,
                        help="check final token-order output against an FP32 reference")
    parser.add_argument("--rdma-return-token-results", action="store_true",
                        help="return completed intra-node combine column groups over existing RDMA QPs")
    parser.add_argument("--check-rdma-return-token-results", action=argparse.BooleanOptionalAction, default=False,
                        help="compare returned payload SHA256 with the remote source after each iteration")
    parser.add_argument("--rdma-return-chunk-bytes", type=int, default=1048576)
    parser.add_argument("--intra-node-combine", action="store_true",
                        help="combine returned and retained down outputs while down GEMM runs")
    parser.add_argument("--intra-node-combine-n-tile-major-output", action="store_true",
                        help="store each source output as [N tiles, input rows, tile N]")
    parser.add_argument("--intra-node-combine-publish-ready", action="store_true",
                        help="publish completed intra-node output column-group prefix in HBM")
    parser.add_argument("--check-intra-node-combine", action=argparse.BooleanOptionalAction, default=False,
                        help="compare intra-node output with an FP32 reference each iteration")
    parser.add_argument("--intra-node-combine-ctas", type=int, default=4)
    parser.add_argument("--intra-node-combine-relax-sm-guard",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="allow down-sm-count <= total SMs - combine CTAs, instead of "
                             "subtracting combine CTAs * down-cluster-m; requires intra-node combine")
    parser.add_argument("--intra-node-combine-elements-per-thread", type=int,
                        choices=(8, 16, 32, 64, 128), default=32)
    parser.add_argument("--intra-node-combine-threads-per-cta", type=int,
                        choices=(256, 512), default=512)
    parser.add_argument("--intra-node-combine-prefetch-peers", type=int,
                        choices=(1, 2, 4), default=4)
    parser.add_argument("--return-token-results", action="store_true",
                        help="return local-combine outputs to their same-node NVLink senders")
    parser.add_argument("--return-token-results-per-group", action="store_true",
                        help="return each ready column group while down GEMM continues")
    parser.add_argument("--check-return-token-results", action=argparse.BooleanOptionalAction, default=False,
                        help="compare received valid bytes with producing GPU outputs each iteration")
    parser.add_argument("--return-token-timeout", type=float, default=120.0,
                        help="same-node return setup/completion and CPU metadata submission timeout in seconds")
    parser.add_argument("--down-projection2", action="store_true")
    parser.add_argument(
        "--down-output-reset", choices=("before-down", "end-of-iteration"), default="before-down",
        help="when to zero down outputs: before down GEMM (default), or after all iteration "
             "consumers and checks; end-of-iteration still includes reset in pipeline timing",
    )
    parser.add_argument(
        "--down-sm-count", type=int, default=None,
        help="cap down GEMM persistent CTAs; positive multiple of --down-cluster-m. "
             "Defaults to hardware capacity; does not reserve physical SMs",
    )
    parser.add_argument("--trace-down-ready", action="store_true",
                        help="poll down-output readiness in the C++ notification receiver")
    parser.add_argument("--local-gpu-combine", action="store_true",
                        help="sum down outputs into separate per-source token buffers, without routing weights")
    for name, default in (("tile-m", 128), ("tile-n", 128), ("cluster-m", 1),
                          ("max-swizzle-size", 1), ("scatter-output-buffers", 2)):
        parser.add_argument(f"--down-{name}", type=int, default=default)
    parser.add_argument("--down-pingpong", action=argparse.BooleanOptionalAction, default=None,
                        help="default: enabled for Hopper waves, disabled for Blackwell")
    for name in ("bulk-smem-padding", "bulk-smem-cols", "bulk-smem-buffers", "bulk-store-warps"):
        parser.add_argument(f"--down-{name}", type=int, default=None,
                            help="explicit SM100 bulk-store compilation setting; omitted keeps QuACK defaults")
    for name in ("scatter-table", "n-tile-major-output",
                 "multi-buffer-scatter", "gather-table", "gather-table-expert-waves",
                 "gather-table-wave-barrier"):
        parser.add_argument(f"--down-{name}", action="store_true")
    parser.add_argument("--down-epilogue-store",
                        choices=("tma", "bulk_rows", "bulk_rows_reduce"), default="tma")
    parser.add_argument(
        "--down-gather-table-publish-ready", action="store_true",
        help="publish completed down-output column groups to a GPU flag; "
             "requires down expert waves and the wave barrier",
    )


def validate_down_arguments(args):
    if args.down_output_reset != "before-down" and not args.down_projection2:
        raise ValueError("--down-output-reset requires --down-projection2")
    if args.cpu_scatter_preparation and not args.local_gpu_combine:
        raise ValueError("--cpu-scatter-preparation requires --local-gpu-combine")
    if args.cpu_scatter_publication is not None and not args.cpu_scatter_preparation:
        raise ValueError("--cpu-scatter-publication requires --cpu-scatter-preparation")
    if args.cpu_scatter_preparation and (
        not math.isfinite(args.return_token_timeout) or args.return_token_timeout <= 0
        or args.return_token_timeout > ((1 << 63) - 1) // 1000
    ):
        raise ValueError("--return-token-timeout must be finite, positive, and fit signed 64-bit milliseconds")
    if args.inter_node_combine_ctas is not None:
        if not args.inter_node_combine or args.inter_node_combine_ctas <= 0:
            raise ValueError("--inter-node-combine-ctas requires --inter-node-combine and a positive count")
    if args.check_inter_node_combine and not args.inter_node_combine:
        raise ValueError("--check-inter-node-combine requires --inter-node-combine")
    if args.inter_node_combine and not (args.intra_node_combine
                                        and args.intra_node_combine_n_tile_major_output):
        raise ValueError("--inter-node-combine requires --intra-node-combine "
                         "and --intra-node-combine-n-tile-major-output")
    if args.check_rdma_return_token_results and not args.rdma_return_token_results:
        raise ValueError("--check-rdma-return-token-results requires --rdma-return-token-results")
    if args.rdma_return_token_results:
        if not (args.intra_node_combine and args.intra_node_combine_n_tile_major_output
                and args.intra_node_combine_publish_ready):
            raise ValueError("--rdma-return-token-results requires --intra-node-combine, "
                             "--intra-node-combine-n-tile-major-output and --intra-node-combine-publish-ready")
        if not 0 < args.rdma_return_chunk_bytes <= 2**32 - 1:
            raise ValueError("--rdma-return-chunk-bytes must be in [1, 2**32-1]")
    if args.intra_node_combine_publish_ready and not args.intra_node_combine:
        raise ValueError("--intra-node-combine-publish-ready requires --intra-node-combine")
    if args.intra_node_combine_n_tile_major_output and not args.intra_node_combine:
        raise ValueError("--intra-node-combine-n-tile-major-output requires --intra-node-combine")
    if args.check_intra_node_combine and not args.intra_node_combine:
        raise ValueError("--check-intra-node-combine requires --intra-node-combine")
    if getattr(args, "intra_node_combine_relax_sm_guard", False) and not args.intra_node_combine:
        raise ValueError("--intra-node-combine-relax-sm-guard requires --intra-node-combine")
    if args.intra_node_combine:
        if not (args.local_gpu_combine and args.return_token_results_per_group):
            raise ValueError("--intra-node-combine requires --local-gpu-combine and "
                             "--return-token-results-per-group with their prerequisites")
        if args.intra_node_combine_ctas <= 0:
            raise ValueError("--intra-node-combine-ctas must be positive")
        if args.down_sm_count is None:
            args.down_sm_count = 40

    if args.return_token_results_per_group and (
        not args.return_token_results or not args.down_n_tile_major_output
    ):
        raise ValueError("--return-token-results-per-group requires --return-token-results "
                         "and --down-n-tile-major-output")
    if args.check_return_token_results and not args.return_token_results:
        raise ValueError("--check-return-token-results requires --return-token-results")
    if args.return_token_results:
        if not args.local_gpu_combine or not args.down_gather_table_publish_ready:
            raise ValueError("--return-token-results requires --local-gpu-combine "
                             "and --down-gather-table-publish-ready")
        if not math.isfinite(args.return_token_timeout) or args.return_token_timeout <= 0:
            raise ValueError("--return-token-timeout must be finite and positive")
    if args.trace_down_ready and (
        not args.down_projection2 or not args.down_gather_table_publish_ready
    ):
        raise ValueError("--trace-down-ready requires --down-projection2 and "
                         "--down-gather-table-publish-ready")
    if args.local_gpu_combine and not args.down_projection2:
        raise ValueError("--local-gpu-combine requires --down-projection2")
    if not args.down_projection2:
        return
    if args.local_gpu_combine:
        # Combine owns the destination count and requires additive scatter.
        args.down_scatter_table = True
        args.down_multi_buffer_scatter = True
        args.down_epilogue_store = "bulk_rows_reduce"
    if not args.concurrent_kernel or args.grouped_gemm_activation in (None, "none"):
        raise ValueError("--down-projection2 requires --concurrent-kernel and a fused activation")
    for name in ("tile_m", "tile_n", "cluster_m", "max_swizzle_size"):
        if getattr(args, "down_" + name) <= 0:
            raise ValueError(f"--down-{name.replace('_', '-')} must be positive")
    if args.down_sm_count is not None and (
        args.down_sm_count <= 0 or args.down_sm_count % args.down_cluster_m
    ):
        raise ValueError("--down-sm-count must be positive and divisible by --down-cluster-m")
    if args.down_scatter_table and args.down_epilogue_store == "tma":
        raise ValueError("down scatter requires a bulk-row epilogue")
    if (args.down_multi_buffer_scatter or args.down_n_tile_major_output) and not args.down_scatter_table:
        raise ValueError("down multi-buffer/tiled output requires --down-scatter-table")
    if (not args.local_gpu_combine and args.down_multi_buffer_scatter
            and args.down_scatter_output_buffers < 2):
        raise ValueError("--down-scatter-output-buffers must be at least 2")
    if args.down_gather_table_expert_waves and not args.down_gather_table:
        raise ValueError("down expert waves require --down-gather-table")
    if args.down_gather_table_publish_ready and (
        not args.down_gather_table_expert_waves or not args.down_gather_table_wave_barrier
    ):
        raise ValueError("--down-gather-table-publish-ready requires "
                         "--down-gather-table-expert-waves and --down-gather-table-wave-barrier")
    if args.down_gather_table_wave_barrier and (
        not args.down_gather_table_expert_waves
        or args.down_epilogue_store == "tma"
    ):
        raise ValueError("down wave barrier requires expert waves and a bulk-row epilogue")
    requested_arch = getattr(args, "gemm_arch", "auto")
    if requested_arch != "auto":
        resolve_down_pingpong(args.down_pingpong, 9 if requested_arch == "hopper" else 10,
                              args.down_gather_table_wave_barrier)


def local_route_positions(torch, work_table, num_sources, routes, n_groups, trace=None):
    """Recover each packed row's exact (source, A_idx list position) on device.

    Multi-buffer QuACK places a table row at sum(start_j), concatenating its
    source ranges. N groups repeat those ranges. Publication order is arbitrary;
    neither expert order nor source order alone describes the packed output.
    Called on the up/down stream after up has consumed the complete table.
    """
    def mark(message):
        if trace is not None:
            trace("mapping: " + message)

    mark("table conversion begin")
    ranges = work_table[::n_groups, 2:].long().reshape(-1, num_sources, 2)
    starts, ends = ranges[:, :, 0], ranges[:, :, 1]
    mark("table conversion submitted; lengths begin")
    lengths = ends - starts
    mark("lengths submitted; sum begin")
    start_sums = starts.sum(dim=1, keepdim=True)
    mark("sum submitted; cumsum begin")
    cumulative_lengths = lengths.cumsum(dim=1)
    mark("cumsum submitted; output ends begin")
    output_ends = start_sums + cumulative_lengths
    del start_sums, cumulative_lengths
    # Empty source segments must never win a search, even at batch boundaries.
    mark("output ends submitted; keys begin")
    keys = torch.where(lengths > 0, output_ends, routes + 1).flatten()
    mark("keys submitted; argsort begin")
    order = keys.argsort()
    mark("argsort submitted; arange begin")
    rows = torch.arange(routes, device=work_table.device, dtype=torch.int64)
    mark("arange submitted; sorted keys gather begin")
    sorted_keys = keys[order]
    mark("sorted keys submitted; searchsorted begin")
    positions = torch.searchsorted(sorted_keys, rows, right=True)
    mark("searchsorted submitted; segment gather begin")
    segments = order[positions]
    del sorted_keys, positions
    mark("segment gather submitted; source and index arithmetic begin")
    source = segments % num_sources
    output_starts = (output_ends - lengths).flatten()
    index = starts.flatten()[segments] + rows - output_starts[segments]
    mark("index arithmetic submitted")
    return source, index


def local_combine_mapping(torch, work_table, a_idx, routes, n_groups, trace=None):
    """Map routed rows to input rows; list positions also distinguish expert weights."""
    source, index = local_route_positions(torch, work_table, len(a_idx), routes, n_groups, trace)
    indices = torch.stack(a_idx)
    token_indices = indices[source, index].long()
    result = torch.stack((source, token_indices), dim=1)
    return result


def compact_combine_mapping(torch, mapping, num_sources, capacity):
    """Compact distinct used rows per source in ascending input-row order.

    Returns route destinations, device-side valid counts, and a capacity-sized
    output-to-input map (-1 beyond each valid prefix). Duplicate expert routes
    share a destination. Only meaningful routes participate, never A_idx padding.
    All operations stay on the calling CUDA stream, with fixed-size allocations.
    """
    flat = mapping[:, 0] * capacity + mapping[:, 1]
    hits = torch.zeros(num_sources * capacity, dtype=torch.int64, device=mapping.device)
    hits.scatter_add_(0, flat, torch.ones_like(flat))
    used = hits.reshape(num_sources, capacity) > 0
    ranks = used.long().cumsum(dim=1) - 1
    compact = torch.stack((mapping[:, 0], ranks.flatten()[flat]), dim=1)
    rows = torch.arange(capacity, device=mapping.device, dtype=torch.int64)
    output_to_input = torch.where(used, rows, capacity).sort(dim=1).values
    output_to_input = torch.where(output_to_input < capacity, output_to_input, -1)
    return compact, used.sum(dim=1), output_to_input


def allocate_cpu_scatter_metadata_storage(torch, address_shape, device, weighted, storage=None):
    """Allocate the address and optional weight views for a single native upload."""
    scatter_bytes = math.prod(address_shape) * 8  # int64 addresses
    weight_offset = (scatter_bytes + 15) // 16 * 16
    total_bytes = weight_offset + address_shape[0] * 4 if weighted else scatter_bytes
    storage = (torch.empty(total_bytes, dtype=torch.uint8, device=device) if storage is None
               else storage.narrow(0, 0, total_bytes))
    addresses = storage[:scatter_bytes].view(torch.int64).reshape(address_shape)
    weights = storage[weight_offset:].view(torch.float32) if weighted else None
    return storage, addresses, weights


def make_cpu_scatter_metadata_storage(torch, scatter_row_addresses, route_weights=None):
    """Pack existing QuACK inputs into the native scatter layout, including for warmup."""
    if (scatter_row_addresses.dtype != torch.int64 or not scatter_row_addresses.is_contiguous()
            or scatter_row_addresses.ndim not in (1, 2)
            or (scatter_row_addresses.ndim == 2 and scatter_row_addresses.shape[1] != 2)
            or scatter_row_addresses.shape[0] <= 0):
        raise ValueError("CPU scatter storage requires contiguous int64 routes or routes-by-2 addresses")
    if route_weights is not None and (
        route_weights.dtype != torch.float32 or route_weights.shape != (scatter_row_addresses.shape[0],)
        or route_weights.device != scatter_row_addresses.device or not route_weights.is_contiguous()
    ):
        raise ValueError("CPU scatter weights require contiguous FP32 storage matching address rows and device")
    storage, addresses, weights = allocate_cpu_scatter_metadata_storage(
        torch, scatter_row_addresses.shape, scatter_row_addresses.device, route_weights is not None)
    addresses.copy_(scatter_row_addresses)
    if route_weights is not None:
        # An odd number of int64s leaves one int64 of padding before the weights.
        if addresses.numel() % 2:
            padding_start = addresses.numel() * addresses.element_size()
            storage[padding_start:padding_start + 8].zero_()
        weights.copy_(route_weights)
    return storage, addresses, weights


class EmptyDownInputs:
    """Down projection state on a rank receiving zero expert assignments."""

    wave_work_ids = None
    scatter_table = None
    wave_ready_host_address = None

    def __init__(self, torch, args, exemplar, outputs):
        self.n_tile_major_tile_n = args.tile_n if args.n_tile_major_output else None
        self.outputs = outputs if outputs is not None else (exemplar.new_empty((0, args.output_dim)),)
        self.scatter_output_rows = tuple(0 for _ in self.outputs)
        self.ready_groups = None
        if args.gather_table_publish_ready:
            tiles = math.ceil(args.output_dim / args.tile_n)
            self.groups = math.ceil(tiles / min(tiles, args.max_swizzle_size))
            self.wave_ready_prefix = torch.tensor([self.groups], dtype=torch.int32, device=exemplar.device)
            self.ready_groups = torch.zeros_like(self.wave_ready_prefix)

    def output_buffers(self):
        return self.outputs

    def zero_up_output(self):
        for buffer in self.outputs:
            buffer.zero_()


class CpuScatterConfig(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("struct_size", "num_sources", "routes", "capacity", "width",
                 "tile_n", "element_bytes")] + [
        ("scatter_addresses", ctypes.c_size_t),
        ("source_pairs", ctypes.POINTER(ctypes.c_int32)),
        ("output_bases", ctypes.POINTER(ctypes.c_uint64)),
        ("counts", ctypes.POINTER(ctypes.c_int64)),
        ("output_to_input", ctypes.POINTER(ctypes.c_int64))]

    @classmethod
    def for_outputs(cls, down, counts, inverse):
        up = down.up
        sources = len(up.source_pairs)
        pairs = (ctypes.c_int32 * (2 * sources))(
            *(value for pair in up.source_pairs for value in pair))
        bases = (ctypes.c_uint64 * sources)(*(buffer.data_ptr() for buffer in down.output_buffers()))
        index_ptr = ctypes.POINTER(ctypes.c_int64)
        return cls(ctypes.sizeof(cls), sources, 0, down.combine_rows, int(up.x_buffers[0].shape[1]),
                   down.combine_tile_n or 0, up.output.element_size(), 0, pairs, bases,
                   ctypes.cast(counts.data_ptr(), index_ptr),
                   ctypes.cast(inverse.data_ptr(), index_ptr) if inverse is not None else None)


# States returned by rdma_proxy_get_native_local_publication.
LOCAL_REQUESTED = 1
LOCAL_NEEDS_STORAGE = 2
LOCAL_UNBOUND = 3


class AsyncScatterLookup:
    """Stable host destinations written by the native scatter thread on refreshed routes."""

    address_storage = None

    def reserve_addresses(self, up, columns):
        # cudaMalloc on cache growth can join a polling up kernel. Reserve before launch.
        capacity = up.output_storage.shape[0]
        size = capacity * columns * 8
        if up.route_weights is not None:
            size = (size + 15) // 16 * 16 + capacity * 4
        if self.address_storage is None or self.address_storage.numel() < size:
            self.address_storage = up.torch.empty(size, dtype=up.torch.uint8, device=up.output.device)
            # Native H2D uses a different stream. Finish a stream-ordered
            # allocation on growth, before a polling kernel can be resident.
            up.torch.cuda.current_stream(up.cuda_device_id).synchronize()

    def register(self, library, handle, routes, slab, weights):
        # Native copies the geometry and output bases before returning. The slab,
        # output tensors and host counts/inverse remain live through retirement.
        self.config.routes = routes
        self.config.scatter_addresses = slab.data_ptr()
        packed_weights = weights.data_ptr() if weights is not None else 0
        status = self.configure(handle, ctypes.byref(self.config), packed_weights, slab.numel())
        require_proxy_success(library, status, "starting native scatter lookup")

    def reserved_views(self, up, routes):
        if self.address_storage is None or not 0 < routes <= up.output.shape[0]:
            raise ValueError("early scatter registration requires reserved storage and a nonempty current route")
        shape = (routes, 2) if self.config.tile_n else (routes,)
        return allocate_cpu_scatter_metadata_storage(
            up.torch, shape, up.output.device, up.route_weights is not None, storage=self.address_storage)

    def register_reserved(self, library, handle, up, routes):
        views = self.reserved_views(up, routes)
        slab, _, weights = views
        self.register(library, handle, routes, slab, weights)
        return views

    @classmethod
    def reserve_if_supported(cls, library, down, options, exchange):
        configure = getattr(library, "rdma_proxy_configure_cpu_scatter_async", None)
        if configure is None or not options.cpu_scatter_preparation or options.cpu_scatter_publication != "threaded":
            return None
        configure.argtypes = [ctypes.c_void_p, ctypes.POINTER(CpuScatterConfig),
                              ctypes.c_size_t, ctypes.c_uint64]
        configure.restype = ctypes.c_int
        torch = down.torch
        sources = len(down.up.source_pairs)
        lookup = cls()
        lookup.configure = configure
        # NVLink return copies read this same pointer after the scatter event and arm.
        lookup.counts = (exchange.rows_host if exchange is not None
                         else torch.empty(sources, dtype=torch.int64, device="cpu"))
        lookup.inverse = (torch.empty((sources, down.combine_rows), dtype=torch.int64, device="cpu")
                          if options.check_projections or options.check_intra_node_combine else None)
        lookup.config = CpuScatterConfig.for_outputs(down, lookup.counts, lookup.inverse)
        print("Native threaded scatter lookup enabled", flush=True)
        return lookup


class _DownKernelOptions:
    """QuACK configuration shared by the routing iterations of one worker."""

    def __init__(self, pregather, up, options):
        parser = argparse.ArgumentParser()
        pregather.add_kernel_arguments(parser)
        args = parser.parse_args([])
        for name, value in vars(options).items():
            if name.startswith("down_") and name not in ("down_projection2", "down_output_reset"):
                setattr(args, name[5:], value)
        args.device_capacity = up.device_capacity
        args.pingpong = resolve_down_pingpong(
            args.pingpong, up.device_capacity, args.gather_table_wave_barrier)
        vars(args).update(experts=int(up.weights.shape[0]), hidden=int(up.output.shape[1]),
                          output_dim=int(up.x_buffers[0].shape[1]), activation=None,
                          down_projection=False, iterations=1, timing_samples=1, warmup=0)
        if options.local_gpu_combine:
            args.scatter_table = True
            args.multi_buffer_scatter = True
            args.epilogue_store = "bulk_rows_reduce"
            args.scatter_output_buffers = len(up.x_buffers)
        self._fixed_args = vars(args)
        self.external_scatter_addresses = (
            options.local_gpu_combine and options.cpu_scatter_preparation
            and options.cpu_scatter_publication == "threaded"
            and "scatter_row_addresses" in inspect.signature(pregather.prepare_grouped_inputs).parameters)

    def for_routes(self, routes):
        # Deferred setup and the launch callable retain their iteration's namespace.
        args = argparse.Namespace(**self._fixed_args)
        args.tokens = args.routes = routes
        return args


class DownProjectionRunner:
    """Compose QuACK's existing plain grouped-GEMM runner with the up output."""

    def __init__(self, torch, up, counts, options, trace_configure=None, previous=None, *,
                 defer_wave_schedule=False, external_wave_schedule=None, async_scatter=None, kernel_options=None,
                 registered_scatter=None):
        # GroupedGemmRunner has already inserted the selected QuACK checkout.
        from run import hopper_pregather_gemm as pregather

        routes = sum(counts)
        if len(counts) != up.weights.shape[0] or any(n < 0 for n in counts) or routes > up.output.shape[0]:
            raise ValueError("expert counts do not describe the up-projection output")
        if kernel_options is None:
            kernel_options = _DownKernelOptions(pregather, up, options)
        self.kernel_options = kernel_options
        args = self.kernel_options.for_routes(routes)
        hidden, output_dim = args.hidden, args.output_dim
        self.combine = options.local_gpu_combine
        self.output_reset_policy = options.down_output_reset
        self.cpu_scatter = options.cpu_scatter_preparation and routes > 0
        self.cpu_scatter_publication = options.cpu_scatter_publication or "deferred"
        self.cpu_scatter_metadata = None
        self._cpu_scatter_wait = None
        self.cpu_scatter_timeout_ms = (max(1, math.ceil(options.return_token_timeout * 1000))
                                       if self.cpu_scatter else 0)
        self.async_scatter = async_scatter
        self.cpu_scatter_counts = async_scatter.counts if async_scatter is not None else None
        self.up = up
        self.torch = torch
        self.routes = routes
        self.route_weights = None
        weighted = getattr(options, "weighted_contributions", False)
        if weighted:
            if up.route_weights is None:
                raise RuntimeError("weighted contributions require native per-source routing weights")
        elif getattr(up, "route_weights", None) is not None:
            raise RuntimeError("native routing weights disagree with --no-weighted-contributions")
        clusters_n = math.ceil(up.gemm_dimension / up.tile_n)
        self.combine_n_groups = clusters_n // min(up.max_swizzle_size, clusters_n)
        if routes:
            validation_args = argparse.Namespace(**vars(args))
            if self.combine:
                # Externally allocated source buffers have independent capacities.
                validation_args.multi_buffer_scatter = False
            pregather.validate_args(validation_args, prepacked=True)
        if previous is None:
            self.weights = torch.randn(
                (len(counts), hidden, output_dim), dtype=up.output.dtype, device=up.output.device)
            self.weights.mul_(1 / math.sqrt(hidden))
        else:
            self.weights = previous.weights
        # A_idx arrays are padded, but only this expert-contiguous prefix is written.
        external = {}
        if self.combine:
            from quack.gemm import build_scatter_row_addresses

            rows = int(up.x_buffers[0].shape[0])
            if previous is None:
                outputs = tuple(torch.empty(
                    (math.ceil(output_dim / args.tile_n), rows * args.tile_n)
                    if args.n_tile_major_output else (rows, output_dim),
                    dtype=up.output.dtype, device=up.output.device,
                ) for _ in up.x_buffers)
                flat = torch.arange(len(outputs) * rows, device=up.output.device)
                all_mapping = torch.stack((flat // rows, flat % rows), dim=1)
                self.combine_addresses = build_scatter_row_addresses(
                    outputs, all_mapping, reduce=True,
                    tile_n=args.tile_n if args.n_tile_major_output else None)
            else:
                # These allocations are also the sources of peer IPC/return descriptors.
                outputs = previous.inputs.output_buffers()
                self.combine_addresses = previous.combine_addresses
            self.combine_rows = rows
            if async_scatter is None:
                self.combine_valid_rows = torch.zeros(
                    len(outputs), dtype=torch.int64, device=up.output.device)
                self.combine_output_to_input = torch.full(
                    (len(outputs), rows), -1, dtype=torch.int64, device=up.output.device)
            else:
                self.combine_valid_rows = async_scatter.counts
                self.combine_output_to_input = async_scatter.inverse
                if not routes:
                    # No scatter worker job for a rank with zero local work.
                    self.combine_valid_rows.zero_()
                    if self.combine_output_to_input is not None:
                        self.combine_output_to_input.fill_(-1)
            self.registered_outputs = outputs
            external["output_buffers"] = outputs
            if previous is not None and self.cpu_scatter and self.kernel_options.external_scatter_addresses:
                # The native publisher fills every live row before down's event wait.
                if registered_scatter is not None:
                    self.cpu_scatter_metadata, addresses, self.route_weights = registered_scatter
                else:
                    address_shape = (routes, 2) if args.n_tile_major_output else (routes,)
                    self.cpu_scatter_metadata, addresses, self.route_weights = allocate_cpu_scatter_metadata_storage(
                        torch, address_shape, up.output.device, weighted,
                        storage=getattr(async_scatter, "address_storage", None))
                external["scatter_row_addresses"] = addresses
            elif routes:
                external["scatter_mapping"] = torch.zeros((routes, 2), dtype=torch.int64, device=up.output.device)
        if weighted:
            if self.route_weights is None:
                self.route_weights = torch.zeros(routes, dtype=torch.float32, device=up.output.device)
            external["route_weights"] = self.route_weights
        needs_ready_address = options.trace_down_ready or options.return_token_results
        if needs_ready_address and trace_configure is None:
            raise ValueError("down readiness requires the proxy trace allocator")
        self.inputs = None
        self._pending_wave_schedule = None
        self._ready_host_address = None
        if (defer_wave_schedule and args.gather_table_publish_ready
                and "scatter_row_addresses" in external):
            if external_wave_schedule is not None:
                external["external_wave_schedule"] = external_wave_schedule
            # These destinations are already bound independently of QuACK's down inputs.
            # Only the registered host flag address is invariant across routing epochs.
            if needs_ready_address:
                self.trace_groups = previous.trace_groups
                self._ready_host_address = previous.inputs.wave_ready_host_address
            self._setup_stream = torch.cuda.current_stream(up.cuda_device_id)
            self._pending_wave_schedule = (args, list(counts), external)
        else:
            if external_wave_schedule is not None:
                raise ValueError("external down waves require deferred QuACK setup and threaded CPU scatter")
            self._prepare_gemm(args, counts, external, trace_configure if needs_ready_address else None)
        self.combine_tile_n = args.tile_n if self.combine and args.n_tile_major_output else None
        self.return_exchange = None
        self.intra_node_combine = None
        self._trace_first_combine_launch = self.combine and previous is None
        self.flops = 2 * routes * hidden * output_dim
        if previous is not None:
            self.reset_start, self.reset_end = previous.reset_start, previous.reset_end
            self.mapping_start, self.gemm_start, self.end = previous.mapping_start, previous.gemm_start, previous.end
            return
        self.reset_start = torch.cuda.Event(enable_timing=True)
        self.reset_end = torch.cuda.Event(enable_timing=True)
        self.mapping_start = torch.cuda.Event(enable_timing=True)
        self.gemm_start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(up.cuda_device_id)
        print(f"DownGEMM configured: device_capacity={up.device_capacity} "
              f"tile=({args.tile_m},{args.tile_n}) cluster_m={args.cluster_m} "
              f"pingpong={args.pingpong} sm_count={args.sm_count} "
              f"wave_work_ids={self.inputs.wave_work_ids} "
              f"bulk_store_config={pregather.bulk_store_config_for_args(args)} "
              f"output_reset={self.output_reset_policy} "
              f"routes={routes} H={hidden} K={output_dim} "
              f"counts={counts} output_shapes="
              f"{[tuple(x.shape) for x in self.inputs.output_buffers()]}", flush=True)
        if self.combine:
            print(f"Local GPU combine: sources={up.source_pairs} "
                  f"rows_per_source={self.combine_rows} epilogue=bulk_rows_reduce "
                  "positions=compact prefixes in input-row order; "
                  f"routing_weights={'FP32' if self.route_weights is not None else 'disabled'}",
                  flush=True)

    def output_buffers(self):
        # IPC and native scatter bind their fixed output bases before wave construction.
        if self.inputs is None:
            return self.registered_outputs
        return self.inputs.output_buffers()

    def _prepare_gemm(self, args, counts, external, trace_configure=None):
        from run import hopper_pregather_gemm as pregather

        up, torch = self.up, self.torch
        if self.routes:
            inputs = pregather.prepare_grouped_inputs(
                args, up.output[:self.routes], self.weights, counts, **external)
        else:
            inputs = EmptyDownInputs(torch, args, up.output, external.get("output_buffers"))
        if (self.cpu_scatter and self.cpu_scatter_metadata is None
                and self.cpu_scatter_publication in ("deferred", "threaded")):
            (self.cpu_scatter_metadata, inputs.scatter_row_addresses,
             self.route_weights) = make_cpu_scatter_metadata_storage(
                torch, inputs.scatter_row_addresses, self.route_weights)
            inputs.route_weights = self.route_weights
        if self._ready_host_address is not None:
            inputs.wave_ready_host_address = self._ready_host_address
        elif trace_configure is not None:
            self.trace_groups = int(inputs.wave_ready_prefix[-1].item())
            address = trace_configure(self.trace_groups)
            inputs.wave_ready_host_address = torch.tensor(
                [address], dtype=torch.int64, device=up.output.device)
        if self.routes:
            self.launch_gemm = pregather.make_launch(args, inputs)
        else:
            self.publish_empty_down = None
            if inputs.wave_ready_host_address is not None:
                from empty_down import prepare_empty_down_publisher
                self.publish_empty_down = prepare_empty_down_publisher(
                    inputs.wave_ready_host_address, inputs.groups, torch.cuda.current_stream(up.cuda_device_id))
            self.launch_gemm = self._complete_empty_down
        self.inputs = inputs

    def finish_wave_schedule(self, wait_for_metadata):
        """Order scatter readiness and finish any deferred work before down reads it."""
        # A QuACK allocation can synchronize the device. First let the native
        # publisher submit every ready row that the resident up kernel can wait on.
        # The scatter event is recorded after the last such submission.
        if wait_for_metadata is not None:
            wait_for_metadata()
        if self.inputs is not None:
            return
        if wait_for_metadata is None:
            raise RuntimeError("deferred down-wave setup requires the current scatter publication")
        compute_stream = self.torch.cuda.current_stream(self.up.cuda_device_id)
        try:
            with self.torch.cuda.stream(self._setup_stream):
                self._prepare_gemm(*self._pending_wave_schedule)
        finally:
            # Include partially submitted setup work in compute-stream teardown, too.
            compute_stream.wait_stream(self._setup_stream)
        self._pending_wave_schedule = None

    def warmup(self):
        """Initialize the down CUDA path before any readiness-gated up launch."""
        torch = self.up.torch
        if isinstance(self.inputs, EmptyDownInputs):
            self.reset_outputs()
            torch.cuda.synchronize(self.up.cuda_device_id)
            return
        print(f"[gpu={self.up.cuda_device_id}] DownGEMM warming up before concurrent iterations", flush=True)
        with torch.inference_mode():
            self.up.output.zero_()
            self.inputs.zero_up_output()
            if self.combine:
                # Exercise the full mapping path without reading the unpublished
                # live table. One synthetic batch covers all routes; remaining
                # rows are empty. Preserve production shapes, strides and dtypes.
                table = torch.empty_like(self.up.work_table)
                table.zero_()
                indices = []
                remaining = self.routes
                for source, real_idx in enumerate(self.up.a_idx):
                    count = min(remaining, real_idx.numel())
                    remaining -= count
                    table[:, 2 + 2 * source:4 + 2 * source] = count
                    table[:self.combine_n_groups, 2 + 2 * source] = 0
                    idx = torch.arange(real_idx.numel(), device=real_idx.device,
                                       dtype=real_idx.dtype) % self.combine_rows
                    indices.append(idx)
                if remaining:
                    raise RuntimeError("combine warmup routes exceed A_idx capacity")
                self._prepare_combine(table, tuple(indices))
                print(f"[gpu={self.up.cuda_device_id}] Combine mapping warmup submitted",
                      flush=True)
            # Warm the selected down kernel using the prepared scatter addresses.
            self.launch_gemm()
            for event in (self.reset_start, self.reset_end, self.mapping_start, self.gemm_start, self.end):
                event.record()
            if self.output_reset_policy == "end-of-iteration":
                # Warmup leaves additive results in the destinations. Establish
                # zeros for the first measured iteration before dropping its reset.
                self.reset_outputs()
        torch.cuda.synchronize(self.up.cuda_device_id)
        print(f"[gpu={self.up.cuda_device_id}] DownGEMM warmup complete", flush=True)

    def _log_combine_startup(self, message):
        if self._trace_first_combine_launch:
            print(f"[gpu={self.up.cuda_device_id}] combine startup: {message}", flush=True)

    def _complete_empty_down(self):
        if self.inputs.ready_groups is None:
            return
        self.inputs.ready_groups.fill_(self.inputs.groups)
        if self.publish_empty_down is not None:
            self.publish_empty_down(self.up.torch.cuda.current_stream(self.up.cuda_device_id))

    def configure_cpu_scatter(self, library, handle, *, already_registered=False):
        """Bind native scatter and configure the down stream's publication wait."""
        torch = self.up.torch
        sources = len(self.up.source_pairs)
        lookup = getattr(self, "async_scatter", None)
        if lookup is not None:
            counts = lookup.counts
            if not already_registered:
                lookup.register(library, handle, self.routes, self.cpu_scatter_metadata, self.route_weights)
        else:
            counts = torch.empty(sources, dtype=torch.int64)
            inverse = torch.empty((sources, self.combine_rows), dtype=torch.int64)
            config = CpuScatterConfig.for_outputs(self, counts, inverse)
            addresses = self.cpu_scatter_metadata if self.inputs is None else self.inputs.scatter_row_addresses
            config.routes = self.routes
            config.scatter_addresses = addresses.data_ptr()
            try:
                configure = library.rdma_proxy_configure_cpu_scatter
            except AttributeError as error:
                raise RuntimeError("--cpu-scatter-preparation requires rebuilding librdma_cpu_proxy") from error
            configure.argtypes = [ctypes.c_void_p, ctypes.POINTER(CpuScatterConfig)]
            configure.restype = ctypes.c_int
            require_proxy_success(library, configure(handle, ctypes.byref(config)),
                                  "configuring CPU scatter preparation")
        if self.route_weights is not None and lookup is None:
            try:
                configure_weights = library.rdma_proxy_configure_cpu_route_weights
            except AttributeError as error:
                raise RuntimeError("weighted CPU preparation requires rebuilding librdma_cpu_proxy") from error
            configure_weights.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64]
            configure_weights.restype = ctypes.c_int
            require_proxy_success(library, configure_weights(
                handle, self.route_weights.data_ptr(), self.routes),
                "configuring CPU packed routing weights")
        if self.cpu_scatter_publication in ("deferred", "threaded"):
            try:
                wait_ready = library.rdma_proxy_wait_cpu_scatter_ready
                if lookup is None:
                    configure_publication = getattr(
                        library, f"rdma_proxy_configure_cpu_scatter_{self.cpu_scatter_publication}")
            except AttributeError as error:
                raise RuntimeError(
                    f"{self.cpu_scatter_publication} CPU scatter requires rebuilding librdma_cpu_proxy") from error
            wait_ready.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_uint64]
            wait_ready.restype = ctypes.c_int
            if lookup is None:
                configure_publication.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
                configure_publication.restype = ctypes.c_int
                require_proxy_success(library, configure_publication(handle, self.cpu_scatter_metadata.numel()),
                                      f"configuring {self.cpu_scatter_publication} CPU scatter publication")
            self._cpu_scatter_wait = wait_ready
            self._cpu_scatter_library, self._cpu_scatter_handle = library, handle
        if lookup is None:
            with torch.inference_mode():
                self.combine_valid_rows.copy_(counts)
                self.combine_output_to_input.copy_(inverse)
            torch.cuda.synchronize(self.up.cuda_device_id)
        self.cpu_scatter_counts = counts
        if self.cpu_scatter_publication == "threaded":
            policy = ("dedicated CPU scatter thread; "
                      f"one {self.cpu_scatter_metadata.numel()}-byte metadata H2D after final gather publication")
        elif self.cpu_scatter_publication == "deferred":
            policy = (f"CPU packing then one {self.cpu_scatter_metadata.numel()}-byte "
                      "metadata H2D after final gather publication")
        else:
            policy = "scatter batches published before gather readiness"
        wave_setup = " quack_down_setup=after_up_submit" if self.inputs is None else ""
        print(f"CPU scatter preparation enabled: compact metadata bound to current routing; {policy}{wave_setup}",
              flush=True)

    def wait_cpu_scatter_ready(self, iteration):
        """Join the current native upload after its event record has been submitted.

        Waiting on a CUDA event before the producer records this iteration would
        observe an old/unrecorded event. The native API first waits for host-side
        submission, then queues the event dependency without synchronizing the GPU.
        """
        if self._cpu_scatter_wait is None:
            raise RuntimeError("CPU scatter publication wait is not configured")
        stream = self.up.torch.cuda.current_stream(self.up.cuda_device_id)
        require_proxy_success(self._cpu_scatter_library, self._cpu_scatter_wait(
            self._cpu_scatter_handle, iteration, int(stream.cuda_stream), self.cpu_scatter_timeout_ms),
            f"waiting for CPU scatter metadata for iteration {iteration}")

    def _prepare_combine(self, work_table, a_idx):
        torch = self.up.torch
        source, index = local_route_positions(
            torch, work_table, len(a_idx), self.routes, self.combine_n_groups,
            trace=self._log_combine_startup if self._trace_first_combine_launch else None)
        mapping = torch.stack((source, torch.stack(a_idx)[source, index].long()), dim=1)
        if self.route_weights is not None:
            self.route_weights.copy_(self.up.route_weights_stacked[source, index])
        self._log_combine_startup("route mapping submitted; compaction begin")
        mapping, valid_rows, output_to_input = compact_combine_mapping(
            self.up.torch, mapping, len(self.up.x_buffers), self.combine_rows,
        )
        self._log_combine_startup("compaction submitted; scatter address updates begin")
        self.combine_valid_rows.copy_(valid_rows)
        self.combine_output_to_input.copy_(output_to_input)
        flat = mapping[:, 0] * self.combine_rows + mapping[:, 1]
        self.inputs.scatter_mapping.copy_(mapping)
        self.inputs.scatter_table.copy_(flat)
        addresses = self.combine_addresses[flat]
        if self.combine_tile_n is not None:
            # Tile zero's addresses are unchanged. Pack subsequent N tiles
            # using the distinct valid-token count, not allocation capacity.
            addresses[:, 1] = (valid_rows[mapping[:, 0]] * self.combine_tile_n
                               * self.up.output.element_size())
        self.inputs.scatter_row_addresses.copy_(addresses)

    def reset_outputs(self):
        """Clear only down destinations on the caller's stream, with independent timing."""
        with self.up.torch.inference_mode():
            self.reset_start.record()
            if self.inputs is None:
                # The deferred path owns fixed, full-capacity scatter destinations.
                for output in self.registered_outputs:
                    output.zero_()
            else:
                self.inputs.zero_up_output()
            self.reset_end.record()

    def finish_iteration(self):
        """Reset after all output readers/checks have joined the caller's stream.

        Return copies, combine streams, peer validation and projection checks
        must finish before this call. The scoped wait charges this reset to the
        completed iteration and leaves the destinations ready for the next one.
        """
        if self.output_reset_policy == "end-of-iteration":
            self.reset_outputs()
            self.reset_end.synchronize()

    def launch(self, wait_for_metadata=None, wait_for_wave_schedule=None, before_combine=None):
        # QuACK prepares output and metadata tensors under inference_mode.
        # Keep their resets and in-place updates in the same mode.
        with self.up.torch.inference_mode():
            if self.output_reset_policy == "before-down":
                self._log_combine_startup("reset begin")
                self.reset_outputs()
            self.mapping_start.record()
            self.finish_wave_schedule(wait_for_metadata)
            if wait_for_wave_schedule is not None:
                # QuACK setup inspected only the views' layout. Join the native
                # upload on this compute stream before down uses their contents.
                wait_for_wave_schedule()
            if self.combine and not self.cpu_scatter and not isinstance(self.inputs, EmptyDownInputs):
                self._log_combine_startup("route mapping begin")
                self._prepare_combine(self.up.work_table, self.up.a_idx)
            elif self.route_weights is not None and not self.cpu_scatter and not isinstance(self.inputs, EmptyDownInputs):
                source, index = local_route_positions(
                    self.up.torch, self.up.work_table, len(self.up.a_idx),
                    self.routes, self.combine_n_groups)
                self.route_weights.copy_(self.up.route_weights_stacked[source, index])
            if self.return_exchange is not None:
                self.return_exchange.publish_counts()
            self.gemm_start.record()
            self._log_combine_startup("scatter updates submitted")
            if self.intra_node_combine is not None:
                self._log_combine_startup("intra-node mapping-ready event record begin")
                self.intra_node_combine.mapping_ready.record()
                self._log_combine_startup("intra-node mapping-ready event record returned")
            self._log_combine_startup("down GEMM launch begin")
            self.launch_gemm()
            self._log_combine_startup("down GEMM launch returned")
            if before_combine is not None:
                # The host may wait for map submission after down has already finished.
                self.end.record()
            if self.intra_node_combine is not None:
                if before_combine is not None:
                    before_combine()
                self.intra_node_combine.launch()
            if before_combine is None:
                self.end.record()
            self._trace_first_combine_launch = False

    def collected_combine_outputs(self):
        """Return one row-major valid tensor per source for inspection/testing.

        Synchronizes this runner's last launch. The timed path uses device-side
        counts and packed buffers directly, without this host count readback.
        QuACK's generic collected_up_output assumes capacity-based tile strides
        and must not be used for these dynamically packed combine outputs.
        """
        if not self.combine:
            raise ValueError("combine outputs require --local-gpu-combine")
        self.end.synchronize()
        counts = self.combine_valid_rows.tolist()
        width = int(self.up.x_buffers[0].shape[1])
        result = []
        for buffer, count in zip(self.inputs.output_buffers(), counts):
            if self.combine_tile_n is None:
                result.append(buffer[:count])
            else:
                tile = self.combine_tile_n
                n_tiles = math.ceil(width / tile)
                result.append(buffer.view(-1)[:n_tiles * count * tile]
                              .view(n_tiles, count, tile).permute(1, 0, 2)
                              .reshape(count, n_tiles * tile)[:, :width])
        return tuple(result)


def get_expert_counts(library, handle):
    try:
        query = library.rdma_proxy_get_computation_expert_counts
    except AttributeError as error:
        raise RuntimeError("--down-projection2 requires rebuilding librdma_cpu_proxy") from error
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64),
                      ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
    query.restype = ctypes.c_int
    size = ctypes.c_uint64()
    require_proxy_success(library, query(handle, None, 0, ctypes.byref(size)), "querying expert count")
    counts = (ctypes.c_uint64 * size.value)()
    require_proxy_success(library, query(handle, counts, size.value, ctypes.byref(size)),
                          "querying per-expert routed rows")
    return list(counts)


def environment_integer(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise RuntimeError(f"{name} is not set; launch this worker with torchrun")
    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from error


def comma_separated_values(name: str, value: str, expected_count: int) -> list[str]:
    values = [item.strip() for item in value.split(",")]
    if any(not item for item in values):
        raise RuntimeError(f"{name} contains an empty entry: {value!r}")
    if len(values) != expected_count:
        raise RuntimeError(
            f"{name} has {len(values)} entries, but torchrun launched "
            f"LOCAL_WORLD_SIZE={expected_count} workers"
        )
    return values


def build_proxy_argv(
    args: argparse.Namespace,
    proxy_args: list[str],
) -> tuple[list[str], dict[str, int | str]]:
    local_rank = environment_integer("LOCAL_RANK")
    local_world_size = environment_integer("LOCAL_WORLD_SIZE")
    rank = environment_integer("RANK")
    world_size = environment_integer("WORLD_SIZE")

    if args.launcher_local_rank is not None and args.launcher_local_rank != local_rank:
        raise RuntimeError(
            f"--local-rank={args.launcher_local_rank} disagrees with LOCAL_RANK={local_rank}"
        )
    if local_world_size <= 0 or world_size <= 0:
        raise RuntimeError("LOCAL_WORLD_SIZE and WORLD_SIZE must be positive")
    if world_size % local_world_size != 0:
        raise RuntimeError(
            f"WORLD_SIZE={world_size} is not divisible by LOCAL_WORLD_SIZE={local_world_size}"
        )
    if local_rank < 0 or local_rank >= local_world_size:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside [0, {local_world_size})"
        )

    num_nodes = world_size // local_world_size
    group_rank_value = os.environ.get("GROUP_RANK")
    node_rank = int(group_rank_value) if group_rank_value is not None else rank // local_world_size
    if node_rank < 0 or node_rank >= num_nodes:
        raise RuntimeError(f"derived node rank {node_rank} is outside [0, {num_nodes})")

    cuda_devices = comma_separated_values(
        "--cuda-device-map", args.cuda_device_map, local_world_size
    )
    rdma_devices = comma_separated_values(
        "--rdma-device-map", args.rdma_device_map, local_world_size
    )
    # Commas separate GPUs; '+' separates the two devices owned by one GPU.
    device_pairs = [[name.strip() for name in entry.split("+")] for entry in rdma_devices]
    for pair in device_pairs:
        if len(pair) not in (1, 2) or any(not name.strip() for name in pair) or len(set(pair)) != len(pair):
            raise RuntimeError("--rdma-device-map requires one device or two distinct devices joined by '+' per GPU")
    local_devices = device_pairs[local_rank]
    try:
        cuda_device_id = int(cuda_devices[local_rank])
    except ValueError as error:
        raise RuntimeError(
            f"CUDA device map entry {cuda_devices[local_rank]!r} is not an integer"
        ) from error

    listen_port = args.listen_port_base + local_rank
    if listen_port < 1 or listen_port > 65535:
        raise RuntimeError(f"derived proxy listen port {listen_port} is invalid")

    # Generated values are appended so torchrun's topology and the per-rank
    # device/NIC maps take precedence over defaults in the JSON configuration.
    generated_args = [
        f"--node_rank={node_rank}",
        f"--num_nodes={num_nodes}",
        f"--num_gpus_per_node={local_world_size}",
        f"--local_gpu_index={local_rank}",
        f"--cuda_device_id={cuda_device_id}",
        f"--rdma_device_name={local_devices[0]}",
        f"--rdma_device_name2={local_devices[1] if len(local_devices) == 2 else ''}",
        f"--listen_port={listen_port}",
        "--router_weighted_contributions_enabled="
        + str(getattr(args, "weighted_contributions", False)).lower(),
    ]
    if args.num_experts is not None:
        generated_args.append(f"--num_experts={args.num_experts}")
    if getattr(args, "pooled_up_gemm", None) is not None:
        generated_args.append(f"--router_pooled_up_gemm_enabled={str(args.pooled_up_gemm).lower()}")
    dynamic = getattr(args, "dynamic_routing", None)
    if getattr(args, "benchmark_metadata_exchange", None):
        if dynamic is False:
            raise ValueError("--benchmark-metadata-exchange requires dynamic routing")
        dynamic = True
    if dynamic is not None:
        generated_args.append(f"--router_dynamic_routing_enabled={str(dynamic).lower()}")
    if getattr(args, "benchmark_metadata_exchange", None) is not None:
        generated_args.append(
            f"--router_benchmark_metadata_enabled={str(args.benchmark_metadata_exchange).lower()}")
    if getattr(args, "benchmark_metadata_validation", None) is not None:
        generated_args.append(
            f"--router_benchmark_metadata_validation_enabled={str(args.benchmark_metadata_validation).lower()}")
    if args.concurrent_kernel:
        generated_args.extend(
            [
                "--router_computation_forwarded_inputs_only=true",
                "--router_local_input_staging_enabled=true",
                f"--expert_gemm_m_tile={args.expert_gemm_m_tile}",
                f"--expert_gemm_n_tile={args.expert_gemm_n_tile}",
                f"--expert_gemm_dimension={projection_dimension(args.expert_gemm_dimension, args.grouped_gemm_activation)}",
                f"--expert_gemm_cluster_m={args.expert_gemm_cluster_m}",
                f"--expert_gemm_max_swizzle_size={args.expert_gemm_max_swizzle_size}",
            ]
        )
        architecture = getattr(args, "gemm_arch", "auto")
        if architecture != "auto":
            generated_args.append(
                f"--expert_gemm_device_capacity={9 if architecture == 'hopper' else 10}")
    if args.inter_node_combine and num_nodes > 1 and not args.rdma_return_token_results:
        raise ValueError("--inter-node-combine requires --rdma-return-token-results with multiple nodes")
    if args.rdma_return_token_results and num_nodes < 2:
        raise ValueError("--rdma-return-token-results requires at least two nodes")
    if args.intra_node_combine:
        generated_args.extend([
            "--nvlink_return_completion_notifications_enabled=true",
            "--nvlink_return_ready_gpu_publication_enabled=true",
        ])
    metadata: dict[str, int | str] = {
        "rank": rank,
        "node_rank": node_rank,
        "local_rank": local_rank,
        "cuda_device_id": cuda_device_id,
        "rdma_device_name": rdma_devices[local_rank],
        "listen_port": listen_port,
        "world_size": world_size,
        "num_nodes": num_nodes,
    }
    return ["rdma_cpu_proxy", *proxy_args, *generated_args], metadata


def check_cuda_runtime_compatibility(torch_cuda_version, library_path, *, mappings=None):
    """Reject mixed runtimes before creating a proxy or starting peer exchanges.

    Match the major ABI, not the patch version. Linux maps repeat a library for
    each segment; the unique path count also preserves QuACK's IPC contract.
    """
    if mappings is None:
        maps_path = Path("/proc/self/maps")
        if not maps_path.exists():
            return  # The real embedded CUDA/RDMA path runs on Linux.
        mappings = maps_path.read_text()
    runtimes = {}
    for line in mappings.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        mapped_path = fields[5]
        path = mapped_path.removesuffix(" (deleted)")
        match = re.fullmatch(r"libcudart\.so(?:\.(\d+)(?:\..*)?)?", Path(path).name)
        if match:
            runtimes[mapped_path] = int(match[1]) if match[1] else None
    expected_major = int(torch_cuda_version.split(".")[0])
    if len(runtimes) != 1 or next(iter(runtimes.values()), None) != expected_major:
        raise RuntimeError(
            f"CUDA runtime mismatch before proxy initialization: PyTorch was built for "
            f"CUDA {torch_cuda_version}, but after loading {library_path} the process has "
            f"these CUDA runtimes: {sorted(runtimes)}. Expected exactly one libcudart.so."
            f"{expected_major}. Rebuild the proxy in a fresh build directory using the "
            f"CUDA {torch_cuda_version} toolkit: set both CMAKE_CUDA_COMPILER to its "
            "bin/nvcc and CUDAToolkit_ROOT to that toolkit root, then point "
            "RDMA_PROXY_BUILD_DIR at the new build. Do not bypass QuACK's runtime check."
        )


def load_proxy_library(path: Path, *, torch_cuda_version=None) -> ctypes.CDLL:
    if not path.is_file():
        raise RuntimeError(
            f"RDMA proxy shared library does not exist: {path}; "
            "build the rdma_cpu_proxy_shared target first"
        )

    library = ctypes.CDLL(str(path))
    if torch_cuda_version is not None:
        check_cuda_runtime_compatibility(torch_cuda_version, path)
    library.rdma_proxy_abi_version.argtypes = []
    library.rdma_proxy_abi_version.restype = ctypes.c_int
    library.rdma_proxy_run_argv.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    library.rdma_proxy_run_argv.restype = ctypes.c_int
    library.rdma_proxy_create.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    library.rdma_proxy_create.restype = ctypes.c_void_p
    library.rdma_proxy_set_device_buffer_allocator.argtypes = [
        ctypes.c_void_p,
        DeviceBufferAllocatorCallback,
        ctypes.c_void_p,
    ]
    library.rdma_proxy_set_device_buffer_allocator.restype = ctypes.c_int
    library.rdma_proxy_initialize.argtypes = [ctypes.c_void_p]
    library.rdma_proxy_initialize.restype = ctypes.c_int
    library.rdma_proxy_run.argtypes = [ctypes.c_void_p]
    library.rdma_proxy_run.restype = ctypes.c_int
    library.rdma_proxy_get_num_iterations.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rdma_proxy_get_num_iterations.restype = ctypes.c_int
    library.rdma_proxy_prepare_iteration.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
    ]
    library.rdma_proxy_prepare_iteration.restype = ctypes.c_int
    library.rdma_proxy_record_computation_end.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    library.rdma_proxy_record_computation_end.restype = ctypes.c_int
    library.rdma_proxy_get_computation_elapsed_ms.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.rdma_proxy_get_computation_elapsed_ms.restype = ctypes.c_int
    library.rdma_proxy_get_computation_num_tokens.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rdma_proxy_get_computation_num_tokens.restype = ctypes.c_int
    library.rdma_proxy_run_iteration.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    library.rdma_proxy_run_iteration.restype = ctypes.c_int
    for name in ("rdma_proxy_run_iteration_payload", "rdma_proxy_complete_iteration"):
        fn = getattr(library, name, None)
        if fn is not None:
            fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
            fn.restype = ctypes.c_int
    library.rdma_proxy_finish.argtypes = [ctypes.c_void_p]
    library.rdma_proxy_finish.restype = ctypes.c_int
    library.rdma_proxy_shutdown.argtypes = [ctypes.c_void_p]
    library.rdma_proxy_shutdown.restype = ctypes.c_int
    library.rdma_proxy_destroy.argtypes = [ctypes.c_void_p]
    library.rdma_proxy_destroy.restype = None
    library.rdma_proxy_last_error.argtypes = []
    library.rdma_proxy_last_error.restype = ctypes.c_char_p

    abi_version = library.rdma_proxy_abi_version()
    if abi_version != EXPECTED_ABI_VERSION:
        raise RuntimeError(
            f"RDMA proxy ABI mismatch: worker expects {EXPECTED_ABI_VERSION}, "
            f"library provides {abi_version}"
        )
    return library


def encode_proxy_argv(proxy_argv: list[str]) -> ctypes.Array:
    encoded_argv = [os.fsencode(argument) for argument in proxy_argv]
    return (ctypes.c_char_p * len(encoded_argv))(*encoded_argv)


def proxy_last_error(library: ctypes.CDLL) -> str:
    error_pointer = library.rdma_proxy_last_error()
    return (
        error_pointer.decode("utf-8", errors="replace")
        if error_pointer
        else "unknown proxy error"
    )


def require_proxy_success(library: ctypes.CDLL, status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed: {proxy_last_error(library)}")


def import_torch_for_proxy(cuda_device_id: int):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("the Python-owned buffer path requires PyTorch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("the Python-owned buffer path requires an available CUDA device")
    torch.cuda.set_device(cuda_device_id)
    return torch


def install_python_device_buffers(
    library: ctypes.CDLL,
    handle: int,
    torch_module,
    cuda_device_id: int,
    *,
    node_rank: int,
    local_gpu_index: int,
) -> TorchDeviceBuffers:
    buffers = TorchDeviceBuffers(
        torch_module,
        cuda_device_id,
        node_rank=node_rank,
        local_gpu_index=local_gpu_index,
        num_gpus_per_node=(environment_integer("LOCAL_WORLD_SIZE") if "LOCAL_WORLD_SIZE" in os.environ
                           else None),
    )
    require_proxy_success(
        library,
        library.rdma_proxy_set_device_buffer_allocator(
            handle, buffers.callback, None
        ),
        "installing Python device-buffer allocator",
    )
    return buffers


# A peer may still own an IPC mapping if native teardown could not confirm
# its close. Keep that allocation unavailable to the torch caching allocator.
_unclosed_input_pools = []


def shutdown_embedded_proxy(library, handle, buffers, initialized, success, return_exchange=None,
                            benchmark_metadata=None):
    pools = buffers.tensors_of_kind(DEVICE_BUFFER_INPUT_POOL) if buffers is not None else []
    error = None
    backend_error = None
    payload_stopped = False
    try:
        # A partially initialized proxy can already have exported its pool.
        if initialized or pools:
            payload_stopped = library.rdma_proxy_shutdown(handle) == 0
            if not payload_stopped:
                error = proxy_last_error(library)
                _unclosed_input_pools.extend(pools)
                print(f"proxy shutdown failed: {error}", file=sys.stderr, flush=True)
    finally:
        try:
            if benchmark_metadata is not None:
                benchmark_metadata.close(clean=success and payload_stopped)
        except Exception as exc:
            backend_error = exc
            print(f"benchmark metadata shutdown failed: {exc}", file=sys.stderr, flush=True)
        finally:
            library.rdma_proxy_destroy(handle)
    if return_exchange is not None:
        return_exchange.close(success and error is None and backend_error is None)
    if success and backend_error is not None:
        raise backend_error
    if pools and success and error is not None:
        raise RuntimeError(f"pooled receive teardown failed: {error}")


def run_proxy(
    library: ctypes.CDLL,
    proxy_argv: list[str],
    cuda_device_id: int,
    *,
    node_rank: int,
    local_gpu_index: int,
) -> int:
    c_argv = encode_proxy_argv(proxy_argv)
    torch = import_torch_for_proxy(cuda_device_id)
    handle = library.rdma_proxy_create(len(proxy_argv), c_argv)
    if not handle:
        raise RuntimeError(f"proxy creation failed: {proxy_last_error(library)}")

    initialized = False
    success = False
    benchmark_metadata = None
    device_buffers: TorchDeviceBuffers | None = None
    try:
        device_buffers = install_python_device_buffers(
            library,
            handle,
            torch,
            cuda_device_id,
            node_rank=node_rank,
            local_gpu_index=local_gpu_index,
        )
        initialize_status = library.rdma_proxy_initialize(handle)
        if initialize_status != 0 and device_buffers.allocation_error is not None:
            raise RuntimeError(
                "proxy initialization failed after Python rejected a buffer request: "
                f"{device_buffers.allocation_error}; C++: {proxy_last_error(library)}"
            ) from device_buffers.allocation_error
        require_proxy_success(library, initialize_status, "proxy initialization")
        initialized = True
        device_buffers.finish_initialization()
        benchmark_metadata = open_benchmark_metadata(library, handle, proxy_argv, torch, device_buffers)
        if benchmark_metadata is None:
            dynamic_routing_enabled(library, handle, proxy_argv)
            require_proxy_success(library, library.rdma_proxy_run(handle), "proxy run")
        else:
            # The external backend supplies the route consumed by prepare/run.
            # The monolithic native loop cannot call back into Python to supply it.
            iterations = ctypes.c_uint64()
            require_proxy_success(library, library.rdma_proxy_get_num_iterations(handle, ctypes.byref(iterations)),
                                  "querying num_iterations")
            if not iterations.value:
                raise RuntimeError("benchmark metadata exchange requires a finite, nonzero --num_iterations")
            for iteration in range(iterations.value):
                _, elapsed_ns = benchmark_metadata.prepare(iteration, compute_extents=False)
                require_proxy_success(library, library.rdma_proxy_prepare_iteration(handle, iteration),
                                      f"preparing proxy iteration {iteration}")
                require_proxy_success(library, library.rdma_proxy_run_iteration(handle, iteration),
                                      f"running proxy iteration {iteration}")
                print_benchmark_routing(node_rank, local_gpu_index, iteration, elapsed_ns,
                                        benchmark_metadata.last_stats)
            require_proxy_success(library, library.rdma_proxy_finish(handle), "proxy finish")
        success = True
        return 0
    finally:
        shutdown_embedded_proxy(library, handle, device_buffers, initialized, success,
                                benchmark_metadata=benchmark_metadata)


class DownReadyTrace:
    """The C++ proxy owns the mapped flag and polls it on its receiver thread."""

    def __init__(self, library, handle):
        self.library, self.handle = library, handle
        try:
            self.configure_fn = library.rdma_proxy_configure_down_ready_trace
            self.read_fn = library.rdma_proxy_read_down_ready_trace
        except AttributeError as error:
            raise RuntimeError("--trace-down-ready requires rebuilding librdma_cpu_proxy") from error
        self.configure_fn.argtypes = [ctypes.c_void_p, ctypes.c_int32,
                                      ctypes.POINTER(ctypes.c_size_t)]
        self.configure_fn.restype = ctypes.c_int
        self.read_fn.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double), ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
        ]
        self.read_fn.restype = ctypes.c_int

    def configure(self, groups):
        if hasattr(self, "groups"):
            if self.groups != groups:
                raise ValueError("down column group geometry changed between routing generations")
            return self.address
        self.groups = groups
        address = ctypes.c_size_t()
        require_proxy_success(self.library, self.configure_fn(
            self.handle, groups, ctypes.byref(address)), "configuring down-ready trace")
        self.address = address.value
        return self.address

    def read(self, iteration):
        groups = (ctypes.c_int32 * self.groups)()
        times = (ctypes.c_double * self.groups)()
        count = ctypes.c_uint64()
        require_proxy_success(self.library, self.read_fn(
            self.handle, iteration, groups, times, self.groups, ctypes.byref(count)),
            "reading down-ready trace")
        return [(groups[i], times[i]) for i in range(count.value)]


def optional_router_mode(library, handle, proxy_argv, query_name, config_key, label):
    query = getattr(library, query_name, None)
    if query is None:
        import json
        # ABI v1 libraries predating this mode silently ignore unknown configuration keys.
        enabled = False
        for i, arg in enumerate(proxy_argv):
            if arg in ("--config", "-c") or arg.startswith("--config="):
                path = proxy_argv[i + 1] if arg in ("--config", "-c") else arg.split("=", 1)[1]
                enabled = json.loads(Path(path).read_text()).get(config_key, False)
        for arg in proxy_argv:
            if arg.startswith(f"--{config_key}="):
                enabled = arg.split("=", 1)[1] in ("yes", "1", "true")
        if enabled:
            raise RuntimeError(f"{label} requires rebuilding librdma_cpu_proxy")
        return False
    enabled = ctypes.c_int32()
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32)]
    query.restype = ctypes.c_int
    require_proxy_success(library, query(handle, ctypes.byref(enabled)), f"querying {label}")
    return bool(enabled.value)


def dynamic_routing_enabled(library, handle, proxy_argv):
    return optional_router_mode(library, handle, proxy_argv, "rdma_proxy_dynamic_routing_enabled",
                                "router_dynamic_routing_enabled", "dynamic routing")


def open_benchmark_metadata(library, handle, argv, torch, buffers, *, concurrent_compute=False):
    if optional_router_mode(library, handle, argv, "rdma_proxy_benchmark_metadata_enabled",
                            "router_benchmark_metadata_enabled", "benchmark metadata exchange"):
        from benchmark_metadata_backend import BenchmarkMetadataBackend
        return BenchmarkMetadataBackend(library, handle, torch, buffers, concurrent_compute=concurrent_compute)
    return None


def print_benchmark_routing(node, gpu, iteration, elapsed_ns, stats, up_submitted_ns=None):
    """All spans are rank-local host wall time. Up submission returns before GPU completion."""
    print(f"  routing node={node} gpu={gpu} iteration={iteration} elapsed_us={elapsed_ns / 1000:.3f}",
          flush=True)
    submit = (f" r_ready_to_up_submit_us={(up_submitted_ns - stats['r_ready_ns']) / 1000:.3f}"
              if up_submitted_ns is not None else "")
    print(f"  benchmark_metadata node={node} gpu={gpu} iteration={iteration} "
          f"phase={'first' if iteration == 0 else 'steady'} "
          f"generation_us={stats['generation_elapsed_ns'] / 1000:.3f} "
          f"exchange_staging_us={stats['exchange_staging_elapsed_ns'] / 1000:.3f} "
          f"install_us={stats['install_elapsed_ns'] / 1000:.3f} "
          f"r_ready_to_install_us={stats['r_ready_to_install_elapsed_ns'] / 1000:.3f} "
          f"retire_us={stats['retire_elapsed_ns'] / 1000:.3f} "
          f"reuse_barrier_us={stats['reuse_barrier_ns'] / 1000:.3f} "
          f"release_barrier_us={stats['release_barrier_ns'] / 1000:.3f}{submit} "
          f"source_route={stats['source_route']} source_route_storage={stats['source_route_storage']} "
          f"a_idx_storage={stats['a_idx_storage']} early_up_launch={int(stats.get('early_up_launch', False))} "
          f"concurrent_local_publication={int(stats.get('concurrent_local_publication', False))} "
          f"native_local_publication={int(stats.get('native_local_publication', False))} "
          f"native_local_publication_growth={int(stats.get('native_local_publication_growth', False))}",
          flush=True)
    waits = stats.get("completion_wait_ns", {})
    if waits:
        fields = [f"{name}={waits[phase] / 1000:.3f}" for phase, name in
                  (("done", "return_done_us"), ("intra-node-done", "combine_done_us")) if phase in waits]
        print(f"  local_completion node={node} gpu={gpu} iteration={iteration} "
              f"{' '.join(fields)}", flush=True)
    marks = stats.get("local_publication_marks")
    if marks is not None:
        submitted = previous = marks["submit"]
        fields = []
        for phase in ("handoff", "setup_context", "expert_counts", "scatter_register", "scatter_views"):
            if phase in marks:
                fields.append(f"{phase}_us={(marks[phase] - previous) / 1000:.3f}")
                previous = marks[phase]
        if "request_end" in marks:
            fields.append(f"submit_to_request_us={(marks['request_begin'] - submitted) / 1000:.3f}")
            fields.append(f"request_call_us={(marks['request_end'] - marks['request_begin']) / 1000:.3f}")
            previous = marks["request_end"]
        for phase in ("down_wave_start", "down_runner"):
            if phase in marks:
                fields.append(f"{phase}_us={(marks[phase] - previous) / 1000:.3f}")
                previous = marks[phase]
        print(f"  local_publication_timing node={node} gpu={gpu} iteration={iteration} "
              f"r_ready_to_submit_us={(submitted - stats['r_ready_ns']) / 1000:.3f} "
              f"{' '.join(fields)}",
              flush=True)


def prepare_routing(library, handle, iteration):
    prepare = library.rdma_proxy_prepare_routing
    prepare.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    prepare.restype = ctypes.c_int
    begin = time.perf_counter_ns()
    status = prepare(handle, iteration)
    elapsed_ns = time.perf_counter_ns() - begin
    require_proxy_success(library, status, f"preparing routing generation {iteration}")
    extents = library.rdma_proxy_get_routing_extents
    extents.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
    extents.restype = ctypes.c_int
    indices, rows = ctypes.c_uint64(), ctypes.c_uint64()
    require_proxy_success(library, extents(handle, ctypes.byref(indices), ctypes.byref(rows)),
                          "querying current routing extents")
    return (indices.value, rows.value), elapsed_ns


def run_proxy_with_concurrent_kernel(
    library: ctypes.CDLL,
    proxy_argv: list[str],
    cuda_device_id: int,
    *,
    node_rank: int,
    local_gpu_index: int,
    world_size: int,
    num_experts: int,
    expert_gemm_dimension: int,
    activation: str | None,
    expert_gemm_m_tile: int,
    expert_gemm_n_tile: int,
    expert_gemm_k_tile: int | None,
    expert_gemm_cluster_m: int,
    expert_gemm_max_swizzle_size: int,
    quack_root: Path,
    down_options: argparse.Namespace,
    up_sm_count: int | None = None,
) -> int:
    """Launch readiness-gated QuACK GroupedGEMM with each proxy iteration."""

    # Validate before native initialization can allocate buffers or contact peers.
    # Like down GEMM, oversized budgets are accepted and capped by QuACK's
    # hardware-aware cluster capacity; this is not physical SM reservation.
    validate_up_sm_count(up_sm_count, expert_gemm_cluster_m)
    run_payload = library.rdma_proxy_run_iteration
    complete_iteration = None
    if down_options.inter_node_combine:
        try:
            run_payload = library.rdma_proxy_run_iteration_payload
            complete_iteration = library.rdma_proxy_complete_iteration
        except AttributeError as error:
            raise RuntimeError("--inter-node-combine requires a rebuilt native proxy on all ranks "
                               "to signal iteration_done after GPU completion") from error
    torch = import_torch_for_proxy(cuda_device_id)
    capacity = resolve_gemm_architecture(
        getattr(down_options, "gemm_arch", "auto"),
        torch.cuda.get_device_capability(cuda_device_id))
    if down_options.pooled_up_gemm and capacity != 9:
        raise ValueError("--pooled-up-gemm requires Hopper SM90")
    if down_options.down_projection2:
        resolve_down_pingpong(down_options.down_pingpong, capacity,
                              down_options.down_gather_table_wave_barrier)
    # Resolve before C++ allocates tables: SM100 MMA M is not per-CTA M.
    proxy_argv = [*proxy_argv, f"--expert_gemm_device_capacity={capacity}"]
    c_argv = encode_proxy_argv(proxy_argv)
    handle = library.rdma_proxy_create(len(proxy_argv), c_argv)
    if not handle:
        raise RuntimeError(f"proxy creation failed: {proxy_last_error(library)}")

    initialized = False
    coordinator = None
    device_buffers: TorchDeviceBuffers | None = None
    trace = None
    return_exchange = None
    benchmark_metadata = None
    success = False
    kernel_stream = None
    down = None
    down_schedule = None
    async_scatter = None
    return_maps = None
    early_up_iteration = None
    try:
        device_buffers = install_python_device_buffers(
            library,
            handle,
            torch,
            cuda_device_id,
            node_rank=node_rank,
            local_gpu_index=local_gpu_index,
        )
        initialize_status = library.rdma_proxy_initialize(handle)
        if initialize_status != 0 and device_buffers.allocation_error is not None:
            raise RuntimeError(
                "proxy initialization failed after Python rejected a buffer request: "
                f"{device_buffers.allocation_error}; C++: {proxy_last_error(library)}"
            ) from device_buffers.allocation_error
        require_proxy_success(library, initialize_status, "proxy initialization")
        initialized = True
        device_buffers.finish_initialization()

        # Older ABI-v1 libraries ignore unrecognized native CLI keys. A pooled
        # timing run must fail explicitly if it would exercise the default path.
        if down_options.pooled_up_gemm and not device_buffers.tensors_of_kind(DEVICE_BUFFER_INPUT_POOL):
            raise RuntimeError("--pooled-up-gemm requires a native library with pooled receive support")
        num_iterations = ctypes.c_uint64()
        require_proxy_success(
            library,
            library.rdma_proxy_get_num_iterations(handle, ctypes.byref(num_iterations)),
            "querying num_iterations",
        )
        if num_iterations.value == 0:
            raise RuntimeError(
                "--concurrent-kernel requires a finite, nonzero --num_iterations"
            )

        benchmark_metadata = open_benchmark_metadata(
            library, handle, proxy_argv, torch, device_buffers, concurrent_compute=True)
        dynamic = benchmark_metadata is not None or dynamic_routing_enabled(library, handle, proxy_argv)
        prepare_epoch = (benchmark_metadata.prepare if benchmark_metadata is not None
                         else partial(prepare_routing, library, handle))
        routing_times_ns = []
        benchmark_timings = []
        extents = None
        if dynamic:
            if (down_options.down_projection2 and down_options.cpu_scatter_preparation
                    and down_options.cpu_scatter_publication == "threaded"):
                # Older ABI-v1 libraries allocate staging during scatter configuration.
                reserve = getattr(library, "rdma_proxy_reserve_threaded_cpu_scatter", None)
                if reserve is not None:
                    reserve.argtypes = [ctypes.c_void_p, ctypes.c_int32]
                    reserve.restype = ctypes.c_int
                    columns = 2 if down_options.down_n_tile_major_output else 1
                    require_proxy_success(library, reserve(handle, columns), "reserving threaded CPU scatter staging")
            extents, elapsed = prepare_epoch(0)
            routing_times_ns.append(elapsed)
        grouped_gemm = GroupedGemmRunner(
            torch,
            device_buffers,
            cuda_device_id=cuda_device_id,
            world_size=world_size,
            num_experts=num_experts,
            output_dimension=expert_gemm_dimension,
            activation=activation,
            tile_m=expert_gemm_m_tile,
            tile_n=expert_gemm_n_tile,
            tile_k=expert_gemm_k_tile,
            cluster_m=expert_gemm_cluster_m,
            max_swizzle_size=expert_gemm_max_swizzle_size,
            quack_root=quack_root,
            sm_count=up_sm_count,
            routing_extents=extents,
        )
        computation_tokens = ctypes.c_uint64()
        require_proxy_success(
            library,
            library.rdma_proxy_get_computation_num_tokens(
                handle, ctypes.byref(computation_tokens)
            ),
            "querying GroupedGEMM routed-token count",
        )
        if computation_tokens.value == 0 and not dynamic:
            raise RuntimeError("GroupedGEMM has zero meaningful routed tokens")
        hidden_dimension = int(grouped_gemm.x_buffers[0].shape[1])
        kernel_stream = torch.cuda.Stream(device=cuda_device_id)
        down = None
        if down_options.down_projection2:
            counts = get_expert_counts(library, handle)
            if sum(counts) != computation_tokens.value:
                raise RuntimeError("per-expert counts disagree with total routed rows")
            if down_options.trace_down_ready or down_options.return_token_results:
                trace = DownReadyTrace(library, handle)
            down = DownProjectionRunner(
                torch, grouped_gemm, counts, down_options,
                trace_configure=trace.configure if trace is not None else None)
            if dynamic and num_iterations.value > 1:
                from down_wave_schedule import DownWaveSchedule
                down_schedule = DownWaveSchedule.reserve_if_supported(
                    torch, library, handle, grouped_gemm, down_options, partial(require_proxy_success, library))
            with torch.cuda.stream(kernel_stream):
                down.warmup()
            if down.cpu_scatter:
                down.configure_cpu_scatter(library, handle)
            if down_options.return_token_results:
                from rdma_return_exchange import ReturnExchange
                return_exchange = ReturnExchange(
                    torch, library, handle, down, down_options,
                    node_rank, local_gpu_index, cuda_device_id, num_iterations.value)
            if dynamic and num_iterations.value > 1:
                async_scatter = AsyncScatterLookup.reserve_if_supported(library, down, down_options, return_exchange)
            if return_exchange is not None and down_options.intra_node_combine:
                from intra_node_combine import IntraNodeCombine
                if async_scatter is not None:
                    from return_maps import ReturnMaps
                    return_maps = ReturnMaps.reserve_if_supported(
                        return_exchange, down.combine_rows, down_options.inter_node_combine,
                        down_options.return_token_timeout)
                    return_exchange.return_maps = return_maps
                if down_options.trace_down_ready:
                    print(f"[gpu={cuda_device_id}] Intra-node combine setup/compilation begin",
                          flush=True)
                down.intra_node_combine = IntraNodeCombine(
                    return_exchange, down_options, dynamic_capacity=down.combine_rows if dynamic else None)
                if down_options.inter_node_combine:
                    from inter_node_combine import InterNodeCombine
                    if down_options.trace_down_ready:
                        print(f"[gpu={cuda_device_id}] Inter-node combine setup/compilation begin",
                              flush=True)
                    down.intra_node_combine.inter_node_combine = InterNodeCombine(
                        down.intra_node_combine, down_options)
            if (down_options.check_projections and down.route_weights is not None
                    and return_exchange is not None):
                from projection_validation import configure_original_reference
                configure_original_reference(
                    grouped_gemm, down, library, handle, return_exchange,
                    num_experts=num_experts)
        iteration_throughputs = []
        pipeline_timings: list[tuple[int, float]] = []
        down_timings = []
        down_traces = []
        kernel_finished = torch.cuda.Event(enable_timing=False, blocking=False)

        prepare_early = getattr(library, "rdma_proxy_prepare_early_up", None)
        cancel_early = getattr(library, "rdma_proxy_cancel_early_up", None)
        # The deferred down path only binds views and starts CPU jobs before
        # gather publication. Older/eager runners can allocate or wait on CUDA.
        can_launch_early = (
            prepare_early is not None and cancel_early is not None and benchmark_metadata is not None
            and capacity == 9 and not down_options.pooled_up_gemm
            and os.environ.get("CUDA_LAUNCH_BLOCKING") != "1"
            and (down is None or (down_schedule is not None and async_scatter is not None
                                  and down.kernel_options.external_scatter_addresses
                                  and (down.intra_node_combine is None or return_maps is not None))))
        if can_launch_early:
            prepare_early.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t,
                                      ctypes.POINTER(ctypes.c_int32)]
            cancel_early.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
            prepare_early.restype = cancel_early.restype = ctypes.c_int

        prepare_local = getattr(library, "rdma_proxy_prepare_local_publication", None)
        if prepare_local is not None:
            prepare_local.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
            prepare_local.restype = ctypes.c_int

        bind_native = getattr(library, "rdma_proxy_bind_native_local_publication", None)
        query_native = getattr(library, "rdma_proxy_get_native_local_publication", None)
        use_native_publication = (can_launch_early and prepare_local is not None and num_iterations.value > 1
                                  and getattr(benchmark_metadata, "device_indices", None) is not None
                                  and bind_native is not None and query_native is not None)

        def bind_publication_storage():
            scatter, size = None, 0
            if async_scatter is not None:
                async_scatter.reserve_addresses(grouped_gemm, 2 if async_scatter.config.tile_n else 1)
                async_scatter.config.scatter_addresses = async_scatter.address_storage.data_ptr()
                scatter = ctypes.byref(async_scatter.config)
                size = async_scatter.address_storage.numel()
            require_proxy_success(library, bind_native(handle, int(kernel_stream.cuda_stream), scatter, size),
                                  "binding local publication storage")

        def publication_state(iteration):
            state = ctypes.c_int32()
            require_proxy_success(library, query_native(handle, iteration, ctypes.byref(state)),
                                  "querying native local publication")
            return state.value

        if use_native_publication:
            bind_native.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(CpuScatterConfig), ctypes.c_size_t]
            query_native.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_int32)]
            bind_native.restype = query_native.restype = ctypes.c_int
            use_native_publication = publication_state(0) == LOCAL_UNBOUND
            if use_native_publication:
                bind_publication_storage()

        def cancel_polling_up(iteration):
            require_proxy_success(library, cancel_early(handle, iteration), "draining early up GEMM")

        def submit_up():
            with torch.cuda.stream(kernel_stream):
                grouped_gemm.launch()
                if benchmark_metadata is not None and not grouped_gemm.empty_epoch:
                    return time.perf_counter_ns()
            return None

        def refresh_runner(iteration, extents, previous, *, up_bound, publication_marks=None, native_requested=False):
            down = previous
            registered_scatter = None
            wave_views = None
            reference_setup_ns = 0
            if return_maps is not None and publication_marks is None:
                # Remote contributions can require both maps even when this
                # computing rank has no expert work this epoch.
                return_maps.start(iteration)
            counts = get_expert_counts(library, handle)
            computation_tokens.value = sum(counts)
            if publication_marks is not None:
                publication_marks["expert_counts"] = time.perf_counter_ns()
                if down is not None:
                    if native_requested:
                        registered_scatter = async_scatter.reserved_views(grouped_gemm, computation_tokens.value)
                        publication_marks["scatter_views"] = time.perf_counter_ns()
                    else:
                        registered_scatter = async_scatter.register_reserved(
                            library, handle, grouped_gemm, computation_tokens.value)
                        publication_marks["scatter_register"] = time.perf_counter_ns()
                if not native_requested:
                    publication_marks["request_begin"] = time.perf_counter_ns()
                    status = prepare_local(handle, iteration)
                    publication_marks["request_end"] = time.perf_counter_ns()
                    require_proxy_success(library, status, "requesting local gather publication")
            if down_schedule is not None and computation_tokens.value:
                wave_views = down_schedule.start(iteration, counts)
                if publication_marks is not None:
                    publication_marks["down_wave_start"] = time.perf_counter_ns()
            if not up_bound:
                grouped_gemm.bind_routing(*extents)
            if down is not None:
                down = DownProjectionRunner(torch, grouped_gemm, counts, down_options,
                                            trace_configure=trace.configure if trace else None,
                                            previous=previous, defer_wave_schedule=True,
                                            external_wave_schedule=wave_views, async_scatter=async_scatter,
                                            kernel_options=previous.kernel_options, registered_scatter=registered_scatter)
                if publication_marks is not None:
                    publication_marks["down_runner"] = time.perf_counter_ns()
                if down.cpu_scatter:
                    down.configure_cpu_scatter(library, handle, already_registered=registered_scatter is not None)
            if publication_marks is not None and return_maps is not None:
                return_maps.start(iteration)
            if down is not None:
                if down.output_reset_policy == "end-of-iteration" and not down.combine:
                    down.reset_outputs()
                if return_exchange is not None:
                    return_exchange.bind_down(down)
                    if down_options.check_projections and down.route_weights is not None:
                        from projection_validation import configure_original_reference
                        reference_begin = time.perf_counter_ns()
                        configure_original_reference(grouped_gemm, down, library, handle, return_exchange,
                                                     num_experts=num_experts, iteration=iteration)
                        reference_setup_ns = time.perf_counter_ns() - reference_begin
            return down, wave_views, reference_setup_ns

        def prepare_proxy(iteration):
            require_proxy_success(library, library.rdma_proxy_prepare_iteration(handle, iteration),
                                  f"preparing proxy iteration {iteration}")

        def prepare_on_coordinator(iteration, extents, previous, setup_stream, publication_marks, native_requested):
            publication_marks["handoff"] = time.perf_counter_ns()
            # CUDA device and current stream are host-thread local.
            with torch.cuda.device(cuda_device_id), torch.cuda.stream(setup_stream):
                publication_marks["setup_context"] = time.perf_counter_ns()
                refreshed = refresh_runner(iteration, extents, previous, up_bound=True,
                                           publication_marks=publication_marks, native_requested=native_requested)
                prepare_proxy(iteration)
                return refreshed

        def run_on_coordinator(iteration, start_barrier, prepare=None, prepared=None):
            try:
                if prepare is not None:
                    prepared.set_result(prepare())
                elif start_barrier is not None:
                    start_barrier.wait()
                require_proxy_success(library, run_payload(handle, iteration),
                                      f"proxy iteration {iteration}")
            except BaseException as error:
                if start_barrier is None:
                    # A partially submitted launch can already be polling the prefix.
                    try:
                        cancel_polling_up(iteration)
                    finally:
                        if prepared is not None and not prepared.done():
                            prepared.set_exception(error)
                raise

        coordinator = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rdma-proxy-iteration-coordinator")

        print(
            "concurrent-kernel mode enabled: Python will launch one readiness-gated "
            "QuACK GroupedGEMM with each proxy iteration",
            flush=True,
        )

        for iteration in range(num_iterations.value):
            pipeline_started_ns = time.perf_counter_ns()
            reference_setup_ns = 0
            wave_views = None
            up_submitted_ns = None
            prepared = None
            publication_marks = None
            native_requested = grew_slab = False
            if dynamic and iteration:
                # All GPU and peer readers of the previous generation have retired.
                # Finish Python's optional checks before the native thread reuses A_idx.
                torch.cuda.synchronize(cuda_device_id)
                extents, elapsed = prepare_epoch(iteration)
                routing_times_ns.append(elapsed)
                if can_launch_early and extents[1]:
                    if use_native_publication:
                        state = publication_state(iteration)
                        if state == LOCAL_NEEDS_STORAGE:
                            # No publication or polling kernel can consume the undersized slab.
                            with torch.cuda.stream(kernel_stream):
                                grouped_gemm.bind_routing(*extents)
                            bind_publication_storage()
                            state = publication_state(iteration)
                            if state != LOCAL_REQUESTED:
                                raise RuntimeError("local publication did not start after scatter slab growth")
                            grew_slab = True
                        native_requested = state == LOCAL_REQUESTED
                        armed = native_requested
                    else:
                        armed_result = ctypes.c_int32()
                        require_proxy_success(library, prepare_early(handle, iteration, int(kernel_stream.cuda_stream),
                                                                      ctypes.byref(armed_result)), "ordering early up GEMM")
                        armed = bool(armed_result.value)
                    if armed:
                        early_up_iteration = iteration
                        if not grew_slab:
                            with torch.cuda.stream(kernel_stream):
                                grouped_gemm.bind_routing(*extents)
                        if down is not None and not native_requested:
                            columns = 2 if down_options.down_n_tile_major_output else 1
                            async_scatter.reserve_addresses(grouped_gemm, columns)
                        if prepare_local is not None:
                            prepared = Future()
                            publication_marks = {}
                            prepare = partial(prepare_on_coordinator, iteration, extents, down,
                                              torch.cuda.current_stream(cuda_device_id), publication_marks,
                                              native_requested)
                            publication_marks["submit"] = time.perf_counter_ns()
                            payload = coordinator.submit(run_on_coordinator, iteration, None, prepare, prepared)
                            del prepare  # The queued callable owns a reference to the previous down runner.
                        # Keep the old runner's allocations through the setup-stream join.
                        previous = down
                        try:
                            up_submitted_ns = submit_up()
                        finally:
                            if prepared is not None:
                                down, wave_views, reference_setup_ns = prepared.result()
                if prepared is None:
                    down, wave_views, reference_setup_ns = refresh_runner(
                        iteration, extents, down, up_bound=early_up_iteration is not None)
                if early_up_iteration is None:
                    torch.cuda.synchronize(cuda_device_id)

            def log_startup(message):
                if iteration == 0 and down is not None and down.combine:
                    print(f"[gpu={cuda_device_id}] iteration=0: {message}", flush=True)

            if early_up_iteration is None and down is not None and down.intra_node_combine is not None:
                down.intra_node_combine.prepare_iteration()
            log_startup("proxy preparation begin")
            start_barrier = threading.Barrier(2) if early_up_iteration is None else None
            if prepared is None:
                prepare_proxy(iteration)
                payload = coordinator.submit(run_on_coordinator, iteration, start_barrier)
            if start_barrier is not None:
                start_barrier.wait()
            log_startup("proxy released; up GEMM launch begin")

            kernel_error: BaseException | None = None
            try:
                if early_up_iteration is None:
                    up_submitted_ns = submit_up()
                log_startup("up GEMM launch returned")
                require_proxy_success(
                    library,
                    library.rdma_proxy_record_computation_end(
                        handle,
                        iteration,
                        int(kernel_stream.cuda_stream),
                    ),
                    f"recording GroupedGEMM end for iteration {iteration}",
                )
                if early_up_iteration is not None and down is not None:
                    # Join setup only for down and compute-stream teardown.
                    kernel_stream.wait_stream(torch.cuda.current_stream(cuda_device_id))
                    del previous
                with torch.cuda.stream(kernel_stream):
                    if early_up_iteration is not None and down is not None and down.intra_node_combine is not None:
                        down.intra_node_combine.prepare_iteration(stream=kernel_stream)
                    async_returns = iteration != 0 and return_maps is not None
                    if down is not None:
                        needs_metadata_wait = (down.cpu_scatter and
                                               down.cpu_scatter_publication in ("deferred", "threaded"))
                        down.launch(
                            wait_for_metadata=partial(down.wait_cpu_scatter_ready, iteration)
                            if needs_metadata_wait else None,
                            wait_for_wave_schedule=partial(down_schedule.wait, iteration, kernel_stream)
                            if wave_views is not None else None,
                            before_combine=partial(return_exchange.arm_and_bind_maps, iteration)
                            if async_returns else None)
                    if return_exchange is not None and not async_returns:
                        return_exchange.arm(iteration)
                    kernel_finished.record(kernel_stream)
            except BaseException as error:
                kernel_error = error
                if early_up_iteration is not None:
                    cancel_polling_up(iteration)
                # Report before waiting for proxy progress, which may be stalled.
                print(f"concurrent CUDA launch failed at iteration {iteration}: {error}",
                      file=sys.stderr, flush=True)

            log_startup("waiting for proxy result")
            try:
                payload.result()
            finally:
                if kernel_error is not None:
                    raise RuntimeError(
                        f"concurrent CUDA kernel failed at iteration {iteration}: {kernel_error}"
                    ) from kernel_error
            log_startup("proxy result received")
            combine = down.intra_node_combine if return_exchange is not None else None
            try:
                if complete_iteration is not None:
                    # payload.result() covers reverse-RDMA completion and visibility.
                    # Down has recorded the local combine event for this iteration.
                    combine.inter_node_combine.run(local_finished=combine.finished)
            finally:
                kernel_finished.synchronize()
                early_up_iteration = None
            log_startup("GPU iteration complete")
            if complete_iteration is not None:
                # InterNodeCombine.run joins its CUDA stream before returning.
                require_proxy_success(library, complete_iteration(handle, iteration),
                                      f"post-combine iteration_done for iteration {iteration}")
            if trace is not None:
                down_traces.append((iteration, trace.read(iteration)))
            if return_exchange is not None and return_exchange.needs_return_counts:
                return_exchange.finish_iteration(iteration)
            if combine is not None:
                combine.finish()
                # No rank may overwrite returned buffers while a peer still combines or validates.
                return_exchange.gather_completion("intra-node-done", iteration)
            # Snapshot existing completion joins before projection validation;
            # append the optional tail reset separately without hiding its cost.
            # Return/combine check flags must also be disabled for performance runs.
            pipeline_ms = (time.perf_counter_ns() - pipeline_started_ns - reference_setup_ns) / 1e6
            if dynamic and iteration == 0:
                pipeline_ms += routing_times_ns[0] / 1e6
            if down_options.check_projections:
                from projection_validation import check_projections
                check_projections(torch, grouped_gemm, down, int(computation_tokens.value))
                if return_exchange is not None:
                    return_exchange.control.gather(f"projections-checked-{iteration}")
                print(f"Projection numerical checks passed: iteration={iteration}", flush=True)
            if down is not None and down.output_reset_policy == "end-of-iteration":
                # All local/remote readers and validation barriers above must
                # precede reuse. Order any check work on the Python stream too.
                # Keep projection validation outside pipeline timing, but include
                # the tail reset and its completion even on the final iteration.
                reset_started_ns = time.perf_counter_ns()
                kernel_stream.wait_stream(torch.cuda.current_stream(cuda_device_id))
                with torch.cuda.stream(kernel_stream):
                    down.finish_iteration()
                pipeline_ms += (time.perf_counter_ns() - reset_started_ns) / 1e6
            pipeline_timings.append((iteration, pipeline_ms))
            if benchmark_metadata is not None:
                stats = dict(benchmark_metadata.last_stats, early_up_launch=start_barrier is None,
                             concurrent_local_publication=prepared is not None,
                             native_local_publication=native_requested, native_local_publication_growth=grew_slab,
                             local_publication_marks=publication_marks)
                if return_exchange is not None:
                    stats["completion_wait_ns"] = return_exchange.completion_wait_ns.copy()
                benchmark_timings.append((stats, up_submitted_ns))
            elapsed_ms = ctypes.c_float()
            require_proxy_success(
                library,
                library.rdma_proxy_get_computation_elapsed_ms(
                    handle, iteration, ctypes.byref(elapsed_ms)
                ),
                f"querying GroupedGEMM timing for iteration {iteration}",
            )
            if elapsed_ms.value <= 0.0 and computation_tokens.value:
                raise RuntimeError(
                    f"GroupedGEMM iteration {iteration} reported invalid "
                    f"elapsed time {elapsed_ms.value} ms"
                )
            useful_flops = 2 * computation_tokens.value * hidden_dimension * grouped_gemm.gemm_dimension
            effective_tflops = useful_flops / (max(elapsed_ms.value, 1e-9) * 1.0e9)
            iteration_throughputs.append(
                (iteration, computation_tokens.value, float(elapsed_ms.value), effective_tflops)
            )
            if down is not None:
                reset_ms = down.reset_start.elapsed_time(down.reset_end)
                mapping_ms = down.mapping_start.elapsed_time(down.gemm_start)
                down_ms = down.gemm_start.elapsed_time(down.end)
                # Sum measured stages; excludes host launch gaps between stages.
                tail_ms = reset_ms + mapping_ms + down_ms
                down_timings.append((iteration, down.flops, reset_ms, mapping_ms, down_ms,
                                     elapsed_ms.value + tail_ms))

        coordinator.shutdown(wait=True)
        coordinator = None
        require_proxy_success(library, library.rdma_proxy_finish(handle), "proxy finish")
        if down_options.benchmark_ready_up_gemm and computation_tokens.value:
            from benchmark_ready_up_gemm import benchmark_ready_up_gemm
            kernel_stream.wait_stream(torch.cuda.current_stream(cuda_device_id))
            with torch.cuda.stream(kernel_stream):
                benchmark_ready_up_gemm(grouped_gemm, int(computation_tokens.value))
        route_count_label = "" if dynamic else f"useful_routes={computation_tokens.value}, "
        print(
            "GroupedGEMM effective throughput by iteration "
            f"({route_count_label}K={hidden_dimension}, "
            f"projection_N={grouped_gemm.gemm_dimension}, "
            f"output_N={expert_gemm_dimension}, activation={activation or 'none'}; "
            "GEMM FLOPs only, timing includes fused activation):",
            flush=True,
        )
        for iteration, routes, elapsed_value, effective_tflops in iteration_throughputs:
            print(
                f"  iteration {iteration}: elapsed_ms={elapsed_value:.4f} "
                f"effective_throughput={effective_tflops:.2f} TFLOP/s"
                + (f" useful_routes={routes}" if dynamic else ""),
                flush=True,
            )
        print("Complete pipeline host timing (preparation through GPU, return/combine "
              "completion, reuse barriers and output resets; excludes projection checks; disable all other "
              "checks for performance):", flush=True)
        for iteration, pipeline_ms in pipeline_timings:
            print(f"  pipeline iteration {iteration}: elapsed_ms={pipeline_ms:.4f}", flush=True)
        if benchmark_metadata is not None:
            print("Benchmark routing elapsed_us covers route generation through local installation, "
                  "including the pre-exchange rank barrier. exchange_staging_us runs from aligned submit "
                  "through benchmark wait. r_ready_to_up_submit_us starts before "
                  "alignment and ends at host return from up-GEMM submission. Iteration zero also includes "
                  "compute setup in this last interval.", flush=True)
            for iteration, (stats, submitted) in enumerate(benchmark_timings):
                print_benchmark_routing(node_rank, local_gpu_index, iteration, routing_times_ns[iteration],
                                        stats, submitted)
        elif dynamic:
            print("Native routing preparation host wall time includes route generation, processing, "
                  "metadata exchange, local staging and native installation through prepare_routing return. "
                  "Python compute rebinding follows this interval.", flush=True)
            for iteration, elapsed in enumerate(routing_times_ns):
                print(f"  routing node={node_rank} gpu={local_gpu_index} iteration={iteration} "
                      f"elapsed_us={elapsed / 1000:.3f}", flush=True)
        for iteration, flops, reset_ms, mapping_ms, down_ms, combined_ms in down_timings:
            print(f"  down iteration {iteration}: reset_ms={reset_ms:.4f} "
                  f"mapping_ms={mapping_ms:.4f} "
                  f"gemm_ms={down_ms:.4f} "
                  f"effective_throughput={flops / (max(down_ms, 1e-9) * 1e9):.2f} TFLOP/s "
                  f"up_plus_down_ms={combined_ms:.4f}", flush=True)
        if down_options.trace_down_ready:
            print("CPU down-ready trace: detection times since iteration preparation "
                  "(includes up GEMM and host delays; not GPU timestamps). "
                  "Ranges are completed group prefixes; jumps share one observation.",
                  flush=True)
            width = min(down_options.down_max_swizzle_size,
                        math.ceil(hidden_dimension / down_options.down_tile_n)) * down_options.down_tile_n
            for iteration, observations in down_traces:
                previous = 0
                for ready, elapsed_us in observations:
                    print(f"  down-ready iteration {iteration}: groups=[{previous},{ready}) "
                          f"columns=[{previous * width},{min(ready * width, hidden_dimension)}) "
                          f"detected_us={elapsed_us:.3f}", flush=True)
                    previous = ready
        success = True
        return 0
    finally:
        if early_up_iteration is not None:
            cancel_polling_up(early_up_iteration)
        if down is not None and down.intra_node_combine is not None:
            down.intra_node_combine.abort()
        # Keep borrowed inputs, scatter destinations, and any mapped host flag
        # alive until the last possible kernel write, including launch failures.
        if kernel_stream is not None:
            try:
                kernel_stream.synchronize()
            except Exception as error:
                print(f"compute stream cleanup failed: {error}", file=sys.stderr, flush=True)
        if coordinator is not None:
            coordinator.shutdown(wait=True)
        shutdown_embedded_proxy(library, handle, device_buffers, initialized, success, return_exchange,
                                benchmark_metadata)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one librdma_cpu_proxy.so instance in a torchrun worker",
        allow_abbrev=False,
    )
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cuda-device-map", required=True)
    parser.add_argument("--rdma-device-map", required=True,
                        help="comma-separated GPU entries; use deviceA+deviceB for two devices per GPU")
    parser.add_argument("--listen-port-base", type=int, default=18515)
    parser.add_argument("--local-rank", "--local_rank", dest="launcher_local_rank", type=int)
    parser.add_argument(
        "--concurrent-kernel",
        action="store_true",
        help="launch readiness-gated QuACK GroupedGEMM with every proxy iteration",
    )
    parser.add_argument("--pooled-up-gemm", action=argparse.BooleanOptionalAction, default=None,
                        help="receive into one activation pool and stream a flat gather index "
                             "to the Hopper up GEMM (default: multi-buffer gather)")
    parser.add_argument("--benchmark-ready-up-gemm", action="store_true",
                        help="replay the final pooled iteration's ready work through both "
                             "up kernels and compare their outputs and timings")
    parser.add_argument(
        "--communication-only",
        action="store_true",
        help="override GEMM, return, and combine options; run only proxy dispatch/forwarding",
    )
    parser.add_argument("--num-experts", "--num_experts", type=int)
    parser.add_argument("--dynamic-routing", action=argparse.BooleanOptionalAction, default=None,
                        help="regenerate, exchange and stage routing before each iteration")
    parser.add_argument("--benchmark-metadata-exchange", action=argparse.BooleanOptionalAction, default=None,
                        help="use the CuTe benchmark metadata exchange and staging backend for dynamic routing")
    parser.add_argument("--benchmark-metadata-validation", action=argparse.BooleanOptionalAction, default=None,
                        help="scan benchmark route values during native installation (default: enabled)")
    parser.add_argument("--gemm-arch", choices=("auto", "hopper", "blackwell"), default="auto",
                        help="resolve GEMM/table geometry before native initialization; explicit values are checked against the GPU")
    parser.add_argument("--check-projections", action=argparse.BooleanOptionalAction, default=False,
                        help="check up/down numerical values each iteration, outside GEMM timing")
    parser.add_argument("--weighted-contributions", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="multiply down FP32 accumulators by fixed normalized routing weights")
    parser.add_argument(
        "--grouped-gemm-activation",
        choices=("none", *ORDINARY_ACTIVATIONS, *GATED_ACTIVATIONS),
        default="none",
        help="fused up-projection activation; requires --concurrent-kernel",
    )
    parser.add_argument("--grouped-gemm-m-tile", dest="expert_gemm_m_tile", type=int, default=128)
    parser.add_argument("--grouped-gemm-n-tile", dest="expert_gemm_n_tile", type=int, default=256)
    parser.add_argument(
        "--grouped-gemm-k-tile", dest="expert_gemm_k_tile", type=int, default=None
    )
    parser.add_argument(
        "--grouped-gemm-output-dimension",
        dest="expert_gemm_dimension",
        type=int,
        default=8192,
        help="post-activation width; gated activations use twice this GEMM width",
    )
    parser.add_argument(
        "--grouped-gemm-cluster-m", dest="expert_gemm_cluster_m", type=int, default=1
    )
    parser.add_argument(
        "--up-sm-count", "--grouped-gemm-sm-count", dest="up_sm_count", type=int, default=None,
        help="cap up GEMM persistent CTAs; positive multiple of --grouped-gemm-cluster-m. "
             "Defaults to hardware capacity; hardware caps larger requests; does not reserve physical SMs",
    )
    parser.add_argument(
        "--grouped-gemm-max-swizzle-size",
        dest="expert_gemm_max_swizzle_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--quack-root",
        type=Path,
        default=Path(os.environ.get("QUACK_ROOT", Path(__file__).resolve().parents[2] / "quack_for_communication_overlap")),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the derived C++ argv without loading the shared library",
    )
    add_down_arguments(parser)
    args, proxy_args = parser.parse_known_args()
    if any(arg.startswith(("--down-", "--no-down-")) for arg in proxy_args):
        parser.error("unsupported --down-* option")
    if args.communication_only:
        # Apply the mode after parsing so it wins regardless of argument order.
        # Reset every option owned by the down/return/combine pipeline together.
        down_parser = argparse.ArgumentParser(add_help=False)
        add_down_arguments(down_parser)
        vars(args).update(vars(down_parser.parse_args([])))
        args.concurrent_kernel = False
        args.pooled_up_gemm = False
        args.benchmark_ready_up_gemm = False
        args.check_projections = False
        args.weighted_contributions = False
        args.grouped_gemm_activation = "none"
        args.up_sm_count = None
        for name in vars(args):
            if name.startswith("expert_gemm_"):
                setattr(args, name, parser.get_default(name))
    try:
        if args.pooled_up_gemm and not args.concurrent_kernel:
            raise ValueError("--pooled-up-gemm requires --concurrent-kernel")
        if args.pooled_up_gemm and args.gemm_arch == "blackwell":
            raise ValueError("--pooled-up-gemm requires Hopper SM90")
        if args.benchmark_ready_up_gemm and not args.pooled_up_gemm:
            raise ValueError("--benchmark-ready-up-gemm requires --pooled-up-gemm")
        if args.up_sm_count is not None and not args.concurrent_kernel:
            raise ValueError("--up-sm-count requires --concurrent-kernel")
        validate_up_sm_count(args.up_sm_count, args.expert_gemm_cluster_m)
        validate_down_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if not args.communication_only and not args.down_projection2 and any(
        arg.startswith(("--down-", "--no-down-")) for arg in sys.argv[1:]
    ):
        parser.error("--down-* options require --down-projection2")
    if args.grouped_gemm_activation != "none" and not args.concurrent_kernel:
        parser.error("--grouped-gemm-activation requires --concurrent-kernel")
    if args.check_projections and not args.concurrent_kernel:
        parser.error("--check-projections requires --concurrent-kernel")
    if args.weighted_contributions and not (args.down_projection2 and args.down_gather_table):
        parser.error("--weighted-contributions requires --down-projection2 and --down-gather-table")
    if args.expert_gemm_dimension <= 0:
        parser.error("--grouped-gemm-output-dimension must be positive")
    args.grouped_gemm_activation = (
        None if args.grouped_gemm_activation == "none" else args.grouped_gemm_activation
    )

    proxy_argv, metadata = build_proxy_argv(args, proxy_args)
    if args.communication_only:
        # Last CLI values override both the launch script and JSON settings.
        proxy_argv.extend([
            "--nvlink_return_completion_notifications_enabled=false",
            "--nvlink_return_local_batch_sync_enabled=false",
            "--nvlink_return_ready_gpu_publication_enabled=false",
        ])
        print("communication-only mode: GEMM, result returns, and combine disabled", flush=True)
    print(
        "starting embedded RDMA CPU proxy "
        f"rank={metadata['rank']} node_rank={metadata['node_rank']} "
        f"local_gpu={metadata['local_rank']} cuda_device={metadata['cuda_device_id']} "
        f"rdma_device={metadata['rdma_device_name']} "
        f"listen_port={metadata['listen_port']}",
        flush=True,
    )
    if args.dry_run:
        if args.concurrent_kernel:
            print("Up GEMM SM budget: "
                  f"requested={args.up_sm_count if args.up_sm_count is not None else 'hardware-default'} "
                  f"cluster_m={args.expert_gemm_cluster_m}; QuACK hardware capacity applies", flush=True)
        if args.concurrent_kernel and args.gemm_arch == "auto":
            print("GEMM architecture: auto (resolved from CUDA before native initialization; "
                  "use --gemm-arch=blackwell or hopper to inspect explicit scheduler geometry)", flush=True)
        print(shlex.join(proxy_argv), flush=True)
        return 0

    # Load PyTorch's runtime first so a matching native SONAME reuses it. Check
    # immediately after dlopen, before native initialization can contact peers.
    torch = import_torch_for_proxy(int(metadata["cuda_device_id"]))
    library = load_proxy_library(args.library.resolve(), torch_cuda_version=torch.version.cuda)
    if args.concurrent_kernel:
        if args.num_experts is None:
            raise RuntimeError("--concurrent-kernel requires --num-experts")
        return run_proxy_with_concurrent_kernel(
            library,
            proxy_argv,
            int(metadata["cuda_device_id"]),
            node_rank=int(metadata["node_rank"]),
            local_gpu_index=int(metadata["local_rank"]),
            world_size=int(metadata["world_size"]),
            num_experts=args.num_experts,
            expert_gemm_dimension=args.expert_gemm_dimension,
            activation=args.grouped_gemm_activation,
            expert_gemm_m_tile=args.expert_gemm_m_tile,
            expert_gemm_n_tile=args.expert_gemm_n_tile,
            expert_gemm_k_tile=args.expert_gemm_k_tile,
            expert_gemm_cluster_m=args.expert_gemm_cluster_m,
            expert_gemm_max_swizzle_size=args.expert_gemm_max_swizzle_size,
            quack_root=args.quack_root,
            down_options=args,
            up_sm_count=args.up_sm_count,
        )
    return run_proxy(
        library,
        proxy_argv,
        int(metadata["cuda_device_id"]),
        node_rank=int(metadata["node_rank"]),
        local_gpu_index=int(metadata["local_rank"]),
    )


def worker_entrypoint() -> int:
    """Preserve a readable local diagnostic before TorchElastic records it."""
    try:
        return main()
    except Exception as error:
        print(f"rdma_proxy_worker failed: {error}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    try:
        from torch.distributed.elastic.multiprocessing.errors import record
    except ImportError:
        # A direct dry-run can still explain a missing PyTorch installation.
        recorded_worker_entrypoint = worker_entrypoint
    else:
        recorded_worker_entrypoint = record(worker_entrypoint)
    raise SystemExit(recorded_worker_entrypoint())
