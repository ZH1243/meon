"""Post-RDMA node reduction, writing directly in original source-token order."""
import ctypes


def inverse_token_rows(token_lists, num_tokens):
    """Validate x3 domains and invert them independently; absent rows stay -1."""
    if num_tokens < 0:
        raise ValueError("num_tokens must be nonnegative")
    result = []
    for tokens in token_lists:
        inverse = [-1] * num_tokens
        for row, token in enumerate(tokens):
            if token < 0 or token >= num_tokens or inverse[token] != -1:
                raise ValueError("x3 contains an out-of-range or duplicate token")
            inverse[token] = row
        result.append(inverse)
    return result


class TokenOrderCombine:
    """All inputs must be complete before run(); run joins before buffer reuse.

    Inputs are in ascending node order. Fixed routes omit empty domains from the
    compiled tuple. Dynamic routes retain all domains so they can grow later.
    """

    def __init__(self, inputs, token_lists, num_tokens, check=False, ctas=None, capacity_buffers=None,
                 *, prepared_rows=None):
        import torch
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from intra_node_combine_kernel import RegisterInterNodeKernel

        if not inputs or (prepared_rows is None and (token_lists is None or len(inputs) != len(token_lists))):
            raise ValueError("inter-node combine requires one x3 list per input")
        if num_tokens < 0:
            raise ValueError("num_tokens must be nonnegative")
        if ctas is not None and ctas <= 0:
            raise ValueError("inter-node combine CTA count must be positive")
        first = inputs[0]
        if first.ndim != 3 or first.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("inter-node combine requires tiled BF16/FP16 inputs")
        if prepared_rows is not None and (
                capacity_buffers is None or len(capacity_buffers) != len(inputs)
                or prepared_rows.shape != (len(inputs), num_tokens + 2)
                or prepared_rows.dtype != torch.int32 or prepared_rows.device != first.device
                or not prepared_rows.is_contiguous()):
            raise ValueError("invalid dynamic inter-node combine row table or capacity buffers")
        tiles, _, tile = first.shape
        width = tiles * tile
        if not width or tile % 8 or num_tokens * width > 2**31 - 1:
            raise ValueError("invalid inter-node width/alignment or int32 index overflow")
        # The native builder has already checked its live counts against the input views.
        counts = ([tensor.shape[1] for tensor in inputs] if prepared_rows is not None
                  else [len(tokens) for tokens in token_lists])
        for tensor, count in zip(inputs, counts):
            if (tensor.shape != (tiles, count, tile) or not tensor.is_contiguous()
                    or tensor.dtype != first.dtype or tensor.device != first.device
                    or tensor.numel() > 2**31 - 1):
                raise ValueError("inter-node input layout/rows/dtype/device mismatch")
        self.torch, self.cuda = torch, cuda
        self.inputs, self.token_lists = tuple(inputs), token_lists
        self.num_tokens, self.width = num_tokens, width
        self.check_enabled = check
        self.output = torch.empty((num_tokens, width), dtype=first.dtype, device=first.device)
        self.stream = torch.cuda.Stream(device=first.device)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.compiled = None
        self.num_ctas = (ctas if ctas is not None else
                         torch.cuda.get_device_properties(first.device).multi_processor_count)
        active = (list(range(len(inputs))) if capacity_buffers is not None else
                  [i for i, count in enumerate(counts) if count])
        if prepared_rows is not None:
            self.rows = prepared_rows
        else:
            inverse = inverse_token_rows(token_lists, num_tokens)
            if capacity_buffers is not None:
                # The kernel reads each live packed tile stride after the inverse row list.
                inverse = [row + [count, 0] for row, count in zip(inverse, counts)]
            self.rows = torch.tensor([inverse[i] for i in active], dtype=torch.int32,
                                     device=first.device) if active and num_tokens else torch.full(
                                         (max(1, len(active)), max(1, num_tokens)), -1,
                                         dtype=torch.int32, device=first.device)
        self.buffers = tuple(inputs[i].view(-1) for i in active) if capacity_buffers is None else tuple(capacity_buffers)
        if num_tokens:
            self.arguments = (
                tuple(from_dlpack(b, assumed_align=16) for b in self.buffers) if active else None,
                from_dlpack(self.rows), from_dlpack(self.output, assumed_align=16), None,
            )
            vec = 32
            while tile % vec:
                vec //= 2
            lanes = 32
            while (tile // vec) % lanes:
                lanes //= 2
            kernel = RegisterInterNodeKernel(
                tuple(num_tokens if capacity_buffers is not None else counts[i] for i in active), width, tile,
                self.num_ctas,
                vec, 256, False, 0)
            kernel.source_tokens = num_tokens
            kernel.dynamic_strides = capacity_buffers is not None
            kernel.lanes_per_group = lanes
            kernel.prefetch_peers = min(4, max(1, len(active)))
            self.compiled = cute.compile(kernel, *self.arguments,
                                         cuda.CUstream(self.stream.cuda_stream))
        # Setup uploads must precede the first launch on the private stream.
        torch.cuda.synchronize(first.device)

    def run(self):
        """Caller has joined native RDMA and local combine, including visibility."""
        self.start.record(self.stream)
        try:
            if self.compiled is not None:
                self.compiled(*self.arguments, self.cuda.CUstream(self.stream.cuda_stream))
            self.end.record(self.stream)
        finally:
            # Also join a partially successful launch before exceptional teardown.
            self.stream.synchronize()
        self.elapsed_ms = self.start.elapsed_time(self.end)
        if self.check_enabled:
            self.check()

    def bind_routes(self, inputs, token_lists):
        """Replace live domains while retaining the registered storage and compiled launch."""
        inverse = inverse_token_rows(token_lists, self.num_tokens)
        self.inputs, self.token_lists = tuple(inputs), token_lists
        self.rows.copy_(self.torch.tensor([row + [len(tokens), 0] for row, tokens in zip(inverse, token_lists)],
                                          dtype=self.rows.dtype, device=self.rows.device))

    def check(self):
        """Independent scatter/add reference, bounded to 4096 output rows."""
        torch = self.torch
        for start in range(0, self.num_tokens, 4096):
            end = min(start + 4096, self.num_tokens)
            expected = torch.zeros((end - start, self.width), dtype=torch.float32,
                                   device=self.output.device)
            for tensor, tokens in zip(self.inputs, self.token_lists):
                selected = [(row, token - start) for row, token in enumerate(tokens)
                            if start <= token < end]
                if selected:
                    rows, destinations = zip(*selected)
                    rows = torch.tensor(rows, dtype=torch.int64, device=tensor.device)
                    destinations = torch.tensor(destinations, dtype=torch.int64, device=tensor.device)
                    values = tensor[:, rows, :].permute(1, 0, 2).reshape(len(selected), self.width)
                    expected.index_add_(0, destinations, values.float())
            torch.testing.assert_close(self.output[start:end], expected.to(self.output.dtype),
                                       rtol=0, atol=0)
        print("Inter-node combine reference check: PASSED", flush=True)


class InterNodeCombine(TokenOrderCombine):
    def __init__(self, combine, options):
        inputs, token_lists, num_tokens, rows = self._init_route_data(combine)
        capacity_buffers = None
        if combine.dynamic_capacity is not None:
            capacity_buffers = tuple(
                combine.storage_for_node(node) if node == combine.exchange.node
                else combine.rdma_return.receive_storage[node] for node in combine.nodes)
        super().__init__(inputs, token_lists, num_tokens, options.check_inter_node_combine,
                         ctas=options.inter_node_combine_ctas, capacity_buffers=capacity_buffers,
                         prepared_rows=rows)
        row_builder = "native" if self._build_rows is not None else "python"
        print(f"Inter-node combine: CTAs={self.num_ctas}, output={tuple(self.output.shape)} row-major, "
              "original token order; runs after local combine and any RDMA returns, "
              f"row_builder={row_builder}", flush=True)

    @staticmethod
    def _live_inputs(combine):
        return tuple(combine.outputs[node] if node == combine.exchange.node else combine.rdma_return.received[node]
                     for node in combine.nodes)

    def _init_route_data(self, combine):
        self._exchange = combine.exchange
        self._nodes = (ctypes.c_int32 * len(combine.nodes))(*combine.nodes)
        inputs = self._live_inputs(combine)
        num_tokens = combine.dynamic_capacity
        self._build_rows = (getattr(self._exchange.library, "rdma_proxy_build_inter_node_combine_rows", None)
                            if num_tokens is not None else None)
        if self._build_rows is None:
            token_lists, num_tokens = self._read_token_lists()
            return inputs, token_lists, num_tokens, None

        torch = self._exchange.torch
        device = inputs[0].device
        self._build_rows.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64,
                                     ctypes.c_uint64, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64]
        self._build_rows.restype = ctypes.c_int
        maps = getattr(self._exchange, "return_maps", None)
        self.rows_host = torch.empty((len(self._nodes), num_tokens + 2), dtype=torch.int32,
                                     device="cpu", pin_memory=maps is None and device.type == "cuda")
        rows = maps.inter_rows if maps is not None else torch.empty_like(self.rows_host, device=device)
        self._upload_native_rows(inputs, rows, num_tokens)
        if maps is not None:
            self.rows_host = None
        return inputs, None, num_tokens, rows

    def _upload_native_rows(self, inputs, rows, num_tokens):
        host = self.rows_host
        self._exchange.require(self._build_rows(
            self._exchange.handle, self._nodes, len(self._nodes), num_tokens,
            ctypes.cast(host.data_ptr(), ctypes.POINTER(ctypes.c_int32)), host.numel()))
        if host[:, -2].tolist() != [tensor.shape[1] for tensor in inputs]:
            raise ValueError("outgoing x3 lengths disagree with inter-node input views")
        # Complete the DMA read before a subsequent generation can reuse host storage.
        rows.copy_(host)

    def _read_token_lists(self):
        exchange = self._exchange
        query = exchange.library.rdma_proxy_get_outgoing_token_indices
        query.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64),
                          ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64),
                          ctypes.POINTER(ctypes.c_uint64)]
        query.restype = ctypes.c_int
        token_lists = []
        num_tokens = None
        for node in self._nodes:
            count, total = ctypes.c_uint64(), ctypes.c_uint64()
            exchange.require(query(exchange.handle, node, None, 0,
                                   ctypes.byref(count), ctypes.byref(total)))
            indices = (ctypes.c_uint64 * count.value)()
            exchange.require(query(exchange.handle, node, indices, count.value,
                                   ctypes.byref(count), ctypes.byref(total)))
            if num_tokens is not None and num_tokens != total.value:
                raise ValueError("inconsistent original token count")
            num_tokens = total.value
            token_lists.append(list(indices))
        return token_lists, num_tokens

    def refresh_routes(self, combine):
        inputs = self._live_inputs(combine)
        if self._build_rows is not None:
            self._upload_native_rows(inputs, self.rows, self.num_tokens)
            self.inputs, self.token_lists = inputs, None
            return
        token_lists, num_tokens = self._read_token_lists()
        if num_tokens != self.num_tokens:
            raise ValueError("original token capacity changed during dynamic routing")
        self.bind_routes(inputs, token_lists)

    def bind_return_views(self, combine, outgoing_counts):
        inputs = self._live_inputs(combine)
        if [t.shape[1] for t in inputs] != list(outgoing_counts):
            raise ValueError("outgoing x3 lengths disagree with inter-node input views")
        self.inputs, self.token_lists = inputs, None

    def check(self):
        # Fetch original x3 independently for the numerical reference after the kernel finishes.
        if self.token_lists is None:
            token_lists, num_tokens = self._read_token_lists()
            if num_tokens != self.num_tokens:
                raise ValueError("original token capacity changed during inter-node check")
            if [len(tokens) for tokens in token_lists] != [tensor.shape[1] for tensor in self.inputs]:
                raise ValueError("outgoing x3 lengths disagree with inter-node input views")
            self.token_lists = token_lists
        super().check()
