"""Source-side intra-node reduction, concurrent with readiness-publishing down GEMM."""
import ctypes


def validate_intra_node_sm_budget(total_sms, down_sm_count, combine_ctas, down_cluster_m,
                                  *, relax_guard=False):
    """Check launch headroom; this does not reserve or pin physical SMs."""
    # The default budgets a whole down cluster per combine CTA to allow for
    # fragmentation. Opting in budgets one SM per CTA, retaining a total cap.
    available = total_sms - combine_ctas * (1 if relax_guard else down_cluster_m)
    if down_sm_count is None or down_sm_count > available:
        policy = "relaxed" if relax_guard else "cluster"
        raise ValueError(f"intra-node combine requires --down-sm-count <= {available} "
                         "to leave room for persistent combine CTAs "
                         f"(SM guard: {policy}; total SMs: {total_sms})")


def contribution_indices(masks, world):
    """Output row k of peer p corresponds to the k-th selected full-input row."""
    if not 2 <= world <= 8 or any(mask <= 0 or mask >= (1 << world) for mask in masks):
        raise ValueError("invalid intra-node routing masks")
    return [[row for row, mask in enumerate(masks) if mask & (1 << peer)]
            for peer in range(world)]


class IntraNodeCombine:
    def __init__(self, exchange, options, *, dynamic_capacity=None):
        import torch
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack
        from intra_node_combine_kernel import RegisterIntraNodeKernel

        self.exchange = exchange
        self.torch, self.cuda = torch, cuda
        self.device = exchange.down.up.output.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.control_stream = torch.cuda.Stream(device=self.device)
        self.mapping_ready = torch.cuda.Event()
        self.finished = torch.cuda.Event()
        self.launched = False
        self.inter_node_combine = None
        self.check_enabled = options.check_intra_node_combine
        self.cancel = torch.zeros(1, dtype=torch.int32, device=self.device)
        width = int(exchange.down.up.x_buffers[0].shape[1])
        tile = options.down_tile_n
        vec = options.intra_node_combine_elements_per_thread
        threads = options.intra_node_combine_threads_per_cta
        ctas = options.intra_node_combine_ctas
        if width % tile or tile % vec:
            raise ValueError("intra-node combine requires tile_n dividing width and elements/thread dividing tile_n")
        properties = torch.cuda.get_device_properties(self.device)
        relax_sm_guard = getattr(options, "intra_node_combine_relax_sm_guard", False)
        validate_intra_node_sm_budget(
            properties.multi_processor_count, options.down_sm_count, ctas,
            options.down_cluster_m, relax_guard=relax_sm_guard,
        )
        query = exchange.library.rdma_proxy_get_intra_node_source_masks
        query.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_uint8),
                          ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        query.restype = ctypes.c_int
        publication = exchange.library.rdma_proxy_get_return_ready_publication
        publication.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                               ctypes.POINTER(ctypes.c_size_t)]
        publication.restype = ctypes.c_int
        host, device = ctypes.c_size_t(), ctypes.c_size_t()
        exchange.require(publication(exchange.handle, ctypes.byref(host), ctypes.byref(device)))
        self.group_tiles = min(options.down_max_swizzle_size, width // tile)
        self.num_groups = (width // tile + self.group_tiles - 1) // self.group_tiles
        self.ready_groups = (torch.zeros(1, dtype=torch.int32, device=self.device)
                             if options.intra_node_combine_publish_ready else None)
        self.completion_slots = (torch.zeros(self.num_groups * ctas, dtype=torch.int32,
                                            device=self.device)
                                 if self.ready_groups is not None else None)
        self.addresses = torch.tensor([device.value, self.cancel.data_ptr(),
                                      self.completion_slots.data_ptr() if self.completion_slots is not None else 0,
                                      self.ready_groups.data_ptr() if self.ready_groups is not None else 0],
                                     dtype=torch.int64, device=self.device)
        self.nodes = sorted({node for node, _ in exchange.down.up.source_pairs})
        self.dynamic_capacity = dynamic_capacity
        self.width, self.tile = width, tile
        self.output_tile_n = tile if options.intra_node_combine_n_tile_major_output else 0
        active = self._init_routes()
        self.buffers = tuple(buffer.view(-1) if dynamic_capacity is not None
                             else buffer.view(-1)[:index.numel() * width]
                             for _, _, _, index, buffer in active)
        self.compiled = None
        self.kernel_output = self.output if self.capacity_rows else torch.empty(
            1, dtype=self.output.dtype, device=self.device)
        self.rdma_return = None
        if options.rdma_return_token_results:
            from rdma_network_return import NetworkReturn
            self.rdma_return = NetworkReturn(self, options)
            self.addresses = torch.cat((self.addresses, torch.tensor(
                [self.rdma_return.device_prefix], dtype=torch.int64, device=self.device)))
        if self.capacity_rows or self.ready_groups is not None:
            self.arguments = (
                tuple(from_dlpack(buffer, assumed_align=16) for buffer in self.buffers)
                if self.buffers else None,
                from_dlpack(self.rows), from_dlpack(self.kernel_output, assumed_align=16),
                from_dlpack(self.addresses),
            )
            kernel = RegisterIntraNodeKernel(
                tuple(dynamic_capacity if dynamic_capacity is not None else entry[3].numel()
                      for entry in active), width, tile, ctas, vec,
                threads, True, self.output_tile_n)
            kernel.dynamic_strides = dynamic_capacity is not None
            kernel.dynamic_domain_capacity = dynamic_capacity
            kernel.domain_peers = exchange.world
            kernel.output_domains = tuple(
                (offset, dynamic_capacity if dynamic_capacity is not None else count)
                for _, offset, count in self.domains)
            first_peer = 0
            input_domains = []
            for node, offset, count in self.domains:
                peers = sum(entry[0] == node for entry in active)
                input_domains.append((offset, dynamic_capacity if dynamic_capacity is not None else count,
                                      first_peer, peers))
                first_peer += peers
            kernel.input_domains = tuple(input_domains)
            kernel.group_tiles = self.group_tiles
            kernel.publish_ready = self.ready_groups is not None
            kernel.publish_rdma_ready = self.rdma_return is not None
            kernel.source_tokens = self.capacity_rows
            kernel.prefetch_peers = options.intra_node_combine_prefetch_peers
            lanes = 32
            while (tile // vec) % lanes:
                lanes //= 2
            kernel.lanes_per_group = lanes
            self.compiled = cute.compile(kernel, *self.arguments,
                                         cuda.CUstream(self.stream.cuda_stream))
        # Compilation/metadata uploads finish before any readiness-gated launch.
        torch.cuda.synchronize(self.device)
        self._warmup()
        print(f"Intra-node combine: register-prefetch, rows_by_source="
              f"{[(n, c) for n, _, c in self.domains]} CTAs={ctas}, "
              f"outputs={'N-tile-major' if self.output_tile_n else 'row-major'} "
              "in staging/RDMA input-row order; "
              f"SM guard={'relaxed' if relax_sm_guard else 'cluster'}, "
              f"down CTA budget={options.down_sm_count}", flush=True)

    def _init_routes(self):
        exchange, torch, capacity = self.exchange, self.torch, self.dynamic_capacity
        self._build_rows = (getattr(exchange.library, "rdma_proxy_build_intra_node_combine_rows", None)
                            if capacity is not None else None)
        if self._build_rows is None:
            self.domains, self.entries = self._read_mask_routes()
        else:
            # Dynamic kernels bind every peer buffer once, including empty peers.
            self.entries = [(node, peer, i * capacity, None, self._source_buffer(node, peer))
                            for i, node in enumerate(self.nodes) for peer in range(exchange.world)]
        self.capacity_rows = (len(self.nodes) * capacity if capacity is not None else
                              sum(count for _, _, count in self.domains))
        if self.capacity_rows * self.width > 2**31 - 1:
            raise ValueError("intra-node combine exceeds the kernel's int32 element-index range")
        active = self.entries if capacity is not None else [entry for entry in self.entries if entry[3].numel()]
        shape = (max(1, len(active)), max(1, self.capacity_rows) + (2 if capacity is not None else 0))
        self.rows = torch.empty(shape, dtype=torch.int32, device=self.device)
        self.output = torch.empty((self.capacity_rows * self.width,) if self.output_tile_n
                                  else (self.capacity_rows, self.width),
                                  dtype=exchange.outputs[0].dtype, device=self.device)
        if self._build_rows is None:
            self._update_views(active)
        else:
            self._build_rows.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64,
                                         ctypes.c_uint64, ctypes.POINTER(ctypes.c_int32), ctypes.c_uint64,
                                         ctypes.POINTER(ctypes.c_uint64)]
            self._build_rows.restype = ctypes.c_int
            self._source_nodes = (ctypes.c_int32 * len(self.nodes))(*self.nodes)
            self._domain_counts = (ctypes.c_uint64 * len(self.nodes))()
            self.rows_host = torch.empty(shape, dtype=torch.int32, device="cpu", pin_memory=self.device.type == "cuda")
            self._update_native_rows()
        return active

    def _update_native_rows(self):
        self.exchange.require(self._build_rows(
            self.exchange.handle, self._source_nodes, len(self.nodes), self.dynamic_capacity,
            ctypes.cast(self.rows_host.data_ptr(), ctypes.POINTER(ctypes.c_int32)),
            self.rows_host.numel(), self._domain_counts))
        # The blocking copy retires the DMA read before this buffer can be refilled.
        self.rows.copy_(self.rows_host)
        self.domains = [(node, i * self.dynamic_capacity, self._domain_counts[i])
                        for i, node in enumerate(self.nodes)]
        self._update_output_views()

    def _source_buffer(self, node, peer):
        exchange = self.exchange
        if peer == exchange.rank:
            return exchange.outputs[exchange.down.up.source_pairs.index((node, peer))]
        return exchange.received[peer, node]

    def _read_mask_routes(self):
        exchange, torch = self.exchange, self.torch
        query = exchange.library.rdma_proxy_get_intra_node_source_masks
        domains, entries = [], []
        total_rows = 0
        for i, node in enumerate(self.nodes):
            count = ctypes.c_uint64()
            exchange.require(query(exchange.handle, node, None, 0, ctypes.byref(count)))
            masks = (ctypes.c_uint8 * count.value)()
            exchange.require(query(exchange.handle, node, masks, count.value, ctypes.byref(count)))
            if self.dynamic_capacity is not None and count.value > self.dynamic_capacity:
                raise ValueError("intra-node route domain exceeds its source-token capacity")
            offset = total_rows if self.dynamic_capacity is None else i * self.dynamic_capacity
            domains.append((node, offset, count.value))
            for peer, selected in enumerate(contribution_indices(list(masks), exchange.world)):
                indices = torch.tensor(selected, dtype=torch.int64, device=self.device)
                entries.append((node, peer, offset, indices, self._source_buffer(node, peer)))
            total_rows += count.value
        return domains, entries

    def _update_views(self, entries):
        torch = self.torch
        self.rows.fill_(-1)
        counts = {node: count for node, _, count in self.domains}
        for peer, (node, _, offset, index, _) in enumerate(entries):
            self.rows[peer, index + offset] = torch.arange(index.numel(), dtype=torch.int32,
                                                           device=self.device)
            if self.dynamic_capacity is not None:
                self.rows[peer, -2] = index.numel()
                self.rows[peer, -1] = counts[node]
        self._update_output_views()

    def _update_output_views(self):
        self.outputs = {
            node: (self.output.view(-1)[offset * self.width:(offset + count) * self.width]
                   .view(self.width // self.tile, count, self.tile)
                   if self.output_tile_n else self.output[offset:offset + count])
            for node, offset, count in self.domains
        }

    def storage_for_node(self, node):
        """Fixed base used for RDMA registration and full-capacity kernel views."""
        offset = self.nodes.index(node) * self.dynamic_capacity * self.width
        return self.output.view(-1).narrow(0, offset, self.dynamic_capacity * self.width)

    def refresh_routes(self):
        if self._build_rows is None:
            self.domains, self.entries = self._read_mask_routes()
            self._update_views(self.entries)
        else:
            self._update_native_rows()
        if self.rdma_return is not None:
            self.rdma_return.refresh(self)
        if self.inter_node_combine is not None:
            self.inter_node_combine.refresh_routes(self)

    def _warmup(self):
        """Exercise the compiled callable before readiness-gated GPU work starts.

        Compilation alone does not exercise first-invocation runtime setup.
        Cancellation lets the actual kernel run without waiting for returns or
        reading their uninitialized payloads, and suppresses ready publication.
        """
        if self.compiled is None:
            return
        print(f"[gpu={self.exchange.down.up.cuda_device_id}] intra-node combine warmup begin",
              flush=True)
        with self.torch.cuda.stream(self.stream):
            self.cancel.fill_(1)
            self.compiled(*self.arguments, self.cuda.CUstream(self.stream.cuda_stream))
            self.finished.record(self.stream)
        # Do not clear cancellation until every warmup CTA has exited.
        self.stream.synchronize()
        with self.torch.cuda.stream(self.stream):
            self.cancel.zero_()
            if self.ready_groups is not None:
                self.ready_groups.zero_()
                self.completion_slots.zero_()
        self.stream.synchronize()
        print(f"[gpu={self.exchange.down.up.cuda_device_id}] intra-node combine warmup complete",
              flush=True)

    def prepare_iteration(self):
        """Reset publication before allowing any work for the next iteration.

        Caller has joined combine and all consumers from the previous iteration.
        Returning guarantees zero is visible, including to other CUDA streams.
        """
        if self.launched:
            raise RuntimeError("cannot reset intra-node readiness while combine is active")
        if self.ready_groups is not None:
            with self.torch.cuda.stream(self.control_stream):
                self.ready_groups.zero_()
                self.completion_slots.zero_()
            self.control_stream.synchronize()

    def launch(self):
        # Called on the down stream after mapping/reset and down launch. This
        # event was recorded BEFORE down launch, so it does not wait for down.
        trace = self.exchange.down._log_combine_startup
        trace("intra-node combine stream wait-event begin")
        self.stream.wait_event(self.mapping_ready)
        trace("intra-node combine stream wait-event returned")
        self.launched = True
        if self.compiled is not None:
            trace("intra-node combine compiled invocation begin")
            self.compiled(*self.arguments, self.cuda.CUstream(self.stream.cuda_stream))
            trace("intra-node combine compiled invocation returned")
        trace("intra-node combine finished-event record begin")
        self.finished.record(self.stream)
        trace("intra-node combine finished-event record returned")

    def finish(self):
        if self.launched:
            self.finished.synchronize()
            self.launched = False
        if self.rdma_return is not None:
            self.rdma_return.check()
        if self.check_enabled:
            if self.ready_groups is not None and self.ready_groups.item() != self.num_groups:
                raise RuntimeError("intra-node output-ready prefix did not reach the final group")
            self.check()

    def check(self):
        """Independent FP32 gather/add reference after all returns are complete."""
        torch = self.torch
        entries = self.entries
        if self._build_rows is not None:
            domains, entries = self._read_mask_routes()
            if domains != self.domains:
                raise RuntimeError("native intra-node domain lengths disagree with the routing masks")
        expected = torch.zeros((self.capacity_rows, self.width), dtype=torch.float32, device=self.device)
        for node, peer, offset, index, buffer in entries:
            count = index.numel()
            if peer == self.exchange.rank:
                source = self.exchange.down.up.source_pairs.index((node, peer))
                actual_count = int(self.exchange.down.combine_valid_rows[source].item())
                actual_index = self.exchange.down.combine_output_to_input[source, :count]
                torch.testing.assert_close(actual_index, index, rtol=0, atol=0)
            else:
                actual_count = self.exchange.received_valid_rows[peer, node]
            if count != actual_count:
                raise RuntimeError("routing masks disagree with combine valid-row count")
            if count:
                values = (buffer.view(-1)[:count * self.width]
                          .view(self.width // self.tile, count, self.tile)
                          .permute(1, 0, 2).reshape(count, self.width))
                expected[index + offset] += values.float()
        for node, offset, count in self.domains:
            actual = self.outputs[node]
            if self.output_tile_n:
                actual = actual.permute(1, 0, 2).reshape(count, self.width)
            torch.testing.assert_close(actual, expected[offset:offset + count].to(self.output.dtype),
                                       rtol=0, atol=0)
        print("Intra-node combine reference check: PASSED", flush=True)

    def abort(self):
        # Unblock polling if a CPU proxy fails before publishing the final group.
        if self.launched:
            with self.torch.cuda.stream(self.control_stream):
                self.cancel.fill_(1)
            self.control_stream.synchronize()
            self.stream.synchronize()
            self.launched = False
