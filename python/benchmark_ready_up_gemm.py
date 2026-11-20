"""Replay the last completed pooled up GEMM using both QuACK gather formats."""

import statistics


def check_ready_outputs(up, routes):
    """Check both warmup launches against the saved, valid live output."""
    if up.ready_rows.item() != up.work_table.shape[0]:
        raise RuntimeError("ready-work comparison requires a fully published gather table")
    if not 0 < routes <= up.output.shape[0]:
        raise ValueError("invalid packed row count for ready-work comparison")

    valid = up.output[:routes]
    live = valid.clone()
    for pooled in (False, True):
        # Reuse D and its strides. Poison first so skipped tiles cannot pass
        # by retaining the preceding launch's result. Ignore capacity padding.
        valid.fill_(float("nan"))
        up.launch(pooled=pooled)
        if not up.torch.equal(valid, live):
            raise AssertionError(f"ready-work replay differs from live packed output (pooled={pooled})")


def benchmark_ready_up_gemm(up, routes):
    """Time interleaved CUDA graph launches after all final-iteration consumers join."""
    torch = up.torch
    with torch.cuda.device(up.cuda_device_id), torch.inference_mode():
        check_ready_outputs(up, routes)

        # ABBA balances first position and cache carryover across replays. Each
        # kernel gets its own external timing events inside the graph so host
        # enqueue gaps and numerical checks do not enter the measurement.
        sequence = (False, True, True, False)
        events = [(torch.cuda.Event(enable_timing=True, external=True),
                   torch.cuda.Event(enable_timing=True, external=True)) for _ in sequence]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for pooled, (begin, end) in zip(sequence, events):
                begin.record()
                up.launch(pooled=pooled)
                end.record()
        graph.replay()
        torch.cuda.current_stream().synchronize()

        timings = {False: [], True: []}
        for _ in range(9):
            graph.replay()
            events[-1][1].synchronize()
            for pooled, (begin, end) in zip(sequence, events):
                timings[pooled].append(begin.elapsed_time(end))

        control_ms = statistics.median(timings[False])
        pooled_ms = statistics.median(timings[True])
        print(f"Ready up-GEMM gpu={up.cuda_device_id} routes={routes} "
              f"table_rows={up.work_table.shape[0]} samples={len(timings[True])} "
              f"multi_ms={control_ms:.4f} pooled_ms={pooled_ms:.4f} "
              f"speedup={control_ms / pooled_ms:.3f}x", flush=True)
