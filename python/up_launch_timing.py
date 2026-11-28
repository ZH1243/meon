"""Host timing for one worker's measured up-GEMM launches."""

from time import perf_counter_ns


class UpLaunchTiming:
    def __init__(self, nvtx):
        self.nvtx = nvtx
        self.marks = {}
        self.iterations = []

    def mark(self, name):
        self.marks[name] = perf_counter_ns()

    def timed(self, name, call):
        """Wrap a submission-thread call. The CUDA stream remains asynchronous."""
        label = f"meon/up/{name}"
        start, end = f"{name}_start", f"{name}_end"

        def measured(*args, **kwargs):
            with self.nvtx.range(label):
                self.mark(start)
                try:
                    return call(*args, **kwargs)
                finally:
                    self.mark(end)

        return measured

    def finish_iteration(self, extents):
        self.iterations.append((extents, self.marks))
        self.marks = {}

    def report(self, node, gpu):
        for iteration, ((source_rows, table_rows), marks) in enumerate(self.iterations):
            fields = []
            for name in ("bind_routing", "coordinator_submit", "up_call"):
                if f"{name}_start" in marks:
                    elapsed = marks[f"{name}_end"] - marks[f"{name}_start"]
                    fields.append(f"{name}_us={elapsed / 1000:.3f}")
            cache = "no_lookup" if table_rows else "empty"
            if "key_start" in marks:
                cache = "hit" if "cache_hit" in marks else "miss"
                lookup_end = marks[f"cache_{cache}"]
                plan_ready = lookup_end if cache == "hit" else marks["plan_built"]
                for name, elapsed in (
                    ("pre_key", marks["key_start"] - marks["up_call_start"]),
                    ("plan_lookup", lookup_end - marks["key_start"]),
                    ("plan_build", plan_ready - lookup_end),
                    ("after_plan", marks["up_call_end"] - plan_ready),
                ):
                    fields.append(f"{name}_us={elapsed / 1000:.3f}")
            print(f"  up_launch_timing node={node} gpu={gpu} iteration={iteration} "
                  f"source_index_rows={source_rows} table_rows={table_rows} plan_cache={cache} "
                  f"{' '.join(fields)}", flush=True)
