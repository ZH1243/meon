import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from benchmark_ready_up_gemm import check_ready_outputs


class ReadyReplayTests(unittest.TestCase):
    def test_valid_rows_and_poison_across_replays(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is not installed")

        # The two formats address the same physical source rows in mixed order.
        pool = torch.arange(12, dtype=torch.float32).view(6, 2)
        x0, x1 = pool.chunk(2)
        weights = torch.tensor([[1., -1.], [2., 4.]])
        expected = torch.stack((pool[4], pool[1], pool[5])) @ weights
        output = torch.full((4, 2), 43.)
        output[:3].copy_(expected)
        calls = []

        def launch(*, pooled):
            calls.append(pooled)
            rows = pool[[4, 1, 5]] if pooled else torch.stack((x1[1], x0[1], x1[2]))
            output[:3].copy_(rows @ weights)

        up = SimpleNamespace(torch=torch, ready_rows=torch.tensor(4),
                             work_table=torch.empty((4, 6)), output=output, launch=launch)
        check_ready_outputs(up, 3)
        self.assertEqual(calls, [False, True])
        torch.testing.assert_close(output[:3], expected)
        torch.testing.assert_close(output[3], torch.full((2,), 43.))

        # Simulate a candidate that forgets the final partial row. The control
        # just produced correct values there, so a replay without poison passes.
        def skip_last(*, pooled):
            if pooled:
                output[:2].copy_(expected[:2])
            else:
                launch(pooled=False)

        up.launch = skip_last
        with self.assertRaisesRegex(AssertionError, "pooled=True"):
            check_ready_outputs(up, 3)

        up.ready_rows.zero_()
        with self.assertRaisesRegex(RuntimeError, "fully published"):
            check_ready_outputs(up, 3)
