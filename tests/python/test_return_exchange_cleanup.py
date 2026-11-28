"""CPU checks for successful CUDA IPC ownership teardown."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import importlib.util
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import weakref


PATH = Path(__file__).resolve().parents[2] / "python" / "rdma_return_exchange.py"
SPEC = importlib.util.spec_from_file_location("rdma_return_exchange_cleanup", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Storage:
    pass


class ReturnCountTests(unittest.TestCase):
    def test_counts_for_output_validation_and_older_peers(self):
        peers = [{"optional_return_counts": True}] * 8
        for combine, check_copy, check_combine, expected in (
                (False, False, False, True), (True, False, False, False),
                (True, True, False, True), (True, False, True, True)):
            options = SimpleNamespace(intra_node_combine=combine, check_return_token_results=check_copy,
                                      check_intra_node_combine=check_combine)
            self.assertEqual(MODULE._needs_return_counts(options, peers), expected)
            self.assertTrue(MODULE._needs_return_counts(options, [*peers[:-1], {}]))


class CompletionTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        records = {}
        lock = threading.Lock()
        self.arrived = threading.Event()
        self.ranks = []
        for rank in range(3):
            def state(stage=None, value=None, rank=rank):
                with lock:
                    if stage is not None:
                        records.setdefault(stage, {})[str(rank)] = value
                        if stage == "done-0" and rank == 0:
                            self.arrived.set()
                    return {key: dict(values) for key, values in records.items()}

            exchange = MODULE.ReturnExchange.__new__(MODULE.ReturnExchange)
            exchange.control = SimpleNamespace(path=directory / "exchange", rank=rank, world=3,
                                               timeout=3, state=state)
            exchange.completion_wait_ns = {}
            exchange.wakeups = MODULE._CompletionWakeups(exchange.control)
            stack.callback(exchange.wakeups.close)
            self.ranks.append(exchange)
        paths = [rank.wakeups.path for rank in self.ranks]
        for rank in self.ranks:
            rank.wakeups.connect(paths)

    def test_late_rank_and_repeated_collectives(self):
        for rank in self.ranks:
            rank.wakeups.directory.cleanup()  # Setup has retired the FIFO paths.
        def run(rank):
            exchange = self.ranks[rank]
            for iteration in range(8):
                counts = exchange.gather_completion("done", iteration, [rank, iteration])
                self.assertEqual(counts, [[peer, iteration] for peer in range(3)])
                self.assertEqual(exchange.gather_completion("intra-node-done", iteration), [True] * 3)
        with ThreadPoolExecutor(3) as pool:
            first = pool.submit(run, 0)
            self.assertTrue(self.arrived.wait(1))
            self.ranks[2].wakeups.notify()  # Unrelated wakeup cannot release rank zero.
            self.assertFalse(first.done())
            others = [pool.submit(run, rank) for rank in (1, 2)]
            for task in [first, *others]:
                task.result()

    def test_abort_wakes_waiter_and_closes_fifo(self):
        with ThreadPoolExecutor(1) as pool:
            waiter = pool.submit(self.ranks[0].gather_completion, "done", 0)
            self.assertTrue(self.arrived.wait(1))
            path = Path(self.ranks[1].wakeups.path)
            self.ranks[1]._abort_completion("injected failure")
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                waiter.result(timeout=1)
            self.assertFalse(path.exists())

    def test_partial_connect_cleanup(self):
        extra = MODULE._CompletionWakeups(self.ranks[0].control)
        try:
            self.assertRaises(FileNotFoundError, extra.connect,
                              [extra.path, self.ranks[1].wakeups.path, extra.path + "-missing"])
        finally:
            extra.close()
        self.assertFalse(Path(extra.path).exists())

    def test_legacy_peer_uses_existing_gather(self):
        exchange = self.ranks[0]
        exchange.wakeups = None
        exchange.control.gather = Mock(return_value=[[1], [0], [3]])
        self.assertEqual(exchange.gather_completion("done", 0, [1]), [[1], [0], [3]])
        exchange.control.gather.assert_called_once_with("done-0", [1])

    def test_timeout_reports_arrived_ranks(self):
        self.ranks[0].control.timeout = 0.01
        with self.assertRaisesRegex(TimeoutError, "done-1.*arrived.*0"):
            self.ranks[0].gather_completion("done", 1)


class CleanupTests(unittest.TestCase):
    def test_success_releases_storage_after_imports_closed(self):
        exchange = MODULE.ReturnExchange.__new__(MODULE.ReturnExchange)
        events = []
        exchange.closed = False
        exchange.wakeups = None
        exchange.control = SimpleNamespace(
            gather=lambda phase: events.append(phase),
            close=lambda: events.append("control_closed"))
        exchange.cuda = SimpleNamespace(close=lambda: events.append("imports_closed_local"))
        storage = Storage()
        reference = weakref.ref(storage)
        export = SimpleNamespace(storage=storage, release=lambda: events.append("counter_released"))
        exchange.exports = [export]
        exchange.outputs = (storage,)
        exchange.return_maps = SimpleNamespace(slab=storage, check=exchange.require)
        exchange.received = {0: storage}
        exchange.received_valid_rows = {0: 1}
        down = SimpleNamespace(return_exchange=exchange)
        exchange.down = down
        del storage, export

        exchange.close(True)
        self.assertEqual(events, ["workers_stopped", "imports_closed_local",
                                  "imports_closed", "counter_released", "control_closed"])
        self.assertIsNone(reference())
        self.assertIsNone(down.return_exchange)
        self.assertIsNone(exchange.down)
        exchange.close(True)
        self.assertEqual(len(events), 5)

    def test_failure_retains_resources(self):
        exchange = MODULE.ReturnExchange.__new__(MODULE.ReturnExchange)
        exchange.closed = False
        exchange.wakeups = None
        exchange.control = Mock()
        exchange.cuda = Mock()
        exchange.exports = [Mock()]
        try:
            exchange.close(False)
            self.assertIn(exchange, MODULE._failed_exchanges)
            exchange.cuda.close.assert_not_called()
            exchange.exports[0].release.assert_not_called()
        finally:
            MODULE._failed_exchanges.remove(exchange)


if __name__ == "__main__":
    unittest.main()
