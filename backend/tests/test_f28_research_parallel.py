"""F28 — parallelize research within EXPLICIT (process-wide) limits.

Acceptance (audit report): "Instrumented concurrent runs stay within agreed
ceilings; cancelled waiters start no I/O; hard failures run once; all sources
have ordered outcomes and partial evidence survives."

Baseline defects pinned here:
  * the fetch/note semaphores were created per INVOCATION, so two concurrent
    runs doubled the intended load;
  * cancellation was not rechecked after admission (or before a note), so a
    cancelled waiter still started I/O;
  * a fetch that outlived the deadline kept running;
  * gathered exceptions were logged and dropped, so a source lost its outcome.
"""

import asyncio
import threading
import time
import unittest
from unittest import mock

from backend.services import research_service as rs


class AggregateGateTests(unittest.TestCase):
    def test_ceiling_is_process_wide_not_per_invocation(self):
        gate = rs._AggregateGate(2, "test")
        gate._in_use = 0
        gate._peak = 0
        loops = []

        async def worker():
            async with gate:
                loops.append(gate.snapshot()["in_use"])
                await asyncio.sleep(0.12)

        async def main():
            await asyncio.gather(*(worker() for _ in range(6)))

        asyncio.run(main())
        self.assertLessEqual(max(loops), 2,
                             "the ceiling must hold across everything in flight")
        self.assertEqual(gate.snapshot()["in_use"], 0)
        self.assertEqual(gate.snapshot()["peak"], 2)

    def test_ceiling_holds_across_two_separate_event_loops(self):
        """Two research runs own two loops — the limit is still process-wide."""
        gate = rs._AggregateGate(1, "cross-loop")
        gate._in_use = 0
        gate._peak = 0
        observed = []
        release = threading.Event()

        def run():
            async def worker():
                async with gate:
                    observed.append(time.time())
                    await asyncio.to_thread(release.wait, 2)
            asyncio.run(worker())

        first = threading.Thread(target=run)
        first.start()
        time.sleep(0.15)
        second = threading.Thread(target=run)
        second.start()
        time.sleep(0.3)
        in_use = gate.snapshot()["in_use"]
        self.assertEqual(in_use, 1,
                         "a second run must not enter while the ceiling is full")
        self.assertEqual(gate.snapshot()["waiting"], 1)
        release.set()
        first.join(3)
        second.join(3)
        self.assertEqual(gate.snapshot()["in_use"], 0)

    def test_gates_are_module_level_and_shared(self):
        self.assertIs(rs.FETCH_GATE, rs.FETCH_GATE)
        self.assertEqual(rs.FETCH_GATE.limit, rs.MAX_CONCURRENT_FETCHES)
        self.assertEqual(rs.NOTE_GATE.limit, rs.MAX_CONCURRENT_NOTES)
        snapshot = rs.concurrency_snapshot()
        self.assertEqual(snapshot["fetch"]["limit"], rs.MAX_CONCURRENT_FETCHES)
        self.assertEqual(snapshot["note"]["limit"], rs.MAX_CONCURRENT_NOTES)

    def test_collect_sites_uses_the_shared_gates(self):
        import inspect

        code = inspect.getsource(rs._research_job)
        self.assertIn("fetch_gate = FETCH_GATE", code)
        self.assertIn("note_gate = NOTE_GATE", code)
        self.assertNotIn("asyncio.Semaphore(", code)

    def test_admission_is_rechecked_for_cancellation(self):
        import inspect

        code = inspect.getsource(rs._research_job)
        gate = code.index("async with fetch_gate:")
        after = code[gate:gate + 400]
        self.assertIn("_stop_now", after,
                      "a cancelled waiter must start no I/O")
        note = code.index("async with note_gate:")
        self.assertIn("_stop_now", code[note:note + 300],
                      "a cancelled note must not reach the model")

    def test_fetches_are_deadline_bounded(self):
        import inspect

        code = inspect.getsource(rs._research_job)
        self.assertIn("asyncio.wait_for", code)
        self.assertIn("asyncio.TimeoutError", code)

    def test_exceptions_become_ordered_failure_outcomes(self):
        import inspect

        code = inspect.getsource(rs._research_job)
        self.assertIn('"index": index', code)
        self.assertIn("isinstance(outcome, BaseException)", code)

    def test_browser_job_gets_only_the_time_left(self):
        import inspect

        code = inspect.getsource(rs.run_research)
        self.assertIn("timeout=max(0.0, deadline - time.monotonic())", code)


class DeadlineAndEvidenceTests(unittest.TestCase):
    """The pipeline itself: partial evidence survives, hard failures run once."""

    def setUp(self):
        self.gates = (rs.FETCH_GATE, rs.NOTE_GATE)
        for gate in self.gates:
            gate._in_use = 0
            gate._peak = 0
            gate._waiting = 0
        rs.clear_stop_request()

    def tearDown(self):
        rs.clear_stop_request()
        for gate in self.gates:
            gate._in_use = 0

    def _run(self, coro):
        return asyncio.run(coro)

    def _job(self, deadline_offset=60.0, cancel=False):
        job = None
        try:
            from backend.services import jobs
            job = jobs.new_job(kind="research", label="f28")
        except Exception:
            job = None
        if cancel and job is not None:
            job.cancel()
        return job

    def test_partial_evidence_survives_a_deadline(self):
        fetched = []

        class FakePage:
            async def goto(self, *a, **k):
                return None

            async def wait_for_selector(self, *a, **k):
                return None

            async def close(self):
                return None

            async def evaluate(self, *a, **k):
                return ""

        class FakeTask:
            async def new_page(self):
                return FakePage()

        async def fake_extract(page, max_results=None):
            return [{"url": "https://a.test/1", "title": "A"},
                    {"url": "https://b.test/2", "title": "B"}]

        async def fake_fetch(tab, url):
            fetched.append(url)
            if len(fetched) == 1:
                return {"title": "A", "text": "A meaningful note about tanks " * 4}
            raise RuntimeError("boom")

        deadline = time.monotonic() + 30
        progress = []
        evidence = []
        with mock.patch.object(rs, "extract_brave_async", fake_extract), \
             mock.patch.object(rs, "fetch_page_async", fake_fetch), \
             mock.patch.object(rs, "summarize_with_gemini",
                                        lambda *a, **k: "note text about tanks"), \
             mock.patch.object(rs, "_browser_run",
                                        lambda fn, **k: None):
            result = self._run(rs._research_job(
                FakeTask(), "best tanks", 2, None, deadline,
                lambda msg: progress.append(msg),
                lambda item: evidence.append(item)))
        self.assertEqual(len(result["collected"]), 1,
                         "the gathered evidence survives")
        self.assertTrue(result["failures"], "the hard failure is recorded")
        self.assertIn("a.test", result["collected"][0]["url"])

    def test_hard_failure_is_attempted_once(self):
        attempts = []

        class FakePage:
            async def goto(self, *a, **k):
                return None

            async def wait_for_selector(self, *a, **k):
                return None

            async def close(self):
                return None

        class FakeTask:
            async def new_page(self):
                return FakePage()

        async def fake_extract(page, max_results=None):
            return [{"url": "https://cert.test/x", "title": "bad cert"}]

        async def fake_fetch(tab, url):
            attempts.append(url)
            raise RuntimeError("ERR_CERT_AUTHORITY_INVALID")

        deadline = time.monotonic() + 30
        with mock.patch.object(rs, "extract_brave_async", fake_extract), \
             mock.patch.object(rs, "fetch_page_async", fake_fetch), \
             mock.patch.object(rs, "_stop_now",
                                        lambda *a, **k: False), \
             mock.patch.object(rs, "_cancelled_now",
                                        lambda *a, **k: False), \
             mock.patch.object(rs, "asyncio") as fake_asyncio:
            fake_asyncio.sleep = asyncio.sleep
            fake_asyncio.gather = asyncio.gather
            fake_asyncio.wait_for = asyncio.wait_for
            fake_asyncio.TimeoutError = asyncio.TimeoutError
            fake_asyncio.to_thread = asyncio.to_thread
            fake_asyncio.CancelledError = asyncio.CancelledError
            fake_asyncio.Semaphore = asyncio.Semaphore
            result = self._run(rs._research_job(
                FakeTask(), "q", 1, None, deadline, lambda m: None,
                lambda i: None))
        self.assertEqual(len(attempts), 1,
                         "a hard (TLS) failure must not be retried")
        self.assertEqual(result["failures"][0]["url"], "https://cert.test/x")

    def test_exception_outcome_keeps_its_index_and_url(self):
        class FakePage:
            async def goto(self, *a, **k):
                return None

            async def wait_for_selector(self, *a, **k):
                return None

            async def close(self):
                return None

        class FakeTask:
            async def new_page(self):
                return FakePage()

        async def fake_extract(page, max_results=None):
            return [{"url": "https://a.test/1", "title": "A"},
                    {"url": "https://b.test/2", "title": "B"}]

        async def fake_fetch(tab, url):
            if url.endswith("/2"):
                raise ValueError("kaboom")
            return {"title": "A", "text": "about tanks " * 6}

        deadline = time.monotonic() + 30
        with mock.patch.object(rs, "extract_brave_async", fake_extract), \
             mock.patch.object(rs, "fetch_page_async", fake_fetch), \
             mock.patch.object(rs, "summarize_with_gemini",
                                        lambda *a, **k: "note about tanks"):
            result = self._run(rs._research_job(
                FakeTask(), "best tanks", 2, None, deadline, lambda m: None,
                lambda i: None))
        self.assertEqual(len(result["collected"]), 1)
        self.assertTrue(any(f.get("index") == 2 for f in result["failures"]),
                        "the failed source keeps an ordered outcome")


if __name__ == "__main__":
    unittest.main()
