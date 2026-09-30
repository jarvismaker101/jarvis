"""[P1-03] Barge-in must never block the real-time capture loop.

The old ``barge_in_on_speech_onset`` ran a blocking GET /voice-state probe
(0.3s) and a blocking POST /speak/stop (0.4s) from inside ``for chunk in
audio_stream`` - up to 0.7s of freeze AT SPEECH ONSET, and a hung backend
stalled the loop until the socket timeout. This pins the new contract:

* queueing a stop returns in microseconds even when the endpoint is hung;
* a stop arriving while one is in flight is never lost (coalesced flag, not a
  queue of duplicates);
* a 401/403 is recorded and logged ONCE, never treated as silence;
* the worker is a daemon and shuts down cleanly.
"""

import io
import os
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from backend.services import listener, local_auth

TOKEN = "test-launch-token-0123456789abcdef"


class _FakeResponse:
    def __init__(self, status=200):
        self.status = status

    def read(self):
        return b"{}"


class _RecordingConn:
    """Fake persistent connection; optionally HANGS like a dead backend."""

    def __init__(self, status=200, on_request=None):
        self.status = status
        self.on_request = on_request
        self.requests = []
        self.closed = False

    def request(self, method, path, body=None, headers=None):
        self.requests.append((method, path, dict(headers or {})))
        if self.on_request is not None:
            self.on_request(len(self.requests))

    def getresponse(self):
        return _FakeResponse(self.status)

    def close(self):
        self.closed = True


class _HangForeverConn(_RecordingConn):
    """Blocks until the test says so - a backend that never answers."""

    def __init__(self, release):
        super().__init__()
        self._release = release
        self.entered = threading.Event()

    def request(self, method, path, body=None, headers=None):
        self.entered.set()
        self._release.wait(timeout=10.0)
        self.requests.append((method, path, dict(headers or {})))


def _worker_with(conn):
    worker = listener._SpeakStopWorker()
    worker._open_connection = lambda: conn
    return worker


def _wait_for(predicate, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


class BargeInLatencyTests(unittest.TestCase):
    """The regression test the audit asks for: onset never waits on HTTP."""

    def test_barge_in_returns_within_milliseconds_when_the_endpoint_hangs(self):
        release = threading.Event()
        conn = _HangForeverConn(release)
        worker = _worker_with(conn)
        self.addCleanup(release.set)
        self.addCleanup(worker.shutdown)

        with patch.object(listener, "_speak_stop", worker), \
             patch.object(listener.listener_state, "is_speaking",
                          return_value=True), \
             patch("backend.services.voice.stop_speaking"):
            started = time.monotonic()
            result = listener.barge_in_on_speech_onset()
            elapsed = time.monotonic() - started

        self.assertTrue(result)
        self.assertLess(elapsed, 0.05,
                        "barge-in waited on the hung HTTP endpoint")
        # The stop is still DELIVERED - just not by the capture thread.
        self.assertTrue(conn.entered.wait(timeout=2.0),
                        "the worker never attempted the stop")

    def test_local_stop_is_synchronous_and_immediate(self):
        """Local audio must die NOW - only the remote notice is async."""
        order = []
        kwargs = {}

        def _stop(**kw):
            kwargs.update(kw)
            order.append("local_stop")

        conn = _RecordingConn(on_request=lambda n: order.append("http"))
        worker = _worker_with(conn)
        self.addCleanup(worker.shutdown)

        with patch.object(listener, "_speak_stop", worker), \
             patch.object(listener.listener_state, "is_speaking",
                          return_value=True), \
             patch("backend.services.voice.stop_speaking", _stop):
            listener.barge_in_on_speech_onset()
            # Within the call, local stop already happened; HTTP comes later.
            self.assertEqual(order, ["local_stop"])
            # [P1-02] ...and it asks for the SILENT stop: no "ready" beep in
            # the middle of the user's sentence.
            self.assertEqual(kwargs, {"signal_ready": False})
            self.assertTrue(_wait_for(lambda: "http" in order, 2.0))

    def test_zero_blocking_network_calls_on_the_capture_thread(self):
        """Acceptance: the capture path opens no socket and probes nothing."""
        conn = _RecordingConn()
        worker = _worker_with(conn)
        self.addCleanup(worker.shutdown)
        opened = []

        def _guard():
            opened.append(1)
            return conn

        with patch.object(listener, "_speak_stop", worker), \
             patch.object(worker, "_open_connection", _guard), \
             patch.object(listener, "_api_is_speaking",
                          side_effect=AssertionError("inline probe")), \
             patch.object(listener.listener_state, "is_speaking",
                          return_value=False):
            listener.barge_in_on_speech_onset()
            # The worker thread runs ASYNC - keep the patches alive until the
            # delivery happens, or the guard is restored before it fires.
            self.assertTrue(_wait_for(lambda: opened == [1], 2.0))
        self.assertEqual(worker.stats["requests"], 1)



class CoalescingTests(unittest.TestCase):
    """Two barge-ins in quick succession still deliver a stop."""

    def test_two_barge_ins_in_quick_succession_deliver_a_stop(self):
        conn = _RecordingConn()
        worker = _worker_with(conn)
        self.addCleanup(worker.shutdown)

        self.assertTrue(worker.request_stop())
        self.assertTrue(worker.request_stop())
        self.assertTrue(_wait_for(lambda: len(conn.requests) >= 1, 2.0),
                        "no stop was delivered after two barge-ins")
        self.assertLessEqual(worker.stats["requests"], 2)

    def test_a_stop_arriving_during_a_send_is_not_lost(self):
        """The newest stop must survive even mid-flight delivery."""
        entered = threading.Event()
        release = threading.Event()
        delivered = []

        def _on_request(count):
            delivered.append(count)
            if count == 1:
                entered.set()
                release.wait(timeout=10.0)

        conn = _RecordingConn(on_request=_on_request)
        worker = _worker_with(conn)
        self.addCleanup(release.set)
        self.addCleanup(worker.shutdown)

        worker.request_stop()                  # starts an in-flight send
        self.assertTrue(entered.wait(timeout=2.0))
        worker.request_stop()                  # arrives WHILE the send blocks
        release.set()                          # unblock the first send
        self.assertTrue(
            _wait_for(lambda: len(delivered) >= 2, 2.0),
            "a barge-in during an in-flight stop was swallowed")

    def test_duplicate_requests_never_pile_up(self):
        """The flag coalesces: N rapid requests cost <= 2 sends, not N."""
        conn = _RecordingConn()
        worker = _worker_with(conn)
        self.addCleanup(worker.shutdown)

        for _ in range(50):
            worker.request_stop()
        self.assertTrue(_wait_for(lambda: conn.requests, 2.0))
        time.sleep(0.2)   # let any queued re-send drain
        self.assertLessEqual(len(conn.requests), 2,
                             "the stop flag became a queue")
        self.assertEqual(worker.stats["requests"], 50)


class FailureVisibilityTests(unittest.TestCase):
    """A failing stop is LOUD once, and never silent."""

    def test_401_is_recorded_and_logged_once(self):
        conn = _RecordingConn(status=401)
        worker = _worker_with(conn)
        self.addCleanup(worker.shutdown)

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            worker._deliver()
            worker._deliver()   # second 401 must not print again
        printed = buffer.getvalue()

        self.assertEqual(worker.stats["auth_failures"], 2)
        self.assertEqual(worker.stats["delivered"], 0)
        self.assertEqual(worker.stats["failed"], 2)
        self.assertIn("401", printed)
        self.assertEqual(printed.count("rejected"), 1,
                         "the auth failure must be loud ONCE, not silent")

    def test_403_is_treated_as_an_auth_failure(self):
        conn = _RecordingConn(status=403)
        worker = _worker_with(conn)
        self.addCleanup(worker.shutdown)
        with redirect_stdout(io.StringIO()):
            worker._deliver()
        self.assertEqual(worker.stats["auth_failures"], 1)

    def test_connection_error_is_recorded_not_raised(self):
        def _boom():
            raise OSError("connection refused")

        worker = listener._SpeakStopWorker()
        worker._open_connection = _boom
        self.addCleanup(worker.shutdown)
        with redirect_stdout(io.StringIO()):
            worker._deliver()
        self.assertEqual(worker.stats["failed"], 1)
        self.assertIn("refused", worker.stats["last_error"])

    def test_success_still_records_status(self):
        conn = _RecordingConn(status=200)
        worker = _worker_with(conn)
        self.addCleanup(worker.shutdown)
        worker._deliver()
        self.assertTrue(worker.stats["last_ok"])
        self.assertEqual(worker.stats["last_status"], 200)
        self.assertEqual(worker.stats["delivered"], 1)



class WorkerLifecycleTests(unittest.TestCase):
    def test_worker_thread_is_daemon(self):
        worker = listener._SpeakStopWorker()
        self.addCleanup(worker.shutdown)
        worker.request_stop()
        self.assertIsNotNone(worker._thread)
        self.assertTrue(worker._thread.daemon)

    def test_shutdown_stops_the_thread_and_closes_the_connection(self):
        conn = _RecordingConn()
        worker = _worker_with(conn)
        worker.request_stop()
        self.assertTrue(_wait_for(lambda: conn.requests, 2.0))
        worker.shutdown(timeout=2.0)
        self.assertFalse(worker._thread.is_alive())
        self.assertTrue(conn.closed)

    def test_shutdown_with_no_work_is_clean(self):
        listener._SpeakStopWorker().shutdown()   # must not raise

    def test_connection_is_persistent_across_deliveries(self):
        """P0-12-style reuse: one socket serves many stops (no per-stop dial)."""
        made = []

        def _make():
            conn = _RecordingConn()
            made.append(conn)
            return conn

        worker = listener._SpeakStopWorker()
        worker._open_connection = _make
        self.addCleanup(worker.shutdown)
        worker._deliver()
        worker._deliver()
        self.assertEqual(len(made), 1, "the connection was not kept alive")
        self.assertEqual(len(made[0].requests), 2)

    def test_token_rides_every_delivery(self):
        conn = _RecordingConn()
        worker = _worker_with(conn)
        self.addCleanup(worker.shutdown)
        with patch.dict(os.environ, {"JARVIS_LOCAL_TOKEN": TOKEN}):
            worker._deliver()
        headers = {k.lower(): v for k, v in conn.requests[0][2].items()}
        self.assertEqual(headers.get(local_auth.HEADER.lower()), TOKEN)


class TelemetryTests(unittest.TestCase):
    def test_stop_duration_and_result_are_recorded(self):
        conn = _RecordingConn(status=200)
        worker = _worker_with(conn)
        self.addCleanup(worker.shutdown)
        worker._deliver()
        self.assertGreaterEqual(worker.stats["last_ms"], 0.0)
        self.assertTrue(worker.stats["last_ok"])

    def test_barge_in_stop_mark_is_emitted_for_p1_19(self):
        conn = _RecordingConn(status=200)
        worker = _worker_with(conn)
        self.addCleanup(worker.shutdown)
        with patch.object(listener._latency, "mark_active") as mock_mark:
            worker._deliver()
        mock_mark.assert_called_once()
        name = mock_mark.call_args[0][0]
        meta = mock_mark.call_args[0][1]
        self.assertEqual(name, "barge_in_stop")
        self.assertIn("ok", meta)
        self.assertIn("ms", meta)

    def test_speak_stop_stats_snapshot_is_isolated(self):
        stats = listener.speak_stop_stats()
        stats["delivered"] = 999
        self.assertNotEqual(listener.speak_stop_stats()["delivered"], 999)


class UnchangedBehaviourTests(unittest.TestCase):
    """Constraints: onset timing and the local stop are untouched."""

    def test_speech_start_constants_are_untouched(self):
        self.assertEqual(listener.SPEECH_START_CONFIRMATION_CHUNKS, 2)
        self.assertEqual(listener.SPEECH_START_MIN_SECONDS, 0.10)
        self.assertEqual(listener.SPEECH_START_WINDOW_CHUNKS, 6)

    def test_local_stop_is_not_deferred_by_a_failing_worker(self):
        """Even when queueing the remote stop fails, local audio still stops."""
        called = []

        def _stop(**_kw):
            called.append(1)

        with patch.object(listener.listener_state, "is_speaking",
                          return_value=True), \
             patch("backend.services.voice.stop_speaking", _stop), \
             patch.object(listener, "_post_backend_speak_stop",
                          return_value=False):
            # Must not raise; the local stop already happened synchronously.
            listener.barge_in_on_speech_onset()
        self.assertEqual(called, [1])


if __name__ == "__main__":
    unittest.main()

