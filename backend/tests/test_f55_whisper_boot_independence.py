"""F55 — boot must not depend on the whisper model load.

Found while investigating "Jarvis does not boot any more, it just hangs on
terminals". On this machine (1 GB RAM free with a game running) the boot path
spent its time on the whisper daemon:

1. The daemon bound port 8767 only AFTER ``WhisperModel`` finished loading
   (10-20s on a healthy machine, minutes when the box is thrashing), so the
   supervisor's readiness probe expired and it loaded a SECOND copy of the
   same ~1.5 GB model in-process. A slow boot became a stuck one.
2. ``ThreadingHTTPServer`` binds with ``allow_reuse_address = True``, which on
   Windows lets a second daemon bind the same port, so two "healthy" daemons
   can share one port and connections land on either.

These tests pin the corrected contract: the port is served before the model is
loaded, ``/health`` says ``ok`` while ``ready`` is still false so a loading
daemon is ADOPTED rather than duplicated, transcription waits for the model
instead of failing, the port cannot be shared, and a listener that holds the
port without serving the API is never doubled by a doomed second binder.
"""

import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend import whisper_daemon


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeModel:
    def transcribe(self, _audio, **_kwargs):
        return [_FakeSegment(" hello")], SimpleNamespace(
            language="en", language_probability=0.91
        )


def _fake_proc(pid=31337, alive=True, returncode=None):
    proc = Mock()
    proc.pid = pid
    proc.poll.return_value = None if alive else (returncode or 1)
    proc.returncode = returncode
    return proc


class _ServingDaemon(unittest.TestCase):
    """A real HTTP server on an ephemeral port, with the model state stubbed."""

    def setUp(self):
        self._saved = {
            "model": whisper_daemon.model,
            "device": whisper_daemon.device,
            "_model_error": whisper_daemon._model_error,
            "_model_load_finished": whisper_daemon._model_load_finished,
            "_model_load_started": whisper_daemon._model_load_started,
            "TRANSCRIBE_MODEL_WAIT": whisper_daemon.TRANSCRIBE_MODEL_WAIT,
        }
        whisper_daemon.model = None
        whisper_daemon.device = "unknown"
        whisper_daemon._model_error = None
        whisper_daemon._model_load_finished = threading.Event()
        whisper_daemon._model_load_started = False
        self.server = whisper_daemon.build_server(0, bind_retry_seconds=0)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop)

    def _stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        for name, value in self._saved.items():
            setattr(whisper_daemon, name, value)

    def _get(self, path):
        with urllib.request.urlopen(
            f"http://127.0.0.1:{self.port}{path}", timeout=10
        ) as response:
            return json.load(response)

    def _post(self, path, body=b"RIFFxxxx"):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=body, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)


class BindsBeforeModelLoadTests(_ServingDaemon):
    """The whole point of F55: adoptable before the model exists."""

    def test_health_answers_while_the_model_is_still_loading(self):
        release = threading.Event()
        finished = threading.Event()

        def slow_load():
            release.wait(timeout=5)
            whisper_daemon.model = _FakeModel()
            whisper_daemon._model_load_finished.set()
            finished.set()

        with patch.object(whisper_daemon, "load_model", side_effect=slow_load):
            whisper_daemon.ensure_model_loading()
            started = time.monotonic()
            payload = self._get("/health")
            elapsed = time.monotonic() - started
            release.set()
            self.assertTrue(finished.wait(timeout=5), "the load must complete")

        self.assertTrue(payload["ok"], "a bound daemon is alive, model or not")
        self.assertEqual(payload["service"], "jarvis-whisper")
        self.assertFalse(payload["ready"])
        self.assertTrue(payload["loading"])
        self.assertIsNone(payload["error"])
        self.assertLess(
            elapsed, 2.0,
            "the listener must answer immediately instead of holding the port "
            "until a 10-20s (or minutes-long) model load finishes")

    def test_health_reports_ready_once_the_model_is_loaded(self):
        whisper_daemon.model = _FakeModel()
        whisper_daemon.device = "cuda"
        whisper_daemon._model_load_finished.set()

        payload = self._get("/health")

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["ready"])
        self.assertFalse(payload["loading"])
        self.assertEqual(payload["device"], "cuda")

    def test_a_second_daemon_cannot_share_the_port(self):
        """Two binders on one port look healthy while nobody is served."""
        with self.assertRaises(OSError):
            whisper_daemon.build_server(self.port, bind_retry_seconds=0)


class TranscribeWaitsForTheModelTests(_ServingDaemon):
    def test_transcribe_waits_for_a_loading_model_and_then_answers(self):
        whisper_daemon.TRANSCRIBE_MODEL_WAIT = 10.0

        def finish_load():
            time.sleep(0.4)
            whisper_daemon.model = _FakeModel()
            whisper_daemon._model_load_finished.set()

        threading.Thread(target=finish_load, daemon=True).start()

        status, payload = self._post("/transcribe")

        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["text"], "hello")

    def test_transcribe_reports_loading_when_the_wait_expires(self):
        whisper_daemon.TRANSCRIBE_MODEL_WAIT = 0.3

        status, payload = self._post("/transcribe")

        self.assertEqual(status, 503)
        self.assertFalse(payload["ok"])
        self.assertIn("still loading", payload["error"])

    def test_transcribe_reports_a_model_load_failure(self):
        whisper_daemon._model_error = "cuda missing"
        whisper_daemon._model_load_finished.set()

        status, payload = self._post("/transcribe")

        self.assertEqual(status, 503)
        self.assertEqual(payload["error"], "cuda missing")


class _WatcherDaemonBase(unittest.TestCase):
    """Hermetic seams for the whisper starter: no real spawn, no real kill."""

    def setUp(self):
        from backend import watcher

        self.watcher = watcher
        self._saved_proc = watcher.whisper_daemon_proc
        self._saved_ok = watcher.whisper_daemon_ok
        watcher.whisper_daemon_proc = None
        watcher.whisper_daemon_ok = False

    def tearDown(self):
        self.watcher.whisper_daemon_proc = self._saved_proc
        self.watcher.whisper_daemon_ok = self._saved_ok


class WatcherAdoptsInsteadOfDuplicatingTests(_WatcherDaemonBase):
    def test_a_daemon_still_loading_its_model_is_adopted(self):
        """ok=true while ready=false must be enough to adopt (F55)."""
        loading = {
            "ok": True,
            "service": "jarvis-whisper",
            "ready": False,
            "loading": True,
            "pid": 555,
        }

        with patch.object(self.watcher, "_whisper_daemon_health", return_value=loading), \
             patch.object(self.watcher, "_attributed_whisper_pid", return_value=555), \
             patch.object(self.watcher, "_pids_on_port", return_value={555}), \
             patch.object(self.watcher, "_register_owned"), \
             patch.object(self.watcher.subprocess, "Popen") as popen:
            ok = self.watcher.ensure_whisper_daemon()

        self.assertTrue(ok)
        popen.assert_not_called()
        self.assertTrue(self.watcher.whisper_daemon_ok)
        self.assertIsNone(
            self.watcher.whisper_daemon_proc,
            "an adopted daemon is tracked by pid, not by a handle we do not own")

    def test_a_foreign_listener_blocks_a_second_binder(self):
        with patch.object(self.watcher, "_whisper_daemon_healthy", return_value=False), \
             patch.object(self.watcher, "_pids_on_port", return_value={777}), \
             patch.object(self.watcher.subprocess, "Popen") as popen:
            ok = self.watcher.ensure_whisper_daemon()

        self.assertFalse(ok)
        popen.assert_not_called()

    def test_a_launcher_exit_gets_a_grace_window_to_bind(self):
        """The Windows venv shim can exit while its interpreter keeps serving."""
        proc = _fake_proc(alive=False, returncode=0)

        with patch.object(self.watcher, "WHISPER_EXIT_GRACE", 0.5), \
             patch.object(self.watcher, "WHISPER_READY_TIMEOUT", 5.0), \
             patch.object(self.watcher, "_whisper_daemon_healthy",
                          side_effect=[False, True]), \
             patch.object(self.watcher, "_pids_on_port", return_value=set()), \
             patch.object(self.watcher, "_register_owned"), \
             patch.object(self.watcher.subprocess, "Popen", return_value=proc):
            ok = self.watcher.ensure_whisper_daemon()

        self.assertTrue(ok)

    def test_a_daemon_that_never_serves_is_not_left_tracked(self):
        proc = _fake_proc(alive=False, returncode=1)

        with patch.object(self.watcher, "WHISPER_EXIT_GRACE", 0.5), \
             patch.object(self.watcher, "WHISPER_READY_TIMEOUT", 5.0), \
             patch.object(self.watcher, "_whisper_daemon_healthy", return_value=False), \
             patch.object(self.watcher, "_pids_on_port", return_value=set()), \
             patch.object(self.watcher, "_register_owned"), \
             patch.object(self.watcher.subprocess, "Popen", return_value=proc):
            ok = self.watcher.ensure_whisper_daemon()

        self.assertFalse(ok)
        self.assertIsNone(
            self.watcher.whisper_daemon_proc,
            "a dead pid must not stay tracked: Windows recycles pids (F52)")

    def test_readiness_window_is_short_and_tunable(self):
        self.assertLessEqual(
            self.watcher.WHISPER_READY_TIMEOUT, 30.0,
            "a bound daemon answers in milliseconds, so this is a failure "
            "window - the old 60s wait was a model-load budget")

        with patch.dict(os.environ, {"JARVIS_WHISPER_TEST_SECS": "not-a-number"}):
            self.assertEqual(
                self.watcher._env_seconds("JARVIS_WHISPER_TEST_SECS", 7.0, 1.0), 7.0)
        with patch.dict(os.environ, {"JARVIS_WHISPER_TEST_SECS": "0.1"}):
            self.assertEqual(
                self.watcher._env_seconds("JARVIS_WHISPER_TEST_SECS", 7.0, 1.0), 1.0)


class LazyModelLoadTests(_ServingDaemon):
    """Booting Jarvis must not pay for the model load at all (F55)."""

    def test_model_load_is_not_started_until_it_is_asked_for(self):
        payload = self._get("/health")

        self.assertTrue(payload["ok"])
        self.assertFalse(payload["ready"])
        self.assertFalse(
            payload["loading"],
            "nothing may load while the backend is still starting up")
        self.assertFalse(whisper_daemon._model_load_started)

    def test_warm_request_starts_the_load_exactly_once(self):
        calls = []

        with patch.object(whisper_daemon, "load_model",
                          side_effect=lambda: calls.append(1)):
            first_status, first = self._post("/warm", body=b"")
            second_status, second = self._post("/warm", body=b"")

        self.assertEqual(first_status, 200)
        self.assertTrue(first["started"])
        self.assertEqual(second_status, 200)
        self.assertFalse(second["started"], "the warm-up is idempotent")
        time.sleep(0.2)
        self.assertEqual(len(calls), 1)

    def test_transcribe_starts_the_load_itself(self):
        """A wake that arrives before the warm-up still transcribes."""

        def fake_load():
            whisper_daemon.model = _FakeModel()
            whisper_daemon.device = "cuda"
            whisper_daemon._model_load_finished.set()

        with patch.object(whisper_daemon, "load_model", side_effect=fake_load):
            status, payload = self._post("/transcribe")

        self.assertEqual(status, 200)
        self.assertEqual(payload["text"], "hello")


class BackendReadinessTests(unittest.TestCase):
    """A launcher shim exit must not throw away a live backend (F55)."""

    def test_a_launcher_exit_does_not_discard_a_live_backend(self):
        from backend import watcher

        proc = _fake_proc(alive=False, returncode=0)
        health = {"ok": True, "service": "jarvis-backend"}

        with patch.object(watcher, "_backend_health", return_value=health):
            started = time.monotonic()
            ready = watcher.wait_for_backend_ready(timeout_seconds=5, proc=proc)
            elapsed = time.monotonic() - started

        self.assertTrue(ready, "the health probe decides, not the shim's exit")
        self.assertLess(elapsed, 1.0)

    def test_a_dead_backend_fails_after_the_grace_window(self):
        from backend import watcher

        proc = _fake_proc(alive=False, returncode=1)

        with patch.object(watcher, "BACKEND_EXIT_GRACE", 0.5), \
             patch.object(watcher, "_backend_health", return_value=None):
            started = time.monotonic()
            ready = watcher.wait_for_backend_ready(timeout_seconds=30, proc=proc)
            elapsed = time.monotonic() - started

        self.assertFalse(ready)
        self.assertLess(
            elapsed, 5.0,
            "a backend that is gone must not burn the full readiness timeout")


class WhisperWarmupTests(unittest.TestCase):
    def test_warmup_request_runs_off_the_calling_thread(self):
        from backend import watcher

        seen = {}
        released = threading.Event()

        class _Response:
            def read(self):
                return b"{}"

            def __enter__(self):
                return self

            def __exit__(self, *_exc):
                return False

        def slow_urlopen(*_args, **_kwargs):
            seen["thread"] = threading.current_thread().name
            released.wait(timeout=5)
            return _Response()

        with patch.object(watcher, "urlopen", slow_urlopen):
            started = time.monotonic()
            thread = watcher.warm_whisper_daemon_model()
            elapsed = time.monotonic() - started
            released.set()
            thread.join(timeout=5)

        self.assertLess(
            elapsed, 0.5,
            "the warm-up must never hold up the launch it runs after")
        self.assertEqual(seen["thread"], "whisper-model-warmup")
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()