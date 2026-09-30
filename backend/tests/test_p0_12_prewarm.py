"""P0-12 — pre-warmed connections on the latency-critical path.

The audit: nothing opened the provider connections before a turn needed them,
the voice worker knew the user was about to speak and did nothing with the head
start, the STT call had no connection pool at all, and no socket carried
keepalive so a held socket could be dropped silently by a VPN/NAT.

These tests pin the contract, not the timing: the endpoint is authenticated,
rate-limited per process, skipped rather than queued while a warm is in flight,
silent-but-counted on a provider outage, actually drains the response body, and
is wired to VAD onset off the capture thread.
"""

import os
import socket
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from backend.services import local_auth, prewarm


def _make_client():
    from fastapi.testclient import TestClient
    from backend.main import app

    return TestClient(app)


class _FakeResponse:
    """Minimal stand-in that records whether its body was read."""

    def __init__(self, status_code=200, body=b'{"data": []}'):
        self.status_code = status_code
        self._body = body
        self.content_reads = 0
        self.closed = False

    @property
    def content(self):
        self.content_reads += 1
        return self._body

    def close(self):
        self.closed = True


class PrewarmAuthTests(unittest.TestCase):
    """Acceptance: /prewarm is a control endpoint, so it fails closed."""

    def setUp(self):
        os.environ.pop("JARVIS_DEV_MODE", None)
        self.token = local_auth.mint_token()
        local_auth.configure(self.token)
        prewarm.reset()
        self.addCleanup(self._disarm)

    def _disarm(self):
        local_auth.configure("")
        os.environ.pop("JARVIS_LOCAL_TOKEN", None)
        os.environ.pop("JARVIS_DEV_MODE", None)
        prewarm.reset()

    def test_prewarm_requires_the_token(self):
        client = _make_client()
        self.assertEqual(client.post("/prewarm").status_code, 401)
        self.assertEqual(client.post("/prewarm?force=true").status_code, 401)
        self.assertEqual(client.get("/prewarm/stats").status_code, 401)

    def test_prewarm_is_reachable_with_the_token(self):
        client = _make_client()
        with patch.object(prewarm, "_warm_one", return_value=("x", "warm", {})):
            response = client.post("/prewarm",
                                   headers={local_auth.HEADER: self.token})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["ok"])


class PrewarmRateLimitTests(unittest.TestCase):
    """Acceptance: once per 20s per process; in-flight warms are skipped."""

    def setUp(self):
        prewarm.reset()
        self.addCleanup(prewarm.reset)

    def test_the_window_is_the_documented_twenty_seconds(self):
        self.assertEqual(prewarm.WARM_INTERVAL_SECONDS, 20.0)

    def test_a_second_warm_inside_the_window_is_skipped(self):
        calls = []

        def fake_warm_one(target):
            calls.append(target["name"])
            return target["name"], "warm", {}

        with patch.object(prewarm, "_warm_one", side_effect=fake_warm_one), \
             patch.object(prewarm, "targets",
                          return_value=[{"name": "chat", "url": "https://x/m"}]):
            first = prewarm.warm()
            second = prewarm.warm()

        self.assertEqual(first["warmed"], ["chat"])
        self.assertEqual(second["warmed"], [])
        self.assertEqual(second["skipped"], "rate_limited")
        self.assertEqual(calls, ["chat"], "the provider was hit twice")
        self.assertEqual(prewarm.stats()["skipped"], 1)

    def test_force_bypasses_the_window_for_startup(self):
        with patch.object(prewarm, "_warm_one",
                          return_value=("chat", "warm", {})), \
             patch.object(prewarm, "targets",
                          return_value=[{"name": "chat", "url": "https://x/m"}]):
            prewarm.warm()
            forced = prewarm.warm(force=True)
        self.assertEqual(forced["warmed"], ["chat"])

    def test_a_wedged_warm_is_skipped_rather_than_queued(self):
        """A warm in flight must not become a bottleneck of its own."""
        started = threading.Event()
        release = threading.Event()

        def wedged(target):
            started.set()
            release.wait(5)
            return target["name"], "warm", {}

        with patch.object(prewarm, "_warm_one", side_effect=wedged), \
             patch.object(prewarm, "targets",
                          return_value=[{"name": "chat", "url": "https://x/m"}]):
            worker = threading.Thread(target=prewarm.warm, daemon=True)
            worker.start()
            self.assertTrue(started.wait(5))
            began = time.monotonic()
            second = prewarm.warm(force=True)
            elapsed = time.monotonic() - began
            release.set()
            worker.join(5)

        self.assertEqual(second["skipped"], "in_flight")
        self.assertLess(elapsed, 1.0, "the skipped call blocked on the warm")
        self.assertEqual(prewarm.stats()["skipped"], 1)


class PrewarmDegradationTests(unittest.TestCase):
    """Acceptance: a provider outage degrades silently and never raises."""

    def setUp(self):
        prewarm.reset()
        self.addCleanup(prewarm.reset)

    def test_a_dead_provider_returns_ok_with_a_degraded_list(self):
        with patch.object(prewarm, "targets",
                          return_value=[{"name": "chat", "url": "https://x/m"}]), \
             patch.object(prewarm._session, "get",
                          side_effect=OSError("connection refused")):
            report = prewarm.warm(force=True)

        self.assertTrue(report["ok"])
        self.assertEqual(report["warmed"], [])
        self.assertEqual(report["degraded"], ["chat"])
        self.assertEqual(prewarm.stats()["degraded"], 1)

    def test_the_response_body_is_fully_read_so_the_socket_is_reused(self):
        """A half-read response never returns its socket to the pool."""
        response = _FakeResponse()
        with patch.object(prewarm, "targets",
                          return_value=[{"name": "chat", "url": "https://x/m"}]), \
             patch.object(prewarm._session, "get", return_value=response):
            report = prewarm.warm(force=True)

        self.assertEqual(report["warmed"], ["chat"])
        self.assertEqual(response.content_reads, 1,
                         "the warm did not read the body")
        self.assertTrue(response.closed)

    def test_a_warm_uses_the_latency_critical_connect_budget(self):
        response = _FakeResponse()
        with patch.object(prewarm, "targets",
                          return_value=[{"name": "chat", "url": "https://x/m"}]), \
             patch.object(prewarm._session, "get",
                          return_value=response) as get:
            prewarm.warm(force=True)

        timeout = get.call_args.kwargs["timeout"]
        self.assertEqual(timeout[0], prewarm.CONNECT_TIMEOUT_SECONDS)
        self.assertLessEqual(timeout[0], 1.0)
        self.assertGreater(timeout[1], timeout[0])

    def test_stats_are_silent_but_counted(self):
        with patch.object(prewarm, "targets",
                          return_value=[{"name": "chat", "url": "https://x/m"}]), \
             patch.object(prewarm._session, "get",
                          side_effect=OSError("down")):
            prewarm.warm(force=True)

        stats = prewarm.stats()
        self.assertEqual(stats["warms"], 1)
        self.assertEqual(stats["degraded"], 1)
        self.assertEqual(stats["errors"], 0)
        self.assertEqual(stats["targets"]["chat"]["outcome"], "degraded")


class KeepAliveTests(unittest.TestCase):
    """Acceptance: the pooled sockets carry keepalive, modestly."""

    def test_socket_options_enable_keepalive(self):
        options = prewarm.keepalive_socket_options()
        self.assertIn((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), options)

    def test_keepalive_is_not_aggressive(self):
        """Deliberately RFC-1122-era values: nothing that reads as abuse."""
        self.assertGreaterEqual(prewarm.KEEPALIVE_IDLE_SECONDS, 30)
        self.assertGreaterEqual(prewarm.KEEPALIVE_INTERVAL_SECONDS, 10)
        self.assertLessEqual(prewarm.KEEPALIVE_PROBES, 5)

    def test_the_adapter_carries_the_options_into_its_pool(self):
        adapter = prewarm.KeepAliveAdapter()
        self.assertIn((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
                      adapter._socket_options)

    def test_the_provider_sessions_are_keepalive_sessions(self):
        from backend.services import gemini_client, openai_compat_client

        for session in (gemini_client._session, openai_compat_client._session):
            adapter = session.get_adapter("https://api.example.com")
            self.assertIsInstance(adapter, prewarm.KeepAliveAdapter)


class SttPoolingTests(unittest.TestCase):
    """Acceptance: STT rides a pooled session instead of a bare requests.post."""

    def test_transcription_uses_one_pooled_session(self):
        from backend.services import transcription

        self.assertIsInstance(transcription._session, __import__("requests").Session)
        self.assertIsNotNone(transcription._session.get_adapter(
            "https://api.inworld.ai/stt/v1/transcribe"))

    def test_the_stt_connect_budget_is_latency_critical(self):
        from backend.services import transcription

        self.assertLessEqual(transcription.STT_CONNECT_TIMEOUT_SECONDS, 1.0)
        self.assertLessEqual(
            transcription.INWORLD_STT_CONNECT_TIMEOUT_SECONDS, 1.0)

    def test_the_inworld_call_goes_through_the_session(self):
        from backend.services import transcription

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "transcription": {"transcript": "hello"}
        }
        audio = MagicMock()
        audio.get_wav_data.return_value = b"RIFF"
        audio.sample_rate = 16000

        with patch.object(transcription, "INWORLD_STT_API_KEY", "k"), \
             patch.object(transcription._session, "post",
                          return_value=response) as post:
            self.assertEqual(transcription.recognize_inworld(audio), "hello")

        self.assertTrue(post.called)
        self.assertEqual(
            post.call_args.kwargs["timeout"], transcription.INWORLD_STT_TIMEOUT)

    def test_a_cloud_stt_host_is_offered_as_a_prewarm_target(self):
        from backend.services import transcription

        with patch.object(transcription, "INWORLD_STT_API_KEY", "k"):
            target = transcription.prewarm_target()
        self.assertIsNotNone(target)
        self.assertIn("api.inworld.ai", target["url"])

    def test_a_local_only_setup_offers_nothing_to_warm(self):
        from backend.services import transcription

        with patch.object(transcription, "INWORLD_STT_API_KEY", ""):
            self.assertIsNone(transcription.prewarm_target())


class VoiceOnsetPrewarmTests(unittest.TestCase):
    """Acceptance: the voice worker warms on VAD onset, off the capture thread."""

    def setUp(self):
        import backend.voice_mode as voice_mode

        self.voice_mode = voice_mode
        voice_mode._last_prewarm_request_at = 0.0
        self.addCleanup(setattr, voice_mode, "_last_prewarm_request_at", 0.0)

    def test_onset_warm_spawns_a_daemon_thread_and_never_blocks(self):
        posted = threading.Event()
        with patch.object(self.voice_mode, "_request_backend_prewarm",
                          side_effect=lambda: posted.set()):
            began = time.monotonic()
            spawned = self.voice_mode._prewarm_backend_async()
            elapsed = time.monotonic() - began
            self.assertTrue(posted.wait(5))

        self.assertTrue(spawned)
        self.assertLess(elapsed, 0.2, "onset must not do network work inline")

    def test_the_onset_warm_is_rate_limited_locally(self):
        with patch.object(self.voice_mode, "_request_backend_prewarm",
                          side_effect=lambda: None) as post:
            self.assertTrue(self.voice_mode._prewarm_backend_async())
            self.assertFalse(self.voice_mode._prewarm_backend_async())
            time.sleep(0.05)
        self.assertEqual(post.call_count, 1,
                         "an onset storm must not spawn a thread each time")

    def test_the_onset_hook_is_registered_with_the_listener(self):
        from backend.services import listener

        self.assertIn(self.voice_mode._on_speech_onset,
                      listener._barge_in_hooks)

    def test_the_warm_post_carries_the_auth_token(self):
        captured = {}

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                captured["read"] = True
                return b'{"ok": true}'

        def fake_urlopen(request, timeout=None):
            captured["headers"] = dict(request.headers)
            captured["url"] = request.full_url
            return _Response()

        with patch.object(self.voice_mode, "urlopen", fake_urlopen), \
             patch.object(self.voice_mode, "_backend_headers",
                          return_value={"X-Jarvis-Token": "tok"}), \
             patch("backend.services.local_auth.auth_headers",
                   return_value={"X-Jarvis-Token": "tok"}):
            self.assertTrue(self.voice_mode._request_backend_prewarm())

        self.assertTrue(captured["url"].endswith("/prewarm"))
        self.assertEqual(
            {k.lower(): v for k, v in captured["headers"].items()}
            .get("x-jarvis-token"), "tok")
        self.assertTrue(captured.get("read"), "the body must be drained")

    def test_a_backend_that_is_down_is_silent(self):
        with patch.object(self.voice_mode, "urlopen",
                          side_effect=OSError("connection refused")):
            self.assertFalse(self.voice_mode._request_backend_prewarm())


class PrewarmTargetTests(unittest.TestCase):
    """The chat provider and the classifier ladder are the hot sessions."""

    def test_the_chat_provider_is_a_target(self):
        targets = prewarm._classifier_targets()
        for target in targets:
            self.assertIn(target["name"], ("gemini", "groq"))
            self.assertTrue(target["url"].startswith("https://"))

    def test_targets_are_deduped_by_url(self):
        duplicate = {"name": "a", "url": "https://same/models"}
        with patch.object(prewarm, "_chat_target", return_value=duplicate), \
             patch.object(prewarm, "_classifier_targets",
                          return_value=[dict(duplicate)]), \
             patch("backend.services.transcription.prewarm_target",
                   return_value=None):
            found = prewarm.targets()
        self.assertEqual(len(found), 1)

    def test_a_provider_without_a_cheap_get_is_skipped(self):
        """Warming is a real GET, never a fake request to look busy."""
        with patch("backend.services.model_registry.get_model_for_role",
                   return_value=("whisper", "local")), \
             patch("backend.services.model_registry.get_provider_credentials",
                   return_value=(None, None)):
            self.assertIsNone(prewarm._chat_target())


if __name__ == "__main__":
    unittest.main()
