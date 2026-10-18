import json
import threading
import unittest
from unittest.mock import MagicMock, patch

from backend import listener_state
from backend import voice_mode
from backend.api import routes


class VoiceModeToggleTests(unittest.TestCase):
    def setUp(self):
        # ensure default state before each test
        listener_state.set_voice_input_enabled(True)

    def tearDown(self):
        listener_state.set_voice_input_enabled(True)

    def test_flag_defaults_true(self):
        self.assertTrue(listener_state.is_voice_input_enabled())
        state = listener_state.get_voice_state()
        self.assertTrue(state.get("voice_input_enabled"))
        # also via routes get_ui_state
        ui = routes.get_ui_state()
        self.assertTrue(ui["state"].get("voice_input_enabled"))
        self.assertTrue(ui.get("voice_input_enabled"))

    def test_set_flips_and_exposes_via_state_and_ui(self):
        listener_state.set_voice_input_enabled(False)
        self.assertFalse(listener_state.is_voice_input_enabled())
        state = listener_state.get_voice_state()
        self.assertFalse(state.get("voice_input_enabled"))
        ui = routes.get_ui_state()
        self.assertFalse(ui["state"].get("voice_input_enabled"))
        self.assertFalse(ui.get("voice_input_enabled"))
        voice_state = routes.get_voice_state()
        self.assertFalse(voice_state.get("voice_input_enabled"))
        # also check GET /voice-mode helper
        resp = routes.get_voice_mode()
        self.assertFalse(resp.get("voice_input_enabled"))

    def test_post_voice_mode_round_trip(self):
        # direct route function call mirrors TestClient pattern without needing HTTP server
        payload = routes.VoiceModeUpdate(enabled=False)
        resp = routes.set_voice_mode(payload)
        self.assertEqual(resp, {"voice_input_enabled": False})
        self.assertFalse(listener_state.is_voice_input_enabled())
        # flip via POST again
        payload2 = routes.VoiceModeUpdate(enabled=True)
        resp2 = routes.set_voice_mode(payload2)
        self.assertEqual(resp2, {"voice_input_enabled": True})
        self.assertTrue(listener_state.is_voice_input_enabled())
        ui = routes.get_ui_state()
        self.assertTrue(ui["state"].get("voice_input_enabled"))

    def test_toggling_true_restores(self):
        listener_state.set_voice_input_enabled(False)
        self.assertFalse(listener_state.is_voice_input_enabled())
        listener_state.set_voice_input_enabled(True)
        self.assertTrue(listener_state.is_voice_input_enabled())
        state = listener_state.get_voice_state()
        self.assertTrue(state.get("voice_input_enabled"))
        ui = routes.get_ui_state()
        self.assertTrue(ui["state"].get("voice_input_enabled"))
        self.assertTrue(ui.get("voice_input_enabled"))


class VoiceFlagHttpPollTests(unittest.TestCase):
    """Contract: the voice PROCESS (separate OS process) consumes the
    text-mode flag via HTTP GET /voice-mode, never via the module import
    (its imported listener_state copy can never see API-side toggles).
    """

    def setUp(self):
        # reset the module-level poll cache before/after each test so
        # cached state never leaks between tests
        voice_mode._voice_flag_last_known = True
        voice_mode._voice_flag_checked_at = 0.0

    def tearDown(self):
        voice_mode._voice_flag_last_known = True
        voice_mode._voice_flag_checked_at = 0.0
        listener_state.set_voice_input_enabled(True)

    @staticmethod
    def _fake_urlopen(body):
        class _FakeResponse:
            def __init__(self, payload):
                self._payload = payload

            def read(self):
                return json.dumps(self._payload).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return lambda url, timeout=None: _FakeResponse(body)

    def test_flag_comes_from_http_not_module_import(self):
        # API side (this process) flips the flag to False, but the HTTP
        # endpoint reports True: the voice process must see True - the
        # HTTP value wins over the stale local listener_state copy.
        listener_state.set_voice_input_enabled(False)
        with patch.object(voice_mode, "urlopen",
                         side_effect=self._fake_urlopen(
                             {"voice_input_enabled": True})):
            self.assertTrue(voice_mode.voice_input_enabled())
        # and the inverse: HTTP says False while the local copy says True
        listener_state.set_voice_input_enabled(True)
        voice_mode._voice_flag_checked_at = 0.0  # force a fresh poll
        with patch.object(voice_mode, "urlopen",
                         side_effect=self._fake_urlopen(
                             {"voice_input_enabled": False})):
            self.assertFalse(voice_mode.voice_input_enabled())

    def test_flag_polls_backend_voice_mode_url_with_short_timeout(self):
        seen = {}

        def fake_urlopen(url, timeout=None):
            # F51: the poll is authenticated now, so it passes a Request
            # object (headers + URL) instead of a bare URL string.
            seen["url"] = getattr(url, "full_url", url)
            seen["request"] = url
            seen["timeout"] = timeout
            return self._fake_urlopen({"voice_input_enabled": True})(url)

        with patch.object(voice_mode, "urlopen", side_effect=fake_urlopen):
            voice_mode.voice_input_enabled()
        self.assertEqual(seen["url"],
                         f"http://127.0.0.1:{voice_mode.BACKEND_PORT}/voice-mode")
        self.assertLess(seen["timeout"], 2.0)

    def test_flag_cached_within_cadence_and_failure_keeps_last_known(self):
        calls = []

        def fake_urlopen(url, timeout=None):
            calls.append(url)
            return self._fake_urlopen({"voice_input_enabled": False})(url)

        with patch.object(voice_mode, "urlopen", side_effect=fake_urlopen), \
             patch.object(voice_mode.time, "monotonic",
                          side_effect=[100.0, 100.5, 101.5]):
            self.assertFalse(voice_mode.voice_input_enabled())  # polls
            self.assertFalse(voice_mode.voice_input_enabled())  # cached, no poll
            self.assertFalse(voice_mode.voice_input_enabled())  # cadence elapsed, polls
        self.assertEqual(len(calls), 2)

        # request failure keeps the last known value (False)
        def broken_urlopen(url, timeout=None):
            raise OSError("backend down")

        with patch.object(voice_mode, "urlopen", side_effect=broken_urlopen), \
             patch.object(voice_mode.time, "monotonic",
                          side_effect=[500.0, 500.0]):
            self.assertFalse(voice_mode.voice_input_enabled())
        self.assertEqual(len(calls), 2)  # broken urlopen never succeeded


class ShutdownGrammarTests(unittest.TestCase):
    """F35: is_shutdown is exact and negation-aware. Only explicit shutdown
    phrases terminate the stack; stop speaking / stop task / cancel approval
    / pause — with or without the Jarvis name — never do."""

    def test_explicit_shutdown_phrases_true(self):
        for phrase in (
            "jarvis shutdown",
            "jarvis shut down",
            "shutdown",
            "shut down",
            "shutdown jarvis",
            "jarvis stop listening",
            "stop listening",
        ):
            self.assertTrue(voice_mode.is_shutdown(phrase), phrase)

    def test_stop_speaking_never_shutdown(self):
        for phrase in (
            "stop speaking",
            "jarvis stop speaking",
            "jarvis, stop speaking",
            "stop talking",
            "jarvis stop talking",
            "be quiet",
            "shut up",
            "silence",
        ):
            self.assertFalse(voice_mode.is_shutdown(phrase), phrase)

    def test_stop_task_cancel_approval_pause_never_shutdown(self):
        for phrase in (
            "stop task",
            "jarvis stop task",
            "stop the task",
            "cancel approval",
            "jarvis cancel approval",
            "pause",
            "jarvis pause",
        ):
            self.assertFalse(voice_mode.is_shutdown(phrase), phrase)

    def test_hindi_shutdown_true(self):
        for phrase in ("band ho jao", "jarvis band ho jao", "sab band karo"):
            self.assertTrue(voice_mode.is_shutdown(phrase), phrase)

    def test_hindi_stop_talking_not_shutdown(self):
        for phrase in ("chup ho jao", "jarvis chup ho jao", "bolna band karo"):
            self.assertFalse(voice_mode.is_shutdown(phrase), phrase)

    def test_bare_stop_or_empty_not_shutdown(self):
        for phrase in ("", "   ", "jarvis stop", "stop"):
            self.assertFalse(voice_mode.is_shutdown(phrase), phrase)


class PublishedStateAuthorityTests(unittest.TestCase):
    """[LF36] The toggle is owned by the API; a worker's mirror cannot undo it.

    Live bug: clicking the toggle sent ``POST /voice-mode`` (flag False) and
    the next ``/ui-state`` poll returned True again, because the voice
    worker's published snapshot carried its stale import-time default and the
    merge let that snapshot override the authoritative copy. The UI flipped
    back to "Voice: On" within one poll and the switch looked broken.
    """

    def setUp(self):
        self._reset()
        self.addCleanup(self._reset)

    @staticmethod
    def _reset():
        routes._published_voice.clear()
        routes._published_voice_seq = 0
        routes._published_voice_publisher = ""
        listener_state.set_voice_input_enabled(True)

    def publish(self, **payload):
        payload.setdefault("status", "listening")
        return routes.publish_voice_state(dict(payload))

    def test_a_stale_worker_flag_cannot_flip_the_ui_flag_back(self):
        listener_state.set_voice_input_enabled(False)

        self.publish(publisher_id="worker-a", state_seq=1,
                     voice_input_enabled=True)

        self.assertFalse(routes.get_voice_state()["voice_input_enabled"])
        ui = routes.get_ui_state()
        self.assertFalse(ui["state"]["voice_input_enabled"])
        self.assertFalse(ui["voice_input_enabled"])

    def test_the_worker_still_owns_the_other_published_fields(self):
        # Only the flag is control-plane state; the worker's real listening
        # state must keep overriding this process's empty module copy.
        listener_state.set_voice_input_enabled(False)
        self.publish(publisher_id="worker-b", state_seq=1, status="speaking")
        self.assertEqual(routes.get_ui_state()["state"]["status"], "speaking")


class WorkerFlagMirrorTests(unittest.TestCase):
    """[LF36] The worker's PUBLISHED state must carry the flag it learned."""

    def setUp(self):
        self._restore()
        self.addCleanup(self._restore)

    @staticmethod
    def _restore():
        listener_state.set_voice_input_enabled(True)
        voice_mode._voice_flag_last_known = True
        voice_mode._voice_flag_checked_at = 0.0

    def test_a_pushed_off_event_updates_the_workers_state_copy(self):
        voice_mode._apply_pushed_state({"voice_input_enabled": False})
        self.assertFalse(voice_mode._voice_flag_last_known)
        self.assertFalse(listener_state.is_voice_input_enabled())
        self.assertFalse(
            listener_state.get_voice_state()["voice_input_enabled"])

    def test_a_polled_off_answer_updates_the_workers_state_copy(self):
        class _FakeResponse:
            def read(self):
                return b'{"voice_input_enabled": false}'

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with patch.object(
                voice_mode, "urlopen",
                side_effect=lambda url, timeout=None: _FakeResponse()):
            self.assertFalse(voice_mode.voice_input_enabled())
        self.assertFalse(listener_state.is_voice_input_enabled())
        self.assertFalse(
            listener_state.get_voice_state()["voice_input_enabled"])


_STOP_SENTINEL = "\x00lf36-stop"


def _once_then_shutdown(text):
    """Serve *text* on the first capture, then end the listener thread."""
    served = {"done": False}

    def _impl(marks=None, **_kwargs):
        if not served["done"]:
            served["done"] = True
            return text
        return _STOP_SENTINEL

    return _impl


def _classify_passthrough():
    real = voice_mode.classify_control

    def _impl(text):
        if text == _STOP_SENTINEL:
            return "shutdown"
        return real(text)

    return _impl


class TextModeCaptureTests(unittest.TestCase):
    """[LF36] A capture already in flight when the toggle lands is not a turn."""

    def setUp(self):
        while True:
            try:
                voice_mode.command_queue.get_nowait()
            except Exception:
                break

    def test_an_in_flight_capture_is_dropped_when_voice_turns_off(self):
        state = MagicMock()
        state.is_speaking.return_value = False
        calls = {"n": 0}

        def flag():
            calls["n"] += 1
            # ON at the loop top, OFF for the post-capture re-check, then ON
            # again so the thread reaches the shutdown sentinel and ends.
            return calls["n"] != 2

        with patch.object(voice_mode, "voice_input_enabled", side_effect=flag), \
                patch.object(voice_mode, "listener_state", state), \
                patch.object(voice_mode, "classify_control",
                             side_effect=_classify_passthrough()), \
                patch.object(voice_mode, "shutdown_everything"), \
                patch.object(voice_mode, "_close_microphone_source") as close_mic, \
                patch.object(voice_mode, "_latency", MagicMock()), \
                patch.object(voice_mode, "listen",
                             side_effect=_once_then_shutdown("open chrome")):
            thread = threading.Thread(target=voice_mode.listener_thread,
                                      daemon=True)
            thread.start()
            thread.join(timeout=3)

        self.assertTrue(close_mic.called,
                        "the mic was not closed when the toggle landed")
        self.assertTrue(
            voice_mode.command_queue.empty(),
            "an utterance captured before the toggle became a turn")

    def test_an_utterance_with_voice_on_is_still_submitted(self):
        state = MagicMock()
        state.is_speaking.return_value = False

        with patch.object(voice_mode, "voice_input_enabled", return_value=True), \
                patch.object(voice_mode, "listener_state", state), \
                patch.object(voice_mode, "classify_control",
                             side_effect=_classify_passthrough()), \
                patch.object(voice_mode, "shutdown_everything"), \
                patch.object(voice_mode, "backend_task_running",
                             return_value=False), \
                patch.object(voice_mode, "_latency", MagicMock()), \
                patch.object(voice_mode, "listen",
                             side_effect=_once_then_shutdown("open chrome")):
            thread = threading.Thread(target=voice_mode.listener_thread,
                                      daemon=True)
            thread.start()
            try:
                item = voice_mode.command_queue.get(timeout=3)
            except Exception:
                item = None
            thread.join(timeout=3)

        self.assertIsNotNone(item)
        self.assertEqual(item[0], "open chrome")


class TextModeSpeechTests(unittest.TestCase):
    """[LF36] Voice OFF = text chat only: the backend does not talk back.

    The toggle is a voice/text MODE switch, not only a mic mute: with voice
    off a typed request must be answered as text (no TTS), and narration must
    stay silent - otherwise "text chat only" still talks out loud.
    """

    def setUp(self):
        listener_state.set_voice_input_enabled(True)
        self.addCleanup(listener_state.set_voice_input_enabled, True)

    def test_a_typed_request_is_spoken_only_while_voice_mode_is_on(self):
        self.assertTrue(routes._typed_speech_allowed(True))
        listener_state.set_voice_input_enabled(False)
        self.assertFalse(routes._typed_speech_allowed(True))

    def test_a_voice_worker_submission_stays_silent_either_way(self):
        self.assertFalse(routes._typed_speech_allowed(False))
        listener_state.set_voice_input_enabled(True)
        self.assertFalse(routes._typed_speech_allowed(False))

    def test_speak_is_suppressed_in_text_mode(self):
        from backend.services import voice

        listener_state.set_voice_input_enabled(False)
        with patch.object(voice, "threading") as fake_threading:
            voice.speak("this reply must stay text")
        fake_threading.Thread.assert_not_called()
        self.assertFalse(listener_state.is_speaking())

    def test_speak_still_runs_when_voice_mode_is_on(self):
        from backend.services import voice

        with patch.object(voice, "threading") as fake_threading:
            voice.speak("this reply is spoken")
        fake_threading.Thread.assert_called_once()


if __name__ == "__main__":
    unittest.main()
