import json
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
