"""[S5] One STT engine, one pass, its output is final.

The user-selected transcription model is the ONLY model that runs, on both the
wake path (``watcher.recognize_candidates``) and the conversation path
(``listener.recognize_multilingual``). No cross-engine ladder, no racing, and no
second pass that could substitute another model's text for the selected one.
"""

import unittest
from unittest.mock import patch

import speech_recognition as sr

from backend import watcher
from backend.services import listener


def _audio(seconds=0.5, rate=16000):
    return sr.AudioData(b"\x00" * int(rate * seconds) * 2, rate, 2)


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _FakeInfo:
    language = "en"
    language_probability = 0.99


class _FakeWhisperModel:
    def __init__(self, text="in process answer"):
        self.text = text
        self.calls = 0

    def transcribe(self, stream, **kwargs):
        self.calls += 1
        return iter([_FakeSegment(self.text)]), _FakeInfo()


class EngineResolutionTests(unittest.TestCase):
    """Both paths resolve the SAME engine, from settings + cloud policy."""

    def test_resolution_follows_the_settings_selection(self):
        for provider in ("whisper", "inworld", "google-or-groq"):
            with patch.object(listener, "_engine_for_listening_role",
                              return_value=provider), \
                 patch.object(listener, "_cloud_stt_policy", return_value="on"):
                self.assertEqual(listener.selected_stt_engine(), provider)

    def test_a_local_only_policy_clamps_a_cloud_selection(self):
        with patch.object(listener, "_engine_for_listening_role",
                          return_value="inworld"), \
             patch.object(listener, "_cloud_stt_policy", return_value="off"):
            self.assertEqual(listener.selected_stt_engine(), "whisper")

    def test_the_wake_path_uses_the_shared_resolution(self):
        seen = {}

        def _resolve():
            seen["called"] = True
            return "google-or-groq"

        with patch.object(listener, "selected_stt_engine", _resolve), \
             patch.object(watcher, "recognize_google_or_groq",
                          return_value="jarvis play music"), \
             patch.object(watcher, "_transcribe_with_daemon") as daemon:
            candidates, wake = watcher.recognize_candidates(_audio())
        self.assertTrue(seen.get("called"), "the wake path resolved its own engine")
        daemon.assert_not_called()
        self.assertEqual(wake, "jarvis play music")
        self.assertEqual(candidates, ["jarvis play music"])

    def test_resolution_failure_degrades_to_local_never_to_cloud(self):
        # A registry hiccup must never turn into "try the cloud instead".
        with patch.object(listener, "selected_stt_engine",
                          side_effect=RuntimeError("registry down")):
            self.assertEqual(watcher._selected_engine(), "whisper")


class WakePathSingleEngineTests(unittest.TestCase):
    """The wake path asks the selected engine ONCE and stops."""

    def setUp(self):
        self.daemon = patch.object(watcher, "_transcribe_with_daemon")
        self.in_process = patch.object(watcher, "whisper_model", None)
        self.daemon_ok = patch.object(watcher, "whisper_daemon_ok", True)
        self.online = patch.object(watcher, "recognize_google_or_groq")
        self.inworld = patch.object(watcher, "recognize_inworld")
        self.mock_daemon = self.daemon.start()
        self.in_process.start()
        self.daemon_ok.start()
        self.mock_online = self.online.start()
        self.mock_inworld = self.inworld.start()
        for p in (self.daemon, self.in_process, self.daemon_ok,
                  self.online, self.inworld):
            self.addCleanup(p.stop)

    def test_a_selected_local_engine_never_reaches_the_cloud(self):
        with patch.object(watcher, "_selected_engine", return_value="whisper"):
            self.mock_daemon.return_value = ("Jarvis, play music", {})
            candidates, wake = watcher.recognize_candidates(_audio())
        self.mock_daemon.assert_called_once()
        self.assertEqual(wake, "Jarvis, play music")
        self.mock_online.assert_not_called()
        self.mock_inworld.assert_not_called()

    def test_the_in_process_model_is_a_transport_fallback_not_a_second_engine(self):
        # Same selected model (local whisper), asked through the in-process
        # model because the daemon could not answer - and STILL no cloud call.
        model = _FakeWhisperModel("jarvis play music")
        with patch.object(watcher, "_selected_engine", return_value="whisper"), \
             patch.object(watcher, "whisper_model", model):
            self.mock_daemon.return_value = None
            candidates, wake = watcher.recognize_candidates(_audio())
        self.assertEqual(model.calls, 1)
        self.assertEqual(wake, "jarvis play music")
        self.mock_online.assert_not_called()
        self.mock_inworld.assert_not_called()

    def test_an_exhausted_local_engine_yields_no_transcript_at_all(self):
        with patch.object(watcher, "_selected_engine", return_value="whisper"):
            self.mock_daemon.return_value = None
            candidates, wake = watcher.recognize_candidates(_audio())
        self.assertEqual(candidates, [])
        self.assertIsNone(wake)
        self.mock_online.assert_not_called()
        self.mock_inworld.assert_not_called()

    def test_a_selected_cloud_engine_asks_that_engine_exactly_once(self):
        self.mock_inworld.return_value = "jarvis hello"
        with patch.object(watcher, "_selected_engine", return_value="inworld"):
            candidates, wake = watcher.recognize_candidates(_audio())
        self.mock_inworld.assert_called_once()
        self.mock_daemon.assert_not_called()
        self.mock_online.assert_not_called()
        self.assertEqual(wake, "jarvis hello")

    def test_the_online_engine_is_asked_once_per_utterance_not_twice(self):
        # The old loop called recognize_google_or_groq with show_all=True and
        # then AGAIN for the same language when that came back empty, once per
        # configured language. One utterance = one call now.
        self.mock_online.return_value = "jarvis hello"
        with patch.object(watcher, "_selected_engine",
                          return_value="google-or-groq"):
            candidates, wake = watcher.recognize_candidates(_audio())
        self.assertEqual(self.mock_online.call_count, 1)
        self.assertEqual(wake, "jarvis hello")
        self.mock_daemon.assert_not_called()
        self.mock_inworld.assert_not_called()

    def test_a_cloud_failure_is_a_failed_turn_not_a_local_retry(self):
        self.mock_inworld.side_effect = sr.UnknownValueError()
        with patch.object(watcher, "_selected_engine", return_value="inworld"):
            candidates, wake = watcher.recognize_candidates(_audio())
        self.mock_inworld.assert_called_once()
        self.mock_daemon.assert_not_called()
        self.assertEqual(candidates, [])
        self.assertIsNone(wake)


class ConversationPathSingleEngineTests(unittest.TestCase):
    """The conversation path keeps exactly one engine, and it is the same one."""

    def test_a_failed_selected_engine_never_consults_another(self):
        with patch.object(listener, "_engine_for_listening_role",
                          return_value="inworld"), \
             patch.object(listener, "_cloud_stt_policy", return_value="on"), \
             patch.object(listener, "recognize_inworld",
                          side_effect=sr.RequestError("down")), \
             patch.object(listener, "recognize_local_whisper") as local, \
             patch.object(listener, "recognize_google_or_groq") as online:
            result = listener.recognize_multilingual(_audio())
        self.assertEqual(result, (None, None, None))
        local.assert_not_called()
        online.assert_not_called()
        self.assertEqual(listener.LAST_STT_FAILURE[0], "inworld")


class NoLadderLeftTests(unittest.TestCase):
    """The removed ladder must not come back by accident."""

    def test_recognize_candidates_holds_no_second_engine_pass(self):
        import inspect

        source = inspect.getsource(watcher.recognize_candidates)
        self.assertNotIn("for language in", source)
        self.assertNotIn("show_all", source)

    def test_the_local_helper_is_the_only_place_the_daemon_is_called(self):
        import inspect

        helper = inspect.getsource(watcher._transcribe_local_once)
        candidates = inspect.getsource(watcher.recognize_candidates)
        self.assertIn("_transcribe_with_daemon", helper)
        self.assertNotIn("_transcribe_with_daemon", candidates)


if __name__ == "__main__":
    unittest.main()