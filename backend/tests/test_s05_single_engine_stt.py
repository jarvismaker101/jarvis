"""[S5] One STT engine, one pass, its output is final.

The user-selected transcription model is the ONLY model that runs on the
CONVERSATION path (``listener.recognize_multilingual``). No cross-engine
ladder, no racing, and no second pass that could substitute another model's
text for the selected one.

[LF36] The WAKE path (``watcher.recognize_candidates``) is deliberately NOT
the selection: it runs the sleeping loop continuously over every ambient
sound, so it is LOCAL-ONLY (whisper) and holds no cloud transcriber at all.
A cloud model picked in settings can never turn that loop into an uploader;
the selection governs the conversation path, which runs per turn, after the
user asked to be heard.
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
    """The CONVERSATION path resolves the settings engine (LF36 split)."""

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

    def test_the_wake_path_never_resolves_the_settings_engine(self):
        # [LF36] The wake path is local-only: even with a cloud model selected
        # in settings (and the policy on), it must not consult the selection.
        # The patch below turns any such resolution into a loud failure.
        with patch.object(listener, "_engine_for_listening_role",
                          return_value="sarvam"), \
             patch.object(listener, "_cloud_stt_policy", return_value="on"), \
             patch.object(listener, "selected_stt_engine",
                          side_effect=AssertionError(
                              "the wake path resolved the settings engine")), \
             patch.object(watcher, "whisper_daemon_ok", True), \
             patch.object(watcher, "whisper_model", None), \
             patch.object(watcher, "_transcribe_with_daemon",
                          return_value=("jarvis play music", {})):
            candidates, wake = watcher.recognize_candidates(_audio())
        self.assertEqual(wake, "jarvis play music")
        self.assertEqual(candidates, ["jarvis play music"])

    def test_a_settings_failure_cannot_break_the_wake_path(self):
        # A registry hiccup used to be resolved (and degraded) on this path;
        # it no longer resolves anything, so the local model keeps working.
        with patch.object(listener, "_engine_for_listening_role",
                          side_effect=RuntimeError("registry down")), \
             patch.object(watcher, "whisper_daemon_ok", True), \
             patch.object(watcher, "whisper_model", None), \
             patch.object(watcher, "_transcribe_with_daemon",
                          return_value=("Jarvis, open Chrome", {})):
            candidates, wake = watcher.recognize_candidates(_audio())
        self.assertEqual(wake, "Jarvis, open Chrome")
        self.assertEqual(candidates, ["Jarvis, open Chrome"])

    def test_the_watcher_holds_no_cloud_transcriber(self):
        # [LF36] The strongest form of "no upload from the sleeping loop":
        # there is no cloud engine in the module that could be called.
        for name in ("recognize_sarvam", "recognize_inworld",
                     "recognize_google_or_groq"):
            self.assertFalse(hasattr(watcher, name),
                             "watcher can still reach %s" % name)


class WakePathLocalOnlyTests(unittest.TestCase):
    """[LF36] The wake path asks the LOCAL model ONCE and stops."""

    def setUp(self):
        self.daemon = patch.object(watcher, "_transcribe_with_daemon")
        self.in_process = patch.object(watcher, "whisper_model", None)
        self.daemon_ok = patch.object(watcher, "whisper_daemon_ok", True)
        self.mock_daemon = self.daemon.start()
        self.in_process.start()
        self.daemon_ok.start()
        for p in (self.daemon, self.in_process, self.daemon_ok):
            self.addCleanup(p.stop)

    def test_the_wake_path_is_local(self):
        self.mock_daemon.return_value = ("Jarvis, play music", {})
        candidates, wake = watcher.recognize_candidates(_audio())
        self.mock_daemon.assert_called_once()
        self.assertEqual(wake, "Jarvis, play music")
        self.assertEqual(candidates, ["Jarvis, play music"])

    def test_the_in_process_model_is_a_transport_fallback_not_a_second_engine(self):
        # Same model (local whisper), asked through the in-process model
        # because the daemon could not answer - and still no other engine.
        model = _FakeWhisperModel("jarvis play music")
        with patch.object(watcher, "whisper_model", model):
            self.mock_daemon.return_value = None
            candidates, wake = watcher.recognize_candidates(_audio())
        self.assertEqual(model.calls, 1)
        self.assertEqual(wake, "jarvis play music")

    def test_an_exhausted_local_model_yields_no_transcript_at_all(self):
        self.mock_daemon.return_value = None
        candidates, wake = watcher.recognize_candidates(_audio())
        self.assertEqual(candidates, [])
        self.assertIsNone(wake)

    def test_the_conversation_path_still_uses_the_cloud_selection(self):
        # [LF36] The user asked for exactly this split: wake = local whisper,
        # conversation = the selected model (sarvam in the live settings).
        # This pins the conversation half so the wake fix cannot break it.
        with patch.object(listener, "_engine_for_listening_role",
                          return_value="sarvam"), \
             patch.object(listener, "_cloud_stt_policy", return_value="on"):
            self.assertEqual(listener.selected_stt_engine(), "sarvam")


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