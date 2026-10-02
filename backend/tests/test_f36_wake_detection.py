"""F36 — separate wake detection from full transcription.

Acceptance (audit report): "Disabled cloud policy produces zero online calls;
wake-only phrases create no task; literal command tails execute once despite
launch/disconnection failures."

The baseline defects pinned here:
  * keyword spotting was never wired to a watcher decision;
  * non-wake speech reached the online STT regardless of policy;
  * the pre-roll ring was fed only AFTER a wake match, so an oversized chunk
    could empty it;
  * verification handed the local daemon an ``sr.AudioData`` where it expects
    WAV BYTES, and then PROCEEDED after the resulting error;
  * shortest-span matching extracted wake-only tails ("jarvis wake up" ->
    "wake up") as commands;
  * forwarding was unauthenticated, unidentified and readiness-blind.

Everything here mocks the hardware/process/HTTP boundary: no microphone is
opened and no HTTP request leaves the machine.
"""

import json
import os
import unittest
from unittest.mock import patch
from urllib.request import Request

import speech_recognition as sr

from backend import watcher
from backend.services import wake_engine


def _audio(seconds=0.5, rate=16000):
    return sr.AudioData(b"\x00" * int(rate * seconds) * 2, rate, 2)


class CloudPolicyTests(unittest.TestCase):
    """Acceptance: disabled cloud policy produces zero online calls."""

    def setUp(self):
        self.whisper_ok = patch.object(watcher, "whisper_daemon_ok", True)
        self.whisper_model = patch.object(watcher, "whisper_model", None)
        self.online = patch.object(watcher, "recognize_google_or_groq")
        self.daemon = patch.object(watcher, "_transcribe_with_daemon")
        self.mock_whisper_ok = self.whisper_ok.start()
        self.whisper_model.start()
        self.mock_online = self.online.start()
        self.mock_daemon = self.daemon.start()
        for p in (self.whisper_ok, self.whisper_model, self.online, self.daemon):
            self.addCleanup(p.stop)
        os.environ.pop("JARVIS_STT_CLOUD_POLICY", None)
        os.environ.pop("JARVIS_STT_LOCAL_ONLY", None)
        self.addCleanup(os.environ.pop, "JARVIS_STT_CLOUD_POLICY", None)
        self.addCleanup(os.environ.pop, "JARVIS_STT_LOCAL_ONLY", None)

    def test_policy_off_makes_zero_online_calls(self):
        os.environ["JARVIS_STT_CLOUD_POLICY"] = "off"
        self.mock_daemon.return_value = ("just some background chatter", {})
        candidates, wake_match = watcher.recognize_candidates(_audio())
        self.assertIsNone(wake_match)
        self.assertEqual(candidates, ["just some background chatter"])
        self.mock_online.assert_not_called()

    def test_policy_off_keeps_local_wake_detection_working(self):
        os.environ["JARVIS_STT_CLOUD_POLICY"] = "off"
        self.mock_daemon.return_value = ("Jarvis, open Chrome", {})
        candidates, wake_match = watcher.recognize_candidates(_audio())
        self.assertEqual(wake_match, "Jarvis, open Chrome")
        self.assertEqual(candidates, ["Jarvis, open Chrome"])
        self.mock_online.assert_not_called()

    def test_local_only_shorthand_is_honoured(self):
        os.environ["JARVIS_STT_LOCAL_ONLY"] = "1"
        self.assertFalse(wake_engine.cloud_stt_allowed())
        self.assertEqual(wake_engine.cloud_stt_policy(), "off")

    def test_default_policy_still_allows_a_cloud_selection(self):
        # [S5] The policy gates whether a cloud engine MAY be used; it does not
        # decide WHICH engine runs. With a cloud engine selected, that engine is
        # the one used.
        self.assertTrue(wake_engine.cloud_stt_allowed())

    def test_a_local_selection_is_never_followed_by_an_online_engine(self):
        # [S5] The old ladder asked the online engine when the daemon came back
        # empty, so the transcript that produced the wake verdict could come
        # from a model the user never selected. A failed SELECTED engine is now
        # a failed turn: no candidates, and zero online calls.
        with patch.object(watcher, "_selected_engine", return_value="whisper"), \
             patch.object(watcher, "recognize_inworld") as inworld:
            self.mock_daemon.return_value = None
            candidates, wake_match = watcher.recognize_candidates(_audio())
        self.assertEqual(candidates, [])
        self.assertIsNone(wake_match)
        self.mock_online.assert_not_called()
        inworld.assert_not_called()

    def test_the_selected_cloud_engine_is_the_only_engine_asked(self):
        # [S5] ...and symmetrically: a selected cloud engine is not preceded by
        # a local guess, so the transcript IS that engine's output.
        with patch.object(watcher, "_selected_engine", return_value="inworld"), \
             patch.object(watcher, "recognize_inworld",
                          return_value="hello there") as inworld:
            candidates, wake = watcher.recognize_candidates(_audio())
        inworld.assert_called_once()
        self.mock_daemon.assert_not_called()
        self.mock_online.assert_not_called()
        self.assertEqual(candidates, ["hello there"])
        self.assertIsNone(wake)


class WakeOnlyPhraseTests(unittest.TestCase):
    """Acceptance: wake-only phrases create no task."""

    def test_wake_only_phrases_yield_no_command(self):
        for phrase in ("wake up jarvis", "jarvis wake up", "jarvis chalu ho",
                       "utho jarvis", "hey jarvis", "jarvis"):
            self.assertEqual(wake_engine.extract_command(phrase), "",
                             "wake-only phrase produced a task: %r" % phrase)

    def test_a_real_tail_is_kept_whole(self):
        self.assertEqual(
            wake_engine.extract_command("wake up jarvis and search for cats"),
            "and search for cats")
        self.assertEqual(
            wake_engine.extract_command("Jarvis, open Chrome"), "open Chrome")

    def test_the_longest_candidate_still_wins(self):
        self.assertEqual(
            wake_engine.extract_command(
                "wake up jarvis",
                candidates=["wake up jarvis", "wake up jarvis open chrome"]),
            "open chrome")

    def test_a_command_verb_that_looks_like_a_wake_word_is_kept(self):
        # 'start'/'listen'/'activate' are wake vocabulary AND ordinary
        # command verbs: after the name they belong to the command.
        self.assertEqual(
            wake_engine.extract_command("Jarvis, start the timer"),
            "start the timer")
        self.assertEqual(
            wake_engine.extract_command("Jarvis, listen to this song"),
            "listen to this song")

    def test_a_multiword_wake_lead_in_is_consumed_not_forwarded(self):
        self.assertEqual(
            wake_engine.extract_command("jarvis wake up and search for cats"),
            "and search for cats")

    def test_command_extraction_never_raises(self):
        self.assertEqual(wake_engine.extract_command(None, None), "")
        self.assertEqual(wake_engine.extract_command("", ["", "  "]), "")


class PreRollTests(unittest.TestCase):
    """The pre-roll is continuous candidate capture: fed BEFORE matching."""

    def setUp(self):
        wake_engine.reset_pre_roll()
        self.addCleanup(wake_engine.reset_pre_roll)
        self.seen_during_match = None

    def test_pre_roll_is_fed_before_the_wake_decision(self):
        def fake_recognize(audio):
            self.seen_during_match = len(wake_engine.pre_roll())
            return [], None

        with patch.object(watcher, "recognize_candidates",
                          side_effect=fake_recognize), \
             patch.object(watcher, "open_microphone") as mic, \
             patch.object(watcher.recognizer, "listen",
                          return_value=_audio(1.0)), \
             patch.object(watcher, "recalibrate_watcher"), \
             patch.object(wake_engine, "keyword_spot", return_value=False):
            mic.return_value.__enter__.return_value = object()
            watcher.listen_once()
        self.assertIsNotNone(self.seen_during_match)
        self.assertGreater(self.seen_during_match, 0,
                           "pre-roll was empty when the wake decision ran")


class VerificationTests(unittest.TestCase):
    """Verification is bounded LOCAL evidence on the correct interface."""

    def setUp(self):
        wake_engine.reset_pre_roll()
        self.addCleanup(wake_engine.reset_pre_roll)
        self.verify = patch.object(wake_engine, "WAKE_ONLINE_VERIFY", True)
        self.verify.start()
        self.addCleanup(self.verify.stop)
        wake_engine.pre_roll().feed(_audio(0.6))

    def _daemon(self, result=None, raises=None):
        captured = {}

        def fake(wav):
            captured["arg"] = wav
            if raises is not None:
                raise raises
            return result

        return fake, captured

    def test_verification_hands_the_daemon_wav_bytes(self):
        fake, captured = self._daemon(("wake up jarvis", {}))
        with patch.object(watcher, "_transcribe_with_daemon",
                          side_effect=fake):
            self.assertTrue(wake_engine.online_verify())
        self.assertIsInstance(captured["arg"], (bytes, bytearray),
                              "verification used an incompatible interface")

    def test_non_wake_pre_roll_rejects_the_launch(self):
        fake, _captured = self._daemon(("some random chatter", {}))
        with patch.object(watcher, "_transcribe_with_daemon",
                          side_effect=fake):
            self.assertFalse(wake_engine.online_verify())

    def test_empty_local_transcript_fails_closed(self):
        fake, _captured = self._daemon((None, {}))
        with patch.object(watcher, "_transcribe_with_daemon",
                          side_effect=fake):
            self.assertFalse(wake_engine.online_verify())

    def test_a_verification_error_fails_closed(self):
        fake, _captured = self._daemon(raises=OSError("daemon down"))
        with patch.object(watcher, "_transcribe_with_daemon",
                          side_effect=fake):
            self.assertFalse(wake_engine.online_verify())

    def test_disabled_verification_proceeds_and_drains_nothing(self):
        with patch.object(wake_engine, "WAKE_ONLINE_VERIFY", False):
            self.assertTrue(wake_engine.online_verify())


class KeywordSpotWiringTests(unittest.TestCase):
    """Keyword spotting participates in the watcher's wake decision."""

    def setUp(self):
        wake_engine.reset_pre_roll()
        self.addCleanup(wake_engine.reset_pre_roll)

    def _listen(self, spotted, verified=True):
        with patch.object(watcher, "recognize_candidates",
                          return_value=([], None)), \
             patch.object(watcher, "open_microphone") as mic, \
             patch.object(watcher.recognizer, "listen",
                          return_value=_audio(1.0)), \
             patch.object(watcher, "recalibrate_watcher"), \
             patch.object(wake_engine, "keyword_spot", return_value=spotted), \
             patch.object(wake_engine, "online_verify",
                          return_value=verified):
            mic.return_value.__enter__.return_value = object()
            return watcher.listen_once()

    def test_a_verified_keyword_spot_is_a_wake_decision(self):
        candidates, wake_match = self._listen(True, verified=True)
        self.assertIsNotNone(wake_match)
        self.assertEqual(candidates, [])

    def test_an_unverified_keyword_spot_does_not_wake(self):
        _candidates, wake_match = self._listen(True, verified=False)
        self.assertIsNone(wake_match)

    def test_no_keyword_spot_and_no_transcript_stays_asleep(self):
        _candidates, wake_match = self._listen(False)
        self.assertIsNone(wake_match)


class ForwardingTests(unittest.TestCase):
    """Forwarding is authenticated, identified, readiness-aware and ONCE."""

    def setUp(self):
        wake_engine.reset_forward_state()
        self.addCleanup(wake_engine.reset_forward_state)
        self.posts = []
        self.health_probes = []

    def _fake_urlopen(self, fail_times=0):
        state = {"failures": fail_times}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"reply": "ok"}'

        def fake(request, timeout=None):
            if request.full_url.endswith("/health"):
                self.health_probes.append(request.full_url)
                if state["failures"] > 0:
                    state["failures"] -= 1
                    raise OSError("backend not up yet")
                return _Resp()
            self.posts.append(request)
            if state["failures"] > 0:
                state["failures"] -= 1
                raise OSError("connection dropped")
            return _Resp()

        return fake

    def test_forward_is_authenticated_and_identified(self):
        import urllib.request as _ur

        with patch.dict(os.environ, {"JARVIS_LOCAL_TOKEN": "t" * 40}), \
             patch.object(_ur, "urlopen", side_effect=self._fake_urlopen()):
            ok = wake_engine.forward_wake_command(
                "open chrome", request_id="wake-req-1")
        self.assertTrue(ok)
        self.assertEqual(len(self.posts), 1)
        request = self.posts[0]
        self.assertEqual(request.get_header("X-jarvis-token"), "t" * 40)
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["request_id"], "wake-req-1")
        self.assertEqual(payload["message"], "open chrome")

    def test_a_literal_tail_is_forwarded_byte_exact(self):
        import urllib.request as _ur

        literal = "run --Dry-Run -C C:\\Users\\Me\\Q4 Report.PDF"
        with patch.object(_ur, "urlopen", side_effect=self._fake_urlopen()):
            self.assertTrue(wake_engine.forward_wake_command(
                literal, request_id="wake-literal"))
        payload = json.loads(self.posts[0].data.decode("utf-8"))
        self.assertEqual(payload["message"], literal)

    def test_the_same_request_id_executes_at_most_once(self):
        import urllib.request as _ur

        with patch.object(_ur, "urlopen", side_effect=self._fake_urlopen()):
            first = wake_engine.forward_wake_command(
                "open chrome", request_id="wake-once")
            second = wake_engine.forward_wake_command(
                "open chrome", request_id="wake-once")
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(self.posts), 1)

    def test_readiness_is_awaited_before_the_single_post(self):
        import urllib.request as _ur

        calls = []

        def wait_ready(timeout_seconds=None, proc=None):
            calls.append(timeout_seconds)
            return True

        with patch.object(watcher, "wait_for_backend_ready",
                          side_effect=wait_ready), \
             patch.object(_ur, "urlopen", side_effect=self._fake_urlopen()):
            ok = wake_engine.forward_wake_command(
                "open chrome", request_id="wake-ready", wait_ready=True)
        self.assertTrue(ok)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(self.posts), 1)

    def test_execution_survives_a_failed_readiness_wait(self):
        """A launch that is slow/unreachable must not duplicate the tail."""
        import urllib.request as _ur

        with patch.object(watcher, "wait_for_backend_ready",
                          return_value=False), \
             patch.object(_ur, "urlopen", side_effect=self._fake_urlopen()):
            ok = wake_engine.forward_wake_command(
                "open chrome", request_id="wake-slow", wait_ready=True)
        self.assertTrue(ok)
        self.assertEqual(len(self.posts), 1)

    def test_a_disconnection_does_not_replay_the_tail(self):
        import urllib.request as _ur

        with patch.object(_ur, "urlopen",
                          side_effect=self._fake_urlopen(fail_times=1)):
            first = wake_engine.forward_wake_command(
                "open chrome", request_id="wake-drop")
            second = wake_engine.forward_wake_command(
                "open chrome", request_id="wake-drop")
        self.assertFalse(first)
        self.assertFalse(second)
        self.assertEqual(len(self.posts), 1)

    def test_empty_command_is_a_noop(self):
        self.assertFalse(wake_engine.forward_wake_command("   "))
        self.assertFalse(wake_engine.forward_wake_command(None))


if __name__ == "__main__":
    unittest.main()
