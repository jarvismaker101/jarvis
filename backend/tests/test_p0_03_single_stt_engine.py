"""P0-03 — exactly ONE STT engine per turn, chosen by settings.

Pinned defects:

  * ``listener.recognize_multilingual()`` chained engines: the selected engine
    first, the OTHER engine second (``_try_inworld() or _try_local_whisper()`` /
    ``_try_local_whisper() or _try_inworld()``), and then — if both failed — it
    looped ``RECOGNITION_LANGUAGES`` calling ``recognize_google_or_groq`` per
    language. Sharing one serial budget of
    Inworld 20s + local whisper 15s + 2*(Google 6s + Groq 35s) = **117s**.
  * a failed turn returned ``(None, None, None)`` with no reason: "the user
    said nothing" and "every engine failed" were indistinguishable in the log.

The user's decision (this audit): ONE engine per turn, no ladder, no streaming.
The cloud policy gate is untouched and still fails closed; the hallucination
gate is untouched and still runs for every engine.

Every engine is stubbed; nothing here touches the network, a microphone or a
device.
"""

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import ANY, patch

import speech_recognition as sr

from backend.services import listener, transcription


def _audio():
    """The engines are stubbed, so the object is only a token."""
    return object()


class _EngineHarness(unittest.TestCase):
    """Patches the three engines and the registry, and counts the calls."""

    def setUp(self):
        self.inworld = patch.object(listener, "recognize_inworld")
        self.local = patch.object(listener, "recognize_local_whisper")
        self.online = patch.object(listener, "recognize_google_or_groq")
        self.mock_inworld = self.inworld.start()
        self.mock_local = self.local.start()
        self.mock_online = self.online.start()
        for item in (self.inworld, self.local, self.online):
            self.addCleanup(item.stop)

        # Cloud egress allowed unless a test says otherwise (F34 policy).
        self.policy = patch.object(listener, "_cloud_stt_policy",
                                   return_value="on")
        self.policy.start()
        self.addCleanup(self.policy.stop)

        for name in ("JARVIS_STT_CLOUD_POLICY", "JARVIS_STT_LOCAL_ONLY"):
            os.environ.pop(name, None)
            self.addCleanup(os.environ.pop, name, None)

    def _select(self, provider):
        registry = patch.object(
            listener.model_registry,
            "get_model_for_role",
            return_value={"provider": provider, "model": f"{provider}/test"},
        )
        registry.start()
        self.addCleanup(registry.stop)

    def _calls(self):
        return (self.mock_inworld.call_count,
                self.mock_local.call_count,
                self.mock_online.call_count)

    def _recognize(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = listener.recognize_multilingual(_audio())
        return result, buffer.getvalue()


class SingleEngineTests(_EngineHarness):
    def test_inworld_failure_does_not_fall_through_to_any_other_engine(self):
        """One engine means ONE attempt — the 117s ladder is gone."""
        self._select("inworld")
        self.mock_inworld.side_effect = sr.RequestError("inworld down")
        self.mock_local.return_value = ("a local second opinion", "en")
        self.mock_online.return_value = "a google second opinion"

        result, log = self._recognize()

        self.assertEqual(result, (None, None, None))
        self.assertEqual(self._calls(), (1, 0, 0),
                         "more than one engine was contacted for one turn")
        self.assertNotIn("local", log.lower().split("stt failed")[-1])

    def test_whisper_failure_does_not_fall_through_to_inworld(self):
        self._select("whisper")
        self.mock_local.side_effect = sr.RequestError("daemon down")
        self.mock_inworld.return_value = "an inworld second opinion"

        result, _log = self._recognize()

        self.assertEqual(result, (None, None, None))
        self.assertEqual(self._calls(), (0, 1, 0))

    def test_a_selected_inworld_never_iterates_the_language_list(self):
        """The per-language Google/Groq loop was a THIRD and FOURTH attempt."""
        self._select("inworld")
        self.mock_inworld.side_effect = sr.RequestError("inworld down")

        with patch.object(listener, "RECOGNITION_LANGUAGES",
                          ("en-IN", "hi-IN", "en-US")):
            self._recognize()

        self.mock_online.assert_not_called()

    def test_a_selected_whisper_runs_once_and_uses_the_final_budget(self):
        self._select("whisper")
        self.mock_local.return_value = ("Open C:\\Temp\\A.TXT", "en")

        raw, normalized, language = listener.recognize_multilingual(_audio())

        self.assertEqual(raw, "Open C:\\Temp\\A.TXT")
        self.assertEqual(normalized, "open c:\\temp\\a.txt")
        self.assertEqual(language, "en")
        self.assertEqual(self._calls(), (0, 1, 0))
        self.mock_local.assert_called_once_with(
            ANY, timeout=listener.FINAL_STT_TIMEOUT_SECONDS)

    def test_the_selected_engine_answers_even_when_the_other_would_have(self):
        self._select("inworld")
        self.mock_inworld.return_value = "Inworld answer"
        self.mock_local.return_value = ("Local answer", "en")

        raw, _normalized, _language = listener.recognize_multilingual(_audio())

        self.assertEqual(raw, "Inworld answer")
        self.assertEqual(self._calls(), (1, 0, 0))

    def test_the_selection_is_read_per_call_so_a_switch_needs_no_restart(self):
        """P1-05 contract: the role is resolved fresh on the NEXT phrase."""
        self._select("whisper")
        self.mock_local.return_value = ("first", "en")
        listener.recognize_multilingual(_audio())

        self._select("inworld")
        self.mock_inworld.return_value = "second"
        raw, _normalized, _language = listener.recognize_multilingual(_audio())

        self.assertEqual(raw, "second")
        self.assertEqual(self._calls(), (1, 1, 0))

    def test_an_unknown_provider_still_resolves_to_one_engine(self):
        self._select("something-else")
        self.mock_inworld.return_value = "default engine"
        raw, _normalized, _language = listener.recognize_multilingual(_audio())
        self.assertEqual(raw, "default engine")
        self.assertEqual(self._calls(), (1, 0, 0))

    def test_the_language_parameterised_engine_keeps_its_language_list(self):
        """The language list survives for the engine that needs it — as the
        SELECTED engine, not as a ladder rung behind the other two."""
        self.mock_online.side_effect = [sr.UnknownValueError(),
                                        "main toh hi"]
        with patch.object(listener, "_engine_for_listening_role",
                          return_value="google-or-groq"), \
                patch.object(listener, "RECOGNITION_LANGUAGES",
                             ("en-IN", "hi-IN")):
            raw, normalized, language = listener.recognize_multilingual(_audio())

        self.assertEqual((raw, normalized, language),
                         ("main toh hi", "main toh hi", "hi-IN"))
        self.assertEqual(self._calls(), (0, 0, 2))
        self.assertEqual(listener.LAST_STT_FAILURE, None)


class CloudPolicyTests(_EngineHarness):
    """F34: the policy gate is unchanged and still fails closed."""

    def test_a_local_only_policy_never_calls_a_cloud_engine(self):
        self.policy.stop()               # replace the "on" default
        self.addCleanup(self.policy.start)
        with patch.object(listener, "_cloud_stt_policy",
                          return_value="local-only"):
            self._select("inworld")      # a CLOUD role under a local policy
            self.mock_local.side_effect = sr.RequestError("daemon down")
            result, _log = self._recognize()

        self.assertEqual(result, (None, None, None))
        self.mock_inworld.assert_not_called()
        self.mock_online.assert_not_called()
        self.mock_local.assert_called_once()

    def test_an_off_policy_uploads_nothing_and_yields_no_transcript(self):
        self.policy.stop()
        self.addCleanup(self.policy.start)
        with patch.object(listener, "_cloud_stt_policy", return_value="off"):
            self._select("inworld")
            self.mock_local.side_effect = sr.RequestError("daemon down")
            result, _log = self._recognize()

        self.assertEqual(result, (None, None, None))
        self.assertEqual(self._calls(), (0, 1, 0))

    def test_a_local_only_policy_keeps_a_local_success_working(self):
        self.policy.stop()
        self.addCleanup(self.policy.start)
        with patch.object(listener, "_cloud_stt_policy",
                          return_value="local-only"):
            self._select("inworld")
            self.mock_local.return_value = ("Local only answer", "en")
            raw, _normalized, language = listener.recognize_multilingual(_audio())

        self.assertEqual(raw, "Local only answer")
        self.assertEqual(language, "en")
        self.assertEqual(self._calls(), (0, 1, 0))


class FailureReasonTests(_EngineHarness):
    """A failed turn must be VISIBLE, not a silent None."""

    def test_a_failure_records_a_reason_code_and_logs_it(self):
        self._select("inworld")
        self.mock_inworld.side_effect = sr.RequestError("INWORLD_STT_API_KEY is missing")

        _result, log = self._recognize()

        self.assertEqual(listener.LAST_STT_FAILURE,
                         ("inworld", "INWORLD_STT_API_KEY is missing"))
        self.assertIn("STT failed: inworld: INWORLD_STT_API_KEY is missing", log)

    def test_the_reason_names_the_engine_that_was_actually_used(self):
        self._select("whisper")
        self.mock_local.side_effect = sr.RequestError("local whisper refused")

        _result, log = self._recognize()

        self.assertEqual(listener.LAST_STT_FAILURE[0], "local-whisper")
        self.assertIn("STT failed: local-whisper: local whisper refused", log)

    def test_no_speech_is_its_own_reason(self):
        self._select("inworld")
        self.mock_inworld.side_effect = sr.UnknownValueError()

        _result, log = self._recognize()

        self.assertEqual(listener.LAST_STT_FAILURE, ("inworld", "no-speech"))
        self.assertIn("STT failed: inworld: no-speech", log)

    def test_an_empty_transcript_says_so(self):
        self._select("inworld")
        self.mock_inworld.return_value = "   "

        _result, _log = self._recognize()

        self.assertEqual(listener.LAST_STT_FAILURE, ("inworld", "empty transcript"))

    def test_a_non_english_script_is_a_failure_not_a_retry(self):
        """Inworld's English hint was ignored: no second engine exists."""
        self._select("inworld")
        self.mock_inworld.return_value = "\u0928\u092e\u0938\u094d\u0924\u0947"

        result, _log = self._recognize()

        self.assertEqual(result, (None, None, None))
        self.assertEqual(listener.LAST_STT_FAILURE,
                         ("inworld", "non-english-script"))
        self.assertEqual(self._calls(), (1, 0, 0))

    def test_a_success_clears_the_reason(self):
        self._select("inworld")
        self.mock_inworld.side_effect = sr.RequestError("first turn failed")
        self._recognize()
        self.assertIsNotNone(listener.LAST_STT_FAILURE)

        self.mock_inworld.side_effect = None
        self.mock_inworld.return_value = "second turn works"
        self._recognize()

        self.assertIsNone(listener.LAST_STT_FAILURE)


class HallucinationGateTests(_EngineHarness):
    """The gate is unchanged: it still covers the single engine."""

    def test_a_hallucination_from_inworld_is_rejected(self):
        self._select("inworld")
        self.mock_inworld.return_value = "jarvis, wake up, jervis, utho, jago, chalu"

        result, log = self._recognize()

        self.assertEqual(result, (None, None, None))
        self.assertIn("Ignoring STT hallucination", log)
        self.assertEqual(listener.LAST_STT_FAILURE, ("inworld", "hallucination"))

    def test_a_hallucination_from_whisper_is_rejected(self):
        self._select("whisper")
        self.mock_local.return_value = ("chalu, chalu, chalu, chalu", "en")

        result, _log = self._recognize()

        self.assertEqual(result, (None, None, None))
        self.assertEqual(listener.LAST_STT_FAILURE,
                         ("local-whisper", "hallucination"))
        self.assertEqual(self._calls(), (0, 1, 0))

    def test_a_real_transcript_still_passes(self):
        self._select("inworld")
        self.mock_inworld.return_value = "Open C:\\Temp\\A.TXT"

        raw, _normalized, _language = listener.recognize_multilingual(_audio())

        self.assertEqual(raw, "Open C:\\Temp\\A.TXT")
        self.assertIsNone(listener.LAST_STT_FAILURE)

    def test_the_gate_is_reachable_for_every_engine(self):
        """Every engine's transcript goes through the same belt."""
        hallucinations = {
            "inworld": lambda: self.mock_inworld,
            "whisper": lambda: self.mock_local,
        }
        for provider, get_mock in hallucinations.items():
            with self.subTest(provider=provider):
                self._select(provider)
                mock = get_mock()
                if provider == "whisper":
                    mock.return_value = ("a ver si te acuerdas de esto", "es")
                else:
                    mock.return_value = "a ver si te acuerdas de esto"
                result, _log = self._recognize()
                self.assertEqual(result, (None, None, None))


class TimeoutBudgetTests(unittest.TestCase):
    """With one engine, the turn's worst case IS that engine's timeout."""

    def test_the_local_final_budget_is_tighter_than_the_generic_default(self):
        self.assertLessEqual(listener.FINAL_STT_TIMEOUT_SECONDS, 10.0)
        self.assertGreater(listener.FINAL_STT_TIMEOUT_SECONDS, 0)

    def test_the_partial_budget_is_unchanged_and_smaller(self):
        self.assertEqual(listener.PARTIAL_STT_TIMEOUT_SECONDS, 4.0)
        self.assertLess(listener.PARTIAL_STT_TIMEOUT_SECONDS,
                        listener.FINAL_STT_TIMEOUT_SECONDS)

    def test_inworld_has_explicit_connect_and_read_budgets(self):
        connect, read = transcription.INWORLD_STT_TIMEOUT
        self.assertLessEqual(connect, 5)
        self.assertLessEqual(read, 15)
        self.assertGreater(connect, 0)
        self.assertGreater(read, 0)

    def test_the_worst_case_of_a_single_engine_turn_is_bounded(self):
        """The 117s serial ladder cannot be reconstructed from these budgets."""
        worst = max(listener.FINAL_STT_TIMEOUT_SECONDS,
                    sum(transcription.INWORLD_STT_TIMEOUT))
        self.assertLess(worst, 20.0)


if __name__ == "__main__":
    unittest.main()
