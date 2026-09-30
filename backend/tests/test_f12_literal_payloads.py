"""F12 — preserve literal task payloads.

Acceptance (audit report): "Mixed-case paths, signed URLs, uppercase flags,
quoted and, indentation/newlines, and literal placeholders survive typed,
voice, and wake-tail routes unchanged."

The baseline defects pinned here:
  * wake processing lowercased and RE-TOKENISED every candidate
    (``" ".join(tokens)``), so punctuation, paths, URLs, flags, quotes and
    whitespace were destroyed before the planner saw them;
  * ``forward_wake_command`` normalized the command again before sending it;
  * ``extract_transcripts`` returned the normalized copy as the candidate, so
    the raw spelling never survived the wake route.

The typed route is pinned against ``backend.core.brain`` (owned by the
coordinating agent for F12 — asserted here, not modified).

No microphone, TTS engine, wake engine or subprocess is opened here.
"""

import os
import unittest
from unittest.mock import patch

import speech_recognition as sr

from backend import watcher
from backend.core import brain
from backend.services import listener
from backend.services import wake_engine

LITERAL_CASES = [
    ('Jarvis, copy "Q4 Report.PDF" to C:\\Users\\Me\\Final',
     'copy "Q4 Report.PDF" to C:\\Users\\Me\\Final'),
    ("Jarvis fetch https://ex.test/a?X-Amz-Signature=AbC123&Expires=99",
     "fetch https://ex.test/a?X-Amz-Signature=AbC123&Expires=99"),
    ("Jarvis run git -C MyRepo STATUS --PORCELAIN",
     "run git -C MyRepo STATUS --PORCELAIN"),
    ('Jarvis, echo "Rock and Roll"', 'echo "Rock and Roll"'),
    ("Jarvis,\n  deploy:\n  stage: prod\n  keep: True",
     "deploy:\n  stage: prod\n  keep: True"),
    ("Jarvis, run deploy {{step0.files.0}}.ps1 --Wait True",
     "run deploy {{step0.files.0}}.ps1 --Wait True"),
    ("Jarvis, read C:\\Temp\\report{{step0}}.txt",
     "read C:\\Temp\\report{{step0}}.txt"),
    ("Jarvis open %USERPROFILE%\\Notes\\Q4 Plan.MD",
     "open %USERPROFILE%\\Notes\\Q4 Plan.MD"),
]


class WakeTailLiteralTests(unittest.TestCase):
    """Acceptance: the wake-tail route forwards literal payloads."""

    def test_literal_payloads_survive_the_wake_tail(self):
        for phrase, expected in LITERAL_CASES:
            self.assertEqual(wake_engine.extract_command(phrase), expected,
                             "wake tail was rewritten for %r" % phrase)

    def test_the_tail_is_not_lowercased_or_retokenised(self):
        tail = wake_engine.extract_command(
            "Jarvis, Copy   C:\\Users\\Me\\Q4 Report.PDF  To  D:\\Backup")
        self.assertIn("Q4 Report.PDF", tail)
        self.assertIn("  To  ", tail,
                      "internal whitespace was reconstructed, not preserved")
        self.assertNotIn(" q4 ", tail)

    def test_a_quoted_and_survives(self):
        tail = wake_engine.extract_command('Jarvis say "Rock and Roll"')
        self.assertEqual(tail, 'say "Rock and Roll"')
        self.assertIn(" and ", tail)

    def test_placeholder_tokens_are_never_resolved_by_the_wake_route(self):
        tail = wake_engine.extract_command(
            "Jarvis, patch {{step0.files.0}} --dry-run")
        self.assertEqual(tail, "patch {{step0.files.0}} --dry-run")

    def test_content_delimiters_inside_a_filename_are_left_alone(self):
        tail = wake_engine.extract_command(
            "Jarvis, open C:\\Temp\\notes{\"a\": 1}.txt")
        self.assertEqual(tail, 'open C:\\Temp\\notes{"a": 1}.txt')


class WatcherRawTranscriptTests(unittest.TestCase):
    """Acceptance: the wake route never hands a normalized copy onward."""

    def test_extract_transcripts_preserves_the_raw_spelling(self):
        result = {"alternative": [{"transcript": "Open C:\\Temp\\My File.TXT"}]}
        self.assertEqual(watcher.extract_transcripts(result),
                         ["Open C:\\Temp\\My File.TXT"])

    def test_dedupe_keeps_the_first_literal_spelling(self):
        result = {"alternative": [
            {"transcript": "Open C:\\Temp\\My File.TXT"},
            {"transcript": "open c:\\temp\\my file.txt"},
        ]}
        self.assertEqual(watcher.extract_transcripts(result),
                         ["Open C:\\Temp\\My File.TXT"])

    def test_find_wake_match_returns_the_raw_transcript(self):
        transcripts = ["Jarvis, Open Chrome", "hello there"]
        self.assertEqual(watcher.find_wake_match(transcripts),
                         "Jarvis, Open Chrome")

    def test_a_filename_is_not_a_wake_word(self):
        # A filename is payload, not a wake phrase: no task may be created.
        self.assertIsNone(watcher.find_wake_match(["open report.txt"]))
        self.assertIsNone(watcher.find_wake_match(
            ["C:\\Temp\\jarvis-notes.txt"]))

    def test_recognize_candidates_returns_raw_candidates(self):
        with patch.object(watcher, "whisper_daemon_ok", True), \
             patch.object(watcher, "whisper_model", None), \
             patch.object(watcher, "_transcribe_with_daemon",
                          return_value=(
                              "Hey Jarvis, Patch {{step0}}.py --Dry-Run",
                              {})), \
             patch.object(watcher, "recognize_google_or_groq") as online:
            candidates, wake_match = watcher.recognize_candidates(
                sr.AudioData(b"\x00" * 3200, 16000, 2))
        self.assertEqual(wake_match, "Hey Jarvis, Patch {{step0}}.py --Dry-Run")
        self.assertEqual(candidates,
                         ["Hey Jarvis, Patch {{step0}}.py --Dry-Run"])
        self.assertEqual(
            wake_engine.extract_command(wake_match, candidates),
            "Patch {{step0}}.py --Dry-Run")
        online.assert_not_called()

    def test_forwarding_does_not_normalize_the_command(self):
        import json as _json
        import urllib.request as _ur

        wake_engine.reset_forward_state()
        self.addCleanup(wake_engine.reset_forward_state)
        literal = 'patch "Q4 Report.TXT" --Dry-Run'
        captured = {}

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"reply": "ok"}'

        def fake(request, timeout=None):
            captured["body"] = request.data.decode("utf-8")
            return _Resp()

        with patch.object(_ur, "urlopen", side_effect=fake):
            self.assertTrue(wake_engine.forward_wake_command(
                literal, request_id="f12-literal"))
        self.assertEqual(_json.loads(captured["body"])["message"], literal)


class VoiceRouteLiteralTests(unittest.TestCase):
    """Acceptance: the voice route hands the literal payload onward."""

    def setUp(self):
        listener._turn_stabilizer.reset()
        self.addCleanup(listener._turn_stabilizer.reset)

    def test_recognize_multilingual_keeps_raw_and_normalized_separate(self):
        with patch.object(listener.model_registry, "get_model_for_role",
                          return_value={"provider": "inworld"}), \
             patch.object(listener, "recognize_inworld",
                          return_value="Create File Q4 Report.TXT"), \
             patch.object(listener, "recognize_local_whisper",
                          side_effect=sr.RequestError("offline")) as whisper:
            raw, normalized, _language = listener.recognize_multilingual(
                object())
        self.assertEqual(raw, "Create File Q4 Report.TXT")
        self.assertEqual(normalized, "create file q4 report.txt")
        # [P0-03] The other engine is not a fallback any more.
        whisper.assert_not_called()

    def test_listen_returns_the_literal_utterance(self):
        literal = 'run {{step0.files.0}} --Dry-Run -C "D:\\Q4 Report"'
        with patch.object(listener, "_capture_audio",
                          return_value=sr.AudioData(b"\x00" * 3200,
                                                    16000, 2)), \
             patch.object(listener, "recognize_multilingual",
                          return_value=(literal, literal.lower(), "en")):
            self.assertEqual(listener.listen(), literal)

    def test_only_a_committed_literal_is_returned(self):
        literal = "Open C:\\Temp\\My File.TXT"
        with patch.object(listener, "_capture_audio",
                          return_value=sr.AudioData(b"\x00" * 3200,
                                                    16000, 2)), \
             patch.object(listener, "recognize_multilingual",
                          return_value=(literal, literal.lower(), "en")):
            returned = listener.listen()
        self.assertEqual(returned, literal)
        self.assertTrue(listener._turn_stabilizer.is_committed(literal))


class TypedRouteLiteralTests(unittest.TestCase):
    """Acceptance: the typed route (brain) preserves literal payloads."""

    def test_payloads_survive_filler_stripping(self):
        cases = [
            ("jarvis, please open C:\\Users\\Me\\Q4 Report.PDF",
             "open C:\\Users\\Me\\Q4 Report.PDF"),
            ("hey Jarvis run --Dry-Run -X POST", "run --Dry-Run -X POST"),
            ("jarvis check https://x.test/a?b=1&sig=AbC",
             "check https://x.test/a?b=1&sig=AbC"),
            ('jarvis python -c "print( 1 )"', 'python -c "print( 1 )"'),
            ("jarvis  deploy:\n  stage: prod\n  keep: True",
             "deploy:\n  stage: prod\n  keep: True"),
            ("jarvis do the thing please sir", "do the thing"),
            ("play it please", "play it"),
        ]
        for spoken, expected in cases:
            self.assertEqual(brain.strip_voice_filler_words(spoken), expected,
                             "typed route rewrote %r" % spoken)

    def test_literal_placeholders_survive_the_typed_route(self):
        self.assertEqual(
            brain.strip_voice_filler_words(
                "jarvis run deploy {{step0.files.0}}.ps1 --Wait True"),
            "run deploy {{step0.files.0}}.ps1 --Wait True")
        self.assertEqual(
            brain.strip_voice_filler_words("jarvis open %USERPROFILE%\\Q4.MD"),
            "open %USERPROFILE%\\Q4.MD")

    def test_case_is_never_lowercased(self):
        self.assertEqual(
            brain.strip_voice_filler_words("Jarvis Open The File.TXT"),
            "Open The File.TXT")


if __name__ == "__main__":
    unittest.main()
