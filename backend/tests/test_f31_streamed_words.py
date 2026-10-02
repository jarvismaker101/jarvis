"""F31 — preserve streamed words and answer channels.

Acceptance (audit report): "hel plus lo remains hello; thought/final mixed
responses never narrate reasoning; announcements do not splice into
unfinished answers."

The baseline defects pinned here:
  * the default Fireworks/custom streams could only carry an untyped iterator,
    so opt-in reasoning text was indistinguishable from answer text;
  * Gemini concatenated EVERY text part, including parts marked
    ``"thought": true``, into the one answer string (chat/TTS/memory);
  * ``StreamSpeaker`` fed announcements through the same text buffer as the
    in-flight answer, so "Working on the task" + "Task finished." became the
    single spliced utterance "Working on the taskTask finished.".

All HTTP, TTS and audio is mocked; nothing here touches the network, a
microphone or a real speech engine.
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

from backend.services import fireworks_client
from backend.services import gemini_client
from backend.services import openai_compat_client
from backend.services import voice as voice_mod
from backend.services.openai_compat_client import (
    FINAL_CHANNEL,
    REASONING_CHANNEL,
    StreamDelta,
    final_only,
)

MESSAGES = [{"role": "user", "content": "hello"}]


def _sse(lines):
    resp = MagicMock()
    resp.status_code = 200
    resp.iter_lines.return_value = lines
    return resp


def _openai_lines(chunks):
    """SSE lines for one OpenAI-dialect stream; *chunks* are delta dicts."""
    out = ["data: " + json.dumps({"choices": [{"delta": d}]}) for d in chunks]
    out.append("data: [DONE]")
    return out


class _SpeakerHarness(unittest.TestCase):
    """Real StreamSpeaker with the playback boundary mocked."""

    def setUp(self):
        with voice_mod._state_lock:
            voice_mod._speech_generation += 1

    def _speaker(self):
        sp = voice_mod.StreamSpeaker()
        captured = []
        orig_put = sp._queue.put

        def cap_put(item):
            if item is not voice_mod._STREAM_STOP:
                captured.append(item)
            return orig_put(item)

        self._patches = [
            patch.object(voice_mod, "prefetch_fish_audio"),
            patch.object(voice_mod, "_speak_chunk", return_value=True),
            patch.object(sp._queue, "put", side_effect=cap_put),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(sp.close)
        return sp, captured


class SplitWordTests(_SpeakerHarness):
    """Acceptance: 'hel' plus 'lo' remains 'hello'."""

    def test_split_word_deltas_stay_joined(self):
        sp, captured = self._speaker()
        sp.feed("hel")
        sp.feed("lo")
        sp.flush()
        time.sleep(0.05)
        self.assertEqual(captured, ["hello"])

    def test_fireworks_stream_then_speaker_keeps_the_word_whole(self):
        """The adapter forwards the model's exact chunks; the speaker's
        concatenation is byte-exact — no re-spacing, no re-joining."""
        lines = _openai_lines([
            {"content": "hel"},
            {"content": "lo"},
            {"content": " there"},
        ])
        sp, captured = self._speaker()
        with patch.object(fireworks_client, "API_KEY", "test-key"), \
             patch.object(fireworks_client, "REASONING_EFFORT", ""), \
             patch.object(fireworks_client._session, "post",
                          return_value=_sse(lines)):
            deltas = list(fireworks_client.ask_fireworks_stream(MESSAGES))
            for delta in deltas:
                sp.feed(delta)
        sp.flush()
        time.sleep(0.05)
        self.assertEqual(deltas, ["hel", "lo", " there"])
        self.assertEqual("".join(captured), "hello there")

    def test_typed_final_deltas_reach_the_speaker_unchanged(self):
        sp, captured = self._speaker()
        for delta in (StreamDelta("hel", FINAL_CHANNEL),
                      StreamDelta("lo", FINAL_CHANNEL)):
            sp.feed(delta)
        sp.flush()
        time.sleep(0.05)
        self.assertEqual(captured, ["hello"])


class ReasoningChannelTests(_SpeakerHarness):
    """Acceptance: thought/final mixed responses never narrate reasoning."""

    def test_fireworks_opt_in_reasoning_is_typed_not_merged(self):
        lines = _openai_lines([
            {"reasoning_content": "let me think..."},
            {"content": "hello"},
        ])
        with patch.object(fireworks_client, "API_KEY", "test-key"), \
             patch.object(fireworks_client, "REASONING_EFFORT", ""), \
             patch.object(fireworks_client._session, "post",
                          return_value=_sse(lines)):
            out = list(fireworks_client.ask_fireworks_stream(
                MESSAGES, include_reasoning=True))
        self.assertEqual(
            [(d.channel, d.text) for d in out],
            [(REASONING_CHANNEL, "let me think..."), (FINAL_CHANNEL, "hello")],
        )

    def test_fireworks_default_stream_is_final_only(self):
        lines = _openai_lines([
            {"reasoning_content": "let me think..."},
            {"content": "hel"},
            {"reasoning_content": "more thinking"},
            {"content": "lo"},
        ])
        with patch.object(fireworks_client, "API_KEY", "test-key"), \
             patch.object(fireworks_client, "REASONING_EFFORT", ""), \
             patch.object(fireworks_client._session, "post",
                          return_value=_sse(lines)):
            out = list(fireworks_client.ask_fireworks_stream(MESSAGES))
        self.assertEqual(out, ["hel", "lo"])

    def test_gemini_thought_parts_never_join_the_answer(self):
        payload = {
            "candidates": [{
                "content": {"parts": [
                    {"text": "weighing options", "thought": True},
                    {"text": "hello"},
                ]},
            }],
        }
        resp = _sse(["data: " + json.dumps(payload)])
        with patch.object(gemini_client, "GEMINI_API_KEY", "test-key"), \
             patch.object(gemini_client._session, "post", return_value=resp):
            out = list(gemini_client.ask_gemini_chat_stream(MESSAGES))
        self.assertEqual(out, ["hello"])

    def test_gemini_thought_parts_are_preserved_on_the_reasoning_channel(self):
        payload = {
            "candidates": [{
                "content": {"parts": [
                    {"text": "weighing options", "thought": True},
                    {"text": "hello"},
                ]},
            }],
        }
        resp = _sse(["data: " + json.dumps(payload)])
        with patch.object(gemini_client, "GEMINI_API_KEY", "test-key"), \
             patch.object(gemini_client._session, "post", return_value=resp):
            out = list(gemini_client.ask_gemini_chat_stream(
                MESSAGES, typed=True))
        self.assertEqual(
            [(d.channel, d.text) for d in out],
            [(REASONING_CHANNEL, "weighing options"),
             (FINAL_CHANNEL, "hello")],
        )

    def test_gemini_non_stream_chat_excludes_thought_parts(self):
        body = {
            "candidates": [{
                "content": {"parts": [
                    {"text": "hidden thinking", "thought": True},
                    {"text": "the answer"},
                ]},
            }],
        }
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = body
        with patch.object(gemini_client, "GEMINI_API_KEY", "test-key"), \
             patch.object(gemini_client._no_retry_session, "post",
                          return_value=resp):
            result = gemini_client.ask_gemini_chat(MESSAGES, no_retry=True)
        self.assertEqual(result["choices"][0]["message"]["content"],
                         "the answer")

    def test_openai_compat_reasoning_field_never_becomes_answer_text(self):
        lines = _openai_lines([
            {"reasoning": "gateway thinking"},
            {"content": "hel"},
            {"content": "lo"},
        ])
        resp = _sse(lines)

        class _FakeSession:
            def post(self, *a, **k):
                return resp

        with patch.object(openai_compat_client, "_session", _FakeSession()):
            plain = list(openai_compat_client.ask_openai_compat_stream(
                MESSAGES, "m", "https://x.test/v1", "k"))
            typed = list(openai_compat_client.ask_openai_compat_stream(
                MESSAGES, "m", "https://x.test/v1", "k", typed=True))
        self.assertEqual(plain, ["hel", "lo"])
        self.assertEqual(
            [(d.channel, d.text) for d in typed],
            [(REASONING_CHANNEL, "gateway thinking"),
             (FINAL_CHANNEL, "hel"), (FINAL_CHANNEL, "lo")],
        )

    def test_the_speaker_preserves_but_never_speaks_reasoning(self):
        sp, captured = self._speaker()
        sp.feed(StreamDelta("thinking hard", REASONING_CHANNEL))
        sp.feed(StreamDelta("hel", FINAL_CHANNEL))
        sp.feed(StreamDelta("lo", FINAL_CHANNEL))
        sp.flush()
        time.sleep(0.05)
        self.assertEqual(sp.reasoning, "thinking hard")
        self.assertEqual(captured, ["hello"])

    def test_final_only_gate_filters_a_mixed_stream(self):
        mixed = [StreamDelta("thinking", REASONING_CHANNEL),
                 StreamDelta("ans", FINAL_CHANNEL), "wer"]
        final = list(final_only(mixed))
        # The gate drops the reasoning channel and keeps the final deltas
        # byte-exact (it never re-joins or re-spaces them).
        self.assertEqual(final, ["ans", "wer"])
        self.assertEqual("".join(final), "answer")

    def test_backend_frames_carrying_a_reasoning_channel_are_not_spoken(self):
        """The voice transport's own delta frames: a reasoning frame must
        never reach the speaker, and must not count as spoken text."""
        from backend import voice_mode

        frames = [
            {"type": "delta", "text": "hidden thinking", "seq": 0,
             "channel": "reasoning"},
            {"type": "reasoning", "text": "more thinking", "seq": 1},
            {"type": "delta", "text": "hel", "seq": 2},
            {"type": "delta", "text": "lo", "seq": 3},
            {"type": "completed", "reply": "hello", "seq": 4},
        ]
        lines = [("data: " + json.dumps(f) + "\n\n").encode("utf-8")
                 for f in frames]

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def __iter__(self):
                return iter(lines)

        spoken = []
        with patch.object(voice_mode, "urlopen",
                          return_value=_Response()):
            reply = voice_mode._ask_backend("hi", "req-f31",
                                            stream_sink=spoken.append)
        self.assertEqual(reply, "hello")
        self.assertEqual(spoken, ["hel", "lo"])


class AnnouncementTests(_SpeakerHarness):
    """Acceptance: announcements do not splice into unfinished answers."""

    def test_announcement_is_a_separate_utterance(self):
        sp, captured = self._speaker()
        sp.feed("Working on the task")
        self.assertTrue(sp.enqueue_external("Task finished."))
        time.sleep(0.05)
        self.assertEqual(captured, ["Working on the task", "Task finished."])

    def test_no_queue_item_mixes_answer_and_announcement_words(self):
        sp, captured = self._speaker()
        sp.feed("The answer is")
        sp.enqueue_external("Deployment complete.")
        sp.feed(" forty two")
        sp.flush()
        time.sleep(0.05)
        for item in captured:
            self.assertFalse("The answer is" in item and "Deployment" in item,
                             "announcement spliced into the answer: %r" % item)
        self.assertIn("The answer is", captured)
        self.assertIn("Deployment complete.", captured)

    def test_announcement_survives_after_the_answer_finised(self):
        sp, captured = self._speaker()
        sp.feed("All done.")
        sp.finish()
        time.sleep(0.05)
        self.assertTrue(sp.enqueue_external("Task finished"))
        time.sleep(0.3)
        self.assertIn("Task finished", captured)


if __name__ == "__main__":
    unittest.main()
