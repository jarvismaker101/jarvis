"""[S6] One LLM call classifies the turn AND answers a chat turn.

The intent router is asked for a verdict and, when the verdict is ``chat``, the
answer itself. Nothing else changes: every non-chat route is untouched, the
deterministic nets still get their chance to upgrade a chat verdict, and an
empty/unusable reply falls back to the normal chat model so a turn is never
lost.
"""

import unittest
from unittest.mock import patch

from backend.core import brain
from backend.services import intent as intent_mod


class RouterReplyParsingTests(unittest.TestCase):
    """The router's reply is normalised, bounded, and only for chat."""

    def test_a_chat_verdict_carries_its_reply(self):
        result = intent_mod._parse_intent_json(
            '{"intent": "chat", "reply": "Blue light scatters most."}',
            "why is the sky blue")
        self.assertEqual(result["intent"], "chat")
        self.assertEqual(result["reply"], "Blue light scatters most.")

    def test_the_reply_is_whitespace_collapsed(self):
        result = intent_mod._parse_intent_json(
            '{"intent": "chat", "reply": "  one\\n two   three  "}', "hi")
        self.assertEqual(result["reply"], "one two three")

    def test_an_absent_blank_or_non_string_reply_is_empty(self):
        for payload in ('{"intent": "chat"}',
                        '{"intent": "chat", "reply": "   "}',
                        '{"intent": "chat", "reply": 42}'):
            result = intent_mod._parse_intent_json(payload, "hi")
            self.assertEqual(result["intent"], "chat")
            self.assertEqual(result["reply"], "", payload)

    def test_a_non_chat_verdict_never_carries_a_reply(self):
        for payload in (
                '{"intent": "tool", "reply": "sure", "steps": '
                '[{"action": "open_website", "input": "site.com"}]}',
                '{"intent": "task", "reply": "working on it", '
                '"task_description": "move files"}',
                '{"intent": "research", "reply": "here you go", '
                '"query": "claude fable 5.1 pricing"}'):
            result = intent_mod._parse_intent_json(payload, "x")
            self.assertNotEqual(result["intent"], "chat")
            self.assertEqual(result["reply"], "", payload)

    def test_unparseable_output_yields_no_reply(self):
        result = intent_mod._parse_intent_json("I think it is just chat", "hi")
        self.assertEqual(result["intent"], "chat")
        self.assertEqual(result["reply"], "")

    def test_the_router_prompt_asks_for_the_reply_in_the_same_call(self):
        prompt = intent_mod._INTENT_PROMPT
        self.assertIn("reply", prompt)
        self.assertIn("ONLY JSON", prompt)
        # The routing contract must survive the change.
        for token in ("tool", "screen", "region", "research", "task",
                      "task_description", "steps", "query"):
            self.assertIn(token, prompt)

    def test_a_reply_is_bounded(self):
        result = intent_mod._parse_intent_json(
            '{"intent": "chat", "reply": "%s"}' % ("x" * 5000), "hi")
        self.assertLessEqual(len(result["reply"]), 1200)


class SelectedModelOwnsTheSpokenReplyTests(unittest.TestCase):
    """A fallback hop may route, but never supplies the spoken answer (S5)."""

    def test_a_fallback_hops_chat_reply_is_dropped(self):
        with patch.object(intent_mod, "_selected_intent_model",
                          return_value=("gemini", "gemini-3.5-flash-lite")), \
             patch.object(intent_mod, "_classify_with_gemini",
                          return_value="") as gem, \
             patch.object(intent_mod, "_classify_with_groq",
                          return_value='{"intent": "chat", "reply": "hi there"}'):
            result = intent_mod.classify_intent("hello")
        self.assertEqual(result["intent"], "chat")
        self.assertEqual(result["reply"], "")
        self.assertEqual(result["_source"], "groq")

    def test_the_selected_hops_reply_is_kept(self):
        with patch.object(intent_mod, "_selected_intent_model",
                          return_value=("gemini", "gemini-3.5-flash-lite")), \
             patch.object(intent_mod, "_classify_with_gemini",
                          return_value='{"intent": "chat", "reply": "hi there"}'):
            result = intent_mod.classify_intent("hello")
        self.assertEqual(result["reply"], "hi there")
        self.assertEqual(result["_source"], "gemini")

    def test_the_total_classifier_failure_carries_no_reply(self):
        with patch.object(intent_mod, "_selected_intent_model",
                          return_value=None), \
             patch.object(intent_mod, "_classify_with_openrouter",
                          return_value=""), \
             patch.object(intent_mod, "_classify_with_gemini",
                          return_value=""), \
             patch.object(intent_mod, "_classify_with_groq",
                          return_value=""):
            result = intent_mod.classify_intent("hello")
        self.assertEqual(result["intent"], "chat")
        self.assertEqual(result["reply"], "")
        self.assertEqual(result["_source"], "none")


class _BrainHarness(unittest.TestCase):
    """Drives _process_message_inner with the router's verdict injected."""

    def setUp(self):
        self.history = []
        started = []
        specs = (
            ("add_message", dict(side_effect=lambda role, text:
                                 self.history.append((role, text)))),
            ("classify_intent", dict(return_value=self.intent_verdict())),
            ("handle_research_intent", dict(return_value="RESEARCHED")),
            ("handle_tool_intent", dict(return_value="TOOLED")),
            ("_ask_chat_nonstream", {}),
            ("_stream_chat_deltas", {}),
            ("_ChatRacer", {}),
        )
        for target, kwargs in specs:
            patcher = patch.object(brain, target, **kwargs)
            started.append((target, patcher.start()))
            self.addCleanup(patcher.stop)
        mocks = dict(started)
        self.nonstream = mocks["_ask_chat_nonstream"]
        self.stream_deltas = mocks["_stream_chat_deltas"]
        brain.memory_store = None

    def intent_verdict(self):
        return {"intent": "chat", "steps": [], "task_description": "",
                "query": "why is the sky blue", "reply": "", "_source": "gemini"}

    def run_turn(self, msg="why is the sky blue", **kwargs):
        return brain._process_message_inner(msg, **kwargs)


class RouterAnsweredChatTests(_BrainHarness):
    def intent_verdict(self):
        v = super().intent_verdict()
        v["reply"] = "Light scatters off air molecules, and blue scatters most."
        return v

    def test_the_chat_model_is_never_called(self):
        response = self.run_turn()
        self.assertEqual(
            response, "Light scatters off air molecules, and blue scatters most.")
        self.nonstream.assert_not_called()
        self.stream_deltas.assert_not_called()

    def test_the_reply_is_committed_to_history(self):
        self.run_turn()
        self.assertIn(
            ("assistant", "Light scatters off air molecules, and blue scatters most."),
            self.history)

    def test_a_streaming_consumer_receives_exactly_one_delta(self):
        deltas = []
        self.run_turn(stream_reply=deltas.append)
        self.assertEqual(
            deltas,
            ["Light scatters off air molecules, and blue scatters most."])
        self.stream_deltas.assert_not_called()

    def test_the_uncertainty_rewrite_never_undoes_spoken_text(self):
        # The reply was already emitted, so F26 must keep it verbatim rather
        # than swapping in a clarification question (one answer, spoken once).
        vague = {"intent": "chat", "steps": [], "task_description": "",
                 "query": "x", "reply": "I'm not sure, maybe.", "_source": "gemini"}
        deltas = []
        with patch.object(brain, "classify_intent", return_value=vague), \
             patch.object(brain, "maybe_proactive_research",
                          return_value=True) as research:
            response = self.run_turn("x", stream_reply=deltas.append)
        research.assert_not_called()
        self.assertEqual(response, "I'm not sure, maybe.")
        self.assertEqual(deltas, ["I'm not sure, maybe."])


class RouterReplyRespectsTheNetsTests(_BrainHarness):
    """A router reply never bypasses the deterministic upgrade nets."""

    def intent_verdict(self):
        v = super().intent_verdict()
        v["reply"] = "Claude Fable 5.1 lists at forty dollars a month."
        return v

    def test_a_fresh_info_question_still_researches(self):
        response = self.run_turn("what is the pricing for claude fable 5.1 model")
        self.assertEqual(response, "RESEARCHED")
        self.nonstream.assert_not_called()

    def test_a_screen_question_still_analyses_the_screen(self):
        with patch.object(brain, "analyze_screen",
                          return_value={"tip": "SCREEN", "evidence": [],
                                        "topic": "", "grounding_links": []}), \
             patch.object(brain, "begin_screen_capture",
                          return_value={"capture_id": "c1", "seq": 1}), \
             patch.object(brain, "push_screen_answer",
                          return_value="answer-1"):
            response = self.run_turn("what is on my screen")
        self.assertEqual(response, "SCREEN")
        self.nonstream.assert_not_called()


class NoRouterReplyFallsBackTests(_BrainHarness):
    """No reply from the router -> the chat model answers, exactly as before."""

    def test_the_chat_model_still_answers(self):
        self.nonstream.return_value = {
            "choices": [{"message": {"content": "from the chat model"}}]}
        response = self.run_turn()
        self.nonstream.assert_called_once()
        self.assertEqual(response, "from the chat model")
        self.assertIn(("assistant", "from the chat model"), self.history)


class HandleChatAnsweredTests(unittest.TestCase):
    """handle_chat(answered=...) is a complete chat reply, not a shortcut."""

    def setUp(self):
        self.history = []
        self.built = patch.object(brain, "_build_chat_messages").start()
        self.nonstream = patch.object(
            brain, "_ask_chat_nonstream").start()
        self.deltas = patch.object(brain, "_stream_chat_deltas").start()
        add = patch.object(
            brain, "add_message",
            side_effect=lambda role, text: self.history.append((role, text)))
        add.start()
        self.addCleanup(self.built.stop)
        self.addCleanup(self.nonstream.stop)
        self.addCleanup(self.deltas.stop)
        self.addCleanup(add.stop)
        brain.memory_store = None

    def test_the_chat_pipeline_is_never_entered(self):
        text = "A direct answer."
        out = brain.handle_chat("why", answered=text)
        self.assertEqual(out, text)
        self.built.assert_not_called()
        self.nonstream.assert_not_called()
        self.deltas.assert_not_called()

    def test_the_user_turn_is_still_recorded(self):
        brain.handle_chat("why is the sky blue", answered="Because scattering.")
        self.assertEqual(self.history[0], ("user", "why is the sky blue"))
        self.assertEqual(self.history[1], ("assistant", "Because scattering."))

    def test_multi_line_router_text_is_flattened_for_speech(self):
        out = brain.handle_chat("x", answered="one\n\ntwo   three")
        self.assertEqual(out, "one two three")


if __name__ == "__main__":
    unittest.main()