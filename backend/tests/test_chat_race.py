import time
import unittest
from unittest.mock import patch, MagicMock

from backend.core import brain
from backend.services import intent as intent_mod


class ChatRaceStreamVsRouterTests(unittest.TestCase):
    def setUp(self):
        brain._pending_confirmation = None
        brain._pending_opencode_task = None
        brain._pending_browser_clarification = None
        brain._proactive_research_fired = False
        brain._pending_browser_clarification = None
        # clear memory
        from backend.core.memory import clear_history, get_history
        clear_history()
        self.get_history = get_history

    def tearDown(self):
        brain._pending_confirmation = None
        brain._pending_opencode_task = None
        brain._pending_browser_clarification = None
        brain._proactive_research_fired = False
        brain._pending_browser_clarification = None
        from backend.core.memory import clear_history
        clear_history()
        brain.set_opencode_task_running(False)

    def _wait(self, pred, timeout=1.5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if pred():
                return True
            time.sleep(0.02)
        return pred()

    # helper to make a fake streaming generator that tracks close()
    def _make_fake_stream(self, chunks, close_tracker):
        class FakeGen:
            def __init__(self, chunks):
                self._chunks = list(chunks)
                self.closed = False
                self.close_tracker = close_tracker
            def __iter__(self):
                for c in self._chunks:
                    yield c
                # keep yielding nothing after? just stop
            def close(self):
                self.closed = True
                close_tracker["closed"] = True
        return FakeGen(chunks)

    def test_chat_verdict_streams_and_commits_once(self):
        # fake deltas
        deltas = ["Hello ", "sir."]
        close_tracker = {"closed": False}

        def fake_stream(messages, temperature, max_tokens, cancel=None):
            # this is called inside _ChatRacer thread via _stream_chat_deltas
            # need to be generator function that yields deltas
            # Use wrapper to track close
            class Gen:
                def __iter__(self):
                    for d in deltas:
                        yield d
                def close(self):
                    close_tracker["closed"] = True
            return Gen()

        collected = []

        def collector(d):
            collected.append(d)

        # build must return llm path
        fake_built = {
            "path": "llm",
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            "system_prompt": "sys",
            "query": "hi",
        }

        chat_intent = {"intent": "chat", "steps": [], "query": "hi", "task_description": "", "original": "hi"}

        with patch.object(brain, "_build_chat_messages", return_value=fake_built), \
             patch.object(brain, "_stream_chat_deltas", side_effect=fake_stream), \
             patch.object(brain, "classify_intent", return_value=chat_intent), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "force_research", return_value=False), \
             patch.object(brain, "is_task_request", return_value=False), \
             patch.object(brain, "is_explicit_task_request", return_value=False), \
             patch.object(brain, "is_code_tool_request", return_value=False):

            resp = brain.process_message("hi", sync_voice=False, stream_reply=collector)

        # deltas reached stream
        self.assertEqual(collected, deltas)
        # response is assembled
        self.assertEqual(resp, "Hello sir.")
        # history should have user + assistant exactly once, no duplicate user
        hist = self.get_history()
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["role"], "user")
        self.assertEqual(hist[0]["content"], "hi")
        self.assertEqual(hist[1]["role"], "assistant")
        self.assertEqual(hist[1]["content"], "Hello sir.")
        self.assertFalse(close_tracker["closed"], "chat path should not have closed due to cancel")

    def test_task_verdict_cancels_and_close_called(self):
        deltas = ["Hello ", "sir."]
        close_tracker = {"closed": False}

        def fake_stream(messages, temperature, max_tokens, cancel=None):
            class Gen:
                def __iter__(self):
                    # infinite so racer stays alive until cancel
                    while True:
                        for d in deltas:
                            yield d
                def close(self):
                    close_tracker["closed"] = True
            return Gen()

        collected = []
        def collector(d):
            collected.append(d)

        fake_built = {
            "path": "llm",
            "messages": [{"role": "system", "content": "sys"}],
            "system_prompt": "sys",
            "query": "hi",
        }
        task_intent = {"intent": "task", "task_description": "create folder x", "steps": [], "query": "hi", "original": "hi"}

        with patch.object(brain, "_build_chat_messages", return_value=fake_built), \
             patch.object(brain, "_stream_chat_deltas", side_effect=fake_stream), \
             patch.object(brain, "classify_intent", return_value=task_intent), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "force_research", return_value=False), \
             patch.object(brain, "is_task_request", return_value=False), \
             patch.object(brain, "is_explicit_task_request", return_value=False), \
             patch.object(brain, "is_code_tool_request", return_value=False), \
             patch.object(brain, "handle_opencode_task", return_value="Handing the task to opencode, sir.") as mock_task:

            resp = brain.process_message("create folder x", sync_voice=False, stream_reply=collector)
            # give racer thread time to notice cancel and close
            time.sleep(0.15)

        self.assertEqual(collected, [], "task verdict must not leak stream")
        self.assertTrue(close_tracker["closed"], "cancel must close generator")
        self.assertEqual(resp, "Handing the task to opencode, sir.")
        # no assistant commit (handle_chat not called)
        hist = self.get_history()
        # for task, no user+assistant via chat path - check no assistant entry with deltas
        self.assertFalse(any(h["role"] == "assistant" and "Hello" in h["content"] for h in hist))
        mock_task.assert_called_once()

    def test_tool_verdict_cancels(self):
        deltas = ["tool delta"]
        close_tracker = {"closed": False}
        def fake_stream(messages, temperature, max_tokens, cancel=None):
            class Gen:
                def __iter__(self):
                    while True:
                        for d in deltas:
                            yield d
                def close(self):
                    close_tracker["closed"] = True
            return Gen()
        collected = []
        def collector(d):
            collected.append(d)
        fake_built = {"path": "llm", "messages": [{"role":"system","content":"sys"}], "system_prompt":"sys", "query":"hi"}
        tool_intent = {"intent": "tool", "steps": [{"action":"open_website","input":"youtube.com","browser":None}], "query":"hi", "task_description":"", "original":"open youtube"}

        with patch.object(brain, "_build_chat_messages", return_value=fake_built), \
             patch.object(brain, "_stream_chat_deltas", side_effect=fake_stream), \
             patch.object(brain, "classify_intent", return_value=tool_intent), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "force_research", return_value=False), \
             patch.object(brain, "is_task_request", return_value=False), \
             patch.object(brain, "is_explicit_task_request", return_value=False), \
             patch.object(brain, "is_code_tool_request", return_value=False), \
             patch.object(brain, "handle_tool_intent", return_value="Opening youtube, sir.") as mock_tool:

            resp = brain.process_message("open youtube", sync_voice=False, stream_reply=collector)
            time.sleep(0.15)

        self.assertEqual(collected, [])
        self.assertTrue(close_tracker["closed"])
        self.assertEqual(resp, "Opening youtube, sir.")
        hist = self.get_history()
        self.assertFalse(any("tool delta" in h["content"] for h in hist))

    def test_is_task_request_fallthrough_cancels(self):
        deltas = ["leak"]
        close_tracker = {"closed": False}
        def fake_stream(messages, temperature, max_tokens, cancel=None):
            class Gen:
                def __iter__(self):
                    while True:
                        for d in deltas:
                            yield d
                def close(self):
                    close_tracker["closed"] = True
            return Gen()
        collected = []
        def collector(d):
            collected.append(d)
        fake_built = {"path": "llm", "messages": [{"role":"system","content":"sys"}], "system_prompt":"sys", "query":"hi"}
        chat_intent = {"intent": "chat", "steps": [], "query": "hi", "task_description": "", "original": "open chrome and automate login"}

        with patch.object(brain, "_build_chat_messages", return_value=fake_built), \
             patch.object(brain, "_stream_chat_deltas", side_effect=fake_stream), \
             patch.object(brain, "classify_intent", return_value=chat_intent), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "force_research", return_value=False), \
             patch.object(brain, "is_task_request", return_value=True), \
             patch.object(brain, "is_explicit_task_request", return_value=False), \
             patch.object(brain, "is_code_tool_request", return_value=False), \
             patch.object(brain, "handle_opencode_task", return_value="handing off") as mock_handoff, \
             patch.object(brain, "handle_task_message", return_value="task handled via heuristic") as mock_task:

            resp = brain.process_message("open chrome and automate login", sync_voice=False, stream_reply=collector)
            time.sleep(0.15)

        self.assertEqual(collected, [], "is_task_request fallthrough must not leak")
        self.assertTrue(close_tracker["closed"])
        # Web-shaped under the default browser_agent engine -> handoff, not raw task message.
        self.assertEqual(resp, "handing off")
        mock_handoff.assert_called_once()
        mock_task.assert_not_called()

    def test_classify_failure_fallback_streams(self):
        deltas = ["fallback ", "ok"]
        collected = []
        def collector(d):
            collected.append(d)
        fake_built = {"path": "llm", "messages": [{"role":"system","content":"sys"}], "system_prompt":"sys", "query":"hi"}
        fallback = {"intent": "chat", "steps": [], "query": "hi", "task_description": "", "original": "hi"}

        def fake_stream(messages, temperature, max_tokens, cancel=None):
            class Gen:
                def __iter__(self):
                    for d in deltas:
                        yield d
                def close(self):
                    pass
            return Gen()

        with patch.object(brain, "_build_chat_messages", return_value=fake_built), \
             patch.object(brain, "_stream_chat_deltas", side_effect=fake_stream), \
             patch.object(brain, "classify_intent", return_value=fallback), \
             patch.object(brain, "maybe_handle_screen_control_message", return_value=None), \
             patch.object(brain, "force_research", return_value=False), \
             patch.object(brain, "is_task_request", return_value=False), \
             patch.object(brain, "is_explicit_task_request", return_value=False), \
             patch.object(brain, "is_code_tool_request", return_value=False):

            resp = brain.process_message("hello", sync_voice=False, stream_reply=collector)

        self.assertEqual(collected, deltas)
        self.assertEqual(resp, "fallback ok")
        hist = self.get_history()
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[1]["content"], "fallback ok")

    def test_browser_search_prebuilt_path(self):
        # directly test handle_chat with browser_search prebuilt
        from backend.core.memory import clear_history
        clear_history()
        collected = []
        def collector(d):
            collected.append(d)
        prebuilt = {"path": "browser_search", "system_prompt": "sys", "query": "what is 2+2?", "search_info": None}
        with patch.object(brain, "execute_multiple", return_value=["ok"]) as mock_exec:
            resp = brain.handle_chat("what is 2+2?", stream=collector, prebuilt=prebuilt, live_stream=None)
        # R14: no unverified "I've opened a search for you" — the lookup ran
        # synchronously, so the reply reports the fact, not a promise.
        self.assertEqual(resp, "Sir, I ran a search for that.")
        self.assertEqual(collected, [resp])
        mock_exec.assert_called_once_with([{"action": "search", "input": "what is 2+2?"}])
        hist = self.get_history()
        # handle_chat should have committed user + assistant exactly once
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["role"], "user")
        self.assertEqual(hist[1]["content"], resp)

    def test_build_chat_messages_voice_compact_trims_history(self):
        # MAX_HISTORY history messages -> only last 6 in LLM messages
        from backend.core.memory import clear_history, add_message, get_history
        from backend.core import memory as memory_mod
        clear_history()
        pairs = memory_mod.MAX_HISTORY // 2 + 10
        for i in range(pairs):
            add_message("user", f"msg {i}")
            add_message("assistant", f"reply {i}")
        # get_history returns last MAX_HISTORY (capped)
        all_hist = get_history()
        self.assertEqual(len(all_hist), memory_mod.MAX_HISTORY)
        # voice_compact True -> only last 6
        with patch.object(brain, "search_internet", return_value=None):
            built = brain._build_chat_messages("final question", voice_compact=True)
        # built messages = system + last6 history (no search_info)
        msgs = built["messages"]
        # first is system
        self.assertEqual(msgs[0]["role"], "system")
        history_part = msgs[1:]
        self.assertEqual(len(history_part), 6)
        # should be last 6 of all_hist
        expected = all_hist[-6:]
        self.assertEqual(history_part, expected)
        # non-voice keeps full MAX_HISTORY
        with patch.object(brain, "search_internet", return_value=None):
            built2 = brain._build_chat_messages("final question", voice_compact=False)
        msgs2 = built2["messages"]
        self.assertEqual(len(msgs2[1:]), memory_mod.MAX_HISTORY)
        self.assertEqual(msgs2[1:], all_hist)
        # also check search_info branch with voice_compact
        with patch.object(brain, "search_internet", return_value="some search info that is long enough to trigger browser? no"):
            # force search via should_search? easier mock force_search
            with patch.object(brain, "force_search", return_value=True), patch.object(brain, "search_internet", return_value="This is a long search info result that is definitely longer than 20 chars."):
                built3 = brain._build_chat_messages("searchy", voice_compact=True)
                # when search_info present, messages = system + history[:-1] + user+search
                # history is trimmed to last 6 first
                trimmed = all_hist[-6:]
                expected_len = len(trimmed[:-1]) + 1  # +1 for final user+search message + system? actually system already
                # msgs = [system, *trimmed[:-1], {user+search}]
                self.assertEqual(len(built3["messages"]), 1 + len(trimmed[:-1]) + 1)
                self.assertEqual(built3["messages"][1:-1], trimmed[:-1])

    def test_intent_prompt_length_and_keys(self):
        p = intent_mod._INTENT_PROMPT
        self.assertLess(len(p), 2000, f"prompt len {len(p)} must be <2000")
        for name in ["chat", "tool", "screen", "region", "research", "task"]:
            self.assertIn(name, p)
        for action in ["open_website", "launch_app", "youtube_play", "search"]:
            self.assertIn(action, p)
        for key in ["intent", "steps", "query", "task_description", "original"]:
            self.assertIn(key, p)
        # must contain meaning across languages hint and hinglish example
        self.assertIn("Hinglish", p)
        self.assertIn("how are you", p.lower())

    def test_racer_thread_joinable_no_sleep(self):
        # verify racer exposes join and finishes deterministically with fake stream returning immediately
        fake_built = {"path": "llm", "messages": [{"role":"system","content":"sys"}], "system_prompt":"sys", "query":"hi"}
        def fake_stream(messages, temperature, max_tokens, cancel=None):
            class Gen:
                def __iter__(self):
                    yield "a"
                    yield "b"
                def close(self):
                    pass
            return Gen()
        with patch.object(brain, "_build_chat_messages", return_value=fake_built), \
             patch.object(brain, "_stream_chat_deltas", side_effect=fake_stream):
            racer = brain._ChatRacer("hi", voice_compact=False)
            # join should succeed quickly, no sleeps determinism
            done = racer.join(timeout=1.0)
            self.assertTrue(done)
            self.assertTrue(racer.is_done)
            built = racer.built()
            self.assertEqual(built, fake_built)
            stream = racer.adopt()
            collected = list(stream)
            self.assertEqual(collected, ["a", "b"])
            # cancel after done should be safe
            racer.cancel()
            self.assertTrue(racer.is_done)

    def test_racer_appends_user_turn_when_history_empty(self):
        # Regression: racer builds messages before handle_chat adds the user
        # message to history; with empty history the LLM request had NO user
        # turn at all (Gemini 400 / degenerate fallback output).
        from backend.core.memory import clear_history
        clear_history()
        captured = {}

        def fake_stream(messages, temperature, max_tokens, cancel=None):
            captured["messages"] = messages
            class Gen:
                def __iter__(self):
                    yield "ok"
                def close(self):
                    pass
            return Gen()

        with patch.object(brain, "_stream_chat_deltas", side_effect=fake_stream):
            racer = brain._ChatRacer("tell me a joke", voice_compact=False)
            self.assertTrue(racer.join(timeout=1.0))

        msgs = racer.built()["messages"]
        self.assertTrue(msgs, "messages must not be empty")
        self.assertEqual(msgs[-1]["role"], "user")
        # [S12] the latest user turn also carries the current date/time note.
        self.assertTrue(msgs[-1]["content"].startswith("tell me a joke"))
        self.assertEqual(captured["messages"][-1]["role"], "user")
        self.assertTrue(
            captured["messages"][-1]["content"].startswith("tell me a joke"))

    def test_racer_appends_user_turn_when_history_ends_with_assistant(self):
        # Regression: history ending in an assistant turn made Gemini reject
        # the request ("Requests ending with a model turn are not supported").
        from backend.core.memory import add_message
        add_message("user", "previous question")
        add_message("assistant", "previous reply")
        captured = {}

        def fake_stream(messages, temperature, max_tokens, cancel=None):
            captured["messages"] = messages
            class Gen:
                def __iter__(self):
                    yield "ok"
                def close(self):
                    pass
            return Gen()

        with patch.object(brain, "_stream_chat_deltas", side_effect=fake_stream):
            racer = brain._ChatRacer("tell me a joke", voice_compact=False)
            self.assertTrue(racer.join(timeout=1.0))

        msgs = racer.built()["messages"]
        self.assertEqual(msgs[-1]["role"], "user")
        self.assertTrue(msgs[-1]["content"].startswith("tell me a joke"))

    def test_racer_no_duplicate_user_turn_when_already_present(self):
        # Search branch already ends with the user turn (msg + search info):
        # the racer must not append a second copy.
        fake_built = {
            "path": "llm",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "what is ai\n\nSearch info: some results"},
            ],
            "system_prompt": "sys",
            "query": "what is ai",
        }
        orig_len = len(fake_built["messages"])
        captured = {}

        def fake_stream(messages, temperature, max_tokens, cancel=None):
            captured["messages"] = messages
            class Gen:
                def __iter__(self):
                    yield "ok"
                def close(self):
                    pass
            return Gen()

        with patch.object(brain, "_build_chat_messages", return_value=fake_built), \
             patch.object(brain, "_stream_chat_deltas", side_effect=fake_stream):
            racer = brain._ChatRacer("what is ai", voice_compact=False)
            self.assertTrue(racer.join(timeout=1.0))

        self.assertEqual(len(fake_built["messages"]), orig_len)
        self.assertEqual(captured["messages"][-1]["content"], "what is ai\n\nSearch info: some results")

