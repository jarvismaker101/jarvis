"""G3 Round 1 (F23/F25/F26/F30): request identity, one event protocol, pure
cancellable speculation, and screen answers published before decorations.

  F23 — execute once per request id; a reconnecting client resumes after its
        last event instead of re-running the message.
  F25 — speculation is pure (no search, no external effect), builds from an
        immutable snapshot, uses a bounded queue, and cancels promptly.
  F26 — one event protocol (delta/replace/progress/completed/interrupted/
        error); the terminal reply is authoritative; decisions that change the
        reply happen before anything is spoken.
  F30 — screen answers are published as soon as vision returns; decorations
        arrive later in a bounded phase that patches the same answer id.
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from backend.api import routes
from backend.core import brain
from backend.services import request_registry


# ─────────────────────────────────────────────────────────────────────────────
# F23 + F26 — request registry / event protocol
# ─────────────────────────────────────────────────────────────────────────────
class RequestRegistryTests(unittest.TestCase):
    """Execute-once identity and the numbered event buffer."""

    def setUp(self):
        # The registry is a process-wide singleton — every test needs its own
        # id so results never bleed between tests.
        self.state, created = request_registry.get_or_create(
            request_registry.new_request_id(), "hello")
        self.assertTrue(created)

    def test_execute_once_guard(self):
        same, created_again = request_registry.get_or_create(
            self.state.request_id, "hello")
        self.assertFalse(created_again)
        self.assertIs(same, self.state)
        self.assertTrue(request_registry.try_start(self.state))
        # A second attach (retry or reconnect) must NOT execute again.
        self.assertFalse(request_registry.try_start(self.state))

    def test_frames_are_numbered_and_tagged(self):
        self.state.delta("Hel")
        self.state.delta("lo")
        self.assertEqual([seq for seq, _ in self.state.events_after(-1)],
                         [0, 1])
        frame = self.state.events_after(0)[0][1]
        self.assertEqual(frame["type"], request_registry.DELTA)
        self.assertEqual(frame["request_id"], self.state.request_id)
        self.assertEqual(frame["text"], "lo")

    def test_resume_after_last_event(self):
        for i in range(5):
            self.state.delta("d%d" % i)
        resumed = self.state.events_after(2)
        self.assertEqual([seq for seq, _ in resumed], [3, 4])

    def test_terminal_frame_is_authoritative(self):
        self.state.delta("partial")
        self.state.complete("the final answer")
        self.assertTrue(self.state.done)
        self.assertEqual(self.state.reply, "the final answer")
        self.assertEqual(self.state.events[-1][1]["type"],
                         request_registry.COMPLETED)

    def test_terminal_frame_survives_buffer_trim(self):
        # A disconnected client must still be able to reach the terminal
        # frame even after the buffer hits its bound.
        for i in range(request_registry.EVENT_LIMIT + 20):
            self.state.delta("x%d" % i)
        self.state.complete("done")
        self.assertEqual(self.state.events[-1][1]["type"],
                         request_registry.COMPLETED)
        self.assertTrue(self.state.done)

    def test_stream_replays_only_newer_events_then_ends(self):
        self.state.delta("a")
        self.state.delta("b")
        self.state.complete("ab")
        frames = list(self.state.stream(last_seq=0, heartbeat=0.05))
        self.assertEqual([f["type"] for f in frames],
                         [request_registry.DELTA, request_registry.COMPLETED])
        self.assertEqual(frames[0]["text"], "b")

    def test_heartbeat_keeps_quiet_stream_alive(self):
        frames = []
        gen = self.state.stream(last_seq=-1, heartbeat=0.05)
        try:
            frames.append(next(gen))
        finally:
            gen.close()
        self.assertEqual(frames[0]["type"], request_registry.PROGRESS)
        self.assertTrue(frames[0].get("heartbeat"))
        # Heartbeats are not numbered events and never advance the cursor.
        self.assertIsNone(frames[0]["seq"])

    def test_replace_frame(self):
        self.state.delta("old")
        self.state.replace("new")
        frame = self.state.events[-1][1]
        self.assertEqual(frame["type"], request_registry.REPLACE)
        self.assertEqual(frame["text"], "new")

    def test_interrupt_active_terminalises_inflight(self):
        state, _ = request_registry.get_or_create(
            request_registry.new_request_id(), "hi")
        request_registry.try_start(state)
        state.delta("working")
        self.assertGreaterEqual(
            request_registry.interrupt_active("stopped by user"), 1)
        self.assertTrue(state.done)
        self.assertEqual(state.events[-1][1]["type"],
                         request_registry.INTERRUPTED)


# ─────────────────────────────────────────────────────────────────────────────
# F23 + F26 — routes
# ─────────────────────────────────────────────────────────────────────────────
class AskRouteTests(unittest.TestCase):
    """A retried /ask must not execute the message twice (F23)."""

    def setUp(self):
        self._patches = [
            patch.object(routes, "stop_speaking"),
            patch.object(routes, "set_narration_enabled"),
            patch.object(routes, "_maybe_speak"),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_retry_with_same_request_id_does_not_reexecute(self):
        with patch.object(routes, "process_message", return_value="first reply") as pm:
            rid = request_registry.new_request_id()
            first = routes.ask(routes.Query(message="open notepad",
                                            request_id=rid))
            self.assertEqual(first["reply"], "first reply")
            self.assertEqual(first["request_id"], rid)

            second = routes.ask(routes.Query(message="open notepad",
                                             request_id=rid))
            self.assertEqual(second["reply"], "first reply")
            self.assertEqual(pm.call_count, 1,
                             "a retry must attach to the finished request, "
                             "not run the action again")

    def test_status_lookup_for_known_request(self):
        rid = request_registry.new_request_id()
        state, _ = request_registry.get_or_create(rid, "hi")
        request_registry.try_start(state)
        state.complete("done already")
        snapshot = routes.ask_status(rid)
        self.assertTrue(snapshot["done"])
        self.assertEqual(snapshot["reply"], "done already")

    def test_status_lookup_for_unknown_request_raises_404(self):
        with self.assertRaises(HTTPException) as ctx:
            routes.ask_status("req-does-not-exist")
        self.assertEqual(ctx.exception.status_code, 404)


# ─────────────────────────────────────────────────────────────────────────────
# F25 — pure, cancellable speculation
# ─────────────────────────────────────────────────────────────────────────────
class SpeculationPurityTests(unittest.TestCase):
    """The speculative build must have no external effect (F25)."""

    def test_speculative_build_never_searches(self):
        with patch.object(brain, "force_search", return_value=True), \
             patch.object(brain, "search_internet") as search:
            built = brain._build_chat_messages("latest news", speculative=True)
        search.assert_not_called()
        self.assertEqual(built["path"], "needs_search")

    def test_selected_build_still_acquires_the_lookup(self):
        with patch.object(brain, "force_search", return_value=True), \
             patch.object(brain, "search_internet",
                          return_value="a" * 40) as search:
            built = brain._build_chat_messages("latest news")
        search.assert_called_once()
        self.assertEqual(built["path"], "llm")

    def test_snapshot_history_is_used_verbatim(self):
        snapshot = [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "two"},
        ]
        with patch.object(brain, "should_search", return_value=False), \
             patch.object(brain, "force_search", return_value=False), \
             patch.object(brain, "get_history", return_value=[]):
            built = brain._build_chat_messages("three", history=snapshot)
        # system + the two snapshot turns, regardless of live history.
        self.assertEqual(built["messages"][1:], snapshot)

    def test_racer_does_not_search(self):
        with patch.object(brain, "force_search", return_value=True), \
             patch.object(brain, "search_internet") as search, \
             patch.object(brain, "_stream_chat_deltas", return_value=iter([])):
            racer = brain._ChatRacer("latest news", voice_compact=False)
            self.assertTrue(racer.join(timeout=2.0))
        search.assert_not_called()
        self.assertEqual(racer.built()["path"], "needs_search")
        self.assertFalse(racer.has_stream())

    def test_racer_builds_from_an_immutable_snapshot(self):
        snapshot = [{"role": "user", "content": "earlier question"}]
        with patch.object(brain, "should_search", return_value=False), \
             patch.object(brain, "force_search", return_value=False), \
             patch.object(brain, "get_history", return_value=snapshot) as live:
            racer = brain._ChatRacer("new question", voice_compact=False)
            self.assertTrue(racer.join(timeout=2.0))
        live.assert_called()
        msgs = racer.built()["messages"]
        self.assertIn({"role": "user", "content": "earlier question"}, msgs)
        # the new turn is appended by the racer, not read from live history
        # (the [S12] current date/time note rides along with it)
        self.assertTrue(msgs[-1]["content"].startswith("new question"))

    def test_racer_queue_is_bounded(self):
        racer = brain._ChatRacer("hi", voice_compact=False)
        self.assertEqual(racer._queue.maxsize, brain._RACER_QUEUE_LIMIT)
        racer.cancel()
        racer.join(timeout=2.0)

    def test_cancel_sets_event_and_releases_consumer(self):
        gate = threading.Event()

        def slow_stream(messages, temperature, max_tokens, cancel=None):
            gate.wait(5.0)
            yield "late delta"

        with patch.object(brain, "_stream_chat_deltas", side_effect=slow_stream):
            racer = brain._ChatRacer("hi", voice_compact=False)
            racer.cancel()
            # adopt() must return promptly instead of waiting for the socket
            started = time.time()
            collected = list(racer.adopt())
            self.assertLess(time.time() - started, 2.0)
        self.assertEqual(collected, [])
        self.assertTrue(racer._cancel.is_set())
        gate.set()

    def test_cancel_closes_the_underlying_stream(self):
        closed = threading.Event()
        started = threading.Event()
        gate = threading.Event()

        class Stream:
            def __iter__(self):
                started.set()
                yield "x"
                gate.wait(5.0)
                yield "y"

            def close(self):
                closed.set()

        with patch.object(brain, "_stream_chat_deltas", return_value=Stream()):
            racer = brain._ChatRacer("hi", voice_compact=False)
            self.assertTrue(started.wait(2.0), "stream never started")
            racer.cancel()
        gate.set()
        self.assertTrue(racer.join(timeout=3.0))
        self.assertTrue(closed.is_set(), "cancelling must close the stream")


# ─────────────────────────────────────────────────────────────────────────────
# F26 — one streaming contract
# ─────────────────────────────────────────────────────────────────────────────
class StreamingContractTests(unittest.TestCase):
    """Terminal authority and decision-before-speech (F26)."""

    def _prebuilt(self):
        return {"path": "llm", "messages": [{"role": "user", "content": "hi"}]}

    def test_unsure_answer_is_not_swapped_after_being_spoken(self):
        collected = []

        def stream(text):
            collected.append(text)

        with patch.object(brain, "_stream_chat_deltas",
                          return_value=iter(["I don't know that, sir."])), \
             patch.object(brain, "maybe_proactive_research") as proactive:
            reply = brain.handle_chat("question", stream=stream,
                                      commit_response=False,
                                      prebuilt=self._prebuilt())
        self.assertEqual(collected, ["I don't know that, sir."])
        # Spoken text and stored reply must agree — no silent swap to a
        # different permission question after the answer was spoken.
        self.assertEqual(reply, "I don't know that, sir.")
        proactive.assert_not_called()

    def test_unsure_answer_becomes_clarification_when_nothing_was_spoken(self):
        with patch.object(brain, "_ask_chat_nonstream",
                          return_value={"choices": [{"message": {
                              "content": "I don't know that, sir."}}]}), \
             patch.object(brain, "maybe_proactive_research", return_value=True):
            reply = brain.handle_chat("question", commit_response=False,
                                      prebuilt=self._prebuilt())
        self.assertEqual(reply, brain._confirmation_question())


# ─────────────────────────────────────────────────────────────────────────────
# F30 — screen answers before decorations
# ─────────────────────────────────────────────────────────────────────────────
class ScreenAnswerDecorationTests(unittest.TestCase):
    """Publish early, enrich later, never overwrite a newer question (F30)."""

    def test_publish_then_patch_same_answer_id(self):
        first = routes.post_screen_answer(
            routes.ScreenAnswer(tip="tip", capture_id="cap-1"))
        answer_id = first["id"]
        self.assertEqual(first["revision"] if "revision" in first else 0, 0)

        patched = routes.post_screen_answer(
            routes.ScreenAnswer(tip="tip", capture_id="cap-1", id=answer_id,
                                links=[routes.LinkItem(label="More",
                                                       url="http://x")]))
        # Same answer id — the overlay refreshes in place instead of showing a
        # second, apparently newer card.
        self.assertEqual(patched["id"], answer_id)
        self.assertTrue(patched["updated"])
        self.assertEqual(patched["revision"], 1)

        stored = routes.get_screen_answer()
        self.assertEqual(stored["id"], answer_id)
        self.assertEqual(len(stored["links"]), 1)
        self.assertTrue(stored["enriched"])

    def test_patch_to_an_old_answer_id_is_rejected(self):
        first = routes.post_screen_answer(
            routes.ScreenAnswer(tip="first", capture_id="cap-a"))
        routes.post_screen_answer(
            routes.ScreenAnswer(tip="second", capture_id="cap-b"))
        with self.assertRaises(HTTPException) as ctx:
            routes.post_screen_answer(
                routes.ScreenAnswer(tip="stale", id=first["id"],
                                    capture_id="cap-a"))
        self.assertEqual(ctx.exception.status_code, 409)

    def test_late_decoration_cannot_overwrite_a_newer_question(self):
        routes.post_screen_answer(
            routes.ScreenAnswer(tip="old question", capture_id="cap-old"))
        newest = routes.post_screen_answer(
            routes.ScreenAnswer(tip="new question", capture_id="cap-new"))
        with self.assertRaises(HTTPException) as ctx:
            routes.post_screen_answer(
                routes.ScreenAnswer(tip="old question", id=newest["id"],
                                    capture_id="cap-old",
                                    images=[routes.ImageItem(url="http://i")]))
        self.assertEqual(ctx.exception.status_code, 409)
        stored = routes.get_screen_answer()
        self.assertEqual(stored["tip"], "new question")
        self.assertEqual(stored["images"], [])

    def test_screen_intent_publishes_before_decorations(self):
        result = {
            "tip": "You are looking at a code editor.",
            "evidence": [{"source": "screen", "snippet": "def foo()"}],
            "topic": "python function",
            "grounding_links": [],
            "show_images": True,
            "region": None,
        }

        def fake_explore(topic):
            return [{"label": "Docs", "url": "http://docs", "icon": "🔗"}]

        def fake_images(topic, max_images=2):
            return [{"url": "http://img", "title": "img"}]

        with patch.object(brain, "classify_intent",
                          return_value={"intent": "screen"}), \
             patch.object(brain, "maybe_handle_screen_control_message",
                          return_value=None), \
             patch.object(brain, "is_task_request", return_value=False), \
             patch.object(brain, "analyze_screen", return_value=result), \
             patch.object(brain, "build_explore_links",
                          side_effect=fake_explore) as explore, \
             patch.object(brain, "fetch_topic_images",
                          side_effect=fake_images) as images, \
             patch.object(brain, "push_screen_answer") as push:
            reply = brain.process_message("what's on my screen",
                                          sync_voice=False)
            # The answer goes out on the first push — decorations are not part
            # of it yet.
            self.assertEqual(reply, "You are looking at a code editor.")
            self.assertGreaterEqual(push.call_count, 1)
            first_call = push.call_args_list[0]
            # Decorations are NOT part of the first push: images are the 4th
            # positional argument and must still be empty.
            self.assertEqual(first_call.args[3], [])
            self.assertTrue(first_call.kwargs.get("capture_id"))

            # The decoration phase runs afterwards, in the background.
            deadline = time.time() + 3.0
            while time.time() < deadline and push.call_count < 2:
                time.sleep(0.02)
            self.assertGreaterEqual(push.call_count, 2)
            explore.assert_called_once()
            images.assert_called_once()
            second_call = push.call_args_list[1]
            self.assertEqual(second_call.kwargs.get("capture_id"),
                             first_call.kwargs.get("capture_id"))
            self.assertTrue(second_call.kwargs.get("answer_id"))

    def test_progress_is_reported_while_vision_runs(self):
        result = {
            "tip": "tip", "evidence": [], "topic": "", "grounding_links": [],
            "show_images": False, "region": None,
        }
        seen = []
        with patch.object(brain, "classify_intent",
                          return_value={"intent": "screen"}), \
             patch.object(brain, "maybe_handle_screen_control_message",
                          return_value=None), \
             patch.object(brain, "is_task_request", return_value=False), \
             patch.object(brain, "analyze_screen", return_value=result), \
             patch.object(brain, "push_screen_answer", return_value=1):
            brain.process_message("what's on my screen", sync_voice=False,
                                  progress=lambda m, **kw: seen.append(m),
                                  request_id="req-screen-1")
        self.assertIn("analysing screen", seen)

    def test_enrichment_deadline_ships_links_without_images(self):
        """F30 — out of budget means "links only", not "discard the links".

        The enrichment phase must never hold the answer back, but it also
        must not throw away decoration work it already completed just
        because the image phase would have overrun the deadline.
        """
        calls = []
        with patch.object(brain, "SCREEN_ENRICH_TIMEOUT", -1), \
             patch.object(brain, "build_explore_links",
                          return_value=[{"label": "More", "url": "http://x"}]), \
             patch.object(brain, "fetch_topic_images",
                          side_effect=AssertionError("images must be skipped")), \
             patch.object(brain, "push_screen_answer",
                          side_effect=lambda *a, **kw: calls.append((a, kw))):
            brain._enrich_screen_answer(
                7, "cap-1", "req-1", "tip", [], [], "topic", True, {})

        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        # push_screen_answer(tip, evidence, links, images, …) — links and
        # images are passed positionally.
        # The freshly built exploration links are shipped…
        self.assertEqual(len(args[2]), 1)
        # …and the slow image phase is skipped entirely.
        self.assertEqual(args[3], [])
        self.assertEqual(kwargs["answer_id"], 7)
        self.assertEqual(kwargs["capture_id"], "cap-1")


if __name__ == "__main__":
    unittest.main()
