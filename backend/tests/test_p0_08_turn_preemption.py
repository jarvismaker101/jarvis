"""P0-08 — barge-in cancels the backend turn, and the reply still lands in
history.

Pinned defects:

  * ``/speak/stop`` only stopped the *audio*; nothing cancelled the running
    request or its job, so the old generation kept going;
  * ``brain_thread`` handled ONE utterance at a time and blocked in
    ``_ask_backend`` until the OLD stream reached its terminal frame, so every
    interruption waited for the reply it was interrupting (and a non-streamed
    reply voiced the OLD answer first).

User decision, pinned here so nobody "fixes" it by accident: an interrupted
turn still commits its FULL generated reply to history. Cancelling is about
audio and pending tool work, never about rewriting what was stored.

Everything is stubbed: no network, no model, no microphone.
"""

import threading
import time
import unittest
from unittest.mock import patch

from backend import voice_mode as vm
from backend.services import request_registry as req_registry
from backend.services import jobs as job_registry


class _SpeakerStub:
    """Records what a turn tried to say, without touching an audio device."""

    def __init__(self):
        self.fed = []
        self.closed = False
        self.finished = False

    def feed(self, delta):
        self.fed.append(delta)

    def close(self):
        self.closed = True

    def finish(self):
        self.finished = True

    @property
    def spoken_any(self):
        return bool(self.fed)


class _TurnHarness(unittest.TestCase):
    def setUp(self):
        self._prev = vm.TURNS
        vm.TURNS = vm._TurnManager()
        self.cancels = []
        self._patches = [
            patch.object(vm, "_cancel_backend_request_async",
                         side_effect=lambda rid, reason: self.cancels.append(
                             (rid, reason)) or True),
            patch.object(vm, "_cancel_backend_request",
                         side_effect=lambda rid, reason="": self.cancels.append(
                             (rid, reason)) or True),
            patch.object(vm, "set_active_stream"),
            patch.object(vm, "get_active_stream", return_value=None),
            patch.object(vm, "_watch_turn_for_shipping"),
            patch.object(vm, "speak"),
            patch.object(vm, "listener_state"),
            patch.object(vm, "_latency"),
            # A real backend probe would make one HTTP call per submit and the
            # test would measure that, not the dispatcher.
            patch.object(vm, "backend_task_running", return_value=False),
            patch.object(vm, "StreamSpeaker", _SpeakerStub),
        ]
        for item in self._patches:
            item.start()
            self.addCleanup(item.stop)

    def tearDown(self):
        vm.TURNS = self._prev
        for _ in range(5):
            try:
                vm.command_queue.get_nowait()
            except Exception:
                break

    def _wait(self, predicate, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()


class BargeInDoesNotWaitTests(_TurnHarness):
    """The regression: the new turn starts without waiting for the old one."""

    def test_submitting_an_utterance_does_not_block_on_a_slow_reply(self):
        """Pre-fix this call sat inside _ask_backend for the whole reply."""
        released = threading.Event()

        def slow_ask(text, request_id, stream_sink=None, client_marks=None,
                      **kw):
            released.wait(5)
            return "old reply"

        with patch.object(vm, "_ask_backend", side_effect=slow_ask):
            started = time.perf_counter()
            vm._respond_to_utterance("first question")
            elapsed = time.perf_counter() - started
            self.addCleanup(released.set)

        self.assertLess(elapsed, 0.5,
                        "the dispatcher waited for the generation (%.2fs)"
                        % elapsed)

    def test_a_new_utterance_reaches_the_backend_while_the_old_one_runs(self):
        """Acceptance: mid-reply interrupt, no wait for the old answer."""
        first_started = threading.Event()
        release_first = threading.Event()
        seen = []

        def ask(text, request_id, stream_sink=None, client_marks=None, **kw):
            seen.append(text)
            if text == "first question":
                first_started.set()
                release_first.wait(5)
            return "reply to %s" % text

        with patch.object(vm, "_ask_backend", side_effect=ask):
            vm._respond_to_utterance("first question")
            self.assertTrue(first_started.wait(2), "the first turn never ran")
            # The user interrupts mid-reply.
            vm._respond_to_utterance("actually, what about X")
            self.assertTrue(
                self._wait(lambda: "actually, what about X" in seen),
                "the new utterance waited for the old turn to finish")
            release_first.set()

        self.assertIn("actually, what about X", seen)
        self.assertGreaterEqual(vm.TURNS.stats["preempted"], 1)
        self.assertTrue(self.cancels, "the old turn was never cancelled")

    def test_the_replaced_turn_is_cancelled_by_request_id(self):
        request_ids = []
        release = threading.Event()

        def ask(text, request_id, stream_sink=None, client_marks=None, **kw):
            request_ids.append(request_id)
            if len(request_ids) == 1:
                release.wait(5)
            return "reply"

        with patch.object(vm, "_ask_backend", side_effect=ask):
            vm._respond_to_utterance("first question")
            self.assertTrue(self._wait(lambda: len(request_ids) == 1))
            vm._respond_to_utterance("second question")
            release.set()

        cancelled_ids = [rid for rid, _ in self.cancels]
        self.assertIn(request_ids[0], cancelled_ids,
                      "the OLD request id must be the one cancelled")

    def test_a_replaced_turns_late_reply_is_not_spoken(self):
        release = threading.Event()
        speakers = []

        def ask(text, request_id, stream_sink=None, client_marks=None, **kw):
            if text == "first question":
                release.wait(5)
            return "reply to %s" % text

        def make_speaker():
            speaker = _SpeakerStub()
            speakers.append(speaker)
            return speaker

        with patch.object(vm, "_ask_backend", side_effect=ask), \
                patch.object(vm, "StreamSpeaker", side_effect=make_speaker):
            vm._respond_to_utterance("first question")
            self.assertTrue(self._wait(lambda: len(speakers) == 1))
            vm._respond_to_utterance("second question")
            self.assertTrue(self._wait(lambda: len(speakers) == 2))
            release.set()
            self.assertTrue(
                self._wait(lambda: vm.TURNS.stats["stale_replies_dropped"] >= 1),
                "the superseded turn was not recognised as stale")

        self.assertTrue(speakers[0].closed,
                        "the replaced turn's audio was never closed")


class TwoVoicesTests(_TurnHarness):
    """One voice at a time: the turn manager owns that invariant."""

    def test_two_rapid_interruptions_leave_exactly_one_current_turn(self):
        speakers = []
        release = threading.Event()

        def make_speaker():
            speaker = _SpeakerStub()
            speakers.append(speaker)
            return speaker

        def ask(text, request_id, stream_sink=None, client_marks=None, **kw):
            # Every turn stays PENDING until released, so the assertions below
            # observe the turn registry rather than thread timing (a completed
            # turn clears itself, which made this race).
            release.wait(3)
            return "reply to %s" % text

        with patch.object(vm, "_ask_backend", side_effect=ask), \
                patch.object(vm, "StreamSpeaker", side_effect=make_speaker):
            vm._respond_to_utterance("one")
            first = vm.TURNS.active_request_id()
            vm.TURNS.cancel_current("barge-in")
            vm._respond_to_utterance("two")
            second = vm.TURNS.active_request_id()
            vm.TURNS.cancel_current("barge-in")
            vm._respond_to_utterance("three")
            third = vm.TURNS.active_request_id()

            self.assertTrue(first and second and third)
            self.assertNotEqual(len({first, second, third}), 1,
                                "each turn must carry its own request id")
            # Exactly one turn is current, and it is the newest.
            self.assertTrue(vm.TURNS.is_current(third))
            self.assertFalse(vm.TURNS.is_current(first))
            self.assertFalse(vm.TURNS.is_current(second))
            # Only the newest speaker may still be live; the older two were
            # closed so their queued audio cannot reach the actor.
            self.assertTrue(speakers[0].closed)
            self.assertTrue(speakers[1].closed)
            self.assertFalse(speakers[2].closed)
            release.set()

        self.assertEqual(vm.TURNS.stats["started"], 3)
        self.assertEqual(vm.TURNS.stats["barge_in_cancels"], 2)

    def test_a_streaming_delta_from_a_stale_turn_is_not_fed_to_the_actor(self):
        speakers = []
        release = threading.Event()
        sink_holder = {}

        def make_speaker():
            speaker = _SpeakerStub()
            speakers.append(speaker)
            return speaker

        def ask(text, request_id, stream_sink=None, client_marks=None, **kw):
            if text == "first question":
                sink_holder["sink"] = stream_sink
                release.wait(5)
            return "reply"

        with patch.object(vm, "_ask_backend", side_effect=ask), \
                patch.object(vm, "StreamSpeaker", side_effect=make_speaker):
            vm._respond_to_utterance("first question")
            self.assertTrue(self._wait(lambda: "sink" in sink_holder))
            vm._respond_to_utterance("second question")   # supersedes turn 1
            # A late delta arrives for the superseded turn.
            sink_holder["sink"]("late words")
            release.set()

        self.assertEqual(speakers[0].fed, [],
                         "a stale turn fed audio to the actor")

    def test_barge_in_with_no_active_turn_is_a_no_op(self):
        self.assertFalse(vm.TURNS.cancel_current("barge-in"))
        self.assertEqual(self.cancels, [])



class CancelEndpointTests(unittest.TestCase):
    """The request-scoped cancel contract, exercised on the registry."""

    def setUp(self):
        from backend.api import routes
        self.routes = routes
        self._made = []

    def tearDown(self):
        for rid in self._made:
            req_registry.REGISTRY._requests.pop(rid, None)

    def _live(self, rid, message):
        state, _created, _conflict = req_registry.admit(rid, message)
        self._made.append(rid)
        return state

    def test_cancelling_one_request_leaves_a_concurrent_one_running(self):
        first = self._live("p0-08-a", "first")
        other = self._live("p0-08-b", "unrelated")
        result = self.routes.cancel_request("p0-08-a", "barge-in")
        self.assertTrue(result["cancelled"])
        self.assertTrue(first.done)
        self.assertTrue(first.interrupted)
        # The unrelated concurrent request is untouched.
        self.assertFalse(other.done)
        self.assertFalse(other.interrupted)

    def test_cancelling_an_unknown_request_is_a_safe_no_op(self):
        result = self.routes.cancel_request("p0-08-never-existed")
        self.assertTrue(result["ok"])
        self.assertFalse(result["cancelled"])
        self.assertEqual(result["reason"], "unknown_request")

    def test_cancelling_an_already_finished_request_never_rewrites_it(self):
        state = self._live("p0-08-done", "already answered")
        state.complete("the full reply")
        result = self.routes.cancel_request("p0-08-done", "barge-in")
        self.assertFalse(result["cancelled"])
        self.assertEqual(result["reason"], "already_finished")
        self.assertEqual(state.reply, "the full reply")
        self.assertFalse(state.interrupted,
                         "a completed turn must not be flipped to interrupted")

    def test_cancel_is_idempotent(self):
        self._live("p0-08-twice", "message")
        first = self.routes.cancel_request("p0-08-twice", "barge-in")
        second = self.routes.cancel_request("p0-08-twice", "barge-in")
        self.assertTrue(first["cancelled"])
        self.assertFalse(second["cancelled"])
        self.assertEqual(second["reason"], "already_interrupted")

    def test_the_cancel_route_never_raises(self):
        with patch.object(req_registry, "get", side_effect=RuntimeError("boom")):
            result = self.routes.cancel_request("p0-08-broken")
        self.assertTrue(result["ok"])
        self.assertFalse(result["cancelled"])
        self.assertEqual(result["reason"], "lookup_failed")



class HistorySurvivesBargeInTests(unittest.TestCase):
    """The user's explicit requirement, pinned end to end."""

    def setUp(self):
        from backend.core import memory
        from backend.api import routes
        self.memory = memory
        self.routes = routes
        self._saved = list(memory.get_history())
        memory.clear_history()

    def tearDown(self):
        self.memory.clear_history()
        for message in self._saved:
            self.memory.add_message(message.get("role"), message.get("content"))
        req_registry.REGISTRY._requests.pop(self.request_id, None)

    def test_the_full_reply_is_still_in_history_after_a_barge_in(self):
        self.request_id = "p0-08-history"
        full_reply = ("Here is the complete answer. It has several sentences, "
                      "because a real reply does. Nothing here is a prefix.")
        started = threading.Event()
        release = threading.Event()

        def fake_process(message, **kwargs):
            # What brain.handle_chat does: commit the assistant turn once the
            # content is produced (unchanged by this audit).
            started.set()
            release.wait(5)
            self.memory.add_message("assistant", full_reply)
            return full_reply

        state, _created, _conflict = req_registry.admit(self.request_id,
                                                       "a question")
        with patch.object(self.routes, "process_message",
                          side_effect=fake_process):
            worker = threading.Thread(
                target=self.routes._run_request_worker,
                kwargs={"state": state, "speak_stream": False,
                        "speak_terminal": False},
                daemon=True,
            )
            worker.start()
            self.assertTrue(started.wait(2), "the worker never started")
            # The user barges in while the reply is being produced.
            result = self.routes.cancel_request(self.request_id, "barge-in")
            self.assertTrue(result["cancelled"])
            release.set()
            worker.join(timeout=5)

        # 1) The cancellation is terminal for the REQUEST...
        self.assertTrue(state.interrupted)
        self.assertIsNone(state.reply,
                          "an interrupted turn must not publish a completion")
        # 2) ...but the reply text was still committed in full.
        assistants = [m["content"] for m in self.memory.get_history()
                      if m.get("role") == "assistant"]
        self.assertEqual(assistants, [full_reply],
                         "the full generated reply must stay readable")

    def test_the_barge_in_marker_is_exposed_without_touching_the_reply(self):
        self.request_id = "p0-08-marker"
        self.routes._note_reply_interrupted(self.request_id, "barge-in")
        marker = self.routes.get_last_reply_interrupted()
        self.assertTrue(marker["interrupted"])
        self.assertEqual(marker["request_id"], self.request_id)
        # Additive only: no reply text lives in the marker.
        self.assertNotIn("response", marker)
        self.assertNotIn("reply", marker)

    def test_a_completed_reply_clears_the_marker(self):
        self.request_id = "p0-08-clear"
        self.routes._note_reply_interrupted("old", "barge-in")
        self.routes._note_reply_completed()
        self.assertFalse(self.routes.get_last_reply_interrupted()["interrupted"])



class ControlPhraseTests(_TurnHarness):
    """A control phrase is a control, never an interrupting utterance.

    Which phrases count is decided by the F35 grammar (``classify_control``),
    which this audit does not touch — the target-aware vocabulary deliberately
    does NOT match a bare "stop", because "stop" alone is ambiguous. These tests
    use phrases the grammar really does own and assert the DISPATCH consequence:
    a control never becomes a turn, because a turn would submit a second
    generation behind the reply the user was trying to silence.
    """

    def test_a_real_control_phrase_is_classified_as_a_control(self):
        self.assertEqual(vm.classify_control("stop speaking"), "speech_stop")
        self.assertEqual(vm.classify_control("pause jarvis"), "pause")

    def test_a_control_phrase_never_starts_a_turn(self):
        with patch.object(vm, "_ask_backend") as ask, \
                patch.object(vm, "dispatch_control") as dispatch:
            vm.handle_queued_item("stop speaking")
        dispatch.assert_called_once()
        ask.assert_not_called()
        self.assertEqual(vm.TURNS.stats["started"], 0)

    def test_a_pause_phrase_never_starts_a_turn(self):
        with patch.object(vm, "_ask_backend") as ask, \
                patch.object(vm, "_deliver_pause") as pause:
            vm.handle_queued_item("pause jarvis")
        pause.assert_called_once()
        ask.assert_not_called()
        self.assertEqual(vm.TURNS.stats["started"], 0)

    def test_a_plain_utterance_does_start_a_turn(self):
        with patch.object(vm, "_ask_backend", return_value="reply") as ask:
            vm.handle_queued_item("what is the weather")
            self.assertTrue(self._wait(lambda: ask.called))
        self.assertEqual(vm.TURNS.stats["started"], 1)


class BargeInHookTests(unittest.TestCase):
    """The listener notifies; the turn manager cancels. Never blocks capture."""

    def setUp(self):
        self._prev = vm.TURNS
        vm.TURNS = vm._TurnManager()

    def tearDown(self):
        vm.TURNS = self._prev

    def test_the_observer_is_registered_with_the_listener(self):
        from backend.services import listener
        self.assertIn(vm._on_barge_in, listener._barge_in_hooks)

    def test_barge_in_onset_cancels_the_active_turn(self):
        vm.TURNS.start("voice-1-999", _SpeakerStub())
        with patch.object(vm, "_cancel_backend_request_async") as cancel:
            vm._on_barge_in()
        cancel.assert_called_once()
        self.assertEqual(cancel.call_args[0][0], "voice-1-999")
        self.assertFalse(vm.TURNS.is_current("voice-1-999"))

    def test_the_observer_never_blocks_on_a_hung_cancel(self):
        """It runs on the real-time capture thread (P1-03)."""
        vm.TURNS.start("voice-2-999", _SpeakerStub())

        def hung(rid, reason):
            time.sleep(3)
            return True

        with patch.object(vm, "_cancel_backend_request", side_effect=hung):
            started = time.perf_counter()
            vm._on_barge_in()
            elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.3,
                        "the capture thread waited on the cancel (%.2fs)"
                        % elapsed)

    def test_the_listener_notify_path_is_fault_tolerant(self):
        from backend.services import listener

        def bad():
            raise RuntimeError("nope")

        listener.register_barge_in_hook(bad)
        self.addCleanup(listener.unregister_barge_in_hook, bad)
        listener._notify_barge_in()      # must not raise


class RegistrationTests(unittest.TestCase):
    def test_turn_request_ids_are_unique_when_submitted_back_to_back(self):
        """A millisecond-resolution id collided; two turns in the same
        millisecond shared one id, which the backend registry treats as a 409
        conflict against a different message."""
        ids = [vm._new_turn_request_id() for _ in range(5)]
        self.assertEqual(len(set(ids)), 5, ids)

    def test_preemption_triggers_on_a_new_speaker_even_with_a_reused_id(self):
        """Playback ownership is the identity that matters, so an id collision
        can never leave two turns registered."""
        manager = vm._TurnManager()
        first = _SpeakerStub()
        second = _SpeakerStub()
        manager.start("same-id", first)
        manager.start("same-id", second)
        self.assertEqual(manager.stats["preempted"], 1)
        self.assertTrue(first.closed, "the old turn's audio was not closed")
        self.assertTrue(manager.is_current("same-id"))

    def test_the_hook_registry_is_idempotent(self):
        from backend.services import listener

        def hook():
            pass

        self.assertTrue(listener.register_barge_in_hook(hook))
        self.assertFalse(listener.register_barge_in_hook(hook))
        self.assertTrue(listener.unregister_barge_in_hook(hook))
        self.assertFalse(listener.unregister_barge_in_hook(hook))

    def test_cancel_worker_is_scoped_and_safe(self):
        """The job behind a request is cancelled; a second call is safe."""
        state, _c, _f = req_registry.admit("p0-08-job", "message")
        self.addCleanup(req_registry.REGISTRY._requests.pop, "p0-08-job", None)
        job = job_registry.new_job(kind="request", label="p0-08-test")
        # The job registry is process-wide: an unfinished job here would make a
        # later unaddressed stop pick the wrong target.
        self.addCleanup(job.finish)
        state.attach_job(job)
        self.assertTrue(state.cancel_worker("barge-in"))
        self.assertTrue(job.cancelled)


if __name__ == "__main__":
    unittest.main()

