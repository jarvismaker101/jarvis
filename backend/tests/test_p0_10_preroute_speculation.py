"""P0-10 — the speculative chat racer starts before the pre-route chain.

The chain between "transcript ready" and "route selected" is dead air: task
request, code tools, screen control, explicit research, orchestrator route,
classifier. The speculation used to start after all of it, so none of that cost
was hidden. P0-10 hoists the start above the chain — which creates the opposite
risk, and that is what most of this file pins: once the speculation starts
early, EVERY early return in the chain must release it.

Three properties are load-bearing and each has an explicit test:

* the speculation starts first, so the predicates it overlaps are actually
  overlapped (not merely reordered);
* an un-adopted speculation is cancelled on the way out of the turn, whatever
  path the routing took (a leaked background stream is the failure mode);
* an ADOPTED speculation is left alone (cancelling it injects the
  end-of-stream sentinel and truncates the reply being streamed from it).

Staying green from the pre-existing suites: ``test_chat_race.py`` (deltas reach
the caller exactly and in order, nothing leaks on a non-chat verdict),
``test_f02_goal_routing.py`` (an orchestrator route never speculates) and
``test_brain_gate.py`` (pending confirmations are consumed before the
classifier — and, per P0-10, before any speculation exists).
"""

import time
import unittest
from unittest.mock import patch

from backend import config
from backend.core import brain
from backend.services import latency


def _chat_verdict(message="hello"):
    return {
        "intent": "chat",
        "steps": [],
        "task_description": "",
        "query": message,
        "original": message,
    }


class _RacerRecorder:
    """Records every ``_ChatRacer`` construction, adoption and cancellation."""

    def __init__(self):
        self.events = []
        self.instances = []

    def build(self, label="racer"):
        """Return a stand-in class to patch over ``brain._ChatRacer``."""
        recorder = self

        class _FakeRacer:
            def __init__(self, msg, voice_compact=False, history=None):
                self.msg = msg
                self.voice_compact = voice_compact
                self.cancelled = False
                self._adopted = False
                recorder.instances.append(self)
                recorder.events.append((label + ":start", msg))

            # ── the surface brain actually uses ──
            def built(self):
                # A racer that built an LLM answer is the one a chat route
                # adopts (``prebuilt["path"] == "llm"``); any other path is
                # consumed by the selected route instead.
                return {"path": "llm", "messages": [], "system_prompt": "",
                        "query": self.msg}

            def has_stream(self):
                return True

            def adopt(self):
                self._adopted = True
                recorder.events.append((label + ":adopt", self.msg))

                def _drain():
                    return iter(())

                return _drain()

            def cancel(self):
                self.cancelled = True
                recorder.events.append((label + ":cancel", self.msg))

            def join(self, timeout=None):
                return True

            @property
            def is_adopted(self):
                return self._adopted

            @property
            def is_done(self):
                return self.cancelled

        return _FakeRacer


class _PreRouteCase(unittest.TestCase):
    def setUp(self):
        # A turn that leaked a racer into this thread would poison the next
        # test, so the slot is cleared before and after every case.
        brain._cancel_orphan_turn_racer()
        brain._pending_confirmation = None
        brain._pending_opencode_task = None
        brain._pending_browser_clarification = None
        brain._proactive_research_fired = False
        from backend.core.memory import clear_history
        clear_history()
        self.recorder = _RacerRecorder()

    def tearDown(self):
        brain._cancel_orphan_turn_racer()
        brain._pending_confirmation = None
        brain._pending_opencode_task = None
        brain._pending_browser_clarification = None
        brain._proactive_research_fired = False
        from backend.core.memory import clear_history
        clear_history()
        brain.set_opencode_task_running(False)

    def _patch_racer(self):
        return patch.object(brain, "_ChatRacer", self.recorder.build())

    def _quiet_routing(self):
        """Everything below the predicate chain, neutralised."""
        return [
            patch.object(brain, "classify_intent",
                         return_value=_chat_verdict()),
            patch.object(brain, "handle_chat", return_value="Hello, sir."),
            patch.object(brain, "maybe_handle_screen_control_message",
                         return_value=None),
        ]


class SpeculationStartsFirstTests(_PreRouteCase):
    """The whole point of P0-10: the race overlaps the predicates."""

    def test_the_chat_route_adopts_the_early_speculation(self):
        """The hoist must not orphan the racer on the route that wants it."""
        with self._patch_racer(), \
             patch.object(brain, "is_explicit_task_request", return_value=False), \
             patch.object(brain, "is_code_tool_request", return_value=False), \
             patch.object(brain, "force_research", return_value=False), \
             patch.object(brain, "classify_intent",
                          return_value=_chat_verdict("tell me about tea")), \
             patch.object(brain, "handle_chat", return_value="Hello, sir."), \
             patch.object(brain, "maybe_handle_screen_control_message",
                          return_value=None):
            response = brain.process_message("tell me about tea", sync_voice=False,
                                             stream_reply=lambda delta: None)

        self.assertEqual(response, "Hello, sir.")
        self.assertEqual([event[0] for event in self.recorder.events],
                         ["racer:start", "racer:adopt"])
        racer = self.recorder.instances[0]
        self.assertTrue(racer.is_adopted)
        self.assertFalse(racer.cancelled)

    def test_the_predicate_runs_after_the_speculation_was_constructed(self):
        """Ordering, proved with two ordered side effects.

        Pre-fix the racer was constructed after this predicate, so the recorded
        order was [predicate, racer] — the assertion below is the regression.
        """
        seen = []
        recorder = self.recorder
        fake_cls = recorder.build()

        def racer_factory(msg, voice_compact=False, history=None):
            seen.append("racer")
            return fake_cls(msg, voice_compact, history)

        def task_predicate(message):
            seen.append("predicate")
            return False

        with patch.object(brain, "_ChatRacer", racer_factory), \
             patch.object(brain, "is_explicit_task_request",
                          side_effect=task_predicate), \
             patch.object(brain, "is_code_tool_request", return_value=False), \
             patch.object(brain, "force_research", return_value=False), \
             patch.object(brain, "classify_intent",
                          return_value=_chat_verdict()), \
             patch.object(brain, "handle_chat", return_value="Hello, sir."), \
             patch.object(brain, "maybe_handle_screen_control_message",
                          return_value=None):
            brain.process_message("tell me about tea", sync_voice=False,
                                  stream_reply=lambda delta: None)

        self.assertEqual(seen[:2], ["racer", "predicate"])


class EarlyReturnReleasesTheSpeculationTests(_PreRouteCase):
    """Every early return in the chain must cancel an un-adopted racer."""

    def test_a_task_verdict_cancels_the_early_speculation(self):
        collected = []
        with self._patch_racer(), \
             patch.object(brain, "is_explicit_task_request", return_value=True), \
             patch.object(brain, "handle_task_message",
                          return_value="Task handled, sir."), \
             patch.object(brain, "classify_intent",
                          return_value=_chat_verdict()), \
             patch.object(brain, "handle_chat", return_value="should not run"):
            response = brain.process_message(
                "create a folder called demo", sync_voice=False,
                stream_reply=collected.append)

        self.assertEqual(response, "Task handled, sir.")
        self.assertEqual(collected, [], "no speculative text may be released")
        self.assertEqual(len(self.recorder.instances), 1,
                         "the task route must not leak a speculation")
        racer = self.recorder.instances[0]
        self.assertTrue(racer.cancelled)
        self.assertFalse(racer.is_adopted)

    def test_a_code_tool_verdict_cancels_the_early_speculation(self):
        collected = []
        with self._patch_racer(), \
             patch.object(brain, "is_explicit_task_request", return_value=False), \
             patch.object(brain, "is_code_tool_request", return_value=True), \
             patch.object(brain, "handle_task_message",
                          return_value="Reading the file, sir."), \
             patch.object(brain, "classify_intent",
                          return_value=_chat_verdict()), \
             patch.object(brain, "handle_chat", return_value="should not run"):
            response = brain.process_message(
                "read the file app.py", sync_voice=False,
                stream_reply=collected.append)

        self.assertEqual(response, "Reading the file, sir.")
        self.assertEqual(collected, [])
        self.assertTrue(self.recorder.instances[0].cancelled)

    def test_a_screen_control_verdict_cancels_the_early_speculation(self):
        collected = []
        with self._patch_racer(), \
             patch.object(brain, "is_explicit_task_request", return_value=False), \
             patch.object(brain, "is_code_tool_request", return_value=False), \
             patch.object(brain, "maybe_handle_screen_control_message",
                          return_value="Clicked it, sir."), \
             patch.object(brain, "classify_intent",
                          return_value=_chat_verdict()), \
             patch.object(brain, "handle_chat", return_value="should not run"):
            response = brain.process_message(
                "just click at the video on my screen", sync_voice=False,
                stream_reply=collected.append)

        self.assertEqual(response, "Clicked it, sir.")
        self.assertEqual(collected, [])
        self.assertTrue(self.recorder.instances[0].cancelled)

    def test_the_fresh_info_reroute_releases_the_speculation(self):
        collected = []
        message = "whats the pricing for claude fable 5.1 model"
        with self._patch_racer(), \
             patch.object(brain, "is_explicit_task_request", return_value=False), \
             patch.object(brain, "is_code_tool_request", return_value=False), \
             patch.object(brain, "maybe_handle_screen_control_message",
                          return_value=None), \
             patch.object(brain, "force_research", return_value=False), \
             patch.object(brain, "classify_intent",
                          return_value=_chat_verdict(message)), \
             patch.object(brain, "handle_research_intent",
                          return_value="researched") as research, \
             patch.object(brain, "handle_chat", return_value="should not run"):
            response = brain.process_message(message, sync_voice=False,
                                             stream_reply=collected.append)

        self.assertEqual(response, "researched")
        research.assert_called_once()
        self.assertEqual(collected, [])
        self.assertTrue(self.recorder.instances[0].cancelled)


class GatesStillComeFirstTests(_PreRouteCase):
    """A gate answer must never race a speculative answer."""

    def test_a_pending_confirmation_answer_never_starts_a_speculation(self):
        collected = []
        with self._patch_racer(), \
             patch.object(brain, "_consume_confirmation",
                          return_value="Looking it up, sir."), \
             patch.object(brain, "classify_intent",
                          return_value=_chat_verdict()), \
             patch.object(brain, "handle_chat", return_value="should not run"):
            response = brain.process_message("yes", sync_voice=False,
                                             stream_reply=collected.append)

        self.assertEqual(response, "Looking it up, sir.")
        self.assertEqual(self.recorder.instances, [],
                         "a confirmation answer must not speculate")
        self.assertEqual(collected, [])

    def test_a_pending_task_confirmation_never_starts_a_speculation(self):
        with self._patch_racer(), \
             patch.object(brain, "consume_task_confirmation",
                          return_value="Creating the demo folder, sir."), \
             patch.object(brain, "classify_intent",
                          return_value=_chat_verdict()), \
             patch.object(brain, "handle_chat", return_value="should not run"):
            response = brain.process_message("confirm task", sync_voice=False,
                                             stream_reply=lambda delta: None)

        self.assertEqual(response, "Creating the demo folder, sir.")
        self.assertEqual(self.recorder.instances, [])

    def test_a_memory_phrase_never_starts_a_speculation(self):
        with self._patch_racer(), \
             patch.object(brain, "classify_intent",
                          return_value=_chat_verdict()), \
             patch.object(brain, "handle_chat", return_value="should not run"):
            class _Store:
                def handle_memory_phrase(self, message):
                    return "Noted, sir."

            with patch.object(brain, "memory_store", _Store()):
                response = brain.process_message(
                    "remember that I prefer tea", sync_voice=False,
                    stream_reply=lambda delta: None)

        self.assertEqual(response, "Noted, sir.")
        self.assertEqual(self.recorder.instances, [])


class OrchestratorRouteTests(_PreRouteCase):
    """F02 keeps the goal route: the orchestrator must not pay for a chat race."""

    def _run(self, orchestrator_reply, message="tell me about tea"):
        with self._patch_racer(), \
             patch.object(config, "ORCHESTRATOR_MODE", "orchestrator"), \
             patch.object(brain, "orchestrator_handle_message",
                          return_value=orchestrator_reply), \
             patch.object(brain, "is_screen_question", return_value=False), \
             patch.object(brain, "is_explicit_task_request", return_value=False), \
             patch.object(brain, "is_code_tool_request", return_value=False), \
             patch.object(brain, "maybe_handle_screen_control_message",
                          return_value=None), \
             patch.object(brain, "force_research", return_value=False), \
             patch.object(brain, "classify_intent",
                          return_value=_chat_verdict()), \
             patch.object(brain, "handle_chat", return_value="legacy reply"):
            return brain.process_message(message, sync_voice=False,
                                         stream_reply=lambda delta: None)

    def test_an_orchestrator_route_never_starts_the_early_speculation(self):
        reply = self._run({"status": "answered", "reply": "Fine, sir.",
                           "plan": None}, message="what is on my screen")
        self.assertEqual(reply, "Fine, sir.")
        self.assertEqual(self.recorder.instances, [])

    def test_a_declined_orchestrator_route_falls_back_without_speculating(self):
        reply = self._run(None)
        self.assertEqual(reply, "legacy reply")
        self.assertEqual(self.recorder.instances, [],
                         "a goal-shaped route must not speculate on fallback")


class CleanupContractTests(_PreRouteCase):
    """The turn owns the speculation's lifetime."""

    def test_the_turn_cleanup_cancels_a_speculation_no_route_adopted(self):
        with self._patch_racer():
            fake = brain._ChatRacer("hi", False)
            brain._register_turn_racer(fake)
            brain._cancel_orphan_turn_racer()
        self.assertTrue(fake.cancelled)
        # The slot is empty again, so a second call is a no-op.
        brain._cancel_orphan_turn_racer()
        self.assertTrue(fake.cancelled)

    def test_the_turn_cleanup_leaves_an_adopted_speculation_alone(self):
        with self._patch_racer():
            fake = brain._ChatRacer("hi", False)
            fake.adopt()
            brain._register_turn_racer(fake)
            brain._cancel_orphan_turn_racer()
        self.assertFalse(
            fake.cancelled,
            "cancel() injects the end-of-stream sentinel — cancelling an "
            "adopted racer would truncate the reply being streamed from it")

    def test_adopt_marks_a_real_racer_as_owned(self):
        with patch.object(brain, "get_history", return_value=[]), \
             patch.object(brain, "_build_chat_messages",
                          return_value={"path": "browser_search",
                                        "messages": []}):
            racer = brain._ChatRacer("hi", voice_compact=False)
            try:
                self.assertFalse(racer.is_adopted)
                racer.adopt()
                self.assertTrue(racer.is_adopted)
            finally:
                racer.cancel()
                racer.join(2.0)

    def test_the_settled_cleanup_is_idempotent(self):
        with self._patch_racer():
            fake = brain._ChatRacer("hi", False)
            brain._register_turn_racer(fake)
            brain._cancel_orphan_turn_racer()
            brain._cancel_orphan_turn_racer()
        self.assertTrue(fake.cancelled)

    def test_a_racer_that_was_never_started_is_not_cancelled(self):
        brain._cancel_orphan_turn_racer()  # must not raise


class PredicateCostTelemetryTests(_PreRouteCase):
    """P0-10 also had to MEASURE the chain, not just reorder it."""

    def _record(self, request_id="req-p010"):
        with self._patch_racer(), \
             patch.object(brain, "is_explicit_task_request", return_value=False), \
             patch.object(brain, "is_code_tool_request", return_value=False), \
             patch.object(brain, "maybe_handle_screen_control_message",
                          return_value=None), \
             patch.object(brain, "force_research", return_value=False), \
             patch.object(brain, "classify_intent",
                          return_value=_chat_verdict("tell me about tea")), \
             patch.object(brain, "handle_chat", return_value="Hello, sir."):
            brain.process_message("tell me about tea", sync_voice=False,
                                  stream_reply=lambda delta: None,
                                  request_id=request_id)
        return latency.finish(request_id)

    def test_every_preroute_predicate_reports_its_measured_cost(self):
        latency.begin("req-p010", origin="ui", label="tell me about tea")
        record = self._record()
        self.assertIsNotNone(record, "the turn must publish a waterfall")
        steps = {step["name"]: step for step in record["steps"]}
        for name in ("preroute_memory_phrase", "preroute_stop_research",
                     "preroute_confirmation", "preroute_task_confirmation",
                     "preroute_opencode_confirmation",
                     "preroute_browser_followup", "preroute_route",
                     "preroute_task_request", "preroute_code_tool",
                     "preroute_screen_control", "preroute_force_research",
                     "racer_start"):
            self.assertIn(name, steps, "%s must be on the waterfall" % name)
        for name, step in steps.items():
            if name.startswith("preroute_"):
                self.assertIn("predicate_ms", step["meta"])
                self.assertGreaterEqual(step["meta"]["predicate_ms"], 0.0)
                self.assertLess(step["meta"]["predicate_ms"], 60_000.0)

    def test_the_marks_stay_within_the_turn_budget(self):
        latency.begin("req-p010-bounded", origin="ui", label="x")
        record = self._record("req-p010-bounded")
        self.assertLessEqual(len(record["marks"]), latency.MAX_MARKS_PER_TURN)

    def test_a_measured_predicate_cost_is_a_real_measurement(self):
        """The number must come from the predicate, not from a constant."""
        def slow_predicate(message):
            time.sleep(0.05)
            return False

        latency.begin("req-p010-slow", origin="ui", label="x")
        with self._patch_racer(), \
             patch.object(brain, "is_explicit_task_request",
                          side_effect=slow_predicate), \
             patch.object(brain, "is_code_tool_request", return_value=False), \
             patch.object(brain, "maybe_handle_screen_control_message",
                          return_value=None), \
             patch.object(brain, "force_research", return_value=False), \
             patch.object(brain, "classify_intent",
                          return_value=_chat_verdict("tell me about tea")), \
             patch.object(brain, "handle_chat", return_value="Hello, sir."):
            brain.process_message("tell me about tea", sync_voice=False,
                                  stream_reply=lambda delta: None,
                                  request_id="req-p010-slow")
        record = latency.finish("req-p010-slow")
        steps = {step["name"]: step for step in record["steps"]}
        self.assertGreaterEqual(
            steps["preroute_task_request"]["meta"]["predicate_ms"], 40.0,
            "a 50ms predicate cannot be reported as free")


if __name__ == "__main__":
    unittest.main()
