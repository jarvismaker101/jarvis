"""LIVE FIX 29 — the live re-test bugs B63-B70 (the screen-command routing
wave), tested SILENTLY and deterministically.

One test class per bug. Every behavior test drives the real preroute/planning
entry points (``brain._process_message_inner`` / ``brain.answer_status_question``
/ ``brain._consume_opencode_confirmation``) with the user's own utterances.

Simulated seams, stated openly:
- no real screen, network, or model call is made anywhere: the screen-control
  planner/executor and the deferred browser handoff are stubbed in the brain /
  screen_control namespaces only where a test must not touch the OS;
- the audit log writer is silenced (it is not under test);
- the vision/LLM cascade is never reached (every deterministic gate under test
  returns before it).
"""

import time

import pytest

from backend.core import brain
from backend.services import approvals, screen_control, screen_state


@pytest.fixture(autouse=True)
def _live_fix_29_state(monkeypatch):
    """Reset shared confirmation, queue, chain, and screen state per test."""
    was_enabled = screen_state.is_enabled()
    screen_state.clear_pending_plan()
    screen_state.clear_interactions()
    screen_state.clear_refused_chain()
    approvals.clear()
    with brain._opencode_confirm_lock:
        brain._pending_opencode_task = None
    with brain._confirmation_lock:
        brain._pending_confirmation = None
        brain._pending_screen_interview = None
    with brain._action_queue_lock:
        brain._pending_action_requests.clear()
    brain._opencode_task_running = False
    with brain._last_action_lock:
        brain._last_action.update({
            "kind": "", "command": "", "summary": "",
            "done": False, "seq": 0, "at": 0.0,
        })
    with brain._chain_followup_lock:
        brain._last_chain.update({"plan": None, "results": None, "at": 0.0})
    monkeypatch.setattr(brain, "_last_offer", None, raising=False)
    # The audit log is not under test — never append to the real file.
    monkeypatch.setattr(screen_control, "_write_screen_command_log",
                        lambda *a, **kw: None)
    yield
    screen_state.clear_pending_plan()
    approvals.clear()
    with brain._opencode_confirm_lock:
        brain._pending_opencode_task = None
    with brain._confirmation_lock:
        brain._pending_screen_interview = None
    with brain._action_queue_lock:
        brain._pending_action_requests.clear()
    brain._opencode_task_running = False
    screen_state.set_enabled(was_enabled)


class TestB63AffirmativeContrastScreenCorrection:
    """B63: 'you are clicking in the youtube search bar but typing in the
    browser url bar' rewrites to a screen command/chain, never opencode."""

    def test_contrast_correction_with_recent_typed_step_builds_chain_command(
            self):
        brain.note_executed_action(
            "screen_control", "type mr.beast in it click enter", "Typed.")
        utterance = (
            "you are clicking in the youtube search bar "
            "but typing in the browser url bar"
        )
        assert brain._screen_action_correction_command(utterance) == (
            "click in the youtube search bar, then type mr.beast in it "
            "click enter"
        )

    def test_contrast_correction_without_typed_history_rewrites_click_target(
            self):
        brain.note_executed_action(
            "screen_control", "click in the youtube search bar", "Clicked.")
        utterance = (
            "you are clicking in the youtube search bar "
            "but typing in the browser url bar"
        )
        assert (brain._screen_action_correction_command(utterance)
                == "click in the youtube search bar")

    def test_contrast_correction_clears_pending_plan_and_never_arms_opencode(
            self, monkeypatch):
        brain.note_executed_action(
            "screen_control", "click in the youtube search bar", "Clicked.")
        screen_state.set_pending_plan(
            {"ok": True, "steps": [{"action": "click", "x": 10, "y": 10}]},
            "click the first video",
        )
        routed = []
        monkeypatch.setattr(
            brain, "maybe_handle_screen_control_message",
            lambda msg: routed.append(msg) or "Clicked the YouTube search bar, sir.",
        )
        reply = brain._process_message_inner(
            "you are clicking in the youtube search bar "
            "but typing in the browser url bar"
        )
        assert reply == "Clicked the YouTube search bar, sir."
        assert routed == ["click in the youtube search bar"]
        assert not screen_state.has_pending_plan()
        assert brain._pending_opencode_task is None

    def test_recall_turn_never_matches_contrast_correction(self):
        brain.note_executed_action(
            "screen_control", "click in the youtube search bar", "Clicked.")
        assert (brain._screen_action_correction_command(
            "what did you just say about clicking the search bar") is None)


class TestB64OpencodeDoesNotShadowArmedScreenPlan:
    """B64: an armed chain screen plan is never shadowed by an armed
    opencode confirmation."""

    def test_yes_confirms_armed_chain_screen_plan_instead_of_opencode(
            self, monkeypatch):
        screen_state.set_enabled(True)
        chain_plan = {
            "ok": True,
            "summary": "Clicking the first video, sir.",
            "steps": [{"action": "click", "x": 100, "y": 200}],
            "screen_generation": screen_state.generation(),
            "command_text": "click the first video from the results that appear",
            "chain_context": {"signature": "sig-1", "step_index": 2},
        }
        screen_state.set_pending_plan(
            chain_plan,
            "click the first video from the results that appear",
            ttl=300.0,
        )
        # A wrongly-armed opencode confirmation coexists with the live chain
        # step-3 plan (the chain-armed plan is preserved at arm time).
        brain._arm_opencode_confirmation(
            "Correct the previous action by clicking the YouTube search bar",
            "you are clicking in the youtube search bar",
        )
        assert screen_state.has_pending_plan()
        assert brain._pending_opencode_task is not None

        executed = []
        monkeypatch.setattr(
            screen_control, "execute_steps",
            lambda steps, **kw: executed.append(steps))
        resumed = []
        monkeypatch.setattr(
            screen_control, "_notify_plan_confirmed",
            lambda ctx, summary: resumed.append((ctx, summary)))
        opencode_called = []
        monkeypatch.setattr(
            brain, "_execute_deferred_opencode",
            lambda *a, **kw: opencode_called.append(a) or
            "Taking over the browser task, sir.")

        reply = brain._process_message_inner("yes")
        assert reply == "Clicking the first video, sir."
        assert len(executed) == 1
        assert resumed == [({"signature": "sig-1", "step_index": 2},
                            "Clicking the first video, sir.")]
        assert opencode_called == []

    def test_arming_screen_plan_disarms_stale_opencode_preview(self):
        brain._arm_opencode_confirmation("Some browser task", "do browser task")
        assert brain._pending_opencode_task is not None
        screen_state.set_pending_plan(
            {"ok": True, "steps": [{"action": "click", "x": 1, "y": 2}]},
            "click submit",
        )
        assert brain._pending_opencode_task is None


class TestB65EngineRouteCorrectionGate:
    """B65: 'that is not a browser taks its a screen command' disarms opencode
    and re-routes the last screen command — never chat."""

    def test_engine_correction_drops_opencode_and_reroutes_last_screen_cmd(
            self, monkeypatch):
        brain._arm_opencode_confirmation(
            "Correct the previous action", "you are clicking...")
        brain.note_executed_action(
            "screen_control",
            "click the first video from the results that appear",
            "",
        )
        rerouted = []
        monkeypatch.setattr(
            brain, "maybe_handle_screen_control_message",
            lambda cmd: rerouted.append(cmd) or "Clicking the first video, sir.",
        )
        reply = brain._process_message_inner(
            "that is not a browser taks its a screen command"
        )
        assert reply == "Clicking the first video, sir."
        assert rerouted == ["click the first video from the results that appear"]
        assert brain._pending_opencode_task is None

    def test_engine_correction_honest_fallback_when_nothing_recent(self):
        reply = brain._process_message_inner(
            "that is not a browser task, it's a screen command"
        )
        assert "screen command" in reply.lower()
        assert "[INTENT]" not in reply
        assert "[TASK]" not in reply

    def test_genuine_browser_task_request_not_matched(self):
        assert not brain._is_engine_route_correction(
            "take over the browser task and login to github")


class TestB66DoneQuestionConsultsLastActionRegistry:
    """B66: 'are you done?' right after a verified screen action reports the
    completed action instead of 'nothing is running right now'."""

    def test_are_you_done_after_verified_minimize_reports_done_summary(
            self, monkeypatch):
        monkeypatch.setattr(brain, "_last_task_result", lambda: (None, ""))
        brain.note_executed_action(
            "screen_control",
            "minimize all the windows on my screen except for the brave window",
            "Minimizing 6 windows, sir. Leaving Brave.",
        )
        reply = brain.answer_status_question("are you done?")
        assert reply == (
            "Yes, sir — done: Minimizing 6 windows, sir. Leaving Brave."
        )

    def test_are_you_done_when_last_action_not_done_answers_honestly(
            self, monkeypatch):
        monkeypatch.setattr(brain, "_last_task_result", lambda: (None, ""))
        brain.note_executed_action(
            "screen_control", "click the search button", "")
        reply = brain.answer_status_question("is it done?")
        assert reply.startswith("No, sir —")
        assert "click the search button" in reply

    def test_unrelated_topic_done_question_does_not_leak_screen_summary(
            self, monkeypatch):
        monkeypatch.setattr(brain, "_last_task_result", lambda: (None, ""))
        brain.note_executed_action(
            "screen_control",
            "minimize all the windows on my screen except for the brave window",
            "Minimizing 6 windows, sir. Leaving Brave.",
        )
        reply = brain.answer_status_question(
            "did you finish the online search for this skin?")
        assert "Leaving Brave" not in reply

    def test_idle_status_question_still_reports_nothing_running(
            self, monkeypatch):
        monkeypatch.setattr(brain, "_last_task_result", lambda: (None, ""))
        assert (brain.answer_status_question("what is running right now?")
                == "Sir, nothing is running right now.")


class TestB67WindowActionComplaintAndInterviewStrip:
    """B67: 'i asked you to minimize but you didnt' re-plans fresh; a
    standalone screen-interview question in chat is stripped."""

    def test_complaint_reruns_last_window_command_fresh(self, monkeypatch):
        brain.note_executed_action(
            "screen_control",
            "minimize all the windows on my screen except for the brave window",
            "Minimizing 6 windows, sir. Leaving Brave.",
        )
        rerouted = []
        monkeypatch.setattr(
            brain, "maybe_handle_screen_control_message",
            lambda cmd: rerouted.append(cmd) or
            "Minimizing 6 windows, sir. Leaving Brave.",
        )
        reply = brain._process_message_inner(
            "i asked you to minimize but you didnt")
        assert reply == "Minimizing 6 windows, sir. Leaving Brave."
        assert rerouted == [
            "minimize all the windows on my screen except for the brave window"
        ]

    def test_standalone_chat_screen_interview_is_stripped(self):
        raw = ("My apologies, sir. Please tell me which window you want to "
               "keep visible.")
        cleaned = brain._strip_unverified_action_claims(raw, "chat")
        assert cleaned == brain._CHAT_SCREEN_FALLBACK
        assert "which window" not in cleaned.lower()

    def test_complaint_with_nothing_recent_answers_honestly(self):
        reply = brain._process_message_inner(
            "i asked you to minimize but you didnt")
        assert "recent screen action" in reply.lower()


class TestB68BareAnswerToScreenInterviewAndTaskDeflection:
    """B68: a bare 'brave' after a screen-interview slot, or a window-
    management task description, routes to screen_control — never opencode."""

    def test_bare_app_answer_resolves_pending_screen_interview_slot(
            self, monkeypatch):
        brain._set_pending_screen_interview(verb="minimize")
        rerouted = []
        monkeypatch.setattr(
            brain, "maybe_handle_screen_control_message",
            lambda cmd: rerouted.append(cmd) or
            "Minimizing 5 windows, sir. Leaving Brave.",
        )
        reply = brain._process_message_inner("brave")
        assert reply == "Minimizing 5 windows, sir. Leaving Brave."
        assert rerouted == [
            "minimize all the windows on my screen except for the brave window"
        ]
        assert brain._pending_opencode_task is None

    def test_handle_opencode_task_deflects_window_management_description(
            self, monkeypatch):
        rerouted = []
        monkeypatch.setattr(
            brain, "maybe_handle_screen_control_message",
            lambda cmd: rerouted.append(cmd) or
            "Minimizing 6 windows, sir. Leaving Brave.",
        )
        reply = brain.handle_opencode_task(
            "Minimize all windows except for the Brave browser window.",
            original_message="brave",
        )
        assert reply == "Minimizing 6 windows, sir. Leaving Brave."
        assert brain._pending_opencode_task is None

    def test_expired_interview_slot_does_not_capture_bare_word(self):
        brain._set_pending_screen_interview(verb="minimize")
        with brain._confirmation_lock:
            brain._pending_screen_interview["at"] = time.time() - 1000.0
        assert brain._consume_pending_screen_interview("brave") is None


class TestB69WindowManagementOpencodeConfirmationNeverLaunchesBrowser:
    """B69: if an opencode confirmation is armed with a window-management job,
    'yes' deflects to screen_control and never launches the browser agent."""

    def test_yes_on_armed_window_management_opencode_task_deflects_to_screen(
            self, monkeypatch):
        brain._arm_opencode_confirmation(
            "Minimize all windows except for the Brave browser window.",
            "brave",
        )
        browser_launched = []
        monkeypatch.setattr(
            brain, "_execute_deferred_opencode",
            lambda *a, **kw: browser_launched.append(a) or
            brain.BROWSER_AGENT_START_PHRASE)
        sc_routed = []
        monkeypatch.setattr(
            brain, "maybe_handle_screen_control_message",
            lambda cmd: sc_routed.append(cmd) or
            "Minimizing 6 windows, sir. Leaving Brave.",
        )
        reply = brain._consume_opencode_confirmation("yes")
        assert reply == "Minimizing 6 windows, sir. Leaving Brave."
        assert sc_routed == [
            "Minimize all windows except for the Brave browser window."
        ]
        assert browser_launched == []


class TestB70RouteFeedbackBypassesS18ActionQueue:
    """B70: 'its not a browser task' while a task runs bypasses the S18 queue,
    stops the wrong task, and re-routes immediately."""

    def test_route_feedback_not_queued_when_opencode_task_running(
            self, monkeypatch):
        brain._opencode_task_running = True
        brain.note_executed_action(
            "screen_control",
            "minimize all the windows on my screen except for the brave window",
            "Minimizing 6 windows, sir. Leaving Brave.",
        )
        stopped = []
        monkeypatch.setattr(
            brain, "handle_stop_message",
            lambda msg, **kw: stopped.append(msg) or "Stopped, sir.")
        monkeypatch.setattr(
            brain, "maybe_handle_screen_control_message",
            lambda cmd: "Minimizing 6 windows, sir. Leaving Brave.")

        assert brain._queue_action_request("its not a browser task", False) is None
        assert brain._pending_action_requests == []

        reply = brain._process_message_inner("its not a browser task")
        assert reply != brain.ACTION_QUEUED_REPLY
        assert "stop that and run it as a screen command" in reply.lower()
        assert brain._pending_action_requests == []
        assert not brain._opencode_task_running
        assert stopped == ["stop the browser task"]

    def test_genuine_action_request_still_queues_while_task_running(self):
        brain._opencode_task_running = True
        held = brain._queue_action_request(
            "create a folder named reports on my desktop", False)
        assert held == brain.ACTION_QUEUED_REPLY
        assert len(brain._pending_action_requests) == 1
