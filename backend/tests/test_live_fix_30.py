"""LIVE FIX 30 — the live-test bugs B71-B77 (phantom census, partial
leftovers, generic execute-look-execute, count-feedback, confirm-question,
chat fast-path), tested SILENTLY and deterministically.

One test class per bug. No real screen, network, or model call is made
anywhere: EnumWindows/ShowWindow/capture/vision are stubbed, ledgers are
cleared per test, and the audit log writer is silenced.
"""

import ctypes
import time

import pytest

from backend.core import brain
from backend.services import approvals, screen_control, screen_state
from backend.services import screen_ui_elements


@pytest.fixture(autouse=True)
def _live_fix_30_state(monkeypatch):
    """Reset shared confirmation, queue, chain, and screen state per test."""
    was_enabled = screen_state.is_enabled()
    screen_state.clear_pending_plan()
    screen_state.clear_interactions()
    screen_state.note_request("")
    try:
        screen_control._clear_window_enumeration()
    except Exception:
        pass
    approvals.clear()
    with brain._opencode_confirm_lock:
        brain._pending_opencode_task = None
    with brain._last_action_lock:
        brain._last_action.update({
            "kind": "", "command": "", "summary": "",
            "done": False, "seq": 0, "at": 0.0,
        })
    monkeypatch.setattr(screen_control, "_write_screen_command_log",
                        lambda *a, **kw: None)
    screen_state.set_enabled(True, reason="test_live_fix_30")
    yield
    screen_state.clear_pending_plan()
    screen_state.clear_interactions()
    screen_state.note_request("")
    try:
        screen_control._clear_window_enumeration()
    except Exception:
        pass
    approvals.clear()
    with brain._opencode_confirm_lock:
        brain._pending_opencode_task = None
    with brain._last_action_lock:
        brain._last_action.update({
            "kind": "", "command": "", "summary": "",
            "done": False, "seq": 0, "at": 0.0,
        })
    screen_state.set_enabled(was_enabled, reason="test_live_fix_30_done")


def _win(hwnd, title, iconic=False):
    return {"hwnd": hwnd, "title": title, "iconic": iconic,
            "rect": None, "process": "fake", "window_class": "Fake"}


def _hwnd_plan(text, windows, exclusions=(), action="minimize_hwnd",
               verb="minimize"):
    plan = screen_control._finalize_window_targets(
        verb, action, windows, list(exclusions), False)
    assert plan["ok"], plan
    plan["confidence"] = 0.99
    plan["source"] = "Direct Match"
    plan["command_text"] = text
    return plan


class TestB71PhantomWindowCensus:
    """B71 — ghost/toolwindow/badge titles never inflate the census."""

    def test_clean_window_title_strips_badge(self):
        assert screen_control._clean_window_title("(50) WhatsApp") == "WhatsApp"
        assert screen_control._clean_window_title("Brave") == "Brave"

    def test_finalize_keeps_raw_titles_but_cleans_spoken_names(self):
        windows = [_win(1, "(50) WhatsApp"), _win(2, "Brave")]
        plan = screen_control._finalize_window_targets(
            "minimize", "minimize_hwnd", windows, ["brave"], False)
        assert plan["ok"]
        assert plan["summary"] == "Minimizing 1 window, sir. Leaving Brave."
        assert plan["steps"][0]["title"] == "(50) WhatsApp"
        assert set(plan["steps"][0].keys()) == {
            "action", "hwnd", "title", "description"}
        assert sorted(plan["window_op"]["plan_time_open_hwnds"]) == [1, 2]
        assert plan["window_op"]["plan_time_iconic_hwnds"] == []

    def test_list_managable_windows_drops_ghosts_and_toolwindows(
            self, monkeypatch):
        table = {
            101: {"title": "(50) WhatsApp", "visible": True,
                  "iconic": False, "exstyle": 0},
            102: {"title": "Program Manager", "visible": True,
                  "iconic": False, "exstyle": 0},
            103: {"title": "Default IME", "visible": True,
                  "iconic": False, "exstyle": 0},
            104: {"title": "Tooltip", "visible": True,
                  "iconic": False, "exstyle": 0x80},
            105: {"title": "Brave", "visible": True,
                  "iconic": False, "exstyle": 0x40000},
            106: {"title": "Hidden", "visible": False,
                  "iconic": False, "exstyle": 0},
        }

        class _FakeUser32:
            def IsWindowVisible(self, hwnd):
                return table[int(hwnd)]["visible"]

            def GetWindowTextW(self, hwnd, buf, n):
                buf.value = table[int(hwnd)]["title"]
                return len(table[int(hwnd)]["title"])

            def MonitorFromWindow(self, hwnd, flags):
                return 1

            def IsIconic(self, hwnd):
                return table[int(hwnd)]["iconic"]

            def GetWindowRect(self, hwnd, ref):
                raise OSError("no rect in test")

            def GetWindowLongW(self, hwnd, idx):
                return table[int(hwnd)]["exstyle"]

            def GetWindowThreadProcessId(self, hwnd, ref):
                raise OSError("no pid in test")

            def GetClassNameW(self, hwnd, buf, n):
                buf.value = "Fake"
                return 4

            def EnumWindows(self, cb, lp):
                for h in table:
                    cb(h, lp)
                return True

        fake = _FakeUser32()
        monkeypatch.setattr(ctypes.windll, "user32", fake)
        out = screen_control._list_managable_windows()
        assert sorted(w["hwnd"] for w in out) == [101, 105]
        assert next(
            w for w in out if w["hwnd"] == 101)["title"] == "(50) WhatsApp"


class TestB72WindowPostconditionLook:
    """B72 — enumeration-only look, one bounded retry, no user-window sweep."""

    def test_leftover_retried_once_without_capture(self, monkeypatch):
        # First attempt moved 11, then refused 12 (partial failure).
        live = {_win(11, "A", iconic=True)["hwnd"]: _win(11, "A", iconic=True),
                _win(12, "B")["hwnd"]: _win(12, "B")}
        monkeypatch.setattr(screen_control, "_list_managable_windows",
                            lambda: list(live.values()))
        captures = []
        monkeypatch.setattr(screen_control, "capture_active_window",
                            lambda *a, **kw: captures.append(1) or {
                                "image_data_url": "data:,"})
        calls = []

        def _fake_exec(steps, **kw):
            calls.append(list(steps))
            if len(calls) == 1:
                raise RuntimeError(
                    "I managed to minimize 1 of 2 windows, sir — "
                    "B wouldn't minimize. The rest are done.")
            live[12] = _win(12, "B", iconic=True)

        monkeypatch.setattr(screen_control, "execute_steps", _fake_exec)
        plan = _hwnd_plan(
            "minimize all windows",
            [_win(11, "A"), _win(12, "B"),
             _win(14, "Old", iconic=True)],
            action="minimize_hwnd", verb="minimize")
        # Plan-time snapshot: 14 was already iconic (no-op drop), 13 is a
        # window the user opens mid-run — neither may ever be retried.
        plan["window_op"]["plan_time_open_hwnds"] = [11, 12, 14]
        plan["window_op"]["plan_time_iconic_hwnds"] = [14]
        live[13] = _win(13, "New")
        live[14] = _win(14, "Old")
        status, spoken = screen_control._execute_plan_tagged(
            plan, "minimize all windows")
        assert status == "executed"
        assert captures == []
        assert len(calls) == 2
        assert [s["hwnd"] for s in calls[1]] == [12]

    def test_user_restored_window_never_swept_up(self, monkeypatch):
        live = [_win(11, "A", iconic=True), _win(14, "Old")]
        monkeypatch.setattr(screen_control, "_list_managable_windows",
                            lambda: list(live))
        plan = _hwnd_plan("minimize all windows", [_win(11, "A")])
        plan["window_op"]["plan_time_open_hwnds"] = [11, 14]
        plan["window_op"]["plan_time_iconic_hwnds"] = [14]
        leftovers = screen_control._collect_window_plan_leftovers(
            plan, plan["steps"])
        assert leftovers == []


class TestB73GenericExecuteLookResolveLoop:
    """B73 — wrong-bar override on the T3 pronoun step, clear-before-retype,
    press/type single-command guard, restricted chain resolve."""

    def test_t3_pronoun_step_overrides_local_true_on_address_bar(
            self, monkeypatch):
        screen_state.add_interaction(
            "click in the youtube search bar on my screen",
            "Clicking the YouTube search bar, sir.",
            verified=True,
        )
        steps = [
            {"action": "type", "text": "mr.beast", "ui_runtime_id": "42.1"},
            {"action": "press", "keys": ["enter"]},
        ]
        monkeypatch.setattr(
            screen_control, "_typed_targets",
            lambda steps: [{"text": "mr.beast", "runtime_id": "42.1"}])
        monkeypatch.setattr(
            screen_control, "_read_typed_field_values",
            lambda typed, hwnd=None: [("mr.beast", "mr.beast")])
        monkeypatch.setattr(
            screen_ui_elements, "read_control_values",
            lambda hwnd=None: [{"name": "Address and search bar",
                                "focused": True, "runtime_id": "42.1",
                                "value": "mr.beast"}])
        monkeypatch.setattr(
            screen_control, "capture_active_window",
            lambda *a, **kw: {"image_data_url": "data:image/png;base64,"})
        monkeypatch.setattr(screen_control, "_verify_action_locally",
                            lambda *a, **kw: True)
        assert screen_control._verify_action_with_steps(
            "Typing 'mr.beast' and pressing Enter, sir.",
            steps=steps, hwnd=55,
            command_text="type mr.beast in it click enter") is False

    def test_inject_clear_before_retype_after_leading_click(self):
        retry_steps = [
            {"action": "click", "x": 450, "y": 140,
             "description": "Click YouTube search bar"},
            {"action": "type", "text": "mr.beast"},
            {"action": "press", "keys": ["enter"]},
        ]
        updated = brain._inject_clear_before_retype(retry_steps)
        assert [s["action"] for s in updated] == [
            "click", "hotkey", "press", "type", "press"]
        assert updated[1]["keys"] == ["ctrl", "a"]
        assert updated[2]["keys"] == ["backspace"]

    def test_single_press_command_does_not_auto_retry(self, monkeypatch):
        press_plan = {
            "ok": True, "confidence": 0.99,
            "summary": "Pressing Enter, sir.", "needs_confirmation": False,
            "steps": [{"action": "press", "keys": ["enter"]}],
        }
        exec_calls = []
        monkeypatch.setattr(
            screen_control, "execute_steps",
            lambda steps, **kw: exec_calls.append(list(steps)))
        monkeypatch.setattr(screen_control, "_verify_action_with_steps",
                            lambda *a, **kw: False)
        replanned = []
        monkeypatch.setattr(screen_control, "plan_command",
                            lambda cmd: replanned.append(cmd) or None)
        status, spoken = screen_control._execute_plan_tagged(
            press_plan, "click enter")
        assert status == "executed"
        assert len(exec_calls) == 1
        assert replanned == []
        assert "not fully sure it worked" in spoken

    def test_mi_step_skips_resolve_on_non_recoverable(self, monkeypatch):
        step = {"kind": "screen_control", "text": "click the hidden button"}
        plan1 = {"ok": True, "confidence": 0.4,
                 "steps": [{"action": "click", "x": 10, "y": 10}]}
        plans = []
        monkeypatch.setattr(
            brain, "plan_command",
            lambda text: plans.append(text) or dict(plan1))
        execs = []
        monkeypatch.setattr(
            brain, "_execute_plan_tagged",
            lambda *a, **kw: execs.append(a) or (
                "blocked",
                "I found a possible target, but not confidently enough "
                "to click it."))
        monkeypatch.setattr(brain, "_chain_stop_requested",
                            lambda: False)
        res = brain._mi_screen_control_step(step)
        assert res["status"] == "failed"
        assert len(plans) == 1
        assert len(execs) == 1


class TestB74CountFeedbackComplaintNet:
    """B74 — T6/T10 feedback re-plans the recent window command; present-tense
    state questions answer from enumeration."""

    T6 = ("look at my screen 3 windows are still open , and 5 windows "
          "werent even there to begin with")
    T10 = "you only minimized 1 window 2 more windows are still left"

    def test_feedback_shapes_match_and_resolve(self):
        brain.note_executed_action(
            "screen_control", "minimize all windows", "Minimized.")
        for utterance in (self.T6, self.T10):
            assert brain._is_screen_count_feedback(utterance)
            assert screen_control._is_window_count_feedback(utterance)
            assert brain._is_screen_action_complaint(utterance)
            assert (brain._screen_action_complaint_command(utterance)
                    == "minimize all windows")
        assert not screen_control._is_window_count_feedback(
            "minimize 3 windows")
        assert not screen_control._is_window_count_feedback(
            "how many windows are still open?")

    def test_feedback_end_to_end_replans(self, monkeypatch):
        live = {1: _win(1, "A"), 2: _win(2, "B")}
        monkeypatch.setattr(screen_control, "_list_managable_windows",
                            lambda: list(live.values()))
        execs = []

        def _fake_exec(steps, **kw):
            execs.append(list(steps))
            for s in steps:
                live[int(s["hwnd"])] = _win(
                    int(s["hwnd"]), live[int(s["hwnd"])]["title"],
                    iconic=True)

        monkeypatch.setattr(screen_control, "execute_steps", _fake_exec)
        monkeypatch.setattr(screen_control, "capture_active_window",
                            lambda *a, **kw: {"image_data_url": "data:,"})
        screen_state.note_request("minimize all windows")
        reply = screen_control.maybe_handle_screen_control_message(self.T6)
        assert reply == "Minimizing 2 windows, sir."
        assert [s["hwnd"] for s in execs[0]] == [1, 2]

    def test_present_tense_state_question(self, monkeypatch):
        monkeypatch.setattr(
            screen_control, "_list_managable_windows",
            lambda: [_win(1, "A"), _win(2, "B"), _win(3, "C", iconic=True)])
        reply = screen_control._parse_present_window_state_question(
            "are my windows still open?")
        assert reply == "Yes, sir — 2 of 3 windows are still open: A, B."


class TestB75ConfirmQuestionHonesty:
    """B75 — confirm-questions are queries, never consent."""

    def test_verdict_never_yes_on_confirm_questions(self):
        for q in ("what are you asking me to confirm ?",
                  "confirm what?",
                  "what is pending?"):
            assert approvals.verdict(q) is None
            assert brain._confirmation_verdict(q) is None
        assert approvals.verdict("confirm screen action") == approvals._YES
        assert brain._confirmation_verdict("yes") == "yes"

    def test_live_pending_restated_without_consuming(self):
        screen_state.set_pending_plan(
            {"ok": True, "confidence": 0.4,
             "summary": "Clicking the button, sir.",
             "steps": [{"action": "click", "x": 1, "y": 2}],
             "command_text": "click the button"},
            "click the button", ttl=120)
        reply = screen_control._parse_confirm_question(
            "what are you asking me to confirm ?")
        assert reply.startswith("I'm asking you to confirm:")
        assert screen_state.has_pending_plan()

    def test_nothing_pending_offers_recent_window_rerun(self):
        screen_state.note_request("minimize all windows")
        reply = screen_control._parse_confirm_question(
            "what are you asking me to confirm ?")
        assert reply.startswith(
            "Nothing is waiting for your confirmation")
        assert "minimize all windows" in reply

    def test_feedback_built_pending_dropped(self):
        screen_state.set_pending_plan(
            {"ok": True, "confidence": 0.4,
             "summary": "Clicking 'windows werent even there', sir.",
             "steps": [{"action": "click", "x": 1, "y": 2}],
             "command_text": TestB74CountFeedbackComplaintNet.T6},
            TestB74CountFeedbackComplaintNet.T6, ttl=120)
        reply = screen_control._parse_confirm_question(
            "what are you asking me to confirm ?")
        assert not screen_state.has_pending_plan()
        assert "Clicking" not in reply


class TestB76ChatFastPathAndR13RerunOffer:
    """B76 — complaints never fast-path to chat; leftover cuts earn offers."""

    def test_complaint_never_plain_chat(self):
        assert not brain.is_definitely_plain_chat(
            TestB74CountFeedbackComplaintNet.T10)
        assert not brain.is_definitely_plain_chat(
            TestB74CountFeedbackComplaintNet.T6)

    def test_leftover_cut_claim_offers_rerun(self):
        screen_state.note_request("minimize all windows")
        offer = brain._recent_screen_rerun_offer(
            ["2 more windows are still left"])
        assert ("say 'minimize the rest'" in offer
                and "'do it again'" in offer)
        assert (brain._chat_claim_fallback(["2 more windows are still left"])
                == offer)

    def test_generic_cut_claim_returns_exact_screen_fallback(self):
        screen_state.note_request("minimize all windows")
        assert (brain._chat_claim_fallback(["the window did nothing"])
                == brain._CHAT_SCREEN_FALLBACK)


class TestB77PartialReportLeftovers:
    """B77 — partial failure reports D-of-T with cleaned names + rest offer."""

    def test_partial_error_reconciles_to_rest_offer(self, monkeypatch):
        live = {21: _win(21, "(50) WhatsApp"), 22: _win(22, "Brave"),
                23: _win(23, "Terminal")}
        monkeypatch.setattr(screen_control, "_list_managable_windows",
                            lambda: list(live.values()))
        calls = []

        def _fake_exec(steps, **kw):
            calls.append(list(steps))
            raise RuntimeError(
                "I managed to minimize 2 of 3 windows, sir — "
                "(50) WhatsApp wouldn't minimize. The rest are done.")

        monkeypatch.setattr(screen_control, "execute_steps", _fake_exec)
        plan = _hwnd_plan("minimize all windows",
                          [live[21], live[22], live[23]])
        for hwnd in (22, 23):
            live[hwnd] = _win(hwnd, live[hwnd]["title"], iconic=True)
        status, spoken = screen_control._execute_plan_tagged(
            plan, "minimize all windows")
        assert status == "blocked"
        assert "The rest are done." not in spoken
        assert "WhatsApp wouldn't minimize" in spoken
        assert "(50)" not in spoken
        assert "Say 'minimize the rest' to try them again." in spoken
        assert "2 of 3 windows" in spoken
        assert len(calls) == 2
        assert [s["hwnd"] for s in calls[1]] == [21]
        assert (screen_state.get_last_request()["command"]
                == "minimize all windows")
