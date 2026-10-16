"""Live Fix 32 regression tests for bugs B82-B88.

One class per bug. Stubbed Win32/UIA/network; ledgers cleared per test.
"""

import os
import tempfile
import time

import pytest

from backend.core import brain
from backend.services import (
    approvals,
    multi_intent,
    screen_capture,
    screen_control,
    screen_state,
    screen_ui_elements,
)


@pytest.fixture(autouse=True)
def _clean_ledgers(monkeypatch):
    """Reset screen_state, approvals, window enum, UIA cache and brain ledgers.

    The toggle state file is redirected to an ISOLATED per-run temp path so a
    test run can never persist "screen controls ON" into the shared
    ``jarvis_screen_toggle.json`` the live backend rehydrates (an empty env
    value is falsy and would fall through to that shared file).
    """
    monkeypatch.setenv(
        "JARVIS_SCREEN_STATE_PATH",
        os.path.join(tempfile.gettempdir(), "jarvis_test_live_fix_32_toggle.json"))
    screen_state.set_enabled(True, reason="test_live_fix_32 setup")
    screen_state.clear_interactions()
    screen_state.clear_pending_plan()
    screen_state.clear_refused_chain()
    approvals.clear()
    screen_control._clear_window_enumeration()
    try:
        screen_ui_elements._element_cache.clear()
    except Exception:
        pass
    with brain._last_action_lock:
        brain._last_action.update({
            "kind": "", "command": "", "summary": "",
            "done": False, "seq": 0, "at": 0.0,
        })
    with brain._chain_followup_lock:
        brain._last_chain.update({"plan": None, "results": None, "at": 0.0})
    yield
    screen_state.set_enabled(False, reason="test_live_fix_32 teardown")
    screen_state.clear_interactions()
    screen_state.clear_pending_plan()
    screen_state.clear_refused_chain()
    approvals.clear()
    screen_control._clear_window_enumeration()


class TestB82ComplaintFeedbackCanonical:
    """B82: a complaint is re-planned, never answered as a census/status."""

    def test_t2_second_window_still_up_is_a_complaint(self):
        msg = "the second window is still up jarvis you didnt minimize both"
        assert brain._is_screen_action_complaint(msg) is True

    def test_t11_named_terminal_window_complaint_is_recognized(self):
        assert brain._is_screen_action_complaint(
            "you didnt minimize the terminal window") is True

    def test_complaint_replans_the_recent_window_command(self):
        cmd = "minimize all the windows on my screen except for brave"
        brain.note_executed_action("screen_control", cmd,
                                   "Minimizing 2 windows, sir. Leaving Brave.")
        screen_state.note_request(cmd)
        msg = "the second window is still up jarvis you didnt minimize both"
        assert brain._screen_action_complaint_command(msg) == cmd

    def test_genuine_question_and_new_command_are_not_complaints(self):
        for t in ("are 3 windows still open?",
                  "did you minimize the window?",
                  "minimize 3 windows",
                  "minimize all the windows on my screen except for brave"):
            assert brain._is_screen_action_complaint(t) is False, t
            assert screen_control._is_window_count_feedback(t) is False, t

    def test_feedback_noop_plan_answers_honestly_not_a_census(self, monkeypatch):
        summary = ("Everything else is already put away, sir — the only "
                   "window up is Brave.")
        monkeypatch.setattr(screen_control, "_window_feedback_retry_command",
                            lambda: "minimize all the windows except brave")
        monkeypatch.setattr(
            screen_control, "_build_direct_plan",
            lambda cmd: {"ok": True, "steps": [], "summary": summary})
        reply = screen_control.maybe_handle_screen_control_message(
            "3 windows are still open")
        assert reply == summary
        assert "window open on your screen" not in reply


class TestB83OwnWindowIdentity:
    """B83: is_own_window matches only Jarvis's real overlay, never a
    repo-path console or any user window that merely mentions 'jarvis'."""

    def _stub(self, monkeypatch, title, cls="", proc=""):
        monkeypatch.setattr(screen_capture, "_get_window_title",
                            lambda hwnd: title)
        monkeypatch.setattr(screen_capture, "_window_class_name",
                            lambda hwnd: cls)
        monkeypatch.setattr(screen_capture, "_window_process_name",
                            lambda hwnd: proc)

    def test_repo_path_console_is_not_own_window(self, monkeypatch):
        self._stub(
            monkeypatch,
            r"C:\Users\mayan\jarvis-assistant\backend\venv\Scripts\python.exe",
            "ConsoleWindowClass", "cmd.exe")
        assert screen_capture.is_own_window(12345) is False

    def test_terminal_titled_jarvis_is_not_own_window(self, monkeypatch):
        self._stub(monkeypatch, "jarvis",
                   "CASCADIA_HOSTING_WINDOW_CLASS", "windowsterminal.exe")
        assert screen_capture.is_own_window(22222) is False

    def test_editor_showing_the_repo_is_not_own_window(self, monkeypatch):
        self._stub(monkeypatch, "jarvis - Visual Studio Code",
                   "Chrome_WidgetWin_1", "code.exe")
        assert screen_capture.is_own_window(33333) is False

    def test_real_jarvis_overlay_is_still_excluded(self, monkeypatch):
        self._stub(monkeypatch, "Jarvis Overlay",
                   "Chrome_WidgetWin_1", "electron.exe")
        assert screen_capture.is_own_window(99999) is True


class TestB84PostActionLook:
    """B84: a real, bounded post-action look + UIA revalidation recovery."""

    def test_local_verify_uses_capture_haystack_when_uia_is_unreadable(
            self, monkeypatch):
        steps = [{"action": "type", "text": "mr.beast", "ui_uid": "obs1:2"}]
        monkeypatch.setattr(screen_control, "_read_typed_field_values",
                            lambda *a, **k: None)
        monkeypatch.setattr(screen_control, "_verification_text_haystack",
                            lambda cap: "youtube search mr.beast results")
        assert screen_control._verify_action_locally(
            steps, capture={"raw_vision_image": object()}) is True
        monkeypatch.setattr(screen_control, "_verification_text_haystack",
                            lambda cap: "youtube search results")
        assert screen_control._verify_action_locally(
            steps, capture={"raw_vision_image": object()}) is None

    def test_submit_key_detection(self):
        assert screen_control._steps_have_submit_key([
            {"action": "type", "text": "x"},
            {"action": "press", "keys": ["enter"]},
        ]) is True
        assert screen_control._steps_have_submit_key([
            {"action": "type", "text": "x"}]) is False

    def test_revalidation_failure_clears_uia_cache_and_replans_once(
            self, monkeypatch):
        screen_ui_elements._element_cache["stale_uid"] = {"wrapper": object()}
        attempts = {"exec": 0}

        def _fake_exec(steps, **kwargs):
            attempts["exec"] += 1
            if attempts["exec"] == 1:
                raise RuntimeError(
                    "The control this step targets could not be revalidated "
                    "in the live accessibility tree. Please ask me again so I "
                    "can re-plan.")

        monkeypatch.setattr(screen_control, "execute_steps", _fake_exec)
        monkeypatch.setattr(
            screen_control, "plan_command",
            lambda cmd: {"ok": True, "confidence": 0.95,
                         "summary": "Typing 'mr.beast', sir.",
                         "steps": [{"action": "type", "text": "mr.beast"}]})
        monkeypatch.setattr(screen_control, "_verify_action_with_steps",
                            lambda *a, **k: True)
        plan = {
            "ok": True,
            "confidence": 0.95,
            "summary": "Typing 'mr.beast', sir.",
            "steps": [{"action": "type", "text": "mr.beast",
                       "ui_uid": "stale_uid"}],
            "screen_generation": screen_state.generation(),
        }
        status, _out = screen_control._execute_plan_tagged(plan, "type mr.beast")
        assert status == "executed"
        assert attempts["exec"] == 2
        assert "stale_uid" not in screen_ui_elements._element_cache

    def test_filtered_out_but_visible_hwnd_is_a_leftover(self, monkeypatch):
        # Enumeration is non-empty (a live list exists) but the targeted hwnd
        # was filtered out; Win32 says it is still on screen and its state did
        # not reach the target — it must be reported, not counted as moved.
        monkeypatch.setattr(
            screen_control, "_list_managable_windows",
            lambda: [{"hwnd": 101, "title": "Brave", "iconic": True}])

        class _U:
            @staticmethod
            def IsWindow(h):
                return 1

            @staticmethod
            def IsWindowVisible(h):
                return 1

        class _W:
            user32 = _U

        class _C:
            windll = _W

        monkeypatch.setattr(screen_control, "ctypes", _C)
        monkeypatch.setattr(screen_control, "_verify_hwnd_state_steps",
                            lambda steps: False)
        leftovers = screen_control._collect_window_plan_leftovers(
            {"ok": True},
            [{"action": "minimize_hwnd", "hwnd": 999, "title": "Console"}])
        assert [l["hwnd"] for l in leftovers] == [999]


class TestB85ChainTypeClassification:
    """B85: a mangled/normal 'type' is a screen-CONTROL step, not a look."""

    def test_t7_stt_mangled_ype_is_a_control_step(self):
        chain = multi_intent.build_chain(
            "ype mr.beast in the search bar on my screen and press enter")
        assert chain is not None
        assert [s["kind"] for s in chain["steps"]] == [
            "screen_control", "screen_control"]
        assert chain["steps"][0]["text"].lower().startswith("type mr.beast")
        assert "look at your screen" not in multi_intent.render_ack(
            chain).lower()

    def test_juxtaposed_type_then_press_enter_splits(self):
        chain = multi_intent.build_chain("type mr.beast in it press enter")
        assert chain is not None
        assert [s["text"] for s in chain["steps"]] == [
            "type mr.beast in it", "press enter"]

    def test_file_write_stays_a_task_not_a_screen_type(self):
        assert multi_intent.build_chain("type the report in the file") is None

    def test_look_only_command_is_not_a_control_step(self):
        assert multi_intent.build_chain(
            "look at my screen and tell me what you see") is None


class TestB88SubmitDependsOnType:
    """B88: a submit step consumes the typing step it submits."""

    def test_submit_step_consumes_the_preceding_type_step(self):
        chain = multi_intent.build_chain(
            "type mr.beast in the search bar on my screen and press enter")
        assert chain is not None
        assert chain["steps"][1]["consumes"] == [0]


class TestB86BareRepeatAfterChain:
    """B86: 'do it again' after a chain re-runs the WHOLE chain."""

    @pytest.mark.parametrize("t", [
        "Again.", "again", "do it again", "one more time", "phir se",
        "Jarvis, again!", "ek baar aur", "yes do it again",
        "yes, do it again please",
    ])
    def test_repeat_shapes_match(self, t):
        assert brain._REPEAT_LAST_RE.match(t.strip())

    @pytest.mark.parametrize("t", [
        "minimize it again", "yes", "again, what did you say", "try again",
        "repeat that song",
    ])
    def test_non_repeat_shapes_do_not_match(self, t):
        assert not brain._REPEAT_LAST_RE.match(t.strip())

    def test_last_action_of_a_chain_is_recognized_as_chain(self):
        chain = multi_intent.build_chain(
            "click in the youtube search bar then type mr.beast in it and "
            "press enter")
        assert chain is not None
        with brain._chain_followup_lock:
            brain._last_chain.update({
                "plan": chain,
                "results": [{"status": "ok"}, {"status": "ok"},
                            {"status": "ok"}],
                "at": time.time(),
            })
        # The final chain step overwrote the action registry.
        brain.note_executed_action("screen_control", "press enter",
                                   "Pressing enter, sir.")
        assert brain._last_action_belongs_to_stored_chain() is True

    def test_standalone_action_after_chain_is_not_a_chain_repeat(self):
        chain = multi_intent.build_chain(
            "click in the youtube search bar then type mr.beast in it")
        with brain._chain_followup_lock:
            brain._last_chain.update({
                "plan": chain, "results": [{"status": "ok"}],
                "at": time.time() - 30.0,
            })
        brain.note_executed_action(
            "screen_control", "minimize notepad",
            "Minimizing Notepad, sir.")
        assert brain._last_action_belongs_to_stored_chain() is False


class TestB87WholeTaskRepeatDeflection:
    """B87: natural whole-task repeats match and never become a browser task."""

    @pytest.mark.parametrize("t", [
        "do the whole task again", "redo the whole task",
        "run the whole thing again", "start it all over again",
        "do it again all over", "do the task from start",
        "do the task from the beginning",
        "dont ask me to press enter do the whole task again im asking you",
    ])
    def test_whole_task_repeat_shapes_match(self, t):
        assert brain._is_whole_task_repeat(t) is True

    @pytest.mark.parametrize("t", [
        "again", "do the rest", "do the whole task", "continue",
        "tell me about the whole task again",
    ])
    def test_non_whole_task_shapes_do_not_match(self, t):
        assert brain._is_whole_task_repeat(t) is False

    def test_handle_opencode_task_deflects_a_screen_repeat(self, monkeypatch):
        monkeypatch.setattr(
            brain, "_rerun_whole_chain",
            lambda msg, **kw: "Sir, I'll do this in 3 steps: click, type, "
                              "press enter. I'll report back when it's done.")
        reply = brain.handle_opencode_task(
            "Repeat the previous task from the beginning.",
            "do the task from start")
        assert "3 steps" in reply
        assert "Do you want me to go ahead and execute it?" not in reply
