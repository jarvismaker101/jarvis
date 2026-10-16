"""Live Fix 34 regression tests for bugs B96-B101.

One class per bug. Stubbed Win32/UIA/network; ledgers cleared per test; the
screen-state path is redirected to an isolated temp file so a test run can
never toggle the live backend's state.
"""

import json
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
    ``jarvis_screen_toggle.json`` the live backend rehydrates.
    """
    monkeypatch.setenv(
        "JARVIS_SCREEN_STATE_PATH",
        os.path.join(tempfile.gettempdir(), "jarvis_test_live_fix_34_toggle.json"))
    screen_state.set_enabled(True, reason="test_live_fix_34 setup")
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
    screen_state.set_enabled(False, reason="test_live_fix_34 teardown")
    screen_state.clear_interactions()
    screen_state.clear_pending_plan()
    screen_state.clear_refused_chain()
    approvals.clear()
    screen_control._clear_window_enumeration()


#: The live 2026-10-15 capture frame: 1920x1080 desktop, normalized 0..1000.
CAP = {
    "vision_width": 1000, "vision_height": 1000,
    "capture_width": 1920, "capture_height": 1080,
    "origin_left": 0, "origin_top": 0,
    "capture_mode": "primary_screen",
    "window_title": "",
    "image_data_url": "data:image/jpeg;base64,stub",
}


def _vision_plan(steps, coordinate_space="normalized_1000"):
    return {
        "ok": True, "confidence": 0.9, "summary": "Doing that, sir.",
        "needs_confirmation": False, "reason": "",
        "coordinate_space": coordinate_space,
        "steps": steps,
    }


def _cascade_response(obj):
    return {"choices": [{"message": {"content": json.dumps(obj)}}]}


class TestB96BoxCentreSemantics:
    """B96: {x, y, width, height} carries the CENTRE in x,y (prompt contract).

    The executor used to add width/2, height/2 to every box centre, so every
    box click landed half a box away (the live YouTube search-bar miss).
    """

    def test_xywh_box_returns_the_centre_itself(self):
        c = screen_control._extract_box_center(
            {"action": "click", "x": 469, "y": 120, "width": 354, "height": 33})
        assert c == (469, 120)
        c = screen_control._extract_box_center(
            {"box": {"x": 305, "y": 410, "width": 250, "height": 260}})
        assert c == (305, 410)

    def test_ltrb_box_still_returns_its_midpoint(self):
        c = screen_control._extract_box_center(
            {"bbox": {"left": 100, "top": 50, "right": 120, "bottom": 60}})
        assert c == (110, 55)

    def test_legacy_left_top_width_height_keeps_corner_math(self):
        c = screen_control._extract_box_center(
            {"box": {"left": 100, "top": 50, "width": 20, "height": 10}})
        assert c == (110, 55)

    def test_plain_point_has_no_box(self):
        assert screen_control._extract_box_center(
            {"action": "click", "x": 100, "y": 50}) is None

    def test_live_search_bar_box_lands_its_centre_in_screen_pixels(self):
        # Live log 14:42:21: the box {x:469,y:120,w:354,h:33} was executed as
        # (1240,147)px — the bar's far edge — instead of the centred point.
        out = screen_control._normalize_vision_plan(
            _vision_plan([{"action": "click", "x": 469, "y": 120,
                           "width": 354, "height": 33}]), dict(CAP))
        assert out["ok"] is True
        step = out["steps"][0]
        assert (step["x"], step["y"]) == (900, 130)
        assert step["space"] == "screen_pixels"

    def test_live_video_box_lands_its_centre_in_screen_pixels(self):
        out = screen_control._normalize_vision_plan(
            _vision_plan([{"action": "click", "x": 305, "y": 410,
                           "width": 250, "height": 260}]), dict(CAP))
        step = out["steps"][0]
        assert (step["x"], step["y"]) == (586, 443)

    def test_prompt_pins_the_centre_contract(self):
        tree_prompt = screen_control._build_tree_prompt(
            "click the search bar", dict(CAP))
        assert "never a corner" in tree_prompt
        assert '"x": 420, "y": 125, "width": 200, "height": 30' in tree_prompt
        image_only_prompt = screen_control._build_tree_prompt(
            "click the search bar", dict(CAP), image_only=True)
        assert "IS the centre" in image_only_prompt


class TestB97CoordinateRepairAndRetry:
    """B97: out-of-space points get a bounded repair, then one corrective
    vision retry — never an instant fatal rejection of a fixable plan."""

    def test_pixel_scale_point_declared_normalized_is_repaired(self):
        r = screen_control._repair_out_of_space_point(
            "normalized_1000", 960, 1040, dict(CAP))
        assert r == ("screen_pixels", 960, 1040)

    def test_point_outside_the_capture_still_rejects(self):
        assert screen_control._repair_out_of_space_point(
            "normalized_1000", 2500, 1400, dict(CAP)) is None
        out = screen_control._normalize_vision_plan(
            _vision_plan([{"action": "click", "x": 2500, "y": 1400}]),
            dict(CAP))
        assert out["ok"] is False
        assert "outside the captured screen area" in out["reason"]

    def test_normalize_reports_the_repaired_screen_point(self):
        out = screen_control._normalize_vision_plan(
            _vision_plan([{"action": "click", "x": 960, "y": 1040}]),
            dict(CAP))
        assert out["ok"] is True
        step = out["steps"][0]
        assert (step["x"], step["y"], step["space"]) == (
            960, 1040, "screen_pixels")

    def test_one_corrective_retry_recovers_a_fixable_plan(self, monkeypatch):
        calls = []

        def fake_cascade(prompt, image, **kw):
            calls.append(prompt)
            if len(calls) == 1:
                return _cascade_response(_vision_plan(
                    [{"action": "click", "x": 2500, "y": 1400}]))
            return _cascade_response(_vision_plan(
                [{"action": "click", "x": 500, "y": 500}]))

        monkeypatch.setattr(screen_control, "_ask_vision_cascade", fake_cascade)
        plan = screen_control._request_vision_plan(
            "click the search bar", dict(CAP))
        assert len(calls) == 2
        assert "CORRECTION" in calls[1]
        assert plan["ok"] is True
        assert (plan["steps"][0]["x"], plan["steps"][0]["y"]) == (960, 540)

    def test_no_retry_for_a_valid_plan(self, monkeypatch):
        calls = []

        def fake_cascade(prompt, image, **kw):
            calls.append(prompt)
            return _cascade_response(_vision_plan(
                [{"action": "click", "x": 500, "y": 500}]))

        monkeypatch.setattr(screen_control, "_ask_vision_cascade", fake_cascade)
        plan = screen_control._request_vision_plan(
            "click the search bar", dict(CAP))
        assert len(calls) == 1
        assert plan["ok"] is True

    def test_no_retry_for_a_non_coordinate_rejection(self, monkeypatch):
        calls = []

        def fake_cascade(prompt, image, **kw):
            calls.append(prompt)
            # No coordinate_space declared -> a schema rejection, not a
            # coordinate-frame problem; a retry cannot fix it.
            return _cascade_response({
                "ok": True, "confidence": 0.9, "summary": "s",
                "needs_confirmation": False, "reason": "",
                "steps": [{"action": "click", "x": 100, "y": 100}],
            })

        monkeypatch.setattr(screen_control, "_ask_vision_cascade", fake_cascade)
        plan = screen_control._request_vision_plan(
            "click the search bar", dict(CAP))
        assert len(calls) == 1
        assert plan["ok"] is False


class TestB98ReferentConsumes:
    """B98: a step whose object is the previous steps' OUTPUT consumes every
    preceding screen-control step, so a failed search skips the results
    click instead of clicking whatever is on screen."""

    def test_live_search_chain_links_the_results_step(self):
        chain = multi_intent.build_chain(
            "now type mr.beast in youtube search bar press enter and play "
            "the first video from the search results")
        assert chain is not None
        steps = chain["steps"]
        assert steps[0]["consumes"] == []
        last = steps[-1]
        assert "first video" in last["text"]
        assert last["consumes"] == [0]

    def test_full_split_chain_consumes_all_preceding_control_steps(self):
        chain = multi_intent.build_chain(
            "click in the youtube search bar type mr.beast and press enter "
            "then play the first video that appears for the search")
        assert chain is not None
        steps = chain["steps"]
        last = steps[-1]
        assert last["consumes"] == [0, 1, 2]

    def test_look_then_play_consumes_nothing(self):
        # The "first video" step depends on screen-CONTROL steps only; a bare
        # prior look is no prerequisite of its own.
        chain = multi_intent.build_chain(
            "look at my screen and play the first video")
        assert chain is not None
        assert chain["steps"][1]["consumes"] == []

    def test_pronoun_chain_keeps_its_ordinal_link(self):
        chain = multi_intent.build_chain(
            "minimize the terminal window and then close it")
        assert chain["steps"][1]["consumes"] == [0]


class TestB98EffectLookAndChainHonesty:
    """B98: a typing plan gets a post-action vision look by default, and a
    look-disproven chain step is a FAILURE so dependents skip it."""

    def _run_verify(self, monkeypatch, steps):
        calls = []

        def fake_cascade(prompt, image, **kw):
            calls.append(prompt)
            return _cascade_response(
                {"verified": False,
                 "observation": "the search bar is empty"})

        monkeypatch.setattr(screen_control, "_verify_hwnd_state_steps",
                            lambda *a, **kw: None)
        monkeypatch.setattr(screen_control, "_verify_not_wrong_bar",
                            lambda *a, **kw: None)
        monkeypatch.setattr(screen_control, "_cloud_verification_enabled",
                            lambda: False)
        monkeypatch.setattr(screen_control, "_verify_action_locally",
                            lambda *a, **kw: None)
        monkeypatch.setattr(screen_control, "capture_active_window",
                            lambda: dict(CAP))
        monkeypatch.setattr(screen_control, "_ask_vision_cascade", fake_cascade)
        monkeypatch.setattr("time.sleep", lambda s: None)
        verdict = screen_control._verify_action_with_steps(
            "Typing 'mr.beast' in the YouTube search bar, sir.",
            steps=steps, hwnd=None,
            command_text="type mr.beast in youtube search bar")
        return verdict, calls

    def test_typed_plan_gets_the_default_effect_look(self, monkeypatch):
        verdict, calls = self._run_verify(
            monkeypatch, [{"action": "click", "x": 1, "y": 1},
                          {"action": "type", "text": "mr.beast"}])
        assert verdict is False
        assert len(calls) == 1
        assert "STATED EFFECT" in calls[0]

    def test_pointer_plan_keeps_the_opt_in_gate(self, monkeypatch):
        verdict, calls = self._run_verify(
            monkeypatch, [{"action": "click", "x": 1, "y": 1}])
        assert verdict is None
        assert calls == []

    def test_unsure_type_step_is_failed_so_dependents_skip(
            self, monkeypatch):
        monkeypatch.setattr(brain, "_chain_stop_requested", lambda: False)
        monkeypatch.setattr(
            brain, "plan_command",
            lambda text: {
                "ok": True, "confidence": 0.95,
                "summary": "Pressing enter, sir.",
                "steps": [{"action": "type", "text": "mr.beast"}],
                "screen_generation": screen_state.generation(),
            })
        monkeypatch.setattr(
            brain, "_execute_plan_tagged",
            lambda plan, text, **kw: (
                "executed",
                "Pressing enter, sir. Though I'm not fully sure it worked - "
                "want me to try again?"))
        res = brain._mi_screen_control_step(
            {"kind": "screen_control", "text": "press enter"})
        assert res["status"] == "failed"
        assert "not fully sure" in res["fragment"]

    def test_confirmed_type_step_is_ok_when_nothing_disproves_it(
            self, monkeypatch):
        monkeypatch.setattr(brain, "_chain_stop_requested", lambda: False)
        monkeypatch.setattr(
            brain, "plan_command",
            lambda text: {
                "ok": True, "confidence": 0.95,
                "summary": "Typing mr.beast, sir.",
                "steps": [{"action": "type", "text": "mr.beast"}],
                "screen_generation": screen_state.generation(),
            })
        monkeypatch.setattr(
            brain, "_execute_plan_tagged",
            lambda plan, text, **kw: ("executed", "Typing mr.beast, sir."))
        res = brain._mi_screen_control_step(
            {"kind": "screen_control", "text": "type mr.beast"})
        assert res["status"] == "ok"


class TestB99EpochGateScope:
    """B99: the capture-epoch refusal protects USER-CONFIRMED previews only;
    auto-executed plans survive our own in-house re-captures."""

    def test_auto_plan_survives_a_moved_epoch(self, monkeypatch):
        marker = object()
        monkeypatch.setattr(screen_capture, "current_capture_epoch",
                            lambda: 8)
        monkeypatch.setattr(
            "backend.services.screen_state.effect_gate",
            lambda stamp: marker)
        gate = screen_control._plan_gate(
            {"screen_generation": 1, "capture_epoch": 7})
        assert gate() is marker

    def test_confirmed_plan_is_refused_when_the_epoch_moved(
            self, monkeypatch):
        monkeypatch.setattr(screen_capture, "current_capture_epoch",
                            lambda: 8)
        gate = screen_control._plan_gate(
            {"screen_generation": 1, "capture_epoch": 7,
             "user_confirmed": True})
        with pytest.raises(RuntimeError) as ctx:
            gate()
        assert "screen changed" in str(ctx.value)


class TestB100DeicticComplaintChainRerun:
    """B100: a deictic repeat or a complaint about the last thing re-runs the
    WHOLE stored chain when the last action belongs to it."""

    CHAIN_TEXT = ("click in the youtube search bar then type mr.beast in it "
                  "and press enter")

    def _seed_chain_as_last_action(self):
        chain = multi_intent.build_chain(self.CHAIN_TEXT)
        assert chain is not None
        with brain._chain_followup_lock:
            brain._last_chain.update({
                "plan": chain,
                "results": [{"status": "ok"}] * len(chain["steps"]),
                "at": time.time(),
            })
        brain.note_executed_action("screen_control", "press enter",
                                   "Pressing enter, sir.")
        return chain

    def _quiet_process(self, monkeypatch):
        monkeypatch.setattr(brain, "_remember_user_turn",
                            lambda *a, **kw: None)
        monkeypatch.setattr(brain, "add_message",
                            lambda *a, **kw: None)

    def test_deictic_repeat_after_a_chain_reruns_the_whole_chain(            self, monkeypatch):
        self._seed_chain_as_last_action()
        self._quiet_process(monkeypatch)
        calls = []

        def fake_rerun(msg, **kw):
            calls.append(msg)
            return "Sir, re-running the whole search chain from the top."

        monkeypatch.setattr(brain, "_rerun_whole_chain", fake_rerun)
        # [LF34 precedence] with no pending source answering the repeat, the
        # stored chain fallback takes over.
        monkeypatch.setattr(brain, "_deictic_rerun_source",
                            lambda: "")
        reply = brain.process_message("do that task again", sync_voice=False)
        assert "whole search chain" in reply
        assert calls == ["do that task again"]

    def test_complaint_after_a_chain_reruns_the_whole_chain(
            self, monkeypatch):
        self._seed_chain_as_last_action()
        self._quiet_process(monkeypatch)
        calls = []
        monkeypatch.setattr(
            brain, "_rerun_whole_chain",
            lambda msg, **kw: calls.append(msg) or
            "Sir, the whole chain didn't take - redoing it from step 1.")
        monkeypatch.setattr(
            brain, "_screen_action_complaint_command",
            lambda msg: "minimize notepad")
        reply = brain.process_message(
            "the second window is still up jarvis you didnt minimize both",
            sync_voice=False)
        assert "redoing it from step 1" in reply
        assert calls


class TestB101LastTaskRepeat:
    """B101: "do the last task again" is a whole-task repeat — it re-runs the
    stored chain instead of being re-confirmed as a new opencode job."""

    @pytest.mark.parametrize("t", [
        "do the last task again", "run the previous task once more",
        "do my last task again", "repeat the last task once more",
        "restart the last task from the start",
        "ok dont ask me again do the last task again",
    ])
    def test_last_task_repeat_shapes_match(self, t):
        assert brain._is_whole_task_repeat(t) is True

    @pytest.mark.parametrize("t", [
        "do the last task", "tell me about the last task again",
        "what was the last task again?", "do the rest",
    ])
    def test_non_repeat_shapes_do_not_match(self, t):
        assert brain._is_whole_task_repeat(t) is False

    def test_task_path_deflects_to_the_chain_rerun(self, monkeypatch):
        monkeypatch.setattr(
            brain, "_rerun_whole_chain",
            lambda msg, **kw: "Sir, re-running the last task from step 1.")
        monkeypatch.setattr(
            brain, "_arm_opencode_confirmation",
            lambda *a, **kw: (_ for _ in ()).throw(
                AssertionError("must not arm a new task")))
        reply = brain.handle_opencode_task(
            "do the last task again", "do the last task again")
        assert "re-running the last task" in reply
        assert "Do you want me to go ahead" not in reply
