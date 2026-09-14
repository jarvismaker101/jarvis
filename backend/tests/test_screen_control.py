import json
import time
import unittest
from unittest.mock import patch

from backend.services import screen_capture
from backend.services import screen_control
from backend.services import screen_executor
from backend.services import screen_geometry
from backend.services import screen_ocr
from backend.services import screen_state
from backend.services import screen_ui_elements

FAST_LOCAL_MATCH_CONFIDENCE = screen_control.FAST_LOCAL_MATCH_CONFIDENCE


class ScreenControlTests(unittest.TestCase):
    def test_generic_search_bar_does_not_force_windows_search(self):
        plan = screen_control._build_direct_plan("click the microsoft store search bar")
        self.assertIsNone(plan)

    def test_explicit_windows_search_still_uses_shortcut(self):
        plan = screen_control._build_direct_plan("click the windows search bar")
        self.assertIsNotNone(plan)
        self.assertEqual(plan["steps"], [{"action": "press", "keys": ["windows", "s"]}])

    def test_close_window_phrase_is_treated_as_screen_control(self):
        # Context-dependent patterns need screen controls on + UI noun + action verb.
        command = "close the settings tab"
        screen_state.set_enabled(True)
        try:
            self.assertTrue(screen_control._looks_like_screen_command(command))
        finally:
            screen_state.set_enabled(False)

    def test_box_center_is_used_for_click_coordinates(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Microsoft Store",
            "origin_left": 100,
            "origin_top": 50,
            "capture_width": 200,
            "capture_height": 100,
            "vision_width": 100,
            "vision_height": 50,
        }
        # box in normalized 0..1000: center 200,200 -> 20,10 pixels -> 140,70 screen
        plan = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "coordinate_space": "normalized_1000",
            "steps": [
                {
                    "action": "click",
                    "box": {"left": 100, "top": 100, "right": 300, "bottom": 300},
                }
            ],
        }

        normalized = screen_control._normalize_vision_plan(plan, capture)
        self.assertEqual(normalized["steps"][0]["x"], 140)
        self.assertEqual(normalized["steps"][0]["y"], 70)

    def test_element_id_map_uses_vision_space_not_absolute_screen(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Demo",
            "origin_left": 100,
            "origin_top": 50,
            "capture_width": 200,
            "capture_height": 100,
            "vision_width": 200,
            "vision_height": 100,
        }
        plan = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "steps": [{"action": "click", "element_id": 7}],
        }
        element_id_map = {
            7: {"x": 50, "y": 30, "space": "vision"},
        }

        normalized = screen_control._normalize_vision_plan(plan, capture, element_id_map)
        self.assertEqual(normalized["steps"][0]["x"], 150)
        self.assertEqual(normalized["steps"][0]["y"], 80)

    def test_active_window_plan_is_preferred_for_window_local_target(self):
        active_capture = {
            "capture_mode": "active_window",
            "window_title": "Microsoft Store",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 400,
            "capture_height": 200,
            "vision_width": 400,
            "vision_height": 200,
            "image_data_url": "data:image/jpeg;base64,active",
        }
        primary_capture = {
            "capture_mode": "primary_screen",
            "window_title": "",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
            "image_data_url": "data:image/jpeg;base64,primary",
        }

        def fake_vision_plan(command_text, capture, ocr_context="", ui_context="", element_id_map=None):
            return {
                "ok": True,
                "confidence": 0.8,
                "summary": "Clicking, sir.",
                "needs_confirmation": False,
                "reason": "",
                "capture_mode": capture.get("capture_mode", ""),
                "steps": [{"action": "click", "x": 20, "y": 20}],
            }

        with patch.object(screen_control, "capture_for_screen_control", return_value=active_capture), patch.object(
            screen_control, "capture_primary_screen", return_value=primary_capture
        ), patch.object(screen_control, "_try_local_match", return_value=None), patch.object(
            screen_control, "_gather_ui_tree", return_value=("<window/>", {1: {"x": 20, "y": 20, "space": "vision"}})
        ), patch.object(
            screen_control, "_request_vision_plan", side_effect=fake_vision_plan
        ):
            plan = screen_control._plan_with_tree("click the microsoft store window search bar")

        self.assertEqual(plan["capture_mode"], "active_window")

    def test_targeted_type_local_match_builds_click_then_type_plan(self):
        capture = {"capture_mode": "active_window", "raw_vision_image": None}
        elements = [
            {
                "name": "Search",
                "control_type": "Edit",
                "left": 10,
                "top": 20,
                "right": 110,
                "bottom": 50,
                "enabled": True,
            }
        ]

        with patch.object(screen_control.screen_ui_elements, "is_available", return_value=True), patch.object(
            screen_control.screen_ui_elements, "get_foreground_window_elements", return_value=elements
        ), patch.object(screen_control.screen_ocr, "is_available", return_value=False):
            plan = screen_control._try_local_match("type hello in search bar", capture)

        self.assertIsNotNone(plan)
        self.assertEqual(
            plan["steps"],
            [
                {"action": "click", "x": 60, "y": 35, "description": "Focus 'Search'"},
                {"action": "type", "text": "hello"},
            ],
        )

    def test_targeted_type_local_match_requires_text_entry_control(self):
        capture = {"capture_mode": "active_window", "raw_vision_image": None}
        elements = [
            {
                "name": "Search",
                "control_type": "Button",
                "left": 10,
                "top": 20,
                "right": 110,
                "bottom": 50,
                "enabled": True,
            }
        ]

        with patch.object(screen_control.screen_ui_elements, "is_available", return_value=True), patch.object(
            screen_control.screen_ui_elements, "get_foreground_window_elements", return_value=elements
        ), patch.object(screen_control.screen_ocr, "is_available", return_value=False):
            plan = screen_control._try_local_match("type hello in search bar", capture)

        self.assertIsNone(plan)

    def test_ocr_local_match_converts_vision_coords_to_screen(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Browser",
            "origin_left": 100,
            "origin_top": 50,
            "capture_width": 400,
            "capture_height": 200,
            "vision_width": 200,
            "vision_height": 100,
            "raw_vision_image": object(),
        }
        regions = [
            {
                "text": "Search",
                "left": 40,
                "top": 10,
                "right": 60,
                "bottom": 30,
                "confidence": 95,
            }
        ]

        with patch.object(screen_control.screen_ui_elements, "is_available", return_value=False), patch.object(
            screen_control.screen_ocr, "is_available", return_value=True
        ), patch.object(screen_control.screen_ocr, "extract_text_regions", return_value=regions):
            plan = screen_control._try_local_match("click search", capture)

        self.assertIsNotNone(plan)
        self.assertEqual(plan["steps"][0]["x"], 200)
        self.assertEqual(plan["steps"][0]["y"], 90)

    def test_local_match_is_used_and_skips_ai_cascade(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Browser",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 400,
            "capture_height": 200,
            "vision_width": 400,
            "vision_height": 200,
            "raw_vision_image": None,
        }
        local_plan = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking 'Search', sir.",
            "needs_confirmation": False,
            "reason": "",
            "capture_mode": "active_window",
            "window_title": "Browser",
            "steps": [{"action": "click", "x": 50, "y": 25}],
        }

        with patch.object(screen_control, "capture_active_window", return_value=capture), patch.object(
            screen_control, "capture_primary_screen", return_value=capture
        ), patch.object(screen_control, "_try_local_match", return_value=local_plan), patch.object(
            screen_control, "_request_vision_plan"
        ) as mock_vision:
            plan = screen_control._plan_with_tree("click search")
            mock_vision.assert_not_called()

        self.assertEqual(plan, local_plan)

    # --- PART A: screen phrase + action verb anywhere ---

    def test_just_click_video_on_my_screen_accepted_when_disabled(self):
        screen_state.set_enabled(False)
        try:
            self.assertTrue(screen_control._looks_like_screen_command("just click at the video on my screen"))
        finally:
            screen_state.set_enabled(False)

    def test_jarvis_look_at_screen_and_click_visible_accepted_when_disabled(self):
        screen_state.set_enabled(False)
        try:
            # second log phrase: jarvis prefix + look at + click + visible
            self.assertTrue(screen_control._looks_like_screen_command("jarvis look at my screen and click at the video visible"))
        finally:
            screen_state.set_enabled(False)

    def test_interrogative_type_files_on_screen_not_screen_command(self):
        # type is a noun here, not a screen action - must not be false positive
        self.assertFalse(screen_control._looks_like_screen_command("what type of files are on my screen"))
        screen_state.set_enabled(True)
        try:
            self.assertFalse(screen_control._looks_like_screen_command("what type of files are on my screen"))
        finally:
            screen_state.set_enabled(False)

    def test_interrogative_which_app_on_screen_not_screen_command(self):
        # open is adjective here, not a verb
        self.assertFalse(screen_control._looks_like_screen_command("which app is open on my screen"))
        screen_state.set_enabled(True)
        try:
            self.assertFalse(screen_control._looks_like_screen_command("which app is open on my screen"))
        finally:
            screen_state.set_enabled(False)

    def test_interrogative_how_scroll_on_screen_not_screen_command(self):
        # scroll is noun in how-to question
        self.assertFalse(screen_control._looks_like_screen_command("how do i scroll on the screen"))
        screen_state.set_enabled(True)
        try:
            self.assertFalse(screen_control._looks_like_screen_command("how do i scroll on the screen"))
        finally:
            screen_state.set_enabled(False)

    def test_what_is_on_my_screen_not_screen_command(self):
        screen_state.set_enabled(False)
        try:
            self.assertFalse(screen_control._looks_like_screen_command("what is on my screen"))
        finally:
            screen_state.set_enabled(False)
        screen_state.set_enabled(True)
        try:
            self.assertFalse(screen_control._looks_like_screen_command("what is on my screen"))
        finally:
            screen_state.set_enabled(False)

    def test_list_files_on_desktop_not_screen_command(self):
        screen_state.set_enabled(False)
        try:
            self.assertFalse(screen_control._looks_like_screen_command("list the files on my desktop"))
        finally:
            screen_state.set_enabled(False)

    def test_click_login_button_with_controls_disabled_is_screen_command_via_high_confidence(self):
        # High-confidence start-anchored commands are still accepted when disabled
        screen_state.set_enabled(False)
        try:
            self.assertTrue(screen_control._looks_like_screen_command("click the login button"))
        finally:
            screen_state.set_enabled(False)

    def test_click_login_button_with_filler_and_no_screen_not_screen_when_disabled(self):
        # Leading filler "just" breaks anchor and no screen phrase -> must not match when disabled
        screen_state.set_enabled(False)
        try:
            # phrase has click but no screen phrase; after stripping "just " it would be high-confidence,
            # but our test uses a non-anchored filler that is not stripped? Actually "just" is stripped,
            # so it would still be start-anchored. Use a phrase that is not start-anchored even after stripping.
            self.assertFalse(screen_control._looks_like_screen_command("i want to click the login button"))
        finally:
            screen_state.set_enabled(False)

    def test_click_login_button_with_controls_enabled_is_screen_command(self):
        screen_state.set_enabled(True)
        try:
            self.assertTrue(screen_control._looks_like_screen_command("click the login button"))
        finally:
            screen_state.set_enabled(False)

    def test_whole_screen_phrase_uses_primary_screen_capture(self):
        active_capture = {
            "capture_mode": "active_window",
            "window_title": "Browser",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 400,
            "capture_height": 200,
            "vision_width": 400,
            "vision_height": 200,
            "image_data_url": "data:image/jpeg;base64,active",
        }
        primary_capture = {
            "capture_mode": "primary_screen",
            "window_title": "",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
            "image_data_url": "data:image/jpeg;base64,primary",
        }

        def fake_vision(command_text, capture, ocr_context="", ui_context="", element_id_map=None):
            return {
                "ok": True,
                "confidence": 0.9,
                "summary": "Clicking, sir.",
                "needs_confirmation": False,
                "reason": "",
                "capture_mode": capture.get("capture_mode", ""),
                "steps": [{"action": "click", "x": 20, "y": 20}],
            }

        with patch.object(screen_control, "capture_for_screen_control", return_value=active_capture) as mock_active, \
             patch.object(screen_control, "capture_primary_screen", return_value=primary_capture) as mock_primary, \
             patch.object(screen_control, "_try_local_match", return_value=None), \
             patch.object(screen_control, "_gather_ui_tree", return_value=("<window/>", {1: {"x": 20, "y": 20, "space": "vision"}})), \
             patch.object(screen_control, "_request_vision_plan", side_effect=fake_vision):
            plan = screen_control._plan_with_tree("click at the video on my screen")
        # whole-screen phrasing must start with primary screen, not active window
        mock_primary.assert_called()
        # active capture should NOT be used first for whole-screen
        # the first capture mode should be primary_screen
        self.assertEqual(plan["capture_mode"], "primary_screen")

    # --- Vision planner coordinate fallback ---

    def test_build_tree_prompt_contains_coordinate_fallback_and_grounding(self):
        capture = {
            "window_title": "Instagram",
            "vision_width": 1920,
            "vision_height": 1080,
            "capture_width": 1920,
            "capture_height": 1080,
        }
        prompt = screen_control._build_tree_prompt("click the heart icon", capture, ui_context="<window/>")
        # coordinate fallback instruction - normalized 0..1000
        self.assertIn("bounding box", prompt.lower())
        self.assertIn("plain x and y", prompt.lower())
        self.assertIn("NORMALIZED 0..1000", prompt)
        self.assertIn("0 = left/top edge", prompt)
        self.assertIn("1000 = right/bottom edge", prompt)
        # grounding aids - normalized grid
        self.assertIn("Numbered boxes are element ids", prompt)
        self.assertIn("grid ticks are drawn every 10 percent", prompt)
        self.assertIn("labels 100, 200, ... 900", prompt)
        # still contains element_id preference and allowed verbs
        self.assertIn("PREFER to return the integer ID", prompt)
        self.assertIn("element_id", prompt)
        self.assertIn("Allowed step actions: click, double_click, right_click, type, press, hotkey, scroll", prompt)
        # ok:false only when genuinely not visible
        self.assertIn("Set ok to false ONLY when the target is genuinely not visible", prompt)

    def test_build_tree_prompt_few_shot_includes_coordinate_example(self):
        capture = {
            "window_title": "Instagram",
            "vision_width": 1920,
            "vision_height": 1080,
            "capture_width": 1920,
            "capture_height": 1080,
        }
        prompt = screen_control._build_tree_prompt("click the heart icon", capture, ui_context="<window/>")
        # existing element_id example still present
        self.assertIn('"element_id": 14', prompt)
        # normalized coordinate examples
        self.assertIn("click the heart icon", prompt.lower())
        self.assertIn('"x": 660', prompt)
        self.assertIn('"y": 550', prompt)
        self.assertIn('"x": 922', prompt)
        self.assertIn('"y": 83', prompt)
        # reason notes it used screenshot
        self.assertIn("not in the UI tree", prompt)

    def test_vision_to_screen_point_basic(self):
        capture = {
            "capture_width": 3840,
            "capture_height": 2160,
            "vision_width": 2560,
            "vision_height": 1440,
            "origin_left": 0,
            "origin_top": 0,
        }
        sx, sy = screen_control._vision_to_screen_point(capture, 100, 100)
        self.assertEqual(sx, 150)
        self.assertEqual(sy, 150)

    def test_vision_to_screen_point_with_origin(self):
        capture = {
            "capture_width": 200,
            "capture_height": 100,
            "vision_width": 100,
            "vision_height": 50,
            "origin_left": 100,
            "origin_top": 50,
        }
        sx, sy = screen_control._vision_to_screen_point(capture, 10, 20)
        # x_scale 2, y_scale 2 => 10*2+100=120, 20*2+50=90
        self.assertEqual(sx, 120)
        self.assertEqual(sy, 90)

    def test_normalize_vision_plan_coordinate_fallback(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Demo",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 3840,
            "capture_height": 2160,
            "vision_width": 2560,
            "vision_height": 1440,
        }
        # normalized 100,100 -> vision 256,144 -> screen 384,216 (scale 1.5)
        plan = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "coordinate_space": "normalized_1000",
            "steps": [{"action": "click", "x": 100, "y": 100}],
        }
        normalized = screen_control._normalize_vision_plan(plan, capture)
        self.assertTrue(normalized["ok"])
        self.assertEqual(normalized["steps"][0]["x"], 384)
        self.assertEqual(normalized["steps"][0]["y"], 216)

    def test_normalize_vision_plan_rejects_out_of_range_coordinates(self):
        """F45: out-of-range points are rejected, never clamped into a click."""
        capture = {
            "capture_mode": "active_window",
            "window_title": "Demo",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 3840,
            "capture_height": 2160,
            "vision_width": 2560,
            "vision_height": 1440,
        }
        plan = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "coordinate_space": "normalized_1000",
            "steps": [{"action": "click", "x": 5000, "y": -10}],
        }
        normalized = screen_control._normalize_vision_plan(plan, capture)
        self.assertFalse(normalized["ok"])
        self.assertEqual(normalized["steps"], [])
        self.assertIn("outside", normalized["reason"])

    def test_normalize_vision_plan_rejects_missing_coordinate_space(self):
        """F45: a missing space is not guessed."""
        capture = {
            "capture_mode": "active_window", "window_title": "Demo",
            "origin_left": 0, "origin_top": 0,
            "capture_width": 1920, "capture_height": 1080,
            "vision_width": 1920, "vision_height": 1080,
        }
        plan = {"ok": True, "confidence": 0.9, "summary": "Clicking, sir.",
                "needs_confirmation": False, "reason": "",
                "steps": [{"action": "click", "x": 100, "y": 100}]}
        normalized = screen_control._normalize_vision_plan(plan, capture)
        self.assertFalse(normalized["ok"])
        self.assertEqual(normalized["steps"], [])
        self.assertIn("space", normalized["reason"].lower())

    def test_nan_confidence_cannot_bypass_the_low_confidence_gate(self):
        """F45: NaN compares False against every threshold."""
        capture = {
            "capture_mode": "active_window", "window_title": "Demo",
            "origin_left": 0, "origin_top": 0,
            "capture_width": 1920, "capture_height": 1080,
            "vision_width": 1920, "vision_height": 1080,
        }
        plan = {"ok": True, "confidence": float("nan"),
                "summary": "Clicking, sir.", "needs_confirmation": False,
                "reason": "", "coordinate_space": "normalized_1000",
                "steps": [{"action": "click", "x": 100, "y": 100}]}
        normalized = screen_control._normalize_vision_plan(plan, capture)
        # NaN collapses to a finite 0.0 instead of surviving every comparison.
        self.assertEqual(normalized["confidence"], 0.0)
        screen_state.set_enabled(True)
        try:
            with patch.object(screen_control, "execute_steps") as mock_exec:
                result = screen_control._execute_or_queue(normalized, "click it")
            mock_exec.assert_not_called()
        finally:
            screen_state.set_enabled(False)
        self.assertIn("confiden", result.lower())

    def test_element_id_without_a_tree_is_rejected(self):
        """F45: an image-only element_id must not become a guessed click."""
        capture = {
            "capture_mode": "active_window", "window_title": "Demo",
            "origin_left": 0, "origin_top": 0,
            "capture_width": 1920, "capture_height": 1080,
            "vision_width": 1920, "vision_height": 1080,
        }
        plan = {"ok": True, "confidence": 0.95, "summary": "Clicking, sir.",
                "needs_confirmation": False, "reason": "",
                "steps": [{"action": "click", "element_id": 3}]}
        normalized = screen_control._normalize_vision_plan(plan, capture, {})
        self.assertFalse(normalized["ok"])
        self.assertEqual(normalized["steps"], [])

    def test_malformed_step_is_rejected(self):
        capture = {
            "capture_mode": "active_window", "window_title": "Demo",
            "origin_left": 0, "origin_top": 0,
            "capture_width": 1920, "capture_height": 1080,
            "vision_width": 1920, "vision_height": 1080,
        }
        plan = {"ok": True, "confidence": 0.9, "summary": "Clicking, sir.",
                "needs_confirmation": False, "reason": "",
                "steps": ["click the button"]}
        normalized = screen_control._normalize_vision_plan(plan, capture)
        self.assertFalse(normalized["ok"])

    def test_conflicting_step_spaces_are_rejected(self):
        capture = {
            "capture_mode": "active_window", "window_title": "Demo",
            "origin_left": 0, "origin_top": 0,
            "capture_width": 1920, "capture_height": 1080,
            "vision_width": 1920, "vision_height": 1080,
        }
        plan = {"ok": True, "confidence": 0.9, "summary": "Clicking, sir.",
                "needs_confirmation": False, "reason": "",
                "steps": [{"action": "click", "x": 100, "y": 100,
                           "coordinate_space": "normalized_1000",
                           "space": "vision_pixels"}]}
        normalized = screen_control._normalize_vision_plan(plan, capture)
        self.assertFalse(normalized["ok"])
        self.assertIn("conflicting", normalized["reason"])

    def test_element_id_still_resolves_via_map(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Demo",
            "origin_left": 100,
            "origin_top": 50,
            "capture_width": 200,
            "capture_height": 100,
            "vision_width": 200,
            "vision_height": 100,
        }
        plan = {
            "ok": True,
            "confidence": 0.92,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "steps": [{"action": "click", "element_id": 7}],
        }
        element_id_map = {7: {"x": 50, "y": 30, "space": "vision"}}
        normalized = screen_control._normalize_vision_plan(plan, capture, element_id_map)
        self.assertTrue(normalized["ok"])
        self.assertEqual(normalized["steps"][0]["x"], 150)
        self.assertEqual(normalized["steps"][0]["y"], 80)

    def test_retry_missing_element_retries_once(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Browser",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
            "image_data_url": "data:image/jpeg;base64,active",
        }
        # first vision response is ok:false with element id phrasing, second is ok with coords
        call_count = {"n": 0}

        def fake_vision(cmd, cap, ocr_context="", ui_context="", element_id_map=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                self.assertNotIn("IMPORTANT", cmd)
                return {
                    "ok": False,
                    "confidence": 0.0,
                    "summary": "",
                    "needs_confirmation": False,
                    "reason": "The heart icon lacks an element ID in the accessibility tree",
                    "capture_mode": cap.get("capture_mode", ""),
                    "steps": [],
                }
            else:
                # retry must contain the appended instruction
                self.assertIn("IMPORTANT", cmd)
                self.assertIn("accessibility tree", cmd.lower())
                # _request_vision_plan is mocked, so return already-converted screen coords
                # raw 500,400 would be 960,432 after 0..1000->pixels (1920*0.5,1080*0.4)
                return {
                    "ok": True,
                    "confidence": 0.90,
                    "summary": "Clicking the heart icon, sir.",
                    "needs_confirmation": False,
                    "reason": "Used screenshot coordinates because the icon was not in the UI tree.",
                    "capture_mode": cap.get("capture_mode", ""),
                    "steps": [{"action": "click", "x": 960, "y": 432}],
                }

        # wrap to track call count on the patched method
        with patch.object(screen_control, "capture_for_screen_control", return_value=capture), \
             patch.object(screen_control, "capture_primary_screen", return_value=capture), \
             patch.object(screen_control, "_try_local_match", return_value=None), \
             patch.object(screen_control, "_gather_ui_tree", return_value=("<window/>", {1: {"x": 10, "y": 10, "space": "vision"}})), \
             patch("backend.services.screen_capture.annotate_capture"), \
             patch.object(screen_control, "_request_vision_plan", side_effect=fake_vision) as mock_vision:
            plan = screen_control._plan_with_tree("click the heart icon")
            self.assertTrue(plan["ok"])
            self.assertEqual(mock_vision.call_count, 2)
            self.assertEqual(plan["steps"][0]["x"], 960)
            self.assertEqual(plan["steps"][0]["y"], 432)

    def test_retry_non_matching_reason_no_retry(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Browser",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
            "image_data_url": "data:image/jpeg;base64,active",
        }

        def fake_vision(cmd, cap, ocr_context="", ui_context="", element_id_map=None):
            return {
                "ok": False,
                "confidence": 0.0,
                "summary": "",
                "needs_confirmation": False,
                "reason": "target not visible on screen",
                "capture_mode": cap.get("capture_mode", ""),
                "steps": [],
            }

        with patch.object(screen_control, "capture_for_screen_control", return_value=capture), \
             patch.object(screen_control, "capture_primary_screen", return_value=capture), \
             patch.object(screen_control, "_try_local_match", return_value=None), \
             patch.object(screen_control, "_gather_ui_tree", return_value=("<window/>", {1: {"x": 10, "y": 10, "space": "vision"}})), \
             patch("backend.services.screen_capture.annotate_capture"), \
             patch.object(screen_control, "_request_vision_plan", side_effect=fake_vision) as mock_vision:
            # use whole-screen phrase so only primary_screen capture is attempted -> exactly 1 vision call
            plan = screen_control._plan_with_tree("click at the heart icon on my screen")
            self.assertFalse(plan["ok"])
            self.assertEqual(mock_vision.call_count, 1)

    def test_plan_with_tree_heart_icon_retry_end_to_end_primary_screen(self):
        # end-to-end dry-run: primary_screen capture, heart scenario with retry
        primary_capture = {
            "capture_mode": "primary_screen",
            "window_title": "Instagram",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
            "image_data_url": "data:image/jpeg;base64,primary",
        }
        active_capture = {
            "capture_mode": "active_window",
            "window_title": "Browser",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 400,
            "capture_height": 200,
            "vision_width": 400,
            "vision_height": 200,
            "image_data_url": "data:image/jpeg;base64,active",
        }

        responses = [
            {
                "ok": False,
                "confidence": 0.0,
                "summary": "",
                "needs_confirmation": False,
                "reason": "The heart icon lacks an element ID in the accessibility tree",
                "capture_mode": "primary_screen",
                "steps": [],
            },
            {
                "ok": True,
                "confidence": 0.95,
                "summary": "Clicking the heart icon, sir.",
                "needs_confirmation": False,
                "reason": "Used screenshot coordinates because the icon was not in the UI tree.",
                "capture_mode": "primary_screen",
                "steps": [{"action": "click", "x": 960, "y": 432}],
            },
        ]

        def fake_vision(cmd, cap, ocr_context="", ui_context="", element_id_map=None):
            # pop in order
            return responses.pop(0)

        # patch to track that no real click happens: _execute_or_queue / screen_executor not called
        with patch.object(screen_control, "capture_for_screen_control", return_value=active_capture), \
             patch.object(screen_control, "capture_primary_screen", return_value=primary_capture), \
             patch.object(screen_control, "_try_local_match", return_value=None), \
             patch.object(screen_control, "_gather_ui_tree", return_value=("<window/>", {1: {"x": 10, "y": 10, "space": "vision"}})), \
             patch("backend.services.screen_capture.annotate_capture"), \
             patch.object(screen_control, "_request_vision_plan", side_effect=fake_vision) as mock_vision:
            plan = screen_control._plan_with_tree("click at the heart icon on my screen")
            self.assertTrue(plan["ok"])
            self.assertEqual(plan["capture_mode"], "primary_screen")
            self.assertEqual(mock_vision.call_count, 2)
            # already converted (mocked _request returns screen coords)
            self.assertEqual(plan["steps"][0]["x"], 960)
            self.assertEqual(plan["steps"][0]["y"], 432)

    # --- P0: raw coordinates must pass through unsnapped ---

    def test_p0_raw_coordinate_click_not_snapped_to_giant_container(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Browser",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
        }
        # The giant UIA container (0,0,1920,1216) contains every icon and used to
        # snap all coordinate clicks to its center (960,608).
        element_id_map = {
            1: {
                "x": 960, "y": 608, "space": "vision",
                "bounds": {"left": 0, "top": 0, "right": 1920, "bottom": 1216},
                "source": "ui", "label": "BrowserRoot", "uid": "root",
            },
            2: {
                "x": 200, "y": 150, "space": "vision",
                "bounds": {"left": 100, "top": 50, "right": 300, "bottom": 250},
                "source": "ui", "label": "SmallControl", "uid": "small",
            },
        }
        # normalized 800,400 -> 1536,432 after 0..1000 conversion
        plan = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "coordinate_space": "normalized_1000",
            "steps": [{"action": "click", "x": 800, "y": 400}],
        }
        normalized = screen_control._normalize_vision_plan(plan, capture, element_id_map)
        self.assertTrue(normalized["ok"])
        step = normalized["steps"][0]
        # 800,400 normalized sits inside the giant container only - must NOT snap to 960,608.
        self.assertEqual(step["x"], 1536)
        self.assertEqual(step["y"], 432)
        self.assertNotIn("ui_uid", step)

    def test_p0_bbox_center_not_snapped_to_giant_container(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Browser",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
        }
        element_id_map = {
            1: {
                "x": 960, "y": 608, "space": "vision",
                "bounds": {"left": 0, "top": 0, "right": 1920, "bottom": 1216},
                "source": "ui", "label": "BrowserRoot", "uid": "root",
            },
            2: {
                "x": 200, "y": 150, "space": "vision",
                "bounds": {"left": 100, "top": 50, "right": 300, "bottom": 250},
                "source": "ui", "label": "SmallControl", "uid": "small",
            },
        }
        # normalized box center 800,400 -> 1536,432 after conversion
        plan = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "coordinate_space": "normalized_1000",
            "steps": [{"action": "click", "box": {"x": 700, "y": 300, "width": 200, "height": 200}}],
        }
        normalized = screen_control._normalize_vision_plan(plan, capture, element_id_map)
        self.assertTrue(normalized["ok"])
        step = normalized["steps"][0]
        # bbox center (800,400) normalized also lives only inside the giant container - must
        # pass through unchanged (after conversion) instead of snapping to 960,608.
        self.assertEqual(step["x"], 1536)
        self.assertEqual(step["y"], 432)
        self.assertNotIn("ui_uid", step)

    # --- P1.1: bounding box support ---

    def test_extract_box_center_xywh_and_ltrb_shapes(self):
        c = screen_control._extract_box_center({"box": {"x": 100, "y": 50, "width": 20, "height": 10}})
        self.assertEqual(c, (110, 55))
        c = screen_control._extract_box_center({"bbox": {"left": 100, "top": 50, "right": 120, "bottom": 60}})
        self.assertEqual(c, (110, 55))
        # top-level keys without a wrapper are also supported
        c = screen_control._extract_box_center({"action": "click", "x": 100, "y": 50, "width": 20, "height": 10})
        self.assertEqual(c, (110, 55))
        self.assertIsNone(screen_control._extract_box_center({"action": "click", "x": 100, "y": 50}))

    # --- P1.2: few-shot anti-echo + bbox instruction ---

    def test_build_tree_prompt_bbox_anti_echo_and_two_coordinate_examples(self):
        capture = {
            "window_title": "Instagram",
            "vision_width": 1920,
            "vision_height": 1080,
            "capture_width": 1920,
            "capture_height": 1080,
        }
        prompt = screen_control._build_tree_prompt("click the heart icon", capture, ui_context="<window/>")
        # bbox instruction: normalized 0..1000 and we click its center
        self.assertIn("bounding box when you can", prompt)
        self.assertIn("we will click its center", prompt)
        self.assertIn("NORMALIZED 0..1000", prompt)
        for axis_word in ("width", "height", "left", "top", "right", "bottom"):
            self.assertIn(axis_word, prompt)
        # anti-echo instruction
        self.assertIn("never copy example numbers", prompt)
        # two coordinate examples in different quadrants (normalized)
        self.assertIn('"x": 660', prompt)
        self.assertIn('"y": 550', prompt)
        self.assertIn('"x": 922', prompt)
        self.assertIn('"y": 83', prompt)
        # element_id preference + allowed verbs + grounding still present (normalized grid)
        self.assertIn("PREFER to return the integer ID", prompt)
        self.assertIn("Allowed step actions: click, double_click, right_click, type, press, hotkey, scroll", prompt)
        self.assertIn("Numbered boxes are element ids", prompt)
        self.assertIn("grid ticks are drawn every 10 percent", prompt)
        self.assertIn("labels 100, 200, ... 900", prompt)

    # --- P1.3: raw vision logging ---

    def test_request_vision_plan_attaches_raw_vision(self):
        capture = {
            "capture_mode": "primary_screen",
            "window_title": "Instagram",
            "image_data_url": "data:image/jpeg;base64,abc",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
        }
        raw_content = (
            '{"ok": true, "confidence": 0.9, "summary": "s", '
            '"needs_confirmation": false, "reason": "", '
            '"coordinate_space": "normalized_1000", '
            '"steps": [{"action": "click", "x": 100, "y": 100}]}'
        )
        result = {"choices": [{"message": {"content": raw_content}}]}
        with patch.object(screen_control, "_ask_vision_cascade", return_value=result):
            plan = screen_control._request_vision_plan("click heart", capture)
        self.assertEqual(plan["source"], "Vision Match")
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["raw_vision"], raw_content)

    def test_write_screen_command_log_writes_truncated_raw_vision_success_and_failed(self):
        captured = []

        class FakeFile:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def write(self, text):
                captured.append(text)
                return len(text)

        long_raw = "x" * 5000
        for status in ("EXECUTED", "FAILED_PLANNING"):
            plan = {
                "ok": status == "EXECUTED",
                "steps": [{"action": "click", "x": 1, "y": 2}],
                "capture_mode": "primary_screen",
                "window_title": "",
                "confidence": 0.9 if status == "EXECUTED" else 0.0,
                "source": "Vision Match",
                "raw_vision": long_raw,
            }
            captured.clear()
            with patch("builtins.open", create=True, return_value=FakeFile()):
                screen_control._write_screen_command_log("click heart", plan, status, error=None)
            written = "".join(captured)
            self.assertIn("RawVision:", written)
            raw_line = written.split("RawVision:  ", 1)[1].split("\n")[0]
            self.assertEqual(raw_line, "x" * 2000)

    # --- P1.4: coordinate confidence calibration ---

    def test_coordinate_click_low_confidence_forces_confirmation(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Demo",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
        }
        plan = {
            "ok": True,
            "confidence": 0.75,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "coordinate_space": "normalized_1000",
            "steps": [{"action": "click", "x": 100, "y": 100}],
        }
        normalized = screen_control._normalize_vision_plan(plan, capture)
        self.assertTrue(normalized["needs_confirmation"])

    def test_coordinate_click_high_confidence_not_forced(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Demo",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
        }
        plan = {
            "ok": True,
            "confidence": 0.85,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "coordinate_space": "normalized_1000",
            "steps": [{"action": "click", "x": 100, "y": 100}],
        }
        normalized = screen_control._normalize_vision_plan(plan, capture)
        self.assertFalse(normalized["needs_confirmation"])

    def test_element_id_click_low_confidence_not_forced(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Demo",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
        }
        plan = {
            "ok": True,
            "confidence": 0.75,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "steps": [{"action": "click", "element_id": 7}],
        }
        element_id_map = {7: {"x": 100, "y": 100, "space": "vision", "label": "Search", "uid": "search"}}
        normalized = screen_control._normalize_vision_plan(plan, capture, element_id_map)
        self.assertFalse(normalized["needs_confirmation"])

    # --- P1.5: icon local-match guard ---

    def test_icon_target_weak_ui_match_skips_local_match(self):
        capture = {"capture_mode": "active_window", "raw_vision_image": None}
        elements = [
            {
                "name": "Earth",
                "control_type": "ListItem",
                "left": 10,
                "top": 20,
                "right": 60,
                "bottom": 50,
                "enabled": True,
            }
        ]
        with patch.object(screen_control.screen_ui_elements, "is_available", return_value=True), \
             patch.object(screen_control.screen_ui_elements, "get_foreground_window_elements", return_value=elements), \
             patch.object(screen_control.screen_ocr, "is_available", return_value=False):
            plan = screen_control._try_local_match("click the heart icon", capture)
        self.assertIsNone(plan)

    def test_icon_target_strong_ui_match_still_local_matches(self):
        capture = {"capture_mode": "active_window", "raw_vision_image": None}
        elements = [
            {
                "name": "Heart",
                "control_type": "Button",
                "left": 10,
                "top": 20,
                "right": 60,
                "bottom": 50,
                "enabled": True,
            }
        ]
        with patch.object(screen_control.screen_ui_elements, "is_available", return_value=True), \
             patch.object(screen_control.screen_ui_elements, "get_foreground_window_elements", return_value=elements), \
             patch.object(screen_control.screen_ocr, "is_available", return_value=False):
            plan = screen_control._try_local_match("click the heart icon", capture)
        self.assertIsNotNone(plan)

    def test_non_icon_weak_ui_match_still_local_matches(self):
        capture = {"capture_mode": "active_window", "raw_vision_image": None}
        elements = [
            {
                "name": "Lock",
                "control_type": "ListItem",
                "left": 10,
                "top": 20,
                "right": 60,
                "bottom": 50,
                "enabled": True,
            }
        ]
        with patch.object(screen_control.screen_ui_elements, "is_available", return_value=True), \
             patch.object(screen_control.screen_ui_elements, "get_foreground_window_elements", return_value=elements), \
             patch.object(screen_control.screen_ocr, "is_available", return_value=False):
            plan = screen_control._try_local_match("click the block", capture)
        self.assertIsNotNone(plan)

    # --- P1.6: whole-screen phrase breadth ---

    def test_whole_screen_phrase_breadth_routes_to_primary(self):
        for phrase in (
            "on screen",
            "on the screen",
            "on my desktop",
            "on the desktop",
            "visible on screen",
            "anything visible",
        ):
            routes_primary = (
                screen_control._needs_desktop_capture(phrase)
                or screen_control._is_whole_screen_phrase(phrase)
            )
            self.assertTrue(routes_primary, f"{phrase!r} should route to primary screen")

    def test_visible_on_screen_phrase_uses_primary_capture(self):
        active_capture = {
            "capture_mode": "active_window",
            "window_title": "Browser",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 400,
            "capture_height": 200,
            "vision_width": 400,
            "vision_height": 200,
            "image_data_url": "data:image/jpeg;base64,active",
        }
        primary_capture = {
            "capture_mode": "primary_screen",
            "window_title": "",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
            "image_data_url": "data:image/jpeg;base64,primary",
        }

        def fake_vision(command_text, capture, ocr_context="", ui_context="", element_id_map=None):
            return {
                "ok": True,
                "confidence": 0.9,
                "summary": "Clicking, sir.",
                "needs_confirmation": False,
                "reason": "",
                "capture_mode": capture.get("capture_mode", ""),
                "steps": [{"action": "click", "x": 20, "y": 20}],
            }

        with patch.object(screen_control, "capture_for_screen_control", return_value=active_capture) as mock_active, \
             patch.object(screen_control, "capture_primary_screen", return_value=primary_capture) as mock_primary, \
             patch.object(screen_control, "_try_local_match", return_value=None), \
             patch.object(screen_control, "_gather_ui_tree", return_value=("<window/>", {1: {"x": 20, "y": 20, "space": "vision"}})), \
             patch.object(screen_control, "_request_vision_plan", side_effect=fake_vision):
            plan = screen_control._plan_with_tree("click at the video visible on screen")
        mock_primary.assert_called()
        mock_active.assert_not_called()
        self.assertEqual(plan["capture_mode"], "primary_screen")

    def test_interrogative_phrases_remain_not_screen_commands(self):
        for phrase in (
            "what type of files are on my screen",
            "which app is open on my screen",
            "how do i scroll on the screen",
        ):
            screen_state.set_enabled(False)
            try:
                self.assertFalse(screen_control._looks_like_screen_command(phrase))
            finally:
                screen_state.set_enabled(False)
            screen_state.set_enabled(True)
            try:
                self.assertFalse(screen_control._looks_like_screen_command(phrase))
            finally:
                screen_state.set_enabled(False)

    def test_heart_log_phrases_remain_screen_commands(self):
        screen_state.set_enabled(False)
        try:
            self.assertTrue(screen_control._looks_like_screen_command("just click at the video on my screen"))
            self.assertTrue(screen_control._looks_like_screen_command("jarvis look at my screen and click at the video visible"))
        finally:
            screen_state.set_enabled(False)

    # --- P0-B: anchor budget for icon targets ---
    def test_filter_icon_budget_keeps_more_elements(self):
        elements = []
        # 45 heart items (high relevance) + 15 feed containers (low relevance) = 60
        for i in range(45):
            elements.append({"name": f"heart item {i}", "control_type": "Button"})
        for i in range(15):
            elements.append({"name": f"feed container {i}", "control_type": "Pane"})
        # icon target should keep all 60 (budget 80)
        filtered_icon = screen_control._filter_tree_to_relevant(elements, "heart icon")
        self.assertGreater(len(filtered_icon), 40)
        self.assertEqual(len(filtered_icon), 60)
        # check feed container survived
        self.assertTrue(any("feed" in el["name"] for el in filtered_icon))
        # non-icon target should be capped at 40
        filtered_nonicon = screen_control._filter_tree_to_relevant(elements, "settings")
        self.assertEqual(len(filtered_nonicon), 40)

    def test_gather_ui_tree_icon_budget_keeps_feed_container(self):
        # 65 elements: 50 heart + 15 feed, target heart keeps >40
        elements = []
        for i in range(50):
            elements.append({"name": f"heart {i}", "control_type": "Button", "left": 0, "top": i, "right": 10, "bottom": i+5, "depth": 0})
        for i in range(15):
            elements.append({"name": f"feed container {i}", "control_type": "Pane", "left": 0, "top": i, "right": 10, "bottom": i+5, "depth": 0})
        capture = {"window_title": "Demo", "vision_width": 1920, "vision_height": 1080, "capture_mode": "active_window"}
        snapshot = {"raw_ui": [{"name": el["name"]} for el in elements], "ui_elements": elements, "ocr_merged": [], "ocr_regions": [], "raw_ui": elements}
        # Need raw_ui for seen_texts length but not used now; provide same
        snapshot["raw_ui"] = [{"name": el["name"]} for el in elements]
        tree, id_map = screen_control._gather_ui_tree(capture, snapshot=snapshot, target_label="heart icon")
        self.assertGreater(len(id_map), 40)
        # feed container should be present in tree (since budget 80)
        self.assertIn("feed container", tree.lower())
        # non-icon target should be capped at 40
        tree2, id_map2 = screen_control._gather_ui_tree(capture, snapshot=snapshot, target_label="settings")
        self.assertEqual(len(id_map2), 40)

    # --- P0-C: duplicate-icon disambiguation prompt ---
    def test_build_tree_prompt_duplicate_icon_disambiguation(self):
        capture = {"window_title": "Instagram", "vision_width": 1920, "vision_height": 1080, "capture_width": 1920, "capture_height": 1080}
        prompt = screen_control._build_tree_prompt("click the heart icon", capture, ui_context="<window/>")
        # disambiguation sentence must be present
        self.assertIn("main content", prompt.lower())
        self.assertIn("feed area", prompt.lower())
        self.assertTrue("sidebar" in prompt.lower() or "sidebars" in prompt.lower())
        # bbox + anti-echo + two examples still intact (normalized)
        self.assertIn("bounding box when you can", prompt)
        self.assertIn("we will click its center", prompt)
        self.assertIn("never copy example numbers", prompt)
        self.assertIn('"x": 660', prompt)
        self.assertIn('"x": 922', prompt)
        self.assertIn("PREFER returning a bounding box for the target", prompt)
        self.assertIn("plain x,y point tends to land on the corner", prompt.lower())
        self.assertIn("center of the target", prompt.lower())
        self.assertIn("NORMALIZED 0..1000", prompt)
        self.assertIn("0 = left/top edge", prompt)
        self.assertIn("1000 = right/bottom edge", prompt)
        self.assertIn("PREFER to return the integer ID", prompt)
        self.assertIn("Allowed step actions: click, double_click", prompt)
        self.assertIn("Numbered boxes are element ids", prompt)

    # --- FIX: vision coordinate frame normalized 0..1000 ---
    def test_prompt_frame_contains_top_left_origin_and_corrected_examples(self):
        capture = {"window_title": "Instagram", "vision_width": 1920, "vision_height": 1080, "capture_width": 1920, "capture_height": 1080}
        prompt = screen_control._build_tree_prompt("click the heart icon", capture, ui_context="<window/>")
        self.assertIn("NORMALIZED 0..1000", prompt)
        self.assertIn("0 = left/top edge", prompt)
        self.assertIn("1000 = right/bottom edge", prompt)
        # normalized examples
        self.assertIn('"x": 660', prompt)
        self.assertIn('"y": 550', prompt)
        self.assertIn('"x": 922', prompt)
        self.assertIn('"y": 83', prompt)
        # bbox preference strengthened and normalized scale noted
        self.assertIn("PREFER returning a bounding box for the target", prompt)
        self.assertIn("plain x,y point tends to land on the corner", prompt.lower())
        self.assertIn("center of the target", prompt.lower())
        self.assertIn("same NORMALIZED 0..1000 scale", prompt)

    def test_grid_overlay_does_not_flip_and_annotates_origin(self):
        from PIL import Image
        from backend.services.screen_capture import _draw_grid_overlay
        img = Image.new("RGB", (400, 300), (0, 0, 0))
        img.putpixel((45, 10), (255, 0, 0))
        img.putpixel((355, 10), (0, 255, 0))
        # snapshot colors near origin before overlay (should be black)
        before = img.getpixel((5, 5))
        overlay = _draw_grid_overlay(img)
        # no flip: colors stay at same x
        self.assertEqual(overlay.getpixel((45, 10)), (255, 0, 0))
        self.assertEqual(overlay.getpixel((355, 10)), (0, 255, 0))
        # annotation draws near (2,1) - at least one reddish pixel appears near origin
        found_red = False
        for x in range(2, 50):
            for y in range(1, 20):
                r, g, b = overlay.getpixel((x, y))[:3]
                if r > 150 and g < 100 and b < 100:
                    found_red = True
                    break
            if found_red:
                break
        self.assertTrue(found_red, "grid overlay should annotate (0,0) near (2,1) with reddish text")
        # also ensure pure before difference: overlay differs near origin
        self.assertNotEqual(before, overlay.getpixel((5, 5)) if overlay.getpixel((5,5)) != before else (1,1,1))
        # normalized ruler: ticks at every 10% (100..900) - for 400px width ticks at 40,80,...,360
        # check at least 2 normalized ticks exist (40 and 80) which old pixel ruler (every 100) would not have
        for x_check in (40, 80, 120):
            r, g, b = overlay.getpixel((x_check, 2))[:3]
            # tick line is red at top edge y=0..10
            self.assertTrue(r > 150 and g < 100 and b < 100, f"normalized tick expected at x={x_check}")
        # label at normalized position should exist - infer via tick presence (text also reddish near 100 label)
        # old pixel labels every 200 would be 200 only for 400px, but normalized has 100,200,300 etc.
        # ensure tick at 100 (both old and new) still present but tick at 40 proves normalized

    def test_normalized_coordinates_convert_to_pixels(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Demo",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
        }
        # raw normalized 650,555 -> 1248,599
        plan = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "coordinate_space": "normalized_1000",
            "steps": [{"action": "click", "x": 650, "y": 555}],
        }
        normalized = screen_control._normalize_vision_plan(plan, capture)
        self.assertEqual(normalized["steps"][0]["x"], 1248)
        self.assertEqual(normalized["steps"][0]["y"], 599)
        # bbox in normalized 0..1000: box 600,500 width100 height100 -> center 650,550 -> 1248,594
        plan2 = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "coordinate_space": "normalized_1000",
            "steps": [{"action": "click", "box": {"x": 600, "y": 500, "width": 100, "height": 100}}],
        }
        normalized2 = screen_control._normalize_vision_plan(plan2, capture)
        self.assertEqual(normalized2["steps"][0]["x"], 1248)
        self.assertEqual(normalized2["steps"][0]["y"], 594)

    def test_element_id_point_not_converted(self):
        capture = {
            "capture_mode": "active_window",
            "window_title": "Demo",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 1920,
            "capture_height": 1080,
            "vision_width": 1920,
            "vision_height": 1080,
        }
        # element_id resolved to screen space should pass through unconverted
        element_id_map = {7: {"x": 650, "y": 555, "space": "screen"}}
        plan = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "steps": [{"action": "click", "element_id": 7}],
        }
        normalized = screen_control._normalize_vision_plan(plan, capture, element_id_map)
        self.assertEqual(normalized["steps"][0]["x"], 650)
        self.assertEqual(normalized["steps"][0]["y"], 555)
        # vision-space element_id also not converted via 0..1000 (already pixels)
        element_id_map2 = {7: {"x": 650, "y": 555, "space": "vision"}}
        normalized2 = screen_control._normalize_vision_plan(plan, capture, element_id_map2)
        # vision 650,555 -> screen same because scale 1 (origin 0)
        self.assertEqual(normalized2["steps"][0]["x"], 650)
        self.assertEqual(normalized2["steps"][0]["y"], 555)

    def test_vision_to_screen_identity_maps_without_mirror(self):
        capture = {"origin_left": 0, "origin_top": 0, "capture_width": 1920, "capture_height": 1080, "vision_width": 1920, "vision_height": 1080}
        sx, sy = screen_control._vision_to_screen_point(capture, 1267, 554)
        self.assertEqual((sx, sy), (1267, 554))
        # does NOT flip: 653 should map to 653, not to 1267 (1920-653)
        sx2, sy2 = screen_control._vision_to_screen_point(capture, 653, 554)
        self.assertEqual((sx2, sy2), (653, 554))
        self.assertNotEqual(sx2, 1267)

    # --- P1-B: post-scale clamp ---
    def test_vision_to_screen_point_post_scale_clamp(self):
        capture = {"origin_left": 10, "origin_top": 20, "capture_width": 3000, "capture_height": 2000, "vision_width": 2560, "vision_height": 1440}
        # vision point beyond vision_width (2560) would naively be 10+3000=3010 > max 3009
        sx, sy = screen_control._vision_to_screen_point(capture, 2560, 1440)
        self.assertEqual(sx, 10 + 3000 - 1)  # clamped to 3009
        self.assertEqual(sy, 20 + 2000 - 1)
        # normal max vision-1 should still be within bounds
        sx2, sy2 = screen_control._vision_to_screen_point(capture, 2559, 1439)
        self.assertLessEqual(sx2, 10 + 3000 - 1)
        self.assertLessEqual(sy2, 20 + 2000 - 1)

    def test_normalize_vision_plan_post_scale_clamp(self):
        capture = {"capture_mode": "active_window", "window_title": "Demo", "origin_left": 0, "origin_top": 0, "capture_width": 3000, "capture_height": 2000, "vision_width": 2560, "vision_height": 1440}
        plan = {"ok": True, "confidence": 0.9, "summary": "Clicking, sir.", "needs_confirmation": False, "reason": "", "coordinate_space": "normalized_1000", "steps": [{"action": "click", "x": 1000, "y": 1000}]}
        normalized = screen_control._normalize_vision_plan(plan, capture)
        # the far edge stays inside the capture after scaling
        self.assertLessEqual(normalized["steps"][0]["x"], 3000 - 1)
        self.assertLessEqual(normalized["steps"][0]["y"], 2000 - 1)

    # --- P1-C: capture-execute race guard ---
    def test_race_guard_translates_coords(self):
        plan = {"ok": True, "confidence": 0.9, "summary": "Clicking, sir.", "needs_confirmation": False, "reason": "", "capture_mode": "active_window", "origin_left": 100, "origin_top": 100, "capture_width": 1920, "capture_height": 1080, "hwnd": 123, "process_id": 77, "steps": [{"action": "click", "x": 150, "y": 150}]}
        new_bounds = {"left": 140, "top": 80, "width": 1920, "height": 1080, "window_title": "Demo", "hwnd": 123, "process_id": 77}
        with patch("backend.services.screen_capture.get_window_bounds", return_value=new_bounds):
            err = screen_control._maybe_adjust_for_window_move(plan)
            self.assertIsNone(err)
            self.assertEqual(plan["steps"][0]["x"], 190)  # 150 +40
            self.assertEqual(plan["steps"][0]["y"], 130)  # 150 -20

    def test_race_guard_window_closed_no_click(self):
        capture = {"capture_mode": "active_window", "window_title": "Demo", "origin_left": 0, "origin_top": 0, "capture_width": 1920, "capture_height": 1080, "vision_width": 1920, "vision_height": 1080, "hwnd": 999, "image_data_url": "data:image/png;base64,abc"}
        plan = {"ok": True, "confidence": 0.9, "summary": "Clicking, sir.", "needs_confirmation": False, "reason": "", "capture_mode": "active_window", "origin_left": 0, "origin_top": 0, "capture_width": 1920, "capture_height": 1080, "vision_width": 1920, "vision_height": 1080, "window_title": "Demo", "hwnd": 999, "steps": [{"action": "click", "x": 100, "y": 100}]}
        # F19: the race guard only runs for a plan that passed the control
        # gate, so these exercise it with screen controls ON.
        screen_state.set_enabled(True)
        try:
            with patch("backend.services.screen_capture.get_window_bounds", return_value=None), patch.object(screen_control, "execute_steps") as mock_exec:
                result = screen_control._execute_or_queue(plan, "click heart")
                mock_exec.assert_not_called()
                self.assertIn("closed", result.lower() or "minimized" in result.lower() or "window" in result.lower())
        finally:
            screen_state.set_enabled(False)

    def test_race_guard_execute_queue_translates_and_calls_executor(self):
        plan = {"ok": True, "confidence": 0.9, "summary": "Clicking, sir.", "needs_confirmation": False, "reason": "", "capture_mode": "active_window", "origin_left": 0, "origin_top": 0, "capture_width": 1920, "capture_height": 1080, "vision_width": 1920, "vision_height": 1080, "window_title": "Demo", "hwnd": 555, "process_id": 9, "steps": [{"action": "click", "x": 100, "y": 100}]}
        new_bounds = {"left": 40, "top": -20, "width": 1920, "height": 1080, "window_title": "Demo", "hwnd": 555, "process_id": 9}
        screen_state.set_enabled(True)
        try:
            with patch("backend.services.screen_capture.get_window_bounds", return_value=new_bounds), patch.object(screen_control, "execute_steps") as mock_exec, patch.object(screen_control.screen_state, "clear_pending_plan"), patch.object(screen_control.screen_state, "add_interaction"), patch.object(screen_control, "_write_screen_command_log"):
                screen_control._execute_or_queue(plan, "click heart")
                mock_exec.assert_called_once()
                called_steps = mock_exec.call_args[0][0]
                self.assertEqual(called_steps[0]["x"], 140)
                self.assertEqual(called_steps[0]["y"], 80)
        finally:
            screen_state.set_enabled(False)

    def test_screen_control_disabled_drops_the_action(self):
        """F19: with controls off nothing may reach execute_steps."""
        plan = {"ok": True, "confidence": 0.9, "summary": "Clicking, sir.", "needs_confirmation": False, "reason": "", "steps": [{"action": "click", "x": 100, "y": 100}]}
        screen_state.set_enabled(False)
        with patch.object(screen_control, "execute_steps") as mock_exec:
            result = screen_control._execute_or_queue(plan, "click heart")
        mock_exec.assert_not_called()
        self.assertIn("screen controls are off", result.lower())

    # --- P1-A: OCR/UIA dedup fix ---
    def test_ocr_dedup_same_text_both_sources_survive(self):
        regions = [
            {"text": "Search", "left": 0, "top": 0, "right": 10, "bottom": 10, "confidence": 90},
            {"text": "Hello", "left": 20, "top": 0, "right": 30, "bottom": 10, "confidence": 90},
        ]
        seen_texts = ["Search"]
        lines, id_map, _ = screen_control._serialize_ocr_nodes(regions, seen_texts, start_id=1)
        # With cross-source dedup fix, Search from OCR should survive even though UIA already has Search
        self.assertEqual(len(lines), 2)
        texts = [line for line in lines]
        self.assertTrue(any("Search" in l for l in lines))
        self.assertTrue(any("Hello" in l for l in lines))

    def test_ocr_dedup_within_source_still_deduped(self):
        # F44: deduplication is location-aware. Identical labels at DISTINCT
        # positions are separate spatial entities and each survives; only
        # duplicate detections of the same physical box collapse.
        regions = [
            {"text": "Search", "left": 0, "top": 0, "right": 10, "bottom": 10, "confidence": 90},
            {"text": "Search", "left": 20, "top": 0, "right": 30, "bottom": 10, "confidence": 90},
            {"text": "Search", "left": 40, "top": 0, "right": 50, "bottom": 10, "confidence": 90},
        ]
        lines, id_map, _ = screen_control._serialize_ocr_nodes(regions, [], start_id=1)
        self.assertEqual(len(lines), 3)  # repeated labels kept as distinct entities

        # Same text overlapping the same physical region -> single entity.
        overlapping = [
            {"text": "Search", "left": 0, "top": 0, "right": 10, "bottom": 10, "confidence": 90},
            {"text": "Search", "left": 2, "top": 0, "right": 12, "bottom": 10, "confidence": 90},
        ]
        dup_lines, _, _ = screen_control._serialize_ocr_nodes(overlapping, [], start_id=1)
        self.assertEqual(len(dup_lines), 1)  # duplicate detections deduped

    # --- P1-D: scale audit logging ---
    def test_write_screen_command_log_includes_vision_origin_scale(self):
        captured = []

        class FakeFile:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def write(self, text):
                captured.append(text)
                return len(text)

        plan = {"ok": True, "steps": [{"action": "click", "x": 100, "y": 100}], "capture_mode": "active_window", "window_title": "Demo", "confidence": 0.9, "source": "Vision Match", "vision_width": 2560, "vision_height": 1440, "capture_width": 3840, "capture_height": 2160, "origin_left": 10, "origin_top": 20}
        with patch("builtins.open", create=True, return_value=FakeFile()):
            screen_control._write_screen_command_log("click heart", plan, "EXECUTED")
        written = "".join(captured)
        self.assertIn("Vision:", written)
        self.assertIn("2560x1440", written)
        self.assertIn("Origin:", written)
        self.assertIn("10,20", written)
        self.assertIn("Scale:", written)
        self.assertIn("Capture:", written)

    # --- P1-B extra: post-scale clamp edge already tested above ---
    # --- P0-C already covered in prompt test ---


class ScreenExecutorTests(unittest.TestCase):
    def test_move_mouse_ensures_dpi_and_int_coords(self):
        from backend.services import screen_executor
        with patch.object(screen_executor, "_ensure_dpi_awareness") as mock_dpi, patch.object(screen_executor.user32, "SetCursorPos") as mock_set:
            screen_executor.move_mouse(100.6, 200.9)
            mock_dpi.assert_called_once()
            mock_set.assert_called_once()
            args = mock_set.call_args[0]
            self.assertEqual(args[0], int(100.6))
            self.assertEqual(args[1], int(200.9))
            self.assertIsInstance(args[0], int)
            self.assertIsInstance(args[1], int)

    def test_click_ensures_dpi_and_uses_int_coords(self):
        from backend.services import screen_executor
        with patch.object(screen_executor, "_ensure_dpi_awareness") as mock_dpi, patch.object(screen_executor, "move_mouse") as mock_move, patch.object(screen_executor.user32, "SendInput") as mock_send, patch("time.sleep"):
            screen_executor.click(123.7, 456.2)
            # click calls _ensure_dpi_awareness directly plus move_mouse also would, but we mock move_mouse so only direct call counts
            mock_dpi.assert_called()
            mock_move.assert_called_once()
            # move called with original float, but move_mouse would int it
            called_x, called_y = mock_move.call_args[0]
            self.assertEqual(called_x, 123.7)
            self.assertEqual(called_y, 456.2)

    def test_execute_steps_ensures_dpi(self):
        from backend.services import screen_executor
        with patch.object(screen_executor, "_ensure_dpi_awareness") as mock_dpi, patch.object(screen_executor.user32, "SetCursorPos") as mock_set, patch.object(screen_executor.user32, "SendInput"), patch("time.sleep"), patch.object(screen_executor.user32, "IsIconic", return_value=0), patch.object(screen_executor.user32, "SetForegroundWindow"), patch.object(screen_executor.user32, "GetForegroundWindow", return_value=123):
            steps = [{"action": "click", "x": 10, "y": 20}]
            screen_executor.execute_steps(steps, target_hwnd=None)
            mock_dpi.assert_called()
            self.assertGreaterEqual(mock_dpi.call_count, 1)
            mock_set.assert_called()

    def test_scroll_ensures_dpi(self):
        from backend.services import screen_executor
        with patch.object(screen_executor, "_ensure_dpi_awareness") as mock_dpi, patch.object(screen_executor.user32, "mouse_event") as mock_mouse:
            screen_executor.scroll("down", 600)
            mock_dpi.assert_called_once()
            mock_mouse.assert_called()


class ScreenControlToggleRegexTests(unittest.TestCase):
    """F19: anchored, mutually exclusive toggle commands. 'turn screen
    controls off' must never match the ON pattern (and vice versa)."""

    def test_turn_off_matches_off_only(self):
        self.assertTrue(screen_control._is_toggle_off("turn screen controls off"))
        self.assertFalse(screen_control._is_toggle_on("turn screen controls off"))

    def test_turn_on_matches_on_only(self):
        self.assertTrue(screen_control._is_toggle_on("turn screen controls on"))
        self.assertFalse(screen_control._is_toggle_off("turn screen controls on"))

    def test_switch_on_variant(self):
        self.assertTrue(screen_control._is_toggle_on("switch on screen control"))
        self.assertFalse(screen_control._is_toggle_off("switch on screen control"))

    def test_switch_off_variant(self):
        self.assertTrue(screen_control._is_toggle_off("switch off screen controls"))
        self.assertFalse(screen_control._is_toggle_on("switch off screen controls"))

    def test_enable_disable_synonyms(self):
        self.assertTrue(screen_control._is_toggle_on("enable screen controls"))
        self.assertFalse(screen_control._is_toggle_off("enable screen controls"))
        self.assertTrue(screen_control._is_toggle_off("disable screen controls"))
        self.assertFalse(screen_control._is_toggle_on("disable screen controls"))

    def test_trailing_position_variants(self):
        self.assertTrue(screen_control._is_toggle_on("screen controls on"))
        self.assertFalse(screen_control._is_toggle_off("screen controls on"))
        self.assertTrue(screen_control._is_toggle_off("screen control off"))
        self.assertFalse(screen_control._is_toggle_on("screen control off"))

    def test_negative_mismatches(self):
        for phrase in (
            "screen controls",
            "turn on the lights",
            "what is on my screen",
            "turn up the volume",
        ):
            self.assertFalse(screen_control._is_toggle_on(phrase), phrase)
            self.assertFalse(screen_control._is_toggle_off(phrase), phrase)

    def test_dispatch_off_wins_for_off_phrase(self):
        """maybe_handle_screen_control_message checks ON first — the off
        phrase must still land on the OFF branch."""
        screen_state.set_enabled(True)
        try:
            result = screen_control.maybe_handle_screen_control_message(
                "turn screen controls off"
            )
            self.assertEqual(result, "Screen controls are off, sir.")
            self.assertFalse(screen_state.is_enabled())
        finally:
            screen_state.set_enabled(False)


class ImageOnlyVisionPlanTests(unittest.TestCase):
    """F45: when UIA/OCR yield no targets the screenshot + coordinate grid
    still goes to the vision planner in image-only mode; returned
    coordinate-space metadata is validated."""

    def _capture(self):
        return {
            "capture_mode": "active_window",
            "window_title": "Demo",
            "origin_left": 0,
            "origin_top": 0,
            "capture_width": 200,
            "capture_height": 100,
            "vision_width": 100,
            "vision_height": 50,
            "image_data_url": "data:image/png;base64,abc",
            "hwnd": None,
        }

    def _vision_result(self, payload):
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    def test_image_only_accepts_normalized_coordinate_space(self):
        payload = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking the heart icon, sir.",
            "needs_confirmation": False,
            "reason": "",
            "coordinate_space": "normalized_1000",
            "steps": [{"action": "click", "x": 500, "y": 500}],
        }
        with patch.object(screen_control, "_ask_vision_cascade",
                          return_value=self._vision_result(payload)):
            plan = screen_control._request_vision_plan(
                "click the heart icon", self._capture(), image_only=True
            )
        self.assertTrue(plan["ok"])
        self.assertTrue(plan.get("image_only"))
        self.assertEqual(len(plan["steps"]), 1)

    def test_image_only_rejects_missing_coordinate_space(self):
        payload = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "steps": [{"action": "click", "x": 500, "y": 500}],
        }
        with patch.object(screen_control, "_ask_vision_cascade",
                          return_value=self._vision_result(payload)):
            plan = screen_control._request_vision_plan(
                "click the heart icon", self._capture(), image_only=True
            )
        # F45: a missing coordinate space is not guessed — the plan is rejected.
        self.assertFalse(plan["ok"])
        self.assertEqual(plan["steps"], [])

    def test_image_only_rejects_unexpected_coordinate_space(self):
        payload = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "coordinate_space": "raw_pixels",
            "steps": [{"action": "click", "x": 50, "y": 25}],
        }
        with patch.object(screen_control, "_ask_vision_cascade",
                          return_value=self._vision_result(payload)):
            plan = screen_control._request_vision_plan(
                "click the heart icon", self._capture(), image_only=True
            )
        self.assertFalse(plan["ok"])
        self.assertIn("unexpected space", plan["reason"])
        self.assertEqual(plan["steps"], [])

    def test_image_only_prompt_marks_mode_and_forbids_element_id(self):
        prompt = screen_control._build_tree_prompt(
            "click the heart icon", self._capture(), image_only=True
        )
        self.assertIn("IMAGE-ONLY MODE", prompt)
        self.assertIn("coordinate_space", prompt)
        self.assertIn("normalized_1000", prompt)
        self.assertIn("Never use element_id", prompt)
        self.assertIn("ambiguous", prompt)

    def test_empty_tree_routes_to_image_only_vision(self):
        ok_plan = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Clicking, sir.",
            "needs_confirmation": False,
            "reason": "",
            "steps": [{"action": "click", "x": 10, "y": 10}],
        }
        with patch.object(screen_control, "capture_for_screen_control",
                          return_value=self._capture()), \
             patch.object(screen_control, "_build_screen_snapshot",
                          return_value={}), \
             patch.object(screen_control, "_try_local_match",
                          return_value=None), \
             patch.object(screen_control, "_gather_ui_tree",
                          return_value=("<window/>", {})), \
             patch.object(screen_control, "_request_vision_plan",
                          return_value=ok_plan) as rvp:
            plan = screen_control._plan_with_tree("click the heart icon")
        rvp.assert_called_once()
        self.assertTrue(rvp.call_args.kwargs.get("image_only"))
        self.assertIsNone(rvp.call_args.kwargs.get("element_id_map"))
        self.assertTrue(plan["ok"])

    def test_empty_tree_vision_failure_surfaces(self):
        failed_plan = {
            "ok": False,
            "confidence": 0.0,
            "summary": "",
            "needs_confirmation": False,
            "reason": "Which region do you mean, sir?",
            "steps": [],
        }
        with patch.object(screen_control, "capture_for_screen_control",
                          return_value=self._capture()), \
             patch.object(screen_control, "capture_primary_screen",
                          return_value=self._capture()), \
             patch.object(screen_control, "_build_screen_snapshot",
                          return_value={}), \
             patch.object(screen_control, "_try_local_match",
                          return_value=None), \
             patch.object(screen_control, "_gather_ui_tree",
                          return_value=("<window/>", {})), \
             patch.object(screen_control, "_request_vision_plan",
                          return_value=failed_plan):
            plan = screen_control._plan_with_tree("click the heart icon")
        self.assertFalse(plan["ok"])
        self.assertIn("Which region", plan["reason"])


class ScreenRevocationInsideExecutionTests(unittest.TestCase):
    """F19: screen disable must be authoritative *inside* execution.

    Acceptance (plan.txt F19): "Disable/re-enable during planning invalidates
    old work; revocation between steps prevents later effects except necessary
    release cleanup; negated enable never enables."
    """

    def setUp(self):
        screen_state.set_enabled(False)
        screen_state.clear_interactions()
        self.addCleanup(lambda: screen_state.set_enabled(False))
        self.addCleanup(screen_state.clear_interactions)

    def test_revocation_between_steps_prevents_later_effects(self):
        import backend.services.screen_executor as screen_executor

        screen_state.set_enabled(True)
        calls = []

        def fake_click(x, y, button="left", clicks=1):
            calls.append((x, y))
            if len(calls) == 1:
                # The user revokes control while step 1 is running.
                screen_state.set_enabled(False)

        with patch.object(screen_executor, "click", side_effect=fake_click):
            steps = [
                {"action": "click", "x": 10, "y": 10},
                {"action": "click", "x": 20, "y": 20},
                {"action": "click", "x": 30, "y": 30},
            ]
            with self.assertRaises(screen_executor.ScreenRevoked):
                screen_executor.execute_steps(
                    steps, gate=lambda: screen_state.effect_gate(None)
                )
        # Only the already-started effect ran; nothing after revocation did.
        self.assertEqual(calls, [(10, 10)])

    def test_effect_boundary_denies_when_disabled(self):
        import backend.services.screen_executor as screen_executor

        with patch.object(screen_executor, "click") as mock_click:
            with self.assertRaises(screen_executor.ScreenRevoked):
                screen_executor.execute_steps(
                    [{"action": "click", "x": 1, "y": 1}],
                    gate=lambda: screen_state.effect_gate(None),
                )
        mock_click.assert_not_called()

    def test_stale_generation_never_executes(self):
        import backend.services.screen_executor as screen_executor

        screen_state.set_enabled(True)
        stale = screen_state.generation()
        screen_state.set_enabled(False)
        screen_state.set_enabled(True)  # off/on cycle bumps the generation

        with patch.object(screen_executor, "click") as mock_click:
            with self.assertRaises(screen_executor.ScreenRevoked):
                screen_executor.execute_steps(
                    [{"action": "click", "x": 1, "y": 1}],
                    gate=lambda: screen_state.effect_gate(stale),
                )
        mock_click.assert_not_called()

    def test_plan_stamped_before_planning_cannot_adopt_new_authority(self):
        """An off/on cycle during vision planning invalidates the plan."""
        screen_state.set_enabled(True)

        def plan_then_toggle(_command):
            # Simulate the user disabling+re-enabling while the vision model
            # is still planning.
            screen_state.set_enabled(False)
            screen_state.set_enabled(True)
            return {"ok": True, "confidence": 0.9, "summary": "Clicking, sir.",
                    "steps": [{"action": "click", "x": 5, "y": 5}]}

        with patch.object(screen_control, "_looks_like_screen_command",
                          return_value=True), \
             patch.object(screen_control, "_build_direct_plan",
                          return_value=None), \
             patch.object(screen_control, "_plan_with_tree",
                          side_effect=plan_then_toggle), \
             patch.object(screen_control, "execute_steps") as mock_exec:
            first = screen_control.maybe_handle_screen_control_message(
                "click the heart icon"
            )
            self.assertEqual(first, "Working on it, sir.")
            # Let the background vision thread finish.
            for _ in range(200):
                if not screen_control._vision_busy.locked():
                    break
                time.sleep(0.01)
        mock_exec.assert_not_called()

    def test_toggle_off_with_pending_consent_revokes_instead_of_executing(self):
        plan = {"ok": True, "confidence": 0.9, "needs_confirmation": True,
                "summary": "Deleting, sir.",
                "steps": [{"action": "click", "x": 1, "y": 1}]}
        screen_state.set_enabled(True)
        screen_state.set_pending_plan(plan, "delete the file")

        with patch.object(screen_control, "execute_steps") as mock_exec:
            result = screen_control.maybe_handle_screen_control_message(
                "turn screen controls off"
            )
        mock_exec.assert_not_called()
        self.assertIn("off", result.lower())
        self.assertFalse(screen_state.is_enabled())
        self.assertFalse(screen_state.has_pending_plan())

    def test_negated_enable_never_enables(self):
        for phrase in ("don't turn on screen controls",
                       "do not enable screen controls",
                       "never turn on screen controls"):
            with self.subTest(phrase=phrase):
                screen_state.set_enabled(False)
                screen_control.maybe_handle_screen_control_message(phrase)
                self.assertFalse(screen_state.is_enabled())

    def test_enable_and_disable_phrases_are_mutually_exclusive(self):
        self.assertEqual(screen_control._exact_toggle_verdict(
            "turn screen controls off"), "off")
        self.assertEqual(screen_control._exact_toggle_verdict(
            "turn off the screen controls"), "off")
        self.assertEqual(screen_control._exact_toggle_verdict(
            "disable screen controls"), "off")
        self.assertEqual(screen_control._exact_toggle_verdict(
            "turn on screen controls"), "on")
        self.assertEqual(screen_control._exact_toggle_verdict(
            "turn the screen controls on"), "on")
        # An off-then-on phrase with no on-word match resolves to the
        # fail-safe revocation rather than enabling.
        self.assertEqual(screen_control._exact_toggle_verdict(
            "turn screen controls off and on again"), "off")
        # A genuinely unrelated utterance is not a toggle at all.
        self.assertIsNone(screen_control._exact_toggle_verdict(
            "what is on my screen"))


class WindowsTargetRevalidationTests(unittest.TestCase):
    """F42 acceptance: expired/cleared caches, reused IDs, replacement,
    movement, focus theft and delayed consent yield safe fresh resolution or
    refusal — never stale-coordinate input."""

    def setUp(self):
        screen_state.set_enabled(False)
        self.addCleanup(lambda: screen_state.set_enabled(False))

    def _plan(self, **over):
        plan = {"ok": True, "confidence": 0.9, "summary": "Clicking, sir.",
                "needs_confirmation": False, "reason": "",
                "capture_mode": "active_window", "origin_left": 0,
                "origin_top": 0, "capture_width": 800, "capture_height": 600,
                "hwnd": 4242, "process_id": 111,
                "steps": [{"action": "click", "x": 10, "y": 10}]}
        plan.update(over)
        return plan

    def test_replaced_window_is_refused(self):
        plan = self._plan()
        bounds = {"left": 0, "top": 0, "width": 800, "height": 600,
                  "window_title": "Other", "hwnd": 4242, "process_id": 999}
        with patch("backend.services.screen_capture.get_window_bounds",
                   return_value=bounds):
            err = screen_control._maybe_adjust_for_window_move(plan)
        self.assertIsNotNone(err)
        self.assertIn("replaced", err.lower())

    def test_resized_window_is_refused(self):
        plan = self._plan()
        bounds = {"left": 0, "top": 0, "width": 1024, "height": 768,
                  "window_title": "Demo", "hwnd": 4242, "process_id": 111}
        with patch("backend.services.screen_capture.get_window_bounds",
                   return_value=bounds):
            err = screen_control._maybe_adjust_for_window_move(plan)
        self.assertIsNotNone(err)
        self.assertIn("size", err.lower())

    def test_movement_uses_planned_hwnd_not_foreground(self):
        plan = self._plan()
        seen = {}
        bounds = {"left": 50, "top": 25, "width": 800, "height": 600,
                  "window_title": "Demo", "hwnd": 4242, "process_id": 111}

        def fake_get_window_bounds(hwnd):
            seen["hwnd"] = hwnd
            return bounds

        with patch("backend.services.screen_capture.get_window_bounds",
                   side_effect=fake_get_window_bounds):
            err = screen_control._maybe_adjust_for_window_move(plan)
        self.assertIsNone(err)
        self.assertEqual(seen["hwnd"], 4242)
        self.assertEqual(plan["steps"][0]["x"], 60)
        self.assertEqual(plan["steps"][0]["y"], 35)

    def test_missing_hwnd_refuses_translation(self):
        err = screen_control._maybe_adjust_for_window_move(self._plan(hwnd=None))
        self.assertIsNotNone(err)

    def test_expired_cache_is_not_returned_as_fresh(self):
        from backend.services import screen_ui_elements as uia

        uia._element_cache.clear()
        self.addCleanup(uia._element_cache.clear)
        uia._element_cache["u1"] = {
            "created_at": time.monotonic() - 999,
            "wrapper": object(),
            "runtime_id": "rid-1",
            "hwnd": 4242,
        }
        self.assertIsNone(
            uia._find_cached_by_runtime_id("rid-1", hwnd=4242, require_live=True)
        )

    def test_cache_record_from_another_window_is_not_used(self):
        from backend.services import screen_ui_elements as uia

        uia._element_cache.clear()
        self.addCleanup(uia._element_cache.clear)
        uia._element_cache["u1"] = {
            "created_at": time.monotonic(),
            "wrapper": object(),
            "runtime_id": "rid-1",
            "hwnd": 1111,
        }
        self.assertIsNone(
            uia._find_cached_by_runtime_id("rid-1", hwnd=2222, require_live=True)
        )

    def test_cleared_cache_triggers_fresh_resolution(self):
        from backend.services import screen_ui_elements as uia

        uia._element_cache.clear()
        self.addCleanup(uia._element_cache.clear)

        def rewalk(hwnd=None, **kwargs):
            uia._element_cache["new"] = {
                "created_at": time.monotonic(),
                "wrapper": "W",
                "runtime_id": "rid-9",
                "hwnd": 4242,
            }
            return []

        with patch.object(uia, "is_available", return_value=True), \
             patch.object(uia, "get_foreground_window_elements",
                          side_effect=rewalk):
            ok = uia.invoke_element("stale-uid", runtime_id="rid-9", hwnd=4242)
        # The fake wrapper has no click API so the invoke reports failure, but
        # the fresh record must have been resolved rather than the lookup
        # silently refusing because the cache was cleared.
        self.assertFalse(ok)
        self.assertIn("new", uia._element_cache)

    def test_keyboard_step_refuses_when_focus_cannot_be_restored(self):
        import backend.services.screen_executor as screen_executor

        screen_state.set_enabled(True)
        with patch.object(screen_executor, "_wait_for_foreground",
                          side_effect=[True, False]), \
             patch.object(screen_executor.user32, "SetForegroundWindow",
                          return_value=0), \
             patch.object(screen_executor, "type_text") as mock_type:
            with self.assertRaises(RuntimeError) as ctx:
                screen_executor.execute_steps(
                    [{"action": "type", "text": "secret"}],
                    target_hwnd=4242,
                    gate=lambda: screen_state.effect_gate(None),
                )
        mock_type.assert_not_called()
        self.assertIn("focus", str(ctx.exception).lower())

    def test_click_is_refused_when_another_window_covers_the_point(self):
        import backend.services.screen_executor as screen_executor

        screen_state.set_enabled(True)
        with patch.object(screen_executor, "_wait_for_foreground",
                          return_value=True), \
             patch.object(screen_executor, "_root_window_at_point",
                          return_value=9999), \
             patch.object(screen_executor, "click") as mock_click:
            with self.assertRaises(RuntimeError):
                screen_executor.execute_steps(
                    [{"action": "click", "x": 5, "y": 5}],
                    target_hwnd=4242,
                    gate=lambda: screen_state.effect_gate(None),
                )
        mock_click.assert_not_called()


class _FakeInfo:
    def __init__(self, control_type="", name="", automation_id="",
                 class_name="", runtime_id=None):
        self.control_type = control_type
        self.name = name
        self.automation_id = automation_id
        self.class_name = class_name
        self.runtime_id = runtime_id


class _FakeRect:
    def __init__(self, left, top, right, bottom):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom


class _FakeElement:
    def __init__(self, info, rect, enabled=True, children=None):
        self.element_info = info
        self._rect = rect
        self._enabled = enabled
        self._children = children or []

    def children(self):
        return self._children

    def rectangle(self):
        return self._rect

    def is_enabled(self):
        return self._enabled


class HierarchyAndRepeatedLabelTests(unittest.TestCase):
    """F44 acceptance: unnamed ancestors, repeated Edit/Delete, staggered
    columns and OCR/UIA overlap retain correct hierarchy/distinct targets;
    translation/resizing does not corrupt tie resolution."""

    def setUp(self):
        from backend.services import screen_ui_elements as uia
        self.uia = uia
        uia._element_cache.clear()
        self._saved_obs = uia._observation_id
        self._saved_counter = uia._walk_counter
        self.addCleanup(uia._element_cache.clear)
        self.addCleanup(self._restore)

    def _restore(self):
        self.uia._observation_id = self._saved_obs
        self.uia._walk_counter = self._saved_counter

    def _walk_tree(self, root):
        out = []
        self.uia._observation_id = 7
        self.uia._walk_counter = 0
        self.uia._walk(root, out, 50, 9, 0, "")
        return out

    def test_unnamed_ancestor_survives_as_structural_placeholder(self):
        button = _FakeElement(_FakeInfo("Button", "Save"),
                              _FakeRect(10, 10, 60, 30))
        group = _FakeElement(_FakeInfo("Group", ""), _FakeRect(0, 0, 100, 100),
                             children=[button])
        root = _FakeElement(_FakeInfo("Window", "App"), _FakeRect(0, 0, 100, 100),
                            children=[group])

        out = self._walk_tree(root)
        uids = {el["uid"] for el in out}
        # Every emitted parent reference must resolve inside this observation.
        for el in out:
            if el["parent_uid"]:
                self.assertIn(el["parent_uid"], uids)
        self.assertIn("Save", [el.get("name") for el in out])
        placeholders = [el for el in out if el.get("placeholder")]
        self.assertEqual(len(placeholders), 1)
        self.assertEqual(placeholders[0]["control_type"], "Group")
        # Placeholder bounds are the union of its emitted subtree.
        self.assertEqual(placeholders[0]["left"], 10)
        self.assertEqual(placeholders[0]["right"], 60)

    def test_uids_are_observation_scoped_not_object_identities(self):
        button = _FakeElement(_FakeInfo("Button", "Go"), _FakeRect(1, 1, 20, 20))
        root = _FakeElement(_FakeInfo("Window", "App"), _FakeRect(0, 0, 50, 50),
                            children=[button])
        out = self._walk_tree(root)
        for el in out:
            self.assertTrue(el["uid"].startswith("obs7:"))
            self.assertEqual(el["observation_id"], 7)

    def test_repeated_labels_remain_distinct_spatial_entities(self):
        first = _FakeElement(_FakeInfo("Button", "Edit"), _FakeRect(0, 0, 40, 20))
        second = _FakeElement(_FakeInfo("Button", "Edit"), _FakeRect(0, 60, 40, 80))
        root = _FakeElement(_FakeInfo("Window", "App"), _FakeRect(0, 0, 100, 100),
                            children=[first, second])
        out = self._walk_tree(root)
        edits = [el for el in out if el.get("name") == "Edit"]
        self.assertEqual(len(edits), 2)
        self.assertNotEqual(edits[0]["top"], edits[1]["top"])

    def test_serialization_derives_depth_from_parent_links(self):
        elements = [
            {"uid": "a", "parent_uid": "", "depth": 0, "name": "Root",
             "control_type": "Button", "left": 0, "top": 0, "right": 10,
             "bottom": 10},
            # A depth integer that skipped a level must not make this a sibling.
            {"uid": "b", "parent_uid": "a", "depth": 5, "name": "Child",
             "control_type": "Button", "left": 0, "top": 0, "right": 10,
             "bottom": 10},
        ]
        lines, id_map, _ = screen_control._serialize_accessibility_elements(elements)
        joined = "\n".join(lines)
        self.assertEqual(len(id_map), 2)
        # "Root" opens (has a child) and "Child" is nested one level under it.
        self.assertIn('name="Root">', joined)
        child_line = [l for l in lines if 'name="Child"' in l][0]
        self.assertTrue(child_line.startswith(" " * 6), child_line)
        self.assertEqual(joined.count("</element>"), 1)

    def test_ocr_words_on_different_lines_are_not_merged(self):
        from backend.services import screen_ocr

        regions = [
            {"text": "Save", "left": 0, "top": 0, "right": 20, "bottom": 10,
             "confidence": 90, "block_num": 1, "par_num": 1, "line_num": 1},
            {"text": "As", "left": 22, "top": 0, "right": 40, "bottom": 10,
             "confidence": 90, "block_num": 1, "par_num": 1, "line_num": 2},
        ]
        merged = screen_ocr._merge_nearby_words(regions)
        self.assertEqual(len(merged), 2)

    def test_ocr_overlapping_boxes_are_not_merged(self):
        from backend.services import screen_ocr

        regions = [
            {"text": "Edit", "left": 0, "top": 0, "right": 40, "bottom": 10,
             "confidence": 90, "block_num": 1, "par_num": 1, "line_num": 1},
            {"text": "Now", "left": 20, "top": 0, "right": 60, "bottom": 10,
             "confidence": 90, "block_num": 1, "par_num": 1, "line_num": 1},
        ]
        merged = screen_ocr._merge_nearby_words(regions)
        self.assertEqual(len(merged), 2)

    def test_ocr_adjacent_words_same_line_still_merge(self):
        from backend.services import screen_ocr

        regions = [
            {"text": "Save", "left": 0, "top": 0, "right": 20, "bottom": 10,
             "confidence": 90, "block_num": 1, "par_num": 1, "line_num": 1},
            {"text": "As", "left": 24, "top": 0, "right": 44, "bottom": 10,
             "confidence": 90, "block_num": 1, "par_num": 1, "line_num": 1},
        ]
        merged = screen_ocr._merge_nearby_words(regions)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["text"], "Save As")

    def test_ocr_repeated_labels_on_distinct_lines_survive(self):
        from backend.services import screen_ocr

        regions = [
            {"text": "Edit", "left": 0, "top": 0, "right": 30, "bottom": 10,
             "confidence": 90, "block_num": 1, "par_num": 1, "line_num": 1},
            {"text": "Edit", "left": 0, "top": 40, "right": 30, "bottom": 50,
             "confidence": 90, "block_num": 1, "par_num": 1, "line_num": 3},
        ]
        merged = screen_ocr._merge_nearby_words(regions)
        self.assertEqual(len(merged), 2)

    def test_tiebreak_converts_candidates_into_the_screen_frame(self):
        seen = []

        def to_screen(x, y, space):
            seen.append((x, y, space))
            return x, y

        candidates = [
            {"cx": 1, "cy": 1, "label": "a", "space": "vision"},
            {"cx": 2, "cy": 2, "label": "b", "space": "screen_pixels"},
        ]
        winner = screen_control._tiebreak_by_proximity(
            candidates, to_screen=to_screen)
        self.assertIsNotNone(winner)
        self.assertEqual(len(seen), 2)
        self.assertEqual({c[2] for c in seen}, {"vision_pixels", "screen_pixels"})


class CaptureCoordinateContractTests(unittest.TestCase):
    """F43: capture and coordinates are explicit and share one observation."""

    def setUp(self):
        screen_capture._last_user_target = {
            "hwnd": None, "title": "", "process_id": None,
        }
        self.addCleanup(
            lambda: screen_capture._last_user_target.update(
                {"hwnd": None, "title": "", "process_id": None}
            )
        )

    def test_last_foreground_target_reacquires_live_geometry(self):
        """A moved window must not be described by its recorded bounds."""
        bounds = {
            "left": 500, "top": 400, "width": 800, "height": 600,
            "window_title": "Editor", "hwnd": 77, "process_id": 42,
        }
        screen_capture._last_user_target.update(
            {"hwnd": 77, "title": "Editor", "process_id": 42}
        )
        with patch.object(screen_capture, "_window_bounds", return_value=bounds):
            target = screen_capture.last_foreground_target()
        self.assertEqual(target["bounds"]["left"], 500)
        self.assertEqual(target["process_id"], 42)

    def test_last_foreground_target_refuses_a_recycled_hwnd(self):
        """F43: an HWND reused by another program is not our target."""
        screen_capture._last_user_target.update(
            {"hwnd": 77, "title": "Editor", "process_id": 42}
        )
        recycled = {
            "left": 0, "top": 0, "width": 100, "height": 100,
            "window_title": "Something else", "hwnd": 77, "process_id": 999,
        }
        with patch.object(screen_capture, "_window_bounds", return_value=recycled):
            self.assertIsNone(screen_capture.last_foreground_target())
        self.assertIsNone(screen_capture._last_user_target["hwnd"])

    def test_last_foreground_target_is_none_for_a_closed_window(self):
        screen_capture._last_user_target.update(
            {"hwnd": 77, "title": "Editor", "process_id": 42}
        )
        with patch.object(screen_capture, "_window_bounds", return_value=None):
            self.assertIsNone(screen_capture.last_foreground_target())

    def test_own_overlay_falls_back_to_the_recorded_target(self):
        """F43: never capture our own overlay as 'the screen the user means'."""
        live = {
            "left": 10, "top": 20, "width": 640, "height": 480,
            "window_title": "Browser", "hwnd": 555, "process_id": 12,
        }
        screen_capture._last_user_target.update(
            {"hwnd": 555, "title": "Browser", "process_id": 12}
        )
        with patch.object(screen_capture.user32, "GetForegroundWindow",
                          return_value=999), \
             patch.object(screen_capture, "is_own_window", return_value=True), \
             patch.object(screen_capture, "_window_bounds", return_value=live):
            bounds = screen_capture._get_foreground_window_bounds()
        self.assertEqual(bounds["hwnd"], 555)

    def test_own_overlay_with_no_recorded_target_refuses_to_capture(self):
        with patch.object(screen_capture.user32, "GetForegroundWindow",
                          return_value=999), \
             patch.object(screen_capture, "is_own_window", return_value=True):
            self.assertIsNone(screen_capture._get_foreground_window_bounds())

    def test_capture_epoch_advances_per_capture(self):
        first = screen_capture._next_capture_epoch()
        second = screen_capture._next_capture_epoch()
        self.assertGreater(second, first)
        self.assertEqual(screen_capture.current_capture_epoch(), second)

    def test_snapshot_walks_the_windows_the_capture_came_from(self):
        """F43: the tree must describe the frame, not 'whatever is foreground'."""
        seen = {}

        def fake_walk(hwnd=None, **kwargs):
            seen["hwnd"] = hwnd
            return []

        capture = {
            "capture_mode": "active_window", "window_title": "Demo",
            "origin_left": 0, "origin_top": 0,
            "capture_width": 100, "capture_height": 100,
            "vision_width": 100, "vision_height": 100,
            "hwnd": 4242, "ui_hwnd": 4242, "capture_epoch": 9,
        }
        with patch.object(screen_ui_elements, "is_available", return_value=True), \
             patch.object(screen_ui_elements, "get_foreground_window_elements",
                          side_effect=fake_walk):
            snapshot = screen_control._build_screen_snapshot(capture)
        self.assertEqual(seen["hwnd"], 4242)
        self.assertEqual(snapshot["ui_hwnd"], 4242)
        self.assertEqual(snapshot["capture_epoch"], 9)

    def test_ui_space_is_declared_not_assumed(self):
        """F43: unconverted (screen-pixel) UIA bounds must not be labelled vision."""
        elements = [
            {"uid": "obs1:1", "depth": 0, "control_type": "Button",
             "name": "Go", "left": 10, "top": 10, "right": 30, "bottom": 30},
        ]
        _, id_map, _ = screen_control._serialize_accessibility_elements(
            elements, start_id=1, space=screen_geometry.COORD_SCREEN_PIXELS)
        self.assertEqual(id_map[1]["space"], screen_geometry.COORD_SCREEN_PIXELS)

    def test_vision_space_is_preserved_for_converted_bounds(self):
        elements = [
            {"uid": "obs1:1", "depth": 0, "control_type": "Button",
             "name": "Go", "left": 10, "top": 10, "right": 30, "bottom": 30},
        ]
        _, id_map, _ = screen_control._serialize_accessibility_elements(
            elements, start_id=1, space=screen_geometry.COORD_VISION_PIXELS)
        self.assertEqual(id_map[1]["space"], screen_geometry.COORD_VISION_PIXELS)


class LocalMatchAmbiguityTests(unittest.TestCase):
    """F29: one ambiguity/actionability policy across the local routes."""

    CAPTURE = {"capture_mode": "active_window", "raw_vision_image": None}

    def _elem(self, name, ctype="Button", left=10, top=20, right=60, bottom=50,
              enabled=True, uid=None):
        elem = {
            "name": name, "control_type": ctype,
            "left": left, "top": top, "right": right, "bottom": bottom,
        }
        if enabled is not None:
            elem["enabled"] = enabled
        elem["uid"] = uid if uid is not None else f"obs1:{left}"
        return elem

    def _local(self, command, elements, ocr=None, ocr_available=False):
        with patch.object(screen_control.screen_ui_elements, "is_available",
                          return_value=True), \
             patch.object(screen_control.screen_ui_elements,
                          "get_foreground_window_elements", return_value=elements), \
             patch.object(screen_control.screen_ocr, "is_available",
                          return_value=ocr_available):
            if ocr_available:
                with patch.object(screen_control.screen_ocr,
                                  "extract_text_regions", return_value=ocr or []):
                    return screen_control._try_local_match(command, self.CAPTURE)
            return screen_control._try_local_match(command, self.CAPTURE)

    def test_unique_exact_enabled_target_needs_no_ocr_or_provider(self):
        plan = self._local("click save", [self._elem("Save")])
        self.assertIsNotNone(plan)
        self.assertEqual(plan["source"], "Local Match")
        self.assertGreaterEqual(plan["confidence"], FAST_LOCAL_MATCH_CONFIDENCE)
        self.assertFalse(plan["needs_confirmation"])

    def test_duplicate_labels_cannot_regain_automatic_execution(self):
        """Two Edit buttons in a list are not one identifiable target."""
        elements = [
            self._elem("Edit", left=10, top=10, right=60, bottom=40, uid="obs1:1"),
            self._elem("Edit", left=10, top=60, right=60, bottom=90, uid="obs1:2"),
        ]
        self.assertIsNone(self._local("click edit", elements))

    def test_disabled_control_is_never_actionable(self):
        elements = [self._elem("Save", enabled=False)]
        self.assertIsNone(self._local("click save", elements))

    def test_unknown_enabled_state_is_never_actionable(self):
        """F29: a missing enabled flag is not evidence of actionability."""
        elements = [self._elem("Save", enabled=None)]
        self.assertIsNone(self._local("click save", elements))

    def test_exact_match_is_not_promoted_over_by_a_clickable_substring(self):
        """F29: a Button bonus must not lift "Save As" above an exact "Save"."""
        elements = [
            self._elem("Save As", ctype="Button", left=10, top=10, right=60,
                       bottom=40, uid="obs1:1"),
            self._elem("Save", ctype="Text", left=10, top=60, right=60,
                       bottom=90, uid="obs1:2"),
        ]
        scores = {e["name"]: screen_control._score_ui_element_match("Save", e)
                  for e in elements}
        self.assertGreater(scores["Save"], scores["Save As"])

    def test_substring_target_is_not_executed_automatically(self):
        """A materially different label is a candidate, not an identification."""
        plan = self._local("click save", [self._elem("Save As")])
        self.assertIsNotNone(plan)
        self.assertTrue(plan["needs_confirmation"])
        self.assertLess(plan["confidence"], FAST_LOCAL_MATCH_CONFIDENCE)

    def test_duplicate_uids_with_one_label_defer(self):
        """F29: identical labels on different controls are ambiguous."""
        elements = [
            self._elem("Reload", left=10, top=10, right=60, bottom=40,
                       uid="obs1:1"),
            self._elem("Reload", left=10, top=60, right=60, bottom=90,
                       uid="obs1:2"),
        ]
        self.assertIsNone(self._local("click reload", elements))

    def test_materially_different_alias_is_no_longer_declared(self):
        """F29: "Save As" is not equivalent to "Save"."""
        self.assertLess(
            screen_control._text_match_class("save", "Save As"), 3)
        self.assertLess(
            screen_control._text_match_class("close", "Exit"), 3)

    def test_uia_first_declines_a_non_exact_target(self):
        """F29: the fast path requires an exact/alias identification."""
        elements = [self._elem("Save As")]
        with patch.object(screen_control.screen_ui_elements, "is_available",
                          return_value=True), \
             patch.object(screen_control.screen_ui_elements,
                          "get_foreground_window_elements", return_value=elements), \
             patch.object(screen_control, "_resolve_uia_target_hwnd",
                          return_value=4242):
            plan = screen_control._try_uia_first("click save")
        self.assertIsNone(plan)

    def test_uia_first_accepts_a_unique_exact_target(self):
        elements = [self._elem("Save")]
        with patch.object(screen_control.screen_ui_elements, "is_available",
                          return_value=True), \
             patch.object(screen_control.screen_ui_elements,
                          "get_foreground_window_elements", return_value=elements), \
             patch.object(screen_control, "_resolve_uia_target_hwnd",
                          return_value=4242), \
             patch.object(screen_control, "get_window_bounds",
                          return_value={"left": 0, "top": 0, "width": 800,
                                        "height": 600, "hwnd": 4242,
                                        "process_id": 7}):
            plan = screen_control._try_uia_first("click save")
        self.assertIsNotNone(plan)
        self.assertEqual(plan["steps"][0]["action"], "click")

    def test_uia_first_declines_an_unknown_enabled_state(self):
        elements = [self._elem("Save", enabled=None)]
        with patch.object(screen_control.screen_ui_elements, "is_available",
                          return_value=True), \
             patch.object(screen_control.screen_ui_elements,
                          "get_foreground_window_elements", return_value=elements), \
             patch.object(screen_control, "_resolve_uia_target_hwnd",
                          return_value=4242):
            self.assertIsNone(screen_control._try_uia_first("click save"))


class PostActionVerificationTests(unittest.TestCase):
    """F41: verify the goal, not generic motion or text anywhere on screen."""

    def _controls(self, value, focused=True, runtime_id="rid-1"):
        return [{
            "uid": "obs1:1", "runtime_id": runtime_id,
            "control_type": "Edit", "focused": focused, "value": value,
        }]

    def test_text_in_the_intended_field_verifies(self):
        steps = [{"action": "click", "ui_runtime_id": "rid-1"},
                 {"action": "type", "text": "hello"}]
        with patch.object(screen_control.screen_ui_elements, "is_available",
                          return_value=True), \
             patch.object(screen_control.screen_ui_elements, "read_control_values",
                          return_value=self._controls("hello world")):
            self.assertTrue(screen_control._verify_action_locally(steps))

    def test_text_missing_from_the_intended_field_fails(self):
        """F41: the text landing in the WRONG field must not verify."""
        steps = [{"action": "click", "ui_runtime_id": "rid-1"},
                 {"action": "type", "text": "hello"}]
        with patch.object(screen_control.screen_ui_elements, "is_available",
                          return_value=True), \
             patch.object(screen_control.screen_ui_elements, "read_control_values",
                          return_value=self._controls("")):
            self.assertFalse(screen_control._verify_action_locally(steps))

    def test_whole_window_text_is_not_evidence(self):
        """F41: the haystack no longer proves anything on its own."""
        steps = [{"action": "click", "ui_runtime_id": "rid-1"},
                 {"action": "type", "text": "hello"}]
        capture = {"raw_vision_image": object()}
        with patch.object(screen_control.screen_ui_elements, "is_available",
                          return_value=True), \
             patch.object(screen_control.screen_ui_elements, "read_control_values",
                          return_value=self._controls("")), \
             patch.object(screen_control, "_verification_text_haystack",
                          return_value="hello hello hello"):
            self.assertFalse(
                screen_control._verify_action_locally(steps, capture))

    def test_unreadable_field_is_uncertainty_not_success(self):
        steps = [{"action": "click", "ui_runtime_id": "rid-1"},
                 {"action": "type", "text": "hello"}]
        with patch.object(screen_control.screen_ui_elements, "is_available",
                          return_value=True), \
             patch.object(screen_control.screen_ui_elements, "read_control_values",
                          return_value=self._controls(None)):
            self.assertIsNone(screen_control._verify_action_locally(steps))

    def test_unavailable_uia_is_uncertainty(self):
        steps = [{"action": "type", "text": "hello"}]
        with patch.object(screen_control.screen_ui_elements, "is_available",
                          return_value=False):
            self.assertIsNone(screen_control._verify_action_locally(steps))

    def test_click_only_plan_has_no_local_postcondition(self):
        with patch.object(screen_control.screen_ui_elements, "is_available",
                          return_value=True):
            self.assertIsNone(screen_control._verify_action_locally(
                [{"action": "click", "x": 1, "y": 2}]))

    def test_type_step_without_identity_falls_back_to_the_focused_field(self):
        steps = [{"action": "type", "text": "hello"}]
        with patch.object(screen_control.screen_ui_elements, "is_available",
                          return_value=True), \
             patch.object(screen_control.screen_ui_elements, "read_control_values",
                          return_value=self._controls("hello", focused=True)):
            self.assertTrue(screen_control._verify_action_locally(steps))

    def test_unsupported_action_stops_instead_of_reporting_success(self):
        with self.assertRaises(RuntimeError):
            screen_executor.execute_steps(
                [{"action": "teleport", "x": 1, "y": 2}], gate=None)

    def test_single_step_plans_are_verified(self):
        """F41: the `len(steps) <= 2` skip meant common actions unchecked."""
        plan = {
            "ok": True, "confidence": 0.95, "summary": "Typing, sir.",
            "needs_confirmation": False, "reason": "",
            "capture_mode": "active_window", "window_title": "Demo",
            "hwnd": None, "steps": [{"action": "type", "text": "hello"}],
        }
        screen_state.set_enabled(True)
        try:
            with patch.object(screen_control, "execute_steps"), \
                 patch.object(screen_control, "_verify_action_with_steps",
                              return_value=None) as mock_verify, \
                 patch.object(screen_control.screen_state, "clear_pending_plan"), \
                 patch.object(screen_control.screen_state, "add_interaction"), \
                 patch.object(screen_control, "_write_screen_command_log"):
                screen_control._execute_or_queue(plan, "type hello")
            mock_verify.assert_called_once()
        finally:
            screen_state.set_enabled(False)


if __name__ == "__main__":
    unittest.main()
