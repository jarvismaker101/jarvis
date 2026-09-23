"""C2 (2026-09-23 audit) — one declared coordinate frame for the screen planner.

Prompt geometry is serialized in NORMALIZED 0..1000 (the same frame the
planner must answer in), and a step that cites an element must not contradict
it with coordinates from another frame.
"""
import unittest

from backend.services import screen_control

CAPTURE = {
    "vision_width": 1280, "vision_height": 800,
    "capture_width": 1280, "capture_height": 800,
    "origin_left": 0, "origin_top": 0,
    "capture_mode": "desktop",
}


class NormGeomTests(unittest.TestCase):
    def test_vision_pixels_normalize_to_1000_frame(self):
        box, center = screen_control._norm_geom(
            600, 500, 720, 600, 660, 550, CAPTURE, "vision")
        self.assertEqual(center, (516, 688))
        self.assertEqual(box, (469, 625, 562, 750))

    def test_missing_capture_metadata_falls_back_to_pixels(self):
        box, center = screen_control._norm_geom(
            600, 500, 720, 600, 660, 550, None, "vision")
        self.assertEqual(center, (660, 550))
        self.assertEqual(box, (600, 500, 720, 600))


class SerializerTests(unittest.TestCase):
    def test_ocr_nodes_serialize_normalized_but_map_keeps_pixels(self):
        regions = [{"text": "Hi", "left": 600, "top": 500,
                    "right": 720, "bottom": 600, "confidence": 90}]
        lines, id_map, _ = screen_control._serialize_ocr_nodes(
            regions, [], start_id=1, capture=CAPTURE)
        self.assertIn('center="516,688"', lines[0])
        self.assertIn('bounds="469,625,562,750"', lines[0])
        self.assertNotIn('center="660,550"', lines[0])
        # internal truth stays in pixels for the executor
        self.assertEqual(id_map[1]["x"], 660)
        self.assertEqual(id_map[1]["y"], 550)

    def test_accessibility_nodes_serialize_normalized(self):
        elements = [{
            "left": 600, "top": 500, "right": 720, "bottom": 600,
            "control_type": "Button", "name": "Go", "uid": "u1",
        }]
        lines, id_map, _ = screen_control._serialize_accessibility_elements(
            elements, start_id=1, space="vision", capture=CAPTURE)
        self.assertIn('center="516,688"', lines[0])
        self.assertEqual(id_map[1]["x"], 660)


class PromptTests(unittest.TestCase):
    def test_prompt_declares_one_frame(self):
        prompt = screen_control._build_tree_prompt(
            "click go",
            {"window_title": "T", "vision_width": 1280, "vision_height": 800,
             "capture_width": 1280, "capture_height": 800},
            ui_context="<window/>")
        self.assertIn("the SAME frame your x,y answers", prompt)
        self.assertNotIn("capture-local vision pixels", prompt)


class PointAgreementTests(unittest.TestCase):
    def _element(self):
        return screen_control._build_element_point(
            150, 80, space="vision",
            bounds={"left": 100, "top": 40, "right": 200, "bottom": 120})

    def test_near_point_agrees(self):
        self.assertTrue(screen_control._point_agrees_with_element(
            "vision", 150, 80, self._element(), CAPTURE))

    def test_far_point_contradicts(self):
        self.assertFalse(screen_control._point_agrees_with_element(
            "vision", 900, 700, self._element(), CAPTURE))

    def test_no_bounds_means_nothing_to_check(self):
        ref = screen_control._build_element_point(150, 80, space="vision")
        self.assertTrue(screen_control._point_agrees_with_element(
            "vision", 900, 700, ref, CAPTURE))


class ContradictionRejectionTests(unittest.TestCase):
    def test_contradicting_coords_are_rejected(self):
        element_id_map = {
            1: screen_control._build_element_point(
                150, 80, space="vision",
                bounds={"left": 100, "top": 40, "right": 200, "bottom": 120})}
        plan = {
            "ok": True, "confidence": 0.95, "needs_confirmation": False,
            "reason": "", "coordinate_space": "normalized_1000",
            "steps": [{"action": "click", "element_id": 1,
                       "x": 900, "y": 700}],
        }
        out = screen_control._normalize_vision_plan(
            plan, dict(CAPTURE, vision_width=1000, vision_height=1000,
                       capture_width=1000, capture_height=1000),
            element_id_map)
        self.assertFalse(out["ok"])
        self.assertEqual(out["steps"], [])
        self.assertIn("contradict", out["reason"])

    def test_agreeing_coords_pass(self):
        element_id_map = {
            1: screen_control._build_element_point(
                150, 80, space="vision",
                bounds={"left": 100, "top": 40, "right": 200, "bottom": 120})}
        plan = {
            "ok": True, "confidence": 0.95, "needs_confirmation": False,
            "reason": "", "coordinate_space": "normalized_1000",
            "steps": [{"action": "click", "element_id": 1,
                       "x": 150, "y": 80}],
        }
        out = screen_control._normalize_vision_plan(
            plan, dict(CAPTURE, vision_width=1000, vision_height=1000,
                       capture_width=1000, capture_height=1000),
            element_id_map)
        self.assertTrue(out["ok"])


if __name__ == "__main__":
    unittest.main()
