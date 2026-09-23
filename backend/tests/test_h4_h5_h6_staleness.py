"""H4 + H5 + H6 (2026-09-23 audit) — staleness, revalidation, coordinate bounds.

H4: plans carry their capture's epoch and the gate refuses when the screen
was re-observed since (the confirmed preview described a different screen).
H5: window identity is fail-closed (IsWindow + process match), hit-tests
refuse when no window is under the point, and move/hover + window-state
actions are revalidated like clicks.
H6: out-of-frame points are rejected, never clamped into an edge click.
"""
import unittest
from unittest.mock import MagicMock, patch

from backend.services import screen_capture, screen_control, screen_executor

CAPTURE = {
    "vision_width": 1000, "vision_height": 1000,
    "capture_width": 1000, "capture_height": 1000,
    "origin_left": 0, "origin_top": 0, "capture_mode": "desktop",
    "capture_epoch": 7,
}


class EpochGateTests(unittest.TestCase):
    def test_plan_carries_the_capture_epoch(self):
        plan = screen_control._normalize_vision_plan(
            {"ok": True, "confidence": 0.95, "needs_confirmation": False,
             "reason": "", "coordinate_space": "normalized_1000",
             "steps": [{"action": "click", "x": 10, "y": 10}]}, dict(CAPTURE))
        self.assertEqual(plan.get("capture_epoch"), 7)

    def test_gate_refuses_when_epoch_moved(self):
        plan = {"screen_generation": 1, "capture_epoch": 7}
        with patch.object(screen_capture, "current_capture_epoch",
                          return_value=8):
            gate = screen_control._plan_gate(plan)
            with self.assertRaises(RuntimeError) as ctx:
                gate()
        self.assertIn("screen changed", str(ctx.exception))

    def test_gate_passes_when_epoch_unchanged(self):
        plan = {"screen_generation": 1, "capture_epoch": 7}
        marker = object()
        with patch.object(screen_capture, "current_capture_epoch",
                          return_value=7), \
             patch("backend.services.screen_state.effect_gate",
                   return_value=marker):
            self.assertIs(screen_control._plan_gate(plan)(), marker)

    def test_mint_stamps_the_recorded_target(self):
        region = {"hwnd": 55}
        with patch.object(screen_capture, "is_own_window",
                          return_value=False), \
             patch.object(screen_capture, "_last_user_target", {}):
            ep = screen_capture._mint_capture_epoch(region)
            self.assertEqual(screen_capture._last_user_target
                             .get("capture_epoch"), ep)


class WindowIdentityTests(unittest.TestCase):
    @staticmethod
    def _fill_pid(value):
        def fake_pid(_hwnd, out):
            getattr(out, "_obj", out).value = value
        return fake_pid

    def test_dead_window_refuses(self):
        with patch.object(screen_executor.user32, "IsWindow", return_value=0):
            with self.assertRaises(RuntimeError):
                screen_executor._verify_window_identity(123)

    def test_recycled_window_refuses_on_process_mismatch(self):
        with patch.object(screen_executor.user32, "IsWindow", return_value=1), \
             patch.object(screen_executor.user32, "GetWindowThreadProcessId",
                          side_effect=self._fill_pid(4321)):
            with self.assertRaises(RuntimeError) as ctx:
                screen_executor._verify_window_identity(123, expected_pid=999)
        self.assertIn("different program", str(ctx.exception))

    def test_matching_identity_passes(self):
        with patch.object(screen_executor.user32, "IsWindow", return_value=1), \
             patch.object(screen_executor.user32, "GetWindowThreadProcessId",
                          side_effect=self._fill_pid(999)):
            screen_executor._verify_window_identity(123, expected_pid=999)


class HitTestTests(unittest.TestCase):
    def test_no_window_under_point_refuses(self):
        with patch.object(screen_executor, "_root_window_at_point",
                          return_value=0):
            with self.assertRaises(RuntimeError) as ctx:
                screen_executor._verify_hit_test({"x": 1, "y": 2}, 123)
        self.assertIn("no window under the spot", str(ctx.exception))


class RevalidationCoverageTests(unittest.TestCase):
    def test_move_hover_go_through_hit_test(self):
        with patch.object(screen_executor, "_verify_hit_test") as hit, \
             patch.object(screen_executor, "move_mouse"), \
             patch("time.sleep"):
            screen_executor.execute_steps([{"action": "move", "x": 1, "y": 2}])
        hit.assert_called_once()

    def test_window_state_actions_bring_the_target_forward(self):
        with patch.object(screen_executor, "_verify_window_identity"), \
             patch.object(screen_executor, "_wait_for_foreground",
                          return_value=True), \
             patch.object(screen_executor.user32, "IsIconic", return_value=0), \
             patch.object(screen_executor.user32, "SetForegroundWindow"), \
             patch.object(screen_executor, "_ensure_target_focus") as focus, \
             patch.object(screen_executor, "minimize_foreground_window"):
            screen_executor.execute_steps([{"action": "minimize_window"}],
                                          target_hwnd=55)
        focus.assert_called_once()


class RejectNotClampTests(unittest.TestCase):
    def test_out_of_frame_point_is_rejected(self):
        self.assertIsNone(screen_executor and
                          screen_control._step_point_to_screen(
                              "normalized_1000", 1500, 500, dict(CAPTURE)))
        self.assertIsNone(screen_control._step_point_to_screen(
            "normalized_1000", 500, -5, dict(CAPTURE)))

    def test_in_range_point_still_maps(self):
        self.assertEqual(
            screen_control._step_point_to_screen(
                "normalized_1000", 500, 500, dict(CAPTURE)),
            (500, 500))


if __name__ == "__main__":
    unittest.main()
