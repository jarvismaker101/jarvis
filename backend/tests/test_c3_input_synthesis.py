"""C3 (2026-09-23 audit) — model output must never become raw OS input.

Key tokens are whitelisted against the KEY_TOKEN_MAP vocabulary, clicks are
bounded 1..10, buttons are left|right only, and the executor refuses any key
string outside its safe grammar.
"""
import unittest
from unittest.mock import MagicMock, patch

from backend.services import screen_control, screen_executor

CAPTURE = {
    "vision_width": 1000, "vision_height": 1000,
    "capture_width": 1000, "capture_height": 1000,
    "origin_left": 0, "origin_top": 0,
    "capture_mode": "desktop",
}


def _plan(step):
    return {
        "ok": True, "confidence": 0.95, "needs_confirmation": False,
        "reason": "", "coordinate_space": "normalized_1000", "steps": [step],
    }


class WhitelistTests(unittest.TestCase):
    def test_vocabulary_tokens_map_to_canonical(self):
        self.assertEqual(screen_control._whitelist_key_tokens(
            ["control", "a"]), ["ctrl", "a"])
        self.assertEqual(screen_control._whitelist_key_tokens(
            ["escape"]), ["esc"])
        self.assertEqual(screen_control._whitelist_key_tokens(
            ["win", "r"]), ["windows", "r"])
        self.assertEqual(screen_control._whitelist_key_tokens(["f5"]), ["f5"])

    def test_combined_or_junk_strings_are_rejected(self):
        self.assertIsNone(screen_control._whitelist_key_tokens(["alt+f4"]))
        self.assertIsNone(screen_control._whitelist_key_tokens(["a+b+c"]))
        self.assertIsNone(screen_control._whitelist_key_tokens(["!!"]))
        self.assertIsNone(screen_control._whitelist_key_tokens(["drop table"]))
        self.assertIsNone(screen_control._whitelist_key_tokens([""]))
        self.assertIsNone(screen_control._whitelist_key_tokens([]))


class PlanGateTests(unittest.TestCase):
    def test_hotkey_with_combined_string_is_rejected(self):
        out = screen_control._normalize_vision_plan(
            _plan({"action": "hotkey", "keys": ["alt+f4"]}), dict(CAPTURE))
        self.assertFalse(out["ok"])
        self.assertIn("don't allow", out["reason"])

    def test_press_maps_vocabulary(self):
        out = screen_control._normalize_vision_plan(
            _plan({"action": "press", "keys": ["escape"]}), dict(CAPTURE))
        self.assertTrue(out["ok"])
        self.assertEqual(out["steps"][0]["keys"], ["esc"])

    def test_clicks_are_clamped_and_button_whitelisted(self):
        out = screen_control._normalize_vision_plan(
            _plan({"action": "click", "x": 100, "y": 100, "clicks": 100000,
                   "button": "middle"}), dict(CAPTURE))
        self.assertTrue(out["ok"])
        self.assertEqual(out["steps"][0]["clicks"], 10)
        self.assertEqual(out["steps"][0]["button"], "left")

    def test_bad_clicks_type_degrades_to_one(self):
        out = screen_control._normalize_vision_plan(
            _plan({"action": "click", "x": 100, "y": 100,
                   "clicks": "abc"}), dict(CAPTURE))
        self.assertTrue(out["ok"])
        self.assertEqual(out["steps"][0]["clicks"], 1)

    def test_right_button_survives(self):
        out = screen_control._normalize_vision_plan(
            _plan({"action": "right_click", "x": 100, "y": 100,
                   "button": "right"}), dict(CAPTURE))
        self.assertTrue(out["ok"])
        self.assertEqual(out["steps"][0]["button"], "right")


class ExecutorGuardTests(unittest.TestCase):
    def test_normalize_key_name_rejects_grammar_violations(self):
        self.assertEqual(screen_executor._normalize_key_name("ctrl"), "ctrl")
        self.assertIsNone(screen_executor._normalize_key_name("a+b+c"))
        self.assertIsNone(screen_executor._normalize_key_name("{LEFT}"))
        self.assertIsNone(screen_executor._normalize_key_name(""))

    def test_press_keys_refuses_unsafe_token(self):
        with patch.object(screen_executor, "_get_keyboard",
                          return_value=MagicMock()):
            with self.assertRaises(ValueError):
                screen_executor.press_keys(["a+b+c"])

    def test_press_keys_joins_whitelisted_chord(self):
        kb = MagicMock()
        with patch.object(screen_executor, "_get_keyboard", return_value=kb):
            screen_executor.press_keys(["ctrl", "a"])
        kb.press_and_release.assert_called_once_with("ctrl+a")

    def test_click_count_is_capped_at_ten(self):
        with patch.object(screen_executor, "move_mouse"), \
             patch.object(screen_executor, "_ensure_dpi_awareness"), \
             patch.object(screen_executor, "user32") as u32, \
             patch("backend.services.screen_executor.time"):
            screen_executor.click(1, 1, clicks=100000)
        self.assertEqual(u32.SendInput.call_count, 20)  # 10 clicks x down+up

    def test_bad_clicks_type_clicks_once(self):
        with patch.object(screen_executor, "move_mouse"), \
             patch.object(screen_executor, "_ensure_dpi_awareness"), \
             patch.object(screen_executor, "user32") as u32, \
             patch("backend.services.screen_executor.time"):
            screen_executor.click(1, 1, clicks="nope")
        self.assertEqual(u32.SendInput.call_count, 2)


if __name__ == "__main__":
    unittest.main()
