"""F21 — keep user secrets out of tool traces.

Acceptance (audit report): "Seeded secrets in text/fills/errors/nested
fields/screenshots/references never reach unauthorized traces or providers."
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from backend.services import tool_policy

SEED = "hunter2-correct-horse"
TOKEN = "sk-abcdefghijklmnopqrstuvwx"


class DeepScrubTests(unittest.TestCase):
    def test_secret_below_the_old_depth_limit_is_masked(self):
        payload = {"password": SEED}
        for _ in range(20):
            payload = {"nested": payload}
        blob = json.dumps(tool_policy.scrub_mapping(payload))
        self.assertNotIn(SEED, blob)

    def test_secret_under_an_innocent_key_name_is_masked(self):
        blob = tool_policy.redact_for_egress({"value": "Bearer " + TOKEN})
        self.assertNotIn(TOKEN, blob)

    def test_nested_secret_under_an_innocent_key_is_masked(self):
        payload = {"result": {"items": [{"label": TOKEN}]}}
        blob = json.dumps(tool_policy.redact_for_egress(payload))
        self.assertNotIn(TOKEN, blob)

    def test_masking_survives_beyond_the_recursion_guard(self):
        payload = {"leaf": SEED}
        for _ in range(60):
            payload = {"n": payload}
        blob = json.dumps(tool_policy.scrub_mapping(payload))
        # Deeper than the guard the subtree is masked wholesale, never passed on.
        self.assertNotIn(SEED, blob)

    def test_reference_cycle_does_not_leak_or_hang(self):
        payload = {"password": SEED}
        payload["self"] = payload
        blob = json.dumps(tool_policy.scrub_mapping(payload))
        self.assertNotIn(SEED, blob)

    def test_plain_values_are_untouched(self):
        payload = {"path": "out/build.txt", "count": 3}
        self.assertEqual(tool_policy.redact_for_egress(payload), payload)

    def test_egress_of_a_non_serialisable_payload_is_masked(self):
        class Weird:
            def __str__(self):
                return "token " + TOKEN

        blob = tool_policy.redact_for_egress(Weird())
        self.assertNotIn(TOKEN, blob)


class VirtualResultTests(unittest.TestCase):
    """A virtual tool's result reaches the model; it must be redacted."""

    def test_virtual_result_is_redacted_before_the_model_sees_it(self):
        from backend.services import browser_agent

        history = []
        call = {"id": "c1", "name": "wait_for",
                "arguments": {"selector": "#x"}}
        with patch.object(browser_agent, "_run_virtual_tool",
                          return_value=("found token " + TOKEN, None)), \
             patch.object(browser_agent, "append_activity_line"), \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "_log_tool_result"), \
             patch.object(browser_agent, "_publish_daemon_tabs"):
            browser_agent._run_one_tool(
                object(), history, call, {"grants": {"privileged_js"}})
        content = history[0]["content"]
        self.assertNotIn(TOKEN, content)
        self.assertIn("***masked***", content)

    def test_tool_arguments_are_redacted_in_the_activity_log(self):
        from backend.services import browser_agent

        logged = []
        call = {"id": "c1", "name": "wait_for",
                "arguments": {"selector": "#x", "password": SEED}}
        with patch.object(browser_agent, "append_activity_line",
                          side_effect=logged.append), \
             patch.object(browser_agent, "narrate_activity"), \
             patch.object(browser_agent, "_log_tool_result"), \
             patch.object(browser_agent, "_publish_daemon_tabs"), \
             patch.object(browser_agent, "_run_virtual_tool",
                          return_value=("ok", None)):
            browser_agent._run_one_tool(
                object(), [], call, {"grants": {"privileged_js"}})
        joined = "".join(logged)
        self.assertNotIn(SEED, joined)
        self.assertIn("***masked***", joined)


class ScreenTraceTests(unittest.TestCase):
    def tearDown(self):
        from backend.services import screen_state
        screen_state.clear_interactions()

    def test_screen_history_masks_marked_typed_input(self):
        from backend.services import screen_state

        screen_state.add_interaction(
            "type the token " + TOKEN,
            "typed the password",
            plan={"steps": [{"action": "type", "text": SEED,
                             "sensitive": True}]},
        )
        entry = screen_state.get_recent_interactions()[0]
        blob = json.dumps(entry)
        # The marked field's value is masked, and the credential-shaped part of
        # the free-text command is masked too.
        self.assertNotIn(SEED, blob)
        self.assertNotIn(TOKEN, blob)

    def test_screen_history_masks_a_label_declared_sensitive(self):
        from backend.services import screen_state

        screen_state.add_interaction(
            "type it",
            "typed the password",
            plan={"steps": [{"action": "type", "text": SEED,
                             "label": "Password"}]},
        )
        entry = screen_state.get_recent_interactions()[0]
        self.assertNotIn(SEED, json.dumps(entry))

    def test_unmarked_screen_history_is_left_readable(self):
        from backend.services import screen_state

        screen_state.add_interaction(
            "click Save",
            "clicked Save",
            plan={"steps": [{"action": "click", "x": 10, "y": 20}]},
        )
        entry = screen_state.get_recent_interactions()[0]
        self.assertIn("click Save", json.dumps(entry))

    def test_screen_command_log_masks_the_error_and_raw_vision(self):
        from backend.services import screen_control

        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "screen_commands.log")
            with patch("backend.config.BASE_DIR", __import__("pathlib").Path(tmp)):
                screen_control._write_screen_command_log(
                    "type the token " + TOKEN,
                    {"steps": [{"action": "type", "text": SEED,
                                "sensitive": True}],
                     "window_title": "Login " + TOKEN,
                     "raw_vision": "saw " + TOKEN},
                    "FAILED",
                    error="provider rejected " + TOKEN,
                )
            with open(log_path, encoding="utf-8") as handle:
                written = handle.read()
        self.assertNotIn(SEED, written)
        self.assertNotIn(TOKEN, written)


class MemoryTraceTests(unittest.TestCase):
    def test_event_detail_dict_is_masked_by_key(self):
        try:
            from backend.core import memory_store
        except Exception:
            self.skipTest("memory store unavailable")
        if not memory_store.MEMORY_ENABLED:
            self.skipTest("memory disabled")
        detail = {"outer": {"password": SEED}}
        blob = memory_store.mask_secrets(json.dumps(
            memory_store.redact_for_egress(detail)))
        self.assertNotIn(SEED, blob)

    def test_memory_store_exposes_the_shared_boundary(self):
        from backend.core import memory_store
        self.assertTrue(callable(memory_store.redact_for_egress))


class ScreenshotRedactionTests(unittest.TestCase):
    def _capture(self, words, elements=None):
        from PIL import Image
        from backend.services import screen_capture

        image = Image.new("RGB", (200, 100), (255, 255, 255))
        capture = {
            "raw_vision_image": image,
            "vision_width": 200,
            "vision_height": 100,
            "image_data_url": "",
        }
        blanked = screen_capture.redact_capture(
            capture, words=words, elements=elements)
        return capture, blanked

    def test_ocr_word_that_is_a_credential_is_blanked(self):
        capture, blanked = self._capture(
            [{"text": TOKEN, "x": 10, "y": 10, "w": 60, "h": 12}])
        self.assertEqual(blanked, 1)
        self.assertEqual(
            capture["raw_vision_image"].getpixel((20, 15)), (0, 0, 0))

    def test_ordinary_text_is_not_blanked(self):
        capture, blanked = self._capture(
            [{"text": "Save", "x": 10, "y": 10, "w": 40, "h": 12}])
        self.assertEqual(blanked, 0)
        self.assertEqual(
            capture["raw_vision_image"].getpixel((20, 15)), (255, 255, 255))

    def test_password_field_element_is_blanked_even_without_a_shape(self):
        capture, blanked = self._capture([], elements=[
            {"name": "Password", "x": 5, "y": 5, "w": 50, "h": 20}])
        self.assertEqual(blanked, 1)

    def test_redaction_refreshes_the_encoded_payload(self):
        capture, _ = self._capture(
            [{"text": TOKEN, "x": 10, "y": 10, "w": 60, "h": 12}])
        self.assertTrue(capture["image_data_url"].startswith("data:image/png"))

    def test_blanking_is_clamped_to_the_image(self):
        from PIL import Image
        from backend.services import screen_capture

        image = Image.new("RGB", (20, 20), (255, 255, 255))
        applied = screen_capture.redact_image_regions(
            image, [{"x": -50, "y": -50, "w": 10, "h": 10},
                    {"x": 100, "y": 100, "w": 10, "h": 10}])
        self.assertEqual(applied, 0)
        applied = screen_capture.redact_image_regions(
            image, [{"x": 5, "y": 5, "w": 999, "h": 999}])
        self.assertEqual(applied, 1)


if __name__ == "__main__":
    unittest.main()
