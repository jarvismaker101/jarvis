"""F38 — Stop dropping Gemini browser images (acceptance pins).

Acceptance clause under test:

    "Fixture and deployed-contract tests cover parallel/repeated-name tools
     and images. Exact API acceptance is unverified, not asserted broken for
     every version."

So this module pins the SHAPE the adapter must produce/consume (a fixture
contract) plus the deployed response contract as parsed from a recorded-style
payload. It explicitly does NOT assert that every deployed Gemini version
accepts the payload — that remains unverified.
"""

import json
import unittest
from unittest.mock import Mock, patch

from backend.services import browser_agent


class GeminiFixtureTests(unittest.TestCase):
    """Fixture: the neutral history -> Gemini contents mapping."""

    def test_two_parallel_calls_with_the_same_name_stay_correlated(self):
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_a", "name": "look", "arguments": {}},
                {"id": "call_b", "name": "look", "arguments": {}},
            ]},
            {"role": "tool", "tool_call_id": "call_a", "name": "look",
             "content": "first observation", "image_b64": "IMG_A"},
            {"role": "tool", "tool_call_id": "call_b", "name": "look",
             "content": "second observation", "image_b64": "IMG_B"},
        ]
        contents = browser_agent._to_gemini_contents(history)
        # ONE model turn and ONE grouped user turn for both results.
        self.assertEqual([c["role"] for c in contents], ["model", "user"])
        calls = [part["functionCall"] for part in contents[0]["parts"]]
        self.assertEqual([c["id"] for c in calls], ["call_a", "call_b"])
        parts = contents[1]["parts"]
        self.assertEqual(len(parts), 2)
        responses = [part["functionResponse"] for part in parts]
        self.assertEqual([r["id"] for r in responses], ["call_a", "call_b"])
        self.assertEqual([r["response"]["result"] for r in responses],
                         ["first observation", "second observation"])
        # The image rides inside the response it belongs to - never in a
        # separate message that could drift to the other call.
        self.assertEqual(
            responses[0]["parts"][1]["inlineData"]["data"], "IMG_A")
        self.assertEqual(
            responses[1]["parts"][1]["inlineData"]["data"], "IMG_B")
        self.assertNotIn("IMG_B", json.dumps(responses[0]))

    def test_single_image_stays_in_its_own_response(self):
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "name": "look", "arguments": {}},
                {"id": "c2", "name": "navigate", "arguments": {"url": "https://x"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "look",
             "content": "page", "image_b64": "ONLY_IMAGE"},
            {"role": "tool", "tool_call_id": "c2", "name": "navigate",
             "content": "opened"},
        ]
        contents = browser_agent._to_gemini_contents(history)
        self.assertEqual(len(contents), 2)
        responses = {part["functionResponse"]["id"]: part["functionResponse"]
                     for part in contents[1]["parts"]}
        self.assertIn("parts", responses["c1"])
        self.assertEqual(responses["c1"]["parts"][1]["inlineData"]["data"],
                         "ONLY_IMAGE")
        self.assertNotIn("parts", responses["c2"])

    def test_repeated_name_without_ids_still_gets_distinct_ids(self):
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"name": "look", "arguments": {}},
                {"name": "look", "arguments": {}},
            ]},
            {"role": "tool", "name": "look", "content": "one"},
            {"role": "tool", "name": "look", "content": "two"},
        ]
        contents = browser_agent._to_gemini_contents(history)
        calls = [part["functionCall"] for part in contents[0]["parts"]]
        self.assertEqual(len({c["id"] for c in calls}), 2)
        responses = [part["functionResponse"] for part in contents[1]["parts"]]
        self.assertEqual(len({r["id"] for r in responses}), 2)
        self.assertEqual([r["response"]["result"] for r in responses],
                         ["one", "two"])

    def test_thought_signature_is_echoed_back(self):
        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "name": "look", "arguments": {},
                 "thought_signature": "SIG-123"},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "look",
             "content": "page"},
        ]
        contents = browser_agent._to_gemini_contents(history)
        part = contents[0]["parts"][0]
        self.assertEqual(part["thoughtSignature"], "SIG-123")
        self.assertEqual(part["functionCall"]["id"], "c1")

    def test_openai_adapter_keeps_image_parts(self):
        messages = [{"role": "tool", "tool_call_id": "c1", "name": "look",
                     "content": "hello", "image_b64": "abc123"}]
        out = browser_agent._to_openai_messages(messages)
        self.assertEqual(out[0]["content"][1]["type"], "image_url")
        self.assertIn("data:image/jpeg;base64,abc123",
                      out[0]["content"][1]["image_url"]["url"])


class DeployedResponseTests(unittest.TestCase):
    """Deployed-contract parsing of a generateContent response."""

    def _gemini_response(self, parts, index=0):
        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "candidates": [{"index": index, "content": {"parts": parts}}]}
        return response

    def _call(self, parts, index=0):
        session = Mock()
        session.post.return_value = self._gemini_response(parts, index)
        with patch.object(browser_agent, "_MODEL_SESSION", session):
            return browser_agent._call_gemini([], [], model="gemini-test")

    def test_api_issued_ids_and_signatures_are_preserved(self):
        _text, calls = self._call([
            {"functionCall": {"id": "fc-1", "name": "look", "args": {}},
             "thoughtSignature": "SIG-1"},
            {"functionCall": {"id": "fc-2", "name": "look", "args": {"a": 1}},
             "thought_signature": "SIG-2"},
        ])
        self.assertEqual([c["id"] for c in calls], ["fc-1", "fc-2"])
        self.assertEqual([c["thought_signature"] for c in calls],
                         ["SIG-1", "SIG-2"])
        self.assertEqual(calls[1]["arguments"], {"a": 1})

    def test_stringified_arguments_are_parsed_but_bad_json_is_refused(self):
        _text, calls = self._call([
            {"functionCall": {"id": "ok", "name": "scroll",
                              "args": "{\"direction\": \"down\"}"}},
            {"functionCall": {"id": "bad", "name": "click_text",
                              "args": "{not json"}},
        ])
        self.assertEqual(calls[0]["arguments"], {"direction": "down"})
        self.assertNotIn("parse_error", calls[0])
        self.assertEqual(calls[1]["arguments"], {})
        self.assertIn("parse_error", calls[1])

    def test_malformed_deployed_call_executes_nothing(self):
        _text, calls = self._call([
            {"functionCall": {"id": "bad", "name": "click_text",
                              "args": "[1, 2, 3]"}},
        ])
        self.assertIn("parse_error", calls[0])

        class NeverCalled:
            def call_tool(self, name, arguments=None):
                raise AssertionError("nothing may be dispatched")

        history = []
        browser_agent._run_one_tool(NeverCalled(), history, calls[0], session={})
        self.assertIn("not valid JSON", history[-1]["content"])

    def test_text_and_call_parts_together(self):
        text, calls = self._call([
            {"text": "I will look at the page."},
            {"functionCall": {"id": "fc-1", "name": "look", "args": {}}},
        ])
        self.assertEqual(text, "I will look at the page.")
        self.assertEqual(len(calls), 1)


class InlineImagePreservationTests(unittest.TestCase):
    """F38: an image the daemon returns inline must not be dropped."""

    def test_image_blocks_are_preserved_alongside_the_text_marker(self):
        from backend.services.brave_mcp_client import BraveMcpClient
        client = BraveMcpClient(base_url="http://x/mcp", token="tok")
        with patch.object(client, "_request", return_value={"content": [
            {"type": "text", "text": "saved"},
            {"type": "image", "mimeType": "image/png", "data": "QUJD"},
            {"type": "image", "data": ""},
        ]}):
            text = client.call_tool("screenshot", {})
        # The text contract is unchanged...
        self.assertEqual(text, "saved\n[image omitted]\n[image omitted]")
        # ... but the payload is no longer unrecoverable.
        self.assertEqual(len(client.last_images), 1)
        self.assertEqual(client.last_images[0]["data"], "QUJD")
        self.assertEqual(client.last_images[0]["mime_type"], "image/png")

    def test_an_inline_screenshot_makes_look_work(self):
        import base64
        import io
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (200, 120), (10, 20, 30)).save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

        class InlineClient:
            def __init__(self):
                self.last_images = [{"mime_type": "image/png",
                                     "data": encoded}]

            def call_tool(self, name, arguments=None):
                arguments = arguments or {}
                if name == "evaluate":
                    return json.dumps({"elements": [], "url": "https://x",
                                       "title": "T", "doc": 1, "mut": 0,
                                       "dpr": 1})
                # The daemon answers the screenshot call but never writes the
                # requested file - the image comes back inline instead.
                return "saved"

        text, image_b64 = browser_agent._handle_look(InlineClient(), {})
        self.assertIsNotNone(image_b64, text)
        self.assertNotIn("did not create file", text)

    def test_look_still_fails_honestly_when_no_image_arrives(self):
        class NoImageClient:
            last_images = []

            def call_tool(self, name, arguments=None):
                if name == "evaluate":
                    return json.dumps({"elements": [], "url": "https://x",
                                       "title": "T", "doc": 1, "mut": 0,
                                       "dpr": 1})
                return "saved"

        text, image_b64 = browser_agent._handle_look(NoImageClient(), {})
        self.assertIsNone(image_b64)
        self.assertIn("did not create file", text)


class ModelAdapterPairTests(unittest.TestCase):
    """F38: "reject incapable model-adapter pairs"."""

    def test_shipped_adapters_keep_tools_and_vision(self):
        for provider in ("fireworks", "groq", "openrouter", "gemini"):
            self.assertTrue(
                browser_agent._adapter_supports_tools_and_vision(provider),
                provider)

    def test_unregistered_provider_is_rejected(self):
        self.assertFalse(browser_agent._adapter_supports_tools_and_vision("nope"))
        self.assertFalse(browser_agent._adapter_supports_tools_and_vision(""))

    def test_registered_provider_without_vision_is_rejected(self):
        with patch("backend.services.model_registry.model_capabilities_for",
                   return_value={"tool_calling"}):
            self.assertFalse(
                browser_agent._adapter_supports_tools_and_vision("custom", "m"))

    def test_registered_provider_with_both_capabilities_is_accepted(self):
        with patch("backend.services.model_registry.model_capabilities_for",
                   return_value={"tool_calling", "vision_input"}):
            self.assertTrue(
                browser_agent._adapter_supports_tools_and_vision("custom", "m"))

    def test_model_turn_rejects_an_incapable_pair(self):
        with patch.object(browser_agent, "_resolve_browser_tool_model",
                          return_value=("mystery", "text-only")), \
             patch("backend.services.model_registry.model_capabilities_for",
                   return_value={"tool_calling"}), \
             patch.object(browser_agent, "_call_openai_compatible") as call:
            with self.assertRaises(RuntimeError) as ctx:
                browser_agent._model_turn([], [])
        call.assert_not_called()
        self.assertIn("selection rejected", str(ctx.exception))

    def test_model_turn_accepts_a_registered_capable_pair(self):
        with patch.object(browser_agent, "_resolve_browser_tool_model",
                          return_value=("mystery", "vl-1")), \
             patch("backend.services.model_registry.model_capabilities_for",
                   return_value={"tool_calling", "vision_input"}), \
             patch("backend.services.model_registry.get_provider_credentials",
                   return_value=("key", "https://api.example/v1")), \
             patch.object(browser_agent, "_call_openai_compatible",
                          return_value=("done", [])) as call:
            text, calls = browser_agent._model_turn([], [])
        self.assertTrue(call.called)
        self.assertEqual(text, "done")
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
