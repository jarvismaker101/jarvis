"""L-14 / BA-06 — retrying probe helper for internal evaluate calls.

Verified 2026-10-04: every virtual handler called ``client.call_tool(
"evaluate", ...)`` directly, so one dropped connection turned a read-only
probe into ``"look failed: evaluate error: ..."`` — costing a full model
turn to recover. All read-only internal sites now go through ``_probe``
(short 8 s timeout, one fast 50 ms retry); the two MUTATING evaluates
(coordinate click, legacy coordinate fill) keep their single attempt, as
does model-issued ``evaluate`` via ``_MUTATION_TOOLS``.
"""
import json
import unittest
from unittest.mock import patch

from backend.services import browser_agent


class BlipClient:
    """Fails the first ``fail_times`` evaluate calls, then answers."""

    def __init__(self, fail_times=1):
        self.fail_times = fail_times
        self.evaluate_calls = 0
        self.timeouts_seen = []

    def call_tool(self, name, arguments=None, timeout=None):
        if name == "screenshot":
            from PIL import Image
            Image.new("RGB", (100, 100), (255, 255, 255)).save(
                arguments["path"], format="PNG")
            return "saved"
        if name == "evaluate":
            self.evaluate_calls += 1
            self.timeouts_seen.append(timeout)
            if self.evaluate_calls <= self.fail_times:
                raise RuntimeError("connection reset by peer")
            expression = (arguments or {}).get("expression", "")
            if "formEls" in expression:
                return json.dumps({
                    "elements": [
                        {"x": 10, "y": 20, "w": 30, "h": 20,
                         "tag": "a", "label": "Home",
                         "cssPath": "a:nth-of-type(1)",
                         "epoch": {"doc": 111, "mut": 0}, "dpr": 1},
                    ],
                    "url": "https://example.com", "title": "Test",
                    "epoch": {"doc": 111, "mut": 0}, "dpr": 1})
            return json.dumps({"doc": 111, "dpr": 1,
                               "url": "https://example.com",
                               "title": "Test"})
        if name == "list_tabs":
            return "no tabs"
        return "ok"


class LegacyClient(BlipClient):
    """Old-style client without the L-15 per-call timeout kwarg."""

    def call_tool(self, name, arguments=None):  # noqa: F811
        return super().call_tool(name, arguments)


class ProbeUnitTests(unittest.TestCase):
    def test_success_costs_one_call_with_short_timeout(self):
        client = BlipClient(fail_times=0)
        result = browser_agent._probe(client, "1+1")
        self.assertIn("example.com", result)
        self.assertEqual(client.evaluate_calls, 1)
        self.assertEqual(client.timeouts_seen, [8.0])

    def test_one_blip_is_absorbed_transparently(self):
        client = BlipClient(fail_times=1)
        with patch.object(browser_agent.time, "sleep") as sleep_mock:
            result = browser_agent._probe(client, "1+1")
        self.assertIn("example.com", result)
        self.assertEqual(client.evaluate_calls, 2)
        sleep_mock.assert_called_once_with(0.05)

    def test_two_blips_raise_the_last_error(self):
        client = BlipClient(fail_times=2)
        with patch.object(browser_agent.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "connection reset"):
                browser_agent._probe(client, "1+1")
        self.assertEqual(client.evaluate_calls, 2)

    def test_legacy_client_without_timeout_kwarg_still_works(self):
        client = LegacyClient(fail_times=1)
        with patch.object(browser_agent.time, "sleep"):
            result = browser_agent._probe(client, "1+1")
        self.assertIn("example.com", result)
        # First attempt: timeout call TypeErrors, fallback answers…
        # …but that answer counted as attempt 1's blip, so attempt 2 probes.
        self.assertGreaterEqual(client.evaluate_calls, 2)


class LookAbsorbsBlipTests(unittest.TestCase):
    def test_forced_single_failure_on_look_is_absorbed(self):
        client = BlipClient(fail_times=1)
        session = {}
        with patch.object(browser_agent.time, "sleep"):
            text, b64 = browser_agent._handle_look(client, session)
        self.assertNotIn("evaluate error", text)
        self.assertIsNotNone(b64)
        self.assertIn("Home", text)
        self.assertEqual(session["marks"][1]["cx"], 25)

    def test_two_consecutive_failures_still_fail_the_look(self):
        client = BlipClient(fail_times=2)
        session = {}
        with patch.object(browser_agent.time, "sleep"):
            text, b64 = browser_agent._handle_look(client, session)
        self.assertIn("look failed: evaluate error", text)
        self.assertIsNone(b64)


class MutationSingleAttemptTests(unittest.TestCase):
    def test_model_issued_evaluate_stays_single_attempt(self):
        self.assertIn("evaluate", browser_agent._MUTATION_TOOLS)

    def test_mutating_click_evaluate_has_no_retry(self):
        import inspect
        source = inspect.getsource(browser_agent._click_and_confirm)
        head, _, _tail = source.partition("time.sleep")
        self.assertIn('call_tool("evaluate"', head)
        self.assertNotIn("_probe", head)


if __name__ == "__main__":
    unittest.main()
