"""L-8 / BA-02: retryable vs. deterministic model errors.

A forced 400 must fail in ONE attempt with ZERO sleeps and surface the
provider's message; a forced 503 must still retry 3x; the provider's
Retry-After must beat the flat 0.5 s gap. Bare errors without a provider
response keep today's retry behaviour (pinned by the pre-existing suite).
"""
import unittest
from unittest.mock import patch

import requests

from backend.services import browser_agent


def _http_error(status, body=None, headers=None):
    """A requests.HTTPError shaped like raise_for_status() produces."""
    resp = requests.Response()
    resp.status_code = status
    resp.headers.update(headers or {})
    if isinstance(body, dict):
        import json as _json
        resp._content = _json.dumps(body).encode("utf-8")
        resp.headers["Content-Type"] = "application/json"
    elif body is not None:
        resp._content = str(body).encode("utf-8")
    else:
        resp._content = b""
    err = requests.HTTPError("400 Client Error: Bad Request")
    err.response = resp
    return err


class IsRetryableTests(unittest.TestCase):
    def test_deterministic_statuses_fail_fast(self):
        for status in (400, 401, 403, 404, 413, 422):
            self.assertFalse(
                browser_agent._is_retryable(_http_error(status)),
                "status %d" % status)

    def test_transient_statuses_retry(self):
        for status in (408, 409, 425, 429, 500, 502, 503, 504):
            self.assertTrue(
                browser_agent._is_retryable(_http_error(status)),
                "status %d" % status)

    def test_transport_errors_retry(self):
        self.assertTrue(
            browser_agent._is_retryable(requests.Timeout("timed out")))
        self.assertTrue(browser_agent._is_retryable(
            requests.ConnectionError("reset")))

    def test_bare_errors_keep_retrying(self):
        # Deliberate deviation from the audit sketch (documented in code):
        # an error with NO provider response is ambiguous, so it retries.
        self.assertTrue(browser_agent._is_retryable(RuntimeError("boom")))
        self.assertTrue(browser_agent._is_retryable(ValueError("x")))
        self.assertTrue(browser_agent._is_retryable(
            _http_error(400, body=None) if False else RuntimeError("no resp")))


class RetryAfterTests(unittest.TestCase):
    def test_seconds_header(self):
        err = _http_error(429, headers={"Retry-After": "3"})
        self.assertEqual(browser_agent._retry_after_s(err), 3.0)

    def test_capped(self):
        err = _http_error(503, headers={"Retry-After": "3600"})
        self.assertEqual(browser_agent._retry_after_s(err),
                         browser_agent._RETRY_AFTER_CAP_S)

    def test_missing_or_garbage_is_zero(self):
        self.assertEqual(browser_agent._retry_after_s(_http_error(500)), 0.0)
        self.assertEqual(
            browser_agent._retry_after_s(
                _http_error(500, headers={"Retry-After": "soon"})), 0.0)

    def test_no_response_is_zero(self):
        self.assertEqual(
            browser_agent._retry_after_s(RuntimeError("x")), 0.0)

    def test_delay_falls_back_to_flat_gap(self):
        self.assertEqual(
            browser_agent._retry_delay_s(RuntimeError("x")),
            browser_agent._RETRY_GAP_S)
        err = _http_error(429, headers={"Retry-After": "2"})
        self.assertEqual(browser_agent._retry_delay_s(err), 2.0)


class ProviderMessageTests(unittest.TestCase):
    def test_json_error_message(self):
        err = _http_error(400, body={"error": {"message": "image too large"}})
        self.assertEqual(browser_agent._provider_message(err), "image too large")

    def test_top_level_message(self):
        err = _http_error(400, body={"message": "context length exceeded"})
        self.assertEqual(browser_agent._provider_message(err),
                         "context length exceeded")

    def test_plain_text_body(self):
        err = _http_error(400, body="bad request: schema")
        self.assertIn("bad request", browser_agent._provider_message(err))

    def test_no_response_is_empty(self):
        self.assertEqual(browser_agent._provider_message(RuntimeError("x")), "")

    def test_failure_message_carries_provider_words(self):
        err = _http_error(400, body={"error": {"message": "image too large"}})
        msg = browser_agent._model_failure_message(err)
        self.assertIn("image too large", msg)


class LoopBehaviourTests(unittest.TestCase):
    def _run(self, side_effect, **kw):
        lines = []
        sleeps = []
        with patch.object(browser_agent, "_model_turn",
                          side_effect=side_effect), \
             patch.object(browser_agent, "append_activity_line",
                          side_effect=lambda s: lines.append(s)), \
             patch.object(browser_agent.time, "sleep",
                          side_effect=lambda s: sleeps.append(s)):
            try:
                browser_agent._model_turn_with_retries(
                    [{"role": "user", "content": "go"}], [], step=7, **kw)
                raised = None
            except RuntimeError as exc:
                raised = exc
        return raised, lines, sleeps

    def test_forced_400_fails_in_one_attempt_with_zero_sleeps(self):
        err = _http_error(400, body={"error": {"message": "image too large"}})
        calls = {"n": 0}

        def once(history, tools):
            calls["n"] += 1
            raise err

        raised, lines, sleeps = self._run(side_effect=once)
        self.assertIsNotNone(raised)
        # ONE model call total, ZERO sleeps: nothing was retried.
        self.assertEqual(calls["n"], 1)
        self.assertEqual(sleeps, [])
        # The single failed attempt's wall time is still recorded (attempt 0),
        # then the deterministic break is logged explicitly.
        retries = [l for l in lines if "model retry step=7 " in l]
        self.assertEqual(len(retries), 1)
        det = [l for l in lines if "model deterministic step=7 " in l]
        self.assertEqual(len(det), 1)
        self.assertIn("image too large", str(raised))
        fail = next(l for l in lines if "ok=false" in l)
        self.assertIn("step=7", fail)

    def test_forced_503_retries_three_times_with_backoff(self):
        err = _http_error(503)
        raised, lines, sleeps = self._run(side_effect=err)
        self.assertIsNotNone(raised)
        retries = [l for l in lines if "model retry step=7 " in l]
        self.assertEqual(len(retries), 2)
        self.assertEqual(len(sleeps), 2)

    def test_retry_after_beats_flat_gap(self):
        err = _http_error(429, headers={"Retry-After": "4"})
        calls = {"n": 0}

        def flaky(history, tools):
            calls["n"] += 1
            if calls["n"] == 1:
                raise err
            return "done", []

        raised, lines, sleeps = self._run(side_effect=flaky)
        self.assertIsNone(raised)
        self.assertEqual(sleeps, [4.0])

    def test_bare_error_still_retries_then_succeeds(self):
        calls = {"n": 0}

        def flaky(history, tools):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("connection reset")
            return "done", []

        raised, lines, sleeps = self._run(side_effect=flaky)
        self.assertIsNone(raised)
        self.assertEqual(len(sleeps), 1)
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
