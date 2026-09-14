"""Round-6 live-bug regression: thinking-only Fireworks models (e.g. GLM)
reject the latency-optimizing reasoning_effort='none' payload field with a
400. Both ask_fireworks_stream and ask_fireworks must retry ONCE with the
field removed, return the successful retry (no Gemini fallback), keep the
existing failure path when the retry also 400s, and never fire a second
call for models that accept the parameter. All HTTP is mocked."""

import json
import unittest
from unittest.mock import MagicMock, patch

from backend.services import fireworks_client

GLM_MODEL = "accounts/fireworks/models/glm-5.3"

BODY_THINKING_ONLY = (
    "{'error':{'object':'error','type':'invalid_request_error',"
    "'code':'invalid_request_error','message':'GLM-5.3 is a thinking-only "
    "model; disabling thinking (reasoning_effort='none') is not "
    "supported.'}}"
)
BODY_INVALID_EFFORT = "{'error':{'message':'Invalid reasoning effort: none'}}"
BODY_UNRELATED = "{'error':{'message':'Invalid model id: nope'}}"


class FireworksClientTestBase(unittest.TestCase):
    """Deterministic module constants + fully mocked HTTP layer."""

    def setUp(self):
        self._api_key = patch.object(fireworks_client, "API_KEY", "test-key")
        self._effort = patch.object(fireworks_client, "REASONING_EFFORT", "none")
        self._post = patch.object(fireworks_client.requests, "post")
        self._api_key.start()
        self._effort.start()
        self.post = self._post.start()
        self.addCleanup(self._api_key.stop)
        self.addCleanup(self._effort.stop)
        self.addCleanup(self._post.stop)
        # Snapshot request payloads at call time: the client mutates the
        # SAME dict (pops reasoning_effort) before retrying, so the mock's
        # stored references would all show the post-pop payload.
        self.sent = []

    def _respond(self, *responses):
        def _post(url, **kwargs):
            self.sent.append(dict(kwargs.get("json") or {}))
            return responses[len(self.sent) - 1]

        self.post.side_effect = _post

    def _sse_response(self, chunks):
        resp = MagicMock()
        resp.status_code = 200
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": text}}]})
            for text in chunks
        ]
        lines.append("data: [DONE]")
        resp.iter_lines.return_value = lines
        return resp

    def _error_response(self, status, body):
        resp = MagicMock()
        resp.status_code = status
        resp.text = body
        return resp

    def _ok_response(self, content):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "choices": [{"message": {"content": content}}]
        }
        return resp


class StreamReasoningRetryTests(FireworksClientTestBase):
    MESSAGES = [{"role": "user", "content": "hello"}]

    def _stream(self):
        return list(
            fireworks_client.ask_fireworks_stream(
                self.MESSAGES, model=GLM_MODEL
            )
        )

    def test_thinking_only_400_retries_once_without_reasoning_effort(self):
        self._respond(
            self._error_response(400, BODY_THINKING_ONLY),
            self._sse_response(["bonjour", " le ", "monde"]),
        )
        self.assertEqual(self._stream(), ["bonjour", " le ", "monde"])
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(self.sent[0]["reasoning_effort"], "none")
        self.assertNotIn("reasoning_effort", self.sent[1])
        self.assertEqual(self.sent[1]["model"], GLM_MODEL)

    def test_invalid_effort_400_also_retries_without_reasoning_effort(self):
        self._respond(
            self._error_response(400, BODY_INVALID_EFFORT),
            self._sse_response(["ok"]),
        )
        self.assertEqual(self._stream(), ["ok"])
        self.assertEqual(self.post.call_count, 2)
        self.assertNotIn("reasoning_effort", self.sent[1])

    def test_double_400_returns_empty_stream(self):
        self._respond(
            self._error_response(400, BODY_THINKING_ONLY),
            self._error_response(400, BODY_INVALID_EFFORT),
        )
        self.assertEqual(self._stream(), [])
        self.assertEqual(self.post.call_count, 2)
        self.assertNotIn("reasoning_effort", self.sent[1])

    def test_accepting_model_never_triggers_second_call(self):
        self._respond(self._sse_response(["salut"]))
        self.assertEqual(self._stream(), ["salut"])
        self.assertEqual(self.post.call_count, 1)
        self.assertEqual(self.sent[0]["reasoning_effort"], "none")

    def test_unrelated_400_does_not_retry(self):
        self._respond(self._error_response(400, BODY_UNRELATED))
        self.assertEqual(self._stream(), [])
        self.assertEqual(self.post.call_count, 1)

    def test_500_mentioning_thinking_does_not_retry(self):
        self._respond(self._error_response(500, "internal error while thinking"))
        self.assertEqual(self._stream(), [])
        self.assertEqual(self.post.call_count, 1)


class NonStreamReasoningRetryTests(FireworksClientTestBase):
    MESSAGES = [{"role": "user", "content": "hello"}]

    def _ask(self):
        return fireworks_client.ask_fireworks(self.MESSAGES, model=GLM_MODEL)

    def test_thinking_only_400_retries_once_without_reasoning_effort(self):
        self._respond(
            self._error_response(400, BODY_THINKING_ONLY),
            self._ok_response("bonjour le monde"),
        )
        result = self._ask()
        self.assertEqual(
            result["choices"][0]["message"]["content"], "bonjour le monde"
        )
        self.assertEqual(self.post.call_count, 2)
        self.assertEqual(self.sent[0]["reasoning_effort"], "none")
        self.assertNotIn("reasoning_effort", self.sent[1])
        self.assertEqual(self.sent[1]["model"], GLM_MODEL)

    def test_double_400_returns_error_failure(self):
        self._respond(
            self._error_response(400, BODY_THINKING_ONLY),
            self._error_response(400, BODY_INVALID_EFFORT),
        )
        result = self._ask()
        self.assertEqual(result.get("error"), 400)
        self.assertEqual(self.post.call_count, 2)
        self.assertNotIn("reasoning_effort", self.sent[1])

    def test_accepting_model_never_triggers_second_call(self):
        self._respond(self._ok_response("salut"))
        result = self._ask()
        self.assertEqual(
            result["choices"][0]["message"]["content"], "salut"
        )
        self.assertEqual(self.post.call_count, 1)
        self.assertEqual(self.sent[0]["reasoning_effort"], "none")

    def test_unrelated_400_does_not_retry(self):
        # F24: the non-stream retry is gated — an unrelated validation 400
        # is never replayed (only transient statuses or the rejected-
        # reasoning-parameter 400 retry).
        self._respond(self._error_response(400, BODY_UNRELATED))
        result = self._ask()
        self.assertEqual(result.get("error"), 400)
        self.assertEqual(self.post.call_count, 1)

    def test_transient_429_retries_once(self):
        self._respond(
            self._error_response(429, "rate limited"),
            self._ok_response("recovered"),
        )
        result = self._ask()
        self.assertEqual(
            result["choices"][0]["message"]["content"], "recovered"
        )
        self.assertEqual(self.post.call_count, 2)

    def test_transient_503_retries_once(self):
        self._respond(
            self._error_response(503, "service unavailable"),
            self._ok_response("back up"),
        )
        result = self._ask()
        self.assertEqual(
            result["choices"][0]["message"]["content"], "back up"
        )
        self.assertEqual(self.post.call_count, 2)

    def test_auth_401_never_retries(self):
        self._respond(self._error_response(401, "invalid api key"))
        result = self._ask()
        self.assertEqual(result.get("error"), 401)
        self.assertEqual(self.post.call_count, 1)

    def test_auth_403_never_retries(self):
        self._respond(self._error_response(403, "forbidden"))
        result = self._ask()
        self.assertEqual(result.get("error"), 403)
        self.assertEqual(self.post.call_count, 1)

    def test_missing_model_404_never_retries(self):
        self._respond(self._error_response(404, "model not found"))
        result = self._ask()
        self.assertEqual(result.get("error"), 404)
        self.assertEqual(self.post.call_count, 1)


class StreamReasoningSeparationTests(FireworksClientTestBase):
    """F31: the default stream carries ONLY final-answer content — reasoning
    deltas never reach chat/TTS/memory unless explicitly opted in."""

    MESSAGES = [{"role": "user", "content": "hello"}]

    def _sse_with_reasoning(self, chunks):
        resp = MagicMock()
        resp.status_code = 200
        lines = []
        for reasoning, content in chunks:
            delta = {}
            if reasoning is not None:
                delta["reasoning_content"] = reasoning
            if content is not None:
                delta["content"] = content
            lines.append("data: " + json.dumps({"choices": [{"delta": delta}]}))
        lines.append("data: [DONE]")
        resp.iter_lines.return_value = lines
        return resp

    def test_reasoning_absent_from_default_stream(self):
        self._respond(self._sse_with_reasoning([
            ("let me think...", None),
            (None, "hel"),
            ("more thinking", None),
            (None, "lo"),
        ]))
        out = list(fireworks_client.ask_fireworks_stream(
            self.MESSAGES, model=GLM_MODEL
        ))
        self.assertEqual(out, ["hel", "lo"])
        self.assertNotIn("let me think...", out)
        self.assertNotIn("more thinking", out)

    def test_reasoning_available_via_opt_in(self):
        # F31: the reasoning channel is TYPED — opt-in reasoning arrives as
        # StreamDelta(channel='reasoning'), never as a bare string that a
        # consumer could concatenate into answer text.
        self._respond(self._sse_with_reasoning([
            ("let me think...", None),
            (None, "hello"),
        ]))
        out = list(fireworks_client.ask_fireworks_stream(
            self.MESSAGES, model=GLM_MODEL, include_reasoning=True
        ))
        self.assertEqual(
            [(delta.channel, delta.text) for delta in out],
            [("reasoning", "let me think..."), ("final", "hello")],
        )


if __name__ == "__main__":
    unittest.main()
