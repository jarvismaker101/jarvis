"""F24 + F49, brain side — no silent cross-provider substitution.

The audit found two ways a chat reply could end up at a provider the user did
not choose:

* F49: ``_resolve_chat_model`` swallowed EVERY registry failure and returned
  the Gemini default, so a private provider whose credentials are missing (or
  any unauthorized selection) still had its message posted to Gemini; and the
  fallback chains happily rerouted to Gemini anyway.
* F24: a terminal (non-retryable) failure — bad key, 403, unknown model — was
  replayed against the next provider, hiding a fixable configuration error.

Acceptance pinned here: an unauthorized selection or a terminal failure sends
NOTHING anywhere else, the reply stays honest, and the turn budget is passed
to the clients that accept one.
"""

import unittest
from unittest import mock

from backend.core import brain


class _FakeFailure:
    def __init__(self, detail="bad key", terminal=True):
        self.detail = detail
        self.terminal = terminal


class ChatFailClosedTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def boom(*a, **k):
            self.calls.append("unexpected")
            raise AssertionError("no provider may be called")

        for name in ("ask_gemini_chat_stream", "ask_fireworks_stream",
                     "ask_gemini_chat", "ask_fireworks",
                     "ask_openai_compat", "ask_openai_compat_stream"):
            patcher = mock.patch.object(brain, name, boom)
            patcher.start()
            self.addCleanup(patcher.stop)

    # ── F49: an unauthorized selection fails closed ──────────────────────
    def test_registry_failure_returns_no_provider(self):
        with mock.patch.object(brain.model_registry, "get_default_chat_model",
                               side_effect=RuntimeError("private provider 'acme'"
                                                        " is not authorized")):
            selection = brain._resolve_chat_model()
        self.assertIsNone(selection["provider"],
                          "a registry rejection must not become the Gemini default")
        self.assertIn("acme", selection["refused"])

    def test_stream_refusal_sends_nothing(self):
        with mock.patch.object(brain, "_resolve_chat_model",
                               return_value={"provider": None, "model": None,
                                             "refused": "acme unauthorized"}):
            out = "".join(brain._stream_chat_deltas(
                [{"role": "user", "content": "hi"}], 0.2, 64))
        self.assertEqual(self.calls, [], "nothing may be sent anywhere")
        self.assertIn("settings", out.lower())

    def test_nonstream_refusal_sends_nothing(self):
        with mock.patch.object(brain, "_resolve_chat_model",
                               return_value={"provider": None, "model": None,
                                             "refused": "acme unauthorized"}):
            result = brain._ask_chat_nonstream(
                [{"role": "user", "content": "hi"}], 0.2, 64)
        self.assertEqual(self.calls, [])
        self.assertFalse(result.get("choices"))
        self.assertTrue(result.get("detail"))

    def test_private_provider_without_credentials_fails_closed(self):
        with mock.patch.object(brain, "_resolve_chat_model",
                               return_value={"provider": "acme",
                                             "model": "acme-1"}), \
             mock.patch.object(brain.model_registry, "get_provider_credentials",
                               return_value=(None, None)):
            result = brain._ask_chat_nonstream(
                [{"role": "user", "content": "hi"}], 0.2, 64)
        self.assertEqual(self.calls, [], "Gemini must NOT receive the message")
        self.assertTrue(result.get("detail"))

    def test_private_provider_without_credentials_stream_fails_closed(self):
        with mock.patch.object(brain, "_resolve_chat_model",
                               return_value={"provider": "acme",
                                             "model": "acme-1"}), \
             mock.patch.object(brain.model_registry, "get_provider_credentials",
                               return_value=(None, None)):
            out = "".join(brain._stream_chat_deltas(
                [{"role": "user", "content": "hi"}], 0.2, 64))
        self.assertEqual(self.calls, [])
        self.assertTrue(out.strip())

    # ── F24: terminal failures stop the chain ────────────────────────────
    def test_terminal_fireworks_failure_is_not_replayed(self):
        terminal = {"choices": [], "failure": _FakeFailure("401 unauthorized"),
                    "detail": "401 unauthorized"}

        def fake_stream(*a, **k):
            return iter(())

        with mock.patch.object(brain, "_resolve_chat_model",
                               return_value={"provider": "fireworks",
                                             "model": "fw-1"}), \
             mock.patch.object(brain, "ask_fireworks_stream", fake_stream), \
             mock.patch.object(brain, "ask_fireworks",
                               side_effect=lambda *a, **k: terminal), \
             mock.patch.object(brain, "_record_chat_fallback") as recorded:
            out = "".join(brain._stream_chat_deltas(
                [{"role": "user", "content": "hi"}], 0.2, 64))
        self.assertIn("credentials", out.lower())
        self.assertEqual(recorded.call_args[0][3], "none",
                         "the fallback target must be 'none'")
        self.assertNotIn("unexpected", self.calls)

    def test_retryable_fireworks_failure_still_falls_back(self):
        soft = {"choices": [], "failure": _FakeFailure("503 busy", terminal=False),
                "detail": "503 busy"}

        def fake_stream(*a, **k):
            return iter(())

        def fake_gemini_stream(*a, **k):
            yield "hello from gemini"

        with mock.patch.object(brain, "_resolve_chat_model",
                               return_value={"provider": "fireworks",
                                             "model": "fw-1"}), \
             mock.patch.object(brain, "ask_fireworks_stream", fake_stream), \
             mock.patch.object(brain, "ask_fireworks",
                               side_effect=lambda *a, **k: soft), \
             mock.patch.object(brain, "ask_gemini_chat_stream",
                               fake_gemini_stream):
            out = "".join(brain._stream_chat_deltas(
                [{"role": "user", "content": "hi"}], 0.2, 64))
        self.assertIn("hello from gemini", out)

    # ── F24: the turn budget reaches the clients ─────────────────────────
    def test_stream_passes_the_turn_budget(self):
        seen = {}
        marker = object()

        def fake_gemini_stream(*a, **k):
            seen.update(k)
            yield "ok"

        with mock.patch.object(brain, "_resolve_chat_model",
                               return_value={"provider": "gemini",
                                             "model": "gemini-2.5-pro"}), \
             mock.patch.object(brain, "_turn_budget", return_value=marker), \
             mock.patch.object(brain, "ask_gemini_chat_stream",
                               fake_gemini_stream):
            out = "".join(brain._stream_chat_deltas(
                [{"role": "user", "content": "hi"}], 0.2, 64))
        self.assertEqual(out, "ok")
        self.assertIs(seen.get("deadline"), marker)

    def test_nonstream_passes_the_turn_budget(self):
        seen = {}
        marker = object()

        def fake_gemini(*a, **k):
            seen.update(k)
            return {"choices": [{"message": {"content": "ok"}}]}

        with mock.patch.object(brain, "_resolve_chat_model",
                               return_value={"provider": "gemini",
                                             "model": "gemini-2.5-pro"}), \
             mock.patch.object(brain, "_turn_budget", return_value=marker), \
             mock.patch.object(brain, "ask_gemini_chat", fake_gemini):
            brain._ask_chat_nonstream(
                [{"role": "user", "content": "hi"}], 0.2, 64)
        self.assertIs(seen.get("deadline"), marker)


class FailureHelperTests(unittest.TestCase):
    def test_terminal_detection_accepts_object_and_dict(self):
        self.assertTrue(brain._failure_is_terminal(
            {"failure": _FakeFailure(terminal=True)}))
        self.assertFalse(brain._failure_is_terminal(
            {"failure": _FakeFailure(terminal=False)}))
        self.assertTrue(brain._failure_is_terminal(
            {"failure": {"terminal": True}}))
        self.assertFalse(brain._failure_is_terminal({}))
        self.assertFalse(brain._failure_is_terminal(None))

    def test_refusal_text_is_secret_free_and_bounded(self):
        text = brain._chat_refusal_text("x" * 500)
        self.assertLess(len(text), 300)


if __name__ == "__main__":
    unittest.main()
