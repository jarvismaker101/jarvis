"""Two live-failure regressions, both found by running the real stack.

1. ``gemini-3.5-flash-lite`` rejects an explicit ``thinkingBudget: 0`` with
   400 INVALID_ARGUMENT, so every chat call failed and the turn fell through to
   a dead fallback chain ("I'm having trouble connecting").
2. A barge-in stop sent on a keep-alive socket the voice process had already
   closed was dropped instead of re-dialled, so Jarvis kept talking over the
   user.
"""

import socket
import unittest
from unittest.mock import patch

from backend.services import gemini_client
from backend.services import listener


class ThinkingBudgetTests(unittest.TestCase):
    def test_a_zero_budget_is_omitted_not_sent_as_zero(self):
        # gemini-3.5-flash-lite: 400 when thinkingBudget is 0, fine when the
        # field is absent.
        with patch.object(gemini_client, "GEMINI_THINKING_BUDGET", "auto"):
            with patch.object(gemini_client, "_FLASH_LITE_THINKING_BUDGET", 0):
                self.assertIsNone(gemini_client._thinking_config(
                    "gemini-3.5-flash-lite"))
            with patch.object(gemini_client, "_FLASH_LITE_THINKING_BUDGET", 128):
                self.assertEqual(
                    gemini_client._thinking_config("gemini-3.5-flash-lite"),
                    {"thinkingBudget": 128})

    def test_a_pinned_zero_budget_is_also_omitted(self):
        with patch.object(gemini_client, "GEMINI_THINKING_BUDGET", "0"):
            self.assertIsNone(gemini_client._thinking_config("gemini-3.5-flash-lite"))

    def test_other_families_still_send_their_budget(self):
        with patch.object(gemini_client, "GEMINI_THINKING_BUDGET", "auto"):
            self.assertEqual(gemini_client._thinking_config("gemini-2.5-flash"),
                             {"thinkingBudget": 512})
            self.assertEqual(
                gemini_client._thinking_config("gemini-3-pro-preview"),
                {"thinkingBudget": 128})
            self.assertIsNone(gemini_client._thinking_config("some-unknown-model"))

    def test_the_chat_body_carries_no_zero_thinking_config(self):
        with patch.object(gemini_client, "GEMINI_THINKING_BUDGET", "auto"), \
             patch.object(gemini_client, "_FLASH_LITE_THINKING_BUDGET", 0):
            body = gemini_client._build_chat_body(
                [{"role": "user", "content": "hi"}], 0.7, 100,
                "gemini-3.5-flash-lite")
        self.assertNotIn("thinkingConfig", body["generationConfig"])

    def test_a_400_replays_once_without_the_thinking_config(self):
        class _Resp:
            def __init__(self, status, text=""):
                self.status_code = status
                self.text = text

            def json(self):
                return {"candidates": [{"content": {"parts": [
                    {"text": "recovered"}]}}]}

        calls = []

        def _post(url, params=None, json=None, timeout=None):
            calls.append(json)
            if len(calls) == 1:
                return _Resp(400, '{"error": {"message": "invalid argument"}}')
            return _Resp(200)

        with patch.object(gemini_client, "GEMINI_API_KEY", "k"), \
             patch.object(gemini_client, "_session") as session, \
             patch.object(gemini_client, "GEMINI_THINKING_BUDGET", "512"):
            session.post.side_effect = _post
            out = gemini_client.ask_gemini_chat(
                [{"role": "user", "content": "hi"}], model="gemini-3.5-flash")
        self.assertEqual(out["choices"][0]["message"]["content"], "recovered")
        self.assertEqual(len(calls), 2)
        self.assertIn("thinkingConfig", calls[0]["generationConfig"])
        self.assertNotIn("thinkingConfig", calls[1]["generationConfig"])

    def test_a_non_400_is_not_replayed(self):
        class _Resp:
            status_code = 500
            text = "boom"

        with patch.object(gemini_client, "GEMINI_API_KEY", "k"), \
             patch.object(gemini_client, "_session") as session, \
             patch.object(gemini_client, "GEMINI_THINKING_BUDGET", "512"):
            session.post.return_value = _Resp()
            out = gemini_client.ask_gemini_chat(
                [{"role": "user", "content": "hi"}], model="gemini-3.5-flash")
        self.assertEqual(out, {})
        self.assertEqual(session.post.call_count, 1)


class _Resp:
    def __init__(self, status=200):
        self.status = status

    def read(self):
        return b"{}"


class _DeadConn:
    """A keep-alive socket the voice process already closed."""

    def __init__(self, error):
        self.error = error
        self.requests = 0

    def request(self, *args, **kwargs):
        self.requests += 1
        raise self.error

    def getresponse(self):  # pragma: no cover - never reached
        raise AssertionError("request should have raised")

    def close(self):
        pass


class _LiveConn:
    def __init__(self):
        self.requests = 0

    def request(self, *args, **kwargs):
        self.requests += 1

    def getresponse(self):
        return _Resp(200)

    def close(self):
        pass


class StaleSocketBargeInTests(unittest.TestCase):
    """[F32 barge-in] a dead keep-alive socket must not eat the stop."""

    def _worker(self, conns):
        worker = listener._SpeakStopWorker()
        queue = list(conns)

        def _open():
            return queue.pop(0)

        worker._open_connection = _open
        return worker

    def test_an_aborted_socket_is_re_dialled_immediately(self):
        dead = _DeadConn(OSError(10053, "An established connection was aborted"))
        fresh = _LiveConn()
        worker = self._worker([dead, fresh])
        worker._deliver()
        self.assertEqual(dead.requests, 1)
        self.assertEqual(fresh.requests, 1, "the stop was never re-dialled")
        self.assertTrue(worker.stats["last_ok"])
        self.assertEqual(worker.stats["failed"], 0)
        self.assertEqual(worker.stats["delivered"], 1)

    def test_a_connection_reset_is_also_re_dialled(self):
        dead = _DeadConn(ConnectionResetError("connection reset"))
        fresh = _LiveConn()
        worker = self._worker([dead, fresh])
        worker._deliver()
        self.assertEqual(fresh.requests, 1)
        self.assertTrue(worker.stats["last_ok"])

    def test_a_working_first_attempt_never_dials_again(self):
        first = _LiveConn()
        spare = _LiveConn()
        worker = self._worker([first, spare])
        worker._deliver()
        self.assertEqual(first.requests, 1)
        self.assertEqual(spare.requests, 0)
        self.assertTrue(worker.stats["last_ok"])

    def test_an_http_answer_is_never_retried(self):
        class _AuthConn(_LiveConn):
            def getresponse(self):
                return _Resp(401)

        worker = self._worker([_AuthConn(), _LiveConn()])
        worker._deliver()
        self.assertEqual(worker.stats["last_status"], 401)
        self.assertEqual(worker.stats["failed"], 1)
        self.assertEqual(worker.stats["auth_failures"], 1)

    def test_two_dead_sockets_report_a_failure_and_do_not_loop(self):
        worker = self._worker([
            _DeadConn(OSError(10053, "aborted")),
            _DeadConn(socket.timeout("timed out")),
        ])
        worker._deliver()
        self.assertFalse(worker.stats["last_ok"])
        self.assertEqual(worker.stats["failed"], 1)

    def test_a_socket_level_failure_is_counted_as_stale(self):
        worker = self._worker([_LiveConn()])
        self.assertFalse(worker._is_stale_socket())
        worker._last_error = OSError(10053, "aborted")
        self.assertTrue(worker._is_stale_socket())
        worker._last_error = RuntimeError("something else entirely")
        self.assertFalse(worker._is_stale_socket())


if __name__ == "__main__":
    unittest.main()