"""F24 — enforce real end-to-end deadlines.

Acceptance (audit report): "Expired budgets send no requests; auth failures
are not replayed; trickling streams cannot extend deadlines; completed
evidence survives as partial results."

The tests pin the four clauses against the real code paths:

  * the shared ``backend.core.deadline`` handle (coercion, slicing, waits,
    thread binding, job propagation) and the central failure classifier;
  * ``gemini_client`` / ``fireworks_client`` transports: no request on a
    spent budget, eligible-only replays within the remaining window, and a
    deadline-aware urllib3 retry strategy that refuses hidden retries;
  * ``executor``'s YouTube lookup (previously unbounded) and its batch wait.
"""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

import requests
from urllib3.exceptions import MaxRetryError

from backend.core import deadline as budget
from backend.core import executor
from backend.services import fireworks_client
from backend.services import gemini_client
from backend.services import jobs as job_registry


class FakeClock:
    """A monotonic clock a mocked transport can advance mid-request."""

    def __init__(self, start=1000.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


def _ok_response(content):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


def _gemini_ok_response(content):
    """Gemini's own (candidates/parts) response shape."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": content}]}}]
    }
    return resp


def _status_response(status, body="boom"):
    resp = MagicMock()
    resp.status_code = status
    resp.text = body
    return resp


# ── the shared handle + classifier ─────────────────────────────────────────
class SharedBudgetTests(unittest.TestCase):
    def test_a_spent_budget_yields_no_timeout_slice(self):
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        self.assertTrue(spent.expired())
        self.assertIsNone(spent.timeout((8, 45)))
        self.assertIsNone(spent.seconds(10))

    def test_a_live_budget_slices_timeouts_to_what_is_left(self):
        live = budget.Deadline.after(0.5)
        connect, read = live.timeout((8, 45))
        self.assertLessEqual(connect, 0.5)
        self.assertLessEqual(read, 0.5)
        self.assertGreater(read, 0)
        self.assertLessEqual(live.seconds(10), 0.5)

    def test_a_child_budget_cannot_outlive_its_parent(self):
        parent = budget.Deadline.after(0.2)
        parent_left = parent.remaining()
        child = parent.child(60)
        self.assertLessEqual(child.remaining(), parent_left + 1e-6)
        # The cancellation travels with the window.
        cancel = MagicMock()
        cancel.is_set.return_value = True
        cancelled_parent = budget.Deadline(None, cancel=cancel)
        self.assertTrue(cancelled_parent.child(5).cancelled())

    def test_a_wait_is_capped_and_never_grants_fresh_time(self):
        live = budget.Deadline.after(0.1)
        started = time.monotonic()
        live.sleep(30)                      # would be 30s without the cap
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertLessEqual(live.remaining(), 0.05)
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        self.assertFalse(spent.sleep(0.1))

    def test_the_handle_is_visible_to_argument_less_layers(self):
        handle = budget.Deadline.after(5)
        self.assertIsNone(budget.current())
        with budget.bound(handle):
            self.assertIs(budget.current(), handle)
            # resolve() inherits the ambient budget when given none.
            self.assertIs(budget.resolve(None), handle)
        self.assertIsNone(budget.current())

    def test_a_job_token_is_one_budget_handle(self):
        job = job_registry.new_job(kind="test", timeout=30.0)
        try:
            handle = budget.Deadline.coerce(job)
            self.assertIsNotNone(handle)
            self.assertFalse(handle.expired())
            self.assertAlmostEqual(handle.remaining(), job.remaining(), delta=0.5)
            job.cancel("stopped by user")
            self.assertTrue(handle.cancelled())
            self.assertTrue(handle.stopped())
        finally:
            job.finish()

    def test_a_job_budget_expires_with_the_job(self):
        job = job_registry.new_job(kind="test", timeout=0.05)
        try:
            handle = job.budget()
            time.sleep(0.1)
            self.assertTrue(handle.expired())
            self.assertTrue(handle.stopped())
        finally:
            job.finish()

    def test_turn_budget_follows_the_bound_turn_job(self):
        job = job_registry.new_job(kind="request", timeout=30.0)
        try:
            self.assertIsNone(job_registry.turn_budget())
            job_registry.bind_turn_job(job)
            handle = job_registry.turn_budget()
            self.assertIsNotNone(handle)
            self.assertAlmostEqual(handle.remaining(), job.remaining(), delta=0.5)
            job.cancel("stopped by user")
            self.assertTrue(job_registry.turn_budget().cancelled())
        finally:
            job_registry.unbind_turn_job(None)
            job.finish()


class FailureClassifierTests(unittest.TestCase):
    """One classifier decides replay eligibility for every transport."""

    def test_auth_failures_are_terminal(self):
        for status in (401, 403):
            failure = budget.classify_status(status, "invalid key")
            self.assertEqual(failure.kind, budget.AUTH)
            self.assertFalse(failure.retryable)
            self.assertTrue(failure.terminal)

    def test_validation_and_not_found_are_terminal(self):
        for status in (400, 404, 410, 422):
            failure = budget.classify_status(status)
            self.assertFalse(failure.retryable, status)
            self.assertTrue(failure.terminal, status)

    def test_transient_statuses_are_replayable(self):
        for status in (408, 429, 500, 502, 503, 504):
            failure = budget.classify_status(status)
            self.assertTrue(failure.retryable, status)

    def test_transport_errors_are_classified(self):
        self.assertEqual(
            budget.classify_exception(requests.Timeout()).kind, budget.TIMEOUT)
        self.assertEqual(
            budget.classify_exception(requests.ConnectionError()).kind,
            budget.CONNECTION)
        self.assertEqual(
            budget.classify_exception(ValueError("nope")).kind, budget.UNKNOWN)
        self.assertFalse(
            budget.classify_exception(ValueError("nope")).retryable)

    def test_a_replayable_failure_is_refused_when_the_budget_is_gone(self):
        failure = budget.classify_status(503)
        self.assertTrue(budget.retry_eligible(failure, None, 1, 2))
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        self.assertFalse(budget.retry_eligible(failure, spent, 1, 2))
        live = budget.Deadline.after(5)
        self.assertTrue(budget.retry_eligible(failure, live, 1, 2))
        # ... but attempts still run out.
        self.assertFalse(budget.retry_eligible(failure, live, 2, 2))


# ── gemini ─────────────────────────────────────────────────────────────────
class GeminiBudgetTests(unittest.TestCase):
    def setUp(self):
        self._key = patch.object(gemini_client, "GEMINI_API_KEY", "test-key")
        self._key.start()
        self.addCleanup(self._key.stop)

    def test_chat_sends_nothing_with_a_spent_budget(self):
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        with patch.object(gemini_client._session, "post") as post:
            for passed in (spent, time.monotonic() - 1.0):
                self.assertEqual(
                    gemini_client.ask_gemini_chat(
                        [{"role": "user", "content": "hi"}], deadline=passed),
                    {},
                )
        post.assert_not_called()

    def test_vision_sends_nothing_with_a_spent_budget(self):
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        with patch.object(gemini_client._session, "post") as post:
            self.assertEqual(
                gemini_client.ask_gemini_vision(
                    "describe", "data:image/png;base64,abc", deadline=spent),
                {},
            )
        post.assert_not_called()

    def test_stream_sends_nothing_with_a_spent_budget(self):
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        with patch.object(gemini_client._session, "post") as post:
            self.assertEqual(
                list(gemini_client.ask_gemini_chat_stream(
                    [{"role": "user", "content": "hi"}], deadline=spent)),
                [],
            )
        post.assert_not_called()

    def test_an_ambient_budget_is_inherited(self):
        """A transport given no deadline spends the turn's budget, not a new one."""
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        with patch.object(gemini_client._session, "post") as post:
            with budget.bound(spent):
                result = gemini_client.ask_gemini_chat(
                    [{"role": "user", "content": "hi"}])
        post.assert_not_called()
        self.assertEqual(result, {})

    def test_auth_failure_is_not_replayed(self):
        with patch.object(gemini_client._session, "post",
                          return_value=_status_response(403, "forbidden")) as post:
            result = gemini_client.ask_gemini_chat(
                [{"role": "user", "content": "hi"}],
                deadline=budget.Deadline.after(30),
            )
        self.assertEqual(result, {})
        self.assertEqual(post.call_count, 1)

    def test_the_request_timeout_is_sliced_to_the_budget(self):
        with patch.object(gemini_client._session, "post",
                          return_value=_gemini_ok_response("hello")) as post:
            result = gemini_client.ask_gemini_chat(
                [{"role": "user", "content": "hi"}],
                deadline=budget.Deadline.after(0.5),
            )
        self.assertEqual(result["choices"][0]["message"]["content"], "hello")
        connect, read = post.call_args.kwargs["timeout"]
        self.assertLessEqual(connect, 0.5)
        self.assertLessEqual(read, 0.5)

    def test_a_transient_transport_failure_is_replayed_within_the_budget(self):
        with patch.object(gemini_client._session, "post",
                          side_effect=[requests.ConnectionError("reset"),
                                       _gemini_ok_response("recovered")]) as post, \
             patch.object(gemini_client.time, "sleep") as sleeper:
            result = gemini_client.ask_gemini_chat(
                [{"role": "user", "content": "hi"}],
                deadline=budget.Deadline.after(30),
            )
        self.assertEqual(result["choices"][0]["message"]["content"], "recovered")
        self.assertEqual(post.call_count, 2)
        # The wait came out of the budget (Deadline.sleep), not a bare sleep.
        self.assertTrue(sleeper.called)

    def test_hidden_adapter_retry_is_refused_once_the_budget_is_spent(self):
        strategy = gemini_client._retry_strategy
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        with budget.bound(spent):
            with self.assertRaises(MaxRetryError):
                strategy.increment(
                    method="POST", url="https://example.invalid", _pool=None)
        # With a live budget the very same call is an ordinary retry.
        self.assertIsNotNone(
            strategy.increment(
                method="POST", url="https://example.invalid", _pool=None))

    def test_adapter_backoff_cannot_outlive_the_budget(self):
        strategy = gemini_client._retry_strategy
        response = MagicMock()
        response.headers = {"Retry-After": "5"}
        live = budget.Deadline.after(0.1)
        started = time.monotonic()
        with budget.bound(live):
            strategy.sleep(response)
        self.assertLess(time.monotonic() - started, 2.0)


class GeminiStreamBudgetTests(unittest.TestCase):
    def setUp(self):
        self._key = patch.object(gemini_client, "GEMINI_API_KEY", "test-key")
        self._key.start()
        self.addCleanup(self._key.stop)

    @staticmethod
    def _sse_lines(texts):
        return [
            "data: " + json.dumps(
                {"candidates": [{"content": {"parts": [{"text": text}]}}]})
            for text in texts
        ]

    def test_a_trickling_stream_cannot_extend_the_deadline(self):
        """Acceptance: trickling streams stop at the deadline and the text
        already delivered survives as the partial result."""
        clock = FakeClock()
        handle = budget.Deadline.at(clock.now + 25.0)
        lines = self._sse_lines(["one", "two", "three", "four"])
        produced = []

        def _iter_lines(decode_unicode=True):
            for line in lines:
                clock.advance(10.0)
                produced.append(line)
                yield line

        response = MagicMock()
        response.status_code = 200
        response.iter_lines.return_value = _iter_lines()
        with patch.object(budget.time, "monotonic", clock), \
             patch.object(gemini_client._session, "post", return_value=response):
            out = list(gemini_client.ask_gemini_chat_stream(
                [{"role": "user", "content": "hi"}], deadline=handle))

        self.assertEqual(out, ["one", "two"])
        # The stream stopped between chunks instead of draining the trickle.
        self.assertEqual(len(produced), 3)


# ── fireworks ──────────────────────────────────────────────────────────────
class FireworksBudgetTests(unittest.TestCase):
    MESSAGES = [{"role": "user", "content": "hello"}]

    def setUp(self):
        self._key = patch.object(fireworks_client, "API_KEY", "test-key")
        self._effort = patch.object(fireworks_client, "REASONING_EFFORT", "none")
        self._post = patch.object(fireworks_client._session, "post")
        self._key.start()
        self._effort.start()
        self.post = self._post.start()
        self.addCleanup(self._key.stop)
        self.addCleanup(self._effort.stop)
        self.addCleanup(self._post.stop)
        self.sent = []

    def _respond(self, *responses):
        def _post(url, **kwargs):
            self.sent.append(dict(kwargs.get("json") or {}))
            return responses[len(self.sent) - 1]

        self.post.side_effect = _post

    def _sse_response(self, lines):
        resp = MagicMock()
        resp.status_code = 200
        resp.iter_lines.return_value = lines
        return resp

    def _sse_lines(self, texts):
        lines = [
            "data: " + json.dumps({"choices": [{"delta": {"content": text}}]})
            for text in texts
        ]
        lines.append("data: [DONE]")
        return lines

    def test_chat_sends_nothing_with_a_spent_budget(self):
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        result = fireworks_client.ask_fireworks(self.MESSAGES, deadline=spent)
        self.post.assert_not_called()
        self.assertEqual(result["failure"].kind, budget.BUDGET)
        self.assertTrue(result["failure"].terminal)

    def test_vision_sends_nothing_with_a_spent_budget(self):
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        result = fireworks_client.ask_fireworks_vision(
            "describe", "data:image/png;base64,abc", deadline=spent)
        self.assertEqual(result, {})
        self.post.assert_not_called()

    def test_stream_sends_nothing_with_a_spent_budget(self):
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        out = list(fireworks_client.ask_fireworks_stream(
            self.MESSAGES, deadline=spent))
        self.assertEqual(out, [])
        self.post.assert_not_called()

    def test_a_cancelled_job_token_is_a_spent_budget(self):
        job = job_registry.new_job(kind="request", timeout=30.0)
        try:
            job.cancel("stopped by user")
            result = fireworks_client.ask_fireworks(self.MESSAGES, deadline=job)
        finally:
            job.finish()
        self.post.assert_not_called()
        self.assertEqual(result["failure"].kind, budget.BUDGET)

    def test_auth_failure_is_not_replayed(self):
        self._respond(_status_response(401, "invalid api key"))
        result = fireworks_client.ask_fireworks(
            self.MESSAGES, deadline=budget.Deadline.after(30))
        self.assertEqual(result["error"], 401)
        self.assertEqual(result["failure"].kind, budget.AUTH)
        self.assertTrue(result["failure"].terminal)
        self.assertEqual(self.post.call_count, 1)

    def test_transient_failure_is_replayed_while_budget_remains(self):
        self._respond(_status_response(503, "unavailable"),
                      _ok_response("recovered"))
        result = fireworks_client.ask_fireworks(
            self.MESSAGES, deadline=budget.Deadline.after(30))
        self.assertEqual(result["choices"][0]["message"]["content"], "recovered")
        self.assertEqual(self.post.call_count, 2)

    def test_a_replay_is_refused_once_the_budget_is_gone(self):
        clock = FakeClock()

        def _slow_503(url, **kwargs):
            clock.advance(30.0)          # the request ate the whole window
            return _status_response(503, "unavailable")

        with patch.object(budget.time, "monotonic", clock):
            handle = budget.Deadline.at(clock.now + 10.0)
            self.post.side_effect = _slow_503
            result = fireworks_client.ask_fireworks(
                self.MESSAGES, deadline=handle)

        self.assertEqual(self.post.call_count, 1)
        self.assertEqual(result["error"], 503)

    def test_the_request_timeout_is_sliced_to_the_budget(self):
        self._respond(_ok_response("hi"))
        fireworks_client.ask_fireworks(
            self.MESSAGES, deadline=budget.Deadline.after(0.4))
        connect, read = self.post.call_args.kwargs["timeout"]
        self.assertLessEqual(connect, 0.4)
        self.assertLessEqual(read, 0.4)

    def test_a_trickling_stream_cannot_extend_the_deadline(self):
        """Acceptance: the trickle stops at the deadline and the text already
        yielded survives as the partial result."""
        clock = FakeClock()
        handle = budget.Deadline.at(clock.now + 25.0)
        lines = self._sse_lines(["one", "two", "three", "four"])
        produced = []

        def _iter_lines(decode_unicode=True):
            for line in lines:
                clock.advance(10.0)
                produced.append(line)
                yield line

        response = MagicMock()
        response.status_code = 200
        response.iter_lines.return_value = _iter_lines()
        self._respond(response)
        with patch.object(budget.time, "monotonic", clock):
            out = list(fireworks_client.ask_fireworks_stream(
                self.MESSAGES, deadline=handle))

        self.assertEqual(out, ["one", "two"])
        self.assertEqual(len(produced), 3)


# ── executor ───────────────────────────────────────────────────────────────
class ExecutorBudgetTests(unittest.TestCase):
    def test_the_youtube_lookup_sends_no_request_on_a_spent_budget(self):
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        with patch.object(executor.requests, "get") as get:
            self.assertIsNone(executor.get_first_youtube_video("lofi", deadline=spent))
        get.assert_not_called()

    def test_the_youtube_lookup_is_bounded_and_sliced(self):
        body = MagicMock()
        body.text = '<a href="/watch?v=abc123">x</a>'
        with patch.object(executor.requests, "get", return_value=body) as get:
            url = executor.get_first_youtube_video("lofi")
        self.assertEqual(url, "https://www.youtube.com/watch?v=abc123")
        self.assertLessEqual(get.call_args.kwargs["timeout"],
                             executor.YOUTUBE_LOOKUP_TIMEOUT)
        self.assertGreater(get.call_args.kwargs["timeout"], 0)

        with patch.object(executor.requests, "get", return_value=body) as get:
            executor.get_first_youtube_video(
                "lofi", deadline=budget.Deadline.after(0.4))
        self.assertLessEqual(get.call_args.kwargs["timeout"], 0.4)

    def test_a_spent_budget_runs_no_action_of_a_batch(self):
        spent = budget.Deadline.at(time.monotonic() - 1.0)
        actions = [
            {"action": "search", "input": "one"},
            {"action": "search", "input": "two"},
        ]
        with patch.object(executor, "execute_action") as run:
            results = executor.execute_multiple(actions, deadline=spent)
        self.assertEqual(results, [False, False])
        run.assert_not_called()

    def test_a_batch_wait_grants_no_fresh_time(self):
        clock = FakeClock()

        def _slow_action(action, input_value, browser=None, deadline=None):
            clock.advance(5.0)
            return True

        actions = [
            {"action": "search", "input": "one"},
            {"action": "search", "input": "two"},
        ]
        with patch.object(budget.time, "monotonic", clock), \
             patch.object(executor, "execute_action", side_effect=_slow_action) as run, \
             patch.object(executor.time, "sleep") as sleeper:
            handle = budget.Deadline.at(clock.now + 1.0)
            results = executor.execute_multiple(actions, deadline=handle)

        self.assertEqual(results, [True, False])
        self.assertEqual(run.call_count, 1)
        # The pause was taken from the budget, and the spent budget did not
        # hand out a fresh second to the next action.
        sleeper.assert_not_called()


if __name__ == "__main__":
    unittest.main()
