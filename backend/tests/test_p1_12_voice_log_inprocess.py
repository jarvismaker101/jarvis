"""P1-12 — no unauthenticated self-HTTP call on the turn path.

The brain POSTed to its own ``/update-voice-log`` with no ``X-Jarvis-Token``.
Auth fails closed here, so that request 401'd on EVERY voice turn: a background
thread, a round trip and a misleading error, every time. The backend is already
in the same process as the voice-log state, so the publish is a direct call.

These tests pin: no self-HTTP from a turn, the log still updates (in-process,
once, after the reply is final), the endpoint stays token-protected for
external callers, and a voice-log failure can never touch the reply.
"""

import threading
import unittest
from unittest.mock import patch

from backend.api import routes
from backend.core import brain
from backend.services import local_auth, request_registry


class SyncVoiceLogTests(unittest.TestCase):
    """The brain's publisher must not open a socket any more."""

    def tearDown(self):
        brain.register_voice_log_sink(routes._publish_voice_log)

    def test_publishing_makes_no_http_request(self):
        with patch("backend.core.brain.requests.post") as post, \
             patch("backend.core.brain.threading.Thread") as thread:
            brain.sync_voice_log("hello", "hi sir")

        post.assert_not_called()
        thread.assert_not_called()

    def test_publishing_reaches_the_registered_sink_directly(self):
        seen = []
        brain.register_voice_log_sink(lambda m, r: seen.append((m, r)))

        self.assertTrue(brain.sync_voice_log("hello", "hi sir"))

        self.assertEqual(seen, [("hello", "hi sir")])

    def test_the_api_layer_registers_its_publisher(self):
        """The sink is the route module's own state, not a copy of it."""
        brain.register_voice_log_sink(routes._publish_voice_log)
        before = routes.last_voice_log_id

        self.assertTrue(brain.sync_voice_log("probe", "reply"))

        self.assertEqual(routes.last_voice_log_id, before + 1)
        self.assertEqual(routes.last_voice_message, "probe")
        self.assertEqual(routes.last_voice_response, "reply")

    def test_a_sink_failure_can_never_affect_the_turn(self):
        def exploding(_message, _response):
            raise RuntimeError("voice log is on fire")

        brain.register_voice_log_sink(exploding)

        self.assertFalse(brain.sync_voice_log("hello", "hi"))

    def test_no_sink_is_a_silent_no_op(self):
        """The voice I/O worker imports this module but owns no route state."""
        brain.register_voice_log_sink(None)

        self.assertFalse(brain.sync_voice_log("hello", "hi"))


class _WorkerTestCase(unittest.TestCase):
    """Drives the real worker once, with the brain faked at its boundary."""

    def setUp(self):
        self.saved = (routes.last_voice_message, routes.last_voice_response,
                      routes.last_voice_log_id)
        self.states = []

    def tearDown(self):
        (routes.last_voice_message, routes.last_voice_response,
         routes.last_voice_log_id) = self.saved
        for state in self.states:
            try:
                if not state.done:
                    state.complete("")
            except Exception:
                pass

    def _run_worker(self, reply="On it, sir.", from_voice=True,
                    message="open brave"):
        state = request_registry.REGISTRY.admit("", message)[0]
        self.states.append(state)
        with patch.object(routes, "process_message", return_value=reply), \
             patch.object(routes, "stop_speaking"), \
             patch.object(routes, "set_narration_enabled"):
            routes._run_request_worker(state, from_voice=from_voice)
        return state


class TurnPathTests(_WorkerTestCase):
    """Acceptance: a normal chat turn makes NO self-HTTP request at all."""

    def test_a_voice_turn_makes_no_self_http_request(self):
        with patch("backend.core.brain.requests.post") as brain_post, \
             patch.object(brain, "threading") as brain_threading:
            self._run_worker(from_voice=True)

        brain_post.assert_not_called()
        brain_threading.Thread.assert_not_called()

    def test_the_worker_asks_the_brain_not_to_publish(self):
        """sync_voice=False is the contract that removes the old self-call."""
        state = request_registry.REGISTRY.admit("", "typed")[0]
        self.states.append(state)
        with patch.object(routes, "process_message",
                          return_value="ok") as call, \
             patch.object(routes, "stop_speaking"), \
             patch.object(routes, "set_narration_enabled"):
            routes._run_request_worker(state, from_voice=True)

        self.assertIs(call.call_args.kwargs["sync_voice"], False)

    def test_the_voice_log_is_updated_in_process_after_the_reply(self):
        self._run_worker(reply="Brave is open, sir.", from_voice=True,
                         message="open brave")

        self.assertEqual(routes.last_voice_response, "Brave is open, sir.")
        self.assertEqual(routes.last_voice_message, "open brave")

    def test_the_log_is_written_once_per_turn(self):
        with patch.object(routes, "_publish_voice_log") as publish:
            self._run_worker(from_voice=True)

        publish.assert_called_once()

    def test_the_command_prefix_is_stripped_from_the_log(self):
        self._run_worker(from_voice=True, message="command open brave")

        self.assertEqual(routes.last_voice_message, "open brave")

    def test_a_typed_turn_does_not_touch_the_voice_log(self):
        before = (routes.last_voice_message, routes.last_voice_response,
                  routes.last_voice_log_id)

        with patch.object(routes, "_publish_voice_log") as publish:
            self._run_worker(from_voice=False, message="a typed question")

        publish.assert_not_called()
        self.assertEqual(
            (routes.last_voice_message, routes.last_voice_response,
             routes.last_voice_log_id), before)

    def test_the_publish_happens_after_the_reply_is_final(self):
        """The log must mirror what the user actually got, not an early draft."""
        order = []
        state = request_registry.REGISTRY.admit("", "open brave")[0]
        self.states.append(state)

        def fake_process(*_args, **_kwargs):
            order.append("process")
            return "final reply"

        def fake_publish(message, response):
            order.append(("publish", message, response))

        with patch.object(routes, "process_message", side_effect=fake_process), \
             patch.object(routes, "_publish_voice_log", side_effect=fake_publish), \
             patch.object(routes, "stop_speaking"), \
             patch.object(routes, "set_narration_enabled"):
            routes._run_request_worker(state, from_voice=True)

        self.assertEqual(order[0], "process")
        self.assertEqual(order[1], ("publish", "open brave", "final reply"))

    def test_a_voice_log_failure_does_not_affect_the_reply(self):
        state = request_registry.REGISTRY.admit("", "hi")[0]
        self.states.append(state)

        def exploding(_message, _response):
            raise RuntimeError("bookkeeping is on fire")

        with patch.object(routes, "process_message",
                          return_value="The answer."), \
             patch.object(routes, "_publish_voice_log", side_effect=exploding), \
             patch.object(routes, "stop_speaking"), \
             patch.object(routes, "set_narration_enabled"):
            routes._run_request_worker(state, from_voice=True)

        self.assertTrue(state.done)
        self.assertEqual(state.reply, "The answer.")


class EndpointTests(unittest.TestCase):
    """Acceptance: /update-voice-log stays available, and stays protected."""

    def setUp(self):
        self.token = local_auth.mint_token()
        local_auth.configure(self.token)
        self.addCleanup(self._disarm)
        self.saved = (routes.last_voice_message, routes.last_voice_response,
                      routes.last_voice_log_id)

    def _disarm(self):
        local_auth.configure("")
        import os
        os.environ.pop("JARVIS_LOCAL_TOKEN", None)
        os.environ.pop("JARVIS_DEV_MODE", None)
        (routes.last_voice_message, routes.last_voice_response,
         routes.last_voice_log_id) = self.saved

    def _client(self):
        from fastapi.testclient import TestClient
        from backend.main import app

        return TestClient(app)

    def test_the_endpoint_still_requires_the_token(self):
        import os
        os.environ.pop("JARVIS_DEV_MODE", None)
        client = self._client()

        response = client.post("/update-voice-log",
                               json={"message": "hi", "response": "hello"})

        self.assertEqual(response.status_code, 401)

    def test_the_endpoint_still_works_for_an_external_caller(self):
        client = self._client()

        response = client.post(
            "/update-voice-log",
            json={"message": "external", "response": "caller"},
            headers={local_auth.HEADER: self.token})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(routes.last_voice_message, "external")
        self.assertEqual(routes.last_voice_response, "caller")

    def test_the_ui_mirror_still_reads_the_published_log(self):
        client = self._client()
        client.post("/update-voice-log",
                    json={"message": "q", "response": "a"},
                    headers={local_auth.HEADER: self.token})

        body = client.get("/voice-log",
                          headers={local_auth.HEADER: self.token}).json()

        self.assertEqual(body["message"], "q")
        self.assertEqual(body["response"], "a")
        self.assertEqual(body["id"], routes.last_voice_log_id)


if __name__ == "__main__":
    unittest.main()
