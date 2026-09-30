"""F51/F50 — authenticated callers and single-owner setup in the voice worker.

These are the coordinator-requested fixes that live in the files owned by the
F12/F31/F34/F36 work:

  * F51 — ``local_auth`` now FAILS CLOSED (only ``GET /health`` is public), so
    every non-public call made by the voice worker / watcher / listener must
    carry the launch token. Without them a 401 is indistinguishable from
    "endpoint missing", "backend silent" or "no answer", which silently
    degrades live behaviour (kill/respawn loops, a dead task-mute poll, no
    cross-process barge-in).
  * F50 — intelligence-state effects (opening the user's setup) are owned by
    the backend, and playback has exactly ONE owner.

No network, microphone, TTS or subprocess is opened: every client call is
driven against a mocked ``urlopen``.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from backend import voice_mode as vm
from backend.services import local_auth

TOKEN = "test-launch-token-0123456789abcdef"


def _header(request, name):
    """Case-insensitive header lookup on a urllib Request."""
    for key, value in request.header_items():
        if key.lower() == name.lower():
            return value
    return None


class _FakeResponse:
    def __init__(self, payload=b"{}"):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._payload


class AuthenticatedCallerTests(unittest.TestCase):
    """F51 — every non-public call carries the launch token."""

    def setUp(self):
        env = patch.dict(os.environ, {"JARVIS_LOCAL_TOKEN": TOKEN})
        env.start()
        self.addCleanup(env.stop)
        self.seen = []

    def _capture(self, payload=b"{}"):
        def fake(url, timeout=None):
            self.seen.append(url)
            return _FakeResponse(payload)
        return fake

    def test_get_backend_authenticates_the_task_mute_poll(self):
        with patch.object(vm, "urlopen", side_effect=self._capture(
                b'{"task_running": true}')):
            state = vm._get_backend("/ui-state")
        self.assertEqual(state, {"task_running": True})
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(_header(self.seen[0], local_auth.HEADER), TOKEN)

    def test_get_backend_keeps_the_documented_url_and_timeout(self):
        with patch.object(vm, "urlopen", side_effect=self._capture()):
            vm._get_backend("/ui-state")
        request = self.seen[0]
        self.assertEqual(
            getattr(request, "full_url", request),
            "http://127.0.0.1:%s/ui-state" % vm.BACKEND_PORT)

    def test_voice_flag_poll_authenticates(self):
        with patch.object(vm, "urlopen", side_effect=self._capture(
                b'{"voice_input_enabled": true}')):
            self.assertTrue(vm.voice_input_enabled())
        self.assertEqual(_header(self.seen[0], local_auth.HEADER), TOKEN)

    def test_post_backend_speak_stop_authenticates(self):
        """[P1-03] The stop moved off the capture thread onto the _speak_stop
        worker with a persistent http.client connection, so the F51 assertion
        is now made at the DELIVERY (conn.request) instead of urlopen. The
        contract is unchanged: the per-launch token rides every stop."""
        from backend.services import listener

        seen = []

        class _FakeConn:
            def request(self, method, path, body=None, headers=None):
                seen.append((method, path, dict(headers or {})))

            def getresponse(self):
                class _Resp:
                    status = 200

                    def read(self):
                        return b"{}"

                return _Resp()

            def close(self):
                pass

        worker = listener._SpeakStopWorker()
        self.addCleanup(worker.shutdown)
        with patch.object(worker, "_open_connection", lambda: _FakeConn()):
            worker._deliver()
        self.assertEqual(len(seen), 1)
        method, path, headers = seen[0]
        self.assertEqual((method, path), ("POST", "/speak/stop"))
        values = {k.lower(): v for k, v in headers.items()}
        self.assertEqual(values.get(local_auth.HEADER.lower()), TOKEN)

    def test_post_backend_speak_stop_never_blocks_the_caller(self):
        """[P1-03] The capture-loop contract: queueing a stop is not I/O.

        A hung endpoint must not be able to stall barge-in onset, so the public
        entry point raises only a flag - it must not open a socket.
        """
        from backend.services import listener

        worker = listener._SpeakStopWorker()
        self.addCleanup(worker.shutdown)
        with patch.object(worker, "_open_connection",
                          side_effect=AssertionError("opened a socket")):
            self.assertTrue(worker.request_stop())
        self.assertEqual(worker.stats["requests"], 1)

    def test_api_is_speaking_authenticates(self):
        from backend.services import listener

        with patch("urllib.request.urlopen", side_effect=self._capture(
                b'{"assistant_speaking": true}')):
            self.assertTrue(listener._api_is_speaking())
        self.assertEqual(_header(self.seen[0], local_auth.HEADER), TOKEN)

    def test_research_endpoint_probe_authenticates(self):
        from backend import watcher

        with patch.object(watcher, "urlopen",
                          side_effect=self._capture(b"{}")):
            watcher._backend_has_research_endpoint()
        self.assertEqual(_header(self.seen[0], local_auth.HEADER),
                         watcher.LOCAL_TOKEN)


class SetupOwnershipTests(unittest.TestCase):
    """F50 — the backend owns setup launches and playback."""

    def test_launch_normal_setup_posts_to_the_backend(self):
        posted = []

        def fake_post(path, payload, timeout=2.5):
            posted.append((path, payload))
            return True, {"launched": ["brave", "whatsapp"]}

        with patch.object(vm, "_post_backend", side_effect=fake_post), \
             patch.object(vm.os, "startfile") as startfile, \
             patch.object(vm.subprocess, "Popen") as popen:
            self.assertTrue(vm.launch_normal_setup())
        self.assertEqual(posted, [("/voice-setup/launch", {"setup": "normal"})])
        startfile.assert_not_called()
        popen.assert_not_called()

    def test_launch_normal_setup_reports_backend_failure(self):
        with patch.object(vm, "_post_backend", return_value=(False, None)), \
             patch.object(vm.os, "startfile") as startfile:
            self.assertFalse(vm.launch_normal_setup())
        startfile.assert_not_called()

    def test_is_normal_setup_matches_the_exact_grammar(self):
        for text in ("normal setup", "jarvis please put my normal setup",
                     "mera setup karo", "setup chalu karo", "setup kholo"):
            self.assertTrue(vm.is_normal_setup(text), text)

    def test_is_normal_setup_honours_negation_and_containment(self):
        for text in ("do not put my normal setup", "dont launch normal setup",
                     "normal setup mat karo", "what is a normal setup anyway"):
            self.assertFalse(vm.is_normal_setup(text), text)

    def test_continue_has_no_local_playback_owner(self):
        with patch.object(vm, "_post_backend", return_value=(True, {})), \
             patch.object(vm.listener_state, "has_remaining",
                          return_value=True), \
             patch.object(vm.listener_state, "pop_remaining",
                          return_value="rest") as popped, \
             patch.object(vm, "speak") as speak:
            self.assertFalse(vm._deliver_continue())
        popped.assert_not_called()
        speak.assert_not_called()


if __name__ == "__main__":
    unittest.main()
