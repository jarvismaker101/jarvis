"""F35 — one exact, target-aware multilingual control grammar.

Acceptance (audit report): "Test both apostrophe forms, Hindi negation,
other-app closing, backend speech, tasks, and prefetch waits; every control
affects only its intended owner."

Baseline defects pinned here:
  * Hindi shutdown was SUBSTRING based, so "music band karo" (stop the music)
    shut the whole stack down;
  * negation/apostrophes were handled only for the research phrases, so
    "don't continue" still resumed the narration;
  * pause had no handler at all (it only appeared in a "never shutdown" list);
  * "stop speaking" was only honoured while THIS process believed it was
    speaking, so it did nothing against backend narration;
  * shutdown sent an UNAUTHENTICATED warm-stop to the supervisor and called
    the stack down when the backend was still running.
"""

import os
import threading
import unittest
from unittest.mock import MagicMock, patch

from backend import voice_mode as vm
from backend.services import voice as voice_service


class GrammarTests(unittest.TestCase):
    def test_the_control_table_is_target_aware(self):
        cases = (
            # (utterance, expected control)
            ("stop speaking", "speech_stop"),
            ("jarvis, stop speaking", "speech_stop"),
            ("stop talking", "speech_stop"),
            ("be quiet", "speech_stop"),
            ("shut up", "speech_stop"),
            ("bolna band karo", "speech_stop"),
            ("chup ho jao", "speech_stop"),
            ("bas karo", "speech_stop"),
            ("stop task", "task_stop"),
            ("cancel the task", "task_stop"),
            ("kaam band karo", "task_stop"),
            ("cancel approval", "approval_cancel"),
            ("don't do that", "approval_cancel"),
            ("pause", "pause"),
            ("jarvis pause", "pause"),
            ("ruk jao", "pause"),
            ("continue", "continue"),
            ("continue karo", "continue"),
            ("sleep", "sleep"),
            ("so jao", "sleep"),
            ("shutdown", "shutdown"),
            ("shut down", "shutdown"),
            ("jarvis stop listening", "shutdown"),
            ("band ho jao", "shutdown"),
            ("band karo", "shutdown"),
            ("sab band karo", "shutdown"),
        )
        for utterance, expected in cases:
            self.assertEqual(vm.classify_control(utterance), expected, utterance)

    def test_other_app_closing_is_not_shutdown(self):
        """Closing another app (or the music) must never take Jarvis down."""
        for utterance in (
            "music band karo", "gaana band karo", "chrome band karo",
            "close chrome", "close whatsapp", "band karo chrome",
            "whatsapp band kar do", "turn off the music",
            "stop the music", "shut the window", "close the window",
        ):
            self.assertIsNone(vm.classify_control(utterance), utterance)
            self.assertFalse(vm.is_shutdown(utterance), utterance)

    def test_apostrophe_forms_are_equivalent(self):
        for utterance in ("don't continue", "dont continue", "do not continue",
                          "never continue", "don't stop the research",
                          "dont stop the research", "do not stop the search"):
            self.assertIsNone(vm.classify_control(utterance), utterance)
            self.assertFalse(vm.is_continue(utterance), utterance)

    def test_hindi_negation_blocks_the_control(self):
        for utterance in ("mat band karo", "band mat karo", "nahi band karo",
                          "band nahi karo"):
            self.assertFalse(vm.is_shutdown(utterance), utterance)
            self.assertIsNone(vm.classify_control(utterance), utterance)

    def test_hindi_stop_talking_is_not_shutdown(self):
        for utterance in ("chup ho jao", "jarvis chup ho jao", "bolna band karo",
                          "bolna band karo jarvis"):
            self.assertFalse(vm.is_shutdown(utterance), utterance)
            self.assertEqual(vm.classify_control(utterance), "speech_stop",
                             utterance)

    def test_bare_stop_and_empty_are_not_controls(self):
        for utterance in ("", "   ", "stop", "jarvis stop", "and", "the"):
            self.assertIsNone(vm.classify_control(utterance), utterance)
            self.assertFalse(vm.is_shutdown(utterance), utterance)

    def test_research_stop_keeps_its_negation_and_tail_tolerance(self):
        self.assertTrue(vm.is_stop_research("stop the research"))
        self.assertTrue(vm.is_stop_research("stop the deepsearch now"))
        self.assertTrue(vm.is_stop_research("stop the research sir"))
        self.assertTrue(vm.is_stop_research("hey jarvis stop the search"))
        self.assertFalse(vm.is_stop_research("don't stop the research"))
        self.assertFalse(vm.is_stop_research("dont stop the research"))
        self.assertFalse(vm.is_stop_research("what is today's weather"))

    def test_every_control_has_exactly_one_owner(self):
        owners = {
            "speech_stop": "speech",
            "pause": "speech",
            "continue": "speech",
            "task_stop": "task",
            "approval_cancel": "approval",
            "sleep": "supervisor",
            "shutdown": "supervisor",
        }
        for command, owner in owners.items():
            self.assertEqual(vm.control_owner(command), owner, command)
        self.assertIsNone(vm.control_owner("chat"))


class DeliveryTests(unittest.TestCase):
    """Every control affects only its intended owner."""

    def test_dispatch_routes_to_one_delivery_only(self):
        deliveries = (
            "_deliver_speech_stop", "_deliver_pause", "_deliver_continue",
            "_deliver_stop_task", "_deliver_cancel_approval",
            "_deliver_sleep", "_deliver_shutdown",
        )
        expected = {
            "speech_stop": "_deliver_speech_stop",
            "pause": "_deliver_pause",
            "continue": "_deliver_continue",
            "task_stop": "_deliver_stop_task",
            "approval_cancel": "_deliver_cancel_approval",
            "sleep": "_deliver_sleep",
            "shutdown": "_deliver_shutdown",
        }
        for command, wanted in expected.items():
            patches = [patch.object(vm, name, return_value=True)
                       for name in deliveries]
            mocks = {}
            for name, item in zip(deliveries, patches):
                mocks[name] = item.start()
                self.addCleanup(item.stop)
            try:
                self.assertTrue(vm.dispatch_control(command), command)
            finally:
                for item in patches:
                    item.stop()
            self.assertEqual(mocks[wanted].call_count, 1, command)
            for name in deliveries:
                if name != wanted:
                    self.assertEqual(mocks[name].call_count, 0,
                                     f"{command} touched {name}")

    def test_speech_stop_reaches_the_backend_regardless_of_local_state(self):
        """The backend narrates; this process's own flag is not the truth."""
        posted = []

        def fake_post(path, payload, timeout=2.5):
            posted.append(path)
            return True, {}

        with patch.object(vm, "listener_state") as state, \
             patch.object(vm, "stop_speaking") as local_stop, \
             patch.object(vm, "_post_backend", side_effect=fake_post):
            state.is_speaking.return_value = False
            self.assertTrue(vm.dispatch_control("speech_stop"))
        self.assertIn("/speak/stop", posted,
                      "a stop must not depend on this process's speaking flag")
        local_stop.assert_called()

    def test_a_broken_local_stop_never_blocks_the_control(self):
        posted = []

        def fake_post(path, payload, timeout=2.5):
            posted.append(path)
            return True, {}

        with patch.object(vm, "stop_speaking", side_effect=RuntimeError("busy")), \
             patch.object(vm, "_post_backend", side_effect=fake_post):
            self.assertTrue(vm.dispatch_control("speech_stop"))
        self.assertIn("/speak/stop", posted)

    def test_prefetch_waits_do_not_delay_a_control(self):
        """A control delivered while playback/prefetch is in flight."""
        started = threading.Event()
        release = threading.Event()

        def slow_prefetch(*args, **kwargs):
            started.set()
            release.wait(5)
            return True

        posted = []

        def fake_post(path, payload, timeout=2.5):
            posted.append(path)
            return True, {}

        from backend.services import fish_voice

        thread = threading.Thread(target=slow_prefetch, args=("hi",), daemon=True)
        with patch.object(fish_voice, "prefetch_fish_audio",
                          side_effect=slow_prefetch), \
             patch.object(vm, "stop_speaking"), \
             patch.object(vm, "_post_backend", side_effect=fake_post):
            thread.start()
            self.assertTrue(started.wait(2))
            done = threading.Event()

            def deliver():
                vm.dispatch_control("speech_stop")
                done.set()

            worker = threading.Thread(target=deliver, daemon=True)
            worker.start()
            self.assertTrue(done.wait(2),
                            "a control must not wait for a TTS prefetch")
        release.set()
        thread.join(5)
        self.assertIn("/speak/stop", posted)

    def test_task_stop_targets_the_task_owner_only(self):
        posted = []

        def fake_post(path, payload, timeout=2.5):
            posted.append(path)
            return True, {}

        with patch.object(vm, "_post_backend", side_effect=fake_post), \
             patch.object(vm, "stop_speaking"):
            vm.dispatch_control("task_stop")
        self.assertEqual(posted, ["/task/stop"])

    def test_approval_cancel_targets_the_approval_owner_only(self):
        posted = []

        def fake_post(path, payload, timeout=2.5):
            posted.append(path)
            return True, {"dropped": True}

        with patch.object(vm, "_post_backend", side_effect=fake_post), \
             patch.object(vm, "stop_speaking"):
            vm.dispatch_control("approval_cancel")
        self.assertEqual(posted, ["/approvals/reset"])

    def test_pause_is_local_when_this_process_owns_the_audio(self):
        """[P1-07] A VOICE reply's audio lives HERE, so the pause happens here.

        The old assertion — "pause asks the backend to keep the remainder" with
        the local ``stop_speaking`` patched out — pinned the defect: for a voice
        turn the backend's active stream is EMPTY, so the backend pause resumed
        nothing while the real audio had already been stopped locally. Pause is
        a LOCAL, non-destructive control now; the backend is only used when
        nothing was playing here (see the next test).
        """
        posted = []

        def fake_post(path, payload, timeout=2.5):
            posted.append(path)
            return True, {"resumable": True, "remaining_chars": 42}

        with patch.object(vm, "pause_speaking", return_value=True) as pause, \
             patch.object(vm, "_post_backend", side_effect=fake_post), \
             patch.object(vm, "stop_speaking") as stop:
            self.assertTrue(vm.dispatch_control("pause"))
        pause.assert_called_once_with()
        stop.assert_not_called()
        self.assertEqual(posted, [])

    def test_pause_falls_back_to_the_backend_owner(self):
        """The BACKEND narrates typed-UI replies, so its pause still matters."""
        posted = []

        def fake_post(path, payload, timeout=2.5):
            posted.append(path)
            return True, {"resumable": True, "remaining_chars": 42}

        with patch.object(vm, "pause_speaking", return_value=False), \
             patch.object(vm, "_post_backend", side_effect=fake_post), \
             patch.object(vm, "stop_speaking"):
            self.assertTrue(vm.dispatch_control("pause"))
        self.assertEqual(posted, ["/speak/pause"])

    def test_continue_targets_the_paused_speech(self):
        posted = []

        def fake_post(path, payload, timeout=2.5):
            posted.append(path)
            return True, {"resumed": True}

        with patch.object(vm, "resume_local_playback", return_value=False), \
             patch.object(vm, "_post_backend", side_effect=fake_post):
            self.assertTrue(vm.dispatch_control("continue"))
        self.assertEqual(posted, ["/speak/resume"])

    def test_continue_never_plays_a_local_remainder(self):
        """F50 single-owner rule: only the backend may resume playback.

        The old assertion pinned a SECOND playback owner here (a local
        ``pop_remaining()`` + ``speak``), which F50 forbids: with the backend
        reporting nothing to resume, no audio may be played from this process.
        """
        with patch.object(vm, "_post_backend", return_value=(True, {})), \
             patch.object(vm.listener_state, "has_remaining",
                          return_value=True), \
             patch.object(vm.listener_state, "pop_remaining",
                          return_value="the rest of the answer") as popped, \
             patch.object(vm, "speak") as fake_speak:
            self.assertFalse(vm.dispatch_control("continue"))
        fake_speak.assert_not_called()
        popped.assert_not_called()


class SupervisorControlTests(unittest.TestCase):
    """Sleep and shutdown are distinct, authenticated supervisor actions."""

    def test_sleep_is_the_warm_sleep_endpoint(self):
        seen = {}

        def fake_urlopen(request, timeout=None):
            seen["url"] = request.full_url
            seen["token"] = request.headers.get("X-jarvis-token") or \
                request.headers.get("x-jarvis-token")
            response = MagicMock()
            response.read.return_value = b'{"ok": true, "mode": "warm-sleep"}'
            response.__enter__ = MagicMock(return_value=response)
            response.__exit__ = MagicMock(return_value=False)
            return response

        with patch.dict(os.environ, {"JARVIS_WATCHER_CONTROL_PORT": "8766",
                                     "JARVIS_LOCAL_TOKEN": "tok-123"}), \
             patch.object(vm, "urlopen", side_effect=fake_urlopen):
            self.assertTrue(vm.dispatch_control("sleep"))
        self.assertIn("/stop", seen["url"])
        self.assertNotIn("/shutdown", seen["url"])
        self.assertEqual(seen["token"], "tok-123",
                         "watcher control must carry the per-launch token")

    def test_shutdown_is_the_full_shutdown_endpoint_and_authenticated(self):
        seen = {}

        def fake_urlopen(request, timeout=None):
            seen["url"] = request.full_url
            seen["token"] = request.headers.get("X-jarvis-token") or \
                request.headers.get("x-jarvis-token")
            response = MagicMock()
            response.read.return_value = b'{"ok": true, "mode": "full-shutdown"}'
            response.__enter__ = MagicMock(return_value=response)
            response.__exit__ = MagicMock(return_value=False)
            return response

        with patch.dict(os.environ, {"JARVIS_WATCHER_CONTROL_PORT": "8766",
                                     "JARVIS_LOCAL_TOKEN": "tok-123"}), \
             patch.object(vm, "urlopen", side_effect=fake_urlopen):
            self.assertTrue(vm.dispatch_control("shutdown"))
        self.assertIn("/shutdown", seen["url"])
        self.assertEqual(seen["token"], "tok-123")

    def test_an_unreachable_supervisor_still_shuts_the_children_down(self):
        killed = []
        with patch.dict(os.environ, {"JARVIS_WATCHER_CONTROL_PORT": "8766",
                                     "JARVIS_BACKEND_PID": "111",
                                     "JARVIS_ELECTRON_PID": "222"}), \
             patch.object(vm, "urlopen", side_effect=OSError("down")), \
             patch.object(vm, "_taskkill_pid",
                          side_effect=lambda pid: killed.append(pid) or True), \
             patch.object(vm, "_stop_backend_port_if_jarvis",
                          return_value=True):
            self.assertFalse(vm.dispatch_control("shutdown"))
        self.assertIn("111", killed)
        self.assertIn("222", killed)

    def test_shutdown_everything_requests_a_full_shutdown(self):
        with patch.object(vm, "stop_speaking"), \
             patch.object(vm, "speak"), \
             patch.object(vm.time, "sleep"), \
             patch.object(vm, "_terminate_self") as terminate, \
             patch.object(vm, "_deliver_shutdown", return_value=True) as deliver:
            vm.shutdown_everything()
        deliver.assert_called_once_with()
        terminate.assert_called_once_with()


class PauseResumeTests(unittest.TestCase):
    def tearDown(self):
        voice_service.listener_state.set_remaining("")

    def test_pause_keeps_the_remainder_resumable(self):
        speaker = voice_service.StreamSpeaker()
        speaker._queue.put("second sentence")
        speaker._queue.put("third sentence")
        previous = voice_service.get_active_stream()
        voice_service.set_active_stream(speaker)
        try:
            with patch.object(voice_service, "_speak_chunk",
                              side_effect=lambda *a, **k: True):
                remaining = voice_service.pending_speaking_text()
        finally:
            voice_service.set_active_stream(previous)
            speaker.close()
        self.assertIn("second sentence", remaining)
        self.assertIn("third sentence", remaining)

    def test_pause_then_resume_speaks_the_same_text(self):
        with patch.object(voice_service, "stop_speaking"), \
             patch.object(voice_service, "pending_speaking_text",
                          return_value="the unplayed part"), \
             patch.object(voice_service, "speak") as fake_speak:
            self.assertTrue(voice_service.pause_speaking())
            self.assertEqual(voice_service.listener_state.get_remaining(),
                             "the unplayed part")
            self.assertEqual(voice_service.resume_speaking(),
                             "the unplayed part")
        fake_speak.assert_called_once_with("the unplayed part")
        self.assertEqual(voice_service.listener_state.get_remaining(), "")

    def test_resume_with_nothing_pending_is_honest(self):
        voice_service.listener_state.set_remaining("")
        with patch.object(voice_service, "speak") as fake_speak:
            self.assertEqual(voice_service.resume_speaking(), "")
        fake_speak.assert_not_called()

    def test_the_stream_speaker_snapshot_keeps_the_order(self):
        speaker = voice_service.StreamSpeaker()
        for text in ("one", "two", "three"):
            speaker._queue.put(text)
        snapshot = speaker.pending_text()
        self.assertEqual(snapshot.split(), ["one", "two", "three"])
        # ...and nothing was dropped.
        self.assertEqual(speaker._queue.qsize(), 3)
        speaker.close()


class RouteTests(unittest.TestCase):
    def test_pause_route_reports_resumability(self):
        from backend.api import routes

        with patch.object(voice_service, "pause_speaking", return_value=True), \
             patch.object(routes, "set_narration_enabled"):
            result = routes.pause_speech()
        self.assertTrue(result["ok"])
        self.assertTrue(result["resumable"])

    def test_resume_route_reports_what_it_resumed(self):
        from backend.api import routes

        with patch.object(voice_service, "resume_speaking",
                          return_value="some text"):
            result = routes.resume_speech()
        self.assertTrue(result["resumed"])
        self.assertEqual(result["resumed_chars"], len("some text"))

    def test_remaining_route_exposes_unplayed_speech(self):
        from backend.api import routes

        voice_service.listener_state.set_remaining("left over")
        try:
            result = routes.speech_remaining()
        finally:
            voice_service.listener_state.set_remaining("")
        self.assertTrue(result["has_remaining"])
        self.assertEqual(result["remaining"], "left over")


if __name__ == "__main__":
    unittest.main()
