import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend import config
from backend.core import brain
from backend import voice_mode
from backend.api import routes


def _force_opencode_engine():
    return patch.object(config, "TASK_ENGINE", "opencode")


def _wait_until(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class OpencodeTaskRunningFlagTests(unittest.TestCase):
    """Dual-voice mute: the task-running flag gates Jarvis's own speech."""

    def tearDown(self):
        brain.set_opencode_task_running(False)

    def test_flag_starts_false(self):
        self.assertFalse(brain.opencode_task_in_progress())

    def test_set_get_roundtrip(self):
        brain.set_opencode_task_running(True)
        self.assertTrue(brain.opencode_task_in_progress())
        brain.set_opencode_task_running(False)
        self.assertFalse(brain.opencode_task_in_progress())

    def test_flag_true_while_task_runs(self):
        seen = {}

        def slow_task(_task, **_kwargs):
            seen["flag"] = brain.opencode_task_in_progress()
            return "ok"

        with _force_opencode_engine(), \
             patch.object(brain, "is_opencode_available", return_value=True), \
             patch.object(brain, "run_opencode_task", side_effect=slow_task), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("create folder x", "create folder x")
            # Wait for the worker to fully finish (notify is its last call)
            # so the thread never outlives the patch window.
            self.assertTrue(_wait_until(lambda: notify.called))
        self.assertTrue(seen["flag"])
        self.assertFalse(brain.opencode_task_in_progress())

    def test_flag_reset_in_finally_on_exception(self):
        with _force_opencode_engine(), \
             patch.object(brain, "is_opencode_available", return_value=True), \
             patch.object(brain, "run_opencode_task",
                          side_effect=RuntimeError("boom")), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("boom task", "boom task")
            self.assertTrue(
                _wait_until(lambda: not brain.opencode_task_in_progress())
            )
            self.assertTrue(_wait_until(lambda: notify.called))
        self.assertFalse(brain.opencode_task_in_progress())
        self.assertEqual(notify.call_count, 1)

    def test_start_phrase_spoken_once_right_after_confirmation(self):
        with _force_opencode_engine(), \
             patch.object(brain, "is_opencode_available", return_value=True), \
             patch.object(brain, "run_opencode_task", return_value="ok"), \
             patch.object(brain, "_notify_async_reply") as notify:
            reply = brain._execute_deferred_opencode(
                "create folder x", "create folder x"
            )
            self.assertEqual(reply, "Handing the task to opencode, sir.")
            # Confirming the armed gate resolves to the same start phrase
            # (and spawns the deferred worker — kept inside this patch
            # window so the thread can never reach the real function).
            brain.handle_opencode_task("create folder x", "create folder x")
            resolved = brain._consume_opencode_confirmation("yes go ahead")
            self.assertEqual(resolved, "Handing the task to opencode, sir.")
            self.assertTrue(_wait_until(lambda: notify.call_count == 2))
        brain._pending_opencode_task = None

    def test_completion_speaks_fixed_phrase(self):
        with _force_opencode_engine(), \
             patch.object(brain, "is_opencode_available", return_value=True), \
             patch.object(brain, "run_opencode_task",
                          return_value="Created folder 'badmoss'."), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("create folder", "create folder")
            self.assertTrue(_wait_until(lambda: notify.called))
        text = notify.call_args[0][0]
        spoken = notify.call_args[1]["spoken"]
        self.assertEqual(spoken, brain._voice_clip(text))
        self.assertIn("folder has been created", text.lower())


class VoiceMuteWhileTaskRunningTests(unittest.TestCase):
    """The voice I/O worker must never reach the backend runtime mid-task."""

    def test_utterance_processing_skipped_while_flag_true(self):
        # G11 / F50 — the flag is the backend's published task state, and the
        # utterance is submitted (not executed in-process); both are patched.
        with patch.object(voice_mode, "backend_task_running",
                          return_value=True), \
             patch.object(voice_mode, "_ask_backend") as fake_submit:
            voice_mode._respond_to_utterance("hello there")
        fake_submit.assert_not_called()

    def test_utterance_processing_runs_while_flag_false(self):
        stream = SimpleNamespace(
            feed=lambda t: None, spoken_any=False,
            finish=lambda: None, close=lambda: None,
        )
        with patch.object(voice_mode, "backend_task_running",
                          return_value=False), \
             patch.object(voice_mode, "_ask_backend",
                          return_value="hi sir") as fake_submit, \
             patch.object(voice_mode, "StreamSpeaker", return_value=stream), \
             patch.object(voice_mode, "speak") as fake_speak:
            voice_mode._respond_to_utterance("hello there")
            # [P0-08] The dispatch returns as soon as the turn is registered —
            # the reply is produced on the turn's own worker thread (the
            # audit's requirement 3), so its effects are awaited INSIDE this
            # patch window. Outside it the worker would reach the real `speak`
            # and block on real TTS.
            self.assertTrue(_wait_until(lambda: fake_submit.called),
                            "the utterance was never submitted")
            self.assertTrue(_wait_until(lambda: fake_speak.called),
                            "the reply was never spoken")
        fake_submit.assert_called_once()
        fake_speak.assert_called_once_with("hi sir")


class AsyncReplyMuteTests(unittest.TestCase):
    """Async reply callbacks must never speak mid-task; the start phrase
    must never be swallowed by the mute."""

    def tearDown(self):
        brain.set_opencode_task_running(False)

    def test_routes_async_reply_suppressed_while_flag_true(self):
        with patch.object(routes, "opencode_task_in_progress",
                          return_value=True), \
             patch.object(routes, "_publish_voice_log") as log, \
             patch.object(routes, "speak") as spk:
            routes._publish_async_opencode_reply(
                "Sir, the folder has been created.",
                spoken="Sir, the task has been completed.",
            )
        spk.assert_not_called()
        log.assert_called_once()

    def test_routes_async_reply_speaks_after_flag_false(self):
        with patch.object(routes, "opencode_task_in_progress",
                          return_value=False), \
             patch.object(routes, "_publish_voice_log"), \
             patch.object(routes, "speak") as spk:
            routes._publish_async_opencode_reply(
                "Sir, the folder has been created.",
                spoken="Sir, the task has been completed.",
            )
            self.assertTrue(_wait_until(lambda: spk.called))
        self.assertEqual(spk.call_args[0][0],
                         "Sir, the task has been completed.")

    def test_voice_mode_async_reply_callbacks_removed(self):
        """G11 / F50 — async replies are spoken by the BACKEND (routes), not
        by dead module-copy callbacks in the I/O worker. The voice process
        no longer carries _on_async_reply or a local task flag at all."""
        self.assertFalse(hasattr(voice_mode, "_on_async_reply"))
        self.assertFalse(hasattr(voice_mode, "opencode_task_in_progress"))

    def test_voice_mode_deliver_stop_research_cuts_task_and_narration(self):
        """F50 — the voice worker's stop-research control goes to the one
        backend over the authed control plane (/task/stop + /speak/stop)."""
        with patch.object(voice_mode, "stop_speaking") as mock_stop, \
             patch.object(voice_mode, "_post_backend") as mock_post:
            voice_mode._deliver_stop_research()
        mock_stop.assert_called_once()
        self.assertEqual(mock_post.call_count, 2)
        paths = sorted(c.args[0] for c in mock_post.call_args_list)
        self.assertEqual(paths, ["/speak/stop", "/task/stop"])

    def test_start_phrase_spoken_even_when_worker_already_started(self):
        with _force_opencode_engine(), \
             patch.object(brain, "is_opencode_available", return_value=True), \
             patch.object(brain, "run_opencode_task", return_value="ok"), \
             patch.object(brain, "_notify_async_reply") as notify:
            reply = brain._execute_deferred_opencode(
                "create folder x", "create folder x"
            )
            # Hold the patches until the worker thread has finished so it
            # can never call the real run_opencode_task after unpatch.
            self.assertTrue(_wait_until(lambda: notify.called))
        # The flag flips before the worker thread starts, so the mute is
        # deterministic at handoff — and the start phrase is exempt.
        self.assertEqual(reply, brain.OPENCODE_START_PHRASE)
        with patch.object(routes, "opencode_task_in_progress",
                          return_value=True), \
             patch.object(routes, "speak") as spk:
            routes._maybe_speak(reply)
            routes._maybe_speak("some unrelated chat reply")
        spk.assert_called_once_with(brain.OPENCODE_START_PHRASE)


if __name__ == "__main__":
    unittest.main()
