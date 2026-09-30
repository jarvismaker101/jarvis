"""P1-06 — a committed transcript is never silently discarded, and the
"speaking" flag means "Jarvis is mid-reply" for the WHOLE reply.

Pinned defects:

  * ``voice_mode.listener_thread`` did ``if listener_state.is_speaking():
    continue``. A transcript committed while a reply was playing was thrown
    away with no handling, no log and no interruption — even though it had
    already passed the VAD, the human-voice gate and the hallucination gate.
  * ``voice.StreamSpeaker._playback_loop`` cleared the speaking flag whenever
    the playback queue was momentarily empty, so the flag flickered OFF in the
    gap between two sentences of the SAME reply. That is exactly what made the
    gate above unpredictable: sometimes an interruption landed, sometimes the
    utterance vanished with no trace.

The fix: speaking during a reply is an INTERRUPTION, so the utterance is handed
to the P0-08 turn manager (which cuts the turn that owns playback) and the flag
is scoped to the reply session — set at reply start, cleared only when the reply
is complete AND everything queued has been played.

Everything is stubbed: no network, no model, no microphone, no audio device.
"""

import ast
import inspect
import textwrap
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from backend import voice_mode as vm
from backend.services import voice as voice_mod


# ─────────────────────────────────────────
# listener_thread — the silent drop
# ─────────────────────────────────────────

class _SpeakerStub:
    """Records what the turn manager did to a turn's audio."""

    def __init__(self):
        self.fed = []
        self.closed = False
        self.finished = False

    def feed(self, delta):
        self.fed.append(delta)

    def close(self):
        self.closed = True

    def finish(self):
        self.finished = True

    @property
    def spoken_any(self):
        return bool(self.fed)


#: A text ``classify_control`` is stubbed to read as "shutdown", which makes
#: ``listener_thread`` leave its loop cleanly after one capture.
_STOP_SENTINEL = "\x00p1-06-stop"


def _once_then_shutdown(text):
    """Serve *text* on the first capture, then end the listener thread.

    ``listener_thread`` breaks out of its loop when a control classifies as
    "shutdown", so the thread stops deterministically while the patches are
    still in place — a thread still looping after the patch is removed would
    reach the real microphone. No exception is raised, so pytest's
    unhandled-thread-exception warning stays quiet.
    """
    served = {"done": False}

    def _impl(marks=None):
        if not served["done"]:
            served["done"] = True
            return text
        return _STOP_SENTINEL

    return _impl


def _classify_passthrough():
    """The real F35 grammar, plus the shutdown sentinel."""
    real = vm.classify_control

    def _impl(text):
        if text == _STOP_SENTINEL:
            return "shutdown"
        return real(text)

    return _impl


class _ListenerHarness(unittest.TestCase):
    def setUp(self):
        self._prev_turns = vm.TURNS
        vm.TURNS = vm._TurnManager()
        self.cancels = []
        self._patches = [
            patch.object(vm, "_cancel_backend_request_async",
                         side_effect=lambda rid, reason: self.cancels.append(
                             (rid, reason)) or True),
            patch.object(vm, "voice_input_enabled", return_value=True),
            patch.object(vm, "backend_task_running", return_value=False),
            patch.object(vm, "set_active_stream"),
            patch.object(vm, "_latency"),
        ]
        for item in self._patches:
            item.start()
            self.addCleanup(item.stop)

    def tearDown(self):
        vm.TURNS = self._prev_turns
        while True:
            try:
                vm.command_queue.get_nowait()
            except Exception:
                break

    def _run_listener(self, text, speaking):
        state = MagicMock()
        state.is_speaking.return_value = speaking
        with patch.object(vm, "listener_state", state), \
                patch.object(vm, "classify_control",
                             side_effect=_classify_passthrough()), \
                patch.object(vm, "shutdown_everything"), \
                patch.object(vm, "listen",
                             side_effect=_once_then_shutdown(text)):
            thread = threading.Thread(target=vm.listener_thread, daemon=True)
            thread.start()
            try:
                item = vm.command_queue.get(timeout=3)
            except Exception:
                item = None
            thread.join(timeout=3)
        return item


class NoSilentDropTests(_ListenerHarness):
    """An utterance committed during a reply must never be thrown away."""

    def test_an_utterance_during_a_reply_reaches_the_turn_manager(self):
        speaker = _SpeakerStub()
        vm.TURNS.start("voice-p106-1", speaker)

        item = self._run_listener("actually, what about the other file",
                                  speaking=True)

        self.assertIsNotNone(
            item, "a committed transcript was discarded while Jarvis spoke")
        self.assertEqual(item[0], "actually, what about the other file")

    def test_an_utterance_during_a_reply_interrupts_the_active_turn(self):
        speaker = _SpeakerStub()
        vm.TURNS.start("voice-p106-2", speaker)

        self._run_listener("actually, what about the other file",
                           speaking=True)

        self.assertFalse(vm.TURNS.is_current("voice-p106-2"),
                         "the reply being interrupted was never cut")
        self.assertTrue(speaker.closed,
                        "the interrupting utterance did not close the old audio")
        self.assertIn("voice-p106-2", [rid for rid, _ in self.cancels],
                      "the old backend request was never cancelled")

    def test_a_normal_utterance_is_unaffected_when_nothing_is_playing(self):
        item = self._run_listener("what is the weather today", speaking=False)

        self.assertIsNotNone(item)
        self.assertEqual(item[0], "what is the weather today")
        self.assertEqual(self.cancels, [],
                         "a normal utterance cancelled a turn")
        self.assertEqual(vm.TURNS.stats["barge_in_cancels"], 0)

    def test_a_control_phrase_is_still_a_control_and_not_a_turn(self):
        """The F35 grammar still wins: a control is never submitted as text."""
        state = MagicMock()
        state.is_speaking.return_value = True
        responses = iter(["speech_stop", "shutdown"])
        with patch.object(vm, "listener_state", state), \
                patch.object(vm, "classify_control",
                             side_effect=lambda text: next(responses)), \
                patch.object(vm, "dispatch_control") as dispatch, \
                patch.object(vm, "shutdown_everything"), \
                patch.object(vm, "listen",
                             side_effect=_once_then_shutdown("stop speaking")):
            thread = threading.Thread(target=vm.listener_thread, daemon=True)
            thread.start()
            thread.join(timeout=3)

        dispatch.assert_called_once_with("speech_stop")
        self.assertTrue(vm.command_queue.empty(),
                        "a control phrase was queued as an utterance")

    def test_no_path_drops_an_utterance_because_jarvis_is_speaking(self):
        """The structural guarantee: nothing keys a discard on the flag.

        Any ``if ... is_speaking ...:`` inside ``listener_thread`` must not have
        a bare ``continue``/``return``/``break`` as a body — that shape is
        exactly the silent drop that must never come back.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(vm.listener_thread)))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if "is_speaking" not in ast.dump(node.test):
                continue
            for stmt in node.body:
                if isinstance(stmt, (ast.Continue, ast.Return, ast.Break)):
                    offenders.append(node.lineno)
        self.assertEqual(
            offenders, [],
            "listener_thread discards a committed transcript while speaking "
            "(line %s)" % offenders)


class InterruptHelperTests(_ListenerHarness):
    """``_interrupt_active_turn`` is the one interrupt entry point."""

    def test_interrupting_with_no_active_turn_is_a_safe_no_op(self):
        self.assertFalse(vm._interrupt_active_turn())
        self.assertEqual(self.cancels, [])

    def test_interrupting_is_idempotent(self):
        vm.TURNS.start("voice-p106-3", _SpeakerStub())
        self.assertTrue(vm._interrupt_active_turn())
        self.assertFalse(vm._interrupt_active_turn())
        self.assertEqual(len(self.cancels), 1)

    def test_the_helper_never_raises(self):
        with patch.object(vm.TURNS, "cancel_current",
                          side_effect=RuntimeError("boom")):
            self.assertFalse(vm._interrupt_active_turn())


# ─────────────────────────────────────────
# StreamSpeaker — the flickering flag
# ─────────────────────────────────────────

class _SpeakingRecorder:
    """Stands in for ``listener_state`` inside ``voice``."""

    def __init__(self):
        self.calls = []

    def set_speaking(self, speaking):
        self.calls.append(bool(speaking))

    def set_remaining(self, text):
        pass

    def set_thinking(self, thinking):
        pass


class _SpeakerHarness(unittest.TestCase):
    def setUp(self):
        with voice_mod._state_lock:
            self._saved_flag = voice_mod.is_speaking
            voice_mod.is_speaking = False
        self.recorder = _SpeakingRecorder()
        self._patches = [
            patch.object(voice_mod, "listener_state", self.recorder),
            patch.object(voice_mod, "_prefetch_tts_audio"),
            patch.object(voice_mod, "play_ready_earcon"),
            patch.object(voice_mod, "play_reply_start_earcon"),
        ]
        for item in self._patches:
            item.start()
            self.addCleanup(item.stop)

    def tearDown(self):
        with voice_mod._state_lock:
            voice_mod.is_speaking = self._saved_flag

    def _wait(self, predicate, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()


class ReplySessionFlagTests(_SpeakerHarness):
    """Requirement 3: the flag spans the reply session, not the queue."""

    def test_the_flag_does_not_flicker_off_between_sentences(self):
        played = []
        first_played = threading.Event()

        def fake_chunk(chunk, generation, is_first_chunk=False, **_kwargs):
            played.append(chunk)
            first_played.set()
            return True

        with patch.object(voice_mod, "_speak_chunk", side_effect=fake_chunk):
            speaker = voice_mod.StreamSpeaker()
            self.addCleanup(speaker.close)
            speaker.feed("First sentence lands here. ")
            self.assertTrue(first_played.wait(2),
                            "the first chunk never reached playback")
            self.assertTrue(self._wait(lambda: not speaker._queue_has_audio()))
            # The classic flicker window: the queue is empty, but the reply is
            # NOT finished — more deltas are still coming.
            time.sleep(0.2)
            self.assertNotIn(
                False, self.recorder.calls,
                "the speaking flag was cleared between two sentences of the "
                "SAME reply")
            self.assertTrue(voice_mod.is_speaking)

    def test_the_flag_survives_a_gap_and_then_a_later_sentence(self):
        played = []
        first_played = threading.Event()

        def fake_chunk(chunk, generation, is_first_chunk=False, **_kwargs):
            played.append(chunk)
            first_played.set()
            return True

        with patch.object(voice_mod, "_speak_chunk", side_effect=fake_chunk):
            speaker = voice_mod.StreamSpeaker()
            self.addCleanup(speaker.close)
            speaker.feed("First sentence lands here. ")
            self.assertTrue(first_played.wait(2))
            time.sleep(0.1)
            speaker.feed("Second sentence arrives later. ")
            self.assertTrue(self._wait(lambda: len(played) >= 2))
            self.assertNotIn(False, self.recorder.calls,
                             "a later sentence of the same reply reset the flag")

    def test_the_flag_clears_once_the_reply_is_complete(self):
        played = []
        first_played = threading.Event()

        def fake_chunk(chunk, generation, is_first_chunk=False, **_kwargs):
            played.append(chunk)
            first_played.set()
            return True

        with patch.object(voice_mod, "_speak_chunk", side_effect=fake_chunk):
            speaker = voice_mod.StreamSpeaker()
            self.addCleanup(speaker.close)
            speaker.feed("The only sentence of this reply. ")
            self.assertTrue(first_played.wait(2))
            self.assertIn(True, self.recorder.calls)
            speaker.finish()          # reply complete, nothing left buffered
            self.assertTrue(self._wait(lambda: False in self.recorder.calls),
                            "the flag stayed set after the reply ended")
            self.assertFalse(voice_mod.is_speaking)

    def test_a_finished_reply_waits_for_the_last_chunk_to_play(self):
        """The flag must outlive the deltas: audio is still coming."""
        played = []
        release = threading.Event()
        playing = threading.Event()

        def fake_chunk(chunk, generation, is_first_chunk=False, **_kwargs):
            played.append(chunk)
            playing.set()
            release.wait(3)
            return True

        with patch.object(voice_mod, "_speak_chunk", side_effect=fake_chunk):
            speaker = voice_mod.StreamSpeaker()
            self.addCleanup(speaker.close)
            speaker.feed("A sentence that takes its time. ")
            self.assertTrue(playing.wait(2))
            speaker.finish()
            time.sleep(0.15)
            self.assertNotIn(False, self.recorder.calls,
                             "the flag cleared while audio was still playing")
            release.set()
            self.assertTrue(self._wait(lambda: False in self.recorder.calls),
                            "the flag never cleared after the audio finished")


class ReplySessionOverTests(_SpeakerHarness):
    """``_reply_session_over`` is the session boundary the loop consults."""

    def _speaker(self):
        speaker = voice_mod.StreamSpeaker()
        # No worker: these tests assert the predicate directly.
        speaker._start_worker = lambda: None
        self.addCleanup(speaker.close)
        return speaker

    def test_a_still_streaming_reply_is_not_over(self):
        speaker = self._speaker()
        speaker._queue.put("something to say")
        self.assertFalse(speaker._reply_session_over())

    def test_a_finished_reply_with_pending_audio_is_not_over(self):
        speaker = self._speaker()
        speaker._queue.put("pending audio")
        speaker.finish()
        self.assertFalse(speaker._reply_session_over())

    def test_a_finished_reply_with_nothing_left_is_over(self):
        speaker = self._speaker()
        speaker.finish()
        self.assertTrue(speaker._reply_session_over())

    def test_a_finished_reply_with_a_buffered_remainder_is_not_over(self):
        speaker = self._speaker()
        with speaker._buffer_lock:
            speaker._buffer = "half a sentence"
        speaker.finish()
        self.assertFalse(speaker._reply_session_over())

    def test_a_closed_speaker_with_nothing_left_is_over(self):
        """The ``_STREAM_STOP`` sentinel is not audio."""
        speaker = self._speaker()
        speaker.close()
        self.assertTrue(speaker._reply_session_over(),
                        "the stop sentinel was mistaken for unplayed audio")


if __name__ == "__main__":
    unittest.main()
