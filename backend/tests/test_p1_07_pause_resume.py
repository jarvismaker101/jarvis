"""P1-07 — pause and resume actually work for voice-driven replies.

Pinned defects:

  * ``voice_mode._deliver_pause`` called the LOCAL ``stop_speaking()`` FIRST —
    which discards the unplayed queue and bumps the generation — and only then
    posted ``/speak/pause`` to the backend, whose active stream is EMPTY for a
    voice turn (the audio lives in the voice process). Net effect: "pause" was
    exactly a stop, and "continue" had nothing to resume.
  * there was no byte-level resume at all: the actor already tracked a spoken
    cursor (``abort()`` returns it, ``resume_from()`` replays the tail), but
    nothing on the pause path used it.

The fix: pause is a LOCAL, non-destructive control. The actor's play loop PARKS
(keeping the producer drained but writing nothing to the device, and NOT
advancing the cursor), the stream speaker keeps its queue, and "continue"
replays exactly the unplayed remainder into the same parked loop. A real
barge-in is still a cancellation: ``abort()`` ends a pause for good.

Audio devices, TTS and the network are stubbed; nothing here opens a device.
"""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from backend import voice_mode as vm
from backend.services import audio_actor
from backend.services import voice as voice_mod


# ─────────────────────────────────────────
# fakes
# ─────────────────────────────────────────

class RecordingStream:
    """Minimal output-stream fake: records the bytes in write order (F32)."""

    def __init__(self, slow=0.0):
        self.written = []
        self.stopped = False
        self.closed = False
        self._slow = slow
        self._lock = threading.Lock()

    def write(self, pcm_bytes):
        if self._slow:
            time.sleep(self._slow)
        with self._lock:
            self.written.append(bytes(pcm_bytes))

    def played(self):
        with self._lock:
            return b"".join(self.written)

    def stop(self):
        self.stopped = True

    def abort(self):
        self.stopped = True

    def close(self):
        self.closed = True


def _actor(stream, **kwargs):
    return audio_actor.AudioActor(stream_factory=lambda: stream, **kwargs)


def _wait(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


# ─────────────────────────────────────────
# AudioActor — pause parks the loop, resume replays the tail
# ─────────────────────────────────────────

class ActorPauseTests(unittest.TestCase):
    PCM_CHUNKS = 10
    CHUNK = 100

    def setUp(self):
        # A deliberately SLOW device: the point of a pause is to catch the
        # utterance mid-playback, so a fake that swallows the whole thing
        # instantly would never exercise it.
        self.stream = RecordingStream(slow=0.02)
        self.actor = _actor(self.stream)
        self.key = "p1-07"
        self.generation = self.actor.begin(self.key)
        self.pcm = b"".join(bytes([index]) * self.CHUNK
                            for index in range(self.PCM_CHUNKS))
        for index in range(0, len(self.pcm), self.CHUNK):
            self.actor.feed_chunk(self.key, self.generation,
                                  self.pcm[index:index + self.CHUNK])
        self.actor.end_utterance(self.key, self.generation)

    def _playing(self, at_least=300):
        waiter = threading.Thread(target=self.actor.play, daemon=True)
        waiter.start()
        self.assertTrue(
            _wait(lambda: self.actor.spoke_bytes() >= at_least),
            "the utterance never started playing")
        return waiter

    def test_pause_stops_the_device_and_keeps_the_utterance(self):
        waiter = self._playing()
        cursor = self.actor.pause()
        self.assertIsNotNone(cursor, "a playing utterance must be pausable")
        self.assertTrue(self.actor.paused())
        heard_at_pause = len(self.stream.played())
        self.assertLess(heard_at_pause, len(self.pcm),
                        "the test never caught the utterance mid-playback")
        time.sleep(0.2)
        self.assertEqual(
            len(self.stream.played()), heard_at_pause,
            "the device kept receiving audio after the pause")
        self.assertEqual(self.actor.spoke_bytes(), cursor,
                         "the spoken cursor advanced while paused")
        self.assertEqual(len(self.actor.utterance_pcm()), len(self.pcm),
                         "the utterance PCM was discarded by the pause")
        self.actor.resume_playback()
        waiter.join(5)

    def test_resume_replays_exactly_the_unplayed_remainder(self):
        waiter = self._playing()
        cursor = self.actor.pause()
        heard_before = self.stream.played()
        self.assertEqual(len(heard_before), cursor,
                         "the pause cursor is not the byte the device reached")
        # Not from the beginning, not from the end: the pause landed strictly
        # inside the utterance, so the resume point is a real mid-sentence point.
        self.assertGreater(cursor, 0)
        self.assertLess(cursor, len(self.pcm))
        self.assertTrue(self.actor.resume_playback(),
                        "the paused utterance did not resume")
        waiter.join(5)
        self.assertFalse(self.actor.paused())
        after = self.stream.played()
        # Nothing lost, nothing duplicated: what was heard before the pause plus
        # what was replayed is the whole utterance, in order.
        self.assertEqual(heard_before + after[len(heard_before):], self.pcm)
        self.assertEqual(after, self.pcm)

    def test_a_paused_loop_keeps_draining_the_producer(self):
        """A pause must not make a backpressured synthesis time out."""
        waiter = self._playing()
        self.actor.pause()
        accepted = []
        for _index in range(4):
            accepted.append(
                self.actor.feed_chunk(self.key, self.generation, b"\x01" * 64))
        self.assertEqual(accepted, [True] * 4,
                         "the producer was backpressured into a timeout by a pause")
        self.actor.resume_playback()
        waiter.join(5)

    def test_pause_with_nothing_playing_is_none(self):
        self.assertIsNone(self.actor.pause())
        self.assertFalse(self.actor.paused())

    def test_resume_with_nothing_paused_is_false(self):
        self.assertFalse(self.actor.resume_playback())

    def test_an_abort_clears_a_pause_for_good(self):
        """A barge-in is a cancellation, not a pause: no resumable remainder."""
        waiter = self._playing()
        self.assertIsNotNone(self.actor.pause())
        self.actor.abort()
        waiter.join(5)
        self.assertFalse(self.actor.paused(),
                         "an abort left the utterance resumable")
        self.assertFalse(self.actor.resume_playback(),
                         "an aborted utterance was resurrected")

    def test_a_new_utterance_clears_a_pause(self):
        waiter = self._playing()
        self.assertIsNotNone(self.actor.pause())
        self.actor.abort()
        waiter.join(5)
        self.actor.begin("next")
        self.assertFalse(self.actor.paused())
        self.assertFalse(self.actor.resume_playback(),
                         "a stale pause bled into the next utterance")


# ─────────────────────────────────────────
# StreamSpeaker — the gate that preserves the queue
# ─────────────────────────────────────────

class _SpeakerHarness(unittest.TestCase):
    def setUp(self):
        with voice_mod._state_lock:
            self._saved_flag = voice_mod.is_speaking
            voice_mod.is_speaking = False
        self._patches = [
            patch.object(voice_mod, "listener_state"),
            patch.object(voice_mod, "_prefetch_tts_audio"),
            patch.object(voice_mod, "play_ready_earcon"),
            patch.object(voice_mod, "play_reply_start_earcon"),
            patch.object(voice_mod, "_pause_actor_playback", return_value=321),
            patch.object(voice_mod, "_resume_actor_playback", return_value=True),
        ]
        for item in self._patches:
            item.start()
            self.addCleanup(item.stop)

    def tearDown(self):
        with voice_mod._state_lock:
            voice_mod.is_speaking = self._saved_flag


class SpeakerPauseTests(_SpeakerHarness):
    def test_pause_holds_the_next_chunk_and_keeps_the_queue(self):
        played = []
        first_played = threading.Event()

        def fake_chunk(chunk, generation, is_first_chunk=False):
            played.append(chunk)
            if len(played) == 1:
                first_played.set()
            return True

        with patch.object(voice_mod, "_speak_chunk", side_effect=fake_chunk):
            speaker = voice_mod.StreamSpeaker()
            self.addCleanup(speaker.close)
            speaker.feed("One two three four. ")
            self.assertTrue(first_played.wait(2))
            self.assertTrue(speaker.pause())
            speaker.feed("Five six seven eight. ")
            time.sleep(0.3)          # several STREAM_POLL_SECONDS
            self.assertEqual(played, ["One two three four."],
                             "a paused reply started another chunk")
            self.assertIn("Five six seven eight",
                          speaker.pending_text(),
                          "the pause discarded the unplayed queue")
            speaker.resume()
            self.assertTrue(_wait(lambda: len(played) >= 2),
                            "the reply never carried on after the resume")
            self.assertEqual(played[1], "Five six seven eight.")

    def test_pause_reports_false_when_there_is_nothing_to_pause(self):
        with patch.object(voice_mod, "_speak_chunk", return_value=True):
            speaker = voice_mod.StreamSpeaker()
            self.addCleanup(speaker.close)
            with patch.object(voice_mod, "_pause_actor_playback",
                              return_value=None):
                self.assertFalse(speaker.pause())
            speaker.resume()

    def test_resume_without_a_pause_is_a_no_op(self):
        speaker = voice_mod.StreamSpeaker()
        self.addCleanup(speaker.close)
        self.assertFalse(speaker.resume())
        self.assertFalse(speaker.paused)

    def test_a_paused_speaker_is_still_speaking(self):
        """The speaking flag must span a pause: the reply has not finished."""
        with patch.object(voice_mod, "_speak_chunk", return_value=True):
            speaker = voice_mod.StreamSpeaker()
            self.addCleanup(speaker.close)
            speaker.feed("A sentence to hold. ")
            self.assertTrue(_wait(lambda: voice_mod.is_speaking))
            speaker.pause()
            self.assertTrue(voice_mod.is_speaking)
            speaker.resume()
            speaker.finish()
            self.assertTrue(_wait(lambda: not voice_mod.is_speaking))


# ─────────────────────────────────────────
# voice service — pause is not stop
# ─────────────────────────────────────────

class PauseServiceTests(unittest.TestCase):
    def tearDown(self):
        voice_mod.listener_state.set_remaining("")

    def _restore_stream(self, previous):
        voice_mod.set_active_stream(previous)

    def test_pause_does_not_stop_and_keeps_the_remainder(self):
        speaker = MagicMock()
        speaker.pause.return_value = True
        previous = voice_mod.get_active_stream()
        voice_mod.set_active_stream(speaker)
        try:
            with patch.object(voice_mod, "stop_speaking") as stop, \
                    patch.object(voice_mod, "pending_speaking_text",
                                 return_value="the unplayed part"):
                self.assertTrue(voice_mod.pause_speaking())
        finally:
            self._restore_stream(previous)
        stop.assert_not_called()
        speaker.pause.assert_called_once_with()
        self.assertEqual(voice_mod.listener_state.get_remaining(),
                         "the unplayed part")

    def test_pause_with_nothing_playing_is_a_safe_no_op(self):
        previous = voice_mod.get_active_stream()
        voice_mod.set_active_stream(None)
        with voice_mod._state_lock:
            generation = voice_mod._speech_generation
        try:
            with patch.object(voice_mod, "_pause_actor_playback",
                              return_value=None), \
                    patch.object(voice_mod, "pending_speaking_text",
                                 return_value=""), \
                    patch.object(voice_mod, "stop_speaking") as stop:
                self.assertFalse(voice_mod.pause_speaking())
        finally:
            self._restore_stream(previous)
        stop.assert_not_called()
        with voice_mod._state_lock:
            self.assertEqual(voice_mod._speech_generation, generation,
                             "a no-op pause bumped the speech generation")

    def test_pause_covers_backend_style_audio_with_no_stream_speaker(self):
        """A ``speak()`` reply is paused through the actor directly."""
        previous = voice_mod.get_active_stream()
        voice_mod.set_active_stream(None)
        try:
            with patch.object(voice_mod, "_pause_actor_playback",
                              return_value=4096) as pause, \
                    patch.object(voice_mod, "pending_speaking_text",
                                 return_value="rest of the sentence"):
                self.assertTrue(voice_mod.pause_speaking())
        finally:
            self._restore_stream(previous)
        pause.assert_called_once_with()

    def test_resume_replays_parked_audio_without_re_speaking_the_snapshot(self):
        voice_mod.listener_state.set_remaining("queued remainder")
        previous = voice_mod.get_active_stream()
        voice_mod.set_active_stream(None)
        try:
            with patch.object(voice_mod, "_resume_actor_playback",
                              return_value=True), \
                    patch.object(voice_mod, "speak") as fake_speak:
                resumed = voice_mod.resume_speaking()
        finally:
            self._restore_stream(previous)
        self.assertEqual(resumed, "queued remainder")
        fake_speak.assert_not_called()
        self.assertEqual(voice_mod.listener_state.get_remaining(), "")

    def test_resume_without_parked_audio_speaks_the_remainder(self):
        voice_mod.listener_state.set_remaining("rest")
        previous = voice_mod.get_active_stream()
        voice_mod.set_active_stream(None)
        try:
            with patch.object(voice_mod, "_resume_actor_playback",
                              return_value=False), \
                    patch.object(voice_mod, "speak") as fake_speak:
                resumed = voice_mod.resume_speaking()
        finally:
            self._restore_stream(previous)
        self.assertEqual(resumed, "rest")
        fake_speak.assert_called_once_with("rest")

    def test_resume_with_nothing_pending_is_honest(self):
        voice_mod.listener_state.set_remaining("")
        previous = voice_mod.get_active_stream()
        voice_mod.set_active_stream(None)
        try:
            with patch.object(voice_mod, "_resume_actor_playback",
                              return_value=False), \
                    patch.object(voice_mod, "speak") as fake_speak:
                self.assertEqual(voice_mod.resume_speaking(), "")
        finally:
            self._restore_stream(previous)
        fake_speak.assert_not_called()

    def test_a_barge_in_is_a_cancellation_not_a_pause(self):
        from backend.services import listener as listener_mod

        with patch.object(listener_mod.listener_state, "is_speaking",
                          return_value=True), \
                patch.object(voice_mod, "stop_speaking") as stop, \
                patch.object(voice_mod, "pause_speaking") as pause, \
                patch.object(listener_mod, "_post_backend_speak_stop",
                             return_value=True):
            listener_mod.barge_in_on_speech_onset()

        stop.assert_called_once_with(signal_ready=False)
        pause.assert_not_called()


# ─────────────────────────────────────────
# voice_mode — the control that was wired to the wrong process
# ─────────────────────────────────────────

class DeliveryTests(unittest.TestCase):
    def test_deliver_pause_prefers_the_local_pause(self):
        with patch.object(vm, "pause_speaking", return_value=True), \
                patch.object(vm, "_post_backend") as post:
            self.assertTrue(vm._deliver_pause())
        post.assert_not_called()

    def test_deliver_pause_never_calls_stop_speaking(self):
        """Pause must no longer BE the implementation of stop."""
        with patch.object(vm, "pause_speaking", return_value=True), \
                patch.object(vm, "stop_speaking") as stop:
            vm._deliver_pause()
        stop.assert_not_called()

    def test_deliver_pause_falls_back_to_the_backend(self):
        with patch.object(vm, "pause_speaking", return_value=False), \
                patch.object(vm, "_post_backend",
                             return_value=(True, {"resumable": True})) as post:
            self.assertTrue(vm._deliver_pause())
        self.assertEqual(post.call_args[0][0], "/speak/pause")

    def test_deliver_pause_reports_nothing_to_resume(self):
        with patch.object(vm, "pause_speaking", return_value=False), \
                patch.object(vm, "_post_backend",
                             return_value=(True, {"resumable": False})):
            self.assertFalse(vm._deliver_pause())

    def test_deliver_continue_prefers_the_local_resume(self):
        with patch.object(vm, "resume_local_playback", return_value=True), \
                patch.object(vm, "_post_backend") as post:
            self.assertTrue(vm._deliver_continue())
        post.assert_not_called()

    def test_deliver_continue_falls_back_when_nothing_is_paused(self):
        with patch.object(vm, "resume_local_playback", return_value=False), \
                patch.object(vm, "_post_backend",
                             return_value=(True, {"resumed": True})) as post:
            self.assertTrue(vm._deliver_continue())
        self.assertEqual(post.call_args[0][0], "/speak/resume")

    def test_deliver_continue_never_re_speaks_from_this_process(self):
        """F50: this process must not become a second playback owner."""
        with patch.object(vm, "resume_local_playback", return_value=False), \
                patch.object(vm, "_post_backend", return_value=(True, {})), \
                patch.object(vm.listener_state, "pop_remaining") as popped, \
                patch.object(vm, "speak") as fake_speak:
            self.assertFalse(vm._deliver_continue())
        popped.assert_not_called()
        fake_speak.assert_not_called()


# ─────────────────────────────────────────
# the backend endpoints still serve backend-spoken audio
# ─────────────────────────────────────────

class BackendRouteTests(unittest.TestCase):
    def test_the_pause_route_still_pauses_backend_spoken_audio(self):
        from backend.api import routes

        with patch.object(voice_mod, "pause_speaking",
                          return_value=True) as pause, \
                patch.object(routes, "set_narration_enabled"):
            result = routes.pause_speech()
        self.assertTrue(result["ok"])
        self.assertTrue(result["resumable"])
        pause.assert_called_once_with()

    def test_the_resume_route_still_resumes_backend_spoken_audio(self):
        from backend.api import routes

        with patch.object(voice_mod, "resume_speaking",
                          return_value="the rest"):
            result = routes.resume_speech()
        self.assertTrue(result["ok"])
        self.assertTrue(result["resumed"])
        self.assertEqual(result["resumed_chars"], len("the rest"))


if __name__ == "__main__":
    unittest.main()
