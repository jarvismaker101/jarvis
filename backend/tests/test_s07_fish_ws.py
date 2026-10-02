"""S7 - one live Fish WebSocket session per reply.

Acceptance: the reply's sentences are FED to a single bidirectional session
and play as one continuous utterance through the one playback owner; a
session that fails before ANY audio reached the device hands every sentence
back to the per-sentence ladder in order; a reply that already spoke is
never replayed; the HTTP path stays the fallback.
"""

import threading
import time
import unittest
from unittest.mock import patch

from backend.services import fish_voice, voice


def _wait_until(predicate, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


class _FakeSession:
    """A session stand-in whose audio is produced by a scripted thread."""

    def __init__(self, fail=False, chunks=(b"\x01\x00" * 4000,), delay=0.01):
        self.fed = []
        self.finished_text = False
        self.closed = False
        self._fail = fail
        self._chunks = list(chunks)
        self._delay = delay
        self._q = []
        self._lock = threading.Lock()
        self._arrived = threading.Event()
        self._failed = False
        self._produced_all = False
        self._start()

    def _start(self):
        thread = threading.Thread(target=self._produce, daemon=True)
        thread.start()

    def _produce(self):
        if self._fail:
            time.sleep(self._delay)
            self._failed = True
            self._arrived.set()
            return
        time.sleep(self._delay)
        for chunk in self._chunks:
            with self._lock:
                self._q.append(chunk)
            self._arrived.set()
            time.sleep(0.005)
        # Scripted stream complete: like a real session whose reader finished.
        self._produced_all = True
        self._arrived.set()

    # session API used by the code under test
    def feed(self, text):
        self.fed.append(text)

    def finish_text(self):
        self.finished_text = True
        self._arrived.set()

    def failed(self):
        return self._failed

    def failure(self):
        return RuntimeError("scripted failure") if self._failed else None

    def audio_chunks(self, _seen=None):
        while True:
            with self._lock:
                chunk = self._q.pop(0) if self._q else None
            if chunk is not None:
                yield chunk
                continue
            if (self._failed or self.closed
                    or (self.finished_text and self._produced_all)):
                return
            self._arrived.wait(0.05)
            self._arrived.clear()

    def close(self):
        self.closed = True
        self._arrived.set()

    def drain(self):
        with self._lock:
            out = list(self._q)
            self._q = []
        return b"".join(out)


class SessionLifecycleTests(unittest.TestCase):
    """The real session object: feed -> audio -> text-finished -> end."""

    def test_feed_then_finish_yields_audio_and_ends(self):
        session = fish_voice._FishReplySession()
        session._read = lambda: None  # no SDK in this unit test
        session.feed("Hello there, sir.")
        session.feed("And a second sentence.")
        session.finish_text()
        chunks = list(session.audio_chunks())
        self.assertEqual(chunks, [])
        self.assertTrue(session._text_done)
        session.close()

    def test_audio_is_stitched_and_boosted_odd_byte_splits(self):
        session = _FakeSession(chunks=(b"\x01\x00\x02", b"\x03\x00"))
        session.finish_text()
        with patch.object(fish_voice, "_boost_pcm_chunk", side_effect=lambda c: c):
            collected = list(fish_voice._ws_boosted_chunks(session))
        # The lone odd byte was stitched into a full sample and flushed at the
        # end: every chunk is sample-aligned and nothing is dropped.
        self.assertEqual(b"".join(collected),
                         b"\x01\x00" b"\x02\x03" b"\x00\x00")
        for chunk in collected:
            self.assertEqual(len(chunk) % 2, 0)

    def test_wrong_engine_is_a_noop(self):
        voice.fish_ws_end_reply()
        with patch.object(voice, "_resolve_session_tts_provider",
                          return_value="gtts"):
            fish_voice.fish_ws_begin_reply()
        self.assertIsNone(fish_voice.fish_ws_session())
        fish_voice.fish_ws_end_reply()

    def test_kill_switch_disables_the_engine(self):
        with patch.dict("os.environ", {fish_voice.FISH_WS_ENV: "0"}):
            self.assertFalse(fish_voice.fish_ws_enabled())


class PlaybackTests(unittest.TestCase):
    """play_ws_reply: one utterance through the one playback owner."""

    def setUp(self):
        self.fed_chunks = []
        self._orig = {}

    def _patch_actor(self):
        """Replace the playback owner + device plumbing with a recorder."""

        def fake_stream_to_actor(key, chunks, handle, out=None):
            for chunk in chunks:
                self.fed_chunks.append(chunk)
                if out is not None:
                    out["generation"] = 7
            return b"".join(self.fed_chunks), False

        return patch.object(fish_voice, "_stream_pcm_to_actor",
                            side_effect=fake_stream_to_actor)

    def test_a_whole_reply_plays_as_one_continuous_stream(self):
        session = _FakeSession(chunks=(b"\x01\x00" * 5000, b"\x03\x00" * 5000))
        session.fed = []
        session.feed("First sentence.")
        session.feed("Second sentence.")
        session.finish_text()
        earcons = []
        with self._patch_actor(), \
                patch.object(fish_voice, "_register_sounddevice_playback",
                             return_value=None), \
                patch.object(fish_voice, "ws_audio_reached_device",
                             return_value=True):
            outcome = fish_voice.play_ws_reply(
                session, earcon=lambda: earcons.append(1))
        self.assertTrue(outcome["audio_started"])
        self.assertEqual(len(earcons), 1)
        self.assertGreater(len(self.fed_chunks), 0)

    def test_a_cancelled_reply_stops_feeding_audio(self):
        session = _FakeSession(chunks=(b"\x01\x00" * 5000,))
        session.feed("Only sentence.")
        session.finish_text()
        with self._patch_actor(), \
                patch.object(fish_voice, "_register_sounddevice_playback",
                             return_value=None), \
                patch.object(fish_voice, "ws_audio_reached_device",
                             return_value=True):
            outcome = fish_voice.play_ws_reply(
                session, is_current=lambda: False)
        self.assertEqual(self.fed_chunks, [])


class _LiveStream:
    """A play_ws_reply stand-in that holds the stream open until released."""

    def __init__(self, audio_started=True, failed=False):
        self.release = threading.Event()
        self.audio_started = audio_started
        self.failed = failed

    def __call__(self, session, is_current=None, earcon=None):
        self.release.wait(5.0)
        return {"audio_started": self.audio_started, "failed": self.failed}


class StreamSpeakerTests(unittest.TestCase):
    """The streaming speaker feeds one session and falls back honestly."""

    def _speaker(self):
        speaker = voice.StreamSpeaker()
        self.addCleanup(speaker.close)
        return speaker

    def test_a_streamed_reply_is_fed_to_one_session_not_spoken_per_sentence(self):
        speaker = self._speaker()
        session = _FakeSession(chunks=(b"\x01\x00" * 20000,))
        stream = _LiveStream()
        played = []
        self.addCleanup(stream.release.set)
        with patch.object(voice, "fish_ws_begin_reply"), \
                patch.object(voice, "fish_ws_session", return_value=session), \
                patch.object(voice, "fish_ws_end_reply"), \
                patch.object(voice, "play_ws_reply", new=stream), \
                patch.object(voice, "_speak_chunk",
                             side_effect=lambda *a, **k: played.append(a[0])):
            speaker.feed("First sentence. ")
            speaker.feed("Second sentence.")
            speaker.finish()
            self.assertTrue(
                _wait_until(lambda: len(session.fed) == 2),
                "sentences were not fed to the session: %r" % (session.fed,))
            # Mid-reply: the session is still streaming and the per-sentence
            # ladder has spoken NOTHING.
            self.assertEqual(played, [])
        self.assertEqual(session.fed,
                         ["First sentence.", "Second sentence."])

    def test_a_session_that_never_spoke_hands_the_reply_back_to_the_ladder(self):
        speaker = self._speaker()
        broken = _FakeSession(fail=True)
        played = []
        with patch.object(voice, "fish_ws_begin_reply"), \
                patch.object(voice, "fish_ws_session", return_value=broken), \
                patch.object(voice, "fish_ws_end_reply"), \
                patch.object(voice, "play_ws_reply",
                             side_effect=lambda *a, **k: {
                                 "audio_started": False,
                                 "failed": True}), \
                patch.object(voice, "ws_audio_reached_device",
                             return_value=False), \
                patch.object(voice, "_speak_chunk",
                             side_effect=lambda *a, **k: played.append(a[0])):
            speaker.feed("First sentence. ")
            speaker.feed("Second sentence.")
            speaker.finish()
            self.assertTrue(_wait_until(lambda: len(played) == 2),
                            "ladder did not speak the reply: %r "
                            "ws_mode=%s fed=%r q=%s finished=%s active=%s"
                            % (played, speaker._ws_mode, speaker._ws_fed,
                               speaker._queue.qsize(), speaker._finished,
                               speaker._active))
        # Every sentence is spoken EXACTLY once, in order, by the ladder.
        self.assertEqual(played, ["First sentence.", "Second sentence."])

    def test_a_stopped_reply_never_replays_what_already_spoke(self):
        speaker = self._speaker()
        session = _FakeSession(fail=True)
        played = []
        with patch.object(voice, "fish_ws_begin_reply"), \
                patch.object(voice, "fish_ws_session", return_value=session), \
                patch.object(voice, "fish_ws_end_reply"), \
                patch.object(voice, "play_ws_reply",
                             return_value={"audio_started": True,
                                           "failed": True}), \
                patch.object(voice, "ws_audio_reached_device",
                             return_value=True), \
                patch.object(voice, "_speak_chunk",
                             side_effect=lambda *a, **k: played.append(a[0])):
            speaker.feed("First sentence. ")
            speaker.feed("Second sentence.")
            speaker.finish()
            self.assertTrue(_wait_until(lambda: not speaker._ws_mode))
            time.sleep(0.1)
        self.assertEqual(played, [])


class SpeakTests(unittest.TestCase):
    """speak(): one session for the whole reply, ladder untouched on failure."""

    def test_the_whole_reply_rides_one_session(self):
        session = _FakeSession(chunks=(b"\x01\x00" * 20000,))
        session.fed = []
        spoken = []
        with patch.object(voice, "fish_ws_begin_reply"), \
                patch.object(voice, "fish_ws_session", return_value=session), \
                patch.object(voice, "fish_ws_end_reply"), \
                patch.object(voice, "play_ws_reply",
                             return_value={"audio_started": True,
                                           "failed": False}), \
                patch.object(voice, "_speak_chunk",
                             side_effect=lambda *a, **k: spoken.append(a[0])):
            voice.speak("Hello there, sir. This is a second sentence.")
            self.assertTrue(_wait_until(lambda: session.finished_text))
        self.assertTrue(session.fed)
        self.assertEqual(spoken, [])

    def test_no_audio_falls_back_to_the_sentence_ladder(self):
        session = _FakeSession(fail=True)
        spoken = []
        with patch.object(voice, "fish_ws_begin_reply"), \
                patch.object(voice, "fish_ws_session", return_value=session), \
                patch.object(voice, "fish_ws_end_reply"), \
                patch.object(voice, "play_ws_reply",
                             return_value={"audio_started": False,
                                           "failed": True}), \
                patch.object(voice, "_speak_chunk",
                             side_effect=lambda *a, **k: spoken.append(a[0])):
            voice.speak("Hello there, sir.")
            self.assertTrue(_wait_until(lambda: spoken))
        self.assertEqual(spoken, ["Hello there, sir."])


if __name__ == "__main__":
    unittest.main()