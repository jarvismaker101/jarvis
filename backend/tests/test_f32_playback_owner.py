"""F32 — give playback one real owner.

Acceptance (audit report): "Stop during prefetch/read/write prevents old
output; long streams lose nothing; repeated resume neither duplicates nor
skips; model changes cannot reuse wrong-identity audio."

What the baseline got wrong, pinned here:
  * ``fish_voice`` created its own ``OutputStream`` (and used ``sd.play``) while
    separately feeding the actor — the actor did not own the device;
  * a stop during a prefetch wait registered a playback handle only AFTER the
    cancellation had already happened;
  * ring overflow silently dropped PCM;
  * the spoken cursor counted SUBMITTED bytes, so a resume replayed or skipped
    audio;
  * cache/in-flight cleanup used the text where the entry was keyed by the
    audio identity, and one branch popped the undefined name ``ck``.
"""

import sys
import threading
import time
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from backend.services import audio_actor
from backend.services import fish_voice as fish_mod


class RecordingStream:
    """Minimal output-stream fake: records the bytes in write order."""

    def __init__(self, slow=0.0, fail_on_write=None):
        self.written = []
        self.started = False
        self.start_calls = 0
        self.stopped = False
        self.closed = False
        self._slow = slow
        self._fail_on_write = fail_on_write
        self._lock = threading.Lock()

    def start(self):
        """A real blocking stream must be started before it accepts data."""
        self.start_calls += 1
        self.started = True

    def write(self, pcm_bytes):
        if self._fail_on_write is not None and len(self.written) >= self._fail_on_write:
            raise RuntimeError("device write failed")
        if self._slow:
            time.sleep(self._slow)
        with self._lock:
            self.written.append(bytes(pcm_bytes))

    def played(self):
        with self._lock:
            return b"".join(self.written)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def _actor(stream, **kwargs):
    return audio_actor.AudioActor(stream_factory=lambda: stream, **kwargs)


class SingleOwnerTests(unittest.TestCase):
    """One real owner: the actor writes; nobody else touches a stream."""

    def test_the_actor_owns_the_stream_and_accounts_consumed_frames(self):
        stream = RecordingStream()
        actor = _actor(stream)
        generation = actor.begin("utt")
        for index in range(4):
            self.assertTrue(actor.feed_chunk("utt", generation, b"%d" % index * 100))
        actor.end_utterance("utt", generation)
        written = actor.play()
        self.assertEqual(written, 400)
        self.assertEqual(actor.spoke_bytes(), 400, "cursor must be consumed bytes")
        self.assertEqual(len(stream.played()), 400)
        self.assertTrue(stream.closed, "the owner closes what it opened")

    def test_a_stale_generation_chunk_is_never_written(self):
        stream = RecordingStream()
        actor = _actor(stream)
        first = actor.begin("one")
        stale = b"\x01" * 10
        self.assertTrue(actor.feed_chunk("one", first, stale))
        second = actor.begin("two")  # a new answer supersedes the old one
        self.assertFalse(actor.feed_chunk("one", first, b"\x02" * 10))
        self.assertTrue(actor.feed_chunk("two", second, b"\x03" * 10))
        actor.end_utterance("two", second)
        actor.play()
        self.assertNotIn(stale, stream.played())
        self.assertIn(b"\x03" * 10, stream.played())

    def test_fish_voice_never_calls_sd_play_or_sd_stop(self):
        """No global playback authority: only the actor opens a stream."""
        fake_sd = MagicMock()
        fake_sd.OutputStream.return_value = RecordingStream()

        def fake_post(*args, **kwargs):
            response = MagicMock()
            response.status_code = 200
            response.iter_content = lambda chunk_size=4096: iter([b"\x01\x00" * 512])
            return response

        with patch.object(fish_mod._session, "post", side_effect=fake_post), \
             patch.object(fish_mod, "FISH_API_KEY", "fake-key"), \
             patch.dict("sys.modules", {"sounddevice": fake_sd}):
            fish_mod._audio_cache.clear()
            fish_mod._in_flight.clear()
            self.assertTrue(fish_mod._do_pcm_stream("one owner", play=True))
        fake_sd.play.assert_not_called()
        fake_sd.stop.assert_not_called()
        fake_sd.OutputStream.assert_called_once()

    def test_the_stop_handle_never_stops_every_stream(self):
        """`sd.stop()` kills unrelated audio; the abort is per-utterance."""
        fake_sd = MagicMock()
        with patch.dict("sys.modules", {"sounddevice": fake_sd}):
            handle = fish_mod._register_sounddevice_playback()
            handle.stop()
        fake_sd.stop.assert_not_called()
        self.assertTrue(handle.stopped)
        self.assertNotEqual(audio_actor.get_actor().state(), "playing")


class CancellationTests(unittest.TestCase):
    def tearDown(self):
        fish_mod._audio_cache.clear()
        fish_mod._in_flight.clear()

    def test_a_stop_during_the_prefetch_wait_plays_nothing(self):
        """The old code registered its handle only after the wait returned."""
        text = "stop during prefetch"
        ck = fish_mod._pcm_cache_key(text)
        owner_done = threading.Event()
        fish_mod._in_flight[ck] = owner_done

        with patch.object(fish_mod, "FISH_API_KEY", "fake-key"), \
             patch.object(fish_mod, "_play_pcm_through_actor") as player, \
             patch.object(fish_mod, "_replay_cached_pcm") as replay:
            result_holder = {}

            def _call():
                result_holder["result"] = fish_mod._do_pcm_stream(text, play=True)

            thread = threading.Thread(target=_call, daemon=True)
            thread.start()
            time.sleep(0.05)
            # The user hits stop while the playback thread is still waiting on
            # the prefetch owner.
            fish_mod.stop_fish_audio()
            owner_done.set()
            thread.join(10)
        player.assert_not_called()
        replay.assert_not_called()
        self.assertFalse(result_holder.get("result"),
                         "a stopped request must not report playback")

    def test_a_stop_mid_stream_prevents_later_output(self):
        """Chunks produced after the stop must never reach the device."""
        written = []
        written_lock = threading.Lock()

        class Stream:
            def write(self, pcm_bytes):
                with written_lock:
                    written.append(bytes(pcm_bytes))

            def stop(self):
                pass

            def close(self):
                pass

        stream = Stream()
        actor = _actor(stream)
        key = "mid-stream"
        generation = actor.begin(key)
        self.assertTrue(actor.feed_chunk(key, generation, b"\x01" * 100))
        # Let the first chunk be written, then stop.
        waiter = threading.Thread(target=actor.play, daemon=True)
        waiter.start()
        deadline = time.monotonic() + 5
        while not written and time.monotonic() < deadline:
            time.sleep(0.01)
        actor.abort()
        # Everything the producer hands over from now on is refused.
        self.assertFalse(actor.feed_chunk(key, generation, b"\x02" * 100))
        waiter.join(5)
        self.assertFalse(waiter.is_alive())
        self.assertIn(b"\x01" * 100, b"".join(written))
        self.assertNotIn(b"\x02" * 100, b"".join(written))

    def test_restarting_a_new_utterance_discards_the_old_ring(self):
        stream = RecordingStream()
        actor = _actor(stream)
        old = actor.begin("old")
        for _ in range(4):
            actor.feed_chunk("old", old, b"\x09" * 64)
        new = actor.begin("new")
        actor.feed_chunk("new", new, b"\x07" * 64)
        actor.end_utterance("new", new)
        actor.play()
        self.assertEqual(stream.played(), b"\x07" * 64,
                         "old-generation buffered PCM must not leak into the "
                         "new answer")


class BackpressureTests(unittest.TestCase):
    def test_a_long_stream_loses_nothing(self):
        """Ring overflow used to popleft() (drop) the oldest chunk."""
        stream = RecordingStream(slow=0.002)
        actor = _actor(stream, ring_chunks=4)
        key = "long"
        generation = actor.begin(key)
        waiter = threading.Thread(target=actor.play, daemon=True)
        waiter.start()
        chunks = [bytes([index % 251]) * 128 for index in range(40)]
        for chunk in chunks:
            self.assertTrue(actor.feed_chunk(key, generation, chunk),
                            "backpressure must wait for room, never drop")
        actor.end_utterance(key, generation)
        waiter.join(20)
        self.assertEqual(actor.dropped_chunks(), 0)
        self.assertEqual(stream.played(), b"".join(chunks),
                         "every byte of a long stream must be played in order")

    def test_a_producer_that_cannot_be_served_is_told(self):
        """A dead consumer must not hang the producer forever."""
        actor = _actor(RecordingStream(), ring_chunks=2)
        key = "stuck"
        generation = actor.begin(key)
        self.assertTrue(actor.feed_chunk(key, generation, b"\x01" * 8))
        self.assertTrue(actor.feed_chunk(key, generation, b"\x02" * 8))
        # No play loop is running, so the ring stays full.
        self.assertFalse(actor.feed_chunk(key, generation, b"\x03" * 8,
                                          timeout=0.05))
        self.assertEqual(actor.dropped_chunks(), 1)
        self.assertIn("ring full", actor.last_error() or "")


class ResumeTests(unittest.TestCase):
    def test_repeated_resume_neither_duplicates_nor_skips(self):
        stream = RecordingStream()
        actor = _actor(stream)
        key = "resume"
        generation = actor.begin(key)
        pcm = b"".join(bytes([index]) * 100 for index in range(10))  # 1000 bytes
        for index in range(0, len(pcm), 100):
            actor.feed_chunk(key, generation, pcm[index:index + 100])
        actor.end_utterance(key, generation)
        waiter = threading.Thread(target=actor.play, daemon=True)
        waiter.start()
        # Interrupt after the device consumed a few chunks.
        deadline = time.monotonic() + 5
        while actor.spoke_bytes() < 300 and time.monotonic() < deadline:
            time.sleep(0.005)
        cursor = actor.abort()
        waiter.join(5)
        heard = len(stream.played())
        self.assertGreaterEqual(cursor, heard)
        # Resume twice: the second resume has nothing left to re-enqueue
        # because the cursor already accounted for everything re-sent.
        first = actor.resume_from(cursor)
        second = actor.resume_from(cursor)
        self.assertEqual(first, len(pcm) - cursor)
        self.assertEqual(second, 0, "resume must not re-send already consumed audio")

    def test_resume_never_replays_what_was_already_heard(self):
        stream = RecordingStream()
        actor = _actor(stream)
        key = "no-duplicate"
        generation = actor.begin(key)
        pcm = bytes([7]) * 400
        for index in range(0, 400, 100):
            actor.feed_chunk(key, generation, pcm[index:index + 100])
        actor.end_utterance(key, generation)
        waiter = threading.Thread(target=actor.play, daemon=True)
        waiter.start()
        deadline = time.monotonic() + 5
        while actor.spoke_bytes() < 200 and time.monotonic() < deadline:
            time.sleep(0.005)
        cursor = actor.abort()
        waiter.join(5)
        before = stream.played()
        # Resume asking to start from zero: the consumed cursor wins.
        replayed = actor.resume_from(0)
        self.assertEqual(replayed, len(pcm) - cursor)
        waiter = threading.Thread(target=actor.play, daemon=True)
        waiter.start()
        actor.end_utterance(key, actor.generation())
        waiter.join(5)
        self.assertTrue(actor.utterance_pcm().endswith(stream.played()[len(before):] + b""))


class CacheIdentityTests(unittest.TestCase):
    def tearDown(self):
        fish_mod._audio_cache.clear()
        fish_mod._in_flight.clear()
        fish_mod._clear_device_cache()

    def test_different_models_never_share_an_audio_slot(self):
        one = audio_actor.audio_cache_key("model-a", "ref", "pcm", "hello")
        two = audio_actor.audio_cache_key("model-b", "ref", "pcm", "hello")
        three = audio_actor.audio_cache_key("model-a", "other-ref", "pcm", "hello")
        self.assertEqual(len({one, two, three}), 3)

    def test_the_pcm_key_tracks_the_live_tts_model(self):
        with patch.object(fish_mod, "_resolve_tts_model", return_value="m-one"):
            first = fish_mod._pcm_cache_key("hello")
        with patch.object(fish_mod, "_resolve_tts_model", return_value="m-two"):
            second = fish_mod._pcm_cache_key("hello")
        self.assertNotEqual(first, second,
                            "a model change must not reuse another model's audio")

    def test_a_prefetched_sentence_is_not_replayed_after_a_model_change(self):
        calls = []

        def fake_post(*args, **kwargs):
            calls.append(kwargs.get("json", {}).get("model"))
            response = MagicMock()
            response.status_code = 200
            response.iter_content = lambda chunk_size=4096: iter([b"\x05\x00" * 64])
            return response

        fake_sd = MagicMock()
        fake_sd.OutputStream.return_value = RecordingStream()
        with patch.object(fish_mod._session, "post", side_effect=fake_post), \
             patch.object(fish_mod, "FISH_API_KEY", "fake-key"), \
             patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with patch.object(fish_mod, "_resolve_tts_model", return_value="m-one"):
                self.assertTrue(fish_mod._do_pcm_stream("same sentence", play=False))
            with patch.object(fish_mod, "_resolve_tts_model", return_value="m-two"):
                self.assertTrue(fish_mod._do_pcm_stream("same sentence", play=False))
        self.assertEqual(calls, ["m-one", "m-two"],
                         "the new model must synthesise instead of replaying the "
                         "old model's cached audio")

    def test_the_in_flight_entry_is_cleared_after_a_stream(self):
        def fake_post(*args, **kwargs):
            response = MagicMock()
            response.status_code = 200
            response.iter_content = lambda chunk_size=4096: iter([b"\x01\x00" * 64])
            return response

        fake_sd = MagicMock()
        fake_sd.OutputStream.return_value = RecordingStream()
        with patch.object(fish_mod._session, "post", side_effect=fake_post), \
             patch.object(fish_mod, "FISH_API_KEY", "fake-key"), \
             patch.dict("sys.modules", {"sounddevice": fake_sd}):
            fish_mod._audio_cache.clear()
            fish_mod._in_flight.clear()
            ck = fish_mod._pcm_cache_key("cleanup")
            self.assertTrue(fish_mod._do_pcm_stream("cleanup", play=True))
            self.assertNotIn(ck, fish_mod._in_flight,
                             "the in-flight entry is keyed by identity and must "
                             "be released (it used to be popped by text)")
            self.assertIn(ck, fish_mod._audio_cache)

    def test_the_decoded_path_uses_a_format_scoped_identity_key(self):
        audio = MagicMock()
        audio.set_frame_rate.return_value = audio
        audio.set_channels.return_value = audio
        audio.raw_data = b"\x00\x01" * 32
        with patch.object(fish_mod, "_request_audio_playable", return_value=audio):
            fish_mod._fetch_audio("decoded sentence")
        expected = audio_actor.audio_cache_key(
            fish_mod._resolve_tts_model(), fish_mod.FISH_REFERENCE_ID, "mp3",
            "decoded sentence")
        self.assertIn(expected, fish_mod._audio_cache)
        # ...and it must never collide with the raw-PCM slot for the same text.
        self.assertNotEqual(expected, fish_mod._pcm_cache_key("decoded sentence"))

    def test_the_stall_recovery_branch_does_not_raise(self):
        """`_in_flight.pop(ck)` referenced an undefined name before F32."""
        text = "stalled fetch"
        ck = audio_actor.audio_cache_key(
            fish_mod._resolve_tts_model(), fish_mod.FISH_REFERENCE_ID, "mp3", text)

        class _StalledOwner:
            """The prefetch owner that never finishes (times out immediately)."""

            def wait(self, timeout=None):
                return False

        fish_mod._in_flight[ck] = _StalledOwner()
        audio = MagicMock()
        with patch.object(fish_mod, "_request_audio_playable", return_value=audio), \
             patch.object(fish_mod, "_boost_volume", side_effect=lambda a: a), \
             patch.object(fish_mod, "_decode_audio", return_value=audio):
            result = fish_mod._fetch_audio(text)
        self.assertIs(result, audio)
        self.assertNotIn(ck, fish_mod._in_flight)


class PortAudioContractStream:
    """Fake output stream that enforces PortAudio's real start() contract.

    ``sd.OutputStream`` opens a BLOCKING stream in the stopped state, and
    ``Pa_WriteStream`` answers ``paStreamIsStopped`` (-9983) until ``start()``
    is called. sounddevice only ever calls ``start()`` from its ``with``
    statement, never from ``__init__``. A wrapper that forgets to start its
    stream therefore fails on the FIRST write and no audio is heard at all —
    whatever engine produced the PCM. This fake reproduces that contract so
    the regression cannot come back silently.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.start_calls = 0
        self.stopped = False
        self.closed = False
        self.writes = []

    def start(self):
        self.start_calls += 1
        self.started = True

    def write(self, data):
        if not self.started:
            raise RuntimeError("Stream is stopped [PaErrorCode -9983]")
        self.writes.append(bytes(np.asarray(data).tobytes()))

    def stop(self):
        self.stopped = True
        self.started = False

    def close(self):
        self.closed = True

    def played(self):
        return b"".join(self.writes)


def _fake_sounddevice(created, cls=PortAudioContractStream):
    """A stand-in ``sounddevice`` module whose OutputStream enforces start()."""
    module = types.ModuleType("sounddevice")

    def OutputStream(**kwargs):
        stream = cls(**kwargs)
        created.append(stream)
        return stream

    module.OutputStream = OutputStream
    return module


class StreamStartContractTests(unittest.TestCase):
    """F32: the owner must START the device stream before it writes to it."""

    def setUp(self):
        # Never inherit a factory injected by another test.
        audio_actor.set_stream_factory(None)

    def tearDown(self):
        audio_actor.set_stream_factory(None)

    def _play(self, actor, pcm, key="k", **play_kwargs):
        generation = actor.begin(key)
        self.assertTrue(actor.feed_chunk(key, generation, pcm))
        actor.end_utterance(key, generation)
        return actor.play(**play_kwargs)

    def test_default_stream_is_started_before_the_first_write(self):
        created = []
        pcm = b"\x00\x10" * 1024
        actor = audio_actor.AudioActor()

        with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
            written = self._play(actor, pcm)

        self.assertEqual(written, len(pcm),
                         "the whole utterance must reach the device")
        self.assertTrue(created, "a device stream must have been opened")
        # Assert on start_calls, not the `started` flag: the play loop's
        # finally block stops the stream, and stopping legitimately clears
        # `started`. What matters is that start() WAS called before writes.
        self.assertGreater(created[0].start_calls, 0,
                           "the stream must be started, or PortAudio rejects writes")
        self.assertEqual(created[0].played(), pcm)

    def test_format_factory_stream_is_started_before_the_first_write(self):
        created = []
        pcm = b"\x00\x10" * 1024
        actor = audio_actor.AudioActor()

        with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
            written = self._play(
                actor, pcm,
                stream_factory=audio_actor.make_sounddevice_factory(
                    device=None, samplerate=44100, channels=1))

        self.assertEqual(written, len(pcm))
        self.assertTrue(created, "a device stream must have been opened")
        self.assertGreater(created[0].start_calls, 0,
                           "the format-specific factory must start its stream too")
        self.assertEqual(created[0].played(), pcm)

    def test_a_caller_supplied_device_is_released_after_the_utterance(self):
        """A device handed TO the actor is still torn down (F32).

        [P0-07] The process-wide default device is deliberately long-lived now
        and is covered by the test below. What must NOT change is that a device
        the caller owns is released the moment the utterance ends.
        """
        created = []
        actor = audio_actor.AudioActor()

        with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
            self._play(actor, b"\x00\x10" * 256,
                       stream_factory=audio_actor.make_sounddevice_factory())

        self.assertTrue(created[0].stopped and created[0].closed,
                        "a stream that is opened must also be stopped and closed")

    def test_the_process_device_is_opened_once_and_released_by_shutdown(self):
        """[P0-07] ONE device per process, not one per sentence or reply.

        This is the P0-07 assertion that replaces the old per-sentence teardown
        check for the default device: opening the speaker per sentence cost a
        measured ~400ms hole between sentences of a reply.
        """
        created = []
        actor = audio_actor.AudioActor()

        with patch.dict(sys.modules, {"sounddevice": _fake_sounddevice(created)}):
            for _ in range(3):                 # three sentences, one reply
                self._play(actor, b"\x00\x10" * 256)
            self.assertEqual(len(created), 1,
                             "the device must be opened once, not per sentence")
            self.assertFalse(created[0].closed,
                             "the process device must outlive the sentence")
            actor.shutdown()

        self.assertTrue(created[0].closed,
                        "shutdown must release the long-lived device")

    def test_a_write_lost_to_an_abort_is_not_reported_as_an_error(self):
        # Barge-in: the stop lands between the generation check and the write,
        # leaving the stream stopped underneath the loop. That is an
        # intentional abort and must end the utterance quietly.
        created = []
        holder = {}

        class AbortsOnWrite(PortAudioContractStream):
            def write(self, data):
                holder["actor"].abort()
                raise RuntimeError("Stream is stopped [PaErrorCode -9983]")

        actor = audio_actor.AudioActor()
        holder["actor"] = actor

        with patch.dict(sys.modules,
                        {"sounddevice": _fake_sounddevice(created, AbortsOnWrite)}):
            written = self._play(actor, b"\x00\x10" * 1024)

        self.assertEqual(written, 0, "an aborted utterance plays nothing further")


if __name__ == "__main__":
    unittest.main()
