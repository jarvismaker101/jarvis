"""[P0-06] The audio actor wakes on a chunk instead of polling every 250ms.

The bug: ``feed_chunk`` appended to the ring and set ``_chunk_wait`` - an event
the play loop NEVER waits on. The loop instead slept in 250ms slices on
``_stop``, which only ``abort()`` ever sets. So the first chunk of an utterance
could sit in the ring for up to a quarter second before it reached the device,
once per utterance, on top of whatever synthesis cost had already been paid.

The fix is to wait on the ring condition variable that producers already
notify, with a short floor purely so a missed notify cannot park the loop.

No microphone, no real audio device and no TTS engine is opened here: the
stream is a recording fake, exactly like the F32 suite.
"""

import threading
import time
import unittest

from backend.services import audio_actor


class RecordingStream:
    """Output-stream fake that records writes and when they arrived."""

    def __init__(self, slow=0.0):
        self.written = []
        self.write_times = []
        self.started = False
        self.stopped = False
        self.closed = False
        self._slow = slow
        self._lock = threading.Lock()

    def start(self):
        self.started = True

    def write(self, pcm_bytes):
        if self._slow:
            time.sleep(self._slow)
        with self._lock:
            self.written.append(bytes(pcm_bytes))
            self.write_times.append(time.monotonic())

    def count(self):
        with self._lock:
            return len(self.written)

    def played(self):
        with self._lock:
            return b"".join(self.written)

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


def _wait_for(predicate, timeout=2.0, interval=0.001):
    """Spin until *predicate* is true. Returns True when it became true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class WakeOnChunkTests(unittest.TestCase):
    """A chunk that lands wakes the player; there is no 250ms timer."""

    def _start_player(self, actor, stream):
        """Run play() on its own thread and wait until it is really waiting.

        The loop is considered parked once the stream exists and nothing has
        been written for a while: an empty ring with no EOF is exactly the
        state that used to sleep.
        """
        created = threading.Event()

        def _factory():
            created.set()
            return stream

        actor._factory = _factory
        done = {}

        def _run():
            try:
                done["written"] = actor.play()
            except Exception as exc:      # pragma: no cover - diagnostic
                done["error"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        self.assertTrue(created.wait(2.0), "play() never created a stream")
        time.sleep(0.1)
        self.assertEqual(stream.count(), 0,
                         "nothing to play, so nothing may be written yet")
        return thread, done

    def test_a_fed_chunk_reaches_the_device_far_faster_than_the_old_tick(self):
        """The regression: wake latency must be nowhere near 250ms.

        Five trials, best one wins, so scheduler jitter on a busy box cannot
        make this flaky. Before the fix every trial sat on the 250ms tick.
        """
        stream = RecordingStream()
        actor = audio_actor.AudioActor(stream_factory=lambda: stream)
        key = "utt"
        generation = actor.begin(key)
        thread, done = self._start_player(actor, stream)
        try:
            latencies = []
            for trial in range(5):
                # Let the loop settle back into its wait, so every trial
                # measures a genuine "parked consumer, then a chunk arrives"
                # wake instead of catching it mid-iteration. Without this the
                # first trial can be fast even on the old 250ms tick, which
                # would let the regression slip through.
                time.sleep(0.1)
                payload = bytes([trial + 1]) * 64
                index = stream.count()
                started = time.monotonic()
                self.assertTrue(actor.feed_chunk(key, generation, payload))
                self.assertTrue(
                    _wait_for(lambda: stream.count() > index, timeout=2.0),
                    "the chunk never reached the device")
                # The stream timestamps its own writes from the play thread, so
                # this is feed -> device write and not the test's own polling
                # granularity (Windows rounds small sleeps up to ~15.6ms, which
                # would otherwise mask the number we care about).
                latencies.append(stream.write_times[index] - started)

            best = min(latencies)
            self.assertLess(
                best, audio_actor.PLAY_WAIT_FLOOR_SECONDS,
                "feed->device took %.1fms; the notify is not waking the "
                "player (latencies: %sms)"
                % (best * 1000,
                   [round(x * 1000, 1) for x in latencies]))
            # And nothing on the audio path polls on the old 250ms interval.
            self.assertLess(best, 0.25)

            actor.end_utterance(key, generation)
            thread.join(2.0)
            self.assertFalse(thread.is_alive())
            self.assertNotIn("error", done)
            self.assertEqual(len(stream.played()), 5 * 64)
        finally:
            actor.abort()
            thread.join(2.0)

    def test_every_fed_chunk_is_written_in_order(self):
        stream = RecordingStream()
        actor = audio_actor.AudioActor(stream_factory=lambda: stream)
        key = "ordered"
        generation = actor.begin(key)
        thread, _ = self._start_player(actor, stream)
        try:
            chunks = [bytes([index]) * 32 for index in range(1, 6)]
            for chunk in chunks:
                self.assertTrue(actor.feed_chunk(key, generation, chunk))
            self.assertTrue(_wait_for(lambda: stream.count() >= 5,
                                      timeout=2.0))
            self.assertEqual(stream.played(), b"".join(chunks))
        finally:
            actor.abort()
            thread.join(2.0)

    def test_the_wait_floor_is_short(self):
        """A floor is kept (missed-notify safety) but it must not be a tick."""
        self.assertLessEqual(audio_actor.PLAY_WAIT_FLOOR_SECONDS, 0.05)



class TerminationTests(unittest.TestCase):
    """The loop still exits promptly on every stop path."""

    def _parked_player(self):
        stream = RecordingStream()
        actor = audio_actor.AudioActor(stream_factory=lambda: stream)
        generation = actor.begin("utt")
        done = {}

        def _run():
            done["written"] = actor.play()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        time.sleep(0.1)
        self.assertEqual(stream.count(), 0)
        return actor, stream, generation, thread, done

    def test_abort_terminates_a_waiting_player_promptly(self):
        actor, stream, _gen, thread, _done = self._parked_player()
        started = time.monotonic()
        cursor = actor.abort()
        thread.join(2.0)
        elapsed = time.monotonic() - started
        self.assertFalse(thread.is_alive(), "abort left the player parked")
        self.assertLess(elapsed, 0.1,
                        "abort took %.1fms to stop the loop" % (elapsed * 1000))
        self.assertEqual(cursor, 0)
        self.assertTrue(stream.closed, "the finally block must still run")

    def test_end_utterance_with_an_empty_ring_terminates_the_player(self):
        actor, stream, generation, thread, done = self._parked_player()
        self.assertTrue(actor.end_utterance("utt", generation))
        thread.join(2.0)
        self.assertFalse(thread.is_alive(),
                         "EOF with an empty ring left the player parked")
        self.assertEqual(done.get("written"), 0)
        self.assertTrue(stream.stopped)
        self.assertTrue(stream.closed)

    def test_abort_mid_stream_stops_before_the_next_chunk_is_heard(self):
        stream = RecordingStream(slow=0.02)
        actor = audio_actor.AudioActor(stream_factory=lambda: stream)
        key = "cut"
        generation = actor.begin(key)
        thread = threading.Thread(target=actor.play, daemon=True)
        thread.start()
        for index in range(10):
            actor.feed_chunk(key, generation, bytes([index]) * 64, timeout=2.0)
        # Abort while the ring is still being drained by a slow device.
        actor.abort()
        thread.join(2.0)
        self.assertFalse(thread.is_alive())
        self.assertLess(len(stream.played()), 10 * 64,
                        "an abort must not let the whole buffer play out")
        self.assertTrue(stream.closed)


class BackpressurePreservedTests(unittest.TestCase):
    """P0-06 must not touch the producer-side backpressure contract."""

    def test_a_full_ring_makes_the_producer_wait_instead_of_dropping(self):
        stream = RecordingStream()
        actor = audio_actor.AudioActor(stream_factory=lambda: stream,
                                       ring_chunks=2)
        key = "wait"
        generation = actor.begin(key)
        self.assertTrue(actor.feed_chunk(key, generation, b"\x01" * 16))
        self.assertTrue(actor.feed_chunk(key, generation, b"\x02" * 16))

        # The ring is full and nothing is consuming it yet. Start the player
        # shortly after, and the producer must WAIT for room, then succeed.
        result = {}

        def _feed():
            result["ok"] = actor.feed_chunk(key, generation, b"\x03" * 16,
                                            timeout=5.0)

        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()
        time.sleep(0.1)
        self.assertNotIn("ok", result, "the producer should still be waiting")
        player = threading.Thread(target=actor.play, daemon=True)
        player.start()
        feeder.join(5.0)
        self.assertTrue(result.get("ok"),
                        "backpressure must wait for room, never drop")
        self.assertEqual(actor.dropped_chunks(), 0)
        actor.abort()
        player.join(2.0)

    def test_a_producer_that_cannot_be_served_is_still_told(self):
        actor = audio_actor.AudioActor(stream_factory=RecordingStream,
                                       ring_chunks=2)
        key = "stuck"
        generation = actor.begin(key)
        self.assertTrue(actor.feed_chunk(key, generation, b"\x01" * 8))
        self.assertTrue(actor.feed_chunk(key, generation, b"\x02" * 8))
        # No consumer at all: the ring stays full for the whole timeout.
        self.assertFalse(actor.feed_chunk(key, generation, b"\x03" * 8,
                                          timeout=0.05))
        self.assertEqual(actor.dropped_chunks(), 1)
        self.assertIn("ring full", actor.last_error() or "")


class PreservedContractsTests(unittest.TestCase):
    """Everything P0-06 must NOT have changed."""

    def test_the_on_chunk_callback_still_fires_once_per_chunk_in_order(self):
        """The AEC reference hook (P0-05 / P1-19) must keep working."""
        stream = RecordingStream()
        actor = audio_actor.AudioActor(stream_factory=lambda: stream)
        key = "hook"
        generation = actor.begin(key)
        chunks = [bytes([index]) * 16 for index in range(1, 4)]
        for chunk in chunks:
            self.assertTrue(actor.feed_chunk(key, generation, chunk))
        actor.end_utterance(key, generation)
        seen = []
        actor.play(on_chunk=lambda pcm: seen.append(bytes(pcm)))
        self.assertEqual(seen, chunks)

    def test_a_stale_chunk_that_reached_the_ring_is_still_discarded(self):
        """The F32 re-check immediately before the device write must survive."""
        stream = RecordingStream()
        actor = audio_actor.AudioActor(stream_factory=lambda: stream)
        generation = actor.begin("live")
        # Bypass feed_chunk: put an OLD-generation chunk into the ring by hand.
        # That is exactly what the pre-write re-check exists to catch.
        with actor._ring_cv:
            actor._ring.append(
                audio_actor.Chunk("live", generation - 1, b"\xaa" * 32))
        self.assertTrue(actor.feed_chunk("live", generation, b"\x11" * 32))
        actor.end_utterance("live", generation)
        written = actor.play()
        self.assertNotIn(b"\xaa" * 32, stream.played(),
                         "a stale-generation chunk was written")
        self.assertEqual(written, 32)

    def test_the_cursor_counts_only_bytes_the_device_took(self):
        stream = RecordingStream()
        actor = audio_actor.AudioActor(stream_factory=lambda: stream)
        key = "cursor"
        generation = actor.begin(key)
        for _ in range(3):
            self.assertTrue(actor.feed_chunk(key, generation, b"\x01" * 100))
        self.assertEqual(actor.spoke_bytes(), 0,
                         "nothing has reached the device yet")
        actor.end_utterance(key, generation)
        written = actor.play()
        self.assertEqual(written, 300)
        self.assertEqual(actor.spoke_bytes(), 300)
        self.assertEqual(actor.cursor(), 300)

    def test_a_new_utterance_still_discards_the_previous_ring(self):
        stream = RecordingStream()
        actor = audio_actor.AudioActor(stream_factory=lambda: stream)
        first = actor.begin("one")
        self.assertTrue(actor.feed_chunk("one", first, b"\x01" * 32))
        second = actor.begin("two")
        self.assertFalse(actor.feed_chunk("one", first, b"\x02" * 32))
        self.assertTrue(actor.feed_chunk("two", second, b"\x03" * 32))
        actor.end_utterance("two", second)
        actor.play()
        self.assertNotIn(b"\x01" * 32, stream.played())
        self.assertIn(b"\x03" * 32, stream.played())


class LockingStressTests(unittest.TestCase):
    """The locking discipline must hold under concurrency, not just in theory.

    Producers take the ring lock and then the generation lock (via
    ``_is_current`` inside the backpressure wait); the play loop now holds the
    ring lock while waiting. Nothing in this change ever takes the generation
    lock and then waits on the ring condition, so the two orders never meet.
    This drives them against each other to prove it.
    """

    def test_concurrent_producers_and_a_consumer_do_not_deadlock(self):
        for attempt in range(5):
            stream = RecordingStream(slow=0.001)
            actor = audio_actor.AudioActor(stream_factory=lambda: stream,
                                           ring_chunks=4)
            key = "stress-%d" % attempt
            generation = actor.begin(key)
            player = threading.Thread(target=actor.play, daemon=True)
            player.start()
            errors = []

            def _produce():
                try:
                    for index in range(30):
                        actor.feed_chunk(key, generation,
                                         bytes([index % 251]) * 64, timeout=5.0)
                except Exception as exc:      # pragma: no cover - diagnostic
                    errors.append(exc)

            producers = [threading.Thread(target=_produce, daemon=True)
                         for _ in range(3)]
            for producer in producers:
                producer.start()
            for producer in producers:
                producer.join(10.0)
            actor.end_utterance(key, generation)
            player.join(10.0)

            self.assertEqual(errors, [])
            self.assertFalse(player.is_alive(),
                             "the play loop deadlocked under concurrency")
            self.assertEqual(actor.dropped_chunks(), 0)

    def test_abort_racing_a_producer_never_deadlocks(self):
        for attempt in range(5):
            stream = RecordingStream(slow=0.002)
            actor = audio_actor.AudioActor(stream_factory=lambda: stream,
                                           ring_chunks=4)
            key = "race-%d" % attempt
            generation = actor.begin(key)
            player = threading.Thread(target=actor.play, daemon=True)
            player.start()

            def _produce():
                for index in range(20):
                    actor.feed_chunk(key, generation, bytes([index]) * 64,
                                     timeout=1.0)

            producer = threading.Thread(target=_produce, daemon=True)
            producer.start()
            time.sleep(0.01)
            actor.abort()
            producer.join(5.0)
            player.join(5.0)
            self.assertFalse(producer.is_alive(), "a producer deadlocked")
            self.assertFalse(player.is_alive(), "the play loop deadlocked")
            self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()

