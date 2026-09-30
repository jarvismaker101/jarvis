"""[P0-05] Prefetch must never make the next sentence wait for synthesis.

The defect: ``StreamSpeaker._enqueue`` prefetched EVERY sentence including the
one about to play. The prefetch claims the in-flight slot, so the playback
thread became a mere waiter and could only start after synthesis had FINISHED —
and the prefetch usually won, because the playback worker first reads the model
registry and starts an earcon thread. Sentence 1 therefore lost streaming
playback, and the first thing the user heard was delayed by a full synthesis.

The fix has two halves, and this file tests both:

* the in-flight entry is JOINABLE — a player reads the growing buffer as bytes
  arrive instead of waiting for the last one;
* ``_enqueue`` never prefetches the sentence that will play next while the
  worker is idle.

Every test drives fakes: no network, no real device, nothing audible.
"""

import contextlib
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from backend.services import audio_actor, fish_voice as fish_mod
from backend.services import voice as voice_mod

TEXT = "Sentence one is long enough to be its own utterance."
PCM_A = b"\x11\x00" * 1024        # 2048 bytes: even, so no residual stitching
PCM_B = b"\x22\x00" * 1024


def _wait_until(predicate, timeout=3.0, interval=0.005):
    """Poll *predicate* until true; returns its final value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class _StreamedResponse:
    """Streams PCM chunks, optionally holding the tail behind a gate.

    The gate is what reproduces the race: while it is closed the synthesis is
    demonstrably NOT finished, so anything the player does in that window
    cannot have been caused by synthesis completing.
    """

    def __init__(self, chunks, release=None, gate_after=1):
        self.status_code = 200
        self._chunks = list(chunks)
        self._release = release
        self._gate_after = gate_after
        self.closed = False

    def iter_content(self, chunk_size=None):
        for index, chunk in enumerate(self._chunks):
            if self._release is not None and index == self._gate_after:
                self._release.wait(10.0)
            yield chunk

    def close(self):
        self.closed = True


class _HangingResponse:
    """A synthesis that never produces a byte until released."""

    status_code = 200

    def __init__(self, release):
        self._release = release

    def iter_content(self, chunk_size=None):
        self._release.wait(10.0)
        return iter(())

    def close(self):
        pass


class _FakeOutput:
    """Records what reached the device, timestamped."""

    def __init__(self, recorder, **kwargs):
        self._recorder = recorder
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def ensure_started(self):
        if not self.started:
            self.start()

    def write(self, data):
        if not self.started:
            raise RuntimeError("Stream is stopped [PaErrorCode -9983]")
        self._recorder.append((time.monotonic(), bytes(np.asarray(data).tobytes())))

    def stop(self):
        self.started = False

    def abort(self):
        self.started = False

    def close(self):
        self.closed = True


def _fake_sounddevice(recorder):
    module = types.ModuleType("sounddevice")
    module.OutputStream = lambda **kwargs: _FakeOutput(recorder, **kwargs)
    return module


class _Harness:
    """Patches the TTS surface, the device and the session for one test."""

    def __init__(self, response):
        self._response = response
        self.writes = []
        self.posts = 0

    def __enter__(self):
        self._stack = contextlib.ExitStack()
        stack = self._stack
        stack.enter_context(patch.object(fish_mod, "FISH_API_KEY", "test-key"))
        stack.enter_context(patch.object(fish_mod, "_boost_pcm_chunk",
                                         side_effect=lambda chunk: chunk))
        stack.enter_context(patch.dict(sys.modules,
                                       {"sounddevice": _fake_sounddevice(self.writes)}))
        stack.enter_context(patch.object(fish_mod._session, "post",
                                         side_effect=self._post))
        # A device cached by an earlier test belongs to that test's binding.
        audio_actor.shutdown_actor()
        fish_mod._audio_cache.clear()
        fish_mod._in_flight.clear()
        return self

    def _post(self, *args, **kwargs):
        self.posts += 1
        return self._response

    def __exit__(self, *exc):
        try:
            fish_mod._audio_cache.clear()
            fish_mod._in_flight.clear()
        finally:
            audio_actor.shutdown_actor()
            return self._stack.__exit__(*exc)



class PrefetchRaceRegressionTests(unittest.TestCase):
    """The regression: a prefetch that WINS must not delay sentence 1."""

    def test_a_prefetch_that_wins_the_race_still_plays_before_synthesis_ends(self):
        """MUST FAIL BEFORE THE FIX.

        The prefetch claims the in-flight slot first, exactly as it does in
        production. Playback has to begin while the tail of the synthesis is
        still gated shut.
        """
        release = threading.Event()
        response = _StreamedResponse([PCM_A, PCM_B], release, gate_after=1)
        played = {}

        try:
            with _Harness(response) as harness:
                # The prefetch thread wins the race for the in-flight slot.
                fish_mod.prefetch_fish_audio(TEXT)
                self.assertTrue(
                    _wait_until(lambda: fish_mod._pcm_cache_key(TEXT)
                                in fish_mod._in_flight),
                    "the prefetch must have claimed the in-flight slot")

                def _play():
                    played["result"] = fish_mod._do_pcm_stream(TEXT, play=True)

                player = threading.Thread(target=_play, daemon=True)
                player.start()

                # Synthesis is STILL gated, so a device write here can only
                # have come from streaming the first chunk.
                self.assertTrue(
                    _wait_until(lambda: bool(harness.writes), timeout=2.0),
                    "playback must start from the first chunk, before the "
                    "synthesis has finished")

                release.set()
                player.join(5.0)
        finally:
            release.set()

        self.assertTrue(played.get("result"), "the joined sentence must play")
        streamed = b"".join(chunk for _ts, chunk in harness.writes)
        self.assertEqual(streamed, PCM_A + PCM_B,
                         "the whole sentence must reach the device, once")

    def test_only_one_synthesis_is_requested_for_the_joined_sentence(self):
        """Losing the race must still not buy a second TTS call.

        The synthesis is gated open only after the player has already started,
        so "the prefetch is in flight when the player arrives" is deterministic
        rather than a race the test can lose.
        """
        release = threading.Event()
        response = _StreamedResponse([PCM_A, PCM_B], release, gate_after=1)
        played = {}
        try:
            with _Harness(response) as harness:
                fish_mod.prefetch_fish_audio(TEXT)
                self.assertTrue(_wait_until(
                    lambda: fish_mod._pcm_cache_key(TEXT) in fish_mod._in_flight))

                def _play():
                    played["result"] = fish_mod._do_pcm_stream(TEXT, play=True)

                player = threading.Thread(target=_play, daemon=True)
                player.start()
                self.assertTrue(_wait_until(lambda: bool(harness.writes)),
                                "the joined sentence must start playing")
                release.set()
                player.join(5.0)
        finally:
            release.set()

        self.assertTrue(played.get("result"))
        self.assertEqual(harness.posts, 1,
                         "the prefetch and the player must share one synthesis")


class JoinableInflightTests(unittest.TestCase):
    """The mechanism itself: bytes are readable as they arrive."""

    def test_a_joiner_reads_bytes_before_the_synthesis_finishes(self):
        inflight = fish_mod._InflightPCM()
        received = []
        done = threading.Event()

        def _consume():
            for chunk in inflight.iter_from(0, timeout=5.0):
                received.append(chunk)
            done.set()

        consumer = threading.Thread(target=_consume, daemon=True)
        consumer.start()

        inflight.publish(PCM_A)
        # The joiner must see the FIRST chunk while the owner is still working.
        self.assertTrue(_wait_until(lambda: received == [PCM_A]),
                        "the first chunk must be playable before publish #2")

        inflight.publish(PCM_B)
        self.assertTrue(_wait_until(lambda: received == [PCM_A, PCM_B]))
        self.assertFalse(done.is_set(), "the synthesis is not finished yet")

        inflight.finish()
        self.assertTrue(done.wait(2.0), "finish() must release the joiner")

    def test_wait_is_event_shaped_so_the_mp3_path_still_works(self):
        """_in_flight holds this object for PCM and a plain Event for MP3."""
        inflight = fish_mod._InflightPCM()
        self.assertFalse(inflight.wait(timeout=0.05),
                         "an unfinished synthesis must time out, not hang")
        inflight.finish()
        self.assertTrue(inflight.wait(timeout=0.05))
        self.assertTrue(inflight.is_done())

    def test_a_hung_synthesis_times_out_instead_of_hanging_its_joiner(self):
        release = threading.Event()
        response = _HangingResponse(release)
        try:
            with patch.object(fish_mod, "_INFLIGHT_JOIN_TIMEOUT_SECONDS", 0.3):
                with _Harness(response) as harness:
                    owner = threading.Thread(
                        target=fish_mod.prefetch_fish_audio, args=(TEXT,),
                        daemon=True)
                    owner.start()
                    self.assertTrue(_wait_until(
                        lambda: fish_mod._pcm_cache_key(TEXT)
                        in fish_mod._in_flight))

                    started = time.monotonic()
                    result = fish_mod._do_pcm_stream(TEXT, play=True)
                    elapsed = time.monotonic() - started

            self.assertFalse(result, "a hung synthesis cannot be reported as played")
            self.assertLess(elapsed, 5.0,
                            "the joiner must give up on its own deadline")
            self.assertEqual(harness.writes, [], "nothing reached the device")
        finally:
            release.set()



    def test_a_playing_owner_is_not_streamed_alongside_by_a_second_player(self):
        """Two players on one sentence would speak it twice.

        A second ``play=True`` caller keeps the documented wait-then-replay
        behaviour when the owner is itself feeding the device.
        """
        inflight = fish_mod._InflightPCM()
        inflight.playing = True
        inflight.publish(PCM_A)
        inflight.finish()
        ck = fish_mod._pcm_cache_key(TEXT)
        with _Harness(_StreamedResponse([PCM_A])):
            with patch.object(fish_mod, "_play_pcm_through_actor",
                              return_value=True) as replay:
                with fish_mod._cache_lock:
                    fish_mod._in_flight[ck] = inflight
                    fish_mod._audio_cache[ck] = PCM_A
                with patch.object(fish_mod, "_play_inflight_stream") as join:
                    result = fish_mod._do_pcm_stream(TEXT, play=True)

        self.assertTrue(result)
        join.assert_not_called()
        replay.assert_called_once()

    def test_the_in_flight_map_stays_bounded(self):
        """Constraint: joined buffers must not grow the map without limit."""
        with fish_mod._cache_lock:
            fish_mod._in_flight.clear()
        for index in range(fish_mod._INFLIGHT_MAX + 5):
            entry, _owner = fish_mod._claim_inflight("bounded-%d" % index)
            entry.finish()
        with fish_mod._cache_lock:
            size = len(fish_mod._in_flight)
            fish_mod._in_flight.clear()
        self.assertLessEqual(size, fish_mod._INFLIGHT_MAX + 1,
                             "completed entries must be evicted oldest-first")


class StreamingPrefetchRuleTests(unittest.TestCase):
    """Rule (1): never prefetch the sentence that will play next, while idle.

    [P1-09] The rule now lives in `_prefetch_tts_audio`, so these tests let the
    REAL helper run and intercept at the engine level: what is asserted is that
    no engine synthesis is started, not which wrapper was called.
    """

    def _speaker(self):
        speaker = voice_mod.StreamSpeaker()
        # Deterministic: no worker thread may pull the item behind our back.
        speaker._start_worker = lambda: None
        return speaker

    def test_the_first_sentence_is_never_prefetched_while_the_worker_is_idle(self):
        speaker = self._speaker()
        with patch.object(voice_mod, "_resolve_tts_provider", return_value="fish"), \
                patch.object(voice_mod, "prefetch_fish_audio") as fish_prefetch:
            speaker._enqueue("Sentence one.")
            fish_prefetch.assert_not_called()

            # The worker is now speaking sentence one, so sentence two cannot
            # be the next thing heard and is safe to synthesise ahead.
            with patch.object(voice_mod, "is_speaking", True):
                speaker._enqueue("Sentence two.")

        fish_prefetch.assert_called_once_with("Sentence two.")

    def test_a_sentence_with_one_queued_ahead_is_still_prefetched(self):
        speaker = self._speaker()
        with patch.object(voice_mod, "_resolve_tts_provider", return_value="fish"), \
                patch.object(voice_mod, "prefetch_fish_audio") as fish_prefetch:
            speaker._queue.put("already queued")
            speaker._enqueue("Sentence two.")

        fish_prefetch.assert_called_once_with("Sentence two.")

    def test_the_first_chunk_split_path_never_prefetches_the_piece_that_plays_first(self):
        """The split path has two layers of "next", and the rule must hold.

        The first piece plays immediately, so it must not be raced by a
        prefetch. The remainder is enqueued *behind* it, so warming those chunks
        is both safe and the point of the prefetch.
        """
        speaker = self._speaker()
        long_text = "First clause, " + ("filler words " * 12) + "and the end."
        with patch.object(voice_mod, "_resolve_tts_provider", return_value="fish"), \
                patch.object(voice_mod, "prefetch_fish_audio") as fish_prefetch:
            speaker._enqueue(long_text)

        first_piece = speaker._queue.queue[0]
        warmed = [call.args[0] for call in fish_prefetch.call_args_list]
        self.assertNotIn(first_piece, warmed,
                         "the piece about to play must not be prefetched")
        self.assertTrue(warmed,
                        "chunks queued BEHIND the first piece are still warmed")


if __name__ == "__main__":
    unittest.main()
