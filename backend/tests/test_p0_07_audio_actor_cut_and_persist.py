"""[P0-07] Cut instead of drain, and keep ONE output device per process.

Two independent defects, both measured on the real default device before the
change:

* ``abort()`` called ``stream.stop()``. PortAudio's ``stop()`` WAITS for
  everything already buffered, so a barge-in kept playing the tail of the
  sentence and blocked the caller: measured 221ms p50 (270ms in an earlier
  pass). ``stream.abort()`` discards the buffer: measured 10.7ms.
* ``play()`` opened and closed the speaker per sentence, and the first write to
  a cold device paid a long warm-up: measured 25ms open + ~180ms first write,
  leaving a ~400ms hole between sentences of one reply. One long-lived device
  measured a ~10ms per-sentence cost.

Everything here drives fake devices: no real speaker is opened, and no audio
leaves the machine.
"""

import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from backend.services import audio_actor


class FakeOutput:
    """Records which primitive was used, and enforces the start() contract."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.start_calls = 0
        self.stopped = False
        self.aborted = False
        self.closed = False
        self.writes = []
        self.fail_write_from = None

    def start(self):
        self.start_calls += 1
        self.started = True

    def ensure_started(self):
        """The actor's reuse contract: never write into a stopped device."""
        if not self.started:
            self.start()

    def write(self, data):
        if not self.started:
            raise RuntimeError("Stream is stopped [PaErrorCode -9983]")
        if (self.fail_write_from is not None
                and len(self.writes) >= self.fail_write_from):
            raise RuntimeError("device vanished")
        self.writes.append(bytes(np.asarray(data).tobytes()))

    def stop(self):
        self.stopped = True
        self.started = False

    def abort(self):
        self.aborted = True
        self.started = False

    def close(self):
        self.closed = True


class FakeStreamWithoutAbort(FakeOutput):
    """A stream with no abort(): the fallback path must still work."""

    abort = None


def fake_sounddevice(created, cls=FakeOutput):
    module = types.ModuleType("sounddevice")

    def OutputStream(**kwargs):
        stream = cls(**kwargs)
        created.append(stream)
        return stream

    module.OutputStream = OutputStream
    return module


def _play(actor, pcm, key="k", **play_kwargs):
    generation = actor.begin(key)
    assert actor.feed_chunk(key, generation, pcm)
    actor.end_utterance(key, generation)
    return actor.play(**play_kwargs)


class AbortCutsInsteadOfDrainingTests(unittest.TestCase):
    """PART 1: a barge-in cuts the device, never drains it."""

    def test_abort_reaches_the_stream_as_abort_not_stop(self):
        created = []
        actor = audio_actor.AudioActor()

        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice(created)}):
            generation = actor.begin("cut")
            self.assertTrue(actor.feed_chunk("cut", generation, b"\x00\x10" * 64))
            player = threading.Thread(target=actor.play, daemon=True)
            player.start()
            time.sleep(0.05)
            actor.abort()
            player.join(2.0)

        self.assertTrue(created[0].aborted,
                        "stop() drains the buffer; abort() must be used")
        self.assertFalse(created[0].stopped,
                         "the drain path must not be the one taken")

    def test_stop_is_the_fallback_when_the_stream_has_no_abort(self):
        created = []
        actor = audio_actor.AudioActor()

        with patch.dict(sys.modules,
                        {"sounddevice": fake_sounddevice(created,
                                                         FakeStreamWithoutAbort)}):
            generation = actor.begin("fallback")
            self.assertTrue(actor.feed_chunk("fallback", generation,
                                             b"\x00\x10" * 64))
            player = threading.Thread(target=actor.play, daemon=True)
            player.start()
            time.sleep(0.05)
            actor.abort()          # must not raise
            player.join(2.0)

        self.assertTrue(created[0].stopped,
                        "a stream without abort() must fall back to stop()")

    def test_abort_on_an_actor_that_never_opened_a_device_is_a_no_op(self):
        actor = audio_actor.AudioActor()
        self.assertEqual(actor.abort(), 0)


class PersistentDeviceTests(unittest.TestCase):
    """PART 2: one process device, reused across sentences and replies."""

    def test_the_factory_is_called_once_across_a_multi_sentence_reply(self):
        created = []
        actor = audio_actor.AudioActor()

        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice(created)}):
            for sentence in range(4):
                _play(actor, b"\x00\x10" * 128, key="s%d" % sentence)
            self.assertEqual(len(created), 1,
                             "one device for the whole reply, not one per sentence")

    def test_the_same_device_is_reused_across_separate_replies(self):
        created = []
        actor = audio_actor.AudioActor()

        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice(created)}):
            _play(actor, b"\x00\x10" * 128, key="reply-1")
            _play(actor, b"\x00\x10" * 128, key="reply-2")
            self.assertEqual(len(created), 1, "the device must outlive a reply")
            self.assertFalse(created[0].closed)
            actor.shutdown()

        self.assertTrue(created[0].closed)

    def test_a_barge_in_then_the_next_sentence_restarts_the_same_device(self):
        """Abort stops a PortAudio stream: the reuse must start it again.

        Getting this wrong means NO audio is ever heard after the first
        barge-in (-9983 on every write), so it is asserted explicitly.
        """
        created = []
        actor = audio_actor.AudioActor()
        pcm = b"\x00\x10" * 64

        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice(created)}):
            generation = actor.begin("first")
            self.assertTrue(actor.feed_chunk("first", generation, pcm))
            player = threading.Thread(target=actor.play, daemon=True)
            player.start()
            time.sleep(0.05)
            actor.abort()
            player.join(2.0)
            self.assertTrue(created[0].aborted)
            self.assertFalse(created[0].started, "abort leaves the device stopped")

            # The next sentence must not write into a stopped stream.
            written = _play(actor, pcm, key="second")

            self.assertEqual(len(created), 1, "no new device was opened")
            self.assertTrue(created[0].started,
                            "the reused device must be started again")
            self.assertEqual(written, len(pcm), "the second sentence must play")
            actor.shutdown()

    def test_shutdown_is_idempotent_and_never_raises(self):
        created = []
        actor = audio_actor.AudioActor()

        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice(created)}):
            _play(actor, b"\x00\x10" * 32)
            actor.shutdown()
            actor.shutdown()            # second call must be harmless

        self.assertTrue(created[0].closed)

    def test_a_caller_supplied_device_is_still_never_reused(self):
        """The explicit stream/factory contract must survive P0-07."""
        created = []
        actor = audio_actor.AudioActor()

        with patch.dict(sys.modules, {"sounddevice": fake_sounddevice(created)}):
            for _ in range(2):
                _play(actor, b"\x00\x10" * 64,
                      stream_factory=audio_actor.make_sounddevice_factory())

        self.assertEqual(len(created), 2,
                         "a caller's own device is per call, as before")
        self.assertTrue(all(stream.closed for stream in created))

    def test_an_injected_factory_is_still_never_reused(self):
        created = []

        def _factory():
            # An injected factory owns the whole lifecycle, including the
            # explicit start() the PortAudio contract requires.
            stream = FakeOutput()
            stream.start()
            created.append(stream)
            return stream

        actor = audio_actor.AudioActor(stream_factory=_factory)
        for _ in range(3):
            _play(actor, b"\x00\x10" * 64)

        self.assertEqual(len(created), 3,
                         "an injected factory owns its own device lifecycle")
        self.assertTrue(all(stream.closed for stream in created))


class ReopenOnDeviceFailureTests(unittest.TestCase):
    """PART 2 requirement 8: one lazy reopen, then surface the failure."""

    def test_a_dead_device_is_reopened_once_and_the_sentence_still_plays(self):
        created = []
        # One-shot across the whole test: the ORIGINAL device dies, the
        # replacement must work - a per-instance counter would fail the retry
        # too and prove nothing.
        state = {"died": False}

        class DiesOnce(FakeOutput):
            def write(self, data):
                if not state["died"]:
                    state["died"] = True
                    raise RuntimeError("device vanished")
                super().write(data)

        pcm = b"\x00\x10" * 64
        actor = audio_actor.AudioActor()

        with patch.dict(sys.modules,
                        {"sounddevice": fake_sounddevice(created, DiesOnce)}):
            written = _play(actor, pcm, key="resilient")

        self.assertEqual(len(created), 2,
                         "exactly one lazy reopen must be attempted")
        self.assertEqual(written, len(pcm),
                         "the sentence must still reach the new device")
        self.assertEqual(b"".join(created[1].writes), pcm)

    def test_a_reopen_that_fails_is_surfaced_through_last_error(self):
        calls = {"count": 0}

        class Dies(FakeOutput):
            def write(self, data):
                raise RuntimeError("device vanished")

        def _factory():
            calls["count"] += 1
            if calls["count"] > 1:
                raise RuntimeError("no output device available")
            stream = Dies()
            stream.start()
            return stream

        actor = audio_actor.AudioActor(stream_factory=_factory)
        actor._owns_device = True          # exercise the persistent path
        actor._factory = _factory

        generation = actor.begin("doomed")
        self.assertTrue(actor.feed_chunk("doomed", generation, b"\x00\x10" * 64))
        actor.end_utterance("doomed", generation)
        with self.assertRaises(Exception):
            actor.play()

        self.assertEqual(calls["count"], 2, "the reopen is attempted exactly once")
        self.assertIn("reopen failed", actor.last_error() or "",
                      "a failed reopen must be visible, never silent")


if __name__ == "__main__":
    unittest.main()

