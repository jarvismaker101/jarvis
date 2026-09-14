"""G10 voice-runtime tests (F32 single playback owner, F33 AEC reference
path, F34 local-agreement stabilization, F36 wake/command separation).

Everything here is hardware-free: sounddevice is faked through
``sys.modules``, the audio actor is driven with its injectable surface, and
no test imports the live listener module (mic side effects).
"""

import sys
import time
import unittest
from unittest import mock

from backend.services import audio_actor, echo_cancel, transcript_stabilizer
from backend.services import wake_engine
from backend.services.audio_actor import AudioActor, audio_cache_key


def _fake_sd_module(recorder):
    """Build a fake `sounddevice` module recording every write."""
    import types

    class FakeOutputStream:
        def __init__(self, samplerate=None, channels=None, dtype=None,
                     device=None, blocksize=None, latency=None):
            self.written = recorder
            self.started = False
            self.stopped = False
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.closed = True
            return False

        def write(self, arr):
            self.written.append(bytes(arr.tobytes()))

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    module = types.ModuleType("sounddevice")
    module.OutputStream = FakeOutputStream
    module.wait = lambda: None

    def _play(samples, samplerate=None, device=None):
        recorder.append(bytes(samples.tobytes()))

    module.play = _play
    return module


class F32CacheIdentityTests(unittest.TestCase):
    def test_same_text_two_references_differ(self):
        k1 = audio_cache_key("m", "ref-1", "pcm", "hello")
        k2 = audio_cache_key("m", "ref-2", "pcm", "hello")
        self.assertNotEqual(k1, k2)

    def test_model_change_differences(self):
        self.assertNotEqual(
            audio_cache_key("m1", "r", "pcm", "hello"),
            audio_cache_key("m2", "r", "pcm", "hello"))

    def test_identical_identity_shares_slot(self):
        self.assertEqual(
            audio_cache_key("m", "r", "pcm", "hello"),
            audio_cache_key("m", "r", "pcm", "hello"))

    def test_fish_key_includes_tts_identity(self):
        from backend.services import fish_voice

        key = fish_voice._pcm_cache_key("hello")
        self.assertEqual(key[2], "pcm")
        self.assertEqual(key[3], "hello")


class F32AudioActorTests(unittest.TestCase):
    def _make_actor(self):
        import threading

        class FakeStream:
            def __init__(self):
                self.writes = []

            def write(self, pcm_bytes):
                self.writes.append(bytes(pcm_bytes))

            def stop(self):
                pass

            def close(self):
                pass

        streams = []

        def factory():
            stream = FakeStream()
            streams.append(stream)
            return stream

        return AudioActor(stream_factory=factory), streams

    def _play_until_write(self, actor, streams, timeout=5.0):
        import threading

        thread = threading.Thread(target=actor.play, daemon=True)
        thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if streams and streams[0].writes:
                break
            time.sleep(0.01)
        return thread

    def test_current_generation_plays(self):
        actor, streams = self._make_actor()
        gen = actor.begin("u1")
        self.assertTrue(actor.feed_chunk("u1", gen, b"\x01\x02"))
        thread = self._play_until_write(actor, streams)
        actor.abort()
        thread.join(timeout=5)
        self.assertEqual(streams[0].writes[-1], b"\x01\x02")
        self.assertGreater(actor.spoke_bytes(), 0)

    def test_stale_generation_discarded(self):
        actor, streams = self._make_actor()
        # Chunk from an un-begun (stale) generation is dropped at the
        # boundary — it can never reach a newer answer's stream.
        self.assertFalse(actor.feed_chunk("u1", 999, b"\x01\x02"))
        self.assertEqual(actor.spoke_bytes(), 0)
        actor.abort()

    def test_begin_bumps_generation_and_clears_utterance(self):
        actor, streams = self._make_actor()
        gen1 = actor.begin("u1")
        actor.feed_chunk("u1", gen1, b"\x01" * 32)
        gen2 = actor.begin("u2")
        self.assertGreater(gen2, gen1)
        self.assertEqual(actor.utterance_pcm(), b"")
        # The old generation's chunks no longer pass the boundary.
        self.assertFalse(actor.feed_chunk("u1", gen1, b"\x03\x04"))
        actor.abort()

    def test_abort_keeps_utterance_for_resume(self):
        actor, streams = self._make_actor()
        gen = actor.begin("u1")
        actor.feed_chunk("u1", gen, b"\x01" * 64)
        actor.feed_chunk("u1", gen, b"\x02" * 64)
        # Deterministic abort before any play loop runs (cursor = 0).
        cursor = actor.abort()
        self.assertEqual(cursor, 0)
        # The full utterance PCM survives the abort so an interrupted
        # sentence can resume from the exact spoken cursor.
        self.assertEqual(len(actor.utterance_pcm()), 128)
        self.assertEqual(actor.resume_from(cursor), 128)
        actor.abort()


class F33EchoCancelTests(unittest.TestCase):
    def setUp(self):
        self.path = echo_cancel.AecSignalPath(
            canceller=echo_cancel.NoOpEchoCanceller(),
            reference_buffer=echo_cancel.ReferencePcmBuffer(max_seconds=5),
        )

    def _pcm(self, seconds, sample_rate=16000):
        return bytes(int(sample_rate * seconds) * 2)

    def test_fresh_reference_aligns(self):
        self.path.feed_reference(self._pcm(1.0), sample_rate=16000,
                                 sample_width=2)
        filtered, had_reference = self.path.cancelled_mic_window(
            self._pcm(0.5))
        self.assertTrue(had_reference)
        # NoOp canceller: the mic window passes through untouched.
        self.assertEqual(filtered, self._pcm(0.5))

    def test_stale_reference_reports_absence(self):
        from collections import deque

        self.path.reference._chunks = deque(
            [(time.monotonic() - 30.0, b"\x00" * 1600)])
        _filtered, had_reference = self.path.cancelled_mic_window(
            self._pcm(0.5))
        self.assertFalse(had_reference)

    def test_reference_downsamples_device_rate(self):
        ref44 = self._pcm(1.0, sample_rate=44100)
        self.path.feed_reference(ref44, sample_rate=44100, sample_width=2)
        aligned = self.path.reference.aligned_reference(len(ref44))
        self.assertGreater(len(aligned), 0)
        # 44.1k -> 16k downsample shrinks the byte count.
        self.assertLess(len(aligned), len(ref44))

    def test_stats_track_usage(self):
        self.path.feed_reference(self._pcm(1.0))
        self.path.cancelled_mic_window(self._pcm(0.2))
        self.assertEqual(self.path.stats["cancelled_analyses"], 1)
        self.assertEqual(self.path.stats["had_reference"], 1)


class F34StabilizerTests(unittest.TestCase):
    def _stabilizer(self):
        return transcript_stabilizer.TranscriptStabilizer()

    def test_final_commits_immediately(self):
        s = self._stabilizer()
        stable = s.push(transcript_stabilizer.TranscriptWindow(
            "w1", "open chrome", final=True))
        self.assertIsNotNone(stable)
        self.assertEqual(s.committed(), "open chrome")
        self.assertTrue(s.is_committed("OPEN CHROME"))

    def test_partial_only_never_committed(self):
        s = self._stabilizer()
        stable = s.push(transcript_stabilizer.TranscriptWindow(
            "p1", "open chr", final=False))
        self.assertIsNone(stable)
        self.assertEqual(s.committed(), "")
        self.assertFalse(s.is_committed("open chr"))

    def test_agreeing_overlapping_partials_commit(self):
        s = self._stabilizer()
        s.push(transcript_stabilizer.TranscriptWindow(
            "p1", "open chrome", final=False, start_ms=0, end_ms=1000))
        stable = s.push(transcript_stabilizer.TranscriptWindow(
            "p2", "open chrome", final=False, start_ms=100, end_ms=1100))
        self.assertIsNotNone(stable)
        self.assertEqual(s.committed(), "open chrome")

    def test_disagreeing_partials_stay_unstable(self):
        s = self._stabilizer()
        s.push(transcript_stabilizer.TranscriptWindow(
            "p1", "open chrome", final=False, start_ms=0, end_ms=1000))
        s.push(transcript_stabilizer.TranscriptWindow(
            "p2", "open browser", final=False, start_ms=100, end_ms=1100))
        self.assertEqual(s.committed(), "")
        self.assertNotEqual(s.unstable(), "")


class F36WakeCommandTests(unittest.TestCase):
    def test_tail_after_wake_window_is_the_command(self):
        command = wake_engine.extract_command(
            "wake up jarvis and search for cats")
        self.assertEqual(command, "and search for cats")

    def test_wake_only_phrase_yields_empty_command(self):
        self.assertEqual(wake_engine.extract_command("wake up jarvis"), "")

    def test_candidate_variants_prefer_longest(self):
        command = wake_engine.extract_command(
            "wake up jarvis",
            candidates=["wake up jarvis", "wake up jarvis open chrome"])
        self.assertEqual(command, "open chrome")

    def test_command_extraction_never_raises(self):
        self.assertEqual(wake_engine.extract_command(None, None), "")
        self.assertEqual(wake_engine.extract_command("", ["", "  "]), "")


class F36PreRollTests(unittest.TestCase):
    def _audio(self, seconds, sample_rate=16000):
        import speech_recognition as sr

        return sr.AudioData(bytes(int(sample_rate * seconds) * 2),
                            sample_rate, 2)

    def setUp(self):
        wake_engine.reset_pre_roll()

    def test_ring_caps_at_pre_roll_seconds(self):
        buffer = wake_engine.WakePreRollBuffer(pre_roll_seconds=1.5)
        for _ in range(5):
            buffer.feed(self._audio(1.0))
        self.assertLessEqual(len(buffer), int(16000 * 2 * 1.5))

    def test_drain_empties_the_ring(self):
        buffer = wake_engine.WakePreRollBuffer(pre_roll_seconds=2.0)
        buffer.feed(self._audio(0.5))
        self.assertGreater(len(buffer), 0)
        self.assertIsNotNone(buffer.drain())
        self.assertEqual(len(buffer), 0)
        self.assertIsNone(buffer.drain())

    def test_disabled_pre_roll_stores_nothing(self):
        buffer = wake_engine.WakePreRollBuffer(pre_roll_seconds=0)
        buffer.feed(self._audio(1.0))
        self.assertEqual(len(buffer), 0)
        self.assertIsNone(buffer.drain())

    def test_online_verify_disabled_proceeds(self):
        # Default env (JARVIS_WAKE_ONLINE_VERIFY unset) → launch proceeds.
        self.assertTrue(wake_engine.online_verify())


class F36ForwardTests(unittest.TestCase):
    def test_forward_posts_command_to_ask(self):
        from backend.config import BACKEND_PORT
        from urllib.request import Request

        captured = {}

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"reply": "ok", "request_id": "r1"}'

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["data"] = request.data
            return FakeResp()

        import urllib.request as _ur

        with mock.patch.object(_ur, "urlopen", fake_urlopen):
            ok = wake_engine.forward_wake_command("open chrome")
        self.assertTrue(ok)
        self.assertTrue(captured["url"].endswith(":%d/ask" % BACKEND_PORT))
        self.assertIn(b"open chrome", captured["data"])

    def test_forward_empty_command_is_noop(self):
        self.assertFalse(wake_engine.forward_wake_command("   "))
        self.assertFalse(wake_engine.forward_wake_command(None))

    def test_forward_backend_down_is_best_effort(self):
        import urllib.request as _ur

        with mock.patch.object(_ur, "urlopen",
                               side_effect=OSError("backend down")):
            self.assertFalse(wake_engine.forward_wake_command("open chrome"))


def _fake_sd_module(recorder):
    """Build a fake `sounddevice` module recording every device write."""
    import types

    class FakeOutputStream:
        def __init__(self, samplerate=None, channels=None, dtype=None,
                     device=None, blocksize=None, latency=None):
            self.recorder = recorder

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, arr):
            self.recorder.append(bytes(arr.tobytes()))

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    module = types.ModuleType("sounddevice")
    module.OutputStream = FakeOutputStream
    module.wait = lambda: None
    return module


class F32FishPlaybackWiringTests(unittest.TestCase):
    """_do_pcm_stream(play=True) drives the AEC reference path + the actor
    utterance through a faked sounddevice module (no hardware)."""

    def setUp(self):
        from backend.services import fish_voice

        self.fish = fish_voice
        self.device_writes = []
        self.fake_sd = _fake_sd_module(self.device_writes)
        echo_cancel.signal_path.reference._chunks.clear()

    def test_play_path_feeds_reference_and_actor(self):
        import numpy as np

        chunk = (np.ones(2048, dtype=np.int16) * 500).astype(np.int16)
        payload_chunks = [chunk.tobytes(), chunk.tobytes()]

        class FakeResp:
            status_code = 200

            def iter_content(self, chunk_size=None):
                for c in payload_chunks:
                    yield c

        actor = audio_actor.get_actor()
        handle = type("H", (), {"stopped": False})()
        with mock.patch.dict(sys.modules, {"sounddevice": self.fake_sd}), \
                mock.patch.object(self.fish, "FISH_API_KEY", "test-key"), \
                mock.patch.object(self.fish, "FISH_REFERENCE_ID",
                                  "ref-under-test"), \
                mock.patch.object(self.fish._session, "post",
                                  return_value=FakeResp()), \
                mock.patch.object(self.fish, "_get_cached_device",
                                  return_value=None), \
                mock.patch.object(self.fish, "_register_sounddevice_playback",
                                  return_value=handle), \
                mock.patch.object(self.fish, "_clear_sounddevice_playback",
                                  lambda h: None), \
                mock.patch.object(self.fish, "_resolve_tts_model",
                                  return_value="test-model"):
            ok = self.fish._do_pcm_stream("g10 wiring", play=True)
        self.assertTrue(ok)
        # [F33] chunks that reached the device are in the reference path.
        self.assertGreater(len(echo_cancel.signal_path.reference), 0)
        # [F32] the actor holds the utterance PCM + generation contract.
        self.assertGreater(len(actor.utterance_pcm()), 0)
        # The fake device actually received the boosted PCM.
        self.assertGreater(len(self.device_writes), 0)
        # Identity-keyed cache slot (model, reference, format, text). The
        # reference is pinned above so this assertion proves the reference is
        # part of the audio identity, instead of silently depending on whether
        # the machine's .env happens to configure a cloned voice.
        self.assertIn(("test-model", "ref-under-test", "pcm", "g10 wiring"),
                      list(self.fish._audio_cache.keys()))
        audio_actor.get_actor().abort()

    def test_stop_aborts_actor(self):
        from backend.services import fish_voice

        with mock.patch.object(audio_actor, "_actor") as fake_actor:
            fish_voice.stop_fish_audio()
            # At least once: stop_fish_audio aborts the owner itself AND the
            # registered handle's stop() aborts it too (idempotent, and the
            # handle may be a leftover from an earlier playback).
            fake_actor.abort.assert_called()
