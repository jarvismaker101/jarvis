"""LIVE FIX 35 - Jarvis must not hear its own voice as the user (speakers).

With laptop speakers instead of earphones the mic hears the reply, and
Jarvis both interrupts its own speech and answers its own words. Four
layers now stop that, and these tests cover each one offline with
synthetic audio:

1. a real in-house echo canceller (``FrequencyDomainEchoCanceller``): the
   speaker->mic delay is locked by cross-correlation, then a per-frequency
   transfer estimate subtracts the echo; pure-echo frames are flagged
   ``suppressed`` and double-talk survives in the residual;
2. a raised VAD onset bar while playback is active (an echo residual must
   not fire a barge-in; a human voice still does);
3. capture-level rules (echo-only and echo-DOMINANT captures are dropped;
   a confirmed human onset keeps the capture);
4. the transcript echo guard (a transcript matching what Jarvis just said
   aloud is never committed as user speech - locally or from the API
   process via GET /speak/last-text).
"""

import time
import types
import unittest
from unittest import mock

import numpy as np

from backend.services import echo_cancel
from backend.services.echo_cancel import (
    AEC_SAMPLE_RATE as RATE,
    AecSignalPath,
    FrequencyDomainEchoCanceller,
)

FRAME = 512
DELAY_SECONDS = 0.18
DELAY = int(DELAY_SECONDS * RATE)


def _speechish(seconds, seed=7, amp=9000.0):
    """Band-limited noise with a slow envelope: speech-like energy without
    pure-tone degeneracy (a single tone makes the transfer estimate
    trivial in a way real audio never is)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    spec = rng.normal(size=n // 2 + 1) + 0j
    spec[int(len(spec) * 0.12):] = 0
    sig = np.fft.irfft(spec, n=n)
    sig = sig / max(1e-9, np.max(np.abs(sig))) * amp
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * np.arange(n) / RATE) ** 2
    return (sig * envelope).astype(np.float64)


def _pcm(samples):
    return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()


def _as_float(pcm_bytes):
    usable = (len(pcm_bytes) // 2) * 2
    return np.frombuffer(pcm_bytes[:usable], dtype="<i2").astype(np.float64)


def _run_path(mic_signal, ref_signal):
    """Feed a reference timeline plus a mic signal through the signal path.

    The mic hears the reference delayed by DELAY_SECONDS at 0.55 gain,
    plus a little sensor noise. Returns the per-frame CaptureFrames.
    """
    path = AecSignalPath(canceller=FrequencyDomainEchoCanceller(),
                         process="test", transport=None)
    token = path.begin_capture()
    base = time.monotonic()
    frames_out = []
    total = min(len(mic_signal), len(ref_signal)) // FRAME
    for i in range(total):
        ref_chunk = ref_signal[i * FRAME:(i + 1) * FRAME]
        path.feed_reference(_pcm(ref_chunk), sample_rate=RATE,
                            t_end=base + (i + 1) * FRAME / RATE)
        mic_chunk = mic_signal[i * FRAME:(i + 1) * FRAME]
        frames_out.append(path.cancelled_capture_frame(
            _pcm(mic_chunk), duration_seconds=FRAME / RATE,
            mic_t_end=base + (i + 1) * FRAME / RATE,
            frame_id=path.frame_id(i, token), sample_rate=RATE,
            sample_width=2))
    return path, frames_out


def _echoed_mic(ref_signal, gain=0.55, noise=25.0, seed=11):
    """The mic signal a speaker setup produces for the given playback."""
    rng = np.random.default_rng(seed)
    mic = np.zeros_like(ref_signal)
    mic[DELAY:] = gain * ref_signal[:-DELAY]
    return mic + rng.normal(size=ref_signal.size) * noise


class TestFdafEchoSuppression(unittest.TestCase):
    """Layer 1: pure echo is subtracted and flagged, never user speech."""

    def test_pure_echo_frames_suppressed_with_real_erle(self):
        ref = _speechish(2.0)
        mic = _echoed_mic(ref)
        path, frames = _run_path(mic, ref)

        # Past warmup + delay onset: every frame must be recognised as our
        # own playback, and the subtraction must be real (>= 12 dB, well
        # over the 10 dB suppress bar), not just flagged.
        checked = 0
        for i in range(40, len(frames)):
            frame = frames[i]
            mic_chunk = mic[i * FRAME:(i + 1) * FRAME]
            self.assertTrue(frame.suppressed, f"frame {i} not suppressed")
            residual = _as_float(frame.pcm)
            mic_energy = float(np.dot(mic_chunk, mic_chunk))
            res_energy = float(np.dot(residual, residual))
            erle = 10 * np.log10(mic_energy / max(res_energy, 1e-9))
            self.assertGreaterEqual(
                erle, 12.0, f"frame {i} ERLE only {erle:.1f} dB")
            checked += 1
        self.assertGreater(checked, 20)

        metrics = path.canceller.metrics
        self.assertTrue(metrics["lag_locked"])
        self.assertAlmostEqual(metrics["lag_seconds"], DELAY_SECONDS,
                                delta=0.05)
        self.assertGreaterEqual(metrics["last_erle_db"], 12.0)

    def test_delay_is_measured_not_assumed(self):
        # A different acoustic delay (0.35 s, near the search limit) must
        # lock and cancel just as well: nothing in fdaf assumes one lag.
        ref = _speechish(2.0, seed=23)
        delay = int(0.35 * RATE)
        rng = np.random.default_rng(3)
        mic = np.zeros_like(ref)
        mic[delay:] = 0.4 * ref[:-delay]
        mic += rng.normal(size=ref.size) * 20.0

        path = AecSignalPath(canceller=FrequencyDomainEchoCanceller(),
                             process="test", transport=None)
        token = path.begin_capture()
        base = time.monotonic()
        total = len(ref) // FRAME
        for i in range(total):
            path.feed_reference(_pcm(ref[i * FRAME:(i + 1) * FRAME]),
                                sample_rate=RATE,
                                t_end=base + (i + 1) * FRAME / RATE)
            path.cancelled_capture_frame(
                _pcm(mic[i * FRAME:(i + 1) * FRAME]),
                duration_seconds=FRAME / RATE,
                mic_t_end=base + (i + 1) * FRAME / RATE,
                frame_id=path.frame_id(i, token), sample_rate=RATE,
                sample_width=2)
        self.assertTrue(path.canceller._lag.locked)
        self.assertAlmostEqual(path.canceller._lag.locked_lag_seconds,
                               0.35, delta=0.05)

    def test_no_reference_passthrough(self):
        canceller = FrequencyDomainEchoCanceller()
        frame = _speechish(0.032, amp=5000.0)
        out = canceller.cancel(_pcm(frame), b"")
        self.assertFalse(canceller.last_suppressed)
        np.testing.assert_array_equal(_as_float(out), frame.astype(np.int16))

    def test_silent_frame_flagged_not_user_speech(self):
        canceller = FrequencyDomainEchoCanceller()
        ref = _speechish(0.5)
        silence = np.zeros(FRAME)
        # A reference span must be long enough to matter (> window).
        span = _pcm(ref[:FRAME * 2])
        out = canceller.cancel(_pcm(silence), span)
        self.assertTrue(canceller.last_suppressed)


class TestFdafDoubleTalk(unittest.TestCase):
    """Layer 1, the other half: a human voice over playback survives."""

    def _signals(self, seconds=5.0):
        """Reference, echoed mic, and a 1050 Hz user tone (outside the
        reference's band) over frames 50-90, at a level well above the
        echo."""
        ref = _speechish(seconds)
        mic = _echoed_mic(ref)
        user = np.zeros_like(ref)
        span = 40 * FRAME
        user[50 * FRAME:50 * FRAME + span] = (
            np.sin(2 * np.pi * 1050.0 * np.arange(span) / RATE) * 6000.0)
        return ref, mic, user

    def test_user_voice_over_echo_is_not_suppressed(self):
        ref, mic, user = self._signals()

        _path, frames = _run_path(mic + user, ref)

        suppressed = 0
        keeps = []
        user_frame = user[60 * FRAME:(60 + 1) * FRAME]
        user_energy = float(np.dot(user_frame, user_frame))
        for i in range(55, 88):
            frame = frames[i]
            residual = _as_float(frame.pcm)
            keeps.append(float(np.dot(residual, residual)) /
                         max(user_energy, 1e-9))
            if frame.suppressed:
                suppressed += 1
        # The overwhelming majority of double-talk frames must NOT be
        # flagged as our own playback...
        self.assertLessEqual(suppressed, 2)
        # ...and the residual must still hold most of the user's voice.
        self.assertGreaterEqual(float(np.median(keeps)), 0.5)

    def test_echo_returns_after_user_stops(self):
        ref, mic, user = self._signals()

        _path, frames = _run_path(mic + user, ref)
        # After the user stops, the path must return to suppressing echo.
        for i in range(100, len(frames)):
            self.assertTrue(frames[i].suppressed,
                            f"frame {i} echo no longer suppressed")


class TestBuildCanceller(unittest.TestCase):
    """fdaf is the canceller every real session now runs."""

    def test_fdaf_selected_when_webrtcaec3_missing(self):
        if echo_cancel._import_webrtcaec3() is not None:
            self.skipTest("webrtcaec3 installed; it outranks fdaf")
        with mock.patch.object(echo_cancel, "AEC_ENABLED", True):
            canceller = echo_cancel.build_canceller()
        self.assertEqual(canceller.name, "fdaf")
        self.assertFalse(canceller.degraded)

    def test_aec_disabled_falls_to_noop(self):
        with mock.patch.object(echo_cancel, "AEC_ENABLED", False):
            canceller = echo_cancel.build_canceller()
        self.assertEqual(canceller.name, "noop")

    def test_state_reports_canceller_metrics(self):
        path = AecSignalPath(canceller=FrequencyDomainEchoCanceller(),
                             process="test", transport=None)
        state = path.state()
        self.assertEqual(state["mode"], "fdaf")
        self.assertFalse(state["degraded"])
        self.assertEqual(state["canceller"]["name"], "fdaf")
        self.assertIn("frames", state["canceller"]["metrics"])
        self.assertIn("lag_locked", state["canceller"]["metrics"])


class TestEchoCaptureVerdict(unittest.TestCase):
    """Layer 3: which captures count as our own playback."""

    def setUp(self):
        from backend.services import listener

        self.listener = listener

    def test_all_frames_suppressed_drops(self):
        verdict = self.listener._echo_capture_verdict(
            suppressed_frames=30, other_frames=0, speech_started=False,
            had_reference_frames=30)
        self.assertEqual(verdict, "drop")

    def test_echo_dominant_without_onset_drops(self):
        verdict = self.listener._echo_capture_verdict(
            suppressed_frames=60, other_frames=40, speech_started=False,
            had_reference_frames=100)
        self.assertEqual(verdict, "drop")

    def test_echo_dominant_with_confirmed_onset_keeps(self):
        # The user spoke over playback (double-talk): the capture must
        # survive every echo rule.
        verdict = self.listener._echo_capture_verdict(
            suppressed_frames=60, other_frames=40, speech_started=True,
            had_reference_frames=100)
        self.assertEqual(verdict, "keep")

    def test_minority_echo_without_onset_keeps(self):
        verdict = self.listener._echo_capture_verdict(
            suppressed_frames=30, other_frames=70, speech_started=False,
            had_reference_frames=100)
        self.assertEqual(verdict, "keep")

    def test_no_frames_keeps(self):
        verdict = self.listener._echo_capture_verdict(
            suppressed_frames=0, other_frames=0, speech_started=False,
            had_reference_frames=0)
        self.assertEqual(verdict, "keep")


class TestOnsetPlaybackThreshold(unittest.TestCase):
    """Layer 2: the barge-in bar rises while playback is active."""

    def setUp(self):
        from backend.services import listener

        self.listener = listener
        self.chunks = [
            types.SimpleNamespace(
                frame_data=_pcm(np.sin(
                    2 * np.pi * 220.0 * np.arange(FRAME) / RATE) * 4000.0),
                sample_rate=RATE, sample_width=2)
            for _ in range(12)
        ]

    def test_ratio_between_the_bars(self):
        # A VAD ratio that clears the normal onset bar (0.18) but not the
        # playback bar (0.32) is user speech in silence and echo during
        # playback.
        stats = {"speech_frames": 25, "total_frames": 100,
                 "speech_ratio": 0.25}
        with mock.patch.object(self.listener, "_get_speech_stats",
                               return_value=stats):
            self.assertFalse(self.listener._should_confirm_speech_start(
                self.chunks, playback_active=True))
            self.assertTrue(self.listener._should_confirm_speech_start(
                self.chunks, playback_active=False))

    def test_loud_human_voice_clears_both_bars(self):
        stats = {"speech_frames": 80, "total_frames": 100,
                 "speech_ratio": 0.80}
        with mock.patch.object(self.listener, "_get_speech_stats",
                               return_value=stats):
            self.assertTrue(self.listener._should_confirm_speech_start(
                self.chunks, playback_active=True))
            self.assertTrue(self.listener._should_confirm_speech_start(
                self.chunks, playback_active=False))


class TestSpokenTextRegistry(unittest.TestCase):
    """Layer 4 substrate: what this process said aloud, with timestamps."""

    def setUp(self):
        from backend.services import voice

        self.voice = voice
        voice._SPOKEN_TEXTS.clear()

    def tearDown(self):
        self.voice._SPOKEN_TEXTS.clear()

    def test_open_entry_visible_while_speaking(self):
        marker = self.voice.note_spoken_text("hello there user")
        texts = self.voice.recent_spoken_texts(8.0)
        self.assertIn(("hello there user", 0.0), texts)
        self.voice.note_spoken_finished(marker)
        # Finished just now: still visible, small age.
        texts = self.voice.recent_spoken_texts(8.0)
        self.assertEqual(len(texts), 1)
        self.assertEqual(texts[0][0], "hello there user")
        self.assertLess(texts[0][1], 1.0)

    def test_finished_entry_expires(self):
        marker = self.voice.note_spoken_text("old reply")
        self.voice.note_spoken_finished(marker)
        # Backdate the entry's end.
        with self.voice._SPOKEN_LOCK:
            text, started, _ended = self.voice._SPOKEN_TEXTS[0]
            self.voice._SPOKEN_TEXTS[0] = (
                text, started, time.monotonic() - 100.0)
        self.assertEqual(self.voice.recent_spoken_texts(8.0), [])

    def test_new_speak_supersedes_and_closes_the_old(self):
        marker_a = self.voice.note_spoken_text("first reply")
        marker_b = self.voice.note_spoken_text("second reply")
        # The first entry must now be closed; the second still open.
        with self.voice._SPOKEN_LOCK:
            entries = list(self.voice._SPOKEN_TEXTS)
        self.assertIsNotNone(entries[0][2])
        self.assertIsNone(entries[1][2])
        # Cleanup of the SUPERSEDED speak must not close the new entry.
        self.voice.note_spoken_finished(marker_a)
        with self.voice._SPOKEN_LOCK:
            entries = list(self.voice._SPOKEN_TEXTS)
        self.assertIsNone(entries[1][2])
        self.voice.note_spoken_finished(marker_b)
        with self.voice._SPOKEN_LOCK:
            entries = list(self.voice._SPOKEN_TEXTS)
        self.assertIsNotNone(entries[1][2])

    def test_empty_text_not_recorded(self):
        self.voice.note_spoken_text("   ")
        self.assertEqual(self.voice.recent_spoken_texts(8.0), [])


class TestEchoTranscriptGuard(unittest.TestCase):
    """Layer 4: a transcript of our own reply is never a user query."""

    SPOKEN = ("the quick brown fox jumps over the lazy dog near the "
              "river bank today")

    def setUp(self):
        from backend.services import listener
        from backend.services import voice

        self.listener = listener
        self.voice = voice
        voice._SPOKEN_TEXTS.clear()
        # Never touch the network from tests.
        patcher = mock.patch.object(listener, "_remote_spoken_texts",
                                    return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)
        marker = voice.note_spoken_text(self.SPOKEN)
        voice.note_spoken_finished(marker)

    def tearDown(self):
        self.voice._SPOKEN_TEXTS.clear()

    def test_exact_slice_of_spoken_reply_is_echo(self):
        # The mic usually catches only a slice of the reply.
        transcript = "fox jumps over the lazy dog"
        self.assertTrue(self.listener._is_echo_of_spoken_reply(transcript))

    def test_corrupted_slice_is_still_echo(self):
        transcript = "fox jumps over a lazy dog near river"
        self.assertTrue(self.listener._is_echo_of_spoken_reply(transcript))

    def test_unrelated_transcript_is_not_echo(self):
        transcript = "open youtube and play some music for me"
        self.assertFalse(self.listener._is_echo_of_spoken_reply(transcript))

    def test_short_commands_bypass_the_guard(self):
        # Real short commands ("stop", "yes") must never be eaten. The
        # guard's minimum is 4 words BY DESIGN, so only <=3-word
        # transcripts bypass it - a 4-word slice of our own reply is
        # already echo territory.
        for transcript in ("stop", "yes please", "quick brown fox"):
            self.assertFalse(
                self.listener._is_echo_of_spoken_reply(transcript))

    def test_expired_reply_is_not_echo(self):
        with self.voice._SPOKEN_LOCK:
            text, started, _ended = self.voice._SPOKEN_TEXTS[0]
            self.voice._SPOKEN_TEXTS[0] = (
                text, started, time.monotonic() - 100.0)
        transcript = "fox jumps over the lazy dog"
        self.assertFalse(self.listener._is_echo_of_spoken_reply(transcript))

    def test_api_process_reply_counts_as_echo(self):
        # Replies voiced by the API process (typed UI) arrive through the
        # remote leg, not the local registry.
        self.voice._SPOKEN_TEXTS.clear()
        with mock.patch.object(
                self.listener, "_remote_spoken_texts",
                return_value=[("what a nice day to build an assistant", 1.0)]):
            transcript = "nice day to build an assistant"
            self.assertTrue(
                self.listener._is_echo_of_spoken_reply(transcript))

    def test_guard_disabled_by_config(self):
        with mock.patch.object(self.listener, "ECHO_GUARD_ENABLED", False):
            transcript = "fox jumps over the lazy dog"
            self.assertFalse(
                self.listener._is_echo_of_spoken_reply(transcript))

    def test_match_ratio_sliding_window(self):
        ratio = self.listener._spoken_text_match_ratio(
            "fox jumps over the lazy dog", self.SPOKEN)
        self.assertGreaterEqual(ratio, 0.95)
        ratio = self.listener._spoken_text_match_ratio(
            "open youtube and play music", self.SPOKEN)
        self.assertLess(ratio, 0.5)


class TestSpeakLastTextRoute(unittest.TestCase):
    """GET /speak/last-text publishes this process's spoken texts."""

    def setUp(self):
        from backend.api import routes
        from backend.services import voice

        self.routes = routes
        self.voice = voice
        voice._SPOKEN_TEXTS.clear()

    def tearDown(self):
        self.voice._SPOKEN_TEXTS.clear()

    def test_endpoint_returns_registry_with_ages(self):
        marker = self.voice.note_spoken_text("typed reply text here")
        self.voice.note_spoken_finished(marker)
        response = self.routes.speak_last_text()
        self.assertIn("texts", response)
        entries = [t["text"] for t in response["texts"]]
        self.assertIn("typed reply text here", entries)
        for item in response["texts"]:
            self.assertIsInstance(item["age_seconds"], float)
            self.assertGreaterEqual(item["age_seconds"], 0.0)

    def test_open_entry_has_age_zero(self):
        self.voice.note_spoken_text("still speaking entry")
        response = self.routes.speak_last_text()
        ages = {t["text"]: t["age_seconds"] for t in response["texts"]}
        self.assertEqual(ages.get("still speaking entry"), 0.0)

    def test_old_entries_not_served(self):
        marker = self.voice.note_spoken_text("ancient reply")
        self.voice.note_spoken_finished(marker)
        with self.voice._SPOKEN_LOCK:
            text, started, _ended = self.voice._SPOKEN_TEXTS[0]
            self.voice._SPOKEN_TEXTS[0] = (
                text, started, time.monotonic() - 100.0)
        response = self.routes.speak_last_text()
        self.assertEqual(response["texts"], [])


class TestStreamSpeakerEchoRegistry(unittest.TestCase):
    """[B102] The REAL conversation voicing path feeds the spoken-text
    registry: drive StreamSpeaker (no manual note_spoken_text seeding) and
    verify the registry grows chunk by chunk, closes on every session end,
    and that the transcript echo guard matches slices of a streamed reply."""

    def setUp(self):
        from backend.services import listener
        from backend.services import voice

        self.listener = listener
        self.voice = voice
        with voice._SPOKEN_LOCK:
            voice._SPOKEN_TEXTS.clear()
            voice._SPOKEN_SEQ[0] = 0
        listener._remote_spoken_cache["at"] = 0.0
        listener._remote_spoken_cache["texts"] = []
        self._patches = [
            mock.patch.object(voice, "_speak_chunk", return_value=True),
            mock.patch.object(voice, "_prefetch_tts_audio", return_value=None),
            mock.patch.object(voice, "_tts_char_limit", return_value=1800),
            mock.patch.object(voice.StreamSpeaker, "_tts_engine",
                              return_value="gtts"),
            mock.patch.object(voice, "play_ready_earcon", return_value=None),
            mock.patch.object(listener, "_remote_spoken_texts",
                              return_value=[]),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self._patches):
            patch.stop()
        with self.voice._SPOKEN_LOCK:
            self.voice._SPOKEN_TEXTS.clear()

    def _close_and_join(self, speaker):
        speaker.close()
        worker = getattr(speaker, "_worker", None)
        if worker is not None:
            worker.join(timeout=3.0)

    def _entries(self):
        with self.voice._SPOKEN_LOCK:
            return list(self.voice._SPOKEN_TEXTS)

    def test_registry_grows_and_guard_matches_stream_slices(self):
        speaker = self.voice.StreamSpeaker()
        try:
            # Reasoning deltas are never spoken, so they never register.
            from backend.services.openai_compat_client import (
                REASONING_CHANNEL, StreamDelta)
            speaker.feed(StreamDelta("internal thoughts only ",
                                     channel=REASONING_CHANNEL))
            self.assertEqual(self.voice.recent_spoken_texts(), [])

            speaker.feed("Jarvis is speaking through the laptop speakers now. ")
            entries = self._entries()
            self.assertEqual(len(entries), 1)
            self.assertIn("laptop speakers now", entries[0][0])
            self.assertIsNone(entries[0][2])  # still open

            # The guard matches a SLICE of the streamed reply without any
            # manual note_spoken_text() seeding.
            self.assertTrue(self.listener._is_echo_of_spoken_reply(
                self.listener._normalize_text(
                    "speaking through the laptop speakers now")))
            # Genuine user words are not eaten.
            self.assertFalse(self.listener._is_echo_of_spoken_reply(
                self.listener._normalize_text(
                    "actually open the browser settings instead")))
            # Short control commands are never checked at all.
            self.assertFalse(self.listener._is_echo_of_spoken_reply(
                self.listener._normalize_text("stop talking")))

            # A second sentence extends the SAME open entry.
            speaker.feed("The second sentence extends the active entry. ")
            entries = self._entries()
            self.assertEqual(len(entries), 1)
            self.assertIn("second sentence extends", entries[0][0])
            self.assertIsNone(entries[0][2])

            # Background announcements ride the same entry.
            self.assertTrue(speaker.enqueue_external(
                "Background research task has finished."))
            entries = self._entries()
            self.assertEqual(len(entries), 1)
            self.assertIn("research task has finished", entries[0][0])
        finally:
            self._close_and_join(speaker)

    def test_completion_closes_entry_via_worker(self):
        speaker = self.voice.StreamSpeaker()
        try:
            speaker.feed("This reply completes and should close its entry. ")
            self.assertIsNone(self._entries()[-1][2])
            speaker.finish()
            deadline = time.time() + 3.0
            ended = None
            while time.time() < deadline:
                ended = self._entries()[-1][2]
                if ended is not None:
                    break
                time.sleep(0.02)
            self.assertIsNotNone(ended)
            # A finished entry stays matchable for the room-tail window.
            self.assertTrue(self.listener._is_echo_of_spoken_reply(
                self.listener._normalize_text(
                    "this reply completes and should close its entry")))
        finally:
            self._close_and_join(speaker)

    def test_close_stop_and_supersede_end_entries(self):
        # Supersede: creating s2 closes s1's open entry immediately.
        s1 = self.voice.StreamSpeaker()
        try:
            s1.feed("First turn reply text is playing aloud now. ")
            self.assertIsNone(self._entries()[-1][2])
            s2 = self.voice.StreamSpeaker()
            try:
                self.assertIsNotNone(self._entries()[-1][2])
                s2.feed("Second turn reply text has replaced the first. ")
                entries = self._entries()
                self.assertEqual(len(entries), 2)
                self.assertIsNotNone(entries[0][2])   # s1 closed
                self.assertIsNone(entries[1][2])      # s2 open

                # Late cleanup on superseded s1 must NOT close s2's entry.
                s1.close()
                self.assertIsNone(self._entries()[1][2])

                # A real stop closes whatever entry is open.
                with mock.patch.object(self.voice, "stop_elevenlabs"), \
                     mock.patch.object(self.voice, "stop_fish_audio"), \
                     mock.patch.object(self.voice, "fish_ws_end_reply"), \
                     mock.patch.object(self.voice, "stop_google_tts"):
                    self.voice.stop_speaking(signal_ready=False)
                self.assertIsNotNone(self._entries()[1][2])
            finally:
                self._close_and_join(s2)
        finally:
            self._close_and_join(s1)

        # close() ends an open entry directly and is idempotent.
        s3 = self.voice.StreamSpeaker()
        try:
            s3.feed("Third turn reply text before an explicit close. ")
            self.assertIsNone(self._entries()[-1][2])
            s3.close()
            self.assertIsNotNone(self._entries()[-1][2])
            s3.close()  # idempotent
            self.assertIsNotNone(self._entries()[-1][2])
        finally:
            worker = getattr(s3, "_worker", None)
            if worker is not None:
                worker.join(timeout=3.0)


class TestVoiceProcessAecTelemetry(unittest.TestCase):
    """[B103] The voice process's echo-defense telemetry: guard counters,
    canceller/signal-path state, the rate-limited commit warning, and the
    ``aec`` field on the published voice-state snapshot."""

    def setUp(self):
        from backend.services import listener

        self.listener = listener
        with listener._echo_guard_lock:
            listener._echo_guard_stats.update({
                "checks": 0,
                "last_candidates": 0,
                "dropped": 0,
                "last_best_ratio": 0.0,
            })
            listener._last_capture_echo_stats.update({
                "had_reference_frames": 0,
                "suppressed_frames": 0,
                "other_frames": 0,
            })
        listener._last_unsuppressed_commit_log_at = 0.0

    def tearDown(self):
        with self.listener._echo_guard_lock:
            self.listener._echo_guard_stats.update({
                "checks": 0,
                "last_candidates": 0,
                "dropped": 0,
                "last_best_ratio": 0.0,
            })
            self.listener._last_capture_echo_stats.update({
                "had_reference_frames": 0,
                "suppressed_frames": 0,
                "other_frames": 0,
            })
        self.listener._last_unsuppressed_commit_log_at = 0.0

    def test_telemetry_reflects_guard_candidates_and_drops(self):
        from backend.services import voice

        with mock.patch.object(self.listener, "_remote_spoken_texts",
                               return_value=[]):
            # 1. A pass with zero candidates.
            self.assertFalse(self.listener._is_echo_of_spoken_reply(
                self.listener._normalize_text(
                    "a transcript with no candidates anywhere")))
            tel = self.listener.aec_voice_telemetry()
            self.assertEqual(tel["echo_guard"]["checks"], 1)
            self.assertEqual(tel["echo_guard"]["candidates"], 0)
            self.assertEqual(tel["echo_guard"]["dropped"], 0)
            self.assertEqual(tel["echo_guard"]["enabled"],
                             self.listener.ECHO_GUARD_ENABLED)

            # 2. Candidates + a drop, driven through StreamSpeaker.
            with voice._SPOKEN_LOCK:
                voice._SPOKEN_TEXTS.clear()
                voice._SPOKEN_SEQ[0] = 0
            patches = [
                mock.patch.object(voice, "_speak_chunk", return_value=True),
                mock.patch.object(voice, "_prefetch_tts_audio",
                                  return_value=None),
                mock.patch.object(voice, "_tts_char_limit", return_value=1800),
                mock.patch.object(voice.StreamSpeaker, "_tts_engine",
                                  return_value="gtts"),
                mock.patch.object(voice, "play_ready_earcon",
                                  return_value=None),
            ]
            for patch in patches:
                patch.start()
            speaker = voice.StreamSpeaker()
            try:
                speaker.feed("Here is the multi word spoken reply from jarvis. ")
                self.assertTrue(self.listener._is_echo_of_spoken_reply(
                    self.listener._normalize_text(
                        "multi word spoken reply from jarvis")))
                tel = self.listener.aec_voice_telemetry()
                self.assertGreaterEqual(tel["echo_guard"]["candidates"], 1)
                self.assertEqual(tel["echo_guard"]["dropped"], 1)
                self.assertGreaterEqual(tel["echo_guard"]["last_best_ratio"],
                                        0.72)
            finally:
                speaker.close()
                worker = getattr(speaker, "_worker", None)
                if worker is not None:
                    worker.join(timeout=3.0)
                for patch in reversed(patches):
                    patch.stop()
                with voice._SPOKEN_LOCK:
                    voice._SPOKEN_TEXTS.clear()

        # 3. Structure: canceller + signal path + lag + capture keys.
        tel = self.listener.aec_voice_telemetry()
        for key in ("mode", "canceller", "lag_locked", "lag_seconds",
                    "metrics", "echo_guard", "signal_path", "last_capture",
                    "errors"):
            self.assertIn(key, tel)
        for key in ("frames", "adapted", "frozen", "suppressed",
                    "last_erle_db", "resets"):
            self.assertIn(key, tel["metrics"])
        self.assertIn("echo_suppressed", tel["signal_path"])
        self.assertIn("had_reference", tel["signal_path"])
        self.assertIn("had_reference_frames", tel["last_capture"])

    def test_publish_voice_state_loop_includes_aec_section(self):
        from backend import listener_state
        from backend import voice_mode

        published = []
        stop_event = mock.MagicMock()
        stop_event.is_set.side_effect = [False, True]

        def _capture_post(path, payload, timeout=1.0):
            published.append((path, dict(payload)))

        try:
            with mock.patch.object(voice_mode, "_post_backend",
                                   side_effect=_capture_post), \
                 mock.patch.object(voice_mode, "_ship_turn_marks",
                                   return_value=None), \
                 mock.patch.object(voice_mode._VOICE_STATE_DIRTY, "wait",
                                   return_value=True):
                voice_mode._publish_voice_state_loop(stop_event=stop_event)
        finally:
            listener_state.unregister_state_hook(
                voice_mode._on_voice_state_change)

        self.assertEqual(len(published), 1)
        path, snapshot = published[0]
        self.assertEqual(path, "/voice-state/publish")
        self.assertIn("aec_errors", snapshot)
        self.assertIn("aec", snapshot)
        self.assertIn("metrics", snapshot["aec"])
        self.assertIn("echo_guard", snapshot["aec"])
        self.assertIn("lag_locked", snapshot["aec"])

    def test_unsuppressed_playback_commit_warning_is_rate_limited(self):
        with self.listener._echo_guard_lock:
            self.listener._last_capture_echo_stats.update({
                "had_reference_frames": 14,
                "suppressed_frames": 0,
                "other_frames": 14,
            })
        with mock.patch("builtins.print") as mocked_print:
            self.assertTrue(
                self.listener._maybe_log_unsuppressed_playback_commit())
            # A second call inside the cooldown window stays silent.
            self.assertFalse(
                self.listener._maybe_log_unsuppressed_playback_commit())
            self.assertEqual(mocked_print.call_count, 1)
        # A capture that WAS suppressed never warns, regardless of cooldown.
        with self.listener._echo_guard_lock:
            self.listener._last_capture_echo_stats.update({
                "had_reference_frames": 14,
                "suppressed_frames": 12,
            })
        with mock.patch("builtins.print") as mocked_print:
            self.assertFalse(
                self.listener._maybe_log_unsuppressed_playback_commit())
            self.assertEqual(mocked_print.call_count, 0)


class TestColdAecOnsetGate(unittest.TestCase):
    """[B104] Cold-start onset gate: while the canceller has not proven it
    can cancel the active playback, voice-shaped echo must not confirm a
    barge-in — and the gate must fail open so real interruption survives."""

    def setUp(self):
        from backend.services import listener

        self.listener = listener
        listener._aec_last_reference_at = None
        listener._aec_last_suppressed_at = None
        listener._aec_gate_blocked_since = None

    def tearDown(self):
        listener = self.listener
        listener._aec_last_reference_at = None
        listener._aec_last_suppressed_at = None
        listener._aec_gate_blocked_since = None

    def test_cold_playback_blocks_onset_until_first_suppressed_frame(self):
        lst = self.listener
        t = 1000.0
        # The reply starts playing: reference flows and the echo is honestly
        # passed through un-suppressed while the fdaf is still locking.
        for i in range(5):
            lst._note_aec_frame(t + i * 0.05, True, False)
        self.assertTrue(lst._cold_aec_blocks_onset(t + 0.25))

        # One suppressed frame proves cancellation works: the gate opens.
        transition = lst._note_aec_frame(t + 0.30, True, True)
        self.assertTrue(transition)
        self.assertFalse(lst._cold_aec_blocks_onset(t + 0.30))

    def test_without_active_playback_the_gate_is_never_active(self):
        lst = self.listener
        self.assertFalse(lst._cold_aec_blocks_onset(2000.0))
        lst._note_aec_frame(2000.0, True, False)
        after_tail = 2000.0 + lst.SPEECH_START_PLAYBACK_TAIL_SECONDS + 0.01
        self.assertFalse(lst._cold_aec_blocks_onset(after_tail))

    def test_fail_open_bounds_one_unproven_stretch(self):
        lst = self.listener
        start = 3000.0
        samples = []
        t = start
        end = start + lst.AEC_ONSET_PROOF_TIMEOUT + 0.25
        while t <= end:
            lst._note_aec_frame(t, True, False)  # playback keeps flowing
            samples.append((round(t - start, 3),
                            lst._cold_aec_blocks_onset(t)))
            t += 0.05
        self.assertTrue(samples[0][1])    # blocked while unproven
        self.assertFalse(samples[-1][1])  # fail-open by the bound

    def test_trust_expiry_rearms_the_gate_for_a_stalling_canceller(self):
        lst = self.listener
        t = 4000.0
        lst._note_aec_frame(t, True, True)  # proven
        self.assertFalse(lst._cold_aec_blocks_onset(t))
        expiry = t + lst.AEC_SUPPRESSION_TRUST_SECONDS + 0.05
        # Playback continues but nothing has been suppressed since t (the
        # estimate went stale, or an internal reset wiped it): re-arm.
        lst._note_aec_frame(expiry, True, False)
        self.assertTrue(lst._cold_aec_blocks_onset(expiry))

    def test_transition_fires_only_on_the_proof_moment(self):
        lst = self.listener
        t = 5000.0
        self.assertTrue(lst._note_aec_frame(t, True, True))  # cold -> proven
        self.assertFalse(lst._note_aec_frame(t + 0.05, True, True))
        self.assertFalse(lst._note_aec_frame(t + 0.10, True, False))
        # After trust expires, the next suppressed frame is a NEW transition.
        expiry = t + lst.AEC_SUPPRESSION_TRUST_SECONDS + 0.5
        self.assertTrue(lst._note_aec_frame(expiry, True, True))

    def test_echo_buffered_cold_is_dropped_at_the_proof_moment(self):
        """The exact live failure: voice-shaped echo at reply start fills the
        onset buffer; it must never confirm onset — not while cold, and not
        after the proof either (the buffer is dropped when trust arrives)."""
        lst = self.listener
        t = 6000.0
        onset_chunks = ["echo-1", "echo-2"]
        for i in range(4):
            frame_t = t + i * 0.05
            if lst._note_aec_frame(frame_t, True, False):
                onset_chunks.clear()
            self.assertTrue(lst._cold_aec_blocks_onset(frame_t))
        # The canceller locks mid-reply: the pre-proof echo buffer is gone.
        proof_t = t + 0.25
        if lst._note_aec_frame(proof_t, True, True):
            onset_chunks.clear()
        self.assertEqual(onset_chunks, [])
        self.assertFalse(lst._cold_aec_blocks_onset(proof_t))

    def test_capture_loop_consults_the_gate_at_both_barge_in_sites(self):
        import inspect

        source = inspect.getsource(self.listener._capture_audio)
        # One check feeds the onset confirmation, one guards the post-capture
        # fallback: both barge-in sites must consult the gate.
        self.assertEqual(source.count("_cold_aec_blocks_onset"), 2)

    def test_telemetry_exposes_cold_gate_state(self):
        lst = self.listener
        lst._note_aec_frame(8000.0, True, True)
        tel = lst.aec_voice_telemetry()
        self.assertIn("cold_gate", tel)
        self.assertIn("trust_active", tel["cold_gate"])
        self.assertIn("suppressed_age", tel["cold_gate"])
        self.assertIn("reference_age", tel["cold_gate"])


class TestB105DelayEvidence(unittest.TestCase):
    """[B105] The fdaf delay lock fits JOINT multi-frame evidence, and its
    diagnostics are real: the live failure showed every frame reporting
    ``lag_locked=False`` with no way to tell a cold window apart from a
    decorrelated path. These tests drive synthetic timelines and assert
    the lock, the configured search range, the passthrough while unproven,
    and the exported diagnostics."""

    @staticmethod
    def _driver(path, mic_signal, ref_signal, frame=1024, count=None,
                feed_local=True):
        """Like _run_path with a configurable frame size and canceller."""
        token = path.begin_capture()
        base = time.monotonic()
        total = min(len(mic_signal), len(ref_signal)) // frame
        if count is not None:
            total = min(total, count)
        out = []
        for index in range(total):
            if feed_local:
                path.feed_reference(
                    _pcm(ref_signal[index * frame:(index + 1) * frame]),
                    sample_rate=RATE,
                    t_end=base + (index + 1) * frame / RATE)
            out.append(path.cancelled_capture_frame(
                _pcm(mic_signal[index * frame:(index + 1) * frame]),
                duration_seconds=frame / RATE,
                mic_t_end=base + (index + 1) * frame / RATE,
                frame_id=path.frame_id(index, token), sample_rate=RATE,
                sample_width=2))
        return out

    def test_joint_window_locks_known_delay(self):
        est = echo_cancel.DelayEstimator()
        frame = 1024
        delay = int(0.25 * RATE)
        ref = _speechish(3.0, seed=5)
        mic = np.zeros_like(ref)
        mic[delay:] = 0.5 * ref[:-delay]

        span = frame + int((est.max_lag_seconds + est.window_seconds) * RATE)
        votes = []
        for index in range(15):
            end = (index + 1) * frame
            vote = est.estimate(_pcm(mic[index * frame:end]),
                                _pcm(ref[max(0, end - span):end]))
            if vote is not None:
                votes.append(index)
            if index == 0:
                # One frame is not evidence: the joint window accumulates.
                self.assertFalse(est.locked)
                self.assertEqual(est.votes, [])
                self.assertEqual(est.last_reason, "collecting-window")

        # Votes arrive once the accumulated window is fully echo (the mic
        # hears nothing before the delay) and are separated by at least one
        # window (3 x 1024 samples): indices 6, 9, 12.
        self.assertEqual(votes[:3], [6, 9, 12])
        self.assertTrue(est.locked)
        self.assertAlmostEqual(est.locked_lag_seconds, 0.25, delta=0.01)
        self.assertEqual(est.last_reason, "locked")

    def test_configured_range_reaches_fdaf_reference_fetch(self):
        class _RecordingTransport:
            """No audio, just records what the path asked the renderer for."""

            def __init__(self):
                self.durations = []

            def fetch_reference(self, duration_seconds, mic_t_end=None):
                self.durations.append(float(duration_seconds))
                return b"", None

            def state(self):
                return {}

            def note_capture_activity(self, *args, **kwargs):
                pass

        ref = _speechish(4.0, seed=9)
        # No local reference is fed, so every frame asks the transport.
        canceller = FrequencyDomainEchoCanceller(max_lag_seconds=0.8)
        transport = _RecordingTransport()
        path = AecSignalPath(canceller=canceller, process="test",
                             transport=transport)
        self._driver(path, ref, ref, count=8, feed_local=False)
        self.assertGreaterEqual(len(transport.durations), 8)
        expected = 0.8 + canceller._lag.window_seconds
        for asked in transport.durations:
            self.assertGreaterEqual(asked, expected - 1e-6,
                                    "the fdaf search range was not fetched")

        # The larger configured range is really searchable: a 0.55 s delay
        # (outside the default 0.4 s) locks through the local fetch.
        canceller2 = FrequencyDomainEchoCanceller(max_lag_seconds=0.8)
        path2 = AecSignalPath(canceller=canceller2, process="test",
                              transport=None)
        delay = int(0.55 * RATE)
        mic = np.zeros_like(ref)
        mic[delay:] = 0.5 * ref[:-delay]
        self._driver(path2, mic, ref)
        self.assertTrue(canceller2._lag.locked)
        self.assertAlmostEqual(canceller2._lag.locked_lag_seconds, 0.55,
                               delta=0.02)

    def test_search_never_exceeds_configured_range(self):
        est = echo_cancel.DelayEstimator(max_lag_seconds=0.2)
        frame = 1024
        ref = _speechish(3.0, seed=21)
        far_delay = int(0.5 * RATE)  # outside the configured range
        mic = np.zeros_like(ref)
        mic[far_delay:] = 0.6 * ref[:-far_delay]

        # The span arrives oversized; the estimator must clamp the search.
        span = frame + int((est.max_lag_seconds + est.window_seconds) * RATE)
        for index in range(14):
            end = (index + 1) * frame
            est.estimate(_pcm(mic[index * frame:end]),
                         _pcm(ref[max(0, end - span):end]))

        self.assertFalse(est.locked)
        self.assertEqual(est.votes, [])
        self.assertLessEqual(est.last_search_seconds,
                             est.max_lag_seconds + 1e-6)
        # The measured candidate (and the exported peak) stay in range.
        self.assertLessEqual(est.last_candidate_seconds,
                             est.max_lag_seconds + 1e-6)
        self.assertIsNotNone(est.last_correlation)

    def test_old_outlier_does_not_block_recent_consensus(self):
        est = echo_cancel.DelayEstimator()
        est.votes.append(0.35)  # one old disagreeing vote
        frame = 1024
        delay = int(0.2 * RATE)
        ref = _speechish(2.0, seed=23)
        mic = np.zeros_like(ref)
        mic[delay:] = 0.5 * ref[:-delay]

        span = frame + int((est.max_lag_seconds + est.window_seconds) * RATE)
        for index in range(15):
            end = (index + 1) * frame
            est.estimate(_pcm(mic[index * frame:end]),
                         _pcm(ref[max(0, end - span):end]))

        # The old outlier would have kept the whole-list spread (0.15 s)
        # above the lock bar forever; the last three consistent votes lock.
        self.assertTrue(est.locked)
        self.assertAlmostEqual(est.locked_lag_seconds, 0.2, delta=0.01)

    def test_independent_audio_does_not_create_a_lock(self):
        est = echo_cancel.DelayEstimator()
        frame = 1024
        ref = _speechish(2.5, seed=31)
        mic = _speechish(2.5, seed=32)  # unrelated broadband audio
        span = frame + int((est.max_lag_seconds + est.window_seconds) * RATE)
        for index in range(14):
            end = (index + 1) * frame
            est.estimate(_pcm(mic[index * frame:end]),
                         _pcm(ref[max(0, end - span):end]))

        self.assertFalse(est.locked)
        self.assertEqual(est.votes, [])
        # The measured peak is EXPORTED instead of disappearing as None.
        self.assertIsNotNone(est.last_correlation)
        self.assertLess(est.last_correlation, est.min_correlation)
        self.assertEqual(est.last_reason, "below-correlation-threshold")

    def test_unproven_fdaf_is_exact_passthrough(self):
        canceller = FrequencyDomainEchoCanceller()
        canceller._lag.min_correlation = 2.0  # can never prove a lock
        frame = 1024
        ref = _speechish(1.2, seed=41)
        mic = _echoed_mic(ref)
        span = frame + int((canceller._lag.max_lag_seconds
                            + canceller._lag.window_seconds) * RATE)
        for index in range(16):
            end = (index + 1) * frame
            mic_frame = _pcm(mic[index * frame:end])
            out = canceller.cancel(mic_frame,
                                   _pcm(ref[max(0, end - span):end]))
            self.assertEqual(out, mic_frame)
        self.assertFalse(canceller._lag.locked)
        self.assertFalse(canceller.last_suppressed)
        self.assertIsNone(canceller.last_erle_db)
        self.assertIsNone(canceller.metrics["last_erle_db"])
        self.assertEqual(canceller.metrics["adapted"], 0)
        self.assertEqual(canceller.metrics["lag_votes"], 0)

    def test_capture_boundary_clears_joint_evidence_not_lock(self):
        ref = _speechish(2.0, seed=51)
        mic = _echoed_mic(ref)
        path, _frames = _run_path(mic, ref)
        self.assertTrue(path.canceller._lag.locked)
        self.assertIsNotNone(path.canceller._h)

        path.begin_capture()
        lag = path.canceller._lag
        self.assertTrue(lag.locked)              # the lock survives
        self.assertIsNotNone(path.canceller._h)  # the fitted H survives
        self.assertEqual(lag._mic_history, b"")  # the window is cleared
        self.assertEqual(lag.last_reason, "window-cleared")

    def test_reset_clears_exported_diagnostics(self):
        ref = _speechish(1.5, seed=61)
        mic = _echoed_mic(ref)
        path, _frames = _run_path(mic, ref)
        canceller = path.canceller
        self.assertTrue(canceller._lag.locked)
        self.assertTrue(canceller.metrics["lag_locked"])
        self.assertEqual(canceller.metrics["lag_votes"], 3)

        canceller.reset()
        self.assertFalse(canceller._lag.locked)
        self.assertFalse(canceller.metrics["lag_locked"])
        self.assertIsNone(canceller.metrics["lag_seconds"])
        self.assertIsNone(canceller.metrics["lag_best_correlation"])
        self.assertIsNone(canceller.metrics["lag_candidate_seconds"])
        self.assertEqual(canceller.metrics["lag_votes"], 0)
        self.assertEqual(canceller.metrics["lag_search_seconds"], 0.0)
        self.assertEqual(canceller.metrics["lag_reason"], "reset")
        self.assertFalse(canceller.last_suppressed)
        self.assertIsNone(canceller.last_erle_db)
        self.assertIsNone(canceller.metrics["last_erle_db"])

    def test_state_uses_fdaf_estimator_not_degraded_estimator(self):
        canceller = FrequencyDomainEchoCanceller()
        path = AecSignalPath(canceller=canceller, process="test",
                             transport=None)
        canceller._lag.votes = [0.21, 0.21, 0.21]
        canceller._lag.locked_lag_seconds = 0.21
        path._lag_estimator.votes = [0.05, 0.05, 0.05]
        path._lag_estimator.locked_lag_seconds = 0.05

        lag = path.state()["lag"]
        self.assertEqual(lag["source"], "fdaf")
        self.assertAlmostEqual(lag["locked_seconds"], 0.21)
        self.assertEqual(lag["votes"], 3)
        self.assertIn("gate_history_frames", lag)
        self.assertEqual(lag["window_seconds"],
                         canceller._lag.window_seconds)

        degraded_path = AecSignalPath(
            canceller=echo_cancel.NoOpEchoCanceller(), process="test",
            transport=None)
        degraded_path._lag_estimator.votes = [0.07, 0.07, 0.07]
        degraded_path._lag_estimator.locked_lag_seconds = 0.07
        lag = degraded_path.state()["lag"]
        self.assertEqual(lag["source"], "signal-path")
        self.assertAlmostEqual(lag["locked_seconds"], 0.07)

    def test_voice_telemetry_preserves_lag_diagnostics(self):
        from backend.services import listener

        tel = listener.aec_voice_telemetry()
        self.assertIn("lag", tel)
        lag = tel["lag"]
        for key in ("source", "locked", "locked_seconds", "votes",
                    "best_correlation", "candidate_seconds",
                    "search_seconds", "max_lag_seconds", "window_seconds",
                    "min_correlation", "reason", "gate_history_frames"):
            self.assertIn(key, lag)
        if tel["canceller"] == "fdaf":
            # The old whitelist dropped the new canceller diagnostics.
            for key in ("lag_best_correlation", "lag_candidate_seconds",
                        "lag_votes", "lag_search_seconds", "lag_reason"):
                self.assertIn(key, tel["metrics"])

    def test_double_talk_during_repaired_lock_path_survives(self):
        ref = _speechish(5.0, seed=81)
        mic = _echoed_mic(ref)
        user = np.zeros_like(ref)
        span = 40 * FRAME
        user[50 * FRAME:50 * FRAME + span] = (
            np.sin(2 * np.pi * 1050.0 * np.arange(span) / RATE) * 6000.0)

        path, frames = _run_path(mic + user, ref)
        # The repaired lock is engaged well before the user speaks.
        self.assertTrue(path.canceller._lag.locked)
        suppressed = 0
        keeps = []
        user_frame = user[60 * FRAME:(60 + 1) * FRAME]
        user_energy = float(np.dot(user_frame, user_frame))
        for index in range(55, 88):
            frame = frames[index]
            residual = _as_float(frame.pcm)
            keeps.append(float(np.dot(residual, residual)) /
                         max(user_energy, 1e-9))
            if frame.suppressed:
                suppressed += 1
        self.assertLessEqual(suppressed, 2)
        self.assertGreaterEqual(float(np.median(keeps)), 0.5)
        # After the user stops, the path returns to suppressing echo.
        for index in range(100, len(frames)):
            self.assertTrue(frames[index].suppressed,
                            f"frame {index} echo no longer suppressed")


class TestB106ShortEchoGuard(unittest.TestCase):
    """[B106] Short and STT-mangled fragments of the reply are rejected
    when the capture overlapped playback, without weakening the one-word
    exemption, the no-overlap behavior, or the existing long matcher."""

    def setUp(self):
        import sys

        if "backend.services.listener" not in sys.modules:
            # The listener opens the microphone at import time; that warm
            # open must never touch the real device during a test run.
            from backend.services import audio_input
            with mock.patch.object(
                    audio_input, "open_microphone_at_preferred_rate",
                    return_value=None):
                from backend.services import listener
        from backend.services import listener, voice

        self.listener = listener
        self.voice = voice
        self.old_stats = dict(listener._echo_guard_stats)
        with listener._echo_guard_lock:
            listener._echo_guard_stats.update({
                "checks": 0,
                "last_candidates": 0,
                "dropped": 0,
                "last_best_ratio": 0.0,
                "last_inspected": False,
                "last_reason": "not-checked",
            })
        with voice._SPOKEN_LOCK:
            self.old_entries = list(voice._SPOKEN_TEXTS)
            self.old_seq = voice._SPOKEN_SEQ[0]
            voice._SPOKEN_TEXTS.clear()
        patcher = mock.patch.object(
            listener, "_remote_spoken_texts", return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        with self.listener._echo_guard_lock:
            self.listener._echo_guard_stats.clear()
            self.listener._echo_guard_stats.update(self.old_stats)
        with self.voice._SPOKEN_LOCK:
            self.voice._SPOKEN_TEXTS.clear()
            self.voice._SPOKEN_TEXTS.extend(self.old_entries)
            self.voice._SPOKEN_SEQ[0] = self.old_seq

    def _spoken(self, text):
        with self.voice._SPOKEN_LOCK:
            self.voice._SPOKEN_TEXTS.clear()
        marker = self.voice.note_spoken_text(text)
        self.voice.note_spoken_finished(marker)

    def test_live_short_fragments_are_rejected_when_overlapped(self):
        cases = [
            (
                "My apologies, sir. That infinite density creates a point.",
                "apologies, sir.",
            ),
            (
                "Here's an interesting one, sir: honey never spoils.",
                'one said, "honey never."',
            ),
            (
                "Yes, sir, honey never spoils when sealed properly.",
                "honey never spoils.",
            ),
            (
                "Yes, sir, if honey is resealed well and kept dry.",
                "re-sealed honey.",
            ),
        ]
        for spoken, transcript in cases:
            with self.subTest(transcript=transcript):
                self._spoken(spoken)
                self.assertTrue(
                    self.listener._is_echo_of_spoken_reply(
                        transcript, playback_overlapped=True))

    def test_one_word_commands_always_survive(self):
        self._spoken("Stop. Yes. No. Cancel. Wait.")
        for transcript in ("stop", "yes", "no", "cancel", "wait"):
            with self.subTest(transcript=transcript):
                self.assertFalse(
                    self.listener._is_echo_of_spoken_reply(
                        transcript, playback_overlapped=True))

    def test_short_slice_without_overlap_keeps_existing_behavior(self):
        self._spoken("The quick brown fox jumps over the lazy dog.")
        self.assertFalse(
            self.listener._is_echo_of_spoken_reply("quick brown fox"))

    def test_unrelated_short_commands_survive_overlap(self):
        self._spoken("Honey never spoils when sealed properly.")
        for transcript in ("open settings", "stop talking", "yes please"):
            with self.subTest(transcript=transcript):
                self.assertFalse(
                    self.listener._is_echo_of_spoken_reply(
                        transcript, playback_overlapped=True))

    def test_negation_is_not_discarded(self):
        self._spoken("Honey never spoils.")
        self.assertFalse(
            self.listener._is_echo_of_spoken_reply(
                "honey spoils", playback_overlapped=True))

    def test_early_return_clears_stale_candidate_telemetry(self):
        self._spoken("Honey never spoils.")
        self.listener._is_echo_of_spoken_reply(
            "honey never spoils", playback_overlapped=True)
        self.listener._is_echo_of_spoken_reply(
            "stop", playback_overlapped=True)
        stats = self.listener._echo_guard_stats
        self.assertEqual(stats["checks"], 2)
        self.assertEqual(stats["last_candidates"], 0)
        self.assertFalse(stats["last_inspected"])
        self.assertEqual(stats["last_reason"], "one-word-exempt")

    def test_existing_correct_drops_still_drop(self):
        cases = [
            ("The quick brown fox jumps over the lazy dog near the "
             "river bank today", "fox jumps over the lazy dog"),
            ("The quick brown fox jumps over the lazy dog near the "
             "river bank today", "fox jumps over a lazy dog near river"),
            ("Jarvis is speaking through the laptop speakers now.",
             "speaking through the laptop speakers now"),
            ("This reply completes and should close its entry.",
             "this reply completes and should close its entry"),
        ]
        for spoken, transcript in cases:
            with self.subTest(transcript=transcript):
                self._spoken(spoken)
                self.assertTrue(
                    self.listener._is_echo_of_spoken_reply(
                        transcript, playback_overlapped=True))
        with mock.patch.object(
                self.listener, "_remote_spoken_texts",
                return_value=[("what a nice day to build an assistant",
                               1.0)]):
            self.assertTrue(
                self.listener._is_echo_of_spoken_reply(
                    "nice day to build an assistant",
                    playback_overlapped=True))

    def test_short_match_from_the_remote_process(self):
        with mock.patch.object(
                self.listener, "_remote_spoken_texts",
                return_value=[("My apologies, sir. That infinite density "
                               "creates a point.", 0.5)]):
            # With overlap evidence the remote fragment is caught...
            self.assertTrue(
                self.listener._is_echo_of_spoken_reply(
                    "apologies, sir.", playback_overlapped=True))
            # ...and without overlap the 2-word bypass still applies.
            self.assertFalse(
                self.listener._is_echo_of_spoken_reply("apologies, sir."))

    def test_expired_candidates_are_not_matched(self):
        self._spoken("Honey never spoils.")
        with self.voice._SPOKEN_LOCK:
            text, started, _ended = self.voice._SPOKEN_TEXTS[0]
            self.voice._SPOKEN_TEXTS[0] = (
                text, started, time.monotonic() - 100.0)
        self.assertFalse(
            self.listener._is_echo_of_spoken_reply(
                "honey never spoils", playback_overlapped=True))

        with mock.patch.object(
                self.listener, "_remote_spoken_texts",
                return_value=[("My apologies, sir. That density.", 99.0)]):
            self.assertFalse(
                self.listener._is_echo_of_spoken_reply(
                    "apologies, sir.", playback_overlapped=True))

    def test_disabled_empty_and_empty_registry(self):
        with mock.patch.object(self.listener, "ECHO_GUARD_ENABLED", False):
            self.assertFalse(
                self.listener._is_echo_of_spoken_reply(
                    "honey never spoils", playback_overlapped=True))
            self.assertEqual(
                self.listener._echo_guard_stats["last_reason"], "disabled")
        self.assertFalse(
            self.listener._is_echo_of_spoken_reply(
                "", playback_overlapped=True))
        with self.voice._SPOKEN_LOCK:
            self.voice._SPOKEN_TEXTS.clear()
        self.assertFalse(
            self.listener._is_echo_of_spoken_reply(
                "honey never spoils", playback_overlapped=True))
        stats = self.listener._echo_guard_stats
        self.assertEqual(stats["last_reason"], "no-candidates")
        self.assertTrue(stats["last_inspected"])

    def test_no_matching_across_two_separate_replies(self):
        with self.voice._SPOKEN_LOCK:
            self.voice._SPOKEN_TEXTS.clear()
        for text in ("alpha bravo charlie delta",
                     "echo foxtrot golf hotel"):
            marker = self.voice.note_spoken_text(text)
            self.voice.note_spoken_finished(marker)
        for transcript in ("bravo golf",
                           "charlie delta echo foxtrot"):
            with self.subTest(transcript=transcript):
                self.assertFalse(
                    self.listener._is_echo_of_spoken_reply(
                        transcript, playback_overlapped=True))

    def test_no_matching_of_distant_two_word_subsets(self):
        self._spoken("the quick brown fox jumps over the lazy dog near "
                     "the river bank")
        self.assertFalse(
            self.listener._is_echo_of_spoken_reply(
                "fox river", playback_overlapped=True))

    def test_canonicalization_never_leaks_into_the_transcript(self):
        transcript = "Open C:\\Users\\Mayan\\notes.txt please and read it"
        raw = transcript
        self.assertFalse(
            self.listener._is_echo_of_spoken_reply(
                self.listener._normalize_text(transcript),
                playback_overlapped=True))
        # The guard only canonicalizes internally; the committed
        # transcript is untouched.
        self.assertEqual(transcript, raw)
        self.assertIn("notes", self.listener._echo_words(transcript))

    def test_echo_rejection_runs_before_the_stabilizer(self):
        import inspect

        source = inspect.getsource(self.listener.listen)
        self.assertLess(
            source.index("_is_echo_of_spoken_reply"),
            source.index("_turn_stabilizer.push"))


class _FakeCaptureChunk:
    """Minimal sr-like chunk: _aec_filter_chunk reads these attributes."""

    def __init__(self, data, rate=16000, width=2):
        self.frame_data = data
        self.sample_rate = rate
        self.sample_width = width


class TestB107NoProofPolicy(unittest.TestCase):
    """[B107] A reference-overlapped capture with zero suppressed frames
    and no confirmed onset is UNCERTAIN, not proven echo: it must be
    labelled for the transcript layer, never dropped as if proven and
    never allowed to authorize a stop from VAD alone."""

    def setUp(self):
        from backend.services import listener

        self.listener = listener

    def test_zero_proof_overlap_requires_transcript_decision(self):
        self.assertEqual(
            self.listener._echo_capture_verdict(
                suppressed_frames=0, other_frames=30,
                speech_started=False, had_reference_frames=30),
            "check-transcript")

    def test_reference_absence_keeps_normal_capture(self):
        self.assertEqual(
            self.listener._echo_capture_verdict(
                suppressed_frames=0, other_frames=30,
                speech_started=False, had_reference_frames=0),
            "keep")

    def test_uncertainty_is_not_an_echo_drop(self):
        verdict = self.listener._echo_capture_verdict(
            suppressed_frames=0, other_frames=30,
            speech_started=False, had_reference_frames=30)
        self.assertNotEqual(verdict, "drop")

    def test_proven_echo_still_drops_before_stt(self):
        self.assertEqual(
            self.listener._echo_capture_verdict(
                suppressed_frames=30, other_frames=0,
                speech_started=False, had_reference_frames=30),
            "drop")

    def test_confirmed_double_talk_keeps_existing_verdict(self):
        self.assertEqual(
            self.listener._echo_capture_verdict(
                suppressed_frames=60, other_frames=40,
                speech_started=True, had_reference_frames=100),
            "keep")

    # ── capture-level integration: the verdict is recorded, and the ──
    # ── uncertain capture is not dropped before STT ──

    def _stale_capture_stats(self):
        """Seed the previous capture's numbers and register a restore."""
        with self.listener._echo_guard_lock:
            saved = dict(self.listener._last_capture_echo_stats)
            self.listener._last_capture_echo_stats.update({
                "had_reference_frames": 99,
                "suppressed_frames": 99,
                "other_frames": 99,
                "onset_confirmed": True,
                "verdict": "stale",
            })
        self.addCleanup(self._restore_capture_stats, saved)

    def _restore_capture_stats(self, saved):
        with self.listener._echo_guard_lock:
            self.listener._last_capture_echo_stats.clear()
            self.listener._last_capture_echo_stats.update(saved)

    def _drive_capture(self, chunks, capture_frames):
        """Drive the real ``_capture_audio`` over *chunks*, one AEC result
        per chunk. No human onset is allowed, matching the live cold-AEC
        miss this wave fixes."""
        from backend.services import listener

        def fake_listen(source, timeout=None, phrase_time_limit=None,
                        stream=False):
            return iter(chunks)

        patches = [
            mock.patch.object(listener, "_get_microphone_source",
                              return_value=mock.MagicMock()),
            mock.patch.object(listener.recognizer, "listen",
                              side_effect=fake_listen),
            mock.patch.object(listener, "_aec_capture_frame",
                              side_effect=capture_frames),
            mock.patch.object(listener, "barge_in_on_speech_onset"),
            mock.patch.object(listener, "play_capture_complete_earcon"),
            mock.patch.object(listener, "_recalibrate_listener"),
            mock.patch.object(listener, "_reset_microphone_source"),
            mock.patch.object(listener, "_should_confirm_speech_start",
                              return_value=False),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        listener.listener_state.mark_user_speaking(False)
        listener.empty_listen_count = 0
        return listener._capture_audio()

    def test_uncertain_overlapped_capture_is_kept_and_records_evidence(self):
        # Playback overlapped every frame and the canceller missed all of
        # them: the live "no suppression proof" capture.
        self._stale_capture_stats()
        mic = _speechish(0.4, seed=91, amp=12000.0)
        pcm = mic.astype(np.int16).tobytes()
        size = 6400  # 0.2 s per frame
        chunks = [_FakeCaptureChunk(pcm[i:i + size])
                  for i in range(0, len(pcm), size)]

        def capture_frames(data, **kwargs):
            return echo_cancel.CaptureFrame(data, True, False, 16000)

        audio = self._drive_capture(chunks, capture_frames)

        self.assertIsNotNone(
            audio, "an uncertain overlapped capture must reach STT - the "
                   "transcript guard decides, not VAD alone")
        stats = self.listener._last_capture_echo_stats
        self.assertEqual(stats["verdict"], "check-transcript")
        self.assertEqual(stats["had_reference_frames"], len(chunks))
        self.assertEqual(stats["suppressed_frames"], 0)
        self.assertEqual(stats["other_frames"], len(chunks))
        self.assertFalse(stats["onset_confirmed"])

    def test_all_suppressed_capture_records_its_evidence_before_drop(self):
        # The observability regression: a capture dropped before STT must
        # leave ITS numbers behind, not the previous capture's.
        self._stale_capture_stats()
        mic = _speechish(0.4, seed=93, amp=12000.0)
        pcm = mic.astype(np.int16).tobytes()
        size = 6400
        chunks = [_FakeCaptureChunk(pcm[i:i + size])
                  for i in range(0, len(pcm), size)]

        def capture_frames(data, **kwargs):
            return echo_cancel.CaptureFrame(b"\x00" * len(data), True, True,
                                            16000)

        audio = self._drive_capture(chunks, capture_frames)

        self.assertIsNone(audio, "proven echo must still be dropped")
        stats = self.listener._last_capture_echo_stats
        self.assertEqual(stats["verdict"], "drop")
        self.assertEqual(stats["had_reference_frames"], len(chunks))
        self.assertEqual(stats["suppressed_frames"], len(chunks))
        self.assertEqual(stats["other_frames"], 0)
        self.assertFalse(stats["onset_confirmed"])


# ─────────────────────────────────────────────────────────────────────
# [B109] trust policy: only a proven canceller may cut the reply
# ─────────────────────────────────────────────────────────────────────

def _drive_capture(case, chunks, capture_frames, confirm=False, cold=False,
                   human=None, consumed=None, confirm_fn=None):
    """Drive the real ``_capture_audio`` with explicit AEC frame results."""
    from backend.services import listener

    def stream():
        for chunk in chunks:
            if consumed is not None:
                consumed.append(chunk)
            yield chunk

    if confirm_fn is not None:
        confirm_patch = mock.patch.object(
            listener, "_should_confirm_speech_start",
            side_effect=confirm_fn)
    else:
        confirm_patch = mock.patch.object(
            listener, "_should_confirm_speech_start",
            return_value=confirm)
    patches = [
        mock.patch.object(listener, "_get_microphone_source",
                          return_value=mock.MagicMock()),
        mock.patch.object(listener.recognizer, "listen",
                          side_effect=lambda *a, **k: stream()),
        mock.patch.object(listener, "_aec_capture_frame",
                          side_effect=capture_frames),
        mock.patch.object(listener, "barge_in_on_speech_onset"),
        mock.patch.object(listener, "play_capture_complete_earcon"),
        mock.patch.object(listener, "_recalibrate_listener"),
        mock.patch.object(listener, "_reset_microphone_source"),
        confirm_patch,
        mock.patch.object(listener, "_cold_aec_blocks_onset",
                          return_value=cold),
    ]
    if human is not None:
        patches.append(mock.patch.object(listener, "is_human_voice",
                                         return_value=human))
    for item in patches:
        item.start()
        case.addCleanup(item.stop)
    listener.listener_state.mark_user_speaking(False)
    listener.empty_listen_count = 0
    return listener._capture_audio()


def _reset_gate(listener):
    listener._aec_last_reference_at = None
    listener._aec_last_suppressed_at = None
    listener._aec_gate_blocked_since = None
    listener._aec_epoch_started_at = None
    # [B113] A new test starts with no path generation memory.
    listener._aec_path_generation = None
    listener._aec_trust_generation = None
    listener._aec_frame_generation = None


class TestB109TrustedBargeIn(unittest.TestCase):
    """[B109] Only proven cancellation may cut playback; an unproven
    onset is a bounded, transcript-qualified interruption."""

    def setUp(self):
        from backend.services import listener
        self.listener = listener
        _reset_gate(listener)
        self.addCleanup(_reset_gate, listener)
        saved = dict(listener._onset_stats)
        self.addCleanup(lambda: listener._onset_stats.update(saved))
        with listener._echo_guard_lock:
            stats = dict(listener._last_capture_echo_stats)
        self.addCleanup(lambda: listener._last_capture_echo_stats
                        .update(stats))

    def _chunks(self, seconds, size=3200):
        pcm = _speechish(seconds, seed=111, amp=12000.0) \
            .astype(np.int16).tobytes()
        return [_FakeCaptureChunk(pcm[i:i + size])
                for i in range(0, len(pcm), size)]

    def test_authorization_rules(self):
        lst, t = self.listener, 100.0
        self.assertTrue(lst._onset_authorizes_stop(t, playback_active=False))
        lst._note_aec_frame(t, True, False)
        self.assertFalse(lst._onset_authorizes_stop(t, playback_active=True))
        lst._note_aec_frame(t + 0.05, True, True)
        self.assertTrue(lst._onset_authorizes_stop(t + 0.05,
                                                   playback_active=True))

    def test_previous_reply_proof_does_not_authorize_a_new_reply(self):
        lst, t = self.listener, 200.0
        lst._note_aec_frame(t, True, True)                 # reply A proven
        new = t + lst.SPEECH_START_PLAYBACK_TAIL_SECONDS + 0.1
        lst._note_aec_frame(new, True, False)              # reply B starts
        self.assertTrue(lst._aec_trust_active(new))        # old rule: trusted
        self.assertFalse(lst._onset_authorizes_stop(new, playback_active=True))

    def test_unproven_echo_onset_never_cuts_playback_and_is_bounded(self):
        chunks, consumed = self._chunks(8.0), []           # 80 x 0.1 s
        with mock.patch.object(self.listener, "AEC_PROVISIONAL_MAX_SECONDS",
                               4.0), \
             mock.patch.object(self.listener, "make_speech_endpoint",
                               return_value=None):
            audio = _drive_capture(
                self, chunks,
                lambda data, **kw: echo_cancel.CaptureFrame(data, True, False,
                                                            16000),
                confirm=True, consumed=consumed)
        self.listener.barge_in_on_speech_onset.assert_not_called()
        self.assertIsNotNone(audio, "the transcript must decide")
        # [B117] The 2 s qualify bound no longer cuts: the capture runs to
        # the hard provisional bound and the whole utterance survives.
        self.assertLessEqual(len(consumed), 2 + 40)
        self.assertGreater(len(consumed), 2 + 20)
        stats = self.listener._last_capture_echo_stats
        self.assertEqual(stats["onset_trust"], "provisional")
        self.assertTrue(stats["qualify_overrun"])

    def test_proven_canceller_keeps_instant_barge_in(self):
        calls = {"n": 0}

        def frames(data, **kw):
            calls["n"] += 1
            proof = calls["n"] <= 2           # echo frames, suppressed
            return echo_cancel.CaptureFrame(data, True, proof, 16000)

        _drive_capture(self, self._chunks(1.0), frames, confirm=True)
        self.listener.barge_in_on_speech_onset.assert_called_once()
        self.assertEqual(self.listener._last_capture_echo_stats[
            "onset_trust"], "trusted")

    def test_fallback_barge_in_is_held_while_unproven(self):
        _drive_capture(
            self, self._chunks(0.6),
            lambda data, **kw: echo_cancel.CaptureFrame(data, True, False,
                                                        16000),
            confirm=False, human=True)
        self.listener.barge_in_on_speech_onset.assert_not_called()
        self.assertEqual(self.listener._onset_stats["last"], "held_fallback")

    def test_both_barge_in_sites_consult_the_trust_rule(self):
        import inspect
        source = inspect.getsource(self.listener._capture_audio)
        self.assertEqual(source.count("_onset_authorizes_stop"), 2)
        self.assertEqual(source.count("barge_in_on_speech_onset()"), 2)

    def test_telemetry_exposes_onset_provenance(self):
        tel = self.listener.aec_voice_telemetry()
        self.assertIn("onset", tel)
        self.assertIn("provisional", tel["onset"])
        self.assertIn("epoch_trusted", tel["cold_gate"])


# ─────────────────────────────────────────────────────────────────────
# [B110] the capture verdict must see the onset's provenance
# ─────────────────────────────────────────────────────────────────────

class TestB110OnsetProvenanceVerdict(unittest.TestCase):
    """[B110] The verdict sees WHERE the onset came from."""

    def setUp(self):
        from backend.services import listener
        self.listener = listener
        self.v = listener._echo_capture_verdict
        _reset_gate(listener)
        self.addCleanup(_reset_gate, listener)

    def test_provisional_onset_is_check_transcript_not_keep(self):
        self.assertEqual(self.v(0, 30, True, 30, onset_trusted=False),
                         "check-transcript")
        self.assertEqual(self.v(60, 40, True, 100, onset_trusted=False),
                         "check-transcript")

    def test_trusted_and_legacy_double_talk_still_keep(self):
        self.assertEqual(self.v(60, 40, True, 100, onset_trusted=True),
                         "keep")
        self.assertEqual(self.v(60, 40, True, 100), "keep")

    def test_all_suppressed_drops_whatever_the_onset(self):
        self.assertEqual(self.v(30, 0, True, 30, onset_trusted=False), "drop")

    def test_no_reference_capture_keeps(self):
        self.assertEqual(self.v(0, 30, True, 0, onset_trusted=False), "keep")

    def test_capture_records_provenance(self):
        pcm = _speechish(0.6, seed=121, amp=12000.0).astype(np.int16) \
            .tobytes()
        chunks = [_FakeCaptureChunk(pcm[i:i + 3200])
                  for i in range(0, len(pcm), 3200)]
        _drive_capture(self, chunks,
                       lambda d, **k: echo_cancel.CaptureFrame(d, True,
                                                               False, 16000),
                       confirm=True)
        stats = self.listener._last_capture_echo_stats
        self.assertEqual(stats["verdict"], "check-transcript")
        self.assertEqual(stats["onset_trust"], "provisional")
        self.assertTrue(stats["onset_confirmed"])


# ─────────────────────────────────────────────────────────────────────
# [B111] the live echoes are rejected on uncertain captures
# ─────────────────────────────────────────────────────────────────────

class TestB111EchoGuardEvidence(unittest.TestCase):
    """[B111] The live echoes are rejected on uncertain captures; the F35
    vocabulary and genuinely new utterances always survive."""

    def setUp(self):
        from backend.services import listener, voice
        self.listener, self.voice = listener, voice
        self.old_stats = dict(listener._echo_guard_stats)
        self.old_classifier = listener._control_classifier
        with voice._SPOKEN_LOCK:
            self.old_entries = list(voice._SPOKEN_TEXTS)
            voice._SPOKEN_TEXTS.clear()
        patcher = mock.patch.object(listener, "_remote_spoken_texts",
                                    return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        with self.listener._echo_guard_lock:
            self.listener._echo_guard_stats.clear()
            self.listener._echo_guard_stats.update(self.old_stats)
        self.listener._control_classifier = self.old_classifier
        with self.voice._SPOKEN_LOCK:
            self.voice._SPOKEN_TEXTS.clear()
            self.voice._SPOKEN_TEXTS.extend(self.old_entries)

    def _speaking(self, text):  # open entry: still playing (age 0.0)
        with self.voice._SPOKEN_LOCK:
            self.voice._SPOKEN_TEXTS.clear()
        self.voice.note_spoken_text(text)

    def _echo(self, transcript):
        return self.listener._is_echo_of_spoken_reply(
            self.listener._normalize_text(transcript),
            playback_overlapped=True, uncertain=True)

    def test_live_one_word_echoes_are_rejected(self):
        for spoken, heard in (
            ("Got it, fifteen thousand.What would you like me to do with "
             "it, sir?", "15000."),
            ("Sir, I can't receive or send payments.If you're", "Sir."),
            ("Understood, sir — it's two oh three AM, Sunday.", "Understood."),
        ):
            with self.subTest(heard=heard):
                self._speaking(spoken)
                self.assertTrue(self._echo(heard))
                self.assertEqual(self.listener._echo_guard_stats[
                    "last_reason"], "one-word-echo")

    def test_live_paraphrase_and_number_mangling_are_rejected(self):
        cases = (
            ("I have noted the fifteen thousand. How would you like me to "
             "proceed with this amount?",
             "I have noted the account. How would you like me to receive "
             "your payment?"),
            ("Noted, sir.It's two oh two AM local on Sunday, October "
             "eighteenth, twenty twenty-six.What do you need?",
             "Set pace. Notice it's 202. And uh October 2026."),
            ("Understood, sir — it's two oh three AM, Sunday, October "
             "eighteenth, twenty twenty-six. Tell me what pace you want set.",
             "Understood. It's too unclear."),
        )
        for spoken, heard in cases:
            with self.subTest(heard=heard):
                self._speaking(spoken)
                self.assertTrue(self._echo(heard))
        # The measured gap that let T4 through stays pinned.
        self.assertLess(self.listener._spoken_text_match_ratio(
            self.listener._normalize_text(cases[0][1]), cases[0][0]),
            self.listener.ECHO_TRANSCRIPT_RATIO)

    def test_control_vocabulary_is_never_eaten(self):
        self._speaking("Stop. Wait. Pause. Cancel. Yes. No. Silence. "
                       "Continue.")
        for word in ("stop", "wait", "pause", "cancel", "yes", "no",
                     "silence", "continue"):
            with self.subTest(word=word):
                self.assertFalse(self._echo(word))
                self.assertEqual(self.listener._echo_guard_stats[
                    "last_reason"], "control-exempt")
        self._speaking("Please stop talking over the music, sir.")
        with mock.patch.object(
                self.listener, "_control_classifier",
                side_effect=lambda t: "speech_stop"
                if t == "stop talking" else None):
            self.assertFalse(self._echo("stop talking"))

    def test_exempt_set_covers_every_single_token_f35_phrase(self):
        from backend import voice_mode
        singles = set()
        for _name, groups in voice_mode.CONTROL_GRAMMAR:
            for phrases in groups:
                for phrase in phrases:
                    tokens = voice_mode._phrase_tokens(phrase)
                    if len(tokens) == 1:
                        singles.add(tokens[0])
        self.assertLessEqual(singles, set(self.listener._ECHO_EXEMPT_WORDS))
        self.assertIs(self.listener._control_classifier,
                      voice_mode.classify_control)

    def test_new_utterances_survive_uncertain_overlap(self):
        self._speaking("It's two oh two AM local on Sunday, October "
                       "eighteenth, twenty twenty-six.")
        for heard in ("what is the weather in delhi tomorrow",
                      "open youtube and play some music",
                      "remind me at five", "spotify"):
            with self.subTest(heard=heard):
                self.assertFalse(self._echo(heard))

    def test_one_word_needs_overlap_and_a_fresh_reply(self):
        # [B114] Overlap, not "uncertain", gates one-word echo evidence.
        self._speaking("Do you want Spotify or YouTube, sir?")
        self.assertFalse(self.listener._is_echo_of_spoken_reply("spotify"))
        self.assertEqual(self.listener._echo_guard_stats["last_reason"],
                         "one-word-exempt")
        self.assertTrue(self.listener._is_echo_of_spoken_reply(
            "spotify", playback_overlapped=True))
        self.assertEqual(self.listener._echo_guard_stats["last_reason"],
                         "one-word-echo")
        self.assertFalse(self.listener._echo_guard_stats["last_uncertain"])
        with self.voice._SPOKEN_LOCK:
            text, marker, _ = self.voice._SPOKEN_TEXTS[0]
            self.voice._SPOKEN_TEXTS[0] = (
                text, marker,
                time.monotonic() - self.listener.ECHO_ONE_WORD_MAX_AGE - 1)
        self.assertFalse(self._echo("spotify"))

    def test_uncertainty_is_derived_from_capture_evidence(self):
        u = self.listener._capture_is_uncertain
        base = {"had_reference_frames": 30, "suppressed_frames": 0,
                "verdict": "keep", "onset_trust": "trusted",
                "speaking_at_onset": True}
        self.assertTrue(u(base))                               # live T3-T8
        self.assertFalse(u(dict(base, suppressed_frames=12)))  # proven
        self.assertTrue(u(dict(base, suppressed_frames=12,
                               onset_trust="provisional")))
        self.assertFalse(u(dict(base, speaking_at_onset=False)))
        self.assertFalse(u(dict(base, had_reference_frames=0)))

    def test_control_command_ahead_of_echo_is_kept(self):
        rescue = self.listener._control_prefix_rescue
        self._speaking("I have noted the fifteen thousand. How would you "
                       "like me to proceed with this amount?")
        with mock.patch.object(
                self.listener, "_control_classifier",
                side_effect=lambda t: "speech_stop"
                if t == "stop talking" else None):
            self.assertEqual(rescue("stop talking how would you like me to "
                                    "proceed with this amount"),
                             "stop talking")
            self._speaking("Stop talking, you said? How would you like me "
                           "to proceed?")
            self.assertIsNone(rescue("stop talking how would you like me "
                                     "to proceed"))

    def test_listen_drops_live_echo_with_recorded_reason(self):
        self._speaking("Got it, fifteen thousand.What would you like me to "
                       "do with it, sir?")
        with self.listener._echo_guard_lock:
            saved = dict(self.listener._last_capture_echo_stats)
        self.addCleanup(lambda: self.listener._last_capture_echo_stats
                        .update(saved))
        audio = self.listener.sr.AudioData(b"\x00\x00" * 1600, 16000, 2)

        def fake_capture(marks=None):
            with self.listener._echo_guard_lock:
                self.listener._last_capture_echo_stats.update({
                    "had_reference_frames": 36, "suppressed_frames": 0,
                    "other_frames": 86, "onset_confirmed": True,
                    "verdict": "keep", "onset_trust": "trusted",
                    "speaking_at_onset": True})
            return audio

        with mock.patch.object(self.listener, "_capture_audio",
                               side_effect=fake_capture), \
             mock.patch.object(self.listener, "recognize_multilingual",
                               return_value=("15000.", "15000.", "auto")), \
             mock.patch.object(self.listener, "is_hallucinated_transcript",
                               return_value=False), \
             mock.patch.object(self.listener, "_record_empty_listen"), \
             mock.patch("builtins.print") as printed:
            self.assertIsNone(self.listener.listen())
        self.assertEqual(self.listener._echo_guard_stats["last_reason"],
                         "one-word-echo")
        self.assertTrue(any("Ignoring echo of own reply" in str(c)
                            for c in printed.call_args_list))


# ─────────────────────────────────────────────────────────────────────
# [B108] a lag is only kept if it MEASURES cancellation
# ─────────────────────────────────────────────────────────────────────

class TestB108LagValidation(unittest.TestCase):
    """[B108] A lag is only kept if it MEASURES cancellation."""

    @staticmethod
    def _echo(ref, delay_s, gain=0.55, seed=3):
        delay = int(delay_s * RATE)
        mic = np.zeros_like(ref)
        mic[delay:] = gain * ref[:-delay]
        return mic + np.random.default_rng(seed).normal(size=ref.size) * 25.0

    @staticmethod
    def _force_lock(canceller, lag):
        canceller._lag.votes = [lag] * echo_cancel.AEC_LAG_CONFIRMATIONS
        canceller._lag.locked_lag_seconds = lag

    def _drive(self, canceller, mic, ref):
        path = AecSignalPath(canceller=canceller, process="test",
                             transport=None)
        return TestB105DelayEvidence._driver(path, mic, ref, frame=FRAME)

    def test_wrong_lock_relocks_to_the_measured_lag(self):
        ref = _speechish(4.0, seed=101)
        canceller = FrequencyDomainEchoCanceller(max_lag_seconds=0.4)
        self._force_lock(canceller, 0.0)           # the live signature
        frames = self._drive(canceller, self._echo(ref, 0.12), ref)
        self.assertGreaterEqual(canceller.metrics["relocks"], 1)
        self.assertAlmostEqual(canceller._lag.locked_lag_seconds, 0.12,
                               delta=0.02)
        self.assertTrue(canceller.metrics["lock_proven"])
        self.assertTrue(all(f.suppressed for f in frames[-30:]))

    def test_out_of_range_echo_rejects_the_lock_instead_of_freezing(self):
        ref = _speechish(7.0, seed=103)
        canceller = FrequencyDomainEchoCanceller(max_lag_seconds=0.4)
        self._force_lock(canceller, 0.05)
        frames = self._drive(canceller, self._echo(ref, 0.7), ref)
        self.assertGreaterEqual(canceller.metrics["relearns"], 1)
        self.assertGreaterEqual(canceller.metrics["lock_rejections"], 1)
        self.assertTrue(any(abs(x - 0.05) < 1e-9
                            for x in canceller._lag.rejected_lags))
        self.assertFalse(any(f.suppressed for f in frames),
                         "never claim cancellation it does not have")
        if canceller._lag.locked:
            self.assertGreater(abs(canceller._lag.locked_lag_seconds - 0.05),
                               echo_cancel.AEC_LAG_SPREAD_SECONDS)

    def test_first_locked_frame_is_never_suppressed_by_in_sample_fit(self):
        canceller = FrequencyDomainEchoCanceller()
        self._force_lock(canceller, 0.1)
        span = FRAME + int((canceller._lag.max_lag_seconds
                            + canceller._lag.window_seconds) * RATE)
        ref = _speechish(1.0, seed=105)[:span]
        # Broadband speech-like mic content (NOT a tone outside the
        # reference band): this is the fixture that actually reproduces the
        # in-sample fit suppressing the user's voice.
        user = _speechish(FRAME / RATE, seed=777)
        canceller.cancel(_pcm(user), _pcm(ref))
        self.assertFalse(canceller.last_suppressed)
        self.assertIsNone(canceller.last_erle_prior_db)

    def test_proven_lock_survives_double_talk(self):
        ref, mic, user = TestFdafDoubleTalk._signals(None)
        path, _frames = _run_path(mic + user, ref)
        m = path.canceller.metrics
        self.assertTrue(m["lock_proven"])
        self.assertEqual((m["relocks"], m["relearns"], m["lock_rejections"]),
                         (0, 0, 0))

    def test_locked_estimator_records_challenges_but_never_moves(self):
        est = echo_cancel.DelayEstimator(window_seconds=FRAME / RATE)
        ref = _speechish(1.0, seed=107)[:FRAME + 4000]

        def at(lag):
            return _pcm(ref[ref.size - FRAME - lag:ref.size - lag])

        for _ in range(echo_cancel.AEC_LAG_CONFIRMATIONS):
            est.estimate(at(1920), _pcm(ref))
        self.assertAlmostEqual(est.locked_lag_seconds, 1920 / RATE, places=3)
        for _ in range(echo_cancel.AEC_LAG_CONFIRMATIONS):
            self.assertIsNone(est.estimate(at(320), _pcm(ref)))
        self.assertAlmostEqual(est.locked_lag_seconds, 1920 / RATE, places=3)
        self.assertAlmostEqual(est.challenger(), 320 / RATE, places=3)

    def test_rejected_lag_is_excluded_from_the_next_lock(self):
        est = echo_cancel.DelayEstimator(window_seconds=FRAME / RATE)
        ref = _speechish(1.0, seed=109)[:FRAME + 4000]
        mic = _pcm(ref[ref.size - FRAME - 1920:ref.size - 1920])
        for _ in range(echo_cancel.AEC_LAG_CONFIRMATIONS):
            est.estimate(mic, _pcm(ref))
        self.assertAlmostEqual(est.reject_lock(), 1920 / RATE, places=3)
        for _ in range(echo_cancel.AEC_LAG_CONFIRMATIONS):
            est.estimate(mic, _pcm(ref))
        if est.locked:
            self.assertGreater(abs(est.locked_lag_seconds - 1920 / RATE),
                               echo_cancel.AEC_LAG_SPREAD_SECONDS)

    def test_reference_is_stamped_with_device_latency(self):
        from backend.services import fish_voice
        for reported, expected, enabled in ((0.2, 0.2, True), (5.0,
                fish_voice.AEC_DEVICE_LATENCY_MAX, True), (None, 0.0, True),
                (0.2, 0.0, False)):
            with self.subTest(reported=reported, enabled=enabled), \
                 mock.patch.object(fish_voice, "_actor_output_latency",
                                   return_value=reported), \
                 mock.patch.object(fish_voice, "AEC_STAMP_DEVICE_LATENCY",
                                   enabled):
                lead = fish_voice._reference_t_end() - time.monotonic()
                self.assertAlmostEqual(lead, expected, delta=0.02)

    def test_stream_latency_probe(self):
        from backend.services import audio_actor
        probe = audio_actor._stream_output_latency
        self.assertEqual(probe(types.SimpleNamespace(
            output_latency_seconds=lambda: 0.18)), 0.18)
        self.assertIsNone(probe(object()))
        self.assertIsNone(probe(types.SimpleNamespace(
            output_latency_seconds=lambda: 3.0)))

    def test_future_stamped_reference_serves_only_what_was_heard(self):
        buf = echo_cancel.ReferencePcmBuffer()
        now = time.monotonic()
        buf.feed(_pcm(_speechish(0.5, seed=113)), sample_rate=RATE,
                 t_end=now + 0.2)
        span = buf.aligned_reference(int(0.5 * RATE) * 2, mic_t_end=now)
        self.assertAlmostEqual(len(span), int(0.3 * RATE) * 2, delta=4)

    def test_commit_warning_carries_lag_and_adaptation_numbers(self):
        from backend.services import listener
        with listener._echo_guard_lock:
            saved = dict(listener._last_capture_echo_stats)
            listener._last_capture_echo_stats.update(
                {"had_reference_frames": 14, "suppressed_frames": 0})
        self.addCleanup(lambda: listener._last_capture_echo_stats
                        .update(saved))
        listener._last_unsuppressed_commit_log_at = 0.0
        with mock.patch("builtins.print") as printed:
            self.assertTrue(listener._maybe_log_unsuppressed_playback_commit())
        line = str(printed.call_args)
        for key in ("lag_s=", "lag_corr=", "lock_proven=", "adapted=",
                    "frozen=", "erle_prior_db=", "verdict=", "onset="):
            self.assertIn(key, line)


# ══════════════════════════════════════════════════════════════════════════════
# [B113] trust is evidence-scoped: a dead H estimate cannot authorize a stop
# ══════════════════════════════════════════════════════════════════════════════

class TestB113TrustIsEvidenceScoped(unittest.TestCase):
    """A path reset/relearn/reject voids every earlier suppressed frame."""

    def setUp(self):
        from backend.services import listener
        self.listener = listener
        _reset_gate(listener)
        self.addCleanup(_reset_gate, listener)
        saved = dict(listener._onset_stats)
        self.addCleanup(lambda: listener._onset_stats.update(saved))
        with listener._echo_guard_lock:
            stats = dict(listener._last_capture_echo_stats)
        self.addCleanup(lambda: listener._last_capture_echo_stats.update(stats))

    def test_capture_frame_generation_defaults_and_mic_window_shape(self):
        self.assertIsNone(echo_cancel.CaptureFrame(
            b"\x01\x00", True, False, 16000).generation)
        path = AecSignalPath(canceller=FrequencyDomainEchoCanceller(),
                             process="test", transport=None)
        out = path.cancelled_mic_window(b"\x10\x00" * FRAME,
                                        duration_seconds=FRAME / RATE)
        self.assertEqual(len(out), 2)

    def test_every_h_drop_bumps_the_generation(self):
        c = FrequencyDomainEchoCanceller()
        g0 = c.path_generation
        c.reset()
        self.assertEqual(c.path_generation, g0 + 1)
        c._drop_estimate(keep_relearns=True)
        self.assertEqual(c.path_generation, g0 + 2)
        self.assertEqual(c.metrics["path_generation"], g0 + 2)

    def test_relearn_and_reject_bump_and_frames_carry_it(self):
        ref = _speechish(7.0, seed=103)
        c = FrequencyDomainEchoCanceller(max_lag_seconds=0.4)
        TestB108LagValidation._force_lock(c, 0.05)
        path = AecSignalPath(canceller=c, process="test", transport=None)
        frames = TestB105DelayEvidence._driver(
            path, TestB108LagValidation._echo(ref, 0.7), ref, frame=FRAME)
        self.assertGreaterEqual(
            c.path_generation,
            c.metrics["relearns"] + c.metrics["lock_rejections"])
        self.assertEqual(frames[-1].generation, c.path_generation)

    def test_generation_change_voids_trust_until_new_proof(self):
        lst, t = self.listener, 100.0
        lst._note_aec_frame(t, True, True, generation=1)
        self.assertTrue(lst._onset_authorizes_stop(t, playback_active=True))
        self.assertFalse(lst._note_aec_frame(t + 0.03, True, False,
                                             generation=2))
        self.assertFalse(lst._aec_trust_active(t + 0.03))
        self.assertFalse(lst._onset_authorizes_stop(t + 0.03, True))
        self.assertTrue(lst._cold_aec_blocks_onset(t + 0.03))
        # Fail-open remains bounded while playback continues.
        tt = t + 0.05
        end = t + 0.03 + lst.AEC_ONSET_PROOF_TIMEOUT + 0.1
        while tt <= end:
            lst._note_aec_frame(tt, True, False, generation=2)
            tt += 0.05
        self.assertFalse(lst._cold_aec_blocks_onset(tt))
        self.assertFalse(lst._onset_authorizes_stop(tt, True))  # provisional
        self.assertTrue(lst._note_aec_frame(tt + 0.01, True, True,
                                            generation=2))   # re-armed
        self.assertTrue(lst._onset_authorizes_stop(tt + 0.01, True))
        self.assertGreaterEqual(
            lst.aec_voice_telemetry()["cold_gate"]["trust_invalidations"], 1)

    def test_legacy_frames_without_generation_are_unchanged(self):
        lst, t = self.listener, 200.0
        lst._note_aec_frame(t, True, True)
        lst._note_aec_frame(t + 0.03, True, False)
        self.assertTrue(lst._onset_authorizes_stop(t + 0.03, True))

    def test_real_reset_mid_reply_never_authorizes_on_echo(self):
        """The live repro: a reset left unsuppressed echo frames with trust
        still TRUE. Now no stop is authorized until a new proof."""
        lst = self.listener
        ref = _speechish(6.0, seed=131)
        mic = _echoed_mic(ref)
        path = AecSignalPath(canceller=FrequencyDomainEchoCanceller(),
                             process="test", transport=None)
        token, base = path.begin_capture(), time.monotonic()
        rows, reset_at = [], None
        for i in range(len(ref) // FRAME):
            t_end = base + (i + 1) * FRAME / RATE
            path.feed_reference(_pcm(ref[i * FRAME:(i + 1) * FRAME]),
                                sample_rate=RATE, t_end=t_end)
            if (reset_at is None and i >= 40 and rows
                    and rows[-1][0].suppressed):
                path.canceller.reset(keep_rejections=True)
                reset_at = i
            frame = path.cancelled_capture_frame(
                _pcm(mic[i * FRAME:(i + 1) * FRAME]),
                duration_seconds=FRAME / RATE, mic_t_end=t_end,
                frame_id=path.frame_id(i, token), sample_rate=RATE,
                sample_width=2)
            lst._note_aec_frame(t_end, frame.had_reference,
                                frame.suppressed,
                                generation=frame.generation)
            rows.append((frame, lst._onset_authorizes_stop(t_end, True)))
        self.assertIsNotNone(reset_at, "premise: proven before the reset")
        after = rows[reset_at:]
        proof = next((k for k, (frame, _a) in enumerate(after)
                      if frame.suppressed), None)
        self.assertIsNotNone(proof, "premise: the canceller re-proves")
        self.assertGreater(proof, 0, "premise: unsuppressed warm-up exists")
        self.assertFalse(any(a for _frame, a in after[:proof]))
        self.assertTrue(after[proof][1])

    def test_real_user_in_reset_window_is_provisional_and_kept(self):
        calls = {"n": 0}

        def frames(data, **kw):
            calls["n"] += 1
            if calls["n"] <= 3:
                return echo_cancel.CaptureFrame(data, True, True, 16000, 1)
            return echo_cancel.CaptureFrame(data, True, False, 16000, 2)

        pcm = _speechish(1.0, seed=141, amp=12000.0).astype(np.int16).tobytes()
        chunks = [_FakeCaptureChunk(pcm[i:i + 3200])
                  for i in range(0, len(pcm), 3200)]
        audio = _drive_capture(
            self, chunks, frames,
            confirm_fn=lambda c, playback_active=False: len(c) >= 2)
        self.listener.barge_in_on_speech_onset.assert_not_called()
        self.assertIsNotNone(audio)
        stats = self.listener._last_capture_echo_stats
        self.assertEqual(stats["onset_trust"], "provisional")
        self.assertEqual(stats["verdict"], "check-transcript")


# ══════════════════════════════════════════════════════════════════════════════
# [B114] short echoes drop whenever playback overlapped, not only "uncertain"
# ══════════════════════════════════════════════════════════════════════════════

class TestB114OverlapShortEchoGuard(unittest.TestCase):
    """The live blind spot: a trusted capture with one suppressed frame made
    ``_capture_is_uncertain`` false, so one-word/short echoes bypassed B111."""

    REPLY = ("Got it, fifteen thousand. What would you like me to do with "
             "it, sir?")

    def setUp(self):
        from backend.services import listener, voice
        self.listener, self.voice = listener, voice
        listener.discard_pending_capture(wait_seconds=0)
        self.addCleanup(listener.discard_pending_capture, 0)
        self.old_stats = dict(listener._echo_guard_stats)
        self.old_classifier = listener._control_classifier
        with voice._SPOKEN_LOCK:
            self.old_entries = list(voice._SPOKEN_TEXTS)
            voice._SPOKEN_TEXTS.clear()
        patcher = mock.patch.object(listener, "_remote_spoken_texts",
                                    return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        with self.listener._echo_guard_lock:
            self.listener._echo_guard_stats.clear()
            self.listener._echo_guard_stats.update(self.old_stats)
        self.listener._control_classifier = self.old_classifier
        with self.voice._SPOKEN_LOCK:
            self.voice._SPOKEN_TEXTS.clear()
            self.voice._SPOKEN_TEXTS.extend(self.old_entries)

    def _speaking(self, text):
        with self.voice._SPOKEN_LOCK:
            self.voice._SPOKEN_TEXTS.clear()
        self.voice.note_spoken_text(text)

    def _check(self, heard, uncertain=False, overlapped=True):
        return self.listener._is_echo_of_spoken_reply(
            self.listener._normalize_text(heard),
            playback_overlapped=overlapped, uncertain=uncertain)

    def _reason(self):
        return self.listener._echo_guard_stats["last_reason"]

    def test_zero_suppression_one_word_still_drops(self):
        self._speaking(self.REPLY)
        self.assertTrue(self._check("15000.", uncertain=True))
        self.assertEqual(self._reason(), "one-word-echo")

    def test_trusted_partial_one_word_now_drops(self):
        self._speaking(self.REPLY)
        for heard in ("15000.", "Sir."):
            with self.subTest(heard=heard):
                self.assertTrue(self._check(heard))
                self.assertEqual(self._reason(), "one-word-echo")
                self.assertFalse(
                    self.listener._echo_guard_stats["last_uncertain"])

    def test_trusted_partial_short_number_variant_now_drops(self):
        self._speaking(self.REPLY)
        self.assertTrue(self._check("got it 15000"))
        self.assertEqual(self._reason(), "relaxed-short-echo")

    def test_control_words_survive(self):
        self._speaking("Stop. Wait. Cancel. Yes, sir.")
        for word in ("stop", "wait", "cancel", "yes"):
            with self.subTest(word=word):
                self.assertFalse(self._check(word))
        with mock.patch.object(self.listener, "_control_classifier",
                               side_effect=lambda t: "speech_stop"
                               if t == "stop talking" else None):
            self._speaking("Please stop talking over the music, sir.")
            self.assertFalse(self._check("stop talking"))

    def test_unrelated_short_queries_survive(self):
        self._speaking(self.REPLY)
        for heard in ("spotify", "open settings", "what time is it",
                      "play some music"):
            with self.subTest(heard=heard):
                self.assertFalse(self._check(heard))

    def test_reply_older_than_age_window_is_not_dropped(self):
        self._speaking(self.REPLY)
        with self.voice._SPOKEN_LOCK:
            text, marker, _ = self.voice._SPOKEN_TEXTS[0]
            self.voice._SPOKEN_TEXTS[0] = (
                text, marker,
                time.monotonic() - self.listener.ECHO_ONE_WORD_MAX_AGE - 1)
        self.assertFalse(self._check("15000."))
        self.assertEqual(self._reason(), "no-match")

    def test_listen_drops_trusted_partial_echo(self):
        self._speaking(self.REPLY)
        with self.listener._echo_guard_lock:
            saved = dict(self.listener._last_capture_echo_stats)
        self.addCleanup(lambda: self.listener._last_capture_echo_stats
                        .update(saved))
        audio = self.listener.sr.AudioData(b"\x00\x00" * 1600, 16000, 2)

        def fake_capture(marks=None):
            with self.listener._echo_guard_lock:
                self.listener._last_capture_echo_stats.update({
                    "had_reference_frames": 36, "suppressed_frames": 20,
                    "other_frames": 86, "onset_confirmed": True,
                    "verdict": "keep", "onset_trust": "trusted",
                    "speaking_at_onset": True})
            return audio

        before = self.listener._echo_guard_stats["overlap_short_drops"]
        with mock.patch.object(self.listener, "_capture_audio",
                               side_effect=fake_capture), \
             mock.patch.object(self.listener, "recognize_multilingual",
                               return_value=("15000.", "15000.", "auto")), \
             mock.patch.object(self.listener, "is_hallucinated_transcript",
                               return_value=False), \
             mock.patch.object(self.listener, "_record_empty_listen"), \
             mock.patch("builtins.print"):
            self.assertIsNone(self.listener.listen())
        stats = self.listener._echo_guard_stats
        self.assertEqual(stats["last_reason"], "one-word-echo")
        self.assertFalse(stats["last_uncertain"])
        self.assertEqual(stats["overlap_short_drops"], before + 1)

    def test_telemetry_exposes_short_drop_counters(self):
        guard = self.listener.aec_voice_telemetry()["echo_guard"]
        self.assertIn("overlap_short_drops", guard)
        self.assertIn("drops_by_reason", guard)


# ══════════════════════════════════════════════════════════════════════════════
# [B115] playback-at-onset evidence must cross the process boundary
# ══════════════════════════════════════════════════════════════════════════════

class _RemoteResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


class TestB115PlaybackProvenance(unittest.TestCase):
    """The local is_speaking flag cannot see API-voiced replies, and sampling
    it after the capture loses a reply that ended first."""

    def setUp(self):
        from backend.api import routes
        from backend.services import listener
        self.listener, self.routes = listener, routes
        listener.discard_pending_capture(wait_seconds=0)
        self.addCleanup(listener.discard_pending_capture, 0)
        _reset_gate(listener)
        self.addCleanup(_reset_gate, listener)
        self.old_cache = dict(listener._remote_spoken_cache)
        self.addCleanup(self._restore_cache)
        with listener._echo_guard_lock:
            self.old_stats = dict(listener._last_capture_echo_stats)
        self.addCleanup(lambda: listener._last_capture_echo_stats
                        .update(self.old_stats))

    def _restore_cache(self):
        self.listener._remote_spoken_cache.clear()
        self.listener._remote_spoken_cache.update(self.old_cache)

    def test_route_reports_open_entry_as_speaking(self):
        self.listener._remote_spoken_cache["at"] = 0.0
        from backend.services import voice
        with voice._SPOKEN_LOCK:
            voice._SPOKEN_TEXTS.clear()
        marker = voice.note_spoken_text("still speaking entry")
        try:
            payload = self.routes.speak_last_text()
            self.assertTrue(payload["speaking"])
        finally:
            voice.note_spoken_finished(marker)
            with voice._SPOKEN_LOCK:
                voice._SPOKEN_TEXTS.clear()
        self.assertFalse(self.routes.speak_last_text()["speaking"])

    def test_remote_payload_parse_sets_the_cached_flag(self):
        self.listener._remote_spoken_cache.update(
            {"at": 0.0, "texts": [], "speaking": False})
        body = (b'{"texts":[{"text":"announcement","age_seconds":0.0}],'
                b'"speaking":true}')
        with mock.patch("urllib.request.urlopen",
                        return_value=_RemoteResponse(body)), \
             mock.patch("backend.services.local_auth.auth_headers",
                        return_value={}):
            texts = self.listener._remote_spoken_texts()
        self.assertEqual(texts, [("announcement", 0.0)])
        self.assertTrue(self.listener._remote_is_speaking())
        with mock.patch("urllib.request.urlopen",
                        return_value=_RemoteResponse(
                            b'{"texts":[{"text":"old","age_seconds":1.0}],'
                            b'"speaking":false}')), \
             mock.patch("backend.services.local_auth.auth_headers",
                        return_value={}):
            self.listener._remote_spoken_cache["at"] = 0.0
            self.listener._remote_spoken_texts()
        self.assertFalse(self.listener._remote_is_speaking())

    def test_playback_evidence_uses_the_remote_flag_and_expires(self):
        lst = self.listener
        now = time.monotonic()
        with mock.patch.object(lst.listener_state, "is_speaking",
                               return_value=False):
            lst._aec_last_reference_at = None
            lst._remote_spoken_cache.update(
                {"at": now, "texts": [], "speaking": True})
            self.assertEqual(lst._playback_evidence(now, False),
                             "remote-flag")
            lst._remote_spoken_cache["at"] = now - 10.0
            self.assertEqual(lst._playback_evidence(now, False), "")

    def test_capture_that_outlives_the_reply_keeps_reference_provenance(self):
        chunks = [_FakeCaptureChunk(bytes(3200)) for _ in range(6)]
        calls = {"n": 0}

        def frames(data, **kw):
            calls["n"] += 1
            return echo_cancel.CaptureFrame(data, calls["n"] <= 3, False,
                                            16000)

        with mock.patch.object(self.listener.listener_state, "is_speaking",
                               return_value=False), \
             mock.patch.object(self.listener, "_remote_is_speaking",
                               return_value=False):
            _drive_capture(self, chunks, frames, human=True)
        stats = self.listener._last_capture_echo_stats
        self.assertTrue(stats["speaking_at_onset"])
        self.assertEqual(stats["speaking_source"], "reference")

    def test_api_voiced_onset_is_provenance_not_the_local_flag(self):
        chunks = [_FakeCaptureChunk(bytes(3200)) for _ in range(6)]

        def frames(data, **kw):
            return echo_cancel.CaptureFrame(data, True, False, 16000)

        with mock.patch.object(self.listener.listener_state, "is_speaking",
                               return_value=False), \
             mock.patch.object(self.listener, "_remote_is_speaking",
                               return_value=False):
            _drive_capture(self, chunks, frames, confirm=True, human=True)
        stats = self.listener._last_capture_echo_stats
        self.assertTrue(stats["speaking_at_onset"])
        self.assertEqual(stats["speaking_source"], "reference")

    def test_no_playback_records_none(self):
        chunks = [_FakeCaptureChunk(bytes(3200)) for _ in range(4)]

        def frames(data, **kw):
            return echo_cancel.CaptureFrame(data, False, False, 16000)

        with mock.patch.object(self.listener.listener_state, "is_speaking",
                               return_value=False), \
             mock.patch.object(self.listener, "_remote_is_speaking",
                               return_value=False):
            _drive_capture(self, chunks, frames, confirm=True, human=True)
        stats = self.listener._last_capture_echo_stats
        self.assertFalse(stats["speaking_at_onset"])
        self.assertEqual(stats["speaking_source"], "none")

    def test_listen_remote_or_unlocks_long_coverage(self):
        from backend.services import voice
        reply = ("I have noted the fifteen thousand. How would you like me "
                 "to proceed with this amount?")
        with voice._SPOKEN_LOCK:
            voice._SPOKEN_TEXTS.clear()
        voice.note_spoken_text(reply)
        self.addCleanup(lambda: voice._SPOKEN_TEXTS.clear())
        audio = self.listener.sr.AudioData(b"\x00\x00" * 1600, 16000, 2)

        def fake_capture(marks=None):
            with self.listener._echo_guard_lock:
                self.listener._last_capture_echo_stats.update({
                    "had_reference_frames": 30, "suppressed_frames": 0,
                    "other_frames": 30, "onset_confirmed": True,
                    "verdict": "keep", "onset_trust": "trusted",
                    "speaking_at_onset": False})
            return audio

        def remote():
            self.listener._remote_spoken_cache.update(
                {"at": time.monotonic(), "texts": [(reply, 0.0)],
                 "speaking": True})
            return [(reply, 0.0)]

        with mock.patch.object(self.listener, "_capture_audio",
                               side_effect=fake_capture), \
             mock.patch.object(self.listener, "recognize_multilingual",
                               return_value=(
                                   "I have noted the account. How would you "
                                   "like me to receive your payment?",
                                   "I have noted the account. How would you "
                                   "like me to receive your payment?", "en")), \
             mock.patch.object(self.listener, "is_hallucinated_transcript",
                               return_value=False), \
             mock.patch.object(self.listener, "_record_empty_listen"), \
             mock.patch.object(self.listener, "_remote_spoken_texts",
                               side_effect=remote), \
             mock.patch("builtins.print"):
            self.assertIsNone(self.listener.listen())
        self.assertTrue(self.listener._echo_guard_stats["last_uncertain"])


# ══════════════════════════════════════════════════════════════════════════════
# [B116] the mic is read while the ONE STT call runs
# ══════════════════════════════════════════════════════════════════════════════

class TestB116SttOverlapCapture(unittest.TestCase):
    """A barge-in during cloud STT must be captured, not lost; one audio, one
    transcription still holds."""

    def setUp(self):
        import threading
        from backend.services import listener
        self.threading, self.listener = threading, listener
        listener.discard_pending_capture(wait_seconds=0)
        self.addCleanup(listener.discard_pending_capture, 0)
        with listener._echo_guard_lock:
            saved = dict(listener._last_capture_echo_stats)
        self.addCleanup(lambda: listener._last_capture_echo_stats.update(saved))
        self.a = listener.sr.AudioData(b"\x01\x00" * 1600, 16000, 2)
        self.b = listener.sr.AudioData(b"\x02\x00" * 1600, 16000, 2)
        self.stab = mock.MagicMock()
        self.stab.push.return_value = None
        self.stab.committed.return_value = ""
        for patch in (mock.patch.object(listener, "is_hallucinated_transcript",
                                        return_value=False),
                      mock.patch.object(listener, "_record_empty_listen"),
                      mock.patch.object(listener, "_remote_spoken_texts",
                                        return_value=[]),
                      mock.patch.object(listener, "_turn_stabilizer",
                                        self.stab)):
            patch.start()
            self.addCleanup(patch.stop)

    def test_barge_in_during_stt_is_captured_and_transcribed_once(self):
        lst, stt_running, calls = self.listener, self.threading.Event(), []

        def fake_capture(marks=None, wait_timeout=None, begin_turn=True):
            if wait_timeout is None:
                calls.append(("direct",))
                return self.a
            stt_running.wait(2.0)
            calls.append(("overlap", stt_running.is_set()))
            return self.b

        def fake_stt(audio):
            calls.append(("stt", audio))
            if audio is self.a:
                stt_running.set()
                time.sleep(0.2)          # the worker reads "the mic" now
                return ("first", "first", "en")
            return ("second", "second", "en")

        with mock.patch.object(lst, "_capture_audio",
                               side_effect=fake_capture), \
             mock.patch.object(lst, "recognize_multilingual",
                               side_effect=fake_stt), \
             mock.patch.object(lst, "_is_echo_of_spoken_reply",
                               return_value=False):
            first = lst.listen(overlap_capture=True)
            second = lst.listen()
        self.assertEqual((first, second), ("first", "second"))
        self.assertEqual([c[1] for c in calls if c[0] == "stt"],
                         [self.a, self.b])                # one STT each
        self.assertIn(("overlap", True), calls)          # read during STT
        self.assertEqual(calls.count(("direct",)), 1)    # no recapture
        names = [c[0] for c in self.stab.mock_calls]
        self.assertLess(names.index("push"), names.index("begin_turn"))

    def test_silent_stt_window_yields_no_extra_turn(self):
        lst, direct = self.listener, iter([self.a, None])

        def fake_capture(marks=None, wait_timeout=None, begin_turn=True):
            if wait_timeout is None:
                return next(direct)
            time.sleep(0.01)
            return None

        stt = mock.MagicMock(return_value=("hello", "hello", "en"))
        with mock.patch.object(lst, "_capture_audio",
                               side_effect=fake_capture), \
             mock.patch.object(lst, "recognize_multilingual", stt), \
             mock.patch.object(lst, "_is_echo_of_spoken_reply",
                               return_value=False):
            self.assertEqual(lst.listen(overlap_capture=True), "hello")
            self.assertIsNone(lst.listen())
        self.assertEqual(stt.call_count, 1)

    def test_capture_stats_travel_with_their_audio(self):
        lst, seen = self.listener, []

        def fake_capture(marks=None, wait_timeout=None, begin_turn=True):
            with lst._echo_guard_lock:
                lst._last_capture_echo_stats.update(
                    {"had_reference_frames": 30 if wait_timeout is None
                     else 0})
            return self.a if wait_timeout is None else self.b

        def guard(normalized, playback_overlapped=False, uncertain=False):
            seen.append(playback_overlapped)
            return False

        def slow_stt(audio):
            time.sleep(0.1)
            return ("x y", "x y", "en")

        with mock.patch.object(lst, "_capture_audio",
                               side_effect=fake_capture), \
             mock.patch.object(lst, "recognize_multilingual",
                               side_effect=slow_stt), \
             mock.patch.object(lst, "_is_echo_of_spoken_reply",
                               side_effect=guard):
            lst.listen(overlap_capture=True)
            lst.listen()
        self.assertEqual(seen, [True, False])

    def test_voice_off_discards_the_pending_capture(self):
        lst, stt_seen = self.listener, []

        def fake_capture(marks=None, wait_timeout=None, begin_turn=True):
            return self.a if wait_timeout is None else self.b

        def fake_stt(audio):
            stt_seen.append(audio)
            time.sleep(0.05)
            return ("first", "first", "en")

        with mock.patch.object(lst, "_capture_audio",
                               side_effect=fake_capture), \
             mock.patch.object(lst, "recognize_multilingual",
                               side_effect=fake_stt), \
             mock.patch.object(lst, "_is_echo_of_spoken_reply",
                               return_value=False):
            self.assertEqual(lst.listen(overlap_capture=True), "first")
            self.assertTrue(lst.discard_pending_capture(wait_seconds=1.0))
            self.assertEqual(lst.listen(), "first")
        self.assertIsNot(stt_seen[-1], self.b)

    def test_overlap_poll_timeout_is_not_an_empty_listen(self):
        lst = self.listener
        with mock.patch.object(lst, "_get_microphone_source",
                               return_value=mock.MagicMock()), \
             mock.patch.object(lst.recognizer, "listen",
                               side_effect=lst.sr.WaitTimeoutError()) as lis:
            self.assertIsNone(lst._capture_audio(None, wait_timeout=0.3,
                                                 begin_turn=False))
        self.assertEqual(lis.call_args.kwargs["timeout"], 0.3)
        lst._record_empty_listen.assert_not_called()
        self.stab.begin_turn.assert_not_called()

    def test_default_listen_never_starts_an_overlap(self):
        with mock.patch.object(self.listener, "_start_overlap_capture") as s, \
             mock.patch.object(self.listener, "_capture_audio",
                               return_value=None):
            self.listener.listen()
        s.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# [B117] the qualify bound holds the utterance, it does not cut it
# ══════════════════════════════════════════════════════════════════════════════

class TestB117QualifyBoundKeepsTheUtterance(unittest.TestCase):
    """Past the 2 s bound the capture continues to the endpoint; only the
    transcript may stop the reply."""

    def setUp(self):
        from backend.services import listener
        self.listener = listener
        _reset_gate(listener)
        self.addCleanup(_reset_gate, listener)
        saved = dict(listener._onset_stats)
        self.addCleanup(lambda: listener._onset_stats.update(saved))
        with listener._echo_guard_lock:
            stats = dict(listener._last_capture_echo_stats)
        self.addCleanup(lambda: listener._last_capture_echo_stats.update(stats))
        self.addCleanup(listener.discard_pending_capture, 0)
        listener.discard_pending_capture(wait_seconds=0)
        patcher = mock.patch.object(listener, "_remote_spoken_texts",
                                    return_value=[])
        patcher.start()
        self.addCleanup(patcher.stop)

    def _chunks(self, seconds, size=3200):
        pcm = _speechish(seconds, seed=111, amp=12000.0) \
            .astype(np.int16).tobytes()
        return [_FakeCaptureChunk(pcm[i:i + size])
                for i in range(0, len(pcm), size)]

    def test_echo_then_user_survives_past_the_bound(self):
        chunks = self._chunks(4.5)

        with mock.patch.object(self.listener, "make_speech_endpoint",
                               return_value=None):
            audio = _drive_capture(
                self, chunks,
                lambda data, **kw: echo_cancel.CaptureFrame(
                    data, True, False, 16000),
                confirm=True, human=True)
        self.assertIsNotNone(audio)
        self.assertGreaterEqual(
            self.listener._audio_duration_seconds(audio), 4.4)
        self.listener.barge_in_on_speech_onset.assert_not_called()
        stats = self.listener._last_capture_echo_stats
        self.assertTrue(stats["qualify_overrun"])
        self.assertEqual(stats["onset_trust"], "provisional")

    def test_pure_echo_past_the_bound_is_still_dropped(self):
        from backend.services import voice
        reply = ("Got it, fifteen thousand. What would you like me to do "
                 "with it, sir?")
        with voice._SPOKEN_LOCK:
            voice._SPOKEN_TEXTS.clear()
        voice.note_spoken_text(reply)
        self.addCleanup(lambda: voice._SPOKEN_TEXTS.clear())
        audio = self.listener.sr.AudioData(b"\x00\x00" * 1600, 16000, 2)

        def fake_capture(marks=None):
            with self.listener._echo_guard_lock:
                self.listener._last_capture_echo_stats.update({
                    "had_reference_frames": 30, "suppressed_frames": 0,
                    "other_frames": 30, "onset_confirmed": True,
                    "verdict": "check-transcript",
                    "onset_trust": "provisional",
                    "speaking_at_onset": True, "qualify_overrun": True})
            return audio

        with mock.patch.object(self.listener, "_capture_audio",
                               side_effect=fake_capture), \
             mock.patch.object(self.listener, "recognize_multilingual",
                               return_value=(
                                   "got it fifteen thousand what would you "
                                   "like me to do with it sir",
                                   "got it fifteen thousand what would you "
                                   "like me to do with it sir", "en")), \
             mock.patch.object(self.listener, "is_hallucinated_transcript",
                               return_value=False), \
             mock.patch.object(self.listener, "_record_empty_listen"), \
             mock.patch("builtins.print"):
            self.assertIsNone(self.listener.listen())
        self.assertEqual(self.listener._echo_guard_stats["last_reason"],
                         "window-echo")

    def test_capture_is_still_bounded_by_the_hard_limit(self):
        before = self.listener._onset_stats["qualify_cuts"]
        chunks, consumed = self._chunks(30.0), []
        with mock.patch.object(self.listener, "AEC_PROVISIONAL_MAX_SECONDS",
                               3.0), \
             mock.patch.object(self.listener, "make_speech_endpoint",
                               return_value=None):
            _drive_capture(
                self, chunks,
                lambda data, **kw: echo_cancel.CaptureFrame(
                    data, True, False, 16000),
                confirm=True, consumed=consumed)
        self.assertLessEqual(len(consumed), 32)
        self.assertEqual(self.listener._onset_stats["qualify_cuts"],
                         before + 1)
        self.listener.barge_in_on_speech_onset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
