"""[P1-05] Stop joining audio chunks that have different sample rates.

The capture list holds two kinds of chunk: an echo-cancelled frame at
AEC_SAMPLE_RATE (16 kHz) when playback overlapped it, and the ORIGINAL
native-rate frame when it did not. ``_combine_audio_chunks`` used to join
whatever it was given and label the result with the FIRST chunk's rate, so a
barge-in capture concatenated 16 kHz and 48 kHz bytes under one label -
time-warped speech and a duration wrong by up to 3x.
"""

import io
import struct
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import speech_recognition as sr

from backend.services import audio_input, echo_cancel, listener

AEC = echo_cancel.AEC_SAMPLE_RATE          # 16000
WIDTH = echo_cancel.AEC_SAMPLE_WIDTH      # 2


def _chunk(seconds, rate, width=WIDTH):
    return sr.AudioData(bytes(int(rate * seconds) * width), rate, width)


class MixedRateJoinTests(unittest.TestCase):
    """The regression test the audit asks for."""

    def setUp(self):
        listener._rate_mismatch_reported = False
        listener._rate_mismatch_count = 0

    def test_joining_two_rates_never_produces_a_mislabelled_blob(self):
        # 1s of 16 kHz + 1s of 48 kHz. Real duration is 2s. The old code
        # returned 1.0s of "16 kHz" audio, i.e. 3x time-warped.
        audio = listener._combine_audio_chunks([_chunk(1.0, AEC), _chunk(1.0, 48000)])

        self.assertIsNotNone(audio)
        self.assertEqual(audio.sample_rate, AEC)
        self.assertEqual(audio.sample_width, WIDTH)
        # Duration must reflect reality (~2s), not the byte count mislabelled.
        self.assertAlmostEqual(listener._audio_duration_seconds(audio), 2.0, delta=0.05)

    def test_order_does_not_matter(self):
        """A 48 kHz FIRST chunk must not be relabelled either.

        This is the case a naive fix misses: a single resampler built from the
        first chunk's rate is a no-op when that rate is already 16 kHz, so the
        48 kHz bytes pass through untouched and still get the wrong label.
        """
        audio = listener._combine_audio_chunks([_chunk(1.0, 48000), _chunk(1.0, AEC)])

        self.assertIsNotNone(audio)
        self.assertEqual(audio.sample_rate, AEC)
        self.assertAlmostEqual(listener._audio_duration_seconds(audio), 2.0, delta=0.05)

    def test_three_rates_all_normalise_to_one(self):
        audio = listener._combine_audio_chunks(
            [_chunk(0.5, 8000), _chunk(0.5, AEC), _chunk(0.5, 44100)])
        self.assertEqual(audio.sample_rate, AEC)
        self.assertAlmostEqual(listener._audio_duration_seconds(audio), 1.5, delta=0.05)


    def test_homogeneous_join_is_unchanged(self):
        """No resampling, no rate change, byte-for-byte the old behaviour."""
        chunks = [_chunk(0.25, 48000), _chunk(0.25, 48000)]
        audio = listener._combine_audio_chunks(chunks)
        self.assertEqual(audio.sample_rate, 48000)
        self.assertEqual(audio.frame_data, chunks[0].frame_data + chunks[1].frame_data)
        self.assertEqual(listener._rate_mismatch_count, 0)

    def test_16k_only_capture_is_untouched(self):
        chunks = [_chunk(0.5, AEC), _chunk(0.5, AEC)]
        audio = listener._combine_audio_chunks(chunks)
        self.assertEqual(audio.sample_rate, AEC)
        self.assertEqual(audio.frame_data, chunks[0].frame_data + chunks[1].frame_data)

    def test_mismatch_is_counted_once_and_not_printed_per_frame(self):
        """Reporting stays once-only: this runs per capture in a real loop."""
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            for _ in range(50):
                listener._combine_audio_chunks([_chunk(0.1, AEC), _chunk(0.1, 48000)])
        printed = buffer.getvalue()
        self.assertEqual(printed.count("Mixed capture rates"), 1)
        self.assertEqual(listener._rate_mismatch_count, 50)

    def test_unconvertible_chunk_refuses_rather_than_lying(self):
        """Route (c): a chunk that cannot be described at the AEC format yields None."""
        wide = sr.AudioData(bytes(int(AEC * 0.1) * 4), AEC, 4)
        self.assertIsNone(
            listener._combine_audio_chunks([_chunk(0.1, AEC), wide]))

    def test_zero_rate_chunk_refuses(self):
        # sr.AudioData refuses rate 0 at construction, so the broken chunk is
        # duck-typed - exactly what a malformed stream would look like.
        class _Broken:
            frame_data = b"\x00" * 64
            sample_rate = 0
            sample_width = WIDTH

        self.assertIsNone(
            listener._combine_audio_chunks([_chunk(0.1, AEC), _Broken()]))

    def test_empty_and_blank_inputs(self):
        self.assertIsNone(listener._combine_audio_chunks([]))
        self.assertIsNone(listener._combine_audio_chunks(None))
        self.assertIsNone(listener._combine_audio_chunks([_chunk(0.0, AEC)]))
        self.assertIsNotNone(listener._combine_audio_chunks([None, _chunk(0.1, AEC)]))

    def test_resampler_failure_never_raises(self):
        with patch.object(listener, "_StatefulResampler",
                          side_effect=RuntimeError("numpy exploded")):
            self.assertIsNone(
                listener._combine_audio_chunks([_chunk(0.1, AEC), _chunk(0.1, 48000)])
            )

    def test_capture_shaped_chunk_list_joins_consistently(self):
        """The real barge-in shape: AEC frames with gaps of native frames."""
        chunks = []
        for index in range(6):
            if index % 2 == 0:            # playback overlapped -> AEC frame
                chunks.append(_chunk(0.032, AEC))
            else:                         # silent -> native mic frame
                chunks.append(_chunk(0.032, 48000))
        audio = listener._combine_audio_chunks(chunks)
        self.assertEqual(audio.sample_rate, AEC)
        # 6 x 32ms of real audio = 0.192s; the old join counted the 3 native
        # frames as if their bytes were 16 kHz, halving the reported duration.
        self.assertAlmostEqual(listener._audio_duration_seconds(audio), 0.192, delta=0.01)


class SttInputInvariantTests(unittest.TestCase):
    """Every AudioData handed to STT carries one rate that describes its bytes."""

    def test_invariant_accepts_well_formed_audio(self):
        self.assertTrue(listener.assert_single_rate_audio(_chunk(1.0, 16000)))
        self.assertTrue(listener.assert_single_rate_audio(_chunk(1.0, 48000)))

    def test_invariant_rejects_zero_rate(self):
        class _Broken:
            frame_data = b"\x00" * 32
            sample_rate = 0
            sample_width = 2

        self.assertFalse(listener.assert_single_rate_audio(_Broken()))

    def test_invariant_rejects_partial_sample(self):
        self.assertFalse(
            listener.assert_single_rate_audio(sr.AudioData(b"\x00" * 33, 16000, 2)))

    def test_invariant_never_raises_on_junk(self):
        self.assertFalse(listener.assert_single_rate_audio(object()))
        self.assertFalse(listener.assert_single_rate_audio(None))

    def test_stt_engine_receives_audio_whose_rate_matches_its_length(self):
        """The acceptance check: the rate sent equals the rate the bytes imply.

        ``recognize_inworld`` builds its WAV from ``audio.sample_rate`` +
        ``get_wav_data()``; this drives the mixed-rate capture through the real
        entry door and inspects what an engine would actually be handed.
        """
        captured = {}

        def _fake_inworld(source, *args, **kwargs):
            captured["rate"] = source.sample_rate
            captured["duration"] = listener._audio_duration_seconds(source)
            captured["wav"] = source.get_wav_data()
            raise sr.UnknownValueError

        audio = listener._combine_audio_chunks([_chunk(1.0, AEC), _chunk(1.0, 48000)])
        self.assertTrue(listener.assert_single_rate_audio(audio))

        with patch.object(listener, "recognize_inworld", _fake_inworld), \
             patch.object(listener, "recognize_local_whisper",
                          side_effect=sr.UnknownValueError), \
             patch.object(listener, "_cloud_stt_policy", return_value="on"):
            listener.recognize_multilingual(audio)

        self.assertEqual(captured["rate"], AEC)
        self.assertAlmostEqual(captured["duration"], 2.0, delta=0.05)
        # A WAV header carries its own rate: it must agree with the label.
        self.assertEqual(struct.unpack_from("<I", captured["wav"], 24)[0], AEC)


class PreferredCaptureRateTests(unittest.TestCase):
    """Route (a): the mic is opened at the AEC rate, with a native fallback."""

    class _Mic:
        def __init__(self, device_index=None, sample_rate=None):
            self.device_index = device_index
            self.SAMPLE_RATE = sample_rate or 48000
            self.stream = object()
            self.entered = 0

        def __enter__(self):
            self.entered += 1
            return self

        def __exit__(self, *args):
            return False

    class _RefusingMic(_Mic):
        def __enter__(self):
            self.entered += 1
            self.stream = None      # what sr.Microphone does on a refused rate
            return self

    def test_preferred_rate_is_the_aec_rate(self):
        self.assertEqual(audio_input.AEC_CAPTURE_RATE, AEC)

    def test_mic_is_opened_at_16k_when_the_device_allows_it(self):
        made = []

        def _factory(device_index=None, sample_rate=None):
            made.append(sample_rate)
            return self._Mic(device_index, sample_rate)

        with patch.object(audio_input, "create_microphone", _factory):
            mic = audio_input.open_microphone_at_preferred_rate(3)

        self.assertEqual(made[0], AEC)
        self.assertEqual(mic.SAMPLE_RATE, AEC)
        self.assertFalse(audio_input.mic_rate_state()["refused"])

    def test_refused_rate_falls_back_to_native_and_keeps_the_mic(self):
        """Route (a) -> (b): never fail the capture over a preferred rate."""
        rates = []

        def _factory(device_index=None, sample_rate=None):
            rates.append(sample_rate)
            if sample_rate == AEC:
                return self._RefusingMic(device_index, sample_rate)
            return self._Mic(device_index, sample_rate)

        with patch.object(audio_input, "create_microphone", _factory):
            mic = audio_input.open_microphone_at_preferred_rate(3)

        self.assertEqual(rates, [AEC, None])       # tried 16k, then native
        self.assertIsNotNone(mic)
        self.assertIsNotNone(mic.stream)          # capture survives
        self.assertEqual(mic.SAMPLE_RATE, 48000)
        self.assertTrue(audio_input.mic_rate_state()["refused"])

    def test_device_that_refuses_everything_returns_none(self):
        with patch.object(audio_input, "create_microphone",
                          lambda device_index=None, sample_rate=None:
                              self._RefusingMic(device_index, sample_rate)):
            self.assertIsNone(audio_input.open_microphone_at_preferred_rate(1))

    def test_helper_returns_an_already_entered_mic(self):
        """sr.Microphone asserts it is not entered twice - pin the contract."""
        mic = self._Mic(0, AEC)
        with patch.object(audio_input, "create_microphone",
                          lambda device_index=None, sample_rate=None: mic):
            got = audio_input.open_microphone_at_preferred_rate(0)
        self.assertIs(got, mic)
        self.assertEqual(mic.entered, 1)


class UnchangedBehaviourTests(unittest.TestCase):
    """Constraints: this audit is a rate fix, nothing else moved."""

    def test_pause_threshold_and_caps_untouched(self):
        self.assertEqual(listener.PAUSE_THRESHOLD_SECONDS, 1.2)
        self.assertEqual(listener.MAX_PHRASE_SECONDS, 15)
        self.assertEqual(listener.LISTEN_TIMEOUT_SECONDS, 10)

    def test_aec_resampling_untouched(self):
        """The AEC's own resampler is not modified by P1-05."""
        self.assertEqual(echo_cancel.AEC_SAMPLE_RATE, AEC)


if __name__ == "__main__":
    unittest.main()
