"""[S29] Neural-VAD-driven utterance endpointing.

The END of the user's utterance used to be decided by speech_recognition's
fixed energy threshold (``energy_threshold=300``): a fan/AC keeps the level
above 300 (the capture runs to its phrase limit), a soft voice dips below it
(trailing words cut off), and Jarvis's own playback keeps the level up
whenever the echo is not fully removed. This module drives the end decision
from a small neural VAD - Silero, ONNX, ~0.2 ms per 32 ms frame on CPU -
running on the ALREADY echo-cancelled capture frames, with hysteresis:

  * voice must last ``SPEECH_ON_SECONDS`` (~100 ms) to count as speech;
  * quiet must last ``SPEECH_OFF_SECONDS`` (~200 ms) to end the utterance.

Because the silence is then trusted, the capture can end on it directly
instead of waiting for the energy detector's slow pause - which is what
makes a short pause window safe.

Fallback: when the Silero model is not loadable, the same hysteresis runs on
top of webrtcvad frame verdicts. When the feature is disabled (or no judge
exists at all), the endpoint reports unavailable and the listener keeps
speech_recognition's own behaviour untouched.
"""

import os

NEURAL_VAD_ENABLED = os.getenv("JARVIS_NEURAL_ENDPOINT", "1") != "0"
SPEECH_ON_SECONDS = float(os.getenv("JARVIS_VAD_SPEECH_ON", "0.10"))
SPEECH_OFF_SECONDS = float(os.getenv("JARVIS_VAD_SPEECH_OFF", "0.20"))
VAD_SPEECH_THRESHOLD = float(os.getenv("JARVIS_VAD_THRESHOLD", "0.5"))
#: Silero v5 wants 512-sample frames at 16 kHz (32 ms).
VAD_FRAME_SAMPLES = 512
AEC_SAMPLE_RATE = 16000

_SILERO_SESSION = None
_SILERO_TRIED = False


def _load_silero_session():
    """The cached ONNX runtime session for the bundled Silero model, or None."""
    global _SILERO_SESSION, _SILERO_TRIED
    if _SILERO_TRIED:
        return _SILERO_SESSION
    _SILERO_TRIED = True
    try:
        import onnxruntime as ort
        import silero_vad

        path = os.path.join(os.path.dirname(silero_vad.__file__), "data",
                            "silero_vad.onnx")
        if not os.path.exists(path):
            return None
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        _SILERO_SESSION = ort.InferenceSession(
            path, sess_options=options, providers=["CPUExecutionProvider"])
    except Exception:
        _SILERO_SESSION = None
    return _SILERO_SESSION


class _SileroJudge:
    """Speech probability per 32 ms frame, via the bundled Silero model."""

    name = "silero"

    def __init__(self, session):
        self._session = session
        self._state = None
        self._carry = b""
        self.reset()

    def reset(self):
        import numpy as np

        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._carry = b""

    def frames(self, pcm_bytes, sample_rate=16000, sample_width=2):
        """Yield ``(probability, seconds)`` per 512-sample frame.

        Frames only flow when the capture is already 16 kHz mono s16 (the
        P1-05 preferred rate); anything else is skipped rather than
        misjudged.
        """
        if int(sample_rate) != AEC_SAMPLE_RATE or int(sample_width) != 2:
            return
        import numpy as np

        data = self._carry + bytes(pcm_bytes)
        usable = (len(data) // (VAD_FRAME_SAMPLES * 2)) * VAD_FRAME_SAMPLES * 2
        self._carry = data[usable:]
        data = data[:usable]
        if not data:
            return
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        sr = self._sr()
        for start in range(0, samples.size - VAD_FRAME_SAMPLES + 1,
                           VAD_FRAME_SAMPLES):
            chunk = samples[start:start + VAD_FRAME_SAMPLES]
            out, self._state = self._session.run(
                None, {"input": chunk[None, :], "state": self._state,
                       "sr": sr})
            yield float(out[0][0]), VAD_FRAME_SAMPLES / AEC_SAMPLE_RATE

    @staticmethod
    def _sr():
        import numpy as np

        return np.array(AEC_SAMPLE_RATE, dtype=np.int64)


class _WebRtcJudge:
    """webrtcvad verdicts per 30 ms frame - the no-model fallback."""

    name = "webrtcvad"

    def __init__(self):
        import webrtcvad

        self._vad = webrtcvad.Vad(2)
        self._carry = b""

    def reset(self):
        self._carry = b""

    def frames(self, pcm_bytes, sample_rate=16000, sample_width=2):
        """Yield ``(1.0/0.0, seconds)`` per 480-sample (30 ms) frame."""
        if int(sample_rate) != AEC_SAMPLE_RATE or int(sample_width) != 2:
            return
        frame_bytes = 480 * 2
        data = self._carry + bytes(pcm_bytes)
        usable = (len(data) // frame_bytes) * frame_bytes
        self._carry = data[usable:]
        data = data[:usable]
        for start in range(0, len(data), frame_bytes):
            frame = data[start:start + frame_bytes]
            try:
                voiced = self._vad.is_speech(frame, AEC_SAMPLE_RATE)
            except Exception:
                continue
            yield (1.0 if voiced else 0.0), 0.030


class SpeechEndpoint:
    """Hysteresis endpoint over per-frame voice decisions (S29).

    One instance per capture. ``feed`` returns ``"ended"`` exactly once,
    after ``SPEECH_OFF_SECONDS`` of quiet following voiced speech - the
    listener breaks its capture loop on that event instead of waiting for
    the energy detector.
    """

    def __init__(self, judge=None, on_seconds=None, off_seconds=None,
                 threshold=None):
        self._judge = judge
        self.on_seconds = (SPEECH_ON_SECONDS if on_seconds is None
                           else float(on_seconds))
        self.off_seconds = (SPEECH_OFF_SECONDS if off_seconds is None
                            else float(off_seconds))
        self.threshold = (VAD_SPEECH_THRESHOLD if threshold is None
                          else float(threshold))
        self.speaking = False
        self.has_voiced = False
        self._ended = False
        self._voice_run = 0.0
        self._quiet_run = 0.0

    @property
    def available(self):
        return self._judge is not None

    @property
    def judge_name(self):
        return getattr(self._judge, "name", None)

    def reset(self):
        self.speaking = False
        self.has_voiced = False
        self._ended = False
        self._voice_run = 0.0
        self._quiet_run = 0.0
        if self._judge is not None:
            self._judge.reset()

    def feed(self, pcm_bytes, sample_rate=16000, sample_width=2):
        """Feed one captured frame; ``"ended"`` when the utterance is over.

        The event fires exactly once: the capture loop breaks on it and
        discards this instance, but the latch also makes a reused endpoint
        inert instead of re-firing mid-quiet.
        """
        if self._judge is None or self._ended:
            return None
        for probability, seconds in self._judge.frames(
                pcm_bytes, sample_rate, sample_width):
            voiced = probability >= self.threshold
            if voiced:
                self._voice_run += seconds
                self._quiet_run = 0.0
            else:
                self._quiet_run += seconds
                self._voice_run = 0.0
            if voiced and not self.has_voiced \
                    and self._voice_run >= self.on_seconds:
                self.has_voiced = True
                self.speaking = True
            if self.has_voiced and self._quiet_run >= self.off_seconds:
                self._quiet_run = 0.0
                self.speaking = False
                self._ended = True
                return "ended"
        return None


def make_speech_endpoint():
    """The endpoint for one capture, or None (feature off / no judge).

    The Silero ONNX session is loaded once per process and shared; the VAD
    state itself is per capture.
    """
    if not NEURAL_VAD_ENABLED:
        return None
    session = _load_silero_session()
    if session is not None:
        return SpeechEndpoint(_SileroJudge(session))
    return SpeechEndpoint(_WebRtcJudge())
