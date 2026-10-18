"""Echo-cancelled full-duplex listening reference path (F33).

The audit asks that assistant playback be fed into WebRTC AEC3 and that
onset/VAD detection run on the *echo-cancelled* microphone signal, instead
of treating Jarvis's own voice as a user interruption. The AEC tuning
itself is marked [ASSUMPTION] - real echo suppression requires recordings
from the actual speaker/headset setup. What this module delivers is the
*missing reference path*: rendered PCM flows from the audio actor into a
monotonic, timestamped reference buffer, and the microphone window is
aligned against it on a SHARED TIMELINE; the resulting (mic, reference)
pair is handed to an AEC implementation.

F33 corrections in this module:

* **Continuous, stateful resampling.** The old conversion used an integer
  stride (``int(round(16000/44100)) == 0`` -> 1), so 44.1 kHz playback was
  handed to the AEC as if it were 16 kHz - an effective ~14.7 kHz reference
  with periodic discontinuities. ``StatefulResampler`` interpolates with a
  fractional phase and a carried-over sample, so chunk boundaries do not
  break the signal and arbitrary chunk sizes are safe.
* **Timestamp alignment, not "newest suffix".** Every reference chunk is
  stored with the monotonic time its last sample was rendered. A mic window
  is matched to the reference span that OVERLAPS it in time.
* **Once-only frame processing.** ``cancelled_mic_window(..., frame_id=...)``
  processes each capture frame exactly once, so the listener's overlapping
  VAD windows cannot re-feed the AEC or double-count statistics. Ids are scoped
  by a process-unique capture token (``begin_capture()``, P0-13), so a frame id
  can never be reused by a later capture.
* **Explicit degraded state.** ``AecSignalPath.state()`` reports whether
  cancellation is real, no-op, or cross-process, and why.

When ``py-webrtc-aec3`` is installed and ``JARVIS_AEC_ENABLED != 0`` the
streaming AEC3 processor is used. Otherwise the pair is still produced and
a loud no-op fallback applies, so the wiring exists and instrumentation
works before a real AEC model is added.
"""

import base64
import itertools
import json
import math
import os
import threading
import time
from collections import deque, namedtuple
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from backend.config import AEC_ENABLED
from backend.services import local_auth

try:  # pragma: no cover - config always provides these, but stay importable
    from backend.config import AEC_REMOTE_ENABLED, AEC_REMOTE_TIMEOUT
except Exception:  # pragma: no cover
    AEC_REMOTE_ENABLED = True
    AEC_REMOTE_TIMEOUT = 0.35

AEC_SAMPLE_RATE = 16000
AEC_SAMPLE_WIDTH = 2
REFERENCE_MAX_SECONDS = 30
#: A mic window is only cancelled against reference PCM rendered within this
#: many seconds of it (a stale reference means "playback did not overlap").
REFERENCE_MAX_DRIFT_SECONDS = 2.0
#: [PERF] How long one fetched AEC reference span may serve consecutive mic
#: frames. Frames arrive every ~32ms and overlap each other, so a short window
#: removes almost all repeat fetches while staying well inside the 2.0s drift
#: bound that decides whether a reference is usable at all.
AEC_CACHE_TTL_SECONDS = float(os.getenv("JARVIS_AEC_CACHE_TTL", "0.25"))
#: [PERF] After this long with no rendered audio, the remote reference is
#: skipped without a request. This is the IDLE case, which is the common one:
#: the transport is consulted per frame, so probing while Jarvis is silent
#: costs a round trip per frame to learn "nothing is playing".
#:
#: [P1-04] The same interval now also bounds the idle RECOVERY probe (see
#: ``RemoteAecTransport._idle_probe_due``). The old skip was a one-way latch:
#: ``_last_fetch_ok_at`` only advanced on a successful NON-EMPTY fetch, and the
#: skip is decided without a request, so once latched the transport could never
#: ask again. A state that only the suppressed thing can exit is a dead
#: transport - a backend-spoken reply (typed UI, async announcement) stayed
#: uncancelled for the rest of the session. Idle is now a hint with a bounded
#: re-check, not a gate.
AEC_IDLE_SKIP_SECONDS = float(os.getenv("JARVIS_AEC_IDLE_SKIP", "1.5"))
#: [P1-04] Circuit breaker. After this many consecutive failed fetches the
#: transport stops dialling for the cooldown instead of paying a failing round
#: trip on every captured frame.
AEC_BREAKER_FAILURES = int(os.getenv("JARVIS_AEC_BREAKER_FAILURES", "3"))
AEC_BREAKER_COOLDOWN_SECONDS = float(
    os.getenv("JARVIS_AEC_BREAKER_COOLDOWN", "15"))
# [S28] Echo alignment. The reference timestamp is recorded when the audio
# actor's blocking write() RETURNS - with latency="high" the sound actually
# leaves the speaker tens to hundreds of ms later, plus the room's acoustic
# delay. A zero-lag comparison over a single 32 ms frame cannot survive that,
# so the degraded path now:
#   * estimates the lag while the first replies play, by cross-correlating the
#     mic against the reference over 0..AEC_MAX_LAG_SECONDS, and locks to the
#     median of a few consistent votes;
#   * fits the gate over the last few frames jointly (~100-200 ms) instead of
#     one 32 ms frame;
#   * keeps the remote reference fresh from a background thread so the
#     capture loop stops paying the fetch inside a real-time frame.
AEC_MAX_LAG_SECONDS = float(os.getenv("JARVIS_AEC_MAX_LAG", "0.6"))
AEC_LAG_CONFIRMATIONS = int(os.getenv("JARVIS_AEC_LAG_CONFIRMATIONS", "3"))
#: A vote below this correlation is noise, not a timing opinion.
AEC_LAG_MIN_CORRELATION = float(
    os.getenv("JARVIS_AEC_LAG_MIN_CORRELATION", "0.30")
)
#: [B105] Joint evidence window: the correlation fit spans this much of the
#: mic history instead of a single ~32 ms frame, so the estimate survives
#: the timing jitter and nonlinearity a laptop-speaker path adds. Votes
#: are separated by at least one window, so this also bounds the cadence.
AEC_LAG_WINDOW_SECONDS = max(
    0.064,
    min(
        0.256,
        float(os.getenv("JARVIS_AEC_LAG_WINDOW", "0.192")),
    ),
)
#: Votes must agree within this spread (about one frame) to lock.
AEC_LAG_SPREAD_SECONDS = 0.04
#: [B108] Lags this session proved WRONG are remembered (bounded) and
#: excluded from later estimates, so a rejected false peak cannot win again.
AEC_LAG_REJECT_MEMORY = 4
#: Frames the gate fits jointly (5 x ~32 ms = ~160 ms of compare window).
AEC_GATE_HISTORY_FRAMES = int(os.getenv("JARVIS_AEC_GATE_HISTORY", "5"))
#: Keep the remote reference prefetching this long after capture activity.
AEC_PREFETCH_SECONDS = float(os.getenv("JARVIS_AEC_PREFETCH", "2.0"))
AEC_PREFETCH_INTERVAL_SECONDS = max(0.05, AEC_CACHE_TTL_SECONDS / 2.0)
# [LF35] The frequency-domain adaptive canceller ("fdaf"). py-webrtc-aec3
# does not exist on PyPI (verified: no distribution under either name), so
# the wiring's preferred model could never install and the signal path ran
# the no-op canceller + correlation gate. Laptop speakers fail that gate -
# OS "audio enhancements" and AGC distort the echo nonlinearly - and every
# frame that slips through is heard as the user: Jarvis barges into its own
# reply and then answers its own words. These knobs tune the in-house
# replacement (see FrequencyDomainEchoCanceller).
#: Transfer smoothing per frame: H = FDAF_ADAPT*H + (1-FDAF_ADAPT)*H_new.
FDAF_ADAPT = float(os.getenv("JARVIS_AEC_FDAF_ADAPT", "0.45"))
#: Adaptation freezes when the residual holds this much of the mic energy
#: AND is uncorrelated with the echo estimate (double-talk).
FDAF_FREEZE_RATIO = float(os.getenv("JARVIS_AEC_FDAF_FREEZE", "0.30"))
#: A window counts as our own playback when subtraction removed at least
#: this many dB (10 dB means the residual is under a third of the mic).
FDAF_SUPPRESS_DB = float(os.getenv("JARVIS_AEC_FDAF_SUPPRESS_DB", "10.0"))
#: Per-bin transfer-magnitude cap (runaway bins on quiet reference).
FDAF_GAIN_MAX = float(os.getenv("JARVIS_AEC_FDAF_GAIN_MAX", "12.0"))
#: Fixed transform size. Covers the mic window PLUS the whole lag search
#: range, so the unknown speaker->mic delay lives inside the per-bin phase
#: of H and never needs a separate delay lock.
FDAF_TRANSFORM_SIZE = 8192
#: After this long without any reference the path estimate is rebuilt from
#: scratch (a device switch or volume change may have moved the path).
FDAF_RESET_AFTER_SECONDS = float(os.getenv("JARVIS_AEC_FDAF_RESET_AFTER", "20"))
#: Per-bin reference-magnitude floor, relative to the mean magnitude: bins
#: where nothing was played carry no transfer information.
FDAF_REF_MAG_FLOOR = 1e-3
#: [LF35] Frames of unconditional adaptation after (re)initialization.
#: Without this the double-talk freeze deadlocks a cold start: the first
#: frames estimate H from whatever the mic has (often not the echo yet),
#: every later frame then looks "loud and uncorrelated" and freezes before
#: H can learn the path.
FDAF_WARMUP_FRAMES = int(os.getenv("JARVIS_AEC_FDAF_WARMUP", "15"))
#: Watchdog: this many consecutive frames where H explains almost nothing
#: mean the estimate went stale - rebuild it (re-enters warmup).
FDAF_LOW_ERLE_STREAK = int(os.getenv("JARVIS_AEC_FDAF_STREAK", "30"))
FDAF_LOW_ERLE_DB = 4.0
#: [B108] A lock must PROVE itself: some frame whose *a-priori* ERLE (the
#: estimate from BEFORE that frame's update - never the in-sample fit of
#: the current window) reaches this many dB within
#: :data:`FDAF_LOCK_PROOF_FRAMES` locked frames. A lock that never proves
#: is re-learned once and then rejected, so a wrong delay can no longer
#: freeze the canceller at 0 dB for the whole session.
FDAF_LOCK_PROOF_DB = float(os.getenv("JARVIS_AEC_FDAF_LOCK_PROOF_DB", "6.0"))
FDAF_LOCK_PROOF_FRAMES = int(
    os.getenv("JARVIS_AEC_FDAF_LOCK_PROOF_FRAMES", "60"))
FDAF_RELOCK_ERLE_FRAMES = 8


class CaptureFrame(namedtuple("CaptureFrame",
                              "pcm had_reference suppressed sample_rate "
                              "generation",
                              defaults=(None,))):
    """One analysed capture frame (F33).

    *pcm* is what the mic contributed after cancellation, *had_reference*
    says playback overlapped the frame, and *suppressed* says the frame was
    recognised as the assistant's own playback (so it must never be committed
    as user speech).

    [B113] *generation* is the canceller's path generation when this frame
    was produced (None for cancellers that do not track one). It changes on
    EVERY drop of the path estimate, so proof from an older generation is
    void: the suppressed frame proved an H that no longer exists.
    """

    __slots__ = ()

    def __bool__(self):
        return bool(self.pcm)


def _import_webrtcaec3():
    """Guarded import - returns the webrtcaec3 module or None."""
    try:
        import webrtcaec3

        return webrtcaec3
    except Exception:
        return None


class StatefulResampler:
    """Continuous sample-rate conversion for a chunked PCM stream.

    Linear interpolation with a global read position and a two-frame tail
    that CARRIES OVER between calls, so a stream split at an arbitrary byte
    boundary resamples to exactly the same signal as the whole stream
    resampled at once (no stride aliasing, no boundary clicks).
    """

    def __init__(self, source_rate, target_rate=AEC_SAMPLE_RATE,
                 channels=1, sample_width=2):
        self.source_rate = int(source_rate or target_rate)
        self.target_rate = int(target_rate or source_rate)
        self.channels = max(1, int(channels or 1))
        self.sample_width = int(sample_width or 2)
        self._step = float(self.source_rate) / float(self.target_rate)
        # Global input frame index of the first frame of the next chunk, and
        # the position (in the same global frame space) of the next output
        # sample. Both survive across calls, so the output is EXACTLY the
        # whole-stream resample for any chunk split.
        self._input_frames = 0   # frames consumed so far
        self._next_pos = 0.0     # global input position of the next output
        self._tail = None        # last <=2 input frames of the previous chunk
        self._carry = b""        # undecoded partial frame across chunks
        self._lock = threading.Lock()

    @property
    def passthrough(self):
        return self.source_rate == self.target_rate and self.channels == 1

    def reset(self):
        with self._lock:
            self._input_frames = 0
            self._next_pos = 0.0
            self._tail = None
            self._carry = b""

    def resample(self, pcm_bytes):
        """Return *pcm_bytes* converted to the target rate (mono s16le)."""
        if not pcm_bytes:
            return b""
        frame_bytes = self.sample_width * self.channels
        with self._lock:
            data = self._carry + bytes(pcm_bytes)
            usable = (len(data) // frame_bytes) * frame_bytes
            self._carry = data[usable:]
            data = data[:usable]
            if not data:
                return b""
            import numpy as np

            if self.sample_width == 2:
                arr = np.frombuffer(data, dtype="<i2").astype(np.float32)
            else:
                arr = np.frombuffer(data, dtype="<i1").astype(np.float32)
            if self.channels > 1:
                arr = arr.reshape((-1, self.channels)).mean(axis=1)
            frames = arr
            n = int(frames.size)
            g0 = self._input_frames
            g_last = g0 + n - 1

            if self.passthrough:
                self._input_frames = g0 + n
                self._next_pos = float(self._input_frames)
                out = frames.astype(np.int16)
                return out.tobytes()

            tail = self._tail
            if tail is None or tail.size == 0:
                source = frames
                source_g0 = g0
            else:
                source = np.concatenate((tail, frames))
                source_g0 = g0 - int(tail.size)

            # Emit every output sample whose interpolation support
            # (floor(pos), floor(pos)+1) is fully inside the data we have.
            positions = []
            pos = self._next_pos
            while int(pos) + 1 <= g_last:
                positions.append(pos)
                pos += self._step
            self._next_pos = pos
            self._input_frames = g0 + n
            # Keep the last two frames for the next chunk's interpolation.
            if n >= 2:
                self._tail = frames[-2:].copy()
            else:
                self._tail = (frames if tail is None
                              else np.concatenate((tail, frames)))[-2:].copy()

            if not positions:
                return b""
            index = np.array(positions, dtype=np.float64)
            low = np.floor(index).astype(np.int64)
            frac = (index - low).astype(np.float32)
            low_idx = low - source_g0
            high_idx = low_idx + 1
            mixed = source[low_idx] * (1.0 - frac) + source[high_idx] * frac
            return np.clip(mixed, -32768, 32767).astype(np.int16).tobytes()


class EchoCanceller:
    """Protocol: combine a mic PCM window with the aligned reference PCM."""

    name = "base"

    def cancel(self, mic_pcm, ref_pcm):
        raise NotImplementedError


class NoOpEchoCanceller(EchoCanceller):
    """Identity fallback: returns the mic signal untouched.

    This is the [ASSUMPTION] boundary - the reference path is proven, but
    measured acoustic performance is a hardware question. It is reported as
    a DEGRADED state (see ``AecSignalPath.state``), never as cancellation.
    """

    name = "noop"
    degraded = True

    def cancel(self, mic_pcm, ref_pcm):
        if ref_pcm:
            self.last_reference_seen = time.monotonic()
        return mic_pcm


class WebRtcAec3Canceller(EchoCanceller):
    """Streaming AEC3 via py-webrtc-aec3 when that package is present."""

    name = "webrtcaec3"
    degraded = False

    def __init__(self, sample_rate=AEC_SAMPLE_RATE):
        webrtcaec3 = _import_webrtcaec3()
        if webrtcaec3 is None:
            raise RuntimeError("py-webrtcaec3 is not installed")
        self._aec = webrtcaec3.StreamingAec3(
            sample_rate=sample_rate,
            frame_length=10,  # ms
            sample_format="S16",
        )

    def cancel(self, mic_pcm, ref_pcm):
        # Feed the reference first so the model has learned recent playback,
        # then cancel the mic window with it.
        if ref_pcm:
            self._aec.feed_reference_pcm(ref_pcm)
        return self._aec.run(mic_pcm)


class DelayEstimator:
    """Cross-correlation lag fit between the mic and the reference (S28).

    The reference span is fetched so that it covers
    ``[window_end - window - max_lag, window_end]``. The mic window is slid
    against the tail of that span and the lag with the best normalised
    correlation wins. A vote is only cast when the best correlation is real
    (:data:`AEC_LAG_MIN_CORRELATION`) so room noise cannot move the estimate,
    and the lock takes :data:`AEC_LAG_CONFIRMATIONS` votes that agree within
    one frame of each other. The lag is a property of the audio path, so it
    is estimated once per session and then simply applied.

    [B105] Evidence is fitted over a JOINT window
    (:data:`AEC_LAG_WINDOW_SECONDS`) accumulated across real capture
    frames instead of one frame: a single ~32 ms window against a delayed,
    imperfectly-linear laptop-speaker path often peaks below the
    correlation bar, and the lock never engages - every later frame then
    passes through uncancelled while the reply is heard as the user. Votes
    are separated by at least one window so consecutive votes carry
    independent evidence, and the lock is decided by the last
    :data:`AEC_LAG_CONFIRMATIONS` votes only, so one old outlier can never
    block three fresh consistent ones.
    """

    name = "delay-estimator"

    def __init__(self, max_lag_seconds=AEC_MAX_LAG_SECONDS,
                 min_correlation=AEC_LAG_MIN_CORRELATION,
                 window_seconds=AEC_LAG_WINDOW_SECONDS):
        self.max_lag_seconds = max(0.0, float(max_lag_seconds))
        self.min_correlation = float(min_correlation)
        self.votes = []
        self.locked_lag_seconds = None
        self.window_seconds = max(0.0, float(window_seconds))
        self._window_samples = max(
            1, int(round(self.window_seconds * AEC_SAMPLE_RATE)))
        #: [B105] Capture-local evidence: the last window of mic samples
        #: (bytes), cleared at capture boundaries and reference gaps.
        self._mic_history = b""
        self._samples_since_vote = 0
        self.last_correlation = None
        self.last_candidate_seconds = None
        self.last_search_seconds = 0.0
        self.last_reason = "not-estimated"
        #: [B108] Independent evidence for a DIFFERENT lag while locked.
        self.challenge_votes = []
        #: [B108] Lags already proved wrong (bounded by
        #: :data:`AEC_LAG_REJECT_MEMORY`); excluded from later estimates.
        self.rejected_lags = []
        self.rejections = 0
        self.relocks = 0
        #: [B108] Peak sharpness diagnostics: how far the winning
        #: correlation stands above the best peak outside its spread.
        self.last_peak_sharpness = None
        self.last_runner_up_seconds = None

    @property
    def locked(self):
        return self.locked_lag_seconds is not None

    def clear_window(self):
        """[B105] Drop capture-local evidence, keep a proven path lag."""
        self._mic_history = b""
        self._samples_since_vote = 0
        self.last_correlation = None
        self.last_candidate_seconds = None
        self.last_search_seconds = 0.0
        self.last_reason = "window-cleared"
        # [B108] Challenges are capture-local too: a gap in the reference
        # must never concatenate evidence across two different playbacks.
        self.challenge_votes = []
        self.last_peak_sharpness = None
        self.last_runner_up_seconds = None

    def state(self):
        """[B105] Plain-dict diagnostics for telemetry (never raises)."""
        return {
            "locked": self.locked,
            "locked_seconds": self.locked_lag_seconds,
            "votes": len(self.votes),
            "best_correlation": self.last_correlation,
            "candidate_seconds": self.last_candidate_seconds,
            "search_seconds": self.last_search_seconds,
            "max_lag_seconds": self.max_lag_seconds,
            "window_seconds": self.window_seconds,
            "min_correlation": self.min_correlation,
            "reason": self.last_reason,
            # [B108] Lock health: challenges and lags proved wrong.
            "challenge_votes": list(self.challenge_votes),
            "rejected_lags": list(self.rejected_lags),
            "rejections": self.rejections,
            "relocks": self.relocks,
            "peak_sharpness": self.last_peak_sharpness,
            "runner_up_seconds": self.last_runner_up_seconds,
        }

    def reset(self, keep_rejections=False):
        self.votes = []
        self.locked_lag_seconds = None
        self.clear_window()
        # [B108] The watchdog reset keeps the knowledge that a lag was
        # already proved wrong; only a device-change reset forgets it.
        if not keep_rejections:
            self.rejected_lags = []
        self.last_reason = "reset"

    @staticmethod
    def _median(values):
        ordered = sorted(values)
        middle = len(ordered) // 2
        return (ordered[middle] if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2.0)

    def _note_locked_evidence(self, correlation, lag_seconds, window_samples):
        """[B108] Record independent evidence while locked, never move on it.

        A lock no longer ends measurement. A strong, independent correlation
        peak at a DIFFERENT lag is a *challenge*; the canceller decides
        whether the challenge is consistent enough (and whether its own
        cancellation is bad enough) to relock.
        """
        self.last_reason = "locked"
        if (correlation < self.min_correlation
                or self._samples_since_vote < window_samples):
            return
        self._samples_since_vote = 0
        if abs(lag_seconds - self.locked_lag_seconds) <= AEC_LAG_SPREAD_SECONDS:
            self.challenge_votes = []          # agreeing evidence
            return
        keep = max(1, AEC_LAG_CONFIRMATIONS) * 2
        self.challenge_votes = (self.challenge_votes + [lag_seconds])[-keep:]
        self.last_reason = "locked-challenged"

    def challenger(self):
        """[B108] Median of the last N consistent challenges, else None."""
        n = max(1, AEC_LAG_CONFIRMATIONS)
        recent = self.challenge_votes[-n:]
        if len(recent) < n or max(recent) - min(recent) > AEC_LAG_SPREAD_SECONDS:
            return None
        return self._median(recent)

    def relock(self, lag_seconds):
        """[B108] Adopt a challenged lag without forgetting it was contested."""
        self.votes = list(self.challenge_votes[-max(1, AEC_LAG_CONFIRMATIONS):])
        self.locked_lag_seconds = float(lag_seconds)
        self.challenge_votes = []
        self.relocks += 1
        self.last_reason = "relocked"

    def reject_lock(self):
        """[B108] A lock that never proved itself is wrong: exclude it."""
        if self.locked_lag_seconds is None:
            return None
        lag = self.locked_lag_seconds
        self.rejected_lags = (self.rejected_lags + [lag])[-AEC_LAG_REJECT_MEMORY:]
        self.locked_lag_seconds = None
        self.votes = []
        self.challenge_votes = []
        self.rejections += 1
        self.last_reason = "lock-rejected"
        return lag

    def estimate(self, mic_pcm, ref_pcm):
        """[B105] Fit bounded joint evidence; return an accepted new vote.

        *mic_pcm* is one capture frame; the last
        :data:`AEC_LAG_WINDOW_SECONDS` of mic audio (this frame included)
        is correlated against *ref_pcm*, whose tail may extend
        :data:`AEC_MAX_LAG_SECONDS` beyond the window. Returns the lag in
        seconds when this call cast a vote, else None. Diagnostics
        (``last_correlation``, ``last_candidate_seconds``,
        ``last_search_seconds``, ``last_reason``) are updated on EVERY
        call, including while locked, so telemetry can tell a cold window
        apart from a decorrelated one.
        """
        import numpy as np

        self.last_correlation = None
        self.last_candidate_seconds = None
        self.last_search_seconds = 0.0

        if not mic_pcm or not ref_pcm:
            self.clear_window()
            self.last_reason = "no-reference-or-mic"
            return None
        if self.max_lag_seconds <= 0.0:
            self.last_reason = "search-disabled"
            return None

        mic_pcm = bytes(mic_pcm[:(len(mic_pcm) // 2) * 2])
        ref_pcm = bytes(ref_pcm[:(len(ref_pcm) // 2) * 2])
        frame_samples = len(mic_pcm) // 2
        if frame_samples == 0:
            self.last_reason = "empty-mic"
            return None

        window_samples = max(frame_samples, self._window_samples)
        self._mic_history = (
            self._mic_history + mic_pcm
        )[-window_samples * 2:]
        self._samples_since_vote += frame_samples

        if len(self._mic_history) < window_samples * 2:
            self.last_reason = "collecting-window"
            return None

        mic = np.frombuffer(self._mic_history, dtype="<i2").astype(np.float64)
        ref = np.frombuffer(ref_pcm, dtype="<i2").astype(np.float64)
        if ref.size < mic.size:
            self.last_reason = "reference-too-short"
            return None

        max_lag_samples = int(round(self.max_lag_seconds * AEC_SAMPLE_RATE))
        # Never search outside the configured range, even if a larger span
        # arrived.
        ref = ref[-(mic.size + max_lag_samples):]
        self.last_search_seconds = (ref.size - mic.size) / float(
            AEC_SAMPLE_RATE)

        mic = mic - mic.mean()
        mic_energy = float(np.dot(mic, mic))
        if mic_energy <= 1e-9:
            self.last_reason = "quiet-mic"
            return None

        # FFT convolution computes the same valid cross-correlation
        # without a growing O(window * search_range) capture-loop cost.
        fft_size = 1 << int(mic.size + ref.size - 2).bit_length()
        convolution = np.fft.irfft(
            np.fft.rfft(ref, n=fft_size)
            * np.fft.rfft(mic[::-1], n=fft_size),
            n=fft_size,
        )
        dots = convolution[mic.size - 1:ref.size]

        sums = np.concatenate(([0.0], np.cumsum(ref)))
        squares = np.concatenate(([0.0], np.cumsum(ref * ref)))
        starts = np.arange(dots.size)
        window_sums = sums[starts + mic.size] - sums[starts]
        ref_energy = (
            squares[starts + mic.size]
            - squares[starts]
            - window_sums * window_sums / mic.size
        )
        ref_energy = np.maximum(ref_energy, 0.0)
        usable = ref_energy > mic_energy * 1e-6
        correlations = np.zeros(dots.size)
        correlations[usable] = (
            np.abs(dots[usable])
            / np.sqrt(mic_energy * ref_energy[usable])
        )
        correlations = np.clip(correlations, 0.0, 1.0)

        # [B108] Lags this session proved WRONG are excluded, so a
        # re-estimate can never land on the same false peak again.
        lags = (dots.size - 1 - np.arange(dots.size)) / float(AEC_SAMPLE_RATE)
        for rejected in self.rejected_lags:
            correlations[np.abs(lags - rejected) <= AEC_LAG_SPREAD_SECONDS] = 0.0

        best = int(np.argmax(correlations))
        correlation = float(correlations[best])
        lag_seconds = float(lags[best])
        self.last_correlation = correlation
        self.last_candidate_seconds = lag_seconds
        # [B108] Peak sharpness: a flatter correlation landscape is a
        # weaker timing opinion even when its absolute peak clears the bar.
        far = np.abs(lags - lag_seconds) > AEC_LAG_SPREAD_SECONDS
        if np.any(far):
            runner = int(np.argmax(np.where(far, correlations, -1.0)))
            runner_corr = float(correlations[runner])
            self.last_runner_up_seconds = float(lags[runner])
            self.last_peak_sharpness = (correlation / runner_corr
                                        if runner_corr > 1e-9 else None)

        if self.locked:
            # [B108] A lock no longer ends measurement: independent,
            # correlated evidence for a DIFFERENT lag is recorded as a
            # challenge (the canceller decides whether to relock).
            self._note_locked_evidence(correlation, lag_seconds, mic.size)
            return None
        if correlation < self.min_correlation:
            self.last_reason = "below-correlation-threshold"
            return None
        if self._samples_since_vote < mic.size:
            self.last_reason = "waiting-for-independent-window"
            return None

        self._samples_since_vote = 0
        self.votes.append(lag_seconds)
        self.votes = self.votes[-24:]

        confirmations = max(1, AEC_LAG_CONFIRMATIONS)
        recent = self.votes[-confirmations:]
        if (
            len(recent) == confirmations
            and max(recent) - min(recent) <= AEC_LAG_SPREAD_SECONDS
        ):
            self.locked_lag_seconds = self._median(recent)
            self.last_reason = "locked"
        else:
            self.last_reason = "accepted-vote"

        return lag_seconds


class ReferenceEchoGate:
    """Decide whether a mic window IS our own playback (degraded path).

    When no real AEC model is installed the mic passes through unfiltered, and
    no downstream VAD can be trusted to tell the assistant's own voice from
    the user's: WebRTC VAD adapts to what it heard before, so it happily calls
    a loud playback window "speech". This gate compares the window against the
    time-aligned reference with a single-tap least-squares fit:

      * ``gain``         best-fit playback scale;
      * ``suppression``  dB of window energy removed by subtracting gain*ref;
      * ``correlation``  |<mic, ref>| / sqrt(E_mic * E_ref).

    A window that is (almost) entirely playback has correlation near 1 and a
    large suppression; genuine double-talk does not. The residual is what the
    user actually said, so callers can keep analysing it.
    """

    name = "echo-gate"

    def __init__(self, min_correlation=0.55, min_suppression_db=8.0,
                 residual_floor=8.0):
        self.min_correlation = float(min_correlation)
        self.min_suppression_db = float(min_suppression_db)
        self.residual_floor = float(residual_floor)

    def analyse(self, mic_pcm, ref_pcm, history_mic=b"", history_ref=b""):
        """Return ``(is_echo, residual_pcm, metrics)``.

        When *history_mic*/*history_ref* carry the immediately preceding
        frames and their aligned references (S28), the least-squares fit runs
        on the concatenation - about 100-200 ms - because a single 32 ms
        window cannot tolerate residual timing jitter. The decision is made
        on the joint fit, but the residual still covers ONLY the current
        frame, which is what the caller returns downstream.
        """
        metrics = {"correlation": 0.0, "suppression_db": 0.0, "is_echo": False}
        if not mic_pcm or not ref_pcm:
            return False, mic_pcm, metrics
        import numpy as np

        def as_float(data):
            usable = (len(data) // 2) * 2
            return np.frombuffer(data[:usable], dtype="<i2").astype(np.float64)

        def as_pair(data, size):
            arr = as_float(data)
            if arr.size < size:
                arr = np.concatenate((np.zeros(size - arr.size), arr))
            elif arr.size > size:
                arr = arr[arr.size - size:]
            return arr

        mic = as_float(mic_pcm)
        ref = as_float(ref_pcm)
        if mic.size == 0 or ref.size == 0:
            return False, mic_pcm, metrics
        # Align lengths: the reference span is asked for the window's length,
        # but a partial span is padded so the fit stays comparable.
        if ref.size < mic.size:
            ref = np.concatenate((np.zeros(mic.size - ref.size), ref))
        elif ref.size > mic.size:
            ref = ref[ref.size - mic.size:]
        # [S28] Joint compare window: the past frames join the fit, the
        # current frame keeps its own residual.
        fit_mic, fit_ref = mic, ref
        if history_mic and history_ref:
            size = mic.size
            past_mic = as_pair(history_mic, size)
            past_ref = as_pair(history_ref, size)
            fit_mic = np.concatenate((past_mic, mic))
            fit_ref = np.concatenate((past_ref, ref))

        mic_energy = float(np.dot(fit_mic, fit_mic))
        ref_energy = float(np.dot(fit_ref, fit_ref))
        if mic_energy <= 0.0 or ref_energy <= 0.0:
            return False, mic_pcm, metrics
        cross = float(np.dot(fit_mic, fit_ref))
        correlation = abs(cross) / math.sqrt(mic_energy * ref_energy)
        gain = cross / ref_energy
        fit_residual = fit_mic - gain * fit_ref
        residual_energy = float(np.dot(fit_residual, fit_residual))
        if residual_energy <= 0.0:
            suppression = 120.0
        else:
            suppression = 10.0 * math.log10(mic_energy / residual_energy)
        is_echo = (correlation >= self.min_correlation and
                   suppression >= self.min_suppression_db)
        metrics.update({"correlation": correlation,
                        "suppression_db": suppression,
                        "gain": gain,
                        "is_echo": is_echo})
        if not is_echo:
            return False, mic_pcm, metrics
        residual = mic - gain * ref
        return True, np.clip(residual, -32768, 32767).astype(np.int16).tobytes(), metrics


class FrequencyDomainEchoCanceller(EchoCanceller):
    """[LF35] Real in-house echo canceller: a block frequency-domain
    adaptive filter.

    The standard block FDAF shape, in numpy, no third-party model:

    * per frequency bin, the transfer H mapping the rendered reference to
      the mic is estimated as Y/X and smoothed over frames (adaptation);
    * the speaker->mic delay is measured FIRST (the same cross-correlation
      lock the degraded gate uses): an unaligned mic/reference pair is
      edge-limited to a few dB no matter how good the estimator is, so H
      is estimated against the lag-ALIGNED reference window, where the
      per-bin relation is exact for a linear echo path; if the delay never
      locks on heavily distorted hardware, the mic passes through
      honestly and the listener's other LF35 layers carry the protection;
    * the estimated echo (H*X, back in time domain) is subtracted; the
      residual is what the user actually contributed, so genuine
      double-talk survives (that is what makes barge-in keep working on
      speakers);
    * adaptation FREEZES while the residual is loud AND uncorrelated with
      the echo estimate (a human speaking over playback), so the user's
      voice is never "adapted away". A path CHANGE leaves the residual
      echo-correlated, so adaptation continues through it.

    ``last_suppressed`` (per cancel() call) says the window was essentially
    entirely our own playback: subtraction removed at least
    :data:`FDAF_SUPPRESS_DB` of energy. The signal path turns that into
    ``CaptureFrame.suppressed``, which the listener must never commit as
    user speech or let fire a barge-in.
    """

    name = "fdaf"
    degraded = False
    #: Wants the reference span covering window + lag range: the extra
    #: history feeds the internal delay estimator and keeps the reference
    #: tail long enough to slice lag-aligned windows from it.
    wants_full_reference = True
    reports_suppression = True

    def __init__(self, adapt=FDAF_ADAPT, freeze_ratio=FDAF_FREEZE_RATIO,
                 suppress_db=FDAF_SUPPRESS_DB, gain_max=FDAF_GAIN_MAX,
                 transform_size=FDAF_TRANSFORM_SIZE,
                 warmup_frames=FDAF_WARMUP_FRAMES,
                 max_lag_seconds=AEC_MAX_LAG_SECONDS):
        self.adapt = float(adapt)
        self.freeze_ratio = float(freeze_ratio)
        self.suppress_db = float(suppress_db)
        self.gain_max = float(gain_max)
        self.transform_size = int(transform_size)
        self.warmup_frames = max(0, int(warmup_frames))
        #: The speaker->mic delay, measured not absorbed (see the class
        #: docstring: alignment first, estimate second).
        self._lag = DelayEstimator(max_lag_seconds=max_lag_seconds)
        self._h = None
        #: Frames of unconditional adaptation left (cold start / after a
        #: watchdog reset). See FDAF_WARMUP_FRAMES.
        self._warmup_left = 0
        #: Freeze hold-off after double-talk: once the user's voice is in
        #: the estimation window, every per-frame Y/X estimate is polluted
        #: until the window has rolled past it, so adaptation stays frozen
        #: for one window-length after the double-talk ends.
        self._dt_holdoff = 0
        #: Consecutive frames where H explained almost nothing.
        self._low_erle_streak = 0
        #: Rolling signal history (float samples). The mic tail holds one
        #: transform window; the reference tail holds window + lag range so
        #: a lag-aligned window can be sliced out of it. Tails advance by
        #: exactly the frame length per call - one shared timeline.
        self._mic_tail = None
        self._ref_tail = None
        self._ref_capacity = int(transform_size +
                                 max_lag_seconds * AEC_SAMPLE_RATE)
        self.last_suppressed = False
        self.last_erle_db = None
        #: [B108] ERLE of the estimate from BEFORE this frame's update.
        #: This is the only honest "did our playback cancel?" number: the
        #: a-posteriori residual contains the frame's own Y/X fit.
        self.last_erle_prior_db = None
        #: [B108] Lock proof / relearn / reject state.
        self._lock_frames = 0
        self._lock_proven = False
        self._lock_relearns = 0
        self._recent_erle = deque(maxlen=FDAF_RELOCK_ERLE_FRAMES)
        #: [B113] Monotonic id of the current H estimate. It bumps on EVERY
        #: H drop (reset, watchdog reset, relearn, relock, lock rejection), so
        #: the listener can invalidate proof granted by a dead estimate.
        self._path_generation = 0
        self.metrics = {"frames": 0, "adapted": 0, "frozen": 0,
                        "suppressed": 0, "resets": 0, "lag_locked": False,
                        "lag_seconds": None, "last_erle_db": None,
                        "lag_best_correlation": None,
                        "lag_candidate_seconds": None,
                        "lag_votes": 0,
                        "lag_search_seconds": 0.0,
                        "lag_reason": "not-estimated",
                        "lock_proven": False, "lock_frames": 0,
                        "relearns": 0, "relocks": 0,
                        "lock_rejections": 0, "last_erle_prior_db": None,
                        "lag_challenge_votes": 0, "lag_rejected": [],
                        "lag_peak_sharpness": None,
                        "path_generation": 0}

    @property
    def path_generation(self):
        """[B113] Monotonic id of the current H estimate (bumps on drop)."""
        return self._path_generation

    def reset(self, keep_rejections=False):
        """Drop the delay estimate, the path estimate and the signal
        history (a long silence may mean a device or volume change moved
        the acoustic path)."""
        self._lag.reset(keep_rejections=keep_rejections)
        self._h = None
        self._mic_tail = None
        self._ref_tail = None
        self._warmup_left = 0
        self._dt_holdoff = 0
        self._low_erle_streak = 0
        # [B108] A reset drops the lock proof and all lock-health counters'
        # per-lock state; cumulative counters (resets/relearns/relocks/
        # rejections) are deliberately kept.
        self._drop_estimate()
        self.last_erle_prior_db = None
        self.metrics["resets"] += 1
        # [B103] A reset drops the lag lock and the ERLE history: exported
        # telemetry must never report a stale delay lock after it.
        # [B105] The lag diagnostics go with it - no stale correlation
        # peak, candidate or vote count survives reset.
        self.metrics["lag_locked"] = False
        self.metrics["lag_seconds"] = None
        self.metrics["last_erle_db"] = None
        self.metrics["last_erle_prior_db"] = None
        self.last_suppressed = False
        self.last_erle_db = None
        self.metrics.update({
            "lag_best_correlation": None,
            "lag_candidate_seconds": None,
            "lag_votes": 0,
            "lag_search_seconds": 0.0,
            "lag_reason": "reset",
            "lag_challenge_votes": 0,
            "lag_rejected": [],
            "lag_peak_sharpness": None,
        })

    def _advance_tail(self, tail, samples, capacity):
        import numpy as np

        if samples.size:
            tail = samples if tail is None else np.concatenate((tail, samples))
        if tail is not None and tail.size > capacity:
            tail = tail[tail.size - capacity:]
        return tail

    def _clip_gain(self, h):
        import numpy as np

        mag = np.abs(h)
        over = mag > self.gain_max
        if np.any(over):
            h = np.where(over, h * (self.gain_max / np.maximum(mag, 1e-12)),
                         h)
        return h

    def _window_from(self, tail, back_samples=0):
        """Last ``transform_size`` samples of *tail*, ending *back_samples*
        before the tail's end (the lag alignment), zero-padded at the
        front when the history is still short."""
        import numpy as np

        m = self.transform_size
        win = np.zeros(m)
        if tail is None or not tail.size:
            return win
        end = tail.size - int(back_samples)
        if end <= 0:
            return win
        take = min(m, end)
        win[m - take:] = tail[end - take:end]
        return win

    # ── [B108] lock health ─────────────────────────────────────────────

    def _drop_estimate(self, keep_relearns=False):
        """Re-learn H from scratch on the current lock (or a new one)."""
        # [B113] Every H drop (reset, watchdog reset, relearn, relock, lock
        # rejection) starts a new estimate: listeners must re-prove trust.
        self._path_generation += 1
        self.metrics["path_generation"] = self._path_generation
        self._h = None
        self._warmup_left = 0
        self._dt_holdoff = 0
        self._low_erle_streak = 0
        self._lock_frames = 0
        self._lock_proven = False
        if not keep_relearns:
            self._lock_relearns = 0
        self._recent_erle.clear()
        self.metrics["lock_proven"] = False
        self.metrics["lock_frames"] = 0

    def _sync_lag_metrics(self):
        s = self._lag.state()
        self.metrics.update({
            "lag_locked": s["locked"], "lag_seconds": s["locked_seconds"],
            "lag_best_correlation": s["best_correlation"],
            "lag_candidate_seconds": s["candidate_seconds"],
            "lag_votes": s["votes"], "lag_search_seconds": s["search_seconds"],
            "lag_reason": s["reason"],
            "lag_challenge_votes": len(s["challenge_votes"]),
            "lag_rejected": [round(x, 3) for x in s["rejected_lags"]],
            "lag_peak_sharpness": s["peak_sharpness"],
        })

    def _erle_recently_low(self):
        if len(self._recent_erle) < 5:
            return False
        ordered = sorted(self._recent_erle)
        return ordered[len(ordered) // 2] < FDAF_LOW_ERLE_DB

    def _note_lock_evidence(self, erle_prior, did_adapt):
        """[B108] Lock proof + watchdog, on a-priori ERLE only.

        A lock that never cancels anything must not survive: after
        :data:`FDAF_LOCK_PROOF_FRAMES` locked frames without one honest
        proof, H is re-learned once (the estimate may simply be wrong);
        if it still never proves, the LAG is wrong and is rejected.
        Frames are counted whether or not adaptation happened, so an
        estimate frozen on every frame can no longer escape the watchdog.
        """
        if erle_prior is None:
            return
        self._recent_erle.append(erle_prior)
        self._lock_frames += 1
        if erle_prior >= FDAF_LOCK_PROOF_DB:
            self._lock_proven = True
        self.metrics["lock_proven"] = self._lock_proven
        self.metrics["lock_frames"] = self._lock_frames
        if not self._lock_proven:
            if self._lock_frames >= FDAF_LOCK_PROOF_FRAMES:
                if self._lock_relearns == 0:
                    # 1st: the estimate may have converged on the wrong
                    # window. Re-learn H on the same lock first.
                    self._lock_relearns = 1
                    self.metrics["relearns"] += 1
                    self._drop_estimate(keep_relearns=True)
                else:
                    # 2nd: H was given a fair chance and still cancels
                    # nothing, so the LAG itself is wrong.
                    self._lag.reject_lock()
                    self.metrics["lock_rejections"] += 1
                    self._drop_estimate()
                self._sync_lag_metrics()
            return
        if erle_prior >= FDAF_LOW_ERLE_DB:
            self._low_erle_streak = 0
        elif did_adapt:
            self._low_erle_streak += 1
        if self._low_erle_streak >= FDAF_LOW_ERLE_STREAK:
            # The gate keeps the rejection memory: a reset drops H, not
            # the knowledge that a particular lag was already proved bad.
            self.reset(keep_rejections=True)

    def cancel(self, mic_pcm, ref_pcm):
        self.last_suppressed = False
        self.last_erle_db = None
        self.last_erle_prior_db = None
        # Never let a previous frame's prior ERLE describe an early return
        # (silence, no reference, unlocked) in telemetry.
        self.metrics["last_erle_prior_db"] = None
        if not mic_pcm:
            return mic_pcm

        import numpy as np

        def as_float(data):
            usable = (len(data) // 2) * 2
            return np.frombuffer(data[:usable], dtype="<i2").astype(np.float64)

        mic = as_float(mic_pcm)
        n = mic.size
        m = self.transform_size
        if n == 0 or m <= n:
            # Degenerate window (no lag room): pass through, never suppress.
            return mic_pcm

        # Advance the shared timeline FIRST: even a silent or reference-free
        # frame moves both tails, so the windows stay time-aligned.
        ref = as_float(ref_pcm) if ref_pcm else np.zeros(0)
        self._mic_tail = self._advance_tail(self._mic_tail, mic, m)
        # Only the FRESHEST n samples of the span join the reference tail -
        # consecutive spans overlap by the lag range, and appending the
        # whole span would duplicate history.
        fresh_ref = np.zeros(n)
        take = min(n, ref.size)
        if take:
            fresh_ref[n - take:] = ref[ref.size - take:]
        self._ref_tail = self._advance_tail(self._ref_tail, fresh_ref,
                                            self._ref_capacity)

        mic_energy = float(np.dot(mic, mic))
        mic_rms = math.sqrt(mic_energy / n)
        if mic_rms < 3.0:
            # Digital silence: certainly not the user; nothing to subtract.
            self.metrics["frames"] += 1
            self.metrics["suppressed"] += 1
            self.last_suppressed = True
            return mic_pcm
        if ref.size <= n:
            # No overlapping playback: pass the mic through untouched and
            # KEEP the path estimate - the acoustic path outlives a gap.
            # [B105] The joint lag window does NOT survive the gap: its
            # evidence must never concatenate across reference-free audio.
            self._lag.clear_window()
            self.metrics["frames"] += 1
            return mic_pcm

        # The speaker->mic delay must be known before H can be estimated.
        # Until the cross-correlation lock agrees, pass through honestly.
        # [B105] Estimation runs on EVERY reference-bearing frame: after
        # the lock it only updates diagnostics (correlation peak, candidate
        # lag, searched range, reason) so telemetry can tell a cold window
        # apart from a genuinely decorrelated path.
        self._lag.estimate(mic_pcm, ref_pcm)
        # [B108] A locked lag that consistently disagrees with the measured
        # correlation peak WHILE H removes nothing is the wrong lag: relock.
        if self._lag.locked and self._erle_recently_low():
            challenger = self._lag.challenger()
            if challenger is not None:
                self._lag.relock(challenger)
                self._drop_estimate()
                self.metrics["relocks"] += 1
        self._sync_lag_metrics()
        if not self._lag.locked:
            self.metrics["frames"] += 1
            self.metrics["last_erle_db"] = None
            return mic_pcm

        lag_samples = int(self._lag.locked_lag_seconds * AEC_SAMPLE_RATE)

        mic_win = self._window_from(self._mic_tail)
        ref_win = self._window_from(self._ref_tail,
                                    back_samples=lag_samples)

        x = np.fft.rfft(ref_win)
        y = np.fft.rfft(mic_win)
        x_mag = np.abs(x)
        x_energy = float(np.dot(x_mag, x_mag))
        if x_energy <= 0.0:
            self.metrics["frames"] += 1
            return mic_pcm

        floor = FDAF_REF_MAG_FLOOR * math.sqrt(x_energy / x_mag.size)
        usable = x_mag > floor
        h_new = np.zeros_like(x)
        h_new[usable] = y[usable] / x[usable]
        h_new = self._clip_gain(h_new)

        def residual_with(h):
            e = np.fft.irfft(h * x, n=m)[m - n:]
            return mic - e, e

        # [B108] ERLE of the estimate from BEFORE this frame's update. The
        # a-posteriori number below includes this frame's own Y/X fit, which
        # is exactly what must never decide "our own playback".
        erle_prior = None
        if self._h is None:
            self._h = h_new
            self._warmup_left = self.warmup_frames
            self.metrics["adapted"] += 1
            did_adapt = True
        else:
            residual, echo = residual_with(self._h)
            res_energy = float(np.dot(residual, residual))
            echo_energy = float(np.dot(echo, echo))
            ratio = res_energy / mic_energy
            erle_prior = 10.0 * math.log10(1.0 / max(ratio, 1e-12))
            if res_energy > 0.0 and echo_energy > 0.0:
                corr = (abs(float(np.dot(residual, echo))) /
                        math.sqrt(res_energy * echo_energy))
            else:
                corr = 0.0
            # Adaptation is allowed while warming up (a cold start must not
            # deadlock behind the double-talk freeze), when the residual is
            # quiet (pure echo, nothing to protect), or when the residual is
            # still echo-CORRELATED (a path change, not a human voice).
            adapt = (self._warmup_left > 0 or
                     ratio <= self.freeze_ratio or corr >= 0.5)
            if self._warmup_left > 0:
                self._warmup_left -= 1
            if not adapt:
                # Fresh double-talk: hold the freeze for one window-length
                # AFTER it ends - the estimation window still contains the
                # user's voice until it has rolled past, and adapting on a
                # polluted window re-learns their voice into H.
                self._dt_holdoff = -(-m // n)
            elif self._dt_holdoff > 0:
                self._dt_holdoff -= 1
                adapt = False
            if adapt:
                blended = self._h.copy()
                blended[usable] = (self.adapt * self._h[usable] +
                                   (1.0 - self.adapt) * h_new[usable])
                self._h = self._clip_gain(blended)
                self.metrics["adapted"] += 1
                did_adapt = True
            else:
                self.metrics["frozen"] += 1
                did_adapt = False

        residual, _echo = residual_with(self._h)
        res_energy = float(np.dot(residual, residual))
        erle = 10.0 * math.log10(mic_energy / max(res_energy, 1e-9))
        self.last_erle_db = erle
        self.last_erle_prior_db = erle_prior
        self.metrics["last_erle_db"] = round(erle, 1)
        self.metrics["last_erle_prior_db"] = (None if erle_prior is None
                                              else round(erle_prior, 1))
        # [B108] "Our own playback" must be PREDICTED cancellation: the
        # a-posteriori residual includes this frame's own Y/X fit, which
        # cancels ANY mic content (on the first locked frame, exactly) -
        # including a user's voice.
        self.last_suppressed = (erle_prior is not None and
                                min(erle, erle_prior) >= self.suppress_db)
        self.metrics["frames"] += 1
        if self.last_suppressed:
            self.metrics["suppressed"] += 1
        self._note_lock_evidence(erle_prior, did_adapt)
        out = np.clip(residual, -32768, 32767).astype(np.int16)
        return out.tobytes()


def build_canceller():
    """Select the best AEC implementation available at runtime."""
    if AEC_ENABLED and _import_webrtcaec3() is not None:
        try:
            return WebRtcAec3Canceller()
        except Exception as exc:
            print(f"[AEC] webrtcaec3 init failed ({exc}) - trying fdaf")
    if AEC_ENABLED:
        # [LF35] The in-house canceller: py-webrtc-aec3 is not installable,
        # so this is the path every real session takes.
        try:
            return FrequencyDomainEchoCanceller()
        except Exception as exc:
            print(f"[AEC] fdaf init failed ({exc}) - no-op path")
    return NoOpEchoCanceller()


class RemoteAecTransport:
    """Fetch the AEC reference from the process that renders playback.

    F33: the reference buffer is process-local, but the API process voices
    replies while the voice process owns the microphone, so a listener-side
    buffer would always be empty (``had_reference`` False forever). This is
    the missing PCM transport: the voice side asks the API for the rendered
    span that overlaps its mic window, together with the age of that span
    measured on the API's own clock - which is what establishes the shared
    timeline without comparing unrelated monotonic clocks.

    Never raises and never blocks longer than ``AEC_REMOTE_TIMEOUT``; a
    failure simply means "no reference", which the caller reports as
    ``had_reference=False`` (unfiltered semantics, explicitly degraded).
    """

    def __init__(self, base_url=None, timeout=None, cache_seconds=None,
                 idle_skip_seconds=None, breaker_failures=None,
                 breaker_cooldown_seconds=None):
        if base_url is None:
            try:
                from backend.config import BACKEND_PORT
                base_url = f"http://127.0.0.1:{BACKEND_PORT}"
            except Exception:  # pragma: no cover
                base_url = "http://127.0.0.1:9999"
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout or AEC_REMOTE_TIMEOUT)
        # Existing keys are kept verbatim for backward compatibility; P1-04
        # only ADDS keys (never renames or removes one).
        self.stats = {"fetches": 0, "hits": 0, "errors": 0,
                      "cache_hits": 0, "skipped_idle": 0,
                      "last_age_seconds": None, "last_error": None,
                      # [P1-04] additions
                      "idle_reprobes": 0, "auth_failures": 0,
                      "last_status": None, "circuit_skips": 0,
                      "consecutive_failures": 0,
                      # [S28] background prefetcher
                      "prefetch_polls": 0, "prefetch_fetches": 0}
        # [PERF] This transport is consulted once per captured mic frame from
        # inside the real-time capture loop. Three guards keep it from turning
        # into a per-frame HTTP round trip:
        #   * a short TTL cache keyed on the mic window - consecutive frames
        #     overlap the same rendered audio, so one fetch serves many frames;
        #   * an idle early-out - when nothing has been rendered recently there
        #     is provably no reference to fetch, so the request is skipped;
        #   * a circuit breaker - a failing endpoint is not re-dialled per frame.
        #
        # [S28] RESOLVED (was the P1-04 residual risk): the fetch no longer
        # runs ON the capture hot path. A background daemon keeps the latest
        # reference span fetched while captures are live (see
        # ``note_capture_activity``/``_prefetch_loop``), so the capture loop's
        # own ``fetch_reference`` is a cache hit except in the rare gap where
        # the prefetcher has not landed yet - the guards above still bound
        # that fallback the same way they always did.
        self._cache_ttl = float(
            cache_seconds if cache_seconds is not None else AEC_CACHE_TTL_SECONDS)
        self._idle_skip = float(
            idle_skip_seconds if idle_skip_seconds is not None
            else AEC_IDLE_SKIP_SECONDS)
        self._breaker_limit = max(
            1, int(breaker_failures if breaker_failures is not None
                   else AEC_BREAKER_FAILURES))
        self._breaker_cooldown = float(
            breaker_cooldown_seconds if breaker_cooldown_seconds is not None
            else AEC_BREAKER_COOLDOWN_SECONDS)
        self._cache_lock = threading.Lock()
        self._cache_pcm = b""
        self._cache_age = None
        self._cache_at = 0.0
        #: [P1-04] The mic window the cached span was fetched FOR. Without this
        #: the cache keyed on time alone and could serve a span rendered for one
        #: mic window to a window at a different point on the timeline.
        self._cache_mic_t_end = None
        self._last_fetch_ok_at = None
        #: [P1-04] When the last idle recovery probe was issued.
        self._last_idle_probe_at = None
        #: [P1-04] Monotonic time the breaker reopens at, plus the last reason
        #: reported through state().
        self._breaker_open_until = 0.0
        self._reason = "never_fetched"
        # [S28] Background prefetch state. The capture loop only notes that it
        # is live; the daemon below keeps the cache fresh so the loop itself
        # never blocks on the network.
        self._prefetch_until = 0.0
        self._prefetch_duration = 0.1
        self._prefetch_thread = None

    # ── [S28] background prefetch ──────────────────────────────────────
    def note_capture_activity(self, duration_seconds=None):
        """Mark the capture loop as live; (re)arm the prefetch daemon.

        Cheap and lock-only: called from ``fetch_reference`` and from
        ``AecSignalPath.begin_capture``, never from the audio thread's
        blocking work.
        """
        with self._cache_lock:
            if duration_seconds is not None:
                try:
                    self._prefetch_duration = max(
                        0.05, min(2.0, float(duration_seconds)))
                except Exception:
                    pass
            self._prefetch_until = time.monotonic() + AEC_PREFETCH_SECONDS
            if self._prefetch_thread is None or \
                    not self._prefetch_thread.is_alive():
                thread = threading.Thread(
                    target=self._prefetch_loop, name="aec-refetch",
                    daemon=True)
                self._prefetch_thread = thread
                thread.start()

    def _prefetch_loop(self):
        """Keep the latest reference span fetched while captures are live.

        Runs the SAME guarded fetch path (cache, idle skip, breaker) as the
        capture loop would - it only ever replaces a blocking fetch with a
        background one, never widens what may be requested.
        """
        while True:
            time.sleep(AEC_PREFETCH_INTERVAL_SECONDS)
            with self._cache_lock:
                active_until = self._prefetch_until
                duration = self._prefetch_duration
            if time.monotonic() > active_until:
                continue
            with self._cache_lock:
                self.stats["prefetch_polls"] += 1
            try:
                fetched = self._fetch_reference(duration, None)
            except Exception:  # pragma: no cover - guarded upstream too
                continue
            if fetched[0]:
                with self._cache_lock:
                    self.stats["prefetch_fetches"] += 1

    def fetch_reference(self, duration_seconds, mic_t_end=None):
        """Return ``(pcm_16k_mono, age_seconds)`` or ``(b"", None)``.

        Never raises: any failure means "no reference", which the caller
        reports as ``had_reference=False`` (an explicitly degraded state the
        listener already handles). [S28] Every live call also re-arms the
        background prefetch, so a cache miss inside the capture loop gets
        rarer the longer the capture runs.
        """
        self.note_capture_activity(duration_seconds)
        try:
            return self._fetch_reference(duration_seconds, mic_t_end)
        except Exception as exc:  # pragma: no cover - last-resort guard
            with self._cache_lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(exc)
                self._reason = "exception"
            return b"", None

    # ── circuit breaker ─────────────────────────────────────────────────
    def _breaker_open(self, now):
        return now < self._breaker_open_until

    def _note_failure(self, reason, status=None):
        """Record a failed fetch; open the breaker once it trips."""
        with self._cache_lock:
            self._reason = reason
            if status is not None:
                self.stats["last_status"] = status
            self.stats["consecutive_failures"] += 1
            if self.stats["consecutive_failures"] >= self._breaker_limit:
                self._breaker_open_until = (time.monotonic() +
                                            self._breaker_cooldown)

    def _note_success(self):
        with self._cache_lock:
            self.stats["consecutive_failures"] = 0
            self._breaker_open_until = 0.0

    def _remote_is_idle(self, mic_t_end=None):
        """True when no rendered audio exists near *mic_t_end* to cancel against.

        ``/aec/reference`` answers with the span the API process actually
        rendered, so when its own clock says nothing was played around this
        window the answer is necessarily empty. Probing costs a round trip on
        every frame, so the last successful non-empty fetch ages out into
        "idle" instead.

        [P1-04] This is now only a HINT, never a gate: the caller still
        re-probes on a bounded cadence (see :meth:`_idle_probe_due`). The old
        behaviour was a one-way latch - ``_last_fetch_ok_at`` advanced only on
        a successful NON-EMPTY fetch, while the skip is decided WITHOUT a
        request, so once latched nothing could ever clear it. The transport
        could not recover on its own and a backend-spoken reply (typed UI,
        async announcement) stayed uncancelled for the rest of the session.
        """
        with self._cache_lock:
            last_ok = self._last_fetch_ok_at
        if last_ok is None:
            # Never seen playback: one probe is still required before we can
            # claim silence, otherwise the first real utterance would be
            # uncancelled.
            return False
        return (time.monotonic() - last_ok) > self._idle_skip

    def _idle_probe_due(self, now):
        """At most ONE recovery probe per ``_idle_skip`` window while idle.

        This is what makes the latch recoverable. The early-out still does the
        bulk of the work (frames arrive every ~32ms, so most calls skip without
        a request), but the transport always re-checks the renderer on its own
        rather than waiting for playback that only it could detect.
        """
        with self._cache_lock:
            last_probe = self._last_idle_probe_at
        return last_probe is None or (now - last_probe) >= self._idle_skip

    def _cache_is_usable(self, now, mic_t_end):
        """Is the cached span still valid FOR THIS MIC WINDOW?

        Three independent conditions. [P1-04] The cache used to key on time
        since the fetch alone, so a span fetched for one mic window could be
        served to a window at a different point on the timeline - a span must
        never be replayed onto a different window.
        """
        with self._cache_lock:
            if not self._cache_pcm:
                return False
            if (now - self._cache_at) >= self._cache_ttl:
                return False
            age, at, cached_window = (self._cache_age, self._cache_at,
                                      self._cache_mic_t_end)
        if age is not None:
            # A span keeps ageing after it was fetched, so compare the age it
            # will have NOW against the same drift bound that decides whether a
            # reference is usable at all.
            if (float(age) + (now - at)) > REFERENCE_MAX_DRIFT_SECONDS:
                return False
        if mic_t_end is not None and cached_window is not None:
            if abs(float(mic_t_end) - float(cached_window)) > REFERENCE_MAX_DRIFT_SECONDS:
                return False
        return True

    def _fetch_reference(self, duration_seconds, mic_t_end):
        now = time.monotonic()
        if self._cache_is_usable(now, mic_t_end):
            with self._cache_lock:
                self.stats["cache_hits"] += 1
                return self._cache_pcm, self._cache_age
        if self._breaker_open(now):
            with self._cache_lock:
                self.stats["circuit_skips"] += 1
                self._reason = "circuit_open"
            return b"", None
        if self._remote_is_idle(mic_t_end):
            # [P1-04] Bounded recovery probe. A latched "idle" must never be a
            # state only the suppressed request could exit, so we re-check the
            # renderer at most once per idle window.
            if not self._idle_probe_due(now):
                with self._cache_lock:
                    self.stats["skipped_idle"] += 1
                return b"", None
            with self._cache_lock:
                self._last_idle_probe_at = now
                self.stats["idle_reprobes"] += 1
        with self._cache_lock:
            self.stats["fetches"] += 1
        url = (f"{self.base_url}/aec/reference"
               f"?seconds={max(0.0, float(duration_seconds or 0.0)):.3f}")
        try:
            # [P1-04] Auth. Every endpoint except GET /health is fail-closed
            # behind X-Jarvis-Token (local_auth), so an unauthenticated fetch
            # 401s, returns b"" and silently leaves every backend-spoken reply
            # uncancelled. auth_headers() is the ONE authenticated client header
            # set (F51) - the same call listener.py makes for POST /speak/stop.
            request = Request(url, method="GET",
                              headers=local_auth.auth_headers())
            with urlopen(request, timeout=self.timeout) as response:
                status = (getattr(response, "status", None)
                          or getattr(response, "code", None))
                payload = json.loads(response.read().decode("utf-8") or "{}")
        except HTTPError as exc:
            status = getattr(exc, "code", None)
            with self._cache_lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = f"HTTP {status}: {exc}"
            if status in (401, 403):
                # Never silently swallow an auth failure: it means the whole
                # transport is dead, not that the reference is absent.
                with self._cache_lock:
                    self.stats["auth_failures"] += 1
                self._note_failure("auth_failed", status=status)
            else:
                self._note_failure(f"http_{status}", status=status)
            return b"", None
        except Exception as exc:
            with self._cache_lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(exc)
            self._note_failure("transport_error")
            return b"", None
        # A well-formed answer proves the endpoint is reachable even when the
        # span is empty ("nothing is playing"), so it clears the breaker.
        self._note_success()
        pcm = payload.get("pcm_b64") or ""
        age = payload.get("age_seconds")
        if not pcm:
            with self._cache_lock:
                self._reason = "no_reference"
            return b"", None
        try:
            data = base64.b64decode(pcm)
        except Exception as exc:
            with self._cache_lock:
                self.stats["errors"] += 1
                self.stats["last_error"] = str(exc)
            self._note_failure("bad_payload")
            return b"", None
        try:
            age = (None if age is None else float(age))
        except Exception:
            age = None
        with self._cache_lock:
            self.stats["hits"] += 1
            self.stats["last_age_seconds"] = age
            self.stats["last_status"] = status
            self._reason = "ok"
            # Only a NON-EMPTY span refreshes the cache and the idle timer: an
            # empty answer means "nothing is playing", which is exactly the
            # state the idle early-out is allowed to assume without asking.
            self._cache_pcm = data
            self._cache_age = age
            self._cache_at = now
            self._cache_mic_t_end = mic_t_end
            self._last_fetch_ok_at = now
        return data, age

    def state(self):
        now = time.monotonic()
        with self._cache_lock:
            open_for = max(0.0, self._breaker_open_until - now)
            reason = ("circuit_open" if open_for > 0.0 else self._reason)
            consecutive = self.stats["consecutive_failures"]
        return {"url": self.base_url, "timeout": self.timeout,
                # [P1-04] A distinct reason instead of pretending the reference
                # is simply absent, so a dead transport is diagnosable from
                # /aec/state alone.
                "reason": reason,
                "breaker": {"open": open_for > 0.0,
                            "consecutive_failures": consecutive,
                            "threshold": self._breaker_limit,
                            "cooldown_remaining_seconds": round(open_for, 3)},
                "stats": dict(self.stats)}


class ReferencePcmBuffer:
    """Monotonic, timestamped ring of rendered playback PCM (16k mono s16).

    The audio actor (or fish_voice) feeds every chunk that actually reached
    the output device. Each chunk carries the monotonic time of its LAST
    sample, so a mic window can be aligned by TIME instead of by "whatever
    was rendered most recently" (F33).
    """

    def __init__(self, max_seconds=REFERENCE_MAX_SECONDS,
                 sample_rate=AEC_SAMPLE_RATE, sample_width=AEC_SAMPLE_WIDTH):
        self.sample_rate = int(sample_rate)
        self.sample_width = int(sample_width)
        self._max_bytes = int(max_seconds * self.sample_rate * self.sample_width)
        self._lock = threading.Lock()
        #: deque of (t_end, pcm) oldest first.
        self._chunks = deque()
        self._resampler = None
        self._resampler_rate = None
        self._channels = 1
        self._total_feeds = 0
        self._last_feed_ts = 0.0
        self._last_t_end = 0.0
        self._resample_failures = 0
        #: [B108] How far ahead of wall-clock the last chunk was stamped
        #: (the device's reported output latency). Telemetry only.
        self._last_lead_seconds = 0.0

    # ── feeding ────────────────────────────────────────────────────────
    def feed(self, pcm_bytes, sample_rate=None, sample_width=None, t_end=None,
             channels=None):
        """Store a rendered-PCM chunk; returns the stored (16k) byte count.

        Resampling is continuous ACROSS calls (``StatefulResampler``), so a
        producer that hands over arbitrary chunk sizes still yields one
        uninterrupted reference timeline. *channels* describes the incoming
        format (device output is often 48 kHz stereo); the reference is always
        stored as 16 kHz mono.
        """
        if not pcm_bytes:
            return 0
        if channels:
            self.set_channels(channels)
        resampled = self._resample_to_16k(pcm_bytes, sample_rate, sample_width)
        if not resampled:
            return 0
        now = time.monotonic() if t_end is None else float(t_end)
        self._last_lead_seconds = max(0.0, now - time.monotonic())
        with self._lock:
            self._chunks.append((now, resampled))
            self._total_feeds += 1
            self._last_feed_ts = time.monotonic()
            self._last_t_end = now
            while len(self._chunks) > 1:
                total = sum(len(c) for _, c in self._chunks)
                if total <= self._max_bytes:
                    break
                self._chunks.popleft()
        return len(resampled)

    def _resample_to_16k(self, pcm_bytes, sample_rate, sample_width):
        sr = int(sample_rate or self.sample_rate)
        sw = int(sample_width or self.sample_width)
        channels = getattr(self, "_channels", 1)
        try:
            with self._lock:
                if (self._resampler is None or self._resampler_rate != (sr, sw,
                                                                        channels)):
                    self._resampler = StatefulResampler(
                        sr, self.sample_rate, channels=channels,
                        sample_width=sw)
                    self._resampler_rate = (sr, sw, channels)
                resampler = self._resampler
            return resampler.resample(pcm_bytes)
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                self._resample_failures += 1
            print(f"[AEC] reference resample failed ({exc})")
            return b""

    def set_channels(self, channels):
        """Tell the buffer how many channels the producer's PCM carries."""
        with self._lock:
            if int(channels) != self._channels:
                self._channels = max(1, int(channels))
                self._resampler = None
                self._resampler_rate = None

    # ── alignment ──────────────────────────────────────────────────────
    def aligned_reference(self, duration_bytes, mic_t_end=None,
                          max_drift_seconds=REFERENCE_MAX_DRIFT_SECONDS):
        """The reference span that OVERLAPS the mic window, or ``b""``.

        *mic_t_end* is the monotonic time of the mic window's last sample
        (the listener timestamps each captured frame). The span returned ends
        at that point and is *duration_bytes* long, so a window captured while
        the assistant spoke is cancelled against exactly what was playing.
        Without *mic_t_end* the window is assumed to end NOW (a live capture),
        so a reference that stopped playing long ago reports "no reference"
        rather than being reused as if it were current.
        """
        duration_bytes = max(0, int(duration_bytes or 0))
        with self._lock:
            if not self._chunks:
                return b""
            newest_t_end = self._chunks[-1][0]
            chunks = list(self._chunks)
        if mic_t_end is None:
            mic_t_end = time.monotonic()
        if abs(mic_t_end - newest_t_end) > max_drift_seconds:
            # Playback did not overlap this window (or the clock is skewed).
            return b""
        # Walk backwards from the window end, keeping chunks whose span
        # overlaps [mic_t_end - duration, mic_t_end]. [S28] A chunk that
        # extends BEYOND mic_t_end is kept only up to the window end: the
        # span must not contain audio the mic window has not heard yet.
        bytes_per_second = float(self.sample_rate * self.sample_width)
        window_start = mic_t_end - (duration_bytes / bytes_per_second)
        kept = []
        newest_kept_t_end = None
        for t_end, chunk in reversed(chunks):
            span = len(chunk) / bytes_per_second
            t_start = t_end - span
            if t_end <= window_start:
                break
            if t_start >= mic_t_end:
                continue
            kept.append(chunk)
            if newest_kept_t_end is None:
                newest_kept_t_end = t_end
        joined = b"".join(reversed(kept))
        if not joined:
            return b""
        if newest_kept_t_end is not None and newest_kept_t_end > mic_t_end:
            # Floor, with a nudge for float jitter around an exact sample
            # boundary (639.999... must become 640, a genuine 639.5 stays 639).
            overflow = int((newest_kept_t_end - mic_t_end) *
                           bytes_per_second + 1e-3)
            if overflow > 0:
                joined = joined[:max(0, len(joined) - overflow)]
        if not joined:
            return b""
        if len(joined) <= duration_bytes:
            return joined
        return joined[len(joined) - duration_bytes:]

    def age_seconds(self):
        with self._lock:
            if not self._chunks:
                return None
            return max(0.0, time.monotonic() - self._chunks[-1][0])

    def stats(self):
        with self._lock:
            return {
                "chunks": len(self._chunks),
                "bytes": sum(len(c) for _, c in self._chunks),
                "feeds": self._total_feeds,
                "last_feed_ts": self._last_feed_ts,
                "last_t_end": self._last_t_end,
                "resample_failures": self._resample_failures,
                "lead_seconds": round(self._last_lead_seconds, 3),
                "age_seconds": (max(0.0, time.monotonic() - self._chunks[-1][0])
                                if self._chunks else None),
            }

    def __len__(self):
        with self._lock:
            return sum(len(c) for _, c in self._chunks)


class AecSignalPath:
    """Ties the reference buffer and the canceller together for the
    listener: rendered PCM is fed by the audio layer; the mic capture
    asks for a cancelled window before VAD/onset analysis.

    F33: every capture frame is processed ONCE (``frame_id``), the mic window
    is aligned on the shared timeline (``mic_t_end``), and the whole path
    reports an explicit state instead of silently pretending to cancel.
    """

    #: Frames kept for the once-only cache (a few seconds of VAD windows).
    FRAME_CACHE_LIMIT = 512

    #: [P0-13] Process-wide capture counter. Frame ids used to restart at 0 for
    #: every capture, so capture N+1's frame 0 collided with capture N's frame 0
    #: and the cache served the PREVIOUS utterance's PCM back as the new one -
    #: the "self-barge-in on turn 2+ / playback bleeding through" symptom. Ids are
    #: now (capture_token, index) and the token comes from here, so they are
    #: unique for the lifetime of the process.
    _capture_counter = itertools.count(1)

    def __init__(self, canceller=None, reference_buffer=None, process=None,
                 transport="auto", echo_gate=None):
        self.canceller = canceller or build_canceller()
        self.reference = reference_buffer or ReferencePcmBuffer()
        self.process = process or "local"
        if transport == "auto":
            self.transport = (RemoteAecTransport()
                              if (AEC_ENABLED and AEC_REMOTE_ENABLED) else None)
        else:
            self.transport = transport
        #: Used only when the canceller is a no-op (no AEC model installed):
        #: without it the "filtered" mic is the raw mic, and a loud playback
        #: window would be captured as user speech.
        self.echo_gate = (echo_gate if echo_gate is not None
                          else ReferenceEchoGate())
        self.stats = {
            "cancelled_analyses": 0,
            "had_reference": 0,
            "replayed_frames": 0,
            "frames_processed": 0,
            "echo_suppressed": 0,
            "captures_started": 0,
            "reference_sources": {"local": 0, "remote": 0, "none": 0},
        }
        self._frame_lock = threading.Lock()
        self._frame_cache = {}
        self._frame_order = deque()
        self._last_frame_id = None
        #: [P0-13] Token identifying the capture in flight; frame ids are
        #: ``(capture_token, index)`` so they cannot repeat across captures.
        self._capture_token = None
        #: Stateful 16k conversion for the MIC side too - the canceller is a
        #: 16 kHz model, and the mic must not be resampled with a stride.
        self._mic_resampler = None
        self._mic_rate = None
        self.mic_sample_rate = AEC_SAMPLE_RATE
        # [S28] Speaker-delay alignment: the lag between "reference recorded"
        # and "sound actually heard", estimated once per session on the
        # degraded (echo-gate) path - a real AEC3 model has its own delay
        # estimator. Plus the joint compare-window history for the gate.
        self._lag_estimator = DelayEstimator()
        self._gate_history = deque(maxlen=max(0, AEC_GATE_HISTORY_FRAMES - 1))
        self.stats["lag_estimates"] = 0
        self.stats["lag_locked_seconds"] = None
        #: [LF35] Monotonic time the last frame with overlapping playback was
        #: seen; fdaf resets its path estimate only after a long silence.
        self._last_had_reference_at = None

    # ── reference side ─────────────────────────────────────────────────
    def feed_reference(self, pcm_bytes, sample_rate=None, sample_width=None,
                       t_end=None, channels=None):
        return self.reference.feed(pcm_bytes, sample_rate, sample_width, t_end,
                                   channels=channels)

    # ── mic side ───────────────────────────────────────────────────────
    def begin_capture(self, turn_id=None):
        """Open a capture and return the token to scope its frame ids with.

        [P0-13] Two layers of defence against the frame-cache collision that
        replayed an old capture's PCM into the next one:

        1. *Identity*: ids handed out from here are ``(token, index)`` with a
           token from a process-wide counter, so capture N+1 can never ask for
           an id capture N already used. *turn_id* (the listener's own turn
           number) is folded in so ids stay readable and stay unique even if
           ``begin_turn`` ever returns a constant.
        2. *Eviction*: the previous capture's entries are dropped, which bounds
           memory over a long session and guarantees no cross-turn reuse even
           if the id scheme is ever changed again.

        Clearing alone would be wrong - it would also discard the WITHIN-capture
        de-duplication the cache exists for (the sliding VAD window re-asks for
        frames it already analysed, and those must still be served from cache
        without re-feeding the AEC).

        Returns an opaque token; pass it to :meth:`frame_id`.
        """
        # [B105] New capture, fresh joint evidence: the lag windows never
        # carry audio across a capture boundary (a proven lock and the
        # fitted H do survive - the acoustic path outlives short gaps).
        self._lag_estimator.clear_window()
        if isinstance(self.canceller, FrequencyDomainEchoCanceller):
            self.canceller._lag.clear_window()
        with self._frame_lock:
            self._frame_cache.clear()
            self._frame_order.clear()
            self._last_frame_id = None
            token = (next(self._capture_counter), turn_id)
            self._capture_token = token
            self.stats["captures_started"] += 1
        # [S28] a live capture re-arms the background reference prefetch.
        note = getattr(self.transport, "note_capture_activity", None)
        if callable(note):
            note()
        return token

    def frame_id(self, index, token=None):
        """Frame id for the *index*-th frame of the current capture."""
        if token is None:
            token = self._capture_token
        return (token, index)

    def _mic_to_aec_rate(self, mic_pcm, sample_rate=None, sample_width=None):
        """Return (16k mono s16 pcm, rate_used) for a captured frame."""
        rate = int(sample_rate or AEC_SAMPLE_RATE)
        width = int(sample_width or AEC_SAMPLE_WIDTH)
        if rate == AEC_SAMPLE_RATE and width == AEC_SAMPLE_WIDTH:
            return mic_pcm, AEC_SAMPLE_RATE
        try:
            with self._frame_lock:
                if self._mic_resampler is None or self._mic_rate != (rate, width):
                    self._mic_resampler = StatefulResampler(
                        rate, AEC_SAMPLE_RATE, channels=1, sample_width=width)
                    self._mic_rate = (rate, width)
                resampler = self._mic_resampler
            return resampler.resample(mic_pcm), AEC_SAMPLE_RATE
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[AEC] mic resample failed ({exc})")
            return mic_pcm, rate

    def cancelled_mic_window(self, mic_pcm, duration_bytes=None,
                             mic_t_end=None, frame_id=None,
                             sample_rate=None, sample_width=None,
                             duration_seconds=None):
        """Return (filtered_pcm, had_reference) for a captured mic window.

        *had_reference* False means there is no overlapping playback to
        cancel (either the reference path is unpopulated or playback did not
        overlap this window) - the caller keeps its existing VAD semantics
        and must NOT treat this as "cancelled".

        *frame_id* makes processing once-only: the SAME capture frame asked
        for twice (the listener's VAD window slides over frames it already
        analysed) returns the cached result and is not fed to the AEC again.

        *mic_t_end* is the monotonic time of the window's last sample, so the
        reference span is chosen by TIME overlap. *seconds* (or
        *duration_seconds*) is the window length measured on the shared
        timeline; *duration_bytes* remains accepted for 16k callers.

        ``cancelled_capture_frame`` is the richer variant: it also reports
        whether the window was identified as our own playback.
        """
        frame = self.cancelled_capture_frame(
            mic_pcm, duration_bytes=duration_bytes, mic_t_end=mic_t_end,
            frame_id=frame_id, sample_rate=sample_rate,
            sample_width=sample_width, duration_seconds=duration_seconds)
        return frame.pcm, frame.had_reference

    def cancelled_capture_frame(self, mic_pcm, duration_bytes=None,
                                mic_t_end=None, frame_id=None,
                                sample_rate=None, sample_width=None,
                                duration_seconds=None):
        """Return ``CaptureFrame(pcm, had_reference, suppressed, rate, gen)``.

        *suppressed* True means this window was recognised as the assistant's
        own playback (see ``ReferenceEchoGate``) and *pcm* is the residual the
        user actually contributed - which is near-silence for assistant-only
        audio. Callers must not let a suppressed frame become user speech.

        [B113] *generation* is the canceller's path generation. A replayed
        (cached) frame keeps the generation it was produced with; a fresh
        frame is stamped AFTER the cancellation call, so the frame on which
        H was dropped already carries the new number.

        [P0-13] *frame_id* must identify a frame uniquely for the WHOLE process,
        not just within one capture: a bare index restarts at 0 on every capture
        and used to make capture N+1 frame 0 return capture N's cached PCM. Use
        :meth:`begin_capture` once per capture and scope ids with its token
        (``path.frame_id(index)``). Any hashable id still works - the cache does
        not care about the type - which keeps callers and tests that pass plain
        ids working.
        """
        if frame_id is not None:
            with self._frame_lock:
                cached = self._frame_cache.get(frame_id)
            if cached is not None:
                with self._frame_lock:
                    self.stats["replayed_frames"] += 1
                return cached
        pcm, had_reference, suppressed = self._cancel_once(
            mic_pcm, duration_bytes, mic_t_end, sample_rate, sample_width,
            duration_seconds)
        result = CaptureFrame(pcm, had_reference, suppressed, AEC_SAMPLE_RATE,
                              getattr(self.canceller, "path_generation", None))
        if frame_id is not None:
            with self._frame_lock:
                self._frame_cache[frame_id] = result
                self._frame_order.append(frame_id)
                while len(self._frame_order) > self.FRAME_CACHE_LIMIT:
                    self._frame_cache.pop(self._frame_order.popleft(), None)
                self._last_frame_id = frame_id
                self.stats["frames_processed"] += 1
        return result

    def _cancel_once(self, mic_pcm, duration_bytes, mic_t_end, sample_rate=None,
                     sample_width=None, duration_seconds=None):
        self.stats["cancelled_analyses"] += 1
        mic_16k, rate = self._mic_to_aec_rate(mic_pcm, sample_rate, sample_width)
        if duration_seconds is None:
            if duration_bytes is None:
                duration_seconds = (len(mic_16k) / float(
                    AEC_SAMPLE_RATE * AEC_SAMPLE_WIDTH)) if mic_16k else 0.0
            else:
                duration_seconds = duration_bytes / float(
                    AEC_SAMPLE_RATE * AEC_SAMPLE_WIDTH)
        want = int(duration_seconds * AEC_SAMPLE_RATE * AEC_SAMPLE_WIDTH)
        # [B105] The active estimator owns the search range: fdaf measures
        # its own delay inside the canceller and absorbs it in H, so the
        # spans it needs come from ITS estimator (a larger AEC_MAX_LAG on
        # the canceller must widen the fetched span, not silently keep the
        # unrelated degraded-path default).
        full_reference = bool(getattr(self.canceller, "wants_full_reference",
                                      False))
        active_estimator = (
            self.canceller._lag
            if full_reference and isinstance(
                self.canceller, FrequencyDomainEchoCanceller)
            else self._lag_estimator
        )
        # [S28] Until the estimate locks, the degraded path fetches a span
        # long enough to search the whole lag range by cross-correlation.
        estimating = (getattr(self.canceller, "degraded", False)
                      and not self._lag_estimator.locked
                      and self._lag_estimator.max_lag_seconds > 0.0)
        # [B105] FDAF performs its own alignment; never apply its lag a
        # second time here.
        lag = (0.0 if full_reference
               else self._lag_estimator.locked_lag_seconds or 0.0)
        # The joint lag window needs one extra window-length of reference
        # on top of the search range so the oldest window the estimator
        # correlates still has its overlap inside the span.
        extra_seconds = 0.0
        if estimating or full_reference:
            extra_seconds = (active_estimator.max_lag_seconds
                             + active_estimator.window_seconds)
        lag_bytes = int(extra_seconds * AEC_SAMPLE_RATE * AEC_SAMPLE_WIDTH)
        fetch_t_end = (mic_t_end - lag) if mic_t_end is not None else None
        ref_pcm = self.reference.aligned_reference(
            want + lag_bytes, mic_t_end=fetch_t_end)
        source = "local" if ref_pcm else "none"
        local_age = self.reference.age_seconds() if ref_pcm else None
        # The API process voices replies while this process owns the mic, so
        # when nothing was rendered locally ask the renderer for the span
        # that overlaps this window (and how old it is on ITS clock).
        if self.transport is not None and (not ref_pcm or
                                           (local_age or 0.0) > 0.25):
            remote, remote_age = self.transport.fetch_reference(
                duration_seconds + extra_seconds,
                mic_t_end=fetch_t_end)
            if remote and (remote_age is None or
                           remote_age <= REFERENCE_MAX_DRIFT_SECONDS):
                fresher = (not ref_pcm or remote_age is None or
                           local_age is None or remote_age < local_age)
                if fresher:
                    ref_pcm = remote
                    source = "remote"
        had_reference = bool(ref_pcm)
        if had_reference:
            self.stats["had_reference"] += 1
            # [LF35] When the last overlapping playback was seen (fdaf path
            # resets its estimate only after a long silence).
            self._last_had_reference_at = time.monotonic()
        sources = self.stats.setdefault("reference_sources",
                                        {"local": 0, "remote": 0, "none": 0})
        sources[source] = sources.get(source, 0) + 1
        # The span the gate compares against: exactly one mic window long,
        # ending at the lag-corrected point (an estimation fetch asked for
        # window + search range, so trim the tail window out of it).
        ref_frame = ref_pcm
        if ref_pcm and len(ref_pcm) > want:
            ref_frame = ref_pcm[len(ref_pcm) - want:]
        # [S28] First votes of the session: correlate the mic against the
        # long span. Once locked, every later fetch already carries the lag.
        if estimating and ref_pcm:
            estimate = self._lag_estimator.estimate(mic_16k, ref_pcm)
            if estimate is not None:
                self.stats["lag_estimates"] += 1
            if self._lag_estimator.locked:
                self.stats["lag_locked_seconds"] = \
                    self._lag_estimator.locked_lag_seconds
                self._gate_history.clear()
        suppressed = False
        if had_reference and getattr(self.canceller, "degraded", False):
            # No real AEC: the canceller returns the mic untouched, so decide
            # explicitly whether this window IS the playback echo. If it is,
            # hand back the residual (near silence for assistant-only audio)
            # and mark the frame so no downstream VAD can commit it.
            # [S28] The fit runs on the last few frames jointly (~100-200 ms);
            # a single 32 ms window cannot tolerate residual timing jitter.
            hist_mic = b"".join(m for m, _ in self._gate_history)
            hist_ref = b"".join(r for _, r in self._gate_history)
            if not hist_mic or not hist_ref:
                hist_mic = hist_ref = b""
            suppressed, residual, _metrics = self.echo_gate.analyse(
                mic_16k, ref_frame, history_mic=hist_mic,
                history_ref=hist_ref)
            # [S28] Only a lag-corrected span may join the joint window: a
            # pre-lock frame's reference is misaligned by the very offset we
            # are still measuring, and would poison the next fit.
            if ref_frame and not estimating:
                self._gate_history.append((mic_16k, ref_frame))
            if suppressed:
                self.stats["echo_suppressed"] += 1
                return residual, True, True
            filtered = self.canceller.cancel(mic_16k, ref_frame)
            return filtered, True, False
        if had_reference and getattr(self.canceller, "reports_suppression",
                                     False):
            # [LF35] fdaf: subtract the estimated echo over the FULL
            # lag-covering span (the delay lives inside H) and take the
            # frame-level echo verdict from the canceller itself. A window
            # where subtraction removed >= FDAF_SUPPRESS_DB is our own
            # playback, never user speech.
            ref_for_cancel = (ref_pcm if full_reference else ref_frame)
            filtered = self.canceller.cancel(mic_16k, ref_for_cancel)
            suppressed = bool(getattr(self.canceller, "last_suppressed",
                                      False))
            if suppressed:
                self.stats["echo_suppressed"] += 1
            return filtered, True, suppressed
        if not had_reference:
            # A playback gap invalidates the joint window; the next reference
            # span must build a fresh history.
            self._gate_history.clear()
            # [LF35] fdaf keeps its path estimate across short gaps (the
            # acoustic path outlives them), but a LONG silence after real
            # playback may mean a device or volume change - rebuild the
            # estimate from scratch.
            last_at = self._last_had_reference_at
            if (last_at is not None and
                    time.monotonic() - last_at > FDAF_RESET_AFTER_SECONDS):
                reset = getattr(self.canceller, "reset", None)
                if callable(reset):
                    reset()
                    self._last_had_reference_at = None
        filtered = self.canceller.cancel(mic_16k, ref_pcm)
        return filtered, had_reference, suppressed

    # ── observability ──────────────────────────────────────────────────
    def state(self):
        """Explicit AEC state — never a silent no-op.

        ``mode`` is ``webrtcaec3`` when real cancellation is active,
        ``noop`` when the reference path exists but the model does not, and
        ``off`` when AEC is disabled by config.
        """
        mode = "off" if not AEC_ENABLED else getattr(self.canceller, "name", "noop")
        degraded = bool(getattr(self.canceller, "degraded", True))
        reason = None
        if not AEC_ENABLED:
            reason = "AEC disabled by configuration"
        elif degraded:
            reason = ("no AEC implementation installed - the reference path is "
                      "wired but the mic passes through unfiltered")
        age = self.reference.age_seconds()
        if age is not None and age > REFERENCE_MAX_DRIFT_SECONDS:
            reason = (reason + "; " if reason else "") + \
                "reference is stale (%.1fs old)" % age
        # [B105] Telemetry must show the estimator that actually decides
        # this path (fdaf measures its own delay inside the canceller), not
        # the unused degraded-path one - the live misses were invisible
        # because state() reported the wrong locks and votes.
        if isinstance(self.canceller, FrequencyDomainEchoCanceller):
            lag_snapshot = self.canceller._lag.state()
            lag_snapshot["source"] = "fdaf"
        else:
            lag_snapshot = self._lag_estimator.state()
            lag_snapshot["source"] = "signal-path"
        lag_snapshot["gate_history_frames"] = len(self._gate_history)
        snapshot = {
            "mode": mode,
            "degraded": degraded,
            "reason": reason,
            "process": self.process,
            "reference_age_seconds": age,
            "mic_sample_rate": self.mic_sample_rate,
            # [LF35] The canceller's own counters (fdaf adaptation/ERLE), so
            # /aec/state shows whether subtraction is actually working.
            "canceller": {
                "name": getattr(self.canceller, "name", "?"),
                "metrics": dict(getattr(self.canceller, "metrics", {})
                                or {}),
            },
            "reference": self.reference.stats(),
            "transport": (self.transport.state()
                          if self.transport is not None else None),
            "counters": dict(self.stats),
            # [B105] The active estimator's full diagnostics: correlation
            # peak, candidate lag, searched range, votes, reason.
            "lag": lag_snapshot,
        }
        return snapshot

    # ── rendering side (the process that plays TTS) ────────────────────
    def reference_span(self, duration_seconds, mic_t_end=None):
        """Return ``(pcm, age_seconds)`` for the /aec/reference endpoint."""
        want = int(max(0.0, float(duration_seconds or 0.0)) *
                   AEC_SAMPLE_RATE * AEC_SAMPLE_WIDTH)
        pcm = self.reference.aligned_reference(want, mic_t_end=mic_t_end)
        if not pcm:
            return b"", None
        return pcm, self.reference.age_seconds()


# Module-level path the listener uses; the audio actor feeds it so the
# mic side and the playback side never need to meet explicitly.
signal_path = AecSignalPath()


def feed_reference(pcm_bytes, sample_rate=None, sample_width=None, t_end=None,
                   channels=None):
    return signal_path.feed_reference(pcm_bytes, sample_rate, sample_width,
                                      t_end=t_end, channels=channels)


def begin_capture(turn_id=None):
    """[P0-13] Open a capture on the shared signal path; returns its token.

    The listener calls this once per ``_capture_audio()`` and scopes its frame
    ids with the returned token, so no id is ever reused across captures.
    """
    return signal_path.begin_capture(turn_id=turn_id)


def cancelled_mic_window(mic_pcm, duration_bytes=None, mic_t_end=None,
                         frame_id=None, sample_rate=None, sample_width=None,
                         duration_seconds=None):
    return signal_path.cancelled_mic_window(
        mic_pcm, duration_bytes, mic_t_end=mic_t_end, frame_id=frame_id,
        sample_rate=sample_rate, sample_width=sample_width,
        duration_seconds=duration_seconds)


def cancelled_capture_frame(mic_pcm, duration_bytes=None, mic_t_end=None,
                            frame_id=None, sample_rate=None, sample_width=None,
                            duration_seconds=None):
    """Richer sibling of ``cancelled_mic_window``: returns a CaptureFrame."""
    return signal_path.cancelled_capture_frame(
        mic_pcm, duration_bytes, mic_t_end=mic_t_end, frame_id=frame_id,
        sample_rate=sample_rate, sample_width=sample_width,
        duration_seconds=duration_seconds)


def aec_state():
    return signal_path.state()
