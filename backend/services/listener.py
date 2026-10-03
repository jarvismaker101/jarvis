import atexit
import http.client as http_exceptions
import os
import re
import threading
import time

import speech_recognition as sr
import webrtcvad

from backend import listener_state
from backend.services import model_registry
from backend.services.audio_input import (
    AEC_CAPTURE_RATE as _AEC_CAPTURE_RATE,
    FIXED_IDLE_ENERGY_THRESHOLD,
    calibrate_recognizer,
    create_microphone,
    list_microphone_names,
    mic_rate_state,
    open_microphone_at_preferred_rate,
    resolve_microphone,
    resolve_working_microphone_index,
)
from backend.services.earcons import play_capture_complete_earcon
from backend.services.neural_vad import make_speech_endpoint
from backend.services.transcription import (
    is_hallucinated_transcript,
    recognize_google_or_groq,
    recognize_inworld,
    recognize_local_whisper,
)
# [F33] onset detection runs on the echo-cancelled mic window; [F34]
# downstream actions may only consume committed transcripts.
from backend.services.echo_cancel import (
    AEC_SAMPLE_RATE as _AEC_RATE,
    AEC_SAMPLE_WIDTH as _AEC_WIDTH,
    StatefulResampler as _StatefulResampler,
    aec_state as _aec_state,
    begin_capture as _aec_begin_capture,
    cancelled_capture_frame as _aec_capture_frame,
)
from backend.services.transcript_stabilizer import (
    TranscriptWindow,
    stabilizer as _turn_stabilizer,
)
from backend.services import latency as _latency

LISTEN_TIMEOUT_SECONDS = 10
MAX_PHRASE_SECONDS = 15
#: [PERF] Trailing silence that ends an utterance. This is a direct
#: latency/accuracy knob: every turn pays it in full, but a value that is too
#: low cuts the user off mid-sentence (thinking pauses, trailing clauses).
#: Shipped at 0.7s per the 2026-10 owner request (was 1.2s), overridable at
#: runtime via JARVIS_PAUSE_THRESHOLD (e.g. 1.2 to restore the old value).
PAUSE_THRESHOLD_SECONDS = float(
    os.getenv("JARVIS_PAUSE_THRESHOLD", "0.7"))
NON_SPEAKING_SECONDS = 0.5
PHRASE_THRESHOLD_SECONDS = 0.12
RECALIBRATE_AFTER_EMPTY_LISTENS = 8
RECALIBRATE_COOLDOWN_SECONDS = 20
SPEECH_START_CONFIRMATION_CHUNKS = 2
SPEECH_START_MIN_SECONDS = 0.10
SPEECH_START_VAD_RATIO = float(os.getenv("JARVIS_SPEECH_START_VAD_RATIO", "0.18"))
SPEECH_START_MIN_FRAMES = 2
SPEECH_START_WINDOW_CHUNKS = 6
FINAL_SPEECH_VAD_RATIO = float(os.getenv("JARVIS_FINAL_SPEECH_VAD_RATIO", "0.08"))
RECOGNITION_LANGUAGES = tuple(
    language.strip()
    for language in os.getenv("JARVIS_STT_LANGUAGES", "en-IN,hi-IN").split(",")
    if language.strip()
)
if not RECOGNITION_LANGUAGES:
    RECOGNITION_LANGUAGES = ("en-IN", "hi-IN")

recognizer = sr.Recognizer()
recognizer.pause_threshold = PAUSE_THRESHOLD_SECONDS
recognizer.non_speaking_duration = NON_SPEAKING_SECONDS
recognizer.phrase_threshold = PHRASE_THRESHOLD_SECONDS
recognizer.dynamic_energy_threshold = False
recognizer.dynamic_energy_ratio = 1.2
recognizer.operation_timeout = 6

MIC_DEVICE_INDEX, MIC_NAME, MIC_SOURCE = resolve_microphone()

if MIC_DEVICE_INDEX is not None:
    try:
        working_index = resolve_working_microphone_index(MIC_DEVICE_INDEX)
    except OSError:
        working_index = None
    if working_index != MIC_DEVICE_INDEX:
        working_names = list_microphone_names()
        working_label = (
            f"{working_names[working_index]!r}"
            if working_index is not None and 0 <= working_index < len(working_names)
            else "system default"
        )
        print(
            f"[LISTENER] Preferred mic {MIC_NAME!r} cannot be opened - "
            f"using {working_label} until it becomes available"
        )
        MIC_DEVICE_INDEX = working_index

empty_listen_count = 0
last_recalibrated_at = 0.0
_microphone_source = None
_microphone_lock = threading.Lock()

try:
    calibrate_recognizer(recognizer, MIC_DEVICE_INDEX, duration=1.0)
except OSError as exc:
    print(f"[LISTENER] Initial mic calibration failed - using fixed threshold: {exc}")
listener_state.register_recognizer(recognizer)
print(
    f"[LISTENER] Ready - mic: {MIC_NAME} ({MIC_SOURCE}) | "
    f"threshold: {int(recognizer.energy_threshold)} fixed (target {FIXED_IDLE_ENERGY_THRESHOLD}) | "
    f"pause: {recognizer.pause_threshold:.2f}s"
)

vad = webrtcvad.Vad(1)

#: [PERF] P1-19 — which STT engine produced the last accepted transcript
#: ("inworld" | "local-whisper" | "google-or-groq" | ""). Telemetry only: it is
#: what lets the waterfall say WHICH engine cost that time (the audit's
#: "record which STT engine produced the transcript").
LAST_STT_ENGINE = ""

#: [P0-03] Why the last STT turn produced NO transcript: ``(engine, reason)``,
#: or None when the selected engine answered. Before this, a failed turn
#: returned None with nothing but a per-engine print, so "the listener heard
#: nothing" and "every engine failed" looked identical in the log.
LAST_STT_FAILURE = None


def _mark_turn(marks, name, meta=None):
    """[PERF] P1-19 — best-effort mark on this turn's voice timeline.

    *marks* is the optional ``latency.LocalTurn`` the capture thread created for
    this utterance. Telemetry must never break capture, so a missing sink or a
    failing sink is silently ignored.
    """
    if marks is None:
        return
    try:
        marks.mark(name, meta)
    except Exception:
        pass


def _close_microphone_source():
    global _microphone_source

    with _microphone_lock:
        source = _microphone_source
        _microphone_source = None

    if source is None:
        return

    try:
        source.__exit__(None, None, None)
    except Exception:
        pass


def _get_microphone_source():
    global _microphone_source

    with _microphone_lock:
        source = _microphone_source
        if source is not None and getattr(source, "stream", None) is not None:
            return source

        # [P1-05] Open AT the AEC rate so a captured frame and an
        # echo-cancelled frame share one rate and no join can mix them. A device
        # that refuses it falls back to its native rate inside the helper - the
        # capture must never be lost over a preferred rate, and
        # ``_combine_audio_chunks`` guarantees one rate either way.
        mic = open_microphone_at_preferred_rate(MIC_DEVICE_INDEX)
        if mic is None or getattr(mic, "stream", None) is None:
            raise OSError(
                f"Failed to open microphone (device index {MIC_DEVICE_INDEX}). "
                "The device may be unavailable, disabled, or already in use."
            )
        _microphone_source = mic
        return mic


def _reset_microphone_source():
    _close_microphone_source()
    return _get_microphone_source()


atexit.register(_close_microphone_source)
try:
    _get_microphone_source()
except Exception as exc:
    print(f"[LISTENER] Microphone warm-open failed: {exc}")


def _normalize_text(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _recalibrate_listener(reason, force=False):
    global empty_listen_count, last_recalibrated_at

    now = time.monotonic()
    if not force:
        if empty_listen_count < RECALIBRATE_AFTER_EMPTY_LISTENS:
            return
        if now - last_recalibrated_at < RECALIBRATE_COOLDOWN_SECONDS:
            return

    try:
        threshold = calibrate_recognizer(
            recognizer,
            MIC_DEVICE_INDEX,
            duration=0.8,
            source=_get_microphone_source(),
        )
    except Exception:
        threshold = calibrate_recognizer(
            recognizer,
            MIC_DEVICE_INDEX,
            duration=0.8,
            source=_reset_microphone_source(),
        )
    listener_state.register_recognizer(recognizer)
    empty_listen_count = 0
    last_recalibrated_at = now
    print(
        f"[LISTENER] Recalibrated ({reason}) - mic: {MIC_NAME} ({MIC_SOURCE}) | "
        f"threshold: {threshold}"
    )


def _record_empty_listen(reason):
    global empty_listen_count
    if listener_state.is_speaking():
        return
    empty_listen_count += 1
    _recalibrate_listener(reason)


def _get_speech_stats(audio):
    try:
        raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
        frame_duration = 30
        frame_size = int(16000 * frame_duration / 1000) * 2
        frames = [
            raw[index:index + frame_size]
            for index in range(0, len(raw) - frame_size + 1, frame_size)
        ]
        if not frames:
            return {"speech_frames": 0, "total_frames": 0, "speech_ratio": 0.0}

        speech_frames = sum(
            1
            for frame in frames
            if len(frame) == frame_size and vad.is_speech(frame, 16000)
        )
        return {
            "speech_frames": speech_frames,
            "total_frames": len(frames),
            "speech_ratio": speech_frames / len(frames),
        }
    except Exception as exc:
        print(f"[LISTENER] VAD error: {exc}")
        return {"speech_frames": 0, "total_frames": 0, "speech_ratio": 1.0}


def is_human_voice(audio, minimum_ratio=FINAL_SPEECH_VAD_RATIO):
    stats = _get_speech_stats(audio)
    return (
        stats["total_frames"] > 0
        and stats["speech_frames"] > 0
        and stats["speech_ratio"] >= minimum_ratio
    )


#: [P0-03] The turn's WHOLE speech-to-text budget. With exactly one engine per
#: turn the budget IS this number - explicit and env-tunable - instead of
#: falling through to ``recognize_local_whisper``'s generic 15s default. The
#: separate partial-window deadline that used to sit beside this constant went
#: with the partial layer (tag ``simple-listening``).
FINAL_STT_TIMEOUT_SECONDS = float(
    os.getenv("JARVIS_STT_FINAL_TIMEOUT", "10"))


def _cloud_stt_policy():
    """The shared explicit cloud-egress policy (F34/F36)."""
    try:
        from backend.services.wake_engine import cloud_stt_policy

        return cloud_stt_policy()
    except Exception:
        return "on"


#: [P0-03] The single-engine map. The registry offers exactly these providers
#: for the listening role (``model_registry.ROLE_PROVIDERS["listening"]``).
#: "google-or-groq" is not offered there yet, but it stays addressable so the
#: engine remains a SELECTION that can be turned on later rather than a
#: capability this change deletes.
LOCAL_STT_ENGINE = "whisper"
CLOUD_STT_ENGINES = frozenset({"inworld", "google-or-groq"})


def _engine_for_listening_role():
    """[P0-03] The ONE engine this turn will use, resolved from settings.

    Resolved PER CALL (P1-05) so a switch in the settings UI takes effect on the
    very next phrase with no restart. Any registry hiccup degrades to the
    shipped default (Inworld first).
    """
    try:
        selected = model_registry.get_model_for_role("listening").get(
            "provider", "inworld"
        )
    except Exception:
        selected = "inworld"
    if selected in (LOCAL_STT_ENGINE, "google-or-groq"):
        return selected
    # Anything unexpected (including the shipped default) is Inworld.
    return "inworld"


def _transcribe_with_engine(engine, audio):
    """Run exactly ONE STT engine. Returns ``(result, engine_label, reason)``.

    [P0-03] There is NO chaining here: the caller picked the single engine for
    this turn, and whatever it returns IS the answer. No ``a or b`` between
    engines and no Google/Groq language loop behind them — which is what bounds
    a turn by that engine's own timeout instead of the 117s serial ladder.

    ``result`` is ``(raw, normalized, language)`` or None; ``reason`` is a short
    code describing why nothing came back (None on success).

    The hallucination gate lives here (and again at the commit boundary in
    ``listen()``): it is never bypassed for any engine.
    """
    global LAST_STT_ENGINE

    if engine == LOCAL_STT_ENGINE:
        label = "local-whisper"
        try:
            transcript, language = recognize_local_whisper(
                audio, timeout=FINAL_STT_TIMEOUT_SECONDS)
        except sr.UnknownValueError:
            return None, label, "no-speech"
        except Exception as exc:
            return None, label, str(exc) or type(exc).__name__
        normalized = _normalize_text(transcript)
        if not normalized:
            return None, label, "empty transcript"
        if is_hallucinated_transcript(normalized):
            print(f"[LISTENER] Ignoring STT hallucination: {normalized}")
            return None, label, "hallucination"
        print(f"[HEARD:local-whisper] {normalized}")
        LAST_STT_ENGINE = label   # [PERF] P1-19
        return (transcript, normalized, language), label, None

    if engine == "google-or-groq":
        label = "google-or-groq"
        reason = "no-speech"
        # This engine IS the selected one, so its own language list still
        # applies — the languages are not a fallback ladder.
        for language in RECOGNITION_LANGUAGES:
            try:
                text = recognize_google_or_groq(
                    recognizer,
                    audio,
                    language,
                    log_prefix="LISTENER",
                )
            except sr.UnknownValueError:
                continue
            except sr.RequestError as exc:
                print(f"[LISTENER] Recognition request failed [{language}]: {exc}")
                reason = str(exc) or "request failed"
                continue
            except Exception as exc:
                print(f"[LISTENER] Recognition error [{language}]: {exc}")
                reason = str(exc) or type(exc).__name__
                continue
            normalized = _normalize_text(text)
            if not normalized:
                continue
            if is_hallucinated_transcript(normalized):
                print(f"[LISTENER] Ignoring STT hallucination: {normalized}")
                reason = "hallucination"
                continue
            print(f"[HEARD:{language}] {normalized}")
            LAST_STT_ENGINE = label   # [PERF] P1-19
            return (text, normalized, language), label, None
        return None, label, reason

    # Inworld, with an English hint (conversation text must stay roman/English
    # script). Also the engine for any unexpected provider value.
    label = "inworld"
    try:
        text = recognize_inworld(audio, language="en")
    except sr.UnknownValueError:
        return None, label, "no-speech"
    except Exception as exc:
        return None, label, str(exc) or type(exc).__name__
    normalized = _normalize_text(text)
    if not normalized:
        return None, label, "empty transcript"
    if is_hallucinated_transcript(normalized):
        print(f"[LISTENER] Ignoring STT hallucination: {normalized}")
        return None, label, "hallucination"
    if any("\u0900" <= ch <= "\u097F" for ch in text):
        # The English hint was ignored. There is no second engine to fall back
        # to (P0-03): an unwanted script is a failed turn, not a retry.
        print("[LISTENER] Inworld STT returned non-English script")
        return None, label, "non-english-script"
    print(f"[HEARD:inworld] {normalized}")
    LAST_STT_ENGINE = label   # [PERF] P1-19
    return (text, normalized, "auto"), label, None


def selected_stt_engine():
    """[S5] THE engine for this turn: the one selected in settings, clamped by
    the shared cloud-egress policy.

    One resolution, shared by the conversation path (:func:`recognize_multilingual`)
    and the wake path (``watcher.recognize_candidates``), so the model that
    decides "Jarvis" and the model that transcribes the command are always the
    SAME one the user picked. A local-only policy clamps a cloud selection to
    local whisper rather than silently uploading audio.
    """
    engine = _engine_for_listening_role()
    if _cloud_stt_policy() != "on":
        # F34: local-only. Inworld and Google/Groq are cloud engines; with the
        # policy off they are never called - a local failure produces no
        # transcript instead of an upload.
        print("[LISTENER] Cloud STT disabled by policy - local whisper only")
        engine = LOCAL_STT_ENGINE
    return engine


def recognize_multilingual(audio):
    """Recognize one utterance with the SINGLE engine this turn selected.

    Returns (raw, normalized, language): the RAW transcription (case,
    newlines and quoted content intact) alongside the normalized control
    text used for logging/matching. (None, None, None) when nothing was
    understood.

    [P0-03] Exactly ONE engine runs per turn, chosen from the settings
    registry: there is no cross-engine fallback ladder, so the worst case of
    a turn is that engine's own timeout instead of the serial sum that could
    reach 117s. When the one engine fails, no transcript is returned AND
    ``LAST_STT_FAILURE`` records ``(engine, reason)`` so the failure is
    visible in the log instead of looking like silence.

    [F34] The cloud policy is explicit and enforced HERE, before any audio
    can leave the machine: under a local-only policy the local whisper
    engine is the only engine tried, and if it fails the utterance simply
    returns no transcript — nothing is transmitted externally.
    """
    global LAST_STT_FAILURE
    LAST_STT_FAILURE = None
    assert_single_rate_audio(audio)

    engine = selected_stt_engine()

    result, label, reason = _transcribe_with_engine(engine, audio)
    if result:
        return result

    reason = reason or "unknown"
    LAST_STT_FAILURE = (label, reason)
    print(f"[LISTENER] STT failed: {label}: {reason}")
    return None, None, None


#: [P1-05] Counters so a mixed-rate capture is visible instead of silent.
_rate_mismatch_reported = False
_rate_mismatch_count = 0


def assert_single_rate_audio(audio):
    """[P1-05] THE single-point invariant for STT input.

    Asserted in exactly one place - the front door of
    ``recognize_multilingual`` - because that is where every engine is reached:
    ``recognize_inworld`` builds its WAV from ``audio.sample_rate`` +
    ``audio.get_wav_data()``, so a wrong label here silently corrupts the
    upload (48 kHz PCM announced as 16 kHz plays back at a third of the speed).

    Checks that the AudioData has one non-zero ``sample_rate``/``sample_width``
    and that the duration implied by the byte length at that rate is finite and
    non-negative - i.e. the label actually describes the bytes. Returns True /
    False; never raises, because a capture must not die on a diagnostic.
    """
    try:
        rate = int(getattr(audio, "sample_rate", 0) or 0)
        width = int(getattr(audio, "sample_width", 0) or 0)
        data = getattr(audio, "frame_data", b"") or b""
    except Exception:
        return False
    if rate <= 0 or width <= 0:
        print(f"[LISTENER] STT input has no usable rate (rate={rate} width={width})")
        return False
    if len(data) % width:
        print(
            f"[LISTENER] STT input is {len(data)} bytes, not a whole number of "
            f"{width}-byte samples at {rate} Hz"
        )
        return False
    return True


def rate_mismatch_stats():
    """[P1-05] Observability: how many joins have seen mixed rates."""
    return {
        "mixed_rate_joins": _rate_mismatch_count,
        "mic_rate": mic_rate_state(),
    }


def _chunk_rates(chunks):
    """Distinct ``(sample_rate, sample_width)`` pairs present in *chunks*."""
    return {
        (int(chunk.sample_rate or 0), int(chunk.sample_width or 0))
        for chunk in chunks
        if chunk and chunk.frame_data
    }


def _combine_audio_chunks(chunks):
    """Join capture chunks into ONE ``sr.AudioData`` at a single known rate.

    [P1-05] This used to ``b"".join`` whatever it was given and label the result
    with the FIRST chunk's rate. That list is not homogeneous: ``_aec_filter_chunk``
    returns an echo-cancelled frame at ``AEC_SAMPLE_RATE`` (16 kHz) when playback
    overlapped it, but the ORIGINAL native-rate frame when it did not. Joining
    16 kHz and 48 kHz bytes concatenates 1 second of audio with 1/3 second of
    audio under one label - time-warped speech, and a duration wrong by up to 3x
    that can make the transcript stabiliser treat the final transcript as stale.

    Invariant enforced here, in the ONE place every STT call goes through: the
    returned ``AudioData`` carries a single ``sample_rate`` that describes every
    byte in it. Two layers uphold it:

    1. The mic is opened at 16 kHz (route (a), ``audio_input``), so in practice
       native and AEC rates coincide and this is a no-op join.
    2. If a device refused that rate, non-16 kHz chunks are resampled up with the
       SAME ``StatefulResampler`` the AEC already uses for the mic side - reused,
       not reimplemented.

    If a chunk cannot be converted, this returns None rather than returning
    audio it cannot describe correctly; callers already treat None as "no audio"
    and fall back (e.g. the unfiltered ``chunks`` list).
    """
    global _rate_mismatch_reported, _rate_mismatch_count

    if not chunks:
        return None

    usable = [chunk for chunk in chunks if chunk and chunk.frame_data]
    if not usable:
        return None

    rates = _chunk_rates(usable)
    first = usable[0]
    target_rate = int(first.sample_rate or 0)
    target_width = int(first.sample_width or 0)
    if target_rate <= 0 or target_width <= 0:
        return None

    if len(rates) == 1:
        # Homogeneous: the original fast path, unchanged.
        frame_data = b"".join(chunk.frame_data for chunk in usable)
        return sr.AudioData(frame_data, target_rate, target_width) if frame_data else None

    # Mixed rates (P1-05). Normalise everything to the AEC rate, which is what
    # every STT engine here expects and what the AEC frames already are.
    _rate_mismatch_count += 1
    if not _rate_mismatch_reported:
        _rate_mismatch_reported = True
        print(
            f"[LISTENER] Mixed capture rates {sorted(rates)} (further ones "
            f"reported by count only) - normalising to {_AEC_RATE} Hz. The mic "
            f"could not open at {_AEC_RATE} Hz."
        )

    try:
        # One resampler PER SOURCE RATE, each streaming 48k->16k, so a stream
        # split at any boundary is the whole-stream resample (seam-free). A
        # single resampler built from the first chunk's rate would be wrong:
        # if the first chunk is already 16 kHz it is a passthrough and would
        # relabel 48 kHz bytes as 16 kHz - exactly the bug being fixed.
        resamplers = {}
        parts = []
        for chunk in usable:
            rate = int(chunk.sample_rate or 0)
            width = int(chunk.sample_width or 0)
            if rate <= 0 or width != _AEC_WIDTH:
                # Cannot describe this chunk as AEC-format PCM: refuse rather
                # than return audio whose label is a lie (P1-05 route (c)).
                return None
            if rate == _AEC_RATE:
                parts.append(bytes(chunk.frame_data))
                continue
            resampler = resamplers.get(rate)
            if resampler is None:
                resampler = resamplers[rate] = _StatefulResampler(
                    source_rate=rate,
                    target_rate=_AEC_RATE,
                    channels=1,
                    sample_width=width,
                )
            converted = resampler.resample(bytes(chunk.frame_data))
            if not converted:
                # Too short to emit an output sample yet (the resampler needs
                # its interpolation tail). Not an error - just no bytes.
                continue
            parts.append(converted)
        frame_data = b"".join(parts)
    except Exception as exc:
        # Never fail a capture over telemetry-grade repair of the audio.
        print(f"[LISTENER] Rate normalisation failed: {exc}")
        return None

    if not frame_data:
        return None
    return sr.AudioData(frame_data, _AEC_RATE, _AEC_WIDTH)


def _audio_duration_seconds(audio):
    bytes_per_second = audio.sample_rate * audio.sample_width
    if bytes_per_second <= 0:
        return 0.0
    return len(audio.frame_data) / bytes_per_second


#: [P1-03] Timeout of ONE stop request. Also the persistent connection's
#: socket timeout - it bounds a hung backend without another mechanism.
_STOP_TIMEOUT_SECONDS = 0.4


class _SpeakStopWorker:
    """[P1-03] Daemon worker that owns POST /speak/stop for the voice process.

    Why this exists: ``barge_in_on_speech_onset`` used to make two BLOCKING
    HTTP calls from inside the real-time capture loop - a ``GET /voice-state``
    probe (0.3s) and the stop POST (0.4s). A hung backend froze the capture
    loop for up to 0.7s AT SPEECH ONSET, i.e. exactly when responsiveness
    matters most.

    Shape: one daemon thread, one persistent ``http.client.HTTPConnection``
    (the handshake is not repeated on every barge-in), and ONE coalescing flag
    rather than a queue. The flag is cleared BEFORE the send, so:

    * repeated barge-ins while a send is pending collapse into one delivery
      (``/speak/stop`` is idempotent - duplicates buy nothing), and
    * a barge-in that arrives WHILE a send is in flight re-arms the loop, so
      the newest stop is delivered again afterwards and is never swallowed.

    Never raises: every failure is swallowed into a first-only log line plus a
    counter in :attr:`stats`. A 401/403 is reported LOUDLY once - an F51-class
    auth bug used to be indistinguishable from "the backend is silent".
    """

    def __init__(self):
        self._stop_needed = threading.Event()
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._conn = None
        # [stale-socket retry] the timing anchor and the error behind the last
        # attempt, so a dead keep-alive socket can be re-dialed immediately.
        self._started = time.monotonic()
        self._last_error = None
        self._auth_warned = False
        self._error_warned = False
        self.stats = {
            "requests": 0,
            "delivered": 0,
            "failed": 0,
            "auth_failures": 0,
            "last_ms": 0.0,
            "last_status": None,
            "last_ok": False,
            "last_error": None,
        }

    # -- lifecycle ---------------------------------------------------------

    def request_stop(self):
        """Flag that a stop is needed. NON-BLOCKING; True once queued.

        The capture loop ends here - no socket, no handshake, no wait.
        """
        try:
            with self._lock:
                self.stats["requests"] += 1
                if self._thread is None or not self._thread.is_alive():
                    self._thread = threading.Thread(
                        target=self._run, name="speak-stop", daemon=True)
                    self._thread.start()
            self._stop_needed.set()
            return True
        except Exception:
            return False

    def shutdown(self, timeout=1.0):
        """Stop the worker and close its connection (listener exit)."""
        try:
            self._shutdown.set()
            self._stop_needed.set()   # wake the loop so it can see the shutdown
            thread = self._thread
            if thread is not None:
                thread.join(timeout=timeout)
        except Exception:
            pass
        self._close_connection()

    # -- worker side -------------------------------------------------------

    def _run(self):
        while True:
            self._stop_needed.wait()
            if self._shutdown.is_set():
                break
            # Clear BEFORE sending: a barge-in arriving during the send sets
            # the flag again and the loop delivers the newest stop after it.
            self._stop_needed.clear()
            self._deliver()
        self._close_connection()

    def _open_connection(self):
        """Create the persistent connection (seam for tests)."""
        from backend.config import BACKEND_PORT as _PORT
        from http.client import HTTPConnection

        return HTTPConnection("127.0.0.1", int(_PORT), timeout=_STOP_TIMEOUT_SECONDS)

    def _close_connection(self):
        conn = self._conn
        self._conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _deliver(self):
        """Send ONE stop request. Never raises; records timing and result.

        A keep-alive socket the voice process has already closed fails on the
        FIRST write with a connection-abort error (WinError 10053/10054 and
        friends). That used to cost the whole barge-in - the stop was dropped
        and only the NEXT utterance retried on a fresh socket, so Jarvis kept
        talking over the user. One immediate re-dial turns that into a stop
        that lands a few milliseconds later instead.
        """
        ok, status = self._attempt()
        if not ok and status is None and self._is_stale_socket():
            self._close_connection()
            ok, status = self._attempt()
        elapsed_ms = (time.monotonic() - self._started) * 1000.0
        with self._lock:
            self.stats["last_ms"] = round(elapsed_ms, 1)
            self.stats["last_status"] = status
            self.stats["last_ok"] = ok
            self.stats["delivered" if ok else "failed"] += 1
        self._mark_delivery(ok, elapsed_ms, status)

    def _attempt(self):
        """One send on the persistent connection. Returns (ok, status)."""
        from backend.services import local_auth

        self._started = time.monotonic()
        ok, status = False, None
        try:
            conn = self._conn
            if conn is None:
                conn = self._conn = self._open_connection()
            # F51: the per-launch token goes on EVERY delivery via the shared
            # header builder - a 401 here is the invisible-barge-in bug class.
            conn.request("POST", "/speak/stop", body=b"{}",
                         headers=local_auth.auth_headers())
            resp = conn.getresponse()
            status = int(resp.status)
            # Drain the body: a connection with an unread response cannot be
            # reused, which would defeat the persistent connection.
            resp.read()
            ok = 200 <= status < 300
            if status in (401, 403):
                self._warn_auth(status)
        except Exception as exc:
            # Drop the connection so the NEXT attempt dials fresh instead of
            # reusing a socket the server has already closed.
            self._close_connection()
            self._last_error = exc
            self._warn_error(exc)
            ok = False
        return ok, status

    def _is_stale_socket(self):
        """True when the last failure was a dead keep-alive socket (not an
        HTTP-level answer). Those are the ones a fresh dial fixes."""
        exc = getattr(self, "_last_error", None)
        if exc is None:
            return False
        if isinstance(exc, (ConnectionError, OSError, http_exceptions.HTTPException)):
            return True
        return "10053" in str(exc) or "10054" in str(exc) or "10055" in str(exc)

    def _warn_auth(self, status):
        with self._lock:
            self.stats["auth_failures"] += 1
            first = not self._auth_warned
            self._auth_warned = True
        if first:
            # Loud ONCE (rate-limited): a 401 must never look like silence.
            print(
                f"[LISTENER] speak/stop rejected with HTTP {status} - the voice "
                f"process token is not accepted, so barge-in cannot stop remote "
                f"TTS (further rejections counted, not printed)"
            )

    def _warn_error(self, exc):
        with self._lock:
            self.stats["last_error"] = str(exc)[:200]
            first = not self._error_warned
            self._error_warned = True
        if first:
            print(f"[LISTENER] speak/stop failed (further ones counted only): {exc}")

    def _mark_delivery(self, ok, elapsed_ms, status):
        """[P1-19] barge-in latency: how long the stop took and its result."""
        try:
            _latency.mark_active("barge_in_stop", {
                "ok": bool(ok),
                "ms": round(elapsed_ms, 1),
                "status": status,
            })
        except Exception:
            pass


#: [P1-03] The one stop worker for this process (started lazily on first use).
_speak_stop = _SpeakStopWorker()
atexit.register(_speak_stop.shutdown)


def speak_stop_stats():
    """[P1-03] Snapshot of the async stop's outcomes (observability)."""
    with _speak_stop._lock:
        return dict(_speak_stop.stats)


def _post_backend_speak_stop():
    """QUEUE a stop of the API-process TTS (separate OS process).

    [P1-03] NON-BLOCKING: this only raises the worker's flag, so the capture
    loop can never stall on it. The actual POST /speak/stop runs on the
    ``_speak_stop`` daemon thread over a persistent connection. Returns True
    once the stop is queued.

    F51: ``/speak/stop`` is a private control endpoint and auth fails
    CLOSED, so the DELIVERY carries the per-launch token via
    ``local_auth.auth_headers`` — without it the barge-in stop silently 401s
    and Jarvis keeps talking over the user.
    """
    return _speak_stop.request_stop()


def _api_is_speaking():
    """Best-effort check: is the API process currently voicing anything?

    The voice process's own listener_state cannot see API-side speech
    (separate OS process).  We poll the API's /voice-state endpoint; a
    connection failure means the API is offline — treat as silent.
    Short timeout, never raises.

    F51: ``/voice-state`` is a private read now, so the poll authenticates
    with the launch token; an unauthenticated 401 would look exactly like
    "the API is silent" and disable cross-process barge-in.
    """
    try:
        from backend.config import BACKEND_PORT as _PORT
        from backend.services import local_auth
        from urllib.request import Request, urlopen
        req = Request(
            f"http://127.0.0.1:{_PORT}/voice-state",
            method="GET",
            headers=local_auth.auth_headers(),
        )
        with urlopen(req, timeout=0.3) as resp:
            import json as _json
            state = _json.loads(resp.read().decode("utf-8"))
        return bool(state.get("assistant_speaking", False))
    except Exception:
        return False


#: [P0-08] Barge-in observers. The listener owns *when* a barge-in happens; the
#: process that owns the in-flight turn (voice_mode's turn manager) registers
#: here to hear about it and cancel its own backend request.
_barge_in_hooks = []
_barge_in_hooks_lock = threading.Lock()


def register_barge_in_hook(fn):
    """Register ``fn()`` to run when a barge-in onset is handled.

    Idempotent, and never raises. Observers run on the CAPTURE thread, so they
    must return immediately (the turn manager fires its cancel on a daemon
    thread for exactly that reason).
    """
    if not callable(fn):
        return False
    try:
        with _barge_in_hooks_lock:
            if fn in _barge_in_hooks:
                return False
            _barge_in_hooks.append(fn)
        return True
    except Exception:
        return False


def unregister_barge_in_hook(fn):
    """Drop a previously registered observer (used by tests). Never raises."""
    try:
        with _barge_in_hooks_lock:
            if fn in _barge_in_hooks:
                _barge_in_hooks.remove(fn)
                return True
    except Exception:
        pass
    return False


def _notify_barge_in():
    """Tell every observer that a barge-in happened. Never raises, never blocks
    on a slow observer — an observer failure must not disturb capture."""
    try:
        with _barge_in_hooks_lock:
            hooks = list(_barge_in_hooks)
    except Exception:
        return
    for hook in hooks:
        try:
            hook()
        except Exception:
            pass


def barge_in_on_speech_onset():
    """Instant barge-in: user started speaking while Jarvis TTS is playing.

    Stops local TTS synchronously (generation bump + fish PCM flush via
    stop_speaking) and queues a stop of the API-process TTS.  VAD-gated
    callers decide *when* — this only decides *how*.  Never raises.

    [P1-03] The remote notification is now ASYNC (the ``_speak_stop`` daemon
    worker), and the old ``GET /voice-state`` probe is GONE: ``/speak/stop`` is
    idempotent, so asking "is the API speaking?" first bought nothing and cost
    a blocking round trip (0.3s) plus the stop POST (0.4s) on the real-time
    capture thread AT SPEECH ONSET. A stop is now queued on every onset.

    [P0-08] Onset also notifies the registered barge-in observers, so the
    process that owns the in-flight turn can cancel THAT backend request.
    Stopping the audio alone left the old generation running and made the next
    utterance wait behind it. Observers run here on the capture thread, so they
    must be non-blocking; a slow one is never awaited and never raises.

    Returns True when a stop was issued (local) or queued (remote) — which is
    every onset, since the idempotent remote stop is queued unconditionally.
    """
    try:
        if listener_state.is_speaking():
            from backend.services.voice import stop_speaking as _stop
            # [P1-02] No "ready" cue on barge-in: the user is already talking,
            # so a beep confirming "I can hear you" is noise in the middle of
            # their sentence. Every other stop path keeps the cue.
            _stop(signal_ready=False)
    except Exception:
        pass  # a local failure must never keep the remote stop from being queued
    _notify_barge_in()
    return _post_backend_speak_stop()


def _should_confirm_speech_start(chunks):
    """VAD onset test over the ALREADY echo-cancelled capture frames.

    [F33] The frames handed in here have been cancelled exactly once (by
    ``_aec_filter_chunk``) against the reference span that overlapped them in
    time; sliding this window therefore cannot reprocess a frame through the
    AEC, and the onset decision is made on the filtered signal.
    """
    if len(chunks) < SPEECH_START_CONFIRMATION_CHUNKS:
        return False

    window = chunks[-SPEECH_START_WINDOW_CHUNKS:]
    audio = _combine_audio_chunks(window)
    if audio is None:
        return False

    if _audio_duration_seconds(audio) < SPEECH_START_MIN_SECONDS:
        return False

    stats = _get_speech_stats(audio)
    if stats["total_frames"] < SPEECH_START_MIN_FRAMES:
        return False

    return stats["speech_ratio"] >= SPEECH_START_VAD_RATIO


#: [PERF] AEC failures are counted, not printed per frame (see
#: _aec_filter_chunk). The count is exposed for diagnostics.
_aec_error_reported = False
_aec_error_count = 0


def _aec_filter_chunk(chunk, frame_id, t_end):
    """Cancel ONE captured frame.

    Returns ``(filtered_chunk, had_reference, suppressed)``.

    [F33] *frame_id* makes the AEC once-only per capture frame; *t_end* is
    the monotonic time of the frame's last sample, so the reference span is
    aligned by TIME. The returned chunk is at the AEC rate (16 kHz mono)
    whenever playback overlapped the frame. *suppressed* True means the frame
    was identified as the assistant's OWN playback - the caller must not let
    it become committed user speech, whatever a VAD says about the residual.
    """
    try:
        frame = _aec_capture_frame(
            chunk.frame_data,
            duration_seconds=_audio_duration_seconds(chunk),
            mic_t_end=t_end,
            frame_id=frame_id,
            sample_rate=chunk.sample_rate,
            sample_width=chunk.sample_width,
        )
    except Exception as exc:
        # [PERF] This runs once per captured frame. A failing AEC path would
        # otherwise print ~30 lines/second to a redirected handle, inside the
        # real-time loop. Report the first failure, then count silently.
        global _aec_error_reported, _aec_error_count
        _aec_error_count += 1
        if not _aec_error_reported:
            _aec_error_reported = True
            print(f"[LISTENER] AEC error (further ones suppressed): {exc}")
        return chunk, False, False
    if not frame.had_reference or not frame.pcm:
        return chunk, False, False
    return (sr.AudioData(frame.pcm, frame.sample_rate or _AEC_RATE,
                         _AEC_WIDTH), True, bool(frame.suppressed))


def _report_aec_degraded_once(during):
    """Print the AEC state when playback overlapped a capture it could not
    actually cancel - an explicit degraded state instead of a silent no-op."""
    if not during.get("frames"):
        return
    if during.get("reported"):
        return
    during["reported"] = True
    state = _aec_state()
    if state.get("degraded"):
        print(f"[LISTENER] AEC degraded ({state.get('mode')}): "
              f"{state.get('reason')} - {during['frames']} captured frame(s) "
              f"overlapped playback")


def _capture_audio(marks=None):
    """Capture one utterance (streaming) and return its combined audio.

    *marks* is the optional [PERF] P1-19 telemetry sink for this turn: the two
    capture boundaries (``speech_end`` = the last frame arrived, ``capture_end``
    = the audio was assembled and handed on) are recorded on it. Optional so a
    caller without a turn (a test, a probe) is unaffected.

    Plain capture: frames are consumed until the pause threshold ends the
    utterance and NOTHING is transcribed here. The partial-window layer that
    used to run a background whisper pass over the trailing audio (and could end
    the capture early on two agreeing windows) is deliberately gone - one
    utterance now costs exactly ONE transcription, after the audio is complete.
    """
    global empty_listen_count

    speech_started = False
    user_marked_speaking = False

    try:
        source = _get_microphone_source()
        print("[LISTENER] Listening...")
        audio_stream = recognizer.listen(
            source,
            timeout=LISTEN_TIMEOUT_SECONDS,
            phrase_time_limit=MAX_PHRASE_SECONDS,
            stream=True,
        )

        chunks = []
        filtered_chunks = []
        onset_chunks = []
        aec_during = {"frames": 0, "reported": False}
        aec_suppressed = {"frames": 0, "other": 0}
        # [F34] One conversation turn per capture: the final window committed
        # for this utterance can never combine with another turn's.
        try:
            turn_id = _turn_stabilizer.begin_turn()
        except Exception:
            turn_id = 0
        # [P0-13] Open the AEC capture BEFORE the first frame and scope every
        # frame id with the returned token. The old code restarted ``frame_id``
        # at 0 on every capture, so capture N+1's frame 0 hit capture N's cache
        # entry and the listener analysed the PREVIOUS utterance's PCM as if it
        # were the new one - heard as Jarvis talking over itself from turn 2 on.
        # Within one capture the ids are still stable, which is what keeps the
        # sliding VAD window from re-feeding the AEC (F33).
        try:
            capture_token = _aec_begin_capture(turn_id=turn_id)
        except Exception:
            capture_token = turn_id
        frame_index = 0
        # [S29] The utterance END is decided by a neural VAD (Silero) over
        # the echo-cancelled frames - not by the fixed energy threshold,
        # which a fan/AC outruns (capture runs to the phrase limit) and a
        # soft voice falls under (trailing words cut off). The endpoint
        # fires "ended" after ~200 ms of trusted silence following voiced
        # speech; until then speech_recognition's own pause logic is the
        # unchanged backstop.
        endpoint = make_speech_endpoint()

        for chunk in audio_stream:
            if not chunk or not chunk.frame_data:
                continue

            chunks.append(chunk)
            # [F33] cancel this frame exactly once, against the reference
            # span that overlapped it in time; every later VAD/STT decision
            # (onset, final VAD, human-voice gate, STT input) uses the
            # filtered frames, so Jarvis's own playback cannot be captured
            # as user speech - including the final fallback path.
            filtered_chunk, had_reference, suppressed = _aec_filter_chunk(
                chunk, (capture_token, frame_index), time.monotonic())
            frame_index += 1
            filtered_chunks.append(filtered_chunk)
            if had_reference:
                aec_during["frames"] += 1
            if suppressed:
                aec_suppressed["frames"] += 1
            else:
                aec_suppressed["other"] += 1
                # Only frames that are NOT the assistant's own playback may
                # contribute to onset detection: a barge-in must be the user,
                # never Jarvis's own voice leaking into the mic path.
                onset_chunks.append(filtered_chunk)

            # [PERF] Onset / barge-in is the only per-frame decision left in
            # this loop: nothing transcribes here, so the moment the user is
            # recognised as having started speaking is never delayed by an
            # engine call.
            if not speech_started and _should_confirm_speech_start(onset_chunks):
                speech_started = True
                listener_state.mark_user_speaking(True)
                user_marked_speaking = True
                # Barge-in: cut ANY in-progress TTS the instant the user
                # starts speaking (VAD-gated above, not phrase-gated).
                barge_in_on_speech_onset()

            # [S29] End the capture on trusted silence: the neural VAD saw
            # voiced speech and has now seen ~200 ms of quiet. A failing or
            # unavailable endpoint just never fires - the sr pause logic
            # remains the backstop.
            if endpoint is not None:
                try:
                    if endpoint.feed(filtered_chunk.frame_data,
                                     sample_rate=filtered_chunk.sample_rate,
                                     sample_width=filtered_chunk.sample_width,
                                     ) == "ended" and speech_started:
                        _mark_turn(marks, "neural_speech_end",
                                   {"frames": frame_index})
                        break
                except Exception as exc:
                    print(f"[LISTENER] neural endpoint error (disabled this "
                          f"capture): {exc}")
                    endpoint = None

        # [PERF] P1-19 — the streaming capture ended: the user stopped talking
        # and this is the last frame of the utterance. Everything from here on
        # is Jarvis's own processing cost.
        _mark_turn(marks, "speech_end", {"frames": len(filtered_chunks)})
        _report_aec_degraded_once(aec_during)
        audio = _combine_audio_chunks(filtered_chunks) or _combine_audio_chunks(chunks)
        if audio is None:
            _record_empty_listen("empty capture")
            return None

        if aec_suppressed["frames"] and not aec_suppressed["other"]:
            # Every frame of this capture was recognised as the assistant's
            # own playback: it is not the user, no matter what a VAD says
            # about the residual (WebRTC VAD adapts and will call a quiet
            # residual "speech"). This keeps TTS from being committed as a
            # user utterance - including the final fallback path.
            print("[LISTENER] Ignoring echo-only capture (assistant playback)")
            _record_empty_listen("echo-only capture")
            return None

        if not speech_started:
            if not is_human_voice(audio):
                print("[LISTENER] Ignoring low-confidence noise capture")
                _record_empty_listen("noise capture")
                return None

            speech_started = True
            listener_state.mark_user_speaking(True)
            user_marked_speaking = True
            # Barge-in fallback path (post-capture human-voice confirm).
            barge_in_on_speech_onset()

        if user_marked_speaking:
            listener_state.mark_user_speaking(False)
            user_marked_speaking = False

        if not listener_state.is_speaking():
            # [P1-02] Kept: it acknowledges the end of the user's turn during the
            # genuinely dead period before STT. Non-blocking (the cue's own
            # daemon thread does the Beep) and the module-level one-cue-at-a-time
            # guard means it can never overlap the reply-start cue.
            play_capture_complete_earcon()
        empty_listen_count = 0
        duration = _audio_duration_seconds(audio)
        print(
            f"[LISTENER] Speech captured - {duration:.2f}s | "
            f"threshold={int(recognizer.energy_threshold)}"
        )

        if not is_human_voice(audio):
            print("[LISTENER] Voice check uncertain - sending to STT anyway")

        # [PERF] P1-19 — the capture is finalised and about to be handed to STT.
        _mark_turn(marks, "capture_end", {"seconds": round(duration, 2)})
        return audio

    except sr.WaitTimeoutError:
        _record_empty_listen("idle")
        return None
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[LISTENER] Voice error: {exc}")
        _reset_microphone_source()
        _record_empty_listen("listen error")
        return None
    finally:
        if user_marked_speaking:
            listener_state.mark_user_speaking(False)


def listen(marks=None):
    """Capture and transcribe one committed utterance (None when there is none).

    *marks* is the optional [PERF] P1-19 telemetry sink for this turn (see
    :func:`_capture_audio`); the STT boundaries are recorded on it, including
    WHICH engine produced the transcript.

    One utterance, one transcription: the capture returns the complete audio and
    exactly ONE engine is asked to transcribe it. There is no partial-window
    layer and no early commit any more, so the transcript always comes from the
    selected engine over the WHOLE utterance instead of a trailing-window
    approximation. It still goes through the hallucination belt, because it is
    about to be acted on.
    """
    global empty_listen_count

    audio = _capture_audio(marks)
    if audio is None:
        return None

    listener_state.set_thinking(True)
    try:
        print("[LISTENER] Processing speech...")
        _mark_turn(marks, "stt_start")
        raw, text, language = recognize_multilingual(audio)
        _mark_turn(marks, "stt_done", {"engine": LAST_STT_ENGINE or "none",
                                       "language": language})
        if not text:
            _record_empty_listen("no transcript")
            return None
        # Final belt at the commit boundary: whatever engine produced the
        # transcript, an STT hallucination is never committed as user speech
        # (observed live: the wake-bias prompt echoed verbatim over TTS echo
        # and answered by the brain as if the user had spoken).
        if is_hallucinated_transcript(raw or text):
            print(f"[LISTENER] Ignoring STT hallucination: {_normalize_text(text)}")
            _record_empty_listen("hallucination")
            return None
        empty_listen_count = 0
        # [F12] The RAW transcription is what leaves this function: case,
        # paths, URLs, uppercase flags, quotes, placeholders and whitespace
        # survive into the planner/tool boundary unchanged; the normalized
        # copy (above) is for matching/logging only.
        spoken = raw if raw else text
        # [F34] The returned utterance is a COMMITTED transcript: the final
        # window commits immediately, and there is no partial-window path left
        # that could return unstable text - so no downstream action can fire on
        # a transcript the stabilizer never committed.
        try:
            stable = _turn_stabilizer.push(TranscriptWindow(
                "conv-%d" % time.monotonic_ns(),
                spoken,
                final=True,
                start_ms=0,
                end_ms=int(_audio_duration_seconds(audio) * 1000),
            ))
            committed = (stable.text if stable is not None
                         else _turn_stabilizer.committed())
        except Exception:
            committed = spoken
        return committed or spoken
    finally:
        listener_state.set_thinking(False)
