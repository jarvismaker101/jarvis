import atexit
import difflib
import http.client as http_exceptions
import itertools
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
    recognize_sarvam,
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
#: [LF35] While Jarvis's own playback is active (or just ended), the onset
#: bar rises: an echo residual that slipped the canceller must clear a
#: higher VAD ratio than real silence would, while a human speaking over
#: the speakers still clears it comfortably.
SPEECH_START_PLAYBACK_VAD_RATIO = float(
    os.getenv("JARVIS_SPEECH_START_PLAYBACK_VAD_RATIO", "0.32"))
#: [LF35] Playback is considered active for this long after the last frame
#: that overlapped rendered audio (device buffer tail + room tail).
SPEECH_START_PLAYBACK_TAIL_SECONDS = float(
    os.getenv("JARVIS_SPEECH_START_PLAYBACK_TAIL", "0.5"))
#: [LF35] A capture is echo-dominated when at least this fraction of its
#: frames were recognised as the assistant's own playback AND no human
#: onset was ever confirmed.
ECHO_CAPTURE_MIN_RATIO = float(os.getenv("JARVIS_ECHO_CAPTURE_RATIO", "0.5"))
#: [LF35/B104] Cold-start onset gate: barge-in must never be confirmed from
#: frames the canceller has not yet PROVEN it can cancel. At the start of a
#: reply the fdaf is still hunting the speaker->mic delay and converging H,
#: so loud voice-shaped echo is honestly passed through un-suppressed - and
#: no VAD bar can exclude a voice. The gate holds onset until one suppressed
#: frame proves cancellation works for THIS playback (see _aec_trust_active)
#: or, fail-open, this bound expires - which also lets the reply survive long
#: enough for H to converge instead of dying in its first 200 ms.
AEC_ONSET_PROOF_TIMEOUT = float(
    os.getenv("JARVIS_AEC_ONSET_PROOF_TIMEOUT", "1.5"))
#: [LF35/B104] How long one suppressed frame keeps the canceller trusted:
#: long enough to bridge normal inter-frame gaps, short enough that a stale
#: path estimate (or an internal reset) re-arms the gate quickly.
AEC_SUPPRESSION_TRUST_SECONDS = float(
    os.getenv("JARVIS_AEC_TRUST_SECONDS", "1.5"))
#: [B109/B117] Audio seconds a provisional (unproven) onset may run before
#: the qualify-overrun is logged: the HOLD bound. It no longer cuts the
#: utterance; the capture continues to the endpoint (or the hard bound below)
#: and the transcript decides.
AEC_ONSET_QUALIFY_SECONDS = float(
    os.getenv("JARVIS_AEC_ONSET_QUALIFY_SECONDS", "2.0"))
#: [B117] Hard bound on a provisional capture. The capture is still bounded
#: so a stuck stream cannot hold the listener forever, but the user's whole
#: utterance survives up to this point.
AEC_PROVISIONAL_MAX_SECONDS = float(
    os.getenv("JARVIS_AEC_PROVISIONAL_MAX_SECONDS", str(MAX_PHRASE_SECONDS)))
#: [B116] Read the mic while the ONE STT call runs (opt-in per listen()).
#: The worker never transcribes: it only prefetches the NEXT utterance's
#: audio, so the one-utterance-one-STT invariant is unchanged.
STT_OVERLAP_CAPTURE = os.getenv("JARVIS_STT_OVERLAP_CAPTURE", "1") != "0"
STT_OVERLAP_POLL_SECONDS = max(
    0.25, float(os.getenv("JARVIS_STT_OVERLAP_POLL", "1.0")))
_STT_OVERLAP_JOIN_SECONDS = (LISTEN_TIMEOUT_SECONDS + MAX_PHRASE_SECONDS
                             + 5.0)
#: [LF35] Transcript echo guard (the guarantee layer): a transcript that
#: matches what Jarvis just said aloud is never committed as user speech.
ECHO_GUARD_ENABLED = os.getenv("JARVIS_ECHO_GUARD", "1") != "0"
ECHO_TRANSCRIPT_MIN_WORDS = int(
    os.getenv("JARVIS_ECHO_TRANSCRIPT_MIN_WORDS", "4"))
ECHO_TRANSCRIPT_RATIO = float(
    os.getenv("JARVIS_ECHO_TRANSCRIPT_RATIO", "0.72"))
ECHO_TRANSCRIPT_MAX_AGE = float(
    os.getenv("JARVIS_ECHO_TRANSCRIPT_WINDOW", "8.0"))
#: [B111] Relaxed, re-transcription-tolerant matching for UNCERTAIN captures
#: only (playback overlapped, speech began while Jarvis was speaking, the
#: canceller proved nothing). Everything else keeps the B106 rules.
ECHO_COVERAGE_RATIO = float(os.getenv("JARVIS_ECHO_COVERAGE_RATIO", "0.6"))
ECHO_COVERAGE_MIN_RUN = int(os.getenv("JARVIS_ECHO_COVERAGE_MIN_RUN", "3"))
ECHO_COVERAGE_MIN_CONTENT = 3
ECHO_ONE_WORD_MAX_AGE = float(os.getenv("JARVIS_ECHO_ONE_WORD_MAX_AGE", "3.0"))
#: [B111] 8 was too small: a transcript with three numeric tokens can need
#: a combination beyond the eighth while the right reading is obvious to a
#: human. 24 keeps the match cheap on the STT thread.
_ECHO_VARIANT_CAP = 24

#: [B111] Single-token commands the guard never eats. A superset of every
#: single-token F35 phrase (pinned against CONTROL_GRAMMAR by a test).
_ECHO_EXEMPT_WORDS = frozenset((
    "stop", "pause", "wait", "resume", "continue", "cancel", "abort",
    "yes", "no", "yeah", "yep", "nope", "confirm", "quiet", "silence",
    "bas", "choop", "ruko", "ruk", "haan", "nahi", "sleep", "standby",
    "shutdown", "close", "exit", "quit",
))
_ECHO_FILLER = frozenset(("a", "an", "the", "and", "uh", "um", "umm",
                          "er", "ah", "hmm", "mm"))
_ECHO_HOMOPHONES = {"to": "two", "too": "two", "for": "four", "won": "one",
                    "ate": "eight", "zero": "oh", "o": "oh", "its": "it's"}
_NUM_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine", "ten", "eleven", "twelve", "thirteen",
             "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
             "nineteen")
_NUM_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty",
             "seventy", "eighty", "ninety")
_TENS_UNITS_RE = re.compile(
    r"(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(one|two|three|four|five|six|seven|eight|nine)")
_DIGIT_GROUP_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")

#: [B111] The F35 classifier, registered by voice_mode (which imports us -
#: the listener cannot import it back).
_control_classifier = None


def register_control_classifier(fn):
    """[B111] Install the F35 grammar in the listener's echo guard."""
    global _control_classifier
    if not callable(fn):
        return False
    _control_classifier = fn
    return True
#: [B103] Echo-defense telemetry: what the mic-side defences are actually
#: doing on this machine, published on the voice-state channel. One warning
#: line per interval when a playback-overlapped capture commits without a
#: single suppressed frame - the suspicious case live failures hide in.
AEC_COMMIT_WARN_INTERVAL = float(
    os.getenv("JARVIS_AEC_COMMIT_WARN_INTERVAL", "10.0"))
FINAL_SPEECH_VAD_RATIO = float(os.getenv("JARVIS_FINAL_SPEECH_VAD_RATIO", "0.08"))
RECOGNITION_LANGUAGES = tuple(
    language.strip()
    for language in os.getenv("JARVIS_STT_LANGUAGES", "en-IN").split(",")
    if language.strip()
)
if not RECOGNITION_LANGUAGES:
    RECOGNITION_LANGUAGES = ("en-IN",)

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
CLOUD_STT_ENGINES = frozenset({"inworld", "sarvam", "google-or-groq"})


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
    if selected in (LOCAL_STT_ENGINE, "sarvam", "google-or-groq"):
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

    if engine == "sarvam":
        # Sarvam (Saaras v4), with the same English hint as Inworld
        # (conversation text must stay roman/English script).
        label = "sarvam"
        try:
            text = recognize_sarvam(audio, language="en")
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
        if any("" <= ch <= "" for ch in text):
            # The English hint was ignored. There is no second engine to
            # fall back to (P0-03): an unwanted script is a failed turn.
            print("[LISTENER] Sarvam STT returned non-English script")
            return None, label, "non-english-script"
        print(f"[HEARD:sarvam] {normalized}")
        LAST_STT_ENGINE = label   # [PERF] P1-19
        return (text, normalized, "auto"), label, None

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
    """[S5] THE engine for this CONVERSATION turn: the one selected in
    settings, clamped by the shared cloud-egress policy.

    [LF36] The wake path (``watcher.recognize_candidates``) is local-only by
    design - it runs the sleeping loop continuously, so uploading every
    ambient phrase must never be possible. THIS selection governs the
    conversation path (:func:`recognize_multilingual`), which runs per turn
    after the user asked to be heard. A local-only policy clamps a cloud
    selection to local whisper rather than silently uploading audio.
    """
    engine = _engine_for_listening_role()
    if _cloud_stt_policy() != "on":
        # F34: local-only. Inworld, Sarvam and Google/Groq are cloud engines;
        # with the policy off they are never called - a local failure produces
        # no transcript instead of an upload.
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
        if not ok:
            # [LIVE FIX 26 / B43] warn only on FINAL failure — a first-attempt
            # stale-socket transient that the immediate re-dial fixes is a
            # recovery, not a failure, and must not look like one in the log.
            try:
                _exc = getattr(self, "_last_error", None)
            except Exception:
                _exc = None
            if _exc is not None:
                self._warn_error(_exc)
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
            # reusing a socket the server has already closed. The warning is
            # emitted by _deliver only if the re-dial also fails (B43).
            self._close_connection()
            self._last_error = exc
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


#: [LF35/B104] Cold-start onset gate state. Written only by the capture
#: thread; the state publisher reads it passively (never mutates).
_aec_last_reference_at = None
_aec_last_suppressed_at = None
_aec_gate_blocked_since = None
#: [B109] When the CURRENT playback epoch began. Proof from a previous
#: reply must never authorize a destructive stop for this one.
_aec_epoch_started_at = None
#: [B113] Path generation last seen on a frame, the generation whose
#: suppressed frame granted the current trust, and the generation of the
#: frame _aec_filter_chunk just produced (capture thread only). A change
#: means H was dropped: proof from the old estimate is void.
_aec_path_generation = None
_aec_trust_generation = None
_aec_frame_generation = None
_aec_trust_invalidations = 0
#: [B109] Onset provenance counters (telemetry / live diagnosis).
_onset_stats = {"trusted": 0, "provisional": 0, "voided": 0,
                "qualify_cuts": 0, "qualify_overruns": 0,
                "held_fallback": 0,
                #: [B119] provisional onsets whose ONLY playback evidence was
                #: a speaking flag (no reference frame: ElevenLabs/SAPI, or a
                #: late/stale reference). Telemetry only.
                "provisional_no_reference": 0,
                "last": "none"}


def _note_onset(kind):
    # [B117] "qualify_cut" historically never incremented "qualify_cuts"
    # because the counter is plural; resolve the key instead of trusting the
    # caller's spelling.
    key = kind if kind in _onset_stats else kind + "s"
    with _echo_guard_lock:
        if key in _onset_stats:
            _onset_stats[key] += 1
        _onset_stats["last"] = kind


def _aec_trust_active(now):
    """[LF35/B104] Has a suppressed frame recently PROVEN the canceller can
    cancel the playback that is currently playing?"""
    if _aec_last_suppressed_at is None:
        return False
    return (now - _aec_last_suppressed_at) <= AEC_SUPPRESSION_TRUST_SECONDS


def _aec_epoch_trusted(now):
    """[B109] Trust proven by a suppressed frame of THIS playback epoch."""
    if not _aec_trust_active(now) or _aec_epoch_started_at is None:
        return False
    return _aec_last_suppressed_at >= _aec_epoch_started_at


def _onset_authorizes_stop(now, playback_active):
    """[B109] May an onset perform the destructive stop right now?

    Nothing playing means no echo risk. With playback active, only proof
    from the CURRENT reply may cut it: an unproven onset must be qualified
    by the transcript instead (see listener_thread).

    [B119] *playback_active* is the caller's AUDIBLE evidence - reference
    recency OR the local speaking flag OR the cached remote flag - never the
    reference timeline alone: engines that do not feed the reference
    (ElevenLabs, SAPI) and a cold/late reference are still audible.
    """
    if not playback_active:
        return True
    return _aec_epoch_trusted(now)


def _note_aec_frame(now, had_reference, suppressed, generation=None):
    """[LF35/B104] Record one capture frame for the cold-start onset gate.

    Returns True exactly on the current epoch's untrusted->trusted
    transition, so the caller can drop the onset frames that accumulated
    while the canceller was still cold: they are probable echo, not
    double-talk.

    [B109] The transition is compared against THIS epoch's proof, not the
    rolling trust window: a suppressed frame from the previous reply may
    still be inside that window, and would otherwise swallow the new
    epoch's first genuine proof (so a held onset would never be voided).

    [B113] *generation* is the canceller's path generation for this frame.
    A change means H was dropped (reset/relearn/relock/reject): every earlier
    suppressed frame proved an estimate that no longer exists, so trust is
    cleared and only a NEW suppressed frame can re-grant it. The fail-open
    budget of ``_cold_aec_blocks_onset`` restarts, so a real user is held at
    most ``AEC_ONSET_PROOF_TIMEOUT`` and then becomes provisional (never
    blocked). ``generation=None`` keeps the legacy behaviour for cancellers
    and tests that do not track one.
    """
    global _aec_last_reference_at, _aec_last_suppressed_at
    global _aec_gate_blocked_since, _aec_epoch_started_at
    global _aec_path_generation, _aec_trust_generation
    global _aec_trust_invalidations

    if generation is not None:
        # A cached replay of a pre-drop frame must never roll the path
        # generation backwards or re-grant trust from dead evidence.
        if (_aec_path_generation is not None
                and generation < _aec_path_generation):
            return False
        if (_aec_path_generation is not None
                and generation > _aec_path_generation):
            if _aec_last_suppressed_at is not None:
                _aec_trust_invalidations += 1
            _aec_last_suppressed_at = None
            _aec_trust_generation = None
        _aec_path_generation = generation

    was_trusted = _aec_epoch_trusted(now)          # AFTER invalidation
    if had_reference:
        previous = _aec_last_reference_at
        _aec_last_reference_at = now
        if (previous is None
                or (now - previous) >= SPEECH_START_PLAYBACK_TAIL_SECONDS):
            # A fresh playback epoch owns a fresh fail-open budget and its
            # own proof requirement.
            _aec_gate_blocked_since = None
            _aec_epoch_started_at = now
    if suppressed:
        _aec_last_suppressed_at = now
        _aec_trust_generation = generation
    if not was_trusted and _aec_epoch_trusted(now):
        _aec_gate_blocked_since = None
        return True
    return False


def _cold_aec_blocks_onset(now):
    """[LF35/B104] Should barge-in onset wait for the canceller to prove?

    True only while playback is active, no recent suppressed frame proves
    the canceller can cancel it, and the fail-open bound since this unproven
    stretch began has not expired. This HOLDS onset confirmation (it never
    stops the capture): a real double-talk utterance still confirms on the
    next suppression-proof - or the bound - and the reply's own echo is left
    to the transcript guarantee layer.
    """
    global _aec_gate_blocked_since

    if _aec_last_reference_at is None:
        return False
    if (now - _aec_last_reference_at) >= SPEECH_START_PLAYBACK_TAIL_SECONDS:
        _aec_gate_blocked_since = None
        return False
    if _aec_trust_active(now):
        return False
    if _aec_gate_blocked_since is None:
        _aec_gate_blocked_since = now
    return (now - _aec_gate_blocked_since) < AEC_ONSET_PROOF_TIMEOUT


def _should_confirm_speech_start(chunks, playback_active=False):
    """VAD onset test over the ALREADY echo-cancelled capture frames.

    [F33] The frames handed in here have been cancelled exactly once (by
    ``_aec_filter_chunk``) against the reference span that overlapped them in
    time; sliding this window therefore cannot reprocess a frame through the
    AEC, and the onset decision is made on the filtered signal.

    [LF35] *playback_active* raises the VAD ratio bar: with speakers (no
    earphones) the mic hears the reply, and an imperfectly cancelled echo
    must not be able to stop Jarvis's own speech. A human speaking over
    the speakers clears the raised bar easily.
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

    ratio = SPEECH_START_VAD_RATIO
    if playback_active:
        ratio = max(ratio, SPEECH_START_PLAYBACK_VAD_RATIO)
    return stats["speech_ratio"] >= ratio


def _echo_capture_verdict(suppressed_frames, other_frames, speech_started,
                          had_reference_frames, onset_trusted=None,
                          audible=False):
    """[LF35] Should this capture be dropped as our own playback?

    ``drop`` when every frame was playback (the pre-LF35 rule), or when the
    capture is dominated by playback frames AND no human onset was ever
    confirmed: a canceller miss on a minority of frames must not turn the
    reply's tail into a user utterance. A CONFIRMED onset keeps the
    capture - that is the double-talk case (the user spoke over playback)
    and it must survive.

    [B107] A reference-overlapped capture with ZERO suppressed frames and
    no confirmed onset is *uncertain*, not proven echo: when the canceller
    has never proven it can cancel this playback, echo and a genuine user
    are indistinguishable at the signal level. Such a capture returns
    "check-transcript" - it must run through recognition and the
    transcript echo guard before any commit, and it must never authorize
    a destructive stop from VAD alone.

    [B110] *onset_trusted* carries the onset's provenance: True = confirmed
    while the canceller was proven for this playback (or nothing played);
    False = provisional (unproven canceller). A provisional onset never
    makes a playback-overlapped capture "keep": the transcript guard
    decides. None = legacy caller (pre-B110 behaviour).

    [B119] *audible* (default False: every pre-wave caller is unchanged)
    says Jarvis was audible at the start of the capture by a speaking flag
    even though no frame had a reference. Such a capture can never prove
    anything at the signal level, so it is "check-transcript" - never
    "drop" (fail open; the transcript guard is the narrow fail-closed layer).
    """
    total = suppressed_frames + other_frames
    if total <= 0:
        return "keep"
    if suppressed_frames and not other_frames:
        return "drop"
    if (speech_started and onset_trusted is False
            and had_reference_frames > 0):
        # The user's words may sit in the minority of frames; a signal-level
        # ratio cannot tell them apart from the echo. The transcript can.
        return "check-transcript"
    if (not speech_started and had_reference_frames > 0 and
            suppressed_frames / float(total) >= ECHO_CAPTURE_MIN_RATIO):
        return "drop"
    if (not speech_started and had_reference_frames > 0 and
            suppressed_frames == 0):
        return "check-transcript"
    if (audible and had_reference_frames <= 0 and suppressed_frames == 0
            and onset_trusted is not True):
        # [B119] Audible without a reference frame: the canceller proved
        # nothing, so this is uncertain, not proven echo.
        return "check-transcript"
    return "keep"


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

    [B113] The canceller's path generation is published to the capture loop
    through ``_aec_frame_generation``; the 3-tuple return stays unchanged so
    every existing caller/patch is unaffected.
    """
    global _aec_error_reported, _aec_error_count, _aec_frame_generation

    _aec_frame_generation = None
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
        _aec_error_count += 1
        if not _aec_error_reported:
            _aec_error_reported = True
            print(f"[LISTENER] AEC error (further ones suppressed): {exc}")
        return chunk, False, False
    _aec_frame_generation = getattr(frame, "generation", None)
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


def _capture_audio(marks=None, wait_timeout=None, begin_turn=True):
    """Capture one utterance (streaming) and return its combined audio.

    *marks* is the optional [PERF] P1-19 telemetry sink for this turn: the two
    capture boundaries (``speech_end`` = the last frame arrived, ``capture_end``
    = the audio was assembled and handed on) are recorded on it. Optional so a
    caller without a turn (a test, a probe) is unaffected.

    [B116] *wait_timeout* is the bounded wait for speech to START; the overlap
    prefetch worker uses a short poll and loops, while the normal capture keeps
    the LISTEN_TIMEOUT_SECONDS default. *begin_turn* is False for a prefetched
    capture: its stabilizer turn is opened later, by ``listen``, AFTER the
    previous transcript has been pushed (F34 turn order).

    Plain capture: frames are consumed until the pause threshold ends the
    utterance and NOTHING is transcribed here. The partial-window layer that
    used to run a background whisper pass over the trailing audio (and could end
    the capture early on two agreeing windows) is deliberately gone - one
    utterance now costs exactly ONE transcription, after the audio is complete.
    """
    global empty_listen_count

    speech_started = False
    user_marked_speaking = False
    #: [B109] A confirmed VAD onset that has NOT been authorized to cut the
    #: reply yet. The capture keeps running until the transcript qualifies
    #: it (or until the bounded qualify cut).
    provisional_onset_at = None
    provisional_audio_seconds = 0.0
    onset_trust = "none"
    onset_speaking = None
    #: [B115] Why Jarvis counted as audible at the start of this capture, and
    #: the evidence source for the recorded speaking_at_onset. Filled from the
    #: first frame so a capture that outlives the reply never records False
    #: just because the local flag was sampled after playback ended.
    playback_at_start = None
    speaking_source = "none"
    #: [B117] The 2 s qualify bound is a hold bound now: once crossed, keep
    #: capturing to the endpoint (bounded by AEC_PROVISIONAL_MAX_SECONDS) and
    #: let the transcript decide.
    qualify_overrun = False

    try:
        source = _get_microphone_source()
        print("[LISTENER] Listening...")
        audio_stream = recognizer.listen(
            source,
            timeout=(LISTEN_TIMEOUT_SECONDS if wait_timeout is None
                     else wait_timeout),
            phrase_time_limit=MAX_PHRASE_SECONDS,
            stream=True,
        )

        chunks = []
        filtered_chunks = []
        onset_chunks = []
        aec_during = {"frames": 0, "reported": False}
        aec_suppressed = {"frames": 0, "other": 0}
        # [B103] Fresh telemetry for this capture: the commit-time warning
        # must never quote a previous capture's numbers. [B107] The
        # verdict/onset fields are part of the same freshness contract.
        with _echo_guard_lock:
            _last_capture_echo_stats.update({
                "had_reference_frames": 0,
                "suppressed_frames": 0,
                "other_frames": 0,
                "onset_confirmed": False,
                "verdict": "not-captured",
                "onset_trust": "none",
                "speaking_at_onset": False,
                "speaking_source": "none",
                "qualify_overrun": False,
            })
        # [LF35] When the last frame that overlapped rendered playback was
        # seen; onset holds itself to a higher VAD bar for this long after.
        last_reference_at = 0.0
        # [F34] One conversation turn per capture: the final window committed
        # for this utterance can never combine with another turn's. [B116]
        # The overlap prefetch worker passes begin_turn=False: its stabilizer
        # turn is opened by listen() when the audio is consumed, after the
        # previous transcript has been pushed.
        turn_id = None
        if begin_turn:
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
                last_reference_at = time.monotonic()
            if suppressed:
                aec_suppressed["frames"] += 1
            else:
                aec_suppressed["other"] += 1
                # Only frames that are NOT the assistant's own playback may
                # contribute to onset detection: a barge-in must be the user,
                # never Jarvis's own voice leaking into the mic path.
                onset_chunks.append(filtered_chunk)

            # [LF35/B104] Cold-start gate, part 1: track whether the
            # canceller has PROVEN it can cancel this playback yet, and drop
            # the onset frames that accumulated before the proof - the fdaf
            # honestly passes echo through while it hunts the delay and
            # converges H, and those frames are probable echo.
            if _note_aec_frame(time.monotonic(), had_reference, suppressed,
                               generation=_aec_frame_generation):
                onset_chunks.clear()
                if provisional_onset_at is not None:
                    # [B109] Proof just arrived: the held onset was confirmed
                    # on pre-proof frames (probable echo) - void it.
                    provisional_onset_at = None
                    provisional_audio_seconds = 0.0
                    onset_trust = "none"
                    _note_onset("voided")

            # [PERF] Onset / barge-in is the only per-frame decision left in
            # this loop: nothing transcribes here, so the moment the user is
            # recognised as having started speaking is never delayed by an
            # engine call.
            now = time.monotonic()
            playback_active = (now - last_reference_at
                               < SPEECH_START_PLAYBACK_TAIL_SECONDS)
            # [B119] "Jarvis is audible" is a first-class boolean: a recent
            # reference frame OR the local speaking flag OR the cached
            # remote flag. ElevenLabs/SAPI never feed the reference and a
            # cold, late or stale reference reads as silence, so the
            # reference alone must not decide the raised VAD bar, the
            # destructive stop, or the qualify bound. Pure, no I/O.
            audible_source = _playback_evidence(now, playback_active)
            audible = bool(audible_source)
            if playback_at_start is None:
                playback_at_start = audible_source
            if (not speech_started
                    and provisional_onset_at is None
                    and not _cold_aec_blocks_onset(now)
                    and _should_confirm_speech_start(
                        onset_chunks, playback_active=audible)):
                onset_speaking = audible
                speaking_source = audible_source or "none"
                if _onset_authorizes_stop(now, audible):
                    speech_started = True
                    onset_trust = "trusted"
                    _note_onset("trusted")
                    listener_state.mark_user_speaking(True)
                    user_marked_speaking = True
                    # Barge-in: cut ANY in-progress TTS the instant the user
                    # starts speaking (VAD-gated above, not phrase-gated).
                    barge_in_on_speech_onset()
                else:
                    # [B109] Unproven canceller: hold the destructive stop.
                    # The transcript qualifies (or drops) this interruption.
                    provisional_onset_at = now
                    onset_trust = "provisional"
                    _note_onset("provisional")
                    if audible_source != "reference":
                        with _echo_guard_lock:
                            _onset_stats["provisional_no_reference"] += 1
                    print("[LISTENER] Onset held: canceller unproven for "
                          "this playback - the transcript decides")
            elif provisional_onset_at is not None:
                provisional_audio_seconds += _audio_duration_seconds(chunk)
                if (audible and not qualify_overrun
                        and provisional_audio_seconds
                        >= AEC_ONSET_QUALIFY_SECONDS):
                    # [B117] The 2 s bound is a HOLD bound: log it once and
                    # keep capturing. The transcript remains the only
                    # authority for a destructive stop while unproven.
                    qualify_overrun = True
                    _note_onset("qualify_overruns")
                    _mark_turn(marks, "onset_qualify_bound",
                               {"frames": frame_index})
                    print("[LISTENER] Provisional onset past the qualify "
                          "bound - capturing to the endpoint; the "
                          "transcript decides")
                if provisional_audio_seconds >= AEC_PROVISIONAL_MAX_SECONDS:
                    _note_onset("qualify_cut")
                    _mark_turn(marks, "onset_qualify_cut",
                               {"frames": frame_index})
                    break

            # [S29] End the capture on trusted silence: the neural VAD saw
            # voiced speech and has now seen ~200 ms of quiet. A failing or
            # unavailable endpoint just never fires - the sr pause logic
            # remains the backstop.
            if endpoint is not None:
                try:
                    if (endpoint.feed(filtered_chunk.frame_data,
                                      sample_rate=filtered_chunk.sample_rate,
                                      sample_width=filtered_chunk.sample_width,
                                      ) == "ended"
                            and (speech_started
                                 or provisional_onset_at is not None)):
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

        # [LF35] Echo-DOMINANT captures: the canceller flagged most frames
        # as our own playback and no human onset was ever confirmed. The
        # old rule only dropped captures where EVERY frame was flagged, so
        # a canceller miss on a minority of frames turned the reply's tail
        # into a user utterance. A confirmed onset keeps the capture -
        # that is the user speaking over playback (double-talk).
        # [B107] An all-suppressed capture (every frame recognised as the
        # assistant's own playback) is folded into the same verdict below,
        # so its telemetry is recorded before the drop like everyone
        # else's: it is not the user no matter what a VAD says about the
        # residual (WebRTC VAD adapts and will call a quiet residual
        # "speech") - including the final fallback path.
        # [B110] The verdict must see WHERE the onset came from. For a
        # provisional onset, `speech_started` is False but the capture is
        # still a possible interruption: pass "an onset happened" as the
        # third argument and its provenance separately.
        onset_any = speech_started or provisional_onset_at is not None
        if onset_speaking is None:
            # [B115] Use the evidence captured at the START of the utterance:
            # sampling listener_state.is_speaking() here would record False
            # for an echo capture that outlived the reply, and it can never
            # see an API-process reply anyway.
            onset_speaking = bool(playback_at_start)
            speaking_source = playback_at_start or "none"
        verdict = _echo_capture_verdict(
            aec_suppressed["frames"], aec_suppressed["other"], onset_any,
            aec_during["frames"],
            onset_trusted=(speech_started if onset_any else None),
            audible=bool(onset_speaking))
        # [B107] Record what the defence actually saw IMMEDIATELY, before
        # either a drop or the post-capture fallback: telemetry for
        # captures dropped before STT must never stay stale - that is what
        # made the live misses invisible.
        with _echo_guard_lock:
            _last_capture_echo_stats.update({
                "had_reference_frames": int(aec_during["frames"]),
                "suppressed_frames": int(aec_suppressed["frames"]),
                "other_frames": int(aec_suppressed["other"]),
                "onset_confirmed": bool(onset_any),
                "verdict": verdict,
                "onset_trust": onset_trust,
                "speaking_at_onset": bool(onset_speaking),
                "speaking_source": speaking_source,
                "qualify_overrun": bool(qualify_overrun),
            })
        if verdict == "drop":
            if aec_suppressed["frames"] and not aec_suppressed["other"]:
                print("[LISTENER] Ignoring echo-only capture "
                      "(assistant playback)")
                _record_empty_listen("echo-only capture")
            else:
                print("[LISTENER] Ignoring echo-dominant capture "
                      "(assistant playback)")
                _record_empty_listen("echo-dominant capture")
            return None

        if not speech_started and provisional_onset_at is None:
            # [LF35] The fallback human-voice check runs on the frames that
            # were NOT our own playback (the same frames onset uses): echo
            # in a canceller-missed frame must not carry the capture
            # through on the combined audio. A provisional onset already
            # has VAD evidence, so it skips this fallback entirely.
            probe = _combine_audio_chunks(onset_chunks) or audio
            if not is_human_voice(probe):
                print("[LISTENER] Ignoring low-confidence noise capture")
                _record_empty_listen("noise capture")
                return None

            speech_started = True
            listener_state.mark_user_speaking(True)
            user_marked_speaking = True
            # Barge-in fallback path (post-capture human-voice confirm).
            # [LF35/B104] Held while the canceller has never proven it can
            # cancel the active playback (echo risk): the reply's own words
            # are still caught by the transcript guarantee layer and are
            # never committed as user speech. [B109] The hold is now the
            # proof rule, not only the cold-start budget.
            now = time.monotonic()
            # [B119] Same audible evidence as the onset site.
            fallback_audible = bool(_playback_evidence(
                now, now - last_reference_at
                < SPEECH_START_PLAYBACK_TAIL_SECONDS))
            if (_cold_aec_blocks_onset(now)
                    or not _onset_authorizes_stop(now, fallback_audible)):
                _note_onset("held_fallback")
                print("[LISTENER] Barge-in held: canceller has not proven "
                      "cancellation for the active playback - the "
                      "transcript decides")
            else:
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

        # [B103] Record what the echo defense actually did on this capture
        # (read at the commit boundary if playback overlapped it).
        with _echo_guard_lock:
            _last_capture_echo_stats["had_reference_frames"] = int(
                aec_during["frames"])
            _last_capture_echo_stats["suppressed_frames"] = int(
                aec_suppressed["frames"])
            _last_capture_echo_stats["other_frames"] = int(
                aec_suppressed["other"])
            _last_capture_echo_stats["speaking_source"] = speaking_source
            _last_capture_echo_stats["qualify_overrun"] = bool(
                qualify_overrun)

        # [PERF] P1-19 — the capture is finalised and about to be handed to STT.
        _mark_turn(marks, "capture_end", {"seconds": round(duration, 2)})
        return audio

    except sr.WaitTimeoutError:
        # [B116] A 1 s overlap poll is not an idle LISTENER: it must not
        # count toward recalibration. Only a real listen timeout does.
        if wait_timeout is None:
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


#: [LF35] Remote spoken-text cache (replies voiced by the API process, e.g.
#: typed-UI answers). Fetched at most once per cache window, on the STT
#: thread (never the capture thread), and never raises.
_remote_spoken_cache = {"at": 0.0, "texts": [], "speaking": False}
_REMOTE_SPOKEN_CACHE_SECONDS = 2.0


def _remote_spoken_texts():
    """[LF35] ``(text, age_seconds)`` the API process said aloud recently.

    The voice process cannot see the API process's ``speak()`` calls, so the
    echo-transcript guard asks the renderer - the same cross-process shape
    ``/aec/reference`` uses, with the age measured on the API's own clock.
    A failure means "no remote texts", never an exception.

    [B115] The payload's ``speaking`` flag (any entry with age 0.0) is cached
    alongside the texts; ``_remote_is_speaking`` reads it without I/O.
    """
    now = time.monotonic()
    if now - _remote_spoken_cache["at"] < _REMOTE_SPOKEN_CACHE_SECONDS:
        return _remote_spoken_cache["texts"]
    texts = []
    speaking = False
    try:
        import json as _json
        from urllib.request import Request, urlopen

        from backend.config import BACKEND_PORT as _PORT
        from backend.services import local_auth

        request = Request(
            f"http://127.0.0.1:{_PORT}/speak/last-text", method="GET",
            headers=local_auth.auth_headers())
        with urlopen(request, timeout=0.3) as response:
            payload = _json.loads(response.read().decode("utf-8") or "{}")
        for item in (payload.get("texts") or []):
            try:
                text = str(item.get("text") or "")
                age = float(item.get("age_seconds") or 0.0)
            except Exception:
                continue
            if text and age <= ECHO_TRANSCRIPT_MAX_AGE:
                texts.append((text, age))
        if "speaking" in payload:
            speaking = bool(payload.get("speaking"))
        else:
            speaking = any(age <= 0.0 for _text, age in texts)
    except Exception:
        texts = []
        speaking = False
    _remote_spoken_cache["at"] = now
    _remote_spoken_cache["texts"] = texts
    _remote_spoken_cache["speaking"] = speaking
    return texts


def _remote_is_speaking(now=None, max_age=_REMOTE_SPOKEN_CACHE_SECONDS):
    """[B115] Cached API-process "an entry is open" flag; never fetches."""
    now = time.monotonic() if now is None else now
    cache = _remote_spoken_cache
    return (bool(cache.get("speaking"))
            and (now - float(cache.get("at") or 0.0)) <= max_age)


def _playback_evidence(now, playback_active):
    """[B115] Why Jarvis counts as audible at *now* ("" = not audible).

    The reference timeline is fed by whichever process plays (the local actor
    or /aec/reference), so it crosses the process boundary; the local speaking
    flag and the cached remote flag are OR'd in. Capture-thread safe: no I/O.
    """
    if playback_active:
        return "reference"
    last = _aec_last_reference_at
    if (last is not None
            and (now - last) < SPEECH_START_PLAYBACK_TAIL_SECONDS):
        return "reference"
    try:
        if listener_state.is_speaking():
            return "local-flag"
    except Exception:
        pass
    return "remote-flag" if _remote_is_speaking(now) else ""


def _echo_words(text):
    """[B106] Canonical words for echo matching only; never modify output.

    STT output and spoken text carry different punctuation (hyphens,
    apostrophes, quotes) for the same utterance, so matching runs on
    canonical words: casefolded, lexical hyphens joined ("re-sealed" and
    "resealed" are the same word), everything else split on non-word
    characters.
    """
    text = (text or "").casefold().replace("\u2019", "'")
    # Join lexical hyphens, but not an em dash between separate words.
    text = re.sub(
        r"(?<=[^\W\d_])[-\u2010\u2011](?=[^\W\d_])",
        "",
        text,
    )
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", text)


def _short_spoken_echo_match(transcript_words, spoken_words):
    """[B106] Return (matched, best_coverage) for a playback-overlapped
    fragment of 2-4 words.

    Short fragments are only checked when the capture overlapped playback
    (see :func:`_is_echo_of_spoken_reply`): a 2-4 word transcript that
    recognizably reproduces a slice of what Jarvis was saying at that
    moment is echo. Two-word fragments require exact anchors (with one
    narrow reordered-form exception); three words require full coverage
    of some window; four words tolerate a single substitution/omission
    but still need three ordered anchors including a meaningful adjacent
    pair unless all four matched.
    """
    count = len(transcript_words)
    if not 2 <= count <= 4 or not spoken_words:
        return False, 0.0

    def same_word(left, right):
        if left == right:
            return True
        # Character fuzz is not allowed for short/control-sized words.
        return (
            min(len(left), len(right)) >= 5
            and difflib.SequenceMatcher(
                None, left, right, autojunk=False
            ).ratio() >= 0.90
        )

    if count == 2:
        # Two words require exact canonical anchors.
        for start in range(len(spoken_words) - 1):
            pair = spoken_words[start:start + 2]
            if transcript_words == pair:
                return True, 1.0

            # STT may reorder a two-word fragment ("re-sealed honey"
            # versus "honey is resealed"). Permit this narrow form only,
            # not arbitrary unordered subsets of the reply.
            triple = spoken_words[start:start + 3]
            if (
                len(triple) == 3
                and triple[1] in {"is", "a", "an", "the"}
                and transcript_words == [triple[2], triple[0]]
            ):
                return True, 1.0
        return False, 0.0

    if count == 3:
        best = 0.0
        for start in range(len(spoken_words) - 2):
            window = spoken_words[start:start + 3]
            coverage = sum(
                same_word(left, right)
                for left, right in zip(transcript_words, window)
            ) / 3.0
            best = max(best, coverage)
            if coverage == 1.0:
                return True, best
        return False, best

    # Four words may contain one STT substitution/omission. Require three
    # ordered exact anchors in a small window, including a meaningful
    # adjacent pair unless all four words matched.
    best = 0.0
    filler = {"a", "an", "and", "is", "one", "please", "said", "sir", "the"}
    for start in range(len(spoken_words)):
        for width in (4, 5, 6):
            window = spoken_words[start:start + width]
            if len(window) < 3:
                continue
            matcher = difflib.SequenceMatcher(
                None, transcript_words, window, autojunk=False
            )
            blocks = matcher.get_matching_blocks()
            anchors = sum(block.size for block in blocks)
            best = max(best, anchors / 4.0)
            if anchors == 4:
                return True, best
            if anchors < 3:
                continue
            meaningful_pair = any(
                block.size >= 2
                and any(
                    window[index] not in filler
                    and window[index + 1] not in filler
                    for index in range(block.b, block.b + block.size - 1)
                )
                for block in blocks
            )
            if meaningful_pair:
                return True, best
    return False, best


def _spoken_text_match_ratio(transcript_norm, spoken_norm):
    """[LF35] Best canonical word similarity against a same-sized window.

    The mic usually catches only a SLICE of the reply (the capture starts
    mid-reply), so the transcript is slid across the spoken text both
    ways. difflib over canonical words tolerates the STT errors an echo
    transcript carries.
    """
    tw = _echo_words(transcript_norm)
    sw = _echo_words(spoken_norm)
    if not tw or not sw:
        return 0.0

    short, long = (tw, sw) if len(tw) <= len(sw) else (sw, tw)
    best = 0.0
    for index in range(len(long) - len(short) + 1):
        ratio = difflib.SequenceMatcher(
            None,
            short,
            long[index:index + len(short)],
            autojunk=False,
        ).ratio()
        best = max(best, ratio)
        if best >= 0.999:
            break
    return best


# ── [B111] re-transcription-tolerant echo matching ─────────────────────
# STT rewrites a reply: digits become words, words become digits, numbers
# are read digit-by-digit or as years, and a few homophones flip. These
# helpers canonicalize BOTH sides for MATCHING ONLY (never for output) and
# decide how much of a transcript is, in order, a window of the reply.

def _cardinal_words(n):
    if n < 20:
        return [_NUM_ONES[n]]
    if n < 100:
        tens, ones = divmod(n, 10)
        return [_NUM_TENS[tens]] + ([_NUM_ONES[ones]] if ones else [])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return ([_NUM_ONES[hundreds], "hundred"]
                + (_cardinal_words(rest) if rest else []))
    for scale, name in ((10 ** 6, "million"), (1000, "thousand")):
        if n >= scale:
            head, rest = divmod(n, scale)
            return (_cardinal_words(head) + [name]
                    + (_cardinal_words(rest) if rest else []))
    return [_NUM_ONES[int(d)] for d in str(n)]


def _number_readings(digits):
    """[B111] How TTS/STT may say a digit string: cardinal, digit-by-digit
    ("two zero two" -> canonical "two oh two"), clock ("two forty five"),
    year ("twenty twenty six", "nineteen oh five")."""
    readings = []
    if len(digits) <= 9 and (len(digits) == 1 or digits[0] != "0"):
        readings.append(_cardinal_words(int(digits)))
    readings.append([_NUM_ONES[int(d)] for d in digits])
    if len(digits) == 3 and digits[0] != "0" and digits[1] != "0":
        readings.append([_NUM_ONES[int(digits[0])]]
                        + _cardinal_words(int(digits[1:])))
    if len(digits) == 4 and digits[0] != "0":
        head, tail = int(digits[:2]), int(digits[2:])
        if tail == 0:
            readings.append(_cardinal_words(head) + ["hundred"])
        elif tail < 10:
            readings.append(_cardinal_words(head) + ["oh", _NUM_ONES[tail]])
        else:
            readings.append(_cardinal_words(head) + _cardinal_words(tail))
    unique = []
    for reading in readings:
        if reading not in unique:
            unique.append(reading)
    return unique


def _echo_match_variants(text, cap=_ECHO_VARIANT_CAP):
    """[B111] Canonical token sequences for MATCHING ONLY (never output)."""
    text = _DIGIT_GROUP_RE.sub("", text or "")
    options = []
    for word in _echo_words(text):
        if word.isascii() and word.isdigit():
            choices = _number_readings(word)
        else:
            joined = _TENS_UNITS_RE.fullmatch(word)
            choices = ([[joined.group(1), joined.group(2)]] if joined
                       else [[word]])
        options.append([[_ECHO_HOMOPHONES.get(t, t) for t in choice]
                        for choice in choices])
    variants = []
    for combo in itertools.product(*options):
        variants.append([t for part in combo for t in part])
        if len(variants) >= cap:
            break
    return [v for v in variants if v]


def _contains_run(sequence, run):
    size = len(run)
    return size > 0 and any(sequence[i:i + size] == run
                            for i in range(len(sequence) - size + 1))


def _echo_coverage(tw, sw):
    """[B111] (coverage, longest_run): share of the transcript's content
    words found IN ORDER inside one bounded window of the spoken words,
    and the longest contiguous content-word block."""
    content = {i for i, w in enumerate(tw) if w not in _ECHO_FILLER}
    if not content or not sw:
        return 0.0, 0
    width = min(len(sw), len(tw) + max(3, len(tw) // 2))
    best = (0.0, 0)
    for start in range(len(sw) - width + 1):
        window = sw[start:start + width]
        blocks = difflib.SequenceMatcher(
            None, tw, window, autojunk=False).get_matching_blocks()
        matched, run = set(), 0
        for block in blocks:
            hits = [i for i in range(block.a, block.a + block.size)
                    if i in content]
            matched.update(hits)
            run = max(run, len(hits))
        score = (len(matched) / float(len(content)), run)
        best = max(best, score)
        if best[0] >= 0.999:
            break
    return best


def _relaxed_echo_match(normalized, spoken, one_word=False):
    """[B111] Number/homophone-tolerant echo match (uncertain captures)."""
    spoken_variants = _echo_match_variants(spoken, cap=4)
    best = 0.0
    for tw in _echo_match_variants(normalized):
        for sw in spoken_variants:
            if one_word:
                if _contains_run(sw, tw):
                    return True, 1.0
                continue
            if 2 <= len(tw) <= 4:
                hit, ratio = _short_spoken_echo_match(tw, sw)
                best = max(best, ratio)
                if hit:
                    return True, ratio
            if sum(1 for t in tw if t not in _ECHO_FILLER) < \
                    ECHO_COVERAGE_MIN_CONTENT:
                continue
            coverage, run = _echo_coverage(tw, sw)
            best = max(best, coverage)
            if (coverage >= ECHO_COVERAGE_RATIO
                    and run >= ECHO_COVERAGE_MIN_RUN):
                return True, coverage
    return False, best


def _is_control_utterance(normalized, words):
    """[B111] F35 commands are exempt from the echo guard, always."""
    if len(words) == 1 and words[0] in _ECHO_EXEMPT_WORDS:
        return True
    classifier = _control_classifier
    if classifier is None:
        return False
    try:
        return bool(classifier(normalized))
    except Exception:
        return False


def _echo_candidates():
    """[B111] ``(text, age)`` the guard matches against (local + remote)."""
    candidates = []
    try:
        # Same-process replies (the voice conversation path).
        from backend.services import voice as _voice
        candidates.extend(_voice.recent_spoken_texts(ECHO_TRANSCRIPT_MAX_AGE))
    except Exception:
        pass
    try:
        # API-process replies (typed UI, announcements) - already
        # age-filtered on the renderer's clock.
        candidates.extend(_remote_spoken_texts())
    except Exception:
        pass
    usable = []
    for text, age in candidates:
        try:
            age = float(age)
        except (TypeError, ValueError):
            continue
        if 0.0 <= age <= ECHO_TRANSCRIPT_MAX_AGE and _echo_words(text):
            usable.append((text, age))
    return usable


def _capture_is_uncertain(capture):
    """[B111] The capture overlapped playback, its speech began while
    Jarvis was speaking, and the canceller proved nothing (or B109/B110
    flagged it). Only then do the relaxed rules apply.

    [B120] A capture with NO reference frame while Jarvis was audible
    (speaking_at_onset from the local/remote flag) is uncertain by
    construction: without a reference the canceller cannot prove anything.
    """
    if not capture.get("speaking_at_onset"):
        return False
    if int(capture.get("had_reference_frames", 0) or 0) <= 0:
        return int(capture.get("suppressed_frames", 0) or 0) == 0
    return (capture.get("verdict") == "check-transcript"
            or capture.get("onset_trust") == "provisional"
            or int(capture.get("suppressed_frames", 0) or 0) == 0)


def _control_prefix_rescue(normalized):
    """[B111] An echo-dominated transcript that STARTS with a real control
    phrase keeps that phrase - unless the reply itself said it."""
    words = (normalized or "").split()
    if len(words) < 2:
        return None
    spoken = [_echo_words(text) for text, _age in _echo_candidates()]
    for size in range(min(4, len(words) - 1), 0, -1):
        prefix = " ".join(words[:size])
        prefix_words = _echo_words(prefix)
        is_control = bool(size == 1 and prefix_words[:1]
                          and prefix_words[0] in _ECHO_EXEMPT_WORDS)
        if not is_control and _control_classifier is not None:
            try:
                is_control = bool(_control_classifier(prefix))
            except Exception:
                return None
        if not is_control:
            continue
        if any(_contains_run(sw, prefix_words) for sw in spoken):
            return None
        return prefix
    return None


#: [B103] Voice-process echo telemetry: guard counters and the last
#: capture's echo numbers, read by :func:`aec_voice_telemetry` on the
#: voice-state publisher thread (never the audio path).
_echo_guard_lock = threading.Lock()
_echo_guard_stats = {
    "checks": 0,
    "last_candidates": 0,
    "dropped": 0,
    "last_best_ratio": 0.0,
    "last_inspected": False,
    "last_reason": "not-checked",
    "last_uncertain": False,
    "control_rescues": 0,
    # [B114] Short-echo drops that happened WITHOUT the uncertain gate, and
    # a per-reason counter so the live test can see which rule fired.
    "overlap_short_drops": 0,
    "drops_by_reason": {},
}
_last_capture_echo_stats = {
    "had_reference_frames": 0,
    "suppressed_frames": 0,
    "other_frames": 0,
    "onset_confirmed": False,
    "verdict": "not-captured",
    "onset_trust": "none",
    "speaking_at_onset": False,
    "speaking_source": "none",
    "qualify_overrun": False,
}
_last_unsuppressed_commit_log_at = 0.0


# ── [B116] STT-overlap capture ──────────────────────────────────────────────
# The listener used to be deaf while the ONE cloud STT call ran: a barge-in
# in that 0.5-2 s window was never sampled. The worker below captures the
# NEXT utterance while the current audio is being transcribed. It never calls
# STT itself, and only one capture runs at a time, so the one-utterance-one-
# transcription invariant and the single-writer gate rule both hold.

class _OverlapCapture:
    """[B116] One bounded capture of the NEXT utterance while STT runs."""

    def __init__(self):
        self.stt_done = threading.Event()
        self.done = threading.Event()
        self.cancelled = threading.Event()
        self.audio = None
        #: Echo-stats snapshot of THIS audio (never the global at consume
        #: time: the worker would otherwise overwrite the current turn's
        #: numbers while its STT is still running).
        self.capture = None


_overlap_lock = threading.Lock()
_overlap_slot = None
_stt_overlap_stats = {"started": 0, "used": 0, "empty": 0,
                      "discarded": 0, "polls": 0}


def _overlap_capture_worker(slot):
    try:
        while not slot.cancelled.is_set():
            with _echo_guard_lock:
                _stt_overlap_stats["polls"] += 1
            audio = _capture_audio(None,
                                   wait_timeout=STT_OVERLAP_POLL_SECONDS,
                                   begin_turn=False)
            if audio is not None:
                with _echo_guard_lock:
                    slot.capture = dict(_last_capture_echo_stats)
                slot.audio = audio
                return
            if slot.stt_done.is_set():
                return
    except Exception as exc:
        print(f"[LISTENER] overlap capture error: {exc}")
    finally:
        slot.done.set()


def _start_overlap_capture():
    global _overlap_slot
    slot = _OverlapCapture()
    with _overlap_lock:
        _overlap_slot = slot
    with _echo_guard_lock:
        _stt_overlap_stats["started"] += 1
    threading.Thread(target=_overlap_capture_worker, args=(slot,),
                     name="listener-overlap-capture", daemon=True).start()
    return slot


def _take_overlap_capture():
    """[B116] (audio, capture_stats) of a finished overlap capture, else
    (None, None). Waits (bounded) for an in-flight one: it owns the mic."""
    global _overlap_slot
    with _overlap_lock:
        slot, _overlap_slot = _overlap_slot, None
    if slot is None:
        return None, None
    slot.stt_done.set()
    finished = slot.done.wait(_STT_OVERLAP_JOIN_SECONDS)
    with _echo_guard_lock:
        if not finished or slot.cancelled.is_set() or slot.audio is None:
            _stt_overlap_stats["empty"] += 1
            if not finished:
                print("[LISTENER] overlap capture did not finish - dropped")
            return None, None
        _stt_overlap_stats["used"] += 1
    return slot.audio, slot.capture


def discard_pending_capture(wait_seconds=None):
    """[B116] Never let a prefetched capture cross a voice-off/shutdown
    boundary. Bounded wait; never raises.

    The slot is deliberately left in place until the worker finishes: the
    next ``_take_overlap_capture`` waits for it and discards its audio. That
    keeps the single-capture-writer rule even when the 1.5 s wait here times
    out (a capture that is already reading the mic cannot be aborted from
    this thread).
    """
    global _overlap_slot
    with _overlap_lock:
        slot = _overlap_slot
    if slot is None:
        return False
    slot.cancelled.set()
    slot.stt_done.set()
    slot.done.wait(STT_OVERLAP_POLL_SECONDS + 0.5
                   if wait_seconds is None else wait_seconds)
    with _overlap_lock:
        if _overlap_slot is slot:
            _overlap_slot = None
    with _echo_guard_lock:
        _stt_overlap_stats["discarded"] += 1
    return True


def _is_echo_of_spoken_reply(normalized, *, playback_overlapped=False,
                             uncertain=False):
    """[LF35] True when *normalized* matches what Jarvis just said aloud.

    The layered signal path (fdaf canceller -> playback onset bar ->
    capture rules) should stop echo much earlier; this is the guarantee
    layer: even if every signal-level defence misses on bad hardware, a
    transcript of our own reply is never committed as user speech.

    [B106] Long transcripts (>= :data:`ECHO_TRANSCRIPT_MIN_WORDS` words)
    are always checked against the spoken-text registry. Short fragments
    of 2-4 words are only checked when *playback_overlapped* says Jarvis
    was audible during the capture (reference frames, a speaking flag at
    onset, or a recently-ended API entry - B120) - the live failure mode
    where the tail of a reply string is transcribed verbatim.

    [B111] *uncertain* (see :func:`_capture_is_uncertain`) enables evidence
    that survives re-transcription: one-word echoes of the reply being
    played, number/homophone-normalized short matches, and ordered content
    coverage for long transcripts. The F35 vocabulary is always exempt.

    [B114] With playback overlap, one-word (age-bounded) and 2-4-word
    number/homophone evidence apply whether or not the capture is
    uncertain; ``uncertain`` still gates long-transcript coverage only.

    Every call updates :data:`_echo_guard_stats` in a ``finally`` block,
    so an early return can never leave a previous decision's numbers
    behind as if they were this call's.
    """
    usable = []
    words = []
    matched = False
    best_ratio = 0.0
    inspected = False
    reason = "empty"
    uncertain_mode = False

    try:
        words = _echo_words(normalized)
        if not ECHO_GUARD_ENABLED:
            reason = "disabled"
            return False
        if not words:
            return False
        uncertain_mode = bool(uncertain and playback_overlapped)
        one_word = len(words) == 1
        if one_word and not playback_overlapped:
            reason = "one-word-exempt"
            return False
        if _is_control_utterance(normalized, words):
            # F35 is never eaten. A certain capture's single command word
            # keeps its B106 reason name.
            reason = ("one-word-exempt" if one_word and not uncertain_mode
                      else "control-exempt")
            return False
        short_check = playback_overlapped and 2 <= len(words) <= 4
        if (not one_word and not uncertain_mode and not short_check
                and len(words) < ECHO_TRANSCRIPT_MIN_WORDS):
            reason = "short-without-overlap"
            return False

        usable = _echo_candidates()
        inspected = True
        reason = "no-candidates" if not usable else "no-match"
        for spoken, age in usable:
            if one_word:
                if age > ECHO_ONE_WORD_MAX_AGE:
                    continue
                hit, ratio = _relaxed_echo_match(normalized, spoken,
                                                 one_word=True)
                kind = "one-word-echo"
            elif short_check:
                hit, ratio = _short_spoken_echo_match(
                    words, _echo_words(spoken))
                kind = "short-echo"
                if not hit:
                    # [B114] Number/homophone evidence for 2-4-word
                    # fragments is no longer gated on "uncertain": the live
                    # self-query survives exactly in the trusted/partially
                    # suppressed captures where the old gate said False.
                    hit, ratio2 = _relaxed_echo_match(normalized, spoken)
                    ratio = max(ratio, ratio2)
                    kind = "relaxed-short-echo"
            else:
                ratio = _spoken_text_match_ratio(normalized, spoken)
                hit = ratio >= ECHO_TRANSCRIPT_RATIO
                kind = "window-echo"
                if not hit and uncertain_mode:
                    hit, ratio2 = _relaxed_echo_match(normalized, spoken)
                    ratio = max(ratio, ratio2)
                    kind = "coverage-echo"
            best_ratio = max(best_ratio, ratio)
            if hit:
                matched = True
                reason = kind
                break
        return matched
    finally:
        # [B103] The guard's decisions are telemetry now: a silent pass
        # with zero candidates and a correct drop look identical from
        # outside. The finally block records EVERY call, including early
        # returns, so the published numbers always describe the last one.
        with _echo_guard_lock:
            _echo_guard_stats["checks"] += 1
            _echo_guard_stats["last_candidates"] = len(usable)
            _echo_guard_stats["last_best_ratio"] = round(best_ratio, 3)
            _echo_guard_stats["last_inspected"] = inspected
            _echo_guard_stats["last_reason"] = reason
            _echo_guard_stats["last_uncertain"] = uncertain_mode
            if matched:
                _echo_guard_stats["dropped"] += 1
                by = _echo_guard_stats.setdefault("drops_by_reason", {})
                by[reason] = by.get(reason, 0) + 1
                if not uncertain_mode and len(words) <= 4:
                    _echo_guard_stats["overlap_short_drops"] = (
                        _echo_guard_stats.get("overlap_short_drops", 0) + 1)


def aec_voice_telemetry():
    """[B103] Plain-dict snapshot of this process's echo-defense state.

    Published as the ``aec`` field of the voice-state snapshot so a live
    speaker test is diagnosable from outside: whether fdaf locked the
    speaker->mic delay, how many frames were adapted/frozen/suppressed,
    whether captures overlapped playback, and whether the transcript echo
    guard saw candidates or dropped transcripts. Never raises.
    """
    try:
        raw = _aec_state() or {}
    except Exception:
        raw = {}
    canceller_info = raw.get("canceller") or {}
    fdaf_metrics = dict(canceller_info.get("metrics") or {})
    counters = raw.get("counters") or {}
    lag_info = raw.get("lag") or {}

    lag_locked = bool(
        fdaf_metrics.get("lag_locked")
        if "lag_locked" in fdaf_metrics
        else lag_info.get("locked", False)
    )
    lag_seconds = (
        fdaf_metrics.get("lag_seconds")
        if fdaf_metrics.get("lag_seconds") is not None
        else lag_info.get("locked_seconds")
    )
    metrics = dict(fdaf_metrics)
    metrics.update({
        "frames": int(fdaf_metrics.get("frames", 0) or 0),
        "adapted": int(fdaf_metrics.get("adapted", 0) or 0),
        "frozen": int(fdaf_metrics.get("frozen", 0) or 0),
        "suppressed": int(fdaf_metrics.get("suppressed", 0) or 0),
        "last_erle_db": fdaf_metrics.get("last_erle_db"),
        "resets": int(fdaf_metrics.get("resets", 0) or 0),
    })
    signal_stats = {
        "echo_suppressed": int(counters.get("echo_suppressed", 0) or 0),
        "had_reference": int(counters.get("had_reference", 0) or 0),
    }
    with _echo_guard_lock:
        guard = {
            "enabled": bool(ECHO_GUARD_ENABLED),
            "checks": int(_echo_guard_stats["checks"]),
            "candidates": int(_echo_guard_stats["last_candidates"]),
            "dropped": int(_echo_guard_stats["dropped"]),
            "last_best_ratio": _echo_guard_stats["last_best_ratio"],
            # [B106] Distinguishes: never inspected / empty registry /
            # inspected with no match / rejected recognizable echo.
            "inspected": bool(_echo_guard_stats.get("last_inspected",
                                                    False)),
            "last_reason": _echo_guard_stats.get("last_reason",
                                                 "not-checked"),
            # [B111] Whether the relaxed rules were allowed for that call.
            "uncertain": bool(_echo_guard_stats.get("last_uncertain",
                                                    False)),
            "control_rescues": int(
                _echo_guard_stats.get("control_rescues", 0) or 0),
            # [B114] Short-echo drops that fired without the uncertain gate,
            # and which rule matched each drop.
            "overlap_short_drops": int(
                _echo_guard_stats.get("overlap_short_drops", 0) or 0),
            "drops_by_reason": dict(
                _echo_guard_stats.get("drops_by_reason") or {}),
        }
        last_capture = dict(_last_capture_echo_stats)
        onset = dict(_onset_stats)
        # [B116] Prefetch counters: started/polls/used/empty/discarded.
        stt_overlap = dict(_stt_overlap_stats)
    now = time.monotonic()
    cold_gate = {
        "trust_active": _aec_trust_active(now),
        "epoch_trusted": _aec_epoch_trusted(now),
        "suppressed_age": (round(now - _aec_last_suppressed_at, 2)
                           if _aec_last_suppressed_at is not None else None),
        "reference_age": (round(now - _aec_last_reference_at, 2)
                          if _aec_last_reference_at is not None else None),
        # [B113] Path-generation provenance: a bump here with a null
        # suppressed_age is a cancelled reset/relearn hand-off, not a cold
        # start; the live test reads both.
        "path_generation": _aec_path_generation,
        "trusted_generation": _aec_trust_generation,
        "trust_invalidations": int(_aec_trust_invalidations),
    }
    return {
        "mode": raw.get("mode", "unknown"),
        "canceller": canceller_info.get("name") or raw.get("mode", "unknown"),
        "degraded": bool(raw.get("degraded", False)),
        "lag_locked": lag_locked,
        "lag_seconds": lag_seconds,
        # [B105] The active estimator's raw diagnostics (source fdaf or
        # signal-path): correlation peak, candidate lag, searched range,
        # votes and reason - what the live speaker test needs to tell a
        # cold window apart from a decorrelated one.
        "lag": dict(lag_info),
        "metrics": metrics,
        "echo_suppressed": signal_stats["echo_suppressed"],
        "had_reference": signal_stats["had_reference"],
        "signal_path": signal_stats,
        "last_capture": last_capture,
        "echo_guard": guard,
        "cold_gate": cold_gate,
        # [B109] Onset provenance: trusted stop vs provisional transcript
        # qualification, plus the bounded qualify cuts.
        "onset": onset,
        # [B116] Whether the mic was read while STT ran, and what it found.
        "stt_overlap": stt_overlap,
        "errors": int(_aec_error_count),
    }


def _maybe_log_unsuppressed_playback_commit(capture=None):
    """[B103] Rate-limited one-liner when a playback-overlapped capture was
    committed without a single suppressed frame.

    That is the exact situation the live speaker failure hid in: frames had
    a rendered reference, the canceller flagged nothing, and the transcript
    still reached the commit boundary. One line per
    :data:`AEC_COMMIT_WARN_INTERVAL` keeps the console readable; the full
    numbers ride the voice-state snapshot.

    [B116] *capture* is the stats snapshot that belongs to THIS audio. The
    global can already describe a prefetched next utterance by the time the
    commit warning runs.
    """
    global _last_unsuppressed_commit_log_at
    source = capture if capture is not None else _last_capture_echo_stats
    with _echo_guard_lock:
        ref_frames = int(source.get("had_reference_frames", 0) or 0)
        sup_frames = int(source.get("suppressed_frames", 0) or 0)
        verdict = source.get("verdict")
        onset = source.get("onset_trust")
    if ref_frames <= 0 or sup_frames > 0:
        return False
    now = time.monotonic()
    if (now - _last_unsuppressed_commit_log_at) < AEC_COMMIT_WARN_INTERVAL:
        return False
    _last_unsuppressed_commit_log_at = now
    tel = aec_voice_telemetry()
    guard = tel.get("echo_guard") or {}
    metrics = tel.get("metrics") or {}
    lag = tel.get("lag") or {}
    corr = lag.get("best_correlation")
    print(
        "[LISTENER] Playback-overlapped capture committed without AEC "
        "suppression (ref_frames=%d, mode=%s, lag_locked=%s, lag_s=%s, "
        "lag_corr=%s, lock_proven=%s, erle_db=%s, erle_prior_db=%s, "
        "adapted=%d, frozen=%d, relocks=%d, rejections=%d, verdict=%s, "
        "onset=%s, guard=%s, guard_candidates=%d)" % (
            ref_frames, tel.get("mode"), tel.get("lag_locked"),
            tel.get("lag_seconds"),
            None if corr is None else round(corr, 2),
            metrics.get("lock_proven"), metrics.get("last_erle_db"),
            metrics.get("last_erle_prior_db"),
            int(metrics.get("adapted", 0) or 0),
            int(metrics.get("frozen", 0) or 0),
            int(metrics.get("relocks", 0) or 0),
            int(metrics.get("lock_rejections", 0) or 0),
            verdict, onset, guard.get("last_reason"),
            guard.get("candidates", 0))
    )
    return True


def listen(marks=None, overlap_capture=False):
    """Capture and transcribe one committed utterance (None when there is none).

    *marks* is the optional [PERF] P1-19 telemetry sink for this turn (see
    :func:`_capture_audio`); the STT boundaries are recorded on it, including
    WHICH engine produced the transcript.

    [B116] *overlap_capture* is opt-in and only the voice listener passes it:
    it starts a prefetch of the NEXT utterance while this one's ONE STT call
    runs. Direct callers (tests, probes) keep the old serial behaviour. The
    prefetched audio is consumed by the next call, and its echo-stats snapshot
    travels with it, so the guard never reads another capture's numbers.

    One utterance, one transcription: the capture returns the complete audio and
    exactly ONE engine is asked to transcribe it. There is no partial-window
    layer and no early commit any more, so the transcript always comes from the
    selected engine over the WHOLE utterance instead of a trailing-window
    approximation. It still goes through the hallucination belt, because it is
    about to be acted on.
    """
    global empty_listen_count

    prefetched, capture = _take_overlap_capture()
    if prefetched is not None:
        audio = prefetched
        capture = capture or {}
        _mark_turn(marks, "capture_prefetched",
                   {"seconds": round(_audio_duration_seconds(audio), 2)})
        # [F34] The prefetched audio's turn starts only now, AFTER the
        # previous transcript's push, so turn order is preserved.
        try:
            _turn_stabilizer.begin_turn()
        except Exception:
            pass
    else:
        audio = _capture_audio(marks)
        if audio is None:
            return None
        # [B116] Snapshot THIS capture's evidence before the prefetch worker
        # can overwrite the global while the STT call runs.
        with _echo_guard_lock:
            capture = dict(_last_capture_echo_stats)
    slot = (_start_overlap_capture()
            if overlap_capture and STT_OVERLAP_CAPTURE else None)

    listener_state.set_thinking(True)
    try:
        print("[LISTENER] Processing speech...")
        _mark_turn(marks, "stt_start")
        stt_started_at = time.monotonic()
        raw, text, language = recognize_multilingual(audio)
        if slot is not None:
            slot.stt_done.set()
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
        # [LF35] Echo of our own reply: with speakers instead of earphones
        # the mic hears the reply, and on bad hardware every signal-level
        # defence (fdaf canceller, onset bar, capture rules) can miss. A
        # transcript of what Jarvis just said aloud is never a user query.
        # [B106] The capture's own overlap evidence decides whether short
        # (2-4 word) fragments are checked; without it, a real short
        # command spoken after playback must not be eaten.
        had_reference = int(
            capture.get("had_reference_frames", 0) or 0) > 0
        # [B115/B120] The capture-time evidence is process-local, and the
        # cached remote flag may be stale on the capture thread. Refresh the
        # cross-process half on this (STT) thread whenever local evidence is
        # missing - NOT only when reference frames exist: a reply voiced by
        # ElevenLabs/SAPI or by the API process has no reference at all,
        # and the old condition made that refresh circular.
        remote_texts = []
        if not capture.get("speaking_at_onset"):
            try:
                remote_texts = list(_remote_spoken_texts() or [])
            except Exception:
                remote_texts = []
            if _remote_is_speaking():
                capture["speaking_at_onset"] = True
                capture["speaking_source"] = "remote-flag"
        # [B120] An API-process entry that ENDED within this capture's span
        # (STT time + audio length + room tail) overlapped the capture too.
        # Bounded by the cache TTL; it only enables the short/one-word
        # matchers, which drop nothing but slices of that reply.
        overlap_window = ((time.monotonic() - stt_started_at)
                          + _audio_duration_seconds(audio)
                          + SPEECH_START_PLAYBACK_TAIL_SECONDS)
        remote_recent = any(age <= overlap_window
                            for _text, age in remote_texts)
        # [B120] Overlap is AUDIBLE evidence, not only reference frames.
        playback_overlapped = (had_reference
                               or bool(capture.get("speaking_at_onset"))
                               or remote_recent)
        normalized = _normalize_text(text)
        rescued_control = False
        if _is_echo_of_spoken_reply(
                normalized,
                playback_overlapped=playback_overlapped,
                uncertain=_capture_is_uncertain(capture)):
            # [B111] A real command mixed into an echo transcript still
            # belongs to the user: keep just the leading control phrase.
            rescued = _control_prefix_rescue(normalized)
            if not rescued:
                print(f"[LISTENER] Ignoring echo of own reply: {normalized}")
                _record_empty_listen("echo transcript")
                return None
            with _echo_guard_lock:
                _echo_guard_stats["control_rescues"] += 1
            print(f"[LISTENER] Control command kept from an echo capture: "
                  f"{rescued}")
            raw = text = rescued
            rescued_control = True
        # [B103] A committed transcript that overlapped playback with zero
        # suppressed frames is the suspicious case live failures hide in:
        # one rate-limited line, then continue exactly as before. A rescued
        # control is a deliberate exception, not a miss.
        if not rescued_control:
            _maybe_log_unsuppressed_playback_commit(capture)
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
        if slot is not None:
            slot.stt_done.set()
        listener_state.set_thinking(False)
