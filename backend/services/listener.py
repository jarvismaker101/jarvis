import atexit
import os
import re
import threading
import time

import speech_recognition as sr
import webrtcvad

from backend import listener_state
from backend.services import model_registry
from backend.services.audio_input import (
    FIXED_IDLE_ENERGY_THRESHOLD,
    calibrate_recognizer,
    create_microphone,
    list_microphone_names,
    resolve_microphone,
    resolve_working_microphone_index,
)
from backend.services.earcons import play_capture_complete_earcon
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
    aec_state as _aec_state,
    cancelled_capture_frame as _aec_capture_frame,
)
from backend.services.transcript_stabilizer import (
    TranscriptWindow,
    stabilizer as _turn_stabilizer,
)

LISTEN_TIMEOUT_SECONDS = 10
MAX_PHRASE_SECONDS = 15
#: [PERF] Trailing silence that ends an utterance. This is a direct
#: latency/accuracy knob: every turn pays it in full, but a value that is too
#: low cuts the user off mid-sentence (thinking pauses, trailing clauses).
#: The default is deliberately UNCHANGED at 1.2s; lowering it is now a runtime
#: decision (JARVIS_PAUSE_THRESHOLD, e.g. 0.8 on a fast local loop) rather than
#: a code edit. Raising it helps accuracy at a straight latency cost.
PAUSE_THRESHOLD_SECONDS = float(
    os.getenv("JARVIS_PAUSE_THRESHOLD", "1.2"))
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

        source = create_microphone(MIC_DEVICE_INDEX)
        source.__enter__()
        if source.stream is None:
            raise OSError(
                f"Failed to open microphone (device index {MIC_DEVICE_INDEX}). "
                "The device may be unavailable, disabled, or already in use."
            )
        _microphone_source = source
        return source


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


# ── F34: local-only partial windows during capture ─────────────────────────
# Active conversation must produce REAL overlapping partial windows BEFORE
# the utterance ends, so local agreement can commit a transcript early and
# unstable text never has to authorize an action. Partials are produced by
# the LOCAL whisper engine only (never a cloud STT), at most every
# JARVIS_PARTIAL_TRANSCRIBE_SECONDS and at most
# JARVIS_MAX_PARTIAL_WINDOWS per utterance, so a long phrase stays bounded.
PARTIAL_TRANSCRIBE_MIN_SECONDS = float(
    os.getenv("JARVIS_PARTIAL_TRANSCRIBE_SECONDS", "1.5"))
MAX_PARTIAL_WINDOWS_PER_UTTERANCE = int(
    os.getenv("JARVIS_MAX_PARTIAL_WINDOWS", "8"))
#: [PERF] Trailing audio ceiling for ONE partial window. Transcribing the whole
#: accumulated utterance every window is quadratic (window k re-sends ~1.5k
#: seconds) and it happens inside the real-time capture loop. Only the tail can
#: change the newest transcript, so older audio is dropped. The stabiliser's
#: end_ms is still absolute, so window ranges keep advancing exactly as before.
PARTIAL_MAX_AUDIO_SECONDS = float(
    os.getenv("JARVIS_PARTIAL_MAX_AUDIO_SECONDS", "20"))
#: [PERF] A partial window is an early hint, never the committed answer, so it
#: must not be able to stall the capture loop for the daemon's full 15s budget.
PARTIAL_STT_TIMEOUT_SECONDS = float(
    os.getenv("JARVIS_PARTIAL_STT_TIMEOUT", "4"))


def _bounded_audio_tail(chunks, max_seconds):
    """Return the trailing ``chunks`` covering at most *max_seconds* of audio.

    Returns the list unchanged when the whole capture already fits, so the
    common short-utterance case is untouched.
    """
    if not chunks or not max_seconds or max_seconds <= 0:
        return chunks
    keep = 0
    total = 0.0
    for chunk in reversed(chunks):
        total += _audio_duration_seconds(chunk)
        keep += 1
        if total >= max_seconds:
            break
    if keep >= len(chunks):
        return chunks
    return chunks[len(chunks) - keep:]


_partial_observers = []
_partial_observer_lock = threading.Lock()


def register_partial_observer(callback):
    """Observe every PARTIAL transcript window as it is produced (F34)."""
    with _partial_observer_lock:
        if callback not in _partial_observers:
            _partial_observers.append(callback)
    return callback


def unregister_partial_observer(callback):
    with _partial_observer_lock:
        try:
            _partial_observers.remove(callback)
        except ValueError:
            pass


def _notify_partial(window):
    with _partial_observer_lock:
        observers = list(_partial_observers)
    for callback in observers:
        try:
            callback(window)
        except Exception:
            pass


def _cloud_stt_policy():
    """The shared explicit cloud-egress policy (F34/F36)."""
    try:
        from backend.services.wake_engine import cloud_stt_policy

        return cloud_stt_policy()
    except Exception:
        return "on"


def _transcribe_partial(audio):
    """Transcribe one partial window with the PARTIAL (short) deadline.

    A partial window is an early hint, not the answer, so it must not be able
    to hold the real-time capture loop for the daemon's full budget. The
    ``timeout`` keyword is passed defensively: many tests replace
    ``recognize_local_whisper`` with a single-argument stub, and a stub that
    cannot express a deadline must not turn a partial into an error.
    """
    try:
        return recognize_local_whisper(
            audio, timeout=PARTIAL_STT_TIMEOUT_SECONDS)
    except TypeError:
        return recognize_local_whisper(audio)


def _emit_partial_window(chunks, turn_id, index, duration_ms):
    """Transcribe the accumulated LOCAL audio and push one partial window.

    Returns the :class:`TranscriptWindow` (or None). Never raises, and never
    contacts anything but the local whisper engine.
    """
    audio = _combine_audio_chunks(chunks)
    if audio is None:
        return None
    try:
        transcript, language = _transcribe_partial(audio)
    except Exception as exc:
        print(f"[LISTENER] Partial transcription unavailable: {exc}")
        return None
    text = (transcript or "").strip()
    if not text:
        return None
    if is_hallucinated_transcript(text):
        # STT degeneracy over noise / TTS-echo audio (looped token, prompt
        # echo, memorised silence phrase): never becomes a partial window and
        # never reaches the stabilizer.
        return None
    window = TranscriptWindow(
        wid="partial-%s-%d" % (turn_id, index),
        text=text,
        final=False,
        language=language,
        start_ms=0,
        end_ms=int(duration_ms),
        turn=turn_id,
    )
    _notify_partial(window)
    try:
        _turn_stabilizer.push(window)
    except Exception as exc:
        print(f"[LISTENER] Stabilizer error: {exc}")
    return window


def recognize_multilingual(audio):
    """Recognize speech across languages.

    Returns (raw, normalized, language): the RAW transcription (case,
    newlines and quoted content intact) alongside the normalized control
    text used for logging/matching. (None, None, None) when nothing was
    understood.

    [F34] The cloud policy is explicit and enforced HERE, before any audio
    can leave the machine: under a local-only policy the local whisper
    engine is the only engine tried, and if it fails the utterance simply
    returns no transcript — nothing is transmitted externally.
    """
    # Selected listening engine (settings UI): resolved PER CALL from the
    # registry so a switch takes effect on the very next phrase, no
    # restart. Any registry hiccup degrades to the shipped default
    # (Inworld first).
    try:
        selected = model_registry.get_model_for_role("listening").get(
            "provider", "inworld"
        )
    except Exception:
        selected = "inworld"

    def _try_inworld():
        # Inworld STT with an English hint (conversation text must stay
        # roman/English script). Empty transcript = no speech -> second
        # opinions below. A Devanagari transcript means the hint was
        # ignored -> fall back.
        try:
            text = recognize_inworld(audio, language="en")
            normalized = _normalize_text(text)
            if normalized and is_hallucinated_transcript(normalized):
                print(f"[LISTENER] Ignoring STT hallucination: {normalized}")
                return None
            if normalized:
                if any("\u0900" <= ch <= "\u097F" for ch in text):
                    print("[LISTENER] Inworld STT returned non-English script, falling back")
                else:
                    print(f"[HEARD:inworld] {normalized}")
                    return text, normalized, "auto"
        except sr.UnknownValueError:
            pass
        except Exception as exc:
            print(f"[LISTENER] Inworld STT failed: {exc}")
        return None

    def _try_local_whisper():
        try:
            transcript, lang = recognize_local_whisper(audio)
            normalized = _normalize_text(transcript)
            if normalized and is_hallucinated_transcript(normalized):
                print(f"[LISTENER] Ignoring STT hallucination: {normalized}")
                return None
            if normalized:
                print(f"[HEARD:local-whisper] {normalized}")
                return transcript, normalized, lang
        except sr.UnknownValueError:
            pass
        except Exception as exc:
            print(f"[LISTENER] Local whisper STT failed: {exc}")
        return None

    if _cloud_stt_policy() != "on":
        # F34: local-only. Inworld and Google/Groq are cloud engines; with
        # the policy off they are never called — a local failure produces no
        # transcript instead of an upload.
        print("[LISTENER] Cloud STT disabled by policy - local whisper only")
        result = _try_local_whisper()
        if result:
            return result
        return None, None, None

    if selected == "whisper":
        # Local whisper primary, Inworld as the first fallback.
        result = _try_local_whisper() or _try_inworld()
    else:
        # Inworld primary (default; also any unexpected provider value).
        result = _try_inworld() or _try_local_whisper()
    if result:
        return result

    for language in RECOGNITION_LANGUAGES:
        try:
            text = recognize_google_or_groq(
                recognizer,
                audio,
                language,
                log_prefix="LISTENER",
            )
            normalized = _normalize_text(text)
            if normalized and is_hallucinated_transcript(normalized):
                print(f"[LISTENER] Ignoring STT hallucination: {normalized}")
                continue
            if normalized:
                print(f"[HEARD:{language}] {normalized}")
                return text, normalized, language
        except sr.UnknownValueError:
            continue
        except sr.RequestError as exc:
            print(f"[LISTENER] Recognition request failed [{language}]: {exc}")
            continue
        except Exception as exc:
            print(f"[LISTENER] Recognition error [{language}]: {exc}")
            continue

    return None, None, None


def _combine_audio_chunks(chunks):
    if not chunks:
        return None

    first = chunks[0]
    frame_data = b"".join(chunk.frame_data for chunk in chunks if chunk and chunk.frame_data)
    if not frame_data:
        return None

    return sr.AudioData(frame_data, first.sample_rate, first.sample_width)


def _audio_duration_seconds(audio):
    bytes_per_second = audio.sample_rate * audio.sample_width
    if bytes_per_second <= 0:
        return 0.0
    return len(audio.frame_data) / bytes_per_second


def _post_backend_speak_stop():
    """Best-effort stop of the API-process TTS (separate OS process).

    The voice process hears the user; chat replies / task narration are
    voiced by the API backend, so a local-only stop would leave them
    playing. Localhost POST, short timeout, never raises.

    F51: ``/speak/stop`` is a private control endpoint and auth fails
    CLOSED, so this request carries the per-launch token — without it the
    barge-in stop silently 401s and Jarvis keeps talking over the user.
    """
    try:
        from backend.config import BACKEND_PORT as _PORT
        from backend.services import local_auth
        from urllib.request import Request, urlopen
        req = Request(
            f"http://127.0.0.1:{_PORT}/speak/stop",
            data=b"{}",
            method="POST",
            headers=local_auth.auth_headers(),
        )
        with urlopen(req, timeout=0.4):
            pass
        return True
    except Exception:
        return False


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


def barge_in_on_speech_onset():
    """Instant barge-in: user started speaking while Jarvis TTS is playing.

    Stops local TTS synchronously (generation bump + fish PCM flush via
    stop_speaking) and stops the API-process TTS too.  When the API is
    speaking but the voice process is silent (cross-process gap), only
    the API POST is issued.  VAD-gated callers decide *when* — this only
    decides *how*.  Never raises.  Returns True if a stop was issued.
    """
    try:
        if listener_state.is_speaking():
            from backend.services.voice import stop_speaking as _stop
            _stop()
            _post_backend_speak_stop()
            return True
    except Exception:
        return False
    # Cross-process gap: the voice process is silent but the API may be
    # voicing a chat reply or websearch narration.  Only the API POST
    # (no local stop — nothing local is playing).
    if _api_is_speaking():
        _post_backend_speak_stop()
        return True
    return False


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


def _capture_audio():
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
        frame_id = 0
        # [F34] One conversation turn per capture: partial windows produced
        # here can only ever agree with windows of THIS utterance.
        try:
            turn_id = _turn_stabilizer.begin_turn()
        except Exception:
            turn_id = 0
        partial_state = {"seconds": 0.0, "emitted": 0, "duration_ms": 0.0}
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
                chunk, frame_id, time.monotonic())
            frame_id += 1
            filtered_chunks.append(filtered_chunk)
            frame_duration = _audio_duration_seconds(filtered_chunk)
            partial_state["duration_ms"] += frame_duration * 1000.0
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
                partial_state["seconds"] += frame_duration

            # [PERF] Onset / barge-in is evaluated BEFORE any partial-window
            # transcription. Transcribing a partial window is a blocking call
            # into the local whisper engine, so doing it first delayed the
            # moment the user is recognised as having started speaking - i.e.
            # it delayed barge-in by the whole transcription. Onset must never
            # queue behind transcription.
            if not speech_started and _should_confirm_speech_start(onset_chunks):
                speech_started = True
                listener_state.mark_user_speaking(True)
                user_marked_speaking = True
                # Barge-in: cut ANY in-progress TTS the instant the user
                # starts speaking (VAD-gated above, not phrase-gated).
                barge_in_on_speech_onset()

            # [F34] overlapping partial windows DURING capture: locally
            # transcribed, bounded, and never produced from echo-only
            # audio. They arrive before the utterance ends, which is what
            # lets local agreement commit (or refuse) early.
            #
            # [PERF] This is a blocking HTTP call into the whisper daemon and it
            # runs inside the real-time capture loop, so it is bounded twice:
            # the audio handed to the engine is capped to a trailing window
            # (older audio cannot change the newest transcript) and the request
            # itself has its own deadline. The local-agreement contract is
            # unchanged - only the amount of audio and the worst-case stall
            # are.
            if not suppressed and (
                partial_state["seconds"] >= PARTIAL_TRANSCRIBE_MIN_SECONDS
                and partial_state["emitted"]
                < MAX_PARTIAL_WINDOWS_PER_UTTERANCE
            ):
                partial_state["seconds"] = 0.0
                partial_state["emitted"] += 1
                window = _emit_partial_window(
                    _bounded_audio_tail(
                        filtered_chunks, PARTIAL_MAX_AUDIO_SECONDS),
                    turn_id, partial_state["emitted"],
                    partial_state["duration_ms"],
                )
                if window is not None:
                    print(
                        f"[LISTENER] Partial window {window.wid} "
                        f"({int(partial_state['duration_ms'])}ms): "
                        f"{_normalize_text(window.text)}"
                    )

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
            play_capture_complete_earcon()
        empty_listen_count = 0
        duration = _audio_duration_seconds(audio)
        print(
            f"[LISTENER] Speech captured - {duration:.2f}s | "
            f"threshold={int(recognizer.energy_threshold)}"
        )

        if not is_human_voice(audio):
            print("[LISTENER] Voice check uncertain - sending to STT anyway")

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


def listen():
    global empty_listen_count

    audio = _capture_audio()
    if audio is None:
        return None

    listener_state.set_thinking(True)
    try:
        print("[LISTENER] Processing speech...")
        raw, text, _language = recognize_multilingual(audio)
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
        # [F34] The returned utterance is a COMMITTED transcript: finals
        # commit immediately, agreeing overlapping partials would too, and a
        # partial-only/unstable transcript is never returned - so no
        # downstream action can fire on unstable text.
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
