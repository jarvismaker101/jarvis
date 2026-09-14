import os
import threading
import time

from backend.services.audio_input import FIXED_IDLE_ENERGY_THRESHOLD

_MAX_THRESHOLD = max(80, int(os.getenv("JARVIS_MAX_ENERGY_THRESHOLD", "700")))

_speaking = False
_thinking = False
_user_speaking = False
_voice_input_enabled = True
_normal_threshold = FIXED_IDLE_ENERGY_THRESHOLD
_interrupt_threshold = FIXED_IDLE_ENERGY_THRESHOLD
_recognizer = None
remaining_speech = ""

_done_speaking = threading.Event()
_done_speaking.set()

_state_lock = threading.Lock()
_last_user_started_at = 0.0
_last_user_finished_at = 0.0


def _clamp_threshold(value, minimum=80, maximum=_MAX_THRESHOLD):
    return max(minimum, min(int(value), maximum))


def register_recognizer(recognizer):
    global _recognizer, _normal_threshold, _interrupt_threshold
    with _state_lock:
        _recognizer = recognizer
        _normal_threshold = _clamp_threshold(FIXED_IDLE_ENERGY_THRESHOLD)
        _interrupt_threshold = _normal_threshold
        _recognizer.dynamic_energy_threshold = False
        active_threshold = _interrupt_threshold if _speaking else _normal_threshold
        if int(getattr(_recognizer, "energy_threshold", active_threshold)) != active_threshold:
            _recognizer.energy_threshold = active_threshold


def _apply_threshold(threshold):
    with _state_lock:
        recognizer = _recognizer
    threshold = _clamp_threshold(threshold)
    if recognizer is not None:
        current = int(getattr(recognizer, "energy_threshold", threshold))
        if current == threshold:
            return False
        recognizer.dynamic_energy_threshold = False
        recognizer.energy_threshold = threshold
        print(f"[MIC] Threshold -> {threshold}")
        return True
    return False


def set_speaking(speaking):
    global _speaking
    with _state_lock:
        _speaking = speaking
        normal_threshold = _normal_threshold
        interrupt_threshold = _interrupt_threshold

    if speaking:
        _done_speaking.clear()
        _apply_threshold(interrupt_threshold)
        print("[VOICE] Jarvis speaking")
    else:
        _done_speaking.set()
        _apply_threshold(normal_threshold)
        print("[VOICE] Listening")


def is_speaking():
    with _state_lock:
        return _speaking


def set_thinking(thinking):
    global _thinking
    with _state_lock:
        _thinking = thinking


def is_thinking():
    with _state_lock:
        return _thinking


def mark_user_speaking(speaking):
    global _user_speaking, _last_user_started_at, _last_user_finished_at
    now = time.monotonic()

    with _state_lock:
        if speaking and not _user_speaking:
            _last_user_started_at = now
            print("[TURN] User started speaking")
        elif not speaking and _user_speaking:
            _last_user_finished_at = now
            print("[TURN] User finished speaking")

        _user_speaking = speaking


def is_user_speaking():
    with _state_lock:
        return _user_speaking


def get_voice_state():
    with _state_lock:
        if _user_speaking:
            status = "hearing"
        elif _speaking:
            status = "speaking"
        elif _thinking:
            status = "thinking"
        else:
            status = "listening"

        threshold = int(getattr(_recognizer, "energy_threshold", _normal_threshold))
        return {
            "status": status,
            "assistant_speaking": _speaking,
            "user_speaking": _user_speaking,
            "thinking": _thinking,
            "threshold": threshold,
            "last_user_started_at": _last_user_started_at,
            "last_user_finished_at": _last_user_finished_at,
            "voice_input_enabled": _voice_input_enabled,
        }


def set_voice_input_enabled(enabled):
    global _voice_input_enabled
    with _state_lock:
        _voice_input_enabled = bool(enabled)
        return _voice_input_enabled


def is_voice_input_enabled():
    with _state_lock:
        return _voice_input_enabled


def set_remaining(text):
    global remaining_speech
    with _state_lock:
        remaining_speech = text


def pop_remaining():
    global remaining_speech
    with _state_lock:
        text = remaining_speech
        remaining_speech = ""
        return text


def has_remaining():
    with _state_lock:
        return bool(remaining_speech)


def get_remaining():
    """Peek at the unplayed speech without consuming it (F35 pause)."""
    with _state_lock:
        return remaining_speech
