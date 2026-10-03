import os
import threading
import time

from backend.services.audio_input import FIXED_IDLE_ENERGY_THRESHOLD
from backend.services import event_bus

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

#: [P1-15] Observers told that the listening state CHANGED. The voice worker
#: registers one so a transition (listening -> thinking -> speaking) is
#: published the moment it happens instead of waiting for its 1s heartbeat —
#: that lag is what made the UI indicator feel untruthful.
#:
#: Hooks run OUTSIDE ``_state_lock`` and must be cheap and non-raising: the
#: capture thread calls these setters, so a hook that blocks (a network POST)
#: would put network latency directly on the audio path. The worker's hook only
#: sets an event.
_state_hooks = []


def register_state_hook(hook):
    """Register ``hook()`` for every state change. Returns the hook."""
    if hook is None:
        return None
    with _state_lock:
        if hook not in _state_hooks:
            _state_hooks.append(hook)
    return hook


def unregister_state_hook(hook):
    with _state_lock:
        try:
            _state_hooks.remove(hook)
        except ValueError:
            pass


def _notify_state_hooks():
    """Tell observers the state changed. Never raises, never holds the lock."""
    with _state_lock:
        hooks = list(_state_hooks)
    for hook in hooks:
        try:
            hook()
        except Exception:
            pass


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
    # [P1-15] Outside the lock: the publisher must see the new state at once.
    _notify_state_hooks()


def is_speaking():
    with _state_lock:
        return _speaking


def set_thinking(thinking):
    global _thinking
    with _state_lock:
        changed = _thinking != thinking
        _thinking = thinking
    if changed:
        _notify_state_hooks()


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

        changed = _user_speaking != speaking
        _user_speaking = speaking
    if changed:
        # [P1-15] "hearing" is a transition too: publish it immediately.
        _notify_state_hooks()


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
        changed = _voice_input_enabled != bool(enabled)
        _voice_input_enabled = bool(enabled)
    if changed:
        # [P1-15] A mute toggle is a visible state change: publish it now.
        _notify_state_hooks()
        # [S19] Push it to every event subscriber so the voice worker and UI
        # see the switch instantly instead of on their next /ui-state poll.
        event_bus.publish("voice_enabled", {"voice_input_enabled": _voice_input_enabled})
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
