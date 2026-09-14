import os
import time
from contextlib import contextmanager

import speech_recognition as sr


def _read_int_env(name, default):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError:
        print(f"[MIC] Ignoring invalid {name}={raw_value!r}")
        return default


_PREFERRED_INPUT_TERMS = (
    "microphone",
    "mic",
    "headset",
    "array",
    "input",
)
_AVOID_INPUT_TERMS = (
    "output",
    "speaker",
    "stereo mix",
    "steam streaming",
)
FIXED_IDLE_ENERGY_THRESHOLD = max(
    80,
    min(_read_int_env("JARVIS_IDLE_ENERGY_THRESHOLD", 300), 5000),
)


def list_microphone_names():
    try:
        return sr.Microphone.list_microphone_names()
    except Exception:
        return []


def _input_capable_indices():
    """Set of device indices with audio input channels, or None if unknown."""
    try:
        pyaudio = sr.Microphone.get_pyaudio().PyAudio()
        try:
            return {
                pyaudio.get_device_info_by_index(index)["index"]
                for index in range(pyaudio.get_device_count())
                if pyaudio.get_device_info_by_index(index).get("maxInputChannels", 0) > 0
            }
        finally:
            pyaudio.terminate()
    except Exception:
        return None


def resolve_microphone():
    names = list_microphone_names()
    input_indices = _input_capable_indices()

    def is_input(index):
        return input_indices is None or index in input_indices

    env_index = os.getenv("JARVIS_MIC_DEVICE_INDEX")
    if env_index:
        try:
            index = int(env_index)
            if 0 <= index < len(names) and is_input(index):
                return index, names[index], "env-index"
            print(f"[MIC] Ignoring invalid JARVIS_MIC_DEVICE_INDEX={env_index}")
        except ValueError:
            print(f"[MIC] Ignoring invalid JARVIS_MIC_DEVICE_INDEX={env_index}")

    env_name = os.getenv("JARVIS_MIC_NAME", "").strip().lower()
    if env_name:
        for index, name in enumerate(names):
            if env_name in name.lower() and is_input(index):
                return index, name, "env-name"
        print(f"[MIC] No input microphone matched JARVIS_MIC_NAME={env_name!r}")

    try:
        pyaudio = sr.Microphone.get_pyaudio().PyAudio()
        try:
            info = pyaudio.get_default_input_device_info()
            index = int(info["index"])
            name = names[index] if 0 <= index < len(names) else info.get("name", f"device {index}")
            return index, name, "default"
        finally:
            pyaudio.terminate()
    except Exception:
        pass

    fallback_candidates = []
    for index, name in enumerate(names):
        lowered = name.lower()
        if any(term in lowered for term in _PREFERRED_INPUT_TERMS) and not any(
            term in lowered for term in _AVOID_INPUT_TERMS
        ):
            fallback_candidates.append((index, name))
    for index, name in fallback_candidates:
        if is_input(index):
            return index, name, "fallback"

    return None, "system default", "implicit-default"


def create_microphone(device_index=None):
    if device_index is None:
        return sr.Microphone()
    return sr.Microphone(device_index=device_index)


_CANDIDATES_CACHE = None
_CANDIDATES_CACHE_AT = 0.0
_CANDIDATES_TTL = 15.0
_PREFERRED_BLOCKED_UNTIL = {}
_PREFERRED_BLOCKED_SECONDS = 120.0
_REPORTED_FALLBACK_SWITCH = {}


def _fallback_candidates():
    """Input-capable device indices, best-named first, cached briefly."""
    global _CANDIDATES_CACHE, _CANDIDATES_CACHE_AT

    now = time.monotonic()
    if _CANDIDATES_CACHE is not None and now - _CANDIDATES_CACHE_AT < _CANDIDATES_TTL:
        return _CANDIDATES_CACHE

    names = list_microphone_names()
    input_indices = _input_capable_indices() or set(range(len(names)))

    def score(index):
        if index >= len(names):
            return 0
        lowered = names[index].lower()
        score = 0
        for term in _PREFERRED_INPUT_TERMS:
            if term in lowered:
                score += 10
        for term in _AVOID_INPUT_TERMS:
            if term in lowered:
                score -= 5
        return score

    _CANDIDATES_CACHE = sorted(
        (index for index in input_indices if index < len(names)),
        key=lambda index: (-score(index), index),
    )
    _CANDIDATES_CACHE_AT = now
    return _CANDIDATES_CACHE


def _device_label(index, names=None):
    if names is None:
        names = list_microphone_names()
    if index is not None and 0 <= index < len(names):
        return f"{names[index]!r} (index {index})"
    if index is not None:
        return f"index {index}"
    return "default input device"


def _enter_microphone(preferred_index, fallback):
    """Open the first usable microphone; returns (microphone, source)."""
    now = time.monotonic()

    candidates = []
    if preferred_index is not None and now >= _PREFERRED_BLOCKED_UNTIL.get(preferred_index, 0):
        candidates.append(preferred_index)
    if fallback:
        for index in _fallback_candidates():
            if index != preferred_index:
                candidates.append(index)
        candidates.append(None)

    names = list_microphone_names()
    for index in candidates:
        microphone = create_microphone(index)
        source = microphone.__enter__()
        if source is not None and source.stream is not None:
            if index != preferred_index and _REPORTED_FALLBACK_SWITCH.get(
                preferred_index
            ) != index:
                _REPORTED_FALLBACK_SWITCH[preferred_index] = index
                print(
                    f"[MIC] Preferred mic {_device_label(preferred_index, names)} unavailable - "
                    f"using {_device_label(index, names)} instead"
                )
            return microphone, source
        if preferred_index is not None and index == preferred_index:
            _PREFERRED_BLOCKED_UNTIL[preferred_index] = now + _PREFERRED_BLOCKED_SECONDS

    raise OSError(
        "No microphone could be opened - every input device failed (busy, disabled, or unavailable)."
    )


@contextmanager
def open_microphone(device_index=None, fallback=True):
    microphone, source = _enter_microphone(device_index, fallback)

    try:
        yield source
    finally:
        if source.stream is not None:
            microphone.__exit__(None, None, None)


def resolve_working_microphone_index(preferred_index=None):
    """Return the index of a microphone that actually opens (preferred, else best fallback)."""
    microphone, source = _enter_microphone(preferred_index, fallback=True)
    try:
        return microphone.device_index
    finally:
        if source.stream is not None:
            source.__exit__(None, None, None)


def apply_fixed_energy_threshold(recognizer, threshold=FIXED_IDLE_ENERGY_THRESHOLD):
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = threshold
    return int(threshold)


def calibrate_recognizer(recognizer, device_index=None, duration=1.0, source=None):
    # The recognizer always runs with dynamic_energy_threshold disabled and a
    # fixed threshold (apply_fixed_energy_threshold overrides any reading), so
    # the ~1s ambient-noise recording from adjust_for_ambient_noise was pure
    # waste. Skipping it removes a microphone open + record from boot time.
    if source is not None and getattr(source, "stream", None) is None:
        raise OSError("Microphone source stream is not open; cannot calibrate.")

    return apply_fixed_energy_threshold(recognizer)
