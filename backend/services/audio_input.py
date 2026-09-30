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


#: [P1-05] The mic is opened AT the AEC rate so a captured frame and an
#: echo-cancelled frame are the same rate and ``_combine_audio_chunks`` can
#: never be handed a mix (joining two rates time-warps the audio - bad STT and
#: a wrong duration). 16 kHz is also what every STT engine downstream wants,
#: so this removes a resample instead of adding one.
#:
#: Some Windows devices REFUSE a requested rate. That must never lose the mic,
#: so every open falls back to the device's native rate and the rate-safe join
#: in the listener becomes the safety net (the two layers are complementary,
#: not either/or).
try:
    from backend.services.echo_cancel import AEC_SAMPLE_RATE as AEC_CAPTURE_RATE
except Exception:  # pragma: no cover - keep audio_input importable standalone
    AEC_CAPTURE_RATE = 16000

#: What the last successful open actually got. Observability for the P1-05
#: fallback: a device that refuses 16 kHz is visible here instead of silently
#: reintroducing mixed-rate chunks.
_MIC_RATE = {"requested": AEC_CAPTURE_RATE, "granted": None, "refused": False}


def mic_rate_state():
    """Snapshot of the capture rate: requested, granted, and whether refused."""
    return dict(_MIC_RATE)


def create_microphone(device_index=None, sample_rate=None):
    if sample_rate is None:
        return sr.Microphone(device_index=device_index)
    return sr.Microphone(device_index=device_index, sample_rate=int(sample_rate))


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


def open_microphone_at_preferred_rate(device_index=None):
    """Open (and ENTER) a microphone AT ``AEC_CAPTURE_RATE``.

    Falls back to the device's native rate if the preferred rate is refused.
    Returns the ENTERED ``sr.Microphone`` (``stream`` is non-None) or None on
    failure. Callers must NOT call ``__enter__`` again - ``sr.Microphone``
    asserts it is not already inside a context manager.

    ``sr.Microphone.__enter__`` swallows the open failure and leaves
    ``stream is None``, so a device that refuses 16 kHz is detected here, not
    raised. Falling back keeps the capture alive - P1-05's rate-safe join then
    guarantees correctness at the native rate instead.
    """
    native = None
    preferred = create_microphone(device_index, sample_rate=AEC_CAPTURE_RATE)
    try:
        preferred.__enter__()
    except Exception:
        preferred = None
    if preferred is not None and getattr(preferred, "stream", None) is not None:
        _MIC_RATE["granted"] = getattr(preferred, "SAMPLE_RATE", AEC_CAPTURE_RATE)
        _MIC_RATE["refused"] = False
        return preferred

    # Refused (or raised): never lose the mic over a preferred rate.
    _MIC_RATE["refused"] = True
    print(
        f"[MIC] Device refused {AEC_CAPTURE_RATE} Hz - capturing at its native "
        f"rate; mixed-rate audio is resampled per frame instead"
    )
    try:
        native = create_microphone(device_index)
        native.__enter__()
    except Exception:
        return None
    if getattr(native, "stream", None) is None:
        return None
    _MIC_RATE["granted"] = getattr(native, "SAMPLE_RATE", None)
    return native


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
        # [P1-05] Try the AEC rate FIRST so native and AEC rates coincide; a
        # device that refuses it falls back to native inside the helper. The
        # mic is returned ALREADY ENTERED, so it is used as-is here.
        microphone = open_microphone_at_preferred_rate(index)
        if microphone is None:
            if preferred_index is not None and index == preferred_index:
                _PREFERRED_BLOCKED_UNTIL[preferred_index] = now + _PREFERRED_BLOCKED_SECONDS
            continue
        source = microphone
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
