import os
import re
import json
import subprocess
import sys
import threading
import time
import io
import atexit
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

# Load DLL search paths on Windows to prevent faster-whisper/ctranslate2 crash in virtual environment
if sys.platform == "win32":
    user_profile = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    possible_paths = [
        os.path.join(user_profile, "AppData", "Local", "Programs", "Ollama", "lib", "ollama", "cuda_v12"),
        os.path.join(user_profile, "anaconda3", "Library", "bin"),
        r"C:\Users\mayan\anaconda3\Library\bin",
    ]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        possible_paths.append(os.path.join(conda_prefix, "Library", "bin"))

    for path in possible_paths:
        if os.path.isdir(path):
            os.environ["PATH"] += os.pathsep + path
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(path)
                except Exception:
                    pass

import speech_recognition as sr

from backend.config import BACKEND_PORT, BASE_DIR

# G11 / F51 — the watcher is the launch parent: it mints the per-launch local
# command token once and injects it into EVERY child it owns (backend, voice,
# electron). Children forward it on every local command; the watcher itself
# uses it for its control-plane POSTs. An inherited JARVIS_LOCAL_TOKEN wins
# so a supervisor chain stays consistent.
from backend.services import local_auth
from backend.services import runtime_identity


def _ensure_local_token():
    token = os.getenv("JARVIS_LOCAL_TOKEN", "")
    if not token:
        token = local_auth.mint_token()
        os.environ["JARVIS_LOCAL_TOKEN"] = token
    return token


LOCAL_TOKEN = _ensure_local_token()

from backend.services.audio_input import (
    FIXED_IDLE_ENERGY_THRESHOLD,
    calibrate_recognizer,
    list_microphone_names,
    open_microphone,
    resolve_microphone,
    resolve_working_microphone_index,
)
from backend.services.transcription import (
    is_hallucinated_transcript,
    recognize_google_or_groq,
    recognize_inworld,
)

LISTEN_TIMEOUT_SECONDS = 8
PHRASE_TIME_LIMIT_SECONDS = 12
RECALIBRATE_AFTER_EMPTY_LISTENS = 12
RECALIBRATE_COOLDOWN_SECONDS = 20
RECOGNITION_LANGUAGES = tuple(
    language.strip()
    for language in os.getenv("JARVIS_STT_LANGUAGES", "en-IN").split(",")
    if language.strip()
)
if not RECOGNITION_LANGUAGES:
    RECOGNITION_LANGUAGES = ("en-IN",)
WATCHER_ENERGY_THRESHOLD = max(
    40,
    min(
        int(os.getenv("JARVIS_WATCHER_ENERGY_THRESHOLD", 80)),
        FIXED_IDLE_ENERGY_THRESHOLD,
    ),
)

recognizer = sr.Recognizer()
# Wake phrases are short, but queries can follow in the same utterance;
# 0.8s of silence still ends the phrase promptly without cutting a breath.
recognizer.pause_threshold = 0.8
recognizer.non_speaking_duration = 0.1
recognizer.phrase_threshold = 0.15
recognizer.dynamic_energy_threshold = False
recognizer.dynamic_energy_ratio = 1.2
recognizer.operation_timeout = 3
recognizer.energy_threshold = WATCHER_ENERGY_THRESHOLD

# Whisper model is hosted by a persistent daemon process (backend.whisper_daemon)
# so it survives watcher restarts — the watcher connects to it over HTTP
# instead of loading the model in-process. `whisper_model` stays as an
# in-process fallback used only when the daemon can't be started.
whisper_model = None
WHISPER_DAEMON_PORT = int(os.getenv("JARVIS_WHISPER_PORT", "8767"))
WHISPER_DAEMON_URL = f"http://127.0.0.1:{WHISPER_DAEMON_PORT}"
whisper_daemon_proc = None
whisper_daemon_ok = False

MIC_DEVICE_INDEX, MIC_NAME, MIC_SOURCE = None, "", ""
_mic_resolved = False


def ensure_mic():
    global MIC_DEVICE_INDEX, MIC_NAME, MIC_SOURCE, _mic_resolved
    if _mic_resolved:
        return MIC_DEVICE_INDEX is not None
    _mic_resolved = True
    try:
        index, name, source = resolve_microphone()
    except Exception as exc:
        print(f"[WATCHER] Mic detection failed: {exc}")
        return False
    MIC_DEVICE_INDEX, MIC_NAME, MIC_SOURCE = index, name, source
    if MIC_DEVICE_INDEX is not None:
        try:
            working_index = resolve_working_microphone_index(MIC_DEVICE_INDEX)
        except OSError as exc:
            working_index = None
            print(f"[WATCHER] Mic check failed: {exc}")
        if working_index != MIC_DEVICE_INDEX:
            working_names = list_microphone_names()
            working_label = (
                f"{working_names[working_index]!r}"
                if working_index is not None and 0 <= working_index < len(working_names)
                else "system default"
            )
            print(
                f"[WATCHER] Preferred mic {MIC_NAME!r} cannot be opened - "
                f"using {working_label} until it becomes available"
            )
    print(
        f"[WATCHER] Mic: {MIC_NAME!r} ({MIC_SOURCE})"
        if MIC_DEVICE_INDEX is not None
        else "[WATCHER] Using system default mic"
    )
    return MIC_DEVICE_INDEX is not None

VENV_PY = BASE_DIR / "backend" / "venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = sys.executable

jarvis_running = False
backend_proc = None
voice_proc = None
electron_proc = None
runtime_lock = threading.RLock()
control_server = None
empty_listen_count = 0
last_recalibrated_at = 0.0
WATCHER_CONTROL_PORT = int(os.getenv("JARVIS_WATCHER_CONTROL_PORT", "8766"))

# ── F52: owned process handles + bounded worker supervision ─────────────────
# Every process this supervisor creates (or explicitly adopts after identity
# verification) is retained below as a creation-identity-bound handle:
# startup, crash recovery and teardown act on the handle/pid WE recorded —
# never on "whoever happens to be listening on the port".
#
# The stop sets are explicit so "sleep" and "shutdown" stop different things
# on purpose:
#   * SLEEP (keep_backend=True)  — voice + UI only; the backend and the
#     resident daemons stay warm for an instant next wake.
#   * SHUTDOWN (keep_backend=False) — everything this supervisor owns,
#     including the whisper daemon, so no unintended daemon is left behind.
SLEEP_STOP_ROLES = ("voice", "electron")
SHUTDOWN_STOP_ROLES = ("backend", "voice", "electron", "whisper_daemon")
SUPERVISED_WORKER_ROLES = ("backend", "voice")


def _env_int(name, default):
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


#: How many crash restarts one worker may consume inside the rolling window.
WORKER_RESTART_BUDGET = max(0, _env_int("JARVIS_WORKER_RESTART_BUDGET", 3))
WORKER_RESTART_WINDOW_SECONDS = max(1.0, _env_float("JARVIS_WORKER_RESTART_WINDOW", 300.0))
WORKER_SUPERVISION_INTERVAL_SECONDS = max(
    0.1, _env_float("JARVIS_WORKER_SUPERVISION_INTERVAL", 2.0)
)
BACKEND_SPAWN_ATTEMPTS = 3
BACKEND_READY_TIMEOUT_SECONDS = 45

_owned_lock = threading.RLock()
_owned_processes = {}
_worker_supervisor_thread = None
_worker_supervisor_stop = threading.Event()


def _register_owned(role, proc=None, pid=None, identity=None, respawn=None, adopted=False):
    """Retain a worker handle bound to the identity it was created with."""
    with _owned_lock:
        previous = _owned_processes.get(role) or {}
        entry = {
            "role": role,
            "proc": proc,
            "pid": pid if pid is not None else getattr(proc, "pid", None),
            "identity": dict(identity or previous.get("identity") or {}),
            "respawn": respawn if respawn is not None else previous.get("respawn"),
            "restarts": list(previous.get("restarts") or []),
            "restart_budget": previous.get("restart_budget"),
            "window_seconds": previous.get("window_seconds"),
            "created_at": time.time(),
            "adopted": adopted,
        }
        _owned_processes[role] = entry
        return entry


def _owned_entry(role):
    with _owned_lock:
        return _owned_processes.get(role)


def _forget_owned(role, proc=None):
    """Drop a retained handle. A newer handle for the role is never dropped."""
    with _owned_lock:
        entry = _owned_processes.get(role)
        if entry is None:
            return None
        if proc is not None and entry.get("proc") is not None and entry.get("proc") is not proc:
            return entry
        return _owned_processes.pop(role)


def _owned_pids():
    with _owned_lock:
        return {entry.get("pid") for entry in _owned_processes.values() if entry.get("pid")}


def _clear_owned():
    """Forget every retained handle (teardown/tests)."""
    with _owned_lock:
        _owned_processes.clear()


def _stop_handle(proc, pid=None):
    """Stop a worker by its retained handle, falling back to its recorded pid."""
    if proc is not None:
        _terminate_process(proc)
        return True
    if pid:
        _taskkill_pid(pid)
        return True
    return False


def _stop_owned_role(role):
    entry = _forget_owned(role)
    if entry is None:
        return False
    return _stop_handle(entry.get("proc"), entry.get("pid"))


def _restart_allowed(entry, now=None):
    """Consume/peek the rolling restart budget for one worker."""
    now = time.monotonic() if now is None else now
    budget = entry.get("restart_budget")
    if budget is None:
        budget = WORKER_RESTART_BUDGET
    window = entry.get("window_seconds") or WORKER_RESTART_WINDOW_SECONDS
    recent = [t for t in (entry.get("restarts") or []) if now - t < window]
    entry["restarts"] = recent
    return len(recent) < max(0, int(budget))


def supervise_owned_workers():
    """F52 — one crash-recovery pass over the owned, supervised workers.

    Only workers this supervisor created (or adopted) are supervised, and
    only while the UI that would show them is still alive. Each role has a
    bounded restart budget inside a rolling window: a worker that keeps dying
    stays down and is reported degraded instead of restart-looping forever.
    Returns the list of roles restarted by this pass.
    """
    restarted = []
    if not jarvis_running:
        return restarted

    ui = _owned_entry("electron")
    ui_proc = ui.get("proc") if ui else None
    if ui_proc is not None and ui_proc.poll() is not None:
        # The UI is gone: that is a close, not a crash. wait_for_shutdown owns
        # the teardown, and a respawn here would race it.
        return restarted

    for role in SUPERVISED_WORKER_ROLES:
        entry = _owned_entry(role)
        if entry is None or entry.get("proc") is None:
            continue
        proc = entry["proc"]
        if proc.poll() is None:
            continue
        respawn = entry.get("respawn")
        if respawn is None:
            print(f"[WATCHER] {role} exited and has no respawn spec - leaving it down.")
            continue
        if not _restart_allowed(entry):
            print(
                f"[WATCHER] {role} crash restart budget exhausted "
                f"({WORKER_RESTART_BUDGET} in {int(WORKER_RESTART_WINDOW_SECONDS)}s) - "
                "leaving it down (degraded)."
            )
            continue
        print(f"[WATCHER] {role} exited unexpectedly - restarting it.")
        entry["restarts"].append(time.monotonic())
        try:
            new_proc = respawn()
        except Exception as exc:
            print(f"[WATCHER] {role} restart failed: {exc}")
            continue
        if new_proc is None:
            continue
        _register_owned(role, new_proc, identity=entry.get("identity"), respawn=respawn)
        restarted.append(role)
    return restarted


def _worker_supervisor_loop():
    while not _worker_supervisor_stop.wait(WORKER_SUPERVISION_INTERVAL_SECONDS):
        try:
            supervise_owned_workers()
        except Exception as exc:
            print(f"[WATCHER] worker supervision error: {exc}")


def start_worker_supervisor():
    """Start the crash-recovery watch over the workers this supervisor owns."""
    global _worker_supervisor_thread
    if _worker_supervisor_thread is not None and _worker_supervisor_thread.is_alive():
        return _worker_supervisor_thread
    _worker_supervisor_stop.clear()
    _worker_supervisor_thread = threading.Thread(
        target=_worker_supervisor_loop,
        name="worker-supervisor",
        daemon=True,
    )
    _worker_supervisor_thread.start()
    return _worker_supervisor_thread


def stop_worker_supervisor():
    """Stop crash recovery: a teardown must never race a respawn."""
    global _worker_supervisor_thread
    _worker_supervisor_stop.set()
    _worker_supervisor_thread = None

JARVIS_VARIANTS = [
    "jarvis",
    "jervis",
    "jarvish",
    "jarvice",
    "jarwis",
    "jarvas",
    "jarbus",
    "jar vis",
    "harvis",
    "garvis",
    "harvey",
]

WAKE_VARIANTS_EN = [
    "wake",
    "wake up",
    "wakeup",
    "makeup",
    "breakup",
    "hey",
    "hello",
    "start",
    "begin",
    "activate",
    "listen",
    "come online",
]

WAKE_VARIANTS_HI = [
    "utho",
    "uth ja",
    "uth jao",
    "jago",
    "jaago",
    "jag ja",
    "chalu karo",
    "chalu ho",
    "shuru karo",
    "shuru ho",
]

SHUTDOWN_VARIANTS_HI = [
    "band ho",
    "band karo",
    "band ho jao",
    "band ho ja",
    "band kar do",
    "bandh karo",
    "chalo band",
    "sab band",
]

WAKE_PATTERNS = [
    "wake up jarvis",
    "wakeup jarvis",
    "makeup jarvis",
    "hey jarvis",
    "hello jarvis",
    "jarvis wake up",
    "start jarvis",
    "jarvis start",
    "utho jarvis",
    "jarvis utho",
    "jago jarvis",
    "jarvis chalu ho",
    "jarvis shuru ho",
]


def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def phrase_windows(words, size):
    for index in range(0, len(words) - size + 1):
        yield " ".join(words[index:index + size])


def fuzzy_contains(text, variants, threshold):
    normalized_text = normalize_text(text)
    words = normalized_text.split()
    if not words:
        return False

    for variant in variants:
        normalized_variant = normalize_text(variant)
        if not normalized_variant:
            continue

        if normalized_variant in normalized_text:
            return True

        variant_words = normalized_variant.split()
        if len(variant_words) == 1:
            for word in words:
                if SequenceMatcher(None, word, normalized_variant).ratio() >= threshold:
                    return True
            continue

        for window in phrase_windows(words, len(variant_words)):
            if SequenceMatcher(None, window, normalized_variant).ratio() >= threshold:
                return True

    return False


def has_partial_jarvis(text):
    tokens = normalize_text(text).split()
    partial_prefixes = ("jar", "jarv", "jerv", "harv", "garv")
    partial_exact = {"jar", "jarr", "jarv", "jervis", "jarvis", "char"}

    for token in tokens:
        if token in partial_exact:
            return True
        if any(token.startswith(prefix) for prefix in partial_prefixes) and len(token) >= 3:
            return True

    for window in phrase_windows(tokens, 2):
        if window in {"jar vis", "jar vice", "jar vish"}:
            return True

    return False


def wake_pattern_match(text):
    normalized_text = normalize_text(text)
    if not normalized_text:
        return False

    for pattern in WAKE_PATTERNS:
        if SequenceMatcher(None, normalized_text, pattern).ratio() >= 0.74:
            return True

    return False


def is_wake_word(text):
    # An STT hallucination (the wake-bias prompt echoed back over noise, a
    # looped token, ...) can look EXACTLY like a wake phrase — never let one
    # launch the stack.
    if is_hallucinated_transcript(text):
        return False
    normalized_text = normalize_text(text)
    words = normalized_text.split()
    has_jarvis = fuzzy_contains(normalized_text, JARVIS_VARIANTS, threshold=0.76)
    has_strong_jarvis = fuzzy_contains(normalized_text, JARVIS_VARIANTS, threshold=0.84)
    has_wake_en = fuzzy_contains(normalized_text, WAKE_VARIANTS_EN, threshold=0.84)
    has_wake_hi = fuzzy_contains(normalized_text, WAKE_VARIANTS_HI, threshold=0.8)
    has_partial_name = has_partial_jarvis(normalized_text)
    short_direct_call = (has_strong_jarvis or has_partial_name) and len(words) <= 3
    if wake_pattern_match(normalized_text):
        return True
    if (has_jarvis or has_partial_name) and (has_wake_en or has_wake_hi):
        return True
    return short_direct_call


def is_shutdown_word(text):
    normalized_text = normalize_text(text)
    has_jarvis = fuzzy_contains(normalized_text, JARVIS_VARIANTS, threshold=0.76)
    english = has_jarvis and (
        "stop" in normalized_text
        or "shutdown" in normalized_text
        or "shut down" in normalized_text
    )
    hindi = has_jarvis and fuzzy_contains(normalized_text, SHUTDOWN_VARIANTS_HI, threshold=0.82)
    return english or hindi


def recalibrate_watcher(reason, force=False):
    global empty_listen_count, last_recalibrated_at

    now = time.monotonic()
    if not force:
        if empty_listen_count < RECALIBRATE_AFTER_EMPTY_LISTENS:
            return
        if now - last_recalibrated_at < RECALIBRATE_COOLDOWN_SECONDS:
            return

    try:
        calibrate_recognizer(recognizer, MIC_DEVICE_INDEX, duration=1.2)
    except (OSError, ValueError) as exc:
        print(f"[WATCHER] Recalibration failed - keeping fixed threshold: {exc}")
    threshold = WATCHER_ENERGY_THRESHOLD
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = threshold
    empty_listen_count = 0
    last_recalibrated_at = now
    print(
        f"[WATCHER] Recalibrated ({reason}) - mic: {MIC_NAME} ({MIC_SOURCE}) | "
        f"threshold: {threshold}"
    )


def extract_transcripts(result):
    """RAW transcripts from one recognition result (F12).

    Text is preserved byte-exact — case, punctuation, paths, URLs, flags,
    quotes and whitespace all survive into the candidate list, because the
    command tail is sliced from this text and forwarded verbatim. The
    normalized form is used ONLY for decisions (dedupe here, matching in
    ``is_wake_word``), never returned as the payload.
    """
    transcripts = []
    seen = set()

    if isinstance(result, str):
        result = {"alternative": [{"transcript": result}]}

    if not isinstance(result, dict):
        return transcripts

    for alternative in result.get("alternative", [])[:5]:
        raw = str(alternative.get("transcript", "") or "").strip()
        if not raw:
            continue
        # Decisions run on the normalized copy; the payload stays raw.
        normalized = normalize_text(raw)
        if len(normalized) < 2:
            continue
        if not any(char.isalpha() for char in raw):
            continue
        if normalized not in seen:
            seen.add(normalized)
            transcripts.append(raw)

    return transcripts


def _add_candidates(candidates, seen, texts):
    """Append RAW transcripts, deduped on their normalized form.

    One phrase heard by two engines ("Open Chrome" / "open chrome") is one
    candidate; the FIRST spelling seen is the one that survives, so a later
    lowercased variant can never replace the literal payload.
    """
    for raw in texts or ():
        text = str(raw or "").strip()
        if not text:
            continue
        if is_hallucinated_transcript(text):
            continue
        key = normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(text)
    return candidates


def find_wake_match(transcripts):
    """The first RAW transcript that is a wake phrase (matching normalizes)."""
    return next((transcript for transcript in transcripts if is_wake_word(transcript)), None)


def _whisper_daemon_health():
    """Return the daemon's /health payload, or None."""
    try:
        request = Request(f"{WHISPER_DAEMON_URL}/health", method="GET")
        with urlopen(request, timeout=1.0) as response:
            data = json.load(response)
            return data if isinstance(data, dict) else None
    except Exception:
        return None


def _whisper_daemon_healthy():
    data = _whisper_daemon_health()
    return bool(data and data.get("ok"))


def _attributed_whisper_pid():
    """Pid attributable to the live whisper daemon, or None (F52).

    The daemon reports its own service identity + pid; an older daemon that
    only stamped itself is accepted when the stamped pid really is the
    listener. Anything unattributable is left alone.
    """
    listeners = _pids_on_port(WHISPER_DAEMON_PORT)
    data = _whisper_daemon_health()
    pid = None
    if data and data.get("service") == "jarvis-whisper":
        candidate = data.get("pid")
        if isinstance(candidate, int) and candidate > 0:
            pid = candidate
    if pid is None:
        stamp = runtime_identity.read_instance_file("whisper_daemon")
        if stamp and stamp.get("role") == "whisper_daemon":
            candidate = stamp.get("pid")
            if isinstance(candidate, int) and candidate > 0:
                pid = candidate
    if pid is not None and pid in listeners:
        return pid
    return None


def _stop_whisper_daemon():
    """Stop the whisper daemon on FULL shutdown only (F52).

    Warm sleep intentionally keeps the resident model; a full shutdown must
    leave no unintended daemon behind, so the owned handle goes first and an
    adopted daemon is stopped by its attributable pid.
    """
    if _stop_owned_role("whisper_daemon"):
        return True
    pid = _attributed_whisper_pid()
    if pid:
        _taskkill_pid(pid)
        return True
    return False


def _env_seconds(name, default, minimum=0.0):
    """A duration from the environment, never a crash (F55)."""
    try:
        return max(minimum, float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return float(default)


# F55 — a bound daemon answers /health in milliseconds, so this window is
# "the spawn failed", not a model-load budget. It is deliberately short: the
# cost of waiting it out is nothing, while the cost of the old 60s window was
# a second resident copy of the model on a machine that had no room for it.
WHISPER_READY_TIMEOUT = _env_seconds("JARVIS_WHISPER_READY_TIMEOUT", 12.0, 1.0)
# F55 — a launcher process can exit while the interpreter it started keeps
# serving (the Windows venv shim does exactly that), so an exit earns a short
# window to bind instead of an immediate in-process model load.
WHISPER_EXIT_GRACE = _env_seconds("JARVIS_WHISPER_EXIT_GRACE", 4.0, 0.5)
# F55 — the same launcher-exit ambiguity applies to the backend: this venv's
# python.exe is a shim that starts the real interpreter, so its exit is not
# proof that the backend died. The health probe decides, and an exit earns a
# short grace first - without it a healthy backend was thrown away and killed.
BACKEND_EXIT_GRACE = _env_seconds("JARVIS_BACKEND_EXIT_GRACE", 4.0, 0.5)


def ensure_whisper_daemon():
    """Connect to the persistent whisper daemon, spawning it if needed.

    The daemon keeps the whisper model resident in memory/VRAM across
    watcher restarts, so re-running the watcher starts listening instantly
    instead of reloading the model. Returns True when the daemon is ready;
    on failure the caller falls back to the in-process model.

    F55 — "ready" means the daemon is alive and serving, which its
    bind-before-load design makes a sub-second event. The old contract only
    answered once a 10-20s model load had finished, so this wait expired on
    cold starts and the caller loaded a SECOND copy of the model in-process:
    a slow boot became a stuck one on any machine short of memory.
    """
    global whisper_daemon_proc, whisper_daemon_ok

    if _whisper_daemon_healthy():
        print(
            f"[WATCHER] Whisper daemon already running on port {WHISPER_DAEMON_PORT} "
            "- reusing it."
        )
        whisper_daemon_ok = True
        # F52 — an adopted daemon is still part of the shutdown set: retain
        # the pid this launch attributed to it.
        adopted_pid = _attributed_whisper_pid()
        if adopted_pid:
            _register_owned("whisper_daemon", None, pid=adopted_pid, adopted=True)
        return True

    # A daemon that is alive and bound is adopted by the health probe below, so
    # a leftover daemon from an earlier launch is REUSED (model and memory
    # included) rather than duplicated - and because a launch that finds one
    # never spawns another, they cannot accumulate.
    holders = _pids_on_port(WHISPER_DAEMON_PORT)
    if holders:
        # A listener that does not answer /health is not this supervisor's to
        # kill (F52), and spawning a second binder would only produce two
        # processes sharing one port on Windows.
        print(
            f"[WATCHER] Port {WHISPER_DAEMON_PORT} is held by pid(s) "
            f"{sorted(holders)} that do not serve the whisper API; refusing to "
            "start a second binder - using in-process model."
        )
        return False

    print(f"[WATCHER] Starting whisper daemon on port {WHISPER_DAEMON_PORT}...")
    try:
        whisper_daemon_proc = subprocess.Popen(
            [str(VENV_PY), "-m", "backend.whisper_daemon"],
            cwd=str(BASE_DIR),
            creationflags=_child_creationflags(),
        )
    except Exception as exc:
        print(f"[WATCHER] Could not start whisper daemon: {exc}")
        return False

    _register_owned("whisper_daemon", whisper_daemon_proc)

    started = time.time()
    deadline = started + WHISPER_READY_TIMEOUT
    exited_at = None
    while True:
        if _whisper_daemon_healthy():
            print(f"[WATCHER] Whisper daemon ready in {time.time() - started:.1f}s.")
            whisper_daemon_ok = True
            return True

        now = time.time()
        if exited_at is None and whisper_daemon_proc.poll() is not None:
            exited_at = now
        if exited_at is not None and now - exited_at >= WHISPER_EXIT_GRACE:
            # Dead pid must not stay tracked: Windows recycles pids (F52).
            print(
                "[WATCHER] Whisper daemon exited before it served "
                f"(exit code {whisper_daemon_proc.returncode}) - using in-process model."
            )
            whisper_daemon_proc = None
            return False
        if now >= deadline:
            print(
                "[WATCHER] Whisper daemon did not serve within "
                f"{WHISPER_READY_TIMEOUT:.0f}s - using in-process model."
            )
            return False
        time.sleep(0.25)


#: [ACCURACY] The in-process model is only the daemon-failure fallback, but it
#: must not silently keep a different size than the daemon: ONE env var drives
#: both (see backend/whisper_daemon.py, which ships "medium" for accuracy).
WHISPER_MODEL_SIZE = os.getenv("JARVIS_WHISPER_MODEL", "medium")
#: [LANGUAGE] English-only, matching the daemon (see whisper_daemon.
#: TRANSCRIBE_LANGUAGE). Auto-detect hallucinates whole sentences in random
#: languages on noisy/echo audio; empty restores auto-detect.
WHISPER_LANGUAGE = os.getenv("JARVIS_WHISPER_LANGUAGE", "en")


def _load_in_process_whisper():
    global whisper_model
    # Imported lazily: faster-whisper/ctranslate2 pulls ~2-4s of CUDA libs at
    # import, and the in-process model is only a daemon-failure fallback.
    from faster_whisper import WhisperModel
    print(f"[WATCHER] Initializing local faster-whisper ({WHISPER_MODEL_SIZE}) "
          "model on CUDA GPU...")
    try:
        whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="float16", local_files_only=True)
        print("[WATCHER] Local Whisper model loaded on GPU (CUDA).")
    except Exception as exc:
        print(f"[WATCHER] CUDA load failed: {exc}. Trying CPU fallback...")
        try:
            whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8", local_files_only=True)
            print("[WATCHER] Local Whisper model loaded on CPU.")
        except Exception as exc_cpu:
            print(f"[WATCHER] Local Whisper model fallback failed: {exc_cpu}. Using API transcription only.")
            whisper_model = None


# F55 — this must outlast the daemon's own wait for a model that is still
# loading (``JARVIS_WHISPER_TRANSCRIBE_WAIT``, 20s), or the first utterance
# after a cold boot fails here as a client timeout instead of being
# transcribed by the daemon that was about to answer.
WHISPER_TRANSCRIBE_TIMEOUT = _env_seconds("JARVIS_WHISPER_TRANSCRIBE_TIMEOUT", 25.0, 5.0)


def _transcribe_with_daemon(wav_bytes):
    """POST wav bytes to the whisper daemon. Returns (text, info) or None."""
    request = Request(
        f"{WHISPER_DAEMON_URL}/transcribe",
        data=wav_bytes,
        headers={
            "Content-Type": "application/octet-stream",
            # Wake matching ASKS for the wake-bias prompt (it helps spell the
            # wake words); the daemon only applies it for this purpose. The
            # conversation path never sends this header.
            "X-Jarvis-Purpose": "wake",
        },
        method="POST",
    )
    with urlopen(request, timeout=WHISPER_TRANSCRIBE_TIMEOUT) as response:
        payload = json.load(response)
    if not payload.get("ok"):
        return None
    return payload.get("text") or "", payload


def _cloud_stt_allowed():
    """The explicit cloud-egress policy for the wake path (F36).

    "off"/local-only means NON-WAKE SPEECH MUST NOT REACH AN ONLINE STT, not
    merely that the transcript is discarded afterwards. The one policy lives
    in ``backend.services.wake_engine`` and is shared with the active
    conversation path, so the two can never drift.
    """
    try:
        from backend.services import wake_engine

        return wake_engine.cloud_stt_allowed()
    except Exception:
        return True


def _selected_engine():
    """[S5] The ONE engine the wake path uses.

    It is the same selection the conversation path uses
    (``listener.selected_stt_engine()``: the settings model, clamped by the
    shared cloud policy), so the transcript that detects "Jarvis" and the
    transcript that becomes the command come from the SAME model the user
    picked. Resolution failure degrades to local whisper - never to a cloud
    engine the user did not select.
    """
    try:
        from backend.services import listener

        return listener.selected_stt_engine()
    except Exception as exc:
        print(f"[WATCHER] Engine resolution failed ({exc}) - using local whisper")
        return "whisper"


def _transcribe_local_once(audio):
    """Local whisper, ONE model, asked exactly once per utterance.

    The persistent daemon is the transport; the in-process model is the SAME
    model (same size, greedy decode) used only when that daemon cannot answer.
    It is not a different engine and never changes which model was selected.
    Returns the transcript text, or None when the selected engine produced
    nothing - the caller then reports no transcript instead of asking another
    engine.
    """
    if whisper_daemon_ok:
        try:
            transcribed = _transcribe_with_daemon(audio.get_wav_data())
            if transcribed:
                text, info = transcribed
                if text:
                    print(f"[HEARD:local-whisper] {text} "
                          f"(lang: {info.get('language')}, "
                          f"prob: {info.get('language_probability')})")
                    return text
        except Exception as exc:
            print(f"[WATCHER] Whisper daemon transcription error: {exc}")

    if whisper_model:
        try:
            wav_bytes = audio.get_wav_data()
            segments, info = whisper_model.transcribe(
                io.BytesIO(wav_bytes),
                temperature=0.0,
                language=WHISPER_LANGUAGE or None,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt="Jarvis, wake up, jervis, utho, jago, chalu",
            )
            text = "".join(seg.text for seg in segments).strip()
            if text:
                print(f"[HEARD:local-whisper] {text} "
                      f"(lang: {info.language}, prob: {info.language_probability:.2f})")
                return text
        except Exception as exc:
            print(f"[WATCHER] Local Whisper transcription error: {exc}")

    return None


def recognize_candidates(audio):
    """[S5] ONE engine, ONE pass, and its output IS the transcript.

    The old code ran a serial ladder - whisper daemon, then the in-process
    model, then an online engine over every configured language, calling the
    online engine TWICE per language when the "show all" variant came back
    empty. Whichever rung happened to answer decided both the wake verdict and
    the transcript the rest of the system saw, so the model actually used was
    decided by a race between engines rather than by the user's selection.

    Now the engine is the one selected in settings (clamped by the cloud
    policy), it is asked once, and whatever it says is final. No other engine
    is contacted, so a selected cloud engine cannot be preceded by a local
    guess and a selected local engine cannot be followed by an upload.
    """
    candidates = []
    seen = set()

    engine = _selected_engine()
    text = None
    if engine == "whisper":
        text = _transcribe_local_once(audio)
    elif engine == "inworld":
        try:
            text = recognize_inworld(audio, language="en")
        except sr.UnknownValueError:
            text = None
        except Exception as exc:
            print(f"[WATCHER] Inworld recognition error: {exc}")
            text = None
        if text:
            print(f"[HEARD:inworld] {text}")
    else:
        # google-or-groq, when it is ever offered for the listening role: ONE
        # call on the configured language. No "show all" probe, no second call
        # per language - the same double-pass this path used to make.
        try:
            text = recognize_google_or_groq(
                recognizer,
                audio,
                RECOGNITION_LANGUAGES[0],
                log_prefix="WATCHER",
            )
        except sr.UnknownValueError:
            text = None
        except sr.RequestError as exc:
            print(f"[WATCHER] Recognition request failed: {exc}")
            text = None
        except Exception as exc:
            print(f"[WATCHER] Recognition error: {exc}")
            text = None
        if text:
            print(f"[HEARD:{RECOGNITION_LANGUAGES[0]}] {text}")

    if not text:
        # The SELECTED engine produced nothing. That is a failed turn, not an
        # invitation to ask a different model (which is what produced wake
        # verdicts and commands from an engine the user never selected).
        return candidates, None

    transcripts = extract_transcripts(text)
    _add_candidates(candidates, seen, transcripts)
    return candidates, find_wake_match(transcripts)


def listen_once():
    global empty_listen_count

    try:
        microphone_context = open_microphone(MIC_DEVICE_INDEX)
        source = microphone_context.__enter__()
    except OSError as exc:
        print(f"[WATCHER] Cannot open microphone: {exc}")
        time.sleep(3)
        return [], None

    try:
        audio = recognizer.listen(
            source,
            timeout=LISTEN_TIMEOUT_SECONDS,
            phrase_time_limit=PHRASE_TIME_LIMIT_SECONDS,
        )

        from backend.services import wake_engine

        # F36: continuous candidate capture — the pre-roll ring is fed the
        # instant audio exists, BEFORE any wake decision is made. The old
        # code fed it only after a match, so an oversized chunk could empty
        # it and verification had nothing to overlap the phrase start.
        try:
            wake_engine.pre_roll().feed(audio)
        except Exception as exc:
            print(f"[WATCHER] Pre-roll capture error: {exc}")

        candidates, wake_match = recognize_candidates(audio)
        if not wake_match:
            # F36: the local keyword spotter is part of the WAKE DECISION,
            # not decoration. When openwakeword is unavailable it reports
            # False and this is a no-op; when it fires, the pre-roll is
            # verified locally before the stack boots.
            try:
                spotted = wake_engine.keyword_spot(audio)
            except Exception:
                spotted = False
            if spotted:
                if wake_engine.online_verify():
                    print("[WAKE] Keyword spot confirmed - launching")
                    return candidates, (candidates[0]
                                        if candidates else "keyword-spot")
                print("[WAKE] Keyword spot not verified - ignoring")
        if wake_match:
            empty_listen_count = 0
            # F36: the phrase tail after the wake window is a command —
            # when one exists, it is itself evidence of a real wake hit and
            # skips online verification; otherwise verify the pre-roll
            # (JARVIS_WAKE_ONLINE_VERIFY) before booting the stack.
            if not wake_engine.extract_command(wake_match, candidates) \
                    and not wake_engine.online_verify():
                wake_match = None
            if wake_match:
                return candidates, wake_match

        if candidates:
            empty_listen_count = 0
            return candidates, None

        empty_listen_count += 1
        recalibrate_watcher("no transcript")
        return [], None

    except sr.WaitTimeoutError:
        empty_listen_count += 1
        recalibrate_watcher("idle")
        return [], None
    except sr.RequestError as exc:
        print(f"[WATCHER] Listen request failed: {exc}")
        time.sleep(1)
        return [], None
    except Exception as exc:
        empty_listen_count += 1
        print(f"[WATCHER] Listen error: {exc}")
        recalibrate_watcher("listen error")
        return [], None
    finally:
        microphone_context.__exit__(None, None, None)


def _child_creationflags():
    return getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def _taskkill_pid(pid):
    if not pid:
        return
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
    )


def _terminate_process(proc):
    if proc is None:
        return
    if proc.poll() is not None:
        return
    _taskkill_pid(proc.pid)


def _backend_health():
    try:
        request = Request(f"http://127.0.0.1:{BACKEND_PORT}/health")
        with urlopen(request, timeout=0.8) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)
    except Exception:
        return None


def _expected_token_fingerprint():
    """The auth fingerprint a worker must report to be reused (F52).

    A restarted watcher mints a NEW per-launch token. A warm backend still
    holding the previous launch's token would 401 every command this
    supervisor sends, so reuse is decided on the fingerprint — never on the
    assumption that "a warm backend is probably ours".
    """
    return local_auth.fingerprint_for(LOCAL_TOKEN)


def _backend_identity(health=None, stamp=None):
    """Identity reported by the live backend (``/health`` + instance stamp)."""
    health = _backend_health() if health is None else health
    if not health or health.get("service") != "jarvis-backend":
        return {}
    stamp = runtime_identity.read_instance_file("backend") if stamp is None else stamp
    identity = {
        "instance_id": health.get("instance_id"),
        "protocol": health.get("protocol"),
        "auth": health.get("auth"),
        "pid": health.get("pid"),
    }
    if stamp and stamp.get("instance_id") == health.get("instance_id"):
        identity["stamp_pid"] = stamp.get("pid")
        identity["started_at"] = stamp.get("started_at")
    return identity


def _backend_matches_stamp(health=None):
    """G11 / F52 — is the live backend OUR current runtime?

    A self-consistent stamp is not enough; three independent checks decide:

    1. *version* — the live protocol contract must be the one this watcher
       speaks, so an old build is replaced instead of reused.
    2. *authority* — the live backend must hold the SAME per-launch token
       this supervisor minted, so a restarted watcher's new token can never
       silently mismatch a warm backend.
    3. *ownership* — the /health instance identity must be the stamped one,
       and the stamped pid must be the pid that is actually listening.
    """
    health = _backend_health() if health is None else health
    if not health or health.get("service") != "jarvis-backend":
        return False
    if health.get("protocol") != runtime_identity.protocol_version():
        return False
    if health.get("auth") != _expected_token_fingerprint():
        return False
    stamp = runtime_identity.read_instance_file("backend")
    if not stamp:
        return False
    if stamp.get("instance_id") != health.get("instance_id"):
        return False
    live_pid = health.get("pid")
    if isinstance(live_pid, int) and stamp.get("pid") != live_pid:
        return False
    return True


def _jarvis_backend_pid(health=None):
    """The pid attributable to the live jarvis backend, or None (F52).

    Attribution comes from the service's own identity report (or the matching
    instance stamp), never from "something is listening on the port": Windows
    can double-bind a port (SO_REUSEADDR), so a port scan alone says nothing
    about which pid is ours, and killing the wrong one would take down a
    foreign process.
    """
    health = _backend_health() if health is None else health
    if not health or health.get("service") != "jarvis-backend":
        return None
    pid = health.get("pid")
    if isinstance(pid, int) and pid > 0:
        return pid
    stamp = runtime_identity.read_instance_file("backend")
    if stamp and stamp.get("instance_id") == health.get("instance_id"):
        pid = stamp.get("pid")
        if isinstance(pid, int) and pid > 0:
            return pid
    owned = _owned_entry("backend")
    if owned and owned.get("pid"):
        return owned["pid"]
    return None


def _kill_backend_listener_if_owned(health=None):
    """Kill the live jarvis backend ONLY when its pid is attributable.

    F52 — foreign listeners survive: a process that merely shares
    BACKEND_PORT is never killed for listening there.
    """
    pid = _jarvis_backend_pid(health)
    if not pid:
        return False
    if pid not in _pids_on_port(BACKEND_PORT) and pid not in _owned_pids():
        return False
    _taskkill_pid(pid)
    return True


def _wait_for_port_release(port, timeout_seconds=5.0):
    """Give a just-killed listener a bounded moment to release *port*."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _pids_on_port(port):
            return True
        time.sleep(0.2)
    return not _pids_on_port(port)


def _invalidate_backend_approvals():
    """G11 / F52 — invalidate stale approvals BEFORE any safe recovery.

    Called when this supervisor is about to replace or tear down a live
    backend: pending consent must never survive into a new worker. Best
    effort over the authed control contract; a dead backend needs no call
    (its in-memory approvals die with it).
    """
    try:
        request = Request(
            f"http://127.0.0.1:{BACKEND_PORT}/approvals/reset",
            data=b"{}",
            method="POST",
            headers={
                "Content-Type": "application/json",
                local_auth.HEADER: LOCAL_TOKEN,
            },
        )
        with urlopen(request, timeout=1.0):
            pass
    except Exception:
        pass


def _backend_has_research_endpoint():
    """True when the running backend knows POST/GET /research-result (new code).

    A backend launched before the research overlay existed answers 404 here;
    the watcher then kills and respawns it so reports can actually flow.

    F51: this is a non-public endpoint, so the probe must carry the launch
    token — without it the fail-closed 401 reads as "endpoint missing" and
    the watcher would kill/respawn a perfectly good backend in a loop.
    """
    try:
        request = Request(
            f"http://127.0.0.1:{BACKEND_PORT}/research-result",
            headers={local_auth.HEADER: LOCAL_TOKEN},
        )
        with urlopen(request, timeout=0.8) as response:
            return response.status == 200
    except Exception:
        return False


def wait_for_backend_ready(timeout_seconds=45, proc=None):
    """Wait until the backend answers /health, or prove it never will.

    F55 — the health probe decides. ``proc`` is only a hint: this venv's
    python.exe is a shim that can exit while the interpreter it started keeps
    serving, so an exit is not proof of death and gets a short grace instead
    of an immediate ``False`` (which used to kill a live backend and spend two
    more 45s attempts on it).
    """
    deadline = time.monotonic() + timeout_seconds
    attempts = 0
    exited_at = None
    while time.monotonic() < deadline:
        health = _backend_health()
        if health and health.get("service") == "jarvis-backend":
            return True

        now = time.monotonic()
        if proc is not None and proc.poll() is not None:
            if exited_at is None:
                exited_at = now
            if now - exited_at >= BACKEND_EXIT_GRACE:
                return False

        attempts += 1
        if attempts % 10 == 1:
            print(f"[WATCHER] Waiting for backend... ({int(deadline - time.monotonic())}s left)")
        time.sleep(0.4)
    return False


def _pids_on_port(port):
    pids = set()
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
        )
    except Exception:
        return pids

    marker = f":{port}"
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if marker not in parts[1] or parts[-2].upper() != "LISTENING":
            continue
        try:
            pids.add(int(parts[-1]))
        except ValueError:
            continue
    return pids


def _stop_backend_port_if_jarvis():
    """Fallback teardown for a jarvis backend we did not spawn in this call.

    F52 — attribution scoped: the port scan is used only to CONFIRM that the
    attributable pid really is the listener. A foreign process sharing the
    port (or an unattributable listener) survives untouched.
    """
    return _kill_backend_listener_if_owned()


def _stop_opencode_server_if_jarvis():
    """Kill the persistent opencode serve daemon on full teardown.

    The warm opencode session (visible console on port 9560) is a child of
    the watcher; when the whole stack shuts down it must go with it, or a
    zombie daemon keeps the model resident (and a console window open)
    forever. Warm shutdowns (keep_backend=True) intentionally leave it
    running so the next wake is instant.
    """
    try:
        from backend.services.opencode_client import shutdown_opencode_server

        shutdown_opencode_server()
    except Exception as exc:
        print(f"[WATCHER] opencode serve daemon teardown error: {exc}")


def stop_runtime(stop_electron=True, keep_backend=False, stop_daemons=None):
    """Tear down the launched stack (F52 — two explicit process sets).

    ``keep_backend=True`` is the WARM SLEEP set (``SLEEP_STOP_ROLES``): only
    the voice worker and the UI go down, while the backend and the resident
    daemons survive so the next wake is instant.

    ``keep_backend=False`` is the FULL SHUTDOWN set (``SHUTDOWN_STOP_ROLES``):
    every process this supervisor owns — backend, voice, UI and the whisper
    daemon — is stopped, the task-engine daemons (opencode serve / brave MCP)
    are shut down, and the instance stamps of the workers we just retired are
    cleared under an ownership check. Nothing is stopped merely for holding a
    port.

    *stop_daemons* overrides the resident-daemon part of that decision; it
    defaults to ``not keep_backend``. An ABORTED LAUNCH passes False: a failed
    boot must not cost the warm whisper model or the task-engine daemons.
    """
    global backend_proc, voice_proc, electron_proc, jarvis_running
    global whisper_daemon_proc, whisper_daemon_ok

    if stop_daemons is None:
        stop_daemons = not keep_backend

    # F52 — a teardown must never race a crash-recovery respawn.
    stop_worker_supervisor()

    with runtime_lock:
        voice = voice_proc
        electron = electron_proc if stop_electron else None
        backend = None if keep_backend else backend_proc
        voice_proc = None
        if not keep_backend:
            backend_proc = None
        if stop_electron:
            electron_proc = None
        jarvis_running = False

    voice_owned = _forget_owned("voice")
    electron_owned = _forget_owned("electron") if stop_electron else None
    backend_owned = _forget_owned("backend") if not keep_backend else None

    _stop_handle(voice, (voice_owned or {}).get("pid"))
    _stop_handle(electron, (electron_owned or {}).get("pid"))

    backend_pid = (backend_owned or {}).get("pid") or getattr(backend, "pid", None)
    if backend is not None or backend_owned is not None:
        # G11 / F52 — invalidate stale approvals and let the backend interrupt
        # its live jobs BEFORE the worker goes away: pending consent and
        # half-finished work must never survive into a replacement worker.
        _invalidate_backend_approvals()
        _stop_handle(backend, (backend_owned or {}).get("pid"))

    if keep_backend:
        # Warm sleep: the backend AND the whisper daemon stay resident.
        # Close the visible activity-tail console but keep the warm fast
        # stack - the opencode serve daemon (port 9560) and the brave MCP
        # daemon stay alive; the tail respawns on the next launch.
        try:
            from backend.services.opencode_client import close_activity_tail

            close_activity_tail()
        except Exception as exc:
            print(f"[WATCHER] activity tail close error: {exc}")
        return

    # Full shutdown: the remaining owned daemons go with it.
    if stop_daemons:
        if "whisper_daemon" in SHUTDOWN_STOP_ROLES:
            _stop_whisper_daemon()
            whisper_daemon_proc = None
            whisper_daemon_ok = False
        # Fallback for old/orphaned Jarvis backend instances (attribution
        # scoped, so a foreign listener on the port survives).
        _stop_backend_port_if_jarvis()
        # The warm opencode serve daemon must not survive a full shutdown.
        _stop_opencode_server_if_jarvis()
    # Ownership-checked stamp cleanup: only the stamps of the workers this
    # supervisor just retired are removed, so an OLD exit can never delete a
    # NEW stamp.
    if backend_pid:
        runtime_identity.clear_instance_file("backend", owned_pid=backend_pid)


class WatcherControlHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *args):
        return

    def do_GET(self):
        if self.path != "/health":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        self._send_json(
            200,
            {
                "ok": True,
                "service": "jarvis-watcher",
                "running": jarvis_running,
            },
        )

    def do_POST(self):
        # G11 / F51 — watcher control endpoints are command endpoints too:
        # any POST must carry the per-launch token the watcher itself minted
        # and injected into its children. (GET /health stays open so a
        # freshly-spawned child can probe liveness pre-injection.)
        if LOCAL_TOKEN and self.headers.get(local_auth.HEADER, "") != LOCAL_TOKEN:
            self._send_json(401, {"ok": False, "error": "unauthorized watcher control"})
            return
        if self.path == "/stop":
            # Warm-sleep semantics: stop the UI/voice but keep the backend
            # warm so the next wake is fast. (Cancel-task is a DISTINCT
            # action: POST /task/stop on the backend — it never stops the
            # stack.)
            threading.Thread(
                target=lambda: stop_runtime(stop_electron=True, keep_backend=True),
                daemon=True,
            ).start()
            self._send_json(200, {"ok": True, "mode": "warm-sleep"})
            return
        if self.path == "/shutdown":
            # Full-shutdown semantics: everything goes down, including the
            # warm backend and the daemons. Distinct from /stop on purpose
            # (F52: users get warm-sleep vs full-shutdown, not one blur).
            threading.Thread(
                target=lambda: stop_runtime(stop_electron=True, keep_backend=False),
                daemon=True,
            ).start()
            self._send_json(200, {"ok": True, "mode": "full-shutdown"})
            return
        self._send_json(404, {"ok": False, "error": "not found"})


def start_control_server():
    global control_server
    if control_server is not None:
        return
    try:
        control_server = ThreadingHTTPServer(
            ("127.0.0.1", WATCHER_CONTROL_PORT),
            WatcherControlHandler,
        )
    except OSError as exc:
        print(f"[WATCHER] Control server unavailable: {exc}")
        return
    threading.Thread(target=control_server.serve_forever, daemon=True).start()
    print(f"[WATCHER] Control server on 127.0.0.1:{WATCHER_CONTROL_PORT}")


def _spawn_with_retry(args, label, **kwargs):
    """Spawn *args* with retries; never raise.

    Under low RAM (this machine routinely sits around 1-2GB free) Windows
    CreateProcess can transiently fail with "Access is denied" / WinError 5
    even though nothing is actually wrong with the command — the same flake
    that once showed as venv "Unable to create process". A 3x retry with a
    pause rides through it; a real error still logs a clear message.
    """
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            return subprocess.Popen(args, **kwargs)
        except OSError as exc:
            print(
                f"[WATCHER] {label} spawn failed (attempt {attempt}/{max_attempts}): "
                f"{type(exc).__name__} {exc}"
            )
            if attempt < max_attempts:
                time.sleep(2)
    print(
        f"[WATCHER] {label} could not be started after {max_attempts} attempts.\n"
        "[WATCHER] This is usually system memory pressure (RAM nearly "
        "exhausted) or antivirus scanning. Close heavy apps and retry."
    )
    return None


def _spawn_backend(child_env, attempts=BACKEND_SPAWN_ATTEMPTS):
    """Boot the backend and wait until it answers; the supervisor's respawn spec.

    F52 — the initial retry budget and a later crash recovery share ONE
    routine, so a worker that comes back after a crash is validated exactly
    like the first attempt (readiness + attributed cleanup) instead of being
    trusted merely because ``Popen`` returned. Returns a Popen or None.
    """
    for attempt in range(1, attempts + 1):
        print(
            f"[WATCHER] Booting backend (attempt {attempt}/{attempts})"
            f" - python: {VENV_PY}"
        )
        try:
            proc = subprocess.Popen(
                [
                    str(VENV_PY),
                    "-u",
                    "-m",
                    "uvicorn",
                    "backend.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(BACKEND_PORT),
                    # [P1-14] no access log: it is synchronous I/O on the event
                    # loop, adding latency to every request including barge-in.
                    "--no-access-log",
                ],
                cwd=str(BASE_DIR),
                env=child_env,
                creationflags=_child_creationflags(),
            )
        except OSError as exc:
            proc = None
            print(f"[WATCHER] Backend spawn failed (OSError): {exc}")

        if proc is not None:
            if wait_for_backend_ready(
                proc=proc, timeout_seconds=BACKEND_READY_TIMEOUT_SECONDS
            ):
                return proc
            print(
                f"[WATCHER] Backend attempt {attempt} did not become ready - "
                "killing and retrying."
            )
            # The pid is OUR OWN just-spawned child, so a tree kill is always
            # legitimate; then any attributed listener leftover is cleaned up.
            _taskkill_pid(getattr(proc, "pid", None))
            try:
                proc.kill()
            except OSError:
                pass
            _kill_backend_listener_if_owned()

        if attempt < attempts:
            time.sleep(2)

    print(
        f"[WATCHER] Backend did not become ready after {attempts} attempts.\n"
        "[WATCHER] Likely system load (low RAM / antivirus / OneDrive "
        "sync churn). Close heavy apps and retry run_jarvis.bat."
    )
    return None


def _warn_if_low_memory():
    try:
        import ctypes

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            free_gb = status.ullAvailPhys / (1024 ** 3)
            if free_gb < 2.0:
                print(
                    f"[WATCHER] WARNING: only {free_gb:.1f} GB RAM free - "
                    "spawns may transiently fail. Close heavy apps if Jarvis "
                    "struggles to launch."
                )
            return free_gb
    except Exception:
        pass
    return None


def launch_jarvis():
    global jarvis_running, backend_proc, voice_proc, electron_proc

    print("[WATCHER] Wake word detected - launching Jarvis...")

    child_env = os.environ.copy()
    child_env["JARVIS_WATCHER_CONTROL_PORT"] = str(WATCHER_CONTROL_PORT)
    # G11 / F51 — every child this supervisor owns carries the per-launch
    # local command token (backend enforces it; voice and electron forward it).
    child_env["JARVIS_LOCAL_TOKEN"] = LOCAL_TOKEN

    # Reuse a healthy already-running backend instead of booting a new one.
    # G11 / F52 — "current" is verified by IDENTITY (version + the per-launch
    # token authority + the instance stamp that matches the listening pid),
    # never by the mere existence of a listener on the port.
    kill_stale_backend = False
    health = _backend_health()
    if health and health.get("service") == "jarvis-backend":
        if _backend_matches_stamp(health) and _backend_has_research_endpoint():
            print("[WATCHER] Warm backend already running - reusing it.")
            # F52 — retain the ADOPTED handle bound to its verified identity.
            # We did not spawn it, but we know exactly which pid is ours, so
            # teardown and crash recovery act on that pid (not on a port
            # scan). Voice mode still wants a shutdown fallback pid too.
            backend_proc = None
            backend_pid = _jarvis_backend_pid(health)
            _register_owned(
                "backend",
                None,
                pid=backend_pid,
                identity=_backend_identity(health),
                respawn=lambda: _spawn_backend(child_env),
                adopted=True,
            )
        else:
            # Stale backend (version / token / identity mismatch) — it can
            # never be trusted as ours. Invalidate its approvals and interrupt
            # its jobs FIRST (F52), then stop the attributed pid and boot a
            # fresh one. A listener we cannot attribute is never killed.
            print(
                "[WATCHER] Warm backend is stale (version/token/identity/"
                "endpoint mismatch) - restarting it."
            )
            _invalidate_backend_approvals()
            _kill_backend_listener_if_owned(health)
            _wait_for_port_release(BACKEND_PORT)
            if _pids_on_port(BACKEND_PORT):
                print(
                    f"[WATCHER] Port {BACKEND_PORT} is still held by a listener "
                    "that is not attributable to this Jarvis runtime; refusing "
                    "to kill it (F52: foreign listeners survive) - launch "
                    "cancelled."
                )
                return
            kill_stale_backend = True
            health = None

    if kill_stale_backend or (health is None and not _pids_on_port(BACKEND_PORT)):
        backend_pid = None
        # One budgeted routine for the first boot AND later crash recovery:
        # a respawned worker is validated exactly like a first launch.
        spawn_backend = lambda: _spawn_backend(child_env)
        backend_proc = spawn_backend()
        if backend_proc is None:
            print(
                "[WATCHER] Backend could not be started after "
                f"{BACKEND_SPAWN_ATTEMPTS} attempts.\n"
                "[WATCHER] This is usually system memory pressure (RAM almost "
                "exhausted, ~1GB free) or antivirus scanning. Close heavy apps "
                "and retry run_jarvis.bat."
            )
            # Aborted launch, not a shutdown: keep the resident daemons warm.
            stop_runtime(stop_electron=False, stop_daemons=False)
            return
        # Creation-identity-bound handle: the pid we spawned + the identity
        # the child reported, kept together for supervision and teardown.
        _register_owned(
            "backend",
            backend_proc,
            identity=_backend_identity(),
            respawn=spawn_backend,
        )
    elif not (health and health.get("service") == "jarvis-backend"):
        print(
            f"[WATCHER] Port {BACKEND_PORT} is already in use by a non-Jarvis "
            "process; launch cancelled."
        )
        return

    voice_env = child_env.copy()
    if backend_pid:
        voice_env["JARVIS_BACKEND_PID"] = str(backend_pid)

    _warn_if_low_memory()

    voice_args = [str(VENV_PY), "-u", "-m", "backend.voice_mode"]
    voice_kwargs = dict(
        cwd=str(BASE_DIR),
        env=voice_env,
        creationflags=_child_creationflags(),
    )
    respawn_voice = lambda: _spawn_with_retry(voice_args, label="Voice", **voice_kwargs)
    voice_proc = respawn_voice()
    if voice_proc is not None:
        # F52 — the voice worker is in BOTH process sets (sleep and shutdown)
        # and is supervised: a crash while the UI lives is restarted within a
        # bounded budget.
        _register_owned("voice", voice_proc, respawn=respawn_voice)

    electron_env = os.environ.copy()
    electron_env["JARVIS_EXTERNAL_RUNTIME"] = "1"
    electron_env["JARVIS_WATCHER_CONTROL_PORT"] = str(WATCHER_CONTROL_PORT)
    # G11 / F51 — the renderer host forwards the same per-launch token.
    electron_env["JARVIS_LOCAL_TOKEN"] = LOCAL_TOKEN

    electron_proc = _spawn_with_retry(
        "npm start",
        label="Electron",
        shell=True,
        cwd=str(BASE_DIR),
        env=electron_env,
    )
    if electron_proc is not None:
        # The UI handle is the "UI is alive" signal for crash recovery: while
        # it runs, crashed workers are restarted; when it exits, that is a
        # close, not a crash.
        _register_owned("electron", electron_proc)

    if voice_proc is None or electron_proc is None:
        print(
            "[WATCHER] Some runtime processes failed to spawn - Jarvis may be "
            "partially up. Free up RAM and run run_jarvis.bat again if needed."
        )

    # Respawn the visible activity-tail console if a warm stop closed it.
    # Idempotent: an alive serve daemon is reused and an existing tail is
    # left alone; only the missing tail gets a fresh console. The native
    # browser-agent engine spawns no opencode serve daemon.
    try:
        from backend import config
        from backend.services.opencode_client import ensure_activity_tail
        from backend.services.opencode_client import kill_stale_opencode_server
        from backend.services.opencode_client import ensure_opencode_server

        if config.TASK_ENGINE == "browser_agent":
            ensure_activity_tail()
            kill_stale_opencode_server()
        else:
            ensure_opencode_server()
    except Exception as exc:
        print(f"[WATCHER] opencode tail respawn failed: {exc}")

    jarvis_running = True
    # F52 — supervise the owned workers from here on: a crash while the UI
    # lives is recovered within a bounded budget instead of leaving the stack
    # half-dead until the next wake.
    start_worker_supervisor()
    # F55 — the runtime is up, so the whisper model may load now. Deferring it
    # to this point keeps the boot window free of a ~1.5 GB load that used to
    # run beside the backend's own model warm-up and starve it on a machine
    # with ~1 GB RAM free (three failed backend attempts, no Jarvis window).
    warm_whisper_daemon_model()
    print("[OK] Jarvis launched - watcher paused\n")


def is_backend_running():
    return bool(_pids_on_port(BACKEND_PORT))


def wait_for_shutdown():
    global jarvis_running, electron_proc

    print("[WATCHER] Paused - waiting for Jarvis to shut down...")

    if electron_proc:
        electron_proc.wait()

    # If the UI was closed directly, keep the backend alive so the models
    # stay warm and the next wake is fast. Only voice/electron are stopped.
    stop_runtime(stop_electron=False, keep_backend=True)

    jarvis_running = False
    electron_proc = None

    print("\n[WATCHER] Jarvis closed - models kept warm. Say 'wake up jarvis' to relaunch\n")


def _cleanup_watcher_exit():
    """Full teardown when the watcher process itself exits.

    Closing the watcher console is the only "shut everything down"
    action now — it also kills any warm backend left running so no
    orphaned Python processes survive.
    """
    stop_runtime(stop_electron=True, keep_backend=False)


atexit.register(_cleanup_watcher_exit)


def warm_whisper_daemon_model():
    """Ask the whisper daemon to load its model, off the caller's thread (F55).

    Best effort and never raising: the warm-up is an optimisation (without it
    the first wake pays for the load), and it must never hold up the launch it
    runs after. Returns the worker thread.
    """

    def _request_warmup():
        try:
            request = Request(
                f"{WHISPER_DAEMON_URL}/warm",
                data=b"",
                headers={"Content-Type": "application/octet-stream"},
                method="POST",
            )
            with urlopen(request, timeout=5.0) as response:
                response.read()
            print("[WATCHER] Whisper model warm-up requested.")
        except Exception as exc:
            print(f"[WATCHER] Whisper model warm-up request failed: {exc}")

    thread = threading.Thread(
        target=_request_warmup, name="whisper-model-warmup", daemon=True
    )
    thread.start()
    return thread


def warm_browser_task_engine():
    """Warm the browser-task daemon OFF the boot path (F54).

    Returns the worker thread so a caller — or a test — can join it. The
    readiness wait belongs to that thread: run inline it held the visible boot
    for its entire timeout whenever the daemon was sick (measured 10.4s
    between the activity console and the next window, which is precisely the
    delay users reported). The starter is idempotent and lock-guarded, so a
    task that needs the daemon calls it again and now fails fast with the real
    reason instead of a connection-refused later.
    """
    from backend.services.opencode_client import ensure_brave_mcp_daemon

    thread = threading.Thread(
        target=ensure_brave_mcp_daemon,
        name="brave-mcp-warmup",
        daemon=True,
    )
    thread.start()
    return thread


def main():
    if not ensure_mic():
        print("[WATCHER] Microphone unavailable this run - continuing anyway")

    launch_on_start = "--launch" in sys.argv

    start_control_server()

    if not ensure_whisper_daemon():
        _load_in_process_whisper()

    # Keep the task engine warm so hand-offs are instant. The opencode
    # engine runs a persistent serve daemon (hidden) + the visible activity
    # tail; the native browser-agent engine skips the serve daemon and only
    # needs the tail plus the brave MCP daemon. Best effort either way.
    try:
        from backend import config
        from backend.services.opencode_client import ensure_activity_tail
        from backend.services.opencode_client import kill_stale_opencode_server
        from backend.services.opencode_client import ensure_opencode_server

        if config.TASK_ENGINE == "browser_agent":
            ensure_activity_tail()
            kill_stale_opencode_server()
        else:
            ensure_opencode_server()
        warm_browser_task_engine()
    except Exception as exc:
        print(f"[WATCHER] opencode serve daemon not started: {exc}")

    if launch_on_start:
        print("[WATCHER] --launch flag set - starting Jarvis now (no wake word needed).")
        launch_jarvis()
    else:
        recalibrate_watcher("startup", force=True)
        print(
            "[WATCHER] Active - say 'wake up jarvis' / 'utho jarvis' anytime | "
            f"threshold: {WATCHER_ENERGY_THRESHOLD}\n"
        )

    while True:
        if jarvis_running:
            wait_for_shutdown()
            continue

        candidates, wake_match = listen_once()
        if wake_match:
            print(f"[WATCHER] Wake match: {wake_match}")
            # F36: extract the command tail BEFORE launch (candidates live
            # in this scope) and forward it ONCE the stack accepts /ask —
            # "wake up jarvis and search for cats" now does both parts. The
            # literal tail is preserved byte-exact and forwarded through an
            # authenticated, identified request after readiness, and a
            # request id is minted per wake so launch/disconnection retries
            # can never replay the same spoken command.
            from backend.services import wake_engine

            command = wake_engine.extract_command(wake_match, candidates)
            request_id = "wake-%d-%d" % (int(time.time() * 1000), os.getpid())
            launch_jarvis()
            if command:
                wake_engine.forward_wake_command(
                    command, request_id=request_id, wait_ready=True)
            continue

        if not candidates:
            continue


if __name__ == "__main__":
    main()
