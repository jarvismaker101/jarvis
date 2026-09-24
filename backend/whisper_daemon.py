"""Persistent faster-whisper daemon for Jarvis.

Loads the local Whisper model once at startup and keeps it resident in
memory/VRAM, serving transcription over a small local HTTP API. The watcher
spawns this process detached on first use and reconnects to it on later
launches, so wake-word listening starts instantly instead of reloading the
model every time the watcher starts.

F55 — the port is bound BEFORE the model is loaded, and the load itself only
starts when it is asked for:

* A cold start used to hold the port until ``WhisperModel`` finished (10-20s
  on CUDA, minutes when the machine is thrashing). Every supervisor that
  waited for the port therefore looked at a daemon that "was not there" and
  loaded a SECOND copy of a ~1.5 GB model in its own process, which is
  exactly how a slow boot turned into a stuck one on a memory-tight machine.
* ``/health`` now answers within milliseconds and reports ``ready`` /
  ``loading`` honestly, so the supervisor adopts this daemon immediately and
  never duplicates the model.
* The load is *on demand* — ``POST /warm`` (sent by the supervisor once the
  runtime is up) or the first ``/transcribe``. Loading it at daemon startup
  instead put a ~1.5 GB spike next to the backend's own model warm-up; on a
  machine with ~1 GB free the backend then never became ready and the
  supervisor killed and retried it three times, leaving nothing but console
  windows on screen.
* ``/transcribe`` waits for the model (bounded) instead of failing, so the
  first utterance after a cold boot still transcribes.
* The listener binds exclusively (no ``SO_REUSEADDR``): on Windows that flag
  lets a second daemon bind the same port and silently steal connections from
  the live one, which looks healthy while nothing is served.

Endpoints:
  GET  /health      -> {"ok": true, "service": "jarvis-whisper", "pid": ...,
                        "instance_id": ..., "protocol": ..., "model": ...,
                        "device": ..., "ready": bool, "loading": bool,
                        "error": str|None}
  POST /warm        -> start loading the model now (idempotent)
  POST /transcribe  -> body = wav bytes; returns
                       {"ok": true, "text": ..., "language": ..., "language_probability": ...}

F52 — the daemon stamps ``data/runtime/whisper_daemon-instance.json`` and
reports its pid so the supervisor can attribute the listener to a process it
owns: a FULL shutdown then leaves no unintended daemon behind, while a warm
sleep still keeps the resident model for the next wake.
"""

import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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

from backend.services import runtime_identity

PORT = int(os.getenv("JARVIS_WHISPER_PORT", "8767"))
MODEL_SIZE = os.getenv("JARVIS_WHISPER_MODEL", "medium")
# Wake-word biasing prompt. ONLY applied for wake matching — the caller asks
# for it with ``X-Jarvis-Purpose: wake`` (watcher). Conversation transcription
# runs WITHOUT it on purpose: on noise / TTS-echo audio the model hallucinates
# and echoes these prompt tokens verbatim ("jarvis, wake up, jervis, utho,
# jago, chalu"), which then reached the brain as user speech.
INITIAL_PROMPT = "Jarvis, wake up, jervis, utho, jago, chalu"
# Bounded wait for a still-loading model on the transcription path (F55).
TRANSCRIBE_MODEL_WAIT = float(os.getenv("JARVIS_WHISPER_TRANSCRIBE_WAIT", "20"))

model = None
device = "unknown"
_model_error = None
# Set when the load ATTEMPT has finished (success or failure), never before.
_model_load_finished = threading.Event()
_model_load_started = False
_load_lock = threading.Lock()
_lock = threading.Lock()


def ensure_model_loading():
    """Start the model load once, on demand (F55).

    Returns True when this call started it. Deferring the load keeps the boot
    window free: the model is warmed by the supervisor once the runtime is up,
    or by the first transcription that needs it.
    """
    global _model_load_started

    with _load_lock:
        if _model_load_started:
            return False
        _model_load_started = True
        threading.Thread(
            target=load_model, name="whisper-model-load", daemon=True
        ).start()
        return True


def load_model():
    """Load the model into memory. Runs on a background thread (F55)."""
    global model, device, _model_error

    print(f"[WHISPER-DAEMON] Loading {MODEL_SIZE} model...", flush=True)
    try:
        # Imported here, not at module scope: the heavy CUDA/ctranslate2 import
        # must not delay the port bind that makes this daemon adoptable.
        from faster_whisper import WhisperModel

        try:
            candidate = WhisperModel(
                MODEL_SIZE, device="cuda", compute_type="float16", local_files_only=True
            )
            device = "cuda"
            print("[WHISPER-DAEMON] Loaded on GPU (CUDA).", flush=True)
        except Exception as exc:
            print(f"[WHISPER-DAEMON] CUDA load failed: {exc}. Trying CPU fallback...", flush=True)
            candidate = WhisperModel(
                MODEL_SIZE, device="cpu", compute_type="int8", local_files_only=True
            )
            device = "cpu"
            print("[WHISPER-DAEMON] Loaded on CPU.", flush=True)
        model = candidate
    except Exception as exc_all:
        _model_error = str(exc_all)
        print(f"[WHISPER-DAEMON] Model load failed: {exc_all}", flush=True)
    finally:
        _model_load_finished.set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, _format, *args):
        return

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        # F52 — identity surface: which daemon (instance + pid) owns this
        # port, so a supervisor can attribute the listener to a process it
        # started instead of killing "whatever is listening".
        # F55 — ``ok`` means "this daemon is alive and bound"; ``ready`` says
        # whether the model can transcribe yet. A supervisor adopts on ``ok``
        # so a loading model is never duplicated in another process.
        self._send_json(
            200,
            {
                "ok": True,
                "service": "jarvis-whisper",
                "pid": os.getpid(),
                "instance_id": runtime_identity.instance_id(),
                "protocol": runtime_identity.protocol_version(),
                "build": runtime_identity.build_id(),
                "model": MODEL_SIZE,
                "device": device,
                "ready": model is not None,
                "loading": _model_load_started and not _model_load_finished.is_set(),
                "error": _model_error,
            },
        )

    def do_POST(self):
        if self.path == "/warm":
            # F55 — the supervisor asks for the warm-up once the runtime is
            # up, so the load never competes with the backend's own startup.
            started = ensure_model_loading()
            self._send_json(
                200,
                {
                    "ok": True,
                    "started": started,
                    "ready": model is not None,
                    "loading": _model_load_started and not _model_load_finished.is_set(),
                },
            )
            return
        if self.path != "/transcribe":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            wav_bytes = self.rfile.read(length)
            if not wav_bytes:
                self._send_json(400, {"ok": False, "error": "empty body"})
                return
            # F55 — a wake that arrives before the warm-up still works: the
            # load starts here and the request waits for it (bounded).
            ensure_model_loading()
            if not _model_load_finished.wait(TRANSCRIBE_MODEL_WAIT):
                self._send_json(503, {"ok": False, "error": "model still loading"})
                return
            if model is None:
                self._send_json(
                    503,
                    {"ok": False, "error": _model_error or "model unavailable"},
                )
                return
            with _lock:
                wake_bias = (
                    (self.headers.get("X-Jarvis-Purpose") or "").strip().lower()
                    == "wake"
                )
                segments, info = model.transcribe(
                    io.BytesIO(wav_bytes),
                    temperature=0.0,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    initial_prompt=INITIAL_PROMPT if wake_bias else None,
                )
                text = "".join(seg.text for seg in segments).strip()
            self._send_json(
                200,
                {
                    "ok": True,
                    "text": text,
                    "language": info.language,
                    "language_probability": round(float(info.language_probability or 0.0), 4),
                },
            )
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})


class _ExclusiveServer(ThreadingHTTPServer):
    """HTTP server that refuses to share its port (F55).

    ``ThreadingHTTPServer`` defaults to ``allow_reuse_address = True``. On
    Windows that is not a mere "reuse a TIME_WAIT port": a second daemon can
    bind an already-listening port, after which connections land on whichever
    socket the stack picks. Two "healthy" daemons then serve one port and the
    adopted one silently stops receiving requests.
    """

    allow_reuse_address = False
    daemon_threads = True


def build_server(port, bind_retry_seconds=6.0):
    """Bind ``port`` exclusively, retrying briefly while it is still busy."""
    deadline = time.time() + max(0.0, bind_retry_seconds)
    while True:
        try:
            return _ExclusiveServer(("127.0.0.1", port), Handler)
        except OSError as exc:
            if time.time() >= deadline:
                raise
            print(f"[WHISPER-DAEMON] Port {port} not free yet ({exc}); retrying...", flush=True)
            time.sleep(0.5)


def main():
    server = build_server(PORT)
    # F52 — ownership stamp for this daemon (best effort): the supervisor
    # reads it to attribute the port to the daemon it launched.
    runtime_identity.write_instance_file(
        "whisper_daemon",
        extra={"port": PORT, "auth": "whisper"},
    )
    print(f"[WHISPER-DAEMON] Serving on 127.0.0.1:{PORT} (model loads on demand)", flush=True)
    # F55 — no model load here: the supervisor warms it once the runtime is
    # up (POST /warm), and a wake that arrives first triggers it itself.
    try:
        server.serve_forever()
    finally:
        runtime_identity.clear_instance_file("whisper_daemon")


if __name__ == "__main__":
    main()