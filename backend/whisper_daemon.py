"""Persistent faster-whisper daemon for Jarvis.

Loads the local Whisper model once at startup and keeps it resident in
memory/VRAM, serving transcription over a small local HTTP API. The watcher
spawns this process detached on first use and reconnects to it on later
launches, so wake-word listening starts instantly instead of reloading the
model every time the watcher starts.

Endpoints:
  GET  /health      -> {"ok": true, "service": "jarvis-whisper", "pid": ...,
                        "instance_id": ..., "protocol": ..., "model": ...,
                        "device": ...}
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

from faster_whisper import WhisperModel

from backend.services import runtime_identity

PORT = int(os.getenv("JARVIS_WHISPER_PORT", "8767"))
MODEL_SIZE = os.getenv("JARVIS_WHISPER_MODEL", "medium")
INITIAL_PROMPT = "Jarvis, wake up, jervis, utho, jago, chalu"

model = None
device = "cpu"
_lock = threading.Lock()


def load_model():
    global model, device
    print(f"[WHISPER-DAEMON] Loading {MODEL_SIZE} model...")
    try:
        model = WhisperModel(
            MODEL_SIZE, device="cuda", compute_type="float16", local_files_only=True
        )
        device = "cuda"
        print("[WHISPER-DAEMON] Loaded on GPU (CUDA).")
    except Exception as exc:
        print(f"[WHISPER-DAEMON] CUDA load failed: {exc}. Trying CPU fallback...")
        try:
            model = WhisperModel(
                MODEL_SIZE, device="cpu", compute_type="int8", local_files_only=True
            )
            device = "cpu"
            print("[WHISPER-DAEMON] Loaded on CPU.")
        except Exception as exc_cpu:
            print(f"[WHISPER-DAEMON] Model load failed: {exc_cpu}")
            sys.exit(1)


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
            },
        )

    def do_POST(self):
        if self.path != "/transcribe":
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            wav_bytes = self.rfile.read(length)
            if not wav_bytes:
                self._send_json(400, {"ok": False, "error": "empty body"})
                return
            with _lock:
                segments, info = model.transcribe(
                    io.BytesIO(wav_bytes),
                    temperature=0.0,
                    vad_filter=True,
                    condition_on_previous_text=False,
                    initial_prompt=INITIAL_PROMPT,
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


def main():
    load_model()
    # F52 — ownership stamp for this daemon (best effort): the supervisor
    # reads it to attribute the port to the daemon it launched.
    runtime_identity.write_instance_file(
        "whisper_daemon",
        extra={"port": PORT, "auth": "whisper"},
    )
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[WHISPER-DAEMON] Serving on 127.0.0.1:{PORT}")
    try:
        server.serve_forever()
    finally:
        runtime_identity.clear_instance_file("whisper_daemon")


if __name__ == "__main__":
    main()
