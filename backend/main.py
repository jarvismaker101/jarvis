import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.config import validate_environment
from backend.services.voice import warm_up_selected_tts
from backend.services import local_auth
from backend.services import runtime_identity
from backend.services.ollama_client import warm_up_ollama

app = FastAPI()

# G11 / F51 — RESTRICTED origins. This used to be ``allow_origins=["*"]``
# with credentials, which let ANY web page a browser visits read responses
# from the loopback control port. Only the Electron renderer origins (a
# file:// page reports ``null``) and loopback dev servers may call in;
# native clients (the supervisor, the voice worker) send no Origin at all.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(local_auth.RENDERER_ORIGINS),
    allow_origin_regex=local_auth.LOOPBACK_ORIGIN_PATTERN,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization",
                   local_auth.HEADER],
)

# G11 / F51 — per-launch local command authentication. When a supervisor
# injected JARVIS_LOCAL_TOKEN into this process, every endpoint except the
# public liveness surface (GET /health) requires it — mutating commands AND
# private reads. Without a token the middleware now FAILS CLOSED; only an
# explicit JARVIS_DEV_MODE=1 declares development mode and opens the surface.
local_auth.install(app)

# All routes (including /ask and /voice-log) live in routes.py
app.include_router(router)


@app.on_event("startup")
def _startup_checks():
    validate_environment()
    # G11 / F52 — durable instance identity: the supervisor matches this
    # stamp against the listening port instead of assuming ownership from a
    # port alone. Best-effort; a stamping failure never blocks startup.
    runtime_identity.write_instance_file(
        role="backend",
        extra={"auth": local_auth.token_fingerprint()},
    )
    threading.Thread(
        target=_warm_up_ollama_background,
        name="ollama-warmup",
        daemon=True,
    ).start()
    threading.Thread(
        target=_warm_up_tts_background,
        name="tts-warmup",
        daemon=True,
    ).start()


def _warm_up_tts_background():
    try:
        warm_up_selected_tts()
    except Exception as exc:
        logging.debug("TTS warm-up skipped: %s", exc)


def _warm_up_ollama_background():
    try:
        warm_up_ollama()
    except Exception as exc:
        logging.debug("Ollama warm-up skipped: %s", exc)


@app.on_event("shutdown")
def _flush_durable_state():
    """P0-11 — a clean shutdown writes out what the background writers hold.

    The debounced history snapshot and the bookkeeping queue are both flushed
    here as well as from their ``atexit`` hooks: depending on how the process
    is stopped, only one of the two paths is guaranteed to run.
    """
    try:
        from backend.core import memory
        memory.flush_history()
    except Exception as exc:
        logging.warning("[MEMORY] shutdown flush failed: %s", exc)
    try:
        from backend.core import memory_store
        if not memory_store.flush_writes(timeout=2.0):
            memory_store.drain_pending_writes()
    except Exception as exc:
        logging.warning("[MEMORY] shutdown write-queue flush failed: %s", exc)
