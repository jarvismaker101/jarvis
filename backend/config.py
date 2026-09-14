import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH, override=True)


def _force_utf8_stdio():
    """Force UTF-8 stdio so emoji prints (🟢, 🤖, ✅…) never crash a process.

    Voice mode, the watcher and the backend print emoji throughout. When
    stdout/stderr is a cp1252 pipe (hidden child processes), those prints
    raise ``UnicodeEncodeError`` and can silently kill the voice process at
    startup. Reconfiguring to UTF-8 with errors="replace" makes every prior
    stream safe; on consoles that can't render the glyph, we get mojibake
    instead of a crash.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdio()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
FISH_API_KEY = os.getenv("FISH_API_KEY")
FISH_MODEL = os.getenv("FISH_MODEL", "s2.1-pro-free")
FISH_REFERENCE_ID = os.getenv("FISH_REFERENCE_ID")
FISH_VOLUME_BOOST_DB = float(os.getenv("JARVIS_FISH_TTS_VOLUME_BOOST_DB", "6"))
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
CLINE_API_KEY = os.getenv("CLINE_API_KEY")
FIREWORKS_MODEL = os.getenv(
    "FIREWORKS_MODEL",
    "accounts/fireworks/models/deepseek-v4-flash-0731",
)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
FIREWORKS_API_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
CLINE_API_URL = "https://api.cline.bot/api/v1/chat/completions"
BACKEND_PORT = int(os.getenv("JARVIS_BACKEND_PORT", "9999"))

# ── Task engine ──
# Which agent executes hand-off tasks: the native browser agent (default,
# strong model driving the brave MCP daemon) or the opencode CLI.
TASK_ENGINE = os.getenv("JARVIS_TASK_ENGINE", "browser_agent")
# ── G8 orchestrator migration mode (F02) ──
# "legacy" (default): the classifier/racer routing runs exactly as before.
# "orchestrator": the native tool-use orchestrator is attempted FIRST for
# non-gated messages; on any decline/failure the legacy routing (and every
# deterministic safety net) still runs. The classifier is retired only after
# orchestrator parity — never by default.
ORCHESTRATOR_MODE = os.getenv("JARVIS_ORCHESTRATOR_MODE", "legacy")

# ── G9 memory & continuity (F06/F07/F10/F09) ──
# One SQLite facts/entities/events/commitments/skills store. Every call is
# a safe no-op when disabled; an empty store changes zero prompts.
MEMORY_ENABLED = os.getenv("JARVIS_MEMORY_ENABLED", "1") != "0"
MEMORY_DB_PATH = os.getenv("JARVIS_MEMORY_DB") or str(
    BASE_DIR / "data" / "jarvis_memory.db"
)

# ── G10 voice runtime (F31/F32/F33/F34/F36) ──
# Single audio actor owns playback; AEC reference path; local Whisper in
# wake/conversation modes; wake detection separated from full transcription.
# Every knob is read at import in the owning module; hardware/model-dependent
# pieces degrade gracefully (see the audit [ASSUMPTION] notes).
AEC_ENABLED = os.getenv("JARVIS_AEC_ENABLED", "1") != "0"
# [F33] The API process voices replies while the voice process listens, so the
# AEC reference must cross the process boundary. The voice side asks the API
# for the rendered PCM that overlaps its mic window (see /aec/reference);
# disable with JARVIS_AEC_REMOTE=0 for an isolated/local-only setup.
AEC_REMOTE_ENABLED = os.getenv("JARVIS_AEC_REMOTE", "1") != "0"
AEC_REMOTE_TIMEOUT = float(os.getenv("JARVIS_AEC_REMOTE_TIMEOUT", "0.35"))
WAKE_ENGINE = os.getenv("JARVIS_WAKE_ENGINE", "auto")
WAKE_MODELS_DIR = os.getenv("JARVIS_WAKE_MODELS_DIR", "")
WAKE_ONLINE_VERIFY = os.getenv("JARVIS_WAKE_ONLINE_VERIFY", "0") != "0"
WAKE_PRE_ROLL_SECONDS = float(os.getenv("JARVIS_WAKE_PRE_ROLL_SECONDS", "1.5"))
WHISPER_MODE = os.getenv("JARVIS_WHISPER_MODE", "conversation")

BROWSER_AGENT_PROVIDER = os.getenv("JARVIS_BROWSER_AGENT_PROVIDER", "fireworks")
BROWSER_AGENT_MODEL = os.getenv(
    "JARVIS_BROWSER_AGENT_MODEL",
    "accounts/fireworks/models/qwen3p7-plus",
)
BROWSER_AGENT_REASONING_EFFORT = os.getenv(
    "JARVIS_BROWSER_AGENT_REASONING_EFFORT", ""
)
BROWSER_AGENT_MAX_STEPS = int(os.getenv("JARVIS_BROWSER_AGENT_MAX_STEPS", "50"))  # safety margin - vision grounding cuts typical step counts, not a license to flail
BROWSER_AGENT_TIMEOUT = int(os.getenv("JARVIS_BROWSER_AGENT_TIMEOUT", "480"))
BROWSER_AGENT_TOOL_TIMEOUT = 120
BROWSER_AGENT_KEEP_LAST_IMAGES = int(os.getenv("JARVIS_BROWSER_AGENT_KEEP_LAST_IMAGES", "1"))
# payload knobs — clamped at read time (env may be stale or test-patched)
try:
    _raw_look_width = int(os.getenv("JARVIS_BROWSER_AGENT_LOOK_WIDTH", "1280"))
except Exception:
    _raw_look_width = 1280
BROWSER_AGENT_LOOK_WIDTH = max(640, min(1920, _raw_look_width))
try:
    _raw_jpeg_quality = int(os.getenv("JARVIS_BROWSER_AGENT_JPEG_QUALITY", "70"))
except Exception:
    _raw_jpeg_quality = 70
BROWSER_AGENT_JPEG_QUALITY = max(40, min(95, _raw_jpeg_quality))

OPTIONAL_API_KEYS = {
    "GROQ_API_KEY": GROQ_API_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "OPENROUTER_API_KEY": OPENROUTER_API_KEY,
    "ELEVENLABS_API_KEY": ELEVENLABS_API_KEY,
    "FISH_API_KEY": FISH_API_KEY,
    "FIREWORKS_API_KEY": FIREWORKS_API_KEY,
    "CLINE_API_KEY": CLINE_API_KEY,
}


def validate_environment():
    missing = [name for name, value in OPTIONAL_API_KEYS.items() if not value]
    if missing:
        logging.warning(
            "[CONFIG] Missing optional API keys: %s",
            ", ".join(missing),
        )
    return missing
