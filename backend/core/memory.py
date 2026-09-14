"""Conversation memory with optional disk persistence.

History is kept in-process for speed, and mirrored to a small JSON file so
context survives backend restarts. Persistence is best-effort: any IO error
is swallowed so a corrupt/missing file never breaks chat.
"""

import json
import logging
import threading

try:
    from backend.config import BASE_DIR
except Exception:  # pragma: no cover - config import fallback
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

MAX_HISTORY = 20
_MEMORY_FILE = BASE_DIR / "data" / "conversation_history.json"

_lock = threading.Lock()
conversation_history = []


def _load_from_disk():
    global conversation_history
    try:
        if not _MEMORY_FILE.exists():
            return
        with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            # keep only well-formed role/content dicts, trimmed to cap
            cleaned = [
                {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
                for m in data
                if isinstance(m, dict) and m.get("role") and m.get("content")
            ]
            conversation_history = cleaned[-MAX_HISTORY:]
            logging.info("[MEMORY] Loaded %d messages from disk", len(conversation_history))
    except Exception as exc:  # corrupt file etc.
        logging.warning("[MEMORY] Could not load history: %s", exc)
        conversation_history = []


def _save_to_disk():
    try:
        _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _MEMORY_FILE.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(conversation_history, f, ensure_ascii=False)
        tmp.replace(_MEMORY_FILE)  # atomic-ish on Windows when target exists? use os.replace
    except Exception as exc:
        logging.warning("[MEMORY] Could not persist history: %s", exc)


def add_message(role: str, content: str):
    with _lock:
        conversation_history.append({"role": role, "content": content})
        if len(conversation_history) > MAX_HISTORY:
            conversation_history.pop(0)
        _save_to_disk()


def get_history():
    with _lock:
        return list(conversation_history)


def clear_history():
    global conversation_history
    with _lock:
        conversation_history = []
        try:
            if _MEMORY_FILE.exists():
                _MEMORY_FILE.unlink()
        except Exception as exc:
            logging.warning("[MEMORY] Could not delete history file: %s", exc)
    print("[MEMORY] Cleared")


_load_from_disk()
