"""Conversation memory with optional disk persistence.

History is kept in-process for speed, and mirrored to a small JSON file so
context survives backend restarts. Persistence is best-effort: any IO error
is swallowed so a corrupt/missing file never breaks chat.

P0-11: the in-memory append is the source of truth and happens IMMEDIATELY;
the JSON file is written by ONE background writer thread, debounced, with the
snapshot taken at write time. Two consequences that matter:

* nothing on the request thread touches the disk before the reply's first
  delta — the old code rewrote this whole file under the lock on the way to
  the user's answer (10-150 ms on Windows, worse with an AV/OneDrive scan);
* because the writer always persists the CURRENT list, a debounce can only
  ever delay a message, never reorder or drop one.

Durability is tracked with generations rather than a dirty flag: an append
bumps ``_write_generation``, a successful write records the generation it
persisted, and "nothing pending" means the two are equal. The tail is not
optional — ``flush_history`` runs on the shutdown path and from an ``atexit``
hook, so the last turn of a session still reaches the file.
"""

import atexit
import json
import logging
import os
import threading
import time

try:
    from backend.config import BASE_DIR
except Exception:  # pragma: no cover - config import fallback
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

#: R20 — 40 turns: every chain/task turn now commits its user half AND its
#: per-step notes, so 20 slots held only ~7 real exchanges before the older
#: half of a working session rolled off exactly when a follow-up needed it.
MAX_HISTORY = 40
_MEMORY_FILE = BASE_DIR / "data" / "conversation_history.json"

#: P0-11 — how long the writer waits for the burst of appends that is a turn
#: (user + assistant message) to settle before it touches the disk.
DEBOUNCE_SECONDS = 0.5
#: P0-11 — how long ``flush_history`` waits for the writer by default.
FLUSH_TIMEOUT_SECONDS = 2.0
#: Windows file locking (an AV or OneDrive holding the target) is the common
#: cause of a failed rename, and it is transient. Retry rather than lose the
#: tail of the conversation.
_REPLACE_ATTEMPTS = 5
_REPLACE_BACKOFF_SECONDS = 0.05

_lock = threading.Lock()
conversation_history = []

_wake = threading.Event()          # "there is a change to persist"
_stop = threading.Event()          # shutdown: drain and exit
_writer = None
_writer_guard = threading.Lock()
#: P0-11 — visibility: how the writer has behaved since process start.
_write_generation = 0              # bumped by every append
_saved_generation = 0              # the generation the file reflects
_failures = 0                      # writes that raised (never surfaced)


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


def _pending():
    """True when the file does not yet reflect the in-memory history."""
    with _lock:
        return _saved_generation < _write_generation


def _write_snapshot(messages):
    """Atomically replace the history file with *messages*. Raises on failure.

    The snapshot is a plain list taken under the lock before this call, so the
    lock is NOT held across the file IO — ``get_history`` never waits on a
    disk write (it did, before P0-11).
    """
    _MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = _MEMORY_FILE.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False)
    last_error = None
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, _MEMORY_FILE)
            return True
        except PermissionError as exc:  # Windows: transient file lock
            last_error = exc
            time.sleep(_REPLACE_BACKOFF_SECONDS * (attempt + 1))
    if last_error is not None:
        raise last_error
    return False


def _persist_once():
    """Persist the current snapshot. Never raises; counts failures."""
    global _failures, _saved_generation
    with _lock:
        messages = list(conversation_history)
        generation = _write_generation
    try:
        _write_snapshot(messages)
    except Exception as exc:
        _failures += 1
        logging.warning("[MEMORY] Could not persist history: %s", exc)
        return False
    with _lock:
        if generation > _saved_generation:
            _saved_generation = generation
    return True


def _writer_loop():
    """The single history writer: debounce, then persist the latest state."""
    while True:
        if not _wake.wait(0.25):
            if _stop.is_set():
                _persist_once()
                return
            continue
        if _stop.is_set():
            _persist_once()
            return
        # Debounce: let the turn's appends (user, then assistant) settle.
        time.sleep(DEBOUNCE_SECONDS)
        _wake.clear()
        _persist_once()


def _ensure_writer():
    global _writer
    if _writer is not None and _writer.is_alive():
        return _writer
    with _writer_guard:
        if _writer is None or not _writer.is_alive():
            _stop.clear()
            _writer = threading.Thread(
                target=_writer_loop, name="history-writer", daemon=True)
            _writer.start()
        return _writer


def flush_history(timeout=FLUSH_TIMEOUT_SECONDS):
    """Persist any pending change NOW and wait for the writer. Never raises.

    Called on the shutdown path, from ``atexit``, and by tests that assert the
    file — the in-memory history is always the source of truth, so this is
    about durability only.
    """
    if _pending() and not _persist_once():
        return False
    deadline = time.monotonic() + max(0.0, float(timeout))
    while time.monotonic() < deadline:
        if not _pending():
            return True
        time.sleep(0.01)
    return not _pending()


def add_message(role: str, content: str):
    """Append one message IN MEMORY; the file follows (P0-11).

    The append is synchronous and immediately visible to ``get_history`` —
    only the durable copy is deferred.
    """
    global _write_generation
    with _lock:
        conversation_history.append({"role": role, "content": content})
        if len(conversation_history) > MAX_HISTORY:
            conversation_history.pop(0)
        _write_generation += 1
    _ensure_writer()
    _wake.set()


def get_history():
    with _lock:
        return list(conversation_history)


def clear_history():
    global conversation_history, _saved_generation
    with _lock:
        conversation_history = []
        # A pending write must not resurrect what was just forgotten: the
        # writer takes its snapshot at write time, so the next persist would
        # write [] anyway — this simply stops it from re-creating the file.
        _wake.clear()
        _saved_generation = _write_generation
        try:
            if _MEMORY_FILE.exists():
                _MEMORY_FILE.unlink()
        except Exception as exc:
            logging.warning("[MEMORY] Could not delete history file: %s", exc)
    print("[MEMORY] Cleared")


def history_write_stats():
    """P0-11 — the history writer's counters, for telemetry and tests."""
    with _lock:
        return {
            "pending": _saved_generation < _write_generation,
            "saved_generation": _saved_generation,
            "write_generation": _write_generation,
            "failures": _failures,
            "writer_alive": bool(_writer and _writer.is_alive()),
        }


def _flush_at_exit():
    """P0-11 — the tail of the conversation must not die with the process."""
    try:
        flush_history(timeout=1.0)
    except Exception:
        pass


atexit.register(_flush_at_exit)

_load_from_disk()
