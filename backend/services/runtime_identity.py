"""Per-process runtime identity for the Jarvis stack (G11 / F52).

The backend is the sole owner of intelligence state, but every supervised
worker still needs a durable identity so the supervisor (watcher.py /
main.js) can tell which instance is really running *before* trusting a
listening port. ``runtime_identity`` provides:

* ``instance_id()`` — a per-process UUID, cached for the life of the process.
* ``protocol_version()`` — the local IPC/health contract version. Bump when
  the health payload or the /ask contract changes shape.
* ``build_id()`` — an explicit build label when ``JARVIS_BUILD_ID`` is set
  (CI/dists set it; local dev reports "dev").
* ``write_instance_file(role)`` / ``read_instance_file(role)`` — an atomic
  ``data/runtime/<role>-instance.json`` stamp (pid, instance_id, started_at,
  protocol). The supervisor keeps this handle instead of inferring ownership
  from a listening port alone.

F52 — the above is only safe when the stamp itself is trustworthy:

* ``started_at()`` is the *process* start identity, frozen when this module
  loads. It used to return ``time.time()`` on every call, so a stamp's start
  time changed on each write and could not identify a run.
* temp files are per-writer (pid + uuid) instead of one shared
  ``<role>-instance.json.tmp`` that concurrent writers clobbered.
* ``clear_instance_file()`` is ownership-checked: it removes a stamp only
  when the caller proves ownership (this process wrote it, or it names the
  pid of a worker the caller created and is retiring). An OLD exit can
  therefore never delete a NEW stamp.
"""

import json
import os
import threading
import time
import uuid

from backend.config import BASE_DIR

PROTOCOL_VERSION = 2
_RUNTIME_DIR = BASE_DIR / "data" / "runtime"

_lock = threading.Lock()
_instance_id = None
# Frozen at import: this is the process's start identity, not "now".
_started_at = time.time()


def instance_id():
    """Stable per-process identity; generated once, cached forever."""
    global _instance_id
    with _lock:
        if _instance_id is None:
            _instance_id = str(uuid.uuid4())
        return _instance_id


def protocol_version():
    return PROTOCOL_VERSION


def build_id():
    return os.getenv("JARVIS_BUILD_ID", "") or "dev"


def started_at():
    """Stable per-process start time (F52).

    Cached when the module loads so every stamp this process writes carries
    the SAME start identity; ``time.time()`` per call made the stamp's start
    time drift and made "same run" unprovable.
    """
    return _started_at


def owns_stamp(stamp):
    """True when *stamp* was written by THIS process (instance id + pid)."""
    if not isinstance(stamp, dict):
        return False
    if stamp.get("instance_id") != instance_id():
        return False
    try:
        return int(stamp.get("pid") or 0) == os.getpid()
    except (TypeError, ValueError):
        return False


# ── Instance stamp file (one per role) ──────────────────────────────────────
def _stamp_path(role):
    return _RUNTIME_DIR / ("%s-instance.json" % (role or "backend"))


def write_instance_file(role="backend", extra=None):
    """Atomically stamp ``data/runtime/<role>-instance.json``.

    The stamping process is the owner: it records its pid + instance id so a
    supervisor can later compare the file against the listening port instead
    of assuming any listener on the port is ours. ``extra`` may carry the
    secret nonce or task-engine info for the supervisor's use.

    F52 — the temp file is unique per writer (pid + uuid). One shared
    ``<role>-instance.json.tmp`` let two concurrent writers (two workers of
    the same role, or a restart racing the old process) truncate each other
    mid-write and publish a half file.
    """
    target = _stamp_path(role)
    tmp = target.with_name(
        "%s.%d.%s.tmp" % (target.name, os.getpid(), uuid.uuid4().hex[:8])
    )
    try:
        _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "role": role,
            "instance_id": instance_id(),
            "pid": os.getpid(),
            "started_at": started_at(),
            "protocol": protocol_version(),
            "build": build_id(),
        }
        if extra:
            payload.update(extra)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        # Atomic publish. Windows can transiently deny the replace while
        # another writer is replacing the SAME destination (WinError 5), so a
        # bounded retry lets concurrent same-role writers converge on the
        # last stamp instead of one of them publishing nothing.
        deadline = time.monotonic() + 1.0
        while True:
            try:
                tmp.replace(target)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        return payload
    except Exception:
        # Identity stamps are best-effort supervision metadata; a failure must
        # never take the backend (or the voice worker) down.
        return None
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def read_instance_file(role="backend"):
    try:
        with open(_stamp_path(role), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def clear_instance_file(role="backend", owned_pid=None):
    """Remove the stamp ONLY when the caller owns it (F52).

    Ownership is proven by either:

    * this process wrote the stamp (same ``instance_id`` AND ``pid``), or
    * *owned_pid* names a worker the caller created and is now retiring, and
      the stamp is that worker's stamp.

    Anything else — a stamp written by a newer run, a foreign/corrupt stamp,
    a missing file — is left untouched. Returns True when a file was removed.
    """
    try:
        stamp = read_instance_file(role)
        if stamp is None:
            return False
        owned = owns_stamp(stamp)
        if not owned and owned_pid:
            try:
                owned = int(stamp.get("pid") or 0) == int(owned_pid)
            except (TypeError, ValueError):
                owned = False
        if not owned:
            return False
        p = _stamp_path(role)
        if p.exists():
            p.unlink()
            return True
        return False
    except Exception:
        return False