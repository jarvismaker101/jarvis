"""Workspace-scoped code grants, a change journal and owned processes.

Fable-5 audit G2 / F22 — make code changes reversible and confined.

Before this module ``code_tools.write_file`` opened the target with ``"w"``
(no backup, no atomicity, no hash check), ``_resolve_path`` happily wrote to
any absolute path in the system, and ``run_script`` rebuilt a shell string by
hand and pushed it through ``shell=True`` — so a timeout killed the wrapper
but not the tree it spawned.

This module supplies the four missing pieces:

  * **scoped grants** — :func:`is_allowed` confines writes/executions to
    configured workspace roots (``JARVIS_WORKSPACE_ROOTS``; the process temp
    dir is a default root so tests and scratch work keep working, and
    ``JARVIS_CODE_SCOPE=strict`` drops it);
  * **an atomic writer with expected-content hashes** — :func:`atomic_write`
    writes to a sibling temp file, fsyncs and ``os.replace``s, so a crash can
    never leave a half-written source file, and refuses to overwrite a file
    that changed since the caller last read it;
  * **a change journal** — every write records a restore point, so
    :func:`restore_last` / :func:`restore` are a real undo path rather than a
    promise;
  * **owned processes** — :func:`spawn` runs a real argv (no shell parsing for
    Python scripts) inside a Windows Job Object, so a timeout or a
    cancellation terminates every descendant instead of orphaning it.

Pure-ish module: stdlib only.
"""

import ctypes
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from typing import Dict, List, Optional, Sequence

JOURNAL_ENV = "JARVIS_CHANGE_JOURNAL"
ROOTS_ENV = "JARVIS_WORKSPACE_ROOTS"
SCOPE_ENV = "JARVIS_CODE_SCOPE"

_CREATE_NO_WINDOW = 0x08000000


# ── workspace scope ────────────────────────────────────────────────────────
def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def default_roots():
    """Roots a write/execution is allowed inside, by default.

    The system temp dir is included so scratch work and the test-suite keep
    behaving; ``JARVIS_CODE_SCOPE=strict`` removes it.
    """
    roots = [_repo_root(), os.getcwd()]
    try:
        tmp = os.path.realpath(tempfile.gettempdir())
    except Exception:
        tmp = None
    if tmp:
        roots.append(tmp)
    configured = os.getenv(ROOTS_ENV, "")
    if configured.strip():
        roots = [p for p in (part.strip() for part in configured.split(os.pathsep))
                 if p]
    elif str(os.getenv(SCOPE_ENV, "")).strip().lower() == "strict":
        roots = [r for r in roots if not tmp or not r.startswith(tmp)]
    return [os.path.normpath(os.path.realpath(r)) for r in roots]


def roots():
    return default_roots()


def resolve(path):
    """Expand and absolutise *path* against the repo root (as code_tools did)."""
    if not path:
        return ""
    expanded = os.path.expandvars(os.path.expanduser(str(path).strip()))
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(_repo_root(), expanded))


def is_allowed(path):
    """True when *path* is inside a granted workspace root."""
    resolved = resolve(path)
    if not resolved:
        return False
    try:
        target = os.path.realpath(resolved)
    except Exception:
        target = os.path.normpath(resolved)
    for root in roots():
        try:
            if target == root or target.startswith(root + os.sep):
                return True
        except Exception:
            continue
    return False


def deny_reason(path):
    return ("path %r is outside the granted workspaces (%s)"
            % (resolve(path), ", ".join(roots())))


def grant_check(path):
    """Return ``(allowed, resolved, reason)``."""
    resolved = resolve(path)
    if is_allowed(resolved):
        return True, resolved, ""
    return False, resolved, deny_reason(resolved)


# ── content hashes ─────────────────────────────────────────────────────────
def content_hash(path=None, text=None):
    """sha256 of a file's bytes or of an in-memory string."""
    digest = hashlib.sha256()
    if text is not None:
        digest.update(text.encode("utf-8", "replace"))
        return digest.hexdigest()
    try:
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        return None


# ── change journal ─────────────────────────────────────────────────────────
_JOURNAL_LOCK = threading.RLock()
_JOURNAL: List[Dict] = []
_JOURNAL_DIR = None
#: F22: the journal is persisted, so "undo works after restart" — an in-memory
#: list lost every restore point when the worker was replaced.
_JOURNAL_FILE_NAME = "journal.json"
_JOURNAL_LOADED = False


def journal_dir():
    """Directory holding restore-point backups."""
    global _JOURNAL_DIR
    if _JOURNAL_DIR:
        return _JOURNAL_DIR
    configured = os.getenv(JOURNAL_ENV, "").strip()
    if configured:
        path = os.path.abspath(configured)
    else:
        path = os.path.join(_repo_root(), "data", "change_journal")
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        path = os.path.join(tempfile.gettempdir(), "jarvis_change_journal")
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
    _JOURNAL_DIR = path
    return path


def _backup_path(path, entry_id):
    base = os.path.basename(path) or "file"
    return os.path.join(journal_dir(), "%s.%s.bak" % (entry_id, base))


def _journal_file():
    return os.path.join(journal_dir(), _JOURNAL_FILE_NAME)


def _load_journal():
    """Load persisted restore points once, so undo survives a restart (F22)."""
    global _JOURNAL_LOADED
    if _JOURNAL_LOADED:
        return
    _JOURNAL_LOADED = True
    try:
        with open(_journal_file(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return
    if not isinstance(data, list):
        return
    with _JOURNAL_LOCK:
        _JOURNAL[:] = [entry for entry in data if isinstance(entry, dict)][-200:]


def _save_journal():
    """Persist the journal atomically; never raises."""
    payload = json.dumps(_JOURNAL[-200:], ensure_ascii=False)
    target = _journal_file()
    tmp = "%s.%d.tmp" % (target, os.getpid())
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except OSError as exc:
        logging.debug("[GRANTS] journal persist failed: %s", exc)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def record(path, before_hash=None, backup=None, kind="write", existed=None):
    """Append a persisted restore point; returns the journal entry.

    ``existed`` MUST be the file's existence BEFORE the write. It used to be
    sampled here — i.e. AFTER ``os.replace`` — so a brand-new file was recorded
    as pre-existing with no backup, and undoing its creation was impossible.
    """
    _load_journal()
    entry = {
        "id": "c%d-%d" % (int(time.time() * 1000), len(_JOURNAL) + 1),
        "path": path,
        "kind": kind,
        # F22: explicit pre-write existence, and the post-write hash used to
        # detect an intervening edit before restoring over it.
        "existed": (os.path.exists(path) if path else False)
                   if existed is None else bool(existed),
        "before_hash": before_hash,
        "after_hash": content_hash(path) if path and os.path.exists(path) else None,
        "backup": backup,
        "ts": time.time(),
    }
    with _JOURNAL_LOCK:
        _JOURNAL.append(entry)
        if len(_JOURNAL) > 200:
            _JOURNAL[:] = _JOURNAL[-200:]
        _save_journal()
    return entry


def entries():
    _load_journal()
    with _JOURNAL_LOCK:
        return list(_JOURNAL)


def last_entry(path=None):
    _load_journal()
    with _JOURNAL_LOCK:
        for entry in reversed(_JOURNAL):
            if path is None or entry.get("path") == path:
                return entry
    return None


def _snapshot(path, entry_id):
    """Copy the current file aside so it can be restored later."""
    if not path or not os.path.exists(path):
        return None
    try:
        target = _backup_path(path, entry_id)
        with open(path, "rb") as src, open(target, "wb") as dst:
            for block in iter(lambda: src.read(65536), b""):
                dst.write(block)
        return target
    except OSError as exc:
        logging.debug("[GRANTS] backup failed for %s: %s", path, exc)
        return None


def restore(entry_id=None, path=None, force=False):
    """Undo a recorded write. Returns ``(ok, message)``.

    F22: an undo refuses to clobber work that happened AFTER the restore point
    was recorded. The old version wrote the backup straight over whatever was
    there, so undoing Jarvis's edit silently destroyed the user's later edit —
    or deleted a file the user had created. ``force=True`` is the explicit
    override for a caller that has already inspected the diff.
    """
    _load_journal()
    with _JOURNAL_LOCK:
        entry = None
        for candidate in reversed(_JOURNAL):
            if entry_id and candidate.get("id") == entry_id:
                entry = candidate
                break
            if not entry_id and path and candidate.get("path") == path:
                entry = candidate
                break
        if entry is None:
            return False, "no restore point found"
    target = entry.get("path")
    backup = entry.get("backup")

    current = content_hash(target) if target and os.path.exists(target) else None
    recorded = entry.get("after_hash")
    if not force:
        if current is not None and recorded is not None and current != recorded:
            return False, (
                "%s changed after this restore point was recorded "
                "(expected %s, found %s) — your later edits are kept; "
                "inspect the diff and pass force to overwrite"
                % (target, str(recorded)[:12], str(current)[:12]))
        if current is not None and recorded is None and not entry.get("existed"):
            return False, (
                "%s was created by someone else after this restore point "
                "was recorded — refusing to delete it" % target)

    try:
        if entry.get("existed"):
            if not backup or not os.path.exists(backup):
                return False, "restore point %s is missing its backup" % entry["id"]
            with open(backup, "rb") as src, open(target, "wb") as dst:
                for block in iter(lambda: src.read(65536), b""):
                    dst.write(block)
        else:
            if os.path.exists(target):
                os.remove(target)
        return True, "restored %s" % target
    except OSError as exc:
        return False, "restore failed: %s" % exc


def restore_last(path=None):
    """Undo the most recent write (optionally the most recent for *path*)."""
    entry = last_entry(path)
    if entry is None:
        return False, "nothing to undo"
    return restore(entry_id=entry["id"])


# ── atomic, journalled writes ──────────────────────────────────────────────
def atomic_write(path, content="", expected_hash=None, encoding="utf-8",
                 expect=None):
    """Write *content* to *path* atomically, journalled and scope-checked.

    ``expected_hash`` is the hash of the file as the caller last saw it: if
    the file changed since, the write is refused instead of silently clobbering
    someone else's edit.

    ``expect`` names the precondition explicitly (F22 — an approved diff is
    bound to a create/replace precondition, not an optional hash):

      * ``"create"``  — the file must NOT exist; a write that would overwrite
        an existing file is refused.
      * ``"replace"`` — the file MUST exist and its hash must match
        ``expected_hash`` when one is given.
      * ``"any"`` / None — legacy behaviour: create or replace, with the hash
        check applied whenever ``expected_hash`` is supplied.

    ``expected_hash`` alone implies ``"replace"`` (a hash can only be known
    for a file that exists). A backup or journal failure is reported as a
    refusal, never as a silent write without a restore point.
    """
    allowed, resolved, reason = grant_check(path)
    if not allowed:
        return {"ok": False, "error": reason, "path": resolved, "content": ""}

    existed = os.path.exists(resolved)
    before_hash = content_hash(resolved) if existed else None

    mode = (expect or "").strip().lower() or None
    if mode is None and expected_hash:
        mode = "replace"
    if mode not in (None, "any", "create", "replace"):
        return {"ok": False, "path": resolved, "content": "",
                "error": "unknown write precondition %r" % expect}
    if mode == "create" and existed:
        return {"ok": False, "path": resolved, "content": "",
                "error": ("refusing to overwrite existing file %s "
                          "(create precondition)" % resolved)}
    if mode == "replace" and not existed:
        return {"ok": False, "path": resolved, "content": "",
                "error": ("%s does not exist (replace precondition needs a "
                          "file to change)" % resolved)}
    if existed and expected_hash and before_hash != expected_hash:
        return {
            "ok": False,
            "error": ("file changed since it was last read "
                      "(expected %s, found %s) — re-read and retry"
                      % (str(expected_hash)[:12], str(before_hash)[:12])),
            "path": resolved,
            "content": "",
        }

    entry_id = "c%d-%d" % (int(time.time() * 1000), len(_JOURNAL) + 1)
    backup = _snapshot(resolved, entry_id) if existed else None
    if existed and not backup:
        # F22: without the backup the write could not be undone. Refusing is
        # the safe failure — "backup/assignment failure is safe".
        return {"ok": False, "path": resolved, "content": "",
                "error": ("could not create a restore point for %s; "
                          "refusing to overwrite it" % resolved)}

    directory = os.path.dirname(resolved) or "."
    tmp_path = os.path.join(directory, ".jarvis-tmp-%d-%s" % (os.getpid(), entry_id))
    try:
        os.makedirs(directory, exist_ok=True)
        with open(tmp_path, "w", encoding=encoding, errors="replace", newline="") as fh:
            fh.write(content or "")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, resolved)
    except OSError as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return {"ok": False, "error": str(exc), "path": resolved, "content": ""}
    entry = record(resolved, before_hash=before_hash, backup=backup,
                   existed=existed)
    if not entry.get("after_hash"):
        # The journal is persisted inside record(); if the entry cannot carry a
        # post-write hash the undo would be unable to detect conflicts.
        logging.warning("[GRANTS] restore point for %s lacks an after-hash",
                        resolved)
    return {
        "ok": True,
        "content": "Wrote %d chars to %s" % (len(content or ""), resolved),
        "error": "",
        "path": resolved,
        "exit_code": 0,
        "before_hash": before_hash,
        "after_hash": content_hash(resolved),
        "restore_id": entry["id"],
    }


def preview_diff(path, content=""):
    """Unified diff preview of what :func:`atomic_write` would change."""
    import difflib
    resolved = resolve(path)
    before = ""
    if os.path.exists(resolved):
        try:
            with open(resolved, "r", encoding="utf-8", errors="replace") as fh:
                before = fh.read()
        except OSError:
            before = ""
    diff = "\n".join(difflib.unified_diff(
        before.splitlines(), (content or "").splitlines(),
        fromfile="a/%s" % os.path.basename(resolved),
        tofile="b/%s" % os.path.basename(resolved), lineterm=""))
    return diff


# ── owned processes (Job Objects on Windows) ───────────────────────────────
#: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE and the info-class constant.
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JobObjectExtendedLimitInformation = 9

if os.name == "nt":
    # F22: the structures are declared and the calls are typed/checked. The
    # old version passed a raw ``(c_ulonglong * 12)`` blob with a guessed
    # offset and ignored every return value, so a rejected limit or a failed
    # assignment looked exactly like success.
    class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", ctypes.c_uint32),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", ctypes.c_uint32),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", ctypes.c_uint32),
            ("SchedulingClass", ctypes.c_uint32),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]


def _typed_kernel32():
    """kernel32 with declared signatures so return values are real ints."""
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.TerminateJobObject.restype = ctypes.c_int
    kernel32.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.CloseHandle.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    return kernel32


class _JobObject:
    """Windows Job Object that kills every descendant on close/terminate."""

    def __init__(self):
        self.handle = None
        self.last_error = ""
        if os.name != "nt":
            return
        try:
            kernel32 = _typed_kernel32()
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                self.last_error = "CreateJobObjectW failed"
                return
            info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = (
                _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
            ok = kernel32.SetInformationJobObject(
                ctypes.c_void_p(handle), _JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info))
            if not ok:
                # Without KILL_ON_JOB_CLOSE the descendants would outlive the
                # job; refuse to hand back a handle that cannot guarantee it.
                self.last_error = "SetInformationJobObject failed"
                kernel32.CloseHandle(ctypes.c_void_p(handle))
                return
            self._kernel32 = kernel32
            self.handle = handle
        except Exception as exc:
            self.last_error = str(exc)
            logging.debug("[GRANTS] job object unavailable: %s", exc)
            self.handle = None

    def assign(self, pid):
        if not self.handle:
            return False
        try:
            kernel32 = self._kernel32
            handle = kernel32.OpenProcess(0x1F0FFF, False, int(pid))
            if not handle:
                self.last_error = "OpenProcess failed"
                return False
            ok = kernel32.AssignProcessToJobObject(
                ctypes.c_void_p(self.handle), ctypes.c_void_p(handle))
            kernel32.CloseHandle(ctypes.c_void_p(handle))
            if not ok:
                self.last_error = "AssignProcessToJobObject failed"
            return bool(ok)
        except Exception as exc:
            self.last_error = str(exc)
            return False

    def terminate(self):
        if not self.handle:
            return False
        try:
            return bool(self._kernel32.TerminateJobObject(
                ctypes.c_void_p(self.handle), 1))
        except Exception:
            return False

    def close(self):
        if not self.handle:
            return
        try:
            self._kernel32.CloseHandle(ctypes.c_void_p(self.handle))
        except Exception:
            pass
        self.handle = None


def build_argv(script_path, args=None):
    """Direct argv for a script — no shell string reconstruction.

    Python scripts run under ``sys.executable`` so the interpreter that owns
    the workspace is the one that runs the file, and quoting can never be
    mis-parsed.
    """
    argv_path = [str(script_path)]
    ext = os.path.splitext(str(script_path))[1].lower()
    if ext == ".py":
        argv_path = [sys.executable, str(script_path)]
    return argv_path + [str(a) for a in (args or [])]


def spawn(argv, cwd=None, timeout=None, job=None, env=None, shell=False):
    """Run *argv* as an owned child process.

    When *job* is a :class:`backend.services.jobs.JobToken` the process is
    attached to it, so cancelling the job terminates the whole tree. On
    Windows the process is additionally placed in a Job Object whose
    ``KILL_ON_JOB_CLOSE`` flag guarantees descendants die too.

    F22: ownership is enforced, not attempted. If the Job Object cannot be
    created or the child cannot be assigned to it, the child is killed and the
    call raises — an unowned process tree that a later timeout could not
    terminate is worse than a failed launch. On POSIX the child gets its own
    session (``start_new_session``), so the group we signal later contains only
    our own descendants.
    """
    job_object = _JobObject() if os.name == "nt" else None
    popen = None
    try:
        popen = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=bool(shell),
            env=env,
            # F22: isolated child ownership on POSIX — never signal a group
            # shared with the backend or another task.
            start_new_session=(os.name != "nt"),
            creationflags=_CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except Exception:
        if job_object:
            job_object.close()
        raise
    if job_object is not None:
        if not job_object.assign(popen.pid):
            job_object.terminate()
            job_object.close()
            _kill_tree_fallback(popen)
            raise RuntimeError(
                "could not take ownership of pid %s (%s); refusing to run an "
                "unbounded process tree"
                % (popen.pid, job_object.last_error or "job assignment failed"))
    if job is not None and hasattr(job, "attach"):
        job.attach(popen)
        if job_object is not None:
            job._job_object = job_object
    elif job_object:
        # No owning job: make sure the handle is released with the process.
        popen._jarvis_job_object = job_object
    return popen


def run_argv(argv, cwd=None, timeout=None, job=None, shell=False, input_text=None):
    """Run *argv* to completion, bounded by *timeout* and job cancellation."""
    popen = spawn(argv, cwd=cwd, timeout=timeout, job=job, shell=shell)
    job_object = getattr(popen, "_jarvis_job_object", None)
    if job_object is None and job is not None:
        job_object = getattr(job, "_job_object", None)
    try:
        try:
            out, _ = popen.communicate(input=input_text, timeout=timeout)
        except subprocess.TimeoutExpired:
            if job_object:
                job_object.terminate()
            _kill_tree_fallback(popen)
            try:
                out, _ = popen.communicate(timeout=5)
            except Exception:
                out = ""
            return {
                "ok": False,
                "error": "command timed out after %ss" % timeout,
                "content": (out or "")[-4000:],
                "exit_code": -1,
                "timed_out": True,
            }
        return {
            "ok": popen.returncode == 0,
            "content": out or "",
            "error": "" if popen.returncode == 0 else
                     "exited with code %s" % popen.returncode,
            "exit_code": popen.returncode,
        }
    finally:
        if job_object:
            job_object.close()
        try:
            if popen.poll() is None:
                popen.kill()
        except Exception:
            pass


def _kill_tree_fallback(popen):
    """Kill only the tree this call owns.

    F22: on POSIX the child is its own session leader, so the group id is the
    child's pid — ``os.getpgid(popen.pid)`` could otherwise resolve to the
    backend's own group and take the whole application down.
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(popen.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5, creationflags=_CREATE_NO_WINDOW)
        else:
            import signal
            try:
                os.killpg(int(popen.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    except Exception:
        try:
            popen.kill()
        except Exception:
            pass
