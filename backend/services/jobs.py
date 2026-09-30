"""Per-job cancellation tokens.

Fable-5 audit G2 / F20 — cancel jobs, not global flags.

Before this module every long-running worker shared one module-level
``threading.Event``:

  * ``browser_agent.run_browser_task`` *cleared* the shared stop event on
    entry, so starting job B silently disarmed the stop the user had just
    issued for job A;
  * ``research_service.run_research`` cleared its own, for the same reason;
  * ``/task/stop`` could not reach the opencode subprocess at all;
  * an idle stop (nothing running) left the event set and poisoned the next
    lookup, which then stopped immediately for no reason.

A :class:`JobToken` carries its own cancellation event, an optional monotonic
deadline, and the subprocess handles it owns so cancelling a job terminates
the whole descendant tree instead of orphaning it. Cancellation is addressed
by job id: stopping an idle id is a no-op, and a new job never clears another
job's cancellation.

Legacy module-level stop events still work — they are wired to the *current*
job (see :func:`request_stop`) so nothing that used them regresses.
"""

import itertools
import logging
import os
import subprocess
import threading
import time
from typing import Dict, List, Optional

from backend.core.deadline import Deadline


class Cancelled(Exception):
    """Raised at a job checkpoint when the job was cancelled."""


# ── F50: every EFFECT is a typed job ───────────────────────────────────────
#: F50 ("Give the Backend Sole Ownership of Intelligence State") requires all
#: effects — typed requests, voice controls, background workers and local
#: setup commands — to run through ONE job runtime with typed authority
#: instead of each surface owning its own gate/thread. The kind is part of
#: the job so cancellation, history and ownership stay addressable per
#: effect surface.
EFFECT_TYPED = "typed"
EFFECT_VOICE = "voice"
EFFECT_BACKGROUND = "background"
EFFECT_SETUP = "setup"
EFFECT_KINDS = frozenset((
    EFFECT_TYPED, EFFECT_VOICE, EFFECT_BACKGROUND, EFFECT_SETUP,
))


class UnknownEffectKind(Exception):
    """Raised when an effect is submitted with an untyped kind (F50)."""


_counter = itertools.count(1)
_lock = threading.RLock()

#: Every live job, by id. Finished jobs are dropped.
_jobs: Dict[str, "JobToken"] = {}

#: The job that should receive an unaddressed stop (most recently started
#: job that is still running).
_current_id: Optional[str] = None

#: Handlers notified when a job is cancelled, by kind (legacy bridges).
_stop_handlers: Dict[str, List] = {}


def register_stop_handler(kind, fn):
    """Register ``fn(job)`` called when a job of *kind* is cancelled.

    Used to keep the pre-existing module-level stop events of
    browser_agent / research_service in sync without making them the source
    of truth.
    """
    with _lock:
        _stop_handlers.setdefault(kind, []).append(fn)


class JobToken:
    """One cancellable unit of work."""

    def __init__(self, job_id, kind="job", timeout=None, label="",
                 generation=None):
        self.job_id = job_id
        self.kind = kind
        self.label = label or kind
        #: F50 — the worker generation that owns this effect (None for legacy
        #: jobs). A superseded generation can no longer claim authority.
        self.generation = generation
        #: Creation order — a monotonic tie-breaker for job identity.
        self.seq = next(_counter)
        self.created_at = time.time()
        self._cancel = threading.Event()
        self._paused = threading.Event()
        self._processes = []
        self._proc_lock = threading.Lock()
        #: [P1-13] open resources (a streaming provider response) that must be
        #: closed when the job is cancelled, so a blocked read is released.
        self._closers = []
        self._closer_lock = threading.Lock()
        self._finished = False
        #: Monotonic deadline; None means "bounded only by the caller".
        self.deadline = (time.monotonic() + timeout) if timeout else None
        self.cancel_reason = ""

    # ── state ──
    @property
    def cancelled(self):
        return self._cancel.is_set()

    @property
    def paused(self):
        return self._paused.is_set() and not self._cancel.is_set()

    def remaining(self):
        """Seconds left on the deadline, or None when unbounded."""
        if self.deadline is None:
            return None
        return self.deadline - time.monotonic()

    def expired(self):
        return self.deadline is not None and time.monotonic() >= self.deadline

    def elapsed(self):
        return time.time() - self.created_at

    # ── control ──
    def cancel(self, reason="cancelled by user"):
        """Cancel the job and terminate every process it owns."""
        self.cancel_reason = reason
        self._cancel.set()
        # A cancelled job must not stay paused: nobody would ever resume it.
        self._paused.clear()
        self.terminate_processes()
        # [P1-13] …and close what a blocked read is waiting on. The event alone
        # cannot interrupt a read that is already in progress.
        self.close_resources()

    def pause(self):
        if not self._cancel.is_set():
            self._paused.set()

    def resume(self):
        self._paused.clear()

    def wait_if_paused(self, timeout=300.0):
        """Block while the job is paused.

        Returns False when the job was cancelled OR when its deadline passed
        while it waited (F20): a pause must not outlive the job's own deadline,
        otherwise pausing is a way to resume effects after the budget expired.
        """
        while self._paused.is_set() and not self._cancel.is_set():
            if self.expired():
                self._paused.clear()
                return False
            self._paused.wait(timeout=0.25)
        return not self._cancel.is_set() and not self.expired()

    def checkpoint(self, timeout=300.0):
        """Pause/cancel/deadline checkpoint.

        Call this before every tool, retry and phase. Raises :class:`Cancelled`
        when the job was cancelled or its deadline passed, and blocks while it
        is paused — re-checking the deadline on the way out, so a job that
        paused past its budget cannot go on to resume effects.
        """
        if self._cancel.is_set():
            raise Cancelled(self.cancel_reason or "cancelled")
        if self.expired():
            raise Cancelled("deadline exceeded")
        if not self.wait_if_paused(timeout=timeout):
            raise Cancelled(self.cancel_reason or "deadline exceeded")
        if self._cancel.is_set():
            raise Cancelled(self.cancel_reason or "cancelled")
        return True

    def should_stop(self):
        """Non-raising variant for loops that prefer to return a result."""
        return self._cancel.is_set() or self.expired()

    # ── F24: one budget handle to propagate ──
    @property
    def cancel_event(self):
        """The job's cancellation event, so a budget handle can share it."""
        return self._cancel

    def budget(self):
        """This job's deadline + cancellation as ONE shared handle (F24).

        Transports take a budget instead of a timeout: the absolute expiry
        travels with the cancellation, so classification, retries, streams
        and tools all read the same clock and the same stop signal.
        """
        return Deadline(self.deadline, cancel=self._cancel,
                        reason=self.cancel_reason or "deadline exceeded")

    # ── owned processes ──
    def attach(self, popen):
        """Take ownership of a Popen so cancellation kills its whole tree."""
        if popen is None:
            return popen
        with self._proc_lock:
            self._processes.append(popen)
        if self._cancel.is_set():
            _terminate(popen)
        return popen

    def terminate_processes(self):
        with self._proc_lock:
            procs = list(self._processes)
            self._processes = []
        for proc in procs:
            _terminate(proc)

    def finish(self):
        global _current_id
        self._finished = True
        with _lock:
            _jobs.pop(self.job_id, None)
            if _current_id == self.job_id:
                _current_id = None

    # ── P1-13: resources a cancellation must close NOW ──
    def register_closer(self, closer):
        """Register ``closer()`` to run the moment this job is cancelled.

        A blocked ``read()`` on a provider socket cannot notice a flag that is
        only checked between lines, so cancelling a turn has to CLOSE the
        socket from the cancelling thread. Clients register their response
        here (see ``openai_compat_client._register_cancel_closer``) and the
        read raises immediately instead of holding the thread.

        If the job is already cancelled the closer runs at once, so a late
        registration can never outlive the cancellation it belongs to.
        """
        if closer is None:
            return None
        already = self._cancel.is_set()
        if not already:
            with self._closer_lock:
                if not self._cancel.is_set():
                    self._closers.append(closer)
                    return closer
        self._run_closer(closer)
        return closer

    def unregister_closer(self, closer):
        """Forget a closer (the client always does this in its ``finally``)."""
        with self._closer_lock:
            try:
                self._closers.remove(closer)
            except ValueError:
                pass

    def close_resources(self):
        """Run and clear every registered closer. Never raises."""
        with self._closer_lock:
            closers = list(self._closers)
            self._closers = []
        for closer in closers:
            self._run_closer(closer)

    @staticmethod
    def _run_closer(closer):
        try:
            closer()
        except Exception as exc:
            logging.debug("[JOBS] closer failed: %s", exc)

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "label": self.label,
            "generation": self.generation,
            "cancelled": self.cancelled,
            "paused": self.paused,
            "elapsed": round(self.elapsed(), 1),
            "remaining": (round(self.remaining(), 1)
                          if self.remaining() is not None else None),
        }


def _terminate(proc):
    """Kill one Popen *and its descendants* (best effort, never raises)."""
    if proc is None:
        return
    try:
        if proc.poll() is not None:
            return
    except Exception:
        return
    try:
        _kill_tree(proc.pid)
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _kill_tree(pid):
    """Terminate a process and all of its descendants (Windows + POSIX)."""
    if not pid:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), 15)
            return
        except Exception:
            pass
    try:
        import signal
        os.kill(pid, getattr(signal, "SIGTERM", 15))
    except Exception:
        pass


# ── registry ───────────────────────────────────────────────────────────────
def new_job(kind="job", timeout=None, label="", generation=None):
    """Create, register and return a fresh JobToken."""
    global _current_id
    job = JobToken("job-%d" % next(_counter), kind=kind, timeout=timeout,
                   label=label, generation=generation)
    with _lock:
        _jobs[job.job_id] = job
        _current_id = job.job_id
    return job


def new_effect_job(kind, label="", timeout=None, generation=None):
    """Create a job for one typed EFFECT (F50).

    Refuses an untyped kind: an effect that cannot name its surface
    (typed / voice / background / setup) cannot be cancelled, attributed or
    journalled consistently, so it must not run through the job runtime.
    """
    if kind not in EFFECT_KINDS:
        raise UnknownEffectKind(
            "untyped effect kind %r; expected one of %s"
            % (kind, ", ".join(sorted(EFFECT_KINDS))))
    return new_job(kind=kind, timeout=timeout, label=label or kind,
                   generation=generation)


def get_job(job_id):
    if not job_id:
        return None
    with _lock:
        return _jobs.get(job_id)


def current_job():
    """The job an unaddressed stop should hit (None when nothing is running)."""
    with _lock:
        job = _jobs.get(_current_id) if _current_id else None
    return job


def live_jobs(kind=None, exclude_kinds=None):
    """Live jobs, optionally limited to one *kind* or a collection of kinds.

    [P1-11] *exclude_kinds* is how an UNADDRESSED stop keeps its hands off chat
    requests (see :func:`request_stop`): "stop" means the thing doing work for
    me, never the conversation I am having.
    """
    with _lock:
        jobs = list(_jobs.values())
    if kind:
        wanted = (kind,) if isinstance(kind, str) else tuple(kind)
        jobs = [j for j in jobs if j.kind in wanted]
    if exclude_kinds:
        unwanted = ((exclude_kinds,) if isinstance(exclude_kinds, str)
                    else tuple(exclude_kinds))
        jobs = [j for j in jobs if j.kind not in unwanted]
    return jobs


def newest_job(kind=None, exclude_kinds=None):
    """The most recently created live job (optionally of one *kind*).

    Ordered by the creation sequence, not by wall-clock time or by job id:
    two jobs created in the same clock tick used to tie, and the id comparison
    is lexicographic ("job-9" > "job-10"), so the unaddressed stop could
    cancel the OLDER job.
    """
    jobs = live_jobs(kind, exclude_kinds)
    if not jobs:
        return None
    return max(jobs, key=lambda j: j.seq)


def cancel_job(job_id=None, reason="cancelled by user"):
    """Cancel one job by id, or the current job when no id is given.

    Returns the list of cancelled job ids. An unaddressed stop with nothing
    running cancels nothing — it must not leave a flag set for the next job.
    """
    global _current_id
    with _lock:
        if job_id:
            job = _jobs.get(job_id)
            targets = [job] if job is not None else []
        else:
            job = _jobs.get(_current_id) if _current_id else None
            if job is None:
                # F20: an unaddressed stop belongs to ONE job — the newest one
                # still running. Cancelling every registered request here was
                # exactly how "stop A" also killed B.
                job = max(_jobs.values(), key=lambda j: j.seq) if _jobs else None
            targets = [job] if job is not None else []
        cancelled = []
        for target in targets:
            if target is None or target.cancelled:
                continue
            target.cancel(reason)
            cancelled.append(target.job_id)
            for handler in _stop_handlers.get(target.kind, []):
                try:
                    handler(target)
                except Exception as exc:
                    logging.debug("[JOBS] stop handler failed: %s", exc)
        if not job_id and not cancelled:
            logging.info("[JOBS] stop requested with no live job — no-op")
        if _current_id in cancelled:
            _current_id = None
        return cancelled


def request_stop(job_id=None, reason="cancelled by user", kinds=None,
                 exclude_kinds=None):
    """Stop by id (or the newest matching job), optionally limited to *kinds*.

    F20: without an id this hits exactly ONE job — the newest still-running one
    whose kind matches. It used to iterate every matching job, so a single
    "stop" interrupted unrelated work.

    [P1-11] An UNADDRESSED stop must not pick a ``request`` job: a bare
    ``/task/stop`` means "stop the task doing work for me", and with more than
    one thing running (P0-08) the newest job of ANY kind is frequently the chat
    request the user is reading. Callers that really mean "stop this chat" name
    the job id or pass ``kinds=("request",)`` explicitly.
    """
    with _lock:
        if job_id:
            return cancel_job(job_id, reason)
        target = newest_job(kinds, exclude_kinds)
    if target is None:
        return []
    return cancel_job(target.job_id, reason)


# ── F20: the job that owns the turn being processed on THIS thread ─────────
# Thread-local, so concurrent requests (a typed stream and a voice request)
# never cancel each other's work. The transport parks its token here; the
# composite/synthesis phases and the task-step loop checkpoint against it
# instead of a global flag.
_TURN_JOB = threading.local()


class TurnCancelled(Exception):
    """Raised at a phase checkpoint when the owning job was cancelled."""


def bind_turn_job(job):
    """Park *job* as this thread's turn job; returns the previous value."""
    previous = getattr(_TURN_JOB, "job", None)
    _TURN_JOB.job = job
    return previous


def unbind_turn_job(previous):
    _TURN_JOB.job = previous


def current_turn_job():
    """The job owning this thread's turn, or None (jobless/background work)."""
    return getattr(_TURN_JOB, "job", None)


def checkpoint_turn(stage="phase"):
    """F20 checkpoint for composite/synthesis phases and effect loops.

    Asks the OWNING JOB whether to continue, not a global flag — and a jobless
    caller is never cancelled here. Raises :class:`TurnCancelled` so the caller
    can stop without running the next effect.
    """
    job = current_turn_job()
    if job is None:
        return True
    try:
        job.checkpoint()
    except Cancelled as exc:
        raise TurnCancelled(str(exc) or "cancelled") from exc
    return True


def turn_cancelled():
    """True when the owning job was cancelled (non-raising form)."""
    job = current_turn_job()
    if job is None:
        return False
    try:
        return bool(job.cancelled)
    except Exception:
        return False


def turn_budget():
    """The budget handle of the job owning this thread's turn (F24).

    ``None`` for jobless/background work — callers then keep their own
    explicit deadline rather than silently inheriting one. Everything that
    runs inside a turn can read this and spend from the SAME window.
    """
    job = current_turn_job()
    if job is None:
        return None
    try:
        return job.budget()
    except Exception:
        return None
