"""F50 — the backend as the sole owner of intelligence state.

Audit F50 ("Give the Backend Sole Ownership of Intelligence State"):

    Correction: All effects through one job runtime, one explicit playback
    owner/reference transport, typed worker-generation state, and centralized
    transactional events/approvals/checkpoints.

The defect: the voice I/O worker ran local effects itself (``os.startfile``
straight from the voice process), published state with no worker incarnation
or expiry (a restart could leave NEW state behind the OLD sequence), and each
surface kept its own gate while several processes could write the durable
store.

This module is the backend-side authority layer that fixes it:

* :func:`run_effect` / :func:`submit_effect` — ONE way for every effect
  (typed, voice, background, setup) to run: a typed backend job from
  :mod:`backend.services.jobs`, bound to the calling thread so F20
  cancellation/deadlines apply, journalled before and after it runs.
* :class:`WorkerRegistry` — typed worker-generation state. A newly started
  worker SUPERSEDES older generations immediately; a publish from a stale
  generation is rejected, and a state whose incarnation expired is not
  returned at all.
* :class:`DurableOwnership` — only the DESIGNATED owner of a durable
  resource may schedule/write it. A second live owner is refused; an expired
  lease can be taken over.
* :class:`EventJournal` — one transactional event history (effects,
  checkpoints, approvals) shared by every surface, so typed/voice/background
  work shares authority and history instead of keeping private logs.

Pure stdlib; only :mod:`backend.services.jobs` is imported.
"""

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple

from backend.services import jobs

# ── Roles / resources ──────────────────────────────────────────────────────
#: The single process that owns speech playback (F35/F50). The voice I/O
#: worker is a reference transport, never a second playback owner.
PLAYBACK_RESOURCE = "playback"
ROLE_VOICE = "voice-io"
ROLE_BACKEND = "backend"
ROLE_TYPED = "typed"
ROLE_BACKGROUND = "background"

#: Durable resources with exactly one writer.
RESOURCE_MEMORY = "memory.db"
RESOURCE_CHECKPOINTS = "checkpoints"
RESOURCE_APPROVALS = "approvals"

DEFAULT_TTL = 3600.0


def _ttl_seconds(ttl):
    """Lease lifetime in seconds (None -> default; 0/negative -> expired)."""
    return DEFAULT_TTL if ttl is None else float(ttl)


class NotDurableOwner(Exception):
    """Raised when a non-designated surface tries to write durable state."""


class StaleGeneration(Exception):
    """Raised when a superseded worker generation tries to publish state."""


# ── Typed worker-generation state ──────────────────────────────────────────
@dataclass(frozen=True)
class WorkerState:
    """One published state of one worker incarnation (F50)."""

    role: str
    owner: str            # incarnation label, e.g. "voice-io@4242#7"
    generation: int
    pid: int
    state: dict
    issued_at: float
    expires_at: float

    def expired(self, now=None):
        now = time.time() if now is None else now
        return now >= self.expires_at

    def to_dict(self):
        return {
            "role": self.role,
            "owner": self.owner,
            "generation": self.generation,
            "pid": self.pid,
            "state": dict(self.state or {}),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "expired": self.expired(),
        }


@dataclass(frozen=True)
class PublishOutcome:
    """Result of a state publish: authority is explicit, never implied."""

    accepted: bool
    stale: bool
    generation: int
    owner: str
    reason: str = ""

    def to_dict(self):
        return {
            "accepted": self.accepted,
            "stale": self.stale,
            "generation": self.generation,
            "owner": self.owner,
            "reason": self.reason,
        }


class WorkerRegistry:
    """Typed state per worker role, with incarnations and generations."""

    def __init__(self):
        self._lock = threading.RLock()
        self._states: Dict[str, WorkerState] = {}
        self._generations: Dict[str, int] = {}
        self._owners: Dict[str, str] = {}

    def generation(self, role):
        with self._lock:
            return int(self._generations.get(role, 0))

    def owner(self, role):
        with self._lock:
            return self._owners.get(role, "")

    def incarnation(self, role):
        """``"<owner>#<generation>"`` of the current worker, or ""."""
        with self._lock:
            owner = self._owners.get(role)
            if not owner:
                return ""
            return "%s#%d" % (owner, self._generations.get(role, 0))

    def supersede(self, role, owner, pid=0, ttl=DEFAULT_TTL, state=None):
        """A NEW worker starts: its generation immediately supersedes older ones.

        Any state published by the previous incarnation is dropped in the
        same critical section, so no later reader can see the old worker's
        truth once the new worker exists (F50 acceptance: "new worker state
        supersedes old generations immediately").
        """
        owner = str(owner or "").strip()
        if not owner:
            raise ValueError("a worker incarnation needs an owner label")
        with self._lock:
            generation = int(self._generations.get(role, 0)) + 1
            self._generations[role] = generation
            self._owners[role] = owner
            if state is None:
                self._states.pop(role, None)
                return self.incarnation(role)
            return self._store(role, owner, generation, pid, state, ttl)

    def publish(self, role, owner, state, pid=0, ttl=DEFAULT_TTL,
                generation=None):
        """Publish state for the CURRENT incarnation of *role*.

        Returns a :class:`PublishOutcome`. A publish from a superseded
        generation (or from an owner that is not the current incarnation) is
        REJECTED and marked ``stale`` instead of overwriting newer truth.
        """
        owner = str(owner or "").strip()
        with self._lock:
            current_gen = int(self._generations.get(role, 0))
            current_owner = self._owners.get(role, "")
            if current_owner == "":
                # No incarnation registered yet: first publisher becomes it.
                current_gen = current_gen or 1
                self._generations[role] = current_gen
                self._owners[role] = owner
            elif owner != current_owner:
                return PublishOutcome(
                    accepted=False, stale=True,
                    generation=self._generations.get(role, 0), owner=owner,
                    reason="publisher %r is not the current incarnation %r"
                           % (owner, current_owner))
            if generation is not None and int(generation) < current_gen:
                return PublishOutcome(
                    accepted=False, stale=True, generation=current_gen,
                    owner=owner,
                    reason="generation %s superseded by %s"
                           % (generation, current_gen))
            stored = self._store(role, owner, current_gen, pid, state, ttl)
            return PublishOutcome(accepted=True, stale=False,
                                  generation=stored.generation, owner=owner)

    def _store(self, role, owner, generation, pid, state, ttl):
        now = time.time()
        stored = WorkerState(
            role=role, owner=owner, generation=int(generation),
            pid=int(pid or 0), state=dict(state or {}),
            issued_at=now, expires_at=now + _ttl_seconds(ttl),
        )
        self._states[role] = stored
        return stored

    def current(self, role):
        """The live state for *role*, or None when absent/expired (F50)."""
        with self._lock:
            stored = self._states.get(role)
        if stored is None or stored.expired():
            if stored is not None:
                with self._lock:
                    if self._states.get(role) is stored:
                        self._states.pop(role, None)
            return None
        return stored

    def clear(self, role=None):
        with self._lock:
            if role is None:
                self._states.clear()
                self._generations.clear()
                self._owners.clear()
            else:
                self._states.pop(role, None)
                self._generations.pop(role, None)
                self._owners.pop(role, None)


# ── Durable-state ownership ────────────────────────────────────────────────
class DurableOwnership:
    """Exactly one designated writer/scheduler per durable resource (F50).

    ``claim`` succeeds only for an unowned, expired, or already-owned lease;
    a second LIVE owner is refused with :class:`NotDurableOwner`. Writers
    call :meth:`assert_writer` (or :func:`run_effect` with a ``resource``),
    so a surface that merely *has* the resource path still cannot write it.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._leases: Dict[str, Tuple[str, float]] = {}

    def claim(self, resource, owner, ttl=DEFAULT_TTL):
        """Designate *owner* as THE writer of *resource*; returns True."""
        owner = str(owner or "").strip()
        if not owner:
            raise ValueError("a durable owner label is required")
        now = time.time()
        with self._lock:
            current = self._leases.get(resource)
            if current is not None:
                current_owner, expires_at = current
                if current_owner != owner and now < expires_at:
                    raise NotDurableOwner(
                        "%s is owned by %r until %.0f" %
                        (resource, current_owner, expires_at))
            self._leases[resource] = (owner, now + _ttl_seconds(ttl))
            return True

    def owner_of(self, resource):
        with self._lock:
            current = self._leases.get(resource)
        if current is None:
            return ""
        owner, expires_at = current
        if time.time() >= expires_at:
            return ""
        return owner

    def writer_allowed(self, resource, owner):
        current = self.owner_of(resource)
        return current == "" or current == str(owner or "").strip()

    def assert_writer(self, resource, owner):
        """Fail closed unless *owner* is the designated writer of *resource*."""
        current = self.owner_of(resource)
        owner = str(owner or "").strip()
        if current and current != owner:
            raise NotDurableOwner(
                "%s is written by %r, not %r" % (resource, current, owner))
        return True

    def release(self, resource, owner):
        with self._lock:
            current = self._leases.get(resource)
            if current is not None and current[0] == str(owner or "").strip():
                self._leases.pop(resource, None)
                return True
        return False

    def clear(self):
        with self._lock:
            self._leases.clear()


# ── Transactional event history ────────────────────────────────────────────
@dataclass(frozen=True)
class WorkEvent:
    seq: int
    at: float
    kind: str          # typed | voice | background | setup
    phase: str         # submitted | started | finished | failed | stopped | checkpoint | approval
    effect: str
    job_id: str
    authority: str
    detail: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "seq": self.seq,
            "at": self.at,
            "kind": self.kind,
            "phase": self.phase,
            "effect": self.effect,
            "job_id": self.job_id,
            "authority": self.authority,
            "detail": dict(self.detail or {}),
        }


class EventJournal:
    """ONE shared, sequenced history of effects/approvals/checkpoints."""

    def __init__(self, limit=1024):
        self._lock = threading.RLock()
        self._seq = itertools.count(1)
        self._events = deque(maxlen=limit)
        self._buffer = None

    def append(self, kind, phase, effect, job_id="", authority="", detail=None):
        event = WorkEvent(
            seq=next(self._seq), at=time.time(), kind=kind, phase=phase,
            effect=effect, job_id=job_id, authority=authority,
            detail=dict(detail or {}),
        )
        with self._lock:
            if self._buffer is not None:
                self._buffer.append(event)
            else:
                self._events.append(event)
        return event

    class _Transaction:
        def __init__(self, journal):
            self._journal = journal

        def __enter__(self):
            journal = self._journal
            with journal._lock:
                if journal._buffer is not None:
                    raise RuntimeError("nested journal transaction")
                journal._buffer = []
            return self

        def __exit__(self, exc_type, exc, tb):
            journal = self._journal
            with journal._lock:
                buffered = journal._buffer or []
                journal._buffer = None
                if exc_type is None:
                    # Commit atomically under one lock: readers see all of the
                    # events or none of them.
                    for event in buffered:
                        journal._events.append(event)
            return False

    def transaction(self):
        """All-or-nothing append block (F50: transactional events)."""
        return EventJournal._Transaction(self)

    def events(self, kind=None, phase=None, limit=None):
        with self._lock:
            items = list(self._events)
        if kind:
            items = [e for e in items if e.kind == kind]
        if phase:
            items = [e for e in items if e.phase == phase]
        if limit:
            items = items[-int(limit):]
        return items

    def clear(self):
        with self._lock:
            self._events.clear()


# ── The one effect runtime ─────────────────────────────────────────────────
@dataclass(frozen=True)
class EffectOutcome:
    job_id: str
    kind: str
    effect: str
    status: str          # completed | failed | stopped
    result: object = None
    error: str = ""
    authority: str = ""
    events: Tuple[WorkEvent, ...] = ()

    def to_dict(self):
        return {
            "job_id": self.job_id,
            "kind": self.kind,
            "effect": self.effect,
            "status": self.status,
            "error": self.error,
            "authority": self.authority,
            "events": [e.to_dict() for e in self.events],
        }


#: Process-wide singletons — ONE registry/journal/ownership per backend.
worker_states = WorkerRegistry()
durable = DurableOwnership()
journal = EventJournal()


def _authority_label(owner=None):
    owner = (owner or ROLE_BACKEND)
    return "%s#%d" % (owner, worker_states.generation(owner)) \
        if worker_states.generation(owner) else str(owner)


def run_effect(kind, effect, handler, label="", timeout=None, owner=None,
               resource=None, require_owner=True):
    """Run ONE effect through the one job runtime (F50).

    *kind* must be one of :data:`backend.services.jobs.EFFECT_KINDS`;
    *handler* receives the :class:`~backend.services.jobs.JobToken` and runs
    on THIS thread with the job bound as the turn job, so F20 checkpoints,
    deadlines and cancellation apply to typed work exactly as to voice and
    background work. Every phase is journalled into the shared history.

    When *resource* is given, the caller must be that durable resource's
    designated writer (fail closed with :class:`NotDurableOwner`).
    """
    if kind not in jobs.EFFECT_KINDS:
        raise jobs.UnknownEffectKind(
            "untyped effect kind %r" % (kind,))
    if resource is not None:
        if require_owner:
            durable.assert_writer(resource, owner or ROLE_BACKEND)
        elif not durable.writer_allowed(resource, owner or ROLE_BACKEND):
            raise NotDurableOwner(
                "%s is written by %r" % (resource, durable.owner_of(resource)))

    authority = _authority_label(owner)
    job = jobs.new_effect_job(
        kind, label=label or effect, timeout=timeout,
        generation=worker_states.generation(owner or ROLE_BACKEND))
    events = [journal.append(kind, "submitted", effect,
                             job_id=job.job_id, authority=authority)]
    previous = jobs.bind_turn_job(job)
    status, result, error = "completed", None, ""
    try:
        events.append(journal.append(kind, "started", effect,
                                     job_id=job.job_id, authority=authority))
        try:
            result = handler(job)
        except jobs.Cancelled as exc:
            status, error = "stopped", str(exc) or "cancelled"
        except Exception as exc:  # noqa: BLE001 - journalled, then re-raised
            status, error = "failed", "%s: %s" % (type(exc).__name__, exc)
            events.append(journal.append(
                kind, "failed", effect, job_id=job.job_id,
                authority=authority, detail={"error": error}))
            raise
    finally:
        jobs.unbind_turn_job(previous)
        job.finish()
        if status == "completed":
            events.append(journal.append(kind, "finished", effect,
                                         job_id=job.job_id,
                                         authority=authority))
        elif status == "stopped":
            events.append(journal.append(kind, "stopped", effect,
                                         job_id=job.job_id,
                                         authority=authority,
                                         detail={"reason": error}))
    return EffectOutcome(
        job_id=job.job_id, kind=kind, effect=effect, status=status,
        result=result, error=error, authority=authority,
        events=tuple(events))


def submit_effect(kind, effect, handler, label="", timeout=None, owner=None,
                  resource=None, require_owner=True):
    """Submit an effect that must NOT block the caller (F50 setup/voice).

    Creates the backend job synchronously — so the caller can report/stop it
    immediately — then runs the handler on a daemon thread through
    :func:`run_effect`'s runtime. Returns the job id.
    """
    if kind not in jobs.EFFECT_KINDS:
        raise jobs.UnknownEffectKind("untyped effect kind %r" % (kind,))
    if resource is not None:
        if require_owner:
            durable.assert_writer(resource, owner or ROLE_BACKEND)
        elif not durable.writer_allowed(resource, owner or ROLE_BACKEND):
            raise NotDurableOwner(
                "%s is written by %r" % (resource, durable.owner_of(resource)))

    authority = _authority_label(owner)
    job = jobs.new_effect_job(
        kind, label=label or effect, timeout=timeout,
        generation=worker_states.generation(owner or ROLE_BACKEND))

    def _run():
        previous = jobs.bind_turn_job(job)
        try:
            journal.append(kind, "started", effect, job_id=job.job_id,
                           authority=authority)
            handler(job)
            journal.append(kind, "finished", effect, job_id=job.job_id,
                           authority=authority)
        except jobs.Cancelled as exc:
            journal.append(kind, "stopped", effect, job_id=job.job_id,
                           authority=authority,
                           detail={"reason": str(exc) or "cancelled"})
        except Exception as exc:  # noqa: BLE001 - backgrounded, journalled
            journal.append(kind, "failed", effect, job_id=job.job_id,
                           authority=authority,
                           detail={"error": "%s: %s" % (type(exc).__name__, exc)})
        finally:
            jobs.unbind_turn_job(previous)
            job.finish()

    thread = threading.Thread(target=_run, name="effect-%s" % effect,
                              daemon=True)
    journal.append(kind, "submitted", effect, job_id=job.job_id,
                   authority=authority)
    thread.start()
    return job


def submit_setup_effect(name, handler, label="", timeout=None, owner=None):
    """Run a local setup command AS a backend job (F50 acceptance).

    The voice process must never launch apps itself: it asks the backend,
    which submits a typed ``setup`` job here. The job id is returned so the
    caller can stop or observe it.
    """
    return submit_effect(jobs.EFFECT_SETUP, name, handler, label=label,
                         timeout=timeout, owner=owner or ROLE_BACKEND)


# ── Playback ownership ─────────────────────────────────────────────────────
def designate_playback_owner(owner, ttl=DEFAULT_TTL):
    """Designate THE playback owner (F50: one explicit playback owner)."""
    if owner is None:
        owner = ROLE_BACKEND
    durable.claim(PLAYBACK_RESOURCE, owner, ttl=ttl)
    return owner


def playback_owner():
    """The designated playback owner, or "" when unowned/expired."""
    return durable.owner_of(PLAYBACK_RESOURCE)


def release_playback_owner(owner):
    return durable.release(PLAYBACK_RESOURCE, owner)


def note_checkpoint(effect, job_id="", authority="", detail=None):
    """Record a checkpoint in the shared history (centralized checkpoints)."""
    return journal.append(ROLE_BACKEND, "checkpoint", effect, job_id=job_id,
                          authority=authority, detail=detail)


def note_approval(effect, job_id="", authority="", detail=None):
    """Record an approval decision in the shared history (F50/F18 bridge)."""
    return journal.append(ROLE_BACKEND, "approval", effect, job_id=job_id,
                          authority=authority, detail=detail)


def note_voice_state(role, owner, state, pid=0, ttl=DEFAULT_TTL,
                     generation=None):
    """Publish typed voice/worker state through the generation registry."""
    outcome = worker_states.publish(role, owner, state, pid=pid, ttl=ttl,
                                    generation=generation)
    journal.append(ROLE_VOICE, "published", "voice-state",
                   authority=owner,
                   detail={"role": role, "accepted": outcome.accepted,
                           "stale": outcome.stale,
                           "generation": outcome.generation})
    return outcome


def reset():
    """Drop all process-wide authority state (tests / supervisor reset)."""
    worker_states.clear()
    durable.clear()
    journal.clear()
