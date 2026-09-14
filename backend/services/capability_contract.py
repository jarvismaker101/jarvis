"""F16 — immutable capability/executor contract carried through consent.

Audit F16 ("Dispatch by Capability, Not One Global Engine"):

    Correction: Carry required capability, selected executor, availability,
    and explicit grant through immutable approval/execution; fail closed on
    incompatible fallback.

``capability_resolver.resolve_engine`` is a pure DECISION function: it can be
logged, printed, and then ignored by execution — which is exactly the defect
the audit found (``brain.py::_execute_deferred_opencode`` still branches on
the live ``config.TASK_ENGINE`` value, so a configuration change after the
user consented silently changes the executor).

This module turns a decision into an :class:`ExecutionContract`: a frozen
record of

  * the **required capability** (coding / editor / browser / generic),
  * the **selected executor** (frozen at consent time),
  * the **availability facts** that were true when the choice was made,
  * the **explicit grant** (the user consent that authorizes the executor),

plus a content digest. Executors verify the contract instead of re-reading
configuration, so:

  * a consent given for one executor can never execute on another one;
  * a configuration change after consent cannot change the executor;
  * a contract that failed closed (no compatible executor) authorizes nothing.

Pure stdlib; no connector imports. The ledger is a bounded in-memory audit
trail of every dispatch decision (never a source of authority — authority
comes only from the frozen contract itself).
"""

import hashlib
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Tuple

#: Availability facts a contract may carry (the resolver's inputs).
AVAILABILITY_FACTS = ("opencode", "editor", "code_tools", "browser")

#: The blocked sentinel mirrors capability_resolver.BLOCKED without importing
#: it (keeps this module import-free of the resolver).
BLOCKED = "blocked"

#: Ledger bound: the audit trail is diagnostic, never unbounded memory.
LEDGER_LIMIT = 256


class ContractViolation(Exception):
    """Raised when an execution does not match its frozen contract."""


def compute_digest(command, capability, executor, availability, grant, opt_in):
    """Content digest of every authority-bearing field of a contract."""
    payload = json.dumps(
        {
            "command": str(command or ""),
            "capability": str(capability or ""),
            "executor": str(executor or ""),
            "availability": [[str(k), bool(v)] for k, v in (availability or ())],
            "grant": str(grant or ""),
            "opt_in": bool(opt_in),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutionContract:
    """One immutable capability dispatch decision, safe to hand to execution."""

    command: str
    capability: str
    executor: str
    availability: Tuple[Tuple[str, bool], ...] = ()
    grant: str = ""
    opt_in: bool = False
    reason: str = ""
    created_at: float = field(default_factory=time.time)
    digest: str = ""

    # ── construction helpers ───────────────────────────────────────────────
    def as_dict(self):
        return {
            "command": self.command,
            "capability": self.capability,
            "executor": self.executor,
            "availability": {k: v for k, v in self.availability},
            "grant": self.grant,
            "opt_in": self.opt_in,
            "reason": self.reason,
            "created_at": self.created_at,
            "digest": self.digest,
        }

    def granted(self, grant_id):
        """Return a NEW contract carrying *grant_id* (the explicit consent).

        Immutability is preserved: the original contract is untouched, and
        the digest is recomputed so the granted copy verifies.
        """
        grant_id = str(grant_id or "")
        return replace(
            self,
            grant=grant_id,
            digest=compute_digest(
                self.command, self.capability, self.executor,
                self.availability, grant_id, self.opt_in,
            ),
        )

    # ── verification ───────────────────────────────────────────────────────
    def authorizes(self, executor=None, capability=None, require_grant=True):
        """True when this contract allows *executor*; never raises."""
        try:
            self.verify(executor=executor, capability=capability,
                        require_grant=require_grant)
            return True
        except ContractViolation:
            return False

    def verify(self, executor=None, capability=None, require_grant=True):
        """Fail closed unless this contract authorizes the execution.

        Raises :class:`ContractViolation` when
          * any authority-bearing field was altered after the fact (digest
            mismatch),
          * the executor/capability asked to run is not the frozen one,
          * the decision failed closed (:data:`BLOCKED`),
          * no explicit grant was ever attached.
        """
        expected = compute_digest(
            self.command, self.capability, self.executor,
            self.availability, self.grant, self.opt_in,
        )
        if expected != self.digest:
            raise ContractViolation(
                "contract digest mismatch: authority fields were altered "
                "after consent")
        if self.executor == BLOCKED:
            raise ContractViolation(
                "capability dispatch failed closed (%s): no compatible "
                "executor" % (self.reason or "no reason recorded"))
        if executor is not None and executor != self.executor:
            raise ContractViolation(
                "executor %r does not match the consented executor %r"
                % (executor, self.executor))
        if capability is not None and capability != self.capability:
            raise ContractViolation(
                "capability %r does not match the consented capability %r"
                % (capability, self.capability))
        if require_grant and not self.grant:
            raise ContractViolation(
                "no explicit grant: capability execution requires consent")
        return True


def freeze(decision, command, grant=""):
    """Build a contract from a resolver decision dict.

    *decision* is the mapping returned by
    ``capability_resolver.resolve_engine`` (engine / reason / opt_in /
    capability / availability / blocked).
    """
    decision = dict(decision or {})
    availability = tuple(sorted(
        (str(k), bool(v))
        for k, v in dict(decision.get("availability") or {}).items()
    ))
    executor = str(decision.get("engine") or "")
    if decision.get("blocked"):
        executor = BLOCKED
    capability = str(decision.get("capability") or "generic")
    opt_in = bool(decision.get("opt_in"))
    grant = str(grant or "")
    return ExecutionContract(
        command=str(command or ""),
        capability=capability,
        executor=executor,
        availability=availability,
        grant=grant,
        opt_in=opt_in,
        reason=str(decision.get("reason") or ""),
        created_at=time.time(),
        digest=compute_digest(
            command, capability, executor, availability, grant, opt_in),
    )


# ── audit ledger (diagnostic only — never a source of authority) ────────────
_ledger_lock = threading.Lock()
_ledger = deque(maxlen=LEDGER_LIMIT)


def record(contract):
    """Append *contract* to the dispatch ledger and return it."""
    with _ledger_lock:
        _ledger.append(contract)
    return contract


def ledger():
    """Every recorded contract, oldest first (a copy)."""
    with _ledger_lock:
        return list(_ledger)


def latest():
    """The most recently recorded contract, or None."""
    with _ledger_lock:
        return _ledger[-1] if _ledger else None


def clear():
    """Drop the ledger (tests / a supervisor reset)."""
    with _ledger_lock:
        _ledger.clear()


def contract_for(decision, command, grant=""):
    """:func:`freeze` + :func:`record` in one step."""
    return record(freeze(decision, command, grant=grant))
