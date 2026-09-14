"""Plan-bound approval records.

Fable-5 audit G2 / F18 — bind consent to the entire plan.

Before this module a pending screen plan was a bare dict with no expiry and
no identity, the confirmation preview showed only the first sensitive step,
and "no, don't do it" matched the *do it* pattern and executed anyway.

An :class:`ApprovalRecord` fixes all three:

  * ``plan_hash``  — the approval is bound to the *whole* plan. If follow-up
    input materially changes the operation (different steps, targets or
    command) the hash no longer matches and reapproval is required.
  * ``expires_at`` — a stale approval cannot be woken up minutes later by an
    unrelated "yes".
  * ``scope``/``targets`` — what the approval actually authorises.
  * ``generation`` — a screen-control generation token; turning screen
    control off (and back on) invalidates every outstanding approval.

Negation is resolved first: an explicit negative can never be read as consent.
"""

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.services import tool_policy

_DEFAULT_TTL_SECONDS = 120.0

_YES = "yes"
_NO = "no"


# ── Negation-aware verdict grammar ─────────────────────────────────────────
# Checked in order; the FIRST match wins. Negations deliberately come before
# affirmations so "no, don't do it" is never accepted as "do it".
_NEGATION_RE = re.compile(
    r"\b(no|nope|nah|never|cancel|abort|stop that|don'?t(?: do it)?|do not|"
    r"forget it|never\s*mind|mat karo|rehne do|nahi)\b",
    re.IGNORECASE,
)

_AFFIRM_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|confirm|confirmed|do it|go ahead|proceed|"
    r"execute|run it|ok(?:ay)?|haan|han|kar do|kar de|theek hai)\b",
    re.IGNORECASE,
)

_REAFFIRM_RE = re.compile(
    r"\bconfirm\s+screen\s+action\b|\bconfirm\s+(?:the\s+)?(?:screen\s+)?action\b",
    re.IGNORECASE,
)


def verdict(text):
    """Return 'yes', 'no' or None for one user utterance.

    Negation wins: an utterance containing both a negation and an
    affirmation ("no ... do it" aside) resolves to 'no'.
    """
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    if _NEGATION_RE.search(raw):
        return _NO
    if _REAFFIRM_RE.search(raw) or _AFFIRM_RE.search(raw):
        return _YES
    return None


# ── Plan identity ──────────────────────────────────────────────────────────
def canonical_plan(plan, command_text=""):
    """The parts of a plan that consent is actually bound to."""
    plan = plan if isinstance(plan, dict) else {}
    steps = []
    for step in plan.get("steps") or []:
        if not isinstance(step, dict):
            steps.append(str(step))
            continue
        steps.append({k: v for k, v in sorted(step.items(), key=lambda kv: kv[0])})
    return {
        "command": str(command_text or plan.get("command_text") or "").strip().lower(),
        "steps": steps,
        "hwnd": plan.get("hwnd"),
        "capture_mode": plan.get("capture_mode"),
    }


def plan_hash(plan, command_text=""):
    """Stable sha256 of the plan's effects — the identity consent is bound to."""
    try:
        blob = json.dumps(canonical_plan(plan, command_text),
                          sort_keys=True, default=str)
    except Exception:
        blob = repr(canonical_plan(plan, command_text))
    return hashlib.sha256(blob.encode("utf-8", "replace")).hexdigest()


@dataclass
class ApprovalRecord:
    """One outstanding request for consent."""

    id: str
    plan_hash: str
    command: str
    effects: List[str] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    scope: str = "screen"
    created_at: float = 0.0
    expires_at: float = 0.0
    generation: int = 0
    #: Full preview text for the UI (every effect, not just the first).
    preview: str = ""
    #: Short spoken summary with action counts.
    spoken: str = ""
    #: The approved plan itself, kept so it cannot drift after approval.
    plan: Optional[Dict[str, Any]] = None

    def is_expired(self, now=None):
        return (now or time.time()) >= self.expires_at

    def remaining(self, now=None):
        return max(0.0, self.expires_at - (now or time.time()))

    def to_dict(self):
        return {
            "id": self.id,
            "plan_hash": self.plan_hash,
            "command": self.command,
            "effects": list(self.effects),
            "targets": list(self.targets),
            "scope": self.scope,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "expires_in": round(self.remaining(), 1),
            "generation": self.generation,
            "preview": self.preview,
            "spoken": self.spoken,
        }


def build_record(plan, command_text="", scope="screen", ttl=None,
                 generation=0, record_id=None):
    """Build an ApprovalRecord describing exactly what is being authorized."""
    effects = tool_policy.describe_plan(plan)
    targets = []
    for step in (plan or {}).get("steps") or []:
        if isinstance(step, dict) and step.get("hwnd"):
            targets.append("hwnd:%s" % step["hwnd"])
    if isinstance(plan, dict) and plan.get("hwnd"):
        targets.append("hwnd:%s" % plan["hwnd"])
    if isinstance(plan, dict) and plan.get("capture_mode"):
        targets.append("capture:%s" % plan["capture_mode"])
    targets = sorted(set(str(t) for t in targets))
    ttl = _DEFAULT_TTL_SECONDS if ttl is None else ttl
    now = time.time()
    record = ApprovalRecord(
        id=record_id or hashlib.sha1(
            ("%s|%s" % (plan_hash(plan, command_text), now)).encode()
        ).hexdigest()[:12],
        plan_hash=plan_hash(plan, command_text),
        command=str(command_text or ""),
        effects=effects,
        targets=targets,
        scope=scope,
        created_at=now,
        expires_at=now + ttl,
        generation=generation,
        plan=plan,
    )
    record.preview = preview_text(record)
    record.spoken = spoken_text(record)
    return record


def preview_text(record):
    """Full, ordered preview for the UI — every effect, not just the first."""
    lines = ["Command: %s" % (record.command or "(pending command)")]
    if record.targets:
        lines.append("Targets: %s" % ", ".join(record.targets))
    lines.append("Actions (%d):" % len(record.effects))
    for i, effect in enumerate(record.effects, 1):
        lines.append("  %d. %s" % (i, effect))
    lines.append("Expires in %ds." % int(record.remaining()))
    return "\n".join(lines)


def spoken_text(record):
    """Concise spoken summary: what it is and how many of each action."""
    return ("%s — %s. Say confirm screen action to proceed, or say cancel."
            % (record.command or "that action",
               tool_policy.effect_counts(record.effects)))


# ── The single outstanding approval ────────────────────────────────────────
_lock = threading.RLock()
_pending: Optional[ApprovalRecord] = None
_last_cancelled: Optional[ApprovalRecord] = None


def arm(plan, command_text="", scope="screen", ttl=None, generation=0):
    """Register a plan as awaiting consent; returns the record."""
    global _pending
    record = build_record(plan, command_text, scope=scope, ttl=ttl,
                          generation=generation)
    with _lock:
        _pending = record
    return record


def pending(now=None):
    """The outstanding approval, or None when there is none (incl. expired)."""
    global _pending
    with _lock:
        record = _pending
        if record is not None and record.is_expired(now):
            _pending = None
            return None
        return record


def take(now=None):
    """Pop the outstanding approval if it is still valid."""
    global _pending
    with _lock:
        record = pending(now)
        _pending = None
        return record


def clear():
    global _pending
    with _lock:
        record = _pending
        _pending = None
        return record


def cancel(reason="cancelled"):
    """Consume the outstanding approval *without* authorising it."""
    global _last_cancelled
    record = clear()
    if record is not None:
        _last_cancelled = record
    return record


def invalidate(reason="invalidated"):
    """G11 / F52 — drop every outstanding approval on worker replacement.

    A supervisor that tears a worker down (or replaces it) must not leave a
    plan awaiting consent that the NEW worker could consume. This clears the
    pending record, mirrors it into ``_last_cancelled`` so the UI can show
    what was dropped, and returns the discarded record (or None).
    """
    record = cancel("invalidated: %s" % (reason,))
    if record is not None:
        print("[APPROVALS] invalidated on %s: %s" % (reason, record.command))
    return record


def last_cancelled():
    with _lock:
        return _last_cancelled


def verify(plan, record, command_text=None, generation=None):
    """Check a plan against the approval that is supposed to cover it.

    Returns ``(ok, reason)``. Reapproval is required when the plan's identity
    changed, the approval expired, or the screen-control generation moved on.
    """
    if record is None:
        return False, "no approval on record"
    if record.is_expired():
        return False, "the approval expired"
    if generation is not None and record.generation != generation:
        return False, "screen controls were re-enabled since the approval"
    if command_text is None:
        command_text = record.command
    if plan_hash(plan, command_text) != record.plan_hash:
        return False, "the plan changed since it was approved"
    return True, ""


def state():
    record = pending()
    if record is None:
        return {"screen_approval_pending": False}
    payload = {"screen_approval_pending": True}
    payload.update(record.to_dict())
    return payload
