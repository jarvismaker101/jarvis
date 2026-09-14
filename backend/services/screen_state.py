"""Screen-control state: enable flag, control generation and pending consent.

Fable-5 audit G2:
  F19-enforcement — a "control generation" counter backs the enable flag.
    Every screen effect must be guarded by :func:`guard` immediately before it
    happens, and every plan carries the generation it was created under, so
    disabling (and re-enabling) screen control invalidates pending approvals
    and in-flight plans instead of only flipping a UI flag.
  F18 — the pending plan is stored together with an approvals.ApprovalRecord
    (plan hash, every effect, targets, expiry, scope) instead of a bare dict
    with no identity and no expiry.
"""

import contextlib
import threading
import time

from backend.services import approvals
from backend.services import tool_policy


_state_lock = threading.RLock()
_enabled = False
_pending_plan = None
_pending_approval = None
_interaction_history = []
_MAX_HISTORY = 5

#: Bumped every time screen control is enabled or disabled. A plan stamped
#: with an older generation can never execute.
_generation = 0

#: Default lifetime of a pending approval (seconds).
APPROVAL_TTL = 120.0


def is_enabled():
    with _state_lock:
        return _enabled


def generation():
    """Current control-generation token."""
    with _state_lock:
        return _generation


def bump_generation():
    """Invalidate every outstanding plan/approval. Returns the new token."""
    global _generation
    with _state_lock:
        _generation += 1
        return _generation


def set_enabled(enabled):
    """Enable/disable screen control.

    Disabling (or re-enabling) bumps the control generation and drops the
    pending plan and approval: consent granted under the old generation must
    not survive into the new one.
    """
    global _enabled, _pending_plan, _pending_approval
    with _state_lock:
        _enabled = bool(enabled)
        bump_generation()
        if not _enabled:
            _pending_plan = None
            _pending_approval = None
            approvals.cancel("screen control disabled")
        return _enabled


def guard(stamp=None):
    """Authoritative screen-effect gate.

    Returns ``(allowed, reason)``. Call this immediately before every screen
    effect (including connector-driven ones and background plans), not only
    when the request arrives. *stamp* is the generation the plan was created
    under; a mismatch means control was revoked and re-granted meanwhile.
    """
    with _state_lock:
        if not _enabled:
            return False, "screen controls are off"
        if stamp is not None and int(stamp) != _generation:
            return False, ("screen controls were turned off and back on since "
                           "this action was planned")
        return True, ""


@contextlib.contextmanager
def effect_gate(stamp=None):
    """F19: hold the control lock across exactly one screen effect.

    This closes the check-then-act window that :func:`guard` alone leaves open.
    ``set_enabled`` (and :func:`bump_generation`) acquire the same lock, so a
    revocation can never be interleaved *inside* a single effect: it waits for
    the in-flight effect to finish, and the very next effect re-checks the
    revoked state and refuses.

    Raises :class:`PermissionError` when the effect is not authorized.
    """
    with _state_lock:
        if not _enabled:
            raise PermissionError("screen controls are off")
        if stamp is not None and int(stamp) != _generation:
            raise PermissionError("screen controls were restarted since "
                                  "this action was planned")
        yield


def set_pending_plan(plan, command_text="", ttl=None, scope="screen"):
    """Store a plan awaiting consent, bound to an approval record.

    The approval carries the plan hash (so a materially changed operation
    needs reapproval), every effect (so the UI can preview the whole plan),
    the target identities, an expiry and the control generation.
    """
    global _pending_plan, _pending_approval
    with _state_lock:
        _pending_plan = plan
        try:
            _pending_approval = approvals.arm(
                plan, command_text or (plan or {}).get("command_text", ""),
                scope=scope, ttl=ttl if ttl is not None else APPROVAL_TTL,
                generation=_generation,
            )
        except Exception:
            _pending_approval = None
        return _pending_plan


def get_pending_plan():
    with _state_lock:
        return _pending_plan


def get_pending_approval():
    """The outstanding ApprovalRecord (already expiry-checked)."""
    global _pending_approval
    with _state_lock:
        record = approvals.pending()
        if record is None:
            _pending_approval = None
            return None
        _pending_approval = record
        return record


def pop_pending_plan():
    """Return and clear the pending plan together with its approval."""
    global _pending_plan, _pending_approval
    with _state_lock:
        plan = _pending_plan
        _pending_plan = None
        _pending_approval = None
        approvals.clear()
        return plan


def consume_pending_plan():
    """F18: atomically take the pending (plan, approval) pair for execution.

    Verification and consumption used to be two separate reads, so a plan
    could be verified and then replaced (or have its coordinates adjusted)
    before the same approval was consumed and executed. This returns an
    immutable SNAPSHOT of the plan together with the record it was armed
    against, and consumes both under one lock — a plan that fails verification
    afterwards has already been dropped and therefore executes nothing.
    """
    global _pending_plan, _pending_approval
    with _state_lock:
        plan = _pending_plan
        record = _pending_approval
        _pending_plan = None
        _pending_approval = None
        approvals.take()
        if plan is None:
            return None, None
        try:
            import copy

            plan = copy.deepcopy(plan)
        except Exception:
            pass
        # The record travels with the snapshot so verification reads the same
        # bytes the approval was bound to.
        if record is None:
            record = None
        return plan, record


def clear_pending_plan():
    global _pending_plan, _pending_approval
    with _state_lock:
        _pending_plan = None
        _pending_approval = None
        approvals.clear()


def has_pending_plan():
    """True only while a *live* (unexpired) plan is awaiting consent."""
    global _pending_plan, _pending_approval
    with _state_lock:
        if _pending_plan is None:
            return False
        record = approvals.pending()
        if record is None:
            # Expired or cancelled elsewhere: drop it rather than letting a
            # stale plan be woken up by an unrelated "yes".
            _pending_plan = None
            _pending_approval = None
            return False
        return True


def pending_preview():
    """Full preview text for the UI, or '' when nothing is pending."""
    record = get_pending_approval()
    return record.preview if record else ""


def pending_spoken():
    """Short spoken summary with action counts."""
    record = get_pending_approval()
    return record.spoken if record else ""


# ---- interaction history for multi-step context ----


def add_interaction(command, action_summary, verified=None, plan=None):
    """Record a screen interaction.  Keeps at most *_MAX_HISTORY* entries.

    F21: history is replayed into the model context and the UI, so it crosses
    the redacted egress boundary on the way IN. The command text and the
    typed steps used to be stored verbatim, which turned every later prompt
    (and every retry) into a second disclosure of the same credential.
    """
    global _interaction_history
    scrubbed_plan = plan
    try:
        scrubbed_plan = tool_policy.redact_for_egress(plan)
    except Exception:
        scrubbed_plan = None
    with _state_lock:
        _interaction_history.append(
            {
                "command": tool_policy.redact_for_egress(command),
                "action": tool_policy.redact_for_egress(action_summary),
                "verified": verified,
                "plan": scrubbed_plan,
                "ts": time.time(),
            }
        )
        if len(_interaction_history) > _MAX_HISTORY:
            _interaction_history = _interaction_history[-_MAX_HISTORY:]


def get_recent_interactions():
    with _state_lock:
        return list(_interaction_history)


def clear_interactions():
    global _interaction_history
    with _state_lock:
        _interaction_history = []


def format_history_for_prompt():
    """Return a compact prompt fragment describing recent screen actions."""
    with _state_lock:
        history = list(_interaction_history)

    if not history:
        return ""

    lines = ["Recent screen actions (for context):"]
    for i, entry in enumerate(history, 1):
        v_note = ""
        if entry.get("verified") is True:
            v_note = " [verified OK]"
        elif entry.get("verified") is False:
            v_note = " [verification uncertain]"
        lines.append(
            f'{i}. User said "{entry["command"]}" -> {entry["action"]}{v_note}'
        )
    return "\n".join(lines) + "\n"


def get_state():
    with _state_lock:
        state = {
            "screen_controls_enabled": _enabled,
            "screen_action_pending": _pending_plan is not None,
            "screen_generation": _generation,
        }
    state.update(approvals.state())
    return state
