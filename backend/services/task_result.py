"""Evidence-backed task results (Fable-5 audit G1: F03 + F05).

TaskResult is the single return type of the task engines
(browser_agent.run_browser_task, opencode_client.run_opencode_task) so a
caller can always tell *completed* apart from *partial* and *failed* -
no more inferring success from substrings of a transcript.

Legacy string compatibility (deliberate, documented):
  * str() reproduces the historical wire strings byte-for-byte, so every
    existing startswith/equality check keeps passing:
      failed  -> 'TASK NOT COMPLETED. Error: <error or summary>'
      stopped -> 'Stopped per your request.'
      anything else -> the summary (like the old returned strings).
  * bool() is False ONLY for 'failed', so the old `if output:` guards
    keep meaning 'the task produced usable output' (the engines used to
    return '' on every failure path).
  * == / in / startswith() against plain strings delegate to str(self)
    for the same reason. New code MUST branch on .status instead.
  * is_failure_text() centralises the 'TASK NOT COMPLETED' prefix check.

FailureTracker is the task-level identical-action failure cap (F05):
keyed by (tool name, canonical JSON of arguments), it counts consecutive
failures of the exact same action; a success resets the counter. The
browser agent blocks the 3rd identical failure without executing it.

Pure data module: stdlib only (no brain/browser_agent imports, no
cycles).
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Valid TaskResult.status values.
COMPLETED = "completed"
PARTIAL = "partial"
FAILED = "failed"
STOPPED = "stopped"
NEEDS_INPUT = "needs_input"

_VALID_STATUSES = frozenset((COMPLETED, PARTIAL, FAILED, STOPPED, NEEDS_INPUT))

#: Legacy wire strings (byte-identical to the pre-TaskResult outputs).
_FAILED_PREFIX = "TASK NOT COMPLETED"
_STOP_MESSAGE = "Stopped per your request."


def is_failure_text(text):
    """True when `text` (str or TaskResult) carries the legacy failure prefix."""
    if text is None:
        return False
    try:
        # NOTE: no `text or ""` here - a failed TaskResult is FALSY by
        # design, and `or` would discard it before str() ever runs.
        return str(text).startswith(_FAILED_PREFIX)
    except Exception:
        return False


# ── F03: completion is an EVIDENCE-BACKED status ──────────────────────────
#
# Callers used to promote whatever a model said into "completed": a refusal, a
# bare "Done.", a report whose own words contain an error, or an exit status
# nobody actually observed. These patterns separate the three cases that
# matter: a KNOWN FAILURE, a claim with NO ACTION behind it, and a report that
# names an actual effect. Anything else stays unverified.

#: A transcript that says only this proves no effect happened.
NO_ACTION_TEXTS = frozenset((
    "done", "done.", "done!", "done sir", "done, sir", "done, sir.",
    "completed", "completed.", "finished", "finished.", "ok", "okay",
    "task completed", "all done", "all done.", "nothing", "no action",
    "no action taken", "n/a",
))

#: Known failure wording. Checked BEFORE any success word, so "failed: file was
#: created" can never be read as a completed creation.
FAILURE_TEXT_RE = re.compile(
    r"(?:"
    r"\bfailed\b|\bfailure\b|\berror\b|\bexception\b|\btraceback\b|"
    r"\bunknown tool\b|\bnot found\b|\bno such file\b|\bdenied\b|"
    r"\bpermission\b|\btimed? ?out\b|\bunavailable\b|\bnot available\b|"
    r"\baborted\b|\brejected\b|\bblocked\b"
    r")", re.IGNORECASE)

#: A refusal, a non-action, or a statement that nothing was done.
REFUSAL_TEXT_RE = re.compile(
    r"(?:"
    r"\bi (?:can'?t|cannot|won'?t|will not|am unable|'?m unable|am not able|"
    r"'?m not able|refuse|decline)\b|"
    r"\bunable to\b|\bnot able to\b|\bno action (?:was |were )?taken\b|"
    r"\bnothing (?:was |were )?(?:done|changed|modified|created|written)\b|"
    r"\bi did not\b|\bcannot comply\b|\brefus(?:e|ed|ing)\b"
    r")", re.IGNORECASE)

#: An actual effect, in the past tense, is the only prose that can support a
#: completion claim.
EFFECT_TEXT_RE = re.compile(
    r"(?:"
    r"\bcreated\b|\bwrote\b|\bwritten\b|\bsaved\b|\bdeleted\b|\bremoved\b|"
    r"\bmoved\b|\bcopied\b|\brenamed\b|\bopened\b|\bclicked\b|\btyped\b|"
    r"\binstalled\b|\bedited\b|\bupdated\b|\bpatched\b|\bran\b|\bexecuted\b|"
    r"\bsent\b|\buploaded\b|\bdownloaded\b|\bnavigated\b|\blaunched\b|"
    r"\bstarted\b|\bcompleted the\b|\bfinished the\b"
    r")", re.IGNORECASE)

#: Status values a completion claim can be built from.
VERIFIED = "verified"
UNVERIFIED = "unverified"
REFUSED = "refused"
NO_ACTION = "no_action"
KNOWN_FAILURE = "known_failure"
QUESTION = "question"


def classify_reported_text(text):
    """Classify a raw engine/model report (F03).

    Returns one of ``KNOWN_FAILURE``, ``REFUSED``, ``NO_ACTION``,
    ``QUESTION``, ``VERIFIED`` or ``UNVERIFIED``. Refusals, bare
    acknowledgements and reports whose own wording carries a failure never
    yield ``VERIFIED`` — that is reserved for text naming an actual effect.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return NO_ACTION
    # Failure wording wins over any success word in the same report, which is
    # what stops "failed: created nothing" from becoming a completion.
    if is_failure_text(cleaned) or FAILURE_TEXT_RE.search(cleaned):
        return KNOWN_FAILURE
    if REFUSAL_TEXT_RE.search(cleaned):
        return REFUSED
    lowered = cleaned.lower().strip(" .!,")
    if lowered in NO_ACTION_TEXTS or len(lowered) <= 3:
        return NO_ACTION
    # A question is a SUSPENSION: the engine is asking the user, so nothing has
    # been completed and nothing has failed.
    if cleaned.rstrip().endswith("?"):
        return QUESTION
    if EFFECT_TEXT_RE.search(cleaned):
        return VERIFIED
    return UNVERIFIED


def result_from_reported_text(text, detail="", artifacts=None,
                              evidence=None, assume_success=False,
                              trust_answer=True):
    """Build the TaskResult for a REPORT-WORDING-ONLY outcome (F03).

    ``assume_success`` is the caller's statement that it independently
    observed terminal success (a real exit code of 0, a verified effect). Even
    then, a refusal, a question or a known failure in the text is never
    upgraded: verified completion needs both the observation AND the absence
    of contradiction.

    ``trust_answer`` (default True) says substantive prose IS the deliverable
    — an informational answer completes its task even though it names no
    effect. Set it False to require an explicit effect.
    """
    verdict = classify_reported_text(text)
    body = (text or "").strip()
    if verdict == KNOWN_FAILURE:
        reason = body.split("Error:", 1)[-1].strip() if "Error:" in body else body
        return TaskResult.failed(reason or "the tool reported an error",
                                 detail=detail or body)
    if verdict == REFUSED:
        return TaskResult.partial(
            body or "The action was not carried out.",
            detail=detail or body,
            evidence=["the model refused or reported no action"])
    if verdict == NO_ACTION:
        return TaskResult.partial(
            body or "Nothing was done.",
            detail=detail or body,
            evidence=["no action was reported"])
    if verdict == QUESTION:
        return TaskResult.needs_input(body, detail=detail or body)
    if verdict == VERIFIED:
        return TaskResult.completed(body, detail=detail or body,
                                    evidence=evidence)
    # A claim with no named effect.
    if trust_answer:
        # The answer itself is the result (a lookup, an explanation, a report).
        return TaskResult.completed(body, detail=detail or body,
                                    evidence=evidence)
    result = TaskResult.partial(
        body or "The task finished without a verified result.",
        detail=detail or body,
        evidence=list(evidence or []) + ["completion not evidenced"],
    )
    if artifacts:
        result.artifacts = list(artifacts)
    return result


@dataclass(eq=False)
class TaskResult:
    """Evidence-backed outcome of one task-engine run.

    status is one of 'completed', 'partial', 'failed', 'stopped',
    'needs_input'. summary is the spoken/chat headline, detail the full
    raw output, evidence a list of short factual strings (tool failures,
    budget notes, block reasons), artifacts optional structured outputs,
    error the short machine-style reason (set for 'failed').
    """

    status: str
    summary: str = ""
    detail: str = ""
    evidence: List[str] = field(default_factory=list)
    artifacts: List[Any] = field(default_factory=list)
    error: Optional[str] = None
    #: F03 — the goals this run set out to achieve but did not. A result with
    #: unmet goals is never 'completed', and callers keep them so a later turn
    #: can finish the work instead of re-planning from scratch.
    unmet_goals: List[str] = field(default_factory=list)
    #: F08 — identity of the suspended checkpoint this result left behind, so
    #: a follow-up RESUMES it (verified progress restored) instead of starting
    #: a new run that would replay committed actions.
    checkpoint: Optional[str] = None
    #: F09 — the verified TRACE of what the run actually did: one dict per
    #: executed step ({"tool", "args", "observation"}). This is what turns a
    #: successful run into a parameterised PROCEDURE instead of goal text.
    trace: List[Dict[str, Any]] = field(default_factory=list)
    #: F09 — the observations that verified the goal (postconditions).
    verification: List[str] = field(default_factory=list)
    # E1 (G1 round-2 re-audit): when True and status == 'failed', str()
    # returns the message AS-IS without the 'TASK NOT COMPLETED. Error: '
    # prefix (plan-not-ok spoken text must stay byte-identical to pre-G1).
    # Status/bool semantics are unchanged. Excluded from == comparison.
    plain: bool = field(default=False, compare=False)

    def __post_init__(self):
        if self.status not in _VALID_STATUSES:
            raise ValueError("invalid TaskResult status: %r" % (self.status,))
        self.summary = self.summary or ""
        self.detail = self.detail if self.detail is not None else ""
        self.evidence = list(self.evidence or [])
        self.artifacts = list(self.artifacts or [])
        self.unmet_goals = list(self.unmet_goals or [])
        # F03: an unverified completion is a contradiction — unmet goals mean
        # the status must say so instead of claiming success.
        if self.status == COMPLETED and self.unmet_goals:
            self.status = PARTIAL

    # ── legacy string compatibility (see module docstring) ──
    def __str__(self):
        if self.status == FAILED:
            message = self.error or self.summary or "unknown error"
            if self.plain:
                return message
            return "%s. Error: %s" % (_FAILED_PREFIX, message)
        if self.status == STOPPED:
            return _STOP_MESSAGE
        return self.summary or ""

    def __repr__(self):
        return "TaskResult(status=%r, summary=%r, error=%r, evidence=%r)" % (
            self.status, self.summary, self.error, self.evidence,
        )

    def __bool__(self):
        # Only 'failed' is falsy: `if result:` keeps meaning 'usable output'.
        return self.status != FAILED

    def __eq__(self, other):
        if isinstance(other, TaskResult):
            return (
                self.status == other.status
                and self.summary == other.summary
                and self.detail == other.detail
                and self.evidence == other.evidence
                and self.error == other.error
            )
        if isinstance(other, str):
            return str(self) == other
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __contains__(self, item):
        return item in str(self)

    def startswith(self, prefix, *args):
        return str(self).startswith(prefix, *args)

    # ── builders ──
    @classmethod
    def completed(cls, summary, detail="", evidence=None, artifacts=None,
                  unmet_goals=None):
        result = cls(status=COMPLETED, summary=summary or "",
                     detail=detail if detail is not None else "",
                     evidence=list(evidence or []))
        result.artifacts = list(artifacts or [])
        # NOTE: __post_init__ already ran, so an unmet-goal completion is
        # downgraded to partial explicitly here (F03).
        if unmet_goals:
            result.unmet_goals = list(unmet_goals)
            result.status = PARTIAL
        return result

    @classmethod
    def verified(cls, summary, evidence, detail="", artifacts=None):
        """F03: a completion that MUST carry its evidence.

        Raises ValueError when no evidence is supplied, so a caller cannot
        publish a verified completion for an effect nobody observed.
        """
        if not evidence:
            raise ValueError(
                "a verified completion requires evidence of the effect")
        return cls.completed(summary, detail=detail, evidence=evidence,
                             artifacts=artifacts)

    @classmethod
    def partial(cls, summary, detail="", evidence=None, error=None,
                artifacts=None, unmet_goals=None):
        result = cls(status=PARTIAL, summary=summary or "",
                     detail=detail if detail is not None else "",
                     evidence=list(evidence or []), error=error)
        result.artifacts = list(artifacts or [])
        result.unmet_goals = list(unmet_goals or [])
        return result

    @classmethod
    def failed(cls, error, detail="", plain=False):
        return cls(status=FAILED, summary="",
                   detail=detail if detail is not None else "",
                   error=error or "unknown error", plain=plain)

    @classmethod
    def stopped(cls):
        return cls(status=STOPPED, summary=_STOP_MESSAGE)

    @classmethod
    def needs_input(cls, question, detail="", checkpoint=None):
        result = cls(status=NEEDS_INPUT, summary=question or "",
                     detail=detail if detail is not None else "")
        result.checkpoint = checkpoint
        return result


class FailureTracker:
    """Counts consecutive failures of the identical tool action (F05).

    Key = (tool_name, canonical json.dumps of arguments, sort_keys=True).
    A successful execution resets that key's counter to 0. Callers decide
    the policy; the tracker only counts. NOTE_AFTER / BLOCK_AFTER encode
    the agreed thresholds: NOTE text on the 2nd failure, block (without
    executing) on the 3rd identical failure.

    F05 — STRUCTURED OUTCOMES. Counting failures is not enough for a mutation
    whose transport died after submission: the effect may or may not have
    happened. Those actions are recorded as OUTCOME-UNKNOWN and may only be
    replayed after an independent observation of the same target proves the
    first attempt did not commit.
    """

    NOTE_AFTER = 2
    BLOCK_AFTER = 3

    REPEAT_NOTE = (
        "NOTE: this action has now failed 2 times; "
        "change approach or finish without it."
    )

    #: Transport failures that leave a mutation's outcome genuinely unknown.
    AMBIGUOUS_ERROR_MARKERS = (
        "timeout", "timed out", "connection", "disconnect", "closed",
        "broken pipe", "reset by peer", "eof", "no response", "transport",
        "socket", "cancelled", "aborted",
    )

    #: Argument fields that identify the target of an action.
    TARGET_FIELDS = ("url", "selector", "name", "href", "path", "query",
                     "text", "value", "tab", "index", "frame")

    def __init__(self):
        self.counts: Dict[Tuple[str, str], int] = {}
        self.outcome_unknown: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.observed: Dict[str, int] = {}
        self._sequence = 0

    @staticmethod
    def key(tool_name, arguments):
        try:
            canonical = json.dumps(arguments or {}, sort_keys=True, default=str)
        except Exception:
            try:
                canonical = repr(sorted((arguments or {}).items()))
            except Exception:
                canonical = repr(arguments)
        return (tool_name or "", canonical)

    @staticmethod
    def target_key(arguments):
        """A stable identifier for WHAT an action targets (F05).

        Two actions on the same selector/url/path share a target key, so an
        observation of that target can reconcile an unknown-outcome mutation.
        """
        if not isinstance(arguments, dict):
            return ""
        parts = []
        for field in FailureTracker.TARGET_FIELDS:
            value = arguments.get(field)
            if isinstance(value, (str, int)) and str(value).strip():
                parts.append("%s=%s" % (field, str(value).strip().lower()))
        return "|".join(parts)

    @staticmethod
    def is_ambiguous_error(error):
        text = str(error or "").lower()
        return any(marker in text for marker in
                   FailureTracker.AMBIGUOUS_ERROR_MARKERS)

    def failures_for(self, key):
        return self.counts.get(key, 0)

    def record_success(self, key):
        self.counts.pop(key, None)

    def record_failure(self, key):
        count = self.counts.get(key, 0) + 1
        self.counts[key] = count
        return count

    # ── F05: ambiguous mutation outcomes ──
    def mark_outcome_unknown(self, key, target="", reason=""):
        """The action was submitted but its result was never observed."""
        self._sequence += 1
        self.outcome_unknown[key] = {
            "target": target or "",
            "reason": reason or "the response was lost",
            "at": self._sequence,
        }
        return self.outcome_unknown[key]

    def outcome_unknown_reason(self, key):
        entry = self.outcome_unknown.get(key)
        return entry["reason"] if entry else None

    def note_observation(self, target="", evidence=""):
        """Record an independent observation of a target (a read)."""
        self._sequence += 1
        self.observed[target or ""] = self._sequence
        return self._sequence

    def replay_decision(self, key, target=""):
        """May this action run again? Returns (allowed, reason)."""
        entry = self.outcome_unknown.get(key)
        if entry is None:
            return True, ""
        observed_at = self.observed.get(entry.get("target") or target or "", -1)
        if observed_at > entry.get("at", 0):
            # An independent observation arrived AFTER the ambiguous attempt:
            # the caller has real evidence of the current state.
            self.outcome_unknown.pop(key, None)
            return True, ""
        return False, entry.get("reason") or "the previous attempt's outcome is unknown"

    def forget_target(self, target):
        self.observed.pop(target or "", None)
