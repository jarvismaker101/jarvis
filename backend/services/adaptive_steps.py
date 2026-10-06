"""Rank 3 — try, check, try smarter: failure diagnosis + differing retries.

Pure decision layer for the adaptive step loop in ``task_agent.agent``:

* :func:`diagnose` reads a failed step's own evidence (its result text and
  structured payload) and names the most likely cause;
* :func:`tactics_for` returns the recovery tactics appropriate to that cause
  and tool, in the order they should be attempted — every tactic is a
  DIFFERENT approach, so a retry can never be an identical repeat of an
  earlier try;
* :func:`attempt_signature` canonicalises (tactic, tool, args) so the loop
  can prove no attempt equals a previous one;
* :data:`MAX_TRIES_PER_STEP` and :data:`MAX_REPLANS_PER_JOB` are the hard
  caps; a job that exhausts them stops and says so honestly.

Nothing here executes anything: the agent supplies the tactic runners.
"""

import json
import re


TRANSIENT = "transient"
OVERLAY = "overlay"
MISSING = "missing"
LOGIN = "login"
EXISTS = "exists"
PERMISSION = "permission"
PREREQ = "prerequisite"
UNKNOWN = "unknown"

#: Hard caps (Rank 3): total executions of one step, and plan-level replans.
MAX_TRIES_PER_STEP = 3
MAX_REPLANS_PER_JOB = 2

#: Only these classes may be retried at all; everything else stops honestly
#: on the first failure (a login wall or an existing file never fixes itself).
RETRYABLE_CLASSES = frozenset({TRANSIENT, OVERLAY, MISSING})

#: Human labels for the spoken/detail trail.
TACTIC_LABELS = {
    "resettle_wait": "waiting a moment and re-reading the window",
    "refresh_evidence": "taking a fresh look at the page",
    "alternate_route": "trying another route",
    "dismiss_overlay": "dismissing the pop-up",
}

#: Which tools each tactic can safely act on. ``None`` means any retryable
#: tool. These are deliberately a small set of action tools with real
#: recovery runners; anything else is never auto-retried.
_TACTIC_TOOLS = {
    "resettle_wait": None,
    "refresh_evidence": frozenset({
        "browser.search_web", "browser.open_url", "windows.screen_action",
    }),
    "dismiss_overlay": frozenset({"windows.screen_action"}),
    "alternate_route": frozenset({
        "browser.search_web", "browser.open_url",
    }),
}

#: Default tactic order per class (before tool filtering).
_BASE_TACTICS = {
    TRANSIENT: ("resettle_wait", "refresh_evidence"),
    OVERLAY: ("refresh_evidence", "dismiss_overlay"),
    MISSING: ("refresh_evidence", "alternate_route"),
}

#: Order overrides where one tactic clearly fits better.
_ORDER_OVERRIDES = {
    (MISSING, "browser.search_web"): ("alternate_route", "refresh_evidence"),
    (MISSING, "browser.open_url"): ("alternate_route", "refresh_evidence"),
}

#: Evidence rules, most specific/authoritative first. (class, regex, why)
_EVIDENCE_RULES = (
    (LOGIN, re.compile(
        r"sign[ -]?in|\blog[ -]?in\b|\blogin\b|password|credential",
        re.IGNORECASE),
     "the page wants a sign-in, which I never do for you"),
    (PERMISSION, re.compile(
        r"\b(denied|permission|forbidden|not allowed|refus(?:ed|ing)|"
        r"read-?only|locked)\b", re.IGNORECASE),
     "permission was refused"),
    (EXISTS, re.compile(
        r"already exists|already there|already present|exists already|"
        r"would overwrite|file exists", re.IGNORECASE),
     "the target already exists"),
    (TRANSIENT, re.compile(
        r"timed? ?out|timeout|still loading|not loaded|not responding|"
        r"temporar|busy|try again|later|connection|network|unreachable|"
        r"hanging|hung", re.IGNORECASE),
     "the page or app was slow to respond"),
    (OVERLAY, re.compile(
        r"pop-?up|overlay|intercept|covered|obscured|not clickable|"
        r"obstruct|consent|cookie banner|dialog box|modal", re.IGNORECASE),
     "something is covering the target"),
    (MISSING, re.compile(
        r"not found|no element|no match|couldn'?t find|could not find|"
        r"couldn'?t locate|could not locate|couldn'?t produce|"
        r"could not produce|no confident|nothing matched|didn'?t match|"
        r"did not match", re.IGNORECASE),
     "the target was not there any more"),
)

#: A write whose parent is missing is a PLAN problem, not a retry problem.
_PREREQ_RE = re.compile(
    r"cannot find the path|path specified|no such file|does not exist|"
    r"doesn'?t exist|missing parent|parent|folder not found|"
    r"directory not found", re.IGNORECASE)


def can_replan(replans_used):
    """True while the job may still change its plan (Rank 3 cap)."""
    try:
        return int(replans_used) < MAX_REPLANS_PER_JOB
    except Exception:
        return False


def diagnose(tool, text, structured=None, external_effect=False):
    """Name the most likely cause of ONE failed step, from its own evidence.

    Returns ``{"class", "retryable", "why", "evidence"}``. ``retryable`` is
    True only for classes with recovery tactics that fit *tool* — and never
    for externally-visible effects (a sent mail is never re-sent).
    """
    error = ""
    if isinstance(structured, dict):
        error = str(structured.get("error") or structured.get("reason")
                    or structured.get("message") or "")
    hay = ("%s %s" % (text or "", error)).strip()
    hay = re.sub(r"\s+", " ", hay)
    verdict = {"class": UNKNOWN, "why": "the failure reason is unclear",
               "evidence": hay[:240]}
    if external_effect:
        verdict["retryable"] = False
        verdict["why"] = "an external effect is never retried automatically"
        return verdict
    if tool == "code.write_file" and _PREREQ_RE.search(hay):
        verdict["class"] = PREREQ
        verdict["why"] = "the folder this write needs is not there"
        verdict["retryable"] = False
        return verdict
    for cls, pattern, why in _EVIDENCE_RULES:
        if pattern.search(hay):
            verdict["class"] = cls
            verdict["why"] = why
            break
    verdict["retryable"] = bool(
        verdict["class"] in RETRYABLE_CLASSES
        and tactics_for(tool, verdict["class"]))
    return verdict


def tactics_for(tool, diagnosis_class):
    """Ordered recovery tactic ids that fit *tool* for this failure class.

    Every id is a different approach; the loop never repeats one it already
    used for the same step, so an identical retry is structurally impossible.
    """
    tool = str(tool or "")
    if diagnosis_class not in RETRYABLE_CLASSES:
        return ()
    order = _ORDER_OVERRIDES.get((diagnosis_class, tool)) \
        or _BASE_TACTICS.get(diagnosis_class, ())
    result = []
    for tactic in order:
        allowed = _TACTIC_TOOLS.get(tactic)
        if allowed is None or tool in allowed:
            result.append(tactic)
    return tuple(result)


def attempt_signature(tool, args, tactic="initial"):
    """Canonical identity of one try: tactic + tool + exact arguments."""
    try:
        payload = json.dumps(args or {}, sort_keys=True, default=str)
    except Exception:
        payload = str(args)
    return "%s|%s|%s" % (tactic, tool, payload)


_STOPWORDS = frozenset(
    "a an and are as at be by for from how i in is it me my of on or please "
    "that the their them then there these this to was were what when where "
    "which who why will with you your dont don't cant can't just really"
    .split()
)


def simplify_query(query, max_terms=6):
    """Drop filler so a failed web search can be retried DIFFERENTLY.

    Returns the simplified query, or ``""`` when simplification changes
    nothing — the caller then must not retry with an identical search.
    """
    original = re.sub(r"\s+", " ", str(query or "")).strip()
    if not original:
        return ""
    tokens = re.findall(r"[A-Za-z0-9']+", original)
    kept = [t for t in tokens if t.lower() not in _STOPWORDS]
    simplified = " ".join(kept[:max_terms]).strip()
    if not simplified or simplified.lower() == original.lower():
        return ""
    return simplified
