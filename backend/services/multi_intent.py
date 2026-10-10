"""Rank 2 — one utterance, many jobs: split a compound request into a chain.

Pure, deterministic, stdlib-only (no codebase imports), so the brain and
the workers can use it freely.

The chain is an ordered list of steps whose kinds are ``screen``,
``research`` and ``task``. A ``task`` step is always a FILE WRITE and is
only allowed as the LAST step — so every chain has at most one approval
pause and it lands at the end, with the earlier steps' output already in
hand. Steps that refer back to earlier output ("about it", "whatever you
find", "write your report in it") carry a ``consumes`` index; the brain's
executor injects the earlier step's output into the later step's query or
file content.

Deliberately conservative: splitting is liberal, but a chain is accepted
only when the pieces classify to distinct action kinds, no ``tool`` clause
is present, and no task clause sits before the end. Anything else returns
None and the turn keeps its normal single-intent routing.
"""

import re

_MAX_STEPS = 4

#: Clause separators: commas/semicolons, "and"/"then" families, and the
#: common Hinglish connectors. Splitting is intentionally liberal — the
#: strict chain acceptance below is what keeps single jobs intact.
_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:,\s*(?:and\s+|then\s+|so\s+|and\s+then\s+|after\s+that\s+)?"
    r"|;\s*"
    r"|\s+(?:and\s+then|and|then|after\s+that|afterwards|aur|phir"
    r"|uske\s+baad)\s+)",
    re.IGNORECASE,
)

#: First-person future talk ("I will look at my screen later and …") is
#: conversation ABOUT future actions, not a request to act now.
_FUTURE_TALK_RE = re.compile(
    r"^\s*(?:i\s*(?:will|'ll|would|might|could)|maybe\s+i|we\s*(?:will|'ll))\b",
    re.IGNORECASE,
)

#: The screen word tolerates the typos speech-to-text actually produces
#: ("scrren", "scren", "screan") — a typo used to drop the screen clause
#: entirely, so "research about it" never bound to it and the whole
#: sentence was typed into the web.
_SCREEN_WORD = r"(?:scr+[ae]*n|monitor|display)"

_SCREEN_RE = re.compile(
    r"\b(look at|check|see|read|watch|analyse|analyze|scan|view|examine)\b"
    r"[^.?!,;]{0,40}\b" + _SCREEN_WORD + r"\b"
    r"|\bon (?:my|the) " + _SCREEN_WORD + r"\b"
    r"|\bwhat(?:'s| is) on (?:my|the) " + _SCREEN_WORD + r"\b"
    r"|\blook at (?:this|the) (?:video|image|picture|photo|window)\b",
    re.IGNORECASE,
)

_RESEARCH_RE = re.compile(
    r"\b(deep\s*research|deepsearch|deep\s+search|research|google|search|"
    r"look\s+(?:it|this|that)\s+up|look\s+up|find\s+out|"
    r"search\s+(?:on\s+|the\s+)?(?:internet|web|online)|"
    r"(?:on|from)\s+the\s+internet|online|internet)\b",
    re.IGNORECASE,
)

_TASK_WRITE_RE = re.compile(
    r"\b(create|make|write|save|put|store)\b[^.?!]{0,60}"
    r"\b(file|report|txt|text|document|note)\b"
    r"|\bwrite (?:your|the|a) report\b"
    r"|\bsave (?:it|this|that|the findings|the results|what you)\b",
    re.IGNORECASE,
)

_TASK_ACTION_RE = re.compile(
    r"\b(create|make|delete|move|rename|copy|edit|run|execute|install)\b"
    r"[^.?!]{0,60}\b(folder|directory|script|code|command|program)\b",
    re.IGNORECASE,
)

#: [LIVE FIX 12] A clause that REPLICATES a structure shown elsewhere
#: ("replicate it exactly on my desktop", "recreate the folder structure
#: on my desktop"). The object of the verb is the thing the SCREEN showed,
#: so this clause is a task half of a screen→task chain and must route as
#: such, never as a standalone folder-name question. Note "copy" lives in
#: both this and _TASK_ACTION_RE: there the object is a folder word, here
#: it is the displayed structure.
_TASK_REPLICATE_RE = re.compile(
    r"\b(?:replicate|recreate|rebuild|duplicate|clone|mirror|copy)\b"
    r"[^.?!]{0,48}\b(?:it|this|that|them|structure|tree|layout|hierarchy)\b",
    re.IGNORECASE,
)

#: [LIVE FIX 12] A displayed STRUCTURE (not a single folder): the marker
#: that makes a screen clause a look-and-replicate job even when the
#: replication fragment was merged into it (no conjunction to split on).
_STRUCTURE_RE = re.compile(
    r"\b(?:structure|tree|hierarchy|layout)\b", re.IGNORECASE)

_TOOL_RE = re.compile(
    # "Open AI" / "OpenAI" is a company name, not the browser verb — but a
    # real tool clause ("open youtube", "open the browser") still counts.
    r"\b(?:open(?!\s+ai\b)|launch|play|start|go\s+to|navigate|visit)\b",
    re.IGNORECASE,
)

#: The step refers back to the previous step's output.
_ANAPHORA_RE = re.compile(
    r"\b(about|on|of|from)\s+(it|this|that|them)\b"
    r"|\b(whatever|what)\s+you\s+(?:find|see|get|learn)\b"
    r"|\byour\s+(?:report|findings|research|answer|summary|results)\b"
    r"|\bthe\s+(?:findings|results|report|research|answer|summary)\b"
    r"|\bin\s+(?:it|that file)\b|\binto\s+(?:it|that file)\b"
    r"|\b(save|write|put|store|add)\s+(it|this|that|them)\b"
    r"|\b(research|google|look\s+up|find\s+out\s+about|search\s+for)\s+"
    r"(it|this|that|them)\b"
    # Live fix: "search this youtube creator", "find anything about that
    # stream" — the search verb and the pointer need not be adjacent.
    r"|\b(search|searches|searching|find|research|google|look\s+up)\b"
    r"[^.?!,;]{0,32}\b(it|this|that|these|those|them)\b"
    # Live fix: "whichever name you find ... by that name" — the task
    # clause consumes the research result even though "that name" is not
    # the usual anaphora shape.
    r"|\b(?:whichever|which|the)\s+(?:name|model|one)\b[^.?!,;]{0,40}\byou\b"
    r"|\bby\s+that\s+name\b"
    r"|\bthat\s+(?:model\s+)?name\b"
    # Live fix 12: "replicate it exactly on my desktop" — the task clause
    # reproduces what the screen step saw (RANK 1's stored observation).
    r"|\b(?:replicate|recreate|rebuild|duplicate|clone|mirror|copy)\b\s+"
    r"(?:it|this|that|them|the\s+(?:(?:folder|directory|project|file)\s+)?"
    r"(?:structure|tree|layout|hierarchy))\b",
    re.IGNORECASE,
)

#: After a screen step, a research clause that points at a screen object
#: ("this streamer", "the creator", "that video") is about what was seen.
_SCREEN_REF_RE = re.compile(
    r"\b(it|this|that|these|those)\b"
    r"|\bthe\s+(?:same|stream|streamer|creator|channel|video|song|movie|"
    r"guy|person|content)\b",
    re.IGNORECASE,
)

#: Anaphoric question whose referent is the thing just shown on screen.
_DEICTIC_RES_RE = re.compile(
    r"\b(what|which)\b[^.?!,;]{0,30}"
    r"\b(it|this|that|these|those|error|video|movie|image|photo|song)\b",
    re.IGNORECASE,
)


def split_clauses(text):
    """Split *text* on conjunction boundaries into candidate clauses."""
    parts = _CLAUSE_SPLIT_RE.split(text or "")
    out = []
    for part in parts:
        part = (part or "").strip(" \t,;.")
        if len(part) < 3:
            continue
        out.append(part)
    return out


def classify_clause(text):
    """The single action kind of one clause, or None for a fragment.

    Fragments (chatter, "tell me what you see", answers) return None and
    are merged into the clause they continue.
    """
    t = (text or "").strip()
    if not t:
        return None
    if _SCREEN_RE.search(t):
        return "screen"
    if _RESEARCH_RE.search(t):
        return "research"
    if (_TASK_WRITE_RE.search(t) or _TASK_ACTION_RE.search(t)
            or _TASK_REPLICATE_RE.search(t)):
        return "task"
    if _TOOL_RE.search(t):
        return "tool"
    return None


def build_chain(text):
    """Parse *text* into a step chain, or None when it is a single job.

    Accepted only when >= 2 steps survive merging, the kinds are distinct,
    no tool clause is present, and a task (file write) clause, if any, is
    last.
    """
    raw = (text or "").strip()
    if not raw or _FUTURE_TALK_RE.match(raw):
        return None
    clauses = split_clauses(raw)
    if not clauses:
        return None
    merged = []
    for clause in clauses:
        kind = classify_clause(clause)
        if kind is None:
            if merged:
                merged[-1]["text"] = merged[-1]["text"] + " and " + clause
            continue
        if merged and merged[-1]["kind"] == kind:
            merged[-1]["text"] = merged[-1]["text"] + " and " + clause
            continue
        merged.append({"kind": kind, "text": clause})
    # Live fix (general): ONE clause that both points at the screen and asks
    # to research it ("find out who this player on my screen is by
    # researching on the internet, and create a folder…") is TWO jobs even
    # when more clauses follow — without this the research half of the
    # request silently died and only the screen look ran. A screen clause
    # that carries research markers gains a research sibling right after it
    # (unless the next clause is already its own research step).
    expanded = []
    for i, m in enumerate(merged):
        expanded.append(m)
        if (m["kind"] == "screen" and _RESEARCH_RE.search(m["text"])
                and not (i + 1 < len(merged)
                         and merged[i + 1]["kind"] == "research")):
            expanded.append({"kind": "research", "text": m["text"]})
    merged = expanded
    # [LIVE FIX 12] The same rule for a screen clause that SHOWS a structure
    # and asks to replicate it ("look at my screen there is a project folder
    # structure replicate it on my desktop" — no conjunction to split on):
    # look, then rebuild exactly that structure on the destination.
    expanded = []
    for i, m in enumerate(merged):
        expanded.append(m)
        if (m["kind"] == "screen" and _TASK_REPLICATE_RE.search(m["text"])
                and _STRUCTURE_RE.search(m["text"])
                and not (i + 1 < len(merged)
                         and merged[i + 1]["kind"] == "task")):
            expanded.append({"kind": "task", "text": m["text"]})
    merged = expanded
    if len(merged) < 2 or len(merged) > _MAX_STEPS:
        return None
    kinds = [m["kind"] for m in merged]
    if "tool" in kinds:
        return None
    task_positions = [i for i, k in enumerate(kinds) if k == "task"]
    if task_positions and task_positions != [len(kinds) - 1]:
        return None
    steps = []
    for i, m in enumerate(merged):
        consumes = []
        if i > 0 and _ANAPHORA_RE.search(m["text"]):
            consumes = [i - 1]
        elif (i > 0 and merged[i - 1]["kind"] == "screen"
              and m["kind"] == "research"
              and (_DEICTIC_RES_RE.search(m["text"])
                   or _SCREEN_REF_RE.search(m["text"]))):
            consumes = [i - 1]
        elif (i > 0 and merged[i - 1]["kind"] == "screen"
              and m["kind"] == "task"
              and _TASK_REPLICATE_RE.search(m["text"])):
            # [LIVE FIX 12] The task clause rebuilds what the screen step
            # just saw — it consumes that step's stored observation.
            consumes = [i - 1]
        steps.append({"kind": m["kind"], "text": m["text"],
                      "index": i, "consumes": consumes})
    return {"ok": True, "steps": steps, "source": raw, "command_text": raw}


_LABELS = {
    "screen": "look at your screen",
    "research": "research it online",
    "task": "save it to a file",
}

_FOLDER_CLAUSE_RE = re.compile(r"\b(folder|directory)\b", re.IGNORECASE)


def _step_label(step):
    """What this step will actually do — folder jobs never say "file"."""
    kind = step.get("kind")
    text = step.get("text") or ""
    if kind == "task" and _TASK_REPLICATE_RE.search(text):
        # [LIVE FIX 12] A replication rebuilds the on-screen structure —
        # never "save it to a file".
        return "recreate that structure on your desktop"
    if kind == "task" and _FOLDER_CLAUSE_RE.search(text):
        if re.search(r"\bfile\b", text, re.IGNORECASE):
            return "create a folder with a file inside"
        return "create a folder"
    return _LABELS.get(kind, kind)


def render_ack(plan, voice_compact=False):
    """One coherent narrative for the whole chain — not per-step chatter."""
    steps = plan.get("steps") or []
    kinds = [s.get("kind") for s in steps]
    n = len(kinds)
    if n < 2:
        return "On it, sir."
    parts = ", ".join(_step_label(s) for s in steps)
    last_text = steps[-1].get("text") or ""
    folder_last = (kinds[-1] == "task"
                   and (_FOLDER_CLAUSE_RE.search(last_text)
                        or _TASK_REPLICATE_RE.search(last_text)))
    if folder_last:
        tail = "I'll ask before creating it."
    elif kinds[-1] == "task":
        tail = "I'll ask before creating the file."
    else:
        tail = "I'll report back when it's done."
    if voice_compact:
        return "Sir, on it in %d steps — %s. %s" % (n, parts, tail)
    return "Sir, I'll do this in %d steps: %s. %s" % (n, parts, tail)
