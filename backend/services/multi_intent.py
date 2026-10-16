"""Rank 2 — one utterance, many jobs: split a compound request into a chain.

Pure, deterministic, stdlib-only (no codebase imports), so the brain and
the workers can use it freely.

The chain is an ordered list of steps whose kinds are ``screen``,
``screen_control``, ``research`` and ``task``. A ``task`` step is always a FILE WRITE and is
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

_MAX_STEPS = 6

#: [LIVE FIX 26 / B40] stray UI/punctuation glyphs that leak into typed
#: commands (the "✕" stop-marker, bullets, arrows) and must never reach
#: parsing or the spoken ack. Only leading/trailing runs are stripped —
#: interior content ("mr.beast", file names) is untouched.
_STRAY_LEAD_RE = re.compile(
    r"^[\s\u2715\u2716\u2717\u2718\u2719\u271a\u00d7\u00b7\u2022\u2023"
    r"\u25e6\u25aa\u25ab\u2192\u00bb>\-–—*#+]+")
_STRAY_TAIL_RE = re.compile(
    r"[\s\u2715\u2716\u2717\u2718\u2719\u271a\u00d7\u00b7\u2022\u2023"
    r"\u25e6\u25aa\u25ab\u2192\u00bb>\-–—*#+.,;!]+$")


def _strip_stray_glyphs(text):
    """Drop leading/trailing stray UI glyphs from a command. Never raises."""
    try:
        t = str(text or "")
    except Exception:
        return ""
    t = _STRAY_LEAD_RE.sub("", t)
    t = _STRAY_TAIL_RE.sub("", t)
    return t

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
#: sentence was typed into the web. Live fix: "at my screen" counts too —
#: the live log's replicate sentence ("there is a project structure
#: visible at my screen, create a exact replica of this on my desktop")
#: said "at", so the screen clause never formed and the chain died.
_SCREEN_WORD = r"(?:scr+[ae]*n|monitor|display)"

_SCREEN_RE = re.compile(
    r"\b(look at|check|see|read|watch|analyse|analyze|scan|view|examine)\b"
    r"[^.?!,;]{0,40}\b" + _SCREEN_WORD + r"\b"
    r"|\b(?:on|at) (?:my|the) " + _SCREEN_WORD + r"\b"
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

#: [LIVE FIX 21] screen-CONTROL clauses — DOING something to the screen or
#: its windows, as opposed to LOOKING at it (kind "screen"). Open/launch/
#: play/navigate stay OUT on purpose: those are the tool verbs (Rank 2
#: rejects tool clauses), and file/report/folder verbs stay OUT (kind
#: "task"). A bare "scroll down" / "press enter" / "minimize" names no
#: target but is a complete command on its own, so those shapes match
#: without a target noun.
#: [LIVE FIX 22 / B9] element-level follow-on verbs ("backspace whatever is
#: written in it", "type mr.beast", "play the first video", bare "search"
#: handled at chain level): a spoken control sequence is one verb per
#: clause, so backspace/type-content/ordinal-play match without a UI noun.
#: "copy" stays OUT of the bare list ("copy it" is the task replicate);
#: "delete/clear/erase" only match field-content shapes, never file/folder
#: nouns ("delete the folder" stays a task); "type <content>" is blocked
#: when a file noun follows ("type the report in the file" stays a task);
#: ordinal "play the first/next X" is a screen click while plain "play X"
#: stays a tool request.
#: [LIVE FIX 32 / B85] Speech-to-text frequently drops or transposes the
#: leading letters of a clause-initial "type <content>" ("ype mr.beast in the
#: search bar"). The control verb is recognised typo-tolerantly so a mangled
#: "type" is still a screen-CONTROL step — never a bare screen LOOK.
_TYPE_VERB_TOKEN = r"(?:type|ype|tye|tpe|tup)"
_TYPE_TYPO_LEAD_RE = re.compile(
    r"^(\s*(?:(?:please|kindly|jarvis)\b[\s,]*)?)(?:ype|tye|tpe|tup)\b(?=\s+\S)",
    re.IGNORECASE,
)


def _repair_type_typo(text):
    """Repair a clause-initial STT-mangled 'ype/tye/tpe/tup' -> 'type'.

    [LIVE FIX 32 / B85] Only at the clause start and only when real content
    follows, and never when a file noun follows (a file write stays a task).
    Never raises.
    """
    try:
        t = str(text or "")
        if _JUXT_FILE_NOUN_RE.search(t[:60]):
            return t
        return _TYPE_TYPO_LEAD_RE.sub(r"\1type", t, count=1)
    except Exception:
        return str(text or "")


_SCREEN_CONTROL_RE = re.compile(
    r"\b(click|double-?click|right-?click|tap|press|hit|" + _TYPE_VERB_TOKEN + r"|enter|write|"
    r"copy|scroll|drag|drop|move|hover|select|focus|minimize|minimise|maximize|"
    r"maximise|restore|switch|close|paste|cut)\b[^.?!]{0,48}\b(screen|windows?|"
    r"desktop|button|link|tabs?|icon|checkbox|menu|toolbar|search\s*bar|"
    r"search\s*box|text\s*box|textbox|field|mouse|cursor|apps?|"
    r"applications?|dialog|popup)\b"
    r"|\b(screen|windows?|desktop|button|link|tabs?|icon|mouse|cursor)\b"
    r"[^.?!]{0,24}\b(click|press|" + _TYPE_VERB_TOKEN + r"|scroll|minimize|minimise)\b"
    r"|\b(scroll\s+(?:up|down|left|right)|press\s+enter|hit\s+enter|"
    r"minimize|minimise|maximize|maximise|restore)\b"
    # A control verb aimed at the previous step's target ("and click it",
    # "then close that") is a follow-on control step, not a fragment.
    r"|\b(click|double-?click|right-?click|tap|press|" + _TYPE_VERB_TOKEN + r"|scroll|"
    r"minimize|minimise|close)\s+(it|this|that|them|these|those)\b"
    # [LIVE FIX 22 / B9] field-content editing without a named target.
    r"|\bbackspace\b"
    r"|\b(?:backspace|delete|clear|erase)\b[^.?!]{0,48}\b(?:whatever|"
    r"what(?:'s| is)?\s+\w+|in\s+(?:it|this|that)|it|this|that|all|"
    r"everything|field|box|bar|text)\b"
    # [LIVE FIX 22 / B9] typing content into the focused element ("type
    # mr.beast", "ype mr.beast") — blocked when a file noun follows (file
    # writes stay task). [LIVE FIX 32 / B85] typo-tolerant verb.
    r"|^\s*(?:(?:please|kindly|jarvis)\b[^.?!]{0,12})?" + _TYPE_VERB_TOKEN + r"\b"
    r"(?![^.?!]{0,48}\b(?:file|folder|directory|report|document|note|txt|"
    r"script|code)\b)[^.?!]{0,48}\S"
    r"|\bpaste\b|\bcut\b"
    # [LIVE FIX 22 / B9] ordinal play is a screen click ("play the first
    # video"); plain "play X" stays a tool request.
    r"|\bplay\b[^.?!]{0,32}\bthe\s+(?:first|second|third|fourth|fifth|"
    r"top|next|last)\b"
    # [LIVE FIX 23 / B14] a control verb aimed at the submit target
    # ("click search", "press search", "tap the submit button") is a
    # screen click — never a web-search step. Bare "search" alone keeps
    # its own meaning (chain-level submit vs lookup).
    r"|\b(click|double-?click|right-?click|tap|press|hit)\s+(?:the\s+)?"
    r"(?:search|submit)\b"
    # [LIVE FIX 24 / B22] submitting what was just typed ("search what
    # you typed") is a screen submit — never a web-search step.
    r"|\b(?:search|submit|look\s+it\s+up|find\s+it)\b[^.?!]{0,48}"
    r"\bwhat\s+(?:you|i)\s+(?:just\s+)?typed\b"
    # [LIVE FIX 24 / B22] the gerund form ("clicking enter") is the same
    # complete submit command as "press enter".
    r"|\bclick(?:ing)?\s+enter\b",
    re.IGNORECASE,
)

#: [LIVE FIX 24 / B22] the submit-what-was-typed shape, shared by the
#: classifier branch above, the step rewrite, and the sibling strip.
_SUBMIT_TYPED_RE = re.compile(
    r"\b(?:search|submit|look\s+it\s+up|find\s+it)\b[^.?!]{0,48}"
    r"\bwhat\s+(?:you|i)\s+(?:just\s+)?typed\b",
    re.IGNORECASE)

#: [LIVE FIX 22 / B22] a "not X" correction fragment ("not tab search")
#: revises the previous action — a fragment that merges into the next
#: clause, never a standalone research step.
_NOT_CORRECTION_RE = re.compile(
    r"^\s*not\b[^.?!]{1,40}$",
    re.IGNORECASE)

#: [LIVE FIX 28 / B56] a BARE pronoun-tail restatement of a window command
#: ("minimize the two terminal windows on my screen, minimize both of
#: them") is the SAME action restated — one control clause, never a second
#: step (and never a vision fallback for the pronoun tail).
_BARE_WINDOW_PRONOUN_RE = re.compile(
    r"^\s*(?:minimi[sz]e|maximi[sz]e|restore)\s+"
    r"(?:(?:all|both)\s+of\s+)?(?:it|them|both|everything|those|these|all)\b"
    r"[\s.,!]*$",
    re.IGNORECASE)

#: [LIVE FIX 28 / B62] a past-tense FEEDBACK fragment ("last time you
#: didn't type in the YouTube search bar", "you actually typed in the
#: browser's URL bar") revises the previous run — never a standalone step.
_PAST_FEEDBACK_RE = re.compile(
    r"^\s*(?:last\s+time\b|you\s+actually\b|you\s+were\s+supposed\s+to\b|"
    r"earlier\s+you\b|previously\s+you\b)",
    re.IGNORECASE)

#: [LIVE FIX 28 / B56] a "screen" clause is only a LOOK when it carries a
#: look verb; a topic-only screen phrase ("all the windows on my screen",
#: "this terminal window on my screen") is the LOCATOR of an adjacent
#: control action — it folds into the neighbouring control clause instead
#: of splitting a deterministic action into look-then-do vision steps.
_LOOK_VERB_RE = re.compile(
    r"\b(look|check|see|read|watch|analyse|analyze|scan|view|examine)\b",
    re.IGNORECASE)
_QUESTION_WORD_RE = re.compile(
    r"\b(what|which|who|where|when|why|how)\b", re.IGNORECASE)


def _is_submit_shape(text):
    """True when the clause is a submit of already-typed input."""
    t = str(text or "")
    if _BARE_SUBMIT_RE.match(t):
        return True
    if _SUBMIT_TYPED_RE.search(t):
        return True
    if re.search(r"\bclick(?:ing)?\s+enter\b", t, re.IGNORECASE):
        return True
    if (re.search(r"\bsearch\s+(?:icon|button)\b", t, re.IGNORECASE)
            and re.search(r"\benter\b", t, re.IGNORECASE)):
        return True
    return False

#: A screen_control step that points at the previous step's target
#: ("minimize the terminal, then close it") runs after it — the executor
#: re-plans against the fresh screen, so the dependency is ordering-only.
_SC_PRONOUN_RE = re.compile(
    r"\b(it|this|that|them|these|those)\b", re.IGNORECASE)

#: [LIVE FIX 22 / B9] a bare submit clause ("search", "search it",
#: "submit", "press enter") means "submit what I just typed" when it
#: follows a control step — never a web-search step.
_BARE_SUBMIT_RE = re.compile(
    r"^\s*(?:search(?:\s+it)?|submit(?:\s+it)?|(?:press|hit)\s+enter)\s*$",
    re.IGNORECASE)

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
    r"[^.?!]{0,48}\b(?:it|this|that|them|structure|tree|layout|hierarchy)\b"
    # Live fix: the NOUN form — "create a exact replica of this on my
    # desktop" (the verb is "create"; "replica" is the object of the job).
    # The live log showed this phrasing sailing past every replicate gate
    # and reaching the browser handoff as a rewritten task description.
    r"|\b(?:create|make|build|generate)\b[^.?!]{0,48}?\breplica\b"
    r"[^.?!]{0,32}\b(?:of\s+)?(?:it|this|that|them|structure|tree|layout|"
    r"hierarchy)\b",
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
        # [LIVE FIX 26 / B40] a stray leading glyph ("✕ click…") must not
        # survive into a clause — it defeats anchored matchers below.
        part = _strip_stray_glyphs((part or "").strip(" \t,;."))
        # [LIVE FIX 32 / B85] repair a clause-initial STT-mangled "type".
        part = _repair_type_typo(part)
        if len(part) < 3:
            continue
        out.append(part)
    return out


#: [LIVE FIX 25 / B28] a bare typing verb JUXTAPOSED after a completed
#: control clause ("click in the search bar on my screen type mr.beast")
#: is its own step. "Completed" = the left part starts with a control
#: verb and ends on a target noun or a screen phrase, so "click the font
#: type dropdown" / "click the type button" never split.
_JUXTAPOSED_TYPE_RE = re.compile(
    # [LIVE FIX 26 / B38] anchor on the first control verb, not absolute
    # start — a stray leading glyph must never defeat the split.
    r"^[^\w]*(?P<left>(?:(?:please|kindly|jarvis)\b[\s,]*)?"
    r"(?:click|double-?click|right-?click|tap|select|focus|hover|press|hit)\b.*?"
    r"(?:\b(?:bar|box|field|button|icon|link|tab|textbox|input|window|it|this|"
    r"that|there)"
    r"|\b(?:on|at|in)\s+(?:my|the)\s+(?:screen|desktop|browser|window)))"
    r"\s+(?P<right>(?:" + _TYPE_VERB_TOKEN + r"|enter|write)\s+\S.*)$",
    re.IGNORECASE)
_JUXT_RIGHT_BLOCK_RE = re.compile(
    r"^(?:type|enter|write)\s+(?:of|key|button|field|box|bar|icon|tab|"
    r"dropdown|menu)\b", re.IGNORECASE)
#: The same file-noun guard _SCREEN_CONTROL_RE applies to "type <content>".
_JUXT_FILE_NOUN_RE = re.compile(
    r"\b(?:file|folder|directory|report|document|note|txt|script|code)\b",
    re.IGNORECASE)

#: [LIVE FIX 32 / B85] a typing clause followed by a separate submit verb with
#: no connector ("type mr.beast in it press enter") is TWO steps — the type and
#: the submit — never one merged command. The left half carries a type verb
#: (typo-tolerant), the right half is a bare press/hit/click enter|return.
_JUXTAPOSED_SUBMIT_RE = re.compile(
    r"^[^\w]*(?P<left>(?:(?:please|kindly|jarvis)\b[\s,]*)?"
    r"(?:" + _TYPE_VERB_TOKEN + r"|write|enter)\b\s+\S.*?"
    r"(?:\b(?:in|into|inside|on|at)\s+(?:it|this|that|there|"
    r"(?:my|the)\s+(?:[\w\-]+\s+){0,3}?(?:bar|box|field|input|screen|window)))?)"
    r"\s+(?P<right>(?:press|hit|click(?:ing)?)\s+(?:enter|return)\b.*)$",
    re.IGNORECASE)

#: [LIVE FIX 32 / B85] a UI "search bar/box/field" is a TARGET noun, not a
#: web-lookup request — stripped before testing for research markers so a
#: topic phrase naming a search bar is not mistaken for research (which used
#: to defeat the topic-folding guard and turn a control clause into a look).
_UI_SEARCH_NOUN_RE = re.compile(
    r"\bsearch\s*(?:bar|box|field|icon|button|input)\b", re.IGNORECASE)


def _strip_ui_search_nouns(text):
    """Remove UI "search bar/box/field/icon/button/input" nouns. Never raises."""
    try:
        return _UI_SEARCH_NOUN_RE.sub(" ", str(text or ""))
    except Exception:
        return str(text or "")


#: [LIVE FIX 34 / B98] a step whose OBJECT is a previous step's OUTPUT
#: ("play the first video from the search results", "click the top result",
#: "open the first link") depends on the whole preceding screen flow: without
#: the search there are no results. Without this link the step ran anyway and
#: clicked whatever happened to be on screen — the live 14:44-14:46 chain
#: clicked the YouTube home feed believing it showed the results of a search
#: that never happened.
_RESULTS_REF_RE = re.compile(
    r"\b(?:from|in|of|among)\s+(?:the|those|these|its|my)\s+"
    r"(?:search\s+|query\s+)?results?\b"
    r"|\b(?:search|query)\s+results?\b"
    r"|\b(?:first|second|third|1st|2nd|3rd|top|next|last)\s+"
    r"(?:video|result|link|item|match|entry|clip)\b",
    re.IGNORECASE,
)


#: [LIVE FIX 26 / B39] ordinal-play is a COMPLETE screen action ("play the
#: first video") — it never grows a research sibling, even when a trailing
#: relative clause names the search ("…that appears for the search").
_ORDINAL_PLAY_RE = re.compile(
    r"\bplay\b[^.?!]{0,32}\bthe\s+(?:first|second|third|fourth|fifth|"
    r"top|next|last)\b",
    re.IGNORECASE)
#: [LIVE FIX 26 / B39] a trailing result-reference ("for the search", "that
#: appears") is a noun pointing at the results, never a lookup request.
_TRAILING_RESULT_RE = re.compile(
    r"\s+(?:for\s+the\s+(?:search|results?)|that\s+appears?"
    r"(?:\s+for\s+the\s+search)?|from\s+the\s+search)\s*$",
    re.IGNORECASE)


def _split_control_juxtaposition(clauses):
    """Cut 'click X type Y' or 'type X press enter' (no connector) into two control clauses.

    [LIVE FIX 32 / B85] the second split handles a typing clause followed by a
    bare submit ("type mr.beast in it press enter") — previously it stayed one
    merged command.
    """
    out = []
    for clause in clauses or []:
        clause = _repair_type_typo(clause or "")
        m = _JUXTAPOSED_TYPE_RE.match(clause)
        if (m and not _JUXT_RIGHT_BLOCK_RE.match(_repair_type_typo(m.group("right")))
                and not _JUXT_FILE_NOUN_RE.search(m.group("right")[:60])):
            left_c = _repair_type_typo(m.group("left").strip(" \t,;."))
            right_c = _repair_type_typo(m.group("right").strip(" \t,;"))
            out.append(left_c)
            # The right half may itself end with a juxtaposed "press enter".
            m_sub = _JUXTAPOSED_SUBMIT_RE.match(right_c)
            if (m_sub
                    and not _JUXT_FILE_NOUN_RE.search(m_sub.group("left")[:60])):
                out.append(_repair_type_typo(
                    m_sub.group("left").strip(" \t,;.")))
                out.append(m_sub.group("right").strip(" \t,;"))
            else:
                out.append(right_c)
            continue
        m_sub = _JUXTAPOSED_SUBMIT_RE.match(clause)
        if m_sub and not _JUXT_FILE_NOUN_RE.search(m_sub.group("left")[:60]):
            out.append(_repair_type_typo(m_sub.group("left").strip(" \t,;.")))
            out.append(m_sub.group("right").strip(" \t,;"))
        else:
            out.append(clause)
    return out


def classify_clause(text):
    """The single action kind of one clause, or None for a fragment.

    Fragments (chatter, "tell me what you see", answers) return None and
    are merged into the clause they continue.
    """
    t = (text or "").strip()
    if not t:
        return None
    # [LIVE FIX 24 / B22] a leading "not X" correction fragment ("not tab
    # search") revises the previous action — it merges into the next
    # clause instead of classifying on its own keywords.
    # [LIVE FIX 28 / B62] same for a past-tense feedback fragment ("last
    # time you didn't type ...").
    if _NOT_CORRECTION_RE.match(t) or _PAST_FEEDBACK_RE.match(t):
        return None
    # [LIVE FIX 22 / B9] control-before-screen: "click the search bar ON MY
    # SCREEN" is a control, not a look — the screen phrase used to win and
    # the clause became a look step. Look-only clauses ("look at my
    # screen") carry no control verb so they still fall through below.
    if _SCREEN_CONTROL_RE.search(t):
        return "screen_control"
    if _SCREEN_RE.search(t):
        return "screen"
    if _RESEARCH_RE.search(t):
        return "research"
    # [LIVE FIX 21] control verbs act on the screen/windows; file verbs
    # act on files — checked before "task" so "close the window" is a
    # screen action while "type the report in the file" stays a task
    # (no UI/screen target noun there).
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
    raw = _strip_stray_glyphs((text or "").strip())
    if not raw or _FUTURE_TALK_RE.match(raw):
        return None
    clauses = split_clauses(raw)
    # [LIVE FIX 25 / B28] "click X type Y" with no connector is two steps.
    clauses = _split_control_juxtaposition(clauses)
    if not clauses:
        return None
    merged = []
    _pending_topic = None
    for clause in clauses:
        kind = classify_clause(clause)
        # [LIVE FIX 28 / B56] a stashed topic-only screen phrase leads the
        # next CONTROL clause ("all the windows on my screen" + "minimize
        # them" -> one control clause), so one deterministic job never
        # becomes a look-then-do chain. Before any other kind it keeps its
        # own step (the historical look behaviour).
        if _pending_topic:
            if kind == "screen_control":
                clause = _pending_topic + " and " + clause
                _pending_topic = None
            else:
                merged.append({"kind": "screen", "text": _pending_topic})
                _pending_topic = None
        # [LIVE FIX 28 / B56] a screen clause with no look verb and no
        # control verb and no question word is a TOPIC phrase ("all the
        # windows on my screen"): fold it into an adjacent control clause,
        # or stash it to lead the next one. Real looks ("look at my
        # screen"), screen questions, and clauses carrying their own
        # research request keep their own step.
        if (kind == "screen" and not _LOOK_VERB_RE.search(clause)
                and not _SCREEN_CONTROL_RE.search(clause)
                and not _QUESTION_WORD_RE.search(clause)
                and not _RESEARCH_RE.search(_strip_ui_search_nouns(clause))
                and not clause.strip().endswith("?")):
            if merged and merged[-1]["kind"] == "screen_control":
                merged[-1]["text"] = merged[-1]["text"] + " and " + clause
                continue
            _pending_topic = clause
            continue
        # [LIVE FIX 22 / B9] a bare submit ("and then search", "then hit
        # enter") right after a control step means "submit what I just
        # typed" — a control step, never a web-search step.
        if (merged and merged[-1]["kind"] == "screen_control"
                and _BARE_SUBMIT_RE.match(clause)):
            merged.append({"kind": "screen_control", "text": "press enter"})
            continue
        if kind is None:
            if merged:
                merged[-1]["text"] = merged[-1]["text"] + " and " + clause
            continue
        # [LIVE FIX 24 / B22] "search what you typed" is the submit action
        # itself — rewrite to the canonical step text so the planner and
        # the ack narrate "press enter", never a web search.
        if kind == "screen_control" and _SUBMIT_TYPED_RE.search(clause):
            clause = "press enter"
        # [LIVE FIX 28 / B56] a bare pronoun-tail restatement of the
        # previous window command ("...minimize both of them") is the same
        # action — merge, never a second step.
        if (merged and merged[-1]["kind"] == "screen_control"
                and kind == "screen_control"
                and _BARE_WINDOW_PRONOUN_RE.match(clause)
                and re.search(r"\bwindows?\b", merged[-1]["text"],
                              re.IGNORECASE)):
            merged[-1]["text"] = merged[-1]["text"] + " and " + clause
            continue
        # [LIVE FIX 24 / B22] a restated submit ("...by clicking enter or
        # the search icon") is the same action as the submit step before
        # it — not a second step.
        if (merged and merged[-1]["kind"] == "screen_control"
                and kind == "screen_control"
                and _is_submit_shape(merged[-1]["text"])
                and _is_submit_shape(clause)):
            continue
        # [LIVE FIX 21] screen_control clauses never merge: each is its own
        # planned+executed step ("minimize all terminals then click the
        # search bar" is two actions, not one command).
        if merged and merged[-1]["kind"] == kind and kind != "screen_control":
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
        # A "search bar/box" is a UI target, not a lookup request — strip
        # those before testing for research markers, or "click the search
        # bar" would grow a bogus web-search sibling.
        lookup_text = re.sub(r"search\s*(?:bar|box|field)", " ",
                             m["text"], flags=re.IGNORECASE)
        # [LIVE FIX 23 / B14] same for a control verb aimed at the submit
        # target ("click search") — the bare search word there is a
        # button, never a lookup request. A REAL lookup keeps its own
        # words ("…to search for flights" survives this strip).
        lookup_text = re.sub(
            r"\b(?:click|double-?click|right-?click|tap|press|hit)\s+"
            r"(?:the\s+)?(?:search|submit)\b", " ",
            lookup_text, flags=re.IGNORECASE)
        # [LIVE FIX 24 / B22] a "search icon/button" is a UI target and
        # "search what you typed" is a submit — neither is a lookup.
        lookup_text = re.sub(
            r"\bsearch\s+(?:icon|button|tab)\b", " ",
            lookup_text, flags=re.IGNORECASE)
        lookup_text = _SUBMIT_TYPED_RE.sub(" ", lookup_text)
        # [LIVE FIX 26 / B39] a trailing result-reference ("for the search",
        # "that appears") names the results, never a lookup request.
        lookup_text = _TRAILING_RESULT_RE.sub(" ", lookup_text)
        # [LIVE FIX 28 / B62] "whatever search result comes up" names the
        # ON-PAGE results of a control action — never a web-lookup request;
        # the bare "search" word there used to grow a bogus research sibling.
        lookup_text = re.sub(r"\b(?:search|query)\s+results?\b", " ",
                             lookup_text, flags=re.IGNORECASE)
        if (m["kind"] == "screen" and _RESEARCH_RE.search(lookup_text)
                and not (i + 1 < len(merged)
                         and merged[i + 1]["kind"] == "research")):
            expanded.append({"kind": "research", "text": m["text"]})
        # [LIVE FIX 21] same for a CONTROL clause carrying a lookup
        # ("click the search bar to search for flights") — the lookup
        # half must not die with the click.
        # [LIVE FIX 26 / B39] ordinal-play is a complete screen action and
        # never grows a sibling.
        if (m["kind"] == "screen_control"
                and not _ORDINAL_PLAY_RE.search(m["text"])
                and _RESEARCH_RE.search(lookup_text)
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
        _pre_controls = [j for j in range(i)
                         if merged[j]["kind"] == "screen_control"]
        if (m["kind"] == "screen_control" and _pre_controls
                and _RESULTS_REF_RE.search(m["text"])):
            # [LIVE FIX 34 / B98] a step whose object IS a previous step's
            # OUTPUT ("play the first video from the search results", "click
            # the top result") depends on every preceding screen-control
            # step: without the search there are no results, so the click is
            # skipped when the typing/submit failed instead of clicking
            # whatever happens to be on screen — the live 14:44-14:46 chain
            # clicked the YouTube home feed believing it showed the results
            # of a search that never ran.
            consumes = _pre_controls
        elif i > 0 and _ANAPHORA_RE.search(m["text"]):
            consumes = [i - 1]
        elif (i > 0 and merged[i - 1]["kind"] == "screen_control"
              and m["kind"] == "screen_control"
              and _SC_PRONOUN_RE.search(m["text"])):
            # [LIVE FIX 21] "then close it" runs after the step it points
            # at (ordering-only: the executor re-plans on the fresh
            # screen, never replays coordinates).
            consumes = [i - 1]
        elif (i > 0 and merged[i - 1]["kind"] == "screen_control"
              and m["kind"] == "screen_control"
              and _is_submit_shape(m["text"])):
            # [LIVE FIX 32 / B88] a submit step ("press enter") depends on the
            # step it submits: if the typing step failed or blocked, the Enter
            # must not fire blindly (the runner skips a step whose consumed
            # step's status is not "ok").
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
    "screen_control": "run that screen action",
    "research": "research it online",
    "task": "save it to a file",
}

_FOLDER_CLAUSE_RE = re.compile(r"\b(folder|directory)\b", re.IGNORECASE)


_SC_LABEL_FILLER_RE = re.compile(
    r"^(?:please\s+|kindly\s+|jarvis\s*,?\s*)+", re.IGNORECASE)
_SC_LABEL_TAIL_RE = re.compile(
    r"\s+(?:on|at)\s+(?:my|the)\s+screen\s*$", re.IGNORECASE)


def _short_control_label(text):
    """Compact verb-object for a control step ("type mr.beast").

    [LIVE FIX 22 / B9] the chain ack narrates what each step will do
    instead of repeating "control your screen" for every step.
    [LIVE FIX 26 / B40] a stray leading glyph never eats real content out
    of the 7-word window.
    """
    t = re.sub(r"\s+", " ", _strip_stray_glyphs(text)).strip()
    t = _SC_LABEL_FILLER_RE.sub("", t)
    t = _SC_LABEL_TAIL_RE.sub("", t)
    label = " ".join(t.split()[:7])
    if len(label) > 42:
        label = label[:42].rstrip() + "…"
    return label or "control your screen"


def _step_label(step):
    """What this step will actually do — folder jobs never say "file"."""
    kind = step.get("kind")
    text = step.get("text") or ""
    if kind == "screen_control":
        # [LIVE FIX 22 / B9] window management reads as windows, the rest
        # narrates its own verb-object — the ack stays truthful per step.
        if re.search(r"\b(minimize|minimise|maximize|maximise|restore)\b",
                     text, re.IGNORECASE):
            return "manage your windows"
        return _short_control_label(text)
    if kind == "task" and _TASK_REPLICATE_RE.search(text):
        # [LIVE FIX 12] A replication rebuilds the on-screen structure —
        # never "save it to a file".
        return "recreate that structure on your desktop"
    if kind == "task" and _FOLDER_CLAUSE_RE.search(text):
        if re.search(r"\bfile\b", text, re.IGNORECASE):
            return "create a folder with a file inside"
        return "create a folder"
    return _LABELS.get(kind, kind)


def consumes_for(prev_kind, text):
    """True when a follow-up clause depends on the previous step.

    [LIVE FIX 21] chain continuation ("then click the search bar"): the
    same reference rules as build_chain, minus the index arithmetic —
    anaphora, a screen_control pronoun ("then close it"), or a research
    clause pointing at what the screen step just showed.
    """
    t = text or ""
    if _ANAPHORA_RE.search(t):
        return True
    if (prev_kind == "screen_control" and _SC_PRONOUN_RE.search(t)):
        return True
    if (prev_kind == "screen"
            and (_DEICTIC_RES_RE.search(t) or _SCREEN_REF_RE.search(t))):
        return True
    return False


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
