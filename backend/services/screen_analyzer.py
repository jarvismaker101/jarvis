"""Screen Q&A — analyse the current screen and answer user questions.

Captures the primary screen, sends it to a vision model, and returns a
structured response with a main *tip*, optional *evidence* items, a *topic*
keyword and links for the floating overlay.

F48 — provenance: a vision prompt cannot verify anything and must never say
it did. Every evidence item carries where it came from (observed on screen,
inferred by the model, or externally checked by a real lookup), an external
lookup actually RUNS when the question asks for verification instead of being
implied by the prompt, and it verifies the claim the screen DISPLAYS rather
than the deictic words of the question.

F37 — vision dispatch: the vision model is reached through ONE
eligible-provider cascade (backend/services/vision_cascade.py): the
registry-selected provider first, then the eligible configured providers, each
at most once, bounded — no provider is dispatched without a credential, and
exceptions/malformed output advance instead of ending the call.
"""

import base64
import io
import json
import logging
import re
import time

from backend.services import context_state
from backend.services import gemini_client
from backend.services import vision_cascade
from backend.services.grok_client import ask_groq_vision
from backend.services.provenance import (
    PROVENANCE_INFERRED,
    PROVENANCE_OBSERVED,
    normalise_provenance,
    utc_now_iso,
)
from backend.services.screen_capture import capture_primary_screen, capture_region_around_cursor


# ---------------------------------------------------------------------------
# Detection — does the user's utterance look like a screen question?
# ---------------------------------------------------------------------------

_REGION_QUESTION_PATTERNS = [
    r"\bwhat(?:'s| is) this area\b",
    r"\bwhat is this area\b",
    r"\bthis area\b",
    r"\bthis region\b",
    r"\bthis thing\b",
    r"\bthis part\b",
    r"\bthis section\b",
    r"\bhighlighted (?:area|part|section|thing|text|region)\b",
    r"\bwhat does this\b.*\b(?:highlighted|mean|means)\b",
    r"\bpointed(?: out)?\b",
    r"\bcursor\b",
    r"\bpointer\b",
    r"\b(?:yeh|is) (?:area|cheez|kya)\b",
    r"\b(?:yahan|idhar|yahin) kya hai\b",
    r"\barea pe kya hai\b",
    r"\bhighlight kiya\b",
]

_REGION_QUESTION_RE = re.compile(
    "|".join(_REGION_QUESTION_PATTERNS), re.IGNORECASE
)

_SCREEN_QUESTION_PATTERNS = [
    # English — explicit
    r"\bwhat(?:'s| is) (?:on |happening on )?(?:my |the )?screen\b",
    r"\bwhat do you see (?:on |in )?(?:my |the )?screen\b",
    r"\bwhat(?:'s| is) (?:being )?(?:shown|displayed|visible) (?:on |in )?(?:my |the )?screen\b",
    r"\btell me (?:about |what(?:'s| is) on )?(?:my |the )?screen\b",
    r"\banalyze (?:my |the )?screen\b",
    r"\banalyse (?:my |the )?screen\b",
    r"\blook at (?:my |the )?screen\b",
    r"\bdescribe (?:my |the )?screen\b",
    r"\bexplain (?:what(?:'s| is) on )?(?:my |the )?screen\b",
    r"\bread (?:my |the )?screen\b",
    r"\bcan you see (?:my |the )?screen\b",
    r"\bscreen (?:me |pe )?(kya|what)\b",
    r"\bwhat(?:'s| is) (?:going on|happening)\b.*screen\b",
    # Hindi / Hinglish
    r"\bscreen pe kya hai\b",
    r"\bscreen pe kya ho raha\b",
    r"\bscreen dekho\b",
    r"\bscreen batao\b",
    r"\bscreen samjhao\b",
]

_SCREEN_QUESTION_RE = re.compile(
    "|".join(_SCREEN_QUESTION_PATTERNS), re.IGNORECASE
)

# Broad screen references for the fallback detection.
_SCREEN_REFS = [
    "my screen", "the screen", "on screen", "on my screen",
    "screen pe", "screen par", "screen me", "screen mein",
    "screen shows", "screen dikha", "screen dikhao",
    "i can see on my screen",
]

_SCREEN_QUESTION_CUES = (
    "?",
    "what",
    "which",
    "who",
    "why",
    "how",
    "tell me",
    "explain",
    "describe",
    "analyze",
    "analyse",
    "read",
    "showing",
    "visible",
    "more",
    "about",
    # "give me the answer to this KBC question on my screen" has a screen
    # reference and an ask, but no wh-word — the ask IS the cue.
    "answer",
    "question",
    "solve",
    "quiz",
    "kya",
    "kaise",
    "kaun",
    "kyun",
    "batao",
    "samjhao",
)

# If the message contains a screen reference but also one of these verbs
# it's a screen-*control* command, not a question.
_SCREEN_CONTROL_VERBS = {
    "click", "tap", "press", "type", "scroll", "swipe",
    "drag", "drop", "move", "resize", "minimize", "maximize",
    "close", "switch", "navigate", "hover", "right-click",
    "double-click", "select", "highlight",
}


def is_screen_question(text: str) -> bool:
    """Return True if *text* looks like a question about the screen content.

    Two-tier detection:
    1. Exact regex patterns (high confidence).
    2. Broad fallback — any mention of "screen" / "my screen" / etc.
       **unless** the message also contains a screen-control verb
       (click, type, scroll…).
    """
    if not text:
        return False

    # Tier 1 — exact patterns.
    if _SCREEN_QUESTION_RE.search(text):
        return True
    if _REGION_QUESTION_RE.search(text):
        return True

    # Tier 2 — broad "screen" mention without control verbs.
    lower = text.lower()
    has_screen_ref = any(ref in lower for ref in _SCREEN_REFS)
    if has_screen_ref:
        has_question_cue = any(cue in lower for cue in _SCREEN_QUESTION_CUES)
        words = set(re.findall(r"[a-z]+", lower))
        if has_question_cue and not (words & _SCREEN_CONTROL_VERBS):
            return True

    return False


def is_region_question(text: str) -> bool:
    """True when the question points at a specific area ("this / this area /
    highlighted area / cursor / this thing") rather than the whole screen."""
    if not text:
        return False
    has_screen_ref = any(ref in text.lower() for ref in _SCREEN_REFS)
    if has_screen_ref and _REGION_QUESTION_RE.search(text):
        return True
    # "what does this highlighted area mean" without the word "screen"
    return bool(_REGION_QUESTION_RE.search(text)) and any(
        cue in text.lower() for cue in ("mean", "means", "is", "what", "kya", "kia")
    )


# ---------------------------------------------------------------------------
# Analysis — capture screen and ask a vision model
# ---------------------------------------------------------------------------

# F48: the prompt no longer claims the model can search. It cannot — the
# vision call runs with search grounding OFF — so asking it to "verify" only
# produced confident unverified text. Instead it must separate what it can
# actually SEE from what it is GUESSING.
_ANALYSIS_PROMPT = (
    "You are Jarvis, a smart AI assistant with vision capabilities. "
    "The user is looking at their screen and asked you a question about it.\n\n"
    "Study the screenshot carefully and answer the user's question. "
    "You have NO internet access: never claim to have searched, checked or "
    "verified anything. Work only from what is visible in the screenshot.\n\n"
    "Never answer that you cannot create files or folders — you can, "
    "through your task routes — and never offer command-line steps instead. "
    "When the user asks to recreate, copy or duplicate something visible, "
    "describe it and tell them to just ask you to recreate it.\n\n"
    "Return a JSON object with exactly these keys:\n"
    '  "tip"  — A concise, direct answer to the user\'s question (1-3 sentences max).\n'
    '  "evidence" — A JSON array of 0-3 supporting detail objects, each with:\n'
    '      "source" — short label for where the detail comes from (app name, website, etc.)\n'
    '      "title"  — short headline of the detail\n'
    '      "snippet" — one-sentence description\n'
    '      "provenance" — one of "observed", "inferred". Use "observed" ONLY '
    "when the detail is literally visible in the screenshot. Use \"inferred\" "
    "when it is your own knowledge or reading-between-the-lines.\n"
    '  "topic" — The main subject/topic visible or discussed on screen '
    "(2-5 words, used for fetching related images and links).\n"
    '  "creator" — The exact name of the person, channel or account that '
    "made the main video/media visible on screen (YouTube channel name, "
    "uploader, streamer handle). Empty string when no media or no name is "
    "visible.\n"
    '  "show_images" — true ONLY if showing 1-2 images would genuinely help '
    "the user understand this (e.g. identifying an object, landmark, artwork, "
    "celebrity, product, chart). Otherwise false — most UI/screen questions "
    "do NOT need images.\n\n"
    "If the screen content is straightforward and doesn't need evidence, "
    "return an empty evidence array.\n\n"
    "Keep the tip conversational and address the user as 'sir' occasionally.\n\n"
    "User question: {question}"
)

# F48: when the user asks for a CHECK, a check has to happen. These are
# deliberately narrow — a web lookup costs seconds and must not fire on every
# screen question.
_EXTERNAL_CHECK_PATTERNS = (
    r"\bverif(y|ied|ication)\b",
    r"\bfact[\s-]?check\b",
    r"\bis (this|that|it) (true|real|accurate|correct|genuine)\b",
    r"\btrue or (not|false)\b",
    r"\breal or (fake|not)\b",
    r"\bcheck (this|that|if|whether)\b",
    r"\bcurrent (price|version|value|status)\b",
    r"\blatest (news|version|price|update)\b",
)


def needs_external_check(question):
    """True when the question asks for verification the screen cannot give.

    F48: a vision prompt cannot verify anything. When the user actually asks
    for a check, a real lookup runs instead of the prompt implying one.
    """
    text = (question or "").lower()
    return any(re.search(pattern, text) for pattern in _EXTERNAL_CHECK_PATTERNS)


# F48: the deictic parts of a check request ("is THIS true", "verify that",
# "fact check this") carry no information — searching them verbatim looks up
# the words "is this true" instead of the claim the user is pointing at. Only
# the scaffolding is stripped; the user's own subject words are preserved
# ("verify the height of the Eiffel Tower" -> "the height of the Eiffel
# Tower").
_DEICTIC_CHECK_RE = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+|could you\s+)?"
    r"(?:verif(?:y|ied|ication)|fact[\s-]?check|check)\s+|"
    r"\b(?:is|are|was|were)\s+(?:this|that|it)\s+"
    r"(?:true|real|accurate|correct|genuine|fake)\b|"
    r"\b(?:true\s+or\s+(?:not|false)|real\s+or\s+fake)\b|"
    r"\b(?:this|that|it)\b|"
    r"^\s*(?:please|sir)\b",
    re.IGNORECASE,
)


def _check_query(question, claim=""):
    """The text a verification lookup should actually search (F48).

    F48 acceptance: "Is this true" must verify the DISPLAYED claim. The
    deictic scaffolding is stripped; the claim the screen actually shows is
    what gets searched (and appended when the question adds its own target,
    e.g. "verify the price of bitcoin").
    """
    claim = " ".join(str(claim or "").split())
    residual = _DEICTIC_CHECK_RE.sub(" ", str(question or ""))
    residual = " ".join(residual.split()).strip(" ?.!,\"'")
    if len(residual) < 3:
        residual = ""
    if residual and claim:
        base = "%s — %s" % (residual, claim)
    else:
        base = residual or claim
    return base.strip()


def _external_check(question, claim=""):
    """Run a real lookup of the DISPLAYED claim and return it as evidence.

    Best-effort: a failed lookup yields no item rather than a fake one — the
    "externally_checked" label is only ever attached to a lookup that ran.
    F48: a deictic question with no claim to check returns None (no lookup is
    fabricated for the words "is this true"), and the result records the
    claim, the exact query and a claim-level span/lookup reference so the
    answer can never present a generic summary as support it does not have.
    """
    from backend.services.provenance import (
        PROVENANCE_EXTERNALLY_CHECKED,
        UNCERTAINTY_UNVERIFIED,
        SourceSpan,
        utc_now_iso,
    )
    from backend.services.quick_search import fetch_ai_overview_text, google_search_url

    query = _check_query(question, claim)
    if not query:
        # Nothing was displayed to check and the question itself is deictic:
        # running a lookup would "verify" nothing at all.
        logging.info("[SCREEN-QA] verification asked with no displayed claim "
                     "to check — no lookup run")
        return None
    try:
        overview = fetch_ai_overview_text(query)
    except Exception as exc:  # noqa: BLE001 - a failed check is not fatal
        logging.warning("[SCREEN-QA] external check failed: %s", exc)
        return None
    text = (overview or "").strip()
    if not text:
        return None
    source_url = google_search_url(query)
    retrieved_at = utc_now_iso()
    span = SourceSpan(
        url=source_url,
        quote=text[:300],
        publisher="search-engine AI summary",
        retrieved_at=retrieved_at,
        # An AI summary of other people's pages is never independent support.
        independent=False,
    )
    return {
        "source": "Web lookup",
        "title": "External check",
        "snippet": text[:300],
        "provenance": PROVENANCE_EXTERNALLY_CHECKED,
        "source_url": source_url,
        "query": query,
        "claim": claim,
        "spans": [span.to_dict()],
        "lookup": {"query": query, "url": source_url,
                   "provider": "quick_search.ai_overview",
                   "retrieved_at": retrieved_at},
        "corroboration": {"independent": 0, "corroborated": False,
                          "level": UNCERTAINTY_UNVERIFIED,
                          "secondary_sources": ["search-engine AI summary"]},
        "uncertainty": ("from a live web lookup (search-engine AI summary) — "
                        "not independent corroboration"),
        "retrieved_at": retrieved_at,
    }


def _displayed_claim(tip, evidence):
    """The claim the screen actually displays (F48).

    A verification request ("is this true?") must be checked against what the
    user is LOOKING AT: the answer plus the observed/read text behind it —
    never against the deictic words of the question, which assert nothing.
    """
    parts = []
    text = " ".join(str(tip or "").split())
    if text:
        parts.append(text)
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        if normalise_provenance(item.get("provenance")) != PROVENANCE_OBSERVED:
            continue
        snippet = " ".join(str(item.get("snippet") or "").split())
        if snippet and snippet not in parts:
            parts.append(snippet)
    claim = " — ".join(parts).strip()
    return claim[:600]


def _extract_json(text: str) -> dict:
    """Best-effort JSON extraction from model output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _extract_result_content(result):
    """Return the assistant text from an OpenAI-shaped vision result."""
    if not result or not result.get("choices"):
        return None
    content = result["choices"][0].get("message", {}).get("content", "")
    if content and content.strip():
        return content
    return None


def _resolve_vision_model():
    """Vision model per call: registry vision_model else env default."""
    try:
        from backend.services import model_registry
        sel = model_registry.get_model_for_role("vision")
        prov = str(sel.get("provider") or "").strip()
        mod = str(sel.get("model") or "").strip()
        if prov and mod:
            return {"provider": prov, "model": mod}
    except Exception:
        pass
    try:
        from backend.services.gemini_client import GEMINI_MODEL as _GM
        return {"provider": "gemini", "model": _GM}
    except Exception:
        return {"provider": "gemini", "model": "gemini-3.5-flash-lite"}


# ── F37: one eligible-provider cascade ────────────────────────────────────
# The old cascade hard-coded a gemini->groq tail and called the final Groq
# adapter OUTSIDE any try/except, so a single-provider installation still
# needed a Groq key and one adapter exception ended the whole call. Eligibility,
# ordering, bounding and the attempt record now live in ONE place
# (backend/services/vision_cascade.py); this module only supplies the adapters
# and the response-shape check.

def _dispatch_gemini(prompt, image_data_url, model, max_completion_tokens, response_format):
    return gemini_client.ask_gemini_vision(
        prompt,
        image_data_url,
        max_completion_tokens=max_completion_tokens,
        response_format=response_format or {"type": "json_object"},
        use_google_search=False,
        model=model,
    )


def _dispatch_openrouter(prompt, image_data_url, model, max_completion_tokens, response_format):
    from backend.services.openrouter_client import ask_openrouter_vision
    result = ask_openrouter_vision(
        prompt,
        image_data_url,
        max_completion_tokens=max_completion_tokens,
        model=model,
        response_format=response_format,
    )
    if result:
        result.setdefault("grounding_links", [])
    return result


def _dispatch_fireworks(prompt, image_data_url, model, max_completion_tokens, response_format):
    from backend.services.fireworks_client import ask_fireworks_vision
    result = ask_fireworks_vision(
        prompt,
        image_data_url,
        max_completion_tokens=max_completion_tokens,
        model=model,
        response_format=response_format,
    )
    if result:
        result.setdefault("grounding_links", [])
    return result


def _dispatch_groq(prompt, image_data_url, model, max_completion_tokens, response_format):
    # NOTE: no response_format — Qwen's  thinking blocks break Groq's strict JSON
    # validation; the prompt asks for JSON and _extract_json parses it. The
    # generous budget covers the tokens Qwen burns on  thinking before answering.
    result = ask_groq_vision(
        prompt,
        image_data_url,
        max_completion_tokens=max(max_completion_tokens, 4096),
        model=model,
    )
    if result:
        result.setdefault("grounding_links", [])
    return result


def _with_empty_grounding(dispatch):
    """Wrap a generic adapter so its result carries the empty
    ``grounding_links`` default every shipped adapter gets here."""
    def wrapped(prompt, image_data_url, model, max_completion_tokens,
                response_format):
        result = dispatch(prompt, image_data_url, model,
                          max_completion_tokens, response_format)
        if result:
            result.setdefault("grounding_links", [])
        return result
    return wrapped


def _vision_dispatchers():
    """provider id -> adapter call for this call site (resolved lazily).

    The adapters are module-level functions so existing patch points
    (``screen_analyzer.ask_groq_vision``, the client modules) keep working.
    User-added custom providers ride the shared generic OpenAI-compatible
    vision adapter, wrapped with this site's empty-grounding default.
    """
    dispatchers = {
        "gemini": _dispatch_gemini,
        "openrouter": _dispatch_openrouter,
        "fireworks": _dispatch_fireworks,
        "groq": _dispatch_groq,
    }
    for pid, fn in vision_cascade.custom_vision_dispatchers().items():
        dispatchers[pid] = _with_empty_grounding(fn)
    return dispatchers


def _screen_vision_schema(result):
    """F37 schema validation: nonempty is NOT success — JSON is.

    The old check (``_extract_result_content(result) is not None``) accepted
    any nonempty text, so a malformed first response ended the cascade. A
    response only counts as usable when it contains a JSON object.
    """
    content = _extract_result_content(result)
    if content is None:
        return False, "no assistant content"
    parsed = _extract_json(content)
    if not isinstance(parsed, dict) or not parsed:
        return False, "content was not a JSON object"
    return True, ""


def _ask_screen_vision_cascade(prompt, image_data_url, max_completion_tokens=800,
                               attempts_out=None):
    """Vision for screen Q&A through the ONE eligible-provider cascade (F37).

    The registry-selected (provider, model) is attempted first when it is
    eligible; the remaining ELIGIBLE configured providers follow, each at most
    once, bounded by the cascade's attempt cap. A provider with no credential
    is never dispatched (no unrelated key is required), adapter exceptions and
    schema-invalid output advance to the next provider, and *attempts_out*
    (optional dict) receives the actual-attempt record.
    """
    result, report = vision_cascade.ask_vision_with_fallback(
        prompt,
        image_data_url,
        dispatchers=_vision_dispatchers(),
        validate=_screen_vision_schema,
        selected=_resolve_vision_model(),
        max_completion_tokens=max_completion_tokens,
        response_format={"type": "json_object"},
        attempts_out=attempts_out,
    )
    if report.get("usable"):
        logging.info("[SCREEN-QA] Vision response from %s (%s)",
                     report.get("provider"), report.get("model") or "default")
        return result
    logging.warning("[SCREEN-QA] %s", report.get("unavailable_reason"))
    return result if result is not None else {}


# ---------------------------------------------------------------------------
# RANK 1 (external redesign, Step 2): capture ONCE, keep the observation, and
# let the VISION model resolve which on-screen item the user is pointing at.
# The base F48 analysis prompt above still asks for the spoken answer and
# evidence; the extra jobs below add the item inventory and pointer
# resolution. A text-only model never picks screen targets.
# ---------------------------------------------------------------------------

#: The extra jobs appended to the F48 analysis prompt (RANK 1, report 1.6).
_IDENTIFY_EXTRA = (
    "\n\nALSO return an \"items\" array (up to 6) of separately nameable "
    "things a user could want researched — playing or featured video, "
    "image or thumbnail, post, comment, headline, product, model card, "
    "verse or quote block, person, or window — most prominent first. Each "
    "item needs: \"id\" (\"i1\", \"i2\", ...); \"label\" with the EXACT "
    "displayed title or name copied character by character — never "
    "paraphrase, translate, complete or correct the spelling; when the UI "
    "cuts it off, copy what is visible and set \"truncated\" to true; when "
    "no text is shown, describe it in at most 6 words and set "
    "\"label_is_text\" to false; \"kind\"; \"creator\" with the exact "
    "displayed channel or account name, otherwise empty; \"bbox\" as "
    "[x0,y0,x1,y1] in a 0-1000 scale (left, top, right, bottom); and "
    "\"primary\" true for the one playing, focused or central item.\n\n"
    "POINTER PHRASE: \"{target_text}\"\n"
    "SPATIAL HINT: {spatial}\n"
    "REJECTED BY USER (never choose): {rejected}\n\n"
    "Then say which item the POINTER PHRASE means: include \"target_ids\" "
    "(list of item ids), \"n_fit\" (how many items plausibly fit), "
    "\"match\" (exact, likely, guess or none) and \"why\" (at most 15 "
    "words citing visible evidence). Prefer the item that is playing, "
    "focused or central over items that merely share a word with the "
    "phrase. When the pointer phrase is empty, use match \"none\" and an "
    "empty target_ids list."
)

_IDENTIFY_REGION_NOTE = (
    "The image I sent you is a CROPPED region of the user's screen near "
    "their cursor/highlight — answer ONLY about what is inside it."
)

#: Instruction scaffolding that must never pass as an identified label or
#: become a research query (report 1.10).
_INSTRUCTION_SCAFFOLD_RE = re.compile(
    r"\b(?:research|search|find\s+out|look\s+up)\b[^.]{0,24}?"
    r"\b(?:about|for|it|this|that)\b|"
    r"\bi\s+want\s+to\s+(?:know|research|find)\b|"
    r"\btell\s+me\b|\bon\s+my\s+scr+[ae]*n\b|\bjarvis\b",
    re.IGNORECASE,
)

#: Grammar/instruction words that name nothing by themselves.
_GRADE_FILLERS = {
    "this", "that", "these", "those", "it", "them", "the", "a", "an", "of",
    "on", "in", "my", "screen", "video", "image", "picture", "thing", "one",
    "item", "please", "sir", "about", "and", "for", "from", "me", "know",
    "want", "research", "find", "out", "tell",
}


def _sanitize_label(value, limit=140):
    """Screen text is DATA: strip control characters, collapse whitespace."""
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", str(value or ""))
    text = " ".join(text.split())
    return text[:limit].strip()


def _clean_bbox(value):
    try:
        nums = [float(v) for v in value[:4]]
    except Exception:
        return (0, 0, 0, 0)
    return tuple(max(0.0, min(1000.0, n)) for n in nums)


def _clean_items(raw):
    """The model's item inventory, sanitized and bounded (max 6)."""
    if not isinstance(raw, list):
        return []
    items = []
    seen = set()
    for idx, entry in enumerate(raw[:6], start=1):
        if not isinstance(entry, dict):
            continue
        item_id = _sanitize_label(entry.get("id"), 12) or ("i%d" % idx)
        if item_id in seen:
            item_id = "%s_%d" % (item_id, idx)
        seen.add(item_id)
        label = _sanitize_label(entry.get("label"))
        label_is_text = bool(entry.get("label_is_text", True))
        if not label and label_is_text:
            continue
        items.append(context_state.Item(
            id=item_id,
            label=label,
            label_is_text=label_is_text,
            truncated=bool(entry.get("truncated", False)),
            kind=_sanitize_label(entry.get("kind"), 24).lower() or "other",
            creator=_sanitize_label(entry.get("creator"), 120),
            bbox=_clean_bbox(entry.get("bbox")),
            primary=bool(entry.get("primary", False)),
        ))
    return items


def _named_tokens(text):
    """Content words the user named (for the "named word missing" cap)."""
    return [t for t in re.findall(r"[a-z0-9]+", str(text or "").lower())
            if len(t) >= 4 and t not in _GRADE_FILLERS]


def grade_target(obs, parsed, target_text="", granularity="single"):
    """Post-validate the vision model's target choice (report 1.6).

    Returns {"ids", "n_fit", "match", "why", "ambiguous", "capped",
    "labels"}. "match" is one of exact | likely | guess | none; an
    ambiguous single-item request (2+ plausible items) is flagged instead
    of silently binding one.
    """
    why = _sanitize_label(parsed.get("why"), 200)
    match = str(parsed.get("match") or "").strip().lower()
    if match not in ("exact", "likely", "guess", "none"):
        match = "guess" if parsed.get("target_ids") else "none"
    raw_ids = parsed.get("target_ids") or []
    if not isinstance(raw_ids, list):
        raw_ids = []
    by_id = {item.id: item for item in (obs.items or [])}
    ids = []
    for raw_id in raw_ids:
        sid = str(raw_id).strip()
        if sid in by_id and sid not in ids:
            ids.append(sid)
    try:
        n_fit = int(parsed.get("n_fit"))
    except Exception:
        n_fit = len(ids)
    n_fit = max(0, min(9, n_fit))
    capped = []
    if not ids:
        match = "none"
    else:
        chosen = [by_id[i] for i in ids]
        if any(item.truncated or not item.label_is_text for item in chosen):
            if match == "exact":
                match = "likely"
            capped.append("truncated_label")
        if match == "exact" and not why:
            match = "likely"
            capped.append("no_why")
        if any(_INSTRUCTION_SCAFFOLD_RE.search(item.label)
               or item.label.strip().lower()
               == str(obs.utterance or "").strip().lower()
               for item in chosen):
            match = "none"
            ids = []
            capped.append("scaffolded_label")
        named = _named_tokens(target_text)
        if ids and named and match in ("exact", "likely"):
            haystack = (" ".join(by_id[i].label for i in ids)
                        + " " + why).lower()
            if not any(tok in haystack for tok in named):
                if match == "exact":
                    match = "likely"
                capped.append("named_word_missing")
    if match == "none":
        ids = []
        labels = []
    else:
        labels = [by_id[i].label for i in ids]
    ambiguous = bool(n_fit >= 2 and granularity == "single")
    return {"ids": ids, "n_fit": n_fit, "match": match, "why": why,
            "ambiguous": ambiguous, "capped": capped, "labels": labels}


def _dhash(image_data_url):
    """64-bit difference hash of a capture ("" when it cannot be computed)."""
    try:
        from PIL import Image  # already a hard dependency of screen_capture
        if not image_data_url or "," not in image_data_url:
            return ""
        raw = base64.b64decode(image_data_url.split(",", 1)[1])
        gray = Image.open(io.BytesIO(raw)).convert("L").resize(
            (9, 8), Image.LANCZOS)
        pixels = gray.tobytes()
        bits = 0
        for row in range(8):
            for col in range(8):
                bits = (bits << 1) | (
                    1 if pixels[row * 9 + col] > pixels[row * 9 + col + 1]
                    else 0)
        return "%016x" % bits
    except Exception:
        return ""


def capture_observation(mode="full"):
    """Capture the screen ONCE (RANK 1). Mode is explicit, never inferred."""
    mode = str(mode or "full").lower()
    if mode == "region":
        capture = capture_region_around_cursor()
    else:
        mode = "full"
        capture = capture_primary_screen()
    data_url = str(capture.get("image_data_url") or "")
    return {
        "image_data_url": data_url,
        "region": capture.get("region") or None,
        "mode": mode,
        "img_hash": _dhash(data_url),
    }


def capture_stored_observation(utterance="", mode="full"):
    """Capture the screen ONCE and store it as an Observation (LIVE FIX 13).

    The construction analyze_screen performs after its vision call — minus
    the vision call — so a caller that needs a fresh stored image (the
    folder-tree reader's fallback ladder) can re-read it later by id.
    Returns the stored Observation.
    """
    cap = capture_observation(mode)
    obs = context_state.Observation(
        id=context_state.OBSERVATIONS.next_id(),
        at=time.time(),
        mode=str(cap.get("mode") or "full"),
        utterance=str(utterance or ""),
        image_data_url=str(cap.get("image_data_url") or ""),
        img_hash=str(cap.get("img_hash") or ""),
        region=cap.get("region") or None,
    )
    context_state.OBSERVATIONS.add(obs)
    return obs


def _identify_prompt(utterance, target_text="", spatial="", rejected=(),
                     mode="full"):
    base = _ANALYSIS_PROMPT.format(question=utterance)
    if str(mode or "full").lower() == "region":
        base = _IDENTIFY_REGION_NOTE + "\n\n" + base
    return base + _IDENTIFY_EXTRA.format(
        target_text=_sanitize_label(target_text, 200) or "(none)",
        spatial=_sanitize_label(spatial, 60) or "none",
        rejected=", ".join(sorted(str(r) for r in (rejected or ()))) or "none",
    )


def _identify(utterance, target_text="", spatial="", rejected=(), image=None,
              mode="full", granularity="single", attempts_out=None):
    """(obs, grade, meta) - shared engine for RANK 1 identification."""
    cap = image
    if not cap:
        try:
            cap = capture_observation(mode)
        except Exception as exc:  # capture failure is asked about, not guessed
            return None, {
                "ids": [], "n_fit": 0, "match": "none", "why": "",
                "ambiguous": False, "capped": ["capture_failed"],
                "labels": [], "error": "capture", "detail": str(exc),
            }, {}
    cap_mode = str(cap.get("mode") or mode or "full")
    prompt = _identify_prompt(utterance, target_text, spatial, rejected,
                              mode=cap_mode)
    attempts = attempts_out if attempts_out is not None else {}
    result = _ask_screen_vision_cascade(
        prompt, str(cap.get("image_data_url") or ""),
        max_completion_tokens=1100, attempts_out=attempts)
    if not result or not result.get("choices"):
        return None, {
            "ids": [], "n_fit": 0, "match": "none", "why": "",
            "ambiguous": False, "capped": ["vision_unavailable"],
            "labels": [], "error": "vision",
            "detail": vision_cascade.unavailable_reason(attempts) or "",
        }, {"attempts": attempts, "cap": cap}
    content = (result.get("choices") or [{}])[0].get(
        "message", {}).get("content", "") or ""
    parsed = _extract_json(content)
    if not isinstance(parsed, dict):
        parsed = {}
    items = _clean_items(parsed.get("items"))
    answer = str(parsed.get("answer") or parsed.get("tip") or "").strip()
    obs = context_state.Observation(
        id=context_state.OBSERVATIONS.next_id(),
        at=time.time(),
        mode=cap_mode,
        utterance=str(utterance or ""),
        answer=answer,
        items=items,
        rejected=set(str(r) for r in (rejected or ())),
        image_data_url=str(cap.get("image_data_url") or ""),
        img_hash=str(cap.get("img_hash") or ""),
        region=cap.get("region") or None,
        vision_provider=attempts.get("provider"),
        vision_model=attempts.get("model"),
        vision_attempts=list(attempts.get("attempts") or []),
        vision_degraded=bool(attempts.get("degraded")),
    )
    context_state.OBSERVATIONS.add(obs)
    grade = grade_target(obs, parsed, target_text=target_text,
                         granularity=granularity)
    meta = {"attempts": attempts, "parsed": parsed, "content": content,
            "cap": cap, "grounding_links": result.get("grounding_links") or []}
    return obs, grade, meta


def identify_on_screen(utterance, target_text="", spatial="", rejected=(),
                       image=None, mode="full", granularity="single",
                       attempts_out=None):
    """One vision call => one stored Observation + the graded target (RANK 1).

    The vision model (which sees the pixels) resolves which on-screen item
    the user means; the observation keeps the image, so later turns and
    corrections re-ask on the SAME picture.
    """
    obs, grade, _meta = _identify(
        utterance, target_text=target_text, spatial=spatial, rejected=rejected,
        image=image, mode=mode, granularity=granularity,
        attempts_out=attempts_out)
    return obs, grade


def reidentify(obs, correction, rejected=()):
    """Re-ask vision on the SAME stored image (no new capture)."""
    if obs is None or not getattr(obs, "image_data_url", ""):
        return None, {
            "ids": [], "n_fit": 0, "match": "none", "why": "",
            "ambiguous": False, "capped": ["no_image"], "labels": [],
            "error": "no_image",
        }
    capture = {"image_data_url": obs.image_data_url,
               "region": getattr(obs, "region", None),
               "mode": getattr(obs, "mode", "full"),
               "img_hash": getattr(obs, "img_hash", "")}
    merged = set(getattr(obs, "rejected", set()) or set())
    merged.update(str(r) for r in (rejected or ()))
    return identify_on_screen(
        str(correction or ""), target_text=str(correction or ""),
        image=capture, mode=getattr(obs, "mode", "full"), rejected=merged)


def get_observation(obs_id):
    """The stored observation with this id, or None (RAM only)."""
    return context_state.OBSERVATIONS.get(obs_id)


# ---------------------------------------------------------------------------
# LIVE FIX 12/13 — read the folder structure the screen shows, so a
# "replicate it exactly on my desktop" request can rebuild it (and a later
# "create the files as well" can fill it in).
# ---------------------------------------------------------------------------

#: The project-tree reader job sheet. Folders AND files in one read — the
#: user wants the structure visible on this screen recreated on their
#: computer. Names are copied exactly; nesting comes from the display.
_PROJECT_TREE_PROMPT = (
    "Screen-reading task: the user wants the folder structure visible on "
    "this screen recreated on their computer. A file tree is often in a "
    "SIDEBAR or PANEL, not the main area — check the left and right edges, "
    "the editor's explorer panel, open terminal output, archive windows and "
    "any listing before concluding there is none. If you see ANY folder or "
    "file listing anywhere, report it; only reply with the empty object "
    "when genuinely NO listing exists on the whole screen.\n\n"
    'Reply with JSON only, exactly: {"folders": [{"name": "<exact displayed '
    'folder name>", "depth": 0}], "files": [{"name": "<exact displayed file '
    'name with extension>", "folder": "<the folder it sits in, as displayed; '
    'empty string at the top level>"}]} — one entry per FOLDER in the first '
    "array, top to bottom as displayed, depth 0 for a top-level folder and "
    "+1 for each nesting level of indentation shown; one entry per FILE in "
    'the second array (never folders). Copy every name character by '
    'character; never put a path inside "name", never list files in the '
    'folders array, never add commentary. Files at the top level use an '
    'empty "folder".\n\n'
    "FORMAT example only — never copy these names: a structure showing a "
    "top folder 'myapp' containing 'src' with 'index.js' and a top-level "
    "'README.md' is "
    '{"folders": [{"name": "myapp", "depth": 0}, {"name": "src", "depth": 1}], '
    '"files": [{"name": "index.js", "folder": "src"}, '
    '{"name": "README.md", "folder": ""}]}.\n\n'
    'If no folder or file listing is visible ANYWHERE on the screen, reply '
    '{"folders": [], "files": []}.'
)


def _flatten_tree_entries(entries):
    """Vision folder entries → flat {"name", "depth"} list (LIVE FIX 13).

    Tolerant of the shapes models actually emit: a top-level list, a
    {"folders": [...]} wrapper, nested "children" arrays and string depths.
    """
    flat = []

    def walk(items, depth):
        for entry in items or []:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            try:
                own = int(entry.get("depth", depth))
            except Exception:
                own = depth
            own = max(0, own)
            flat.append({"name": name, "depth": own})
            children = entry.get("children")
            if isinstance(children, list):
                walk(children, own + 1)

    if isinstance(entries, dict):
        entries = entries.get("folders")
    if isinstance(entries, list):
        walk(entries, 0)
    return flat


def extract_project_tree(obs):
    """The folders AND files a stored screen observation shows (LIVE FIX 13).

    One vision read of the SAME stored image (never a re-capture). Returns
    (folder_entries, file_entries) — flat {"name", "depth"} dicts and raw
    {"name", "folder"} dicts — or ([], []) when the screen shows no
    readable structure. Never raises.
    """
    try:
        image = str(getattr(obs, "image_data_url", "") or "")
        if not image:
            return [], []
        attempts = {}
        result = _ask_screen_vision_cascade(
            _PROJECT_TREE_PROMPT, image, max_completion_tokens=2000,
            attempts_out=attempts)
        content = (result.get("choices") or [{}])[0].get(
            "message", {}).get("content", "") or ""
        parsed = _extract_json(content)
        if isinstance(parsed, list):
            if not parsed:
                flat = re.sub(r"\s+", " ", content).strip()[:200]
                print("[SCREEN] Project-tree read: %s returned an empty list "
                      "(reply: %s)" % (attempts.get("provider") or "?", flat))
            return _flatten_tree_entries(parsed), []
        if not isinstance(parsed, dict) or not parsed:
            flat = re.sub(r"\s+", " ", content).strip()[:200]
            print("[SCREEN] Project-tree read: unparseable reply from %s: %s"
                  % (attempts.get("provider") or "?", flat))
            return [], []
        folders = _flatten_tree_entries(parsed.get("folders"))
        files = parsed.get("files")
        files = files if isinstance(files, list) else []
        if not folders:
            # Parsed-but-empty is the quiet killer: a valid JSON object with
            # an empty folders array (the prompt's own give-up shape) used to
            # read as "no structure" with NO trace anywhere. Always leave a
            # line with the provider and what it actually said.
            flat = re.sub(r"\s+", " ", content).strip()[:200]
            print("[SCREEN] Project-tree read: %s returned 0 folders "
                  "(reply: %s)" % (attempts.get("provider") or "?", flat))
        return folders, files
    except Exception as exc:
        logging.warning("[SCREEN] Project-tree read failed: %s", exc)
        return [], []


def extract_folder_tree(obs):
    """Compat (LIVE FIX 12 callers): folders only, flat vision entries."""
    return extract_project_tree(obs)[0]


# ---------------------------------------------------------------------------
# RANK 10 — the model composes the search query from the observation
# ---------------------------------------------------------------------------

#: The query-writer job sheet. The model may merge the target with the
#: user's question terms — it may never invent a name or write instructions.
_QUERY_WRITER_SYSTEM = (
    "You write ONE web search query for a voice assistant's browser. You "
    "are given what is on the user's screen (exact text read from the "
    "screen, item kinds, creators, and the vision model's description) and "
    "the user's request. Compose the query that best finds exactly what the "
    "user asked about. Rules: keep every name and title exactly as shown; "
    "merge the user's question terms with the target (for example a movie "
    "or video title plus the thing they asked about); never invent names, "
    "people or dates; never include instruction words (research, search, "
    "look up, tell me, on my screen, please, sir); output ONLY the query, "
    "one line, under 120 characters. Never answer the question yourself."
)

#: Instruction scaffolding that must never reach a search box.
_QUERY_SCAFFOLD_RE = re.compile(
    r"\b(?:research|search|look\s*up|tell\s+me|on\s+my\s+screen|"
    r"please|jarvis|sir)\b",
    re.IGNORECASE,
)


def _clean_composed_query(text):
    """First line, unquoted, scaffold-free — "" when nothing usable."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    line = raw.splitlines()[0].strip().strip("\"'`*# ")
    line = re.sub(r"\s+", " ", line)
    if len(line) < 2 or _QUERY_SCAFFOLD_RE.search(line):
        return ""
    if len(line) > 180:
        cut = line.rfind(" ", 0, 177)
        line = (line[:cut] if cut > 0 else line[:177]).rstrip() + "..."
    return line


def compose_search_query(observation, request_text, fallback=""):
    """The model composes the query from the target description + request.

    The observation carries what the eyes saw (exact item labels, kinds,
    creators, the vision description); the request carries what the user
    wants to know. The model merges them into one search phrase. Any
    failure — provider down, malformed answer, leaked instruction words —
    returns *fallback* (the caller's deterministic subject+aspect compose),
    so a search never depends on the composer.
    """
    fallback = str(fallback or "").strip()
    if observation is None:
        return fallback
    items = list(getattr(observation, "items", []) or [])
    answer = str(getattr(observation, "answer", "") or "").strip()
    if not items and not answer:
        return fallback
    try:
        rows = []
        for item in items[:6]:
            bits = [str(getattr(item, "label", "") or "")]
            kind = str(getattr(item, "kind", "") or "")
            creator = str(getattr(item, "creator", "") or "")
            if kind:
                bits.append("(%s)" % kind)
            if creator:
                bits.append("by %s" % creator)
            if getattr(item, "primary", False):
                bits.append("[primary]")
            if getattr(item, "truncated", False):
                bits.append("[title cut off]")
            rows.append(" ".join(b for b in bits if b))
        seen = "SCREEN ITEMS (exact text read from the screen):\n" + \
            ("\n".join("- " + row for row in rows) if rows else "- (none)")
        if answer:
            seen += "\nVISION DESCRIPTION: " + answer[:400]
        question = str(getattr(observation, "utterance", "") or "").strip()
        if question:
            seen += "\nTHE USER THEN SAID: " + question[:300]
        user = ("USER REQUEST: %s\nDETERMINISTIC GUESS: %s\n\n"
                "Search query:" % (str(request_text or "")[:300],
                                   fallback or "(none)"))
        from backend.services.gemini_client import ask_gemini_chat
        response = ask_gemini_chat(
            [{"role": "system", "content": _QUERY_WRITER_SYSTEM},
             {"role": "user", "content": seen + "\n\n" + user}],
            temperature=0.1, max_tokens=60, timeout=(6, 12), no_retry=True)
        if not response or not response.get("choices"):
            return fallback
        content = response["choices"][0].get("message", {}).get("content", "")
        return _clean_composed_query(content) or fallback
    except Exception:
        return fallback


def analyze_screen(question: str) -> dict:
    """Capture the screen and ask Gemini Vision about it.

    Returns a dict with keys:
    - ``tip`` (str) — main answer
    - ``evidence`` (list[dict]) — supporting details, each tagged with a
      ``provenance`` (``observed`` / ``inferred`` / ``externally_checked``)
      so the answer never blurs what was seen into what was assumed (F48)
    - ``topic`` (str) — main subject for image/link generation
    - ``grounding_links`` (list[dict]) — source URLs, when the provider
      actually returns any (search grounding is OFF, so usually empty)
    - ``show_images`` (bool) — whether related images are worth fetching
    - ``region`` (dict|None) — captured region bounds when the question
      pointed at a specific area
    - ``observed_at`` (str|None) — when the screenshot was captured
    - ``observation_id`` (str) — the stored Observation this answer came
      from (RANK 1); corrections re-ask on its image instead of a new
      capture.

    No provider precondition here: the vision cascade attempts the selected
    provider and its eligible fallbacks, and reports unavailable only when
    none of them can serve the request.
    """
    region_question = is_region_question(question)
    mode = "region" if region_question else "full"
    attempts = {}
    obs, grade, meta = _identify(question, target_text="", mode=mode,
                                 attempts_out=attempts)

    if obs is None:
        if grade.get("error") == "capture":
            logging.warning("[SCREEN-QA] Capture failed: %s",
                            grade.get("detail"))
            return {
                "tip": "I couldn't capture your screen, sir. Please try again.",
                "evidence": [],
                "topic": "",
                "grounding_links": [],
                "show_images": False,
                "region": None,
                "observed_at": None,
                # F37: no provider was attempted — nothing was captured to send.
                "vision_provider": None,
                "vision_model": None,
                "vision_attempts": [],
            }
        # F37: total unavailability is reported with the ACTUAL attempted
        # providers instead of a bare "the model failed".
        reason = (vision_cascade.unavailable_reason(attempts)
                  or grade.get("detail")
                  or "no vision provider returned a result")
        logging.warning("[SCREEN-QA] %s", reason)
        attempted = vision_cascade.attempted_providers(attempts)
        return {
            "tip": ("I couldn't get a response from the vision model, sir. "
                    "(tried: %s)" % ", ".join(attempted)
                    if attempted
                    else "I couldn't get a response from the vision model, sir. "
                         "No vision provider is configured or eligible."),
            "evidence": [],
            "topic": "",
            "grounding_links": [],
            "show_images": False,
            "region": ((meta.get("cap") or {}).get("region") or None)
                      if region_question else None,
            "observed_at": utc_now_iso(),
            "vision_provider": None,
            "vision_model": None,
            "vision_attempts": list(attempts.get("attempts") or []),
            "vision_unavailable_reason": reason,
        }

    # F48: when the screen was actually looked at — the observation time every
    # evidence item inherits.
    parsed = meta.get("parsed") or {}
    content = str(meta.get("content") or "")
    observed_at = utc_now_iso()

    tip = obs.answer
    if not tip:
        # Fallback: use the raw text as the tip.
        tip = content.strip() if content.strip() else \
            "I couldn't interpret the screen, sir."

    evidence = parsed.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []

    # Sanitise evidence items — and keep the provenance the model declared.
    # F48: an item the model labels "observed" is drawn from the screenshot;
    # its own knowledge is "inferred". A MISSING or invented label is no
    # longer promoted to "observed" (that was the "missing labels become
    # observed" defect) — an unknown label degrades to "inferred", and a model
    # may never invent a stronger one such as "externally_checked".
    clean_evidence = []
    for item in evidence[:3]:
        if not isinstance(item, dict):
            continue
        declared = normalise_provenance(
            item.get("provenance"),
            default=PROVENANCE_INFERRED,
        )
        if declared not in (PROVENANCE_OBSERVED, PROVENANCE_INFERRED):
            declared = PROVENANCE_INFERRED
        snippet = str(item.get("snippet", "")).strip()
        source_label = str(item.get("source", "")).strip()
        entry = {
            "source": source_label,
            "title": str(item.get("title", "")).strip(),
            "snippet": snippet,
            "provenance": declared,
            "uncertainty": (
                "seen on screen" if declared == PROVENANCE_OBSERVED
                else "model inference — not stated on screen"
            ),
            "observed_at": observed_at,
        }
        if declared == PROVENANCE_OBSERVED:
            # F48: an observed item cites its actual SPAN — the text it was
            # read from — instead of a bare generic prefix.
            entry["spans"] = [{
                "quote": snippet,
                "selector": "screen",
                "publisher": source_label,
                "retrieved_at": observed_at,
                "independent": False,
                "domain": "",
            }]
        clean_evidence.append(entry)

    # F48: the user asked for a check, so a check runs — against the claim the
    # screen actually DISPLAYS (the answer plus what it was read from), not
    # against the deictic words of the question.
    if needs_external_check(question):
        displayed_claim = _displayed_claim(tip, clean_evidence)
        checked = _external_check(question, claim=displayed_claim)
        if checked:
            clean_evidence.append(checked)

    # RANK 1: prefer the exact label of the primary on-screen item; fall
    # back to the model's topic when the inventory named none.
    primary = next((item for item in obs.items if item.primary),
                   obs.items[0] if obs.items else None)
    topic = ""
    creator = ""
    if primary is not None:
        if primary.label_is_text:
            topic = primary.label
        creator = primary.creator
    topic = topic or (parsed.get("topic") or "").strip()
    creator = (creator or str(parsed.get("creator") or "").strip())[:120]
    show_images = bool(parsed.get("show_images", False))

    # Source URLs, when the provider grounds its answer at all. Search
    # grounding is OFF for this call, so this is normally empty — it is never
    # presented to the user as "verified sources".
    grounding_links = meta.get("grounding_links") or []

    return {
        "tip": tip,
        "evidence": clean_evidence,
        "topic": topic,
        "creator": creator,
        "grounding_links": grounding_links,
        "show_images": show_images,
        "region": (obs.region or None) if region_question else None,
        "observed_at": observed_at,
        # F37: the ACTUAL provider/model that served this answer travels with
        # it (and whether the output only survived as a degraded last resort).
        "vision_provider": obs.vision_provider,
        "vision_model": obs.vision_model,
        "vision_attempts": list(obs.vision_attempts or []),
        "vision_degraded": bool(obs.vision_degraded),
        # RANK 1: the stored observation later turns/corrections re-ask on.
        "observation_id": obs.id,
    }
