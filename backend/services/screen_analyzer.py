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

import json
import logging
import re

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

    No provider precondition here: the vision cascade attempts the selected
    provider and its eligible fallbacks, and reports unavailable only when
    none of them can serve the request.
    """
    region_question = is_region_question(question)
    try:
        if region_question:
            capture = capture_region_around_cursor()
            region_note = (
                "The image I sent you is a CROPPED region of the user's screen "
                "near their cursor/highlight — answer ONLY about what is inside it."
            )
        else:
            capture = capture_primary_screen()
            region_note = ""
    except RuntimeError as exc:
        logging.warning("[SCREEN-QA] Capture failed: %s", exc)
        observed_at = None
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

    # F48: when the screen was actually looked at — the observation time every
    # evidence item inherits.
    observed_at = utc_now_iso()

    prompt = _ANALYSIS_PROMPT.format(question=question)
    if region_note:
        prompt = region_note + "\n\n" + prompt

    # F37: the cascade fills this with the ACTUAL attempted providers/models.
    attempts = {}
    result = _ask_screen_vision_cascade(
        prompt,
        capture["image_data_url"],
        max_completion_tokens=800,
        attempts_out=attempts,
    )

    if not result or not result.get("choices"):
        # F37: total unavailability is reported with the ACTUAL attempted
        # providers instead of a bare "the model failed".
        reason = vision_cascade.unavailable_reason(attempts) or (
            "no vision provider returned a result")
        logging.warning("[SCREEN-QA] %s", reason)
        return {
            "tip": ("I couldn't get a response from the vision model, sir. "
                    "(tried: %s)" % ", ".join(
                        vision_cascade.attempted_providers(attempts))
                    if vision_cascade.attempted_providers(attempts)
                    else "I couldn't get a response from the vision model, sir. "
                         "No vision provider is configured or eligible."),
            "evidence": [],
            "topic": "",
            "grounding_links": [],
            "show_images": False,
            "region": (capture.get("region") or None) if region_question else None,
            "observed_at": observed_at,
            "vision_provider": None,
            "vision_model": None,
            "vision_attempts": list(attempts.get("attempts") or []),
            "vision_unavailable_reason": reason,
        }

    content = result["choices"][0].get("message", {}).get("content", "")
    parsed = _extract_json(content)

    tip = (parsed.get("tip") or "").strip()
    if not tip:
        # Fallback: use the raw text as the tip.
        tip = content.strip() if content.strip() else "I couldn't interpret the screen, sir."

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

    topic = (parsed.get("topic") or "").strip()
    creator = (parsed.get("creator") or "").strip()[:120]
    show_images = bool(parsed.get("show_images", False))

    # Source URLs, when the provider grounds its answer at all. Search
    # grounding is OFF for this call, so this is normally empty — it is never
    # presented to the user as "verified sources".
    grounding_links = result.get("grounding_links", [])

    return {
        "tip": tip,
        "evidence": clean_evidence,
        "topic": topic,
        "creator": creator,
        "grounding_links": grounding_links,
        "show_images": show_images,
        "region": (capture.get("region") or None) if region_question else None,
        "observed_at": observed_at,
        # F37: the ACTUAL provider/model that served this answer travels with
        # it (and whether the output only survived as a degraded last resort).
        "vision_provider": attempts.get("provider"),
        "vision_model": attempts.get("model"),
        "vision_attempts": list(attempts.get("attempts") or []),
        "vision_degraded": bool(attempts.get("degraded")),
    }
