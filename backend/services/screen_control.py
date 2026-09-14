import difflib
import json
import logging
import math
import os
import re
import threading
import time

from backend.services.grok_client import ask_groq_vision
from backend.services.screen_capture import (
    capture_active_window,
    capture_for_screen_control,
    capture_primary_screen,
    get_window_bounds,
    is_own_window,
    last_foreground_target,
)
from backend.services.screen_executor import execute_steps
from backend.services import screen_state
from backend.services import screen_ocr
from backend.services import screen_ui_elements
from backend.services import gemini_client
from backend.services import approvals
from backend.services import tool_policy
from backend.services import screen_geometry
from backend.services import vision_cascade


# Anchored, mutually exclusive toggle patterns: ON requires an actual
# on-word (or an enable-family verb), OFF requires an off-word (or a
# disable-family verb). 'turn screen controls off' can never match ON.
_TOGGLE_ON_RE = re.compile(
    r"\b(?:turn|switch|put)\s+on\s+(?:the\s+)?screen\s+control(?:s)?\b|"
    r"\b(?:turn|switch|put)\s+(?:the\s+)?screen\s+control(?:s)?\s+on\b|"
    r"\bscreen\s+control(?:s)?\s+on\b|"
    r"\b(?:enable|start|activate|engage)\s+(?:the\s+)?screen\s+control(?:s)?\b",
    re.IGNORECASE,
)

_TOGGLE_OFF_RE = re.compile(
    r"\b(?:turn|switch|put)\s+off\s+(?:the\s+)?screen\s+control(?:s)?\b|"
    r"\b(?:turn|switch|put)\s+(?:the\s+)?screen\s+control(?:s)?\s+off\b|"
    r"\bscreen\s+control(?:s)?\s+off\b|"
    r"\b(?:disable|stop|deactivate|disengage)\s+(?:the\s+)?screen\s+control(?:s)?\b",
    re.IGNORECASE,
)

_CONFIRM_RE = re.compile(
    r"\b(yes|confirm|do it|go ahead|proceed|execute|run it|ok(?:ay)?)\b",
    re.IGNORECASE,
)

_CANCEL_RE = re.compile(
    r"\b(cancel|never\s*mind|don'?t do it|abort|stop that)\b|"
    r"(?<!\w)no(?!\w)",
    re.IGNORECASE,
)

LEADING_FILLER_PATTERN = re.compile(
    r"^(?:\s*(?:command|hey jarvis|jarvis|jervis|jarvish|please|sir|hey|ok|okay|just|kindly|can you|could you|would you)\b[\s,.-]*)+",
    re.IGNORECASE,
)

RISKY_TERMS = (
    "delete",
    "remove",
    "submit",
    "send",
    "purchase",
    "buy",
    "pay",
    "checkout",
    "close",
    "quit",
    "exit",
    "transfer",
)

DESKTOP_CAPTURE_TERMS = (
    "desktop",
    "taskbar",
    "system tray",
    "notification area",
    "tray",
    "start menu",
    "start button",
    "windows search",
)

WINDOWS_SEARCH_COMMAND_TERMS = (
    "windows search",
    "taskbar search",
    "start menu search",
    "start search",
    "search on the taskbar",
    "search in the taskbar",
    "search from start",
)

LOW_CONFIDENCE_CONFIRMATION = 0.82
FAST_LOCAL_MATCH_CONFIDENCE = 0.90
MIN_ACTIONABLE_CONFIDENCE = 0.58
CLOUD_VERIFY_ENV = "JARVIS_CLOUD_VERIFY"

# F29 — the UIA-first fast path only fires on an ENABLED, UNAMBIGUOUS
# structural match: the winner must clear the strong-match bar and beat the
# runner-up by a strict margin (two identical labels = ambiguous = no fast
# path, exactly the F44 repeated-label rule).
UIA_FIRST_MIN_CONFIDENCE = 0.90
UIA_FIRST_UNAMBIGUOUS_MARGIN = 0.08

# F43 — the coordinate spaces _normalize_vision_plan will convert once, named
# using the shared vocabulary in screen_geometry.
_SUPPORTED_STEP_SPACES = (
    screen_geometry.COORD_NORMALIZED_1000,
    screen_geometry.COORD_VISION_PIXELS,
    screen_geometry.COORD_SCREEN_PIXELS,
    screen_geometry.COORD_WINDOW_PIXELS,
)

# Raw-coordinate / bounding-box clicks are inherently less certain than a
# resolved element_id, so a click whose target came from coordinates gets a
# confirmation gate whenever the planner's confidence is below this bar.
COORDINATE_CONFIRMATION_THRESHOLD = 0.80

# Icon-like targets matched via fuzzy UI/OCR text are visually ambiguous
# (a "Heart" caption vs the like button), so local match requires a stronger
# score before it is allowed to skip the vision model.
ICON_MATCH_THRESHOLD = 0.75

_ICON_NOUNS = {
    "heart",
    "star",
    "arrow",
    "bell",
    "camera",
    "like",
    "share",
    "notification",
    "gear",
    "magnifier",
}

UI_ALIASES = {
    "x button": ["close", "dismiss"],
    # F29: "exit" is NOT an alias of "close" — one closes a window, the other
    # ends the application. Declaring them equivalent let a materially
    # different effect be reached through the semantically-equivalent route.
    "close": ["x", "dismiss"],
    "search": ["find", "search box", "search bar", "lookup"],
    "settings": ["options", "preferences", "gear", "config"],
    "back": ["go back", "previous", "navigate back"],
    "forward": ["go forward", "next", "navigate forward"],
    "menu": ["hamburger", "three dots", "more", "..."],
    "refresh": ["reload"],
    "home": ["home page", "start page"],
    # F29: "save as" is a different action from "save" (it changes the target
    # path). It was listed as an alias, which is how a Button bonus could
    # promote "Save As" over an exact "Save".
    "save": ["save file"],
    "undo": ["undo action"],
    "redo": ["redo action"],
}

CLICKABLE_CONTROL_TYPES = {
    "Button",
    "Hyperlink",
    "MenuItem",
    "RadioButton",
    "CheckBox",
    "Tab",
    "TabItem",
}

PASSIVE_CONTROL_TYPES = {
    "Text",
    "Image",
    "Pane",
    "Group",
}

KEY_PHRASE_REPLACEMENTS = (
    ("page down", "pagedown"),
    ("page up", "pageup"),
    ("arrow up", "up"),
    ("arrow down", "down"),
    ("arrow left", "left"),
    ("arrow right", "right"),
    ("windows key", "windows"),
    ("win key", "windows"),
)

KEY_TOKEN_MAP = {
    "control": "ctrl",
    "ctrl": "ctrl",
    "shift": "shift",
    "alt": "alt",
    "windows": "windows",
    "win": "windows",
    "enter": "enter",
    "return": "enter",
    "tab": "tab",
    "escape": "esc",
    "esc": "esc",
    "space": "space",
    "spacebar": "space",
    "backspace": "backspace",
    "delete": "delete",
    "home": "home",
    "end": "end",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "pagedown": "page down",
    "pageup": "page up",
}

TEXT_ENTRY_CONTROL_TYPES = {
    "ComboBox",
    "Document",
    "Edit",
}


def _strip_leading_fillers(text):
    cleaned = text.strip()
    while True:
        updated = LEADING_FILLER_PATTERN.sub("", cleaned).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned.strip(" ,.")


def _normalize_text(text):
    cleaned = _strip_leading_fillers(text)
    return " ".join(cleaned.lower().split())


def _escape_xml_attr(value):
    text = str(value or "")
    return (
        text.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _is_toggle_on(text):
    return bool(_TOGGLE_ON_RE.search(_normalize_text(text)))


def _is_toggle_off(text):
    return bool(_TOGGLE_OFF_RE.search(_normalize_text(text)))


#: Negation words that invalidate an *enable* request ("don't turn on screen
#: controls", "never enable screen controls"). F19: a negated enable must
#: never enable.
_NEGATED_TOGGLE_RE = re.compile(
    r"\b(?:don'?t|do\s+not|does\s+not|never|no\s+longer|not|without|"
    r"mat\s+karo|nahi|na)\b",
    re.IGNORECASE,
)


def _exact_toggle_verdict(text):
    """Resolve one utterance to an exact screen-control toggle intent.

    Returns ``"on"``, ``"off"``, ``"negated_on"`` or ``None``.

    F19: the patterns are anchored and mutually exclusive, so an off-phrase
    can never match the on-pattern. Both matching at once (e.g. "turn them off
    and on again") is ambiguous and resolves to ``None`` rather than guessing.
    A negated enable resolves to ``"negated_on"`` — the caller leaves the
    current state alone instead of enabling.
    """
    normalized = _normalize_text(text)
    on = bool(_TOGGLE_ON_RE.search(normalized))
    off = bool(_TOGGLE_OFF_RE.search(normalized))
    if on == off:
        return None
    if on:
        if _NEGATED_TOGGLE_RE.search(normalized):
            return "negated_on"
        return "on"
    return "off"


def _is_confirmation(text):
    return bool(_CONFIRM_RE.search(_normalize_text(text)))


def _is_cancel(text):
    return bool(_CANCEL_RE.search(_normalize_text(text)))


_SCREEN_CMD_HIGH_CONFIDENCE = re.compile(
    r"^(?:can you |could you |would you |please )?"
    r"(click|double.?click|right.?click|scroll|type into|press the|"
    r"minimize|minimise|maximize|maximise|restore|hover over|"
    r"move\s+(?:the\s+)?mouse)\b",
    re.IGNORECASE,
)

_SCREEN_CMD_UI_NOUN = re.compile(
    r"\b(button|link|search\s*bar|search\s*box|text\s*box|textbox|"
    r"field|tab|icon|checkbox|dropdown|menu\s*item|toolbar)\b",
    re.IGNORECASE,
)

_SCREEN_CMD_ACTION_RE = re.compile(
    r"\b(click|type|open|close|select|focus|tap|press|hit|enter)\b",
    re.IGNORECASE,
)

# PART A - screen phrase + action verb detection anywhere (regardless of toggle)
_SCREEN_ACTION_VERBS_RE = re.compile(
    r"\b(click|double\s*click|double-?click|press|tap|open|close|play|pause|scroll|type|drag)\b",
    re.IGNORECASE,
)

_SCREEN_PHRASE_RE = re.compile(
    r"\b(?:visible\s+(?:on|at)\s+(?:my\s+|the\s+)?screen|"
    r"(?:on|at)\s+(?:my\s+|the\s+)?screen|"
    r"(?:on|at)\s+(?:my\s+|the\s+)?desktop|"
    r"anything\s+visible)\b",
    re.IGNORECASE,
)

# P1.6 extra whole-screen phrases that must route to primary/desktop capture
_WHOLE_SCREEN_EXTRA_RE = re.compile(
    r"\b(?:on\s+screen|on\s+the\s+screen|on\s+my\s+desktop|on\s+the\s+desktop|visible\s+on\s+screen|anything\s+visible)\b",
    re.IGNORECASE,
)

_INTERROGATIVE_PREFIX_RE = re.compile(r'^(what|which|how|where|who|why|when|whose)\b', re.IGNORECASE)

_MISSING_ELEMENT_PATTERNS = [
    "element id",
    "element_id",
    "not found in the ui tree",
    "missing from the tree",
    "no corresponding",
    "accessibility tree",
]

_RETRY_INSTRUCTION = (
    "IMPORTANT: the target may not be exposed in the accessibility tree. "
    "Look at the screenshot itself and return click coordinates (x, y in vision pixels) "
    "for the visible target instead of failing."
)


def _is_missing_element_failure(reason):
    lower = (reason or "").lower()
    return any(p in lower for p in _MISSING_ELEMENT_PATTERNS)


def _looks_like_screen_command(text):
    normalized = _normalize_text(text)
    if not normalized:
        return False
    # PART A: anywhere action verb + screen phrase -> screen command regardless of toggle
    if _SCREEN_ACTION_VERBS_RE.search(normalized) and _SCREEN_PHRASE_RE.search(normalized):
        if not _INTERROGATIVE_PREFIX_RE.search(normalized):
            return True
    if _SCREEN_CMD_HIGH_CONFIDENCE.search(normalized):
        return True
    # Context-dependent: needs screen controls on AND both a UI noun + action verb
    if screen_state.is_enabled():
        has_ui_noun = bool(_SCREEN_CMD_UI_NOUN.search(normalized))
        has_action = bool(_SCREEN_CMD_ACTION_RE.search(normalized))
        return has_ui_noun and has_action
    return False


def _is_whole_screen_phrase(text):
    """True when the command explicitly references the whole screen."""
    normalized = _normalize_text(text)
    return bool(_SCREEN_PHRASE_RE.search(normalized) or _WHOLE_SCREEN_EXTRA_RE.search(normalized))


def _is_risky_command(text):
    normalized = _normalize_text(text)
    return any(term in normalized for term in RISKY_TERMS)


def _needs_desktop_capture(command_text):
    normalized = _normalize_text(command_text)
    if any(term in normalized for term in DESKTOP_CAPTURE_TERMS):
        return True
    # P1.6 also treat explicit whole-screen phrases as needing desktop-wide (primary) capture
    return bool(_WHOLE_SCREEN_EXTRA_RE.search(normalized))


def _parse_scroll_command(command_text):
    normalized = _normalize_text(command_text)
    if "scroll" not in normalized:
        return None

    direction = None
    for candidate in ("down", "up", "right", "left"):
        if re.search(rf"\b{candidate}\b", normalized):
            direction = candidate
            break

    if direction is None:
        direction = "down"

    amount = 600
    if any(word in normalized for word in ("little", "slightly", "bit", "small")):
        amount = 360
    elif any(word in normalized for word in ("page", "lot", "more", "fast", "far")):
        amount = 960

    return {
        "ok": True,
        "confidence": 0.99,
        "summary": f"Scrolling {direction}, sir.",
        "needs_confirmation": False,
        "reason": "",
        "steps": [{"action": "scroll", "direction": direction, "amount": amount}],
    }


def _parse_windows_shell_command(command_text):
    normalized = _normalize_text(command_text)

    if (
        ("start menu" in normalized or "start button" in normalized)
        and any(word in normalized for word in ("click", "open", "show", "press"))
    ):
        return {
            "ok": True,
            "confidence": 0.99,
            "summary": "Opening Start, sir.",
            "needs_confirmation": False,
            "reason": "",
            "steps": [{"action": "press", "keys": ["windows"]}],
        }

    if any(term in normalized for term in WINDOWS_SEARCH_COMMAND_TERMS) and any(
        word in normalized for word in ("click", "open", "show", "focus", "select")
    ):
        return {
            "ok": True,
            "confidence": 0.99,
            "summary": "Opening Windows search, sir.",
            "needs_confirmation": False,
            "reason": "",
            "steps": [{"action": "press", "keys": ["windows", "s"]}],
        }

    if (
        ("search bar" in normalized or "search box" in normalized)
        and any(term in normalized for term in ("taskbar", "start menu", "start button", "windows search"))
        and any(word in normalized for word in ("click", "open", "show", "focus", "select"))
    ):
        return {
            "ok": True,
            "confidence": 0.99,
            "summary": "Opening Windows search, sir.",
            "needs_confirmation": False,
            "reason": "",
            "steps": [{"action": "press", "keys": ["windows", "s"]}],
        }

    return None


def _parse_window_command(command_text):
    normalized = _normalize_text(command_text)

    if any(word in normalized for word in ("minimize", "minimise")):
        return {
            "ok": True,
            "confidence": 0.99,
            "summary": "Minimizing the active window, sir.",
            "needs_confirmation": False,
            "reason": "",
            "steps": [{"action": "minimize_window"}],
        }

    if any(word in normalized for word in ("maximize", "maximise")):
        return {
            "ok": True,
            "confidence": 0.99,
            "summary": "Maximizing the active window, sir.",
            "needs_confirmation": False,
            "reason": "",
            "steps": [{"action": "maximize_window"}],
        }

    if "restore" in normalized:
        return {
            "ok": True,
            "confidence": 0.99,
            "summary": "Restoring the active window, sir.",
            "needs_confirmation": False,
            "reason": "",
            "steps": [{"action": "restore_window"}],
        }

    return None


def _parse_key_sequence(raw_keys):
    normalized = " ".join(raw_keys.lower().split())
    for source, target in KEY_PHRASE_REPLACEMENTS:
        normalized = normalized.replace(source, target)

    normalized = normalized.replace(" plus ", "+")
    normalized = normalized.replace(" and ", "+")
    normalized = normalized.replace(",", "+")
    normalized = normalized.replace("-", "+")
    normalized = normalized.replace(" then ", " ")

    pieces = [piece for piece in re.split(r"[+\s]+", normalized) if piece]
    keys = []
    for piece in pieces:
        if piece in KEY_TOKEN_MAP:
            keys.append(KEY_TOKEN_MAP[piece])
        elif re.fullmatch(r"f\d{1,2}", piece):
            keys.append(piece)
        elif re.fullmatch(r"[a-z0-9]", piece):
            keys.append(piece)

    return keys


def _parse_press_command(command_text):
    cleaned = _strip_leading_fillers(command_text)
    match = re.match(r"^(?:press|hit|tap)\s+(.+)$", cleaned, flags=re.IGNORECASE)
    if not match:
        return None

    keys = _parse_key_sequence(match.group(1))
    if not keys:
        return None

    key_words = ", ".join(keys)
    return {
        "ok": True,
        "confidence": 0.98,
        "summary": f"Pressing {key_words}, sir.",
        "needs_confirmation": _is_risky_command(command_text),
        "reason": "",
        "steps": [{"action": "press", "keys": keys}],
    }


def _unquote_spoken_text(text):
    cleaned = (text or "").strip()
    if cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')) and len(cleaned) >= 2:
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def _clean_action_target_text(text):
    cleaned = (text or "").strip().lower()
    for prefix in ("the ", "a ", "an "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    for suffix in (
        " button",
        " tab",
        " icon",
        " menu",
        " bar",
        " link",
        " field",
        " box",
        " in front of me",
        " in front of you",
        " for me",
        " please",
    ):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned.strip()


def _extract_targeted_text_entry(command_text):
    cleaned = _strip_leading_fillers(command_text)
    match = re.match(
        r"^(?:type|write|enter)\s+(.+?)\s+(?:in|into)\s+(.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    text = _unquote_spoken_text(match.group(1))
    target = _clean_action_target_text(match.group(2))
    if not text or not target:
        return None

    return {"text": text, "target": target}


def _extract_text_to_type(command_text):
    cleaned = _strip_leading_fillers(command_text)

    if _extract_targeted_text_entry(cleaned):
        return None

    direct_match = re.match(
        r"^(?:type|enter)\s+(.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if not direct_match:
        return None

    return _unquote_spoken_text(direct_match.group(1))


def _parse_type_command(command_text):
    text = _extract_text_to_type(command_text)
    if text is None:
        return None
    if not text:
        return None

    return {
        "ok": True,
        "confidence": 0.98,
        "summary": "Typing that now, sir.",
        "needs_confirmation": False,
        "reason": "",
        "steps": [{"action": "type", "text": text}],
    }


def _build_direct_plan(command_text):
    for parser in (
        _parse_windows_shell_command,
        _parse_window_command,
        _parse_scroll_command,
        _parse_press_command,
        _parse_type_command,
    ):
        plan = parser(command_text)
        if plan:
            plan["source"] = "Direct Match"
            return plan
    return None


def _extract_json_object(content):
    if not content:
        return {}

    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _coerce_int(value, default=0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    # F45: NaN/infinity are not usable integers. ``int(round(inf))`` raises
    # OverflowError, which used to escape and abort plan normalization.
    if not math.isfinite(number):
        return default
    try:
        return int(round(number))
    except (OverflowError, ValueError):
        return default


def _coerce_float(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    # F45: a NaN confidence made every threshold comparison False
    # (``nan < MIN_ACTIONABLE_CONFIDENCE`` is False), so a plan with no usable
    # confidence sailed past the low-confidence and confirmation gates.
    if not math.isfinite(number):
        return default
    return number


def _bounded_confidence(value, default=0.0):
    """F45: confidence is a finite number inside [0, 1]."""
    return max(0.0, min(1.0, _coerce_float(value, default)))


def _declared_space(source):
    """The coordinate space explicitly declared by a plan or step.

    Returns ``(space, declared)``:

      * ``(token, True)``  — declared and recognised;
      * ``(None, True)``   — declared but unrecognised (caller must reject);
      * ``(None, False)``  — nothing was declared (caller must not guess).
    """
    if not isinstance(source, dict):
        return None, False
    stated = source.get("coordinate_space")
    if stated is None or str(stated).strip() == "":
        stated = source.get("space")
    if stated is None or str(stated).strip() == "":
        return None, False
    return screen_geometry.normalize_space(stated, default=None), True


def _spaces_conflict(raw_step):
    """F45: ``space`` and ``coordinate_space`` on one step must agree."""
    if not isinstance(raw_step, dict):
        return False
    first = raw_step.get("coordinate_space")
    second = raw_step.get("space")
    if first is None or second is None:
        return False
    a = screen_geometry.normalize_space(first, default=None)
    b = screen_geometry.normalize_space(second, default=None)
    return a != b


def _point_in_declared_space(space, x, y, capture):
    """F45: a coordinate must lie inside the frame it claims to be in.

    Out-of-range points used to be *clamped* into the capture, which turned a
    malformed or hallucinated coordinate into a real click at the edge of the
    screen. They are now rejected.
    """
    try:
        fx, fy = float(x), float(y)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(fx) and math.isfinite(fy)):
        return False

    if space == screen_geometry.COORD_NORMALIZED_1000:
        return 0 <= fx <= 1000 and 0 <= fy <= 1000

    if space == screen_geometry.COORD_VISION_PIXELS:
        vw = _coerce_int(capture.get("vision_width"), 0)
        vh = _coerce_int(capture.get("vision_height"), 0)
        if vw <= 0 or vh <= 0:
            return False
        return 0 <= fx <= vw - 1 and 0 <= fy <= vh - 1

    if space == screen_geometry.COORD_WINDOW_PIXELS:
        cw = _coerce_int(capture.get("capture_width"), 0)
        ch = _coerce_int(capture.get("capture_height"), 0)
        if cw <= 0 or ch <= 0:
            return False
        return 0 <= fx <= cw - 1 and 0 <= fy <= ch - 1

    if space == screen_geometry.COORD_SCREEN_PIXELS:
        ol = capture.get("origin_left")
        ot = capture.get("origin_top")
        cw = _coerce_int(capture.get("capture_width"), 0)
        ch = _coerce_int(capture.get("capture_height"), 0)
        if ol is None or ot is None or cw <= 0 or ch <= 0:
            return False
        return (int(ol) <= fx <= int(ol) + cw - 1
                and int(ot) <= fy <= int(ot) + ch - 1)

    return False


def _command_references_window_title(command_text, title):
    normalized_command = _normalize_text(command_text)
    normalized_title = _normalize_text(title)
    if not normalized_command or not normalized_title:
        return False

    if normalized_title in normalized_command:
        return True

    title_tokens = [token for token in re.findall(r"[a-z0-9]+", normalized_title) if len(token) >= 4]
    if not title_tokens:
        return False

    shared = [
        token
        for token in title_tokens
        if re.search(rf"\b{re.escape(token)}\b", normalized_command)
    ]
    required_matches = 1 if len(title_tokens) == 1 else min(2, len(title_tokens))
    return len(shared) >= required_matches


def _is_window_local_command(command_text, capture=None):
    normalized = _normalize_text(command_text)
    if not normalized or _needs_desktop_capture(command_text):
        return False

    if any(term in normalized for term in ("window", "app", "application", "dialog", "popup", "title bar")):
        return True

    if re.search(r"\b(its|this|that)\b", normalized) and any(
        term in normalized
        for term in ("search bar", "search box", "button", "field", "textbox", "text box", "tab", "icon")
    ):
        return True

    if capture and _command_references_window_title(command_text, capture.get("window_title", "")):
        return True

    return False


def _extract_box_center(raw_step):
    for key in ("box", "bbox", "bounds"):
        box = raw_step.get(key)
        if not isinstance(box, dict):
            continue

        left = _coerce_int(box.get("left", box.get("x", box.get("x1"))), None)
        top = _coerce_int(box.get("top", box.get("y", box.get("y1"))), None)
        right = _coerce_int(box.get("right", box.get("x2")), None)
        bottom = _coerce_int(box.get("bottom", box.get("y2")), None)

        if right is None and left is not None:
            width = _coerce_int(box.get("width"), None)
            if width is not None:
                right = left + width

        if bottom is None and top is not None:
            height = _coerce_int(box.get("height"), None)
            if height is not None:
                bottom = top + height

        if None in (left, top, right, bottom):
            continue
        if right <= left or bottom <= top:
            continue

        return (
            int(round((left + right) / 2.0)),
            int(round((top + bottom) / 2.0)),
        )

    # Also support bbox returned directly as top-level keys (e.g. {"x":10,"y":20,"width":100,"height":50}
    # or {"left":10,"top":20,"right":110,"bottom":70}) without wrapper - plain x/y remains fallback
    if isinstance(raw_step, dict) and any(k in raw_step for k in ("width", "height", "right", "bottom", "x2", "y2")):
        left = _coerce_int(raw_step.get("left", raw_step.get("x", raw_step.get("x1"))), None)
        top = _coerce_int(raw_step.get("top", raw_step.get("y", raw_step.get("y1"))), None)
        right = _coerce_int(raw_step.get("right", raw_step.get("x2")), None)
        bottom = _coerce_int(raw_step.get("bottom", raw_step.get("y2")), None)
        if right is None and left is not None:
            width = _coerce_int(raw_step.get("width"), None)
            if width is not None:
                right = left + width
        if bottom is None and top is not None:
            height = _coerce_int(raw_step.get("height"), None)
            if height is not None:
                bottom = top + height
        if None not in (left, top, right, bottom) and right > left and bottom > top:
            return (
                int(round((left + right) / 2.0)),
                int(round((top + bottom) / 2.0)),
            )

    return None


def _vision_to_screen_point(capture, x, y):
    x_scale = capture["capture_width"] / float(capture["vision_width"])
    y_scale = capture["capture_height"] / float(capture["vision_height"])
    screen_x = capture["origin_left"] + int(round(x * x_scale))
    screen_y = capture["origin_top"] + int(round(y * y_scale))
    # P1-B: post-scale clamp to capture bounds (vision clamp then scale can overshoot by 1px)
    max_x = capture["origin_left"] + capture["capture_width"] - 1
    max_y = capture["origin_top"] + capture["capture_height"] - 1
    screen_x = max(capture["origin_left"], min(screen_x, max_x))
    screen_y = max(capture["origin_top"], min(screen_y, max_y))
    return screen_x, screen_y


def _build_element_point(x, y, space="vision", bounds=None, source="", label="", uid="", runtime_id=""):
    point = {
        "x": _coerce_int(x, None),
        "y": _coerce_int(y, None),
        # F43: canonical space token — never a loose spelling that a later
        # comparison (`space == "screen"`) could miss.
        "space": screen_geometry.normalize_space(space, screen_geometry.COORD_VISION_PIXELS),
    }
    if bounds:
        point["bounds"] = dict(bounds)
    if source:
        point["source"] = source
    if label:
        point["label"] = label
    if uid:
        point["uid"] = uid
    if runtime_id:
        # F42: the semantic runtime identity travels with the element point so
        # the executor can re-resolve the control through a FRESH UIA walk.
        point["runtime_id"] = runtime_id
    return point


def _build_screen_snapshot(capture):
    snapshot = {
        "raw_ui": [],
        "ui_elements": [],
        "ui_space": screen_geometry.COORD_SCREEN_PIXELS,
        "ocr_regions": [],
        "ocr_merged": [],
        # F43: the observation this snapshot came from, and the window it
        # describes. Consumers can compare this epoch with the capture's to
        # refuse mixing a tree with a different frame.
        "capture_epoch": capture.get("capture_epoch"),
        "ui_hwnd": capture.get("ui_hwnd") or capture.get("hwnd"),
    }

    if screen_ui_elements.is_available():
        try:
            # F43: walk the window the CAPTURE came from — not "whatever is
            # foreground now". When Jarvis's overlay owns the foreground the
            # screenshot is of the recorded user target, and a foreground walk
            # would feed the planner a tree describing a completely different
            # window than the pixels it is looking at.
            raw_ui = screen_ui_elements.get_foreground_window_elements(
                hwnd=capture.get("ui_hwnd") or capture.get("hwnd")
            )
            snapshot["raw_ui"] = raw_ui
            if all(
                key in capture
                for key in ("capture_width", "capture_height", "vision_width", "vision_height")
            ):
                snapshot["ui_elements"] = screen_ui_elements.to_vision_coords(raw_ui, capture)
                snapshot["ui_space"] = screen_geometry.COORD_VISION_PIXELS
            else:
                snapshot["ui_elements"] = raw_ui
                snapshot["ui_space"] = screen_geometry.COORD_SCREEN_PIXELS
        except Exception as exc:
            logging.warning("UI Automation snapshot failed: %s", exc)

    raw_image = capture.get("raw_vision_image")
    if raw_image and screen_ocr.is_available():
        try:
            regions = screen_ocr.extract_text_regions(raw_image)
            snapshot["ocr_regions"] = regions
            snapshot["ocr_merged"] = screen_ocr._merge_nearby_words(regions)
        except Exception as exc:
            logging.warning("OCR snapshot failed: %s", exc)

    return snapshot


def _resolve_element_point(element_ref):
    if isinstance(element_ref, dict):
        x = _coerce_int(element_ref.get("x"), None)
        y = _coerce_int(element_ref.get("y"), None)
        if x is None or y is None:
            return None
        return {
            "x": x,
            "y": y,
            # F43: canonical token, so a later comparison against the screen
            # space cannot miss an alias spelling.
            "space": screen_geometry.normalize_space(
                element_ref.get("space"), screen_geometry.COORD_VISION_PIXELS),
            "uid": element_ref.get("uid", ""),
            "runtime_id": element_ref.get("runtime_id", ""),
        }

    if isinstance(element_ref, (list, tuple)) and len(element_ref) >= 2:
        x = _coerce_int(element_ref[0], None)
        y = _coerce_int(element_ref[1], None)
        if x is None or y is None:
            return None
        return {"x": x, "y": y, "space": screen_geometry.COORD_VISION_PIXELS,
                "uid": "", "runtime_id": ""}

    return None


def _resolve_observation_depth(elements, by_uid):
    """F44: effective depth from explicit parent links, not a raw integer.

    A depth value recorded during the walk can skip a level when the walk was
    truncated (``max_elements``) or when a degenerate container was dropped.
    Trusting it then misparents children. Deriving depth from the emitted
    ``parent_uid`` chain keeps parent/child relationships exact; the recorded
    depth is the fallback for element dicts that carry no parent information.
    """
    depths = {}
    for el in elements:
        uid = el.get("uid")
        if not uid or el.get("parent_uid") is None:
            continue
        hops = 0
        parent = el.get("parent_uid")
        seen = set()
        while parent and parent in by_uid and hops < 64:
            if parent in seen:
                break  # defensive: never loop on a corrupt chain
            seen.add(parent)
            hops += 1
            parent = by_uid[parent].get("parent_uid")
        depths[uid] = hops
    return depths


def _serialize_accessibility_elements(elements, start_id=1, space="vision"):
    lines = []
    element_id_map = {}
    next_id = start_id
    open_depths = []
    total = len(elements)

    # F43: the caller declares which frame these bounds live in. Defaulting to
    # "vision" is only correct for the converted (to_vision_coords) path; the
    # unconverted fallback is screen pixels and must say so. An unrecognised
    # label is treated as screen pixels — the conservative choice, because
    # claiming "vision" for raw pixels would convert them a second time.
    declared_space = screen_geometry.normalize_space(
        space, screen_geometry.COORD_SCREEN_PIXELS)
    if declared_space not in (screen_geometry.COORD_SCREEN_PIXELS,
                              screen_geometry.COORD_VISION_PIXELS):
        declared_space = screen_geometry.COORD_SCREEN_PIXELS

    by_uid = {}
    for el in elements:
        uid = el.get("uid")
        if uid:
            by_uid.setdefault(uid, el)
    chain_depths = _resolve_observation_depth(elements, by_uid)

    def _depth_of(index, el):
        uid = el.get("uid")
        if uid and uid in chain_depths:
            return chain_depths[uid]
        return max(0, _coerce_int(el.get("depth"), 0))

    for index, el in enumerate(elements):
        depth = _depth_of(index, el)
        while open_depths and open_depths[-1] >= depth:
            close_depth = open_depths.pop()
            lines.append(f'{"  " * (close_depth + 2)}</element>')

        left = _coerce_int(el.get("left"), 0)
        top = _coerce_int(el.get("top"), 0)
        right = _coerce_int(el.get("right"), 0)
        bottom = _coerce_int(el.get("bottom"), 0)
        cx = int(round((left + right) / 2.0))
        cy = int(round((top + bottom) / 2.0))
        control_type = (el.get("control_type") or "element").strip() or "element"
        name = (el.get("name") or "").strip()
        automation_id = (el.get("automation_id") or "").strip()
        class_name = (el.get("class_name") or "").strip()
        enabled = bool(el.get("enabled", True))

        attrs = [
            f'id="{next_id}"',
            f'role="{_escape_xml_attr(control_type)}"',
            f'enabled="{str(enabled).lower()}"',
            f'bounds="{left},{top},{right},{bottom}"',
            f'center="{cx},{cy}"',
        ]
        if name:
            attrs.append(f'name="{_escape_xml_attr(name)}"')
        if automation_id:
            attrs.append(f'automation_id="{_escape_xml_attr(automation_id)}"')
        if class_name:
            attrs.append(f'class="{_escape_xml_attr(class_name)}"')
        if el.get("placeholder"):
            # F44: explicit structural container kept only for hierarchy.
            attrs.append('structural="true"')

        indent = "  " * (depth + 2)
        next_depth = (
            _depth_of(index + 1, elements[index + 1])
            if index + 1 < total
            else -1
        )

        element_id_map[next_id] = _build_element_point(
            cx,
            cy,
            space=declared_space,
            bounds={"left": left, "top": top, "right": right, "bottom": bottom},
            source="ui",
            label=name or control_type,
            uid=el.get("uid", ""),
            runtime_id=el.get("runtime_id", ""),
        )

        if next_depth > depth:
            lines.append(f'{indent}<element {" ".join(attrs)}>')
            open_depths.append(depth)
        else:
            lines.append(f'{indent}<element {" ".join(attrs)}/>')
        next_id += 1

    while open_depths:
        close_depth = open_depths.pop()
        lines.append(f'{"  " * (close_depth + 2)}</element>')

    return lines, element_id_map, next_id


def _serialize_ocr_nodes(regions, seen_texts, start_id=1):
    lines = []
    element_id_map = {}
    next_id = start_id
    # F44: identical labels at DISTINCT locations are separate spatial entities
    # and each must survive as its own node (two "Edit"/"Delete"/"Search"
    # controls in a dense app are distinguishable by position). Only boxes that
    # overlap the same physical region are duplicate detections of one entity.
    emitted = []

    for region in regions:
        text = (region.get("text") or "").strip()
        normalized_text = re.sub(r"\s+", " ", text.lower())
        if not text or len(text) < 2:
            continue

        left = _coerce_int(region.get("left"), 0)
        top = _coerce_int(region.get("top"), 0)
        right = _coerce_int(region.get("right"), 0)
        bottom = _coerce_int(region.get("bottom"), 0)
        if right <= left or bottom <= top:
            continue

        bounds = {"left": left, "top": top, "right": right, "bottom": bottom}
        duplicate = False
        for prior in emitted:
            if prior["text"] != normalized_text:
                continue
            if screen_geometry.rects_duplicate(bounds, prior["bounds"]):
                duplicate = True
                break
        if duplicate:
            continue

        address = (region.get("block_num"), region.get("par_num"), region.get("line_num"))
        cx = int(round((left + right) / 2.0))
        cy = int(round((top + bottom) / 2.0))
        confidence = _coerce_int(region.get("confidence"), 0)

        lines.append(
            '    <text id="{id}" value="{value}" bounds="{bounds}" center="{center}" confidence="{confidence}"/>'.format(
                id=next_id,
                value=_escape_xml_attr(text),
                bounds=f"{left},{top},{right},{bottom}",
                center=f"{cx},{cy}",
                confidence=confidence,
            )
        )
        element_id_map[next_id] = _build_element_point(
            cx,
            cy,
            space="vision",
            bounds=bounds,
            source="ocr",
            label=text,
        )
        emitted.append({"text": normalized_text, "bounds": bounds, "address": address})
        next_id += 1

    return lines, element_id_map, next_id


def _plan_selection_score(plan, command_text):
    score = _coerce_float(plan.get("confidence"), 0.0)
    if plan.get("capture_mode") != "active_window":
        return score

    if _is_window_local_command(command_text, {"window_title": plan.get("window_title", "")}):
        score += 0.08

    if _command_references_window_title(command_text, plan.get("window_title", "")):
        score += 0.06

    return score


def _normalize_action_name(raw_action):
    text = (raw_action or "").strip().lower().replace("-", " ").replace("_", " ")
    if not text:
        return ""

    if re.search(r"\bright\s+click\b", text):
        return "right_click"
    if re.search(r"\bdouble\s+click\b", text):
        return "double_click"

    match = re.search(
        r"\b(click|type|enter|write|press|hotkey|scroll|hover|move|select|focus)\b",
        text,
    )
    if not match:
        return text.replace(" ", "_")

    verb = match.group(1)
    if verb in {"select", "focus"}:
        return "click"
    if verb in {"enter", "write"}:
        return "type"
    return verb


def _step_point_to_screen(space, x, y, capture):
    """Convert ONE raw planner point (declared *space*) to screen pixels.

    F43 — every provider point is converted exactly once, by its named
    coordinate space, instead of assuming the 0..1000 frame. Returns
    (screen_x, screen_y), or None when that space cannot be expressed against
    this capture (e.g. window-relative coordinates for a desktop capture).
    """
    if x is None or y is None:
        return None
    x = int(x)
    y = int(y)

    if space == screen_geometry.COORD_SCREEN_PIXELS:
        return x, y

    if space == screen_geometry.COORD_WINDOW_PIXELS:
        if capture.get("capture_mode") != "active_window":
            return None
        if any(k not in capture for k in ("origin_left", "origin_top")):
            return None
        return int(capture["origin_left"]) + x, int(capture["origin_top"]) + y

    if space == screen_geometry.COORD_VISION_PIXELS:
        return _vision_to_screen_point(capture, x, y)

    # normalized 0..1000 — clamp to the vision frame, then scale once (P1-B).
    vw = max(1, int(capture.get("vision_width") or 1000))
    vh = max(1, int(capture.get("vision_height") or 1000))
    vx = min(max(int(round(x * vw / 1000.0)), 0), vw - 1)
    vy = min(max(int(round(y * vh / 1000.0)), 0), vh - 1)
    return _vision_to_screen_point(capture, vx, vy)


def _mark_step_sensitive(step_payload, raw_step, element_id_map=None):
    """F21: flag a typed step whose target is a sensitive field.

    The flag is what the redacted egress boundary keys on — without it, a step
    that types a password is indistinguishable from one that types a city name
    because both store the value under the key ``text``.

    Evidence only: an explicit marker from the planner, a sensitive word in the
    step's own description/label, or the identity of the element the step
    targets. A plain typed step is left unmarked so logs stay readable.
    """
    if raw_step.get("sensitive") is True or raw_step.get("secret") is True:
        step_payload["sensitive"] = True
        return True
    if tool_policy.is_sensitive_field(
            raw_step.get("description"), raw_step.get("label"),
            raw_step.get("target"), raw_step.get("name"),
            raw_step.get("placeholder")):
        step_payload["sensitive"] = True
        return True

    lookup = element_id_map if isinstance(element_id_map, dict) else {}
    keys = []
    el_id = _coerce_int(raw_step.get("element_id"), None)
    if el_id is not None:
        keys.append(el_id)
    for key in keys:
        element = lookup.get(key)
        if not isinstance(element, dict):
            continue
        if element.get("is_password") or tool_policy.is_sensitive_field(
                element.get("name"), element.get("label"),
                element.get("text"), element.get("placeholder"),
                element.get("control_type"), element.get("automation_id"),
                element.get("className")):
            step_payload["sensitive"] = True
            return True
    return False


def _normalize_vision_plan(plan, capture, element_id_map=None):
    if not isinstance(plan, dict):
        return {
            "ok": False,
            "confidence": 0.0,
            "summary": "",
            "needs_confirmation": False,
            "reason": "I couldn't parse the screen action plan.",
            "steps": [],
        }

    normalized = {
        "ok": bool(plan.get("ok")),
        "confidence": _bounded_confidence(plan.get("confidence"), 0.0),
        "summary": (plan.get("summary") or "").strip(),
        "needs_confirmation": bool(plan.get("needs_confirmation")),
        "reason": (plan.get("reason") or "").strip(),
        "capture_mode": capture.get("capture_mode", ""),
        "window_title": capture.get("window_title", ""),
        "hwnd": capture.get("hwnd"),
        "origin_left": capture.get("origin_left"),
        "origin_top": capture.get("origin_top"),
        "capture_width": capture.get("capture_width"),
        "capture_height": capture.get("capture_height"),
        "vision_width": capture.get("vision_width"),
        "vision_height": capture.get("vision_height"),
        # F43/F42: identity + coordinate contract travel with the plan so the
        # executor can revalidate the target by HWND, process id and monitor.
        "process_id": capture.get("process_id"),
        "monitor_index": capture.get("monitor_index"),
        "dpi_scale": capture.get("dpi_scale"),
        "steps": [],
    }

    if not normalized["ok"]:
        if not normalized["reason"]:
            normalized["reason"] = "I couldn't find a confident screen action for that."
        return normalized

    def _reject(reason):
        normalized["ok"] = False
        normalized["reason"] = reason
        normalized["steps"] = []
        return normalized

    # F45: validate the schema before trusting any of it.
    raw_steps = plan.get("steps", [])
    if not isinstance(raw_steps, list):
        return _reject("The screen planner returned a malformed plan.")

    raw_coord_click = False
    plan_space, plan_space_declared = _declared_space(plan)
    if plan_space_declared and plan_space is None:
        return _reject(
            "The screen planner returned coordinates in an unrecognised "
            "coordinate space."
        )
    if plan_space is not None and plan_space not in _SUPPORTED_STEP_SPACES:
        return _reject(
            "The screen planner returned coordinates in an unsupported space."
        )

    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            return _reject("The screen planner returned a malformed step.")
        raw_action = (raw_step.get("action") or "").strip().lower()
        action = _normalize_action_name(raw_action)
        if not action:
            return _reject("The screen planner returned a step with no action.")

        if action in {"click", "double_click", "right_click", "move", "hover"}:
            x, y = None, None
            point_space = screen_geometry.COORD_VISION_PIXELS

            # 1. Primary: Set-of-Mark element_id
            el_id = _coerce_int(raw_step.get("element_id"), None)
            if el_id is not None:
                if not element_id_map:
                    # F45: an element id with no tree at all is invalid — the
                    # image-only path must never silently degrade to a guessed
                    # coordinate click on a hallucinated control.
                    return _reject(
                        "The screen planner referenced a control that is not in "
                        "the current UI tree."
                    )
                if el_id not in element_id_map:
                    logging.warning(
                        "Screen plan referenced missing element_id=%s for action=%s",
                        el_id, raw_action,
                    )
                    return _reject(
                        "The local screen planner referenced a control that is "
                        "not in the current UI tree."
                    )
                point = _resolve_element_point(element_id_map[el_id])
                if point is not None:
                    x = point["x"]
                    y = point["y"]
                    point_space = point["space"]
            else:
                # 2. Fallback: bounding box or direct x, y in the DECLARED space.
                # F45: the space must be declared explicitly — a missing or
                # unknown space is rejected instead of being guessed.
                raw_coord_click = True
                if _spaces_conflict(raw_step):
                    return _reject(
                        "The screen planner declared conflicting coordinate "
                        "spaces for one step."
                    )
                step_space, step_declared = _declared_space(raw_step)
                if not step_declared:
                    step_space = plan_space
                if step_space is None:
                    return _reject(
                        "The screen planner did not declare which coordinate "
                        "space its coordinates use."
                    )
                if step_space not in _SUPPORTED_STEP_SPACES:
                    return _reject(
                        "The screen planner returned coordinates in an "
                        "unsupported space."
                    )

                box_center = _extract_box_center(raw_step)
                if box_center is not None:
                    x, y = box_center
                else:
                    x = _coerce_int(raw_step.get("x"), None)
                    y = _coerce_int(raw_step.get("y"), None)
                if x is not None and y is not None:
                    # F45: out-of-range points are rejected, not clamped into
                    # an actionable pixel at the edge of the capture.
                    if not _point_in_declared_space(step_space, x, y, capture):
                        return _reject(
                            "The screen planner returned a coordinate outside "
                            "the captured screen area."
                        )
                    converted = _step_point_to_screen(step_space, x, y, capture)
                    if converted is None:
                        return _reject(
                            "The screen planner returned window-relative "
                            "coordinates but this capture is not a single window."
                        )
                    x, y = converted
                    point_space = screen_geometry.COORD_SCREEN_PIXELS

            if x is None or y is None:
                continue

            # P0: Do NOT snap pure-coordinate clicks - model coordinates are already
            # screen-grounded. Previous snapping used smallest containing element
            # which for browsers is the giant document root (0,0,1920,1216) and
            # snapped every icon click to its center (960,608).
            raw_ui_uid = raw_step.get("ui_uid", "")

            if point_space == screen_geometry.COORD_SCREEN_PIXELS:
                screen_x, screen_y = x, y
            else:
                x = min(max(x, 0), capture["vision_width"] - 1)
                y = min(max(y, 0), capture["vision_height"] - 1)
                screen_x, screen_y = _vision_to_screen_point(capture, x, y)

            # P1-B: post-scale clamp (also covers screen-space case and 1px overshoot on far edge)
            if all(k in capture for k in ("origin_left", "origin_top", "capture_width", "capture_height")):
                max_x = capture["origin_left"] + capture["capture_width"] - 1
                max_y = capture["origin_top"] + capture["capture_height"] - 1
                screen_x = max(capture["origin_left"], min(screen_x, max_x))
                screen_y = max(capture["origin_top"], min(screen_y, max_y))

            # F43: emitted steps carry an explicit coordinate-space contract.
            step_payload = {
                "action": action,
                "x": screen_x,
                "y": screen_y,
                "button": raw_step.get("button", "left"),
                "clicks": raw_step.get("clicks", 1),
                "space": screen_geometry.COORD_SCREEN_PIXELS,
            }
            # Bug #9: Carry element label into step for menu-open heuristic.
            if el_id is not None and element_id_map and el_id in element_id_map:
                el_ref = element_id_map[el_id]
                step_payload["label"] = el_ref.get("label", "") if isinstance(el_ref, dict) else ""
                resolved = _resolve_element_point(el_ref)
                if resolved and resolved.get("uid"):
                    step_payload["ui_uid"] = resolved["uid"]
                if resolved and resolved.get("runtime_id"):
                    step_payload["ui_runtime_id"] = resolved["runtime_id"]
            elif raw_ui_uid:
                step_payload["ui_uid"] = raw_ui_uid
                if raw_step.get("ui_runtime_id"):
                    step_payload["ui_runtime_id"] = raw_step["ui_runtime_id"]

            normalized["steps"].append(step_payload)
        elif action == "scroll":
            direction = str(raw_step.get("direction", "down") or "down").strip().lower()
            if direction not in {"up", "down", "left", "right"}:
                return _reject("The screen planner returned an invalid scroll direction.")
            normalized["steps"].append(
                {
                    "action": "scroll",
                    "direction": direction,
                    "amount": max(1, min(_coerce_int(raw_step.get("amount"), 600), 10000)),
                }
            )
        elif action == "type":
            text = raw_step.get("text", "")
            if not isinstance(text, str):
                # F45: a non-string payload would be coerced by the executor's
                # typing layer into something the planner never produced.
                return _reject("The screen planner returned a malformed typing step.")
            step_payload = {"action": "type", "text": text}

            el_id = _coerce_int(raw_step.get("element_id"), None)
            if el_id is not None and element_id_map and el_id in element_id_map:
                resolved = _resolve_element_point(element_id_map[el_id])
                if resolved and resolved.get("uid"):
                    step_payload["ui_uid"] = resolved["uid"]
                if resolved and resolved.get("runtime_id"):
                    step_payload["ui_runtime_id"] = resolved["runtime_id"]
            elif el_id is not None and element_id_map:
                logging.warning(
                    "Screen plan referenced missing element_id=%s for type action",
                    el_id,
                )
                return _reject(
                    "The local screen planner referenced a text field that is "
                    "not in the current UI tree."
                )
            elif raw_step.get("ui_uid"):
                step_payload["ui_uid"] = raw_step["ui_uid"]
                if raw_step.get("ui_runtime_id"):
                    step_payload["ui_runtime_id"] = raw_step["ui_runtime_id"]

            # F21: mark a typing step whose target is a sensitive field, so the
            # redacted egress boundary knows the innocuous "text" key holds a
            # credential. Evidence comes from the step itself or the element it
            # names — never from a guess.
            _mark_step_sensitive(step_payload, raw_step, element_id_map)

            normalized["steps"].append(step_payload)
        elif action in {"press", "hotkey"}:
            keys = raw_step.get("keys", [])
            if isinstance(keys, str):
                keys = [keys]
            if not isinstance(keys, list) or not keys:
                return _reject("The screen planner returned a malformed key step.")
            if any(not isinstance(key, str) or not key.strip() for key in keys):
                return _reject("The screen planner returned a malformed key step.")
            normalized["steps"].append(
                {"action": action, "keys": [k.strip() for k in keys]}
            )
        else:
            # F45: an unsupported action is not silently dropped — a plan whose
            # steps cannot all be honoured must not run partially.
            return _reject(
                "The screen planner asked for an action I don't support."
            )

    # P1.4: Raw-coordinate / bbox clicks are less certain than resolved
    # element_ids: force the confirmation gate when confidence is below the
    # coordinate bar. Global thresholds (0.82 / 0.90 / 0.58) are untouched;
    # element_id targets are never affected by this rule.
    if raw_coord_click and normalized["ok"] and normalized["confidence"] < COORDINATE_CONFIRMATION_THRESHOLD:
        normalized["needs_confirmation"] = True

    if not normalized["steps"]:
        normalized["ok"] = False
        if not normalized["reason"]:
            normalized["reason"] = "I couldn't produce any screen steps for that command."

    return normalized


def _build_tree_prompt(command_text, capture, ui_context="", image_only=False):
    title = capture["window_title"] or "Unknown window"
    history_context = screen_state.format_history_for_prompt()

    if image_only:
        coordinate_examples = (
            'Example - user says "click the heart icon" (visible in the screenshot near grid x=660, y=550):\n'
            '{"ok": true, "confidence": 0.90, "summary": "Clicking the heart icon, sir.", '
            '"needs_confirmation": false, "coordinate_space": "normalized_1000", '
            '"reason": "Image-only mode: located from the screenshot grid.", '
            '"steps": [{"action": "click", "x": 660, "y": 550}]}\n\n'
        )
        return (
            "You are an autonomous computer agent. You are driving the computer by looking at a screenshot.\n"
            "Return only a JSON object with keys: ok, confidence, summary, needs_confirmation, reason, steps, coordinate_space.\n"
            f"{coordinate_examples}"
            f"Active window title: {title}\n"
            f"{history_context}"
            "IMAGE-ONLY MODE: UI Automation and OCR found no readable controls on this screen, so "
            "the screenshot and its coordinate grid are your ONLY evidence. Absence from the "
            "accessibility tree does NOT mean absence from the image — locate the target visually.\n"
            "Allowed step actions: click, double_click, right_click, type, press, hotkey, scroll.\n"
            "CRITICAL: You MUST use EXACTLY one of these allowed action verbs. Do NOT write descriptive actions.\n"
            "Never use element_id in this mode — there is no UI tree. Return x, y (and any bounding "
            "box) in NORMALIZED 0..1000 coordinates on each axis: 0 = left/top edge of the image, "
            "1000 = right/bottom edge. Grid ticks are drawn every 10 percent with labels 100, 200, "
            "... 900 in both axes. Prefer a bounding box ({x, y, width, height} or {left, top, "
            "right, bottom}, same normalized scale) over a plain point when the target is larger "
            "than a few pixels; we click the box center. Estimate coordinates from the screenshot "
            "grid; never copy example numbers.\n"
            "Set \"coordinate_space\" to \"normalized_1000\" in every reply so we can validate the "
            "coordinate space you used.\n"
            "If the target is ambiguous (several visually similar candidates), return ok:false with "
            "a reason asking the user which region or element they mean instead of guessing.\n"
            "For type steps include text. For press/hotkey steps include keys as a JSON array.\n"
            "Set ok to false ONLY when the target is genuinely not visible or you cannot locate it precisely.\n"
            "Set needs_confirmation true for risky actions like send, delete, submit, pay, buy, purchase, close, quit, or exit.\n"
            f"User command: {command_text}"
        )

    few_shot_examples = (
        'Example - user says "click the search bar":\n'
        '{"ok": true, "confidence": 0.92, "summary": "Clicking the search bar, sir.", '
        '"needs_confirmation": false, "reason": "", '
        '"steps": [{"action": "click", "element_id": 14}]}\n\n'
        'Example - user says "type hello world" (cursor already in a text field):\n'
        '{"ok": true, "confidence": 0.95, "summary": "Typing that now, sir.", '
        '"needs_confirmation": false, "reason": "", '
        '"steps": [{"action": "type", "text": "hello world"}]}\n\n'
        'Example - user says "type hello into the search box":\n'
        '{"ok": true, "confidence": 0.94, "summary": "Typing into Search, sir.", '
        '"needs_confirmation": false, "reason": "", '
        '"steps": [{"action": "click", "element_id": 9}, {"action": "type", "text": "hello"}]}\n\n'
        'Example - user says "click the heart icon" (icon not in the UI tree, visible in the screenshot at x=660, y=550):\n'
        '{"ok": true, "confidence": 0.90, "summary": "Clicking the heart icon, sir.", '
        '"needs_confirmation": false, "coordinate_space": "normalized_1000", '
        '"reason": "Used screenshot coordinates because the icon was not in the UI tree.", '
        '"steps": [{"action": "click", "x": 660, "y": 550}]}\n\n'
        'Example - user says "click the search icon" (icon not in the UI tree, visible in the screenshot at x=922, y=83):\n'
        '{"ok": true, "confidence": 0.89, "summary": "Clicking the search icon, sir.", '
        '"needs_confirmation": false, "coordinate_space": "normalized_1000", '
        '"reason": "Used screenshot coordinates because the icon was not in the UI tree.", '
        '"steps": [{"action": "click", "x": 922, "y": 83}]}\n\n'
    )

    return (
        "You are an autonomous computer agent. You are driving the computer by reading its active Accessibility Tree.\n"
        "Return only a JSON object with keys: ok, confidence, summary, needs_confirmation, reason, steps, coordinate_space.\n"
        f"{few_shot_examples}"
        f"Active window title: {title}\n"
        f"{history_context}"
        "Below is the XML-like structure of the active window. Every element includes its role, bounds, center, and any known labels.\n"
        "Bounds and centers are in capture-local vision pixels, not raw screen pixels.\n"
        "Use the tree hierarchy and the metadata to disambiguate repeated labels.\n"
        "--- UI TREE START ---\n"
        f"{ui_context}\n"
        "--- UI TREE END ---\n"
        "Allowed step actions: click, double_click, right_click, type, press, hotkey, scroll.\n"
        "CRITICAL: You MUST use EXACTLY one of these allowed action verbs. Do NOT write descriptive actions like 'Locate element' or 'Simulate click event'. Just write 'click'.\n"
        "For click-like steps, PREFER to return the integer ID of the target element in the 'element_id' field when a listed ID matches the target. Only use element_id values that appear verbatim in the UI tree below.\n"
        "If no listed ID matches but the target is clearly visible in the screenshot, PREFER returning a bounding box for the target. A plain x,y point tends to land on the corner of an icon instead of its center and cause misses - return a bounding box whenever the target is larger than a few pixels, and make x,y the CENTER of the target. Return the step with a bounding box when you can -- either {x, y, width, height} or {left, top, right, bottom} (as {\"x\":, \"y\":, \"width\":, \"height\":} or {\"left\":, \"top\":, \"right\":, \"bottom\":}) with all box coordinates in the same NORMALIZED 0..1000 scale as x,y and we will click its center; plain x and y remains an acceptable fallback if you cannot estimate the box. Return x, y, and any bounding box in NORMALIZED 0..1000 coordinates on each axis: 0 = left/top edge of the image, 1000 = right/bottom edge. Numbered boxes are element ids from the UI tree; grid ticks are drawn every 10 percent with labels 100, 200, ... 900 in both axes (normalized 0..1000). When several visually similar icons are on screen, prefer the one in the main content or feed area over sidebars and navigation panels. Estimate coordinates from the screenshot grid and boxes; never copy example numbers.\n"
        "For type steps include text. For press/hotkey steps include keys as a JSON array.\n"
        "If the user wants text entered into a visible field, include a click step on the field's ID before the type step.\n"
        "Prefer UI Automation elements over OCR-only text when both describe the same target.\n"
        "F45: any raw coordinate you return MUST be accompanied by \"coordinate_space\": "
        "\"normalized_1000\". A reply containing coordinates without a declared coordinate space is "
        "rejected, so always state it.\n"
        "Set ok to false ONLY when the target is genuinely not visible or you cannot locate it precisely.\n"
        "Set needs_confirmation true for risky actions like send, delete, submit, pay, buy, purchase, close, quit, or exit.\n"
        f"User command: {command_text}"
    )


def _build_vision_prompt(command_text, capture, ocr_context="", ui_context="", image_only=False):
    merged_context = "\n".join(
        fragment.strip()
        for fragment in (ui_context, ocr_context)
        if fragment and fragment.strip()
    )
    return _build_tree_prompt(command_text, capture, ui_context=merged_context, image_only=image_only)


def _extract_response_content(result):
    """Pull the text content from a vision API response.

    Some models (e.g. Nemotron) put the answer in a 'reasoning' field
    instead of 'content'.  This helper checks both.
    """
    if not result or not result.get("choices"):
        return None
    msg = result["choices"][0].get("message", {})
    content = msg.get("content") or ""
    if content.strip():
        return content
    # Fallback: some reasoning models put the JSON inside 'reasoning'.
    reasoning = msg.get("reasoning") or ""
    if reasoning.strip():
        return reasoning
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


# ── F37: one eligible-provider cascade (shared with screen Q&A) ────────────
# The old cascade only offered the registry-selected provider as "primary" and
# then always ended with an UNGUARDED Groq attempt — a supported
# single-provider installation needed an unrelated Groq key, one adapter
# exception escaped the call, and any nonempty text (including malformed JSON)
# counted as success. Eligibility, ordering, bounding and the actual-attempt
# record now live in backend/services/vision_cascade.py.

def _dispatch_gemini(prompt, image_data_url, model, max_completion_tokens, response_format):
    return gemini_client.ask_gemini_vision(
        prompt,
        image_data_url,
        max_completion_tokens=max_completion_tokens,
        response_format=response_format,
        model=model,
    )


def _dispatch_openrouter(prompt, image_data_url, model, max_completion_tokens, response_format):
    from backend.services.openrouter_client import ask_openrouter_vision
    return ask_openrouter_vision(
        prompt,
        image_data_url,
        max_completion_tokens=max_completion_tokens,
        model=model,
        response_format=response_format,
    )


def _dispatch_fireworks(prompt, image_data_url, model, max_completion_tokens, response_format):
    from backend.services.fireworks_client import ask_fireworks_vision
    return ask_fireworks_vision(
        prompt,
        image_data_url,
        max_completion_tokens=max_completion_tokens,
        model=model,
        response_format=response_format,
    )


def _dispatch_groq(prompt, image_data_url, model, max_completion_tokens, response_format):
    # NOTE: no response_format here -- Qwen's  thinking blocks break Groq's
    # strict JSON validation; prompts already request JSON and the
    # _extract_json_object helper parses it. Generous token budget because
    # Qwen burns tokens on  thinking before the JSON answer.
    return ask_groq_vision(
        prompt,
        image_data_url,
        max_completion_tokens=max(max_completion_tokens, 4096),
        model=model,
    )


def _vision_dispatchers():
    """provider id -> adapter call for this call site (resolved lazily)."""
    return {
        "gemini": _dispatch_gemini,
        "openrouter": _dispatch_openrouter,
        "fireworks": _dispatch_fireworks,
        "groq": _dispatch_groq,
    }


def _screen_control_vision_schema(result):
    """F37 schema validation: a response is usable only when it carries a JSON
    object — nonempty prose/malformed JSON must ADVANCE the cascade."""
    content = _extract_response_content(result)
    if content is None:
        return False, "no assistant content"
    parsed = _extract_json_object(content)
    if not isinstance(parsed, dict) or not parsed:
        return False, "content was not a JSON object"
    return True, ""


def _ask_vision_cascade(prompt, image_data_url, max_completion_tokens=800,
                        response_format=None, attempts_out=None):
    """Vision plan through the ONE eligible-provider cascade (F37).

    The registry-selected (provider, model) is attempted first when eligible;
    every other ELIGIBLE configured provider follows, each at most once and
    bounded by the cascade cap. Providers without a credential are never
    dispatched, adapter exceptions and schema-invalid output advance safely,
    and *attempts_out* (optional dict) receives the actual-attempt record.
    """
    result, report = vision_cascade.ask_vision_with_fallback(
        prompt,
        image_data_url,
        dispatchers=_vision_dispatchers(),
        validate=_screen_control_vision_schema,
        selected=_resolve_vision_model(),
        max_completion_tokens=max_completion_tokens,
        response_format=response_format,
        attempts_out=attempts_out,
    )
    if report.get("usable"):
        logging.info("Vision response from %s (%s)",
                     report.get("provider"), report.get("model") or "default")
    else:
        logging.warning("%s", report.get("unavailable_reason"))
    return result if result is not None else {}


# Coordinate spaces accepted from image-only vision plans. The planner is
# taught the normalized 0..1000 grid; anything else is rejected rather than
# mis-scaled into clicks.
_IMAGE_ONLY_COORD_SPACES = ("normalized_1000", "normalized", "0..1000")


def _request_vision_plan(command_text, capture, ocr_context="", ui_context="", element_id_map=None, image_only=False):
    prompt = _build_vision_prompt(command_text, capture, ocr_context, ui_context, image_only=image_only)

    # F37: the ACTUAL attempted providers/models are recorded per call, so the
    # plan can name the provider that really answered instead of "unknown".
    attempts = {}
    result = _ask_vision_cascade(
        prompt,
        capture["image_data_url"],
        max_completion_tokens=800,
        response_format={"type": "json_object"},
        attempts_out=attempts,
    )

    content = _extract_response_content(result)
    provider = attempts.get("provider")
    if not provider and isinstance(result, dict):
        provider = result.get("model") or result.get("vision_provider") or result.get("vision_model")
    model = attempts.get("model")
    if content is None:
        return {
            "ok": False,
            "confidence": 0.0,
            "summary": "",
            "needs_confirmation": False,
            "reason": "The screen-vision model did not return a result. %s"
                      % (vision_cascade.unavailable_reason(attempts),),
            "capture_mode": capture.get("capture_mode", ""),
            "steps": [],
            "source": "Vision Match",
            "raw_vision": str(result)[:2000] if result else "",
            "vision_provider": provider or "unknown",
            "vision_model": model,
            "vision_attempts": list(attempts.get("attempts") or []),
        }

    raw_obj = _extract_json_object(content)
    if image_only:
        # F45: validate the DECLARED coordinate space before trusting any
        # point. A missing space is no longer defaulted to normalized_1000 —
        # guessing the frame is exactly how a hallucinated coordinate became a
        # real click. Missing or unrecognised spaces reject the plan outright.
        space = str(raw_obj.get("coordinate_space", "") or "").strip().lower()
        if not space or space not in _IMAGE_ONLY_COORD_SPACES:
            return {
                "ok": False,
                "confidence": 0.0,
                "summary": "",
                "needs_confirmation": False,
                "reason": ("The screen-vision model returned coordinates in an "
                           "unexpected space."),
                "capture_mode": capture.get("capture_mode", ""),
                "steps": [],
                "source": "Vision Match",
                "raw_vision": content,
                "vision_provider": provider or "unknown",
                "vision_model": model,
                "vision_attempts": list(attempts.get("attempts") or []),
            }
    plan = _normalize_vision_plan(raw_obj, capture, element_id_map)
    plan["source"] = "Vision Match"
    plan["raw_vision"] = content
    plan["vision_model"] = model
    plan["vision_attempts"] = list(attempts.get("attempts") or [])
    if image_only:
        plan["image_only"] = True
    if provider:
        plan["vision_provider"] = provider
    else:
        # fallback: try to infer from logging context; keep minimal
        plan["vision_provider"] = "unknown"
    return plan


def _filter_tree_to_relevant(elements, target_label, max_elements=40):
    """Keep elements that mention the target word or are clickable.

    F44: candidate nodes are RANKED by relevance, but the returned list
    preserves the ORIGINAL walk order and re-attaches every kept candidate's
    ancestors. The downstream depth-based XML serialization therefore
    reconstructs the real hierarchy instead of a score-sorted flat list with
    dangling children.
    """
    # P0-B: icon targets get larger anchor budget so feed/scrollable containers survive
    if target_label and _is_icon_target(target_label):
        max_elements = max(max_elements, 80)
    if not target_label:
        return list(elements)

    target_words = set(target_label.lower().split())
    by_uid = {}
    for el in elements:
        uid = el.get("uid")
        if uid:
            by_uid.setdefault(uid, el)

    ranked = []
    for index, el in enumerate(elements):
        name = (el.get("name") or "").lower()
        name_words = set(name.split())
        relevance = len(target_words & name_words)
        is_clickable = el.get("control_type", "") in CLICKABLE_CONTROL_TYPES
        ranked.append(((relevance * 2 + int(is_clickable)), index, el))
    ranked.sort(key=lambda x: -x[0])
    top = ranked[:max_elements]

    kept_ids = set()
    bare_top = set()
    for _, _, el in top:
        uid = el.get("uid")
        if not uid:
            bare_top.add(id(el))
            continue
        # F44: walk the parent chain so every ancestor of a ranked candidate
        # survives in the tree (stable parent identity, original hierarchy).
        cur = uid
        guard = 0
        while cur and cur not in kept_ids and guard < 64:
            kept_ids.add(cur)
            parent = by_uid.get(cur, {}).get("parent_uid")
            cur = parent
            guard += 1

    if not kept_ids and not bare_top:
        return []

    kept = [
        el for el in elements
        if (el.get("uid") in kept_ids) or (id(el) in bare_top)
    ]
    return kept


def _gather_ui_tree(capture, snapshot=None, target_label=None):
    """Build an XML-like Accessibility Tree for the local Text AI to read.

    Combines native Windows UI Automation with Tesseract OCR to ensure
    hidden web apps (like Electron) are still readable.
    """
    tree_lines = []
    element_id_map = {}
    id_counter = 1
    snapshot = snapshot or _build_screen_snapshot(capture)
    raw_ui = snapshot.get("raw_ui", [])
    ui_elements = snapshot.get("ui_elements", [])

    # Bug #7: Filter large trees to relevant elements. P0-B: icon targets keep feed containers
    budget = 80 if target_label and _is_icon_target(target_label) else 40
    if target_label and len(ui_elements) > budget:
        ui_elements = _filter_tree_to_relevant(ui_elements, target_label, max_elements=budget)

    title = _escape_xml_attr(capture.get("window_title", "Unknown"))
    capture_mode = _escape_xml_attr(capture.get("capture_mode", "unknown"))
    tree_lines.append(
        f'<window title="{title}" capture_mode="{capture_mode}" size="{capture.get("vision_width", 0)}x{capture.get("vision_height", 0)}">'
    )

    if ui_elements:
        ui_lines, ui_map, id_counter = _serialize_accessibility_elements(
            ui_elements,
            start_id=id_counter,
            # F43: never label pixels with a space they are not in. When the
            # capture lacks the size metadata needed for the screen->vision
            # transform the elements stay in screen pixels, and declaring them
            # "vision" would make the executor convert them a second time.
            space=snapshot.get("ui_space", "screen"),
        )
        tree_lines.append("  <accessibility>")
        tree_lines.extend(ui_lines)
        tree_lines.append("  </accessibility>")
        element_id_map.update(ui_map)

    # 2. Visual OCR (Essential for Electron apps like Antigravity)
    merged = snapshot.get("ocr_merged", [])
    if merged:
        try:
            seen_texts = [el.get("name", "") for el in raw_ui]
            ocr_lines, ocr_map, id_counter = _serialize_ocr_nodes(
                merged,
                seen_texts,
                start_id=id_counter,
            )
            if ocr_lines:
                tree_lines.append("  <ocr>")
                tree_lines.extend(ocr_lines)
                tree_lines.append("  </ocr>")
                element_id_map.update(ocr_map)
        except Exception as exc:
            logging.warning("OCR tree extraction failed: %s", exc)

    tree_lines.append('</window>')
    return "\n".join(tree_lines), element_id_map


def _extract_action_target(command_text):
    """Extract the likely target element name from a voice command.

    E.g. 'click the search bar' -> 'search'
         'close the settings tab' -> 'settings'
         'click on File menu' -> 'file'
    """
    targeted_entry = _extract_targeted_text_entry(command_text)
    if targeted_entry:
        return targeted_entry["target"]

    text = command_text.lower().strip()
    # Strip common action verbs and filler words.
    for prefix in (
        "click on the ", "click on ", "click the ", "click ",
        "tap on the ", "tap on ", "tap the ", "tap ",
        "press the ", "press ", "hit the ", "hit ",
        "open the ", "open ", "close the ", "close ",
        "select the ", "select ", "go to the ", "go to ",
        "switch to the ", "switch to ",
        "minimize the ", "minimize ", "maximise the ", "maximise ",
        "maximize the ", "maximize ",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return _clean_action_target_text(text)


def _text_match_class(target, candidate):
    """How a candidate label relates to the target, as a match class (F29).

    The classes are ordered and *disjoint*:

      4 — the labels are the same text;
      3 — a declared UI alias makes them semantically equivalent;
      2 — one label contains the other;
      1 — strong fuzzy similarity (sequence ratio or word overlap);
      0 — no usable relation.

    The caller scores from the class, so a control-type bonus can never lift
    a weaker class above a stronger one.
    """
    t = target.lower().strip()
    c = candidate.lower().strip()
    if not t or not c:
        return 0
    if t == c:
        return 4

    # Check UI aliases for semantic equivalence.
    for alias_key, aliases in UI_ALIASES.items():
        t_matches = t == alias_key or t in aliases
        c_matches = c == alias_key or c in aliases
        if t_matches and c_matches:
            return 3

    if t in c or c in t:
        return 2

    # Sequence-level similarity (replaces loose character-set overlap).
    if len(t) > 3 and len(c) > 3:
        if difflib.SequenceMatcher(None, t, c).ratio() > 0.75:
            return 1

    # Word overlap.
    t_words = set(t.split())
    c_words = set(c.split())
    if t_words and c_words:
        overlap = len(t_words & c_words) / max(len(t_words), len(c_words))
        if overlap >= 0.5:
            return 1
    return 0


#: F29: disjoint score bands, keyed by match class. ``(floor, width)``. The
#: narrowest inter-band gap (0.05) is wider than the largest control bonus
#: (0.04), so additive bonuses can reorder candidates *within* a class but
#: never promote a materially different fuzzy target above an exact one.
_MATCH_BANDS = {
    4: (0.95, 0.05),
    3: (0.85, 0.05),
    2: (0.70, 0.08),
    1: (0.55, 0.10),
}


def _score_text_match(target, candidate):
    """Return 0-1 similarity between target text and a candidate label.

    F29: the score is a class-disjoint band rather than a free-running
    similarity, so "Save As" cannot outrank an exact "Save" merely because
    one of them is a Button and the other is not.
    """
    rank = _text_match_class(target, candidate)
    if rank <= 0:
        return 0.0

    floor, width = _MATCH_BANDS[rank]
    if rank >= 3:
        fraction = 1.0
    else:
        t = target.lower().strip()
        c = candidate.lower().strip()
        if rank == 2:
            shorter = float(min(len(t), len(c)))
            longer = float(max(len(t), len(c))) or 1.0
            fraction = max(0.0, min(1.0, shorter / longer))
        elif len(t) > 3 and len(c) > 3:
            fraction = max(
                0.0, min(1.0, difflib.SequenceMatcher(None, t, c).ratio()))
        else:
            t_words = set(t.split())
            c_words = set(c.split())
            if t_words and c_words:
                fraction = len(t_words & c_words) / float(
                    max(len(t_words), len(c_words)))
            else:
                fraction = 0.5
    return floor + fraction * width


def _control_priority(control_type, source):
    if source == "ocr":
        return 0
    if control_type in CLICKABLE_CONTROL_TYPES:
        return 4
    if control_type in TEXT_ENTRY_CONTROL_TYPES:
        return 3
    if control_type in PASSIVE_CONTROL_TYPES:
        return 1
    return 2


def _score_band_limits(value):
    """The ``(low, high)`` score band that *value* already falls inside (F29)."""
    for _rank, (floor, width) in _MATCH_BANDS.items():
        if floor - 1e-9 <= value <= floor + width + 1e-9:
            return floor, floor + width
    return None


def _score_ui_element_match(target, elem):
    name = elem.get("name", "")
    ctype = elem.get("control_type", "")
    label = f"{name} {ctype}".strip()
    raw = max(_score_text_match(target, name), _score_text_match(target, label))
    if raw <= 0:
        return 0.0

    # F29: bonuses order candidates WITHIN their match class only. They are
    # never allowed to lift a materially different target — "Save As" on a
    # Button — above an exact "Save", so the total is clamped back into the
    # band the text match already earned.
    bonus = 0.0
    if ctype in CLICKABLE_CONTROL_TYPES:
        bonus += 0.04
    elif ctype in TEXT_ENTRY_CONTROL_TYPES:
        bonus += 0.02
    elif ctype in PASSIVE_CONTROL_TYPES:
        bonus -= 0.03
    # F29: an UNKNOWN enabled state is not evidence of actionability. Only an
    # explicit True earns the bonus; anything else gets nothing.
    if elem.get("enabled") is True:
        bonus += 0.01

    score = raw + bonus
    limits = _score_band_limits(raw)
    if limits:
        score = max(limits[0], min(score, limits[1]))
    return max(0.0, min(score, 1.0))


def _is_better_match(score, priority, best_score, best_priority):
    if score > best_score + 0.001:
        return True
    if abs(score - best_score) <= 0.001 and priority > best_priority:
        return True
    return False


def _extract_action_verb(command_text):
    """Determine click type from the command text."""
    normalized = _normalize_text(command_text)
    if re.search(r"\bright.?click\b", normalized):
        return "right_click"
    if re.search(r"\bdouble.?click\b", normalized):
        return "double_click"
    return "click"


def _tiebreak_by_proximity(candidates, capture=None, to_screen=None):
    """When scores tie, prefer the element nearest the current cursor.

    F44: the cursor position is in **screen pixels**, while candidates may be
    recorded in the vision frame. Comparing them directly mixed coordinate
    frames (a candidate at vision (100, 100) was treated as if it were at
    screen (100, 100)). Every candidate is converted into the screen frame
    first, so a tie breaks on real distance; if the conversion is unavailable
    the candidate keeps its own coordinates rather than being compared across
    frames.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    try:
        import ctypes
        from ctypes import wintypes
        pt = wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        cursor_x, cursor_y = pt.x, pt.y
    except Exception:
        return candidates[0]

    def _screen_point(candidate):
        x = candidate.get("cx", 0)
        y = candidate.get("cy", 0)
        if to_screen is None:
            return x, y
        try:
            converted = to_screen(x, y, screen_geometry.normalize_space(
                candidate.get("space"), screen_geometry.COORD_VISION_PIXELS))
        except Exception:
            return x, y
        return converted if converted else (x, y)

    def _distance(candidate):
        x, y = _screen_point(candidate)
        return abs(x - cursor_x) + abs(y - cursor_y)

    return min(candidates, key=_distance)


def _is_icon_target(target):
    """P1.5: icon guard - true for icon-like targets that should not be
    fuzzy-matched via low UI/OCR scores (a "Heart" caption vs the like button).

    True when the noun explicitly names an icon/button, or when it is a bare
    common icon glyph (heart, star, arrow, bell, camera, ...).
    """
    if not target:
        return False
    low = target.lower()
    if re.search(r"\b(icon|button)\b", low):
        return True
    words = set(re.findall(r"[a-z0-9]+", low))
    return bool(words & _ICON_NOUNS)


def _try_local_match(command_text, capture, snapshot=None):
    """Try to find the target element using OCR/UI data, no API call.

    Returns a plan dict if a confident match is found, else None.
    """
    targeted_entry = _extract_targeted_text_entry(command_text)
    target = targeted_entry["target"] if targeted_entry else _extract_action_target(command_text)
    if not target or len(target) < 2:
        return None

    best_score = 0.0
    best_x = 0
    best_y = 0
    best_label = ""
    best_source = ""
    best_space = screen_geometry.COORD_VISION_PIXELS
    best_priority = -1
    best_uid = ""
    tied_candidates = []  # Bug #6: collect ties for disambiguation
    # F29: a local fallback pick may only auto-execute when the target was
    # identified uniquely. Duplicate labels, or a tie between materially
    # different targets that cursor proximity would silently break, must not
    # regain automatic execution — the screenshot path handles those.
    ambiguous = False
    snapshot = snapshot or _build_screen_snapshot(capture)

    # -- Check UI Automation elements first (more reliable coordinates) --
    for elem in snapshot.get("ui_elements", []):
        name = elem.get("name", "")
        ctype = elem.get("control_type", "")
        if targeted_entry and ctype not in TEXT_ENTRY_CONTROL_TYPES:
            continue
        # F29: fail closed on actionability. A control that is not explicitly
        # enabled must never be clicked just because it matched by text.
        if elem.get("enabled") is not True:
            continue
        score = _score_ui_element_match(target, elem)
        if score <= 0:
            continue
        priority = _control_priority(ctype, "ui")
        if _is_better_match(score, priority, best_score, best_priority):
            best_score = score
            best_priority = priority
            best_x = (elem["left"] + elem["right"]) // 2
            best_y = (elem["top"] + elem["bottom"]) // 2
            best_label = name or ctype
            best_source = "ui"
            best_space = snapshot.get("ui_space", screen_geometry.COORD_VISION_PIXELS)
            best_uid = elem.get("uid", "")
            tied_candidates = [{"cx": best_x, "cy": best_y, "label": best_label,
                                "source": "ui", "space": best_space, "uid": best_uid}]
        elif abs(score - best_score) <= 0.001 and priority >= best_priority:
            tied_candidates.append({
                "cx": (elem["left"] + elem["right"]) // 2,
                "cy": (elem["top"] + elem["bottom"]) // 2,
                "label": name or ctype, "source": "ui",
                "space": snapshot.get("ui_space", screen_geometry.COORD_VISION_PIXELS),
                "uid": elem.get("uid", ""),
            })

    if not targeted_entry:
        # -- Check OCR regions --
        for region in snapshot.get("ocr_merged", []):
            text = region.get("text", "")
            # F29: the OCR fallback is the weakest evidence, so it takes an
            # extra class-preserving penalty. It can still match, but a
            # different-text target cannot be promoted to automatic execution
            # by the fallback route.
            score = max(0.0, _score_text_match(target, text) - 0.02)
            if score <= 0:
                continue
            priority = _control_priority("", "ocr")
            if _is_better_match(score, priority, best_score, best_priority):
                best_score = score
                best_priority = priority
                best_x = (region["left"] + region["right"]) // 2
                best_y = (region["top"] + region["bottom"]) // 2
                best_label = text
                best_source = "ocr"
                best_space = screen_geometry.COORD_VISION_PIXELS
                best_uid = ""

    # Need a strong match to skip the vision model entirely.
    # P1.5 icon guard: icon-like targets need higher confidence from UI/OCR (not exact UIA name)
    if (_is_icon_target(target) or _is_icon_target(command_text)) and best_source in ("ui", "ocr"):
        if best_score < ICON_MATCH_THRESHOLD:
            return None
    else:
        if best_score < 0.6:
            return None

    # Bug #6: Break ties by proximity to current cursor position.
    # F44: both sides are put in the screen frame before the distance is
    # computed, so the comparison is never across coordinate spaces.
    if len(tied_candidates) > 1:
        # F29: two candidates are the same target only when they are the SAME
        # control (same observation uid). Repeated labels — the repeated Edit
        # / Delete buttons of a list — and different fuzzy targets are
        # ambiguous, and picking one by cursor proximity would silently choose
        # between materially different actions.
        uids = {(c.get("uid") or "") for c in tied_candidates}
        labels = {(c.get("label") or "").strip().lower() for c in tied_candidates}
        if "" in uids or len(uids) > 1 or len(labels) > 1:
            ambiguous = True
        else:
            winner = _tiebreak_by_proximity(
                tied_candidates,
                capture=capture,
                to_screen=lambda x, y, space: _step_point_to_screen(
                    screen_geometry.normalize_space(space,
                                                    screen_geometry.COORD_VISION_PIXELS),
                    x, y, capture),
            )
            if winner:
                best_x = winner["cx"]
                best_y = winner["cy"]
                best_label = winner["label"]
                best_source = winner["source"]
                best_space = winner["space"]
                best_uid = winner["uid"]

    if ambiguous:
        # Defer to the screenshot planner instead of guessing: it sees the
        # numbered boxes and can tell the two candidates apart, or decline.
        logging.info(
            "Local match for '%s' is ambiguous (%d tied candidates) — deferring "
            "to the vision planner.", target, len(tied_candidates),
        )
        return None

    if targeted_entry and best_source != "ui":
        return None

    logging.info(
        "Local match: '%s' -> '%s' (score=%.2f) at (%d, %d)",
        target, best_label, best_score, best_x, best_y,
    )

    if best_space == screen_geometry.COORD_VISION_PIXELS and all(
        key in capture
        for key in ("capture_width", "capture_height", "vision_width", "vision_height")
    ):
        best_x, best_y = _vision_to_screen_point(capture, best_x, best_y)

    # F29: only an exact / semantically equivalent label identifies the target
    # uniquely enough for the deterministic route to act without asking. A
    # substring or fuzzy label ("Save As" for "Save", "Lock" for "block") is a
    # candidate, not an identification, so it is held below the auto-execute
    # threshold and the caller confirms it or tries the screenshot planner.
    identified = _text_match_class(target, best_label) >= 3

    if targeted_entry:
        confidence = min(best_score, 0.95)
        if not identified:
            confidence = min(confidence, FAST_LOCAL_MATCH_CONFIDENCE - 0.01)
        steps = [
            {
                "action": "click",
                "x": best_x,
                "y": best_y,
                "description": f"Focus '{best_label}'",
            },
            {
                "action": "type",
                "text": targeted_entry["text"],
            },
        ]
        if best_uid:
            steps[0]["ui_uid"] = best_uid

        return {
            "ok": True,
            "confidence": confidence,
            "summary": f"Typing into '{best_label}', sir.",
            "needs_confirmation": confidence < FAST_LOCAL_MATCH_CONFIDENCE,
            "reason": "",
            "capture_mode": capture.get("capture_mode", ""),
            "window_title": capture.get("window_title", ""),
            "hwnd": capture.get("hwnd"),
            "steps": steps,
            "source": "Local Match",
        }

    # Determine the action type from the command.
    action = _extract_action_verb(command_text)
    action_label = action.replace("_", " ").title()

    confidence = min(best_score, 0.95)
    if not identified:
        confidence = min(confidence, FAST_LOCAL_MATCH_CONFIDENCE - 0.01)

    step = {
        "action": action,
        "x": best_x,
        "y": best_y,
        "description": f"{action_label} on '{best_label}'",
    }
    if best_uid:
        step["ui_uid"] = best_uid

    return {
        "ok": True,
        "confidence": confidence,
        "summary": f"{action_label}ing '{best_label}', sir.",
        "needs_confirmation": confidence < FAST_LOCAL_MATCH_CONFIDENCE,
        "reason": "",
        "capture_mode": capture.get("capture_mode", ""),
        "window_title": capture.get("window_title", ""),
        "hwnd": capture.get("hwnd"),
        "steps": [step],
        "source": "Local Match",
    }


def _build_som_elements(element_id_map):
    """Build a list of SoM overlay elements from the element_id_map."""
    som_elements = []
    for eid, point in element_id_map.items():
        bounds = point.get("bounds") if isinstance(point, dict) else None
        if not bounds:
            continue
        som_elements.append({
            "id": eid,
            "left": bounds.get("left", 0),
            "top": bounds.get("top", 0),
            "right": bounds.get("right", 0),
            "bottom": bounds.get("bottom", 0),
            "source": point.get("source", ""),
        })
    return som_elements


def _resolve_uia_target_hwnd():
    """The HWND the UIA-first fast path matches against (F29).

    Prefers the real foreground window; if Jarvis's own overlay owns the
    foreground, falls back to the last recorded non-Jarvis target (F43).
    """
    import ctypes

    try:
        fg = ctypes.windll.user32.GetForegroundWindow()
        if fg and not is_own_window(fg):
            return int(fg)
    except Exception:
        pass
    target = last_foreground_target() or {}
    hwnd = target.get("hwnd")
    return int(hwnd) if hwnd else 0
def _plan_with_tree(command_text):

    # F29: the UIA semantic fast path runs BEFORE any image capture or OCR.
    # Straightforward native controls are matched structurally and executed
    # through their UID; only missing/ambiguous/icon targets reach the camera.
    uia_first_plan = _try_uia_first(command_text)
    if uia_first_plan is not None:
        logging.info("UIA-first fast path matched (%s).", uia_first_plan.get("source"))
        return uia_first_plan
    from backend.services.screen_capture import annotate_capture

    def _attempt_capture(capture_function):
        try:
            capture = capture_function()
        except RuntimeError as exc:
            return None, {
                "ok": False,
                "confidence": 0.0,
                "summary": "",
                "needs_confirmation": False,
                "reason": str(exc),
                "capture_mode": "",
                "steps": [],
            }

        snapshot = _build_screen_snapshot(capture)
        # F21: remove anything that provably holds a secret from the pixels
        # BEFORE they are annotated, sent to the vision model or persisted.
        # OCR/AIA detection runs on the vision image, so the rectangles and the
        # image share one coordinate space.
        try:
            from backend.services import screen_capture as _screen_capture

            redact_elements = None
            if snapshot.get("ui_space") == screen_geometry.COORD_VISION_PIXELS:
                redact_elements = snapshot.get("ui_elements") or None
            blanked = _screen_capture.redact_capture(
                capture,
                words=(snapshot.get("ocr_regions")
                       or snapshot.get("ocr_merged")),
                elements=redact_elements,
            )
            if blanked:
                logging.info(
                    "F21: redacted %d sensitive region(s) from the screenshot.",
                    blanked)
        except Exception as exc:
            logging.warning("Screenshot redaction failed: %s", exc)
        best_plan = None
        best_failure = None

        # Fast path: use deterministic UIA/OCR matching before any model.
        local_plan = _try_local_match(command_text, capture, snapshot)
        if local_plan:
            if best_plan is None or _plan_selection_score(local_plan, command_text) > _plan_selection_score(
                best_plan, command_text
            ):
                best_plan = local_plan
            if local_plan.get("confidence", 0.0) >= FAST_LOCAL_MATCH_CONFIDENCE:
                logging.info("Fast local match - skipping AI planners.")
                return local_plan, None

        ui_tree, element_id_map = _gather_ui_tree(
            capture, snapshot,
            target_label=_extract_action_target(command_text),
        )
        if not element_id_map:
            # UIA and OCR provided no targets — absence from the tree must
            # not mean absence from the image. Send the screenshot (with its
            # coordinate grid) to the vision planner in image-only mode.
            logging.info("\n========== [VISION CASCADE CALL - IMAGE ONLY] ==========\n")
            vision_plan = _request_vision_plan(
                command_text, capture, ui_context="", element_id_map=None,
                image_only=True,
            )
            if vision_plan.get("ok"):
                logging.info(
                    "[IMAGE-ONLY VISION PLAN]\n%s\n============================================\n",
                    json.dumps(vision_plan, indent=2),
                )
                if best_plan is None or _plan_selection_score(vision_plan, command_text) > _plan_selection_score(
                    best_plan, command_text
                ):
                    best_plan = vision_plan
                return best_plan, None
            if best_failure is None:
                best_failure = vision_plan
            return best_plan, best_failure

        # Draw Set-of-Mark numbered bounding boxes on the screenshot
        # so the vision model can visually identify elements by their ID.
        som_elements = _build_som_elements(element_id_map)
        if som_elements:
            try:
                annotate_capture(capture, som_elements)
            except Exception as exc:
                logging.warning("SoM annotation failed: %s", exc)

        # ---- Vision cascade: Gemini Flash (primary) / Groq (fallback) ----
        # The vision model actually sees the SoM-annotated screenshot,
        # making it far more accurate than text-only Ollama at picking
        # the correct element by its numbered bounding box.
        logging.info("\n========== [VISION CASCADE CALL] ==========\n")
        vision_plan = _request_vision_plan(
            command_text, capture, ui_context=ui_tree, element_id_map=element_id_map,
        )

        if vision_plan.get("ok"):
            logging.info("[VISION PLAN]\n%s\n============================================\n", json.dumps(vision_plan, indent=2))
            if best_plan is None or _plan_selection_score(vision_plan, command_text) > _plan_selection_score(
                best_plan, command_text
            ):
                best_plan = vision_plan
            return best_plan, None

        # TARGETED RETRY: icon-only target missing from UI tree -> ask for coordinates
        if (
            not vision_plan.get("ok")
            and not vision_plan.get("steps")
            and _is_missing_element_failure(vision_plan.get("reason", ""))
        ):
            retry_command = command_text + "\n\n" + _RETRY_INSTRUCTION
            retry_plan = _request_vision_plan(
                retry_command, capture, ui_context=ui_tree, element_id_map=element_id_map,
            )
            if retry_plan.get("ok"):
                logging.info(
                    "[VISION RETRY PLAN]\n%s\n============================================\n",
                    json.dumps(retry_plan, indent=2),
                )
                if best_plan is None or _plan_selection_score(
                    retry_plan, retry_command
                ) > _plan_selection_score(best_plan, command_text):
                    best_plan = retry_plan
                return best_plan, None
            # retry also failed - surface its failure
            vision_plan = retry_plan

        # Neither planner succeeded — return the best we have.
        if best_plan is not None:
            return best_plan, None

        if best_failure is None:
            best_failure = vision_plan

        return best_plan, best_failure

    needs_desktop = _needs_desktop_capture(command_text)
    needs_whole_screen = _is_whole_screen_phrase(command_text)
    needs_primary = needs_desktop or needs_whole_screen
    first_capture = capture_primary_screen if needs_primary else capture_for_screen_control

    best_plan, best_failure = _attempt_capture(first_capture)
    if best_plan is not None:
        return best_plan

    if not needs_primary:
        retry_plan, retry_failure = _attempt_capture(capture_primary_screen)
        if retry_plan is not None:
            return retry_plan
        if retry_failure is not None:
            return retry_failure

    if best_failure is not None:
        return best_failure

    return {
        "ok": False,
        "confidence": 0.0,
        "summary": "",
        "needs_confirmation": False,
        "reason": "I couldn't find a confident screen action for that.",
        "capture_mode": "",
        "steps": [],
    }



def _try_uia_first(command_text):
    """F29 — run the UIA semantic fast path BEFORE any screenshot or OCR.

    Queries the live accessibility tree for an ENABLED, UNAMBIGUOUS target
    and builds a plan that executes through the control's UID (revalidated by
    fresh UIA resolution at execution time, F42). No image is captured, no
    OCR runs and no cloud vision is called.

    Returns a plan in screen-pixel space, or None when the target is missing,
    icon-like or ambiguous so ``_plan_with_tree`` falls through to the full
    capture + OCR + vision pipeline.
    """
    if not screen_ui_elements.is_available():
        return None

    targeted_entry = _extract_targeted_text_entry(command_text)
    target = (
        targeted_entry["target"] if targeted_entry
        else _extract_action_target(command_text)
    )
    if not target or len(target) < 2:
        return None
    # Icon-only targets are visually ambiguous: a UIA caption may name a
    # sibling rather than the control itself, so they stay on the vision path.
    if _is_icon_target(target):
        return None

    hwnd = _resolve_uia_target_hwnd()
    if not hwnd:
        return None
    try:
        raw_ui = screen_ui_elements.get_foreground_window_elements(hwnd=hwnd)
    except Exception as exc:
        logging.debug("UIA-first walk failed: %s", exc)
        return None
    if not raw_ui:
        return None

    scored = []
    for elem in raw_ui:
        ctype = elem.get("control_type", "")
        if targeted_entry and ctype not in TEXT_ENTRY_CONTROL_TYPES:
            continue
        # F29: unknown enabled state is not evidence of actionability.
        if elem.get("enabled") is not True:
            continue
        score = _score_ui_element_match(target, elem)
        if score <= 0:
            continue
        scored.append((score, _control_priority(ctype, "ui"), elem))

    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], -t[1]))
    best_score, _best_prio, best_elem = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0

    # Unambiguous: a strict winning margin. Two controls sharing the label
    # (F44's repeated labels) make the fast path decline instead of guessing.
    if (best_score - second_score) < UIA_FIRST_UNAMBIGUOUS_MARGIN:
        return None
    if best_score < UIA_FIRST_MIN_CONFIDENCE:
        return None

    left = _coerce_int(best_elem.get("left"), 0)
    top = _coerce_int(best_elem.get("top"), 0)
    right = _coerce_int(best_elem.get("right"), 0)
    bottom = _coerce_int(best_elem.get("bottom"), 0)
    cx = int(round((left + right) / 2.0))
    cy = int(round((top + bottom) / 2.0))
    label = (
        (best_elem.get("name") or "").strip()
        or (best_elem.get("control_type") or "element")
    )
    uid = best_elem.get("uid", "")
    runtime_id = best_elem.get("runtime_id", "")

    try:
        bounds = get_window_bounds(hwnd)
    except Exception:
        bounds = None
    if not bounds:
        return None

    confidence = min(best_score, 0.95)

    if targeted_entry:
        step = {
            "action": "click",
            "x": cx,
            "y": cy,
            "description": f"Focus '{label}'",
            "space": screen_geometry.COORD_SCREEN_PIXELS,
        }
        if uid:
            step["ui_uid"] = uid
        if runtime_id:
            step["ui_runtime_id"] = runtime_id
        steps = [step, {"action": "type", "text": targeted_entry["text"]}]
        # F21: a local match knows the target element, so a sensitive field can
        # be marked here with real evidence (name/control type/automation id).
        if best_elem.get("is_password") or tool_policy.is_sensitive_field(
                label, best_elem.get("name"), best_elem.get("control_type"),
                best_elem.get("automation_id"), best_elem.get("className")):
            steps[1]["sensitive"] = True
        summary = f"Typing into '{label}', sir."
    else:
        action = _extract_action_verb(command_text)
        action_label = action.replace("_", " ").title()
        step = {
            "action": action,
            "x": cx,
            "y": cy,
            "description": f"{action_label} on '{label}'",
            "space": screen_geometry.COORD_SCREEN_PIXELS,
        }
        if uid:
            step["ui_uid"] = uid
        if runtime_id:
            step["ui_runtime_id"] = runtime_id
        steps = [step]
        summary = f"{action_label}ing '{label}', sir."

    return {
        "ok": True,
        "confidence": confidence,
        "summary": summary,
        "needs_confirmation": False,
        "reason": "",
        "capture_mode": "active_window",
        "window_title": bounds.get("window_title", ""),
        "hwnd": int(hwnd),
        "process_id": bounds.get("process_id", 0),
        "origin_left": bounds.get("left"),
        "origin_top": bounds.get("top"),
        "capture_width": bounds.get("width"),
        "capture_height": bounds.get("height"),
        "vision_width": bounds.get("width"),
        "vision_height": bounds.get("height"),
        "steps": steps,
        "source": "UIA First",
    }


def _plan_with_vision(command_text):
    """Backward-compatible alias for the old planner name."""
    return _plan_with_tree(command_text)


def _verify_action(steps_description):
    """Backward-compatible wrapper for older callers."""
    return _verify_action_with_steps(steps_description, steps=None)


def _cloud_verification_enabled():
    return os.getenv(CLOUD_VERIFY_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _typed_texts_from_steps(steps):
    texts = []
    for step in steps or []:
        if step.get("action") != "type":
            continue
        text = str(step.get("text") or "").strip()
        if text:
            texts.append(text)
    return texts


def _verification_text_haystack(capture):
    parts = []

    if screen_ui_elements.is_available():
        try:
            for elem in screen_ui_elements.get_foreground_window_elements(
                max_elements=80,
                max_depth=6,
            ):
                parts.extend(
                    value
                    for value in (
                        elem.get("name"),
                        elem.get("automation_id"),
                        elem.get("class_name"),
                    )
                    if value
                )
        except Exception as exc:
            logging.debug("Local UI verification scan failed: %s", exc)

    raw_image = capture.get("raw_vision_image")
    if raw_image and screen_ocr.is_available():
        try:
            regions = screen_ocr.extract_text_regions(raw_image)
            merged = screen_ocr._merge_nearby_words(regions)
            parts.extend(region.get("text", "") for region in merged)
        except Exception as exc:
            logging.debug("Local OCR verification scan failed: %s", exc)

    return re.sub(r"\s+", " ", " ".join(parts).lower()).strip()


def _typed_targets(steps):
    """F41: pair every type step with the identity of the field it typed into.

    A type step usually carries no identity of its own — the plan clicks the
    field first. Inheriting the nearest preceding click target is what makes
    "did the text land in the INTENDED field?" answerable at all.
    """
    typed = []
    last_target = {"ui_uid": "", "ui_runtime_id": ""}
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        action = (step.get("action") or "").lower()
        if action in {"click", "double_click", "right_click"}:
            last_target = {
                "ui_uid": step.get("ui_uid") or "",
                "ui_runtime_id": step.get("ui_runtime_id") or "",
            }
        elif action == "type":
            text = str(step.get("text") or "")
            if not text.strip():
                continue
            typed.append({
                "text": text,
                "ui_uid": step.get("ui_uid") or last_target["ui_uid"],
                "ui_runtime_id": (
                    step.get("ui_runtime_id") or last_target["ui_runtime_id"]
                ),
            })
    return typed


def _read_typed_field_values(typed, hwnd=None):
    """F41: read back the field each type step wrote into.

    Returns a list of ``(intended_text, observed_value)``, or **None** when the
    values could not be observed at all. None means uncertainty, and uncertainty
    must never be reported as verified success.
    """
    if not typed:
        return []
    if not screen_ui_elements.is_available():
        return None
    try:
        controls = screen_ui_elements.read_control_values(hwnd=hwnd)
    except Exception as exc:
        logging.debug("F41 typed-field read failed: %s", exc)
        return None
    if not controls:
        return None

    focused = next((c for c in controls if c.get("focused")), None)
    by_runtime = {
        c["runtime_id"]: c for c in controls if c.get("runtime_id")
    }
    by_uid = {c["uid"]: c for c in controls if c.get("uid")}

    observed = []
    for target in typed:
        control = None
        runtime_id = target.get("ui_runtime_id") or ""
        if runtime_id:
            control = by_runtime.get(runtime_id)
        if control is None and target.get("ui_uid"):
            control = by_uid.get(target["ui_uid"])
        if control is None:
            # No identity survived planning: the keyboard went to whatever
            # held focus, so that is the only field we can honestly inspect.
            control = focused
        if control is None or control.get("value") is None:
            return None
        observed.append((target["text"], str(control["value"])))
    return observed


def _verify_action_locally(steps, capture=None, hwnd=None):
    """F41: verify the value of the field the plan actually typed into.

    The old check searched the whole window's text, so text that landed in the
    WRONG field still "verified". Only the intended field's own value counts
    now, and an unreadable field yields uncertainty rather than success.
    """
    typed = _typed_targets(steps)
    if not typed:
        # No text postcondition exists locally for a click/scroll plan.
        return None

    observed = _read_typed_field_values(typed, hwnd=hwnd)
    if observed is None:
        return None

    for intended, actual in observed:
        needle = re.sub(r"\s+", " ", intended.lower()).strip()
        hay = re.sub(r"\s+", " ", actual.lower()).strip()
        if needle and needle not in hay:
            return False
    return True


def _verify_action_with_steps(steps_description, steps=None, hwnd=None):
    """Verify action state locally first; cloud vision is opt-in."""
    need_capture = bool(_typed_texts_from_steps(steps)) or _cloud_verification_enabled()

    if need_capture:
        try:
            time.sleep(0.25)  # let the UI settle
            capture = capture_active_window()
        except RuntimeError:
            capture = None
        local_result = _verify_action_locally(steps or [], capture, hwnd=hwnd)
        if local_result is not None:
            return local_result
        if capture is None:
            return None
    elif not _cloud_verification_enabled():
        return None

    if not _cloud_verification_enabled():
        return None

    prompt = (
        "You just executed a screen action on this desktop. "
        f"The action was: {steps_description}\n"
        "Look at the screenshot and determine whether the action appears to have succeeded. "
        "Signs of success include: the expected element changing state, a dialog appearing, "
        "text being entered, a new page loading, or any visible change matching the intent.\n"
        "Return a JSON object with two keys:\n"
        '  verified (boolean) - true if the action likely succeeded\n'
        '  observation (string) - one-sentence description of what you see\n'
    )

    # F37: verification uses the SAME eligible-provider cascade as planning —
    # the selected provider first, then every other ELIGIBLE provider, each at
    # most once, with adapter exceptions and malformed output advancing
    # instead of ending the check.
    result = _ask_vision_cascade(
        prompt,
        capture["image_data_url"],
        max_completion_tokens=2048,
    )
    content = _extract_response_content(result)
    if content is None:
        return None

    parsed = _extract_json_object(content)
    return parsed.get("verified") if isinstance(parsed.get("verified"), bool) else None


_LOG_MAX_BYTES = 2 * 1024 * 1024  # 2MB cap, then rotate


def _write_screen_command_log(command, plan, status, error=None, verified=None, raw_vision=None):
    """Write an audit entry to screen_commands.log."""
    try:
        from backend.config import BASE_DIR
        log_file = BASE_DIR / "screen_commands.log"

        # Rotate when the log grows past the cap: rename to .log.1 (keep 1 backup)
        try:
            if log_file.exists() and log_file.stat().st_size > _LOG_MAX_BYTES:
                backup = log_file.with_suffix(".log.1")
                if backup.exists():
                    backup.unlink()
                log_file.rename(backup)
        except OSError:
            pass  # rotation is best-effort; never block logging on it

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # F21: this log persists typed text. Secrets and credential-shaped
        # strings are scrubbed before anything is written to disk.
        steps_str = tool_policy.mask_secrets(
            json.dumps(tool_policy.scrub_mapping(plan.get("steps", [])), indent=2))
        capture_mode = plan.get("capture_mode", "unknown")
        window_title = plan.get("window_title", "")
        confidence = plan.get("confidence", 0.0)
        source = plan.get("source", "AI Vision" if plan.get("ok") else "Failed")
        
        entry = (
            f"==================================================\n"
            f"[{timestamp}] SCREEN COMMAND: \"{command}\"\n"
            f"--------------------------------------------------\n"
            f"Status:      {status}\n"
            f"Source:      {source}\n"
            f"Confidence:  {confidence:.2f}\n"
            f"CaptureMode: {capture_mode}\n"
            f"Window:      \"{window_title}\"\n"
        )
        provider = plan.get("vision_provider") or plan.get("vision_model") or plan.get("model")
        if provider:
            entry += f"Provider:    {provider}\n"
        # P1-D: scale audit - EXECUTED lines include vision/origin/scale for one-glance diagnosis
        if status in ("EXECUTED", "CONFIRMED_EXECUTED"):
            vw = plan.get("vision_width")
            vh = plan.get("vision_height")
            cw = plan.get("capture_width")
            ch = plan.get("capture_height")
            ol = plan.get("origin_left")
            ot = plan.get("origin_top")
            if vw and vh:
                entry += f"Vision:      {vw}x{vh}\n"
            if ol is not None and ot is not None:
                entry += f"Origin:      {ol},{ot}\n"
            if cw and ch and vw and vh:
                try:
                    x_scale = cw / float(vw)
                    y_scale = ch / float(vh)
                    if abs(x_scale - y_scale) < 0.0005:
                        entry += f"Scale:       {x_scale:.4f}\n"
                    else:
                        entry += f"Scale:       x={x_scale:.4f} y={y_scale:.4f}\n"
                    entry += f"Capture:     {cw}x{ch}\n"
                except Exception:
                    pass
        if error:
            entry += f"Error:       {error}\n"
        if verified is not None:
            entry += f"Verified:    {verified}\n"
        # P1.3 raw vision logging - persisted truncated to 2000 chars for debuggability
        raw = raw_vision if raw_vision is not None else plan.get("raw_vision")
        if raw:
            raw_str = str(raw)
            if len(raw_str) > 2000:
                raw_str = raw_str[:2000]
            entry += f"RawVision:  {raw_str}\n"
            
        entry += f"Steps:\n{steps_str}\n"
        entry += "==================================================\n\n"

        # F21: one redacted egress boundary for the WHOLE record — the command
        # text, the window title, the raw vision payload and the error string
        # used to be written verbatim, so a secret that reached any of them
        # landed on disk even though the steps were scrubbed.
        entry = tool_policy.redact_for_egress(entry)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(entry)
    except Exception as exc:
        logging.warning("Failed to write screen command log: %s", exc)


def _maybe_adjust_for_window_move(plan):
    """P1-C / F42: re-acquire the PLANNED window and refuse an altered target.

    Translating stored coordinates is only safe when the target is still the
    *same* window, in the same process, at the same size:

      * a **moved** window is translated by the origin delta;
      * a **closed/minimised** window is refused;
      * a **replaced** window (HWND reused by a different process) is refused;
      * a **resized** window is refused, because the coordinates were planned
        against a different geometry and must be re-planned.

    The bounds are read from the plan's own HWND — never from "whatever is
    foreground now", which is how a click could previously be translated with
    window A's coordinates using window B's origin.
    """
    if plan.get("capture_mode") != "active_window":
        return None
    orig_left = plan.get("origin_left")
    orig_top = plan.get("origin_top")
    if orig_left is None or orig_top is None:
        return None
    planned_hwnd = plan.get("hwnd")
    if not planned_hwnd:
        return ("I don't have a verified target window for that action, sir - "
                "please ask me again so I can re-plan it safely.")
    try:
        from backend.services import screen_capture

        new_bounds = screen_capture.get_window_bounds(planned_hwnd)
    except Exception:
        new_bounds = None
    if not new_bounds:
        return "The window I captured seems to have closed or minimized, sir - please bring it back and try again."

    # F42: process identity — an HWND can be recycled after a window closes.
    expected_pid = plan.get("process_id")
    actual_pid = new_bounds.get("process_id")
    if expected_pid and actual_pid and int(expected_pid) != int(actual_pid):
        logging.warning(
            "Screen target HWND %s changed process (%s -> %s) — refusing.",
            planned_hwnd, expected_pid, actual_pid,
        )
        return ("The window I planned against has been replaced by a different "
                "program, sir - I've stopped instead of clicking blind. Please "
                "ask me again.")

    cap_w = plan.get("capture_width")
    cap_h = plan.get("capture_height")
    if cap_w and cap_h:
        try:
            resized = (int(new_bounds["width"]) != int(cap_w)
                       or int(new_bounds["height"]) != int(cap_h))
        except Exception:
            resized = False
        if resized:
            logging.info(
                "Screen target HWND %s resized (%sx%s -> %sx%s) — refusing "
                "stale coordinates.", planned_hwnd, cap_w, cap_h,
                new_bounds.get("width"), new_bounds.get("height"),
            )
            return ("The window changed size after I planned that action, sir - "
                    "I've stopped rather than click at stale coordinates. "
                    "Please ask me again.")

    delta_x = new_bounds["left"] - orig_left
    delta_y = new_bounds["top"] - orig_top
    if delta_x != 0 or delta_y != 0:
        for step in plan.get("steps", []) or []:
            if "x" in step and "y" in step:
                try:
                    step["x"] = int(step["x"]) + delta_x
                    step["y"] = int(step["y"]) + delta_y
                except Exception:
                    continue
        plan["origin_left"] = new_bounds["left"]
        plan["origin_top"] = new_bounds["top"]
        logging.info("Window moved by (%d,%d) - adjusted click coordinates", delta_x, delta_y)
    return None


def _plan_gate(plan):
    """F19: an effect boundary bound to the plan's captured generation.

    Passing this into :func:`screen_executor.execute_steps` makes revocation
    authoritative *inside* execution: focus acquisition and every step are
    re-checked against the generation the request was created under, and the
    check is serialized against :func:`screen_state.set_enabled`.
    """
    stamp = plan.get("screen_generation")
    return lambda: screen_state.effect_gate(stamp)


def _execute_or_queue(plan, command_text):
    # F19-enforcement: the control boundary is checked HERE, at the point of
    # effect, not only where the request arrived. A plan produced before the
    # user turned screen control off must never reach execute_steps.
    allowed, gate_reason = screen_state.guard(plan.get("screen_generation"))
    if not allowed:
        reason = ("Screen controls are off — I've dropped that action, sir."
                  if gate_reason == "screen controls are off"
                  else "That action was planned before screen controls were "
                       "restarted, sir. Please ask again.")
        _write_screen_command_log(command_text, plan, "BLOCKED_SCREEN_DISABLED",
                                  error=gate_reason)
        return reason

    if not plan.get("ok"):
        reason = plan.get("reason") or "I couldn't complete that screen command, sir."
        _write_screen_command_log(command_text, plan, "FAILED_PLANNING", error=reason)
        return reason

    confidence = _coerce_float(plan.get("confidence"), 0.0)
    confirmation_reason = None

    if confidence < MIN_ACTIONABLE_CONFIDENCE:
        reason = plan.get("reason") or "I found a possible target, but not confidently enough to click it."
        _write_screen_command_log(command_text, plan, "LOW_CONFIDENCE_BLOCKED", error=reason)
        return reason

    if _is_risky_command(command_text):
        plan["needs_confirmation"] = True
        confirmation_reason = "risky"
    elif confidence < LOW_CONFIDENCE_CONFIRMATION:
        plan["needs_confirmation"] = True
        confirmation_reason = "low_confidence"
    elif plan.get("needs_confirmation"):
        confirmation_reason = "low_confidence"

    if plan.get("needs_confirmation"):
        plan["command_text"] = command_text
        # F18: stamp the control generation at the moment the plan is queued
        # for consent, so disabling (and re-enabling) invalidates it, and
        # store a plan-hash approval record covering EVERY effect.
        # F19: ``setdefault`` — a plan that already captured the generation at
        # request creation keeps that token, so a revocation during planning
        # cannot be laundered into fresh authority by re-stamping it here.
        plan.setdefault("screen_generation", screen_state.generation())
        screen_state.set_pending_plan(plan, command_text)
        _write_screen_command_log(command_text, plan, "PENDING_CONFIRMATION")
        spoken = screen_state.pending_spoken()
        if spoken:
            # F18: the spoken summary names the command and the action counts
            # — the full ordered preview travels with the approval record for
            # the UI to render.
            return spoken
        if confirmation_reason == "low_confidence":
            return (
                "I found a likely target, but confidence is limited. "
                "Say confirm screen action to proceed, or say cancel."
            )
        return "That action looks risky. Say confirm screen action to proceed, or say cancel."

    # Immediate execution path: re-check the gate right before the effect —
    # the confirmation delay is exactly the window this protects.
    allowed, gate_reason = screen_state.guard(plan.get("screen_generation"))
    if not allowed:
        _write_screen_command_log(command_text, plan, "BLOCKED_SCREEN_DISABLED",
                                  error=gate_reason)
        return ("Screen controls are off — I've dropped that action, sir."
                if gate_reason == "screen controls are off" else
                "That action was planned before screen controls were "
                "restarted, sir. Please ask again.")

    target_hwnd = plan.get("hwnd")
    race_error = _maybe_adjust_for_window_move(plan)
    if race_error:
        _write_screen_command_log(command_text, plan, "EXECUTION_FAILED", error=race_error)
        return race_error
    error = None
    try:
        execute_steps(plan["steps"], target_hwnd=target_hwnd, gate=_plan_gate(plan))
    except RuntimeError as exc:
        error = str(exc)
        _write_screen_command_log(command_text, plan, "EXECUTION_FAILED", error=error)
        return error
    screen_state.clear_pending_plan()

    summary = plan.get("summary") or "Done, sir."

    # F41: EVERY executed plan is checked, including a single confirmed
    # click or type. The old `len(steps) <= 2` skip meant the most common
    # actions were reported as done without any postcondition at all.
    verified = None
    steps = plan.get("steps") or []
    try:
        verified = _verify_action_with_steps(
            summary, steps=steps, hwnd=plan.get("hwnd"),
        )
    except Exception as exc:
        logging.warning("Post-action verification failed: %s", exc)

    # Record in context memory
    screen_state.add_interaction(command_text, summary, verified=verified, plan=plan)
    _write_screen_command_log(command_text, plan, "EXECUTED", verified=verified)

    if verified is False:
        return f"{summary} Though I'm not fully sure it worked - want me to try again?"
    return summary


_MENU_OPEN_HINTS = re.compile(
    r"\b(menu|dropdown|file|edit|view|tools|options|context)\b",
    re.IGNORECASE,
)


def _step_likely_opens_menu(step):
    """Heuristic: does this click step probably open a menu or dropdown?"""
    desc = step.get("description", "") + " " + step.get("label", "")
    return step.get("action") == "click" and bool(_MENU_OPEN_HINTS.search(desc))


_RETRY_RE = re.compile(
    r"\b(try again|repeat that|redo that|do (?:it|that) again)\b",
    re.IGNORECASE,
)


# ---- in-progress guard for vision commands ----
_vision_busy = threading.Lock()
_response_callback = None


def set_response_callback(callback):
    """Register a callback(text) that delivers async screen-control results.

    The brain thread sets this so background vision results can be spoken.
    """
    global _response_callback
    _response_callback = callback


def maybe_handle_screen_control_message(text):
    # F19: exact, negation-aware enable/disable commands are resolved BEFORE
    # anything else — including pending-consent consumption. "Turn screen
    # controls off" while an approval is pending must revoke control (which
    # also drops the approval), never be read as an approval verdict and run
    # the very plan the user is trying to cancel.
    toggle = _exact_toggle_verdict(text)
    if toggle == "negated_on":
        if screen_state.is_enabled():
            return "Understood, sir — I'll leave screen controls as they are."
        return "Screen controls stay off, sir — I won't enable them."
    if toggle == "on":
        if screen_state.is_enabled():
            return "Screen controls are already on, sir."
        screen_state.set_enabled(True)
        screen_state.clear_interactions()
        return "Screen controls are on, sir."
    if toggle == "off":
        if not screen_state.is_enabled():
            return "Screen controls are already off, sir."
        # Disabling bumps the control generation and drops any pending plan /
        # approval, so an in-flight multi-step plan cannot continue.
        screen_state.set_enabled(False)
        screen_state.clear_interactions()
        return "Screen controls are off, sir."

    if screen_state.has_pending_plan():
        # F18: resolve the verdict with negation FIRST — "no, don't do it"
        # must never be read as "do it". The approval is also bound to the
        # whole plan (hash), expires, and dies with the control generation.
        answer = approvals.verdict(text)
        if answer is None and _is_confirmation(text):
            answer = approvals._YES
        if answer is None and _is_cancel(text):
            answer = approvals._NO
        if answer == approvals._YES:
            # F18: take the plan AND its approval atomically, then verify the
            # snapshot. A plan whose coordinates changed (or whose window moved)
            # after verification cannot slip through, because the consumed
            # snapshot is the only thing that can execute — and if verification
            # fails, the record is already gone, so NOTHING runs.
            plan, record = screen_state.consume_pending_plan()
            if record is None or plan is None:
                screen_state.clear_pending_plan()
                return "That screen action expired, sir. Please ask again."
            cmd_text = plan.get("command_text", "pending command")
            ok, why = approvals.verify(plan, record, cmd_text,
                                       generation=screen_state.generation())
            if not ok:
                _write_screen_command_log(cmd_text, plan, "APPROVAL_INVALID", error=why)
                return "That approval no longer applies (%s), sir. Please ask again." % why
            allowed, gate_reason = screen_state.guard(plan.get("screen_generation"))
            if not allowed:
                _write_screen_command_log(cmd_text, plan, "BLOCKED_SCREEN_DISABLED",
                                          error=gate_reason)
                return ("Screen controls are off, sir — I've dropped that action."
                        if gate_reason == "screen controls are off" else
                        "Screen controls were restarted, sir — please ask again.")
            target_hwnd = plan.get("hwnd")
            race_error = _maybe_adjust_for_window_move(plan)
            if race_error:
                _write_screen_command_log(cmd_text, plan, "CONFIRMED_EXECUTION_FAILED", error=race_error)
                return race_error
            error = None
            try:
                execute_steps(plan["steps"], target_hwnd=target_hwnd,
                              gate=_plan_gate(plan))
            except RuntimeError as exc:
                error = str(exc)
                _write_screen_command_log(cmd_text, plan, "CONFIRMED_EXECUTION_FAILED", error=error)
                return error
            summary = plan.get("summary") or "Done, sir."
            screen_state.add_interaction("confirm", summary, plan=plan)
            _write_screen_command_log(cmd_text, plan, "CONFIRMED_EXECUTED")
            return summary
        if answer == approvals._NO:
            plan = screen_state.pop_pending_plan()
            screen_state.clear_pending_plan()
            _write_screen_command_log(plan.get("command_text", "pending command") if plan else "pending command", plan or {}, "CANCELLED")
            return "Cancelled the pending screen action, sir."

    # Bug #10: "Try again" / "repeat that" support.
    if screen_state.is_enabled() and _RETRY_RE.search(_normalize_text(text)):
        recent = screen_state.get_recent_interactions()
        if recent and recent[-1].get("plan"):
            age = time.time() - recent[-1].get("ts", 0)
            if age > 30:
                # Plan is stale — re-plan from scratch. F19: capture the
                # generation *before* planning so a revocation during this
                # re-plan cannot be adopted by the finished plan.
                retry_generation = screen_state.generation()
                retry_plan = _plan_with_tree(recent[-1]["command"])
                if isinstance(retry_plan, dict):
                    retry_plan["screen_generation"] = retry_generation
                return _execute_or_queue(retry_plan, recent[-1]["command"])
            return _execute_or_queue(recent[-1]["plan"], recent[-1]["command"])
        return "I don't have a recent screen action to repeat, sir."

    if not _looks_like_screen_command(text):
        return None

    if not screen_state.is_enabled():
        return "Screen controls are off. Say turn on screen controls first."

    command_text = _strip_leading_fillers(text)

    # F19: capture the control generation at REQUEST CREATION, before any
    # planning work happens. A plan carries the authority that existed when
    # the user asked; an enable/disable cycle during planning (which can take
    # seconds) therefore invalidates the plan instead of letting it adopt the
    # new authority when it finally reaches the gate.
    request_generation = screen_state.generation()

    # Direct plans (hotkey-based) are instant -- run synchronously.
    direct_plan = _build_direct_plan(command_text)
    if direct_plan:
        direct_plan["screen_generation"] = request_generation
        return _execute_or_queue(direct_plan, command_text)

    # Vision plans are slow -- guard against duplicates and run in background.
    if not _vision_busy.acquire(blocking=False):
        return "Still working on the last screen command, sir."

    def _bg_vision():
        try:
            vision_plan = _plan_with_tree(command_text)
            if isinstance(vision_plan, dict):
                # Pin the plan to the generation captured before planning.
                vision_plan["screen_generation"] = request_generation
            result = _execute_or_queue(vision_plan, command_text)
            if _response_callback:
                _response_callback(result)
            else:
                print("[SCREEN] (no callback) Result:", result)
        except Exception as exc:
            logging.warning("Background vision failed: %s", exc)
            if _response_callback:
                _response_callback("Something went wrong with the screen command, sir.")
        finally:
            _vision_busy.release()

    threading.Thread(target=_bg_vision, daemon=True).start()
    return "Working on it, sir."
