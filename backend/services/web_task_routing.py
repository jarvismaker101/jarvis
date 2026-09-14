"""Web-shaped multi-step task detection for the browser-task routing net.

A message is web-shaped when it names a web/browser target AND an in-page
interaction verb, and carries no local-machine hint. Used by the brain's
is_task_request branch to hand browser automation to the confirmation-gated
browser agent instead of the raw screen-action planner.
"""

import re

_WEB_HINT_RE = re.compile(
    r"browser|website|web ?site|webpage|web page|site|chrome|edge|brave|"
    r"firefox|\.com|\.org|\.net|\.io|\.to\b|http",
    re.IGNORECASE,
)

_INTERACTION_VERB_RE = re.compile(
    r"search|play|click|login|log in|download|fill|type|scroll|watch|automate",
    re.IGNORECASE,
)

_LOCAL_HINT_RE = re.compile(
    r"file|folder|workspace|vs ?code|vscode|cursor|windsurf|editor|code|"
    r"install|python|script|project|setting|config|terminal|shell",
    re.IGNORECASE,
)


def is_web_shaped_task(text):
    """True for multi-step web/browser interaction, False for local tasks."""
    text = text or ""
    return bool(
        _WEB_HINT_RE.search(text)
        and _INTERACTION_VERB_RE.search(text)
        and not _LOCAL_HINT_RE.search(text)
    )
