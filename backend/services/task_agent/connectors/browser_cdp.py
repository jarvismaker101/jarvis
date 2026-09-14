"""Browser connector using Chrome DevTools Protocol discovery when available.

This is intentionally dependency-light. If Chrome/Brave/Edge is started with
`--remote-debugging-port=9222`, Jarvis can see open tabs through CDP. Actions
fall back to the existing browser executor until a WebSocket CDP action layer is
installed.

G6/F47: tab identity is no longer this connector's private business. Every
discovery pass publishes the REAL tab identities (id + webSocketDebuggerUrl,
which used to be reduced to a boolean) to the browser-session broker, and
`open_url` asks the broker how the URL should reach a browser instead of
blindly launching the default one — a URL that is already open, or a tab the
caller pointed at, is attached to rather than duplicated in another profile.

F47 hardening:
  * discovery is published in FULL (the 8-tab UI display limit no longer hides
    later tabs from every other subsystem);
  * the ws handle is a string end to end — a boolean was being stringified
    into the literal "True";
  * "attach"/"focus" perform a REAL activation (the CDP HTTP control endpoint)
    instead of only returning a sentence;
  * an explicitly named tab that cannot be reached is a REFUSAL — it never
    falls through to launching a different browser/profile;
  * several matching tabs are reported as ambiguous instead of silently
    picking the first one.
"""

import os
import urllib.parse

import requests

from backend.core.executor import execute_multiple, open_in_browser
from backend.services import browser_session_broker


CDP_URL = os.getenv("JARVIS_CDP_URL", "http://127.0.0.1:9222").rstrip("/")

# F47: one stable, pinned session id for this discovery endpoint.
_BROKER_SESSION_ID = "cdp-discovery"

#: How many tabs the DISPLAY list returns. Discovery itself is unlimited — the
#: UI limit must not decide which tabs the rest of the system can address.
DEFAULT_DISPLAY_LIMIT = 8

_DISCOVERY_TIMEOUT_S = 0.8


class CdpUnavailable(RuntimeError):
    """The CDP HTTP endpoint did not answer — nothing may be assumed."""


def _get(path, timeout=_DISCOVERY_TIMEOUT_S, method="get"):
    """One CDP HTTP control call. Raises CdpUnavailable on any failure."""
    url = "%s/%s" % (CDP_URL, path.lstrip("/"))
    try:
        response = getattr(requests, method)(url, timeout=timeout)
    except Exception as exc:
        raise CdpUnavailable(str(exc))
    if response.status_code >= 400:
        raise CdpUnavailable("HTTP %s from %s" % (response.status_code, url))
    return response


def _norm_tab(tab):
    """One /json target -> the broker/display shape (identities intact)."""
    if not isinstance(tab, dict):
        return None
    handle = tab.get("webSocketDebuggerUrl")
    handle = handle if isinstance(handle, str) else ""
    return {
        "id": tab.get("id", ""),
        "type": tab.get("type", ""),
        "title": tab.get("title", ""),
        "url": tab.get("url", ""),
        # F47: keep the boolean for old consumers, but never throw the actual
        # handle away — and never publish a stringified boolean as one.
        "webSocketDebuggerUrl": bool(handle),
        "debugger_url": handle,
        "ws_url": handle,
        "identity_source": "structured" if tab.get("id") else "derived",
    }


def discover_tabs():
    """EVERY page target, with real identities (F47: discovery is unlimited).

    Raises :class:`CdpUnavailable` when the endpoint cannot be read, so a
    caller can tell "no tabs" from "could not look".
    """
    response = _get("json")
    try:
        payload = response.json()
    except Exception as exc:
        raise CdpUnavailable("unparseable /json payload: %s" % exc)
    if not isinstance(payload, list):
        raise CdpUnavailable("/json did not return a target list")
    tabs = []
    for tab in payload:
        normalised = _norm_tab(tab)
        if normalised and normalised["id"]:
            tabs.append(normalised)
    return tabs


def _publish_tabs(tabs):
    """F47: hand the broker the real identities, ws URLs included."""
    try:
        browser_session_broker.register_session(
            owner="cdp",
            endpoint=CDP_URL,
            kind="cdp-discovery",
            label="CDP discovery endpoint",
            session_id=_BROKER_SESSION_ID,
        )
        browser_session_broker.publish_tabs(_BROKER_SESSION_ID, [
            {
                "tab_id": tab.get("id"),
                "url": tab.get("url"),
                "title": tab.get("title"),
                # The real ws handle (string), never a boolean.
                "ws_url": tab.get("debugger_url") or tab.get("ws_url") or "",
                "identity_source": "structured",
            }
            for tab in tabs
        ])
    except Exception:
        pass


def is_available():
    try:
        response = requests.get(f"{CDP_URL}/json/version", timeout=0.5)
        return response.status_code == 200
    except Exception:
        return False


def list_tabs(limit=DEFAULT_DISPLAY_LIMIT):
    """The tabs to DISPLAY (up to *limit*), while publishing ALL of them.

    F47: the old version truncated the list to ``limit`` and published the
    truncated copy, so every tab after the eighth was invisible to the whole
    system — "all tabs remain addressable" cannot hold that way.
    """
    try:
        tabs = discover_tabs()
    except Exception:
        return []
    _publish_tabs(tabs)
    if limit is None:
        return tabs
    try:
        limit = int(limit)
    except Exception:
        limit = DEFAULT_DISPLAY_LIMIT
    if limit <= 0:
        return []
    return tabs[:limit]


def snapshot():
    available = is_available()
    return {
        "connector": "browser_cdp",
        "available": available,
        "endpoint": CDP_URL,
        "tabs": list_tabs() if available else [],
        "capabilities": [
            "browser.search_web",
            "browser.open_url",
            "browser.inspect_tabs" if available else "browser.launch_required_for_dom",
        ],
        "setup_hint": (
            ""
            if available
            else "Start Chrome/Brave/Edge with --remote-debugging-port=9222 for DOM-level browser control."
        ),
    }


def search_web(query, browser=None):
    execute_multiple([{"action": "search", "input": query, "browser": browser}])
    return f"Opened a web search for {query}."


def activate_tab(tab_id):
    """Bring a real CDP tab to the front (F47: attach/focus must DO something).

    The old branches only returned a sentence, so "attached to that tab" was
    a claim with no effect at all. Returns a result dict; never raises.
    """
    tab_id = str(tab_id or "").strip()
    if not tab_id:
        return {"activated": False, "reason": "no tab id"}
    try:
        # Chrome's HTTP control endpoint activates a target by its id.
        _get("json/activate/%s" % urllib.parse.quote(tab_id, safe=""), method="get")
        return {"activated": True, "tab_id": tab_id}
    except Exception as exc:
        return {"activated": False, "tab_id": tab_id, "reason": str(exc)}


def open_tab(url):
    """Open *url* as a NEW real CDP tab. Never raises.

    The new target's own identity is returned when the endpoint describes it,
    so the tab is immediately addressable by every other subsystem instead of
    existing only as a sentence (F47).
    """
    target = str(url or "").strip()
    if not target:
        return {"opened": False, "reason": "no url"}
    try:
        response = requests.put(
            "%s/json/new?%s" % (CDP_URL, urllib.parse.quote(target, safe="")),
            timeout=_DISCOVERY_TIMEOUT_S,
        )
        if response.status_code >= 400:
            return {"opened": False, "reason": "HTTP %s" % response.status_code}
        result = {"opened": True, "url": target}
        try:
            created = response.json()
        except Exception:
            created = None
        if isinstance(created, dict):
            handle = created.get("webSocketDebuggerUrl")
            result["tab_id"] = str(created.get("id") or "")
            result["ws_url"] = handle if isinstance(handle, str) else ""
            result["title"] = str(created.get("title") or "")
            if created.get("url"):
                result["url"] = str(created["url"])
        return result
    except Exception as exc:
        return {"opened": False, "reason": str(exc)}


def _live_cdp_session():
    """The broker's record for THIS endpoint, when it is live (F47)."""
    try:
        session = browser_session_broker.get_session(_BROKER_SESSION_ID)
    except Exception:
        return None
    if session and session.get("live"):
        return session
    return None


def _publish_opened_tab(result, fallback_url):
    """Publish a newly opened tab's real identity (F47: typed publication)."""
    tab_id = str(result.get("tab_id") or "").strip()
    if not tab_id:
        return
    try:
        browser_session_broker.publish_tabs(_BROKER_SESSION_ID, [{
            "tab_id": tab_id,
            "url": str(result.get("url") or fallback_url),
            "title": str(result.get("title") or ""),
            "ws_url": str(result.get("ws_url") or ""),
            "identity_source": "structured",
        }], replace=False)
    except Exception:
        pass


def open_url(url, prefer_tab=None, browser=None):
    """Open *url* in the browser the user actually means (F47).

    *prefer_tab* names a specific tab (id, URL or origin): the broker attaches
    to exactly that tab, or refuses — never a silent substitution. Without a
    preference, an already-open copy is focused, the warm research browser
    gets a new page when it is live, and only as the last resort does the
    user's default browser get a fresh launch (which is then recorded).
    """
    target = (url or "").strip()
    if not target:
        return "No URL was provided."
    if not target.startswith(("http://", "https://")):
        if "." in target and " " not in target:
            target = "https://" + target
        else:
            target = "https://www.google.com/search?q=" + urllib.parse.quote_plus(target)

    try:
        route = browser_session_broker.route_open(target, prefer_tab=prefer_tab)
    except Exception:
        route = {"action": "external"}
    action = route.get("action")

    explicit = bool(str(prefer_tab or "").strip())

    if action == "error":
        # F47: an unknown tab is a refusal, not a fallback.
        return ("No open tab matches %r — not opening a different browser. "
                "List tabs first or open it explicitly." % (prefer_tab,))

    if action == "ambiguous":
        candidates = route.get("candidates") or []
        described = "; ".join(
            "%s (%s)" % (c.get("title") or c.get("url") or c.get("tab_id"),
                         c.get("qualified_id") or c.get("tab_id"))
            for c in candidates[:6])
        return ("%d open tabs match %r — not guessing which one. Candidates: %s. "
                "Tell me which tab you mean (by title or full id)."
                % (len(candidates), prefer_tab or target, described or "unknown"))

    if action == "attach":
        tab = route["tab"]
        result = activate_tab(tab.get("tab_id"))
        if not result.get("activated"):
            # F47: the explicit target could NOT be reached. Do not launch a
            # different browser behind the user's back.
            return ("Tab '%s' (%s) exists but could not be brought forward (%s) — "
                    "not opening a different browser or profile. Try again or "
                    "activate it yourself."
                    % (tab.get("title") or tab["tab_id"], tab.get("url"),
                       result.get("reason") or "activation failed"))
        return ("Attached to tab '%s' (%s) in the %s session — it is now the "
                "active tab." % (tab.get("title") or tab["tab_id"], tab.get("url"),
                                 route["session"].get("owner")))

    if action == "focus":
        tab = route["tab"]
        result = activate_tab(tab.get("tab_id"))
        state = "it is now the active tab" if result.get("activated") \
            else "I could not bring it forward: %s" % (result.get("reason") or "activation failed")
        return ("'%s' is already open (%s) in the %s session — %s instead of "
                "opening a second copy."
                % (tab.get("title") or tab.get("url"), tab.get("url"),
                   route["session"].get("owner"), state))

    if action == "research_page":
        opened = _open_in_research_browser(target)
        if opened:
            return ("Opened %s as a new page in the warm research browser "
                    "(same profile and authentication)." % target)
        # F47: the warm browser was chosen deliberately; failing over to a
        # cold default launch would silently move the page into another
        # profile with different authentication.
        return ("The warm research browser is running but would not open %s — "
                "not launching a second browser on a different profile." % target)

    if explicit:
        # Defensive: an explicit target must never reach the launch fallback.
        return ("Could not attach to the requested tab %r — not opening a "
                "different browser." % (prefer_tab,))

    # F47 "real navigation": when THIS endpoint's browser is known and live,
    # a new page belongs in THAT browser (same profile, same authentication)
    # as a real CDP tab — not in a cold default browser.
    if _live_cdp_session():
        opened = open_tab(target)
        if opened.get("opened"):
            _publish_opened_tab(opened, target)
            return ("Opened %s as a new tab of the CDP browser at %s (same "
                    "profile) — tab %s."
                    % (opened.get("url") or target, CDP_URL,
                       opened.get("tab_id") or "unknown"))
        # Reported, not silent: the endpoint is live but refused the page.
        open_in_browser(target, browser)
        browser_session_broker.note_external_launch(target, browser)
        return ("The CDP browser at %s would not open %s (%s), so it was opened "
                "in the default browser instead."
                % (CDP_URL, target, opened.get("reason") or "unknown reason"))

    open_in_browser(target, browser)
    browser_session_broker.note_external_launch(target, browser)
    return f"Opened {target}."


def _open_in_research_browser(url):
    """Best-effort new page on the warm research worker (F27/F47)."""
    try:
        from backend.services import research_browser

        def _goto(page):
            try:
                page.goto(url, timeout=45000, wait_until="domcontentloaded")
                return True
            except Exception:
                return False

        return bool(research_browser.submit(_goto, timeout=60.0))
    except Exception:
        return False
