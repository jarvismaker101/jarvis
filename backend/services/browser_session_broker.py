"""Fable-5 audit G6 / F47 — one owner of browser session identity.

Before this module every subsystem picked its own browser and none of them
knew about the others:

  * ``executor.open_in_browser`` launched chrome.exe/brave.exe/webbrowser on
    the user's DEFAULT profile and forgot about it immediately;
  * ``task_agent/connectors/browser_cdp.py`` ran its own CDP discovery
    endpoint (``http://127.0.0.1:9222/json``) and threw away the
    ``webSocketDebuggerUrl`` — the one handle that identifies a tab;
  * ``research_browser`` (G5/F27) owns the automation profile through one
    persistent Playwright context;
  * the MCP browser agent drives a daemon-owned browser whose tabs only the
    daemon knows.

So "open this in the browser" meant "open it in SOME browser", and a task
that referenced a tab the user was looking at could silently land in a
different profile with different authentication and page state — which is
F47's exact complaint.

The broker is the single place that answers "which browser, which profile,
which tab":

  * every browser-ish actor REGISTERS a session (owner, profile, channel,
    endpoint, kind);
  * a persistent PROFILE has exactly ONE owning LIVE session at a time — a
    second registrant is refused instead of silently starting a second
    browser on the same profile, and a claim held only by a DEAD record is
    reclaimed rather than honored (F47: a process-local record is not proof
    of exclusive live ownership);
  * tabs are published with real, instance-qualified identities (session id
    + tab id + URL + origin + the CDP ``webSocketDebuggerUrl`` when known,
    kept as a STRING — never a stringified boolean) and can be ATTACHED by
    id;
  * attaching an unknown tab REFUSES rather than opening something else, and
    an id/URL that names several live tabs reports the AMBIGUITY with the
    candidates instead of silently picking the first one;
  * ``route_open`` turns "open this URL" into an explicit decision (attach /
    focus / research page / external / ambiguous) so callers never guess.

No I/O happens here: the broker only holds identity. Every helper is
best-effort by design in the callers — a registry problem must never stop a
browser from opening.
"""

import hashlib
import itertools
import os
import threading
import time
from urllib.parse import urlsplit

#: Sessions that claim a profile_dir hold it exclusively while live.
_PROFILE_KINDS = frozenset(("persistent-context", "daemon"))

#: F47: how long a record whose owning process cannot be confirmed here
#: (a foreign pid, or no pid at all) may still authorize anything. Records
#: that carry THIS process's pid stay valid until they are unregistered.
_LIVENESS_TTL_S = 900.0

_session_ids = itertools.count(1)
_external_tab_ids = itertools.count(1)
_lock = threading.RLock()
_sessions = {}          # session_id -> session dict
_profile_owners = {}    # normalized profile -> session_id


class ProfileOwnershipError(RuntimeError):
    """Another live session already owns this persistent profile."""


class UnknownTabError(LookupError):
    """No known session has this tab — refuse instead of opening another browser."""


class AmbiguousTabError(LookupError):
    """Several live tabs match — refuse to guess, surface the candidates.

    F47: "duplicate targets resolve ambiguously". Resolving to the first
    match silently redirected a request to an arbitrary one of several tabs
    (the classic duplicate-URL case).
    """

    def __init__(self, message, candidates=None):
        LookupError.__init__(self, message)
        self.candidates = list(candidates or [])


def normalize_profile(profile_dir):
    """Instance identity of a profile directory (F47).

    Two ALIASES of the same directory must collide — case, ``/`` vs ``\\``,
    a trailing separator, ``..`` segments, a relative vs an absolute path and
    ``%VARS%``/``~`` expansion all name the SAME profile — and the same raw
    string naming two different directories must not be treated as one.
    """
    raw = str(profile_dir or "").strip()
    if not raw:
        return ""
    try:
        expanded = os.path.expandvars(os.path.expanduser(raw))
        if not os.path.isabs(expanded):
            expanded = os.path.join(os.getcwd(), expanded)
        return os.path.normcase(os.path.normpath(os.path.abspath(expanded)))
    except Exception:
        return raw.replace("\\", "/").rstrip("/").lower()


def origin_of(url):
    """scheme://host[:port] of *url* ("" when unparseable)."""
    try:
        parts = urlsplit((url or "").strip())
    except Exception:
        return ""
    if not parts.netloc:
        return ""
    return "%s://%s" % ((parts.scheme or "http").lower(), parts.netloc.lower())


def canonical_ws_url(value):
    """The CDP websocket handle as a STRING, or "".

    F47: ``webSocketDebuggerUrl`` used to be published as a boolean
    (``bool(...)``) and then re-stringified by ``str()``, so the broker's
    "handle" became the literal ``"True"`` — the one field that identifies a
    tab, destroyed. Anything that is not a real ws/http URL is no handle at
    all; a boolean is never coerced into one.
    """
    if isinstance(value, str):
        text = value.strip()
        if text.startswith(("ws://", "wss://", "http://", "https://")):
            return text
    return ""


def _instance_token(session_id):
    """Stable, instance-qualified prefix for tab ids of one session."""
    return hashlib.sha1(str(session_id or "").encode("utf-8", "replace")).hexdigest()[:8]


def qualified_tab_id(session_id, tab_id):
    """F47: ``<instance>:<tab id>`` — a tab id is only meaningful inside the
    browser instance that issued it. Same numbers in two browsers are two
    different tabs."""
    tab_id = str(tab_id or "").strip()
    if not tab_id:
        return ""
    return "%s:%s" % (_instance_token(session_id), tab_id)


def _session_live(session, now=None):
    """Is this record proof of a LIVE browser right now? (F47)

    A record only counts when its owning process is THIS process (a
    leftover record from another process proves nothing), it has not been
    retired, and — for records with no confirmed owner pid — it is still
    inside the liveness window.
    """
    if not session:
        return False
    if session.get("dead"):
        return False
    owner_pid = session.get("pid")
    if owner_pid is None:
        stamp = float(session.get("last_seen") or session.get("registered_at") or 0)
        now = time.time() if now is None else now
        return bool(_LIVENESS_TTL_S) and (now - stamp) <= _LIVENESS_TTL_S
    try:
        if int(owner_pid) != os.getpid():
            return False
    except Exception:
        return False
    return True


def is_live(session_id):
    """True when *session_id* names a live session."""
    with _lock:
        return _session_live(_sessions.get(session_id))


def heartbeat(session_id):
    """Refresh a session's liveness stamp (owners that stay alive call this)."""
    with _lock:
        session = _sessions.get(session_id)
        if session:
            session["last_seen"] = time.time()
            return True
    return False


def mark_dead(session_id):
    """Retire a session: its profile claim and tabs authorize nothing (F47)."""
    if not session_id:
        return False
    with _lock:
        session = _sessions.get(session_id)
        if not session:
            return False
        session["dead"] = True
        for key, owner_id in list(_profile_owners.items()):
            if owner_id == session_id:
                _profile_owners.pop(key, None)
        return True


def _new_session(owner, profile_dir=None, channel=None, endpoint=None,
                 kind="external-launch", label=None, pid=None, instance=None):
    return {
        "session_id": "bs-%d" % next(_session_ids),
        "owner": owner or "unknown",
        "profile_dir": str(profile_dir) if profile_dir else None,
        "channel": channel or None,
        "endpoint": endpoint or None,
        "kind": kind,
        "label": label or None,
        "instance": instance or None,
        "pid": os.getpid() if pid is None else pid,
        "registered_at": time.time(),
        "last_seen": time.time(),
        "dead": False,
        "tabs": {},          # tab_id -> tab dict
    }


def _profile_owner_session(profile_key, now=None):
    """The LIVE session holding *profile_key*, or None (dead claims dropped)."""
    owner_id = _profile_owners.get(profile_key)
    if not owner_id:
        return None
    session = _sessions.get(owner_id)
    if session and _session_live(session, now):
        return session
    # F47: a claim held by a dead/foreign record is not ownership.
    _profile_owners.pop(profile_key, None)
    return None


def register_session(owner, profile_dir=None, channel=None, endpoint=None,
                     kind="external-launch", label=None, session_id=None,
                     pid=None, instance=None):
    """Declare a browser session. Returns its session dict.

    Profile ownership (F47): a persistent profile is held by ONE LIVE
    session. Registering the same profile under a different owner raises
    :class:`ProfileOwnershipError` instead of letting two browsers fight
    over one profile directory. The same owner re-registering gets its
    existing session back (idempotent), and a claim left by a DEAD record is
    reclaimed rather than honored.

    *session_id* pins the id (e.g. ``"cdp-discovery"``, ``"external"``) so
    well-known actors have stable handles instead of minted ones. *pid* is
    the owning process (defaults to this one) — a record published by
    another process never authorizes anything here.
    """
    profile_key = normalize_profile(profile_dir) if profile_dir and kind in _PROFILE_KINDS else None
    with _lock:
        if session_id and session_id in _sessions:
            existing = _sessions[session_id]
            if profile_key:
                other = _profile_owner_session(profile_key)
                if other and other["session_id"] != session_id:
                    if other["owner"] != (owner or "unknown"):
                        raise ProfileOwnershipError(
                            "profile %s is owned by the %s session (%s) — attach "
                            "to it instead of starting a second browser on the "
                            "same profile" % (profile_dir, other["owner"],
                                              other["session_id"]))
                    _profile_owners[profile_key] = session_id
            existing["last_seen"] = time.time()
            existing["dead"] = False
            if endpoint:
                existing["endpoint"] = endpoint
            if instance:
                existing["instance"] = instance
            return existing
        if profile_key:
            existing = _profile_owner_session(profile_key)
            if existing:
                if existing["owner"] != (owner or "unknown"):
                    raise ProfileOwnershipError(
                        "profile %s is owned by the %s session (%s) — attach to it "
                        "instead of starting a second browser on the same profile"
                        % (profile_dir, existing["owner"], existing["session_id"]))
                existing["last_seen"] = time.time()
                return existing
        session = _new_session(owner, profile_dir, channel, endpoint, kind,
                               label, pid=pid, instance=instance)
        if session_id:
            session["session_id"] = session_id
        session["instance"] = session.get("instance") or session["session_id"]
        _sessions[session["session_id"]] = session
        if profile_key:
            _profile_owners[profile_key] = session["session_id"]
        return session


def unregister_session(session_id):
    """Release a session and its profile claim. Idempotent."""
    if not session_id:
        return
    with _lock:
        session = _sessions.pop(session_id, None)
        if not session:
            return
        for key, owner_id in list(_profile_owners.items()):
            if owner_id == session_id:
                _profile_owners.pop(key, None)


def get_session(session_id):
    with _lock:
        session = _sessions.get(session_id)
        if not session:
            return None
        snapshot = dict(session, tabs=dict(session["tabs"]))
        snapshot["live"] = _session_live(session)
        return snapshot


def publish_tabs(session_id, tabs, replace=True):
    """Publish a session's tab list (e.g. from list_tabs / CDP /json).

    *tabs* is a list of dicts with at least ``tab_id``; ``url``, ``title``
    and ``ws_url`` are kept when present. The CDP ``webSocketDebuggerUrl`` is
    preserved here as a STRING instead of being reduced to a boolean — it is
    the handle a future action layer needs (F47 names that loss explicitly).

    *replace* is False for callers that publish one discovery pass worth of
    tabs while another pass may still be authoritative; the default keeps the
    historical "this is the whole list" behavior. Every tab carries its
    instance-qualified id (``<instance>:<tab id>``) so two browsers' "0" are
    not one tab.
    """
    if not session_id or not isinstance(tabs, (list, tuple)):
        return
    with _lock:
        session = _sessions.get(session_id)
        if not session:
            return
        existing = {} if replace else dict(session["tabs"])
        instance = session.get("instance") or session_id
        for tab in tabs:
            if not isinstance(tab, dict):
                continue
            tab_id = str(tab.get("tab_id") or tab.get("id") or "").strip()
            if not tab_id:
                continue
            url = str(tab.get("url") or "")
            entry = {
                "tab_id": tab_id,
                "session_id": session_id,
                "instance": instance,
                "qualified_id": qualified_tab_id(instance, tab_id),
                "url": url,
                "origin": origin_of(url),
                "title": str(tab.get("title") or ""),
                # F47: a boolean (or anything non-URL) is NOT a handle.
                "ws_url": canonical_ws_url(tab.get("ws_url")
                                           or tab.get("webSocketDebuggerUrl")),
                # F17: where this identity came from — "structured" when the
                # daemon issued a typed id, "derived" when it was computed
                # from content, "text" when it was parsed out of display text
                # (never trusted for binding).
                "identity_source": str(tab.get("identity_source") or "structured"),
                "updated_at": time.time(),
            }
            existing[tab_id] = entry
        session["tabs"] = existing
        session["last_seen"] = time.time()


def note_tab(session_id, tab_id, url=None, title=None, ws_url=None,
             identity_source=None):
    """Upsert one tab on a session (best-effort, unknown session is a no-op)."""
    if not session_id or not tab_id:
        return
    with _lock:
        session = _sessions.get(session_id)
        if not session:
            return
        tab = session["tabs"].get(str(tab_id)) or {}
        if url is not None:
            tab["url"] = str(url)
        if title is not None:
            tab["title"] = str(title)
        handle = canonical_ws_url(ws_url)
        if handle:
            tab["ws_url"] = handle
        tab.setdefault("url", "")
        tab.setdefault("title", "")
        tab.setdefault("ws_url", "")
        tab["tab_id"] = str(tab_id)
        tab["session_id"] = session_id
        instance = session.get("instance") or session_id
        tab["instance"] = instance
        tab["qualified_id"] = qualified_tab_id(instance, tab_id)
        tab["origin"] = origin_of(tab.get("url"))
        tab["identity_source"] = str(
            identity_source or tab.get("identity_source") or "structured")
        tab["updated_at"] = time.time()
        session["tabs"][str(tab_id)] = tab
        session["last_seen"] = time.time()


def drop_tab(session_id, tab_id):
    if not session_id or not tab_id:
        return
    with _lock:
        session = _sessions.get(session_id)
        if session:
            session["tabs"].pop(str(tab_id), None)


def _candidate(tab):
    return {
        "session_id": tab.get("session_id"),
        "instance": tab.get("instance"),
        "tab_id": tab.get("tab_id"),
        "qualified_id": tab.get("qualified_id"),
        "url": tab.get("url"),
        "title": tab.get("title"),
        "owner": None,
    }


def _find_tab_matches(tab_id, now=None):
    """Every LIVE (session, tab) whose raw or instance-qualified id matches."""
    wanted = str(tab_id or "").strip()
    if not wanted:
        return []
    matches = []
    for session in _sessions.values():
        if not _session_live(session, now):
            continue
        for key, tab in session["tabs"].items():
            if str(key) == wanted or str(tab.get("qualified_id") or "") == wanted:
                candidate = _candidate(tab)
                candidate["owner"] = session.get("owner")
                matches.append((session, tab, candidate))
    return matches


def _resolve_matches(target, now=None):
    """LIVE (session, tab) matches for a tab id, URL or origin (F47)."""
    text = str(target or "").strip()
    if not text:
        return []
    by_id = _find_tab_matches(text, now)
    if by_id:
        return by_id
    url_matches = []
    origin_matches = []
    for session in _sessions.values():
        if not _session_live(session, now):
            continue
        for tab in session["tabs"].values():
            url = tab.get("url") or ""
            origin = tab.get("origin") or ""
            if not url:
                continue
            if text == url:
                candidate = _candidate(tab)
                candidate["owner"] = session.get("owner")
                url_matches.append((session, tab, candidate))
            elif origin and text == origin:
                candidate = _candidate(tab)
                candidate["owner"] = session.get("owner")
                origin_matches.append((session, tab, candidate))
    return url_matches or origin_matches


def attach_tab(tab_id):
    """Attach to a real tab by id (F47: the explicit, non-guessing path).

    Raises :class:`UnknownTabError` when no LIVE session has this tab — the
    caller must surface that instead of quietly opening some other browser —
    and :class:`AmbiguousTabError` when several live tabs match, listing the
    candidates so the user can resolve it.
    """
    with _lock:
        matches = _find_tab_matches(tab_id)
        if not matches:
            raise UnknownTabError(
                "tab %r is not open in any LIVE known browser session — "
                "refusing to open a different profile or browser; ask the user "
                "or list tabs first" % (tab_id,))
        if len(matches) > 1:
            candidates = [candidate for _s, _t, candidate in matches]
            raise AmbiguousTabError(
                "tab %r matches %d live tabs (%s) — refusing to guess which one; "
                "ask the user or use a full instance-qualified id"
                % (tab_id, len(matches),
                   ", ".join(sorted(str(c.get("qualified_id") or c.get("tab_id"))
                                    for c in candidates))),
                candidates)
        session, tab, _candidate_entry = matches[0]
        return {"session": dict(session, tabs={}), "tab": dict(tab)}


def resolve_targets(target):
    """All live matches for *target*, with an explicit ambiguity flag (F47).

    Returns ``{"matches": [...], "ambiguous": bool}``; the singular
    :func:`resolve_target` keeps its old "None when unknown or ambiguous"
    contract for callers that only need one best answer.
    """
    with _lock:
        matches = _resolve_matches(target)
        candidates = [candidate for _s, _t, candidate in matches]
        return {"matches": [_candidate_entry_from(session, tab)
                            for session, tab, _c in matches],
                "ambiguous": len(candidates) > 1,
                "candidates": candidates}


def _candidate_entry_from(session, tab):
    return {"session": dict(session, tabs={}), "tab": dict(tab)}


def resolve_target(target):
    """Best (session, tab) matching a tab id, URL or origin — None if unknown
    or ambiguous (F47: an ambiguous target is NOT silently the first one)."""
    with _lock:
        matches = _resolve_matches(target)
        if not matches:
            return None
        if len(matches) > 1:
            return None
        session, tab, _candidate_entry = matches[0]
        return {"session": dict(session, tabs={}), "tab": dict(tab)}


def route_open(url, prefer_tab=None):
    """Decide how "open *url*" should reach a browser. Pure, no side effects.

    Returns one of:
      ``{"action": "attach",  "session", "tab"}`` — a specific tab was asked
          for and exists: use THAT tab, never a new browser;
      ``{"action": "focus",   "session", "tab"}``  — the URL is already open
          in a known session: bring that tab forward instead of opening a
          second copy;
      ``{"action": "ambiguous", "candidates", "reason"}`` — several live tabs
          match: the caller must ask the user, not guess;
      ``{"action": "research_page", "session"}``   — no tab matches and the
          warm research browser is live: open a page in IT (same profile,
          same authentication) rather than launching a cold default browser;
      ``{"action": "error", "reason"}``            — an explicitly named tab
          does not exist: refuse, never launch elsewhere;
      ``{"action": "external"}``                   — nothing is known: fall
          back to the user's default browser (and record the launch).
    """
    if prefer_tab:
        try:
            attached = attach_tab(prefer_tab)
            attached["action"] = "attach"
            return attached
        except AmbiguousTabError as exc:
            return {"action": "ambiguous", "reason": str(exc),
                    "candidates": exc.candidates}
        except UnknownTabError:
            return {"action": "error", "reason": "unknown tab %r" % (prefer_tab,)}

    resolved = resolve_targets(url)
    if resolved["ambiguous"]:
        return {"action": "ambiguous", "candidates": resolved["candidates"],
                "reason": ("%d live tabs match %s — ask the user which one "
                           "before opening anything" % (len(resolved["candidates"]), url))}
    if resolved["matches"]:
        match = resolved["matches"][0]
        match["action"] = "focus"
        return match

    with _lock:
        for session in _sessions.values():
            if session["kind"] == "persistent-context" and session["owner"] == "research" \
                    and _session_live(session):
                return {"action": "research_page", "session": dict(session, tabs={})}
    return {"action": "external"}


def note_external_launch(url, browser=None):
    """Record a launch into the user's (unmanaged) default browser.

    The profile is unknown and nothing is claimed — this exists so the broker
    can at least say "this URL was opened externally at <time>" instead of
    knowing nothing (F47: identity, not control).
    """
    try:
        register_session("external", kind="external-launch", session_id="external",
                         label="user's default browser (unmanaged)")
        tab_id = "ext-%d" % next(_external_tab_ids)
        note_tab("external", tab_id, url=url, title="")
        return tab_id
    except Exception:
        return None


def describe():
    """Snapshot of every known session and tab (diagnostics / snapshots)."""
    with _lock:
        sessions = []
        for sess in _sessions.values():
            entry = {k: v for k, v in sess.items() if k != "tabs"}
            entry["live"] = _session_live(sess)
            entry["tabs"] = [dict(tab) for tab in sess["tabs"].values()]
            sessions.append(entry)
        return {"sessions": sessions}


def reset():
    """Drop all identity (tests only)."""
    with _lock:
        _sessions.clear()
        _profile_owners.clear()
