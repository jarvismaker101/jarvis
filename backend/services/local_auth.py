"""Fable-5 audit G11 / F51 — localhost command authentication.

Every POST that mutates Jarvis state (/ask, /ask/stream, /task/stop,
/speak/stop, /settings/*, /research-*, /screen-answer, …) AND every read of
private state (/ui-state, /voice-state, /voice-log, /screen-answer,
/research-*, /ask/status/*) requires the ``X-Jarvis-Token`` header to carry
the per-launch secret minted by the process supervisor that owns the backend
(watcher or main.js). The secret travels to the backend ONLY through the
child-process environment (``JARVIS_LOCAL_TOKEN``) — never on disk, never in
source, never in a URL.

Audit F51 ("Separate Rendering From Computer-Control Authority"):

    Correction: Fail closed outside explicit development mode; authenticate
    private reads, restrict origins/publishers, centralize the authenticated
    client, verify Electron boundaries, and rotate exposed credentials.

What this module now enforces:

* **Fail closed outside explicit development mode.** An unarmed backend
  (no ``JARVIS_LOCAL_TOKEN``) no longer disables enforcement: only an
  explicit ``JARVIS_DEV_MODE=1`` does. A supervisor-less production process
  now rejects every non-public request instead of serving it.
* **Private reads are authenticated.** The public surface is exactly
  ``GET /health`` (liveness + the supervisor attribution contract); state
  polls that reveal what Jarvis is doing, hearing, seeing or researching
  require the token — so renderer content that cannot obtain the token
  cannot read private data either.
* **Origins/publishers are restricted.** A cross-origin (web page) request
  is refused before the token check, and a browser-origin caller can only
  ever see the minimal health payload — the supervisor identity/fingerprint
  fields are served to no-Origin (native) callers only.
* **One authenticated client.** :func:`auth_headers` is the single place a
  client (voice worker, watcher, main process, tests) builds its headers.
* **Rotation.** :func:`rotate` mints a replacement and immediately
  invalidates the old secret.

Security posture (honest, not theatrical):
- The threat model is loopback-only abuse (another local user/process, a
  malicious web page that finds an open localhost port, a stale second
  Jarvis copy). A per-launch random token defeats cross-process replay and
  cross-launch reuse; it does NOT defend against a same-user keylogger or
  memory scraper — nothing short of OS privilege separation would.
- uvicorn is always bound to 127.0.0.1 by the launchers; this module does
  not (and cannot) enforce the bind itself.
- Every response carries ``X-Jarvis-Auth: on|off|closed`` so clients can
  see which posture applies.

IN-TREE CALLERS THAT MUST NOW PRESENT THE TOKEN (they run inside supervised
child processes that already hold ``JARVIS_LOCAL_TOKEN``, so the fix is a
one-line header change each — see the F51 report):

* ``watcher._backend_has_research_endpoint`` — GET /research-result;
* ``voice_mode._get_backend`` — GET /ui-state (task-mute poll);
* ``listener._api_is_speaking`` — GET /voice-state (cross-process barge-in);
* ``listener._post_backend_speak_stop`` — POST /speak/stop (already 401 when
  armed; it never sent the token).

Route functions must publish state to the backend (F50) instead of reading
another process's module copy.
"""

import hashlib
import hmac
import json
import os
import re
import secrets

HEADER = "X-Jarvis-Token"
STATUS_HEADER = "X-Jarvis-Auth"

_ENV_VAR = "JARVIS_LOCAL_TOKEN"
#: Explicit development mode. The ONLY way to run without a token.
_DEV_ENV = "JARVIS_DEV_MODE"
_DEV_VALUES = frozenset(("1", "true", "yes", "on", "dev", "development"))

_token: str = ""
_enabled: bool = False
_rotations: int = 0


def _valid_token_shape(value: str) -> bool:
    return isinstance(value, str) and len(value) >= 32


def development_mode() -> bool:
    """True only when development mode was EXPLICITLY declared (F51).

    A missing token is not a development declaration: inferring dev mode
    from absence is exactly the fail-open behaviour the audit found.
    """
    return os.getenv(_DEV_ENV, "").strip().lower() in _DEV_VALUES


def configure(token: str = "") -> bool:
    """Arm auth with *token*; return True when a token is armed.

    An empty/short token disarms the token check. Outside explicit
    development mode that means FAIL CLOSED (see :func:`enforcement_active`),
    not "open".
    """
    global _token, _enabled
    token = (token or "").strip()
    if _valid_token_shape(token):
        _token = token
        _enabled = True
    else:
        _token = ""
        _enabled = False
    return _enabled


def configure_from_env() -> bool:
    """Arm auth from the supervisor-supplied environment. Idempotent."""
    return configure(os.getenv(_ENV_VAR, ""))


def mint_token() -> str:
    """Mint one launch secret. Called ONLY by the process supervisor."""
    return secrets.token_urlsafe(32)


def rotate() -> str:
    """Replace the live secret and return the new one (F51).

    The previous token stops matching IMMEDIATELY (there is no grace list),
    so a leaked or logged secret cannot be replayed after rotation.
    """
    global _rotations
    new_token = mint_token()
    configure(new_token)
    _rotations += 1
    return new_token


def rotations() -> int:
    return _rotations


def is_enabled() -> bool:
    return _enabled


def enforcement_active() -> bool:
    """True when non-public requests MUST present a valid token.

    Armed -> True. Unarmed + explicit development mode -> False (open).
    Unarmed without development mode -> True (fail closed).
    """
    return _enabled or not development_mode()


def posture() -> str:
    """``"on"`` (armed) / ``"off"`` (explicit dev mode) / ``"closed"``."""
    if _enabled:
        return "on"
    return "off" if development_mode() else "closed"


def token_from_env():
    """The launch token this process was given, or "" (centralized, F51)."""
    return os.getenv(_ENV_VAR, "")


def auth_headers(extra=None, token=None):
    """The ONE authenticated client header set (F51).

    Every in-process client — voice worker, watcher control calls, tests —
    builds its headers here instead of hand-rolling them, so a change to the
    header name or shape lands in exactly one place.
    """
    headers = {"Content-Type": "application/json"}
    value = token if token is not None else token_from_env()
    if value:
        headers[HEADER] = value
    if extra:
        headers.update(extra)
    return headers


def token_fingerprint() -> str:
    """First 8 hex of sha256(token) for logs — never log the token itself."""
    if not _enabled:
        return "off"
    return fingerprint_for(_token)


def fingerprint_for(token: str) -> str:
    """Fingerprint of *token* as a *client* sees it (F52).

    The supervisor must compare the authority a live worker reports on
    ``/health`` against the token it minted for THIS launch — without ever
    putting the token itself on the wire or on disk. ``"off"`` for an
    empty/short token mirrors /health's reporting for an unarmed worker.
    """
    token = (token or "").strip()
    if not _valid_token_shape(token):
        return "off"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


def check(provided: str) -> bool:
    """Constant-time comparison against the launch secret.

    With no armed token this returns True ONLY in explicit development mode:
    an unarmed production process authorizes nothing (fail closed).
    """
    if not _enabled:
        return development_mode()
    if not provided:
        return False
    try:
        return hmac.compare_digest(str(provided), _token)
    except Exception:
        return False


# ── Public surface ─────────────────────────────────────────────────────────
# Only liveness/attribution stays open: a stale supervisor must be able to
# probe whether a warm backend is listening, and the F52 identity checks
# need pid/protocol/instance_id. Every state read is private (F51).
PUBLIC_PATHS = frozenset({
    "GET /health",
})

#: Backwards-compatible alias (older callers/tests read OPEN_PATHS).
OPEN_PATHS = PUBLIC_PATHS

#: Fields an UNAUTHENTICATED caller may see for a public path.
PUBLIC_RESPONSE_FIELDS = {
    "GET /health": ("ok", "service", "pid", "protocol"),
}

#: Additional fields served to a no-Origin (native/supervisor) caller
#: without a token: the supervisor attribution contract (F52). Never served
#: to renderer content, which can therefore not read the token fingerprint.
SUPERVISOR_FIELDS = ("instance_id", "auth")

# ── Origins / publishers (F51) ─────────────────────────────────────────────
#: Renderer origins the Electron shells load from. A file:// page reports
#: ``null`` as its Origin.
RENDERER_ORIGINS = ("null", "file://", "app://jarvis")
#: Local HTTP origins (dev servers, the capsule) are allowed too.
LOOPBACK_ORIGIN_PATTERN = r"^https?://(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$"
_LOOPBACK_ORIGIN_RE = re.compile(LOOPBACK_ORIGIN_PATTERN)


def origin_allowed(origin):
    """True when *origin* may talk to this control surface.

    No Origin header means a native client (the supervisor, the voice
    worker, a test) — not renderer content — and is allowed to send
    requests; it still needs the token for anything non-public.
    """
    origin = (origin or "").strip().lower()
    if not origin:
        return True
    if origin in RENDERER_ORIGINS:
        return True
    return bool(_LOOPBACK_ORIGIN_RE.match(origin))


def renderer_content(origin) -> bool:
    """True when the request carries a browser/renderer Origin header."""
    return bool((origin or "").strip())


def is_open(method: str, path: str) -> bool:
    """True when (method, path) is readable without a token."""
    key = "%s %s" % ((method or "GET").upper(), (path or "/"))
    return key in PUBLIC_PATHS


def path_key(method, path):
    return "%s %s" % ((method or "GET").upper(), (path or "/"))


def public_payload(key, payload, origin=""):
    """Filter a public path's payload for an UNauthenticated caller.

    F51 acceptance: "public health is minimal". Identity/fingerprint fields
    are only served to no-Origin (native supervisor) callers; renderer
    content sees liveness alone.
    """
    allowed = set(PUBLIC_RESPONSE_FIELDS.get(key, ()))
    if not renderer_content(origin):
        allowed.update(SUPERVISOR_FIELDS)
    return {k: v for k, v in dict(payload or {}).items() if k in allowed}


# ── ASGI middleware (installed by backend.main) ──────────────────────────────
# Runs before every route handler:
#   1. a cross-origin (web page) request is refused outright;
#   2. a public path is served (with its payload reduced for anonymous
#      renderer callers);
#   3. everything else requires the launch token — unless development mode
#      was explicitly declared.

#: Cap on buffered public responses (health is a small JSON document).
_MAX_PUBLIC_BODY = 64 * 1024


class LocalTokenMiddleware:
    """Starlette-compatible middleware enforcing the launch token."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        origin = request.headers.get("origin", "")
        if not origin_allowed(origin):
            response = JSONResponse(
                status_code=403,
                content={
                    "ok": False,
                    "error": "origin not permitted",
                    "hint": "renderer content cannot call the local control "
                            "surface from another origin",
                },
                headers={STATUS_HEADER: posture()},
            )
            await response(scope, receive, send)
            return

        method = request.method
        path = request.url.path
        key = path_key(method, path)
        authenticated = check(request.headers.get(HEADER, ""))

        if is_open(method, path):
            if key in PUBLIC_RESPONSE_FIELDS and not authenticated:
                await self._send_filtered(scope, receive, send, key, origin)
                return
            await self.app(scope, receive, send)
            return

        if not enforcement_active():
            await self.app(scope, receive, send)
            return

        if not authenticated:
            response = JSONResponse(
                status_code=401,
                content={
                    "ok": False,
                    "error": "unauthorized local command",
                    "hint": "missing or invalid per-launch local token",
                },
                headers={STATUS_HEADER: posture()},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)

    async def _send_filtered(self, scope, receive, send, key, origin):
        """Serve a public path with its payload reduced for this caller."""
        captured = []

        async def _capture(message):
            captured.append(message)

        await self.app(scope, receive, _capture)

        start = None
        chunks = []
        for message in captured:
            if message["type"] == "http.response.start" and start is None:
                start = message
            elif message["type"] == "http.response.body":
                if len(b"".join(chunks)) < _MAX_PUBLIC_BODY:
                    chunks.append(message.get("body", b""))
        if start is None:
            for message in captured:
                await send(message)
            return

        body = b"".join(chunks)
        headers = [(k, v) for (k, v) in start.get("headers", [])]
        content_type = ""
        for name, value in headers:
            if name.decode("latin-1").lower() == "content-type":
                content_type = value.decode("latin-1").lower()
                break
        if start.get("status") == 200 and "application/json" in content_type:
            try:
                payload = json.loads(body.decode("utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                body = json.dumps(public_payload(key, payload, origin),
                                  ensure_ascii=False).encode("utf-8")
                headers = [(k, v) for (k, v) in headers
                           if k.decode("latin-1").lower() != "content-length"]
                headers.append((b"content-length", str(len(body)).encode()))
        headers.append((STATUS_HEADER.lower().encode(), posture().encode()))

        await send({"type": "http.response.start", "status": start["status"],
                    "headers": headers})
        await send({"type": "http.response.body", "body": body,
                    "more_body": False})


def install(app) -> bool:
    """Arm auth from env and attach the middleware to a FastAPI app.

    Returns True when a token is armed after install. A token already
    configured by a caller (e.g. a test that arms auth before importing the
    app) is never clobbered by the env read. With no token the middleware
    now FAILS CLOSED unless development mode was explicitly declared.
    """
    if not is_enabled():
        configure_from_env()
    app.add_middleware(LocalTokenMiddleware)
    return is_enabled()
