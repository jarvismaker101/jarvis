"""P0-12 — pre-warm and keep alive the connections on the latency-critical path.

Pooled sessions already existed (gemini_client, openai_compat_client) but were
never PRE-CONNECTED, so the first call of a session paid a fresh TCP + TLS
handshake on the LLM, classifier and STT calls. Meanwhile the voice worker
knows the user is about to speak (VAD onset) several hundred milliseconds
before the transcript exists, and did nothing with that head start.

This module is the one place that:

* holds the keepalive socket options (an idle pooled socket is dropped by a VPN
  or NAT silently otherwise) — see :func:`keepalive_socket_options`;
* opens the connection for a target and READS THE RESPONSE FULLY, because a
  half-read response never returns its socket to the pool;
* rate-limits itself (once per :data:`WARM_INTERVAL_SECONDS`, per process),
  skips while a warm is already in flight, and NEVER raises: pre-warming is an
  optimisation and must be invisible when a provider is down.

It is advisory by construction. If it never runs, every call path works exactly
as it did before — it only removes a handshake that would otherwise happen on
the user's clock.
"""

import logging
import os
import socket
import threading
import time

import requests
from requests.adapters import HTTPAdapter

#: P0-12 — the rate limit. Onsets are frequent; a warm per onset would be
#: abusive traffic, so a warm inside this window is skipped (counted).
WARM_INTERVAL_SECONDS = float(os.getenv("JARVIS_PREWARM_INTERVAL", "20"))
#: Latency-critical CONNECT budget: the pool is pre-warmed precisely so this is
#: only reached when there is no warm socket (a cold TLS handshake on a VPN can
#: exceed it, which is why the read side stays generous).
CONNECT_TIMEOUT_SECONDS = float(os.getenv("JARVIS_PREWARM_CONNECT_TIMEOUT", "1.0"))
READ_TIMEOUT_SECONDS = float(os.getenv("JARVIS_PREWARM_READ_TIMEOUT", "4.0"))
#: TCP keepalive — modest, RFC-1122-era values. Deliberately not aggressive:
#: a provider's traffic systems must not read this as abuse.
KEEPALIVE_IDLE_SECONDS = 30
KEEPALIVE_INTERVAL_SECONDS = 10
KEEPALIVE_PROBES = 3

_lock = threading.Lock()
_last_warm_at = 0.0
_in_flight = False
_stats = {"warms": 0, "skipped": 0, "degraded": 0, "errors": 0,
          "last_result": None, "last_at": None, "targets": {}}


def keepalive_socket_options():
    """The socket options that keep a pooled connection from going stale.

    Returns a list of ``(level, optname, value)`` triples; the TCP_* tuning is
    added only where the platform exposes it (Windows and Linux both do, with
    different constant availability).
    """
    options = []
    try:
        options.append((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1))
    except Exception:
        return options
    for name, value in (("TCP_KEEPIDLE", KEEPALIVE_IDLE_SECONDS),
                        ("TCP_KEEPINTVL", KEEPALIVE_INTERVAL_SECONDS),
                        ("TCP_KEEPCNT", KEEPALIVE_PROBES)):
        option = getattr(socket, name, None)
        if option is None:
            continue
        try:
            options.append((socket.IPPROTO_TCP, option, value))
        except Exception:
            continue
    return options


class KeepAliveAdapter(HTTPAdapter):
    """An ``HTTPAdapter`` whose pool opens sockets with keepalive enabled."""

    def __init__(self, *args, **kwargs):
        self._socket_options = keepalive_socket_options()
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["socket_options"] = list(self._socket_options)
        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["socket_options"] = list(self._socket_options)
        return super().proxy_manager_for(*args, **kwargs)


def pooled_session(pool_connections=4, pool_maxsize=8):
    """A keepalive-enabled session for a latency-critical client."""
    session = requests.Session()
    adapter = KeepAliveAdapter(pool_connections=pool_connections,
                               pool_maxsize=pool_maxsize)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


#: The module's own session: one per process, reused for every warm so the warm
#: itself does not pay a handshake the next turn will not benefit from.
_session = pooled_session()


def _client_session(module_name, attr="_session"):
    """A client module's pooled session, or None (never raises).

    S26: warming prewarm's own pool while the real call runs on the client's
    pool would pre-connect a socket nobody uses. The warm must open the
    connection in the SAME pool the request will draw from.
    """
    try:
        import importlib

        return getattr(importlib.import_module(module_name), attr)
    except Exception:
        return None


def _classifier_targets():
    """The classifier's ladder — the hottest sessions after the chat provider."""
    targets = []
    try:
        from backend.config import GEMINI_API_KEY, GROQ_API_KEY
    except Exception:
        GEMINI_API_KEY = GROQ_API_KEY = ""
    if GEMINI_API_KEY:
        # The classifier calls Gemini through ask_gemini_chat(no_retry=True),
        # whose pooled socket lives in _no_retry_session (S26).
        targets.append({
            "name": "gemini",
            "url": ("https://generativelanguage.googleapis.com/v1beta/models"
                    "?key=%s" % GEMINI_API_KEY),
            "headers": {},
            "session": _client_session(
                "backend.services.gemini_client", "_no_retry_session"),
        })
    if GROQ_API_KEY:
        targets.append({
            "name": "groq",
            "url": "https://api.groq.com/openai/v1/models",
            "headers": {"Authorization": "Bearer %s" % GROQ_API_KEY},
            "session": _client_session("backend.services.grok_client"),
        })
    return targets


def _chat_target():
    """The provider the CHAT role will actually call, if it has an HTTP GET."""
    try:
        from backend.services import model_registry
        provider, _model = model_registry.get_model_for_role("chat")
    except Exception:
        return None
    provider = str(provider or "").strip()
    if not provider:
        return None
    try:
        api_key, base_url = model_registry.get_provider_credentials(provider)
    except Exception:
        return None
    if provider == "gemini":
        if not api_key:
            return None
        return {
            "name": "gemini",
            "url": ("https://generativelanguage.googleapis.com/v1beta/models"
                    "?key=%s" % api_key),
            "headers": {},
            "session": _client_session("backend.services.gemini_client"),
        }
    if provider == "fireworks":
        if not api_key:
            return None
        return {
            "name": "fireworks",
            "url": "https://api.fireworks.ai/inference/v1/models",
            "headers": {"Authorization": "Bearer %s" % api_key},
            "session": _client_session("backend.services.fireworks_client"),
        }
    if provider == "groq":
        if not api_key:
            return None
        return {
            "name": "groq",
            "url": "https://api.groq.com/openai/v1/models",
            "headers": {"Authorization": "Bearer %s" % api_key},
            "session": _client_session("backend.services.grok_client"),
        }
    if not base_url:
        # native/adapter providers without a cheap authenticated GET (whisper,
        # fish, gtts, a custom gateway without a /models route) are skipped:
        # warming them would mean a fake request, not a cheaper real one.
        return None
    headers = {}
    if api_key:
        headers["Authorization"] = "Bearer %s" % api_key
    return {"name": provider, "url": str(base_url).rstrip("/") + "/models",
            "headers": headers,
            "session": _client_session("backend.services.openai_compat_client")}


def targets():
    """The hot sessions to warm, deduped by URL. Never raises."""
    found = []
    try:
        chat = _chat_target()
    except Exception:
        chat = None
    if chat:
        found.append(chat)
    try:
        found.extend(_classifier_targets())
    except Exception:
        pass
    try:
        from backend.services import transcription
        stt = transcription.prewarm_target()
    except Exception:
        stt = None
    if stt:
        found.append(stt)
    seen = set()
    unique = []
    for target in found:
        url = target.get("url")
        # S26: the same URL may be served by two different pools (Gemini chat
        # vs the classifier's no_retry leg) — each pool needs its own warm, so
        # the dedup key is (url, session).
        key = (url, id(target.get("session")))
        if not url or key in seen:
            continue
        seen.add(key)
        unique.append(target)
    return unique


def read_fully(response):
    """Consume a response body so its socket returns to the pool.

    A warm that leaves the body unread leaves the socket unusable — the next
    real call then pays the handshake the warm was supposed to save. Returns the
    number of bytes read (0 when the body was already consumed).
    """
    try:
        body = response.content
    except Exception:
        body = b""
    try:
        response.close()
    except Exception:
        pass
    return len(body or b"")


def _warm_one(target):
    """Open (or reuse) the pooled connection for one target. Never raises.

    S26: the warm must land in the CLIENT's pool (``target["session"]``) — a
    socket opened in prewarm's own pool would never be drawn on by the real
    call. Targets without a session (tests, exotic providers) use this
    module's pool as before.
    """
    name = str(target.get("name") or "unknown")
    session = target.get("session") or _session
    try:
        response = session.get(
            target["url"],
            headers=target.get("headers") or None,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        )
        bytes_read = read_fully(response)
        return name, "warm", {"status": response.status_code,
                              "bytes": bytes_read}
    except Exception as exc:
        return name, "degraded", {"error": str(exc)[:200]}


def warm(force=False):
    """Warm every hot session. Returns a report; NEVER raises.

    Rate-limited per process to one warm per :data:`WARM_INTERVAL_SECONDS`
    (a caller may pass ``force=True``). While a warm is in flight a second
    caller is skipped rather than queued: pre-warming must not become a
    bottleneck of its own.
    """
    global _last_warm_at, _in_flight, _stats
    now = time.monotonic()
    with _lock:
        if _in_flight:
            _stats["skipped"] += 1
            return {"ok": True, "skipped": "in_flight", "warmed": [],
                    "degraded": []}
        if not force and (now - _last_warm_at) < WARM_INTERVAL_SECONDS:
            _stats["skipped"] += 1
            return {"ok": True, "skipped": "rate_limited", "warmed": [],
                    "degraded": []}
        _in_flight = True

    warmed = []
    degraded = []
    try:
        for target in targets():
            name, outcome, detail = _warm_one(target)
            if outcome == "warm":
                warmed.append(name)
            else:
                degraded.append(name)
            _stats["targets"][name] = {"outcome": outcome,
                                       "at": time.time(), **detail}
    except Exception as exc:  # a warm must never surface as an error
        _stats["errors"] += 1
        logging.debug("[PREWARM] warm failed: %s", exc)
        degraded.append("warm")
    finally:
        with _lock:
            _in_flight = False
            _last_warm_at = time.monotonic()
            _stats["warms"] += 1
            if degraded:
                _stats["degraded"] += 1
            _stats["last_result"] = {"warmed": warmed, "degraded": degraded}
            _stats["last_at"] = time.time()
    return {
        "ok": True,
        # "degraded" is a real, useful answer: the optimisation is advisory, so
        # a provider being down is reported, not raised.
        "degraded": degraded,
        "warmed": warmed,
    }


def warm_async(force=False):
    """Fire a warm on a daemon thread. Returns immediately.

    The voice worker calls this from the real-time capture thread at VAD onset,
    where a network call (or even a queue) must never appear.
    """
    try:
        threading.Thread(target=warm, args=(force,), name="prewarm",
                         daemon=True).start()
        return True
    except Exception:
        return False


def stats():
    """P0-12 — counters for telemetry and tests (never raises)."""
    with _lock:
        snapshot = dict(_stats)
        snapshot["targets"] = dict(_stats["targets"])
        snapshot["in_flight"] = _in_flight
        snapshot["seconds_since_warm"] = (
            None if not _last_warm_at else time.monotonic() - _last_warm_at)
    return snapshot


def reset():
    """Test hook: forget the rate limit and the counters."""
    global _last_warm_at, _in_flight, _stats
    with _lock:
        _last_warm_at = 0.0
        _in_flight = False
        _stats = {"warms": 0, "skipped": 0, "degraded": 0, "errors": 0,
                  "last_result": None, "last_at": None, "targets": {}}
