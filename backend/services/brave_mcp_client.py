"""Raw JSON-RPC client for the brave-control MCP HTTP daemon.

Talks the MCP streamable-HTTP handshake over plain HTTP (no SDK): POST
initialize captures the mcp-session-id response header, which every later
request echoes; notifications/initialized is a JSON-RPC notification (no
id); tools/list and tools/call are ordinary requests.
"""

import json
import logging
import time

import requests

from backend.core import deadline as budget
from backend.services.opencode_client import BRAVE_MCP_PORT, BRAVE_MCP_TOKEN

_DEFAULT_URL = "http://127.0.0.1:%d/mcp" % BRAVE_MCP_PORT
_INITIALIZE_ID = 1

#: L-6: the ONLY failures a request may be replayed after. An HTTP 404 on a
#: sessioned POST, or a JSON-RPC error naming the session itself, proves the
#: server rejected the request BEFORE dispatch — so replaying it after a
#: fresh handshake cannot double-execute a mutation. Every other failure
#: (500s, timeouts, transport drops) still propagates untouched: the
#: agent's own retry policy owns those, and a possibly-committed mutation
#: is never auto-replayed after an ambiguous failure.
_SESSION_DEAD_MARKERS = (
    "invalid session",
    "unknown session",
    "session not found",
    "session expired",
    "no such session",
    "session closed",
)


def _is_session_death(exc):
    """True when *exc* proves the daemon forgot our MCP session."""
    try:
        status = int(getattr(getattr(exc, "response", None),
                             "status_code", None))
    except (TypeError, ValueError):
        status = None
    if status == 404:
        return True
    try:
        message = ("%s" % exc).lower()
    except Exception:
        return False
    return any(marker in message for marker in _SESSION_DEAD_MARKERS)


class BraveMcpClient:
    """Minimal MCP client over requests.Session, one shared browser session.

    The daemon keeps the browser warm and the DOM-indexed tools alive; this
    client only speaks the wire protocol. Errors surface as RuntimeError so
    the agent loop can retry and fail cleanly.
    """

    #: BA-03: per-class wire budgets (seconds). The old code gave EVERY call
    #: — a 2 s location.href read, a file download, the LLM-sized default —
    #: the same 120 s via one shared self.timeout, so a hung CDP evaluate
    #: could block a task for the full budget. Each class now gets its own:
    #: probes 8 s, real-input actions 15 s, navigation 30 s, screenshot 15 s,
    #: file transfer 45 s. Anything not listed falls back to self.timeout.
    #: Every value is still capped to the bound task deadline at send time,
    #: so a task never outlives BROWSER_AGENT_TIMEOUT by more than the one
    #: call already in flight.
    _TOOL_CLASS_TIMEOUTS = {
        # probes / reads
        "evaluate": 8,
        "list_tabs": 8,
        "read_file": 8,
        "list_dir": 8,
        # real-input actions
        "click_locator": 15,
        "fill_locator": 15,
        "scroll": 15,
        "select_option": 15,
        "set_checked": 15,
        "switch_tab": 15,
        "new_tab": 15,
        "open_brave": 15,
        "screenshot": 15,
        # navigation
        "navigate": 30,
        # file transfer
        "download": 45,
        "upload_file": 45,
        "drag_drop": 45,
        # BA-14: the event-driven waiter may legitimately hold the round
        # trip for the full requested wait (virtual timeout_ms caps at
        # 10 s) plus settle margin.
        "wait_for": 15,
    }

    def __init__(self, base_url=None, token=None, timeout=None):
        self.base_url = base_url or _DEFAULT_URL
        self.token = token if token is not None else BRAVE_MCP_TOKEN
        self.timeout = timeout if timeout is not None else 120
        self.session = requests.Session()
        self._session_id = None
        self._next_id = _INITIALIZE_ID
        self._last_mcp_ms = 0
        #: L-6: successful session re-handshakes performed by reconnect().
        #: Lets pool owners tell a reborn session (tool cache is suspect)
        #: from the one they cached against.
        self.reconnects = 0
        #: F38: non-text content blocks of the LAST call, preserved verbatim
        #: instead of being discarded. An image the daemon returned INLINE
        #: (rather than writing ``path``) used to reach the agent as the bare
        #: marker "[image omitted]" with the payload unrecoverable, so the
        #: visual steps went blind with no error to explain it.
        self.last_images = []

    def _headers(self, with_session=True):
        headers = {
            "Authorization": "Bearer %s" % self.token,
            # The MCP streamable-HTTP transport rejects requests that do not
            # accept both content types with 406 Not Acceptable.
            "Accept": "application/json, text/event-stream",
        }
        if with_session and self._session_id:
            headers["mcp-session-id"] = self._session_id
        return headers

    def _parse_body(self, response, expected_id):
        """Read a JSON-RPC body that may arrive as JSON or as an SSE stream."""
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("text/event-stream"):
            return response.json()
        body = None
        for line in (response.text or "").splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                message = json.loads(line[len("data:"):].strip())
            except ValueError:
                continue
            if message.get("id") == expected_id:
                body = message
                break
            # Surface error messages even when the server replies with a
            # null/mismatched id (e.g. invalid session); results for other
            # ids are never our answer.
            if "error" in message:
                body = message
                break
        if body is None:
            raise RuntimeError(
                "MCP SSE response had no message for id %s" % expected_id
            )
        return body

    def connect(self):
        """Run the initialize handshake; store the session id for later."""
        payload = {
            "jsonrpc": "2.0",
            "id": _INITIALIZE_ID,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "jarvis-browser-agent", "version": "1.0"},
            },
        }
        response = self.session.post(
            self.base_url,
            headers=self._headers(with_session=False),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        self._session_id = response.headers.get("mcp-session-id")
        if not self._session_id:
            raise RuntimeError("MCP initialize returned no mcp-session-id")
        body = self._parse_body(response, _INITIALIZE_ID)
        error = body.get("error")
        if error is not None:
            raise RuntimeError(error.get("message", str(error)))
        # JSON-RPC notification: no id, response body not parsed.
        self.session.post(
            self.base_url,
            headers=self._headers(),
            json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
            timeout=self.timeout,
        )

    def reconnect(self):
        """Drop the dead session and hand-shake a fresh one (L-6).

        Used when the daemon was restarted or upgraded underneath us: the
        old session id 404s, so a new initialize is the only way back.
        The TCP session is replaced too — a keep-alive connection to the
        dead daemon is not worth reusing. Raises when the fresh handshake
        fails, leaving _session_id None so pool owners drop this client.
        """
        try:
            self.session.close()
        except Exception:
            pass
        self.session = requests.Session()
        self._session_id = None
        try:
            self.connect()
        except Exception:
            self._session_id = None
            raise
        self.reconnects += 1
        logging.info("[BRAVE-MCP] session re-established (reconnect #%d)",
                     self.reconnects)

    def _request(self, method, params=None, _healed=False, timeout=None):
        self._next_id += 1
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._next_id,
            "params": params or {},
        }
        # BA-03: the wire timeout is the class budget (or the explicit
        # override) sliced to the bound task deadline. A spent budget sends
        # NO request at all — the caller must stop, not start fresh work on
        # expired time.
        wire_timeout = budget.seconds_for(
            None, timeout if timeout is not None else self.timeout)
        if wire_timeout is None:
            raise budget.BudgetExhausted(
                "browser task budget exhausted - MCP %s not sent" % method)
        try:
            _sw_t0 = time.monotonic()
            response = self.session.post(
                self.base_url,
                headers=self._headers(),
                json=payload,
                timeout=wire_timeout,
            )
            self._last_mcp_ms = int((time.monotonic() - _sw_t0) * 1000)
            response.raise_for_status()
            body = self._parse_body(response, payload["id"])
            error = body.get("error")
            if error is not None:
                raise RuntimeError(error.get("message", str(error)))
            return body.get("result", {})
        except Exception as exc:
            # L-6: exactly ONE re-handshake + replay, and only for proven
            # never-dispatched requests (see _SESSION_DEAD_MARKERS). A
            # second session-death, or any other failure, propagates to
            # the agent's own retry policy untouched.
            if not _healed and _is_session_death(exc):
                self.reconnect()
                return self._request(method, params, _healed=True,
                                     timeout=timeout)
            raise

    def list_tools(self, timeout=None):
        """Return the tool descriptors [{name, description, input_schema}]."""
        result = self._request("tools/list", timeout=timeout)
        tools = []
        for tool in result.get("tools", []):
            tools.append(
                {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("inputSchema", {}),
                }
            )
        return tools

    @staticmethod
    def _image_block(block):
        """F38: an inline image content block, preserved instead of dropped."""
        if not isinstance(block, dict):
            return None
        data = block.get("data")
        if not isinstance(data, str) or not data.strip():
            return None
        return {
            "mime_type": str(block.get("mimeType") or block.get("mime_type")
                             or "image/png"),
            "data": data,
        }

    def call_tool(self, name, arguments=None, timeout=None):
        """Call one tool; return the concatenated text of its content blocks.

        Non-text blocks keep their text marker ("[image omitted]") so the text
        contract is unchanged, but any inline image is ALSO preserved in
        :attr:`last_images` — F38: an image the model needed must never exist
        only as the marker that says it was dropped. A JSON-RPC error response
        raises RuntimeError with the server's message.

        BA-03: *timeout* overrides the per-class wire budget
        (:attr:`_TOOL_CLASS_TIMEOUTS`, falling back to ``self.timeout`` for
        unlisted tools) for this call only.
        """
        if timeout is None:
            timeout = self._TOOL_CLASS_TIMEOUTS.get(name, self.timeout)
        result = self._request(
            "tools/call", {"name": name, "arguments": arguments or {}},
            timeout=timeout,
        )
        self.last_images = []
        parts = []
        for block in result.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
                continue
            image = self._image_block(block)
            if image is not None:
                self.last_images.append(image)
            parts.append("[image omitted]")
        return "\n".join(parts)

    def close(self):
        """Close the MCP session; swallow every error (best effort)."""
        try:
            self.session.delete(self.base_url, headers=self._headers(), timeout=self.timeout)
        except Exception as exc:
            logging.warning("[BRAVE-MCP] close error: %s", exc)
        try:
            self.session.close()
        except Exception:
            pass
