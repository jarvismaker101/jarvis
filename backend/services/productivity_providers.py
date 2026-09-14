"""HTTP provider adapters for the F14 productivity connectors.

Fable-5 audit G4 / F14 — "add typed least-privilege read/draft operations
with separately approved sending and externally visible changes".

This module is the *network* half of that correction: it turns the typed
operations of :mod:`backend.services.productivity_connector` into concrete
HTTP requests for the services the user actually confirmed, and normalises
every response back into the connector's typed shapes. It contains no
authority logic — grants, revocation and effect approval live in the
connector, so a transport can never be reached before those gates have run.

Design notes
------------
* Only true externals (HTTP) live here. The session is injectable
  (``session=``) so tests can exercise the real request/parse paths without
  a network.
* Three profiles are available: ``google`` (Gmail / Google Calendar /
  People API), ``microsoft`` (Graph mail / calendar / contacts) and
  ``custom``, a declarative REST profile driven by endpoints supplied in the
  user's own service configuration — the honest answer to "confirm actual
  user services": a service we did not model can still be wired up without
  new code, and an unmodelled service is never *assumed*.
* Access tokens are held in memory only, are sent in the ``Authorization``
  header, and are never echoed in a result, an error or a repr.
* A non-2xx response raises :class:`ProviderError` carrying the status and a
  bounded, secret-scrubbed body — a failed send is reported, never silently
  treated as success.
"""

import base64
import json
from email.message import EmailMessage
from urllib.parse import quote

import requests

from backend.services import tool_policy

DEFAULT_TIMEOUT = 15.0
#: Bounded body kept from a failed response (never the whole thing).
_MAX_ERROR_BODY = 300

PROVIDER_NAMES = ("google", "microsoft", "custom")


class ProviderError(Exception):
    """A provider call failed at the transport level."""

    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


# ── normalisation helpers ──────────────────────────────────────────────────
def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _text(value):
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _b64url_decode(data):
    if not data:
        return ""
    try:
        padded = data + "=" * (-len(data) % 4)
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode(
            "utf-8", "replace")
    except Exception:
        return ""


def _b64url_encode(text):
    return base64.urlsafe_b64encode(
        (text or "").encode("utf-8")).decode("ascii").rstrip("=")


def _address_of(entry):
    """Pull an address out of a Google/Graph address object."""
    if isinstance(entry, str):
        return entry
    if not isinstance(entry, dict):
        return ""
    if entry.get("address"):
        return _text(entry["address"])
    email = entry.get("emailAddress") or {}
    if isinstance(email, dict) and email.get("address"):
        return _text(email["address"])
    if entry.get("value"):
        return _text(entry["value"])
    if entry.get("email"):
        return _text(entry["email"])
    return ""


def draft_to_message(draft):
    """Normalise a stored mail draft into the fields every profile needs."""
    fields = (draft or {}).get("fields") or {}
    return {
        "to": [_text(a) for a in _as_list(fields.get("to")) if _text(a)],
        "cc": [_text(a) for a in _as_list(fields.get("cc")) if _text(a)],
        "bcc": [_text(a) for a in _as_list(fields.get("bcc")) if _text(a)],
        "subject": _text(fields.get("subject")),
        "body": _text(fields.get("body")),
    }


def draft_to_event(draft):
    """Normalise a stored calendar draft into the fields every profile needs."""
    fields = (draft or {}).get("fields") or {}
    return {
        "title": _text(fields.get("title")),
        "start": _text(fields.get("start")),
        "end": _text(fields.get("end")),
        "location": _text(fields.get("location")),
        "description": _text(fields.get("description")),
        "attendees": [_text(a) for a in _as_list(fields.get("attendees"))
                      if _text(a)],
        "calendar": _text(fields.get("calendar")) or "primary",
    }


def _rfc822(message):
    """Render a mail draft as an RFC 822 message for Gmail's ``raw`` upload."""
    mail = EmailMessage()
    mail["To"] = ", ".join(message["to"])
    if message["cc"]:
        mail["Cc"] = ", ".join(message["cc"])
    if message["bcc"]:
        mail["Bcc"] = ", ".join(message["bcc"])
    mail["Subject"] = message["subject"]
    mail.set_content(message["body"] or "")
    return mail.as_string()


# ── profiles ───────────────────────────────────────────────────────────────
class _Profile:
    """One provider's request/parse mapping.

    Each ``route_*`` method returns ``(method, url, params, body)``; each
    ``parse_*`` method turns the provider payload into the connector's typed
    shape. No authority decisions are made here.
    """

    name = ""

    def __init__(self, config=None):
        self.config = config or {}

    # -- directory (contacts / calendars) --
    def route_directory(self, service, account, query, limit):
        raise NotImplementedError

    def parse_directory(self, service, payload):
        return []

    # -- mail --
    def route_list_messages(self, account, query, limit, unread_only):
        raise NotImplementedError

    def parse_list_messages(self, payload):
        return []

    def route_get_message(self, account, message_id):
        raise NotImplementedError

    def parse_get_message(self, payload):
        return {}

    def route_send_message(self, account, message):
        raise NotImplementedError

    def parse_send_message(self, payload):
        return {}

    # -- calendar --
    def route_list_events(self, account, start, end, limit):
        raise NotImplementedError

    def parse_list_events(self, payload):
        return []

    def route_free_busy(self, account, start, end):
        raise NotImplementedError

    def parse_free_busy(self, payload, account):
        return {"account": account, "busy": []}

    def route_create_event(self, account, event):
        raise NotImplementedError

    def parse_create_event(self, payload):
        return {}


class GoogleProfile(_Profile):
    """Gmail + Google Calendar + People API."""

    name = "google"
    MAIL = "https://gmail.googleapis.com/gmail/v1/users/me"
    CALENDAR = "https://www.googleapis.com/calendar/v3"
    PEOPLE = "https://people.googleapis.com/v1"

    def route_directory(self, service, account, query, limit):
        if service == "contacts":
            return ("GET", self.PEOPLE + "/people:searchContacts",
                    {"query": query or "", "pageSize": int(limit),
                     "readMask": "names,emailAddresses"},
                    None)
        return ("GET", self.CALENDAR + "/users/me/calendarList",
                {"maxResults": int(limit)}, None)

    def parse_directory(self, service, payload):
        payload = payload if isinstance(payload, dict) else {}
        if service == "contacts":
            out = []
            for item in _as_list(payload.get("results")):
                person = (item or {}).get("person") or {}
                names = _as_list(person.get("names"))
                emails = _as_list(person.get("emailAddresses"))
                out.append({
                    "id": _text(person.get("resourceName")) or _address_of(
                        emails[0] if emails else ""),
                    "name": _text((names[0] or {}).get("displayName"))
                            if names else "",
                    "email": _address_of(emails[0]) if emails else "",
                    "kind": "contact",
                })
            return out
        out = []
        for item in _as_list(payload.get("items")):
            out.append({
                "id": _text((item or {}).get("id")),
                "name": _text((item or {}).get("summary")),
                "email": "",
                "kind": "calendar",
                "primary": bool((item or {}).get("primary")),
            })
        return out

    def route_list_messages(self, account, query, limit, unread_only):
        params = {"maxResults": int(limit)}
        parts = [query or ""]
        if unread_only:
            parts.append("is:unread")
        joined = " ".join(p for p in parts if p).strip()
        if joined:
            params["q"] = joined
        return ("GET", self.MAIL + "/messages", params, None)

    def parse_list_messages(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        return [{"id": _text((item or {}).get("id")),
                 "thread_id": _text((item or {}).get("threadId")),
                 "subject": "", "from": "", "to": [], "date": "",
                 "snippet": "", "unread": None}
                for item in _as_list(payload.get("messages"))]

    def route_get_message(self, account, message_id):
        return ("GET", "%s/messages/%s" % (self.MAIL, quote(str(message_id))),
                {"format": "full"}, None)

    def parse_get_message(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        headers = {}
        body = ""
        part = payload.get("payload") or {}
        for header in _as_list(part.get("headers")):
            name = _text((header or {}).get("name")).lower()
            if name:
                headers[name] = _text((header or {}).get("value"))
        body = _b64url_decode((part.get("body") or {}).get("data"))
        if not body:
            for sub in _as_list(part.get("parts")):
                data = ((sub or {}).get("body") or {}).get("data")
                if data:
                    body = _b64url_decode(data)
                    break
        labels = [_text(label) for label in _as_list(payload.get("labelIds"))]
        return {
            "id": _text(payload.get("id")),
            "thread_id": _text(payload.get("threadId")),
            "subject": headers.get("subject", ""),
            "from": headers.get("from", ""),
            "to": [part.strip() for part in headers.get("to", "").split(",")
                   if part.strip()],
            "date": headers.get("date", ""),
            "snippet": _text(payload.get("snippet")),
            "body": body,
            "labels": labels,
            "unread": "UNREAD" in labels,
        }

    def route_send_message(self, account, message):
        raw = _b64url_encode(_rfc822(message))
        return ("POST", self.MAIL + "/messages/send", None, {"raw": raw})

    def parse_send_message(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        return {
            "id": _text(payload.get("id")),
            "thread_id": _text(payload.get("threadId")),
            "sent": True,
            "to": [],
        }

    def route_list_events(self, account, start, end, limit):
        params = {"maxResults": int(limit), "singleEvents": "true",
                  "orderBy": "startTime"}
        if start:
            params["timeMin"] = start
        if end:
            params["timeMax"] = end
        return ("GET", "%s/calendars/%s/events" % (
            self.CALENDAR, quote(account or "primary")), params, None)

    def parse_list_events(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        out = []
        for item in _as_list(payload.get("items")):
            item = item or {}
            out.append(_event(
                item.get("id"), item.get("summary"),
                (item.get("start") or {}).get("dateTime")
                or (item.get("start") or {}).get("date"),
                (item.get("end") or {}).get("dateTime")
                or (item.get("end") or {}).get("date"),
                item.get("location"),
                [_address_of(a) for a in _as_list(item.get("attendees"))],
                item.get("status")))
        return out

    def route_free_busy(self, account, start, end):
        body = {"timeMin": start, "timeMax": end,
                "items": [{"id": account or "primary"}]}
        return ("POST", self.CALENDAR + "/freeBusy", None, body)

    def parse_free_busy(self, payload, account):
        payload = payload if isinstance(payload, dict) else {}
        calendars = payload.get("calendars") or {}
        entry = calendars.get(account) or {}
        if not entry and calendars:
            entry = list(calendars.values())[0]
        busy = [{"start": _text((slot or {}).get("start")),
                 "end": _text((slot or {}).get("end")),
                 "status": "busy"}
                for slot in _as_list((entry or {}).get("busy"))]
        return {"account": account, "busy": busy}

    def route_create_event(self, account, event):
        body = {"summary": event["title"]}
        if event["start"]:
            body["start"] = {"dateTime": event["start"]}
        if event["end"]:
            body["end"] = {"dateTime": event["end"]}
        if event["location"]:
            body["location"] = event["location"]
        if event["description"]:
            body["description"] = event["description"]
        if event["attendees"]:
            body["attendees"] = [{"email": a} for a in event["attendees"]]
        return ("POST", "%s/calendars/%s/events" % (
            self.CALENDAR, quote(account or "primary")), None, body)

    def parse_create_event(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        return {
            "id": _text(payload.get("id")),
            "title": _text(payload.get("summary")),
            "start": _text(((payload.get("start") or {}).get("dateTime"))
                           or ((payload.get("start") or {}).get("date"))),
            "end": _text(((payload.get("end") or {}).get("dateTime"))
                         or ((payload.get("end") or {}).get("date"))),
            "created": True,
        }


class MicrosoftProfile(_Profile):
    """Microsoft Graph mail + calendar + contacts."""

    name = "microsoft"
    GRAPH = "https://graph.microsoft.com/v1.0"

    def route_directory(self, service, account, query, limit):
        if service == "contacts":
            params = {"$top": int(limit)}
            if query:
                params["$search"] = '"%s"' % query
            return ("GET", self.GRAPH + "/me/contacts", params, None)
        return ("GET", self.GRAPH + "/me/calendars", {"$top": int(limit)},
                None)

    def parse_directory(self, service, payload):
        out = []
        for item in _as_list((payload or {}).get("value")):
            item = item or {}
            emails = _as_list(item.get("emailAddresses"))
            out.append({
                "id": _text(item.get("id")),
                "name": _text(item.get("displayName") or item.get("name")),
                "email": _address_of(emails[0]) if emails else "",
                "kind": "contact" if service == "contacts" else "calendar",
                "primary": bool(item.get("isDefaultCalendar")),
            })
        return out

    def route_list_messages(self, account, query, limit, unread_only):
        params = {"$top": int(limit)}
        filters = []
        if query:
            params["$search"] = '"%s"' % query
        if unread_only:
            filters.append("isRead eq false")
        if filters:
            params["$filter"] = " and ".join(filters)
        return ("GET", self.GRAPH + "/me/messages", params, None)

    def parse_list_messages(self, payload):
        out = []
        for item in _as_list((payload or {}).get("value")):
            item = item or {}
            out.append({
                "id": _text(item.get("id")),
                "thread_id": _text(item.get("conversationId")),
                "subject": _text(item.get("subject")),
                "from": _address_of(item.get("from")),
                "to": [_address_of(r) for r in _as_list(item.get("toRecipients"))],
                "date": _text(item.get("receivedDateTime")),
                "snippet": _text(item.get("bodyPreview")),
                "unread": (not item.get("isRead"))
                          if item.get("isRead") is not None else None,
            })
        return out

    def route_get_message(self, account, message_id):
        return ("GET", "%s/me/messages/%s" % (self.GRAPH, quote(str(message_id))),
                None, None)

    def parse_get_message(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        body = _text((payload.get("body") or {}).get("content"))
        return {
            "id": _text(payload.get("id")),
            "thread_id": _text(payload.get("conversationId")),
            "subject": _text(payload.get("subject")),
            "from": _address_of(payload.get("from")),
            "to": [_address_of(r) for r in _as_list(payload.get("toRecipients"))],
            "date": _text(payload.get("receivedDateTime")),
            "snippet": _text(payload.get("bodyPreview")),
            "body": body,
            "labels": [],
            "unread": (not payload.get("isRead"))
                      if payload.get("isRead") is not None else None,
        }

    def route_send_message(self, account, message):
        body = {
            "message": {
                "subject": message["subject"],
                "body": {"contentType": "Text", "content": message["body"]},
                "toRecipients": [
                    {"emailAddress": {"address": a}} for a in message["to"]],
                "ccRecipients": [
                    {"emailAddress": {"address": a}} for a in message["cc"]],
                "bccRecipients": [
                    {"emailAddress": {"address": a}} for a in message["bcc"]],
            },
            "saveToSentItems": True,
        }
        return ("POST", self.GRAPH + "/me/sendMail", None, body)

    def parse_send_message(self, payload):
        # Graph answers 202 with an empty body: the effect happened, there is
        # no id to report.
        return {"id": "", "thread_id": "", "sent": True, "to": []}

    def route_list_events(self, account, start, end, limit):
        params = {"$top": int(limit), "$orderby": "start/dateTime"}
        if start:
            params["startDateTime"] = start
        if end:
            params["endDateTime"] = end
        return ("GET", self.GRAPH + "/me/events", params, None)

    def parse_list_events(self, payload):
        out = []
        for item in _as_list((payload or {}).get("value")):
            item = item or {}
            out.append(_event(
                item.get("id"), item.get("subject"),
                (item.get("start") or {}).get("dateTime"),
                (item.get("end") or {}).get("dateTime"),
                (item.get("location") or {}).get("displayName"),
                [_address_of(a) for a in _as_list(item.get("attendees"))],
                item.get("showAs")))
        return out

    def route_free_busy(self, account, start, end):
        body = {"schedules": [account] if account else [],
                "startTime": {"dateTime": start, "timeZone": "UTC"},
                "endTime": {"dateTime": end, "timeZone": "UTC"}}
        return ("POST", self.GRAPH + "/me/calendar/getSchedule", None, body)

    def parse_free_busy(self, payload, account):
        busy = []
        for schedule in _as_list((payload or {}).get("value")):
            for item in _as_list((schedule or {}).get("scheduleItems")):
                item = item or {}
                busy.append({
                    "start": _text((item.get("start") or {}).get("dateTime")),
                    "end": _text((item.get("end") or {}).get("dateTime")),
                    "status": _text(item.get("status")) or "busy",
                })
        return {"account": account, "busy": busy}

    def route_create_event(self, account, event):
        body = {
            "subject": event["title"],
            "start": {"dateTime": event["start"], "timeZone": "UTC"},
            "end": {"dateTime": event["end"], "timeZone": "UTC"},
            "body": {"contentType": "Text", "content": event["description"]},
            "attendees": [
                {"emailAddress": {"address": a}, "type": "required"}
                for a in event["attendees"]],
        }
        if event["location"]:
            body["location"] = {"displayName": event["location"]}
        return ("POST", self.GRAPH + "/me/events", None, body)

    def parse_create_event(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        return {
            "id": _text(payload.get("id")),
            "title": _text(payload.get("subject")),
            "start": _text((payload.get("start") or {}).get("dateTime")),
            "end": _text((payload.get("end") or {}).get("dateTime")),
            "created": True,
        }


class CustomProfile(_Profile):
    """Declarative REST profile for a service we did not model.

    Endpoints come from the user's own service configuration; the JSON shape
    is the connector's documented one, so "confirm actual user services"
    never turns into "add a provider we guessed at".
    """

    name = "custom"

    def __init__(self, config=None):
        super().__init__(config)
        self.base = _text((config or {}).get("base_url")).rstrip("/")
        self.endpoints = dict((config or {}).get("endpoints") or {})

    def _url(self, key, default):
        path = _text(self.endpoints.get(key)) or default
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return self.base + path

    def route_directory(self, service, account, query, limit):
        return ("GET", self._url("directory", "/directory"),
                {"query": query or "", "limit": int(limit)}, None)

    def parse_directory(self, service, payload):
        items = payload
        if isinstance(payload, dict):
            items = payload.get("items") or payload.get("value") or []
        return [{
            "id": _text((item or {}).get("id")),
            "name": _text((item or {}).get("name")),
            "email": _text((item or {}).get("email")),
            "kind": _text((item or {}).get("kind")) or service,
            "primary": bool((item or {}).get("primary")),
        } for item in _as_list(items)]

    def route_list_messages(self, account, query, limit, unread_only):
        return ("GET", self._url("messages", "/messages"),
                {"query": query or "", "limit": int(limit),
                 "unread_only": "true" if unread_only else "false"}, None)

    def parse_list_messages(self, payload):
        items = payload
        if isinstance(payload, dict):
            items = payload.get("items") or payload.get("value") or []
        out = []
        for item in _as_list(items):
            item = item or {}
            out.append({
                "id": _text(item.get("id")),
                "thread_id": _text(item.get("thread_id")),
                "subject": _text(item.get("subject")),
                "from": _text(item.get("from")),
                "to": list(item.get("to") or []),
                "date": _text(item.get("date")),
                "snippet": _text(item.get("snippet")),
                "unread": item.get("unread"),
            })
        return out

    def route_get_message(self, account, message_id):
        return ("GET",
                self._url("message", "/messages/{id}").replace(
                    "{id}", quote(str(message_id))),
                None, None)

    def parse_get_message(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        return {
            "id": _text(payload.get("id")),
            "thread_id": _text(payload.get("thread_id")),
            "subject": _text(payload.get("subject")),
            "from": _text(payload.get("from")),
            "to": list(payload.get("to") or []),
            "date": _text(payload.get("date")),
            "snippet": _text(payload.get("snippet")),
            "body": _text(payload.get("body")),
            "labels": list(payload.get("labels") or []),
            "unread": payload.get("unread"),
        }

    def route_send_message(self, account, message):
        return ("POST", self._url("send", "/send"), None, message)

    def parse_send_message(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        return {"id": _text(payload.get("id")), "thread_id": "",
                "sent": bool(payload.get("sent", True)),
                "to": list(payload.get("to") or [])}

    def route_list_events(self, account, start, end, limit):
        return ("GET", self._url("events", "/events"),
                {"start": start or "", "end": end or "", "limit": int(limit)},
                None)

    def parse_list_events(self, payload):
        items = payload
        if isinstance(payload, dict):
            items = payload.get("items") or payload.get("value") or []
        return [_event(item.get("id"), item.get("title"), item.get("start"),
                       item.get("end"), item.get("location"),
                       item.get("attendees"), item.get("status"))
                for item in (_as_list(items)) if isinstance(item, dict)]

    def route_free_busy(self, account, start, end):
        return ("POST", self._url("freebusy", "/freebusy"), None,
                {"start": start, "end": end, "account": account})

    def parse_free_busy(self, payload, account):
        payload = payload if isinstance(payload, dict) else {}
        return {"account": account,
                "busy": [{"start": _text((slot or {}).get("start")),
                          "end": _text((slot or {}).get("end")),
                          "status": _text((slot or {}).get("status")) or "busy"}
                         for slot in _as_list(payload.get("busy"))]}

    def route_create_event(self, account, event):
        return ("POST", self._url("create_event", "/events"), None, event)

    def parse_create_event(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        return {"id": _text(payload.get("id")),
                "title": _text(payload.get("title")),
                "start": _text(payload.get("start")),
                "end": _text(payload.get("end")),
                "created": bool(payload.get("created", True))}


def _event(event_id, title, start, end, location, attendees, status):
    return {
        "id": _text(event_id),
        "title": _text(title),
        "start": _text(start),
        "end": _text(end),
        "location": _text(location),
        "attendees": [_address_of(a) for a in _as_list(attendees)],
        "status": _text(status),
    }


_PROFILES = {
    "google": GoogleProfile,
    "microsoft": MicrosoftProfile,
    "custom": CustomProfile,
}


def profile_for(provider, config=None):
    """Instantiate the profile for *provider* (``None`` when unmodelled)."""
    factory = _PROFILES.get(_text(provider).strip().lower())
    if factory is None:
        return None
    return factory(config)


def supported_providers():
    return tuple(sorted(_PROFILES))


# ── the transport ──────────────────────────────────────────────────────────
class HttpTransport:
    """Typed provider calls over HTTP.

    The connector only ever reaches this object after its authority gates
    (service confirmation, delegated scope, effect approval) have passed, so
    the methods here are deliberately unconditional: they perform the request
    or raise.
    """

    def __init__(self, provider, token, config=None, session=None,
                 timeout=DEFAULT_TIMEOUT):
        self.provider = _text(provider).strip().lower()
        self._token = _text(token)
        self.profile = profile_for(self.provider, config)
        if self.profile is None:
            raise ProviderError(
                "provider %r is not supported (known: %s)"
                % (self.provider, ", ".join(supported_providers())))
        self._session = session if session is not None else requests.Session()
        self.timeout = float(timeout or DEFAULT_TIMEOUT)

    def __repr__(self):  # never leak the token
        return "<HttpTransport provider=%s token=%s>" % (
            self.provider, "set" if self._token else "unset")

    # -- request plumbing --
    def _headers(self):
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = "Bearer %s" % self._token
        return headers

    def _request(self, route):
        method, url, params, body = route
        if not url:
            raise ProviderError(
                "the %s profile has no endpoint configured for this operation"
                % self.provider)
        if params:
            params = {key: value for key, value in params.items()
                      if value not in (None, "")}
        try:
            response = self._session.request(
                method, url, params=params or None,
                json=body if body is not None else None,
                headers=self._headers(), timeout=self.timeout)
        except Exception as exc:  # network/transport failure
            raise ProviderError("the %s request failed: %s"
                                % (self.provider, exc))
        status = getattr(response, "status_code", 0)
        if status < 200 or status >= 300:
            detail = tool_policy.mask_secrets(
                _text(getattr(response, "text", ""))[:_MAX_ERROR_BODY])
            raise ProviderError(
                "%s rejected the request (HTTP %s)%s"
                % (self.provider, status, (": " + detail) if detail else ""),
                status=status)
        try:
            return response.json()
        except Exception:
            return {}

    # -- typed operations (called only behind the connector's gates) --
    def directory(self, service, account, query="", limit=20):
        payload = self._request(self.profile.route_directory(
            service, account, query, max(1, int(limit))))
        return self.profile.parse_directory(service, payload)

    def list_messages(self, account, query="", limit=20, unread_only=False):
        payload = self._request(self.profile.route_list_messages(
            account, query, max(1, int(limit)), bool(unread_only)))
        return self.profile.parse_list_messages(payload)

    def get_message(self, account, message_id):
        payload = self._request(self.profile.route_get_message(
            account, message_id))
        return self.profile.parse_get_message(payload)

    def send_message(self, account, draft):
        """EXTERNAL EFFECT: hand a mail draft to the provider for delivery."""
        message = draft_to_message(draft)
        payload = self._request(self.profile.route_send_message(account, message))
        result = self.profile.parse_send_message(payload)
        result.setdefault("to", message["to"])
        return result

    def list_events(self, account, start="", end="", limit=20):
        payload = self._request(self.profile.route_list_events(
            account, start, end, max(1, int(limit))))
        return self.profile.parse_list_events(payload)

    def free_busy(self, account, start, end):
        payload = self._request(self.profile.route_free_busy(account, start, end))
        return self.profile.parse_free_busy(payload, account)

    def create_event(self, account, draft):
        """EXTERNAL EFFECT: create a calendar event others will see."""
        event = draft_to_event(draft)
        payload = self._request(self.profile.route_create_event(account, event))
        result = self.profile.parse_create_event(payload)
        result.setdefault("title", event["title"])
        return result


class UnconfiguredTransport:
    """The transport used when nothing has been confirmed.

    Every method refuses. This is the default so an unconfigured install can
    never fall back to "some provider we assumed".
    """

    provider = "unconfigured"

    def __repr__(self):
        return "<UnconfiguredTransport>"

    def _refuse(self):
        raise ProviderError(
            "no productivity service is configured and confirmed, so no "
            "request was made")

    def directory(self, *args, **kwargs):
        self._refuse()

    def list_messages(self, *args, **kwargs):
        self._refuse()

    def get_message(self, *args, **kwargs):
        self._refuse()

    def send_message(self, *args, **kwargs):
        self._refuse()

    def list_events(self, *args, **kwargs):
        self._refuse()

    def free_busy(self, *args, **kwargs):
        self._refuse()

    def create_event(self, *args, **kwargs):
        self._refuse()


def build_transport(provider, token, config=None, session=None, timeout=None):
    """Build the transport for a confirmed service, or a refusing one."""
    if not provider:
        return UnconfiguredTransport()
    kwargs = {}
    if timeout:
        kwargs["timeout"] = timeout
    return HttpTransport(provider, token, config=config, session=session,
                         **kwargs)


def json_dumps(payload, **kwargs):
    """Small helper so callers never persist an unserialisable payload."""
    return json.dumps(payload, ensure_ascii=False, default=str, **kwargs)
