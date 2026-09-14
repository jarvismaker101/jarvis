"""Semantic productivity connectors — calendar, mail and contacts.

Fable-5 audit G4 / F14 — the inspected interfaces covered code/editor/browser/
Windows actions, not the calendar/mail/contact operations a user actually
asks for. The correction is:

    "Confirm actual user services, then add typed least-privilege read/draft
     operations with separately approved sending and externally visible
     changes."

What this module implements, mapped onto that sentence:

* **Confirm actual user services** — nothing is assumed. Services are
  configured from evidence (``discover_services`` reports environment hints)
  and stay inert until the user confirms them (``confirm_services``). An
  unconfirmed or unauthenticated service refuses every operation, and the
  default transport (:class:`~backend.services.productivity_providers.
  UnconfiguredTransport`) refuses to make a request at all.
* **Typed** — :data:`OPERATIONS` is the single source of truth: each
  operation declares its service, effect class, required scope and argument
  types. Tool advertisement (:func:`tool_schemas`), argument validation and
  dispatch are all generated from it, so they cannot diverge. A malformed or
  unknown call refuses without touching a provider.
* **Least-privilege read/draft** — reading needs ``<service>.read``; composing
  a draft needs ``<service>.draft``. Drafts are local objects: creating one
  performs **no external effect** (no request is made), so the user can
  inspect, edit or discard it freely.
* **Separately approved sending and externally visible changes** — holding a
  ``mail.send`` / ``calendar.write`` grant is not enough. The effect needs a
  second, per-effect consent (:func:`request_effect_approval`) that is bound
  to the draft's content hash and to the grant epoch at the moment it was
  requested. :func:`commit_draft` re-verifies and consumes that approval
  immediately before executing, so an edited draft, a replayed approval, a
  cross-draft approval, an expired approval or a revoked grant all execute
  nothing.
* **Account/entity resolution** — :func:`resolve_account` maps a spoken
  account onto a configured one locally; :func:`resolve_entity` resolves
  "mom" / "the work calendar" against the provider directory (a read) and
  reports ambiguity instead of guessing.
* **Revocation** — :func:`revoke` drops scopes, bumps the grant epoch (which
  invalidates every outstanding approval bound to it) and kills pending
  approvals, durably. :func:`unconfigure_service` additionally withdraws the
  service itself.

Persistence: a small JSON file under ``data/`` (gitignored), overridable with
``JARVIS_PRODUCTIVITY_CONFIG`` so tests and alternative installs never touch
the real one. Access tokens never appear in any returned or logged payload.

This module is deliberately free of agent/planner imports (no cycles): wiring
the advertised tools (:func:`tool_schemas`) into the task planner's allowlist
and dispatch table is the remaining integration step, reported separately.
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
from datetime import datetime

from backend.config import BASE_DIR
from backend.services import productivity_providers
from backend.services import tool_policy
from backend.services.productivity_providers import ProviderError

# ── operation classes ──────────────────────────────────────────────────────
# read/draft are local or read-only; send/create reach the outside world and
# therefore need both a delegated grant AND a per-effect approval.
OPERATION_READ = "read"
OPERATION_DRAFT = "draft"
OPERATION_SEND = "send"
OPERATION_CREATE = "create"

EXTERNAL_EFFECTS = frozenset((OPERATION_SEND, OPERATION_CREATE))

SERVICES = ("mail", "calendar", "contacts")

# ── scopes (least privilege: one scope per capability, no wildcards) ───────
MAIL_READ = "mail.read"
MAIL_DRAFT = "mail.draft"
MAIL_SEND = "mail.send"
CALENDAR_READ = "calendar.read"
CALENDAR_DRAFT = "calendar.draft"
CALENDAR_WRITE = "calendar.write"
CONTACTS_READ = "contacts.read"

ALL_SCOPES = frozenset((
    MAIL_READ, MAIL_DRAFT, MAIL_SEND,
    CALENDAR_READ, CALENDAR_DRAFT, CALENDAR_WRITE,
    CONTACTS_READ,
))

SERVICE_SCOPES = {
    "mail": frozenset((MAIL_READ, MAIL_DRAFT, MAIL_SEND)),
    "calendar": frozenset((CALENDAR_READ, CALENDAR_DRAFT, CALENDAR_WRITE)),
    "contacts": frozenset((CONTACTS_READ,)),
}

#: Scopes whose exercise is externally visible. Granting one requires an
#: explicit acknowledgement, and USING one additionally requires a per-effect
#: approval — the "separately approved sending" half of the correction.
ELEVATED_SCOPES = frozenset((MAIL_SEND, CALENDAR_WRITE))

#: The effect a committed draft actually performs. It is always one of these;
#: anything else is refused rather than guessed.
EXTERNAL_EFFECT_VALUES = frozenset((MAIL_SEND, CALENDAR_WRITE))

CONFIG_ENV = "JARVIS_PRODUCTIVITY_CONFIG"
DEFAULT_CONFIG_PATH = os.path.join(str(BASE_DIR), "data",
                                   "productivity_services.json")

APPROVAL_TTL_SECONDS = 120.0
DEFAULT_READ_LIMIT = 10
MAX_READ_LIMIT = 50

#: Environment hints used by :func:`discover_services`. Discovery only
#: *proposes* a service; nothing is adopted until the user confirms it.
ENV_HINTS = (
    ("mail", "google", "JARVIS_GMAIL_TOKEN"),
    ("mail", "google", "JARVIS_GOOGLE_ACCESS_TOKEN"),
    ("mail", "microsoft", "JARVIS_GRAPH_TOKEN"),
    ("calendar", "google", "JARVIS_GOOGLE_CALENDAR_TOKEN"),
    ("calendar", "google", "JARVIS_GOOGLE_ACCESS_TOKEN"),
    ("calendar", "microsoft", "JARVIS_GRAPH_TOKEN"),
    ("contacts", "google", "JARVIS_GOOGLE_PEOPLE_TOKEN"),
    ("contacts", "google", "JARVIS_GOOGLE_ACCESS_TOKEN"),
    ("contacts", "microsoft", "JARVIS_GRAPH_TOKEN"),
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?"
    r"(Z|[+-]\d{2}:?\d{2})?)?$")

_LOCK = threading.RLock()


# ── typed operation registry ───────────────────────────────────────────────
def _arg(kind, doc="", required=False, enum=None, default=None):
    spec = {"type": kind, "doc": doc, "required": required}
    if enum:
        spec["enum"] = list(enum)
    if default is not None:
        spec["default"] = default
    return spec


def _op(service, operation, scope, summary, args, effect="", validator=None):
    required = tuple(name for name, spec in args.items() if spec.get("required"))
    return {
        "service": service,
        "operation": operation,
        "scope": scope,
        "summary": summary,
        "args": args,
        "required": required,
        "effect": effect,
        "validator": validator,
    }


OPERATIONS = {
    "productivity.services": _op(
        None, OPERATION_READ, None,
        "List the productivity services this user has actually confirmed, "
        "with the scopes delegated for each account (never a token).",
        {}),
    "mail.list_messages": _op(
        "mail", OPERATION_READ, MAIL_READ,
        "List recent mail messages (read-only).",
        {"account": _arg("string", "Configured mail account.", required=True),
         "query": _arg("string", "Optional search text."),
         "limit": _arg("integer", "Max messages (1-%d)." % MAX_READ_LIMIT),
         "unread_only": _arg("boolean", "Only unread messages.")}),
    "mail.read_message": _op(
        "mail", OPERATION_READ, MAIL_READ,
        "Read one message by id (read-only).",
        {"account": _arg("string", "Configured mail account.", required=True),
         "message_id": _arg("string", "Provider message id.", required=True)}),
    "mail.create_draft": _op(
        "mail", OPERATION_DRAFT, MAIL_DRAFT,
        "Compose a mail draft (local only — nothing is sent).",
        {"account": _arg("string", "Configured mail account.", required=True),
         "to": _arg("array", "Recipient addresses (resolve names first).",
                    required=True),
         "cc": _arg("array", "Cc addresses."),
         "bcc": _arg("array", "Bcc addresses."),
         "subject": _arg("string", "Subject line."),
         "body": _arg("string", "Plain-text body.")},
        effect=MAIL_SEND, validator="mail"),
    "mail.send_draft": _op(
        "mail", OPERATION_SEND, MAIL_SEND,
        "Send a previously composed draft. Needs a separate per-draft "
        "approval id.",
        {"account": _arg("string", "Configured mail account.", required=True),
         "draft_id": _arg("string", "Draft to send.", required=True),
         "approval_id": _arg("string",
                             "Approval bound to this draft's content.",
                             required=True)},
        effect=MAIL_SEND),
    "calendar.list_events": _op(
        "calendar", OPERATION_READ, CALENDAR_READ,
        "List calendar events (read-only).",
        {"account": _arg("string", "Configured calendar account.",
                         required=True),
         "start": _arg("string", "ISO start bound (inclusive)."),
         "end": _arg("string", "ISO end bound (exclusive)."),
         "limit": _arg("integer", "Max events (1-%d)." % MAX_READ_LIMIT)}),
    "calendar.check_availability": _op(
        "calendar", OPERATION_READ, CALENDAR_READ,
        "Free/busy lookup for a window (read-only).",
        {"account": _arg("string", "Configured calendar account.",
                         required=True),
         "start": _arg("string", "ISO window start.", required=True),
         "end": _arg("string", "ISO window end.", required=True)}),
    "calendar.propose_event": _op(
        "calendar", OPERATION_DRAFT, CALENDAR_DRAFT,
        "Propose a calendar event (local only — nothing is created).",
        {"account": _arg("string", "Configured calendar account.",
                         required=True),
         "title": _arg("string", "Event title.", required=True),
         "start": _arg("string", "ISO start.", required=True),
         "end": _arg("string", "ISO end.", required=True),
         "attendees": _arg("array", "Attendee addresses."),
         "location": _arg("string", "Location."),
         "description": _arg("string", "Notes."),
         "calendar": _arg("string", "Target calendar id.")},
        effect=CALENDAR_WRITE, validator="event"),
    "calendar.commit_event": _op(
        "calendar", OPERATION_CREATE, CALENDAR_WRITE,
        "Create a proposed event on the calendar. Needs a separate approval "
        "id.",
        {"account": _arg("string", "Configured calendar account.",
                         required=True),
         "draft_id": _arg("string", "Proposal to create.", required=True),
         "approval_id": _arg("string",
                             "Approval bound to this proposal's content.",
                             required=True)},
        effect=CALENDAR_WRITE),
    "contacts.search": _op(
        "contacts", OPERATION_READ, CONTACTS_READ,
        "Search the contact directory (read-only).",
        {"account": _arg("string", "Configured contacts account.",
                         required=True),
         "query": _arg("string", "Search text.", required=True),
         "limit": _arg("integer", "Max results (1-%d)." % MAX_READ_LIMIT)}),
    "contacts.resolve": _op(
        "contacts", OPERATION_READ, CONTACTS_READ,
        "Resolve a spoken name to one contact; ambiguity is reported, never "
        "guessed.",
        {"account": _arg("string", "Configured contacts account.",
                         required=True),
         "query": _arg("string", "Name, address or id.", required=True)}),
    "calendar.resolve": _op(
        "calendar", OPERATION_READ, CALENDAR_READ,
        "Resolve a spoken calendar name to one calendar id.",
        {"account": _arg("string", "Configured calendar account.",
                         required=True),
         "query": _arg("string", "Calendar name or id.", required=True)}),
}

TOOL_NAMES = tuple(sorted(OPERATIONS))


def is_productivity_tool(name):
    """True when *name* is one of the typed operations registered here."""
    return str(name or "").strip() in OPERATIONS


def planner_tool_lines():
    """Advertisement lines for the task planner's tool list.

    The planner lives in ``task_agent/agent.py`` (owned by other work); this
    is the plumbing it needs to advertise the connector without duplicating
    the registry or its schemas.
    """
    lines = []
    for name in TOOL_NAMES:
        spec = OPERATIONS[name]
        required = ", ".join("args.%s" % arg for arg in spec["required"])
        suffix = ""
        if spec["operation"] in EXTERNAL_EFFECTS:
            suffix = (" — externally visible: needs a separate approval id "
                      "from request_effect_approval()")
        elif spec["operation"] == OPERATION_DRAFT:
            suffix = " — local draft, nothing is sent or created"
        if required:
            suffix = " (%s)%s" % (required, suffix)
        lines.append("- %s: %s%s" % (name, spec["summary"], suffix))
    return lines


def tool_schemas():
    """Advertise the typed operations (generated from :data:`OPERATIONS`)."""
    schemas = {}
    for name, spec in OPERATIONS.items():
        properties = {}
        for arg_name, arg in spec["args"].items():
            entry = {"type": arg["type"], "description": arg.get("doc", "")}
            if arg.get("enum"):
                entry["enum"] = list(arg["enum"])
            properties[arg_name] = entry
        schemas[name] = {
            "description": spec["summary"],
            "operation": spec["operation"],
            "external_effect": spec["operation"] in EXTERNAL_EFFECTS,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(spec["required"]),
                "additionalProperties": False,
            },
        }
    return schemas


# ── config store ───────────────────────────────────────────────────────────
def config_path():
    configured = os.getenv(CONFIG_ENV, "").strip()
    if configured:
        return os.path.abspath(configured)
    return DEFAULT_CONFIG_PATH


def _empty_config():
    return {"services": {}, "grants": [], "epochs": {}, "drafts": [],
            "approvals": []}


def _load():
    """Read the store; a missing/corrupt file means 'nothing confirmed'."""
    try:
        with open(config_path(), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return _empty_config()
    if not isinstance(data, dict):
        return _empty_config()
    config = _empty_config()
    for key in config:
        value = data.get(key)
        if isinstance(value, type(config[key])):
            config[key] = value
    return config


def _save(config):
    """Atomic write; never raises. Returns True on success."""
    path = config_path()
    directory = os.path.dirname(path) or "."
    tmp = "%s.%d.tmp" % (path, os.getpid())
    try:
        os.makedirs(directory, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(config, ensure_ascii=False, default=str))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logging.debug("[F14] could not persist productivity config: %s", exc)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


#: How long a spent/revoked approval is kept so a replay is refused with an
#: accurate reason ("already used") instead of "unknown approval".
_APPROVAL_HISTORY_SECONDS = 3600.0


def _num(value, default=0.0):
    """A tolerant numeric read: a hand-edited store must never crash a gate."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _prune(config, now=None):
    """Drop approvals that can never be used again."""
    now = time.time() if now is None else now
    approvals = []
    for record in config.get("approvals") or []:
        if not isinstance(record, dict):
            continue
        if record.get("consumed") or record.get("revoked"):
            spent_at = _num(record.get("consumed_at")
                            or record.get("revoked_at"), now)
            if spent_at + _APPROVAL_HISTORY_SECONDS > now:
                approvals.append(record)
            continue
        expires_at = _num(record.get("expires_at"))
        if expires_at <= now:
            # An expired approval is kept for the same short window so a late
            # attempt is refused with "expired" rather than "unknown".
            if expires_at + _APPROVAL_HISTORY_SECONDS > now:
                approvals.append(record)
            continue
        approvals.append(record)
    config["approvals"] = approvals
    drafts = [d for d in (config.get("drafts") or []) if isinstance(d, dict)]
    config["drafts"] = drafts[-100:]
    return config


# ── account identity ───────────────────────────────────────────────────────
def _normalize(value):
    return " ".join(str(value or "").strip().casefold().split())


def _account_entries(entry):
    """The account list of one service, always non-empty when configured."""
    accounts = [a for a in (entry.get("accounts") or []) if isinstance(a, dict)]
    if not accounts and entry.get("account"):
        accounts = [{"id": entry["account"], "label": "", "primary": True}]
    return accounts


def _primary_account(entry):
    accounts = _account_entries(entry)
    for account in accounts:
        if account.get("primary"):
            return account
    return accounts[0] if accounts else {}


def resolve_account(service, query=None):
    """Resolve a spoken account name to one configured account (no network).

    Local by design: identity resolution must not need a provider round trip,
    and it must never invent an account the user did not configure.
    """
    service = str(service or "").strip().lower()
    with _LOCK:
        config = _load()
    entry = (config.get("services") or {}).get(service)
    if not entry:
        return {"ok": False, "status": "unconfigured", "service": service,
                "account": None, "candidates": [],
                "reason": "no %s service is configured" % (service or "?")}
    accounts = _account_entries(entry)
    if not query or not str(query).strip():
        primary = _primary_account(entry)
        return {"ok": True, "status": "resolved", "service": service,
                "account": primary.get("id"), "candidates": [primary],
                "reason": ""}
    wanted = _normalize(query)
    exact = [a for a in accounts
             if wanted in (_normalize(a.get("id")), _normalize(a.get("label")))]
    if len(exact) == 1:
        return {"ok": True, "status": "resolved", "service": service,
                "account": exact[0].get("id"), "candidates": exact, "reason": ""}
    if len(exact) > 1:
        return {"ok": False, "status": "ambiguous", "service": service,
                "account": None, "candidates": exact,
                "reason": "%d accounts match %r" % (len(exact), query)}
    partial = [a for a in accounts
               if wanted in _normalize(a.get("id"))
               or wanted in _normalize(a.get("label"))]
    if len(partial) == 1:
        return {"ok": True, "status": "resolved", "service": service,
                "account": partial[0].get("id"), "candidates": partial,
                "reason": ""}
    if len(partial) > 1:
        return {"ok": False, "status": "ambiguous", "service": service,
                "account": None, "candidates": partial,
                "reason": "%d accounts match %r" % (len(partial), query)}
    return {"ok": False, "status": "unresolved", "service": service,
            "account": None, "candidates": accounts,
            "reason": "no configured %s account matches %r" % (service, query)}


def _require_account(service, account, entry):
    """(ok, resolved_account_or_None, reason)."""
    accounts = _account_entries(entry)
    if not accounts:
        return False, None, "no account is configured for %s" % service
    if not account or not str(account).strip():
        primary = _primary_account(entry)
        return (bool(primary.get("id")), primary.get("id"),
                "" if primary.get("id") else
                "no default %s account is configured" % service)
    wanted = _normalize(account)
    for candidate in accounts:
        if wanted in (_normalize(candidate.get("id")),
                      _normalize(candidate.get("label"))):
            return True, candidate.get("id"), ""
    return False, None, (
        "%r is not a configured %s account (configured: %s)"
        % (account, service,
           ", ".join(str(a.get("id")) for a in accounts)))


# ── service configuration & confirmation ───────────────────────────────────
def configure_service(service, provider, account, token=None, token_env="",
                      accounts=None, base_url="", endpoints=None,
                      confirmed=False, now=None):
    """Record a service the user says they use.

    The service stays inert until it is confirmed (``confirmed=True`` or a
    later :func:`confirm_services`), so nothing is ever assumed about which
    provider or account the user actually has.
    """
    service = str(service or "").strip().lower()
    provider = str(provider or "").strip().lower()
    if service not in SERVICES:
        return {"ok": False, "error": "unknown service %r (known: %s)"
                % (service, ", ".join(SERVICES))}
    if provider not in productivity_providers.supported_providers():
        return {"ok": False, "error":
                "unsupported provider %r (known: %s)"
                % (provider,
                   ", ".join(productivity_providers.supported_providers()))}
    account = str(account or "").strip()
    if not account:
        return {"ok": False,
                "error": "an account id is required so permissions can be "
                         "delegated and revoked per account"}
    if provider == "custom" and not str(base_url or "").strip():
        return {"ok": False,
                "error": "the custom provider needs a base_url"}

    entries = []
    for index, candidate in enumerate(accounts or []):
        if isinstance(candidate, str):
            candidate = {"id": candidate}
        if not isinstance(candidate, dict) or not str(candidate.get("id") or "").strip():
            continue
        entries.append({
            "id": str(candidate["id"]).strip(),
            "label": str(candidate.get("label") or "").strip(),
            "primary": bool(candidate.get("primary")) or index == 0,
        })
    if not entries:
        entries = [{"id": account, "label": "", "primary": True}]
    if not any(a["id"] == account for a in entries):
        entries.insert(0, {"id": account, "label": "", "primary": True})
    if not any(a.get("primary") for a in entries):
        entries[0]["primary"] = True

    now = time.time() if now is None else now
    with _LOCK:
        config = _prune(_load(), now)
        previous = config["services"].get(service) or {}
        config["services"][service] = {
            "provider": provider,
            "account": account,
            "accounts": entries,
            "token": str(token or ""),
            "token_env": str(token_env or "").strip(),
            "base_url": str(base_url or "").strip(),
            "endpoints": dict(endpoints or {}),
            "confirmed": bool(confirmed) or bool(previous.get("confirmed")),
            "configured_at": previous.get("configured_at") or now,
            "confirmed_at": previous.get("confirmed_at")
                            or (now if confirmed else None),
        }
        _save(config)
    return {"ok": True, "service": service, "provider": provider,
            "account": account, "confirmed": bool(confirmed),
            # Named "has_key" (model_registry's convention) rather than
            # "authenticated"/"has_credentials": those names match the secret
            # key patterns, so the hygiene pass would mask the flag itself.
            "has_key": _has_credentials(
                config["services"][service]),
            "note": ("confirmed — operations may run"
                     if confirmed else
                     "configured but NOT confirmed; every operation refuses "
                     "until the user confirms this service")}


def confirm_services(services, confirmed=True, now=None):
    """Confirm (or withdraw confirmation for) the user's actual services."""
    if isinstance(services, str):
        services = [services]
    now = time.time() if now is None else now
    changed = []
    with _LOCK:
        config = _prune(_load(), now)
        for name in services or []:
            service = str(name or "").strip().lower()
            entry = config["services"].get(service)
            if not entry:
                continue
            entry["confirmed"] = bool(confirmed)
            entry["confirmed_at"] = now if confirmed else None
            changed.append(service)
        _save(config)
    return {"ok": True, "changed": sorted(changed), "confirmed": bool(confirmed)}


def unconfigure_service(service, now=None):
    """Withdraw a service entirely: drop it, its grants and its approvals."""
    service = str(service or "").strip().lower()
    now = time.time() if now is None else now
    with _LOCK:
        config = _prune(_load(), now)
        existed = config["services"].pop(service, None)
        revoked = _revoke_locked(config, service, None, None, now)
        config["drafts"] = [d for d in config["drafts"]
                            if d.get("service") != service]
        _save(config)
    return {"ok": bool(existed), "service": service,
            "revoked": revoked["revoked"],
            "invalidated_approvals": revoked["invalidated_approvals"],
            "error": "" if existed else "no %s service was configured" % service}


def _has_credentials(entry):
    if not entry:
        return False
    if str(entry.get("token") or "").strip():
        return True
    token_env = str(entry.get("token_env") or "").strip()
    return bool(token_env and os.getenv(token_env, "").strip())


def _token_for(entry):
    token = str((entry or {}).get("token") or "").strip()
    if token:
        return token
    token_env = str((entry or {}).get("token_env") or "").strip()
    if token_env:
        return os.getenv(token_env, "").strip()
    return ""


def discover_services():
    """Propose services from the environment — never adopt them."""
    with _LOCK:
        config = _load()
    hints = []
    seen = set()
    for service, provider, env_var in ENV_HINTS:
        if (service, provider, env_var) in seen:
            continue
        seen.add((service, provider, env_var))
        if os.getenv(env_var, "").strip():
            hints.append({"service": service, "provider": provider,
                          "env_var": env_var})
    return {
        "configured": sorted(config["services"]),
        "confirmed": sorted(name for name, entry in config["services"].items()
                            if entry.get("confirmed")),
        "env_hints": hints,
        "supported_providers": list(productivity_providers.supported_providers()),
        "note": ("Discovery only reports evidence. Call configure_service() "
                 "and then confirm_services() for the services the user says "
                 "they actually use; until then every operation refuses."),
    }


def service_inventory():
    """What is actually available — configuration only, no network."""
    with _LOCK:
        config = _prune(_load(), time.time())
    out = []
    for service in SERVICES:
        entry = config["services"].get(service)
        granted = {}
        for grant_record in config["grants"]:
            if grant_record.get("service") == service:
                granted[grant_record.get("account")] = sorted(
                    grant_record.get("scopes") or [])
        if not entry:
            out.append({"service": service, "configured": False,
                        "confirmed": False, "has_key": False,
                        "provider": "", "accounts": [], "granted_scopes": {},
                        "operations": sorted(
                            name for name, spec in OPERATIONS.items()
                            if spec["service"] == service),
                        "setup_hint": ("Nothing is configured for %s; ask the "
                                       "user which service they use." % service)})
            continue
        out.append({
            "service": service,
            "configured": True,
            "confirmed": bool(entry.get("confirmed")),
            "has_key": _has_credentials(entry),
            "provider": entry.get("provider", ""),
            "accounts": [{"id": a.get("id"), "label": a.get("label"),
                          "primary": bool(a.get("primary"))}
                         for a in _account_entries(entry)],
            "granted_scopes": granted,
            "operations": sorted(name for name, spec in OPERATIONS.items()
                                 if spec["service"] == service),
            "elevated_operations": sorted(
                name for name, spec in OPERATIONS.items()
                if spec["service"] == service
                and spec["operation"] in EXTERNAL_EFFECTS),
            "setup_hint": "",
        })
    return tool_policy.redact_for_egress(out)


def verify_service(service, transport=None, now=None):
    """Prove the confirmed credentials work with one bounded read."""
    service = str(service or "").strip().lower()
    now = time.time() if now is None else now
    with _LOCK:
        config = _prune(_load(), now)
    ok, reason, entry = _require_service(service, config, now)
    if not ok:
        return {"ok": False, "service": service, "verified": False,
                "error": reason}
    read_scope = {"mail": MAIL_READ, "calendar": CALENDAR_READ,
                  "contacts": CONTACTS_READ}[service]
    account = _primary_account(entry).get("id")
    ok, reason = _require_scope(config, service, account, read_scope, now)
    if not ok:
        return {"ok": False, "service": service, "verified": False,
                "account": account, "error": reason}
    client = transport or _transport_for(entry)
    try:
        if service == "contacts":
            client.directory("contacts", account, "", 1)
        elif service == "calendar":
            client.list_events(account, "", "", 1)
        else:
            client.list_messages(account, "", 1, False)
    except ProviderError as exc:
        return {"ok": False, "service": service, "verified": False,
                "account": account, "error": str(exc)}
    return {"ok": True, "service": service, "verified": True,
            "account": account, "error": ""}


def _transport_for(entry):
    return productivity_providers.build_transport(
        (entry or {}).get("provider"), _token_for(entry),
        config=entry)


def _require_service(service, config, now=None):
    entry = (config.get("services") or {}).get(service)
    if not entry:
        return False, ("the %s service is not configured and confirmed; ask "
                       "the user which %s service they use before acting"
                       % (service or "?", service or "?")), None
    if not entry.get("confirmed"):
        return False, ("the %s service is configured but not confirmed by "
                       "the user, so nothing was done" % service), None
    if not _has_credentials(entry):
        return False, ("the %s service has no credentials configured, so no "
                       "request was made" % service), None
    return True, "", entry


# ── delegated permissions ──────────────────────────────────────────────────
def _grant_key(service, account):
    return "%s|%s" % (service, account)


def _find_grant(config, service, account):
    for grant_record in config.get("grants") or []:
        if (grant_record.get("service") == service
                and grant_record.get("account") == account):
            return grant_record
    return None


def _epoch(config, service, account):
    return _int((config.get("epochs") or {}).get(
        _grant_key(service, account), 0))


def _bump_epoch(config, service, account):
    key = _grant_key(service, account)
    config.setdefault("epochs", {})
    config["epochs"][key] = int(config["epochs"].get(key, 0)) + 1
    return config["epochs"][key]


def grant(service, account, scopes, ttl=None, acknowledge_elevated=False,
          now=None):
    """Delegate least-privilege scopes for one account.

    Unknown scopes, wildcards, scopes that do not belong to the service and
    accounts the user did not configure are all refused. An elevated scope
    (``mail.send`` / ``calendar.write``) additionally needs
    ``acknowledge_elevated=True`` — the capability to change the outside
    world is never granted as a side effect of a read request.
    """
    service = str(service or "").strip().lower()
    account = str(account or "").strip()
    now = time.time() if now is None else now
    requested = scopes if isinstance(scopes, (list, tuple, set, frozenset)) \
        else [scopes]
    cleaned = sorted({str(scope or "").strip() for scope in requested
                      if str(scope or "").strip()})
    if not cleaned:
        return {"ok": False, "error": "no scopes requested"}
    if any(scope in ("*", "all", "full") for scope in cleaned):
        return {"ok": False,
                "error": "wildcard scopes are not allowed; request the exact "
                         "scopes needed (e.g. mail.read)"}
    unknown = [scope for scope in cleaned if scope not in ALL_SCOPES]
    if unknown:
        return {"ok": False, "error": "unknown scope(s): %s"
                % ", ".join(unknown)}
    wrong = [scope for scope in cleaned
             if scope not in (SERVICE_SCOPES.get(service) or frozenset())]
    if wrong:
        return {"ok": False,
                "error": "scope(s) %s do not belong to the %s service"
                         % (", ".join(wrong), service or "?")}
    elevated = sorted(scope for scope in cleaned if scope in ELEVATED_SCOPES)
    if elevated and not acknowledge_elevated:
        return {"ok": False, "elevated_scopes": elevated,
                "error": ("%s can change things outside this machine; "
                          "granting it requires acknowledge_elevated=True, "
                          "and using it still requires a separate approval"
                          % ", ".join(elevated))}

    with _LOCK:
        config = _prune(_load(), now)
        ok, reason, entry = _require_service(service, config, now)
        if not ok:
            return {"ok": False, "error": reason}
        ok, resolved, reason = _require_account(service, account, entry)
        if not ok:
            return {"ok": False, "error": reason}
        account = resolved
        epoch = _bump_epoch(config, service, account)
        existing = _find_grant(config, service, account)
        record = {
            "grant_id": "g-%s-%d" % (
                hashlib.sha1(_grant_key(service, account).encode()).hexdigest()[:8],
                epoch),
            "service": service,
            "account": account,
            "scopes": cleaned,
            "elevated_scopes": elevated,
            "epoch": epoch,
            "delegated_at": now,
            "expires_at": (now + _num(ttl)) if ttl else None,
        }
        if existing:
            config["grants"] = [g for g in config["grants"] if g is not existing]
        config["grants"].append(record)
        _save(config)
    return {"ok": True, "grant_id": record["grant_id"], "service": service,
            "account": account, "scopes": cleaned, "elevated_scopes": elevated,
            "epoch": epoch, "expires_at": record["expires_at"],
            "note": ("externally visible operations still need a separate "
                     "per-effect approval" if elevated else "")}


def revoke(service, account=None, scopes=None, now=None):
    """Withdraw delegated permissions and kill every approval they covered.

    Revocation is durable and immediate: the grant epoch is bumped, so an
    approval that was requested before the revocation can no longer be
    consumed even if the draft is unchanged.
    """
    service = str(service or "").strip().lower()
    now = time.time() if now is None else now
    with _LOCK:
        config = _prune(_load(), now)
        result = _revoke_locked(config, service, account, scopes, now)
        _save(config)
    return result


def _revoke_locked(config, service, account, scopes, now):
    removed = []
    kept = []
    for grant_record in config.get("grants") or []:
        if grant_record.get("service") != service:
            kept.append(grant_record)
            continue
        if account and grant_record.get("account") != account:
            kept.append(grant_record)
            continue
        if scopes:
            wanted = {str(scope) for scope in scopes}
            remaining = [scope for scope in (grant_record.get("scopes") or [])
                         if scope not in wanted]
            if remaining:
                grant_record["scopes"] = remaining
                grant_record["elevated_scopes"] = [
                    scope for scope in remaining if scope in ELEVATED_SCOPES]
                _bump_epoch(config, service, grant_record.get("account"))
                grant_record["epoch"] = _epoch(
                    config, service, grant_record.get("account"))
                removed.append(grant_record)
                kept.append(grant_record)
                continue
        removed.append(grant_record)
        _bump_epoch(config, service, grant_record.get("account"))
    config["grants"] = kept

    keys = {_grant_key(service, g.get("account")) for g in removed}
    invalidated = 0
    for approval in config.get("approvals") or []:
        if approval.get("service") != service:
            continue
        if keys and _grant_key(approval.get("service"),
                               approval.get("account")) not in keys:
            continue
        if account and approval.get("account") != account:
            continue
        approval["revoked"] = True
        approval["revoked_at"] = now
        invalidated += 1
    if not removed and account:
        # No grant left to drop, but a stale epoch still kills any approval.
        _bump_epoch(config, service, account)
    # Revoked approvals are kept (marked) for a short window so a late replay
    # is refused with "revoked" instead of "unknown"; _prune ages them out.
    config["approvals"] = [a for a in config.get("approvals") or []
                           if isinstance(a, dict)]
    return {"ok": True, "service": service, "account": account,
            "revoked": [g.get("grant_id") for g in removed],
            "invalidated_approvals": invalidated,
            "error": ""}


def effective_scopes(service, account=None, now=None):
    """The scopes currently delegated for (service, account)."""
    service = str(service or "").strip().lower()
    now = time.time() if now is None else now
    with _LOCK:
        config = _prune(_load(), now)
        entry = (config.get("services") or {}).get(service)
        if not entry:
            return []
        if not account:
            account = _primary_account(entry).get("id")
        grant_record = _find_grant(config, service, account)
    if not grant_record:
        return []
    expires_at = grant_record.get("expires_at")
    if expires_at and _num(expires_at) <= now:
        return []
    return sorted(grant_record.get("scopes") or [])


def _require_scope(config, service, account, scope, now):
    """(ok, reason) — the authority gate for one operation."""
    if not scope:
        return True, ""
    grant_record = _find_grant(config, service, account)
    if not grant_record:
        return False, (
            "no permission is delegated for %s account %r; grant %s first "
            "(nothing was done)" % (service, account, scope))
    expires_at = grant_record.get("expires_at")
    if expires_at and _num(expires_at) <= now:
        return False, ("the %s permission for %s expired; re-grant it"
                       % (scope, account))
    if scope not in (grant_record.get("scopes") or []):
        hint = ""
        if scope in ELEVATED_SCOPES:
            hint = (" — %s is an externally visible capability and must be "
                    "delegated separately, then approved per effect" % scope)
        return False, ("the grant for %s account %r does not include %s%s"
                       % (service, account, scope, hint))
    return True, ""


# ── semantic entity resolution ─────────────────────────────────────────────
def _match_entity(query, entries):
    wanted = _normalize(query)
    if not wanted:
        return "unresolved", [], "no query was given"
    exact = [e for e in entries
             if wanted in (_normalize(e.get("name")), _normalize(e.get("email")),
                           _normalize(e.get("id")))]
    if len(exact) == 1:
        return "resolved", exact, ""
    if len(exact) > 1:
        return "ambiguous", exact, "%d entries match %r exactly" % (
            len(exact), query)
    partial = [e for e in entries
               if wanted in _normalize(e.get("name"))
               or wanted in _normalize(e.get("email"))
               or _normalize(e.get("name")).startswith(wanted)]
    if len(partial) == 1:
        return "resolved", partial, ""
    if len(partial) > 1:
        return "ambiguous", partial, "%d entries match %r" % (
            len(partial), query)
    return "unresolved", [], "nothing matched %r" % query


def resolve_entity(service, query, account=None, limit=20, transport=None,
                   now=None):
    """Resolve a spoken name to exactly one entity, or report ambiguity.

    Refuses without the read scope: resolution is a provider read, and an
    unapproved read is still a read.
    """
    service = str(service or "").strip().lower()
    if service not in ("contacts", "calendar"):
        return {"ok": False, "status": "unsupported", "service": service,
                "resolved": None, "candidates": [],
                "reason": "entity resolution applies to contacts and "
                          "calendars; mail recipients resolve through "
                          "contacts.resolve"}
    now = time.time() if now is None else now
    with _LOCK:
        config = _prune(_load(), now)
    ok, reason, entry = _require_service(service, config, now)
    if not ok:
        return {"ok": False, "status": "unavailable", "service": service,
                "resolved": None, "candidates": [], "reason": reason}
    ok, account, reason = _require_account(service, account, entry)
    if not ok:
        return {"ok": False, "status": "unresolved", "service": service,
                "resolved": None, "candidates": [], "reason": reason}
    scope = CONTACTS_READ if service == "contacts" else CALENDAR_READ
    ok, reason = _require_scope(config, service, account, scope, now)
    if not ok:
        return {"ok": False, "status": "forbidden", "service": service,
                "account": account, "resolved": None, "candidates": [],
                "reason": reason}
    client = transport or _transport_for(entry)
    try:
        entries = [e for e in client.directory(
            service, account, query or "", max(1, int(limit)))
            if isinstance(e, dict) and (e.get("name") or e.get("email")
                                        or e.get("id"))]
    except ProviderError as exc:
        return {"ok": False, "status": "error", "service": service,
                "account": account, "resolved": None, "candidates": [],
                "reason": str(exc)}
    status, matches, reason = _match_entity(query, entries)
    return {
        "ok": status == "resolved",
        "status": status,
        "service": service,
        "account": account,
        "query": query,
        "resolved": matches[0] if status == "resolved" else None,
        "candidates": matches if status == "ambiguous" else entries[:limit],
        "reason": reason,
    }


# ── drafts: locally composed, no external effect ───────────────────────────
def _content_hash(record):
    blob = json.dumps({
        "service": record.get("service"),
        "account": record.get("account"),
        "effect": record.get("effect"),
        "fields": record.get("fields"),
    }, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _address_list(value, field):
    if value is None:
        return [], ""
    if not isinstance(value, (list, tuple)):
        return [], "%s must be a list of addresses" % field
    out = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        if not _EMAIL_RE.match(text):
            return [], (
                "%s entry %r is not an email address — resolve the contact "
                "first (contacts.resolve) and pass the resolved address"
                % (field, text))
        out.append(text)
    return out, ""


def _validate_mail(arguments):
    to, error = _address_list(arguments.get("to"), "to")
    if error:
        return None, error
    if not to:
        return None, "at least one recipient address is required"
    cc, error = _address_list(arguments.get("cc"), "cc")
    if error:
        return None, error
    bcc, error = _address_list(arguments.get("bcc"), "bcc")
    if error:
        return None, error
    return {"to": to, "cc": cc, "bcc": bcc,
            "subject": str(arguments.get("subject") or ""),
            "body": str(arguments.get("body") or "")}, ""


def _validate_iso(value, field):
    text = str(value or "").strip()
    if not text:
        return "", "%s is required" % field
    if not _ISO_RE.match(text):
        return "", "%s must be an ISO-8601 date or date-time (%r given)" % (
            field, text)
    normalized = text.replace("Z", "+00:00")
    try:
        datetime.fromisoformat(normalized)
    except ValueError:
        return "", "%s is not a valid date/time (%r)" % (field, text)
    return text, ""


def _validate_event(arguments):
    title = str(arguments.get("title") or "").strip()
    if not title:
        return None, "a title is required"
    start, error = _validate_iso(arguments.get("start"), "start")
    if error:
        return None, error
    end, error = _validate_iso(arguments.get("end"), "end")
    if error:
        return None, error
    attendees, error = _address_list(arguments.get("attendees"), "attendees")
    if error:
        return None, error
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if end_dt <= start_dt:
            return None, "end must be after start"
    except ValueError:
        return None, "start/end could not be ordered"
    return {"title": title, "start": start, "end": end,
            "attendees": attendees,
            "location": str(arguments.get("location") or ""),
            "description": str(arguments.get("description") or ""),
            "calendar": str(arguments.get("calendar") or "")}, ""


def create_draft(service, account, fields, effect, now=None):
    """Store a locally composed draft. Performs NO provider request."""
    service = str(service or "").strip().lower()
    now = time.time() if now is None else now
    with _LOCK:
        config = _prune(_load(), now)
        ok, reason, entry = _require_service(service, config, now)
        if not ok:
            return {"ok": False, "error": reason}
        ok, account, reason = _require_account(service, account, entry)
        if not ok:
            return {"ok": False, "error": reason}
        scope = MAIL_DRAFT if service == "mail" else CALENDAR_DRAFT
        ok, reason = _require_scope(config, service, account, scope, now)
        if not ok:
            return {"ok": False, "error": reason}
        draft_id = "d-%d-%d" % (int(now * 1000), len(config["drafts"]) + 1)
        record = {"draft_id": draft_id, "service": service, "account": account,
                  "effect": effect, "fields": fields, "created_at": now}
        record["content_hash"] = _content_hash(record)
        record["preview"] = describe_draft(record)
        config["drafts"].append(record)
        _save(config)
    return {
        "ok": True,
        "draft_id": draft_id,
        "service": service,
        "account": account,
        "effect": effect,
        "content_hash": record["content_hash"],
        "preview": record["preview"],
        "external_effect": False,
        "note": ("draft saved locally — nothing has been sent or created; "
                 "request_effect_approval() is required before committing"),
    }


def describe_draft(draft):
    """One-line, secret-scrubbed description used in the approval preview."""
    draft = draft or {}
    fields = draft.get("fields") or {}
    if draft.get("effect") == MAIL_SEND:
        return tool_policy.mask_secrets(
            "Send mail from %s to %s — subject %r (%d chars)"
            % (draft.get("account"), ", ".join(fields.get("to") or []),
               fields.get("subject") or "", len(fields.get("body") or "")))
    return tool_policy.mask_secrets(
        "Create calendar event %r on %s from %s to %s (%d attendees)"
        % (fields.get("title"), fields.get("calendar") or draft.get("account"),
           fields.get("start"), fields.get("end"),
           len(fields.get("attendees") or [])))


def get_draft(draft_id, now=None):
    with _LOCK:
        config = _prune(_load(), time.time() if now is None else now)
    for record in config["drafts"]:
        if record.get("draft_id") == draft_id:
            return dict(record)
    return None


def list_drafts(service=None, now=None):
    with _LOCK:
        config = _prune(_load(), time.time() if now is None else now)
    return [dict(record) for record in config["drafts"]
            if not service or record.get("service") == service]


# ── separately approved effects ────────────────────────────────────────────
def request_effect_approval(draft_id, ttl=None, now=None):
    """Ask for consent for one specific draft's externally visible effect.

    Refuses unless the standing grant already includes the elevated scope:
    delegation and per-effect consent are two independent gates, and both
    must be satisfied.
    """
    now = time.time() if now is None else now
    ttl = APPROVAL_TTL_SECONDS if ttl is None else _num(ttl,
                                                        APPROVAL_TTL_SECONDS)
    with _LOCK:
        config = _prune(_load(), now)
        draft = None
        for record in config["drafts"]:
            if record.get("draft_id") == draft_id:
                draft = record
                break
        if draft is None:
            return {"ok": False, "error": "unknown draft %r" % draft_id}
        service = draft.get("service")
        account = draft.get("account")
        scope = MAIL_SEND if draft.get("effect") == MAIL_SEND else CALENDAR_WRITE
        ok, reason = _require_scope(config, service, account, scope, now)
        if not ok:
            return {"ok": False, "error": reason, "scope": scope}
        approval_id = "a-%d-%d" % (int(now * 1000), len(config["approvals"]) + 1)
        record = {
            "approval_id": approval_id,
            "draft_id": draft_id,
            "service": service,
            "account": account,
            "effect": draft.get("effect"),
            "scope": scope,
            "content_hash": draft.get("content_hash"),
            "grant_epoch": _epoch(config, service, account),
            "created_at": now,
            "expires_at": now + ttl,
            "consumed": False,
            "preview": draft.get("preview"),
        }
        config["approvals"].append(record)
        _save(config)
    return {"ok": True, "approval_id": approval_id, "draft_id": draft_id,
            "effect": record["effect"], "service": service, "account": account,
            "scope": scope, "expires_at": record["expires_at"],
            "content_hash": record["content_hash"],
            "preview": record["preview"],
            "spoken": ("Say confirm to %s, or cancel."
                       % ("send this mail" if record["effect"] == MAIL_SEND
                          else "create this event"))}


def _verify_approval(config, draft, approval_id, expected_effect, now):
    """(ok, reason, approval_record). Fail-closed at every step."""
    record = None
    for candidate in config.get("approvals") or []:
        if candidate.get("approval_id") == approval_id:
            record = candidate
            break
    if record is None:
        return False, "no approval %r is on record" % approval_id, None
    if record.get("consumed"):
        return False, "this approval was already used", None
    if record.get("revoked"):
        return False, "this approval was revoked with its permissions", None
    if _num(record.get("expires_at")) <= now:
        return False, "this approval expired", None
    if record.get("draft_id") != draft.get("draft_id"):
        return False, ("this approval covers a different draft (%s), so it "
                       "cannot authorise this one"
                       % record.get("draft_id")), None
    if draft.get("effect") != expected_effect:
        return False, ("this draft's effect is %r, not %r"
                       % (draft.get("effect"), expected_effect)), None
    if record.get("effect") != expected_effect:
        return False, ("this approval is for %r, not %r"
                       % (record.get("effect"), expected_effect)), None
    if record.get("content_hash") != draft.get("content_hash"):
        return False, ("the draft changed after it was approved; re-read it "
                       "and approve again"), None
    # The stored hash is not trusted on its own: recompute it from the fields
    # that would actually be executed, so an edit that did not refresh the
    # hash is still caught.
    if _content_hash(draft) != draft.get("content_hash"):
        return False, ("the draft's content no longer matches its recorded "
                       "hash; re-read it and approve again"), None
    service, account = draft.get("service"), draft.get("account")
    if _epoch(config, service, account) != _int(record.get("grant_epoch")):
        return False, ("the delegated permissions changed after this approval "
                       "was given, so it is no longer valid"), None
    scope = MAIL_SEND if expected_effect == MAIL_SEND else CALENDAR_WRITE
    ok, reason = _require_scope(config, service, account, scope, now)
    if not ok:
        return False, reason, None
    return True, "", record


def commit_draft(draft_id, approval_id, expected_effect=None, transport=None,
                 now=None):
    """Execute one approved, unchanged draft. Consumes the approval first.

    Order matters: verify -> consume -> execute. If the provider call fails
    afterwards the approval is already gone, so the caller must ask again —
    replaying an ambiguous delivery is worse than a second confirmation.
    """
    now = time.time() if now is None else now
    with _LOCK:
        config = _prune(_load(), now)
        draft = None
        for record in config["drafts"]:
            if record.get("draft_id") == draft_id:
                draft = record
                break
        if draft is None:
            return {"ok": False, "error": "unknown draft %r" % draft_id}
        # F14: an explicitly expected effect is used EXACTLY as given. Reading
        # it as ``expected_effect or draft["effect"]`` meant asking for a
        # calendar create with a mail draft quietly executed the mail send
        # instead — the draft's own effect must never override the caller's
        # declared intent.
        if expected_effect is None:
            effect = draft.get("effect")
        else:
            effect = expected_effect
        if effect not in EXTERNAL_EFFECT_VALUES:
            return {"ok": False, "error": "unsupported effect %r" % effect,
                    "draft_id": draft_id, "external_effect": False}
        ok, reason, approval = _verify_approval(
            config, draft, approval_id, effect, now)
        if not ok:
            return {"ok": False, "error": reason, "effect": effect,
                    "draft_id": draft_id, "external_effect": False}
        ok, reason, entry = _require_service(draft.get("service"), config, now)
        if not ok:
            return {"ok": False, "error": reason, "effect": effect,
                    "external_effect": False}
        approval["consumed"] = True
        approval["consumed_at"] = now
        _save(config)
        payload = dict(draft)

    client = transport or _transport_for(entry)
    try:
        if effect == MAIL_SEND:
            result = client.send_message(draft.get("account"), payload)
        elif effect == CALENDAR_WRITE:
            result = client.create_event(draft.get("account"), payload)
        else:
            return {"ok": False, "error": "unsupported effect %r" % effect,
                    "external_effect": False}
    except ProviderError as exc:
        return {"ok": False, "error": str(exc), "effect": effect,
                "draft_id": draft_id, "external_effect": False,
                "approval_consumed": True}
    return {
        "ok": True,
        "effect": effect,
        "draft_id": draft_id,
        "approval_id": approval_id,
        "external_effect": True,
        "result": result,
        "content": ("%s — %s" % (
            "Sent" if effect == MAIL_SEND else "Created",
            draft.get("preview", ""))),
    }


# ── typed dispatch ─────────────────────────────────────────────────────────
def _refuse(tool, reason, **extra):
    payload = {"ok": False, "tool": tool, "error": reason, "reason": reason,
               "external_effect": False}
    payload.update(extra)
    return tool_policy.redact_for_egress(payload)


def _validate_arguments(spec, arguments):
    """(ok, reason) — required fields, types, enums and no unknown keys."""
    if not isinstance(arguments, dict):
        return False, "arguments must be an object"
    allowed = set(spec["args"])
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        return False, ("unknown argument(s) for this typed operation: %s"
                       % ", ".join(unknown))
    for name in spec["required"]:
        value = arguments.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            return False, "missing required argument %r" % name
    for name, value in arguments.items():
        if value is None:
            continue
        expected = spec["args"][name]["type"]
        if not tool_policy._matches_type(value, expected):
            return False, "argument %r must be %s" % (name, expected)
        enum = spec["args"][name].get("enum")
        if enum and value not in enum:
            return False, "argument %r must be one of %s" % (name, enum)
    return True, ""


def _limit(arguments, default=DEFAULT_READ_LIMIT):
    value = arguments.get("limit")
    if value is None:
        return default
    try:
        return max(1, min(MAX_READ_LIMIT, int(value)))
    except (TypeError, ValueError):
        return default


def _render(tool, data):
    """A short human/planner-readable rendering of a typed result."""
    if tool == "mail.list_messages":
        return "\n".join(
            "- %s | %s | %s | %s" % (m.get("id"), m.get("date"),
                                     m.get("from"), m.get("subject"))
            for m in data) or "no messages"
    if tool == "mail.read_message":
        return "%s\nFrom: %s\n%s" % (data.get("subject"), data.get("from"),
                                     (data.get("body") or data.get("snippet")
                                      or "")[:2000])
    if tool == "calendar.list_events":
        return "\n".join(
            "- %s | %s | %s .. %s | %s" % (
                e.get("id"), e.get("title"), e.get("start"), e.get("end"),
                e.get("location")) for e in data) or "no events"
    if tool == "calendar.check_availability":
        return "busy slots: %s" % (data.get("busy") or "none")
    if tool in ("contacts.search", "calendar.resolve"):
        return "\n".join(
            "- %s | %s | %s" % (c.get("name"), c.get("email") or c.get("id"),
                                c.get("kind")) for c in data) or "no matches"
    if tool == "contacts.resolve":
        if data.get("status") == "resolved":
            return "resolved %r -> %s" % (
                data.get("query"), (data.get("resolved") or {}).get("email")
                or (data.get("resolved") or {}).get("id"))
        return "not resolved (%s): %s" % (data.get("status"),
                                          data.get("reason"))
    return json.dumps(data, ensure_ascii=False, default=str)[:2000]


def run_operation(name, arguments=None, transport=None, now=None):
    """The single typed entry point for every productivity operation.

    Nothing else in this module executes a provider call on behalf of the
    model: malformed arguments, an unknown tool, an unconfirmed service, a
    missing scope, an unresolvable account or an unapproved effect all end
    here with ``ok: False`` and no external effect.
    """
    name = str(name or "").strip()
    arguments = dict(arguments) if isinstance(arguments, dict) else {}
    if not isinstance(arguments, dict):
        arguments = {}
    spec = OPERATIONS.get(name)
    if spec is None:
        return _refuse(name, "unknown productivity operation %r (known: %s)"
                       % (name, ", ".join(TOOL_NAMES)))
    ok, reason = _validate_arguments(spec, arguments)
    if not ok:
        return _refuse(name, reason, service=spec["service"],
                       operation=spec["operation"])

    now = time.time() if now is None else now
    if spec["service"] is None:
        if name == "productivity.services":
            data = service_inventory()
            return tool_policy.redact_for_egress(
                {"ok": True, "tool": name, "operation": spec["operation"],
                 "external_effect": False, "data": data,
                 "content": _render(name, data)})
        return _refuse(name, "operation %r has no service handler" % name)

    service = spec["service"]
    with _LOCK:
        config = _prune(_load(), now)
    ok, reason, entry = _require_service(service, config, now)
    if not ok:
        return _refuse(name, reason, service=service,
                       operation=spec["operation"])
    ok, account, reason = _require_account(
        service, arguments.get("account"), entry)
    if not ok:
        return _refuse(name, reason, service=service,
                       operation=spec["operation"])
    ok, reason = _require_scope(config, service, account, spec["scope"], now)
    if not ok:
        return _refuse(name, reason, service=service, account=account,
                       operation=spec["operation"],
                       scope=spec["scope"])

    # Externally visible effects go through the approval gate only.
    if spec["operation"] in EXTERNAL_EFFECTS:
        result = commit_draft(arguments.get("draft_id"),
                              arguments.get("approval_id"),
                              expected_effect=spec["effect"],
                              transport=transport, now=now)
        wrapped = dict(result)
        wrapped.setdefault("tool", name)
        wrapped.setdefault("service", service)
        wrapped.setdefault("account", account)
        wrapped.setdefault("operation", spec["operation"])
        return tool_policy.redact_for_egress(wrapped)

    if spec["validator"] and name in ("mail.create_draft",
                                      "calendar.propose_event"):
        fields, error = (_validate_mail if spec["validator"] == "mail"
                         else _validate_event)(arguments)
        if error:
            return _refuse(name, error, service=service, account=account,
                           operation=spec["operation"])
        result = create_draft(
            service, account, fields, spec["effect"], now=now)
        if not result.get("ok"):
            return _refuse(name, result.get("error") or "draft refused",
                           service=service, account=account,
                           operation=spec["operation"])
        return tool_policy.redact_for_egress({
            "ok": True, "tool": name, "service": service, "account": account,
            "operation": spec["operation"], "external_effect": False,
            "draft_id": result["draft_id"],
            "content_hash": result["content_hash"],
            "preview": result["preview"], "data": result,
            "content": ("draft %s ready — nothing sent: %s"
                        % (result["draft_id"], result["preview"])),
        })

    client = transport or _transport_for(entry)
    try:
        if name == "mail.list_messages":
            data = client.list_messages(account, arguments.get("query") or "",
                                        _limit(arguments),
                                        bool(arguments.get("unread_only")))
        elif name == "mail.read_message":
            data = client.get_message(account, arguments["message_id"])
        elif name == "calendar.list_events":
            data = client.list_events(account, arguments.get("start") or "",
                                      arguments.get("end") or "",
                                      _limit(arguments))
        elif name == "calendar.check_availability":
            data = client.free_busy(account, arguments["start"],
                                    arguments["end"])
        elif name == "contacts.search":
            data = client.directory("contacts", account,
                                    arguments.get("query") or "",
                                    _limit(arguments))
        elif name == "contacts.resolve":
            data = resolve_entity("contacts", arguments["query"], account,
                                  limit=DEFAULT_READ_LIMIT, transport=client,
                                  now=now)
        elif name == "calendar.resolve":
            data = resolve_entity("calendar", arguments["query"], account,
                                  limit=DEFAULT_READ_LIMIT, transport=client,
                                  now=now)
        else:  # pragma: no cover - the registry and this block are in sync
            return _refuse(name, "operation %r has no handler" % name)
    except ProviderError as exc:
        return _refuse(name, str(exc), service=service, account=account,
                       operation=spec["operation"])
    return tool_policy.redact_for_egress({
        "ok": True, "tool": name, "service": service, "account": account,
        "operation": spec["operation"], "external_effect": False,
        "data": data, "content": _render(name, data),
    })
