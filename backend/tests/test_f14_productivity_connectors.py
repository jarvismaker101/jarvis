"""F14 — semantic productivity connectors (mail / calendar / contacts).

Acceptance (audit report): "Provide source/tests for account/entity
resolution, delegated permissions, revocation, and no unauthorized
send/create effects."

Correction implemented: "Confirm actual user services, then add typed
least-privilege read/draft operations with separately approved sending and
externally visible changes."

The tests below exercise the real connector and the real provider profiles;
only true externals are faked — the HTTP session (a fake ``requests``
session) and the provider transport itself.
"""

import json
import os
import tempfile
import unittest

from backend.services import productivity_connector as pc
from backend.services import productivity_providers as pp


# ── fakes for the true externals ───────────────────────────────────────────
class FakeTransport:
    """Stands in for a provider; records every call it receives."""

    provider = "fake"

    def __init__(self, directory=None, messages=None, message=None,
                 events=None, busy=None, sent=None, created=None):
        self.calls = []
        self._directory = directory if directory is not None else []
        self._messages = messages if messages is not None else []
        self._message = message or {}
        self._events = events if events is not None else []
        self._busy = busy or {"busy": []}
        self._sent = sent or {"id": "m-1", "sent": True}
        self._created = created or {"id": "e-1", "created": True}

    @property
    def effects(self):
        return [call for call in self.calls
                if call[0] in ("send_message", "create_event")]

    def directory(self, service, account, query="", limit=20):
        self.calls.append(("directory", service, account, query, limit))
        return list(self._directory)

    def list_messages(self, account, query="", limit=20, unread_only=False):
        self.calls.append(("list_messages", account, query, limit, unread_only))
        return list(self._messages)

    def get_message(self, account, message_id):
        self.calls.append(("get_message", account, message_id))
        return dict(self._message)

    def send_message(self, account, draft):
        self.calls.append(("send_message", account, draft))
        return dict(self._sent)

    def list_events(self, account, start="", end="", limit=20):
        self.calls.append(("list_events", account, start, end, limit))
        return list(self._events)

    def free_busy(self, account, start, end):
        self.calls.append(("free_busy", account, start, end))
        return dict(self._busy)

    def create_event(self, account, draft):
        self.calls.append(("create_event", account, draft))
        return dict(self._created)


class _ExplodingTransport(FakeTransport):
    """Any provider call at all is a test failure."""

    def _boom(self, *args, **kwargs):
        raise AssertionError("a provider request was made when none was "
                             "allowed")

    directory = list_messages = get_message = _boom
    send_message = list_events = free_busy = create_event = _boom


MAIL = "me@example.com"
WORK = "work@example.com"
SEND_ALL = (pc.MAIL_READ, pc.MAIL_DRAFT, pc.MAIL_SEND)


class ConnectorTestCase(unittest.TestCase):
    """Isolated config store; nothing here touches the real data/ file."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._saved_env = os.environ.get(pc.CONFIG_ENV)
        os.environ[pc.CONFIG_ENV] = os.path.join(self._tmp.name,
                                                 "productivity.json")
        self.transport = FakeTransport()

    def tearDown(self):
        if self._saved_env is None:
            os.environ.pop(pc.CONFIG_ENV, None)
        else:
            os.environ[pc.CONFIG_ENV] = self._saved_env
        self._tmp.cleanup()

    # -- helpers --
    def configure_mail(self, scopes=(pc.MAIL_READ, pc.MAIL_DRAFT),
                       accounts=None, confirmed=True, token="test-token"):
        result = pc.configure_service(
            "mail", "google", MAIL, token=token, accounts=accounts,
            confirmed=confirmed)
        self.assertTrue(result["ok"], result)
        if scopes:
            granted = pc.grant("mail", MAIL, list(scopes),
                               acknowledge_elevated=pc.MAIL_SEND in scopes)
            self.assertTrue(granted["ok"], granted)
        return result

    def configure_contacts(self, scopes=(pc.CONTACTS_READ,)):
        result = pc.configure_service("contacts", "google", MAIL,
                                      token="test-token", confirmed=True)
        self.assertTrue(result["ok"], result)
        if scopes:
            granted = pc.grant("contacts", MAIL, list(scopes))
            self.assertTrue(granted["ok"], granted)

    def configure_calendar(self, scopes=(pc.CALENDAR_READ, pc.CALENDAR_DRAFT)):
        result = pc.configure_service("calendar", "google", MAIL,
                                      token="test-token", confirmed=True)
        self.assertTrue(result["ok"], result)
        if scopes:
            granted = pc.grant("calendar", MAIL, list(scopes),
                               acknowledge_elevated=pc.CALENDAR_WRITE in scopes)
            self.assertTrue(granted["ok"], granted)

    def mail_draft(self, to=("friend@example.com",), body="hello",
                   subject="hi", transport=None):
        result = pc.run_operation(
            "mail.create_draft",
            {"account": MAIL, "to": list(to), "subject": subject, "body": body},
            transport=transport or self.transport)
        self.assertTrue(result["ok"], result)
        return result


# ── 1. confirm actual user services ────────────────────────────────────────
class ConfirmedServicesTests(ConnectorTestCase):
    def test_nothing_is_assumed_before_the_user_confirms(self):
        """F14: no connector is assumed to exist — no mail, calendar or
        contacts interface is reachable on a fresh install."""
        inventory = pc.service_inventory()
        self.assertEqual([entry["service"] for entry in inventory],
                         list(pc.SERVICES))
        self.assertFalse(any(entry["configured"] for entry in inventory))
        result = pc.run_operation("mail.list_messages", {"account": MAIL},
                                  transport=self.transport)
        self.assertFalse(result["ok"])
        self.assertIn("not configured", result["error"])
        self.assertEqual(self.transport.calls, [])

    def test_a_configured_but_unconfirmed_service_refuses_every_operation(self):
        pc.configure_service("mail", "google", MAIL, token="test-token")
        self.assertEqual(pc.service_inventory()[0]["confirmed"], False)
        result = pc.run_operation("mail.list_messages", {"account": MAIL},
                                  transport=self.transport)
        self.assertFalse(result["ok"])
        self.assertIn("not confirmed", result["error"])
        self.assertEqual(self.transport.calls, [])

    def test_permissions_cannot_be_delegated_for_an_unconfirmed_service(self):
        pc.configure_service("mail", "google", MAIL, token="test-token")
        refused = pc.grant("mail", MAIL, [pc.MAIL_READ])
        self.assertFalse(refused["ok"])
        self.assertIn("not confirmed", refused["error"])

    def test_discovery_proposes_env_hints_but_adopts_nothing(self):
        os.environ["JARVIS_GMAIL_TOKEN"] = "sk-not-a-real-token-value-123456"
        try:
            discovered = pc.discover_services()
        finally:
            os.environ.pop("JARVIS_GMAIL_TOKEN", None)
        hints = [hint for hint in discovered["env_hints"]
                 if hint["env_var"] == "JARVIS_GMAIL_TOKEN"]
        self.assertTrue(hints)
        self.assertEqual(discovered["confirmed"], [])
        # The hint is evidence, not consent: nothing became usable.
        self.assertFalse(pc.service_inventory()[0]["configured"])

    def test_confirming_unlocks_reads_and_only_then_does_a_request(self):
        self.configure_mail()
        result = pc.run_operation("mail.list_messages", {"account": MAIL},
                                  transport=self.transport)
        self.assertTrue(result["ok"], result)
        self.assertEqual([call[0] for call in self.transport.calls],
                         ["list_messages"])

    def test_an_unauthenticated_service_makes_no_request(self):
        pc.configure_service("mail", "google", MAIL, confirmed=True)
        result = pc.run_operation("mail.list_messages", {"account": MAIL},
                                  transport=self.transport)
        self.assertFalse(result["ok"])
        self.assertIn("no credentials", result["error"])
        self.assertEqual(self.transport.calls, [])

    def test_inventory_and_results_never_carry_the_token(self):
        secret = "sk-live-abcdefghijklmnopqrstuvwxyz012345"
        self.configure_mail(token=secret)
        blob = json.dumps(pc.service_inventory())
        self.assertNotIn(secret, blob)
        self.assertNotIn("token", blob.lower().replace("token_env", ""))
        result = pc.run_operation("mail.list_messages", {"account": MAIL},
                                  transport=self.transport)
        self.assertNotIn(secret, json.dumps(result, default=str))

    def test_a_token_env_indirection_is_reported_but_not_returned(self):
        os.environ["JARVIS_F14_TEST_TOKEN"] = "sk-env-abcdefghijklmnopqrstuvwxyz"
        try:
            result = pc.configure_service("mail", "google", MAIL,
                                          token_env="JARVIS_F14_TEST_TOKEN",
                                          confirmed=True)
            self.assertTrue(result["ok"], result)
            self.assertTrue(result["has_key"])
            self.assertNotIn("sk-env-abcdefghijklmnopqrstuvwxyz",
                             json.dumps(result))
        finally:
            os.environ.pop("JARVIS_F14_TEST_TOKEN", None)

    def test_inventory_flags_stay_booleans_through_the_hygiene_pass(self):
        self.configure_mail()
        entry = pc.service_inventory()[0]
        self.assertIs(entry["configured"], True)
        self.assertIs(entry["confirmed"], True)
        self.assertIs(entry["has_key"], True)

    def test_an_unsupported_provider_is_refused(self):
        refused = pc.configure_service("mail", "dropbox", MAIL, token="x")
        self.assertFalse(refused["ok"])
        self.assertIn("unsupported provider", refused["error"])

    def test_verify_service_proves_the_credentials_with_one_bounded_read(self):
        self.configure_mail(scopes=(pc.MAIL_READ,))
        verified = pc.verify_service("mail", transport=self.transport)
        self.assertTrue(verified["verified"], verified)
        self.assertEqual(len(self.transport.calls), 1)
        self.assertLessEqual(self.transport.calls[0][3], 1)  # limit == 1

    def test_verify_service_refuses_without_a_read_scope(self):
        self.configure_mail(scopes=(pc.MAIL_DRAFT,))
        verified = pc.verify_service("mail", transport=self.transport)
        self.assertFalse(verified["verified"])
        self.assertEqual(self.transport.calls, [])


# ── 2. account and entity resolution ───────────────────────────────────────
class AccountResolutionTests(ConnectorTestCase):
    def setUp(self):
        super().setUp()
        self.configure_mail(
            accounts=[{"id": MAIL, "label": "personal", "primary": True},
                      {"id": WORK, "label": "work calendar"}])

    def test_the_default_account_resolves_locally_without_a_request(self):
        resolved = pc.resolve_account("mail")
        self.assertTrue(resolved["ok"])
        self.assertEqual(resolved["account"], MAIL)
        self.assertEqual(self.transport.calls, [])

    def test_a_label_resolves_to_its_account(self):
        resolved = pc.resolve_account("mail", "work calendar")
        self.assertTrue(resolved["ok"])
        self.assertEqual(resolved["account"], WORK)

    def test_an_ambiguous_account_is_reported_not_guessed(self):
        resolved = pc.resolve_account("mail", "example.com")
        self.assertFalse(resolved["ok"])
        self.assertEqual(resolved["status"], "ambiguous")
        self.assertIsNone(resolved["account"])
        self.assertEqual(len(resolved["candidates"]), 2)

    def test_an_unknown_account_is_unresolved(self):
        resolved = pc.resolve_account("mail", "nobody")
        self.assertFalse(resolved["ok"])
        self.assertEqual(resolved["status"], "unresolved")

    def test_an_unconfigured_service_cannot_resolve_an_account(self):
        resolved = pc.resolve_account("calendar")
        self.assertFalse(resolved["ok"])
        self.assertEqual(resolved["status"], "unconfigured")

    def test_an_operation_cannot_use_an_account_the_user_did_not_configure(self):
        result = pc.run_operation("mail.list_messages",
                                  {"account": "attacker@example.com"},
                                  transport=_ExplodingTransport())
        self.assertFalse(result["ok"])
        self.assertIn("not a configured mail account", result["error"])

    def test_account_resolution_prefers_the_primary_by_default(self):
        resolved = pc.resolve_account("mail", None)
        self.assertEqual(resolved["account"], MAIL)
        self.assertTrue(resolved["candidates"][0]["primary"])


class EntityResolutionTests(ConnectorTestCase):
    def test_a_spoken_name_resolves_to_exactly_one_contact(self):
        self.configure_contacts()
        directory = [{"id": "p1", "name": "Mom", "email": "mom@example.com",
                      "kind": "contact"},
                     {"id": "p2", "name": "Bob", "email": "bob@example.com",
                      "kind": "contact"}]
        transport = FakeTransport(directory=directory)
        resolved = pc.resolve_entity("contacts", "mom", transport=transport)
        self.assertTrue(resolved["ok"], resolved)
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(resolved["resolved"]["email"], "mom@example.com")
        self.assertEqual(len(transport.calls), 1)

    def test_an_ambiguous_name_reports_candidates_instead_of_guessing(self):
        self.configure_contacts()
        directory = [{"id": "p1", "name": "Mom Smith", "email": "a@x.com"},
                     {"id": "p2", "name": "Mom Jones", "email": "b@x.com"}]
        resolved = pc.resolve_entity("contacts", "mom",
                                     transport=FakeTransport(directory=directory))
        self.assertFalse(resolved["ok"])
        self.assertEqual(resolved["status"], "ambiguous")
        self.assertIsNone(resolved["resolved"])
        self.assertEqual(len(resolved["candidates"]), 2)

    def test_an_unmatched_name_is_unresolved(self):
        self.configure_contacts()
        resolved = pc.resolve_entity(
            "contacts", "zzz", transport=FakeTransport(directory=[]))
        self.assertFalse(resolved["ok"])
        self.assertEqual(resolved["status"], "unresolved")

    def test_resolution_without_the_read_scope_makes_no_request(self):
        self.configure_contacts(scopes=())
        resolved = pc.resolve_entity("contacts", "mom",
                                     transport=_ExplodingTransport())
        self.assertFalse(resolved["ok"])
        self.assertEqual(resolved["status"], "forbidden")
        self.assertIn("contacts.read", resolved["reason"])

    def test_a_spoken_calendar_name_resolves_through_the_directory(self):
        self.configure_calendar(scopes=(pc.CALENDAR_READ,))
        directory = [{"id": "work-id", "name": "Work", "kind": "calendar"},
                     {"id": "home-id", "name": "Home", "kind": "calendar"}]
        resolved = pc.resolve_entity("calendar", "work",
                                     transport=FakeTransport(directory=directory))
        self.assertTrue(resolved["ok"], resolved)
        self.assertEqual(resolved["resolved"]["id"], "work-id")

    def test_unresolved_names_are_refused_by_the_mail_draft(self):
        """A display name is not an address: resolve it first."""
        self.configure_mail()
        result = pc.run_operation(
            "mail.create_draft",
            {"account": MAIL, "to": ["Mom"], "subject": "hi", "body": "x"},
            transport=_ExplodingTransport())
        self.assertFalse(result["ok"])
        self.assertIn("contacts.resolve", result["error"])

    def test_contacts_resolve_tool_returns_the_same_typed_answer(self):
        self.configure_contacts()
        directory = [{"id": "p1", "name": "Mom", "email": "mom@example.com"}]
        transport = FakeTransport(directory=directory)
        result = pc.run_operation("contacts.resolve",
                                  {"account": MAIL, "query": "mom"},
                                  transport=transport)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["data"]["resolved"]["email"],
                         "mom@example.com")

    def test_entity_resolution_is_not_offered_for_mail(self):
        self.configure_mail()
        resolved = pc.resolve_entity("mail", "mom",
                                     transport=_ExplodingTransport())
        self.assertFalse(resolved["ok"])
        self.assertEqual(resolved["status"], "unsupported")


# ── 3. delegated permissions (least privilege) ─────────────────────────────
class DelegatedPermissionTests(ConnectorTestCase):
    def test_a_read_only_grant_cannot_compose_a_draft(self):
        self.configure_mail(scopes=(pc.MAIL_READ,))
        result = pc.run_operation(
            "mail.create_draft",
            {"account": MAIL, "to": ["a@b.com"], "subject": "s", "body": "b"},
            transport=self.transport)
        self.assertFalse(result["ok"])
        self.assertIn(pc.MAIL_DRAFT, result["error"])
        self.assertEqual(self.transport.calls, [])

    def test_a_draft_grant_cannot_read_the_mailbox(self):
        self.configure_mail(scopes=(pc.MAIL_DRAFT,))
        result = pc.run_operation("mail.list_messages", {"account": MAIL},
                                  transport=self.transport)
        self.assertFalse(result["ok"])
        self.assertIn(pc.MAIL_READ, result["error"])
        self.assertEqual(self.transport.calls, [])

    def test_a_read_grant_does_not_imply_the_send_capability(self):
        self.configure_mail(scopes=(pc.MAIL_READ, pc.MAIL_DRAFT))
        draft = self.mail_draft()
        approval = pc.request_effect_approval(draft["draft_id"])
        self.assertFalse(approval["ok"])
        self.assertIn(pc.MAIL_SEND, approval["error"])

    def test_wildcard_and_unknown_scopes_are_refused(self):
        self.configure_mail(scopes=())
        for bad in (["*"], ["all"], ["mail.everything"], ["calendar.read"]):
            refused = pc.grant("mail", MAIL, bad)
            self.assertFalse(refused["ok"], bad)
        self.assertEqual(pc.effective_scopes("mail", MAIL), [])

    def test_an_elevated_scope_needs_an_explicit_acknowledgement(self):
        self.configure_mail(scopes=())
        refused = pc.grant("mail", MAIL, [pc.MAIL_SEND])
        self.assertFalse(refused["ok"])
        self.assertIn("acknowledge_elevated", refused["error"])
        accepted = pc.grant("mail", MAIL, [pc.MAIL_SEND],
                            acknowledge_elevated=True)
        self.assertTrue(accepted["ok"], accepted)
        self.assertEqual(accepted["elevated_scopes"], [pc.MAIL_SEND])
        self.assertEqual(pc.effective_scopes("mail", MAIL), [pc.MAIL_SEND])

    def test_a_grant_is_bound_to_the_account_it_was_delegated_for(self):
        self.configure_mail(
            scopes=(),
            accounts=[{"id": MAIL, "primary": True}, {"id": WORK, "label": "w"}])
        granted = pc.grant("mail", WORK, [pc.MAIL_READ])
        self.assertTrue(granted["ok"], granted)
        # The same service, a different account: no delegated authority.
        refused = pc.run_operation("mail.list_messages", {"account": MAIL},
                                   transport=_ExplodingTransport())
        self.assertFalse(refused["ok"])
        self.assertIn("no permission is delegated", refused["error"])
        allowed = pc.run_operation("mail.list_messages", {"account": WORK},
                                   transport=self.transport)
        self.assertTrue(allowed["ok"], allowed)

    def test_a_grant_expires(self):
        pc.configure_service("mail", "google", MAIL, token="t", confirmed=True)
        self.assertTrue(pc.grant("mail", MAIL, [pc.MAIL_READ],
                                 ttl=10, now=1000.0)["ok"])
        refused = pc.run_operation("mail.list_messages", {"account": MAIL},
                                   transport=_ExplodingTransport(),
                                   now=2000.0)
        self.assertFalse(refused["ok"])
        self.assertIn("expired", refused["error"])

    def test_grant_state_is_persisted_for_the_next_process(self):
        self.configure_mail(scopes=(pc.MAIL_READ,))
        # Nothing is cached in-process: the store on disk is the authority.
        with open(pc.config_path(), encoding="utf-8") as handle:
            on_disk = json.load(handle)
        self.assertEqual(on_disk["grants"][0]["scopes"], [pc.MAIL_READ])
        self.assertNotIn("token", on_disk["grants"][0])


# ── 4. revocation ──────────────────────────────────────────────────────────
class RevocationTests(ConnectorTestCase):
    def test_revocation_stops_reads_immediately(self):
        self.configure_mail(scopes=(pc.MAIL_READ,))
        self.assertTrue(pc.run_operation("mail.list_messages",
                                         {"account": MAIL},
                                         transport=self.transport)["ok"])
        pc.revoke("mail", MAIL)
        refused = pc.run_operation("mail.list_messages", {"account": MAIL},
                                   transport=_ExplodingTransport())
        self.assertFalse(refused["ok"])
        self.assertEqual(pc.effective_scopes("mail", MAIL), [])

    def test_revocation_kills_an_outstanding_approval(self):
        self.configure_mail(scopes=SEND_ALL)
        draft = self.mail_draft()
        approval = pc.request_effect_approval(draft["draft_id"])
        self.assertTrue(approval["ok"], approval)
        pc.revoke("mail", MAIL, scopes=[pc.MAIL_SEND])
        transport = FakeTransport()
        refused = pc.run_operation(
            "mail.send_draft",
            {"account": MAIL, "draft_id": draft["draft_id"],
             "approval_id": approval["approval_id"]},
            transport=transport)
        self.assertFalse(refused["ok"])
        self.assertEqual(transport.effects, [])

    def test_revocation_invalidates_an_approval_even_when_the_draft_is_unchanged(self):
        self.configure_mail(scopes=SEND_ALL)
        draft = self.mail_draft()
        approval = pc.request_effect_approval(draft["draft_id"])
        pc.revoke("mail", MAIL)
        # Re-granting the ability to send does not resurrect the old consent:
        # the epoch moved on when the permissions were withdrawn.
        self.assertTrue(pc.grant("mail", MAIL, list(SEND_ALL),
                                 acknowledge_elevated=True)["ok"])
        transport = FakeTransport()
        refused = pc.run_operation(
            "mail.send_draft",
            {"account": MAIL, "draft_id": draft["draft_id"],
             "approval_id": approval["approval_id"]},
            transport=transport)
        self.assertFalse(refused["ok"])
        self.assertTrue("revoked" in refused["error"]
                        or "permissions changed" in refused["error"],
                        refused["error"])
        self.assertEqual(transport.effects, [])

    def test_revoking_one_account_leaves_the_others_permissions_alone(self):
        self.configure_mail(
            scopes=(),
            accounts=[{"id": MAIL, "primary": True}, {"id": WORK, "label": "w"}])
        pc.grant("mail", MAIL, [pc.MAIL_READ])
        pc.grant("mail", WORK, [pc.MAIL_READ])
        pc.revoke("mail", MAIL)
        refused = pc.run_operation("mail.list_messages", {"account": MAIL},
                                   transport=_ExplodingTransport())
        self.assertFalse(refused["ok"])
        allowed = pc.run_operation("mail.list_messages", {"account": WORK},
                                   transport=FakeTransport())
        self.assertTrue(allowed["ok"], allowed)

    def test_withdrawing_a_service_revokes_its_grants_and_approvals(self):
        self.configure_mail(scopes=SEND_ALL)
        draft = self.mail_draft()
        approval = pc.request_effect_approval(draft["draft_id"])
        removed = pc.unconfigure_service("mail")
        self.assertTrue(removed["ok"], removed)
        self.assertTrue(removed["revoked"])
        self.assertGreaterEqual(removed["invalidated_approvals"], 1)
        self.assertEqual(pc.effective_scopes("mail", MAIL), [])
        transport = FakeTransport()
        refused = pc.run_operation(
            "mail.send_draft",
            {"account": MAIL, "draft_id": draft["draft_id"],
             "approval_id": approval["approval_id"]},
            transport=transport)
        self.assertFalse(refused["ok"])
        self.assertEqual(transport.effects, [])

    def test_revocation_survives_a_reload_of_the_store(self):
        self.configure_mail(scopes=(pc.MAIL_READ,))
        self.assertTrue(pc.run_operation("mail.list_messages",
                                         {"account": MAIL},
                                         transport=self.transport)["ok"])
        pc.revoke("mail", MAIL)
        self.assertEqual(pc._load()["grants"], [])
        self.assertEqual(pc.effective_scopes("mail", MAIL), [])
        refused = pc.run_operation("mail.list_messages", {"account": MAIL},
                                   transport=_ExplodingTransport())
        self.assertFalse(refused["ok"])


# ── 5. no unauthorized send/create effects ─────────────────────────────────
class NoUnauthorizedEffectsTests(ConnectorTestCase):
    def test_every_externally_visible_operation_requires_an_approval_id(self):
        for name, spec in pc.OPERATIONS.items():
            if spec["operation"] in pc.EXTERNAL_EFFECTS:
                self.assertIn("approval_id", spec["required"],
                              "%s can change the outside world without "
                              "consent" % name)

    def test_no_registered_operation_is_a_hidden_send_or_create(self):
        visible = {name for name, spec in pc.OPERATIONS.items()
                   if spec["operation"] in pc.EXTERNAL_EFFECTS}
        self.assertEqual(visible, {"mail.send_draft", "calendar.commit_event"})
        # Each visible effect declares exactly the effect it performs, so a
        # caller cannot ask for one capability and get another.
        self.assertEqual(pc.OPERATIONS["mail.send_draft"]["effect"],
                         pc.MAIL_SEND)
        self.assertEqual(pc.OPERATIONS["calendar.commit_event"]["effect"],
                         pc.CALENDAR_WRITE)
        for name, spec in pc.OPERATIONS.items():
            if spec["operation"] in pc.EXTERNAL_EFFECTS:
                self.assertIn(spec["effect"], pc.ELEVATED_SCOPES, name)

    def test_holding_the_send_grant_still_sends_nothing_without_approval(self):
        self.configure_mail(scopes=SEND_ALL)
        draft = self.mail_draft()
        transport = FakeTransport()
        refused = pc.run_operation(
            "mail.send_draft",
            {"account": MAIL, "draft_id": draft["draft_id"],
             "approval_id": "made-up"},
            transport=transport)
        self.assertFalse(refused["ok"])
        self.assertEqual(transport.effects, [])

    def test_composing_a_draft_is_local_only(self):
        self.configure_mail()
        result = pc.run_operation(
            "mail.create_draft",
            {"account": MAIL, "to": ["a@b.com"], "subject": "s", "body": "b"},
            transport=_ExplodingTransport())
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["external_effect"])
        self.assertIn("draft", result["content"])

    def test_sending_needs_the_grant_and_the_approval_together(self):
        self.configure_mail(scopes=(pc.MAIL_READ, pc.MAIL_DRAFT))
        draft = self.mail_draft()
        # No elevated scope: consent cannot even be requested.
        self.assertFalse(
            pc.request_effect_approval(draft["draft_id"])["ok"])
        pc.grant("mail", MAIL, list(SEND_ALL), acknowledge_elevated=True)
        approval = pc.request_effect_approval(draft["draft_id"])
        self.assertTrue(approval["ok"], approval)
        transport = FakeTransport()
        sent = pc.run_operation(
            "mail.send_draft",
            {"account": MAIL, "draft_id": draft["draft_id"],
             "approval_id": approval["approval_id"]},
            transport=transport)
        self.assertTrue(sent["ok"], sent)
        self.assertTrue(sent["external_effect"])
        self.assertEqual([call[0] for call in transport.effects],
                         ["send_message"])

    def test_an_approval_is_bound_to_the_draft_it_was_requested_for(self):
        self.configure_mail(scopes=SEND_ALL)
        first = self.mail_draft(body="first")
        second = self.mail_draft(body="second")
        approval = pc.request_effect_approval(first["draft_id"])
        transport = FakeTransport()
        refused = pc.run_operation(
            "mail.send_draft",
            {"account": MAIL, "draft_id": second["draft_id"],
             "approval_id": approval["approval_id"]},
            transport=transport)
        self.assertFalse(refused["ok"])
        self.assertIn("different draft", refused["error"])
        self.assertEqual(transport.effects, [])

    def test_a_changed_draft_cannot_use_its_old_approval(self):
        self.configure_mail(scopes=SEND_ALL)
        draft = self.mail_draft(body="original")
        approval = pc.request_effect_approval(draft["draft_id"])
        # Simulate an edit of the stored draft after consent was given.
        config = pc._load()
        for record in config["drafts"]:
            if record["draft_id"] == draft["draft_id"]:
                record["fields"]["body"] = "wire the money"
        pc._save(config)
        transport = FakeTransport()
        refused = pc.run_operation(
            "mail.send_draft",
            {"account": MAIL, "draft_id": draft["draft_id"],
             "approval_id": approval["approval_id"]},
            transport=transport)
        self.assertFalse(refused["ok"])
        self.assertEqual(transport.effects, [])

    def test_an_approval_is_single_use(self):
        self.configure_mail(scopes=SEND_ALL)
        draft = self.mail_draft()
        approval = pc.request_effect_approval(draft["draft_id"])
        transport = FakeTransport()
        self.assertTrue(pc.run_operation(
            "mail.send_draft",
            {"account": MAIL, "draft_id": draft["draft_id"],
             "approval_id": approval["approval_id"]},
            transport=transport)["ok"])
        replayed = pc.run_operation(
            "mail.send_draft",
            {"account": MAIL, "draft_id": draft["draft_id"],
             "approval_id": approval["approval_id"]},
            transport=transport)
        self.assertFalse(replayed["ok"])
        self.assertIn("already used", replayed["error"])
        self.assertEqual(len(transport.effects), 1)

    def test_an_expired_approval_sends_nothing(self):
        pc.configure_service("mail", "google", MAIL, token="t", confirmed=True)
        pc.grant("mail", MAIL, list(SEND_ALL), acknowledge_elevated=True,
                 now=1000.0)
        draft = pc.run_operation(
            "mail.create_draft",
            {"account": MAIL, "to": ["a@b.com"], "subject": "s", "body": "b"},
            transport=FakeTransport(), now=1000.0)
        approval = pc.request_effect_approval(draft["draft_id"], ttl=10,
                                              now=1000.0)
        transport = FakeTransport()
        refused = pc.run_operation(
            "mail.send_draft",
            {"account": MAIL, "draft_id": draft["draft_id"],
             "approval_id": approval["approval_id"]},
            transport=transport, now=2000.0)
        self.assertFalse(refused["ok"])
        self.assertIn("expired", refused["error"])
        self.assertEqual(transport.effects, [])

    def test_a_mail_approval_cannot_authorise_a_calendar_create(self):
        """Asking for one effect must never execute another."""
        self.configure_mail(scopes=SEND_ALL)
        self.configure_calendar(scopes=(pc.CALENDAR_READ, pc.CALENDAR_DRAFT,
                                        pc.CALENDAR_WRITE))
        draft = self.mail_draft()
        approval = pc.request_effect_approval(draft["draft_id"])
        transport = FakeTransport()
        refused = pc.run_operation(
            "calendar.commit_event",
            {"account": MAIL, "draft_id": draft["draft_id"],
             "approval_id": approval["approval_id"]},
            transport=transport)
        self.assertFalse(refused["ok"])
        self.assertIn("effect", refused["error"])
        # Neither an event was created nor the mail sent.
        self.assertEqual(transport.effects, [])

    def test_a_calendar_event_is_created_only_after_its_own_approval(self):
        self.configure_calendar(
            scopes=(pc.CALENDAR_READ, pc.CALENDAR_DRAFT, pc.CALENDAR_WRITE))
        proposal = pc.run_operation(
            "calendar.propose_event",
            {"account": MAIL, "title": "Standup",
             "start": "2026-03-02T09:00:00", "end": "2026-03-02T09:15:00",
             "attendees": ["a@b.com"]},
            transport=_ExplodingTransport())
        self.assertTrue(proposal["ok"], proposal)
        self.assertFalse(proposal["external_effect"])
        approval = pc.request_effect_approval(proposal["draft_id"])
        self.assertTrue(approval["ok"], approval)
        transport = FakeTransport()
        created = pc.run_operation(
            "calendar.commit_event",
            {"account": MAIL, "draft_id": proposal["draft_id"],
             "approval_id": approval["approval_id"]},
            transport=transport)
        self.assertTrue(created["ok"], created)
        self.assertEqual([call[0] for call in transport.effects],
                         ["create_event"])
        self.assertEqual(transport.effects[0][2]["fields"]["title"], "Standup")

    def test_a_calendar_proposal_needs_a_valid_window(self):
        self.configure_calendar()
        backwards = pc.run_operation(
            "calendar.propose_event",
            {"account": MAIL, "title": "Nope",
             "start": "2026-03-02T10:00:00", "end": "2026-03-02T09:00:00"},
            transport=_ExplodingTransport())
        self.assertFalse(backwards["ok"])
        self.assertIn("after start", backwards["error"])

    def test_malformed_calls_do_nothing(self):
        self.configure_mail(scopes=SEND_ALL)
        cases = [
            ("mail.list_messages", {"account": MAIL, "limit": "many"}),
            ("mail.list_messages", {"account": MAIL, "bogus": 1}),
            ("mail.read_message", {"account": MAIL}),          # missing id
            ("mail.create_draft", {"account": MAIL}),          # missing to
            ("mail.send_draft", {"account": MAIL, "draft_id": "d"}),  # no appr
            ("not.a.tool", {"account": MAIL}),
        ]
        transport = _ExplodingTransport()
        for name, arguments in cases:
            result = pc.run_operation(name, arguments, transport=transport)
            self.assertFalse(result["ok"], (name, result))

    def test_an_empty_arguments_payload_is_refused_not_defaulted(self):
        self.configure_mail(scopes=SEND_ALL)
        transport = _ExplodingTransport()
        for name in ("mail.list_messages", "mail.read_message",
                     "mail.create_draft", "mail.send_draft",
                     "calendar.list_events", "calendar.commit_event",
                     "contacts.search", "contacts.resolve"):
            result = pc.run_operation(name, {}, transport=transport)
            self.assertFalse(result["ok"], name)

    def test_a_hand_edited_store_cannot_crash_a_gate_or_send(self):
        self.configure_mail(scopes=SEND_ALL)
        draft = self.mail_draft()
        approval = pc.request_effect_approval(draft["draft_id"])
        config = pc._load()
        for record in config["approvals"]:
            record["expires_at"] = "not-a-number"
            record["grant_epoch"] = "not-a-number"
        for grant_record in config["grants"]:
            grant_record["expires_at"] = "not-a-number"
        pc._save(config)
        transport = FakeTransport()
        refused = pc.run_operation(
            "mail.send_draft",
            {"account": MAIL, "draft_id": draft["draft_id"],
             "approval_id": approval["approval_id"]},
            transport=transport)
        self.assertFalse(refused["ok"])
        self.assertEqual(transport.effects, [])

    def test_the_typed_registry_is_self_describing(self):
        schemas = pc.tool_schemas()
        self.assertEqual(set(schemas), set(pc.TOOL_NAMES))
        for name, schema in schemas.items():
            self.assertFalse(
                schema["parameters"]["additionalProperties"],
                "%s accepts untyped extra arguments" % name)
            for required in schema["parameters"]["required"]:
                self.assertIn(required, schema["parameters"]["properties"])
        self.assertTrue(schemas["mail.send_draft"]["external_effect"])
        self.assertFalse(schemas["mail.create_draft"]["external_effect"])

    def test_the_planner_advertisement_matches_the_registry(self):
        lines = pc.planner_tool_lines()
        self.assertEqual(len(lines), len(pc.TOOL_NAMES))
        self.assertTrue(pc.is_productivity_tool("mail.send_draft"))
        self.assertFalse(pc.is_productivity_tool("mail.summon_demon"))
        send_line = [line for line in lines if line.startswith(
            "- mail.send_draft:")][0]
        self.assertIn("approval", send_line)
        draft_line = [line for line in lines if line.startswith(
            "- mail.create_draft:")][0]
        self.assertIn("nothing is sent", draft_line)


# ── 6. the real provider profiles (HTTP faked) ─────────────────────────────
class _FakeResponse:
    def __init__(self, payload=None, status=200, text=None):
        self._payload = payload if payload is not None else {}
        self.status_code = status
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _FakeSession:
    def __init__(self, responses=None):
        self.requests = []
        self._responses = list(responses or [])

    def request(self, method, url, params=None, json=None, headers=None,
                timeout=None):
        self.requests.append({"method": method, "url": url,
                              "params": params or {}, "body": json,
                              "headers": headers or {}, "timeout": timeout})
        payload = self._responses.pop(0) if self._responses else {}
        if isinstance(payload, _FakeResponse):
            return payload
        return _FakeResponse(payload)


class ProviderTransportTests(unittest.TestCase):
    def test_google_read_maps_to_a_bearer_get(self):
        session = _FakeSession([{"messages": [{"id": "1", "threadId": "t"}]}])
        transport = pp.HttpTransport("google", "sk-secret-token-value-000",
                                     session=session)
        messages = transport.list_messages(MAIL, "from:bob", 5, True)
        self.assertEqual([m["id"] for m in messages], ["1"])
        request = session.requests[0]
        self.assertEqual(request["method"], "GET")
        self.assertIn("gmail.googleapis.com", request["url"])
        self.assertEqual(request["params"]["maxResults"], 5)
        self.assertIn("is:unread", request["params"]["q"])
        self.assertEqual(request["headers"]["Authorization"],
                         "Bearer sk-secret-token-value-000")

    def test_google_send_posts_a_raw_rfc822_message(self):
        session = _FakeSession([{"id": "m1", "threadId": "t1"}])
        transport = pp.HttpTransport("google", "sk-secret-token-value-000",
                                     session=session)
        result = transport.send_message(MAIL, {
            "service": "mail", "account": MAIL,
            "fields": {"to": ["a@b.com"], "subject": "Report",
                       "body": "line"}})
        self.assertTrue(result["sent"])
        request = session.requests[0]
        self.assertEqual(request["method"], "POST")
        self.assertTrue(request["url"].endswith("/messages/send"))
        raw = request["body"]["raw"]
        import base64
        decoded = base64.urlsafe_b64decode(
            raw + "=" * (-len(raw) % 4)).decode("utf-8")
        self.assertIn("To: a@b.com", decoded)
        self.assertIn("Subject: Report", decoded)

    def test_graph_send_posts_sendmail_with_typed_recipients(self):
        session = _FakeSession([_FakeResponse(payload=None, status=202,
                                              text="")])
        transport = pp.HttpTransport("microsoft", "sk-secret-token-value-000",
                                     session=session)
        transport.send_message(MAIL, {
            "service": "mail", "account": MAIL,
            "fields": {"to": ["a@b.com"], "subject": "Hi", "body": "x"}})
        request = session.requests[0]
        self.assertTrue(request["url"].endswith("/me/sendMail"))
        self.assertEqual(
            request["body"]["message"]["toRecipients"][0]["emailAddress"]
            ["address"], "a@b.com")
        self.assertTrue(request["body"]["saveToSentItems"])

    def test_graph_create_event_posts_the_event(self):
        session = _FakeSession([{"id": "e1", "subject": "Standup"}])
        transport = pp.HttpTransport("microsoft", "sk-secret-token-value-000",
                                     session=session)
        result = transport.create_event(MAIL, {
            "service": "calendar", "account": MAIL,
            "fields": {"title": "Standup", "start": "2026-03-02T09:00:00",
                       "end": "2026-03-02T09:15:00",
                       "attendees": ["a@b.com"]}})
        self.assertTrue(result["created"])
        body = session.requests[0]["body"]
        self.assertEqual(body["subject"], "Standup")
        self.assertEqual(body["attendees"][0]["emailAddress"]["address"],
                         "a@b.com")

    def test_google_calendar_directory_parses_into_resolvable_entries(self):
        session = _FakeSession([
            {"items": [{"id": "primary", "summary": "Personal",
                        "primary": True},
                       {"id": "work@group", "summary": "Work"}]}])
        transport = pp.HttpTransport("google", "sk-secret-token-value-000",
                                     session=session)
        entries = transport.directory("calendar", MAIL, "", 10)
        self.assertEqual([entry["id"] for entry in entries],
                         ["primary", "work@group"])
        self.assertTrue(entries[0]["primary"])

    def test_a_failed_response_raises_and_never_leaks_the_token(self):
        token = "sk-live-abcdefghijklmnopqrstuvwxyz012345"
        session = _FakeSession([_FakeResponse(
            payload=None, status=401,
            text="Unauthorized: bearer %s rejected" % token)])
        transport = pp.HttpTransport("google", token, session=session)
        with self.assertRaises(pp.ProviderError) as caught:
            transport.list_messages(MAIL)
        self.assertEqual(caught.exception.status, 401)
        self.assertNotIn(token, str(caught.exception))

    def test_the_transport_repr_never_shows_the_token(self):
        transport = pp.HttpTransport("google", "sk-secret-token-value-000",
                                     session=_FakeSession())
        self.assertNotIn("sk-secret-token-value-000", repr(transport))

    def test_an_unmodelled_provider_is_refused(self):
        with self.assertRaises(pp.ProviderError):
            pp.HttpTransport("dropbox", "x", session=_FakeSession())

    def test_an_unconfigured_transport_makes_no_request(self):
        transport = pp.build_transport("", "")
        self.assertIsInstance(transport, pp.UnconfiguredTransport)
        with self.assertRaises(pp.ProviderError):
            transport.list_messages(MAIL)
        with self.assertRaises(pp.ProviderError):
            transport.send_message(MAIL, {})

    def test_the_custom_profile_uses_the_users_own_endpoints(self):
        session = _FakeSession([{"items": [{"id": "c1", "name": "Work"}]}])
        transport = pp.HttpTransport(
            "custom", "sk-secret-token-value-000", session=session,
            config={"base_url": "https://intranet.example/api",
                    "endpoints": {"directory": "/dir"}})
        entries = transport.directory("calendar", MAIL, "work", 5)
        self.assertEqual(entries[0]["id"], "c1")
        self.assertEqual(session.requests[0]["url"],
                         "https://intranet.example/api/dir")

    def test_the_connector_drives_the_real_profile_behind_its_gates(self):
        """End-to-end: connector gates + real Google profile + fake HTTP."""
        tmp = tempfile.TemporaryDirectory()
        saved = os.environ.get(pc.CONFIG_ENV)
        os.environ[pc.CONFIG_ENV] = os.path.join(tmp.name, "config.json")
        try:
            pc.configure_service("mail", "google", MAIL, token="sk-token-abcdefghijklmnop", confirmed=True)
            pc.grant("mail", MAIL, [pc.MAIL_READ])
            session = _FakeSession([
                {"messages": [{"id": "1", "threadId": "t"}]}])
            transport = pp.HttpTransport("google", "sk-token-abcdefghijklmnop",
                                         session=session)
            listed = pc.run_operation("mail.list_messages", {"account": MAIL},
                                      transport=transport)
            self.assertTrue(listed["ok"], listed)
            self.assertEqual(listed["data"][0]["id"], "1")
            # Without the draft scope no draft is stored and no request is
            # made for it.
            refused = pc.run_operation(
                "mail.create_draft",
                {"account": MAIL, "to": ["a@b.com"], "subject": "s",
                 "body": "b"}, transport=transport)
            self.assertFalse(refused["ok"])
            self.assertEqual(len(session.requests), 1)
        finally:
            if saved is None:
                os.environ.pop(pc.CONFIG_ENV, None)
            else:
                os.environ[pc.CONFIG_ENV] = saved
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
