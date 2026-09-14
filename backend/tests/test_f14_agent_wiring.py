"""F14 — the semantic productivity connectors wired into the task agent.

Acceptance (audit report): "Provide source/tests for account/entity
resolution, delegated permissions, revocation, and no unauthorized
send/create effects."

The connector itself (registry, grants, revocation, approval binding) is
covered by ``test_f14_productivity_connectors``. This module pins the OTHER
half — the wiring in ``task_agent/agent.py``:

  * the planner is advertised the connector's own typed operations;
  * dispatch goes through ``productivity_connector.run_operation`` beside the
    native code tools;
  * a send/create obtains a SEPARATE per-draft approval
    (``request_effect_approval``), whose scrubbed preview is what the user
    approves, and executes only after that approval is re-verified;
  * a step without the approval, without the elevated grant, or carrying
    another draft's approval executes nothing.

Only the provider HTTP layer is faked (the transport); nothing here reaches a
real service, model or editor.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from backend.services import approvals
from backend.services import productivity_connector as pc
from backend.services.task_agent import agent

MAIL = "me@example.test"
TO = "friend@example.test"


class FakeTransport:
    """Stands in for the mail provider; records every call it receives."""

    provider = "fake"

    def __init__(self, messages=None, sent=None):
        self.calls = []
        self._messages = messages if messages is not None else [
            {"id": "m-1", "date": "2024-01-01", "from": "a@b.test",
             "subject": "Quarterly numbers"},
        ]
        self._sent = sent or {"id": "m-sent", "sent": True}

    @property
    def effects(self):
        return [call for call in self.calls
                if call[0] in ("send_message", "create_event")]

    def directory(self, service, account, query="", limit=20):
        self.calls.append(("directory", service, account, query, limit))
        return []

    def list_messages(self, account, query="", limit=20, unread_only=False):
        self.calls.append(("list_messages", account, query, limit, unread_only))
        return list(self._messages)

    def get_message(self, account, message_id):
        self.calls.append(("get_message", account, message_id))
        return dict(self._messages[0]) if self._messages else {}

    def send_message(self, account, draft):
        self.calls.append(("send_message", account, draft))
        return dict(self._sent)

    def list_events(self, account, start="", end="", limit=20):
        self.calls.append(("list_events", account, start, end, limit))
        return []

    def free_busy(self, account, start, end):
        self.calls.append(("free_busy", account, start, end))
        return {"busy": []}

    def create_event(self, account, draft):
        self.calls.append(("create_event", account, draft))
        return {"id": "e-1", "created": True}


class AgentProductivityTestCase(unittest.TestCase):
    """Isolated connector store; the provider transport is faked."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._saved_env = os.environ.get(pc.CONFIG_ENV)
        os.environ[pc.CONFIG_ENV] = os.path.join(self._tmp.name,
                                                 "productivity.json")
        self.addCleanup(self._restore_env)
        self.transport = FakeTransport()
        transport_patch = patch.object(pc, "_transport_for",
                                       return_value=self.transport)
        transport_patch.start()
        self.addCleanup(transport_patch.stop)
        agent._pending_task_action = None
        self.addCleanup(setattr, agent, "_pending_task_action", None)

    def _restore_env(self):
        if self._saved_env is None:
            os.environ.pop(pc.CONFIG_ENV, None)
        else:
            os.environ[pc.CONFIG_ENV] = self._saved_env
        approvals.clear()
        agent._pending_task_action = None

    # -- helpers --
    def configure_mail(self, scopes=(pc.MAIL_READ, pc.MAIL_DRAFT)):
        result = pc.configure_service("mail", "google", MAIL,
                                      token="test-token", confirmed=True)
        self.assertTrue(result["ok"], result)
        granted = pc.grant("mail", MAIL, list(scopes),
                           acknowledge_elevated=pc.MAIL_SEND in scopes)
        self.assertTrue(granted["ok"], granted)

    def make_draft(self, subject="hi", body="hello there"):
        result = pc.run_operation(
            "mail.create_draft",
            {"account": MAIL, "to": [TO], "subject": subject, "body": body},
            transport=self.transport)
        self.assertTrue(result["ok"], result)
        return result

    def send_plan(self, draft_id, risk="safe"):
        return {
            "ok": True,
            "confidence": 0.9,
            "summary": "Sending the mail.",
            "steps": [{"tool": "mail.send_draft",
                       "args": {"account": MAIL, "draft_id": draft_id},
                       "risk": risk, "reason": "Sending the prepared mail."}],
        }


# ── 1. the planner is advertised the typed connector operations ────────────
class PlannerAdvertisementTests(AgentProductivityTestCase):

    def test_planner_prompt_lists_the_connector_operations(self):
        prompt = agent._build_planner_prompt("what's on my calendar", {})
        for name in pc.TOOL_NAMES:
            self.assertIn(name + ":", prompt,
                          "the planner must be offered %s" % name)
        self.assertIn("productivity.services:", prompt)
        self.assertIn("mail.send_draft:", prompt)
        self.assertIn("request_effect_approval", prompt)
        self.assertIn("local draft, nothing is sent or created", prompt)

    def test_read_and_draft_operations_are_least_privilege(self):
        # F14: reads/drafts are gated by the connector's delegated scope, not
        # by an extra whole-plan confirmation; sends/creates are not "safe".
        self.assertIn("mail.list_messages", agent.SAFE_TOOLS)
        self.assertIn("mail.create_draft", agent.SAFE_TOOLS)
        self.assertNotIn("mail.send_draft", agent.SAFE_TOOLS)
        self.assertIn("mail.send_draft", agent.CONFIRM_TOOLS)
        plan = agent._normalize_plan(
            {"ok": True, "summary": "s", "steps": [
                {"tool": "mail.list_messages", "args": {"account": MAIL},
                 "risk": "safe", "reason": "t"}]},
            "check my mail")
        self.assertFalse(plan["requires_confirmation"])


# ── 2. dispatch goes through the connector ─────────────────────────────────
class ConnectorDispatchTests(AgentProductivityTestCase):

    def test_a_read_is_dispatched_to_the_connector(self):
        self.configure_mail()
        text = agent._execute_step(
            {"tool": "mail.list_messages", "args": {"account": MAIL}}, {})
        self.assertEqual([call[0] for call in self.transport.calls],
                         ["list_messages"])
        self.assertIn("Quarterly numbers", text)

    def test_the_structured_connector_payload_is_preserved(self):
        self.configure_mail()
        text, structured = agent._execute_step_structured(
            {"tool": "mail.list_messages", "args": {"account": MAIL}}, {})
        self.assertIn("Quarterly numbers", text)
        self.assertTrue(structured["ok"])
        self.assertEqual(structured["tool"], "mail.list_messages")
        self.assertEqual(structured["data"][0]["id"], "m-1")

    def test_an_unconfigured_service_reports_a_failure_with_no_request(self):
        text, structured = agent._execute_step_structured(
            {"tool": "mail.list_messages", "args": {"account": MAIL}}, {})
        self.assertTrue(text.startswith("mail.list_messages failed:"),
                        text)
        self.assertFalse(structured["ok"])
        self.assertEqual(self.transport.calls, [])

    def test_another_draft_s_approval_cannot_authorise_this_effect(self):
        self.configure_mail(scopes=(pc.MAIL_READ, pc.MAIL_DRAFT, pc.MAIL_SEND))
        first = self.make_draft(subject="first")
        other = self.make_draft(subject="other")
        approval = pc.request_effect_approval(first["draft_id"])
        self.assertTrue(approval["ok"], approval)
        step = {"tool": "mail.send_draft",
                "args": {"account": MAIL, "draft_id": other["draft_id"],
                         "approval_id": approval["approval_id"]},
                "risk": "confirm", "reason": "t"}
        result = agent.execute_plan(
            {"ok": True, "summary": "s", "steps": [step]}, {}, confirmed=True)
        self.assertNotEqual(result.status, "completed")
        self.assertEqual(self.transport.effects, [])


# ── 3. a send/create needs its own approval, shown to the user ─────────────
class EffectApprovalWiringTests(AgentProductivityTestCase):

    def test_planning_obtains_the_approval_and_shows_its_preview(self):
        self.configure_mail(scopes=(pc.MAIL_READ, pc.MAIL_DRAFT, pc.MAIL_SEND))
        draft = self.make_draft()
        self.assertEqual(self.transport.effects, [],
                         "composing a draft is local-only")
        normalized = agent._normalize_plan(
            self.send_plan(draft["draft_id"]), "send the mail")
        step = normalized["steps"][0]
        self.assertTrue(normalized["requires_confirmation"],
                        "an externally visible send is never implicit")
        self.assertTrue(step["args"].get("approval_id"),
                        "the step must carry a separate approval id")
        preview = step.get("effect_preview") or ""
        self.assertIn("Send mail from %s to %s" % (MAIL, TO), preview)
        spoken = agent._confirmation_preview(normalized)
        self.assertIn(preview, spoken,
                      "the approval preview must be what the user hears")
        self.assertIn("approval", spoken)
        self.assertEqual(self.transport.effects, [],
                         "planning performed no send")

    def test_the_confirmed_effect_executes_exactly_once(self):
        self.configure_mail(scopes=(pc.MAIL_READ, pc.MAIL_DRAFT, pc.MAIL_SEND))
        draft = self.make_draft(body="the real body")
        normalized = agent._normalize_plan(
            self.send_plan(draft["draft_id"]), "send the mail")
        result = agent.execute_plan(normalized, {}, confirmed=True)
        self.assertEqual(result.status, "completed")
        self.assertEqual(len(self.transport.effects), 1)
        self.assertEqual(
            self.transport.effects[0][2]["fields"]["body"], "the real body")

        # Replaying the same confirmed plan cannot send twice: the approval
        # was consumed, and the connector refuses the replay.
        replay = agent.execute_plan(normalized, {}, confirmed=True)
        self.assertNotEqual(replay.status, "completed")
        self.assertEqual(len(self.transport.effects), 1)
        self.assertTrue(any("already used" in str(line)
                            for line in replay.evidence),
                        replay.evidence)

    def test_a_send_step_without_an_approval_executes_nothing(self):
        self.configure_mail(scopes=(pc.MAIL_READ, pc.MAIL_DRAFT, pc.MAIL_SEND))
        draft = self.make_draft()
        # A raw plan that never went through the normalizer has no approval.
        result = agent.execute_plan(
            {"ok": True, "summary": "s",
             "steps": [{"tool": "mail.send_draft",
                        "args": {"account": MAIL,
                                 "draft_id": draft["draft_id"]},
                        "risk": "confirm", "reason": "t"}]},
            {}, confirmed=True)
        self.assertNotEqual(result.status, "completed")
        self.assertEqual(self.transport.effects, [])
        self.assertTrue(any("no separate approval" in str(line)
                            for line in result.evidence), result.evidence)

    def test_a_missing_draft_is_a_precondition_failure_with_no_effect(self):
        self.configure_mail(scopes=(pc.MAIL_READ, pc.MAIL_DRAFT, pc.MAIL_SEND))
        plan = agent._normalize_plan(
            {"ok": True, "summary": "s", "steps": [
                {"tool": "mail.send_draft",
                 "args": {"account": MAIL}, "risk": "confirm", "reason": "t"}]},
            "send it")
        step = plan["steps"][0]
        self.assertIn("precondition_error", step)
        result = agent.execute_plan(plan, {}, confirmed=True)
        self.assertNotEqual(result.status, "completed")
        self.assertEqual(self.transport.effects, [])

    def test_the_consent_gate_shows_the_effect_and_sends_once(self):
        """The whole user flow: propose (preview spoken) -> confirm -> send."""
        self.configure_mail(scopes=(pc.MAIL_READ, pc.MAIL_DRAFT, pc.MAIL_SEND))
        draft = self.make_draft(body="the approved body")
        plan = agent._normalize_plan(
            self.send_plan(draft["draft_id"]), "send the mail")
        proposed = agent.execute_plan(plan, {}, confirmed=False)
        self.assertEqual(proposed.status, "needs_input")
        self.assertIn("Send mail from %s to %s" % (MAIL, TO), str(proposed))
        self.assertEqual(self.transport.effects, [],
                         "nothing is sent before the confirmation")
        outcome = agent.consume_task_confirmation("confirm task")
        self.assertEqual(len(self.transport.effects), 1)
        self.assertIn("Sent", outcome)

    def test_without_the_elevated_grant_the_effect_is_refused(self):
        # mail.send was never delegated: the draft is fine, the approval is
        # not grantable, and the send executes nothing.
        self.configure_mail(scopes=(pc.MAIL_READ, pc.MAIL_DRAFT))
        draft = self.make_draft()
        plan = agent._normalize_plan(
            self.send_plan(draft["draft_id"]), "send the mail")
        step = plan["steps"][0]
        self.assertTrue(plan["requires_confirmation"])
        self.assertIn("precondition_error", step)
        self.assertNotIn("approval_id", step["args"])
        result = agent.execute_plan(plan, {}, confirmed=True)
        self.assertNotEqual(result.status, "completed")
        self.assertEqual(self.transport.effects, [])
        self.assertTrue(any("mail.send" in str(line)
                            for line in result.evidence), result.evidence)


if __name__ == "__main__":
    unittest.main()
