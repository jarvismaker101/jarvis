"""G8 Round 1 (F02/F16/F46/F49): the orchestrator migration behind a flag.

  F46 — one bounded request-scoped ContextEnvelope; fields budgeted before
        serialization with EXPLICIT omitted-field indicators; oversized
        connector shapes are summarized, never halved.
  F16 — dispatch by capability: deterministic code tools / editor bridge /
        browser agent; AVAILABILITY separated from PERMISSION; the
        opencode-installation prerequisite for browser recovery is removed;
        opencode stays opt-in (never a fallback).
  F49 — planner role + capability validation in the model registry;
        task-agent _model_plan resolved through the registry (one registry).
  F02 — native tool-use orchestrator behind the migration mode flag:
        tool arguments validated, invalid output distinguished from a
        deliberate conversational answer, mutations only ever PROPOSED.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from backend.services import capability_resolver as resolver
from backend.services import context_envelope as envelope_mod
from backend.services import model_registry
from backend.services import orchestrator
from backend.services.task_agent import agent as task_agent


def _fake_tool_response(tool, args, content=""):
    return {
        "choices": [{
            "message": {
                "content": content,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(args)},
                }],
            }
        }]
    }


def _fake_answer_response(text):
    return {"choices": [{"message": {"content": text, "tool_calls": []}}]}


class ContextEnvelopeTests(unittest.TestCase):
    """F46 — budgeted fields, explicit omissions, resolved references."""

    def test_long_utterance_clipped_explicitly(self):
        env = envelope_mod.ContextEnvelope(utterance="word " * 900)
        self.assertLessEqual(len(env.fields["utterance"]),
                             envelope_mod.FIELD_BUDGETS["utterance"])
        self.assertIn("…[clipped]", env.fields["utterance"])

    def test_oversized_references_summarized_not_halved(self):
        huge = {"tabs": ["tab%d" % i for i in range(400)]}
        env = envelope_mod.ContextEnvelope(utterance="hi", references=huge)
        rendered = env.render()
        self.assertLessEqual(len(rendered), envelope_mod.TOTAL_BUDGET)
        self.assertIn("_note", json.dumps(env.fields["references"]))
        self.assertTrue(env.omissions or "_note" in json.dumps(env.fields["references"]))

    def test_total_budget_drops_whole_fields_with_omission_list(self):
        env = envelope_mod.ContextEnvelope(
            utterance="look",
            goal_state="x " * 1500,
            editor_text="y " * 1500,
            browser_target="z " * 1500,
            memory_hints="m " * 1500,
            capture_identity="c " * 1500,
        )
        rendered = env.render()
        self.assertLessEqual(len(rendered), envelope_mod.TOTAL_BUDGET)
        self.assertTrue(env.omissions, "whole-field drops must be listed")
        self.assertIn("[omitted:", rendered)
        # The utterance is never dropped — only optional fields.
        self.assertIn("utterance: look", rendered)

    def test_build_envelope_resolves_references(self):
        connectors = {
            "windows": {"active_window": {"title": "VS Code"}},
            "editor": {"available": True, "state": {"activeFile": {
                "path": "C:/x/a.py", "version": 7}}},
            "browser": {"available": True, "tabs": [{"title": "GitHub"}]},
        }
        env = envelope_mod.build_envelope("fix this error", connectors=connectors)
        self.assertEqual(env.fields["references"]["active_window"], "VS Code")
        self.assertEqual(env.fields["references"]["editor_file"], "C:/x/a.py")
        self.assertEqual(env.fields["references"]["editor_version"], 7)
        self.assertIn("GitHub", env.fields["references"]["browser_tabs"])


class CapabilityResolverTests(unittest.TestCase):
    """F16 — capability dispatch; availability vs permission."""

    def test_deterministic_local_file_op_never_hands_off(self):
        decision = resolver.resolve_engine("read C:/x/notes.txt")
        self.assertEqual(decision["engine"], resolver.CODE_TOOLS)

    def test_web_target_goes_to_browser_agent(self):
        decision = resolver.resolve_engine("open https://youtube.com and play")
        self.assertEqual(decision["engine"], resolver.BROWSER_AGENT)

    def test_opencode_never_starts_when_unavailable(self):
        # F16 (updated): an explicitly requested coding agent whose CLI is
        # unavailable now FAILS CLOSED. The browser agent is not a coding
        # executor, so it is not a compatible fallback for a coding request.
        decision = resolver.resolve_engine(
            "take over and fix the build", availability={"opencode": False})
        self.assertEqual(decision["engine"], resolver.BLOCKED)
        self.assertTrue(decision["opt_in"])
        self.assertFalse(decision["compatible"])

    def test_explicit_opt_in_and_available_uses_opencode(self):
        decision = resolver.resolve_engine(
            "take over and fix the build", availability={"opencode": True})
        self.assertEqual(decision["engine"], resolver.OPENCODE)

    def test_recovery_never_requires_opencode_installation(self):
        decision = resolver.recovery_after_failure(
            "open youtube", availability={"opencode": False})
        self.assertEqual(decision["engine"], resolver.BROWSER_AGENT)
        self.assertIn("recovery", decision["reason"])

    def test_recovery_keeps_explicit_opencode_opt_in(self):
        decision = resolver.recovery_after_failure(
            "take over and fix the build", availability={"opencode": True})
        self.assertEqual(decision["engine"], resolver.OPENCODE)

    def test_configured_opencode_default_falls_back_to_browser_agent(self):
        decision = resolver.resolve_engine(
            "do the whole thing", availability={"opencode": False},
            task_engine="opencode")
        self.assertEqual(decision["engine"], resolver.BROWSER_AGENT)

    def test_unavailable_editor_skips_editor_engine(self):
        decision = resolver.resolve_engine(
            "fix unsaved selection in editor", availability={"editor": False})
        self.assertNotEqual(decision["engine"], resolver.EDITOR_TOOLS)


class PlannerCapabilityTests(unittest.TestCase):
    """F49 — planner role + capability validation (one registry)."""

    def test_planner_role_is_valid_and_default_resolves(self):
        self.assertIn("planner", model_registry.VALID_ROLES)
        snapshot = model_registry.get_model_config("planner")
        self.assertEqual(snapshot["provider"], "fireworks")
        self.assertIn("qwen3p7-plus", snapshot["model"])
        self.assertIn("tool_calling", snapshot["required"])

    def test_capability_validation_rejects_unfit_providers(self):
        # fish has audio_output only — it lacks the planner's requirements.
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.validate_role_capabilities("planner", "fish", "m")
        # whisper has speech_input only — no chat streaming.
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.validate_role_capabilities("chat", "whisper", "w")
        # F49: an UNREGISTERED provider id inherits NOTHING — "acme" is not a
        # stored custom provider in this test, so the selection fails closed
        # instead of silently borrowing the OpenAI-compatible floor.
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.validate_role_capabilities("planner", "acme", "m")
        # A fit env provider passes.
        ok = model_registry.validate_role_capabilities("planner", "fireworks", "m")
        self.assertEqual(ok["role"], "planner")

    def test_set_model_for_role_rejects_capability_mismatch(self):
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role("planner", "whisper", "w")

    def test_unknown_role_raises(self):
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.get_model_config("bogus")


class ModelPlanRegistryWiringTests(unittest.TestCase):
    """F49 — task-agent planning goes through the registry, not a bare default."""

    def test_model_plan_uses_registry_snapshot(self):
        snapshot = {
            "role": "planner", "provider": "fireworks",
            "model": "accounts/fireworks/models/qwen3p7-plus",
            "required": ["tool_calling"], "capabilities": ["tool_calling"],
            "resolved_at": None,
        }
        captured = {}

        def fake_fireworks(messages, temperature=0.7, max_tokens=None, model=None, timeout=None):
            captured["model"] = model
            return {"choices": [{"message": {"content": json.dumps({
                "ok": True, "confidence": 0.9, "summary": "planned",
                "requires_confirmation": False,
                "steps": [{"tool": "code.read_file", "args": {"path": "C:/x/a.txt"},
                          "risk": "safe", "reason": "r"}],
                "response": "",
            })}}]}

        with patch.object(model_registry, "get_model_config", return_value=snapshot), \
             patch.object(model_registry, "get_provider_credentials",
                          return_value=("key", None)), \
             patch.object(task_agent, "ask_fireworks", side_effect=fake_fireworks):
            plan = task_agent._model_plan("do something", {})

        self.assertEqual(captured["model"], "accounts/fireworks/models/qwen3p7-plus")
        self.assertTrue(plan.get("ok"))


class OrchestratorTests(unittest.TestCase):
    """F02 — native tool use behind the migration flag; nothing executes
    mutations; invalid output is never mistaken for an answer."""

    def test_default_mode_is_legacy_so_handle_message_declines(self):
        self.assertIsNone(orchestrator.handle_message("hello", screen_question=False))

    def test_live_mode_flag_read(self):
        from backend import config
        self.assertFalse(orchestrator.orchestrator_mode())
        with patch.object(config, "ORCHESTRATOR_MODE", "orchestrator"):
            self.assertTrue(orchestrator.orchestrator_mode())

    def test_conversational_turn_is_the_answer(self):
        env = envelope_mod.ContextEnvelope(utterance="how are you")
        with patch.object(orchestrator, "_chat", return_value=_fake_answer_response("Fine, sir.")):
            outcome = orchestrator.run_orchestrator("how are you", env)
        self.assertEqual(outcome["status"], "answered")
        self.assertEqual(outcome["reply"], "Fine, sir.")
        self.assertEqual(outcome["actions"], [])

    def test_tool_sequence_inspect_then_answer(self):
        env = envelope_mod.ContextEnvelope(utterance="whats the price of claude fable 5.1")
        chat = MagicMock(side_effect=[
            _fake_tool_response("research.lookup", {"query": "claude fable 5.1 price"}),
            _fake_answer_response("It costs 5 credits, sir."),
        ])
        with patch.object(orchestrator, "_chat", chat), \
             patch.object(orchestrator, "_tool_research_lookup",
                          return_value="5 credits per month"):
            outcome = orchestrator.run_orchestrator("price of claude fable 5.1", env)
        self.assertEqual(outcome["status"], "answered")
        self.assertEqual(outcome["reply"], "It costs 5 credits, sir.")
        self.assertEqual(outcome["actions"][0]["tool"], "research.lookup")
        self.assertEqual(chat.call_count, 2)

    def test_invalid_tool_arguments_fed_back_never_executed(self):
        env = envelope_mod.ContextEnvelope(utterance="look up something")
        bad_call = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "c1", "type": "function",
                        "function": {"name": "research.lookup",
                                     "arguments": "this is not json"},
                    }],
                }
            }]
        }
        handler = MagicMock()
        with patch.object(orchestrator, "_chat",
                          MagicMock(side_effect=[bad_call, _fake_answer_response("Here it is.")])), \
             patch.dict(orchestrator._TOOL_HANDLERS,
                        {"research.lookup": handler}, clear=False):
            outcome = orchestrator.run_orchestrator("look up something", env)
        handler.assert_not_called()
        self.assertEqual(outcome["status"], "answered")
        self.assertTrue(any("tool rejected" in e for e in outcome["evidence"]))
        self.assertEqual(outcome["reply"], "Here it is.")

    def test_task_propose_never_executes_only_awaits_confirmation(self):
        env = envelope_mod.ContextEnvelope(utterance="create a folder named demo")
        plan = {"ok": True, "summary": "Create the demo folder?",
                "requires_confirmation": True,
                "steps": [{"tool": "code.create_folder",
                           "args": {"path": "C:/x/demo"}, "risk": "safe", "reason": "r"}]}
        with patch.object(orchestrator, "_chat",
                          MagicMock(side_effect=[_fake_tool_response(
                              "task.propose", {"description": "create a folder named demo"})])) as chat, \
             patch.object(task_agent, "plan_task", return_value=plan):
            outcome = orchestrator.run_orchestrator("create a folder named demo", env)
            # Exactly ONE model turn — the proposal is returned before any
            # turn that could add authority to the mutation.
            chat.assert_called_once()
        self.assertEqual(outcome["status"], "proposal")
        self.assertIn("demo folder", outcome["reply"])
        self.assertEqual(outcome["plan"]["summary"], "Create the demo folder?")

    def test_screen_question_net_fires_deterministically_before_model_turn(self):
        env = envelope_mod.ContextEnvelope(utterance="whats on my screen",
                                           screen_question=True)
        observe = MagicMock(return_value="A code editor is open.")
        with patch.object(orchestrator, "_chat",
                          MagicMock(side_effect=[_fake_answer_response("VS Code is open.")])), \
             patch.object(orchestrator, "_tool_screen_observe", observe):
            outcome = orchestrator.run_orchestrator("whats on my screen", env)
        # The screen observation ran (same analyze_screen capability) — the
        # LLM never decided whether to look.
        observe.assert_called_once()
        self.assertEqual(outcome["status"], "answered")

    def test_planner_failure_with_no_actions_declines(self):
        env = envelope_mod.ContextEnvelope(utterance="something")
        with patch.object(orchestrator, "_chat", return_value={}):
            outcome = orchestrator.run_orchestrator("something", env)
        self.assertIsNone(outcome)

    def test_unknown_tool_call_rejected(self):
        args, error = orchestrator.validate_tool_call("bogus.tool", "{}")
        self.assertIsNone(args)
        self.assertIn("unknown tool", error)


if __name__ == "__main__":
    unittest.main()
