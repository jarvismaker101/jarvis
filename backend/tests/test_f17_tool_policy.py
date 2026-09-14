"""F17 — enforce tool policy at dispatch.

Acceptance (audit report): "Computed calls, hidden bypasses, wrong-origin
effects, malformed nested arguments, forged tab text, and grantless direct
invocation fail before effects."
"""

import json
import os
import unittest
import unittest.mock
from unittest.mock import patch

from backend.services import browser_agent, tool_policy


class ComputedProbeCallTests(unittest.TestCase):
    """A member named through a string is still a call the policy must judge."""

    def test_bracketed_mutation_is_rejected(self):
        reason = tool_policy.probe_expression_error(
            "document.querySelector('button')['click']()")
        self.assertIsNotNone(reason)
        self.assertIn("computed member access", reason)

    def test_bracketed_mutation_built_from_concatenated_literals(self):
        reason = tool_policy.probe_expression_error(
            "document.querySelector('button')['cl'+'ick']()")
        self.assertIsNotNone(reason)

    def test_bracketed_submit_is_rejected(self):
        reason = tool_policy.probe_expression_error(
            "document.querySelector('form')['submit']()")
        self.assertIsNotNone(reason)

    def test_template_literal_member_is_rejected(self):
        reason = tool_policy.probe_expression_error(
            "document.querySelector('button')[`click`]()")
        self.assertIsNotNone(reason)

    def test_statement_then_mutation_is_rejected(self):
        reason = tool_policy.probe_expression_error(
            "document.querySelector('a'); window.open('http://x')")
        self.assertIsNotNone(reason)

    def test_fetch_is_rejected(self):
        reason = tool_policy.probe_expression_error(
            "document.querySelector('a').href + fetch('http://x')")
        self.assertIsNotNone(reason)

    def test_direct_click_call_is_rejected(self):
        reason = tool_policy.probe_expression_error(
            "document.querySelector('a').click()")
        self.assertIsNotNone(reason)

    def test_assignment_is_rejected(self):
        reason = tool_policy.probe_expression_error(
            "document.querySelector('a').value = 'x'")
        self.assertIsNotNone(reason)

    # ── the useful part must keep working ──
    def test_property_read_is_allowed(self):
        self.assertIsNone(tool_policy.probe_expression_error(
            "document.querySelector('#price').innerText"))

    def test_length_read_is_allowed(self):
        self.assertIsNone(tool_policy.probe_expression_error(
            "document.querySelectorAll('.row').length"))

    def test_allowed_member_named_through_a_string_is_allowed(self):
        self.assertIsNone(tool_policy.probe_expression_error(
            "document.querySelector('button')['innerText']"))

    def test_numeric_index_is_allowed(self):
        self.assertIsNone(tool_policy.probe_expression_error(
            "document.querySelectorAll('li')[0].textContent"))

    def test_pipeline_free_read_is_allowed(self):
        self.assertIsNone(tool_policy.probe_expression_error(
            "document.querySelectorAll('h2').length"))


class DispatchBoundaryTests(unittest.TestCase):
    """One typed effect boundary for every authority class."""

    def test_model_call_without_an_allowlist_fails_closed(self):
        decision = tool_policy.validate_dispatch(
            "navigate", {"url": "https://example.com"}, None)
        self.assertFalse(decision.allowed)
        self.assertIn("allowlist", decision.reason)

    def test_hidden_primitive_is_not_dispatchable_by_the_model(self):
        for name in ("read_file", "list_dir", "click_locator", "fill_locator"):
            self.assertNotIn(name, browser_agent._MODEL_DISPATCH_ALLOWLIST)
            decision = tool_policy.validate_dispatch(
                name, {"path": "x"}, browser_agent._MODEL_DISPATCH_ALLOWLIST)
            self.assertFalse(decision.allowed, name)

    def test_internal_origin_may_reach_a_primitive(self):
        decision = tool_policy.validate_dispatch(
            "read_file", {"path": "x"},
            browser_agent._TOOL_DISPATCH_ALLOWLIST, origin="internal")
        self.assertTrue(decision.allowed, decision.reason)

    def test_internal_origin_still_needs_the_privileged_grant(self):
        decision = tool_policy.validate_dispatch(
            "evaluate", {"expression": "1+1"},
            browser_agent._TOOL_DISPATCH_ALLOWLIST, grants=set(),
            origin="internal")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.operation, tool_policy.PRIVILEGED)

    def test_grantless_privileged_invocation_fails_before_effects(self):
        decision = tool_policy.validate_dispatch(
            "evaluate", {"expression": "1+1"},
            browser_agent._MODEL_DISPATCH_ALLOWLIST, grants=set())
        self.assertFalse(decision.allowed)
        self.assertIn("not granted", decision.reason)

    def test_granted_privileged_invocation_is_allowed(self):
        decision = tool_policy.validate_dispatch(
            "evaluate", {"expression": "1+1"},
            browser_agent._MODEL_DISPATCH_ALLOWLIST,
            grants={"privileged_js"})
        self.assertTrue(decision.allowed, decision.reason)

    def test_malformed_nested_arguments_are_rejected(self):
        schema = {
            "type": "object",
            "required": ["steps"],
            "properties": {
                "steps": {
                    "type": "array",
                },
            },
        }
        decision = tool_policy.validate_dispatch(
            "navigate", {"steps": "not-an-array"},
            browser_agent._MODEL_DISPATCH_ALLOWLIST, schema=schema)
        self.assertFalse(decision.allowed)
        self.assertIn("must be array", decision.reason)

    def test_missing_required_argument_is_rejected(self):
        schema = {"type": "object", "required": ["url"]}
        decision = tool_policy.validate_dispatch(
            "navigate", {}, browser_agent._MODEL_DISPATCH_ALLOWLIST,
            schema=schema)
        self.assertFalse(decision.allowed)
        self.assertIn("missing required argument", decision.reason)

    def test_batch_probe_rejects_a_computed_mutation(self):
        decision = tool_policy.validate_dispatch(
            "batch_probe",
            {"expressions": ["document.querySelector('b')['click']()"]},
            browser_agent._MODEL_DISPATCH_ALLOWLIST)
        self.assertFalse(decision.allowed)
        self.assertIn("batch_probe rejected", decision.reason)

    def test_batch_probe_rejects_non_list_expressions(self):
        decision = tool_policy.validate_dispatch(
            "batch_probe", {"expressions": "document.title"},
            browser_agent._MODEL_DISPATCH_ALLOWLIST)
        self.assertFalse(decision.allowed)
        self.assertIn("expressions array", decision.reason)

    def test_unknown_tool_fails_closed_as_mutating(self):
        decision = tool_policy.validate_dispatch(
            "delete_everything", {}, browser_agent._MODEL_DISPATCH_ALLOWLIST)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.operation, tool_policy.MUTATE)


class EmptyGrantTests(unittest.TestCase):
    """An explicitly empty grant set must not revive the environment default."""

    def test_explicitly_empty_grants_do_not_revive_the_env_default(self):
        session = {"grants": set()}
        with patch.dict(os.environ, {"JARVIS_BROWSER_PRIVILEGED_JS": "1"}):
            self.assertEqual(browser_agent._grants_for(session), set())

    def test_explicitly_empty_list_grants_do_not_revive_the_env_default(self):
        session = {"grants": []}
        with patch.dict(os.environ, {"JARVIS_BROWSER_PRIVILEGED_JS": "1"}):
            self.assertEqual(browser_agent._grants_for(session), set())

    def test_absent_grants_still_use_the_environment_default(self):
        with patch.dict(os.environ, {"JARVIS_BROWSER_PRIVILEGED_JS": "1"}):
            self.assertEqual(browser_agent._grants_for({}),
                             {"privileged_js"})

    def test_explicit_grants_are_honoured(self):
        session = {"grants": ["privileged_js"]}
        with patch.dict(os.environ, {"JARVIS_BROWSER_PRIVILEGED_JS": "0"}):
            self.assertEqual(browser_agent._grants_for(session),
                             {"privileged_js"})


class SecretScrubTests(unittest.TestCase):
    """F21 groundwork: nothing sensitive reaches a trace."""

    def test_sensitive_arguments_are_masked_in_the_decision(self):
        decision = tool_policy.validate_dispatch(
            "fill", {"selector": "#p", "password": "hunter2"},
            browser_agent._MODEL_DISPATCH_ALLOWLIST)
        self.assertIn("password", decision.masked)
        self.assertNotIn("hunter2", json.dumps(decision.arguments))

    def test_bearer_token_is_scrubbed_from_free_text(self):
        scrubbed = tool_policy.mask_secrets(
            "Authorization: Bearer abcdef1234567890xyz")
        self.assertNotIn("abcdef1234567890xyz", scrubbed)


class CodeToolBoundaryTests(unittest.TestCase):
    """The native/editor path must go through the same boundary."""

    def _spy(self, name, replacement):
        """Replace one entry of the tool registry (what dispatch reads)."""
        from backend.services import code_tools
        return patch.dict(code_tools.TOOL_REGISTRY,
                          {name: (replacement, "spy")})

    def test_grantless_command_execution_fails_before_effects(self):
        from backend.services import code_tools
        with self._spy("code.run_command", unittest.mock.Mock()) as registry:
            result = code_tools.call_tool(
                "code.run_command", {"command": "echo hi"})
            registry["code.run_command"][0].assert_not_called()
        self.assertFalse(result["ok"])
        self.assertIn("blocked by policy", result["error"])

    def test_grantless_write_fails_before_effects(self):
        from backend.services import code_tools
        with self._spy("code.write_file", unittest.mock.Mock()) as registry:
            result = code_tools.call_tool(
                "code.write_file", {"path": "x.txt", "content": "x"})
            registry["code.write_file"][0].assert_not_called()
        self.assertFalse(result["ok"])
        self.assertIn("blocked by policy", result["error"])

    def test_read_tool_needs_no_grant(self):
        from backend.services import code_tools
        seen = {}

        def fake_read_file(path):
            seen["path"] = path
            return {"ok": True, "content": "hi"}

        with self._spy("code.read_file", fake_read_file):
            result = code_tools.call_tool("code.read_file", {"path": "x.txt"})
        self.assertTrue(result["ok"], result)
        self.assertEqual(seen["path"], "x.txt")

    def test_agent_grants_allow_the_orchestrated_path(self):
        from backend.services import code_tools
        seen = {}

        def fake_run_command(command, cwd=None, timeout=None):
            seen["command"] = command
            return {"ok": True, "content": "hi"}

        with self._spy("code.run_command", fake_run_command):
            result = code_tools.call_tool(
                "code.run_command", {"command": "echo hi"},
                grants=code_tools.agent_grants())
        self.assertTrue(result["ok"], result)
        self.assertEqual(seen["command"], "echo hi")

    def test_malformed_argument_type_is_refused_before_effects(self):
        from backend.services import code_tools
        with self._spy("code.run_command", unittest.mock.Mock()) as registry:
            result = code_tools.call_tool(
                "code.run_command", {"command": ["not", "a", "string"]},
                grants=code_tools.agent_grants())
            registry["code.run_command"][0].assert_not_called()
        self.assertFalse(result["ok"])
        self.assertIn("must be string", result["error"])

    def test_missing_required_argument_is_refused_before_effects(self):
        from backend.services import code_tools
        with self._spy("code.write_file", unittest.mock.Mock()) as registry:
            result = code_tools.call_tool(
                "code.write_file", {"content": "x"},
                grants=code_tools.agent_grants())
            registry["code.write_file"][0].assert_not_called()
        self.assertFalse(result["ok"])
        self.assertIn("missing required argument", result["error"])

    def test_unknown_code_tool_still_reports_unknown(self):
        from backend.services import code_tools
        result = code_tools.call_tool("does.not.exist", {})
        self.assertFalse(result["ok"])
        self.assertIn("Unknown tool", result["error"])

    def test_code_operations_are_classified(self):
        self.assertEqual(
            tool_policy.classify_operation("code.read_file"), tool_policy.READ)
        self.assertEqual(
            tool_policy.classify_operation("code.write_file"),
            tool_policy.MUTATE)
        self.assertEqual(
            tool_policy.classify_operation("code.run_command"),
            tool_policy.PRIVILEGED)
        self.assertEqual(
            tool_policy.classify_operation("editor.read_buffer"),
            tool_policy.READ)
        self.assertEqual(
            tool_policy.classify_operation("editor.apply_workspace_edit"),
            tool_policy.MUTATE)

    def test_schemas_are_derived_from_the_real_signatures(self):
        from backend.services import code_tools
        schema = code_tools.TOOL_SCHEMAS["code.run_command"]
        self.assertIn("command", schema["required"])
        self.assertEqual(schema["properties"]["command"]["type"], "string")
        self.assertEqual(schema["properties"]["timeout"]["type"], "integer")
        self.assertEqual(
            code_tools.TOOL_SCHEMAS["code.apply_patch"]["properties"]
            ["dry_run"]["type"], "boolean")


if __name__ == "__main__":
    unittest.main()
