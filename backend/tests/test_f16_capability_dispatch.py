"""F16 — Dispatch by Capability, Not One Global Engine.

Pins the finding's Acceptance clauses against real code paths:

1. **Negated opencode requests never enable it.** A negated opt-in spelling
   must not select (or grant) the coding agent, while a positive one still
   does.
2. **Browser recovery needs no CLI.** Browser-shaped work and recovery after
   failed local execution never consult opencode availability.
3. **Local refactoring does not become browser work.** A local coding request
   is served by a coding executor or FAILS CLOSED — never delegated to the
   browser engine.
4. **Configuration changes after consent cannot change executor.** The
   decision is frozen into an immutable
   :class:`~backend.services.capability_contract.ExecutionContract`; the
   executor verifies it instead of re-reading ``config.TASK_ENGINE``, and a
   tampered/foreign/grantless contract is refused before any spawn.

Plus the fail-closed config read (``_opencode_engine_enabled``) that used to
authorize a spawn when the configuration raised.
"""

import sys
import unittest
from dataclasses import replace
from unittest.mock import patch

from backend import config
from backend.services import capability_contract as contracts
from backend.services import capability_resolver as resolver
from backend.services import opencode_client


class _FakeStream:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else ""


class _FakeProc:
    """Minimal Popen stand-in: no real process is ever spawned."""

    def __init__(self, lines=("line one\n", "line two\n"), returncode=0):
        self.stdout = _FakeStream(list(lines))
        self.returncode = returncode
        self.pid = 4242
        self.terminated = False

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


def _run_with_contract(contract, task="take over and fix the build"):
    """Run run_opencode_task with every process/network boundary mocked."""
    proc = _FakeProc()
    with patch.object(opencode_client, "_resolve_opencode",
                      return_value=r"C:\fake\opencode.exe"), \
         patch.object(opencode_client, "is_opencode_server_alive",
                      return_value=False), \
         patch.object(opencode_client, "_build_command",
                      return_value=["opencode", "run"]), \
         patch.object(opencode_client, "_truncate_activity_log"), \
         patch.object(opencode_client, "_append_activity"), \
         patch.object(opencode_client, "_narrate_line"), \
         patch.object(opencode_client, "_reset_narration_state"), \
         patch.object(opencode_client.subprocess, "Popen",
                      return_value=proc) as popen:
        result = opencode_client.run_opencode_task(task, timeout=5,
                                                   contract=contract)
    return result, popen, proc


class NegatedOptInTests(unittest.TestCase):
    """Acceptance 1 — negated opencode requests never enable it."""

    NEGATED = (
        "don't use opencode, just refactor the local function in jobs.py",
        "do not take over with the coding agent, fix the function instead",
        "no opencode, refactor this module locally",
        "without opencode please fix the build error",
        "skip agent mode and rename the method",
    )

    def test_negated_opt_in_never_selects_opencode(self):
        for command in self.NEGATED:
            decision = resolver.resolve_engine(
                command, availability={"opencode": True})
            self.assertNotEqual(decision["engine"], resolver.OPENCODE, command)
            self.assertFalse(decision["opt_in"], command)

    def test_negated_opt_in_never_blocks_a_local_task_on_the_cli(self):
        # Negated + unavailable CLI must behave like an ordinary local task,
        # not like a refused coding-agent request.
        decision = resolver.resolve_engine(
            "don't use opencode, refactor the function in jobs.py",
            availability={"opencode": False, "editor": False})
        self.assertEqual(decision["engine"], resolver.CODE_TOOLS)
        self.assertFalse(decision["opt_in"])
        self.assertFalse(decision["blocked"])

    def test_positive_opt_in_still_selects_opencode(self):
        decision = resolver.resolve_engine(
            "use opencode to fix the build error",
            availability={"opencode": True})
        self.assertEqual(decision["engine"], resolver.OPENCODE)
        self.assertTrue(decision["opt_in"])


class BrowserRecoveryTests(unittest.TestCase):
    """Acceptance 2 — browser recovery needs no CLI."""

    def test_browser_work_ignores_opencode_availability(self):
        decision = resolver.resolve_engine(
            "go to github.com and open the issues tab",
            availability={"opencode": False})
        self.assertEqual(decision["engine"], resolver.BROWSER_AGENT)
        self.assertEqual(decision["capability"], resolver.CAPABILITY_BROWSER)

    def test_recovery_never_requires_the_cli(self):
        decision = resolver.recovery_after_failure(
            "open youtube and play lofi", availability={"opencode": False})
        self.assertEqual(decision["engine"], resolver.BROWSER_AGENT)
        self.assertIn("recovery", decision["reason"])
        self.assertFalse(decision["blocked"])

    def test_web_shaped_task_never_claims_coding_capability(self):
        decision = resolver.resolve_engine(
            "search the web for a local refactor guide",
            availability={"opencode": False})
        self.assertEqual(decision["capability"], resolver.CAPABILITY_BROWSER)


class LocalRefactorTests(unittest.TestCase):
    """Acceptance 3 — local refactoring does not become browser work."""

    COMMAND = "refactor the local function in backend/services/jobs.py"

    def test_local_refactor_never_becomes_browser_work(self):
        decision = resolver.resolve_engine(
            self.COMMAND, availability={"opencode": False, "editor": False},
            task_engine="browser_agent")
        self.assertEqual(decision["engine"], resolver.CODE_TOOLS)
        self.assertEqual(decision["capability"], resolver.CAPABILITY_CODING)
        self.assertNotEqual(decision["engine"], resolver.BROWSER_AGENT)
        self.assertTrue(decision["compatible"])

    def test_editor_bridge_serves_local_coding_when_connected(self):
        decision = resolver.resolve_engine(
            self.COMMAND, availability={"opencode": False, "editor": True})
        self.assertEqual(decision["engine"], resolver.EDITOR_TOOLS)

    def test_local_refactor_fails_closed_without_a_coding_executor(self):
        decision = resolver.resolve_engine(
            self.COMMAND,
            availability={"opencode": False, "editor": False,
                          "code_tools": False},
            task_engine="browser_agent")
        self.assertEqual(decision["engine"], resolver.BLOCKED)
        self.assertFalse(decision["compatible"])
        self.assertTrue(decision["blocked"])
        self.assertNotEqual(decision["engine"], resolver.BROWSER_AGENT)

    def test_explicit_coding_agent_request_without_cli_fails_closed(self):
        # "take over and fix the build" is a coding request; a browser engine
        # is NOT a compatible fallback, so this refuses instead of silently
        # becoming browser work.
        decision = resolver.resolve_engine(
            "take over and fix the build", availability={"opencode": False})
        self.assertEqual(decision["engine"], resolver.BLOCKED)
        self.assertTrue(decision["opt_in"])
        self.assertFalse(decision["compatible"])

    def test_recovery_does_not_substitute_an_incompatible_engine(self):
        decision = resolver.recovery_after_failure(
            "refactor the local function in backend/services/jobs.py",
            availability={"opencode": False, "editor": False,
                          "code_tools": False})
        self.assertEqual(decision["engine"], resolver.BLOCKED)


class ContractImmutabilityTests(unittest.TestCase):
    """Acceptance 4 — a decision travels as an immutable contract."""

    def setUp(self):
        contracts.clear()

    def test_freeze_carries_capability_executor_availability_and_grant(self):
        contract = resolver.begin_dispatch(
            "use opencode to fix the build error",
            availability={"opencode": True, "editor": False},
            task_engine="opencode", grant="consent-42")
        self.assertEqual(contract.executor, resolver.OPENCODE)
        self.assertEqual(contract.capability, resolver.CAPABILITY_CODING)
        self.assertEqual(dict(contract.availability),
                         {"opencode": True, "editor": False,
                          "code_tools": True})
        self.assertEqual(contract.grant, "consent-42")
        self.assertTrue(contract.opt_in)
        self.assertTrue(contract.verify(executor=resolver.OPENCODE))
        self.assertEqual(contracts.latest().digest, contract.digest)

    def test_tampered_executor_is_detected(self):
        contract = resolver.begin_dispatch(
            "use opencode to fix the build error",
            availability={"opencode": True}, grant="consent-42")
        tampered = replace(contract, executor=resolver.BROWSER_AGENT)
        with self.assertRaises(contracts.ContractViolation):
            tampered.verify()
        self.assertFalse(tampered.authorizes(executor=resolver.BROWSER_AGENT))

    def test_foreign_executor_is_refused(self):
        contract = resolver.begin_dispatch(
            "use opencode to fix the build error",
            availability={"opencode": True}, grant="consent-42")
        with self.assertRaises(contracts.ContractViolation):
            contract.verify(executor=resolver.BROWSER_AGENT)

    def test_grantless_contract_authorizes_nothing(self):
        contract = resolver.begin_dispatch(
            "use opencode to fix the build error",
            availability={"opencode": True})
        self.assertEqual(contract.grant, "")
        with self.assertRaises(contracts.ContractViolation):
            contract.verify(executor=resolver.OPENCODE)
        # Attaching the grant keeps the contract immutable but valid.
        granted = contract.granted("consent-1")
        self.assertEqual(contract.grant, "")
        self.assertTrue(granted.verify(executor=resolver.OPENCODE))

    def test_blocked_decision_freezes_as_blocked(self):
        contract = resolver.begin_dispatch(
            "refactor the local function in jobs.py",
            availability={"opencode": False, "editor": False,
                          "code_tools": False},
            grant="consent-1")
        self.assertEqual(contract.executor, contracts.BLOCKED)
        with self.assertRaises(contracts.ContractViolation):
            contract.verify()


class ExecutorHonoursConsentTests(unittest.TestCase):
    """Acceptance 4 — configuration changes cannot change the executor."""

    def setUp(self):
        contracts.clear()

    def test_consented_opencode_contract_runs_even_if_config_flips(self):
        contract = resolver.begin_dispatch(
            "use opencode to fix the build error",
            availability={"opencode": True}, task_engine="opencode",
            grant="consent-1")
        # The user already consented; configuration now says browser_agent.
        with patch.object(config, "TASK_ENGINE", "browser_agent"):
            self.assertFalse(opencode_client._opencode_engine_enabled())
            result, popen, _proc = _run_with_contract(contract)
        popen.assert_called_once()
        self.assertEqual(result.status, "completed")

    def test_contract_for_another_executor_is_refused_before_spawning(self):
        # task_engine is pinned so the decision cannot depend on whatever
        # TASK_ENGINE the surrounding suite happens to leave configured.
        contract = resolver.begin_dispatch(
            "open youtube and play", availability={"opencode": True},
            task_engine="browser_agent", grant="consent-1")
        result, popen, _proc = _run_with_contract(contract, task="open youtube")
        popen.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertIn("dispatch refused", result.error)

    def test_grantless_contract_is_refused_before_spawning(self):
        contract = resolver.begin_dispatch(
            "use opencode to fix the build error",
            availability={"opencode": True})
        result, popen, _proc = _run_with_contract(contract)
        popen.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertIn("grant", result.error)

    def test_tampered_contract_is_refused_before_spawning(self):
        contract = resolver.begin_dispatch(
            "use opencode to fix the build error",
            availability={"opencode": True}, grant="consent-1")
        tampered = replace(contract, capability=resolver.CAPABILITY_GENERIC)
        result, popen, _proc = _run_with_contract(tampered)
        popen.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertIn("digest", result.error)

    def test_ledger_records_every_dispatch_decision(self):
        first = resolver.begin_dispatch(
            "use opencode to fix the build error",
            availability={"opencode": True}, grant="consent-1")
        second = resolver.begin_dispatch(
            "refactor the local function in jobs.py",
            availability={"opencode": False, "editor": False})
        digests = [c.digest for c in contracts.ledger()]
        self.assertIn(first.digest, digests)
        self.assertIn(second.digest, digests)


class FailClosedConfigTests(unittest.TestCase):
    """The engine gate fails CLOSED when the configuration cannot be read."""

    def test_unreadable_configuration_never_authorizes_a_spawn(self):
        with patch.object(opencode_client, "_task_engine",
                          side_effect=RuntimeError("boom")):
            self.assertFalse(opencode_client._opencode_engine_enabled())
            result, popen, _proc = _run_with_contract(None)
        popen.assert_not_called()
        self.assertEqual(result.status, "failed")
        self.assertIn("opencode", result.error)

    def test_legacy_path_still_refused_when_engine_is_browser_agent(self):
        with patch.object(config, "TASK_ENGINE", "browser_agent"):
            self.assertFalse(opencode_client._opencode_engine_enabled())
            result, popen, _proc = _run_with_contract(None)
        popen.assert_not_called()
        self.assertFalse(result)

    def test_legacy_path_runs_when_configuration_selects_opencode(self):
        with patch.object(config, "TASK_ENGINE", "opencode"):
            self.assertTrue(opencode_client._opencode_engine_enabled())
            result, popen, _proc = _run_with_contract(None)
        popen.assert_called_once()
        self.assertEqual(result.status, "completed")


if __name__ == "__main__":
    unittest.main()
