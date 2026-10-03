import os
import tempfile
import time
import unittest
from unittest.mock import patch

from backend.services import code_tools
from backend.services.task_agent import agent


class CodeToolRegistryTests(unittest.TestCase):
    def test_registry_has_all_tools(self):
        self.assertIn("code.read_file", code_tools.TOOL_REGISTRY)
        self.assertIn("code.write_file", code_tools.TOOL_REGISTRY)
        self.assertIn("code.list_directory", code_tools.TOOL_REGISTRY)
        self.assertIn("code.create_folder", code_tools.TOOL_REGISTRY)
        self.assertIn("code.run_command", code_tools.TOOL_REGISTRY)
        self.assertIn("code.run_script", code_tools.TOOL_REGISTRY)

    def test_unknown_tool_returns_failure(self):
        result = code_tools.call_tool("does.not.exist", {})
        self.assertFalse(result["ok"])
        self.assertIn("Unknown tool", result["error"])

    def test_read_missing_file(self):
        result = code_tools.read_file("C:/definitely/not/here.txt")
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"].lower())

    def test_write_then_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "hello.txt")
            w = code_tools.write_file(path, "hello jarvis")
            self.assertTrue(w["ok"], w)
            r = code_tools.read_file(path)
            self.assertTrue(r["ok"])
            self.assertIn("hello jarvis", r["content"])

    def test_write_creates_missing_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a", "b", "c.txt")
            w = code_tools.write_file(path, "x")
            self.assertTrue(w["ok"], w)
            self.assertTrue(os.path.exists(path))

    def test_create_folder_makes_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deep", "newdir")
            r = code_tools.create_folder(path)
            self.assertTrue(r["ok"], r)
            self.assertTrue(os.path.isdir(path))

    def test_create_folder_exists_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(code_tools.create_folder(tmp)["ok"])
            again = code_tools.create_folder(os.path.join(tmp, "again"))
            self.assertTrue(again["ok"], again)
            self.assertTrue(os.path.isdir(os.path.join(tmp, "again")))

    def test_run_command_success(self):
        result = code_tools.run_command("echo hello-from-jarvis")
        self.assertTrue(result["ok"], result)
        self.assertIn("hello-from-jarvis", result["content"])

    def test_run_command_failure(self):
        result = code_tools.run_command("exit /b 3")
        self.assertFalse(result["ok"])
        self.assertEqual(result["exit_code"], 3)

    def test_run_script_inline_python(self):
        result = code_tools.run_script(code="print(6 * 7)")
        self.assertTrue(result["ok"], result)
        self.assertIn("42", result["content"])

    def test_script_tool_rejects_unknown_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.csv")
            with open(path, "w") as fh:
                fh.write("a,b")
            result = code_tools.run_script(path=path)
            self.assertFalse(result["ok"])
            self.assertIn("Unsupported script type", result["error"])

    def test_clip_limits_output(self):
        code_tools._MAX_OUTPUT = 50
        try:
            self.assertIn("truncated", code_tools._clip("x" * 500))
        finally:
            code_tools._MAX_OUTPUT = 8000


class CodeToolRoutingTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "windows": {"active_window": {"title": "x"}, "visible_controls": []},
            "editor": {"available": False},
            "browser": {"available": False, "tabs": []},
        }

    def test_read_request_routes_to_read_file(self):
        plan = agent.plan_task("read server.py", self.context)
        self.assertEqual(plan["steps"][0]["tool"], "code.read_file")
        self.assertEqual(plan["steps"][0]["args"]["path"], "server.py")
        self.assertFalse(plan["requires_confirmation"])

    def test_run_command_routes_to_run_command(self):
        plan = agent.plan_task("run pip list", self.context)
        self.assertEqual(plan["steps"][0]["tool"], "code.run_command")
        self.assertEqual(plan["steps"][0]["args"]["command"], "pip list")

    def test_run_script_by_extension(self):
        plan = agent.plan_task("run setup.py", self.context)
        self.assertEqual(plan["steps"][0]["tool"], "code.run_script")
        self.assertEqual(plan["steps"][0]["args"]["path"], "setup.py")

    def test_search_still_uses_browser(self):
        plan = agent.plan_task("search python decorators", self.context)
        self.assertEqual(plan["steps"][0]["tool"], "browser.search_web")

    def test_folder_creation_routes_to_create_folder_and_gates(self):
        plan = agent.plan_task("create a folder named badmoss", self.context)
        self.assertTrue(plan["requires_confirmation"])
        self.assertEqual(plan["steps"][0]["tool"], "code.create_folder")
        self.assertEqual(plan["steps"][0]["args"]["path"], "badmoss")

    def test_folder_on_desktop_routes_and_gates(self):
        plan = agent.plan_task(
            "create a folder on the desktop and name it demo", self.context
        )
        self.assertTrue(plan["requires_confirmation"])
        self.assertEqual(plan["steps"][0]["tool"], "code.create_folder")
        self.assertIn("Desktop", plan["steps"][0]["args"]["path"])
        self.assertTrue(plan["steps"][0]["args"]["path"].endswith("demo"))

    def test_folder_by_the_name_routes_to_desktop(self):
        # Live phrasing that once fell through to chat ("by the name X"):
        # routes into the gated task path AND resolves the desktop path.
        plan = agent.plan_task(
            "Create a folder on the desktop by the name Mayank Malik",
            self.context,
        )
        self.assertTrue(plan["requires_confirmation"])
        self.assertEqual(plan["steps"][0]["tool"], "code.create_folder")
        self.assertIn("Desktop", plan["steps"][0]["args"]["path"])
        self.assertTrue(
            plan["steps"][0]["args"]["path"].endswith("Mayank Malik"))

    def test_multi_file_plan_gates_with_write_steps(self):
        plan = agent.plan_task("create 4 files .txt .py .js .html", self.context)
        self.assertTrue(plan["requires_confirmation"])
        tools = [s["tool"] for s in plan["steps"]]
        self.assertEqual(tools, ["code.write_file"] * 4)
        paths = [s["args"]["path"] for s in plan["steps"]]
        self.assertEqual(paths, ["file1.txt", "file2.py", "file3.js", "file4.html"])

    def test_multi_file_named_plan_uses_given_names(self):
        plan = agent.plan_task(
            "create 4 files: a.txt, b.py, c.js, d.html", self.context
        )
        self.assertTrue(plan["requires_confirmation"])
        names = [s["args"]["path"] for s in plan["steps"]]
        self.assertEqual(names, ["a.txt", "b.py", "c.js", "d.html"])

    def test_multi_file_plan_pads_to_spoken_count(self):
        plan = agent.plan_task("create 3 files a.txt b.py", self.context)
        self.assertTrue(plan["requires_confirmation"])
        tools = [s["tool"] for s in plan["steps"]]
        self.assertEqual(tools, ["code.write_file"] * 3)
        paths = [s["args"]["path"] for s in plan["steps"]]
        self.assertEqual(paths, ["file1.txt", "file2.py", "file3.txt"])

    def test_folder_plus_multi_file_plan(self):
        plan = agent.plan_task(
            "create a folder named badmoss on the Desktop and create 4 files "
            "inside it - a .txt, a .py, a .js, and an .html",
            self.context,
        )
        self.assertTrue(plan["requires_confirmation"])
        tools = [s["tool"] for s in plan["steps"]]
        self.assertEqual(tools[0], "code.create_folder")
        self.assertEqual(len(tools), 5)
        for step in plan["steps"][1:]:
            self.assertEqual(step["tool"], "code.write_file")
            self.assertIn("badmoss", step["args"]["path"])

    def test_is_code_tool_request_routing(self):
        self.assertTrue(agent.is_code_tool_request("read PROJECT_MAP.md"))
        self.assertTrue(agent.is_code_tool_request("read src/main.py"))
        self.assertTrue(agent.is_code_tool_request("read http_server.py"))
        self.assertTrue(agent.is_code_tool_request("write file notes.txt"))
        self.assertTrue(agent.is_code_tool_request("write file notes with hello"))
        self.assertTrue(agent.is_code_tool_request("run pip list"))
        self.assertTrue(agent.is_code_tool_request("run setup.py"))
        self.assertTrue(agent.is_code_tool_request("run curl http://localhost/health"))
        # folder / multi-file creation must route into the gated task path
        self.assertTrue(agent.is_code_tool_request("create a folder named badmoss"))
        self.assertTrue(agent.is_code_tool_request("create a folder on the desktop and name it demo"))
        self.assertTrue(agent.is_code_tool_request("create a directory named logs"))
        self.assertTrue(agent.is_code_tool_request(
            "Create a folder on the desktop by the name Mayank Malik"))
        self.assertTrue(agent.is_code_tool_request("create 4 files .txt .py .js .html"))
        self.assertTrue(agent.is_code_tool_request("create 4 files: a.txt, b.py, c.js, d.html"))
        # conversational hijack phrases must NOT be caught
        self.assertFalse(agent.is_code_tool_request("read me a story"))
        self.assertFalse(agent.is_code_tool_request("write me a poem"))
        self.assertFalse(agent.is_code_tool_request("write a note about file systems"))
        self.assertFalse(agent.is_code_tool_request("update the config file"))
        self.assertFalse(agent.is_code_tool_request("create a folder"))
        self.assertFalse(agent.is_code_tool_request("create a directory listing"))
        self.assertFalse(agent.is_code_tool_request("create files"))
        self.assertFalse(agent.is_code_tool_request("run me a bath"))
        self.assertFalse(agent.is_code_tool_request("display my screen"))
        self.assertFalse(agent.is_code_tool_request("open file in chrome"))
        # web/browser targets must NOT be caught by the code-tool router
        self.assertFalse(agent.is_code_tool_request("open http://example.com"))
        self.assertFalse(agent.is_code_tool_request("open youtube in chrome"))
        self.assertFalse(agent.is_code_tool_request("search google for python"))
        # conversation / unknown verbs must not be caught either
        self.assertFalse(agent.is_code_tool_request("what is the weather"))


class CodeToolConfirmationGateTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            "windows": {"active_window": {"title": "x"}, "visible_controls": []},
            "editor": {"available": False},
            "browser": {"available": False, "tabs": []},
        }
        agent._pending_task_action = None

    def tearDown(self):
        agent._pending_task_action = None

    def test_sensitive_plans_require_confirmation(self):
        cases = [
            ("run pip list", "code.run_command"),
            ("write file x.txt with hello", "code.write_file"),
            ("run setup.py", "code.run_script"),
        ]
        for text, tool in cases:
            plan = agent.plan_task(text, self.context)
            self.assertTrue(plan["requires_confirmation"], text)
            self.assertEqual(plan["steps"][0]["tool"], tool)

    def test_read_plan_stays_confirmation_free(self):
        plan = agent.plan_task("read server.py", self.context)
        self.assertEqual(plan["steps"][0]["tool"], "code.read_file")
        self.assertFalse(plan["requires_confirmation"])

    def test_sensitive_plan_not_executed_without_confirmation(self):
        with patch.object(agent.code_tools, "call_tool") as fake:
            plan = agent.plan_task("run pip list", self.context)
            response = agent.execute_plan(plan, self.context)
            fake.assert_not_called()
        self.assertIn("pip list", response)
        self.assertIn("confirm task", response)
        self.assertIsNotNone(agent._pending_task_action)

    def test_write_preview_surfaces_path_and_content(self):
        plan = agent.plan_task("write file x.txt with hello", self.context)
        response = agent.execute_plan(plan, self.context)
        self.assertIn("x.txt", response)
        self.assertIn("hello", response)

    def test_consume_confirm_executes_stored_plan(self):
        plan = agent.plan_task("run pip list", self.context)
        agent.execute_plan(plan, self.context)
        ok = {"ok": True, "content": "ran pip list", "error": "",
              "exit_code": 0}
        with patch.object(agent.code_tools, "call_tool",
                          return_value=ok) as fake:
            response = agent.consume_task_confirmation("confirm task")
        fake.assert_called_once_with(
            "code.run_command", {"command": "pip list"},
            grants=code_tools.agent_grants())
        self.assertIn("ran pip list", response)
        self.assertIsNone(agent._pending_task_action)

    def test_multi_file_plan_arms_single_gate_and_executes_all_steps(self):
        plan = agent.plan_task("create 4 files .txt .py .js .html", self.context)
        response = agent.execute_plan(plan, self.context)
        self.assertIn("confirm task", response)
        self.assertIsNotNone(agent._pending_task_action)
        handle = tempfile.NamedTemporaryFile(delete=False)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        ok = {"ok": True, "content": "ok", "error": "",
              "path": handle.name, "exit_code": 0}
        with patch.object(agent.code_tools, "call_tool",
                          return_value=ok) as fake:
            response = agent.consume_task_confirmation("confirm task")
        self.assertEqual(fake.call_count, 4)
        self.assertIsNone(agent._pending_task_action)

    def test_consume_cancel_declines(self):
        plan = agent.plan_task("run pip list", self.context)
        agent.execute_plan(plan, self.context)
        with patch.object(agent, "_execute_step") as fake:
            response = agent.consume_task_confirmation("cancel")
        fake.assert_not_called()
        self.assertEqual(response, "As you wish, sir. I will skip that.")
        self.assertIsNone(agent._pending_task_action)

    def test_consume_off_topic_returns_none_and_clears(self):
        plan = agent.plan_task("run pip list", self.context)
        agent.execute_plan(plan, self.context)
        response = agent.consume_task_confirmation("what is the weather")
        self.assertIsNone(response)
        self.assertIsNone(agent._pending_task_action)

    def test_consume_with_no_pending_returns_none(self):
        self.assertIsNone(agent.consume_task_confirmation("confirm task"))

    def test_consume_expired_window_returns_none(self):
        plan = agent.plan_task("run pip list", self.context)
        agent.execute_plan(plan, self.context)
        agent._pending_task_action["expires"] = time.time() - 1
        self.assertIsNone(agent.consume_task_confirmation("confirm task"))
        self.assertIsNone(agent._pending_task_action)

    def test_consume_no_phrases_never_execute(self):
        plan = agent.plan_task("run pip list", self.context)
        phrases = (
            "ok but do not run it", "dont run it", "do not proceed", "stop",
            "not sure", "i am not sure about that", "not ready", "not yet",
        )
        for phrase in phrases:
            agent.execute_plan(plan, self.context)
            response = agent.consume_task_confirmation(phrase)
            self.assertEqual(
                response, "As you wish, sir. I will skip that.", phrase
            )
            self.assertIsNone(agent._pending_task_action)

    def test_has_pending_task_confirmation(self):
        self.assertFalse(agent.has_pending_task_confirmation())
        plan = agent.plan_task("run pip list", self.context)
        agent.execute_plan(plan, self.context)
        self.assertTrue(agent.has_pending_task_confirmation())

    def test_normalize_plan_forces_confirmation_for_sensitive_step(self):
        plan = {
            "ok": True,
            "confidence": 0.9,
            "summary": "Running the command.",
            "requires_confirmation": False,
            "steps": [
                {
                    "tool": "code.run_command",
                    "args": {"command": "pip list"},
                    "risk": "safe",
                    "reason": "Running the command.",
                }
            ],
        }
        normalized = agent._normalize_plan(plan, "run pip list")
        self.assertTrue(normalized["requires_confirmation"])

    def test_confirmed_edit_selection_calls_editor_bridge(self):
        # F15: the plan-time target is addressed by document uri + version +
        # explicit range; a text-only edit (no identity, no version) is
        # refused before the bridge is ever reached, so the call here must
        # carry the version/range the planner inspected.
        plan = {
            "ok": True,
            "summary": "Editing the selection.",
            "requires_confirmation": True,
            "steps": [
                {
                    "tool": "editor.edit_active_selection",
                    "args": {"text": "replacement text"},
                    "risk": "risky",
                    "reason": "Editing the selection.",
                }
            ],
        }
        selection = {"start": {"line": 1, "character": 0},
                     "end": {"line": 1, "character": 4}}
        context = {
            "windows": {"active_window": {"title": "x"}, "visible_controls": []},
            "editor": {"available": True,
                       "state": {"activeFile": {
                           "uri": "file:///c:/proj/a.py",
                           "path": "C:\\proj\\a.py",
                           "version": 7,
                           "selection": selection}}},
            "browser": {"available": False, "tabs": []},
        }
        with patch.object(agent.editor_bridge, "active_selection_matches",
                          return_value=True), \
             patch.object(
                 agent.editor_bridge,
                 "edit_active_selection",
                 return_value={"message": "Edited the selection."},
             ) as fake:
            response = agent.execute_plan(plan, context, confirmed=True)
        fake.assert_called_once()
        kwargs = fake.call_args.kwargs
        self.assertEqual(kwargs.get("uri"), "file:///c:/proj/a.py")
        self.assertEqual(kwargs.get("version"), 7)
        self.assertEqual(kwargs.get("selection"), selection)
        self.assertEqual(fake.call_args.args[0], "replacement text")
        self.assertIn("Edited the selection.", response)


if __name__ == "__main__":
    unittest.main()
