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
        # Live follow-up phrasing: filler + adjective + located, nameless
        # write — once fell through to chat, which promised work no tool did.
        self.assertTrue(agent.is_code_tool_request(
            "now create a text file inside that folder and inside that "
            "text file just write hello"))
        self.assertTrue(agent.is_code_tool_request(
            "create a text file inside that folder and inside that text "
            "file just write hello"))
        self.assertTrue(agent.is_code_tool_request(
            "please create a text file in mayankmalik and write hello"))
        self.assertTrue(agent.is_code_tool_request(
            "create a text file in that folder with hello"))
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
        # Content phrases are not locations ("in english" is what to write).
        self.assertFalse(agent.is_code_tool_request(
            "write hello in english"))
        # web/browser targets must NOT be caught by the code-tool router
        self.assertFalse(agent.is_code_tool_request("open http://example.com"))
        self.assertFalse(agent.is_code_tool_request("open youtube in chrome"))
        self.assertFalse(agent.is_code_tool_request("search google for python"))
        # conversation / unknown verbs must not be caught either
        self.assertFalse(agent.is_code_tool_request("what is the weather"))


class R5LocalInspectTests(unittest.TestCase):
    """R5: local folder inspection is a LOCAL read, never a browser job.

    "Check whether folder Malik exists", "have a quick look at that
    folder", "search that folder and see what is inside" must route into
    the native code path (code.list_directory) — "look"/"navigate" never
    imply the browser; only a web URL / web target does.
    """

    def test_inspect_routes_to_code_tools(self):
        for text in (
            "check whether folder Malik exists",
            "have a quick look at that folder",
            "search that folder and see what is inside",
            "is there a folder named Malik",
            "list that folder",
            "tell me what is in there",
            "see what is inside mayankmalik",
        ):
            self.assertTrue(agent.is_code_tool_request(text), text)

    def test_web_targets_stay_out(self):
        self.assertFalse(agent.is_code_tool_request("search google for python"))
        self.assertFalse(agent.is_code_tool_request("open http://example.com"))
        self.assertFalse(agent.is_code_tool_request("navigate to example.com"))

    def test_create_shaped_turns_not_inspect(self):
        self.assertFalse(agent.is_code_tool_request("create a directory listing"))

    def test_heuristic_plans_local_list_not_browser(self):
        plan = agent._heuristic_plan(
            "have a quick look at that folder", {"windows": {}})
        if plan is not None:
            tools = [s.get("tool") for s in plan.get("steps", [])]
            self.assertNotIn("browser.open_url", tools)
            self.assertNotIn("browser.search_web", tools)


class R4ConfirmationVerdictTests(unittest.TestCase):
    """R4: the WHOLE confirmation sentence decides, including the tail."""

    def test_clean_yes(self):
        for text in ("yes", "yes please", "ok", "yes, please do it",
                     "haan kar do"):
            self.assertEqual(agent.classify_confirmation(text), "yes", text)

    def test_decline(self):
        for text in ("no", "no thanks", "yes, don't create it",
                     "don't create it"):
            self.assertEqual(agent.classify_confirmation(text), "no", text)

    def test_rename_needs_new_preview(self):
        verdict = agent.classify_confirmation("yes, but call it another name")
        self.assertTrue(verdict.startswith("rename:"), verdict)
        verdict = agent.classify_confirmation("yes, but name it demo please")
        self.assertEqual(verdict, "rename:demo", verdict)

    def test_inspect_tail_holds_write(self):
        self.assertEqual(
            agent.classify_confirmation("yes, a quick look"), "inspect")

    def test_extra_action_flagged(self):
        self.assertEqual(
            agent.classify_confirmation("yes, and also delete the old one"),
            "extra")

    def test_question_is_not_yes(self):
        self.assertEqual(
            agent.classify_confirmation("did I say yes?"), "unclear")
        # "Not yet" declines the CURRENT preview (existing NO-first rule);
        # it is never assent.
        self.assertEqual(agent.classify_confirmation("not yet"), "no")


class R9SttSafeConfirmationTests(unittest.TestCase):
    """R9: Whisper tails and time-splits never smuggle a write."""

    def test_dropped_leading_yes_still_inspect(self):
        self.assertEqual(
            agent.classify_confirmation("s, a quick look"), "inspect")
        self.assertEqual(
            agent.classify_confirmation("yes, a quick look"), "inspect")

    def test_time_split_is_decline(self):
        self.assertEqual(
            agent.classify_confirmation("yes yesterday, no now"), "no")
        self.assertEqual(
            agent.classify_confirmation("yes, but not now"), "no")

    def test_echo_question_is_not_assent(self):
        self.assertEqual(
            agent.classify_confirmation("did you say yes?"), "unclear")
        self.assertEqual(
            agent.classify_confirmation("did I say ok?"), "unclear")

    def test_empty_preview_says_empty_with_full_path(self):
        plan = agent._code_tool_plan(
            "code.write_file",
            {"path": "C:\\d\\New Zealand.txt", "content": ""},
            "Writing the file.")
        preview = agent._confirmation_preview(plan)
        self.assertIn("New Zealand.txt", preview)
        self.assertIn("(empty)", preview)

    def test_original_text_never_rewritten(self):
        raw = "yes yesterday, no now"
        self.assertEqual(agent.classify_confirmation(raw), "no")
        self.assertEqual(raw, "yes yesterday, no now")


class R15CarefulNameMatchingTests(unittest.TestCase):
    """R15: lookup ignores case, preview shows canon, writes recheck."""

    def test_lowercase_candidates_canonical(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "Mayank Malik"))
            kind, value = agent.resolve_folder_name(
                "malik", roots=[root])
            self.assertEqual(kind, "candidates")
            self.assertIn("Mayank Malik", value[0])

    def test_exact_case_insensitive(self):
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "Mayank Malik"))
            kind, value = agent.resolve_folder_name(
                "mayank malik", roots=[root])
            self.assertEqual(kind, "exact")
            self.assertTrue(value.endswith("Mayank Malik"))

    def test_no_match_is_none(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(
                agent.resolve_folder_name("zznope", roots=[root]),
                ("none", ""))

    def test_cued_repair_never_touches_filenames(self):
        kind, value = agent.resolve_folder_name("cued task", roots=[])
        self.assertEqual((kind, value), ("none", ""))

    def test_precheck_stops_when_file_appeared(self):
        with tempfile.TemporaryDirectory() as root:
            target = os.path.join(root, "New Zealand.txt")
            ok, _ = agent.pre_execution_recheck(target, "create")
            self.assertTrue(ok)
            with open(target, "w") as fh:
                fh.write("someone else")
            ok2, why2 = agent.pre_execution_recheck(target, "create")
            self.assertFalse(ok2)
            self.assertIn("appeared", why2)

    def test_exact_vs_candidate_answer(self):
        from backend.core import brain as _brain
        reply = _brain.answer_exact_vs_candidate(
            "Malik", ("candidates", ["C:\\d\\Mayank Malik"]))
        self.assertIn("no exact folder named malik", reply.lower())
        self.assertIn("Mayank Malik", reply)


class R1MeaningNotFirstWordTests(unittest.TestCase):
    """R1: the WANT anywhere in the sentence is the request."""

    def test_declarative_want_routes(self):
        self.assertTrue(agent.is_code_tool_request(
            "on my desktop there is a folder Mayank Malik, "
            "I want a text file inside it"))
        self.assertTrue(agent.is_code_tool_request(
            "there should be a new file in that folder"))
        self.assertTrue(agent.is_code_tool_request(
            "I need a file inside that folder"))

    def test_bare_existence_stays_chat(self):
        self.assertFalse(agent.is_code_tool_request("there is a file"))
        self.assertFalse(
            agent.is_code_tool_request("there is a file on the desktop"))

    def test_declarative_plans_located_write(self):
        plan = agent._heuristic_plan(
            "on my desktop there is a folder, I want a text file inside it",
            {"windows": {}})
        # Unresolvable pronoun -> asks which folder, never a browser job.
        if plan is not None:
            tools = [s.get("tool") for s in plan.get("steps", [])]
            self.assertNotIn("browser.open_url", tools)
            self.assertNotIn("browser.search_web", tools)


class R3AllSlotsBeforeActTests(unittest.TestCase):
    """R3: exact path + exact name + no-overwrite bound before acting."""

    def setUp(self):
        self.context = {
            "windows": {"active_window": {"title": "x"}, "visible_controls": []},
            "editor": {"available": False},
            "browser": {"available": False, "tabs": []},
        }
        agent._pending_task_action = None
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._saved_result = agent.last_task_result()
        self.addCleanup(agent._remember_task_result, *self._saved_result)

    def tearDown(self):
        agent._pending_task_action = None

    def _plan_in_folder(self, text, folder):
        from backend.core import brain as _brain
        _brain.notebook_record_entity(
            os.path.basename(folder), folder, kind="folder")
        self.addCleanup(_brain._notebook_entities.clear)
        return agent._heuristic_plan(text, self.context)

    def test_explicit_name_lands_in_plan(self):
        folder = os.path.join(self._tmp.name, "mayankmalik")
        os.makedirs(folder)
        plan = self._plan_in_folder(
            "now create a text file inside that folder, name it New Zealand, "
            "write hello inside it", folder)
        self.assertEqual(plan["steps"][0]["tool"], "code.write_file")
        self.assertEqual(
            plan["steps"][0]["args"]["path"],
            os.path.join(folder, "New Zealand.txt"))

    def test_name_it_anything_uses_default(self):
        folder = os.path.join(self._tmp.name, "mayankmalik")
        os.makedirs(folder)
        plan = self._plan_in_folder(
            "create a text file inside that folder, name it anything, "
            "write hello inside it", folder)
        self.assertEqual(
            plan["steps"][0]["args"]["path"],
            os.path.join(folder, agent._DEFAULT_TEXT_FILE_NAME))

    def test_missing_content_asks_one_question(self):
        folder = os.path.join(self._tmp.name, "mayankmalik")
        os.makedirs(folder)
        plan = self._plan_in_folder(
            "create a text file inside that folder, name it q-test", folder)
        self.assertEqual(plan["steps"], [])
        self.assertIn("what to write", plan["response"])

    def test_create_binds_no_overwrite(self):
        folder = os.path.join(self._tmp.name, "mayankmalik")
        os.makedirs(folder)
        plan = self._plan_in_folder(
            "create a text file inside that folder, name it q-test, "
            "write hello inside it", folder)
        self.assertTrue(plan["steps"][0]["args"].get("create_only"))

    def test_preview_speaks_exact_target(self):
        preview = agent._confirmation_preview({
            "steps": [{"tool": "code.write_file",
                       "args": {"path": "C:\\d\\New Zealand.txt",
                                "content": "hello", "create_only": True},
                       "risk": "safe"}]})
        self.assertIn("New Zealand.txt", preview)


class LocatedWriteTests(unittest.TestCase):
    """Live bug: "now create a text file inside that folder ... write hello"
    fell through every route into chat, which promised the file while no tool
    ran. These pin the routing, the cross-turn folder resolution, the default
    name, and the end-to-end write."""

    def setUp(self):
        self.context = {
            "windows": {"active_window": {"title": "x"}, "visible_controls": []},
            "editor": {"available": False},
            "browser": {"available": False, "tabs": []},
        }
        agent._pending_task_action = None
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._saved_result = agent.last_task_result()
        self.addCleanup(agent._remember_task_result, *self._saved_result)

    def tearDown(self):
        agent._pending_task_action = None

    def _remember_folder(self, folder):
        from backend.services.task_result import TaskResult
        agent._remember_task_result(
            TaskResult.completed("done",
                                 artifacts=[{"step": 0, "path": folder}]),
            "create folder")

    def test_that_folder_resolves_to_last_run_folder(self):
        folder = os.path.join(self._tmp.name, "mayankmalik")
        os.makedirs(folder)
        self._remember_folder(folder)
        plan = agent.plan_task(
            "now create a text file inside that folder and inside that "
            "text file just write hello",
            self.context,
        )
        self.assertTrue(plan["requires_confirmation"])
        self.assertEqual(plan["steps"][0]["tool"], "code.write_file")
        self.assertEqual(plan["steps"][0]["args"]["path"],
                         os.path.join(folder, "hello.txt"))
        self.assertEqual(plan["steps"][0]["args"]["content"], "hello")

    def test_bare_folder_name_resolves_under_desktop(self):
        plan = agent.plan_task(
            "create a text file in mayankmalik and write hello", self.context)
        self.assertEqual(plan["steps"][0]["tool"], "code.write_file")
        self.assertTrue(plan["steps"][0]["args"]["path"].endswith(
            os.path.join("mayankmalik", "hello.txt")))
        self.assertEqual(plan["steps"][0]["args"]["content"], "hello")

    def test_unresolvable_pronoun_asks_instead_of_guessing(self):
        agent._remember_task_result(None)
        plan = agent.plan_task(
            "create a text file inside that folder and write hello",
            self.context,
        )
        self.assertEqual(plan["steps"], [])
        self.assertIn("Which folder", plan["response"])

    def test_second_failed_ask_stops_instead_of_looping(self):
        cmd = "create a file in somewherexyz and write hello"
        first = agent.plan_task(cmd, self.context)
        self.assertIn("Which folder", first["response"])
        second = agent.plan_task(cmd, self.context)
        self.assertIn("Which folder", second["response"])
        third = agent.plan_task(cmd, self.context)
        self.assertIn("stopping rather than guessing", third["response"])
        self.assertEqual(third["steps"], [])
        # A fresh request starts fresh — the counter does not leak.
        agent._clarify_reset(cmd)
        fresh = agent.plan_task(cmd, self.context)
        self.assertIn("Which folder", fresh["response"])
        agent._clarify_reset(cmd)

    def test_second_missing_slot_ask_stops(self):
        cmd = "create a file in that folder"
        agent._remember_task_result(None)
        first = agent.plan_task(cmd, self.context)
        self.assertIn("Which folder", first["response"])
        agent._clarify_reset(cmd)

    def test_unclear_answer_twice_stops(self):
        plan = agent.plan_task("run pip list", self.context)
        agent.execute_plan(plan, self.context)
        command = plan.get("command_text") or ""
        self.assertIsNone(agent.consume_task_confirmation("maybe later"))
        agent.execute_plan(plan, self.context)
        self.assertIsNone(agent.consume_task_confirmation("hmm what"))
        agent.execute_plan(plan, self.context)
        stopped = agent.consume_task_confirmation("er hello")
        self.assertIn("stopping rather than guessing", stopped)
        self.assertIsNone(agent._pending_task_action)
        agent._clarify_reset(command)

    def test_inspect_then_create_runs_same_previewed_effect(self):
        folder = os.path.join(self._tmp.name, "mayankmalik")
        os.makedirs(folder)
        self._remember_folder(folder)
        plan = agent.plan_task(
            "now create a text file inside that folder and inside that "
            "text file just write hello",
            self.context,
        )
        agent.execute_plan(plan, self.context)
        held_q = agent.consume_task_confirmation("yes, a quick look")
        self.assertIn("Create the file, sir, or only check", held_q)
        creating = agent.consume_task_confirmation("create it")
        self.assertIn("exactly what was previewed", creating)
        agent.execute_plan(plan, self.context)
        done = agent.consume_task_confirmation("yes")
        target = os.path.join(folder, "hello.txt")
        self.assertTrue(os.path.exists(target))
        self.assertIn("hello.txt", done)

    def test_inspect_then_check_runs_read_only(self):
        folder = os.path.join(self._tmp.name, "mayankmalik")
        os.makedirs(folder)
        self._remember_folder(folder)
        plan = agent.plan_task(
            "now create a text file inside that folder and inside that "
            "text file just write hello",
            self.context,
        )
        agent.execute_plan(plan, self.context)
        agent.consume_task_confirmation("yes, a quick look")
        checked = agent.consume_task_confirmation("just check it")
        self.assertIn("checked only", checked)
        self.assertFalse(os.path.exists(os.path.join(folder, "hello.txt")))

    def test_replayed_yes_without_approval_reasks(self):
        plan = agent.plan_task("run pip list", self.context)
        agent.execute_plan(plan, self.context)  # arms the shared record
        agent._pending_task_action = None  # ...then the gate state is lost
        with patch.object(agent.code_tools, "call_tool") as fake:
            response = agent.execute_plan(
                plan, self.context, task_text="run pip list",
                confirmed=True)
        fake.assert_not_called()
        self.assertIn("confirm task", str(response))

    def test_refusals_stop_honestly(self):
        for cmd, needle in (
            ("navigate there", "without a target"),
            ("undo the file I created, delete it back",
             "cannot undo"),
        ):
            plan = agent.plan_task(cmd, self.context)
            self.assertEqual(plan["steps"], [])
            self.assertIn(needle, plan["response"])
        refuse, line = agent.should_refuse("x", resolved_kind="candidates")
        self.assertTrue(refuse)
        self.assertIn("more than one folder", line)
        refuse, line = agent.should_refuse("x", resolved_kind="none")
        self.assertTrue(refuse)
        self.assertIn("not guessing", line)
        refuse, _line = agent.should_refuse("create hello.txt with hi")
        self.assertFalse(refuse)

    def test_confirm_executes_the_write_end_to_end(self):
        folder = os.path.join(self._tmp.name, "mayankmalik")
        os.makedirs(folder)
        self._remember_folder(folder)
        plan = agent.plan_task(
            "now create a text file inside that folder and inside that "
            "text file just write hello",
            self.context,
        )
        preview = str(agent.execute_plan(plan, self.context))
        self.assertIn("hello.txt", preview)
        self.assertIn("hello", preview)
        response = str(agent.consume_task_confirmation("confirm task"))
        target = os.path.join(folder, "hello.txt")
        self.assertTrue(os.path.exists(target))
        with open(target, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "hello")
        self.assertIn("hello.txt", response)


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


class R20IntentCreateAndCheckTests(unittest.TestCase):
    """R20: the requested effect decides the route, not the sentence opening.

    Live transcript: "jarvis, can you create a folder on desktop by the name
    history and inside that folder can you create a txt file ... write
    hello" never reached the task path (every create matcher was anchored to
    a leading verb), so chat looped; and "check if there is a folder by the
    name Mayank Malik" lost the name and was answered with "Which folder
    should I create that file in".
    """

    COMPOUND = (
        "jarvis, can you create a folder on desktop by the name history and "
        "inside that folder can you create a txt file and inside that file "
        "just write hello from jarvis.")
    NAMED_CHECK = ("check if there is a folder by the name mayank malik on "
                   "my desktop.")

    def setUp(self):
        agent._clarify_attempts.clear()
        agent._pending_task_action = None
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.desktop = os.path.join(self._tmp.name, "Desktop")
        os.makedirs(self.desktop)
        self.folders = {
            "desktop": self.desktop,
            "documents": os.path.join(self._tmp.name, "Documents"),
            "downloads": os.path.join(self._tmp.name, "Downloads"),
            "home": self._tmp.name,
        }

    def test_question_wrapped_compound_create_routes(self):
        self.assertTrue(agent.is_code_tool_request(self.COMPOUND))

    def test_compound_create_plans_folder_then_file(self):
        with patch.object(agent, "_known_folders",
                          return_value=self.folders):
            plan = agent._heuristic_plan(self.COMPOUND, {})
        self.assertTrue(plan["requires_confirmation"])
        self.assertEqual([s["tool"] for s in plan["steps"]],
                         ["code.create_folder", "code.write_file"])
        self.assertEqual(plan["steps"][0]["args"]["path"],
                         os.path.join(self.desktop, "history"))
        write = plan["steps"][1]["args"]
        self.assertEqual(write["path"],
                         os.path.join(self.desktop, "history", "hello.txt"))
        self.assertIn("hello from jarvis", write["content"])
        self.assertTrue(write["create_only"])

    def test_compound_relation_after_the_file_clause(self):
        text = ("can you make a folder called demo and create a file "
                "inside it write hello")
        with patch.object(agent, "_known_folders",
                          return_value=self.folders):
            plan = agent._folder_file_create_plan(text)
        self.assertEqual([s["tool"] for s in plan["steps"]],
                         ["code.create_folder", "code.write_file"])
        self.assertEqual(plan["steps"][1]["args"]["path"],
                         os.path.join(self.desktop, "demo", "hello.txt"))
        self.assertEqual(plan["steps"][1]["args"]["content"], "hello")

    def test_capability_question_never_becomes_a_plan(self):
        with patch.object(agent, "_known_folders",
                          return_value=self.folders):
            self.assertIsNone(agent._folder_file_create_plan(
                "can you create folders on my desktop?"))

    def test_interrogative_simple_write_routes(self):
        self.assertTrue(agent.is_code_tool_request(
            "can you create a text file in mayankmalik and write hello"))

    def test_named_inspect_target_beats_the_folder_pronoun(self):
        self.assertEqual(
            agent._local_inspect_folder(agent._normalize(self.NAMED_CHECK)),
            ("named", "mayank malik"))
        self.assertEqual(
            agent._local_inspect_folder(agent._normalize(
                "is there a folder named Malik")),
            ("named", "malik"))
        self.assertEqual(
            agent._local_inspect_folder(agent._normalize(
                "have a quick look at that folder")),
            ("pronoun", ""))

    def test_named_check_that_resolves_is_a_local_read(self):
        target = os.path.join(self.desktop, "Mayank Malik")
        with patch.object(agent, "_resolve_folder_hint",
                          return_value=target):
            plan = agent._heuristic_plan(self.NAMED_CHECK, {})
        self.assertEqual(plan["steps"][0]["tool"], "code.list_directory")
        self.assertEqual(plan["steps"][0]["args"]["path"], target)

    def test_named_check_that_does_not_resolve_never_asks_to_create(self):
        with patch.object(agent, "_resolve_folder_hint", return_value=None), \
             patch.object(agent, "resolve_folder_name",
                          return_value=("none", "")):
            plan = agent._heuristic_plan(self.NAMED_CHECK, {})
        self.assertEqual(plan["steps"], [])
        self.assertIn("could not find a folder named", plan["response"])
        self.assertNotIn("create that file", plan["response"])

    def test_named_check_names_the_real_candidate(self):
        candidate = os.path.join(self.desktop, "Mayank Malik")
        with patch.object(agent, "_resolve_folder_hint", return_value=None), \
             patch.object(agent, "resolve_folder_name",
                          return_value=("candidates", [candidate])):
            plan = agent._heuristic_plan(
                "check if there is a folder named malik", {})
        self.assertIn("Mayank Malik", plan["response"])
        self.assertNotIn("create that file", plan["response"])

    def test_unresolved_check_asks_about_checking(self):
        with patch.object(agent, "_resolve_folder_hint", return_value=None):
            plan = agent._heuristic_plan(
                "have a quick look at that folder", {})
        self.assertIn("Which folder should I check", plan["response"])

    def test_screen_sentence_is_never_a_local_folder_check(self):
        # LIVE FIX 12, live log: "look at my screen there is a project
        # folder structure visible i want you to see it and replicate it
        # exactly on my desktop" asked "Which folder should I check, sir?"
        # — the inspect verb belongs to the screen, not to a local folder.
        text = ("look at my screen there is a project folder structure "
                "visible i want you to see it and replicate it exactly on "
                "my desktop")
        self.assertIsNone(agent._local_inspect_folder(agent._normalize(text)))
        self.assertFalse(agent.is_code_tool_request(text))

    def test_screen_sentence_planner_never_asks_which_folder(self):
        text = ("look at my screen there is a project folder structure "
                "visible i want you to see it and replicate it exactly on "
                "my desktop")
        plan = agent._heuristic_plan(text, {}) or {}
        self.assertNotIn("Which folder", plan.get("response") or "")

    def test_screen_typo_still_counts_as_the_screen(self):
        text = "look at my scrren and check the folder structure there"
        self.assertIsNone(agent._local_inspect_folder(agent._normalize(text)))


if __name__ == "__main__":
    unittest.main()
