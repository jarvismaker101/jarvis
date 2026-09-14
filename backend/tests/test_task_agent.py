import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from backend.core import brain
from backend.services.task_agent import agent
from backend.services.task_result import TaskResult


class TaskAgentTests(unittest.TestCase):
    def test_explicit_task_prefix_routes_to_task_agent(self):
        self.assertTrue(agent.is_task_request("task search Python decorators"))
        self.assertTrue(agent.is_task_request("agent install extension in Antigravity"))

    def test_antigravity_action_routes_to_task_agent(self):
        self.assertTrue(agent.is_task_request("install the python extension in Antigravity"))

    def test_plain_question_does_not_route_to_task_agent(self):
        self.assertFalse(agent.is_task_request("what is Antigravity"))

    def test_simple_search_uses_browser_search_tool(self):
        context = {
            "windows": {"active_window": {"title": "Browser"}, "visible_controls": []},
            "editor": {"available": False, "active_window_looks_like_editor": False},
            "browser": {"available": False, "tabs": []},
        }
        plan = agent.plan_task("search python decorators", context)
        self.assertFalse(plan["requires_confirmation"])
        self.assertEqual(plan["steps"][0]["tool"], "browser.search_web")
        self.assertEqual(plan["steps"][0]["args"]["query"], "python decorators")

    def test_brain_routes_task_request_before_chat(self):
        with patch.object(brain, "handle_task_message", return_value="task handled") as task_handler:
            response = brain.process_message("task search python decorators", sync_voice=False)

        self.assertEqual(response, "task handled")
        task_handler.assert_called_once()

    def test_handle_task_message_voice_compact_cuts_at_sentence(self):
        long_response = ("foo " * 43) + ". " + ("bar " * 60)
        with patch.object(agent, "gather_context"), \
             patch.object(agent, "plan_task"), \
             patch.object(agent, "execute_plan", return_value=long_response):
            out = agent.handle_task_message("task do something", voice_compact=True)
        self.assertLessEqual(len(out), 180)
        self.assertTrue(out.endswith("."))
        self.assertNotIn("...", out)

    def test_handle_task_message_voice_compact_word_boundary_ellipsis(self):
        long_response = "word " * 60
        with patch.object(agent, "gather_context"), \
             patch.object(agent, "plan_task"), \
             patch.object(agent, "execute_plan", return_value=long_response):
            out = agent.handle_task_message("task do something", voice_compact=True)
        self.assertLessEqual(len(out), 180)
        self.assertTrue(out.endswith("..."))
        self.assertNotRegex(out, r"\w\w$")

    def test_handle_task_message_voice_compact_short_passthrough(self):
        short_response = "Done, sir."
        with patch.object(agent, "gather_context"), \
             patch.object(agent, "plan_task"), \
             patch.object(agent, "execute_plan", return_value=short_response):
            out = agent.handle_task_message("task do something", voice_compact=True)
        self.assertEqual(out, short_response)


class PlannerSafetyTests(unittest.TestCase):
    """Round 15: Fireworks planner swap + confirmation-gated fallbacks."""

    def test_model_plan_uses_fireworks_not_grok(self):
        import json as _json

        from backend.services import fireworks_client

        payload = {
            "choices": [
                {
                    "message": {
                        "content": _json.dumps({
                            "ok": True,
                            "confidence": 0.9,
                            "summary": "planned",
                            "requires_confirmation": False,
                            "steps": [],
                            "response": "hi",
                        })
                    }
                }
            ]
        }
        with patch.object(agent, "ask_fireworks", return_value=payload) as mock_fw:
            self.assertFalse(hasattr(agent, "ask_grok"))
            plan = agent._model_plan("do something", {})
        mock_fw.assert_called_once()
        _, kwargs = mock_fw.call_args
        self.assertEqual(kwargs.get("temperature"), 0.1)
        self.assertEqual(kwargs.get("max_tokens"), 650)
        # F49 (G8): the planner model is resolved THROUGH the registry —
        # the call carries the capability-validated planner model
        # explicitly (the snapshot travels with the call), no longer the
        # bare Fireworks default.
        self.assertEqual(
            kwargs.get("model"),
            "accounts/fireworks/models/qwen3p7-plus",
        )
        self.assertEqual(plan["summary"], "planned")

    def test_plan_task_fallback_requires_confirmation(self):
        with patch.object(agent, "_heuristic_plan", return_value=None), \
             patch.object(agent, "_model_plan", return_value=None):
            plan = agent.plan_task(
                "Go to 1hd.to website, search for One Piece Movie Red, and play it",
                {"windows": {}, "editor": {}, "browser": {}},
            )
        self.assertTrue(plan["requires_confirmation"])
        self.assertEqual(plan["steps"][0]["tool"], "windows.screen_action")

    def test_normalize_plan_empty_steps_fallback_requires_confirmation(self):
        plan = agent._normalize_plan(
            {"ok": True, "confidence": 0.5, "summary": "x"}, "do a thing"
        )
        self.assertTrue(plan["requires_confirmation"])
        self.assertEqual(plan["steps"][0]["tool"], "windows.screen_action")


class RawPayloadPreservationTests(unittest.TestCase):
    """F12: the heuristic planner detects verbs case-insensitively but keeps
    the raw transcription for every payload — paths, shell flags, URLs and
    file contents cross the planner/tool boundary unchanged."""

    def test_write_content_keeps_case(self):
        plan = agent._heuristic_plan(
            "create a file called Notes.txt with HelloWorld", {}
        )
        step = plan["steps"][0]
        self.assertEqual(step["tool"], "code.write_file")
        self.assertEqual(step["args"]["path"], "Notes.txt")
        self.assertEqual(step["args"]["content"], "HelloWorld")

    def test_run_command_keeps_uppercase_shell_flags(self):
        plan = agent._heuristic_plan("run dir /A /S /B", {})
        step = plan["steps"][0]
        self.assertEqual(step["tool"], "code.run_command")
        self.assertEqual(step["args"]["command"], "dir /A /S /B")

    def test_open_url_keeps_mixed_case(self):
        plan = agent._heuristic_plan("open GitHub.com/MyRepo", {})
        step = plan["steps"][0]
        self.assertEqual(step["tool"], "browser.open_url")
        self.assertEqual(step["args"]["url"], "GitHub.com/MyRepo")

    def test_write_content_keeps_newlines(self):
        plan = agent._heuristic_plan(
            "create a file called notes.txt with line one\nline two", {}
        )
        step = plan["steps"][0]
        self.assertEqual(step["args"]["content"], "line one\nline two")

    def test_read_path_keeps_spaces_and_case(self):
        plan = agent._heuristic_plan(
            "read My Documents/Report Final.txt", {}
        )
        step = plan["steps"][0]
        self.assertEqual(step["tool"], "code.read_file")
        self.assertEqual(step["args"]["path"], "My Documents/Report Final.txt")

    def test_search_query_keeps_case(self):
        plan = agent._heuristic_plan("search Python Decorators", {})
        step = plan["steps"][0]
        self.assertEqual(step["tool"], "browser.search_web")
        self.assertEqual(step["args"]["query"], "Python Decorators")

    def test_verb_detection_still_case_insensitive(self):
        plan = agent._heuristic_plan("RUN dir /A", {})
        self.assertEqual(plan["steps"][0]["tool"], "code.run_command")
        self.assertEqual(plan["steps"][0]["args"]["command"], "dir /A")
        plan2 = agent._heuristic_plan("Read C:\\Temp\\File.txt", {})
        self.assertEqual(plan2["steps"][0]["tool"], "code.read_file")
        self.assertEqual(plan2["steps"][0]["args"]["path"], "C:\\Temp\\File.txt")


class ListenerRawTranscriptionTests(unittest.TestCase):
    """F12: recognize_multilingual returns the raw transcription alongside
    the normalized control text; listen() hands the raw text onward."""

    def test_recognize_multilingual_returns_raw_and_normalized(self):
        import speech_recognition as sr

        from backend.services import listener as listener_mod

        with patch.object(
            listener_mod, "recognize_inworld",
            side_effect=sr.RequestError("offline in tests"),
        ), patch.object(
            listener_mod, "recognize_local_whisper",
            side_effect=sr.RequestError("offline in tests"),
        ), patch.object(
            listener_mod, "recognize_google_or_groq",
            return_value="Create File Hello.txt",
        ):
            raw, normalized, language = listener_mod.recognize_multilingual(
                object()
            )
        self.assertEqual(raw, "Create File Hello.txt")
        self.assertEqual(normalized, "create file hello.txt")
        self.assertEqual(language, listener_mod.RECOGNITION_LANGUAGES[0])

    def test_recognize_multilingual_no_transcript(self):
        import speech_recognition as sr

        from backend.services import listener as listener_mod

        with patch.object(
            listener_mod, "recognize_inworld",
            side_effect=sr.RequestError("offline in tests"),
        ), patch.object(
            listener_mod, "recognize_local_whisper",
            side_effect=sr.RequestError("offline in tests"),
        ), patch.object(
            listener_mod, "recognize_google_or_groq",
            side_effect=sr.UnknownValueError(),
        ):
            result = listener_mod.recognize_multilingual(object())
        self.assertEqual(result, (None, None, None))


class G1Round2ClosedLoopTests(unittest.TestCase):
    """G1 Round 2 (F01): closed-loop execution - postconditions, dependency
    skipping, honest summaries, explicit truncation."""

    def tearDown(self):
        agent._pending_task_action = None

    def _plan(self, steps, **kw):
        plan = {"ok": True, "confidence": 0.9, "summary": "Doing things.",
                "requires_confirmation": False, "steps": steps,
                "response": ""}
        plan.update(kw)
        return plan

    def _step(self, tool, args=None, **kw):
        step = {"tool": tool, "args": args or {}, "risk": "safe",
                "reason": "test step"}
        step.update(kw)
        return step

    def test_all_ok_returns_completed_done_sir(self):
        plan = self._plan([self._step("browser.search_web", {"query": "q"})])
        with patch.object(agent, "_execute_step",
                          return_value="Search results: x"):
            result = agent.execute_plan(plan, {})
        self.assertIsInstance(result, TaskResult)
        self.assertEqual(result.status, "completed")
        self.assertTrue(str(result).startswith("Done, sir."))
        self.assertIn("Search results: x", str(result))

    def test_plan_not_ok_returns_failed_preserving_wording(self):
        result = agent.execute_plan({"ok": False, "response": ""}, {})
        self.assertEqual(result.status, "failed")
        self.assertFalse(result)
        self.assertEqual(str(result), "I could not plan that task.")
        custom = agent.execute_plan({"ok": False, "response": "Nope."}, {})
        self.assertEqual(custom.status, "failed")
        self.assertEqual(str(custom), "Nope.")

    def test_response_without_steps_returned_as_is(self):
        plan = self._plan([], response="Just an answer.")
        result = agent.execute_plan(plan, {})
        self.assertEqual(result.status, "completed")
        self.assertEqual(str(result), "Just an answer.")

    def test_postcondition_failure_is_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            ghost = os.path.join(tmp, "ghost.txt")
            ok_but_missing = {"ok": True, "content": "Wrote 5 chars",
                              "error": "", "path": ghost, "exit_code": 0}
            plan = self._plan(
                [self._step("code.write_file",
                            {"path": ghost, "content": "x"})])
            with patch.object(agent.code_tools, "call_tool",
                              return_value=ok_but_missing):
                result = agent.execute_plan(plan, {})
        self.assertEqual(result.status, "partial")
        self.assertTrue(result.evidence)
        self.assertIn("postcondition failed", result.evidence[0])
        self.assertIn(ghost, result.evidence[0])

    def test_folder_failure_skips_inside_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "newdir")
            target = os.path.join(folder, "a.txt")
            calls = []

            def fake_call(tool, args, **kwargs):
                calls.append(tool)
                if tool == "code.create_folder":
                    return {"ok": False, "error": "denied", "content": "",
                            "path": folder}
                raise AssertionError("must not execute: %s" % tool)

            plan = self._plan([
                self._step("code.create_folder", {"path": folder}),
                self._step("code.write_file",
                           {"path": target, "content": "x"}),
            ])
            with patch.object(agent.code_tools, "call_tool",
                              side_effect=fake_call):
                result = agent.execute_plan(plan, {})
        self.assertEqual(calls, ["code.create_folder"])
        self.assertEqual(result.status, "partial")
        kinds = [line.split(":")[1].strip() if ":" in line else ""
                 for line in result.evidence]
        self.assertIn("failed", kinds)
        self.assertIn("skipped", kinds)
        self.assertTrue(any("prerequisite failed: folder" in line
                            for line in result.evidence))

    def test_depends_on_honored(self):
        calls = []

        def fake_call(tool, args, **kwargs):
            calls.append(tool)
            if tool == "code.run_command":
                return {"ok": False, "error": "boom", "content": "",
                        "exit_code": 1}
            raise AssertionError("must not execute: %s" % tool)

        plan = self._plan([
            self._step("code.run_command", {"command": "boom"}),
            self._step("code.write_file", {"path": "b.txt", "content": "x"},
                       depends_on=[0]),
        ])
        with patch.object(agent.code_tools, "call_tool",
                          side_effect=fake_call):
            result = agent.execute_plan(plan, {})
        self.assertEqual(calls, ["code.run_command"])
        self.assertEqual(result.status, "partial")
        self.assertTrue(any("prerequisite failed: step 0" in line
                            for line in result.evidence))

    def test_independent_steps_continue_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            real_file = os.path.join(tmp, "real.txt")
            with open(real_file, "w", encoding="utf-8") as handle:
                handle.write("already here")
            calls = []

            def fake_call(tool, args, **kwargs):
                calls.append(tool)
                if tool == "code.run_command":
                    return {"ok": False, "error": "boom", "content": "",
                            "exit_code": 1}
                return {"ok": True, "content": "wrote it", "error": "",
                        "path": real_file, "exit_code": 0}

            plan = self._plan([
                self._step("code.run_command", {"command": "boom"}),
                self._step("code.write_file",
                           {"path": real_file, "content": "x"}),
            ])
            with patch.object(agent.code_tools, "call_tool",
                              side_effect=fake_call):
                result = agent.execute_plan(plan, {})
        self.assertEqual(calls, ["code.run_command", "code.write_file"])
        self.assertEqual(result.status, "partial")
        self.assertIn("Partly done, sir.", str(result))

    def test_response_override_removed_after_steps_run(self):
        plan = self._plan(
            [self._step("browser.search_web", {"query": "q"})],
            response="Pre-authored answer.")
        with patch.object(agent, "_execute_step",
                          return_value="Results here."):
            result = agent.execute_plan(plan, {})
        self.assertNotIn("Pre-authored", str(result))
        self.assertIn("Done, sir.", str(result))

    def test_unknown_tool_is_failure_not_silent_ok(self):
        plan = self._plan([self._step("nope.unknown", {})])
        result = agent.execute_plan(plan, {})
        self.assertEqual(result.status, "partial")
        self.assertTrue(any("unknown tool" in line
                            for line in result.evidence))

    def test_truncation_flag_and_note(self):
        # F01: the cap is a configurable step BUDGET (default 24), and what it
        # omits is recorded as unmet goals instead of vanishing.
        budget = agent.TASK_MAX_STEPS
        raw = {"ok": True, "confidence": 0.9, "summary": "s",
               "requires_confirmation": False,
               "steps": [{"tool": "browser.search_web",
                          "args": {"query": "q%d" % i}}
                         for i in range(budget + 2)],
               "response": "Working."}
        plan = agent._normalize_plan(raw, "search lots")
        self.assertEqual(len(plan["steps"]), budget)
        self.assertTrue(plan["truncated"])
        self.assertEqual(plan["intended_steps"], budget + 2)
        self.assertEqual(len(plan["omitted_steps"]), 2)
        self.assertIn("Plan limited to the first %d steps" % budget,
                      plan["response"])
        self.assertIn("Plan limited to the first %d steps" % budget,
                      plan["summary"])
        short = agent._normalize_plan(
            {"ok": True, "steps": [{"tool": "browser.search_web",
                                    "args": {"query": "q"}}]}, "cmd")
        self.assertFalse(short["truncated"])
        self.assertNotIn("Plan limited", short["response"])

    def test_confirmation_preview_byte_identical(self):
        plan = self._plan(
            [self._step("code.run_command", {"command": "pip list"})],
            requires_confirmation=True)
        expected = "%s Say confirm task to proceed, or cancel." % (
            agent._confirmation_preview(plan))
        result = agent.execute_plan(plan, {})
        self.assertEqual(result.status, "needs_input")
        self.assertEqual(str(result), expected)
        self.assertIsNotNone(agent._pending_task_action)

    def test_consume_yes_executes_no_skips(self):
        plan = self._plan(
            [self._step("browser.search_web", {"query": "q"})],
            requires_confirmation=True)
        with patch.object(agent, "_execute_step",
                          return_value="Search done."):
            agent.execute_plan(plan, {})
            yes = agent.consume_task_confirmation("yes, go ahead")
            self.assertIn("Done, sir.", yes)
            self.assertIn("Search done.", yes)
            self.assertIsNone(agent._pending_task_action)
            agent.execute_plan(plan, {})
            no = agent.consume_task_confirmation("no thanks")
            self.assertEqual(no, "As you wish, sir. I will skip that.")
            self.assertIsNone(agent._pending_task_action)


class PlannerPathSanitizerTests(unittest.TestCase):
    """Live fix: the planner hallucinated ~/Desktop paths and the heuristic
    hardcoded ~/Desktop (wrong under OneDrive redirection). All fs facts
    come from agent._known_folders, patched here for determinism."""

    def _folders(self, home, desktop):
        return {"username": "mayan", "home": home, "desktop": desktop,
                "documents": os.path.join(home, "Documents"),
                "downloads": os.path.join(home, "Downloads")}

    def test_username_template_path_redirects_to_real_desktop(self):
        desk = tempfile.mkdtemp(prefix="jarvis_desk_")
        self.addCleanup(shutil.rmtree, desk, True)
        home = os.path.expanduser("~")
        raw_folder = os.path.join("C:" + os.sep, "Users", "<username>",
                                 "Desktop", "astra")
        raw_file = os.path.join(raw_folder, "flower.txt")
        plan = {"ok": True, "confidence": 0.9, "summary": "s",
                "requires_confirmation": False, "response": "",
                "steps": [
                    {"tool": "code.create_folder",
                     "args": {"path": raw_folder},
                     "risk": "safe", "reason": "t"},
                    {"tool": "code.write_file",
                     "args": {"path": raw_file, "content": "petals"},
                     "risk": "safe", "reason": "t"},
                ]}
        with patch.object(agent, "_known_folders",
                          return_value=self._folders(home, desk)):
            normalized = agent._normalize_plan(
                plan, "create a folder on desktop by the name astra")
            self.assertEqual(normalized["steps"][0]["args"]["path"],
                             os.path.join(desk, "astra"))
            self.assertEqual(normalized["steps"][1]["args"]["path"],
                             os.path.join(desk, "astra", "flower.txt"))
            # Real tools, tmp-backed desktop: the exact live scenario.
            result = agent.execute_plan(normalized, {}, confirmed=True)
        self.assertEqual(result.status, "completed")
        with open(os.path.join(desk, "astra", "flower.txt"),
                  encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "petals")

    def test_placeholder_variants_substitute(self):
        home = os.path.join("C:" + os.sep, "Users", "mayan")
        folders = self._folders(home, os.path.join(home, "Desktop"))
        with patch.object(agent, "_known_folders", return_value=folders):
            self.assertEqual(
                agent._sanitize_fs_text("C:\\Users\\{username}\\notes.txt"),
                "C:\\Users\\mayan\\notes.txt")
            self.assertEqual(
                agent._sanitize_fs_text("C:\\Users\\%username%\\notes.txt"),
                "C:\\Users\\mayan\\notes.txt")
            self.assertEqual(
                agent._sanitize_fs_text("%USERPROFILE%\\notes.txt"),
                home + "\\notes.txt")
            self.assertEqual(agent._sanitize_fs_text("~/notes.txt"),
                             home + "/notes.txt")
            self.assertEqual(
                agent._sanitize_fs_text("C:\\Users\\<USERNAME>\\x"),
                "C:\\Users\\mayan\\x")

    def test_redirect_desktop_noop_when_not_redirected(self):
        home = os.path.join("C:" + os.sep, "Users", "mayan")
        local = os.path.join(home, "Desktop")
        folders = self._folders(home, local)
        target = os.path.join(local, "a.txt")
        with patch.object(agent, "_known_folders", return_value=folders):
            self.assertEqual(agent._redirect_desktop(target), target)

    def test_invalid_chars_reject_without_executing(self):
        stray = tempfile.mkdtemp(prefix="jarvis_bad_")
        self.addCleanup(shutil.rmtree, stray, True)
        home = os.path.expanduser("~")
        plan = {"ok": True, "confidence": 0.9, "summary": "s",
                "requires_confirmation": False, "response": "",
                "steps": [{"tool": "code.write_file",
                           "args": {"path": os.path.join(stray, "a<b.txt"),
                                    "content": "x"},
                           "risk": "safe", "reason": "t"}]}
        with patch.object(agent, "_known_folders",
                          return_value=self._folders(home, home)), \
             patch.object(agent.code_tools, "call_tool") as call_tool:
            result = agent.execute_plan(
                agent._normalize_plan(plan, "write it"), {}, confirmed=True)
        call_tool.assert_not_called()
        self.assertEqual(result.status, "partial")
        self.assertTrue(any("invalid characters in path" in line
                            for line in result.evidence))

    def test_single_quote_path_is_legal_double_quote_rejected(self):
        # A single quote is a VALID Windows filename character (e.g.
        # "don't stop.txt") and must execute; a double quote is invalid.
        stray = tempfile.mkdtemp(prefix="jarvis_quote_")
        self.addCleanup(shutil.rmtree, stray, True)
        home = os.path.expanduser("~")
        legal = {"ok": True, "confidence": 0.9, "summary": "s",
                 "requires_confirmation": False, "response": "",
                 "steps": [{"tool": "code.write_file",
                            "args": {"path": os.path.join(stray, "don't.txt"),
                                     "content": "x"},
                            "risk": "safe", "reason": "t"}]}
        with patch.object(agent, "_known_folders",
                          return_value=self._folders(home, home)):
            result = agent.execute_plan(
                agent._normalize_plan(legal, "write it"), {}, confirmed=True)
        self.assertEqual(result.status, "completed")
        self.assertTrue(os.path.exists(os.path.join(stray, "don't.txt")))

        illegal = {"ok": True, "confidence": 0.9, "summary": "s",
                   "requires_confirmation": False, "response": "",
                   "steps": [{"tool": "code.write_file",
                              "args": {"path": os.path.join(stray, 'a"b.txt'),
                                       "content": "x"},
                              "risk": "safe", "reason": "t"}]}
        with patch.object(agent, "_known_folders",
                          return_value=self._folders(home, home)), \
             patch.object(agent.code_tools, "call_tool") as call_tool:
            result = agent.execute_plan(
                agent._normalize_plan(illegal, "write it"), {}, confirmed=True)
        call_tool.assert_not_called()
        self.assertEqual(result.status, "partial")
        self.assertTrue(any("invalid characters in path" in line
                            for line in result.evidence))

    def test_run_command_is_never_rewritten_and_keeps_pipes(self):
        # F12: a command line is payload, not a template. Placeholder
        # substitution used to rewrite an argument that is a command line
        # (`<username>` -> the local user), which also mangled quoted code;
        # only designated path fields get the path sanitizer now. The command
        # keeps every byte, pipes included, and the shell keeps its own
        # placeholders (%USERPROFILE%) to expand.
        home = os.path.expanduser("~")
        plan = {"ok": True, "confidence": 0.9, "summary": "s",
                "requires_confirmation": False, "response": "",
                "steps": [{"tool": "code.run_command",
                           "args": {"command": "echo hello <username> | sort"},
                           "risk": "safe", "reason": "t"},
                          {"tool": "code.run_command",
                           "args": {"command": "dir > out.txt"},
                           "risk": "safe", "reason": "t"}]}
        with patch.object(agent, "_known_folders",
                          return_value=self._folders(home, home)):
            normalized = agent._normalize_plan(plan, "run it")
            self.assertEqual(
                normalized["steps"][0]["args"]["command"],
                "echo hello <username> | sort")
            self.assertEqual(normalized["steps"][1]["args"]["command"],
                             "dir > out.txt")
            ok = {"ok": True, "content": "hi", "error": "",
                  "path": "", "exit_code": 0}
            with patch.object(agent.code_tools, "call_tool",
                              return_value=ok) as call_tool:
                result = agent.execute_plan(normalized, {}, confirmed=True)
        self.assertEqual(result.status, "completed")
        self.assertEqual(
            call_tool.call_args_list[0][0][1]["command"],
            "echo hello <username> | sort")

    def test_planner_prompt_carries_filesystem_context(self):
        home = os.path.join("C:" + os.sep, "Users", "mayan")
        desk = os.path.join(home, "OneDrive", "Desktop")
        with patch.object(agent, "_known_folders",
                          return_value=self._folders(home, desk)):
            prompt = agent._build_planner_prompt("make a folder", {})
        self.assertIn("mayan", prompt)
        self.assertIn(home, prompt)
        self.assertIn(desk, prompt)
        self.assertIn("NEVER invent paths", prompt)

    def test_folder_plan_path_uses_known_desktop(self):
        desk = os.path.join("C:" + os.sep, "desk")
        home = os.path.join("C:" + os.sep, "Users", "mayan")
        with patch.object(agent, "_known_folders",
                          return_value=self._folders(home, desk)):
            self.assertEqual(
                agent._folder_plan_path(
                    "create a folder on desktop named astra"),
                os.path.join(desk, "astra"))


class KnownFolderGuidTests(unittest.TestCase):
    """Regression guard: two known-folder GUIDs once shipped wrong and the
    per-folder fallback silently masked it (OneDrive machines resolved the
    local ~/Desktop instead of the real one). Pin the canonical values."""

    _CANONICAL = {
        "desktop": "{b4bfcc3a-db2c-424c-b029-7fe99a87c641}",
        "documents": "{fdd39ad0-238f-46af-adb4-6c85480369c7}",
        "downloads": "{374de290-123f-4565-9164-39c4925e467b}",
    }

    def test_known_folder_guids_are_canonical(self):
        for key, guid in self._CANONICAL.items():
            self.assertEqual(
                agent._KNOWN_FOLDER_IDS.get(key, "").lower(), guid,
                "known-folder GUID for %s must be the canonical value" % key)

    @unittest.skipIf(os.name != "nt", "Windows known-folder live check")
    def test_known_folders_match_dotnet_on_windows(self):
        # Live cross-check: SHGetKnownFolderPath must agree with .NET's
        # [Environment]::GetFolderPath (which follows OneDrive redirection).
        import subprocess
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "[Environment]::GetFolderPath('Desktop'); "
                 "[Environment]::GetFolderPath('MyDocuments')"],
                stderr=subprocess.DEVNULL, timeout=30)
        except Exception:
            self.skipTest("powershell unavailable")
        dotnet = out.decode("mbcs" if os.name == "nt" else "utf-8",
                            errors="replace").splitlines()
        dotnet = [ln.strip() for ln in dotnet if ln.strip()]
        if len(dotnet) < 2:
            self.skipTest("powershell returned no folder paths")
        folders = agent._known_folders()
        self.assertEqual(folders["desktop"].rstrip("\\"),
                         dotnet[0].rstrip("\\"))
        self.assertEqual(folders["documents"].rstrip("\\"),
                         dotnet[1].rstrip("\\"))


class G4CodingInterfaceTests(unittest.TestCase):
    """G4 (F11 + F15): the coding interface.

    F11 — native coding gets a real inspection/edit loop (code.search,
    bounded code.read_range, code.apply_patch, code.inspect_diff,
    code.run_checks) with structured artifacts and continuation cursors
    instead of a clipped whole-file read.
    F15 — the editor bridge becomes a structured coding interface
    (buffers, diagnostics, symbols, references, workspace search, test
    results, version-checked WorkspaceEdit) that returns actual
    inspection data to the planning loop.
    """

    def tearDown(self):
        agent._pending_task_action = None

    # ── F11: registration + planner advertisement ───────────────────────
    def test_f11_and_f15_tools_are_registered_and_advertised(self):
        for tool in ("code.search", "code.read_range", "code.apply_patch",
                     "code.inspect_diff", "code.run_checks"):
            self.assertIn(tool, agent.code_tools.TOOL_REGISTRY,
                          "%s must be in TOOL_REGISTRY" % tool)
            self.assertIn(tool, agent._CODE_TOOLS)
        prompt = agent._build_planner_prompt("do the thing", {})
        for tool in ("code.search", "code.read_range", "code.apply_patch",
                     "code.inspect_diff", "code.run_checks",
                     "editor.read_buffer", "editor.diagnostics",
                     "editor.symbols", "editor.references",
                     "editor.workspace_search", "editor.test_results",
                     "editor.apply_workspace_edit"):
            self.assertIn(tool + ":", prompt,
                          "planner must advertise %s" % tool)

    def test_read_only_inspection_tools_need_no_confirmation(self):
        for tool in ("code.search", "code.read_range", "code.inspect_diff"):
            self.assertIn(tool, agent.SAFE_TOOLS)
            self.assertNotIn(tool, agent.CONFIRM_TOOLS)
        # Patch and check runs are writes/executions: they stay gated.
        for tool in ("code.apply_patch", "code.run_checks"):
            self.assertIn(tool, agent.CONFIRM_TOOLS)

    # ── F11: structured search + continuation cursor ────────────────────
    def test_search_returns_matches_and_resumable_cursor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("needle one\nfiller\nneedle two\nneedle three\n")
            first = agent.code_tools.search("needle", path=tmp, max_results=2)
            self.assertTrue(first["ok"])
            self.assertEqual(first["match_count"], 2)
            self.assertEqual([m["line"] for m in first["matches"]], [1, 3])
            self.assertTrue(first["has_more"])
            self.assertIn("next_cursor", first)

            second = agent.code_tools.search(
                "needle", path=tmp, max_results=2,
                cursor=first["next_cursor"])
            self.assertTrue(second["ok"])
            # Resumed exactly after line 3: the tail is neither repeated
            # nor silently dropped.
            self.assertEqual([m["line"] for m in second["matches"]], [4])
            self.assertFalse(second["has_more"])

    def test_read_range_reports_cursor_and_content_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "b.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join("line %d" % i for i in range(1, 11)))
            window = agent.code_tools.read_range(path, start_line=1,
                                                 max_lines=3)
            self.assertTrue(window["ok"])
            self.assertEqual(window["end_line"], 3)
            self.assertEqual(window["total_lines"], 10)
            self.assertTrue(window["has_more"])
            self.assertEqual(window["next_start"], 4)
            self.assertTrue(window["content_hash"])
            self.assertIn("line 1", window["content"])

            rest = agent.code_tools.read_range(path, start_line=4,
                                               max_lines=10)
            self.assertEqual(rest["start_line"], 4)
            self.assertFalse(rest["has_more"])
            self.assertIn("line 10", rest["content"])

    def test_continuation_note_survives_clipping(self):
        long_content = "x" * 2000
        rendered = agent._code_tool_text({
            "ok": True, "content": long_content,
            "has_more": True, "next_start": 201,
        })
        self.assertIn("start_line=201", rendered)
        self.assertIn("cursor", agent._continuation_note(
            {"has_more": True,
             "next_cursor": {"file": 2, "after_line": 7}}))

    # ── F11: surgical edits ─────────────────────────────────────────────
    def test_apply_patch_dry_run_then_real_write_and_inspect_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "c.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("line1\nline2\nline3\n")
            patch = ("--- a/c.txt\n+++ b/c.txt\n@@ -1,3 +1,3 @@\n"
                     " line1\n-line2\n+LINE2\n line3\n")

            dry = agent.code_tools.apply_patch(path, patch, dry_run=True)
            self.assertTrue(dry["ok"])
            self.assertFalse(dry["applied"])
            self.assertIn("+LINE2", dry["diff"])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "line1\nline2\nline3\n")

            applied = agent.code_tools.apply_patch(path, patch)
            self.assertTrue(applied["ok"])
            self.assertTrue(applied["applied"])
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "line1\nLINE2\nline3\n")

            diff = agent.code_tools.inspect_diff(path)
            self.assertTrue(diff["ok"])
            self.assertTrue(diff["has_changes"])
            self.assertIn("+LINE2", diff["diff"])

    def test_apply_patch_rejects_malformed_or_unapplicable_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "d.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("alpha\nbeta\n")
            bad = agent.code_tools.apply_patch(path, "not a diff at all")
            self.assertFalse(bad["ok"])
            self.assertIn("Malformed patch", bad["error"])

            stale = agent.code_tools.apply_patch(
                path, "--- a/d.txt\n+++ b/d.txt\n@@ -1,2 +1,2 @@\n"
                      "-nope\n-alsonope\n+changed\n+x\n")
            self.assertFalse(stale["ok"])
            self.assertIn("does not apply", stale["error"])

    def test_apply_patch_postcondition_failure_is_partial(self):
        plan = {"ok": True, "confidence": 0.9, "summary": "Patching.",
                "requires_confirmation": False, "response": "",
                "steps": [{"tool": "code.apply_patch",
                           "args": {"path": "ghost.txt", "patch": "x"},
                           "risk": "safe", "reason": "test"}]}
        ghost = os.path.join(tempfile.gettempdir(),
                             "jarvis_g4_ghost_%d.txt" % os.getpid())
        if os.path.exists(ghost):
            os.remove(ghost)
        try:
            with patch.object(agent.code_tools, "call_tool",
                              return_value={"ok": True, "content": "Patched",
                                            "error": "", "path": ghost,
                                            "exit_code": 0}):
                result = agent.execute_plan(plan, {})
        finally:
            if os.path.exists(ghost):
                os.remove(ghost)
        self.assertEqual(result.status, "partial")
        self.assertIn("postcondition failed", result.evidence[0])

    def test_run_checks_reports_a_structured_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.py")
            with open(good, "w", encoding="utf-8") as handle:
                handle.write("value = 1\n")
            passed = agent.code_tools.run_checks("py_compile", good)
            self.assertTrue(passed["ok"])
            self.assertTrue(passed["checks"])
            self.assertTrue(passed["checks"][0]["ok"])
            self.assertIn("PASS", passed["content"])

            broken = os.path.join(tmp, "broken.py")
            with open(broken, "w", encoding="utf-8") as handle:
                handle.write("def (\n")
            failed = agent.code_tools.run_checks("py_compile", broken)
            self.assertFalse(failed["ok"])
            self.assertFalse(failed["checks"][0]["ok"])
            self.assertIn("FAIL", failed["content"])

            bogus = agent.code_tools.run_checks("nonsense")
            self.assertFalse(bogus["ok"])
            self.assertIn("Unknown check kind", bogus["error"])

    def test_step_target_falls_back_to_pattern(self):
        step = {"tool": "code.search", "args": {"pattern": "needle"}}
        self.assertEqual(agent._step_target(step), "needle")
        self.assertIn("Searched needle.",
                      agent._ok_fragment(step, "y" * 200))

    # ── F15: structured editor inspections ──────────────────────────────
    def _editor_context(self, state=None):
        return {
            "windows": {"active_window": {"title": "code.exe"},
                        "visible_controls": []},
            "editor": {"available": True, "state": state or {}},
            "browser": {"available": False, "tabs": []},
        }

    def test_inspect_workspace_returns_real_state_not_a_stub(self):
        state = {
            "workspaceFolders": ["C:\\proj"],
            "activeFile": {
                "path": "C:\\proj\\app.py", "uri": "file:///c:/proj/app.py",
                "fileName": "app.py", "languageId": "python",
                "isDirty": True, "lineCount": 120, "version": 7,
                "selection": {"start": {"line": 9, "character": 0},
                              "end": {"line": 11, "character": 4}},
            },
            "diagnostics": [{"file": "C:\\proj\\app.py", "severity": 0,
                             "message": "undefined name 'x'", "code": "F821",
                             "range": {"start": {"line": 11,
                                                 "character": 2}}}],
        }
        text = agent._execute_step(
            {"tool": "editor.inspect_workspace", "args": {}},
            self._editor_context(state))
        self.assertNotIn("available through the bridge", text)
        self.assertIn("app.py", text)
        self.assertIn("version 7", text)
        self.assertIn("10:1-12:5", text)  # 0-based -> 1-based selection
        self.assertIn("1 error", text)
        self.assertIn("F821", text)

    def test_read_buffer_renders_version_and_numbered_lines(self):
        with patch.object(agent.editor_bridge, "read_buffer",
                          return_value={"ok": True, "path": "C:\\proj\\a.py",
                                        "uri": "file:///c:/proj/a.py",
                                        "version": 3, "lineCount": 40,
                                        "startLine": 1, "endLine": 2,
                                        "text": "def foo():\n    return 1",
                                        "textTruncated": False}):
            text = agent._execute_step(
                {"tool": "editor.read_buffer", "args": {"path": "a.py"}},
                self._editor_context())
        self.assertIn("version 3", text)
        self.assertIn("1 | def foo():", text)
        self.assertIn("2 |     return 1", text)

    def test_diagnostics_and_symbols_and_references_render(self):
        with patch.object(agent.editor_bridge, "diagnostics",
                          return_value={"ok": True, "diagnostics": [
                              {"file": "C:\\proj\\a.py", "severity": 1,
                               "message": "unused import",
                               "range": {"start": {"line": 4,
                                                   "character": 0}}}]}):
            text = agent._execute_step(
                {"tool": "editor.diagnostics", "args": {}},
                self._editor_context())
        self.assertIn("warning", text)
        self.assertIn("a.py:5:1", text)
        self.assertIn("unused import", text)

        with patch.object(agent.editor_bridge, "symbols",
                          return_value={"ok": True, "symbols": [
                              {"name": "Foo", "kind": 4,
                               "range": {"start": {"line": 3,
                                                   "character": 0}}},
                              {"name": "bar", "kind": 5,
                               "containerName": "Foo",
                               "range": {"start": {"line": 8,
                                                   "character": 4}}}]}):
            text = agent._execute_step(
                {"tool": "editor.symbols", "args": {}},
                self._editor_context())
        self.assertIn("class Foo", text)
        self.assertIn("method bar (in Foo)", text)

        with patch.object(agent.editor_bridge, "references",
                          return_value={"ok": True, "references": [
                              {"path": "C:\\proj\\b.py",
                               "range": {"start": {"line": 11,
                                                   "character": 6}}}]}):
            text = agent._execute_step(
                {"tool": "editor.references", "args": {}},
                self._editor_context())
        self.assertIn("1 reference", text)
        self.assertIn("b.py:12:7", text)

    def test_workspace_search_and_test_results_render(self):
        with patch.object(agent.editor_bridge, "workspace_search",
                          return_value={"ok": True, "query": "needle",
                                        "matches": [{"path": "C:\\proj\\a.py",
                                                     "line": 3,
                                                     "text": "needle = 1"}],
                                        "scannedFiles": 42,
                                        "truncated": False}):
            text = agent._execute_step(
                {"tool": "editor.workspace_search",
                 "args": {"query": "needle"}}, self._editor_context())
        self.assertIn('1 match for "needle"', text)
        self.assertIn("a.py:3: needle = 1", text)

        with patch.object(agent.editor_bridge, "test_results",
                          return_value={"ok": False, "command": "pytest -q",
                                        "cwd": "C:\\proj", "exitCode": 1,
                                        "output": "1 failed",
                                        "error": "exited with code 1"}):
            text = agent._execute_step(
                {"tool": "editor.test_results", "args": {}},
                self._editor_context())
        self.assertIn("Tests failed (exit 1)", text)
        self.assertIn("pytest -q", text)

    # ── F15: honest failures + version safety ───────────────────────────
    def test_editor_tool_failure_is_recorded_not_swallowed(self):
        plan = {"ok": True, "confidence": 0.9, "summary": "Inspecting.",
                "requires_confirmation": False, "response": "",
                "steps": [{"tool": "editor.symbols", "args": {},
                           "risk": "safe", "reason": "test"}]}
        with patch.object(agent.editor_bridge, "symbols",
                          return_value={"ok": False,
                                        "error": "No active text editor."}):
            result = agent.execute_plan(plan, self._editor_context())
        self.assertEqual(result.status, "partial")
        self.assertIn("editor.symbols", result.evidence[0])
        self.assertIn("failed", result.evidence[0])
        self.assertIn("No active text editor.", result.evidence[0])

    def test_version_mismatch_tells_the_loop_to_re_read(self):
        with patch.object(agent.editor_bridge, "apply_workspace_edit",
                          return_value={"ok": False, "status": 409,
                                        "error": "Version mismatch for a.py: "
                                                 "document is at version 5, "
                                                 "expected 3.",
                                        "expectedVersion": 3,
                                        "actualVersion": 5}):
            text = agent._execute_step(
                {"tool": "editor.apply_workspace_edit",
                 "args": {"edits": [{"path": "a.py", "new_text": "x"}]}},
                self._editor_context())
        self.assertIn("editor.apply_workspace_edit failed", text)
        self.assertIn("editor.read_buffer", text)

    def test_stale_selection_is_refused_before_editing(self):
        # F15: the plan-time target carries its document VERSION too — the
        # selection alone is not a stable edit target (content can change at
        # the same position). A drifted target edits nothing.
        context = self._editor_context({
            "activeFile": {"uri": "file:///c:/proj/a.py",
                           "version": 7,
                           "selection": {"start": {"line": 1,
                                                   "character": 0},
                                         "end": {"line": 1,
                                                 "character": 4}}}})
        with patch.object(agent.editor_bridge, "active_selection_matches",
                          return_value=False) as check:
            with patch.object(agent.editor_bridge, "edit_active_selection",
                              return_value={"message": "edited"}) as fake:
                text = agent._execute_step(
                    {"tool": "editor.edit_active_selection",
                     "args": {"text": "new"}}, context)
        fake.assert_not_called()
        self.assertEqual(check.call_args.kwargs.get("expected_version"), 7)
        self.assertIn("selection changed", text)

    def test_workspace_edit_confirmation_preview_names_files(self):
        plan = {"steps": [{"tool": "editor.apply_workspace_edit",
                           "args": {"edits": [
                               {"path": "C:\\proj\\a.py", "new_text": "x"},
                               {"path": "C:\\proj\\b.py", "new_text": "y"}]},
                           "risk": "risky", "reason": "Fixing."}]}
        preview = agent._confirmation_preview(plan)
        self.assertIn("Ready to edit", preview)
        self.assertIn("a.py", preview)
        self.assertIn("b.py", preview)
        self.assertIn("editor.apply_workspace_edit",
                      agent.CONFIRM_SENSITIVE_PREVIEW)


if __name__ == "__main__":
    unittest.main()
