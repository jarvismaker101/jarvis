"""F12 — preserve literal task payloads (task-agent half).

Acceptance (audit report): "Mixed-case paths, signed URLs, uppercase flags,
quoted and, indentation/newlines, and literal placeholders survive typed,
voice, and wake-tail routes unchanged."

The typed/voice/wake halves of that sentence are pinned by
``test_f12_literal_payloads``. This module pins the remaining half named by
the audit — "placeholder replacement can alter quoted code":

  * ``_resolve_step_args`` substituted into EVERY string argument, so an
    observation from an earlier step rewrote command lines, script bodies,
    quoted literals and free text that merely contained a placeholder;
  * ``_normalize_plan`` ran the filesystem placeholder sanitizer over
    ``code.run_command``'s command line, rewriting the user's own bytes;
  * the write-request content delimiter matched inside a file name
    ("Q4 Report With Care.TXT" was truncated at its own "With").

Everything here is pure (no model, no network, no editor, no shell).
"""

import unittest
from unittest.mock import patch

from backend.services.task_agent import agent


class PlaceholderSubstitutionScopeTests(unittest.TestCase):
    """Acceptance: placeholders resolve only where they are templates."""

    OBSERVATIONS = {
        0: {"path": "C:\\Users\\Me\\Q4 Report.TXT",
            "name": "Q4 Report",
            "next_start": 12,
            "matches": ["m"],
            "lines": [{"text": "keep  spaces"}]},
    }

    def test_command_lines_and_script_bodies_are_never_rewritten(self):
        args = {"command": "patch {{step0.path}} --Dry-Run -X POST",
                "script": "deploy {{step0.name}}.ps1 --Wait True"}
        resolved = agent._resolve_step_args(args, self.OBSERVATIONS)
        self.assertEqual(resolved["command"],
                         "patch {{step0.path}} --Dry-Run -X POST")
        self.assertEqual(resolved["script"],
                         "deploy {{step0.name}}.ps1 --Wait True")

    def test_quoted_code_keeps_every_byte(self):
        command = "python -c \"print('{{step0.path}}')\" --Dry-Run"
        resolved = agent._resolve_step_args({"command": command},
                                            self.OBSERVATIONS)
        self.assertEqual(resolved["command"], command)

    def test_quoted_phrases_and_mixed_case_paths_survive(self):
        line = 'echo "Rock and Roll" > "D:\\Q4 Report.TXT"'
        resolved = agent._resolve_step_args(
            {"content": line, "command": line + " --Dry-Run"},
            self.OBSERVATIONS)
        self.assertEqual(resolved["content"], line)
        self.assertEqual(resolved["command"], line + " --Dry-Run")

    def test_free_text_indentation_and_newlines_survive(self):
        content = "line one\n  indented {{step0.path}}\nkeep: True"
        resolved = agent._resolve_step_args({"content": content,
                                             "text": content},
                                            self.OBSERVATIONS)
        self.assertEqual(resolved["content"], content)
        self.assertEqual(resolved["text"], content)

    def test_designated_path_fields_resolve_without_touching_other_bytes(self):
        resolved = agent._resolve_step_args(
            {"path": "{{step0.path}}",
             "glob_pattern": "*.TXT",
             "command": "dir /A /S /B"},
            self.OBSERVATIONS)
        self.assertEqual(resolved["path"], "C:\\Users\\Me\\Q4 Report.TXT")
        self.assertEqual(resolved["command"], "dir /A /S /B")
        self.assertEqual(resolved["glob_pattern"], "*.TXT")

    def test_embedded_placeholder_in_a_path_keeps_the_rest_verbatim(self):
        resolved = agent._resolve_step_args(
            {"path": "{{step0.path}}\\Q4 Report.TXT"},
            {0: {"path": "C:\\Users\\Me"}})
        self.assertEqual(resolved["path"], "C:\\Users\\Me\\Q4 Report.TXT")

    def test_nested_containers_keep_payload_arguments_verbatim(self):
        resolved = agent._resolve_step_args(
            {"steps": [{"command": "echo {{step0.path}}",
                        "path": "{{step0.path}}"}]},
            self.OBSERVATIONS)
        self.assertEqual(resolved["steps"][0]["command"],
                         "echo {{step0.path}}")
        self.assertEqual(resolved["steps"][0]["path"],
                         "C:\\Users\\Me\\Q4 Report.TXT")

    def test_an_argument_that_is_one_reference_is_still_a_template(self):
        # F01 compatibility: a value that IS a reference (nothing around it)
        # is an explicit template, not literal payload.
        resolved = agent._resolve_step_args(
            {"pattern": "{{last.matches}}",
             "cursor": "{{step0.next_start}}",
             "text": "{{step0.lines.0.text}}",
             "items": [{"p": "{{step0.path}}"}]},
            self.OBSERVATIONS)
        self.assertEqual(resolved["pattern"], '["m"]')
        self.assertEqual(resolved["cursor"], "12")
        self.assertEqual(resolved["text"], "keep  spaces")
        self.assertEqual(resolved["items"][0]["p"], "C:\\Users\\Me\\Q4 Report.TXT")

    def test_literal_payload_reaches_the_tool_unchanged_through_execute(self):
        calls = []

        def fake_call(tool, args, grants=None):
            calls.append((tool, dict(args)))
            return {"ok": True, "content": "done", "path": "",
                    "exit_code": 0}

        steps = [
            {"tool": "code.list_directory", "args": {"path": "C:/tmp"}},
            {"tool": "code.run_command",
             "args": {"command": 'patch "Q4 Report.TXT" --Dry-Run {{step0.path}}'}},
        ]
        plan = agent._normalize_plan({"ok": True, "steps": steps}, "run it")
        with patch.object(agent.code_tools, "call_tool",
                          side_effect=fake_call):
            result = agent.execute_plan(plan, {}, confirmed=True)
        self.assertEqual(result.status, "completed")
        self.assertEqual(calls[1][1]["command"],
                         'patch "Q4 Report.TXT" --Dry-Run {{step0.path}}')


class PlanPayloadPreservationTests(unittest.TestCase):
    """Acceptance: mixed case, flags, URLs and indentation reach the tool."""

    def _folders(self):
        return {"username": "mayan", "home": "C:\\Users\\mayan",
                "desktop": "C:\\Users\\mayan\\Desktop",
                "documents": "C:\\Users\\mayan\\Documents",
                "downloads": "C:\\Users\\mayan\\Downloads"}

    def test_mixed_case_paths_and_uppercase_flags_survive_the_plan(self):
        plan = agent._heuristic_plan("read C:\\Temp\\Q4 Report.PDF", {})
        self.assertEqual(plan["steps"][0]["tool"], "code.read_file")
        self.assertEqual(plan["steps"][0]["args"]["path"],
                         "C:\\Temp\\Q4 Report.PDF")

        run = agent._heuristic_plan("run git -C MyRepo STATUS --PORCELAIN", {})
        self.assertEqual(run["steps"][0]["tool"], "code.run_command")
        self.assertEqual(run["steps"][0]["args"]["command"],
                         "git -C MyRepo STATUS --PORCELAIN")

    def test_a_command_is_not_rewritten_by_plan_normalization(self):
        # F12: the command line is payload. `_normalize_plan` used to run the
        # filesystem sanitizer over it, rewriting the user's own bytes.
        plan = {"ok": True, "summary": "s", "steps": [
            {"tool": "code.run_command",
             "args": {"command": "echo %USERPROFILE% <username> | sort"},
             "risk": "safe", "reason": "t"}]}
        with patch.object(agent, "_known_folders",
                          return_value=self._folders()):
            normalized = agent._normalize_plan(plan, "run it")
        self.assertEqual(normalized["steps"][0]["args"]["command"],
                         "echo %USERPROFILE% <username> | sort")

    def test_a_path_is_still_sanitized_and_redirected(self):
        # The designated path field keeps its placeholder handling.
        plan = {"ok": True, "summary": "s", "steps": [
            {"tool": "code.read_file",
             "args": {"path": "C:\\Users\\<username>\\notes.txt"},
             "risk": "safe", "reason": "t"}]}
        with patch.object(agent, "_known_folders",
                          return_value=self._folders()):
            normalized = agent._normalize_plan(plan, "read it")
        self.assertEqual(normalized["steps"][0]["args"]["path"],
                         "C:\\Users\\mayan\\notes.txt")

    def test_a_signed_url_survives_plan_normalization(self):
        url = "https://ex.test/a?X-Amz-Signature=AbC123&Expires=99"
        plan = {"ok": True, "summary": "s", "steps": [
            {"tool": "browser.open_url", "args": {"url": url},
             "risk": "safe", "reason": "t"}]}
        normalized = agent._normalize_plan(plan, "open it")
        self.assertEqual(normalized["steps"][0]["args"]["url"], url)

    def test_write_content_keeps_indentation_and_newlines(self):
        plan = agent._heuristic_plan(
            "create a file called notes.txt with line one\n  line two: True",
            {})
        self.assertEqual(plan["steps"][0]["tool"], "code.write_file")
        self.assertEqual(plan["steps"][0]["args"]["path"], "notes.txt")
        self.assertEqual(plan["steps"][0]["args"]["content"],
                         "line one\n  line two: True")


class WriteContentDelimiterTests(unittest.TestCase):
    """Acceptance: a content delimiter never matches inside a file name."""

    def test_delimiter_word_inside_a_file_name_is_part_of_the_name(self):
        plan = agent._heuristic_plan(
            "create a file named Q4 Report With Care.TXT", {})
        self.assertEqual(plan["steps"][0]["args"]["path"],
                         "Q4 Report With Care.TXT")
        self.assertEqual(plan["steps"][0]["args"]["content"], "")

    def test_one_complete_name_then_a_real_delimiter_still_splits(self):
        plan = agent._heuristic_plan(
            "create a file named Q4 Report With Care.TXT with the final "
            "numbers", {})
        self.assertEqual(plan["steps"][0]["args"]["path"],
                         "Q4 Report With Care.TXT")
        self.assertEqual(plan["steps"][0]["args"]["content"],
                         "the final numbers")

    def test_delimiter_words_inside_the_content_are_preserved(self):
        plan = agent._heuristic_plan(
            "create a file called deploy.log with step one as planned", {})
        self.assertEqual(plan["steps"][0]["args"]["path"], "deploy.log")
        self.assertEqual(plan["steps"][0]["args"]["content"],
                         "step one as planned")

    def test_extract_write_content_agrees_with_the_plan(self):
        self.assertEqual(
            agent._extract_write_content("create a file called a.txt with V1"),
            "V1")
        self.assertEqual(
            agent._extract_write_content(
                "create a file named Q4 Report With Care.TXT"),
            "")


if __name__ == "__main__":
    unittest.main()
