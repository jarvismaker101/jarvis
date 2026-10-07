"""Rank 4 — PROVE IT BEFORE CLAIMING IT.

The per-tool postcondition proofs: a file write is only done when the file
on disk still matches what was written; a folder create is only done when
the directory exists; a media playback goal is only done with a PLAYING
verdict from the task's own verify_playing probe.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from backend.services import browser_agent
from backend.services import proof as proof_layer
from backend.services.task_agent import agent


class ProofModuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "note.txt")

    def _write(self, text):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def test_matching_content_is_proved(self):
        self._write("hello world")
        verdict = proof_layer.prove_step(
            "code.write_file",
            {"path": self.path, "content": "hello world"},
            {"ok": True, "path": self.path})
        self.assertEqual(verdict["state"], proof_layer.PROVED)
        self.assertIn("content verified", verdict["evidence"])

    def test_content_mismatch_is_failed(self):
        self._write("something else")
        verdict = proof_layer.prove_step(
            "code.write_file",
            {"path": self.path, "content": "hello world"},
            {"ok": True, "path": self.path})
        self.assertEqual(verdict["state"], proof_layer.FAILED)
        self.assertIn("does not match", verdict["evidence"])

    def test_missing_file_is_failed(self):
        verdict = proof_layer.prove_step(
            "code.write_file",
            {"path": self.path, "content": "x"},
            {"ok": True, "path": self.path})
        self.assertEqual(verdict["state"], proof_layer.FAILED)
        self.assertIn("missing", verdict["evidence"])

    def test_after_hash_is_used_when_content_absent(self):
        self._write("payload")
        after = proof_layer.code_grants.content_hash(path=self.path)
        verdict = proof_layer.prove_step(
            "code.write_file", {"path": self.path},
            {"ok": True, "path": self.path, "after_hash": after})
        self.assertEqual(verdict["state"], proof_layer.PROVED)

    def test_folder_proved_only_when_directory(self):
        ok = proof_layer.prove_step(
            "code.create_folder", {"path": self.tmp.name},
            {"ok": True, "path": self.tmp.name})
        self.assertEqual(ok["state"], proof_layer.PROVED)
        self._write("not a folder")
        bad = proof_layer.prove_step(
            "code.create_folder", {"path": self.path},
            {"ok": True, "path": self.path})
        self.assertEqual(bad["state"], proof_layer.FAILED)

    def test_patch_present_file_is_proved(self):
        self._write("patched")
        verdict = proof_layer.prove_step(
            "code.apply_patch", {"path": self.path},
            {"ok": True, "path": self.path})
        self.assertEqual(verdict["state"], proof_layer.PROVED)

    def test_unproofable_tool_returns_none(self):
        self.assertIsNone(proof_layer.prove_step(
            "code.run_command", {"command": "dir"}, {"ok": True}))


class StepVerdictTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "note.txt")

    def _write(self, text):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(text)

    def test_failure_verdict_catches_content_mismatch(self):
        self._write("actual")
        failed, reason = agent._step_failure_verdict(
            "code.write_file", {"path": self.path, "content": "expected"},
            "Wrote file.", {"ok": True, "path": self.path})
        self.assertTrue(failed)
        self.assertIn("postcondition failed", reason)
        self.assertIn("does not match", reason)

    def test_failure_verdict_passes_matching_write(self):
        self._write("expected")
        failed, reason = agent._step_failure_verdict(
            "code.write_file", {"path": self.path, "content": "expected"},
            "Wrote file.", {"ok": True, "path": self.path})
        self.assertFalse(failed)
        self.assertEqual(reason, "")


class ConfirmedStepsProofGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "note.txt")

    def _run_with_world(self, world_text, claimed_text):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(world_text)
        plan = {"steps": [{
            "tool": "code.write_file",
            "args": {"path": self.path, "content": claimed_text},
        }]}
        with patch.object(
                agent, "_execute_step_structured",
                return_value=("Wrote the file.", {"ok": True,
                                                  "path": self.path})):
            return agent._run_confirmed_steps(plan, {})

    def test_lying_ok_step_cannot_be_completed(self):
        result = self._run_with_world("the real content", "what I claimed")
        self.assertEqual(result.status, "partial")
        self.assertTrue(result.summary.startswith("Partly done, sir."))
        self.assertIn("does not match", result.summary)
        self.assertFalse(result.verification, "no proof may be advertised")

    def test_matching_write_is_completed_with_proof(self):
        result = self._run_with_world("faithful", "faithful")
        self.assertEqual(result.status, "completed")
        self.assertTrue(any("verified on disk" in line
                            for line in result.verification))

    def test_empty_write_claim_fails(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("")
        plan = {"steps": [{
            "tool": "code.write_file",
            "args": {"path": self.path, "content": "expected"},
        }]}
        with patch.object(
                agent, "_execute_step_structured",
                return_value=("Wrote the file.", {"ok": True,
                                                  "path": self.path})):
            result = agent._run_confirmed_steps(plan, {})
        self.assertEqual(result.status, "partial")


class MediaProofGateTests(unittest.TestCase):
    def test_play_goal_without_observation_needs_hedge(self):
        verdict = browser_agent._unproven_media_verdict(
            "play lofi beats on youtube", {})
        self.assertIsNotNone(verdict)
        self.assertIn("could not confirm", verdict[0])

    def test_playing_verdict_proves_completion(self):
        session = {"_media_verdict":
                   "verify_playing: PLAYING - media time advanced 1.00s -> "
                   "2.20s over 1.2s (player 0, paused=False)."}
        self.assertIsNone(browser_agent._unproven_media_verdict(
            "play lofi beats on youtube", session))

    def test_paused_verdict_downgrades(self):
        session = {"_media_verdict": "verify_playing: PAUSED - the media "
                                     "element is paused."}
        verdict = browser_agent._unproven_media_verdict(
            "play the video", session)
        self.assertIsNotNone(verdict)
        self.assertIn("could not confirm", verdict[0])

    def test_non_media_goal_is_not_gated(self):
        self.assertIsNone(browser_agent._unproven_media_verdict(
            "open gmail and send the invoice", {}))


if __name__ == "__main__":
    unittest.main()
