"""F04 — research notes and synthesis must answer the ACTUAL question.

Audit defect: "The actual query reaches note generation, with relevant-claim/
quotation/date/applicability instructions and some no-evidence filtering.
Output remains unvalidated prose, filtering recognizes narrow phrases, and
synthesis imposes release/upcoming-oriented headings on unrelated questions."

Correction: "Validate structured relevance/evidence and choose synthesis
structure from the actual question."

Acceptance: "Historical, troubleshooting, regional, and compatibility
questions reach every note call intact; irrelevant notes are excluded
regardless of wording; unrelated future-release framing is absent."

Every browser and LLM call here is faked — these tests never launch Chrome.
"""

import asyncio
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.services import research_service

HISTORICAL = "What happened to the Concorde fleet in 2000?"
TROUBLESHOOTING = ("Chrome keeps failing with error 0x80070005 — how do I fix "
                   "the profile?")
REGIONAL = "Is the Pixel 9 Pro available in Japan and South Korea?"
COMPATIBILITY = ("Does Windows 11 24H2 work with the Focusrite Scarlett 2i2, "
                 "and does it support ASIO drivers?")
FUTURE_RELEASE = "What's new in the next Claude release, and when will it ship?"

#: The four question kinds the acceptance criterion names by hand.
ORIGINAL_QUESTIONS = (
    ("historical", HISTORICAL),
    ("troubleshooting", TROUBLESHOOTING),
    ("regional", REGIONAL),
    ("compatibility", COMPATIBILITY),
)


def _research_task():
    """A MagicMock browser task — no Chrome is ever launched."""
    page = MagicMock()
    task = MagicMock()

    async def _new_page():
        return page

    async def _close_pages():
        return None

    task.new_page = _new_page
    task.close_pages = _close_pages
    return task, page


def _run_pipeline(question, summaries,
                  urls=("https://a.example/x", "https://b.example/y")):
    """Drive research_service._research_job for *question* with fakes.

    Returns everything the pipeline saw: the argument every note call was
    made with, the search URL, the progress messages and the collected items.
    """
    task, page = _research_task()
    results = [{"title": "Page " + u, "url": u, "snippet": ""} for u in urls]
    seen = {"queries": [], "progress": []}

    async def fake_fetch(_page, url):
        return {"title": "Fetched " + url, "text": "Body text for " + url,
                "youtube": False}

    def fake_summarize(query, title, url, text):
        seen["queries"].append(query)
        return summaries(url)

    async def job():
        return await research_service._research_job(
            task, question, len(results), None, time.monotonic() + 60,
            seen["progress"].append, None)

    with patch.object(research_service, "extract_brave_async",
                      new_callable=AsyncMock, return_value=results), \
         patch.object(research_service, "fetch_page_async",
                      side_effect=fake_fetch), \
         patch.object(research_service, "summarize_with_gemini",
                      side_effect=fake_summarize):
        seen["outcome"] = asyncio.run(job())

    seen["search_url"] = page.goto.call_args[0][0]
    return seen


def _fake_browser_run(coro_fn, task_id=None, timeout=None, profile_dir=None,
                      channel=None):
    """Stand-in for research_service._browser_run (F27 warm worker seam)."""
    task, _page = _research_task()
    return asyncio.run(coro_fn(task))


class F04ResearchQuestionTests(unittest.TestCase):
    """One question, threaded intact; only real evidence; question-shaped
    synthesis headings."""

    def setUp(self):
        research_service.clear_stop_request()

    def _synthesis_prompt(self, question):
        """Run consolidate_summaries for *question* and return its prompts."""
        items = [{
            "result_title": "Source",
            "url": "https://s.example/page",
            "summary": "Notes about %s with dates and quotations." % question,
            "provenance": research_service.PROVENANCE_OBSERVED,
            "uncertainty": "single source — not independently verified",
        }]
        with patch("backend.services.gemini_client.ask_gemini_chat") as gchat:
            gchat.return_value = {"choices": [{"message": {"content": "## X\nbody"}}]}
            out = research_service.consolidate_summaries(question, items)
        self.assertEqual(out, "## X\nbody",
                         "consolidate_summaries must still return a synthesis")
        messages = gchat.call_args.args[0]
        return messages[0]["content"], messages[1]["content"]

    # ── 1. the ORIGINAL question reaches every note call intact ───────────
    def test_original_question_reaches_every_note_call_verbatim(self):
        """Historical / troubleshooting / regional / compatibility questions
        arrive at each per-site note call exactly as asked."""
        for label, question in ORIGINAL_QUESTIONS:
            with self.subTest(kind=label):
                state = _run_pipeline(question,
                                      lambda url, q=question: "Notes about %s." % q)
                self.assertEqual(len(state["queries"]), 2,
                                 "one note call per fetched site")
                for seen in state["queries"]:
                    self.assertEqual(seen, question,
                                     "the note must get the original question, "
                                     "not a tidied-up topic")
                # the search itself is made with the same intact question
                self.assertIn(research_service.quote_plus(question),
                              state["search_url"])

    def test_a_long_question_is_not_truncated_anywhere(self):
        question = ("Please explain, in detail, why the 2011 Tohoku tsunami "
                    "warning sirens in Sendai were reported as late by some "
                    "residents, and what the official inquiry concluded about "
                    "the delay, including the exact minute the first warning "
                    "was broadcast and who signed off on the alert wording?")
        state = _run_pipeline(question,
                              lambda url: "Notes about the 2011 Tohoku tsunami "
                                          "warning sirens and the inquiry.")
        for seen in state["queries"]:
            self.assertEqual(seen, question)
            self.assertEqual(len(seen), len(question))

    def test_note_generation_prompt_carries_the_question_verbatim(self):
        with patch("backend.services.gemini_client.ask_gemini_chat") as gchat:
            gchat.return_value = {"choices": [{"message": {"content": "a note"}}]}
            research_service.summarize_with_gemini(
                COMPATIBILITY, "Title", "https://u.example", "page text")
        user_prompt = gchat.call_args.args[0][1]["content"]
        self.assertIn("QUESTION: " + COMPATIBILITY, user_prompt)

    def test_run_research_threads_the_question_into_notes_and_synthesis(self):
        note_calls, synthesis_calls = [], []

        def fake_summarize(query, title, url, text):
            note_calls.append(query)
            return ("Windows 11 24H2 works with the Scarlett 2i2 over USB-C.")

        # F04: the question must reach synthesis unchanged as well.
        def fake_consolidate(query, items):
            synthesis_calls.append(query)
            return "## Compatibility\nit works."

        async def fake_fetch(_page, url):
            return {"title": "Fetched", "text": "page text", "youtube": False}

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(research_service, "extract_brave_async",
                          new_callable=AsyncMock,
                          return_value=[{"title": "A", "url": "https://a.example/x",
                                         "snippet": ""}]), \
             patch.object(research_service, "fetch_page_async",
                          side_effect=fake_fetch), \
             patch.object(research_service, "summarize_with_gemini",
                          side_effect=fake_summarize), \
             patch.object(research_service, "consolidate_summaries",
                          side_effect=fake_consolidate), \
             patch.object(research_service, "build_spoken_summary",
                          return_value="spoken"), \
             patch.object(research_service, "_browser_run",
                          side_effect=_fake_browser_run):
            result = research_service.run_research(COMPATIBILITY,
                                                   reports_dir=tmp)

        self.assertEqual(note_calls, [COMPATIBILITY])
        self.assertEqual(synthesis_calls, [COMPATIBILITY])
        self.assertEqual(result["query"], COMPATIBILITY)

    # ── 2. irrelevant notes are excluded by MEANING, not by phrase ────────
    def test_off_topic_note_is_dropped_without_the_old_phrase(self):
        question = "Why does my Bosch dishwasher keep failing with a drain error?"
        notes = {
            "https://a.example/x": ("The filter basket clogs, the drain pump then "
                                    "reports error E24, and clearing the sump "
                                    "restores normal draining."),
            "https://b.example/y": ("Bananas are a good source of potassium and "
                                    "grow best in tropical climates."),
        }
        off_topic = notes["https://b.example/y"]
        # The old check only recognised these literal phrases — this note has
        # neither, so only a real relevance rule can catch it.
        self.assertNotIn("no relevant evidence", off_topic.lower())
        self.assertNotIn("no useful info", off_topic.lower())

        state = _run_pipeline(question, lambda url: notes[url])

        collected = state["outcome"]["collected"]
        self.assertEqual([c["url"] for c in collected], ["https://a.example/x"])
        self.assertTrue(any("not about this question" in m
                            for m in state["progress"]),
                        "the skip must be reported, not silent")

    def test_a_note_that_is_about_the_question_survives_other_wording(self):
        question = "Why does my Bosch dishwasher keep failing with a drain error?"
        note = ("The drain sump was blocked, so the machine could not pump the "
                "water away and stopped mid-cycle.")
        self.assertTrue(research_service.note_is_relevant(question, note))
        # …and the label says what it matched, so the call is checkable.
        self.assertIn("drain", research_service.relevance_label(question, note))

    def test_a_note_that_states_it_is_evidence_is_kept(self):
        note = ("This page is relevant evidence for the question: the reactor "
                "test began at 01:23 local time.")
        self.assertTrue(research_service.note_is_relevant(
            "What happened at Chernobyl?", note))

    def test_a_note_that_denies_relevance_is_not_kept_by_that_wording(self):
        note = ("The page provides no relevant evidence for the question; it is "
                "a shopping list.")
        self.assertFalse(research_service.note_is_relevant(
            "What happened at Chernobyl?", note))

    def test_explicitly_marked_evidence_and_empty_questions_survive(self):
        # The pinned search overview is an answer by construction.
        self.assertTrue(research_service.note_is_relevant(
            "What happened at Chernobyl?", "An unrelated sentence.",
            explicit_evidence=True))
        # A question with no meaningful terms carries nothing to judge on —
        # nothing is dropped because of it.
        self.assertTrue(research_service.note_is_relevant(
            "what is it", "An unrelated sentence."))

    def test_the_stronger_no_evidence_guard_is_still_in_place(self):
        self.assertTrue(research_service._is_no_evidence_note(
            "NO RELEVANT EVIDENCE"))
        state = _run_pipeline(
            "Why does my Bosch dishwasher keep failing with a drain error?",
            lambda url: ("No relevant evidence." if url.endswith("/x")
                         else "The drain error comes from a clogged filter."))
        self.assertEqual([c["url"] for c in state["outcome"]["collected"]],
                         ["https://b.example/y"])

    def test_synthesis_drops_off_topic_notes_that_reached_it(self):
        items = [
            {"result_title": "Good", "url": "https://u1",
             "summary": "The drain error E24 comes from a clogged filter basket."},
            {"result_title": "Off-topic", "url": "https://u2",
             "summary": "Bananas are rich in potassium."},
        ]
        with patch("backend.services.gemini_client.ask_gemini_chat") as gchat:
            gchat.return_value = {"choices": [{"message": {"content": "## What is happening\nx"}}]}
            research_service.consolidate_summaries(
                "Why does my Bosch dishwasher keep failing with a drain error?",
                items)
        entries = gchat.call_args.args[0][1]["content"]
        self.assertIn("clogged filter basket", entries)
        self.assertNotIn("Bananas", entries)

    # ── 3. synthesis structure comes from the question ───────────────────
    def test_historical_question_gets_history_sections_not_release_framing(self):
        self.assertEqual(research_service.detect_question_kind(HISTORICAL),
                         research_service.QUESTION_KIND_HISTORICAL)
        sections = research_service.question_sections(HISTORICAL)
        self.assertEqual(sections, ("Background", "What happened", "Aftermath"))

        system_prompt, user_prompt = self._synthesis_prompt(HISTORICAL)
        for heading in sections:
            self.assertIn("## " + heading, system_prompt)
        for heading in research_service.RELEASE_HEADINGS:
            self.assertNotIn(heading, system_prompt)
        self.assertIn("QUESTION: " + HISTORICAL, user_prompt)

    def test_troubleshooting_question_gets_causes_and_fixes_sections(self):
        self.assertEqual(research_service.detect_question_kind(TROUBLESHOOTING),
                         research_service.QUESTION_KIND_TROUBLESHOOTING)
        sections = research_service.question_sections(TROUBLESHOOTING)
        self.assertEqual(sections, ("What is happening", "Likely causes",
                                    "Fixes to try"))
        system_prompt, _user = self._synthesis_prompt(TROUBLESHOOTING)
        for heading in sections:
            self.assertIn("## " + heading, system_prompt)
        for heading in research_service.RELEASE_HEADINGS:
            self.assertNotIn(heading, system_prompt)

    def test_regional_question_gets_availability_sections(self):
        self.assertEqual(research_service.detect_question_kind(REGIONAL),
                         research_service.QUESTION_KIND_REGIONAL)
        sections = research_service.question_sections(REGIONAL)
        self.assertEqual(sections, ("Availability", "Regional differences"))
        system_prompt, _user = self._synthesis_prompt(REGIONAL)
        for heading in sections:
            self.assertIn("## " + heading, system_prompt)
        for heading in research_service.RELEASE_HEADINGS:
            self.assertNotIn(heading, system_prompt)

    def test_compatibility_question_gets_compatibility_sections(self):
        self.assertEqual(research_service.detect_question_kind(COMPATIBILITY),
                         research_service.QUESTION_KIND_COMPATIBILITY)
        sections = research_service.question_sections(COMPATIBILITY)
        self.assertEqual(sections, ("Compatibility", "Requirements",
                                    "Known limits"))
        system_prompt, _user = self._synthesis_prompt(COMPATIBILITY)
        for heading in sections:
            self.assertIn("## " + heading, system_prompt)
        for heading in research_service.RELEASE_HEADINGS:
            self.assertNotIn(heading, system_prompt)

    def test_comparison_and_history_by_year_are_detected(self):
        self.assertEqual(
            research_service.detect_question_kind(
                "Postgres vs MySQL for a small app — which should I pick?"),
            research_service.QUESTION_KIND_COMPARISON)
        self.assertEqual(
            research_service.detect_question_kind("What took place in 1989?"),
            research_service.QUESTION_KIND_HISTORICAL)

    def test_a_real_future_release_question_still_gets_release_framing(self):
        self.assertEqual(research_service.detect_question_kind(FUTURE_RELEASE),
                         research_service.QUESTION_KIND_RELEASE)
        sections = research_service.question_sections(FUTURE_RELEASE)
        self.assertEqual(sections, research_service.RELEASE_HEADINGS)
        system_prompt, _user = self._synthesis_prompt(FUTURE_RELEASE)
        for heading in research_service.RELEASE_HEADINGS:
            self.assertIn("## " + heading, system_prompt)

    def test_no_other_question_kind_may_use_release_headings(self):
        for kind in research_service.QUESTION_KINDS:
            if kind == research_service.QUESTION_KIND_RELEASE:
                continue
            with self.subTest(kind=kind):
                sections = research_service.synthesis_sections(kind)
                self.assertTrue(sections)
                for heading in research_service.RELEASE_HEADINGS:
                    self.assertNotIn(heading, sections)

    # ── 4. not enough evidence is said out loud, not papered over ────────
    def test_no_relevant_evidence_means_no_synthesis_at_all(self):
        items = [{
            "result_title": "Off-topic",
            "url": "https://u.example",
            "summary": "Bananas are rich in potassium.",
            "provenance": research_service.PROVENANCE_OBSERVED,
        }]
        with patch("backend.services.gemini_client.ask_gemini_chat") as gchat:
            out = research_service.consolidate_summaries(HISTORICAL, items)
        self.assertIsNone(out, "no evidence must not produce invented structure")
        gchat.assert_not_called()

    def test_synthesis_is_told_to_admit_thin_evidence(self):
        system_prompt, _user = self._synthesis_prompt(HISTORICAL)
        self.assertIn("not carry enough evidence", system_prompt)
        self.assertIn("instead of inventing facts", system_prompt)


if __name__ == "__main__":
    unittest.main()
