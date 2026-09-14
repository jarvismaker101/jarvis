"""F48 — Separate Observation From Verified Knowledge.

Acceptance clauses pinned by this module:

  * "Is this true verifies the displayed claim" — a deictic verification
    request is bound to the claim the screen actually DISPLAYS; with no
    displayed claim no lookup is fabricated.
  * "late-page claims cite actual spans" — a span must be verbatim, non
    boilerplate and actually about the claim, otherwise it is rejected as a
    "generic prefix" and the claim cannot stay labelled observed.
  * "mirrors/boilerplate/AI summaries do not independently corroborate" —
    syndicated copies, navigation text and AI summaries are counted
    separately from independent sources.
  * "contradictions affect answers and metadata survives round trips" — the
    corroboration verdict drives the uncertainty wording, and claim metadata
    (spans, lookup, timestamps, corroboration) survives JSON/evidence round
    trips unchanged.
  * aware timestamps: ``utc_now_iso`` names an unambiguous UTC instant.

No OS/vision boundary is touched: the capture and vision calls are patched.
"""

import json
import unittest
from unittest.mock import patch

from backend.services import provenance as prov
from backend.services import screen_analyzer


def _vision_result(tip, evidence):
    content = json.dumps({"tip": tip, "evidence": evidence,
                          "topic": "t", "show_images": False})
    return {"choices": [{"message": {"content": content}}],
            "grounding_links": []}


class TimestampAwarenessTests(unittest.TestCase):
    """F48: UTC-named timestamps must not be naive local time."""

    def test_utc_now_iso_is_aware(self):
        stamp = prov.utc_now_iso()
        self.assertIn("+00:00", stamp)
        parsed = prov.parse_timestamp(stamp)
        self.assertIsNotNone(parsed)
        self.assertIsNotNone(parsed.tzinfo)
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)

    def test_naive_timestamps_are_never_handed_back_naive(self):
        parsed = prov.parse_timestamp("2026-01-01T00:00:00")
        self.assertIsNotNone(parsed.tzinfo)
        self.assertEqual(parsed.utcoffset().total_seconds(), 0)

    def test_z_suffix_and_offsets_parse(self):
        for text in ("2026-01-01T00:00:00Z", "2026-01-01T05:30:00+05:30"):
            parsed = prov.parse_timestamp(text)
            self.assertIsNotNone(parsed)
            self.assertIsNotNone(parsed.tzinfo)
        self.assertIsNone(prov.parse_timestamp("not a timestamp"))
        self.assertIsNone(prov.parse_timestamp(""))

    def test_awareness_helper(self):
        self.assertTrue(prov.is_aware_timestamp(prov.utc_now_iso()))
        self.assertTrue(prov.is_aware_timestamp("2026-01-01T00:00:00Z"))
        self.assertFalse(prov.is_aware_timestamp("nonsense"))


class SpanValidationTests(unittest.TestCase):
    """Acceptance: late-page claims cite actual spans (no generic prefixes)."""

    def test_boilerplate_is_not_a_supporting_span(self):
        supported, reason = prov.spans_support(
            "Skip to main content", "The tower is 330 metres tall",
            "Skip to main content The tower is 330 metres tall.")
        self.assertFalse(supported)
        self.assertIn("boilerplate", reason)

    def test_generic_prefix_cannot_masquerade_as_support(self):
        page = ("Home About Contact Privacy Policy The tower is 330 metres "
                "tall according to the survey.")
        supported, reason = prov.spans_support(
            "Home About Contact Privacy Policy",
            "The Eiffel Tower is 330 metres tall", page)
        self.assertFalse(supported)
        self.assertIn("generic prefix", reason)

    def test_span_must_appear_verbatim_in_its_source(self):
        supported, reason = prov.spans_support(
            "The tower is 330 metres tall",
            "The Eiffel Tower is 330 metres tall",
            "The tower was measured at 330 metres by the survey team.")
        self.assertFalse(supported)
        self.assertIn("verbatim", reason)

    def test_a_real_supporting_span_is_accepted(self):
        supported, reason = prov.spans_support(
            "The tower is 330 metres tall",
            "How tall is the Eiffel Tower? The tower is 330 metres tall today.",
            "intro The tower is 330 metres tall today. outro")
        self.assertTrue(supported, reason)

    def test_build_claim_rejects_bad_spans_and_degrades_the_label(self):
        claim = prov.build_claim(
            "The Eiffel Tower is 330 metres tall",
            sources=[{"url": "https://a.com/x", "quote": "Skip to main content"}],
            provenance=prov.PROVENANCE_OBSERVED)
        self.assertEqual(claim.provenance, prov.PROVENANCE_INFERRED)
        self.assertEqual(claim.spans, ())
        self.assertTrue(claim.corroboration["rejected_spans"])

    def test_build_claim_keeps_a_validated_span(self):
        quote = "The Eiffel Tower is 330 metres tall"
        claim = prov.build_claim(
            "The Eiffel Tower is 330 metres tall",
            sources=[{"url": "https://a.com/x", "quote": quote,
                      "source_text": "news " + quote}])
        self.assertEqual(claim.provenance, prov.PROVENANCE_OBSERVED)
        self.assertEqual(len(claim.spans), 1)
        self.assertEqual(claim.spans[0].quote, quote)
        self.assertEqual(claim.spans[0].to_dict()["domain"], "a.com")


class IndependentCorroborationTests(unittest.TestCase):
    """Acceptance: mirrors/boilerplate/AI summaries do not corroborate."""

    def test_mirrors_of_one_article_are_not_two_sources(self):
        article = "The regulator announced a new rule on Tuesday."
        verdict = prov.independent_corroboration([
            {"url": "https://news-one.com/story", "quote": article},
            {"url": "https://mirror-two.net/story", "quote": article},
            {"url": "https://syndicate-three.org/story", "quote": article},
        ])
        self.assertFalse(verdict["corroborated"])
        self.assertEqual(verdict["independent"], 1)
        self.assertEqual(verdict["level"], prov.UNCERTAINTY_SINGLE_SOURCE)
        self.assertEqual(sorted(verdict["mirrors"]),
                         ["mirror-two.net", "syndicate-three.org"])

    def test_ai_summaries_never_corroborate(self):
        verdict = prov.independent_corroboration([
            {"url": "https://ai-summary.example/a", "quote": "It says so.",
             "provenance": prov.PROVENANCE_SECONDARY},
            {"url": "https://ai-summary.example/b", "quote": "It also says so.",
             "provenance": prov.PROVENANCE_SECONDARY},
        ])
        self.assertFalse(verdict["corroborated"])
        self.assertEqual(verdict["independent"], 0)
        self.assertEqual(verdict["level"], prov.UNCERTAINTY_UNVERIFIED)
        self.assertEqual(len(verdict["secondary_sources"]), 2)

    def test_boilerplate_does_not_corroborate(self):
        verdict = prov.independent_corroboration([
            {"url": "https://a.com", "quote": "Accept all cookies"},
            {"url": "https://b.com", "quote": "Subscribe to our newsletter"},
        ])
        self.assertEqual(verdict["independent"], 0)
        self.assertFalse(verdict["corroborated"])
        self.assertEqual(len(verdict["boilerplate"]), 2)

    def test_two_independent_primary_sources_corroborate(self):
        verdict = prov.independent_corroboration([
            {"url": "https://a.com/story",
             "quote": "The regulator announced a new rule on Tuesday."},
            {"url": "https://b.org/report",
             "quote": "Officials confirmed the rule takes effect in March."},
        ])
        self.assertTrue(verdict["corroborated"])
        self.assertEqual(verdict["independent"], 2)
        self.assertEqual(verdict["level"], prov.UNCERTAINTY_CORROBORATED)

    def test_same_host_twice_is_not_independent(self):
        verdict = prov.independent_corroboration([
            {"url": "https://a.com/one", "quote": "First distinct sentence here."},
            {"url": "https://www.a.com/two",
             "quote": "Second, entirely different sentence about it."},
        ])
        self.assertEqual(verdict["independent"], 1)
        self.assertFalse(verdict["corroborated"])


class ClaimRoundTripTests(unittest.TestCase):
    """Acceptance: contradictions affect answers and metadata survives."""

    def _claim(self):
        retrieved = prov.utc_now_iso()
        return prov.build_claim(
            "The Eiffel Tower is 330 metres tall",
            sources=[
                {"url": "https://a.com/x",
                 "quote": "The Eiffel Tower is 330 metres tall",
                 "publisher": "A News", "retrieved_at": retrieved},
                {"url": "https://b.org/y",
                 "quote": "Surveyors measured the Eiffel Tower at 330 metres",
                 "publisher": "B Report", "retrieved_at": retrieved},
            ],
            lookup={"query": "eiffel tower height", "url": "https://s.example/q",
                    "provider": "quick_search", "retrieved_at": retrieved},
            observed_at=prov.utc_now_iso(),
            relevance=0.87)

    def test_claim_dict_round_trip_is_lossless(self):
        claim = self._claim()
        data = json.loads(json.dumps(claim.to_dict()))
        restored = prov.Claim.from_dict(data)
        self.assertEqual(claim, restored)
        self.assertEqual(len(restored.spans), 2)
        self.assertEqual(restored.lookup["provider"], "quick_search")
        self.assertEqual(restored.corroboration["level"],
                         prov.UNCERTAINTY_CORROBORATED)
        self.assertEqual(restored.relevance, 0.87)
        self.assertTrue(prov.is_aware_timestamp(restored.observed_at))

    def test_evidence_round_trip_keeps_every_field(self):
        claim = self._claim()
        evidence = json.loads(json.dumps(
            prov.claim_to_evidence(claim, title="Height report")))
        self.assertEqual(evidence["result_title"], "Height report")
        self.assertEqual(evidence["provenance"], prov.PROVENANCE_OBSERVED)
        self.assertEqual(evidence["uncertainty"],
                         prov.UNCERTAINTY_CORROBORATED)
        self.assertEqual(evidence["lookup"]["query"], "eiffel tower height")
        self.assertEqual(evidence["spans"][0]["url"], "https://a.com/x")
        restored = prov.evidence_to_claim(evidence)
        self.assertEqual(restored, claim)

    def test_contradiction_changes_the_reported_uncertainty(self):
        text = "The Eiffel Tower is 330 metres tall"
        single = prov.build_claim(text, sources=[
            {"url": "https://a.com/x", "quote": text}])
        self.assertEqual(single.uncertainty, prov.UNCERTAINTY_SINGLE_SOURCE)
        mirror = prov.build_claim(text, sources=[
            {"url": "https://a.com/x", "quote": text},
            {"url": "https://mirror.net/x", "quote": text}])
        self.assertEqual(mirror.uncertainty, prov.UNCERTAINTY_SINGLE_SOURCE)
        self.assertFalse(mirror.corroboration["corroborated"])
        independent = prov.build_claim(text, sources=[
            {"url": "https://a.com/x", "quote": text},
            {"url": "https://b.org/y",
             "quote": "Surveyors measured the Eiffel Tower at 330 metres"}])
        self.assertEqual(independent.uncertainty, prov.UNCERTAINTY_CORROBORATED)

    def test_unknown_label_degrades_to_inferred_in_the_contract(self):
        claim = prov.build_claim("A claim", sources=[],
                                 provenance="definitely true")
        self.assertEqual(claim.provenance, prov.PROVENANCE_INFERRED)
        self.assertEqual(claim.uncertainty, prov.UNCERTAINTY_UNVERIFIED)


class DeicticVerificationTests(unittest.TestCase):
    """Acceptance: "Is this true" verifies the displayed claim."""

    def setUp(self):
        self.capture = {"image_data_url": "data:image/png;base64,abc",
                        "region": None}

    def _analyze(self, question, tip, evidence, overview="Checked out."):
        seen = {}

        def _overview(query):
            seen["query"] = query
            return overview

        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=self.capture), \
             patch.object(screen_analyzer, "is_region_question",
                          return_value=False), \
             patch.object(screen_analyzer, "_ask_screen_vision_cascade",
                          return_value=_vision_result(tip, evidence)), \
             patch("backend.services.quick_search.fetch_ai_overview_text",
                   side_effect=_overview), \
             patch("backend.services.quick_search.google_search_url",
                   return_value="https://search.example/q=claim"):
            result = screen_analyzer.analyze_screen(question)
        return result, seen

    def test_is_this_true_verifies_the_displayed_claim(self):
        tip = "The Eiffel Tower is 330 metres tall."
        result, seen = self._analyze("is this true?", tip, [])
        checked = result["evidence"][-1]
        self.assertEqual(checked["provenance"], "externally_checked")
        self.assertEqual(checked["claim"], tip)
        self.assertIn("330 metres", seen["query"])
        self.assertNotEqual(seen["query"].strip().lower(), "is this true?")
        self.assertEqual(checked["lookup"]["query"], seen["query"])
        self.assertEqual(checked["spans"][0]["quote"], "Checked out.")
        # An AI summary of other pages is not independent corroboration.
        self.assertFalse(checked["corroboration"]["corroborated"])
        self.assertEqual(checked["corroboration"]["independent"], 0)
        self.assertTrue(prov.is_aware_timestamp(checked["retrieved_at"]))

    def test_claim_includes_the_observed_evidence_behind_the_answer(self):
        result, seen = self._analyze(
            "is this accurate?",
            "The tower is tall.",
            [{"source": "Wikipedia", "title": "Tower",
              "snippet": "The tower is 330 metres tall.", "provenance": "observed"}])
        checked = result["evidence"][-1]
        self.assertIn("330 metres", checked["claim"])
        self.assertIn("330 metres", seen["query"])

    def test_an_explicit_target_is_searched_together_with_the_claim(self):
        result, seen = self._analyze("verify the height of the Eiffel Tower",
                                     "The header says 330 m.", [])
        self.assertIn("height of the Eiffel Tower", seen["query"])
        self.assertIn("330 m", seen["query"])
        self.assertEqual(result["evidence"][-1]["provenance"],
                         "externally_checked")

    def test_no_displayed_claim_means_no_fabricated_lookup(self):
        with patch("backend.services.quick_search.fetch_ai_overview_text") as fetch:
            self.assertIsNone(screen_analyzer._external_check("is this true?"))
        fetch.assert_not_called()

    def test_verification_question_without_a_checkable_claim_appends_nothing(self):
        # The model returned no tip at all; the fallback sentence is not a
        # claim about the world the screen could verify — but the lookup is
        # still bound to whatever was displayed, never to "verify this".
        seen = {}

        def _overview(query):
            seen["query"] = query
            return ""

        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=self.capture), \
             patch.object(screen_analyzer, "is_region_question",
                          return_value=False), \
             patch.object(screen_analyzer, "_ask_screen_vision_cascade",
                          return_value=_vision_result("", [])), \
             patch("backend.services.quick_search.fetch_ai_overview_text",
                   side_effect=_overview):
            result = screen_analyzer.analyze_screen("verify this")
        self.assertFalse(any(e["provenance"] == "externally_checked"
                             for e in result["evidence"]))
        self.assertNotEqual(seen.get("query", "").strip().lower(), "verify this")


class ScreenObservationTests(unittest.TestCase):
    """F48: observation and inference stay separated on the screen path."""

    def setUp(self):
        self.capture = {"image_data_url": "data:image/png;base64,abc",
                        "region": None}

    def _analyze(self, question, evidence):
        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=self.capture), \
             patch.object(screen_analyzer, "is_region_question",
                          return_value=False), \
             patch.object(screen_analyzer, "_ask_screen_vision_cascade",
                          return_value=_vision_result("tip", evidence)), \
             patch.object(screen_analyzer, "_external_check",
                          return_value=None):
            return screen_analyzer.analyze_screen(question)

    def test_unlabelled_evidence_is_inferred_not_observed(self):
        result = self._analyze("what is on my screen",
                               [{"source": "x", "title": "y", "snippet": "z"}])
        item = result["evidence"][0]
        self.assertEqual(item["provenance"], "inferred")
        self.assertIn("inference", item["uncertainty"])

    def test_invented_strong_label_is_rejected(self):
        result = self._analyze("what is on my screen",
                               [{"source": "x", "title": "y", "snippet": "z",
                                 "provenance": "externally_checked"}])
        self.assertEqual(result["evidence"][0]["provenance"], "inferred")

    def test_observed_evidence_cites_its_span(self):
        result = self._analyze(
            "what is on my screen",
            [{"source": "VS Code", "title": "Editor",
              "snippet": "def total():", "provenance": "observed"}])
        item = result["evidence"][0]
        self.assertEqual(item["provenance"], "observed")
        self.assertEqual(item["spans"][0]["quote"], "def total():")
        self.assertEqual(item["spans"][0]["selector"], "screen")
        self.assertTrue(prov.is_aware_timestamp(item["observed_at"]))

    def test_inferred_evidence_cites_no_span(self):
        result = self._analyze(
            "what is on my screen",
            [{"source": "model", "title": "Guess", "snippet": "Probably Python.",
              "provenance": "inferred"}])
        self.assertNotIn("spans", result["evidence"][0])

    def test_observed_at_is_aware_and_shared_with_the_evidence(self):
        result = self._analyze(
            "what is on my screen",
            [{"source": "VS Code", "title": "Editor", "snippet": "x",
              "provenance": "observed"}])
        self.assertTrue(prov.is_aware_timestamp(result["observed_at"]))
        self.assertEqual(result["evidence"][0]["observed_at"],
                         result["observed_at"])


if __name__ == "__main__":
    unittest.main()
