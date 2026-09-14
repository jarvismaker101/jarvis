"""F48, delivery side — provenance must survive to the overlay and the report.

The audit found the observation/verification metadata being built and then
dropped: ``routes.EvidenceItem`` declared only source/title/snippet (pydantic
silently discarded spans, provenance, uncertainty, lookup and timestamps), and
the research synthesis emitted prose with a provenance word but no verifiable
spans or corroboration accounting.

Pinned here: the API model keeps every F48 field (and unknown future ones), the
screen-answer endpoints round-trip them, and the research items carry the shared
claim contract.
"""

import unittest

from backend.api import routes
from backend.services import research_service as rs


class EvidenceModelTests(unittest.TestCase):
    def test_every_f48_field_survives_the_model(self):
        item = routes.EvidenceItem(
            source="screen", title="Docs", snippet="the claim",
            spans=[{"url": "https://x.test/a", "quote": "the claim"}],
            provenance="observed",
            corroboration={"level": "single", "independent": 1},
            uncertainty="single source — not independently verified",
            observed_at="2026-03-10T09:00:00+00:00",
            lookup={"query": "is this true?", "url": "https://x.test/a"},
            claim="the claim", query="is this true?", url="https://x.test/a",
        )
        dumped = item.model_dump()
        for key in ("spans", "provenance", "corroboration", "uncertainty",
                    "observed_at", "lookup", "claim", "query", "url"):
            self.assertIn(key, dumped)
        self.assertEqual(dumped["spans"][0]["quote"], "the claim")
        self.assertEqual(dumped["provenance"], "observed")

    def test_unknown_future_fields_are_not_dropped(self):
        item = routes.EvidenceItem(source="s", future_metric=0.75)
        self.assertEqual(item.model_dump()["future_metric"], 0.75)

    def test_screen_answer_round_trips_provenance(self):
        item = routes.EvidenceItem(source="screen", provenance="observed",
                                   spans=[{"quote": "q"}], claim="c")
        answer = routes.ScreenAnswer(tip="t", evidence=[item])
        restored = routes.ScreenAnswer(**answer.model_dump())
        self.assertEqual(restored.evidence[0].provenance, "observed")
        self.assertEqual(restored.evidence[0].spans, [{"quote": "q"}])
        self.assertEqual(restored.evidence[0].claim, "c")


class ResearchClaimTests(unittest.TestCase):
    def _item(self, url, summary, span):
        return {
            "result_title": "A page",
            "url": url,
            "summary": summary,
            "supporting_span": span,
            "text_excerpt": span,
            "provenance": "observed",
            "retrieved_at": "2026-03-10T09:00:00",
        }

    def test_items_get_spans_lookup_and_corroboration(self):
        items = [
            self._item("https://a.test/1",
                       "tanks hold 120 mm guns and thick armour plating",
                       "tanks hold 120 mm guns and thick armour plating"),
            self._item("https://b.test/2",
                       "tanks hold 120 mm guns and thick armour plating",
                       "tanks hold 120 mm guns and thick armour plating"),
        ]
        rs.attach_claim_provenance(items, "what guns do tanks hold")
        first = items[0]
        self.assertTrue(first["spans"], "a supported claim keeps a verbatim span")
        self.assertEqual(first["spans"][0]["quote"],
                         "tanks hold 120 mm guns and thick armour plating")
        self.assertEqual(first["lookup"]["query"], "what guns do tanks hold")
        self.assertTrue(first["observed_at"].endswith("+00:00"),
                        "timestamps must be timezone-aware (F48)")
        self.assertIn("level", first["corroboration"])
        self.assertEqual(first["claim"],
                         "tanks hold 120 mm guns and thick armour plating")

    def test_unsupported_claim_is_never_labelled_observed(self):
        items = [self._item("https://a.test/1", "the moon is made of cheese",
                            "completely unrelated boilerplate about shipping")]
        rs.attach_claim_provenance(items, "what is the moon made of")
        self.assertNotEqual(items[0]["provenance"], "observed",
                            "a strong label with no surviving span must degrade")

    def test_report_markdown_keeps_the_claim_data(self):
        items = [self._item("https://a.test/1",
                            "tanks hold 120 mm guns",
                            "tanks hold 120 mm guns")]
        rs.attach_claim_provenance(items, "what guns do tanks hold")
        markdown = rs.build_details_markdown(
            "what guns do tanks hold", "summary", [], items, [],
            {"visited": 1, "failed": 0})
        self.assertIn("120 mm guns", markdown)


if __name__ == "__main__":
    unittest.main()
