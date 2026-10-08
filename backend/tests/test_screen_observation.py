"""RANK 1 of the external redesign: capture ONCE, keep the observation, and
let the VISION model resolve which on-screen item the user is pointing at.

Covers:
  * the observation store (caps, TTLs, image on the newest two only);
  * the item-inventory cleaner and the target grader (report 1.6 rules);
  * identify_on_screen / reidentify / the analyze_screen wrapper mapping;
  * the chain wiring: the research step re-reads the SAME stored image
    (no re-capture) and asks instead of silently searching a "likely".
"""
import json
import time
import unittest
from unittest.mock import patch

from backend.core import brain
from backend.services import context_state, screen_analyzer


def _vision_content(payload):
    return {"choices": [{"message": {"content": json.dumps(payload)}}],
            "grounding_links": []}


def _capture(url="data:image/png;base64,AAAA"):
    return {"image_data_url": url, "region": None}


def _item(item_id="i1", label="Karna Vs Arjun Ko Secret Message",
          primary=True, **kwargs):
    return context_state.Item(id=item_id, label=label, primary=primary,
                              **kwargs)


class ObservationStoreTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.store = context_state.ObservationStore(clock=lambda: self.now)

    def _obs(self, obs_id, image=""):
        return context_state.Observation(id=obs_id, at=self.now,
                                         image_data_url=image)

    def test_add_get_and_next_id(self):
        obs = self.store.add(self._obs("O1"))
        self.assertIs(self.store.get("O1"), obs)
        self.assertIsNone(self.store.get(""))
        self.assertIsNone(self.store.get("O9"))
        self.assertEqual(self.store.next_id(), "O1")
        self.assertEqual(self.store.next_id(), "O2")

    def test_only_the_last_three_metadata_records_are_kept(self):
        for idx in range(1, 6):
            self.store.add(self._obs("O%d" % idx))
        self.assertIsNone(self.store.get("O1"))
        self.assertIsNone(self.store.get("O2"))
        self.assertIsNotNone(self.store.get("O3"))
        self.assertIsNotNone(self.store.get("O5"))

    def test_image_kept_on_the_newest_two_only(self):
        for idx in range(1, 4):
            self.store.add(self._obs("O%d" % idx, image="img%d" % idx))
        self.assertEqual(self.store.get("O1").image_data_url, "")
        self.assertEqual(self.store.get("O2").image_data_url, "img2")
        self.assertEqual(self.store.get("O3").image_data_url, "img3")

    def test_image_expires_before_metadata(self):
        self.store.add(self._obs("O1", image="img"))
        self.now += context_state.IMAGE_TTL_S + 1
        obs = self.store.get("O1")
        self.assertIsNotNone(obs)
        self.assertEqual(obs.image_data_url, "")

    def test_metadata_expires(self):
        self.store.add(self._obs("O1", image="img"))
        self.now += context_state.OBS_TTL_S + 1
        self.assertIsNone(self.store.get("O1"))
        self.assertIsNone(self.store.latest())

    def test_latest_with_image_and_max_age(self):
        self.store.add(self._obs("O1", image="img"))
        self.now += 10
        older = self.store.add(context_state.Observation(
            id="O2", at=self.now, image_data_url=""))
        self.assertIs(self.store.latest(), older)
        self.assertIs(self.store.latest(with_image=True),
                      self.store.get("O1"))
        self.assertIsNone(self.store.latest(max_age=5, with_image=True))


class CleanItemsTests(unittest.TestCase):
    def test_caps_at_six_and_sanitizes(self):
        raw = [{"id": "i%d" % n, "label": "Item\x00 %d\n" % n}
               for n in range(1, 9)]
        items = screen_analyzer._clean_items(raw)
        self.assertEqual(len(items), 6)
        self.assertEqual(items[0].label, "Item 1")
        self.assertEqual(items[0].id, "i1")

    def test_duplicate_ids_are_disambiguated(self):
        items = screen_analyzer._clean_items([
            {"id": "i1", "label": "a"}, {"id": "i1", "label": "b"}])
        self.assertNotEqual(items[0].id, items[1].id)

    def test_empty_text_label_is_dropped_and_description_kept(self):
        items = screen_analyzer._clean_items([
            {"id": "i1", "label": ""},
            {"id": "i2", "label": "a red thumbnail", "label_is_text": False},
        ])
        self.assertEqual([item.id for item in items], ["i2"])
        self.assertFalse(items[0].label_is_text)

    def test_bad_bbox_is_zeroed_and_ranges_clamped(self):
        items = screen_analyzer._clean_items([
            {"id": "i1", "label": "a", "bbox": "nope"},
            {"id": "i2", "label": "b", "bbox": [-50, 10, 2000, 70]},
        ])
        self.assertEqual(items[0].bbox, (0, 0, 0, 0))
        self.assertEqual(items[1].bbox, (0.0, 10.0, 1000.0, 70.0))


class GradeTargetTests(unittest.TestCase):
    def _obs(self, **kwargs):
        return context_state.Observation(
            utterance="research about it and tell me",
            items=[_item(**kwargs)],
        )

    def test_exact_single_binding(self):
        grade = screen_analyzer.grade_target(
            self._obs(), {"target_ids": ["i1"], "n_fit": 1,
                          "match": "exact", "why": "the playing video"})
        self.assertEqual(grade["match"], "exact")
        self.assertEqual(grade["ids"], ["i1"])
        self.assertFalse(grade["ambiguous"])

    def test_truncated_label_caps_at_likely(self):
        grade = screen_analyzer.grade_target(
            self._obs(truncated=True),
            {"target_ids": ["i1"], "n_fit": 1, "match": "exact",
             "why": "top card"})
        self.assertEqual(grade["match"], "likely")
        self.assertIn("truncated_label", grade["capped"])

    def test_exact_without_a_why_caps_at_likely(self):
        grade = screen_analyzer.grade_target(
            self._obs(), {"target_ids": ["i1"], "n_fit": 1,
                          "match": "exact", "why": ""})
        self.assertEqual(grade["match"], "likely")
        self.assertIn("no_why", grade["capped"])

    def test_unknown_id_is_none(self):
        grade = screen_analyzer.grade_target(
            self._obs(), {"target_ids": ["i9"], "n_fit": 1,
                          "match": "exact", "why": "x"})
        self.assertEqual(grade["match"], "none")
        self.assertEqual(grade["ids"], [])

    def test_scaffolded_label_is_rejected(self):
        grade = screen_analyzer.grade_target(
            self._obs(label="research about it and tell me"),
            {"target_ids": ["i1"], "n_fit": 1, "match": "exact",
             "why": "x"})
        self.assertEqual(grade["match"], "none")
        self.assertIn("scaffolded_label", grade["capped"])

    def test_named_word_missing_caps_at_likely(self):
        obs = context_state.Observation(
            utterance="research it", items=[_item(label="Some Other Video")])
        grade = screen_analyzer.grade_target(
            obs, {"target_ids": ["i1"], "n_fit": 1, "match": "exact",
                  "why": "x"},
            target_text="the secret message")
        self.assertEqual(grade["match"], "likely")
        self.assertIn("named_word_missing", grade["capped"])

    def test_two_plausible_items_flag_ambiguity(self):
        grade = screen_analyzer.grade_target(
            self._obs(), {"target_ids": ["i1"], "n_fit": 2,
                          "match": "exact", "why": "x"})
        self.assertTrue(grade["ambiguous"])


class IdentifyTests(unittest.TestCase):
    def setUp(self):
        context_state.OBSERVATIONS.clear()

    def tearDown(self):
        context_state.OBSERVATIONS.clear()

    PAYLOAD = {
        "tip": "Yes sir, that is the Karna short.",
        "items": [
            {"id": "i1", "label": "Karna Vs Arjun Ko Secret Message",
             "kind": "video", "creator": "Some Channel",
             "bbox": [10, 10, 900, 700], "primary": True},
            {"id": "i2", "label": "QTI The Secret Betr",
             "truncated": True, "bbox": [900, 10, 1000, 300]},
        ],
        "target_ids": ["i1"], "n_fit": 1, "match": "exact",
        "why": "the playing shorts video",
    }

    def test_identify_stores_the_observation_and_grades(self):
        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=_capture()), \
             patch.object(screen_analyzer, "_ask_screen_vision_cascade",
                          return_value=_vision_content(self.PAYLOAD)):
            obs, grade = screen_analyzer.identify_on_screen(
                "research about it", target_text="that secret message")
        self.assertIsNotNone(obs)
        self.assertIs(context_state.OBSERVATIONS.get(obs.id), obs)
        self.assertEqual(grade["match"], "exact")
        self.assertEqual(grade["ids"], ["i1"])
        self.assertEqual(obs.items[0].creator, "Some Channel")
        self.assertTrue(obs.items[1].truncated)

    def test_identify_never_captures_when_an_image_is_given(self):
        with patch.object(screen_analyzer, "capture_primary_screen",
                          side_effect=AssertionError("must not re-capture")), \
             patch.object(screen_analyzer, "_ask_screen_vision_cascade",
                          return_value=_vision_content(self.PAYLOAD)):
            obs, _grade = screen_analyzer.identify_on_screen(
                "research about it", image=_capture("data:image/png;base64,SEED"))
        self.assertEqual(obs.image_data_url, "data:image/png;base64,SEED")

    def test_analyze_screen_maps_the_primary_item_and_observation(self):
        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=_capture()), \
             patch.object(screen_analyzer, "is_region_question",
                          return_value=False), \
             patch.object(screen_analyzer, "_ask_screen_vision_cascade",
                          return_value=_vision_content(self.PAYLOAD)):
            result = screen_analyzer.analyze_screen("what is on my screen")
        self.assertEqual(result["topic"], "Karna Vs Arjun Ko Secret Message")
        self.assertEqual(result["creator"], "Some Channel")
        self.assertEqual(result["tip"], "Yes sir, that is the Karna short.")
        self.assertTrue(result["observation_id"])
        self.assertIsNotNone(
            context_state.OBSERVATIONS.get(result["observation_id"]))

    def test_analyze_screen_keeps_the_legacy_tip_fallback(self):
        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=_capture()), \
             patch.object(screen_analyzer, "is_region_question",
                          return_value=False), \
             patch.object(screen_analyzer, "_ask_screen_vision_cascade",
                          return_value=_vision_content(
                              {"tip": "legacy tip", "topic": "editor"})):
            result = screen_analyzer.analyze_screen("what is on my screen")
        self.assertEqual(result["tip"], "legacy tip")
        self.assertEqual(result["topic"], "editor")

    def test_reidentify_reuses_the_stored_image(self):
        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=_capture("data:image/png;base64,ORIG")), \
             patch.object(screen_analyzer, "_ask_screen_vision_cascade",
                          return_value=_vision_content(self.PAYLOAD)):
            obs, _grade = screen_analyzer.identify_on_screen("look at it")
        payload = dict(self.PAYLOAD, target_ids=["i2"], match="likely")
        with patch.object(screen_analyzer, "capture_primary_screen",
                          side_effect=AssertionError("no new capture")), \
             patch.object(screen_analyzer, "_ask_screen_vision_cascade",
                          return_value=_vision_content(payload)) as cascade:
            fresh, grade = screen_analyzer.reidentify(
                obs, "no not that, the image to the left", rejected=["i1"])
        self.assertIsNotNone(fresh)
        self.assertEqual(cascade.call_args[0][1], "data:image/png;base64,ORIG")
        self.assertIn("i1", fresh.rejected)
        self.assertEqual(grade["match"], "likely")

    def test_capture_failure_is_reported_not_guessed(self):
        with patch.object(screen_analyzer, "capture_primary_screen",
                          side_effect=RuntimeError("no mss")):
            result = screen_analyzer.analyze_screen("what is on my screen")
        self.assertIn("couldn't capture your screen", result["tip"])
        self.assertIsNone(context_state.OBSERVATIONS.latest())


class BrainWiringTests(unittest.TestCase):
    LABEL = "Karna Vs Arjun Ko Secret Message"

    def setUp(self):
        context_state.OBSERVATIONS.clear()
        brain._clear_pending_screen_clarify()

    def tearDown(self):
        context_state.OBSERVATIONS.clear()
        brain._clear_pending_screen_clarify()

    def _seed(self):
        obs = context_state.OBSERVATIONS.add(context_state.Observation(
            id="O1", at=time.time(),
            image_data_url="data:image/png;base64,SEED",
            mode="full", items=[_item(label="generic topic")]))
        return obs

    def _fresh(self):
        return context_state.Observation(id="O2", items=[_item(label=self.LABEL)])

    def test_screen_step_carries_the_observation_id(self):
        with patch.object(brain, "analyze_screen", return_value={
                "tip": "I looked.", "topic": "topic",
                "observation_id": "O7"}):
            result = brain._mi_screen_step({"text": "what is this"})
        self.assertEqual(result["output_obs_id"], "O7")

    def test_vision_target_reuses_the_stored_observation_image(self):
        self._seed()
        grade = {"ids": ["i1"], "n_fit": 1, "match": "exact",
                 "ambiguous": False, "capped": [], "labels": [self.LABEL]}
        with patch.object(brain, "analyze_screen",
                          side_effect=AssertionError("must not re-capture")), \
             patch.object(brain, "identify_on_screen",
                          return_value=(self._fresh(), grade)) as ident:
            label, returned = brain._mi_vision_screen_target(
                "research about it", obs_id="O1")
        self.assertEqual(label, self.LABEL)
        self.assertEqual(returned["match"], "exact")
        sent = ident.call_args.kwargs
        self.assertEqual(sent["image"]["image_data_url"],
                         "data:image/png;base64,SEED")

    def test_research_step_searches_the_graded_binding(self):
        self._seed()
        prior = {"kind": "screen", "status": "ok",
                 "output_query": "YouTube live chat message",
                 "output_content": "A chat panel about a hidden promo code.",
                 "output_obs_id": "O1"}
        grade = {"ids": ["i1"], "n_fit": 1, "match": "exact",
                 "ambiguous": False, "capped": [], "labels": [self.LABEL]}
        searches = []
        with patch.object(brain, "identify_on_screen",
                          return_value=(self._fresh(), grade)), \
             patch.object(brain, "run_quick_search",
                          side_effect=lambda q, *a, **k:
                          searches.append(q) or {"query": q,
                                                 "spoken_summary": "s"}):
            result = brain._mi_research_step(
                {"kind": "research", "consumes": [0],
                 "text": "research on the internet about it and tell me"},
                [prior])
        self.assertEqual(searches, [self.LABEL])
        self.assertEqual(result["status"], "ok")

    def test_research_step_asks_when_the_binding_is_only_likely(self):
        self._seed()
        prior = {"kind": "screen", "status": "ok",
                 "output_query": "YouTube live chat message",
                 "output_content": "A chat panel about a hidden promo code.",
                 "output_obs_id": "O1"}
        grade = {"ids": ["i1"], "n_fit": 1, "match": "likely",
                 "ambiguous": False, "capped": [], "labels": [self.LABEL]}
        with patch.object(brain, "identify_on_screen",
                          return_value=(self._fresh(), grade)), \
             patch.object(brain, "_arm_confirmation") as arm, \
             patch.object(brain, "run_quick_search",
                          side_effect=AssertionError("must ask first")):
            result = brain._mi_research_step(
                {"kind": "research", "consumes": [0],
                 "text": "research on the internet about it and tell me"},
                [prior])
        self.assertEqual(result["status"], "asked")
        self.assertEqual(arm.call_args.kwargs.get("query"), self.LABEL)
        pending = brain._get_pending_screen_clarify()
        self.assertEqual(pending.get("obs_id"), "O1")

    ASPECT_LABEL = "I Tested Nano banana 2.1 vs GPT 2.5 (AI Image)"

    def test_possessive_aspect_composes_instead_of_asking(self):
        context_state.OBSERVATIONS.add(context_state.Observation(
            id="O1", at=time.time(),
            utterance="look at my screen",
            image_data_url="data:image/png;base64,SEED",
            items=[_item(label=self.ASPECT_LABEL)]))
        prior = {"kind": "screen", "status": "ok",
                 "output_query": self.ASPECT_LABEL,
                 "output_content": "A video comparing image models.",
                 "output_obs_id": "O1"}
        searches = []
        with patch.object(brain, "analyze_screen",
                          side_effect=AssertionError("no re-analysis")), \
             patch.object(brain, "identify_on_screen",
                          side_effect=AssertionError("no second vision call")), \
             patch.object(brain, "run_quick_search",
                          side_effect=lambda q, *a, **k:
                          searches.append(q) or {"query": q,
                                                 "spoken_summary": "s"}):
            result = brain._mi_research_step(
                {"kind": "research", "consumes": [0],
                 "text": "research about their release dates on the internet "
                         "and tell me what you find"},
                [prior])
        self.assertEqual(searches,
                         [self.ASPECT_LABEL + " release dates"])
        self.assertEqual(result["status"], "ok")

    def test_deictic_research_composes_the_screen_clause_aspect(self):
        context_state.OBSERVATIONS.add(context_state.Observation(
            id="O1", at=time.time(),
            utterance="what is this video about dimensions on my screen",
            image_data_url="data:image/png;base64,SEED",
            items=[_item(label="YOU IN 7D?")]))
        prior = {"kind": "screen", "status": "ok",
                 "output_query": "YOU IN 7D?",
                 "output_content": "A video is playing.",
                 "output_obs_id": "O1"}
        searches = []
        with patch.object(brain, "analyze_screen",
                          side_effect=AssertionError("no re-analysis")), \
             patch.object(brain, "identify_on_screen",
                          side_effect=AssertionError("no second vision call")), \
             patch.object(brain, "run_quick_search",
                          side_effect=lambda q, *a, **k:
                          searches.append(q) or {"query": q,
                                                 "spoken_summary": "s"}):
            result = brain._mi_research_step(
                {"kind": "research", "consumes": [0],
                 "text": "research about it and tell me"},
                [prior])
        self.assertEqual(searches, ["YOU IN 7D? dimensions"])
        self.assertEqual(result["status"], "ok")

    def test_research_fragment_keeps_the_full_answer(self):
        prior = {"kind": "screen", "status": "ok",
                 "output_query": "Some Video Title",
                 "output_content": "A video."}
        full = ("The video explains seven dimensional thinking and why "
                "creators keep returning to it. It compares the geometry "
                "with everyday intuition and closes with a practical demo. "
                "Viewers remember the final example word for word.")
        with patch.object(brain, "run_quick_search",
                          return_value={"query": "Some Video Title",
                                        "spoken_summary": full}):
            result = brain._mi_research_step(
                {"kind": "research", "consumes": [0],
                 "text": "research about it on the internet and tell me"},
                [prior])
        self.assertEqual(result["fragment"], full)

    def test_finish_reply_is_not_cut_at_the_old_limit(self):
        messages = []
        long_frag = "word " * 220
        with patch.object(brain, "_record_chain_run"), \
             patch.object(brain, "_notify_async_reply",
                          side_effect=lambda text, spoken=None:
                          messages.append(text)):
            brain._finish_multi_intent({"source": "x"}, [
                {"kind": "research", "status": "ok",
                 "fragment": long_frag.strip(),
                 "output_query": "some video"}])
        self.assertTrue(messages)
        self.assertGreater(len(messages[0]), 600)
        self.assertTrue(messages[0].endswith("word"))


class SubjectAspectTests(unittest.TestCase):
    """A possessive back-reference adds an ASPECT to the identified subject;
    it is composed in code instead of being re-resolved by a vision call."""

    LABEL = "I Tested Nano banana 2.1 vs GPT 2.5 (AI Image)"
    CLAUSE = ("research about their release dates on the internet and "
              "tell me what you find")

    def setUp(self):
        context_state.OBSERVATIONS.clear()

    def tearDown(self):
        context_state.OBSERVATIONS.clear()

    def _seed(self, label=None, truncated=False):
        context_state.OBSERVATIONS.add(context_state.Observation(
            id="O1", at=time.time(), image_data_url="u",
            items=[_item(label=label if label is not None else self.LABEL,
                         truncated=truncated)]))

    def test_possessive_composes_subject_and_aspect(self):
        self._seed()
        self.assertEqual(
            brain._mi_subject_aspect_query(self.CLAUSE, "O1"),
            self.LABEL + " release dates")

    def test_without_a_possessive_the_vision_resolver_keeps_ownership(self):
        self._seed()
        self.assertEqual(
            brain._mi_subject_aspect_query(
                "research about these image generation models on my screen",
                "O1"),
            "")

    def test_truncated_primary_label_is_never_composed(self):
        self._seed(truncated=True)
        self.assertEqual(
            brain._mi_subject_aspect_query(self.CLAUSE, "O1"), "")

    def test_generic_primary_label_is_never_composed(self):
        self._seed(label="YouTube live chat message")
        self.assertEqual(
            brain._mi_subject_aspect_query(self.CLAUSE, "O1"), "")

    def test_no_aspect_words_is_never_composed(self):
        self._seed()
        self.assertEqual(
            brain._mi_subject_aspect_query(
                "research about their video and tell me", "O1"),
            "")

    def test_missing_observation_is_never_composed(self):
        self.assertEqual(
            brain._mi_subject_aspect_query(self.CLAUSE, "O9"), "")


class AspectSuffixTests(unittest.TestCase):
    """The aspect may live in the screen clause when the request is deictic."""

    def setUp(self):
        context_state.OBSERVATIONS.clear()

    def tearDown(self):
        context_state.OBSERVATIONS.clear()

    def test_aspect_comes_from_the_screen_clause_when_deictic(self):
        context_state.OBSERVATIONS.add(context_state.Observation(
            id="O1", at=time.time(),
            utterance="what is this video about dimensions on my screen",
            items=[_item(label="YOU IN 7D?")]))
        self.assertEqual(
            brain._mi_aspect_suffix("research about it and tell me",
                                    "YOU IN 7D?", "YOU IN 7D?", "O1"),
            ["dimensions"])

    def test_aspect_already_in_the_query_is_not_doubled(self):
        self.assertEqual(
            brain._mi_aspect_suffix(
                "research about their release dates", "Inception",
                "Inception release dates", ""),
            [])

    def test_filler_words_are_never_aspects(self):
        self.assertEqual(
            brain._mi_aspect_suffix("what is this video very really about",
                                    "YOU IN 7D?", "YOU IN 7D?", ""),
            [])

    def test_instruction_words_are_never_aspects(self):
        self.assertEqual(
            brain._mi_aspect_suffix(
                "research about it on the internet and tell me",
                "YOU IN 7D?", "YOU IN 7D?", ""),
            [])


if __name__ == "__main__":
    unittest.main()
