"""F37 — Remove the Unrelated Groq Prerequisite: the vision cascade.

Acceptance clauses pinned by this module:

  * "Supported single-provider installations need no unrelated key" — only the
    configured provider is ever dispatched; a provider without a credential is
    recorded as skipped with its reason and NEVER called.
  * "exceptions/malformed output advance safely" — an adapter exception and a
    nonempty-but-schema-invalid response both advance to the next eligible
    provider instead of ending (or escaping) the cascade.
  * "total unavailability is accurate and names actual attempted providers" —
    the failure report/message names exactly the providers that were actually
    dispatched (and distinguishes them from ineligible skips).

No network: every adapter is mocked at the boundary; the OS/screen boundary is
never touched (`capture_*` is patched).
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.services import model_registry
from backend.services import screen_analyzer
from backend.services import screen_control
from backend.services import vision_cascade


def _tip_result(tip):
    content = json.dumps({
        "tip": tip, "evidence": [], "topic": "t", "show_images": False})
    return {"choices": [{"message": {"content": content}}], "grounding_links": []}


def _json_result(payload):
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def _prose_result(text="Sure sir, here is what I can see."):
    """Nonempty output that is NOT the JSON object the prompt asked for."""
    return {"choices": [{"message": {"content": text}}]}


class VisionCascadeBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._settings_path = Path(self._tmp.name) / "jarvis_settings.json"
        self._orig_settings = model_registry.SETTINGS_FILE
        self._orig_keys = {
            name: getattr(model_registry, name)
            for name in ("GEMINI_API_KEY", "FIREWORKS_API_KEY",
                         "GROQ_API_KEY", "OPENROUTER_API_KEY")
        }
        model_registry.SETTINGS_FILE = self._settings_path
        model_registry.GEMINI_API_KEY = "test-gem"
        model_registry.FIREWORKS_API_KEY = "test-fw"
        model_registry.GROQ_API_KEY = "test-groq"
        model_registry.OPENROUTER_API_KEY = "test-or"
        self.capture = {"image_data_url": "data:image/png;base64,abc",
                        "region": None}

    def tearDown(self):
        model_registry.SETTINGS_FILE = self._orig_settings
        for name, value in self._orig_keys.items():
            setattr(model_registry, name, value)
        self._tmp.cleanup()

    def _only_gemini_configured(self):
        """A supported single-provider installation: Gemini and nothing else."""
        model_registry.GEMINI_API_KEY = "test-gem"
        model_registry.FIREWORKS_API_KEY = ""
        model_registry.GROQ_API_KEY = ""
        model_registry.OPENROUTER_API_KEY = ""
        model_registry.set_model_for_role("vision", "gemini", "gm-vision")


class SingleProviderInstallationTests(VisionCascadeBase):
    """Acceptance: a supported single-provider installation needs no
    unrelated key."""

    def test_analyze_screen_serves_from_gemini_without_any_groq_key(self):
        self._only_gemini_configured()
        groq = MagicMock(name="ask_groq_vision", return_value={})
        fireworks = MagicMock(name="ask_fireworks_vision", return_value={})
        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=self.capture), \
             patch.object(screen_analyzer.gemini_client, "is_available",
                          return_value=True), \
             patch.object(screen_analyzer.gemini_client, "ask_gemini_vision",
                          return_value=_tip_result("gm answer")) as gemini, \
             patch.object(screen_analyzer, "ask_groq_vision", groq), \
             patch("backend.services.fireworks_client.ask_fireworks_vision", fireworks):
            result = screen_analyzer.analyze_screen("what is on my screen")

        self.assertEqual(result["tip"], "gm answer")
        gemini.assert_called_once()
        self.assertEqual(gemini.call_args.kwargs.get("model"), "gm-vision")
        # The unrelated providers were never dispatched at all — their keys do
        # not exist in this installation.
        groq.assert_not_called()
        fireworks.assert_not_called()
        self.assertEqual(result["vision_provider"], "gemini")
        self.assertEqual(result["vision_model"], "gm-vision")
        attempted = [a["provider"] for a in result["vision_attempts"]
                     if a["outcome"] != vision_cascade.OUTCOME_SKIPPED]
        self.assertEqual(attempted, ["gemini"])

    def test_ineligible_providers_are_skipped_with_a_reason_not_dispatched(self):
        self._only_gemini_configured()
        with patch("backend.services.gemini_client.is_available",
                   return_value=True):
            eligible, skipped = vision_cascade.provider_candidates(
                selected={"provider": "gemini", "model": "gm-vision"})
        self.assertEqual([c["provider"] for c in eligible], ["gemini"])
        skipped_by_provider = {s["provider"]: s["reason"] for s in skipped}
        for provider in ("groq", "fireworks", "openrouter"):
            self.assertIn(provider, skipped_by_provider)
            self.assertIn("credential", skipped_by_provider[provider])

    def test_screen_control_cascade_never_reaches_an_unconfigured_provider(self):
        self._only_gemini_configured()
        groq = MagicMock(name="ask_groq_vision", return_value={})
        fireworks = MagicMock(name="ask_fireworks_vision", return_value={})
        with patch.object(screen_control.gemini_client, "is_available",
                          return_value=True), \
             patch.object(screen_control.gemini_client, "ask_gemini_vision",
                          return_value={}), \
             patch.object(screen_control, "ask_groq_vision", groq), \
             patch("backend.services.fireworks_client.ask_fireworks_vision", fireworks):
            attempts = {}
            screen_control._ask_vision_cascade(
                "prompt", "data:image/png;base64,abc", attempts_out=attempts)
        groq.assert_not_called()
        fireworks.assert_not_called()
        self.assertEqual(vision_cascade.attempted_providers(attempts), ["gemini"])
        self.assertFalse(attempts["usable"])


class EligibilityDiscoveryTests(VisionCascadeBase):
    """The eligibility gates: role allowlist, capability, credentials."""

    def test_selected_provider_comes_first(self):
        eligible = vision_cascade.eligible_vision_providers(
            selected={"provider": "fireworks", "model": "fw-vision"},
            availability={"gemini": True, "fireworks": True, "groq": True,
                          "openrouter": True})
        self.assertEqual(eligible[0]["provider"], "fireworks")
        self.assertEqual(eligible[0]["model"], "fw-vision")

    def test_fallback_providers_use_their_own_adapter_default_model(self):
        eligible = vision_cascade.eligible_vision_providers(
            selected={"provider": "fireworks", "model": "fw-vision"},
            availability={"gemini": True, "fireworks": True, "groq": True,
                          "openrouter": True})
        by_provider = {c["provider"]: c for c in eligible}
        # The selected model must not be leaked onto another provider's call.
        self.assertIsNone(by_provider["gemini"]["model"])
        self.assertIsNone(by_provider["groq"]["model"])

    def test_provider_not_allowed_for_the_role_is_ineligible(self):
        eligible, skipped = vision_cascade.provider_candidates(
            selected={"provider": "fish", "model": "s1"},
            availability={"fish": True, "gemini": True})
        providers = [c["provider"] for c in eligible]
        self.assertNotIn("fish", providers)
        self.assertIn("gemini", providers)
        fish = [s for s in skipped if s["provider"] == "fish"]
        self.assertEqual(len(fish), 1)
        self.assertIn("vision role", fish[0]["reason"])

    def test_unknown_provider_is_ineligible(self):
        eligible, skipped = vision_cascade.provider_candidates(
            selected={"provider": "mystery", "model": "m"},
            availability={"mystery": True, "gemini": True})
        self.assertNotIn("mystery", [c["provider"] for c in eligible])
        self.assertTrue(any(s["provider"] == "mystery" for s in skipped))


class MalformedAndExceptionAdvanceTests(VisionCascadeBase):
    """Acceptance: exceptions/malformed output advance safely."""

    def test_adapter_exception_advances_to_the_next_eligible_provider(self):
        model_registry.set_model_for_role(
            "vision", "fireworks", "fw-vision")
        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=self.capture), \
             patch("backend.services.fireworks_client.ask_fireworks_vision",
                   side_effect=RuntimeError("boom")) as fw, \
             patch.object(screen_analyzer.gemini_client, "is_available",
                          return_value=True), \
             patch.object(screen_analyzer.gemini_client, "ask_gemini_vision",
                          return_value=_tip_result("gm answered")) as gemini:
            result = screen_analyzer.analyze_screen("what is on my screen")

        fw.assert_called_once()
        gemini.assert_called_once()
        self.assertEqual(result["tip"], "gm answered")
        self.assertEqual(result["vision_provider"], "gemini")
        outcomes = {a["provider"]: a["outcome"] for a in result["vision_attempts"]}
        self.assertEqual(outcomes["fireworks"], vision_cascade.OUTCOME_ERROR)

    def test_malformed_nonempty_output_does_not_end_the_cascade(self):
        model_registry.set_model_for_role(
            "vision", "openrouter", "vendor/vision-model")
        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=self.capture), \
             patch("backend.services.openrouter_client.ask_openrouter_vision",
                   return_value=_prose_result("Sure! I can see a code editor.")) as primary, \
             patch.object(screen_analyzer.gemini_client, "is_available",
                          return_value=True), \
             patch.object(screen_analyzer.gemini_client, "ask_gemini_vision",
                          return_value=_tip_result("gm answer")) as gemini:
            result = screen_analyzer.analyze_screen("what is on my screen")

        primary.assert_called_once()
        gemini.assert_called_once()
        # The prose never became the answer: a schema-valid response did.
        self.assertEqual(result["tip"], "gm answer")
        self.assertEqual(result["vision_provider"], "gemini")
        outcomes = {a["provider"]: a["outcome"] for a in result["vision_attempts"]}
        self.assertEqual(outcomes["openrouter"], vision_cascade.OUTCOME_MALFORMED)
        self.assertFalse(result["vision_degraded"])

    def test_screen_control_plan_validation_also_advances_on_malformed(self):
        model_registry.set_model_for_role(
            "vision", "openrouter", "vendor/vision-model")
        with patch("backend.services.openrouter_client.ask_openrouter_vision",
                   return_value=_prose_result("I think you should click there.")), \
             patch.object(screen_control.gemini_client, "is_available",
                          return_value=True), \
             patch.object(screen_control.gemini_client, "ask_gemini_vision",
                          return_value=_json_result({"ok": True})):
            attempts = {}
            result = screen_control._ask_vision_cascade(
                "prompt", "data:image/png;base64,abc", attempts_out=attempts)
        self.assertTrue(attempts["usable"])
        self.assertEqual(attempts["provider"], "gemini")
        self.assertEqual(result, _json_result({"ok": True}))

    def test_degraded_last_resort_is_reported_not_silently_served(self):
        """When EVERY eligible provider only returns malformed text, the last
        such result is handed back FLAGGED as degraded — never as usable."""
        self._only_gemini_configured()
        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=self.capture), \
             patch.object(screen_analyzer.gemini_client, "is_available",
                          return_value=True), \
             patch.object(screen_analyzer.gemini_client, "ask_gemini_vision",
                          return_value=_prose_result("raw prose answer")):
            result = screen_analyzer.analyze_screen("what is on my screen")
        self.assertTrue(result["vision_degraded"])
        self.assertIsNone(result["vision_provider"])
        outcomes = {a["outcome"] for a in result["vision_attempts"]}
        self.assertIn(vision_cascade.OUTCOME_MALFORMED, outcomes)


class BoundedFallbackTests(VisionCascadeBase):
    """Bounded fallback: each provider at most ONCE, attempts capped."""

    def test_selected_provider_is_attempted_exactly_once(self):
        calls = []

        def make(provider, result):
            def _call(prompt, image, model, max_tokens, response_format):
                calls.append(provider)
                return result
            return _call

        dispatchers = {
            "gemini": make("gemini", {}),
            "fireworks": make("fireworks", {}),
            "groq": make("groq", {}),
            "openrouter": make("openrouter", {}),
        }
        result, report = vision_cascade.ask_vision_with_fallback(
            "p", "img",
            dispatchers=dispatchers,
            validate=lambda r: (True, ""),
            selected={"provider": "groq", "model": "qwen/x"},
            availability={"gemini": True, "fireworks": True, "groq": True,
                          "openrouter": True},
        )
        self.assertEqual(calls.count("groq"), 1)
        # No provider is ever retried later in the chain.
        self.assertEqual(sorted(calls), sorted(set(calls)))
        # Nothing was served, and nothing malformed survived either.
        self.assertIsNone(result)
        self.assertFalse(report["usable"])

    def test_attempts_are_bounded(self):
        calls = []

        def _call(provider):
            def inner(prompt, image, model, max_tokens, response_format):
                calls.append(provider)
                return {}
            return inner

        dispatchers = {p: _call(p) for p in
                       ("gemini", "fireworks", "groq", "openrouter")}
        vision_cascade.ask_vision_with_fallback(
            "p", "img",
            dispatchers=dispatchers,
            validate=lambda r: (True, ""),
            selected={"provider": "gemini", "model": "m"},
            availability={p: True for p in dispatchers},
            max_attempts=2,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls, ["gemini", "fireworks"])


class TotalUnavailabilityTests(VisionCascadeBase):
    """Acceptance: total unavailability is accurate and names the ACTUAL
    attempted providers."""

    def test_message_names_the_attempted_providers(self):
        model_registry.set_model_for_role("vision", "gemini", "gm-vision")
        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=self.capture), \
             patch.object(screen_analyzer.gemini_client, "is_available",
                          return_value=True), \
             patch.object(screen_analyzer.gemini_client, "ask_gemini_vision",
                          return_value={}), \
             patch.object(screen_analyzer, "ask_groq_vision",
                          return_value={}) as groq, \
             patch("backend.services.fireworks_client.ask_fireworks_vision",
                   return_value={}):
            result = screen_analyzer.analyze_screen("what is on my screen")

        self.assertIn("couldn't get a response", result["tip"])
        self.assertIn("tried:", result["tip"])
        self.assertIn("gemini", result["tip"])
        self.assertIn("groq", result["tip"])
        # Every named provider was really dispatched.
        attempted = vision_cascade.attempted_providers(
            {"attempted": [a["provider"] for a in result["vision_attempts"]
                           if a["outcome"] != vision_cascade.OUTCOME_SKIPPED]})
        for provider in attempted:
            self.assertIn(provider, result["tip"])
        self.assertIn("gemini", attempted)
        self.assertIn("groq", attempted)
        # Accuracy: the reason text is the honest attempt report.
        self.assertIn("no vision provider served the request",
                      result["vision_unavailable_reason"])
        groq.assert_called_once()

    def test_nothing_eligible_is_reported_as_nothing_attempted(self):
        model_registry.GEMINI_API_KEY = ""
        model_registry.FIREWORKS_API_KEY = ""
        model_registry.GROQ_API_KEY = ""
        model_registry.OPENROUTER_API_KEY = ""
        model_registry.set_model_for_role("vision", "gemini", "gm-vision")
        with patch.object(screen_analyzer, "capture_primary_screen",
                          return_value=self.capture), \
             patch.object(screen_analyzer.gemini_client, "is_available",
                          return_value=False):
            result = screen_analyzer.analyze_screen("what is on my screen")
        self.assertIn("couldn't get a response", result["tip"])
        self.assertIn("No vision provider is configured or eligible",
                      result["tip"])
        self.assertEqual(
            vision_cascade.attempted_providers(
                {"attempted": [a["provider"] for a in result["vision_attempts"]
                               if a["outcome"] != vision_cascade.OUTCOME_SKIPPED]}),
            [])
        self.assertIn("none was attempted", result["vision_unavailable_reason"])

    def test_unavailable_reason_never_names_an_unattempted_provider(self):
        report = {
            "attempted": [],
            "attempts": [
                {"provider": "gemini", "outcome": "skipped",
                 "detail": "no credentials configured"},
            ],
        }
        reason = vision_cascade.unavailable_reason(report)
        self.assertIn("none was attempted", reason)
        self.assertIn("gemini: skipped", reason)


class VerificationUsesTheSharedCascadeTests(VisionCascadeBase):
    """The screen-action verification path must use the SAME cascade — the old
    inline gemini->groq tail with an unguarded final Groq call is gone."""

    def test_verification_routes_through_the_cascade(self):
        valid = _json_result({"verified": True, "observation": "text is there"})
        with patch("backend.services.screen_control.capture_active_window",
                   return_value={"image_data_url": "data:image/png;base64,abc"}), \
             patch("backend.services.screen_control._cloud_verification_enabled",
                   return_value=True), \
             patch("backend.services.screen_control._typed_texts_from_steps",
                   return_value=["hello"]), \
             patch("backend.services.screen_control._verification_text_haystack",
                   return_value=None), \
             patch.object(screen_control, "_ask_vision_cascade",
                          return_value=valid) as cascade:
            verified = screen_control._verify_action_with_steps(
                "typed hello", steps=[{"action": "type", "text": "hello"}])
        self.assertTrue(verified)
        cascade.assert_called_once()

    def test_verification_reports_none_when_nothing_served(self):
        with patch("backend.services.screen_control.capture_active_window",
                   return_value={"image_data_url": "data:image/png;base64,abc"}), \
             patch("backend.services.screen_control._cloud_verification_enabled",
                   return_value=True), \
             patch("backend.services.screen_control._typed_texts_from_steps",
                   return_value=["hello"]), \
             patch("backend.services.screen_control._verification_text_haystack",
                   return_value=None), \
             patch.object(screen_control, "_ask_vision_cascade",
                          return_value={}):
            verified = screen_control._verify_action_with_steps(
                "typed hello", steps=[{"action": "type", "text": "hello"}])
        self.assertIsNone(verified)


if __name__ == "__main__":
    unittest.main()
