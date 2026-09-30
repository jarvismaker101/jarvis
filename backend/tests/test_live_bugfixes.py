import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.services import model_registry, fireworks_client
from backend.core import brain
from backend.api import routes
from fastapi import HTTPException


class LiveBugfixBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._settings_path = Path(self._tmp.name) / "jarvis_settings.json"
        self._orig_settings = model_registry.SETTINGS_FILE
        self._orig_gemini = model_registry.GEMINI_API_KEY
        self._orig_fw = model_registry.FIREWORKS_API_KEY
        self._orig_groq = model_registry.GROQ_API_KEY
        self._orig_fish = model_registry.FISH_API_KEY
        self._orig_openrouter = model_registry.OPENROUTER_API_KEY
        model_registry.SETTINGS_FILE = self._settings_path
        model_registry.GEMINI_API_KEY = "test-gem"
        model_registry.FIREWORKS_API_KEY = "test-fw"
        model_registry.GROQ_API_KEY = "test-groq"
        model_registry.FISH_API_KEY = "test-fish"
        model_registry.OPENROUTER_API_KEY = "test-or"
        # reset last fallback
        brain._last_chat_fallback = None

    def tearDown(self):
        model_registry.SETTINGS_FILE = self._orig_settings
        model_registry.GEMINI_API_KEY = self._orig_gemini
        model_registry.FIREWORKS_API_KEY = self._orig_fw
        model_registry.GROQ_API_KEY = self._orig_groq
        model_registry.FISH_API_KEY = self._orig_fish
        model_registry.OPENROUTER_API_KEY = self._orig_openrouter
        self._tmp.cleanup()
        brain._last_chat_fallback = None


class BroadenedReasoningGateTests(LiveBugfixBase):
    def test_mentions_reasoning_broadened(self):
        self.assertTrue(fireworks_client._mentions_reasoning("unsupported parameter: reasoning_effort"))
        self.assertTrue(fireworks_client._mentions_reasoning("extra fields: reasoning_effort"))
        self.assertTrue(fireworks_client._mentions_reasoning("unknown field reasoning_effort"))
        self.assertTrue(fireworks_client._mentions_reasoning("Invalid reasoning effort: none"))
        self.assertTrue(fireworks_client._mentions_reasoning("Invalid parameter: effort"))
        self.assertTrue(fireworks_client._mentions_reasoning("thinking is not supported"))
        self.assertFalse(fireworks_client._mentions_reasoning("model not found"))
        self.assertFalse(fireworks_client._mentions_reasoning("rate limit exceeded"))

    def test_stream_retries_without_reasoning_on_unsupported(self):
        # Simulate fireworks stream 400 with unsupported parameter error, then success on retry
        messages = [{"role": "user", "content": "hi"}]
        # first response is 400 with extra fields, second is 200 with one chunk
        first = MagicMock()
        first.status_code = 400
        first.text = '{"error":"unsupported parameter: reasoning_effort"}'
        second = MagicMock()
        second.status_code = 200
        second.iter_lines.return_value = [
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            'data: [DONE]',
        ]
        with patch.object(fireworks_client, "API_KEY", "test-key"), \
             patch.object(fireworks_client.requests, "post", side_effect=[first, second]) as post:
            deltas = list(fireworks_client.ask_fireworks_stream(messages, temperature=0.7, max_tokens=10, model="accounts/fireworks/models/deepseek-v4-flash-vision-exp"))
            self.assertEqual(deltas, ["ok"])
            self.assertEqual(post.call_count, 2)
            # second retry must have dropped reasoning_effort
            second_payload = post.call_args_list[1].kwargs["json"]
            self.assertNotIn("reasoning_effort", second_payload)


class StreamFallbackTests(LiveBugfixBase):
    def test_stream_empty_retries_same_model_nonstream_before_gemini(self):
        messages = [{"role": "user", "content": "hello"}]
        model_registry.set_model_for_role("chat", "fireworks", "accounts/fireworks/models/deepseek-v4-flash-vision-exp")
        # stream yields nothing (error case), non-stream succeeds
        with patch.object(brain, "ask_fireworks_stream", return_value=iter([])) as mock_stream, \
             patch.object(brain, "ask_fireworks", return_value={"choices":[{"message":{"content":"nonstream ok"}}]}) as mock_nonstream, \
             patch.object(brain, "ask_gemini_chat_stream", return_value=iter([])) as mock_gem:
            deltas = list(brain._stream_chat_deltas(messages, 0.7, 100))
            self.assertEqual(deltas, ["nonstream ok"])
            mock_stream.assert_called_once()
            mock_nonstream.assert_called_once()
            # should not have fallen back to gemini
            mock_gem.assert_not_called()
            # no fallback recorded because same provider succeeded via non-stream
            self.assertIsNone(brain.get_last_chat_fallback())

    def test_stream_and_nonstream_both_fail_records_fallback_and_goes_to_gemini(self):
        messages = [{"role": "user", "content": "hi"}]
        model_registry.set_model_for_role("chat", "fireworks", "accounts/fireworks/models/deepseek-v4-flash-vision-exp")
        def fake_gem_stream(*a, **kw):
            yield "gemini ok"
        with patch.object(brain, "ask_fireworks_stream", return_value=iter([])), \
             patch.object(brain, "ask_fireworks", return_value={}), \
             patch.object(brain, "ask_gemini_chat_stream", side_effect=fake_gem_stream):
            deltas = list(brain._stream_chat_deltas(messages, 0.7, 100))
            self.assertEqual(deltas, ["gemini ok"])
            fb = brain.get_last_chat_fallback()
            self.assertIsNotNone(fb)
            self.assertEqual(fb["provider"], "fireworks")
            self.assertEqual(fb["model"], "accounts/fireworks/models/deepseek-v4-flash-vision-exp")
            self.assertEqual(fb["fallback_provider"], "gemini")


class LastFallbackExposureTests(LiveBugfixBase):
    def test_last_fallback_exposed_via_settings_and_scrubbed(self):
        # record a fallback with fake key material in error
        brain._record_chat_fallback("fireworks", "m", "error with Bearer sk-12345678901234567890", "gemini")
        resp = routes.get_settings()
        self.assertIn("last_fallback", resp)
        fb = resp["last_fallback"]
        self.assertEqual(fb["provider"], "fireworks")
        self.assertNotIn("sk-123", json.dumps(fb))
        self.assertIn("REDACTED", fb["error"] or "")

    def test_last_fallback_none_initially(self):
        resp = routes.get_settings()
        self.assertIsNone(resp["last_fallback"])


class AllowlistValidationTests(LiveBugfixBase):
    def test_chat_rejects_fish(self):
        with self.assertRaises(model_registry.ModelRegistryError) as ctx:
            model_registry.set_model_for_role("chat", "fish", "s1")
        self.assertIn("not allowed", str(ctx.exception).lower())

    def test_tts_rejects_fireworks(self):
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role("tts", "fireworks", "accounts/fireworks/models/qwen3p7-plus")

    def test_vision_rejects_custom(self):
        with patch.object(model_registry, "_list_openai_compat_models", return_value=[{"id":"m","display":"m"}]):
            model_registry.add_custom_provider("acme", "Acme", "sk-1", "https://acme.example/v1")
        with self.assertRaises(model_registry.ModelRegistryError):
            model_registry.set_model_for_role("vision", "acme", "m")

    def test_vision_accepts_fireworks_groq_openrouter_gemini(self):
        for prov, model in [("fireworks","accounts/fireworks/models/deepseek-v4-flash-vision-exp"),("groq","qwen/qwen3.6-27b"),("openrouter","google/gemma-4-31b-it:free"),("gemini","gemini-2.5-flash")]:
            model_registry.set_model_for_role("vision", prov, model)
            self.assertEqual(model_registry.get_model_for_role("vision")["provider"], prov)

    def test_browser_tool_accepts_custom(self):
        with patch.object(model_registry, "_list_openai_compat_models", return_value=[{"id":"m","display":"m"}]):
            model_registry.add_custom_provider("acme2", "Acme2", "sk-1", "https://acme2.example/v1")
        model_registry.set_model_for_role("browser_tool", "acme2", "acme-model")
        self.assertEqual(model_registry.get_model_for_role("browser_tool")["provider"], "acme2")

    def test_routes_returns_400_for_incompatible(self):
        payload = routes.ModelUpdate(role="tts", provider="fireworks", model="x")
        with self.assertRaises(HTTPException) as ctx:
            routes.set_model(payload)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_role_allowed_map_exposed(self):
        resp = routes.get_settings()
        self.assertIn("role_allowed", resp)
        self.assertIn("chat", resp["role_allowed"])
        self.assertIn("fireworks", resp["role_allowed"]["chat"])
        self.assertNotIn("fish", resp["role_allowed"]["chat"])
        self.assertIn("fish", resp["role_allowed"]["tts"])
        self.assertNotIn("acme", resp["role_allowed"]["tts"])


class FireworksVisionDispatchTests(LiveBugfixBase):
    def test_screen_analyzer_uses_fireworks_first(self):
        from backend.services import screen_analyzer
        img = "data:image/png;base64,abc"
        model_registry.set_model_for_role("vision", "fireworks", "accounts/fireworks/models/deepseek-v4-flash-vision-exp")
        fake_result = {"choices":[{"message":{"content":'{"tip":"ok","evidence":[],"topic":"t","show_images":false}'}}], "grounding_links":[]}
        with patch.object(screen_analyzer, "ask_groq_vision") as mock_groq, \
             patch("backend.services.fireworks_client.ask_fireworks_vision", return_value=fake_result) as mock_fw, \
             patch.object(screen_analyzer.gemini_client, "is_available", return_value=True), \
             patch.object(screen_analyzer.gemini_client, "ask_gemini_vision", return_value={}) as mock_gem:
            result = screen_analyzer._ask_screen_vision_cascade("prompt", img)
            mock_fw.assert_called_once()
            # verify model and image passed
            self.assertEqual(mock_fw.call_args.kwargs.get("model"), "accounts/fireworks/models/deepseek-v4-flash-vision-exp")
            self.assertEqual(mock_fw.call_args.args[1], img)
            mock_gem.assert_not_called()
            mock_groq.assert_not_called()

    def test_screen_control_uses_fireworks_first(self):
        from backend.services import screen_control
        img = "data:image/png;base64,abc"
        model_registry.set_model_for_role("vision", "fireworks", "accounts/fireworks/models/deepseek-v4-flash-vision-exp")
        fake_result = {"choices":[{"message":{"content":'{"ok":true}'}}]}
        with patch("backend.services.fireworks_client.ask_fireworks_vision", return_value=fake_result) as mock_fw, \
             patch.object(screen_control.gemini_client, "is_available", return_value=True), \
             patch.object(screen_control.gemini_client, "ask_gemini_vision", return_value={}) as mock_gem, \
             patch("backend.services.screen_control.ask_groq_vision") as mock_groq:
            result = screen_control._ask_vision_cascade("prompt", img)
            mock_fw.assert_called_once()
            self.assertEqual(mock_fw.call_args.kwargs.get("model"), "accounts/fireworks/models/deepseek-v4-flash-vision-exp")
            mock_gem.assert_not_called()

    def test_verification_uses_fireworks(self):
        from backend.services import screen_control
        # need to mock capture and verification
        model_registry.set_model_for_role("vision", "fireworks", "accounts/fireworks/models/deepseek-v4-flash-vision-exp")
        fake_result = {"choices":[{"message":{"content":'{"verified":true,"observation":"ok"}'}}]}
        with patch("backend.services.screen_control.capture_active_window", return_value={"image_data_url":"data:image/png;base64,abc"}), \
             patch("backend.services.screen_control._cloud_verification_enabled", return_value=True), \
             patch("backend.services.screen_control._typed_texts_from_steps", return_value=["hello"]), \
             patch("backend.services.screen_control._verification_text_haystack", return_value=None), \
             patch("backend.services.fireworks_client.ask_fireworks_vision", return_value=fake_result) as mock_fw, \
             patch.object(screen_control.gemini_client, "is_available", return_value=False):
            result = screen_control._verify_action_with_steps("typed hello", steps=[{"action":"type","text":"hello"}])
            mock_fw.assert_called_once()
            self.assertTrue(result is True or result is None)  # allow parsing


class GroqVisionDispatchTests(LiveBugfixBase):
    def test_screen_analyzer_uses_groq_first_and_passes_selected_model(self):
        from backend.services import screen_analyzer
        img = "data:image/png;base64,abc"
        model_registry.set_model_for_role("vision", "groq", "qwen/qwen3.6-27b-custom")
        fake_result = {"choices":[{"message":{"content":'{"tip":"ok","evidence":[],"topic":"t","show_images":false}'}}], "grounding_links":[]}
        with patch.object(screen_analyzer, "ask_groq_vision", return_value=fake_result) as mock_groq, \
             patch.object(screen_analyzer.gemini_client, "is_available", return_value=True), \
             patch.object(screen_analyzer.gemini_client, "ask_gemini_vision", return_value={}) as mock_gem, \
             patch("backend.services.fireworks_client.ask_fireworks_vision") as mock_fw:
            result = screen_analyzer._ask_screen_vision_cascade("prompt", img)
            mock_groq.assert_called_once()
            self.assertEqual(mock_groq.call_args.kwargs.get("model"), "qwen/qwen3.6-27b-custom")
            self.assertEqual(mock_groq.call_args.args[1], img)
            self.assertNotIn("response_format", mock_groq.call_args.kwargs)
            mock_gem.assert_not_called()
            mock_fw.assert_not_called()

    def test_screen_analyzer_fallback_groq_uses_selected_model(self):
        from backend.services import screen_analyzer
        img = "data:image/png;base64,abc"
        # F37: the selected groq is attempted EXACTLY ONCE with the selected
        # model; an empty/failed attempt advances to the next eligible
        # provider instead of retrying the same provider (the old cascade's
        # redundant second groq attempt).
        model_registry.set_model_for_role("vision", "groq", "qwen/qwen3.6-27b-custom")
        with patch.object(screen_analyzer, "ask_groq_vision", return_value={}) as mock_groq, \
             patch.object(screen_analyzer.gemini_client, "is_available", return_value=True), \
             patch.object(screen_analyzer.gemini_client, "ask_gemini_vision", return_value={}) as mock_gem:
            screen_analyzer._ask_screen_vision_cascade("prompt", img)
            self.assertEqual(mock_groq.call_count, 1)
            self.assertEqual(mock_groq.call_args.kwargs.get("model"), "qwen/qwen3.6-27b-custom")
            self.assertNotIn("response_format", mock_groq.call_args.kwargs)
            # the cascade advanced to the next eligible provider (gemini)
            self.assertEqual(mock_gem.call_count, 1)

    def test_screen_control_uses_groq_first_and_passes_selected_model(self):
        from backend.services import screen_control
        img = "data:image/png;base64,abc"
        model_registry.set_model_for_role("vision", "groq", "qwen/qwen3.6-27b-custom")
        fake_result = {"choices":[{"message":{"content":'{"ok":true}'}}]}
        with patch("backend.services.screen_control.ask_groq_vision", return_value=fake_result) as mock_groq, \
             patch.object(screen_control.gemini_client, "is_available", return_value=True), \
             patch.object(screen_control.gemini_client, "ask_gemini_vision", return_value={}) as mock_gem, \
             patch("backend.services.fireworks_client.ask_fireworks_vision") as mock_fw:
            result = screen_control._ask_vision_cascade("prompt", img)
            mock_groq.assert_called_once()
            self.assertEqual(mock_groq.call_args.kwargs.get("model"), "qwen/qwen3.6-27b-custom")
            self.assertEqual(mock_groq.call_args.args[1], img)
            self.assertNotIn("response_format", mock_groq.call_args.kwargs)
            mock_gem.assert_not_called()
            mock_fw.assert_not_called()

    def test_verification_uses_groq(self):
        from backend.services import screen_control
        model_registry.set_model_for_role("vision", "groq", "qwen/qwen3.6-27b-custom")
        fake_result = {"choices":[{"message":{"content":'{"verified":true,"observation":"ok"}'}}]}
        with patch("backend.services.screen_control.capture_active_window", return_value={"image_data_url":"data:image/png;base64,abc"}), \
             patch("backend.services.screen_control._cloud_verification_enabled", return_value=True), \
             patch("backend.services.screen_control._typed_texts_from_steps", return_value=["hello"]), \
             patch("backend.services.screen_control._verification_text_haystack", return_value=None), \
             patch("backend.services.screen_control.ask_groq_vision", return_value=fake_result) as mock_groq, \
             patch.object(screen_control.gemini_client, "is_available", return_value=False):
            result = screen_control._verify_action_with_steps("typed hello", steps=[{"action":"type","text":"hello"}])
            mock_groq.assert_called_once()
            self.assertNotIn("response_format", mock_groq.call_args.kwargs)
            self.assertEqual(mock_groq.call_args.kwargs.get("model"), "qwen/qwen3.6-27b-custom")


class BrowserShortSummaryTests(LiveBugfixBase):
    def _wait(self, pred, timeout=3.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if pred():
                return True
            time.sleep(0.02)
        return pred()

    def test_summarizer_trims_2000_char_output(self):
        long_output = ("Visited https://example.com and extracted the price $42. "
                       "Checked three more pages for stock status. ") * 25
        self.assertGreater(len(long_output), 2000)
        summary = brain._summarize_browser_output(long_output, "find the price on example.com")
        self.assertLessEqual(len(summary), 300, f"len={len(summary)}: {summary}")
        self.assertLessEqual(len(summary.splitlines()), 3)
        self.assertIn("example.com", summary)

    def test_summarizer_failure_short(self):
        out = "TASK NOT COMPLETED. Error: " + ("page timed out waiting for selector. " * 30)
        summary = brain._summarize_browser_output(out, "open the dashboard")
        self.assertLessEqual(len(summary), 300)
        self.assertLessEqual(len(summary.splitlines()), 3)
        self.assertIn("could not be completed", summary.lower())

    def test_browser_task_reply_short_and_detail_kept(self):
        from backend import config
        long_output = ("Opened https://example.com, clicked the product, read price $42. " * 40)
        self.assertGreater(len(long_output), 2000)
        with patch.object(config, "TASK_ENGINE", "browser_agent"), \
             patch.object(brain, "run_browser_task", return_value=long_output), \
             patch.object(brain, "_notify_async_reply") as notify:
            brain._execute_deferred_opencode("find price on example.com", "find price on example.com")
            self.assertTrue(self._wait(lambda: notify.called))
        text = notify.call_args[0][0]
        spoken = notify.call_args[1].get("spoken")
        self.assertLessEqual(len(text), 300, f"chat headline len={len(text)}")
        self.assertLessEqual(len(text.splitlines()), 3)
        self.assertLessEqual(len(spoken), 300)
        # full text remains reachable in the detail path
        self.assertEqual(brain.get_last_browser_full_output(), long_output)


class BargeInStopTests(unittest.TestCase):
    def test_ask_stops_speaking_before_processing(self):
        import backend.api.routes as routes_mod
        parent = MagicMock()
        parent.attach_mock(MagicMock(), "stop")
        parent.attach_mock(MagicMock(return_value="ok reply"), "process")
        with patch.object(routes_mod, "stop_speaking", parent.stop), \
             patch.object(routes_mod, "process_message", parent.process), \
             patch.object(routes_mod, "_maybe_speak"):
            resp = routes_mod.ask(routes_mod.Query(message="barge-in-order-probe-%d" % time.time_ns()))
        # F23 — /ask answers with a request id so a retry can attach instead
        # of executing again.
        self.assertEqual(resp["reply"], "ok reply")
        self.assertTrue(resp["request_id"])
        # order: stop before process
        names = [c[0] for c in parent.mock_calls]
        self.assertIn("stop", names)
        self.assertIn("process", names)
        self.assertLess(names.index("stop"), names.index("process"))

    def test_speak_stop_endpoint_calls_stop_speaking(self):
        import backend.api.routes as routes_mod
        with patch.object(routes_mod, "stop_speaking") as mock_stop:
            resp = routes_mod.stop_speech()
        mock_stop.assert_called_once_with()
        self.assertEqual(resp, {"ok": True})

    def test_speech_onset_stops_when_speaking(self):
        from backend.services import listener as listener_mod
        with patch.object(listener_mod.listener_state, "is_speaking", return_value=True), \
             patch("backend.services.voice.stop_speaking") as mock_stop, \
             patch.object(listener_mod, "_post_backend_speak_stop", return_value=True) as mock_post:
            result = listener_mod.barge_in_on_speech_onset()
        self.assertTrue(result)
        # [P1-02] Barge-in asks for the silent stop: no "ready" beep while the
        # user is mid-sentence. The local stop itself is unchanged.
        mock_stop.assert_called_once_with(signal_ready=False)
        mock_post.assert_called_once_with()

    def test_speech_onset_still_queues_the_stop_when_local_silent(self):
        """[P1-03] Was "noop when silent". The old gate needed a blocking
        GET /voice-state probe on the capture thread, which the audit removes:
        /speak/stop is idempotent and now rides a background worker, so every
        onset queues one. LOCAL audio is still untouched when nothing plays
        locally - that half of the old assertion is preserved."""
        from backend.services import listener as listener_mod
        with patch.object(listener_mod.listener_state, "is_speaking", return_value=False), \
             patch("backend.services.voice.stop_speaking") as mock_stop, \
             patch.object(listener_mod, "_post_backend_speak_stop") as mock_post:
            result = listener_mod.barge_in_on_speech_onset()
        self.assertTrue(result)
        mock_stop.assert_not_called()
        mock_post.assert_called_once_with()

    def test_stop_invalidates_generation_and_clears_fish_buffer(self):
        from backend.services import voice as voice_mod
        from backend.services import fish_voice as fish_mod
        # quiet pre-state: not speaking so no earcon side effects
        with voice_mod._state_lock:
            voice_mod.is_speaking = False
        voice_mod.stop_speaking()
        with voice_mod._state_lock:
            gen = voice_mod._speech_generation
        # fake in-flight fish playback handle = buffered PCM still held
        fake = MagicMock()
        with fish_mod._playback_lock:
            fish_mod._current_playback = fake
        voice_mod.stop_speaking()
        self.assertFalse(voice_mod._is_current_generation(gen),
                         "generation token must invalidate so queued chunks abort")
        with fish_mod._playback_lock:
            current = fish_mod._current_playback
        self.assertIsNone(current, "fish playback buffer handle must be cleared")
        fake.stop.assert_called()


class ScreenAnalyzerNoGroqPreconditionTests(LiveBugfixBase):
    """F37: analyze_screen carries no GROQ_API_KEY precondition. The cascade
    attempts the selected provider and eligible fallbacks, and only reports
    unavailable when none can serve the request."""

    def _capture(self):
        return {"image_data_url": "data:image/png;base64,abc", "region": None}

    def _tip_result(self, tip):
        content = json.dumps({"tip": tip, "evidence": [], "topic": "t", "show_images": False})
        return {
            "choices": [{"message": {"content": content}}],
            "grounding_links": [],
        }

    def test_fireworks_primary_serves_without_groq(self):
        from backend.services import screen_analyzer
        with patch.object(screen_analyzer, "capture_primary_screen", return_value=self._capture()), \
             patch.object(screen_analyzer, "_resolve_vision_model",
                          return_value={"provider": "fireworks", "model": "fw-vision"}), \
             patch("backend.services.fireworks_client.ask_fireworks_vision",
                   return_value=self._tip_result("fw answer")) as mock_fw, \
             patch.object(screen_analyzer, "ask_groq_vision", return_value={}) as mock_groq:
            result = screen_analyzer.analyze_screen("what is on my screen")
        self.assertEqual(result["tip"], "fw answer")
        mock_fw.assert_called_once()
        mock_groq.assert_not_called()

    def test_gemini_primary_serves_without_groq(self):
        from backend.services import screen_analyzer
        with patch.object(screen_analyzer, "capture_primary_screen", return_value=self._capture()), \
             patch.object(screen_analyzer, "_resolve_vision_model",
                          return_value={"provider": "gemini", "model": "gm-vision"}), \
             patch.object(screen_analyzer.gemini_client, "is_available", return_value=True), \
             patch.object(screen_analyzer.gemini_client, "ask_gemini_vision",
                          return_value=self._tip_result("gm answer")) as mock_gem, \
             patch.object(screen_analyzer, "ask_groq_vision", return_value={}) as mock_groq:
            result = screen_analyzer.analyze_screen("what is on my screen")
        self.assertEqual(result["tip"], "gm answer")
        mock_gem.assert_called_once()
        mock_groq.assert_not_called()

    def test_no_provider_serves_reports_unavailable(self):
        from backend.services import screen_analyzer
        with patch.object(screen_analyzer, "capture_primary_screen", return_value=self._capture()), \
             patch.object(screen_analyzer, "_resolve_vision_model",
                          return_value={"provider": "gemini", "model": "gm-vision"}), \
             patch.object(screen_analyzer.gemini_client, "is_available", return_value=False), \
             patch.object(screen_analyzer, "ask_groq_vision", return_value={}):
            result = screen_analyzer.analyze_screen("what is on my screen")
        self.assertIn("couldn't get a response", result["tip"])
        self.assertEqual(result["evidence"], [])

    def test_unavailable_primary_falls_back_to_configured_provider(self):
        from backend.services import screen_analyzer
        with patch.object(screen_analyzer, "capture_primary_screen", return_value=self._capture()), \
             patch.object(screen_analyzer, "_resolve_vision_model",
                          return_value={"provider": "fireworks", "model": "fw-vision"}), \
             patch("backend.services.fireworks_client.ask_fireworks_vision", return_value={}), \
             patch.object(screen_analyzer.gemini_client, "is_available", return_value=True), \
             patch.object(screen_analyzer.gemini_client, "ask_gemini_vision",
                          return_value=self._tip_result("fallback answer")), \
             patch.object(screen_analyzer, "ask_groq_vision", return_value={}):
            result = screen_analyzer.analyze_screen("what is on my screen")
        self.assertEqual(result["tip"], "fallback answer")


if __name__ == "__main__":
    unittest.main()
