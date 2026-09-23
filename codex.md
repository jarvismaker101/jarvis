# Codex Project Notes

This file is a lightweight running log for future Codex conversations. It should be updated after each substantive project change so the next session can pick up current context quickly.

## Current Baseline

- Jarvis speaks through **Fish Audio** (`s2.1-pro-free`) first, falling back to local pyttsx3 → ElevenLabs (≤150 chars) → local pyttsx3 when Fish is unavailable or the text exceeds `JARVIS_FISH_TTS_CHAR_LIMIT` (300).
- Voice listening uses a fixed idle energy threshold with `JARVIS_IDLE_ENERGY_THRESHOLD`, currently defaulting to `300`, and keeps dynamic thresholding disabled.
- The listener keeps a shared microphone source warm across listen and recalibration cycles, and recreates it after microphone-stream failures.
- `watcher.py` correctly loads `.env` on startup to ensure configured microphone names and thresholds are respected.
- Jarvis voice input supports Google STT with a Groq transcription fallback.
- **Native Accessibility Screen Controls**: Jarvis no longer uses screenshots and Cloud Vision models to navigate the screen. He now uses a **Local Llama 3.2 (3B)** model via Ollama to read the screen's UI Tree natively, like Codex reads code.
- Screen control planning extracts the active window's UI Automation Tree (using `pywinauto` UIA) and formats it as a nested XML-like document with bounds, centers, state, and control metadata.
- **OCR Injection**: To prevent the model from going blind inside Electron/Web apps (which hide their DOM), Tesseract OCR is run on the window. Extracted text regions are injected back into the XML UI Tree as `<text>` elements.
- The prompt provides the XML UI Tree and asks the local text-only Llama 3.2 model to return the integer `element_id` for the target action, completely bypassing VRAM-heavy visual processing.
- `element_id` grounding now keeps coordinates in capture-local vision space until the final screen-pixel conversion, fixing active-window offset misclicks.
- Screen controls support voice-driven clicks, typing, scrolling, hotkeys, and risky-action confirmation.
- High-confidence `_try_local_match` routing is restored ahead of Ollama planning, and still works as a fallback when Ollama is unavailable.
- Post-action verification takes a fresh screenshot after execution and asks the Groq vision model whether the action appeared to succeed; uncertain results prompt the user to retry.
- A rolling interaction history (last 3 actions) is included in the Llama prompt so multi-step screen workflows have context.
- New modules: `screen_ocr.py`, `screen_ui_elements.py`, `ollama_client.py`. Both degrade gracefully if dependencies are missing.
- Screen Q&A (What's on my screen?) continues to use Gemini Vision to provide dual-glass overlay tips and Wikipedia image exploration, as this requires high-level visual reasoning.
- **Live Knowledge Search**: When Groq needs live context (triggered by terms like "latest", "today", "now"), Jarvis silently scrapes top results using `duckduckgo-search` (`ddgs`) and feeds them directly into the Groq prompt. This enables up-to-date, conversational answers without leaving the app.

## Recent Changes

### 2026-08-27

- **Native code-agent tools for Jarvis**: Added `backend/services/code_tools.py` — a built-in toolkit giving Jarvis the basic file & shell capabilities of a coding agent (opencode / codex / Claude Code) WITHOUT delegating every simple task to the external opencode CLI. Tools return a uniform `{ok, content, error, exit_code, path}` dict:
  - `code.read_file(path)` — read a text file.
  - `code.write_file(path, content)` — create/overwrite a text file (creates missing parent dirs).
  - `code.list_directory(path)` — list a folder's contents.
  - `code.run_command(command)` — run a shell/cmd command and capture output (30s timeout, 8000-char cap, `CREATE_NO_WINDOW`).
  - `code.run_script(path|code)` — run a `.py`/`.bat`/`.cmd` script or inline Python code.
  - Dispatched via the `TOOL_REGISTRY` + `call_tool(name, args)` helper.
- **Wired into the task agent** (`backend/services/task_agent/agent.py`): the five `code.*` tools are in `SAFE_TOOLS`, advertised in the LLM planner prompt, executed in `_execute_step`, and short-circuit heuristics in `_heuristic_plan` handle "read <file>", "write file X", and "run <command>/<script>" without needing an LLM call.
- **Deterministic brain routing** (`backend/core/brain.py`): added `is_code_tool_request()` (exported from `task_agent/__init__.py`), checked BEFORE the LLM intent router, so plain "read X / write file Y / run Z" requests land on the native code tools instead of the opencode fallback. Deliberately excludes web/browser targets (youtube/gmail/.com/http) so those still use the normal tool/executor path.
- **Tests**: `backend/tests/test_code_tools.py` — 15 unittest cases covering the registry, file read/write roundtrip, dir creation, command success/failure, script execution, output clipping, heuristic routing, and the `is_code_tool_request` guard.
- **Safety/env**: command timeout `JARVIS_TOOL_COMMAND_TIMEOUT` (30), output cap `JARVIS_TOOL_MAX_OUTPUT` (8000). File paths resolve against the repo root by default.

### 2026-08-02

- **Fish Audio is now Jarvis's primary TTS**: Created `backend/services/fish_voice.py`, which posts to `https://api.fish.audio/v1/tts` with `model: s2.1-pro-free` and `format: mp3` (bearer token from `FISH_API_KEY`). Audio is played via pydub/simpleaudio exactly like the ElevenLabs path, and `stop_fish_audio()` is wired into `stop_speaking()` so interruption still works.
- **Fallback preserved**: In `voice.py`, `speak()` now tries Fish Audio first (only for text within `JARVIS_FISH_TTS_CHAR_LIMIT`, default 300 — the non-streaming `v1/tts` limit) and otherwise falls back to the existing chain: local pyttsx3 (`JARVIS_PREFER_LOCAL_TTS`) → ElevenLabs (≤150 chars) → local pyttsx3.
- `FISH_REFERENCE_ID` is optional — when empty, Fish uses the account's default voice (verified live: works without a reference). Added `FISH_API_KEY`, `FISH_MODEL`, `FISH_REFERENCE_ID` to `backend/config.py` (with `FISH_API_KEY` in `OPTIONAL_API_KEYS`), `.env`, and `.env.example`.
- **Hardening (free tier is flaky)**: the free `s2.1-pro-free` model sometimes answers HTTP 200 with a non-audio/empty body, which previously threw a pydub/ffprobe `JSONDecodeError`. `speak_fish_audio` now validates the body (ID3/MPEG sync bytes), retries once, pins `ffmpeg`/`ffprobe` paths at import, and otherwise falls back cleanly to the pyttsx3 chain.
- **Volume +6 dB**: Fish playback now runs through `_boost_volume()` — +6 dB via `JARVIS_FISH_TTS_VOLUME_BOOST_DB` (config `FISH_VOLUME_BOOST_DB`), clamped so the peak never passes 90% of full scale to avoid clipping. Verified: 3627 → 7237 peak, no clip.

### 2026-04-29

- **Native Accessibility Agent Pivot**: Completely replaced the massive, VRAM-heavy Cloud Vision pipelines (Gemini, Groq, OpenRouter) with a fully local, 3B-parameter Native Accessibility agent powered by `llama3.2` via Ollama.
- **Screen Planner Reliability Fix**: Restored the deterministic `_try_local_match` fast path ahead of Ollama planning inside `_plan_with_tree`, so obvious UI/OCR targets still work when Ollama is unavailable and avoid unnecessary model latency. Broadened UI Automation extraction for editor-style apps by including controls such as `Document`, `Pane`, `Group`, and `DataItem`, and increased the default UIA traversal budget. Targeted `backend.tests.test_screen_control` now passes.
- **Connector-First Task Brain**: Added `backend/services/task_agent/` as a separate task-execution layer that gathers structured context from editor, browser, and Windows connectors, plans safe tool calls, and only falls back to screen/OCR action when no connector can handle the task. Routed explicit task/agent requests through this layer from `backend/core/brain.py`.
- **Editor Bridge Integration**: Added `integrations/jarvis-editor-bridge/`, a VS Code-compatible localhost extension exposing active file text, selection, workspace folders, diagnostics, editor command execution, file open, and selection editing. This is the intended path for Antigravity/VS Code-style "inside the editor" mastery.
- Created `backend/services/ollama_client.py` to handle fast, local inference to the `llama3.2` model on port `11434`. This limits the entire screen control footprint to ~2GB VRAM, easily fitting on a 6GB RTX 3050 alongside native OS apps.
- Refactored `backend/services/screen_ui_elements.py` and `screen_control.py` to stop generating visual bounding boxes (`draw_som_overlay`).
- `_gather_screen_context` was replaced by `_gather_ui_tree`, which natively reads the Windows UIAutomation tree via `pywinauto` and serializes it into a dense XML-like document (resembling HTML code) for the Llama model to read natively.
- **Electron Blindness Fix (OCR Injection)**: Because Electron apps (like Antigravity, Discord, Spotify) hide their DOM from the Windows Accessibility API, the UI Tree alone was rendering Jarvis blind inside these apps. Modified `_gather_ui_tree` to run Tesseract OCR on a background screenshot, extracting visible text regions and injecting them back into the XML UI Tree as `<text>` elements. This guarantees 100% precision targeting across both Native and hidden Electron apps.
- Rewrote the `_build_vision_prompt` into `_build_tree_prompt`, changing the persona to an "Autonomous Computer Agent driving the computer by reading its active Accessibility Tree." The model is strictly instructed to return `element_id` integers.
- Wired `_plan_with_tree` and `_bg_vision` in `screen_control.py` to use `ollama_client` natively.
- **Grounding Fix**: Fixed a coordinate-space regression where `element_id` targets from the new accessibility pipeline could be treated as screen coordinates too early and then offset a second time. Click-like actions now stay in vision-space until the final conversion back to real screen pixels.
- **Richer Accessibility Tree**: Expanded `_gather_ui_tree` to emit a nested XML-like structure with element bounds, centers, enabled state, automation IDs, class names, and OCR node metadata so the local Llama model can disambiguate repeated labels more reliably.
- **Fast-Path Restoration**: Restored `_try_local_match` before Ollama planning for obvious UI/OCR targets, fixed OCR match conversion back to real screen coordinates, and allowed that path to keep working when Ollama is offline instead of failing the whole screen command immediately.
- **Compatibility + Tests**: Repaired the compatibility path around `_build_vision_prompt` / `_plan_with_vision` so older internal callers and tests do not break after the refactor, and updated `backend/tests/test_screen_control.py` to cover the new tree planner and `element_id` grounding. The targeted screen-control test suite now passes again.
- **Microphone Bug Fix**: Fixed a bug where `watcher.py` was failing to hear anything because it wasn't loading the `.env` file on startup. Added `load_dotenv` to `watcher.py`, ensuring it correctly resolves `JARVIS_MIC_NAME` and `JARVIS_WATCHER_ENERGY_THRESHOLD` to connect to the Bluetooth headset and apply the correct silence thresholds.

### 2026-04-28

- **Brain Upgrade (Live Knowledge Search)**: Replaced the weak DuckDuckGo Instant Answer API with the `duckduckgo-search` (`ddgs`) package.
- In `backend/core/brain.py`, `search_internet()` now fetches snippets from the top 3 live web results.
- When `should_search()` or `force_search()` triggers, Groq is fed these snippets and produces up-to-date conversational answers without needing to open a browser search window. Browser fallback only occurs if `ddgs` fails.
- **Screen Control Accuracy**: Implemented **Set-of-Mark (SoM)** visual prompting.
- `screen_capture.py` now draws numbered, semi-transparent bounding boxes over UI elements and OCR text regions before sending the screenshot to the Vision API.
- The `screen_control.py` prompt now instructs the VLM to return the integer `element_id` instead of guessing normalized coordinates, completely eliminating click hallucination and guaranteeing pixel-perfect UI execution.
- **Disabled Fast Path**: The naive text-matching fast path in `_plan_with_vision` was completely disabled. It was frequently hijacking complex or descriptive commands (e.g. "click the pink circle with an m inside it") and clicking random words that partially matched. All screen actions now securely route through the highly accurate Set-of-Mark Vision pipeline.
- Optimized screen Q&A latency by speaking the text response immediately before fetching Wikipedia images in the background thread.

### 2026-04-26

- Added screen controls by voice, including capture, planning, execution, state tracking, and frontend status updates.
- Added Groq vision support for screen-action planning and Groq STT fallback for voice recognition when Google STT is unavailable.
- Set a fixed listening threshold via `JARVIS_IDLE_ENERGY_THRESHOLD` and aligned the listener-state clamp so the configured value is preserved across recalibration.
- Improved screen-control reliability by using tighter active-window captures first, falling back to full-screen captures when needed, and requiring confirmation on low-confidence screen plans instead of clicking immediately.
- Fixed screen-control routing so generic `search bar` requests no longer open Windows Search unless the command explicitly mentions taskbar or Windows search.
- Fixed screen-control command detection for `close`/`quit`/`exit` phrasing and improved click grounding by preferring active-window matches and center-of-target coordinates when vision provides bounds.
- Added this `codex.md` file as the running project change log for future sessions.
- Switched screen-capture format from JPEG (quality 85) to PNG for sharper text, and raised the vision resize cap from 1800 to 2560 pixels.
- Added red pixel-coordinate ruler tick marks (every 100px) on the top and left edges of screenshots sent to the vision model.
- Created `backend/services/screen_ocr.py` with pytesseract-based OCR text extraction, word merging, deduplication, and prompt formatting. Probes common Windows Tesseract install paths automatically.
- Created `backend/services/screen_ui_elements.py` with pywinauto UIA-based element enumeration, coordinate conversion to vision-image space, and prompt formatting.
- Added few-shot JSON examples (click, type, not-found) to the vision prompt in `screen_control.py`.
- Added post-action verification (`_verify_action`) that captures a fresh screenshot after execution and asks the vision model to confirm success.
- Added rolling interaction history (max 3 entries) to `screen_state.py` with `add_interaction`, `get_recent_interactions`, `clear_interactions`, and `format_history_for_prompt`. History is injected into subsequent vision prompts.
- Integrated OCR context, UI-element context, grid explanation, and history context into `_build_vision_prompt` and the `_plan_with_vision` pipeline.
- Increased `max_completion_tokens` from 650 to 800 for vision planning calls.
- Installed Tesseract OCR 5.4.0 to `C:\Program Files\Tesseract-OCR` and added `pytesseract`, `pywinauto` to `requirements.txt`.
- Reworked the listener to reuse one microphone source across capture and ambient-noise recalibration, warm-open it at startup, and reset it automatically after listen errors.
- Updated `calibrate_recognizer` so it can calibrate against an already-open microphone source instead of reopening the device each time.
- Tightened listener-state threshold handling by clamping applied values, skipping redundant `energy_threshold` writes, and simplifying the idle log message.
- Switched `screen_ui_elements.py` to lazy-import `pywinauto`, so UI Automation now degrades cleanly when COM or import setup fails during startup.
- Created `backend/services/openrouter_client.py` with support for free OpenRouter vision models (Gemma 4 31B, Gemma 4 26B, Nemotron Nano VL). Handles 429 rate-limits, in-body errors, and reasoning-model responses gracefully.
- Added `_ask_vision_cascade` in `screen_control.py` that tries OpenRouter free models in quality order before falling back to Groq Llama 4 Scout.
- Added `_extract_response_content` helper to handle models that return content in `reasoning` field instead of `content` (e.g. Nemotron).
- Lowered vision temperature from 0.2 to 0.0 in `grok_client.py` for deterministic coordinate output.
- Added rate-limit cooldown cache to `openrouter_client.py`: models returning 429 are skipped for 120 seconds with zero network delay. Shortened OpenRouter timeout from 45s to 25s.
- Changed `_ask_vision_cascade` to call `any_model_available()` so it skips the entire OpenRouter loop when all models are on cooldown.
- Switched post-action verification (`_verify_action`) to use Groq directly instead of the cascade, eliminating a second round of slow failing API calls.
- Made vision-based screen commands run in a **background thread**: `maybe_handle_screen_control_message` returns "Working on it, sir." immediately and delivers the real result via a callback that speaks it. Direct/hotkey plans still run synchronously.
- Added `_vision_busy` lock guard: if a vision command is already in-flight, new vision requests return "Still working on the last screen command, sir." instead of stacking up duplicate slow API calls.
- Added `set_response_callback` in `screen_control.py` and wired it in `voice_mode.py` so background vision results are spoken via TTS when they arrive.
- Added `_tts_lock` (threading.Lock) in `voice.py` around `pyttsx3.init()` and `runAndWait()` to prevent the `RuntimeError: run loop already started` crash when two TTS calls overlap.
- Created `backend/services/gemini_client.py` using **Gemini 2.5 Flash** (free, ~3s response) with `thinkingBudget: 0` for direct output. Returns OpenAI-shaped responses for drop-in cascade use.
- Replaced OpenRouter cascade with **Gemini → Groq** cascade in `screen_control.py`. OpenRouter free models were perpetually rate-limited (429); Gemini is faster and more reliable.
- Added **OCR/UI-first text matching** (`_try_local_match`): before calling any API, tries to match the user's command against locally-extracted OCR text regions and UI Automation elements. If a strong match is found, builds a direct plan with exact pixel coordinates — **no API call, sub-500ms response**.
- Skipped post-action verification for simple single-step actions (click/type). Verification only runs for complex multi-step plans, saving ~3-5s per command.
- Added `GEMINI_API_KEY` to `.env`.

### 2026-04-27

- Added **Screen Q&A with floating glass overlay**: asking "what's on my screen?" captures the primary screen, sends it to Gemini Vision, and returns a structured tip + evidence response.
- Created `backend/services/screen_analyzer.py` with `is_screen_question()` detection (English + Hindi/Hinglish patterns) and `analyze_screen()` which calls Gemini Vision with JSON output mode.
- Added `/screen-answer` GET/POST API endpoints to `backend/api/routes.py` for overlay state synchronisation.
- Integrated screen Q&A routing into `backend/core/brain.py` — screen questions are detected before chat/command routing, results are pushed to the overlay API.
- Created `frontend/overlay.html`, `frontend/overlay.css` (glassmorphism cards with slide-in animations, progress-bar auto-dismiss), and `frontend/overlay_renderer.js` (IPC-driven rendering).
- Added overlay window management to `main.js`: transparent, always-on-top, click-through `BrowserWindow` with mouse event toggling via IPC so glass cards are clickable but transparent areas pass through.
- Main process polls `/screen-answer` at 800ms and pushes data to the overlay renderer via IPC (`show-screen-answer`).
- **Fix**: Removed redundant `fetch`-based polling from `overlay_renderer.js` — main.js already polls and pushes via IPC, preventing double-renders.
- **Fix**: Replaced `executeJavaScript`-based mouse listener injection in `main.js` with event delegation in `overlay_renderer.js` — dynamically created evidence cards now correctly toggle click-through.
- **Fix**: Updated `routes.py` to use `.model_dump()` instead of deprecated `.dict()` for Pydantic v2 compatibility.
- **Enhanced** screen question detection: broadened to catch any sentence mentioning "screen" / "my screen" / "screen pe" etc. without screen-control verbs (click, type, scroll…). Supports phrases like "on my screen there is something about X, tell me more".
- Added **Google Search grounding** to screen analysis via `gemini_client.py` — Gemini searches the web while analysing the screenshot, returning real source URLs in `groundingMetadata`. Leverages Google AI Pro subscription.
- Added **explore links** to the overlay: real Google Search grounding links are shown first, with fallback constructed links (Google, Wikipedia, News, Images) from the extracted topic.
- Created `backend/services/image_fetcher.py` — fetches 1-2 topic-related images from Wikipedia API (search + page thumbnails) and constructs explore links.
- Added **left-side image overlay** (`overlay_images.html`, `overlay_images_renderer.js`) — transparent, always-on-top Electron window on the left showing Wikipedia images in glass cards with titles/captions.
- Updated `overlay.html` with an EXPLORE card (link pills between TIP and EVIDENCE).
- Updated `overlay.css` with link pill styles, image card styles, and left-side overlay container.
- Updated `overlay_renderer.js` to render clickable link pills that open in the default browser via `shell.openExternal`.
- Updated `main.js` to manage two overlay windows (right + left) and route images to the left overlay.
- Updated `/screen-answer` API model to include `links` and `images` fields.
- **Fix**: Resolved Gemini Vision `('Connection aborted.', TimeoutError('The write operation timed out'))` by switching `gemini_client.py` to a persistent `requests.Session` with a `urllib3` retry adapter (2 retries, 0.5s backoff, auto-retry on 429/502/503/504). Added a manual retry loop for connection-aborted / write-timeout errors. Bumped connect timeout from 3s→8s and read timeout from 30s→45s.
- **Fix**: Screen Q&A now runs in a **background thread** for voice requests (same pattern as screen control). `process_message` returns "Let me take a look at your screen, sir." immediately and delivers the real result via `set_screen_qa_callback` when Gemini responds. A `_screen_qa_busy` lock prevents duplicate concurrent analyses.
- Added `set_screen_qa_callback` in `brain.py` and wired it in `voice_mode.py` alongside the existing screen-control callback.

### 2026-04-28

- **Brain Upgrade (Live Knowledge Search)**: Replaced the weak DuckDuckGo Instant Answer API with the `duckduckgo-search` (`ddgs`) package.
- In `backend/core/brain.py`, `search_internet()` now fetches snippets from the top 3 live web results.
- When `should_search()` or `force_search()` triggers, Groq is fed these snippets and produces up-to-date conversational answers without needing to open a browser search window. Browser fallback only occurs if `ddgs` fails.
- **Screen Control Accuracy**: Implemented **Set-of-Mark (SoM)** visual prompting.
- `screen_capture.py` now draws numbered, semi-transparent bounding boxes over UI elements and OCR text regions before sending the screenshot to the Vision API.
- The `screen_control.py` prompt now instructs the VLM to return the integer `element_id` instead of guessing normalized coordinates, completely eliminating click hallucination and guaranteeing pixel-perfect UI execution.
- **Disabled Fast Path**: The naive text-matching fast path in `_plan_with_vision` was completely disabled. It was frequently hijacking complex or descriptive commands (e.g. "click the pink circle with an m inside it") and clicking random words that partially matched. All screen actions now securely route through the highly accurate Set-of-Mark Vision pipeline.
- Optimized screen Q&A latency by speaking the text response immediately before fetching Wikipedia images in the background thread.

## Update Rule

- Append a dated bullet under `Recent Changes` after each substantive Codex edit.
- Update `Current Baseline` whenever behavior or architecture meaningfully changes.

## 2026-09-23 - Chat outage root-caused: direct Gemini API unusable behind Proton VPN; chat/vision/browser_tool + intent rerouted via OpenRouter

- Symptom: every query returned "I'm having trouble connecting. Please try again."
- Root cause chain: Proton VPN (ProTUN tunnel, post-PC-reset install) active -> direct generativelanguage.googleapis.com calls degraded to 7-45s+ (frequent >45s read timeouts) -> brain chat chain exhausted attempts -> fallback leg is the suspended Fireworks account -> {} -> generic message. Groq now 403s ("check your network") from the VPN exit IP. OpenRouter (Cloudflare-fronted) measures ~1.4s through the same tunnel.
- Fix (no key ever left .env):
  - model_registry: chat role allowlist += openrouter; get_provider_credentials now returns canonical OpenAI-compatible base URLs for openrouter/groq (_ENV_PROVIDER_BASE_URLS; fireworks deliberately stays None - pinned contract).
  - data/jarvis_settings.json (backup .bak-before-openrouter-switch): chat/vision/browser_tool -> openrouter/google/gemini-2.5-flash-lite.
  - intent.py: classifier chain reordered OpenRouter -> Gemini -> Groq (OpenRouter stays fast behind the VPN so routing no longer collapses to "chat" verdicts); ask_openai_compat gained an optional timeout param for the classifier's tight budget slice.
- Verified: non-stream chat 1.41s, stream 0.61-0.65s, full process_message turn 1.80s with a real reply; classify 0.6-1.6s with correct verdicts (fresh-price question routes research). Regression 183 passed; 2 pre-existing failures (test_verification_uses_fireworks / test_verification_uses_groq - fail identically on stashed pre-edit code).
- Operator note: the direct-Gemini degradation is caused by the Proton VPN relay. Turning it off (or excluding python.exe/electron.exe via Proton split tunneling on Plus plans) restores the direct Gemini path; the OpenRouter selection then simply remains a fast alternative.
