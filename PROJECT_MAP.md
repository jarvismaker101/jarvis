# Jarvis Assistant Project Map

This document is meant to give another model enough context to work on the repo without first re-discovering the architecture from scratch. Refreshed 2026-09-11 (G0–G11 Fable-5 audit remediation landed; F37–F45 screen grounding, F50–F52 process architecture & security included; the classifier-retirement step 5 remains).

## What This Project Is

Jarvis Assistant is a Windows-only desktop assistant with four tightly connected parts:

1. An Electron desktop UI in `frontend/` with `main.js` as the Electron main process.
2. A FastAPI backend in `backend/` that handles chat, command parsing, voice status APIs, research, and screen-control orchestration.
3. A continuous voice loop in `backend/voice_mode.py` for active hands-free conversation once Jarvis is running.
4. A passive wake-word watcher in `backend/watcher.py` that listens for wake phrases and launches the full stack.

The project is heavily local-machine oriented. It assumes:

- Windows APIs are available.
- The repo lives at `C:\Users\mayan\jarvis-assistant`.
- A Python virtual environment exists at `backend\venv`.
- Electron and Python processes are allowed to spawn child processes and drive local input devices.

## The One-Sentence Mental Model

Typed UI requests and spoken requests both end up in `backend.core.brain.process_message()`. That function races a speculative chat stream against a cloud intent classifier, applies deterministic safety nets (screen questions, fresh-info search, web-shaped tasks) over the classifier verdict, then routes to chat, tool actions, screen Q&A, research, browser-agent tasks, or memory commands, and returns a short English reply that may also be spoken aloud.

## Fable-5 Audit Remediation State

The Fable-5 audit's remediation plan groups work into G0–G11. State as of this refresh:

- **G0 (landed)** — quick wins: streaming/reasoning fixes, payload-safe lowercasing, research notes carry the question, vision-when-tree-empty (F45), env-sourced browser token.
- **G1 (landed)** — closed-loop execution core: `backend/services/task_result.py` (`TaskResult` with `completed|partial|failed|stopped|needs_input`, evidence, legacy-string compatibility), recovery limits, closed-loop plans in `task_agent/agent.py`.
- **G2 (landed)** — policy & consent backbone: `backend/services/approvals.py` (F18 plan-hash approvals), `backend/services/tool_policy.py` (F17 dispatch-bound tool validation + F21 secret masking), `backend/services/code_grants.py` (F22 scoped grants + change journal), `backend/services/jobs.py` (F20 per-job cancellation tokens).
- **G3 (landed)** — request identity & streaming: `backend/services/request_registry.py` + `/ask` event frames (`delta|replace|progress|completed|interrupted|error`), `/ask/status/{request_id}`, reconnect-resume without re-execution, pure cancellable speculation (`_ChatRacer`), publish-early-enrich-later screen answers.
- **G4 (landed)** — coding interface: `code.search/read_range/inspect_diff/apply_patch/run_checks` in `code_tools.py`; structured editor bridge renderers (F15) in `task_agent/agent.py`.
- **G5 (landed)** — research pipeline: question-carrying evidence schema, `backend/services/provenance.py` (observed vs checked provenance, F48), bounded parallelism (`test_g5_research_pipeline.py`).
- **G6 (landed)** — browser grounding: `backend/services/browser_session_broker.py` (profile/tab/origin identity), marks bound to real targets with epochs, Playwright input primitives, extended toolset.
- **G7 (landed)** — Windows screen grounding: `screen_geometry.py` (coordinate-space helpers, monitor bookkeeping, F44 same-frame rect comparisons), `screen_ui_elements.py` (F42 runtime-id re-resolution, F44 preserved hierarchy), `screen_capture.py` (F43 multi-monitor/DPI/last non-Jarvis target), F29 UIA-first fast path in `screen_control.py`, F45 image-only vision fallback.
- **G8 (landed)** — orchestrator migration behind the migration flag: `backend/services/orchestrator.py` (F02 native tool-use loop over the registry-resolved qwen3p7-plus planner; tool args validated; invalid output distinguished from a conversational answer; mutations only PROPOSED through the existing gates), `backend/services/capability_resolver.py` (F16 engine dispatch; availability ≠ permission; opencode opt-in only; browser recovery needs no CLI), `backend/services/context_envelope.py` (F46 bounded request-scoped envelope with explicit omissions), planner role + capability validation in `model_registry.py` (F49; wired into `task_agent._model_plan`). Enable with `JARVIS_ORCHESTRATOR_MODE=orchestrator`; the legacy classifier routing and all deterministic safety nets stay until parity (the screen-question net replays deterministically inside the orchestrator).
- **G9 (landed)** — memory & continuity: `backend/core/memory_store.py`, one SQLite store (FTS5 with graceful LIKE fallback) with `facts`/`entities` (F06: provenance, confidence, sensitivity, superseded-by revision chain, tombstoned scoped forgetting, bounded `memory_context` injection in `_build_chat_messages`), `events` (F07: every chat exchange, async reply and task result recorded bounded + secret-masked via `tool_policy.mask_secrets`, grounding "use the report you just researched" follow-ups), `commitments` (F10: explicitly-armed deadline/job-completed triggers with a state machine and scheduler daemon; delivery rides the same async-reply UI/speech path as background results — notification only, never computer control; calendar/file-watch triggers deliberately rejected as assumption-flagged), `skills` (F09: verified completions capture approval-gated candidates; only user-approved skills ground `_model_plan`; versioned + invalidated after repeated failures; webpage instructions are never promoted). Deterministic phrase ops (`remember that…`, `forget…`, `what do you remember about…`, `remind me to … in N minutes/at HH:MM`, `cancel the reminder`, `approve/retire the … skill`) run before any other routing and are gated by `JARVIS_MEMORY_ENABLED` (default on; empty store changes zero prompts).
- **G10 (landed)** — voice runtime: `backend/services/audio_actor.py` (F32 single playback owner with a generation token — chunks carry the generation they were produced for and stale-generation PCM is dropped at every await/play boundary; the PCM ring + spoken cursor make interrupted resumes exact; `abort()` stops **this** stream, never `sd.stop()` for everyone; audio cache is identity-keyed by model+reference+format+text), `backend/services/echo_cancel.py` (F33 AEC3 reference path: every chunk that reaches the device feeds a monotonic reference ring resampled to 16k, the listener's VAD/onset window is AEC-cancelled before it can look like a barge-in — `py-webrtc-aec3` when installed, otherwise a loud no-op at the [ASSUMPTION] boundary), `backend/services/transcript_stabilizer.py` (F34 local-agreement conversation windows: finals commit immediately, agreeing overlapping partials after N windows, and `listen()` only ever returns *committed* transcripts so unstable partials can never trigger an action). F36 wake/command separation in `backend/services/wake_engine.py` + `watcher.py`: the phrase tail after the wake window is now extracted and forwarded to `/ask` ("wake up jarvis and search for cats" acts on both parts), an optional pre-roll ring (`JARVIS_WAKE_PRE_ROLL_SECONDS`) plus online Whisper verification of false positives (`JARVIS_WAKE_ONLINE_VERIFY`), and `openwakeword` keyword spotting when installed (`JARVIS_WAKE_ENGINE`) with graceful fallback to the fuzzy path. Fish playback feeds the actor + reference path on every device write; `test_voice_latency.py` cache assertions were updated to the identity-key contract. AEC tuning and keyword-spot quality stay [ASSUMPTION]-flagged (need the real speaker/headset recordings).
- **G11 (landed)** — process architecture & security (audit F50/F51/F52):

  * **F50 backend sole owner of intelligence state** (`backend/voice_mode.py`): the voice process is now a pure I/O worker — it no longer imports or executes `backend.core.brain` (the audit's "second, never-authoritative copy" bug). Every utterance is submitted to the ONE backend runtime via the authenticated `/ask/stream` SSE contract (`speak=False`, `origin=voice`); streamed deltas feed the same `StreamSpeaker`. Listening state is **published** (`POST /voice-state/publish`, authed) and the backend serves `/voice-state` + `/ui-state` from that snapshot instead of its empty `listener_state` module copy. Task state is **polled** from `/ui-state` (`backend_task_running`, 1s TTL, last-known-kept) — never a module copy. Stop/stop-research/cancel-approval controls route through the authed HTTP control plane (`/task/stop`, `/speak/stop`, `/approvals/reset`). A negation-aware `is_stop_research` lives in the worker (mirrors brain). `backend/api/routes.py`: `Query.speak`/`Query.origin` flags gate backend double-speaking; `_run_request_worker` gains a `speak_terminal` parameter so a voice submission stays silent; `/voice-state/publish` (authed) merges into `/voice-state` and `/ui-state` (monotonic `state_seq`); `/approvals/reset` invalidates pending consent + clears the pending screen plan + interrupts live registered requests before safe recovery.
  * **F51 sandboxed renderers + authenticated localhost** (`frontend/preload.js`, `main.js`, `frontend/capsule_main.js`, and all 5 renderer JS files): every window is now `contextIsolation:true, nodeIntegration:false, sandbox:true` with the minimal `contextBridge` preload bridge — validated IPC channels only, http/https-validated `openExternal`, and a token-attaching `jarvisAPI.backend()` proxy so the backend token rides on every call (overlays, which render web-derived content, can **never** read the token). Renderer-initiated navigation is denied and `window.open` is denied. `backend/services/local_auth.py` (`X-Jarvis-Token`, per-launch secret, constant-time compare) enforces the token on every mutating endpoint; GET `/health`, `/ui-state`, `/voice-state`, `/screen-answer`, `/research-*`, `/ask/status/*` stay open for liveness. `backend/services/runtime_identity.py` writes an atomic `data/runtime/backend-instance.json` stamp (pid + instance_id + protocol + build) at startup, and `/health` exposes it. A missing token disables enforcement entirely (dev/backward-compatible). `frontend/*.html` carry a `Content-Security-Policy` meta tag. `BRAVE_MCP_TOKEN` is already env-based in source (rotation is a user step on `.env`/`~/.config/opencode/opencode.jsonc`).
  * **F52 supervised, versioned, observable runtime** (`main.js`, `watcher.py`): child processes are spawned with stdio piped and drained into bounded `data/logs/<label>.log` (cap + half-rotate) — never a blocking full pipe. `registerManagedProcess` retains the handle and restarts required workers within a restart **budget** (2 backend, 1 voice) on unexpected exit; deliberate stops never count. A backend replacement best-effort POSTs `/approvals/reset` first (invalidate stale approvals before recovery). `watcher.py` mints the per-launch token, injects it into every child (backend/voice/electron), and verifies the backend via **identity** (instance stamp vs. `/health`) instead of the mere existence of `/research-result`; it invalidates the live backend's approvals before replacing a stale one and exposes distinct warm-sleep (`/stop`, authed) and full-shutdown (`/shutdown`, authed) control semantics.

## Current Runtime Topology

There are three practical ways the app runs:

### 1. Direct Electron startup

Main file: `main.js`

Flow:

- Electron creates the window and loads `frontend/index.html`.
- Unless `JARVIS_EXTERNAL_RUNTIME=1` is already set, Electron starts:
  - FastAPI with `python -m uvicorn backend.main:app --host 127.0.0.1 --port 9999` (port from `JARVIS_BACKEND_PORT`, default `9999`)
  - Voice mode with `python -m backend.voice_mode` (only after the backend reports ready on `/health`)

This is the normal `npm start` path when not launched by the watcher.

### 2. Combined launcher batch file

Main file: `run_jarvis.bat`

Flow:

- Starts the FastAPI backend in a visible terminal.
- Waits 3 seconds.
- Starts `backend.voice_mode` in another terminal.
- Waits 2 seconds.
- Starts Electron with `npm start`.

### 3. Wake-word watcher mode

Main files: `run_watcher.bat`, `backend/watcher.py`

Flow:

- A lightweight watcher listens for wake phrases like `wake up jarvis` or `utho jarvis`.
- When it detects a wake match, it launches:
  - the FastAPI backend
  - the voice mode process
  - Electron
- It sets `JARVIS_EXTERNAL_RUNTIME=1` for Electron so `main.js` does not start the backend and voice mode a second time.
- Before spawning, the watcher kills any stale backend listening on the configured port (`watcher.py` `_pids_on_port(BACKEND_PORT)` + `taskkill`), so a relaunch always owns port 9999.
- While Jarvis is running, the watcher pauses.
- When Electron exits and the backend is no longer on the configured port (`JARVIS_BACKEND_PORT`, default `9999`), the watcher resumes listening.

## End-to-End Request Flows

### Typed UI flow

Path:

`frontend/renderer.js`
-> `POST /ask`
-> `backend/api/routes.py`
-> `backend/core/brain.py` (`process_message`)
-> intent classification (`backend/services/intent.py`) raced against a speculative chat stream (`_ChatRacer`)
-> deterministic safety nets may rewrite the verdict (see brain.py section)
-> one of:
- chat via the model-registry-selected provider chain (`gemini_client` / `fireworks_client` / custom OpenAI-compatible)
- tool actions via `backend/core/executor.py`
- screen Q&A via `backend/services/screen_analyzer.py`
- research via `backend/services/quick_search.py` / `research_service.py`
- browser-agent task via `backend/services/browser_agent.py` (confirmation-gated)
- screen control via `backend/services/screen_control.py`
- memory reset via `backend/core/memory.py`
-> response returned to UI
-> same response is also spoken through `backend/services/voice.py`

Notes:

- The UI has `chat` mode and `command` mode.
- In command mode, the renderer prefixes the request with `command ` before sending it to `/ask`.
- The `/ask` route deduplicates identical requests received within 1 second.

### Intent classification + chat race (inside process_message)

- `_ChatRacer` (`brain.py:772`) starts building chat messages and streaming the speculative reply the moment a message arrives.
- `classify_intent` (`backend/services/intent.py:180`) asks Gemini 3.5 Flash Lite first (3s timeout, no retry), falls back once to Qwen 3.6 27B on Groq, and lands on a `chat` verdict if both fail.
- Any non-chat verdict cancels the racer (`brain.py:1887` holdback); a chat verdict adopts the already-running stream, so plain chat answers start with near-zero router latency.

### Voice conversation flow

Path:

`backend/voice_mode.py`
-> `backend/services/listener.py`
-> recognized text
-> `backend/core/brain.py`
-> response text
-> `backend/services/voice.py`
-> optional sync back to backend voice-log API
-> `frontend/renderer.js` polls and mirrors the exchange in the UI

Voice mode is split into two loops:

- `listener_thread()`
  - continuously captures speech with `listen()`
  - if Jarvis is currently speaking, only interruption commands are honored
  - otherwise recognized text is pushed into a queue
- `brain_thread()`
  - consumes queued text
  - handles shutdown, continue/resume, and "normal setup" shortcuts
  - otherwise routes the message through `process_message(..., from_voice=True, voice_compact=True)`

### Wake-word flow

Path:

`backend/watcher.py`
-> microphone capture
-> Local GPU-accelerated faster-whisper (medium) model (with Google/Groq fallback)
-> fuzzy wake-phrase matching
-> spawn backend + voice + Electron

The watcher is separate from active voice mode. It is optimized for short wake phrases, not full conversation, running locally on CUDA GPU.

### Screen Q&A flow

Path:

message
-> `backend/core/brain.py` (screen/region verdict from the classifier, or the deterministic screen-question net)
-> `backend/services/screen_analyzer.py`
-> `_ask_screen_vision_cascade` (`screen_analyzer.py:235`): registry-selected vision provider first (currently Fireworks `qwen3p7-plus` per `data/jarvis_settings.json`), then fall-through Gemini (`GEMINI_VISION_MODEL`, default `gemini-3.5-flash-lite`), then Groq Qwen (`qwen/qwen3.6-27b`)
-> fetches topic images via `backend/services/image_fetcher.py`
-> pushes structured JSON to `/screen-answer`
-> `main.js` polls and routes to `overlay_renderer.js` and `overlay_images_renderer.js`
-> displays floating glassmorphism overlays on desktop

### Research / quick-search flow

Path:

message (research verdict, or the fresh-info auto-search net)
-> `brain.handle_research_intent` (`brain.py:1326`) â€” acks immediately, works in a background thread
-> default mode: `backend/services/quick_search.py` â€” headed Brave Search on the shared research Chrome profile, reads the AI Overview answer box (Ask-tab fallback, container-first extraction), returns a short spoken+text summary; no site scraping
-> deepsearch mode (explicit "deepsearch" keyword): AI Overview PLUS `backend/services/research_service.py` multi-site flow â€” Brave search, scrape top-N results, Gemini per-site notes, one deduped consolidated report pushed to the glass overlay via `/research-result` and saved under `data/research_reports/`
-> research is interruptible ("stop the research") via a stop event checked between sites

### Screen-control flow

Path:

message
-> `backend/core/brain.py`
-> `backend/services/screen_control.py`
-> either:
- direct deterministic plan
- vision plan (registry vision model cascade) from an XML UI Accessibility Tree (UI Automation + OCR)
-> `backend/services/screen_executor.py`
-> mouse / keyboard / window-control actions on Windows

This subsystem is off by default and must be enabled by saying or typing a phrase like `turn on screen controls`.

## High-Value Files and Their Real Roles

### Electron and UI

- `main.js`
  - Electron main process.
  - Owns app startup and stop IPC behavior.
  - Starts backend and voice mode unless the watcher already did (`JARVIS_EXTERNAL_RUNTIME=1`).
  - Stop behavior is targeted: tracked child PIDs are killed with `taskkill /PID <pid> /T /F`; in external-runtime mode it asks the watcher via HTTP `/stop` first, and only if that fails runs `fallbackExternalCleanup()` (kills the port-9999 backend by port ownership plus voice/whisper-daemon processes by command line). It does NOT kill port 9999 at startup â€” that stale-backend kill lives in the watcher's launch path.
  - Manages floating glassmorphism overlay windows for Screen Q&A, polling `/screen-answer` to update them, plus a research overlay polling `/research-result`.

- `frontend/overlay.html`, `overlay.css`, `overlay_renderer.js`, etc.
  - Transparent, always-on-top windows for the Screen Q&A feature.
  - Shows tips, evidence, topic images, and search grounding links.

- `frontend/index.html`
  - Single-window shell for the desktop app.
  - Shows status, chat history, a text input with chat/command modes, and the model-switcher sidebar.

- `frontend/renderer.js`
  - Sends `/ask` requests.
  - Adds the `command ` prefix in command mode.
  - Polls the fused `/ui-state` endpoint â€” 250 ms while active (hearing/thinking/speaking), 600 ms when idle, exponential backoff up to 3 s on failures.
  - Updates the UI badge/ring for `listening`, `hearing`, `thinking`, `speaking`, or `offline`.
  - Shows screen-control status and whether a screen action is awaiting confirmation.

- `frontend/style.css`
  - Pure styling. No logic.

### FastAPI backend

- `backend/main.py`
  - Primary backend entrypoint used by the launch commands.
  - Creates the FastAPI app.
  - Adds permissive CORS.
  - Includes routes from `backend/api/routes.py`.

- `backend/app.py`
  - Secondary minimal FastAPI entrypoint.
  - Also includes the router, but current launch scripts use `backend.main:app`, not this file.

- `backend/api/routes.py`
  - `POST /ask`
    - deduplicates near-identical rapid repeats
    - calls `process_message(...)`
    - speaks the reply in a background thread
  - `POST /update-voice-log`
    - stores the latest voice input/reply pair
  - `GET /voice-log`
    - returns only the most recent voice exchange
  - `GET /voice-state`
    - merges voice state from `backend.listener_state`
    - merges screen-control state from `backend.services.screen_state`
  - `GET /ui-state`
    - fused endpoint returning `{state, voice_log}` in one response â€” halves renderer polling
  - `GET /screen-answer` / `POST /screen-answer`
    - handles state synchronization for the floating Screen Q&A overlays
  - `GET /research-result` / `POST /research-result`
    - deep-research report delivery for the research overlay
  - `GET/POST /settings`
    - model-registry selections (chat/tts/vision/browser_tool roles) and custom providers
  - `POST /task/stop`
    - cancels a running browser-agent task

### Core decision layer

- `backend/core/brain.py`
  - This is the main router for user intent.
  - Most important file in the repo for behavior changes.

Routing order inside `process_message` (non-`command` messages):

1. Pending confirmation/follow-up gates (task confirm, browser clarification).
2. `maybe_handle_screen_control_message(...)` (bypassed if prefixed with `command`).
3. Speculative `_ChatRacer` start, then `classify_intent(msg)` (`brain.py:1878`).
4. **Screen-question safety net** (`brain.py:1879`): if the verdict is `chat` or `research` and `is_screen_question(msg)` (deterministic regex in `screen_analyzer.py`), the verdict is rewritten to `region`/`screen`. tool/task verdicts are exempt because they carry structured steps the upgrade would discard.
5. Racer holdback: any non-chat verdict cancels the speculative stream.
6. **Fresh-info auto-search net** (`brain.py:1892`): a `chat` verdict that hits `should_search()` (pricing/cost/latest/news keywords, `brain.py:274`) and is question-shaped but not greeting-like reroutes to `handle_research_intent` (quick-search tier), so stale chat answers are never served for current-world facts.
7. Tool steps branch, research branch, screen/region branch (screen Q&A), task branch (opencode/browser-agent handoff, confirmation-gated).
8. **Web-task routing net** (`brain.py:2029`): the broad `is_task_request` heuristic, when `TASK_ENGINE=browser_agent`, checks `is_web_shaped_task` (`backend/services/web_task_routing.py`: web hint + interaction verb, no local hint) and hands the task to the confirmation-gated browser-agent path instead of the raw task-message path.
9. Memory reset phrases, command-mode parsing, legacy chat path.

Important chat behavior:

- The chat model is resolved per message by the model registry (`_resolve_chat_model`, `brain.py:576`) â€” a UI model switch takes effect on the next reply, no restart.
- Provider chain: registry-selected provider first (currently Fireworks `qwen3p7-plus` per `data/jarvis_settings.json`; env default is Gemini `GEMINI_BRAIN_MODEL`, default `gemini-3.5-flash-lite`), with same-model non-stream retry before any provider fallback, then the Gemini -> Fireworks fallback chain (`_stream_chat_deltas` `brain.py:622`, `_ask_chat_nonstream` `brain.py:723`). Provider fallbacks are recorded for UI warning (`_record_chat_fallback`, keys scrubbed).
- Groq is NOT the chat provider: `grok_client.DEFAULT_MODEL` (`llama-3.3-70b-versatile`) is retired upstream (404) and no longer used for chat.
- Detects Hindi/Hinglish vs English and still forces English replies.
- Uses short speech-oriented prompts when `voice_compact=True`.
- Legacy search injection: `search_internet()` (`brain.py:487`, `ddgs`) can still inject live snippets into chat context, but the primary lookup path is the Brave-based quick-search pipeline (see Research flow).

Important command behavior:

- Parses `command ...` text into structured actions.
- Supports:
  - opening websites
  - Google searches
  - YouTube play commands
- Supports browser preference detection for Chrome, Edge, and Brave.
- Supports chaining with `and`.
- Supports simple repetition counts such as `open youtube two times`.

- `backend/core/executor.py`
  - Executes the structured browser actions and local application launches from command mode.
  - Implements `launch_app(app_name)` to dynamically resolve and open desktop applications (using registry system aliases, browser protocols, and recursively searching user/system Start Menu and Desktop shortcuts) with a graceful fallback to a website URL.
  - Uses hardcoded executable paths for Chrome, Edge, and Brave where available.
  - Falls back to the system browser through Python's `webbrowser`.
  - For YouTube play, it scrapes the first `/watch?v=` match from the YouTube results page.

- `backend/core/memory.py`
  - Conversation history capped at 20 messages, in-memory for the session.
  - Also mirrored to `data/conversation_history.json` so context survives backend restarts.
  - Persistence is best-effort; a corrupt/missing file never breaks chat.

### Intent router and deterministic nets

- `backend/services/intent.py`
  - `classify_intent(message, timeout_ms=3500)` routes every message to chat/tool/screen/region/research/task â€” no keyword pre-check.
  - Chain: Gemini 3.5 Flash Lite primary (`GEMINI_INTENT_MODEL` / `GEMINI_BRAIN_MODEL`, 3s timeout, `no_retry`) -> single fallback Qwen 3.6 27B on Groq (`GROQ_INTENT_MODEL` / `GROQ_VISION_MODEL`) -> `chat` verdict on total failure. Chat is therefore never broken, but classifier throttles silently degrade to chat verdicts â€” which is exactly why the deterministic nets in `brain.py` exist.
  - The prompt is length-capped (test asserts <2000 chars) and covers English/Hindi/Hinglish meaning, not keywords.

- `backend/services/web_task_routing.py`
  - `is_web_shaped_task(text)`: web hint (`website|chrome|edge|.com|.to|httpâ€¦`) AND interaction verb (`search|play|click|loginâ€¦`) AND no local hint (`file|folder|vscode|pythonâ€¦`). Feeds the brain's web-task routing net.

- `backend/services/screen_analyzer.py`
  - `is_screen_question` (two-tier: exact regex incl. Hinglish, then broad screen-ref + question cue minus control verbs) and `is_region_question` power the screen-question net.
  - `analyze_screen` captures the screen and runs the vision cascade (see Screen Q&A flow).

### Quick-search and research pipeline

- `backend/services/quick_search.py`
  - Fast default websearch: AI Overview first, no site scraping.
  - Opens headed Brave Search on the shared research Chrome profile (`data/chrome_profile_jarvis`, human-looking, captcha-solvable; `JARVIS_RESEARCH_CHANNEL`/`JARVIS_RESEARCH_PROFILE` override).
  - Reads the AI answer from the rendered DOM (`chatllm-answer`/`chatllm-content` selectors) with a completion poll; Ask-tab fallback (`_ASK_TAB_SELECTORS`) with container-first answer extraction (`_ASK_ANSWER_CONTAINER_SELECTORS`, guarded `rfind(query)` fallback) and a trailing-chrome cut (disclaimer/follow-up labels/site crumbs stripped).
  - Degrades lightly to the top result's snippet when there is no AI Overview.

- `backend/services/research_service.py`
  - Deep research: Brave Search on the real Chrome profile headed, scrape top-N results (no ranking by us), Gemini per-site notes, one deduped consolidated summary; YouTube links become related videos when descriptions are empty.
  - Returns both a short spoken summary and a long markdown report (glass overlay + saved file under `data/research_reports/`).
  - Interruptible via a stop event checked between site fetches.

### Task engines

- `backend/services/browser_agent.py`
  - The default task engine (`config.TASK_ENGINE == 'browser_agent'`): a strong model drives the brave-control MCP daemon with vision grounding.
  - Model comes from the registry `browser_tool` role (env default Fireworks `accounts/fireworks/models/qwen3p7-plus`; `data/jarvis_settings.json` currently overrides to `deepseek-v4-flash-vision-exp`), with `MAX_STEPS` (50) / `TIMEOUT` (480s) backstops and a stop event (`/task/stop`).
  - Every handoff is confirmation-gated in the brain before anything executes.

- `backend/services/brave_mcp_client.py`
  - Streamable-HTTP MCP client for the brave-control daemon (`BRAVE_MCP_PORT`, default 9570).

- `backend/services/opencode_client.py`
  - The alternative engine (`JARVIS_TASK_ENGINE=opencode`): hard guards refuse to spawn anything unless that engine is selected.

- `backend/services/task_agent/agent.py`
  - Connector-first task planning brain (heuristics -> model plan -> Windows control fallback).
  - The model planner (`_model_plan`, `agent.py:578`) calls `ask_fireworks` (temperature 0.1, max_tokens 650, no model arg -> `FIREWORKS_MODEL` default `accounts/fireworks/models/deepseek-v4-flash-0731`). It no longer uses Groq â€” the old `llama-3.3-70b-versatile` planner 404'd after the model was retired.
  - BOTH no-plan fallbacks (empty-steps screen_action append in `_normalize_plan`, and the `plan_task` Windows-control fallback) are confirmation-gated (`requires_confirmation=True`), so an unstructured improvisation never runs unconfirmed.
  - Gathers structured context from editor, browser, and Windows connectors; uses UI Automation / OCR screen actions only as a fallback.

- `backend/services/task_agent/connectors/editor_bridge.py`
  - Talks to a local VS Code-compatible editor bridge at `http://127.0.0.1:8765`.

- `backend/services/task_agent/connectors/browser_cdp.py`
  - Detects Chrome DevTools Protocol at `http://127.0.0.1:9222`.

- `backend/services/task_agent/connectors/windows_connector.py`
  - Reads active-window metadata and UI Automation controls; bridges fallback actions into the screen-control planner.

- `integrations/jarvis-editor-bridge/`
  - Local VS Code-compatible extension that exposes structured editor state to Jarvis over localhost.

### LLM clients and model registry

- `backend/services/model_registry.py`
  - The single runtime source of truth for model selection across four roles: `chat`, `tts`, `vision`, `browser_tool`.
  - Persisted overrides live in `data/jarvis_settings.json` (set from the UI model switcher, immediate effect, no restart); missing/corrupt settings degrade to env defaults.
  - Env defaults per role: chat -> gemini/`GEMINI_CHAT_MODEL`; tts -> fish/`FISH_MODEL`; vision -> gemini/`GEMINI_MODEL`; browser_tool -> `BROWSER_AGENT_PROVIDER`/`BROWSER_AGENT_MODEL`.
  - Per-role provider allowlists; custom (user-added OpenAI-compatible) providers allowed for chat and browser_tool only; API keys never leave the module (masked listings).
  - Current live overrides: chat and vision on Fireworks `qwen3p7-plus`, browser_tool on Fireworks `deepseek-v4-flash-vision-exp`.

- `backend/services/gemini_client.py`
  - Google Gemini client â€” vision and the text brain, both in OpenAI-shaped format so cascade code is provider-agnostic.
  - `GEMINI_MODEL` (`GEMINI_VISION_MODEL`, default `gemini-3.5-flash-lite`) for vision; `GEMINI_CHAT_MODEL` (`GEMINI_BRAIN_MODEL`, default `gemini-3.5-flash-lite`) for chat/intent.
  - Supports Google Search grounding for screen Q&A.

- `backend/services/fireworks_client.py`
  - Fireworks chat client; `DEFAULT_MODEL` = `FIREWORKS_MODEL` (default `accounts/fireworks/models/deepseek-v4-flash-0731`); retry logic drops `reasoning_effort` when the error body indicates a thinking-only model.
  - Also provides streaming and vision entry points used by the chat and screen cascades.

- `backend/services/grok_client.py`
  - Despite the filename, this is a Groq API client, not xAI Grok.
  - `DEFAULT_MODEL` is still `llama-3.3-70b-versatile` in code, but that model is RETIRED upstream (404) â€” nothing routes chat through it anymore; callers pass explicit models.
  - `VISION_MODEL` default is `qwen/qwen3.6-27b` (used as the last-resort screen-Q&A vision fallback and as the intent-router fallback model).

- `backend/services/openrouter_client.py`
  - OpenRouter vision client (free vision models) for screen-control grounding and as an optional first vision provider in the cascade.

- `backend/services/openai_compat_client.py`
  - Chat client for user-added custom providers (any /v1-compatible endpoint).

- `backend/services/transcription.py`
  - Shared speech-to-text fallback logic. Google STT primary; Groq Whisper-style transcription (`whisper-large-v3-turbo`) as the network fallback.

- `backend/services/ollama_client.py`
  - Local inference to the `llama3.2` model on port `11434` for the accessibility screen-control planner.

### Active voice subsystem

- `backend/voice_mode.py`
  - Main always-on voice orchestrator once Jarvis is active.
  - Has custom phrase handling for:
    - stop speaking
    - shutdown
    - continue/resume
    - normal setup
  - The `normal setup` shortcut launches Brave, VS Code, WhatsApp, and Edge from hardcoded Windows locations.

- `backend/services/listener.py`
  - Active conversation microphone capture.
  - Uses `speech_recognition` with `stream=True`.
  - Uses `webrtcvad` to reject obvious noise and low-confidence captures.
  - Uses multilingual recognition across `JARVIS_STT_LANGUAGES`, default `en-IN,hi-IN`.
  - Uses Google Speech Recognition first, then falls back to Groq STT if Google requests fail.
  - Registers its recognizer with `backend/listener_state.py` so speaking/listening thresholds can be adjusted globally.

- `backend/services/audio_input.py`
  - Shared microphone-selection and threshold utilities.
  - Resolves microphone by:
    - `JARVIS_MIC_DEVICE_INDEX`
    - `JARVIS_MIC_NAME`
    - default input device
    - fallback name heuristics
  - Defines `FIXED_IDLE_ENERGY_THRESHOLD`.

- `backend/listener_state.py`
  - Global voice-state registry used across listener and TTS.
  - Tracks speaking/thinking/user-speaking booleans, active recognizer, energy threshold, remaining speech text, and speech timestamps.
  - `get_voice_state()` is what the UI ultimately renders.

- `backend/services/voice.py`
  - Text-to-speech orchestrator. The current ladder is easy to misunderstand:
    - **Fish Audio is tried first** (`speak_fish_audio`, streaming PCM via simpleaudio/WASAPI â€” works even from a hidden Electron child, with next-chunk prefetch pipelining to hide round-trips)
    - local `pyttsx3`/SAPI5 second (with a silent-playback sanity check)
    - ElevenLabs for short chunks (`JARVIS_REMOTE_TTS_CHAR_LIMIT`)
    - local SAPI5 again as the final safety net
  - Sentence chunking + background prefetch of the next chunk; interruption via `stop_speaking()` (generation bump + Fish PCM flush) for instant barge-in.

- `backend/services/fish_voice.py`
  - Fish Audio TTS client: registry `tts` role model resolution (env default `FISH_MODEL`, default `s2.1-pro-free`), PCM streaming playback, prefetch/warm-up, output-device selection.

- `backend/services/elevenlabs_voice.py`
  - Optional remote TTS provider (ElevenLabs API, MP3 via pydub).

- `backend/services/earcons.py`
  - Plays simple Windows beep patterns for ready/capture/reply cues.

### Wake watcher

- `backend/watcher.py`
  - Passive wake-word launcher.
  - Uses a dedicated recognizer with its own thresholds and fuzzy phrase matching.
  - Applies a fixed watcher energy threshold from `JARVIS_WATCHER_ENERGY_THRESHOLD`, clamped against the shared idle threshold.
  - Pre-loads Anaconda and Ollama CUDA v12 DLL search paths dynamically to enable local CUDA GPU speech transcription.
  - Uses a local GPU-accelerated `faster-whisper` (medium) model running with greedy decoding (temperature=0.0), VAD filtering, disabled repetition conditioning, and wake-word biasing prompts; a persistent `whisper_daemon.py` child keeps the model warm.
  - Uses Google STT with Groq fallback only as a backup.
  - Kills stale port-9999 backends before launching the stack, and boots the activity-tail console + brave MCP daemon in browser_agent mode.

### Screen-control subsystem

- `backend/services/screen_control.py`
  - Entry point for natural-language screen actions.
  - Important behavior:
    - screen controls are toggleable and off by default
    - pending risky or low-confidence actions require spoken/text confirmation
    - deterministic direct actions are attempted before vision planning

Direct actions it can parse without vision:

- open Start menu
- open Windows search
- minimize / maximize / restore active window
- scroll
- press keys / hotkeys
- type raw text

If the command is not directly parseable:

- It captures the active window's UI Automation Accessibility Tree (via pywinauto) and extracts visual OCR text (via Tesseract) to catch hidden Electron DOMs.
- It formats these into an XML-like document and asks the registry-selected vision model (local Llama 3.2 via Ollama remains an option).
- It expects strict JSON back containing the target `element_id`.
- It maps the `element_id` back to its exact screen coordinates.
- It executes the resulting steps if confidence is high enough.

Risk handling:

- risky actions like `delete`, `send`, `submit`, `purchase`, `close`, `quit`, and similar terms require confirmation
- low-confidence actions also require confirmation
- confirmation state is stored in `screen_state`

- `backend/services/screen_capture.py`
  - Captures either the foreground window or the primary monitor.
  - Uses Windows APIs plus `mss`.
  - Resizes the screenshot for vision model use, draws grid/SoM overlays, and returns a base64 data URL.

- `backend/services/screen_executor.py`
  - Executes low-level mouse, keyboard, scroll, and window-control steps.
  - Uses ctypes / Win32 `SendInput` for hardware mouse clicks, `SetForegroundWindow` for focus, and the `keyboard` package for typing and hotkeys.

- `backend/services/screen_state.py`
  - Stores whether screen controls are enabled and whether a pending plan is waiting for confirmation.

## State Model

Nearly all state is process-local module state; there are only three durable artifacts on disk.

### Memory state

Location: `backend/core/memory.py`

- Stores recent chat history (capped at 20).
- Mirrored best-effort to `data/conversation_history.json`.

### Model selections (durable)

Location: `data/jarvis_settings.json`

- Persisted model-registry overrides for the `chat`, `tts`, `vision`, `browser_tool` roles plus any user-added custom providers.
- Read per message/per call â€” changes take effect immediately, no restart.

### Voice UI mirror state

Location: `backend/api/routes.py`

- `last_voice_message`, `last_voice_response`, `last_voice_log_id` â€” exists only so the frontend can poll and append the latest voice exchange into the visible chat log.

### Voice runtime state

Location: `backend/listener_state.py`

- speaking/thinking/user-speaking booleans, active recognizer threshold, timestamps for user speech events, any stored remaining speech text.

### Screen-control runtime state

Location: `backend/services/screen_state.py`

- whether screen controls are enabled; any pending plan waiting for confirmation.

### Pending task/confirmation gates

Location: `backend/core/brain.py` + `backend/services/task_agent/agent.py`

- `_pending_opencode_task` / `_pending_confirmation` / `_pending_browser_clarification` (45s expiry windows) â€” the arm-confirm-execute gates for task handoffs.

## Environment Variables and What They Actually Affect

Do not put real secret values into docs or prompts. The important thing is the variable names and their purpose.

### API keys (all optional individually â€” features degrade per provider)

- `GROQ_API_KEY` â€” Groq: intent-router fallback classifier, screen-Q&A last-resort vision, STT fallback.
- `GEMINI_API_KEY` â€” Gemini: chat env-default provider, intent primary classifier, screen Q&A vision, research per-site notes.
- `FIREWORKS_API_KEY` â€” Fireworks: current chat/vision override provider, task-agent planner, browser-agent default provider.
- `FISH_API_KEY` â€” Fish Audio TTS (the primary spoken-voice engine).
- `ELEVENLABS_API_KEY` â€” enables ElevenLabs TTS for short chunks.
- `OPENROUTER_API_KEY` â€” optional vision provider (free models) for screen grounding.
- `CLINE_API_KEY` â€” optional browser-agent provider.

### Model selection

- `GEMINI_BRAIN_MODEL` â€” Gemini chat/intent text model, default `gemini-3.5-flash-lite`.
- `GEMINI_VISION_MODEL` â€” Gemini vision model, default `gemini-3.5-flash-lite`.
- `GEMINI_INTENT_MODEL` â€” intent-router Gemini override.
- `GROQ_MODEL` â€” Groq text default (retired model; not used for chat anymore).
- `GROQ_VISION_MODEL` â€” Groq vision/Qwen default, `qwen/qwen3.6-27b`; also the intent fallback model source.
- `GROQ_INTENT_MODEL` â€” intent-router Groq override.
- `FIREWORKS_MODEL` â€” Fireworks default model, `accounts/fireworks/models/deepseek-v4-flash-0731` (used by the task-agent planner).
- `FIREWORKS_REASONING_EFFORT` â€” default `none`.
- `FISH_MODEL` â€” Fish TTS model, default `s2.1-pro-free`.
- `GROQ_STT_MODEL` / `GROQ_STT_URL` â€” Groq transcription model/endpoint.

### Task engine

- `JARVIS_TASK_ENGINE` â€” `browser_agent` (default) or `opencode`.
- `JARVIS_BROWSER_AGENT_PROVIDER` â€” default `fireworks`.
- `JARVIS_BROWSER_AGENT_MODEL` â€” default `accounts/fireworks/models/qwen3p7-plus`.
- `JARVIS_BROWSER_AGENT_MAX_STEPS` â€” safety backstop, default `50`.
- `JARVIS_BROWSER_AGENT_TIMEOUT` â€” task timeout seconds, default `480`.
- `JARVIS_BROWSER_AGENT_REASONING_EFFORT`, `JARVIS_BROWSER_AGENT_KEEP_LAST_IMAGES`, `JARVIS_BROWSER_AGENT_LOOK_WIDTH`, `JARVIS_BROWSER_AGENT_JPEG_QUALITY` â€” payload knobs.
- `BRAVE_MCP_PORT` / `BRAVE_MCP_SERVER_DIR` â€” brave-control MCP daemon coordination.
- `JARVIS_OPENCODE_PORT` / `JARVIS_OPENCODE_CMD` / `JARVIS_OPENCODE_TIMEOUT` â€” opencode engine (only when selected).

### Research browser

- `JARVIS_RESEARCH_CHANNEL` â€” browser channel for research (`chrome` default; `msedge` used on this machine).
- `JARVIS_RESEARCH_PROFILE` â€” headed browser profile dir, default `data/chrome_profile_jarvis`.
- `JARVIS_RESEARCH_REPORTS` â€” deep-research report output dir.

### Audio / microphone tuning

- `JARVIS_STT_LANGUAGES` â€” comma-separated speech-recognition languages, default `en-IN,hi-IN`.
- `JARVIS_IDLE_ENERGY_THRESHOLD` â€” main listener idle threshold.
- `JARVIS_WATCHER_ENERGY_THRESHOLD` â€” watcher-only threshold.
- `JARVIS_MIC_DEVICE_INDEX` / `JARVIS_MIC_NAME` â€” microphone selection.
- `JARVIS_SPEECH_START_VAD_RATIO` / `JARVIS_FINAL_SPEECH_VAD_RATIO` â€” VAD ratios.

### TTS tuning

- `JARVIS_PREFER_LOCAL_TTS` â€” legacy flag; Fish is tried first regardless, local SAPI5 is the fallback.
- `JARVIS_LOCAL_TTS_RATE` â€” `pyttsx3` speaking rate.
- `JARVIS_REMOTE_TTS_CHAR_LIMIT` â€” max text length eligible for ElevenLabs.
- `JARVIS_FISH_TTS_CHAR_LIMIT` â€” max chunk length for Fish.
- `JARVIS_FISH_TTS_VOLUME_BOOST_DB` â€” Fish playback gain.
- `JARVIS_TTS_OUTPUT_DEVICE` â€” Fish output device selection.

### Internal process coordination

- `JARVIS_EXTERNAL_RUNTIME` â€” set by the watcher before launching Electron; tells `main.js` not to spawn backend and voice mode again.
- `JARVIS_BACKEND_PORT` â€” backend port, default `9999`.
- `JARVIS_BACKEND_PID` / `JARVIS_ELECTRON_PID` â€” passed to the voice process for targeted shutdown.
- `JARVIS_WATCHER_CONTROL_PORT` â€” watcher HTTP control endpoint used by Electron stop.

### Running the test suite

The standard recipe (17 modules, run from the repo root with the backend venv):

```
& backend\venv\Scripts\python.exe -m unittest backend.tests.test_code_tools backend.tests.test_task_agent backend.tests.test_brain_gate backend.tests.test_voice_task_mute backend.tests.test_opencode_lifecycle backend.tests.test_browser_agent backend.tests.test_screen_control backend.tests.test_voice_mode_toggle backend.tests.test_chat_race backend.tests.test_voice_latency backend.tests.test_model_registry backend.tests.test_settings_routes backend.tests.test_fireworks_reasoning backend.tests.test_model_roles_wiring backend.tests.test_live_bugfixes backend.tests.test_websearch_interruption backend.tests.test_websearch_modes
```

512 tests currently pass. Wall time varies run-to-run (roughly 15s-300s) â€” the variance is known and comes from a few unmocked live network calls in `test_websearch_modes`, not from flaky assertions.

## Non-Obvious Behaviors Another Model Should Know

### 1. `backend/main.py` is the real backend entrypoint

`backend/app.py` exists, but the active launch commands use `backend.main:app`.

### 2. `grok_client.py` is named misleadingly â€” and its default model is dead

It talks to Groq, not xAI Grok. Its code default `llama-3.3-70b-versatile` is retired upstream (HTTP 404); every live caller passes an explicit model (e.g. `qwen/qwen3.6-27b`). Never route new chat traffic through its default.

### 3. The classifier can silently degrade â€” the deterministic nets exist because of it

`classify_intent` lands on a `chat` verdict whenever both cloud providers fail or throttle, and can genuinely misread screen questions as chat/research. The three deterministic nets in `process_message` (screen-question net, fresh-info auto-search, web-task routing) are regex-based backstops that rewrite or reroute such verdicts. When changing routing, check the nets, not just the classifier branch.

### 4. Screen control is not always active

The screen subsystem exists even when idle, but actual natural-language screen actions are rejected until the user turns screen controls on.

### 5. Voice replies are normally spoken for typed UI requests too

`POST /ask` triggers `voice.speak(...)` even for typed messages from the Electron UI.

### 6. Stop behavior is targeted, and the port-9999 kill is NOT at Electron startup

`stop-jarvis` kills only tracked backend/voice PIDs (`taskkill /PID <pid> /T /F`). External-runtime mode asks the watcher via HTTP `/stop` before falling back to `fallbackExternalCleanup()` (port-ownership kill of the backend + command-line kill of voice/whisper processes). The stale-backend kill at LAUNCH lives in the watcher, not in `main.js`. Deploying backend changes = restart the app (or let the watcher relaunch), since nothing hot-reloads.

### 7. The repo has hardcoded machine-specific paths

Examples include:

- repo base path
- desktop shortcut locations
- Brave executable location
- the expected virtualenv path
- the brave-control MCP server directory (`C:\Users\mayan\mcp-servers\brave-control`)

### 8. Dependency expectations are split and a little inconsistent

There are both `requirements.txt` and `backend/requirements.txt`, and neither file perfectly documents every runtime import used across the whole project. If something fails due to missing packages, inspect actual imports before trusting one requirements file.

### 9. The frontend is polling, not event-driven

Still no websocket â€” the renderer polls one adaptive `/ui-state` endpoint (~250ms active, 600ms idle), plus separate polls for `/screen-answer` and `/research-result` overlays.

### 10. Model selection is live and persisted

`data/jarvis_settings.json` overrides env defaults for chat/tts/vision/browser_tool per message â€” reading only `.env`/`config.py` will give you the wrong picture of which model actually answers. The registry masks API keys; never log or echo them.

## Where To Change Things

If another model is asked to make a specific kind of change, this is the shortest path to the right files:

### Change chat behavior or LLM prompting

Start in:

- `backend/core/brain.py` (`_build_chat_messages`, `_stream_chat_deltas`, `_resolve_chat_model`)
- `backend/services/gemini_client.py`, `backend/services/fireworks_client.py`
- `backend/services/model_registry.py` (provider/model selection)

### Change intent routing or the safety nets

Start in:

- `backend/services/intent.py` (classifier prompt + chain)
- `backend/core/brain.py` (nets in `process_message`, `_ChatRacer`)
- `backend/services/web_task_routing.py`, `backend/services/screen_analyzer.py` (deterministic predicates)

### Change websearch / research behavior

Start in:

- `backend/services/quick_search.py` (default AI-Overview tier)
- `backend/services/research_service.py` (deep multi-site tier)
- `backend/core/brain.py` (`handle_research_intent`, `should_search`)

### Change browser automation commands

Start in:

- `backend/core/brain.py`
- `backend/core/executor.py`

### Change task handoffs (browser agent / opencode)

Start in:

- `backend/services/browser_agent.py`, `backend/services/brave_mcp_client.py`
- `backend/services/task_agent/agent.py` (planner, confirmation gates)
- `backend/services/opencode_client.py`
- `backend/core/brain.py` (`handle_opencode_task`, confirmation consumption)

### Change native code-agent tools (file read/write, shell/script execution)

Start in:

- `backend/services/code_tools.py` (the tool implementations + registry)
- `backend/services/task_agent/agent.py` (SAFE_TOOLS, planner prompt, heuristics, exec)
- `backend/core/brain.py` (`is_code_tool_request` routing)

### Change wake-word behavior

Start in:

- `backend/watcher.py`
- `backend/services/transcription.py`
- `backend/services/audio_input.py`

### Change active voice listening or interruption behavior

Start in:

- `backend/services/listener.py`
- `backend/listener_state.py`
- `backend/voice_mode.py`

### Change TTS or spoken reply behavior

Start in:

- `backend/services/voice.py` (the Fish -> SAPI5 -> ElevenLabs ladder)
- `backend/services/fish_voice.py`
- `backend/services/elevenlabs_voice.py`
- `backend/services/earcons.py`

### Change screen controls

Start in:

- `backend/services/screen_control.py` (UIA-first fast path F29, planning, vision cascade)
- `backend/services/screen_capture.py` (F43 coordinate contract, monitor/DPI bookkeeping, last non-Jarvis target)
- `backend/services/screen_executor.py` (F42 revalidate-before-act execution)
- `backend/services/screen_geometry.py` (coordinate-space helpers, monitor bookkeeping, F44 rect comparison)
- `backend/services/screen_ui_elements.py` (UIA cache with runtime-id re-resolution, F44 preserved hierarchy)
- `backend/services/screen_ocr.py` (Tesseract word boxes with block/par/line identity, F44 merging)
- `backend/services/screen_analyzer.py` (screen Q&A prompts)
- `backend/services/screen_state.py` (enabled/pending-approval state)

### Change UI behavior

Start in:

- `frontend/renderer.js`
- `frontend/index.html`
- `frontend/style.css`

### Change HTTP API behavior

Start in:

- `backend/api/routes.py`
- `backend/main.py`

## Minimal File Map

Top-level directories and files that matter:

- `frontend/`
  - Electron renderer UI
- `backend/`
  - Python backend, voice mode, watcher, services
- `main.js`
  - Electron main process
- `run_jarvis.bat`
  - manual full-stack launcher
- `run_watcher.bat`
  - wake-word launcher
- `PROJECT_MAP.md`
  - this map
- `screen_commands.log`
  - persistent audit log of all screen control actions and execution steps
- `data/`
  - `jarvis_settings.json` (model selections), `conversation_history.json`, `chrome_profile_jarvis/` (research browser profile), `research_reports/`

Mostly non-runtime or secondary:

- `graphify-out/`
  - generated architecture artifacts
- `integrations/jarvis-editor-bridge/`
  - editor bridge extension
- `backend/tests/`
  - the unittest suite (screen control incl. G7, task agent incl. G1/G4, G3 request streaming, G5 research pipeline, G6 browser grounding, brain gates, model registry, voice latency, …)

## Short Architecture Summary

If you need the shortest accurate summary possible, use this:

- Electron provides the desktop shell and launches the backend/voice processes unless the watcher already did.
- FastAPI exposes `/ask`, `/voice-log`, `/voice-state`, `/ui-state`, `/screen-answer`, `/research-result`, `/settings`, and `/task/stop`.
- `backend/core/brain.py` is the central intent router: a speculative chat stream races the cloud classifier (Gemini Flash Lite -> Groq Qwen -> chat verdict), and three deterministic nets (screen-question, fresh-info auto-search, web-task routing) backstop classifier misfires.
- Chat is served by the model-registry-selected provider (currently Fireworks `qwen3p7-plus`) with a Gemini -> Fireworks fallback chain â€” NOT Groq, whose old default model is retired.
- Web lookups default to the Brave AI-Overview quick-search tier; "deepsearch" adds the multi-site research service with a glass-overlay report.
- Screen Q&A ("What's on my screen?") runs a vision cascade (registry provider first, then Gemini, then Groq Qwen) and shows floating desktop overlays.
- `command ...` messages trigger browser actions or launch local Windows applications (via `launch_app` resolver in executor) with web URL fallback.
- Multi-step web tasks hand off to the confirmation-gated browser agent (brave-control MCP daemon, Fireworks model, 50-step/480s backstops); the opencode CLI remains an opt-in engine.
- Voice mode continuously listens, routes recognized text through the same brain, and speaks the reply through the Fish Audio -> SAPI5 -> ElevenLabs TTS ladder.
- The watcher is a separate passive wake-word launcher running local GPU-accelerated Whisper (medium) with zero-temperature VAD filtering.
- Screen control is a distinct subsystem that combines direct command parsing, native UI Automation tree extraction, visual OCR text extraction, vision-model planning, and Windows input automation.
- All screen control actions are logged with full details in `screen_commands.log`.
- State is mostly in-memory module state; the durable artifacts are `data/jarvis_settings.json`, `data/conversation_history.json`, and the research reports.

## Recent Improvements (2026-08)

Chronological work on `master`, newest last. Test counts in each entry are for the full suite at that commit.

### Confirmation gate for opencode handoffs + folder/multi-file code tools (ae624eb)

Introduced an arm-confirm-execute gate before every opencode `run --auto` handoff: complex commands arm a spoken confirmation ("Sir, this is what I understood â€” ...") that expires after 45 seconds or can be declined, and the gate is disarmed cross-engine with the research and task gates. `is_code_tool_request` now routes folder/directory and multi-file phrasings into the task-agent path, and a new `code.create_folder` tool is gated via `CONFIRM_TOOLS`, with spoken counts honored in multi-file plans. Key files: `backend/core/brain.py`, `backend/services/code_tools.py`, `backend/services/task_agent/agent.py`, plus new `backend/tests/test_brain_gate.py` and `backend/tests/test_code_tools.py`.

### Single-voice rule during opencode tasks (9b92bcc)

While an opencode task runs, an `opencode_task_in_progress` flag in `brain.py` flips true before the worker thread spawns and resets on every exit path. Jarvis speaks exactly two phrases â€” "Handing the task to opencode, sir." and "Sir, the task has been completed." â€” with nothing in between, while chat text still publishes to the UI. Mute choke points cover the voice-mode listener/queue/brain-thread guards, route speech guards, and async-reply callbacks. Key files: `backend/core/brain.py`, `backend/voice_mode.py`, `backend/api/routes.py`, new `backend/tests/test_voice_task_mute.py`.

### Warm opencode session in a visible console, torn down with jarvis (0a9b0d3)

`ensure_opencode_server` now spawns `opencode serve` in its own visible console via `CREATE_NEW_CONSOLE`, which doubles as the live log; the spawned pid is tracked and `shutdown_opencode_server` kills by pid with a port-based taskkill fallback. Teardown is wired into the watcher stop path and the Electron stop-jarvis port kill, while a warm restart intentionally keeps the daemon alive for instant wake. Key files: `backend/services/opencode_client.py`, `backend/watcher.py`, `main.js`, new `backend/tests/test_opencode_lifecycle.py`.

### Live-stream opencode task output into a visible activity console (28f91af)

`run_opencode_task` switches to Popen streaming: every output line is appended to `backend/data/opencode_activity.log` with timestamped task headers and timeout markers, while the transcript still feeds the completion summary and `proc.wait()` runs before the exit-code check. The serve daemon is hidden again and a single visible console live-tails the activity log. Also fixes a race in `test_voice_task_mute` where worker threads could escape their patch windows. Key files: `backend/services/opencode_client.py`, `backend/tests/test_opencode_lifecycle.py`, `backend/tests/test_voice_task_mute.py`.

### Resilient opencode activity console + browser-task narrator (0bbbdbc)

Adds per-task activity-log truncation with public truncate/append/narrate wrappers, a shrink-aware `tail_activity.ps1` that starts at EOF and clears the host on shrink, `close_activity_tail` on warm stop, and narration phrases for fast browser tasks plus the BRAVE_MCP daemon constants. Key files: `backend/scripts/tail_activity.ps1`, `backend/services/opencode_client.py`.

### Native browser-automation agent on the brave MCP daemon (954fb37)

Adds a streamable-HTTP MCP client (`backend/services/brave_mcp_client.py`) and a tool-calling browser-agent loop (`backend/services/browser_agent.py`) with a 2-retry rule and `MAX_STEPS`/`TIMEOUT` backstops; `TASK_ENGINE` defaults to `browser_agent`, with opencode still selectable via `JARVIS_TASK_ENGINE=opencode`. Providers cover fireworks/groq/openrouter/gemini plus cline data-wrapper unwrap, with MiniMax M3 as the default model and default reasoning. The brain's engine branch sits behind the same confirmation gate, and the watcher boots the activity tail and brave daemon only in browser_agent mode. Key files: `backend/services/browser_agent.py`, `backend/services/brave_mcp_client.py`, `backend/config.py`, `backend/watcher.py`, `backend/tests/test_browser_agent.py`.

### Fully detach the opencode engine while browser_agent is default (1e9d48f)

The activity-tail console is retitled "jarvis - task activity" (was "opencode"), and hard guards make `run_opencode_task` and `ensure_opencode_server` refuse to spawn anything unless `TASK_ENGINE=opencode`. The watcher kills stale serve daemons on port 9560 at boot in browser_agent mode, and MiniMax models never receive a `reasoning_effort` argument (default reasoning). Key files: `backend/services/opencode_client.py`, `backend/scripts/tail_activity.ps1`, `backend/watcher.py`, `backend/services/browser_agent.py`.

### Switch browser agent to Fireworks MiniMax M3 (c0c9ce0)

`load_dotenv(override=True)` makes `.env` beat stale inherited process environment variables â€” fixing a stale `FIREWORKS_API_KEY` that was shadowing the updated key â€” and the provider default moves `cline` to `fireworks` with the model default `accounts/fireworks/models/minimax-m3`, live-verified for tool calls. Regression tests cover the dotenv-override-beats-stale-env case and the new defaults. Key files: `backend/config.py`, `backend/tests/test_browser_agent.py`. (Note: the browser-agent default model has since moved to `qwen3p7-plus` â€” see config.py.)

### Raise browser-agent limits for longer tasks (606e06f)

`BROWSER_AGENT_MAX_STEPS` goes 12 -> 35 and is now env-overridable via `JARVIS_BROWSER_AGENT_MAX_STEPS`; the task timeout goes 180s -> 480s via `JARVIS_BROWSER_AGENT_TIMEOUT`. Both defaults are covered by `ConfigDefaultsTests` reload assertions. Key files: `backend/config.py`, `backend/tests/test_browser_agent.py`. (Note: the steps cap has since been raised to 50.)

### Never truncate the voice confirmation question mid-sentence (472c09d)

`handle_opencode_task` now budgets only the task text when `voice_compact`: the closing question "Do you want me to go ahead and execute it?" always survives intact and total length stays at or below 180 characters, using a word-boundary cut via the `_truncate_at_word` helper. `handle_task_message` cuts at the last sentence terminator when it is at least 100 characters in, else at a word boundary with an ellipsis, so replies never end mid-word. Six new regression tests bring the suite to 133/133. Key files: `backend/core/brain.py`, `backend/services/task_agent/agent.py`, `backend/tests/test_brain_gate.py`, `backend/tests/test_task_agent.py`.

### Note: event-driven page-settle latency overhaul (brave-control repo, not versioned here)

This change lives in the separate brave-control MCP server repo at `C:\Users\mayan\mcp-servers\brave-control` (file `server.mjs`), which is not a git repo and is not versioned in this project. A `settlePage(page, opts)` helper replaced the slow `networkidle` waits in `navigate`/`new_tab` (now `waitForEvent('load', 3s)` with a catch, then settle 300ms) and added a 200ms settle after `click_element`. It resolves on a main-frame `framenavigated` event or DOM-mutation quiescence (200ms debounce, 2500ms hard cap), disconnects its MutationObserver cleanly, and is wrapped in try/catch so settling can never fail a tool call. `ask_chat` polls every 300ms instead of 1500ms, and `copy_code_block` switched to a 100ms clipboard poll capped at 2000ms. Measured navigate ~390-406ms (was 2-4s+) and click ~16ms; that repo's node tests pass 18/18.

## Recent Improvements (2026-09)

Chronological work on `master`, newest last. Test counts are the number of test methods across the current 17 suite modules at that commit.

### Short browser task summaries and instant TTS barge-in stop (901fd6e)

Browser-agent task completions now produce short spoken summaries instead of raw transcripts, and TTS barge-in stops instantly: the listener's stop path bumps the speech generation and flushes Fish PCM playback mid-stream so a user interruption cuts the voice without lag. Key files: `backend/core/brain.py`, `backend/api/routes.py`, `backend/services/listener.py`, `backend/services/fish_voice.py`, `frontend/renderer.js`, `frontend/capsule_renderer.js`, new `backend/tests/test_live_bugfixes.py` coverage. Suite: 440.

### Tiered web search with Brave AI Overview and interruptible research (69f9b07)

Introduces the two-tier lookup contract: default mode reads the Brave Search AI Overview answer box via headed Chrome on the shared research profile (no site scraping, short spoken+text summary, snippet fallback), while explicit "deepsearch" pins the overview into the existing multi-site `run_research` flow. Research runs in a background thread with an immediate ack, is stoppable mid-run ("stop the research"), and reports push to the UI. Key files: new `backend/services/quick_search.py`, `backend/core/brain.py` (`handle_research_intent`), `backend/services/research_service.py`, `backend/api/routes.py`, `backend/services/listener.py`, `backend/voice_mode.py`, new `backend/tests/test_websearch_interruption.py` and `backend/tests/test_websearch_modes.py`. Suite: 485.

### Plain-spoken AI overview answer, print to chat UI, TTS markdown strip and fallback cleanup (4608a94)

The quick-search answer is rewritten into a plain-spoken spoken summary, the lookup result prints into the chat UI, TTS strips markdown before speaking, and dead fallback branches are cleaned up. Key files: `backend/services/quick_search.py`, `backend/core/brain.py`, `backend/tests/test_websearch_modes.py`, `backend/tests/test_browser_agent.py`. Suite: 485.

### Websearch permission removal, ask tab answer fallback, fast-fail classify and key redaction (37ae46b)

Removes the browser permission prompt from the websearch flow, adds the Ask-tab answer fallback with container-first extraction (`div.message.assistant.llm-output`) plus a guarded `rfind(query)` fallback for when Brave's AI answer renders in the Ask view, makes `classify_intent` fast-fail (3s timeout, `no_retry`, no urllib3 retry multiplication) with a Groq Qwen fallback, and redacts API keys (`key=<redacted>`) from all logged URLs/headers in the Gemini client. Key files: `backend/services/quick_search.py`, `backend/services/intent.py`, `backend/services/gemini_client.py`, `backend/services/grok_client.py`, `backend/core/brain.py`, `backend/tests/test_websearch_modes.py`. Suite: 496.

### Fireworks planner swap, confirmation-gated fallbacks, web-task routing, fresh-info auto-search (2c55775)

The task-agent planner moves from retired Groq `llama-3.3-70b-versatile` (404) to `ask_fireworks` (`deepseek-v4-flash-0731`, temperature 0.1), and BOTH no-plan fallbacks become confirmation-gated (`requires_confirmation=True`). New deterministic nets in `process_message`: web-shaped task requests (`backend/services/web_task_routing.py`) route to the confirmation-gated browser-agent handoff instead of the raw task path, and chat verdicts with fresh-info keywords (pricing/cost/latestâ€¦) that are question-shaped but not greetings auto-route to the quick-search tier. Intent prompt gains pricing/research and multi-step-web-task examples. Key files: `backend/services/task_agent/agent.py`, `backend/core/brain.py`, new `backend/services/web_task_routing.py`, `backend/services/intent.py`, `backend/tests/test_brain_gate.py`, `backend/tests/test_task_agent.py`, `backend/tests/test_chat_race.py`. Suite: 504.

### Deterministic screen-question net over classifier chat/research misreads (1fdff1c)

"what's on my screen jarvis" was answered "I cannot see your screen" because the cloud classifier genuinely misclassifies screen questions as chat (and sometimes research), not only on provider outages. A deterministic net now fires right after `classify_intent`: on a `chat` or `research` verdict, `is_screen_question` rewrites the intent to `screen`/`region` before the racer holdback and fresh-info nets, so the existing screen branch handles analysis. tool/task verdicts are exempt (their structured steps would be discarded). Intent prompt gains the apostrophe+wake-word example. Key files: `backend/core/brain.py`, `backend/services/intent.py`, `backend/tests/test_brain_gate.py` (8 ScreenQuestionNetTests). Suite: 512.
