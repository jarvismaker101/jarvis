# Jarvis Assistant Project Map

This document gives another model enough context to work on the repo without re-discovering the architecture from scratch. **Refreshed 2026-09-30.**

State as of this refresh:

- The **Fable-5 audit remediation** (G0–G11, F01–F55) is landed except the **G8 classifier-retirement step**, which remains open.
- The **CODE_REVIEW_REPORT hardening wave** is landed for **C1–C3** and **H1–H9**, plus the **STT-hallucination gate**. The review's M/L/structural items are still open.
- A **responsiveness audit** (43 findings, P0-01…P1-19) was completed 2026-09-29 and triaged by the owner. **13 of its items are now implemented** — the 2026-09-30 wave, commits `f37a519`…`6c89095`. Per-item detail is in "Responsiveness audit implementation state" below; the remaining items are still open, and `AUDIT_IMPLEMENTATION_PROMPTS.html` holds the per-item implementation prompts.

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

Typed UI requests and spoken requests both end up in `backend.core.brain.process_message()` (defined at `brain.py:3266`, delegating to `_process_message_inner` at `brain.py:3308`). Deterministic phrase routes (memory, explicit stops, pending confirmations) run first; then explicit task/code-tool, screen-control and research routes, plus the optional G8 orchestrator. Otherwise that function races a speculative chat stream (`_ChatRacer`, `brain.py:1568`, started at `brain.py:3482`) against a cloud intent classifier, applies deterministic safety nets (screen questions, fresh-info search, all-search-steps reroute, web-shaped tasks) over the classifier verdict, then routes to chat, tool actions, screen Q&A, research, browser-agent tasks, or memory commands, and returns a short English reply that may also be spoken aloud.


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
  * **F51 sandboxed renderers + authenticated localhost** (`frontend/preload.js`, `main.js`, `frontend/capsule_main.js`, and all 5 renderer JS files): every window is now `contextIsolation:true, nodeIntegration:false, sandbox:true` with the minimal `contextBridge` preload bridge — validated IPC channels only, http/https-validated `openExternal`, and a token-attaching `jarvisAPI.backend()` proxy so the backend token rides on every call (overlays, which render web-derived content, can **never** read the token). Renderer-initiated navigation is denied and `window.open` is denied. `backend/services/local_auth.py` (`X-Jarvis-Token`, per-launch secret, constant-time compare) enforces the token on every endpoint EXCEPT the public liveness surface (`GET /health`) — that includes private READS such as `/ui-state`, `/voice-state`, `/screen-answer`, `/research-*` and `/ask/status/*`, which is why the renderer and the voice worker always attach the header. `backend/services/runtime_identity.py` writes an atomic `data/runtime/backend-instance.json` stamp (pid + instance_id + protocol + build) at startup, and `/health` exposes it. A missing token no longer disables enforcement: `local_auth` FAILS CLOSED outside explicit development mode, and only `JARVIS_DEV_MODE=1` reopens the surface. The old `allow_origins=["*"]` CORS is gone too — `backend/main.py` now allows the renderer origins plus loopback only, with credentials off. `frontend/*.html` carry a `Content-Security-Policy` meta tag. `BRAVE_MCP_TOKEN` is already env-based in source (rotation is a user step on `.env`/`~/.config/opencode/opencode.jsonc`).
  * **F52 supervised, versioned, observable runtime** (`main.js`, `watcher.py`): child processes are spawned with stdio piped and drained into bounded `data/logs/<label>.log` (cap + half-rotate) — never a blocking full pipe. `registerManagedProcess` retains the handle and restarts required workers within a restart **budget** (2 backend, 1 voice) on unexpected exit; deliberate stops never count. A backend replacement best-effort POSTs `/approvals/reset` first (invalidate stale approvals before recovery). `watcher.py` mints the per-launch token, injects it into every child (backend/voice/electron), and verifies the backend via **identity** (instance stamp vs. `/health`) instead of the mere existence of `/research-result`; it invalidates the live backend's approvals before replacing a stale one and exposes distinct warm-sleep (`/stop`, authed) and full-shutdown (`/shutdown`, authed) control semantics.

## Code-Review Remediation State (2026-09)

`CODE_REVIEW_REPORT.txt` / `.pdf` (generated 2026-09-23; also summarised in `codex.md`) is a second, independent audit of the whole repo: 55 numbered items (3 critical, 9 high, 14 medium, 17 low, 12 structural). State as of this refresh:

- **C1 (landed)** — screen-planner prompt injection: all screen-derived text (OCR, UIA names, window title, interaction history) is untrusted data wrapped in spoof-proof `<<<SCREEN_TEXT_UNTRUSTED>>>` delimiters, with role-prefix and control-phrase stripping behind a single choke point (`_build_tree_prompt`). Tests: `backend/tests/test_c1_prompt_injection.py`.
- **C2 (landed)** — one declared coordinate frame: prompt geometry (UI tree + OCR nodes) is serialised in NORMALIZED 0..1000 at serialisation time, so an echoed pixel coordinate can no longer be reinterpreted as a percentage; a step whose cited element and its coordinates disagree is rejected instead of silently preferring one half. Tests: `test_c2_coordinate_frames.py`.
- **C3 (landed)** — model output can no longer become raw OS input: key/hotkey tokens are whitelisted against the key vocabulary (a combined string like `alt+f4` as ONE token is rejected), click buttons are whitelisted `{left,right}`, click counts are clamped 1..10, and `screen_executor` rejects any token outside its safe grammar. Tests: `test_c3_input_synthesis.py`.
- **H1 (landed)** — `requirements.txt` regenerated from the working venv via pip freeze (numpy 1.26.4 ABI pin, pywinauto 0.6.9, and the dozen-plus runtime packages the old file omitted — the review counted ~12, the regenerated file added 14 pins). Pin guards in `test_h1_requirements.py`.
- **H2 (landed)** — `executor.launch_app` deleted its `shell=True` fallback: Edge launches by protocol, VS Code by argv list, against a fixed system-app allowlist. Tests: `test_h2_no_shell_launch.py`.
- **H3/H7 (landed)** — `screen_ui_elements`: a `set_focus` failure now ABORTS a type action instead of typing into whatever holds focus; legacy bare-wrapper cache entries are treated as stale; cache TTL 8s -> 3s; per-thread COM apartment initialisation (`comtypes.CoInitialize`) at every UIA entry fixes the silent `RPC_E_CHANGED_MODE` degradation to blind coordinate clicks. Tests: `test_h3_h7_ui_safety.py`.
- **H4/H5/H6 (landed)** — execution staleness and geometry: plans carry a `capture_epoch` probed before EVERY effect that refuses on mismatch; `_verify_window_identity` (IsWindow + process id) runs before every effect and at focus acquisition, and a hit-test with no window under the point now REFUSES; out-of-frame normalised points are rejected rather than clamped into edge clicks. Tests: `test_h4_h5_h6_staleness.py`. (The report's DPI/monitor double-scaling claim did not survive verification: captured pixels are physical end-to-end.)
- **H8 (landed)** — the retired Groq model `llama-3.3-70b-versatile` is gone from `grok_client`'s default and from the offered catalog.
- **H9 (landed)** — the research overlay accepts only `http(s)`/`mailto` markdown hrefs; `javascript:`/`data:`/`vbscript:` render as plain label text. Tests: `tests/research-overlay-href-scheme.test.js` (node).
- **STT hallucination gate (landed)** — `backend/services/transcription.py::is_hallucinated_transcript()` rejects prompt-vocabulary echoes (5+ tokens drawn only from the wake-bias vocabulary), looped tokens/phrases, memorised silence phrases ("a ver si te acuerdas de esto", "thanks for watching", ...) and filler-only utterances, while never rejecting real commands, short wake phrases, literal payloads or repeated safety words. It is wired into the listener's partial windows, each STT engine's accept, the final `listen()` commit belt, the watcher's `is_wake_word`, and `_add_candidates`. `whisper_daemon`'s wake-bias `INITIAL_PROMPT` is now opt-in via `X-Jarvis-Purpose: wake` (sent only by the watcher), so conversation transcription runs unbiased. Tests: `test_stt_hallucination_gate.py`.
- **Chat-outage root cause (2026-09-23, fixed)** — direct `generativelanguage.googleapis.com` degraded to 7-45s+ through the user's Proton VPN tunnel while the Fireworks account was suspended, so every query fell through to the generic failure message. Fixes: the `chat` role allowlist gained `openrouter`, `get_provider_credentials` returns canonical OpenAI-compatible base URLs for openrouter/groq, and `intent.py` now classifies OpenRouter -> Gemini -> Groq. The current `data/jarvis_settings.json` has vision and browser_tool on `openrouter/google/gemini-2.5-flash-lite` (chat is on `gemini/gemini-3.5-flash-lite`).
- **Still open** — the report's M1 (approvals verdict grammar: affirmation words anywhere count as consent) and the remaining M/L/structural items, plus the G8 classifier-retirement step.

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

### 2. Watcher-launched full stack (`run_jarvis.bat`)

Main files: `run_jarvis.bat`, `backend/watcher.py`

Flow:

- `run_jarvis.bat` no longer starts the three processes itself. It launches `python -m backend.watcher --launch` in a MINIMISED console and appends that console's output to `data\logs\watcher.log` (the file to read first when a launch does not come up).
- `--launch` makes the watcher bring the stack up immediately instead of waiting for a wake phrase; the watcher is therefore the launch supervisor in this path too — it mints the per-launch `JARVIS_LOCAL_TOKEN`, injects it into every child, and identifies the backend through `/health` before adopting it.
- `run_jarvis_noisy.bat` is the same launch with loud-room microphone thresholds preset (`JARVIS_IDLE_ENERGY_THRESHOLD=2500`, `JARVIS_WATCHER_ENERGY_THRESHOLD=2000`, `JARVIS_MAX_ENERGY_THRESHOLD=5000`, VAD ratios 0.5/0.35). `run_watcher_noisy.bat` is its listen-only twin.

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
- It exposes an authed HTTP control plane on `JARVIS_WATCHER_CONTROL_PORT` with two DIFFERENT semantics: a warm sleep (`/stop`) tears down backend/voice/Electron but deliberately keeps the resident `whisper_daemon` (and the opencode/brave daemons) alive for an instant next wake, while a full shutdown (`/shutdown`) leaves nothing behind. Electron's stop button uses `/stop` and only falls back to killing port owners if the watcher does not answer.
- When Electron exits and the backend is no longer on the configured port (`JARVIS_BACKEND_PORT`, default `9999`), the watcher resumes listening.

## End-to-End Request Flows

### Typed UI flow

Path:

`frontend/renderer.js`
-> `POST /ask/stream` (SSE; F23 request id + reconnect; plain `POST /ask` is still the non-streaming form)
-> `backend/api/routes.py` (admission -> one job -> `_run_request_worker` -> `process_message`)
-> `backend/core/brain.py` (`process_message` -> `_process_message_inner`)
-> deterministic phrase routes first (memory / commitments / skills, explicit stops, pending confirmations and clarifications)
-> then, in order: explicit task and code-tool handoff, screen control, explicit research, and — in `orchestrator` mode only — the G8 native tool-use loop
-> otherwise the legacy path: `classify_intent` raced against the speculative `_ChatRacer`, with the deterministic nets rewriting the verdict (see brain.py section)
-> one of:
- chat via the model-registry-selected provider chain (`gemini_client` / `fireworks_client` / OpenAI-compatible for every other provider)
- tool actions via `backend/core/executor.py`
- screen Q&A via `backend/services/screen_analyzer.py`
- research via `backend/services/quick_search.py` / `research_service.py`
- browser-agent task via `backend/services/browser_agent.py` (confirmation-gated)
- screen control via `backend/services/screen_control.py`
- memory ops via `backend/core/memory_store.py`; memory reset via `backend/core/memory.py`
- native code tools via `backend/services/code_tools.py`
-> response returned to the UI as numbered SSE events
-> the same response is also spoken through `backend/services/voice.py` — unless the caller sent `speak=False`, which the voice worker always does

Notes:

- The UI has `chat` mode and `command` mode.
- In command mode, the renderer prefixes the request with `command ` before sending it to `/ask/stream`.
- Request admission lives in `request_registry`: a retried request id reuses its result instead of re-executing, a reused id carrying a different message is rejected, and identical messages inside 1 second are deduplicated.

### Route selection and the chat race (inside process_message)

`backend/core/brain.py` is now 3827 lines. `process_message` (line 3266) binds the F20 turn job and delegates to `_process_message_inner` (line 3308). The order today:

1. The `clear memory` phrase list.
2. G9 memory / commitment / skill phrase ops (`memory_store.handle_memory_phrase`) — `remember that…`, `forget…`, `what do you remember about…`, `remind me to …`, `cancel the reminder`, `approve the … skill`. These run before every other route.
3. Explicit research stop (`is_stop_research`), then the pending research / task-action / opencode-handoff confirmations and the browser-clarification follow-up.
4. `is_explicit_task_request` and `is_code_tool_request` hand straight to `handle_task_message`.
5. `maybe_handle_screen_control_message` (skipped for `command …`).
6. `force_research` explicit research phrasings (including the `deepsearch` keyword).
7. **G8 orchestrator route selection** (`orchestrator_select_route`, line 3451): with `JARVIS_ORCHESTRATOR_MODE=orchestrator` the native tool-use loop is attempted first and only a decline falls through to legacy routing; in the default `legacy` mode (the shipped default, `config.py:69`) the orchestrator is not selected at all.
8. Legacy routing: `_ChatRacer` speculative stream (started line 3482) + `classify_intent` (line 3514, budgeted by `INTENT_BUDGET_VOICE_MS`) -> screen-question net -> racer holdback on any non-chat verdict -> fresh-info auto-search net -> tool / research / screen / region / task branches -> implicit `is_task_request` web-shaped handoff -> chat -> `command …` parsing.

`classify_intent` (`backend/services/intent.py:228`) asks **OpenRouter first** (Gemini 2.5 Flash Lite, `google/gemini-2.5-flash-lite`), then Gemini 3.5 Flash Lite direct, then Qwen 3.6 27B on Groq, and lands on a `chat` verdict if all three fail — which is exactly why the deterministic nets below exist. All three hops share ONE monotonic deadline (`_budget_timeout`), so the advertised `timeout_ms` is real rather than per-hop. The order was flipped to OpenRouter-first on 2026-09-23 because the Cloudflare-fronted OpenRouter endpoint stayed ~1.4 s while direct Gemini degraded to 7–45 s behind the VPN. **The module docstring still describes the old Gemini-first order and is stale.**

### Voice conversation flow

Path:

`backend/voice_mode.py` (pure I/O worker since G11/F50 — it deliberately does NOT import `backend.core.brain`)
-> `backend/services/listener.py` (transcripts only ever committed after the F34 stabiliser and the STT hallucination gate)
-> `POST /ask/stream` on the one backend runtime, authenticated with the per-launch `X-Jarvis-Token`, sent with `speak=False` and `origin=voice`
-> `backend/core/brain.py` (`process_message`)
-> streamed deltas feed this process's `StreamSpeaker` (the backend stays silent for these requests)
-> `POST /voice-state/publish` publishes real listening state; task state is polled from `/ui-state`
-> `frontend/renderer.js` polls `/ui-state` and mirrors the exchange in the UI

Voice mode is split into two loops:

- `listener_thread()`
  - continuously captures speech with `listen()`
  - if Jarvis is currently speaking, only interruption commands are honored
  - otherwise recognized text is pushed into a queue
- `brain_thread()`
  - consumes queued text
  - handles shutdown, continue/resume, and "normal setup" shortcuts
  - otherwise submits the utterance to the backend over HTTP (see above) — never by calling `process_message` locally

Controls route through the authed HTTP control plane rather than module copies: `/task/stop`, `/speak/stop`, `/speak/pause`, `/speak/resume`, `/approvals/reset`, and `/voice-setup/launch` (the "normal setup" app launches are now a typed backend job).

### Wake-word flow

Path:

`backend/watcher.py`
-> microphone capture
-> `backend/services/wake_engine.py` (F36): `openwakeword` keyword spotting when installed (`JARVIS_WAKE_ENGINE`, default `auto`), with the fuzzy phrase path as the graceful fallback
-> local GPU-accelerated faster-whisper served by the persistent `backend/whisper_daemon.py` (with Google/Groq fallback), optionally preceded by the `JARVIS_WAKE_PRE_ROLL_SECONDS` pre-roll ring and followed by online Whisper verification (`JARVIS_WAKE_ONLINE_VERIFY`)
-> fuzzy/online wake-phrase matching, vetted by the STT hallucination gate (a prompt-vocabulary echo must never false-launch the stack)
-> the phrase TAIL after the wake window is extracted and forwarded to `/ask` ("wake up jarvis and search for cats" acts on both parts)
-> spawn backend + voice + Electron

`whisper_daemon.py` binds its port BEFORE loading the model (F55): `/health` answers in milliseconds and reports `ready`/`loading` honestly, the ~1.5 GB model load starts on demand (`POST /warm`) or on the first `/transcribe`, and the listener binds exclusively so a second daemon cannot silently steal connections. The watcher is separate from active voice mode: it is optimized for short wake phrases, not full conversation, running locally on the CUDA GPU.

### Screen Q&A flow

Path:

message
-> `backend/core/brain.py` (screen/region verdict from the classifier, or the deterministic screen-question net)
-> `backend/services/screen_analyzer.py`
-> `_ask_screen_vision_cascade` via `backend/services/vision_cascade.py` (F37): the registry-selected provider is dispatched first, then the eligible configured providers in a deterministic bounded order (`gemini`, `fireworks`, `groq`, `openrouter`, at most 3 dispatched attempts), each provider at most once, with empty/malformed output advancing instead of ending the call and no provider ever dispatched without a credential
-> fetches topic images via `backend/services/image_fetcher.py`
-> pushes structured JSON to `/screen-answer`
-> `main.js` polls and routes to `overlay_renderer.js` and `overlay_images_renderer.js`
-> displays floating glassmorphism overlays on desktop

### Research / quick-search flow

Path:

message (research verdict, or the fresh-info auto-search net)
-> `brain.handle_research_intent` â€” acks immediately, works in a background thread
-> default mode: `backend/services/quick_search.py` â€” headed Brave Search on the shared research Chrome profile, reads the AI Overview answer box (Ask-tab fallback, container-first extraction), returns a short spoken+text summary; no site scraping
-> deepsearch mode (explicit "deepsearch" keyword): AI Overview PLUS `backend/services/research_service.py` multi-site flow â€” Brave search, scrape top-N results, Gemini per-site notes, one deduped consolidated report pushed to the glass overlay via `/research-result` and saved under `data/research_reports/`
-> research is interruptible ("stop the research") via a stop event checked between sites
-> both tiers now share ONE long-lived browser worker (`backend/services/research_browser.py`, F27/F28): a single daemon thread owns one asyncio loop and one persistent Playwright context for the whole process; callers submit jobs (`submit` for sync bodies, `run` for coroutines that drive several pages concurrently); cancellation closes only that task's pages; bounded to 4 in-flight jobs and retired after an idle TTL
-> progress and incremental evidence are published out-of-band on `/research-progress`, because the originating SSE stream is already closed by the time a deep run produces them

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

Every plan step is gated at execution time (see the 2026-09 review wave): the planner prompt treats all screen-derived text as untrusted data with spoof-proof delimiters, prompt geometry is serialised in one declared NORMALIZED 0..1000 frame, key/click payloads are whitelisted and clamped, a plan carries the `capture_epoch` it was built from and every effect refuses on an epoch or window-identity mismatch, and out-of-frame points are rejected instead of clamped.

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
  - CORS is RESTRICTED (G11/F51): only the Electron renderer origins (`file://` reports `null`) and loopback dev servers, with credentials off. It used to be `allow_origins=["*"]`.
  - Installs `local_auth`, which FAILS CLOSED: with a supervisor-injected `JARVIS_LOCAL_TOKEN` every non-public endpoint requires it, and without one the surface is closed unless `JARVIS_DEV_MODE=1` explicitly declares development mode.
  - On startup: `validate_environment()`, writes the `data/runtime/backend-instance.json` identity stamp, and starts the Ollama and TTS warm-up threads.
  - Includes routes from `backend/api/routes.py`.

- `backend/app.py`
  - Secondary minimal FastAPI entrypoint.
  - Also includes the router, but current launch scripts use `backend.main:app`, not this file.

- `backend/api/routes.py` — the whole local control plane (~1180 lines). Every mutating endpoint AND every private read requires the per-launch `X-Jarvis-Token` unless `JARVIS_DEV_MODE=1`; only the public liveness surface stays open.
  - `POST /ask` — non-streaming request; F23 request-id admission (a retried id reuses its result instead of re-executing; the same id carrying a different message is a 409), a 1-second duplicate guard, and it speaks the reply unless the caller passed `speak=False`.
  - `POST /ask/stream` (SSE) — the primary UI and voice path. One frame shape `{type, seq, request_id, ...}` with `delta|replace|progress|completed|interrupted|error`; reconnecting with `last_event_id` resumes without re-executing; a voice submission (`speak=False`, `origin=voice`) runs the same runtime but stays silent.
  - `GET /ask/status/{request_id}` — status lookup for reconnect checks.
  - `POST /update-voice-log`, `GET /voice-log` — the latest single voice exchange for the UI mirror.
  - `GET /voice-mode`, `POST /voice-mode` — voice input on/off.
  - `POST /voice-setup/launch` — runs the "normal setup" applications as a typed backend job (F50).
  - `GET /voice-state`, `POST /voice-state/publish` — the voice I/O worker publishes its real listening state (owner incarnation + generation; a stale generation or an expired snapshot is rejected) and the backend serves it.
  - `GET /ui-state` — fused `{state, voice_log, voice_input_enabled, task_running}`; halves renderer polling.
  - `POST /speak/stop`, `POST /speak/pause`, `POST /speak/resume`, `GET /speak/remaining` — F35 barge-in / pause / continue controls (the pause remainder is published here because the voice process is a separate OS process).
  - `GET /aec/state`, `GET /aec/reference` — F33 echo-cancellation diagnostics and the cross-process rendered-PCM reference span.
  - `POST /task/stop` — F20: cancels ONE identified job (or the newest still-running one) at its next checkpoint; an idle stop cancels nothing.
  - `POST /approvals/reset` — invalidates a pending consent, clears the pending screen plan and interrupts live registered requests before a supervisor replacement.
  - `GET /health` — liveness plus identity (`instance_id`, `pid`, `protocol`, `build`, `auth` fingerprint).
  - `GET`/`POST /screen-answer` — F30 publish-or-patch with capture-generation and revision guards (a stale patch gets 409 instead of clobbering a newer answer).
  - `GET`/`POST /research-result` — deep-research report delivery for the research overlay.
  - `GET`/`POST /research-progress` — F28 out-of-band research progress and incremental evidence.
  - `GET /settings`, `POST /settings/model`, `POST /settings/chat-model`, `POST /settings/provider`, `GET /providers/{id}/models` — the model-registry surface for all six roles; listings are masked (`has_key` booleans only) and selections are capability-validated.

### Core decision layer

- `backend/core/brain.py`
  - This is the main router for user intent.
  - Most important file in the repo for behavior changes.

Routing order inside `process_message` is listed in "Route selection and the chat race" above (it changed and grew: memory phrases now run before everything, the G8 orchestrator route selection now precedes the legacy classifier, and there is a fourth branch — tool-search steps reroute to research). The deterministic nets are unchanged in intent:

1. Pending confirmation/follow-up gates (research confirm, task confirm, opencode confirm, browser clarification).
2. `maybe_handle_screen_control_message(...)` (bypassed if prefixed with `command`).
3. Legacy branch only: speculative `_ChatRacer` start, then `classify_intent(msg)`.
4. **Screen-question safety net**: if the verdict is `chat` or `research` and `is_screen_question(msg)` (deterministic regex in `screen_analyzer.py`), the verdict is rewritten to `region`/`screen`. tool/task verdicts are exempt because they carry structured steps the upgrade would discard.
5. Racer holdback: any non-chat verdict cancels the speculative stream.
6. **Fresh-info auto-search net**: a `chat` verdict that hits `should_search()` (pricing/cost/latest/news keywords) and is question-shaped but not greeting-like reroutes to `handle_research_intent` (quick-search tier), so stale chat answers are never served for current-world facts.
7. Tool steps branch (an all-`search` step list reroutes to research), research branch, screen/region branch (screen Q&A), task branch (opencode/browser-agent handoff, confirmation-gated).
8. **Web-task routing net**: the broad `is_task_request` heuristic, when `TASK_ENGINE=browser_agent`, checks `is_web_shaped_task` (`backend/services/web_task_routing.py`: web hint + interaction verb, no local hint) and hands the task to the confirmation-gated browser-agent path instead of the raw task-message path.
9. Memory reset phrases, command-mode parsing, legacy chat path.

Important chat behavior:

- The chat model is resolved per message by the model registry â€” a UI model switch takes effect on the next reply, no restart.
- Provider chain (`_stream_chat_deltas` / `_ask_chat_nonstream`): the registry-selected provider goes first. `gemini` and `fireworks` use their dedicated clients; any other provider (openrouter, groq, a user-added custom provider) is carried by the OpenAI-compatible path via `get_provider_credentials`. An empty stream retries the SAME model non-stream before any provider fallback, and a terminal failure (auth/permission/validation) REFUSES instead of silently answering with a different model (F24/F49). The tail fallback chain is still Gemini -> Fireworks. Fallbacks are recorded for a UI warning (`_record_chat_fallback`, keys scrubbed).
- Current selection per `data/jarvis_settings.json` (verified 2026-09-30, `revision` 16): chat = `gemini`/`gemini-3.5-flash-lite`, vision = `openrouter`/`google/gemini-2.5-flash-lite`, browser_tool = `openrouter`/`google/gemini-2.5-flash-lite`, tts = `fish`/`s2.1-pro-free`, listening = `whisper`/`whisper-local`. Reading only `.env` or `config.py` will tell you the wrong story — this file is the live truth.
- Groq is NOT the chat provider: `grok_client`'s retired `llama-3.3-70b-versatile` default is gone (H8); callers pass explicit models.
- Detects Hindi/Hinglish vs English and still forces English replies.
- Uses short speech-oriented prompts when `voice_compact=True`.
- Legacy search injection: `search_internet()` (`ddgs`) can still inject live snippets into chat context, but the primary lookup path is the Brave-based quick-search pipeline (see Research flow).

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
  - The single runtime source of truth for model selection across SIX roles: `chat`, `tts`, `vision`, `browser_tool`, `listening`, `planner` (`VALID_ROLES`).
  - Persisted overrides live in `data/jarvis_settings.json` (set from the UI model switcher, immediate effect, no restart); missing/corrupt settings degrade to env defaults.
  - Env providers: `gemini`, `fireworks`, `groq`, `fish`, `gtts` (Google Translate TTS), `openrouter`, `whisper` (local), `inworld` (STT). Custom OpenAI-compatible providers are allowed for `chat` and `browser_tool` only.
  - Per-role allowlists: chat `{gemini, fireworks, openrouter}`; tts `{fish, gtts}`; vision and browser_tool `{gemini, fireworks, groq, openrouter}`; listening `{whisper, inworld}`; planner `{fireworks}` only.
  - **F49 capability-aware selection**: every role declares what it REQUIRES (`chat` streaming, `tts` audio_output, `vision` vision_input, `browser_tool` tool_calling + structured_output + vision_input, `listening` speech_input, `planner` tool_calling + structured_output + streaming) and a (provider, model) pair is only usable when those capabilities are positively established from the adapter floor, the provider record, model-family name rules, or capability metadata the provider published. Anything unknown FAILS CLOSED, resolution is validated at use time from one locked snapshot, and a persisted selection that stops validating surfaces as a `model_errors` entry in `GET /settings` instead of running.
  - API keys never leave the module — everything the routes return is masked (`has_key` booleans), and log lines are scrubbed.
  - Current live selections (verified 2026-09-30, `revision` 16): chat `gemini`/`gemini-3.5-flash-lite`; tts `fish`/`s2.1-pro-free`; vision `openrouter`/`google/gemini-2.5-flash-lite`; browser_tool `openrouter`/`google/gemini-2.5-flash-lite`; listening `whisper`/`whisper-local`; no `planner` override (so the planner falls back to its env default). The file also carries an `observed_capabilities` map populated from live provider probes. Backup copies sit next to it (`*.bak-*`).

- `backend/services/gemini_client.py`
  - Google Gemini client â€” vision and the text brain, both in OpenAI-shaped format so cascade code is provider-agnostic.
  - `GEMINI_MODEL` (`GEMINI_VISION_MODEL`, default `gemini-3.5-flash-lite`) for vision; `GEMINI_CHAT_MODEL` (`GEMINI_BRAIN_MODEL`, default `gemini-3.5-flash-lite`) for chat/intent.
  - Supports Google Search grounding for screen Q&A.

- `backend/services/fireworks_client.py`
  - Fireworks chat client; `DEFAULT_MODEL` = `FIREWORKS_MODEL` (default `accounts/fireworks/models/deepseek-v4-flash-0731`); retry logic drops `reasoning_effort` when the error body indicates a thinking-only model.
  - Also provides streaming and vision entry points used by the chat and screen cascades.

- `backend/services/grok_client.py`
  - Despite the filename, this is a Groq API client, not xAI Grok.
  - Its old default `llama-3.3-70b-versatile` is RETIRED upstream (404) and has been dropped from both the module default and the offered catalog (H8). Every live caller passes an explicit model.
  - `VISION_MODEL` default is `qwen/qwen3.6-27b`, still reachable as one eligible provider in the screen-Q&A vision cascade and as the intent-router's last fallback.

- `backend/services/openrouter_client.py`
  - OpenRouter client. It is now a first-class env provider rather than just a free-vision helper: the `chat` role allowlist includes it (added after the 2026-09-23 VPN incident), `get_provider_credentials` returns its canonical OpenAI-compatible base URL, and it also appears in the vision and browser_tool allowlists.

- `backend/services/openai_compat_client.py`
  - Chat client for user-added custom providers (any /v1-compatible endpoint).

- `backend/services/transcription.py`
  - Shared speech-to-text ladder: Google STT primary, Groq Whisper-style (`whisper-large-v3-turbo`) network fallback, local Whisper over the persistent `whisper_daemon` (`JARVIS_WHISPER_PORT`, default 8767), and Inworld STT (`INWORLD_STT_*`).
  - Hosts the STT hallucination gate (`is_hallucinated_transcript`) that every transcript-commit path consults — see the 2026-09 review wave above.

- `backend/services/ollama_client.py`
  - Local inference to the `llama3.2` model on port `11434` for the accessibility screen-control planner.

### Modules added by the Fable-5 remediation wave

One-line roles for the modules the G0-G11 work introduced that are not described above, so a future model knows where to look:

- `backend/services/vision_cascade.py` (F37) — the one eligible-provider vision dispatcher used by screen Q&A: ordering, bounding, attempt metadata.
- `backend/services/research_browser.py` (F27/F28) — one long-lived research browser worker; `submit`/`run` job API, per-task page cancellation, idle TTL.
- `backend/services/intelligence_state.py` (F50) — backend authority layer: `run_effect`/`submit_effect` (one job runtime for every effect), `WorkerRegistry` (typed worker generations; stale publishes rejected), `DurableOwnership` (single-writer leases), `EventJournal` (one shared transactional history).
- `backend/services/capability_contract.py` (F16) — immutable `ExecutionContract` (capability + selected executor + availability + grant + digest) that executors verify instead of re-reading configuration.
- `backend/services/deadline.py` (F24) — one absolute monotonic deadline/cancellation handle plus a shared replay-eligibility classifier, propagated into urllib3 retries.
- `backend/services/provenance.py` (F48) — observed/inferred/externally-checked provenance and corroboration accounting for research evidence.
- `backend/services/productivity_connector.py` + `productivity_providers.py` (F14) — typed least-privilege calendar/mail/contact operations: services stay inert until confirmed, drafts are local, and sending needs a second per-effect consent bound to the draft hash and grant epoch.
- `backend/services/tool_policy.py` (F17/F21) — dispatch-bound tool validation and secret masking; `code_grants.py` (F22) — scoped code grants + change journal.
- `backend/services/approvals.py` (F18) — plan-hash approvals; `jobs.py` (F20) — per-job cancellation tokens and turn checkpoints; `request_registry.py` (F23/F26) — request identity, numbered event buffers, reconnect.
- `backend/services/local_auth.py` (F51) — `X-Jarvis-Token` minting/verification (constant-time), renderer origins, `JARVIS_DEV_MODE` escape hatch; `runtime_identity.py` (F52) — atomic per-process instance stamps under `data/runtime/`.
- `backend/services/audio_actor.py` (F32), `echo_cancel.py` (F33), `transcript_stabilizer.py` (F34), `wake_engine.py` (F36) — the voice runtime pieces described in G10.
- `backend/services/screen_geometry.py`, `screen_ui_elements.py`, `screen_ocr.py` (F42/F43/F44) — coordinate-space helpers, the UIA cache with runtime-id re-resolution, and Tesseract word boxes with block/par/line identity.
- `backend/services/browser_session_broker.py` (F06/G6) — profile/tab/origin identity with epochs for browser grounding.
- `backend/services/context_envelope.py` (F46) — bounded request-scoped context envelope with explicit omissions.
- `backend/services/goal`-side helpers (F01/F03/F05) live in `backend/services/task_agent/` and `task_result.py` rather than as separate modules.
- `backend/core/memory_store.py` (F06/F07/F09/F10) — the single SQLite store: facts/entities, events, commitments + scheduler daemon, skills. See the G9 bullet above.
- `backend/whisper_daemon.py` (F52/F55) — the persistent local Whisper HTTP daemon (`/health`, `/warm`, `/transcribe`).
- `backend/services/google_tts.py` — the key-less Google Translate TTS engine (an interchangeable `tts` role engine alongside Fish).
- `backend/services/quick_search.py` / `research_service.py` — the two research tiers (see the research flow).

### Active voice subsystem

- `backend/voice_mode.py`
  - The always-on voice I/O WORKER once Jarvis is active (G11/F50). It captures, transcribes, submits to the backend and speaks — nothing else. It deliberately does NOT import `backend.core.brain`, because that would give this process a second, never-authoritative copy of the intelligence state.
  - **`_TurnManager` (P0-08) owns the ONE active voice turn.** `_respond_to_utterance` registers the turn and returns; `_run_turn` does the round trip on its own daemon thread, because the brain thread dispatches rather than waits. A new utterance pre-empts the old one: the old speaker is closed immediately and its backend request is cancelled through `POST /ask/cancel/{request_id}` (fired from a daemon thread — barge-in onset runs on the capture thread). `is_current(request_id)` gates both the streaming sink and the terminal reply, so a stale turn can never speak over its replacement.
  - `handle_queued_item` holds the control-vs-utterance decision (extracted from `brain_thread` so it is directly testable). A control phrase must never become a turn — that would queue a second generation behind the reply it was meant to silence.
  - Turn request ids are `voice-<epoch-ms>-<pid>-<counter>`; the counter matters, because two turns submitted in the same millisecond used to share an id and a reused id with a different message is a 409 at the request registry.
  - Has custom phrase handling for:
    - stop speaking (`POST /speak/stop`)
    - shutdown
    - continue/resume
    - normal setup (`POST /voice-setup/launch` — the app launches are now a typed backend job, not local `os.startfile` calls)
  - Publishes real listening state to `POST /voice-state/publish` and reads task state from `/ui-state`.

- `backend/services/listener.py`
  - Active conversation microphone capture.
  - Uses `speech_recognition` with `stream=True`.
  - Uses `webrtcvad` to reject obvious noise and low-confidence captures.
  - Uses multilingual recognition across `JARVIS_STT_LANGUAGES`, default `en-IN,hi-IN`.
  - Engine order comes from `transcription.recognize_multilingual` (`listener.py:746`): the registry-selected `listening`-role engine is tried first — **Inworld by default**, with local Whisper as the alternative, and Google STT then Groq as further fallbacks (one call per language in `RECOGNITION_LANGUAGES`). The live setting is currently `whisper`/`whisper-local`, so the local daemon is primary in practice. A hallucinated result falls through to the next engine. This is the order the code actually does; older notes that say "Google STT first" are stale. (P0-03, which reduces this to ONE engine, is still open.)
  - Commits a transcript only through the F34 stabiliser and the STT hallucination gate (partial windows are filtered too).
  - **Nothing blocking in the capture loop** (P1-03/P0-04): the `/speak/stop` POST goes through the `_SpeakStopWorker` daemon thread, partial transcription goes through `_PartialWorker`, and barge-in onset only notifies observers (`register_barge_in_hook`) on a daemon thread. Do not add a synchronous HTTP call or a blocking engine call back into `_capture_audio`/`barge_in_on_speech_onset`.
  - The capture loop normalises everything to one sample rate (P1-05, `assert_single_rate_audio` at the STT entry) and frame ids are `(capture_token, index)` with `AecSignalPath.begin_capture()` clearing the cache per capture (P0-13).
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
  - Since G11/F50 this module's copy is authoritative only INSIDE the owning process: the voice worker publishes its real snapshot to `POST /voice-state/publish` and the backend serves that (newest generation wins, stale generations and expired snapshots rejected) — that fused snapshot is what the UI ultimately renders.

- `backend/services/voice.py`
  - Text-to-speech orchestrator. The ladder depends on the registry `tts` role engine, which makes it easy to misunderstand:
    - **the selected engine is tried first** — Fish Audio (`speak_fish_audio`, streaming PCM through the single `audio_actor`, with next-chunk prefetch pipelining), or the key-less `google_tts` engine
    - local `pyttsx3`/SAPI5 second (with a silent-playback sanity check)
    - ElevenLabs for short chunks (`JARVIS_REMOTE_TTS_CHAR_LIMIT`)
    - local SAPI5 again as the final safety net
  - Sentence chunking + background prefetch of the next chunk. `_prefetch_tts_audio(text, plays_next=…)` is the ONLY prefetch entry point (P1-09): it resolves the selected engine per call and no-ops for an engine without a prefetch implementation, so a non-Fish engine never bills Fish. It also enforces the P0-05 rule — never warm the sentence that is about to play while the worker is idle.
  - Streaming chunking (P1-01) flushes the first chunk at the first clause boundary (≥3 words) or the last word boundary (≥30 chars), later chunks at sentence boundaries (~200 chars), and **never mid-word**; `STREAM_POLL_SECONDS` (0.05) must stay below the 120 ms punctuated-buffer settle, which must stay below the 0.3 s stall flush.
  - `stop_speaking(signal_ready=True)`: the barge-in path passes `signal_ready=False` (P1-02) so no "I can hear you" beep lands inside the user's sentence. Interruption is a generation bump + PCM flush, with the paused remainder republished on `GET /speak/remaining`.

- `backend/services/audio_actor.py`
  - The ONE playback owner for a reply (F32/F50) — nothing else may write to the output device.
  - `play()` consumes the ring and waits on `_ring_cv` with a 0.05 s floor, and `feed_chunk()` notifies that condition, so a chunk reaches the device in well under a millisecond instead of on a 250 ms poll (P0-06). The producer still waits for ring space rather than dropping audio.
  - One lazily-opened, long-lived output stream per process (`blocksize=1024`, `latency="low"`), reused across sentences and replies, restarted after a barge-in, reopened once on a write error (P0-07). `abort()` cuts pending buffers; `stop()` drains them — use abort for barge-in and never hold `_gen_lock` while waiting on `_ring_cv`.
  - The F32 pre-write generation/stale-chunk re-check and the `_spoken_bytes` cursor accounting are load-bearing (P0-08's resume path reads them). Do not "simplify" either.

- `backend/services/fish_voice.py`
  - Fish Audio TTS client: registry `tts` role model resolution (env default `FISH_MODEL`, default `s2.1-pro-free`), PCM streaming playback, prefetch/warm-up, output-device selection.

- `backend/services/google_tts.py`
  - The zero-cost, key-less Google Translate TTS engine, selectable as the `tts` role engine (`gtts` provider). Splits text at the endpoint's ~200-character limit and time-stretches 1.5x with ffmpeg `atempo` (pitch preserved). It deliberately reuses `fish_voice`'s playback primitives so there is still exactly ONE playback owner (F32).

- `backend/services/elevenlabs_voice.py`
  - Optional remote TTS provider (ElevenLabs API, MP3 via pydub).

- `backend/services/earcons.py`
  - Plays simple Windows beep patterns for ready/capture/reply cues. **Already non-blocking** (a daemon thread per cue; caller cost is sub-millisecond), so no extra worker thread is needed here.
  - The ready cue is not played on barge-in (P1-02), the reply-start cue is off by default (`JARVIS_REPLY_START_EARCON=1` restores it), and a one-cue-at-a-time claim coalesces a burst instead of spawning overlapping beeps. Earcons must never route through the audio actor — the actor is the reply's single playback owner.

### Wake watcher

- `backend/watcher.py`
  - Passive wake-word launcher AND the full-stack supervisor for `run_jarvis.bat --launch`.
  - Mints the per-launch `JARVIS_LOCAL_TOKEN` and injects it into every child it owns (backend, voice, Electron); its own control plane (`/stop` warm sleep, `/shutdown` full stop) is token-authed too.
  - Uses a dedicated recognizer with its own thresholds, `backend/services/wake_engine.py` keyword spotting (F36) and fuzzy phrase matching as the fallback.
  - Applies a fixed watcher energy threshold from `JARVIS_WATCHER_ENERGY_THRESHOLD`, clamped against the shared idle threshold.
  - Pre-loads Anaconda and Ollama CUDA v12 DLL search paths dynamically to enable local CUDA GPU speech transcription.
  - Uses a local GPU-accelerated `faster-whisper` (medium) model running with greedy decoding (temperature=0.0), VAD filtering, disabled repetition conditioning, and a wake-bias prompt applied ONLY to wake transcription (`X-Jarvis-Purpose: wake`); a persistent `whisper_daemon.py` child keeps the model warm, binds its port before loading, and reports `ready`/`loading` honestly (F55).
  - STT goes through `transcription.recognize_multilingual`'s registry-selected engine ladder (Inworld by default, local Whisper over the resident daemon, then Google/Groq per language), and every committed transcript passes the STT hallucination gate (a prompt echo must never false-launch the stack).
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

Much state is still process-local module state, but the durable artifacts are no longer "only three". On disk today: `data/jarvis_settings.json` (model selections), `data/conversation_history.json`, `data/jarvis_memory.db` (the G9 SQLite store, plus `-wal`/`-shm`), `data/change_journal/` (F22 code-grant file backups), `data/runtime/` (per-process instance stamps), `data/logs/` (bounded supervisor child logs), `data/research_reports/` and the productivity connector's grant JSON — all under `data/`, which is gitignored in full.

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
- The voice process additionally owns the P0-08 turn manager (`voice_mode.TURNS`: one active turn, pre-emption statistics, `active_request_id`) and the P1-19 mark timeline it ships to the backend (`latency.LocalTurn`, one per capture, merged into the backend record under the same `request_id`). Both are process-local by design and reach the backend over HTTP, never by module import.

### Screen-control runtime state

Location: `backend/services/screen_state.py`

- whether screen controls are enabled; any pending plan waiting for confirmation.

### Pending task/confirmation gates

Location: `backend/core/brain.py` + `backend/services/task_agent/agent.py` + `backend/services/approvals.py`

- `_pending_opencode_task` / `_pending_confirmation` / `_pending_browser_clarification` (45s expiry windows, browser clarification 90s) — the arm-confirm-execute gates for task handoffs. Only one gate may be armed at a time.
- Deferred browser runs carry a generation id (F26/F03): a superseded run may not publish a result, arm a clarification or speak.
- Approvals themselves are records in `backend/services/approvals.py`, invalidated by `/approvals/reset` before any supervisor replacement so a pending consent can never survive into a new worker.

### Work / effect state (shared history)

Location: `backend/services/intelligence_state.py`, `backend/services/jobs.py`, `backend/services/request_registry.py`

- F20 job registry: every request, task and setup effect is a typed job with its own cancellation token and deadline; the registry is the single stop surface (`/task/stop`).
- F23 request registry: request id -> numbered event buffer, so a retry or reconnect reattaches instead of re-executing.
- F50 `intelligence_state`: worker generations, durable-writer leases (memory db, checkpoints, approvals) and one transactional `EventJournal` (effects, checkpoints, approvals) shared by the typed UI, the voice worker and background work.
- F52 `data/runtime/*-instance.json`: pid + instance id + protocol + build, which is how the watcher/supervisor proves WHICH runtime owns a port.

### Audio / AEC state

Location: `backend/services/audio_actor.py`, `backend/services/echo_cancel.py`

- One playback owner with a generation token (stale-generation PCM is dropped), the PCM ring plus spoken cursor for exact interrupted resumes, and the AEC reference ring. The API process renders audio while the voice process owns the microphone, so the reference crosses the process boundary through `GET /aec/reference`.
- Since P0-07 the owner keeps ONE long-lived output stream (`blocksize=1024`, `latency="low"`) for the whole process instead of opening a device per sentence, and `abort()` cuts pending buffers where `stop()` drains them. Since P0-06 the player is woken by the ring condition variable rather than a 250 ms poll.
- The AEC frame cache is keyed by `(capture_token, index)` and cleared per capture (P0-13); `RemoteAecTransport` (the backend-spoken-audio reference path) authenticates, re-probes when idle, expires cached spans by `mic_t_end` drift, and reports a breaker/auth failure in `state()` instead of pretending the reference is merely absent (P1-04).

## Environment Variables and What They Actually Affect

Do not put real secret values into docs or prompts. The important thing is the variable names and their purpose.

### API keys (all optional individually â€” features degrade per provider)

- `GROQ_API_KEY` â€” Groq: intent-router last fallback classifier, an eligible screen-Q&A vision provider, STT fallback.
- `GEMINI_API_KEY` â€” Gemini: intent primary classifier, chat/vision env-default provider, research per-site notes.
- `FIREWORKS_API_KEY` â€” Fireworks: the ONLY allowed `planner` provider, and the browser-agent env default. NOTE: this account was suspended/failing on 2026-09-23, which is what pushed chat and vision onto OpenRouter.
- `FISH_API_KEY` â€” Fish Audio TTS (the metered primary spoken-voice engine; the `gtts` engine needs no key at all).
- `ELEVENLABS_API_KEY` â€” enables ElevenLabs TTS for short chunks.
- `OPENROUTER_API_KEY` â€” OpenRouter: now a first-class provider for chat, vision and browser_tool (Cloudflare-fronted, so it stayed fast behind the VPN that broke direct Gemini).
- `INWORLD_STT_API_KEY` â€” Inworld STT, the alternative `listening` engine.
- `CLINE_API_KEY` â€” optional browser-agent provider.

### Model selection

- `GEMINI_BRAIN_MODEL` â€” Gemini chat/intent text model, default `gemini-3.5-flash-lite`.
- `GEMINI_VISION_MODEL` â€” Gemini vision model, default `gemini-3.5-flash-lite`.
- `GEMINI_INTENT_MODEL` â€” intent-router Gemini override.
- `GROQ_MODEL` â€” legacy Groq text default. The retired `llama-3.3-70b-versatile` is no longer referenced anywhere (H8).
- `JARVIS_WHISPER_PORT` â€” local Whisper daemon port, default `8767` (also read by `whisper_daemon.py`).
- `INWORLD_STT_MODEL` / `INWORLD_STT_URL` â€” Inworld STT model/endpoint for the `listening` role.
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

### Voice runtime, AEC and wake detection (G10 / F31-F36)

- `JARVIS_AEC_ENABLED` â€” acoustic echo cancellation on/off, default on.
- `JARVIS_AEC_REMOTE` / `JARVIS_AEC_REMOTE_TIMEOUT` â€” let the voice process fetch the AEC reference from the API process over HTTP; this is required because the API process voices replies while the voice process owns the microphone. `JARVIS_AEC_REMOTE=0` is for an isolated local-only setup.
- `JARVIS_WAKE_ENGINE` â€” `auto` (default) / openwakeword / fuzzy matching.
- `JARVIS_WAKE_MODELS_DIR`, `JARVIS_WAKE_ONLINE_VERIFY`, `JARVIS_WAKE_PRE_ROLL_SECONDS` â€” wake-model directory, online whisper verification toggle, and how much audio is pre-rolled so the first syllable is not clipped.
- `JARVIS_WHISPER_MODE` â€” `conversation` (default) or wake-oriented local Whisper use.

### Memory & continuity (G9)

- `JARVIS_MEMORY_ENABLED` â€” default on; `0` makes every memory call a safe no-op (an empty store changes zero prompts).
- `JARVIS_MEMORY_DB` â€” SQLite path, default `data/jarvis_memory.db`.

### Routing / orchestration

- `JARVIS_ORCHESTRATOR_MODE` â€” `legacy` (default) or `orchestrator`. In `orchestrator` mode the native tool-use loop is tried first and the legacy routing (plus every deterministic net) still runs on a decline. Orchestrator parity is NOT established, so do not default this to `orchestrator`.

### Local control plane & security (G11 / F51)

- `JARVIS_LOCAL_TOKEN` â€” the per-launch command token, minted by the watcher (or `main.js` in direct-launch mode) and injected into every child. With it set, every non-public endpoint requires the `X-Jarvis-Token` header; without it, `local_auth` FAILS CLOSED.
- `JARVIS_DEV_MODE` â€” `1` explicitly opens the local surface for development. This is the ONLY way to get the old permissive behavior.

### Productivity connectors (F14)

- `JARVIS_PRODUCTIVITY_CONFIG` â€” path of the connector's grant JSON (gitignored, under `data/`). Services stay inert until the user confirms them, and an unconfirmed/unauthenticated service refuses every operation.

### Browser automation / brave-control

- `BRAVE_MCP_MODE` â€” `stdio` (default; spawned per opencode session) or `http` (persistent daemon, `BRAVE_MCP_PORT` default `9570`, `BRAVE_MCP_TOKEN`, `BRAVE_MCP_IDLE_MIN` browser recycle).
- `JARVIS_BROWSER_AGENT_*` â€” see the task-engine section above.

### Internal process coordination

- `JARVIS_EXTERNAL_RUNTIME` â€” set by the watcher before launching Electron; tells `main.js` not to spawn backend and voice mode again.
- `JARVIS_BACKEND_PORT` â€” backend port, default `9999`.
- `JARVIS_BACKEND_PID` / `JARVIS_ELECTRON_PID` â€” passed to the voice process for targeted shutdown.
- `JARVIS_WATCHER_CONTROL_PORT` â€” watcher HTTP control endpoint used by Electron stop.

### Running the test suite

The suite is no longer 17 modules. `backend/tests/` now holds **105 `test_*.py` modules** (the F01-F55 audit suites, the G3-G11 suites, the C1-C3 and H1-H9 review suites, the 13 `test_p*_*` modules added by the 2026-09-30 responsiveness wave, plus the original behaviour suites), and `tests/` holds 4 Node test files. The root `conftest.py` guards the developer's real `.env` against any test that would modify or delete it.

Run everything with the backend venv from the repo root (pytest is pinned in `requirements.txt`, and the root `conftest.py` is pytest-shaped):

```
& backend\venv\Scripts\python.exe -m pytest backend\tests -q
```

The original 17-module unittest invocation still works and still names a useful fast subset:

```
& backend\venv\Scripts\python.exe -m unittest backend.tests.test_code_tools backend.tests.test_task_agent backend.tests.test_brain_gate backend.tests.test_voice_task_mute backend.tests.test_opencode_lifecycle backend.tests.test_browser_agent backend.tests.test_screen_control backend.tests.test_voice_mode_toggle backend.tests.test_chat_race backend.tests.test_voice_latency backend.tests.test_model_registry backend.tests.test_settings_routes backend.tests.test_fireworks_reasoning backend.tests.test_model_roles_wiring backend.tests.test_live_bugfixes backend.tests.test_websearch_interruption backend.tests.test_websearch_modes
```

A single module runs as `python -m unittest backend.tests.test_screen_control`. The Node tests have no harness entry (`npm test` in `package.json` is still a stub) and are run directly:

```
node --test tests\backend-request-policy.test.js
node --test tests\model-sidebar.test.js
node --test tests\overlay-ipc-contract.test.js
node --test tests\research-overlay-href-scheme.test.js
```

On counts: the last full-suite figure recorded in this map is **2507 passed, 4 failed, 1 skipped (2026-09-30, 105 modules)**. Do not quote the old 512 (2026-09-11, 17 modules) figure. Wall time varies run-to-run (roughly 15s–300s) — the variance is known and comes from a few unmocked live network calls in `test_websearch_modes`, not from flaky assertions. The 4 failures in that run were `test_browser_agent.py` (×3) and `test_f02_goal_routing.py` (×1), which are **order-dependent** rather than broken: they pass in isolation. A number of other failures (`test_verification_uses_fireworks`, `test_verification_uses_groq`, `test_f37_groq_prerequisite`, the `test_g3_request_streaming` racer test, the `live_bugfixes` vision tests) also appear and disappear run-to-run; the reliable way to attribute a failure is to A/B it against a `git stash` of your own change rather than trusting a single run. Advice that survives all of that: run the full suite once per change, and confirm any new failure against HEAD before believing it is yours.

## Non-Obvious Behaviors Another Model Should Know

### 1. `backend/main.py` is the real backend entrypoint

`backend/app.py` exists, but the active launch commands use `backend.main:app`.

### 2. `grok_client.py` is named misleadingly — and its old default model is dead

It talks to Groq, not xAI Grok. Its former code default `llama-3.3-70b-versatile` is retired upstream (HTTP 404) and has now been removed from both the module and the offered catalog (H8); every live caller passes an explicit model (e.g. `qwen/qwen3.6-27b`). Never route new chat traffic through a bare Groq default.

### 3. The classifier can silently degrade — the deterministic nets exist because of it

`classify_intent` lands on a `chat` verdict whenever its whole chain fails or throttles (today OpenRouter -> Gemini -> Groq), and it can genuinely misread screen questions as chat/research. Four deterministic backstops in `process_message` rewrite or reroute such verdicts: the screen-question net, the fresh-info auto-search net, the all-`search`-steps -> research reroute, and the web-task routing net. When changing routing, check the nets, not just the classifier branch.

### 4. Voice-path defects: the three confirmed ones are FIXED (2026-09-30)

The three defects that were confirmed present in the running code have all been fixed. This block is kept so a future session knows they were real, what the fixes are, and what to reach for when the voice path misbehaves again:

- **The AEC frame cache replaying the previous utterance's audio — FIXED** (P0-13, `deee60f`). `listener._capture_audio` restarted `frame_id` at 0 each capture while `_frame_cache` is a process-lifetime singleton, so from the second capture onward frames came back from the prior turn's cache — stale PCM plus stale `had_reference`/`suppressed`, never re-processed by the AEC. That was the cause of self-barge-in. Frames are now identified by `(capture_token, index)` from a process-wide counter, and `AecSignalPath.begin_capture()` clears the cache/order/last-id at the start of every capture. The within-capture dedup the cache exists for is deliberately preserved.
- **The remote AEC reference transport — FIXED** (P1-04, `f550642`). It sent no token (every fetch 401'd under fail-closed auth), its idle guard latched off permanently, its TTL cache ignored `mic_t_end`, and it had no breaker. All four are addressed; `state()` now distinguishes `auth_failed` / `circuit_open` from "no reference". It remains the BACKEND-spoken-audio path only — the local reference ring is still what serves voice turns.
- **Barge-in stopping audio but not the turn — FIXED** (P0-08, `6c89095`). `/speak/stop` only called `stop_speaking()`; the running request kept generating and the next utterance queued behind it. `POST /ask/cancel/{request_id}` now cancels that one request, the voice worker runs each turn on its own thread, and barge-in onset cancels the active turn. **The full generated reply still commits to history** — that is an owner decision, not an oversight; do not "fix" it into a spoken-prefix-only record.

Two things found while fixing these, worth knowing:

- The voice turn's request id was `voice-<epoch-ms>-<pid>` and **collided** when two turns were submitted inside one millisecond. Mostly theoretical before, but pre-emptive dispatch submits back-to-back and a reused id carrying a different message is a **409 conflict** at the request registry. It now carries a per-process counter, and `_TurnManager.start()` additionally pre-empts on a changed speaker so an id collision can never leave two turns registered.
- **The listener's blocking work is gone but the async replacements are only as good as their timeouts**: `_SpeakStopWorker` (P1-03) and the turn manager's cancel both post from daemon threads, so a hung endpoint costs a thread, not the capture loop. If you see barge-in latency, look at `speak_stop_stats()` and the `barge_in_stop{ok,ms,status}` latency mark before changing any VAD constant.

### 5. Screen control is not always active

The screen subsystem exists even when idle, but actual natural-language screen actions are rejected until the user turns screen controls on.

### 6. Voice replies are normally spoken for typed UI requests too

`POST /ask` / `POST /ask/stream` speak the reply unless the caller passes `speak=False`. The typed UI keeps the default; the voice I/O worker sets `speak=False` (plus `origin=voice`) because IT owns playback, and the backend must never be a second playback authority (F50).

### 7. Stop behavior is targeted, and the port-9999 kill is NOT at Electron startup

`stop-jarvis` kills only tracked backend/voice PIDs (`taskkill /PID <pid> /T /F`). External-runtime mode asks the watcher via HTTP `/stop` before falling back to `fallbackExternalCleanup()` (port-ownership kill of the backend + command-line kill of voice/whisper processes). The stale-backend kill at LAUNCH lives in the watcher, not in `main.js`. `/task/stop` cancels ONE job at its next checkpoint rather than flipping a global flag, and the watcher distinguishes a warm sleep from a full shutdown. Deploying backend changes = restart the app (or let the watcher relaunch), since nothing hot-reloads; model selection is the exception (it is per-message state).

### 8. The repo has hardcoded machine-specific paths

Examples include:

- repo base path
- desktop shortcut locations
- Brave executable location
- the expected virtualenv path
- the brave-control MCP server directory (`C:\Users\mayan\mcp-servers\brave-control`) — note a second copy is now vendored at `integrations/brave-control/`
- the Chrome/Brave profile and Start Menu search roots used by `executor.launch_app`

### 9. Dependencies: `requirements.txt` is the proven set, `backend/requirements.txt` is legacy

After H1, the root `requirements.txt` was regenerated by pip freeze from the working venv and now carries a header saying so — that IS the proven install set (numpy 1.26.4 ABI, pywinauto 0.6.9, the ~12 packages the old file omitted). `backend/requirements.txt` still exists, and neither file perfectly documents every runtime import; if something fails on a missing package, inspect the actual imports before trusting either file.

### 10. The frontend is polling, not event-driven

Still no websocket â€” the renderer polls one adaptive `/ui-state` endpoint (fast while a request or voice activity is in flight, slower when idle, with failure backoff), plus separate polls for the `/screen-answer` and `/research-result` overlays. The one genuinely streaming path is `/ask/stream` (SSE).

### 11. Model selection is live and persisted

`data/jarvis_settings.json` overrides env defaults for all SIX roles per message â€” reading only `.env`/`config.py` will give you the wrong picture of which model actually answers. The registry masks API keys; never log or echo them. `data/` is gitignored precisely because custom-provider keys live only there — never in `.env`, never in git.

### 12. The local control plane FAILS CLOSED (this bites everyone once)

`backend.main` installs `local_auth`. With a supervisor-injected `JARVIS_LOCAL_TOKEN`, every endpoint except `GET /health` requires the `X-Jarvis-Token` header — including private READS such as `/ui-state` and `/settings`. Without any token the surface is closed, not open. If you hand-call the API with curl/Invoke-RestMethod and get 401s, that is the design: launch through `run_jarvis.bat` / `npm start`, or set `JARVIS_DEV_MODE=1` for a dev shell. CORS is restricted the same way (renderer origins plus loopback only, credentials off).

### 13. Only the trusted chat window can obtain the token

`main.js` registers the trusted webContents id at window creation and the preload exposes `get-local-secret` to that window only. Overlay windows (which render web-derived content) get an empty secret and must go through the main-process request proxy, whose per-sender policy is a pure function covered by `tests/backend-request-policy.test.js`. The capsule window gets a narrower channel still: an action enum (`ask` / `task-stop` / `speak-stop`) mapped in main to fixed endpoints, never a URL.

### 14. `integrations/brave-control/` is now vendored in this repo

Older notes describe brave-control as a separate, unversioned directory at `C:\Users\mayan\mcp-servers\brave-control`. A copy now lives in-tree at `integrations/brave-control/` (`server.mjs` plus `lib/dom_inventory.mjs`, `lib/fs_tools.mjs`, `lib/retry.mjs`, `lib/tab_tools.mjs`), and `BRAVE_MCP_SERVER_DIR` decides which copy actually runs. The runtime default may still point at the external path — check before assuming which one is live.

### 15. `.env` is load-bearing

`backend/config.py` calls `load_dotenv(ENV_PATH, override=True)` at import time, so real environment variables do NOT win over `.env` unless that file is absent. The root `conftest.py` snapshots `.env` once per test session and fails (then restores) any test that modified or deleted it.

## Where To Change Things

If another model is asked to make a specific kind of change, this is the shortest path to the right files:

### Change chat behavior or LLM prompting

Start in:

- `backend/core/brain.py` (`_build_chat_messages`, `_stream_chat_deltas`, `_resolve_chat_model`)
- `backend/services/gemini_client.py`, `backend/services/fireworks_client.py`
- `backend/services/model_registry.py` (provider/model selection)

### Change intent routing or the safety nets

Start in:

- `backend/services/intent.py` (classifier prompt + chain — **the real hop order is OpenRouter → Gemini direct → Groq**, sharing one monotonic budget via `_budget_timeout`; the module docstring's "Gemini first" description is stale, and the comment at line 286 is also mislabelled "3")
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
- `backend/services/wake_engine.py` (F36 keyword spotting vs fuzzy fallback)
- `backend/whisper_daemon.py` (the resident local Whisper; `/warm`, `/health`, purpose-scoped prompt)
- `backend/services/transcription.py` (engine ladder AND the hallucination gate)
- `backend/services/audio_input.py`

### Change active voice listening or interruption behavior

Start in:

- `backend/services/listener.py`
- `backend/listener_state.py`
- `backend/voice_mode.py` (the I/O worker; it must never import `backend.core.brain`)
- `backend/services/audio_actor.py` (the single playback owner), `echo_cancel.py` (AEC), `transcript_stabilizer.py` (F34)
- `backend/services/intelligence_state.py` (worker generations, playback ownership, event journal)

**Measure before you change anything here.** `/latency` (P1-19) already renders the whole turn as an ordered waterfall with p50/p90/max per step, sorted slowest-first, and it stitches the voice process's capture/STT/TTS marks into the backend's record under one `request_id`. If you are chasing latency, read that first: the marks (`speech_end`, `capture_end`, `stt_start`, `stt_done{engine}`, `http_in`, `racer_start`, `classify_done{hop}`, `first_token`, `provider_headers`, `tts_first_byte`, `playback_started`, `barge_in_stop{ok,ms,status}`) tell you which stage is actually slow instead of inviting a guess. `speak_stop_stats()` covers the P1-03 stop worker, and `voice_mode.TURNS.snapshot()` covers turn pre-emption.

Two hard rules that have already bitten this area: nothing blocking may enter `_capture_audio` or `barge_in_on_speech_onset` (they run on the real-time capture thread), and `PAUSE_THRESHOLD_SECONDS` / the end-of-speech decision / `MAX_PHRASE_SECONDS` / `LISTEN_TIMEOUT_SECONDS` are frozen by an explicit owner decision (P0-02, declined).

### Change TTS or spoken reply behavior

Start in:

- `backend/services/voice.py` (the Fish -> local SAPI5 -> ElevenLabs -> local SAPI5 ladder)
- `backend/services/fish_voice.py` (Fish PCM streaming)
- `backend/services/google_tts.py` (the key-less `gtts` engine)
- `backend/services/elevenlabs_voice.py`
- `backend/services/earcons.py`

### Change screen controls

Start in:

- `backend/services/screen_control.py` (UIA-first fast path F29, planning, untrusted-text delimiters, the single declared coordinate frame, key/click whitelisting with clamped counts)
- `backend/services/screen_capture.py` (F43 coordinate contract, monitor/DPI bookkeeping, last non-Jarvis target, `capture_epoch`)
- `backend/services/screen_executor.py` (F42 revalidate-before-act execution; epoch + window-identity refusal per effect)
- `backend/services/screen_geometry.py` (coordinate-space helpers, monitor bookkeeping, F44 rect comparison)
- `backend/services/screen_ui_elements.py` (UIA cache with runtime-id re-resolution, F44 preserved hierarchy, H3 abort-on-focus-failure, H7 COM apartment init)
- `backend/services/screen_ocr.py` (Tesseract word boxes with block/par/line identity, F44 merging)
- `backend/services/screen_analyzer.py` (screen Q&A prompts + provenance)
- `backend/services/vision_cascade.py` (F37 eligible-provider vision dispatch and ordering)
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

### Change memory / commitments / skills (G9)

Start in:

- `backend/core/memory_store.py` (single SQLite store: facts/entities, events, commitments + scheduler daemon, skills, and the explicit phrase ops)
- `backend/core/brain.py` (memory phrase routing runs before every other route)
- `backend/services/intelligence_state.py` (it holds the durable-writer lease for the memory db)

### Change approvals, tool policy or code grants

Start in:

- `backend/services/approvals.py` (F18 plan-hash approvals — note M1 is still open: affirmation words anywhere in a reply count as consent)
- `backend/services/tool_policy.py` (F17 dispatch-bound validation, F21 secret masking)
- `backend/services/code_grants.py` (F22 scoped grants + `data/change_journal/`)
- `backend/services/capability_resolver.py` + `backend/services/capability_contract.py` (F16: the executor choice is frozen at consent)
- `backend/services/productivity_connector.py` (F14: sending needs a second per-effect consent bound to the draft hash and grant epoch)

### Change model selection or capability rules

Start in:

- `backend/services/model_registry.py` (`VALID_ROLES`, `ROLE_CAPABILITIES`, `_ROLE_ALLOWED_ENV`, `_ROLE_ALLOWS_CUSTOM`, `_MODEL_CAPABILITY_RULES`, `add_custom_provider`, `test_custom_provider`)
- `backend/api/routes.py` (`/settings/*`, `/settings/provider`, `/settings/provider/test`, `/providers/{id}/models`)
- `frontend/renderer.js` (the per-role provider sections; the per-functionality add/test/save/choose form)
- `data/jarvis_settings.json` (live state incl. custom provider keys; gitignored)

### Change process supervision, auth or the local control plane

Start in:

- `main.js` (window factory, IPC policy, managed-child supervision and restart budget, bounded child logs)
- `backend/services/local_auth.py` (token mint/verify, renderer origins, `JARVIS_DEV_MODE`)
- `backend/watcher.py` (launch/stop, warm sleep vs full shutdown, control port)
- `backend/services/runtime_identity.py` (`data/runtime/*-instance.json`)
- `tests/backend-request-policy.test.js` (the pure policy functions)

### Change or add tests

Start in:

- `backend/tests/` — 120 unittest modules; per-feature suites are named `test_fNN_*` (Fable-5), `test_gNN_*` (G-groups), `test_cN_*` (review criticals), `test_hN_*` (review highs), `test_pNN_*` (priority/audit waves), plus 21 unnumbered feature suites (including `test_custom_provider_ui.py` and `test_stt_hallucination_gate.py`)
- `tests/` — Node `node --test` suites
- `conftest.py` — the `.env` session guard

Do not quote a stale total from an older copy of this map. Two suites were already failing before any recent work — `test_verification_uses_fireworks` and `test_verification_uses_groq` — and they fail identically on unmodified code, so they are not a regression from your change.

## Minimal File Map

Top-level directories and files that matter:

- `frontend/`
  - Electron renderer UI: `index.html`, `renderer.js`, `style.css`, plus the overlay renderers (`overlay_renderer.js`, `overlay_images_renderer.js`, `research_overlay_renderer.js`), the capsule (`capsule.html`, `capsule_renderer.js`, `capsule.css`, `capsule_main.js`), `preload.js`, and the `snapshot-v3.html` design snapshot
- `backend/`
  - Python backend (`main.py`, `api/routes.py`, `core/`, `services/`), `voice_mode.py`, `watcher.py`, `whisper_daemon.py`, `scripts/tail_activity.ps1`
- `main.js`
  - Electron main process: window factory, IPC policy, managed-child supervision
- `run_jarvis.bat`
  - full-stack launcher (now delegates to `backend.watcher --launch`)
- `run_watcher.bat`
  - wake-word-only launcher
- `run_jarvis_noisy.bat` / `run_watcher_noisy.bat`
  - the same two launchers with loud-room mic thresholds preset
- `tests/`
  - Node (`node --test`) suites for the Electron/main-process policy layers
- `integrations/brave-control/`
  - vendored copy of the brave-control MCP server (`server.mjs` + `lib/`)
- `integrations/jarvis-editor-bridge/`
  - VS Code-compatible editor bridge extension
- `PROJECT_MAP.md`
  - this map
- `AUDIT_IMPLEMENTATION_PROMPTS.html`
  - the 2026-09-30 responsiveness-audit triage: the owner's decision per finding plus a self-contained implementation prompt for each, in phase order. **13 of the 43 items have now been implemented** (see "Responsiveness audit implementation state"); the remaining prompts are still the place to start for the rest.
- `codex.md`
  - the project's running change log (newest entries at the bottom) — read it for the 2026-08/09 history this map summarises
- `CODE_REVIEW_REPORT.txt` / `.pdf`
  - the 2026-09-23 independent code review (55 findings) this map's remediation section tracks
- `screen_commands.log`
  - persistent audit log of all screen control actions and execution steps
- `conftest.py`
  - pytest session guard for the real `.env`
- `data/` (gitignored, all of it)
  - `jarvis_settings.json` (+ backups), `conversation_history.json`, `jarvis_memory.db`, `change_journal/`, `runtime/`, `logs/`, `chrome_profile_jarvis/`, `research_reports/`

Mostly non-runtime or secondary:

- `graphify-out/`
  - generated architecture artifacts
- `.audit_tmp/`, `_plan.txt`, `_repro*.py`, `_r3.txt`–`_r5.txt`, `.git-broken-20260914-154720/`
  - scratch/repro/backup leftovers from the audit work; not part of the runtime

## Responsiveness Audit State (2026-09-29 triage — 13 of 43 implemented on 2026-09-30)

A responsiveness audit produced 43 findings (P0-01…P1-19). The owner triaged every item; the decision and a pasteable implementation prompt for each lives in `AUDIT_IMPLEMENTATION_PROMPTS.html` at the repo root. **13 items have since been implemented, one at a time, in the phase order that document gives (P1-19 telemetry first)** — see "Responsiveness audit implementation state" below for the commit and the substance of each. The standing decisions below are recorded so a future session does not re-litigate them.

**The owner's standing decisions:**

- **P0-01 (slow Gemini over VPN) — no action.** The VPN was turned off, so the measured 7–45 s figure no longer applies. **Do not change the chat provider.** Re-measure once P1-19 telemetry exists. (Note: the audit's cited model `gemini-3.8-flash` was stale; the live selection is `gemini-3.5-flash-lite`.)
- **P0-02 (1.2 s energy-only end-of-speech) — DECLINED, keep as is.** This is now a hard constraint: **no change may be made to `PAUSE_THRESHOLD_SECONDS`, the end-of-speech decision, `MAX_PHRASE_SECONDS`, or `LISTEN_TIMEOUT_SECONDS`.** Several planned items (P0-04, P1-05) carry explicit constraints to that effect.
- **P0-03 (serial STT ladder) — implement, but as ONE engine only.** The cross-engine fallback chain and the Google/Groq language loop are to be removed. This also makes P0-04's partial-agreement early-exit meaningful, since both come from the same engine.
- **P0-08 (barge-in does not cancel the backend turn) — implement, with one carve-out. IMPLEMENTED (`6c89095`).** The full generated reply **stays committed to history** after an interruption, because the owner wants to read what was missed. The audit's "store only the spoken prefix" suggestion is explicitly rejected; an additive `last_reply_interrupted` flag is the substitute. **Do not reverse the carve-out** — the reply text is untouched by design.
- **P0-09 (classifier gating) — suggestion only, do not implement yet.** See the analysis in `AUDIT_IMPLEMENTATION_PROMPTS.html`; the short version is that a local Qwen3-0.6B via the existing Ollama client is the recommended candidate, used as a local-first / cloud-fallback cascade rather than a replacement, and only after a labelled evaluation set exists.
- **P0-07 — implement only if it measurably helps. IMPLEMENTED (`6d1bedc`) after the gate fired.** It was split into a measurement gate, an `abort()` fix to do regardless, and a persistent-output-stream refactor to do only if the gate justified it; the gate measured a 400 ms inter-sentence gap on a Bluetooth default device (and a 221 ms `stop()` drain), so **both** parts were done. If you change the device setup, re-run that gate before trusting the persistent stream.
- **P1-17 (wake cold-start) and P1-18 (Whisper daemon settings) — SKIPPED.**

Everything else (P0-04, P0-05, P0-06, P0-10…P0-13, P1-01…P1-16, P1-19) is approved for implementation, one item at a time, in the phase order given in the prompts document (P1-19 telemetry first). **Landed so far: P0-04, P0-05, P0-06, P0-07, P0-08, P0-13, P1-01, P1-02, P1-03, P1-04, P1-05, P1-09, P1-19.** Still pending from that list: P0-03 (its own decision above), P0-10…P0-12 and P1-06…P1-08, P1-10…P1-16.

### Responsiveness audit implementation state (the 2026-09-30 wave)

**Landed, in order.** Each entry gives the commit, what actually changed, and the measured before/after where the item was about latency. Every item has a dedicated `backend/tests/test_p*_*.py` suite; all 13 measured their defect before fixing it.

- **P1-19 — real per-turn latency waterfall (`f37a519`).** `backend/services/latency.py` was rebuilt: a record is now a list of `(name, perf_counter_ns, meta)` tuples holding **absolute timestamps only**, and every duration is differenced at read time (the old ring mixed offsets and durations — the actual bug). Backend marks: `http_in` (in the ROUTE, not the worker thread), `racer_start`, `classify_done{hop}` naming which classifier hop answered, `provider_headers`, `first_token`. Voice marks: `speech_end`, `capture_end`, `stt_start`, `stt_done{engine,language}`, `tts_first_byte`, `playback_started`. `POST /latency/client` (authenticated) merges the voice process's marks into the same `request_id`, and the early batch rides the existing `/ask/stream` submission (`client_marks` + `client_now_ns`) — never a second id, with one clock-offset per turn so late batches cannot drift. `/latency` no longer imports `listener` (the AEC counter arrives in the published snapshot); reporting adds a `waterfall` of p50/p90/max per step sorted slowest-first while keeping the legacy `median_ms`/`p90_ms`/`max_ms` keys. `test_p1_19_latency_waterfall.py`.
- **P0-13 — AEC frame-cache collision (`deee60f`).** See "Voice-path defects" above. Proven pre-fix: two consecutive captures returned **identical** STT input with `replayed_frames` 4/4; post-fix `0` and distinct audio. `test_p0_13_aec_frame_cache.py` (plus one F33 assertion updated to the new id shape).
- **P1-04 — remote AEC transport (`f550642`).** Auth header, the idle latch replaced by a bounded re-probe (`idle_reprobes`), the cache keyed on `mic_t_end` with read-time drift expiry, a breaker (`AEC_BREAKER_FAILURES`/`COOLDOWN`), and a non-raising guard. It stays synchronous on purpose — a `RESIDUAL RISK` comment records why threading was deferred. `test_p1_04_remote_aec_transport.py`. **One pinned test was changed deliberately**: `test_latency_reductions.py::test_idle_period_skips_the_request_entirely` asserted the unrecoverable latch, so it was replaced by a recovery test plus a bounded-probe test.
- **P1-05 — mixed sample rates on one capture (`07796f9`).** Mic now opens at 16 kHz so AEC and native rates coincide; when a device refuses that, mixed chunks are resampled with the existing `StatefulResampler` (one per source rate) and anything that cannot be described at the AEC format refuses to join (`None`) rather than being mislabelled. `assert_single_rate_audio` enforces the invariant at the STT entry (`recognize_multilingual`), where `recognize_inworld` builds its WAV. Pre-fix the joiner claimed **4.000 s for 2.000 s of audio**. `test_p1_05_sample_rate_join.py`.
- **P1-03 — blocking HTTP out of the capture loop (`11ddee7`).** `_SpeakStopWorker`: one daemon thread, one persistent `HTTPConnection`, a coalescing "stop needed" flag cleared *before* the send (so the newest stop is never lost and duplicates never queue), the `GET /voice-state` probe deleted, 401/403 logged once and counted (`auth_failures`), and a `barge_in_stop{ok,ms,status}` mark. Measured against a hung endpoint: `barge_in_on_speech_onset` **10,016 ms → 0 ms**. `test_p1_03_async_speak_stop.py`; four existing assertions about the probe/silent-everywhere no-op were updated.
- **P0-04 — partial transcription off the capture loop, and actually used (`e491c40`).** `_PartialWorker` over `queue.Queue(maxsize=1)` (newest-only, dropped-oldest pending, in-flight never dropped), the deadline-less `TypeError` retry removed, partials only start after onset is confirmed, `PARTIAL_MAX_AUDIO_SECONDS` 20 → 6 so the cap is finally below `MAX_PHRASE_SECONDS`, a cached whisper-daemon readiness gate (fails OPEN when unreachable), and two agreeing partials + 250 ms of trailing silence now end the capture early using the stabilizer's commit — with the hallucination gate unchanged on that path. Measured: capture loop **2.45 s → 0.05 s** for the same six windows. `test_p0_04_async_partials.py`.
- **P0-06 — the audio actor wakes on chunk arrival (`6662bd8`).** `feed_chunk` appended to the ring but notified `_chunk_wait` — an Event `play()` never waited on — while the player slept on `_stop.wait(0.25)`. It now waits on the existing `_ring_cv` with a 0.05 s floor and `feed_chunk` notifies that condition. Measured feed→device-write **125–203 ms → <0.1 ms**. `_chunk_wait` is vestigial and was left alone. `test_p0_06_audio_actor_wake.py` (includes two concurrency stress tests).
- **P0-07 — abort cuts instead of draining, and one output device per process (`6d1bedc`).** Part 0 measured before choosing: inter-sentence gap p50 **400 ms** on the Bluetooth default device (vs a ~30 ms threshold) and `stop()` drain 221 ms, so **both** parts were implemented. Part 1: `abort()` on `SoundDeviceStream` and `RawPcmStream` with a `stop()` fallback, one actor abort per stop (not two), pyttsx3 interrupted only while its event loop actually runs. Part 2: one lazily-opened, long-lived stream per actor, `blocksize=1024`/`latency="low"`, restart (not reopen) after a barge-in, one lazy reopen on a write error then `last_error`, `shutdown_actor()` at exit. Result: **9 device opens → 1** across 9 sentences, gap p50 **−0.10 ms**, abort-to-silence 8.94/1.86/1.39 ms. `test_p0_07_audio_actor_cut_and_persist.py`. Two existing tests updated (`test_f32_playback_owner` teardown split; `test_voice_latency` stream params).
- **P0-05 — an in-flight synthesis is JOINED, not waited for (`1c82a0a`).** `_InflightPCM` publishes chunks into a growing buffer under a `Condition`; a `play=True` caller reads it as bytes arrive (`iter_from`), so a prefetch already in flight stops being a penalty. Both owner and joiner stream through one shared `_stream_pcm_to_actor` (a second playback implementation would have drifted). The prefetch guard never warms the sentence about to play while the worker is idle, and the in-flight map is bounded (`_INFLIGHT_MAX`). Measured time-to-first-audio with the prefetch deliberately winning the race: **281 ms → 47 ms**. `test_p0_05_prefetch_stream.py`.
- **P1-09 — every prefetch goes through the engine-aware helper (`f74a304`).** Both direct `prefetch_fish_audio` call sites now route through `_prefetch_tts_audio`, which resolves the engine once per call and does nothing for an engine without a prefetch implementation — the old `if gtts … else fish` shape silently billed Fish for every non-Fish engine. The implementation map is built **per call** from the module namespace so `patch.object(voice_mod, "prefetch_fish_audio")` keeps working (a module-level map of callables broke the suite and would have let a "no Fish call" test pass while a real metered call ran). Measured with Google selected: Fish calls **1 → 0**, Google **0 → 1**. `test_p1_09_engine_aware_prefetch.py`.
- **P1-02 — earcons off the hot path (`bf82dfa`).** The audit's assumption was wrong and was verified first: `earcons.py` was **already** non-blocking (a daemon thread per cue; caller cost sub-millisecond), so no worker thread was invented. What did change: the ready cue is dropped on **barge-in** (`stop_speaking(signal_ready=False)`), the reply-start cue is off by default (`JARVIS_REPLY_START_EARCON`) because instant speech is the better cue, and a non-blocking one-cue-at-a-time claim coalesces bursts instead of spawning N overlapping beeps. `test_p1_02_earcons_off_hot_path.py`; four `stop_speaking()` call-contract assertions updated.
- **P1-01 — streamed chunking cuts sooner and never mid-word (`7720cdd`).** First chunk flushes at the first clause boundary `[,;:—.!?]` once ≥3 words are buffered — punctuation may be the **last** character of the delta, which is what removes the wait for the next token — or at the last word boundary once ≥30 chars are buffered. Later chunks flush at sentence boundaries up to ~200 chars, always at a word boundary, and a punctuated buffer flushes after a 120 ms settle (`STREAM_POLL_SECONDS` lowered to 0.05 so that rule can fire). The 40-char mid-word flush is gone; a chunk that cannot end on a word boundary waits instead. `test_p1_01_stream_chunking.py`. Note the deliberate trade-off: a period that is the last character of a delta counts as a clause end, so a decimal split exactly across deltas could be cut there (pin: mid-buffer `3.5` is never cut).
- **P0-08 — barge-in cancels the backend turn (`6c89095`).** See "Voice-path defects" above. `POST /ask/cancel/{request_id}` is request-scoped and refuses to touch a finished turn; `_TurnManager` owns the one active turn; each turn runs on its own worker thread; `listener.register_barge_in_hook` notifies onset (non-blocking) and `_on_barge_in` cancels. Measured: submission of the interrupting utterance **3.015 s → 0.005 s**, and the old turn is now cancelled (it never was). `/ui-state` gains an additive `last_reply_interrupted` marker that carries **no reply text** — the reply itself is untouched in history. `test_p0_08_turn_preemption.py`; three existing tests updated for the async handover.

**Still open from the audit:** P0-03 (collapse the STT ladder to ONE engine — its decision was a prerequisite for P0-04's agreement path, which is implemented but currently only meaningful once one engine is used), P0-09 (classifier gating — suggestion only), P0-10…P0-12, and P1-06…P1-08 / P1-10…P1-16. P0-01 has no action, P0-02 is declined, P1-17/P1-18 are skipped. The 43-finding list is the source of truth; do not invent extra scope.

**Cross-cutting lessons from the wave, worth reading before touching this area:**

- **Async-ing a caller breaks tests that observed the old synchronous side effect.** Existing assertions in `test_p1_19_latency_waterfall`, `test_voice_task_mute`, `test_p1_03_async_speak_stop`, `test_websearch_interruption`, `test_live_bugfixes` and `test_f50_f51_voice_ownership` all had to be updated for it, because the effect now lands on a worker thread. When a fix moves work off a thread, grep for the tests that assert its *timing*, not just its outcome.
- **When you wait for a thread's side effect in a test, wait INSIDE the patch window.** A wait placed after the `with patch...` block lets the worker reach the real function (in one case the real `speak()`, which blocked on TTS) — the failure looks like "the reply was never spoken" and has nothing to do with the assertion.
- **`voice.py` resolves TTS engines per call from the module namespace**; do not hoist engine callables into module-level constants or `patch.object` stops working (and a metered-provider test can pass while a real call runs).
- **Telemetry/marks must never raise into the request path** and the latency record is bounded (200 turns, per-turn mark cap). Anything you add there follows the same contract.

## Short Architecture Summary

If you need the shortest accurate summary possible, use this:

- Electron provides the desktop shell and launches the backend/voice processes unless the watcher already did (`JARVIS_EXTERNAL_RUNTIME=1`). Model switches are live per message; nothing else hot-reloads.
- FastAPI exposes `/ask`, `/ask/stream`, `/ask/status/{id}`, `/ask/cancel/{request_id}`, `/voice-log`, `/voice-mode`, `/voice-state` + `/voice-state/publish`, `/ui-state`, `/speak/stop|pause|resume|remaining`, `/latency` + `/latency/client`, `/aec/state|reference`, `/screen-answer`, `/research-result`, `/research-progress`, `/settings*` + `/providers/{id}/models`, `/task/stop`, `/approvals/reset`, `/voice-setup/launch`, `/health`. Everything except `/health` needs the per-launch `X-Jarvis-Token` (fails closed; `JARVIS_DEV_MODE=1` is the only bypass).
- `backend/core/brain.py` is the central intent router. Memory phrases run first; then explicit task/code-tool handoffs, screen control, explicit research; then (if `JARVIS_ORCHESTRATOR_MODE=orchestrator`) the native tool-use orchestrator; then the legacy path: a speculative chat stream races the cloud classifier (OpenRouter Flash Lite -> Gemini Flash Lite -> Groq Qwen -> `chat` verdict, all hops sharing one deadline), and four deterministic backstops (screen-question net, fresh-info auto-search, all-search-steps -> research reroute, web-task routing) correct classifier misfires.
- Chat is served by the model-registry-selected provider (currently `gemini/gemini-3.5-flash-lite`) with a same-model non-stream retry and a Gemini -> Fireworks tail — NOT Groq, whose old default model is retired. A terminal auth/validation failure refuses instead of substituting another model.
- Web lookups default to the Brave AI-Overview quick-search tier; "deepsearch" adds the multi-site research service with a glass-overlay report. Both tiers share one long-lived Playwright worker.
- Screen Q&A ("What's on my screen?") runs the F37 eligible-provider vision cascade (registry selection first, bounded fallbacks, no credential-free dispatch) and shows floating desktop overlays.
- `command ...` messages trigger browser actions or launch local Windows applications (via the `launch_app` resolver in executor, no shell fallback) with web URL fallback.
- Multi-step web tasks hand off to the confirmation-gated browser agent (brave-control MCP daemon, 50-step/480s backstops); the opencode CLI is an opt-in engine selected by capability, and the executor choice is frozen in an F16 contract at consent time.
- Voice mode is a pure I/O worker: it captures and transcribes, submits to `/ask/stream` with `speak=False`, publishes its listening state, and owns playback through the Fish Audio -> local SAPI5 -> ElevenLabs -> local SAPI5 ladder. The backend remains the single intelligence authority.
- The watcher is a separate passive wake-word launcher running local GPU-accelerated Whisper via the resident `whisper_daemon` (port bound before model load), with the STT hallucination gate vetoing prompt echoes. It also supervises the full stack in `run_jarvis.bat` mode and distinguishes warm sleep from full shutdown.
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

### Note: event-driven page-settle latency overhaul (brave-control MCP server)

This change lives in the brave-control MCP server (`server.mjs`). It was originally made only in the external copy at `C:\Users\mayan\mcp-servers\brave-control`, but a copy is now vendored in-tree at `integrations/brave-control/` — check `BRAVE_MCP_SERVER_DIR` to see which one the app actually spawns. A `settlePage(page, opts)` helper replaced the slow `networkidle` waits in `navigate`/`new_tab` (now `waitForEvent('load', 3s)` with a catch, then settle 300ms) and added a 200ms settle after `click_element`. It resolves on a main-frame `framenavigated` event or DOM-mutation quiescence (200ms debounce, 2500ms hard cap), disconnects its MutationObserver cleanly, and is wrapped in try/catch so settling can never fail a tool call. `ask_chat` polls every 300ms instead of 1500ms, and `copy_code_block` switched to a 100ms clipboard poll capped at 2000ms. Measured navigate ~390-406ms (was 2-4s+) and click ~16ms; that repo's node tests pass 18/18.

## Recent Improvements (2026-09)

### Per-functionality custom providers from the Electron UI (custom-provider-ui, 2026-10-01)

Each model section of the sidebar (TEXT CHAT, VISION MODEL, BROWSER TOOL MODEL, PLANNER MODEL) now has its own **+ ADD CUSTOM PROVIDER** entry next to the providers already available for that functionality. The form asks for a name (optional — falls back to the endpoint host), the **base URL** and the **API key**, and below the fields offers **TEST** and **SAVE**: TEST live-checks the (base_url, api_key) pair with one model-list round trip and **stores nothing** (`POST /settings/provider/test`); SAVE persists the provider (still validated live before the write, key stored only in the gitignored `data/jarvis_settings.json`) and then **auto-fetches the models on that key** so one can be chosen for that functionality immediately from the dropdown (`USE MODEL` → `POST /settings/model`), or later from the usual provider → model list.

- Scope: custom providers now serve every LLM-backed role (`_ROLE_ALLOWS_CUSTOM = chat, vision, browser_tool, planner` — was chat + browser_tool); the voice roles (tts, listening) still refuse them because Fish/gTTS/Whisper/Inworld are dedicated audio engines, not base-url + key endpoints. `GET /settings` reports `custom_provider_roles` so the UI shows the entry exactly where it is accepted.
- Vision support for custom providers rides ONE generic adapter (`openai_compat_client.ask_openai_compat_vision`, standard `image_url` content part, optional `response_format` passthrough) registered per provider through `vision_cascade.custom_vision_dispatchers()` and merged into both call sites (`screen_control._vision_dispatchers`, `screen_analyzer._vision_dispatchers`). `vision_cascade.provider_available` now accepts a registered custom provider (key **and** base URL required) — the env-only gate would have silently skipped a selected custom provider.
- Planner and chat needed no dispatch change: `orchestrator._chat` and the chat chain already resolve any snapshot through `ask_openai_compat`; `browser_agent` already had a custom-provider branch.
- Test contracts adapted (deliberate, documented in the tests): `test_model_registry.py::test_custom_provider_selectable_for_any_role` and `test_live_bugfixes.py::test_vision_rejects_custom` (→ `test_vision_accepts_custom`) pinned the old chat/browser_tool-only allowlist. Suite: `backend/tests/test_custom_provider_ui.py` (15 tests). Key files: `frontend/renderer.js`, `frontend/index.html`, `frontend/style.css`, `backend/api/routes.py`, `backend/services/model_registry.py`, `backend/services/openai_compat_client.py`, `backend/services/vision_cascade.py`.

### Responsiveness audit wave II: P1-06, P1-07, P0-03, P0-10…P0-12, P1-11…P1-16, P1-08, P1-10 (2026-10-01, 14 commits, tagged per item)

The second implementation pass over the same 43-finding responsiveness audit, one item at a time in the order the owner fixed, `2e85ebb` → `3b460d0`. Every item carries an annotated tag with its audit id (`P1-06`, `P1-07`, `P0-03`, `P0-10`, `P0-11`, `P0-12`, `P1-11`, `P1-12`, `P1-13`, `P1-14`, `P1-15`, `P1-08`, `P1-10`, `P1-16`), so `git tag -l` is the index of which commit implemented which finding. Headline results, each measured before and after:

- **Turn start:** a chat turn no longer pays two durable writes before the first character (P0-11 — history JSON and the F07 `begin_request` commits are now a debounced background writer plus a FIFO bookkeeping queue with read-your-writes `_write_barrier()`); the speculative racer is built before the pre-route chain instead of after it and its release is guaranteed by the turn (P0-10 — the chain itself measured at **~0.034 ms per message**, so the hoist's real value is the guaranteed release plus the `predicate_ms` telemetry); provider TCP+TLS handshakes move onto the VAD head start (P0-12).
- **Voice:** "pause" is a real pause and "continue" resumes from the exact byte (P1-07 — was a `stop_speaking()` in disguise, so resume had nothing left); a transcript that arrived while the reply was speaking is an INTERRUPTION, never a silent drop (P1-06 — the speaking flag is now scoped to the reply session, so it stops flickering between sentences of the same reply); one STT engine per turn with no serial fallback ladder, worst case **117 s → the one engine's explicit timeout** with a recorded failure reason (P0-03).
- **Control plane:** `/task/stop` cancels precisely what was asked for and interrupts only the requests that job produced (P1-11); the voice-log publish is an in-process call instead of a 401-ing self-POST on every turn (P1-12); four concurrent control calls can no longer starve the loop, and **45 open SSE streams leave the thread pool at zero** (P1-14); voice state from a restarted worker is accepted and transitions publish immediately (P1-15); a paused job parked on an event burns **under 50 ms of CPU across a 0.4 s window** and resume/cancel wake it in under 150 ms (P1-10).
- **Reliability / cost:** the streaming client is bounded (time-to-first-token budget, idle budget once tokens flow, a cancelled read actually stops, in-stream errors surface as terminal states instead of empty turns — P1-13); the settings file is parsed once per change and a corrupt read falls back to the last good copy instead of silently resetting roles (P1-08); personal memory retrieval is ONE FTS query over content words with a relevance gate and a write-generation cache, its FTS rebuild is gated on `PRAGMA user_version`, its event table is pruned with its FTS rows, and memory writes now require explicit wording while "forget it" asks instead of deleting (P1-16).

Per item, with the source and the regression suite it added (every suite fails on the pre-fix tree):

- **P1-06 — never silently discard a committed transcript** (`2e85ebb`). `backend/voice_mode.py`, `backend/services/voice.py`. `test_p1_06_no_silent_drop.py` (17 assertions, 13 fail pre-fix).
- **P1-07 — pause/resume for voice replies** (`dcce09d`). `backend/services/voice.py` (`AudioActor.pause`/`resume_playback`/`_requeue_for_resume`, one `_device_lock` step for pop+paused-check+write), `backend/services/fish_voice.py`, `backend/voice_mode.py`. `test_p1_07_pause_resume.py` (27 tests, 24 fail pre-fix).
- **P0-03 — one STT engine per turn** (`62a624e`). `backend/voice_mode.py` (`_engine_for_listening_role`, `_transcribe_with_engine`, `LAST_STT_FAILURE`), `backend/services/transcription.py`. `test_p0_03_single_stt_engine.py` (25 tests, 17 fail pre-fix).
- **P0-10 — speculative racer before the pre-route chain** (`1ca1089`). `backend/core/brain.py` (`_register_turn_racer`, `_cancel_orphan_turn_racer`, `_ChatRacer.adopt`/`is_adopted`, `_mark_latency_duration`). `test_p0_10_preroute_speculation.py` (19 tests, all fail pre-fix).
- **P0-11 — synchronous writes off the path to the first delta** (`5d9f8b4`). `backend/core/memory.py` (debounced history writer, `os.replace` retry, generations, `flush_history()` on `atexit` + shutdown), `backend/core/memory_store.py` (FIFO writer thread, in-memory event ids, `_write_barrier()`, `read_without_barrier()`, one transaction for row+FTS, writer-thread `assert_writer`). `test_p0_11_deferred_writes.py` (15 tests, 12 fail pre-fix); `test_f50_setup_ownership.py` updated to the blocked-not-raised refusal contract.
- **P0-12 — pre-warm the provider connections on the VAD head start** (`36497a0`). New `backend/services/prewarm.py` (`pooled_session`, `KeepAliveAdapter`, rate-limited non-blocking `warm`, fully-read warm responses, `warm_async`), `POST /prewarm` + `GET /prewarm/stats` in `backend/api/routes.py`, the onset hook in `backend/voice_mode.py`, pooled STT with a 1 s connect budget in `backend/services/transcription.py`, keepalive on the Gemini/OpenAI-compatible sessions. `test_p0_12_prewarm.py` (27 tests, 13 fail pre-fix). Deliberate deviation: chat/classifier CONNECT timeouts stay 8 s / 5.05 s — capping them would make a cold VPN handshake a *failed* call.
- **P1-11 — `/task/stop` cancels precisely what was asked for** (`5cced12`). `backend/api/routes.py`, `backend/services/jobs.py`, `backend/services/request_registry.py` (`interrupt_job`), `voice_mode.py`. Response reports `cancelled` + `interrupted`; a bare stop no longer sweeps `request` jobs; `interrupt_active` is kept for the legacy path. `test_p1_11_task_stop_scope.py` (22 tests, 13 fail pre-fix).
- **P1-12 — remove the unauthenticated self-HTTP call from the turn path** (`561c11b`). `backend/core/brain.py` (`sync_voice_log` in-process, `register_voice_log_sink`), `backend/api/routes.py` (registers `_publish_voice_log`, worker publishes once after `state.complete`). `test_p1_12_voice_log_inprocess.py` (16 tests, 10 fail pre-fix).
- **P1-13 — bound the streaming client** (`36c41df`). `backend/services/openai_compat_client.py` (`DEFAULT_FIRST_TOKEN_TIMEOUT`, `DEFAULT_STREAM_IDLE_TIMEOUT`, cancellable reads, terminal `new_stream_outcome`, pinned utf-8). `test_p1_13_stream_bounds.py` (32 tests, 15 fail pre-fix).
- **P1-14 — SSE without a thread per stream** (`5ac5ef9`). `backend/services/request_registry.py` (`astream()`, batched frames, `STREAM_FLUSH_SECONDS = 0.03`, atomic `delta`/`replace`), `backend/api/routes.py` (async generator body, `CONTROL_LIMITER` of 4 for the control plane), `main.js` + `backend/watcher.py` (`--no-access-log`). `test_p1_14_sse_capacity.py` (23 tests, 18 fail pre-fix). Deviation: `_on_control_plane` uses `limiter=` rather than `async with`.
- **P1-15 — voice-state freshness across a worker restart** (`5393f79`). `backend/api/routes.py` (`_voice_state_publisher`, per-publisher high-water mark, additive `publisher_id`), `backend/voice_mode.py` (`_VOICE_LAUNCH_ID`, event-driven publish loop with a 1 s heartbeat), `backend/services/listener_state.py` (state hooks). `test_p1_15_voice_state_freshness.py` (17 tests, 12 fail pre-fix).
- **P1-08 — parse the settings file once per change** (`8ef2547`). `backend/services/model_registry.py` (stat-keyed cache, last-good-copy fallback, deepcopy to callers, `_save_unlocked` adopts its write, `_forget_cached_settings_unlocked`), `backend/services/voice.py` (TTS provider resolved once per reply session). `test_p1_08_settings_cache.py` (15 tests, 14 fail pre-fix). Two test-contract adaptations: `test_model_roles_wiring`'s corrupt-registry assertion now expects the last good copy, and `_speak_chunk` fakes accept `**_kwargs`.
- **P1-10 — park a paused job on an event** (`bb397fe`). `backend/services/jobs.py` (`JobToken._gate`, `wait_if_paused` on the gate, `killing`/`state`, `terminate_processes` off the registry lock, `terminate_processes_now`, `wait_for_kill`), `cancel_job` releases `_lock` before cancelling. `test_p1_10_job_runtime.py` (15 tests, 10 fail pre-fix).
- **P1-16 — memory retrieval cost, retention and phrase safety** (`3b460d0`). `backend/core/memory_store.py`: `relevant_facts` is ONE FTS query over `_retrieval_terms` plus ONE batched row/revision load (query count no longer grows with terms or stored facts), `_relevance` gates on whole-word/5+-char stem hits, results are cached on a write generation bumped by every `commit()` (via `_MemoryConnection`) and every queued write; the whole-index FTS rebuild is gated on `PRAGMA user_version` (`MEMORY_SCHEMA_VERSION`) and `prune_events()` adds events retention with its FTS rows; the scheduler sleeps until the next due commitment (`_next_commitment_delay`, `SCHEDULER_MAX_WAIT`, `SCHEDULER_SETTLE_SECONDS`) instead of polling every 5 s; corrections/aliases require explicit memory wording, and forget phrases are punctuation-normalised and resolve by subject/key only (never by value) or ask. `test_p1_16_memory_cost.py` (26 tests, 19 fail pre-fix). Two test-contract adaptations, both because the assertion pinned the audited bug: `test_g9_memory.test_negation_idioms_are_not_memory_ops` ("forget it" now asks which memory, deleting nothing; "never mind" stays `None`) and `test_f06_personal_memory.test_alias_known_as_phrase` (the bare "X is also known as Y" sentence writes nothing; the explicit "remember that …" form does).

Verification: one regression suite per item, each run against a stashed pre-fix tree first (fail counts above), then the full backend suite after each commit. Final: **2802 passed, 5 failed, 1 skipped, 56 subtests passed** (`& backend\venv\Scripts\python.exe -m pytest backend\tests -q`, ~2.5 min), with the same 5 pre-existing failures throughout: `test_f02_goal_routing.py::OutcomeContractTests::test_a_proposal_status_is_proposal_not_needs_input`, `test_f37_groq_prerequisite.py::VerificationUsesTheSharedCascadeTests` (×2) and `test_live_bugfixes.py` vision dispatch (×2). Deliberate deviations to remember: P1-11 retains `interrupt_active`, P0-11 moves `_mark_latency_duration` onto the turn, P0-12 keeps the 8 s / 5.05 s chat/classifier connect timeouts, P1-14 uses `limiter=` only, P1-15 adds publisher identity additively, P1-08 and P1-16 adapt two test contracts each (listed above).

### Responsiveness audit wave: P1-19, P0-04…P0-08, P0-13, P1-01…P1-05, P1-09 (2026-09-30, 13 commits)

The first implementation pass over the 43-finding responsiveness audit, one item at a time, `f37a519` → `6c89095`. Headline measured results, each taken before and after on the same probe:

- **Barge-in:** `barge_in_on_speech_onset` **10,016 ms → 0 ms** against a hung endpoint (P1-03), `stop()` drain **221 ms → ~9 ms** with `abort()`, abort-to-silence 1.4–8.9 ms (P0-07).
- **Interrupting a reply:** submission of the interrupting utterance **3.015 s → 0.005 s**, and the old turn is actually cancelled now (P0-08).
- **Capture loop:** **2.45 s → 0.05 s** per capture with partials moved off-thread (P0-04).
- **Audio output:** feed→device-write **125–203 ms → <0.1 ms** (P0-06); inter-sentence gap p50 **400 ms → −0.10 ms** and **9 device opens → 1** with one persistent low-latency stream (P0-07); time-to-first-audio with a prefetch in flight **281 ms → 47 ms** (P0-05).
- **Correctness:** two consecutive captures no longer return identical audio (P0-13, this was the self-barge-in cause); the **4.000 s-for-2.000 s** sample-rate mislabel is gone (P1-05); a non-Fish TTS engine no longer bills a Fish synthesis (P1-09, **1 → 0** metered calls); the first spoken chunk no longer waits for a following token and no chunk ever ends mid-word (P1-01).
- **Observability:** `/latency` now renders a real per-turn waterfall stitched across both processes (P1-19) — absolute-stamp marks only, p50/p90/max per step, sorted slowest first, with the legacy keys preserved.

Verification: 13 new `backend/tests/test_p*_*.py` suites (all of them fail before their fix), focused runs of every touched suite, and a full-suite run per item (**2507 passed, 4 failed, 1 skipped** at the end, with the 4 pre-existing and order-dependent). Existing suites that had to be updated deliberately — each because its assertion pinned the old synchronous timing or the exact bug being fixed, and each commented in place with the reason: `test_f33_echo_cancel`, `test_latency_reductions`, `test_p1_03_async_speak_stop`, `test_websearch_interruption`, `test_live_bugfixes`, `test_f50_f51_voice_ownership`, `test_f32_playback_owner`, `test_voice_latency`, `test_p1_19_latency_waterfall`, `test_voice_task_mute`.

Chronological work, newest last. The "Suite:" figures below are the counts recorded at the time of each change (they stopped being a single 17-module number after 2026-09-11 — see "Running the test suite").

### Short browser task summaries and instant TTS barge-in stop (901fd6e)

Browser-agent task completions now produce short spoken summaries instead of raw transcripts, and TTS barge-in stops instantly: the listener's stop path bumps the speech generation and flushes Fish PCM playback mid-stream so a user interruption cuts the voice without lag. Key files: `backend/core/brain.py`, `backend/api/routes.py`, `backend/services/listener.py`, `backend/services/fish_voice.py`, `frontend/renderer.js`, `frontend/capsule_renderer.js`, new `backend/tests/test_live_bugfixes.py` coverage. Suite: 440.

### Tiered web search with Brave AI Overview and interruptible research (69f9b07)

Introduces the two-tier lookup contract: default mode reads the Brave Search AI Overview answer box via headed Chrome on the shared research profile (no site scraping, short spoken+text summary, snippet fallback), while explicit "deepsearch" pins the overview into the existing multi-site `run_research` flow. Research runs in a background thread with an immediate ack, is stoppable mid-run ("stop the research"), and reports push to the UI. Key files: new `backend/services/quick_search.py`, `backend/core/brain.py` (`handle_research_intent`), `backend/services/research_service.py`, `backend/api/routes.py`, `backend/services/listener.py`, `backend/voice_mode.py`, new `backend/tests/test_websearch_interruption.py` and `backend/tests/test_websearch_modes.py`. Suite: 485.

### Plain-spoken AI overview answer, print to chat UI, TTS markdown strip and fallback cleanup (4608a94)

The quick-search answer is rewritten into a plain-spoken spoken summary, the lookup result prints into the chat UI, TTS strips markdown before speaking, and dead fallback branches are cleaned up. Key files: `backend/services/quick_search.py`, `backend/core/brain.py`, `backend/tests/test_websearch_modes.py`, `backend/tests/test_browser_agent.py`. Suite: 485.

### Websearch permission removal, ask tab answer fallback, fast-fail classify and key redaction (37ae46b)

Removes the browser permission prompt from the websearch flow, adds the Ask-tab answer fallback with container-first extraction (`div.message.assistant.llm-output`) plus a guarded `rfind(query)` fallback for when Brave's AI answer renders in the Ask view, makes `classify_intent` fast-fail (3s timeout, `no_retry`, no urllib3 retry multiplication) with a Groq Qwen fallback, and redacts API keys (`key=<redacted>`) from all logged URLs/headers in the Gemini client. Key files: `backend/services/quick_search.py`, `backend/services/intent.py`, `backend/services/gemini_client.py`, `backend/services/grok_client.py`, `backend/core/brain.py`, `backend/tests/test_websearch_modes.py`. Suite: 496.

### Fireworks planner swap, confirmation-gated fallbacks, web-task routing, fresh-info auto-search (2c55775)

The task-agent planner moves from retired Groq `llama-3.3-70b-versatile` (404) to `ask_fireworks` (`deepseek-v4-flash-0731`, temperature 0.1), and BOTH no-plan fallbacks become confirmation-gated (`requires_confirmation=True`). New deterministic nets in `process_message`: web-shaped task requests (`backend/services/web_task_routing.py`) route to the confirmation-gated browser-agent handoff instead of the raw task path, and chat verdicts with fresh-info keywords (pricing/cost/latest…) that are question-shaped but not greetings auto-route to the quick-search tier. Intent prompt gains pricing/research and multi-step-web-task examples. Key files: `backend/services/task_agent/agent.py`, `backend/core/brain.py`, new `backend/services/web_task_routing.py`, `backend/services/intent.py`, `backend/tests/test_brain_gate.py`, `backend/tests/test_task_agent.py`, `backend/tests/test_chat_race.py`. Suite: 504.

### Deterministic screen-question net over classifier chat/research misreads (1fdff1c)

"what's on my screen jarvis" was answered "I cannot see your screen" because the cloud classifier genuinely misclassifies screen questions as chat (and sometimes research), not only on provider outages. A deterministic net now fires right after `classify_intent`: on a `chat` or `research` verdict, `is_screen_question` rewrites the intent to `screen`/`region` before the racer holdback and fresh-info nets, so the existing screen branch handles analysis. tool/task verdicts are exempt (their structured steps would be discarded). Intent prompt gains the apostrophe+wake-word example. Key files: `backend/core/brain.py`, `backend/services/intent.py`, `backend/tests/test_brain_gate.py` (8 ScreenQuestionNetTests). Suite: 512.

### Fable-5 remediation wave: F01-F52 implemented (2026-09-12)

One long session turned the audit's correction list into named feature suites, taking `backend/tests/` from 17 modules into the dozens. Landed in group order: **G1** closed-loop task results and recovery limits (`task_result.py`; `test_f01`/`f02`/`f03`/`f05`); **G2** the policy and consent backbone (`approvals.py`, `tool_policy.py`, `code_grants.py`, `jobs.py`; `test_f17`/`f18`/`f20`/`f21`/`f22`); **G3** request identity, event buffers, reconnect and pure cancellable speculation (`request_registry.py`; `test_g3_request_streaming`, `test_f23_reconnect`); **G4** the coding interface (`code_tools.py` plus the structured editor bridge; `test_code_tools`, `test_f15_editor_interface`); **G5** the research pipeline with question-carrying evidence and provenance (`provenance.py`; `test_g5_research_pipeline`, `test_f48_provenance`); **G6** browser grounding (`browser_session_broker.py`; `test_g6_browser_grounding`, `test_f38`/`f39`/`f40`); **G7** Windows screen grounding (`screen_geometry.py`, `screen_ui_elements.py`, `screen_ocr.py`, `screen_capture.py`; the `test_f37`-`f45` suites); **G8** goal routing, capability dispatch, one deadline and capability-aware model selection (`orchestrator.py`, `capability_resolver.py`, `capability_contract.py`, `deadline.py`; `test_g8_orchestrator`, `test_f16`/`f24`/`f49`); **G9** the single persistent memory store (`backend/core/memory_store.py`; `test_g9_memory`, `test_f06`-`f10`); **G10** the voice runtime (`audio_actor.py`, `echo_cancel.py`, `transcript_stabilizer.py`, `wake_engine.py`, `whisper_daemon.py`; `test_g10_voice_runtime`, `test_f31`-`f36`); **G11** process architecture and security (`local_auth.py`, `intelligence_state.py`, `runtime_identity.py`; `test_g11_process_security`, `test_f50`-`f52`). The classifier retirement (G8 step 5) deliberately did NOT land: `JARVIS_ORCHESTRATOR_MODE` still defaults to `legacy`.

### Voice/runtime hardening wave: F53-F55 (2026-09-13/14)

- `backend/services/google_tts.py` added: the key-less Google Translate TTS engine, selectable as the `tts` role engine (`gtts` provider) and deliberately sharing the ONE `audio_actor` playback owner instead of becoming a second playback authority (F32). Not the `gTTS` package — its `click<8.2` pin breaks `typer`/`uvicorn` in this venv; text is split at ~200 characters and time-stretched 1.5x with ffmpeg `atempo`. `test_gtts_fallback.py`.
- `backend/services/research_browser.py` (F27/F28) became the single long-lived research browser worker; `test_f53_research_browser_channel.py` pins the out-of-band channel.
- F54/F55: boot and task-engine ownership (`test_f54_boot_and_task_engine.py`) and the whisper daemon's port-before-model boot order with an honest `/health` (`test_f55_whisper_boot_independence.py`).
- Root `conftest.py` + `test_env_file_guard.py`: a test may never modify or delete the real `.env`.
- The overlay renderers were reworked alongside `tests/overlay-ipc-contract.test.js`.

### Chat outage root-caused: Proton VPN + suspended Fireworks; OpenRouter added (2026-09-23)

Every query answered "I'm having trouble connecting. Please try again." Root-cause chain: Proton VPN's ProTUN tunnel degraded direct `generativelanguage.googleapis.com` calls to 7-45s+ (frequent read timeouts), the chat chain's other leg (Fireworks) was suspended, and Groq 403'd from the VPN exit IP. OpenRouter (Cloudflare-fronted) answered the same model in ~1.4s through that same tunnel. Fixes: the `chat` role allowlist gained `openrouter`; `get_provider_credentials` now returns canonical OpenAI-compatible base URLs for openrouter/groq; `intent.py` was reordered to Gemini -> OpenRouter -> Groq, with `ask_openai_compat` gaining an optional timeout so the classifier's tight budget slice is respected. Verified: non-stream chat 1.41s, stream 0.61-0.65s, a full `process_message` turn 1.80s. Key files: `backend/services/model_registry.py`, `backend/services/intent.py`, `backend/services/openai_compat_client.py`, `data/jarvis_settings.json`. Operator note: turning the VPN off (or split-tunnelling `python.exe`/`electron.exe`) restores the direct Gemini path.

### CODE_REVIEW_REPORT C1-C3 implemented (2026-09-23)

- **C1 — prompt injection.** Screen-derived text (OCR, UIA names, window title, interaction history) is untrusted data: spoof-proof `<<<SCREEN_TEXT_UNTRUSTED>>>` delimiters (marker spoofing neutralised first), role-prefix and control-phrase stripping via `_sanitize_screen_fragment`, and a static never-instructions header that pins step targets to element ids and type payloads to the user command. `_build_tree_prompt` is the single choke point. `test_c1_prompt_injection.py` (10 cases). Full screen suite: 149 passed.
- **C2 — one declared coordinate frame.** Prompt geometry (UI tree + OCR nodes) is serialised in NORMALIZED 0..1000 at serialisation time (`_norm_geom`), the same frame the planner answers in, so the 660,550 -> 66% exploit is gone; internal element-id maps keep pixel truth for the executor. A step that cites an element AND returns coordinates must have them agree (bounds grown 25%, minimum 40px) or the plan is rejected. `test_c2_coordinate_frames.py` (12 cases). Suites: screen 141 passed, C1+C2 18 passed.
- **C3 — model output can no longer become raw OS input.** Keys are whitelisted against the key vocabulary (a combined `alt+f4` as ONE token is rejected), click buttons are whitelisted `{left,right}`, click counts are coerced and clamped 1..10, `_normalize_key_name` rejects any token outside a safe grammar (no plus sign — chords are built from separate whitelisted tokens), and the two `execute_steps` handlers widened from `except RuntimeError` to `except Exception` so a malformed step cannot abort mid-sequence after earlier steps already fired. `test_c3_input_synthesis.py` (12 cases). Suites: C1+C2+C3 + screen_control 171 passed.

### CODE_REVIEW_REPORT H1-H9 implemented (2026-09-23)

- **H1**: `requirements.txt` regenerated by pip freeze — numpy 1.26.4 ABI pin (2.2.6 broke the ctranslate2/av wheels), pywinauto 0.6.9, plus ~14 previously missing runtime packages; 3 pin-guard tests.
- **H2**: `executor.launch_app` deleted its `isalnum` `shell=True` tail; Edge launches via protocol, VS Code via an argv list, against a fixed `system_apps` allowlist; 3 tests.
- **H3/H7**: `screen_ui_elements` — a `set_focus` failure now ABORTS the type action instead of typing into whatever holds focus; legacy bare-wrapper cache entries are treated as stale and re-resolved; cache TTL 8s -> 3s; per-thread COM apartment init at every UIA entry fixes the `RPC_E_CHANGED_MODE` silent degradation to blind coordinate clicks.
- **H8**: the retired Groq model `llama-3.3-70b-versatile` was dropped from `grok_client`'s default and from the offered catalog.
- **H9**: research-overlay markdown links allow only `http(s)`/`mailto`; `javascript:`/`data:`/`vbscript:` render as plain label text; 3 VM tests, `node --test` green.
- **H4/H5/H6**: `capture_epoch` is probed before EVERY effect and refuses on mismatch; `_verify_window_identity` (IsWindow + process-id match) runs before every effect and at focus acquisition; a hit-test that finds no window under the point now REFUSES (it used to pass silently); out-of-frame normalised points are REJECTED rather than clamped into edge clicks. The report's DPI/monitor double-scaling claim did NOT survive verification — captured pixels are physical end-to-end and window origins already use DWM extended frame bounds, so applying `dpi_scale` again would have double-scaled. `test_h4_h5_h6_staleness.py` (12 cases); Screen+C+F24/F49 surface: 199 passed.

### STT hallucinations were being answered as user speech (2026-09-24)

With the headset AEC degraded, 26 frames of Jarvis's own TTS were captured into the microphone and local Whisper transcribed the noise into memorised filler that the brain then answered as commands ("jarvis, a ver si te acuerdas de esto", "jervis, wake up, jervis, utho, jago, chalu", and "chalu, chalu, ..." loops — the wake-bias prompt echoed verbatim). Three fixes: (1) `whisper_daemon.py` had applied the wake-bias `INITIAL_PROMPT` to EVERY transcription — it is now opt-in via `X-Jarvis-Purpose: wake`, sent only by `watcher._transcribe_with_daemon`, so conversation transcription runs unbiased; (2) a deterministic `is_hallucinated_transcript()` gate in `backend/services/transcription.py` rejects prompt-vocabulary echoes (5+ tokens from the wake vocabulary), looped tokens/phrases, memorised silence phrases and filler-only utterances, while never rejecting real commands, short wake phrases, literal payloads or repeated safety words; (3) the gate is wired at every commit boundary — listener partial windows, each STT engine's accept inside `recognize_multilingual` (a hallucination falls through to the next engine), the final `listen()` commit belt, the watcher's `is_wake_word` (a prompt echo could otherwise false-LAUNCH the stack) and `_add_candidates`. `test_stt_hallucination_gate.py` (26 cases); adjacent suites 240 green.
