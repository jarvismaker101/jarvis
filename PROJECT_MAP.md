# Jarvis Assistant Project Map

This document gives another model enough context to work on the repo without re-discovering the architecture from scratch. **Refreshed 2026-10-03.**

State as of this refresh:

- The **Fable-5 audit remediation** (G0–G11, F01–F55) is landed except the **G8 classifier-retirement step**, which remains open.
- The **CODE_REVIEW_REPORT hardening wave** is landed for **C1–C3** and **H1–H9**, plus the **STT-hallucination gate**. The review's M/L/structural items are still open.
- A **responsiveness audit** was completed 2026-09-29 and triaged by the owner. **27 of its items are now implemented** — 13 in the 2026-09-30 wave (`f37a519`…`6c89095`) and 14 in the 2026-10-01 wave (`2e85ebb`…`3b460d0`). Per-item detail is in "Responsiveness audit implementation state" below. The original triage prompts file was never committed to this repo, so the per-item prompts are **not recoverable**; git commit messages and the per-item test suites are the record.
- **The S-series latency/intelligence wave (2026-10-02/03)** is the newest work and is the single biggest architectural change since F56: neural-VAD endpointing (S29), AEC alignment (S28), one Fish WebSocket session per reply (S7), one STT engine per utterance (S5), a single-call router that also answers chat (S6), conversation during a running task (S18), background results in history (S13), state pushed over one event channel (S19), and a voice-path latency batch (S1/S12/S23/S26/S27). See "Recent Improvements (2026-10)" below.
- **F56 (landed, tag `local-models`)** — the LOCAL Ollama server is a first-class model provider, and the intent classifier is a selectable role. `chat` and the new `intent` role accept `ollama/<installed model>` from the Electron model switcher; `intent` gets a whole new sidebar section. See "Local models as selectable brains (F56)" below — including the MEASURED latency numbers, which are the reason this is a selection and not a new default.

## Local models as selectable brains (F56)

The user asked for the locally-installed Ollama models to be selectable like any
other model — one for replies and one for routing — and that is exactly what
landed. Nothing about the shipped defaults changed: with no selection made, chat
still answers with `gemini/gemini-3.5-flash-lite` and the router still starts on
OpenRouter Lite.

- **`ollama` is an ENV provider, not a custom provider** (`model_registry.ENV_PROVIDERS`): key-less, endpoint from `ollama_client.OLLAMA_BASE_URL` + `/v1` (ONE source of truth for the local port), model list read live from `GET /api/tags`. Its credential is the constant placeholder `ollama-local`, because the generic OpenAI-compatible adapter and brain's "no credentials → fail closed" rule both require a truthy key; Ollama ignores the header, and no API surface ever returns it.
- **Roles**: `chat` and the NEW `intent` role accept `ollama`. `vision`, `tts`, `listening`, `planner` and `browser_tool` do NOT — the provider floor declares `tool_calling`/`structured_output`/`streaming` and deliberately NO `vision_input`, so a local text model can never be selected to answer a screen question.
- **The model list is filtered, not dumped**: whatever the daemon reports minus models that cannot answer a chat/intent call (an installed embedder such as `nomic-embed-text` must not appear as a selectable brain). A stopped daemon is a clean 400 in the picker, not an empty list.
- **Thinking is OFF for a local endpoint, inside the client** (`openai_compat_client.LOCAL_REASONING_EFFORT`). Measured against Ollama 0.34.4, the OpenAI-compatible surface IGNORES the native `think` field AND a `/no_think` marker, but honours `reasoning_effort`: `none` disables thinking, any enabling value enables it, and a non-thinking model (llama3.2) ACCEPTS `none` while rejecting every enabling value with a 400. With thinking on, qwen3 1.7B took 1.41s to its first ANSWER token and 2.2s per classification; with it off, 0.06s / 0.4s. A 4xx that names the reasoning setting is replayed once without it (the `fireworks_client` pattern).
- **A budgeted hop gets exactly ONE attempt** (`ask_openai_compat(single_attempt=True)`, `_single_attempt_session`): the classifier splits one shared deadline across hops, and urllib3's `Retry(total=N)` also retries READ timeouts, so a 3.5s hop could cost ~3×3.5s plus backoff (measured: a 3.5s budget produced a 5.6s classification). The classifier's next hop IS its retry.
- **The intent role is hop 1; the shipped chain is the fallback** (`intent._SHIPPED_CHAIN`): selected `(provider, model)` first, then OpenRouter Lite → Gemini Flash Lite → Qwen on Groq, with the selected provider skipped in the chain (never pay the same endpoint twice inside one budget). The registry's env default for `intent` resolves to the SAME first hop as before (`intent.DEFAULT_OPENROUTER_MODEL`, read by `model_registry._env_default_for_role`), so an untouched install routes identically. A selection that stops validating is dropped by the registry and the chain runs.
- **UI**: a new `INTENT CLASSIFIER MODEL` section in the model switcher (`intent-provider-list`, `activeIntentModel`), rendered by the same provider→model list as every other role; `GET /settings` gained `intent_model` + `role_allowed["intent"]`; `POST /settings/model` accepts `role: "intent"` and answers `intent_model`. The Ollama provider shows as `local · no key needed` (`KEYLESS_PROVIDERS_BY_ROLE` / `LOCAL_PROVIDERS`).
- **MEASURED, on this machine (i5-13420H / RTX 3050 6GB), full 1866-char production classifier prompt**: qwen3:1.7b thinking-off **0.4–0.7s warm (prompt-cache hit)**, 2.7–3.3s on the first call of a burst (prefill) and up to 5.6s on a cold model load; llama3.2 same shape; OpenRouter `google/gemini-2.5-flash-lite` **1.5s** (1.48–1.78). So steady-state local routing is FASTER than the cloud, but the first call after idle is slower — and the voice classifier budget (`INTENT_BUDGET_VOICE_MS = 1200`) cannot fit a cold local prefill at all, in which case the turn pays that budget and still falls back to the cloud. Keep that in mind before selecting a local model for `intent` while using voice.
- **Accuracy, same 8 labelled queries through the real `classify_intent`** (production prompt, temp settings file, real daemon): qwen3:1.7b reached the cloud baseline's verdicts including the Hinglish `deepseek kya hai, dhundho → research` and the multi-step `1hd.to … play it → task`, and missed the same two hard cases the cloud misses (`capital of france → research`, `google python decorators → research`). llama3.2's early 2/8 was timeout contamination from cold loads, not accuracy — once resident it matched on the same cases.
- Tests: `backend/tests/test_f56_local_models.py` (34 cases) — provider/credential/allowlist/capability contract, model-list filtering, thinking-off on both wire paths plus the one-shot replay, single-attempt budgeting, hop-1 selection + fallback + no-duplicate-provider, and both settings routes.

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

Typed UI requests and spoken requests both end up in `backend.core.brain.process_message()` (defined at `brain.py:3672`, delegating to `_process_message_inner` at `brain.py:3723`). Deterministic phrase routes (memory, explicit stops, pending confirmations) run first. Then — since P0-10 — **route selection and a speculative chat racer run FIRST, above the task/code-tool/screen-control/research chain** (`orchestrator_select_route` at `brain.py:3849`, then `_ChatRacer` at `brain.py:3857` for any non-`command` message with a stream). Only then do explicit task (`3864`), native code-tool (`3878`), screen-control (`3891`) and research (`3901`) gates run, plus the optional G8 orchestrator. Otherwise that function applies deterministic safety nets over the classifier verdict, then routes to chat, tool actions, screen Q&A, research, browser-agent tasks, or memory commands, and returns a short English reply that may also be spoken aloud.

**Two changes since 2026-09-30 make this materially different from older descriptions of the flow:**

1. **The router answers the turn (S6, `1fd635a`).** A deterministic fast path (`JARVIS_CHAT_FASTPATH`, default on) skips the network entirely for plain chat via `is_definitely_plain_chat` (`brain.py:942`, verdict synthesised at `3978-3991`). Otherwise the intent router's JSON carries a `reply` field (`intent.py:60-61`) which the brain speaks via `handle_chat(..., answered=router_reply)` (`brain.py:4209-4236`) — **no second chat completion**. A `chat` verdict no longer implies "fall through to the chat model".
2. **State is pushed, not polled (S19, `a0ec729`).** The voice worker subscribes to the `GET /events` SSE channel (`backend/services/event_bus.py`) instead of polling `/ui-state` and `/voice-mode` on a 1s TTL. The 1s polls remain only as a fallback if the channel drops.


## Fable-5 Audit Remediation State

The Fable-5 audit's remediation plan groups work into G0–G11. State as of this refresh:

- **G0 (landed)** — quick wins: streaming/reasoning fixes, payload-safe lowercasing, research notes carry the question, vision-when-tree-empty (F45), env-sourced browser token.
- **G1 (landed)** — closed-loop execution core: `backend/services/task_result.py` (`TaskResult` with `completed|partial|failed|stopped|needs_input`, evidence, legacy-string compatibility), recovery limits, closed-loop plans in `task_agent/agent.py`.
- **G2 (landed)** — policy & consent backbone: `backend/services/approvals.py` (F18 plan-hash approvals), `backend/services/tool_policy.py` (F17 dispatch-bound tool validation + F21 secret masking), `backend/services/code_grants.py` (F22 scoped grants + change journal), `backend/services/jobs.py` (F20 per-job cancellation tokens).
- **G3 (landed)** — request identity & streaming: `backend/services/request_registry.py` + `/ask` event frames (`delta|replace|progress|completed|interrupted|error`), `/ask/status/{request_id}`, reconnect-resume without re-execution, pure cancellable speculation (`_ChatRacer`), publish-early-enrich-later screen answers.
- **G4 (landed)** — coding interface: `code.search/read_range/inspect_diff/apply_patch/run_checks` in `code_tools.py`; structured editor bridge renderers (F15) in `task_agent/agent.py`.
- **G5 (landed)** — research pipeline: question-carrying evidence schema, `backend/services/provenance.py` (observed vs checked provenance, F48), bounded parallelism (`test_g5_research_pipeline.py`).
- **G6 (landed)** — browser grounding: `backend/services/browser_session_broker.py` (profile/tab/origin identity), marks bound to real targets with epochs, Playwright input primitives, extended toolset.
- **G7 (landed)** — Windows screen grounding: `backend/services/screen_geometry.py` (coordinate-space helpers, monitor bookkeeping, F44 same-frame rect comparisons), `backend/services/screen_ui_elements.py` (F42 runtime-id re-resolution, F44 preserved hierarchy), `backend/services/screen_capture.py` (F43 multi-monitor/DPI/last non-Jarvis target), F29 UIA-first fast path in `screen_control.py`, F45 image-only vision fallback.
- **G8 (landed)** — orchestrator migration behind the migration flag: `backend/services/orchestrator.py` (F02 native tool-use loop over the registry-resolved qwen3p7-plus planner; tool args validated; invalid output distinguished from a conversational answer; mutations only PROPOSED through the existing gates), `backend/services/capability_resolver.py` (F16 engine dispatch; availability ≠ permission; opencode opt-in only; browser recovery needs no CLI), `backend/services/context_envelope.py` (F46 bounded request-scoped envelope with explicit omissions), planner role + capability validation in `model_registry.py` (F49; wired into `task_agent._model_plan`). Enable with `JARVIS_ORCHESTRATOR_MODE=orchestrator`; the legacy classifier routing and all deterministic safety nets stay until parity (the screen-question net replays deterministically inside the orchestrator).
- **G9 (landed)** — memory & continuity: `backend/core/memory_store.py`, one SQLite store (FTS5 with graceful LIKE fallback) with `facts`/`entities` (F06: provenance, confidence, sensitivity, superseded-by revision chain, tombstoned scoped forgetting, bounded `memory_context` injection in `_build_chat_messages`), `events` (F07: every chat exchange, async reply and task result recorded bounded + secret-masked via `tool_policy.mask_secrets`, grounding "use the report you just researched" follow-ups), `commitments` (F10: explicitly-armed deadline/job-completed triggers with a state machine and scheduler daemon; delivery rides the same async-reply UI/speech path as background results — notification only, never computer control; calendar/file-watch triggers deliberately rejected as assumption-flagged), `skills` (F09: verified completions capture approval-gated candidates; only user-approved skills ground `_model_plan`; versioned + invalidated after repeated failures; webpage instructions are never promoted). Deterministic phrase ops (`remember that…`, `forget…`, `what do you remember about…`, `remind me to … in N minutes/at HH:MM`, `cancel the reminder`, `approve/retire the … skill`) run before any other routing and are gated by `JARVIS_MEMORY_ENABLED` (default on; empty store changes zero prompts).
- **G10 (landed)** — voice runtime: `backend/services/audio_actor.py` (F32 single playback owner with a generation token — chunks carry the generation they were produced for and stale-generation PCM is dropped at every await/play boundary; the PCM ring + spoken cursor make interrupted resumes exact; `abort()` stops **this** stream, never `sd.stop()` for everyone; audio cache is identity-keyed by model+reference+format+text), `backend/services/echo_cancel.py` (F33 AEC3 reference path: every chunk that reaches the device feeds a monotonic reference ring resampled to 16k, the listener's VAD/onset window is AEC-cancelled before it can look like a barge-in — `py-webrtc-aec3` when installed, otherwise a loud no-op at the [ASSUMPTION] boundary; **S28 `80c094e` extended this substantially with `DelayEstimator` cross-correlation lag lock, a joint last-5-frame compare window, off-thread reference prefetch, and a `lag` block in `state()` exposed by `GET /aec/state`**), `backend/services/transcript_stabilizer.py` (F34 local-agreement conversation windows: finals commit immediately, agreeing overlapping partials after N windows, and `listen()` only ever returns *committed* transcripts so unstable partials can never trigger an action). **Utterance endpointing is now neural (S29 `acc366e`): `backend/services/neural_vad.py` runs Silero-on-ONNX with 100 ms-on / 200 ms-off hysteresis and falls back to `webrtcvad` only when the model is unavailable (`JARVIS_NEURAL_ENDPOINT=0` is the kill switch). The old partial-window layer was deleted outright in `7afefc0`, and S5 `c7b2128` collapses STT to exactly ONE engine per utterance whose output IS the transcript — there is no engine ladder and no fall-through.** F36 wake/command separation in `backend/services/wake_engine.py` + `watcher.py`: the phrase tail after the wake window is now extracted and forwarded to `/ask` ("wake up jarvis and search for cats" acts on both parts), an optional pre-roll ring (`JARVIS_WAKE_PRE_ROLL_SECONDS`) plus online Whisper verification of false positives (`JARVIS_WAKE_ONLINE_VERIFY`), and `openwakeword` keyword spotting when installed (`JARVIS_WAKE_ENGINE`) with graceful fallback to the fuzzy path. Fish playback feeds the actor + reference path on every device write; `test_voice_latency.py` cache assertions were updated to the identity-key contract. AEC tuning and keyword-spot quality stay [ASSUMPTION]-flagged (need the real speaker/headset recordings).
- **G11 (landed)** — process architecture & security (audit F50/F51/F52):

  * **F50 backend sole owner of intelligence state** (`backend/voice_mode.py`): the voice process is now a pure I/O worker — it no longer imports or executes `backend.core.brain` (the audit's "second, never-authoritative copy" bug). Every utterance is submitted to the ONE backend runtime via the authenticated `/ask/stream` SSE contract (`speak=False`, `origin=voice`); streamed deltas feed the same `StreamSpeaker`. Listening state is **published** (`POST /voice-state/publish`, authed) and the backend serves `/voice-state` + `/ui-state` from that snapshot instead of its empty `listener_state` module copy. Task state is **pushed** over the `GET /events` SSE channel from `backend/services/event_bus.py` (`backend_task_running`, plus speaking/thinking and the voice toggle; **S19 `a0ec729`**) — the worker subscribes and the old 1s-TTL `/ui-state` poll is retained only as a fallback if the channel drops. Stop/stop-research/cancel-approval controls route through the authed HTTP control plane (`/task/stop`, `/speak/stop`, `/approvals/reset`). A negation-aware `is_stop_research` lives in the worker (mirrors brain). `backend/api/routes.py`: `Query.speak`/`Query.origin` flags gate backend double-speaking; `_run_request_worker` gains a `speak_terminal` parameter so a voice submission stays silent; `/voice-state/publish` (authed) merges into `/voice-state` and `/ui-state` (monotonic `state_seq`); `/approvals/reset` invalidates pending consent + clears the pending screen plan + interrupts live registered requests before safe recovery.
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
- **STT hallucination gate (landed)** — `backend/services/transcription.py::is_hallucinated_transcript()` rejects prompt-vocabulary echoes (5+ tokens drawn only from the wake-bias vocabulary), looped tokens/phrases, memorised silence phrases ("a ver si te acuerdas de esto", "thanks for watching", ...) and filler-only utterances, while never rejecting real commands, short wake phrases, literal payloads or repeated safety words. It is wired into the single STT engine's accept path, the final `listen()` commit belt, the watcher's `is_wake_word`, and `_add_candidates`. (**The listener's partial-window layer it used to guard was deleted in `7afefc0`, and the multi-engine ladder collapsed to one engine per turn in S5 `c7b2128`** — `listener.py:1084`, `1270`, `1307` document the removal.) `whisper_daemon`'s wake-bias `INITIAL_PROMPT` is now opt-in via `X-Jarvis-Purpose: wake` (sent only by the watcher), so conversation transcription runs unbiased. Tests: `test_stt_hallucination_gate.py`.
- **Chat-outage root cause (2026-09-23, fixed)** — direct `generativelanguage.googleapis.com` degraded to 7-45s+ through the user's Proton VPN tunnel while the Fireworks account was suspended, so every query fell through to the generic failure message. Fixes: the `chat` role allowlist gained `openrouter`, `get_provider_credentials` returns canonical OpenAI-compatible base URLs for openrouter/groq, and `intent.py` now classifies OpenRouter -> Gemini -> Groq. **Live selections have moved since (settings `revision` 61): chat is `gemini/gemini-3.5-flash-lite`; vision is `fireworks-2/accounts/fireworks/models/qwen3p8-max`; browser_tool and planner are `fireworks-2/accounts/fireworks/models/deepseek-v4p1-flash`; intent is `gemini/gemini-3.1-flash-lite`; tts is `fish/s2.1-pro-free`; listening is `whisper/whisper-local`. `fireworks-2` is a CUSTOM provider entry, not the `fireworks` env provider.**
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
- It exposes an authed HTTP control plane on `JARVIS_WATCHER_CONTROL_PORT` with two DIFFERENT semantics: a warm sleep (`/stop`) brings down **only the voice worker and the UI** — `watcher.py:1471-1478` calls `stop_runtime(stop_electron=True, keep_backend=True)`, so the backend **stays warm and listening** along with the resident `whisper_daemon` (and the opencode/brave daemons) for an instant next wake — while a full shutdown (`/shutdown`) leaves nothing behind. Electron's stop button uses `/stop` and only falls back to killing port owners if the watcher does not answer.
- When Electron exits and the backend is no longer on the configured port (`JARVIS_BACKEND_PORT`, default `9999`), the watcher resumes listening.

## End-to-End Request Flows

### Typed UI flow

Path:

`frontend/renderer.js`
-> `POST /ask/stream` (SSE; F23 request id + reconnect; plain `POST /ask` is still the non-streaming form)
-> `backend/api/routes.py` (admission -> one job -> `_run_request_worker` -> `process_message`)
-> `backend/core/brain.py` (`process_message` -> `_process_message_inner`)
-> deterministic phrase routes first (memory / commitments / skills, explicit stops, pending confirmations and clarifications)
-> then, since P0-10: **route selection + speculative `_ChatRacer` first** (`brain.py:3846-3861`), then explicit task and code-tool handoff, screen control, explicit research, and — in `orchestrator` mode only — the G8 native tool-use loop
-> otherwise the legacy path: `classify_intent` raced against the speculative `_ChatRacer`, with the deterministic nets rewriting the verdict (see brain.py section). **S6 changed this: a plain-chat fast path can skip the LLM entirely, and a non-fast-path `chat` verdict is ANSWERED by the router's own `reply` field rather than by a second chat completion**
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
-> the same response is also spoken through `backend/services/voice.py` — unless the caller sent `speak=False`, which the voice worker always does. **Since S7 `066220c` the primary playback path is ONE live bidirectional Fish Audio WebSocket session per reply (`fish_voice.py:67-69`, `_FishReplySession`, `open_ws_reply`/`play_ws_reply` on the audio actor); the HTTP ladder below it is now the fallback, switchable with `JARVIS_FISH_WS`.**

Notes:

- The UI has `chat` mode and `command` mode.
- In command mode, the renderer prefixes the request with `command ` before sending it to `/ask/stream`.
- Request admission lives in `request_registry`: a retried request id reuses its result instead of re-executing, and a reused id carrying a different message is rejected. **The 1-second identical-message dedup is NOT in `request_registry` — it is in `backend/api/routes.py:500-510` on `POST /ask` only, and it is gated on `not query.request_id`. Since `frontend/renderer.js:73` always sends a client-generated id, that guard can never fire on the real UI path, and `/ask/stream` has no equivalent guard.**

### Route selection and the chat race (inside process_message)

`backend/core/brain.py` is now ~4360 lines. `process_message` (line 3672) binds the F20 turn job and delegates to `_process_message_inner` (line 3723). The order today:

1. The `clear memory` phrase list.
2. G9 memory / commitment / skill phrase ops (`memory_store.handle_memory_phrase`) — `remember that…`, `forget…`, `what do you remember about…`, `remind me to …`, `cancel the reminder`, `approve the … skill`. These run before every other route.
3. Explicit research stop (`is_stop_research`, `brain.py:3777`), then the pending research / task-action / opencode-handoff confirmations and the browser-clarification follow-up.
4. **Route selection and the speculative racer — FIRST since P0-10** (`brain.py:3846-3861`): `is_explicit_command` → `orchestrator_select_route` (`3849`) → `_ChatRacer` start (`3857`) for any non-`command` message that has a stream. The comment at `brain.py:3825` states this explicitly. **The racer is therefore no longer "legacy branch only" — it starts for every routable message, including ones that go on to explicit task or code-tool handling.**
5. `is_explicit_task_request` (`3864`) and `is_code_tool_request` (`3878`) hand straight to `handle_task_message`.
6. `maybe_handle_screen_control_message` (`3891`, skipped for `command …`).
7. `force_research` explicit research phrasings (`3901`, including the `deepsearch` keyword).
8. **G8 orchestrator handling** (`3936`): with `JARVIS_ORCHESTRATOR_MODE=orchestrator` the native tool-use loop is attempted and only a decline falls through to legacy routing; in the default `legacy` mode (the shipped default, `config.py:69`) the orchestrator is not selected at all.
9. Legacy routing: **deterministic chat fast path** (`brain.py:3978-3991` — `JARVIS_CHAT_FASTPATH`, default on, `is_definitely_plain_chat` at `brain.py:942` synthesises a `chat` verdict with **no network call at all**) → otherwise `classify_intent` (`3993`, budgeted by `INTENT_BUDGET_MS=3500` typed / `INTENT_BUDGET_VOICE_MS=1200` voice) → screen-question net → racer holdback on any non-chat verdict → fresh-info auto-search net → tool / research / screen / region / task branches → implicit `is_task_request` web-shaped handoff → chat → `command …` parsing. **Since S6 `1fd635a` a `chat` verdict from the classifier carries the router's own `reply` (`intent.py:60-61`, normalised `100-110`, defaulted `418-420`) and the brain speaks it via `_finalize_chat_reply` (`brain.py:2061`, `4209-4236`) — the second chat completion only happens if the router returned no usable reply.**

`classify_intent` (`backend/services/intent.py:394`) tries the **registry-selected `intent` role first**, then the shipped fallback chain: OpenRouter (`google/gemini-2.5-flash-lite`), then Gemini direct, then Qwen 3.6 27B on Groq — landing on a `chat` verdict if everything fails, which is exactly why the deterministic nets exist. The selected hop is skipped inside the chain so the same endpoint is never paid twice in one budget. All hops share ONE monotonic deadline (`_budget_timeout`), so the advertised `timeout_ms` is real rather than per-hop. Order was flipped to OpenRouter-first on 2026-09-23 because the Cloudflare-fronted OpenRouter endpoint stayed ~1.4 s while direct Gemini degraded to 7–45 s behind the VPN; **since F56 the selected role is hop 1 and the live override is `gemini/gemini-3.1-flash-lite`, so in practice hop 1 is Gemini, not OpenRouter.** **The module docstring still describes the old Gemini-first order and is stale** (`intent.py:22-24`), and the old hop-number comment the map used to cite at `intent.py:286` no longer exists.

### Voice conversation flow

Path:

`backend/voice_mode.py` (pure I/O worker since G11/F50 — it deliberately does NOT import `backend.core.brain`)
-> `backend/services/listener.py` (transcripts only ever committed after the F34 stabiliser and the STT hallucination gate; endpointing is Silero neural VAD since S29, and exactly ONE STT engine runs per utterance since S5)
-> `POST /ask/stream` on the one backend runtime, authenticated with the per-launch `X-Jarvis-Token`, sent with `speak=False` and `origin=voice`
-> `backend/core/brain.py` (`process_message`)
-> streamed deltas feed this process's `StreamSpeaker` (the backend stays silent for these requests)
-> `POST /voice-state/publish` publishes real listening state; task state arrives by **push** over `GET /events` (`backend/services/event_bus.py`, S19) with the 1s `/ui-state` poll retained as a fallback
-> `frontend/renderer.js` still polls `/ui-state` and mirrors the exchange in the UI (**the renderer has NOT adopted `/events`** — it cannot send the auth header from an `EventSource`)

**Since S18 `66bacbc` the full voice mute is gone.** While a task is running the listener still submits committed utterances to `/ask/stream`; the backend answers them conversationally and QUEUES action requests (max 3 in flight) instead of refusing them. `test_voice_task_mute.py` was rewritten for this. **Since S13 `2ce627f` a delivered background result joins the conversation history** rather than being spoken as an orphan.

**Barge-in robustness (`9839fe8`):** `_SpeakStopWorker` re-dials a fresh socket immediately when the keep-alive socket is dead (`_is_stale_socket`), so a stop is never lost to a stale connection. HTTP answers (401/403) are still not retried.

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
-> local GPU-accelerated faster-whisper served by the persistent `backend/whisper_daemon.py`, **medium** model by default (`JARVIS_WHISPER_MODEL`, `bedd9e0`), greedy decode, VAD filtering, disabled repetition conditioning. **There is NO Google/Groq fallback ladder in the watcher since S5 `c7b2128` — `watcher.py:934-948` runs exactly ONE engine, ONE pass, and its output IS the transcript; another provider is contacted only if it is itself the selected listening engine.** Transcription is pinned to English (`JARVIS_WHISPER_LANGUAGE=en`, `8ae2294`), so Whisper no longer auto-detects and cannot hallucinate in a random language. Optionally preceded by the `JARVIS_WAKE_PRE_ROLL_SECONDS` pre-roll ring and followed by online Whisper verification (`JARVIS_WAKE_ONLINE_VERIFY`)
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
  - Shows status, chat history, a text input with chat/command modes, and the model-switcher sidebar (per-role sections: TEXT CHAT, INTENT CLASSIFIER MODEL, VOICE MODEL (TTS), LISTENING MODEL (STT), VISION MODEL, BROWSER TOOL, PLANNER — the intent section is F56).

- `frontend/renderer.js`
  - Sends `/ask` requests.
  - Adds the `command ` prefix in command mode.
  - Polls the fused `/ui-state` endpoint — **120 ms** while active (hearing/thinking/speaking; lowered from 250 ms, `renderer.js:515-519`), 600 ms when idle, exponential backoff up to 3 s on failures. It does **not** use the `GET /events` push channel (an `EventSource` cannot send the auth header).
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
  - On startup: `validate_environment()`, writes the `data/runtime/backend-instance.json` identity stamp, and starts **three** warm-up threads — Ollama (`main.py:53`), TTS (`main.py:58`), and provider pre-warm (`main.py:62-76`, calling `backend/services/prewarm.py::warm(force=True)`, landed by P0-12).
  - Includes routes from `backend/api/routes.py`.

- `backend/app.py`
  - Secondary minimal FastAPI entrypoint.
  - Also includes the router, but current launch scripts use `backend.main:app`, not this file.

- `backend/api/routes.py` — the whole local control plane (~1835 lines). Every mutating endpoint AND every private read requires the per-launch `X-Jarvis-Token` unless `JARVIS_DEV_MODE=1`; only `GET /health` stays open.
  - `POST /ask` — non-streaming request; F23 request-id admission (a retried id reuses its result instead of re-executing; the same id carrying a different message is a 409), a 1-second duplicate guard **that only fires when the caller sends no `request_id`** (the renderer always sends one, so it never fires in practice), and it speaks the reply unless the caller passed `speak=False`.
  - `POST /ask/stream` (SSE) — the primary UI and voice path. One frame shape `{type, seq, request_id, ...}` with `delta|replace|progress|completed|interrupted|error`; reconnecting with `last_event_id` resumes without re-executing; a voice submission (`speak=False`, `origin=voice`) runs the same runtime but stays silent.
  - `GET /ask/status/{request_id}` — status lookup for reconnect checks.
  - `POST /ask/cancel/{request_id}` (`routes.py:648`) — explicit client-side cancellation of one identified request; added with the 2026-09-30 voice-path fixes.
  - **`GET /events` (SSE, decorator at `routes.py:973`, handler `events_stream` at `:974`)** — the S19 push channel: one `event_bus` stream carrying `snapshot` first, then live `task_running` / `voice_enabled` / `voice_state` events, with a 15 s `: ping` heartbeat. It is an async generator that holds **no thread-pool slot**, so it cannot starve barge-in. Authed like any other private read.
  - `POST /update-voice-log`, `GET /voice-log` — the latest single voice exchange for the UI mirror.
  - `GET /voice-mode`, `POST /voice-mode` — voice input on/off.
  - `POST /voice-setup/launch` — runs the "normal setup" applications as a typed backend job (F50).
  - `GET /voice-state`, `POST /voice-state/publish` — the voice I/O worker publishes its real listening state (owner incarnation + generation; a stale generation or an expired snapshot is rejected) and the backend serves it.
  - `GET /ui-state` — fused `{state, voice_log, voice_input_enabled, task_running}`; still the renderer's read, and the voice worker's **fallback** since S19 pushed the same data over `/events`.
  - `POST /speak/stop`, `POST /speak/pause`, `POST /speak/resume`, `GET /speak/remaining` — F35 barge-in / pause / continue controls (the pause remainder is published here because the voice process is a separate OS process).
  - `GET /aec/state`, `GET /aec/reference` — F33 echo-cancellation diagnostics and the cross-process rendered-PCM reference span. `GET /aec/state` now also reports the S28 `lag` block.
  - `GET /latency`, `POST /latency/client` (`routes.py:1274`, `:1319`) — the per-turn waterfall that stitches model/STT/TTS/playback marks under one request id (P1-19; `backend/services/latency.py`).
  - `POST /prewarm`, `GET /prewarm/stats` (`routes.py:1765`, `:1791`) — connection pre-warm control and counters (P0-12).
  - `POST /task/stop` — F20: cancels ONE identified job (or the newest still-running one) at its next checkpoint; an idle stop cancels nothing.
  - `POST /approvals/reset` — invalidates a pending consent, clears the pending screen plan and interrupts live registered requests before a supervisor replacement.
  - `GET /health` — liveness plus identity (`instance_id`, `pid`, `protocol`, `build`, `auth` fingerprint).
  - `GET`/`POST /screen-answer` — F30 publish-or-patch with capture-generation and revision guards (a stale patch gets 409 instead of clobbering a newer answer).
  - `GET`/`POST /research-result` — deep-research report delivery for the research overlay.
  - `GET`/`POST /research-progress` — F28 out-of-band research progress and incremental evidence.
  - `GET /settings`, `POST /settings/model`, `POST /settings/chat-model`, `POST /settings/provider`, `POST /settings/provider/test` (`routes.py:1820`), `GET /providers/{provider_id}/models` — the model-registry surface for all **seven** roles (`chat`, `tts`, `vision`, `browser_tool`, `listening`, `planner`, `intent`); listings are masked (`has_key` booleans only) and selections are capability-validated.

### Core decision layer

- `backend/core/brain.py`
  - This is the main router for user intent.
  - Most important file in the repo for behavior changes.

Routing order inside `process_message` is listed in "Route selection and the chat race" above (it changed and grew: memory phrases run before everything; **route selection and the `_ChatRacer` start moved ABOVE the task/code-tool/screen/research chain in P0-10, so the racer is no longer "legacy branch only"**; the G8 orchestrator route selection precedes the legacy classifier; and there is a fourth branch — tool-search steps reroute to research). The deterministic nets are unchanged in intent:

1. Pending confirmation/follow-up gates (research confirm, task confirm, opencode confirm, browser clarification).
2. `maybe_handle_screen_control_message(...)` (bypassed if prefixed with `command`).
3. Deterministic chat fast path (`JARVIS_CHAT_FASTPATH`, default on) — plain chat skips the LLM entirely; otherwise a speculative `_ChatRacer` start plus `classify_intent(msg)`. **Since S6 a classifier `chat` verdict is answered by the router's own `reply` field.**
4. **Screen-question safety net**: if the verdict is `chat` or `research` and `is_screen_question(msg)` (deterministic regex in `screen_analyzer.py`), the verdict is rewritten to `region`/`screen`. tool/task verdicts are exempt because they carry structured steps the upgrade would discard.
5. Racer holdback: any non-chat verdict cancels the speculative stream.
6. **Fresh-info auto-search net**: a `chat` verdict that hits `should_search()` (pricing/cost/latest/news keywords) and is question-shaped but not greeting-like reroutes to `handle_research_intent` (quick-search tier), so stale chat answers are never served for current-world facts.
7. Tool steps branch (an all-`search` step list reroutes to research), research branch, screen/region branch (screen Q&A), task branch (opencode/browser-agent handoff, confirmation-gated).
8. **Web-task routing net**: the broad `is_task_request` heuristic, when `TASK_ENGINE=browser_agent`, checks `is_web_shaped_task` (`backend/services/web_task_routing.py`: web hint + interaction verb, no local hint) and hands the task to the confirmation-gated browser-agent path instead of the raw task-message path.
9. Memory reset phrases, command-mode parsing, legacy chat path.

Important chat behavior:

- The chat model is resolved per message by the model registry â€” a UI model switch takes effect on the next reply, no restart.
- Provider chain (`_stream_chat_deltas` / `_ask_chat_nonstream`): the registry-selected provider goes first. `gemini` and `fireworks` use their dedicated clients; any other provider (openrouter, groq, a user-added custom provider) is carried by the OpenAI-compatible path via `get_provider_credentials`. An empty stream retries the SAME model non-stream before any provider fallback, and a terminal failure (auth/permission/validation) REFUSES instead of silently answering with a different model (F24/F49). The tail fallback chain is still Gemini -> Fireworks. Fallbacks are recorded for a UI warning (`_record_chat_fallback`, keys scrubbed).
- Current selection per `data/jarvis_settings.json` (verified 2026-10-03, `revision` 61): chat = `gemini`/`gemini-3.5-flash-lite`, vision = `fireworks-2`/`accounts/fireworks/models/qwen3p8-max`, browser_tool = `fireworks-2`/`accounts/fireworks/models/deepseek-v4p1-flash`, **planner = `fireworks-2`/`accounts/fireworks/models/deepseek-v4p1-flash`**, **intent = `gemini`/`gemini-3.1-flash-lite`**, tts = `fish`/`s2.1-pro-free`, listening = `whisper`/`whisper-local`. `fireworks-2` is a **custom** provider entry in the settings file, distinct from the `fireworks` env provider. Reading only `.env` or `config.py` will tell you the wrong story — this file is the live truth.
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
  - `classify_intent(message, timeout_ms=3500)` routes every message to chat/tool/screen/region/research/task — no keyword pre-check. **Its JSON also carries a `reply` field since S6 `1fd635a`: on a `chat` verdict the brain speaks the router's own answer (`intent.py:60-61`, normalised `:100-110`, defaulted `:418-420`, consumed by `brain._finalize_chat_reply`), so a normal chat turn costs ONE LLM call, not two.**
  - Hop 1 since F56 is the `(provider, model)` selected for the registry's `intent` role (UI model switcher) — including the local Ollama models, which is what makes routing free/offline and fast enough for the typed budget. The shipped chain follows as the fallback: OpenRouter `google/gemini-2.5-flash-lite` (fastest cloud hop since 2026-09-23) -> Gemini 3.5 Flash Lite direct (`GEMINI_INTENT_MODEL` / `GEMINI_BRAIN_MODEL`, 3s timeout, `no_retry`) -> Qwen 3.6 27B on Groq (`GROQ_INTENT_MODEL` / `GROQ_VISION_MODEL`) -> `chat` verdict on total failure. The selected provider is skipped in the chain, and the registry's env default for `intent` IS the OpenRouter hop, so an untouched install routes exactly as before. **The live override is `gemini/gemini-3.1-flash-lite`, so in practice hop 1 is Gemini today, not OpenRouter.** Chat is therefore never broken, but classifier throttles silently degrade to chat verdicts — which is exactly why the deterministic nets in `brain.py` exist.
  - Each hop gets a slice of ONE monotonic deadline and is dispatched explicitly (`_classify_with_provider`: gemini/fireworks/groq have dedicated clients, everything else rides the generic OpenAI-compatible adapter with `single_attempt=True`, so a read timeout cannot be replayed under a spent budget). The prompt is length-capped (test asserts <2000 chars) and covers English/Hindi/Hinglish meaning, not keywords. **The budget is 3500 ms for typed turns but only `INTENT_BUDGET_VOICE_MS = 1200` for voice (`brain.py:3993-3994`).**
  - **The module docstring (`intent.py:22-24`) still describes the old Gemini-first hop order and is stale.**
  - Note the practical limit F56 measured: a hop is bounded by its socket read timeout, not by a wall clock, so an endpoint that sends headers then stalls can take up to ~2x its slice.

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
  - Model comes from the registry `browser_tool` role (env default Fireworks `accounts/fireworks/models/qwen3p7-plus`; **`data/jarvis_settings.json` currently overrides to the custom provider `fireworks-2`/`accounts/fireworks/models/deepseek-v4p1-flash`**), with `MAX_STEPS` (50) / `TIMEOUT` (480s) backstops and a stop event (`/task/stop`). Resolution is per-turn via `_resolve_browser_tool_model`.
  - Every handoff is confirmation-gated in the brain before anything executes.
  - **BA-00 (committed `ff86fc1`, 2026-10-03)**: turn-level instrumentation — `_Span` (bounded 64/turn ring, covers look/model/act/wait phases; NO `model.ttfb` span because POSTs are non-streaming) + `_SwStats` spans/census (`model_turns`, `wasted_turns`, `stale_refusals`, `stale_refusals_other`, `offscreen_refusals`, `look_count`, `image_tokens_est`, `upload_bytes_total`) uploaded per task via `_perf_record_task` into the `_PERF_MAX_SAMPLES=512` cross-task reservoir, published at `GET /latency` under the `browser_agent` block, and `_wasted_pct` flushed in the summary STOPWATCH line. Refusal/outcome classification for the census comes from `_ba00_classify_result` (exact virtual-tool error prefixes). Tests: `backend/tests/test_ba00_instrumentation.py` (24).
  - **L-8 / BA-02 (committed `f3855bd`, 2026-10-03)**: `_model_turn_with_retries` fails FAST on deterministic HTTP rejections (status outside `_RETRYABLE_STATUS={408,409,425,429,500,502,503,504}` breaks after attempt 0 with a `model deterministic` STOPWATCH line) and retries only transient/transport errors (3-attempt budget kept); backoff honours the provider's `Retry-After` (capped `_RETRY_AFTER_CAP_S=10.0`) over the flat `_RETRY_GAP_S=0.5`; the final raise carries the provider's own message (`_provider_message`: JSON `error.message`/`message`/plain text, `_model_failure_message`), and bare errors without a provider response still retry. Tests: `backend/tests/test_ba02_retry_classification.py` (18).
  - **L-6 (pooled MCP session, committed — see below — 2026-10-03)**: verified against current code first — every task built a fresh `BraveMcpClient`, paid initialize + notifications/initialized + a `tools/list` round trip, then closed, while the daemon stays warm (idempotent `ensure_brave_mcp_daemon`, watcher boot) and the tool list is code-static (`server.tool()` in `server.mjs`, client frozenset allowlist). So sequential tasks now share ONE pooled session (`_borrow_mcp_client`/`_return_mcp_client`, 180s idle TTL, checkout flag so concurrent tasks never share request ids — the second takes the legacy private path) plus a raw-def tool cache (dropped whenever the session is reborn, guarding daemon upgrades); the per-task `ensure_daemon` call STAYS (pool never skips a dead-daemon restart). Reuse is optimistic with no borrow-time health RTT — a restarted daemon heals inside the client (reconnect + exactly one replay, ONLY for proven-never-dispatched 404/invalid-session signals; mutation replay stays forbidden otherwise). Only real clients pool (factory-identity check), so patched-mock test paths behave byte-identically. Speculative navigation was verified but deliberately LEFT OUT (changes first-turn semantics, needs benchmark proof). While the user reads the confirmation prompt, a daemon thread prewarms session + cache (`prewarm_mcp_pool`, no navigation, never spawns the daemon), gated by `_browser_prewarm_due` (frozen contract executor wins, else `TASK_ENGINE`). Kill switches (default on): `JARVIS_MCP_POOL=0`, `JARVIS_MCP_PREWARM=0`. Tests: `backend/tests/test_l06_mcp_pool.py` (25).
  - **L-1 / BA-05 (committed — see below — 2026-10-03)**: the DOM-mutation epoch demoted from refusal to hint. A drifted `mut` counter no longer refuses the click (it fired on unrelated noise: class toggles, aria-live, lazy images, spinners) — it only widens the mark rect tolerance 3x (`_MARK_RECT_TOLERANCE` 12 → 36), while the element is still re-queried by cssPath and must exist, be visible and sit within tolerance. Document/URL/DPR/frame/tab/exists/visible refusals are UNCHANGED (F39 identity intact); legacy refusal restorable via `JARVIS_BROWSER_STALE_ON_MUTATION=1`. The look inventory's MutationObserver is scoped to `{childList: true}` (no subtree/attributes) so the counter tracks structural changes only. Fallout fix included: `_parse_locator_outcome("")` IndexError'd on empty daemon replies (a path the new policy makes hotter) — now a clean failure. BA-00's `stale_refusals` census is expected to drop to near zero. Migrated pins: `test_f39_mark_identity.py` (3 mut tests), `test_g6_browser_grounding.py` (SPA-update test). Tests: `backend/tests/test_ba05_staleness_hint.py` (17).
  - **L-15 / BA-03 (committed — see below — 2026-10-03)**: the single shared 120 s budget is split three ways. `BraveMcpClient.call_tool` takes a per-call `timeout` and carries per-class wire budgets (probes 8 s, real-input actions + screenshot 15 s, navigate 30 s, file transfer 45 s, unlisted tools fall back to the client default); `config` gains `BROWSER_AGENT_MODEL_TIMEOUT` (~90 s) while `BROWSER_AGENT_TOOL_TIMEOUT` drops to ~30 s (both env-overridable). One absolute `core/deadline.py` task budget, built from the same `started` origin the timeout message reports, is bound to the task thread in `run_browser_task` and passed explicitly to `_agent_loop`/`_agent_loop_inner` — the step-top check is now `deadline.expired()` instead of the ad-hoc monotonic comparison, every MCP POST and model POST slices its timeout to what is actually left, a spent budget sends nothing (`BudgetExhausted`), the model retry loop never retries a spent budget, and both retry-gap sleeps are deadline-capped (plain sleeps when unbound, so tests and one-off callers behave exactly as configured). Task wall-clock can never exceed `BROWSER_AGENT_TIMEOUT` by more than the one call already in flight; `overshoot_ms` in `_emit_summary` stays as the observational check. Tests: `backend/tests/test_ba03_timeouts.py` (14 + 9 subtests).
  - **L-16 / BA-15 (committed — see below — 2026-10-03)**: off-screen marks are no longer refused when they carry an addressable identity. `click_locator` already scrolled into view inside the same real-input call (`scrollIntoViewIfNeeded`); `fill_locator` gained the identical line (daemon parity), so the refuse -> scroll -> look -> click four-turn dance collapses to one turn reporting `scrolled: true` on success (viewport moved — look again before trusting coordinates). Success-only: failed clicks/fills report no scroll, and identity refusals (navigated/reloaded/mutated targets) still refuse with look-again before any scroll. Only identity-less coordinate marks keep the up-front refusal (no selector = nothing to scroll to); the typed-interaction resolver (`_resolve_mark_or_css`) is unchanged, so `offscreen_refusals` approaches but need not hit zero. Model-facing `click_mark`/`fill_mark` descriptions now say off-screen targets auto-scroll. Tests: `backend/tests/test_ba15_offscreen_click.py` (8).
  - **L-5 / BA-14 (committed — see below — 2026-10-03)**: `wait_for` is event-driven instead of polling every 200 ms. The daemon owns a real `wait_for` tool (`selector?`/`text?`/`state?`/`timeout_ms`, backed by `locator.waitFor`/`getByText`, frame-aware, zero-timeout = one immediate check, invalid selectors report `found:false` with an error note like the old polling semantics) returning `{found, elapsed_ms, url, title}` in one round trip, plus opt-in `awaitPromise` on `evaluate` so async expressions stop being structurally impossible (default path byte-identical). The agent prefers the daemon waiter whenever the task's daemon tool list advertises it (`session["daemon_tools"]`, captured before allowlisting since capability is about what the daemon HAS, not what the model may SEE) and keeps the Python polling loop as the capability-gated fallback for old daemons — including an unknown-tool race fallback that triggers ONLY on unknown-tool markers, never on real daemon errors (those fail loudly, as does daemon garbage). The BA-00 `wait.polls` span closes with `polls=1` on the event path; the new tool gets a 15 s class wire budget (BA-03). Model-facing `wait_for` description no longer says "polling every 200ms". Tests: `backend/tests/test_ba14_event_wait.py` (11).
  - **L-14 / BA-06 (committed — see below — 2026-10-04)**: internal read-only `evaluate` probes no longer die on one transport blip. New `_probe(client, expression, attempts=2, timeout=8.0)` (short timeout, one 50 ms retry) fronts all eleven read-only internal sites (`_handle_look`, `_capture_state`, `_mark_target_state`, `_after_state`, `_click_and_confirm` url re-read, `_page_url`, `_real_locate`, `_read_control_state`, `batch_probe`, the old-daemon `wait_for` poll loop, `_probe_media_state`); the two MUTATING evaluates (coordinate click, legacy coordinate fill) keep their single attempt, as does model-issued `evaluate` via `_MUTATION_TOOLS`. Clients without the L-15 per-call `timeout` kwarg (older daemons, test fakes) fall back to the client default via `TypeError`. Migrated pin: `test_click_mark_second_evaluate_failure_degrades_gracefully` now needs two consecutive failures (a single blip is absorbed) plus a new single-blip-absorbed test. Tests: `backend/tests/test_ba06_probe_retry.py` (8).
  - **I-2 / BA-07 (committed — see below — 2026-10-04)**: the prompt no longer describes tools that do not exist, and `evaluate` is no longer advertised-then-forbidden. All four phantom/prohibition blocks deleted from `_SYSTEM_PROMPT` (5,648 → 4,971 chars; the estimate was 773, the four blocks measure 677 — the removed source lines total 820); `evaluate` left `_TOOL_ALLOWLIST` so its schema never reaches the model, but stays in `_TOOL_DISPATCH_ALLOWLIST` for the internal handlers (model-issued `evaluate` now refuses at the dispatch name gate even WITH the `privileged_js` grant; the SyntaxError/IIFE guidance still serves the internal origin). Import-time `_PHANTOM_TOOL_MENTIONS` guard fails any future drift ("screenshot" deliberately excluded — plain English in the prompt, never advertised). Migrated pins: allowlist-filter test, dispatch-membership test, three prompt-content tests, two F17 grant tests, the truncation test (now on `navigate`); model-path `evaluate` in tests now only pins the refusal. Tests: `backend/tests/test_ba07_phantom_tools.py` (11).
  - **L-18 / BA-08 (committed — see below — 2026-10-04)**: activity logging is async and non-blocking. `append_activity_line`/`narrate_activity` only enqueue (never open/flush/speak on the request path, never raise — rule 19); one daemon writer thread in `opencode_client` drains the bounded queue (`JARVIS_ACTIVITY_QUEUE_MAX`, default 1000) in batches of up to 200 — ONE file open per batch, redaction moved to the writer, overflow drops with a counter (`activity_queue_stats`), narration speaks on the writer after claiming its throttle slot on the caller (nanoseconds). `truncate_activity_log` drains (bounded 2 s) then truncates under the file lock so a batch cannot land after it. The engine path (`_append_activity` direct callers) stays synchronous. Drive-by fix: the `_on_job_cancelled` stop handler armed the module-global `_STOP_REQUESTED` for ANY cancelled browser job — including jobs this module never owned (e.g. a /task/stop test creating/cancelling jobs directly) — leaking "user pressed STOP" into the next unrelated loop (the pre-existing websearch→ba14/ba03 order failure, verified present without these changes too). Now only runs owned via `run_browser_task` (tracked in `_OWNED_JOB_IDS`) arm it. Tests: `backend/tests/test_ba08_async_logging.py` (12).
  - **Live chat-routing fixes (2026-10-04, committed `28cc112` — user-reported)**: three failures from one live session, each traced to a deterministic cause. (1) "Create a folder on the desktop by the name Mayank Malik" matched NO folder regex (`_FOLDER_TAIL_HINT_RE` lacked `on`/`by the name`, `_folder_plan_path` lacked the `by the name` capture) so it fell through every route into tool-less chat, where the model invented "I cannot create files or folders". Both regexes now cover the phrasing; the request routes to the gated `code.create_folder` task path with the OneDrive-aware desktop path. (2) The "unable to browse websites" reply is the same shape of failure one layer up: the turn resolved to plain chat (zero tools attached), so the refusal is model confabulation, not a capability check — the English chat system prompt now grounds capabilities (task routes do actions; never claim inability) and recall (report only what the turns show; deny absent premises instead of agreeing). (3) Memory itself was faithful — NZ + the folder refusal genuinely were in the 20-entry window — but SEVEN identical `[background result] folder has been created` turns (a completion re-firing) crowded real conversation out of the window; `_notify_async_reply` no longer appends a repeat of the immediately-preceding entry (delivery/callback semantics unchanged). Tests: new `by the name` routing + planner pins in `test_code_tools.py`, repeat-dedupe pins in `test_s13_background_history.py`.
  - **Desktop grant + blank-tab fixes (2026-10-04, committed — see below — user-reported)**: the follow-up live session surfaced two more deterministic bugs. (1) The folder request now PLANNED fine but was denied AFTER confirmation: `code_grants.default_roots()` (repo root, cwd, tempdir) never included the Desktop, and confirmation never expands scope — consent given, work refused. The real Desktop dir (OneDrive-aware via `FOLDERID_Desktop`, `~/Desktop` fallback) is now a default root; explicit `JARVIS_WORKSPACE_ROOTS` still replaces the defaults. Safe because Desktop writes are already user-explicit + spoken-previewed + confirmation-gated. (2) Every research lookup stranded one `about:blank` tab: Playwright persistent contexts spawn with a default blank page, but `ResearchTask.new_page()` always opened a SECOND tab and cleanup closed only tracked pages. Now `new_page()` adopts an untracked blank page when one exists (callers `goto` it anyway), `submit()` goes through the same path, and `close_pages()` additionally sweeps any leftover blank page (real-URL pages are never touched — they may belong to a concurrent task). Tests: `test_f22` desktop-root pins, `test_f27` blank-reuse suite + f27/g5 fake-context default-blank updates.
  - **Follow-up false-completion fix (2026-10-04, committed — user-reported)**: "now create a text file inside that folder ... write hello" was answered with "I will get that created" while NO tool ran. Root cause chain: (1) `is_code_tool_request`'s anchored verb regexes rejected the leading "now", the "text file" adjective, and the nameless located shape ("inside that folder"), so the turn fell through every deterministic route into plain chat; (2) even routed, `_split_write_request` had no "create ... inside X ... write Y" shape and nothing resolved "that folder" across turns. Fix: routing strips leading spoken fillers and accepts located writes (location + content = intent); the heuristic planner resolves "that folder/inside it" from the last native run's artifacts (create_folder path first, else parent of last write; trace-args recovery when the artifacts list is empty; asks which folder when unresolvable instead of guessing); nameless files default to `hello.txt`, spoken verbatim in the confirmation preview; a defensive `is_code_tool_request` re-check runs after the classifier so a misrouted action still reaches the task path; the chat prompt now forbids promising/done-claiming from the tool-less route. Tests: new `LocatedWriteTests` (routing, resolution, clarification, confirm→write end-to-end) + content-is-not-location pins.
  - **Screen Q&A history fix (2026-10-04, committed — user-reported)**: "look at my screen ... find out when this volume releases" was analysed and answered live, but NEITHER half of the exchange entered conversation history (only the voice log saw it), so the very next turn — "search on the internet for the release date of this volume" — had no referent and the chat model asked "which specific volume are you referring to?". The legacy screen/region branch in `_process_message_inner` now commits both halves exactly like `handle_chat`: `add_message("user", msg)` when the analysis starts, and `_commit_chat("assistant", tip)` for the spoken tip under `commit_response`. Test: `test_screen_turn_is_committed_to_history` in `test_brain_gate.py`.

- `backend/services/brave_mcp_client.py`
  - Streamable-HTTP MCP client for the brave-control daemon (`BRAVE_MCP_PORT`, default 9570).
  - **L-6 (2026-10-03)**: self-healing session — `_is_session_death` (HTTP 404 or invalid/unknown/expired-session messages, which prove the request never dispatched) triggers exactly one `reconnect()` (fresh TCP session + handshake, `reconnects` generation counter) plus one replay; every other failure propagates to the agent's retry policy untouched, so a possibly-committed mutation is never auto-replayed.

- `backend/services/opencode_client.py`
  - The alternative engine (`JARVIS_TASK_ENGINE=opencode`): hard guards refuse to spawn anything unless that engine is selected.

- `backend/services/task_agent/agent.py`
  - Connector-first task planning brain (heuristics -> model plan -> Windows control fallback).
  - The model planner (`_model_plan`, **`agent.py:931`**) calls `ask_fireworks` (temperature 0.1, max_tokens 650, no model arg -> `FIREWORKS_MODEL` default `accounts/fireworks/models/deepseek-v4-flash-0731`). It no longer uses Groq — the old `llama-3.3-70b-versatile` planner 404'd after the model was retired.
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
  - The single runtime source of truth for model selection across SEVEN roles: `chat`, `tts`, `vision`, `browser_tool`, `listening`, `planner`, `intent` (`VALID_ROLES`). The `intent` role (F56) is what `services/intent.py` classifies with.
  - Persisted overrides live in `data/jarvis_settings.json` (set from the UI model switcher, immediate effect, no restart); missing/corrupt settings degrade to env defaults.
  - Env providers: `gemini`, `fireworks`, `groq`, `fish`, `gtts` (Google Translate TTS), `openrouter`, `whisper` (local), `inworld` (STT), `ollama` (the LOCAL server, F56 — key-less, base URL from `ollama_client`, model list live from `/api/tags`). Custom OpenAI-compatible providers are allowed for `chat`, `vision`, `browser_tool`, `planner` **and `intent`** only (`_ROLE_ALLOWS_CUSTOM`, `model_registry.py:111`) — this set CHANGED after the map was first written; the live `fireworks-2` custom provider is what carries the current selections.
  - Per-role allowlists: chat `{gemini, fireworks, openrouter, ollama}`; tts `{fish, gtts}`; vision and browser_tool `{gemini, fireworks, groq, openrouter}`; listening `{whisper, inworld}`; planner `{fireworks}` only; intent `{gemini, fireworks, groq, openrouter, ollama}`.
  - **F49 capability-aware selection**: every role declares what it REQUIRES (`chat` streaming, `tts` audio_output, `vision` vision_input, `browser_tool` tool_calling + structured_output + vision_input, `listening` speech_input, `planner` tool_calling + structured_output + streaming, `intent` structured_output) and a (provider, model) pair is only usable when those capabilities are positively established from the adapter floor, the provider record, model-family name rules, or capability metadata the provider published. Anything unknown FAILS CLOSED, resolution is validated at use time from one locked snapshot, and a persisted selection that stops validating surfaces as a `model_errors` entry in `GET /settings` instead of running.
  - The `ollama` provider floor advertises `tool_calling` + `structured_output` + `streaming` and NO `vision_input` (the installed local models are text-only), and its `_reasoning_for` entry declares `reasoning_effort="none"` — the one value that turns thinking off for a thinking model and is accepted by a non-thinking one (F56, measured).
  - API keys never leave the module — everything the routes return is masked (`has_key` booleans), and log lines are scrubbed.
  - Current live selections (verified 2026-10-03, `revision` 61): chat `gemini`/`gemini-3.5-flash-lite`; tts `fish`/`s2.1-pro-free`; vision `fireworks-2`/`accounts/fireworks/models/qwen3p8-max`; browser_tool `fireworks-2`/`accounts/fireworks/models/deepseek-v4p1-flash`; **planner `fireworks-2`/`accounts/fireworks/models/deepseek-v4p1-flash`** (an override now exists — the earlier "no planner override" note is obsolete); **intent `gemini`/`gemini-3.1-flash-lite`**; listening `whisper`/`whisper-local`. The file also carries an `observed_capabilities` map populated from live provider probes. Backup copies sit next to it (`*.bak-*`, written by P1-08).

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
  - Chat client for user-added custom providers (any /v1-compatible endpoint) — and, since F56, the carrier for the LOCAL Ollama provider and the intent classifier's generic hops. It defaults `reasoning_effort="none"` for a local Ollama endpoint (thinking off; the native `think` field is ignored there) and replays once without the field if an endpoint rejects it, and it offers `single_attempt=True` for callers that own one slice of a shared deadline (see "Local models as selectable brains (F56)").

- `backend/services/transcription.py`
  - Shared speech-to-text ladder: Google STT primary, Groq Whisper-style (`whisper-large-v3-turbo`) network fallback, local Whisper over the persistent `whisper_daemon` (`JARVIS_WHISPER_PORT`, default 8767), and Inworld STT (`INWORLD_STT_*`).
  - Hosts the STT hallucination gate (`is_hallucinated_transcript`) that every transcript-commit path consults — see the 2026-09 review wave above.

- `backend/services/ollama_client.py`
  - Local inference to the `llama3.2` model on port `11434` for the accessibility screen-control planner. Its `OLLAMA_BASE_URL` is the ONE source of truth for the local endpoint — `model_registry` appends `/v1` to it for the generic chat/intent adapter (F56) rather than hardcoding the port a second time.

### Modules added by the Fable-5 remediation wave

One-line roles for the modules the G0-G11 work introduced that are not described above, so a future model knows where to look:

- `backend/services/vision_cascade.py` (F37) — the one eligible-provider vision dispatcher used by screen Q&A: ordering, bounding, attempt metadata.
- `backend/services/research_browser.py` (F27/F28) — one long-lived research browser worker; `submit`/`run` job API, per-task page cancellation, idle TTL.
- `backend/services/intelligence_state.py` (F50) — backend authority layer: `run_effect`/`submit_effect` (one job runtime for every effect), `WorkerRegistry` (typed worker generations; stale publishes rejected), `DurableOwnership` (single-writer leases), `EventJournal` (one shared transactional history).
- `backend/services/capability_contract.py` (F16) — immutable `ExecutionContract` (capability + selected executor + availability + grant + digest) that executors verify instead of re-reading configuration.
- **`backend/core/deadline.py` (F24)** — one absolute monotonic deadline/cancellation handle plus a shared replay-eligibility classifier, propagated into urllib3 retries. **NOTE the path: it is `backend/core/deadline.py`, NOT `backend/services/deadline.py` — older revisions of this map named the wrong path.** `Deadline`, `classify_exception`, `budget_allows`, `BudgetedRetry`, `budget.seconds_for(handle, default)`. `backend/core/executor.py` uses it correctly (`budget.seconds_for(handle, timeout)`); **`backend/services/browser_agent.py` does NOT use it at all — it rolls its own `started` clock against `BROWSER_AGENT_TIMEOUT`.**
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
- `backend/whisper_daemon.py` (F52/F55) — the persistent local Whisper HTTP daemon (`/health`, `/warm`, `/transcribe`). Model default is **medium** (`JARVIS_WHISPER_MODEL`, `bedd9e0`), transcription pinned to **English** (`JARVIS_WHISPER_LANGUAGE=en`, `8ae2294`).
- `backend/services/google_tts.py` — the key-less Google Translate TTS engine (an interchangeable `tts` role engine alongside Fish).
- `backend/services/quick_search.py` / `research_service.py` — the two research tiers (see the research flow).

### Modules added by the S-series wave (2026-10-02/03)

These post-date the Fable-5 work and are easy to miss:

- `backend/services/event_bus.py` (S19, `a0ec729`) — dependency-free thread-safe pub/sub behind `GET /events`: `subscribe`/`unsubscribe`/`publish`/`snapshot`, bounded queue (64, drops oldest), optional cross-thread `wakeup`, plus a `reset()` test seam. A late or reconnecting subscriber gets a merged snapshot replayed first.
- `backend/services/neural_vad.py` (S29, `acc366e`) — Silero-on-ONNX utterance endpointing with hysteresis (100 ms on / 200 ms off) and a `webrtcvad` fallback judge; `JARVIS_NEURAL_ENDPOINT=0` is the kill switch.
- `backend/services/latency.py` (S1/S12) — the per-stage latency marker store behind `GET /latency` + `/latency/client`; `brain.py` writes ~20 `_mark_latency` sites and the P1-19 waterfall mark names live here.
- `backend/services/prewarm.py` (P0-12) — connection pre-warm for the provider clients; wired into `main.py` and exposed as `/prewarm` + `/prewarm/stats`. Covers LLM providers, **not** the brave MCP daemon.
- `backend/services/fish_voice.py` — also gained the S7 live WebSocket reply session (see the voice subsystem below).

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
  - Publishes real listening state to `POST /voice-state/publish` and reads task state from the **`GET /events` push channel** (`_apply_pushed_state`, `_task_state_push_loop`, started in `start_voice_mode`), with the 1s `/ui-state` poll retained as a fallback if the channel drops (S19).
  - **Since S18 `66bacbc` it keeps submitting utterances while a task runs** — the backend answers conversationally and queues action requests instead of refusing them.

- `backend/services/listener.py`
  - Active conversation microphone capture.
  - Uses `speech_recognition` with `stream=True`.
  - **Utterance endpointing is neural, not energy-based (S29 `acc366e`)**: `make_speech_endpoint` from `backend/services/neural_vad.py` (Silero on ONNX, hysteresis) owns speech start/end. `webrtcvad` survives only as `Vad(2)` inside `neural_vad.py` when the model is unavailable; `listener.py:119` still instantiates a coarse `Vad(1)` prefilter. Kill switch: `JARVIS_NEURAL_ENDPOINT=0`.
  - `JARVIS_STT_LANGUAGES` defaults to **`en-IN` only** (English-only since `8ae2294`; it used to be `en-IN,hi-IN`), and the local whisper daemon **now passes `language=` explicitly** (`whisper_daemon.TRANSCRIBE_LANGUAGE`, default `en`) instead of letting Whisper auto-detect. Auto-detect on noisy/echo audio was hallucinating random-language sentences, which then looked like the user speaking them.
  - Engine order comes from `transcription.recognize_multilingual`: exactly ONE engine per turn — the registry-selected `listening`-role engine (P0-03, landed). The live setting is `whisper`/`whisper-local`, so the local daemon answers. A failed/hallucinated result is a failed turn; there is no fall-through to another engine.
  - Commits a transcript only through the F34 stabilizer (final window) and the STT hallucination gate. Since tag `simple-listening` the partial-window layer is gone: one utterance costs exactly one transcription of the complete audio.
  - **Nothing blocking in the capture loop** (P1-03): the `/speak/stop` POST goes through the `_SpeakStopWorker` daemon thread and barge-in onset only notifies observers (`register_barge_in_hook`) on a daemon thread. The capture loop makes no transcription call at all. Do not add a synchronous HTTP call or a blocking engine call back into `_capture_audio`/`barge_in_on_speech_onset`. **`_SpeakStopWorker._deliver` re-dials a fresh socket when the keep-alive socket is dead (`_is_stale_socket`), so a barge-in stop is never lost (`9839fe8`); HTTP answers (401/403) are still not retried.**
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
  - Text-to-speech orchestrator. **Since S7 `066220c` the PRIMARY path is ONE live bidirectional Fish Audio WebSocket session per reply** (`fish_ws_begin_reply`/`fish_ws_session`/`fish_ws_end_reply` in `voice.py:13-21`, `_FishReplySession` in `fish_voice.py:93-94`, playback via `open_ws_reply`/`play_ws_reply` on the audio actor). `JARVIS_FISH_WS` is the kill switch (tests force it to 0 via `backend/tests/conftest.py`). The HTTP ladder below is now the **fallback**:
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
  - Fish Audio TTS client: registry `tts` role model resolution (env default `FISH_MODEL`, default `s2.1-pro-free`), PCM streaming playback, prefetch/warm-up, output-device selection. **Since S7 it also owns the live WebSocket reply session** (`fish_voice.py:67-69`, `open_ws_reply` `:230`, `play_ws_reply` `:302`); the older per-HTTP-POST path still exists but is no longer primary.

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
- Does NOT load a Whisper model in-process — `watcher.py:106` sets `whisper_model = None` because the model is hosted by the persistent daemon. Those decode settings now live in `whisper_daemon.py`: **medium** model (`:79`), cuda float16 / cpu int8 (`:145`/`:152`), `temperature=0.0` (`:246`), `beam_size=1` (`:255`), `vad_filter=True` (`:256`), `condition_on_previous_text=False` (`:257`), English-only (`:102`, applied `:249`); the wake-bias prompt (`watcher.py:843`, `:921`) is still wake-only via `X-Jarvis-Purpose: wake`.
- STT goes through `transcription.recognize_multilingual`'s registry-selected engine — **but there is NO ladder since S5 `c7b2128`**: `watcher.py:934-948` runs exactly ONE engine, ONE pass, and its output IS the transcript; `_selected_engine()` (`:952`) resolves it. Another provider is contacted only if it is itself the selected listening engine. The live setting is `whisper`/`whisper-local`, so the resident daemon answers. Every committed transcript passes the STT hallucination gate (a prompt echo must never false-launch the stack).
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

Much state is still process-local module state, but the durable artifacts are no longer "only three". On disk today: `data/jarvis_settings.json` (model selections, plus `jarvis_settings.json.bak-*` backups written by P1-08), `data/conversation_history.json` (created lazily — it may not exist on a fresh checkout), `data/jarvis_memory.db` (the G9 SQLite store, plus `-wal`/`-shm`), `data/change_journal/` (F22 code-grant file backups), `data/runtime/` (per-process instance stamps), `data/logs/` (bounded supervisor child logs), `data/chrome_profile_jarvis/` and `data/edge_profile_jarvis/` (browser profiles), `data/opencode_activity.log`, `data/opencode_server.log`, `data/research_reports/` and the productivity connector's grant JSON — all under `data/`, which is gitignored in full.

### Memory state

Location: `backend/core/memory.py`

- Stores recent chat history (capped at 20).
- Mirrored best-effort to `data/conversation_history.json`.

### Model selections (durable)

Location: `data/jarvis_settings.json`

- Persisted model-registry overrides for the `chat`, `tts`, `vision`, `browser_tool`, `listening`, `planner` and `intent` roles plus any user-added custom providers.
- Read per message/per call â€” changes take effect immediately, no restart.
- Since F56 the `intent` role is durable here too (`intent_model`): it is what the classifier's hop 1 resolves to, and deleting the key restores the shipped OpenRouter-first chain.

### Voice UI mirror state

Location: `backend/api/routes.py`

- `last_voice_message`, `last_voice_response`, `last_voice_log_id` — the latest voice exchange for the UI. **No frontend file reads `/voice-log` at all**; `/ui-state` is the fused voice-state + voice-log read (`routes.py:936-971`). Since S19 the mirror is not the only path: state changes also go out over `GET /events`.

### Voice runtime state

Location: `backend/listener_state.py`

- speaking/thinking/user-speaking booleans, active recognizer threshold, timestamps for user speech events, any stored remaining speech text.
- The voice process additionally owns the P0-08 turn manager (`voice_mode.TURNS`: one active turn, pre-emption statistics, `active_request_id`) and the P1-19 mark timeline it ships to the backend (`latency.LocalTurn`, one per capture, merged into the backend record under the same `request_id`). Both are process-local by design and reach the backend over HTTP, never by module import.

### Screen-control runtime state

Location: `backend/services/screen_state.py`

- whether screen controls are enabled; any pending plan waiting for confirmation.

### Pending task/confirmation gates

Location: `backend/core/brain.py` + `backend/services/task_agent/agent.py` + `backend/services/approvals.py`

- `_pending_opencode_task` / `_pending_confirmation` / `_pending_browser_clarification` (45s expiry windows, browser clarification 90s) plus **`_pending_task_action`** (`task_agent/agent.py:141`, with `has_pending_task_confirmation()`) — the arm-confirm-execute gates for task handoffs. Only one gate may be armed at a time.
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
- **S28 (`80c094e`) added real AEC alignment**: `DelayEstimator` cross-correlation lag lock, a joint last-5-frame compare window (`AEC_GATE_HISTORY_FRAMES`, `echo_cancel.py:108`), off-thread reference prefetch (`:109-111`), fp-exact overflow trim, and a `lag` block in `state()` now exposed by `GET /aec/state`. Tunable via eight `JARVIS_AEC_*` env vars (max lag, lag confirmations, gate history, prefetch, cache TTL, idle skip, breaker failures, breaker cooldown).
- The AEC frame cache is keyed by `(capture_token, index)` and cleared per capture (P0-13); `RemoteAecTransport` (the backend-spoken-audio reference path) authenticates, re-probes when idle, expires cached spans by `mic_t_end` drift, and reports a breaker/auth failure in `state()` instead of pretending the reference is merely absent (P1-04).

## Environment Variables and What They Actually Affect

Do not put real secret values into docs or prompts. The important thing is the variable names and their purpose.

### API keys (all optional individually â€” features degrade per provider)

- `GROQ_API_KEY` — Groq: intent-router last fallback classifier, an eligible screen-Q&A vision provider.
- `GEMINI_API_KEY` — Gemini: chat/vision env-default provider, research per-site notes, and the **live** `intent` hop 1.
- `FIREWORKS_API_KEY` — Fireworks: the ONLY env provider allowed for `planner`, and the browser-agent env default. NOTE: this account was suspended/failing on 2026-09-23 and again on 2026-10-03 (HTTP 412), which is what pushed chat onto Gemini and the others onto OpenRouter/a custom `fireworks-2` entry. **Do not hide Gemini failures behind Fireworks.**
- `FISH_API_KEY` — Fish Audio TTS (the metered primary spoken-voice engine; the `gtts` engine needs no key at all).
- `ELEVENLABS_API_KEY` — enables ElevenLabs TTS for short chunks.
- `OPENROUTER_API_KEY` — OpenRouter: a first-class provider for chat, vision and browser_tool (Cloudflare-fronted, so it stayed fast behind the VPN that broke direct Gemini).
- `INWORLD_STT_API_KEY` — Inworld STT, the alternative `listening` engine.
- `CLINE_API_KEY` — optional browser-agent provider.

### Model selection

- `GEMINI_BRAIN_MODEL` â€” Gemini chat/intent text model, default `gemini-3.5-flash-lite`.
- `GEMINI_VISION_MODEL` â€” Gemini vision model, default `gemini-3.5-flash-lite`.
- `GEMINI_INTENT_MODEL` â€” intent-router Gemini override.
- `GROQ_MODEL` â€” legacy Groq text default. The retired `llama-3.3-70b-versatile` is no longer referenced anywhere (H8).
- `JARVIS_WHISPER_PORT` â€” local Whisper daemon port, default `8767` (also read by `whisper_daemon.py`).
- `INWORLD_STT_MODEL` / `INWORLD_STT_URL` — Inworld STT model/endpoint for the `listening` role.
- `JARVIS_WHISPER_MODEL` — **local Whisper model size, default `medium`** (`whisper_daemon.py:79`, `watcher.py:799`; changed from `base` in `bedd9e0`). On CUDA medium decodes in ~1.5 s; on CPU-only it is far slower and `base` is the better setting.
- `JARVIS_WHISPER_LANGUAGE` — **transcription language, default `en`** (`whisper_daemon.py:102`, applied at `:249`; `watcher.py:803`). English-only was deliberate (`8ae2294`): Whisper auto-detect hallucinated random-language sentences on noisy/echo audio and they were then treated as user speech.
- `JARVIS_WHISPER_TRANSCRIBE_WAIT` — daemon transcribe wait, default `20` (`whisper_daemon.py:95`).
- `GROQ_VISION_MODEL` â€” Groq vision/Qwen default, `qwen/qwen3.6-27b`; also the intent fallback model source.
- `GROQ_INTENT_MODEL` â€” intent-router Groq override.
- `FIREWORKS_MODEL` â€” Fireworks default model, `accounts/fireworks/models/deepseek-v4-flash-0731` (used by the task-agent planner).
- `FIREWORKS_REASONING_EFFORT` â€” default `none`.
- `FISH_MODEL` â€” Fish TTS model, default `s2.1-pro-free`.
- `GROQ_STT_MODEL` / `GROQ_STT_URL` — Groq transcription model/endpoint.
- `INTENT_OPENROUTER_MODEL` — the shipped OpenRouter hop for the classifier, default `google/gemini-2.5-flash-lite` (`services/intent.py:42`).
- `GEMINI_THINKING_BUDGET` — Gemini thinking budget, default `auto` (`gemini_client.py:50`). **Set to `0`/off to stop paying for thinking; `gemini-3.5-flash-lite` 400s if you send an explicit `{"thinkingBudget": 0}`, so it is omitted entirely when the budget is 0 (`9839fe8`), and a 400 naming a bad argument replays once without `thinkingConfig`.**
- `OPENROUTER_REASONING_EFFORT` — default `low` (`model_registry.py:654`).
- `GROQ_REASONING_EFFORT` — default `none` (`grok_client.py:38`).
- `JARVIS_OLLAMA_KEEP_ALIVE` / `JARVIS_OLLAMA_WARMUP_TIMEOUT` — F56 local-model keep-alive and warmup (`ollama_client.py:21`, `:17`).

### Task engine

- `JARVIS_TASK_ENGINE` â€” `browser_agent` (default) or `opencode`.
- `JARVIS_BROWSER_AGENT_PROVIDER` â€” default `fireworks`.
- `JARVIS_BROWSER_AGENT_MODEL` â€” default `accounts/fireworks/models/qwen3p7-plus`.
- `JARVIS_BROWSER_AGENT_MAX_STEPS` â€” safety backstop, default `50`.
- `JARVIS_BROWSER_AGENT_TIMEOUT` â€” task timeout seconds, default `480`.
- `JARVIS_BROWSER_AGENT_REASONING_EFFORT`, `JARVIS_BROWSER_AGENT_KEEP_LAST_IMAGES`, `JARVIS_BROWSER_AGENT_LOOK_WIDTH`, `JARVIS_BROWSER_AGENT_JPEG_QUALITY` â€” payload knobs.
- `BRAVE_MCP_PORT` / `BRAVE_MCP_SERVER_DIR` — brave-control MCP daemon coordination.
- `JARVIS_OPENCODE_PORT` / `JARVIS_OPENCODE_CMD` / `JARVIS_OPENCODE_TIMEOUT` — opencode engine (only when selected).
- `JARVIS_TASK_MAX_STEPS` — task-agent step cap, default `24` (`services/task_agent/agent.py:73`).
- `JARVIS_OPENCODE_AUTO` (default `1`) / `JARVIS_OPENCODE_SERVE` (default `1`) — opencode auto-launch and serve mode (`opencode_client.py:43`, `:38`).
- `JARVIS_BRAVE_MCP_READY_TIMEOUT` — how long to wait for the MCP daemon (`opencode_client.py:102`).
- `JARVIS_PROJECT_DIR` — the project folder handed to a task (`opencode_client.py:522`).
- `JARVIS_BROWSER_PRIVILEGED_JS` — browser-agent JS policy gate (`browser_agent.py:57`).

### Research browser

- `JARVIS_RESEARCH_CHANNEL` â€” browser channel for research (`chrome` default; `msedge` used on this machine).
- `JARVIS_RESEARCH_PROFILE` â€” headed browser profile dir, default `data/chrome_profile_jarvis`.
- `JARVIS_RESEARCH_REPORTS` — deep-research report output dir.
- `JARVIS_RESEARCH_IDLE_TTL` — how long the shared research browser worker is retained before retiring (`services/research_browser.py:167`).

### Audio / microphone tuning

- `JARVIS_STT_LANGUAGES` — comma-separated speech-recognition languages, **default `en-IN` only** (English-only since `8ae2294`; it used to be `en-IN,hi-IN`). Read by both `listener.py:69` and `watcher.py:78`.
- `JARVIS_IDLE_ENERGY_THRESHOLD` — main listener idle threshold.
- `JARVIS_WATCHER_ENERGY_THRESHOLD` — watcher-only threshold.
- `JARVIS_MIC_DEVICE_INDEX` / `JARVIS_MIC_NAME` — microphone selection.
- `JARVIS_SPEECH_START_VAD_RATIO` / `JARVIS_FINAL_SPEECH_VAD_RATIO` — VAD ratios.
- **`JARVIS_PAUSE_THRESHOLD` — utterance-end pause, default `0.7` s** (`services/listener.py:55-56`; lowered from `1.2` in `bedd9e0`). This is env-overridable and is NOT a frozen constant — an earlier version of this map recorded the end-of-speech decision as frozen by an owner decision (P0-02 declined); that is no longer true.
- `JARVIS_MAX_ENERGY_THRESHOLD` — loud-room ceiling, default `700` (`backend/listener_state.py:8`).
- `JARVIS_STT_FINAL_TIMEOUT` — final-transcript budget, default `10` (`listener.py:287`).
- **S29 neural endpointing** (`services/neural_vad.py:26-29`): `JARVIS_NEURAL_ENDPOINT` (default `1`; `0` reverts to `webrtcvad`), `JARVIS_VAD_SPEECH_ON` (`0.10`), `JARVIS_VAD_SPEECH_OFF` (`0.20`), `JARVIS_VAD_THRESHOLD` (`0.5`).

### TTS tuning

- `JARVIS_PREFER_LOCAL_TTS` â€” legacy flag; Fish is tried first regardless, local SAPI5 is the fallback.
- `JARVIS_LOCAL_TTS_RATE` â€” `pyttsx3` speaking rate.
- `JARVIS_REMOTE_TTS_CHAR_LIMIT` â€” max text length eligible for ElevenLabs.
- `JARVIS_FISH_TTS_CHAR_LIMIT` â€” max chunk length for Fish.
- `JARVIS_FISH_TTS_VOLUME_BOOST_DB` â€” Fish playback gain.
- `JARVIS_TTS_OUTPUT_DEVICE` — Fish output device selection.
- **`JARVIS_FISH_WS` — S7 live-WebSocket session kill switch, default ON** (`services/fish_voice.py:78`, `:88`). `backend/tests/conftest.py` forces it to `0` for the whole test run so no test opens a real Fish socket.
- `JARVIS_GOOGLE_TTS_CHAR_LIMIT` — gTTS chunk limit, default `1800` (`services/voice.py:44`); `JARVIS_GOOGLE_TTS_SPEED` (`google_tts.py:140`).
- `FISH_REFERENCE_ID` — Fish reference clip for AEC (`config.py:44`).
- `JARVIS_EARCONS_ENABLED` / `JARVIS_REPLY_START_EARCON` — earcon toggles (`earcons.py:10`, `:20`).

### Voice runtime, AEC and wake detection (G10 / F31-F36)

- `JARVIS_AEC_ENABLED` â€” acoustic echo cancellation on/off, default on.
- `JARVIS_AEC_REMOTE` / `JARVIS_AEC_REMOTE_TIMEOUT` â€” let the voice process fetch the AEC reference from the API process over HTTP; this is required because the API process voices replies while the voice process owns the microphone. `JARVIS_AEC_REMOTE=0` is for an isolated local-only setup.
- `JARVIS_WAKE_ENGINE` â€” `auto` (default) / openwakeword / fuzzy matching.
- `JARVIS_WAKE_MODELS_DIR`, `JARVIS_WAKE_ONLINE_VERIFY`, `JARVIS_WAKE_PRE_ROLL_SECONDS` â€” wake-model directory, online whisper verification toggle, and how much audio is pre-rolled so the first syllable is not clipped.
- `JARVIS_WHISPER_MODE` — `conversation` (default) or wake-oriented local Whisper use.
- **S28 AEC tuning — all eight knobs** (`services/echo_cancel.py:68-110`): `JARVIS_AEC_MAX_LAG` (`0.4`), `JARVIS_AEC_LAG_CONFIRMATIONS` (`3`), `JARVIS_AEC_GATE_HISTORY` (`5`), `JARVIS_AEC_PREFETCH` (`2.0`), `JARVIS_AEC_CACHE_TTL` (`0.25`), `JARVIS_AEC_IDLE_SKIP` (`1.5`), `JARVIS_AEC_BREAKER_FAILURES` (`3`), `JARVIS_AEC_BREAKER_COOLDOWN` (`15`).
- `JARVIS_STT_CLOUD` — cloud STT toggle for the wake engine (`services/wake_engine.py:65`).

### Memory & continuity (G9)

- `JARVIS_MEMORY_ENABLED` â€” default on; `0` makes every memory call a safe no-op (an empty store changes zero prompts).
- `JARVIS_MEMORY_DB` — SQLite path, default `data/jarvis_memory.db`.
- `JARVIS_MEMORY_EVENT_RETENTION_DAYS` (default `30`) / `JARVIS_MEMORY_EVENT_MAX_ROWS` (default `5000`) — event-log pruning (`backend/core/memory_store.py:154`, `:157`; P1-16).

### Routing / orchestration

- `JARVIS_ORCHESTRATOR_MODE` — `legacy` (default) or `orchestrator`. In `orchestrator` mode the native tool-use loop is tried first and the legacy routing (plus every deterministic net) still runs on a decline. Orchestrator parity is NOT established, so do not default this to `orchestrator`.
- `JARVIS_INTENT_BUDGET_MS` (`3500`) / `JARVIS_INTENT_BUDGET_VOICE_MS` (`1200`) — classifier budgets, typed and voice (`backend/core/brain.py:817-818`).
- **`JARVIS_CHAT_FASTPATH` — default ON** (`brain.py:828`): lets definitely-plain chat skip the classifier network call entirely (S6).

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
- `JARVIS_WATCHER_CONTROL_PORT` — watcher HTTP control endpoint used by Electron stop (default `8766`, `watcher.py:163`).
- `JARVIS_DESKTOP_PATH` — desktop folder used by the voice worker (`voice_mode.py:945`).
- `JARVIS_CHROME_PATH` / `JARVIS_EDGE_PATH` / `JARVIS_BRAVE_PATH` — browser binary overrides (`core/executor.py:152`, `:161`, `:170`).
- `JARVIS_TOOL_COMMAND_TIMEOUT` / `JARVIS_TOOL_MAX_OUTPUT` / `JARVIS_TOOL_PAGE_CHARS` — native code-tool limits (`services/code_tools.py:74-75`, `:358`).
- `JARVIS_CDP_URL`, `JARVIS_EDITOR_BRIDGE_URL` / `JARVIS_EDITOR_BRIDGE_TOKEN` — connector endpoints.
- `JARVIS_PREWARM_INTERVAL` / `JARVIS_PREWARM_CONNECT_TIMEOUT` / `JARVIS_PREWARM_READ_TIMEOUT` — P0-12 pre-warm cadence and budgets (`services/prewarm.py:35-40`).

### Running the test suite

The suite is no longer 17 modules. `backend/tests/` now holds **131 `test_*.py` modules** (the F01-F56 audit suites, the G3-G11 suites, the C1-C3 and H1-H9 review suites, the 26 `test_p*_*` modules added by the two responsiveness waves, the 9 `test_s*_*` modules from the 2026-10 S-series, plus the original behaviour suites), and `tests/` holds 4 Node test files. The root `conftest.py` guards the developer's real `.env` against any test that would modify or delete it; **`backend/tests/conftest.py` additionally forces `JARVIS_FISH_WS=0` for the whole run so no test opens a real Fish socket.**

Run everything with the backend venv from the repo root (pytest is pinned in `requirements.txt`, and the root `conftest.py` is pytest-shaped):

```
& backend\venv\Scripts\python.exe -m pytest backend\tests -q
```

The original 17-module unittest invocation still names a useful fast subset and all 17 modules still exist — **but it no longer passes cleanly: at HEAD it reports `Ran 725 tests … FAILED (failures=5, errors=1)`** (e.g. `ERROR: test_stream_custom_provider_primary (test_model_registry.BrainChatModelResolutionTests)`). Use pytest for the authoritative run; treat the unittest path as a convenience subset, not a gate:

```
& backend\venv\Scripts\python.exe -m unittest backend.tests.test_code_tools backend.tests.test_task_agent backend.tests.test_brain_gate backend.tests.test_voice_task_mute backend.tests.test_opencode_lifecycle backend.tests.test_browser_agent backend.tests.test_screen_control backend.tests.test_voice_mode_toggle backend.tests.test_chat_race backend.tests.test_voice_latency backend.tests.test_model_registry backend.tests.test_settings_routes backend.tests.test_fireworks_reasoning backend.tests.test_model_roles_wiring backend.tests.test_live_bugfixes backend.tests.test_websearch_interruption backend.tests.test_websearch_modes
```

A single module runs as `python -m unittest backend.tests.test_screen_control`. The Node tests have no harness entry (`npm test` in `package.json` is still a stub) and are run directly. Note that two of them (`model-sidebar.test.js`, `research-overlay-href-scheme.test.js`) are renderer/DOM tests, not main-process policy layers:

```
node --test tests\backend-request-policy.test.js
node --test tests\model-sidebar.test.js
node --test tests\overlay-ipc-contract.test.js
node --test tests\research-overlay-href-scheme.test.js
```

On counts: the current collection is **2979 tests across 131 modules** (`pytest backend\tests --collect-only -q`). The last full-suite *pass* figure recorded in this map is **2802 passed, 5 failed, 1 skipped (2026-10-01, wave II)** and is necessarily stale — many commits and suites landed after it. Do not quote any older figure (512, 2507, or 2802) as current. Wall time varies run-to-run (roughly 15s–300s) — the variance is known and comes from a few unmocked live network calls in `test_websearch_modes`, not from flaky assertions.

**Known non-existent suites:** an earlier version of this map told you to disregard failures in `test_verification_uses_fireworks` and `test_verification_uses_groq`. **Neither file exists anywhere in the repo** — that advice was wrong. Real intermittent failures include `test_browser_agent.py`, `test_f02_goal_routing.py` (order-dependent, pass in isolation), `test_f37_groq_prerequisite`, the `test_g3_request_streaming` racer test, and the `live_bugfixes` vision tests. `test_task_agent.py::PlannerSafetyTests::test_model_plan_uses_fireworks_not_grok` is a **settings-dependent** failure: the persisted planner selection is now the custom `fireworks-2` provider, so the test's expectation no longer matches the live configuration. The reliable way to attribute a failure is to A/B it against a `git stash` of your own change rather than trusting a single run. Advice that survives all of that: run the full suite once per change, and confirm any new failure against HEAD before believing it is yours.

## Non-Obvious Behaviors Another Model Should Know

### 1. `backend/main.py` is the real backend entrypoint

`backend/app.py` exists, but the active launch commands use `backend.main:app`.

### 2. `grok_client.py` is named misleadingly — and its old default model is dead

It talks to Groq, not xAI Grok. Its former code default `llama-3.3-70b-versatile` is retired upstream (HTTP 404) and has now been removed from both the module and the offered catalog (H8); every live caller passes an explicit model (e.g. `qwen/qwen3.6-27b`). Never route new chat traffic through a bare Groq default.

### 3. The classifier can silently degrade — the deterministic nets exist because of it

`classify_intent` lands on a `chat` verdict whenever its whole chain fails or throttles, and it can genuinely misread screen questions as chat/research. **Since F56 hop 1 is the registry-selected `intent` role (live: `gemini/gemini-3.1-flash-lite`) and the OpenRouter -> Gemini -> Groq order is only the shipped FALLBACK chain; since S6 the router's JSON also carries a `reply` and the brain SPEAKS it, so a `chat` verdict no longer means "fall through to the chat model".** Four deterministic backstops in `process_message` rewrite or reroute such verdicts: the screen-question net, the fresh-info auto-search net, the all-`search`-steps -> research reroute, and the web-task routing net. Plus `JARVIS_CHAT_FASTPATH` (default on) short-circuits definitely-plain chat before any network call. When changing routing, check the nets, not just the classifier branch.

### 4. Voice-path defects: the three confirmed ones are FIXED (2026-09-30) — and five more changes landed since

The three defects that were confirmed present in the running code have all been fixed. This block is kept so a future session knows they were real, what the fixes are, and what to reach for when the voice path misbehaves again:

- **The AEC frame cache replaying the previous utterance's audio — FIXED** (P0-13, `deee60f`). `listener._capture_audio` restarted `frame_id` at 0 each capture while `_frame_cache` is a process-lifetime singleton, so from the second capture onward frames came back from the prior turn's cache — stale PCM plus stale `had_reference`/`suppressed`, never re-processed by the AEC. That was the cause of self-barge-in. Frames are now identified by `(capture_token, index)` from a process-wide counter, and `AecSignalPath.begin_capture()` clears the cache/order/last-id at the start of every capture. The within-capture dedup the cache exists for is deliberately preserved.
- **The remote AEC reference transport — FIXED** (P1-04, `f550642`). It sent no token (every fetch 401'd under fail-closed auth), its idle guard latched off permanently, its TTL cache ignored `mic_t_end`, and it had no breaker. All four are addressed; `state()` now distinguishes `auth_failed` / `circuit_open` from "no reference". It remains the BACKEND-spoken-audio path only — the local reference ring is still what serves voice turns.
- **Barge-in stopping audio but not the turn — FIXED** (P0-08, `6c89095`). `/speak/stop` only called `stop_speaking()`; the running request kept generating and the next utterance queued behind it. `POST /ask/cancel/{request_id}` now cancels that one request, the voice worker runs each turn on its own thread, and barge-in onset cancels the active turn. **The full generated reply still commits to history** — that is an owner decision, not an oversight; do not "fix" it into a spoken-prefix-only record.

Two things found while fixing these, worth knowing:

- The voice turn's request id was `voice-<epoch-ms>-<pid>` and **collided** when two turns were submitted inside one millisecond. Mostly theoretical before, but pre-emptive dispatch submits back-to-back and a reused id carrying a different message is a **409 conflict** at the request registry. It now carries a per-process counter, and `_TurnManager.start()` additionally pre-empts on a changed speaker so an id collision can never leave two turns registered.
- **The listener's blocking work is gone but the async replacements are only as good as their timeouts**: `_SpeakStopWorker` (P1-03) and the turn manager's cancel both post from daemon threads, so a hung endpoint costs a thread, not the capture loop. If you see barge-in latency, look at `speak_stop_stats()` and the `barge_in_stop{ok,ms,status}` latency mark before changing any VAD constant.

**Five later voice-path changes that this block originally did not mention (all 2026-10-02/03):**

- **S28 AEC alignment** (`80c094e`) — `echo_cancel.py` grew ~350 lines: `DelayEstimator` cross-correlation lag lock, a joint last-5-frame compare window, off-thread reference prefetch, and a `lag` block in `state()`. If barge-in suppression misbehaves, look here before touching VAD constants.
- **S29 neural endpointing** (`acc366e`) — utterance start/end is decided by `backend/services/neural_vad.py` (Silero on ONNX, 100 ms-on / 200 ms-off hysteresis), not by a fixed energy threshold. Kill switch `JARVIS_NEURAL_ENDPOINT=0`.
- **Whisper medium + 0.7 s pause** (`bedd9e0`) — `PAUSE_THRESHOLD_SECONDS` is **0.7** via `JARVIS_PAUSE_THRESHOLD`, down from 1.2. On CPU-only, set `JARVIS_WHISPER_MODEL=base`.
- **English-only transcription** (`8ae2294`) — `JARVIS_WHISPER_LANGUAGE=en`. Whisper no longer auto-detects, which is what stopped random-language hallucinations being treated as user speech.
- **Barge-in stale-socket recovery** (`9839fe8`) — `_SpeakStopWorker._deliver` re-dials when the keep-alive socket is dead, so a stop is never silently dropped.

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
- **an Anaconda `Library\bin` DLL search path at `backend/watcher.py:20` AND `backend/whisper_daemon.py:61`** — runtime-critical, since that is how CUDA is found
- `extract_pdf.py` paths

Note: there is **no hardcoded repo base path** — `config.py:9` derives `BASE_DIR` from `__file__`. The nearest literal is an `HTTP-Referer` hostname in `openrouter_client.py:103`.

### 9. Dependencies: `requirements.txt` is the proven set; `backend/requirements.txt` is a delegating shim

After H1, the root `requirements.txt` was regenerated by pip freeze from the working venv and now carries a header saying so — that IS the proven install set (numpy 1.26.4 ABI, pywinauto 0.6.9, the ~12 packages the old file omitted). **`backend/requirements.txt` is NOT a stale legacy list — it is a single line, `-r ../requirements.txt`, so installing from either path yields the same set.**

**Known gap:** the root file pins `onnxruntime==1.23.2` but has **no `silero-vad` entry**, while `backend/services/neural_vad.py:46` does `import silero_vad`. On a fresh install the default S29 endpointing path therefore **silently degrades to the `webrtcvad` judge**. Add the pin if you rebuild the environment.

### 10. The frontend is polling, not event-driven — but a push channel now exists for the voice worker

The renderer still polls one adaptive `/ui-state` endpoint (`renderer.js:569-610`, 120 ms active / 600 ms idle / backoff to 3 s), plus separate polls for the `/screen-answer` and `/research-result` overlays; `main.js` references `/events` nowhere. **But since S19 the backend exposes `GET /events`** (`routes.py:973`), an SSE channel fed by `backend/services/event_bus.py`, and the **voice worker already uses it** (`voice_mode._task_state_push_loop`). The renderer deliberately has not adopted it because a browser `EventSource` cannot attach the `X-Jarvis-Token` header.

So there are now **two** streaming paths: `/ask/stream` (the reply deltas) and `/events` (state push).

### 11. Model selection is live and persisted

`data/jarvis_settings.json` overrides env defaults for all **SEVEN** roles per message â€” reading only `.env`/`config.py` will give you the wrong picture of which model actually answers. The registry masks API keys; never log or echo them. `data/` is gitignored precisely because custom-provider keys live only there — never in `.env`, never in git.

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

- `backend/core/brain.py` (`_build_chat_messages`, `_stream_chat_deltas`, `_resolve_chat_model`, `_finalize_chat_reply`)
- **`backend/services/intent.py` — since S6 the chat ANSWER can originate here, via the router JSON's `reply` field. A prompt change that only edits the chat model will not be the whole story.**
- `backend/services/gemini_client.py`, `backend/services/fireworks_client.py`
- `backend/services/model_registry.py` (provider/model selection)
- The generic dispatch that carries openrouter/groq/custom-provider traffic: `backend/services/openai_compat_client.py`, `openrouter_client.py`, `grok_client.py`, `ollama_client.py`

### Change intent routing or the safety nets

Start in:

- `backend/services/intent.py` (classifier prompt + chain — **hop 1 is the registry-selected `intent` role** (live `gemini/gemini-3.1-flash-lite`), then the shipped fallback chain OpenRouter → Gemini direct → Groq, sharing one monotonic budget via `_budget_timeout`; the module docstring's "Gemini first" description is stale). **Since S6 this file also owns the chat reply text.**
- `backend/core/brain.py` (nets in `process_message`, `_ChatRacer`, the `JARVIS_CHAT_FASTPATH` fast path)
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
- **`backend/services/neural_vad.py` — since S29 this owns the utterance-END decision that `listener.py` consults. It was missing from this list, which is why endpointing changes were hard to place.**
- `backend/listener_state.py` (`selected_stt_engine()` — the S5 single-engine resolution)
- `backend/whisper_daemon.py` (model size / language pins)
- `backend/voice_mode.py` (the I/O worker; it must never import `backend.core.brain`)
- `backend/services/audio_actor.py` (the single playback owner), `echo_cancel.py` (AEC + S28 lag alignment), `transcript_stabilizer.py` (F34)
- `backend/services/intelligence_state.py` (worker generations, playback ownership, event journal)
- `backend/services/event_bus.py` (S19 state push)

**Measure before you change anything here.** `/latency` (P1-19) already renders the whole turn as an ordered waterfall with p50/p90/max per step, sorted slowest-first, and it stitches the voice process's capture/STT/TTS marks into the backend's record under one `request_id`. If you are chasing latency, read that first: the marks (`speech_end`, `capture_end`, `stt_start`, `stt_done{engine}`, `http_in`, `racer_start`, `classify_done{hop}`, `first_token`, `provider_headers`, `tts_first_byte`, `playback_started`, `barge_in_stop{ok,ms,status}`) tell you which stage is actually slow instead of inviting a guess. `speak_stop_stats()` covers the P1-03 stop worker, and `voice_mode.TURNS.snapshot()` covers turn pre-emption.

Two hard rules that have already bitten this area: nothing blocking may enter `_capture_audio` or `barge_in_on_speech_onset` (they run on the real-time capture thread), and `MAX_PHRASE_SECONDS` / `LISTEN_TIMEOUT_SECONDS` are frozen by an explicit owner decision (P0-02, declined).

**`PAUSE_THRESHOLD_SECONDS` is NOT frozen.** An earlier version of this map listed it alongside those; that is obsolete. It is now `float(os.getenv("JARVIS_PAUSE_THRESHOLD", "0.7"))` — 0.7 s since `bedd9e0`, down from 1.2. And the **end-of-speech decision itself is no longer a fixed energy threshold at all** — S29 moved it to Silero-on-ONNX with hysteresis in `neural_vad.py`. Change endpointing there, not in `listener.py`.

### Change TTS or spoken reply behavior

Start in:

- `backend/services/voice.py` (**since S7 the primary path is ONE live Fish WebSocket session per reply** — `fish_ws_begin_reply`/`fish_ws_session`/`fish_ws_end_reply`; the HTTP ladder below it is the fallback)
- `backend/services/fish_voice.py` (Fish PCM streaming **and** the S7 WebSocket session; kill switch `JARVIS_FISH_WS`)
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

- `backend/tests/` — 131 unittest modules; per-feature suites are named `test_fNN_*` (Fable-5), `test_gNN_*` (G-groups), `test_cN_*` (review criticals), `test_hN_*` (review highs), `test_pNN_*` (responsiveness waves), `test_sNN_*` (the 2026-10 S-series), plus 28 unnumbered feature suites (including `test_custom_provider_ui.py`, `test_stt_hallucination_gate.py`, `test_simple_listening.py`, `test_voice_task_mute.py`, `test_live_chat_and_bargein_fixes.py`)
- `tests/` — Node `node --test` suites (two are renderer/DOM tests, not main-process policy)
- `conftest.py` (root) — the `.env` session guard; `backend/tests/conftest.py` — forces `JARVIS_FISH_WS=0`

Do not quote a stale total from an older copy of this map. An older version claimed two suites were "already failing": `test_verification_uses_fireworks` and `test_verification_uses_groq`. **Neither file exists in this repository** — disregard that claim entirely. The genuine settings-dependent failure is `test_task_agent.py::PlannerSafetyTests::test_model_plan_uses_fireworks_not_grok`, because the persisted planner selection is now the custom `fireworks-2` provider.

## Minimal File Map

Top-level directories and files that matter:

- `frontend/`
  - Electron renderer UI: `index.html`, `renderer.js`, `style.css`, plus the overlay renderers (`overlay_renderer.js`, `overlay_images_renderer.js`, `research_overlay_renderer.js`), the capsule (`capsule.html`, `capsule_renderer.js`, `capsule.css`, `capsule_main.js`), `preload.js`, and the `snapshot-v3.html` design snapshot
- `backend/`
  - Python backend (`main.py`, `api/routes.py`, `core/`, `services/`), `voice_mode.py`, `watcher.py`, `whisper_daemon.py`, `listener_state.py`, `scripts/tail_activity.ps1`
- `main.js`
  - Electron main process: window factory, IPC policy, managed-child supervision
- `run_jarvis.bat`
  - full-stack launcher (now delegates to `backend.watcher --launch`)
- `run_watcher.bat`
  - wake-word-only launcher
- `run_jarvis_noisy.bat` / `run_watcher_noisy.bat`
  - the same two launchers with loud-room mic thresholds preset
- `tests/`
  - Node (`node --test`) suites; `backend-request-policy` and `overlay-ipc-contract` cover main-process policy, while `model-sidebar` and `research-overlay-href-scheme` are renderer/DOM tests. `npm test` is still a stub — run them directly.
- `integrations/brave-control/`
  - vendored copy of the brave-control MCP server (`server.mjs` + `lib/`)
- `integrations/jarvis-editor-bridge/`
  - VS Code-compatible editor bridge extension
- `PROJECT_MAP.md`
  - this map
- `BROWSER_AGENT_AUDIT_BACKLOG.md`
  - **the accepted 2026-10 browser-agent audit backlog** (36 items `BA-00`–`BA-35`): per-item problem, evidence, solution, acceptance criteria, invariants, a 10-case benchmark suite, and a section listing the five claims that were investigated and **dismissed as false**. Approved by the owner; implementation is underway commit-per-audit — `BA-00` (`ff86fc1`), `BA-02`/L-8 (`f3855bd`), L-6 (pooled session), L-1/`BA-05` (staleness hint) and L-15/`BA-03` (split timeouts), L-16/`BA-15` (auto-scroll click) and L-5/`BA-14` (event-driven wait) are landed; L-14/`BA-06` (retrying probe helper), I-2/`BA-07` (phantom-tool prose) and L-18/`BA-08` (async logging) are landed; the remaining backlog items are pending.
- `codex.md`
  - a change log (newest entries at the bottom) covering the 2026-08/09 history this map summarises. **It is NOT running: it is ~228 lines and its newest entry is the 2026-09-24 STT-hallucination fix.** The 2026-09-30/10-01 waves and the whole S-series are not in it — use the sections below instead.
- `CODE_REVIEW_REPORT.txt` / `.pdf`
  - the 2026-09-23 independent code review (55 findings) this map's remediation section tracks
- `screen_commands.log`
  - persistent audit log of all screen control actions and execution steps
- `conftest.py`
  - pytest session guard for the real `.env`
- `data/` (gitignored, all of it)
  - `jarvis_settings.json` (+ `*.bak-*`), `jarvis_memory.db`, `change_journal/`, `runtime/`, `logs/`, `chrome_profile_jarvis/`, `edge_profile_jarvis/`, `research_reports/`, `opencode_activity.log`, `opencode_server.log`. Note `conversation_history.json` is created lazily and may be absent on a fresh checkout.

Mostly non-runtime or secondary:

- `graphify-out/`
  - generated architecture artifacts
- `.audit_tmp/`, `_plan.txt`, `_repro*.py`, `_r3.txt`–`_r5.txt`, `.git-broken-20260914-154720/`
  - scratch/repro/backup leftovers from the audit work; not part of the runtime

## Responsiveness Audit State (2026-09-29 triage — 27 items implemented across two waves)

A responsiveness audit produced 43 findings. The owner triaged every item. **27 items are now implemented** — 13 in the 2026-09-30 wave (`f37a519`…`6c89095`) and 14 in the 2026-10-01 wave (`2e85ebb`…`3b460d0`, each tagged per item). See "Responsiveness audit implementation state" below for commits and substance.

**Two honesty notes about this section:**

1. **The triage document no longer exists.** Older revisions of this map pointed at `AUDIT_IMPLEMENTATION_PROMPTS.html` for the per-item decisions and pasteable prompts. **That file was never committed to this repository** — it is absent from the working tree and from the entire git history. The owner's per-item decisions are therefore **not recoverable from the repo**; what survives is the standing-decision list below, the commit messages, and the per-item test suites.
2. **The stated id range does not add up.** `P0-01…P0-13` is 13 ids and `P1-01…P1-19` is 19 — **32 ids, not 43**, and there is no `P2-*` id anywhere. So roughly 11 of the original 43 findings were never given an id that survives in any file here. Do not assume the id list is complete.

**The owner's standing decisions:**

- **P0-01 (slow Gemini over VPN) — no action.** The VPN was turned off, so the measured 7–45 s figure no longer applies. **Do not change the chat provider.** Re-measure once P1-19 telemetry exists. (Note: the audit's cited model `gemini-3.8-flash` was stale; the live selection is `gemini-3.5-flash-lite`.)
- **P0-02 (1.2 s energy-only end-of-speech) — DECLINED at the time, but SINCE SUPERSEDED. This entry is the one standing decision that no longer holds.** It originally read "no change may be made to `PAUSE_THRESHOLD_SECONDS`, the end-of-speech decision, `MAX_PHRASE_SECONDS`, or `LISTEN_TIMEOUT_SECONDS`." As of `bedd9e0` and `acc366e` that is no longer true: `PAUSE_THRESHOLD_SECONDS` is **0.7 s** (env `JARVIS_PAUSE_THRESHOLD`) and the end-of-speech decision moved to Silero neural VAD in `backend/services/neural_vad.py`. `MAX_PHRASE_SECONDS` and `LISTEN_TIMEOUT_SECONDS` remain untouched. **Do not cite P0-02 as a reason to avoid touching endpointing.**
- **P0-03 (serial STT ladder) — implement, but as ONE engine only. DONE, and reworked by S5 `c7b2128`.** The cross-engine fallback chain and the Google/Groq language loop are gone; `listener.py:443-476` and `watcher.py:934-948` each run exactly one engine whose output is final.
- **P0-08 (barge-in does not cancel the backend turn) — implement, with one carve-out. IMPLEMENTED (`6c89095`).** The full generated reply **stays committed to history** after an interruption, because the owner wants to read what was missed. The audit's "store only the spoken prefix" suggestion is explicitly rejected; an additive `last_reply_interrupted` flag is the substitute. **Do not reverse the carve-out** — the reply text is untouched by design.
- **P0-09 (classifier gating) — suggestion only, do not implement yet.** The short version is that a local model via the existing Ollama client is the recommended candidate, used as a local-first / cloud-fallback cascade rather than a replacement, and only after a labelled evaluation set exists.
- **P0-07 — implement only if it measurably helps. IMPLEMENTED (`6d1bedc`) after the gate fired.** It was split into a measurement gate, an `abort()` fix to do regardless, and a persistent-output-stream refactor to do only if the gate justified it; the gate measured a 400 ms inter-sentence gap on a Bluetooth default device (and a 221 ms `stop()` drain), so **both** parts were done. If you change the device setup, re-run that gate before trusting the persistent stream.
- **P1-17 (wake cold-start) and P1-18 (Whisper daemon settings) — SKIPPED.**

**Landed: 27 of 43.** Wave I (13): P0-04, P0-05, P0-06, P0-07, P0-08, P0-13, P1-01, P1-02, P1-03, P1-04, P1-05, P1-09, P1-19. Wave II (14): P0-03, P0-10, P0-11, P0-12, P1-06, P1-07, P1-08, P1-10, P1-11, P1-12, P1-13, P1-14, P1-15, P1-16.

**Still open: none of the previously-listed items.** An earlier version of this map listed P0-03, P0-10…P0-12 and P1-06…P1-08, P1-10…P1-16 as "still pending" — **all 14 landed in wave II**, 120 lines further down in the same document. The only genuinely unresolved entries are P0-01 (no action), P0-09 (suggestion only), and P1-17/P1-18 (skipped).

**One wave-I entry is superseded and should be read with care:** P0-04's partial-transcription layer (`_PartialWorker`, `test_p0_04_async_partials.py`) was **deleted outright** in `7afefc0` as part of the simple-listening change. Its measurement (capture loop 2.45 s → 0.05 s) was real at the time, but the code it describes no longer exists.

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
- FastAPI exposes `/ask`, `/ask/stream`, `/ask/status/{id}`, `/ask/cancel/{request_id}`, **`/events` (SSE state push)**, `/voice-log`, `/voice-mode`, `/voice-state` + `/voice-state/publish`, `/ui-state`, `/speak/stop|pause|resume|remaining`, `/latency` + `/latency/client`, `/prewarm` + `/prewarm/stats`, `/aec/state|reference`, `/screen-answer`, `/research-result`, `/research-progress`, `/settings*` + `/providers/{provider_id}/models` + `/settings/provider/test`, `/task/stop`, `/approvals/reset`, `/voice-setup/launch`, `/health`, `/update-voice-log`. Everything except `/health` needs the per-launch `X-Jarvis-Token` (fails closed; `JARVIS_DEV_MODE=1` is the only bypass).
- `backend/core/brain.py` is the central intent router. Memory phrases run first; then **route selection plus a speculative chat racer (moved above the task/code-tool/screen/research chain in P0-10)**; then explicit task/code-tool handoffs, screen control, explicit research; then (if `JARVIS_ORCHESTRATOR_MODE=orchestrator`) the native tool-use orchestrator; then the legacy path: a speculative chat stream races the classifier (hop 1 = the registry-selected `intent` role, currently `gemini/gemini-3.1-flash-lite`; shipped fallback chain OpenRouter Flash Lite → Gemini Flash Lite → Groq Qwen → `chat` verdict, all hops sharing one deadline), and four deterministic backstops (screen-question net, fresh-info auto-search, all-search-steps → research reroute, web-task routing) correct classifier misfires. **Since S6 the router's JSON carries a `reply` and the brain speaks it, so a normal chat turn is ONE LLM call; `JARVIS_CHAT_FASTPATH` can also skip the network entirely for plain chat.**
- Chat is served by the model-registry-selected provider (currently `gemini/gemini-3.5-flash-lite`) with a same-model non-stream retry and a Gemini → Fireworks tail — NOT Groq, whose old default model is retired. A terminal auth/validation failure refuses instead of substituting another model.
- Web lookups default to the Brave AI-Overview quick-search tier; "deepsearch" adds the multi-site research service with a glass-overlay report. Both tiers share one long-lived Playwright worker.
- Screen Q&A ("What's on my screen?") runs the F37 eligible-provider vision cascade (registry selection first, bounded fallbacks, no credential-free dispatch) and shows floating desktop overlays.
- `command ...` messages trigger browser actions or launch local Windows applications (via the `launch_app` resolver in executor, no shell fallback) with web URL fallback.
- Multi-step web tasks hand off to the confirmation-gated browser agent (brave-control MCP daemon, 50-step/480s backstops); the opencode CLI is an opt-in engine selected by capability, and the executor choice is frozen in an F16 contract at consent time.
- Voice mode is a pure I/O worker: it captures and transcribes (**one STT engine per utterance, endpointing by Silero neural VAD**), submits to `/ask/stream` with `speak=False`, publishes its listening state, receives task state by **push over `/events`**, and owns playback through **one live Fish Audio WebSocket session per reply**, with the HTTP ladder (Fish → local SAPI5 → ElevenLabs → local SAPI5) as the fallback. The backend remains the single intelligence authority. **Since S18 it keeps accepting conversation while a task runs, queueing action requests rather than refusing them.**
- The watcher is a separate passive wake-word launcher running local GPU-accelerated Whisper (**medium**, English-only) via the resident `whisper_daemon` (port bound before model load), with the STT hallucination gate vetoing prompt echoes and **no provider fallback ladder since S5**. It also supervises the full stack in `run_jarvis.bat` mode and distinguishes warm sleep from full shutdown.
- Screen control is a distinct subsystem that combines direct command parsing, native UI Automation tree extraction, visual OCR text extraction, vision-model planning, and Windows input automation.
- All screen control actions are logged with full details in `screen_commands.log`.
- State is mostly in-memory module state; the durable artifacts are `data/jarvis_settings.json` (revision 61, seven role selections), `data/jarvis_memory.db`, the browser profiles, and the research reports. Conversation history is mirrored best-effort by a debounced background writer (`backend/core/memory.py`) and flushed on shutdown.

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

### Simple listening: partials removed, Whisper + greedy decode (simple-listening, 2026-10-01; model default revised twice since)

Owner-requested latency pass on the speech-to-text path, made after **measuring**
the real cost on the target laptop (RTX 3050 6 GB + i5-13420H), one 7.24 s English
utterance, using the daemon's own transcribe flags:

These are the numbers that justified choosing a model size **at the time**. Note the column labels are now HISTORICAL: `base` was the new default in this pass, but the owner immediately moved it back to `medium` (`bedd9e0`), which is the shipped default today (`JARVIS_WHISPER_MODEL`). On CPU-only, `base` is still the better setting — medium cpu int8 is 3.82× realtime.

| configuration | model load | decode | speed |
| --- | --- | --- | --- |
| medium cuda float16 | 3.66 s | 1.53 s | 0.21× realtime |
| base cuda float16 (the default chosen in this pass, since revised back to medium) | 0.53 s | 0.42 s | 0.06× realtime |
| tiny cuda float16 | 0.37 s | 0.37 s | 0.05× realtime |
| medium cpu int8 (the fallback path) | 6.92 s | 27.65 s | 3.82× realtime |

The perceived "speech → text" gap was three things stacked, and only one of them
was STT: the fixed **pause threshold** (then 1.2 s, since reduced to 0.7 s; `PAUSE_THRESHOLD_SECONDS`, paid on
every turn before STT even starts), the decode itself, and **queueing behind the
live partials** (they shared the daemon's single `_lock` with the final request).
Three changes, all reversible:

1. **The partial-window layer is GONE** (`backend/services/listener.py`, −468
   lines). Removed: `_PartialWorker`/`_partial_worker`/`_submit_partial`,
   `_emit_partial_window`/`_transcribe_partial`,
   `register_partial_observer`/`unregister_partial_observer`/`_notify_partial`,
   `probe_whisper_daemon`/`whisper_daemon_ready`/`partial_daemon_state`,
   `partial_worker_stats`, `_bounded_audio_tail`, all six `PARTIAL_*` env
   constants, the per-frame trailing-silence accounting, and the early-commit
   path (two agreeing partial windows + 250 ms of silence ending a capture
   early). `_capture_audio` takes no `early` out-parameter and transcribes
   nothing; `listen()` is now simply *capture the whole utterance, then ask
   exactly ONE engine to transcribe it*. This deliberately reverses P0-04's
   "partials are actually used" behaviour above (that work stays in history at
   `e491c40`). The reason: with a warm GPU the partials' only latency win was
   the early commit, while their cost was a second consumer of the daemon's one
   `_lock`, so the final transcription could sit behind an in-flight partial
   (0–1.2 s, felt as random jitter). Accuracy improves too — the committed text
   is now always the FULL-utterance transcription, never a trailing-window
   (≤6 s) approximation of it.
   - **Deliberately KEPT:** AEC + echo-only rejection, onset/barge-in (now the
      only per-frame decision), the 0.7 s pause threshold (reduced from 1.2 s by
      the 2026-10 owner request; `JARVIS_PAUSE_THRESHOLD` overrides), the
      human-voice gate,
     the capture-complete earcon, the `speech_end`/`capture_end`/`stt_start`/
     `stt_done` marks, `LAST_STT_ENGINE`/`LAST_STT_FAILURE`, one engine per turn
     (P0-03) and the F34 final-window commit through `_turn_stabilizer`
     (`begin_turn()` per capture is also what scopes the AEC frame token,
     P0-13). Do **not** add a blocking engine call back into `_capture_audio`.
2. **Model default — REVERTED to `medium`** (`backend/whisper_daemon.py::MODEL_SIZE`,
   and `watcher.WHISPER_MODEL_SIZE` for the in-process fallback). The latency
   pass had switched it to `base`; the 2026-10 owner request moved it back,
   because base mangled Hindi/Hinglish, names and numbers. Still one env
   var — `JARVIS_WHISPER_MODEL=base` (or `small`) restores the faster, weaker
   behaviour, and both models are already cached on disk. On the owner's
   question of whether dropping multilingual support would help: **the language
   list was never the latency cost** (it is read only by the `google-or-groq`
   engine, which is not the selected one on this machine —
   `data/jarvis_settings.json` → `listening = whisper/whisper-local`). As of the
   2026-10 owner request the daemon is **English-only** (`TRANSCRIBE_LANGUAGE`,
   `JARVIS_WHISPER_LANGUAGE`, default `en`), because whisper's auto-detect
   hallucinated whole sentences in random languages on noisy/echo audio; the
   recognition language list defaults to `en-IN`. Model size is the lever that
   moves the clock.
3. **Greedy decode**: `beam_size=1` added to the daemon's
   `model.transcribe(...)` — faster-whisper defaults to a beam of 5 plus a
   re-rank, measured at roughly 1.5–2× the decode cost. Deterministic at
   `temperature=0.0`; delete the line to restore beam search.

**Tests:** `test_p0_04_async_partials.py` was **deleted** (its entire subject no
longer exists) and the partial-only contracts were removed from
`test_f34_whisper_conversation.py` (its docstring records exactly why the audit's
first acceptance clause is no longer pinned), `test_latency_reductions.py`,
`test_p0_03_single_stt_engine.py`, `test_p0_13_aec_frame_cache.py`,
`test_p1_19_latency_waterfall.py` and `test_stt_hallucination_gate.py`.
`test_f55_whisper_boot_independence.py` also had to be made hermetic: three of
its `/transcribe` tests let the REAL `load_model` run in-process and passed only
because `medium` was slow enough to lose the race — with `base` the real model
was handed a fake `RIFFxxxx` body and answered 500, so the loader is now stubbed
explicitly (they pin the WAITING behaviour, not how fast a model loads). New
`test_simple_listening.py` pins the replacement contract: one utterance → exactly
one engine call over the whole audio, no engine call inside the capture loop, no
early exit, no `early` parameter, and **the removed API staying removed** (a
symbol list that fails the moment the layer is reintroduced), plus the shipped
`base` default. `test_stt_hallucination_gate.py` gained a pin that conversation
transcription decodes with `beam_size=1` at temperature 0.

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

## Recent Improvements (2026-10)

The S-series latency/intelligence wave. This is the newest work in the repo and the largest architectural change since F56. **Every one of these commits was previously absent from this map.**

### S19 — state pushed over one event channel instead of polled (`a0ec729`)

The voice worker used to ASK the backend "is a task running?" and "is voice enabled?" on a 1 s-cached poll, so a change landed up to a second late and the listener could hold off or mute wrongly. S19 pushes each change the moment it happens.

- **`backend/services/event_bus.py`** (new) — dependency-free thread-safe pub/sub: `subscribe`/`unsubscribe`/`publish`/`snapshot`, bounded queue (64, drops OLDEST so a wedged reader cannot grow unbounded), optional cross-thread `wakeup`, and a `reset()` test seam. A late or reconnecting subscriber gets a merged snapshot replayed first.
- **`GET /events`** (SSE) — one channel carrying `snapshot` then live `task_running` / `voice_enabled` / `voice_state`, with a 15 s `: ping` heartbeat. It is an **async generator that holds no thread-pool slot**, so unlike the chat streams it cannot starve barge-in.
- **Emitters** — `brain.set_opencode_task_running`, `listener_state.set_voice_input_enabled`, and `routes.publish_voice_state` → `_emit_voice_activity` (change-detected, so the 1 s heartbeat never floods the channel).
- **`voice_mode.py`** — `_apply_pushed_state` folds pushed truth into the two poll caches (refreshing their TTLs); `_task_state_push_loop` reconnects with backoff. **The 1 s polls stay as the fallback** if the channel drops.
- The Electron renderer can adopt `/events` in place of its `/ui-state` poll; that swap is deliberately left to the UI layer (a browser `EventSource` cannot send the auth header). Tests: `backend/tests/test_s19_event_channel.py` (17).

### English-only whisper transcription (`8ae2294`)

`whisper_daemon.TRANSCRIBE_LANGUAGE` (default `en`, override `JARVIS_WHISPER_LANGUAGE`) is now passed as `language=` to `model.transcribe`; `watcher.WHISPER_LANGUAGE` applies the same pin on the in-process fallback. `RECOGNITION_LANGUAGES` defaults to `("en-IN",)` in both `watcher.py` and `services/listener.py`.

Root cause of the reported "Telugu" transcripts: Whisper **auto-detect** hallucinating random-language sentences on noisy/echo audio, not translation. The brain then answered them as user speech.

### Whisper medium + a 0.7 s utterance-end pause (`bedd9e0`)

`whisper_daemon.MODEL_SIZE` and `watcher.WHISPER_MODEL_SIZE` default `base`→**`medium`**; `listener.PAUSE_THRESHOLD_SECONDS` default `1.2`→**`0.7`**. Caveat: medium is fine on CUDA (~1.5 s decode) but much slower CPU-only — `JARVIS_WHISPER_MODEL=base` reverts. **This supersedes the P0-02 "frozen 1.2 s" standing decision above.**

### Live chat 400s + a barge-in stop lost to a stale socket (`9839fe8`)

- `_thinking_config()` now returns `None` (omitting `thinkingConfig`) when the budget is 0 — `gemini-3.5-flash-lite` rejects an explicit `{"thinkingBudget": 0}` with 400 INVALID_ARGUMENT. On a 400 naming a bad argument, `ask_gemini_chat` replays once without it.
- `_SpeakStopWorker._deliver` re-dials a fresh socket immediately when the keep-alive socket is dead (`_is_stale_socket`), so a barge-in stop is never lost. HTTP answers (401/403) are still not retried.
- Tests: `backend/tests/test_live_chat_and_bargein_fixes.py` (12).

### S6 — one LLM call classifies the turn and answers it (`1fd635a`)

The router's JSON now carries a `reply`, and the brain speaks it (`handle_chat(..., answered=router_reply)`). **A normal chat turn costs one LLM call instead of two.** Paired with `JARVIS_CHAT_FASTPATH` (default on), which lets definitely-plain chat skip the network entirely via `is_definitely_plain_chat`. Tests: `test_s06_router_answers_chat.py`.

### S5 — one STT engine per utterance, its output final (`c7b2128`)

`listener.py:443-476` and `watcher.py:934-948` each run exactly ONE engine, ONE pass; a failure sets `LAST_STT_FAILURE` and there is no fall-through. This is the final shape of P0-03.

### S7 — one live Fish WebSocket session per reply (`066220c`)

`fish_voice.py` gained `_FishReplySession`, `open_ws_reply` and `play_ws_reply`: one bidirectional Fish session per reply instead of an HTTP POST per chunk. The HTTP ladder survives as the fallback; `JARVIS_FISH_WS` is the kill switch and `backend/tests/conftest.py` forces it to 0 for the whole test run. Tests: `test_s07_fish_ws.py` (324 lines).

### S18 — conversation continues during a task; actions queue (`66bacbc`)

**The full voice mute is gone.** Committed utterances are submitted *while a task runs*; the backend answers conversationally and queues action requests (max 3 in flight) instead of refusing them. This relaxes the single-voice rule described in the 2026-08 entry above. `test_voice_task_mute.py` was rewritten.

### S13 — delivered background results join the conversation history (`2ce627f`)

A delivered async result becomes part of the history rather than an orphan spoken fragment. Tests: `test_s13_background_history.py`.

### S29 — neural-VAD-driven utterance endpointing (`acc366e`)

**NEW `backend/services/neural_vad.py`** (227 lines): Silero on ONNX with 100 ms-on / 200 ms-off hysteresis, wired into `listener.py` (`make_speech_endpoint`). `webrtcvad` survives only as the no-model fallback judge. Kill switch `JARVIS_NEURAL_ENDPOINT=0`. **This is the second supersession of the P0-02 end-of-speech standing decision.** Tests: `test_s29_neural_endpointing.py`.

### S28 — AEC alignment: lag estimation, joint window, off-thread prefetch (`80c094e`)

`echo_cancel.py` grew ~350 lines: `DelayEstimator` cross-correlation lag lock, a joint last-5-frame compare window (`AEC_GATE_HISTORY_FRAMES`), off-thread reference prefetch, fp-exact overflow trim, and a `lag` block in `state()` now surfaced by `GET /aec/state`. Tunable via eight `JARVIS_AEC_*` vars. Tests: `test_s28_aec_alignment.py`.

### S1 + S12 + S23 + S26 + S27 — voice-path latency & clarity batch (`8d89dae`)

One commit carrying five audit items: a whole-word gate on memory search; a spoken-style `voice_compact` prompt with a minute-precision time note; a smoothed TTS output limiter (removes ~6 dB per-chunk jumps); pooled keep-alive sessions for fireworks/grok/gemini with prewarm warming the clients' own pools; and explicit reasoning controls (Gemini `thinkingConfig`, Groq `reasoning_effort=none`, OpenRouter via the registry snapshot). Tests: `test_s27_reasoning_controls.py`.

### F56 — local Ollama models as selectable brains (`f004e36`, `cfd1f7f`)

Tag `local-models`. Documented in full at the top of this map ("Local models as selectable brains"). `cfd1f7f` additionally fixes the active checkmark in the INTENT CLASSIFIER MODEL list.

### Astra R13 — strict speech gateway: tool-less chat has zero action authority (`8837f3c`)

`backend/core/brain.py`: `_ACTION_CLAIM_RE` / `_ACTION_OFFER_RE` detect action sentences; `_CHAT_NO_ACTION_FALLBACK` ("Understood, sir. Nothing was started — which exact folder and file name should I use?") replaces any unverified claim via `_strip_unverified_action_claims`, applied in `_finalize_chat_reply`; the chat prompt's "say you will get it done" instruction removed; `_orchestrator_reply` ANSWERED gated. Task/result narrators untouched. Tests: `R13SpeechGatewayTests` in `test_brain_gate.py`.

### Astra R12 — status answered from live state, never chat recall (`f5ee444`)

`is_status_question` / `_status_snapshot` (armed confirmations, running flag, S18 queue, clarification, last verified result) / `answer_status_question` (Astra §4 controlled forms; running+queued names both; "cued"→"queued" only in status; "q test" asks instead of denying). Status turns skip all three confirmation gates so "is it done?" never discards the preview it asks about. Tests: `R12StatusGroundingTests` in `test_brain_gate.py`.

### Astra R5 — local folder inspection is a local read, never browser (uncommitted at map time)

`backend/services/task_agent/agent.py`: `_INSPECT_VERB_RE` + `_INSPECT_FOLDER_RE` + `_local_inspect_folder` ("check whether folder Malik exists", "have a quick look at that folder", "see what is inside" → `code.list_directory`); checked FIRST in `is_code_tool_request` and `_heuristic_plan` (before the web "look up" branch, which now yields to local targets); bare "navigate to <folder>" no longer browser-shaped without URL tokens. "Create a directory listing" stays a create, not an inspect. Tests: `R5LocalInspectTests` in `test_code_tools.py`.

### Astra R6 — stop-then-redirect: control first, redirect held behind quiescence (uncommitted at map time)

`backend/core/brain.py`: `STOP_RESEARCH_PHRASES` gains the browser-task stop phrases (previously no brain text route); `split_stop_and_redirect` splits "stop X and do Y" (status turns never split); `handle_stop_then_redirect` signals the worker NOW — nothing running → honest "no browser task is running" + redirect runs immediately (Trace A); running → redirect held in `_held_redirect`, reply "held until it stops" (Trace B), released by `_release_held_redirect` on a later quiescent turn; `_record_last_work_request` / `_resolve_last_command` track the last real work (skips status/yes-no/stop) so "execute the last command" resolves to the user's file request. Tests: `R6StopThenRedirectTests` in `test_brain_gate.py`.

### Astra R4 — whole-sentence yes check: tails decide (uncommitted at map time)

`backend/services/task_agent/agent.py`: `classify_confirmation` returns yes/no/rename/inspect/extra/unclear — "yes, don't create it" declines, "yes, but call it X" re-previews under the new name via `_repreview_with_name` + `arm_task_confirmation` (old yes dead), "yes, a quick look" HOLDS the write and asks "create, or only check?", extras never inherit, "did I say yes?" is a question not assent. `consume_task_confirmation` and brain `_consume_opencode_confirmation` both use it (opencode inspect re-arms so a clear "create" next turn approves). Tests: `R4ConfirmationVerdictTests` in `test_code_tools.py`, `R4OpencodeGateTests` in `test_brain_gate.py`.

### Astra R1 — meaning, not first word (`450472c`)

Declarative/desire phrasing routes deterministically: `_WANT_CREATE_RE` catches "I want / I need / there should be ... file" ANYWHERE in the sentence ("on my desktop there is a folder Mayank Malik, I want a file inside it"), while `_WANT_ASSERT_ONLY_RE` keeps bare existence ("there is a file") conversational. Both `is_code_tool_request` and `_heuristic_plan` handle the R1 shape (located-write plan with defaulted name). Tests: `R1MeaningNotFirstWordTests` in `test_code_tools.py`.

### Astra R2 — notebook across turns (`7e99bda`)

`backend/core/brain.py`: bounded in-memory ledger (`_notebook_requests` max 20, `_notebook_entities` max 12) — `notebook_record_request` (id req-N, kind, state seen→…) fed by every `_record_last_work_request`, `notebook_mark_state`, `notebook_record_entity` upsert with focus-head order, `notebook_focus_folder`, `notebook_snapshot` (requests + entities + live `_status_snapshot`). It is a READ model over the existing globals, not a replacement. `_resolve_folder_hint` in `task_agent/agent.py` reads the focus head first ("that folder" = active-task folder); `code_tools.create_folder` / `list_directory` record observed entities. Tests: `R2NotebookTests` in `test_brain_gate.py`.

### Astra R14 — chat has no action voice (`e97bbed`)
`backend/core/brain.py`: `_ACTION_CLAIM_RE` extended to the optimistic pre-execution acks ("On it", "Playing/Opening/Checking/Searching/Navigating ... now", "Back in a moment", "I've opened ... for you"); `generate_command_response` now only NAMES the request ("{song}, sir." — completion comes from the result narrator); `handle_research_intent` acks name the request ("Quick lookup for that, sir."); `browser_search` chat path reports the fact ("I ran a search for that."); crash path no longer claims "I started the task"; `_orchestrator_reply` gates suspension/error planner text too; `_ChatRacer.adopt` gates at sentence boundaries (byte-identical passthrough when nothing cut, held-claim redelivery at stream end) so live speech and stored text agree. Tests: `R14NoActionVoiceTests` in `test_brain_gate.py` + updated `test_browser_search_prebuilt_path` expectation in `test_chat_race.py`. **Follow-up (`pending`): live pieces + completed-sentence head were both emitted by `adopt()._drain`, duplicating the reply's first word ("LoudLoud and clear") in every streamed voice/UI reply; an `emitted` offset now flushes only the not-yet-spoken tail, and the sentinel emits the un-emitted remainder (a cut claim still redelivers its gated replacement once).**

### Astra R3 — all slots before acting (`6d31829`)

`backend/services/task_agent/agent.py`: `_explicit_file_name` ("name it New Zealand" → `New Zealand.txt`; "name it anything/whatever" → the spoken default, never a vague plan); both located-write branches (R1 declarative + imperative) bind exact parent path + frozen name, ask ONE bundled question when content is missing (R11-shaped: "before I create that I still need: …"), and build via `_code_tool_plan` which binds `create_only=True` so an approved create can never silently clobber a file that appeared after the preview; `_confirmation_preview` speaks the exact target ("Ready to create new file … (refusing if it already exists)"). Tests: `R3AllSlotsBeforeActTests` in `test_code_tools.py`.

### Astra R7 — correction replaces, never adds (`8909df4`)

`backend/core/brain.py`: `is_correction` ("that was meant/supposed to be", "I meant check…", "actually just look…", "correction:") + `handle_correction` runs BEFORE the confirmation gates so the old preview can never eat the revision as "unclear"; `_cancel_armed_gates` kills every armed approval (native via new `cancel_pending_task_confirmation` in `task_agent/agent.py`, opencode, research, shared record, browser clarification) so the old yes dies; the superseded notebook record is marked; a check-shaped correction executes read-only immediately ("understood — checking instead"), anything else returns an explicit replace ack. Tests: `R7CorrectionReplacesTests` in `test_brain_gate.py`.

### Astra R10 — that/it/there/last/queued resolve from the notebook (`76d56e6`)

`backend/core/brain.py`: one resolver family over the R2 ledger — `resolve_that_folder`/`resolve_there` (focus head; "there" is always a place, never a file/page), `resolve_bare_it` (folder / ask-once when file+folder both fit / none), `resolve_last_command` (last non-superseded notebook request — skips own/yes/stop/status by construction), `resolve_queued_task` (oldest S18 entry / held R6 redirect / armed approval / "Nothing is queued"); `_resolve_folder_hint` in `task_agent/agent.py` now calls the shared `resolve_that_folder`; the R12 queue-status branch speaks the REAL entry ("Queued: … — not started") instead of an invented count. Tests: `R10ResolveFromNotebookTests` in `test_brain_gate.py`.

### Astra R8 — one turn can be two jobs (`132ba68`)

`backend/core/brain.py`: `split_compound_turn` (status-half must be a real status question, work-half a non-trivial action sentence; corrections/confirmations/bare status never split) + a role-detector block in `_process_message_inner` AFTER held-redirect release and correction but BEFORE the confirmation gates — the status half is answered from live state NOW (`answer_status_question`), the work half re-enters `_process_message_inner` as a FRESH turn (its own preview/approval, nothing inherited), and the two replies are joined. Stop+redirect compounds keep the dedicated control-first R6 path; this covers status+work. Tests: `R8OneTurnTwoJobsTests` in `test_brain_gate.py`.

### Astra R9 — confirmations survive STT noise (`4036720`)

`backend/services/task_agent/agent.py`: `_stt_confirmation_verdict` hypotheses run BEFORE the R4 word-match classifier — "yes yesterday, no now" declines (latest word wins), a dropped leading yes ("s, a quick look") still holds as inspect, "did I/you say yes/no/ok?" stays a question; the ORIGINAL text is never rewritten, only the verdict changes. `_describe_native_step` now says the emptiness out loud ("Ready to create new file … (empty)", full path always shown). Tests: `R9SttSafeConfirmationTests` in `test_code_tools.py`.

### Astra R15 — fix names carefully (`37153a1`)

`backend/services/task_agent/agent.py`: `_norm_folder_token` (case/space-folded lookup) + `_folder_name_candidates` (exact→prefix→containment over the real folders, on-disk canonical paths only) + `resolve_folder_name` (exact / candidates / none); `_resolve_folder_hint` resolves named folders through it — ambiguity and no-match return None so the caller asks once instead of guessing; `pre_execution_recheck` re-verifies JUST before a confirmed write runs (parent still exists, grant still covers it, file must not have appeared — any change stops, never renames/overwrites). `backend/core/brain.py`: `answer_exact_vs_candidate` ("no exact folder named Malik — but Mayank Malik exists. Shall I use it?"). The cued→queued repair stays status-only, never filenames. Tests: `R15CarefulNameMatchingTests` in `test_code_tools.py`.

### Astra R11 — ask once, then stop (`pending`)

`backend/services/task_agent/agent.py`: `_clarify_attempts` counter keyed by request text (`_clarify_key/note/count/reset`, max 2 asks) — missing-slot bundles (`_slot_complete_write_plan`), unknown-folder asks (`_folder_hint_clarification`), unclear answers and inspect-holds in `consume_task_confirmation` all count under the same request; the third ask returns `_clarify_stop_line` ("still not sure … stopping rather than guessing … Nothing was started") instead of another question. The inspect hold keeps the exact write plan aside in `_held_inspect_plan` (NOT armed — no yes can fire it): a clear "create" next turn re-arms the SAME previewed effect, "check it" runs read-only (`list_directory`, never the write). `cancel_pending_task_confirmation` clears the hold + the ask count so corrections start fresh. Tests in `test_code_tools.py` (`LocatedWriteTests` additions).

### Astra R16 — one approval, one run (`pending`)

`backend/services/task_agent/agent.py`: the 45s pending window + atomic `approvals.take()` ARE the expiry and one-shot rules — a late yes finds nothing armed ("no longer available — yes lasts one run only"), a replayed yes finds the taken record gone, a changed plan fails `verify`; all three re-ask. `execute_plan(confirmed=True)` WITHOUT the taken record but WITH a live pending record (replayed yes, stale queued write, direct prod call) re-arms-and-asks instead of running; harness plans (no pending record) keep the old shortcut. `consume_task_confirmation` passes its taken `approval` record into `execute_plan` so the gate-blessed run is distinguishable from a stray call. Nothing is persisted — restart clears every armed approval by construction. Test: `test_replayed_yes_without_approval_reasks` in `test_code_tools.py`.

### Astra R17 — both routes, one gate (`pending`)

Gate-blessed runs flow `consume_task_confirmation → execute_plan(approval=record) → loop`; the orchestrator's `task.propose` arms through the same `_arm_plan_confirmation`/`approvals` record and its PROPOSAL reply uses the same `confirmation_prompt`. `backend/core/brain.py::_orchestrator_reply` refuses to voice a PROPOSAL unless `has_pending_task_confirmation()` is live ("could not arm that action … nothing was started"). No settings flag bypasses arm/verify/speech/cancel. Test update: `test_consent_executes_the_very_plan_that_was_proposed` fake accepts `approval`.

### Astra R18 — know what to refuse (`pending`)

`backend/services/task_agent/agent.py::should_refuse` — the one deliberate refusal list: vague "navigate there" (no target), "browser stopped" claims with no worker ack, undo-a-done-write corrections (history is not rewritten; removal needs its own explicit ask), candidate/none folder names (never pick, never invent), stale verification (>10min old check proves nothing now). `_heuristic_plan` calls it BEFORE any planning — a refusal is a completed honest stop, never a guessed plan. Related: non-existent named folders no longer auto-resolve to `Desktop\<name>` in `_resolve_folder_hint` (returns None → ask/refuse). Tests: `test_refusals_stop_honestly` in `test_code_tools.py`.

### R19 — every task engine leaves a short history summary

`backend/core/brain.py`: `_remember_tool_summary(kind, summary)` — one bounded (240-char, word-cut) in-band `[<kind>]` assistant entry, deduped against the immediately-preceding entry (the S13 stacking guard); a write failure never breaks the turn. `_record_native_task_outcome` now emits `[task] <request> — <outcome headline>` for every terminal native run (request capped at 100 chars, outcome at 160 so the shared cap can never truncate away what happened), so local file/folder work is recallable ("what file did you create?") instead of living only in work events. Browser/opencode/research terminal results already join history as `[background result]` via S13. Tests: `ToolHistoryTests` in `test_f09_native_capture.py` (8).

### R20 — intent shape decides the route, not the sentence opening

Live transcript: "jarvis, can you create a folder on desktop by the name history and inside that folder can you create a txt file … write hello" never reached the task path — every create/write matcher was anchored to a leading verb and only R1's "I want…" form escaped, so chat looped on "I can create …"; and "check if there is a folder by the name Mayank Malik" lost the name to the bare-word "folder" pronoun shortcut and was answered "Which folder should I create that file in". `backend/services/task_agent/agent.py`: `_CODE_TOOL_FILLER_RE` now strips question auxiliaries ("can/could/would/will you") so a wrapped request is the same effect; NEW `_intent_flat` + `_folder_file_create_plan` — a folder-create clause anywhere plus a file-create clause that references it ("inside that folder" / "inside it") becomes ONE two-step plan (create_folder then create_only write_file, one confirmation; a capability question with no name never becomes a plan); `_inspect_named_target` resolves "folder by the name X" / "folder named X" / "is there a folder X" BEFORE the pronoun shortcut, and `_folder_hint_clarification(..., verb="check")` answers a check with the honest absence (or names the real candidate — R15) instead of write wording. Tests: `R20IntentCreateAndCheckTests` in `test_code_tools.py` (9) + `test_question_wrapped_create_routes_to_task_message` in `test_brain_gate.py`.

### EX1 — understand 'this one', 'it', 'that file' (pending)

Execution-brain Rank 1. New `backend/core/entity_ledger.py` (stdlib-only, no import cycles): an upsert ledger of every object Jarvis has seen or created (id, kind, display name, canonical ref, provenance + trust weight, confidence, salience, version, mention/focus timestamps, 30-entry cap) plus `resolve_mention` — demonstrative ('that file'), definite-description ('the file you just made'), explicit quoted/named, and bare-pronoun mentions bind by deterministic scoring (name 0.4 / recency 0.3 / salience + provenance trust, kind-substitute + hard-constraint gates, path-liveness factor; bind at >= 0.5, near-tie asks once naming both options). Rejections are negative evidence (10-minute TTL) so a rejected entity is never offered again. `backend/core/brain.py`: `notebook_record_entity` mirrors every entity into the ledger (one funnel, no drift); new `_REFERENCE_CORRECTION_RE` + `handle_reference_correction` pre-route gate before R7 and the confirmation gates — 'no, not that one' / 'I meant the other one' cancels armed approvals, rejects the last-bound entity, re-resolves the raw mention text, and confirms with the new name or asks for the exact name; R7 check/create phrases are excluded and stay on their path. `backend/services/code_tools.py`: `write_file` now records created files (create_folder/list_directory already did). Tests: `backend/tests/test_entity_ledger.py` (17: ledger, resolver, correction).

### EX2 — one utterance, many jobs (pending)

Execution-brain Rank 2. New `backend/services/multi_intent.py` (pure, stdlib-only): `build_chain` splits a compound request on conjunction boundaries (including Hinglish `aur`/`phir`), classifies each piece (screen / research / task / tool), merges fragments into the clause they continue and same-kind neighbours into one step, then accepts a chain only when >= 2 distinct kinds survive, no `tool` clause is present, and any file-write (`task`) clause is LAST. Reference cues (`about it`, `whatever you find`, `write your report in it`, `save it on desktop`) set each step's `consumes` index so output flows forward: screen observation → research query → file content. `render_ack` speaks ONE narrative ("I'll do this in 3 steps … I'll ask before creating the file"). `backend/core/brain.py`: `handle_multi_intent` + `_run_multi_intent_chain` run the steps on a worker thread — screen via the existing `analyze_screen` (shared `_screen_qa_busy` lock; topic recorded as a ledger entity), research synchronously via `run_quick_search`/`run_research` (deep reports still push the overlay), and the final write is only ARMED through the normal task confirmation (`_arm_plan_confirmation` + `confirmation_prompt`) — nothing is written without the user's yes, and one armed approval at a time is respected. Failures are honest and local: a failed step skips only its dependents ("I skipped the research step because the screen step didn't finish") and the one summary says exactly where it stands. The new pre-classifier gate in `_process_message_inner` sits after every confirmation/correction/stop/status gate and before the racer/classifier, so single jobs (build_chain returns None) and explicit `command ` turns are untouched. Tests: `backend/tests/test_multi_intent.py` (17: splitter, executor with fakes, gate routing).

### EX3 — try, check, try smarter (pending)

Execution-brain Rank 3. New `backend/services/adaptive_steps.py` (pure, stdlib-only): `diagnose` reads a failed step's own evidence (its result text + structured payload) and names the cause (transient / overlay / missing target / login wall / existing target / permission / missing parent), `tactics_for` returns the DIFFERENT approaches that fit the tool, in order — a tactic runs at most once per step, so an identical retry is structurally impossible — plus `simplify_query` (a failed web search reruns with filler stripped; refuses when nothing would change), `attempt_signature` (canonical identity of each try) and the hard caps `MAX_TRIES_PER_STEP = 3`, `MAX_REPLANS_PER_JOB = 2`. `backend/services/task_agent/agent.py`: every step in the confirmed loop now runs `_execute_step_adaptive` — do → check (`_step_failure_verdict`, the exact evidence rules the loop always used, extracted unchanged) → diagnose → recovery runner. Runners never send new effects: `resettle_wait` (a beat, then fresh evidence), `refresh_evidence` (re-snapshot tabs/windows/editor), `alternate_route` (simplified search query, or activate the already-open tab for the URL), `dismiss_overlay` (one Escape, only while the foreground window is still the one the plan was built against — if focus moved it refuses). Non-recoverable evidence stops immediately and honestly: a sign-in wall is never retried (Jarvis never signs in for you), an existing file is never retried (never overwritten), permission denials and externally-visible productivity effects are never retried. A write whose parent folder is missing is a PLAN problem, not a retry problem: within the replan budget the folder becomes an explicit `code.create_folder` step (later `depends_on` indices re-mapped) and the changed effect set is re-approved through the normal gate — the old yes never authorizes an un-previewed step. Exhausted steps report the trail ("… (after 3 tries)") in the outcome, the evidence and the spoken fragment. Tests: `backend/tests/test_adaptive_steps.py` (24: diagnosis, tactics, no-identical-retry, hard caps, replan, honest report).

### EX4 — prove it before claiming it (pending)

Execution-brain Rank 4. New `backend/services/proof.py` (pure, stdlib-only): `prove_step(tool, args, structured)` verifies the postcondition of a step that reported success and returns `{"state": "proved"|"failed", "evidence": …}` — `code.write_file` must exist on disk AND hash-match what was written (the content arg, or the tool's own `after_hash`; a name-only claim only proves non-empty presence), `code.create_folder` must exist as a directory, `code.apply_patch` must leave a present file; tools without a proof rule return None and keep today's behaviour. `backend/services/task_agent/agent.py`: `_step_failure_verdict` now fails a lying write inside the adaptive loop ("postcondition failed: <proof>", diagnosed under the existing evidence rules), and the confirmed-step loop computes each proof BEFORE publishing an ok outcome — a failed proof becomes a failed step carrying the proof as its reason/fragment, while a proved one joins `result.verification` ("file content verified on disk: <path> (sha256 …)"), so "Done, sir." can only be spoken for effects the filesystem confirms. `backend/services/browser_agent.py`: media-playback goals (play/watch wording) may finish `completed` ONLY with a `verify_playing: PLAYING` verdict — the verdict is recorded on the task session, and a PAUSED / UNCERTAIN / STATIC / never-probed play goal downgrades to `partial` with the honest hedge ("I pressed play, but I could not confirm it started …") instead of a false done. Tests: `backend/tests/test_proof_layer.py` (16: module rules, verdict integration, completion gate, media gate).

### EX5 — stop means stop (pending)

Execution-brain Rank 5. `backend/core/brain.py`: a bare typed stop ("stop", "stop it", "ruko") and the broad "stop everything" (both negation-guarded) now have their own gate — after the dedicated research/browser phrases, before the confirmation gates, so a stop is never eaten as an unclear answer to an armed preview. `handle_stop_message` cuts speech (`voice.stop_speaking` + narration off), cancels exactly the watched work job (`jobs.request_stop` — never the chat request, never older background work) or every work job for "stop everything", and speaks "Stopped, sir." ONLY when the cancelled work is observably still (`_stop_targets_still`: research flag, browser quiescence, cancelled jobs finished killing) — otherwise "Stopping, sir." now and an 8-second quiescence watcher delivers the truthful async report ("still moving" when it cannot confirm). A stopped browser run publishes what it had already done (`browser_agent._publish_stop_report` / `consume_stop_report`, from the session's completed actions), so the stop reply reads "Stopped, sir. I had already clicked Play; opened youtube.com. Nothing is moving now." Voice: the new `task_stop_all` control ("stop everything", "stop all tasks") posts `/task/stop?scope=all`, the routes' `stop_task` scope=all cancels every non-request job, and both voice dispatch entry points route the new kind. Tests: `backend/tests/test_stop_protocol.py` (16: phrases, scopes, quiescence, report, voice grammar/dispatch, gate routing).

### REDESIGN R1 — capture once, let the eyes resolve (external redesign RANK 1)

New `backend/services/context_state.py` (stdlib-only, RLock): `Item` / `Observation` dataclasses + `ObservationStore` — every screen look keeps its RAM-only image (newest 2, 5-min TTL), metadata (newest 3, 15-min TTL), the item inventory and the rejected set; `OBSERVATIONS` singleton. `backend/services/screen_analyzer.py`: `capture_observation(mode)` (mode explicit, never inferred from question text; dHash for same-screen detection), `identify_on_screen(utterance, target_text, …)` — ONE vision call whose prompt appends an item-inventory job (up to 6 items: EXACT displayed label, truncation flag, kind, creator, bbox, primary) and a pointer-resolution job to the F48 analysis prompt — plus `_clean_items` (labels sanitized as data), `grade_target` (report 1.6 post-validation: unknown id → none; truncated/described label → capped at likely; exact without why → likely; scaffolded label or label == utterance → none; a user-named word missing from the chosen label → likely; 2+ plausible items flagged ambiguous instead of silently binding), `reidentify` (re-asks on the SAME stored image; rejected items stay rejected) and `get_observation`. `analyze_screen` is now a wrapper over the same engine: it stores an observation on every screen Q&A (`observation_id` in the result), maps `topic`/`creator` from the PRIMARY inventory item (exact label) falling back to the model's topic/creator, and keeps F48 evidence/provenance, the external-check path and the F37 cascade/error reporting unchanged. `backend/core/brain.py`: `_mi_screen_step` returns `output_obs_id`; the chain's research step calls `_mi_vision_screen_target(clause, obs_id)` — when the screen step stored an observation the SAME image is re-used (no re-capture; a Shorts feed that advanced cannot swap the subject) and the vision model resolves the pointer against the inventory; only an exact, single-fit, non-ambiguous binding searches silently, a likely binding asks "is this what you mean?", and with no stored observation the legacy focused-vision path + text-extractor fallback are untouched; a pending screen clarify now carries `obs_id`, so a correction ("no not that, the image to the left") re-asks on the same picture. A possessive back-reference ("research about their release dates") is an ASPECT of the already-identified subject, not a new target: `_mi_subject_aspect_query` composes the observation's non-truncated primary label with the aspect words the user added (`_mi_aspect_tokens` strips instruction tails like "and tell me what you find"), so the stateless second vision call — which has no antecedent for "their" — is never asked and the chain stops answering "I couldn't tell exactly what to search for"; truncated/generic/absent primaries keep the existing ask flow. Tests: `backend/tests/test_screen_observation.py` (34: store caps/TTLs, cleaner, grader rules, identify/reidentify/analyze mapping, chain wiring, possessive-aspect composition).

### LIVE FIX 8 — full answers, aspect-aware search

Two live defects from the same chain replay. (1) The research step's chat fragment was hard-sliced `summary[:200]` (mid-word: "He claims O"), while the chain report was clipped at 600 — the spoken summary is condensed for voice, not short for reading, so the report now carries it up to 900 and the final reply up to 1400 (the repeat-echo path 600), with `_mi_clip`'s word-boundary "..." as the only cut. (2) The search query was only the identified label/thumbnail title: "what is this video about dimensions on my screen, research about it and tell me" searched 'YOU IN 7D?' alone and answered about the unrelated Disney show. The aspect of a deictic request ("...about dimensions, research it") lives in the SCREEN clause, not the research clause, so `_mi_aspect_suffix(clause, subject, query, obs_id)` appends aspect words from the research clause first and the stored observation's own utterance second (deduped against both the subject and the query) — but never when the vision or text resolver already carried the user's words (`used_text_extractor`), when the creator branch overrides the subject (`creator_override`), or when a possessive compose already ran (`aspect_composed`); `_mi_aspect_tokens`/`_mi_clause_specific_tokens` now strip verb inflections ("researching") and bare "on internet", and the stop sets absorb filler adverbs/quantifiers ("very", "some") and look/pointer words ("look", "at", "my"). Verified live replays: PewDiePie utterance keeps the exact label (no filler appended), "dimensions" composes to 'YOU IN 7D? dimensions', possessive composes to '... release dates' with no screen-clause leakage. Tests: `test_screen_observation.py` +6 (deictic aspect composition, full-fragment, uncut report, suffix rules).

### LIVE FIX 9 — the model writes the search phrase; the answer-ask reaches the screen

RANK 10, model side. `backend/services/screen_analyzer.py`: `compose_search_query(observation, request_text, fallback)` sends the query-writer job sheet the target description the eyes already produced — exact item labels, kinds, creators, primary/truncated flags, the vision description and the screen-clause utterance stored on the observation — plus the user's request, and asks for ONE search phrase (keep names exactly, merge the question terms, never invent, never answer, no instruction words). Validation `_clean_composed_query` takes the first line, strips quotes, rejects scaffolding ("research", "on my screen", "please"…), caps 180; any failure (provider down, junk answer, no items/answer) returns the caller's deterministic query. `backend/core/brain.py` `_mi_research_step`: the deterministic subject+aspect compose remains as the model's input hint and fallback; the model's phrase replaces it when valid (live probe: "The Menu movie review is it good" instead of the raw thumbnail title); `creator_override` still wins and the model is skipped without a stored observation. KBC misroute fixed in the same pass: "give me the answer to this KBC question on my screen" has a screen reference and an ask but no wh-word, so `_SCREEN_QUESTION_CUES` now includes answer/question/solve/quiz — both the chat fast path and the screen-question net route it to screen analysis instead of chat. Tests: `ComposeSearchQueryTests` (model answer used, prompt carries label+guess, scaffolding rejected, provider failure/no-items fallback), `test_model_composed_query_is_used`, KBC cue tests in `test_screen_analyzer.py` + `test_latency_reductions.py`.

### R20 — the conversation remembers its own work (memory/reference fixes)

Live diagnosis (history dump): chain/tool requests, the folder-name answers and the "confirm" answers never entered history — `add_message("user", …)` lived only on the chat and screen-QA paths — so history held dangling assistant notes with no request above them; the chain's step findings (the identified player, the research summary) were never written anywhere; and the task path that must resolve "that folder" read no history at all. Fixes: (1) every consumed action turn commits its user half — the chain gate (plus its ack), the folder-name gate, both confirmation gates, and all five `handle_task_message` gates (`_remember_user_turn`); (2) every finished chain step leaves a compact R19 note (`[screen] I looked at your screen: <label>`, `[research] I searched "<q>" — …`, `[task] <preview/fragment>`), and the latest real output is stored TTL'd (`_set_chain_findings`) for the later folder-name turn; (3) `consume_pending_folder_name` answers against the STORED clause (the name injected at its first folder mention — an appended tail lands past the file mention where the folder-name search never looks), so the file half of the original request is no longer dropped, and its file content falls back to the chain findings when the clause content is a pointer ("write the information you found"); (4) `_mi_folder_name` now reads "by/with/in the name X" (folder half only — the file half's "by the name info" can never name the folder) with pointer names ("of this website") left to the findings extraction, and "by" joins the name-cut words; (5) `build_chain` expands a screen clause carrying research markers ("find out who this player on my screen is by researching on the internet, and create a folder…") into screen+research even when more clauses follow — the research half of the request no longer silently dies; (6) `MAX_HISTORY` 20 → 40 (each chain now adds its user half + step notes, so 20 slots held ~7 real exchanges); (7) the task agent sees the conversation: `gather_context()` carries the last 12 turns (240 chars each), `_build_planner_prompt` renders them as a RECENT CONVERSATION block, and `_resolve_folder_references` deterministically substitutes "in that folder"/"that folder you made" with the most recent "Folder ready: <path>" from history (only when the path exists). Tests: `test_reference_memory.py` (20) + the parametrized cap in `test_chat_race.py`.

### LIVE FIX 10 — the file lands in the EXISTING folder, under the name the user gave it

Live replay: "research … and store all the information you find in a txt file inside information folder on my desktop" → "the folder is already there on the desktop by the name information , create a txt file inside that folder…" → "name it random.txt" created a NEW folder `random` with `info.txt` in it, and the chain read the full research summary aloud though the user only asked to store it. Root causes and fixes, all in `backend/core/brain.py`: (1) positional folder names — "inside information folder" names the folder by position (the name rides before the folder word, past the file mention where the phrase search never looks), so `_mi_folder_name` = `_mi_folder_name_phrase` (the old head-restricted "named/called/by the name" search) OR `_mi_folder_name_positional` (`in/inside/into/on/at/under the|my|our <1–3 words> folder|directory`; articles, pointers, grammar and file words name nothing); (2) inline answers — `consume_pending_folder_name` also consumes a full clarification ("the folder is already there … by the name information, create a txt file inside that folder…") as the answer (phrase-named folder + folder word required; the incoming text becomes the clause, carrying the file half), so the ask can no longer sit stale while the turn falls into chat; (3) the existing folder is a LOCATION, never a duplicate — `_FOLDER_EXISTS_RE` ("already there/exists/created", "existing/pre-existing folder"): an existing folder hosting the file is reused as-is (no "information (2)", no create step — the plan is the file write only), and "already there" with nothing to put inside asks what to create; (4) the FILE gets its own ask — `_FILE_NAME_RE`/`_mi_file_name` (explicit "file named X", stopped at comma/and/inside/…), and when the findings will be written but the file is unnamed the task step ASKS for the file name (never guesses "info.txt"), remembered in `_pending_file_name` so the new `consume_pending_file_name` gate (wired after the folder gate in `process_message`; answer regex name it/call it/file name is, plus inline full requests; the name injected at the FIRST file mention; both turns committed to history) lets "name it random.txt" name the FILE inside the resolved folder; (5) the chain report never reads the findings aloud — when any task step of the chain stores them in a file (`stores_findings`), the research step's fragment is "The findings are ready for your file." (the full findings still reach the file write, `_set_chain_findings`, and a one-line history note, so "remind me what we just did" no longer recites them). Tests: `test_reference_memory.py` +11 (positional names, articles/pointers rejected, existing-folder reuse, the file-name ask, the file-name answer, the exact three-turn live replay, the saved-not-spoken report).
