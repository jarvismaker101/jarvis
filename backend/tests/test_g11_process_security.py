"""G11 — Process Architecture & Security (audit F50 / F51 / F52).

Covers:

* F51  local-auth token semantics: mint/arm/check, open paths, and full
  enforcement through the FastAPI middleware (401 without the token, live
  /health liveness stays open).
* F51  backend identity surface: /health exposes instance/build/protocol.
* F50  voice I/O worker decoupling: the worker attaches the token on its
  backend calls, fails gracefully when the backend is unreachable, keeps the
  last-known published task state, and its local research-stop matcher
  mirrors the brain contract (negation-aware).
* F50  published voice state: /voice-state/publish is accepted (authed) and
  /voice-state merges it over the backend's own listener_state copy.
* F52  instance stamps: atomic write/read/clear in data/runtime and
  approvals.invalidate() on supervisor reset.
"""

import os

import pytest


# ── F51: local token semantics ──────────────────────────────────────────────
def test_local_auth_mint_is_long_and_unique():
    from backend.services import local_auth

    a = local_auth.mint_token()
    b = local_auth.mint_token()
    assert len(a) >= 32
    assert a != b


def test_local_auth_unarmed_fails_closed(monkeypatch):
    # F51 (updated): an unarmed backend no longer disables enforcement. Only
    # an explicit JARVIS_DEV_MODE declares development mode; otherwise every
    # non-public request is refused.
    from backend.services import local_auth

    monkeypatch.delenv("JARVIS_LOCAL_TOKEN", raising=False)
    monkeypatch.delenv("JARVIS_DEV_MODE", raising=False)
    local_auth.configure("")
    assert local_auth.is_enabled() is False
    assert local_auth.enforcement_active() is True
    assert local_auth.check("") is False         # fail closed
    assert local_auth.check("anything") is False
    assert local_auth.posture() == "closed"
    assert local_auth.token_fingerprint() == "off"


def test_local_auth_dev_mode_is_the_only_way_to_open(monkeypatch):
    from backend.services import local_auth

    monkeypatch.delenv("JARVIS_LOCAL_TOKEN", raising=False)
    monkeypatch.setenv("JARVIS_DEV_MODE", "1")
    local_auth.configure("")
    assert local_auth.is_enabled() is False
    assert local_auth.enforcement_active() is False
    assert local_auth.check("") is True          # explicit development mode
    assert local_auth.posture() == "off"
    monkeypatch.delenv("JARVIS_DEV_MODE", raising=False)
    local_auth.configure("")


def test_local_auth_armed_enforces_constant_time():
    from backend.services import local_auth

    token = local_auth.mint_token()
    assert local_auth.configure(token) is True
    assert local_auth.is_enabled() is True
    assert local_auth.check(token) is True
    assert local_auth.check(token + "x") is False
    assert local_auth.check("") is False
    assert local_auth.check(None) is False
    assert local_auth.token_fingerprint() != "off"
    local_auth.configure("")


def test_local_auth_open_paths():
    from backend.services import local_auth

    # F51 (updated): only liveness is public; every private read (ui-state,
    # voice state/log, screen answers, research results) needs the token.
    assert local_auth.is_open("GET", "/health") is True
    assert local_auth.is_open("GET", "/ui-state") is False
    assert local_auth.is_open("GET", "/voice-state") is False
    assert local_auth.is_open("GET", "/voice-log") is False
    assert local_auth.is_open("GET", "/ask/status/req-x") is False
    assert local_auth.is_open("GET", "/screen-answer") is False
    assert local_auth.is_open("GET", "/research-result") is False
    assert local_auth.is_open("POST", "/health") is False
    assert local_auth.is_open("POST", "/task/stop") is False
    assert local_auth.is_open("POST", "/ask") is False
    assert local_auth.is_open("GET", "/settings") is False


# ── F52: runtime identity stamps ────────────────────────────────────────────
def test_runtime_identity_stable_instance_id():
    from backend.services import runtime_identity

    assert runtime_identity.instance_id() == runtime_identity.instance_id()
    assert len(runtime_identity.instance_id()) >= 16
    assert isinstance(runtime_identity.protocol_version(), int)
    assert runtime_identity.protocol_version() >= 2


def test_runtime_identity_stamp_roundtrip(tmp_path, monkeypatch):
    from backend.services import runtime_identity

    monkeypatch.setattr(runtime_identity, "_RUNTIME_DIR", tmp_path)
    stamp = runtime_identity.write_instance_file(
        role="backend", extra={"auth": "fp123"}
    )
    assert stamp is not None
    assert stamp["role"] == "backend"
    assert stamp["instance_id"] == runtime_identity.instance_id()

    read = runtime_identity.read_instance_file("backend")
    assert read is not None
    assert read["instance_id"] == stamp["instance_id"]
    assert read["auth"] == "fp123"
    assert "protocol" in read and "build" in read and "started_at" in read

    runtime_identity.clear_instance_file("backend")
    assert runtime_identity.read_instance_file("backend") is None


# ── F52: approval invalidation on worker replacement ───────────────────────
def test_approvals_invalidate_drops_pending():
    from backend.services import approvals

    approvals.clear()
    record = approvals.arm(
        {"steps": [{"action": "click", "target": "x"}], "command_text": "do it"},
        command_text="do it",
    )
    assert approvals.pending() is not None
    dropped = approvals.invalidate("supervisor reset")
    assert dropped is not None and dropped.plan_hash == record.plan_hash
    assert approvals.pending() is None


def test_approvals_invalidate_no_pending_is_noop():
    from backend.services import approvals

    approvals.clear()
    assert approvals.invalidate("test") is None


# ── F50/F51: backend HTTP surface ───────────────────────────────────────────
def _make_client():
    from fastapi.testclient import TestClient
    from backend.main import app

    return TestClient(app)


def test_health_exposes_identity_and_stays_open():
    from backend.services import local_auth
    from backend.services import runtime_identity

    local_auth.configure(local_auth.mint_token())
    try:
        client = _make_client()
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["service"] == "jarvis-backend"
        assert body["instance_id"] == runtime_identity.instance_id()
        assert body["protocol"] == runtime_identity.protocol_version()
        assert body["auth"] != "off"
    finally:
        local_auth.configure("")


def test_protected_post_requires_token():
    from backend.services import local_auth

    token = local_auth.mint_token()
    local_auth.configure(token)
    try:
        client = _make_client()
        # Without the token → 401.
        res = client.post("/voice-state/publish", json={"status": "listening"})
        assert res.status_code == 401
        # With the token → accepted.
        res = client.post(
            "/voice-state/publish",
            json={"status": "hearing", "thinking": True},
            headers={"X-Jarvis-Token": token},
        )
        assert res.status_code == 200
        assert res.json().get("ok") is True
    finally:
        local_auth.configure("")


def test_published_voice_state_merges_over_local_copy():
    from backend.services import local_auth

    token = local_auth.mint_token()
    local_auth.configure(token)
    try:
        client = _make_client()
        client.post(
            "/voice-state/publish",
            json={"status": "speaking", "publisher_pid": 424242, "state_seq": 7},
            headers={"X-Jarvis-Token": token},
        )
        state = client.get("/voice-state",
                           headers={"X-Jarvis-Token": token}).json()
        assert state.get("status") == "speaking"
        assert state.get("publisher_pid") == 424242
        # Stale (older seq) publishes are rejected.
        res = client.post(
            "/voice-state/publish",
            json={"status": "listening", "state_seq": 3},
            headers={"X-Jarvis-Token": token},
        )
        assert res.json().get("stale") is True
        state = client.get("/voice-state",
                           headers={"X-Jarvis-Token": token}).json()
        assert state.get("status") == "speaking"
    finally:
        local_auth.configure("")


def test_ui_state_includes_published_voice_state():
    from backend.services import local_auth

    token = local_auth.mint_token()
    local_auth.configure(token)
    try:
        client = _make_client()
        client.post(
            "/voice-state/publish",
            json={"status": "thinking", "state_seq": 10},
            headers={"X-Jarvis-Token": token},
        )
        ui = client.get("/ui-state",
                        headers={"X-Jarvis-Token": token}).json()
        assert ui["state"].get("status") == "thinking"
    finally:
        local_auth.configure("")


# ── F50: voice I/O worker decoupling ────────────────────────────────────────
def test_voice_worker_attaches_token_header(monkeypatch):
    monkeypatch.setenv("JARVIS_LOCAL_TOKEN", "tok-" + "x" * 40)
    import backend.voice_mode as vm

    headers = vm._backend_headers()
    assert headers.get("X-Jarvis-Token") == "tok-" + "x" * 40
    assert headers.get("Content-Type") == "application/json"


def test_voice_worker_post_backend_fails_gracefully(monkeypatch):
    monkeypatch.delenv("JARVIS_LOCAL_TOKEN", raising=False)
    import backend.voice_mode as vm

    ok, reply = vm._post_backend("/approvals/reset", {}, timeout=0.1)
    assert ok is False and reply is None  # no backend is listening in tests


def test_voice_worker_ask_backend_fails_gracefully(monkeypatch):
    monkeypatch.delenv("JARVIS_LOCAL_TOKEN", raising=False)
    import backend.voice_mode as vm

    assert vm._ask_backend("hello", "req-1", timeout=0.3) is None


def test_voice_worker_keeps_last_known_task_state(monkeypatch):
    import backend.voice_mode as vm

    monkeypatch.setattr(vm, "_task_running_last_known", True)
    monkeypatch.setattr(vm, "_task_running_checked_at", 0.0)
    monkeypatch.setattr(vm, "_get_backend", lambda *a, **k: None)
    assert vm.backend_task_running() is True  # failure keeps last known


def test_voice_worker_research_stop_matcher():
    import backend.voice_mode as vm

    assert vm.is_stop_research("jarvis stop the research") is True
    assert vm.is_stop_research("stop the deepsearch now") is True
    assert vm.is_stop_research("don't stop the research") is False
    assert vm.is_stop_research("do not stop the search") is False
    assert vm.is_stop_research("what is today's weather") is False
    assert vm.is_stop_research("") is False


def test_voice_worker_has_no_brain_execution_surface():
    """F50: the voice I/O worker must not execute a second process_message
    copy — its namespace carries neither brain's process_message nor the
    module-copy task flag."""
    import backend.voice_mode as vm

    assert not hasattr(vm, "process_message")
    assert not hasattr(vm, "opencode_task_in_progress")