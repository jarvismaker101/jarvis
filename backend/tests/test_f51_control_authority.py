"""F51 — Separate Rendering From Computer-Control Authority.

Pins the finding's Acceptance clauses against real code paths:

1. **Missing/wrong/old tokens cannot mutate or read private data.** The
   middleware fails closed outside explicit development mode, all private
   reads require the launch token, and a rotated (old) token stops matching
   immediately.
2. **All legitimate controls work.** A correctly authenticated client — the
   one built by :func:`local_auth.auth_headers` — can still publish voice
   state, reset approvals, toggle voice mode and read private state.
3. **Renderer content cannot gain unrestricted authority.** A foreign
   (web-page) Origin is refused before the token check; a renderer-origin
   caller can never read the supervisor identity/token fingerprint fields.
4. **Public health is minimal.** ``GET /health`` stays open for liveness and
   the supervisor attribution contract, and exposes nothing else.
"""

import os
import unittest

from backend.services import local_auth


def _make_client():
    from fastapi.testclient import TestClient
    from backend.main import app

    return TestClient(app)


class _AuthTestCase(unittest.TestCase):
    """Arms one launch token for the test and cleans the posture up after."""

    def setUp(self):
        os.environ.pop("JARVIS_DEV_MODE", None)
        self.token = local_auth.mint_token()
        local_auth.configure(self.token)
        self.addCleanup(self._disarm)

    def _disarm(self):
        local_auth.configure("")
        os.environ.pop("JARVIS_LOCAL_TOKEN", None)
        os.environ.pop("JARVIS_DEV_MODE", None)

    def headers(self, token=None):
        return {local_auth.HEADER: self.token if token is None else token}


class FailClosedPostureTests(unittest.TestCase):
    """Acceptance 1 — fail closed outside explicit development mode."""

    def tearDown(self):
        local_auth.configure("")
        os.environ.pop("JARVIS_DEV_MODE", None)
        os.environ.pop("JARVIS_LOCAL_TOKEN", None)

    def test_unarmed_backend_without_dev_mode_is_closed(self):
        os.environ.pop("JARVIS_DEV_MODE", None)
        local_auth.configure("")
        self.assertFalse(local_auth.is_enabled())
        self.assertTrue(local_auth.enforcement_active())
        self.assertEqual(local_auth.posture(), "closed")
        self.assertFalse(local_auth.check(""))
        self.assertFalse(local_auth.check("some-other-token"))

    def test_explicit_dev_mode_is_the_only_open_posture(self):
        os.environ["JARVIS_DEV_MODE"] = "1"
        local_auth.configure("")
        self.assertFalse(local_auth.is_enabled())
        self.assertFalse(local_auth.enforcement_active())
        self.assertEqual(local_auth.posture(), "off")
        self.assertTrue(local_auth.check(""))

    def test_missing_token_cannot_read_private_state_over_http(self):
        os.environ.pop("JARVIS_DEV_MODE", None)
        local_auth.configure("")
        client = _make_client()
        for path in ("/ui-state", "/voice-state", "/voice-log",
                     "/screen-answer", "/research-result",
                     "/research-progress", "/ask/status/req-x"):
            response = client.get(path)
            self.assertEqual(response.status_code, 401, path)
        self.assertEqual(
            client.post("/voice-state/publish", json={"status": "x"})
            .status_code, 401)
        self.assertEqual(client.post("/ask", json={"message": "hi"})
                         .status_code, 401)


class PrivateReadTests(_AuthTestCase):
    """Acceptance 1 — private reads require the token; legit reads work."""

    PRIVATE_GETS = ("/ui-state", "/voice-state", "/voice-log",
                    "/voice-mode", "/screen-answer", "/research-result",
                    "/research-progress")

    def test_private_reads_require_the_token(self):
        client = _make_client()
        for path in self.PRIVATE_GETS:
            self.assertEqual(client.get(path).status_code, 401, path)

    def test_private_reads_work_with_the_token(self):
        client = _make_client()
        for path in self.PRIVATE_GETS:
            self.assertNotEqual(
                client.get(path, headers=self.headers()).status_code, 401,
                path)

    def test_ask_status_requires_the_token(self):
        client = _make_client()
        self.assertEqual(client.get("/ask/status/req-x").status_code, 401)
        self.assertNotEqual(
            client.get("/ask/status/req-x", headers=self.headers())
            .status_code, 401)

    def test_wrong_token_is_never_accepted(self):
        client = _make_client()
        wrong = local_auth.mint_token()
        self.assertEqual(client.get("/ui-state", headers=self.headers(wrong))
                         .status_code, 401)
        self.assertEqual(
            client.post("/voice-state/publish", json={"status": "x"},
                        headers=self.headers(wrong)).status_code, 401)


class RotationTests(_AuthTestCase):
    """Acceptance 1 — an OLD token stops working immediately."""

    def test_rotated_token_stops_matching(self):
        client = _make_client()
        old = self.token
        self.assertEqual(
            client.post("/voice-state/publish", json={"status": "listening"},
                        headers=self.headers(old)).status_code, 200)

        new = local_auth.rotate()
        self.assertNotEqual(new, old)
        self.assertNotEqual(local_auth.fingerprint_for(new),
                            local_auth.fingerprint_for(old))
        self.assertGreaterEqual(local_auth.rotations(), 1)

        self.assertEqual(
            client.post("/voice-state/publish", json={"status": "listening"},
                        headers=self.headers(old)).status_code, 401)
        self.assertEqual(
            client.post("/voice-state/publish", json={"status": "listening"},
                        headers=self.headers(new)).status_code, 200)


class CentralizedClientTests(_AuthTestCase):
    """Acceptance 2 — one authenticated client, all controls work."""

    def test_auth_headers_builds_the_authenticated_client(self):
        os.environ["JARVIS_LOCAL_TOKEN"] = "env-" + "t" * 40
        try:
            headers = local_auth.auth_headers()
            self.assertEqual(headers[local_auth.HEADER], "env-" + "t" * 40)
            self.assertEqual(headers["Content-Type"], "application/json")
            merged = local_auth.auth_headers({"X-Extra": "1"})
            self.assertEqual(merged["X-Extra"], "1")
            self.assertNotIn(local_auth.HEADER, local_auth.auth_headers(
                token=""))
        finally:
            os.environ.pop("JARVIS_LOCAL_TOKEN", None)

    def test_centralized_headers_authenticate_a_real_request(self):
        client = _make_client()
        headers = local_auth.auth_headers(extra=None, token=self.token)
        response = client.post("/voice-state/publish",
                               json={"status": "listening"}, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json().get("ok") is not False)

    def test_legitimate_controls_still_work(self):
        client = _make_client()
        reset = client.post("/approvals/reset", json={},
                            headers=self.headers())
        self.assertEqual(reset.status_code, 200)
        self.assertTrue(reset.json().get("ok"))

        mode = client.post("/voice-mode", json={"enabled": True},
                           headers=self.headers())
        self.assertEqual(mode.status_code, 200)
        self.assertIn("voice_input_enabled", mode.json())

        publish = client.post("/voice-state/publish",
                              json={"status": "listening"},
                              headers=self.headers())
        self.assertEqual(publish.status_code, 200)
        state = client.get("/voice-state", headers=self.headers())
        self.assertEqual(state.status_code, 200)
        self.assertEqual(state.json().get("status"), "listening")


class OriginBoundaryTests(_AuthTestCase):
    """Acceptance 3 — renderer content gains no unrestricted authority."""

    def test_foreign_origin_is_refused_before_the_token_check(self):
        client = _make_client()
        # A web page that somehow learned the token still cannot steer the
        # control surface: the Origin is refused first.
        response = client.post(
            "/voice-state/publish", json={"status": "x"},
            headers=dict(self.headers(), Origin="https://evil.example"))
        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            client.get("/health", headers={"Origin": "https://evil.example"})
            .status_code, 403)

    def test_renderer_origin_can_only_see_minimal_health(self):
        client = _make_client()
        body = client.get("/health", headers={"Origin": "null"}).json()
        self.assertEqual(body.get("service"), "jarvis-backend")
        self.assertNotIn("auth", body)
        self.assertNotIn("instance_id", body)
        self.assertNotIn("build", body)
        self.assertLessEqual(set(body),
                             set(local_auth.PUBLIC_RESPONSE_FIELDS[
                                 "GET /health"]))

    def test_supervisor_attribution_fields_need_no_origin(self):
        client = _make_client()
        body = client.get("/health").json()
        # The supervisor (no Origin, no token) still gets the attribution
        # contract F52 needs to decide whether a warm backend is ours.
        self.assertIn("instance_id", body)
        self.assertIn("auth", body)
        self.assertIn("pid", body)
        self.assertIn("protocol", body)
        self.assertNotIn("build", body)

    def test_origin_policy_accepts_renderer_and_loopback_origins(self):
        self.assertTrue(local_auth.origin_allowed(""))
        self.assertTrue(local_auth.origin_allowed("null"))
        self.assertTrue(local_auth.origin_allowed("file://"))
        self.assertTrue(local_auth.origin_allowed("http://127.0.0.1:5173"))
        self.assertTrue(local_auth.origin_allowed("http://localhost:8000"))
        self.assertFalse(local_auth.origin_allowed("https://evil.example"))
        self.assertFalse(local_auth.origin_allowed("http://127.0.0.1.evil.com"))


class CorsConfigurationTests(unittest.TestCase):
    """Acceptance 3 — the deployment no longer opens CORS to everyone."""

    def test_cors_is_not_wildcard_with_credentials(self):
        from backend.main import app

        entries = [
            entry for entry in app.user_middleware
            if "CORSMiddleware" in str(entry.cls)
            or "CORSMiddleware" in str(getattr(entry, "args", ()))
        ]
        self.assertTrue(entries, "no CORS middleware configured")
        options = dict(entries[0].kwargs)
        self.assertNotIn("*", list(options.get("allow_origins") or []))
        self.assertFalse(options.get("allow_credentials"))
        self.assertIn("null", options.get("allow_origins"))
        self.assertTrue(options.get("allow_origin_regex"))


if __name__ == "__main__":
    unittest.main()
