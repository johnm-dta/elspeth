"""Smoke test for composer guided fixtures.

Verifies that the composer_test_client and audit_recorder fixtures are
properly configured and the session creation endpoint works end-to-end.
"""

from __future__ import annotations

from elspeth.web.composer.audit import BufferingRecorder
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


class TestFixtureSmoke:
    """Verify that fixtures are properly wired."""

    def test_session_creation_works(self, composer_test_client: TestClient) -> None:
        """POST /api/sessions creates a session and returns 201."""
        resp = composer_test_client.post(
            "/api/sessions",
            json={"title": "Test Session"},
        )
        assert resp.status_code == 201
        body = resp.json()
        assert "id" in body
        assert body["title"] == "Test Session"
        assert body["user_id"] == "alice"

    def test_audit_recorder_is_present(self, composer_test_client: TestClient) -> None:
        """App state has composer_recorder fixture."""
        recorder = composer_test_client.app.state.composer_recorder
        assert recorder is not None

    def test_audit_recorder_fixture_accessible(self, audit_recorder: BufferingRecorder) -> None:
        """audit_recorder fixture yields a BufferingRecorder with empty initial state."""
        assert audit_recorder.invocations == ()

    def test_session_list_works(self, composer_test_client: TestClient) -> None:
        """GET /api/sessions returns list of sessions."""
        # Create a session first
        create_resp = composer_test_client.post(
            "/api/sessions",
            json={"title": "Session 1"},
        )
        assert create_resp.status_code == 201

        # List sessions
        list_resp = composer_test_client.get("/api/sessions")
        assert list_resp.status_code == 200
        body = list_resp.json()
        assert isinstance(body, list)
        assert len(body) > 0
        assert body[0]["title"] == "Session 1"
