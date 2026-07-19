"""Closed-stage Chat rejection and GET invariant sanitisation."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from elspeth.web.composer.guided.errors import InvariantError
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.sessions.routes.composer import guided as guided_route
from tests.integration.web.composer.guided.test_chat_schema8_atomic import _chat_operation_count, _persist_guided
from tests.integration.web.composer.guided.test_step_chat import _create_session
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def test_step_3_chat_is_rejected_before_provider_reservation_or_blob_work(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema-8 Step-3 Chat has no turn when there is no pending intent to manage."""
    client = composer_test_client
    session_id = _create_session(client)
    _persist_guided(client, session_id, GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS))
    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("Step 3 called the provider")),
    )
    reserve = AsyncMock(side_effect=AssertionError("Step 3 attempted blob custody"))
    monkeypatch.setattr(client.app.state.blob_service, "reserve_inline_custody", reserve)

    response = client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json={
            "operation_id": str(uuid4()),
            "turn_token": "a" * 64,
            "message": "Revise the transform proposal.",
        },
    )

    assert response.status_code == 409, response.json()
    assert response.json()["detail"] == {
        "code": "guided_chat_stage_unsupported",
        "detail": "Schema-8 CHAT is not available for step_3_transforms.",
    }
    assert _chat_operation_count(client, session_id) == 0
    reserve.assert_not_awaited()


class TestGetGuidedInvariantSanitisation:
    def test_get_guided_rebuild_invariant_error_returns_sanitised_500(self, composer_test_client: TestClient) -> None:
        """GET never returns Tier-1 invariant detail containing sample rows."""
        client = composer_test_client
        created = client.post("/api/sessions", json={"title": "guided-get-invariant"})
        assert created.status_code == 201, created.json()
        session_id = created.json()["id"]

        secret_marker = "TIER3-SAMPLE-ROW-CONTENT"
        with patch(
            "elspeth.web.sessions.routes.composer.guided._build_get_guided_turn",
            side_effect=InvariantError(f"corrupted record {{'sample_rows': '{secret_marker}'}}"),
        ):
            response = client.get(f"/api/sessions/{session_id}/guided")

        assert response.status_code == 500
        assert response.json()["detail"] == "Server invariant violated. See application audit log for diagnostic detail."
        assert secret_marker not in response.text
