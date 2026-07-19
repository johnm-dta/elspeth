"""Authoritative, non-sensitive reconciliation for cold guided-start custody."""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

from fastapi import HTTPException

from elspeth.web.auth.middleware import get_current_user
from elspeth.web.sessions.protocol import GuidedOperationClaimed
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _create_session(client: TestClient) -> str:
    response = client.post("/api/sessions", json={"title": "guided-start-reconciliation"})
    assert response.status_code == 201
    return response.json()["id"]


def _reconcile(client: TestClient, session_id: str, operation_id: str):
    return client.post(f"/api/sessions/{session_id}/guided/start/{operation_id}/reconcile")


def test_reconciliation_reports_absent_with_exact_non_sensitive_shape(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)

    response = _reconcile(composer_test_client, session_id, str(uuid4()))

    assert response.status_code == 200
    assert response.json() == {"status": "absent"}


def test_reconciliation_reports_unexpired_without_exposing_operation_custody(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    operation_id = str(uuid4())
    service = composer_test_client.app.state.session_service
    claim = asyncio.run(
        service.reserve_guided_operation(
            session_id=UUID(session_id),
            operation_id=operation_id,
            kind="guided_start",
            request_hash="1" * 64,
            actor="worker",
            lease_seconds=30,
        )
    )
    assert isinstance(claim, GuidedOperationClaimed)

    response = _reconcile(composer_test_client, session_id, operation_id)

    assert response.status_code == 200
    assert response.json() == {"status": "in_progress"}
    assert claim.fence.lease_token not in response.text
    assert "request_hash" not in response.text
    assert "lease" not in response.text


def test_reconciliation_reports_closed_failure_only(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    operation_id = str(uuid4())
    service = composer_test_client.app.state.session_service
    claim = asyncio.run(
        service.reserve_guided_operation(
            session_id=UUID(session_id),
            operation_id=operation_id,
            kind="guided_start",
            request_hash="2" * 64,
            actor="worker",
            lease_seconds=30,
        )
    )
    assert isinstance(claim, GuidedOperationClaimed)
    asyncio.run(service.fail_guided_operation(claim.fence, failure_code="provider_timeout", actor="worker"))

    response = _reconcile(composer_test_client, session_id, operation_id)

    assert response.status_code == 200
    assert response.json() == {"status": "failed", "failure_code": "provider_timeout"}


def test_reconciliation_reports_completed_safe_state_locator(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    operation_id = str(uuid4())
    started = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/start",
        json={"profile": "live", "intent": "Build a pipeline", "operation_id": operation_id},
    )
    assert started.status_code == 200, started.json()
    state_id = started.json()["composition_state"]["id"]

    response = _reconcile(composer_test_client, session_id, operation_id)

    assert response.status_code == 200
    assert response.json() == {"status": "completed", "composition_state_id": state_id}
    assert set(response.json()) == {"status", "composition_state_id"}


def test_reconciliation_rejects_wrong_operation_kind(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    operation_id = str(uuid4())
    service = composer_test_client.app.state.session_service
    asyncio.run(
        service.reserve_guided_operation(
            session_id=UUID(session_id),
            operation_id=operation_id,
            kind="guided_chat",
            request_hash="3" * 64,
            actor="worker",
            lease_seconds=30,
        )
    )

    response = _reconcile(composer_test_client, session_id, operation_id)

    assert response.status_code == 409
    assert response.json() == {"detail": "Operation id is already bound to a different guided action."}


def test_reconciliation_is_session_owned_and_requires_authentication(composer_test_client: TestClient) -> None:
    alice_session = _create_session(composer_test_client)
    bob_session = asyncio.run(composer_test_client.app.state.session_service.create_session("bob", "Bob", "local"))
    operation_id = str(uuid4())

    assert _reconcile(composer_test_client, str(bob_session.id), operation_id).status_code == 404
    assert _reconcile(composer_test_client, str(uuid4()), operation_id).status_code == 404

    original_override = composer_test_client.app.dependency_overrides[get_current_user]

    async def unauthenticated():
        raise HTTPException(status_code=401, detail="Authentication required")

    composer_test_client.app.dependency_overrides[get_current_user] = unauthenticated
    try:
        response = _reconcile(composer_test_client, alice_session, operation_id)
    finally:
        composer_test_client.app.dependency_overrides[get_current_user] = original_override
    assert response.status_code == 401
