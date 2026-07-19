"""Public-route persistence checks for schema-8 fork and revert."""

from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.sessions.converters import state_from_record
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def test_early_guided_checkpoint_survives_public_fork_and_revert_without_proposal_authority(
    composer_test_client: TestClient,
) -> None:
    created = composer_test_client.post("/api/sessions", json={"title": "schema-8 persistence"})
    assert created.status_code == 201, created.json()
    parent_id = created.json()["id"]

    started = composer_test_client.post(
        f"/api/sessions/{parent_id}/guided/start",
        json={"operation_id": str(uuid4()), "profile": "live", "intent": "Build a live pipeline"},
    )
    assert started.status_code == 200, started.json()
    target_state_id = started.json()["composition_state"]["id"]
    assert started.json()["guided_session"]["step"] == GuidedStep.STEP_1_SOURCE.value

    service = composer_test_client.app.state.session_service
    target_record = asyncio.run(service.get_state_in_session(UUID(target_state_id), UUID(parent_id)))
    target_guided = state_from_record(target_record).guided_session
    assert target_guided is not None
    assert target_guided.active_proposal is None
    assert target_guided.active_edit_target is None
    fork_message = asyncio.run(
        service.add_message(
            UUID(parent_id),
            "user",
            "Build from this checkpoint.",
            composition_state_id=UUID(target_state_id),
            writer_principal="route_user_message",
        )
    )
    forked = composer_test_client.post(
        f"/api/sessions/{parent_id}/fork",
        json={
            "operation_id": str(uuid4()),
            "from_message_id": str(fork_message.id),
            "new_message_content": "Build the edited request.",
        },
    )
    assert forked.status_code == 201, forked.json()
    child_id = UUID(forked.json()["session_id"])
    child_record = asyncio.run(service.get_current_state(child_id))
    assert child_record is not None
    child_guided = state_from_record(child_record).guided_session
    assert child_guided is not None
    assert child_guided.step is GuidedStep.STEP_1_SOURCE
    assert child_guided.active_proposal is None
    assert child_guided.active_edit_target is None

    reverted = composer_test_client.post(
        f"/api/sessions/{parent_id}/state/revert",
        json={"operation_id": str(uuid4()), "state_id": target_state_id},
    )
    assert reverted.status_code == 200, reverted.json()
    restored_record = asyncio.run(service.get_current_state(UUID(parent_id)))
    assert restored_record is not None
    restored_guided = state_from_record(restored_record).guided_session
    assert restored_guided is not None
    assert restored_guided.step is GuidedStep.STEP_1_SOURCE
    assert restored_guided.active_proposal is None
    assert restored_guided.active_edit_target is None
