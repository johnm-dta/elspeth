"""Retry/replay regressions for schema-8 guided mutation routes."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from uuid import UUID, uuid4

from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import TerminalKind, TerminalReason, TerminalState, TurnRecord
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.routes._helpers import _initial_composition_state_with_guided_session
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _create_session(client: TestClient) -> str:
    response = client.post("/api/sessions", json={"title": "retry-safe-route"})
    assert response.status_code == 201
    return response.json()["id"]


def _seed_exited_wire_state(client: TestClient, session_id: str) -> None:
    service = client.app.state.session_service
    state = _initial_composition_state_with_guided_session()
    assert state.guided_session is not None
    guided = replace(
        state.guided_session,
        step=GuidedStep.STEP_4_WIRE,
        history=(
            TurnRecord(
                step=GuidedStep.STEP_4_WIRE,
                turn_type=TurnType.CONFIRM_WIRING,
                payload_hash="a" * 64,
                response_hash="b" * 64,
                emitter="server",
                summary="Exited wire review",
            ),
        ),
        terminal=TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml=None,
        ),
    )
    state = replace(state, guided_session=guided)
    state_data_raw = state.to_dict()
    asyncio.run(
        service.save_composition_state(
            UUID(session_id),
            CompositionStateData(
                sources=state_data_raw["sources"],
                nodes=state_data_raw["nodes"],
                edges=state_data_raw["edges"],
                outputs=state_data_raw["outputs"],
                metadata_=state_data_raw["metadata"],
                is_valid=False,
                validation_errors=None,
                composer_meta={"guided_session": guided.to_dict()},
            ),
            provenance="session_seed",
        )
    )


def test_reenter_replays_exact_located_response_without_a_second_state(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    _seed_exited_wire_state(composer_test_client, session_id)
    operation_id = str(uuid4())

    first = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/reenter",
        json={"operation_id": operation_id},
    )
    replay = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/reenter",
        json={"operation_id": operation_id},
    )

    assert first.status_code == 200, first.json()
    assert replay.status_code == 200, replay.json()
    assert replay.json() == first.json()
    service = composer_test_client.app.state.session_service
    versions = asyncio.run(service.get_state_versions(UUID(session_id)))
    assert [state.version for state in versions] == [1, 2]


def test_reenter_rejects_missing_operation_id_before_mutation(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    _seed_exited_wire_state(composer_test_client, session_id)

    response = composer_test_client.post(f"/api/sessions/{session_id}/guided/reenter", json={})

    assert response.status_code == 422
    service = composer_test_client.app.state.session_service
    versions = asyncio.run(service.get_state_versions(UUID(session_id)))
    assert [state.version for state in versions] == [1]
