"""Retry/replay regressions for schema-8 guided mutation routes."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from elspeth.contracts.freeze import deep_thaw
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.resolved import SourceResolved
from elspeth.web.composer.guided.state_machine import (
    GuidedSession,
    SinkIntent,
    SourceIntent,
    TerminalKind,
    TerminalReason,
    TerminalState,
    TurnRecord,
)
from elspeth.web.composer.source_inspection import SourceInspectionFacts
from elspeth.web.sessions.guided_replay import guided_turn_token, load_guided_json_payload
from elspeth.web.sessions.protocol import CompositionStateData, GuidedOperationSettlementConflictError
from elspeth.web.sessions.routes._helpers import _initial_composition_state_with_guided_session
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


def _create_session(client: TestClient) -> str:
    response = client.post("/api/sessions", json={"title": "retry-safe-route"})
    assert response.status_code == 201
    session_id = response.json()["id"]
    assert type(session_id) is str
    return session_id


def _seed_exited_wire_state(client: TestClient, session_id: str) -> str:
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
    prior_live = replace(
        guided,
        history=(replace(guided.history[0], response_hash=None, summary=None),),
        terminal=None,
    )
    prior_turn_token = guided_turn_token(prior_live)
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
    return prior_turn_token


def _guided_turn_emitted_args(client: TestClient, session_id: str) -> list[dict]:
    messages = asyncio.run(client.app.state.session_service.get_messages(UUID(session_id), limit=None))
    events: list[dict] = []
    for message in messages:
        for tool_call in message.tool_calls or ():
            invocation = tool_call.get("invocation", {})
            if invocation.get("tool_name") == "guided_turn_emitted":
                events.append(json.loads(invocation["arguments_canonical"]))
    return events


_SOURCE_ID = "00000000-0000-4000-8000-000000000101"
_OUTPUT_ID = "00000000-0000-4000-8000-000000000102"
_INSPECTION_WARNING = "csv_jagged_rows: row 2 has fewer fields"


def _source_inspection() -> SourceInspectionFacts:
    return SourceInspectionFacts(
        source_kind="csv",
        redacted_identity={"filename": "input.csv", "mime_type": "text/csv"},
        byte_range_inspected=(0, 32),
        sample_row_count=1,
        observed_headers=("id", "name"),
        inferred_types={"id": "int", "name": "str"},
        url_candidates=(),
        warnings=(_INSPECTION_WARNING,),
    )


def _reviewed_source() -> SourceResolved:
    return SourceResolved(
        name="source",
        plugin="csv",
        options={"path": "/data/input.csv", "schema": {"mode": "observed"}},
        observed_columns=("id", "name"),
        sample_rows=({"id": 1, "name": "Ada"},),
        on_validation_failure="discard",
    )


def _earlier_checkpoint(name: str) -> tuple[GuidedSession, TurnType]:
    source = _reviewed_source()
    if name == "source_initial":
        return GuidedSession.initial(), TurnType.SINGLE_SELECT
    if name == "source_plugin_selection":
        return (
            replace(
                GuidedSession.initial(),
                source_order=(_SOURCE_ID,),
                pending_source_intents={
                    _SOURCE_ID: SourceIntent(
                        name="source",
                        phase="plugin_selection",
                        plugin=None,
                        options=None,
                        inspection_facts=None,
                        observed_columns=(),
                        sample_rows=(),
                    )
                },
            ),
            TurnType.SINGLE_SELECT,
        )
    if name == "source_plugin_options":
        return (
            replace(
                GuidedSession.initial(),
                source_order=(_SOURCE_ID,),
                pending_source_intents={
                    _SOURCE_ID: SourceIntent(
                        name="source",
                        phase="plugin_options",
                        plugin="csv",
                        options=None,
                        inspection_facts=None,
                        observed_columns=(),
                        sample_rows=(),
                    )
                },
            ),
            TurnType.SCHEMA_FORM,
        )
    if name == "source_inspection_review":
        return (
            replace(
                GuidedSession.initial(),
                source_order=(_SOURCE_ID,),
                pending_source_intents={
                    _SOURCE_ID: SourceIntent(
                        name="source",
                        phase="inspection_review",
                        plugin="csv",
                        options={"path": "/data/input.csv"},
                        inspection_facts=_source_inspection(),
                        observed_columns=("id", "name"),
                        sample_rows=({"id": 1, "name": "Ada"},),
                    )
                },
            ),
            TurnType.INSPECT_AND_CONFIRM,
        )
    base = replace(
        GuidedSession.initial(),
        step=GuidedStep.STEP_2_SINK,
        source_order=(_SOURCE_ID,),
        reviewed_sources={_SOURCE_ID: source},
    )
    if name == "output_initial":
        return base, TurnType.SINGLE_SELECT
    if name == "output_plugin_selection":
        return (
            replace(
                base,
                output_order=(_OUTPUT_ID,),
                pending_output_intents={_OUTPUT_ID: SinkIntent(name="main", phase="plugin_selection", plugin=None, options=None)},
            ),
            TurnType.SINGLE_SELECT,
        )
    if name == "output_plugin_options":
        return (
            replace(
                base,
                output_order=(_OUTPUT_ID,),
                pending_output_intents={_OUTPUT_ID: SinkIntent(name="main", phase="plugin_options", plugin="json", options=None)},
            ),
            TurnType.SCHEMA_FORM,
        )
    if name == "output_field_review":
        return (
            replace(
                base,
                output_order=(_OUTPUT_ID,),
                pending_output_intents={
                    _OUTPUT_ID: SinkIntent(
                        name="main",
                        phase="field_review",
                        plugin="json",
                        options={"path": "/data/output.jsonl"},
                    )
                },
            ),
            TurnType.MULTI_SELECT_WITH_CUSTOM,
        )
    raise AssertionError(f"unknown checkpoint {name}")


def _seed_exited_checkpoint(client: TestClient, session_id: str, checkpoint: GuidedSession, turn_type: TurnType) -> None:
    service = client.app.state.session_service
    state = _initial_composition_state_with_guided_session()
    exited = replace(
        checkpoint,
        history=(
            TurnRecord(
                step=checkpoint.step,
                turn_type=turn_type,
                payload_hash="c" * 64,
                response_hash="d" * 64,
                emitter="server",
                summary="Checkpoint before exit",
            ),
        ),
        terminal=TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml=None,
        ),
    )
    state = replace(state, guided_session=exited)
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
                composer_meta={"guided_session": exited.to_dict()},
            ),
            provenance="session_seed",
        )
    )


def test_reenter_replays_exact_located_response_without_a_second_state(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    prior_turn_token = _seed_exited_wire_state(composer_test_client, session_id)
    operation_id = str(uuid4())

    first = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/reenter",
        json={"operation_id": operation_id},
    )
    with patch(
        "elspeth.web.sessions.routes.composer.guided._build_get_guided_turn",
        side_effect=AssertionError("completed reentry replay must not rebuild live policy"),
    ):
        replay = composer_test_client.post(
            f"/api/sessions/{session_id}/guided/reenter",
            json={"operation_id": operation_id},
        )

    assert first.status_code == 200, first.json()
    assert replay.status_code == 200, replay.json()
    assert replay.json() == first.json()
    body = first.json()
    assert len(body["guided_session"]["history"]) == 2
    assert body["guided_session"]["history"][0]["response_hash"] == "b" * 64
    assert body["guided_session"]["history"][1]["response_hash"] is None
    assert body["next_turn"]["turn_token"] != prior_turn_token
    loaded = load_guided_json_payload(
        composer_test_client.app.state.payload_store,
        payload_id=body["guided_session"]["history"][-1]["payload_hash"],
        purpose="turn",
    )
    assert deep_thaw(loaded.payload) == body["next_turn"]["payload"]
    fetched = composer_test_client.get(f"/api/sessions/{session_id}/guided")
    assert fetched.status_code == 200
    assert fetched.json()["next_turn"]["turn_token"] == body["next_turn"]["turn_token"]
    emissions = _guided_turn_emitted_args(composer_test_client, session_id)
    assert len(emissions) == 1
    assert emissions[0]["payload_hash"] == body["guided_session"]["history"][-1]["payload_hash"]
    assert emissions[0]["payload_payload_id"] == emissions[0]["payload_hash"]
    service = composer_test_client.app.state.session_service
    versions = asyncio.run(service.get_state_versions(UUID(session_id)))
    assert [state.version for state in versions] == [1, 2]


def test_reenter_audit_insert_failure_rolls_back_new_occurrence(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    _seed_exited_wire_state(composer_test_client, session_id)
    service = composer_test_client.app.state.session_service

    with (
        patch.object(
            service,
            "_insert_prepared_guided_audit_rows_on_connection",
            side_effect=RuntimeError("injected audit insert failure"),
        ),
        pytest.raises(RuntimeError, match="injected audit insert failure"),
    ):
        composer_test_client.post(
            f"/api/sessions/{session_id}/guided/reenter",
            json={"operation_id": str(uuid4())},
        )

    versions = asyncio.run(service.get_state_versions(UUID(session_id)))
    assert [state.version for state in versions] == [1]
    assert _guided_turn_emitted_args(composer_test_client, session_id) == []


def test_reenter_settlement_head_conflict_is_terminal_and_exactly_replayed(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    _seed_exited_wire_state(composer_test_client, session_id)
    service = composer_test_client.app.state.session_service
    request_body = {"operation_id": str(uuid4())}

    with patch.object(
        service,
        "settle_guided_state_operation",
        side_effect=GuidedOperationSettlementConflictError(),
    ):
        first = composer_test_client.post(f"/api/sessions/{session_id}/guided/reenter", json=request_body)
    replay = composer_test_client.post(f"/api/sessions/{session_id}/guided/reenter", json=request_body)

    assert first.status_code == replay.status_code == 409
    assert first.json() == replay.json()
    assert first.json()["detail"]["failure_code"] == "stale_conflict"


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "source_initial",
        "source_plugin_selection",
        "source_plugin_options",
        "source_inspection_review",
        "output_initial",
        "output_plugin_selection",
        "output_plugin_options",
        "output_field_review",
    ],
)
def test_reenter_schema8_earlier_checkpoint_replays_exact_hashed_response(
    composer_test_client: TestClient,
    checkpoint_name: str,
) -> None:
    session_id = _create_session(composer_test_client)
    checkpoint, expected_turn_type = _earlier_checkpoint(checkpoint_name)
    _seed_exited_checkpoint(composer_test_client, session_id, checkpoint, expected_turn_type)
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
    assert first.json()["next_turn"]["type"] == expected_turn_type.value
    if checkpoint_name == "source_inspection_review":
        assert first.json()["next_turn"]["payload"]["observed"]["warnings"] == [_INSPECTION_WARNING]
        assert replay.json()["next_turn"]["payload"]["observed"]["warnings"] == [_INSPECTION_WARNING]
    versions = asyncio.run(composer_test_client.app.state.session_service.get_state_versions(UUID(session_id)))
    assert [state.version for state in versions] == [1, 2]


def test_reenter_rejects_missing_operation_id_before_mutation(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    _seed_exited_wire_state(composer_test_client, session_id)

    response = composer_test_client.post(f"/api/sessions/{session_id}/guided/reenter", json={})

    assert response.status_code == 422
    service = composer_test_client.app.state.session_service
    versions = asyncio.run(service.get_state_versions(UUID(session_id)))
    assert [state.version for state in versions] == [1]
