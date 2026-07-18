"""Current-schema error, exit, and re-entry behavior for guided responses."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from elspeth.contracts.freeze import deep_thaw
from elspeth.web.composer.guided.protocol import TurnType
from elspeth.web.composer.guided.state_machine import (
    GuidedSession,
    GuidedStep,
    TerminalKind,
    TerminalReason,
    TerminalState,
    TurnRecord,
)
from elspeth.web.composer.state import CompositionState, OutputSpec, PipelineMetadata, SourceSpec
from elspeth.web.sessions.converters import state_from_record
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.routes import _initial_composition_state_with_guided_session
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient

_CURRENT_TOKEN = object()


def _create_session(client: TestClient) -> str:
    response = client.post("/api/sessions", json={"title": "error-path-test"})
    assert response.status_code == 201, response.json()
    return response.json()["id"]


def _get_guided(client: TestClient, session_id: str) -> dict:
    response = client.get(f"/api/sessions/{session_id}/guided")
    assert response.status_code == 200, response.json()
    return response.json()


def _respond_raw(
    client: TestClient,
    session_id: str,
    *,
    operation_id: str | None = None,
    turn_token: str | None | object = _CURRENT_TOKEN,
    **response_fields: object,
):
    if turn_token is _CURRENT_TOKEN:
        current = _get_guided(client, session_id)
        turn = current["next_turn"]
        turn_token = turn["turn_token"] if turn is not None else None
    return client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": operation_id or str(uuid4()),
            "turn_token": turn_token,
            **response_fields,
        },
    )


def _respond(client: TestClient, session_id: str, **response_fields: object) -> dict:
    response = _respond_raw(client, session_id, **response_fields)
    assert response.status_code == 200, response.json()
    return response.json()


def _reenter_raw(client: TestClient, session_id: str, *, operation_id: str | None = None):
    return client.post(
        f"/api/sessions/{session_id}/guided/reenter",
        json={"operation_id": operation_id or str(uuid4())},
    )


def _current_state(client: TestClient, session_id: str) -> tuple[object, CompositionState]:
    service = client.app.state.session_service
    record = asyncio.run(service.get_current_state(UUID(session_id)))
    assert record is not None
    return record, state_from_record(record)


def _persist_state(
    client: TestClient,
    session_id: str,
    state: CompositionState,
    *,
    meta_updates: dict[str, object] | None = None,
) -> None:
    service = client.app.state.session_service
    current = asyncio.run(service.get_current_state(UUID(session_id)))
    existing_meta = dict(deep_thaw(current.composer_meta)) if current is not None and current.composer_meta else {}
    guided = state.guided_session
    assert guided is not None
    existing_meta.update(meta_updates or {})
    existing_meta["guided_session"] = guided.to_dict()
    data = state.to_dict()
    asyncio.run(
        service.save_composition_state(
            UUID(session_id),
            CompositionStateData(
                sources=data["sources"],
                nodes=data["nodes"],
                edges=data["edges"],
                outputs=data["outputs"],
                metadata_=data["metadata"],
                is_valid=True,
                validation_errors=None,
                composer_meta=existing_meta,
            ),
            provenance="session_seed",
        )
    )


def _seed_terminal(client: TestClient, session_id: str, terminal: TerminalState) -> None:
    state = _initial_composition_state_with_guided_session()
    assert state.guided_session is not None
    record = TurnRecord(
        step=GuidedStep.STEP_1_SOURCE,
        turn_type=TurnType.SINGLE_SELECT,
        payload_hash="a" * 64,
        response_hash=None,
        emitter="server",
    )
    guided = replace(state.guided_session, history=(record,), terminal=terminal)
    _persist_state(client, session_id, replace(state, guided_session=guided))


def _seed_completed_pipeline(client: TestClient, session_id: str) -> None:
    upload = client.post(
        f"/api/sessions/{session_id}/blobs/inline",
        json={"filename": "rows.csv", "content": "price\n1.99\n", "mime_type": "text/csv"},
    )
    assert upload.status_code == 201, upload.json()
    blob = asyncio.run(client.app.state.blob_service.get_blob(UUID(upload.json()["id"])))
    output_dir = Path(client.app.state.settings.data_dir) / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    record = TurnRecord(
        step=GuidedStep.STEP_4_WIRE,
        turn_type=TurnType.CONFIRM_WIRING,
        payload_hash="a" * 64,
        response_hash="b" * 64,
        emitter="server",
        summary="Pipeline wiring confirmed.",
    )
    guided = replace(
        GuidedSession.initial(),
        step=GuidedStep.STEP_4_WIRE,
        history=(record,),
        terminal=TerminalState(
            kind=TerminalKind.COMPLETED,
            reason=None,
            pipeline_yaml="version: 1\n",
        ),
    )
    state = CompositionState(
        source=SourceSpec(
            plugin="csv",
            on_success="main",
            options={
                "path": blob.storage_path,
                "schema": {"mode": "observed"},
                "on_validation_failure": "discard",
            },
            on_validation_failure="discard",
        ),
        nodes=(),
        edges=(),
        outputs=(
            OutputSpec(
                name="main",
                plugin="json",
                options={
                    "path": str(output_dir / "rows.jsonl"),
                    "schema": {"mode": "observed"},
                    "mode": "write",
                    "collision_policy": "auto_increment",
                },
                on_write_failure="discard",
            ),
        ),
        metadata=PipelineMetadata(name="Completed pipeline"),
        version=1,
        guided_session=guided,
    )
    _persist_state(client, session_id, state)


def test_exit_from_current_turn_is_persisted_without_completion_marker(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)

    body = _respond(composer_test_client, session_id, control_signal="exit_to_freeform")

    assert body["terminal"] == {
        "kind": "exited_to_freeform",
        "reason": "user_pressed_exit",
        "pipeline_yaml": None,
    }
    assert body["next_turn"] is None
    persisted = _get_guided(composer_test_client, session_id)
    assert persisted["terminal"] == body["terminal"]
    record, _state = _current_state(composer_test_client, session_id)
    meta = dict(deep_thaw(record.composer_meta))
    assert "guided_completed_terminal_before_user_exit" not in meta


def test_exit_from_schema_form_uses_its_exact_turn_token(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    selected = _respond(composer_test_client, session_id, chosen=["csv"])
    assert selected["next_turn"]["type"] == "schema_form"

    exited = _respond(composer_test_client, session_id, control_signal="exit_to_freeform")

    assert exited["terminal"]["kind"] == "exited_to_freeform"
    assert exited["next_turn"] is None


def test_reentry_rejects_non_user_terminal(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    _seed_terminal(
        composer_test_client,
        session_id,
        TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.SOLVER_EXHAUSTED,
            pipeline_yaml=None,
        ),
    )

    response = _reenter_raw(composer_test_client, session_id)

    assert response.status_code == 409
    assert "user exit" in response.json()["detail"].lower()


def test_completed_exit_preserves_pipeline_and_emits_only_drop_audit(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    _seed_completed_pipeline(composer_test_client, session_id)
    before = _get_guided(composer_test_client, session_id)["composition_state"]

    exited = _respond(composer_test_client, session_id, turn_token=None, control_signal="exit_to_freeform")

    assert exited["terminal"] == {
        "kind": "exited_to_freeform",
        "reason": "user_pressed_exit",
        "pipeline_yaml": None,
    }
    after = exited["composition_state"]
    for field in ("sources", "nodes", "edges", "outputs"):
        assert after[field] == before[field]
    messages = asyncio.run(composer_test_client.app.state.session_service.get_messages(UUID(session_id), limit=None))
    tool_names: list[str] = []
    for message in messages:
        for tool_call in message.tool_calls or ():
            invocation = tool_call.get("invocation", {})
            name = invocation.get("tool_name")
            if isinstance(name, str):
                tool_names.append(name)
                if name == "guided_dropped_to_freeform":
                    arguments = json.loads(invocation["arguments_canonical"])
                    assert arguments["prev_step"] == "step_4_wire"
                    assert arguments["drop_reason"] == "user_pressed_exit"
    assert "guided_dropped_to_freeform" in tool_names
    assert "guided_turn_answered" not in tool_names


def test_completed_exit_reentry_restores_unchanged_completion(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    _seed_completed_pipeline(composer_test_client, session_id)
    _respond(composer_test_client, session_id, turn_token=None, control_signal="exit_to_freeform")

    response = _reenter_raw(composer_test_client, session_id)

    assert response.status_code == 200, response.json()
    assert response.json()["terminal"]["kind"] == "completed"
    assert response.json()["terminal"]["pipeline_yaml"]
    assert response.json()["next_turn"] is None
    record, _state = _current_state(composer_test_client, session_id)
    assert "guided_completed_terminal_before_user_exit" not in dict(deep_thaw(record.composer_meta))


def test_completed_exit_reentry_requires_reconfirmation_after_content_change(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)
    _seed_completed_pipeline(composer_test_client, session_id)
    _respond(composer_test_client, session_id, turn_token=None, control_signal="exit_to_freeform")
    _record, state = _current_state(composer_test_client, session_id)
    _persist_state(
        composer_test_client,
        session_id,
        replace(state, metadata=PipelineMetadata(name="Changed in freeform")),
    )

    response = _reenter_raw(composer_test_client, session_id)

    assert response.status_code == 200, response.json()
    assert response.json()["terminal"] is None
    assert response.json()["next_turn"]["type"] == "confirm_wiring"


@pytest.mark.parametrize("corrupt_marker", [None, {"composition_hash": 42}])
def test_completed_exit_reentry_fails_closed_on_corrupt_marker(
    composer_test_client: TestClient,
    corrupt_marker: object,
) -> None:
    session_id = _create_session(composer_test_client)
    _seed_completed_pipeline(composer_test_client, session_id)
    _respond(composer_test_client, session_id, turn_token=None, control_signal="exit_to_freeform")
    _before_record, state = _current_state(composer_test_client, session_id)
    _persist_state(
        composer_test_client,
        session_id,
        state,
        meta_updates={"guided_completed_terminal_before_user_exit": corrupt_marker},
    )
    service = composer_test_client.app.state.session_service
    versions_before = asyncio.run(service.get_state_versions(UUID(session_id)))

    response = _reenter_raw(composer_test_client, session_id)

    assert response.status_code == 500
    assert response.json()["detail"] == "Server invariant violated. See application audit log for diagnostic detail."
    after_record, _state = _current_state(composer_test_client, session_id)
    after_meta = dict(deep_thaw(after_record.composer_meta))
    assert after_meta["guided_completed_terminal_before_user_exit"] == corrupt_marker
    assert asyncio.run(service.get_state_versions(UUID(session_id))) == versions_before


def test_fresh_respond_requires_operation_id_and_exact_turn_token(composer_test_client: TestClient) -> None:
    session_id = _create_session(composer_test_client)

    missing = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={"chosen": ["csv"]},
    )
    stale = _respond_raw(composer_test_client, session_id, turn_token="a" * 64, chosen=["csv"])

    assert missing.status_code == 422
    assert stale.status_code == 409
    assert _respond(composer_test_client, session_id, chosen=["csv"])["next_turn"]["type"] == "schema_form"
