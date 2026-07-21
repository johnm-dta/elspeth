"""Schema-8 atomic CHAT route contracts."""

from __future__ import annotations

import ast
import asyncio
import inspect
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
import structlog
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select, text

from elspeth.contracts.composer_llm_audit import ComposerChatTurnStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.guided.chat_solver import Step1SourceChatResolution
from elspeth.web.composer.guided.protocol import GuidedStep
from elspeth.web.composer.guided.resolved import SinkOutputResolved, SinkResolved
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.sessions._guided_step_chat import (
    GuidedStepChatOnlyResult,
    Step1SourceResolvedResult,
    Step2SinkResolvedResult,
    StepChatResult,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import guided_operations_table
from elspeth.web.sessions.protocol import CompositionStateData
from elspeth.web.sessions.routes._helpers import _initial_composition_state_with_guided_session
from elspeth.web.sessions.routes.composer import guided as guided_route
from elspeth.web.sessions.routes.composer.guided_chat_atomic import GuidedChatProviderOutcome
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.schemas import GuidedChatRequest
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry
from tests.unit.web._sync_asgi_client import SyncASGITestClient as TestClient


@pytest.fixture
def file_composer_test_client(composer_test_client: TestClient, tmp_path: Path) -> Iterator[TestClient]:
    """Rebind the minimal app to file SQLite for real multi-connection races."""
    engine = create_session_engine(f"sqlite:///{tmp_path / 'chat-races.db'}")
    initialize_session_schema(engine)
    composer_test_client.app.state.session_engine = engine
    composer_test_client.app.state.session_service = SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.guided.chat.races"),
    )
    try:
        yield composer_test_client
    finally:
        engine.dispose()


def _create_session(client: TestClient) -> str:
    response = client.post("/api/sessions", json={"title": "schema-8 chat"})
    assert response.status_code == 201, response.json()
    session_id = response.json()["id"]
    start = client.post(
        f"/api/sessions/{session_id}/guided/start",
        json={
            "profile": "live",
            "intent": "Begin this guided chat session.",
            "operation_id": str(uuid4()),
        },
    )
    assert start.status_code == 200, start.json()
    return session_id


def _chat_body(turn: dict, *, operation_id: str | None = None, message: str = "Use CSV") -> dict[str, str]:
    return {
        "operation_id": operation_id or str(uuid4()),
        "turn_token": turn["turn_token"],
        "message": message,
    }


def _chat_operation_count(client: TestClient, session_id: str) -> int:
    with client.app.state.session_engine.connect() as connection:
        return int(
            connection.execute(
                select(func.count())
                .select_from(guided_operations_table)
                .where(
                    guided_operations_table.c.session_id == session_id,
                    guided_operations_table.c.kind == "guided_chat",
                )
            ).scalar_one()
        )


def test_step_2_singular_sink_resolution_maps_to_the_live_transition() -> None:
    from elspeth.web.sessions.routes.composer.guided_chat_atomic import _transition_request

    body = GuidedChatRequest.model_validate(
        {
            "operation_id": str(uuid4()),
            "turn_token": "a" * 64,
            "message": "Use JSON",
        },
        strict=True,
    )
    sink = SinkResolved(
        outputs=(
            SinkOutputResolved(
                name="result",
                plugin="json",
                options={"path": "out.jsonl"},
                required_fields=(),
                schema_mode="observed",
                on_write_failure="discard",
            ),
        )
    )

    request = _transition_request(
        body=body,
        guided=SimpleNamespace(step=GuidedStep.STEP_2_SINK),
        current_turn={"type": "single_select", "step_index": 1, "payload": {}},
        source_resolution=None,
        sink_resolution=sink,
    )

    assert request is not None
    assert request.chosen == ["json"]


def _choose_source(client: TestClient, session_id: str, turn: dict, plugin: str = "csv") -> dict:
    response = client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": turn["turn_token"],
            "chosen": [plugin],
        },
    )
    assert response.status_code == 200, response.json()
    return response.json()


def _source_resolution() -> Step1SourceChatResolution:
    return Step1SourceChatResolution(
        assistant_message="I prepared the CSV source.",
        plugin="csv",
        filename="source.csv",
        mime_type="text/csv",
        content="name,value\nalice,1\n",
        options={"schema": {"mode": "observed"}},
        observed_columns=("name", "value"),
        sample_rows=({"name": "alice", "value": "1"},),
        on_validation_failure="discard",
    )


async def _resolved_source_provider(**_kwargs: object) -> GuidedChatProviderOutcome:
    resolution = _source_resolution()
    return Step1SourceResolvedResult(
        chat=StepChatResult(
            assistant_message=resolution.assistant_message,
            status=ComposerChatTurnStatus.SUCCESS,
            latency_ms=1,
            error_class=None,
        ),
        resolution=resolution,
    )


def _persist_guided(client: TestClient, session_id: str, guided: GuidedSession) -> None:
    state = replace(_initial_composition_state_with_guided_session(), guided_session=guided)
    state_dict = state.to_dict()
    asyncio.run(
        client.app.state.session_service.save_composition_state(
            UUID(session_id),
            CompositionStateData(
                sources=state_dict["sources"],
                nodes=state_dict["nodes"],
                edges=state_dict["edges"],
                outputs=state_dict["outputs"],
                metadata_=state_dict["metadata"],
                is_valid=False,
                composer_meta={"guided_session": guided.to_dict()},
            ),
            provenance="session_seed",
        )
    )


async def _advisory_provider(**_kwargs: object) -> GuidedChatProviderOutcome:
    return GuidedStepChatOnlyResult(
        chat=StepChatResult(
            assistant_message="Review the current source choices.",
            status=ComposerChatTurnStatus.SUCCESS,
            latency_ms=1,
            error_class=None,
        ),
    )


def test_advisory_chat_settles_once_and_exact_replay_ignores_mutable_provider_and_policy(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elspeth.web.sessions.routes.composer import guided_chat_atomic

    session_id = _create_session(composer_test_client)
    turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    body = _chat_body(turn)
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _advisory_provider, raising=False)

    first = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)

    assert first.status_code == 200, first.json()
    first_json = first.json()
    assert first_json["assistant_message"] == "Review the current source choices."
    assert first_json["assistant_message_kind"] == "assistant"
    assert first_json["next_turn"]["turn_token"] == turn["turn_token"]
    assert [item["role"] for item in first_json["guided_session"]["chat_history"][-2:]] == ["user", "assistant"]
    assert _chat_operation_count(composer_test_client, session_id) == 1

    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("replay called provider")),
        raising=False,
    )
    monkeypatch.setattr(
        guided_chat_atomic,
        "_request_plugin_policy_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("replay consulted mutable policy")),
    )
    replay = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)

    assert replay.status_code == 200, replay.json()
    assert replay.json() == first_json
    assert _chat_operation_count(composer_test_client, session_id) == 1


def test_reused_operation_id_with_different_message_conflicts_without_provider(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _create_session(composer_test_client)
    turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    body = _chat_body(turn)
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _advisory_provider, raising=False)
    first = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)
    assert first.status_code == 200, first.json()
    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("conflict called provider")),
        raising=False,
    )

    conflict = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json={**body, "message": "Use JSON instead"},
    )

    assert conflict.status_code == 409, conflict.json()
    assert conflict.json()["detail"] == "Operation id is already bound to a different request."
    assert _chat_operation_count(composer_test_client, session_id) == 1


def test_schema8_chat_rejects_step3_without_current_turn_before_provider_or_reservation(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _create_session(composer_test_client)
    _persist_guided(composer_test_client, session_id, GuidedSession(step=GuidedStep.STEP_3_TRANSFORMS))
    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unsupported stage called provider")),
        raising=False,
    )

    response = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json={"operation_id": str(uuid4()), "turn_token": "a" * 64, "message": "Build transforms"},
    )

    assert response.status_code == 409, response.json()
    assert response.json()["detail"] == {
        "code": "guided_chat_stage_unsupported",
        "detail": "Schema-8 CHAT is not available for step_3_transforms.",
    }
    assert _chat_operation_count(composer_test_client, session_id) == 0


def test_chat_route_has_no_legacy_decoders_direct_writers_or_chain_solver() -> None:
    from elspeth.web.sessions.routes.composer import guided_chat_atomic

    source = inspect.getsource(guided_route.post_guided_chat)
    implementation = inspect.getsource(guided_chat_atomic.post_guided_chat_schema8)
    tree = ast.parse(source + "\n" + implementation)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    assert "body.step_index" not in implementation
    assert "handle_step_1_source" not in names
    assert "handle_step_2_sink" not in names
    assert "solve_chain" not in names
    assert "save_composition_state" not in attributes
    assert "settle_guided_state_operation" in attributes


def test_expired_invalid_attempt_fails_preflight_without_attempt_bump_or_provider(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elspeth.web.sessions.guided_operations import guided_operation_request_hash
    from elspeth.web.sessions.protocol import GuidedOperationClaimed
    from elspeth.web.sessions.schemas import GuidedChatRequest

    session_id = _create_session(composer_test_client)
    turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    body = _chat_body({"turn_token": "0" * 64})
    request_model = GuidedChatRequest.model_validate(body, strict=True)
    service = composer_test_client.app.state.session_service
    claim = asyncio.run(
        service.reserve_guided_operation(
            session_id=UUID(session_id),
            operation_id=body["operation_id"],
            kind="guided_chat",
            request_hash=guided_operation_request_hash(
                session_id=UUID(session_id),
                kind="guided_chat",
                request=request_model,
            ),
            actor="composer_route",
            lease_seconds=300,
        )
    )
    assert isinstance(claim, GuidedOperationClaimed)
    with composer_test_client.app.state.session_engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE guided_operations SET lease_expires_at = :expired WHERE session_id = :session_id AND operation_id = :operation_id"
            ),
            {
                "expired": datetime.now(UTC) - timedelta(seconds=1),
                "session_id": session_id,
                "operation_id": body["operation_id"],
            },
        )
    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("invalid expired operation called provider")),
        raising=False,
    )

    response = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)

    assert response.status_code == 409, response.json()
    assert response.json()["detail"] == "turn_token does not identify the current unanswered turn."
    assert body["turn_token"] != turn["turn_token"]
    with composer_test_client.app.state.session_engine.connect() as connection:
        operation = (
            connection.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == session_id,
                    guided_operations_table.c.operation_id == body["operation_id"],
                )
            )
            .mappings()
            .one()
        )
    assert operation["attempt"] == 1
    assert operation["status"] == "in_progress"


def test_schema_form_source_resolution_is_advisory_without_blob_mutation_and_replays_exactly(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _create_session(composer_test_client)
    initial_turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    schema_turn = _choose_source(composer_test_client, session_id, initial_turn)["next_turn"]
    body = _chat_body(schema_turn)
    state_before = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["composition_state"]
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _resolved_source_provider, raising=False)
    reserve = AsyncMock(
        spec=composer_test_client.app.state.blob_service.reserve_inline_custody,
        side_effect=AssertionError("advisory Chat attempted blob custody"),
    )
    delete = AsyncMock(
        spec=composer_test_client.app.state.blob_service.delete_blob,
        side_effect=AssertionError("advisory Chat attempted blob deletion"),
    )
    monkeypatch.setattr(composer_test_client.app.state.blob_service, "reserve_inline_custody", reserve)
    monkeypatch.setattr(composer_test_client.app.state.blob_service, "delete_blob", delete)

    first = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)

    assert first.status_code == 200, first.json()
    first_json = first.json()
    assert first_json["assistant_message"] == (
        "I did not apply generated source content. Review the current source form and submit it through the wizard controls."
    )
    assert first_json["assistant_message_kind"] == "synthetic_failure"
    assert first_json["guided_session"]["chat_history"][-1]["synthetic_failure_reason"] == "not_applied"
    assert first_json["next_turn"]["turn_token"] == schema_turn["turn_token"]
    assert first_json["next_turn"]["payload"] == schema_turn["payload"]
    for key in ("sources", "nodes", "edges", "outputs", "metadata"):
        assert first_json["composition_state"][key] == state_before[key]
    assert asyncio.run(composer_test_client.app.state.blob_service.list_blobs(UUID(session_id))) == []
    reserve.assert_not_awaited()
    delete.assert_not_awaited()
    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("replay called provider")),
        raising=False,
    )
    replay = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)

    assert replay.status_code == 200, replay.json()
    assert replay.json() == first_json
    assert asyncio.run(composer_test_client.app.state.blob_service.list_blobs(UUID(session_id))) == []
    reserve.assert_not_awaited()
    delete.assert_not_awaited()


def test_same_operation_concurrent_callers_join_one_provider_result_outside_compose_lock(
    file_composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = file_composer_test_client
    session_id = _create_session(client)
    turn = client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    body = _chat_body(turn)
    initial_versions = asyncio.run(client.app.state.session_service.get_state_versions(UUID(session_id)))
    provider_calls = 0

    async def race() -> list[object]:
        nonlocal provider_calls
        provider_started = asyncio.Event()
        release_provider = asyncio.Event()

        async def controlled_provider(**kwargs: object) -> GuidedChatProviderOutcome:
            nonlocal provider_calls
            provider_calls += 1
            compose_lock = await client.app.state.session_compose_lock_registry.get_lock(str(kwargs["session_id"]))
            assert not compose_lock.locked(), "provider work must run outside the per-session compose lock"
            provider_started.set()
            await release_provider.wait()
            return await _advisory_provider()

        monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", controlled_provider, raising=False)
        async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://test") as async_client:
            winner = asyncio.create_task(async_client.post(f"/api/sessions/{session_id}/guided/chat", json=body))
            await asyncio.wait_for(provider_started.wait(), timeout=3)
            joiner = asyncio.create_task(async_client.post(f"/api/sessions/{session_id}/guided/chat", json=body))
            await asyncio.sleep(0)
            release_provider.set()
            return list(await asyncio.wait_for(asyncio.gather(winner, joiner), timeout=5))

    winner_response, joined_response = asyncio.run(race())

    assert winner_response.status_code == joined_response.status_code == 200
    assert winner_response.json() == joined_response.json()
    assert provider_calls == 1
    assert _chat_operation_count(client, session_id) == 1
    assert len(asyncio.run(client.app.state.session_service.get_state_versions(UUID(session_id)))) == len(initial_versions) + 1


def test_settlement_failure_rolls_back_chat_state_and_evidence_and_replays_typed_failure(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _create_session(composer_test_client)
    turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    body = _chat_body(turn)
    service = composer_test_client.app.state.session_service
    initial_versions = asyncio.run(service.get_state_versions(UUID(session_id)))
    initial_messages = asyncio.run(service.get_messages(UUID(session_id), limit=None))
    secret_canary = "/private/operator/chat-settlement-secret.csv"

    async def fail_settlement(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(secret_canary)

    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _advisory_provider, raising=False)
    monkeypatch.setattr(service, "settle_guided_state_operation", fail_settlement)
    first = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)
    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("terminal replay called provider")),
        raising=False,
    )
    replay = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)

    assert first.status_code == replay.status_code == 500
    assert replay.json() == first.json()
    assert first.json()["detail"]["failure_code"] == "operation_failed"
    assert secret_canary not in first.text
    assert asyncio.run(service.get_state_versions(UUID(session_id))) == initial_versions
    assert asyncio.run(service.get_messages(UUID(session_id), limit=None)) == initial_messages
    with composer_test_client.app.state.session_engine.connect() as connection:
        operation = (
            connection.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == session_id,
                    guided_operations_table.c.operation_id == body["operation_id"],
                )
            )
            .mappings()
            .one()
        )
    assert operation["status"] == "failed"
    assert operation["result_state_id"] is None
    assert operation["response_hash"] is None
    assert secret_canary not in str(dict(operation))


def test_provider_head_drift_fails_closed_without_settling_chat(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _create_session(composer_test_client)
    turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    body = _chat_body(turn)
    service = composer_test_client.app.state.session_service
    initial_versions = asyncio.run(service.get_state_versions(UUID(session_id)))
    initial_messages = asyncio.run(service.get_messages(UUID(session_id), limit=None))

    async def drifting_provider(**kwargs: object) -> GuidedChatProviderOutcome:
        state = kwargs["state"]
        state_dict = state.to_dict()  # type: ignore[union-attr]
        await service.save_composition_state(
            UUID(session_id),
            CompositionStateData(
                sources=state_dict["sources"],
                nodes=state_dict["nodes"],
                edges=state_dict["edges"],
                outputs=state_dict["outputs"],
                metadata_=state_dict["metadata"],
                is_valid=False,
                composer_meta={"guided_session": state.guided_session.to_dict()},  # type: ignore[union-attr]
            ),
            provenance="session_seed",
        )
        return await _advisory_provider()

    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", drifting_provider, raising=False)
    response = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)
    replay = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)

    assert response.status_code == replay.status_code == 409
    assert response.json() == replay.json()
    assert response.json()["detail"]["failure_code"] == "stale_conflict"
    assert len(asyncio.run(service.get_state_versions(UUID(session_id)))) == len(initial_versions) + 1
    assert asyncio.run(service.get_messages(UUID(session_id), limit=None)) == initial_messages


def test_exact_replay_fails_closed_when_current_turn_payload_is_tampered(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _create_session(composer_test_client)
    turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    body = _chat_body(turn)
    initial_versions = asyncio.run(composer_test_client.app.state.session_service.get_state_versions(UUID(session_id)))
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _advisory_provider, raising=False)
    first = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)
    assert first.status_code == 200
    store = composer_test_client.app.state.payload_store

    monkeypatch.setattr(type(store), "retrieve", lambda _self, _content_hash: b"{}")
    monkeypatch.setattr(
        guided_route,
        "_run_guided_chat_provider_attempt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("tampered replay called provider")),
        raising=False,
    )

    with pytest.raises(AuditIntegrityError, match="bytes do not match"):
        composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)

    assert _chat_operation_count(composer_test_client, session_id) == 1
    assert (
        len(asyncio.run(composer_test_client.app.state.session_service.get_state_versions(UUID(session_id)))) == len(initial_versions) + 1
    )


def test_expired_operation_takeover_fences_stale_worker_and_both_join_winner(
    file_composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = file_composer_test_client
    session_id = _create_session(client)
    turn = client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    body = _chat_body(turn)
    service = client.app.state.session_service
    initial_versions = asyncio.run(service.get_state_versions(UUID(session_id)))
    engine = client.app.state.session_engine
    provider_calls = 0

    async def race() -> list[object]:
        nonlocal provider_calls
        stale_provider_started = asyncio.Event()
        release_stale_provider = asyncio.Event()
        takeover_provider_started = asyncio.Event()

        async def controlled_provider(**_kwargs: object) -> GuidedChatProviderOutcome:
            nonlocal provider_calls
            provider_calls += 1
            if provider_calls == 1:
                stale_provider_started.set()
                await release_stale_provider.wait()
            else:
                takeover_provider_started.set()
            return await _advisory_provider()

        monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", controlled_provider, raising=False)
        async with AsyncClient(transport=ASGITransport(app=client.app), base_url="http://test") as async_client:
            stale = asyncio.create_task(async_client.post(f"/api/sessions/{session_id}/guided/chat", json=body))
            await asyncio.wait_for(stale_provider_started.wait(), timeout=3)
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE guided_operations SET lease_expires_at = :expired "
                        "WHERE session_id = :session_id AND operation_id = :operation_id"
                    ),
                    {
                        "expired": datetime.now(UTC) - timedelta(seconds=1),
                        "session_id": session_id,
                        "operation_id": body["operation_id"],
                    },
                )
            winner = asyncio.create_task(async_client.post(f"/api/sessions/{session_id}/guided/chat", json=body))
            await asyncio.wait_for(takeover_provider_started.wait(), timeout=3)
            winner_response = await asyncio.wait_for(winner, timeout=3)
            release_stale_provider.set()
            stale_response = await asyncio.wait_for(stale, timeout=3)
            return [stale_response, winner_response]

    stale_response, winner_response = asyncio.run(race())

    assert stale_response.status_code == winner_response.status_code == 200
    assert stale_response.json() == winner_response.json()
    assert provider_calls == 2
    with engine.connect() as connection:
        operation = (
            connection.execute(
                select(guided_operations_table).where(
                    guided_operations_table.c.session_id == session_id,
                    guided_operations_table.c.operation_id == body["operation_id"],
                )
            )
            .mappings()
            .one()
        )
    assert operation["status"] == "completed"
    assert operation["attempt"] == 2
    assert len(asyncio.run(service.get_state_versions(UUID(session_id)))) == len(initial_versions) + 1


def test_single_select_inline_source_resolution_materializes_blob_and_prefills_schema_form(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inline resolve_source content with no uploaded blob becomes a session blob.

    The plugin-selection transition must carry inspection facts derived from
    the materialized bytes so the next turn is a blob-backed, continuable
    schema form instead of the bare ``options: null`` stall.
    """
    session_id = _create_session(composer_test_client)
    initial_turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    assert initial_turn["type"] == "single_select"
    body = _chat_body(initial_turn, message="The rows are name,value pairs; create the source inline.")
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _resolved_source_provider, raising=False)

    first = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=body)

    assert first.status_code == 200, first.json()
    first_json = first.json()
    assert first_json["assistant_message"] == "I prepared the CSV source."
    assert first_json["assistant_message_kind"] == "assistant"

    blobs = asyncio.run(composer_test_client.app.state.blob_service.list_blobs(UUID(session_id)))
    assert len(blobs) == 1
    blob = blobs[0]
    assert blob.filename == "source.csv"
    assert blob.mime_type == "text/csv"
    assert blob.created_by == "assistant"
    assert blob.status == "ready"
    content = asyncio.run(composer_test_client.app.state.blob_service.read_blob_content(blob.id))
    assert content == b"name,value\nalice,1\n"

    next_turn = first_json["next_turn"]
    assert next_turn["type"] == "schema_form"
    prefilled = next_turn["payload"]["prefilled"]
    assert prefilled["path"] == f"blob:{blob.id}"
    assert prefilled["on_validation_failure"] == "discard"
    assert prefilled["schema"]["mode"] in {"flexible", "observed"}

    record = asyncio.run(composer_test_client.app.state.session_service.get_current_state(UUID(session_id)))
    assert record is not None
    persisted_guided = record.composer_meta["guided_session"]
    intent = next(iter(persisted_guided["pending_source_intents"].values()))
    assert intent["phase"] == "plugin_options"
    assert intent["plugin"] == "csv"
    assert intent["inspection_facts"] is not None


def test_inline_source_walks_prefilled_form_through_inspection_to_resolved_source(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The materialized inline blob drives the wizard to a reviewed source."""
    session_id = _create_session(composer_test_client)
    initial_turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _resolved_source_provider, raising=False)
    chat = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=_chat_body(initial_turn))
    assert chat.status_code == 200, chat.json()
    schema_turn = chat.json()["next_turn"]
    assert schema_turn["type"] == "schema_form"
    prefilled = schema_turn["payload"]["prefilled"]
    assert prefilled["path"].startswith("blob:")

    form = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": schema_turn["turn_token"],
            "edited_values": {
                "plugin": "csv",
                "options": {
                    "path": prefilled["path"],
                    "schema": prefilled["schema"],
                    "on_validation_failure": prefilled["on_validation_failure"],
                },
            },
        },
    )
    assert form.status_code == 200, form.json()
    inspect_turn = form.json()["next_turn"]
    assert inspect_turn["type"] == "inspect_and_confirm"

    confirm = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": inspect_turn["turn_token"],
            "edited_values": {"columns": ["name", "value"]},
        },
    )
    assert confirm.status_code == 200, confirm.json()

    record = asyncio.run(composer_test_client.app.state.session_service.get_current_state(UUID(session_id)))
    assert record is not None
    persisted_guided = record.composer_meta["guided_session"]
    assert persisted_guided["pending_source_intents"] == {}
    source = next(iter(persisted_guided["reviewed_sources"].values()))
    assert source["plugin"] == "csv"
    assert tuple(source["observed_columns"]) == ("name", "value")
    assert source["options"]["path"] == prefilled["path"]


async def _resolved_sink_provider(**_kwargs: object) -> GuidedChatProviderOutcome:
    sink = SinkResolved(
        outputs=(
            SinkOutputResolved(
                name="result",
                plugin="json",
                options={"path": "out.json", "schema": {"mode": "observed"}},
                required_fields=(),
                schema_mode="observed",
                on_write_failure="discard",
            ),
        )
    )
    return Step2SinkResolvedResult(
        chat=StepChatResult(
            assistant_message="I set up the JSON sink.",
            status=ComposerChatTurnStatus.SUCCESS,
            latency_ms=1,
            error_class=None,
        ),
        sink=sink,
    )


def test_sink_resolution_prefills_schema_form_from_chat_options(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sink resolution's options must survive plugin selection as prefill.

    One chat message per stage is the tutorial contract: after the resolution
    answers the sink single_select, the schema form must render with the
    resolution's options (path included) so the wizard's Continue is live —
    not the bare ``path: Not set`` stall.
    """
    session_id = _create_session(composer_test_client)
    initial_turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _resolved_source_provider, raising=False)
    chat = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=_chat_body(initial_turn))
    schema_turn = chat.json()["next_turn"]
    prefilled = schema_turn["payload"]["prefilled"]
    form = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": schema_turn["turn_token"],
            "edited_values": {
                "plugin": "csv",
                "options": {
                    "path": prefilled["path"],
                    "schema": prefilled["schema"],
                    "on_validation_failure": prefilled["on_validation_failure"],
                },
            },
        },
    )
    inspect_turn = form.json()["next_turn"]
    confirm = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": inspect_turn["turn_token"],
            "edited_values": {"columns": ["name", "value"]},
        },
    )
    review_turn = confirm.json()["next_turn"]
    assert review_turn["type"] == "review_components"
    finish = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/respond",
        json={
            "operation_id": str(uuid4()),
            "turn_token": review_turn["turn_token"],
            "component_action": {"action": "finish", "component_kind": "source"},
        },
    )
    assert finish.status_code == 200, finish.json()
    sink_select_turn = finish.json()["next_turn"]
    assert sink_select_turn["type"] == "single_select"

    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _resolved_sink_provider, raising=False)
    sink_chat = composer_test_client.post(
        f"/api/sessions/{session_id}/guided/chat",
        json=_chat_body(sink_select_turn, message="Save the results to a JSON file."),
    )

    assert sink_chat.status_code == 200, sink_chat.json()
    sink_form_turn = sink_chat.json()["next_turn"]
    assert sink_form_turn["type"] == "schema_form"
    sink_prefilled = sink_form_turn["payload"]["prefilled"]
    assert sink_prefilled["path"] == "out.json"
    assert sink_prefilled["schema"]["mode"] == "observed"
    assert sink_prefilled["on_write_failure"] == "discard"


def test_inline_source_defers_to_existing_ready_uploaded_blob(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An uploaded blob stays authoritative; inline content is not stored."""
    session_id = _create_session(composer_test_client)
    uploaded = asyncio.run(
        composer_test_client.app.state.blob_service.create_blob(
            UUID(session_id),
            "uploaded.csv",
            b"name,value\nuploaded,9\n",
            "text/csv",
            created_by="user",
        )
    )
    initial_turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _resolved_source_provider, raising=False)

    first = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=_chat_body(initial_turn))

    assert first.status_code == 200, first.json()
    blobs = asyncio.run(composer_test_client.app.state.blob_service.list_blobs(UUID(session_id)))
    assert [blob.id for blob in blobs] == [uploaded.id]
    next_turn = first.json()["next_turn"]
    assert next_turn["type"] == "schema_form"
    assert next_turn["payload"]["prefilled"]["path"] == f"blob:{uploaded.id}"


def test_inline_source_unencodable_content_settles_as_advisory_without_blob(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lone-surrogate content from the provider must not 500 the turn."""
    session_id = _create_session(composer_test_client)
    initial_turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]

    async def surrogate_provider(**_kwargs: object) -> GuidedChatProviderOutcome:
        resolution = replace(_source_resolution(), content="name,value\n\ud800,1\n")
        return Step1SourceResolvedResult(
            chat=StepChatResult(
                assistant_message=resolution.assistant_message,
                status=ComposerChatTurnStatus.SUCCESS,
                latency_ms=1,
                error_class=None,
            ),
            resolution=resolution,
        )

    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", surrogate_provider, raising=False)

    first = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=_chat_body(initial_turn))

    assert first.status_code == 200, first.json()
    first_json = first.json()
    assert first_json["assistant_message_kind"] == "synthetic_failure"
    assert first_json["guided_session"]["chat_history"][-1]["synthetic_failure_reason"] == "not_applied"
    assert first_json["next_turn"]["turn_token"] == initial_turn["turn_token"]
    assert asyncio.run(composer_test_client.app.state.blob_service.list_blobs(UUID(session_id))) == []


def test_inline_source_quota_failure_settles_as_advisory_without_blob(
    composer_test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A quota-rejected inline source must not pretend it was applied."""
    from elspeth.web.blobs.service import BlobQuotaExceededError

    session_id = _create_session(composer_test_client)
    initial_turn = composer_test_client.get(f"/api/sessions/{session_id}/guided").json()["next_turn"]
    monkeypatch.setattr(guided_route, "_run_guided_chat_provider_attempt", _resolved_source_provider, raising=False)
    quota = AsyncMock(
        spec=composer_test_client.app.state.blob_service.create_blob,
        side_effect=BlobQuotaExceededError(session_id, current_bytes=10, limit_bytes=10),
    )
    monkeypatch.setattr(composer_test_client.app.state.blob_service, "create_blob", quota)

    first = composer_test_client.post(f"/api/sessions/{session_id}/guided/chat", json=_chat_body(initial_turn))

    assert first.status_code == 200, first.json()
    first_json = first.json()
    assert first_json["assistant_message_kind"] == "synthetic_failure"
    assert first_json["guided_session"]["chat_history"][-1]["synthetic_failure_reason"] == "not_applied"
    assert first_json["next_turn"]["turn_token"] == initial_turn["turn_token"]
    assert asyncio.run(composer_test_client.app.state.blob_service.list_blobs(UUID(session_id))) == []
