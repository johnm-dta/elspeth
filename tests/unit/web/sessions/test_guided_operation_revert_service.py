"""Atomic state-revert settlement under a guided-operation fence."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import structlog
from sqlalchemy import func, select, update
from sqlalchemy.pool import StaticPool

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import stable_hash
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import chat_messages_table, composition_states_table, guided_operations_table
from elspeth.web.sessions.protocol import (
    CompositionStateData,
    GuidedCompositionStateResult,
    GuidedOperationClaimed,
    GuidedOperationCompleted,
    GuidedOperationFenceLostError,
    GuidedOperationTakenOver,
    GuidedStartStateConverged,
    GuidedStartStateSeeded,
)
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


@pytest.fixture
def engine():
    engine = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(engine)
    return engine


@pytest.fixture
def service(engine):
    return SessionServiceImpl(engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test"))


@pytest.fixture
def file_engine(tmp_path: Path):
    engine = create_session_engine(f"sqlite:///{tmp_path / 'guided-start.db'}")
    initialize_session_schema(engine)
    try:
        yield engine
    finally:
        engine.dispose()


async def _claim(
    service,
    session_id,
    operation_id="00000000-0000-4000-8000-000000000001",
    *,
    kind="state_revert",
    request_hash="a" * 64,
):
    outcome = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind=kind,
        request_hash=request_hash,
        actor="route",
        lease_seconds=60,
    )
    assert isinstance(outcome, GuidedOperationClaimed)
    return outcome.fence


@pytest.mark.asyncio
async def test_revert_state_and_system_message_settle_in_fence_transaction(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    first = await service.save_composition_state(
        session.id,
        CompositionStateData(sources={"source": {"type": "csv"}}, is_valid=True),
        provenance="session_seed",
    )
    await service.save_composition_state(
        session.id,
        CompositionStateData(sources={"source": {"type": "api"}}, is_valid=True),
        provenance="session_seed",
    )
    fence = await _claim(service, session.id)

    reverted = await service.revert_state_for_guided_operation(
        fence,
        state_id=first.id,
        actor="route",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id), "version": state.version}),
    )

    assert reverted.version == 3
    messages = await service.get_messages(session.id, limit=None)
    assert messages[-1].content == "Pipeline reverted to version 1."
    outcome = await service.get_guided_operation(
        session_id=session.id,
        operation_id=fence.operation_id,
        kind="state_revert",
        request_hash="a" * 64,
    )
    assert outcome == GuidedOperationCompleted(
        result=GuidedCompositionStateResult(state_id=reverted.id),
        response_hash=stable_hash({"state_id": str(reverted.id), "version": 3}),
    )


@pytest.mark.asyncio
async def test_stale_revert_fence_writes_nothing(service, engine) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    first = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
    stale_fence = await _claim(service, session.id)
    with engine.begin() as conn:
        conn.execute(
            update(guided_operations_table)
            .where(guided_operations_table.c.session_id == str(session.id))
            .where(guided_operations_table.c.operation_id == stale_fence.operation_id)
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
    takeover = await service.reserve_guided_operation(
        session_id=session.id,
        operation_id=stale_fence.operation_id,
        kind="state_revert",
        request_hash="a" * 64,
        actor="route-b",
        lease_seconds=60,
    )
    assert isinstance(takeover, GuidedOperationTakenOver)

    with pytest.raises(GuidedOperationFenceLostError):
        await service.revert_state_for_guided_operation(
            stale_fence,
            state_id=first.id,
            actor="route-a",
            response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
        )

    with engine.connect() as conn:
        assert (
            conn.execute(
                select(func.count()).select_from(composition_states_table).where(composition_states_table.c.session_id == str(session.id))
            ).scalar_one()
            == 1
        )
        assert (
            conn.execute(
                select(func.count()).select_from(chat_messages_table).where(chat_messages_table.c.session_id == str(session.id))
            ).scalar_one()
            == 0
        )


@pytest.mark.asyncio
async def test_guided_state_save_system_message_and_settlement_are_atomic(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    fence = await _claim(service, session.id, kind="guided_convert")
    state_data = CompositionStateData(composer_meta={"guided_session": {"schema_version": 8}}, is_valid=True)

    saved = await service.save_state_for_guided_operation(
        fence,
        expected_current_state_id=None,
        expected_current_state_version=None,
        state=state_data,
        provenance="session_seed",
        actor="route",
        system_message="Switched to guided mode.",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
    )

    current = await service.get_current_state(session.id)
    assert current is not None and current.id == saved.id
    assert [message.content for message in await service.get_messages(session.id, limit=None)] == ["Switched to guided mode."]
    outcome = await service.get_guided_operation(
        session_id=session.id,
        operation_id=fence.operation_id,
        kind="guided_convert",
        request_hash="a" * 64,
    )
    assert outcome == GuidedOperationCompleted(
        result=GuidedCompositionStateResult(state_id=saved.id),
        response_hash=stable_hash({"state_id": str(saved.id)}),
    )


@pytest.mark.asyncio
async def test_existing_guided_state_can_settle_without_new_state_version(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    existing = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
    fence = await _claim(service, session.id, kind="guided_start")

    settled = await service.complete_existing_state_guided_operation(
        fence,
        state_id=existing.id,
        expected_current_state_id=existing.id,
        expected_current_state_version=existing.version,
        actor="route",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
    )

    assert settled == existing
    assert [state.id for state in await service.get_state_versions(session.id)] == [existing.id]


@pytest.mark.asyncio
async def test_guided_start_atomic_seed_persists_and_settles_empty_session(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    fence = await _claim(service, session.id, kind="guided_start")
    state_data = CompositionStateData(
        composer_meta={"guided_session": {"schema_version": 8}},
        is_valid=True,
    )

    outcome = await service.seed_or_complete_guided_start_operation(
        fence,
        state=state_data,
        provenance="session_seed",
        actor="route",
        response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
    )

    assert isinstance(outcome, GuidedStartStateSeeded)
    assert await service.get_current_state(session.id) == outcome.state
    operation = await service.get_guided_operation(
        session_id=session.id,
        operation_id=fence.operation_id,
        kind="guided_start",
        request_hash="a" * 64,
    )
    assert operation == GuidedOperationCompleted(
        result=GuidedCompositionStateResult(state_id=outcome.state.id),
        response_hash=stable_hash({"state_id": str(outcome.state.id)}),
    )


@pytest.mark.asyncio
async def test_guided_start_atomic_seed_converges_exact_existing_guided_head(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    existing = await service.save_composition_state(
        session.id,
        CompositionStateData(
            composer_meta={"guided_session": {"schema_version": 8}},
            is_valid=True,
        ),
        provenance="session_seed",
    )
    fence = await _claim(service, session.id, kind="guided_start")

    def guided_response_hash(state):
        assert state.composer_meta is not None
        assert "guided_session" in state.composer_meta
        return stable_hash({"state_id": str(state.id)})

    outcome = await service.seed_or_complete_guided_start_operation(
        fence,
        state=CompositionStateData(is_valid=True),
        provenance="session_seed",
        actor="route",
        response_hash_factory=guided_response_hash,
    )

    assert outcome == GuidedStartStateConverged(state=existing)
    assert [state.id for state in await service.get_state_versions(session.id)] == [existing.id]


@pytest.mark.asyncio
async def test_guided_start_atomic_seed_does_not_treat_generic_integrity_error_as_convergence(service) -> None:
    session = await service.create_session("alice", "Pipeline", "local")
    freeform = await service.save_composition_state(
        session.id,
        CompositionStateData(is_valid=True),
        provenance="post_compose",
    )
    fence = await _claim(service, session.id, kind="guided_start")

    def reject_freeform(_state):
        raise AuditIntegrityError("current head is not a valid guided checkpoint")

    with pytest.raises(AuditIntegrityError, match="not a valid guided checkpoint"):
        await service.seed_or_complete_guided_start_operation(
            fence,
            state=CompositionStateData(
                composer_meta={"guided_session": {"schema_version": 8}},
                is_valid=True,
            ),
            provenance="session_seed",
            actor="route",
            response_hash_factory=reject_freeform,
        )

    assert [state.id for state in await service.get_state_versions(session.id)] == [freeform.id]


@pytest.mark.asyncio
async def test_guided_start_atomic_seed_serializes_two_sqlite_services(file_engine) -> None:
    service_a = SessionServiceImpl(file_engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test.a"))
    service_b = SessionServiceImpl(file_engine, telemetry=build_sessions_telemetry(), log=structlog.get_logger("test.b"))
    session = await service_a.create_session("alice", "Pipeline", "local")
    fence_a = await _claim(
        service_a,
        session.id,
        operation_id="00000000-0000-4000-8000-000000000011",
        kind="guided_start",
        request_hash="b" * 64,
    )
    fence_b = await _claim(
        service_b,
        session.id,
        operation_id="00000000-0000-4000-8000-000000000012",
        kind="guided_start",
        request_hash="c" * 64,
    )
    state_data = CompositionStateData(
        composer_meta={"guided_session": {"schema_version": 8}},
        is_valid=True,
    )

    outcomes = await asyncio.gather(
        service_a.seed_or_complete_guided_start_operation(
            fence_a,
            state=state_data,
            provenance="session_seed",
            actor="route-a",
            response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
        ),
        service_b.seed_or_complete_guided_start_operation(
            fence_b,
            state=state_data,
            provenance="session_seed",
            actor="route-b",
            response_hash_factory=lambda state: stable_hash({"state_id": str(state.id)}),
        ),
    )

    assert {type(outcome) for outcome in outcomes} == {
        GuidedStartStateSeeded,
        GuidedStartStateConverged,
    }
    assert outcomes[0].state.id == outcomes[1].state.id
    assert len(await service_a.get_state_versions(session.id)) == 1
