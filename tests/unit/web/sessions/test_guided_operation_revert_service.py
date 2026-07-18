"""Atomic state-revert settlement under a guided-operation fence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import structlog
from sqlalchemy import func, select, update
from sqlalchemy.pool import StaticPool

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


async def _claim(service, session_id, operation_id="00000000-0000-4000-8000-000000000001"):
    outcome = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="state_revert",
        request_hash="a" * 64,
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
