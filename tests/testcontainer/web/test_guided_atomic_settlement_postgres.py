"""Real PostgreSQL parity proofs for atomic guided RESPOND settlement."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import structlog
from sqlalchemy import Engine, select, update
from testcontainers.postgres import PostgresContainer

from elspeth.contracts.composer_llm_audit import (
    ComposerChatInitiator,
    ComposerChatTurn,
    ComposerChatTurnStatus,
)
from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import chat_messages_table, composition_states_table, guided_operations_table
from elspeth.web.sessions.protocol import (
    CompositionStateData,
    GuidedAuditEvidence,
    GuidedOperationClaimed,
    GuidedOperationFence,
    GuidedOperationFenceLostError,
    GuidedOperationTakenOver,
    GuidedOriginatingUserMessageDraft,
    GuidedResponseDescriptor,
    GuidedStateOperationCommand,
)
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry

pytestmark = pytest.mark.testcontainer


@pytest.fixture(scope="module")
def postgres_engine() -> Iterator[Engine]:
    with PostgresContainer("postgres:16-alpine", driver="psycopg") as postgres:
        engine = create_session_engine(postgres.get_connection_url())
        initialize_session_schema(engine)
        try:
            yield engine
        finally:
            engine.dispose()


@pytest.fixture
def postgres_service(postgres_engine: Engine, tmp_path: Path) -> SessionServiceImpl:
    return SessionServiceImpl(
        postgres_engine,
        data_dir=tmp_path,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.guided.atomic.postgres"),
    )


def _command(
    fence: GuidedOperationFence,
    *,
    with_messages: bool = False,
) -> GuidedStateOperationCommand:
    now = datetime.now(UTC)
    evidence = (
        GuidedAuditEvidence(
            chat_turns=(
                ComposerChatTurn(
                    step="step_1_source",
                    initiator=ComposerChatInitiator.USER,
                    chat_turn_seq=0,
                    user_message_hash="a" * 64,
                    assistant_message_hash="b" * 64,
                    latency_ms=1,
                    model="test-model",
                    status=ComposerChatTurnStatus.SUCCESS,
                    started_at=now,
                    finished_at=now,
                ),
            )
        )
        if with_messages
        else GuidedAuditEvidence()
    )
    return GuidedStateOperationCommand(
        fence=fence,
        expected_current_state_id=None,
        expected_current_state_version=None,
        expected_current_content_hash=None,
        state_id=uuid4(),
        state=CompositionStateData(
            is_valid=False,
            composer_meta={"guided_session": GuidedSession.initial().to_dict()},
        ),
        provenance="convergence_persist",
        actor="worker",
        response=GuidedResponseDescriptor(kind="guided_respond", next_turn=None, assistant_turn_seq=None),
        audit_evidence=evidence,
        originating_message=(GuidedOriginatingUserMessageDraft(message_id=uuid4(), content="continue") if with_messages else None),
    )


@pytest.mark.asyncio
async def test_postgres_commits_state_messages_and_operation_as_one_cohort(
    postgres_service: SessionServiceImpl,
    postgres_engine: Engine,
) -> None:
    session_id = (await postgres_service.create_session("alice", "PG guided cohort", "local")).id
    claimed = await postgres_service.reserve_guided_operation(
        session_id=session_id,
        operation_id="pg-respond-cohort",
        kind="guided_respond",
        request_hash="c" * 64,
        actor="worker",
        lease_seconds=60,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    command = _command(claimed.fence, with_messages=True)

    settlement = await postgres_service.settle_guided_state_operation(command)

    with postgres_engine.connect() as conn:
        states = conn.execute(select(composition_states_table).where(composition_states_table.c.session_id == str(session_id))).all()
        messages = conn.execute(
            select(chat_messages_table)
            .where(chat_messages_table.c.session_id == str(session_id))
            .order_by(chat_messages_table.c.sequence_no)
        ).all()
        operation = conn.execute(select(guided_operations_table).where(guided_operations_table.c.session_id == str(session_id))).one()
    assert [row.id for row in states] == [str(command.state_id)]
    assert [row.role for row in messages] == ["user", "audit"]
    assert operation.status == "completed"
    assert operation.result_state_id == str(settlement.result_state.id)
    assert operation.response_hash == settlement.response_hash


@pytest.mark.asyncio
async def test_postgres_stale_fence_attempt_cannot_write_after_takeover(
    postgres_service: SessionServiceImpl,
    postgres_engine: Engine,
) -> None:
    session_id = (await postgres_service.create_session("alice", "PG stale fence", "local")).id
    operation_id = "pg-respond-stale"
    first = await postgres_service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_respond",
        request_hash="d" * 64,
        actor="worker-a",
        lease_seconds=60,
    )
    assert isinstance(first, GuidedOperationClaimed)
    with postgres_engine.begin() as conn:
        conn.execute(
            update(guided_operations_table)
            .where(guided_operations_table.c.session_id == str(session_id))
            .values(lease_expires_at=datetime.now(UTC) - timedelta(minutes=1))
        )
    second = await postgres_service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_respond",
        request_hash="d" * 64,
        actor="worker-b",
        lease_seconds=60,
    )
    assert isinstance(second, GuidedOperationTakenOver)

    with pytest.raises(GuidedOperationFenceLostError):
        await postgres_service.settle_guided_state_operation(_command(first.fence))
    winning = _command(second.fence)
    settlement = await postgres_service.settle_guided_state_operation(winning)

    with postgres_engine.connect() as conn:
        states = conn.execute(select(composition_states_table).where(composition_states_table.c.session_id == str(session_id))).all()
    assert [row.id for row in states] == [str(winning.state_id)]
    assert settlement.result_state.id == winning.state_id
