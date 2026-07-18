"""Guided-operation reservation, replay, and stale-worker fencing."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
import structlog
from pydantic import BaseModel, ConfigDict
from sqlalchemy import insert, select, update

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.hashing import stable_hash
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.guided_operations import guided_operation_request_hash
from elspeth.web.sessions.models import (
    chat_messages_table,
    composition_proposals_table,
    composition_states_table,
    guided_operation_events_table,
    guided_operations_table,
)
from elspeth.web.sessions.protocol import (
    CompositionStateData,
    GuidedCompositionStateResult,
    GuidedOperationActive,
    GuidedOperationClaimed,
    GuidedOperationCompleted,
    GuidedOperationConflictError,
    GuidedOperationFailed,
    GuidedOperationFence,
    GuidedOperationFenceLostError,
    GuidedOperationTakenOver,
    GuidedProposalResult,
    GuidedSessionResult,
)
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


class _StrictRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    operation_id: str
    message: str
    include_defaults: bool = True


def _service(engine) -> SessionServiceImpl:
    return SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test.guided-operations"),
    )


@pytest.fixture
def file_engine(tmp_path: Path):
    engine = create_session_engine(f"sqlite:///{tmp_path / 'sessions.db'}")
    initialize_session_schema(engine)
    try:
        yield engine
    finally:
        engine.dispose()


async def _create_session(service: SessionServiceImpl) -> UUID:
    return (await service.create_session("alice", "Guided operation", "local")).id


def test_fence_lost_error_never_retains_or_logs_lease_token(caplog: pytest.LogCaptureFixture) -> None:
    secret = "lease-token-must-not-escape"
    fence = GuidedOperationFence(
        session_id=uuid4(),
        operation_id="operation-safe-error",
        lease_token=secret,
        attempt=7,
    )
    error = GuidedOperationFenceLostError(fence)

    assert error.args == ("Guided operation fence is no longer current",)
    assert error.session_id == fence.session_id
    assert error.operation_id == fence.operation_id
    assert error.attempt == fence.attempt
    assert not hasattr(error, "fence")
    assert secret not in str(error)
    assert secret not in repr(error)
    assert secret not in repr(vars(error))
    with caplog.at_level(logging.ERROR):
        logging.getLogger("test.guided-fence").error(
            "guided operation failed",
            exc_info=(type(error), error, error.__traceback__),
        )
    assert secret not in caplog.text


def _expire_operation(engine, *, session_id: UUID, operation_id: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            update(guided_operations_table)
            .where(
                guided_operations_table.c.session_id == str(session_id),
                guided_operations_table.c.operation_id == operation_id,
            )
            .values(lease_expires_at=datetime.now(UTC) - timedelta(minutes=1))
        )


def _seed_state(engine, *, session_id: UUID) -> UUID:
    state_id = uuid4()
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(composition_states_table).values(
                id=str(state_id),
                session_id=str(session_id),
                version=1,
                is_valid=False,
                provenance="session_seed",
                created_at=now,
            )
        )
    return state_id


def _seed_proposal(engine, *, session_id: UUID) -> UUID:
    proposal_id = uuid4()
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(
            insert(composition_proposals_table).values(
                id=str(proposal_id),
                session_id=str(session_id),
                tool_call_id=f"guided-plan-{proposal_id}",
                tool_name="set_pipeline",
                status="pending",
                summary="Guided plan",
                rationale="Guided operation replay locator",
                affects=[],
                arguments_json={},
                arguments_redacted_json={},
                created_at=now,
                updated_at=now,
            )
        )
    return proposal_id


def test_request_hash_excludes_operation_id_and_materializes_defaults() -> None:
    session_id = uuid4()
    first = _StrictRequest(operation_id="client-a", message="hello")
    retry = _StrictRequest(operation_id="client-b", message="hello", include_defaults=True)

    assert guided_operation_request_hash(session_id=session_id, kind="guided_chat", request=first) == guided_operation_request_hash(
        session_id=session_id,
        kind="guided_chat",
        request=retry,
    )
    assert guided_operation_request_hash(session_id=session_id, kind="guided_chat", request=first) != guided_operation_request_hash(
        session_id=uuid4(),
        kind="guided_chat",
        request=first,
    )
    assert guided_operation_request_hash(session_id=session_id, kind="guided_chat", request=first) == stable_hash(
        {
            "schema": "guided-operation-request.v1",
            "session_id": str(session_id),
            "kind": "guided_chat",
            "request": {"message": "hello", "include_defaults": True},
        }
    )
    assert guided_operation_request_hash(session_id=session_id, kind="guided_chat", request=first) != guided_operation_request_hash(
        session_id=session_id,
        kind="guided_respond",
        request=first,
    )


def test_operation_decoders_reject_kind_locator_drift_and_status_residue() -> None:
    completed = {
        "status": "completed",
        "kind": "session_fork",
        "result_kind": "composition_state",
        "result_state_id": str(uuid4()),
        "result_session_id": None,
        "proposal_id": None,
        "response_hash": "a" * 64,
        "failure_code": None,
        "lease_token": None,
        "lease_expires_at": None,
        "settled_at": datetime.now(UTC),
    }
    failed = {
        **completed,
        "status": "failed",
        "result_kind": None,
        "result_state_id": None,
        "response_hash": "b" * 64,
        "failure_code": "provider_timeout",
    }

    with pytest.raises(AuditIntegrityError, match=r"kind.*locator"):
        SessionServiceImpl._guided_terminal_outcome(cast("Any", completed))
    with pytest.raises(AuditIntegrityError, match=r"failure.*residue"):
        SessionServiceImpl._guided_terminal_outcome(cast("Any", failed))
    with pytest.raises(AuditIntegrityError, match=r"in-progress.*lease"):
        SessionServiceImpl._guided_in_progress_expiry(
            cast(
                "Any",
                {
                    "kind": "guided_start",
                    "lease_token": "token",
                    "lease_expires_at": None,
                    "settled_at": None,
                    "result_kind": None,
                    "result_state_id": None,
                    "result_session_id": None,
                    "proposal_id": None,
                    "response_hash": None,
                    "failure_code": None,
                },
            )
        )


@pytest.mark.asyncio
async def test_claim_active_join_and_request_conflict_without_mutation(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    request_hash = "a" * 64

    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="operation-1",
        kind="guided_start",
        request_hash=request_hash,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    assert claimed.fence.attempt == 1

    active = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="operation-1",
        kind="guided_start",
        request_hash=request_hash,
        actor="worker-b",
        lease_seconds=30,
    )
    assert isinstance(active, GuidedOperationActive)
    assert active.attempt == 1

    with pytest.raises(GuidedOperationConflictError):
        await service.reserve_guided_operation(
            session_id=session_id,
            operation_id="operation-1",
            kind="guided_chat",
            request_hash="b" * 64,
            actor="worker-b",
            lease_seconds=30,
        )

    with file_engine.connect() as conn:
        row = conn.execute(select(guided_operations_table)).one()
        events = conn.execute(select(guided_operation_events_table)).all()
    assert row.kind == "guided_start"
    assert row.request_hash == request_hash
    assert row.attempt == 1
    assert [event.event_kind for event in events] == ["claimed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["reserve", "get"])
async def test_corrupt_in_progress_row_is_integrity_error_before_conflict(file_engine, surface: str) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    operation_id = "operation-corrupt-active"
    request_hash = "7" * 64
    await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash=request_hash,
        actor="worker-a",
        lease_seconds=30,
    )
    with file_engine.begin() as conn:
        conn.exec_driver_sql("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            update(guided_operations_table)
            .where(
                guided_operations_table.c.session_id == str(session_id),
                guided_operations_table.c.operation_id == operation_id,
            )
            .values(kind="corrupt-kind", attempt=0)
        )

    with pytest.raises(AuditIntegrityError, match=r"guided operation.*(kind|request_hash|attempt)"):
        if surface == "reserve":
            await service.reserve_guided_operation(
                session_id=session_id,
                operation_id=operation_id,
                kind="guided_chat",
                request_hash="8" * 64,
                actor="worker-b",
                lease_seconds=30,
            )
        else:
            await service.get_guided_operation(
                session_id=session_id,
                operation_id=operation_id,
                kind="guided_chat",
                request_hash="8" * 64,
            )


@pytest.mark.asyncio
async def test_corrupt_terminal_row_is_integrity_error_before_conflict(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    state_id = _seed_state(file_engine, session_id=session_id)
    operation_id = "operation-corrupt-terminal"
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash="9" * 64,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    await service.complete_guided_operation(
        claimed.fence,
        result=GuidedCompositionStateResult(state_id=state_id),
        response_hash="a" * 64,
        actor="worker-a",
    )
    with file_engine.begin() as conn:
        conn.exec_driver_sql("DROP TRIGGER trg_guided_operations_terminal_immutable")
        conn.exec_driver_sql("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            update(guided_operations_table)
            .where(
                guided_operations_table.c.session_id == str(session_id),
                guided_operations_table.c.operation_id == operation_id,
            )
            .values(kind="guided_chat", response_hash="corrupt-response-hash")
        )

    with pytest.raises(AuditIntegrityError, match=r"guided operation"):
        await service.reserve_guided_operation(
            session_id=session_id,
            operation_id=operation_id,
            kind="guided_respond",
            request_hash="b" * 64,
            actor="worker-b",
            lease_seconds=30,
        )


@pytest.mark.asyncio
async def test_expired_takeover_rotates_fence_and_old_worker_cannot_bind_or_settle(file_engine) -> None:
    service_a = _service(file_engine)
    service_b = _service(file_engine)
    session_id = await _create_session(service_a)
    state_id = _seed_state(file_engine, session_id=session_id)
    request_hash = "c" * 64
    operation_id = "operation-race"
    first = await service_a.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash=request_hash,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(first, GuidedOperationClaimed)
    _expire_operation(file_engine, session_id=session_id, operation_id=operation_id)

    takeover = await service_b.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash=request_hash,
        actor="worker-b",
        lease_seconds=30,
    )
    assert isinstance(takeover, GuidedOperationTakenOver)
    assert takeover.prior_attempt == 1
    assert takeover.fence.attempt == 2
    assert takeover.fence.lease_token != first.fence.lease_token
    with pytest.raises(GuidedOperationFenceLostError):
        await service_a.bind_guided_operation(first.fence, result_state_id=state_id)
    with pytest.raises(GuidedOperationFenceLostError):
        await service_a.complete_guided_operation(
            first.fence,
            result=GuidedCompositionStateResult(state_id=state_id),
            response_hash="d" * 64,
            actor="worker-a",
        )
    await service_b.bind_guided_operation(takeover.fence, result_state_id=state_id)
    completed = await service_b.complete_guided_operation(
        takeover.fence,
        result=GuidedCompositionStateResult(state_id=state_id),
        response_hash="d" * 64,
        actor="worker-b",
    )
    assert completed == GuidedOperationCompleted(
        result=GuidedCompositionStateResult(state_id=state_id),
        response_hash="d" * 64,
    )

    replay = await service_a.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash=request_hash,
        actor="worker-c",
        lease_seconds=30,
    )
    assert replay == GuidedOperationCompleted(
        result=GuidedCompositionStateResult(state_id=state_id),
        response_hash="d" * 64,
    )
    with file_engine.connect() as conn:
        row = conn.execute(select(guided_operations_table)).one()
        events = conn.execute(select(guided_operation_events_table).order_by(guided_operation_events_table.c.sequence)).all()
    assert row.attempt == 2
    assert row.result_state_id == str(state_id)
    assert [event.event_kind for event in events] == ["claimed", "taken_over", "completed"]


@pytest.mark.asyncio
async def test_guided_seed_rejects_cross_service_head_drift_before_writes(file_engine) -> None:
    service_a = _service(file_engine)
    service_b = _service(file_engine)
    session_id = await _create_session(service_a)
    claim = await service_a.reserve_guided_operation(
        session_id=session_id,
        operation_id="operation-head-race",
        kind="guided_convert",
        request_hash="e" * 64,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(claim, GuidedOperationClaimed)

    winning_state = await service_b.save_composition_state(
        session_id,
        CompositionStateData(is_valid=True, composer_meta={}),
        provenance="session_seed",
    )

    with pytest.raises(AuditIntegrityError, match="current state"):
        await service_a.save_state_for_guided_operation(
            claim.fence,
            expected_current_state_id=None,
            expected_current_state_version=None,
            state=CompositionStateData(is_valid=True, composer_meta={"guided_session": {"schema_version": 8}}),
            provenance="session_seed",
            actor="worker-a",
            system_message="Wrong stale breadcrumb.",
            response_hash_factory=lambda record: stable_hash({"state_id": str(record.id)}),
        )

    assert [record.id for record in await service_a.get_state_versions(session_id)] == [winning_state.id]
    with file_engine.connect() as conn:
        message_count = conn.execute(select(chat_messages_table.c.id).where(chat_messages_table.c.session_id == str(session_id))).all()
    assert message_count == []


@pytest.mark.asyncio
async def test_existing_guided_settlement_rejects_cross_service_head_drift(file_engine) -> None:
    service_a = _service(file_engine)
    service_b = _service(file_engine)
    session_id = await _create_session(service_a)
    observed = await service_a.save_composition_state(
        session_id,
        CompositionStateData(is_valid=True),
        provenance="session_seed",
    )
    claim = await service_a.reserve_guided_operation(
        session_id=session_id,
        operation_id="operation-existing-head-race",
        kind="guided_convert",
        request_hash="f" * 64,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(claim, GuidedOperationClaimed)
    winning_state = await service_b.save_composition_state(
        session_id,
        CompositionStateData(is_valid=True),
        provenance="session_seed",
    )

    with pytest.raises(AuditIntegrityError, match="current state"):
        await service_a.complete_existing_state_guided_operation(
            claim.fence,
            state_id=observed.id,
            expected_current_state_id=observed.id,
            expected_current_state_version=observed.version,
            actor="worker-a",
            response_hash_factory=lambda record: stable_hash({"state_id": str(record.id)}),
        )

    assert [record.id for record in await service_a.get_state_versions(session_id)] == [observed.id, winning_state.id]


@pytest.mark.asyncio
async def test_renew_after_expiry_cannot_revive_fence(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="operation-expired",
        kind="guided_chat",
        request_hash="e" * 64,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    _expire_operation(file_engine, session_id=session_id, operation_id="operation-expired")

    with pytest.raises(GuidedOperationFenceLostError):
        await service.renew_guided_operation(claimed.fence, actor="worker-a", lease_seconds=30)

    with file_engine.connect() as conn:
        events = conn.execute(select(guided_operation_events_table)).all()
    assert [event.event_kind for event in events] == ["claimed"]


@pytest.mark.asyncio
async def test_live_renewal_uses_same_fence_and_appends_evidence(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="operation-renewed",
        kind="guided_chat",
        request_hash="5" * 64,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(claimed, GuidedOperationClaimed)

    assert await service.renew_guided_operation(claimed.fence, actor="worker-a", lease_seconds=60) == claimed.fence
    active = await service.get_guided_operation(
        session_id=session_id,
        operation_id="operation-renewed",
        kind="guided_chat",
        request_hash="5" * 64,
    )
    assert isinstance(active, GuidedOperationActive)
    assert active.expired is False
    assert active.lease_expires_at > claimed.lease_expires_at
    with file_engine.connect() as conn:
        events = conn.execute(select(guided_operation_events_table).order_by(guided_operation_events_table.c.sequence)).all()
    assert [event.event_kind for event in events] == ["claimed", "renewed"]


@pytest.mark.asyncio
async def test_connection_fence_helper_rejects_check_outside_session_write_lock(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="operation-lock-precondition",
        kind="guided_start",
        request_hash="6" * 64,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(claimed, GuidedOperationClaimed)

    with file_engine.begin() as conn, pytest.raises(RuntimeError, match="_session_write_lock"):
        service.require_guided_operation_fence_on_connection(conn, claimed.fence)


@pytest.mark.asyncio
async def test_file_backed_sqlite_concurrent_takeover_has_one_winner(file_engine) -> None:
    service_a = _service(file_engine)
    service_b = _service(file_engine)
    service_c = _service(file_engine)
    session_id = await _create_session(service_a)
    state_id = _seed_state(file_engine, session_id=session_id)
    operation_id = "operation-contended"
    request_hash = "3" * 64
    first = await service_a.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash=request_hash,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(first, GuidedOperationClaimed)
    _expire_operation(file_engine, session_id=session_id, operation_id=operation_id)

    barrier = threading.Barrier(2)

    def contend(service: SessionServiceImpl, actor: str):
        barrier.wait()
        return asyncio.run(
            service.reserve_guided_operation(
                session_id=session_id,
                operation_id=operation_id,
                kind="guided_start",
                request_hash=request_hash,
                actor=actor,
                lease_seconds=30,
            )
        )

    outcomes = await asyncio.gather(
        asyncio.to_thread(contend, service_b, "worker-b"),
        asyncio.to_thread(contend, service_c, "worker-c"),
    )
    winners = [outcome for outcome in outcomes if isinstance(outcome, GuidedOperationTakenOver)]
    active = [outcome for outcome in outcomes if isinstance(outcome, GuidedOperationActive)]
    assert len(winners) == len(active) == 1

    winner = winners[0]
    await service_b.complete_guided_operation(
        winner.fence,
        result=GuidedCompositionStateResult(state_id=state_id),
        response_hash="4" * 64,
        actor="winner",
    )
    with file_engine.connect() as conn:
        events = conn.execute(select(guided_operation_events_table).order_by(guided_operation_events_table.c.sequence)).all()
    assert [event.event_kind for event in events] == ["claimed", "taken_over", "completed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "result_factory"),
    [
        ("guided_start", "state"),
        ("guided_respond", "state_with_proposal"),
        ("guided_chat", "state_with_proposal"),
        ("guided_convert", "state"),
        ("guided_reenter", "state"),
        ("state_revert", "state"),
        ("guided_plan", "proposal"),
        ("session_fork", "session"),
    ],
)
async def test_terminal_replay_uses_closed_per_kind_locator(file_engine, kind: str, result_factory: str) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    state_id = _seed_state(file_engine, session_id=session_id)
    proposal_id = _seed_proposal(file_engine, session_id=session_id)
    child_session_id = (
        await service.create_session(
            "alice",
            "Fork result",
            "local",
            forked_from_session_id=session_id,
        )
    ).id
    result = {
        "state": GuidedCompositionStateResult(state_id=state_id),
        "state_with_proposal": GuidedCompositionStateResult(state_id=state_id, proposal_id=proposal_id),
        "proposal": GuidedProposalResult(proposal_id=proposal_id),
        "session": GuidedSessionResult(session_id=child_session_id),
    }[result_factory]
    operation_id = f"operation-{kind}"
    request_hash = "f" * 64
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind=kind,
        request_hash=request_hash,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(claimed, GuidedOperationClaimed)

    if isinstance(result, GuidedProposalResult):
        await service.bind_guided_operation(claimed.fence, proposal_id=result.proposal_id, result_state_id=state_id)
    elif isinstance(result, GuidedSessionResult):
        await service.bind_guided_operation(claimed.fence, result_session_id=result.session_id)

    completed = await service.complete_guided_operation(
        claimed.fence,
        result=result,
        response_hash="1" * 64,
        actor="worker-a",
    )
    replay = await service.get_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind=kind,
        request_hash=request_hash,
    )
    assert completed == replay == GuidedOperationCompleted(result=result, response_hash="1" * 64)


@pytest.mark.asyncio
@pytest.mark.parametrize("target_kind", ["ordinary", "unrelated", "cross_user"])
async def test_fork_completion_rejects_target_without_exact_lineage_and_principal_custody(file_engine, target_kind: str) -> None:
    service = _service(file_engine)
    parent_id = await _create_session(service)
    if target_kind == "ordinary":
        target = await service.create_session("alice", "Ordinary", "local")
    elif target_kind == "unrelated":
        other_parent = await service.create_session("alice", "Other parent", "local")
        target = await service.create_session(
            "alice",
            "Wrong lineage",
            "local",
            forked_from_session_id=other_parent.id,
        )
    else:
        target = await service.create_session(
            "bob",
            "Cross-user child",
            "local",
            forked_from_session_id=parent_id,
        )
    claimed = await service.reserve_guided_operation(
        session_id=parent_id,
        operation_id=f"operation-fork-{target_kind}",
        kind="session_fork",
        request_hash="c" * 64,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(claimed, GuidedOperationClaimed)

    with pytest.raises(AuditIntegrityError, match=r"fork result session.*custody"):
        await service.complete_guided_operation(
            claimed.fence,
            result=GuidedSessionResult(session_id=target.id),
            response_hash="d" * 64,
            actor="worker-a",
        )

    active = await service.get_guided_operation(
        session_id=parent_id,
        operation_id=f"operation-fork-{target_kind}",
        kind="session_fork",
        request_hash="c" * 64,
    )
    assert isinstance(active, GuidedOperationActive)


@pytest.mark.asyncio
async def test_failed_operation_replays_only_closed_safe_failure(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    state_id = _seed_state(file_engine, session_id=session_id)
    proposal_id = _seed_proposal(file_engine, session_id=session_id)
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="operation-failed",
        kind="guided_chat",
        request_hash="2" * 64,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    originating_message = await service.add_message(
        session_id,
        "user",
        "Original guided request",
        writer_principal="route_user_message",
    )
    await service.bind_guided_operation(
        claimed.fence,
        originating_message_id=originating_message.id,
        proposal_id=proposal_id,
        result_state_id=state_id,
    )
    failed = await service.fail_guided_operation(
        claimed.fence,
        failure_code="provider_timeout",
        actor="worker-a",
    )
    assert failed == GuidedOperationFailed(failure_code="provider_timeout")

    replay = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="operation-failed",
        kind="guided_chat",
        request_hash="2" * 64,
        actor="worker-b",
        lease_seconds=30,
    )
    assert replay == failed

    with file_engine.connect() as conn:
        row = conn.execute(select(guided_operations_table)).one()
    assert row.originating_message_id == str(originating_message.id)
    assert row.proposal_id is None
    assert row.result_state_id is None
    assert row.result_session_id is None
