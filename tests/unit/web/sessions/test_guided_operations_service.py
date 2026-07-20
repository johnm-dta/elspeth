"""Guided-operation reservation, replay, and stale-worker fencing."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
import structlog
from pydantic import BaseModel, ConfigDict
from sqlalchemy import delete, func, insert, select, update

from elspeth.contracts.composer_llm_audit import ComposerLLMCall, ComposerLLMCallStatus
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import stable_hash
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.guided_operations import guided_operation_request_hash
from elspeth.web.sessions.models import (
    chat_messages_table,
    composition_proposals_table,
    guided_operation_admission_blocks_table,
    guided_operation_events_table,
    guided_operations_table,
    sessions_table,
)
from elspeth.web.sessions.protocol import (
    CompositionStateData,
    GuidedAuditEvidence,
    GuidedCompositionStateResult,
    GuidedOperationActive,
    GuidedOperationClaimed,
    GuidedOperationCompleted,
    GuidedOperationConflictError,
    GuidedOperationFailed,
    GuidedOperationFailureCommand,
    GuidedOperationFence,
    GuidedOperationFenceLostError,
    GuidedOperationSettlementConflictError,
    GuidedOperationTakenOver,
    GuidedPipelineProposalResult,
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


@pytest.fixture(params=("sqlite", "postgres"))
def durable_engine(request: pytest.FixtureRequest, tmp_path: Path):
    """Production-shaped SQLite plus opt-in PostgreSQL reconciliation races."""
    if request.param == "postgres":
        url = os.environ.get("ELSPETH_TEST_POSTGRES_URL")
        if url is None:
            pytest.skip("ELSPETH_TEST_POSTGRES_URL is required for guided reconciliation races")
        engine = create_session_engine(url)
    else:
        engine = create_session_engine(f"sqlite:///{tmp_path / 'guided-reconciliation-races.db'}")
    initialize_session_schema(engine)
    try:
        yield engine
    finally:
        engine.dispose()


async def _create_session(service: SessionServiceImpl) -> UUID:
    return (await service.create_session("alice", "Guided operation", "local")).id


def _failed_llm_call(marker: str) -> ComposerLLMCall:
    now = datetime.now(UTC)
    return ComposerLLMCall(
        model_requested="test/planner",
        model_returned=None,
        status=ComposerLLMCallStatus.API_ERROR,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        latency_ms=1,
        provider_request_id=None,
        messages_hash=stable_hash({"marker": marker}),
        tools_spec_hash=None,
        declared_tool_names=(),
        started_at=now,
        finished_at=now,
        error_class="ProviderFailure",
        error_message=f"secret-{marker}",
        temperature=None,
        seed=None,
    )


def _failure_command(
    claim: GuidedOperationClaimed,
    *,
    marker: str,
    evidence_count: int = 1,
) -> GuidedOperationFailureCommand:
    return GuidedOperationFailureCommand(
        fence=claim.fence,
        failure_code="operation_failed",
        actor=f"worker-{marker}",
        audit_evidence=GuidedAuditEvidence(llm_calls=tuple(_failed_llm_call(f"{marker}-{index}") for index in range(evidence_count))),
    )


def _expected_failure_audit_cohort(messages) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for message in messages:
        assert message.sequence_no is not None
        rows.append(
            {
                "message_id": str(message.id),
                "sequence_no": message.sequence_no,
                "content_hash": stable_hash(message.content),
                "tool_calls_hash": stable_hash(deep_thaw(message.tool_calls)),
            }
        )
    authority: dict[str, object] = {
        "schema": "guided_failure_audit_cohort.v1",
        "count": len(rows),
        "rows": rows,
    }
    return {**authority, "aggregate_digest": stable_hash(authority)}


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
    with pytest.raises(AttributeError):
        _ = error.fence
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


async def _seed_state(engine, *, session_id: UUID) -> UUID:
    state = await _service(engine).save_composition_state(
        session_id,
        CompositionStateData(is_valid=False),
        provenance="session_seed",
    )
    return state.id


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
async def test_two_empty_session_failures_have_exact_distinct_read_verified_lineage(file_engine) -> None:
    service_a = _service(file_engine)
    service_b = _service(file_engine)
    session_id = await _create_session(service_a)
    claims: list[GuidedOperationClaimed] = []
    for operation_id, request_hash, service in (
        ("failure-operation-a", "a" * 64, service_a),
        ("failure-operation-b", "b" * 64, service_b),
    ):
        claim = await service.reserve_guided_operation(
            session_id=session_id,
            operation_id=operation_id,
            kind="guided_plan",
            request_hash=request_hash,
            actor=f"claim-{operation_id}",
            lease_seconds=30,
        )
        assert isinstance(claim, GuidedOperationClaimed)
        claims.append(claim)

    await asyncio.gather(
        service_a.fail_guided_operation_with_audit(_failure_command(claims[0], marker="a")),
        service_b.fail_guided_operation_with_audit(_failure_command(claims[1], marker="b")),
    )

    messages = await service_a.get_messages(session_id, limit=None)
    assert len(messages) == 2
    lineages = sorted(
        (message.tool_calls[0]["_guided_failure_lineage"] for message in messages if message.tool_calls is not None),
        key=lambda lineage: lineage["operation_id"],
    )
    assert lineages == [
        {
            "schema": "guided_failure_audit_lineage.v1",
            "session_id": str(session_id),
            "operation_id": "failure-operation-a",
            "attempt": 1,
            "request_hash": "a" * 64,
            "cohort_id": stable_hash(
                {
                    "schema": "guided_failure_audit_lineage.v1",
                    "session_id": str(session_id),
                    "operation_id": "failure-operation-a",
                    "attempt": 1,
                    "request_hash": "a" * 64,
                }
            ),
        },
        {
            "schema": "guided_failure_audit_lineage.v1",
            "session_id": str(session_id),
            "operation_id": "failure-operation-b",
            "attempt": 1,
            "request_hash": "b" * 64,
            "cohort_id": stable_hash(
                {
                    "schema": "guided_failure_audit_lineage.v1",
                    "session_id": str(session_id),
                    "operation_id": "failure-operation-b",
                    "attempt": 1,
                    "request_hash": "b" * 64,
                }
            ),
        },
    ]
    with file_engine.connect() as conn:
        failed_events = conn.execute(
            select(
                guided_operation_events_table.c.session_id,
                guided_operation_events_table.c.operation_id,
                guided_operation_events_table.c.attempt,
                guided_operation_events_table.c.request_hash,
                guided_operation_events_table.c.failure_audit_cohort,
            )
            .where(guided_operation_events_table.c.event_kind == "failed")
            .order_by(guided_operation_events_table.c.operation_id)
        ).all()
    messages_by_operation = {
        lineage["operation_id"]: [
            message
            for message in messages
            if message.tool_calls is not None
            and message.tool_calls[0]["_guided_failure_lineage"]["operation_id"] == lineage["operation_id"]
        ]
        for lineage in lineages
    }
    assert [tuple(row[:4]) for row in failed_events] == [
        (str(session_id), "failure-operation-a", 1, "a" * 64),
        (str(session_id), "failure-operation-b", 1, "b" * 64),
    ]
    assert failed_events[0].failure_audit_cohort == _expected_failure_audit_cohort(messages_by_operation["failure-operation-a"])
    assert failed_events[1].failure_audit_cohort == _expected_failure_audit_cohort(messages_by_operation["failure-operation-b"])


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", ["missing", "partial", "attempt", "request_hash", "cross_swap", "ambiguous_event"])
async def test_failure_audit_read_rejects_lineage_tamper(file_engine, tamper: str) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    claims: list[GuidedOperationClaimed] = []
    for operation_id, request_hash in (("tamper-a", "c" * 64), ("tamper-b", "d" * 64)):
        claim = await service.reserve_guided_operation(
            session_id=session_id,
            operation_id=operation_id,
            kind="guided_plan",
            request_hash=request_hash,
            actor=f"claim-{operation_id}",
            lease_seconds=30,
        )
        assert isinstance(claim, GuidedOperationClaimed)
        claims.append(claim)
    await service.fail_guided_operation_with_audit(_failure_command(claims[0], marker="tamper-a"))
    await service.fail_guided_operation_with_audit(_failure_command(claims[1], marker="tamper-b"))

    forged_clone: tuple[str, list[dict[str, Any]]] | None = None
    with file_engine.begin() as conn:
        rows = conn.execute(
            select(
                chat_messages_table.c.id,
                chat_messages_table.c.content,
                chat_messages_table.c.tool_calls,
                chat_messages_table.c.sequence_no,
            )
            .where(chat_messages_table.c.session_id == str(session_id))
            .order_by(chat_messages_table.c.sequence_no)
        ).all()
        first_envelope = dict(rows[0].tool_calls[0])
        second_envelope = dict(rows[1].tool_calls[0])
        if tamper == "missing":
            first_envelope.pop("_guided_failure_lineage")
        elif tamper == "partial":
            first_envelope["_guided_failure_lineage"] = {
                key: value for key, value in first_envelope["_guided_failure_lineage"].items() if key != "cohort_id"
            }
        elif tamper in {"attempt", "request_hash"}:
            forged_lineage = dict(first_envelope["_guided_failure_lineage"])
            forged_lineage[tamper] = 2 if tamper == "attempt" else "e" * 64
            forged_lineage["cohort_id"] = stable_hash({key: value for key, value in forged_lineage.items() if key != "cohort_id"})
            first_envelope["_guided_failure_lineage"] = forged_lineage
            forged_content = json.loads(rows[0].content)
            forged_content["_guided_failure_lineage"] = forged_lineage
            forged_clone = (json.dumps(forged_content), [first_envelope])
        elif tamper == "cross_swap":
            first_envelope = second_envelope
        elif tamper == "ambiguous_event":
            failed_event = conn.execute(
                select(guided_operation_events_table)
                .where(guided_operation_events_table.c.session_id == str(session_id))
                .where(guided_operation_events_table.c.operation_id == "tamper-a")
                .where(guided_operation_events_table.c.event_kind == "failed")
            ).one()
            conn.execute(
                insert(guided_operation_events_table).values(
                    session_id=failed_event.session_id,
                    operation_id=failed_event.operation_id,
                    sequence=failed_event.sequence + 1,
                    event_kind="failed",
                    actor="tamper",
                    attempt=failed_event.attempt,
                    prior_attempt=None,
                    lease_expires_at=None,
                    request_hash=failed_event.request_hash,
                    failure_audit_cohort=failed_event.failure_audit_cohort,
                    occurred_at=datetime.now(UTC),
                )
            )
        if tamper not in {"attempt", "request_hash", "ambiguous_event"}:
            conn.execute(update(chat_messages_table).where(chat_messages_table.c.id == rows[0].id).values(tool_calls=[first_envelope]))

    if forged_clone is not None:
        content, tool_calls = forged_clone
        await service.add_message(
            session_id,
            "audit",
            content,
            writer_principal="compose_loop",
            tool_calls=tool_calls,
        )

    with pytest.raises(AuditIntegrityError, match="guided failure audit lineage"):
        await service.get_messages(session_id, limit=None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tamper",
    [
        "content",
        "tool_calls_error_class",
        "extra_clone",
        "missing_row",
        "reordered_sequences",
        "modified_event_commitment",
        "duplicate_event_row_commitment",
        "missing_event_commitment",
        "ambiguous_event_commitment",
    ],
)
async def test_failure_audit_read_rejects_exact_cohort_tamper(file_engine, tamper: str) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    claim = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="exact-cohort-tamper",
        kind="guided_plan",
        request_hash="2" * 64,
        actor="claim-exact-cohort",
        lease_seconds=30,
    )
    assert isinstance(claim, GuidedOperationClaimed)
    await service.fail_guided_operation_with_audit(_failure_command(claim, marker="exact-cohort", evidence_count=2))
    assert len(await service.get_messages(session_id, limit=None)) == 2

    cloned_row: tuple[str, list[dict[str, Any]], str] | None = None
    with file_engine.begin() as conn:
        rows = conn.execute(
            select(chat_messages_table)
            .where(chat_messages_table.c.session_id == str(session_id))
            .order_by(chat_messages_table.c.sequence_no)
        ).all()
        event = conn.execute(
            select(guided_operation_events_table)
            .where(guided_operation_events_table.c.session_id == str(session_id))
            .where(guided_operation_events_table.c.operation_id == "exact-cohort-tamper")
            .where(guided_operation_events_table.c.event_kind == "failed")
        ).one()
        if tamper == "content":
            conn.exec_driver_sql("DROP TRIGGER trg_chat_messages_immutable_content")
            content = json.loads(rows[0].content)
            content["error_class"] = "ForgedFailure"
            conn.execute(update(chat_messages_table).where(chat_messages_table.c.id == rows[0].id).values(content=json.dumps(content)))
        elif tamper == "tool_calls_error_class":
            tool_calls = copy.deepcopy(rows[0].tool_calls)
            tool_calls[0]["call"]["error_class"] = "ForgedFailure"
            conn.execute(update(chat_messages_table).where(chat_messages_table.c.id == rows[0].id).values(tool_calls=tool_calls))
        elif tamper == "extra_clone":
            cloned_row = (rows[0].content, rows[0].tool_calls, rows[0].writer_principal)
        elif tamper == "missing_row":
            conn.exec_driver_sql("DROP TRIGGER trg_chat_messages_no_delete")
            conn.execute(delete(chat_messages_table).where(chat_messages_table.c.id == rows[0].id))
        elif tamper == "reordered_sequences":
            temporary_sequence = rows[-1].sequence_no + 100
            conn.execute(update(chat_messages_table).where(chat_messages_table.c.id == rows[0].id).values(sequence_no=temporary_sequence))
            conn.execute(update(chat_messages_table).where(chat_messages_table.c.id == rows[1].id).values(sequence_no=rows[0].sequence_no))
            conn.execute(update(chat_messages_table).where(chat_messages_table.c.id == rows[0].id).values(sequence_no=rows[1].sequence_no))
        elif tamper in {
            "modified_event_commitment",
            "duplicate_event_row_commitment",
            "missing_event_commitment",
        }:
            conn.exec_driver_sql("DROP TRIGGER trg_guided_operation_events_no_update")
            conn.exec_driver_sql("PRAGMA ignore_check_constraints = ON")
            if tamper == "missing_event_commitment":
                commitment = None
            elif tamper == "modified_event_commitment":
                commitment = copy.deepcopy(event.failure_audit_cohort)
                commitment["rows"][0]["content_hash"] = "f" * 64
                commitment["aggregate_digest"] = stable_hash({key: value for key, value in commitment.items() if key != "aggregate_digest"})
            else:
                commitment = copy.deepcopy(event.failure_audit_cohort)
                commitment["rows"].append(copy.deepcopy(commitment["rows"][0]))
                commitment["count"] = len(commitment["rows"])
                commitment["aggregate_digest"] = stable_hash({key: value for key, value in commitment.items() if key != "aggregate_digest"})
            conn.execute(
                update(guided_operation_events_table)
                .where(guided_operation_events_table.c.session_id == event.session_id)
                .where(guided_operation_events_table.c.operation_id == event.operation_id)
                .where(guided_operation_events_table.c.sequence == event.sequence)
                .values(failure_audit_cohort=commitment)
            )
        elif tamper == "ambiguous_event_commitment":
            conn.execute(
                insert(guided_operation_events_table).values(
                    session_id=event.session_id,
                    operation_id=event.operation_id,
                    sequence=event.sequence + 1,
                    event_kind="failed",
                    actor="duplicate-event",
                    attempt=event.attempt,
                    prior_attempt=None,
                    lease_expires_at=None,
                    request_hash=event.request_hash,
                    failure_audit_cohort=event.failure_audit_cohort,
                    occurred_at=datetime.now(UTC),
                )
            )

    if cloned_row is not None:
        content, tool_calls, writer_principal = cloned_row
        await service.add_message(
            session_id,
            "audit",
            content,
            writer_principal=writer_principal,
            tool_calls=tool_calls,
        )

    with pytest.raises(AuditIntegrityError, match="guided failure audit cohort"):
        await service.get_messages(session_id, limit=1)


@pytest.mark.asyncio
async def test_failure_without_evidence_writes_only_an_exact_terminal_event(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    claim = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="zero-evidence",
        kind="guided_plan",
        request_hash="f" * 64,
        actor="claim-zero",
        lease_seconds=30,
    )
    assert isinstance(claim, GuidedOperationClaimed)

    await service.fail_guided_operation_with_audit(
        GuidedOperationFailureCommand(
            fence=claim.fence,
            failure_code="operation_failed",
            actor="worker-zero",
            audit_evidence=GuidedAuditEvidence(),
        )
    )

    assert await service.get_messages(session_id, limit=None) == []
    with file_engine.connect() as conn:
        events = conn.execute(
            select(
                guided_operation_events_table.c.event_kind,
                guided_operation_events_table.c.operation_id,
                guided_operation_events_table.c.attempt,
                guided_operation_events_table.c.request_hash,
                guided_operation_events_table.c.failure_audit_cohort,
            )
            .where(guided_operation_events_table.c.session_id == str(session_id))
            .order_by(guided_operation_events_table.c.sequence)
        ).all()
    assert [tuple(row) for row in events] == [
        ("claimed", "zero-evidence", 1, "f" * 64, None),
        (
            "failed",
            "zero-evidence",
            1,
            "f" * 64,
            {
                "schema": "guided_failure_audit_cohort.v1",
                "count": 0,
                "rows": [],
                "aggregate_digest": stable_hash(
                    {
                        "schema": "guided_failure_audit_cohort.v1",
                        "count": 0,
                        "rows": [],
                    }
                ),
            },
        ),
    ]


@pytest.mark.asyncio
async def test_failure_event_fault_rolls_back_the_bound_audit_cohort(
    file_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    claim = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="rollback-bound-cohort",
        kind="guided_plan",
        request_hash="1" * 64,
        actor="claim-rollback",
        lease_seconds=30,
    )
    assert isinstance(claim, GuidedOperationClaimed)

    def fail_after_audit_insert(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("injected terminal event failure")

    monkeypatch.setattr(service, "fail_guided_operation_on_connection", fail_after_audit_insert)
    with pytest.raises(RuntimeError, match="injected terminal event failure"):
        await service.fail_guided_operation_with_audit(_failure_command(claim, marker="rollback"))

    assert await service.get_messages(session_id, limit=None) == []
    with file_engine.connect() as conn:
        operation = conn.execute(
            select(guided_operations_table.c.status)
            .where(guided_operations_table.c.session_id == str(session_id))
            .where(guided_operations_table.c.operation_id == "rollback-bound-cohort")
        ).one()
        events = (
            conn.execute(
                select(guided_operation_events_table.c.event_kind)
                .where(guided_operation_events_table.c.session_id == str(session_id))
                .where(guided_operation_events_table.c.operation_id == "rollback-bound-cohort")
                .order_by(guided_operation_events_table.c.sequence)
            )
            .scalars()
            .all()
        )
    assert operation.status == "in_progress"
    assert events == ["claimed"]


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
    state_id = await _seed_state(file_engine, session_id=session_id)
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
    state_id = await _seed_state(file_engine, session_id=session_id)
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

    with pytest.raises(GuidedOperationSettlementConflictError):
        await service_a.save_state_for_guided_operation(
            claim.fence,
            expected_current_state_id=None,
            expected_current_state_version=None,
            state=CompositionStateData(is_valid=True, composer_meta={"guided_session": {"schema_version": 9}}),
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

    with pytest.raises(GuidedOperationSettlementConflictError):
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
async def test_reconcile_guided_start_seals_absent_operation_as_request_cancelled(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    operation_id = "missing-guided-start"

    first = await service.reconcile_guided_start_operation(
        session_id=session_id,
        operation_id=operation_id,
        actor="reconciler",
    )
    second = await service.reconcile_guided_start_operation(
        session_id=session_id,
        operation_id=operation_id,
        actor="different-reconciler",
    )

    assert first == second == GuidedOperationFailed(failure_code="request_cancelled")
    with file_engine.connect() as conn:
        assert conn.execute(select(guided_operations_table)).all() == []
        assert conn.execute(select(guided_operation_events_table)).all() == []
        block = conn.execute(select(guided_operation_admission_blocks_table)).one()
    assert block.session_id == str(session_id)
    assert block.operation_id == operation_id
    assert block.kind == "guided_start"
    assert block.failure_code == "request_cancelled"
    assert block.actor == "reconciler"
    assert isinstance(block.created_at, datetime)


@pytest.mark.asyncio
async def test_blocked_guided_start_reservation_replays_cancellation_without_request_binding(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    operation_id = "blocked-guided-start"
    blocked = await service.reconcile_guided_start_operation(
        session_id=session_id,
        operation_id=operation_id,
        actor="reconciler",
    )

    original = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash="1" * 64,
        actor="original-worker",
        lease_seconds=30,
    )
    changed_payload = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash="2" * 64,
        actor="late-worker",
        lease_seconds=30,
    )

    assert blocked == original == changed_payload == GuidedOperationFailed(failure_code="request_cancelled")
    with file_engine.connect() as conn:
        assert conn.execute(select(guided_operations_table)).all() == []
        assert conn.execute(select(guided_operation_events_table)).all() == []


@pytest.mark.asyncio
async def test_blocked_guided_start_lookup_replays_cancellation_without_request_binding(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    operation_id = "blocked-guided-start-lookup"
    await service.reconcile_guided_start_operation(
        session_id=session_id,
        operation_id=operation_id,
        actor="reconciler",
    )

    first = await service.get_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash="1" * 64,
    )
    changed_payload = await service.get_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash="2" * 64,
    )

    assert first == changed_payload == GuidedOperationFailed(failure_code="request_cancelled")


@pytest.mark.asyncio
async def test_admission_block_rejects_reservation_as_another_guided_kind(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    operation_id = "blocked-wrong-kind"
    await service.reconcile_guided_start_operation(
        session_id=session_id,
        operation_id=operation_id,
        actor="reconciler",
    )

    with pytest.raises(GuidedOperationConflictError):
        await service.reserve_guided_operation(
            session_id=session_id,
            operation_id=operation_id,
            kind="guided_chat",
            request_hash="3" * 64,
            actor="chat-worker",
            lease_seconds=30,
        )


@pytest.mark.asyncio
async def test_reconcile_guided_start_reports_unexpired_attempt_without_mutation(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="live-guided-start",
        kind="guided_start",
        request_hash="6" * 64,
        actor="worker",
        lease_seconds=30,
    )
    assert isinstance(claimed, GuidedOperationClaimed)

    outcome = await service.reconcile_guided_start_operation(
        session_id=session_id,
        operation_id="live-guided-start",
        actor="reconciler",
    )

    assert outcome == GuidedOperationActive(attempt=1, lease_expires_at=claimed.lease_expires_at)
    with file_engine.connect() as conn:
        row = conn.execute(select(guided_operations_table)).one()
        events = conn.execute(select(guided_operation_events_table).order_by(guided_operation_events_table.c.sequence)).all()
    assert row.status == "in_progress"
    assert row.lease_token == claimed.fence.lease_token
    assert [event.event_kind for event in events] == ["claimed"]


@pytest.mark.asyncio
async def test_reconcile_guided_start_settles_expired_attempt_once_with_exact_empty_audit_cohort(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    claimed = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="expired-guided-start",
        kind="guided_start",
        request_hash="7" * 64,
        actor="worker",
        lease_seconds=30,
    )
    assert isinstance(claimed, GuidedOperationClaimed)
    _expire_operation(file_engine, session_id=session_id, operation_id="expired-guided-start")

    first = await service.reconcile_guided_start_operation(
        session_id=session_id,
        operation_id="expired-guided-start",
        actor="reconciler",
    )
    second = await service.reconcile_guided_start_operation(
        session_id=session_id,
        operation_id="expired-guided-start",
        actor="reconciler-again",
    )

    assert first == second == GuidedOperationFailed(failure_code="request_cancelled")
    with file_engine.connect() as conn:
        row = conn.execute(select(guided_operations_table)).one()
        events = conn.execute(select(guided_operation_events_table).order_by(guided_operation_events_table.c.sequence)).all()
    assert row.status == "failed"
    assert row.failure_code == "request_cancelled"
    assert row.lease_token is None
    assert row.lease_expires_at is None
    assert row.settled_at is not None
    assert [event.event_kind for event in events] == ["claimed", "failed"]
    assert events[-1].attempt == claimed.fence.attempt
    assert events[-1].actor == "reconciler"
    assert events[-1].failure_audit_cohort == {
        "schema": "guided_failure_audit_cohort.v1",
        "count": 0,
        "rows": [],
        "aggregate_digest": stable_hash(
            {
                "schema": "guided_failure_audit_cohort.v1",
                "count": 0,
                "rows": [],
            }
        ),
    }
    assert await service.get_messages(session_id, limit=None) == []


@pytest.mark.asyncio
async def test_reconcile_guided_start_returns_existing_terminal_outcome(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    state_id = await _seed_state(file_engine, session_id=session_id)
    completed_claim = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="completed-guided-start",
        kind="guided_start",
        request_hash="8" * 64,
        actor="worker",
        lease_seconds=30,
    )
    failed_claim = await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="failed-guided-start",
        kind="guided_start",
        request_hash="9" * 64,
        actor="worker",
        lease_seconds=30,
    )
    assert isinstance(completed_claim, GuidedOperationClaimed)
    assert isinstance(failed_claim, GuidedOperationClaimed)
    completed = await service.complete_guided_operation(
        completed_claim.fence,
        result=GuidedCompositionStateResult(state_id=state_id),
        response_hash="a" * 64,
        actor="worker",
    )
    failed = await service.fail_guided_operation(
        failed_claim.fence,
        failure_code="operation_failed",
        actor="worker",
    )

    assert (
        await service.reconcile_guided_start_operation(
            session_id=session_id,
            operation_id="completed-guided-start",
            actor="reconciler",
        )
        == completed
    )
    assert (
        await service.reconcile_guided_start_operation(
            session_id=session_id,
            operation_id="failed-guided-start",
            actor="reconciler",
        )
        == failed
    )


@pytest.mark.asyncio
async def test_reconcile_guided_start_rejects_operation_bound_to_another_kind(file_engine) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    await service.reserve_guided_operation(
        session_id=session_id,
        operation_id="wrong-kind",
        kind="guided_chat",
        request_hash="b" * 64,
        actor="worker",
        lease_seconds=30,
    )

    with pytest.raises(GuidedOperationConflictError):
        await service.reconcile_guided_start_operation(
            session_id=session_id,
            operation_id="wrong-kind",
            actor="reconciler",
        )


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
    state_id = await _seed_state(file_engine, session_id=session_id)
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
async def test_expired_guided_start_reconciliation_race_with_takeover_has_one_authority(durable_engine) -> None:
    reconcile_service = _service(durable_engine)
    takeover_service = _service(durable_engine)
    session_id = await _create_session(reconcile_service)
    operation_id = f"reconcile-takeover-{uuid4()}"
    request_hash = "e" * 64
    claim = await reconcile_service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash=request_hash,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(claim, GuidedOperationClaimed)
    _expire_operation(durable_engine, session_id=session_id, operation_id=operation_id)
    barrier = threading.Barrier(2)

    def reconcile():
        barrier.wait(timeout=5)
        return asyncio.run(
            reconcile_service.reconcile_guided_start_operation(
                session_id=session_id,
                operation_id=operation_id,
                actor="reconciler",
            )
        )

    def takeover():
        barrier.wait(timeout=5)
        return asyncio.run(
            takeover_service.reserve_guided_operation(
                session_id=session_id,
                operation_id=operation_id,
                kind="guided_start",
                request_hash=request_hash,
                actor="worker-b",
                lease_seconds=30,
            )
        )

    reconciled, reserved = await asyncio.gather(
        asyncio.to_thread(reconcile),
        asyncio.to_thread(takeover),
    )

    with durable_engine.connect() as conn:
        events = conn.execute(
            select(guided_operation_events_table)
            .where(
                guided_operation_events_table.c.session_id == str(session_id),
                guided_operation_events_table.c.operation_id == operation_id,
            )
            .order_by(guided_operation_events_table.c.sequence)
        ).all()
    if isinstance(reconciled, GuidedOperationFailed):
        assert reconciled.failure_code == "request_cancelled"
        assert reserved == reconciled
        assert [event.event_kind for event in events] == ["claimed", "failed"]
    else:
        assert isinstance(reconciled, GuidedOperationActive)
        assert isinstance(reserved, GuidedOperationTakenOver)
        assert reconciled.attempt == reserved.fence.attempt == 2
        assert [event.event_kind for event in events] == ["claimed", "taken_over"]


@pytest.mark.asyncio
async def test_expired_guided_start_reconciliation_race_rejects_stale_completion(durable_engine) -> None:
    reconcile_service = _service(durable_engine)
    completion_service = _service(durable_engine)
    session_id = await _create_session(reconcile_service)
    state_id = await _seed_state(durable_engine, session_id=session_id)
    operation_id = f"reconcile-complete-{uuid4()}"
    claim = await reconcile_service.reserve_guided_operation(
        session_id=session_id,
        operation_id=operation_id,
        kind="guided_start",
        request_hash="f" * 64,
        actor="worker-a",
        lease_seconds=30,
    )
    assert isinstance(claim, GuidedOperationClaimed)
    _expire_operation(durable_engine, session_id=session_id, operation_id=operation_id)
    barrier = threading.Barrier(2)

    def reconcile():
        barrier.wait(timeout=5)
        return asyncio.run(
            reconcile_service.reconcile_guided_start_operation(
                session_id=session_id,
                operation_id=operation_id,
                actor="reconciler",
            )
        )

    def complete():
        barrier.wait(timeout=5)
        try:
            return asyncio.run(
                completion_service.complete_guided_operation(
                    claim.fence,
                    result=GuidedCompositionStateResult(state_id=state_id),
                    response_hash="1" * 64,
                    actor="stale-worker",
                )
            )
        except GuidedOperationFenceLostError as exc:
            return exc

    reconciled, completed = await asyncio.gather(
        asyncio.to_thread(reconcile),
        asyncio.to_thread(complete),
    )

    assert reconciled == GuidedOperationFailed(failure_code="request_cancelled")
    assert isinstance(completed, GuidedOperationFenceLostError)
    with durable_engine.connect() as conn:
        events = conn.execute(
            select(guided_operation_events_table)
            .where(
                guided_operation_events_table.c.session_id == str(session_id),
                guided_operation_events_table.c.operation_id == operation_id,
            )
            .order_by(guided_operation_events_table.c.sequence)
        ).all()
    assert [event.event_kind for event in events] == ["claimed", "failed"]


@pytest.mark.asyncio
async def test_absent_reconciliation_race_with_original_reservation_has_one_admission_authority(
    durable_engine,
) -> None:
    reconcile_service = _service(durable_engine)
    original_service = _service(durable_engine)
    session_id = await _create_session(reconcile_service)
    operation_id = f"absent-original-race-{uuid4()}"
    barrier = threading.Barrier(2)

    def reconcile():
        barrier.wait(timeout=5)
        return asyncio.run(
            reconcile_service.reconcile_guided_start_operation(
                session_id=session_id,
                operation_id=operation_id,
                actor="reconciler",
            )
        )

    def reserve_original():
        barrier.wait(timeout=5)
        return asyncio.run(
            original_service.reserve_guided_operation(
                session_id=session_id,
                operation_id=operation_id,
                kind="guided_start",
                request_hash="a" * 64,
                actor="original-worker",
                lease_seconds=30,
            )
        )

    reconciled, reserved = await asyncio.gather(
        asyncio.to_thread(reconcile),
        asyncio.to_thread(reserve_original),
    )

    with durable_engine.connect() as conn:
        block_count = conn.scalar(
            select(func.count())
            .select_from(guided_operation_admission_blocks_table)
            .where(
                guided_operation_admission_blocks_table.c.session_id == str(session_id),
                guided_operation_admission_blocks_table.c.operation_id == operation_id,
            )
        )
        operation_count = conn.scalar(
            select(func.count())
            .select_from(guided_operations_table)
            .where(
                guided_operations_table.c.session_id == str(session_id),
                guided_operations_table.c.operation_id == operation_id,
            )
        )
        events = conn.execute(
            select(guided_operation_events_table).where(
                guided_operation_events_table.c.session_id == str(session_id),
                guided_operation_events_table.c.operation_id == operation_id,
            )
        ).all()

    assert block_count + operation_count == 1
    if isinstance(reserved, GuidedOperationClaimed):
        assert isinstance(reconciled, GuidedOperationActive)
        assert (block_count, operation_count) == (0, 1)
        assert [event.event_kind for event in events] == ["claimed"]
    else:
        assert reconciled == reserved == GuidedOperationFailed(failure_code="request_cancelled")
        assert (block_count, operation_count) == (1, 0)
        assert events == []


@pytest.mark.asyncio
async def test_absent_reconciliation_race_with_revised_start_leaves_only_revised_operation_admitted(
    durable_engine,
) -> None:
    reconcile_service = _service(durable_engine)
    revised_service = _service(durable_engine)
    session_id = await _create_session(reconcile_service)
    old_operation_id = f"old-start-{uuid4()}"
    revised_operation_id = f"revised-start-{uuid4()}"
    barrier = threading.Barrier(2)

    def block_old():
        barrier.wait(timeout=5)
        return asyncio.run(
            reconcile_service.reconcile_guided_start_operation(
                session_id=session_id,
                operation_id=old_operation_id,
                actor="reconciler",
            )
        )

    def reserve_revised():
        barrier.wait(timeout=5)
        return asyncio.run(
            revised_service.reserve_guided_operation(
                session_id=session_id,
                operation_id=revised_operation_id,
                kind="guided_start",
                request_hash="b" * 64,
                actor="revised-worker",
                lease_seconds=30,
            )
        )

    blocked, revised = await asyncio.gather(
        asyncio.to_thread(block_old),
        asyncio.to_thread(reserve_revised),
    )
    late_original = await revised_service.reserve_guided_operation(
        session_id=session_id,
        operation_id=old_operation_id,
        kind="guided_start",
        request_hash="c" * 64,
        actor="late-original-worker",
        lease_seconds=30,
    )

    assert blocked == late_original == GuidedOperationFailed(failure_code="request_cancelled")
    assert isinstance(revised, GuidedOperationClaimed)
    with durable_engine.connect() as conn:
        blocks = (
            conn.execute(
                select(guided_operation_admission_blocks_table.c.operation_id).where(
                    guided_operation_admission_blocks_table.c.session_id == str(session_id)
                )
            )
            .scalars()
            .all()
        )
        operations = (
            conn.execute(select(guided_operations_table.c.operation_id).where(guided_operations_table.c.session_id == str(session_id)))
            .scalars()
            .all()
        )
        events = (
            conn.execute(
                select(guided_operation_events_table.c.operation_id).where(guided_operation_events_table.c.session_id == str(session_id))
            )
            .scalars()
            .all()
        )
    assert blocks == [old_operation_id]
    assert operations == [revised_operation_id]
    assert events == [revised_operation_id]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "result_factory"),
    [
        ("guided_start", "state"),
        ("guided_respond", "state_with_proposal"),
        ("guided_chat", "state_with_proposal"),
        ("guided_convert", "state"),
        ("guided_reenter", "state"),
        ("guided_plan", "pipeline_proposal"),
        ("state_revert", "state"),
        ("session_fork", "session"),
    ],
)
async def test_terminal_replay_uses_closed_per_kind_locator(file_engine, kind: str, result_factory: str) -> None:
    service = _service(file_engine)
    session_id = await _create_session(service)
    state_id = await _seed_state(file_engine, session_id=session_id)
    proposal_id = _seed_proposal(file_engine, session_id=session_id)
    child_session_id = (await service.create_session("alice", "Fork result", "local")).id
    with file_engine.begin() as conn:
        conn.execute(
            update(sessions_table).where(sessions_table.c.id == str(child_session_id)).values(forked_from_session_id=str(session_id))
        )
    result = {
        "state": GuidedCompositionStateResult(state_id=state_id),
        "state_with_proposal": GuidedCompositionStateResult(state_id=state_id, proposal_id=proposal_id),
        "pipeline_proposal": GuidedPipelineProposalResult(proposal_id=proposal_id, checkpoint_state_id=state_id),
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

    if isinstance(result, GuidedSessionResult):
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
        target = await service.create_session("alice", "Wrong lineage", "local")
        with file_engine.begin() as conn:
            conn.execute(
                update(sessions_table).where(sessions_table.c.id == str(target.id)).values(forked_from_session_id=str(other_parent.id))
            )
    else:
        target = await service.create_session("bob", "Cross-user child", "local")
        with file_engine.begin() as conn:
            conn.execute(update(sessions_table).where(sessions_table.c.id == str(target.id)).values(forked_from_session_id=str(parent_id)))
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
    state_id = await _seed_state(file_engine, session_id=session_id)
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
