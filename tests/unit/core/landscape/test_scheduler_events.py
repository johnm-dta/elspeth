"""Scheduler event audit trail regression coverage."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from sqlalchemy import insert, select
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError

import elspeth.core.landscape.database as database_module
from elspeth.contracts import NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import AuditIntegrityError, SchedulerLeaseLostError
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.landscape.database import LandscapeDB, Tier1Engine
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.schema import (
    coalesce_branch_losses_table,
    metadata,
    nodes_table,
    rows_table,
    run_coordination_table,
    run_workers_table,
    runs_table,
    token_outcomes_table,
    token_work_items_table,
    tokens_table,
)
from tests.fixtures.landscape import make_recorder_with_run, register_test_node

if TYPE_CHECKING:
    from elspeth.contracts.scheduler import BranchLossSpec, TokenWorkItem
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

# Epoch-1 coordination seat token for the "run-1" test run.
# The ratcheted verbs (mark_pending_sink_terminal*, terminalize_pending_sinks*)
# require a non-None token; _insert_scheduler_prerequisites seeds the matching row.
_COORD_TOKEN = CoordinationToken(run_id="run-1", worker_id="test-leader", leader_epoch=1)


def test_scheduler_events_schema_is_required_run_scoped_contract() -> None:
    from elspeth.core.landscape.schema import scheduler_events_table

    assert {
        "event_id",
        "run_id",
        "token_id",
        "work_item_id",
        "node_id",
        "event_type",
        "from_status",
        "to_status",
        "from_lease_owner",
        "to_lease_owner",
        "from_lease_expires_at",
        "to_lease_expires_at",
        "from_attempt",
        "to_attempt",
        "recorded_at",
        "caller_owner",
        "context_json",
    } <= set(scheduler_events_table.c.keys())

    required_columns = {column for table, column in database_module._REQUIRED_COLUMNS if table == "scheduler_events"}
    assert {
        "event_id",
        "run_id",
        "token_id",
        "work_item_id",
        "event_type",
        "to_status",
        "to_attempt",
        "recorded_at",
        "context_json",
        "from_lease_expires_at",
        "to_lease_expires_at",
    } <= required_columns
    required_constraints = {constraint for table, constraint in database_module._REQUIRED_CHECK_CONSTRAINTS if table == "scheduler_events"}
    assert {
        "ck_scheduler_events_event_type",
        "ck_scheduler_events_from_status",
        "ck_scheduler_events_to_status",
        "ck_scheduler_events_from_attempt_non_negative",
        "ck_scheduler_events_to_attempt_non_negative",
    } <= required_constraints
    assert ("scheduler_events", "run_id", "runs") in database_module._REQUIRED_FOREIGN_KEYS
    assert (
        "scheduler_events",
        ("token_id", "run_id"),
        "tokens",
        ("token_id", "run_id"),
    ) in database_module._REQUIRED_COMPOSITE_FOREIGN_KEYS
    assert ("scheduler_events", "ix_scheduler_events_run_token_time") in database_module._REQUIRED_INDEXES
    assert all(fk.column.table.name != "token_work_items" for fk in scheduler_events_table.foreign_keys)


def test_enqueue_ready_records_single_idempotent_scheduler_event() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)

    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    duplicate = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )

    events = _scheduler_events(engine)
    assert duplicate.work_item_id == item.work_item_id
    assert [(event.event_type, event.work_item_id) for event in events] == [(SchedulerEventType.ENQUEUE.value, item.work_item_id)]
    event = events[0]
    assert event.token_id == "token-1"
    assert event.node_id == "normalize"
    assert event.from_status is None
    assert event.to_status == TokenWorkStatus.READY.value
    assert event.from_attempt is None
    assert event.to_attempt == 1
    assert event.from_lease_owner is None
    assert event.to_lease_owner is None
    assert event.caller_owner is None
    assert json.loads(event.context_json) == {}


def test_enqueue_ready_mismatch_diagnostics_redact_row_payload_values() -> None:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    secret = "sk-" + ("a" * 32)
    alternate_payload = TokenSchedulerRepository.serialize_row_payload(
        PipelineRow({"id": 1, "api_key": secret}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
    )

    repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )

    with pytest.raises(LandscapeRecordError) as exc_info:
        repo.enqueue_ready(
            run_id="run-1",
            token_id="token-1",
            row_id="row-1",
            node_id="normalize",
            step_index=1,
            ingest_sequence=0,
            available_at=now,
            row_payload_json=alternate_payload,
        )

    message = str(exc_info.value)
    assert "row_payload_json" in message
    assert "sha256=" in message
    assert "bytes=" in message
    assert secret not in message
    assert alternate_payload not in message


def test_enqueue_ready_claimed_records_enqueue_and_claim_events_in_one_operation() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)

    claimed = repo.enqueue_ready_claimed(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
        lease_owner="worker-a",
        lease_seconds=30,
        now=now + timedelta(seconds=1),
    )

    assert claimed.status is TokenWorkStatus.LEASED
    assert claimed.lease_owner == "worker-a"
    assert claimed.lease_expires_at == now + timedelta(seconds=31)

    events = _scheduler_events(engine)
    assert [event.event_type for event in events] == [
        SchedulerEventType.ENQUEUE.value,
        SchedulerEventType.CLAIM_READY.value,
    ]
    enqueue_event, claim_event = events
    assert enqueue_event.work_item_id == claimed.work_item_id
    assert enqueue_event.from_status is None
    assert enqueue_event.to_status == TokenWorkStatus.READY.value
    assert enqueue_event.from_attempt is None
    assert enqueue_event.to_attempt == 1
    assert enqueue_event.caller_owner is None

    assert claim_event.work_item_id == claimed.work_item_id
    assert claim_event.from_status == TokenWorkStatus.READY.value
    assert claim_event.to_status == TokenWorkStatus.LEASED.value
    assert claim_event.from_lease_owner is None
    assert claim_event.to_lease_owner == "worker-a"
    assert claim_event.from_lease_expires_at is None
    assert claim_event.to_lease_expires_at == _stored_datetime(now + timedelta(seconds=31))
    assert claim_event.from_attempt == 1
    assert claim_event.to_attempt == 1
    assert claim_event.caller_owner == "worker-a"


def test_claim_and_terminal_events_record_status_and_lease_ownership() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)

    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now + timedelta(seconds=1))
    assert claimed is not None
    repo.mark_terminal(work_item_id=item.work_item_id, now=now + timedelta(seconds=2), expected_lease_owner="worker-a")

    events = _scheduler_events(engine)
    assert [event.event_type for event in events] == [
        SchedulerEventType.ENQUEUE.value,
        SchedulerEventType.CLAIM_READY.value,
        SchedulerEventType.MARK_TERMINAL.value,
    ]

    claim_event = events[1]
    assert claim_event.from_status == TokenWorkStatus.READY.value
    assert claim_event.to_status == TokenWorkStatus.LEASED.value
    assert claim_event.from_lease_owner is None
    assert claim_event.to_lease_owner == "worker-a"
    assert claim_event.from_lease_expires_at is None
    assert claim_event.to_lease_expires_at == _stored_datetime(now + timedelta(seconds=31))
    assert claim_event.from_attempt == 1
    assert claim_event.to_attempt == 1
    assert claim_event.caller_owner == "worker-a"

    terminal_event = events[2]
    assert terminal_event.from_status == TokenWorkStatus.LEASED.value
    assert terminal_event.to_status == TokenWorkStatus.TERMINAL.value
    assert terminal_event.from_lease_owner == "worker-a"
    assert terminal_event.to_lease_owner is None
    assert terminal_event.from_lease_expires_at == _stored_datetime(now + timedelta(seconds=31))
    assert terminal_event.to_lease_expires_at is None
    assert terminal_event.caller_owner == "worker-a"


def test_recover_expired_leases_records_attempt_bump_and_previous_work_item() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)

    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now)
    assert claimed is not None

    recovered = repo.recover_expired_leases(
        run_id="run-1",
        now=now + timedelta(seconds=31),
        caller_owner="worker-b",
    )

    events = _scheduler_events(engine)
    recovery_event = events[-1]
    assert recovered == 1
    assert recovery_event.event_type == SchedulerEventType.RECOVER_EXPIRED_LEASE.value
    assert recovery_event.work_item_id != item.work_item_id
    assert recovery_event.token_id == "token-1"
    assert recovery_event.from_status == TokenWorkStatus.LEASED.value
    assert recovery_event.to_status == TokenWorkStatus.READY.value
    assert recovery_event.from_lease_owner == "worker-a"
    assert recovery_event.to_lease_owner is None
    assert recovery_event.from_lease_expires_at == _stored_datetime(now + timedelta(seconds=30))
    assert recovery_event.to_lease_expires_at is None
    assert recovery_event.from_attempt == claimed.attempt
    assert recovery_event.to_attempt == claimed.attempt + 1
    assert recovery_event.caller_owner == "worker-b"
    assert json.loads(recovery_event.context_json) == {"previous_work_item_id": item.work_item_id}


def test_heartbeat_lease_lost_records_event_when_current_row_is_peer_owned() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)

    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now)
    assert claimed is not None

    with engine.begin() as conn:
        conn.execute(
            token_work_items_table.update()
            .where(token_work_items_table.c.work_item_id == item.work_item_id)
            .values(
                lease_owner="worker-b",
                lease_expires_at=now + timedelta(seconds=30),
                updated_at=now + timedelta(seconds=1),
            )
        )

    with pytest.raises(SchedulerLeaseLostError):
        repo.heartbeat_lease(
            run_id="run-1",
            work_item_id=item.work_item_id,
            lease_owner="worker-a",
            lease_seconds=30,
            now=now + timedelta(seconds=2),
        )

    event = _scheduler_events(engine)[-1]
    assert event.event_type == SchedulerEventType.LEASE_LOST.value
    assert event.from_status == TokenWorkStatus.LEASED.value
    assert event.to_status == TokenWorkStatus.LEASED.value
    assert event.from_lease_owner == "worker-a"
    assert event.to_lease_owner == "worker-b"
    assert event.from_lease_expires_at is None
    assert event.to_lease_expires_at == _stored_datetime(now + timedelta(seconds=30))
    assert event.from_attempt == claimed.attempt
    assert event.to_attempt == claimed.attempt
    assert event.caller_owner == "worker-a"


def test_heartbeat_lease_lost_records_event_when_expired_lease_was_recovered() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)

    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now)
    assert claimed is not None
    recovered = repo.recover_expired_leases(
        run_id="run-1",
        now=now + timedelta(seconds=31),
        caller_owner="worker-b",
    )
    assert recovered == 1

    with pytest.raises(SchedulerLeaseLostError):
        repo.heartbeat_lease(
            run_id="run-1",
            work_item_id=item.work_item_id,
            lease_owner="worker-a",
            lease_seconds=30,
            now=now + timedelta(seconds=32),
        )

    events = _scheduler_events(engine)
    recovery_event = events[-2]
    lease_lost_event = events[-1]
    assert recovery_event.event_type == SchedulerEventType.RECOVER_EXPIRED_LEASE.value
    assert lease_lost_event.event_type == SchedulerEventType.LEASE_LOST.value
    assert lease_lost_event.work_item_id == item.work_item_id
    assert lease_lost_event.from_status == TokenWorkStatus.LEASED.value
    assert lease_lost_event.to_status == TokenWorkStatus.READY.value
    assert lease_lost_event.from_lease_owner == "worker-a"
    assert lease_lost_event.to_lease_owner is None
    assert lease_lost_event.from_lease_expires_at == _stored_datetime(now + timedelta(seconds=30))
    assert lease_lost_event.to_lease_expires_at is None
    assert lease_lost_event.from_attempt == claimed.attempt
    assert lease_lost_event.to_attempt == claimed.attempt + 1
    assert lease_lost_event.caller_owner == "worker-a"
    assert json.loads(lease_lost_event.context_json) == {
        "reason": "heartbeat_cas_miss_after_recovery",
        "recovered_work_item_id": recovery_event.work_item_id,
        "recovery_event_id": recovery_event.event_id,
    }


def test_mark_blocked_and_mark_failed_record_transition_events() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    assert repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now + timedelta(seconds=1)) is not None

    blocked = repo.mark_blocked(
        work_item_id=item.work_item_id,
        queue_key="queue-a",
        barrier_key=None,
        now=now + timedelta(seconds=2),
        expected_lease_owner="worker-a",
    )

    events = _scheduler_events(engine)
    assert blocked.status is TokenWorkStatus.BLOCKED
    assert [event.event_type for event in events] == [
        SchedulerEventType.ENQUEUE.value,
        SchedulerEventType.CLAIM_READY.value,
        SchedulerEventType.MARK_BLOCKED.value,
    ]
    assert events[-1].from_status == TokenWorkStatus.LEASED.value
    assert events[-1].to_status == TokenWorkStatus.BLOCKED.value

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    assert repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now + timedelta(seconds=1)) is not None
    failed = repo.mark_failed(work_item_id=item.work_item_id, now=now + timedelta(seconds=2), expected_lease_owner="worker-a")

    events = _scheduler_events(engine)
    assert failed.status is TokenWorkStatus.FAILED
    assert [event.event_type for event in events] == [
        SchedulerEventType.ENQUEUE.value,
        SchedulerEventType.CLAIM_READY.value,
        SchedulerEventType.MARK_FAILED.value,
    ]
    assert events[-1].from_status == TokenWorkStatus.LEASED.value
    assert events[-1].to_status == TokenWorkStatus.FAILED.value


def test_pending_sink_claim_and_terminalization_record_transition_events() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    assert repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now + timedelta(seconds=1)) is not None
    repo.mark_pending_sink(
        work_item_id=item.work_item_id,
        row_payload_json=payload,
        sink_name="sink-a",
        outcome="success",
        path="completed",
        error_hash=None,
        error_message=None,
        now=now + timedelta(seconds=2),
        expected_lease_owner="worker-a",
    )
    assert repo.claim_pending_sink(run_id="run-1", lease_owner="worker-b", lease_seconds=30, now=now + timedelta(seconds=3)) is not None
    terminalized = repo.mark_pending_sink_terminal(
        run_id="run-1",
        token_id="token-1",
        now=now + timedelta(seconds=4),
        expected_lease_owner="worker-b",
        coordination_token=_COORD_TOKEN,
    )

    events = _scheduler_events(engine)
    assert terminalized == 1
    assert [event.event_type for event in events] == [
        SchedulerEventType.ENQUEUE.value,
        SchedulerEventType.CLAIM_READY.value,
        SchedulerEventType.MARK_PENDING_SINK.value,
        SchedulerEventType.CLAIM_PENDING_SINK.value,
        SchedulerEventType.MARK_PENDING_SINK_TERMINAL.value,
    ]
    assert events[-1].from_status == TokenWorkStatus.LEASED.value
    assert events[-1].to_status == TokenWorkStatus.TERMINAL.value
    assert events[-1].caller_owner == "worker-b"


@pytest.mark.parametrize("verb", ["mark_terminal", "mark_failed", "mark_blocked", "mark_pending_sink"])
def test_normal_dispositions_refuse_reclaimed_sink_redrive_without_mutation(verb: str) -> None:
    """TS-07 through TS-10 are transform-lease-only dispositions.

    A reclaimed sink handoff is also LEASED, but ``pending_sink_name`` makes it
    the sink-redrive subtype.  Refusal must occur before any part of the
    transactional state/event/branch-loss image changes.
    """
    from elspeth.contracts.scheduler import BranchLossSpec, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = _make_pending_sink(repo, run_id="run-1", token_id="token-1", row_id="row-1", payload=payload, now=now)
    reclaimed = repo.claim_pending_sink(
        run_id="run-1",
        lease_owner="worker-b",
        lease_seconds=30,
        now=now + timedelta(seconds=3),
    )
    assert reclaimed is not None
    assert reclaimed.status is TokenWorkStatus.LEASED
    assert reclaimed.pending_sink_name == "sink-a"

    before = _disposition_state_snapshot(engine, work_item_id=item.work_item_id)
    branch_loss = BranchLossSpec(
        coalesce_name="merge",
        row_id="row-1",
        branch_name="left",
        token_id="token-1",
        reason="refused-normal-disposition",
        recorded_by="worker-b",
    )

    with pytest.raises(AuditIntegrityError, match="transform-lease row"):
        _invoke_normal_disposition(
            repo,
            verb=verb,
            work_item_id=item.work_item_id,
            payload=payload,
            now=now + timedelta(seconds=4),
            branch_loss=branch_loss,
        )

    assert _disposition_state_snapshot(engine, work_item_id=item.work_item_id) == before


@pytest.mark.parametrize(
    ("verb", "expected_status"),
    [
        ("mark_terminal", "terminal"),
        ("mark_failed", "failed"),
        ("mark_blocked", "blocked"),
        ("mark_pending_sink", "pending_sink"),
    ],
)
def test_normal_dispositions_still_accept_transform_leases(verb: str, expected_status: str) -> None:
    """The subtype gate must preserve every legal transform-lease arm."""
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-b", lease_seconds=30, now=now + timedelta(seconds=1))
    assert claimed is not None
    assert claimed.pending_sink_name is None

    transitioned = _invoke_normal_disposition(
        repo,
        verb=verb,
        work_item_id=item.work_item_id,
        payload=payload,
        now=now + timedelta(seconds=2),
    )

    assert transitioned.status.value == expected_status


@pytest.mark.parametrize("verb", ["mark_pending_sink_terminal", "mark_pending_sink_terminal_many"])
def test_dedicated_sink_redrive_terminalizers_still_accept_reclaimed_sink_leases(verb: str) -> None:
    """The normal-disposition gate must not block the dedicated redrive exit."""
    from elspeth.contracts.scheduler import TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = _make_pending_sink(repo, run_id="run-1", token_id="token-1", row_id="row-1", payload=payload, now=now)
    reclaimed = repo.claim_pending_sink(
        run_id="run-1",
        lease_owner="worker-b",
        lease_seconds=30,
        now=now + timedelta(seconds=3),
    )
    assert reclaimed is not None
    assert reclaimed.status is TokenWorkStatus.LEASED
    assert reclaimed.pending_sink_name == "sink-a"

    if verb == "mark_pending_sink_terminal":
        terminalized = repo.mark_pending_sink_terminal(
            run_id="run-1",
            token_id="token-1",
            now=now + timedelta(seconds=4),
            expected_lease_owner="worker-b",
            coordination_token=_COORD_TOKEN,
        )
    else:
        terminalized = repo.mark_pending_sink_terminal_many(
            run_id="run-1",
            token_ids=("token-1",),
            now=now + timedelta(seconds=4),
            expected_lease_owner="worker-b",
            coordination_token=_COORD_TOKEN,
        )

    assert terminalized == 1
    with engine.connect() as conn:
        status = conn.execute(
            select(token_work_items_table.c.status).where(token_work_items_table.c.work_item_id == item.work_item_id)
        ).scalar_one()
    assert status == TokenWorkStatus.TERMINAL.value


def test_normal_disposition_rolls_back_when_scheduler_event_insert_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared disposition transaction is atomic with its event and loss."""
    from elspeth.contracts.scheduler import BranchLossSpec, SchedulerEventType
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    assert repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now + timedelta(seconds=1)) is not None
    before = _disposition_state_snapshot(engine, work_item_id=item.work_item_id)
    original_record_scheduler_event = repo.events.record

    def fail_failed_event(conn, *, event_type, **kwargs):
        if event_type is SchedulerEventType.MARK_FAILED:
            raise LandscapeRecordError("forced disposition event failure")
        return original_record_scheduler_event(conn, event_type=event_type, **kwargs)

    monkeypatch.setattr(repo.events, "record", fail_failed_event)

    with pytest.raises(LandscapeRecordError, match="forced disposition event failure"):
        repo.mark_failed(
            work_item_id=item.work_item_id,
            now=now + timedelta(seconds=2),
            expected_lease_owner="worker-a",
            branch_loss=BranchLossSpec(
                coalesce_name="merge",
                row_id="row-1",
                branch_name="left",
                token_id="token-1",
                reason="failed",
                recorded_by="worker-a",
            ),
        )

    assert _disposition_state_snapshot(engine, work_item_id=item.work_item_id) == before


def test_pending_sink_batch_terminalization_records_per_token_events() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    with engine.begin() as conn:
        conn.execute(
            insert(rows_table).values(
                row_id="row-2",
                run_id="run-1",
                source_node_id="source-0",
                row_index=1,
                source_row_index=1,
                ingest_sequence=1,
                source_data_hash="hash-row-2",
                created_at=now,
            )
        )
        conn.execute(
            insert(tokens_table).values(
                token_id="token-2",
                row_id="row-2",
                run_id="run-1",
                created_at=now,
            )
        )

    first = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    second = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-2",
        row_id="row-2",
        node_id="normalize",
        step_index=1,
        ingest_sequence=1,
        available_at=now,
        row_payload_json=payload,
    )
    assert repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now + timedelta(seconds=1)) is not None
    repo.mark_pending_sink(
        work_item_id=first.work_item_id,
        row_payload_json=payload,
        sink_name="sink-a",
        outcome="success",
        path="completed",
        error_hash=None,
        error_message=None,
        now=now + timedelta(seconds=2),
        expected_lease_owner="worker-a",
    )
    assert repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now + timedelta(seconds=3)) is not None
    repo.mark_pending_sink(
        work_item_id=second.work_item_id,
        row_payload_json=payload,
        sink_name="sink-a",
        outcome="success",
        path="completed",
        error_hash=None,
        error_message=None,
        now=now + timedelta(seconds=4),
        expected_lease_owner="worker-a",
    )

    terminalized = repo.mark_pending_sink_terminal_many(
        run_id="run-1",
        token_ids=("token-1", "token-2"),
        now=now + timedelta(seconds=5),
        expected_lease_owner="worker-a",
        coordination_token=_COORD_TOKEN,
    )

    events = _scheduler_events(engine)
    terminal_events = [event for event in events if event.event_type == SchedulerEventType.MARK_PENDING_SINK_TERMINAL.value]
    assert terminalized == 2
    assert len(terminal_events) == 2
    assert {event.token_id for event in terminal_events} == {"token-1", "token-2"}
    assert {event.from_status for event in terminal_events} == {TokenWorkStatus.PENDING_SINK.value}
    assert {event.to_status for event in terminal_events} == {TokenWorkStatus.TERMINAL.value}
    assert {event.caller_owner for event in terminal_events} == {"worker-a"}


def test_pending_sink_batch_terminalization_rejects_duplicate_token_ids() -> None:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = _make_pending_sink(repo, run_id="run-1", token_id="token-1", row_id="row-1", payload=payload, now=now)

    with pytest.raises(AuditIntegrityError, match="duplicate token_id"):
        repo.mark_pending_sink_terminal_many(
            run_id="run-1",
            token_ids=(item.token_id, item.token_id),
            now=now + timedelta(seconds=3),
            expected_lease_owner="worker-a",
            coordination_token=_COORD_TOKEN,
        )


def test_pending_sink_batch_terminalization_requires_every_requested_token() -> None:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = _make_pending_sink(repo, run_id="run-1", token_id="token-1", row_id="row-1", payload=payload, now=now)

    with pytest.raises(AuditIntegrityError, match="missing token_id"):
        repo.mark_pending_sink_terminal_many(
            run_id="run-1",
            token_ids=(item.token_id, "token-missing"),
            now=now + timedelta(seconds=3),
            expected_lease_owner="worker-a",
            coordination_token=_COORD_TOKEN,
        )


def test_pending_sink_batch_terminalization_rejects_wrong_lease_owner() -> None:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    _make_pending_sink(repo, run_id="run-1", token_id="token-1", row_id="row-1", payload=payload, now=now)
    claimed = repo.claim_pending_sink(
        run_id="run-1",
        lease_owner="worker-b",
        lease_seconds=30,
        now=now + timedelta(seconds=3),
    )
    assert claimed is not None

    with pytest.raises(AuditIntegrityError, match="lease_owner"):
        repo.mark_pending_sink_terminal_many(
            run_id="run-1",
            token_ids=("token-1",),
            now=now + timedelta(seconds=4),
            expected_lease_owner="worker-a",
            coordination_token=_COORD_TOKEN,
        )


def test_pending_sink_with_terminal_outcome_is_repaired_without_reclaiming_sink() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = _make_pending_sink(repo, run_id="run-1", token_id="token-1", row_id="row-1", payload=payload, now=now)
    with engine.begin() as conn:
        conn.execute(
            insert(token_outcomes_table).values(
                outcome_id="out-token-1",
                run_id="run-1",
                token_id="token-1",
                outcome=TerminalOutcome.SUCCESS.value,
                path=TerminalPath.DEFAULT_FLOW.value,
                completed=1,
                recorded_at=now + timedelta(seconds=3),
                sink_name="sink-a",
            )
        )

    terminalized = repo.terminalize_pending_sinks_with_terminal_outcomes(
        run_id="run-1",
        now=now + timedelta(seconds=4),
        caller_owner="resume-repair",
        coordination_token=_COORD_TOKEN,
    )

    assert terminalized == 1
    assert repo.claim_pending_sink(run_id="run-1", lease_owner="worker-b", lease_seconds=30, now=now + timedelta(seconds=5)) is None
    with engine.connect() as conn:
        status = conn.execute(
            select(token_work_items_table.c.status).where(token_work_items_table.c.work_item_id == item.work_item_id)
        ).scalar_one()
    assert status == TokenWorkStatus.TERMINAL.value
    terminal_events = [
        event for event in _scheduler_events(engine) if event.event_type == SchedulerEventType.MARK_PENDING_SINK_TERMINAL.value
    ]
    assert len(terminal_events) == 1
    assert terminal_events[0].from_status == TokenWorkStatus.PENDING_SINK.value
    assert terminal_events[0].caller_owner == "resume-repair"


def test_blocked_barrier_terminalization_records_transition_event() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    assert repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now + timedelta(seconds=1)) is not None
    blocked = repo.mark_blocked(
        work_item_id=item.work_item_id,
        queue_key=None,
        barrier_key="join:1",
        now=now + timedelta(seconds=2),
        expected_lease_owner="worker-a",
    )
    terminalized = repo.mark_blocked_barrier_terminal(
        run_id="run-1",
        barrier_key="join:1",
        token_ids=("token-1",),
        now=now + timedelta(seconds=3),
    )

    events = _scheduler_events(engine)
    assert blocked.status is TokenWorkStatus.BLOCKED
    assert terminalized == 1
    assert [event.event_type for event in events] == [
        SchedulerEventType.ENQUEUE.value,
        SchedulerEventType.CLAIM_READY.value,
        SchedulerEventType.MARK_BLOCKED.value,
        SchedulerEventType.MARK_BLOCKED_BARRIER_TERMINAL.value,
    ]
    assert events[-1].from_status == TokenWorkStatus.BLOCKED.value
    assert events[-1].to_status == TokenWorkStatus.TERMINAL.value
    assert json.loads(events[-1].context_json) == {"barrier_key": "join:1"}


def test_blocked_barrier_pending_sink_handoff_records_state_and_event() -> None:
    from elspeth.contracts.scheduler import BlockedPendingSinkHandoff, SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    assert repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now + timedelta(seconds=1)) is not None
    repo.mark_blocked(
        work_item_id=item.work_item_id,
        queue_key=None,
        barrier_key="agg-1",
        now=now + timedelta(seconds=2),
        expected_lease_owner="worker-a",
    )
    sink_payload = TokenSchedulerRepository.serialize_row_payload(
        PipelineRow({"id": 2}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
    )

    transitioned = repo.mark_blocked_barrier_pending_sink_many(
        run_id="run-1",
        barrier_key="agg-1",
        handoffs={
            "token-1": BlockedPendingSinkHandoff(
                row_payload_json=sink_payload,
                sink_name="sink-a",
                outcome=TerminalOutcome.SUCCESS.value,
                path=TerminalPath.DEFAULT_FLOW.value,
                error_hash=None,
                error_message=None,
            )
        },
        now=now + timedelta(seconds=3),
    )

    with engine.connect() as conn:
        row = (
            conn.execute(
                select(
                    token_work_items_table.c.status,
                    token_work_items_table.c.row_payload_json,
                    token_work_items_table.c.pending_sink_name,
                    token_work_items_table.c.pending_outcome,
                    token_work_items_table.c.pending_path,
                ).where(token_work_items_table.c.work_item_id == item.work_item_id)
            )
            .mappings()
            .one()
        )

    events = _scheduler_events(engine)
    assert transitioned == 1
    assert row.status == TokenWorkStatus.PENDING_SINK.value
    assert row.row_payload_json == sink_payload
    assert row.pending_sink_name == "sink-a"
    assert row.pending_outcome == TerminalOutcome.SUCCESS.value
    assert row.pending_path == TerminalPath.DEFAULT_FLOW.value
    assert [event.event_type for event in events] == [
        SchedulerEventType.ENQUEUE.value,
        SchedulerEventType.CLAIM_READY.value,
        SchedulerEventType.MARK_BLOCKED.value,
        SchedulerEventType.MARK_PENDING_SINK.value,
    ]
    assert events[-1].from_status == TokenWorkStatus.BLOCKED.value
    assert events[-1].to_status == TokenWorkStatus.PENDING_SINK.value
    assert json.loads(events[-1].context_json) == {"barrier_key": "agg-1"}


def test_claim_ready_rolls_back_work_item_update_when_scheduler_event_insert_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = _make_scheduler_engine()
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = _insert_scheduler_prerequisites(engine, now=now)
    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    original_record_scheduler_event = repo.events.record

    def fail_claim_event(conn, *, event_type, **kwargs):
        if event_type is SchedulerEventType.CLAIM_READY:
            raise LandscapeRecordError("forced scheduler event failure")
        return original_record_scheduler_event(conn, event_type=event_type, **kwargs)

    # The shared SchedulerEventStore instance is the single event-plane seam:
    # every component records through it, so patching it intercepts the
    # lease repository's internal CLAIM_READY write.
    monkeypatch.setattr(repo.events, "record", fail_claim_event)

    with pytest.raises(LandscapeRecordError, match="forced scheduler event failure"):
        repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now + timedelta(seconds=1))

    with engine.connect() as conn:
        row = conn.execute(
            select(token_work_items_table.c.status, token_work_items_table.c.lease_owner).where(
                token_work_items_table.c.work_item_id == item.work_item_id
            )
        ).one()
    assert row == (TokenWorkStatus.READY.value, None)
    assert [event.event_type for event in _scheduler_events(engine)] == [SchedulerEventType.ENQUEUE.value]


def test_query_repository_lists_scheduler_events_by_token_history() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType
    from elspeth.core.landscape.lineage import explain

    setup = make_recorder_with_run(run_id="run-1", source_node_id="source-0", source_plugin_name="csv")
    db, factory = setup.db, setup.factory
    register_test_node(factory.data_flow, "run-1", "normalize", plugin_name="identity")
    factory.data_flow.create_row(
        "run-1",
        "source-0",
        0,
        {"id": 1},
        row_id="row-1",
        source_row_index=0,
        ingest_sequence=0,
    )
    factory.data_flow.create_token("row-1", token_id="token-1")
    now = datetime.now(UTC)
    payload = factory.scheduler.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))

    factory.scheduler.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    with db.engine.begin() as conn:
        conn.execute(
            insert(run_workers_table).values(
                worker_id="worker-a",
                run_id="run-1",
                role="follower",
                status="active",
                registered_at=now,
                heartbeat_expires_at=now + timedelta(hours=1),
            )
        )
    factory.scheduler.claim_ready(
        run_id="run-1",
        lease_owner="worker-a",
        lease_seconds=30,
        now=now + timedelta(seconds=1),
    )

    events = factory.query.get_scheduler_events(run_id="run-1", token_id="token-1")
    assert [event.event_type for event in events] == [SchedulerEventType.ENQUEUE, SchedulerEventType.CLAIM_READY]
    assert [event.run_id for event in events] == ["run-1", "run-1"]
    assert [event.token_id for event in events] == ["token-1", "token-1"]
    lineage = explain(factory.query, factory.data_flow, "run-1", token_id="token-1")
    assert lineage is not None
    assert [event.event_type for event in lineage.scheduler_events] == [SchedulerEventType.ENQUEUE, SchedulerEventType.CLAIM_READY]
    db.close()


def _invoke_normal_disposition(
    repo: TokenSchedulerRepository,
    *,
    verb: str,
    work_item_id: str,
    payload: str,
    now: datetime,
    branch_loss: BranchLossSpec | None = None,
) -> TokenWorkItem:
    if verb == "mark_terminal":
        return repo.mark_terminal(
            work_item_id=work_item_id,
            now=now,
            expected_lease_owner="worker-b",
            branch_loss=branch_loss,
        )
    if verb == "mark_failed":
        return repo.mark_failed(
            work_item_id=work_item_id,
            now=now,
            expected_lease_owner="worker-b",
            branch_loss=branch_loss,
        )
    if verb == "mark_blocked":
        return repo.mark_blocked(
            work_item_id=work_item_id,
            queue_key="queue-a",
            barrier_key=None,
            now=now,
            expected_lease_owner="worker-b",
        )
    if verb == "mark_pending_sink":
        return repo.mark_pending_sink(
            work_item_id=work_item_id,
            row_payload_json=payload,
            sink_name="replacement-sink",
            outcome="quarantined",
            path="quarantined",
            error_hash="replacement-error-hash",
            error_message="replacement error",
            now=now,
            expected_lease_owner="worker-b",
            branch_loss=branch_loss,
        )
    raise AssertionError(f"unknown normal disposition {verb!r}")


def _disposition_state_snapshot(
    engine: Tier1Engine, *, work_item_id: str
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    from elspeth.core.landscape.schema import scheduler_events_table

    with engine.connect() as conn:
        work_item = dict(
            conn.execute(select(token_work_items_table).where(token_work_items_table.c.work_item_id == work_item_id)).mappings().one()
        )
        coordination = dict(conn.execute(select(run_coordination_table).where(run_coordination_table.c.run_id == "run-1")).mappings().one())
        events = [
            dict(row)
            for row in conn.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.run_id == "run-1")
                .order_by(scheduler_events_table.c.recorded_at, scheduler_events_table.c.event_id)
            )
            .mappings()
            .all()
        ]
        branch_losses = [
            dict(row)
            for row in conn.execute(select(coalesce_branch_losses_table).order_by(coalesce_branch_losses_table.c.loss_id)).mappings().all()
        ]
    return work_item, coordination, events, branch_losses


def _make_scheduler_engine() -> Tier1Engine:
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:", echo=False)
    LandscapeDB._configure_sqlite(engine)
    LandscapeDB._verify_sqlite_pragmas(engine, "sqlite:///:memory:")
    metadata.create_all(engine)
    return Tier1Engine(engine)


def _insert_scheduler_prerequisites(engine: Tier1Engine, *, now: datetime) -> str:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    row_payload_json = TokenSchedulerRepository.serialize_row_payload(
        PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
    )
    with engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id="run-1",
                started_at=now,
                config_hash="config",
                settings_json="{}",
                canonical_version="v1",
                status="running",
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        conn.execute(
            insert(nodes_table).values(
                run_id="run-1",
                node_id="source-0",
                plugin_name="csv",
                node_type=NodeType.SOURCE.value,
                plugin_version="1.0",
                determinism="deterministic",
                config_hash="config",
                config_json="{}",
                registered_at=now,
            )
        )
        conn.execute(
            insert(nodes_table).values(
                run_id="run-1",
                node_id="normalize",
                plugin_name="identity",
                node_type=NodeType.TRANSFORM.value,
                plugin_version="1.0",
                determinism="deterministic",
                config_hash="config",
                config_json="{}",
                registered_at=now,
            )
        )
        conn.execute(
            insert(rows_table).values(
                row_id="row-1",
                run_id="run-1",
                source_node_id="source-0",
                row_index=0,
                source_row_index=0,
                ingest_sequence=0,
                source_data_hash="hash-row-1",
                created_at=now,
            )
        )
        conn.execute(
            insert(tokens_table).values(
                token_id="token-1",
                row_id="row-1",
                run_id="run-1",
                created_at=now,
            )
        )
        # Epoch-1 coordination seat for mark_pending_sink_terminal* and
        # terminalize_pending_sinks_with_terminal_outcomes (slice-4 REQUIRED).
        conn.execute(
            insert(run_coordination_table).values(
                run_id="run-1",
                leader_worker_id="test-leader",
                leader_epoch=1,
                leader_heartbeat_expires_at=now + timedelta(hours=1),
                updated_at=now,
            )
        )
    return row_payload_json


def _make_pending_sink(
    repo,
    *,
    run_id: str,
    token_id: str,
    row_id: str,
    payload: str,
    now: datetime,
):
    item = repo.enqueue_ready(
        run_id=run_id,
        token_id=token_id,
        row_id=row_id,
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    assert repo.claim_ready(run_id=run_id, lease_owner="worker-a", lease_seconds=30, now=now + timedelta(seconds=1)) is not None
    repo.mark_pending_sink(
        work_item_id=item.work_item_id,
        row_payload_json=payload,
        sink_name="sink-a",
        outcome="success",
        path="completed",
        error_hash=None,
        error_message=None,
        now=now + timedelta(seconds=2),
        expected_lease_owner="worker-a",
    )
    return item


def _scheduler_events(engine: Tier1Engine):
    from elspeth.core.landscape.schema import scheduler_events_table

    with engine.connect() as conn:
        return (
            conn.execute(
                select(scheduler_events_table)
                .where(scheduler_events_table.c.run_id == "run-1")
                .order_by(scheduler_events_table.c.recorded_at, scheduler_events_table.c.event_id)
            )
            .mappings()
            .all()
        )


def _stored_datetime(value: datetime) -> datetime:
    """SQLite's DateTime adapter returns raw row-mapping timestamps without tzinfo."""
    return value.replace(tzinfo=None)


def _insert_raw_recovery_event(
    engine: Tier1Engine,
    *,
    event_id: str,
    work_item_id: str,
    context_json: str,
    now: datetime,
) -> None:
    """Insert a RECOVER_EXPIRED_LEASE audit row directly, bypassing the store's writer."""
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.schema import scheduler_events_table

    with engine.begin() as conn:
        conn.execute(
            insert(scheduler_events_table).values(
                event_id=event_id,
                run_id="run-1",
                token_id="token-1",
                work_item_id=work_item_id,
                node_id=None,
                event_type=SchedulerEventType.RECOVER_EXPIRED_LEASE.value,
                from_status=None,
                to_status=TokenWorkStatus.READY.value,
                from_lease_owner=None,
                to_lease_owner=None,
                from_lease_expires_at=None,
                to_lease_expires_at=None,
                from_attempt=None,
                to_attempt=1,
                recorded_at=now,
                caller_owner=None,
                context_json=context_json,
            )
        )


def test_scheduler_event_store_wraps_database_rejection_as_landscape_record_error() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler.events import SchedulerEventStore

    class _RejectingConn:
        def execute(self, statement: object) -> object:
            raise OperationalError("INSERT INTO scheduler_events", {}, Exception("disk I/O error"))

    store = SchedulerEventStore()
    with pytest.raises(LandscapeRecordError, match="database rejected audit write: OperationalError"):
        store.record(
            cast(Connection, _RejectingConn()),
            event_type=SchedulerEventType.RECOVER_EXPIRED_LEASE,
            run_id="run-1",
            token_id="token-1",
            work_item_id="work-1",
            node_id=None,
            from_status=None,
            to_status=TokenWorkStatus.READY,
            from_lease_owner=None,
            to_lease_owner=None,
            from_attempt=None,
            to_attempt=0,
            recorded_at=datetime.now(UTC),
        )


def test_scheduler_event_store_rejects_unexpected_returned_identity() -> None:
    from elspeth.contracts.scheduler import SchedulerEventType, TokenWorkStatus
    from elspeth.core.landscape.scheduler.events import SchedulerEventStore

    class _WrongIdentityConn:
        def execute(self, statement: object) -> SimpleNamespace:
            return SimpleNamespace(scalar_one=lambda: "wrong-event-id")

    store = SchedulerEventStore()
    with pytest.raises(LandscapeRecordError, match="returned unexpected event_id"):
        store.record(
            cast(Connection, _WrongIdentityConn()),
            event_type=SchedulerEventType.RECOVER_EXPIRED_LEASE,
            run_id="run-1",
            token_id="token-1",
            work_item_id="work-1",
            node_id=None,
            from_status=None,
            to_status=TokenWorkStatus.READY,
            from_lease_owner=None,
            to_lease_owner=None,
            from_attempt=None,
            to_attempt=0,
            recorded_at=datetime.now(UTC),
        )


def test_recovery_event_reader_rejects_unparseable_context_json() -> None:
    from elspeth.core.landscape.scheduler.events import SchedulerEventStore

    engine = _make_scheduler_engine()
    now = datetime.now(UTC)
    _insert_scheduler_prerequisites(engine, now=now)
    _insert_raw_recovery_event(engine, event_id="evt-corrupt-parse", work_item_id="w-old", context_json="{not json", now=now)

    with engine.connect() as conn, pytest.raises(AuditIntegrityError, match="Corrupt scheduler recovery event context_json"):
        SchedulerEventStore.recovery_event_for_previous_work_item(conn, run_id="run-1", previous_work_item_id="w-old")


def test_recovery_event_reader_rejects_non_object_context_json() -> None:
    from elspeth.core.landscape.scheduler.events import SchedulerEventStore

    engine = _make_scheduler_engine()
    now = datetime.now(UTC)
    _insert_scheduler_prerequisites(engine, now=now)
    _insert_raw_recovery_event(
        engine, event_id="evt-corrupt-shape", work_item_id="w-old", context_json='["previous_work_item_id"]', now=now
    )

    with engine.connect() as conn, pytest.raises(AuditIntegrityError, match="expected object, got list"):
        SchedulerEventStore.recovery_event_for_previous_work_item(conn, run_id="run-1", previous_work_item_id="w-old")


def test_recovery_event_reader_rejects_non_string_previous_work_item_id() -> None:
    from elspeth.core.landscape.scheduler.events import SchedulerEventStore

    engine = _make_scheduler_engine()
    now = datetime.now(UTC)
    _insert_scheduler_prerequisites(engine, now=now)
    _insert_raw_recovery_event(
        engine, event_id="evt-corrupt-type", work_item_id="w-old", context_json='{"previous_work_item_id": 42}', now=now
    )

    with engine.connect() as conn, pytest.raises(AuditIntegrityError, match="previous_work_item_id must be str, got int"):
        SchedulerEventStore.recovery_event_for_previous_work_item(conn, run_id="run-1", previous_work_item_id="w-old")


def test_recovery_event_reader_skips_unrelated_events_and_returns_none() -> None:
    from elspeth.core.landscape.scheduler.events import SchedulerEventStore

    engine = _make_scheduler_engine()
    now = datetime.now(UTC)
    _insert_scheduler_prerequisites(engine, now=now)
    _insert_raw_recovery_event(engine, event_id="evt-no-context-key", work_item_id="w-old", context_json='{"unrelated": true}', now=now)
    _insert_raw_recovery_event(
        engine, event_id="evt-other-item", work_item_id="w-old", context_json='{"previous_work_item_id": "w-different"}', now=now
    )

    with engine.connect() as conn:
        result = SchedulerEventStore.recovery_event_for_previous_work_item(conn, run_id="run-1", previous_work_item_id="w-old")
    assert result is None
