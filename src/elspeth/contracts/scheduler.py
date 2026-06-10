"""Durable token scheduler contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from elspeth.contracts.freeze import require_int


class TokenWorkStatus(StrEnum):
    """Durable token work-item lifecycle states."""

    READY = "ready"
    LEASED = "leased"
    BLOCKED = "blocked"
    PENDING_SINK = "pending_sink"
    TERMINAL = "terminal"
    FAILED = "failed"


class SchedulerEventType(StrEnum):
    """Durable scheduler state-transition audit event types."""

    ENQUEUE = "enqueue"
    RESTORE_BLOCKED = "restore_blocked"
    CLAIM_READY = "claim_ready"
    CLAIM_PENDING_SINK = "claim_pending_sink"
    RECOVER_EXPIRED_LEASE = "recover_expired_lease"
    LEASE_LOST = "lease_lost"
    MARK_BLOCKED = "mark_blocked"
    MARK_TERMINAL = "mark_terminal"
    MARK_FAILED = "mark_failed"
    MARK_PENDING_SINK = "mark_pending_sink"
    MARK_PENDING_SINK_TERMINAL = "mark_pending_sink_terminal"
    MARK_BLOCKED_BARRIER_TERMINAL = "mark_blocked_barrier_terminal"


def _validate_scheduler_enum(value: object, enum_type: type, field_name: str) -> None:
    if value is not None and type(value) is not enum_type:
        raise TypeError(f"{field_name} must be {enum_type.__name__}, got {type(value).__name__}: {value!r}")


@dataclass(frozen=True, slots=True)
class TokenWorkItem:
    """A schedulable token continuation.

    The scheduler row is authoritative for resume. It carries the current row
    payload, node cursor, terminal sink context, token lineage, and coalesce
    cursor needed to rebuild a ``WorkItem`` in a fresh process without relying
    on in-memory ``pending_items`` state.
    """

    work_item_id: str
    run_id: str
    token_id: str
    row_id: str
    node_id: str | None
    step_index: int
    ingest_sequence: int
    row_payload_json: str
    status: TokenWorkStatus
    attempt: int
    available_at: datetime
    created_at: datetime
    updated_at: datetime
    queue_key: str | None = None
    barrier_key: str | None = None
    on_success_sink: str | None = None
    pending_sink_name: str | None = None
    pending_outcome: str | None = None
    pending_path: str | None = None
    pending_error_hash: str | None = None
    pending_error_message: str | None = None
    branch_name: str | None = None
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None
    coalesce_node_id: str | None = None
    coalesce_name: str | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    barrier_blocked_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class BlockedPendingSinkHandoff:
    """Sink handoff metadata for a BLOCKED scheduler row released by a barrier."""

    row_payload_json: str
    sink_name: str
    outcome: str
    path: str
    error_hash: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class SchedulerEvent:
    """Immutable audit event for a scheduler work-item transition."""

    event_id: str
    run_id: str
    token_id: str
    work_item_id: str
    event_type: SchedulerEventType
    to_status: TokenWorkStatus
    to_attempt: int
    recorded_at: datetime
    node_id: str | None = None
    from_status: TokenWorkStatus | None = None
    from_lease_owner: str | None = None
    to_lease_owner: str | None = None
    from_lease_expires_at: datetime | None = None
    to_lease_expires_at: datetime | None = None
    from_attempt: int | None = None
    caller_owner: str | None = None
    context_json: str = "{}"

    def __post_init__(self) -> None:
        _validate_scheduler_enum(self.event_type, SchedulerEventType, "event_type")
        _validate_scheduler_enum(self.from_status, TokenWorkStatus, "from_status")
        _validate_scheduler_enum(self.to_status, TokenWorkStatus, "to_status")
        require_int(self.from_attempt, "from_attempt", optional=True, min_value=0)
        require_int(self.to_attempt, "to_attempt", min_value=0)
