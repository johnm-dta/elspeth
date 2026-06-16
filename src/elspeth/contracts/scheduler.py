"""Durable token scheduler contracts."""

from __future__ import annotations

from collections.abc import Mapping
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


@dataclass(frozen=True)
class BatchMembershipSpec:
    """Aggregation-arm adoption payload: the ``batch_members`` row to write.

    Coalesce adoptions pass ``None`` (their held-arrival durable bookkeeping
    is a node_states row written by ``begin_node_state``; the adoption's
    durable payload is the CAS marker alone).
    """

    batch_id: str
    ordinal: int


@dataclass(frozen=True)
class BufferedOutcomeSpec:
    """Aggregation-arm adoption payload: the BUFFERED ``token_outcomes`` row.

    ``batch_id`` is the same batch as the membership spec — kept explicit
    because the outcome row carries its own ``batch_id`` column (ADR-019
    BUFFERED rule: ``batch_id`` REQUIRED for ``path='buffered'``).
    """

    batch_id: str
    context: Mapping[str, object] | None = None


@dataclass(frozen=True)
class BranchLossSpec:
    """Durable branch-loss record riding a lossy disposition (§E.5).

    Passed to ``mark_failed`` / ``mark_pending_sink`` when the disposed item
    is a fork-lineage branch feeding a coalesce: the loss row commits in the
    SAME lease-fenced transaction as the disposition (record-then-notify
    uniformity rule, design §E.5).
    """

    coalesce_name: str
    row_id: str
    branch_name: str
    token_id: str
    reason: str
    recorded_by: str


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
    # ADR-030 §E.2 (slice 3): the leader epoch that durably adopted this
    # BLOCKED barrier hold into executor memory via adopt_blocked_barrier_item.
    # NULL = intake-pending (deposited but not yet adopted); the per-iteration
    # journal-first intake filters on IS NULL.
    barrier_adopted_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class BarrierEmission:
    """One output emitted by an atomic barrier completion (``complete_barrier``).

    Two lanes consume this contract:

    - ``emitted_pending_sink``: a sink-bound output. If the token already has a
      BLOCKED row under the completing barrier (a buffered passthrough token),
      that row transitions BLOCKED -> PENDING_SINK in place and only the
      handoff bundle (``row_payload_json``, ``sink_name``, ``outcome``,
      ``path``, ``error_hash``, ``error_message``) is read. Otherwise a fresh
      PENDING_SINK row is inserted on the node_id-NULL terminal lane, which
      additionally requires ``row_id``, ``step_index`` and ``ingest_sequence``
      (and ``node_id`` must stay ``None``).
    - ``emitted_ready``: a READY continuation (an aggregation/coalesce output
      re-entering the DAG). Requires ``row_id``, ``step_index``,
      ``ingest_sequence`` and the target ``node_id`` cursor.

    Lineage and coalesce cursor fields mirror ``TokenWorkItem``: they are
    durable resume metadata for the inserted row.
    """

    token_id: str
    row_payload_json: str
    sink_name: str | None = None
    outcome: str | None = None
    path: str | None = None
    error_hash: str | None = None
    error_message: str | None = None
    row_id: str | None = None
    node_id: str | None = None
    step_index: int | None = None
    ingest_sequence: int | None = None
    queue_key: str | None = None
    barrier_key: str | None = None
    on_success_sink: str | None = None
    branch_name: str | None = None
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None
    coalesce_node_id: str | None = None
    coalesce_name: str | None = None
    attempt: int = 1

    def __post_init__(self) -> None:
        require_int(self.attempt, "attempt", min_value=1)
        require_int(self.step_index, "step_index", optional=True, min_value=0)
        require_int(self.ingest_sequence, "ingest_sequence", optional=True, min_value=0)


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
