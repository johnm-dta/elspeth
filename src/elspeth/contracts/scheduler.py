"""Durable token scheduler contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class TokenWorkStatus(StrEnum):
    """Durable token work-item lifecycle states."""

    READY = "ready"
    LEASED = "leased"
    WAITING = "waiting"
    BLOCKED = "blocked"
    TERMINAL = "terminal"
    FAILED = "failed"


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
    branch_name: str | None = None
    fork_group_id: str | None = None
    join_group_id: str | None = None
    expand_group_id: str | None = None
    coalesce_node_id: str | None = None
    coalesce_name: str | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
