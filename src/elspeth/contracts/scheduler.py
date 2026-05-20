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
    """A schedulable token continuation."""

    work_item_id: str
    run_id: str
    token_id: str
    row_id: str
    node_id: str
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
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
