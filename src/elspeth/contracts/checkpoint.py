"""Checkpoint and recovery domain contracts.

These types are used for checkpoint validation and resume operations.
They are NOT persisted to the audit trail (those are in audit.py).
"""

from dataclasses import dataclass
from typing import Any

from elspeth.contracts.audit import Checkpoint


@dataclass(frozen=True)
class ResumeCheck:
    """Result of checking if a run can be resumed.

    Used by RecoveryManager and CheckpointCompatibilityValidator to
    communicate whether resume is possible and why/why not.
    """

    can_resume: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.can_resume and self.reason is not None:
            raise ValueError("can_resume=True should not have a reason")
        if not self.can_resume and self.reason is None:
            raise ValueError("can_resume=False must have a reason explaining why")


@dataclass
class ResumePoint:
    """Information needed to resume a run.

    Contains all the data needed by Orchestrator.resume() to continue
    processing from where a failed run left off.
    """

    checkpoint: Checkpoint
    token_id: str
    node_id: str
    sequence_number: int
    aggregation_state: dict[str, Any] | None
