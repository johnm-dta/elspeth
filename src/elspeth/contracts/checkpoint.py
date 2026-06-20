"""Checkpoint and recovery domain contracts.

These types are used for checkpoint validation and resume operations.
They are NOT persisted to the audit trail (those are in audit.py).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from elspeth.contracts.audit import Checkpoint
from elspeth.contracts.barrier_scalars import BarrierScalars
from elspeth.contracts.freeze import freeze_fields, require_int
from elspeth.contracts.types import NodeID


@dataclass(frozen=True, slots=True)
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


@dataclass(frozen=True, slots=True)
class ResumePoint:
    """Information needed to resume a run.

    Contains all the data needed by Orchestrator.resume() to continue
    processing from where a failed run left off.
    """

    checkpoint: Checkpoint
    sequence_number: int
    barrier_scalars: BarrierScalars | None = None

    def __post_init__(self) -> None:
        """Validate resume point fields — Tier 1 crash on invalid data.

        Per CLAUDE.md Data Manifesto: Checkpoints are Tier 1 audit data.
        Wrong types indicate corrupted checkpoint data — crash immediately
        with distinct error messages.
        """
        if not isinstance(self.checkpoint, Checkpoint):
            raise TypeError(f"ResumePoint.checkpoint must be Checkpoint, got {type(self.checkpoint).__name__}")
        require_int(self.sequence_number, "ResumePoint.sequence_number", min_value=0)
        if self.barrier_scalars is not None and not isinstance(self.barrier_scalars, BarrierScalars):
            raise TypeError(f"ResumePoint.barrier_scalars must be BarrierScalars or None, got {type(self.barrier_scalars).__name__}")
        # Invariant: the duplicated field must match the embedded Checkpoint.
        # It exists for convenience access but is derived data, not an
        # independent input. Mismatch = corrupted construction.
        if self.sequence_number != self.checkpoint.sequence_number:
            raise ValueError(
                f"ResumePoint.sequence_number ({self.sequence_number}) does not match "
                f"checkpoint.sequence_number ({self.checkpoint.sequence_number})"
            )


@dataclass(frozen=True, slots=True)
class ResumedRow:
    """A single row recovered from the audit trail for resume processing.

    Per ADR-025 Decision §4: every persisted row carries
    ``rows.source_node_id`` (NOT NULL in the schema), and resume must
    look up that row's schema contract via source node identity. The
    previous ``(row_id, row_index, row_data) | (row_id, row_index,
    source_node_id, row_data)`` 3|4-tuple union — discriminated at the
    consumer by ``len()`` — was the carrier for the singular/plural
    dual-truth surface that ADR-025 deletes. This dataclass replaces
    both shapes with a single, non-optional carrier for
    ``source_node_id``.

    ``row_data`` is typed as ``Mapping[str, Any]`` and deep-frozen in
    ``__post_init__`` (per CLAUDE.md's frozen-dataclass deep-freeze
    contract — no loose mutable dicts on frozen records). Consumers
    that need a mutable dict (notably ``PipelineRow``, which demands
    ``type(data) is dict`` as a Tier-1 anti-coercion check) construct
    one explicitly at the boundary via ``dict(row.row_data)``;
    ``PipelineRow.__init__`` then immediately re-freezes the copy
    via ``deep_freeze``. No mutation surface exists at any point in
    the chain.
    """

    row_id: str
    row_index: int
    source_node_id: NodeID
    row_data: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Tier-1 read-side validation — crash on garbage from our own DB.

        Per CLAUDE.md Data Manifesto: rows recovered from the audit
        trail are Tier 1 data. Wrong types or empty identifiers
        indicate corruption.
        """
        if not isinstance(self.row_id, str):
            raise TypeError(f"ResumedRow.row_id must be str, got {type(self.row_id).__name__}: {self.row_id!r}")
        if not self.row_id:
            raise ValueError("ResumedRow.row_id must not be empty")
        require_int(self.row_index, "ResumedRow.row_index", min_value=0)
        if not isinstance(self.source_node_id, str):
            raise TypeError(
                f"ResumedRow.source_node_id must be NodeID (str), got {type(self.source_node_id).__name__}: {self.source_node_id!r}"
            )
        if not self.source_node_id:
            raise ValueError("ResumedRow.source_node_id must not be empty")
        if not isinstance(self.row_data, Mapping):
            raise TypeError(f"ResumedRow.row_data must be Mapping, got {type(self.row_data).__name__}")
        freeze_fields(self, "row_data")
