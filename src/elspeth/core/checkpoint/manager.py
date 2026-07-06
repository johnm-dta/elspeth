"""CheckpointManager for creating and loading checkpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import asc, delete, desc, select

from elspeth.contracts import Checkpoint, CheckpointDraft
from elspeth.contracts.coordination import DEFAULT_RUN_LIVENESS_WINDOW_SECONDS
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.core.checkpoint.compatibility import IncompatibleCheckpointError as IncompatibleCheckpointError
from elspeth.core.checkpoint.serialization import checkpoint_dumps
from elspeth.core.landscape.database import LandscapeDB, begin_write
from elspeth.core.landscape.run_coordination_repository import fenced_leader_transaction
from elspeth.core.landscape.schema import checkpoints_table

_MAX_BARRIER_SCALARS_BYTES = 10_000_000

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from sqlalchemy.engine import Connection

    from elspeth.contracts.coordination import CoordinationToken


class CheckpointCorruptionError(Exception):
    """Raised when checkpoint data integrity verification fails.

    This indicates corruption in the audit trail - a Tier 1 failure
    that must be treated as unrecoverable per CLAUDE.md data manifesto.
    """

    pass


def _validate_barrier_scalars_json_size(serialized: str) -> None:
    """Hard-fail size guard at the single checkpoint persistence boundary.

    Post-F1, the checkpoint carries only scalar barrier metadata (two float
    trigger latches per in-flight aggregation node plus lost-branch records
    per pending coalesce key) — payloads are tiny by construction. A payload
    anywhere near the limit indicates corrupted state construction upstream,
    not a large pipeline, so this is a crash, not a warning.
    """
    serialized_bytes = len(serialized.encode("utf-8"))
    if serialized_bytes <= _MAX_BARRIER_SCALARS_BYTES:
        return
    raise OrchestrationInvariantError(
        f"Checkpoint barrier_scalars size {serialized_bytes / 1_000_000:.1f}MB exceeds 10MB limit. "
        f"Barrier scalars carry only trigger latches and lost-branch records; "
        f"a payload this large indicates a bug in barrier state construction."
    )


class CheckpointManager:
    """Manages checkpoint creation and retrieval.

    Checkpoints capture run progress at sink-durability boundaries,
    enabling resume after crash. Each checkpoint records:
    - A monotonic sequence number for ordering
    - The full-topology hash for compatibility validation
    - Optional scalar barrier metadata (BarrierScalars) for in-flight
      aggregation/coalesce barriers — buffered tokens themselves live in
      token_work_items journal BLOCKED rows (F1 durability unification)
    """

    def __init__(self, db: LandscapeDB) -> None:
        """Initialize with Landscape database.

        Args:
            db: LandscapeDB instance for storage
        """
        self._db = db

    def _fenced_or_plain_write(
        self,
        *,
        coordination_token: CoordinationToken | None,
        verb: str,
    ) -> AbstractContextManager[Connection]:
        """One write-intent transaction, leader-fenced when a token is supplied.

        ADR-030 §C.4 row 5: the verify-and-extend epoch fence runs as the
        FIRST statement of the checkpoint write transaction — a deposed
        leader's checkpoint INSERT/DELETE is refused before the
        duplicate-sequence guard or the UNIQUE constraint is even reached
        (both stay beneath as the durable backstop). ``None`` preserves the
        unfenced legacy arm for direct repository-level callers (tests,
        tooling); the orchestrator's CheckpointCoordinator always threads the
        token it bound at run/resume start.
        """
        if coordination_token is None:
            return begin_write(self._db.engine)
        return fenced_leader_transaction(
            self._db.engine,
            token=coordination_token,
            now=datetime.now(UTC),
            window_seconds=DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
            verb=verb,
        )

    def create_checkpoint(
        self,
        *,
        draft: CheckpointDraft,
        coordination_token: CoordinationToken | None = None,
    ) -> Checkpoint:
        """Create a checkpoint at current progress point.

        Args:
            draft: Persistence-ready checkpoint data. The topology hash is
                computed by the orchestration/compatibility boundary before
                reaching this repository.
            coordination_token: Leader fencing token (ADR-030). When
                supplied, the verify-and-extend epoch fence is the first
                statement of the write transaction; a stale epoch raises
                ``RunLeadershipLostError`` with zero mutation.

        Returns:
            The created Checkpoint
        """
        if not isinstance(draft, CheckpointDraft):
            raise TypeError(f"draft must be CheckpointDraft, got {type(draft).__name__}")

        # All checkpoint data generation happens INSIDE transaction for atomicity
        with self._fenced_or_plain_write(coordination_token=coordination_token, verb="create_checkpoint") as conn:
            existing_sequence = conn.execute(
                select(checkpoints_table.c.checkpoint_id)
                .where((checkpoints_table.c.run_id == draft.run_id) & (checkpoints_table.c.sequence_number == draft.sequence_number))
                .limit(1)
            ).fetchone()
            if existing_sequence is not None:
                raise OrchestrationInvariantError(
                    f"Duplicate checkpoint sequence_number={draft.sequence_number} for run '{draft.run_id}' "
                    f"would make resume ordering ambiguous; existing checkpoint={existing_sequence.checkpoint_id}"
                )

            # Generate IDs and timestamps within transaction boundary
            checkpoint_id = f"cp-{uuid.uuid4().hex}"
            created_at = datetime.now(UTC)

            # Serialize barrier scalars JSON.
            # checkpoint_dumps() handles:
            # - NaN/Infinity rejection per CLAUDE.md audit integrity requirements
            # Note: We don't use canonical_json because it normalizes floats to
            # integers, breaking round-trip for the float trigger-offset latches.
            scalars_json: str | None = None
            if draft.barrier_scalars is not None and draft.barrier_scalars.has_state:
                scalars_json = checkpoint_dumps(draft.barrier_scalars.to_dict())
                _validate_barrier_scalars_json_size(scalars_json)

            conn.execute(
                checkpoints_table.insert().values(
                    checkpoint_id=checkpoint_id,
                    run_id=draft.run_id,
                    sequence_number=draft.sequence_number,
                    barrier_scalars_json=scalars_json,
                    created_at=created_at,
                    upstream_topology_hash=draft.upstream_topology_hash,
                    format_version=draft.format_version,
                )
            )
            # begin() auto-commits on clean exit, auto-rollbacks on exception

        return Checkpoint(
            checkpoint_id=checkpoint_id,
            run_id=draft.run_id,
            sequence_number=draft.sequence_number,
            created_at=created_at,
            upstream_topology_hash=draft.upstream_topology_hash,
            barrier_scalars_json=scalars_json,
            format_version=draft.format_version,
        )

    def get_latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for a run.

        Args:
            run_id: The run to get checkpoint for

        Returns:
            Latest Checkpoint or None if no checkpoints exist

        This is a raw persistence read. Resume compatibility policy is enforced
        by CheckpointCompatibilityValidator, not by the repository boundary.
        """
        with self._db.engine.connect() as conn:
            result = conn.execute(
                select(checkpoints_table)
                .where(checkpoints_table.c.run_id == run_id)
                .order_by(desc(checkpoints_table.c.sequence_number))
                .limit(1)
            ).fetchone()

        if result is None:
            return None

        try:
            checkpoint = Checkpoint(
                checkpoint_id=result.checkpoint_id,
                run_id=result.run_id,
                sequence_number=result.sequence_number,
                created_at=result.created_at,
                upstream_topology_hash=result.upstream_topology_hash,
                barrier_scalars_json=result.barrier_scalars_json,
                format_version=result.format_version,  # None for legacy checkpoints
            )
        except ValueError as e:
            raise CheckpointCorruptionError(
                f"Checkpoint corruption detected for run '{run_id}', checkpoint '{result.checkpoint_id}': {e}"
            ) from e

        return checkpoint

    def get_checkpoints(self, run_id: str) -> list[Checkpoint]:
        """Get all checkpoints for a run, ordered by sequence.

        Args:
            run_id: The run to get checkpoints for

        Returns:
            List of Checkpoints ordered by sequence_number
        """
        with self._db.engine.connect() as conn:
            results = conn.execute(
                select(checkpoints_table).where(checkpoints_table.c.run_id == run_id).order_by(asc(checkpoints_table.c.sequence_number))
            ).fetchall()

        checkpoints = []
        for r in results:
            try:
                checkpoints.append(
                    Checkpoint(
                        checkpoint_id=r.checkpoint_id,
                        run_id=r.run_id,
                        sequence_number=r.sequence_number,
                        created_at=r.created_at,
                        upstream_topology_hash=r.upstream_topology_hash,
                        barrier_scalars_json=r.barrier_scalars_json,
                        format_version=r.format_version,  # None for legacy checkpoints
                    )
                )
            except ValueError as e:
                raise CheckpointCorruptionError(
                    f"Checkpoint corruption detected for run '{run_id}', checkpoint '{r.checkpoint_id}': {e}"
                ) from e
        return checkpoints

    def delete_checkpoints(self, run_id: str, *, coordination_token: CoordinationToken | None = None) -> int:
        """Delete all checkpoints for a completed run.

        Called after successful run completion to clean up. Checkpoints are deletable
        progress state — node_states.resume_checkpoint_id is a marker-only id (no FK),
        so the resume-provenance fact endures on node_states even after its checkpoint
        row is purged here.

        Args:
            run_id: The run to clean up
            coordination_token: Leader fencing token (ADR-030 §C.4 row 5) —
                a deposed leader must not destroy the new leader's resume
                anchors. Fence-first when supplied.

        Returns:
            Number of checkpoints deleted
        """
        with self._fenced_or_plain_write(coordination_token=coordination_token, verb="delete_checkpoints") as conn:
            result = conn.execute(delete(checkpoints_table).where(checkpoints_table.c.run_id == run_id))
            # begin() auto-commits on clean exit, auto-rollbacks on exception
            return result.rowcount
