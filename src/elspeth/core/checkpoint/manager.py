"""CheckpointManager for creating and loading checkpoints."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from sqlalchemy import asc, delete, desc, select

from elspeth.contracts import Checkpoint
from elspeth.contracts.errors import AuditIntegrityError, OrchestrationInvariantError
from elspeth.core.canonical import compute_full_topology_hash, stable_hash
from elspeth.core.checkpoint.serialization import checkpoint_dumps
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import checkpoints_table, node_states_table, tokens_table

logger = logging.getLogger(__name__)

_LARGE_AGGREGATION_CHECKPOINT_BYTES = 1_000_000
_MAX_CHECKPOINT_STATE_BYTES = 10_000_000

if TYPE_CHECKING:
    from elspeth.contracts.aggregation_checkpoint import AggregationCheckpointState
    from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState
    from elspeth.core.dag import ExecutionGraph


class IncompatibleCheckpointError(Exception):
    """Raised when attempting to load a checkpoint from an incompatible version."""

    pass


class CheckpointCorruptionError(Exception):
    """Raised when checkpoint data integrity verification fails.

    This indicates corruption in the audit trail - a Tier 1 failure
    that must be treated as unrecoverable per CLAUDE.md data manifesto.
    """

    pass


def _validate_checkpoint_state_json_size(
    *,
    state_name: Literal["aggregation", "coalesce"],
    serialized: str,
    total_rows: int | None = None,
    node_count: int | None = None,
    pending_joins: int | None = None,
) -> None:
    """Validate serialized checkpoint state at the single persistence boundary."""
    serialized_bytes = len(serialized.encode("utf-8"))
    size_mb = serialized_bytes / 1_000_000

    if state_name == "aggregation" and serialized_bytes > _LARGE_AGGREGATION_CHECKPOINT_BYTES:
        logger.warning(
            "Large checkpoint: %.1fMB for %d buffered rows across %d nodes",
            size_mb,
            total_rows or 0,
            node_count or 0,
        )

    if serialized_bytes <= _MAX_CHECKPOINT_STATE_BYTES:
        return

    if state_name == "aggregation":
        raise OrchestrationInvariantError(
            f"Checkpoint size {size_mb:.1f}MB exceeds 10MB limit. "
            f"Buffer contains {total_rows or 0} total rows across {node_count or 0} nodes. "
            f"Solutions: (1) Reduce aggregation count trigger to <5000 rows, "
            f"(2) Reduce row_data payload size, or (3) Implement checkpoint retention "
            f"policy"
        )

    raise RuntimeError(f"Coalesce checkpoint size {size_mb:.1f}MB exceeds 10MB limit. Pending joins: {pending_joins or 0}.")


class CheckpointManager:
    """Manages checkpoint creation and retrieval.

    Checkpoints capture run progress at row boundaries, enabling
    resume after crash. Each checkpoint records:
    - Which token was being processed
    - Which node it was at
    - A monotonic sequence number for ordering
    - Optional aggregation state for stateful plugins
    """

    def __init__(self, db: LandscapeDB) -> None:
        """Initialize with Landscape database.

        Args:
            db: LandscapeDB instance for storage
        """
        self._db = db

    def create_checkpoint(
        self,
        run_id: str,
        token_id: str,
        node_id: str,
        sequence_number: int,
        graph: ExecutionGraph,
        aggregation_state: AggregationCheckpointState | None = None,
        coalesce_state: CoalesceCheckpointState | None = None,
    ) -> Checkpoint:
        """Create a checkpoint at current progress point.

        Args:
            run_id: The run being checkpointed
            token_id: Current token being processed
            node_id: Current node in the pipeline
            sequence_number: Monotonic progress marker
            graph: Execution graph for topology validation (REQUIRED)
            aggregation_state: Optional serializable aggregation buffers
            coalesce_state: Optional serializable pending coalesce state

        Returns:
            The created Checkpoint

        Raises:
            ValueError: If graph is None or node_id not in graph
        """
        # Validate parameters early
        if graph is None:
            raise ValueError("graph parameter is required for checkpoint creation")
        if not graph.has_node(node_id):
            raise ValueError(f"node_id '{node_id}' does not exist in graph")

        # All checkpoint data generation happens INSIDE transaction for atomicity
        with self._db.engine.begin() as conn:
            # Verify token belongs to the specified run (Tier 1 invariant).
            # Cross-run checkpoint contamination is audit corruption.
            token_row = conn.execute(select(tokens_table.c.run_id).where(tokens_table.c.token_id == token_id)).fetchone()
            if token_row is None:
                raise AuditIntegrityError(f"Cannot create checkpoint: token '{token_id}' does not exist")
            if token_row.run_id != run_id:
                raise AuditIntegrityError(
                    f"Cannot create checkpoint: token '{token_id}' belongs to run "
                    f"'{token_row.run_id}' but checkpoint targets run '{run_id}'. "
                    f"Cross-run checkpoint contamination is audit corruption."
                )
            existing_sequence = conn.execute(
                select(checkpoints_table.c.checkpoint_id)
                .where((checkpoints_table.c.run_id == run_id) & (checkpoints_table.c.sequence_number == sequence_number))
                .limit(1)
            ).fetchone()
            if existing_sequence is not None:
                raise OrchestrationInvariantError(
                    f"Duplicate checkpoint sequence_number={sequence_number} for run '{run_id}' "
                    f"would make resume ordering ambiguous; existing checkpoint={existing_sequence.checkpoint_id}"
                )

            # Generate IDs and timestamps within transaction boundary
            checkpoint_id = f"cp-{uuid.uuid4().hex}"
            created_at = datetime.now(UTC)

            # Prepare aggregation state JSON
            # checkpoint_dumps() handles:
            # - datetime serialization with type tags for round-trip fidelity
            # - NaN/Infinity rejection per CLAUDE.md audit integrity requirements
            # Note: We don't use canonical_json because it normalizes floats to integers,
            # breaking round-trip for aggregation state
            agg_json: str | None = None
            if aggregation_state is not None:
                agg_json = checkpoint_dumps(aggregation_state.to_dict())
                _validate_checkpoint_state_json_size(
                    state_name="aggregation",
                    serialized=agg_json,
                    total_rows=sum(len(node.tokens) for node in aggregation_state.nodes.values()),
                    node_count=len(aggregation_state.nodes),
                )

            coalesce_json: str | None = None
            if coalesce_state is not None:
                coalesce_json = checkpoint_dumps(coalesce_state.to_dict())
                _validate_checkpoint_state_json_size(
                    state_name="coalesce",
                    serialized=coalesce_json,
                    pending_joins=len(coalesce_state.pending),
                )

            # Compute topology hashes inside transaction
            # This ensures hash matches graph state at exact moment of checkpoint creation
            # Use FULL topology hash instead of upstream-only hash.
            # This ensures changes to ANY branch (including sibling sink branches)
            # are detected during resume validation, enforcing "one run = one config".
            upstream_topology_hash = compute_full_topology_hash(graph)
            node_info = graph.get_node_info(node_id)
            checkpoint_node_config_hash = stable_hash(node_info.config)

            conn.execute(
                checkpoints_table.insert().values(
                    checkpoint_id=checkpoint_id,
                    run_id=run_id,
                    token_id=token_id,
                    node_id=node_id,
                    sequence_number=sequence_number,
                    aggregation_state_json=agg_json,
                    coalesce_state_json=coalesce_json,
                    created_at=created_at,
                    upstream_topology_hash=upstream_topology_hash,
                    checkpoint_node_config_hash=checkpoint_node_config_hash,
                    format_version=Checkpoint.CURRENT_FORMAT_VERSION,
                )
            )
            # begin() auto-commits on clean exit, auto-rollbacks on exception

        return Checkpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            token_id=token_id,
            node_id=node_id,
            sequence_number=sequence_number,
            created_at=created_at,
            upstream_topology_hash=upstream_topology_hash,
            checkpoint_node_config_hash=checkpoint_node_config_hash,
            aggregation_state_json=agg_json,
            coalesce_state_json=coalesce_json,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )

    def get_latest_checkpoint(self, run_id: str) -> Checkpoint | None:
        """Get the most recent checkpoint for a run.

        Args:
            run_id: The run to get checkpoint for

        Returns:
            Latest Checkpoint or None if no checkpoints exist

        Raises:
            IncompatibleCheckpointError: If checkpoint predates deterministic node IDs
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
                token_id=result.token_id,
                node_id=result.node_id,
                sequence_number=result.sequence_number,
                created_at=result.created_at,
                upstream_topology_hash=result.upstream_topology_hash,
                checkpoint_node_config_hash=result.checkpoint_node_config_hash,
                aggregation_state_json=result.aggregation_state_json,
                coalesce_state_json=result.coalesce_state_json,
                format_version=result.format_version,  # None for legacy checkpoints
            )
        except ValueError as e:
            raise CheckpointCorruptionError(
                f"Checkpoint corruption detected for run '{run_id}', checkpoint '{result.checkpoint_id}': {e}"
            ) from e

        # Validate checkpoint compatibility before returning
        self._validate_checkpoint_compatibility(checkpoint)

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
                        token_id=r.token_id,
                        node_id=r.node_id,
                        sequence_number=r.sequence_number,
                        created_at=r.created_at,
                        upstream_topology_hash=r.upstream_topology_hash,
                        checkpoint_node_config_hash=r.checkpoint_node_config_hash,
                        aggregation_state_json=r.aggregation_state_json,
                        coalesce_state_json=r.coalesce_state_json,
                        format_version=r.format_version,  # None for legacy checkpoints
                    )
                )
            except ValueError as e:
                raise CheckpointCorruptionError(
                    f"Checkpoint corruption detected for run '{run_id}', checkpoint '{r.checkpoint_id}': {e}"
                ) from e
        return checkpoints

    def delete_checkpoints(self, run_id: str) -> int:
        """Delete all checkpoints for a completed run.

        Called after successful run completion to clean up.

        Checkpoints referenced by node_states.resume_checkpoint_id are PRESERVED
        as audit evidence: they are the provenance anchor for resumed node writes
        (used by explain() to distinguish resume re-drives from run-1 retries via
        `resume_checkpoint_id IS NOT NULL`). Deleting them would violate the FK
        constraint and destroy the audit lineage they represent. Only progress
        checkpoints not referenced by any node_state are deleted.

        Args:
            run_id: The run to clean up

        Returns:
            Number of checkpoints deleted
        """
        # Subquery: checkpoint_ids from THIS run that are referenced as resume provenance
        # anchors. Scoped to run_id to avoid a full-table scan on node_states as the audit
        # DB grows. These must not be deleted — they are part of the audit trail (FK from
        # node_states.resume_checkpoint_id). Deleting them would (a) violate the FK
        # constraint and (b) destroy the query-separation property that explain() relies
        # on to identify resumed node writes (`resume_checkpoint_id IS NOT NULL`).
        referenced_ids = (
            select(node_states_table.c.resume_checkpoint_id)
            .where(
                node_states_table.c.run_id == run_id,
                node_states_table.c.resume_checkpoint_id.is_not(None),
            )
            .distinct()
        ).scalar_subquery()

        with self._db.engine.begin() as conn:
            result = conn.execute(
                delete(checkpoints_table).where(
                    checkpoints_table.c.run_id == run_id,
                    ~checkpoints_table.c.checkpoint_id.in_(referenced_ids),
                )
            )
            # begin() auto-commits on clean exit, auto-rollbacks on exception
            return result.rowcount

    def _validate_checkpoint_compatibility(self, checkpoint: Checkpoint) -> None:
        """Verify checkpoint was created with compatible format version.

        CRITICAL: Node IDs changed from random UUID to deterministic hash-based
        in format version 2. Old checkpoints cannot be resumed because node IDs
        will not match between checkpoint and current graph.

        Args:
            checkpoint: Checkpoint to validate

        Raises:
            IncompatibleCheckpointError: If checkpoint format version is incompatible
        """
        if checkpoint.format_version is None:
            raise IncompatibleCheckpointError(
                f"Checkpoint '{checkpoint.checkpoint_id}' is missing format_version. "
                "Resume not supported for unversioned checkpoints. "
                "Please restart pipeline from beginning."
            )

        # CRITICAL: Reject BOTH older AND newer versions - cross-version resume is unsupported
        if checkpoint.format_version != Checkpoint.CURRENT_FORMAT_VERSION:
            raise IncompatibleCheckpointError(
                f"Checkpoint '{checkpoint.checkpoint_id}' has incompatible format version "
                f"(checkpoint: v{checkpoint.format_version}, current: v{Checkpoint.CURRENT_FORMAT_VERSION}). "
                "Resume requires exact format version match. "
                "Please restart pipeline from beginning."
            )
