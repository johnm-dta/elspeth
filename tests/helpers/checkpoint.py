"""Checkpoint test helpers."""

from __future__ import annotations

from elspeth.contracts import Checkpoint, CheckpointDraft
from elspeth.contracts.barrier_scalars import BarrierScalars
from elspeth.contracts.coordination import CoordinationToken
from elspeth.core.checkpoint import CheckpointCompatibilityValidator
from elspeth.core.checkpoint.manager import CheckpointManager
from elspeth.core.dag import ExecutionGraph


def checkpoint_draft(
    *,
    run_id: str,
    sequence_number: int,
    graph: ExecutionGraph,
    barrier_scalars: BarrierScalars | None = None,
) -> CheckpointDraft:
    """Build the persistence-ready checkpoint record expected by the manager."""
    return CheckpointDraft(
        run_id=run_id,
        sequence_number=sequence_number,
        barrier_scalars=barrier_scalars,
        upstream_topology_hash=CheckpointCompatibilityValidator().compute_full_topology_hash(graph),
    )


def create_checkpoint(
    checkpoint_manager: CheckpointManager,
    *,
    run_id: str,
    sequence_number: int,
    graph: ExecutionGraph,
    barrier_scalars: BarrierScalars | None = None,
    coordination_token: CoordinationToken | None = None,
) -> Checkpoint:
    """Create a checkpoint through the persistence-ready draft boundary."""
    return checkpoint_manager.create_checkpoint(
        draft=checkpoint_draft(
            run_id=run_id,
            sequence_number=sequence_number,
            graph=graph,
            barrier_scalars=barrier_scalars,
        ),
        coordination_token=coordination_token,
    )
