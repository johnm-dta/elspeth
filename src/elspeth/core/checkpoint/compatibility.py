"""Checkpoint compatibility validation for resume operations.

Validates that a checkpoint can be safely resumed with the current
pipeline configuration by checking topological compatibility.
"""

from elspeth.contracts import Checkpoint, ResumeCheck
from elspeth.core.canonical import compute_full_topology_hash
from elspeth.core.dag import ExecutionGraph


class CheckpointCompatibilityValidator:
    """Validates checkpoint compatibility with current execution graph.

    Separates topology validation logic from RecoveryManager's concern
    of "can this run be resumed?" (status checks, checkpoint existence).

    A checkpoint is compatible iff the FULL topology (ALL nodes + edges)
    is unchanged. The full-topology hash embeds every node's id, plugin,
    type, and config hash, so node-removal and node-config drift are
    detected by the single hash comparison (the former per-anchor-node
    existence/config checks were strictly subsumed by it and were deleted
    with the token/node checkpoint anchor, F2 2026-06-10).

    In multi-sink DAGs, upstream-only validation allowed changes to sibling
    branches (other sink paths) to go undetected, causing a single run to
    contain outputs produced under different pipeline configurations.

    ANY topology change (including downstream or sibling branches)
    invalidates the checkpoint, enforcing: one run_id = one configuration.
    """

    def validate(
        self,
        checkpoint: Checkpoint,
        current_graph: ExecutionGraph,
    ) -> ResumeCheck:
        """Validate checkpoint compatibility with current graph topology.

        Args:
            checkpoint: The checkpoint to validate
            current_graph: Current execution graph from config

        Returns:
            ResumeCheck with can_resume=True if compatible,
            or can_resume=False with specific reason if not.
        """
        # FULL topology (ALL nodes + edges + per-node config hashes) must be
        # unchanged. Validate the entire DAG; this catches node removal,
        # node-config drift, and changes to sibling branches in multi-sink DAGs.
        current_topology_hash = self.compute_full_topology_hash(current_graph)
        checkpoint_topology_hash = checkpoint.full_topology_hash

        if checkpoint_topology_hash != current_topology_hash:
            # Provide detailed diagnostic
            return self._create_topology_mismatch_error(
                checkpoint,
                current_graph,
                checkpoint_topology_hash,
                current_topology_hash,
            )

        # All validations passed
        return ResumeCheck(can_resume=True)

    def compute_full_topology_hash(
        self,
        graph: ExecutionGraph,
    ) -> str:
        """Delegate to canonical.compute_full_topology_hash().

        Changed from upstream-only to full DAG hashing.
        """
        return compute_full_topology_hash(graph)

    def _create_topology_mismatch_error(
        self,
        checkpoint: Checkpoint,
        current_graph: ExecutionGraph,
        expected_hash: str,
        actual_hash: str,
    ) -> ResumeCheck:
        """Create detailed error message for topology mismatch.

        Args:
            checkpoint: The checkpoint being validated
            current_graph: Current execution graph
            expected_hash: Topology hash from checkpoint
            actual_hash: Topology hash from current graph

        Returns:
            ResumeCheck with detailed mismatch information.
        """
        # Could add more diagnostics here: which nodes changed, etc.
        # For now, provide hash comparison for audit trail
        return ResumeCheck(
            can_resume=False,
            reason=f"Pipeline configuration changed since checkpoint was created. "
            f"Resuming would produce outputs under a different configuration, "
            f"violating audit integrity (one run_id must map to one configuration). "
            f"(Expected topology hash: {expected_hash[:8]}..., Actual: {actual_hash[:8]}...)",
        )
