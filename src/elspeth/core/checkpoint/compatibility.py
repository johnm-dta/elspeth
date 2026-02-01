"""Checkpoint compatibility validation for resume operations.

Validates that a checkpoint can be safely resumed with the current
pipeline configuration by checking topological compatibility.
"""

import structlog

from elspeth.contracts import Checkpoint, ResumeCheck
from elspeth.core.canonical import compute_full_topology_hash, stable_hash
from elspeth.core.dag import ExecutionGraph


class CheckpointCompatibilityValidator:
    """Validates checkpoint compatibility with current execution graph.

    Separates topology validation logic from RecoveryManager's concern
    of "can this run be resumed?" (status checks, checkpoint existence).

    A checkpoint is compatible if:
    1. Checkpoint node exists in current graph
    2. Checkpoint node config hasn't changed
    3. FULL topology (ALL nodes + edges) is unchanged

    BUG-COMPAT-01 FIX: Changed from upstream-only to full DAG validation.
    In multi-sink DAGs, upstream-only validation allowed changes to sibling
    branches (other sink paths) to go undetected, causing a single run to
    contain outputs produced under different pipeline configurations.

    Now ANY topology change (including downstream or sibling branches)
    invalidates the checkpoint, enforcing: one run_id = one configuration.
    """

    def __init__(self) -> None:
        """Initialize validator."""
        self._logger = structlog.get_logger(__name__)

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
        # Validation 1: Checkpoint node must exist
        if not current_graph.has_node(checkpoint.node_id):
            return ResumeCheck(
                can_resume=False,
                reason=f"Checkpoint node '{checkpoint.node_id}' no longer exists in pipeline config. "
                "The transform at the checkpoint position may have been removed or its config changed.",
            )

        # Validation 2: Checkpoint node config must be unchanged
        current_node_info = current_graph.get_node_info(checkpoint.node_id)
        current_config_hash = stable_hash(current_node_info.config)

        if checkpoint.checkpoint_node_config_hash != current_config_hash:
            return ResumeCheck(
                can_resume=False,
                reason=f"Checkpoint node '{checkpoint.node_id}' configuration has changed. "
                "Resuming would process remaining rows differently than original run. "
                f"(Original config hash: {checkpoint.checkpoint_node_config_hash[:8]}..., "
                f"Current: {current_config_hash[:8]}...)",
            )

        # Validation 3: FULL topology (ALL nodes + edges) must be unchanged
        # BUG-COMPAT-01 fix: Validate entire DAG, not just upstream of checkpoint.
        # This catches changes to sibling branches in multi-sink DAGs.
        current_topology_hash = self.compute_full_topology_hash(current_graph)

        if checkpoint.upstream_topology_hash != current_topology_hash:
            # Provide detailed diagnostic
            return self._create_topology_mismatch_error(checkpoint, current_graph, checkpoint.upstream_topology_hash, current_topology_hash)

        # All validations passed
        return ResumeCheck(can_resume=True)

    def compute_full_topology_hash(
        self,
        graph: ExecutionGraph,
    ) -> str:
        """Delegate to canonical.compute_full_topology_hash().

        BUG-COMPAT-01: Changed from upstream-only to full DAG hashing.
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
