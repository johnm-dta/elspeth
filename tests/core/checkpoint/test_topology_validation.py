# tests/core/checkpoint/test_topology_validation.py
"""Tests for checkpoint topology validation.

Critical audit integrity tests: Verifies that resume with modified
pipeline configuration is correctly rejected, preventing "one run,
two configs" corruption.

This file contains focused tests for simple linear pipeline scenarios
that complement the more complex DAG tests in test_compatibility_validator.py.
"""

from __future__ import annotations

from datetime import UTC, datetime

from elspeth.contracts import Checkpoint
from elspeth.contracts.enums import NodeType, RoutingMode
from elspeth.core.canonical import compute_full_topology_hash, stable_hash
from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator
from elspeth.core.dag import ExecutionGraph


class TestCheckpointTopologyValidation:
    """Tests for topology hash validation during resume."""

    def _create_linear_graph(self, num_transforms: int = 2) -> ExecutionGraph:
        """Create a simple linear graph: source -> transforms -> sink."""
        graph = ExecutionGraph()

        # Add source
        graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            config={"path": "data.csv"},
            plugin_name="csv_source",
        )

        # Add transforms
        prev_node = "source_1"
        for i in range(num_transforms):
            node_id = f"transform_{i}"
            graph.add_node(
                node_id,
                node_type=NodeType.TRANSFORM,
                config={"operation": f"op_{i}"},
                plugin_name="passthrough",
            )
            graph.add_edge(prev_node, node_id, label="continue", mode=RoutingMode.MOVE)
            prev_node = node_id

        # Add sink
        graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            config={"path": "output.csv"},
            plugin_name="csv_sink",
        )
        graph.add_edge(prev_node, "sink_1", label="continue", mode=RoutingMode.MOVE)

        return graph

    def _create_checkpoint_for_graph(
        self,
        graph: ExecutionGraph,
        node_id: str = "transform_0",
    ) -> Checkpoint:
        """Create a checkpoint with topology hash from the given graph."""
        topology_hash = compute_full_topology_hash(graph)
        node_info = graph.get_node_info(node_id)
        config_hash = stable_hash(node_info.config)

        return Checkpoint(
            checkpoint_id="cp-test123",
            run_id="run-123",
            token_id="token-456",
            node_id=node_id,
            sequence_number=100,
            created_at=datetime.now(UTC),
            upstream_topology_hash=topology_hash,
            checkpoint_node_config_hash=config_hash,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )

    def test_identical_graph_validates_successfully(self) -> None:
        """Checkpoint from identical graph should validate successfully."""
        original_graph = self._create_linear_graph(num_transforms=2)
        checkpoint = self._create_checkpoint_for_graph(original_graph)

        # Create identical graph for resume
        resume_graph = self._create_linear_graph(num_transforms=2)

        validator = CheckpointCompatibilityValidator()
        result = validator.validate(checkpoint, resume_graph)

        assert result.can_resume is True
        assert result.reason is None

    def test_added_transform_causes_validation_failure(self) -> None:
        """Adding a transform after checkpoint must fail validation.

        Scenario: Original pipeline had 2 transforms, resume has 3.
        Even though checkpoint node still exists, topology changed.
        """
        original_graph = self._create_linear_graph(num_transforms=2)
        checkpoint = self._create_checkpoint_for_graph(original_graph)

        # Resume graph has extra transform
        modified_graph = self._create_linear_graph(num_transforms=3)

        validator = CheckpointCompatibilityValidator()
        result = validator.validate(checkpoint, modified_graph)

        assert result.can_resume is False
        assert result.reason is not None
        assert "topology" in result.reason.lower() or "configuration changed" in result.reason.lower()

    def test_removed_transform_causes_validation_failure(self) -> None:
        """Removing a transform after checkpoint must fail validation."""
        original_graph = self._create_linear_graph(num_transforms=3)
        # Checkpoint at transform_0 (which exists in both)
        checkpoint = self._create_checkpoint_for_graph(original_graph, node_id="transform_0")

        # Resume graph has fewer transforms
        modified_graph = self._create_linear_graph(num_transforms=2)

        validator = CheckpointCompatibilityValidator()
        result = validator.validate(checkpoint, modified_graph)

        assert result.can_resume is False
        assert result.reason is not None
        assert "topology" in result.reason.lower() or "configuration changed" in result.reason.lower()

    def test_modified_sink_config_causes_validation_failure(self) -> None:
        """Changing sink configuration after checkpoint must fail validation.

        BUG-COMPAT-01 scenario: Even changes to sibling branches (different
        sinks) must invalidate the checkpoint because "one run = one config".
        """
        original_graph = self._create_linear_graph(num_transforms=2)
        checkpoint = self._create_checkpoint_for_graph(original_graph)

        # Create graph with different sink config
        modified_graph = ExecutionGraph()
        modified_graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            config={"path": "data.csv"},
            plugin_name="csv_source",
        )
        for i in range(2):
            node_id = f"transform_{i}"
            modified_graph.add_node(
                node_id,
                node_type=NodeType.TRANSFORM,
                config={"operation": f"op_{i}"},
                plugin_name="passthrough",
            )
            prev = "source_1" if i == 0 else f"transform_{i - 1}"
            modified_graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)

        # DIFFERENT sink config
        modified_graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            config={"path": "DIFFERENT_OUTPUT.csv"},  # Changed!
            plugin_name="csv_sink",
        )
        modified_graph.add_edge("transform_1", "sink_1", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        result = validator.validate(checkpoint, modified_graph)

        assert result.can_resume is False
        assert result.reason is not None
        assert "topology" in result.reason.lower() or "configuration changed" in result.reason.lower()

    def test_checkpoint_node_missing_causes_validation_failure(self) -> None:
        """Checkpoint node not existing in new graph must fail validation."""
        original_graph = self._create_linear_graph(num_transforms=3)
        # Checkpoint at transform_2 which won't exist in smaller graph
        checkpoint = self._create_checkpoint_for_graph(original_graph, node_id="transform_2")

        # Resume graph doesn't have transform_2
        modified_graph = self._create_linear_graph(num_transforms=2)

        validator = CheckpointCompatibilityValidator()
        result = validator.validate(checkpoint, modified_graph)

        assert result.can_resume is False
        # Should mention either node or topology mismatch
        assert result.reason is not None
        assert "node" in result.reason.lower() or "topology" in result.reason.lower()

    def test_checkpoint_node_config_changed_causes_validation_failure(self) -> None:
        """Changing checkpoint node's config must fail validation.

        This is separate from topology - even if graph structure is same,
        the specific node's config must also match.
        """
        original_graph = self._create_linear_graph(num_transforms=2)
        checkpoint = self._create_checkpoint_for_graph(original_graph, node_id="transform_0")

        # Create graph with same structure but different config at checkpoint node
        modified_graph = ExecutionGraph()
        modified_graph.add_node(
            "source_1",
            node_type=NodeType.SOURCE,
            config={"path": "data.csv"},
            plugin_name="csv_source",
        )
        # transform_0 has DIFFERENT config
        modified_graph.add_node(
            "transform_0",
            node_type=NodeType.TRANSFORM,
            config={"operation": "DIFFERENT_OP"},  # Changed!
            plugin_name="passthrough",
        )
        modified_graph.add_edge("source_1", "transform_0", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_node(
            "transform_1",
            node_type=NodeType.TRANSFORM,
            config={"operation": "op_1"},
            plugin_name="passthrough",
        )
        modified_graph.add_edge("transform_0", "transform_1", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_node(
            "sink_1",
            node_type=NodeType.SINK,
            config={"path": "output.csv"},
            plugin_name="csv_sink",
        )
        modified_graph.add_edge("transform_1", "sink_1", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        result = validator.validate(checkpoint, modified_graph)

        assert result.can_resume is False
        # Should fail on either topology (config affects hash) or node config check


class TestTopologyHashDeterminism:
    """Tests for topology hash determinism and edge cases."""

    def test_topology_hash_is_deterministic(self) -> None:
        """Same graph built twice should produce identical topology hash."""
        graph1 = ExecutionGraph()
        graph1.add_node("source", node_type=NodeType.SOURCE, config={"x": 1}, plugin_name="csv")
        graph1.add_node("sink", node_type=NodeType.SINK, config={"y": 2}, plugin_name="csv")
        graph1.add_edge("source", "sink", label="continue", mode=RoutingMode.MOVE)

        graph2 = ExecutionGraph()
        graph2.add_node("source", node_type=NodeType.SOURCE, config={"x": 1}, plugin_name="csv")
        graph2.add_node("sink", node_type=NodeType.SINK, config={"y": 2}, plugin_name="csv")
        graph2.add_edge("source", "sink", label="continue", mode=RoutingMode.MOVE)

        hash1 = compute_full_topology_hash(graph1)
        hash2 = compute_full_topology_hash(graph2)

        assert hash1 == hash2, "Same graph should produce same hash"

    def test_different_configs_produce_different_hashes(self) -> None:
        """Different node configs should produce different topology hashes."""
        graph1 = ExecutionGraph()
        graph1.add_node("source", node_type=NodeType.SOURCE, config={"x": 1}, plugin_name="csv")
        graph1.add_node("sink", node_type=NodeType.SINK, config={}, plugin_name="csv")
        graph1.add_edge("source", "sink", label="continue", mode=RoutingMode.MOVE)

        graph2 = ExecutionGraph()
        graph2.add_node("source", node_type=NodeType.SOURCE, config={"x": 2}, plugin_name="csv")  # Different!
        graph2.add_node("sink", node_type=NodeType.SINK, config={}, plugin_name="csv")
        graph2.add_edge("source", "sink", label="continue", mode=RoutingMode.MOVE)

        hash1 = compute_full_topology_hash(graph1)
        hash2 = compute_full_topology_hash(graph2)

        assert hash1 != hash2, "Different configs should produce different hashes"

    def test_different_edge_labels_produce_different_hashes(self) -> None:
        """Different edge labels should produce different topology hashes."""
        graph1 = ExecutionGraph()
        graph1.add_node("source", node_type=NodeType.SOURCE, config={}, plugin_name="csv")
        graph1.add_node("sink", node_type=NodeType.SINK, config={}, plugin_name="csv")
        graph1.add_edge("source", "sink", label="continue", mode=RoutingMode.MOVE)

        graph2 = ExecutionGraph()
        graph2.add_node("source", node_type=NodeType.SOURCE, config={}, plugin_name="csv")
        graph2.add_node("sink", node_type=NodeType.SINK, config={}, plugin_name="csv")
        graph2.add_edge("source", "sink", label="different_label", mode=RoutingMode.MOVE)  # Different!

        hash1 = compute_full_topology_hash(graph1)
        hash2 = compute_full_topology_hash(graph2)

        assert hash1 != hash2, "Different edge labels should produce different hashes"

    def test_different_routing_modes_produce_different_hashes(self) -> None:
        """Different routing modes should produce different topology hashes."""
        graph1 = ExecutionGraph()
        graph1.add_node("source", node_type=NodeType.SOURCE, config={}, plugin_name="csv")
        graph1.add_node("sink", node_type=NodeType.SINK, config={}, plugin_name="csv")
        graph1.add_edge("source", "sink", label="continue", mode=RoutingMode.MOVE)

        graph2 = ExecutionGraph()
        graph2.add_node("source", node_type=NodeType.SOURCE, config={}, plugin_name="csv")
        graph2.add_node("sink", node_type=NodeType.SINK, config={}, plugin_name="csv")
        graph2.add_edge("source", "sink", label="continue", mode=RoutingMode.COPY)  # Different!

        hash1 = compute_full_topology_hash(graph1)
        hash2 = compute_full_topology_hash(graph2)

        assert hash1 != hash2, "Different routing modes should produce different hashes"


class TestResumeAuditIntegrity:
    """Tests verifying that mismatched topology resume CRASHES, not silently continues.

    These tests verify the core audit integrity guarantee: resuming with a
    different pipeline topology is completely rejected, preventing "one run,
    two configs" corruption that would make audit results meaningless.
    """

    def test_mismatch_produces_clear_rejection_not_silent_accept(self) -> None:
        """Topology mismatch must produce explicit rejection with reason.

        CRITICAL: Silent acceptance of topology mismatch would corrupt audit trail.
        """
        # Original: source -> transform_A -> sink
        original = ExecutionGraph()
        original.add_node("source", node_type=NodeType.SOURCE, config={}, plugin_name="csv")
        original.add_node("transform_A", node_type=NodeType.TRANSFORM, config={"version": 1}, plugin_name="passthrough")
        original.add_node("sink", node_type=NodeType.SINK, config={}, plugin_name="csv")
        original.add_edge("source", "transform_A", label="continue", mode=RoutingMode.MOVE)
        original.add_edge("transform_A", "sink", label="continue", mode=RoutingMode.MOVE)

        checkpoint = Checkpoint(
            checkpoint_id="cp-audit-test",
            run_id="run-audit",
            token_id="token-audit",
            node_id="transform_A",
            sequence_number=50,
            created_at=datetime.now(UTC),
            upstream_topology_hash=compute_full_topology_hash(original),
            checkpoint_node_config_hash=stable_hash({"version": 1}),
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )

        # Modified: same structure, different transform config
        modified = ExecutionGraph()
        modified.add_node("source", node_type=NodeType.SOURCE, config={}, plugin_name="csv")
        modified.add_node("transform_A", node_type=NodeType.TRANSFORM, config={"version": 2}, plugin_name="passthrough")  # Changed!
        modified.add_node("sink", node_type=NodeType.SINK, config={}, plugin_name="csv")
        modified.add_edge("source", "transform_A", label="continue", mode=RoutingMode.MOVE)
        modified.add_edge("transform_A", "sink", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        result = validator.validate(checkpoint, modified)

        # MUST be explicit rejection
        assert result.can_resume is False, "Topology mismatch MUST be rejected"
        assert result.reason is not None, "Rejection MUST include reason"
        assert len(result.reason) > 0, "Reason MUST not be empty"

    def test_empty_graph_vs_populated_graph_rejected(self) -> None:
        """Resume with fundamentally different graph structure must be rejected."""
        # Original: substantial pipeline
        original = ExecutionGraph()
        original.add_node("source", node_type=NodeType.SOURCE, config={}, plugin_name="csv")
        original.add_node("t1", node_type=NodeType.TRANSFORM, config={}, plugin_name="passthrough")
        original.add_node("t2", node_type=NodeType.TRANSFORM, config={}, plugin_name="passthrough")
        original.add_node("sink", node_type=NodeType.SINK, config={}, plugin_name="csv")
        original.add_edge("source", "t1", label="continue", mode=RoutingMode.MOVE)
        original.add_edge("t1", "t2", label="continue", mode=RoutingMode.MOVE)
        original.add_edge("t2", "sink", label="continue", mode=RoutingMode.MOVE)

        checkpoint = Checkpoint(
            checkpoint_id="cp-size-test",
            run_id="run-size",
            token_id="token-size",
            node_id="t1",
            sequence_number=10,
            created_at=datetime.now(UTC),
            upstream_topology_hash=compute_full_topology_hash(original),
            checkpoint_node_config_hash=stable_hash({}),
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )

        # Modified: minimal pipeline (t1 exists but everything else is different)
        modified = ExecutionGraph()
        modified.add_node("source", node_type=NodeType.SOURCE, config={}, plugin_name="csv")
        modified.add_node("t1", node_type=NodeType.TRANSFORM, config={}, plugin_name="passthrough")
        modified.add_node("sink", node_type=NodeType.SINK, config={}, plugin_name="csv")
        modified.add_edge("source", "t1", label="continue", mode=RoutingMode.MOVE)
        modified.add_edge("t1", "sink", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        result = validator.validate(checkpoint, modified)

        assert result.can_resume is False, "Structurally different graph must be rejected"
