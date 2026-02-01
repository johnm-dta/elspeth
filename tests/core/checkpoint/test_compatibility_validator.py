"""Tests for checkpoint topology compatibility validation.

BUG-COMPAT-01 FIX: Tests now validate FULL DAG topology, not just upstream.
ANY topology change (upstream, downstream, or sibling branches) invalidates
checkpoint resume to enforce: one run_id = one configuration.

This prevents audit integrity violations where a single run could contain
outputs produced under different pipeline configurations.
"""

from datetime import UTC, datetime

from elspeth.contracts import Checkpoint, NodeType, RoutingMode
from elspeth.core.canonical import stable_hash
from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator
from elspeth.core.dag import ExecutionGraph


class TestForkJoinTopology:
    """Tests for fork/join (coalesce) topologies - CRITICAL GAP identified by QA review."""

    def test_resume_rejects_fork_path_change(self) -> None:
        """Changing one fork path should reject if it feeds checkpoint."""
        # Original graph: source → gate → path_A → coalesce → checkpoint
        #                              → path_B ────┘
        original_graph = ExecutionGraph()
        original_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        original_graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate")
        original_graph.add_node("path_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"version": "v1"})
        original_graph.add_node("path_B", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"version": "v1"})
        original_graph.add_node("coalesce", node_type=NodeType.COALESCE, plugin_name="coalesce")
        original_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})

        original_graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("gate", "path_A", label="branch_a", mode=RoutingMode.COPY)
        original_graph.add_edge("gate", "path_B", label="branch_b", mode=RoutingMode.COPY)
        original_graph.add_edge("path_A", "coalesce", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("path_B", "coalesce", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("coalesce", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)

        # Create checkpoint with original topology
        validator = CheckpointCompatibilityValidator()
        original_topology_hash = validator.compute_full_topology_hash(original_graph)
        original_config_hash = stable_hash({"prompt": "test"})

        checkpoint = Checkpoint(
            checkpoint_id="ckpt-001",
            run_id="run-001",
            token_id="tok-001",
            node_id="checkpoint_node",
            sequence_number=10,
            created_at=datetime.now(UTC),
            upstream_topology_hash=original_topology_hash,
            checkpoint_node_config_hash=original_config_hash,
        )

        # Modified graph: change path_A config
        modified_graph = ExecutionGraph()
        modified_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        modified_graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate")
        modified_graph.add_node("path_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"version": "v2"})  # CHANGED
        modified_graph.add_node("path_B", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"version": "v1"})
        modified_graph.add_node("coalesce", node_type=NodeType.COALESCE, plugin_name="coalesce")
        modified_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})

        modified_graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("gate", "path_A", label="branch_a", mode=RoutingMode.COPY)
        modified_graph.add_edge("gate", "path_B", label="branch_b", mode=RoutingMode.COPY)
        modified_graph.add_edge("path_A", "coalesce", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("path_B", "coalesce", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("coalesce", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)

        # Should FAIL - topology changed (path_A config modified)
        result = validator.validate(checkpoint, modified_graph)

        assert not result.can_resume
        assert result.reason is not None
        assert "configuration changed" in result.reason.lower() or "topology" in result.reason.lower()

    def test_resume_rejects_downstream_fork_path_addition(self) -> None:
        """BUG-COMPAT-01: Downstream changes MUST be rejected for audit integrity.

        Even though the change is downstream of the checkpoint node, allowing it
        would mean unprocessed rows could flow through a path that didn't exist
        when the run started. This creates mixed-configuration runs.

        Rule: one run_id = one complete pipeline configuration
        """
        # Original: source → checkpoint → gate → path_A → sink_A
        original_graph = ExecutionGraph()
        original_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        original_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})
        original_graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate")
        original_graph.add_node("path_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        original_graph.add_node("sink_A", node_type=NodeType.SINK, plugin_name="csv")

        original_graph.add_edge("source", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("checkpoint_node", "gate", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("gate", "path_A", label="branch_a", mode=RoutingMode.COPY)
        original_graph.add_edge("path_A", "sink_A", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        original_topology_hash = validator.compute_full_topology_hash(original_graph)
        original_config_hash = stable_hash({"prompt": "test"})

        checkpoint = Checkpoint(
            checkpoint_id="ckpt-002",
            run_id="run-002",
            token_id="tok-002",
            node_id="checkpoint_node",
            sequence_number=5,
            created_at=datetime.now(UTC),
            upstream_topology_hash=original_topology_hash,
            checkpoint_node_config_hash=original_config_hash,
        )

        # Modified: Add new fork path after checkpoint (DOWNSTREAM change)
        modified_graph = ExecutionGraph()
        modified_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        modified_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})
        modified_graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate")
        modified_graph.add_node("path_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        modified_graph.add_node("path_C", node_type=NodeType.TRANSFORM, plugin_name="new_transform")  # NEW
        modified_graph.add_node("sink_A", node_type=NodeType.SINK, plugin_name="csv")
        modified_graph.add_node("sink_C", node_type=NodeType.SINK, plugin_name="csv")  # NEW

        modified_graph.add_edge("source", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("checkpoint_node", "gate", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("gate", "path_A", label="branch_a", mode=RoutingMode.COPY)
        modified_graph.add_edge("gate", "path_C", label="branch_c", mode=RoutingMode.COPY)  # NEW
        modified_graph.add_edge("path_A", "sink_A", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("path_C", "sink_C", label="continue", mode=RoutingMode.MOVE)  # NEW

        # BUG-COMPAT-01: Should FAIL - ANY topology change invalidates resume
        result = validator.validate(checkpoint, modified_graph)

        assert not result.can_resume
        assert result.reason is not None
        assert "configuration changed" in result.reason.lower() or "topology" in result.reason.lower()


class TestTransitiveUpstreamChanges:
    """Tests for transitive upstream changes - CRITICAL GAP identified by QA review."""

    def test_resume_rejects_grandparent_node_change(self) -> None:
        """Changing node 2+ hops upstream should reject."""
        # Original: source → A → B → checkpoint
        original_graph = ExecutionGraph()
        original_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", config={"path": "input.csv"})
        original_graph.add_node("transform_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        original_graph.add_node("transform_B", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        original_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})

        original_graph.add_edge("source", "transform_A", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("transform_A", "transform_B", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("transform_B", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        original_topology_hash = validator.compute_full_topology_hash(original_graph)
        original_config_hash = stable_hash({"prompt": "test"})

        checkpoint = Checkpoint(
            checkpoint_id="ckpt-003",
            run_id="run-003",
            token_id="tok-003",
            node_id="checkpoint_node",
            sequence_number=15,
            created_at=datetime.now(UTC),
            upstream_topology_hash=original_topology_hash,
            checkpoint_node_config_hash=original_config_hash,
        )

        # Modified: change source config (grandparent of checkpoint)
        modified_graph = ExecutionGraph()
        modified_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", config={"path": "input_v2.csv"})  # CHANGED
        modified_graph.add_node("transform_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        modified_graph.add_node("transform_B", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        modified_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})

        modified_graph.add_edge("source", "transform_A", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("transform_A", "transform_B", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("transform_B", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)

        # Should FAIL - topology changed (source config modified)
        result = validator.validate(checkpoint, modified_graph)

        assert not result.can_resume
        assert result.reason is not None
        assert "configuration changed" in result.reason.lower() or "topology" in result.reason.lower()

    def test_resume_rejects_deep_upstream_insertion(self) -> None:
        """Inserting transform far upstream should reject."""
        # Original: source → A → B → C → checkpoint
        original_graph = ExecutionGraph()
        original_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        original_graph.add_node("transform_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        original_graph.add_node("transform_B", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        original_graph.add_node("transform_C", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        original_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})

        original_graph.add_edge("source", "transform_A", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("transform_A", "transform_B", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("transform_B", "transform_C", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("transform_C", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        original_topology_hash = validator.compute_full_topology_hash(original_graph)
        original_config_hash = stable_hash({"prompt": "test"})

        checkpoint = Checkpoint(
            checkpoint_id="ckpt-004",
            run_id="run-004",
            token_id="tok-004",
            node_id="checkpoint_node",
            sequence_number=20,
            created_at=datetime.now(UTC),
            upstream_topology_hash=original_topology_hash,
            checkpoint_node_config_hash=original_config_hash,
        )

        # Modified: insert NEW transform far upstream (between source and A)
        modified_graph = ExecutionGraph()
        modified_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        modified_graph.add_node("transform_NEW", node_type=NodeType.TRANSFORM, plugin_name="field_mapper")  # NEW
        modified_graph.add_node("transform_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        modified_graph.add_node("transform_B", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        modified_graph.add_node("transform_C", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        modified_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})

        modified_graph.add_edge("source", "transform_NEW", label="continue", mode=RoutingMode.MOVE)  # NEW
        modified_graph.add_edge("transform_NEW", "transform_A", label="continue", mode=RoutingMode.MOVE)  # NEW
        modified_graph.add_edge("transform_A", "transform_B", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("transform_B", "transform_C", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("transform_C", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)

        # Should FAIL - topology changed (new transform inserted)
        result = validator.validate(checkpoint, modified_graph)

        assert not result.can_resume
        assert result.reason is not None
        assert "configuration changed" in result.reason.lower() or "topology" in result.reason.lower()


class TestParallelEdgeValidation:
    """Tests for MultiDiGraph parallel edge validation - CRITICAL GAP identified by QA review."""

    def test_topology_hash_includes_edge_keys(self) -> None:
        """Parallel edges must have different keys in hash."""
        # Gate with fork creates parallel edges to same transform
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate")
        graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        graph.add_node("checkpoint", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})

        # Two parallel edges from gate to transform (different route labels)
        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "transform", label="route_A", mode=RoutingMode.COPY)
        graph.add_edge("gate", "transform", label="route_B", mode=RoutingMode.COPY)
        graph.add_edge("transform", "checkpoint", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        topology_hash_with_both_edges = validator.compute_full_topology_hash(graph)

        # Create second graph with only one edge
        graph2 = ExecutionGraph()
        graph2.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph2.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate")
        graph2.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        graph2.add_node("checkpoint", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})

        graph2.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph2.add_edge("gate", "transform", label="route_A", mode=RoutingMode.COPY)  # Only route_A
        graph2.add_edge("transform", "checkpoint", label="continue", mode=RoutingMode.MOVE)

        topology_hash_with_one_edge = validator.compute_full_topology_hash(graph2)

        # Hashes MUST be different - parallel edges matter
        assert topology_hash_with_both_edges != topology_hash_with_one_edge

    def test_resume_rejects_parallel_edge_addition(self) -> None:
        """Adding second edge between same nodes should reject."""
        # Original: source → gate --[approved]--> transform → checkpoint
        original_graph = ExecutionGraph()
        original_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        original_graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate")
        original_graph.add_node("upstream_transform", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        original_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})

        original_graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("gate", "upstream_transform", label="approved", mode=RoutingMode.MOVE)
        original_graph.add_edge("upstream_transform", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        original_topology_hash = validator.compute_full_topology_hash(original_graph)
        original_config_hash = stable_hash({"prompt": "test"})

        checkpoint = Checkpoint(
            checkpoint_id="ckpt-005",
            run_id="run-005",
            token_id="tok-005",
            node_id="checkpoint_node",
            sequence_number=25,
            created_at=datetime.now(UTC),
            upstream_topology_hash=original_topology_hash,
            checkpoint_node_config_hash=original_config_hash,
        )

        # Modified: Add second parallel edge between gate and upstream_transform
        modified_graph = ExecutionGraph()
        modified_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        modified_graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate")
        modified_graph.add_node("upstream_transform", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        modified_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})

        modified_graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("gate", "upstream_transform", label="approved", mode=RoutingMode.MOVE)
        modified_graph.add_edge("gate", "upstream_transform", label="verified", mode=RoutingMode.COPY)  # NEW parallel edge
        modified_graph.add_edge("upstream_transform", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)

        # Should FAIL - topology changed (new parallel edge added)
        result = validator.validate(checkpoint, modified_graph)

        assert not result.can_resume
        assert result.reason is not None
        assert "configuration changed" in result.reason.lower() or "topology" in result.reason.lower()


class TestCheckpointNodeValidation:
    """Tests for checkpoint node existence and config validation.

    These tests cover the basic rejection paths in the validator:
    1. Checkpoint node removed from graph
    2. Checkpoint node config changed

    These were identified as missing by quality audit - P1 priority.
    """

    def test_resume_rejects_missing_checkpoint_node(self) -> None:
        """Resume must reject when checkpoint node is removed from graph."""
        # Original: source → transform → checkpoint_node
        original_graph = ExecutionGraph()
        original_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        original_graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        original_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "test"})

        original_graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("transform", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        original_topology_hash = validator.compute_full_topology_hash(original_graph)
        original_config_hash = stable_hash({"prompt": "test"})

        checkpoint = Checkpoint(
            checkpoint_id="ckpt-missing-001",
            run_id="run-missing-001",
            token_id="tok-001",
            node_id="checkpoint_node",
            sequence_number=5,
            created_at=datetime.now(UTC),
            upstream_topology_hash=original_topology_hash,
            checkpoint_node_config_hash=original_config_hash,
        )

        # Modified: checkpoint_node is REMOVED
        modified_graph = ExecutionGraph()
        modified_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        modified_graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="passthrough")
        # NOTE: checkpoint_node is NOT added

        modified_graph.add_edge("source", "transform", label="continue", mode=RoutingMode.MOVE)

        # Should FAIL - checkpoint node no longer exists
        result = validator.validate(checkpoint, modified_graph)

        assert not result.can_resume
        assert result.reason is not None
        assert "checkpoint_node" in result.reason or "not found" in result.reason.lower() or "missing" in result.reason.lower()

    def test_resume_rejects_checkpoint_config_change(self) -> None:
        """Resume must reject when checkpoint node config changes."""
        # Original: source → checkpoint_node with config {"prompt": "original"}
        original_graph = ExecutionGraph()
        original_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        original_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "original"})

        original_graph.add_edge("source", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        original_topology_hash = validator.compute_full_topology_hash(original_graph)
        original_config_hash = stable_hash({"prompt": "original"})

        checkpoint = Checkpoint(
            checkpoint_id="ckpt-config-001",
            run_id="run-config-001",
            token_id="tok-001",
            node_id="checkpoint_node",
            sequence_number=10,
            created_at=datetime.now(UTC),
            upstream_topology_hash=original_topology_hash,
            checkpoint_node_config_hash=original_config_hash,
        )

        # Modified: SAME topology, DIFFERENT checkpoint node config
        modified_graph = ExecutionGraph()
        modified_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        modified_graph.add_node("checkpoint_node", node_type=NodeType.TRANSFORM, plugin_name="llm", config={"prompt": "changed"})  # CHANGED

        modified_graph.add_edge("source", "checkpoint_node", label="continue", mode=RoutingMode.MOVE)

        # Should FAIL - checkpoint node config changed
        result = validator.validate(checkpoint, modified_graph)

        assert not result.can_resume
        assert result.reason is not None
        assert "configuration" in result.reason.lower() or "config" in result.reason.lower()


# Bug #7 fix: Legacy checkpoint test class removed
# With nullable=False on topology hash fields, legacy checkpoints cannot exist in the system.
# This enforces the "No Legacy Code Policy" - old checkpoints are incompatible by design.


class TestMultiSinkBranchValidation:
    """BUG-COMPAT-01: Tests for multi-sink DAG parallel branch validation.

    These tests verify that changes to ANY sink branch are detected,
    not just the branch containing the checkpoint node.

    The core bug: upstream-only validation allowed changes to sibling
    branches (other sink paths) to go undetected, causing mixed-config runs.
    """

    def test_resume_rejects_parallel_sink_branch_change(self) -> None:
        """CORE BUG-COMPAT-01 TEST: Changing sibling sink branch must reject.

        Topology: Source → Gate → Transform_A → Sink_A (checkpoint here)
                              → Transform_B → Sink_B (modified after checkpoint)

        OLD BEHAVIOR (buggy): Resume allowed because only Sink_A ancestors checked
        NEW BEHAVIOR (fixed): Resume rejected because full DAG is validated
        """
        # Original graph with two sink branches
        original_graph = ExecutionGraph()
        original_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        original_graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate", config={"threshold": 50})
        original_graph.add_node("transform_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"version": "v1"})
        original_graph.add_node("transform_B", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"version": "v1"})
        original_graph.add_node("sink_A", node_type=NodeType.SINK, plugin_name="csv", config={"path": "output_a.csv"})
        original_graph.add_node("sink_B", node_type=NodeType.SINK, plugin_name="csv", config={"path": "output_b.csv"})

        original_graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("gate", "transform_A", label="high", mode=RoutingMode.MOVE)
        original_graph.add_edge("gate", "transform_B", label="low", mode=RoutingMode.MOVE)
        original_graph.add_edge("transform_A", "sink_A", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("transform_B", "sink_B", label="continue", mode=RoutingMode.MOVE)

        # Create checkpoint at sink_A (the "high" branch)
        validator = CheckpointCompatibilityValidator()
        original_topology_hash = validator.compute_full_topology_hash(original_graph)
        original_config_hash = stable_hash({"path": "output_a.csv"})

        checkpoint = Checkpoint(
            checkpoint_id="ckpt-multi-001",
            run_id="run-multi-001",
            token_id="tok-001",
            node_id="sink_A",  # Checkpoint at sink_A
            sequence_number=100,
            created_at=datetime.now(UTC),
            upstream_topology_hash=original_topology_hash,
            checkpoint_node_config_hash=original_config_hash,
        )

        # Modified: Change sink_B branch (sibling of checkpoint branch)
        modified_graph = ExecutionGraph()
        modified_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        modified_graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate", config={"threshold": 50})
        modified_graph.add_node("transform_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"version": "v1"})
        modified_graph.add_node("transform_B", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"version": "v2"})  # CHANGED
        modified_graph.add_node("sink_A", node_type=NodeType.SINK, plugin_name="csv", config={"path": "output_a.csv"})
        modified_graph.add_node("sink_B", node_type=NodeType.SINK, plugin_name="csv", config={"path": "output_b.csv"})

        modified_graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("gate", "transform_A", label="high", mode=RoutingMode.MOVE)
        modified_graph.add_edge("gate", "transform_B", label="low", mode=RoutingMode.MOVE)
        modified_graph.add_edge("transform_A", "sink_A", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("transform_B", "sink_B", label="continue", mode=RoutingMode.MOVE)

        # BUG-COMPAT-01: Must REJECT - sibling branch changed
        result = validator.validate(checkpoint, modified_graph)

        assert not result.can_resume, "Resume should be rejected when sibling sink branch changes"
        assert result.reason is not None
        assert "configuration changed" in result.reason.lower() or "topology" in result.reason.lower()

    def test_resume_rejects_new_sink_added(self) -> None:
        """Adding a new sink after checkpoint must reject.

        Rule: Adding a new sink creates a path that didn't exist at run start.
        Rows processed after resume would flow through a different configuration.
        """
        # Original: Source → Gate → Sink_A
        original_graph = ExecutionGraph()
        original_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        original_graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate")
        original_graph.add_node("sink_A", node_type=NodeType.SINK, plugin_name="csv", config={"path": "output.csv"})

        original_graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("gate", "sink_A", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        original_topology_hash = validator.compute_full_topology_hash(original_graph)
        original_config_hash = stable_hash({"path": "output.csv"})

        checkpoint = Checkpoint(
            checkpoint_id="ckpt-newsink-001",
            run_id="run-newsink-001",
            token_id="tok-001",
            node_id="sink_A",
            sequence_number=50,
            created_at=datetime.now(UTC),
            upstream_topology_hash=original_topology_hash,
            checkpoint_node_config_hash=original_config_hash,
        )

        # Modified: Add sink_B as new routing destination
        modified_graph = ExecutionGraph()
        modified_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        modified_graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate")
        modified_graph.add_node("sink_A", node_type=NodeType.SINK, plugin_name="csv", config={"path": "output.csv"})
        modified_graph.add_node("sink_B", node_type=NodeType.SINK, plugin_name="csv", config={"path": "errors.csv"})  # NEW

        modified_graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("gate", "sink_A", label="pass", mode=RoutingMode.MOVE)
        modified_graph.add_edge("gate", "sink_B", label="fail", mode=RoutingMode.MOVE)  # NEW

        # Must REJECT - new sink path added
        result = validator.validate(checkpoint, modified_graph)

        assert not result.can_resume, "Resume should be rejected when new sink is added"

    def test_resume_allows_same_config_multi_sink(self) -> None:
        """Resume ALLOWED when multi-sink config is unchanged.

        Verifies that multi-sink DAGs work correctly when no changes made.
        """
        # Multi-sink graph
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="threshold_gate", config={"threshold": 50})
        graph.add_node("sink_high", node_type=NodeType.SINK, plugin_name="csv", config={"path": "high.csv"})
        graph.add_node("sink_low", node_type=NodeType.SINK, plugin_name="csv", config={"path": "low.csv"})

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "sink_high", label="high", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "sink_low", label="low", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        topology_hash = validator.compute_full_topology_hash(graph)
        config_hash = stable_hash({"path": "high.csv"})

        checkpoint = Checkpoint(
            checkpoint_id="ckpt-same-001",
            run_id="run-same-001",
            token_id="tok-001",
            node_id="sink_high",
            sequence_number=200,
            created_at=datetime.now(UTC),
            upstream_topology_hash=topology_hash,
            checkpoint_node_config_hash=config_hash,
        )

        # Validate with SAME graph (no changes)
        result = validator.validate(checkpoint, graph)

        assert result.can_resume, "Resume should be allowed when config is unchanged"

    def test_three_way_fork_checkpoint_rejects_any_branch_change(self) -> None:
        """Three-way fork: changing ANY branch must reject.

        Tests the N-way fork case (N>2) which has different combinatorics.
        """
        # Original: Source → Gate → Branch_A → Sink_A (checkpointed)
        #                        → Branch_B → Sink_B
        #                        → Branch_C → Sink_C
        original_graph = ExecutionGraph()
        original_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        original_graph.add_node("gate", node_type=NodeType.GATE, plugin_name="classifier_gate")
        original_graph.add_node("branch_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"label": "A"})
        original_graph.add_node("branch_B", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"label": "B"})
        original_graph.add_node("branch_C", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"label": "C"})
        original_graph.add_node("sink_A", node_type=NodeType.SINK, plugin_name="csv", config={"path": "a.csv"})
        original_graph.add_node("sink_B", node_type=NodeType.SINK, plugin_name="csv", config={"path": "b.csv"})
        original_graph.add_node("sink_C", node_type=NodeType.SINK, plugin_name="csv", config={"path": "c.csv"})

        original_graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("gate", "branch_A", label="type_a", mode=RoutingMode.MOVE)
        original_graph.add_edge("gate", "branch_B", label="type_b", mode=RoutingMode.MOVE)
        original_graph.add_edge("gate", "branch_C", label="type_c", mode=RoutingMode.MOVE)
        original_graph.add_edge("branch_A", "sink_A", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("branch_B", "sink_B", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("branch_C", "sink_C", label="continue", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        original_hash = validator.compute_full_topology_hash(original_graph)
        config_hash = stable_hash({"path": "a.csv"})

        checkpoint = Checkpoint(
            checkpoint_id="ckpt-3way-001",
            run_id="run-3way-001",
            token_id="tok-001",
            node_id="sink_A",  # Checkpoint at sink_A
            sequence_number=300,
            created_at=datetime.now(UTC),
            upstream_topology_hash=original_hash,
            checkpoint_node_config_hash=config_hash,
        )

        # Modified: Change branch_C (distant sibling of checkpoint branch)
        modified_graph = ExecutionGraph()
        modified_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        modified_graph.add_node("gate", node_type=NodeType.GATE, plugin_name="classifier_gate")
        modified_graph.add_node("branch_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"label": "A"})
        modified_graph.add_node("branch_B", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"label": "B"})
        modified_graph.add_node(
            "branch_C", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"label": "C_MODIFIED"}
        )  # CHANGED
        modified_graph.add_node("sink_A", node_type=NodeType.SINK, plugin_name="csv", config={"path": "a.csv"})
        modified_graph.add_node("sink_B", node_type=NodeType.SINK, plugin_name="csv", config={"path": "b.csv"})
        modified_graph.add_node("sink_C", node_type=NodeType.SINK, plugin_name="csv", config={"path": "c.csv"})

        modified_graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("gate", "branch_A", label="type_a", mode=RoutingMode.MOVE)
        modified_graph.add_edge("gate", "branch_B", label="type_b", mode=RoutingMode.MOVE)
        modified_graph.add_edge("gate", "branch_C", label="type_c", mode=RoutingMode.MOVE)
        modified_graph.add_edge("branch_A", "sink_A", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("branch_B", "sink_B", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("branch_C", "sink_C", label="continue", mode=RoutingMode.MOVE)

        # Must REJECT - branch_C changed (even though checkpoint is at sink_A)
        result = validator.validate(checkpoint, modified_graph)

        assert not result.can_resume, "Resume should be rejected when ANY branch changes in N-way fork"

    def test_diamond_dag_multi_sink_rejects_any_path_change(self) -> None:
        """Diamond DAG with multi-sink terminals: any change rejects.

        Topology: Source → Fork → Path_A → Coalesce → Gate → Sink_X (checkpoint)
                              → Path_B ─────────┘        → Sink_Y (modified)

        Tests complex fork/coalesce + multi-sink pattern.
        """
        # Original diamond DAG
        original_graph = ExecutionGraph()
        original_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        original_graph.add_node("fork_gate", node_type=NodeType.GATE, plugin_name="fork_gate")
        original_graph.add_node("path_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"path": "A"})
        original_graph.add_node("path_B", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"path": "B"})
        original_graph.add_node("coalesce", node_type=NodeType.COALESCE, plugin_name="coalesce")
        original_graph.add_node("final_gate", node_type=NodeType.GATE, plugin_name="threshold_gate")
        original_graph.add_node("sink_X", node_type=NodeType.SINK, plugin_name="csv", config={"path": "x.csv"})
        original_graph.add_node("sink_Y", node_type=NodeType.SINK, plugin_name="csv", config={"path": "y.csv"})

        original_graph.add_edge("source", "fork_gate", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("fork_gate", "path_A", label="path_a", mode=RoutingMode.COPY)
        original_graph.add_edge("fork_gate", "path_B", label="path_b", mode=RoutingMode.COPY)
        original_graph.add_edge("path_A", "coalesce", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("path_B", "coalesce", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("coalesce", "final_gate", label="continue", mode=RoutingMode.MOVE)
        original_graph.add_edge("final_gate", "sink_X", label="pass", mode=RoutingMode.MOVE)
        original_graph.add_edge("final_gate", "sink_Y", label="fail", mode=RoutingMode.MOVE)

        validator = CheckpointCompatibilityValidator()
        original_hash = validator.compute_full_topology_hash(original_graph)
        config_hash = stable_hash({"path": "x.csv"})

        checkpoint = Checkpoint(
            checkpoint_id="ckpt-diamond-001",
            run_id="run-diamond-001",
            token_id="tok-001",
            node_id="sink_X",  # Checkpoint at sink_X
            sequence_number=500,
            created_at=datetime.now(UTC),
            upstream_topology_hash=original_hash,
            checkpoint_node_config_hash=config_hash,
        )

        # Modified: Change sink_Y config (sibling of checkpoint in final routing)
        modified_graph = ExecutionGraph()
        modified_graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        modified_graph.add_node("fork_gate", node_type=NodeType.GATE, plugin_name="fork_gate")
        modified_graph.add_node("path_A", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"path": "A"})
        modified_graph.add_node("path_B", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config={"path": "B"})
        modified_graph.add_node("coalesce", node_type=NodeType.COALESCE, plugin_name="coalesce")
        modified_graph.add_node("final_gate", node_type=NodeType.GATE, plugin_name="threshold_gate")
        modified_graph.add_node("sink_X", node_type=NodeType.SINK, plugin_name="csv", config={"path": "x.csv"})
        modified_graph.add_node("sink_Y", node_type=NodeType.SINK, plugin_name="csv", config={"path": "y_modified.csv"})  # CHANGED

        modified_graph.add_edge("source", "fork_gate", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("fork_gate", "path_A", label="path_a", mode=RoutingMode.COPY)
        modified_graph.add_edge("fork_gate", "path_B", label="path_b", mode=RoutingMode.COPY)
        modified_graph.add_edge("path_A", "coalesce", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("path_B", "coalesce", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("coalesce", "final_gate", label="continue", mode=RoutingMode.MOVE)
        modified_graph.add_edge("final_gate", "sink_X", label="pass", mode=RoutingMode.MOVE)
        modified_graph.add_edge("final_gate", "sink_Y", label="fail", mode=RoutingMode.MOVE)

        # Must REJECT - sink_Y config changed (sibling of checkpoint sink)
        result = validator.validate(checkpoint, modified_graph)

        assert not result.can_resume, "Resume should be rejected when sibling sink config changes in diamond DAG"
