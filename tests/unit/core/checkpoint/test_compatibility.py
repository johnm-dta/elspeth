"""Unit tests for CheckpointCompatibilityValidator."""

from __future__ import annotations

from datetime import UTC, datetime

from elspeth.contracts import Checkpoint, NodeType
from elspeth.core.canonical import compute_full_topology_hash
from elspeth.core.checkpoint.compatibility import CheckpointCompatibilityValidator
from elspeth.core.dag import ExecutionGraph


def _graph(*, checkpoint_config: dict[str, object] | None = None, include_checkpoint: bool = True) -> ExecutionGraph:
    graph = ExecutionGraph()
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", config={})
    if include_checkpoint:
        graph.add_node(
            "checkpoint-node",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform",
            config=checkpoint_config or {"version": 1},
        )
        graph.add_edge("source", "checkpoint-node", label="continue")
    return graph


def _checkpoint_for_graph(graph: ExecutionGraph) -> Checkpoint:
    return Checkpoint(
        checkpoint_id="cp-compat-001",
        run_id="run-compat-001",
        sequence_number=7,
        created_at=datetime.now(UTC),
        upstream_topology_hash=compute_full_topology_hash(graph),
        format_version=Checkpoint.CURRENT_FORMAT_VERSION,
    )


def test_validate_rejects_node_removal_via_topology_hash() -> None:
    """Removing a node still invalidates the checkpoint.

    The full-topology hash embeds every node, so node removal is rejected
    without a per-node anchor (the former anchor-node existence check was
    strictly subsumed by the hash comparison).
    """
    original_graph = _graph(checkpoint_config={"version": 1})
    checkpoint = _checkpoint_for_graph(original_graph)

    current_graph = _graph(include_checkpoint=False)
    validator = CheckpointCompatibilityValidator()
    result = validator.validate(checkpoint, current_graph)

    assert result.can_resume is False
    assert result.reason is not None
    assert "Pipeline configuration changed since checkpoint was created." in result.reason


def test_validate_rejects_node_config_change_via_topology_hash() -> None:
    """Changing any node's config still invalidates the checkpoint.

    The full-topology hash embeds every node's config hash, so config drift
    is rejected without a per-node anchor (the former anchor-node config
    check was strictly subsumed by the hash comparison).
    """
    original_graph = _graph(checkpoint_config={"version": 1})
    checkpoint = _checkpoint_for_graph(original_graph)

    changed_graph = _graph(checkpoint_config={"version": 2})
    validator = CheckpointCompatibilityValidator()
    result = validator.validate(checkpoint, changed_graph)

    assert result.can_resume is False
    assert result.reason is not None
    assert "Pipeline configuration changed since checkpoint was created." in result.reason


def test_validate_rejects_topology_hash_mismatch() -> None:
    original_graph = _graph(checkpoint_config={"version": 1})
    checkpoint = _checkpoint_for_graph(original_graph)

    changed_graph = _graph(checkpoint_config={"version": 1})
    changed_graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv_sink", config={})
    changed_graph.add_edge("checkpoint-node", "sink", label="continue")

    validator = CheckpointCompatibilityValidator()
    result = validator.validate(checkpoint, changed_graph)

    assert result.can_resume is False
    assert result.reason is not None
    assert "Pipeline configuration changed since checkpoint was created." in result.reason
    assert "Expected topology hash" in result.reason


def test_validate_accepts_unchanged_graph() -> None:
    graph = _graph(checkpoint_config={"version": 1})
    checkpoint = _checkpoint_for_graph(graph)

    validator = CheckpointCompatibilityValidator()
    result = validator.validate(checkpoint, graph)
    assert result.can_resume is True
    assert result.reason is None
