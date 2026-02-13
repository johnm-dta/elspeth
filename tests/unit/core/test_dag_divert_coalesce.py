# tests/unit/core/test_dag_divert_coalesce.py
"""Tests for DIVERT+coalesce build-time warning detection.

When a branch transform has on_error routing (DIVERT edge) and feeds a
require_all coalesce, rows diverted on error will never reach the coalesce.
The coalesce will wait indefinitely for the missing branch. This test suite
verifies that the warning system detects these interactions correctly.
"""

from __future__ import annotations

import pytest

from elspeth.contracts import RoutingMode
from elspeth.contracts.enums import NodeType
from elspeth.contracts.types import NodeID
from elspeth.core.config import CoalesceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.models import GraphValidationWarning


def _make_coalesce_config(
    name: str = "merge",
    branches: dict[str, str] | None = None,
    policy: str = "require_all",
    **kwargs: object,
) -> CoalesceSettings:
    """Build a CoalesceSettings for testing."""
    if branches is None:
        branches = {"path_a": "path_a", "path_b": "path_b"}
    # best_effort requires timeout_seconds; supply a default if not given
    if policy == "best_effort" and "timeout_seconds" not in kwargs:
        kwargs["timeout_seconds"] = 30.0
    return CoalesceSettings(
        name=name,
        branches=branches,
        policy=policy,
        merge="union",
        **kwargs,  # type: ignore[arg-type]
    )


def _build_graph_with_divert(
    *,
    divert_on: str | None = None,
    policy: str = "require_all",
    multi_step: bool = False,
) -> tuple[ExecutionGraph, dict[NodeID, CoalesceSettings]]:
    """Build a fork/coalesce graph with optional DIVERT edge.

    Creates: source → gate → [path_a transforms, path_b transforms] → coalesce → sink

    Args:
        divert_on: Transform node to add a DIVERT edge to (or None).
        policy: Coalesce policy.
        multi_step: If True, add two transforms per branch instead of one.

    Returns:
        (graph, coalesce_configs) tuple for passing to warn_divert_coalesce_interactions.
    """
    graph = ExecutionGraph()

    # Source
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test-source", config={})

    # Gate
    graph.add_node("gate", node_type=NodeType.GATE, plugin_name="test-gate", config={})
    graph.add_edge("source", "gate", label="continue")

    # Error sink (for DIVERT edges)
    graph.add_node("error_sink", node_type=NodeType.SINK, plugin_name="error-sink", config={})

    if multi_step:
        # Path A: t_a1 → t_a2
        graph.add_node("t_a1", node_type=NodeType.TRANSFORM, plugin_name="transform-a1", config={})
        graph.add_node("t_a2", node_type=NodeType.TRANSFORM, plugin_name="transform-a2", config={})
        graph.add_edge("gate", "t_a1", label="path_a", mode=RoutingMode.MOVE)
        graph.add_edge("t_a1", "t_a2", label="continue", mode=RoutingMode.MOVE)
        last_a = "t_a2"

        # Path B: t_b1 → t_b2
        graph.add_node("t_b1", node_type=NodeType.TRANSFORM, plugin_name="transform-b1", config={})
        graph.add_node("t_b2", node_type=NodeType.TRANSFORM, plugin_name="transform-b2", config={})
        graph.add_edge("gate", "t_b1", label="path_b", mode=RoutingMode.MOVE)
        graph.add_edge("t_b1", "t_b2", label="continue", mode=RoutingMode.MOVE)
        last_b = "t_b2"
    else:
        # Path A: single transform
        graph.add_node("t_a", node_type=NodeType.TRANSFORM, plugin_name="transform-a", config={})
        graph.add_edge("gate", "t_a", label="path_a", mode=RoutingMode.MOVE)
        last_a = "t_a"

        # Path B: single transform
        graph.add_node("t_b", node_type=NodeType.TRANSFORM, plugin_name="transform-b", config={})
        graph.add_edge("gate", "t_b", label="path_b", mode=RoutingMode.MOVE)
        last_b = "t_b"

    # Coalesce
    branches = {"path_a": "path_a", "path_b": "path_b"}
    coalesce_config = {
        "branches": branches,
        "policy": policy,
        "merge": "union",
    }
    graph.add_node("coalesce", node_type=NodeType.COALESCE, plugin_name="coalesce:merge", config=coalesce_config)
    graph.add_edge(last_a, "coalesce", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge(last_b, "coalesce", label="continue", mode=RoutingMode.MOVE)

    # Sink
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test-sink", config={})
    graph.add_edge("coalesce", "sink", label="continue")

    # Add DIVERT edge if requested
    if divert_on is not None:
        graph.add_edge(divert_on, "error_sink", label=f"__error_{divert_on}__", mode=RoutingMode.DIVERT)

    coal_settings = _make_coalesce_config(policy=policy, branches=branches)
    coalesce_configs = {NodeID("coalesce"): coal_settings}

    return graph, coalesce_configs


class TestDivertCoalesceWarning:
    """Tests for warn_divert_coalesce_interactions()."""

    def test_divert_on_branch_transform_with_require_all_emits_warning(self) -> None:
        """Transform with DIVERT + require_all coalesce → warning emitted."""
        graph, configs = _build_graph_with_divert(divert_on="t_a")
        warnings = graph.warn_divert_coalesce_interactions(configs)

        assert len(warnings) == 1
        w = warnings[0]
        assert w.code == "DIVERT_COALESCE_REQUIRE_ALL"
        assert "t_a" in w.message
        assert "coalesce" in w.message
        assert "t_a" in w.node_ids
        assert "coalesce" in w.node_ids

    def test_divert_on_branch_transform_with_best_effort_no_warning(self) -> None:
        """Transform with DIVERT + best_effort coalesce → no warning."""
        graph, configs = _build_graph_with_divert(
            divert_on="t_a",
            policy="best_effort",
        )
        warnings = graph.warn_divert_coalesce_interactions(configs)
        assert len(warnings) == 0

    def test_no_divert_edges_no_warning(self) -> None:
        """No DIVERT edges → no warnings (even with require_all)."""
        graph, configs = _build_graph_with_divert(divert_on=None)
        warnings = graph.warn_divert_coalesce_interactions(configs)
        assert len(warnings) == 0

    def test_identity_branch_with_divert_elsewhere_no_false_positive(self) -> None:
        """Identity branches (COPY edges, no transforms) should not trigger warnings."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="src", config={})
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="gate", config={})
        graph.add_edge("source", "gate", label="continue")

        # Identity branches (COPY edges from gate to coalesce)
        coalesce_config = {"branches": {"path_a": "path_a", "path_b": "path_b"}, "policy": "require_all", "merge": "union"}
        graph.add_node("coalesce", node_type=NodeType.COALESCE, plugin_name="coalesce:merge", config=coalesce_config)
        graph.add_edge("gate", "coalesce", label="path_a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "coalesce", label="path_b", mode=RoutingMode.COPY)

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="sink", config={})
        graph.add_edge("coalesce", "sink", label="continue")

        # Even though a transform somewhere has a DIVERT edge, the identity
        # branches should not trigger warnings
        graph.add_node("other_t", node_type=NodeType.TRANSFORM, plugin_name="other", config={})
        graph.add_node("err_sink", node_type=NodeType.SINK, plugin_name="err", config={})
        graph.add_edge("other_t", "err_sink", label="__error__", mode=RoutingMode.DIVERT)

        coal_settings = _make_coalesce_config()
        configs = {NodeID("coalesce"): coal_settings}
        warnings = graph.warn_divert_coalesce_interactions(configs)
        assert len(warnings) == 0

    def test_multi_step_branch_divert_on_first_transform(self) -> None:
        """Multi-step branch: divert on first transform → warning emitted."""
        graph, configs = _build_graph_with_divert(divert_on="t_a1", multi_step=True)
        warnings = graph.warn_divert_coalesce_interactions(configs)

        assert len(warnings) == 1
        assert "t_a1" in warnings[0].message

    def test_multi_step_branch_divert_on_second_transform(self) -> None:
        """Multi-step branch: divert on second transform → warning emitted."""
        graph, configs = _build_graph_with_divert(divert_on="t_a2", multi_step=True)
        warnings = graph.warn_divert_coalesce_interactions(configs)

        assert len(warnings) == 1
        assert "t_a2" in warnings[0].message

    def test_multiple_branches_only_one_has_divert(self) -> None:
        """Only the branch with DIVERT should produce a warning."""
        graph, configs = _build_graph_with_divert(divert_on="t_a")
        warnings = graph.warn_divert_coalesce_interactions(configs)

        assert len(warnings) == 1
        w = warnings[0]
        # Should reference the specific transform, not the other branch's
        assert "t_a" in w.node_ids
        assert "t_b" not in w.node_ids

    def test_both_branches_have_divert_two_warnings(self) -> None:
        """Both branches with DIVERT → two warnings."""
        graph, configs = _build_graph_with_divert(divert_on="t_a")
        # Add DIVERT edge to t_b as well
        graph.add_edge("t_b", "error_sink", label="__error_t_b__", mode=RoutingMode.DIVERT)

        warnings = graph.warn_divert_coalesce_interactions(configs)
        assert len(warnings) == 2
        warning_transforms = {w.node_ids[0] for w in warnings}
        assert "t_a" in warning_transforms
        assert "t_b" in warning_transforms

    def test_quorum_policy_no_warning(self) -> None:
        """DIVERT + quorum coalesce → no warning (quorum tolerates missing branches)."""
        branches = {"path_a": "path_a", "path_b": "path_b", "path_c": "path_c"}
        coal_settings = CoalesceSettings(
            name="merge",
            branches=branches,
            policy="quorum",
            quorum_count=2,
            merge="union",
        )

        # Build graph with 3 branches
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="src", config={})
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="gate", config={})
        graph.add_edge("source", "gate", label="continue")

        for branch in ["path_a", "path_b", "path_c"]:
            t_name = f"t_{branch}"
            graph.add_node(t_name, node_type=NodeType.TRANSFORM, plugin_name=f"transform-{branch}", config={})
            graph.add_edge("gate", t_name, label=branch, mode=RoutingMode.MOVE)
            graph.add_edge(t_name, "coalesce", label="continue", mode=RoutingMode.MOVE)

        coalesce_config = {"branches": branches, "policy": "quorum", "merge": "union", "quorum_count": 2}
        graph.add_node("coalesce", node_type=NodeType.COALESCE, plugin_name="coalesce:merge", config=coalesce_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="sink", config={})
        graph.add_edge("coalesce", "sink", label="continue")
        graph.add_node("err", node_type=NodeType.SINK, plugin_name="err", config={})
        graph.add_edge("t_path_a", "err", label="__error__", mode=RoutingMode.DIVERT)

        configs = {NodeID("coalesce"): coal_settings}
        warnings = graph.warn_divert_coalesce_interactions(configs)
        assert len(warnings) == 0

    def test_empty_coalesce_configs_no_crash(self) -> None:
        """Empty coalesce_configs should return empty list."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="src", config={})
        warnings = graph.warn_divert_coalesce_interactions({})
        assert warnings == []

    def test_warning_is_frozen_dataclass(self) -> None:
        """GraphValidationWarning should be immutable."""
        w = GraphValidationWarning(
            code="TEST",
            message="test message",
            node_ids=("a", "b"),
        )
        assert w.code == "TEST"
        with pytest.raises(AttributeError):
            w.code = "MODIFIED"  # type: ignore[misc]
