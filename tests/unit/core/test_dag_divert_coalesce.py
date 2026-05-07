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
from elspeth.contracts.types import BranchName, CoalesceName, NodeID
from elspeth.core.config import CoalesceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.dag.models import BranchInfo, GraphValidationWarning


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
        **kwargs,
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

    def test_intermediate_gate_branch_preserves_require_all_warning(self) -> None:
        """Intermediate gates must not hide upstream DIVERT transforms."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="src", config={})
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork", config={})
        graph.add_node("t_a1", node_type=NodeType.TRANSFORM, plugin_name="transform-a1", config={})
        graph.add_node("inner_gate", node_type=NodeType.GATE, plugin_name="route-gate", config={})
        graph.add_node("t_a2", node_type=NodeType.TRANSFORM, plugin_name="transform-a2", config={})
        graph.add_node("t_b", node_type=NodeType.TRANSFORM, plugin_name="transform-b", config={})
        graph.add_node("coalesce", node_type=NodeType.COALESCE, plugin_name="coalesce:merge", config={})
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="sink", config={})
        graph.add_node("error_sink", node_type=NodeType.SINK, plugin_name="error", config={})

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_a1", label="path_a", mode=RoutingMode.MOVE)
        graph.add_edge("t_a1", "inner_gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("inner_gate", "t_a2", label="approved", mode=RoutingMode.MOVE)
        graph.add_edge("t_a2", "coalesce", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_b", label="path_b", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("t_a1", "error_sink", label="__error_t_a1__", mode=RoutingMode.DIVERT)
        graph.add_edge("coalesce", "sink", label="continue", mode=RoutingMode.MOVE)

        graph.set_branch_info(
            {
                BranchName("path_a"): BranchInfo(
                    coalesce_name=CoalesceName("merge"),
                    gate_node_id=NodeID("gate"),
                ),
                BranchName("path_b"): BranchInfo(
                    coalesce_name=CoalesceName("merge"),
                    gate_node_id=NodeID("gate"),
                ),
            }
        )
        graph.set_coalesce_id_map({CoalesceName("merge"): NodeID("coalesce")})

        warnings = graph.warn_divert_coalesce_interactions(
            {
                NodeID("coalesce"): CoalesceSettings(
                    name="merge",
                    branches={"path_a": "path_a", "path_b": "path_b"},
                    policy="require_all",
                    merge="union",
                )
            }
        )

        assert len(warnings) == 1
        warning = warnings[0]
        assert warning.code == "DIVERT_COALESCE_REQUIRE_ALL"
        assert "t_a1" in warning.message
        assert "t_a1" in warning.node_ids

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


class TestDivertCoalesceExclusiveFields:
    """Tests for DIVERT_COALESCE_EXCLUSIVE_FIELDS warning.

    This warning is emitted when a branch with DIVERT transforms carries
    fields that no other branch provides. If the branch is diverted, those
    fields silently disappear from the merged output.
    """

    def _build_graph_with_schemas(
        self,
        *,
        branch_a_fields: tuple[str, ...] | None = None,
        branch_b_fields: tuple[str, ...] | None = None,
        divert_on: str | None = None,
        policy: str = "best_effort",
        merge: str = "union",
    ) -> tuple[ExecutionGraph, dict[NodeID, CoalesceSettings]]:
        """Build a graph with explicit schema configs on transforms.

        Args:
            branch_a_fields: Fields guaranteed by branch A (None = observed mode)
            branch_b_fields: Fields guaranteed by branch B (None = observed mode)
            divert_on: Transform to add DIVERT edge to
            policy: Coalesce policy
            merge: Coalesce merge strategy
        """
        from elspeth.contracts.schema import FieldDefinition, SchemaConfig

        graph = ExecutionGraph()

        # Source
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="src", config={})

        # Gate
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="gate", config={})
        graph.add_edge("source", "gate", label="continue")

        # Error sink
        graph.add_node("error_sink", node_type=NodeType.SINK, plugin_name="err", config={})

        # Build schema configs
        def _make_schema(fields: tuple[str, ...] | None) -> SchemaConfig:
            if fields is None:
                return SchemaConfig(mode="observed", fields=None)
            return SchemaConfig(
                mode="flexible",
                fields=tuple(FieldDefinition(name=f, field_type="str", required=True) for f in fields),
                guaranteed_fields=fields,
            )

        # Transform A with schema
        schema_a = _make_schema(branch_a_fields)
        graph.add_node(
            "t_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform-a",
            config={},
            output_schema_config=schema_a,
        )
        graph.add_edge("gate", "t_a", label="path_a", mode=RoutingMode.MOVE)

        # Transform B with schema
        schema_b = _make_schema(branch_b_fields)
        graph.add_node(
            "t_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform-b",
            config={},
            output_schema_config=schema_b,
        )
        graph.add_edge("gate", "t_b", label="path_b", mode=RoutingMode.MOVE)

        # Coalesce
        branches = {"path_a": "path_a", "path_b": "path_b"}
        coalesce_config = {
            "branches": branches,
            "policy": policy,
            "merge": merge,
        }
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config=coalesce_config,
        )
        graph.add_edge("t_a", "coalesce", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="continue", mode=RoutingMode.MOVE)

        # Sink
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="sink", config={})
        graph.add_edge("coalesce", "sink", label="continue")

        # Add DIVERT edge if requested
        if divert_on is not None:
            graph.add_edge(divert_on, "error_sink", label=f"__error_{divert_on}__", mode=RoutingMode.DIVERT)

        # Build BranchInfo for _trace_branch_endpoints and get_coalesce_branch_schemas
        from elspeth.contracts.types import BranchName, CoalesceName
        from elspeth.core.dag.models import BranchInfo

        graph.set_branch_info(
            {
                BranchName("path_a"): BranchInfo(
                    coalesce_name=CoalesceName("merge"),
                    gate_node_id=NodeID("gate"),
                    schema=schema_a,
                ),
                BranchName("path_b"): BranchInfo(
                    coalesce_name=CoalesceName("merge"),
                    gate_node_id=NodeID("gate"),
                    schema=schema_b,
                ),
            }
        )
        graph.set_coalesce_id_map({CoalesceName("merge"): NodeID("coalesce")})

        coal_settings = CoalesceSettings(
            name="merge",
            branches=branches,
            policy=policy,
            merge=merge,
            timeout_seconds=30.0 if policy == "best_effort" else None,
        )
        configs = {NodeID("coalesce"): coal_settings}

        return graph, configs

    def test_exclusive_fields_warning_emitted(self) -> None:
        """Branch with exclusive fields + DIVERT → DIVERT_COALESCE_EXCLUSIVE_FIELDS warning."""
        # Branch A has exclusive field "only_a"
        # Branch B has shared field "shared"
        graph, configs = self._build_graph_with_schemas(
            branch_a_fields=("shared", "only_a"),
            branch_b_fields=("shared",),
            divert_on="t_a",
            policy="best_effort",
        )
        warnings = graph.warn_divert_coalesce_interactions(configs)

        # Should have one warning for exclusive fields (not require_all, so no timing warning)
        assert len(warnings) == 1
        w = warnings[0]
        assert w.code == "DIVERT_COALESCE_EXCLUSIVE_FIELDS"
        assert "only_a" in w.message
        assert "path_a" in w.message
        assert "t_a" in w.message

    def test_no_exclusive_fields_no_warning(self) -> None:
        """Both branches have same fields + DIVERT → no EXCLUSIVE_FIELDS warning."""
        graph, configs = self._build_graph_with_schemas(
            branch_a_fields=("shared", "also_shared"),
            branch_b_fields=("shared", "also_shared"),
            divert_on="t_a",
            policy="best_effort",
        )
        warnings = graph.warn_divert_coalesce_interactions(configs)

        # No warnings — fields are not exclusive
        assert len(warnings) == 0

    def test_observed_schema_no_warning(self) -> None:
        """Observed schema (no guaranteed fields) + DIVERT → no warning."""
        graph, configs = self._build_graph_with_schemas(
            branch_a_fields=None,  # observed mode
            branch_b_fields=None,  # observed mode
            divert_on="t_a",
            policy="best_effort",
        )
        warnings = graph.warn_divert_coalesce_interactions(configs)

        # No warnings — observed schemas have no guaranteed fields to lose
        assert len(warnings) == 0

    def test_nested_merge_no_exclusive_fields_warning(self) -> None:
        """Nested merge strategy → no EXCLUSIVE_FIELDS warning (not applicable)."""
        graph, configs = self._build_graph_with_schemas(
            branch_a_fields=("only_a",),
            branch_b_fields=("only_b",),
            divert_on="t_a",
            policy="best_effort",
            merge="nested",
        )
        warnings = graph.warn_divert_coalesce_interactions(configs)

        # No warnings — nested merge keys branches separately
        assert len(warnings) == 0

    def test_require_all_gets_both_warnings(self) -> None:
        """require_all + exclusive fields → both REQUIRE_ALL and EXCLUSIVE_FIELDS warnings."""
        graph, configs = self._build_graph_with_schemas(
            branch_a_fields=("shared", "only_a"),
            branch_b_fields=("shared",),
            divert_on="t_a",
            policy="require_all",
        )
        warnings = graph.warn_divert_coalesce_interactions(configs)

        # Should have both warnings
        assert len(warnings) == 2
        codes = {w.code for w in warnings}
        assert "DIVERT_COALESCE_REQUIRE_ALL" in codes
        assert "DIVERT_COALESCE_EXCLUSIVE_FIELDS" in codes

    def test_intermediate_gate_branch_preserves_exclusive_fields_warning(self) -> None:
        """Intermediate gates must not hide branch-exclusive field warnings."""
        from elspeth.contracts.schema import FieldDefinition, SchemaConfig

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="src", config={})
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork", config={})
        graph.add_node(
            "t_a1",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform-a1",
            config={},
            output_schema_config=SchemaConfig(
                mode="flexible",
                fields=(
                    FieldDefinition("shared", "str", required=True),
                    FieldDefinition("only_a", "str", required=True),
                ),
                guaranteed_fields=("shared", "only_a"),
            ),
        )
        graph.add_node("inner_gate", node_type=NodeType.GATE, plugin_name="route-gate", config={})
        graph.add_node(
            "t_a2",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform-a2",
            config={},
            output_schema_config=SchemaConfig(
                mode="flexible",
                fields=(
                    FieldDefinition("shared", "str", required=True),
                    FieldDefinition("only_a", "str", required=True),
                ),
                guaranteed_fields=("shared", "only_a"),
            ),
        )
        graph.add_node(
            "t_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="transform-b",
            config={},
            output_schema_config=SchemaConfig(
                mode="flexible",
                fields=(FieldDefinition("shared", "str", required=True),),
                guaranteed_fields=("shared",),
            ),
        )
        graph.add_node("coalesce", node_type=NodeType.COALESCE, plugin_name="coalesce:merge", config={})
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="sink", config={})
        graph.add_node("error_sink", node_type=NodeType.SINK, plugin_name="error", config={})

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_a1", label="path_a", mode=RoutingMode.MOVE)
        graph.add_edge("t_a1", "inner_gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("inner_gate", "t_a2", label="approved", mode=RoutingMode.MOVE)
        graph.add_edge("t_a2", "coalesce", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_b", label="path_b", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("t_a1", "error_sink", label="__error_t_a1__", mode=RoutingMode.DIVERT)
        graph.add_edge("coalesce", "sink", label="continue", mode=RoutingMode.MOVE)

        graph.set_branch_info(
            {
                BranchName("path_a"): BranchInfo(
                    coalesce_name=CoalesceName("merge"),
                    gate_node_id=NodeID("gate"),
                    schema=SchemaConfig(
                        mode="flexible",
                        fields=(
                            FieldDefinition("shared", "str", required=True),
                            FieldDefinition("only_a", "str", required=True),
                        ),
                        guaranteed_fields=("shared", "only_a"),
                    ),
                ),
                BranchName("path_b"): BranchInfo(
                    coalesce_name=CoalesceName("merge"),
                    gate_node_id=NodeID("gate"),
                    schema=SchemaConfig(
                        mode="flexible",
                        fields=(FieldDefinition("shared", "str", required=True),),
                        guaranteed_fields=("shared",),
                    ),
                ),
            }
        )
        graph.set_coalesce_id_map({CoalesceName("merge"): NodeID("coalesce")})

        warnings = graph.warn_divert_coalesce_interactions(
            {
                NodeID("coalesce"): CoalesceSettings(
                    name="merge",
                    branches={"path_a": "path_a", "path_b": "path_b"},
                    policy="best_effort",
                    merge="union",
                    timeout_seconds=30.0,
                )
            }
        )

        assert len(warnings) == 1
        warning = warnings[0]
        assert warning.code == "DIVERT_COALESCE_EXCLUSIVE_FIELDS"
        assert "only_a" in warning.message
        assert "t_a1" in warning.message

    def test_divert_on_branch_without_exclusive_fields(self) -> None:
        """DIVERT on branch B (no exclusive fields) → no EXCLUSIVE_FIELDS warning."""
        # Branch A has exclusive field, but DIVERT is on B (which has no exclusive fields)
        graph, configs = self._build_graph_with_schemas(
            branch_a_fields=("shared", "only_a"),
            branch_b_fields=("shared",),
            divert_on="t_b",  # DIVERT on B, not A
            policy="best_effort",
        )
        warnings = graph.warn_divert_coalesce_interactions(configs)

        # No warnings — B has no exclusive fields
        assert len(warnings) == 0

    def test_multiple_exclusive_fields_listed(self) -> None:
        """Branch with multiple exclusive fields → all listed in warning."""
        graph, configs = self._build_graph_with_schemas(
            branch_a_fields=("shared", "only_a1", "only_a2", "only_a3"),
            branch_b_fields=("shared",),
            divert_on="t_a",
            policy="best_effort",
        )
        warnings = graph.warn_divert_coalesce_interactions(configs)

        assert len(warnings) == 1
        w = warnings[0]
        assert w.code == "DIVERT_COALESCE_EXCLUSIVE_FIELDS"
        assert "only_a1" in w.message
        assert "only_a2" in w.message
        assert "only_a3" in w.message
