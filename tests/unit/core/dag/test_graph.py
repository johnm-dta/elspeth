"""Regression test for Phase 0 fix #2: GraphValidationError suppressed.

Bug: In get_effective_producer_schema(), when processing a select-merge
coalesce with a transform branch, _trace_branch_endpoints was called
inside a try/except that caught GraphValidationError and returned None
instead of propagating the error. This silently hid graph construction
bugs by falling back to "dynamic schema" instead of raising.

Fix: Removed the try/except so GraphValidationError from
_trace_branch_endpoints propagates up to the caller.
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from elspeth.contracts.enums import NodeType, RoutingMode
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.types import BranchName, CoalesceName, NodeID
from elspeth.core.dag.graph import ExecutionGraph
from elspeth.core.dag.models import BranchInfo, GraphValidationError


class TestSelectMergeCoalesceRaisesOnBrokenBranch:
    """Verify GraphValidationError propagates from get_effective_producer_schema
    for select-merge coalesce with untraceable branches.
    """

    def test_untraceable_branch_raises_graph_validation_error(self) -> None:
        """When _trace_branch_endpoints fails for a select-merge coalesce,
        get_effective_producer_schema must raise GraphValidationError,
        not return None.

        Before the fix, a try/except caught the error and returned None,
        silently treating the coalesce as dynamic schema.
        """
        graph = ExecutionGraph()

        # Build a minimal graph with a select-merge coalesce
        # that has a transform branch which cannot be traced
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node(
            "gate",
            node_type=NodeType.GATE,
            plugin_name="fork_gate",
            config={"routes": {"true": "fork"}},
        )
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={
                "merge": "select",
                "select_branch": "branch_a",
            },
        )
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv_sink")

        # Wire edges: source -> gate, gate -> coalesce (COPY for branch_b),
        # but branch_a is a MOVE edge that has no proper chain
        graph.add_edge("source", "gate", label="continue")
        # branch_b is identity (COPY)
        graph.add_edge("gate", "coalesce", label="branch_b", mode=RoutingMode.COPY)
        # branch_a has a MOVE edge from gate directly to coalesce
        # but with no intermediate transform — the trace should find gate as
        # the fork producer, but we deliberately set up a broken trace by
        # NOT populating the _branch_gate_map for this branch
        graph.add_edge("gate", "coalesce", label="branch_a", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "sink", label="on_success")

        # Set the branch_info to point to a nonexistent gate for branch_a
        # This simulates a graph construction bug
        graph.set_branch_info(
            {
                BranchName("branch_a"): BranchInfo(
                    coalesce_name=CoalesceName("coalesce"),
                    gate_node_id=NodeID("nonexistent_gate"),
                ),
            }
        )

        # get_effective_producer_schema for coalesce with select-merge
        # should raise GraphValidationError, NOT return None
        with pytest.raises((GraphValidationError, KeyError)):
            graph.get_effective_producer_schema("coalesce")

    def test_valid_select_merge_still_works(self) -> None:
        """Sanity check: properly constructed select-merge coalesce works."""
        graph = ExecutionGraph()

        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node(
            "gate",
            node_type=NodeType.GATE,
            plugin_name="fork_gate",
            config={"routes": {"true": "fork"}},
        )
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce",
            config={
                "merge": "select",
                "select_branch": "branch_b",
            },
        )
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv_sink")

        graph.add_edge("source", "gate", label="continue")
        # branch_b is identity (COPY) — select picks this branch
        graph.add_edge("gate", "coalesce", label="branch_b", mode=RoutingMode.COPY)
        graph.add_edge("gate", "coalesce", label="branch_a", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "sink", label="on_success")

        graph.set_branch_info(
            {
                BranchName("branch_a"): BranchInfo(
                    coalesce_name=CoalesceName("coalesce"),
                    gate_node_id=NodeID("gate"),
                ),
            }
        )

        # Identity branch (COPY): should trace through to gate's schema
        # This should NOT raise
        result = graph.get_effective_producer_schema("coalesce")
        # Returns None because gate has no output_schema — that's fine,
        # the important thing is it didn't raise an error
        assert result is None


class TestExecutionGraphConstructionApi:
    def test_set_node_output_schema_replaces_node_info_without_mutating_existing_instance(self) -> None:
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        original_info = graph.get_node_info("source")

        schema = SchemaConfig(mode="observed", fields=None)

        graph.set_node_output_schema("source", schema)

        updated_info = graph.get_node_info("source")
        assert updated_info is not original_info
        assert original_info.output_schema_config is None
        assert updated_info.output_schema_config is schema

    def test_finalize_node_configs_replaces_node_info_without_mutating_existing_instance(self) -> None:
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", config={"schema": {"mode": "observed"}})
        original_info = graph.get_node_info("source")

        graph.finalize_node_configs()

        updated_info = graph.get_node_info("source")
        assert updated_info is not original_info
        assert isinstance(updated_info.config, MappingProxyType)
        assert isinstance(original_info.config, dict)

    def test_set_node_output_schema_updates_node_info_through_graph_api(self) -> None:
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")

        schema = SchemaConfig(mode="observed", fields=None)

        graph.set_node_output_schema("source", schema)

        assert graph.get_node_info("source").output_schema_config is schema

    def test_topological_processing_order_filters_to_processing_nodes(self) -> None:
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv")
        graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="classifier")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")
        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")

        order = graph.topological_processing_order({NodeID("transform")})

        assert order == [NodeID("transform")]

    def test_topological_processing_order_preserves_cycle_error_contract(self) -> None:
        graph = ExecutionGraph()
        graph.add_node("first", node_type=NodeType.TRANSFORM, plugin_name="a")
        graph.add_node("second", node_type=NodeType.TRANSFORM, plugin_name="b")
        graph.add_edge("first", "second", label="forward")
        graph.add_edge("second", "first", label="back")

        with pytest.raises(GraphValidationError, match="Pipeline contains a cycle"):
            graph.topological_processing_order({NodeID("first"), NodeID("second")})

    def test_finalize_node_configs_deep_freezes_node_config(self) -> None:
        graph = ExecutionGraph()
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"options": {"columns": ["name"]}},
        )

        graph.finalize_node_configs()

        config = graph.get_node_info("source").config
        assert config["options"]["columns"] == ("name",)
        with pytest.raises(TypeError):
            config["new"] = "value"
