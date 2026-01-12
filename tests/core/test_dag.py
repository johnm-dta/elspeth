# tests/core/test_dag.py
"""Tests for DAG validation and operations."""

import pytest


class TestDAGBuilder:
    """Building execution graphs from configuration."""

    def test_empty_dag(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_add_node(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")

        assert graph.node_count == 1
        assert graph.has_node("source")

    def test_add_edge(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("transform", node_type="transform", plugin_name="validate")
        graph.add_edge("source", "transform", label="continue")

        assert graph.edge_count == 1

    def test_linear_pipeline(self) -> None:
        from elspeth.core.dag import ExecutionGraph

        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="csv")
        graph.add_node("t1", node_type="transform", plugin_name="enrich")
        graph.add_node("t2", node_type="transform", plugin_name="classify")
        graph.add_node("sink", node_type="sink", plugin_name="csv")

        graph.add_edge("source", "t1", label="continue")
        graph.add_edge("t1", "t2", label="continue")
        graph.add_edge("t2", "sink", label="continue")

        assert graph.node_count == 4
        assert graph.edge_count == 3
