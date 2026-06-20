# tests/property/core/test_dag_complex_topologies.py
"""Property-based tests for complex DAG topologies.

The existing test_dag_properties.py covers linear pipelines and single
diamond (2-4 branch) topologies. This module extends coverage to:

- Sequential forks: fork → coalesce → fork → coalesce
- Deep forks: >4 branches (up to 8)
- Mixed destinations: fork branches to both sinks and coalesce
- Parallel coalesces: two independent fork→coalesce paths
- Transforms between fork/coalesce boundaries

These topologies represent real user scenarios that were added with
the full DAG execution model but lack property test coverage.
"""

from __future__ import annotations

import networkx as nx
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.contracts.enums import NodeType
from elspeth.contracts.types import NodeID
from elspeth.core.dag import ExecutionGraph

# =============================================================================
# Complex Topology Strategies
# =============================================================================


@st.composite
def deep_fork_pipelines(draw: st.DrawFn) -> ExecutionGraph:
    """Generate pipelines with deep forks (5-8 branches).

    source → [branch_0..branch_N] → merge → sink
    """
    num_branches = draw(st.integers(min_value=5, max_value=8))
    graph = ExecutionGraph()

    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source")

    for i in range(num_branches):
        branch_id = f"branch_{i}"
        graph.add_node(branch_id, node_type=NodeType.TRANSFORM, plugin_name="test_transform")
        graph.add_edge("source", branch_id, label=f"path_{i}")

    graph.add_node("merge", node_type=NodeType.COALESCE, plugin_name="test_coalesce")
    for i in range(num_branches):
        graph.add_edge(f"branch_{i}", "merge", label=f"join_{i}")

    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")
    graph.add_edge("merge", "sink", label="continue")

    return graph


@st.composite
def sequential_fork_pipelines(draw: st.DrawFn) -> ExecutionGraph:
    """Generate pipelines with two sequential fork-coalesce pairs.

    source → gate1(fork) → [a,b] → coalesce1 → transform → gate2(fork) → [c,d] → coalesce2 → sink

    The number of branches in each fork varies independently.
    """
    branches_1 = draw(st.integers(min_value=2, max_value=4))
    branches_2 = draw(st.integers(min_value=2, max_value=4))

    graph = ExecutionGraph()

    # Source
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source")

    # First fork
    for i in range(branches_1):
        bid = f"fork1_branch_{i}"
        graph.add_node(bid, node_type=NodeType.TRANSFORM, plugin_name="test_transform")
        graph.add_edge("source", bid, label=f"fork1_path_{i}")

    # First coalesce
    graph.add_node("coalesce1", node_type=NodeType.COALESCE, plugin_name="test_coalesce")
    for i in range(branches_1):
        graph.add_edge(f"fork1_branch_{i}", "coalesce1", label=f"fork1_join_{i}")

    # Middle transform
    graph.add_node("mid_transform", node_type=NodeType.TRANSFORM, plugin_name="test_transform")
    graph.add_edge("coalesce1", "mid_transform", label="continue")

    # Second fork
    for i in range(branches_2):
        bid = f"fork2_branch_{i}"
        graph.add_node(bid, node_type=NodeType.TRANSFORM, plugin_name="test_transform")
        graph.add_edge("mid_transform", bid, label=f"fork2_path_{i}")

    # Second coalesce
    graph.add_node("coalesce2", node_type=NodeType.COALESCE, plugin_name="test_coalesce")
    for i in range(branches_2):
        graph.add_edge(f"fork2_branch_{i}", "coalesce2", label=f"fork2_join_{i}")

    # Sink
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")
    graph.add_edge("coalesce2", "sink", label="continue")

    return graph


@st.composite
def parallel_coalesce_pipelines(draw: st.DrawFn) -> ExecutionGraph:
    """Generate pipelines with two independent fork-coalesce paths.

    source → transform → gate →
        path_a: [a1, a2] → coalesce_a → sink_a
        path_b: [b1, b2] → coalesce_b → sink_b

    Both fork-coalesce pairs operate independently.
    """
    branches_a = draw(st.integers(min_value=2, max_value=3))
    branches_b = draw(st.integers(min_value=2, max_value=3))

    graph = ExecutionGraph()

    # Source and initial transform
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source")
    graph.add_node("gate", node_type=NodeType.GATE, plugin_name="test_gate")
    graph.add_edge("source", "gate", label="continue")

    # Fork A branches
    for i in range(branches_a):
        bid = f"a_branch_{i}"
        graph.add_node(bid, node_type=NodeType.TRANSFORM, plugin_name="test_transform")
        graph.add_edge("gate", bid, label=f"a_path_{i}")

    # Coalesce A
    graph.add_node("coalesce_a", node_type=NodeType.COALESCE, plugin_name="test_coalesce")
    for i in range(branches_a):
        graph.add_edge(f"a_branch_{i}", "coalesce_a", label=f"a_join_{i}")

    # Sink A
    graph.add_node("sink_a", node_type=NodeType.SINK, plugin_name="test_sink")
    graph.add_edge("coalesce_a", "sink_a", label="continue")

    # Fork B branches
    for i in range(branches_b):
        bid = f"b_branch_{i}"
        graph.add_node(bid, node_type=NodeType.TRANSFORM, plugin_name="test_transform")
        graph.add_edge("gate", bid, label=f"b_path_{i}")

    # Coalesce B
    graph.add_node("coalesce_b", node_type=NodeType.COALESCE, plugin_name="test_coalesce")
    for i in range(branches_b):
        graph.add_edge(f"b_branch_{i}", "coalesce_b", label=f"b_join_{i}")

    # Sink B
    graph.add_node("sink_b", node_type=NodeType.SINK, plugin_name="test_sink")
    graph.add_edge("coalesce_b", "sink_b", label="continue")

    return graph


@st.composite
def fork_with_branch_transforms(draw: st.DrawFn) -> ExecutionGraph:
    """Generate pipelines with transforms on fork branches.

    source → [branch_0: t0a→t0b, branch_1: t1a→t1b, ...] → merge → sink

    Each branch has 1-3 transforms, varying independently.
    """
    num_branches = draw(st.integers(min_value=2, max_value=4))
    graph = ExecutionGraph()

    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source")

    branch_tails = []
    for i in range(num_branches):
        num_transforms = draw(st.integers(min_value=1, max_value=3))
        prev = "source"
        for j in range(num_transforms):
            tid = f"b{i}_t{j}"
            graph.add_node(tid, node_type=NodeType.TRANSFORM, plugin_name="test_transform")
            label = f"path_{i}" if j == 0 else f"b{i}_continue_{j}"
            graph.add_edge(prev, tid, label=label)
            prev = tid
        branch_tails.append(prev)

    graph.add_node("merge", node_type=NodeType.COALESCE, plugin_name="test_coalesce")
    for i, tail in enumerate(branch_tails):
        graph.add_edge(tail, "merge", label=f"join_{i}")

    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")
    graph.add_edge("merge", "sink", label="continue")

    return graph


# =============================================================================
# Topological Order Properties for Complex Topologies
# =============================================================================


class TestDeepForkTopology:
    """Property tests for deep forks (5-8 branches)."""

    @given(graph=deep_fork_pipelines())
    @settings(max_examples=50)
    def test_topo_order_respects_all_edges(self, graph: ExecutionGraph) -> None:
        """Property: Topological order respects all edges in deep forks."""
        topo_order = graph.topological_order()
        index_map = {node: idx for idx, node in enumerate(topo_order)}

        for edge in graph.get_edges():
            assert index_map[edge.from_node] < index_map[edge.to_node], f"Edge {edge.from_node} → {edge.to_node} violates topological order"

    @given(graph=deep_fork_pipelines())
    @settings(max_examples=50)
    def test_deep_fork_is_acyclic(self, graph: ExecutionGraph) -> None:
        """Property: Deep fork pipelines are acyclic."""
        assert graph.is_acyclic()

    @given(graph=deep_fork_pipelines())
    @settings(max_examples=50)
    def test_all_nodes_reachable(self, graph: ExecutionGraph) -> None:
        """Property: All nodes reachable from source in deep forks."""
        source = graph.get_sources()[0]
        nx_graph = graph.get_nx_graph()
        reachable = {NodeID(n) for n in nx.descendants(nx_graph, source)} | {source}
        all_nodes = {info.node_id for info in graph.get_nodes()}
        assert reachable == all_nodes

    @given(graph=deep_fork_pipelines())
    @settings(max_examples=50)
    def test_source_before_branches_before_merge(self, graph: ExecutionGraph) -> None:
        """Property: source < all branches < merge < sink in topo order."""
        topo_order = graph.topological_order()
        index_map = {node: idx for idx, node in enumerate(topo_order)}

        branch_ids = [n for n in index_map if str(n).startswith("branch_")]

        for bid in branch_ids:
            assert index_map[NodeID("source")] < index_map[bid]
            assert index_map[bid] < index_map[NodeID("merge")]

        assert index_map[NodeID("merge")] < index_map[NodeID("sink")]


class TestSequentialForkTopology:
    """Property tests for sequential fork-coalesce pairs."""

    @given(graph=sequential_fork_pipelines())
    @settings(max_examples=50)
    def test_topo_order_respects_all_edges(self, graph: ExecutionGraph) -> None:
        """Property: All edges respected in sequential fork topology."""
        topo_order = graph.topological_order()
        index_map = {node: idx for idx, node in enumerate(topo_order)}

        for edge in graph.get_edges():
            assert index_map[edge.from_node] < index_map[edge.to_node], f"Edge {edge.from_node} → {edge.to_node} violates topo order"

    @given(graph=sequential_fork_pipelines())
    @settings(max_examples=50)
    def test_sequential_forks_acyclic(self, graph: ExecutionGraph) -> None:
        """Property: Sequential fork-coalesce pipelines are acyclic."""
        assert graph.is_acyclic()

    @given(graph=sequential_fork_pipelines())
    @settings(max_examples=50)
    def test_coalesce_ordering(self, graph: ExecutionGraph) -> None:
        """Property: coalesce1 < mid_transform < coalesce2 in topo order."""
        topo_order = graph.topological_order()
        index_map = {node: idx for idx, node in enumerate(topo_order)}

        assert index_map[NodeID("coalesce1")] < index_map[NodeID("mid_transform")]
        assert index_map[NodeID("mid_transform")] < index_map[NodeID("coalesce2")]

    @given(graph=sequential_fork_pipelines())
    @settings(max_examples=50)
    def test_all_nodes_reachable(self, graph: ExecutionGraph) -> None:
        """Property: All nodes reachable from source."""
        source = graph.get_sources()[0]
        nx_graph = graph.get_nx_graph()
        reachable = {NodeID(n) for n in nx.descendants(nx_graph, source)} | {source}
        all_nodes = {info.node_id for info in graph.get_nodes()}
        assert reachable == all_nodes

    @given(graph=sequential_fork_pipelines())
    @settings(max_examples=30)
    def test_fork1_branches_before_coalesce1(self, graph: ExecutionGraph) -> None:
        """Property: Fork1 branches all precede coalesce1."""
        topo_order = graph.topological_order()
        index_map = {node: idx for idx, node in enumerate(topo_order)}

        fork1_branches = [n for n in index_map if str(n).startswith("fork1_branch_")]
        for bid in fork1_branches:
            assert index_map[bid] < index_map[NodeID("coalesce1")]

    @given(graph=sequential_fork_pipelines())
    @settings(max_examples=30)
    def test_fork2_branches_before_coalesce2(self, graph: ExecutionGraph) -> None:
        """Property: Fork2 branches all precede coalesce2."""
        topo_order = graph.topological_order()
        index_map = {node: idx for idx, node in enumerate(topo_order)}

        fork2_branches = [n for n in index_map if str(n).startswith("fork2_branch_")]
        for bid in fork2_branches:
            assert index_map[bid] < index_map[NodeID("coalesce2")]


class TestParallelCoalesceTopology:
    """Property tests for two independent fork-coalesce paths."""

    @given(graph=parallel_coalesce_pipelines())
    @settings(max_examples=50)
    def test_topo_order_respects_all_edges(self, graph: ExecutionGraph) -> None:
        """Property: All edges respected in parallel coalesce topology."""
        topo_order = graph.topological_order()
        index_map = {node: idx for idx, node in enumerate(topo_order)}

        for edge in graph.get_edges():
            assert index_map[edge.from_node] < index_map[edge.to_node]

    @given(graph=parallel_coalesce_pipelines())
    @settings(max_examples=50)
    def test_parallel_coalesce_acyclic(self, graph: ExecutionGraph) -> None:
        """Property: Parallel coalesce pipelines are acyclic."""
        assert graph.is_acyclic()

    @given(graph=parallel_coalesce_pipelines())
    @settings(max_examples=50)
    def test_two_sinks_present(self, graph: ExecutionGraph) -> None:
        """Property: Parallel coalesce pipeline has exactly 2 sinks."""
        sinks = graph.get_sinks()
        assert len(sinks) == 2

    @given(graph=parallel_coalesce_pipelines())
    @settings(max_examples=50)
    def test_all_nodes_reachable(self, graph: ExecutionGraph) -> None:
        """Property: All nodes reachable from source."""
        source = graph.get_sources()[0]
        nx_graph = graph.get_nx_graph()
        reachable = {NodeID(n) for n in nx.descendants(nx_graph, source)} | {source}
        all_nodes = {info.node_id for info in graph.get_nodes()}
        assert reachable == all_nodes

    @given(graph=parallel_coalesce_pipelines())
    @settings(max_examples=30)
    def test_a_branches_precede_coalesce_a(self, graph: ExecutionGraph) -> None:
        """Property: A-path branches all precede coalesce_a."""
        topo_order = graph.topological_order()
        index_map = {node: idx for idx, node in enumerate(topo_order)}

        a_branches = [n for n in index_map if str(n).startswith("a_branch_")]
        for bid in a_branches:
            assert index_map[bid] < index_map[NodeID("coalesce_a")]

    @given(graph=parallel_coalesce_pipelines())
    @settings(max_examples=30)
    def test_b_branches_precede_coalesce_b(self, graph: ExecutionGraph) -> None:
        """Property: B-path branches all precede coalesce_b."""
        topo_order = graph.topological_order()
        index_map = {node: idx for idx, node in enumerate(topo_order)}

        b_branches = [n for n in index_map if str(n).startswith("b_branch_")]
        for bid in b_branches:
            assert index_map[bid] < index_map[NodeID("coalesce_b")]


class TestBranchTransformTopology:
    """Property tests for forks with transforms on branches."""

    @given(graph=fork_with_branch_transforms())
    @settings(max_examples=50)
    def test_topo_order_respects_all_edges(self, graph: ExecutionGraph) -> None:
        """Property: All edges respected with branch transforms."""
        topo_order = graph.topological_order()
        index_map = {node: idx for idx, node in enumerate(topo_order)}

        for edge in graph.get_edges():
            assert index_map[edge.from_node] < index_map[edge.to_node]

    @given(graph=fork_with_branch_transforms())
    @settings(max_examples=50)
    def test_branch_transforms_acyclic(self, graph: ExecutionGraph) -> None:
        """Property: Fork with branch transforms is acyclic."""
        assert graph.is_acyclic()

    @given(graph=fork_with_branch_transforms())
    @settings(max_examples=50)
    def test_all_nodes_reachable(self, graph: ExecutionGraph) -> None:
        """Property: All nodes reachable from source."""
        source = graph.get_sources()[0]
        nx_graph = graph.get_nx_graph()
        reachable = {NodeID(n) for n in nx.descendants(nx_graph, source)} | {source}
        all_nodes = {info.node_id for info in graph.get_nodes()}
        assert reachable == all_nodes

    @given(graph=fork_with_branch_transforms())
    @settings(max_examples=50)
    def test_sinks_have_no_outgoing(self, graph: ExecutionGraph) -> None:
        """Property: Sink nodes have zero out-degree."""
        nx_graph = graph.get_nx_graph()
        for sink in graph.get_sinks():
            assert nx_graph.out_degree(sink) == 0

    @given(graph=fork_with_branch_transforms())
    @settings(max_examples=30)
    def test_node_count_consistent(self, graph: ExecutionGraph) -> None:
        """Property: Topological order contains all nodes."""
        topo_order = graph.topological_order()
        assert len(topo_order) == graph.node_count
