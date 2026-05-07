# tests/property/core/test_dag_properties.py
"""Property-based tests for DAG (Directed Acyclic Graph) operations.

These tests verify the fundamental invariants of ELSPETH's execution graph:
- Topological sort respects all edges
- Validation guarantees structural correctness
- Acyclicity is correctly detected

The DAG is the backbone of pipeline execution - incorrect behavior here
would cause transforms to execute out of order or produce corrupt audit trails.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from elspeth.contracts.enums import NodeType
from elspeth.contracts.schema import FieldDefinition, SchemaConfig
from elspeth.contracts.types import NodeID
from elspeth.core.dag import (
    ExecutionGraph,
    GraphValidationError,
    merge_guaranteed_fields,
    merge_union_fields,
)
from tests.helpers.coalesce import _add_coalesce_with_computed_schema

# Type alias matching FieldDefinition.field_type
_FieldType = Literal["str", "int", "float", "bool", "any"]

# =============================================================================
# Strategies for generating valid DAG structures
# =============================================================================

# Node type choices
node_types = st.sampled_from(["transform"])  # Start simple, add source/sink explicitly

# Valid node IDs (alphanumeric, no special chars)
node_ids = st.text(
    min_size=1,
    max_size=20,
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
).filter(lambda s: s[0].isalpha())


@st.composite
def linear_pipelines(draw: st.DrawFn, min_transforms: int = 1, max_transforms: int = 5) -> ExecutionGraph:
    """Generate a valid linear pipeline: source -> transform(s) -> sink."""
    num_transforms = draw(st.integers(min_value=min_transforms, max_value=max_transforms))

    graph = ExecutionGraph()

    # Add source
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source")

    # Add transforms
    prev_node = "source"
    for i in range(num_transforms):
        node_id = f"transform_{i}"
        graph.add_node(node_id, node_type=NodeType.TRANSFORM, plugin_name="test_transform")
        graph.add_edge(prev_node, node_id, label="continue")
        prev_node = node_id

    # Add sink
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")
    graph.add_edge(prev_node, "sink", label="continue")

    return graph


@st.composite
def diamond_pipelines(draw: st.DrawFn) -> ExecutionGraph:
    """Generate a diamond-shaped pipeline: source -> [branches...] -> merge -> sink.

    Varies the number of parallel branches (2-4) so Hypothesis explores
    different fan-out widths in the DAG.
    """
    num_branches = draw(st.integers(min_value=2, max_value=4))
    graph = ExecutionGraph()

    # Source
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source")

    # Parallel branches
    for i in range(num_branches):
        branch_id = f"branch_{i}"
        graph.add_node(branch_id, node_type=NodeType.TRANSFORM, plugin_name="test_transform")
        graph.add_edge("source", branch_id, label=f"path_{i}")

    # Merge point (simplified - in real ELSPETH this would be a coalesce)
    graph.add_node("merge", node_type=NodeType.TRANSFORM, plugin_name="test_transform")
    for i in range(num_branches):
        graph.add_edge(f"branch_{i}", "merge", label=f"continue_{i}")

    # Sink
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")
    graph.add_edge("merge", "sink", label="continue")

    return graph


@st.composite
def multi_sink_pipelines(draw: st.DrawFn) -> ExecutionGraph:
    """Generate a pipeline with multiple sinks.

    Varies the number of sinks (2-5) so Hypothesis explores different
    fan-out patterns at the terminal end of the DAG.
    """
    num_sinks = draw(st.integers(min_value=2, max_value=5))
    graph = ExecutionGraph()

    # Source
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source")

    # Transform
    graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="test_transform")
    graph.add_edge("source", "transform", label="continue")

    # Multiple sinks with unique edge labels
    for i in range(num_sinks):
        sink_id = f"sink_{i}"
        graph.add_node(sink_id, node_type=NodeType.SINK, plugin_name="test_sink")
        graph.add_edge("transform", sink_id, label=f"route_{i}")

    return graph


# =============================================================================
# Topological Order Property Tests
# =============================================================================


class TestTopologicalOrderProperties:
    """Property tests for topological_order()."""

    @given(graph=linear_pipelines())
    @settings(max_examples=100)
    def test_topological_order_respects_edges(self, graph: ExecutionGraph) -> None:
        """Property: For every edge (u, v), u appears before v in topological order.

        This is THE fundamental property of topological sort.
        """
        topo_order = graph.topological_order()
        index_map = {node: idx for idx, node in enumerate(topo_order)}

        # Check every edge
        for edge in graph.get_edges():
            from_idx = index_map[edge.from_node]
            to_idx = index_map[edge.to_node]
            assert from_idx < to_idx, (
                f"Edge {edge.from_node} -> {edge.to_node} violates topological order: "
                f"index({edge.from_node})={from_idx} >= index({edge.to_node})={to_idx}"
            )

    @given(graph=linear_pipelines())
    @settings(max_examples=100)
    def test_topological_order_contains_all_nodes(self, graph: ExecutionGraph) -> None:
        """Property: Topological order contains exactly all nodes."""
        topo_order = graph.topological_order()

        assert len(topo_order) == graph.node_count, f"Topological order has {len(topo_order)} nodes, graph has {graph.node_count}"

        # All nodes present
        topo_set = set(topo_order)
        for node_info in graph.get_nodes():
            assert node_info.node_id in topo_set, f"Node {node_info.node_id} missing from topological order"

    @given(graph=linear_pipelines())
    @settings(max_examples=100)
    def test_topological_order_source_first(self, graph: ExecutionGraph) -> None:
        """Property: Source node appears first in topological order."""
        topo_order = graph.topological_order()
        source = graph.get_source()

        assert source is not None, "Graph should have a source"
        assert topo_order[0] == source, f"Source {source} is not first, got {topo_order[0]}"

    @given(graph=linear_pipelines())
    @settings(max_examples=100)
    def test_topological_order_sinks_last(self, graph: ExecutionGraph) -> None:
        """Property: Sink nodes appear after all non-sink nodes."""
        topo_order = graph.topological_order()
        sinks = set(graph.get_sinks())

        # Find where sinks start in the order
        first_sink_idx = None
        for idx, node in enumerate(topo_order):
            if node in sinks:
                first_sink_idx = idx
                break

        if first_sink_idx is not None:
            # All nodes after first sink should be sinks
            for idx in range(first_sink_idx, len(topo_order)):
                assert topo_order[idx] in sinks, f"Non-sink {topo_order[idx]} appears after sink at index {first_sink_idx}"

    @given(graph=diamond_pipelines())
    @settings(max_examples=50)
    def test_topological_order_diamond_respects_all_paths(self, graph: ExecutionGraph) -> None:
        """Property: Diamond topology respects all paths through the graph."""
        topo_order = graph.topological_order()
        index_map = {node: idx for idx, node in enumerate(topo_order)}

        # Source before all branches
        branch_ids = [n for n in index_map if n.startswith("branch_")]
        for branch_id in branch_ids:
            assert index_map["source"] < index_map[branch_id]

        # All branches before merge
        for branch_id in branch_ids:
            assert index_map[branch_id] < index_map["merge"]

        # Merge before sink
        assert index_map["merge"] < index_map["sink"]

    @given(graph=linear_pipelines())
    @settings(max_examples=50)
    def test_topological_order_is_deterministic(self, graph: ExecutionGraph) -> None:
        """Property: Calling topological_order() twice returns identical result."""
        order1 = graph.topological_order()
        order2 = graph.topological_order()

        assert order1 == order2, "Topological order is not deterministic"


# =============================================================================
# Graph Validation Property Tests
# =============================================================================


class TestValidationProperties:
    """Property tests for validate()."""

    @given(graph=linear_pipelines())
    @settings(max_examples=100)
    def test_validate_implies_acyclic(self, graph: ExecutionGraph) -> None:
        """Property: After validate() succeeds, is_acyclic() == True."""
        graph.validate()  # Should not raise
        assert graph.is_acyclic(), "Graph passed validation but is_acyclic() is False"

    @given(graph=linear_pipelines())
    @settings(max_examples=100)
    def test_validate_implies_one_source(self, graph: ExecutionGraph) -> None:
        """Property: After validate() succeeds, exactly one source exists."""
        graph.validate()
        source = graph.get_source()
        assert source is not None, "Graph passed validation but has no source"

    @given(graph=linear_pipelines())
    @settings(max_examples=100)
    def test_validate_implies_has_sink(self, graph: ExecutionGraph) -> None:
        """Property: After validate() succeeds, at least one sink exists."""
        graph.validate()
        sinks = graph.get_sinks()
        assert len(sinks) >= 1, "Graph passed validation but has no sinks"

    @given(graph=linear_pipelines())
    @settings(max_examples=50)
    def test_validate_is_idempotent(self, graph: ExecutionGraph) -> None:
        """Property: Calling validate() twice has same result."""
        # First call
        graph.validate()

        # Second call should also succeed
        graph.validate()  # Should not raise

    @given(graph=multi_sink_pipelines())
    @settings(max_examples=50)
    def test_validate_accepts_multiple_sinks(self, graph: ExecutionGraph) -> None:
        """Property: Graphs with multiple sinks are valid."""
        graph.validate()  # Should not raise
        sinks = graph.get_sinks()
        assert len(sinks) >= 2, f"Expected at least 2 sinks, got {len(sinks)}"


class TestValidationFailureProperties:
    """Property tests for validation failure cases."""

    def test_validate_rejects_cycle(self) -> None:
        """Property: Cyclic graphs are rejected."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test")
        graph.add_node("a", node_type=NodeType.TRANSFORM, plugin_name="test")
        graph.add_node("b", node_type=NodeType.TRANSFORM, plugin_name="test")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test")

        # Create cycle: a -> b -> a
        graph.add_edge("source", "a", label="continue")
        graph.add_edge("a", "b", label="continue")
        graph.add_edge("b", "a", label="back")  # Creates cycle
        graph.add_edge("b", "sink", label="continue")

        import pytest

        with pytest.raises(GraphValidationError, match="cycle"):
            graph.validate()

    def test_validate_rejects_no_source(self) -> None:
        """Property: Graphs without source are rejected."""
        graph = ExecutionGraph()
        graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="test")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test")
        graph.add_edge("transform", "sink", label="continue")

        import pytest

        with pytest.raises(GraphValidationError, match="source"):
            graph.validate()

    def test_validate_rejects_multiple_sources(self) -> None:
        """Property: Graphs with multiple sources are rejected."""
        graph = ExecutionGraph()
        graph.add_node("source1", node_type=NodeType.SOURCE, plugin_name="test")
        graph.add_node("source2", node_type=NodeType.SOURCE, plugin_name="test")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test")
        graph.add_edge("source1", "sink", label="path1")
        graph.add_edge("source2", "sink", label="path2")

        import pytest

        with pytest.raises(GraphValidationError, match="exactly one source"):
            graph.validate()

    def test_validate_rejects_no_sink(self) -> None:
        """Property: Graphs without sink are rejected."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test")
        graph.add_node("transform", node_type=NodeType.TRANSFORM, plugin_name="test")
        graph.add_edge("source", "transform", label="continue")

        import pytest

        with pytest.raises(GraphValidationError, match="sink"):
            graph.validate()

    def test_validate_rejects_duplicate_edge_labels(self) -> None:
        """Property: Duplicate edge labels from same node are rejected."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test")
        graph.add_node("sink1", node_type=NodeType.SINK, plugin_name="test")
        graph.add_node("sink2", node_type=NodeType.SINK, plugin_name="test")

        # Same label "continue" to two different sinks
        graph.add_edge("source", "sink1", label="continue")
        graph.add_edge("source", "sink2", label="continue")  # Duplicate label!

        import pytest

        with pytest.raises(GraphValidationError, match="duplicate"):
            graph.validate()


# =============================================================================
# Acyclicity Property Tests
# =============================================================================


class TestAcyclicityProperties:
    """Property tests for is_acyclic()."""

    @given(graph=linear_pipelines())
    @settings(max_examples=100)
    def test_linear_pipeline_is_acyclic(self, graph: ExecutionGraph) -> None:
        """Property: Linear pipelines are always acyclic."""
        assert graph.is_acyclic(), "Linear pipeline should be acyclic"

    @given(graph=diamond_pipelines())
    @settings(max_examples=50)
    def test_diamond_pipeline_is_acyclic(self, graph: ExecutionGraph) -> None:
        """Property: Diamond pipelines (fork/join) are acyclic."""
        assert graph.is_acyclic(), "Diamond pipeline should be acyclic"

    @given(graph=linear_pipelines())
    @settings(max_examples=50)
    def test_is_acyclic_is_deterministic(self, graph: ExecutionGraph) -> None:
        """Property: is_acyclic() returns same result on repeated calls."""
        result1 = graph.is_acyclic()
        result2 = graph.is_acyclic()
        assert result1 == result2, "is_acyclic() is not deterministic"

    def test_adding_back_edge_breaks_acyclicity(self) -> None:
        """Property: Adding edge from later to earlier node creates cycle."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test")
        graph.add_node("a", node_type=NodeType.TRANSFORM, plugin_name="test")
        graph.add_node("b", node_type=NodeType.TRANSFORM, plugin_name="test")
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test")

        # Linear: source -> a -> b -> sink
        graph.add_edge("source", "a", label="continue")
        graph.add_edge("a", "b", label="continue")
        graph.add_edge("b", "sink", label="continue")

        assert graph.is_acyclic(), "Should be acyclic before adding back edge"

        # Add back edge: b -> a (creates cycle)
        graph.add_edge("b", "a", label="back")

        assert not graph.is_acyclic(), "Should be cyclic after adding back edge"


# =============================================================================
# Node/Edge Consistency Property Tests
# =============================================================================


class TestGraphConsistencyProperties:
    """Property tests for graph structure consistency."""

    @given(graph=linear_pipelines())
    @settings(max_examples=100)
    def test_edge_count_matches_structure(self, graph: ExecutionGraph) -> None:
        """Property: Edge count matches expected for linear pipeline."""
        # Linear pipeline: source -> T1 -> T2 -> ... -> sink
        # edges = node_count - 1 (for linear graph)
        expected_edges = graph.node_count - 1
        assert graph.edge_count == expected_edges, f"Expected {expected_edges} edges, got {graph.edge_count}"

    @given(graph=linear_pipelines())
    @settings(max_examples=100)
    def test_all_nodes_reachable_from_source(self, graph: ExecutionGraph) -> None:
        """Property: All nodes are reachable from source."""
        source = graph.get_source()
        assert source is not None

        nx_graph = graph.get_nx_graph()
        # Cast to set[NodeID] since nx.descendants returns set[str] but our graph uses NodeID
        reachable: set[NodeID] = {NodeID(n) for n in nx.descendants(nx_graph, source)} | {source}

        all_nodes = {info.node_id for info in graph.get_nodes()}
        assert reachable == all_nodes, f"Not all nodes reachable from source. Unreachable: {all_nodes - reachable}"

    @given(graph=linear_pipelines())
    @settings(max_examples=100)
    def test_sinks_have_no_outgoing_edges(self, graph: ExecutionGraph) -> None:
        """Property: Sink nodes have no outgoing edges."""
        nx_graph = graph.get_nx_graph()
        sinks = graph.get_sinks()

        for sink in sinks:
            out_degree = nx_graph.out_degree(sink)
            assert out_degree == 0, f"Sink {sink} has {out_degree} outgoing edges"

    @given(graph=linear_pipelines())
    @settings(max_examples=100)
    def test_source_has_no_incoming_edges(self, graph: ExecutionGraph) -> None:
        """Property: Source node has no incoming edges."""
        source = graph.get_source()
        assert source is not None

        nx_graph = graph.get_nx_graph()
        in_degree = nx_graph.in_degree(source)
        assert in_degree == 0, f"Source has {in_degree} incoming edges"


# =============================================================================
# Schema Contract (Guaranteed Fields) Property Tests
# =============================================================================


# Strategies for generating field sets
# Use ASCII letters only - schema validation requires valid Python identifiers
# that can be expressed in config files (stricter than Python's isidentifier())
ascii_field_names = st.text(
    min_size=1,
    max_size=10,
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
)

field_sets = st.frozensets(ascii_field_names, min_size=0, max_size=5)


@st.composite
def disjoint_field_sets(draw: st.DrawFn) -> tuple[frozenset[str], frozenset[str]]:
    """Generate two sets where the second contains fields NOT in the first."""
    guaranteed = draw(field_sets)
    # Generate additional fields not in guaranteed
    extra_fields = draw(
        st.frozensets(ascii_field_names, min_size=1, max_size=3).filter(
            lambda s: not s & guaranteed  # Ensure no overlap
        )
    )
    return guaranteed, extra_fields


@st.composite
def superset_field_sets(draw: st.DrawFn) -> tuple[frozenset[str], frozenset[str]]:
    """Generate (guaranteed, required) where guaranteed >= required."""
    # Use ASCII-only field names for schema validation compatibility
    required = draw(st.frozensets(ascii_field_names, min_size=1, max_size=3))
    # Guaranteed includes required plus possibly more
    extra = draw(field_sets)
    guaranteed = required | extra
    return guaranteed, required


class TestGuaranteedFieldsProperties:
    """Property tests for schema contract validation (guaranteed_fields, required_fields).

    NOTE: Schema config requires "fields" key. For dynamic schemas with guarantees, use:
    {"schema": {"mode": "observed", "guaranteed_fields": [...]}}
    """

    @given(sets=superset_field_sets())
    @settings(max_examples=100)
    def test_superset_guarantees_satisfy_requirements(self, sets: tuple[frozenset[str], frozenset[str]]) -> None:
        """Property: If guaranteed >= required, validation passes.

        This is the fundamental contract: producers must guarantee at least
        what consumers require.
        """
        guaranteed, required = sets

        graph = ExecutionGraph()

        # Source with guaranteed fields (dynamic schema with guarantees)
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="test_source",
            config={"schema": {"mode": "observed", "guaranteed_fields": list(guaranteed)}},
        )

        # Transform with required fields
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="test_transform",
            config={"required_input_fields": list(required)},
        )

        # Sink
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")

        # Connect
        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")

        # Schema contract validation should pass - guaranteed includes all required
        graph.validate_edge_compatibility()  # Should not raise

    @given(sets=disjoint_field_sets())
    @settings(max_examples=100)
    def test_missing_fields_detected(self, sets: tuple[frozenset[str], frozenset[str]]) -> None:
        """Property: If required - guaranteed != empty, validation fails with correct missing set.

        When a consumer requires fields the producer doesn't guarantee,
        validation must fail and identify exactly which fields are missing.
        """
        guaranteed, extra_required = sets
        required = guaranteed | extra_required  # required has fields not in guaranteed

        graph = ExecutionGraph()

        # Source with limited guarantees (or no guarantees if empty)
        if guaranteed:
            source_config = {"schema": {"mode": "observed", "guaranteed_fields": list(guaranteed)}}
        else:
            source_config = {}  # No schema config = no guarantees
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="test_source",
            config=source_config,
        )

        # Transform requires more than guaranteed
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="test_transform",
            config={"required_input_fields": list(required)},
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")

        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")

        # Schema contract validation should fail
        import pytest

        with pytest.raises(ValueError, match="Schema contract violation"):
            graph.validate_edge_compatibility()

    @given(
        common_fields=st.frozensets(
            st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,7}", fullmatch=True),
            min_size=1,
            max_size=3,
        ),
        branch_a_only=st.frozensets(
            st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,7}", fullmatch=True),
            min_size=0,
            max_size=2,
        ),
        branch_b_only=st.frozensets(
            st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,7}", fullmatch=True),
            min_size=0,
            max_size=2,
        ),
    )
    @settings(max_examples=100)
    def test_coalesce_intersection_property(
        self,
        common_fields: frozenset[str],
        branch_a_only: frozenset[str],
        branch_b_only: frozenset[str],
    ) -> None:
        """Property: Under best_effort, coalesce guaranteed = intersection of branch guarantees.

        When parallel branches rejoin at a coalesce node under best_effort
        policy, branches may be lost — only fields guaranteed by ALL branches
        remain guaranteed (the "lowest common denominator"). Under require_all
        the result would be the union (every branch always arrives) — see
        test_dag_coalesce_optionality.py::test_require_all_* for that path.
        """
        # Ensure disjoint branch-specific fields
        branch_a_only = branch_a_only - common_fields - branch_b_only
        branch_b_only = branch_b_only - common_fields - branch_a_only

        branch_a_guarantees = common_fields | branch_a_only
        branch_b_guarantees = common_fields | branch_b_only
        expected_after_coalesce = common_fields  # Only common fields survive intersection

        graph = ExecutionGraph()

        # Source
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source")

        # Two branches with different guarantees
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="test_transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": list(branch_a_guarantees)}},
        )
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="test_transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": list(branch_b_guarantees)}},
        )

        # Coalesce node (merge point) — best_effort uses intersection of guarantees
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce",
            ["branch_a", "branch_b"],
            policy="best_effort",
            extra_config={"branches": {"branch_a": {}, "branch_b": {}}},
        )

        # Sink
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")

        # Connect: source -> [branch_a, branch_b] -> coalesce -> sink
        graph.add_edge("source", "branch_a", label="path_a")
        graph.add_edge("source", "branch_b", label="path_b")
        graph.add_edge("branch_a", "coalesce", label="continue")
        graph.add_edge("branch_b", "coalesce", label="continue")
        graph.add_edge("coalesce", "sink", label="continue")

        # Get effective guarantees after coalesce
        effective = graph.get_effective_guaranteed_fields("coalesce")

        assert effective == expected_after_coalesce, (
            f"Coalesce should guarantee intersection: {expected_after_coalesce}, "
            f"got {effective}. "
            f"Branch A: {branch_a_guarantees}, Branch B: {branch_b_guarantees}"
        )

    @given(
        common_fields=st.frozensets(
            st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,7}", fullmatch=True),
            min_size=0,
            max_size=3,
        ),
        branch_a_only=st.frozensets(
            st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,7}", fullmatch=True),
            min_size=0,
            max_size=2,
        ),
        branch_b_only=st.frozensets(
            st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,7}", fullmatch=True),
            min_size=0,
            max_size=2,
        ),
    )
    @settings(max_examples=100)
    def test_coalesce_require_all_union_property(
        self,
        common_fields: frozenset[str],
        branch_a_only: frozenset[str],
        branch_b_only: frozenset[str],
    ) -> None:
        """Property: Under require_all, coalesce guaranteed = UNION of branch guarantees.

        When parallel branches rejoin under require_all, every branch always
        arrives (CoalesceExecutor._should_merge enforces this) and _merge_data's
        dict.update preserves every branch's keys. So a field guaranteed by
        ANY branch is guaranteed in the merged row — the merged spec is the
        union of branch guarantees, not the intersection.

        This is the symmetric counterpart to test_coalesce_intersection_property
        (which exercises best_effort).
        """
        # Ensure disjoint branch-specific fields
        branch_a_only = branch_a_only - common_fields - branch_b_only
        branch_b_only = branch_b_only - common_fields - branch_a_only

        # The union property is only observable when at least one branch has
        # exclusive fields. Without this, union == intersection == common_fields,
        # making the test degenerate. QA review identified this gap.
        assume(branch_a_only or branch_b_only)

        branch_a_guarantees = common_fields | branch_a_only
        branch_b_guarantees = common_fields | branch_b_only
        # Under require_all, every branch always arrives → union of guarantees.
        expected_after_coalesce = branch_a_guarantees | branch_b_guarantees

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source")
        graph.add_node(
            "branch_a",
            node_type=NodeType.TRANSFORM,
            plugin_name="test_transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": list(branch_a_guarantees)}},
        )
        graph.add_node(
            "branch_b",
            node_type=NodeType.TRANSFORM,
            plugin_name="test_transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": list(branch_b_guarantees)}},
        )

        # Coalesce node — require_all uses union of guarantees (every branch arrives)
        _add_coalesce_with_computed_schema(
            graph,
            "coalesce",
            ["branch_a", "branch_b"],
            policy="require_all",
            extra_config={"branches": {"branch_a": {}, "branch_b": {}}},
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")

        graph.add_edge("source", "branch_a", label="path_a")
        graph.add_edge("source", "branch_b", label="path_b")
        graph.add_edge("branch_a", "coalesce", label="continue")
        graph.add_edge("branch_b", "coalesce", label="continue")
        graph.add_edge("coalesce", "sink", label="continue")

        effective = graph.get_effective_guaranteed_fields("coalesce")

        # Both branches' guarantees survive under require_all (union semantics).
        # However: a branch with guaranteed_fields=None abstains and contributes
        # nothing. With both branches having explicit (possibly empty) guarantees,
        # we expect the union of both.
        assert effective == expected_after_coalesce, (
            f"Coalesce under require_all should guarantee union: "
            f"{expected_after_coalesce}, got {effective}. "
            f"Branch A: {branch_a_guarantees}, Branch B: {branch_b_guarantees}"
        )

    @given(guaranteed=field_sets.filter(lambda s: len(s) > 0))
    @settings(max_examples=50)
    def test_gate_passthrough_inheritance(self, guaranteed: frozenset[str]) -> None:
        """Property: Gates use schema propagated by builder from upstream.

        Gates don't transform data - they only route. In production, the
        builder copies the upstream schema to the gate. The effective
        guarantees after a gate should equal the propagated guarantees.
        """
        graph = ExecutionGraph()

        # Source with guarantees
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="test_source",
            config={"schema": {"mode": "observed", "guaranteed_fields": list(guaranteed)}},
        )

        # Gate (builder would propagate source schema here)
        graph.add_node(
            "gate",
            node_type=NodeType.GATE,
            plugin_name="test_gate",
            config={"schema": {"mode": "observed", "guaranteed_fields": list(guaranteed)}},
        )

        # Sink
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")

        # Connect
        graph.add_edge("source", "gate", label="continue")
        graph.add_edge("gate", "sink", label="continue")

        # Effective guarantees after gate should equal source guarantees
        effective = graph.get_effective_guaranteed_fields("gate")

        assert effective == guaranteed, f"Gate should inherit upstream guarantees: {guaranteed}, got {effective}"

    @given(
        g1=st.frozensets(ascii_field_names, min_size=1, max_size=3),
        g2=st.frozensets(ascii_field_names, min_size=1, max_size=3),
    )
    @settings(max_examples=50)
    def test_guarantees_propagate_through_chain(self, g1: frozenset[str], g2: frozenset[str]) -> None:
        """Property: Later nodes define their own guarantees for downstream.

        In a chain: source(g1) -> transform(g2) -> sink
        The transform's effective output is g2 (its declared guarantees),
        not a union with upstream.
        """
        graph = ExecutionGraph()

        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="test_source",
            config={"schema": {"mode": "observed", "guaranteed_fields": list(g1)}},
        )

        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="test_transform",
            config={"schema": {"mode": "observed", "guaranteed_fields": list(g2)}},
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")

        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")

        # Source's effective guarantees
        source_effective = graph.get_effective_guaranteed_fields("source")
        assert source_effective == g1

        # Transform declares its own guarantees - these become the effective output
        transform_effective = graph.get_effective_guaranteed_fields("transform")
        assert transform_effective == g2, f"Transform with own guarantees should use those: {g2}, got {transform_effective}"

    def test_empty_guarantees_fails_any_requirement(self) -> None:
        """Property: No guarantees + any requirement = validation failure.

        A producer with no declared guarantees cannot satisfy
        any consumer requirements.
        """
        graph = ExecutionGraph()

        # Source with no guarantees (no schema config)
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source")

        # Transform requires a field
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="test_transform",
            config={"required_input_fields": ["must_have_field"]},
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")

        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")

        import pytest

        with pytest.raises(ValueError, match="Schema contract violation"):
            graph.validate_edge_compatibility()

    def test_empty_requirements_always_satisfied(self) -> None:
        """Property: No requirements are always satisfied.

        A consumer with no declared requirements can accept
        any producer output.
        """
        graph = ExecutionGraph()

        # Source with arbitrary guarantees
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="test_source",
            config={"schema": {"mode": "observed", "guaranteed_fields": ["field_a", "field_b"]}},
        )

        # Transform with no requirements
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="test_transform",
        )

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test_sink")

        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")

        # Schema contract validation should pass - no requirements to violate
        graph.validate_edge_compatibility()  # Should not raise


# =============================================================================
# Coalesce Topology Property Tests (N-branch, field optionality)
# =============================================================================


# Strategies for field definitions with mixed required/optional
valid_field_types: st.SearchStrategy[_FieldType] = st.sampled_from(["int", "str", "float", "bool"])


@st.composite
def field_definitions(draw: st.DrawFn) -> FieldDefinition:
    """Generate a single FieldDefinition with random name, type, and required status."""
    name = draw(st.from_regex(r"[a-z][a-z0-9_]{0,7}", fullmatch=True))
    field_type = draw(valid_field_types)
    required = draw(st.booleans())
    return FieldDefinition(name=name, field_type=field_type, required=required)


@st.composite
def branch_schemas(
    draw: st.DrawFn,
    min_fields: int = 1,
    max_fields: int = 4,
) -> SchemaConfig:
    """Generate a SchemaConfig with random fields and required statuses."""
    num_fields = draw(st.integers(min_value=min_fields, max_value=max_fields))

    # Generate unique field names with types and required flags
    field_names = draw(
        st.lists(
            st.from_regex(r"[a-z][a-z0-9_]{0,5}", fullmatch=True),
            min_size=num_fields,
            max_size=num_fields,
            unique=True,
        )
    )
    fields = tuple(
        FieldDefinition(
            name=name,
            field_type=draw(valid_field_types),
            required=draw(st.booleans()),
        )
        for name in field_names
    )

    return SchemaConfig(mode="flexible", fields=fields)


@st.composite
def coalesce_branch_sets(
    draw: st.DrawFn,
    min_branches: int = 2,
    max_branches: int = 5,
) -> dict[str, SchemaConfig]:
    """Generate a set of branch schemas for a coalesce node."""
    num_branches = draw(st.integers(min_value=min_branches, max_value=max_branches))

    branches = {}
    for i in range(num_branches):
        branch_name = f"branch_{i}"
        # Each branch gets its own schema
        branches[branch_name] = draw(branch_schemas())

    return branches


class TestCoalesceNBranchProperties:
    """Property tests for coalesce behavior with varying branch counts (2-N).

    These tests verify invariants that must hold regardless of the number of
    branches feeding into a coalesce node. The RC5-UX change set introduced
    significant coalesce logic — these tests ensure the invariants hold under
    arbitrary branch counts, not just the 2-branch case covered by unit tests.
    """

    @given(branches=coalesce_branch_sets(min_branches=2, max_branches=5))
    @settings(max_examples=100)
    def test_best_effort_guarantees_are_intersection(self, branches: dict[str, SchemaConfig]) -> None:
        """Property: Under best_effort, merged guarantees = intersection of branch guarantees.

        For N branches, any branch might be lost, so only fields guaranteed by
        ALL branches survive in the merged output.
        """
        # Compute effective guarantees for each branch
        branch_guarantees = [schema.get_effective_guaranteed_fields() for schema in branches.values()]

        # Intersection: only fields guaranteed by ALL branches survive
        if branch_guarantees:
            expected = set.intersection(*[set(g) for g in branch_guarantees])
        else:
            expected = set()

        # Use production function
        result = merge_guaranteed_fields(branches, require_all=False)

        # Result should be intersection (or None if no branch has guarantees)
        if result is None:
            # None means no branch had effective guarantees
            assert all(not schema.has_effective_guarantees for schema in branches.values()), (
                "Result None but some branch has effective guarantees"
            )
        else:
            assert set(result) == expected, f"best_effort should produce intersection: expected {expected}, got {set(result)}"

    @given(branches=coalesce_branch_sets(min_branches=2, max_branches=5))
    @settings(max_examples=100)
    def test_require_all_guarantees_are_union(self, branches: dict[str, SchemaConfig]) -> None:
        """Property: Under require_all, merged guarantees = union of branch guarantees.

        When all branches always arrive, any field guaranteed by ANY branch
        is guaranteed in the merged output.
        """
        branch_guarantees = [schema.get_effective_guaranteed_fields() for schema in branches.values()]

        # Union: fields guaranteed by ANY branch survive
        if branch_guarantees:
            expected = set.union(*[set(g) for g in branch_guarantees])
        else:
            expected = set()

        result = merge_guaranteed_fields(branches, require_all=True)

        if result is None:
            assert all(not schema.has_effective_guarantees for schema in branches.values()), (
                "Result None but some branch has effective guarantees"
            )
        else:
            assert set(result) == expected, f"require_all should produce union: expected {expected}, got {set(result)}"

    @given(branches=coalesce_branch_sets(min_branches=2, max_branches=4))
    @settings(max_examples=100)
    def test_best_effort_guarantees_subset_of_any_branch(self, branches: dict[str, SchemaConfig]) -> None:
        """Property: Under best_effort, merged guarantees ⊆ each branch's guarantees.

        The intersection is always a subset of (or equal to) each input set.
        This is the "monotonicity" property — guarantees can only shrink.
        """
        result = merge_guaranteed_fields(branches, require_all=False)

        if result is not None:
            result_set = set(result)
            for branch_name, schema in branches.items():
                if schema.has_effective_guarantees:
                    branch_set = set(schema.get_effective_guaranteed_fields())
                    assert result_set <= branch_set, (
                        f"Merged guarantees {result_set} not subset of branch {branch_name}'s guarantees {branch_set}"
                    )

    @given(branches=coalesce_branch_sets(min_branches=2, max_branches=4))
    @settings(max_examples=100)
    def test_require_all_guarantees_superset_of_any_branch(self, branches: dict[str, SchemaConfig]) -> None:
        """Property: Under require_all, merged guarantees ⊇ each branch's guarantees.

        The union is always a superset of (or equal to) each input set.
        """
        result = merge_guaranteed_fields(branches, require_all=True)

        if result is not None:
            result_set = set(result)
            for branch_name, schema in branches.items():
                if schema.has_effective_guarantees:
                    branch_set = set(schema.get_effective_guaranteed_fields())
                    assert result_set >= branch_set, (
                        f"Merged guarantees {result_set} not superset of branch {branch_name}'s guarantees {branch_set}"
                    )


def _has_type_conflicts(branches: dict[str, SchemaConfig]) -> bool:
    """Check if branches have fields with same name but different types."""
    field_types: dict[str, str] = {}
    for schema in branches.values():
        if schema.fields is None:
            continue
        for fd in schema.fields:
            if fd.name in field_types:
                if field_types[fd.name] != fd.field_type:
                    return True
            else:
                field_types[fd.name] = fd.field_type
    return False


class TestCoalesceFieldOptionalityProperties:
    """Property tests for field required/optional semantics in union merge.

    These tests verify the OR/AND semantics for the `required` flag when
    merging typed fields from multiple branches.

    Note: These tests filter out type conflicts (same field name, different
    types) since that's a separate invariant tested elsewhere.
    """

    @given(branches=coalesce_branch_sets(min_branches=2, max_branches=4))
    @settings(max_examples=100)
    def test_best_effort_shared_field_required_only_if_all_require(self, branches: dict[str, SchemaConfig]) -> None:
        """Property: Under best_effort, shared field is required only if ALL branches require it.

        AND semantics: a field must be required in every contributing branch
        to be required in the merged output. "Shared" means present in ALL
        contributing branches (branches with fields defined).
        """
        # Skip examples with type conflicts — those raise GraphValidationError
        assume(not _has_type_conflicts(branches))

        merged = merge_union_fields(branches, require_all=False)

        if merged.fields is None:
            return  # Observed mode, no fields to check

        # Identify contributing branches (those with fields defined)
        contributing_branches = [name for name, schema in branches.items() if schema.fields is not None]
        num_contributing = len(contributing_branches)

        if num_contributing == 0:
            return  # No contributing branches

        # Build map of field names to merged required status
        merged_fields = {fd.name: fd.required for fd in merged.fields}

        # For each merged field, check AND semantics for truly shared fields
        for field_name, merged_required in merged_fields.items():
            # Count how many branches have this field and their required statuses
            branch_statuses = []
            for branch_name in contributing_branches:
                schema = branches[branch_name]
                if schema.fields is not None:
                    for fd in schema.fields:
                        if fd.name == field_name:
                            branch_statuses.append(fd.required)
                            break

            # Only check AND semantics for fields in ALL contributing branches
            if len(branch_statuses) == num_contributing:
                # Truly shared field — AND semantics: required only if ALL require
                expected = all(branch_statuses)
                assert merged_required == expected, (
                    f"Field {field_name}: expected required={expected} (AND of {branch_statuses}), got required={merged_required}"
                )

    @given(branches=coalesce_branch_sets(min_branches=2, max_branches=4))
    @settings(max_examples=100)
    def test_require_all_shared_field_required_if_any_requires(self, branches: dict[str, SchemaConfig]) -> None:
        """Property: Under require_all, shared field is required if ANY branch requires it.

        OR semantics: a field is required in the merged output if at least
        one branch requires it. "Shared" means present in ALL contributing branches.
        """
        assume(not _has_type_conflicts(branches))

        merged = merge_union_fields(branches, require_all=True)

        if merged.fields is None:
            return

        # Identify contributing branches
        contributing_branches = [name for name, schema in branches.items() if schema.fields is not None]
        num_contributing = len(contributing_branches)

        if num_contributing == 0:
            return

        merged_fields = {fd.name: fd.required for fd in merged.fields}

        for field_name, merged_required in merged_fields.items():
            branch_statuses = []
            for branch_name in contributing_branches:
                schema = branches[branch_name]
                if schema.fields is not None:
                    for fd in schema.fields:
                        if fd.name == field_name:
                            branch_statuses.append(fd.required)
                            break

            # Only check OR semantics for fields in ALL contributing branches
            if len(branch_statuses) == num_contributing:
                # Truly shared field — OR semantics: required if ANY requires
                expected = any(branch_statuses)
                assert merged_required == expected, (
                    f"Field {field_name}: expected required={expected} (OR of {branch_statuses}), got required={merged_required}"
                )

    @given(branches=coalesce_branch_sets(min_branches=2, max_branches=4))
    @settings(max_examples=100)
    def test_best_effort_branch_exclusive_field_always_optional(self, branches: dict[str, SchemaConfig]) -> None:
        """Property: Under best_effort, branch-exclusive fields are always optional.

        A field that exists in only one branch cannot be guaranteed in the
        merged output — the branch that has it might not arrive.
        """
        assume(not _has_type_conflicts(branches))

        merged = merge_union_fields(branches, require_all=False)

        if merged.fields is None:
            return

        # Identify branch-exclusive fields
        field_to_branches: dict[str, list[str]] = {}
        for branch_name, schema in branches.items():
            if schema.fields is not None:
                for fd in schema.fields:
                    if fd.name not in field_to_branches:
                        field_to_branches[fd.name] = []
                    field_to_branches[fd.name].append(branch_name)

        exclusive_fields = {name for name, branch_list in field_to_branches.items() if len(branch_list) == 1}

        for fd in merged.fields:
            if fd.name in exclusive_fields:
                assert fd.required is False, (
                    f"Branch-exclusive field {fd.name} should be optional under best_effort, but got required={fd.required}"
                )

    @given(branches=coalesce_branch_sets(min_branches=2, max_branches=4))
    @settings(max_examples=100)
    def test_merged_fields_superset_of_all_branch_fields(self, branches: dict[str, SchemaConfig]) -> None:
        """Property: Merged schema contains all fields from all branches (union merge).

        Under union merge strategy, the merged schema includes every field
        that appears in any branch (unlike select merge which picks one).
        """
        assume(not _has_type_conflicts(branches))

        merged = merge_union_fields(branches, require_all=False)

        # Collect all field names from all branches
        all_field_names: set[str] = set()
        for schema in branches.values():
            if schema.fields is not None:
                for fd in schema.fields:
                    all_field_names.add(fd.name)

        if not all_field_names:
            # All branches are observed mode
            assert merged.fields is None
            return

        assert merged.fields is not None, "Expected merged fields when branches have fields"
        merged_names = {fd.name for fd in merged.fields}

        assert merged_names == all_field_names, f"Merged should contain all branch fields: expected {all_field_names}, got {merged_names}"
