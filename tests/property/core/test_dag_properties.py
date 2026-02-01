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

import networkx as nx
from hypothesis import given, settings
from hypothesis import strategies as st

from elspeth.core.dag import ExecutionGraph, GraphValidationError

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
    graph.add_node("source", node_type="source", plugin_name="test_source")

    # Add transforms
    prev_node = "source"
    for i in range(num_transforms):
        node_id = f"transform_{i}"
        graph.add_node(node_id, node_type="transform", plugin_name="test_transform")
        graph.add_edge(prev_node, node_id, label="continue")
        prev_node = node_id

    # Add sink
    graph.add_node("sink", node_type="sink", plugin_name="test_sink")
    graph.add_edge(prev_node, "sink", label="continue")

    return graph


@st.composite
def diamond_pipelines(draw: st.DrawFn) -> ExecutionGraph:
    """Generate a diamond-shaped pipeline: source -> [A, B] -> merge -> sink."""
    # Draw is required by @st.composite but not used for deterministic structure
    _ = draw(st.just(None))
    graph = ExecutionGraph()

    # Source
    graph.add_node("source", node_type="source", plugin_name="test_source")

    # Two parallel transforms
    graph.add_node("branch_a", node_type="transform", plugin_name="test_transform")
    graph.add_node("branch_b", node_type="transform", plugin_name="test_transform")
    graph.add_edge("source", "branch_a", label="path_a")
    graph.add_edge("source", "branch_b", label="path_b")

    # Merge point (simplified - in real ELSPETH this would be a coalesce)
    graph.add_node("merge", node_type="transform", plugin_name="test_transform")
    graph.add_edge("branch_a", "merge", label="continue")
    graph.add_edge("branch_b", "merge", label="continue")

    # Sink
    graph.add_node("sink", node_type="sink", plugin_name="test_sink")
    graph.add_edge("merge", "sink", label="continue")

    return graph


@st.composite
def multi_sink_pipelines(draw: st.DrawFn, num_sinks: int = 2) -> ExecutionGraph:
    """Generate a pipeline with multiple sinks."""
    # Draw is required by @st.composite but not used for deterministic structure
    _ = draw(st.just(None))
    graph = ExecutionGraph()

    # Source
    graph.add_node("source", node_type="source", plugin_name="test_source")

    # Transform
    graph.add_node("transform", node_type="transform", plugin_name="test_transform")
    graph.add_edge("source", "transform", label="continue")

    # Multiple sinks with unique edge labels
    for i in range(num_sinks):
        sink_id = f"sink_{i}"
        graph.add_node(sink_id, node_type="sink", plugin_name="test_sink")
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

        # Source before branches
        assert index_map["source"] < index_map["branch_a"]
        assert index_map["source"] < index_map["branch_b"]

        # Branches before merge
        assert index_map["branch_a"] < index_map["merge"]
        assert index_map["branch_b"] < index_map["merge"]

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

    @given(graph=multi_sink_pipelines(num_sinks=3))
    @settings(max_examples=50)
    def test_validate_accepts_multiple_sinks(self, graph: ExecutionGraph) -> None:
        """Property: Graphs with multiple sinks are valid."""
        graph.validate()  # Should not raise
        sinks = graph.get_sinks()
        assert len(sinks) == 3, f"Expected 3 sinks, got {len(sinks)}"


class TestValidationFailureProperties:
    """Property tests for validation failure cases."""

    def test_validate_rejects_cycle(self) -> None:
        """Property: Cyclic graphs are rejected."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="test")
        graph.add_node("a", node_type="transform", plugin_name="test")
        graph.add_node("b", node_type="transform", plugin_name="test")
        graph.add_node("sink", node_type="sink", plugin_name="test")

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
        graph.add_node("transform", node_type="transform", plugin_name="test")
        graph.add_node("sink", node_type="sink", plugin_name="test")
        graph.add_edge("transform", "sink", label="continue")

        import pytest

        with pytest.raises(GraphValidationError, match="source"):
            graph.validate()

    def test_validate_rejects_multiple_sources(self) -> None:
        """Property: Graphs with multiple sources are rejected."""
        graph = ExecutionGraph()
        graph.add_node("source1", node_type="source", plugin_name="test")
        graph.add_node("source2", node_type="source", plugin_name="test")
        graph.add_node("sink", node_type="sink", plugin_name="test")
        graph.add_edge("source1", "sink", label="path1")
        graph.add_edge("source2", "sink", label="path2")

        import pytest

        with pytest.raises(GraphValidationError, match="exactly one source"):
            graph.validate()

    def test_validate_rejects_no_sink(self) -> None:
        """Property: Graphs without sink are rejected."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="test")
        graph.add_node("transform", node_type="transform", plugin_name="test")
        graph.add_edge("source", "transform", label="continue")

        import pytest

        with pytest.raises(GraphValidationError, match="sink"):
            graph.validate()

    def test_validate_rejects_duplicate_edge_labels(self) -> None:
        """Property: Duplicate edge labels from same node are rejected."""
        graph = ExecutionGraph()
        graph.add_node("source", node_type="source", plugin_name="test")
        graph.add_node("sink1", node_type="sink", plugin_name="test")
        graph.add_node("sink2", node_type="sink", plugin_name="test")

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
        graph.add_node("source", node_type="source", plugin_name="test")
        graph.add_node("a", node_type="transform", plugin_name="test")
        graph.add_node("b", node_type="transform", plugin_name="test")
        graph.add_node("sink", node_type="sink", plugin_name="test")

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
        reachable = set(nx.descendants(nx_graph, source)) | {source}

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
    """Generate (guaranteed, required) where guaranteed ⊇ required."""
    # Use ASCII-only field names for schema validation compatibility
    required = draw(st.frozensets(ascii_field_names, min_size=1, max_size=3))
    # Guaranteed includes required plus possibly more
    extra = draw(field_sets)
    guaranteed = required | extra
    return guaranteed, required


class TestGuaranteedFieldsProperties:
    """Property tests for schema contract validation (guaranteed_fields, required_fields).

    NOTE: Schema config requires "fields" key. For dynamic schemas with guarantees, use:
    {"schema": {"fields": "dynamic", "guaranteed_fields": [...]}}
    """

    @given(sets=superset_field_sets())
    @settings(max_examples=100)
    def test_superset_guarantees_satisfy_requirements(self, sets: tuple[frozenset[str], frozenset[str]]) -> None:
        """Property: If guaranteed ⊇ required, validation passes.

        This is the fundamental contract: producers must guarantee at least
        what consumers require.
        """
        guaranteed, required = sets

        graph = ExecutionGraph()

        # Source with guaranteed fields (dynamic schema with guarantees)
        graph.add_node(
            "source",
            node_type="source",
            plugin_name="test_source",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": list(guaranteed)}},
        )

        # Transform with required fields
        graph.add_node(
            "transform",
            node_type="transform",
            plugin_name="test_transform",
            config={"required_input_fields": list(required)},
        )

        # Sink
        graph.add_node("sink", node_type="sink", plugin_name="test_sink")

        # Connect
        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")

        # Schema contract validation should pass - guaranteed includes all required
        graph.validate_edge_compatibility()  # Should not raise

    @given(sets=disjoint_field_sets())
    @settings(max_examples=100)
    def test_missing_fields_detected(self, sets: tuple[frozenset[str], frozenset[str]]) -> None:
        """Property: If required - guaranteed ≠ ∅, validation fails with correct missing set.

        When a consumer requires fields the producer doesn't guarantee,
        validation must fail and identify exactly which fields are missing.
        """
        guaranteed, extra_required = sets
        required = guaranteed | extra_required  # required has fields not in guaranteed

        graph = ExecutionGraph()

        # Source with limited guarantees (or no guarantees if empty)
        if guaranteed:
            source_config = {"schema": {"fields": "dynamic", "guaranteed_fields": list(guaranteed)}}
        else:
            source_config = {}  # No schema config = no guarantees
        graph.add_node(
            "source",
            node_type="source",
            plugin_name="test_source",
            config=source_config,
        )

        # Transform requires more than guaranteed
        graph.add_node(
            "transform",
            node_type="transform",
            plugin_name="test_transform",
            config={"required_input_fields": list(required)},
        )

        graph.add_node("sink", node_type="sink", plugin_name="test_sink")

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
        """Property: After coalesce, guaranteed = intersection of branch guarantees.

        When parallel branches rejoin at a coalesce node, only fields
        guaranteed by ALL branches remain guaranteed. This is the
        "lowest common denominator" of field availability.
        """
        # Ensure disjoint branch-specific fields
        branch_a_only = branch_a_only - common_fields - branch_b_only
        branch_b_only = branch_b_only - common_fields - branch_a_only

        branch_a_guarantees = common_fields | branch_a_only
        branch_b_guarantees = common_fields | branch_b_only
        expected_after_coalesce = common_fields  # Only common fields survive

        graph = ExecutionGraph()

        # Source
        graph.add_node("source", node_type="source", plugin_name="test_source")

        # Two branches with different guarantees
        graph.add_node(
            "branch_a",
            node_type="transform",
            plugin_name="test_transform",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": list(branch_a_guarantees)}},
        )
        graph.add_node(
            "branch_b",
            node_type="transform",
            plugin_name="test_transform",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": list(branch_b_guarantees)}},
        )

        # Coalesce node (merge point)
        graph.add_node("coalesce", node_type="coalesce", plugin_name="test_coalesce")

        # Sink
        graph.add_node("sink", node_type="sink", plugin_name="test_sink")

        # Connect: source -> [branch_a, branch_b] -> coalesce -> sink
        graph.add_edge("source", "branch_a", label="path_a")
        graph.add_edge("source", "branch_b", label="path_b")
        graph.add_edge("branch_a", "coalesce", label="continue")
        graph.add_edge("branch_b", "coalesce", label="continue")
        graph.add_edge("coalesce", "sink", label="continue")

        # Get effective guarantees after coalesce
        effective = graph._get_effective_guaranteed_fields("coalesce")

        assert effective == expected_after_coalesce, (
            f"Coalesce should guarantee intersection: {expected_after_coalesce}, "
            f"got {effective}. "
            f"Branch A: {branch_a_guarantees}, Branch B: {branch_b_guarantees}"
        )

    @given(guaranteed=field_sets.filter(lambda s: len(s) > 0))
    @settings(max_examples=50)
    def test_gate_passthrough_inheritance(self, guaranteed: frozenset[str]) -> None:
        """Property: Gates inherit guarantees from upstream (pass-through).

        Gates don't transform data - they only route. The effective
        guarantees after a gate should equal the upstream guarantees.
        """
        graph = ExecutionGraph()

        # Source with guarantees
        graph.add_node(
            "source",
            node_type="source",
            plugin_name="test_source",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": list(guaranteed)}},
        )

        # Gate (pass-through)
        graph.add_node("gate", node_type="gate", plugin_name="test_gate")

        # Sink
        graph.add_node("sink", node_type="sink", plugin_name="test_sink")

        # Connect
        graph.add_edge("source", "gate", label="continue")
        graph.add_edge("gate", "sink", label="continue")

        # Effective guarantees after gate should equal source guarantees
        effective = graph._get_effective_guaranteed_fields("gate")

        assert effective == guaranteed, f"Gate should inherit upstream guarantees: {guaranteed}, got {effective}"

    @given(
        g1=st.frozensets(
            st.text(min_size=1, max_size=6, alphabet=st.characters(whitelist_categories=("L",))),
            min_size=1,
            max_size=3,
        ),
        g2=st.frozensets(
            st.text(min_size=1, max_size=6, alphabet=st.characters(whitelist_categories=("L",))),
            min_size=1,
            max_size=3,
        ),
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
            node_type="source",
            plugin_name="test_source",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": list(g1)}},
        )

        graph.add_node(
            "transform",
            node_type="transform",
            plugin_name="test_transform",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": list(g2)}},
        )

        graph.add_node("sink", node_type="sink", plugin_name="test_sink")

        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")

        # Source's effective guarantees
        source_effective = graph._get_effective_guaranteed_fields("source")
        assert source_effective == g1

        # Transform declares its own guarantees - these become the effective output
        transform_effective = graph._get_effective_guaranteed_fields("transform")
        assert transform_effective == g2, f"Transform with own guarantees should use those: {g2}, got {transform_effective}"

    def test_empty_guarantees_fails_any_requirement(self) -> None:
        """Property: No guarantees + any requirement = validation failure.

        A producer with no declared guarantees cannot satisfy
        any consumer requirements.
        """
        graph = ExecutionGraph()

        # Source with no guarantees (no schema config)
        graph.add_node("source", node_type="source", plugin_name="test_source")

        # Transform requires a field
        graph.add_node(
            "transform",
            node_type="transform",
            plugin_name="test_transform",
            config={"required_input_fields": ["must_have_field"]},
        )

        graph.add_node("sink", node_type="sink", plugin_name="test_sink")

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
            node_type="source",
            plugin_name="test_source",
            config={"schema": {"fields": "dynamic", "guaranteed_fields": ["field_a", "field_b"]}},
        )

        # Transform with no requirements
        graph.add_node(
            "transform",
            node_type="transform",
            plugin_name="test_transform",
        )

        graph.add_node("sink", node_type="sink", plugin_name="test_sink")

        graph.add_edge("source", "transform", label="continue")
        graph.add_edge("transform", "sink", label="continue")

        # Schema contract validation should pass - no requirements to violate
        graph.validate_edge_compatibility()  # Should not raise
