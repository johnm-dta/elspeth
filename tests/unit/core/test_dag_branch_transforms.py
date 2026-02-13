# tests/unit/core/test_dag_branch_transforms.py
"""Unit tests for per-branch transforms (ARCH-15).

Tests DAG builder wiring and graph metadata for fork→transform→coalesce
topologies. These use make_graph_fork() (manual construction) which is
appropriate for unit tests — integration tests must use from_plugin_instances().

Test strategy:
- Builder wiring: verify correct edge topology for identity vs transform branches
- Schema validation: strategy-aware validation (union/nested/select)
- get_branch_first_nodes(): verify correct first-node resolution per branch
"""

from __future__ import annotations

import pytest

from elspeth.contracts.enums import NodeType
from elspeth.contracts.types import NodeID
from elspeth.core.dag import ExecutionGraph
from tests.fixtures.factories import make_graph_fork

# =============================================================================
# Builder: Fork with per-branch transforms (edge topology)
# =============================================================================


class TestBuilderBranchTransforms:
    """Verify edge topology for fork/coalesce with per-branch transforms."""

    def test_fork_with_single_transform_branch(self) -> None:
        """Fork → single transform → coalesce should wire correctly.

        Both branches have a single transform. The graph should have:
        - gate → transform edges (labelled with branch name)
        - transform → coalesce edges (labelled "continue")
        """
        graph = make_graph_fork(
            {
                "path_a": ["transform_a"],
                "path_b": ["transform_b"],
            }
        )

        edges = graph.get_edges()
        edge_set = {(e.from_node, e.to_node, e.label) for e in edges}

        # Gate → transforms
        assert (NodeID("gate-node"), NodeID("transform_a"), "path_a") in edge_set
        assert (NodeID("gate-node"), NodeID("transform_b"), "path_b") in edge_set

        # Transforms → coalesce
        assert (NodeID("transform_a"), NodeID("coalesce-node"), "continue") in edge_set
        assert (NodeID("transform_b"), NodeID("coalesce-node"), "continue") in edge_set

        # Coalesce → sink
        assert (NodeID("coalesce-node"), NodeID("sink-node"), "continue") in edge_set

    def test_fork_with_chained_transforms(self) -> None:
        """Fork → T1 → T2 → coalesce chain on a branch.

        Chained transforms on a single branch should create a sequential
        MOVE-edge chain: gate → T1, T1 → T2, T2 → coalesce.
        """
        graph = make_graph_fork(
            {
                "path_a": ["t_a1", "t_a2"],
                "path_b": ["t_b1"],
            }
        )

        edges = graph.get_edges()
        edge_set = {(e.from_node, e.to_node, e.label) for e in edges}

        # Path A: gate → t_a1 → t_a2 → coalesce
        assert (NodeID("gate-node"), NodeID("t_a1"), "path_a") in edge_set
        assert (NodeID("t_a1"), NodeID("t_a2"), "continue") in edge_set
        assert (NodeID("t_a2"), NodeID("coalesce-node"), "continue") in edge_set

        # Path B: gate → t_b1 → coalesce
        assert (NodeID("gate-node"), NodeID("t_b1"), "path_b") in edge_set
        assert (NodeID("t_b1"), NodeID("coalesce-node"), "continue") in edge_set

    def test_fork_mixed_identity_and_transform(self) -> None:
        """Mixed: one branch direct (empty list), one through transforms.

        An empty transform list means the branch is identity — gate routes
        directly to coalesce. The make_graph_fork factory uses a "continue"
        edge label for this case (the full builder uses COPY edges with
        branch names — tested via from_plugin_instances in integration tests).
        """
        graph = make_graph_fork(
            {
                "path_a": [],  # identity branch
                "path_b": ["enrich"],  # transform branch
            }
        )

        edges = graph.get_edges()
        edge_set = {(e.from_node, e.to_node, e.label) for e in edges}

        # Identity branch: gate → coalesce directly (factory uses "continue" label)
        assert (NodeID("gate-node"), NodeID("coalesce-node"), "continue") in edge_set

        # Transform branch: gate → enrich → coalesce
        assert (NodeID("gate-node"), NodeID("enrich"), "path_b") in edge_set
        assert (NodeID("enrich"), NodeID("coalesce-node"), "continue") in edge_set

    def test_fork_branch_connection_namespace_collision(self) -> None:
        """Duplicate node names across branches cause graph corruption.

        NetworkX silently overwrites duplicate node_ids, so make_graph_fork
        with a shared node name across branches produces a degenerate graph
        where one branch's transform is silently merged with the other.
        The full builder catches this via the producer registry.

        This test verifies that the degenerate graph has fewer transform
        nodes than expected — confirming the collision happens.
        """
        graph = make_graph_fork(
            {
                "path_a": ["shared_node"],
                "path_b": ["shared_node"],  # Same name overwrites path_a's node
            }
        )

        # NetworkX silently overwrites: we expect only 1 transform, not 2
        transform_nodes = [info for info in graph.get_nodes() if info.node_type == NodeType.TRANSFORM]
        assert len(transform_nodes) == 1, "Expected 1 transform (collision merged two into one)"


# =============================================================================
# Schema validation: strategy-aware
# =============================================================================


class TestSchemaValidationByStrategy:
    """Schema validation behaviour depends on coalesce merge strategy.

    - union: requires compatible types on shared fields
    - nested: no cross-branch constraint
    - select: no cross-branch constraint
    """

    def test_union_schema_validation_all_pairs(self) -> None:
        """Union merge rejects incompatible types between branches.

        When two branches produce different types for the same field name,
        union merge should detect the conflict.
        """
        from elspeth.contracts import PluginSchema, RoutingMode

        class BranchASchema(PluginSchema):
            shared_field: int
            only_a: str

        class BranchBSchema(PluginSchema):
            shared_field: str  # INCOMPATIBLE: int vs str
            only_b: float

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=BranchASchema)
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork")
        graph.add_node("t_a", node_type=NodeType.TRANSFORM, plugin_name="a", input_schema=BranchASchema, output_schema=BranchASchema)
        graph.add_node("t_b", node_type=NodeType.TRANSFORM, plugin_name="b", input_schema=BranchASchema, output_schema=BranchBSchema)
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={"branches": {"path_a": "path_a", "path_b": "path_b"}, "policy": "require_all", "merge": "union"},
        )
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv")

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_a", label="path_a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "t_b", label="path_b", mode=RoutingMode.COPY)
        graph.add_edge("t_a", "coalesce", label="path_a", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="path_b", mode=RoutingMode.MOVE)
        graph.add_edge("coalesce", "sink", label="continue", mode=RoutingMode.MOVE)

        with pytest.raises(ValueError):
            graph.validate_edge_compatibility()

    def test_nested_schema_reflects_branch_structure(self) -> None:
        """Nested merge with different branch schemas passes coalesce validation.

        _validate_coalesce_compatibility is strategy-aware: nested merge
        skips cross-branch schema checks because each branch's data is
        keyed separately in the output.

        The coalesce node is left terminal (no downstream edge) to test the
        coalesce validation in isolation — the full builder path tests cover
        end-to-end schema propagation.
        """
        from elspeth.contracts import PluginSchema, RoutingMode

        class BranchASchema(PluginSchema):
            score: float

        class BranchBSchema(PluginSchema):
            rank: int

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=BranchASchema)
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork")
        graph.add_node("t_a", node_type=NodeType.TRANSFORM, plugin_name="a", input_schema=BranchASchema, output_schema=BranchASchema)
        graph.add_node("t_b", node_type=NodeType.TRANSFORM, plugin_name="b", input_schema=BranchASchema, output_schema=BranchBSchema)
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={
                "branches": {"branch_a": "branch_a", "branch_b": "branch_b"},
                "policy": "require_all",
                "merge": "nested",
            },
        )

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_a", label="branch_a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "t_b", label="branch_b", mode=RoutingMode.COPY)
        graph.add_edge("t_a", "coalesce", label="branch_a", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="branch_b", mode=RoutingMode.MOVE)

        # Nested merge should NOT reject different schemas —
        # _validate_coalesce_compatibility returns early for nested strategy
        graph.validate_edge_compatibility()

    def test_select_schema_uses_correct_branch(self) -> None:
        """Select merge should not reject schema differences.

        Select takes a single branch's output — other branches' schemas
        are irrelevant. _validate_coalesce_compatibility returns early for
        select strategy.

        The coalesce node is left terminal (no downstream edge) to test
        coalesce validation in isolation.
        """
        from elspeth.contracts import PluginSchema, RoutingMode

        class BranchASchema(PluginSchema):
            score: float

        class BranchBSchema(PluginSchema):
            rank: int  # Completely different from BranchA

        graph = ExecutionGraph()
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="csv", output_schema=BranchASchema)
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="fork")
        graph.add_node("t_a", node_type=NodeType.TRANSFORM, plugin_name="a", input_schema=BranchASchema, output_schema=BranchASchema)
        graph.add_node("t_b", node_type=NodeType.TRANSFORM, plugin_name="b", input_schema=BranchASchema, output_schema=BranchBSchema)
        graph.add_node(
            "coalesce",
            node_type=NodeType.COALESCE,
            plugin_name="coalesce:merge",
            config={
                "branches": {"branch_a": "branch_a", "branch_b": "branch_b"},
                "policy": "require_all",
                "merge": "select",
                "select_branch": "branch_a",
            },
        )

        graph.add_edge("source", "gate", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("gate", "t_a", label="branch_a", mode=RoutingMode.COPY)
        graph.add_edge("gate", "t_b", label="branch_b", mode=RoutingMode.COPY)
        graph.add_edge("t_a", "coalesce", label="branch_a", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="branch_b", mode=RoutingMode.MOVE)

        # Select merge should NOT reject different schemas
        graph.validate_edge_compatibility()


# =============================================================================
# get_branch_first_nodes(): first-node resolution per branch
# =============================================================================


class TestGetBranchFirstNodes:
    """Tests for ExecutionGraph.get_branch_first_nodes().

    This method returns {branch_name: first_processing_node_id} for every
    branch in the coalesce configuration. Identity branches map to the
    coalesce node; transform branches map to the first transform.

    These tests manually construct graphs with correct edge modes (COPY/MOVE)
    to match what the full builder produces, since make_graph_fork uses
    simplified edge labelling that doesn't match the builder's output.
    """

    @staticmethod
    def _build_graph_with_modes(
        branches: dict[str, list[str]],
    ) -> ExecutionGraph:
        """Build a fork/join graph with correct COPY/MOVE edge modes.

        Identity branches (empty list): gate --[COPY, branch_name]--> coalesce
        Transform branches: gate --[MOVE, branch_name]--> T1 --[MOVE, continue]--> ... --> coalesce
        """
        from elspeth.contracts import RoutingMode
        from elspeth.contracts.types import BranchName, CoalesceName

        graph = ExecutionGraph()
        gate = "gate-node"
        coalesce = "coalesce-node"

        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test-source", config={})
        graph.add_node(gate, node_type=NodeType.GATE, plugin_name="test-gate", config={})
        graph.add_node(coalesce, node_type=NodeType.COALESCE, plugin_name="coalesce", config={})

        graph.add_edge("source", gate, label="continue")

        for branch_name, transforms in branches.items():
            if not transforms:
                # Identity branch: COPY edge from gate to coalesce
                graph.add_edge(gate, coalesce, label=branch_name, mode=RoutingMode.COPY)
            else:
                # Transform branch: MOVE edges through the chain
                prev = gate
                for t_name in transforms:
                    graph.add_node(t_name, node_type=NodeType.TRANSFORM, plugin_name="test-transform", config={})
                    edge_label = branch_name if prev == gate else "continue"
                    graph.add_edge(prev, t_name, label=edge_label, mode=RoutingMode.MOVE)
                    prev = t_name
                graph.add_edge(prev, coalesce, label="continue", mode=RoutingMode.MOVE)

        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test-sink", config={})
        graph.add_edge(coalesce, "sink", label="continue")

        # Set metadata required by get_branch_first_nodes()
        graph.set_branch_to_coalesce({BranchName(b): CoalesceName("coalesce-node") for b in branches})
        graph.set_coalesce_id_map(
            {
                CoalesceName("coalesce-node"): NodeID("coalesce-node"),
            }
        )

        # Map each branch to its producing gate node ID
        graph.set_branch_gate_map({BranchName(b): NodeID(gate) for b in branches})

        return graph

    def test_get_branch_first_nodes_identity(self) -> None:
        """Identity branches (no transforms) map to the coalesce node.

        When a branch has no transforms, the token goes straight from
        the gate to the coalesce — the "first node" IS the coalesce node.
        """
        graph = self._build_graph_with_modes(
            {
                "path_a": [],
                "path_b": [],
            }
        )

        result = graph.get_branch_first_nodes()

        assert result["path_a"] == NodeID("coalesce-node")
        assert result["path_b"] == NodeID("coalesce-node")

    def test_get_branch_first_nodes_transform(self) -> None:
        """Transform branches map to the first transform node.

        When a branch has transforms, the "first node" is the first
        transform in the chain, not the coalesce.
        """
        graph = self._build_graph_with_modes(
            {
                "path_a": ["t_a1", "t_a2"],
                "path_b": ["t_b1"],
            }
        )

        result = graph.get_branch_first_nodes()

        assert result["path_a"] == NodeID("t_a1")
        assert result["path_b"] == NodeID("t_b1")

    def test_get_branch_first_nodes_mixed(self) -> None:
        """Mixed identity + transform branches return correct first nodes.

        One identity branch maps to coalesce, one transform branch maps
        to the first transform — both in the same result dict.
        """
        graph = self._build_graph_with_modes(
            {
                "path_a": [],  # identity → coalesce
                "path_b": ["enrich"],  # transform → first transform
            }
        )

        result = graph.get_branch_first_nodes()

        assert result["path_a"] == NodeID("coalesce-node")
        assert result["path_b"] == NodeID("enrich")

    def test_get_branch_first_nodes_label_collision_with_intermediate_gate(self) -> None:
        """Intermediate gate whose MOVE label matches branch name must not confuse trace-back.

        Regression: _trace_branch_endpoints() stops at the first incoming MOVE edge
        whose label matches branch_name. If an intermediate gate within the branch
        re-uses the branch name as a route label, the walk stops too early and returns
        a downstream node instead of the true branch start.

        Graph: fork_gate --(b1 MOVE)--> g1 --(b1 MOVE)--> g2 --(continue MOVE)--> coalesce
        Expected first node: g1 (receives the b1 MOVE from the fork gate)
        Bug behavior: returns g2 (matches label b1 on the g1→g2 edge)
        """
        from elspeth.contracts import RoutingMode
        from elspeth.contracts.types import BranchName, CoalesceName

        graph = ExecutionGraph()
        gate = "fork-gate"
        coalesce = "coalesce-node"

        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test-source", config={})
        graph.add_node(gate, node_type=NodeType.GATE, plugin_name="test-gate", config={})
        # g1 is the first node in branch "b1" — it's a gate that routes with label "b1"
        graph.add_node("g1", node_type=NodeType.GATE, plugin_name="inner-gate", config={})
        # g2 is downstream — it receives a MOVE edge labelled "b1" from g1
        graph.add_node("g2", node_type=NodeType.TRANSFORM, plugin_name="test-transform", config={})
        graph.add_node(coalesce, node_type=NodeType.COALESCE, plugin_name="coalesce", config={})
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test-sink", config={})

        # Wiring: source → fork_gate → g1 → g2 → coalesce → sink
        graph.add_edge("source", gate, label="continue")
        graph.add_edge(gate, "g1", label="b1", mode=RoutingMode.MOVE)  # fork creates branch b1
        graph.add_edge("g1", "g2", label="b1", mode=RoutingMode.MOVE)  # intermediate gate re-uses "b1" label
        graph.add_edge("g2", coalesce, label="continue", mode=RoutingMode.MOVE)
        graph.add_edge(coalesce, "sink", label="continue")

        # Metadata
        graph.set_branch_to_coalesce({BranchName("b1"): CoalesceName("coalesce-node")})
        graph.set_coalesce_id_map({CoalesceName("coalesce-node"): NodeID("coalesce-node")})
        # Map branch to its producing gate node ID
        graph.set_branch_gate_map({BranchName("b1"): NodeID("fork-gate")})

        result = graph.get_branch_first_nodes()

        # Must resolve to g1 (the true branch start), NOT g2
        assert result["b1"] == NodeID("g1")

    def test_get_branch_first_nodes_multi_gate_coalesce(self) -> None:
        """Branches from different gates feeding one coalesce resolve correctly.

        Regression: _trace_branch_endpoints() assumed one gate per coalesce
        via _coalesce_gate_index, so branches from a second gate would fail
        the from_id == fork_gate_nid check and raise GraphValidationError.

        Graph: source → gate_a → [t_a] → coalesce → sink
               source → gate_b → [t_b] → coalesce
        """
        from elspeth.contracts import RoutingMode
        from elspeth.contracts.types import BranchName, CoalesceName

        graph = ExecutionGraph()

        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test-source", config={})
        graph.add_node("gate_a", node_type=NodeType.GATE, plugin_name="test-gate-a", config={})
        graph.add_node("gate_b", node_type=NodeType.GATE, plugin_name="test-gate-b", config={})
        graph.add_node("t_a", node_type=NodeType.TRANSFORM, plugin_name="transform-a", config={})
        graph.add_node("t_b", node_type=NodeType.TRANSFORM, plugin_name="transform-b", config={})
        graph.add_node("coalesce", node_type=NodeType.COALESCE, plugin_name="coalesce", config={})
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="test-sink", config={})

        graph.add_edge("source", "gate_a", label="continue")
        graph.add_edge("source", "gate_b", label="continue")

        # Gate A produces path_a
        graph.add_edge("gate_a", "t_a", label="path_a", mode=RoutingMode.MOVE)
        graph.add_edge("t_a", "coalesce", label="continue", mode=RoutingMode.MOVE)

        # Gate B produces path_b
        graph.add_edge("gate_b", "t_b", label="path_b", mode=RoutingMode.MOVE)
        graph.add_edge("t_b", "coalesce", label="continue", mode=RoutingMode.MOVE)

        graph.add_edge("coalesce", "sink", label="continue")

        # Metadata: both branches feed the same coalesce, from different gates
        graph.set_branch_to_coalesce(
            {
                BranchName("path_a"): CoalesceName("coalesce"),
                BranchName("path_b"): CoalesceName("coalesce"),
            }
        )
        graph.set_coalesce_id_map({CoalesceName("coalesce"): NodeID("coalesce")})
        graph.set_branch_gate_map(
            {
                BranchName("path_a"): NodeID("gate_a"),
                BranchName("path_b"): NodeID("gate_b"),
            }
        )

        result = graph.get_branch_first_nodes()

        # Both branches must resolve despite coming from different gates
        assert result["path_a"] == NodeID("t_a")
        assert result["path_b"] == NodeID("t_b")
