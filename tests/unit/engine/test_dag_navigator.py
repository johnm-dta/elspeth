# tests/unit/engine/test_dag_navigator.py
"""Unit tests for DAGNavigator — pure DAG topology queries.

DAGNavigator was extracted from RowProcessor to create a clean service boundary
for navigation concerns. All methods are pure queries on immutable data.

Test strategy:
- Construct minimal DAGNavigator instances with 2-3 node topologies
- Verify each resolution method's happy path and invariant errors
- Verify WorkItem coalesce invariant (moved from test_processor.py)
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from elspeth.contracts import TransformProtocol
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.types import CoalesceName, NodeID
from elspeth.core.config import GateSettings
from elspeth.engine.dag_navigator import DAGNavigator
from elspeth.engine.processor import DAGTraversalContext
from elspeth.engine.work_items import WorkItem, WorkItemFactory
from elspeth.testing import make_token_info

# =============================================================================
# Helpers
# =============================================================================


def _make_mock_transform(
    *,
    node_id: str = "transform-1",
    name: str = "test_transform",
    on_success: str | None = None,
) -> Mock:
    """Create a mock TransformProtocol with on_success."""
    t = Mock(spec=TransformProtocol)
    t.node_id = node_id
    t.name = name
    t.on_success = on_success
    return t


def _make_nav(
    *,
    node_to_plugin: dict[NodeID, TransformProtocol | GateSettings] | None = None,
    node_to_next: dict[NodeID, NodeID | None] | None = None,
    coalesce_node_ids: dict[CoalesceName, NodeID] | None = None,
    structural_node_ids: frozenset[NodeID] | None = None,
    coalesce_name_by_node_id: dict[NodeID, CoalesceName] | None = None,
    coalesce_on_success_map: dict[CoalesceName, str] | None = None,
    sink_names: frozenset[str] | None = None,
    branch_first_node: dict[str, NodeID] | None = None,
) -> DAGNavigator:
    """Create a DAGNavigator with sensible defaults."""
    _node_to_plugin = node_to_plugin or {}
    _node_to_next = node_to_next or {NodeID("source-0"): None}
    _coalesce_node_ids = coalesce_node_ids or {}
    _coalesce_name_by_node_id = coalesce_name_by_node_id or {}
    _structural = (
        structural_node_ids if structural_node_ids is not None else frozenset(nid for nid in _node_to_next if nid not in _node_to_plugin)
    )

    return DAGNavigator(
        node_to_plugin=_node_to_plugin,
        node_to_next=_node_to_next,
        coalesce_node_ids=_coalesce_node_ids,
        structural_node_ids=_structural,
        coalesce_name_by_node_id=_coalesce_name_by_node_id,
        coalesce_on_success_map=coalesce_on_success_map or {},
        sink_names=sink_names or frozenset(),
        branch_first_node=branch_first_node or {},
    )


# =============================================================================
# WorkItem coalesce invariant (moved from TestWorkItemCoalesceInvariant)
# =============================================================================


class TestWorkItemCoalesceInvariant:
    """WorkItem must carry complete coalesce metadata together."""

    def test_missing_coalesce_name_with_coalesce_node_id_raises(self) -> None:
        """Coalesce node without coalesce name is an invariant violation."""
        token = make_token_info(data={"value": 1})
        with pytest.raises(OrchestrationInvariantError, match="coalesce fields must be both set or both None"):
            WorkItem(
                token=token,
                current_node_id=NodeID("coalesce::merge"),
                coalesce_node_id=NodeID("coalesce::merge"),
                coalesce_name=None,
            )

    def test_missing_coalesce_node_id_with_coalesce_name_raises(self) -> None:
        """Coalesce name without coalesce node is an invariant violation."""
        token = make_token_info(data={"value": 1})
        with pytest.raises(OrchestrationInvariantError, match="coalesce fields must be both set or both None"):
            WorkItem(
                token=token,
                current_node_id=NodeID("coalesce::merge"),
                coalesce_node_id=None,
                coalesce_name=CoalesceName("merge"),
            )

    def test_both_none_is_valid(self) -> None:
        """Both coalesce fields None is valid (non-coalesce work item)."""
        token = make_token_info(data={"value": 1})
        item = WorkItem(token=token, current_node_id=NodeID("t-1"))
        assert item.coalesce_node_id is None
        assert item.coalesce_name is None

    def test_both_set_is_valid(self) -> None:
        """Both coalesce fields set is valid (coalesce work item)."""
        token = make_token_info(data={"value": 1})
        item = WorkItem(
            token=token,
            current_node_id=NodeID("coalesce::merge"),
            coalesce_node_id=NodeID("coalesce::merge"),
            coalesce_name=CoalesceName("merge"),
        )
        assert item.coalesce_node_id == NodeID("coalesce::merge")
        assert item.coalesce_name == CoalesceName("merge")


# =============================================================================
# resolve_plugin_for_node
# =============================================================================


class TestResolvePluginForNode:
    """Tests for plugin/structural node resolution."""

    def test_returns_plugin_for_known_node(self) -> None:
        """Known plugin nodes return their plugin."""
        transform = _make_mock_transform()
        nav = _make_nav(
            node_to_plugin={NodeID("transform-1"): transform},
            node_to_next={NodeID("source-0"): NodeID("transform-1"), NodeID("transform-1"): None},
        )
        assert nav.resolve_plugin_for_node(NodeID("transform-1")) is transform

    def test_returns_none_for_structural_node(self) -> None:
        """Structural nodes (coalesce points) return None."""
        nav = _make_nav(
            node_to_next={
                NodeID("source-0"): NodeID("coalesce::merge"),
                NodeID("coalesce::merge"): None,
            },
            structural_node_ids=frozenset({NodeID("coalesce::merge")}),
        )
        assert nav.resolve_plugin_for_node(NodeID("coalesce::merge")) is None

    def test_raises_for_unknown_node(self) -> None:
        """Unknown nodes raise OrchestrationInvariantError."""
        nav = _make_nav()
        with pytest.raises(OrchestrationInvariantError, match="neither a plugin node nor a known structural node"):
            nav.resolve_plugin_for_node(NodeID("unknown-node"))


# =============================================================================
# resolve_next_node
# =============================================================================


class TestResolveNextNode:
    """Tests for next-node resolution."""

    def test_returns_next_node(self) -> None:
        """Known nodes return their successor."""
        nav = _make_nav(
            node_to_next={NodeID("source-0"): NodeID("t-1"), NodeID("t-1"): None},
        )
        assert nav.resolve_next_node(NodeID("source-0")) == NodeID("t-1")

    def test_returns_none_for_terminal(self) -> None:
        """Terminal nodes return None."""
        nav = _make_nav(node_to_next={NodeID("source-0"): None})
        assert nav.resolve_next_node(NodeID("source-0")) is None

    def test_raises_for_missing_node(self) -> None:
        """Missing nodes raise OrchestrationInvariantError."""
        nav = _make_nav()
        with pytest.raises(OrchestrationInvariantError, match="missing from traversal next-node map"):
            nav.resolve_next_node(NodeID("nonexistent"))


# =============================================================================
# resolve_coalesce_sink
# =============================================================================


class TestResolveCoalesceSink:
    """Tests for coalesce terminal sink resolution."""

    def test_returns_mapped_sink(self) -> None:
        """Known coalesce names return their mapped sink."""
        nav = _make_nav(
            coalesce_on_success_map={CoalesceName("merge"): "output_sink"},
        )
        assert nav.resolve_coalesce_sink(CoalesceName("merge"), context="test") == "output_sink"

    def test_raises_for_missing_coalesce(self) -> None:
        """Missing coalesce names raise OrchestrationInvariantError."""
        nav = _make_nav()
        with pytest.raises(OrchestrationInvariantError, match="not in on_success map"):
            nav.resolve_coalesce_sink(CoalesceName("unknown"), context="test")


# =============================================================================
# resolve_jump_target_sink
# =============================================================================


class TestResolveJumpTargetSink:
    """Tests for jump-target terminal sink resolution."""

    def test_resolves_through_transform_on_success(self) -> None:
        """Walk resolves sink from transform's on_success field."""
        transform = _make_mock_transform(
            node_id="branch-t1",
            on_success="branch_sink",
        )
        nav = _make_nav(
            node_to_plugin={NodeID("branch-t1"): transform},
            node_to_next={
                NodeID("source-0"): NodeID("branch-t1"),
                NodeID("branch-t1"): None,
            },
            sink_names=frozenset({"branch_sink"}),
        )
        assert nav.resolve_jump_target_sink(NodeID("branch-t1")) == "branch_sink"

    def test_returns_none_for_gate_path(self) -> None:
        """Paths containing a gate return None (gate self-routes)."""
        gate = GateSettings(name="gate1", input="in_conn", condition="True", routes={"true": "out", "false": "err"})
        nav = _make_nav(
            node_to_plugin={NodeID("gate-1"): gate},
            node_to_next={
                NodeID("source-0"): NodeID("gate-1"),
                NodeID("gate-1"): None,
            },
        )
        assert nav.resolve_jump_target_sink(NodeID("gate-1")) is None

    def test_raises_when_no_sink_and_no_gate(self) -> None:
        """Transform-only path with no valid sink raises invariant error."""
        transform = _make_mock_transform(
            node_id="branch-t1",
            on_success="nonexistent_conn",
        )
        nav = _make_nav(
            node_to_plugin={NodeID("branch-t1"): transform},
            node_to_next={
                NodeID("source-0"): NodeID("branch-t1"),
                NodeID("branch-t1"): None,
            },
            sink_names=frozenset({"valid_sink"}),
        )
        with pytest.raises(OrchestrationInvariantError, match="no sink"):
            nav.resolve_jump_target_sink(NodeID("branch-t1"))

    def test_resolves_terminal_coalesce_on_success(self) -> None:
        """Walk resolves sink from terminal coalesce's on_success map."""
        coalesce_node = NodeID("coalesce::merge")
        nav = _make_nav(
            node_to_next={
                NodeID("source-0"): coalesce_node,
                coalesce_node: None,
            },
            structural_node_ids=frozenset({coalesce_node}),
            coalesce_name_by_node_id={coalesce_node: CoalesceName("merge")},
            coalesce_on_success_map={CoalesceName("merge"): "merged_sink"},
        )
        assert nav.resolve_jump_target_sink(coalesce_node) == "merged_sink"


# =============================================================================
# WorkItemFactory.create
# =============================================================================


class TestCreateWorkItem:
    """Tests for work item factory."""

    def test_creates_basic_work_item(self) -> None:
        """Creates work item with token and node."""
        token = make_token_info(data={"v": 1})
        nav = _make_nav()
        item = WorkItemFactory(nav).create(token=token, current_node_id=NodeID("t-1"))
        assert item.token is token
        assert item.current_node_id == NodeID("t-1")
        assert item.coalesce_node_id is None

    def test_resolves_coalesce_node_from_name(self) -> None:
        """Coalesce name is resolved to coalesce node ID."""
        token = make_token_info(data={"v": 1})
        nav = _make_nav(
            coalesce_node_ids={CoalesceName("merge"): NodeID("coalesce::merge")},
        )
        item = WorkItemFactory(nav).create(
            token=token,
            current_node_id=NodeID("coalesce::merge"),
            coalesce_name=CoalesceName("merge"),
        )
        assert item.coalesce_node_id == NodeID("coalesce::merge")
        assert item.coalesce_name == CoalesceName("merge")

    def test_preserves_on_success_sink(self) -> None:
        """on_success_sink is preserved on the work item."""
        token = make_token_info(data={"v": 1})
        nav = _make_nav()
        item = WorkItemFactory(nav).create(
            token=token,
            current_node_id=None,
            on_success_sink="branch_sink",
        )
        assert item.on_success_sink == "branch_sink"


# =============================================================================
# WorkItemFactory.create_continuation
# =============================================================================


class TestCreateContinuationWorkItem:
    """Tests for continuation work item factory."""

    def test_continues_to_next_node(self) -> None:
        """Without coalesce, continues to next node in traversal."""
        token = make_token_info(data={"v": 1})
        nav = _make_nav(
            node_to_next={NodeID("t-1"): NodeID("t-2"), NodeID("t-2"): None},
        )
        item = WorkItemFactory(nav).create_continuation(
            token=token,
            current_node_id=NodeID("t-1"),
        )
        assert item.current_node_id == NodeID("t-2")

    def test_jumps_to_branch_start_when_fork_gate_origin(self) -> None:
        """With coalesce name from a gate node, routes to branch first node."""
        gate = GateSettings(name="fork_gate", input="in", condition="True", routes={"true": "out", "false": "err"})
        gate_node = NodeID("gate-1")
        token = make_token_info(data={"v": 1}, branch_name="path_a")
        coalesce_node = NodeID("coalesce::merge")
        nav = _make_nav(
            node_to_plugin={gate_node: gate},
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            node_to_next={gate_node: None, coalesce_node: None},
            branch_first_node={"path_a": coalesce_node},
        )
        item = WorkItemFactory(nav).create_continuation(
            token=token,
            current_node_id=gate_node,
            coalesce_name=CoalesceName("merge"),
        )
        assert item.current_node_id == coalesce_node
        assert item.coalesce_name == CoalesceName("merge")

    def test_preserves_on_success_sink(self) -> None:
        """on_success_sink is propagated to continuation item."""
        token = make_token_info(data={"v": 1})
        nav = _make_nav(
            node_to_next={NodeID("t-1"): NodeID("t-2"), NodeID("t-2"): None},
        )
        item = WorkItemFactory(nav).create_continuation(
            token=token,
            current_node_id=NodeID("t-1"),
            on_success_sink="branch_sink",
        )
        assert item.on_success_sink == "branch_sink"


# =============================================================================
# from_traversal_context factory
# =============================================================================


class TestFromTraversalContext:
    """Tests for the ergonomic factory constructor."""

    def test_derives_structural_node_ids(self) -> None:
        """Structural nodes are automatically derived from topology difference."""
        transform = _make_mock_transform()
        coalesce_node = NodeID("coalesce::merge")

        traversal = DAGTraversalContext(
            node_step_map={NodeID("source-0"): 0, NodeID("transform-1"): 1, coalesce_node: 2},
            node_to_plugin={NodeID("transform-1"): transform},
            node_to_next={
                NodeID("source-0"): NodeID("transform-1"),
                NodeID("transform-1"): coalesce_node,
                coalesce_node: None,
            },
            coalesce_node_map={CoalesceName("merge"): coalesce_node},
        )

        nav = DAGNavigator.from_traversal_context(traversal)

        # coalesce::merge and source-0 are in node_to_next but not in node_to_plugin
        assert nav.resolve_plugin_for_node(coalesce_node) is None  # structural
        assert nav.resolve_plugin_for_node(NodeID("transform-1")) is transform

    def test_derives_coalesce_name_by_node_id(self) -> None:
        """Coalesce reverse map is derived correctly."""
        coalesce_node = NodeID("coalesce::merge")
        traversal = DAGTraversalContext(
            node_step_map={NodeID("source-0"): 0, coalesce_node: 1},
            node_to_plugin={},
            node_to_next={NodeID("source-0"): coalesce_node, coalesce_node: None},
            coalesce_node_map={CoalesceName("merge"): coalesce_node},
        )

        nav = DAGNavigator.from_traversal_context(
            traversal,
            coalesce_on_success_map={CoalesceName("merge"): "output_sink"},
        )

        # Verify the walk can resolve through coalesce
        assert nav.resolve_jump_target_sink(coalesce_node) == "output_sink"

    def test_unmapped_plugin_node_fails_closed(self) -> None:
        """A traversal node with no plugin and no structural role raises, not skips.

        Regression for elspeth-c522931bd1: the complement derivation classified
        every plugin-less node_to_next entry as structural, so a transform
        missing from node_to_plugin was silently skipped (fail-open).
        """
        transform = _make_mock_transform()
        traversal = DAGTraversalContext(
            node_step_map={NodeID("transform-1"): 1, NodeID("transform-2"): 2},
            node_to_plugin={NodeID("transform-1"): transform},
            node_to_next={
                NodeID("transform-1"): NodeID("transform-2"),
                NodeID("transform-2"): None,
            },
            coalesce_node_map={},
        )

        nav = DAGNavigator.from_traversal_context(traversal)

        with pytest.raises(OrchestrationInvariantError, match="transform-2"):
            nav.resolve_plugin_for_node(NodeID("transform-2"))

    def test_explicit_structural_node_ids_are_honored(self) -> None:
        """Nodes in the context's explicit structural set resolve to None (skip)."""
        traversal = DAGTraversalContext(
            node_step_map={},
            node_to_plugin={},
            node_to_next={NodeID("queue-1"): None},
            coalesce_node_map={},
            structural_node_ids=frozenset({NodeID("queue-1")}),
        )

        nav = DAGNavigator.from_traversal_context(traversal)

        assert nav.resolve_plugin_for_node(NodeID("queue-1")) is None

    def test_coalesce_nodes_are_always_structural(self) -> None:
        """Coalesce map entries are unioned into the structural set by the context."""
        coalesce_node = NodeID("coalesce::merge")
        traversal = DAGTraversalContext(
            node_step_map={},
            node_to_plugin={},
            node_to_next={coalesce_node: None},
            coalesce_node_map={CoalesceName("merge"): coalesce_node},
            structural_node_ids=frozenset(),
        )

        assert coalesce_node in traversal.structural_node_ids

    def test_passes_through_supplementary_params(self) -> None:
        """Supplementary params (sink_names, coalesce_on_success_map) are passed through."""
        traversal = DAGTraversalContext(
            node_step_map={NodeID("source-0"): 0},
            node_to_plugin={},
            node_to_next={NodeID("source-0"): None},
            coalesce_node_map={},
        )

        nav = DAGNavigator.from_traversal_context(
            traversal,
            coalesce_on_success_map={CoalesceName("merge"): "out"},
            sink_names=frozenset({"out", "err"}),
        )
        assert nav.resolve_coalesce_sink(CoalesceName("merge"), context="test") == "out"


# =============================================================================
# ARCH-15: Per-branch transform routing
# =============================================================================


class TestBranchTransformRouting:
    """Tests for per-branch transform routing in WorkItemFactory.create_continuation.

    ARCH-15 introduces per-branch transforms: fork children can now be routed
    to the first transform in their branch chain instead of directly to the
    coalesce node. The _branch_first_node mapping controls this.
    """

    def test_continuation_routes_through_branch_transform(self) -> None:
        """Fork child should go to the first transform, not coalesce.

        When branch_first_node maps a branch to a transform node (not the
        coalesce), the fork child should be routed to that transform first.
        """
        gate = GateSettings(name="fork_gate", input="in", condition="True", routes={"true": "out", "false": "err"})
        gate_node = NodeID("gate-1")
        transform = _make_mock_transform(node_id="branch-t1", name="enrich")
        coalesce_node = NodeID("coalesce::merge")
        branch_t1 = NodeID("branch-t1")

        nav = _make_nav(
            node_to_plugin={gate_node: gate, branch_t1: transform},
            node_to_next={
                gate_node: None,
                branch_t1: coalesce_node,
                coalesce_node: None,
            },
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            structural_node_ids=frozenset({coalesce_node}),
            branch_first_node={"path_a": branch_t1},  # Transform branch
        )

        token = make_token_info(data={"v": 1}, branch_name="path_a")
        item = WorkItemFactory(nav).create_continuation(
            token=token,
            current_node_id=gate_node,
            coalesce_name=CoalesceName("merge"),
        )

        # Should route to the branch transform, not directly to coalesce
        assert item.current_node_id == branch_t1
        assert item.coalesce_name == CoalesceName("merge")
        assert item.coalesce_node_id == coalesce_node

    def test_continuation_identity_branch_goes_to_coalesce(self) -> None:
        """Identity branch should route directly to the coalesce node.

        When branch_first_node maps a branch to the coalesce node itself,
        the fork child goes straight there (preserving pre-ARCH-15 behavior).
        """
        gate = GateSettings(name="fork_gate", input="in", condition="True", routes={"true": "out", "false": "err"})
        gate_node = NodeID("gate-1")
        coalesce_node = NodeID("coalesce::merge")

        nav = _make_nav(
            node_to_plugin={gate_node: gate},
            node_to_next={
                gate_node: None,
                coalesce_node: None,
            },
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            structural_node_ids=frozenset({coalesce_node}),
            branch_first_node={"path_a": coalesce_node},  # Identity branch
        )

        token = make_token_info(data={"v": 1}, branch_name="path_a")
        item = WorkItemFactory(nav).create_continuation(
            token=token,
            current_node_id=gate_node,
            coalesce_name=CoalesceName("merge"),
        )

        assert item.current_node_id == coalesce_node
        assert item.coalesce_name == CoalesceName("merge")

    def test_work_item_coalesce_name_propagation(self) -> None:
        """coalesce_name must be preserved through the branch transform chain.

        When a work item is created for a branch transform, the coalesce_name
        and coalesce_node_id must be carried forward so the processor knows
        to route to the coalesce node after the branch chain completes.
        """
        gate = GateSettings(name="fork_gate", input="in", condition="True", routes={"true": "out", "false": "err"})
        gate_node = NodeID("gate-1")
        transform = _make_mock_transform(node_id="branch-t1", name="enrich")
        coalesce_node = NodeID("coalesce::merge")
        branch_t1 = NodeID("branch-t1")
        branch_t2 = NodeID("branch-t2")

        nav = _make_nav(
            node_to_plugin={gate_node: gate, branch_t1: transform},
            node_to_next={
                gate_node: None,
                branch_t1: branch_t2,
                branch_t2: coalesce_node,
                coalesce_node: None,
            },
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            structural_node_ids=frozenset({coalesce_node}),
            branch_first_node={"path_a": branch_t1},
        )

        token = make_token_info(data={"v": 1}, branch_name="path_a")

        # First: fork creates work item at branch start
        item1 = WorkItemFactory(nav).create_continuation(
            token=token,
            current_node_id=gate_node,
            coalesce_name=CoalesceName("merge"),
        )
        assert item1.current_node_id == branch_t1
        assert item1.coalesce_name == CoalesceName("merge")
        assert item1.coalesce_node_id == coalesce_node

        # Second: processor continues from branch-t1 to branch-t2
        # This uses normal continuation (no coalesce_name — it's mid-chain)
        item2 = WorkItemFactory(nav).create_continuation(
            token=token,
            current_node_id=branch_t1,
        )
        assert item2.current_node_id == branch_t2


# =============================================================================
# Regression: P1-2026-02-14 — coalesce_name misroutes non-fork continuations
# =============================================================================


class TestContinuationCoalesceNonForkRegression:
    """Regression tests for P1-2026-02-14.

    Bug: continuation work-item creation treated ANY token with coalesce_name
    as a fresh fork child and routed it to branch_first_node. This caused
    backward jumps for non-fork continuations (deaggregation, aggregation flush)
    that are already mid-branch.

    Fix: Only route to branch_first_node when current_node_id is a gate node
    (fork origin). Non-fork continuations advance via resolve_next_node while
    preserving coalesce metadata.
    """

    def test_deaggregation_child_advances_forward_not_to_branch_start(self) -> None:
        """Deaggregation child in a branch must advance to next node, not jump back.

        Simulates: fork gate -> branch_t1 (deagg transform) -> branch_t2 -> coalesce
        A child token from expand_token at branch_t1 should continue to branch_t2,
        not jump back to branch_t1.
        """
        gate = GateSettings(name="fork_gate", input="in", condition="True", routes={"true": "out", "false": "err"})
        gate_node = NodeID("gate-1")
        deagg_transform = _make_mock_transform(node_id="branch-t1", name="deagg")
        next_transform = _make_mock_transform(node_id="branch-t2", name="next_step")
        coalesce_node = NodeID("coalesce::merge")
        branch_t1 = NodeID("branch-t1")
        branch_t2 = NodeID("branch-t2")

        nav = _make_nav(
            node_to_plugin={gate_node: gate, branch_t1: deagg_transform, branch_t2: next_transform},
            node_to_next={
                gate_node: None,
                branch_t1: branch_t2,
                branch_t2: coalesce_node,
                coalesce_node: None,
            },
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            structural_node_ids=frozenset({coalesce_node}),
            branch_first_node={"path_a": branch_t1},
        )

        # Expanded child token inherits branch_name from parent
        child_token = make_token_info(data={"v": 1}, branch_name="path_a")

        # Deaggregation continuation: current_node_id is the deagg transform (NOT the gate)
        item = WorkItemFactory(nav).create_continuation(
            token=child_token,
            current_node_id=branch_t1,
            coalesce_name=CoalesceName("merge"),
        )

        # Must advance to branch_t2 (next node), NOT jump back to branch_t1
        assert item.current_node_id == branch_t2
        # Coalesce metadata must be preserved for eventual coalesce handling
        assert item.coalesce_name == CoalesceName("merge")
        assert item.coalesce_node_id == coalesce_node

    def test_aggregation_flush_advances_forward_not_to_branch_start(self) -> None:
        """Aggregation flush continuation in a branch must advance forward.

        Simulates: fork gate -> branch_t1 (agg transform) -> branch_t2 -> coalesce
        After aggregation flush at branch_t1, tokens should continue to branch_t2.
        """
        gate = GateSettings(name="fork_gate", input="in", condition="True", routes={"true": "out", "false": "err"})
        gate_node = NodeID("gate-1")
        agg_transform = _make_mock_transform(node_id="branch-t1", name="batch_agg")
        post_agg = _make_mock_transform(node_id="branch-t2", name="post_agg")
        coalesce_node = NodeID("coalesce::merge")
        branch_t1 = NodeID("branch-t1")
        branch_t2 = NodeID("branch-t2")

        nav = _make_nav(
            node_to_plugin={gate_node: gate, branch_t1: agg_transform, branch_t2: post_agg},
            node_to_next={
                gate_node: None,
                branch_t1: branch_t2,
                branch_t2: coalesce_node,
                coalesce_node: None,
            },
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            structural_node_ids=frozenset({coalesce_node}),
            branch_first_node={"path_a": branch_t1},
        )

        # Token from aggregation flush (passthrough or transform mode)
        flush_token = make_token_info(data={"v": 1}, branch_name="path_a")

        # Aggregation flush continuation: current_node_id is the agg transform
        item = WorkItemFactory(nav).create_continuation(
            token=flush_token,
            current_node_id=branch_t1,
            coalesce_name=CoalesceName("merge"),
        )

        # Must advance to branch_t2, NOT jump back to branch_t1
        assert item.current_node_id == branch_t2
        assert item.coalesce_name == CoalesceName("merge")
        assert item.coalesce_node_id == coalesce_node

    def test_fork_gate_origin_still_routes_to_branch_start(self) -> None:
        """Fork gate origin must still route to branch_first_node (not regressed).

        The fix must not break the fork-origin path: when the continuation
        originates from a gate node, fresh fork children should still be
        routed to the branch's first processing node.
        """
        gate = GateSettings(name="fork_gate", input="in", condition="True", routes={"true": "out", "false": "err"})
        gate_node = NodeID("gate-1")
        branch_t1 = _make_mock_transform(node_id="branch-t1", name="enrich")
        coalesce_node = NodeID("coalesce::merge")

        nav = _make_nav(
            node_to_plugin={gate_node: gate, NodeID("branch-t1"): branch_t1},
            node_to_next={
                gate_node: None,
                NodeID("branch-t1"): coalesce_node,
                coalesce_node: None,
            },
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            structural_node_ids=frozenset({coalesce_node}),
            branch_first_node={"path_a": NodeID("branch-t1")},
        )

        fork_child = make_token_info(data={"v": 1}, branch_name="path_a")
        item = WorkItemFactory(nav).create_continuation(
            token=fork_child,
            current_node_id=gate_node,
            coalesce_name=CoalesceName("merge"),
        )

        # Fork gate origin: must route to branch_first_node
        assert item.current_node_id == NodeID("branch-t1")
        assert item.coalesce_name == CoalesceName("merge")

    def test_non_fork_continuation_preserves_on_success_sink(self) -> None:
        """Non-fork continuation with coalesce_name must preserve on_success_sink.

        Deaggregation children inherit on_success_sink from their parent.
        This must survive the coalesce-aware continuation routing.
        """
        gate = GateSettings(name="fork_gate", input="in", condition="True", routes={"true": "out", "false": "err"})
        gate_node = NodeID("gate-1")
        deagg_transform = _make_mock_transform(node_id="branch-t1", name="deagg")
        coalesce_node = NodeID("coalesce::merge")
        branch_t1 = NodeID("branch-t1")

        nav = _make_nav(
            node_to_plugin={gate_node: gate, branch_t1: deagg_transform},
            node_to_next={
                gate_node: None,
                branch_t1: coalesce_node,
                coalesce_node: None,
            },
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            structural_node_ids=frozenset({coalesce_node}),
            branch_first_node={"path_a": branch_t1},
        )

        child_token = make_token_info(data={"v": 1}, branch_name="path_a")
        item = WorkItemFactory(nav).create_continuation(
            token=child_token,
            current_node_id=branch_t1,
            coalesce_name=CoalesceName("merge"),
            on_success_sink="inherited_sink",
        )

        assert item.current_node_id == coalesce_node  # next node after branch_t1
        assert item.on_success_sink == "inherited_sink"
        assert item.coalesce_name == CoalesceName("merge")

    def test_non_fork_continuation_without_coalesce_still_works(self) -> None:
        """Non-fork continuation without coalesce_name (linear pipeline) is unaffected."""
        transform = _make_mock_transform(node_id="t-1", name="step1")
        nav = _make_nav(
            node_to_plugin={NodeID("t-1"): transform},
            node_to_next={NodeID("t-1"): NodeID("t-2"), NodeID("t-2"): None},
        )

        token = make_token_info(data={"v": 1})
        item = WorkItemFactory(nav).create_continuation(
            token=token,
            current_node_id=NodeID("t-1"),
        )

        assert item.current_node_id == NodeID("t-2")
        assert item.coalesce_name is None
        assert item.coalesce_node_id is None


# =============================================================================
# resolve_jump_target_sink: cycle-detection guard
# =============================================================================


class TestResolveJumpTargetSinkCycleGuard:
    """Tests for the cycle-detection guard in resolve_jump_target_sink."""

    def test_cycle_in_traversal_map_raises(self) -> None:
        """Cycle in node_to_next causes iteration guard to fire."""
        # Create a cycle: t-1 -> t-2 -> t-1
        transform1 = _make_mock_transform(node_id="t-1", name="step1")
        transform2 = _make_mock_transform(node_id="t-2", name="step2")
        nav = _make_nav(
            node_to_plugin={NodeID("t-1"): transform1, NodeID("t-2"): transform2},
            node_to_next={
                NodeID("source-0"): NodeID("t-1"),
                NodeID("t-1"): NodeID("t-2"),
                NodeID("t-2"): NodeID("t-1"),  # cycle back
            },
        )
        with pytest.raises(OrchestrationInvariantError, match=r"exceeded.*iterations.*Possible cycle"):
            nav.resolve_jump_target_sink(NodeID("t-1"))


# =============================================================================
# resolve_jump_target_sink: post-resolution invalid-sink validation
# =============================================================================


class TestResolveJumpTargetSinkInvalidSinkGuard:
    """Tests for the post-resolution invalid-sink guard in resolve_jump_target_sink."""

    def test_resolved_sink_not_in_configured_sinks_raises(self) -> None:
        """Sink resolved via on_success that is not in sink_names raises invariant error."""
        # Transform has on_success pointing to a sink that exists in its own name
        # but is NOT in the configured sink_names set.
        # We need the walk to pick up the sink AND for it to not be in sink_names.
        # The first check (line 216) filters candidates by sink_names, so resolved_sink
        # stays None for non-matching sinks. To trigger the post-resolution guard at
        # line 235, we need resolved_sink to be set via the coalesce path (which
        # doesn't filter by sink_names).
        coalesce_node = NodeID("coalesce::merge")
        nav = _make_nav(
            node_to_next={
                NodeID("source-0"): coalesce_node,
                coalesce_node: None,
            },
            structural_node_ids=frozenset({coalesce_node}),
            coalesce_name_by_node_id={coalesce_node: CoalesceName("merge")},
            coalesce_on_success_map={CoalesceName("merge"): "ghost_sink"},
            sink_names=frozenset({"real_sink"}),  # ghost_sink is NOT here
        )
        with pytest.raises(OrchestrationInvariantError, match="not a configured sink"):
            nav.resolve_jump_target_sink(coalesce_node)


# =============================================================================
# WorkItemFactory.create_continuation: fork child with None branch_name guard
# =============================================================================


class TestContinuationNullBranchNameGuard:
    """Tests for the branch_name=None guard in WorkItemFactory.create_continuation."""

    def test_fork_child_without_branch_name_raises(self) -> None:
        """Fork child token with branch_name=None raises invariant error."""
        gate = GateSettings(
            name="fork_gate",
            input="in",
            condition="True",
            routes={"true": "out", "false": "err"},
        )
        gate_node = NodeID("gate-1")
        coalesce_node = NodeID("coalesce::merge")

        nav = _make_nav(
            node_to_plugin={gate_node: gate},
            node_to_next={gate_node: None, coalesce_node: None},
            coalesce_node_ids={CoalesceName("merge"): coalesce_node},
            structural_node_ids=frozenset({coalesce_node}),
            branch_first_node={"path_a": coalesce_node},
        )

        # Token with branch_name=None (the default)
        token = make_token_info(data={"v": 1})
        assert token.branch_name is None  # precondition

        with pytest.raises(OrchestrationInvariantError, match="branch_name is None"):
            WorkItemFactory(nav).create_continuation(
                token=token,
                current_node_id=gate_node,
                coalesce_name=CoalesceName("merge"),
            )
