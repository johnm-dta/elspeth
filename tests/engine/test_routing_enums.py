"""Tests for engine routing behavior with RoutingKind enum.

This module tests how the engine actually uses RoutingKind values during
gate execution and routing decisions. Contract tests for RoutingKind itself
are in tests/contracts/test_routing.py and tests/contracts/test_enums.py.
"""

from unittest.mock import MagicMock

import pytest

from elspeth.contracts import NodeID, RoutingAction, RoutingKind, TokenInfo
from elspeth.engine.executors import GateExecutor, GateOutcome, MissingEdgeError
from elspeth.plugins.context import PluginContext


class TestGateExecutorRoutingBehavior:
    """Test how GateExecutor handles different RoutingKind values."""

    def _make_executor(
        self,
        edge_map: dict[tuple[str, str], str] | None = None,
        route_resolution_map: dict[tuple[str, str], str] | None = None,
    ) -> GateExecutor:
        """Create a GateExecutor with mocked dependencies."""
        recorder = MagicMock()
        # Mock begin_node_state to return a proper state object
        state_mock = MagicMock()
        state_mock.state_id = "state-1"
        recorder.begin_node_state.return_value = state_mock

        span_factory = MagicMock()
        # Make span context manager work
        span_factory.gate_span.return_value.__enter__ = MagicMock(return_value=None)
        span_factory.gate_span.return_value.__exit__ = MagicMock(return_value=False)

        # Convert str keys to NodeID for type compatibility
        typed_edge_map: dict[tuple[NodeID, str], str] | None = None
        if edge_map is not None:
            typed_edge_map = {(NodeID(k[0]), k[1]): v for k, v in edge_map.items()}

        typed_route_map: dict[tuple[NodeID, str], str] | None = None
        if route_resolution_map is not None:
            typed_route_map = {(NodeID(k[0]), k[1]): v for k, v in route_resolution_map.items()}

        return GateExecutor(
            recorder=recorder,
            span_factory=span_factory,
            edge_map=typed_edge_map or {},
            route_resolution_map=typed_route_map or {},
        )

    def _make_token(self) -> TokenInfo:
        """Create a test token."""
        return TokenInfo(
            row_id="row-1",
            token_id="token-1",
            row_data={"status": "active", "value": 100},
            branch_name=None,
        )

    def _make_context(self) -> PluginContext:
        """Create a test plugin context."""
        return PluginContext(
            run_id="run-1",
            config={},
            node_id="gate-1",
        )

    def test_continue_action_produces_continue_outcome(self) -> None:
        """When gate returns CONTINUE, executor should produce outcome with no sink routing."""
        # Edge map: (node_id, label) -> edge_id
        # For CONTINUE, the executor internally routes to "continue" label
        edge_map = {("gate-1", "continue"): "edge-continue"}
        executor = self._make_executor(edge_map=edge_map)
        token = self._make_token()
        ctx = self._make_context()

        # Mock gate that returns continue action
        gate = MagicMock()
        gate.node_id = "gate-1"
        gate.evaluate.return_value = MagicMock(
            success=True,
            row=token.row_data,
            action=RoutingAction.continue_(),
        )

        outcome = executor.execute_gate(
            gate=gate,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert isinstance(outcome, GateOutcome)
        assert outcome.sink_name is None
        assert outcome.child_tokens == []

    def test_route_action_resolves_to_sink(self) -> None:
        """When gate returns ROUTE with label, executor should resolve to sink name."""
        # Route resolution map: (gate_id, label) -> sink_name
        route_map = {("gate-1", "above"): "high_value_sink"}
        # Edge map for audit recording
        edge_map = {("gate-1", "above"): "edge-above"}
        executor = self._make_executor(route_resolution_map=route_map, edge_map=edge_map)
        token = self._make_token()
        ctx = self._make_context()

        # Mock gate that routes to "above" label
        gate = MagicMock()
        gate.node_id = "gate-1"
        gate.evaluate.return_value = MagicMock(
            success=True,
            row=token.row_data,
            action=RoutingAction.route("above"),
        )

        outcome = executor.execute_gate(
            gate=gate,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        assert outcome.sink_name == "high_value_sink"
        assert outcome.child_tokens == []

    def test_route_action_to_continue_label(self) -> None:
        """When gate routes to label that resolves to 'continue', should not set sink_name."""
        route_map = {("gate-1", "pass"): "continue"}
        edge_map = {("gate-1", "continue"): "edge-continue"}
        executor = self._make_executor(route_resolution_map=route_map, edge_map=edge_map)
        token = self._make_token()
        ctx = self._make_context()

        # Mock gate that routes to "pass" label which resolves to "continue"
        gate = MagicMock()
        gate.node_id = "gate-1"
        gate.evaluate.return_value = MagicMock(
            success=True,
            row=token.row_data,
            action=RoutingAction.route("pass"),
        )

        outcome = executor.execute_gate(
            gate=gate,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
        )

        # "continue" is special - no sink routing
        assert outcome.sink_name is None

    def test_fork_action_creates_child_tokens(self) -> None:
        """When gate returns FORK_TO_PATHS, executor should create child tokens."""
        # Edge map for fork paths
        edge_map = {
            ("gate-1", "branch_a"): "edge-branch-a",
            ("gate-1", "branch_b"): "edge-branch-b",
        }
        executor = self._make_executor(edge_map=edge_map)
        token = self._make_token()
        ctx = self._make_context()

        # Mock token manager for fork operations
        token_manager = MagicMock()
        child_token_1 = TokenInfo(
            row_id="row-1",
            token_id="token-1-branch-a",
            row_data=token.row_data,
            branch_name="branch_a",
        )
        child_token_2 = TokenInfo(
            row_id="row-1",
            token_id="token-1-branch-b",
            row_data=token.row_data,
            branch_name="branch_b",
        )
        token_manager.fork_token.return_value = ([child_token_1, child_token_2], "fork_group_1")

        # Mock gate that forks to two paths
        gate = MagicMock()
        gate.node_id = "gate-1"
        gate.evaluate.return_value = MagicMock(
            success=True,
            row=token.row_data,
            action=RoutingAction.fork_to_paths(["branch_a", "branch_b"]),
        )

        outcome = executor.execute_gate(
            gate=gate,
            token=token,
            ctx=ctx,
            step_in_pipeline=1,
            token_manager=token_manager,
        )

        assert outcome.sink_name is None
        assert len(outcome.child_tokens) == 2
        assert outcome.child_tokens[0].branch_name == "branch_a"
        assert outcome.child_tokens[1].branch_name == "branch_b"

    def test_fork_without_token_manager_raises(self) -> None:
        """Fork action without TokenManager raises RuntimeError."""
        edge_map = {
            ("gate-1", "path_a"): "edge-a",
            ("gate-1", "path_b"): "edge-b",
        }
        executor = self._make_executor(edge_map=edge_map)
        token = self._make_token()
        ctx = self._make_context()

        # Mock gate that forks but no token manager provided
        gate = MagicMock()
        gate.node_id = "gate-1"
        gate.evaluate.return_value = MagicMock(
            success=True,
            row=token.row_data,
            action=RoutingAction.fork_to_paths(["path_a", "path_b"]),
        )

        with pytest.raises(RuntimeError, match="fork_to_paths but no TokenManager"):
            executor.execute_gate(
                gate=gate,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
                token_manager=None,  # No token manager
            )

    def test_route_missing_resolution_raises(self) -> None:
        """Route to unknown label raises MissingEdgeError."""
        # Empty route map - no resolutions configured
        executor = self._make_executor(route_resolution_map={})
        token = self._make_token()
        ctx = self._make_context()

        # Mock gate that routes to unconfigured label
        gate = MagicMock()
        gate.node_id = "gate-1"
        gate.evaluate.return_value = MagicMock(
            success=True,
            row=token.row_data,
            action=RoutingAction.route("unknown_label"),
        )

        with pytest.raises(MissingEdgeError):
            executor.execute_gate(
                gate=gate,
                token=token,
                ctx=ctx,
                step_in_pipeline=1,
            )


class TestRoutingActionKindEnum:
    """Verify RoutingAction correctly uses RoutingKind enum."""

    def test_continue_action_has_continue_kind(self) -> None:
        """RoutingAction.continue_() returns action with CONTINUE kind."""
        action = RoutingAction.continue_()
        assert action.kind is RoutingKind.CONTINUE

    def test_route_action_has_route_kind(self) -> None:
        """RoutingAction.route() returns action with ROUTE kind."""
        action = RoutingAction.route("sink_name")
        assert action.kind is RoutingKind.ROUTE

    def test_fork_action_has_fork_kind(self) -> None:
        """RoutingAction.fork_to_paths() returns action with FORK_TO_PATHS kind."""
        action = RoutingAction.fork_to_paths(["path_a", "path_b"])
        assert action.kind is RoutingKind.FORK_TO_PATHS

    def test_routing_kind_is_str_enum(self) -> None:
        """RoutingKind is (str, Enum) for serialization."""
        # This enables database storage without explicit .value calls
        # Using .value for explicit string comparison instead of relying on implicit coercion
        assert RoutingKind.CONTINUE.value == "continue"
        assert RoutingKind.ROUTE.value == "route"
        assert RoutingKind.FORK_TO_PATHS.value == "fork_to_paths"

    def test_routing_kind_used_for_dispatch(self) -> None:
        """Verify RoutingKind values can be used in if/elif dispatch.

        This tests that the actual dispatch pattern used in GateExecutor works
        correctly with the enum values.
        """
        actions = [
            RoutingAction.continue_(),
            RoutingAction.route("sink"),
            RoutingAction.fork_to_paths(["a", "b"]),
        ]

        expected_kinds = [
            RoutingKind.CONTINUE,
            RoutingKind.ROUTE,
            RoutingKind.FORK_TO_PATHS,
        ]

        for action, expected_kind in zip(actions, expected_kinds, strict=True):
            # Test the dispatch pattern used in engine
            if action.kind == RoutingKind.CONTINUE:
                matched_kind: RoutingKind | None = RoutingKind.CONTINUE
            elif action.kind == RoutingKind.ROUTE:
                matched_kind = RoutingKind.ROUTE
            elif action.kind == RoutingKind.FORK_TO_PATHS:
                matched_kind = RoutingKind.FORK_TO_PATHS
            else:
                # Exhaustive match - unreachable if all cases handled
                matched_kind = None  # type: ignore[unreachable]

            assert matched_kind is expected_kind
