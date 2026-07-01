# tests/unit/contracts/test_routing.py
"""Tests for routing contracts."""

from typing import Any

import pytest

from elspeth.contracts import (
    EdgeInfo,
    NodeID,
    RouteDestination,
    RouteDestinationKind,
    RoutingAction,
    RoutingKind,
    RoutingMode,
    RoutingSpec,
    SinkName,
)
from elspeth.contracts.errors import ConfigGateReason


class TestRoutingAction:
    """Tests for RoutingAction dataclass."""

    def test_continue_with_reason(self) -> None:
        """continue_ can include audit reason."""
        reason = ConfigGateReason(condition="passed", result="true")
        action = RoutingAction.continue_(reason=reason)
        assert action.reason is not None
        assert action.reason["condition"] == "passed"  # type: ignore[typeddict-item]

    def test_route_default_move(self) -> None:
        """route defaults to MOVE mode."""
        action = RoutingAction.route("above")
        assert action.kind == RoutingKind.ROUTE
        assert action.destinations == ("above",)
        assert action.mode == RoutingMode.MOVE

    def test_route_with_copy_raises(self) -> None:
        """route with COPY mode raises ValueError (architectural limitation).

        COPY mode is only valid for FORK_TO_PATHS because it creates child tokens,
        each with their own terminal state. ROUTE with COPY would require a single
        token to have dual terminal states (ROUTED + COMPLETED), which violates
        ELSPETH's single-terminal-state audit model.

        Users should use fork_to_paths() to route to a sink and continue processing.
        """
        with pytest.raises(
            ValueError,
            match=r"COPY mode not supported for ROUTE kind.*Use FORK_TO_PATHS",
        ):
            RoutingAction.route("above", mode=RoutingMode.COPY)

    def test_route_with_string_mode_raises_type_error(self) -> None:
        """route requires RoutingMode enum, not a raw string."""
        with pytest.raises(TypeError, match="mode must be RoutingMode"):
            RoutingAction.route("above", mode="move")  # type: ignore[arg-type]

    def test_route_with_reason(self) -> None:
        """route can include audit reason."""
        reason = ConfigGateReason(condition="value below threshold", result="below")
        action = RoutingAction.route(
            "below",
            reason=reason,
        )
        assert action.reason is not None
        assert action.reason["condition"] == "value below threshold"  # type: ignore[typeddict-item]

    def test_fork_always_copy(self) -> None:
        """fork_to_paths always uses COPY mode."""
        action = RoutingAction.fork_to_paths(["path_a", "path_b"])
        assert action.kind == RoutingKind.FORK_TO_PATHS
        assert action.destinations == ("path_a", "path_b")
        assert action.mode == RoutingMode.COPY

    def test_fork_with_reason(self) -> None:
        """fork_to_paths can include audit reason."""
        reason = ConfigGateReason(condition="parallel_strategy", result="split")
        action = RoutingAction.fork_to_paths(
            ["a", "b"],
            reason=reason,
        )
        assert action.reason is not None
        assert action.reason["condition"] == "parallel_strategy"  # type: ignore[typeddict-item]

    def test_reason_mutation_prevented_by_deep_copy(self) -> None:
        """Mutating original dict should not affect stored reason (deep copy)."""
        original: dict[str, Any] = {"rule": "test", "matched_value": 42}
        action = RoutingAction.continue_(reason=original)  # type: ignore[arg-type]

        # Mutate original - should not affect action.reason
        original["rule"] = "mutated"
        assert action.reason is not None
        assert action.reason["rule"] == "test"  # type: ignore[typeddict-item]

    def test_reason_deep_copied(self) -> None:
        """Mutating original nested dict should not affect frozen reason."""
        # Use nested dict in matched_value (which accepts Any)
        original: dict[str, Any] = {"rule": "test", "matched_value": {"nested": "value"}}
        action = RoutingAction.continue_(reason=original)  # type: ignore[arg-type]

        # Mutate original nested structure
        original["matched_value"]["nested"] = "modified"

        # Frozen reason should be unchanged
        assert action.reason is not None
        assert action.reason["matched_value"]["nested"] == "value"  # type: ignore[typeddict-item]

    def test_fork_to_paths_rejects_empty_list(self) -> None:
        """fork_to_paths must have at least one destination.

        Per CLAUDE.md "no silent drops" invariant, empty forks would cause
        tokens to disappear without audit trail. This MUST raise immediately.
        """
        with pytest.raises(ValueError, match="at least one destination"):
            RoutingAction.fork_to_paths([])

    def test_fork_to_paths_rejects_duplicate_paths(self) -> None:
        """fork_to_paths must have unique path names.

        Duplicate paths would cause ambiguous routing and audit integrity issues.
        """
        with pytest.raises(ValueError, match=r"unique path names.*duplicates"):
            RoutingAction.fork_to_paths(["path_a", "path_a", "path_b"])

    def test_continue_with_destinations_raises(self) -> None:
        """CONTINUE kind with non-empty destinations raises ValueError.

        CONTINUE means "proceed to next node" - destinations are resolved
        from the pipeline graph, not specified in the action.
        """
        with pytest.raises(ValueError, match="CONTINUE must have empty destinations"):
            RoutingAction(
                kind=RoutingKind.CONTINUE,
                destinations=("sink_a",),
                mode=RoutingMode.MOVE,
            )

    def test_continue_with_copy_mode_raises(self) -> None:
        """CONTINUE kind with COPY mode raises ValueError.

        Bug: P3-2026-01-31-routing-action-continue-copy-allowed

        COPY mode is ONLY valid for FORK_TO_PATHS because it creates child tokens.
        CONTINUE simply advances to the next node - no token cloning occurs.
        """
        with pytest.raises(ValueError, match="CONTINUE must use MOVE mode"):
            RoutingAction(
                kind=RoutingKind.CONTINUE,
                destinations=(),
                mode=RoutingMode.COPY,
            )

    def test_continue_with_divert_mode_raises(self) -> None:
        """CONTINUE kind must use MOVE mode, not DIVERT."""
        with pytest.raises(ValueError, match="CONTINUE must use MOVE mode"):
            RoutingAction(
                kind=RoutingKind.CONTINUE,
                destinations=(),
                mode=RoutingMode.DIVERT,
            )

    def test_fork_to_paths_with_move_mode_raises(self) -> None:
        """FORK_TO_PATHS kind with MOVE mode raises ValueError.

        FORK creates child tokens - MOVE would violate the fork semantics
        by destroying the parent token prematurely.
        """
        with pytest.raises(ValueError, match="FORK_TO_PATHS must use COPY mode"):
            RoutingAction(
                kind=RoutingKind.FORK_TO_PATHS,
                destinations=("path_a", "path_b"),
                mode=RoutingMode.MOVE,
            )

    def test_fork_to_paths_with_empty_destinations_raises(self) -> None:
        """FORK_TO_PATHS direct construction must reject empty destinations."""
        with pytest.raises(ValueError, match="at least one destination"):
            RoutingAction(
                kind=RoutingKind.FORK_TO_PATHS,
                destinations=(),
                mode=RoutingMode.COPY,
            )

    def test_route_with_zero_destinations_raises(self) -> None:
        """ROUTE kind with zero destinations raises ValueError.

        ROUTE must specify exactly one destination - zero destinations
        would cause token to silently drop without audit trail.
        """
        with pytest.raises(ValueError, match="ROUTE must have exactly one destination"):
            RoutingAction(
                kind=RoutingKind.ROUTE,
                destinations=(),
                mode=RoutingMode.MOVE,
            )

    def test_route_with_multiple_destinations_raises(self) -> None:
        """ROUTE kind with multiple destinations raises ValueError.

        ROUTE is single-destination routing. For multi-destination,
        use FORK_TO_PATHS which creates separate token lineages.
        """
        with pytest.raises(ValueError, match="ROUTE must have exactly one destination"):
            RoutingAction(
                kind=RoutingKind.ROUTE,
                destinations=("sink_a", "sink_b"),
                mode=RoutingMode.MOVE,
            )

    def test_constructor_with_string_mode_raises_type_error(self) -> None:
        """Direct constructor requires enum mode."""
        with pytest.raises(TypeError, match="mode must be RoutingMode"):
            RoutingAction(
                kind=RoutingKind.ROUTE,
                destinations=("sink_a",),
                mode="move",  # type: ignore[arg-type]
            )


class TestRouteDestination:
    """Tests for RouteDestination dataclass invariants."""

    def test_sink_destination_requires_sink_name(self) -> None:
        """SINK destination must include sink_name."""
        with pytest.raises(ValueError, match="requires non-empty sink_name"):
            RouteDestination(kind=RouteDestinationKind.SINK)

    def test_processing_destination_requires_next_node_id(self) -> None:
        """PROCESSING_NODE destination must include next_node_id."""
        with pytest.raises(ValueError, match="requires non-empty next_node_id"):
            RouteDestination(kind=RouteDestinationKind.PROCESSING_NODE)

    def test_continue_destination_rejects_payload_fields(self) -> None:
        """CONTINUE destination cannot carry sink or node payloads."""
        with pytest.raises(ValueError, match="must not include sink_name or next_node_id"):
            RouteDestination(kind=RouteDestinationKind.CONTINUE, sink_name=SinkName("out"))

        with pytest.raises(ValueError, match="must not include sink_name or next_node_id"):
            RouteDestination(kind=RouteDestinationKind.CONTINUE, next_node_id=NodeID("transform-1"))

    def test_processing_destination_rejects_sink_name(self) -> None:
        """PROCESSING_NODE destination cannot carry sink_name."""
        with pytest.raises(ValueError, match="must not include sink_name"):
            RouteDestination(
                kind=RouteDestinationKind.PROCESSING_NODE,
                sink_name=SinkName("out"),
                next_node_id=NodeID("transform-1"),
            )


class TestRoutingSpec:
    """Tests for RoutingSpec dataclass."""

    def test_parse_from_string(self) -> None:
        """RoutingSpec parses mode from string value."""
        spec = RoutingSpec(edge_id="edge-1", mode=RoutingMode.COPY)
        assert spec.mode == RoutingMode.COPY
        assert isinstance(spec.mode, RoutingMode)


class TestEdgeInfo:
    """Tests for EdgeInfo dataclass."""

    def test_edge_info_label(self) -> None:
        """EdgeInfo preserves label."""
        edge = EdgeInfo(
            from_node=NodeID("gate-1"),
            to_node=NodeID("sink-1"),
            label="above",
            mode=RoutingMode.MOVE,
        )
        assert edge.label == "above"
