# tests/contracts/test_routing.py
"""Tests for routing contracts."""

import pytest


class TestRoutingAction:
    """Tests for RoutingAction dataclass."""

    def test_has_mode_field(self) -> None:
        """RoutingAction MUST have mode field per architecture."""
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

        action = RoutingAction(
            kind=RoutingKind.ROUTE,
            destinations=("sink_a",),
            mode=RoutingMode.MOVE,
            reason={"rule": "test", "matched_value": None},
        )
        assert action.mode is RoutingMode.MOVE

    def test_continue_action(self) -> None:
        """continue_ creates action with MOVE mode and no destinations."""
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

        action = RoutingAction.continue_()
        assert action.kind == RoutingKind.CONTINUE
        assert action.destinations == ()
        assert action.mode == RoutingMode.MOVE

    def test_continue_with_reason(self) -> None:
        """continue_ can include audit reason."""
        from elspeth.contracts import RoutingAction

        action = RoutingAction.continue_(reason={"rule": "passed", "matched_value": True})
        assert action.reason is not None
        assert action.reason["rule"] == "passed"  # type: ignore[typeddict-item]

    def test_route_default_move(self) -> None:
        """route defaults to MOVE mode."""
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

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
        from elspeth.contracts import RoutingAction, RoutingMode

        with pytest.raises(
            ValueError,
            match=r"COPY mode not supported for ROUTE kind.*Use FORK_TO_PATHS",
        ):
            RoutingAction.route("above", mode=RoutingMode.COPY)

    def test_route_with_reason(self) -> None:
        """route can include audit reason."""
        from elspeth.contracts import RoutingAction

        action = RoutingAction.route(
            "below",
            reason={
                "rule": "value below threshold",
                "matched_value": 500,
            },
        )
        assert action.reason is not None
        assert action.reason["matched_value"] == 500  # type: ignore[typeddict-item]

    def test_fork_always_copy(self) -> None:
        """fork_to_paths always uses COPY mode."""
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

        action = RoutingAction.fork_to_paths(["path_a", "path_b"])
        assert action.kind == RoutingKind.FORK_TO_PATHS
        assert action.destinations == ("path_a", "path_b")
        assert action.mode == RoutingMode.COPY

    def test_fork_with_reason(self) -> None:
        """fork_to_paths can include audit reason."""
        from elspeth.contracts import RoutingAction

        action = RoutingAction.fork_to_paths(
            ["a", "b"],
            reason={
                "rule": "parallel_strategy",
                "matched_value": "split",
            },
        )
        assert action.reason is not None
        assert action.reason["rule"] == "parallel_strategy"  # type: ignore[typeddict-item]

    def test_reason_mutation_prevented_by_deep_copy(self) -> None:
        """Mutating original dict should not affect stored reason (deep copy)."""
        from typing import Any

        from elspeth.contracts import RoutingAction

        original: dict[str, Any] = {"rule": "test", "matched_value": 42}
        action = RoutingAction.continue_(reason=original)  # type: ignore[arg-type]

        # Mutate original - should not affect action.reason
        original["rule"] = "mutated"
        assert action.reason is not None
        assert action.reason["rule"] == "test"  # type: ignore[typeddict-item]

    def test_reason_deep_copied(self) -> None:
        """Mutating original nested dict should not affect frozen reason."""
        from typing import Any

        from elspeth.contracts import RoutingAction

        # Use nested dict in matched_value (which accepts Any)
        original: dict[str, Any] = {"rule": "test", "matched_value": {"nested": "value"}}
        action = RoutingAction.continue_(reason=original)  # type: ignore[arg-type]

        # Mutate original nested structure
        original["matched_value"]["nested"] = "modified"

        # Frozen reason should be unchanged
        assert action.reason is not None
        assert action.reason["matched_value"]["nested"] == "value"  # type: ignore[typeddict-item]

    def test_frozen(self) -> None:
        """RoutingAction should be immutable."""
        from elspeth.contracts import RoutingAction

        action = RoutingAction.continue_()
        with pytest.raises(AttributeError):
            action.kind = "other"  # type: ignore[misc,assignment]  # Testing frozen

    def test_fork_to_paths_rejects_empty_list(self) -> None:
        """fork_to_paths must have at least one destination.

        Per CLAUDE.md "no silent drops" invariant, empty forks would cause
        tokens to disappear without audit trail. This MUST raise immediately.
        """
        from elspeth.contracts import RoutingAction

        with pytest.raises(ValueError, match="at least one destination"):
            RoutingAction.fork_to_paths([])

    def test_fork_to_paths_rejects_duplicate_paths(self) -> None:
        """fork_to_paths must have unique path names.

        Duplicate paths would cause ambiguous routing and audit integrity issues.
        """
        from elspeth.contracts import RoutingAction

        with pytest.raises(ValueError, match=r"unique path names.*duplicates"):
            RoutingAction.fork_to_paths(["path_a", "path_a", "path_b"])

    def test_continue_with_destinations_raises(self) -> None:
        """CONTINUE kind with non-empty destinations raises ValueError.

        CONTINUE means "proceed to next node" - destinations are resolved
        from the pipeline graph, not specified in the action.
        """
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

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
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

        with pytest.raises(ValueError, match="CONTINUE must use MOVE mode"):
            RoutingAction(
                kind=RoutingKind.CONTINUE,
                destinations=(),
                mode=RoutingMode.COPY,
            )

    def test_fork_to_paths_with_move_mode_raises(self) -> None:
        """FORK_TO_PATHS kind with MOVE mode raises ValueError.

        FORK creates child tokens - MOVE would violate the fork semantics
        by destroying the parent token prematurely.
        """
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

        with pytest.raises(ValueError, match="FORK_TO_PATHS must use COPY mode"):
            RoutingAction(
                kind=RoutingKind.FORK_TO_PATHS,
                destinations=("path_a", "path_b"),
                mode=RoutingMode.MOVE,
            )

    def test_route_with_zero_destinations_raises(self) -> None:
        """ROUTE kind with zero destinations raises ValueError.

        ROUTE must specify exactly one destination - zero destinations
        would cause token to silently drop without audit trail.
        """
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

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
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

        with pytest.raises(ValueError, match="ROUTE must have exactly one destination"):
            RoutingAction(
                kind=RoutingKind.ROUTE,
                destinations=("sink_a", "sink_b"),
                mode=RoutingMode.MOVE,
            )


class TestRoutingSpec:
    """Tests for RoutingSpec dataclass."""

    def test_correct_usage_with_enum(self) -> None:
        """Document correct usage - repository layer converts before construction.

        Per Data Manifesto: dataclasses are strict contracts. Type enforcement
        happens at static analysis time (mypy), not runtime. The repository
        layer must convert: RoutingSpec(edge_id=row.edge_id, mode=RoutingMode(row.mode))
        """
        from elspeth.contracts import RoutingMode, RoutingSpec

        # Correct: repository converts DB string to enum before constructing
        spec = RoutingSpec(edge_id="edge-1", mode=RoutingMode.COPY)
        assert spec.mode == RoutingMode.COPY
        assert isinstance(spec.mode, RoutingMode)

    def test_create_with_move(self) -> None:
        """RoutingSpec can be created with MOVE mode."""
        from elspeth.contracts import RoutingMode, RoutingSpec

        spec = RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE)
        assert spec.edge_id == "edge-1"
        assert spec.mode == RoutingMode.MOVE

    def test_create_with_copy(self) -> None:
        """RoutingSpec can be created with COPY mode."""
        from elspeth.contracts import RoutingMode, RoutingSpec

        spec = RoutingSpec(edge_id="edge-2", mode=RoutingMode.COPY)
        assert spec.mode == RoutingMode.COPY

    def test_frozen(self) -> None:
        """RoutingSpec should be immutable."""
        from elspeth.contracts import RoutingMode, RoutingSpec

        spec = RoutingSpec(edge_id="edge-1", mode=RoutingMode.MOVE)
        with pytest.raises(AttributeError):
            spec.edge_id = "changed"  # type: ignore[misc]


class TestEdgeInfo:
    """Tests for EdgeInfo dataclass."""

    def test_create_edge_info(self) -> None:
        """EdgeInfo can be created with all required fields."""
        from elspeth.contracts import EdgeInfo, RoutingMode

        edge = EdgeInfo(
            from_node="gate-1",
            to_node="sink-1",
            label="above",
            mode=RoutingMode.MOVE,
        )
        assert edge.from_node == "gate-1"
        assert edge.to_node == "sink-1"
        assert edge.label == "above"
        assert edge.mode == RoutingMode.MOVE

    def test_edge_info_with_copy(self) -> None:
        """EdgeInfo supports COPY mode."""
        from elspeth.contracts import EdgeInfo, RoutingMode

        edge = EdgeInfo(
            from_node="gate-1",
            to_node="sink-1",
            label="fork_path",
            mode=RoutingMode.COPY,
        )
        assert edge.mode == RoutingMode.COPY

    def test_frozen(self) -> None:
        """EdgeInfo should be immutable."""
        from elspeth.contracts import EdgeInfo, RoutingMode

        edge = EdgeInfo(
            from_node="gate-1",
            to_node="sink-1",
            label="above",
            mode=RoutingMode.MOVE,
        )
        with pytest.raises(AttributeError):
            edge.from_node = "changed"  # type: ignore[misc]
