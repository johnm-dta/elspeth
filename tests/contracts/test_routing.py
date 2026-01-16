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
            reason={},
        )
        assert hasattr(action, "mode")
        assert action.mode == RoutingMode.MOVE

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

        action = RoutingAction.continue_(reason={"rule": "passed"})
        assert dict(action.reason) == {"rule": "passed"}

    def test_route_default_move(self) -> None:
        """route defaults to MOVE mode."""
        from elspeth.contracts import RoutingAction, RoutingKind, RoutingMode

        action = RoutingAction.route("above")
        assert action.kind == RoutingKind.ROUTE
        assert action.destinations == ("above",)
        assert action.mode == RoutingMode.MOVE

    def test_route_with_copy(self) -> None:
        """route can specify COPY mode."""
        from elspeth.contracts import RoutingAction, RoutingMode

        action = RoutingAction.route("above", mode=RoutingMode.COPY)
        assert action.mode == RoutingMode.COPY

    def test_route_with_reason(self) -> None:
        """route can include audit reason."""
        from elspeth.contracts import RoutingAction

        action = RoutingAction.route("below", reason={"value": 500})
        assert dict(action.reason) == {"value": 500}

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
            ["a", "b"], reason={"strategy": "parallel"}
        )
        assert dict(action.reason) == {"strategy": "parallel"}

    def test_reason_is_immutable(self) -> None:
        """reason field should be immutable MappingProxyType."""
        from types import MappingProxyType

        from elspeth.contracts import RoutingAction

        action = RoutingAction.continue_(reason={"key": "value"})
        assert isinstance(action.reason, MappingProxyType)

        with pytest.raises(TypeError):
            action.reason["new_key"] = "new_value"  # type: ignore[index]

    def test_reason_deep_copied(self) -> None:
        """Mutating original dict should not affect frozen reason."""
        from elspeth.contracts import RoutingAction

        original = {"nested": {"key": "value"}}
        action = RoutingAction.continue_(reason=original)

        # Mutate original
        original["nested"]["key"] = "modified"

        # Frozen reason should be unchanged
        assert action.reason["nested"]["key"] == "value"

    def test_frozen(self) -> None:
        """RoutingAction should be immutable."""
        from elspeth.contracts import RoutingAction

        action = RoutingAction.continue_()
        with pytest.raises(AttributeError):
            action.kind = "other"  # type: ignore[misc]


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
