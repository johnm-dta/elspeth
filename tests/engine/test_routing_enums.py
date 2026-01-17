"""Tests for enum-based routing comparisons."""

from elspeth.contracts.enums import RoutingKind


class TestRoutingKindUsage:
    """Verify RoutingKind enum is used for comparisons."""

    def test_routing_kind_continue(self) -> None:
        """CONTINUE should be comparable to action.kind."""
        kind = RoutingKind.CONTINUE
        assert kind == RoutingKind.CONTINUE
        assert kind.value == "continue"

    def test_routing_kind_route(self) -> None:
        """ROUTE should be comparable to action.kind."""
        kind = RoutingKind.ROUTE
        assert kind == RoutingKind.ROUTE
        assert kind.value == "route"

    def test_routing_kind_fork(self) -> None:
        """FORK_TO_PATHS should be comparable to action.kind."""
        kind = RoutingKind.FORK_TO_PATHS
        assert kind == RoutingKind.FORK_TO_PATHS
        assert kind.value == "fork_to_paths"

    def test_routing_action_uses_enum(self) -> None:
        """RoutingAction.kind should return RoutingKind enum."""
        from elspeth.contracts import RoutingAction

        action = RoutingAction.continue_()
        assert isinstance(action.kind, RoutingKind)
        assert action.kind is RoutingKind.CONTINUE

        action = RoutingAction.route("sink1")
        assert isinstance(action.kind, RoutingKind)
        assert action.kind is RoutingKind.ROUTE
