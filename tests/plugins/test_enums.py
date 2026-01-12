# tests/plugins/test_enums.py
"""Tests for plugin type enums."""


class TestNodeType:
    """Node type enumeration."""

    def test_all_node_types_defined(self) -> None:
        from elspeth.plugins.enums import NodeType

        assert NodeType.SOURCE.value == "source"
        assert NodeType.TRANSFORM.value == "transform"
        assert NodeType.GATE.value == "gate"
        assert NodeType.AGGREGATION.value == "aggregation"
        assert NodeType.COALESCE.value == "coalesce"
        assert NodeType.SINK.value == "sink"

    def test_node_type_is_string_enum(self) -> None:
        """NodeType should be usable as a string in comparisons."""
        from elspeth.plugins.enums import NodeType

        # str(Enum) with str base allows direct comparison
        # Use .value for explicit type safety with mypy --strict
        assert NodeType.SOURCE.value == "source"
        # For f-strings, use .value explicitly
        assert f"type:{NodeType.TRANSFORM.value}" == "type:transform"


class TestRoutingKind:
    """Routing decision kinds."""

    def test_all_routing_kinds_defined(self) -> None:
        from elspeth.plugins.enums import RoutingKind

        assert RoutingKind.CONTINUE.value == "continue"
        assert RoutingKind.ROUTE_TO_SINK.value == "route_to_sink"
        assert RoutingKind.FORK_TO_PATHS.value == "fork_to_paths"


class TestRoutingMode:
    """Token routing modes."""

    def test_routing_modes_defined(self) -> None:
        from elspeth.plugins.enums import RoutingMode

        assert RoutingMode.MOVE.value == "move"
        assert RoutingMode.COPY.value == "copy"


class TestDeterminism:
    """Plugin determinism classification."""

    def test_determinism_levels_defined(self) -> None:
        from elspeth.plugins.enums import Determinism

        assert Determinism.DETERMINISTIC.value == "deterministic"
        assert Determinism.SEEDED.value == "seeded"
        assert Determinism.NONDETERMINISTIC.value == "nondeterministic"
