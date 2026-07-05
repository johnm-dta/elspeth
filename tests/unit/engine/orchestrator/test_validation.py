# tests/unit/engine/orchestrator/test_validation.py
"""Tests for pipeline configuration validation functions.

These functions run at pipeline initialization (before any rows process)
to catch config errors early. The tests verify that invalid route
destinations, transform on_error sinks, and source quarantine destinations
are rejected with clear error messages.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from elspeth.contracts import RouteDestination
from elspeth.contracts.types import GateName, NodeID, SinkName
from elspeth.engine.orchestrator.types import RouteValidationError
from elspeth.engine.orchestrator.validation import (
    validate_route_destinations,
    validate_source_quarantine_destination,
    validate_transform_error_sinks,
)

# =============================================================================
# Helpers
# =============================================================================


@dataclass
class FakeTransform:
    node_id: str
    name: str
    on_error: str | None = None


@dataclass
class FakeSource:
    name: str = "csv-source"
    _on_validation_failure: str = "discard"


@dataclass
class FakeConfigGate:
    name: str


def _make_transform(*, node_id: str, name: str, on_error: str | None = None) -> FakeTransform:
    return FakeTransform(node_id=node_id, name=name, on_error=on_error)


def _make_source(*, name: str = "csv-source", on_validation_failure: str = "discard") -> FakeSource:
    return FakeSource(name=name, _on_validation_failure=on_validation_failure)


# =============================================================================
# validate_route_destinations
# =============================================================================


class TestValidateRouteDestinations:
    """Tests for validate_route_destinations()."""

    def test_config_gates_validated(self) -> None:
        """Config-driven gates are also validated via config_gate_id_map."""
        config_gate = FakeConfigGate(name="config_gate")
        gate_id_map = {GateName("config_gate"): NodeID("cfg-gate-1")}
        route_map = {(NodeID("cfg-gate-1"), "route_a"): RouteDestination.sink(SinkName("missing_sink"))}
        sinks = {"output"}

        with pytest.raises(RouteValidationError, match=r"config_gate.*missing_sink"):
            validate_route_destinations(
                route_resolution_map=route_map,
                available_sinks=sinks,
                transform_id_map={},
                transforms=[],
                config_gate_id_map=gate_id_map,
                config_gates=[config_gate],
            )

    def test_empty_route_map_passes(self) -> None:
        """No routes to validate means no errors."""
        validate_route_destinations(
            route_resolution_map={},
            available_sinks=set(),
            transform_id_map={},
            transforms=[],
        )

    def test_gate_discard_route_passes_without_matching_sink(self) -> None:
        """Gate route target 'discard' is a virtual terminal destination, not a sink."""
        config_gate = FakeConfigGate(name="quality_gate")
        gate_id_map = {GateName("quality_gate"): NodeID("cfg-gate-1")}
        route_map = {(NodeID("cfg-gate-1"), "false"): RouteDestination.discard()}

        validate_route_destinations(
            route_resolution_map=route_map,
            available_sinks={"output"},
            transform_id_map={},
            transforms=[],
            config_gate_id_map=gate_id_map,
            config_gates=[config_gate],
        )


# =============================================================================
# validate_transform_error_sinks
# =============================================================================


class TestValidateTransformErrorSinks:
    """Tests for validate_transform_error_sinks()."""

    def test_discard_on_error_passes_with_no_sinks(self) -> None:
        """Transform with on_error='discard' passes even with no sinks defined.

        on_error is now required at config time (TransformSettings), so None
        never reaches the validator. This test verifies 'discard' (the minimum
        valid value) works regardless of available sinks.
        """
        transform = _make_transform(node_id="t-1", name="mapper", on_error="discard")
        validate_transform_error_sinks([transform], set())

    def test_discard_passes(self) -> None:
        """on_error='discard' is a special value, not a sink."""
        transform = _make_transform(node_id="t-1", name="mapper", on_error="discard")
        validate_transform_error_sinks([transform], set())

    def test_valid_error_sink_passes(self) -> None:
        """on_error referencing existing sink passes."""
        transform = _make_transform(node_id="t-1", name="mapper", on_error="error_sink")
        validate_transform_error_sinks([transform], {"output", "error_sink"})

    def test_invalid_error_sink_raises(self) -> None:
        """on_error referencing non-existent sink raises."""
        transform = _make_transform(node_id="t-1", name="classifier", on_error="missing")
        with pytest.raises(RouteValidationError, match=r"classifier.*missing"):
            validate_transform_error_sinks([transform], {"output"})

    def test_error_message_includes_available_sinks(self) -> None:
        """Error message lists available sinks."""
        transform = _make_transform(node_id="t-1", name="t", on_error="bad")
        with pytest.raises(RouteValidationError, match="output"):
            validate_transform_error_sinks([transform], {"output"})

    def test_multiple_transforms_all_validated(self) -> None:
        """All transforms are checked, not just the first."""
        t1 = _make_transform(node_id="t-1", name="first", on_error="output")
        t2 = _make_transform(node_id="t-2", name="second", on_error="missing")

        with pytest.raises(RouteValidationError, match=r"second.*missing"):
            validate_transform_error_sinks([t1, t2], {"output"})

    def test_empty_transforms_passes(self) -> None:
        """Empty transform list means nothing to validate."""
        validate_transform_error_sinks([], {"output"})


# =============================================================================
# validate_source_quarantine_destination
# =============================================================================


class TestValidateSourceQuarantineDestination:
    """Tests for validate_source_quarantine_destination()."""

    def test_discard_passes(self) -> None:
        """on_validation_failure='discard' is always valid."""
        source = _make_source(on_validation_failure="discard")
        validate_source_quarantine_destination(source, {"output"})

    def test_valid_quarantine_sink_passes(self) -> None:
        """on_validation_failure referencing existing sink passes."""
        source = _make_source(on_validation_failure="quarantine_sink")
        validate_source_quarantine_destination(source, {"output", "quarantine_sink"})

    def test_invalid_quarantine_sink_raises(self) -> None:
        """on_validation_failure referencing non-existent sink raises."""
        source = _make_source(name="csv-source", on_validation_failure="missing")
        with pytest.raises(RouteValidationError, match=r"csv-source.*missing"):
            validate_source_quarantine_destination(source, {"output"})

    def test_error_message_includes_available_sinks(self) -> None:
        """Error message lists available sinks for debugging."""
        source = _make_source(on_validation_failure="bad")
        with pytest.raises(RouteValidationError, match="output"):
            validate_source_quarantine_destination(source, {"output"})

    def test_error_message_suggests_discard(self) -> None:
        """Error message suggests 'discard' as alternative."""
        source = _make_source(on_validation_failure="bad")
        with pytest.raises(RouteValidationError, match="discard"):
            validate_source_quarantine_destination(source, {"output"})
