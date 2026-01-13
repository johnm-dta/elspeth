"""Tests for ThresholdGate.

ThresholdGate returns route LABELS ("above"/"below"), not sink names.
The routes config in settings.yaml maps labels to sinks.
"""

import pytest

from elspeth.plugins.context import PluginContext
from elspeth.plugins.protocols import GateProtocol


class TestThresholdGate:
    """Tests for ThresholdGate plugin."""

    @pytest.fixture
    def ctx(self) -> PluginContext:
        """Create minimal plugin context."""
        return PluginContext(run_id="test-run", config={})

    def test_implements_protocol(self) -> None:
        """ThresholdGate implements GateProtocol."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
        })
        assert isinstance(gate, GateProtocol)

    def test_has_required_attributes(self) -> None:
        """ThresholdGate has name and schemas."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        assert ThresholdGate.name == "threshold_gate"
        assert hasattr(ThresholdGate, "input_schema")
        assert hasattr(ThresholdGate, "output_schema")

    def test_route_above_threshold(self, ctx: PluginContext) -> None:
        """Route to 'above' label when value > threshold."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
        })
        row = {"id": 1, "score": 75}

        result = gate.evaluate(row, ctx)

        assert result.action.kind == "route"
        assert result.action.destinations == ("above",)  # Label, not sink
        assert result.row == row

    def test_route_below_threshold(self, ctx: PluginContext) -> None:
        """Route to 'below' label when value < threshold."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
        })
        row = {"id": 1, "score": 25}

        result = gate.evaluate(row, ctx)

        assert result.action.kind == "route"
        assert result.action.destinations == ("below",)  # Label, not sink

    def test_equal_routes_to_below(self, ctx: PluginContext) -> None:
        """Equal value routes to 'below' label by default."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
        })
        row = {"id": 1, "score": 50}

        result = gate.evaluate(row, ctx)
        assert result.action.destinations == ("below",)

    def test_equal_routes_to_above_when_inclusive(self, ctx: PluginContext) -> None:
        """Equal value routes to 'above' label when inclusive=True."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "inclusive": True,  # >= routes to above
        })
        row = {"id": 1, "score": 50}

        result = gate.evaluate(row, ctx)
        assert result.action.destinations == ("above",)

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Access nested field with dot notation."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "metrics.score",
            "threshold": 50,
        })
        row = {"id": 1, "metrics": {"score": 75}}

        result = gate.evaluate(row, ctx)
        assert result.action.destinations == ("above",)

    def test_missing_field_raises_error(self, ctx: PluginContext) -> None:
        """Error when required field is missing."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
        })
        row = {"id": 1}  # No score field

        with pytest.raises(ValueError, match="score"):
            gate.evaluate(row, ctx)

    def test_non_numeric_string_casts_when_enabled(self, ctx: PluginContext) -> None:
        """String values are cast to float when cast=True (default)."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "cast": True,  # Default
        })
        row = {"id": 1, "score": "75"}

        result = gate.evaluate(row, ctx)
        assert result.action.destinations == ("above",)

    def test_non_numeric_string_raises_when_cast_disabled(
        self, ctx: PluginContext
    ) -> None:
        """Error when field is string and cast=False."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "cast": False,
        })
        row = {"id": 1, "score": "75"}

        with pytest.raises(TypeError, match="numeric"):
            gate.evaluate(row, ctx)

    def test_routing_includes_reason(self, ctx: PluginContext) -> None:
        """RoutingAction includes reason with threshold details."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
        })
        row = {"id": 1, "score": 75}

        result = gate.evaluate(row, ctx)

        # Reason should explain why the routing decision was made
        assert "threshold" in result.action.reason
        assert result.action.reason["value"] == 75
