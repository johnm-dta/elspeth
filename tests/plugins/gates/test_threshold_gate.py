"""Tests for ThresholdGate."""

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
            "above_sink": "high_scores",
            "below_sink": "low_scores",
        })
        assert isinstance(gate, GateProtocol)

    def test_has_required_attributes(self) -> None:
        """ThresholdGate has name and schemas."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        assert ThresholdGate.name == "threshold_gate"
        assert hasattr(ThresholdGate, "input_schema")
        assert hasattr(ThresholdGate, "output_schema")

    def test_route_above_threshold(self, ctx: PluginContext) -> None:
        """Route to above_sink when value > threshold."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high_scores",
            "below_sink": "low_scores",
        })
        row = {"id": 1, "score": 75}

        result = gate.evaluate(row, ctx)

        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("high_scores",)
        assert result.row == row

    def test_route_below_threshold(self, ctx: PluginContext) -> None:
        """Route to below_sink when value < threshold."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high_scores",
            "below_sink": "low_scores",
        })
        row = {"id": 1, "score": 25}

        result = gate.evaluate(row, ctx)

        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("low_scores",)

    def test_equal_routes_to_below(self, ctx: PluginContext) -> None:
        """Equal value routes to below_sink by default."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high",
            "below_sink": "low",
        })
        row = {"id": 1, "score": 50}

        result = gate.evaluate(row, ctx)
        assert result.action.destinations == ("low",)

    def test_equal_routes_to_above_when_inclusive(self, ctx: PluginContext) -> None:
        """Equal value routes to above_sink when inclusive=True."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high",
            "below_sink": "low",
            "inclusive": True,  # >= routes to above
        })
        row = {"id": 1, "score": 50}

        result = gate.evaluate(row, ctx)
        assert result.action.destinations == ("high",)

    def test_continue_when_no_below_sink(self, ctx: PluginContext) -> None:
        """Continue to next transform when below_sink not specified."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high_scores",
            # No below_sink - continue to next transform
        })
        row = {"id": 1, "score": 25}

        result = gate.evaluate(row, ctx)

        assert result.action.kind == "continue"
        assert result.row == row

    def test_continue_when_no_above_sink(self, ctx: PluginContext) -> None:
        """Continue to next transform when above_sink not specified."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "below_sink": "low_scores",
            # No above_sink - continue to next transform
        })
        row = {"id": 1, "score": 75}

        result = gate.evaluate(row, ctx)

        assert result.action.kind == "continue"

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Access nested field with dot notation."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "metrics.score",
            "threshold": 50,
            "above_sink": "high",
            "below_sink": "low",
        })
        row = {"id": 1, "metrics": {"score": 75}}

        result = gate.evaluate(row, ctx)
        assert result.action.destinations == ("high",)

    def test_missing_field_raises_error(self, ctx: PluginContext) -> None:
        """Error when required field is missing."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high",
        })
        row = {"id": 1}  # No score field

        with pytest.raises(ValueError, match="score"):
            gate.evaluate(row, ctx)

    def test_non_numeric_field_raises_error(self, ctx: PluginContext) -> None:
        """Error when field is not numeric."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "name",
            "threshold": 50,
            "above_sink": "high",
        })
        row = {"id": 1, "name": "alice"}

        with pytest.raises(TypeError, match="numeric"):
            gate.evaluate(row, ctx)

    def test_routing_includes_reason(self, ctx: PluginContext) -> None:
        """RoutingAction includes reason with threshold details."""
        from elspeth.plugins.gates.threshold_gate import ThresholdGate

        gate = ThresholdGate({
            "field": "score",
            "threshold": 50,
            "above_sink": "high",
            "below_sink": "low",
        })
        row = {"id": 1, "score": 75}

        result = gate.evaluate(row, ctx)

        # Reason should explain why the routing decision was made
        assert "threshold" in result.action.reason
        assert result.action.reason["value"] == 75
