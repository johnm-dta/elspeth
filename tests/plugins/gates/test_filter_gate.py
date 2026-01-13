"""Tests for FilterGate plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from elspeth.plugins.context import PluginContext


@pytest.fixture
def ctx() -> "PluginContext":
    """Create minimal plugin context."""
    from elspeth.plugins.context import PluginContext

    return PluginContext(
        run_id="test-run",
        config={},
        landscape=MagicMock(),
    )


class TestFilterGate:
    """Test FilterGate routes instead of silent drops."""

    def test_passing_row_continues(self, ctx: "PluginContext") -> None:
        """Row that passes filter continues to next stage."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "score",
            "greater_than": 0.5,
            "discard_sink": "filtered_out",
        })

        result = gate.evaluate({"id": 1, "score": 0.8}, ctx)

        assert result.action.kind == "continue"

    def test_failing_row_routes_to_discard(self, ctx: "PluginContext") -> None:
        """Row that fails filter routes to discard sink."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "score",
            "greater_than": 0.5,
            "discard_sink": "filtered_out",
        })

        result = gate.evaluate({"id": 1, "score": 0.3}, ctx)

        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("filtered_out",)
        assert "filtered" in result.action.reason.get("result", "")

    def test_reason_includes_filter_details(self, ctx: "PluginContext") -> None:
        """Routing reason includes why the row was filtered."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "score",
            "greater_than": 0.5,
            "discard_sink": "trash",
        })

        result = gate.evaluate({"id": 1, "score": 0.3}, ctx)

        assert result.action.reason["field"] == "score"
        assert result.action.reason["value"] == 0.3
        assert result.action.reason["condition"] == "greater_than"
        assert result.action.reason["threshold"] == 0.5

    def test_missing_field_routes_to_discard(self, ctx: "PluginContext") -> None:
        """Row with missing field routes to discard sink by default."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "score",
            "greater_than": 0.5,
            "discard_sink": "filtered_out",
        })

        result = gate.evaluate({"id": 1}, ctx)  # No "score" field

        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("filtered_out",)
        assert "missing" in result.action.reason.get("result", "")

    def test_missing_field_passes_when_allowed(self, ctx: "PluginContext") -> None:
        """Row with missing field continues if allow_missing=True."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "score",
            "greater_than": 0.5,
            "discard_sink": "filtered_out",
            "allow_missing": True,
        })

        result = gate.evaluate({"id": 1}, ctx)  # No "score" field

        assert result.action.kind == "continue"


class TestFilterGateValidation:
    """Test FilterGate validation errors."""

    def test_multiple_operators_raises_error(self) -> None:
        """Specifying multiple operators raises ValueError."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        with pytest.raises(ValueError, match="Multiple comparison operators"):
            FilterGate({
                "field": "score",
                "greater_than": 0.5,
                "less_than": 0.9,
                "discard_sink": "filtered_out",
            })

    def test_no_operator_raises_error(self) -> None:
        """Specifying no operator raises ValueError."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        with pytest.raises(ValueError, match="No comparison operator specified"):
            FilterGate({
                "field": "score",
                "discard_sink": "filtered_out",
            })


class TestFilterGateOperators:
    """Test all comparison operators."""

    @pytest.mark.parametrize(
        ("operator", "threshold", "value", "expected_kind"),
        [
            # less_than
            ("less_than", 0.5, 0.3, "continue"),
            ("less_than", 0.5, 0.5, "route_to_sink"),
            ("less_than", 0.5, 0.8, "route_to_sink"),
            # equals
            ("equals", "active", "active", "continue"),
            ("equals", "active", "inactive", "route_to_sink"),
            ("equals", 100, 100, "continue"),
            ("equals", 100, 99, "route_to_sink"),
            # not_equals
            ("not_equals", "active", "inactive", "continue"),
            ("not_equals", "active", "active", "route_to_sink"),
            # greater_than_or_equal
            ("greater_than_or_equal", 0.5, 0.5, "continue"),
            ("greater_than_or_equal", 0.5, 0.6, "continue"),
            ("greater_than_or_equal", 0.5, 0.4, "route_to_sink"),
            # less_than_or_equal
            ("less_than_or_equal", 0.5, 0.5, "continue"),
            ("less_than_or_equal", 0.5, 0.4, "continue"),
            ("less_than_or_equal", 0.5, 0.6, "route_to_sink"),
        ],
    )
    def test_operator(
        self,
        ctx: "PluginContext",
        operator: str,
        threshold: float | str | int,
        value: float | str | int,
        expected_kind: str,
    ) -> None:
        """Parametrized test for all operators."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "score",
            operator: threshold,
            "discard_sink": "filtered_out",
        })

        result = gate.evaluate({"id": 1, "score": value}, ctx)

        assert result.action.kind == expected_kind


class TestFilterGateNestedFields:
    """Test nested field access via dot notation."""

    def test_nested_field_access(self, ctx: "PluginContext") -> None:
        """Dot notation accesses nested fields."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "metrics.score",
            "greater_than": 0.5,
            "discard_sink": "filtered_out",
        })

        result = gate.evaluate(
            {"id": 1, "metrics": {"score": 0.8}},
            ctx,
        )

        assert result.action.kind == "continue"

    def test_nested_field_routes_when_fails(self, ctx: "PluginContext") -> None:
        """Nested field that fails condition routes to discard sink."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "metrics.score",
            "greater_than": 0.5,
            "discard_sink": "filtered_out",
        })

        result = gate.evaluate(
            {"id": 1, "metrics": {"score": 0.3}},
            ctx,
        )

        assert result.action.kind == "route_to_sink"
        assert result.action.destinations == ("filtered_out",)

    def test_missing_nested_field_routes_to_discard(
        self, ctx: "PluginContext"
    ) -> None:
        """Missing nested field routes to discard sink."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "metrics.score",
            "greater_than": 0.5,
            "discard_sink": "filtered_out",
        })

        result = gate.evaluate({"id": 1, "metrics": {}}, ctx)

        assert result.action.kind == "route_to_sink"
        assert "missing" in result.action.reason.get("result", "")

    def test_deeply_nested_field(self, ctx: "PluginContext") -> None:
        """Deeply nested field access works correctly."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate({
            "field": "data.analysis.metrics.confidence",
            "greater_than_or_equal": 0.9,
            "discard_sink": "low_confidence",
        })

        result = gate.evaluate(
            {"id": 1, "data": {"analysis": {"metrics": {"confidence": 0.95}}}},
            ctx,
        )

        assert result.action.kind == "continue"
