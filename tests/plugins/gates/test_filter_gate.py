"""Tests for FilterGate plugin.

FilterGate returns route LABELS ("pass"/"discard"), not sink names.
The routes config in settings.yaml maps labels to sinks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from elspeth.plugins.context import PluginContext


@pytest.fixture
def ctx() -> PluginContext:
    """Create minimal plugin context."""
    from elspeth.plugins.context import PluginContext

    return PluginContext(
        run_id="test-run",
        config={},
        landscape=MagicMock(),
    )


class TestFilterGate:
    """Test FilterGate returns route labels."""

    def test_passing_row_routes_to_pass_label(self, ctx: PluginContext) -> None:
        """Row that passes filter returns 'pass' label."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate(
            {
                "field": "score",
                "greater_than": 0.5,
            }
        )

        result = gate.evaluate({"id": 1, "score": 0.8}, ctx)

        assert result.action.kind == "route"
        assert result.action.destinations == ("pass",)

    def test_failing_row_routes_to_discard_label(self, ctx: PluginContext) -> None:
        """Row that fails filter returns 'discard' label."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate(
            {
                "field": "score",
                "greater_than": 0.5,
            }
        )

        result = gate.evaluate({"id": 1, "score": 0.3}, ctx)

        assert result.action.kind == "route"
        assert result.action.destinations == ("discard",)

    def test_reason_includes_filter_details(self, ctx: PluginContext) -> None:
        """Routing reason includes why the row was filtered."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate(
            {
                "field": "score",
                "greater_than": 0.5,
            }
        )

        result = gate.evaluate({"id": 1, "score": 0.3}, ctx)

        assert result.action.reason["field"] == "score"
        assert result.action.reason["value"] == 0.3
        assert result.action.reason["condition"] == "greater_than"
        assert result.action.reason["threshold"] == 0.5
        assert result.action.reason["result"] == "discard"

    def test_missing_field_routes_to_discard(self, ctx: PluginContext) -> None:
        """Row with missing field returns 'discard' label by default."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate(
            {
                "field": "score",
                "greater_than": 0.5,
            }
        )

        result = gate.evaluate({"id": 1}, ctx)  # No "score" field

        assert result.action.kind == "route"
        assert result.action.destinations == ("discard",)
        assert "missing" in result.action.reason.get("result", "")

    def test_missing_field_passes_when_allowed(self, ctx: PluginContext) -> None:
        """Row with missing field returns 'pass' if allow_missing=True."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate(
            {
                "field": "score",
                "greater_than": 0.5,
                "allow_missing": True,
            }
        )

        result = gate.evaluate({"id": 1}, ctx)  # No "score" field

        assert result.action.kind == "route"
        assert result.action.destinations == ("pass",)


class TestFilterGateValidation:
    """Test FilterGate validation errors."""

    def test_multiple_operators_raises_error(self) -> None:
        """Specifying multiple operators raises PluginConfigError."""
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.gates.filter_gate import FilterGate

        with pytest.raises(PluginConfigError, match="Multiple comparison operators"):
            FilterGate(
                {
                    "field": "score",
                    "greater_than": 0.5,
                    "less_than": 0.9,
                }
            )

    def test_no_operator_raises_error(self) -> None:
        """Specifying no operator raises PluginConfigError."""
        from elspeth.plugins.config_base import PluginConfigError
        from elspeth.plugins.gates.filter_gate import FilterGate

        with pytest.raises(PluginConfigError, match="No comparison operator specified"):
            FilterGate(
                {
                    "field": "score",
                }
            )


class TestFilterGateOperators:
    """Test all comparison operators."""

    @pytest.mark.parametrize(
        ("operator", "threshold", "value", "expected_label"),
        [
            # less_than
            ("less_than", 0.5, 0.3, "pass"),
            ("less_than", 0.5, 0.5, "discard"),
            ("less_than", 0.5, 0.8, "discard"),
            # equals
            ("equals", "active", "active", "pass"),
            ("equals", "active", "inactive", "discard"),
            ("equals", 100, 100, "pass"),
            ("equals", 100, 99, "discard"),
            # not_equals
            ("not_equals", "active", "inactive", "pass"),
            ("not_equals", "active", "active", "discard"),
            # greater_than_or_equal
            ("greater_than_or_equal", 0.5, 0.5, "pass"),
            ("greater_than_or_equal", 0.5, 0.6, "pass"),
            ("greater_than_or_equal", 0.5, 0.4, "discard"),
            # less_than_or_equal
            ("less_than_or_equal", 0.5, 0.5, "pass"),
            ("less_than_or_equal", 0.5, 0.4, "pass"),
            ("less_than_or_equal", 0.5, 0.6, "discard"),
        ],
    )
    def test_operator(
        self,
        ctx: PluginContext,
        operator: str,
        threshold: float | str | int,
        value: float | str | int,
        expected_label: str,
    ) -> None:
        """Parametrized test for all operators."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate(
            {
                "field": "score",
                operator: threshold,
            }
        )

        result = gate.evaluate({"id": 1, "score": value}, ctx)

        assert result.action.kind == "route"
        assert result.action.destinations == (expected_label,)


class TestFilterGateNestedFields:
    """Test nested field access via dot notation."""

    def test_nested_field_access(self, ctx: PluginContext) -> None:
        """Dot notation accesses nested fields."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate(
            {
                "field": "metrics.score",
                "greater_than": 0.5,
            }
        )

        result = gate.evaluate(
            {"id": 1, "metrics": {"score": 0.8}},
            ctx,
        )

        assert result.action.kind == "route"
        assert result.action.destinations == ("pass",)

    def test_nested_field_routes_when_fails(self, ctx: PluginContext) -> None:
        """Nested field that fails condition routes to 'discard'."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate(
            {
                "field": "metrics.score",
                "greater_than": 0.5,
            }
        )

        result = gate.evaluate(
            {"id": 1, "metrics": {"score": 0.3}},
            ctx,
        )

        assert result.action.kind == "route"
        assert result.action.destinations == ("discard",)

    def test_missing_nested_field_routes_to_discard(self, ctx: PluginContext) -> None:
        """Missing nested field routes to 'discard'."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate(
            {
                "field": "metrics.score",
                "greater_than": 0.5,
            }
        )

        result = gate.evaluate({"id": 1, "metrics": {}}, ctx)

        assert result.action.kind == "route"
        assert result.action.destinations == ("discard",)
        assert "missing" in result.action.reason.get("result", "")

    def test_deeply_nested_field(self, ctx: PluginContext) -> None:
        """Deeply nested field access works correctly."""
        from elspeth.plugins.gates.filter_gate import FilterGate

        gate = FilterGate(
            {
                "field": "data.analysis.metrics.confidence",
                "greater_than_or_equal": 0.9,
            }
        )

        result = gate.evaluate(
            {"id": 1, "data": {"analysis": {"metrics": {"confidence": 0.95}}}},
            ctx,
        )

        assert result.action.kind == "route"
        assert result.action.destinations == ("pass",)
