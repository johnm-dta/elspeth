"""Tests for BatchThresholdSummary aggregation transform."""

from typing import Any

import pytest

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.testing import make_field, make_row
from tests.fixtures.factories import make_context

DYNAMIC_SCHEMA = {"mode": "observed"}


def _make_row(data: dict[str, Any]):
    """Create a PipelineRow with OBSERVED contract for testing."""
    fields = tuple(
        make_field(key, type(value) if value is not None else object, original_name=key, required=False, source="inferred")
        for key, value in data.items()
    )
    contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
    return make_row(data, contract=contract)


class TestBatchThresholdSummary:
    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_has_required_attributes(self) -> None:
        from elspeth.plugins.transforms.batch_threshold_summary import BatchThresholdSummary

        assert BatchThresholdSummary.name == "batch_threshold_summary"
        assert BatchThresholdSummary.is_batch_aware is True

    def test_emits_one_row_per_threshold(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_threshold_summary import BatchThresholdSummary

        transform = BatchThresholdSummary(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "score",
                "thresholds": [
                    {"name": "good", "operator": ">=", "value": 0.8},
                    {"name": "poor", "operator": "<", "value": 0.5},
                ],
            }
        )
        rows = [
            _make_row({"score": 0.9}),
            _make_row({"score": 0.75}),
            _make_row({"score": 0.4}),
            _make_row({"score": None}),
            _make_row({"score": float("inf")}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        summaries = {row["threshold_name"]: row for row in result.rows}
        assert summaries["good"]["batch_size"] == 5
        assert summaries["good"]["valid_count"] == 3
        assert summaries["good"]["missing_count"] == 1
        assert summaries["good"]["non_finite_count"] == 1
        assert summaries["good"]["match_count"] == 1
        assert summaries["good"]["match_rate"] == pytest.approx(1 / 3)
        assert summaries["poor"]["match_count"] == 1
        assert summaries["poor"]["non_match_count"] == 2

    def test_non_numeric_value_raises_type_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_threshold_summary import BatchThresholdSummary

        transform = BatchThresholdSummary(
            {
                "schema": DYNAMIC_SCHEMA,
                "value_field": "score",
                "thresholds": [{"name": "good", "operator": ">=", "value": 0.8}],
            }
        )

        with pytest.raises(TypeError, match="must be numeric"):
            transform.process([_make_row({"score": "high"})], ctx)


class TestBatchThresholdSummaryConfig:
    @pytest.mark.parametrize(
        "config",
        [
            {"value_field": "", "thresholds": [{"name": "good", "operator": ">=", "value": 0.8}]},
            {"value_field": "score", "thresholds": []},
            {
                "value_field": "score",
                "thresholds": [
                    {"name": "good", "operator": ">=", "value": 0.8},
                    {"name": "good", "operator": "<", "value": 0.5},
                ],
            },
        ],
    )
    def test_invalid_config_rejected_at_config_boundary(self, config: dict[str, Any]) -> None:
        from elspeth.plugins.transforms.batch_threshold_summary import BatchThresholdSummary

        with pytest.raises(PluginConfigError):
            BatchThresholdSummary({"schema": DYNAMIC_SCHEMA, **config})

    def test_excessive_thresholds_rejected_at_config_boundary(self) -> None:
        from elspeth.plugins.transforms.batch_threshold_summary import BatchThresholdSummary

        thresholds = [{"name": f"threshold_{index}", "operator": ">=", "value": float(index)} for index in range(129)]

        with pytest.raises(PluginConfigError, match="thresholds must contain at most 128 entries"):
            BatchThresholdSummary({"schema": DYNAMIC_SCHEMA, "value_field": "score", "thresholds": thresholds})
