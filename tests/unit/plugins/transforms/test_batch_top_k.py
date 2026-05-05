"""Tests for BatchTopK aggregation transform."""

from typing import Any

import pytest

from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.plugins.infrastructure.config_base import PluginConfigError
from elspeth.testing import make_field, make_row
from tests.fixtures.factories import make_context

DYNAMIC_SCHEMA = {"mode": "observed"}


def _make_row(data: dict[str, Any]):
    """Create a PipelineRow with OBSERVED contract for testing."""
    fields = tuple(
        make_field(
            key,
            type(value) if value is None or type(value) in (str, int, float, bool) else object,
            original_name=key,
            required=False,
            source="inferred",
        )
        for key, value in data.items()
    )
    contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
    return make_row(data, contract=contract)


class TestBatchTopK:
    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_has_required_attributes(self) -> None:
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        assert BatchTopK.name == "batch_top_k"
        assert BatchTopK.is_batch_aware is True

    def test_reports_top_values_for_scalar_field(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        transform = BatchTopK({"schema": DYNAMIC_SCHEMA, "field": "label", "k": 2})
        rows = [
            _make_row({"label": "a"}),
            _make_row({"label": "a"}),
            _make_row({"label": "b"}),
            _make_row({"label": "c"}),
            _make_row({"label": "a"}),
            _make_row({"label": None}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["field"] == "label"
        assert result.row["batch_size"] == 6
        assert result.row["count"] == 5
        assert result.row["missing_count"] == 1
        assert result.row["distinct_count"] == 3
        assert deep_thaw(result.row["top_values"]) == [
            {"value": "a", "count": 3, "rate": 0.6},
            {"value": "b", "count": 1, "rate": 0.2},
        ]

    def test_group_by_emits_one_row_per_group(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        transform = BatchTopK({"schema": DYNAMIC_SCHEMA, "field": "label", "group_by": "cohort", "k": 1})
        rows = [
            _make_row({"cohort": "A", "label": "x"}),
            _make_row({"cohort": "A", "label": "x"}),
            _make_row({"cohort": "A", "label": "y"}),
            _make_row({"cohort": "B", "label": "z"}),
            _make_row({"cohort": "B", "label": "z"}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert [row["group_value"] for row in result.rows] == ["A", "B"]
        assert deep_thaw(result.rows[0]["top_values"]) == [{"value": "x", "count": 2, "rate": pytest.approx(2 / 3)}]
        assert deep_thaw(result.rows[1]["top_values"]) == [{"value": "z", "count": 2, "rate": 1.0}]

    def test_non_scalar_value_raises_type_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        transform = BatchTopK({"schema": DYNAMIC_SCHEMA, "field": "label"})

        with pytest.raises(TypeError, match="must be a scalar top-k value"):
            transform.process([_make_row({"label": ["a"]})], ctx)


class TestBatchTopKConfig:
    @pytest.mark.parametrize("config", [{"field": ""}, {"field": "label", "group_by": "label"}, {"field": "label", "k": 0}])
    def test_invalid_config_rejected_at_config_boundary(self, config: dict[str, Any]) -> None:
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        with pytest.raises(PluginConfigError):
            BatchTopK({"schema": DYNAMIC_SCHEMA, **config})
