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

    @pytest.mark.parametrize("group_value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_group_key_returns_error_before_success(self, ctx: PluginContext, group_value: float) -> None:
        """Non-finite group_by key must error before producing any output (B4.5-d)."""
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        transform = BatchTopK({"schema": DYNAMIC_SCHEMA, "field": "label", "group_by": "cohort", "k": 1})
        rows = [
            _make_row({"cohort": "A", "label": "x"}),
            _make_row({"cohort": group_value, "label": "y"}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "non_finite_group_key"
        assert result.reason["field"] == "cohort"
        assert not result.retryable

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

    def test_group_by_preserves_bool_and_int_buckets(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        transform = BatchTopK({"schema": DYNAMIC_SCHEMA, "field": "label", "group_by": "cohort", "k": 1})
        rows = [
            _make_row({"cohort": True, "label": "bool"}),
            _make_row({"cohort": 1, "label": "int"}),
            _make_row({"cohort": True, "label": "bool"}),
            _make_row({"cohort": 1, "label": "int"}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert [(type(row["group_value"]).__name__, row["group_value"]) for row in result.rows] == [("bool", True), ("int", 1)]
        assert [row["batch_size"] for row in result.rows] == [2, 2]

    def test_non_scalar_value_raises_type_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        transform = BatchTopK({"schema": DYNAMIC_SCHEMA, "field": "label"})

        with pytest.raises(TypeError, match="must be a scalar top-k value"):
            transform.process([_make_row({"label": ["a"]})], ctx)

    def test_bool_and_int_do_not_collide(self, ctx: PluginContext) -> None:
        # Python: True == 1 and hash(True) == hash(1). Treating them as the same
        # frequency bucket merges semantically distinct scalars, undercounts
        # distinct_count, and freezes top_values[].value to whichever type
        # appeared first in the batch.
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        transform = BatchTopK({"schema": DYNAMIC_SCHEMA, "field": "flag", "k": 4})
        rows = [
            _make_row({"flag": True}),
            _make_row({"flag": 1}),
            _make_row({"flag": True}),
            _make_row({"flag": 1}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["distinct_count"] == 2
        top_values_by_value = {(type(tv["value"]).__name__, tv["value"]): tv for tv in deep_thaw(result.row["top_values"])}
        assert ("bool", True) in top_values_by_value
        assert ("int", 1) in top_values_by_value
        assert top_values_by_value[("bool", True)]["count"] == 2
        assert top_values_by_value[("int", 1)]["count"] == 2

    def test_int_and_float_do_not_collide(self, ctx: PluginContext) -> None:
        # 1 == 1.0 and hash(1) == hash(1.0). The transform's TopKValue contract
        # admits both — the source could legitimately emit a column that mixes
        # int and float representations of the same numeric value, and they
        # represent distinct provider-emitted facts that the audit trail must
        # preserve.
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        transform = BatchTopK({"schema": DYNAMIC_SCHEMA, "field": "score", "k": 4})
        rows = [
            _make_row({"score": 1}),
            _make_row({"score": 1.0}),
            _make_row({"score": 1}),
            _make_row({"score": 1.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["distinct_count"] == 2
        top_values_by_value = {(type(tv["value"]).__name__, tv["value"]): tv for tv in deep_thaw(result.row["top_values"])}
        assert ("int", 1) in top_values_by_value
        assert ("float", 1.0) in top_values_by_value
        assert top_values_by_value[("int", 1)]["count"] == 2
        assert top_values_by_value[("float", 1.0)]["count"] == 2

    def test_homogeneous_int_batch_still_merges(self, ctx: PluginContext) -> None:
        # Regression guard: the type-identity fix must not split same-type
        # values into separate buckets. A batch of plain ints must produce
        # distinct_count == 1.
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        transform = BatchTopK({"schema": DYNAMIC_SCHEMA, "field": "score", "k": 4})
        rows = [_make_row({"score": 1}) for _ in range(5)]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["distinct_count"] == 1
        top_values = deep_thaw(result.row["top_values"])
        assert top_values == [{"value": 1, "count": 5, "rate": 1.0}]

    def test_zero_int_and_false_do_not_collide(self, ctx: PluginContext) -> None:
        # 0 == False and hash(0) == hash(False). Symmetric to the True/1 case.
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        transform = BatchTopK({"schema": DYNAMIC_SCHEMA, "field": "flag", "k": 4})
        rows = [
            _make_row({"flag": 0}),
            _make_row({"flag": False}),
            _make_row({"flag": 0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["distinct_count"] == 2
        top_values_by_value = {(type(tv["value"]).__name__, tv["value"]): tv for tv in deep_thaw(result.row["top_values"])}
        assert ("int", 0) in top_values_by_value
        assert ("bool", False) in top_values_by_value


class TestBatchTopKConfig:
    @pytest.mark.parametrize("config", [{"field": ""}, {"field": "label", "group_by": "label"}, {"field": "label", "k": 0}])
    def test_invalid_config_rejected_at_config_boundary(self, config: dict[str, Any]) -> None:
        from elspeth.plugins.transforms.batch_top_k import BatchTopK

        with pytest.raises(PluginConfigError):
            BatchTopK({"schema": DYNAMIC_SCHEMA, **config})
