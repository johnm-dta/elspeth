"""Tests for BatchDriftCompare aggregation transform."""

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


class TestBatchDriftCompare:
    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_has_required_attributes(self) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        assert BatchDriftCompare.name == "batch_drift_compare"
        assert BatchDriftCompare.is_batch_aware is True

    def test_numeric_drift_compares_first_seen_baseline_to_other_cohort(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        transform = BatchDriftCompare({"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "score"})

        rows = [
            _make_row({"cohort": "baseline", "score": 1.0}),
            _make_row({"cohort": "current", "score": 2.0}),
            _make_row({"cohort": "baseline", "score": 2.0}),
            _make_row({"cohort": "current", "score": 3.0}),
            _make_row({"cohort": "baseline", "score": 3.0}),
            _make_row({"cohort": "current", "score": 4.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["cohort_field"] == "cohort"
        assert result.row["value_field"] == "score"
        assert result.row["value_type"] == "numeric"
        assert result.row["baseline_cohort"] == "baseline"
        assert result.row["cohort"] == "current"
        assert result.row["batch_size"] == 6
        assert result.row["baseline_count"] == 3
        assert result.row["cohort_count"] == 3
        assert result.row["baseline_mean"] == 2.0
        assert result.row["cohort_mean"] == 3.0
        assert result.row["mean_delta"] == 1.0
        assert result.row["ks_statistic"] == pytest.approx(1 / 3)
        assert result.row["baseline_missing_count"] == 0
        assert result.row["cohort_missing_count"] == 0
        assert result.row["baseline_non_finite_count"] == 0
        assert result.row["cohort_non_finite_count"] == 0

    def test_configured_baseline_is_used_even_when_not_first_seen(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        transform = BatchDriftCompare(
            {
                "schema": DYNAMIC_SCHEMA,
                "cohort_field": "cohort",
                "value_field": "score",
                "baseline_cohort": "baseline",
            }
        )

        rows = [
            _make_row({"cohort": "current", "score": 5.0}),
            _make_row({"cohort": "baseline", "score": 3.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["baseline_cohort"] == "baseline"
        assert result.row["cohort"] == "current"
        assert result.row["mean_delta"] == 2.0

    def test_multiple_current_cohorts_emit_one_comparison_each(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        transform = BatchDriftCompare({"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "score"})

        rows = [
            _make_row({"cohort": "baseline", "score": 1.0}),
            _make_row({"cohort": "week_1", "score": 2.0}),
            _make_row({"cohort": "week_2", "score": 3.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert [row["cohort"] for row in result.rows] == ["week_1", "week_2"]
        assert [row["baseline_cohort"] for row in result.rows] == ["baseline", "baseline"]

    def test_numeric_missing_and_non_finite_values_are_skipped_and_reported(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        transform = BatchDriftCompare({"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "score"})

        rows = [
            _make_row({"cohort": "baseline", "score": 1.0}),
            _make_row({"cohort": "baseline", "score": None}),
            _make_row({"cohort": "current", "score": 2.0}),
            _make_row({"cohort": "current", "score": float("inf")}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["baseline_count"] == 1
        assert result.row["cohort_count"] == 1
        assert result.row["baseline_missing_count"] == 1
        assert result.row["cohort_non_finite_count"] == 1

    def test_numeric_mean_overflow_returns_transform_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        transform = BatchDriftCompare({"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "score"})

        rows = [
            _make_row({"cohort": "baseline", "score": 1e308}),
            _make_row({"cohort": "baseline", "score": 1e308}),
            _make_row({"cohort": "current", "score": 1.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "float_overflow"
        assert result.reason["operation"] == "baseline_mean"
        assert result.reason["group_value"] == "current"

    def test_categorical_drift_uses_total_variation_and_chi_square_summary(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        transform = BatchDriftCompare(
            {"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "label", "value_type": "categorical"}
        )

        rows = [
            _make_row({"cohort": "baseline", "label": "a"}),
            _make_row({"cohort": "baseline", "label": "a"}),
            _make_row({"cohort": "baseline", "label": "b"}),
            _make_row({"cohort": "current", "label": "a"}),
            _make_row({"cohort": "current", "label": "b"}),
            _make_row({"cohort": "current", "label": "b"}),
            _make_row({"cohort": "current", "label": "c"}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["value_type"] == "categorical"
        assert result.row["total_variation"] == pytest.approx(5 / 12)
        assert result.row["chi_square_statistic"] == pytest.approx(1.375)
        assert result.row["new_category_count"] == 1
        assert tuple(result.row["new_categories"]) == ("c",)
        shifts = {entry["value"]: entry for entry in result.row["category_shifts"]}
        assert shifts["a"]["baseline_proportion"] == pytest.approx(2 / 3)
        assert shifts["a"]["cohort_proportion"] == 0.25
        assert shifts["c"]["baseline_proportion"] == 0.0
        assert shifts["c"]["cohort_proportion"] == 0.25

    def test_categorical_drift_preserves_bool_and_int_buckets(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        transform = BatchDriftCompare(
            {"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "label", "value_type": "categorical"}
        )
        rows = [
            _make_row({"cohort": "baseline", "label": True}),
            _make_row({"cohort": "current", "label": 1}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["total_variation"] == 1.0
        assert result.row["new_category_count"] == 1
        assert [(type(value).__name__, value) for value in result.row["new_categories"]] == [("int", 1)]
        shifts = {(type(entry["value"]).__name__, entry["value"]): entry for entry in result.row["category_shifts"]}
        assert shifts[("bool", True)]["baseline_count"] == 1
        assert shifts[("bool", True)]["cohort_count"] == 0
        assert shifts[("int", 1)]["baseline_count"] == 0
        assert shifts[("int", 1)]["cohort_count"] == 1

    def test_numeric_non_numeric_values_raise_type_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        transform = BatchDriftCompare({"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "score"})

        rows = [
            _make_row({"cohort": "baseline", "score": 1.0}),
            _make_row({"cohort": "current", "score": "high"}),
        ]

        with pytest.raises(TypeError, match="must be numeric"):
            transform.process(rows, ctx)

    def test_no_valid_values_for_cohort_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        transform = BatchDriftCompare({"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "score"})

        rows = [
            _make_row({"cohort": "baseline", "score": 1.0}),
            _make_row({"cohort": "current", "score": None}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "no_valid_values"
        assert result.reason["group_value"] == "current"

    def test_empty_batch_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        transform = BatchDriftCompare({"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "score"})

        result = transform.process([], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_batch"
        assert not result.retryable

    @pytest.mark.parametrize("cohort_value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_group_key_returns_error_before_success(self, ctx: PluginContext, cohort_value: float) -> None:
        """Non-finite cohort key must error before producing any output (B4.5-d)."""
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        transform = BatchDriftCompare({"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "score"})
        rows = [
            _make_row({"cohort": "baseline", "score": 1.0}),
            _make_row({"cohort": cohort_value, "score": 2.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "non_finite_group_key"
        assert result.reason["field"] == "cohort"
        assert not result.retryable


class TestBatchDriftCompareConfig:
    @pytest.mark.parametrize("field_name", ["cohort_field", "value_field"])
    @pytest.mark.parametrize("blank_value", ["", "   "])
    def test_blank_field_names_rejected_at_config_boundary(self, field_name: str, blank_value: str) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        config = {"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "score"}
        config[field_name] = blank_value

        with pytest.raises(PluginConfigError, match=f"{field_name} must not be empty"):
            BatchDriftCompare(config)

    def test_cohort_and_value_fields_must_differ(self) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        with pytest.raises(PluginConfigError, match="cohort_field and value_field must differ"):
            BatchDriftCompare({"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "cohort"})

    def test_blank_baseline_cohort_rejected_at_config_boundary(self) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        with pytest.raises(PluginConfigError, match="baseline_cohort must not be empty"):
            BatchDriftCompare({"schema": DYNAMIC_SCHEMA, "cohort_field": "cohort", "value_field": "score", "baseline_cohort": " "})

    def test_output_schema_config_guarantees_numeric_fields(self) -> None:
        from elspeth.plugins.transforms.batch_drift_compare import BatchDriftCompare

        transform = BatchDriftCompare(
            {
                "schema": {
                    "mode": "flexible",
                    "fields": ["cohort: str", "score: float", "input_only: str"],
                    "required_fields": ["cohort", "score"],
                    "guaranteed_fields": ["input_only"],
                },
                "cohort_field": "cohort",
                "value_field": "score",
            }
        )

        cfg = transform._output_schema_config
        assert cfg is not None
        assert cfg.fields is None
        assert cfg.required_fields is None
        assert "input_only" not in (cfg.guaranteed_fields or ())
        assert frozenset(cfg.guaranteed_fields or ()) == frozenset(
            {
                "baseline_cohort",
                "baseline_count",
                "baseline_mean",
                "baseline_missing_count",
                "baseline_non_finite_count",
                "baseline_total_count",
                "batch_size",
                "cohort",
                "cohort_count",
                "cohort_field",
                "cohort_mean",
                "cohort_missing_count",
                "cohort_non_finite_count",
                "cohort_total_count",
                "ks_statistic",
                "mean_delta",
                "value_field",
                "value_type",
            }
        )
