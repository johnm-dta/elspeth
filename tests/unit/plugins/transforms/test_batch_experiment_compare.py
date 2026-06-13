"""Tests for BatchExperimentCompare aggregation transform."""

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


class TestBatchExperimentCompare:
    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_has_required_attributes(self) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        assert BatchExperimentCompare.name == "batch_experiment_compare"
        assert BatchExperimentCompare.is_batch_aware is True

    def test_compares_first_seen_baseline_to_other_variant(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        transform = BatchExperimentCompare({"schema": DYNAMIC_SCHEMA, "variant_field": "prompt_variant", "score_field": "score"})

        rows = [
            _make_row({"prompt_variant": "control", "score": 0.4}),
            _make_row({"prompt_variant": "treatment", "score": 0.7}),
            _make_row({"prompt_variant": "control", "score": 0.6}),
            _make_row({"prompt_variant": "treatment", "score": 0.9}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["variant_field"] == "prompt_variant"
        assert result.row["score_field"] == "score"
        assert result.row["baseline_variant"] == "control"
        assert result.row["variant"] == "treatment"
        assert result.row["baseline_count"] == 2
        assert result.row["variant_count"] == 2
        assert result.row["baseline_total_count"] == 2
        assert result.row["variant_total_count"] == 2
        assert result.row["baseline_mean"] == 0.5
        assert result.row["variant_mean"] == 0.8
        assert result.row["mean_delta"] == pytest.approx(0.3)
        assert result.row["relative_lift"] == pytest.approx(0.6)
        assert result.row["baseline_stdev"] == pytest.approx(0.14142135623730948)
        assert result.row["variant_stdev"] == pytest.approx(0.14142135623730948)
        assert result.row["standard_error"] == pytest.approx(0.14142135623730948)
        assert result.row["z_score"] == pytest.approx(2.121320343559643)
        assert result.row["confidence_95_low"] == pytest.approx(0.022814142, abs=1e-9)
        assert result.row["confidence_95_high"] == pytest.approx(0.577185858, abs=1e-9)

    def test_configured_baseline_is_used_even_when_not_first_seen(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        transform = BatchExperimentCompare(
            {
                "schema": DYNAMIC_SCHEMA,
                "variant_field": "prompt_variant",
                "score_field": "score",
                "baseline_variant": "control",
            }
        )

        rows = [
            _make_row({"prompt_variant": "treatment", "score": 0.8}),
            _make_row({"prompt_variant": "control", "score": 0.5}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["baseline_variant"] == "control"
        assert result.row["variant"] == "treatment"
        assert result.row["mean_delta"] == pytest.approx(0.3)

    def test_multiple_variants_emit_one_comparison_per_non_baseline_variant(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        transform = BatchExperimentCompare({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score"})

        rows = [
            _make_row({"variant": "A", "score": 0.5}),
            _make_row({"variant": "B", "score": 0.7}),
            _make_row({"variant": "C", "score": 0.2}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert [row["variant"] for row in result.rows] == ["B", "C"]
        assert [row["baseline_variant"] for row in result.rows] == ["A", "A"]

    def test_variant_buckets_preserve_bool_and_int_identity(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        transform = BatchExperimentCompare({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score"})
        rows = [
            _make_row({"variant": True, "score": 1.0}),
            _make_row({"variant": 1, "score": 3.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert type(result.row["baseline_variant"]) is bool
        assert result.row["baseline_variant"] is True
        assert type(result.row["variant"]) is int
        assert result.row["variant"] == 1
        assert result.row["baseline_count"] == 1
        assert result.row["variant_count"] == 1

    def test_missing_and_non_finite_scores_are_skipped_and_reported(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        transform = BatchExperimentCompare({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score"})

        rows = [
            _make_row({"variant": "A", "score": 1.0}),
            _make_row({"variant": "A", "score": None}),
            _make_row({"variant": "B", "score": 2.0}),
            _make_row({"variant": "B", "score": float("nan")}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["baseline_count"] == 1
        assert result.row["variant_count"] == 1
        assert result.row["baseline_missing_count"] == 1
        assert result.row["variant_non_finite_count"] == 1
        assert result.row["baseline_missing_indices"] == (1,)
        assert result.row["variant_non_finite_indices"] == (3,)

    def test_baseline_missing_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        transform = BatchExperimentCompare(
            {
                "schema": DYNAMIC_SCHEMA,
                "variant_field": "variant",
                "score_field": "score",
                "baseline_variant": "control",
            }
        )

        result = transform.process([_make_row({"variant": "treatment", "score": 1.0})], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "baseline_variant_missing"
        assert result.reason["expected"] == "control"

    def test_insufficient_variants_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        transform = BatchExperimentCompare({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score"})

        result = transform.process([_make_row({"variant": "A", "score": 1.0})], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "insufficient_variants"
        assert not result.retryable

    def test_variant_with_no_finite_scores_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        transform = BatchExperimentCompare({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score"})

        rows = [
            _make_row({"variant": "A", "score": 1.0}),
            _make_row({"variant": "B", "score": None}),
            _make_row({"variant": "B", "score": float("inf")}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "variant_has_no_finite_scores"
        assert result.reason["group_value"] == "B"
        assert result.reason["row_errors"] == [
            {"row_index": 1, "reason": "missing_value"},
            {"row_index": 2, "reason": "non_finite_value"},
        ]

    def test_non_numeric_scores_raise_type_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        transform = BatchExperimentCompare({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score"})

        rows = [
            _make_row({"variant": "A", "score": 1.0}),
            _make_row({"variant": "B", "score": "not_a_number"}),
        ]

        with pytest.raises(TypeError, match="must be numeric"):
            transform.process(rows, ctx)

    def test_empty_batch_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        transform = BatchExperimentCompare({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score"})

        result = transform.process([], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_batch"
        assert not result.retryable

    def test_single_value_group_reports_none_stdev_and_ci(self, ctx: PluginContext) -> None:
        """n=1 stdev is undefined, se=0 -> CI bounds must be None (B4.5-a-experiment_compare)."""
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        transform = BatchExperimentCompare({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score"})
        rows = [
            _make_row({"variant": "control", "score": 3.0}),
            _make_row({"variant": "treatment", "score": 5.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["baseline_count"] == 1
        assert result.row["variant_count"] == 1
        # stdev undefined at n=1 -- honest None, never 0.0
        assert result.row["baseline_stdev"] is None
        assert result.row["variant_stdev"] is None
        # standard_error=0 when both stdevs are None -> CI bounds undefined
        assert result.row["z_score"] is None
        assert result.row["confidence_95_low"] is None
        assert result.row["confidence_95_high"] is None


class TestBatchExperimentCompareConfig:
    @pytest.mark.parametrize("blank_field", ["", "   "])
    def test_blank_variant_field_rejected_at_config_boundary(self, blank_field: str) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        with pytest.raises(PluginConfigError, match="variant_field must not be empty"):
            BatchExperimentCompare({"schema": DYNAMIC_SCHEMA, "variant_field": blank_field, "score_field": "score"})

    @pytest.mark.parametrize("blank_field", ["", "   "])
    def test_blank_score_field_rejected_at_config_boundary(self, blank_field: str) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        with pytest.raises(PluginConfigError, match="score_field must not be empty"):
            BatchExperimentCompare({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": blank_field})

    def test_variant_and_score_fields_must_differ(self) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        with pytest.raises(PluginConfigError, match="variant_field and score_field must differ"):
            BatchExperimentCompare({"schema": DYNAMIC_SCHEMA, "variant_field": "score", "score_field": "score"})

    def test_blank_baseline_variant_rejected_at_config_boundary(self) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        with pytest.raises(PluginConfigError, match="baseline_variant must not be empty"):
            BatchExperimentCompare(
                {"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score", "baseline_variant": "   "}
            )

    def test_output_schema_config_guarantees_comparison_fields(self) -> None:
        from elspeth.plugins.transforms.batch_experiment_compare import BatchExperimentCompare

        transform = BatchExperimentCompare(
            {
                "schema": {
                    "mode": "flexible",
                    "fields": ["id: int", "variant: str", "score: float", "input_only: str"],
                    "required_fields": ["variant", "score"],
                    "guaranteed_fields": ["input_only"],
                },
                "variant_field": "variant",
                "score_field": "score",
            }
        )

        cfg = transform._output_schema_config
        assert cfg is not None
        assert cfg.fields is None
        assert cfg.required_fields is None
        assert "input_only" not in (cfg.guaranteed_fields or ())
        assert frozenset(cfg.guaranteed_fields or ()) == frozenset(
            {
                "baseline_count",
                "baseline_mean",
                "baseline_missing_count",
                "baseline_non_finite_count",
                "baseline_stdev",
                "baseline_total_count",
                "baseline_variant",
                "batch_size",
                "confidence_95_high",
                "confidence_95_low",
                "mean_delta",
                "relative_lift",
                "score_field",
                "standard_error",
                "variant",
                "variant_count",
                "variant_field",
                "variant_mean",
                "variant_missing_count",
                "variant_non_finite_count",
                "variant_stdev",
                "variant_total_count",
                "z_score",
            }
        )
