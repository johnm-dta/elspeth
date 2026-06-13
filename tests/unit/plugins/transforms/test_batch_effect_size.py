"""Tests for BatchEffectSize aggregation transform."""

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


class TestBatchEffectSize:
    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_has_required_attributes(self) -> None:
        from elspeth.plugins.transforms.batch_effect_size import BatchEffectSize

        assert BatchEffectSize.name == "batch_effect_size"
        assert BatchEffectSize.is_batch_aware is True

    def test_computes_cohens_d_and_hedges_g(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_effect_size import BatchEffectSize

        transform = BatchEffectSize(
            {"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score", "baseline_variant": "control"}
        )
        rows = [
            _make_row({"variant": "control", "score": 1.0}),
            _make_row({"variant": "control", "score": 2.0}),
            _make_row({"variant": "control", "score": 3.0}),
            _make_row({"variant": "treatment", "score": 2.0}),
            _make_row({"variant": "treatment", "score": 3.0}),
            _make_row({"variant": "treatment", "score": 4.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["baseline_variant"] == "control"
        assert result.row["variant"] == "treatment"
        assert result.row["baseline_mean"] == 2.0
        assert result.row["variant_mean"] == 3.0
        assert result.row["mean_delta"] == 1.0
        assert result.row["pooled_stdev"] == 1.0
        assert result.row["cohens_d"] == 1.0
        assert result.row["hedges_g"] == pytest.approx(0.8)

    def test_missing_and_non_finite_scores_are_skipped_and_reported(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_effect_size import BatchEffectSize

        transform = BatchEffectSize({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score"})
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

    @pytest.mark.parametrize("variant_value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_variant_returns_error_before_success_output(self, ctx: PluginContext, variant_value: float) -> None:
        from elspeth.plugins.transforms.batch_effect_size import BatchEffectSize

        transform = BatchEffectSize({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score"})
        rows = [
            _make_row({"variant": "A", "score": 1.0}),
            _make_row({"variant": variant_value, "score": 2.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "non_finite_variant"
        assert result.reason["field"] == "variant"
        assert result.reason["row_errors"] == [{"row_index": 1, "reason": "non_finite_variant"}]
        assert not result.retryable

    def test_non_numeric_score_raises_type_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_effect_size import BatchEffectSize

        transform = BatchEffectSize({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score"})

        with pytest.raises(TypeError, match="must be numeric"):
            transform.process([_make_row({"variant": "A", "score": 1.0}), _make_row({"variant": "B", "score": "high"})], ctx)

    def test_single_value_group_reports_none_stdev(self, ctx: PluginContext) -> None:
        """n=1 stdev is undefined -- must emit None, never 0.0 (B4.5-a-effect_size-stdev)."""
        from elspeth.plugins.transforms.batch_effect_size import BatchEffectSize

        transform = BatchEffectSize({"schema": DYNAMIC_SCHEMA, "variant_field": "variant", "score_field": "score"})
        rows = [
            _make_row({"variant": "A", "score": 5.0}),
            _make_row({"variant": "B", "score": 7.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["baseline_count"] == 1
        assert result.row["variant_count"] == 1
        # stdev undefined at n=1 -- honest None, never 0.0
        assert result.row["baseline_stdev"] is None
        assert result.row["variant_stdev"] is None


class TestBatchEffectSizeConfig:
    @pytest.mark.parametrize(
        "config",
        [
            {"variant_field": "", "score_field": "score"},
            {"variant_field": "variant", "score_field": ""},
            {"variant_field": "score", "score_field": "score"},
        ],
    )
    def test_invalid_config_rejected_at_config_boundary(self, config: dict[str, Any]) -> None:
        from elspeth.plugins.transforms.batch_effect_size import BatchEffectSize

        with pytest.raises(PluginConfigError):
            BatchEffectSize({"schema": DYNAMIC_SCHEMA, **config})
