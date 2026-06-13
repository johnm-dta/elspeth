"""Tests for BatchPairedPreference aggregation transform."""

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


class TestBatchPairedPreference:
    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_has_required_attributes(self) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        assert BatchPairedPreference.name == "batch_paired_preference"
        assert BatchPairedPreference.is_batch_aware is True

    def test_compares_first_seen_baseline_to_other_variant(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "variant", "score_field": "score"}
        )

        rows = [
            _make_row({"case_id": "p1", "variant": "A", "score": 0.4}),
            _make_row({"case_id": "p1", "variant": "B", "score": 0.7}),
            _make_row({"case_id": "p2", "variant": "A", "score": 0.8}),
            _make_row({"case_id": "p2", "variant": "B", "score": 0.5}),
            _make_row({"case_id": "p3", "variant": "A", "score": 0.6}),
            _make_row({"case_id": "p3", "variant": "B", "score": 0.6}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["pair_field"] == "case_id"
        assert result.row["variant_field"] == "variant"
        assert result.row["score_field"] == "score"
        assert result.row["baseline_variant"] == "A"
        assert result.row["variant"] == "B"
        assert result.row["batch_size"] == 6
        assert result.row["total_pair_count"] == 3
        assert result.row["compared_pair_count"] == 3
        assert result.row["incomplete_pair_count"] == 0
        assert result.row["baseline_mean"] == pytest.approx(0.6)
        assert result.row["variant_mean"] == pytest.approx(0.6)
        assert result.row["mean_paired_delta"] == pytest.approx(0.0)
        assert result.row["wins"] == 1
        assert result.row["losses"] == 1
        assert result.row["ties"] == 1
        assert result.row["win_rate"] == pytest.approx(1 / 3)
        assert result.row["loss_rate"] == pytest.approx(1 / 3)
        assert result.row["tie_rate"] == pytest.approx(1 / 3)
        assert result.row["preference_rate"] == 0.5
        assert result.row["standard_error_delta"] == pytest.approx(0.17320508075688773)
        assert result.row["confidence_95_low"] == pytest.approx(-0.33948195828349996)
        assert result.row["confidence_95_high"] == pytest.approx(0.33948195828349996)

    def test_configured_baseline_is_used_even_when_not_first_seen(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {
                "schema": DYNAMIC_SCHEMA,
                "pair_field": "case_id",
                "variant_field": "variant",
                "score_field": "score",
                "baseline_variant": "A",
            }
        )

        rows = [
            _make_row({"case_id": "p1", "variant": "B", "score": 0.7}),
            _make_row({"case_id": "p1", "variant": "A", "score": 0.5}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["baseline_variant"] == "A"
        assert result.row["variant"] == "B"
        assert result.row["mean_paired_delta"] == pytest.approx(0.2)
        assert result.row["wins"] == 1

    def test_multiple_variants_emit_one_comparison_per_non_baseline_variant(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "variant", "score_field": "score"}
        )

        rows = [
            _make_row({"case_id": "p1", "variant": "A", "score": 0.5}),
            _make_row({"case_id": "p1", "variant": "B", "score": 0.7}),
            _make_row({"case_id": "p1", "variant": "C", "score": 0.1}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert [row["variant"] for row in result.rows] == ["B", "C"]
        assert [row["baseline_variant"] for row in result.rows] == ["A", "A"]

    def test_variant_buckets_preserve_bool_and_int_identity(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "variant", "score_field": "score"}
        )

        rows = [
            _make_row({"case_id": "p1", "variant": True, "score": 1.0}),
            _make_row({"case_id": "p1", "variant": 1, "score": 3.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["baseline_variant"] is True
        assert type(result.row["baseline_variant"]) is bool
        assert result.row["variant"] == 1
        assert type(result.row["variant"]) is int
        assert result.row["compared_pair_count"] == 1
        assert result.row["mean_paired_delta"] == pytest.approx(2.0)

    def test_incomplete_pairs_are_reported_but_complete_pairs_still_compare(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "variant", "score_field": "score"}
        )

        rows = [
            _make_row({"case_id": "p1", "variant": "A", "score": 0.4}),
            _make_row({"case_id": "p1", "variant": "B", "score": 0.6}),
            _make_row({"case_id": "p2", "variant": "A", "score": 0.5}),
            _make_row({"case_id": "p2", "variant": "B", "score": None}),
            _make_row({"case_id": "p3", "variant": "A", "score": 0.5}),
            _make_row({"case_id": "p3", "variant": "B", "score": float("nan")}),
            _make_row({"case_id": "p4", "variant": "A", "score": 0.5}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["compared_pair_count"] == 1
        assert result.row["incomplete_pair_count"] == 3
        assert result.row["missing_score_count"] == 1
        assert result.row["non_finite_score_count"] == 1
        assert tuple(result.row["incomplete_pairs"]) == ("p2", "p3", "p4")
        assert result.row["wins"] == 1
        assert result.row["win_rate"] == 1.0

    def test_overflowed_paired_delta_returns_transform_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "variant", "score_field": "score"}
        )

        rows = [
            _make_row({"case_id": "p1", "variant": "A", "score": -1e308}),
            _make_row({"case_id": "p1", "variant": "B", "score": 1e308}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "float_overflow"
        assert result.reason["operation"] == "paired_delta"
        assert result.reason["group_value"] == "B"

    def test_non_numeric_scores_raise_type_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "variant", "score_field": "score"}
        )

        rows = [
            _make_row({"case_id": "p1", "variant": "A", "score": 0.5}),
            _make_row({"case_id": "p1", "variant": "B", "score": "high"}),
        ]

        with pytest.raises(TypeError, match="must be numeric"):
            transform.process(rows, ctx)

    @pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_group_key_returns_error_before_success(self, ctx: PluginContext, non_finite: float) -> None:
        """Non-finite variant key must error before producing output (B4.5-d)."""
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "variant", "score_field": "score"}
        )
        rows = [
            _make_row({"case_id": "p1", "variant": "A", "score": 0.5}),
            _make_row({"case_id": "p1", "variant": non_finite, "score": 0.8}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "non_finite_variant"
        assert result.reason["field"] == "variant"
        assert not result.retryable

    def test_no_complete_pairs_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "variant", "score_field": "score"}
        )

        rows = [
            _make_row({"case_id": "p1", "variant": "A", "score": 0.5}),
            _make_row({"case_id": "p2", "variant": "B", "score": 0.8}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "no_complete_pairs"
        assert result.reason["group_value"] == "B"

    def test_empty_batch_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "variant", "score_field": "score"}
        )

        result = transform.process([], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_batch"
        assert not result.retryable

    def test_all_ties_reports_none_preference_rate(self, ctx: PluginContext) -> None:
        """All-tie batch: wins+losses==0 -> preference_rate must be None (B4.5-b)."""
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "variant", "score_field": "score"}
        )
        rows = [
            _make_row({"case_id": "p1", "variant": "A", "score": 0.5}),
            _make_row({"case_id": "p1", "variant": "B", "score": 0.5}),
            _make_row({"case_id": "p2", "variant": "A", "score": 0.8}),
            _make_row({"case_id": "p2", "variant": "B", "score": 0.8}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["wins"] == 0
        assert result.row["losses"] == 0
        assert result.row["ties"] == 2
        # preference_rate = wins/(wins+losses) is 0/0 -- honest None, never 0.0
        assert result.row["preference_rate"] is None

    def test_single_compared_pair_reports_none_ci(self, ctx: PluginContext) -> None:
        """compared<=1 -> se=0 -> CI bounds must be None (B4.5-a-paired_pref-CI)."""
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "variant", "score_field": "score"}
        )
        rows = [
            _make_row({"case_id": "p1", "variant": "A", "score": 0.4}),
            _make_row({"case_id": "p1", "variant": "B", "score": 0.7}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["compared_pair_count"] == 1
        # standard_error undefined at n<=1 -- CI bounds must be None, never 0.0
        assert result.row["standard_error_delta"] is None
        assert result.row["confidence_95_low"] is None
        assert result.row["confidence_95_high"] is None


class TestBatchPairedPreferenceConfig:
    @pytest.mark.parametrize("field_name", ["pair_field", "variant_field", "score_field"])
    @pytest.mark.parametrize("blank_value", ["", "   "])
    def test_blank_field_names_rejected_at_config_boundary(self, field_name: str, blank_value: str) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        config = {"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "variant", "score_field": "score"}
        config[field_name] = blank_value

        with pytest.raises(PluginConfigError, match=f"{field_name} must not be empty"):
            BatchPairedPreference(config)

    def test_key_fields_must_be_distinct(self) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        with pytest.raises(PluginConfigError, match="pair_field, variant_field, and score_field must be distinct"):
            BatchPairedPreference({"schema": DYNAMIC_SCHEMA, "pair_field": "case_id", "variant_field": "score", "score_field": "score"})

    def test_blank_baseline_variant_rejected_at_config_boundary(self) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        with pytest.raises(PluginConfigError, match="baseline_variant must not be empty"):
            BatchPairedPreference(
                {
                    "schema": DYNAMIC_SCHEMA,
                    "pair_field": "case_id",
                    "variant_field": "variant",
                    "score_field": "score",
                    "baseline_variant": " ",
                }
            )

    def test_output_schema_config_guarantees_comparison_fields(self) -> None:
        from elspeth.plugins.transforms.batch_paired_preference import BatchPairedPreference

        transform = BatchPairedPreference(
            {
                "schema": {
                    "mode": "flexible",
                    "fields": ["case_id: str", "variant: str", "score: float", "input_only: str"],
                    "required_fields": ["case_id", "variant", "score"],
                    "guaranteed_fields": ["input_only"],
                },
                "pair_field": "case_id",
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
                "baseline_mean",
                "baseline_variant",
                "batch_size",
                "compared_pair_count",
                "confidence_95_high",
                "confidence_95_low",
                "incomplete_pair_count",
                "loss_rate",
                "losses",
                "mean_paired_delta",
                "missing_score_count",
                "non_finite_score_count",
                "pair_field",
                "preference_rate",
                "score_field",
                "standard_error_delta",
                "tie_rate",
                "ties",
                "total_pair_count",
                "variant",
                "variant_field",
                "variant_mean",
                "win_rate",
                "wins",
            }
        )
