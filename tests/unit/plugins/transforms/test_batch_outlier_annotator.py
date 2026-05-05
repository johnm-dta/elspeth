"""Tests for BatchOutlierAnnotator aggregation transform."""

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


class TestBatchOutlierAnnotator:
    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_has_required_attributes(self) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        assert BatchOutlierAnnotator.name == "batch_outlier_annotator"
        assert BatchOutlierAnnotator.is_batch_aware is True
        assert BatchOutlierAnnotator.passes_through_input is False

    def test_annotates_each_valid_row_with_batch_outlier_scores(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        transform = BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        rows = [
            _make_row({"id": 1, "score": 10.0}),
            _make_row({"id": 2, "score": 11.0}),
            _make_row({"id": 3, "score": 12.0}),
            _make_row({"id": 4, "score": 100.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.is_multi_row
        assert result.rows is not None
        assert len(result.rows) == 4
        assert [row["id"] for row in result.rows] == [1, 2, 3, 4]

        outlier = result.rows[3]
        assert outlier["outlier_value_field"] == "score"
        assert outlier["outlier_row_index"] == 3
        assert outlier["outlier_batch_size"] == 4
        assert outlier["outlier_valid_count"] == 4
        assert outlier["outlier_mean"] == pytest.approx(33.25)
        assert outlier["outlier_median"] == pytest.approx(11.5)
        assert outlier["outlier_mad"] == pytest.approx(1.0)
        assert outlier["outlier_robust_z_score"] == pytest.approx(59.69325)
        assert outlier["outlier_is_outlier"] is True
        assert outlier["outlier_reason"] == "robust_z_score"

        inlier = result.rows[1]
        assert inlier["outlier_is_outlier"] is False
        assert inlier["outlier_reason"] == ""

    def test_missing_and_non_finite_values_are_skipped_and_reported(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        transform = BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        rows = [
            _make_row({"id": 1, "score": 10.0}),
            _make_row({"id": 2, "score": None}),
            _make_row({"id": 3, "score": float("inf")}),
            _make_row({"id": 4, "score": 12.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 2
        assert [row["id"] for row in result.rows] == [1, 4]
        for row in result.rows:
            assert row["outlier_batch_size"] == 4
            assert row["outlier_valid_count"] == 2
            assert row["outlier_missing_count"] == 1
            assert row["outlier_non_finite_count"] == 1
            assert row["outlier_skipped_count"] == 2
            assert row["outlier_missing_indices"] == (1,)
            assert row["outlier_non_finite_indices"] == (2,)

    def test_all_missing_or_non_finite_values_return_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        transform = BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        rows = [
            _make_row({"id": 1, "score": None}),
            _make_row({"id": 2, "score": float("nan")}),
            _make_row({"id": 3, "score": float("-inf")}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "no_finite_values"
        assert result.reason["batch_size"] == 3
        assert result.reason["valid_count"] == 0
        assert result.reason["skipped_count"] == 3
        assert result.reason["row_errors"] == [
            {"row_index": 0, "reason": "missing_value"},
            {"row_index": 1, "reason": "non_finite_value"},
            {"row_index": 2, "reason": "non_finite_value"},
        ]

    def test_non_numeric_values_raise_type_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        transform = BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        rows = [
            _make_row({"id": 1, "score": 10.0}),
            _make_row({"id": 2, "score": "large"}),
        ]

        with pytest.raises(TypeError, match="must be numeric"):
            transform.process(rows, ctx)

    def test_empty_batch_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        transform = BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        result = transform.process([], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_batch"
        assert not result.retryable


class TestBatchOutlierAnnotatorConfig:
    @pytest.mark.parametrize("blank_value_field", ["", "   "])
    def test_blank_value_field_rejected_at_config_boundary(self, blank_value_field: str) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        with pytest.raises(PluginConfigError, match="value_field must not be empty"):
            BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": blank_value_field})

    @pytest.mark.parametrize("blank_output_prefix", ["", "   "])
    def test_blank_output_prefix_rejected_at_config_boundary(self, blank_output_prefix: str) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        with pytest.raises(PluginConfigError, match="output_prefix must not be empty"):
            BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score", "output_prefix": blank_output_prefix})

    def test_value_field_must_not_collide_with_annotation_fields(self) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        with pytest.raises(PluginConfigError, match="collides with outlier annotation output key"):
            BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "outlier_is_outlier"})

    def test_thresholds_must_be_positive(self) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        with pytest.raises(PluginConfigError, match="greater than 0"):
            BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score", "robust_z_threshold": 0.0})

    def test_output_schema_config_guarantees_input_and_annotation_fields(self) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        transform = BatchOutlierAnnotator(
            {
                "schema": {
                    "mode": "flexible",
                    "fields": ["id: int", "score: float", "upstream_only: str"],
                    "required_fields": ["score"],
                    "guaranteed_fields": ["id", "upstream_only"],
                },
                "value_field": "score",
            }
        )

        cfg = transform._output_schema_config
        assert cfg is not None
        assert cfg.fields is not None
        assert set(cfg.required_fields or ()) == {"score"}
        assert frozenset(cfg.guaranteed_fields or ()) == frozenset(
            {
                "id",
                "outlier_batch_size",
                "outlier_is_outlier",
                "outlier_mad",
                "outlier_mean",
                "outlier_median",
                "outlier_missing_count",
                "outlier_missing_indices",
                "outlier_non_finite_count",
                "outlier_non_finite_indices",
                "outlier_reason",
                "outlier_robust_z_score",
                "outlier_robust_z_threshold",
                "outlier_row_index",
                "outlier_skipped_count",
                "outlier_stdev",
                "outlier_valid_count",
                "outlier_value",
                "outlier_value_field",
                "outlier_z_score",
                "outlier_z_threshold",
                "upstream_only",
            }
        )
