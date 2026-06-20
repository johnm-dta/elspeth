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

    def test_mad_zero_uses_meanad_fallback_not_fabricated_zero(self, ctx: PluginContext) -> None:
        """When MAD=0 (>50% identical) the masked outlier gets an honest robust-z.

        For [1,1,1,100] the median is 1 and MAD=median([0,0,0,99])=0, so the old
        code fabricated robust_z_score=0.0 for the 100 — silently disabling robust
        detection exactly when it is needed and asserting "at median" for a clear
        outlier. The Iglewicz-Hoaglin MeanAD fallback gives the 100 a real,
        non-zero modified z-score; the genuine inliers (value at the median) stay
        0.0 (plugins review C3).
        """
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        transform = BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        rows = [
            _make_row({"id": 1, "score": 1.0}),
            _make_row({"id": 2, "score": 1.0}),
            _make_row({"id": 3, "score": 1.0}),
            _make_row({"id": 4, "score": 100.0}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 4

        outlier = result.rows[3]
        assert outlier["outlier_mad"] == pytest.approx(0.0)
        # 0.7979 * (100 - 1) / mean(|x-median|=24.75) = 0.7979 * 4.0
        assert outlier["outlier_robust_z_score"] == pytest.approx(3.1916)
        assert outlier["outlier_robust_z_score"] != 0.0

        # An inlier sitting exactly at the median legitimately scores 0.0.
        inlier = result.rows[0]
        assert inlier["outlier_robust_z_score"] == pytest.approx(0.0)

    def test_stdev_zero_reports_none_z_score(self, ctx: PluginContext) -> None:
        """All-identical batch: stdev=0 -> z_score undefined, must be None (B4.5-c)."""
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        transform = BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        rows = [_make_row({"id": i, "score": 5.0}) for i in range(1, 5)]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 4
        for row in result.rows:
            assert row["outlier_stdev"] == pytest.approx(0.0)
            # z_score = (x - mean) / stdev is x/0 = undefined -- honest None, never 0.0
            assert row["outlier_z_score"] is None
            assert row["outlier_is_outlier"] is False

    def test_all_identical_batch_reports_none_robust_z(self, ctx: PluginContext) -> None:
        """An all-identical batch has no spread: robust-z is undefined, emit None.

        MAD=0 and MeanAD=0, so there is no honest modified z-score. Per the
        honest-absence doctrine the value is None, never a fabricated 0.0
        (plugins review C3).
        """
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        transform = BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        rows = [_make_row({"id": i, "score": 5.0}) for i in range(1, 5)]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 4
        for row in result.rows:
            assert row["outlier_mad"] == pytest.approx(0.0)
            assert row["outlier_robust_z_score"] is None
            assert row["outlier_is_outlier"] is False

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

    def test_backward_invariant_probe_exercises_dropped_row_shape(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        transform = BatchOutlierAnnotator(BatchOutlierAnnotator.probe_config())
        probe = _make_row({"id": 1, "keep": "yes"})

        probe_rows = transform.backward_invariant_probe_rows(probe)
        result = transform.execute_backward_invariant_probe(probe_rows, ctx)

        assert result.status == "success"
        emitted_rows = result.rows if result.rows is not None else [result.row]
        assert all(row is not None for row in emitted_rows)
        input_fields = frozenset(field for row in probe_rows for field in row)
        emitted_fields = frozenset(field for row in emitted_rows if row is not None for field in row)
        assert "batch_outlier_annotator_dropped_probe_field" in input_fields
        assert "batch_outlier_annotator_dropped_probe_field" not in emitted_fields

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

    def test_huge_integer_overflow_returns_transform_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        transform = BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score"})

        result = transform.process([_make_row({"id": 1, "score": 10**1000})], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "float_overflow"
        assert result.reason["operation"] == "float_conversion"
        assert not result.retryable

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

    def test_excessive_batch_size_is_rejected_before_outlier_work(self, ctx: PluginContext, monkeypatch: pytest.MonkeyPatch) -> None:
        import elspeth.plugins.transforms.batch_outlier_annotator as module
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        monkeypatch.setattr(module, "_MAX_BATCH_ROWS", 3, raising=False)
        transform = BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score"})
        rows = [_make_row({"id": index, "score": float(index)}) for index in range(4)]

        result = transform.process(rows, ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "batch_too_large"
        assert result.reason["batch_size"] == 4
        assert result.reason["expected"] == "at most 3 rows"
        assert not result.retryable

    def test_large_skipped_index_details_are_bounded_per_output_row(
        self,
        ctx: PluginContext,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import elspeth.plugins.transforms.batch_outlier_annotator as module
        from elspeth.plugins.transforms.batch_outlier_annotator import BatchOutlierAnnotator

        monkeypatch.setattr(module, "_MAX_SKIPPED_INDEX_DETAILS", 3, raising=False)
        transform = BatchOutlierAnnotator({"schema": DYNAMIC_SCHEMA, "value_field": "score"})
        rows = [
            _make_row({"id": 0, "score": 10.0}),
            _make_row({"id": 1, "score": 11.0}),
            *[_make_row({"id": index, "score": None}) for index in range(2, 7)],
            *[_make_row({"id": index, "score": float("inf")}) for index in range(7, 12)],
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.rows is not None
        assert len(result.rows) == 2
        for row in result.rows:
            assert row["outlier_missing_count"] == 5
            assert row["outlier_non_finite_count"] == 5
            assert row["outlier_missing_indices"] == (2, 3, 4)
            assert row["outlier_non_finite_indices"] == (7, 8, 9)


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
