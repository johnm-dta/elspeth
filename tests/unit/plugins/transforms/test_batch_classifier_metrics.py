"""Tests for BatchClassifierMetrics aggregation transform."""

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


class TestBatchClassifierMetrics:
    @pytest.fixture
    def ctx(self) -> PluginContext:
        return make_context()

    def test_has_required_attributes(self) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        assert BatchClassifierMetrics.name == "batch_classifier_metrics"
        assert BatchClassifierMetrics.is_batch_aware is True

    def test_computes_multiclass_confusion_and_f_scores(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        transform = BatchClassifierMetrics({"schema": DYNAMIC_SCHEMA, "actual_field": "actual", "predicted_field": "predicted"})

        rows = [
            _make_row({"actual": "A", "predicted": "A"}),
            _make_row({"actual": "A", "predicted": "B"}),
            _make_row({"actual": "B", "predicted": "B"}),
            _make_row({"actual": "C", "predicted": "B"}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["actual_field"] == "actual"
        assert result.row["predicted_field"] == "predicted"
        assert result.row["batch_size"] == 4
        assert result.row["count"] == 4
        assert result.row["missing_count"] == 0
        assert result.row["accuracy"] == 0.5
        assert tuple(result.row["labels"]) == ("A", "B", "C")
        # Label C is never predicted -> its precision (tp/(tp+fp) = 0/0) is UNDEFINED,
        # reported as None (not a fabricated 0.0). Its f1 is therefore None too. So the
        # macro/weighted aggregates are computed over the DEFINED per-label values only.
        # macro_f1 = mean(A=0.6667, B=0.5) = 0.5833 (C's None excluded).
        # weighted_f1 = (0.6667*2 + 0.5*1) / (2+1) = 0.6111 (C contributes no defined f1).
        # micro_f1 is unchanged (its denominators are non-zero).
        assert result.row["macro_f1"] == pytest.approx(0.5833333333333333)
        assert result.row["weighted_f1"] == pytest.approx(0.6111111111111112)
        assert result.row["micro_f1"] == 0.5
        assert result.row["summary"] == (
            "Classifier metrics for actual vs predicted: 4 rows, 4 valid label pairs, "
            "2 correct, accuracy 0.500, macro F1 0.583, micro F1 0.500."
        )

        per_label = {entry["label"]: entry for entry in result.row["per_label"]}
        assert per_label["A"]["tp"] == 1
        assert per_label["A"]["fp"] == 0
        assert per_label["A"]["fn"] == 1
        assert per_label["A"]["precision"] == 1.0
        assert per_label["A"]["recall"] == 0.5
        assert per_label["A"]["f1"] == pytest.approx(0.6666666666666666)
        assert per_label["B"]["precision"] == pytest.approx(1 / 3)
        assert per_label["B"]["recall"] == 1.0
        assert per_label["B"]["f1"] == 0.5
        # C: precision undefined (0/0) -> None; recall defined (0/1) -> 0.0; f1 -> None.
        assert per_label["C"]["precision"] is None
        assert per_label["C"]["recall"] == 0.0
        assert per_label["C"]["f1"] is None

        cells = {(entry["actual"], entry["predicted"]): entry["count"] for entry in result.row["confusion_matrix"]}
        assert cells == {("A", "A"): 1, ("A", "B"): 1, ("B", "B"): 1, ("C", "B"): 1}

    def test_positive_label_metrics_are_emitted_when_configured(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        transform = BatchClassifierMetrics(
            {"schema": DYNAMIC_SCHEMA, "actual_field": "actual", "predicted_field": "predicted", "positive_label": "spam"}
        )

        rows = [
            _make_row({"actual": "spam", "predicted": "spam"}),
            _make_row({"actual": "spam", "predicted": "ham"}),
            _make_row({"actual": "ham", "predicted": "spam"}),
            _make_row({"actual": "ham", "predicted": "ham"}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["positive_label"] == "spam"
        assert result.row["binary_tp"] == 1
        assert result.row["binary_fp"] == 1
        assert result.row["binary_fn"] == 1
        assert result.row["binary_tn"] == 1
        assert result.row["binary_precision"] == 0.5
        assert result.row["binary_recall"] == 0.5
        assert result.row["binary_f1"] == 0.5

    def test_bool_and_int_labels_do_not_collide(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        transform = BatchClassifierMetrics({"schema": DYNAMIC_SCHEMA, "actual_field": "actual", "predicted_field": "predicted"})
        rows = [
            _make_row({"actual": True, "predicted": True}),
            _make_row({"actual": 1, "predicted": 1}),
            _make_row({"actual": True, "predicted": 1}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert [(type(label).__name__, label) for label in result.row["labels"]] == [("bool", True), ("int", 1)]
        cells = {
            (type(entry["actual"]).__name__, entry["actual"], type(entry["predicted"]).__name__, entry["predicted"]): entry["count"]
            for entry in result.row["confusion_matrix"]
        }
        assert cells == {
            ("bool", True, "bool", True): 1,
            ("bool", True, "int", 1): 1,
            ("int", 1, "int", 1): 1,
        }

    def test_missing_labels_are_skipped_and_reported(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        transform = BatchClassifierMetrics({"schema": DYNAMIC_SCHEMA, "actual_field": "actual", "predicted_field": "predicted"})

        rows = [
            _make_row({"actual": "yes", "predicted": "yes"}),
            _make_row({"actual": None, "predicted": "yes"}),
            _make_row({"actual": "no", "predicted": None}),
        ]

        result = transform.process(rows, ctx)

        assert result.status == "success"
        assert result.row is not None
        assert result.row["batch_size"] == 3
        assert result.row["count"] == 1
        assert result.row["missing_count"] == 2
        assert result.row["missing_indices"] == (1, 2)
        assert result.row["accuracy"] == 1.0

    def test_all_missing_labels_return_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        transform = BatchClassifierMetrics({"schema": DYNAMIC_SCHEMA, "actual_field": "actual", "predicted_field": "predicted"})

        result = transform.process([_make_row({"actual": None, "predicted": "yes"})], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "validation_failed"
        assert result.reason["cause"] == "no_valid_label_pairs"
        assert result.reason["batch_size"] == 1
        assert result.reason["row_errors"] == [{"row_index": 0, "reason": "missing_label"}]

    def test_positive_label_absent_computes_honest_none(self, ctx: PluginContext) -> None:
        """A configured positive_label absent from the batch is legitimate data.

        Rare-positive monitoring (e.g. fraud, defects) is the whole reason to set
        positive_label, so an all-negative batch is expected, not a config fault.
        The binary metrics are computed honestly: tp=0, fn=0, tn=count, with
        precision/recall undefined (None) rather than fabricated 0.0 or an error
        (plugins review Batch 3 item 11).
        """
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        transform = BatchClassifierMetrics(
            {"schema": DYNAMIC_SCHEMA, "actual_field": "actual", "predicted_field": "predicted", "positive_label": "maybe"}
        )

        result = transform.process(
            [
                _make_row({"actual": "yes", "predicted": "no"}),
                _make_row({"actual": "no", "predicted": "no"}),
            ],
            ctx,
        )

        assert result.status == "success"
        assert result.row is not None
        assert result.row["positive_label"] == "maybe"
        assert result.row["binary_tp"] == 0
        assert result.row["binary_fn"] == 0
        assert result.row["binary_tn"] == 2
        assert result.row["binary_precision"] is None
        assert result.row["binary_recall"] is None
        assert result.row["binary_f1"] is None

    def test_non_scalar_labels_raise_type_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        transform = BatchClassifierMetrics({"schema": DYNAMIC_SCHEMA, "actual_field": "actual", "predicted_field": "predicted"})

        with pytest.raises(TypeError, match="must be a scalar label"):
            transform.process([_make_row({"actual": object(), "predicted": "yes"})], ctx)

    def test_empty_batch_returns_error(self, ctx: PluginContext) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        transform = BatchClassifierMetrics({"schema": DYNAMIC_SCHEMA, "actual_field": "actual", "predicted_field": "predicted"})

        result = transform.process([], ctx)

        assert result.status == "error"
        assert result.reason is not None
        assert result.reason["reason"] == "empty_batch"
        assert not result.retryable


class TestBatchClassifierMetricsConfig:
    @pytest.mark.parametrize("blank_field", ["", "   "])
    def test_blank_actual_field_rejected_at_config_boundary(self, blank_field: str) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        with pytest.raises(PluginConfigError, match="actual_field must not be empty"):
            BatchClassifierMetrics({"schema": DYNAMIC_SCHEMA, "actual_field": blank_field, "predicted_field": "predicted"})

    @pytest.mark.parametrize("blank_field", ["", "   "])
    def test_blank_predicted_field_rejected_at_config_boundary(self, blank_field: str) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        with pytest.raises(PluginConfigError, match="predicted_field must not be empty"):
            BatchClassifierMetrics({"schema": DYNAMIC_SCHEMA, "actual_field": "actual", "predicted_field": blank_field})

    def test_actual_and_predicted_fields_must_differ(self) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        with pytest.raises(PluginConfigError, match="actual_field and predicted_field must differ"):
            BatchClassifierMetrics({"schema": DYNAMIC_SCHEMA, "actual_field": "label", "predicted_field": "label"})

    def test_blank_positive_label_rejected_at_config_boundary(self) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        with pytest.raises(PluginConfigError, match="positive_label must not be empty"):
            BatchClassifierMetrics(
                {"schema": DYNAMIC_SCHEMA, "actual_field": "actual", "predicted_field": "predicted", "positive_label": "   "}
            )

    def test_output_schema_config_guarantees_metric_fields(self) -> None:
        from elspeth.plugins.transforms.batch_classifier_metrics import BatchClassifierMetrics

        transform = BatchClassifierMetrics(
            {
                "schema": {
                    "mode": "flexible",
                    "fields": ["actual: str", "predicted: str", "input_only: str"],
                    "required_fields": ["actual", "predicted"],
                    "guaranteed_fields": ["input_only"],
                },
                "actual_field": "actual",
                "predicted_field": "predicted",
                "positive_label": "yes",
            }
        )

        cfg = transform._output_schema_config
        assert cfg is not None
        assert cfg.fields is None
        assert cfg.required_fields is None
        assert "input_only" not in (cfg.guaranteed_fields or ())
        assert frozenset(cfg.guaranteed_fields or ()) == frozenset(
            {
                "accuracy",
                "actual_field",
                "batch_size",
                "confusion_matrix",
                "count",
                "labels",
                "macro_f1",
                "macro_precision",
                "macro_recall",
                "micro_f1",
                "micro_precision",
                "micro_recall",
                "missing_count",
                "per_label",
                "predicted_field",
                "summary",
                "weighted_f1",
                "binary_f1",
                "binary_fn",
                "binary_fp",
                "binary_precision",
                "binary_recall",
                "binary_tn",
                "binary_tp",
                "positive_label",
            }
        )
