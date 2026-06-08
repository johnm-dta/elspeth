"""Batch classifier metrics transform plugin."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from pydantic import Field, field_validator, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.errors import RowErrorEntry, TransformErrorReason
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.transforms._scalar_buckets import (
    ScalarBucketKey,
    append_unique_bucket_value,
    same_scalar_bucket_value,
    scalar_bucket_contains,
    scalar_bucket_key,
)

type LabelValue = str | int | bool
type BatchClassifierMetricsRow = dict[str, object]

_BASE_METRIC_FIELDS = frozenset(
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
    }
)
_BINARY_METRIC_FIELDS = frozenset(
    {
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


@dataclass(frozen=True, slots=True)
class _LabelPair:
    actual: LabelValue
    predicted: LabelValue


@dataclass(frozen=True, slots=True)
class _PerLabelStats:
    label: LabelValue
    tp: int
    fp: int
    fn: int
    tn: int
    support: int
    # None marks a metric that is UNDEFINED for this label (a 0/0 ratio — e.g. a label
    # that was never predicted has undefined precision). None is honest absence, not a
    # fabricated 0.0: an auditor can distinguish "model predicted this label and was
    # always wrong" (0.0) from "model never predicted this label" (None).
    precision: float | None
    recall: float | None
    f1: float | None

    def to_row(self) -> dict[str, object]:
        return {
            "label": self.label,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "support": self.support,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


class BatchClassifierMetricsConfig(TransformDataConfig):
    """Configuration for batch classifier metrics transform."""

    actual_field: str = Field(description="Name of the field containing ground-truth labels")
    predicted_field: str = Field(description="Name of the field containing predicted labels")
    positive_label: str | int | bool | None = Field(
        default=None,
        description="Optional positive label for binary precision/recall/F1 metrics.",
    )

    @field_validator("actual_field")
    @classmethod
    def _reject_empty_actual_field(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("actual_field must not be empty")
        return v

    @field_validator("predicted_field")
    @classmethod
    def _reject_empty_predicted_field(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("predicted_field must not be empty")
        return v

    @field_validator("positive_label")
    @classmethod
    def _reject_empty_positive_label(cls, v: str | int | bool | None) -> str | int | bool | None:
        if type(v) is str and not v.strip():
            raise ValueError("positive_label must not be empty")
        return v

    @model_validator(mode="after")
    def _reject_field_collision(self) -> BatchClassifierMetricsConfig:
        if self.actual_field == self.predicted_field:
            raise ValueError("actual_field and predicted_field must differ")
        return self


class BatchClassifierMetrics(BaseTransform):
    """Compute classifier confusion matrix and F-score metrics over a batch.

    Phase 6A B6 — opts into Phase 6B's narrative-mode result rendering via
    ``capability_tags = ("narrative-summary",)``. The frontend reads this tag
    on the catalog response and, when set, renders the run result as a
    narrative panel (consuming any Phase 5b interpretation events as an
    overlay) rather than the default tabular preview. The wire contract for
    the tag is: the transform's output schema must include a ``summary`` field
    that the narrative renderer surfaces.
    """

    name = "batch_classifier_metrics"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:1b14e60f23f8848e"
    config_model = BatchClassifierMetricsConfig
    is_batch_aware = True
    capability_tags: tuple[str, ...] = ("narrative-summary",)

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Computes classifier accuracy, confusion counts, and precision/recall/F1 metrics for a batch.",
                composer_hints=(
                    "Use batch_classifier_metrics under aggregations with a trigger; it emits metric summary rows.",
                    "actual_field and predicted_field must be distinct scalar label fields.",
                    "Set positive_label only for binary metrics; macro, micro, and weighted metrics are always computed.",
                    "Output is classifier metrics, not per-row predictions; downstream fields should reference the metric keys.",
                ),
            )
        return None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "actual_field": "batch_classifier_metrics_probe_actual",
            "predicted_field": "batch_classifier_metrics_probe_predicted",
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchClassifierMetricsConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._actual_field = cfg.actual_field
        self._predicted_field = cfg.predicted_field
        self._positive_label = cfg.positive_label

        declared_output_fields = set(_BASE_METRIC_FIELDS)
        if cfg.positive_label is not None:
            declared_output_fields.update(_BINARY_METRIC_FIELDS)
        self.declared_output_fields = frozenset(declared_output_fields)

        base_required = set(cfg.schema_config.required_fields or ())
        base_required.update({cfg.actual_field, cfg.predicted_field})
        if base_required != set(cfg.schema_config.required_fields or ()):
            schema_config = SchemaConfig(
                mode=cfg.schema_config.mode,
                fields=cfg.schema_config.fields,
                guaranteed_fields=cfg.schema_config.guaranteed_fields,
                audit_fields=cfg.schema_config.audit_fields,
                required_fields=tuple(base_required),
            )
        else:
            schema_config = cfg.schema_config

        self._schema_config = schema_config
        self.input_schema, self.output_schema = self._create_schemas(
            schema_config,
            "BatchClassifierMetrics",
            adds_fields=True,
        )
        self._output_schema_config = self._build_output_schema_config(schema_config)

    def _build_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
        """Describe metric output without propagating input fields."""
        return SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=tuple(sorted(self.declared_output_fields)),
            required_fields=None,
            audit_fields=None,
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the metric output path for the backward invariant."""
        row = self._augment_invariant_probe_row(
            probe,
            field_name=self._actual_field,
            value="yes",
        )
        row = self._augment_invariant_probe_row(
            row,
            field_name=self._predicted_field,
            value="yes",
        )
        return [row]

    @staticmethod
    def _validate_label(value: object, *, field_name: str, row_index: int) -> LabelValue:
        if type(value) not in (str, int, bool):
            raise TypeError(
                f"Field '{field_name}' must be a scalar label (str, int, or bool), "
                f"got {type(value).__name__} in row {row_index}. "
                f"This indicates an upstream validation bug - check source schema or prior transforms."
            )
        return cast(LabelValue, value)

    def _collect_pairs(self, rows: list[PipelineRow]) -> tuple[list[_LabelPair], list[int]]:
        pairs: list[_LabelPair] = []
        missing_indices: list[int] = []

        for row_index, row in enumerate(rows):
            actual_value = row[self._actual_field]
            predicted_value = row[self._predicted_field]
            if actual_value is None or predicted_value is None:
                missing_indices.append(row_index)
                continue

            actual = self._validate_label(actual_value, field_name=self._actual_field, row_index=row_index)
            predicted = self._validate_label(predicted_value, field_name=self._predicted_field, row_index=row_index)
            pairs.append(_LabelPair(actual=actual, predicted=predicted))

        return pairs, missing_indices

    @staticmethod
    def _labels_for(pairs: list[_LabelPair]) -> list[LabelValue]:
        labels: list[LabelValue] = []
        for pair in pairs:
            append_unique_bucket_value(labels, pair.actual)
            append_unique_bucket_value(labels, pair.predicted)
        return labels

    @staticmethod
    def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
        # A 0/0 ratio is UNDEFINED, not zero. Returning None preserves that distinction
        # (fabrication test: a downstream consumer can tell "real 0" from "no data");
        # returning 0.0 would fabricate a value the data does not support.
        if denominator == 0:
            return None
        return float(numerator / denominator)

    @classmethod
    def _f1(cls, precision: float | None, recall: float | None) -> float | None:
        # F1 is undefined when either component is undefined, or when precision+recall
        # is 0 (both zero -> _safe_ratio returns None).
        if precision is None or recall is None:
            return None
        return cls._safe_ratio(2 * precision * recall, precision + recall)

    @staticmethod
    def _mean_defined(values: list[float | None]) -> float | None:
        """Unweighted mean over the DEFINED (non-None) values, or None if all undefined.

        Macro averages exclude labels whose per-label metric is undefined rather than
        treating them as 0.0 — averaging in a fabricated 0.0 would understate the metric
        and assert a value the data never supported.
        """
        defined = [v for v in values if v is not None]
        if not defined:
            return None
        return sum(defined) / len(defined)

    @staticmethod
    def _weighted_mean_defined(value_weight_pairs: list[tuple[float | None, int]]) -> float | None:
        """Support-weighted mean over DEFINED values, or None if no defined contributor.

        A label with an undefined metric contributes neither to the numerator nor the
        denominator, so the weighting stays honest for the labels that do have a value.
        """
        numerator = 0.0
        denominator = 0
        for value, weight in value_weight_pairs:
            if value is None:
                continue
            numerator += value * weight
            denominator += weight
        if denominator == 0:
            return None
        return numerator / denominator

    @staticmethod
    def _format_metric(value: float | None) -> str:
        if value is None:
            return "undefined"
        return f"{value:.3f}"

    def _per_label_metrics(
        self,
        *,
        labels: list[LabelValue],
        pairs: list[_LabelPair],
        confusion: Counter[tuple[ScalarBucketKey, ScalarBucketKey]],
    ) -> list[_PerLabelStats]:
        metrics: list[_PerLabelStats] = []
        total_count = len(pairs)
        for label in labels:
            label_key = scalar_bucket_key(label)
            tp = confusion[(label_key, label_key)]
            fp = sum(count for (actual, predicted), count in confusion.items() if actual != label_key and predicted == label_key)
            fn = sum(count for (actual, predicted), count in confusion.items() if actual == label_key and predicted != label_key)
            tn = total_count - tp - fp - fn
            precision = self._safe_ratio(tp, tp + fp)
            recall = self._safe_ratio(tp, tp + fn)
            metrics.append(
                _PerLabelStats(
                    label=label,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    tn=tn,
                    support=tp + fn,
                    precision=precision,
                    recall=recall,
                    f1=self._f1(precision, recall),
                )
            )
        return metrics

    @staticmethod
    def _confusion_matrix_rows(
        *,
        labels: list[LabelValue],
        confusion: Counter[tuple[ScalarBucketKey, ScalarBucketKey]],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for actual in labels:
            for predicted in labels:
                count = confusion[(scalar_bucket_key(actual), scalar_bucket_key(predicted))]
                if count:
                    rows.append({"actual": actual, "predicted": predicted, "count": count})
        return rows

    def _build_result_row(
        self,
        *,
        batch_size: int,
        pairs: list[_LabelPair],
        missing_indices: list[int],
    ) -> tuple[BatchClassifierMetricsRow, TransformResult | None]:
        labels = self._labels_for(pairs)
        if self._positive_label is not None and not scalar_bucket_contains(labels, self._positive_label):
            reason: TransformErrorReason = {
                "reason": "validation_failed",
                "cause": "positive_label_missing",
                "expected": str(self._positive_label),
                "errors": [str(label) for label in labels],
            }
            return {}, TransformResult.error(reason, retryable=False)

        confusion: Counter[tuple[ScalarBucketKey, ScalarBucketKey]] = Counter()
        for pair in pairs:
            key = (scalar_bucket_key(pair.actual), scalar_bucket_key(pair.predicted))
            confusion[key] += 1

        count = len(pairs)
        correct = sum(1 for pair in pairs if same_scalar_bucket_value(pair.actual, pair.predicted))
        accuracy = self._safe_ratio(correct, count)
        per_label = self._per_label_metrics(labels=labels, pairs=pairs, confusion=confusion)

        # Macro = unweighted mean over labels with DEFINED metrics; weighted = mean over
        # defined per-label f1 weighted by support. Undefined (None) per-label values are
        # excluded rather than counted as 0.0 (see _mean_defined / _weighted_mean_defined).
        macro_precision = self._mean_defined([entry.precision for entry in per_label])
        macro_recall = self._mean_defined([entry.recall for entry in per_label])
        macro_f1 = self._mean_defined([entry.f1 for entry in per_label])
        weighted_f1 = self._weighted_mean_defined([(entry.f1, entry.support) for entry in per_label])

        micro_tp = sum(entry.tp for entry in per_label)
        micro_fp = sum(entry.fp for entry in per_label)
        micro_fn = sum(entry.fn for entry in per_label)
        micro_precision = self._safe_ratio(micro_tp, micro_tp + micro_fp)
        micro_recall = self._safe_ratio(micro_tp, micro_tp + micro_fn)
        micro_f1 = self._f1(micro_precision, micro_recall)

        result: BatchClassifierMetricsRow = {
            "actual_field": self._actual_field,
            "predicted_field": self._predicted_field,
            "batch_size": batch_size,
            "count": count,
            "missing_count": len(missing_indices),
            "labels": labels,
            "confusion_matrix": self._confusion_matrix_rows(labels=labels, confusion=confusion),
            "per_label": [entry.to_row() for entry in per_label],
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "summary": (
                f"Classifier metrics for {self._actual_field} vs {self._predicted_field}: "
                f"{batch_size} rows, {count} valid label pairs, {correct} correct, "
                f"accuracy {self._format_metric(accuracy)}, macro F1 {self._format_metric(macro_f1)}, "
                f"micro F1 {self._format_metric(micro_f1)}."
            ),
        }

        if missing_indices:
            result["missing_indices"] = missing_indices

        if self._positive_label is not None:
            positive_metrics = next(entry for entry in per_label if same_scalar_bucket_value(entry.label, self._positive_label))
            result.update(
                {
                    "positive_label": self._positive_label,
                    "binary_tp": positive_metrics.tp,
                    "binary_fp": positive_metrics.fp,
                    "binary_fn": positive_metrics.fn,
                    "binary_tn": positive_metrics.tn,
                    "binary_precision": positive_metrics.precision,
                    "binary_recall": positive_metrics.recall,
                    "binary_f1": positive_metrics.f1,
                }
            )

        return result, None

    def _output_contract_for(self, results: list[BatchClassifierMetricsRow]) -> SchemaContract:
        """Build one shared output contract for classifier metric rows."""
        field_names = list(dict.fromkeys(key for result in results for key in result))
        fields = tuple(
            FieldContract(
                normalized_name=key,
                original_name=key,
                python_type=object,
                required=False,
                source="inferred",
            )
            for key in field_names
        )
        output_contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
        return self._align_output_contract(output_contract)

    def process(  # type: ignore[override] # Batch signature: list[PipelineRow] instead of PipelineRow
        self, rows: list[PipelineRow], ctx: TransformContext
    ) -> TransformResult:
        """Compute classifier metrics over a batch."""
        if not rows:
            return TransformResult.error({"reason": "empty_batch"}, retryable=False)

        pairs, missing_indices = self._collect_pairs(rows)
        if not pairs:
            row_errors: list[RowErrorEntry] = [{"row_index": row_index, "reason": "missing_label"} for row_index in missing_indices]
            reason: TransformErrorReason = {
                "reason": "validation_failed",
                "cause": "no_valid_label_pairs",
                "batch_size": len(rows),
                "valid_count": 0,
                "skipped_count": len(missing_indices),
                "row_errors": row_errors,
            }
            return TransformResult.error(
                reason,
                retryable=False,
            )

        result, error = self._build_result_row(batch_size=len(rows), pairs=pairs, missing_indices=missing_indices)
        if error is not None:
            return error

        output_contract = self._output_contract_for([result])
        fields_added = [field.normalized_name for field in output_contract.fields]
        return TransformResult.success(
            PipelineRow(result, output_contract),
            success_reason={"action": "processed", "fields_added": fields_added},
        )

    def close(self) -> None:
        """No resources to release."""
        pass
