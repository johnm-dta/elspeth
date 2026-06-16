"""Batch drift comparison transform plugin."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, cast

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
    same_scalar_bucket_value,
    scalar_bucket_key,
)

type BatchDriftCompareRow = dict[str, object]
type CategoricalValue = str | int | bool
ValueType = Literal["numeric", "categorical"]

_COMMON_OUTPUT_FIELDS = frozenset(
    {
        "baseline_cohort",
        "baseline_count",
        "baseline_missing_count",
        "baseline_non_finite_count",
        "baseline_total_count",
        "batch_size",
        "cohort",
        "cohort_count",
        "cohort_field",
        "cohort_missing_count",
        "cohort_non_finite_count",
        "cohort_total_count",
        "value_field",
        "value_type",
    }
)
_NUMERIC_OUTPUT_FIELDS = _COMMON_OUTPUT_FIELDS | frozenset(
    {
        "baseline_mean",
        "cohort_mean",
        "ks_statistic",
        "mean_delta",
    }
)
_CATEGORICAL_OUTPUT_FIELDS = _COMMON_OUTPUT_FIELDS | frozenset(
    {
        "category_shifts",
        "chi_square_statistic",
        "new_categories",
        "new_category_count",
        "total_variation",
    }
)
_MAX_BATCH_ROWS = 4096


@dataclass(frozen=True, slots=True)
class _CohortValues:
    cohort: Any
    total_count: int
    values: tuple[object, ...]
    missing_count: int = 0
    non_finite_count: int = 0

    @property
    def count(self) -> int:
        return len(self.values)


class BatchDriftCompareConfig(TransformDataConfig):
    """Configuration for batch drift comparison transform."""

    cohort_field: str = Field(description="Field that partitions rows into baseline/current cohorts")
    value_field: str = Field(description="Field whose distribution should be compared")
    value_type: ValueType = Field(
        default="numeric",
        description="Distribution type to compare: numeric uses empirical CDF distance; categorical uses total variation.",
    )
    baseline_cohort: str | int | bool | None = Field(
        default=None,
        description="Optional baseline cohort. When omitted, the first-seen cohort is the baseline.",
    )

    @field_validator("cohort_field")
    @classmethod
    def _reject_empty_cohort_field(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("cohort_field must not be empty")
        return v

    @field_validator("value_field")
    @classmethod
    def _reject_empty_value_field(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("value_field must not be empty")
        return v

    @field_validator("baseline_cohort")
    @classmethod
    def _reject_empty_baseline_cohort(cls, v: str | int | bool | None) -> str | int | bool | None:
        if type(v) is str and not v.strip():
            raise ValueError("baseline_cohort must not be empty")
        return v

    @model_validator(mode="after")
    def _reject_field_collision(self) -> BatchDriftCompareConfig:
        if self.cohort_field == self.value_field:
            raise ValueError("cohort_field and value_field must differ")
        return self


class BatchDriftCompare(BaseTransform):
    """Compare baseline and current cohort distributions over a batch."""

    name = "batch_drift_compare"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:3efb65134bc8c78a"
    config_model = BatchDriftCompareConfig
    is_batch_aware = True

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Compares baseline and current cohort distributions across a batch.",
                composer_hints=(
                    "Use batch_drift_compare under aggregations with a trigger; it compares distributions after a batch flush.",
                    "cohort_field and value_field must differ; baseline_cohort defaults to the first-seen cohort.",
                    "Set value_type=numeric for mean and KS-style distance, or categorical for category shifts and total variation.",
                    "Output is cohort comparison rows and does not preserve the original row shape.",
                ),
            )
        return None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "cohort_field": "batch_drift_compare_probe_cohort",
            "value_field": "batch_drift_compare_probe_value",
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchDriftCompareConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._cohort_field = cfg.cohort_field
        self._value_field = cfg.value_field
        self._value_type = cfg.value_type
        self._baseline_cohort = cfg.baseline_cohort
        self.declared_output_fields = _NUMERIC_OUTPUT_FIELDS if cfg.value_type == "numeric" else _CATEGORICAL_OUTPUT_FIELDS

        base_required = set(cfg.schema_config.required_fields or ())
        base_required.update({cfg.cohort_field, cfg.value_field})
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
            "BatchDriftCompare",
            adds_fields=True,
        )
        self._output_schema_config = self._build_output_schema_config(schema_config)

    def _build_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
        """Describe drift comparison rows without propagating input fields."""
        return SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=tuple(sorted(self.declared_output_fields)),
            required_fields=None,
            audit_fields=None,
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the drift output path for the backward invariant."""
        baseline = self._augment_invariant_probe_row(probe, field_name=self._cohort_field, value="baseline")
        baseline = self._augment_invariant_probe_row(baseline, field_name=self._value_field, value=1.0)
        current = self._augment_invariant_probe_row(probe, field_name=self._cohort_field, value="current")
        current = self._augment_invariant_probe_row(current, field_name=self._value_field, value=2.0)
        return [baseline, current]

    @staticmethod
    def _is_non_finite_group_key(value: object) -> bool:
        if type(value) is float:
            return not math.isfinite(value)
        if type(value) is Decimal:
            return not value.is_finite()
        return False

    def _non_finite_group_key_error(self, rows: list[PipelineRow]) -> TransformResult | None:
        """Return an error if any row carries a non-finite cohort key (B4.5-d)."""
        row_errors: list[RowErrorEntry] = []
        for row_index, row in enumerate(rows):
            if self._is_non_finite_group_key(row[self._cohort_field]):
                row_errors.append({"row_index": row_index, "reason": "non_finite_group_key"})
        if not row_errors:
            return None
        reason: TransformErrorReason = {
            "reason": "validation_failed",
            "cause": "non_finite_group_key",
            "field": self._cohort_field,
            "row_errors": row_errors,
        }
        return TransformResult.error(reason, retryable=False)

    def _collect_cohorts(self, rows: list[PipelineRow]) -> list[tuple[Any, list[PipelineRow]]]:
        cohorts: list[tuple[Any, list[PipelineRow]]] = []
        for row in rows:
            cohort_value = row[self._cohort_field]
            for existing_cohort, cohort_rows in cohorts:
                if same_scalar_bucket_value(cohort_value, existing_cohort):
                    cohort_rows.append(row)
                    break
            else:
                cohorts.append((cohort_value, [row]))
        return cohorts

    def _numeric_values_for(self, cohort: Any, rows: list[PipelineRow]) -> _CohortValues:
        values: list[int | float] = []
        missing_count = 0
        non_finite_count = 0

        for row_index, row in enumerate(rows):
            raw_value = row[self._value_field]
            if raw_value is None:
                missing_count += 1
                continue
            if type(raw_value) not in (int, float):
                raise TypeError(
                    f"Field '{self._value_field}' must be numeric (int or float), "
                    f"got {type(raw_value).__name__} in row {row_index}. "
                    f"This indicates an upstream validation bug - check source schema or prior transforms."
                )
            if type(raw_value) is float and not math.isfinite(raw_value):
                non_finite_count += 1
                continue
            values.append(raw_value)

        return _CohortValues(
            cohort=cohort,
            total_count=len(rows),
            values=tuple(values),
            missing_count=missing_count,
            non_finite_count=non_finite_count,
        )

    def _categorical_values_for(self, cohort: Any, rows: list[PipelineRow]) -> _CohortValues:
        values: list[CategoricalValue] = []
        missing_count = 0

        for row_index, row in enumerate(rows):
            raw_value = row[self._value_field]
            if raw_value is None:
                missing_count += 1
                continue
            if type(raw_value) not in (str, int, bool):
                raise TypeError(
                    f"Field '{self._value_field}' must be a scalar category (str, int, or bool), "
                    f"got {type(raw_value).__name__} in row {row_index}. "
                    f"This indicates an upstream validation bug - check source schema or prior transforms."
                )
            values.append(raw_value)

        return _CohortValues(
            cohort=cohort,
            total_count=len(rows),
            values=tuple(values),
            missing_count=missing_count,
            non_finite_count=0,
        )

    def _values_for(self, cohort: Any, rows: list[PipelineRow]) -> _CohortValues:
        if self._value_type == "numeric":
            return self._numeric_values_for(cohort, rows)
        return self._categorical_values_for(cohort, rows)

    @staticmethod
    def _error_for_no_values(stats: _CohortValues, *, batch_size: int) -> TransformResult:
        reason: TransformErrorReason = {
            "reason": "validation_failed",
            "cause": "no_valid_values",
            "group_value": stats.cohort,
            "batch_size": batch_size,
            "valid_count": 0,
            "skipped_count": stats.missing_count + stats.non_finite_count,
        }
        return TransformResult.error(reason, retryable=False)

    @staticmethod
    def _error_for_batch_too_large(*, batch_size: int) -> TransformResult:
        reason: TransformErrorReason = {
            "reason": "validation_failed",
            "cause": "batch_too_large",
            "batch_size": batch_size,
            "expected": f"at most {_MAX_BATCH_ROWS} rows",
        }
        return TransformResult.error(reason, retryable=False)

    @staticmethod
    def _require_finite(value: float, *, operation: str) -> float:
        if not math.isfinite(value):
            raise OverflowError(operation)
        return value

    @staticmethod
    def _ks_statistic(left: tuple[int | float, ...], right: tuple[int | float, ...]) -> float:
        left_values = sorted(float(value) for value in left)
        right_values = sorted(float(value) for value in right)
        thresholds = sorted(set(left_values + right_values))
        max_distance = 0.0
        left_index = 0
        right_index = 0
        left_count = len(left_values)
        right_count = len(right_values)
        for threshold in thresholds:
            while left_index < left_count and left_values[left_index] <= threshold:
                left_index += 1
            while right_index < right_count and right_values[right_index] <= threshold:
                right_index += 1
            left_cdf = left_index / left_count
            right_cdf = right_index / right_count
            max_distance = max(max_distance, abs(left_cdf - right_cdf))
        return max_distance

    @staticmethod
    def _categorical_shifts(baseline: _CohortValues, cohort: _CohortValues) -> tuple[list[dict[str, object]], float, float, list[object]]:
        baseline_counts: Counter[ScalarBucketKey] = Counter(scalar_bucket_key(value) for value in baseline.values)
        cohort_counts: Counter[ScalarBucketKey] = Counter(scalar_bucket_key(value) for value in cohort.values)
        values_by_key: dict[ScalarBucketKey, object] = {}
        for value in baseline.values + cohort.values:
            values_by_key.setdefault(scalar_bucket_key(value), value)
        values = list(values_by_key.values())

        shifts: list[dict[str, object]] = []
        total_variation_sum = 0.0
        chi_square = 0.0
        new_categories: list[object] = []
        for value in values:
            value_key = scalar_bucket_key(value)
            baseline_count = baseline_counts[value_key]
            cohort_count = cohort_counts[value_key]
            baseline_prop = baseline_count / baseline.count
            cohort_prop = cohort_count / cohort.count
            total_variation_sum += abs(baseline_prop - cohort_prop)
            expected = baseline_prop * cohort.count
            if expected > 0:
                chi_square += ((cohort_count - expected) ** 2) / expected
            if baseline_count == 0 and cohort_count > 0:
                new_categories.append(value)
            shifts.append(
                {
                    "value": value,
                    "baseline_count": baseline_count,
                    "cohort_count": cohort_count,
                    "baseline_proportion": baseline_prop,
                    "cohort_proportion": cohort_prop,
                    "proportion_delta": cohort_prop - baseline_prop,
                }
            )

        return shifts, 0.5 * total_variation_sum, chi_square, new_categories

    def _base_result(self, *, baseline: _CohortValues, cohort: _CohortValues, batch_size: int) -> BatchDriftCompareRow:
        return {
            "cohort_field": self._cohort_field,
            "value_field": self._value_field,
            "value_type": self._value_type,
            "baseline_cohort": baseline.cohort,
            "cohort": cohort.cohort,
            "batch_size": batch_size,
            "baseline_total_count": baseline.total_count,
            "cohort_total_count": cohort.total_count,
            "baseline_count": baseline.count,
            "cohort_count": cohort.count,
            "baseline_missing_count": baseline.missing_count,
            "cohort_missing_count": cohort.missing_count,
            "baseline_non_finite_count": baseline.non_finite_count,
            "cohort_non_finite_count": cohort.non_finite_count,
        }

    def _numeric_result(self, *, baseline: _CohortValues, cohort: _CohortValues, batch_size: int) -> BatchDriftCompareRow:
        baseline_values = cast("tuple[int | float, ...]", baseline.values)
        cohort_values = cast("tuple[int | float, ...]", cohort.values)
        baseline_mean = self._require_finite(
            sum(float(value) for value in baseline_values) / baseline.count,
            operation="baseline_mean",
        )
        cohort_mean = self._require_finite(
            sum(float(value) for value in cohort_values) / cohort.count,
            operation="cohort_mean",
        )
        mean_delta = self._require_finite(cohort_mean - baseline_mean, operation="mean_delta")
        ks_statistic = self._require_finite(self._ks_statistic(baseline_values, cohort_values), operation="ks_statistic")
        result = self._base_result(baseline=baseline, cohort=cohort, batch_size=batch_size)
        result.update(
            {
                "baseline_mean": baseline_mean,
                "cohort_mean": cohort_mean,
                "mean_delta": mean_delta,
                "ks_statistic": ks_statistic,
            }
        )
        return result

    def _categorical_result(self, *, baseline: _CohortValues, cohort: _CohortValues, batch_size: int) -> BatchDriftCompareRow:
        shifts, total_variation, chi_square, new_categories = self._categorical_shifts(baseline, cohort)
        result = self._base_result(baseline=baseline, cohort=cohort, batch_size=batch_size)
        result.update(
            {
                "category_shifts": shifts,
                "total_variation": total_variation,
                "chi_square_statistic": chi_square,
                "new_category_count": len(new_categories),
                "new_categories": new_categories,
            }
        )
        return result

    def _comparison_for(
        self,
        *,
        baseline: _CohortValues,
        cohort: _CohortValues,
        batch_size: int,
    ) -> tuple[BatchDriftCompareRow, TransformResult | None]:
        if baseline.count == 0:
            return {}, self._error_for_no_values(baseline, batch_size=batch_size)
        if cohort.count == 0:
            return {}, self._error_for_no_values(cohort, batch_size=batch_size)
        if self._value_type == "numeric":
            try:
                return self._numeric_result(baseline=baseline, cohort=cohort, batch_size=batch_size), None
            except OverflowError as exc:
                reason: TransformErrorReason = {
                    "reason": "float_overflow",
                    "operation": str(exc) or "numeric_drift",
                    "group_value": cohort.cohort,
                    "value": str(baseline.cohort),
                }
                return {}, TransformResult.error(reason, retryable=False)
        return self._categorical_result(baseline=baseline, cohort=cohort, batch_size=batch_size), None

    def _output_contract_for(self, results: list[BatchDriftCompareRow]) -> SchemaContract:
        """Build one shared output contract for drift comparison rows."""
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
        """Compare baseline and current cohort distributions over a batch."""
        if not rows:
            return TransformResult.error({"reason": "empty_batch"}, retryable=False)
        if len(rows) > _MAX_BATCH_ROWS:
            return self._error_for_batch_too_large(batch_size=len(rows))

        non_finite_error = self._non_finite_group_key_error(rows)
        if non_finite_error is not None:
            return non_finite_error

        grouped = self._collect_cohorts(rows)
        cohort_values = [(cohort, self._values_for(cohort, cohort_rows)) for cohort, cohort_rows in grouped]
        baseline_cohort = self._baseline_cohort if self._baseline_cohort is not None else cohort_values[0][0]
        baseline = None
        for cohort, values in cohort_values:
            if same_scalar_bucket_value(cohort, baseline_cohort):
                baseline = values
                break
        if baseline is None:
            return TransformResult.error(
                {
                    "reason": "validation_failed",
                    "cause": "baseline_cohort_missing",
                    "expected": str(baseline_cohort),
                    "errors": [str(cohort) for cohort, _values in cohort_values],
                },
                retryable=False,
            )

        candidates = [(cohort, values) for cohort, values in cohort_values if not same_scalar_bucket_value(cohort, baseline_cohort)]
        if not candidates:
            return TransformResult.error(
                {
                    "reason": "validation_failed",
                    "cause": "insufficient_cohorts",
                    "count": len(cohort_values),
                },
                retryable=False,
            )

        results: list[BatchDriftCompareRow] = []
        for _cohort, values in candidates:
            comparison, error = self._comparison_for(baseline=baseline, cohort=values, batch_size=len(rows))
            if error is not None:
                return error
            results.append(comparison)

        output_contract = self._output_contract_for(results)
        fields_added = [field.normalized_name for field in output_contract.fields]
        pipeline_rows = [PipelineRow(result, output_contract) for result in results]

        if len(pipeline_rows) > 1:
            return TransformResult.success_multi(
                pipeline_rows,
                success_reason={"action": "processed", "fields_added": fields_added},
            )

        return TransformResult.success(
            pipeline_rows[0],
            success_reason={"action": "processed", "fields_added": fields_added},
        )

    def close(self) -> None:
        """No resources to release."""
        pass
