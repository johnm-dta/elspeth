"""Batch effect size transform plugin."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

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
from elspeth.plugins.transforms._scalar_buckets import same_scalar_bucket_value

type BatchEffectSizeRow = dict[str, object]

_EFFECT_SIZE_OUTPUT_FIELDS = frozenset(
    {
        "baseline_count",
        "baseline_mean",
        "baseline_missing_count",
        "baseline_non_finite_count",
        "baseline_stdev",
        "baseline_total_count",
        "baseline_variant",
        "batch_size",
        "cohens_d",
        "hedges_g",
        "mean_delta",
        "pooled_stdev",
        "score_field",
        "variant",
        "variant_count",
        "variant_field",
        "variant_mean",
        "variant_missing_count",
        "variant_non_finite_count",
        "variant_stdev",
        "variant_total_count",
    }
)


@dataclass(frozen=True, slots=True)
class _VariantStats:
    value: Any
    total_count: int
    values: tuple[int | float, ...]
    missing_indices: tuple[int, ...]
    non_finite_indices: tuple[int, ...]

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def missing_count(self) -> int:
        return len(self.missing_indices)

    @property
    def non_finite_count(self) -> int:
        return len(self.non_finite_indices)


class BatchEffectSizeConfig(TransformDataConfig):
    """Configuration for batch effect size transform."""

    variant_field: str = Field(description="Name of the field containing prompt/treatment variant labels")
    score_field: str = Field(description="Name of the numeric outcome field to compare")
    baseline_variant: str | int | bool | None = Field(
        default=None,
        description="Optional baseline variant. When omitted, the first-seen variant is the baseline.",
    )

    @field_validator("variant_field")
    @classmethod
    def _reject_empty_variant_field(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("variant_field must not be empty")
        return v

    @field_validator("score_field")
    @classmethod
    def _reject_empty_score_field(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("score_field must not be empty")
        return v

    @field_validator("baseline_variant")
    @classmethod
    def _reject_empty_baseline_variant(cls, v: str | int | bool | None) -> str | int | bool | None:
        if type(v) is str and not v.strip():
            raise ValueError("baseline_variant must not be empty")
        return v

    @model_validator(mode="after")
    def _reject_field_collision(self) -> BatchEffectSizeConfig:
        if self.variant_field == self.score_field:
            raise ValueError("variant_field and score_field must differ")
        return self


class BatchEffectSize(BaseTransform):
    """Compute Cohen's d and Hedges' g for batch variant comparisons."""

    name = "batch_effect_size"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:ead9c6381cdb2985"
    config_model = BatchEffectSizeConfig
    is_batch_aware = True

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Computes Cohen's d and Hedges' g between baseline and variant score groups.",
                composer_hints=(
                    "Use batch_effect_size under aggregations with a trigger; it summarizes complete batch groups.",
                    "variant_field and score_field must differ; baseline_variant defaults to the first-seen variant.",
                    "score_field must be finite numeric data; missing and non-finite scores are counted and may make a group invalid.",
                    "Output is effect-size comparison rows, not pass-through source rows.",
                ),
            )
        return None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "variant_field": "batch_effect_size_probe_variant",
            "score_field": "batch_effect_size_probe_score",
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchEffectSizeConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._variant_field = cfg.variant_field
        self._score_field = cfg.score_field
        self._baseline_variant = cfg.baseline_variant
        self.declared_output_fields = _EFFECT_SIZE_OUTPUT_FIELDS

        base_required = set(cfg.schema_config.required_fields or ())
        base_required.update({cfg.variant_field, cfg.score_field})
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
            "BatchEffectSize",
            adds_fields=True,
        )
        self._output_schema_config = self._build_output_schema_config(schema_config)

    def _build_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
        """Describe effect size rows without propagating input fields."""
        return SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=tuple(sorted(self.declared_output_fields)),
            required_fields=None,
            audit_fields=None,
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the effect-size output path for the backward invariant."""
        baseline = self._augment_invariant_probe_row(probe, field_name=self._variant_field, value="baseline")
        baseline = self._augment_invariant_probe_row(baseline, field_name=self._score_field, value=1.0)
        candidate = self._augment_invariant_probe_row(probe, field_name=self._variant_field, value="candidate")
        candidate = self._augment_invariant_probe_row(candidate, field_name=self._score_field, value=2.0)
        return [baseline, candidate]

    @staticmethod
    def _is_non_finite_variant(value: object) -> bool:
        if type(value) is float:
            return not math.isfinite(value)
        if type(value) is Decimal:
            return not value.is_finite()
        return False

    def _non_finite_variant_error(self, rows: list[PipelineRow]) -> TransformResult | None:
        row_errors: list[RowErrorEntry] = []
        for row_index, row in enumerate(rows):
            variant_value = row[self._variant_field]
            if self._is_non_finite_variant(variant_value):
                row_errors.append({"row_index": row_index, "reason": "non_finite_variant"})

        if not row_errors:
            return None

        reason: TransformErrorReason = {
            "reason": "validation_failed",
            "cause": "non_finite_variant",
            "field": self._variant_field,
            "row_errors": row_errors,
        }
        return TransformResult.error(reason, retryable=False)

    def _group_rows(self, rows: list[PipelineRow]) -> list[tuple[Any, list[tuple[int, PipelineRow]]]]:
        """Partition rows by variant while preserving first-seen order."""
        groups: list[tuple[Any, list[tuple[int, PipelineRow]]]] = []
        for row_index, row in enumerate(rows):
            variant_value = row[self._variant_field]
            for existing_value, grouped_rows in groups:
                if same_scalar_bucket_value(variant_value, existing_value):
                    grouped_rows.append((row_index, row))
                    break
            else:
                groups.append((variant_value, [(row_index, row)]))
        return groups

    def _stats_for_group(self, variant_value: Any, grouped_rows: list[tuple[int, PipelineRow]]) -> _VariantStats:
        values: list[int | float] = []
        missing_indices: list[int] = []
        non_finite_indices: list[int] = []

        for row_index, row in grouped_rows:
            raw_value = row[self._score_field]
            if raw_value is None:
                missing_indices.append(row_index)
                continue

            if type(raw_value) not in (int, float):
                raise TypeError(
                    f"Field '{self._score_field}' must be numeric (int or float), "
                    f"got {type(raw_value).__name__} in row {row_index}. "
                    f"This indicates an upstream validation bug - check source schema or prior transforms."
                )

            if type(raw_value) is float and not math.isfinite(raw_value):
                non_finite_indices.append(row_index)
                continue

            values.append(raw_value)

        return _VariantStats(
            value=variant_value,
            total_count=len(grouped_rows),
            values=tuple(values),
            missing_indices=tuple(missing_indices),
            non_finite_indices=tuple(non_finite_indices),
        )

    def _no_finite_score_error(self, stats: _VariantStats, *, baseline: bool) -> TransformResult:
        row_errors: list[RowErrorEntry] = []
        for row_index in stats.missing_indices:
            row_errors.append({"row_index": row_index, "reason": "missing_value"})
        for row_index in stats.non_finite_indices:
            row_errors.append({"row_index": row_index, "reason": "non_finite_value"})
        reason: TransformErrorReason = {
            "reason": "validation_failed",
            "cause": "baseline_has_no_finite_scores" if baseline else "variant_has_no_finite_scores",
            "group_value": stats.value,
            "total_count": stats.total_count,
            "valid_count": 0,
            "skipped_count": stats.missing_count + stats.non_finite_count,
            "row_errors": row_errors,
        }
        return TransformResult.error(reason, retryable=False)

    @staticmethod
    def _mean(stats: _VariantStats) -> float:
        return sum(stats.values) / stats.count

    @staticmethod
    def _stdev(stats: _VariantStats) -> float | None:
        if stats.count == 1:
            # stdev is undefined at n=1 -- emit None, never 0.0 (B4.5-a)
            return None
        return statistics.stdev(stats.values)

    @staticmethod
    def _require_finite(value: float | None, *, operation: str) -> float | None:
        if value is not None and not math.isfinite(value):
            raise OverflowError(operation)
        return value

    def _effect_size_row(
        self,
        *,
        batch_size: int,
        baseline: _VariantStats,
        variant: _VariantStats,
    ) -> tuple[BatchEffectSizeRow, TransformResult | None]:
        try:
            baseline_mean = self._require_finite(self._mean(baseline), operation="baseline_mean")
            variant_mean = self._require_finite(self._mean(variant), operation="variant_mean")
            # _stdev returns None at n=1 (undefined); _require_finite propagates None
            baseline_stdev = self._require_finite(self._stdev(baseline), operation="baseline_stdev")
            variant_stdev = self._require_finite(self._stdev(variant), operation="variant_stdev")
            assert baseline_mean is not None
            assert variant_mean is not None

            mean_delta = self._require_finite(variant_mean - baseline_mean, operation="mean_delta")
            assert mean_delta is not None
            pooled_variance_denominator = baseline.count + variant.count - 2
            pooled_stdev: float | None
            if pooled_variance_denominator <= 0:
                # Both groups are singletons (n1+n2-2 <= 0): the pooled dispersion
                # is UNDEFINED, not zero. Emit None rather than a misleading real
                # 0.0 -- consistent with baseline_stdev/variant_stdev being None at
                # n=1 (B4.5-b). cohens_d stays None (guarded below).
                pooled_stdev = None
            else:
                # When stdev is None (n=1) the (count-1)*stdev^2 term is 0*? = 0,
                # so treat None as 0.0 only for the pooled intermediate.
                baseline_stdev_val = baseline_stdev if baseline_stdev is not None else 0.0
                variant_stdev_val = variant_stdev if variant_stdev is not None else 0.0
                pooled_variance = (
                    ((baseline.count - 1) * (baseline_stdev_val**2)) + ((variant.count - 1) * (variant_stdev_val**2))
                ) / pooled_variance_denominator
                pooled_stdev_candidate = self._require_finite(math.sqrt(pooled_variance), operation="pooled_stdev")
                assert pooled_stdev_candidate is not None
                pooled_stdev = pooled_stdev_candidate

            cohens_d = None if (pooled_stdev is None or pooled_stdev == 0) else mean_delta / pooled_stdev
            cohens_d = self._require_finite(cohens_d, operation="cohens_d")
            hedges_denominator = (4 * (baseline.count + variant.count)) - 9
            if cohens_d is None or hedges_denominator <= 0:
                hedges_g = None
            else:
                hedges_g = cohens_d * (1 - (3 / hedges_denominator))
                hedges_g = self._require_finite(hedges_g, operation="hedges_g")
        except OverflowError as exc:
            reason: TransformErrorReason = {
                "reason": "float_overflow",
                "operation": str(exc) or "effect_size",
                "group_value": variant.value,
                "value": str(baseline.value),
            }
            return {}, TransformResult.error(reason, retryable=False)

        result: BatchEffectSizeRow = {
            "variant_field": self._variant_field,
            "score_field": self._score_field,
            "baseline_variant": baseline.value,
            "variant": variant.value,
            "batch_size": batch_size,
            "baseline_count": baseline.count,
            "variant_count": variant.count,
            "baseline_total_count": baseline.total_count,
            "variant_total_count": variant.total_count,
            "baseline_missing_count": baseline.missing_count,
            "variant_missing_count": variant.missing_count,
            "baseline_non_finite_count": baseline.non_finite_count,
            "variant_non_finite_count": variant.non_finite_count,
            "baseline_mean": baseline_mean,
            "variant_mean": variant_mean,
            "mean_delta": mean_delta,
            "baseline_stdev": baseline_stdev,
            "variant_stdev": variant_stdev,
            "pooled_stdev": pooled_stdev,
            "cohens_d": cohens_d,
            "hedges_g": hedges_g,
        }
        if baseline.missing_indices:
            result["baseline_missing_indices"] = list(baseline.missing_indices)
        if baseline.non_finite_indices:
            result["baseline_non_finite_indices"] = list(baseline.non_finite_indices)
        if variant.missing_indices:
            result["variant_missing_indices"] = list(variant.missing_indices)
        if variant.non_finite_indices:
            result["variant_non_finite_indices"] = list(variant.non_finite_indices)

        return result, None

    def _output_contract_for(self, results: list[BatchEffectSizeRow]) -> SchemaContract:
        """Build one shared output contract for effect size rows."""
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
        """Compute effect-size rows over a batch."""
        if not rows:
            return TransformResult.error({"reason": "empty_batch"}, retryable=False)

        non_finite_variant_error = self._non_finite_variant_error(rows)
        if non_finite_variant_error is not None:
            return non_finite_variant_error

        grouped = self._group_rows(rows)
        stats_by_variant = [self._stats_for_group(variant_value, grouped_rows) for variant_value, grouped_rows in grouped]

        if len(stats_by_variant) < 2:
            return TransformResult.error(
                {
                    "reason": "validation_failed",
                    "cause": "insufficient_variants",
                    "count": len(stats_by_variant),
                },
                retryable=False,
            )

        if self._baseline_variant is not None and not any(
            same_scalar_bucket_value(stats.value, self._baseline_variant) for stats in stats_by_variant
        ):
            return TransformResult.error(
                {
                    "reason": "validation_failed",
                    "cause": "baseline_variant_missing",
                    "expected": str(self._baseline_variant),
                    "message": f"Baseline variant {self._baseline_variant!r} was not present in the batch.",
                    "errors": [str(stats.value) for stats in stats_by_variant],
                },
                retryable=False,
            )

        baseline_value = self._baseline_variant if self._baseline_variant is not None else stats_by_variant[0].value
        baseline = next((stats for stats in stats_by_variant if same_scalar_bucket_value(stats.value, baseline_value)), None)
        if baseline is None:
            return TransformResult.error(
                {
                    "reason": "validation_failed",
                    "cause": "baseline_variant_missing",
                    "expected": str(baseline_value),
                    "message": f"Baseline variant {baseline_value!r} was not present in the batch.",
                    "errors": [str(stats.value) for stats in stats_by_variant],
                },
                retryable=False,
            )
        if baseline.count == 0:
            return self._no_finite_score_error(baseline, baseline=True)

        results: list[BatchEffectSizeRow] = []
        for variant in stats_by_variant:
            if same_scalar_bucket_value(variant.value, baseline.value):
                continue
            if variant.count == 0:
                return self._no_finite_score_error(variant, baseline=False)
            effect, error = self._effect_size_row(batch_size=len(rows), baseline=baseline, variant=variant)
            if error is not None:
                return error
            results.append(effect)

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
