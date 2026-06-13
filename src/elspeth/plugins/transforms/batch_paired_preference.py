"""Batch paired preference transform plugin."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
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
from elspeth.plugins.transforms._scalar_buckets import (
    append_unique_bucket_value,
    same_scalar_bucket_value,
    scalar_bucket_contains,
)

type BatchPairedPreferenceRow = dict[str, object]

_PAIRED_OUTPUT_FIELDS = frozenset(
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


@dataclass(frozen=True, slots=True)
class _ScoreEntry:
    variant: Any
    score: int | float | None
    missing: bool = False
    non_finite: bool = False


class BatchPairedPreferenceConfig(TransformDataConfig):
    """Configuration for batch paired preference transform."""

    pair_field: str = Field(description="Field that identifies paired evaluation items")
    variant_field: str = Field(description="Field that identifies the prompt/model/treatment variant")
    score_field: str = Field(description="Numeric score field to compare within each pair")
    baseline_variant: str | int | bool | None = Field(
        default=None,
        description="Optional baseline variant. When omitted, the first-seen variant is the baseline.",
    )

    @field_validator("pair_field")
    @classmethod
    def _reject_empty_pair_field(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("pair_field must not be empty")
        return v

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
    def _reject_field_collisions(self) -> BatchPairedPreferenceConfig:
        if len({self.pair_field, self.variant_field, self.score_field}) != 3:
            raise ValueError("pair_field, variant_field, and score_field must be distinct")
        return self


class BatchPairedPreference(BaseTransform):
    """Compare paired variant scores over an aggregation batch."""

    name = "batch_paired_preference"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:02cf90a2063b06d7"
    config_model = BatchPairedPreferenceConfig
    is_batch_aware = True

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Compares paired variant scores and emits win/loss/tie preference metrics.",
                composer_hints=(
                    "Use batch_paired_preference under aggregations with a trigger; it compares variants within pair_field groups.",
                    "pair_field, variant_field, and score_field must be three distinct fields.",
                    "baseline_variant defaults to the first-seen variant; incomplete pairs are counted but not compared.",
                    "Output is paired preference metrics, not the original rows.",
                ),
            )
        return None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "pair_field": "batch_paired_preference_probe_pair",
            "variant_field": "batch_paired_preference_probe_variant",
            "score_field": "batch_paired_preference_probe_score",
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchPairedPreferenceConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._pair_field = cfg.pair_field
        self._variant_field = cfg.variant_field
        self._score_field = cfg.score_field
        self._baseline_variant = cfg.baseline_variant
        self.declared_output_fields = _PAIRED_OUTPUT_FIELDS

        base_required = set(cfg.schema_config.required_fields or ())
        base_required.update({cfg.pair_field, cfg.variant_field, cfg.score_field})
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
            "BatchPairedPreference",
            adds_fields=True,
        )
        self._output_schema_config = self._build_output_schema_config(schema_config)

    def _build_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
        """Describe paired comparison rows without propagating input fields."""
        return SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=tuple(sorted(self.declared_output_fields)),
            required_fields=None,
            audit_fields=None,
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the paired comparison output path for the backward invariant."""
        baseline = self._augment_invariant_probe_row(probe, field_name=self._pair_field, value="probe")
        baseline = self._augment_invariant_probe_row(baseline, field_name=self._variant_field, value="baseline")
        baseline = self._augment_invariant_probe_row(baseline, field_name=self._score_field, value=1.0)
        candidate = self._augment_invariant_probe_row(probe, field_name=self._pair_field, value="probe")
        candidate = self._augment_invariant_probe_row(candidate, field_name=self._variant_field, value="candidate")
        candidate = self._augment_invariant_probe_row(candidate, field_name=self._score_field, value=2.0)
        return [baseline, candidate]

    def _score_entry_for(self, row: PipelineRow, *, row_index: int) -> _ScoreEntry:
        raw_score = row[self._score_field]
        variant = row[self._variant_field]

        if raw_score is None:
            return _ScoreEntry(variant=variant, score=None, missing=True)

        if type(raw_score) not in (int, float):
            raise TypeError(
                f"Field '{self._score_field}' must be numeric (int or float), "
                f"got {type(raw_score).__name__} in row {row_index}. "
                f"This indicates an upstream validation bug - check source schema or prior transforms."
            )

        if type(raw_score) is float and not math.isfinite(raw_score):
            return _ScoreEntry(variant=variant, score=None, non_finite=True)

        return _ScoreEntry(variant=variant, score=raw_score)

    @staticmethod
    def _is_non_finite_variant(value: object) -> bool:
        if type(value) is float:
            return not math.isfinite(value)
        return False

    def _non_finite_variant_error(self, rows: list[PipelineRow]) -> TransformResult | None:
        """Return an error if any row carries a non-finite variant key (B4.5-d)."""
        row_errors: list[RowErrorEntry] = []
        for row_index, row in enumerate(rows):
            if self._is_non_finite_variant(row[self._variant_field]):
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

    def _collect_pairs(self, rows: list[PipelineRow]) -> tuple[list[tuple[Any, list[_ScoreEntry]]], list[Any]]:
        pairs: list[tuple[Any, list[_ScoreEntry]]] = []
        variants: list[Any] = []

        for row_index, row in enumerate(rows):
            pair_id = row[self._pair_field]
            entry = self._score_entry_for(row, row_index=row_index)

            append_unique_bucket_value(variants, entry.variant)

            for existing_pair, entries in pairs:
                if same_scalar_bucket_value(pair_id, existing_pair):
                    entries.append(entry)
                    break
            else:
                pairs.append((pair_id, [entry]))

        return pairs, variants

    @staticmethod
    def _find_variant_entry(entries: list[_ScoreEntry], variant: Any) -> _ScoreEntry | None:
        for entry in entries:
            if same_scalar_bucket_value(entry.variant, variant):
                return entry
        return None

    @staticmethod
    def _mean(values: list[int | float]) -> float:
        return sum(values) / len(values)

    @staticmethod
    def _standard_error(values: list[float]) -> float | None:
        if len(values) <= 1:
            # se undefined at n<=1 -- emit None, never 0.0 (B4.5-a)
            return None
        return statistics.stdev(values) / math.sqrt(len(values))

    @staticmethod
    def _require_finite(value: float, *, operation: str) -> float:
        if not math.isfinite(value):
            raise OverflowError(operation)
        return value

    def _comparison_for(
        self,
        *,
        pairs: list[tuple[Any, list[_ScoreEntry]]],
        baseline_variant: Any,
        variant: Any,
        batch_size: int,
    ) -> tuple[BatchPairedPreferenceRow, TransformResult | None]:
        baseline_scores: list[int | float] = []
        variant_scores: list[int | float] = []
        deltas: list[float] = []
        incomplete_pairs: list[Any] = []
        missing_score_count = 0
        non_finite_score_count = 0
        wins = 0
        losses = 0
        ties = 0

        for pair_id, entries in pairs:
            baseline = self._find_variant_entry(entries, baseline_variant)
            candidate = self._find_variant_entry(entries, variant)
            if baseline is None or candidate is None:
                incomplete_pairs.append(pair_id)
                continue

            missing_score_count += int(baseline.missing) + int(candidate.missing)
            non_finite_score_count += int(baseline.non_finite) + int(candidate.non_finite)
            if baseline.score is None or candidate.score is None:
                incomplete_pairs.append(pair_id)
                continue

            try:
                delta = self._require_finite(float(candidate.score - baseline.score), operation="paired_delta")
            except OverflowError as exc:
                overflow_reason: TransformErrorReason = {
                    "reason": "float_overflow",
                    "operation": str(exc) or "paired_delta",
                    "group_value": variant,
                    "value": str(baseline_variant),
                }
                return {}, TransformResult.error(overflow_reason, retryable=False)

            baseline_scores.append(baseline.score)
            variant_scores.append(candidate.score)
            deltas.append(delta)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1

        compared_count = len(deltas)
        if compared_count == 0:
            no_complete_pairs_reason: TransformErrorReason = {
                "reason": "validation_failed",
                "cause": "no_complete_pairs",
                "group_value": variant,
                "batch_size": batch_size,
                "count": len(pairs),
            }
            return {}, TransformResult.error(no_complete_pairs_reason, retryable=False)

        try:
            mean_delta = self._require_finite(sum(deltas) / compared_count, operation="mean_paired_delta")
            baseline_mean = self._require_finite(self._mean(baseline_scores), operation="baseline_mean")
            variant_mean = self._require_finite(self._mean(variant_scores), operation="variant_mean")
            # _standard_error returns None when n<=1 (undefined) -- CI bounds are
            # also undefined in that case; emit None (honest-absence, B4.5-a)
            se_raw = self._standard_error(deltas)
            standard_error_delta: float | None
            confidence_95_low: float | None
            confidence_95_high: float | None
            if se_raw is None:
                standard_error_delta = None
                confidence_95_low = None
                confidence_95_high = None
            else:
                standard_error_delta = self._require_finite(se_raw, operation="standard_error_delta")
                confidence_95_low = self._require_finite(mean_delta - 1.96 * standard_error_delta, operation="confidence_95_low")
                confidence_95_high = self._require_finite(mean_delta + 1.96 * standard_error_delta, operation="confidence_95_high")
        except OverflowError as exc:
            aggregate_overflow_reason: TransformErrorReason = {
                "reason": "float_overflow",
                "operation": str(exc) or "paired_preference",
                "group_value": variant,
                "value": str(baseline_variant),
            }
            return {}, TransformResult.error(aggregate_overflow_reason, retryable=False)

        preference_denominator = wins + losses
        # preference_rate = wins/(wins+losses) is 0/0 when all pairs tie --
        # emit None (honest-absence), never 0.0 (B4.5-b)
        preference_rate: float | None = None if preference_denominator == 0 else wins / preference_denominator

        result: BatchPairedPreferenceRow = {
            "pair_field": self._pair_field,
            "variant_field": self._variant_field,
            "score_field": self._score_field,
            "baseline_variant": baseline_variant,
            "variant": variant,
            "batch_size": batch_size,
            "total_pair_count": len(pairs),
            "compared_pair_count": compared_count,
            "incomplete_pair_count": len(incomplete_pairs),
            "missing_score_count": missing_score_count,
            "non_finite_score_count": non_finite_score_count,
            "baseline_mean": baseline_mean,
            "variant_mean": variant_mean,
            "mean_paired_delta": mean_delta,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / compared_count,
            "loss_rate": losses / compared_count,
            "tie_rate": ties / compared_count,
            "preference_rate": preference_rate,
            "standard_error_delta": standard_error_delta,
            "confidence_95_low": confidence_95_low,
            "confidence_95_high": confidence_95_high,
        }
        if incomplete_pairs:
            result["incomplete_pairs"] = incomplete_pairs
        return result, None

    def _output_contract_for(self, results: list[BatchPairedPreferenceRow]) -> SchemaContract:
        """Build one shared output contract for paired preference rows."""
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
        """Compute paired preference summaries over a batch."""
        if not rows:
            return TransformResult.error({"reason": "empty_batch"}, retryable=False)

        non_finite_error = self._non_finite_variant_error(rows)
        if non_finite_error is not None:
            return non_finite_error

        pairs, variants = self._collect_pairs(rows)
        baseline_variant = self._baseline_variant if self._baseline_variant is not None else variants[0]
        if not scalar_bucket_contains(variants, baseline_variant):
            return TransformResult.error(
                {
                    "reason": "validation_failed",
                    "cause": "baseline_variant_missing",
                    "expected": str(baseline_variant),
                    "errors": [str(variant) for variant in variants],
                },
                retryable=False,
            )

        candidates = [variant for variant in variants if not same_scalar_bucket_value(variant, baseline_variant)]
        if not candidates:
            return TransformResult.error(
                {
                    "reason": "validation_failed",
                    "cause": "insufficient_variants",
                    "count": len(variants),
                },
                retryable=False,
            )

        results: list[BatchPairedPreferenceRow] = []
        for variant in candidates:
            comparison, error = self._comparison_for(
                pairs=pairs,
                baseline_variant=baseline_variant,
                variant=variant,
                batch_size=len(rows),
            )
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
