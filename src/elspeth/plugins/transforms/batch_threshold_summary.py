"""Batch threshold summary transform plugin."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.errors import RowErrorEntry, TransformErrorReason
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult

type ThresholdOperator = Literal["<", "<=", ">", ">=", "==", "!="]
type BatchThresholdSummaryRow = dict[str, object]

_THRESHOLD_OUTPUT_FIELDS = frozenset(
    {
        "batch_size",
        "match_count",
        "match_rate",
        "missing_count",
        "non_finite_count",
        "non_match_count",
        "operator",
        "threshold",
        "threshold_name",
        "valid_count",
        "value_field",
    }
)
_MAX_THRESHOLDS = 128


class ThresholdSpec(BaseModel):
    """One named threshold to evaluate against the configured value field."""

    model_config = {"extra": "forbid", "frozen": True}

    name: str = Field(description="Stable label for this threshold")
    operator: ThresholdOperator = Field(description="Comparison operator")
    value: float = Field(description="Finite numeric threshold value")

    @field_validator("name")
    @classmethod
    def _reject_empty_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("threshold name must not be empty")
        return v

    @field_validator("value")
    @classmethod
    def _reject_non_finite_value(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("threshold value must be finite")
        return v


class BatchThresholdSummaryConfig(TransformDataConfig):
    """Configuration for batch threshold summary transform."""

    value_field: str = Field(description="Name of the numeric field to evaluate")
    thresholds: list[ThresholdSpec] = Field(description="Named thresholds to evaluate")

    @field_validator("value_field")
    @classmethod
    def _reject_empty_value_field(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("value_field must not be empty")
        return v

    @field_validator("thresholds")
    @classmethod
    def _reject_empty_thresholds(cls, v: list[ThresholdSpec]) -> list[ThresholdSpec]:
        if not v:
            raise ValueError("thresholds must contain at least one threshold")
        if len(v) > _MAX_THRESHOLDS:
            raise ValueError(f"thresholds must contain at most {_MAX_THRESHOLDS} entries")
        return v

    @model_validator(mode="after")
    def _reject_duplicate_threshold_names(self) -> BatchThresholdSummaryConfig:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for threshold in self.thresholds:
            if threshold.name in seen:
                duplicates.add(threshold.name)
            seen.add(threshold.name)
        if duplicates:
            raise ValueError(f"Duplicate threshold names: {', '.join(sorted(duplicates))}")
        return self


class BatchThresholdSummary(BaseTransform):
    """Report threshold match counts and rates for finite numeric batch values."""

    name = "batch_threshold_summary"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:93afacbaaa867db3"
    config_model = BatchThresholdSummaryConfig
    is_batch_aware = True

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Counts how many finite numeric batch values match named thresholds.",
                composer_hints=(
                    "Use batch_threshold_summary under aggregations with a trigger; it emits one summary row per threshold.",
                    "value_field must be numeric; missing and non-finite values are skipped and counted.",
                    "Each threshold needs a unique name, an operator from < <= > >= == !=, and a finite numeric value.",
                    "Output is threshold summary rows, not pass-through source data.",
                ),
            )
        return None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "value_field": "batch_threshold_summary_probe_value",
            "thresholds": [{"name": "positive", "operator": ">=", "value": 0.0}],
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchThresholdSummaryConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._value_field = cfg.value_field
        self._thresholds = tuple(cfg.thresholds)
        self.declared_output_fields = _THRESHOLD_OUTPUT_FIELDS

        base_required = set(cfg.schema_config.required_fields or ())
        base_required.add(cfg.value_field)
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
            "BatchThresholdSummary",
            adds_fields=True,
        )
        self._output_schema_config = self._build_output_schema_config(schema_config)

    def _build_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
        """Describe threshold summary rows without propagating input fields."""
        return SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=tuple(sorted(self.declared_output_fields)),
            required_fields=None,
            audit_fields=None,
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the threshold output path for the backward invariant."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._value_field,
                value=1.0,
            )
        ]

    def _finite_values_for(self, rows: list[PipelineRow]) -> tuple[list[int | float], list[int], list[int]]:
        values: list[int | float] = []
        missing_indices: list[int] = []
        non_finite_indices: list[int] = []

        for row_index, row in enumerate(rows):
            value = row[self._value_field]
            if value is None:
                missing_indices.append(row_index)
                continue

            if type(value) not in (int, float):
                raise TypeError(
                    f"Field '{self._value_field}' must be numeric (int or float), "
                    f"got {type(value).__name__} in row {row_index}. "
                    f"This indicates an upstream validation bug - check source schema or prior transforms."
                )

            if type(value) is float and not math.isfinite(value):
                non_finite_indices.append(row_index)
                continue

            values.append(value)

        return values, missing_indices, non_finite_indices

    def _error_for_no_finite_values(
        self,
        *,
        batch_size: int,
        missing_indices: list[int],
        non_finite_indices: list[int],
    ) -> TransformResult:
        row_errors: list[RowErrorEntry] = []
        for row_index in missing_indices:
            row_errors.append({"row_index": row_index, "reason": "missing_value"})
        for row_index in non_finite_indices:
            row_errors.append({"row_index": row_index, "reason": "non_finite_value"})
        reason: TransformErrorReason = {
            "reason": "validation_failed",
            "cause": "no_finite_values",
            "batch_size": batch_size,
            "valid_count": 0,
            "skipped_count": len(missing_indices) + len(non_finite_indices),
            "row_errors": row_errors,
        }
        return TransformResult.error(reason, retryable=False)

    @staticmethod
    def _matches(value: int | float, threshold: ThresholdSpec) -> bool:
        match threshold.operator:
            case "<":
                return value < threshold.value
            case "<=":
                return value <= threshold.value
            case ">":
                return value > threshold.value
            case ">=":
                return value >= threshold.value
            case "==":
                return value == threshold.value
            case "!=":
                return value != threshold.value

    def _summary_rows_for(
        self,
        *,
        batch_size: int,
        values: list[int | float],
        missing_count: int,
        non_finite_count: int,
    ) -> list[BatchThresholdSummaryRow]:
        valid_count = len(values)
        results: list[BatchThresholdSummaryRow] = []
        for threshold in self._thresholds:
            match_count = sum(1 for value in values if self._matches(value, threshold))
            results.append(
                {
                    "value_field": self._value_field,
                    "threshold_name": threshold.name,
                    "operator": threshold.operator,
                    "threshold": threshold.value,
                    "batch_size": batch_size,
                    "valid_count": valid_count,
                    "missing_count": missing_count,
                    "non_finite_count": non_finite_count,
                    "match_count": match_count,
                    "non_match_count": valid_count - match_count,
                    "match_rate": match_count / valid_count,
                }
            )
        return results

    def _output_contract_for(self, results: list[BatchThresholdSummaryRow]) -> SchemaContract:
        """Build one shared output contract for threshold summary rows."""
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
        """Compute threshold summaries over a batch."""
        if not rows:
            return TransformResult.error({"reason": "empty_batch"}, retryable=False)

        values, missing_indices, non_finite_indices = self._finite_values_for(rows)
        if not values:
            return self._error_for_no_finite_values(
                batch_size=len(rows),
                missing_indices=missing_indices,
                non_finite_indices=non_finite_indices,
            )

        results = self._summary_rows_for(
            batch_size=len(rows),
            values=values,
            missing_count=len(missing_indices),
            non_finite_count=len(non_finite_indices),
        )
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
