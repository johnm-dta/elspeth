"""Batch top-k frequency transform plugin."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
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
from elspeth.plugins.transforms._scalar_buckets import same_scalar_bucket_value

type TopKValue = str | int | float | bool | None
type BatchTopKRow = dict[str, object]

_TOP_K_OUTPUT_FIELDS = frozenset(
    {
        "batch_size",
        "count",
        "distinct_count",
        "field",
        "group_by",
        "group_value",
        "k",
        "missing_count",
        "non_finite_count",
        "top_values",
    }
)


@dataclass(slots=True)
class _FrequencyEntry:
    value: TopKValue
    count: int = 0


class BatchTopKConfig(TransformDataConfig):
    """Configuration for batch top-k transform."""

    field: str = Field(description="Field whose most frequent values should be summarized")
    k: int = Field(default=10, gt=0, description="Number of values to include")
    group_by: str | None = Field(default=None, description="Optional field that partitions the batch")
    include_missing: bool = Field(default=False, description="Whether None values are counted as a top-k value")

    @field_validator("field")
    @classmethod
    def _reject_empty_field(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field must not be empty")
        return v

    @field_validator("group_by")
    @classmethod
    def _reject_empty_group_by(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("group_by must not be empty")
        return v

    @model_validator(mode="after")
    def _reject_field_collision(self) -> BatchTopKConfig:
        if self.group_by is not None and self.group_by == self.field:
            raise ValueError("group_by and field must differ")
        return self


class BatchTopK(BaseTransform):
    """Report most frequent scalar values over a batch."""

    name = "batch_top_k"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:23e4c938c1cebcc8"
    config_model = BatchTopKConfig
    is_batch_aware = True

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Reports the most frequent scalar values in a batch.",
                composer_hints=(
                    "Use batch_top_k under aggregations with a trigger; it summarizes a flushed batch.",
                    "field must be scalar data such as str, int, float, bool, or None; arrays and objects are invalid.",
                    "Set include_missing=True only when missing values should appear in top_values.",
                    "Output is top-value summary row(s), not the original rows; group_by partitions the summary.",
                ),
            )
        return None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "field": "batch_top_k_probe_value",
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchTopKConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._field = cfg.field
        self._k = cfg.k
        self._group_by = cfg.group_by
        self._include_missing = cfg.include_missing
        self.declared_output_fields = _TOP_K_OUTPUT_FIELDS

        base_required = set(cfg.schema_config.required_fields or ())
        base_required.add(cfg.field)
        if cfg.group_by is not None:
            base_required.add(cfg.group_by)
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
            "BatchTopK",
            adds_fields=True,
        )
        self._output_schema_config = self._build_output_schema_config(schema_config)

    def _build_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
        """Describe top-k rows without propagating input fields."""
        return SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=tuple(sorted(self.declared_output_fields)),
            required_fields=None,
            audit_fields=None,
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the top-k output path for the backward invariant."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._field,
                value="probe",
            )
        ]

    @staticmethod
    def _is_non_finite_group_key(value: object) -> bool:
        if type(value) is float:
            return not math.isfinite(value)
        if type(value) is Decimal:
            return not value.is_finite()
        return False

    def _non_finite_group_key_error(self, rows: list[PipelineRow]) -> TransformResult | None:
        """Return an error if any row carries a non-finite group_by key (B4.5-d)."""
        if self._group_by is None:
            return None
        row_errors: list[RowErrorEntry] = []
        for row_index, row in enumerate(rows):
            if self._is_non_finite_group_key(row[self._group_by]):
                row_errors.append({"row_index": row_index, "reason": "non_finite_group_key"})
        if not row_errors:
            return None
        reason: TransformErrorReason = {
            "reason": "validation_failed",
            "cause": "non_finite_group_key",
            "field": self._group_by,
            "row_errors": row_errors,
        }
        return TransformResult.error(reason, retryable=False)

    def _group_rows(self, rows: list[PipelineRow]) -> list[tuple[object | None, list[tuple[int, PipelineRow]]]]:
        """Partition rows by group_by value while preserving first-seen order."""
        if self._group_by is None:
            return [(None, list(enumerate(rows)))]

        groups: list[tuple[object | None, list[tuple[int, PipelineRow]]]] = []
        for row_index, row in enumerate(rows):
            group_value = row[self._group_by]
            for existing_value, grouped_rows in groups:
                if same_scalar_bucket_value(group_value, existing_value):
                    grouped_rows.append((row_index, row))
                    break
            else:
                groups.append((group_value, [(row_index, row)]))
        return groups

    @staticmethod
    def _validate_value(value: object, *, field_name: str, row_index: int) -> TopKValue:
        if value is None:
            return None

        if type(value) not in (str, int, float, bool):
            raise TypeError(
                f"Field '{field_name}' must be a scalar top-k value (str, int, float, bool, or None), "
                f"got {type(value).__name__} in row {row_index}. "
                f"This indicates an upstream validation bug - check source schema or prior transforms."
            )

        return cast(TopKValue, value)

    @staticmethod
    def _increment_frequency(entries: list[_FrequencyEntry], value: TopKValue) -> None:
        # Type-identity gate: Python's value equality treats True/1, 1/1.0, and
        # 0/False as equal. The shared helper preserves the runtime scalar type
        # before equality so bool values never collapse into int buckets.
        for entry in entries:
            if same_scalar_bucket_value(entry.value, value):
                entry.count += 1
                return
        entries.append(_FrequencyEntry(value=value, count=1))

    def _top_k_row_for(self, group_value: object | None, grouped_rows: list[tuple[int, PipelineRow]]) -> BatchTopKRow:
        entries: list[_FrequencyEntry] = []
        missing_count = 0
        non_finite_count = 0

        for row_index, row in grouped_rows:
            value = self._validate_value(row[self._field], field_name=self._field, row_index=row_index)
            if value is None:
                missing_count += 1
                if not self._include_missing:
                    continue

            if type(value) is float and not math.isfinite(value):
                non_finite_count += 1
                continue

            self._increment_frequency(entries, value)

        count = sum(entry.count for entry in entries)
        sorted_entries = sorted(enumerate(entries), key=lambda item: (-item[1].count, item[0]))
        top_values = [
            {
                "value": entry.value,
                "count": entry.count,
                "rate": 0.0 if count == 0 else entry.count / count,
            }
            for _, entry in sorted_entries[: self._k]
        ]
        return {
            "field": self._field,
            "group_by": self._group_by,
            "group_value": group_value,
            "batch_size": len(grouped_rows),
            "count": count,
            "missing_count": missing_count,
            "non_finite_count": non_finite_count,
            "distinct_count": len(entries),
            "k": self._k,
            "top_values": top_values,
        }

    def _output_contract_for(self, results: list[BatchTopKRow]) -> SchemaContract:
        """Build one shared output contract for top-k rows."""
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
        """Compute top-k frequency rows over a batch."""
        if not rows:
            return TransformResult.error({"reason": "empty_batch"}, retryable=False)

        non_finite_error = self._non_finite_group_key_error(rows)
        if non_finite_error is not None:
            return non_finite_error

        results = [self._top_k_row_for(group_value, grouped_rows) for group_value, grouped_rows in self._group_rows(rows)]
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
