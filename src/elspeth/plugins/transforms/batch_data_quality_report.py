"""Batch data quality report transform plugin.

Summarizes completeness and scalar validity for configured fields over one
aggregation batch. The plugin is intentionally shape-changing: it emits one
quality row per inspected field rather than passing through source rows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pydantic import Field, field_validator, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.transforms._scalar_buckets import append_unique_bucket_value

type BatchDataQualityReportRow = dict[str, object]

_QUALITY_OUTPUT_FIELDS = frozenset(
    {
        "batch_size",
        "blank_string_count",
        "blank_string_rate",
        "distinct_count",
        "duplicate_count",
        "field",
        "missing_count",
        "missing_rate",
        "non_finite_count",
        "non_scalar_count",
        "observed_count",
        "observed_type_counts",
        "valid_count",
        "valid_rate",
    }
)


@dataclass(frozen=True, slots=True)
class _QualityStats:
    missing_count: int
    blank_string_count: int
    non_finite_count: int
    non_scalar_count: int
    observed_type_counts: Mapping[str, int]
    valid_values: tuple[object, ...]

    def __post_init__(self) -> None:
        # ``observed_type_counts`` is Mapping[str, int]; producers may
        # pass a mutable dict. ``valid_values`` is tuple[object, ...] —
        # the tuple itself is immutable, but the elements are
        # ``object`` (typically scalar bucket entries from
        # ``append_unique_bucket_value``); deep_freeze is identity-
        # preserving for already-immutable scalars and would crash
        # loudly if a future caller passed a nested mutable.
        freeze_fields(self, "observed_type_counts", "valid_values")

    @property
    def observed_count(self) -> int:
        return sum(self.observed_type_counts.values())

    @property
    def valid_count(self) -> int:
        return len(self.valid_values)


class BatchDataQualityReportConfig(TransformDataConfig):
    """Configuration for batch data quality report transform."""

    inspect_fields: list[str] = Field(description="Fields to profile for missing, invalid, and duplicate values")

    @field_validator("inspect_fields")
    @classmethod
    def _reject_empty_inspect_fields(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("inspect_fields must contain at least one field")
        for index, field_name in enumerate(v):
            if not field_name.strip():
                raise ValueError(f"inspect_fields[{index}] must not be empty")
        return v

    @model_validator(mode="after")
    def _reject_duplicate_inspect_fields(self) -> BatchDataQualityReportConfig:
        duplicates = sorted({field_name for field_name in self.inspect_fields if self.inspect_fields.count(field_name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate inspect_fields values: {', '.join(duplicates)}")
        return self


class BatchDataQualityReport(BaseTransform):
    """Report field-level batch quality counts and rates."""

    name = "batch_data_quality_report"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:5f69026e933b8bbd"
    config_model = BatchDataQualityReportConfig
    is_batch_aware = True

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Emits data-quality counts for configured fields across a batch.",
                composer_hints=(
                    "Use batch_data_quality_report under aggregations with a trigger; it inspects a flushed batch.",
                    "inspect_fields must name existing input fields and must not be empty or duplicated.",
                    "It emits one report row per inspected field with missing, blank, non-finite, non-scalar, and type counts.",
                    "Output rows replace the source row shape; downstream stages should consume fields like field, missing_count, valid_rate, and batch_size.",
                ),
            )
        return None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "inspect_fields": ["batch_data_quality_report_probe_value"],
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchDataQualityReportConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._inspect_fields = tuple(cfg.inspect_fields)
        self.declared_output_fields = _QUALITY_OUTPUT_FIELDS

        base_required = set(cfg.schema_config.required_fields or ())
        base_required.update(cfg.inspect_fields)
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
            "BatchDataQualityReport",
            adds_fields=True,
        )
        self._output_schema_config = self._build_output_schema_config(schema_config)

    def _build_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
        """Describe quality report rows without propagating input fields."""
        return SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=tuple(sorted(self.declared_output_fields)),
            required_fields=None,
            audit_fields=None,
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the report output path for the backward invariant."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._inspect_fields[0],
                value=1.0,
            )
        ]

    @staticmethod
    def _is_valid_scalar(value: object) -> bool:
        if type(value) is str:
            return bool(value.strip())
        if type(value) is float:
            return math.isfinite(value)
        return type(value) in (int, bool)

    @staticmethod
    def _distinct_count(values: tuple[object, ...]) -> int:
        distinct_values: list[object] = []
        for value in values:
            append_unique_bucket_value(distinct_values, value)
        return len(distinct_values)

    def _stats_for_field(self, rows: list[PipelineRow], field_name: str) -> _QualityStats:
        missing_count = 0
        blank_string_count = 0
        non_finite_count = 0
        non_scalar_count = 0
        observed_type_counts: dict[str, int] = {}
        valid_values: list[object] = []

        for row in rows:
            value = row[field_name]
            if value is None:
                missing_count += 1
                continue

            type_name = type(value).__name__
            if type_name in observed_type_counts:
                observed_type_counts[type_name] += 1
            else:
                observed_type_counts[type_name] = 1

            if type(value) is str and not value.strip():
                blank_string_count += 1
                continue

            if type(value) is float and not math.isfinite(value):
                non_finite_count += 1
                continue

            if type(value) not in (str, int, float, bool):
                non_scalar_count += 1
                continue

            if self._is_valid_scalar(value):
                valid_values.append(value)

        return _QualityStats(
            missing_count=missing_count,
            blank_string_count=blank_string_count,
            non_finite_count=non_finite_count,
            non_scalar_count=non_scalar_count,
            observed_type_counts=MappingProxyType(observed_type_counts.copy()),
            valid_values=tuple(valid_values),
        )

    def _quality_row_for(self, rows: list[PipelineRow], field_name: str) -> BatchDataQualityReportRow:
        stats = self._stats_for_field(rows, field_name)
        batch_size = len(rows)
        distinct_count = self._distinct_count(stats.valid_values)
        valid_count = stats.valid_count
        return {
            "field": field_name,
            "batch_size": batch_size,
            "missing_count": stats.missing_count,
            "missing_rate": stats.missing_count / batch_size,
            "observed_count": stats.observed_count,
            "blank_string_count": stats.blank_string_count,
            "blank_string_rate": stats.blank_string_count / batch_size,
            "non_finite_count": stats.non_finite_count,
            "non_scalar_count": stats.non_scalar_count,
            "valid_count": valid_count,
            "valid_rate": valid_count / batch_size,
            "distinct_count": distinct_count,
            "duplicate_count": valid_count - distinct_count,
            "observed_type_counts": dict(stats.observed_type_counts),
        }

    def _output_contract_for(self, results: list[BatchDataQualityReportRow]) -> SchemaContract:
        """Build one shared output contract for quality report rows."""
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
        """Compute quality report rows over a batch."""
        if not rows:
            return TransformResult.error({"reason": "empty_batch"}, retryable=False)

        results = [self._quality_row_for(rows, field_name) for field_name in self._inspect_fields]
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
