"""Batch outlier annotation transform plugin."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any

from pydantic import Field, field_validator, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.errors import PluginContractViolation, RowErrorEntry, TransformErrorReason
from elspeth.contracts.field_collision import detect_field_collisions
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import PluginConfigError, TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult

type BatchOutlierAnnotationRow = dict[str, object]

_ANNOTATION_FIELD_SUFFIXES = (
    "batch_size",
    "is_outlier",
    "mad",
    "mean",
    "median",
    "missing_count",
    "missing_indices",
    "non_finite_count",
    "non_finite_indices",
    "reason",
    "robust_z_score",
    "robust_z_threshold",
    "row_index",
    "skipped_count",
    "stdev",
    "valid_count",
    "value",
    "value_field",
    "z_score",
    "z_threshold",
)
_BACKWARD_INVARIANT_DROPPED_FIELD = "batch_outlier_annotator_dropped_probe_field"


def _annotation_fields(output_prefix: str) -> frozenset[str]:
    return frozenset(f"{output_prefix}_{suffix}" for suffix in _ANNOTATION_FIELD_SUFFIXES)


@dataclass(frozen=True, slots=True)
class _FiniteEntry:
    row_index: int
    row: PipelineRow
    value: int | float


@dataclass(frozen=True, slots=True)
class _BatchStats:
    batch_size: int
    mean: float
    median: float
    stdev: float
    mad: float
    mean_abs_dev: float
    missing_indices: tuple[int, ...]
    non_finite_indices: tuple[int, ...]

    @property
    def valid_count(self) -> int:
        return self.batch_size - self.missing_count - self.non_finite_count

    @property
    def missing_count(self) -> int:
        return len(self.missing_indices)

    @property
    def non_finite_count(self) -> int:
        return len(self.non_finite_indices)

    @property
    def skipped_count(self) -> int:
        return self.missing_count + self.non_finite_count


class BatchOutlierAnnotatorConfig(TransformDataConfig):
    """Configuration for batch outlier annotator transform."""

    value_field: str = Field(description="Name of the numeric field to annotate")
    output_prefix: str = Field(
        default="outlier",
        description="Prefix used for emitted annotation fields",
    )
    z_threshold: float = Field(
        default=3.0,
        gt=0,
        description="Absolute sample z-score threshold for outlier annotation",
    )
    robust_z_threshold: float = Field(
        default=3.5,
        gt=0,
        description="Absolute modified z-score threshold based on median absolute deviation",
    )

    @field_validator("value_field")
    @classmethod
    def _reject_empty_value_field(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("value_field must not be empty")
        return v

    @field_validator("output_prefix")
    @classmethod
    def _reject_invalid_output_prefix(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("output_prefix must not be empty")
        if not v.isidentifier():
            raise ValueError("output_prefix must be a valid Python identifier prefix")
        return v

    @model_validator(mode="after")
    def _reject_value_field_collision(self) -> BatchOutlierAnnotatorConfig:
        output_fields = _annotation_fields(self.output_prefix)
        if self.value_field in output_fields:
            raise ValueError(
                f"value_field '{self.value_field}' collides with outlier annotation output key. "
                f"Choose a value_field that is not one of: {', '.join(sorted(output_fields))}"
            )
        return self


class BatchOutlierAnnotator(BaseTransform):
    """Annotate batch rows with z-score and robust-z outlier signals.

    The transform emits one row per finite numeric input value. Missing values
    and non-finite floats are skipped and reported on every emitted row so the
    audit trail remains canonical while preserving data-quality context.
    """

    name = "batch_outlier_annotator"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:79c0566e539c382b"
    config_model = BatchOutlierAnnotatorConfig
    is_batch_aware = True
    passes_through_input = False

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Annotates finite numeric rows with batch z-score and robust-z outlier fields.",
                composer_hints=(
                    "Use batch_outlier_annotator under aggregations with a trigger; it needs the batch distribution.",
                    "value_field must be numeric; missing and non-finite values are skipped and reported.",
                    "output_prefix creates many annotation fields, so choose a prefix that cannot collide with input fields.",
                    "It emits one annotated row per finite input value and may drop skipped rows from success output.",
                ),
            )
        return None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "value_field": "batch_outlier_annotator_probe_value",
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchOutlierAnnotatorConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._value_field = cfg.value_field
        self._output_prefix = cfg.output_prefix
        self._z_threshold = cfg.z_threshold
        self._robust_z_threshold = cfg.robust_z_threshold
        self.declared_output_fields = _annotation_fields(cfg.output_prefix)

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
        self._reject_explicit_output_field_collision(cfg)
        self.input_schema, self.output_schema = self._create_schemas(
            schema_config,
            "BatchOutlierAnnotator",
            adds_fields=True,
        )
        self._output_schema_config = self._build_output_schema_config(schema_config)

    def _reject_explicit_output_field_collision(self, cfg: BatchOutlierAnnotatorConfig) -> None:
        """Reject explicit schemas that would always collide with annotations."""
        if cfg.schema_config.fields is None:
            return

        declared_schema_fields = {field.name for field in cfg.schema_config.fields}
        collisions = detect_field_collisions(declared_schema_fields, self.declared_output_fields)
        if collisions is None:
            return

        cause = (
            "BatchOutlierAnnotator schema declares field(s) "
            f"{collisions!r}, but annotation output would overwrite them. "
            "Remove the colliding field from the explicit schema or choose a different output_prefix."
        )
        raise PluginConfigError(
            cause,
            cause=cause,
            plugin_class=self.config_model.__name__,
            plugin_name=self.name,
            component_type="transform",
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the skipped-row path for the backward invariant."""
        dropped_row = self._augment_invariant_probe_row(
            probe,
            field_name=_BACKWARD_INVARIANT_DROPPED_FIELD,
            value=True,
        )
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._value_field,
                value=1.0,
            ),
            self._augment_invariant_probe_row(
                dropped_row,
                field_name=self._value_field,
                value=None,
            ),
        ]

    def _reject_runtime_output_field_collision(self, rows: list[PipelineRow]) -> None:
        for row_index, row in enumerate(rows):
            collisions = detect_field_collisions(set(row.keys()), self.declared_output_fields)
            if collisions is None:
                continue
            raise PluginContractViolation(
                f"Transform '{self.name}' would overwrite existing input fields {collisions} "
                f"in row {row_index}. This is a pipeline configuration error — choose a different output_prefix."
            )

    def _finite_entries_for(
        self,
        rows: list[PipelineRow],
    ) -> tuple[list[_FiniteEntry], tuple[int, ...], tuple[int, ...]]:
        entries: list[_FiniteEntry] = []
        missing_indices: list[int] = []
        non_finite_indices: list[int] = []

        for row_index, row in enumerate(rows):
            raw_value = row[self._value_field]
            if raw_value is None:
                missing_indices.append(row_index)
                continue

            if type(raw_value) not in (int, float):
                raise TypeError(
                    f"Field '{self._value_field}' must be numeric (int or float), "
                    f"got {type(raw_value).__name__} in row {row_index}. "
                    f"This indicates an upstream validation bug - check source schema or prior transforms."
                )

            if type(raw_value) is float and not math.isfinite(raw_value):
                non_finite_indices.append(row_index)
                continue

            entries.append(_FiniteEntry(row_index=row_index, row=row, value=raw_value))

        return entries, tuple(missing_indices), tuple(non_finite_indices)

    @staticmethod
    def _error_for_no_finite_values(
        *,
        batch_size: int,
        missing_indices: tuple[int, ...],
        non_finite_indices: tuple[int, ...],
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
    def _require_finite_float(value: float, *, operation: str) -> float:
        if not math.isfinite(value):
            raise OverflowError(operation)
        return value

    @classmethod
    def _coerce_finite_float(cls, value: int | float, *, operation: str) -> float:
        try:
            coerced = float(value)
        except OverflowError as exc:
            raise OverflowError(operation) from exc
        return cls._require_finite_float(coerced, operation=operation)

    @staticmethod
    def _float_overflow_error(*, operation: str, batch_size: int, valid_count: int) -> TransformResult:
        reason: TransformErrorReason = {
            "reason": "float_overflow",
            "operation": operation or "outlier_annotation",
            "batch_size": batch_size,
            "valid_count": valid_count,
        }
        return TransformResult.error(reason, retryable=False)

    @staticmethod
    def _median(values: list[float]) -> float:
        return float(statistics.median(values))

    def _stats_for(
        self,
        *,
        batch_size: int,
        entries: list[_FiniteEntry],
        missing_indices: tuple[int, ...],
        non_finite_indices: tuple[int, ...],
    ) -> tuple[_BatchStats, TransformResult | None]:
        try:
            values = [self._coerce_finite_float(entry.value, operation="float_conversion") for entry in entries]
            mean = self._require_finite_float(sum(values) / len(values), operation="mean")
            median = self._require_finite_float(self._median(values), operation="median")
            deviations = [abs(value - median) for value in values]
            mad = self._require_finite_float(self._median(deviations), operation="mad")
            mean_abs_dev = self._require_finite_float(sum(deviations) / len(deviations), operation="mean_abs_dev")
            stdev = 0.0 if len(values) == 1 else self._require_finite_float(statistics.stdev(values), operation="stdev")
        except OverflowError as exc:
            return (
                _BatchStats(
                    batch_size=batch_size,
                    mean=0.0,
                    median=0.0,
                    stdev=0.0,
                    mad=0.0,
                    mean_abs_dev=0.0,
                    missing_indices=missing_indices,
                    non_finite_indices=non_finite_indices,
                ),
                self._float_overflow_error(
                    operation=str(exc),
                    batch_size=batch_size,
                    valid_count=len(entries),
                ),
            )

        return (
            _BatchStats(
                batch_size=batch_size,
                mean=mean,
                median=median,
                stdev=stdev,
                mad=mad,
                mean_abs_dev=mean_abs_dev,
                missing_indices=missing_indices,
                non_finite_indices=non_finite_indices,
            ),
            None,
        )

    def _field(self, suffix: str) -> str:
        return f"{self._output_prefix}_{suffix}"

    def _annotation_for(self, entry: _FiniteEntry, stats: _BatchStats) -> dict[str, object]:
        value = self._coerce_finite_float(entry.value, operation="float_conversion")
        z_score = 0.0 if stats.stdev == 0.0 else self._require_finite_float((value - stats.mean) / stats.stdev, operation="z_score")
        robust_z_score: float | None
        if stats.mad != 0.0:
            robust_z_score = self._require_finite_float(0.6745 * (value - stats.median) / stats.mad, operation="robust_z_score")
        elif stats.mean_abs_dev != 0.0:
            # Iglewicz-Hoaglin fallback: MAD collapses to 0 whenever >50% of values
            # are identical (common for score/count data), but real spread remains.
            # The mean-absolute-deviation modified z-score keeps robust detection
            # alive instead of fabricating 0.0 for a masked outlier.
            robust_z_score = self._require_finite_float(
                0.7979 * (value - stats.median) / stats.mean_abs_dev, operation="robust_z_score_meanad"
            )
        else:
            # All values identical — no spread at all. The robust z-score is
            # genuinely undefined; emit None (honest-absence doctrine), never 0.0.
            robust_z_score = None

        reasons: list[str] = []
        if abs(z_score) >= self._z_threshold:
            reasons.append("z_score")
        if robust_z_score is not None and abs(robust_z_score) >= self._robust_z_threshold:
            reasons.append("robust_z_score")

        return {
            self._field("value_field"): self._value_field,
            self._field("value"): entry.value,
            self._field("row_index"): entry.row_index,
            self._field("batch_size"): stats.batch_size,
            self._field("valid_count"): stats.valid_count,
            self._field("missing_count"): stats.missing_count,
            self._field("non_finite_count"): stats.non_finite_count,
            self._field("skipped_count"): stats.skipped_count,
            self._field("missing_indices"): stats.missing_indices,
            self._field("non_finite_indices"): stats.non_finite_indices,
            self._field("mean"): stats.mean,
            self._field("median"): stats.median,
            self._field("stdev"): stats.stdev,
            self._field("mad"): stats.mad,
            self._field("z_threshold"): self._z_threshold,
            self._field("robust_z_threshold"): self._robust_z_threshold,
            self._field("z_score"): z_score,
            self._field("robust_z_score"): robust_z_score,
            self._field("is_outlier"): bool(reasons),
            self._field("reason"): ",".join(reasons),
        }

    def _output_contract_for(self, entries: list[_FiniteEntry]) -> SchemaContract:
        """Build one shared output contract for annotated rows."""
        first_contract = entries[0].row.contract
        for entry in entries[1:]:
            if entry.row.contract.mode != first_contract.mode:
                raise ValueError(
                    f"Heterogeneous contract modes in batch: row 0 has mode "
                    f"'{first_contract.mode}', row {entry.row_index} has mode '{entry.row.contract.mode}'. "
                    f"All rows in a batch must share the same contract mode."
                )

        merged_fields: dict[str, FieldContract] = {}
        for entry in entries:
            for field in entry.row.contract.fields:
                if field.normalized_name not in merged_fields:
                    merged_fields[field.normalized_name] = field

        for field_name in sorted(self.declared_output_fields):
            merged_fields[field_name] = FieldContract(
                normalized_name=field_name,
                original_name=field_name,
                python_type=object,
                required=False,
                source="inferred",
            )

        output_contract = SchemaContract(
            mode=first_contract.mode,
            fields=tuple(merged_fields.values()),
            locked=True,
        )
        return self._align_output_contract(output_contract)

    def process(  # type: ignore[override] # Batch signature: list[PipelineRow] instead of PipelineRow
        self, rows: list[PipelineRow], ctx: TransformContext
    ) -> TransformResult:
        """Annotate finite numeric rows in a batch with outlier metrics."""
        if not rows:
            return TransformResult.error({"reason": "empty_batch"}, retryable=False)

        self._reject_runtime_output_field_collision(rows)
        entries, missing_indices, non_finite_indices = self._finite_entries_for(rows)
        if not entries:
            return self._error_for_no_finite_values(
                batch_size=len(rows),
                missing_indices=missing_indices,
                non_finite_indices=non_finite_indices,
            )

        stats, error = self._stats_for(
            batch_size=len(rows),
            entries=entries,
            missing_indices=missing_indices,
            non_finite_indices=non_finite_indices,
        )
        if error is not None:
            return error

        output_contract = self._output_contract_for(entries)
        fields_added = sorted(self.declared_output_fields)
        try:
            pipeline_rows = [
                PipelineRow(
                    {
                        **entry.row.to_dict(),
                        **self._annotation_for(entry, stats),
                    },
                    output_contract,
                )
                for entry in entries
            ]
        except OverflowError as exc:
            return self._float_overflow_error(
                operation=str(exc),
                batch_size=len(rows),
                valid_count=len(entries),
            )

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
