"""Batch distribution profile transform plugin.

Computes descriptive statistics over buffered aggregation batches. This plugin
is intentionally shape-changing: it emits profile rows rather than passing
through source rows.
"""

from __future__ import annotations

import math
import statistics
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import Field, field_validator, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.errors import RowErrorEntry, TransformErrorReason
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.transforms._scalar_buckets import same_scalar_bucket_value

if TYPE_CHECKING:
    from elspeth.contracts.plugin_assistance import PluginAssistance

type BatchDistributionProfileRow = dict[str, object]

_GUARANTEED_PROFILE_FIELDS = frozenset(
    {
        "batch_size",
        "count",
        "field",
        "max",
        "mean",
        "median",
        "min",
        "missing_count",
        "non_finite_count",
        "p25",
        "p75",
        "stdev",
        "summary",
    }
)
_CONDITIONAL_PROFILE_FIELDS = frozenset({"missing_indices", "non_finite_indices"})
_PROFILE_OUTPUT_KEYS = _GUARANTEED_PROFILE_FIELDS | _CONDITIONAL_PROFILE_FIELDS


class BatchDistributionProfileConfig(TransformDataConfig):
    """Configuration for batch distribution profile transform."""

    value_field: str = Field(description="Name of the numeric field to profile")
    group_by: str | None = Field(
        default=None,
        description=(
            "Optional field that partitions the batch and emits one profile row per distinct value. "
            "The group_by field is included in each output row."
        ),
    )

    @field_validator("value_field")
    @classmethod
    def _reject_empty_value_field(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("value_field must not be empty")
        return v

    @field_validator("group_by")
    @classmethod
    def _reject_empty_group_by(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("group_by must not be empty")
        return v

    @model_validator(mode="after")
    def _reject_group_by_collision(self) -> BatchDistributionProfileConfig:
        if self.group_by is None:
            return self
        if self.group_by in _PROFILE_OUTPUT_KEYS:
            raise ValueError(
                f"group_by field '{self.group_by}' collides with profile output key. "
                f"Choose a group_by field name that is not one of: {', '.join(sorted(_PROFILE_OUTPUT_KEYS))}"
            )
        return self


class BatchDistributionProfile(BaseTransform):
    """Compute distribution summaries over aggregation batches.

    The transform emits one profile row per batch, or one profile row per
    group when ``group_by`` is configured. ``None`` values are treated as
    missing data. NaN and infinity are type-valid but operation-unsafe, so they
    are excluded from statistics and reported separately.

    Phase 6A B6 — opts into Phase 6B's narrative-mode result rendering via
    ``capability_tags = ("narrative-summary",)``. The frontend reads this tag
    on the catalog response and, when set, renders the run result as a
    narrative panel (consuming any Phase 5b interpretation events as an
    overlay) rather than the default tabular preview. The wire contract for
    the tag is: the transform's output schema must include a ``summary`` field
    that the narrative renderer surfaces.
    """

    name = "batch_distribution_profile"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:3ee882a8c9a35a99"
    config_model = BatchDistributionProfileConfig
    is_batch_aware = True
    capability_tags: tuple[str, ...] = ("narrative-summary",)

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "value_field": "batch_distribution_profile_probe_value",
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchDistributionProfileConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._value_field = cfg.value_field
        self._group_by = cfg.group_by

        declared_output_fields = set(_GUARANTEED_PROFILE_FIELDS)
        if cfg.group_by is not None:
            declared_output_fields.add(cfg.group_by)
        self.declared_output_fields = frozenset(declared_output_fields)

        base_required = set(cfg.schema_config.required_fields or ())
        base_required.add(cfg.value_field)
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
            "BatchDistributionProfile",
            adds_fields=True,
        )
        self._output_schema_config = self._build_output_schema_config(schema_config)

    def _build_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
        """Describe the profile output shape without propagating input fields."""
        guaranteed = tuple(sorted(self.declared_output_fields)) if self.declared_output_fields else None
        return SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=guaranteed,
            required_fields=None,
            audit_fields=None,
        )

    @classmethod
    def get_agent_assistance(
        cls,
        *,
        issue_code: str | None = None,
    ) -> PluginAssistance | None:
        from elspeth.contracts.plugin_assistance import PluginAssistance

        if issue_code is None:
            return PluginAssistance(
                plugin_name="batch_distribution_profile",
                issue_code=None,
                summary="Aggregate numeric descriptive statistics — mean, stddev, quartiles, optionally per group. Numeric-only; categorical counts go to batch_top_k.",
                composer_hints=(
                    "Use batch_distribution_profile under aggregations with a trigger; it summarizes a flushed batch.",
                    "value_field must be int or float. Strings (theme/category names) belong in batch_top_k, not here.",
                    "group_by partitions by a categorical field; omit it for a single distribution over all rows.",
                    "Output is descriptive-statistic summary rows and does not preserve the original row shape.",
                    "Words like 'distribution', 'theme frequency', 'category counts' usually mean batch_top_k unless the user clearly wants numeric stats.",
                ),
            )
        if issue_code != "batch_distribution_profile.value_field.numeric":
            return None
        return PluginAssistance(
            plugin_name="batch_distribution_profile",
            issue_code="batch_distribution_profile.value_field.numeric",
            summary=(
                "batch_distribution_profile computes numeric descriptive statistics only. "
                "Its value_field must be int or float; categorical counts and theme frequencies belong in batch_top_k."
            ),
            suggested_fixes=(
                "Use batch_top_k with field set to the categorical column for barrier counts, theme frequencies, or distributions over strings.",
                "Keep batch_distribution_profile only for numeric measures such as amounts, scores, durations, or counts.",
                "When grouping numeric profiles, set group_by to the categorical partition field and value_field to the numeric measure.",
            ),
            composer_hints=(
                "Words like distribution, barrier counts, theme frequency, category counts, or categorical summary should map to batch_top_k unless the requested statistic is numeric.",
            ),
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the aggregate output path for the backward invariant."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._value_field,
                value=1.0,
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

    def _group_rows(self, rows: list[PipelineRow]) -> list[tuple[Any, list[tuple[int, PipelineRow]]]]:
        """Partition rows by group_by value while preserving first-seen order."""
        if self._group_by is None:
            return [(None, list(enumerate(rows)))]

        groups: list[tuple[Any, list[tuple[int, PipelineRow]]]] = []
        for row_index, row in enumerate(rows):
            group_value = row[self._group_by]
            for existing_value, grouped_rows in groups:
                if same_scalar_bucket_value(group_value, existing_value):
                    grouped_rows.append((row_index, row))
                    break
            else:
                groups.append((group_value, [(row_index, row)]))
        return groups

    def _finite_values_for(
        self,
        grouped_rows: list[tuple[int, PipelineRow]],
    ) -> tuple[list[int | float], list[int], list[int]]:
        """Collect finite numeric values and data-quality indices for one group."""
        values: list[int | float] = []
        missing_indices: list[int] = []
        non_finite_indices: list[int] = []

        for row_index, row in grouped_rows:
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

            values.append(raw_value)

        return values, missing_indices, non_finite_indices

    def _error_for_no_finite_values(
        self,
        grouped_rows: list[tuple[int, PipelineRow]],
        group_value: Any,
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
            "batch_size": len(grouped_rows),
            "valid_count": 0,
            "skipped_count": len(missing_indices) + len(non_finite_indices),
            "row_errors": row_errors,
        }
        if self._group_by is not None:
            reason["group_by"] = self._group_by
            reason["group_value"] = group_value
        return TransformResult.error(reason, retryable=False)

    @staticmethod
    def _percentile(sorted_values: list[int | float], quantile: float) -> float:
        """Return a linearly interpolated percentile for already-sorted values."""
        if len(sorted_values) == 1:
            return float(sorted_values[0])

        position = (len(sorted_values) - 1) * quantile
        lower_index = math.floor(position)
        upper_index = math.ceil(position)
        if lower_index == upper_index:
            return float(sorted_values[lower_index])

        lower_value = sorted_values[lower_index]
        upper_value = sorted_values[upper_index]
        fraction = position - lower_index
        return float(lower_value + (upper_value - lower_value) * fraction)

    @staticmethod
    def _require_finite_float(value: float, *, operation: str) -> float:
        """Return value if finite, otherwise classify the aggregate as invalid."""
        if not math.isfinite(value):
            raise OverflowError(operation)
        return value

    @staticmethod
    def _format_number(value: int | float) -> str:
        return f"{float(value):.3f}"

    def _summary_prefix(self, group_value: Any) -> str:
        if self._group_by is None:
            return f"Distribution profile for {self._value_field}"
        return f"Distribution profile for {self._value_field} grouped by {self._group_by}={group_value}"

    def _aggregate_group(
        self,
        grouped_rows: list[tuple[int, PipelineRow]],
        group_value: Any,
    ) -> tuple[BatchDistributionProfileRow, TransformResult | None]:
        values, missing_indices, non_finite_indices = self._finite_values_for(grouped_rows)
        if not values:
            return {}, self._error_for_no_finite_values(grouped_rows, group_value, missing_indices, non_finite_indices)

        sorted_values = sorted(values)
        count = len(values)

        try:
            total = sum(values)
            mean = self._require_finite_float(total / count, operation="mean")
            median = self._require_finite_float(self._percentile(sorted_values, 0.5), operation="median")
            p25 = self._require_finite_float(self._percentile(sorted_values, 0.25), operation="p25")
            p75 = self._require_finite_float(self._percentile(sorted_values, 0.75), operation="p75")
            # stdev is undefined at n=1 -- emit None, never 0.0 (B4.5-a)
            stdev: float | None = None if count == 1 else self._require_finite_float(statistics.stdev(values), operation="stdev")
        except OverflowError as exc:
            reason: TransformErrorReason = {
                "reason": "float_overflow",
                "operation": str(exc) or "distribution_profile",
                "batch_size": len(grouped_rows),
                "valid_count": count,
            }
            if self._group_by is not None:
                reason["group_by"] = self._group_by
                reason["group_value"] = group_value
            return {}, TransformResult.error(reason, retryable=False)

        result: BatchDistributionProfileRow = {
            "field": self._value_field,
            "count": count,
            "batch_size": len(grouped_rows),
            "missing_count": len(missing_indices),
            "non_finite_count": len(non_finite_indices),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": mean,
            "median": median,
            "p25": p25,
            "p75": p75,
            "stdev": stdev,
            "summary": (
                f"{self._summary_prefix(group_value)}: {len(grouped_rows)} rows, {count} finite values, "
                f"{len(missing_indices)} missing, {len(non_finite_indices)} non-finite, "
                f"mean {self._format_number(mean)}, median {self._format_number(median)}, "
                f"range {self._format_number(sorted_values[0])} to {self._format_number(sorted_values[-1])}."
            ),
        }

        if missing_indices:
            result["missing_indices"] = missing_indices
        if non_finite_indices:
            result["non_finite_indices"] = non_finite_indices
        if self._group_by is not None:
            result[self._group_by] = group_value

        return result, None

    def _output_contract_for(self, results: list[BatchDistributionProfileRow]) -> SchemaContract:
        """Build one shared output contract for profile result rows."""
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
        """Compute distribution profile rows over a batch."""
        if not rows:
            return TransformResult.error({"reason": "empty_batch"}, retryable=False)

        non_finite_error = self._non_finite_group_key_error(rows)
        if non_finite_error is not None:
            return non_finite_error

        results: list[BatchDistributionProfileRow] = []
        for group_value, grouped_rows in self._group_rows(rows):
            profile, error = self._aggregate_group(grouped_rows, group_value)
            if error is not None:
                return error
            results.append(profile)

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
