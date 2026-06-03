"""Batch statistics transform plugin.

Computes aggregate statistics (sum, count, mean) over batches of rows.
Demonstrates batch-aware transform processing for aggregation pipelines.

IMPORTANT: This transform uses is_batch_aware=True, meaning the engine
will buffer rows and call process() with a list when the trigger fires.
"""

import math
from typing import Any

from pydantic import Field, field_validator, model_validator

from elspeth.contracts import Determinism
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.errors import TransformErrorReason
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.transforms._scalar_buckets import same_scalar_bucket_value

type BatchStatsAggregateRow = dict[str, object]


class BatchStatsConfig(TransformDataConfig):
    """Configuration for batch statistics transform.

    Requires a numeric field to aggregate. When group_by is configured, rows
    are partitioned by that field and one aggregate row is emitted per group.
    """

    value_field: str = Field(description="Name of the numeric field to aggregate (sum/mean)")
    group_by: str | None = Field(
        default=None,
        description=(
            "Optional field that partitions the batch and emits one aggregate row per distinct value. "
            "The group_by field is included in each output row."
        ),
    )
    compute_mean: bool = Field(
        default=True,
        description="Whether to compute mean in addition to sum/count",
    )

    @field_validator("value_field")
    @classmethod
    def _reject_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("value_field must not be empty")
        return v

    @field_validator("group_by")
    @classmethod
    def _reject_empty_group_by(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("group_by must not be empty")
        return v

    @model_validator(mode="after")
    def _reject_group_by_collision(self) -> "BatchStatsConfig":
        """Reject group_by names that collide with aggregate output keys.

        The set of output keys is deterministic from compute_mean — no runtime
        state needed. Moved from BatchStats.__init__ so that from_dict()
        catches it (pre-validation / engine-validation agreement).
        """
        if self.group_by is None:
            return self
        stat_fields = {"count", "sum", "batch_size", "skipped_non_finite", "skipped_non_finite_indices"}
        if self.compute_mean:
            stat_fields.add("mean")
        if self.group_by in stat_fields:
            raise ValueError(
                f"group_by field '{self.group_by}' collides with aggregate output key. "
                f"Choose a group_by field name that is not one of: {', '.join(sorted(stat_fields))}"
            )
        return self


class BatchStats(BaseTransform):
    """Compute aggregate statistics over a batch of rows.

    This is a batch-aware transform that receives multiple rows at once
    when an aggregation trigger fires. Without group_by, it emits one
    aggregate row for the full batch. With group_by, it emits one aggregate
    row per distinct group value. It computes:
    - count: Number of rows in the batch
    - sum: Sum of the value_field across all rows
    - mean: Average of value_field (if compute_mean=True)

    Config options:
        schema: Required. Schema for input validation
        value_field: Required. Numeric field to aggregate
        group_by: Optional. Field to partition by; included in each output row
        compute_mean: Whether to compute mean (default: True)

    Example YAML:
        aggregations:
          - name: daily_totals
            plugin: batch_stats
            trigger:
              count: 100
            options:
              schema:
                mode: observed
              value_field: amount
              group_by: category
              compute_mean: true
    """

    name = "batch_stats"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:7506d083ef17eb83"
    config_model = BatchStatsConfig
    is_batch_aware = True  # CRITICAL: Engine buffers rows for batch processing

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name=cls.name,
                issue_code=None,
                summary="Aggregates a numeric field into count, sum, and optional mean rows.",
                composer_hints=(
                    "Use batch_stats under aggregations with a trigger; it is not a per-row transform.",
                    "value_field must contain numeric int or float values; missing and non-finite values are skipped and reported.",
                    "Output is derived summary row(s) and does not preserve original row fields.",
                    "Set group_by only when you want one summary row per group; avoid names that collide with output fields.",
                ),
            )
        return None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal config for the ADR-009 backward invariant."""
        return {
            "schema": {"mode": "observed"},
            "value_field": "batch_stats_probe_value",
            "compute_mean": True,
        }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchStatsConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._value_field = cfg.value_field
        self._group_by = cfg.group_by
        self._compute_mean = cfg.compute_mean

        # Guaranteed output fields — always present in every successful result.
        # skipped_non_finite/skipped_non_finite_indices are conditional (only when
        # non-finite values exist) so they are NOT guaranteed. They are declared
        # separately for collision detection at init time.
        stat_fields: set[str] = {"count", "sum", "batch_size"}
        if cfg.compute_mean:
            stat_fields.add("mean")
        # _all_possible_output_keys is stat fields + conditional fields (without group_by)
        # used at runtime for field-set operations.
        self._all_possible_output_keys = frozenset(stat_fields | {"skipped_non_finite", "skipped_non_finite_indices"})

        # group_by collision is now caught by BatchStatsConfig._reject_group_by_collision
        # model_validator — from_dict() above already enforced it.

        # group_by is guaranteed on every successful result when configured.
        # Added after the collision check so it doesn't self-collide.
        if cfg.group_by is not None:
            stat_fields.add(cfg.group_by)
        self.declared_output_fields = frozenset(stat_fields)

        # Declare group_by as required input when configured, so the DAG builder
        # can validate upstream output guarantees. value_field is enforced at
        # runtime via direct field access (KeyError = upstream bug) rather than
        # build-time schema requirements — observed-mode schemas don't declare
        # guaranteed fields, so a build-time requirement would reject valid pipelines.
        if cfg.group_by is not None:
            base_required = set(cfg.schema_config.required_fields or ())
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
        else:
            schema_config = cfg.schema_config

        self._schema_config = schema_config

        self.input_schema, self.output_schema = self._create_schemas(
            schema_config,
            "BatchStats",
            adds_fields=True,
        )
        self._output_schema_config = self._build_output_schema_config(schema_config)

    def _build_output_schema_config(self, schema_config: SchemaConfig) -> SchemaConfig:
        """Override (elspeth-f5f798f797): aggregation output is independent of input shape.

        The user's ``schema:`` block describes the INPUT contract that batch_stats
        consumes — what fields/types upstream produces and which the consumer
        requires. The aggregation's OUTPUT is computed: stat fields plus an
        optional group_by, all OBSERVED-typed (``python_type=object``). It does
        not carry the user's input field declarations.

        The base class implementation copies ``fields``, ``required_fields``, and
        the user's input-side ``guaranteed_fields`` into the output config, which
        causes ``SchemaConfigModeViolation`` at runtime: ``get_effective_guaranteed_fields``
        unions the input fields into the expected output, and the OBSERVED-typed
        emission then misses fields like ``amount`` and mismatches metadata on
        ``customer_tier`` (declared ``str``, observed ``object``).

        Returns a config that honestly describes batch_stats output:
          - ``mode='observed'`` — output types are inferred from aggregate values.
          - ``fields=None`` — no explicit field declarations on output.
          - ``guaranteed_fields`` — the actual emitted set (declared_output_fields).
          - ``required_fields=None`` — input-only consumer requirement, irrelevant on output.
          - ``audit_fields=None`` — input-side audit hints don't apply to derived output.
        """
        guaranteed = tuple(sorted(self.declared_output_fields)) if self.declared_output_fields else None
        return SchemaConfig(
            mode="observed",
            fields=None,
            guaranteed_fields=guaranteed,
            required_fields=None,
            audit_fields=None,
        )

    def backward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Exercise the real aggregate output path for the backward invariant."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name=self._value_field,
                value=1.0,
            )
        ]

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

    def _aggregate_group(
        self,
        grouped_rows: list[tuple[int, PipelineRow]],
        group_value: Any,
    ) -> tuple[BatchStatsAggregateRow, TransformResult | None]:
        """Aggregate one already-partitioned group."""
        values: list[int | float] = []
        skipped_non_finite_indices: list[int] = []
        for row_index, row in grouped_rows:
            # Direct access - field must exist (KeyError = upstream bug)
            raw_value = row[self._value_field]

            # Contract enforcement: value_field must be numeric (int or float)
            # Tier 2 pipeline data - wrong types indicate upstream bug
            # Use type() instead of isinstance() to reject bool (bool is subclass of int)
            if type(raw_value) not in (int, float):
                raise TypeError(
                    f"Field '{self._value_field}' must be numeric (int or float), "
                    f"got {type(raw_value).__name__} in row {row_index}. "
                    f"This indicates an upstream validation bug - check source schema or prior transforms."
                )

            # NaN/Inf are type-valid floats but operation-unsafe — they produce
            # garbage in arithmetic and crash downstream canonical JSON (RFC 8785).
            # Integers are always finite — only check floats.
            if isinstance(raw_value, float) and not math.isfinite(raw_value):
                skipped_non_finite_indices.append(row_index)
                continue

            # Preserve original type: int stays int (arbitrary precision),
            # float stays float. Avoids precision loss for ints > 2^53.
            values.append(raw_value)

        count = len(values)

        # All-non-finite is the same condition as empty batch — no real data to aggregate.
        # Fabricating sum=0/count=0 would produce phantom statistics indistinguishable
        # from a legitimate computation over zero-valued data.
        if count == 0 and skipped_non_finite_indices:
            reason: TransformErrorReason = {
                "reason": "all_non_finite",
                "batch_size": len(grouped_rows),
                "skipped_non_finite": len(skipped_non_finite_indices),
                "skipped_non_finite_indices": skipped_non_finite_indices,
            }
            if self._group_by is not None:
                reason["group_by"] = self._group_by
                reason["group_value"] = group_value
            return {}, TransformResult.error(
                reason,
                retryable=False,
            )

        # At this point, values is guaranteed non-empty: empty batch returns error
        # before grouping, and all-non-finite returns error above. count > 0.
        total = sum(values)

        # Guard against overflow: summing many large-but-valid floats can produce inf.
        # Integer sums use arbitrary precision and cannot overflow.
        if isinstance(total, float) and not math.isfinite(total):
            overflow_reason: TransformErrorReason = {
                "reason": "float_overflow",
                "batch_size": len(grouped_rows),
                "valid_count": count,
            }
            if self._group_by is not None:
                overflow_reason["group_by"] = self._group_by
                overflow_reason["group_value"] = group_value
            return {}, TransformResult.error(overflow_reason, retryable=False)

        result: BatchStatsAggregateRow = {
            "count": count,
            "sum": total,
            "batch_size": len(grouped_rows),
        }

        if self._compute_mean:
            try:
                result["mean"] = total / count
            except OverflowError:
                mean_error_reason: TransformErrorReason = {
                    "reason": "float_overflow",
                    "operation": "mean",
                    "batch_size": len(grouped_rows),
                    "valid_count": count,
                }
                if self._group_by is not None:
                    mean_error_reason["group_by"] = self._group_by
                    mean_error_reason["group_value"] = group_value
                return {}, TransformResult.error(mean_error_reason, retryable=False)

        if skipped_non_finite_indices:
            result["skipped_non_finite"] = len(skipped_non_finite_indices)
            result["skipped_non_finite_indices"] = skipped_non_finite_indices

        if self._group_by is not None:
            result[self._group_by] = group_value

        return result, None

    def _output_contract_for(self, results: list[BatchStatsAggregateRow]) -> SchemaContract:
        """Build one shared output contract for aggregate result rows."""
        field_names = list(dict.fromkeys(key for result in results for key in result))
        fields = tuple(
            FieldContract(
                normalized_name=key,
                original_name=key,
                python_type=object,  # OBSERVED mode - infer all as object type
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
        """Compute statistics over a batch of rows.

        Args:
            rows: List of input rows (batch-aware receives list[PipelineRow])
            ctx: Plugin context

        Returns:
            TransformResult with aggregated statistics

        Raises:
            KeyError: If value_field is missing from any row (upstream bug)
            TypeError: If value_field is not numeric (upstream bug)
        """
        if not rows:
            # Empty batch is an anomaly — return error, not fabricated statistics.
            # Phantom sum=0/count=0/mean=None would flow to sinks as real data.
            return TransformResult.error(
                {"reason": "empty_batch"},
                retryable=False,
            )

        results: list[BatchStatsAggregateRow] = []
        for group_value, grouped_rows in self._group_rows(rows):
            aggregate, error = self._aggregate_group(grouped_rows, group_value)
            if error is not None:
                return error
            results.append(aggregate)

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
