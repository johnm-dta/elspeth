"""Batch statistics transform plugin.

Computes aggregate statistics (sum, count, mean) over batches of rows.
Demonstrates batch-aware transform processing for aggregation pipelines.

IMPORTANT: This transform uses is_batch_aware=True, meaning the engine
will buffer rows and call process() with a list when the trigger fires.
"""

import math
from typing import Any

from pydantic import Field

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


class BatchStatsConfig(TransformDataConfig):
    """Configuration for batch statistics transform.

    Requires a numeric field to aggregate and optionally a group_by field.
    """

    value_field: str = Field(description="Name of the numeric field to aggregate (sum/mean)")
    group_by: str | None = Field(
        default=None,
        description="Optional field to include in output for grouping context",
    )
    compute_mean: bool = Field(
        default=True,
        description="Whether to compute mean in addition to sum/count",
    )


class BatchStats(BaseTransform):
    """Compute aggregate statistics over a batch of rows.

    This is a batch-aware transform that receives multiple rows at once
    when an aggregation trigger fires. It computes:
    - count: Number of rows in the batch
    - sum: Sum of the value_field across all rows
    - mean: Average of value_field (if compute_mean=True)

    Config options:
        schema: Required. Schema for input validation
        value_field: Required. Numeric field to aggregate
        group_by: Optional. Field to include in output for context
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
    plugin_version = "1.0.0"
    is_batch_aware = True  # CRITICAL: Engine buffers rows for batch processing

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchStatsConfig.from_dict(config)
        self._value_field = cfg.value_field
        self._group_by = cfg.group_by
        self._compute_mean = cfg.compute_mean

        self._schema_config = cfg.schema_config

        # Create input schema from config
        self.input_schema = create_schema_from_config(
            cfg.schema_config,
            "BatchStatsInputSchema",
            allow_coercion=False,
        )

        # Output schema MUST be dynamic because BatchStats outputs a completely
        # different shape: {count, sum, mean, batch_size, group_by?}
        # The output shape has no relation to the input schema.
        # Per P1-2026-01-19-shape-changing-transforms-output-schema-mismatch
        self.output_schema = create_schema_from_config(
            SchemaConfig.from_dict({"mode": "observed"}),
            "BatchStatsOutputSchema",
            allow_coercion=False,
        )

    def process(  # type: ignore[override] # Batch signature: list[PipelineRow] instead of PipelineRow
        self, rows: list[PipelineRow], ctx: PluginContext
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
            # Empty batch - should not happen in normal operation
            result_data: dict[str, Any] = {"count": 0, "sum": 0}
            fields_added = ["count", "sum"]
            if self._compute_mean:
                result_data["mean"] = None
                fields_added.append("mean")
            result_data["batch_empty"] = True
            fields_added.append("batch_empty")

            # Create OBSERVED contract for transform mode (processor.py:712 requires it)
            fields = tuple(
                FieldContract(
                    normalized_name=key,
                    original_name=key,
                    python_type=object,
                    required=False,
                    source="inferred",
                )
                for key in result_data
            )
            output_contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)

            return TransformResult.success(
                PipelineRow(result_data, output_contract),
                success_reason={
                    "action": "processed",
                    "fields_added": fields_added,
                    "metadata": {"empty_batch": True},
                },
            )
        # Extract numeric values - enforce type contract
        # Tier 2 pipeline data should already be validated; wrong types = upstream bug
        values: list[int | float] = []
        skipped_non_finite = 0
        for i, row in enumerate(rows):
            # Direct access - field must exist (KeyError = upstream bug)
            raw_value = row[self._value_field]

            # Contract enforcement: value_field must be numeric (int or float)
            # Tier 2 pipeline data - wrong types indicate upstream bug
            if not isinstance(raw_value, (int, float)):
                raise TypeError(
                    f"Field '{self._value_field}' must be numeric (int or float), "
                    f"got {type(raw_value).__name__} in row {i}. "
                    f"This indicates an upstream validation bug - check source schema or prior transforms."
                )

            # NaN/Inf are type-valid floats but operation-unsafe — they produce
            # garbage in arithmetic and crash downstream canonical JSON (RFC 8785).
            # Integers are always finite — only check floats.
            if isinstance(raw_value, float) and not math.isfinite(raw_value):
                skipped_non_finite += 1
                continue

            # Preserve original type: int stays int (arbitrary precision),
            # float stays float. Avoids precision loss for ints > 2^53.
            values.append(raw_value)

        count = len(values)
        total = sum(values) if values else 0

        # Guard against overflow: summing many large-but-valid floats can produce inf.
        # Integer sums use arbitrary precision and cannot overflow.
        if count > 0 and isinstance(total, float) and not math.isfinite(total):
            return TransformResult.error(
                {"reason": "float_overflow", "batch_size": len(rows), "valid_count": count},
                retryable=False,
            )

        result: dict[str, Any] = {
            "count": count,
            "sum": total,
            "batch_size": len(rows),  # Total rows, including those with missing values
        }

        if self._compute_mean and count > 0:
            result["mean"] = total / count
        elif self._compute_mean:
            result["mean"] = None

        if skipped_non_finite > 0:
            result["skipped_non_finite"] = skipped_non_finite

        # Include group_by field — validate homogeneity across batch.
        # group_by is configured contract, so missing field is an upstream bug.
        if self._group_by and rows:
            group_value = rows[0][self._group_by]
            for row in rows[1:]:
                val = row[self._group_by]
                if val != group_value:
                    raise ValueError(
                        f"Heterogeneous '{self._group_by}' values in batch: "
                        f"first row has {group_value!r}, found {val!r}. "
                        f"Configure the aggregation trigger to group by "
                        f"'{self._group_by}' or remove group_by config."
                    )
            result[self._group_by] = group_value

        # Determine which fields were added
        fields_added = ["count", "sum", "batch_size"]
        if self._compute_mean:
            fields_added.append("mean")
        if skipped_non_finite > 0:
            fields_added.append("skipped_non_finite")
        if self._group_by and rows:
            fields_added.append(self._group_by)

        # Create OBSERVED contract from output fields for transform mode
        # Aggregations in transform mode create new tokens (processor.py:712 requires contract)
        fields = tuple(
            FieldContract(
                normalized_name=key,
                original_name=key,
                python_type=object,  # OBSERVED mode - infer all as object type
                required=False,
                source="inferred",
            )
            for key in result
        )
        output_contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)

        return TransformResult.success(
            PipelineRow(result, output_contract),
            success_reason={"action": "processed", "fields_added": fields_added},
        )

    def close(self) -> None:
        """No resources to release."""
        pass
