"""Batch statistics transform plugin.

Computes aggregate statistics (sum, count, mean) over batches of rows.
Demonstrates batch-aware transform processing for aggregation pipelines.

IMPORTANT: This transform uses is_batch_aware=True, meaning the engine
will buffer rows and call process() with a list when the trigger fires.
"""

from typing import Any

from pydantic import Field

from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
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
                fields: dynamic
              value_field: amount
              group_by: category
              compute_mean: true
    """

    name = "batch_stats"
    is_batch_aware = True  # CRITICAL: Engine buffers rows for batch processing

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchStatsConfig.from_dict(config)
        self._value_field = cfg.value_field
        self._group_by = cfg.group_by
        self._compute_mean = cfg.compute_mean
        self._on_error = cfg.on_error

        # TransformDataConfig validates schema_config is not None
        assert cfg.schema_config is not None
        self._schema_config = cfg.schema_config

        # Create schema from config
        schema = create_schema_from_config(
            self._schema_config,
            "BatchStatsSchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

    @staticmethod
    def _try_convert_to_float(value: Any) -> float | None:
        """Try to convert a value to float, returning None if not possible."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def process(  # type: ignore[override]
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> TransformResult:
        """Compute statistics over a batch of rows.

        Args:
            rows: List of input rows (batch-aware receives list, not single row)
            ctx: Plugin context

        Returns:
            TransformResult with aggregated statistics
        """
        if not rows:
            # Empty batch - should not happen in normal operation
            return TransformResult.success({"count": 0, "sum": 0, "mean": None, "batch_empty": True})

        # Extract numeric values, skipping non-convertible ones
        # (Their data can have non-numeric values - this is expected, not a bug)
        values: list[float] = []
        for row in rows:
            if self._value_field in row:
                raw_value = row[self._value_field]
                converted = self._try_convert_to_float(raw_value)
                if converted is not None:
                    values.append(converted)

        count = len(values)
        total = sum(values) if values else 0

        result: dict[str, Any] = {
            "count": count,
            "sum": total,
            "batch_size": len(rows),  # Total rows, including those with missing values
        }

        if self._compute_mean and count > 0:
            result["mean"] = total / count
        elif self._compute_mean:
            result["mean"] = None

        # Include group_by field from first row for context
        if self._group_by and rows and self._group_by in rows[0]:
            result[self._group_by] = rows[0][self._group_by]

        return TransformResult.success(result)

    def close(self) -> None:
        """No resources to release."""
        pass
