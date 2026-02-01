"""Batch replicate transform plugin.

Demonstrates output_mode: transform deaggregation by replicating rows.
Each input row is replicated based on a 'copies' field, producing more
output rows than input rows (N inputs -> M outputs where M >= N).

IMPORTANT: This transform uses is_batch_aware=True, meaning the engine
will buffer rows and call process() with a list when the trigger fires.

For output_mode: transform, the engine creates NEW tokens for each output
row, with parent linkage to track deaggregation lineage.
"""

from typing import Any

from pydantic import Field

from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


class BatchReplicateConfig(TransformDataConfig):
    """Configuration for batch replicate transform.

    Requires a field that specifies how many copies of each row to produce.
    """

    copies_field: str = Field(
        default="copies",
        description="Name of the field containing the number of copies to make",
    )
    default_copies: int = Field(
        default=1,
        ge=1,
        description="Default number of copies if copies_field is missing or invalid",
    )
    include_copy_index: bool = Field(
        default=True,
        description="Whether to add a 'copy_index' field (0-based) to each output row",
    )


class BatchReplicate(BaseTransform):
    """Replicate rows based on a copies field.

    This is a batch-aware transform that demonstrates output_mode: transform
    deaggregation. It receives N input rows and produces M output rows where
    M is the sum of all copies values.

    Example: If input has 3 rows with copies=2,3,1 respectively:
    - Input: 3 rows
    - Output: 6 rows (2 + 3 + 1)
    - Each output row has copy_index showing which copy it is

    Config options:
        schema: Required. Schema for input validation
        copies_field: Field name for copy count (default: "copies")
        default_copies: Fallback if field missing (default: 1)
        include_copy_index: Add copy_index field (default: True)

    Example YAML:
        aggregations:
          - name: replicate_batch
            plugin: batch_replicate
            trigger:
              count: 5
            output_mode: transform  # CRITICAL: Creates new tokens for outputs
            options:
              schema:
                fields: dynamic
              copies_field: quantity
              default_copies: 1
    """

    name = "batch_replicate"
    plugin_version = "1.0.0"
    is_batch_aware = True  # CRITICAL: Engine buffers rows for batch processing

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = BatchReplicateConfig.from_dict(config)
        self._copies_field = cfg.copies_field
        self._default_copies = cfg.default_copies
        self._include_copy_index = cfg.include_copy_index
        self._on_error = cfg.on_error

        self._schema_config = cfg.schema_config

        # Input schema from config
        self.input_schema = create_schema_from_config(
            cfg.schema_config,
            "BatchReplicateInputSchema",
            allow_coercion=False,
        )

        # Output schema MUST be dynamic because BatchReplicate adds copy_index field
        # Per P1-2026-01-19-shape-changing-transforms-output-schema-mismatch
        self.output_schema = create_schema_from_config(
            SchemaConfig.from_dict({"fields": "dynamic"}),
            "BatchReplicateOutputSchema",
            allow_coercion=False,
        )

    def process(  # type: ignore[override]
        self, rows: list[dict[str, Any]], ctx: PluginContext
    ) -> TransformResult:
        """Replicate each row based on its copies field.

        Args:
            rows: List of input rows (batch-aware receives list)
            ctx: Plugin context

        Returns:
            TransformResult.success_multi() with replicated rows
        """
        if not rows:
            # Empty batch - should not happen in normal operation
            # Return success with single empty-marker row
            return TransformResult.success(
                {"batch_empty": True},
                success_reason={"action": "processed", "metadata": {"empty_batch": True}},
            )

        output_rows: list[dict[str, Any]] = []

        for row in rows:
            # Get copies count - field is optional, type must be correct if present
            if self._copies_field not in row:
                # Field missing - use default (valid scenario)
                copies = self._default_copies
            else:
                raw_copies = row[self._copies_field]

                # Contract enforcement: copies_field must be int if present
                # Tier 2 pipeline data - wrong types indicate upstream bug
                if not isinstance(raw_copies, int):
                    raise TypeError(
                        f"Field '{self._copies_field}' must be int, got {type(raw_copies).__name__}. "
                        f"This indicates an upstream validation bug - check source schema or prior transforms."
                    )

                # Validate value is positive
                if raw_copies < 1:
                    raise ValueError(
                        f"Field '{self._copies_field}' must be >= 1, got {raw_copies}. "
                        f"This indicates invalid data - check source validation."
                    )

                copies = raw_copies

            # Create copies of this row
            for copy_idx in range(copies):
                output = dict(row)  # Shallow copy preserves original data
                if self._include_copy_index:
                    output["copy_index"] = copy_idx
                output_rows.append(output)

        # Return multiple rows - engine will create new tokens for each
        return TransformResult.success_multi(
            output_rows,
            success_reason={
                "action": "processed",
                "fields_added": ["copy_index"] if self._include_copy_index else [],
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass
