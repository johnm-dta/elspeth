"""JSONExplode deaggregation transform.

Transforms one row containing an array field into multiple rows, one for each
element in the array. This is the inverse of aggregation (1-to-N expansion).

THREE-TIER TRUST MODEL COMPLIANCE:

Per the plugin protocol, transforms TRUST that pipeline data types are correct:
- Source validates that required fields exist and have correct types
- Transforms access fields directly without defensive checks
- Type violations (missing field, wrong type) indicate UPSTREAM BUGS and should CRASH

JSONExplode does NOT return TransformResult.error() for type violations because:
1. Missing field = source should have validated -> crash surfaces config bug
2. Wrong type = source should have validated -> crash surfaces config bug
3. There are no VALUE-level operations that can fail in this transform

Therefore, JSONExplode inherits from DataPluginConfig (NOT TransformDataConfig)
and has no on_error configuration.
"""

from typing import Any

from pydantic import Field

from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import DataPluginConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


class JSONExplodeConfig(DataPluginConfig):
    """Configuration for JSON explode transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {fields: dynamic}' for dynamic field handling.

    Attributes:
        array_field: Name of the array field to explode (required)
        output_field: Name for the exploded element (default: "item")
        include_index: Whether to include item_index field (default: True)
    """

    array_field: str = Field(..., description="Name of the array field to explode")
    output_field: str = Field(default="item", description="Name for the exploded element")
    include_index: bool = Field(default=True, description="Whether to include item_index field")


class JSONExplode(BaseTransform):
    """Explode a JSON array field into multiple rows.

    This is a deaggregation transform that expands one input row into multiple
    output rows, one for each element in the array field. The creates_tokens=True
    flag signals to the engine that new token IDs should be created for each
    output row with parent linkage to the input token.

    Config options:
        schema: Required. Schema for input/output (use {fields: dynamic} for any fields)
        array_field: Required. Name of the array field to explode
        output_field: Name for the exploded element (default: "item")
        include_index: Whether to include item_index field (default: True)

    Example:
        Input:  {"id": 1, "items": [{"name": "a"}, {"name": "b"}]}
        Output: [
            {"id": 1, "item": {"name": "a"}, "item_index": 0},
            {"id": 1, "item": {"name": "b"}, "item_index": 1},
        ]

    TRUST MODEL:
        This transform trusts that the source validated:
        - array_field exists in the row
        - array_field value is a list/array

        If these invariants are violated, the transform CRASHES (KeyError, TypeError)
        to surface the upstream bug. This is intentional - see module docstring.
    """

    name = "json_explode"
    plugin_version = "1.0.0"
    creates_tokens = True  # CRITICAL: enables new token creation for deaggregation

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the JSONExplode transform.

        Args:
            config: Configuration dict containing array_field and optional settings

        Raises:
            PluginConfigError: If required config is missing or invalid
        """
        super().__init__(config)
        cfg = JSONExplodeConfig.from_dict(config)
        self._array_field = cfg.array_field
        self._output_field = cfg.output_field
        self._include_index = cfg.include_index

        # Input schema from config for validation
        self.input_schema = create_schema_from_config(cfg.schema_config, "JSONExplodeInputSchema", allow_coercion=False)

        # Output schema MUST be dynamic because JSONExplode changes row shape:
        # - Removes array_field
        # - Adds output_field (e.g., "item")
        # - Adds item_index (if include_index=True)
        # The output shape depends on config, not input schema.
        # Per P1-2026-01-19-shape-changing-transforms-output-schema-mismatch
        self.output_schema = create_schema_from_config(
            SchemaConfig.from_dict({"fields": "dynamic"}),
            "JSONExplodeOutputSchema",
            allow_coercion=False,
        )

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Explode array field into multiple rows.

        Args:
            row: Input row containing the array field
            ctx: Plugin context

        Returns:
            TransformResult with multiple output rows (success_multi) or
            single row for empty arrays (success)

        Raises:
            KeyError: If array_field is missing (upstream bug)
            TypeError: If array_field is not a list (upstream bug)
        """
        # Direct access - TRUST that source validated field exists
        # KeyError here = upstream bug (source didn't validate field exists)
        array_value = row[self._array_field]

        # Contract enforcement: array_field must be list
        # Strings/dicts are iterable but would produce garbage - fail explicitly
        if not isinstance(array_value, list):
            raise TypeError(
                f"Field '{self._array_field}' must be a list, got {type(array_value).__name__}. "
                f"This indicates an upstream validation bug - check source schema or prior transforms."
            )

        # Build base output (all fields except the array field)
        base = {k: v for k, v in row.items() if k != self._array_field}

        # Handle empty array - return single row, not multi
        if len(array_value) == 0:
            output = dict(base)
            output[self._output_field] = None
            if self._include_index:
                output["item_index"] = None
            fields_added = [self._output_field]
            if self._include_index:
                fields_added.append("item_index")
            return TransformResult.success(
                output,
                success_reason={
                    "action": "transformed",
                    "fields_added": fields_added,
                    "fields_removed": [self._array_field],
                },
            )

        # Explode array into multiple rows
        output_rows = []
        for i, item in enumerate(array_value):
            output = dict(base)
            output[self._output_field] = item
            if self._include_index:
                output["item_index"] = i
            output_rows.append(output)

        fields_added = [self._output_field]
        if self._include_index:
            fields_added.append("item_index")
        return TransformResult.success_multi(
            output_rows,
            success_reason={
                "action": "transformed",
                "fields_added": fields_added,
                "fields_removed": [self._array_field],
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass
