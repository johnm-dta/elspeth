"""Truncate transform plugin.

Truncates specified string fields to maximum lengths.

IMPORTANT: Transforms use allow_coercion=False to catch upstream bugs.
If the source outputs wrong types, the transform crashes immediately.
"""

import copy
from typing import Any

from pydantic import Field

from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


class TruncateConfig(TransformDataConfig):
    """Configuration for truncate transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {fields: dynamic}' for dynamic field handling.
    """

    fields: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of field names to maximum lengths",
    )
    suffix: str = Field(
        default="",
        description="Suffix to append when truncating (e.g., '...'). Counts toward max length.",
    )
    strict: bool = Field(
        default=False,
        description="If True, error when a specified field is missing (default: False)",
    )


class Truncate(BaseTransform):
    """Truncate string fields to specified maximum lengths.

    Use cases:
    - Enforce maximum field lengths before database insertion
    - Trim verbose text fields for display
    - Ensure data fits within schema constraints

    Config options:
        schema: Required. Schema for input/output (use {fields: dynamic} for any fields)
        fields: Dict of field_name -> max_length (e.g., {"title": 100, "description": 500})
        suffix: String to append when truncating (e.g., "..."). Included in max length.
        strict: If True, error on missing fields (default: False, skip missing fields)
        on_error: Sink to route errors to (default: None, will quarantine)

    Example config:
        - plugin: truncate
          options:
            schema:
              fields: dynamic
            fields:
              title: 100
              description: 500
              notes: 1000
            suffix: "..."
    """

    name = "truncate"
    plugin_version = "1.0.0"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = TruncateConfig.from_dict(config)
        self._fields = cfg.fields
        self._suffix = cfg.suffix
        self._strict = cfg.strict
        self._on_error: str | None = cfg.on_error

        # Validate suffix length doesn't exceed any max length
        suffix_len = len(self._suffix)
        for field_name, max_len in self._fields.items():
            if suffix_len >= max_len:
                raise ValueError(f"Suffix length ({suffix_len}) must be less than max length for field '{field_name}' ({max_len})")

        self._schema_config = cfg.schema_config

        # Create schema from config
        # CRITICAL: allow_coercion=False - wrong types are source bugs
        schema = create_schema_from_config(
            cfg.schema_config,
            "TruncateSchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Truncate specified fields in row.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with truncated field values
        """
        output = copy.deepcopy(row)

        for field_name, max_len in self._fields.items():
            if field_name not in output:
                if self._strict:
                    return TransformResult.error(
                        {
                            "reason": "missing_field",
                            "field": field_name,
                        }
                    )
                continue  # Skip missing fields in non-strict mode

            value = output[field_name]

            # Only truncate strings
            if not isinstance(value, str):
                # Non-string values pass through unchanged
                # (This is their data - if it's wrong type, source should have caught it)
                continue

            # Truncate if needed
            if len(value) > max_len:
                if self._suffix:
                    # Truncate leaving room for suffix
                    truncate_at = max_len - len(self._suffix)
                    output[field_name] = value[:truncate_at] + self._suffix
                else:
                    output[field_name] = value[:max_len]

        # Track which fields were actually truncated
        fields_modified = [
            field_name
            for field_name, max_len in self._fields.items()
            if field_name in row and isinstance(row[field_name], str) and len(row[field_name]) > max_len
        ]

        return TransformResult.success(
            output,
            success_reason={
                "action": "transformed",
                "fields_modified": fields_modified,
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass
