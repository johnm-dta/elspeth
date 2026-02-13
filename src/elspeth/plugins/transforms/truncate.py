"""Truncate transform plugin.

Truncates specified string fields to maximum lengths.

IMPORTANT: Transforms use allow_coercion=False to catch upstream bugs.
If the source outputs wrong types, the transform crashes immediately.
"""

import copy
from typing import Any

from pydantic import Field

from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


class TruncateConfig(TransformDataConfig):
    """Configuration for truncate transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {mode: observed}' for dynamic field handling.
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
        schema: Required. Schema for input/output (use {mode: observed} for any fields)
        fields: Dict of field_name -> max_length (e.g., {"title": 100, "description": 500})
        suffix: String to append when truncating (e.g., "..."). Included in max length.
        strict: If True, error on missing configured fields (default: False, skip missing fields)
        on_error: Sink to route errors to (default: None, will quarantine)

    Example config:
        - plugin: truncate
          options:
            schema:
              mode: observed
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

    def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
        """Truncate specified fields in row.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with truncated field values
        """
        output = copy.deepcopy(row.to_dict())
        fields_modified: list[str] = []

        for field_name, max_len in self._fields.items():
            if field_name not in row:
                if self._strict:
                    return TransformResult.error(
                        {
                            "reason": "missing_field",
                            "field": field_name,
                        }
                    )
                continue  # Skip missing fields in non-strict mode

            value = row[field_name]
            if field_name in output:
                normalized_field_name = field_name
            else:
                normalized_field_name = row.contract.resolve_name(field_name)

            # Type mismatches are upstream contract bugs; always surface explicitly.
            if type(value) is not str:
                return TransformResult.error(
                    {
                        "reason": "type_mismatch",
                        "field": field_name,
                        "expected": "str",
                        "actual": type(value).__name__,
                    }
                )

            # Truncate if needed
            if len(value) > max_len:
                if self._suffix:
                    # Truncate leaving room for suffix
                    truncate_at = max_len - len(self._suffix)
                    output[normalized_field_name] = value[:truncate_at] + self._suffix
                else:
                    output[normalized_field_name] = value[:max_len]
                fields_modified.append(field_name)

        return TransformResult.success(
            PipelineRow(output, row.contract),
            success_reason={
                "action": "transformed",
                "fields_modified": fields_modified,
            },
        )

    def close(self) -> None:
        """No resources to release."""
        pass
