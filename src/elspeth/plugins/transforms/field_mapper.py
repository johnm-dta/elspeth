"""FieldMapper transform plugin.

Renames, selects, and reorganizes row fields.

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
from elspeth.plugins.sentinels import MISSING


class FieldMapperConfig(TransformDataConfig):
    """Configuration for field mapper transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {fields: dynamic}' for dynamic field handling.
    """

    mapping: dict[str, str] = Field(default_factory=dict)
    select_only: bool = False
    strict: bool = False
    validate_input: bool = False  # Optional input validation


class FieldMapper(BaseTransform):
    """Map, rename, and select row fields.

    Config options:
        schema: Required. Schema for input/output (use {fields: dynamic} for any fields)
        mapping: Dict of source_field -> target_field
            - Simple: {"old": "new"} renames old to new
            - Nested: {"meta.source": "origin"} extracts nested field
        select_only: If True, only include mapped fields (default: False)
        strict: If True, error on missing source fields (default: False)
        validate_input: If True, validate input against schema (default: False)
    """

    name = "field_mapper"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = FieldMapperConfig.from_dict(config)
        self._mapping: dict[str, str] = cfg.mapping
        self._select_only: bool = cfg.select_only
        self._strict: bool = cfg.strict
        self._validate_input: bool = cfg.validate_input
        self._on_error: str | None = cfg.on_error

        # TransformDataConfig validates schema_config is not None
        assert cfg.schema_config is not None
        self._schema_config = cfg.schema_config

        # Create schema from config
        # CRITICAL: allow_coercion=False - wrong types are source bugs
        schema = create_schema_from_config(
            self._schema_config,
            "FieldMapperSchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Apply field mapping to row.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with mapped row data

        Raises:
            ValidationError: If validate_input=True and row fails schema validation.
                This indicates a bug in the upstream source/transform.
        """
        # Optional input validation - crash on wrong types (source bug!)
        if self._validate_input and not self._schema_config.is_dynamic:
            self.input_schema.model_validate(row)  # Raises on failure

        # Start with empty or copy depending on select_only
        if self._select_only:
            output: dict[str, Any] = {}
        else:
            output = copy.deepcopy(row)

        # Apply mappings
        for source, target in self._mapping.items():
            value = self._get_nested(row, source)

            if value is MISSING:
                if self._strict:
                    return TransformResult.error(
                        {"message": f"Required field '{source}' not found in row"}
                    )
                continue  # Skip missing fields in non-strict mode

            # Remove old key if it exists (for rename within same dict)
            if not self._select_only and "." not in source and source in output:
                del output[source]

            output[target] = value

        return TransformResult.success(output)

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation.

        Args:
            data: Source dictionary
            path: Dot-separated path (e.g., "meta.source")

        Returns:
            Value at path or MISSING sentinel
        """
        parts = path.split(".")
        current: Any = data

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return MISSING
            current = current[part]

        return current

    def close(self) -> None:
        """No resources to release."""
        pass
