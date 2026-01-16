"""FieldMapper transform plugin.

Renames, selects, and reorganizes row fields.
"""

import copy
from typing import Any

from pydantic import Field

from elspeth.contracts import PluginSchema
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import PluginConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.sentinels import MISSING


class FieldMapperSchema(PluginSchema):
    """Dynamic schema - fields determined by mapping."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic class-level config


class FieldMapperConfig(PluginConfig):
    """Configuration for field mapper transform."""

    mapping: dict[str, str] = Field(default_factory=dict)
    select_only: bool = False
    strict: bool = False


class FieldMapper(BaseTransform):
    """Map, rename, and select row fields.

    Config options:
        mapping: Dict of source_field -> target_field
            - Simple: {"old": "new"} renames old to new
            - Nested: {"meta.source": "origin"} extracts nested field
        select_only: If True, only include mapped fields (default: False)
        strict: If True, error on missing source fields (default: False)
    """

    name = "field_mapper"
    input_schema = FieldMapperSchema
    output_schema = FieldMapperSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = FieldMapperConfig.from_dict(config)
        self._mapping: dict[str, str] = cfg.mapping
        self._select_only: bool = cfg.select_only
        self._strict: bool = cfg.strict

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Apply field mapping to row.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with mapped row data
        """
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
