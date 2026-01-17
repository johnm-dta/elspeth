"""PassThrough transform plugin.

Passes rows through unchanged. Useful for testing and debugging pipelines.

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


class PassThroughConfig(TransformDataConfig):
    """Configuration for passthrough transform.

    Requires 'schema' in config to define input/output expectations.
    Use 'schema: {fields: dynamic}' for dynamic field handling.
    """

    validate_input: bool = Field(
        default=False,
        description="If True, validate input against schema (default: False)",
    )


class PassThrough(BaseTransform):
    """Pass rows through unchanged.

    Use cases:
    - Testing pipeline wiring without modification
    - Debugging data flow (add logging in subclass)
    - Placeholder for future transform logic

    Config options:
        schema: Required. Schema for input/output (use {fields: dynamic} for any fields)
        validate_input: If True, validate input against schema (default: False)
    """

    name = "passthrough"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = PassThroughConfig.from_dict(config)
        self._validate_input = cfg.validate_input
        self._on_error: str | None = cfg.on_error

        # TransformDataConfig validates schema_config is not None
        assert cfg.schema_config is not None
        self._schema_config = cfg.schema_config

        # Create schema from config
        # CRITICAL: allow_coercion=False - wrong types are source bugs
        schema = create_schema_from_config(
            self._schema_config,
            "PassThroughSchema",
            allow_coercion=False,
        )
        self.input_schema = schema
        self.output_schema = schema

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Return row unchanged (deep copy to prevent mutation).

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with unchanged row data

        Raises:
            ValidationError: If validate_input=True and row fails schema validation.
                This indicates a bug in the upstream source/transform.
        """
        # Optional input validation - crash on wrong types (source bug!)
        if self._validate_input and not self._schema_config.is_dynamic:
            self.input_schema.model_validate(row)  # Raises on failure

        return TransformResult.success(copy.deepcopy(row))

    def close(self) -> None:
        """No resources to release."""
        pass
