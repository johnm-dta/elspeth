"""PassThrough transform plugin.

Passes rows through unchanged. Useful for testing and debugging pipelines.
"""

import copy
from typing import Any

from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schemas import PluginSchema


class PassThroughSchema(PluginSchema):
    """Dynamic schema - accepts any fields."""

    model_config = {"extra": "allow"}


class PassThrough(BaseTransform):
    """Pass rows through unchanged.

    Use cases:
    - Testing pipeline wiring without modification
    - Debugging data flow (add logging in subclass)
    - Placeholder for future transform logic

    Config options:
        None (accepts empty config)
    """

    name = "passthrough"
    input_schema = PassThroughSchema
    output_schema = PassThroughSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Return row unchanged (deep copy to prevent mutation).

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with unchanged row data
        """
        return TransformResult.success(copy.deepcopy(row))

    def close(self) -> None:
        """No resources to release."""
        pass
