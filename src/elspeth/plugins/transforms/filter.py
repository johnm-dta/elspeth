"""Filter transform plugin.

Filters rows based on field conditions.
"""

import copy
import re
from typing import Any

from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schemas import PluginSchema


class FilterSchema(PluginSchema):
    """Dynamic schema - accepts any fields."""

    model_config = {"extra": "allow"}


class Filter(BaseTransform):
    """Filter rows based on field conditions.

    Returns success with row if condition passes, success with row=None if filtered.

    Config options:
        field: Field to check (supports dot notation for nested fields)
        allow_missing: If True, missing fields pass filter (default: False)

        Conditions (exactly one required):
        - equals: Field must equal this value
        - not_equals: Field must not equal this value
        - greater_than: Field must be > this value (numeric)
        - less_than: Field must be < this value (numeric)
        - contains: Field must contain this substring
        - matches: Field must match this regex pattern
        - in: Field must be one of these values (list)
    """

    name = "filter"
    input_schema = FilterSchema
    output_schema = FilterSchema

    # Condition types we support
    _CONDITION_KEYS = {
        "equals",
        "not_equals",
        "greater_than",
        "less_than",
        "contains",
        "matches",
        "in",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._field: str = config["field"]
        self._allow_missing: bool = config.get("allow_missing", False)

        # Find which condition is specified
        found_conditions = self._CONDITION_KEYS & set(config.keys())
        if not found_conditions:
            raise ValueError(
                f"Filter requires a condition. "
                f"Valid conditions: {sorted(self._CONDITION_KEYS)}"
            )

        # Store condition type and value
        self._condition_type = found_conditions.pop()
        self._condition_value = config[self._condition_type]

        # Pre-compile regex if using matches
        if self._condition_type == "matches":
            self._regex: re.Pattern[str] | None = re.compile(self._condition_value)
        else:
            self._regex = None

    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Apply filter condition to row.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult with row if passes, row=None if filtered
        """
        field_value = self._get_nested(row, self._field)

        # Handle missing field
        if field_value is _MISSING:
            if self._allow_missing:
                return TransformResult.success(copy.deepcopy(row))
            return TransformResult.success(None)  # Filtered out

        # Apply condition
        passes = self._evaluate_condition(field_value)

        if passes:
            return TransformResult.success(copy.deepcopy(row))
        return TransformResult.success(None)  # Filtered out

    def _evaluate_condition(self, value: Any) -> bool:
        """Evaluate the condition against a field value.

        Args:
            value: Field value to check

        Returns:
            True if condition passes, False if filtered
        """
        match self._condition_type:
            case "equals":
                return value == self._condition_value
            case "not_equals":
                return value != self._condition_value
            case "greater_than":
                return value > self._condition_value
            case "less_than":
                return value < self._condition_value
            case "contains":
                return self._condition_value in str(value)
            case "matches":
                return bool(self._regex.search(str(value)))
            case "in":
                return value in self._condition_value
            case _:
                return False  # Should never reach here

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation.

        Args:
            data: Source dictionary
            path: Dot-separated path (e.g., "meta.status")

        Returns:
            Value at path or _MISSING sentinel
        """
        parts = path.split(".")
        current: Any = data

        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return _MISSING
            current = current[part]

        return current

    def close(self) -> None:
        """No resources to release."""
        pass


# Sentinel for missing values
class _MissingSentinel:
    """Sentinel to distinguish missing fields from None values."""

    pass


_MISSING = _MissingSentinel()
