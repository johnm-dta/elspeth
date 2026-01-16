"""ThresholdGate plugin.

Routes rows based on numeric threshold comparison.
"""

from typing import Any

from elspeth.contracts import PluginSchema
from elspeth.plugins.base import BaseGate
from elspeth.plugins.config_base import PluginConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult, RoutingAction
from elspeth.plugins.sentinels import MISSING


class ThresholdGateSchema(PluginSchema):
    """Dynamic schema - accepts threshold gate config."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic class-level config


class ThresholdGateConfig(PluginConfig):
    """Configuration for threshold gate plugin."""

    field: str
    threshold: float
    inclusive: bool = False
    cast: bool = True


class ThresholdGate(BaseGate):
    """Route rows based on numeric threshold comparison.

    Compares a numeric field to a threshold and returns route labels
    "above" or "below" based on the comparison result.

    Config options:
        field: Field to compare (supports dot notation for nested fields)
        threshold: Numeric threshold value
        inclusive: If True, >= returns "above" (default: False, > returns "above")
        cast: If True, attempt to cast string values to numbers (default: True)

    Route labels returned:
        "above": Value is above (or equal if inclusive) threshold
        "below": Value is below threshold

    The routes config maps these labels to destinations:
        routes:
          above: high_scores  # or "continue"
          below: low_scores   # or "continue"
    """

    name = "threshold_gate"
    input_schema = ThresholdGateSchema
    output_schema = ThresholdGateSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = ThresholdGateConfig.from_dict(config)
        self._field: str = cfg.field
        self._threshold: float = cfg.threshold
        self._inclusive: bool = cfg.inclusive
        self._cast: bool = cfg.cast

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Evaluate threshold and return routing decision.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            GateResult with routing action

        Raises:
            ValueError: If the field is missing
            TypeError: If the field is not numeric
        """
        value = self._get_nested(row, self._field)

        # Check for missing field
        if value is MISSING:
            raise ValueError(f"Required field '{self._field}' not found in row")

        # Handle type coercion for string values (common with CSV sources)
        if isinstance(value, str) and self._cast:
            try:
                value = float(value)
            except ValueError as err:
                raise TypeError(
                    f"Field '{self._field}' must be numeric, got non-numeric string: '{value}'"
                ) from err
        elif not isinstance(value, int | float):
            raise TypeError(
                f"Field '{self._field}' must be numeric, got {type(value).__name__}"
            )

        # Determine if above threshold
        if self._inclusive:
            is_above = value >= self._threshold
        else:
            is_above = value > self._threshold

        # Determine route label
        route_label = "above" if is_above else "below"

        # Build reason for audit trail
        reason = {
            "field": self._field,
            "value": value,
            "threshold": self._threshold,
            "comparison": ">=" if self._inclusive else ">",
            "result": route_label,
        }

        # Return route label - executor resolves via routes config
        return GateResult(
            row=row,
            action=RoutingAction.route(route_label, reason=reason),
        )

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation.

        Args:
            data: Source dictionary
            path: Dot-separated path (e.g., "metrics.score")

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
