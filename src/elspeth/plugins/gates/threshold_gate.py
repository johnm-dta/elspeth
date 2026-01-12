"""ThresholdGate plugin.

Routes rows based on numeric threshold comparison.
"""

from typing import Any

from elspeth.plugins.base import BaseGate
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult, RoutingAction
from elspeth.plugins.schemas import PluginSchema


class ThresholdGateSchema(PluginSchema):
    """Dynamic schema - accepts threshold gate config."""

    model_config = {"extra": "allow"}


class ThresholdGate(BaseGate):
    """Route rows based on numeric threshold comparison.

    Compares a numeric field to a threshold and routes to different sinks
    based on whether the value is above or below.

    Config options:
        field: Field to compare (supports dot notation for nested fields)
        threshold: Numeric threshold value
        above_sink: Sink name for values above threshold (optional)
        below_sink: Sink name for values below threshold (optional)
        inclusive: If True, >= routes to above_sink (default: False, > routes above)

    If above_sink or below_sink is not specified, rows that would go there
    continue to the next transform instead.
    """

    name = "threshold_gate"
    input_schema = ThresholdGateSchema
    output_schema = ThresholdGateSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._field: str = config["field"]
        self._threshold: float = config["threshold"]
        self._above_sink: str | None = config.get("above_sink")
        self._below_sink: str | None = config.get("below_sink")
        self._inclusive: bool = config.get("inclusive", False)

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
        if value is _MISSING:
            raise ValueError(f"Required field '{self._field}' not found in row")

        # Check for numeric type
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"Field '{self._field}' must be numeric, got {type(value).__name__}"
            )

        # Determine if above threshold
        if self._inclusive:
            is_above = value >= self._threshold
        else:
            is_above = value > self._threshold

        # Build reason for audit trail
        reason = {
            "field": self._field,
            "value": value,
            "threshold": self._threshold,
            "comparison": ">=" if self._inclusive else ">",
            "result": "above" if is_above else "below",
        }

        # Route based on comparison
        if is_above:
            if self._above_sink:
                return GateResult(
                    row=row,
                    action=RoutingAction.route_to_sink(self._above_sink, reason=reason),
                )
            return GateResult(row=row, action=RoutingAction.continue_(reason=reason))
        else:
            if self._below_sink:
                return GateResult(
                    row=row,
                    action=RoutingAction.route_to_sink(self._below_sink, reason=reason),
                )
            return GateResult(row=row, action=RoutingAction.continue_(reason=reason))

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation.

        Args:
            data: Source dictionary
            path: Dot-separated path (e.g., "metrics.score")

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
