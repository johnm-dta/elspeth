"""FilterGate plugin.

Routes rows based on filter conditions instead of silently dropping them.
This maintains the "no silent drops" invariant - filtered rows are explicitly
routed to a discard sink for audit trail completeness.
"""

from typing import Any, Callable

from pydantic import model_validator

from elspeth.plugins.base import BaseGate
from elspeth.plugins.config_base import PluginConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult, RoutingAction
from elspeth.plugins.schemas import PluginSchema


class FilterGateSchema(PluginSchema):
    """Dynamic schema - accepts filter gate config."""

    model_config = {"extra": "allow"}


# Supported comparison operators and their evaluation functions
_OPERATORS: dict[str, tuple[str, Callable[[Any, Any], bool]]] = {
    "greater_than": (">", lambda v, t: v > t),
    "less_than": ("<", lambda v, t: v < t),
    "greater_than_or_equal": (">=", lambda v, t: v >= t),
    "less_than_or_equal": ("<=", lambda v, t: v <= t),
    "equals": ("==", lambda v, t: v == t),
    "not_equals": ("!=", lambda v, t: v != t),
}


class FilterGateConfig(PluginConfig):
    """Configuration for filter gate plugin."""

    field: str
    allow_missing: bool = False

    # Comparison operators - exactly one must be provided
    greater_than: float | int | None = None
    less_than: float | int | None = None
    greater_than_or_equal: float | int | None = None
    less_than_or_equal: float | int | None = None
    equals: Any = None
    not_equals: Any = None

    @model_validator(mode="after")
    def validate_exactly_one_operator(self) -> "FilterGateConfig":
        """Ensure exactly one comparison operator is specified."""
        operators_present = self._get_operators_present()

        if len(operators_present) == 0:
            raise ValueError(
                f"No comparison operator specified. Use one of: "
                f"{', '.join(_OPERATORS.keys())}"
            )
        if len(operators_present) > 1:
            raise ValueError(
                f"Multiple comparison operators specified: "
                f"{' and '.join(operators_present)}"
            )
        return self

    def _get_operators_present(self) -> list[str]:
        """Return list of operator names that have non-None values."""
        # Direct attribute access - no getattr with default
        present = []
        if self.greater_than is not None:
            present.append("greater_than")
        if self.less_than is not None:
            present.append("less_than")
        if self.greater_than_or_equal is not None:
            present.append("greater_than_or_equal")
        if self.less_than_or_equal is not None:
            present.append("less_than_or_equal")
        if self.equals is not None:
            present.append("equals")
        if self.not_equals is not None:
            present.append("not_equals")
        return present

    def get_operator(self) -> tuple[str, Any]:
        """Return the operator name and threshold value.

        Must be called after validation - assumes exactly one operator is set.

        Returns:
            Tuple of (operator_name, threshold_value)
        """
        # Direct attribute access - checked in order
        if self.greater_than is not None:
            return ("greater_than", self.greater_than)
        if self.less_than is not None:
            return ("less_than", self.less_than)
        if self.greater_than_or_equal is not None:
            return ("greater_than_or_equal", self.greater_than_or_equal)
        if self.less_than_or_equal is not None:
            return ("less_than_or_equal", self.less_than_or_equal)
        if self.equals is not None:
            return ("equals", self.equals)
        if self.not_equals is not None:
            return ("not_equals", self.not_equals)
        # Should never reach here after validation
        raise ValueError("No operator specified")


class FilterGate(BaseGate):
    """Route rows based on filter conditions.

    Instead of silently dropping rows that don't pass the filter, routes them
    with a "discard" label. This maintains the "no silent drops" invariant
    required by the architecture.

    Config options:
        field: Field to evaluate (supports dot notation for nested fields)
        allow_missing: If True, rows with missing field get "pass" label (default: False)

    Comparison operators (exactly one required):
        greater_than: Value must be > threshold
        less_than: Value must be < threshold
        greater_than_or_equal: Value must be >= threshold
        less_than_or_equal: Value must be <= threshold
        equals: Value must equal threshold (supports any type)
        not_equals: Value must not equal threshold

    Route labels returned:
        "pass": Row passes the filter condition
        "discard": Row fails the filter condition

    The routes config maps these labels to destinations:
        routes:
          pass: continue  # or a sink name
          discard: discard_sink  # or "continue" to not discard

    Example:
        FilterGate({
            "field": "score",
            "greater_than": 0.5,
        })
    """

    name = "filter_gate"
    input_schema = FilterGateSchema
    output_schema = FilterGateSchema

    # Expose operators for evaluate method
    _OPERATORS = _OPERATORS

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = FilterGateConfig.from_dict(config)
        self._field: str = cfg.field
        self._allow_missing: bool = cfg.allow_missing

        # Get the validated operator (already validated by config)
        self._operator_name, self._threshold = cfg.get_operator()

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Evaluate filter condition and return routing decision.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            GateResult with route label:
            - "pass": Row passes filter
            - "discard": Row fails filter
        """
        value = self._get_nested(row, self._field)

        # Handle missing field
        if value is _MISSING:
            if self._allow_missing:
                return GateResult(
                    row=row,
                    action=RoutingAction.route("pass", reason={
                        "field": self._field,
                        "result": "missing_field_allowed",
                    }),
                )
            return GateResult(
                row=row,
                action=RoutingAction.route("discard", reason={
                    "field": self._field,
                    "result": "filtered_missing_field",
                }),
            )

        # Evaluate the comparison
        symbol, compare_fn = self._OPERATORS[self._operator_name]
        passes = compare_fn(value, self._threshold)

        # Build reason for audit trail
        reason = {
            "field": self._field,
            "value": value,
            "condition": self._operator_name,
            "threshold": self._threshold,
            "comparison": symbol,
        }

        if passes:
            reason["result"] = "pass"
            return GateResult(
                row=row,
                action=RoutingAction.route("pass", reason=reason),
            )

        reason["result"] = "discard"
        return GateResult(
            row=row,
            action=RoutingAction.route("discard", reason=reason),
        )

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
