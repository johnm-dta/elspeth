"""FieldMatchGate plugin.

Routes rows based on field value matching (exact, regex, or list membership).
"""

import re
from typing import Any

from elspeth.plugins.base import BaseGate
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult, RoutingAction
from elspeth.plugins.schemas import PluginSchema


class FieldMatchGateSchema(PluginSchema):
    """Dynamic schema - accepts field match gate config."""

    model_config = {"extra": "allow"}


class FieldMatchGate(BaseGate):
    """Route rows based on field value matching.

    Supports exact matching, regex patterns, and comma-separated lists.

    Config options:
        field: Field to match (supports dot notation for nested fields)
        routes: Dict mapping match values/patterns to sink names
        mode: "exact" (default) or "regex"
        default_sink: Sink for non-matching values (optional, else continue)
        strict: If True, error on missing field (default: False)
        case_insensitive: If True, match case-insensitively (default: False)

    Route keys can be:
        - Single value: "active" -> "active_sink"
        - Comma-separated: "US,CA,MX" -> "north_america_sink"
        - Regex pattern (when mode="regex"): r".*@example\\.com$" -> "internal_sink"
    """

    name = "field_match_gate"
    input_schema = FieldMatchGateSchema
    output_schema = FieldMatchGateSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._field: str = config["field"]
        self._routes: dict[str, str] = config["routes"]
        self._mode: str = config.get("mode", "exact")
        self._default_sink: str | None = config.get("default_sink")
        self._strict: bool = config.get("strict", False)
        self._case_insensitive: bool = config.get("case_insensitive", False)

        # Pre-process routes based on mode
        if self._mode == "regex":
            # Compile regex patterns
            flags = re.IGNORECASE if self._case_insensitive else 0
            self._regex_routes: list[tuple[re.Pattern[str], str]] = [
                (re.compile(pattern, flags), sink)
                for pattern, sink in self._routes.items()
            ]
            self._expanded_routes: dict[str, str] = {}
        else:
            # Expand comma-separated keys into individual mappings
            self._expanded_routes = {}
            for key, sink in self._routes.items():
                for value in key.split(","):
                    normalized = value.strip()
                    if self._case_insensitive:
                        normalized = normalized.lower()
                    self._expanded_routes[normalized] = sink
            self._regex_routes = []

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Evaluate field and return routing decision.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            GateResult with routing action

        Raises:
            ValueError: If the field is missing and strict mode is enabled
        """
        value = self._get_nested(row, self._field)

        # Handle missing field
        if value is _MISSING:
            if self._strict:
                raise ValueError(f"Required field '{self._field}' not found in row")
            return GateResult(
                row=row,
                action=RoutingAction.continue_(reason={
                    "field": self._field,
                    "result": "missing_field",
                }),
            )

        # Convert to string for matching
        str_value = str(value)

        # Build base reason for audit
        reason = {
            "field": self._field,
            "value": value,
            "mode": self._mode,
        }

        # Try to find matching route
        sink_name = self._find_matching_sink(str_value)

        if sink_name:
            reason["matched_sink"] = sink_name
            return GateResult(
                row=row,
                action=RoutingAction.route_to_sink(sink_name, reason=reason),
            )

        # No match - use default or continue
        if self._default_sink:
            reason["result"] = "default"
            reason["matched_sink"] = self._default_sink
            return GateResult(
                row=row,
                action=RoutingAction.route_to_sink(self._default_sink, reason=reason),
            )

        reason["result"] = "no_match"
        return GateResult(row=row, action=RoutingAction.continue_(reason=reason))

    def _find_matching_sink(self, value: str) -> str | None:
        """Find sink for the given value.

        Args:
            value: String value to match

        Returns:
            Sink name if matched, None otherwise
        """
        if self._mode == "regex":
            for pattern, sink in self._regex_routes:
                if pattern.search(value):
                    return sink
            return None
        else:
            lookup_value = value.lower() if self._case_insensitive else value
            return self._expanded_routes.get(lookup_value)

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation.

        Args:
            data: Source dictionary
            path: Dot-separated path (e.g., "meta.type")

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
