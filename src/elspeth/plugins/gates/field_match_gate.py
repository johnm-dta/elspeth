"""FieldMatchGate plugin.

Routes rows based on field value matching (exact, regex, or list membership).
"""

import re
from typing import Any, Literal

from elspeth.plugins.base import BaseGate
from elspeth.plugins.config_base import PluginConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import GateResult, RoutingAction
from elspeth.plugins.schemas import PluginSchema
from elspeth.plugins.sentinels import MISSING


class FieldMatchGateSchema(PluginSchema):
    """Dynamic schema - accepts field match gate config."""

    model_config = {"extra": "allow"}  # noqa: RUF012 - Pydantic class-level config


class FieldMatchGateConfig(PluginConfig):
    """Configuration for field match gate plugin."""

    field: str
    matches: dict[str, str]
    mode: Literal["exact", "regex"] = "exact"
    default_label: str = "no_match"
    strict: bool = False
    case_insensitive: bool = False


class FieldMatchGate(BaseGate):
    """Route rows based on field value matching.

    Supports exact matching, regex patterns, and comma-separated lists.
    Returns route labels that are resolved by the executor via routes config.

    Config options:
        field: Field to match (supports dot notation for nested fields)
        matches: Dict mapping match values/patterns to route labels
        mode: "exact" (default) or "regex"
        default_label: Route label for non-matching values (optional, else "no_match")
        strict: If True, error on missing field (default: False)
        case_insensitive: If True, match case-insensitively (default: False)

    Match keys can be:
        - Single value: "active" -> "active_label"
        - Comma-separated: "US,CA,MX" -> "north_america"
        - Regex pattern (when mode="regex"): r".*@example\\.com$" -> "internal"

    The routes config maps these labels to destinations:
        routes:
          active_label: active_sink
          north_america: na_sink
          internal: internal_sink
          no_match: continue  # or a sink name
    """

    name = "field_match_gate"
    input_schema = FieldMatchGateSchema
    output_schema = FieldMatchGateSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = FieldMatchGateConfig.from_dict(config)
        self._field: str = cfg.field
        self._matches: dict[str, str] = cfg.matches
        self._mode: str = cfg.mode
        self._default_label: str = cfg.default_label
        self._strict: bool = cfg.strict
        self._case_insensitive: bool = cfg.case_insensitive

        # Pre-process matches based on mode
        if self._mode == "regex":
            # Compile regex patterns
            flags = re.IGNORECASE if self._case_insensitive else 0
            self._regex_matches: list[tuple[re.Pattern[str], str]] = [
                (re.compile(pattern, flags), label)
                for pattern, label in self._matches.items()
            ]
            self._expanded_matches: dict[str, str] = {}
        else:
            # Expand comma-separated keys into individual mappings
            self._expanded_matches = {}
            for key, label in self._matches.items():
                for value in key.split(","):
                    normalized = value.strip()
                    if self._case_insensitive:
                        normalized = normalized.lower()
                    self._expanded_matches[normalized] = label
            self._regex_matches = []

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Evaluate field and return routing decision.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            GateResult with route label

        Raises:
            ValueError: If the field is missing and strict mode is enabled
        """
        value = self._get_nested(row, self._field)

        # Handle missing field
        if value is MISSING:
            if self._strict:
                raise ValueError(f"Required field '{self._field}' not found in row")
            return GateResult(
                row=row,
                action=RoutingAction.route(
                    "missing_field",
                    reason={
                        "field": self._field,
                        "result": "missing_field",
                    },
                ),
            )

        # Convert to string for matching
        str_value = str(value)

        # Build base reason for audit
        reason = {
            "field": self._field,
            "value": value,
            "mode": self._mode,
        }

        # Try to find matching route label
        route_label = self._find_matching_label(str_value)

        if route_label:
            reason["matched_label"] = route_label
            return GateResult(
                row=row,
                action=RoutingAction.route(route_label, reason=reason),
            )

        # No match - use default label
        reason["result"] = "default"
        reason["matched_label"] = self._default_label
        return GateResult(
            row=row,
            action=RoutingAction.route(self._default_label, reason=reason),
        )

    def _find_matching_label(self, value: str) -> str | None:
        """Find route label for the given value.

        Args:
            value: String value to match

        Returns:
            Route label if matched, None otherwise
        """
        if self._mode == "regex":
            for pattern, label in self._regex_matches:
                if pattern.search(value):
                    return label
            return None
        else:
            lookup_value = value.lower() if self._case_insensitive else value
            return self._expanded_matches.get(lookup_value)

    def _get_nested(self, data: dict[str, Any], path: str) -> Any:
        """Get value from nested dict using dot notation.

        Args:
            data: Source dictionary
            path: Dot-separated path (e.g., "meta.type")

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
