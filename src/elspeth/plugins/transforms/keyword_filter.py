"""Keyword filter transform for blocking content matching regex patterns."""

import re
from typing import Any

from pydantic import Field, field_validator

from elspeth.contracts import Determinism
from elspeth.contracts.contexts import TransformContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.transforms.safety_utils import get_fields_to_scan
from elspeth.plugins.transforms.safety_utils import validate_fields_not_empty as _validate_fields

# ReDoS detection: patterns with nested quantifiers cause catastrophic backtracking
# on adversarial input. E.g., (a+)+ on "aaa...!" is O(2^n).
#
# Known limitations (defense-in-depth, not comprehensive):
#   - Does not detect alternation-based attacks: (a|a)+
#   - Does not detect overlapping character class repetition: [a-z]+[a-z]+
# These gaps are mitigated by _MAX_PATTERN_LENGTH and the fact that patterns
# come from operator-authored settings.yaml, not arbitrary user input.

# Maximum pattern length — long patterns increase backtracking risk
_MAX_PATTERN_LENGTH = 1000


def _brace_quantifier_end(pattern: str, start: int) -> int | None:
    """Return the exclusive end offset for a valid brace quantifier."""
    if start >= len(pattern) or pattern[start] != "{":
        return None

    i = start + 1
    digits_start = i
    while i < len(pattern) and pattern[i].isdigit():
        i += 1
    if i == digits_start:
        return None

    if i < len(pattern) and pattern[i] == "}":
        return i + 1

    if i >= len(pattern) or pattern[i] != ",":
        return None
    i += 1

    while i < len(pattern) and pattern[i].isdigit():
        i += 1
    if i < len(pattern) and pattern[i] == "}":
        return i + 1
    return None


def _nested_repetition_detected(pattern: str) -> bool:
    """Detect nested repeated groups with escape and character-class awareness."""
    group_contains_repetition: list[bool] = []
    ignore_quantifier_positions: set[int] = set()
    i = 0

    while i < len(pattern):
        if i in ignore_quantifier_positions:
            i += 1
            continue

        ch = pattern[i]

        if ch == "\\":
            i += 2
            continue

        if ch == "[":
            i += 1
            while i < len(pattern):
                if pattern[i] == "\\":
                    i += 2
                    continue
                if pattern[i] == "]":
                    i += 1
                    break
                i += 1
            continue

        if ch == "(":
            group_contains_repetition.append(False)
            if i + 1 < len(pattern) and pattern[i + 1] == "?":
                ignore_quantifier_positions.add(i + 1)
            i += 1
            continue

        if ch == ")":
            if not group_contains_repetition:
                i += 1
                continue

            closed_group_contains_repetition = group_contains_repetition.pop()
            quantifier_end = None
            if i + 1 < len(pattern):
                next_ch = pattern[i + 1]
                if next_ch in {"+", "*"}:
                    quantifier_end = i + 2
                elif next_ch == "{":
                    quantifier_end = _brace_quantifier_end(pattern, i + 1)

            if quantifier_end is not None and closed_group_contains_repetition:
                return True

            quantified_group = quantifier_end is not None
            if group_contains_repetition and (closed_group_contains_repetition or quantified_group):
                group_contains_repetition[-1] = True

            i = quantifier_end if quantifier_end is not None else i + 1
            continue

        quantifier_end = None
        if ch in {"+", "*"}:
            quantifier_end = i + 1
        elif ch == "{":
            quantifier_end = _brace_quantifier_end(pattern, i)

        if quantifier_end is not None:
            if group_contains_repetition:
                group_contains_repetition[-1] = True
            i = quantifier_end
            continue

        i += 1

    return False


def _validate_regex_safety(pattern: str) -> None:
    """Reject regex patterns with known ReDoS-prone constructs.

    Checks for nested quantifiers (the primary cause of catastrophic
    backtracking) and excessive pattern length.

    Args:
        pattern: Regex pattern string to validate

    Raises:
        ValueError: If pattern contains ReDoS-prone constructs
    """
    if len(pattern) > _MAX_PATTERN_LENGTH:
        raise ValueError(f"Regex pattern exceeds maximum length ({_MAX_PATTERN_LENGTH} chars): {pattern[:50]}...")
    if _nested_repetition_detected(pattern):
        raise ValueError(f"Regex pattern contains nested quantifiers (ReDoS risk): {pattern}")


class KeywordFilterConfig(TransformDataConfig):
    """Configuration for keyword filter transform.

    Requires:
        fields: Field name(s) to scan, or 'all' for all string fields
        blocked_patterns: Regex patterns that trigger blocking
        schema: Schema configuration for input/output validation
    """

    fields: str | list[str] = Field(
        ...,  # Required, no default
        description="Field name(s) to scan, or 'all' for all string fields",
    )
    blocked_patterns: list[str] = Field(
        ...,  # Required, no default
        description="Regex patterns that trigger blocking",
    )

    @field_validator("fields")
    @classmethod
    def validate_fields_not_empty(cls, v: str | list[str]) -> str | list[str]:
        """Reject empty fields — security transform must scan at least one field."""
        return _validate_fields(v)

    @field_validator("blocked_patterns")
    @classmethod
    def validate_patterns_not_empty(cls, v: list[str]) -> list[str]:
        """Ensure patterns are non-empty, valid regex, and ReDoS-safe."""
        if not v:
            raise ValueError("blocked_patterns cannot be empty")
        for i, pattern in enumerate(v):
            if pattern == "":
                raise ValueError(f"blocked_patterns[{i}] cannot be empty (empty regex matches everything)")
            _validate_regex_safety(pattern)
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"blocked_patterns[{i}] is not a valid regex: {exc}") from exc
        return v


class KeywordFilter(BaseTransform):
    """Filter rows containing blocked content patterns.

    Scans configured fields for regex pattern matches. Rows with matches
    are routed to the on_error sink; rows without matches pass through.

    Config options:
        fields: Field name(s) to scan, or 'all' for all string fields (required)
        blocked_patterns: Regex patterns that trigger blocking (required)
        schema: Schema configuration (required)
        on_error: Sink for blocked rows (required when patterns might match)

    Example YAML:
        transforms:
          - plugin: keyword_filter
            options:
              fields: [message, subject]
              blocked_patterns:
                - "\\\\bpassword\\\\b"
                - "(?i)confidential"
              on_error: quarantine_sink
              schema:
                mode: observed
    """

    name = "keyword_filter"
    determinism = Determinism.DETERMINISTIC
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:77db6f5c05af2cfd"
    config_model = KeywordFilterConfig
    is_batch_aware = False
    creates_tokens = False
    passes_through_input = True

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        cfg = KeywordFilterConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)
        self._fields = cfg.fields
        self._schema_config = cfg.schema_config
        self._output_schema_config = self._build_output_schema_config(cfg.schema_config)

        # Patterns already validated (regex syntax + ReDoS safety) by config validator
        self._compiled_patterns: list[tuple[str, re.Pattern[str]]] = [(pattern, re.compile(pattern)) for pattern in cfg.blocked_patterns]

        self.input_schema, self.output_schema = self._create_schemas(cfg.schema_config, "KeywordFilter")

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        return {
            "schema": {"mode": "observed"},
            "fields": ["keyword_filter_probe_1"],
            "blocked_patterns": [r"(?!x)x"],
        }

    def forward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name="keyword_filter_probe_1",
                value="safe probe content",
            )
        ]

    def process(
        self,
        row: PipelineRow,
        ctx: TransformContext,
    ) -> TransformResult:
        """Scan configured fields for blocked patterns.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult.success(row) if no patterns match
            TransformResult.error(reason) if any pattern matches
        """
        fields_to_scan = get_fields_to_scan(self._fields, row)
        named_fields = self._fields != "all"

        for field_name in fields_to_scan:
            # Use PipelineRow access semantics so configured fields can be either
            # original names or normalized names from the contract.
            if field_name not in row:
                if named_fields:
                    return TransformResult.error(
                        {
                            "reason": "missing_scan_field",
                            "field": field_name,
                            "message": f"Configured scan field '{field_name}' not found in row",
                        },
                        retryable=False,
                    )
                continue

            value = row[field_name]

            # Non-string values in explicitly named fields must fail-closed.
            # The operator configured this field for scanning — if the value
            # isn't scannable, the row must be quarantined, not passed through.
            # In "all" mode, non-strings are already excluded by _get_fields_to_scan.
            if not isinstance(value, str):
                if named_fields:
                    return TransformResult.error(
                        {
                            "reason": "non_string_field",
                            "field": field_name,
                            "actual_type": type(value).__name__,
                        },
                        retryable=False,
                    )
                continue

            # Check each pattern
            for pattern_str, compiled_pattern in self._compiled_patterns:
                match = compiled_pattern.search(value)
                if match:
                    return TransformResult.error(
                        {
                            "reason": "blocked_content",
                            "field": field_name,
                            "matched_pattern": pattern_str,
                            "match_position": match.start(),
                            "match_length": match.end() - match.start(),
                            "field_length": len(value),
                        },
                        retryable=False,
                    )

        # No matches - pass through unchanged
        return TransformResult.success(
            self._align_output_row_contract(row),
            success_reason={"action": "filtered"},
        )

    def close(self) -> None:
        """Release resources."""
        pass
