"""Keyword filter transform for blocking content matching regex patterns."""

import re
from typing import Any

from pydantic import Field, field_validator

from elspeth.contracts import Determinism
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config

# ReDoS detection: patterns with nested quantifiers cause catastrophic backtracking
# on adversarial input. E.g., (a+)+ on "aaa...!" is O(2^n).
#
# Known limitations (defense-in-depth, not comprehensive):
#   - Does not detect {n,} brace quantifiers inside groups: (a{2,})+
#   - Cannot see past nested group boundaries: ((a+)b)+
#   - Does not detect alternation-based attacks: (a|a)+
#   - Does not detect overlapping character class repetition: [a-z]+[a-z]+
# These gaps are mitigated by _MAX_PATTERN_LENGTH and the fact that patterns
# come from operator-authored settings.yaml, not arbitrary user input.
_NESTED_QUANTIFIER_RE = re.compile(
    r"[+*]\)["  # quantified group followed by
    r"+*{]"  # another quantifier
    r"|"
    r"\([^)]*[+*][^)]*\)["  # group containing quantifier, followed by
    r"+*{]"  # another quantifier
)

# Maximum pattern length — long patterns increase backtracking risk
_MAX_PATTERN_LENGTH = 1000


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
    if _NESTED_QUANTIFIER_RE.search(pattern):
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

    @field_validator("blocked_patterns")
    @classmethod
    def validate_patterns_not_empty(cls, v: list[str]) -> list[str]:
        """Ensure at least one pattern is provided."""
        if not v:
            raise ValueError("blocked_patterns cannot be empty")
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
    is_batch_aware = False
    creates_tokens = False

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        cfg = KeywordFilterConfig.from_dict(config)
        self._fields = cfg.fields

        # Compile patterns at init - fail fast on invalid regex
        # Validate for ReDoS-prone constructs before compiling
        for pattern in cfg.blocked_patterns:
            _validate_regex_safety(pattern)
        self._compiled_patterns: list[tuple[str, re.Pattern[str]]] = [(pattern, re.compile(pattern)) for pattern in cfg.blocked_patterns]

        # Create schema
        schema = create_schema_from_config(
            cfg.schema_config,
            "KeywordFilterSchema",
            allow_coercion=False,  # Transforms do NOT coerce
        )
        self.input_schema = schema
        self.output_schema = schema

    def process(
        self,
        row: PipelineRow,
        ctx: PluginContext,
    ) -> TransformResult:
        """Scan configured fields for blocked patterns.

        Args:
            row: Input row data
            ctx: Plugin context

        Returns:
            TransformResult.success(row) if no patterns match
            TransformResult.error(reason) if any pattern matches
        """
        fields_to_scan = self._get_fields_to_scan(row)

        for field_name in fields_to_scan:
            # Use PipelineRow access semantics so configured fields can be either
            # original names or normalized names from the contract.
            if field_name not in row:
                continue  # Skip fields not present in this row

            value = row[field_name]

            # Only scan string values
            if not isinstance(value, str):
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
            row,
            success_reason={"action": "filtered"},
        )

    def _get_fields_to_scan(self, row: PipelineRow) -> list[str]:
        """Determine which fields to scan based on config."""
        if self._fields == "all":
            # Scan all string-valued fields
            return [field_name for field_name in row if isinstance(row[field_name], str)]
        elif isinstance(self._fields, str):
            return [self._fields]
        else:
            return self._fields

    def close(self) -> None:
        """Release resources."""
        pass
