"""Keyword filter transform for blocking content matching regex patterns."""

import re
from typing import Any

from pydantic import Field, field_validator

from elspeth.contracts import Determinism
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from elspeth.plugins.schema_factory import create_schema_from_config


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
                fields: dynamic
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
        self._on_error = cfg.on_error

        # Compile patterns at init - fail fast on invalid regex
        self._compiled_patterns: list[tuple[str, re.Pattern[str]]] = [
            (pattern, re.compile(pattern)) for pattern in cfg.blocked_patterns
        ]

        # Create schema
        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            "KeywordFilterSchema",
            allow_coercion=False,  # Transforms do NOT coerce
        )
        self.input_schema = schema
        self.output_schema = schema

    def process(
        self,
        row: dict[str, Any],
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
                    context = self._extract_context(value, match)
                    return TransformResult.error(
                        {
                            "reason": "blocked_content",
                            "field": field_name,
                            "matched_pattern": pattern_str,
                            "match_context": context,
                        }
                    )

        # No matches - pass through unchanged
        return TransformResult.success(row)

    def _get_fields_to_scan(self, row: dict[str, Any]) -> list[str]:
        """Determine which fields to scan based on config."""
        if self._fields == "all":
            # Scan all string-valued fields
            return [k for k, v in row.items() if isinstance(v, str)]
        elif isinstance(self._fields, str):
            return [self._fields]
        else:
            return self._fields

    def _extract_context(
        self,
        text: str,
        match: re.Match[str],
        context_chars: int = 40,
    ) -> str:
        """Extract surrounding context around a match."""
        start = max(0, match.start() - context_chars)
        end = min(len(text), match.end() + context_chars)

        context = text[start:end]

        # Add ellipsis markers if truncated
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."

        return context

    def close(self) -> None:
        """Release resources."""
        pass
