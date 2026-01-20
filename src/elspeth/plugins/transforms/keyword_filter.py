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
        """Process a single row - placeholder for now."""
        return TransformResult.success(row)

    def close(self) -> None:
        """Release resources."""
        pass
