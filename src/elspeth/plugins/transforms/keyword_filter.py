"""Keyword filter transform for blocking content matching regex patterns."""

from pydantic import Field, field_validator

from elspeth.plugins.config_base import TransformDataConfig


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
