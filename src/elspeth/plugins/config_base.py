# src/elspeth/plugins/config_base.py
"""Base classes for typed plugin configurations.

This module provides base classes that plugins inherit from to get:
- Strict validation (reject unknown fields)
- Factory methods with clear error messages
- Common validation patterns (path handling, etc.)

Example usage:
    class CSVSourceConfig(PathConfig):
        delimiter: str = ","
        encoding: str = "utf-8"

    cfg = CSVSourceConfig.from_dict(config)
    path = cfg.path  # Direct access, fails fast if missing
"""

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from elspeth.contracts.schema import SchemaConfig


class PluginConfigError(Exception):
    """Raised when plugin configuration is invalid."""

    pass


class PluginConfig(BaseModel):
    """Base class for typed plugin configurations.

    Provides common validation patterns and helpful error messages.
    All plugin configs should inherit from this class.

    Schema is optional in the base class. Subclasses that process data
    (DataPluginConfig) require schema to be specified.
    """

    model_config = {"extra": "forbid"}  # Reject unknown fields

    schema_config: SchemaConfig | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
        """Create config from dict with clear error on validation failure.

        Args:
            config: Dictionary of configuration values.

        Returns:
            Validated configuration instance.

        Raises:
            PluginConfigError: If configuration is invalid.
        """
        try:
            config_copy = dict(config)
            if "schema" in config_copy:
                schema_dict = config_copy.pop("schema")
                # Type guard: schema must be a dict (not None, string, list, etc.)
                if not isinstance(schema_dict, dict):
                    raise PluginConfigError(
                        f"Invalid configuration for {cls.__name__}: "
                        f"'schema' must be a dict, got {type(schema_dict).__name__}. "
                        f"Use 'schema: {{fields: dynamic}}' or provide explicit field definitions."
                    )
                config_copy["schema_config"] = SchemaConfig.from_dict(schema_dict)
            return cls.model_validate(config_copy)
        except ValidationError as e:
            raise PluginConfigError(f"Invalid configuration for {cls.__name__}: {e}") from e
        except ValueError as e:
            raise PluginConfigError(f"Invalid configuration for {cls.__name__}: {e}") from e


class DataPluginConfig(PluginConfig):
    """Base class for data-processing plugin configurations.

    Used by sources, transforms, and sinks that handle data rows.
    Schema is REQUIRED to ensure auditable schema choices.

    Use 'schema: {fields: dynamic}' to accept any fields, or provide
    explicit field definitions with mode (strict/free).

    Type Safety:
        This class overrides schema_config from Optional to required.
        Pydantic validates this at construction time, and mypy sees the
        narrowed type - no cast() needed in plugin implementations.
    """

    # Override parent's Optional field with required field.
    # This provides both runtime validation (Pydantic) and static typing (mypy).
    schema_config: SchemaConfig = Field(
        ...,
        description=(
            "Schema configuration for data validation. "
            "Use 'schema: {fields: dynamic}' to accept any fields, or "
            "provide explicit field definitions with mode (strict/free)."
        ),
    )


class PathConfig(DataPluginConfig):
    """Base for configs that include file paths.

    Extends DataPluginConfig because file-based plugins process data rows
    and therefore require schema configuration.

    Provides path validation and resolution relative to a base directory.
    """

    path: str

    @field_validator("path")
    @classmethod
    def validate_path_not_empty(cls, v: str) -> str:
        """Validate that path is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("path cannot be empty")
        return v

    def resolved_path(self, base_dir: Path | None = None) -> Path:
        """Resolve path relative to base directory if provided.

        Args:
            base_dir: Base directory for relative path resolution.
                     If None, path is returned as-is.

        Returns:
            Resolved Path object.
        """
        p = Path(self.path)
        if base_dir and not p.is_absolute():
            return base_dir / p
        return p


class SourceDataConfig(PathConfig):
    """Base config for source plugins with quarantine routing.

    Extends PathConfig to add required on_validation_failure field.
    All sources must specify where non-conformant rows go.
    """

    on_validation_failure: str = Field(
        ...,  # Required - no default
        description="Sink name for non-conformant rows, or 'discard' for explicit drop",
    )

    @field_validator("on_validation_failure")
    @classmethod
    def validate_on_validation_failure(cls, v: str) -> str:
        """Ensure on_validation_failure is not empty."""
        if not v or not v.strip():
            raise ValueError("on_validation_failure must be a sink name or 'discard'")
        return v.strip()


class TabularSourceDataConfig(SourceDataConfig):
    """Config for sources that read tabular external data with headers.

    Extends SourceDataConfig with field normalization options:
    - columns: Explicit column names for headerless files
    - normalize_fields: Auto-normalize messy headers to identifiers
    - field_mapping: Override specific normalized names

    See docs/plans/2026-01-29-field-normalization-design.md for full specification.
    """

    columns: list[str] | None = None
    normalize_fields: bool = False
    field_mapping: dict[str, str] | None = None

    @model_validator(mode="after")
    def _validate_normalization_options(self) -> Self:
        """Validate field normalization option interactions."""
        from elspeth.core.identifiers import validate_field_names

        # normalize_fields + columns is invalid
        if self.columns is not None and self.normalize_fields:
            raise ValueError("normalize_fields cannot be used with columns config. The columns config already provides clean names.")

        # field_mapping requires normalize_fields or columns
        if self.field_mapping is not None and not self.normalize_fields and self.columns is None:
            raise ValueError("field_mapping requires normalize_fields: true or columns config")

        # Validate columns entries are valid identifiers and not keywords
        if self.columns is not None:
            validate_field_names(self.columns, "columns")

        # Validate field_mapping values are valid identifiers and not keywords
        if self.field_mapping is not None and self.field_mapping:
            validate_field_names(list(self.field_mapping.values()), "field_mapping values")

        return self


class TransformDataConfig(DataPluginConfig):
    """Base config for transform plugins with error routing.

    Extends DataPluginConfig to add optional on_error field.
    Transforms that can return TransformResult.error() should configure
    where those rows go.

    Input Field Requirements:
        Transforms can declare which fields they require in their input using
        `required_input_fields`. This enables DAG validation to catch missing
        field errors at configuration time rather than runtime.

        For template-based transforms (like LLM transforms), use
        `elspeth.core.templates.extract_jinja2_fields()` to discover which
        fields your template references, then declare them explicitly.
    """

    on_error: str | None = Field(
        default=None,
        description="Sink name for rows that cannot be processed, or 'discard'. Required if transform can return errors.",
    )

    required_input_fields: list[str] | None = Field(
        default=None,
        description=(
            "Fields this transform requires in input. Used for DAG validation "
            "to catch missing field errors at config time. For templates, use "
            "elspeth.core.templates.extract_jinja2_fields() to discover fields."
        ),
    )

    @field_validator("on_error")
    @classmethod
    def validate_on_error(cls, v: str | None) -> str | None:
        """Ensure on_error is not empty string."""
        if v is not None and not v.strip():
            raise ValueError("on_error must be a sink name, 'discard', or omitted entirely")
        return v.strip() if v else None

    @field_validator("required_input_fields")
    @classmethod
    def validate_required_input_fields(cls, v: list[str] | None) -> list[str] | None:
        """Validate required_input_fields contains valid identifiers.

        Important distinction for LLM transforms:
        - None = not specified (triggers error if template has row references)
        - [] = explicit opt-out (accepts runtime risk, no error)
        - [fields...] = explicit declaration (DAG validates these)
        """
        if v is None:
            return None

        # Empty list is INTENTIONALLY preserved - it's an explicit opt-out
        # for LLM templates that accept runtime risk of missing fields
        if len(v) == 0:
            return []

        result: list[str] = []
        for i, name in enumerate(v):
            if not isinstance(name, str):
                raise ValueError(f"required_input_fields[{i}] must be a string, got {type(name).__name__}")
            name = name.strip()
            if not name:
                raise ValueError(f"required_input_fields[{i}] cannot be empty")
            if not name.isidentifier():
                raise ValueError(f"required_input_fields[{i}] must be a valid Python identifier, got '{name}'")
            result.append(name)

        # Check for duplicates
        if len(result) != len(set(result)):
            duplicates = sorted({n for n in result if result.count(n) > 1})
            raise ValueError(f"Duplicate field names in required_input_fields: {', '.join(duplicates)}")

        return result
