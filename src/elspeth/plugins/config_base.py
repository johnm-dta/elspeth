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
                config_copy["schema_config"] = SchemaConfig.from_dict(schema_dict)
            return cls.model_validate(config_copy)
        except ValidationError as e:
            raise PluginConfigError(
                f"Invalid configuration for {cls.__name__}: {e}"
            ) from e
        except ValueError as e:
            raise PluginConfigError(
                f"Invalid configuration for {cls.__name__}: {e}"
            ) from e


class DataPluginConfig(PluginConfig):
    """Base class for data-processing plugin configurations.

    Used by sources, transforms, and sinks that handle data rows.
    Schema is REQUIRED to ensure auditable schema choices.

    Use 'schema: {fields: dynamic}' to accept any fields, or provide
    explicit field definitions with mode (strict/free).
    """

    @model_validator(mode="after")
    def _require_schema(self) -> Self:
        """Validate that schema is provided."""
        if self.schema_config is None:
            raise ValueError(
                "Data plugins require 'schema' configuration. "
                "Use 'schema: {fields: dynamic}' to accept any fields, or "
                "provide explicit field definitions with mode (strict/free)."
            )
        return self


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
