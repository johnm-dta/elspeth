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

from pydantic import BaseModel, ValidationError, field_validator


class PluginConfigError(Exception):
    """Raised when plugin configuration is invalid."""

    pass


class PluginConfig(BaseModel):
    """Base class for typed plugin configurations.

    Provides common validation patterns and helpful error messages.
    All plugin configs should inherit from this class.
    """

    model_config = {"extra": "forbid"}  # Reject unknown fields

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
            return cls(**config)
        except ValidationError as e:
            raise PluginConfigError(
                f"Invalid configuration for {cls.__name__}: {e}"
            ) from e


class PathConfig(PluginConfig):
    """Base for configs that include file paths.

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
