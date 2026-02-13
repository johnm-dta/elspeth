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

from elspeth.contracts.header_modes import HeaderMode, parse_header_mode
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
        if not isinstance(config, dict):
            raise PluginConfigError(f"Invalid configuration for {cls.__name__}: config must be a dict, got {type(config).__name__}.")

        try:
            config_copy = dict(config)
            if "schema" in config_copy:
                schema_dict = config_copy.pop("schema")
                # Type guard: schema must be a dict (not None, string, list, etc.)
                if not isinstance(schema_dict, dict):
                    raise PluginConfigError(
                        f"Invalid configuration for {cls.__name__}: "
                        f"'schema' must be a dict, got {type(schema_dict).__name__}. "
                        f"Use 'schema: {{mode: observed}}' or provide explicit field definitions."
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

    Use 'schema: {mode: observed}' to infer types from data, or provide
    explicit field definitions with mode (fixed/flexible).

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
            "Use 'schema: {mode: observed}' to infer types from data, or "
            "provide explicit field definitions with mode (fixed/flexible)."
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

    Note: on_success routing is defined at the settings level
    (SourceSettings in core/config.py), not here. The bridge in
    cli_helpers.py injects on_success after plugin construction.
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


class SinkPathConfig(PathConfig):
    """Base config for file-based sink plugins with header output mode.

    Extends PathConfig to add header output configuration for output formatting.
    Sinks own their output format - headers are resolved at sink level,
    not carried through the pipeline.

    Header Mode Options:
        headers: Unified header mode setting. Can be:
            - "normalized": Use Python identifier names (default)
            - "original": Restore original source header names
            - dict: Custom mapping from normalized to output names
    """

    headers: str | dict[str, str] | None = Field(
        default=None,
        description=("Header output mode: 'normalized', 'original', or {field: header} mapping"),
    )

    @field_validator("headers")
    @classmethod
    def _validate_headers(cls, v: str | dict[str, str] | None) -> str | dict[str, str] | None:
        """Validate headers field value.

        Must be 'normalized', 'original', a dict mapping, or None.
        """
        if v is None:
            return v

        if isinstance(v, dict):
            return v

        if isinstance(v, str):
            if v not in ("normalized", "original"):
                raise ValueError(f"Invalid header mode '{v}'. Expected 'normalized', 'original', or mapping dict.")
            return v

        raise ValueError(f"headers must be 'normalized', 'original', or a dict mapping, got {type(v).__name__}")

    @property
    def headers_mode(self) -> HeaderMode:
        """Get resolved header mode.

        Returns NORMALIZED when headers is not set.
        """
        if self.headers is not None:
            return parse_header_mode(self.headers)
        return HeaderMode.NORMALIZED

    @property
    def headers_mapping(self) -> dict[str, str] | None:
        """Get custom header mapping if CUSTOM mode.

        Returns the explicit header mapping dict if headers mode is CUSTOM,
        otherwise None.
        """
        if isinstance(self.headers, dict):
            return self.headers
        return None


class TransformDataConfig(DataPluginConfig):
    """Base config for transform plugins.

    Routing fields (on_success, on_error) are defined at the settings level
    (TransformSettings in core/config.py), not here. This class provides
    plugin-specific configuration like schema and required_input_fields.

    Input Field Requirements:
        Transforms can declare which fields they require in their input using
        `required_input_fields`. This enables DAG validation to catch missing
        field errors at configuration time rather than runtime.

        For template-based transforms (like LLM transforms), use
        `elspeth.core.templates.extract_jinja2_fields()` to discover which
        fields your template references, then declare them explicitly.
    """

    required_input_fields: list[str] | None = Field(
        default=None,
        description=(
            "Fields this transform requires in input. Used for DAG validation "
            "to catch missing field errors at config time. For templates, use "
            "elspeth.core.templates.extract_jinja2_fields() to discover fields."
        ),
    )

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
