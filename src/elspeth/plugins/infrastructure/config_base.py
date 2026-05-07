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
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from elspeth.contracts.header_modes import HeaderMode, parse_header_mode
from elspeth.contracts.schema import SchemaConfig

OutputCollisionPolicy = Literal["fail_if_exists", "auto_increment", "append_or_create"]


class PluginConfigError(Exception):
    """Raised when plugin configuration is invalid.

    Attributes:
        cause: The underlying validation message without the class-name prefix.
               Callers should read this instead of parsing str(exc).
               None only when the raise site omits the keyword argument (legacy
               or non-from_dict paths such as plugin __init__ guards).
        plugin_class: The config class name (e.g. ``'CSVSourceConfig'``), or
                      None when raised outside ``from_dict()``. Callers can use
                      the suffix to infer component type without regex-parsing.
        plugin_name: The runtime plugin name (e.g. ``'csv'``), as it appears
                     in pipeline YAML. Set by ``from_dict()`` when provided by
                     the caller. None when the caller doesn't know the name
                     (direct ``from_dict()`` calls without context) or when
                     raised outside ``from_dict()``.
        component_type: The pipeline component type (``'source'``, ``'sink'``,
                        ``'transform'``), or None when raised outside
                        ``from_dict()`` or when the type is unknown.
                        Set from the config class hierarchy's
                        ``_plugin_component_type`` attribute.
    """

    def __init__(
        self,
        message: str,
        *,
        cause: str | None = None,
        plugin_class: str | None = None,
        plugin_name: str | None = None,
        component_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.cause: str | None = cause
        self.plugin_class: str | None = plugin_class
        self.plugin_name: str | None = plugin_name
        self.component_type: str | None = component_type


def _format_validation_error_location(loc: tuple[Any, ...]) -> str:
    """Render a Pydantic error location as a stable user-facing field path."""
    if not loc:
        return "config"

    parts: list[str] = []
    for item in loc:
        match item:
            case int() as index:
                if parts:
                    parts[-1] = f"{parts[-1]}[{index}]"
                else:
                    parts.append(f"[{index}]")
            case _:
                parts.append(str(item))
    return ".".join(parts)


def _format_validation_error_cause(exc: ValidationError) -> str:
    """Build a class-name-free validation summary from structured Pydantic data."""
    lines: list[str] = []
    for error in exc.errors():
        location = _format_validation_error_location(tuple(error["loc"]))
        message = error["msg"]
        lines.append(f"{location}: {message}")
    return "\n".join(lines)


def _validate_schema_config(value: Any, *, require_dict: bool) -> SchemaConfig | None:
    """Validate and parse schema config from dict or pass through SchemaConfig instance.

    This validator enables Pydantic to accept schema dicts with string field specs
    like "line: str" that need parsing via SchemaConfig.from_dict().

    Args:
        value: The schema config value to validate.
        require_dict: If True, value must be a dict (rejects None and other types).
            Used when 'schema' key was explicitly provided in the input.
    """
    if value is None:
        if require_dict:
            raise ValueError(
                f"'schema' must be a dict, got {type(value).__name__}. "
                f"Use 'schema: {{mode: observed}}' or provide explicit field definitions."
            )
        return None
    if isinstance(value, SchemaConfig):
        return value
    if isinstance(value, dict):
        return SchemaConfig.from_dict(value)
    raise ValueError(
        f"'schema' must be a dict, got {type(value).__name__}. Use 'schema: {{mode: observed}}' or provide explicit field definitions."
    )


class PluginConfig(BaseModel):
    """Base class for typed plugin configurations.

    Provides common validation patterns and helpful error messages.
    All plugin configs should inherit from this class.

    Schema is optional in the base class. Subclasses that process data
    (DataPluginConfig) require schema to be specified.

    The schema field is exposed as "schema" in JSON Schema (for LLM tools and YAML)
    but stored internally as "schema_config" to avoid shadowing Python's built-in.
    """

    _plugin_component_type: ClassVar[str | None] = None

    model_config = {
        "extra": "forbid",
        "frozen": True,
        "populate_by_name": True,  # Allow both "schema" (alias) and "schema_config" (field name)
    }

    schema_config: SchemaConfig | None = Field(
        default=None,
        alias="schema",
        description="Schema configuration for data validation.",
    )

    @field_validator("schema_config", mode="before")
    @classmethod
    def _parse_schema_config(cls, value: Any) -> SchemaConfig | None:
        """Parse schema dict to SchemaConfig, handling string field specs.

        Accepts:
        - SchemaConfig instance (passthrough)
        - dict (parsed via SchemaConfig.from_dict())
        - None (allowed for PluginConfig, rejected by DataPluginConfig override)

        Raises ValueError for other types (str, list, int, etc.)

        Error messages use 'schema' to match the user-facing alias.
        """
        return _validate_schema_config(value, require_dict=False)

    @classmethod
    def from_dict(
        cls,
        config: object,
        *,
        plugin_name: str | None = None,
    ) -> Self:
        """Create config from dict with clear error on validation failure.

        Args:
            config: Configuration value — expected to be a dict but typed as
                ``object`` because callers may pass unsanitized YAML output.
                Non-dict values are rejected with PluginConfigError.
                Accepts "schema" key (preferred, matches YAML format) or
                "schema_config" for backwards compatibility.
            plugin_name: Runtime plugin name (e.g. ``'csv'``). Threaded into
                PluginConfigError so consumers get both the internal class name
                and the user-facing plugin name without string parsing.

        Returns:
            Validated configuration instance.

        Raises:
            PluginConfigError: If configuration is invalid.
        """
        _err_fields: dict[str, Any] = {
            "cause": "",
            "plugin_class": cls.__name__,
            "plugin_name": plugin_name,
            "component_type": cls._plugin_component_type,
        }

        if not isinstance(config, dict):
            _err_fields["cause"] = f"config must be a dict, got {type(config).__name__}."
            raise PluginConfigError(
                f"Invalid configuration for {cls.__name__}: {_err_fields['cause']}",
                **_err_fields,
            )

        # Pre-validate: reject explicit non-dict schema values.
        # Pydantic can't distinguish "key absent" from "key explicitly None",
        # so we check here before passing to model_validate.
        for key in ("schema", "schema_config"):
            if key in config:
                value = config[key]
                if not isinstance(value, (dict, SchemaConfig)):
                    _err_fields["cause"] = (
                        f"'schema' must be a dict, got {type(value).__name__}. "
                        f"Use 'schema: {{mode: observed}}' or provide explicit field definitions."
                    )
                    raise PluginConfigError(
                        f"Invalid configuration for {cls.__name__}: {_err_fields['cause']}",
                        **_err_fields,
                    )

        try:
            # model_validate handles the schema alias and field_validator parses dicts
            return cls.model_validate(config)
        except ValidationError as e:
            _err_fields["cause"] = _format_validation_error_cause(e)
            raise PluginConfigError(
                f"Invalid configuration for {cls.__name__}: {_err_fields['cause']}",
                **_err_fields,
            ) from e
        except ValueError as e:
            _err_fields["cause"] = str(e)
            raise PluginConfigError(
                f"Invalid configuration for {cls.__name__}: {e}",
                **_err_fields,
            ) from e


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

    Component Type Enforcement:
        All subclasses must set ``_plugin_component_type`` to ``'source'``,
        ``'sink'``, or ``'transform'`` — either directly or by inheriting
        from an intermediate base (SourceDataConfig, SinkPathConfig,
        TransformDataConfig). Intermediate bases that intentionally leave
        the type unset must declare ``_component_type_exempt = True`` in
        their own class body; this exemption does not propagate to children.
    """

    _component_type_exempt: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "_component_type_exempt" in cls.__dict__:
            return
        if cls._plugin_component_type is None:
            raise TypeError(
                f"{cls.__qualname__} inherits from DataPluginConfig but does not "
                f"set _plugin_component_type. Set it to 'source', 'sink', or "
                f"'transform' (directly or via an intermediate base like "
                f"SourceDataConfig, SinkPathConfig, or TransformDataConfig)."
            )

    # Override parent's Optional field with required field.
    # This provides both runtime validation (Pydantic) and static typing (mypy).
    # Alias matches parent - required for Pydantic to properly override.
    # The parent's model_validator handles parsing; this just enforces required.
    schema_config: SchemaConfig = Field(
        ...,
        alias="schema",
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

    _component_type_exempt: ClassVar[bool] = True

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

    _plugin_component_type: ClassVar[str | None] = "source"

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
    - field_mapping: Override specific normalized names

    Field normalization is mandatory at all source boundaries. Raw headers
    are always normalized to valid Python identifiers. The columns config
    provides clean names for headerless files (mutually exclusive path).
    """

    columns: list[str] | None = None
    field_mapping: dict[str, str] | None = None

    @model_validator(mode="after")
    def _validate_normalization_options(self) -> Self:
        """Validate field normalization option interactions."""
        from elspeth.core.identifiers import validate_field_names

        # Validate columns entries are valid identifiers and not keywords
        if self.columns is not None:
            validate_field_names(self.columns, "columns")

        # Validate field_mapping values are valid identifiers and not keywords
        if self.field_mapping is not None and self.field_mapping:
            validate_field_names(list(self.field_mapping.values()), "field_mapping values")

        return self


def validate_headers_value(v: str | dict[str, str] | None) -> str | dict[str, str] | None:
    """Validate a headers config value — shared by SinkPathConfig and AzureBlobSinkConfig.

    Returns the validated value unchanged. Raises ValueError for invalid inputs.
    """
    if v is None:
        return v
    if isinstance(v, dict):
        from elspeth.core.identifiers import validate_field_names

        if not v:
            raise ValueError("headers custom mapping must not be empty — use 'normalized' or 'original' for non-custom modes")
        validate_field_names(list(v.keys()), "headers custom mapping keys")
        for source_name, target_name in v.items():
            if not target_name.strip():
                raise ValueError(f"headers custom mapping value for {source_name!r} must not be empty or whitespace-only")
        targets = list(v.values())
        duplicates = [t for t in targets if targets.count(t) > 1]
        if duplicates:
            raise ValueError(f"Duplicate header mapping targets: {sorted(set(duplicates))}. Each output header must be unique.")
        return v
    if isinstance(v, str):
        if v not in ("normalized", "original"):
            raise ValueError(f"Invalid header mode '{v}'. Expected 'normalized', 'original', or mapping dict.")
        return v
    raise ValueError(f"headers must be 'normalized', 'original', or a dict mapping, got {type(v).__name__}")


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

    _plugin_component_type: ClassVar[str | None] = "sink"

    headers: str | dict[str, str] | None = Field(
        default=None,
        description=("Header output mode: 'normalized', 'original', or {field: header} mapping"),
    )
    collision_policy: OutputCollisionPolicy | None = Field(
        default=None,
        description=(
            "Optional local output collision policy. "
            "'fail_if_exists' refuses an existing write target; "
            "'auto_increment' picks a free sibling path; "
            "'append_or_create' is valid with append mode."
        ),
    )

    @field_validator("headers")
    @classmethod
    def _validate_headers(cls, v: str | dict[str, str] | None) -> str | dict[str, str] | None:
        return validate_headers_value(v)

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

    _plugin_component_type: ClassVar[str | None] = "transform"

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

    @property
    def declared_input_fields(self) -> frozenset[str]:
        """Normalized runtime view of ``required_input_fields``.

        ``required_input_fields`` stays list-shaped in config models so JSON
        schema and YAML remain natural. Runtime declaration contracts need an
        immutable, order-insensitive set surface, so transforms normalize the
        value through this property at construction time.
        """
        if self.required_input_fields is None:
            return frozenset()
        return frozenset(self.required_input_fields)
