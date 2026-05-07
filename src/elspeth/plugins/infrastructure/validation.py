"""Plugin configuration validation subsystem.

Validates plugin configurations BEFORE instantiation, providing clear error
messages and enabling test fixtures to bypass validation when needed.

Design:
- Validation is separate from plugin construction
- Returns structured errors (not exceptions) for better error messages
- Validates against Pydantic config models (CSVSourceConfig, etc.)
- Does NOT instantiate plugins (just validates config)

Usage:
    from elspeth.plugins.infrastructure.validation import validate_source_config

    errors = validate_source_config("csv", config)
    if errors:
        raise ValueError(f"Invalid config: {errors}")
    source = CSVSource(config)  # Assumes config is valid
"""

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from elspeth.contracts.plugin_protocols import PluginConfigProtocol
from elspeth.plugins.infrastructure.config_base import PluginConfigError


class UnknownPluginTypeError(ValueError):
    """Raised when a plugin type string is not in the validator's mapping."""


@dataclass(frozen=True, slots=True)
class ValidationError:
    """Structured validation error.

    Frozen: error records are immutable evidence — once created, the
    captured field, message, and value must not be modified.

    Attributes:
        field: Field name that failed validation
        message: Human-readable error message
        value: The invalid value (for debugging)
    """

    field: str
    message: str
    value: Any

    def __post_init__(self) -> None:
        if not self.field:
            raise ValueError("ValidationError.field must not be empty")
        if not self.message:
            raise ValueError("ValidationError.message must not be empty")


def validate_source_config(
    source_type: str,
    config: dict[str, Any],
) -> list[ValidationError]:
    """Validate source plugin configuration.

    Args:
        source_type: Plugin type name (e.g., "csv", "json")
        config: Plugin configuration dict

    Returns:
        List of validation errors (empty if valid)
    """
    # Get config model for source type
    config_model = get_source_config_model(source_type)

    # Handle special case: null_source has no config class
    if config_model is None:
        return []  # No validation needed

    # Validate using Pydantic
    try:
        config_model.from_dict(config, plugin_name=source_type)
        return []  # Valid
    except PydanticValidationError as e:
        return _extract_errors(e)
    except PluginConfigError as e:
        return _extract_wrapped_plugin_config_error(e, config)


def get_source_config_model(source_type: str) -> type[PluginConfigProtocol] | None:
    """Get Pydantic config model for source type.

    Resolves the plugin class via pluggy discovery (PluginManager), then
    calls get_config_model() on it.

    Returns:
        Config model class, or None for sources with no config (e.g., null_source)

    Raises:
        UnknownPluginTypeError: If source_type is not a registered plugin.
    """
    from elspeth.plugins.infrastructure.manager import PluginNotFoundError, get_shared_plugin_manager

    try:
        plugin_cls = get_shared_plugin_manager().get_source_by_name(source_type)
    except PluginNotFoundError as exc:
        raise UnknownPluginTypeError(f"Unknown source type: {source_type}") from exc
    return plugin_cls.get_config_model()


def validate_transform_config(
    transform_type: str,
    config: dict[str, Any],
) -> list[ValidationError]:
    """Validate transform plugin configuration.

    Args:
        transform_type: Plugin type name (e.g., "passthrough", "field_mapper")
        config: Plugin configuration dict

    Returns:
        List of validation errors (empty if valid)
    """
    # Get config model for transform type (config needed for provider dispatch)
    config_model = get_transform_config_model(transform_type, config)

    if config_model is None:
        return []  # No validation needed

    # Validate using Pydantic
    try:
        config_model.from_dict(config, plugin_name=transform_type)
        return []  # Valid
    except PydanticValidationError as e:
        return _extract_errors(e)
    except PluginConfigError as e:
        return _extract_wrapped_plugin_config_error(e, config)


def validate_sink_config(
    sink_type: str,
    config: dict[str, Any],
) -> list[ValidationError]:
    """Validate sink plugin configuration.

    Args:
        sink_type: Plugin type name (e.g., "csv", "json")
        config: Plugin configuration dict

    Returns:
        List of validation errors (empty if valid)
    """
    # Get config model for sink type
    config_model = get_sink_config_model(sink_type)

    if config_model is None:
        return []  # No validation needed

    # Validate using Pydantic
    try:
        config_model.from_dict(config, plugin_name=sink_type)
        return []  # Valid
    except PydanticValidationError as e:
        return _extract_errors(e)
    except PluginConfigError as e:
        return _extract_wrapped_plugin_config_error(e, config)


def validate_schema_config(
    schema_config: dict[str, Any],
) -> list[ValidationError]:
    """Validate schema configuration independently of plugin.

    Args:
        schema_config: Schema configuration dict (contents of 'schema' key)

    Returns:
        List of validation errors (empty if valid)
    """
    from elspeth.contracts.schema import SchemaConfig

    try:
        SchemaConfig.from_dict(schema_config)
        return []
    except ValueError as e:
        return [
            ValidationError(
                field="schema",
                message=str(e),
                value=schema_config,
            )
        ]


def _extract_wrapped_plugin_config_error(
    error: PluginConfigError,
    config: dict[str, object],
) -> list[ValidationError]:
    """Convert wrapped PluginConfigError causes into structured errors.

    PluginConfig.from_dict() wraps:
    - PydanticValidationError for model-level validation failures
    - ValueError for schema parsing failures before model validation
    """
    cause = error.__cause__

    if cause is None:
        return [ValidationError(field="config", message=str(error), value=config)]

    if type(cause) is PydanticValidationError:
        return _extract_errors(cause)

    if type(cause) is ValueError:
        if "schema" in config:
            return [ValidationError(field="schema", message=str(cause), value=config["schema"])]
        return [ValidationError(field="config", message=str(cause), value=config)]

    raise error from cause


def get_transform_config_model(
    transform_type: str,
    config: dict[str, Any] | None = None,
) -> type[PluginConfigProtocol] | None:
    """Get Pydantic config model for transform type.

    Resolves the plugin class via pluggy discovery (PluginManager), then
    calls get_config_model(config) on it. The config parameter enables
    provider dispatch (e.g. LLMTransform selects provider-specific model).

    Args:
        transform_type: Plugin type name
        config: Plugin configuration dict (needed for provider dispatch on "llm")

    Returns:
        Config model class for the transform type

    Raises:
        UnknownPluginTypeError: If transform_type is not a registered plugin.
    """
    from elspeth.plugins.infrastructure.manager import PluginNotFoundError, get_shared_plugin_manager

    try:
        plugin_cls = get_shared_plugin_manager().get_transform_by_name(transform_type)
    except PluginNotFoundError as exc:
        raise UnknownPluginTypeError(f"Unknown transform type: {transform_type}") from exc
    return plugin_cls.get_config_model(config)


def get_sink_config_model(sink_type: str) -> type[PluginConfigProtocol] | None:
    """Get Pydantic config model for sink type.

    Resolves the plugin class via pluggy discovery (PluginManager), then
    calls get_config_model() on it.

    Returns:
        Config model class for the sink type

    Raises:
        UnknownPluginTypeError: If sink_type is not a registered plugin.
    """
    from elspeth.plugins.infrastructure.manager import PluginNotFoundError, get_shared_plugin_manager

    try:
        plugin_cls = get_shared_plugin_manager().get_sink_by_name(sink_type)
    except PluginNotFoundError as exc:
        raise UnknownPluginTypeError(f"Unknown sink type: {sink_type}") from exc
    return plugin_cls.get_config_model()


def _extract_errors(
    pydantic_error: PydanticValidationError,
) -> list[ValidationError]:
    """Convert Pydantic errors to structured ValidationError list."""
    errors: list[ValidationError] = []

    for err in pydantic_error.errors():
        # Pydantic error dict has: loc, msg, type, ctx
        # Model-level validators (@model_validator) produce loc=() — empty tuple.
        # Use "__model__" sentinel so the field is never empty.
        field_path = ".".join(str(loc) for loc in err["loc"]) or "__model__"
        message = err["msg"]

        # Pydantic error dict includes failing input value.
        value = err["input"]

        errors.append(
            ValidationError(
                field=field_path,
                message=message,
                value=value,
            )
        )

    return errors
