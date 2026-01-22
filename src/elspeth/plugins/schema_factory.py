"""Factory for creating Pydantic schemas from configuration.

This module creates runtime Pydantic models based on SchemaConfig,
enabling config-driven schema validation for plugins.

CRITICAL: The `allow_coercion` parameter enforces the three-tier trust model:
- Sources (allow_coercion=True): May coerce "42" -> 42
- Transforms/Sinks (allow_coercion=False): Reject wrong types (upstream bug)
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, create_model

from elspeth.contracts import PluginSchema
from elspeth.contracts.schema import FieldDefinition, SchemaConfig

# Type alias for extra field handling modes
ExtraMode = Literal["allow", "forbid"]

# Finite float type that rejects NaN and Infinity at the source boundary.
# Per CLAUDE.md Three-Tier Trust Model and canonical.py policy:
# - NaN/Infinity cannot be represented in RFC 8785 canonical JSON
# - They must be rejected at the source boundary (Tier 3), not crash downstream
# - Use None for intentional missing values, not NaN
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

# Python type mapping for schema field types
# NOTE: float uses FiniteFloat to reject NaN/Infinity at source boundary
TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": FiniteFloat,  # type: ignore[dict-item]
    "bool": bool,
    "any": Any,
}


def create_schema_from_config(
    config: SchemaConfig,
    name: str,
    allow_coercion: bool = True,
) -> type[PluginSchema]:
    """Create a Pydantic schema class from configuration.

    Args:
        config: Schema configuration specifying fields and mode
        name: Name for the generated schema class
        allow_coercion: If True, coerce types (e.g., "42" -> 42). Default True.
            - Sources should use True (normalize external data)
            - Transforms/Sinks should use False (wrong types = upstream bug)

    Returns:
        A PluginSchema subclass with the specified fields and validation

    The generated schema:
    - Dynamic mode: extra="allow", accepts any fields (no type checking)
    - Strict mode: extra="forbid", rejects unknown fields
    - Free mode: extra="allow", requires specified fields, allows extras

    Examples:
        # Source - coerces external data
        source_schema = create_schema_from_config(config, "CSVRow", allow_coercion=True)

        # Transform - expects clean data from upstream
        transform_schema = create_schema_from_config(config, "Input", allow_coercion=False)
    """
    if config.is_dynamic:
        # Dynamic schema - accept anything (no type validation either way)
        return _create_dynamic_schema(name)

    # Explicit schema - strict or free mode
    return _create_explicit_schema(config, name, allow_coercion)


def _create_dynamic_schema(name: str) -> type[PluginSchema]:
    """Create a schema that accepts any fields.

    Note: Dynamic schemas don't do type checking, so coercion is irrelevant.
    """
    return create_model(
        name,
        __base__=PluginSchema,
        __module__=__name__,
        __config__=ConfigDict(
            extra="allow",
            # No strict setting needed - no fields to validate types against
        ),
    )


def _create_explicit_schema(
    config: SchemaConfig,
    name: str,
    allow_coercion: bool,
) -> type[PluginSchema]:
    """Create a schema with explicit field definitions."""
    assert config.fields is not None
    assert config.mode is not None

    # Build field definitions for create_model
    # Format: field_name=(type, default) or field_name=(type, ...)
    field_definitions: dict[str, Any] = {}

    for field_def in config.fields:
        python_type = _get_python_type(field_def)

        if field_def.required:
            # Required field - use ... (Ellipsis) as default
            field_definitions[field_def.name] = (python_type, ...)
        else:
            # Optional field - default to None
            field_definitions[field_def.name] = (python_type, None)

    # Determine extra field handling
    extra_mode: ExtraMode = "allow" if config.mode == "free" else "forbid"

    # Coercion control: strict=True means NO coercion (Pydantic's semantics)
    # allow_coercion=True  -> strict=False (coerce)
    # allow_coercion=False -> strict=True  (reject wrong types)
    use_strict = not allow_coercion

    return create_model(
        name,
        __base__=PluginSchema,
        __module__=__name__,
        __config__=ConfigDict(
            extra=extra_mode,
            strict=use_strict,
        ),
        **field_definitions,
    )


def _get_python_type(field_def: FieldDefinition) -> Any:
    """Convert field definition to Python type annotation.

    For optional fields, returns a Union type (base_type | None).
    For required fields, returns the base type directly.

    Returns Any to satisfy mypy - the actual return is a type or UnionType.
    """
    base_type = TYPE_MAP[field_def.field_type]

    if field_def.required:
        return base_type
    else:
        # Optional: allow None
        return base_type | None
