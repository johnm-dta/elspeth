"""Factory for creating Pydantic schemas from configuration.

This module creates runtime Pydantic models based on SchemaConfig,
enabling config-driven schema validation for plugins.

CRITICAL: The `allow_coercion` parameter enforces the three-tier trust model:
- Sources (allow_coercion=True): May coerce "42" -> 42
- Transforms/Sinks (allow_coercion=False): Reject wrong types (upstream bug)
"""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
from pydantic import ConfigDict, create_model, model_validator

from elspeth.contracts import PluginSchema
from elspeth.contracts.schema import FIELD_TYPE_MAP as TYPE_MAP
from elspeth.contracts.schema import FieldDefinition, SchemaConfig

# Type alias for extra field handling modes
ExtraMode = Literal["allow", "forbid"]


def _find_non_finite_value_path(value: Any, path: str = "$") -> str | None:
    """Find the first path containing a non-finite float value."""
    value_type = type(value)

    if value_type is float and not math.isfinite(value):
        return path

    if value_type is dict:
        for key, nested in value.items():
            nested_path = _find_non_finite_value_path(nested, f"{path}.{key!s}")
            if nested_path is not None:
                return nested_path
        return None

    if value_type in {list, tuple}:
        for idx, nested in enumerate(value):
            nested_path = _find_non_finite_value_path(nested, f"{path}[{idx}]")
            if nested_path is not None:
                return nested_path
        return None

    # Catch numpy floating scalars (longdouble, float16, float32, float64, etc.)
    # Use np.isfinite() — math.isfinite(float(value)) overflows for np.longdouble
    # values outside IEEE 754 double range, falsely treating finite values as inf.
    if isinstance(value, np.floating) and not np.isfinite(value):
        return path

    # NumPy arrays: scan for non-finite elements and report a useful path
    if isinstance(value, np.ndarray) and value.size > 0:
        try:
            if np.any(~np.isfinite(value)):
                flat = value.flat
                for idx, elem in enumerate(flat):
                    if isinstance(elem, float | np.floating) and not np.isfinite(elem):
                        indices = np.unravel_index(idx, value.shape)
                        index_str = "][".join(str(i) for i in indices)
                        return f"{path}[{index_str}]"
        except TypeError:
            # np.isfinite raises TypeError for non-numeric dtypes (e.g., strings)
            pass

    return None


def _reject_non_finite_observed_values(data: Any) -> Any:
    """Reject NaN/Infinity in observed schemas at source boundary."""
    offending_path = _find_non_finite_value_path(data)
    if offending_path is not None:
        raise ValueError(f"Non-finite float at {offending_path}. Use null/None for missing values, not NaN/Infinity.")
    return data


class _ObservedPluginSchema(PluginSchema):
    """PluginSchema base for observed mode with non-finite rejection."""

    @model_validator(mode="before")
    @classmethod
    def _validate_non_finite_values(cls, data: Any) -> Any:
        return _reject_non_finite_observed_values(data)


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
    - Observed mode: extra="allow", accepts any fields (no type checking)
    - Fixed mode: extra="forbid", rejects unknown fields
    - Flexible mode: extra="allow", requires specified fields, allows extras

    Examples:
        # Source - coerces external data
        source_schema = create_schema_from_config(config, "CSVRow", allow_coercion=True)

        # Transform - expects clean data from upstream
        transform_schema = create_schema_from_config(config, "Input", allow_coercion=False)
    """
    if config.is_observed:
        # Observed schema - accept anything (no type validation either way)
        return _create_dynamic_schema(name)

    # Explicit schema - fixed or flexible mode
    return _create_explicit_schema(config, name, allow_coercion)


def _create_dynamic_schema(name: str) -> type[PluginSchema]:
    """Create a schema that accepts any fields.

    Note: Dynamic schemas don't do type checking, so coercion is irrelevant.
    """
    return create_model(
        name,
        __base__=_ObservedPluginSchema,
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
    if config.fields is None or config.mode == "observed":
        raise ValueError("_create_explicit_schema requires fields and non-observed mode")

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
    extra_mode: ExtraMode = "allow" if config.mode == "flexible" else "forbid"

    # Coercion control: strict=True means NO coercion (Pydantic's semantics)
    # allow_coercion=True  -> strict=False (coerce)
    # allow_coercion=False -> strict=True  (reject wrong types)
    use_strict = not allow_coercion

    # At source boundary (allow_coercion=True), use _ObservedPluginSchema base
    # to reject NaN/Infinity in 'any' fields and flexible-mode extras.
    # FiniteFloat handles typed float fields; the model validator catches the rest.
    base_class = _ObservedPluginSchema if allow_coercion else PluginSchema

    return create_model(
        name,
        __base__=base_class,
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
