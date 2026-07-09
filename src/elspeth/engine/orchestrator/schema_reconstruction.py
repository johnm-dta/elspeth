"""Resume-time reconstruction of source schemas from stored JSON Schema."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

from elspeth.contracts.errors import AuditIntegrityError


def reconstruct_schema_from_json(schema_dict: Mapping[str, object]) -> type:
    """Reconstruct Pydantic schema class from JSON schema dict.

    Handles complete Pydantic JSON schema including:
    - Primitive types: string, integer, number, boolean
    - datetime: string with format="date-time"
    - Decimal: anyOf with number/string (for precision preservation)
    - Arrays: type="array" with items schema
    - Nested objects: type="object" with properties schema

    Args:
        schema_dict: Pydantic JSON schema dict (from model_json_schema())

    Returns:
        Dynamically created Pydantic model class

    Raises:
        AuditIntegrityError: If schema is malformed, empty, or contains unsupported types
            (stored schema data is Tier 1 - our data from the Landscape DB)
    """
    from pydantic import ConfigDict, create_model

    from elspeth.contracts import PluginSchema

    # Extract field definitions from Pydantic JSON schema
    # This is OUR data (from Landscape DB) - crash if malformed
    if "properties" not in schema_dict:
        raise AuditIntegrityError(
            "Resume failed: Schema JSON has no 'properties' field. This indicates a malformed schema. Cannot reconstruct types."
        )
    properties = cast(Mapping[str, object], schema_dict["properties"])

    # Handle observed/dynamic schemas: empty properties with additionalProperties=true
    # This is the normal JSON schema output for schema.mode=observed (dynamic schemas)
    # See schema_factory._create_dynamic_schema for the creation side
    if not properties:
        if "additionalProperties" in schema_dict and schema_dict["additionalProperties"] is True:
            # Dynamic schema - accepts any fields, no fixed properties
            return create_model(
                "RestoredDynamicSchema",
                __base__=PluginSchema,
                __config__=ConfigDict(extra="allow"),
            )
        # Empty properties WITHOUT additionalProperties=true is genuinely malformed
        raise AuditIntegrityError(
            "Resume failed: Schema has zero fields defined and additionalProperties is not true. "
            "Cannot resume with empty fixed schema - this would silently discard all row data. "
            "For dynamic schemas, additionalProperties must be true."
        )

    # "required" is optional in JSON Schema spec - empty list is valid default
    required_fields = set(cast(list[str], schema_dict["required"])) if "required" in schema_dict else set()

    # Resolve top-level fields recursively, preserving array item types and
    # nested object property schemas.
    return _create_schema_model(
        model_name="RestoredSourceSchema",
        properties=properties,
        required_fields=required_fields,
        schema_defs=cast(Mapping[str, object], schema_dict["$defs"]) if "$defs" in schema_dict else None,
        create_model=create_model,
        schema_base=PluginSchema,
    )


def _create_schema_model(
    model_name: str,
    properties: Mapping[str, object],
    required_fields: set[str],
    *,
    schema_defs: Mapping[str, object] | None,
    create_model: Any,
    schema_base: Any,
) -> type:
    """Create a Pydantic model from JSON schema properties."""
    field_definitions: dict[str, Any] = {}

    for field_name, raw_field_info in properties.items():
        field_info = cast(Mapping[str, object], raw_field_info)
        field_type = _json_schema_to_python_type(
            field_name,
            field_info,
            schema_defs=schema_defs,
            create_model=create_model,
            schema_base=schema_base,
        )
        field_definitions[field_name] = (field_type, ... if field_name in required_fields else None)

    return cast(type, create_model(model_name, __base__=schema_base, **field_definitions))


def _model_name_for_field(field_name: str) -> str:
    """Build a deterministic nested model name from a field name."""
    tokens = re.findall(r"[A-Za-z0-9]+", field_name)
    if not tokens:
        return "RestoredNestedSchema"
    title_cased = "".join(token[:1].upper() + token[1:] for token in tokens)
    if title_cased[0].isdigit():
        title_cased = f"Field{title_cased}"
    return f"Restored{title_cased}Schema"


def _json_schema_to_python_type(
    field_name: str,
    field_info: Mapping[str, object],
    *,
    schema_defs: Mapping[str, object] | None = None,
    create_model: Any | None = None,
    schema_base: Any | None = None,
) -> Any:
    """Map Pydantic JSON schema field to Python type.

    Handles Pydantic's type mapping including special cases:
    - datetime: {"type": "string", "format": "date-time"}
    - date: {"type": "string", "format": "date"}
    - time: {"type": "string", "format": "time"}
    - timedelta: {"type": "string", "format": "duration"}
    - UUID: {"type": "string", "format": "uuid"}
    - Decimal: {"anyOf": [{"type": "number"}, {"type": "string"}]}
    - Nullable: {"anyOf": [{"type": "T"}, {"type": "null"}]} -> T
    - Nullable ref: {"anyOf": [{"$ref": "#/$defs/M"}, {"type": "null"}]} -> M
    - list[T]: {"type": "array", "items": {...}}
    - dict: {"type": "object"} without properties

    Args:
        field_name: Field name (for error messages)
        field_info: JSON schema field definition

    Returns:
        Python type for Pydantic field

    Raises:
        AuditIntegrityError: If field type is not supported (prevents silent degradation)
    """
    from datetime import date, datetime, time, timedelta
    from decimal import Decimal
    from uuid import UUID

    # Handle anyOf patterns FIRST (before checking for "type" key)
    # anyOf is used for: Decimal, nullable types (T | None), nullable Decimal
    if "anyOf" in field_info:
        any_of_items = cast(list[Mapping[str, object]], field_info["anyOf"])
        type_strs = {cast(str, item["type"]) for item in any_of_items if "type" in item}
        has_null = "null" in type_strs
        non_null_items = [item for item in any_of_items if "type" not in item or item["type"] != "null"]
        non_null_types = {cast(str, item["type"]) for item in non_null_items if "type" in item}

        # Pattern 1: Decimal or Optional[Decimal]
        # Decimal: {"anyOf": [{"type": "number"}, {"type": "string"}]}
        # Optional[Decimal]: {"anyOf": [{"type": "number"}, {"type": "string"}, {"type": "null"}]}
        if {"number", "string"}.issubset(non_null_types) and len(non_null_items) == 2:
            return Decimal | None if has_null else Decimal

        # Pattern 2: Nullable - {"anyOf": [{"type": "T", ...}, {"type": "null"}]}
        #   or with $ref:  {"anyOf": [{"$ref": "#/$defs/M"}, {"type": "null"}]}
        # Extract the non-null type and recursively resolve it
        if has_null and len(non_null_items) == 1:
            # Recursively resolve the non-null type, then wrap as Optional.
            # Returning T | None (not bare T) is critical: Pydantic model types
            # reject None unless the type annotation explicitly includes it.
            inner_type = _json_schema_to_python_type(
                field_name,
                non_null_items[0],
                schema_defs=schema_defs,
                create_model=create_model,
                schema_base=schema_base,
            )
            return inner_type | None

        # Unsupported anyOf pattern (e.g., Union[str, int] without null)
        raise AuditIntegrityError(
            f"Resume failed: Field '{field_name}' has unsupported anyOf pattern. "
            f"Supported patterns: Decimal (number|string), nullable (T|null), nullable Decimal (number|string|null). "
            f"Schema definition: {field_info}. "
            f"This is a bug in schema reconstruction - please report this."
        )

    # Resolve local references in Pydantic schemas (e.g., "#/$defs/NestedModel")
    if "$ref" in field_info:
        ref = cast(str, field_info["$ref"])
        ref_prefix = "#/$defs/"
        if not ref.startswith(ref_prefix):
            raise AuditIntegrityError(
                f"Resume failed: Field '{field_name}' has unsupported $ref '{ref}'. Only local refs under '#/$defs/' are supported."
            )
        if schema_defs is None:
            raise AuditIntegrityError(f"Resume failed: Field '{field_name}' references '{ref}' but schema has no $defs section.")
        def_name = ref[len(ref_prefix) :]
        if def_name not in schema_defs:
            raise AuditIntegrityError(f"Resume failed: Field '{field_name}' references missing schema def '{def_name}'.")
        return _json_schema_to_python_type(
            field_name,
            cast(Mapping[str, object], schema_defs[def_name]),
            schema_defs=schema_defs,
            create_model=create_model,
            schema_base=schema_base,
        )

    # Get basic type - required for all non-anyOf fields
    if "type" not in field_info:
        raise AuditIntegrityError(
            f"Resume failed: Field '{field_name}' has no 'type' in schema. "
            f"Schema definition: {field_info}. "
            f"Cannot determine Python type for field."
        )
    field_type_str = cast(str, field_info["type"])

    # Handle string types with format specifiers
    if field_type_str == "string":
        fmt = None
        if "format" in field_info:
            fmt = field_info["format"]
        if fmt == "date-time":
            return datetime
        if fmt == "date":
            return date
        if fmt == "time":
            return time
        if fmt == "duration":
            return timedelta
        if fmt == "uuid":
            return UUID
        # Plain string (no format or unknown format)
        return str

    # Handle array types
    if field_type_str == "array":
        # "items" is optional in JSON Schema arrays. When present, recursively
        # restore item type fidelity (e.g., list[int], list[NestedSchema]).
        if "items" in field_info:
            item_info = cast(Mapping[str, object], field_info["items"])
            item_type = _json_schema_to_python_type(
                f"{field_name}_item",
                item_info,
                schema_defs=schema_defs,
                create_model=create_model,
                schema_base=schema_base,
            )
            return list.__class_getitem__(item_type)
        return list

    # Handle nested object types
    if field_type_str == "object":
        # Typed nested object: recursively create a nested schema model.
        if "properties" in field_info:
            properties = cast(Mapping[str, object], field_info["properties"])
        else:
            properties = None
        if properties:
            nested_required = set(cast(list[str], field_info["required"])) if "required" in field_info else set()
            nested_name = _model_name_for_field(field_name)
            return _create_schema_model(
                model_name=nested_name,
                properties=properties,
                required_fields=nested_required,
                schema_defs=schema_defs,
                create_model=create_model,
                schema_base=schema_base,
            )

        # Map additionalProperties schemas when present (e.g., dict[str, int]).
        if "additionalProperties" in field_info:
            additional = field_info["additionalProperties"]
            if additional is True:
                return dict[str, Any]
            if type(additional) is dict:
                value_type = _json_schema_to_python_type(
                    f"{field_name}_value",
                    cast(Mapping[str, object], additional),
                    schema_defs=schema_defs,
                    create_model=create_model,
                    schema_base=schema_base,
                )
                return dict.__class_getitem__((str, value_type))
            if additional is False:
                return dict
            raise AuditIntegrityError(f"Resume failed: Field '{field_name}' has invalid additionalProperties value: {additional!r}.")

        # Generic dict (no specific structure)
        return dict

    # Handle other primitive types
    primitive_type_map = {
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    if field_type_str in primitive_type_map:
        return primitive_type_map[field_type_str]

    # Unknown type - CRASH instead of silent degradation
    raise AuditIntegrityError(
        f"Resume failed: Field '{field_name}' has unsupported type '{field_type_str}'. "
        f"Supported types: string, integer, number, boolean, date-time, date, time, "
        f"duration, uuid, Decimal, array, object. "
        f"Schema definition: {field_info}. "
        f"This is a bug in schema reconstruction - please report this."
    )
