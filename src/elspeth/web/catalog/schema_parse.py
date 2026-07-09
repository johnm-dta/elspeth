"""Typed views over the JSON-Schema documents ``model_json_schema()`` emits.

The *values* in these documents are first-party (our own plugin config models
produced them — system code), but the *presence* of individual keys is
governed by the JSON Schema specification, which we do not author:
``required`` is omitted when no field is mandatory, ``default`` is omitted
when a field has none, top-level ``type`` is absent for ``anyOf`` properties,
``$ref`` is absent for inline ``oneOf`` entries. Parsing each fragment into
one of these permissive models makes the spec-optional keys explicit typed
fields with honest defaults (absent ``required`` -> empty list, absent
``default`` -> ``None``) so traversals access typed attributes directly
instead of guessing with ``.get(key, default)``. A ``ValidationError`` here
means our own schema generation produced a structurally impossible document —
a first-party bug — so it is intentionally left to propagate (crash), never
swallowed.

Shared by the catalog service (plugin card summaries) and the composer
discovery availability gates, which must agree on what a plugin schema
requires — most critically the required *secret* fields, where silent
under-detection would fail open.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from elspeth.core.secrets import is_secret_field

# JSON-Schema $ref prefix for local $defs used by Pydantic discriminated unions.
DEFS_REF_PREFIX = "#/$defs/"


class SchemaProperty(BaseModel):
    """One entry under a JSON-Schema ``properties`` map."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    type: str | None = None
    description: str | None = None
    default: Any = None
    any_of: list[SchemaProperty] = Field(default_factory=list, alias="anyOf")


class SchemaObject(BaseModel):
    """A JSON-Schema object document (top-level model or ``$defs`` variant)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    properties: dict[str, SchemaProperty] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class OneOfEntry(BaseModel):
    """One entry of a discriminated-union ``oneOf`` list.

    JSON Schema permits each entry to be either a ``$ref`` into ``$defs`` or
    an inline object schema; an inline entry simply omits ``$ref``, so the
    field defaults to the empty string and the caller skips it.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ref: str = Field(default="", alias="$ref")


def required_secret_fields_from_json_schema(schema: dict[str, Any]) -> tuple[str, ...]:
    """Required secret-field names a plugin config JSON schema declares.

    Secret-requirement computation is a security gate: a structurally
    impossible schema must crash (first-party bug) rather than silently
    under-detect required secret fields, so both branches reify through the
    typed models above.
    """
    if "oneOf" in schema and "$defs" in schema:
        return common_required_secret_fields_from_discriminated(schema)

    parsed = SchemaObject.model_validate(schema)
    return tuple(field_name for field_name in parsed.required if is_secret_field(field_name))


def common_required_secret_fields_from_discriminated(schema: dict[str, Any]) -> tuple[str, ...]:
    """Secret fields required by EVERY ``oneOf`` variant of a discriminated union."""
    defs: dict[str, dict[str, Any]] = schema["$defs"]
    variant_required_secret_fields: list[set[str]] = []
    ordered_first_variant: list[str] = []
    for raw_entry in schema["oneOf"]:
        entry = OneOfEntry.model_validate(raw_entry)
        if not entry.ref.startswith(DEFS_REF_PREFIX):
            continue
        variant = SchemaObject.model_validate(defs[entry.ref[len(DEFS_REF_PREFIX) :]])
        fields = {field_name for field_name in variant.required if is_secret_field(field_name)}
        if not variant_required_secret_fields:
            ordered_first_variant = [field_name for field_name in variant.required if field_name in fields]
        variant_required_secret_fields.append(fields)

    if not variant_required_secret_fields:
        return ()
    common = set.intersection(*variant_required_secret_fields)
    return tuple(field_name for field_name in ordered_first_variant if field_name in common)
