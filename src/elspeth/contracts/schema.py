"""Schema configuration types for config-driven plugin schemas.

This module provides the configuration types that allow users to specify
plugin schemas in their pipeline configuration files. Schemas can be:

1. Dynamic: Accept any fields (logged for audit)
2. Strict: Accept exactly the specified fields (no more, no less)
3. Free: Accept at least the specified fields (extras allowed)

Example YAML:
    plugins:
      csv_source:
        path: data.csv
        schema:
          mode: strict
          fields:
            - "id: int"
            - "name: str"
            - "score: float?"  # Optional field

Note: Field specs must be quoted strings in YAML. Unquoted `- id: int` is parsed
as a dict `{id: int}` by YAML loaders, not as the string `"id: int"`. Both formats
are accepted, but quoted strings are recommended for clarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

# Supported field types for schema definitions
SUPPORTED_TYPES = frozenset({"str", "int", "float", "bool", "any"})

# Pattern: "field_name: type" or "field_name: type?"
# Field names must be valid Python identifiers (letters, digits, underscores)
# NOTE: Hyphens and dots are NOT supported in field names
#   - "user-id: int"  -> INVALID (use "user_id: int")
#   - "data.field: str" -> INVALID (use "data_field: str")
# This is intentional: field names map to Python attributes/dict keys
FIELD_PATTERN = re.compile(r"^(\w+):\s*(str|int|float|bool|any)(\?)?$")


@dataclass(frozen=True)
class FieldDefinition:
    """Definition of a single field in a schema.

    Attributes:
        name: Field name (must be valid Python identifier)
        field_type: One of: str, int, float, bool, any
        required: If False, field can be missing or None
    """

    name: str
    field_type: Literal["str", "int", "float", "bool", "any"]
    required: bool = True

    @classmethod
    def parse(cls, spec: str) -> FieldDefinition:
        """Parse a field specification string.

        Args:
            spec: Field spec like "name: str" or "score: float?"

        Returns:
            FieldDefinition instance

        Raises:
            ValueError: If spec is malformed or type is unknown
        """
        spec = spec.strip()
        match = FIELD_PATTERN.match(spec)

        if not match:
            # Check if it's a type issue vs format issue
            if ":" in spec:
                parts = spec.split(":", 1)
                name_part = parts[0].strip()
                type_part = parts[1].strip().rstrip("?")

                # Check for invalid type
                if type_part not in SUPPORTED_TYPES:
                    raise ValueError(
                        f"Unknown type '{type_part}' in field spec '{spec}'. Supported types: {', '.join(sorted(SUPPORTED_TYPES))}"
                    )

                # Check for invalid field name (hyphens, dots, etc.)
                if not name_part.isidentifier():
                    raise ValueError(
                        f"Invalid field name '{name_part}' in field spec '{spec}'. "
                        f"Field names must be valid Python identifiers "
                        f"(letters, digits, underscores only). "
                        f"Use '{name_part.replace('-', '_').replace('.', '_')}' instead."
                    )

            raise ValueError(f"Invalid field spec '{spec}'. Expected format: 'field_name: type' or 'field_name: type?'")

        name, field_type, optional_marker = match.groups()

        # Validate that name is a valid Python identifier
        # (regex allows numeric-prefixed names like "123field" which aren't valid)
        if not name.isidentifier():
            raise ValueError(
                f"Invalid field name '{name}' in field spec '{spec}'. "
                f"Field names must be valid Python identifiers "
                f"(cannot start with a digit)."
            )

        return cls(
            name=name,
            field_type=field_type,  # type: ignore[arg-type]
            required=optional_marker is None,
        )

    def to_dict(self) -> dict[str, str | bool]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.field_type,
            "required": self.required,
        }


def _normalize_field_spec(spec: Any, *, index: int) -> str:
    """Normalize a field spec to string form.

    Accepts:
    - String: "field_name: type" or "field_name: type?"
    - Dict (from YAML): {"field_name": "type"} or {"field_name": "type?"}

    Args:
        spec: Field specification (string or single-key dict)
        index: Index in fields list (for error messages)

    Returns:
        Normalized string spec like "field_name: type"

    Raises:
        ValueError: If spec format is invalid
    """
    if isinstance(spec, str):
        return spec

    if isinstance(spec, dict):
        # YAML `- id: int` parses as {"id": "int"}
        if len(spec) != 1:
            raise ValueError(
                f"Field spec at index {index} is a dict with {len(spec)} keys. "
                f"Expected single-key dict like {{'field_name': 'type'}} or a string like 'field_name: type'."
            )
        name, type_spec = next(iter(spec.items()))
        if not isinstance(name, str) or not isinstance(type_spec, str):
            raise ValueError(
                f"Field spec at index {index}: dict keys and values must be strings, "
                f"got {{{type(name).__name__}: {type(type_spec).__name__}}}."
            )
        return f"{name}: {type_spec}"

    raise ValueError(
        f"Field spec at index {index} must be a string like 'field_name: type' "
        f"or a dict like {{'field_name': 'type'}}, got {type(spec).__name__}."
    )


@dataclass(frozen=True)
class SchemaConfig:
    """Configuration for a plugin's data schema.

    A schema can be either dynamic (accept anything) or explicit
    (validate against specified fields).

    Attributes:
        mode: "strict" (exact fields), "free" (at least these), or None (dynamic)
        fields: List of FieldDefinitions, or None if dynamic
        is_dynamic: True if schema accepts any fields
    """

    mode: Literal["strict", "free"] | None
    fields: tuple[FieldDefinition, ...] | None
    is_dynamic: bool

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> SchemaConfig:
        """Parse schema configuration from dict.

        Args:
            config: Dict with 'fields' key (required) and optional 'mode'

        Returns:
            SchemaConfig instance

        Raises:
            ValueError: If config is invalid
        """
        if "fields" not in config:
            raise ValueError("'fields' key is required in schema config. Use 'fields: dynamic' or provide explicit field list.")

        # Handle serialized dynamic schema (mode="dynamic" from to_dict())
        if config.get("mode") == "dynamic":
            return cls(mode=None, fields=None, is_dynamic=True)

        fields_value = config["fields"]

        # Dynamic schema (original input format: fields="dynamic")
        if fields_value == "dynamic":
            return cls(
                mode=None,
                fields=None,
                is_dynamic=True,
            )

        # Explicit schema - requires mode
        if "mode" not in config:
            raise ValueError(
                "'mode' key is required when specifying explicit fields. "
                "Use 'mode: strict' (exactly these fields) or "
                "'mode: free' (at least these fields)."
            )

        mode = config["mode"]
        if mode not in ("strict", "free"):
            raise ValueError(f"Invalid schema mode '{mode}'. Expected 'strict' or 'free'.")

        # Parse field list
        if not isinstance(fields_value, list):
            raise ValueError(f"Schema fields must be a list, got {type(fields_value).__name__}")

        if len(fields_value) == 0:
            raise ValueError("Schema must define at least one field. Use 'fields: dynamic' to accept any fields.")

        # Normalize field specs: accept both string and dict forms
        normalized_specs = []
        for i, f in enumerate(fields_value):
            spec = _normalize_field_spec(f, index=i)
            normalized_specs.append(spec)

        parsed_fields = tuple(FieldDefinition.parse(spec) for spec in normalized_specs)

        # Check for duplicate field names
        names = [f.name for f in parsed_fields]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate field names in schema: {', '.join(sorted(set(duplicates)))}")

        return cls(
            mode=mode,
            fields=parsed_fields,
            is_dynamic=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for audit logging.

        Note: For dynamic schemas, mode is stored as None internally but
        serialized as "dynamic" for clarity in audit logs. This is intentional:
        - Internal: mode=None, is_dynamic=True (distinguishes from explicit modes)
        - Serialized: mode="dynamic" (clear in audit trail)
        """
        if self.is_dynamic:
            return {"mode": "dynamic", "fields": None}
        return {
            "mode": self.mode,
            "fields": [f.to_dict() for f in self.fields] if self.fields else [],
        }

    @property
    def allows_extra_fields(self) -> bool:
        """Whether extra fields beyond schema are allowed."""
        return self.is_dynamic or self.mode == "free"
