"""Schema configuration types for config-driven plugin schemas.

This module provides the configuration types that allow users to specify
plugin schemas in their pipeline configuration files. Schemas can be:

1. Observed: Accept any fields, types inferred from data (logged for audit)
2. Fixed: Accept exactly the specified fields (no more, no less)
3. Flexible: Accept at least the specified fields (extras allowed)

Example YAML:
    plugins:
      csv_source:
        path: data.csv
        schema:
          mode: fixed
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

        # FIELD_PATTERN regex guarantees field_type is one of the supported literals
        # This check is defense-in-depth in case regex is modified incorrectly
        if field_type not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported field type: {field_type}")
        # Cast needed because mypy can't infer type narrowing from set membership
        typed_field: Literal["str", "int", "float", "bool", "any"] = field_type  # type: ignore[assignment]  # narrowed by SUPPORTED_TYPES membership check above

        return cls(
            name=name,
            field_type=typed_field,
            required=optional_marker is None,
        )

    def to_dict(self) -> dict[str, str | bool]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.field_type,
            "required": self.required,
        }


def _parse_field_names_list(value: Any, field_name: str) -> tuple[str, ...] | None:
    """Parse a list of field names for guaranteed_fields/required_fields.

    Args:
        value: Raw value from config (should be None or list of strings)
        field_name: Name of the config field for error messages

    Returns:
        Tuple of field names, or None if value is None

    Raises:
        ValueError: If value is not None, a list, or contains non-strings
    """
    if value is None:
        return None

    if not isinstance(value, list):
        raise ValueError(f"'{field_name}' must be a list of field names, got {type(value).__name__}")

    if len(value) == 0:
        return None  # Empty list is treated as unspecified

    result: list[str] = []
    for i, name in enumerate(value):
        if not isinstance(name, str):
            raise ValueError(f"'{field_name}[{i}]' must be a string, got {type(name).__name__}")
        name = name.strip()
        if not name:
            raise ValueError(f"'{field_name}[{i}]' cannot be empty or whitespace-only")
        if not name.isidentifier():
            raise ValueError(
                f"'{field_name}[{i}]' must be a valid Python identifier, got '{name}'. "
                f"Field names must contain only letters, digits, and underscores, "
                f"and cannot start with a digit."
            )
        result.append(name)

    # Check for duplicates
    if len(result) != len(set(result)):
        duplicates = sorted({n for n in result if result.count(n) > 1})
        raise ValueError(f"Duplicate field names in '{field_name}': {', '.join(duplicates)}")

    return tuple(result)


def _validate_contract_fields_subset(
    contract_fields: tuple[str, ...] | None,
    field_name: str,
    declared_names: frozenset[str],
) -> None:
    """Validate that contract fields are subsets of declared fields.

    For explicit schemas (mode=fixed/flexible), contract fields like guaranteed_fields,
    required_fields, and audit_fields must reference fields that actually exist in
    the declared schema. Typos would otherwise create false audit claims.

    Args:
        contract_fields: The contract field tuple to validate, or None
        field_name: Name of the field for error messages (e.g., "guaranteed_fields")
        declared_names: Set of declared field names to validate against

    Raises:
        ValueError: If any contract field is not in declared_names
    """
    if contract_fields is None:
        return

    undefined = set(contract_fields) - declared_names
    if undefined:
        raise ValueError(
            f"'{field_name}' contains fields not declared in schema: "
            f"{', '.join(sorted(undefined))}. "
            f"Declared fields are: {', '.join(sorted(declared_names))}."
        )


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
        # Handle to_dict() round-trip format: {"name": "x", "type": "str", "required": true}
        # This enables SchemaConfig.from_dict(schema_config.to_dict()) to work.
        if "name" in spec and "type" in spec:
            name = spec["name"]
            type_spec = spec["type"]
            optional = not spec.get("required", True)
            return f"{name}: {type_spec}{'?' if optional else ''}"

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

    A schema can be either observed (accept anything, infer types from data)
    or explicit (validate against specified fields).

    Schema Contracts (for DAG validation):
        - guaranteed_fields: Fields the producer GUARANTEES will exist AND are
          part of the stable API contract. Downstream can safely depend on these.
        - required_fields: Fields the consumer REQUIRES in input.
        - audit_fields: Fields that exist in output but are NOT part of the
          stability contract. These are for audit trail reconstruction and may
          change between versions. DAG validation does NOT enforce these.

    For explicit schemas (mode='fixed' or 'flexible'), the declared fields are
    implicitly guaranteed. Use guaranteed_fields/required_fields to express
    contracts for observed schemas that still have known field requirements.

    Example YAML for an observed schema with explicit contracts:
        schema:
          mode: observed
          guaranteed_fields: [customer_id, timestamp]  # Producer guarantees these

        schema:
          mode: observed
          required_fields: [customer_id, amount]  # Consumer requires these

        schema:
          mode: observed
          guaranteed_fields: [response, response_usage]  # Stable API
          audit_fields: [response_template_hash]  # May change between versions

    Attributes:
        mode: "fixed" (exact fields), "flexible" (at least these), or "observed" (infer from data)
        fields: List of FieldDefinitions, or None if observed
        is_observed: True if schema types are inferred from data
        guaranteed_fields: Field names the producer guarantees (for observed schemas)
        required_fields: Field names the consumer requires (for observed schemas)
        audit_fields: Field names that exist but are not part of stability contract
    """

    mode: Literal["fixed", "flexible", "observed"]
    fields: tuple[FieldDefinition, ...] | None
    guaranteed_fields: tuple[str, ...] | None = None
    required_fields: tuple[str, ...] | None = None
    audit_fields: tuple[str, ...] | None = None

    @property
    def is_observed(self) -> bool:
        """True if schema types are inferred from data (OBSERVED mode)."""
        return self.mode == "observed"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> SchemaConfig:
        """Parse schema configuration from dict.

        Args:
            config: Dict with 'mode' key (required). For fixed/flexible modes,
                   'fields' is also required. Optional: 'guaranteed_fields',
                   'required_fields', 'audit_fields' for contracts.

        Returns:
            SchemaConfig instance

        Raises:
            ValueError: If config is invalid
        """
        # Parse contract fields (valid for all schema modes)
        guaranteed_fields = _parse_field_names_list(config.get("guaranteed_fields"), "guaranteed_fields")
        required_fields = _parse_field_names_list(config.get("required_fields"), "required_fields")
        audit_fields = _parse_field_names_list(config.get("audit_fields"), "audit_fields")

        # Get mode - required for all schemas
        mode = config.get("mode")

        # Handle observed schema (mode: observed)
        if mode == "observed":
            # Observed schemas may have optional fields for contracts (guaranteed_fields)
            # but don't define explicit field types
            fields_value = config.get("fields")
            # Allow only None or [] for backwards compatibility; any other value
            # is an explicit schema declaration and must be rejected.
            if fields_value is not None and fields_value != []:
                raise ValueError(
                    "Observed schemas (mode: observed) cannot have explicit field definitions. "
                    "Use guaranteed_fields/required_fields for contracts instead."
                )
            return cls(
                mode="observed",
                fields=None,
                guaranteed_fields=guaranteed_fields,
                required_fields=required_fields,
                audit_fields=audit_fields,
            )

        # Explicit schema - requires mode and fields
        if mode is None:
            raise ValueError(
                "'mode' key is required in schema config. "
                "Use 'mode: observed' (infer types from data), "
                "'mode: fixed' (exactly these fields), or "
                "'mode: flexible' (at least these fields)."
            )

        if mode not in ("fixed", "flexible"):
            raise ValueError(f"Invalid schema mode '{mode}'. Expected 'fixed', 'flexible', or 'observed'.")

        # Explicit schemas require fields
        if "fields" not in config:
            raise ValueError(
                f"'fields' key is required for mode '{mode}'. "
                f"Provide explicit field definitions, or use 'mode: observed' to infer types from data."
            )

        fields_value = config["fields"]

        # Parse field list
        if not isinstance(fields_value, list):
            raise ValueError(f"Schema fields must be a list, got {type(fields_value).__name__}")

        if len(fields_value) == 0:
            raise ValueError("Schema must define at least one field for fixed/flexible modes. Use 'mode: observed' to accept any fields.")

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

        # Validate contract fields are subsets of declared fields
        # For explicit schemas, typos in guaranteed/required/audit_fields would create
        # false audit claims - catch them at config load time
        declared_names = frozenset(names)
        _validate_contract_fields_subset(guaranteed_fields, "guaranteed_fields", declared_names)
        _validate_contract_fields_subset(required_fields, "required_fields", declared_names)
        _validate_contract_fields_subset(audit_fields, "audit_fields", declared_names)

        return cls(
            mode=mode,
            fields=parsed_fields,
            guaranteed_fields=guaranteed_fields,
            required_fields=required_fields,
            audit_fields=audit_fields,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for audit logging.

        Contract fields (guaranteed_fields, required_fields) are included
        when present, omitted when None for cleaner audit output.
        """
        result: dict[str, Any]
        if self.is_observed:
            result = {"mode": "observed", "fields": None}
        else:
            result = {
                "mode": self.mode,
                "fields": [f.to_dict() for f in self.fields] if self.fields else [],
            }

        # Include contract fields only when specified (cleaner audit output)
        if self.guaranteed_fields is not None:
            result["guaranteed_fields"] = list(self.guaranteed_fields)
        if self.required_fields is not None:
            result["required_fields"] = list(self.required_fields)
        if self.audit_fields is not None:
            result["audit_fields"] = list(self.audit_fields)

        return result

    @property
    def allows_extra_fields(self) -> bool:
        """Whether extra fields beyond schema are allowed."""
        return self.is_observed or self.mode == "flexible"

    def get_effective_guaranteed_fields(self) -> frozenset[str]:
        """Get all fields this schema guarantees will exist.

        For explicit schemas (fixed/flexible mode), REQUIRED declared fields are
        implicitly guaranteed. Optional fields (marked with ?) are NOT guaranteed
        since producers are allowed to omit them. For observed schemas, only
        explicit guaranteed_fields are considered.

        Returns:
            Frozenset of field names that are guaranteed to exist.
        """
        # Start with explicit guaranteed_fields if any
        explicit = frozenset(self.guaranteed_fields) if self.guaranteed_fields else frozenset()

        # For explicit schemas, only REQUIRED fields are implicitly guaranteed
        # Optional fields (f.required == False) may be missing, so they're not guaranteed
        if self.fields is not None:
            declared_required = frozenset(f.name for f in self.fields if f.required)
            return explicit | declared_required

        return explicit

    def get_effective_required_fields(self) -> frozenset[str]:
        """Get all fields this schema requires in input.

        For explicit schemas (fixed/flexible mode), required declared fields
        (not optional) are implicitly required. For observed schemas, only
        explicit required_fields are considered.

        Returns:
            Frozenset of field names that are required.
        """
        # Start with explicit required_fields if any
        explicit = frozenset(self.required_fields) if self.required_fields else frozenset()

        # For explicit schemas, required declared fields are implicitly required
        if self.fields is not None:
            declared_required = frozenset(f.name for f in self.fields if f.required)
            return explicit | declared_required

        return explicit
