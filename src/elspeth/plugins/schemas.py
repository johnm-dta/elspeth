# src/elspeth/plugins/schemas.py
"""Pydantic-based schema system for plugins.

Every plugin declares input and output schemas using Pydantic models.
This enables:
- Runtime validation of row data
- Pipeline validation at config time (Phase 3)
- Documentation generation
- Landscape context recording
"""

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, ValidationError

T = TypeVar("T", bound="PluginSchema")


class PluginSchema(BaseModel):
    """Base class for plugin input/output schemas.

    Plugins define schemas by subclassing:

        class MyInputSchema(PluginSchema):
            temperature: float
            humidity: float

    Features:
    - Extra fields ignored (rows may have more fields than schema requires)
    - Coercive type validation (int->float allowed, strict=False)
    - Easy conversion to/from row dicts
    """

    model_config = ConfigDict(
        extra="ignore",  # Rows may have extra fields
        strict=False,    # Allow coercion (e.g., int -> float)
        frozen=False,    # Allow modification
    )

    def to_row(self) -> dict[str, Any]:
        """Convert schema instance to row dict."""
        return self.model_dump()

    @classmethod
    def from_row(cls: type[T], row: dict[str, Any]) -> T:
        """Create schema instance from row dict.

        Extra fields in row are ignored.
        """
        return cls.model_validate(row)


class SchemaValidationError:
    """A validation error for a specific field."""

    def __init__(self, field: str, message: str, value: Any = None) -> None:
        self.field = field
        self.message = message
        self.value = value

    def __str__(self) -> str:
        return f"{self.field}: {self.message}"

    def __repr__(self) -> str:
        return f"SchemaValidationError({self.field!r}, {self.message!r})"


def validate_row(
    row: dict[str, Any],
    schema: type[PluginSchema],
) -> list[SchemaValidationError]:
    """Validate a row against a schema.

    Args:
        row: Row data to validate
        schema: PluginSchema subclass

    Returns:
        List of validation errors (empty if valid)
    """
    try:
        schema.model_validate(row)
        return []
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            errors.append(SchemaValidationError(
                field=field,
                message=error["msg"],
                value=error.get("input"),
            ))
        return errors


@dataclass
class CompatibilityResult:
    """Result of schema compatibility check."""

    compatible: bool
    missing_fields: list[str] = dataclass_field(default_factory=list)
    type_mismatches: list[tuple[str, str, str]] = dataclass_field(default_factory=list)

    @property
    def error_message(self) -> str | None:
        """Human-readable error message if incompatible."""
        if self.compatible:
            return None

        parts = []
        if self.missing_fields:
            parts.append(f"Missing fields: {', '.join(self.missing_fields)}")
        if self.type_mismatches:
            mismatches = [
                f"{name} (expected {expected}, got {actual})"
                for name, expected, actual in self.type_mismatches
            ]
            parts.append(f"Type mismatches: {', '.join(mismatches)}")

        return "; ".join(parts)


def check_compatibility(
    producer_schema: type[PluginSchema],
    consumer_schema: type[PluginSchema],
) -> CompatibilityResult:
    """Check if producer output is compatible with consumer input.

    Uses Pydantic model_fields metadata for accurate compatibility checking.
    This handles optional fields, unions, constrained types, and defaults.

    Compatibility means:
    - All REQUIRED fields in consumer are provided by producer
    - Fields with defaults in consumer are optional
    - Field types are compatible (exact match or coercible)

    Args:
        producer_schema: Output schema of upstream plugin
        consumer_schema: Input schema of downstream plugin

    Returns:
        CompatibilityResult indicating compatibility and any issues
    """
    # Use Pydantic v2 model_fields for accurate field introspection
    producer_fields = producer_schema.model_fields
    consumer_fields = consumer_schema.model_fields

    missing: list[str] = []
    mismatches: list[tuple[str, str, str]] = []

    for field_name, consumer_field in consumer_fields.items():
        # Check if field is required (no default value)
        is_required = consumer_field.is_required()

        if field_name not in producer_fields:
            # Missing field - only a problem if required
            if is_required:
                missing.append(field_name)
        else:
            producer_field = producer_fields[field_name]
            if not _types_compatible(producer_field.annotation, consumer_field.annotation):
                mismatches.append((
                    field_name,
                    _type_name(consumer_field.annotation),
                    _type_name(producer_field.annotation),
                ))

    compatible = len(missing) == 0 and len(mismatches) == 0

    return CompatibilityResult(
        compatible=compatible,
        missing_fields=missing,
        type_mismatches=mismatches,
    )


def _type_name(t: Any) -> str:
    """Get readable name for a type annotation."""
    if hasattr(t, "__name__"):
        return str(t.__name__)
    return str(t)


def _is_union_type(t: Any) -> bool:
    """Check if type is a Union (typing.Union or types.UnionType)."""
    origin = get_origin(t)
    return origin is Union or isinstance(t, UnionType)


def _types_compatible(actual: Any, expected: Any) -> bool:
    """Check if actual type is compatible with expected type.

    Handles:
    - Exact matches
    - Numeric compatibility (int -> float)
    - Optional[X] on consumer side (producer can send X or X | None)
    - Union types (both typing.Union and X | Y syntax)
    """
    # Exact match
    if actual == expected:
        return True

    # Numeric compatibility (int -> float is OK)
    if expected is float and actual is int:
        return True

    # Handle Optional/Union types (both typing.Union and types.UnionType)
    if _is_union_type(expected):
        expected_args = get_args(expected)
        # Check if actual type matches any of the union members
        if actual in expected_args:
            return True
        # Check if actual is a Union that's a subset
        if _is_union_type(actual):
            actual_args = get_args(actual)
            return all(a in expected_args for a in actual_args)

    return False
