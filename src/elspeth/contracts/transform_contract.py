"""Contract utilities for transform input/output validation.

Transforms have explicit schemas (PluginSchema subclasses) that define
their input and output contracts. This module bridges PluginSchema
(Pydantic) with SchemaContract (frozen dataclass).
"""

from __future__ import annotations

from types import UnionType
from typing import Annotated, Any, Union, cast, get_args, get_origin

from elspeth.contracts.data import PluginSchema
from elspeth.contracts.errors import ContractViolation
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.type_normalization import ALLOWED_CONTRACT_TYPES


def _is_union_type(t: Any) -> bool:
    """Check if type is a Union (typing.Union or types.UnionType)."""
    origin = get_origin(t)
    return origin is Union or isinstance(t, UnionType)


def _unwrap_annotated(annotation: Any) -> Any:
    """Unwrap typing.Annotated recursively to its underlying type."""
    current = annotation
    while get_origin(current) is Annotated:
        args = get_args(current)
        if not args:
            return current
        current = args[0]
    return current


def _get_python_type(annotation: Any) -> type:
    """Extract Python type from type annotation.

    Handles Optional, Union, etc. by taking the first non-None type.
    Unknown types return `object` (the 'any' type in contracts).

    Args:
        annotation: Type annotation from schema

    Returns:
        Python primitive type, or object for unknown types
    """
    unwrapped = _unwrap_annotated(annotation)

    # Handle Optional[X] which is Union[X, None] or X | None
    if _is_union_type(unwrapped):
        # Union type - get first non-None arg
        args = get_args(unwrapped)
        saw_non_none = False
        for arg in args:
            if arg is not type(None):
                saw_non_none = True
                resolved = _get_python_type(arg)
                if resolved is not object:
                    return resolved
        if saw_non_none:
            return object
        return type(None)

    # Simple type - return if in allowed set, or 'object' for unknown types
    if unwrapped in ALLOWED_CONTRACT_TYPES:
        return cast(type, unwrapped)
    return object


def create_output_contract_from_schema(
    schema_class: type[PluginSchema],
) -> SchemaContract:
    """Create SchemaContract from PluginSchema class.

    Extracts field types from schema annotations. The contract is
    always locked since transform schemas are static.

    Args:
        schema_class: PluginSchema subclass

    Returns:
        Locked SchemaContract with declared fields
    """
    # Check if schema allows extra fields
    # NOTE: We control all schemas via PluginSchema base class which sets model_config["extra"].
    # Direct access is correct per Tier 1 trust model - missing key would be our bug.
    extra = schema_class.model_config["extra"]

    if extra == "allow":
        mode = "FLEXIBLE"
    elif extra == "forbid":
        mode = "FIXED"
    else:
        # extra="ignore": Use FLEXIBLE mode to allow extra fields to pass through
        # Sources may load data with extra fields not in schema (e.g., CSV with extra columns)
        # Transforms may receive rows with extra fields from upstream
        # FLEXIBLE mode allows these extras while enforcing declared field requirements
        mode = "FLEXIBLE"

    # Get field information from Pydantic model_fields
    # This gives us accurate required/optional info
    model_fields = schema_class.model_fields

    # Build FieldContracts
    fields: list[FieldContract] = []

    for name, field_info in model_fields.items():
        if name.startswith("_"):
            continue  # Skip private fields

        # Get the annotation from field_info
        annotation = field_info.annotation
        python_type = _get_python_type(annotation)

        # Field is required if no default and not Optional
        required = field_info.is_required()

        fields.append(
            FieldContract(
                normalized_name=name,
                original_name=name,  # No resolution for transform outputs
                python_type=python_type,
                required=required,
                source="declared",
            )
        )

    return SchemaContract(
        mode=mode,  # type: ignore[arg-type]  # mode is a valid SchemaMode literal; narrowing from str to Literal not expressible here
        fields=tuple(fields),
        locked=True,  # Transform schemas are static
    )


def validate_output_against_contract(
    output: dict[str, Any],
    contract: SchemaContract,
) -> list[ContractViolation]:
    """Validate transform output against output contract.

    Args:
        output: Transform output row
        contract: Output schema contract

    Returns:
        List of violations (empty if valid)
    """
    return contract.validate(output)
