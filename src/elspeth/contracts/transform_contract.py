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
    return origin is Union or origin is UnionType


def _unwrap_annotated(annotation: Any) -> Any:
    """Unwrap typing.Annotated recursively to its underlying type."""
    current = annotation
    while get_origin(current) is Annotated:
        args = get_args(current)
        if not args:
            return current
        current = args[0]
    return current


def _get_python_type(annotation: Any) -> tuple[type, bool]:
    """Extract Python type and nullability from type annotation.

    Handles Optional/Union by detecting T | None pattern.
    Unknown types raise TypeError (only explicit Any maps to object).

    Args:
        annotation: Type annotation from schema

    Returns:
        Tuple of (python_type, is_nullable)

    Raises:
        TypeError: If annotation is an unsupported concrete type or multi-type union
    """
    unwrapped = _unwrap_annotated(annotation)

    # Handle Optional[X] which is Union[X, None] or X | None
    if _is_union_type(unwrapped):
        args = get_args(unwrapped)
        non_none_args = [a for a in args if a is not type(None)]
        has_none = len(non_none_args) < len(args)

        if len(non_none_args) == 0:
            # Union of only NoneType (weird but handle it)
            return (type(None), False)

        if len(non_none_args) > 1:
            # Multi-type union like int | float - not supported
            type_names = [str(a) for a in non_none_args]
            raise TypeError(
                f"Multi-type union '{' | '.join(type_names)}' not supported in schema contracts. "
                f"Use 'Any' for fields that accept multiple types."
            )

        # Exactly one non-None type: T | None pattern
        resolved_type, _ = _get_python_type(non_none_args[0])
        return (resolved_type, has_none)  # nullable = True if None was in the union

    # Explicit Any -> object (intentional wildcard)
    if unwrapped is Any:
        return (object, False)

    # Simple type - must be in allowed set
    if unwrapped in ALLOWED_CONTRACT_TYPES:
        return (cast(type, unwrapped), False)

    # Unsupported concrete type (e.g., list[str], dict[str, Any])
    type_name = str(unwrapped)
    raise TypeError(
        f"Unsupported type annotation '{type_name}' in schema contract. "
        f"Allowed types: {', '.join(sorted(t.__name__ for t in ALLOWED_CONTRACT_TYPES))}. "
        f"Use 'Any' for fields that accept any value."
    )


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
        python_type, is_nullable = _get_python_type(annotation)

        # Field is required if no default and not Optional
        required = field_info.is_required()

        fields.append(
            FieldContract(
                normalized_name=name,
                original_name=name,  # No resolution for transform outputs
                python_type=python_type,
                required=required,
                source="declared",
                nullable=is_nullable,
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
