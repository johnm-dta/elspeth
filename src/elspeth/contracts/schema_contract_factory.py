"""Factory for creating SchemaContract from configuration.

Bridges the gap between user-facing SchemaConfig (YAML) and runtime
SchemaContract used for validation and dual-name access.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from elspeth.contracts.contract_builder import ContractBuilder
from elspeth.contracts.schema_contract import FieldContract, SchemaContract

if TYPE_CHECKING:
    from elspeth.contracts.schema import SchemaConfig


# Type mapping from SchemaConfig field types to Python types
_FIELD_TYPE_MAP: dict[str, type] = {
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
    "any": object,  # 'any' accepts anything - use object as base type
}


def map_schema_mode(
    mode: Literal["fixed", "flexible", "observed"],
) -> Literal["FIXED", "FLEXIBLE", "OBSERVED"]:
    """Map SchemaConfig mode to SchemaContract mode.

    YAML uses lowercase (fixed/flexible/observed) per YAML convention,
    runtime uses uppercase (FIXED/FLEXIBLE/OBSERVED) for enum-like behavior.

    Args:
        mode: SchemaConfig mode ('fixed', 'flexible', or 'observed')

    Returns:
        SchemaContract mode literal (uppercase)
    """
    # Simple uppercase conversion - YAML lowercase to runtime uppercase
    return mode.upper()  # type: ignore[return-value]  # str.upper() returns str, not Literal; callers validate mode beforehand


def expected_runtime_output_contract(
    config: SchemaConfig,
) -> tuple[Literal["FIXED", "FLEXIBLE", "OBSERVED"], bool]:
    """ADR-014 expected emitted-contract semantics for a declared output schema.

    The single statement of what a transform's EMITTED ``PipelineRow.contract``
    must look like for a given ``SchemaConfig`` declaration: the mapped runtime
    mode, and ``locked=True`` — once a contract is attached to an emitted row
    it is locked, even for ``flexible``/``observed`` config modes whose
    pre-emission builders may begin unlocked.

    Both the producer alignment (``BaseTransform._align_output_contract``) and
    the engine verifier (``verify_schema_config_mode``) derive their
    expectation from here so mode/lock-policy changes cannot drift apart
    (filigree elspeth-986cfb43e5).
    """
    return map_schema_mode(config.mode), True


def create_contract_from_config(
    config: SchemaConfig,
    field_resolution: Mapping[str, str] | None = None,
) -> SchemaContract:
    """Create SchemaContract from SchemaConfig.

    For fixed schemas, creates a locked contract with declared fields.
    For flexible/observed schemas, creates an unlocked contract that will
    infer additional fields from the first valid row.

    Args:
        config: Schema configuration from YAML
        field_resolution: Optional mapping of original->normalized names.
            If provided, original_name on FieldContract will use the
            original header; otherwise, original_name = normalized_name.

    Returns:
        SchemaContract ready for validation
    """
    mode = map_schema_mode(config.mode)

    # Validate normalized mode is a known value
    if mode not in ("FIXED", "FLEXIBLE", "OBSERVED"):
        raise ValueError(f"Invalid schema mode '{config.mode}' (normalized: '{mode}'). Expected 'fixed', 'flexible', or 'observed'.")

    # Validate invariant: FIXED and FLEXIBLE require explicit fields
    if mode in ("FIXED", "FLEXIBLE") and config.fields is None:
        raise ValueError(f"Schema mode '{config.mode}' requires explicit field definitions. Use 'mode: observed' to infer types from data.")

    # Build reverse mapping for looking up original names
    # field_resolution is original->normalized, we need normalized->original
    normalized_to_original: dict[str, str] = {}
    if field_resolution:
        normalized_to_original = ContractBuilder._normalized_to_original(field_resolution)

    # For explicit schemas, create FieldContracts from FieldDefinitions
    fields: tuple[FieldContract, ...] = ()

    if config.fields is not None:
        field_contracts: list[FieldContract] = []
        for fd in config.fields:
            # Look up the original header for this declared field. A field
            # absent from the resolution mapping was not renamed, so its
            # original name equals its normalized name — a supported
            # partial-resolution contract (see test_partial_resolution).
            # Branch explicitly rather than using .get(fd.name, fd.name): the
            # absence-means-identity decision is deliberate, not a silently
            # fabricated fallback default.
            if fd.name in normalized_to_original:
                original = normalized_to_original[fd.name]
            else:
                original = fd.name

            fc = FieldContract(
                normalized_name=fd.name,
                original_name=original,
                python_type=_FIELD_TYPE_MAP[fd.field_type],
                required=fd.required,
                source="declared",
                nullable=fd.nullable,
            )
            field_contracts.append(fc)
        fields = tuple(field_contracts)

    # Derive locked from normalized mode, not raw config.mode.
    # FIXED schemas start locked (types are fully declared).
    # FLEXIBLE/OBSERVED start unlocked and lock after first valid row.
    locked = mode == "FIXED"

    return SchemaContract(
        mode=mode,
        fields=fields,
        locked=locked,
    )
