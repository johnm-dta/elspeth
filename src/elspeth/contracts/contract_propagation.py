"""Contract propagation through transform pipeline.

Contracts flow through the pipeline, carrying field metadata (types,
original names) from source to sink. Transforms may add fields, which
get inferred types, or remove fields (narrowing the contract).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import structlog

from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.type_normalization import normalize_type_for_contract

log = structlog.get_logger()


def propagate_contract(
    input_contract: SchemaContract,
    output_row: dict[str, Any],
    *,
    transform_adds_fields: bool = True,
) -> SchemaContract:
    """Propagate contract through transform, inferring new field types.

    For passthrough transforms: returns input contract unchanged.
    For transforms adding fields: infers types from output values.

    Args:
        input_contract: Contract from input row
        output_row: Transform output data
        transform_adds_fields: If True, infer types for new fields

    Returns:
        Contract for output row
    """
    if not transform_adds_fields:
        # Passthrough - same contract
        return input_contract

    # Check for new fields in output
    existing_names = {f.normalized_name for f in input_contract.fields}
    new_fields: list[FieldContract] = []

    for name, value in output_row.items():
        if name not in existing_names:
            # New field - try to infer type
            try:
                python_type = normalize_type_for_contract(value)
            except TypeError:
                # Preserve common complex JSON structures as "any" while
                # preserving prior skip behavior for other unsupported types.
                if type(value) in (dict, list):
                    python_type = object
                else:
                    continue

            new_fields.append(
                FieldContract(
                    normalized_name=name,
                    original_name=name,  # No original for transform-created fields
                    python_type=python_type,
                    required=False,  # Inferred fields are never required
                    source="inferred",
                )
            )

    if not new_fields:
        return input_contract

    # Create new contract with additional fields
    return SchemaContract(
        mode=input_contract.mode,
        fields=input_contract.fields + tuple(new_fields),
        locked=True,
    )


def narrow_contract_to_output(
    input_contract: SchemaContract,
    output_row: dict[str, Any],
    *,
    renamed_fields: Mapping[str, str] | None = None,
) -> SchemaContract:
    """Narrow contract to match output row fields (handles field removal/renaming).

    For transforms that remove or rename fields, we need to:
    1. Remove fields not in output (e.g., JSONExplode removes array_field)
    2. Add new fields in output (e.g., FieldMapper adds target, JSONExplode adds output_field)

    Args:
        input_contract: Contract from input row
        output_row: Transform output data
        renamed_fields: Optional source->target mapping for renames that were
            actually applied by the transform. When provided, metadata from
            the source field is preserved on the renamed target field.

    Returns:
        Contract containing fields from input that still exist + new fields

    Note:
        TODO: Extract shared field inference logic with propagate_contract() - 90% overlap
    """
    output_field_names = set(output_row.keys())

    # Keep fields from input contract that exist in output
    kept_fields = [fc for fc in input_contract.fields if fc.normalized_name in output_field_names]

    # Find NEW fields in output (not in input contract)
    existing_names = {f.normalized_name for f in input_contract.fields}
    new_fields: list[FieldContract] = []
    renamed_targets: list[str] = []
    skipped_fields: list[str] = []

    # Build target->source lookup for metadata preservation.
    # If multiple sources map to the same target, last mapping wins.
    source_by_target: dict[str, str] = {}
    if renamed_fields is not None:
        for source, target in renamed_fields.items():
            source_by_target[target] = source
    original_to_normalized = {fc.original_name: fc.normalized_name for fc in input_contract.fields}

    for name, value in output_row.items():
        if name not in existing_names:
            source_contract = None
            if name in source_by_target:
                source_name = source_by_target[name]
                normalized_source_name = source_name
                if source_name in original_to_normalized:
                    normalized_source_name = original_to_normalized[source_name]
                source_contract = input_contract.find_field(normalized_source_name)
            if source_contract is not None:
                renamed_targets.append(name)
                new_fields.append(
                    FieldContract(
                        normalized_name=name,
                        original_name=source_contract.original_name,
                        python_type=source_contract.python_type,
                        required=source_contract.required,
                        source=source_contract.source,
                    )
                )
                continue

            try:
                python_type = normalize_type_for_contract(value)
            except TypeError as e:
                if type(value) in (dict, list):
                    python_type = object
                else:
                    # Skip unsupported non-dict/list types to preserve prior behavior.
                    skipped_fields.append(name)
                    log.debug(
                        "contract_field_skipped",
                        field_name=name,
                        reason=type(e).__name__,
                        value_type=type(value).__name__,
                    )
                    continue
            except ValueError as e:
                # Skip invalid values (NaN, Infinity)
                skipped_fields.append(name)
                log.debug(
                    "contract_field_skipped",
                    field_name=name,
                    reason=type(e).__name__,
                    value_type=type(value).__name__,
                )
                continue

            new_fields.append(
                FieldContract(
                    normalized_name=name,
                    original_name=name,  # New fields have no original name
                    python_type=python_type,
                    required=False,  # Inferred fields are never required
                    source="inferred",
                )
            )

    # B4: Observability for contract modifications
    log.debug(
        "contract_narrowed",
        input_field_count=len(input_contract.fields),
        output_field_count=len(kept_fields) + len(new_fields),
        fields_kept=[f.normalized_name for f in kept_fields],
        fields_renamed=renamed_targets,
        fields_inferred=[f.normalized_name for f in new_fields if f.normalized_name not in renamed_targets],
        fields_skipped=skipped_fields,
    )

    return SchemaContract(
        mode=input_contract.mode,
        fields=tuple(kept_fields + new_fields),
        locked=True,
    )


def merge_contract_with_output(
    input_contract: SchemaContract,
    output_schema_contract: SchemaContract,
) -> SchemaContract:
    """Merge input contract with transform's output schema.

    The output schema contract defines what the transform guarantees.
    We merge this with input contract to preserve original names
    while adding any new guaranteed fields.

    Args:
        input_contract: Contract from input (has original names)
        output_schema_contract: Contract from transform.output_schema

    Returns:
        Merged contract with original names and output guarantees
    """
    # Build lookup for input contract original names
    input_originals = {f.normalized_name: f.original_name for f in input_contract.fields}

    # Build merged fields
    merged_fields: list[FieldContract] = []

    for output_field in output_schema_contract.fields:
        # Preserve original name from input if available
        original = input_originals.get(
            output_field.normalized_name,
            output_field.original_name,
        )

        merged_fields.append(
            FieldContract(
                normalized_name=output_field.normalized_name,
                original_name=original,
                python_type=output_field.python_type,
                required=output_field.required,
                source=output_field.source,
            )
        )

    # Use most restrictive mode
    mode_order: dict[Literal["FIXED", "FLEXIBLE", "OBSERVED"], int] = {
        "FIXED": 0,
        "FLEXIBLE": 1,
        "OBSERVED": 2,
    }
    merged_mode = min(
        input_contract.mode,
        output_schema_contract.mode,
        key=lambda m: mode_order[m],
    )

    return SchemaContract(
        mode=merged_mode,
        fields=tuple(merged_fields),
        locked=True,
    )
