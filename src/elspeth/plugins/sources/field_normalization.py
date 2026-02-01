"""Field name normalization for external data sources.

This module normalizes messy external headers (e.g., "CaSE Study1 !!!! xx!")
to valid Python identifiers (e.g., "case_study1_xx") at the source boundary.

Per ELSPETH's Three-Tier Trust Model, this is Tier 3 (external data) handling:
- Sources ARE allowed to normalize/coerce external data
- Transforms expect normalized names (no coercion downstream)

Algorithm Stability:
    The normalization algorithm is versioned and frozen per major version.
    NORMALIZATION_ALGORITHM_VERSION is stored in the audit trail to enable
    debugging cross-run field name drift when algorithm evolves.
"""

from __future__ import annotations

import keyword
import re
import unicodedata
from dataclasses import dataclass

# Algorithm version for audit trail - frozen per major version
# Increment when algorithm changes affect output
NORMALIZATION_ALGORITHM_VERSION = "1.0.0"


def get_normalization_version() -> str:
    """Return current algorithm version for audit trail storage."""
    return NORMALIZATION_ALGORITHM_VERSION


# Pre-compiled regex patterns (module level for efficiency)
_NON_IDENTIFIER_CHARS = re.compile(r"[^\w]+")
_CONSECUTIVE_UNDERSCORES = re.compile(r"_+")


def normalize_field_name(raw: str) -> str:
    """Normalize messy header to valid Python identifier.

    Rules applied in order:
    1. Unicode NFC normalization (canonical composition)
    2. Strip leading/trailing whitespace
    3. Lowercase
    4. Replace non-identifier chars with underscore
    5. Collapse consecutive underscores
    6. Strip leading/trailing underscores
    7. Prefix with underscore if starts with digit
    8. Append underscore if result is Python keyword
    9. Raise error if result is empty

    Args:
        raw: Original messy header name

    Returns:
        Valid Python identifier

    Raises:
        ValueError: If header normalizes to empty string
    """
    # Step 1: Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", raw)

    # Step 2: Strip whitespace
    normalized = normalized.strip()

    # Step 3: Lowercase
    normalized = normalized.lower()

    # Step 4: Replace non-identifier chars with underscore
    normalized = _NON_IDENTIFIER_CHARS.sub("_", normalized)

    # Step 5: Collapse consecutive underscores
    normalized = _CONSECUTIVE_UNDERSCORES.sub("_", normalized)

    # Step 6: Strip leading/trailing underscores
    normalized = normalized.strip("_")

    # Step 7: Prefix if starts with digit
    if normalized and normalized[0].isdigit():
        normalized = f"_{normalized}"

    # Step 8: Handle Python keywords
    if keyword.iskeyword(normalized):
        normalized = f"{normalized}_"

    # Step 9: Validate non-empty result
    if not normalized:
        raise ValueError(f"Header '{raw}' normalizes to empty string")

    # Defense-in-depth: verify result is valid identifier
    if not normalized.isidentifier():
        raise ValueError(
            f"Header '{raw}' normalized to '{normalized}' which is not a valid identifier. This is a bug in the normalization algorithm."
        )

    return normalized


def check_normalization_collisions(raw_headers: list[str], normalized_headers: list[str]) -> None:
    """Check for collisions after normalization.

    Args:
        raw_headers: Original header names
        normalized_headers: Normalized header names (same order)

    Raises:
        ValueError: If multiple raw headers normalize to same value,
                   with ALL colliding headers and their positions listed
    """
    seen: dict[str, list[tuple[int, str]]] = {}

    for i, (raw, norm) in enumerate(zip(raw_headers, normalized_headers, strict=True)):
        seen.setdefault(norm, []).append((i, raw))

    collisions = {norm: sources for norm, sources in seen.items() if len(sources) > 1}

    if collisions:
        details = []
        for norm, sources in sorted(collisions.items()):
            source_desc = ", ".join(f"column {i} ('{raw}')" for i, raw in sources)
            details.append(f"  '{norm}' <- {source_desc}")

        raise ValueError("Field name collision after normalization:\n" + "\n".join(details))


def check_mapping_collisions(
    pre_mapping: list[str],
    post_mapping: list[str],
    field_mapping: dict[str, str],
) -> None:
    """Check for collisions created by field_mapping.

    Args:
        pre_mapping: Headers before mapping applied
        post_mapping: Headers after mapping applied
        field_mapping: The mapping that was applied

    Raises:
        ValueError: If mapping causes multiple fields to have same final name
    """
    if len(post_mapping) != len(set(post_mapping)):
        # Build mapping from target to all sources (both mapped and passthrough)
        target_to_sources: dict[str, list[str]] = {}
        for source, target in zip(pre_mapping, post_mapping, strict=True):
            target_to_sources.setdefault(target, []).append(source)

        collisions = {t: s for t, s in target_to_sources.items() if len(s) > 1}

        if collisions:
            details = [f"  '{target}' <- {', '.join(repr(s) for s in sources)}" for target, sources in sorted(collisions.items())]
            raise ValueError("field_mapping creates collision:\n" + "\n".join(details))


@dataclass(frozen=True)
class FieldResolution:
    """Result of field name resolution.

    Attributes:
        final_headers: List of final field names to use
        resolution_mapping: Mapping from original -> final names (for audit trail)
        normalization_version: Algorithm version used, or None if no normalization
    """

    final_headers: list[str]
    resolution_mapping: dict[str, str]
    normalization_version: str | None


def resolve_field_names(
    *,
    raw_headers: list[str] | None,
    normalize_fields: bool,
    field_mapping: dict[str, str] | None,
    columns: list[str] | None,
) -> FieldResolution:
    """Resolve final field names from raw headers and config.

    Args:
        raw_headers: Headers from file, or None if using columns config
        normalize_fields: Whether to apply normalization algorithm
        field_mapping: Optional mapping overrides (keys are effective names)
        columns: Explicit column names for headerless mode

    Returns:
        FieldResolution with final headers, audit mapping, and algorithm version

    Raises:
        ValueError: On collision, invalid mapping key, or configuration error
    """
    # Track whether normalization was used
    used_normalization = False

    # Determine source of headers
    if columns is not None:
        # Headerless mode - use explicit columns
        original_names = columns
        effective_headers = list(columns)
    elif raw_headers is not None:
        original_names = raw_headers
        if normalize_fields:
            effective_headers = [normalize_field_name(h) for h in raw_headers]
            check_normalization_collisions(raw_headers, effective_headers)
            used_normalization = True
        else:
            effective_headers = list(raw_headers)
    else:
        raise ValueError("Either raw_headers or columns must be provided")

    # Apply field mapping if provided
    if field_mapping:
        # Validate all mapping keys exist
        available = set(effective_headers)
        missing = set(field_mapping.keys()) - available
        if missing:
            raise ValueError(f"field_mapping keys not found in headers: {sorted(missing)}. Available: {sorted(available)}")

        # Apply mapping: mapped headers use new name, unmapped pass through
        # Explicit 'if in' check preferred over .get() per no-bug-hiding policy
        final_headers = [
            field_mapping[h] if h in field_mapping else h  # noqa: SIM401
            for h in effective_headers
        ]

        # Check for collisions after mapping
        check_mapping_collisions(effective_headers, final_headers, field_mapping)
    else:
        final_headers = effective_headers

    # Build resolution mapping for audit trail
    resolution_mapping = dict(zip(original_names, final_headers, strict=True))

    return FieldResolution(
        final_headers=final_headers,
        resolution_mapping=resolution_mapping,
        normalization_version=get_normalization_version() if used_normalization else None,
    )
