# src/elspeth/core/canonical.py
"""
Canonical JSON serialization for deterministic hashing.

Two-phase approach:
1. Normalize: Convert pandas/numpy types to JSON-safe primitives (our code)
2. Serialize: Produce deterministic JSON per RFC 8785/JCS (rfc8785 package)

IMPORTANT: NaN and Infinity are strictly REJECTED, not silently converted.
This is defense-in-depth for audit integrity.
"""

import math
from typing import Any

# Version string stored with every run for hash verification
CANONICAL_VERSION = "sha256-rfc8785-v1"


def _normalize_value(obj: Any) -> Any:
    """Convert a single value to JSON-safe primitive.

    Args:
        obj: Any Python value

    Returns:
        JSON-serializable primitive

    Raises:
        ValueError: If value contains NaN or Infinity
    """
    # Primitives pass through unchanged
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    # Floats: check for non-finite values FIRST
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(
                f"Cannot canonicalize non-finite float: {obj}. "
                "Use None for missing values, not NaN."
            )
        return obj

    return obj
