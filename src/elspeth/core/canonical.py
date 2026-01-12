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

import numpy as np
import pandas as pd

# Version string stored with every run for hash verification
CANONICAL_VERSION = "sha256-rfc8785-v1"


def _normalize_value(obj: Any) -> Any:
    """Convert a single value to JSON-safe primitive.

    Handles pandas and numpy types that appear in real pipeline data.

    NaN Policy: STRICT REJECTION
    - NaN and Infinity are invalid input states, not "missing"
    - Use None/pd.NA/NaT for intentional missing values
    - This prevents silent data corruption in audit records

    Args:
        obj: Any Python value

    Returns:
        JSON-serializable primitive

    Raises:
        ValueError: If value contains NaN or Infinity
    """
    # Check for NaN/Infinity FIRST (before type coercion)
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(
                f"Cannot canonicalize non-finite float: {obj}. "
                "Use None for missing values, not NaN."
            )
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    # Primitives pass through unchanged
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj

    # NumPy scalar types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_normalize_value(x) for x in obj.tolist()]

    # Pandas types
    if isinstance(obj, pd.Timestamp):
        # Naive timestamps assumed UTC (explicit policy)
        if obj.tz is None:
            return obj.tz_localize("UTC").isoformat()
        return obj.tz_convert("UTC").isoformat()

    # Intentional missing values (NOT NaN - that's rejected above)
    if obj is pd.NA or (isinstance(obj, type(pd.NaT)) and obj is pd.NaT):
        return None

    return obj
