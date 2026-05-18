"""Type normalization for schema contracts.

Converts numpy/pandas types to Python primitives for consistent
contract storage and validation.

Fast path: Uses type() with frozenset membership for standard Python types (performance).
Slow path: Uses isinstance() checks for numpy/pandas type hierarchies.
Avoids string matching on __name__ per CLAUDE.md.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Final, cast

# NOTE: numpy and pandas are imported LAZILY inside normalize_type_for_contract()
# to avoid breaking the contracts leaf module boundary. Importing them at module
# level pulls in 400+ modules just for type normalization.

# Canonical type registry: string name → Python type.
# Single source of truth for all contract type maps in the codebase.
# Used for checkpoint serialization/deserialization, type validation,
# and annotation resolution across contracts/ modules.
CONTRACT_TYPE_MAP: dict[str, type] = {
    "int": int,
    "str": str,
    "float": float,
    "bool": bool,
    "NoneType": type(None),
    "datetime": datetime,
    "object": object,  # 'any' type for fields that accept any value
}

# Types that can be serialized in checkpoint and restored in from_checkpoint()
# Derived from CONTRACT_TYPE_MAP to stay in sync.
ALLOWED_CONTRACT_TYPES: frozenset[type] = frozenset(CONTRACT_TYPE_MAP.values())


class _UnsupportedContractTypeSignal:
    """Singleton signal for runtime values that cannot be checkpointed as field types."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSUPPORTED_CONTRACT_TYPE"


UNSUPPORTED_CONTRACT_TYPE: Final = _UnsupportedContractTypeSignal()
NormalizedContractType = type | _UnsupportedContractTypeSignal


def _unsupported_contract_type_message(value: Any) -> str:
    final_type = type(value)
    return (
        f"Unsupported type '{final_type.__name__}' for schema contract. "
        f"Allowed types: {', '.join(sorted(t.__name__ for t in ALLOWED_CONTRACT_TYPES))}. "
        f"Use 'any' type declaration for fields with complex/dynamic types."
    )


def normalize_type_for_contract(value: Any) -> NormalizedContractType:
    """Convert value's type to Python primitive for contract storage.

    Args:
        value: Any Python value

    Returns:
        Python primitive type or UNSUPPORTED_CONTRACT_TYPE for unknowns

    Raises:
        ValueError: If value is NaN or Infinity (invalid for audit trail)

    Note:
        numpy and pandas are imported lazily inside this function to avoid
        pulling in heavy dependencies when the contracts package is imported.
    """
    if value is None:
        return type(None)

    # Fast path: standard Python types skip numpy/pandas imports entirely.
    # Exclude float — must fall through to NaN/Infinity rejection below.
    fast_type = type(value)
    if fast_type is not float and fast_type in ALLOWED_CONTRACT_TYPES:
        return fast_type

    # Lazy imports to maintain contracts as a leaf module
    import numpy as np
    import pandas as pd

    # Missing-value sentinels normalize to NoneType
    if value is pd.NA:
        return type(None)

    # pd.NaT = "Not a Time" (missing datetime), treat like None
    if value is pd.NaT:
        return type(None)

    # CRITICAL: Reject NaN/Infinity (Tier 1 audit integrity)
    # Must check before numpy type normalization to catch np.float64(nan) etc.
    if isinstance(value, (float, np.floating)) and (math.isnan(value) or math.isinf(value)):
        raise ValueError(f"Cannot infer type from non-finite float: {value}. NaN/Infinity are invalid in audit trail.")

    # Normalize numpy/pandas types to primitives
    if isinstance(value, np.integer):
        return int
    if isinstance(value, np.floating):
        return float
    if isinstance(value, np.bool_):
        return bool
    if isinstance(value, pd.Timestamp):
        return datetime
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return type(None)
        return datetime
    if isinstance(value, np.str_):
        return str
    # NOTE: np.bytes_ is NOT normalized to str (or bytes). It falls through
    # to the ALLOWED_CONTRACT_TYPES check and returns the unsupported signal. This prevents
    # silent misclassification of binary data as text in schema contracts.

    # Return an explicit unsupported signal instead of raising TypeError as a
    # protocol value. Inference callers can map this to object/any, while
    # fail-fast callers can opt in via require_supported_contract_type().
    final_type = type(value)
    if final_type not in ALLOWED_CONTRACT_TYPES:
        return UNSUPPORTED_CONTRACT_TYPE
    return final_type


def require_supported_contract_type(value: Any) -> type:
    """Normalize a value and raise TypeError when it cannot be checkpointed.

    This keeps fail-fast callers explicit without requiring inference callers
    to catch TypeError as a control-flow protocol.
    """
    normalized_type = normalize_type_for_contract(value)
    if normalized_type is UNSUPPORTED_CONTRACT_TYPE:
        raise TypeError(_unsupported_contract_type_message(value))
    return cast(type, normalized_type)


def classify_runtime_type(value: Any) -> type:
    """Classify a value's type for runtime validation comparison.

    Unlike normalize_type_for_contract(), this function NEVER raises.
    It normalizes numpy/pandas wrappers to Python primitives (so that
    numpy.int64(42) compares equal to int), but returns type(value) for
    anything it doesn't recognize — letting the caller produce a
    TypeMismatchViolation instead of crashing the pipeline.

    This is the correct function for SchemaContract.validate(), where
    values come from Tier 3 external data and exotic types should be
    quarantined, not crash the run.

    Args:
        value: Any Python value from a pipeline row

    Returns:
        Python primitive type for known types, or type(value) for unknowns.
        Never raises.
    """
    if value is None:
        return type(None)

    # Fast path: standard Python types (including float — no NaN rejection
    # at validation time, that's a contract-inference concern)
    fast_type = type(value)
    if fast_type in ALLOWED_CONTRACT_TYPES:
        return fast_type

    # Lazy imports to maintain contracts as a leaf module
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return fast_type

    # Missing-value sentinels
    if value is pd.NA or value is pd.NaT:
        return type(None)

    # Non-finite floats: return float (the type is correct even if the
    # value is problematic — validation compares types, not values)
    if isinstance(value, (float, np.floating)) and (math.isnan(value) or math.isinf(value)):
        return float

    # Normalize numpy/pandas types to primitives
    if isinstance(value, np.integer):
        return int
    if isinstance(value, np.floating):
        return float
    if isinstance(value, np.bool_):
        return bool
    if isinstance(value, pd.Timestamp):
        return datetime
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return type(None)
        return datetime
    if isinstance(value, np.str_):
        return str

    # Unknown type — return as-is for comparison. The caller will see
    # actual_type != expected_type and produce a TypeMismatchViolation.
    return fast_type
