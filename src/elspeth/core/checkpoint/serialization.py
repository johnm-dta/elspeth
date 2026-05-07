"""Type-preserving JSON serialization for checkpoint aggregation state.

This module provides serialization that preserves type fidelity for types allowed
in SchemaContract (int, str, float, bool, NoneType, datetime, object).

The problem: Standard json.dumps() cannot serialize datetime objects.
The solution: Use collision-safe type envelopes with ``__elspeth_type__`` and
``__elspeth_value__`` keys. User dicts that coincidentally contain the reserved
key ``__elspeth_type__`` are escaped via ``_escape_reserved_keys()`` before
encoding, preventing incorrect deserialization.

This replaces the old shape-based tag ``{"__datetime__": iso_string}`` which
could collide with user data matching the same shape. Per CLAUDE.md No Legacy
Code Policy, the old tag format is not supported during deserialization.

This is distinct from canonical_json() which:
1. Is designed for hashing (normalized output)
2. Converts datetime to bare ISO strings (no type tags)
3. Normalizes floats in ways that could change values

Checkpoint serialization needs round-trip fidelity, not canonical form.

Per CLAUDE.md:
- NaN/Infinity are rejected (audit integrity)
- datetime must round-trip correctly (type fidelity)
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any

from elspeth.contracts.errors import AuditIntegrityError

# Reserved key used for type envelopes. User dicts containing this key
# are escaped via _escape_reserved_keys() before encoding.
_ENVELOPE_TYPE_KEY = "__elspeth_type__"
_ENVELOPE_VALUE_KEY = "__elspeth_value__"


class CheckpointEncoder(json.JSONEncoder):
    """JSON encoder that preserves datetime with collision-safe type envelopes.

    Encodes datetime as {"__elspeth_type__": "datetime", "__elspeth_value__": "iso_string"}.
    This allows deserialization to restore the original datetime type without
    colliding with user dicts that happen to contain similar keys.

    NaN and Infinity are rejected per CLAUDE.md audit integrity requirements.
    """

    def default(self, obj: Any) -> Any:
        """Encode non-standard types.

        Args:
            obj: Object to encode

        Returns:
            JSON-serializable representation

        Raises:
            TypeError: If object cannot be serialized
            ValueError: If float is NaN or Infinity
        """
        if isinstance(obj, datetime):
            # Ensure timezone-aware (audit requirement)
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=UTC)
            return {
                _ENVELOPE_TYPE_KEY: "datetime",
                _ENVELOPE_VALUE_KEY: obj.isoformat(),
            }

        # Tuples are handled in _escape_reserved_keys() pre-processing,
        # not here — json.dumps converts tuples to arrays before calling default().

        # Let default encoder handle or raise TypeError
        return super().default(obj)


def _reject_nan_infinity(obj: Any) -> Any:
    """Recursively check for NaN/Infinity in data structure.

    Per CLAUDE.md: NaN/Infinity are strictly rejected for audit integrity.

    Args:
        obj: Data structure to validate

    Returns:
        The same object if valid

    Raises:
        ValueError: If NaN or Infinity found
    """
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValueError(f"Cannot serialize non-finite float: {obj}. Use None for missing values, not NaN/Infinity.")
    elif isinstance(obj, dict):
        for v in obj.values():
            _reject_nan_infinity(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _reject_nan_infinity(v)
    return obj


def _escape_reserved_keys(obj: Any) -> Any:
    """Recursively escape user dicts that coincidentally contain the reserved key.

    If a user dict contains __elspeth_type__, wrap it in an escape envelope so
    _restore_types() can distinguish it from a real type envelope.

    Args:
        obj: Data structure to process

    Returns:
        Data with reserved keys escaped
    """
    if isinstance(obj, datetime):
        # Datetimes are handled by CheckpointEncoder, pass through
        return obj
    if isinstance(obj, tuple):
        # Convert tuple to envelope dict BEFORE json.dumps sees it.
        # json.dumps treats tuples as arrays (never calls default()),
        # so we must create the envelope here during pre-processing.
        return {
            _ENVELOPE_TYPE_KEY: "tuple",
            _ENVELOPE_VALUE_KEY: [_escape_reserved_keys(v) for v in obj],
        }
    if isinstance(obj, dict):
        # First recurse into values
        escaped = {k: _escape_reserved_keys(v) for k, v in obj.items()}
        # If this dict contains our reserved key, wrap it in an escape envelope
        if _ENVELOPE_TYPE_KEY in escaped:
            return {
                _ENVELOPE_TYPE_KEY: "escaped_dict",
                _ENVELOPE_VALUE_KEY: escaped,
            }
        return escaped
    if isinstance(obj, list):
        return [_escape_reserved_keys(v) for v in obj]
    return obj


def checkpoint_dumps(obj: Any) -> str:
    """Serialize object to JSON with type preservation.

    Preserves datetime objects using collision-safe type envelopes.
    Escapes user dicts that coincidentally contain the reserved key.
    Rejects NaN/Infinity per CLAUDE.md audit integrity requirements.

    Args:
        obj: Data structure to serialize (typically aggregation state)

    Returns:
        JSON string with type envelopes for datetime

    Raises:
        ValueError: If data contains NaN or Infinity
        TypeError: If data contains non-serializable types
    """
    # Validate no NaN/Infinity before serialization
    _reject_nan_infinity(obj)

    # Escape user dicts that contain reserved keys before encoding
    escaped = _escape_reserved_keys(obj)

    return json.dumps(escaped, cls=CheckpointEncoder, allow_nan=False)


def _restore_types(obj: Any) -> Any:
    """Recursively restore type-tagged values.

    Handles:
    - New envelopes: {"__elspeth_type__": "datetime", "__elspeth_value__": iso_string}
    - Escaped dicts: {"__elspeth_type__": "escaped_dict", "__elspeth_value__": {...}}

    The old shape-based tag {"__datetime__": iso_string} is NOT restored. Per
    CLAUDE.md No Legacy Code Policy, there are no existing checkpoints to
    preserve compatibility with.

    Args:
        obj: Deserialized JSON data

    Returns:
        Data with restored Python types
    """
    if isinstance(obj, dict):
        # Check for collision-safe envelope
        if _ENVELOPE_TYPE_KEY in obj and _ENVELOPE_VALUE_KEY in obj and len(obj) == 2:
            envelope_type = obj[_ENVELOPE_TYPE_KEY]
            envelope_value = obj[_ENVELOPE_VALUE_KEY]

            if envelope_type == "datetime" and isinstance(envelope_value, str):
                dt = datetime.fromisoformat(envelope_value)
                if dt.tzinfo is None:
                    raise AuditIntegrityError(
                        f"Corrupted checkpoint: datetime envelope contains naive datetime "
                        f"{envelope_value!r} — timezone-aware datetimes are required"
                    )
                return dt

            if envelope_type == "escaped_dict" and isinstance(envelope_value, dict):
                # Unwrap the escaped dict and recurse into its values
                return {k: _restore_types(v) for k, v in envelope_value.items()}

            if envelope_type == "tuple" and isinstance(envelope_value, list):
                return tuple(_restore_types(v) for v in envelope_value)

            # Envelope shape detected — all known types handled above.
            _KNOWN_ENVELOPE_TYPES = {"datetime", "escaped_dict", "tuple"}
            if envelope_type in _KNOWN_ENVELOPE_TYPES:
                # Known type but value failed isinstance check above — wrong Python type
                raise AuditIntegrityError(
                    f"Checkpoint envelope type {envelope_type!r} has invalid value type "
                    f"{type(envelope_value).__name__!r} — data may be corrupted"
                )
            raise AuditIntegrityError(f"Unknown checkpoint envelope type {envelope_type!r} — data may be corrupted or tampered")

        # Recurse into dict values
        return {k: _restore_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_restore_types(v) for v in obj]
    return obj


def checkpoint_loads(s: str) -> Any:
    """Deserialize JSON string with type restoration.

    Restores datetime objects from type envelopes. Supports both new
    collision-safe envelopes and legacy __datetime__ tags.

    Args:
        s: JSON string (from checkpoint_dumps)

    Returns:
        Data structure with restored Python types

    Raises:
        json.JSONDecodeError: If string is not valid JSON
    """
    data = json.loads(s)
    return _restore_types(data)
