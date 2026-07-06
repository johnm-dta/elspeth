"""Type-preserving JSON serialization for checkpoint aggregation state.

This module provides serialization that preserves type fidelity for types allowed
in SchemaContract (int, str, float, bool, NoneType, datetime, object) plus the
non-JSON-native types that flow through `object`-typed contract fields and
FLEXIBLE/OBSERVED-mode rows: Decimal, date, time, bytes, and UUID. numpy scalars
are normalized to Python primitives (mirroring canonical_json's numpy handling)
rather than tagged.

This serializer runs on the happy path, not only on resume: the F1 token_data_ref
envelope (expand/coalesce payload persistence) calls checkpoint_dumps on row
payloads during normal execution. It must therefore preserve every type a row
payload can legitimately carry, and raise a clear audit-fidelity error — never a
cryptic stdlib TypeError — on a genuinely unserializable value.

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

import base64
import binascii
import json
import math
from collections.abc import Callable
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from typing import Any, cast
from uuid import UUID

import numpy as np

from elspeth.contracts.errors import AuditIntegrityError

# numpy is a guaranteed dependency (pandas requires it; canonical.py hard-imports
# it in the same layer). numpy scalars are converted to Python primitives —
# mirroring canonical_json's normalization (np.integer→int, np.floating→float,
# np.bool_→bool). numpy-ness is not semantic data: the value, not its container
# type, is what the audit trail records.

# Reserved key used for type envelopes. User dicts containing this key
# are escaped via _escape_reserved_keys() before encoding.
_ENVELOPE_TYPE_KEY = "__elspeth_type__"
_ENVELOPE_VALUE_KEY = "__elspeth_value__"
_KNOWN_ENVELOPE_TYPES = frozenset(
    {
        "datetime",
        "decimal",
        "date",
        "time",
        "bytes",
        "uuid",
        "escaped_dict",
        "tuple",
    }
)


class CheckpointEncoder(json.JSONEncoder):
    """JSON encoder that preserves rich types with collision-safe type envelopes.

    Encodes each preserved type as
    {"__elspeth_type__": <tag>, "__elspeth_value__": <json-native value>}.
    This allows deserialization to restore the original Python type without
    colliding with user dicts that happen to contain similar keys.

    Preserved types (round-trip exactly via the envelope):
    - datetime → ISO 8601 string (timezone-normalized to UTC if naive)
    - Decimal  → decimal string (str(obj); rejects non-finite NaN/Infinity)
    - date     → ISO 8601 date string
    - time     → ISO 8601 time string
    - bytes    → base64-ascii string
    - UUID     → canonical UUID string

    numpy scalars are NOT envelope-tagged: they are converted to Python primitives
    (int/float/bool), mirroring canonical_json's normalization. numpy-ness is not
    semantic data.

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
            ValueError: If a float/Decimal is NaN or Infinity
        """
        # datetime MUST be checked before date — datetime is a subclass of date,
        # so isinstance(dt, date) is True for datetimes. Order is load-bearing.
        if isinstance(obj, datetime):
            # Ensure timezone-aware (audit requirement)
            if obj.tzinfo is None:
                obj = obj.replace(tzinfo=UTC)
            return {
                _ENVELOPE_TYPE_KEY: "datetime",
                _ENVELOPE_VALUE_KEY: obj.isoformat(),
            }

        if isinstance(obj, Decimal):
            # Reject non-finite Decimals — same audit-integrity invariant as float.
            # _reject_nan_infinity only inspects Python float; Decimal('NaN') would
            # otherwise round-trip a NaN back into the Tier-1 audit trail.
            if not obj.is_finite():
                raise ValueError(f"Cannot serialize non-finite Decimal: {obj}. Use None for missing values, not NaN/Infinity.")
            return {
                _ENVELOPE_TYPE_KEY: "decimal",
                _ENVELOPE_VALUE_KEY: str(obj),
            }

        if isinstance(obj, date):
            return {
                _ENVELOPE_TYPE_KEY: "date",
                _ENVELOPE_VALUE_KEY: obj.isoformat(),
            }

        if isinstance(obj, time):
            return {
                _ENVELOPE_TYPE_KEY: "time",
                _ENVELOPE_VALUE_KEY: obj.isoformat(),
            }

        if isinstance(obj, bytes):
            return {
                _ENVELOPE_TYPE_KEY: "bytes",
                _ENVELOPE_VALUE_KEY: base64.b64encode(obj).decode("ascii"),
            }

        if isinstance(obj, UUID):
            return {
                _ENVELOPE_TYPE_KEY: "uuid",
                _ENVELOPE_VALUE_KEY: str(obj),
            }

        # numpy scalars → Python primitive (mirror canonical_json normalization).
        # numpy-ness carries no semantic meaning; the value is what the audit records.
        # No envelope tag: these become real int/float/bool and round-trip natively.
        if isinstance(obj, np.floating):
            # Use numpy-native finiteness — math.isnan downcasts np.longdouble,
            # overflowing the IEEE-754 double range (matches canonical_json).
            if not np.isfinite(obj):
                raise ValueError(f"Cannot serialize non-finite float: {obj}. Use None for missing values, not NaN/Infinity.")
            converted = float(obj)
            # Secondary guard: a value finite in np.longdouble can overflow to inf
            # when narrowed to a Python float (which json then rejects). Raise the
            # clear ELSPETH message instead of json's "Out of range float values".
            # Mirrors canonical_json's post-conversion check.
            if not math.isfinite(converted):
                raise ValueError(
                    f"Cannot serialize {type(obj).__name__} value: exceeds IEEE 754 double range. "
                    f"Value is finite in native representation but overflows JSON number format."
                )
            return converted
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)

        # Tuples are handled in _escape_reserved_keys() pre-processing,
        # not here — json.dumps converts tuples to arrays before calling default().

        # Offensive-programming boundary: the SchemaContract `object` type is an
        # unbounded catch-all, so the serializer cannot be made total. Replace the
        # cryptic stdlib "Object of type X is not JSON serializable" with a clear
        # audit-fidelity error that names the type without previewing raw payload
        # data. Tier-1 checkpoints (token_data_ref envelopes, aggregation state)
        # must contain only JSON-native values or one of the envelope-tagged types.
        raise TypeError(
            f"Cannot serialize value of type {type(obj).__name__!r} into a checkpoint payload "
            f"at the Tier-1 audit-fidelity boundary. "
            f"every value must be JSON-native (int, float, str, bool, None, list, dict) or one of "
            f"the type-preserving envelopes (datetime, Decimal, date, time, bytes, UUID, tuple). "
            f"numpy scalars are accepted and normalized to Python primitives. To record this value, "
            f"convert it to a supported type upstream, or record None for absence."
        )


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

    If a user dict contains a reserved envelope key, wrap it in an escape envelope so
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
        # If this dict contains a reserved envelope key, wrap it in an escape envelope.
        if _ENVELOPE_TYPE_KEY in escaped or _ENVELOPE_VALUE_KEY in escaped:
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

    Preserves rich types (datetime, Decimal, date, time, bytes, UUID) using
    collision-safe type envelopes; numpy scalars convert to Python primitives.
    Escapes user dicts that coincidentally contain the reserved key.
    Rejects NaN/Infinity per CLAUDE.md audit integrity requirements.

    Args:
        obj: Data structure to serialize (typically aggregation state)

    Returns:
        JSON string with type envelopes for rich types
        (datetime/Decimal/date/time/bytes/UUID).

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

    Handles the collision-safe ``__elspeth_type__``/``__elspeth_value__`` envelopes:
    - Rich types: datetime, decimal, date, time, bytes, uuid (restored to their
      Python types from the json-native envelope value).
    - Tuple: {"__elspeth_type__": "tuple", "__elspeth_value__": [...]} → tuple.
    - Escaped dicts: {"__elspeth_type__": "escaped_dict", "__elspeth_value__": {...}}
      → the original user dict that happened to contain the reserved key.

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
        if _ENVELOPE_TYPE_KEY in obj or _ENVELOPE_VALUE_KEY in obj:
            if _ENVELOPE_TYPE_KEY not in obj or _ENVELOPE_VALUE_KEY not in obj or len(obj) != 2:
                raise AuditIntegrityError(
                    f"Corrupted checkpoint: invalid envelope shape for type {obj.get(_ENVELOPE_TYPE_KEY)!r} - "
                    f"reserved envelope keys must appear together and must not be mixed with extra keys"
                )
            envelope_type = obj[_ENVELOPE_TYPE_KEY]
            envelope_value = obj[_ENVELOPE_VALUE_KEY]

            if not isinstance(envelope_type, str):
                raise AuditIntegrityError(
                    f"Checkpoint envelope type tag must be str, got {type(envelope_type).__name__!r} - data may be corrupted"
                )

            if envelope_type == "datetime":
                value = _require_envelope_value_type(envelope_type, envelope_value, str)
                dt = _parse_string_envelope(envelope_type, value, datetime.fromisoformat)
                if dt.tzinfo is None:
                    raise AuditIntegrityError(
                        f"Corrupted checkpoint: datetime envelope contains naive datetime {value!r} — timezone-aware datetimes are required"
                    )
                return dt

            if envelope_type == "decimal":
                value = _require_envelope_value_type(envelope_type, envelope_value, str)
                restored = _parse_string_envelope(envelope_type, value, Decimal)
                # Re-validate finiteness on restore — symmetric with the write-side
                # reject at line 115. Decimal("NaN")/Decimal("Infinity") construct
                # successfully, so without this guard a corrupted or tampered
                # checkpoint round-trips a live non-finite Decimal back into the
                # Tier-1 audit trail. Mirrors the datetime arm's tz-awareness check.
                if not restored.is_finite():
                    raise AuditIntegrityError(
                        f"Corrupted checkpoint: decimal envelope contains non-finite value "
                        f"{value!r} — NaN/Infinity are not valid audit values"
                    )
                return restored

            if envelope_type == "date":
                value = _require_envelope_value_type(envelope_type, envelope_value, str)
                return _parse_string_envelope(envelope_type, value, date.fromisoformat)

            if envelope_type == "time":
                value = _require_envelope_value_type(envelope_type, envelope_value, str)
                return _parse_string_envelope(envelope_type, value, time.fromisoformat)

            if envelope_type == "bytes":
                value = _require_envelope_value_type(envelope_type, envelope_value, str)
                return _restore_bytes_envelope(value)

            if envelope_type == "uuid":
                value = _require_envelope_value_type(envelope_type, envelope_value, str)
                return _parse_string_envelope(envelope_type, value, UUID)

            if envelope_type == "escaped_dict":
                escaped_value = _require_envelope_value_type(envelope_type, envelope_value, dict)
                # Unwrap the escaped dict and recurse into its values
                return {k: _restore_types(v) for k, v in escaped_value.items()}

            if envelope_type == "tuple":
                tuple_value = _require_envelope_value_type(envelope_type, envelope_value, list)
                return tuple(_restore_types(v) for v in tuple_value)

            # Envelope shape detected — all known types handled above.
            if envelope_type in _KNOWN_ENVELOPE_TYPES:
                # Known type but value failed isinstance check above — wrong Python type
                _raise_invalid_envelope_value_type(envelope_type, envelope_value)
            raise AuditIntegrityError(f"Unknown checkpoint envelope type {envelope_type!r} — data may be corrupted or tampered")

        # Recurse into dict values
        return {k: _restore_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_restore_types(v) for v in obj]
    return obj


def _require_envelope_value_type[T](envelope_type: str, envelope_value: object, value_type: type[T]) -> T:
    if not isinstance(envelope_value, value_type):
        _raise_invalid_envelope_value_type(envelope_type, envelope_value)
    return cast(T, envelope_value)


def _raise_invalid_envelope_value_type(envelope_type: str, envelope_value: object) -> None:
    raise AuditIntegrityError(
        f"Checkpoint envelope type {envelope_type!r} has invalid value type {type(envelope_value).__name__!r} — data may be corrupted"
    )


def _parse_string_envelope[T](envelope_type: str, envelope_value: str, parser: Callable[[str], T]) -> T:
    try:
        return parser(envelope_value)
    except (InvalidOperation, ValueError) as exc:
        raise AuditIntegrityError(
            f"Corrupted checkpoint: {envelope_type} envelope contains invalid value {envelope_value!r} - data may be corrupted or tampered"
        ) from exc


def _restore_bytes_envelope(envelope_value: str) -> bytes:
    try:
        restored = base64.b64decode(envelope_value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AuditIntegrityError(
            f"Corrupted checkpoint: bytes envelope contains invalid Base64 {envelope_value!r} - data may be corrupted or tampered"
        ) from exc

    canonical = base64.b64encode(restored).decode("ascii")
    if canonical != envelope_value:
        raise AuditIntegrityError(
            f"Corrupted checkpoint: bytes envelope contains non-canonical Base64 {envelope_value!r} - expected {canonical!r}"
        )
    return restored


def _reject_json_constant(constant: str) -> Any:
    """Reject the non-finite float literals ``json.loads`` accepts by default.

    Python's ``json.loads`` extends strict JSON by parsing the bare tokens
    ``NaN``, ``Infinity``, and ``-Infinity`` into ``float`` values. ``parse_constant``
    is invoked with exactly one of those three strings. Raising here makes the read
    side symmetric with the write side's ``json.dumps(allow_nan=False)``: a corrupted
    or tampered Tier-1 checkpoint carrying a bare non-finite float must crash, not
    silently restore a live ``float('nan')`` into the audit trail.

    (Decimal non-finites travel through the envelope arm instead, where
    ``_restore_types`` applies the matching ``is_finite()`` guard.)
    """
    raise AuditIntegrityError(f"Corrupted checkpoint: non-finite JSON constant {constant!r} — NaN/Infinity are not valid audit values")


def _reject_duplicate_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    restored: dict[str, object] = {}
    for key, value in pairs:
        if key in restored:
            raise AuditIntegrityError(f"Corrupted checkpoint: duplicate JSON object key {key!r} — data may be corrupted or tampered")
        restored[key] = value
    return restored


def checkpoint_loads(s: str) -> Any:
    """Deserialize JSON string with type restoration.

    Restores Python types from the collision-safe ``__elspeth_type__`` envelopes
    (datetime, decimal, date, time, bytes, uuid), plus escaped dicts and tuples.
    The legacy ``__datetime__`` shape tag is intentionally NOT restored (No Legacy
    Code Policy — see ``_restore_types``).

    Non-finite values are rejected on restore (Tier-1 audit integrity): bare float
    literals via ``parse_constant`` below, Decimal envelopes via ``_restore_types``.

    Args:
        s: JSON string (from checkpoint_dumps)

    Returns:
        Data structure with restored Python types

    Raises:
        json.JSONDecodeError: If string is not valid JSON
        AuditIntegrityError: If the payload carries a non-finite NaN/Infinity value
    """
    data = json.loads(s, parse_constant=_reject_json_constant, object_pairs_hook=_reject_duplicate_object_pairs)
    return _restore_types(data)
