"""Tier-3 coerce-and-record audit serialization (split from ``DataFlowRepository``).

External-origin data (source/quarantined rows, transform payloads, transform
error details) may legitimately contain NaN/Infinity or otherwise
non-canonical values that the canonical serializers reject. These helpers are
the sanctioned coerce-and-record boundary: canonical output when possible, an
explicit recorded fallback otherwise — the absence of a canonical value is
recorded as a distinct representation, never fabricated and never silently
swallowed.
"""

from __future__ import annotations

import json
from typing import Any

from elspeth.contracts import NonCanonicalMetadata
from elspeth.contracts.hashing import repr_hash
from elspeth.core.canonical import canonical_json, stable_hash

__all__ = [
    "canonical_or_recorded_error_details_json",
    "canonical_or_recorded_hash",
    "canonical_or_recorded_json",
    "canonical_or_recorded_repr_payload",
]


def canonical_or_recorded_hash(data: Any) -> str:
    """Hash external row data, recording a non-canonical fallback on failure.

    Tier-3 boundary. ``data`` is external-origin (a source/quarantined row or
    transform-result payload) that may legitimately contain NaN/Infinity or
    otherwise non-canonical values which ``stable_hash`` rejects. This is the
    sanctioned coerce-and-record boundary: we return the canonical hash when
    possible, and otherwise return an explicit ``repr_hash`` fallback. The
    absence of a canonical hash is recorded as a distinct (repr-based) value,
    never fabricated and never silently swallowed.
    """
    try:
        return stable_hash(data)
    except (ValueError, TypeError):
        # Non-canonical external data: return the explicit repr-based fallback
        # so the audit row still records what we received.
        return repr_hash(data)


def canonical_or_recorded_json(data: Any) -> str:
    """Serialize external row data, recording a non-canonical fallback on failure.

    Tier-3 boundary companion to :func:`canonical_or_recorded_hash`. Returns
    canonical JSON when ``data`` is canonicalizable, otherwise returns an
    explicit :class:`NonCanonicalMetadata` envelope (repr + type + error)
    serialized as JSON. The failure is recorded as structured metadata in the
    audit trail, not discarded.
    """
    try:
        return canonical_json(data)
    except (ValueError, TypeError) as exc:
        # Non-canonical external data: return an explicit structured envelope
        # capturing what we saw (repr, type, serialization error).
        return json.dumps(NonCanonicalMetadata.from_error(data, exc).to_dict(), allow_nan=False)


def canonical_or_recorded_error_details_json(error_details: Any) -> str:
    """Serialize transform error_details, recording a non-canonical fallback.

    Tier-3 boundary. ``error_details`` originates from transform results and
    may carry arbitrary row-derived data (NaN/Infinity, non-serializable
    objects from exception context). Returns canonical JSON when possible,
    otherwise an explicit ``__non_canonical__`` envelope (repr + error) — the
    failure is recorded as structured audit data, not discarded.
    """
    try:
        return canonical_json(error_details)
    except (ValueError, TypeError) as exc:
        return json.dumps(
            {
                "__non_canonical__": True,
                "repr": repr(error_details)[:500],
                "serialization_error": str(exc),
            },
            allow_nan=False,
        )


def canonical_or_recorded_repr_payload(data: Any) -> str:
    """Serialize a quarantined payload, recording a repr sentinel on failure.

    Tier-3 boundary for the payload store. ``data`` is quarantined external
    data that may contain non-canonical values. Returns canonical JSON when
    possible, otherwise an explicit ``{"_repr": repr(data)}`` sentinel that
    the query repository recognizes on read-back — the absence of a canonical
    payload is recorded, never fabricated or swallowed.
    """
    try:
        return canonical_json(data)
    except (ValueError, TypeError):
        return json.dumps({"_repr": repr(data)}, allow_nan=False)
