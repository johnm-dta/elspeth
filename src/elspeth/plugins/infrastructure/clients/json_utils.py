"""Shared strict JSON parsing with NaN/Infinity rejection.

Extracted from AuditedHTTPClient to be reusable by any client that handles
external JSON responses at the Tier 3 boundary. Both AuditedHTTPClient and
DataverseClient callers import from this module.

Python's stdlib json module accepts non-finite float values (NaN, Infinity)
by default, but these cannot be canonicalized via RFC 8785 for audit hashing.
Detecting them at the HTTP boundary produces clean, Tier-3-attributed errors
rather than cryptic canonicalization crashes downstream.
"""

from __future__ import annotations

import json
import math
from json import JSONDecodeError
from typing import Any


def contains_non_finite(obj: Any) -> bool:
    """Recursively check if object contains NaN or Infinity float values.

    This is a Tier 3 boundary check: external JSON may contain non-finite values
    (Python's json module accepts them), but canonicalization rejects them. We
    detect these at the HTTP boundary to record as parse failure rather than
    crashing during audit recording.

    Args:
        obj: Any JSON-parsed value (dict, list, or primitive)

    Returns:
        True if any float value is NaN or Infinity
    """
    if isinstance(obj, float):
        return math.isnan(obj) or math.isinf(obj)
    if isinstance(obj, dict):
        return any(contains_non_finite(v) for v in obj.values())
    if isinstance(obj, list):
        return any(contains_non_finite(v) for v in obj)
    return False


class DuplicateJSONKeyError(ValueError):
    """Raised when a JSON object contains duplicate keys at a given nesting level."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject JSON objects with duplicate keys at this nesting level.

    Used as ``object_pairs_hook`` for ``json.loads`` to enforce unique keys.
    Python's default parser silently applies last-wins for duplicates, which
    means ``{"a": 1, "a": 2}`` becomes ``{"a": 2}`` with no error — data loss
    that violates Tier 3 recording integrity.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(constant: str) -> None:
    """Reject NaN/Infinity during JSON parsing via parse_constant callback.

    Raises ValueError instead of returning a Python float, so non-finite
    values are caught during parsing rather than requiring a recursive
    post-parse walk (which is vulnerable to RecursionError on deep nesting).
    """
    raise ValueError(f"JSON contains non-finite value: {constant}")


def parse_json_strict(text: str) -> tuple[Any, str | None]:
    """Parse JSON with strict rejection of NaN/Infinity.

    Python's stdlib json module accepts non-finite values by default, but
    these cannot be canonicalized. This function rejects them during parsing
    via parse_constant, avoiding a recursive post-parse walk that would be
    vulnerable to RecursionError on deeply nested external payloads.

    Args:
        text: JSON string to parse

    Returns:
        Tuple of (parsed_value, error_message)
        - On success: (parsed_dict_or_list, None)
        - On failure: (None, error_message)
    """
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except (JSONDecodeError, DuplicateJSONKeyError, ValueError) as e:
        return None, str(e)

    return parsed, None
