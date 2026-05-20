"""Validation helpers for identity values that leave ELSPETH over the wire."""

from __future__ import annotations

_PLACEHOLDER_LABELS: frozenset[str] = frozenset(
    {
        "changeme",
        "notprovided",
        "operatorrequired",
        "placeholder",
        "required",
        "tbd",
        "todo",
        "unknown",
        "unset",
    }
)


def _normalized_placeholder_label(value: str) -> str:
    return "".join(ch for ch in value.strip().lower() if ch.isalnum())


def is_wire_visible_placeholder(value: str) -> bool:
    """Return true when a wire-visible identity field carries a sentinel value."""
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.startswith("<") and stripped.endswith(">"):
        return True
    return _normalized_placeholder_label(stripped) in _PLACEHOLDER_LABELS
