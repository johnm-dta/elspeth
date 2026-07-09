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
_OPERATOR_REQUIRED_PLACEHOLDER_LABEL = "operatorrequired"


def _normalized_placeholder_label(value: str) -> str:
    return "".join(ch for ch in value.strip().lower() if ch.isalnum())


def is_placeholder_value(value: str) -> bool:
    """Return true when a user/config value carries a placeholder sentinel."""
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.startswith("<") and stripped.endswith(">"):
        return True
    return _normalized_placeholder_label(stripped) in _PLACEHOLDER_LABELS


def reject_placeholder_value(value: str, *, field_name: str) -> str:
    """Return value unless it is a placeholder sentinel."""
    if is_placeholder_value(value):
        raise ValueError(f"{field_name} must be a real configured value; placeholder values are not valid")
    return value


def is_operator_required_placeholder_value(value: str) -> bool:
    """Return true when value carries the explicit operator-required sentinel."""
    return _normalized_placeholder_label(value) == _OPERATOR_REQUIRED_PLACEHOLDER_LABEL


def reject_operator_required_placeholder_value(value: str, *, field_name: str) -> str:
    """Return resource identifier values unless they are explicit operator-required sentinels."""
    if is_operator_required_placeholder_value(value):
        raise ValueError(f"{field_name} must be a real configured value; operator-required placeholder values are not valid")
    return value


def is_wire_visible_placeholder(value: str) -> bool:
    """Return true when a wire-visible identity field carries a sentinel value."""
    return is_placeholder_value(value)
