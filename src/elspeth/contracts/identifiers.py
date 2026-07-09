"""Shared identifier validation policy."""

from __future__ import annotations

import keyword
from collections.abc import Sequence


def validate_field_name(
    name: object,
    context: str,
    *,
    strip: bool = False,
    allow_empty: bool = False,
    invalid_identifier_message: str | None = None,
) -> str:
    """Validate and return one field name under the shared identifier policy."""
    if type(name) is not str:
        raise ValueError(f"{context} must be a string, got {type(name).__name__}")

    value = name.strip() if strip else name
    if not value:
        if allow_empty:
            return value
        if invalid_identifier_message is not None:
            raise ValueError(invalid_identifier_message)
        if strip:
            raise ValueError(f"{context} cannot be empty or whitespace-only")
        raise ValueError(f"{context} '{value}' is not a valid Python identifier")
    if not value.isidentifier():
        raise ValueError(invalid_identifier_message or f"{context} '{value}' is not a valid Python identifier")
    if keyword.iskeyword(value):
        raise ValueError(f"{context} '{value}' is a Python keyword")
    return value


def validate_field_names(
    names: Sequence[object],
    context: str,
    *,
    strip: bool = False,
    allow_empty_sequence: bool = True,
    allow_duplicates: bool = False,
) -> tuple[str, ...]:
    """Validate and return field names under the shared identifier policy."""
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise ValueError(f"{context} must be a sequence of field names, got {type(names).__name__}")
    if not names and not allow_empty_sequence:
        raise ValueError(f"{context} must not be empty")

    result = tuple(validate_field_name(name, f"{context}[{i}]", strip=strip) for i, name in enumerate(names))

    if not allow_duplicates:
        seen: set[str] = set()
        for name in result:
            if name in seen:
                raise ValueError(f"Duplicate field names in {context}: {name}")
            seen.add(name)
    return result


__all__ = [
    "validate_field_name",
    "validate_field_names",
]
