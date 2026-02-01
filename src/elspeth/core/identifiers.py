"""Identifier validation utilities.

This module provides validation for field names and other Python identifiers
used throughout ELSPETH. It's in core/ to be accessible from any subsystem
without creating cross-subsystem imports.
"""

from __future__ import annotations

import keyword


def validate_field_names(names: list[str], context: str) -> None:
    """Validate field names are valid identifiers and not keywords.

    Args:
        names: List of field names to validate
        context: Description for error messages (e.g., "columns", "field_mapping values")

    Raises:
        ValueError: If any name is invalid identifier, is Python keyword, or is duplicate
    """
    seen: set[str] = set()
    for i, name in enumerate(names):
        if not name.isidentifier():
            raise ValueError(f"{context}[{i}] '{name}' is not a valid Python identifier")
        if keyword.iskeyword(name):
            raise ValueError(f"{context}[{i}] '{name}' is a Python keyword")
        if name in seen:
            raise ValueError(f"Duplicate field name '{name}' in {context}")
        seen.add(name)
