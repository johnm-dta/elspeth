"""Shared utilities for the plugin system.

This module provides common functions used across multiple plugin types
to avoid code duplication and ensure consistent behavior.
"""

from typing import Any

from elspeth.plugins.sentinels import MISSING


def get_nested_field(
    data: dict[str, Any],
    path: str,
    default: Any = MISSING,
) -> Any:
    """Get value from nested dict using dot notation.

    Traverses a nested dictionary structure using a dot-separated path.
    Returns the MISSING sentinel (or custom default) if the path doesn't exist.

    This function does NOT coerce values - it returns exactly what is found
    or the default if the path is missing. This is appropriate for pipeline
    data which has elevated trust (validated at source boundaries).

    Args:
        data: Source dictionary to traverse
        path: Dot-separated path (e.g., "user.profile.name")
        default: Value to return if path not found (default: MISSING sentinel)

    Returns:
        Value at path, or default if not found

    Examples:
        >>> data = {"user": {"name": "Alice", "age": 30}}
        >>> get_nested_field(data, "user.name")
        'Alice'
        >>> from elspeth.plugins.sentinels import MISSING
        >>> get_nested_field(data, "user.email") is MISSING
        True
        >>> get_nested_field(data, "user.email", default="unknown")
        'unknown'
    """
    parts = path.split(".")
    current: Any = data

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]

    return current
