"""Shared sentinel values for the plugin system.

This module provides sentinel objects used to distinguish between "value not found"
and "value is explicitly None" when accessing fields in row data.

The sentinel pattern is essential for field extraction operations where:
- A field might legitimately contain None as a value
- A field might be missing entirely from the data structure
- These two cases need to be handled differently

Example usage:
    from elspeth.plugins.sentinels import MISSING

    def get_field(data: dict, key: str) -> Any:
        if key not in data:
            return MISSING
        return data[key]

    value = get_field(row, "optional_field")
    if value is MISSING:
        # Field was not present in the data
        handle_missing_field()
    elif value is None:
        # Field was present but explicitly set to None
        handle_null_value()
"""

from typing import Final


class MissingSentinel:
    """Sentinel class to distinguish missing fields from None values.

    This is a singleton - use the MISSING instance, not the class directly.
    Comparison should always use `is` identity, never equality.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<MISSING>"


MISSING: Final[MissingSentinel] = MissingSentinel()
"""Singleton sentinel indicating a field was not found.

Use identity comparison: `if value is MISSING:`
"""
