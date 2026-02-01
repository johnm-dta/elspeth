"""Common helper functions for landscape modules.

These are extracted from recorder.py to be shared across repositories.
"""

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import TypeVar

E = TypeVar("E", bound=Enum)


def now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(UTC)


def generate_id() -> str:
    """Generate a unique ID (UUID4 hex)."""
    return uuid.uuid4().hex


def coerce_enum(value: str | E, enum_type: type[E]) -> E:
    """Coerce a string or enum value to the target enum type.

    Per Data Manifesto: This is for Tier 1 data (our audit DB).
    Invalid values CRASH - no silent coercion.

    Args:
        value: String or enum value to coerce
        enum_type: Target enum type

    Returns:
        Enum value

    Raises:
        ValueError: If string is not a valid enum value
    """
    if isinstance(value, enum_type):
        return value
    return enum_type(value)
