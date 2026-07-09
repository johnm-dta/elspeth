"""Common helper functions for landscape modules.

These are extracted from recorder.py to be shared across landscape modules.
"""

from datetime import UTC, datetime

from elspeth.core.ids import generate_id

__all__ = ["generate_id", "now"]


def now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(UTC)
