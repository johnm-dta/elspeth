"""Identifier validation utilities.

The canonical identifier policy lives in :mod:`elspeth.contracts.identifiers`
so contracts and higher layers share the same validation boundary.
"""

from __future__ import annotations

from elspeth.contracts.identifiers import validate_field_name, validate_field_names

__all__ = [
    "validate_field_name",
    "validate_field_names",
]
