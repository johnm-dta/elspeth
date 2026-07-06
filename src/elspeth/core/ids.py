"""Neutral core ID generation primitives."""

from __future__ import annotations

import uuid


def generate_id() -> str:
    """Generate a unique ID as UUID4 lowercase hex."""
    return uuid.uuid4().hex


__all__ = ["generate_id"]
