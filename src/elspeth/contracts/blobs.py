"""Blob storage contracts shared below the web layer.

Layer: L0. No upward imports.

This module hosts MIME-type contracts used by both the web blob service
and lower-layer inline blob content resolution. When a value type is
needed below the web layer, the dependency direction is preserved by
moving the contract down instead of importing upward from L3.
"""

from __future__ import annotations

from typing import Literal, get_args

AllowedMimeType = Literal[
    "text/csv",
    "text/plain",
    "application/json",
    "application/x-jsonlines",
    "application/jsonl",
    "text/jsonl",
]
"""Closed set of MIME types accepted for data-oriented blob uploads."""

ALLOWED_MIME_TYPES: frozenset[str] = frozenset(get_args(AllowedMimeType))
"""Runtime view derived from ``AllowedMimeType`` to prevent drift."""
