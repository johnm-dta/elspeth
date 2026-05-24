"""Pin blob storage contracts that must live at L0."""

from __future__ import annotations

from typing import get_args

from elspeth.contracts.blobs import ALLOWED_MIME_TYPES, AllowedMimeType


def test_allowed_mime_types_at_l0() -> None:
    """The MIME contract is importable without touching the web layer."""
    assert "text/csv" in ALLOWED_MIME_TYPES
    assert "application/json" in ALLOWED_MIME_TYPES
    assert "text/plain" in ALLOWED_MIME_TYPES
    assert isinstance(ALLOWED_MIME_TYPES, frozenset)


def test_allowed_mime_type_literal_get_args_consistency() -> None:
    """Anti-drift: the Literal alias and frozenset are co-derived."""
    assert frozenset(get_args(AllowedMimeType)) == ALLOWED_MIME_TYPES
