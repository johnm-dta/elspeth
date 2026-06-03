"""Pin blob storage contracts that must live at L0."""

from __future__ import annotations

from typing import get_args

from elspeth.contracts.blobs import (
    ALLOWED_MIME_TYPES,
    AllowedMimeType,
    BlobActiveRunError,
    BlobContentMissingError,
    BlobError,
    BlobIntegrityError,
    BlobNotFoundError,
    BlobQuotaExceededError,
    BlobServiceProtocol,
    BlobStateError,
)


def test_allowed_mime_types_at_l0() -> None:
    """The MIME contract is importable without touching the web layer."""
    assert "text/csv" in ALLOWED_MIME_TYPES
    assert "application/json" in ALLOWED_MIME_TYPES
    assert "text/plain" in ALLOWED_MIME_TYPES
    assert isinstance(ALLOWED_MIME_TYPES, frozenset)


def test_allowed_mime_type_literal_get_args_consistency() -> None:
    """Anti-drift: the Literal alias and frozenset are co-derived."""
    assert frozenset(get_args(AllowedMimeType)) == ALLOWED_MIME_TYPES


def test_blob_exception_family_lives_at_l0() -> None:
    """Core inline blob resolution catches these without importing L3 web code."""
    for cls in (
        BlobNotFoundError,
        BlobActiveRunError,
        BlobQuotaExceededError,
        BlobStateError,
        BlobIntegrityError,
        BlobContentMissingError,
    ):
        assert issubclass(cls, BlobError)
        assert cls.__module__ == "elspeth.contracts.blobs"


def test_blob_service_protocol_lives_at_l0() -> None:
    """The L1 inline resolver can type against blob service without an upward import."""
    assert BlobServiceProtocol.__module__ == "elspeth.contracts.blobs"
