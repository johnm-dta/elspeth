"""Contracts for widened ``blob_ref`` inline content markers.

Layer: L0. No upward imports.

This module defines the shared marker grammar and value objects for
audited inline blob content. Web composer code, validation code, and the
core resolver all consume these contracts, so the definitions live here
instead of in the web blob service.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Literal, cast, get_args
from uuid import UUID

from elspeth.contracts.blobs import ALLOWED_MIME_TYPES, AllowedMimeType

ContentEncoding = Literal["utf-8", "utf-8-sig", "utf-16", "latin-1"]
"""Closed set of decoders allowed for inline content."""

ALLOWED_CONTENT_ENCODINGS: frozenset[str] = frozenset(get_args(ContentEncoding))
"""Runtime view derived from ``ContentEncoding`` to prevent drift."""

BlobRefMode = Literal["bind_source", "inline_content"]
"""Allowed widened ``blob_ref`` modes."""

ALLOWED_BLOB_REF_MODES: frozenset[str] = frozenset(get_args(BlobRefMode))
"""Runtime view derived from ``BlobRefMode`` to prevent drift."""

BlobInlineValidationCategory = Literal["missing", "oversized", "not_ready", "hash_mismatch", "malformed"]
"""Validation failure categories for inline-content blob refs."""

_SHA256_HEX_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_UUID_PATTERN: Final = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_FIELD_PATH_PATTERN: Final = re.compile(
    r"^(?:source(?::[^.\[\]]+)?|node:[^.\[\]]+|output:[^.\[\]]+)"
    r"\.options(?:\.[A-Za-z_][A-Za-z0-9_-]*)+$"
)
_ALLOWED_MARKER_KEYS: Final = frozenset({"blob_ref", "mode", "path", "sha256", "encoding"})


def _validate_sha256(owner: str, value: str) -> None:
    if _SHA256_HEX_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{owner}: sha256 must be 64 lowercase hex characters")


def _validate_field_path(owner: str, value: str) -> None:
    if _FIELD_PATH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{owner}: field_path must use source/node/output identity prefixes and .options.<field> segments")


@dataclass(frozen=True, slots=True)
class WidenedBlobRefShape:
    """Parsed widened ``blob_ref`` marker."""

    blob_id: UUID
    mode: BlobRefMode
    sha256: str | None = None
    path: str | None = None
    encoding: ContentEncoding = "utf-8"

    def __post_init__(self) -> None:
        if self.mode not in ALLOWED_BLOB_REF_MODES:
            raise ValueError(f"{type(self).__name__}: mode must be one of {sorted(ALLOWED_BLOB_REF_MODES)}")
        if self.encoding not in ALLOWED_CONTENT_ENCODINGS:
            raise ValueError(f"{type(self).__name__}: encoding must be one of {sorted(ALLOWED_CONTENT_ENCODINGS)}")
        if self.mode == "inline_content":
            if self.path is not None:
                raise ValueError(f"{type(self).__name__}: inline_content markers cannot carry path")
            if self.sha256 is None:
                raise ValueError(f"{type(self).__name__}: inline_content markers require sha256")
            _validate_sha256(type(self).__name__, self.sha256)
            return
        if self.sha256 is not None:
            raise ValueError(f"{type(self).__name__}: bind_source markers cannot carry sha256")
        if self.encoding != "utf-8":
            raise ValueError(f"{type(self).__name__}: bind_source markers cannot carry encoding")


@dataclass(frozen=True, slots=True)
class BlobInlineRef:
    """Inline content ref discovered during config tree-walk."""

    field_path: str
    blob_id: UUID
    sha256: str
    encoding: ContentEncoding

    def __post_init__(self) -> None:
        _validate_field_path(type(self).__name__, self.field_path)
        _validate_sha256(type(self).__name__, self.sha256)
        if self.encoding not in ALLOWED_CONTENT_ENCODINGS:
            raise ValueError(f"{type(self).__name__}: encoding must be one of {sorted(ALLOWED_CONTENT_ENCODINGS)}")


@dataclass(frozen=True, slots=True)
class BlobInlineValidationViolation:
    """Structured validate-path violation for inline blob content refs."""

    category: BlobInlineValidationCategory
    field_path: str
    detail: str


@dataclass(frozen=True, slots=True)
class ResolvedBlobContent:
    """Audit-row payload for one successfully resolved inline content ref."""

    field_path: str
    blob_id: UUID
    content_hash: str
    byte_length: int
    mime_type: AllowedMimeType
    encoding: ContentEncoding

    def __post_init__(self) -> None:
        _validate_field_path(type(self).__name__, self.field_path)
        _validate_sha256(type(self).__name__, self.content_hash)
        if self.byte_length < 0:
            raise ValueError(f"{type(self).__name__}: byte_length must be non-negative")
        if self.mime_type not in ALLOWED_MIME_TYPES:
            raise ValueError(f"{type(self).__name__}: mime_type must be one of {sorted(ALLOWED_MIME_TYPES)}")
        if self.encoding not in ALLOWED_CONTENT_ENCODINGS:
            raise ValueError(f"{type(self).__name__}: encoding must be one of {sorted(ALLOWED_CONTENT_ENCODINGS)}")


def is_widened_blob_ref(value: Any) -> WidenedBlobRefShape | None:
    """Return a parsed widened ``blob_ref`` marker, or ``None``.

    Mode-less dictionaries with a ``blob_ref`` key are malformed markers
    and return ``None`` here. Resolver and validation walkers can report
    those as structural errors by separately checking for the key.
    """
    if type(value) is not dict:
        return None
    marker = cast(dict[str, object], value)
    if "blob_ref" not in marker:
        return None
    if not set(marker).issubset(_ALLOWED_MARKER_KEYS):
        return None
    blob_ref_value = marker["blob_ref"]
    if type(blob_ref_value) is not str:
        return None
    if _UUID_PATTERN.fullmatch(blob_ref_value) is None:
        return None
    blob_id = UUID(blob_ref_value)

    if "mode" not in marker:
        return None
    mode_value = marker["mode"]
    if type(mode_value) is not str:
        return None
    if mode_value not in ALLOWED_BLOB_REF_MODES:
        return None

    if mode_value == "bind_source":
        if "sha256" in marker or "encoding" in marker:
            return None
        path_value: str | None = None
        if "path" in marker:
            raw_path = marker["path"]
            if type(raw_path) is not str:
                return None
            path_value = raw_path
        return WidenedBlobRefShape(
            blob_id=blob_id,
            mode="bind_source",
            path=path_value,
        )

    if "path" in marker:
        return None
    if "sha256" not in marker:
        return None
    sha256_value = marker["sha256"]
    if type(sha256_value) is not str:
        return None
    if _SHA256_HEX_PATTERN.fullmatch(sha256_value) is None:
        return None
    encoding_value: object = "utf-8"
    if "encoding" in marker:
        encoding_value = marker["encoding"]
    if type(encoding_value) is not str:
        return None
    if encoding_value not in ALLOWED_CONTENT_ENCODINGS:
        return None
    return WidenedBlobRefShape(
        blob_id=blob_id,
        mode="inline_content",
        sha256=sha256_value,
        encoding=cast(ContentEncoding, encoding_value),
    )


class BlobContentResolutionError(Exception):
    """Batched recoverable failures from inline blob content resolution."""

    missing: tuple[str, ...]
    oversized: tuple[tuple[str, int, int], ...]
    undecodable: tuple[tuple[str, str], ...]
    not_ready: tuple[tuple[str, str], ...]
    cross_session: tuple[str, ...]
    malformed: tuple[tuple[str, str], ...]

    def __init__(
        self,
        *,
        missing: list[str] | tuple[str, ...] | None = None,
        oversized: list[tuple[str, int, int]] | tuple[tuple[str, int, int], ...] | None = None,
        undecodable: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
        not_ready: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
        cross_session: list[str] | tuple[str, ...] | None = None,
        malformed: list[tuple[str, str]] | tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        self.missing = tuple(missing or ())
        self.oversized = tuple(oversized or ())
        self.undecodable = tuple(undecodable or ())
        self.not_ready = tuple(not_ready or ())
        self.cross_session = tuple(cross_session or ())
        self.malformed = tuple(malformed or ())
        super().__init__(self._summary())

    def _summary(self) -> str:
        counts = {
            "missing": len(self.missing),
            "oversized": len(self.oversized),
            "undecodable": len(self.undecodable),
            "not_ready": len(self.not_ready),
            "cross_session": len(self.cross_session),
            "malformed": len(self.malformed),
        }
        non_zero = [f"{key}={value}" for key, value in counts.items() if value]
        if not non_zero:
            return "Blob inline content resolution failed"
        return "Blob inline content resolution failed: " + ", ".join(non_zero)
