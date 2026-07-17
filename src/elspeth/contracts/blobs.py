"""Blob storage contracts shared below the web layer.

Layer: L0. No upward imports.

This module hosts the blob closed sets, record DTOs, exception family,
and service protocol used by both the web blob service and lower-layer
inline blob content resolution. When a value type is needed below the
web layer, the dependency direction is preserved by moving the contract
down instead of importing upward from L3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar, Literal, Protocol, get_args, runtime_checkable
from uuid import UUID

from elspeth.contracts.enums import CreationModality
from elspeth.contracts.freeze import freeze_fields

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

BlobStatus = Literal["ready", "pending", "error"]
FinalizeBlobStatus = Literal["ready", "error"]
BlobCreator = Literal["user", "assistant", "pipeline"]
BlobRunLinkDirection = Literal["input", "output"]

# Runtime frozensets are derived from the Literal aliases so static and
# runtime views share one edit site. These mirror DB CHECK constraints
# and are used by write-boundary assertions and Tier 1 read guards.
BLOB_STATUSES: frozenset[str] = frozenset(get_args(BlobStatus))
FINALIZE_BLOB_STATUSES: frozenset[str] = frozenset(get_args(FinalizeBlobStatus))
BLOB_CREATORS: frozenset[str] = frozenset(get_args(BlobCreator))
BLOB_RUN_LINK_DIRECTIONS: frozenset[str] = frozenset(get_args(BlobRunLinkDirection))


@dataclass(frozen=True, slots=True)
class BlobRecord:
    """Represents a row from the blobs table.

    Inline-blob provenance fields are populated only for LLM-authored
    modalities. The database enforces their all-or-nothing invariant;
    web blob read guards mirror the enum-membership checks.
    """

    id: UUID
    session_id: UUID
    filename: str
    mime_type: AllowedMimeType
    size_bytes: int
    content_hash: str | None
    storage_path: str
    created_at: datetime
    created_by: BlobCreator
    source_description: str | None
    status: BlobStatus
    creation_modality: CreationModality
    created_from_message_id: str | None
    creating_model_identifier: str | None
    creating_model_version: str | None
    creating_provider: str | None
    creating_composer_skill_hash: str | None
    creating_arguments_hash: str | None


@dataclass(frozen=True, slots=True)
class InlineCustodyRequest:
    """Exact bytes and provenance for one idempotent inline-source write.

    ``content`` is deliberately excluded from ``repr`` because these requests
    can cross exception and diagnostic boundaries. The deterministic identity
    is derived by the blob service from its SHA-256 digest, never by rendering
    the bytes into logs or audit records.
    """

    session_id: UUID
    filename: str
    content: bytes = field(repr=False)
    mime_type: AllowedMimeType
    source_description: str | None
    creation_modality: CreationModality
    created_from_message_id: str
    creating_model_identifier: str | None
    creating_model_version: str | None
    creating_provider: str | None
    creating_composer_skill_hash: str | None
    creating_arguments_hash: str | None


@dataclass(frozen=True, slots=True)
class BlobRunLinkRecord:
    """Represents a row from the blob_run_links table."""

    blob_id: UUID
    run_id: UUID
    direction: BlobRunLinkDirection


def _guard_frozen_attr(instance: Exception, name: str, value: object) -> None:
    """Prevent post-construction mutation of declared exception payloads.

    Exception-chain dunders remain writable so ``raise ... from ...`` and
    ``add_note()`` continue to work. First-time writes during ``__init__``
    are allowed; subsequent reassignment raises.
    """
    frozen: frozenset[str] = type(instance)._FROZEN_ATTRS  # type: ignore[attr-defined]
    if name in frozen and name in instance.__dict__:
        raise AttributeError(
            f"{type(instance).__name__}.{name} is frozen after construction; "
            "exception attributes flow into HTTP responses and audit telemetry."
        )
    Exception.__setattr__(instance, name, value)


class BlobError(Exception):
    """Base class for structured blob lifecycle errors."""


class BlobNotFoundError(BlobError):
    """Raised when a blob lookup fails."""

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"blob_id"})

    def __init__(self, blob_id: str) -> None:
        super().__init__(f"Blob {blob_id} not found")
        self.blob_id = blob_id

    def __setattr__(self, name: str, value: object) -> None:
        _guard_frozen_attr(self, name, value)


class BlobActiveRunError(BlobError):
    """Raised when attempting to delete a blob linked to an active run."""

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"blob_id", "run_id"})

    def __init__(self, blob_id: str, *, run_id: str) -> None:
        super().__init__(f"Blob {blob_id} is linked to active run {run_id} and cannot be deleted")
        self.blob_id = blob_id
        self.run_id = run_id

    def __setattr__(self, name: str, value: object) -> None:
        _guard_frozen_attr(self, name, value)


class BlobPendingProposalError(BlobError):
    """Raised when a pending proposal still authorizes use of a blob."""

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"blob_id", "proposal_id"})

    def __init__(self, blob_id: str, *, proposal_id: str) -> None:
        super().__init__(f"Blob {blob_id} is referenced by pending proposal {proposal_id} and cannot be deleted")
        self.blob_id = blob_id
        self.proposal_id = proposal_id

    def __setattr__(self, name: str, value: object) -> None:
        _guard_frozen_attr(self, name, value)


class BlobQuotaExceededError(BlobError):
    """Raised when a blob creation would exceed the session storage quota."""

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"session_id", "current_bytes", "limit_bytes"})

    def __init__(self, session_id: str, *, current_bytes: int, limit_bytes: int) -> None:
        super().__init__(f"Session {session_id} blob storage ({current_bytes} bytes) would exceed quota ({limit_bytes} bytes)")
        self.session_id = session_id
        self.current_bytes = current_bytes
        self.limit_bytes = limit_bytes

    def __setattr__(self, name: str, value: object) -> None:
        _guard_frozen_attr(self, name, value)


class BlobStateError(BlobError):
    """Raised when a blob's status precludes the requested operation."""

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"blob_id"})

    def __init__(self, blob_id: str, *, message: str) -> None:
        super().__init__(message)
        self.blob_id = blob_id

    def __setattr__(self, name: str, value: object) -> None:
        _guard_frozen_attr(self, name, value)


class BlobIntegrityError(BlobError):
    """Raised when a blob's on-disk content does not match its stored hash.

    This is a Tier 1 integrity violation: the system wrote both the file
    and the hash, so mismatch means corruption, tampering, or a write-path
    bug. Callers must propagate it rather than batching or suppressing it.
    """

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"blob_id", "expected_hash", "actual_hash"})

    def __init__(self, blob_id: str, *, expected: str, actual: str) -> None:
        super().__init__(f"Blob {blob_id} content integrity failure: stored hash {expected[:16]}... != computed hash {actual[:16]}...")
        self.blob_id = blob_id
        self.expected_hash = expected
        self.actual_hash = actual

    def __setattr__(self, name: str, value: object) -> None:
        _guard_frozen_attr(self, name, value)


class BlobContentMissingError(BlobError):
    """Raised when a ready blob row points at an absent backing file.

    Distinct from ``BlobNotFoundError``: metadata exists and claims the
    blob is ready, but the committed bytes are gone. This is a Tier 1
    integrity failure.
    """

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"blob_id", "storage_path"})

    def __init__(self, blob_id: str, *, storage_path: str) -> None:
        super().__init__(f"Blob {blob_id} content missing: ready metadata points at absent backing file {storage_path}")
        self.blob_id = blob_id
        self.storage_path = storage_path

    def __setattr__(self, name: str, value: object) -> None:
        _guard_frozen_attr(self, name, value)


@dataclass(frozen=True, slots=True)
class BlobFinalizationError:
    """Record of a per-blob finalization failure.

    Returned in ``BlobFinalizationResult.errors`` so callers decide how
    to surface failures without the blob service owning that policy.
    """

    blob_id: UUID
    exc_type: str
    detail: str


@dataclass(frozen=True, slots=True)
class BlobFinalizationResult:
    """Result of batch blob finalization: successes and per-blob errors.

    Partial failure is expected: one blob's operational error must not
    prevent finalization of remaining blobs.
    """

    finalized: Sequence[BlobRecord]
    errors: Sequence[BlobFinalizationError]

    def __post_init__(self) -> None:
        freeze_fields(self, "finalized", "errors")


@runtime_checkable
class BlobServiceProtocol(Protocol):
    """Protocol for blob persistence and lifecycle operations."""

    async def create_blob(
        self,
        session_id: UUID,
        filename: str,
        content: bytes,
        mime_type: AllowedMimeType,
        created_by: BlobCreator = "user",
        source_description: str | None = None,
    ) -> BlobRecord:
        """Create a blob from content bytes.

        Writes content to storage, computes its hash, and persists
        metadata.
        """
        ...

    async def reserve_inline_custody(self, request: InlineCustodyRequest) -> BlobRecord:
        """Idempotently materialize one deterministic inline-source blob."""
        ...

    async def create_pending_blob(
        self,
        session_id: UUID,
        filename: str,
        mime_type: AllowedMimeType,
        created_by: BlobCreator = "pipeline",
        source_description: str | None = None,
    ) -> BlobRecord:
        """Reserve a pending output blob.

        The backing file does not exist yet; a pipeline sink writes it
        before ``finalize_blob`` marks the record ready or error.
        """
        ...

    async def finalize_blob(
        self,
        blob_id: UUID,
        status: FinalizeBlobStatus,
        size_bytes: int | None = None,
        content_hash: str | None = None,
    ) -> BlobRecord:
        """Update a pending blob to ready or error after execution."""
        ...

    async def get_blob(self, blob_id: UUID) -> BlobRecord:
        """Get blob metadata. Raises ``BlobNotFoundError`` if missing."""
        ...

    async def list_blobs(
        self,
        session_id: UUID,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[BlobRecord]:
        """List blobs for a session, newest first."""
        ...

    async def delete_blob(self, blob_id: UUID) -> None:
        """Delete blob metadata and backing file.

        Raises ``BlobActiveRunError`` if linked to an active run and
        ``BlobNotFoundError`` if the blob does not exist.
        """
        ...

    async def read_blob_content(self, blob_id: UUID) -> bytes:
        """Read the raw content of a ready blob.

        Only ready blobs are readable. The stored hash is verified before
        bytes are returned. Operational misses raise ``BlobNotFoundError``
        or ``BlobStateError``; integrity anomalies raise
        ``BlobContentMissingError`` or ``BlobIntegrityError``.
        """
        ...

    async def link_blob_to_run(
        self,
        blob_id: UUID,
        run_id: UUID,
        direction: BlobRunLinkDirection,
    ) -> None:
        """Record a blob-to-run linkage.

        Raises ``RuntimeError`` if direction is outside the declared
        closed set or the blob and run belong to different sessions.
        """
        ...

    async def get_blob_run_links(
        self,
        blob_id: UUID,
    ) -> list[BlobRunLinkRecord]:
        """Get all run links for a blob."""
        ...

    async def copy_blobs_for_fork(
        self,
        source_session_id: UUID,
        target_session_id: UUID,
    ) -> dict[UUID, BlobRecord]:
        """Copy all ready blobs from source to target session.

        Returns old blob ID to new record mappings so callers can remap
        source references in forked state.
        """
        ...

    async def finalize_run_output_blobs(
        self,
        run_id: UUID,
        success: bool,
    ) -> BlobFinalizationResult:
        """Finalize pending output blobs for a completed or failed run.

        Processes each blob independently and returns both successful
        finalizations and per-blob error records.
        """
        ...
