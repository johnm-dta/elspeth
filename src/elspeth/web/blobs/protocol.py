"""BlobService protocol and record dataclasses.

Record types are frozen dataclasses representing database rows.
BlobCreateData is the input DTO for creating new blobs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Literal, Protocol, get_args, runtime_checkable
from uuid import UUID

from elspeth.contracts.freeze import freeze_fields

# Lifecycle literal aliases.
#
# BlobStatus is the full enum stored on a BlobRecord.  FinalizeBlobStatus
# is the narrower set a pending blob may transition to — pending itself
# is not a valid finalize target.  BlobCreator mirrors ck_blobs_created_by,
# BlobRunLinkDirection mirrors ck_blob_run_links_direction, and
# AllowedMimeType is the closed set of data-oriented MIME types accepted
# for uploads.
#
# The Literal aliases are authoritative — their matching runtime
# frozensets are *derived* via typing.get_args() so adding a member is
# a single-site edit.  The derivation IS the anti-drift mechanism:
# because the frozensets below are built from ``get_args(Literal[...])``
# rather than declared as independent literals, there is no second
# definition that could fall out of sync.  Any edit to a Literal alias
# changes the frozenset produced here in lockstep, so the Tier 1 read
# guards in ``BlobServiceImpl._row_to_record`` and the boundary
# assertions at write sites cannot silently disagree with the static
# type.
BlobStatus = Literal["ready", "pending", "error"]
FinalizeBlobStatus = Literal["ready", "error"]
BlobCreator = Literal["user", "assistant", "pipeline"]
BlobRunLinkDirection = Literal["input", "output"]
AllowedMimeType = Literal[
    "text/csv",
    "text/plain",
    "application/json",
    "application/x-jsonlines",
    "application/jsonl",
    "text/jsonl",
]

# Runtime frozensets derived from the Literal aliases above.  These are
# used by the DB CHECK-mirroring read guards (_row_to_record) and by
# boundary assertions at write sites (create_blob, create_pending_blob).
# Deriving via get_args guarantees the static and runtime views cannot
# drift apart — a single-edit-site contract.
#   ready   — content is available for download/use
#   pending — placeholder for an output blob not yet written
#   error   — run failed before writing the output blob
BLOB_STATUSES: frozenset[str] = frozenset(get_args(BlobStatus))
#   ready   — pending blob whose backing content has been written
#   error   — pending blob whose run failed before writing content
# Narrower than BLOB_STATUSES: ``pending`` is the starting state, not a
# valid finalize target.
FINALIZE_BLOB_STATUSES: frozenset[str] = frozenset(get_args(FinalizeBlobStatus))
#   user      — uploaded by the user via REST or drag-and-drop
#   assistant — materialised by the assistant via create_blob tool
#   pipeline  — produced as output of a pipeline run
BLOB_CREATORS: frozenset[str] = frozenset(get_args(BlobCreator))
#   input  — blob consumed by a run as source data
#   output — blob produced by a run as a pipeline result
BLOB_RUN_LINK_DIRECTIONS: frozenset[str] = frozenset(get_args(BlobRunLinkDirection))
# MIME types accepted for data-oriented uploads.
ALLOWED_MIME_TYPES: frozenset[str] = frozenset(get_args(AllowedMimeType))


@dataclass(frozen=True, slots=True)
class BlobRecord:
    """Represents a row from the blobs table.

    All fields are scalars or None — no freeze guard needed.
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


@dataclass(frozen=True, slots=True)
class BlobRunLinkRecord:
    """Represents a row from the blob_run_links table.

    All fields are scalars — no freeze guard needed.
    """

    blob_id: UUID
    run_id: UUID
    direction: BlobRunLinkDirection


# ─── Blob exception family ───────────────────────────────────────────
#
# Pattern parity with ``elspeth.web.composer.protocol.ComposerServiceError``
# and siblings: every declared attribute is frozen after construction
# via ``_FROZEN_ATTRS`` + ``__setattr__`` override, matching the
# commit-landed contract in the composer family (see composer/protocol.py
# for the full rationale block).  Why apply it here: these exception
# instances flow into HTTP response bodies (404/409/413/500) and into
# structured audit/telemetry emission sites; allowing post-construction
# reassignment would let any intermediate layer silently rewrite what
# downstream consumers see.
#
# Divergence from composer family — NO ``capture()`` classmethod:
# the composer classes use ``capture()`` to encapsulate a derivation
# rule (``partial_state = state if state.version > initial_version
# else None``) as a single source of truth across raise sites.  Blob
# exceptions have no such rule — every attribute is an input from the
# raise site, not derived from surrounding scope.  Adding a pro-forma
# ``capture()`` here would be ceremony without purpose.  If a future
# blob exception grows a real derivation rule, introduce ``capture()``
# at that point so the rule has one home.
#
# Identity args (``blob_id``, ``session_id``) are positional; secondary
# payload args (``run_id``, byte counts, hashes, message text) are
# keyword-only.  This matches the composer pattern's split and makes
# raise sites self-documenting — operators scanning ``raise
# BlobActiveRunError(blob_id, run_id=run)`` immediately see which
# identifier is the subject and which is the context.


def _guard_frozen_attr(instance: Exception, name: str, value: object) -> None:
    """Shared freeze-guard helper for the blob exception family.

    Exception-chain dunders (``__cause__``, ``__context__``,
    ``__suppress_context__``, ``__traceback__``, ``__notes__``) remain
    writable so ``raise ... from ...`` and ``add_note()`` continue to
    work.  First-time write during ``__init__`` is allowed; subsequent
    reassignment raises.
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
    """Raised when a blob lookup fails.

    Route handlers catching this error should return 404.
    """

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"blob_id"})

    def __init__(self, blob_id: str) -> None:
        super().__init__(f"Blob {blob_id} not found")
        self.blob_id = blob_id

    def __setattr__(self, name: str, value: object) -> None:
        _guard_frozen_attr(self, name, value)


class BlobActiveRunError(BlobError):
    """Raised when attempting to delete a blob linked to an active run.

    Route handlers catching this error should return 409.
    """

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"blob_id", "run_id"})

    def __init__(self, blob_id: str, *, run_id: str) -> None:
        super().__init__(f"Blob {blob_id} is linked to active run {run_id} and cannot be deleted")
        self.blob_id = blob_id
        self.run_id = run_id

    def __setattr__(self, name: str, value: object) -> None:
        _guard_frozen_attr(self, name, value)


class BlobQuotaExceededError(BlobError):
    """Raised when a blob creation would exceed the session storage quota.

    Route handlers catching this error should return 413.
    """

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"session_id", "current_bytes", "limit_bytes"})

    def __init__(self, session_id: str, *, current_bytes: int, limit_bytes: int) -> None:
        super().__init__(f"Session {session_id} blob storage ({current_bytes} bytes) would exceed quota ({limit_bytes} bytes)")
        self.session_id = session_id
        self.current_bytes = current_bytes
        self.limit_bytes = limit_bytes

    def __setattr__(self, name: str, value: object) -> None:
        _guard_frozen_attr(self, name, value)


class BlobStateError(BlobError):
    """Raised when a blob's status precludes the requested operation.

    Distinct from RuntimeError so per-blob catch clauses can be precise:
    BlobStateError is an operational condition (concurrent finalization,
    status already terminal), while RuntimeError indicates a code-level
    anomaly (e.g. row vanishing mid-transaction) that should propagate.
    """

    _FROZEN_ATTRS: ClassVar[frozenset[str]] = frozenset({"blob_id"})

    def __init__(self, blob_id: str, *, message: str) -> None:
        super().__init__(message)
        self.blob_id = blob_id

    def __setattr__(self, name: str, value: object) -> None:
        _guard_frozen_attr(self, name, value)


class BlobIntegrityError(BlobError):
    """Raised when a blob's on-disk content does not match its stored hash.

    This is a Tier 1 integrity violation: we wrote both the file and the
    hash, so a mismatch means filesystem corruption, tampering, or a bug
    in our write path.  The web layer should return 500 (not suppress it).

    Deliberately NOT in BlobServiceImpl._PER_BLOB_SUPPRESSED — integrity
    failures must propagate, never be swallowed.
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

    Distinct from BlobNotFoundError: the metadata row exists and claims
    the blob is ``ready``, but the bytes we committed are gone.  This is
    a Tier 1 integrity failure and the web layer should return 500.
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

    Returned in BlobFinalizationResult.errors so callers can decide
    how to surface failures (telemetry, logging, or corrective action)
    without the blob service owning that decision.
    """

    blob_id: UUID
    exc_type: str
    detail: str


@dataclass(frozen=True, slots=True)
class BlobFinalizationResult:
    """Result of batch blob finalization — successes and errors.

    Partial failure is expected: one blob's operational error (concurrent
    deletion, I/O failure) must not prevent finalization of remaining
    blobs.  Callers inspect ``errors`` to determine severity.
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

        Writes content to filesystem, computes hash, and persists metadata.
        """
        ...

    async def create_pending_blob(
        self,
        session_id: UUID,
        filename: str,
        mime_type: AllowedMimeType,
        created_by: BlobCreator = "pipeline",
        source_description: str | None = None,
    ) -> BlobRecord:
        """Reserve a pending output blob (status='pending').

        The backing file doesn't exist yet — it will be written by a
        pipeline sink. Call finalize_blob() after the run completes.
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
        """Get blob metadata. Raises BlobNotFoundError if missing."""
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

        Raises BlobActiveRunError if linked to a pending/running run.
        Raises BlobNotFoundError if the blob doesn't exist.
        """
        ...

    async def read_blob_content(self, blob_id: UUID) -> bytes:
        """Read the raw content of a blob.

        Only ``ready`` blobs are readable.  Verifies the stored
        ``content_hash`` against the bytes on disk before returning.

        Raises BlobNotFoundError if the blob doesn't exist.
        Raises BlobStateError if the blob is not in ``ready`` status.
        Raises BlobContentMissingError if ready metadata exists but the backing file is absent.
        Raises BlobIntegrityError if the content hash doesn't match.
        """
        ...

    async def link_blob_to_run(
        self,
        blob_id: UUID,
        run_id: UUID,
        direction: BlobRunLinkDirection,
    ) -> None:
        """Record a blob-to-run linkage (input or output).

        Raises RuntimeError if direction is outside the declared Literal
        set or if blob and run belong to different sessions.
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

        Creates new blob records with new IDs and new storage paths.
        Copies backing files. Respects the per-session quota.

        Returns a mapping of old blob ID → new BlobRecord, enabling
        callers to remap source references in the forked state.
        """
        ...

    async def finalize_run_output_blobs(
        self,
        run_id: UUID,
        success: bool,
    ) -> BlobFinalizationResult:
        """Finalize pending output blobs for a completed/failed run.

        Processes each blob independently — a per-blob operational error
        (concurrent deletion, I/O failure, DB hiccup) does not abort
        finalization of remaining blobs.  Failed blobs are transitioned
        to ``error`` status on a best-effort basis.

        Returns a BlobFinalizationResult with both successfully finalized
        blobs and per-blob error records.  Callers inspect ``result.errors``
        to decide how to surface failures.
        """
        ...
