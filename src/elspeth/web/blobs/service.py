"""BlobServiceImpl — filesystem-backed blob persistence."""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID, uuid4

from sqlalchemy import Engine, func, select
from sqlalchemy.engine import Connection
from sqlalchemy.exc import SQLAlchemyError

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.blobs.protocol import (
    ALLOWED_MIME_TYPES,
    BLOB_CREATORS,
    BLOB_RUN_LINK_DIRECTIONS,
    BLOB_STATUSES,
    FINALIZE_BLOB_STATUSES,
    AllowedMimeType,
    BlobActiveRunError,
    BlobContentMissingError,
    BlobCreator,
    BlobFinalizationError,
    BlobFinalizationResult,
    BlobIntegrityError,
    BlobNotFoundError,
    BlobQuotaExceededError,
    BlobRecord,
    BlobRunLinkDirection,
    BlobRunLinkRecord,
    BlobStateError,
    FinalizeBlobStatus,
)
from elspeth.web.sessions.models import (
    blob_run_links_table,
    blobs_table,
    composition_states_table,
    runs_table,
)

_T = TypeVar("_T")


def content_hash(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw content bytes.

    This is the shared hash helper referenced by AD-5 and AD-7 in
    docs/plans/rc4.2-ux-remediation/2026-03-30-02-blob-manager-subplan.md.
    When a pipeline reads from a blob, the engine records the raw data
    hash in PayloadStore. Using the same algorithm here guarantees the
    hashes match when the bytes match. Output is SHA-256 hex, 64
    lowercase characters — the canonical form validated by
    ``_validate_finalize_hash`` at the write side and compared via
    ``hmac.compare_digest`` at the read side.
    """
    return hashlib.sha256(data).hexdigest()


def sanitize_filename(filename: str) -> str:
    """Extract a safe basename from a potentially malicious filename.

    Strips all directory components (path traversal protection) and
    rejects empty results or dot-only names.
    """
    sanitized = Path(filename).name
    if not sanitized or sanitized in (".", ".."):
        raise ValueError(f"Invalid filename: {filename!r}")
    # Cap length to leave room for UUID prefix in storage path
    if len(sanitized.encode("utf-8")) > 200:
        # Preserve the extension
        stem = Path(sanitized).stem
        suffix = Path(sanitized).suffix
        max_stem = 200 - len(suffix.encode("utf-8"))
        sanitized = stem.encode("utf-8")[:max_stem].decode("utf-8", errors="ignore") + suffix
    return sanitized


def _source_references_blob(
    source: Any,
    blob_id: str,
    storage_path: str,
) -> bool:
    """Check whether a composition state source references a specific blob.

    Returns True if the source's options contain a matching ``blob_ref``
    OR a ``path``/``file`` that matches the blob's storage_path.

    Tier 1 guards: if source or options have wrong types, that is DB
    corruption — crash immediately rather than silently passing the guard.
    Explicit raises (not ``assert``) because ``python -O`` strips asserts
    and would turn every corruption-detection site here into a silent
    pass-through.
    """
    if source is None:
        return False
    if not isinstance(source, dict):
        raise AuditIntegrityError(f"Tier 1: composition_states.source is {type(source).__name__}, expected dict")
    options = source.get("options")
    if options is None:
        return False
    if not isinstance(options, dict):
        raise AuditIntegrityError(f"Tier 1: composition_states.source.options is {type(options).__name__}, expected dict")
    # Check blob_ref (canonical blob reference)
    if options.get("blob_ref") == blob_id:
        return True
    # Check path/file (a run can read a blob's backing file via plain
    # set_source without blob_ref — the execution service only creates
    # blob_run_links when blob_ref is present)
    return any(options.get(key) == storage_path for key in ("path", "file"))


def _assert_blob_run_same_session(
    conn: Connection,
    *,
    blob_id: str,
    run_id: str,
    caller: str,
) -> None:
    """Offensive guard: blob and run must belong to the same session.

    ``link_blob_to_run()`` is an internal write boundary. A cross-session
    linkage is a caller bug, not user input, so crash with RuntimeError
    before persisting contradictory ownership into ``blob_run_links``.
    """
    blob_session_id = conn.execute(select(blobs_table.c.session_id).where(blobs_table.c.id == blob_id)).scalar()
    if blob_session_id is None:
        raise RuntimeError(f"{caller}: blob_id={blob_id!r} does not exist")

    run_session_id = conn.execute(select(runs_table.c.session_id).where(runs_table.c.id == run_id)).scalar()
    if run_session_id is None:
        raise RuntimeError(f"{caller}: run_id={run_id!r} does not exist")

    if blob_session_id != run_session_id:
        raise RuntimeError(
            f"{caller}: blob_id={blob_id!r} belongs to session "
            f"{blob_session_id!r}, run_id={run_id!r} belongs to session "
            f"{run_session_id!r} — cross-session reference is a contract violation"
        )


def _guard_blob_row_literals(row: Any) -> None:
    """Validate closed-set blob row fields at the DB read boundary."""
    # Tier 1 read guards — BlobRecord's fields are declared as closed
    # Literal types, but the DB can be tampered with via direct SQL
    # or a migration bug. Crash on any value outside the enum so the
    # audit trail never silently returns a record whose static type
    # is a lie. Aligns with the frozenset CHECK constraints in
    # web/sessions/models.py (ck_blobs_status, ck_blobs_created_by)
    # and the MIME allowlist enforced at create_blob().
    #
    # Explicit raise (not ``assert``): ``python -O`` strips asserts,
    # so an optimised interpreter would silently pass a tampered row
    # through these guards. AuditIntegrityError is the contract for
    # Tier 1 DB-corruption conditions and survives ``-O`` execution.
    if row.status not in BLOB_STATUSES:
        raise AuditIntegrityError(f"Tier 1: blobs.status is {row.status!r}, expected one of {sorted(BLOB_STATUSES)}")
    if row.created_by not in BLOB_CREATORS:
        raise AuditIntegrityError(f"Tier 1: blobs.created_by is {row.created_by!r}, expected one of {sorted(BLOB_CREATORS)}")
    if row.mime_type not in ALLOWED_MIME_TYPES:
        raise AuditIntegrityError(f"Tier 1: blobs.mime_type is {row.mime_type!r}, not in the allowed MIME set")


def _row_to_blob_record(row: Any) -> BlobRecord:
    """Convert a blobs row into a guarded BlobRecord."""
    _guard_blob_row_literals(row)
    return BlobRecord(
        id=UUID(row.id),
        session_id=UUID(row.session_id),
        filename=row.filename,
        mime_type=row.mime_type,
        size_bytes=row.size_bytes,
        content_hash=row.content_hash,
        storage_path=row.storage_path,
        created_at=row.created_at,
        created_by=row.created_by,
        source_description=row.source_description,
        status=row.status,
    )


class BlobServiceImpl:
    """Filesystem-backed blob service.

    Follows the same async-over-sync pattern as SessionServiceImpl:
    all public methods are async, database I/O runs in a thread pool
    executor via _run_sync().
    """

    def __init__(self, engine: Engine, data_dir: Path, max_storage_per_session: int = 500 * 1024 * 1024) -> None:
        self._engine = engine
        self._data_dir = data_dir.expanduser().resolve()
        self._max_storage_per_session = max_storage_per_session

    async def _run_sync(self, func: Callable[[], _T]) -> _T:
        return await run_sync_in_worker(func)

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _blob_dir(self, session_id: str) -> Path:
        return self._data_dir / "blobs" / session_id

    def _storage_path(self, session_id: str, blob_id: str, filename: str) -> Path:
        return self._blob_dir(session_id) / f"{blob_id}_{filename}"

    def _row_to_record(self, row: Any) -> BlobRecord:
        return _row_to_blob_record(row)

    def _row_to_link_record(self, row: Any) -> BlobRunLinkRecord:
        # Tier 1 read guard — mirrors the ck_blob_run_links_direction
        # CHECK constraint.  A row with a bogus direction would leave
        # BlobRunLinkRecord.direction (typed BlobRunLinkDirection)
        # carrying a value outside its Literal set.  Explicit raise (not
        # ``assert``) so the guard survives ``python -O``.
        if row.direction not in BLOB_RUN_LINK_DIRECTIONS:
            raise AuditIntegrityError(
                f"Tier 1: blob_run_links.direction is {row.direction!r}, expected one of {sorted(BLOB_RUN_LINK_DIRECTIONS)}"
            )
        return BlobRunLinkRecord(
            blob_id=UUID(row.blob_id),
            run_id=UUID(row.run_id),
            direction=row.direction,
        )

    async def create_blob(
        self,
        session_id: UUID,
        filename: str,
        content: bytes,
        mime_type: AllowedMimeType,
        created_by: BlobCreator = "user",
        source_description: str | None = None,
    ) -> BlobRecord:
        """Create a blob from content bytes."""
        # Programmer-bug guards on Literal-typed parameters.  Explicit
        # raises (not ``assert``) so the guard survives ``python -O`` —
        # mirrors the RuntimeError at ``link_blob_to_run`` (direction) and
        # ``_finalize_blob_sync`` (status).
        if created_by not in BLOB_CREATORS:
            raise RuntimeError(f"Invalid created_by {created_by!r} — must be one of {sorted(BLOB_CREATORS)}")
        if mime_type not in ALLOWED_MIME_TYPES:
            raise RuntimeError(f"Invalid mime_type {mime_type!r} — not in the allowed MIME set")
        safe_filename = sanitize_filename(filename)
        blob_id = str(uuid4())
        session_id_str = str(session_id)
        file_hash = content_hash(content)
        storage = self._storage_path(session_id_str, blob_id, safe_filename)

        def _sync() -> BlobRecord:
            # Write file first (before DB transaction)
            storage.parent.mkdir(parents=True, exist_ok=True)
            storage.write_bytes(content)

            # Single transaction: quota check + insert (atomic, no TOCTOU)
            now = self._now()
            try:
                with self._engine.begin() as conn:
                    # Quota check inside the write transaction
                    current_total = conn.execute(
                        select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == session_id_str)
                    ).scalar()
                    # COALESCE guarantees an exact int; bool/subclasses or any
                    # other type are Tier 1 anomalies. Explicit raise (not
                    # assert) so the guard survives -O.
                    if type(current_total) is not int:
                        raise AuditIntegrityError(f"Tier 1: COALESCE(SUM) returned {type(current_total).__name__}, expected int")
                    if current_total + len(content) > self._max_storage_per_session:
                        raise BlobQuotaExceededError(
                            session_id_str,
                            current_bytes=current_total,
                            limit_bytes=self._max_storage_per_session,
                        )

                    conn.execute(
                        blobs_table.insert().values(
                            id=blob_id,
                            session_id=session_id_str,
                            filename=safe_filename,
                            mime_type=mime_type,
                            size_bytes=len(content),
                            content_hash=file_hash,
                            storage_path=str(storage),
                            created_at=now,
                            created_by=created_by,
                            source_description=source_description,
                            status="ready",
                        )
                    )
            except Exception:
                # Clean up file on quota exceeded or any DB failure
                if storage.exists():
                    storage.unlink()
                raise

            return BlobRecord(
                id=UUID(blob_id),
                session_id=session_id,
                filename=safe_filename,
                mime_type=mime_type,
                size_bytes=len(content),
                content_hash=file_hash,
                storage_path=str(storage),
                created_at=now,
                created_by=created_by,
                source_description=source_description,
                status="ready",
            )

        return await self._run_sync(_sync)

    async def create_pending_blob(
        self,
        session_id: UUID,
        filename: str,
        mime_type: AllowedMimeType,
        created_by: BlobCreator = "pipeline",
        source_description: str | None = None,
    ) -> BlobRecord:
        """Reserve a pending output blob."""
        # Programmer-bug guard on Literal-typed parameter.  Explicit raise
        # so the check survives ``python -O`` (mirrors create_blob()).
        if created_by not in BLOB_CREATORS:
            raise RuntimeError(f"Invalid created_by {created_by!r} — must be one of {sorted(BLOB_CREATORS)}")
        safe_filename = sanitize_filename(filename)
        blob_id = str(uuid4())
        session_id_str = str(session_id)
        storage = self._storage_path(session_id_str, blob_id, safe_filename)

        def _sync() -> BlobRecord:
            # Ensure directory exists (file will be written by sink later)
            storage.parent.mkdir(parents=True, exist_ok=True)

            now = self._now()
            with self._engine.begin() as conn:
                conn.execute(
                    blobs_table.insert().values(
                        id=blob_id,
                        session_id=session_id_str,
                        filename=safe_filename,
                        mime_type=mime_type,
                        size_bytes=0,
                        content_hash=None,
                        storage_path=str(storage),
                        created_at=now,
                        created_by=created_by,
                        source_description=source_description,
                        status="pending",
                    )
                )

            return BlobRecord(
                id=UUID(blob_id),
                session_id=session_id,
                filename=safe_filename,
                mime_type=mime_type,
                size_bytes=0,
                content_hash=None,
                storage_path=str(storage),
                created_at=now,
                created_by=created_by,
                source_description=source_description,
                status="pending",
            )

        return await self._run_sync(_sync)

    async def finalize_blob(
        self,
        blob_id: UUID,
        status: FinalizeBlobStatus,
        size_bytes: int | None = None,
        content_hash: str | None = None,
    ) -> BlobRecord:
        """Update a pending blob to ready or error after execution."""
        blob_id_str = str(blob_id)
        # Runtime guard for dynamic callers — the Literal narrowing gives
        # static callers the correct shape, but the Protocol boundary is
        # still called by code that mypy may not fully verify (tests,
        # factory-constructed services).  Keep the check as a belt.
        if status not in FINALIZE_BLOB_STATUSES:
            raise RuntimeError(f"Invalid finalize status '{status}' — must be one of {sorted(FINALIZE_BLOB_STATUSES)}")

        def _sync() -> BlobRecord:
            with self._engine.begin() as conn:
                row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
                if row is None:
                    raise BlobNotFoundError(blob_id_str)

                if row.status != "pending":
                    raise BlobStateError(
                        blob_id_str,
                        message=f"Cannot finalize blob {blob_id_str} — status is '{row.status}', expected 'pending'",
                    )
                # Hash-format validation runs after the state check so a
                # callers confused by a stale blob hear about the lifecycle
                # problem first.  See _validate_finalize_hash() docstring.
                _validate_finalize_hash(blob_id_str, status, content_hash)

                updates: dict[str, Any] = {"status": status}
                if size_bytes is not None:
                    updates["size_bytes"] = size_bytes
                if content_hash is not None:
                    updates["content_hash"] = content_hash

                conn.execute(blobs_table.update().where(blobs_table.c.id == blob_id_str).values(**updates))

                updated = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
                if updated is None:
                    raise RuntimeError(f"Blob {blob_id_str} vanished during finalize — concurrent deletion?")
                return self._row_to_record(updated)

        return await self._run_sync(_sync)

    async def get_blob(self, blob_id: UUID) -> BlobRecord:
        """Get blob metadata."""
        blob_id_str = str(blob_id)

        def _sync() -> BlobRecord:
            with self._engine.connect() as conn:
                row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
                if row is None:
                    raise BlobNotFoundError(blob_id_str)
                return self._row_to_record(row)

        return await self._run_sync(_sync)

    async def list_blobs(
        self,
        session_id: UUID,
        limit: int | None = 50,
        offset: int = 0,
    ) -> list[BlobRecord]:
        """List blobs for a session, newest first."""
        session_id_str = str(session_id)

        def _sync() -> list[BlobRecord]:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(blobs_table)
                    .where(blobs_table.c.session_id == session_id_str)
                    .order_by(blobs_table.c.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                ).fetchall()
                return [self._row_to_record(r) for r in rows]

        return await self._run_sync(_sync)

    async def delete_blob(self, blob_id: UUID) -> None:
        """Delete blob metadata and backing file."""
        blob_id_str = str(blob_id)

        def _sync() -> None:
            with self._engine.begin() as conn:
                # Check blob exists
                row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
                if row is None:
                    raise BlobNotFoundError(blob_id_str)

                # Active-run guard (two checks):
                #
                # 1. Explicit link: blob_run_links already points at an active run.
                active_link = conn.execute(
                    select(blob_run_links_table)
                    .join(
                        runs_table,
                        blob_run_links_table.c.run_id == runs_table.c.id,
                    )
                    .where(blob_run_links_table.c.blob_id == blob_id_str)
                    .where(runs_table.c.status.in_(["pending", "running"]))
                ).first()
                if active_link is not None:
                    raise BlobActiveRunError(blob_id_str, run_id=active_link.run_id)

                # 2. Pre-link window: _execute_locked() creates the run record
                #    before link_blob_to_run() inserts the blob_run_links row.
                #    During that gap the explicit-link check above sees nothing,
                #    but the backing file is about to be needed.
                #
                #    Scoped to THIS blob: join runs → composition_states and
                #    check whether the active run's source references this
                #    blob via blob_ref OR via a path/file that matches this
                #    blob's storage_path.  Runs whose source doesn't touch
                #    this blob must not block unrelated blob deletions.
                active_run = conn.execute(
                    select(runs_table.c.id, composition_states_table.c.source)
                    .join(
                        composition_states_table,
                        runs_table.c.state_id == composition_states_table.c.id,
                    )
                    .where(runs_table.c.session_id == row.session_id)
                    .where(runs_table.c.status.in_(["pending", "running"]))
                ).first()
                if active_run is not None and _source_references_blob(active_run.source, blob_id_str, row.storage_path):
                    raise BlobActiveRunError(blob_id_str, run_id=active_run.id)

                # Delete backing file first — orphaned DB row is recoverable,
                # orphaned file with no metadata is not
                storage = Path(row.storage_path)
                if storage.exists():
                    storage.unlink()

                # Delete metadata (cascades to blob_run_links)
                conn.execute(blobs_table.delete().where(blobs_table.c.id == blob_id_str))

        await self._run_sync(_sync)

    async def read_blob_content(self, blob_id: UUID) -> bytes:
        """Read the raw content of a blob.

        Enforces two invariants before returning bytes:

        1. **Lifecycle guard**: only ``ready`` blobs are readable.
           Pending blobs have no finalized content; error blobs
           represent failed runs whose output is not trustworthy.

        2. **Integrity verification**: a ready blob must still have a
           backing file on disk, and its bytes must match the stored
           ``content_hash``. Missing bytes or hash mismatch indicate
           filesystem corruption, silent data loss, tampering, or a
           write-path bug — all Tier 1 anomalies.
        """
        blob_id_str = str(blob_id)

        def _sync() -> bytes:
            with self._engine.connect() as conn:
                row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
                if row is None:
                    raise BlobNotFoundError(blob_id_str)

                # Lifecycle guard — only ready blobs have finalized content
                if row.status != "ready":
                    raise BlobStateError(
                        blob_id_str,
                        message=f"Cannot read blob {blob_id_str} — status is '{row.status}', expected 'ready'",
                    )

                storage = Path(row.storage_path)
                if not storage.exists():
                    raise BlobContentMissingError(blob_id_str, storage_path=row.storage_path)

                data = storage.read_bytes()

                # Integrity verification — Tier 1: our data must be pristine.
                # A ready blob must always have a content_hash — it is set
                # by create_blob() and required by _finalize_blob_sync()
                # when transitioning to ready.  NULL here is a DB anomaly.
                # Explicit raise so the guard survives ``python -O``.
                if row.content_hash is None:
                    raise AuditIntegrityError(
                        f"Tier 1: ready blob {blob_id_str} has NULL content_hash — DB integrity anomaly, cannot verify"
                    )
                actual = content_hash(data)
                if not hmac.compare_digest(actual, row.content_hash):
                    raise BlobIntegrityError(blob_id_str, expected=row.content_hash, actual=actual)

                return data

        return await self._run_sync(_sync)

    async def link_blob_to_run(
        self,
        blob_id: UUID,
        run_id: UUID,
        direction: BlobRunLinkDirection,
    ) -> None:
        """Record a blob-to-run linkage."""
        if direction not in BLOB_RUN_LINK_DIRECTIONS:
            raise RuntimeError(f"Invalid link direction '{direction}' — must be one of {sorted(BLOB_RUN_LINK_DIRECTIONS)}")

        def _sync() -> None:
            with self._engine.begin() as conn:
                _assert_blob_run_same_session(
                    conn,
                    blob_id=str(blob_id),
                    run_id=str(run_id),
                    caller="BlobServiceImpl.link_blob_to_run",
                )
                conn.execute(
                    blob_run_links_table.insert().values(
                        blob_id=str(blob_id),
                        run_id=str(run_id),
                        direction=direction,
                    )
                )

        await self._run_sync(_sync)

    async def get_blob_run_links(
        self,
        blob_id: UUID,
    ) -> list[BlobRunLinkRecord]:
        """Get all run links for a blob."""
        blob_id_str = str(blob_id)

        def _sync() -> list[BlobRunLinkRecord]:
            with self._engine.connect() as conn:
                rows = conn.execute(select(blob_run_links_table).where(blob_run_links_table.c.blob_id == blob_id_str)).fetchall()
                return [self._row_to_link_record(r) for r in rows]

        return await self._run_sync(_sync)

    # Per-blob operational errors that should not abort the finalization
    # loop.  BlobStateError covers status-guard conditions (blob already
    # finalized by a concurrent call).  RuntimeError is deliberately
    # excluded — it covers the Tier 1 "blob vanished mid-transaction"
    # anomaly, which must propagate.  Programmer bugs (TypeError,
    # AttributeError, AssertionError) also propagate per offensive
    # programming policy.
    _PER_BLOB_SUPPRESSED: tuple[type[BaseException], ...] = (
        BlobNotFoundError,
        BlobStateError,
        OSError,
        SQLAlchemyError,
    )

    async def finalize_run_output_blobs(
        self,
        run_id: UUID,
        success: bool,
    ) -> BlobFinalizationResult:
        """Finalize pending output blobs for a completed/failed run.

        On success: compute content_hash and size_bytes from the backing
        file, set status to 'ready'. If the file wasn't written, mark
        as 'error'.
        On failure: delete the backing file (if any) and set status to
        'error', leaving size/hash as None.  This ensures the filesystem
        matches the DB metadata and prevents orphaned files from escaping
        quota accounting.

        Processes each blob independently — a per-blob operational error
        does not abort finalization of remaining blobs.  Failed blobs are
        transitioned to ``error`` status on a best-effort basis.

        Returns a BlobFinalizationResult with both successfully finalized
        blobs and per-blob error records.
        """
        run_id_str = str(run_id)

        def _sync() -> BlobFinalizationResult:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    select(blobs_table)
                    .join(
                        blob_run_links_table,
                        blob_run_links_table.c.blob_id == blobs_table.c.id,
                    )
                    .where(blob_run_links_table.c.run_id == run_id_str)
                    .where(blob_run_links_table.c.direction == "output")
                    .where(blobs_table.c.status == "pending")
                ).fetchall()

            finalized: list[BlobRecord] = []
            errors: list[BlobFinalizationError] = []
            for row in rows:
                blob_id = UUID(row.id)
                try:
                    if success:
                        storage = Path(row.storage_path)
                        if storage.exists():
                            file_bytes = storage.read_bytes()
                            try:
                                record = self._finalize_blob_sync(
                                    blob_id,
                                    "ready",
                                    size_bytes=len(file_bytes),
                                    content_hash_val=content_hash(file_bytes),
                                )
                            except BlobQuotaExceededError:
                                # Run succeeded but this blob would breach the
                                # session quota — mark as error so the run
                                # finalization isn't aborted entirely.
                                # Delete the backing file to prevent untracked
                                # disk growth from repeated over-quota outputs.
                                if storage.exists():
                                    storage.unlink()
                                record = self._finalize_blob_sync(blob_id, "error")
                        else:
                            record = self._finalize_blob_sync(blob_id, "error")
                    else:
                        # Run failed — delete the backing file so the
                        # filesystem matches the DB metadata (size_bytes=0,
                        # content_hash=None).  Without this, repeated
                        # failed runs can grow disk usage without bound
                        # while quota accounting sees only zero-byte
                        # error rows.
                        failed_storage = Path(row.storage_path)
                        if failed_storage.exists():
                            failed_storage.unlink()
                        record = self._finalize_blob_sync(blob_id, "error")
                    finalized.append(record)
                except self._PER_BLOB_SUPPRESSED as exc:
                    # Best-effort: transition the failed blob to "error"
                    # so it doesn't remain permanently pending.  The WHERE
                    # on status='pending' makes this a no-op if already
                    # finalized or deleted.
                    recovery_exc: BaseException | None = None
                    try:
                        with self._engine.begin() as err_conn:
                            err_conn.execute(
                                blobs_table.update()
                                .where(blobs_table.c.id == str(blob_id))
                                .where(blobs_table.c.status == "pending")
                                .values(status="error")
                            )
                    except (SQLAlchemyError, OSError) as rec_exc:
                        # Narrow to DB/IO faults — programmer bugs
                        # (TypeError, AttributeError, AssertionError) must
                        # propagate per offensive-programming policy.
                        # The blob stays pending; we record the recovery
                        # failure alongside the primary exception so the
                        # audit trail carries both causes.
                        recovery_exc = rec_exc
                    errors.append(
                        BlobFinalizationError(
                            blob_id=blob_id,
                            exc_type=type(exc).__name__,
                            detail=str(exc),
                        )
                    )
                    if recovery_exc is not None:
                        errors.append(
                            BlobFinalizationError(
                                blob_id=blob_id,
                                exc_type=f"RecoveryFailed[{type(recovery_exc).__name__}]",
                                detail=str(recovery_exc),
                            )
                        )
            return BlobFinalizationResult(finalized=finalized, errors=errors)

        return await self._run_sync(_sync)

    async def copy_blobs_for_fork(
        self,
        source_session_id: UUID,
        target_session_id: UUID,
    ) -> dict[UUID, BlobRecord]:
        """Copy all ready blobs from source session to target session.

        Pre-checks total source blob size against the target session's
        quota before copying any files. This eliminates partial-write
        scenarios — either all blobs are copied or none are.

        On any failure during the copy loop, cleans up files already
        written before re-raising.
        """
        source_blobs = await self.list_blobs(source_session_id, limit=None)
        ready_blobs = [b for b in source_blobs if b.status == "ready"]

        if not ready_blobs:
            return {}

        # Pre-check: will the total source blob size fit in the target quota?
        total_source_bytes = sum(b.size_bytes for b in ready_blobs)
        target_session_id_str = str(target_session_id)

        def _check_quota() -> int:
            with self._engine.connect() as conn:
                current = conn.execute(
                    select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == target_session_id_str)
                ).scalar()
                # COALESCE guarantees an exact int; bool/subclasses or any
                # other type are Tier 1 anomalies. Explicit raise so the
                # guard survives ``python -O``.
                if type(current) is not int:
                    raise AuditIntegrityError(f"Tier 1: COALESCE(SUM) returned {type(current).__name__}, expected int")
                return current

        current_usage = await self._run_sync(_check_quota)
        if current_usage + total_source_bytes > self._max_storage_per_session:
            raise BlobQuotaExceededError(
                target_session_id_str,
                current_bytes=current_usage,
                limit_bytes=self._max_storage_per_session,
            )

        # Copy blobs — clean up partial writes on any failure.
        # Build old_id → new_blob mapping for source reference rewriting.
        blob_map: dict[UUID, BlobRecord] = {}
        copied: list[BlobRecord] = []
        try:
            for blob in ready_blobs:
                content = await self.read_blob_content(blob.id)
                new_blob = await self.create_blob(
                    session_id=target_session_id,
                    filename=blob.filename,
                    content=content,
                    mime_type=blob.mime_type,
                    created_by=blob.created_by,
                    source_description=f"copied from session fork (original: {blob.id})",
                )
                copied.append(new_blob)
                blob_map[blob.id] = new_blob
        except Exception as primary_exc:
            # Clean up both files AND database rows for any blobs already
            # committed. create_blob() commits each blob atomically, so
            # without this cleanup the forked session would have "ready"
            # blob metadata pointing at files we're about to delete.
            #
            # Cleanup failures must NOT be silently swallowed: a failed
            # delete_blob leaves an orphan DB row in the target session
            # that auditors would interpret as a successfully copied blob.
            # Mirror the RecoveryFailed[...] convention used by
            # ``BlobServiceImpl.finalize_run_output_blobs`` (the per-blob
            # error-record path inside its nested ``_sync`` closure): narrow
            # the catch to (SQLAlchemyError, OSError) — programmer bugs must
            # propagate — collect every cleanup failure, and attach them
            # as notes on primary_exc.  The fallback file unlink stays for
            # disk-quota recovery, but the DB-row orphan is now visible
            # to operators reading the traceback.  Bare `raise` re-raises
            # primary_exc (sys.exc_info() reverts after each nested except),
            # preserving the original copy failure as the headline.
            cleanup_failures: list[tuple[UUID, BaseException]] = []
            for written_blob in copied:
                try:
                    await self.delete_blob(written_blob.id)
                except (SQLAlchemyError, OSError) as cleanup_exc:
                    cleanup_failures.append((written_blob.id, cleanup_exc))
                    storage = Path(written_blob.storage_path)
                    if storage.exists():
                        storage.unlink(missing_ok=True)
            for orphan_id, recorded_exc in cleanup_failures:
                primary_exc.add_note(
                    f"RecoveryFailed[{type(recorded_exc).__name__}]: "
                    f"could not delete partially-copied blob {orphan_id} from "
                    f"target session {target_session_id_str} "
                    f"({recorded_exc}). "
                    f"Storage file was unlinked, but the DB row remains and "
                    f"will appear as a 'ready' blob in the target session — "
                    f"manual cleanup of blobs.id={orphan_id} required."
                )
            raise

        return blob_map

    def _finalize_blob_sync(
        self,
        blob_id: UUID,
        status: FinalizeBlobStatus,
        size_bytes: int | None = None,
        content_hash_val: str | None = None,
    ) -> BlobRecord:
        """Synchronous single-blob finalize for use inside _run_sync closures."""
        blob_id_str = str(blob_id)
        # Invalid status is a programmer bug at a Protocol boundary, not a
        # per-blob operational condition.  RuntimeError propagates past
        # _PER_BLOB_SUPPRESSED so the loop in finalize_run_output_blobs
        # crashes loudly instead of silently converting the caller's typo
        # into an "error" record the auditor cannot distinguish from a
        # genuine run failure.  Mirrors the RuntimeError in finalize_blob().
        if status not in FINALIZE_BLOB_STATUSES:
            raise RuntimeError(f"Invalid finalize status '{status}' — must be one of {sorted(FINALIZE_BLOB_STATUSES)}")
        # Single source of truth for the ready-requires-valid-hash rule.
        # See _validate_finalize_hash() docstring.
        _validate_finalize_hash(blob_id_str, status, content_hash_val)

        with self._engine.begin() as conn:
            row = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
            if row is None:
                raise BlobNotFoundError(blob_id_str)
            if row.status != "pending":
                raise BlobStateError(
                    blob_id_str,
                    message=f"Cannot finalize blob {blob_id_str} — status is '{row.status}', expected 'pending'",
                )

            # Enforce quota when finalizing with a real size — pending blobs
            # were reserved at size_bytes=0, so this is the first time the
            # actual size is known.  Without this check, pipeline-generated
            # output could bypass the per-session storage cap entirely.
            if status == "ready" and size_bytes is not None and size_bytes > 0:
                session_id_str = row.session_id
                current_total = conn.execute(
                    select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(
                        blobs_table.c.session_id == session_id_str,
                        blobs_table.c.id != blob_id_str,
                    )
                ).scalar()
                # COALESCE guarantees an exact int; bool/subclasses or any
                # other type are Tier 1 anomalies. Explicit raise so the
                # guard survives ``python -O``.
                if type(current_total) is not int:
                    raise AuditIntegrityError(f"Tier 1: COALESCE(SUM) returned {type(current_total).__name__}, expected int")
                if current_total + size_bytes > self._max_storage_per_session:
                    raise BlobQuotaExceededError(
                        session_id_str,
                        current_bytes=current_total,
                        limit_bytes=self._max_storage_per_session,
                    )

            updates: dict[str, Any] = {"status": status}
            if size_bytes is not None:
                updates["size_bytes"] = size_bytes
            if content_hash_val is not None:
                updates["content_hash"] = content_hash_val

            conn.execute(blobs_table.update().where(blobs_table.c.id == blob_id_str).values(**updates))
            updated = conn.execute(select(blobs_table).where(blobs_table.c.id == blob_id_str)).first()
            if updated is None:
                raise RuntimeError(f"Blob {blob_id_str} vanished during finalize — concurrent deletion?")
            return self._row_to_record(updated)


# SHA-256 hex digest: exactly 64 lowercase hex characters.  Must match
# FilesystemPayloadStore's validator (core/payload_store.py) — a blob
# whose content_hash round-trips through the audit trail must use the
# same canonical form everywhere.  Used with ``fullmatch`` (NOT
# ``match``) because Python's ``$`` anchor matches at end-of-string OR
# just before a final ``\n``, so the naive ``^[a-f0-9]{64}$`` pattern
# would accept ``"a" * 64 + "\n"`` — letting a newline-terminated hash
# slip past the pre-check and land at the DB CHECK as an opaque
# IntegrityError rather than the structured BlobStateError this
# validator is supposed to raise.
_SHA256_HEX_PATTERN = re.compile(r"[a-f0-9]{64}")


def _validate_finalize_hash(
    blob_id_str: str,
    status: FinalizeBlobStatus,
    content_hash_val: str | None,
) -> None:
    """Service-layer pre-check for the ``ready`` content_hash invariant.

    This is the FIRST of two walls enforcing the Tier-1 integrity
    contract that makes ``read_blob_content`` verifiable (AD-5/AD-7 in
    docs/plans/rc4.2-ux-remediation/2026-03-30-02-blob-manager-subplan.md).
    A ``ready`` blob MUST carry a SHA-256 hex digest; before this
    pre-check existed, a caller could finalize with a bogus string like
    ``"abc123"`` and the DB would happily store it, leaving a ``ready``
    row whose hash cannot be produced by any real bytes on disk.

    Division of responsibility
    --------------------------
    This function is the SERVICE-LAYER pre-check. It runs on every
    ``finalize_blob`` / ``_finalize_blob_sync`` write-path call and
    raises :class:`BlobStateError` — a structured, caller-friendly
    diagnostic — before any SQL is issued. The DB-level CHECK
    constraint ``ck_blobs_ready_hash`` is the AUTHORITATIVE guard: it
    closes the same invariant for any writer that bypasses this service
    (direct SQL or an ORM call path that skips finalize). If these two
    guards disagree, the DB CHECK wins and the service pre-check is the
    bug.

    Keeping both guards means a service regression surfaces as a clean
    BlobStateError at the write-path entry point (easy to debug),
    while a writer that skips the service still cannot corrupt the
    audit trail. The shape rule is kept in agreement between the two
    sites by design — the current session schema declares the DB-side
    guard, and the tests in
    ``tests/unit/web/blobs/test_service.py::TestBlobsReadyHashDBConstraint``
    pin the DB guard independently of this one.
    """
    if status != "ready":
        return
    if content_hash_val is None:
        raise BlobStateError(
            blob_id_str,
            message=f"Tier 1: cannot finalize blob {blob_id_str} as 'ready' without content_hash — audit integrity requires a hash",
        )
    # ``fullmatch`` (not ``match``) — see the _SHA256_HEX_PATTERN comment
    # above for why ``^...$`` + ``match`` admits trailing newlines.
    if not _SHA256_HEX_PATTERN.fullmatch(content_hash_val):
        raise BlobStateError(
            blob_id_str,
            message=f"Tier 1: content_hash must be 64 lowercase hex characters (SHA-256), got {content_hash_val!r}",
        )
