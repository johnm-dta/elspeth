"""Tests for BlobServiceImpl — audit-critical blob persistence and lifecycle.

Security boundaries tested:
- Content hash integrity (AD-5/AD-7: hash must match for lineage verification)
- Session-scoped isolation (blobs cannot leak across sessions)
- Active-run deletion guard (cannot destroy evidence during a live run)
- Filename sanitization (path traversal defense at the storage layer)
- Status lifecycle (pending -> ready/error only; no backwards transitions)
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy.pool import StaticPool

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.blobs.protocol import (
    BlobActiveRunError,
    BlobNotFoundError,
)
from elspeth.web.blobs.service import (
    BlobServiceImpl,
    content_hash,
    sanitize_filename,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import sessions_table
from elspeth.web.sessions.schema import initialize_session_schema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_engine():
    """In-memory SQLite engine with all session tables created."""
    engine = create_session_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    initialize_session_schema(engine)
    return engine


@pytest.fixture()
def session_id(db_engine) -> UUID:
    """Insert a session row and return its ID — blobs have FK to sessions."""
    sid = str(uuid4())
    now = datetime.now(UTC)
    with db_engine.begin() as conn:
        conn.execute(
            sessions_table.insert().values(
                id=sid,
                user_id="test-user",
                auth_provider_type="local",
                title="Test Session",
                created_at=now,
                updated_at=now,
            )
        )
    return UUID(sid)


@pytest.fixture()
def blob_service(db_engine, tmp_path) -> BlobServiceImpl:
    """BlobServiceImpl backed by the shared engine and a temp directory."""
    return BlobServiceImpl(db_engine, tmp_path)


# ---------------------------------------------------------------------------
# sanitize_filename — path traversal defense
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    """B5: filename sanitization prevents path traversal at the storage layer."""

    def test_path_traversal_strips_directory_components(self) -> None:
        assert sanitize_filename("../../etc/passwd") == "passwd"

    def test_absolute_path_strips_to_basename(self) -> None:
        assert sanitize_filename("/absolute/path/file.csv") == "file.csv"

    def test_normal_filename_passes_through(self) -> None:
        assert sanitize_filename("normal.csv") == "normal.csv"

    def test_dot_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid filename"):
            sanitize_filename(".")

    def test_dotdot_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid filename"):
            sanitize_filename("..")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid filename"):
            sanitize_filename("")

    def test_long_filename_truncated(self) -> None:
        long_name = "a" * 300 + ".csv"
        result = sanitize_filename(long_name)
        assert len(result.encode("utf-8")) <= 200


# ---------------------------------------------------------------------------
# content_hash — audit integrity
# ---------------------------------------------------------------------------


class TestContentHash:
    """AD-5/AD-7: content hash must be SHA-256 for lineage verification."""

    def test_known_input(self) -> None:
        expected = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        assert content_hash(b"hello") == expected

    def test_stability(self) -> None:
        data = b"audit-critical-content"
        assert content_hash(data) == content_hash(data)

    def test_empty_bytes(self) -> None:
        expected = hashlib.sha256(b"").hexdigest()
        assert content_hash(b"") == expected


# ---------------------------------------------------------------------------
# create_blob + read_blob_content — round-trip integrity
# ---------------------------------------------------------------------------


class TestCreateAndRead:
    """Blob creation writes to filesystem and DB; read returns identical bytes."""

    @pytest.mark.asyncio
    async def test_create_blob_and_read(self, blob_service, session_id, tmp_path) -> None:
        content = b"col1,col2\na,b\nc,d"
        record = await blob_service.create_blob(
            session_id=session_id,
            filename="data.csv",
            content=content,
            mime_type="text/csv",
            created_by="user",
        )

        # Record fields
        assert isinstance(record.id, UUID)
        assert record.session_id == session_id
        assert record.filename == "data.csv"
        assert record.mime_type == "text/csv"
        assert record.size_bytes == len(content)
        assert record.status == "ready"
        assert record.created_by == "user"

        # Read back content
        read_back = await blob_service.read_blob_content(record.id)
        assert read_back == content

        # File exists on disk
        assert Path(record.storage_path).exists()

    @pytest.mark.asyncio
    async def test_create_blob_with_relative_data_dir_stores_absolute_storage_path(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Blob storage paths are internal paths, not data-dir-relative source paths."""
        monkeypatch.chdir(tmp_path)
        blob_service = BlobServiceImpl(db_engine, Path("data"))

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="tickets.csv",
            content=b"ticket_id\nT-001\n",
            mime_type="text/csv",
            created_by="user",
        )

        storage_path = Path(record.storage_path)
        assert storage_path.is_absolute()
        assert storage_path == tmp_path / "data" / "blobs" / str(session_id) / f"{record.id}_tickets.csv"
        assert storage_path.exists()

    @pytest.mark.asyncio
    async def test_create_blob_stores_correct_hash(self, blob_service, session_id) -> None:
        """AD-7: stored hash must match content_hash() for the same bytes."""
        content = b"audit-trail-integrity-check"
        record = await blob_service.create_blob(
            session_id=session_id,
            filename="audit.txt",
            content=content,
            mime_type="text/plain",
            created_by="user",
        )
        assert record.content_hash == content_hash(content)


# ---------------------------------------------------------------------------
# list_blobs — session-scoped isolation
# ---------------------------------------------------------------------------


class TestListBlobs:
    """Session scoping: blobs from one session must not leak into another."""

    @pytest.mark.asyncio
    async def test_list_blobs_returns_session_scoped(self, blob_service, db_engine) -> None:
        now = datetime.now(UTC)
        s1_id = UUID(str(uuid4()))
        s2_id = UUID(str(uuid4()))

        with db_engine.begin() as conn:
            for sid, uid, title in [
                (str(s1_id), "user-a", "Session 1"),
                (str(s2_id), "user-b", "Session 2"),
            ]:
                conn.execute(
                    sessions_table.insert().values(
                        id=sid,
                        user_id=uid,
                        auth_provider_type="local",
                        title=title,
                        created_at=now,
                        updated_at=now,
                    )
                )

        await blob_service.create_blob(
            session_id=s1_id,
            filename="s1.csv",
            content=b"session-1",
            mime_type="text/csv",
            created_by="user",
        )
        await blob_service.create_blob(
            session_id=s2_id,
            filename="s2.csv",
            content=b"session-2",
            mime_type="text/csv",
            created_by="user",
        )

        s1_blobs = await blob_service.list_blobs(s1_id)
        s2_blobs = await blob_service.list_blobs(s2_id)

        assert len(s1_blobs) == 1
        assert s1_blobs[0].filename == "s1.csv"
        assert len(s2_blobs) == 1
        assert s2_blobs[0].filename == "s2.csv"


# ---------------------------------------------------------------------------
# delete_blob — file cleanup and active-run guard
# ---------------------------------------------------------------------------


class TestDeleteBlob:
    """Deletion removes file and record; active-run guard prevents evidence destruction."""

    @pytest.mark.asyncio
    async def test_delete_blob_removes_file_and_record(self, blob_service, session_id) -> None:
        from pathlib import Path

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="delete-me.csv",
            content=b"temporary",
            mime_type="text/csv",
            created_by="user",
        )

        storage = Path(record.storage_path)
        assert storage.exists()

        await blob_service.delete_blob(record.id)

        assert not storage.exists()
        with pytest.raises(BlobNotFoundError):
            await blob_service.get_blob(record.id)

    @pytest.mark.asyncio
    async def test_delete_blob_rejects_when_active_run_linked(self, blob_service, session_id, db_engine) -> None:
        """Active-run guard: cannot delete a blob that is evidence for a live run."""
        from elspeth.web.sessions.models import (
            blob_run_links_table,
            composition_states_table,
            runs_table,
        )

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="evidence.csv",
            content=b"important",
            mime_type="text/csv",
            created_by="user",
        )

        # Insert a composition state (runs FK to composition_states)
        state_id = str(uuid4())
        session_id_str = str(session_id)
        run_id = str(uuid4())

        with db_engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=session_id_str,
                    version=1,
                    is_valid=True,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=run_id,
                    session_id=session_id_str,
                    state_id=state_id,
                    status="running",
                    started_at=datetime(2026, 1, 1, tzinfo=UTC),
                    rows_processed=0,
                    rows_failed=0,
                )
            )
            conn.execute(
                blob_run_links_table.insert().values(
                    blob_id=str(record.id),
                    run_id=run_id,
                    direction="input",
                )
            )

        with pytest.raises(BlobActiveRunError):
            await blob_service.delete_blob(record.id)

    @pytest.mark.asyncio
    async def test_delete_blob_allows_when_completed_run_linked(self, blob_service, session_id, db_engine) -> None:
        """Completed runs do not block deletion — evidence is already recorded."""
        from elspeth.web.sessions.models import (
            blob_run_links_table,
            composition_states_table,
            runs_table,
        )

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="done.csv",
            content=b"finished",
            mime_type="text/csv",
            created_by="user",
        )

        state_id = str(uuid4())
        session_id_str = str(session_id)
        run_id = str(uuid4())

        with db_engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=session_id_str,
                    version=1,
                    is_valid=True,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=run_id,
                    session_id=session_id_str,
                    state_id=state_id,
                    status="completed",
                    started_at=datetime(2026, 1, 1, tzinfo=UTC),
                    rows_processed=10,
                    rows_failed=0,
                )
            )
            conn.execute(
                blob_run_links_table.insert().values(
                    blob_id=str(record.id),
                    run_id=run_id,
                    direction="input",
                )
            )

        # Should succeed — completed run does not block deletion
        await blob_service.delete_blob(record.id)

        with pytest.raises(BlobNotFoundError):
            await blob_service.get_blob(record.id)

    @pytest.mark.asyncio
    async def test_delete_blob_rejects_when_active_run_exists_without_link(self, blob_service, session_id, db_engine) -> None:
        """Pre-link window: active run exists but blob_run_links row hasn't been created yet.

        _execute_locked() creates the run record before link_blob_to_run()
        inserts the link row.  During that gap, the explicit-link guard sees
        nothing.  The composition-state guard must block deletion because
        the run's source references this blob via blob_ref.
        """
        from elspeth.web.sessions.models import (
            composition_states_table,
            runs_table,
        )

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="pre-link.csv",
            content=b"important",
            mime_type="text/csv",
            created_by="user",
        )

        state_id = str(uuid4())
        session_id_str = str(session_id)
        run_id = str(uuid4())

        with db_engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=session_id_str,
                    version=1,
                    # Source references this blob via blob_ref — the run is
                    # about to link it once link_blob_to_run() fires.
                    source={
                        "plugin": "csv",
                        "options": {"blob_ref": str(record.id), "path": str(record.storage_path)},
                    },
                    is_valid=True,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=run_id,
                    session_id=session_id_str,
                    state_id=state_id,
                    status="pending",
                    started_at=datetime(2026, 1, 1, tzinfo=UTC),
                    rows_processed=0,
                    rows_failed=0,
                )
            )
            # Deliberately NO blob_run_links row — simulating the pre-link window

        with pytest.raises(BlobActiveRunError):
            await blob_service.delete_blob(record.id)

    @pytest.mark.asyncio
    async def test_delete_blob_allows_when_active_run_uses_different_source(self, blob_service, session_id, db_engine) -> None:
        """Active run using source.path (no blob_ref) must not block unrelated blob deletion.

        Regression test: the original session-level guard blocked ALL blobs
        when ANY run was active, even if that run used a file-path source
        with no blob_ref.  The scoped guard checks the composition state's
        source.options.blob_ref and only blocks if it matches this blob.
        """
        from elspeth.web.sessions.models import (
            composition_states_table,
            runs_table,
        )

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="unrelated.csv",
            content=b"not used by run",
            mime_type="text/csv",
            created_by="user",
        )

        state_id = str(uuid4())
        session_id_str = str(session_id)
        run_id = str(uuid4())

        with db_engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=session_id_str,
                    version=1,
                    # Source uses file path, NOT blob_ref — run is unrelated
                    # to the blob being deleted.
                    source={
                        "plugin": "csv",
                        "options": {"path": "/data/external/other.csv"},
                    },
                    is_valid=True,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=run_id,
                    session_id=session_id_str,
                    state_id=state_id,
                    status="pending",
                    started_at=datetime(2026, 1, 1, tzinfo=UTC),
                    rows_processed=0,
                    rows_failed=0,
                )
            )

        # Should succeed — active run does not reference this blob
        await blob_service.delete_blob(record.id)

        with pytest.raises(BlobNotFoundError):
            await blob_service.get_blob(record.id)

    @pytest.mark.asyncio
    async def test_delete_blob_rejects_when_active_run_path_matches_storage(self, blob_service, session_id, db_engine) -> None:
        """Active run using source.path matching this blob's storage_path must block.

        A run can read a blob's backing file via plain set_source with
        options.path (no blob_ref).  The guard must check path/file matches
        in addition to blob_ref.
        """
        from elspeth.web.sessions.models import (
            composition_states_table,
            runs_table,
        )

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="path-backed.csv",
            content=b"path match",
            mime_type="text/csv",
            created_by="user",
        )

        state_id = str(uuid4())
        session_id_str = str(session_id)
        run_id = str(uuid4())

        with db_engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=session_id_str,
                    version=1,
                    # Source references this blob via path, NOT blob_ref.
                    source={
                        "plugin": "csv",
                        "options": {"path": record.storage_path},
                    },
                    is_valid=True,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=run_id,
                    session_id=session_id_str,
                    state_id=state_id,
                    status="pending",
                    started_at=datetime(2026, 1, 1, tzinfo=UTC),
                    rows_processed=0,
                    rows_failed=0,
                )
            )

        with pytest.raises(BlobActiveRunError):
            await blob_service.delete_blob(record.id)

    @pytest.mark.asyncio
    async def test_delete_blob_allows_when_completed_run_exists_without_link(self, blob_service, session_id, db_engine) -> None:
        """Completed runs (without link row) must not block deletion."""
        from elspeth.web.sessions.models import (
            composition_states_table,
            runs_table,
        )

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="completed-no-link.csv",
            content=b"done",
            mime_type="text/csv",
            created_by="user",
        )

        state_id = str(uuid4())
        session_id_str = str(session_id)
        run_id = str(uuid4())

        with db_engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=session_id_str,
                    version=1,
                    is_valid=True,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=run_id,
                    session_id=session_id_str,
                    state_id=state_id,
                    status="completed",
                    started_at=datetime(2026, 1, 1, tzinfo=UTC),
                    rows_processed=0,
                    rows_failed=0,
                )
            )

        # Should succeed — completed run does not block deletion
        await blob_service.delete_blob(record.id)

        with pytest.raises(BlobNotFoundError):
            await blob_service.get_blob(record.id)


# ---------------------------------------------------------------------------
# finalize_blob — pending lifecycle transitions
# ---------------------------------------------------------------------------


class TestFinalizeBlob:
    """Pending -> ready/error lifecycle: only valid transitions allowed."""

    @pytest.mark.asyncio
    async def test_finalize_blob_transitions_pending_to_ready(self, blob_service, session_id) -> None:
        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )
        assert pending.status == "pending"

        # Valid SHA-256 hex is required when transitioning to 'ready' —
        # see _validate_finalize_hash().  Using content_hash() here
        # anchors the test to the same helper production code uses.
        valid_hash = content_hash(b"pretend-output-bytes")
        finalized = await blob_service.finalize_blob(
            blob_id=pending.id,
            status="ready",
            size_bytes=42,
            content_hash=valid_hash,
        )
        assert finalized.status == "ready"
        assert finalized.size_bytes == 42
        assert finalized.content_hash == valid_hash

    @pytest.mark.asyncio
    async def test_finalize_blob_rejects_missing_hash_for_ready(self, blob_service, session_id) -> None:
        """Tier 1 invariant: finalizing as 'ready' without a hash is refused."""
        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        from elspeth.web.blobs.protocol import BlobStateError

        with pytest.raises(BlobStateError, match="content_hash"):
            await blob_service.finalize_blob(
                blob_id=pending.id,
                status="ready",
                size_bytes=42,
            )

    @pytest.mark.asyncio
    async def test_finalize_blob_rejects_non_sha256_hash(self, blob_service, session_id) -> None:
        """Tier 1 invariant: content_hash must be 64 lowercase hex chars."""
        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        from elspeth.web.blobs.protocol import BlobStateError

        with pytest.raises(BlobStateError, match="64 lowercase hex"):
            await blob_service.finalize_blob(
                blob_id=pending.id,
                status="ready",
                size_bytes=42,
                content_hash="abc123",  # too short, not SHA-256
            )

    @pytest.mark.asyncio
    async def test_finalize_blob_rejects_uppercase_hex_hash(self, blob_service, session_id) -> None:
        """Canonical form is lowercase — uppercase hex is a bifurcation risk.

        FilesystemPayloadStore writes the lowercase form, and
        read_blob_content compares via hmac.compare_digest byte-for-byte.
        Admitting uppercase at the write side would silently create
        blobs whose hash does not match the stored form anywhere else
        in the audit trail.  Mirrors the same assertion on the sync
        path (TestFinalizeBlobSyncHashValidation) so both entry points
        are pinned.
        """
        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        from elspeth.web.blobs.protocol import BlobStateError

        uppercase_hash = content_hash(b"real-bytes").upper()
        with pytest.raises(BlobStateError, match="64 lowercase hex"):
            await blob_service.finalize_blob(
                blob_id=pending.id,
                status="ready",
                size_bytes=10,
                content_hash=uppercase_hash,
            )

    @pytest.mark.asyncio
    async def test_finalize_blob_rejects_trailing_newline_hash(self, blob_service, session_id) -> None:
        """``^[a-f0-9]{64}$`` + ``re.match`` accepts trailing ``\\n``; fullmatch rejects it.

        Python's ``$`` anchor matches either end-of-string OR just
        before a final newline.  A 64-hex hash followed by a single
        ``\\n`` therefore slipped through the service-layer pre-check
        under the old regex and landed at the DB, where the CHECK
        constraint rejected it as an IntegrityError — the wrong failure
        surface (opaque DB error rather than the clean BlobStateError
        this validator is supposed to raise, and coverage on the
        DB-authoritative guard only).  The service pre-check uses
        ``fullmatch`` so the error path is always the structured
        BlobStateError, and the DB CHECK remains the belt for any
        writer that bypasses the service entirely.
        """
        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        from elspeth.web.blobs.protocol import BlobStateError

        trailing_newline_hash = content_hash(b"real-bytes") + "\n"
        with pytest.raises(BlobStateError, match="64 lowercase hex"):
            await blob_service.finalize_blob(
                blob_id=pending.id,
                status="ready",
                size_bytes=10,
                content_hash=trailing_newline_hash,
            )

    @pytest.mark.asyncio
    async def test_finalize_blob_as_error_without_hash_succeeds(self, blob_service, session_id) -> None:
        """The hash invariant applies only to 'ready' — 'error' needs no hash.

        Pins the ``status != 'ready'`` exemption branch of
        _validate_finalize_hash.  A regression that tightened the
        invariant to require hashes for error blobs would break every
        failed-run cleanup path, and the failure mode would be
        non-obvious (pipeline-level errors finalizing per-blob errors).
        This positive test keeps the exemption honest.
        """
        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="failed-output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        record = await blob_service.finalize_blob(
            blob_id=pending.id,
            status="error",
            # deliberately no content_hash, no size_bytes
        )
        assert record.status == "error"
        assert record.content_hash is None

    @pytest.mark.asyncio
    async def test_finalize_blob_rejects_non_pending(self, blob_service, session_id) -> None:
        """Cannot finalize a blob that is already ready — status rollback is forbidden."""
        record = await blob_service.create_blob(
            session_id=session_id,
            filename="already-ready.csv",
            content=b"done",
            mime_type="text/csv",
            created_by="user",
        )
        assert record.status == "ready"

        from elspeth.web.blobs.protocol import BlobStateError

        with pytest.raises(BlobStateError, match="expected 'pending'"):
            await blob_service.finalize_blob(
                blob_id=record.id,
                status="ready",
                size_bytes=4,
            )

    @pytest.mark.asyncio
    async def test_finalize_blob_rejects_invalid_status(self, blob_service, session_id) -> None:
        """Only 'ready' and 'error' are valid finalize targets."""
        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        # Deliberate type-contract violation: we're exercising the
        # runtime guard for dynamic callers that bypass static typing.
        # `blob_service` is a pytest fixture whose type mypy treats as
        # Any, so no `# type: ignore` is needed here to suppress the
        # arg-type error.
        with pytest.raises(RuntimeError, match="Invalid finalize status"):
            await blob_service.finalize_blob(
                blob_id=pending.id,
                status="deleted",
            )


# ---------------------------------------------------------------------------
# Blob quota — per-session storage limit (AD-10)
# ---------------------------------------------------------------------------


class TestBlobQuota:
    """Per-session cumulative storage quota prevents unbounded disk growth."""

    @pytest.mark.asyncio
    async def test_quota_rejects_when_exceeded(self, db_engine, session_id, tmp_path) -> None:
        """Upload that would exceed the session quota returns BlobQuotaExceededError."""
        from elspeth.web.blobs.protocol import BlobQuotaExceededError

        # Tiny quota: 100 bytes
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)

        # First blob: 60 bytes — fits
        await service.create_blob(
            session_id=session_id,
            filename="a.csv",
            content=b"x" * 60,
            mime_type="text/csv",
            created_by="user",
        )

        # Second blob: 60 bytes — total would be 120 > 100
        with pytest.raises(BlobQuotaExceededError):
            await service.create_blob(
                session_id=session_id,
                filename="b.csv",
                content=b"x" * 60,
                mime_type="text/csv",
                created_by="user",
            )

    @pytest.mark.asyncio
    async def test_quota_allows_within_limit(self, db_engine, session_id, tmp_path) -> None:
        """Uploads within the quota succeed."""
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=200)

        await service.create_blob(
            session_id=session_id,
            filename="a.csv",
            content=b"x" * 90,
            mime_type="text/csv",
            created_by="user",
        )
        record = await service.create_blob(
            session_id=session_id,
            filename="b.csv",
            content=b"x" * 90,
            mime_type="text/csv",
            created_by="user",
        )
        assert record.status == "ready"


# ---------------------------------------------------------------------------
# copy_blobs_for_fork — rollback on partial failure
# ---------------------------------------------------------------------------


class TestCopyBlobsForForkRollback:
    """Rollback path: partial blob copies are cleaned up on failure.

    copy_blobs_for_fork creates blobs atomically (one at a time). If the
    second create_blob fails, the first already-committed blob must be
    deleted — both its DB row and its file on disk.
    """

    @pytest.fixture()
    def target_session_id(self, db_engine) -> UUID:
        """Second session for the fork target."""
        sid = str(uuid4())
        now = datetime.now(UTC)
        with db_engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=sid,
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Forked Session",
                    created_at=now,
                    updated_at=now,
                )
            )
        return UUID(sid)

    @pytest.mark.asyncio
    async def test_rollback_cleans_up_partial_copies(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
        tmp_path,
    ) -> None:
        """When create_blob fails mid-copy, already-created blobs are removed."""
        # Create two blobs in the source session
        await blob_service.create_blob(
            session_id=session_id,
            filename="first.csv",
            content=b"first file content",
            mime_type="text/csv",
            created_by="user",
        )
        await blob_service.create_blob(
            session_id=session_id,
            filename="second.csv",
            content=b"second file content",
            mime_type="text/csv",
            created_by="user",
        )

        # Patch create_blob to fail on the second call
        original_create = blob_service.create_blob
        call_count = 0

        async def _failing_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise RuntimeError("Simulated disk failure on second blob")
            return await original_create(*args, **kwargs)

        blob_service.create_blob = _failing_create  # type: ignore[method-assign]

        with pytest.raises(RuntimeError, match="Simulated disk failure"):
            await blob_service.copy_blobs_for_fork(session_id, target_session_id)

        # Verify rollback: no blobs should remain in the target session
        target_blobs = await blob_service.list_blobs(target_session_id)
        assert target_blobs == [], f"Expected 0 blobs after rollback, found {len(target_blobs)}: {[b.filename for b in target_blobs]}"

    @pytest.mark.asyncio
    async def test_empty_source_returns_empty_map(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
    ) -> None:
        """No blobs in source session → empty mapping, no errors."""
        result = await blob_service.copy_blobs_for_fork(session_id, target_session_id)
        assert result == {}

    @pytest.mark.asyncio
    async def test_quota_exceeded_before_any_copy(
        self,
        blob_service: BlobServiceImpl,
        db_engine,
        session_id: UUID,
        target_session_id: UUID,
        tmp_path,
    ) -> None:
        """Quota check happens before copying — no partial writes."""
        from elspeth.web.blobs.protocol import BlobQuotaExceededError

        # Create a blob using the default (large-quota) service
        await blob_service.create_blob(
            session_id=session_id,
            filename="big.csv",
            content=b"x" * 100,
            mime_type="text/csv",
            created_by="user",
        )

        # Fork with a small-quota service — quota pre-check should reject
        small_quota = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=10)

        with pytest.raises(BlobQuotaExceededError):
            await small_quota.copy_blobs_for_fork(session_id, target_session_id)

        # Verify: no blobs in target (pre-check prevented any copies)
        target_blobs = await blob_service.list_blobs(target_session_id)
        assert target_blobs == []

    @pytest.mark.asyncio
    async def test_cleanup_failures_attached_as_notes_not_swallowed(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
    ) -> None:
        """Rollback delete_blob failures must surface as notes on the primary exception.

        The original silent-failure was an inner ``except Exception: pass`` on
        the rollback delete_blob call.  A failed delete_blob leaves the DB row
        orphaned — the target session would carry phantom blob metadata that
        auditors interpret as successfully-copied blobs while the file is
        gone.  Mirrors the RecoveryFailed[...] convention used by
        ``BlobServiceImpl.finalize_run_output_blobs`` (the per-blob
        recovery branch inside its nested ``_sync`` closure).

        Contract: primary copy exception is the headline; every cleanup
        failure is attached as an ``add_note()`` entry naming the orphan
        blob_id and the underlying exception type.
        """
        # Create two source blobs — second create_blob will fail to trigger rollback
        await blob_service.create_blob(
            session_id=session_id,
            filename="first.csv",
            content=b"first",
            mime_type="text/csv",
            created_by="user",
        )
        await blob_service.create_blob(
            session_id=session_id,
            filename="second.csv",
            content=b"second",
            mime_type="text/csv",
            created_by="user",
        )

        original_create = blob_service.create_blob
        original_delete = blob_service.delete_blob
        create_calls = 0

        async def _failing_create(*args, **kwargs):
            nonlocal create_calls
            create_calls += 1
            if create_calls >= 2:
                raise RuntimeError("Simulated copy failure on second blob")
            return await original_create(*args, **kwargs)

        # Make the rollback delete_blob also fail.  OSError is one of the
        # narrowly-caught recovery faults; programmer bugs would propagate.
        delete_failure = OSError(5, "I/O error during cleanup")

        async def _failing_delete(*_args, **_kwargs):
            raise delete_failure

        blob_service.create_blob = _failing_create  # type: ignore[method-assign]
        blob_service.delete_blob = _failing_delete  # type: ignore[method-assign]

        try:
            with pytest.raises(RuntimeError, match="Simulated copy failure") as exc_info:
                await blob_service.copy_blobs_for_fork(session_id, target_session_id)
        finally:
            # Restore so cleanup in fixtures doesn't break
            blob_service.create_blob = original_create  # type: ignore[method-assign]
            blob_service.delete_blob = original_delete  # type: ignore[method-assign]

        # Headline exception must remain the primary copy failure
        assert type(exc_info.value) is RuntimeError, f"Cleanup OSError masked primary exception: got {type(exc_info.value).__name__}"

        notes = getattr(exc_info.value, "__notes__", [])
        assert notes, (
            "Cleanup failure was silently swallowed — expected add_note() to attach RecoveryFailed diagnostic to primary exception"
        )

        recovery_notes = [n for n in notes if "RecoveryFailed[OSError]" in n]
        assert recovery_notes, f"Missing RecoveryFailed[OSError] note in {notes!r}"
        # Note must identify the orphaned blob_id and target session for triage
        assert any("manual cleanup" in n.lower() for n in recovery_notes), "Note must direct operator to manual cleanup"
        assert any(str(target_session_id) in n for n in recovery_notes), "Note must identify the target session containing the orphan row"


# ---------------------------------------------------------------------------
# finalize_run_output_blobs — run-level batch finalization
# ---------------------------------------------------------------------------


class TestFinalizeRunOutputBlobs:
    """Batch finalization of pending output blobs when a run completes or fails."""

    @pytest.fixture()
    def run_env(self, blob_service, session_id, db_engine):
        """Set up a composition state and run, return (run_id, session_id_str)."""
        from elspeth.web.sessions.models import (
            composition_states_table,
            runs_table,
        )

        state_id = str(uuid4())
        session_id_str = str(session_id)
        run_id = str(uuid4())

        with db_engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=session_id_str,
                    version=1,
                    is_valid=True,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=run_id,
                    session_id=session_id_str,
                    state_id=state_id,
                    status="running",
                    started_at=datetime(2026, 1, 1, tzinfo=UTC),
                    rows_processed=0,
                    rows_failed=0,
                )
            )
        return UUID(run_id), session_id_str

    @pytest.mark.asyncio
    async def test_success_path_sets_ready_with_size_and_hash(self, blob_service, session_id, db_engine, run_env) -> None:
        """Pending blob with file written -> ready with size_bytes and content_hash."""
        from elspeth.web.sessions.models import blob_run_links_table

        run_id, _ = run_env

        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )
        assert pending.status == "pending"

        # Write content to the storage path (simulating sink output)
        from pathlib import Path as _Path

        file_content = b"col1,col2\na,b\nc,d"
        _Path(pending.storage_path).write_bytes(file_content)

        # Link blob to run as output
        with db_engine.begin() as conn:
            conn.execute(
                blob_run_links_table.insert().values(
                    blob_id=str(pending.id),
                    run_id=str(run_id),
                    direction="output",
                )
            )

        result = await blob_service.finalize_run_output_blobs(run_id, success=True)
        assert len(result.finalized) == 1
        assert len(result.errors) == 0
        assert result.finalized[0].status == "ready"
        assert result.finalized[0].size_bytes == len(file_content)
        assert result.finalized[0].content_hash == content_hash(file_content)

    @pytest.mark.asyncio
    async def test_file_not_written_sets_error(self, blob_service, session_id, db_engine, run_env) -> None:
        """Pending blob without file on disk -> error status on success=True."""
        from elspeth.web.sessions.models import blob_run_links_table

        run_id, _ = run_env

        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="missing.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        # Do NOT write any file — simulate sink that didn't produce output

        with db_engine.begin() as conn:
            conn.execute(
                blob_run_links_table.insert().values(
                    blob_id=str(pending.id),
                    run_id=str(run_id),
                    direction="output",
                )
            )

        result = await blob_service.finalize_run_output_blobs(run_id, success=True)
        assert len(result.finalized) == 1
        assert len(result.errors) == 0
        assert result.finalized[0].status == "error"

    @pytest.mark.asyncio
    async def test_run_failed_sets_error(self, blob_service, session_id, db_engine, run_env) -> None:
        """Pending blob with success=False -> error regardless of file state."""
        from pathlib import Path as _Path

        from elspeth.web.sessions.models import blob_run_links_table

        run_id, _ = run_env

        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        # Write file — but the run failed, so it should still be marked error
        _Path(pending.storage_path).write_bytes(b"partial-output")

        with db_engine.begin() as conn:
            conn.execute(
                blob_run_links_table.insert().values(
                    blob_id=str(pending.id),
                    run_id=str(run_id),
                    direction="output",
                )
            )

        result = await blob_service.finalize_run_output_blobs(run_id, success=False)
        assert len(result.finalized) == 1
        assert len(result.errors) == 0
        assert result.finalized[0].status == "error"


# ---------------------------------------------------------------------------
# Partial-failure resilience — elspeth-9f31c32cce
# ---------------------------------------------------------------------------


class TestFinalizeRunOutputBlobsPartialFailure:
    """Per-blob errors must not abort finalization of remaining blobs.

    Bug: elspeth-9f31c32cce — finalize_run_output_blobs aborts on per-blob
    failure, leaving remaining blobs permanently pending for terminal runs.
    """

    @pytest.fixture()
    def run_env(self, blob_service, session_id, db_engine):
        """Set up a composition state and run, return (run_id, session_id_str)."""
        from elspeth.web.sessions.models import (
            composition_states_table,
            runs_table,
        )

        state_id = str(uuid4())
        session_id_str = str(session_id)
        run_id = str(uuid4())

        with db_engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=session_id_str,
                    version=1,
                    is_valid=True,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=run_id,
                    session_id=session_id_str,
                    state_id=state_id,
                    status="running",
                    started_at=datetime(2026, 1, 1, tzinfo=UTC),
                    rows_processed=0,
                    rows_failed=0,
                )
            )
        return UUID(run_id), session_id_str

    async def _create_linked_blob(
        self,
        blob_service,
        session_id: UUID,
        run_id: UUID,
        db_engine,
        filename: str,
        content: bytes | None = None,
    ):
        """Create a pending blob, optionally write content, and link to run."""
        from elspeth.web.sessions.models import blob_run_links_table

        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename=filename,
            mime_type="text/csv",
            created_by="pipeline",
        )
        if content is not None:
            from pathlib import Path as _Path

            _Path(pending.storage_path).write_bytes(content)

        with db_engine.begin() as conn:
            conn.execute(
                blob_run_links_table.insert().values(
                    blob_id=str(pending.id),
                    run_id=str(run_id),
                    direction="output",
                )
            )
        return pending

    @pytest.mark.asyncio
    async def test_continues_after_concurrent_deletion(
        self,
        blob_service,
        session_id,
        db_engine,
        run_env,
    ) -> None:
        """When blob 2 of 3 is concurrently deleted (between initial query
        and per-blob finalize), blobs 1 and 3 still finalize."""
        from elspeth.web.blobs.protocol import BlobNotFoundError

        run_id, _ = run_env

        b1 = await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b1.csv", b"data1")
        b2 = await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b2.csv", b"data2")
        b3 = await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b3.csv", b"data3")

        # Patch _finalize_blob_sync to simulate concurrent deletion of b2
        # in the window between the initial SELECT and per-blob finalize.
        original = blob_service._finalize_blob_sync

        def _patched(blob_id, *args, **kwargs):
            if blob_id == b2.id:
                raise BlobNotFoundError(str(blob_id))
            return original(blob_id, *args, **kwargs)

        blob_service._finalize_blob_sync = _patched
        try:
            result = await blob_service.finalize_run_output_blobs(run_id, success=True)
        finally:
            blob_service._finalize_blob_sync = original

        assert len(result.finalized) == 2, f"Expected 2 finalized, got {len(result.finalized)}"
        assert len(result.errors) == 1, f"Expected 1 error, got {len(result.errors)}"
        assert result.errors[0].blob_id == b2.id
        assert result.errors[0].exc_type == "BlobNotFoundError"
        finalized_ids = {r.id for r in result.finalized}
        assert b1.id in finalized_ids
        assert b3.id in finalized_ids

    @pytest.mark.asyncio
    async def test_continues_after_already_finalized(
        self,
        blob_service,
        session_id,
        db_engine,
        run_env,
    ) -> None:
        """When blob 2 raises BlobStateError (already finalized), loop continues."""
        from elspeth.web.blobs.protocol import BlobStateError

        run_id, _ = run_env

        await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b1.csv", b"data1")
        b2 = await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b2.csv", b"data2")
        await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b3.csv", b"data3")

        # Patch _finalize_blob_sync to simulate b2 already finalized
        original = blob_service._finalize_blob_sync

        def _patched(blob_id, *args, **kwargs):
            if blob_id == b2.id:
                raise BlobStateError(str(blob_id), message="Cannot finalize — status is 'ready', expected 'pending'")
            return original(blob_id, *args, **kwargs)

        blob_service._finalize_blob_sync = _patched
        try:
            result = await blob_service.finalize_run_output_blobs(run_id, success=True)
        finally:
            blob_service._finalize_blob_sync = original

        assert len(result.finalized) == 2
        assert len(result.errors) == 1
        assert result.errors[0].blob_id == b2.id
        assert result.errors[0].exc_type == "BlobStateError"

    @pytest.mark.asyncio
    async def test_continues_after_os_error_reading_file(
        self,
        blob_service,
        session_id,
        db_engine,
        run_env,
        tmp_path,
    ) -> None:
        """When file read raises OSError, loop continues to next blob."""
        run_id, _ = run_env

        await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b1.csv", b"data1")
        b2 = await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b2.csv", b"data2")

        # Make b2's backing file unreadable
        from pathlib import Path as _Path

        b2_path = _Path(b2.storage_path)
        b2_path.chmod(0o000)

        try:
            result = await blob_service.finalize_run_output_blobs(run_id, success=True)
        finally:
            # Restore permissions for cleanup
            b2_path.chmod(0o644)

        assert len(result.finalized) == 1
        assert len(result.errors) == 1
        assert result.errors[0].blob_id == b2.id
        assert "OSError" in result.errors[0].exc_type or "PermissionError" in result.errors[0].exc_type

    @pytest.mark.asyncio
    async def test_propagates_type_error(
        self,
        blob_service,
        session_id,
        db_engine,
        run_env,
    ) -> None:
        """Programmer bugs (TypeError) must crash, not be caught."""
        run_id, _ = run_env

        await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b1.csv", b"data1")

        # Inject a TypeError via patching _finalize_blob_sync
        original = blob_service._finalize_blob_sync

        def _broken_finalize(*args, **kwargs):
            raise TypeError("unexpected keyword argument")

        blob_service._finalize_blob_sync = _broken_finalize
        try:
            with pytest.raises(TypeError, match="unexpected keyword argument"):
                await blob_service.finalize_run_output_blobs(run_id, success=True)
        finally:
            blob_service._finalize_blob_sync = original

    @pytest.mark.asyncio
    async def test_all_blobs_fail_returns_empty_finalized_with_errors(
        self,
        blob_service,
        session_id,
        db_engine,
        run_env,
    ) -> None:
        """When all blobs fail, result has empty finalized and N errors."""
        from elspeth.web.blobs.protocol import BlobNotFoundError

        run_id, _ = run_env

        await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b1.csv", b"data1")
        await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b2.csv", b"data2")

        # Patch to simulate all blobs concurrently deleted
        original = blob_service._finalize_blob_sync

        def _all_missing(blob_id, *args, **kwargs):
            raise BlobNotFoundError(str(blob_id))

        blob_service._finalize_blob_sync = _all_missing
        try:
            result = await blob_service.finalize_run_output_blobs(run_id, success=True)
        finally:
            blob_service._finalize_blob_sync = original

        assert len(result.finalized) == 0
        assert len(result.errors) == 2

    @pytest.mark.asyncio
    async def test_zero_pending_blobs_returns_empty_result(
        self,
        blob_service,
        session_id,
        db_engine,
        run_env,
    ) -> None:
        """Run with no pending output blobs returns empty result."""
        run_id, _ = run_env

        result = await blob_service.finalize_run_output_blobs(run_id, success=True)

        assert len(result.finalized) == 0
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_best_effort_error_recovery_marks_blob_as_error(
        self,
        blob_service,
        session_id,
        db_engine,
        run_env,
    ) -> None:
        """When per-blob catch fires, the failed blob is set to 'error' status."""
        from elspeth.web.sessions.models import blobs_table as bt

        run_id, _ = run_env

        b1 = await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b1.csv", b"data1")
        b2 = await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b2.csv", b"data2")

        # Make b1's file unreadable — triggers OSError, caught per-blob
        from pathlib import Path as _Path

        b1_path = _Path(b1.storage_path)
        b1_path.chmod(0o000)

        try:
            result = await blob_service.finalize_run_output_blobs(run_id, success=True)
        finally:
            b1_path.chmod(0o644)

        # b1 should have been moved to "error" by the best-effort recovery
        with db_engine.connect() as conn:
            row = conn.execute(bt.select().where(bt.c.id == str(b1.id))).first()
        assert row is not None
        assert row.status == "error", f"Expected 'error', got '{row.status}' — recovery should mark failed blobs"

        # b2 should be finalized normally
        assert len(result.finalized) == 1
        assert result.finalized[0].id == b2.id

    @pytest.mark.asyncio
    async def test_runtime_error_from_vanished_blob_propagates(
        self,
        blob_service,
        session_id,
        db_engine,
        run_env,
    ) -> None:
        """RuntimeError (Tier 1 anomaly: blob vanished mid-transaction) propagates."""
        run_id, _ = run_env

        await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b1.csv", b"data1")

        original = blob_service._finalize_blob_sync

        def _vanishing_finalize(*args, **kwargs):
            raise RuntimeError("Blob abc vanished during finalize — concurrent deletion?")

        blob_service._finalize_blob_sync = _vanishing_finalize
        try:
            with pytest.raises(RuntimeError, match="vanished during finalize"):
                await blob_service.finalize_run_output_blobs(run_id, success=True)
        finally:
            blob_service._finalize_blob_sync = original


# ---------------------------------------------------------------------------
# read_blob_content — lifecycle and integrity guards (elspeth-6082ad9636)
# ---------------------------------------------------------------------------


class TestReadBlobContentLifecycleGuard:
    """read_blob_content must enforce blob lifecycle state and content integrity.

    Bug: elspeth-6082ad9636 — read_blob_content() returns bytes without
    checking blob status or verifying the stored content_hash.
    """

    @pytest.mark.asyncio
    async def test_rejects_pending_blob(self, blob_service, session_id) -> None:
        """Pending blobs have no finalized content — reading must fail."""
        from pathlib import Path as _Path

        from elspeth.web.blobs.protocol import BlobStateError

        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )
        # Write a file so the only guard is status, not file existence
        _Path(pending.storage_path).write_bytes(b"partial-content")

        with pytest.raises(BlobStateError):
            await blob_service.read_blob_content(pending.id)

    @pytest.mark.asyncio
    async def test_rejects_error_blob(self, blob_service, session_id) -> None:
        """Error blobs represent failed runs — content must not be served."""
        from pathlib import Path as _Path

        from elspeth.web.blobs.protocol import BlobStateError

        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )
        _Path(pending.storage_path).write_bytes(b"partial-content")
        await blob_service.finalize_blob(pending.id, status="error")

        with pytest.raises(BlobStateError):
            await blob_service.read_blob_content(pending.id)

    @pytest.mark.asyncio
    async def test_detects_content_hash_mismatch(self, blob_service, session_id) -> None:
        """Tier 1 integrity: if stored hash doesn't match file bytes, crash."""
        from pathlib import Path as _Path

        from elspeth.web.blobs.protocol import BlobIntegrityError

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="tampered.csv",
            content=b"original-content",
            mime_type="text/csv",
            created_by="user",
        )
        assert record.status == "ready"
        assert record.content_hash is not None

        # Tamper with the file on disk after creation
        _Path(record.storage_path).write_bytes(b"tampered-content")

        with pytest.raises(BlobIntegrityError):
            await blob_service.read_blob_content(record.id)

    @pytest.mark.asyncio
    async def test_rejects_ready_blob_with_missing_backing_file(self, blob_service, session_id) -> None:
        """Ready metadata without backing bytes is an integrity failure, not 404."""
        from pathlib import Path as _Path

        from elspeth.web.blobs.protocol import BlobContentMissingError

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="missing.csv",
            content=b"original-content",
            mime_type="text/csv",
            created_by="user",
        )
        _Path(record.storage_path).unlink()

        with pytest.raises(BlobContentMissingError, match="backing file"):
            await blob_service.read_blob_content(record.id)

    @pytest.mark.asyncio
    async def test_rejects_pending_blob_without_file(self, blob_service, session_id) -> None:
        """Pending blob with no file must raise BlobStateError, not BlobNotFoundError.

        Guards exception ordering: the status check must fire before
        the file-existence check, otherwise a missing file would mask
        the lifecycle violation.
        """
        from elspeth.web.blobs.protocol import BlobStateError

        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="no-file.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )
        # Deliberately do NOT write a file

        with pytest.raises(BlobStateError, match="expected 'ready'"):
            await blob_service.read_blob_content(pending.id)

    @pytest.mark.asyncio
    async def test_ready_blob_with_valid_hash_succeeds(self, blob_service, session_id) -> None:
        """Ready blob with matching hash returns content normally."""
        content = b"valid-content"
        record = await blob_service.create_blob(
            session_id=session_id,
            filename="good.csv",
            content=content,
            mime_type="text/csv",
            created_by="user",
        )

        result = await blob_service.read_blob_content(record.id)
        assert result == content


# ---------------------------------------------------------------------------
# finalize_run_output_blobs — error path file cleanup (elspeth-0a2644dcb9)
# ---------------------------------------------------------------------------


class TestFinalizeRunOutputBlobsErrorCleanup:
    """Failed run outputs must not leave orphaned backing files.

    Bug: elspeth-0a2644dcb9 — finalize to "error" only updates metadata,
    leaving the backing file on disk while size_bytes=0 and content_hash=None.
    """

    @pytest.fixture()
    def run_env(self, blob_service, session_id, db_engine):
        """Set up a composition state and run, return (run_id, session_id_str)."""
        from elspeth.web.sessions.models import (
            composition_states_table,
            runs_table,
        )

        state_id = str(uuid4())
        session_id_str = str(session_id)
        run_id = str(uuid4())

        with db_engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=session_id_str,
                    version=1,
                    is_valid=True,
                    created_at=datetime(2026, 1, 1, tzinfo=UTC),
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=run_id,
                    session_id=session_id_str,
                    state_id=state_id,
                    status="running",
                    started_at=datetime(2026, 1, 1, tzinfo=UTC),
                    rows_processed=0,
                    rows_failed=0,
                )
            )
        return UUID(run_id), session_id_str

    @pytest.mark.asyncio
    async def test_failure_deletes_backing_file(self, blob_service, session_id, db_engine, run_env) -> None:
        """When run fails, backing file must be deleted — not left orphaned."""
        from pathlib import Path as _Path

        from elspeth.web.sessions.models import blob_run_links_table

        run_id, _ = run_env

        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        # Simulate sink writing partial output before run failure
        storage = _Path(pending.storage_path)
        storage.write_bytes(b"partial-output-before-crash")
        assert storage.exists()

        with db_engine.begin() as conn:
            conn.execute(
                blob_run_links_table.insert().values(
                    blob_id=str(pending.id),
                    run_id=str(run_id),
                    direction="output",
                )
            )

        result = await blob_service.finalize_run_output_blobs(run_id, success=False)
        assert len(result.finalized) == 1
        blob_result = result.finalized[0]
        assert blob_result.status == "error"

        # THE BUG: file must NOT exist after error finalization
        assert not storage.exists(), "Backing file still exists after error finalization — orphaned file will escape quota accounting"

        # Metadata must reflect no content — size_bytes=0, content_hash=None.
        # If these don't match, quota accounting diverges from filesystem.
        assert blob_result.size_bytes == 0, f"Expected size_bytes=0 for error blob, got {blob_result.size_bytes}"
        assert blob_result.content_hash is None, f"Expected content_hash=None for error blob, got {blob_result.content_hash}"

    @pytest.mark.asyncio
    async def test_failure_without_file_still_sets_error(self, blob_service, session_id, db_engine, run_env) -> None:
        """When run fails and no file was written, status is still error (no crash)."""
        from elspeth.web.sessions.models import blob_run_links_table

        run_id, _ = run_env

        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="never-written.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        with db_engine.begin() as conn:
            conn.execute(
                blob_run_links_table.insert().values(
                    blob_id=str(pending.id),
                    run_id=str(run_id),
                    direction="output",
                )
            )

        result = await blob_service.finalize_run_output_blobs(run_id, success=False)
        assert len(result.finalized) == 1
        assert result.finalized[0].status == "error"


# ---------------------------------------------------------------------------
# Database-level integrity constraint — ck_blobs_ready_hash (elspeth-e435b147b7)
# ---------------------------------------------------------------------------


class TestBlobsReadyHashDBConstraint:
    """The DB refuses status='ready' rows without a content_hash.

    Service-level validation in _validate_finalize_hash is the first line
    of defence, but the CHECK constraint in the current schema is the belt:
    even raw SQL / direct ORM writes that bypass the service cannot
    commit a violating row.
    """

    def test_inserting_ready_without_hash_raises(self, db_engine, session_id) -> None:
        """Direct INSERT violating the invariant is rejected at commit time."""
        from datetime import UTC, datetime

        from sqlalchemy.exc import IntegrityError

        from elspeth.web.sessions.models import blobs_table

        session_id_str = str(session_id)
        with pytest.raises(IntegrityError), db_engine.begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=str(uuid4()),
                    session_id=session_id_str,
                    filename="illegal.csv",
                    mime_type="text/csv",
                    size_bytes=1,
                    content_hash=None,  # <-- the violation
                    storage_path="/tmp/never",
                    created_at=datetime.now(UTC),
                    created_by="user",
                    status="ready",
                )
            )

    def test_inserting_pending_without_hash_is_allowed(self, db_engine, session_id) -> None:
        """Pending and error rows may carry NULL hashes — only 'ready' is constrained."""
        from datetime import UTC, datetime

        from elspeth.web.sessions.models import blobs_table

        session_id_str = str(session_id)
        with db_engine.begin() as conn:
            conn.execute(
                blobs_table.insert().values(
                    id=str(uuid4()),
                    session_id=session_id_str,
                    filename="pending.csv",
                    mime_type="text/csv",
                    size_bytes=0,
                    content_hash=None,
                    storage_path="/tmp/pending",
                    created_at=datetime.now(UTC),
                    created_by="pipeline",
                    status="pending",
                )
            )

    @pytest.mark.asyncio
    async def test_update_ready_hash_to_null_rejected(self, blob_service, db_engine, session_id) -> None:
        """Can't bypass the guard by mutating an existing ready row."""
        from sqlalchemy import update
        from sqlalchemy.exc import IntegrityError

        from elspeth.web.sessions.models import blobs_table

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="legit.csv",
            content=b"a,b,c\n1,2,3\n",
            mime_type="text/csv",
            created_by="user",
        )

        with pytest.raises(IntegrityError), db_engine.begin() as conn:
            conn.execute(update(blobs_table).where(blobs_table.c.id == str(record.id)).values(content_hash=None))

    @pytest.mark.parametrize(
        "bad_hash",
        [
            "abc123",  # too short
            "a" * 63,  # off-by-one: 63 chars
            "a" * 65,  # off-by-one: 65 chars
            "A" * 64,  # uppercase
            "g" * 64,  # non-hex letter
            "a" * 63 + "Z",  # mostly-hex with one non-hex char
            "",  # empty
            "a" * 64 + "\n",  # trailing newline — ``^...$`` regex accepts this, ``fullmatch`` rejects
        ],
    )
    @pytest.mark.asyncio
    async def test_update_ready_hash_to_malformed_rejected(self, blob_service, db_engine, session_id, bad_hash: str) -> None:
        """Updating a ready row's hash to a malformed value is rejected.

        The service-level write path goes through ``_validate_finalize_hash``
        which rejects malformed hashes before SQL.  This test bypasses the
        service entirely and asserts the database CHECK is the second wall
        — so a future caller that builds an UPDATE statement directly (or
        a migration script that touches content_hash) cannot leave the row
        in a "ready but unverifiable" state.
        """
        from sqlalchemy import update
        from sqlalchemy.exc import IntegrityError

        from elspeth.web.sessions.models import blobs_table

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="legit.csv",
            content=b"a,b,c\n1,2,3\n",
            mime_type="text/csv",
            created_by="user",
        )

        with pytest.raises(IntegrityError), db_engine.begin() as conn:
            conn.execute(update(blobs_table).where(blobs_table.c.id == str(record.id)).values(content_hash=bad_hash))


# ---------------------------------------------------------------------------
# _finalize_blob_sync — mirrors finalize_blob's hash validation but on the
# path actually used by the pipeline output finalizer.  Coverage asymmetry
# between the two entry points would let a regression strip validation
# from the pipeline path while the REST path stayed healthy — the worst
# kind of bifurcation for audit integrity.
# ---------------------------------------------------------------------------


class TestFinalizeBlobSyncHashValidation:
    """_validate_finalize_hash must engage on the sync pipeline path too."""

    @pytest.mark.asyncio
    async def test_sync_path_rejects_missing_hash_for_ready(self, blob_service, session_id) -> None:
        """Invoking _finalize_blob_sync with ready+None hash raises BlobStateError."""
        from elspeth.web.blobs.protocol import BlobStateError

        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="pipe.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        with pytest.raises(BlobStateError, match="content_hash"):
            blob_service._finalize_blob_sync(
                pending.id,
                "ready",
                size_bytes=42,
                content_hash_val=None,
            )

    @pytest.mark.asyncio
    async def test_sync_path_rejects_non_sha256_hash(self, blob_service, session_id) -> None:
        """Invoking _finalize_blob_sync with a malformed hash raises BlobStateError."""
        from elspeth.web.blobs.protocol import BlobStateError

        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="pipe.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        with pytest.raises(BlobStateError, match="64 lowercase hex"):
            blob_service._finalize_blob_sync(
                pending.id,
                "ready",
                size_bytes=42,
                content_hash_val="abc123",  # too short
            )

    @pytest.mark.asyncio
    async def test_sync_path_rejects_uppercase_hex_hash(self, blob_service, session_id) -> None:
        """The canonical form is lowercase; uppercase hex is a bifurcation risk.

        FilesystemPayloadStore writes lowercase, and read_blob_content
        compares via hmac.compare_digest — byte-for-byte.  If the
        write-side validator silently admitted uppercase, a pipeline
        could commit a blob whose hash does not match the stored form
        anywhere else in the audit trail.
        """
        from elspeth.web.blobs.protocol import BlobStateError

        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="pipe.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        uppercase_hash = content_hash(b"real-bytes").upper()

        with pytest.raises(BlobStateError, match="64 lowercase hex"):
            blob_service._finalize_blob_sync(
                pending.id,
                "ready",
                size_bytes=10,
                content_hash_val=uppercase_hash,
            )

    @pytest.mark.asyncio
    async def test_sync_path_allows_error_status_without_hash(self, blob_service, session_id) -> None:
        """The hash invariant applies only to 'ready'; 'error' requires nothing."""
        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="pipe.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        record = blob_service._finalize_blob_sync(
            pending.id,
            "error",
            size_bytes=None,
            content_hash_val=None,
        )
        assert record.status == "error"
        assert record.content_hash is None

    @pytest.mark.asyncio
    async def test_sync_path_invalid_status_raises_runtime_error(self, blob_service, session_id) -> None:
        """Invalid status on the sync path must propagate as RuntimeError.

        _PER_BLOB_SUPPRESSED deliberately excludes RuntimeError so a
        programmer bug (typo'd status literal) crashes the pipeline
        finalization loop rather than being converted silently into a
        per-blob 'error' record.  BlobStateError would have been
        suppressed — so this test pins the crash-not-suppress contract.
        """
        pending = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="pipe.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        with pytest.raises(RuntimeError, match="Invalid finalize status"):
            blob_service._finalize_blob_sync(
                pending.id,
                "deleted",
                size_bytes=None,
                content_hash_val=None,
            )


# ---------------------------------------------------------------------------
# link_blob_to_run — runtime guard on BlobRunLinkDirection (elspeth-b6ac739b83)
# ---------------------------------------------------------------------------


class TestLinkBlobToRunDirectionGuard:
    """link_blob_to_run rejects direction values outside the Literal set."""

    @staticmethod
    def _make_run(db_engine, session_id: UUID) -> UUID:
        """Seed a composition state and run for FK satisfaction."""
        from elspeth.web.sessions.models import (
            composition_states_table,
            runs_table,
        )

        state_id = str(uuid4())
        run_id = str(uuid4())
        session_id_str = str(session_id)
        now = datetime.now(UTC)
        with db_engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=session_id_str,
                    version=1,
                    is_valid=True,
                    created_at=now,
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=run_id,
                    session_id=session_id_str,
                    state_id=state_id,
                    status="running",
                    started_at=now,
                    rows_processed=0,
                    rows_failed=0,
                )
            )
        return UUID(run_id)

    @pytest.mark.asyncio
    async def test_rejects_invalid_direction(self, blob_service, session_id, db_engine) -> None:
        """A typo'd direction must raise RuntimeError before touching the DB.

        Mirrors finalize_blob's invariant: the Literal alias narrows
        static callers, but the runtime guard catches dynamic / untyped
        call sites.  RuntimeError is the crash-not-suppress classification
        for "caller passed a value outside the Literal set."
        """
        run_id = self._make_run(db_engine, session_id)
        blob = await blob_service.create_blob(
            session_id=session_id,
            filename="input.csv",
            content=b"a,b,c\n1,2,3\n",
            mime_type="text/csv",
            created_by="user",
        )

        with pytest.raises(RuntimeError, match="Invalid link direction"):
            await blob_service.link_blob_to_run(
                blob_id=blob.id,
                run_id=run_id,
                direction="inout",
            )

    @pytest.mark.asyncio
    async def test_accepts_input_and_output(self, blob_service, session_id, db_engine) -> None:
        """Positive control: both valid directions commit without error."""
        run_id = self._make_run(db_engine, session_id)
        blob = await blob_service.create_blob(
            session_id=session_id,
            filename="input.csv",
            content=b"a,b,c\n1,2,3\n",
            mime_type="text/csv",
            created_by="user",
        )

        await blob_service.link_blob_to_run(blob.id, run_id, "input")
        await blob_service.link_blob_to_run(blob.id, run_id, "output")

        links = await blob_service.get_blob_run_links(blob.id)
        directions = sorted(link.direction for link in links)
        assert directions == ["input", "output"]


class TestLinkBlobToRunSessionGuard:
    """link_blob_to_run must reject cross-session references at the write boundary."""

    @pytest.mark.asyncio
    async def test_rejects_cross_session_link(self, blob_service, session_id, db_engine) -> None:
        """Blob and run from different sessions must not be linkable."""
        session_b = UUID(str(uuid4()))
        now = datetime.now(UTC)
        with db_engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=str(session_b),
                    user_id="test-user-b",
                    auth_provider_type="local",
                    title="Session B",
                    created_at=now,
                    updated_at=now,
                )
            )

        foreign_run_id = TestLinkBlobToRunDirectionGuard._make_run(db_engine, session_b)
        blob = await blob_service.create_blob(
            session_id=session_id,
            filename="input.csv",
            content=b"a,b,c\n1,2,3\n",
            mime_type="text/csv",
            created_by="user",
        )

        with pytest.raises(RuntimeError, match="cross-session reference"):
            await blob_service.link_blob_to_run(blob.id, foreign_run_id, "input")

        assert await blob_service.get_blob_run_links(blob.id) == []


# ---------------------------------------------------------------------------
# Tier-1 read guards — audit-trail integrity for DB-sourced rows
# ---------------------------------------------------------------------------


class TestRowToRecordTierOneGuards:
    """Tier-1 read guards in ``_row_to_record`` / ``_row_to_link_record``.

    Context
    -------
    ``BlobRecord.status``, ``BlobRecord.created_by``, ``BlobRecord.mime_type``,
    and ``BlobRunLinkRecord.direction`` are declared as closed ``Literal``
    types. The write paths enforce this via CHECK constraints
    (``ck_blobs_status``, ``ck_blobs_created_by``, ``ck_blob_run_links_direction``)
    and an ``ALLOWED_MIME_TYPES`` membership check at create time.

    The read paths add a second line of defence: assertions inside
    ``_row_to_record`` / ``_row_to_link_record`` that crash if a row ever
    reaches Python with a value outside the declared enum. This matters
    because CHECK constraints can be bypassed by:

    - Direct driver writes (raw SQL, another service writing to the file)
    - A migration bug that drops or loosens the constraint
    - ``PRAGMA ignore_check_constraints`` during maintenance
    - Binary corruption of the sqlite file

    Without the Python-side guard, the returned ``BlobRecord`` would carry
    a ``status`` value that is a lie about its static type, and the
    audit trail would confidently return fabricated data.

    These tests synthesise raw row-like objects (``SimpleNamespace``) and
    feed them through the private helpers to confirm the guard trips. The
    tests deliberately do *not* route through the DB — the point is that
    even a row that somehow slipped past the write-side constraints is
    caught at the read boundary. If anyone weakens the guards (deletes an
    assertion, loosens a membership set, swaps ``in`` for an always-true
    comparison), these tests will fail.

    Note on ``python -O``: the guards are implemented with explicit
    ``raise AuditIntegrityError(...)`` (not ``assert``) so they survive
    optimised interpreter execution.  The Tier-1 DB-corruption contract
    is AuditIntegrityError; tests below pin that type so a silent
    downgrade back to ``assert`` (which ``-O`` strips) would fail here.
    """

    @staticmethod
    def _fake_blob_row(**overrides) -> SimpleNamespace:
        """Build a SQLAlchemy-Row-shaped stand-in with valid defaults.

        Any field can be overridden to force the guard under test.
        """
        defaults = {
            "id": str(uuid4()),
            "session_id": str(uuid4()),
            "filename": "data.csv",
            "mime_type": "text/csv",
            "size_bytes": 42,
            "content_hash": hashlib.sha256(b"x").hexdigest(),
            "storage_path": "/tmp/blobs/x.csv",
            "created_at": datetime.now(UTC),
            "created_by": "user",
            "source_description": None,
            "status": "ready",
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    @staticmethod
    def _fake_link_row(**overrides) -> SimpleNamespace:
        defaults = {
            "blob_id": str(uuid4()),
            "run_id": str(uuid4()),
            "direction": "input",
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    # ---- positive control -------------------------------------------------

    def test_valid_row_returns_record(self, blob_service) -> None:
        """Positive control: a row with all-valid values round-trips.

        Without this, a bug that makes every row fail would be
        indistinguishable from the guard tripping correctly.
        """
        row = self._fake_blob_row()
        record = blob_service._row_to_record(row)
        assert record.status == "ready"
        assert record.created_by == "user"
        assert record.mime_type == "text/csv"

    def test_valid_link_row_returns_record(self, blob_service) -> None:
        row = self._fake_link_row(direction="output")
        record = blob_service._row_to_link_record(row)
        assert record.direction == "output"

    # ---- status guard -----------------------------------------------------

    def test_status_outside_enum_trips_guard(self, blob_service) -> None:
        """A tampered/corrupt row with ``status`` outside BLOB_STATUSES
        must crash with a Tier-1 assertion message before the BlobRecord
        is constructed with the lie."""
        row = self._fake_blob_row(status="corrupted")
        with pytest.raises(AuditIntegrityError, match=r"Tier 1: blobs\.status is 'corrupted'"):
            blob_service._row_to_record(row)

    def test_status_none_trips_guard(self, blob_service) -> None:
        """NULL status — e.g. from a dropped NOT NULL + DEFAULT during
        migration — is outside the enum and must crash."""
        row = self._fake_blob_row(status=None)
        with pytest.raises(AuditIntegrityError, match=r"Tier 1: blobs\.status"):
            blob_service._row_to_record(row)

    # ---- created_by guard ------------------------------------------------

    def test_created_by_outside_enum_trips_guard(self, blob_service) -> None:
        """An attacker who inserted a row directly (bypassing CHECK) with
        ``created_by = 'root'`` would otherwise surface as a valid record
        whose audit attribution is fabricated."""
        row = self._fake_blob_row(created_by="root")
        with pytest.raises(AuditIntegrityError, match=r"Tier 1: blobs\.created_by is 'root'"):
            blob_service._row_to_record(row)

    def test_created_by_empty_string_trips_guard(self, blob_service) -> None:
        row = self._fake_blob_row(created_by="")
        with pytest.raises(AuditIntegrityError, match=r"Tier 1: blobs\.created_by"):
            blob_service._row_to_record(row)

    # ---- mime_type guard -------------------------------------------------

    def test_mime_type_outside_allowlist_trips_guard(self, blob_service) -> None:
        """A row with an unallowed MIME type (e.g. ``application/x-sh``) must
        crash — the allowlist exists to constrain what the composer/pipeline
        layer will accept, and a laundered MIME would silently bypass it."""
        row = self._fake_blob_row(mime_type="application/x-sh")
        with pytest.raises(AuditIntegrityError, match=r"Tier 1: blobs\.mime_type is 'application/x-sh'"):
            blob_service._row_to_record(row)

    def test_mime_type_case_mismatch_trips_guard(self, blob_service) -> None:
        """Membership in ``ALLOWED_MIME_TYPES`` is case-sensitive by
        construction (the Literal values are lowercase). A row with
        ``TEXT/CSV`` has the wrong casing and must be rejected — not
        coerced, because coercion at the Tier-1 boundary is forbidden."""
        row = self._fake_blob_row(mime_type="TEXT/CSV")
        with pytest.raises(AuditIntegrityError, match=r"Tier 1: blobs\.mime_type"):
            blob_service._row_to_record(row)

    # ---- direction guard -------------------------------------------------

    def test_link_direction_outside_enum_trips_guard(self, blob_service) -> None:
        """``BlobRunLinkRecord.direction`` is typed as the Literal pair
        ``('input', 'output')``. A row with ``direction='inout'`` (the exact
        value the write-side test rejects) must also be rejected on read."""
        row = self._fake_link_row(direction="inout")
        with pytest.raises(AuditIntegrityError, match=r"Tier 1: blob_run_links\.direction is 'inout'"):
            blob_service._row_to_link_record(row)

    def test_link_direction_none_trips_guard(self, blob_service) -> None:
        row = self._fake_link_row(direction=None)
        with pytest.raises(AuditIntegrityError, match=r"Tier 1: blob_run_links\.direction"):
            blob_service._row_to_link_record(row)

    # ---- guard-fires-before-record-construction --------------------------

    def test_bad_status_crashes_before_uuid_parse(self, blob_service) -> None:
        """The Tier-1 guard must fire before any field coercion (e.g.
        ``UUID(row.id)``). This pins the guard's position at the top of
        ``_row_to_record`` — a refactor that moves assertions after the
        ``BlobRecord(...)`` call would pass through a fabricated record to
        anything that catches the later error."""
        # ``id`` is a non-parseable string; if the guard were moved, the
        # UUID constructor would raise ValueError first and mask the
        # tampered-status condition.
        row = self._fake_blob_row(status="corrupted", id="not-a-uuid")
        with pytest.raises(AuditIntegrityError, match=r"Tier 1: blobs\.status"):
            blob_service._row_to_record(row)

    def test_bad_direction_crashes_before_uuid_parse(self, blob_service) -> None:
        row = self._fake_link_row(direction="inout", blob_id="not-a-uuid")
        with pytest.raises(AuditIntegrityError, match=r"Tier 1: blob_run_links\.direction"):
            blob_service._row_to_link_record(row)
