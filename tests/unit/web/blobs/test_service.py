"""Tests for BlobServiceImpl — audit-critical blob persistence and lifecycle.

Security boundaries tested:
- Content hash integrity (AD-5/AD-7: hash must match for lineage verification)
- Session-scoped isolation (blobs cannot leak across sessions)
- Active-run deletion guard (cannot destroy evidence during a live run)
- Filename sanitization (path traversal defense at the storage layer)
- Status lifecycle (pending -> ready/error only; no backwards transitions)
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import json
import multiprocessing
import os
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import structlog
from sqlalchemy import delete, event, func, insert, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

from elspeth.contracts.enums import CreationModality
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.blobs import service as blob_service_module
from elspeth.web.blobs.protocol import (
    BlobActiveRunError,
    BlobForkCleanupError,
    BlobForkCleanupResult,
    BlobForkFenceLostError,
    BlobForkPlanEntry,
    BlobForkWriteFence,
    BlobGuidedOperationFenceLostError,
    BlobGuidedOperationWriteFence,
    BlobInProgressForkError,
    BlobIntegrityError,
    BlobNotFoundError,
    BlobQuotaExceededError,
    fork_blob_id,
)
from elspeth.web.blobs.service import (
    BlobServiceImpl,
    content_hash,
    sanitize_filename,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    blobs_table,
    chat_messages_table,
    composition_proposals_table,
    guided_operations_table,
    sessions_table,
)
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import _FakeCounter, build_sessions_telemetry

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
    async def test_delete_blob_commit_failure_restores_file_and_row_after_restart(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        db_engine,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed metadata commit must not strand a live row without bytes."""
        content = b"commit-boundary-content"
        record = await blob_service.create_blob(
            session_id=session_id,
            filename="commit-boundary.csv",
            content=content,
            mime_type="text/csv",
            created_by="user",
        )
        storage = Path(record.storage_path)
        original_do_commit = db_engine.dialect.do_commit
        fail_next_commit = True

        def fail_delete_commit(dbapi_connection) -> None:
            nonlocal fail_next_commit
            if fail_next_commit:
                fail_next_commit = False
                raise RuntimeError("injected blob delete commit failure")
            original_do_commit(dbapi_connection)

        monkeypatch.setattr(db_engine.dialect, "do_commit", fail_delete_commit)

        with pytest.raises(RuntimeError, match="injected blob delete commit failure"):
            await blob_service.delete_blob(record.id)

        restarted = BlobServiceImpl(db_engine, tmp_path)
        restored = await restarted.get_blob(record.id)
        assert restored.id == record.id
        assert storage.read_bytes() == content
        assert await restarted.read_blob_content(record.id) == content
        assert list(storage.parent.glob(f".{storage.name}.delete-*")) == []

    @pytest.mark.asyncio
    async def test_delete_blob_sql_failure_restores_file_before_stage_escapes(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        db_engine,
        tmp_path: Path,
    ) -> None:
        """A DELETE failure inside the helper restores its unreturned stage."""
        content = b"delete-statement-boundary-content"
        record = await blob_service.create_blob(
            session_id=session_id,
            filename="delete-statement-boundary.csv",
            content=content,
            mime_type="text/csv",
            created_by="user",
        )
        storage = Path(record.storage_path)

        def fail_blob_delete(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:
            if statement.lstrip().upper().startswith("DELETE FROM BLOBS"):
                raise RuntimeError("injected blob DELETE failure")

        event.listen(db_engine, "before_cursor_execute", fail_blob_delete)
        try:
            with pytest.raises(RuntimeError, match="injected blob DELETE failure"):
                await blob_service.delete_blob(record.id)
        finally:
            event.remove(db_engine, "before_cursor_execute", fail_blob_delete)

        restarted = BlobServiceImpl(db_engine, tmp_path)
        assert (await restarted.get_blob(record.id)).id == record.id
        assert await restarted.read_blob_content(record.id) == content
        assert list(storage.parent.glob(f".{storage.name}.delete-*")) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool_name", "arguments"),
        [
            ("set_pipeline", lambda blob_id: {"source": {"blob_id": blob_id}}),
            ("set_source_from_blob", lambda blob_id: {"blob_id": blob_id}),
            ("update_blob", lambda blob_id: {"blob_id": blob_id}),
            ("wire_blob_inline_ref", lambda blob_id: {"blob_id": blob_id}),
        ],
    )
    async def test_delete_blob_rejects_blob_referenced_by_pending_proposal(
        self,
        blob_service,
        session_id,
        db_engine,
        tool_name,
        arguments,
    ) -> None:
        contracts = importlib.import_module("elspeth.contracts.blobs")
        pending_error = contracts.BlobPendingProposalError
        record = await blob_service.create_blob(
            session_id=session_id,
            filename="pending-review.csv",
            content=b"review me",
            mime_type="text/csv",
            created_by="assistant",
        )
        now = datetime.now(UTC)
        with db_engine.begin() as conn:
            conn.execute(
                insert(composition_proposals_table).values(
                    id=str(uuid4()),
                    session_id=str(session_id),
                    tool_call_id="call_pending_blob_delete_guard",
                    tool_name=tool_name,
                    status="pending",
                    summary="Review blob-backed pipeline",
                    rationale="Server generated",
                    affects=["source"],
                    arguments_json=arguments(str(record.id)),
                    arguments_redacted_json=arguments(str(record.id)),
                    base_state_id=None,
                    committed_state_id=None,
                    audit_event_id=None,
                    created_at=now,
                    updated_at=now,
                )
            )

        with pytest.raises(pending_error):
            await blob_service.delete_blob(record.id)

        assert Path(record.storage_path).exists()
        assert await blob_service.get_blob(record.id) == record

    @pytest.mark.asyncio
    async def test_delete_blob_allows_blob_only_referenced_by_rejected_proposal(self, blob_service, session_id, db_engine) -> None:
        record = await blob_service.create_blob(
            session_id=session_id,
            filename="rejected-review.csv",
            content=b"no longer retained",
            mime_type="text/csv",
            created_by="assistant",
        )
        now = datetime.now(UTC)
        with db_engine.begin() as conn:
            conn.execute(
                insert(composition_proposals_table).values(
                    id=str(uuid4()),
                    session_id=str(session_id),
                    tool_call_id="call_rejected_blob_delete_guard",
                    tool_name="set_pipeline",
                    status="rejected",
                    summary="Rejected blob-backed pipeline",
                    rationale="Server generated",
                    affects=["source"],
                    arguments_json={"source": {"blob_id": str(record.id)}},
                    arguments_redacted_json={"source": {"blob_id": str(record.id)}},
                    base_state_id=None,
                    committed_state_id=None,
                    audit_event_id=None,
                    created_at=now,
                    updated_at=now,
                )
            )

        await blob_service.delete_blob(record.id)

        with pytest.raises(BlobNotFoundError):
            await blob_service.get_blob(record.id)

    @pytest.mark.asyncio
    async def test_pending_delete_proposal_does_not_retain_its_own_target(self, blob_service, session_id, db_engine) -> None:
        record = await blob_service.create_blob(
            session_id=session_id,
            filename="delete-target.csv",
            content=b"delete me",
            mime_type="text/csv",
            created_by="assistant",
        )
        now = datetime.now(UTC)
        with db_engine.begin() as conn:
            conn.execute(
                insert(composition_proposals_table).values(
                    id=str(uuid4()),
                    session_id=str(session_id),
                    tool_call_id="call_delete_blob_proposal",
                    tool_name="delete_blob",
                    status="pending",
                    summary="Delete blob",
                    rationale="Server generated",
                    affects=["blob"],
                    arguments_json={"blob_id": str(record.id)},
                    arguments_redacted_json={"blob_id": str(record.id)},
                    base_state_id=None,
                    committed_state_id=None,
                    audit_event_id=None,
                    created_at=now,
                    updated_at=now,
                )
            )

        await blob_service.delete_blob(record.id)
        with pytest.raises(BlobNotFoundError):
            await blob_service.get_blob(record.id)

    @pytest.mark.asyncio
    async def test_unrelated_nested_blob_id_does_not_create_pending_retention(self, blob_service, session_id, db_engine) -> None:
        record = await blob_service.create_blob(
            session_id=session_id,
            filename="unrelated.csv",
            content=b"not a source binding",
            mime_type="text/csv",
            created_by="assistant",
        )
        now = datetime.now(UTC)
        with db_engine.begin() as conn:
            conn.execute(
                insert(composition_proposals_table).values(
                    id=str(uuid4()),
                    session_id=str(session_id),
                    tool_call_id="call_unrelated_blob_id",
                    tool_name="set_pipeline",
                    status="pending",
                    summary="Unrelated nested value",
                    rationale="Server generated",
                    affects=["node"],
                    arguments_json={
                        "source": {"plugin": "csv", "options": {}},
                        "nodes": [{"options": {"blob_id": str(record.id)}}],
                    },
                    arguments_redacted_json={"source": {"plugin": "csv", "options": {}}},
                    base_state_id=None,
                    committed_state_id=None,
                    audit_event_id=None,
                    created_at=now,
                    updated_at=now,
                )
            )

        await blob_service.delete_blob(record.id)
        with pytest.raises(BlobNotFoundError):
            await blob_service.get_blob(record.id)

    @pytest.mark.asyncio
    async def test_pending_proposal_retention_does_not_block_blob_finalization(self, blob_service, session_id, db_engine) -> None:
        record = await blob_service.create_pending_blob(
            session_id=session_id,
            filename="pending-output.csv",
            mime_type="text/csv",
            created_by="assistant",
        )
        content = b"ready\n1\n"
        Path(record.storage_path).write_bytes(content)
        now = datetime.now(UTC)
        with db_engine.begin() as conn:
            conn.execute(
                insert(composition_proposals_table).values(
                    id=str(uuid4()),
                    session_id=str(session_id),
                    tool_call_id="call_pending_finalize",
                    tool_name="set_pipeline",
                    status="pending",
                    summary="Retain pending output",
                    rationale="Server generated",
                    affects=["source"],
                    arguments_json={"source": {"blob_id": str(record.id)}},
                    arguments_redacted_json={"source": {"blob_id": str(record.id)}},
                    base_state_id=None,
                    committed_state_id=None,
                    audit_event_id=None,
                    created_at=now,
                    updated_at=now,
                )
            )

        finalized = await blob_service.finalize_blob(
            record.id,
            "ready",
            size_bytes=len(content),
            content_hash=content_hash(content),
        )
        assert finalized.status == "ready"

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
                    # Plan §2294: every test-side direct composition_states
                    # insert must supply provenance after Task 3's CHECK
                    # constraint. ``session_seed`` is the broadened-semantics
                    # default for setup-only rows that don't model a real
                    # compose-loop transition.
                    provenance="session_seed",
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
                    # Plan §2294: every test-side direct composition_states
                    # insert must supply provenance after Task 3's CHECK
                    # constraint. ``session_seed`` is the broadened-semantics
                    # default for setup-only rows that don't model a real
                    # compose-loop transition.
                    provenance="session_seed",
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
    async def test_delete_blob_preserves_completed_inline_resolution_audit_rows(self, blob_service, session_id, db_engine) -> None:
        """Completed inline-content audit rows must not turn blob deletion into a 500."""
        from elspeth.web.sessions.models import (
            blob_inline_resolutions_table,
            blob_run_links_table,
            composition_states_table,
            runs_table,
        )

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="prompt.txt",
            content=b"finished prompt",
            mime_type="text/plain",
            created_by="user",
        )

        state_id = str(uuid4())
        session_id_str = str(session_id)
        run_id = str(uuid4())
        now = datetime(2026, 1, 1, tzinfo=UTC)

        with db_engine.begin() as conn:
            conn.execute(
                composition_states_table.insert().values(
                    id=state_id,
                    session_id=session_id_str,
                    version=1,
                    is_valid=True,
                    provenance="session_seed",
                    created_at=now,
                )
            )
            conn.execute(
                runs_table.insert().values(
                    id=run_id,
                    session_id=session_id_str,
                    state_id=state_id,
                    status="completed",
                    started_at=now,
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
            conn.execute(
                blob_inline_resolutions_table.insert().values(
                    run_id=run_id,
                    attempt=1,
                    field_path="node:classify.options.system_prompt",
                    blob_id=str(record.id),
                    content_hash=record.content_hash,
                    byte_length=record.size_bytes,
                    mime_type=record.mime_type,
                    encoding="utf-8",
                    resolved_at=now,
                )
            )

        await blob_service.delete_blob(record.id)

        with db_engine.connect() as conn:
            rows = conn.execute(select(blob_inline_resolutions_table)).fetchall()

        assert len(rows) == 1
        assert rows[0].blob_id == str(record.id)
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
                        "on_success": "output",
                        "on_validation_failure": "quarantine",
                        "options": {"blob_ref": str(record.id), "path": str(record.storage_path)},
                    },
                    nodes=[],
                    edges=[],
                    outputs=[],
                    metadata_={"name": "Test", "description": ""},
                    is_valid=True,
                    # Plan §2294: every test-side direct composition_states
                    # insert must supply provenance after Task 3's CHECK
                    # constraint. ``session_seed`` is the broadened-semantics
                    # default for setup-only rows that don't model a real
                    # compose-loop transition.
                    provenance="session_seed",
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
                        "on_success": "output",
                        "on_validation_failure": "quarantine",
                        "options": {"path": "/data/external/other.csv"},
                    },
                    nodes=[],
                    edges=[],
                    outputs=[],
                    metadata_={"name": "Test", "description": ""},
                    is_valid=True,
                    # Plan §2294: every test-side direct composition_states
                    # insert must supply provenance after Task 3's CHECK
                    # constraint. ``session_seed`` is the broadened-semantics
                    # default for setup-only rows that don't model a real
                    # compose-loop transition.
                    provenance="session_seed",
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
    async def test_delete_blob_rejects_when_transform_option_references_blob(self, blob_service, session_id, db_engine) -> None:
        """Pre-link guard walks canonical pipeline sections beyond source.options."""
        from elspeth.web.sessions.models import (
            composition_states_table,
            runs_table,
        )

        record = await blob_service.create_blob(
            session_id=session_id,
            filename="prompt.txt",
            content=b"Classify the row.",
            mime_type="text/plain",
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
                    source={
                        "plugin": "csv",
                        "on_success": "classify",
                        "on_validation_failure": "quarantine",
                        "options": {"path": "/data/external/other.csv"},
                    },
                    nodes=[
                        {
                            "id": "classify",
                            "node_type": "transform",
                            "plugin": "llm",
                            "input": "source_out",
                            "on_success": "output",
                            "on_error": "discard",
                            "options": {
                                "system_prompt": {
                                    "blob_ref": str(record.id),
                                    "mode": "inline_content",
                                    "sha256": record.content_hash,
                                }
                            },
                        }
                    ],
                    edges=[],
                    outputs=[],
                    metadata_={"name": "Test", "description": ""},
                    is_valid=True,
                    provenance="session_seed",
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
                        "on_success": "output",
                        "on_validation_failure": "quarantine",
                        "options": {"path": record.storage_path},
                    },
                    nodes=[],
                    edges=[],
                    outputs=[],
                    metadata_={"name": "Test", "description": ""},
                    is_valid=True,
                    # Plan §2294: every test-side direct composition_states
                    # insert must supply provenance after Task 3's CHECK
                    # constraint. ``session_seed`` is the broadened-semantics
                    # default for setup-only rows that don't model a real
                    # compose-loop transition.
                    provenance="session_seed",
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
                    # Plan §2294: every test-side direct composition_states
                    # insert must supply provenance after Task 3's CHECK
                    # constraint. ``session_seed`` is the broadened-semantics
                    # default for setup-only rows that don't model a real
                    # compose-loop transition.
                    provenance="session_seed",
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


class TestCreatePendingBlob:
    """Pending blob reservation must enforce the same literal guards as ready writes."""

    @pytest.mark.asyncio
    async def test_create_pending_blob_rejects_disallowed_mime_type(self, blob_service, session_id) -> None:
        """Pending rows must not persist MIME values that read guards classify as corruption."""
        with pytest.raises(RuntimeError, match="Invalid mime_type"):
            await blob_service.create_pending_blob(
                session_id=session_id,
                filename="output.png",
                mime_type="image/png",  # type: ignore[arg-type]
                created_by="pipeline",
            )


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

    def test_quota_lock_statement_serializes_session_writers_on_postgresql(self) -> None:
        """Quota writers must lock the session row on MVCC databases."""
        statement = blob_service_module._session_quota_lock_statement("session-1")

        compiled = str(statement.compile(dialect=postgresql.dialect()))

        assert "FROM sessions" in compiled
        assert "WHERE sessions.id = " in compiled
        assert "FOR UPDATE" in compiled

    @pytest.mark.asyncio
    async def test_create_blob_locks_session_before_quota_sum(self, db_engine, session_id, tmp_path, monkeypatch) -> None:
        """create_blob must serialize same-session quota writers before SUM+insert."""
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=200)
        locked_sessions: list[str] = []
        original_lock = blob_service_module._lock_session_for_blob_quota

        def recording_lock(conn, session_id_str: str) -> None:
            locked_sessions.append(session_id_str)
            original_lock(conn, session_id_str)

        monkeypatch.setattr(blob_service_module, "_lock_session_for_blob_quota", recording_lock)

        await service.create_blob(
            session_id=session_id,
            filename="serialized.csv",
            content=b"x" * 50,
            mime_type="text/csv",
            created_by="user",
        )

        assert locked_sessions == [str(session_id)]

    @pytest.mark.asyncio
    async def test_finalize_blob_locks_session_before_quota_sum(self, db_engine, session_id, tmp_path, monkeypatch) -> None:
        """finalize_blob must serialize same-session quota writers before SUM+update."""
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=200)
        pending = await service.create_pending_blob(
            session_id=session_id,
            filename="serialized-output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )
        locked_sessions: list[str] = []
        original_lock = blob_service_module._lock_session_for_blob_quota

        def recording_lock(conn, session_id_str: str) -> None:
            locked_sessions.append(session_id_str)
            original_lock(conn, session_id_str)

        monkeypatch.setattr(blob_service_module, "_lock_session_for_blob_quota", recording_lock)

        await service.finalize_blob(
            blob_id=pending.id,
            status="ready",
            size_bytes=50,
            content_hash=content_hash(b"finalized"),
        )

        assert locked_sessions == [str(session_id)]

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

    @pytest.mark.asyncio
    async def test_finalize_blob_rejects_ready_size_that_exceeds_quota(self, db_engine, session_id, tmp_path) -> None:
        """Public finalize_blob must enforce quota when pending output size becomes known."""
        from elspeth.web.blobs.protocol import BlobQuotaExceededError

        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=10)
        pending = await service.create_pending_blob(
            session_id=session_id,
            filename="oversized-output.csv",
            mime_type="text/csv",
            created_by="pipeline",
        )

        with pytest.raises(BlobQuotaExceededError):
            await service.finalize_blob(
                blob_id=pending.id,
                status="ready",
                size_bytes=100,
                content_hash=content_hash(b"oversized-output"),
            )

        record = await service.get_blob(pending.id)
        assert record.status == "pending"
        assert record.size_bytes == 0
        assert record.content_hash is None


# ---------------------------------------------------------------------------
# inline custody — deterministic identity, recovery, and exactly-once quota
# ---------------------------------------------------------------------------


def _inline_custody_contract() -> tuple[type[object], object]:
    """Load the desired Task-2 contract inside the test for a clean RED."""
    contracts = importlib.import_module("elspeth.contracts.blobs")
    return contracts.InlineCustodyRequest, blob_service_module.inline_custody_blob_id


def _seed_custody_message(db_engine, session_id: UUID) -> str:
    message_id = str(uuid4())
    now = datetime.now(UTC)
    with db_engine.begin() as conn:
        conn.execute(
            insert(chat_messages_table).values(
                id=message_id,
                session_id=str(session_id),
                role="user",
                content="Create the inline source.",
                raw_content=None,
                tool_calls=None,
                tool_call_id=None,
                sequence_no=1,
                writer_principal="route_user_message",
                created_at=now,
                composition_state_id=None,
                parent_assistant_id=None,
            )
        )
    return message_id


def _custody_request(db_engine, session_id: UUID, *, content: bytes = b"value\n42\n", description: str | None = "candidate") -> object:
    request_type, _ = _inline_custody_contract()
    return request_type(
        session_id=session_id,
        filename="candidate.csv",
        content=content,
        mime_type="text/csv",
        source_description=description,
        creation_modality=CreationModality.VERBATIM,
        created_from_message_id=_seed_custody_message(db_engine, session_id),
        creating_model_identifier=None,
        creating_model_version=None,
        creating_provider=None,
        creating_composer_skill_hash=None,
        creating_arguments_hash=None,
    )


def _custody_process(
    database_url: str,
    data_dir: str,
    request_fields: dict[str, object],
    start_event: object,
    result_queue: object,
) -> None:
    """Spawn-safe worker proving PostgreSQL exclusion crosses processes."""
    request_type, _ = _inline_custody_contract()
    engine = create_session_engine(database_url)
    normalized_fields = dict(request_fields)
    normalized_fields["session_id"] = UUID(str(request_fields["session_id"]))
    normalized_fields["creation_modality"] = CreationModality(str(request_fields["creation_modality"]))
    request = request_type(**normalized_fields)
    try:
        if not start_event.wait(timeout=15):  # type: ignore[attr-defined]
            raise RuntimeError("PostgreSQL custody process start barrier timed out")
        record = asyncio.run(BlobServiceImpl(engine, Path(data_dir), max_storage_per_session=100).reserve_inline_custody(request))
        result_queue.put(("ok", str(record.id)))  # type: ignore[attr-defined]
    except BaseException as exc:
        result_queue.put(("error", type(exc).__name__, str(exc)))  # type: ignore[attr-defined]
    finally:
        engine.dispose()


class TestInlineCustody:
    @staticmethod
    def _guided_operation_write_fence(
        db_engine,
        session_id: UUID,
        *,
        kind: str = "guided_plan",
    ) -> BlobGuidedOperationWriteFence:
        operation_id = str(uuid4())
        lease_token = uuid4().hex
        now = datetime.now(UTC)
        with db_engine.begin() as conn:
            conn.execute(
                guided_operations_table.insert().values(
                    session_id=str(session_id),
                    operation_id=operation_id,
                    kind=kind,
                    status="in_progress",
                    request_hash="a" * 64,
                    lease_token=lease_token,
                    lease_expires_at=now + timedelta(hours=1),
                    attempt=1,
                    created_at=now,
                    updated_at=now,
                )
            )
        return BlobGuidedOperationWriteFence(
            session_id=session_id,
            operation_id=operation_id,
            lease_token=lease_token,
            attempt=1,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("kind", ["guided_plan", "guided_respond"])
    async def test_guided_inline_custody_accepts_closed_planning_operation_kinds(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
        kind: str,
    ) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        request = _custody_request(db_engine, session_id)
        fence = self._guided_operation_write_fence(db_engine, session_id, kind=kind)

        record = await service.reserve_inline_custody(request, write_fence=fence)

        assert record.status == "ready"
        assert Path(record.storage_path).read_bytes() == request.content

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalidity", ["wrong_kind", "wrong_token", "wrong_attempt"])
    async def test_guided_inline_custody_requires_live_fence_at_reservation(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
        invalidity: str,
    ) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        request = _custody_request(db_engine, session_id)
        fence = self._guided_operation_write_fence(
            db_engine,
            session_id,
            kind="guided_chat" if invalidity == "wrong_kind" else "guided_plan",
        )
        if invalidity == "wrong_token":
            fence = replace(fence, lease_token="wrong-token")
        elif invalidity == "wrong_attempt":
            fence = replace(fence, attempt=2)

        with pytest.raises(BlobGuidedOperationFenceLostError):
            await service.reserve_inline_custody(request, write_fence=fence)

        with db_engine.connect() as conn:
            assert conn.execute(select(func.count()).select_from(blobs_table)).scalar_one() == 0
        assert tuple(path for path in (tmp_path / "blobs").rglob("*") if path.is_file()) == ()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "takeover_values",
        [
            {"kind": "guided_chat"},
            {"lease_token": "takeover-lease"},
            {"attempt": 2},
        ],
        ids=["wrong-kind", "wrong-token", "wrong-attempt"],
    )
    async def test_guided_inline_custody_rechecks_fence_at_ready_write(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        takeover_values: dict[str, object],
    ) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        request = _custody_request(db_engine, session_id)
        fence = self._guided_operation_write_fence(db_engine, session_id)
        original_write = blob_service_module._write_or_validate_reserved_blob

        def _write_after_takeover(**kwargs):
            wrote = original_write(**kwargs)
            with db_engine.begin() as conn:
                changed = conn.execute(
                    guided_operations_table.update()
                    .where(guided_operations_table.c.session_id == str(session_id))
                    .where(guided_operations_table.c.operation_id == fence.operation_id)
                    .where(guided_operations_table.c.lease_token == fence.lease_token)
                    .where(guided_operations_table.c.attempt == fence.attempt)
                    .values(**takeover_values, updated_at=datetime.now(UTC))
                ).rowcount
            assert changed == 1
            return wrote

        monkeypatch.setattr(blob_service_module, "_write_or_validate_reserved_blob", _write_after_takeover)

        with pytest.raises(BlobGuidedOperationFenceLostError):
            await service.reserve_inline_custody(request, write_fence=fence)

        with db_engine.connect() as conn:
            row = conn.execute(select(blobs_table.c.status).where(blobs_table.c.session_id == str(session_id))).one()
        assert row.status == "pending"

    @pytest.mark.asyncio
    async def test_nonidempotent_duplicate_does_not_delete_existing_ready_file(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        fixed_blob_id = uuid4()
        monkeypatch.setattr(blob_service_module, "uuid4", lambda: fixed_blob_id)

        winner = await service.create_blob(
            session_id=session_id,
            filename="winner.csv",
            content=b"value\n42\n",
            mime_type="text/csv",
        )

        with pytest.raises(AuditIntegrityError, match="Unexpected duplicate blob id"):
            await service.create_blob(
                session_id=session_id,
                filename="winner.csv",
                content=b"value\n42\n",
                mime_type="text/csv",
            )

        assert Path(winner.storage_path).read_bytes() == b"value\n42\n"
        assert (await service.get_blob(fixed_blob_id)).status == "ready"

    @pytest.mark.asyncio
    async def test_nonidempotent_failure_preserves_preexisting_orphan_file(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from elspeth.web.blobs.protocol import BlobIntegrityError

        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        fixed_blob_id = uuid4()
        monkeypatch.setattr(blob_service_module, "uuid4", lambda: fixed_blob_id)
        storage = tmp_path.resolve() / "blobs" / str(session_id) / f"{fixed_blob_id}_orphan.csv"
        storage.parent.mkdir(parents=True)
        storage.write_bytes(b"preexisting integrity evidence")

        with pytest.raises(BlobIntegrityError):
            await service.create_blob(
                session_id=session_id,
                filename="orphan.csv",
                content=b"new bytes",
                mime_type="text/csv",
            )

        assert storage.read_bytes() == b"preexisting integrity evidence"
        with db_engine.connect() as conn:
            assert conn.execute(select(func.count()).select_from(blobs_table)).scalar_one() == 0

    def test_atomic_write_fsyncs_parent_directory_after_replace(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        storage = tmp_path / "blobs" / "session" / "blob_candidate.csv"
        fsynced: list[Path] = []
        monkeypatch.setattr(
            blob_service_module,
            "_fsync_parent_directory",
            lambda path: fsynced.append(path),
            raising=False,
        )

        blob_service_module._atomic_write_blob(storage, b"value\n42\n")

        assert storage.read_bytes() == b"value\n42\n"
        assert fsynced == [storage.parent]

    def test_uuid5_identity_covers_every_authority_field_except_final_arguments_hash(
        self,
        db_engine,
        session_id: UUID,
    ) -> None:
        request_type, derive_blob_id = _inline_custody_contract()
        base = request_type(
            session_id=session_id,
            filename="candidate.csv",
            content=b"value\n42\n",
            mime_type="text/csv",
            source_description="candidate",
            creation_modality=CreationModality.LLM_GENERATED,
            created_from_message_id=_seed_custody_message(db_engine, session_id),
            creating_model_identifier="model-a",
            creating_model_version="version-a",
            creating_provider="provider-a",
            creating_composer_skill_hash="a" * 64,
            creating_arguments_hash="b" * 64,
        )
        baseline = derive_blob_id(base)
        second_session = uuid4()
        with db_engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=str(second_session),
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Second Session",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
        variants = (
            replace(base, session_id=second_session),
            replace(base, created_from_message_id=str(uuid4())),
            replace(base, content=b"value\n43\n"),
            replace(base, mime_type="text/plain"),
            replace(base, filename="different.csv"),
            replace(base, source_description="different purpose"),
            replace(base, creation_modality=CreationModality.DISAMBIGUATED),
            replace(base, creating_model_identifier="model-b"),
            replace(base, creating_model_version="version-b"),
            replace(base, creating_provider="provider-b"),
            replace(base, creating_composer_skill_hash="c" * 64),
        )

        assert all(derive_blob_id(variant) != baseline for variant in variants)
        assert derive_blob_id(replace(base, filename="nested/candidate.csv")) == baseline
        assert derive_blob_id(replace(base, creating_arguments_hash="d" * 64)) == baseline

    @pytest.mark.parametrize("field_name", ["creating_composer_skill_hash", "creating_arguments_hash"])
    @pytest.mark.parametrize("invalid_hash", ["A" * 64, "a" * 63, "g" * 64, ""])
    def test_custody_rejects_noncanonical_provenance_hashes_without_echoing_values(
        self,
        db_engine,
        session_id: UUID,
        field_name: str,
        invalid_hash: str,
    ) -> None:
        request_type, derive_blob_id = _inline_custody_contract()
        values = {
            "session_id": session_id,
            "filename": "candidate.csv",
            "content": b"value\n42\n",
            "mime_type": "text/csv",
            "source_description": "candidate",
            "creation_modality": CreationModality.LLM_GENERATED,
            "created_from_message_id": _seed_custody_message(db_engine, session_id),
            "creating_model_identifier": "model-a",
            "creating_model_version": "version-a",
            "creating_provider": "provider-a",
            "creating_composer_skill_hash": "a" * 64,
            "creating_arguments_hash": "b" * 64,
        }
        values[field_name] = invalid_hash

        with pytest.raises(AuditIntegrityError) as exc_info:
            derive_blob_id(request_type(**values))

        assert field_name in str(exc_info.value)
        if invalid_hash:
            assert invalid_hash not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_reuses_uuid5_blob_and_charges_quota_once(self, db_engine, session_id: UUID, tmp_path: Path) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        request = _custody_request(db_engine, session_id)

        first = await service.reserve_inline_custody(request)
        retried = await service.reserve_inline_custody(request)

        assert first == retried
        assert first.id.version == 5
        assert first.status == "ready"
        assert Path(first.storage_path).read_bytes() == request.content
        with db_engine.connect() as conn:
            rows = conn.execute(select(func.count()).select_from(blobs_table)).scalar_one()
            charged = conn.execute(
                select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == str(session_id))
            ).scalar_one()
        assert rows == 1
        assert charged == len(request.content)

    @pytest.mark.asyncio
    async def test_concurrent_retries_converge_on_one_ready_blob(self, db_engine, session_id: UUID, tmp_path: Path) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        request = _custody_request(db_engine, session_id)

        records = await asyncio.gather(*(service.reserve_inline_custody(request) for _ in range(8)))

        assert {record.id for record in records} == {records[0].id}
        with db_engine.connect() as conn:
            assert conn.execute(select(func.count()).select_from(blobs_table)).scalar_one() == 1

    @pytest.mark.asyncio
    async def test_matching_pending_file_is_adopted_and_finalized(self, db_engine, session_id: UUID, tmp_path: Path) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        request = _custody_request(db_engine, session_id)
        _, derive_blob_id = _inline_custody_contract()
        blob_id = derive_blob_id(request)
        storage = tmp_path.resolve() / "blobs" / str(session_id) / f"{blob_id}_candidate.csv"
        storage.parent.mkdir(parents=True)
        storage.write_bytes(request.content)
        with db_engine.begin() as conn:
            conn.execute(
                insert(blobs_table).values(
                    id=str(blob_id),
                    session_id=str(session_id),
                    filename=request.filename,
                    mime_type=request.mime_type,
                    size_bytes=len(request.content),
                    content_hash=content_hash(request.content),
                    storage_path=str(storage),
                    created_at=datetime.now(UTC),
                    created_by="assistant",
                    source_description=request.source_description,
                    status="pending",
                    creation_modality=request.creation_modality.value,
                    created_from_message_id=request.created_from_message_id,
                    creating_model_identifier=None,
                    creating_model_version=None,
                    creating_provider=None,
                    creating_composer_skill_hash=None,
                    creating_arguments_hash=None,
                )
            )

        record = await service.reserve_inline_custody(request)

        assert record.status == "ready"
        assert record.id == blob_id

    @pytest.mark.asyncio
    async def test_retry_reconciles_deterministic_and_legacy_stale_temp_files(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
    ) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        request = _custody_request(db_engine, session_id)
        _, derive_blob_id = _inline_custody_contract()
        blob_id = derive_blob_id(request)
        storage = tmp_path.resolve() / "blobs" / str(session_id) / f"{blob_id}_candidate.csv"
        storage.parent.mkdir(parents=True)
        deterministic_temp = storage.with_name(f".{storage.name}.custody.tmp")
        legacy_temp = storage.with_name(f".{storage.name}.orphan.tmp")
        deterministic_temp.write_bytes(b"partial")
        legacy_temp.write_bytes(b"partial")

        record = await service.reserve_inline_custody(request)

        assert record.status == "ready"
        assert storage.read_bytes() == request.content
        assert not deterministic_temp.exists()
        assert not legacy_temp.exists()

    @pytest.mark.asyncio
    async def test_delete_removes_reconcilable_temp_artifacts(self, db_engine, session_id: UUID, tmp_path: Path) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        record = await service.create_blob(
            session_id=session_id,
            filename="candidate.csv",
            content=b"value\n42\n",
            mime_type="text/csv",
        )
        storage = Path(record.storage_path)
        deterministic_temp = storage.with_name(f".{storage.name}.custody.tmp")
        deterministic_temp.write_bytes(b"partial")

        await service.delete_blob(record.id)

        assert not storage.exists()
        assert not deterministic_temp.exists()

    @pytest.mark.asyncio
    async def test_mismatched_pending_file_fails_closed(self, db_engine, session_id: UUID, tmp_path: Path) -> None:
        from elspeth.web.blobs.protocol import BlobIntegrityError

        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        request = _custody_request(db_engine, session_id)
        _, derive_blob_id = _inline_custody_contract()
        blob_id = derive_blob_id(request)
        storage = tmp_path.resolve() / "blobs" / str(session_id) / f"{blob_id}_candidate.csv"
        storage.parent.mkdir(parents=True)
        storage.write_bytes(b"tampered")
        with db_engine.begin() as conn:
            conn.execute(
                insert(blobs_table).values(
                    id=str(blob_id),
                    session_id=str(session_id),
                    filename=request.filename,
                    mime_type=request.mime_type,
                    size_bytes=len(request.content),
                    content_hash=content_hash(request.content),
                    storage_path=str(storage),
                    created_at=datetime.now(UTC),
                    created_by="assistant",
                    source_description=request.source_description,
                    status="pending",
                    creation_modality=request.creation_modality.value,
                    created_from_message_id=request.created_from_message_id,
                    creating_model_identifier=None,
                    creating_model_version=None,
                    creating_provider=None,
                    creating_composer_skill_hash=None,
                    creating_arguments_hash=None,
                )
            )

        with pytest.raises(BlobIntegrityError):
            await service.reserve_inline_custody(request)

        assert storage.read_bytes() == b"tampered"

    @pytest.mark.asyncio
    async def test_retry_recovers_orphan_file_after_interruption_before_row_finalization(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        request = _custody_request(db_engine, session_id)
        _, derive_blob_id = _inline_custody_contract()
        blob_id = derive_blob_id(request)
        storage = tmp_path.resolve() / "blobs" / str(session_id) / f"{blob_id}_candidate.csv"
        original_write = blob_service_module._atomic_write_blob

        def _write_then_interrupt(path: Path, content: bytes) -> None:
            original_write(path, content)
            raise RuntimeError("simulated interruption after file write")

        monkeypatch.setattr(blob_service_module, "_atomic_write_blob", _write_then_interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            await service.reserve_inline_custody(request)

        assert storage.read_bytes() == request.content
        with db_engine.connect() as conn:
            row = conn.execute(select(blobs_table)).one()
        assert row.status == "pending"

        monkeypatch.setattr(blob_service_module, "_atomic_write_blob", original_write)
        recovered = await service.reserve_inline_custody(request)
        assert recovered.id == blob_id
        assert recovered.status == "ready"
        with db_engine.connect() as conn:
            assert conn.execute(select(func.count()).select_from(blobs_table)).scalar_one() == 1

    @pytest.mark.asyncio
    async def test_failed_file_write_leaves_durable_pending_reservation_for_retry(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        request = _custody_request(db_engine, session_id)
        original_write = blob_service_module._atomic_write_blob

        def _interrupt_before_write(_path: Path, _content: bytes) -> None:
            raise RuntimeError("simulated interruption before file write")

        monkeypatch.setattr(blob_service_module, "_atomic_write_blob", _interrupt_before_write)
        with pytest.raises(RuntimeError, match="before file write"):
            await service.reserve_inline_custody(request)

        with db_engine.connect() as conn:
            row = conn.execute(select(blobs_table)).one()
        assert row.status == "pending"
        assert not Path(row.storage_path).exists()

        monkeypatch.setattr(blob_service_module, "_atomic_write_blob", original_write)
        recovered = await service.reserve_inline_custody(request)
        assert recovered.status == "ready"
        assert Path(recovered.storage_path).read_bytes() == request.content
        with db_engine.connect() as conn:
            charged = conn.execute(
                select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == str(session_id))
            ).scalar_one()
        assert charged == len(request.content)

    @pytest.mark.asyncio
    async def test_failed_ready_transition_leaves_pending_row_and_file_for_retry(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        request = _custody_request(db_engine, session_id)
        original_update = blobs_table.update

        def _interrupt_before_ready_update():
            raise RuntimeError("simulated interruption before ready finalization")

        monkeypatch.setattr(blobs_table, "update", _interrupt_before_ready_update)
        with pytest.raises(RuntimeError, match="before ready finalization"):
            await service.reserve_inline_custody(request)

        with db_engine.connect() as conn:
            row = conn.execute(select(blobs_table)).one()
        assert row.status == "pending"
        assert Path(row.storage_path).read_bytes() == request.content

        monkeypatch.setattr(blobs_table, "update", original_update)
        recovered = await service.reserve_inline_custody(request)
        assert recovered.status == "ready"
        with db_engine.connect() as conn:
            assert conn.execute(select(func.count()).select_from(blobs_table)).scalar_one() == 1

    @pytest.mark.asyncio
    async def test_matching_bytes_with_mismatched_reservation_metadata_fail_closed(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
    ) -> None:
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        request = _custody_request(db_engine, session_id)
        _, derive_blob_id = _inline_custody_contract()
        blob_id = derive_blob_id(request)
        storage = tmp_path.resolve() / "blobs" / str(session_id) / f"{blob_id}_candidate.csv"
        storage.parent.mkdir(parents=True)
        storage.write_bytes(request.content)
        with db_engine.begin() as conn:
            conn.execute(
                insert(blobs_table).values(
                    id=str(blob_id),
                    session_id=str(session_id),
                    filename=request.filename,
                    mime_type=request.mime_type,
                    size_bytes=len(request.content),
                    content_hash=content_hash(request.content),
                    storage_path=str(storage),
                    created_at=datetime.now(UTC),
                    created_by="assistant",
                    source_description="different description",
                    status="pending",
                    creation_modality=request.creation_modality.value,
                    created_from_message_id=request.created_from_message_id,
                    creating_model_identifier=None,
                    creating_model_version=None,
                    creating_provider=None,
                    creating_composer_skill_hash=None,
                    creating_arguments_hash=None,
                )
            )

        with pytest.raises(AuditIntegrityError, match="mismatched source_description"):
            await service.reserve_inline_custody(request)

        assert storage.read_bytes() == request.content

    @pytest.mark.asyncio
    async def test_separate_engines_for_same_sqlite_database_share_custody_lock(self, tmp_path: Path) -> None:
        database_path = tmp_path / "custody.sqlite3"
        database_url = f"sqlite:///{database_path}"
        first_engine = create_session_engine(database_url)
        second_engine = create_session_engine(database_url)
        initialize_session_schema(first_engine)
        shared_session_id = uuid4()
        now = datetime.now(UTC)
        with first_engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=str(shared_session_id),
                    user_id="test-user",
                    auth_provider_type="local",
                    title="Shared database session",
                    created_at=now,
                    updated_at=now,
                )
            )
        request = _custody_request(first_engine, shared_session_id)
        first_service = BlobServiceImpl(first_engine, tmp_path / "data", max_storage_per_session=100)
        second_service = BlobServiceImpl(second_engine, tmp_path / "data", max_storage_per_session=100)

        try:
            first, second = await asyncio.gather(
                first_service.reserve_inline_custody(request),
                second_service.reserve_inline_custody(request),
            )
            assert first == second
            with second_engine.connect() as conn:
                assert conn.execute(select(func.count()).select_from(blobs_table)).scalar_one() == 1
        finally:
            first_engine.dispose()
            second_engine.dispose()

    def test_sqlite_separate_processes_converge_on_one_blob_and_quota_charge(self, tmp_path: Path) -> None:
        database_path = tmp_path / "custody.sqlite3"
        database_url = f"sqlite:///{database_path}"
        engine = create_session_engine(database_url)
        shared_session_id = uuid4()
        now = datetime.now(UTC)
        try:
            initialize_session_schema(engine)
            with engine.begin() as conn:
                conn.execute(
                    sessions_table.insert().values(
                        id=str(shared_session_id),
                        user_id="sqlite-custody-test",
                        auth_provider_type="local",
                        title="SQLite custody concurrency",
                        created_at=now,
                        updated_at=now,
                    )
                )
            request = _custody_request(engine, shared_session_id)
            request_fields = {
                "session_id": str(request.session_id),
                "filename": request.filename,
                "content": request.content,
                "mime_type": request.mime_type,
                "source_description": request.source_description,
                "creation_modality": request.creation_modality.value,
                "created_from_message_id": request.created_from_message_id,
                "creating_model_identifier": request.creating_model_identifier,
                "creating_model_version": request.creating_model_version,
                "creating_provider": request.creating_provider,
                "creating_composer_skill_hash": request.creating_composer_skill_hash,
                "creating_arguments_hash": request.creating_arguments_hash,
            }
            context = multiprocessing.get_context("spawn")
            start_event = context.Event()
            result_queue = context.Queue()
            processes = [
                context.Process(
                    target=_custody_process,
                    args=(database_url, str(tmp_path / "data"), request_fields, start_event, result_queue),
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            start_event.set()
            for process in processes:
                process.join(timeout=30)
                assert process.exitcode == 0

            results = [result_queue.get(timeout=5) for _ in processes]
            assert all(result[0] == "ok" for result in results), results
            assert results[0][1] == results[1][1]
            with engine.connect() as conn:
                assert conn.execute(select(func.count()).select_from(blobs_table)).scalar_one() == 1
                charged = conn.execute(
                    select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == str(shared_session_id))
                ).scalar_one()
            assert charged == len(request.content)
        finally:
            engine.dispose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mismatch", [False, True], ids=["matching-winner", "mismatched-winner"])
    async def test_insert_conflict_reloads_and_validates_winner(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mismatch: bool,
    ) -> None:
        request_type, _ = _inline_custody_contract()
        message_id = _seed_custody_message(db_engine, session_id)
        first_request = request_type(
            session_id=session_id,
            filename="candidate.csv",
            content=b"value\n42\n",
            mime_type="text/csv",
            source_description="candidate",
            creation_modality=CreationModality.LLM_GENERATED,
            created_from_message_id=message_id,
            creating_model_identifier="model-a",
            creating_model_version="version-a",
            creating_provider="provider-a",
            creating_composer_skill_hash="a" * 64,
            creating_arguments_hash="b" * 64,
        )
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        winner = await service.reserve_inline_custody(first_request)
        retry = replace(first_request, creating_arguments_hash="c" * 64) if mismatch else first_request
        original_phase_transaction = blob_service_module._blob_phase_transaction
        phase_count = 0

        class _InsertConflictConnection:
            def __init__(self, conn) -> None:
                self._conn = conn
                self.dialect = conn.dialect
                self._missed_blob_lookup = False

            def execute(self, statement, *args, **kwargs):
                if statement.is_select and not self._missed_blob_lookup:
                    selected = tuple(statement.selected_columns)
                    if selected and getattr(selected[0], "table", None) is blobs_table:
                        self._missed_blob_lookup = True
                        return SimpleNamespace(first=lambda: None)
                if statement.is_insert and statement.table is blobs_table:
                    raise IntegrityError("simulated concurrent winner", {}, RuntimeError("duplicate"))
                return self._conn.execute(statement, *args, **kwargs)

            def begin_nested(self):
                return self._conn.begin_nested()

        @contextlib.contextmanager
        def _conflicting_first_phase(engine, held_connection):
            nonlocal phase_count
            phase_count += 1
            with original_phase_transaction(engine, held_connection) as conn:
                yield _InsertConflictConnection(conn) if phase_count == 1 else conn

        monkeypatch.setattr(blob_service_module, "_blob_phase_transaction", _conflicting_first_phase)

        if mismatch:
            with pytest.raises(AuditIntegrityError, match="mismatched creating_arguments_hash"):
                await service.reserve_inline_custody(retry)
        else:
            assert await service.reserve_inline_custody(retry) == winner

    @pytest.mark.skipif(
        not os.environ.get("ELSPETH_TEST_POSTGRES_URL"),
        reason="ELSPETH_TEST_POSTGRES_URL is required for the server-backend custody exercise",
    )
    def test_postgres_separate_processes_converge_on_one_blob_and_quota_charge(self, tmp_path: Path) -> None:
        database_url = os.environ["ELSPETH_TEST_POSTGRES_URL"]
        first_engine = create_session_engine(database_url)
        shared_session_id = uuid4()
        now = datetime.now(UTC)
        try:
            initialize_session_schema(first_engine)
            with first_engine.begin() as conn:
                conn.execute(
                    sessions_table.insert().values(
                        id=str(shared_session_id),
                        user_id="postgres-custody-test",
                        auth_provider_type="local",
                        title="Postgres custody concurrency",
                        created_at=now,
                        updated_at=now,
                    )
                )
            request = _custody_request(first_engine, shared_session_id)
            request_fields = {
                "session_id": str(request.session_id),
                "filename": request.filename,
                "content": request.content,
                "mime_type": request.mime_type,
                "source_description": request.source_description,
                "creation_modality": request.creation_modality.value,
                "created_from_message_id": request.created_from_message_id,
                "creating_model_identifier": request.creating_model_identifier,
                "creating_model_version": request.creating_model_version,
                "creating_provider": request.creating_provider,
                "creating_composer_skill_hash": request.creating_composer_skill_hash,
                "creating_arguments_hash": request.creating_arguments_hash,
            }
            context = multiprocessing.get_context("spawn")
            start_event = context.Event()
            result_queue = context.Queue()
            processes = [
                context.Process(
                    target=_custody_process,
                    args=(database_url, str(tmp_path / "data"), request_fields, start_event, result_queue),
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            start_event.set()
            for process in processes:
                process.join(timeout=30)
                assert process.exitcode == 0

            results = [result_queue.get(timeout=5) for _ in processes]
            assert all(result[0] == "ok" for result in results), results
            assert results[0][1] == results[1][1]
            with first_engine.connect() as conn:
                assert (
                    conn.execute(
                        select(func.count()).select_from(blobs_table).where(blobs_table.c.session_id == str(shared_session_id))
                    ).scalar_one()
                    == 1
                )
                charged = conn.execute(
                    select(func.coalesce(func.sum(blobs_table.c.size_bytes), 0)).where(blobs_table.c.session_id == str(shared_session_id))
                ).scalar_one()
            assert charged == len(request.content)
        finally:
            with first_engine.begin() as conn:
                conn.execute(delete(sessions_table).where(sessions_table.c.id == str(shared_session_id)))
            first_engine.dispose()

    @pytest.mark.asyncio
    async def test_arguments_hash_is_excluded_from_identity_but_mismatched_reuse_fails_closed(
        self,
        db_engine,
        session_id: UUID,
        tmp_path: Path,
    ) -> None:
        request_type, derive_blob_id = _inline_custody_contract()
        message_id = _seed_custody_message(db_engine, session_id)
        first_request = request_type(
            session_id=session_id,
            filename="candidate.csv",
            content=b"value\n42\n",
            mime_type="text/csv",
            source_description="candidate",
            creation_modality=CreationModality.LLM_GENERATED,
            created_from_message_id=message_id,
            creating_model_identifier="model-a",
            creating_model_version="version-a",
            creating_provider="provider-a",
            creating_composer_skill_hash="a" * 64,
            creating_arguments_hash="b" * 64,
        )
        changed_hash = replace(first_request, creating_arguments_hash="c" * 64)
        service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)

        assert derive_blob_id(first_request) == derive_blob_id(changed_hash)
        await service.reserve_inline_custody(first_request)
        with pytest.raises(AuditIntegrityError, match="mismatched creating_arguments_hash"):
            await service.reserve_inline_custody(changed_hash)

        with db_engine.connect() as conn:
            assert conn.execute(select(func.count()).select_from(blobs_table)).scalar_one() == 1


# ---------------------------------------------------------------------------
# copy_blobs_for_fork — deterministic replay and whole-child cleanup
# ---------------------------------------------------------------------------


class TestCopyBlobsForFork:
    """Fork copies converge on deterministic rows and clean up explicitly."""

    @staticmethod
    async def _checkpoint() -> None:
        return None

    @classmethod
    async def _plan(
        cls,
        service: BlobServiceImpl,
        source_session_id: UUID,
        target_session_id: UUID,
    ) -> tuple[BlobForkPlanEntry, ...]:
        ready = [blob for blob in await service.list_blobs(source_session_id, limit=None) if blob.status == "ready"]
        return tuple(
            BlobForkPlanEntry(
                source_blob_id=blob.id,
                target_blob_id=fork_blob_id(target_session_id=target_session_id, source_blob_id=blob.id),
                content_hash=blob.content_hash,
                size_bytes=blob.size_bytes,
            )
            for blob in sorted(ready, key=lambda blob: str(blob.id))
        )

    @classmethod
    async def _copy(cls, service: BlobServiceImpl, source_session_id, target_session_id):
        if type(source_session_id) is UUID and type(target_session_id) is UUID and source_session_id != target_session_id:
            plan = await cls._plan(service, source_session_id, target_session_id)
            write_fence = await cls._authorize_copy(service, source_session_id, target_session_id, plan)
        else:
            plan = ()
            write_fence = object()
        return await service.copy_blobs_for_fork(
            source_session_id,
            target_session_id,
            plan,
            write_fence,  # type: ignore[arg-type]
            checkpoint=cls._checkpoint,
        )

    @staticmethod
    async def _authorize_copy(
        service: BlobServiceImpl,
        source_session_id: UUID,
        target_session_id: UUID,
        plan: tuple[BlobForkPlanEntry, ...],
    ) -> BlobForkWriteFence:
        operation_id = f"test-fork-{target_session_id}"
        lease_token = f"test-lease-{target_session_id}"
        now = datetime.now(UTC)
        with service._engine.connect() as conn:
            if conn.execute(select(sessions_table.c.id).where(sessions_table.c.id == str(target_session_id))).one_or_none() is None:
                return BlobForkWriteFence(
                    source_session_id=source_session_id,
                    target_session_id=target_session_id,
                    operation_id=operation_id,
                    lease_token=lease_token,
                    attempt=1,
                )
            operation = conn.execute(
                select(guided_operations_table.c.operation_id).where(
                    guided_operations_table.c.session_id == str(source_session_id),
                    guided_operations_table.c.operation_id == operation_id,
                )
            ).one_or_none()
        if operation is None:
            session_service = SessionServiceImpl(
                service._engine,
                telemetry=build_sessions_telemetry(),
                log=structlog.get_logger("test.blob-fork-custody"),
            )
            await session_service.add_message(
                target_session_id,
                "audit",
                json.dumps(
                    {
                        "schema": "session-fork-blob-plan.v1",
                        "source_session_id": str(source_session_id),
                        "child_session_id": str(target_session_id),
                        "operation_id": operation_id,
                        "source_blobs": [
                            {
                                "source_blob_id": str(entry.source_blob_id),
                                "target_blob_id": str(entry.target_blob_id),
                                "content_hash": entry.content_hash,
                                "size_bytes": entry.size_bytes,
                            }
                            for entry in plan
                        ],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                writer_principal="session_fork",
            )
        with service._engine.begin() as conn:
            conn.execute(sessions_table.update().where(sessions_table.c.id == str(target_session_id)).values(archived_at=now))
            if operation is None:
                conn.execute(
                    guided_operations_table.insert().values(
                        session_id=str(source_session_id),
                        operation_id=operation_id,
                        kind="session_fork",
                        status="in_progress",
                        request_hash="a" * 64,
                        lease_token=lease_token,
                        lease_expires_at=now + timedelta(hours=1),
                        attempt=1,
                        result_session_id=str(target_session_id),
                        created_at=now,
                        updated_at=now,
                    )
                )
        return BlobForkWriteFence(
            source_session_id=source_session_id,
            target_session_id=target_session_id,
            operation_id=operation_id,
            lease_token=lease_token,
            attempt=1,
        )

    @staticmethod
    def _fail_fork(service: BlobServiceImpl, source_session_id: UUID, target_session_id: UUID) -> str:
        operation_id = f"test-fork-{target_session_id}"
        now = datetime.now(UTC)
        with service._engine.begin() as conn:
            changed = conn.execute(
                guided_operations_table.update()
                .where(
                    guided_operations_table.c.session_id == str(source_session_id),
                    guided_operations_table.c.operation_id == operation_id,
                    guided_operations_table.c.status == "in_progress",
                )
                .values(
                    status="failed",
                    lease_token=None,
                    lease_expires_at=None,
                    result_session_id=None,
                    failure_code="operation_failed",
                    settled_at=now,
                    updated_at=now,
                )
            ).rowcount
        assert changed == 1
        return operation_id

    @pytest.fixture()
    def target_session_id(self, db_engine, session_id: UUID) -> UUID:
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
                    archived_at=now,
                    forked_from_session_id=str(session_id),
                )
            )
        return UUID(sid)

    @staticmethod
    def _insert_session(
        db_engine,
        *,
        user_id: str = "test-user",
        auth_provider_type: str = "local",
        forked_from_session_id: UUID | None = None,
    ) -> UUID:
        session_id = uuid4()
        now = datetime.now(UTC)
        with db_engine.begin() as conn:
            conn.execute(
                sessions_table.insert().values(
                    id=str(session_id),
                    user_id=user_id,
                    auth_provider_type=auth_provider_type,
                    title="Test Session",
                    created_at=now,
                    updated_at=now,
                    archived_at=now if forked_from_session_id is not None else None,
                    forked_from_session_id=(str(forked_from_session_id) if forked_from_session_id is not None else None),
                )
            )
        return session_id

    def test_fork_blob_id_has_frozen_uuid5_contract_vector(self) -> None:
        target = UUID("11111111-1111-4111-8111-111111111111")
        source = UUID("22222222-2222-4222-8222-222222222222")

        expected = fork_blob_id(
            target_session_id=target,
            source_blob_id=source,
        )

        assert expected == UUID("7db1b79c-2cad-5fc0-87c8-45652bd6cfd4")
        assert (
            fork_blob_id(
                target_session_id=UUID("33333333-3333-4333-8333-333333333333"),
                source_blob_id=source,
            )
            != expected
        )
        assert (
            fork_blob_id(
                target_session_id=target,
                source_blob_id=UUID("44444444-4444-4444-8444-444444444444"),
            )
            != expected
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("invalid_argument", ["source", "target"])
    async def test_copy_requires_exact_uuid_session_ids(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
        invalid_argument: str,
    ) -> None:
        source: UUID | str = session_id
        target: UUID | str = target_session_id
        if invalid_argument == "source":
            source = str(source)
        else:
            target = str(target)

        with pytest.raises(TypeError, match=f"{invalid_argument}_session_id must be UUID"):
            await self._copy(blob_service, source, target)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_copy_rejects_source_as_its_own_target(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
    ) -> None:
        with pytest.raises(ValueError, match="source and target sessions must differ"):
            await self._copy(blob_service, session_id, session_id)

    @pytest.mark.asyncio
    async def test_copy_rejects_unrelated_target_before_blob_work(
        self,
        blob_service: BlobServiceImpl,
        db_engine,
        session_id: UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        unrelated_target = self._insert_session(db_engine)
        source = await blob_service.create_blob(session_id, "source.csv", b"source", "text/csv")
        plan = (
            BlobForkPlanEntry(
                source_blob_id=source.id,
                target_blob_id=fork_blob_id(target_session_id=unrelated_target, source_blob_id=source.id),
                content_hash=source.content_hash,
                size_bytes=source.size_bytes,
            ),
        )

        async def _unexpected_blob_list(*_args, **_kwargs):
            pytest.fail("fork custody must be verified before listing blobs")

        monkeypatch.setattr(blob_service, "list_blobs", _unexpected_blob_list)

        with pytest.raises(AuditIntegrityError, match="not a fork child"):
            await blob_service.copy_blobs_for_fork(
                session_id,
                unrelated_target,
                plan,
                await self._authorize_copy(blob_service, session_id, unrelated_target, plan),
                checkpoint=self._checkpoint,
            )

        with db_engine.connect() as conn:
            assert (
                conn.execute(
                    select(func.count()).select_from(blobs_table).where(blobs_table.c.session_id == str(unrelated_target))
                ).scalar_one()
                == 0
            )

    @pytest.mark.asyncio
    async def test_copy_rejects_missing_target_before_blob_work(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
    ) -> None:
        await blob_service.create_blob(session_id, "source.csv", b"source", "text/csv")

        with pytest.raises(AuditIntegrityError, match=r"target session .* does not exist"):
            await self._copy(blob_service, session_id, uuid4())

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("user_id", "auth_provider_type"),
        [("different-user", "local"), ("test-user", "oidc")],
    )
    async def test_copy_rejects_cross_principal_fork_child(
        self,
        blob_service: BlobServiceImpl,
        db_engine,
        session_id: UUID,
        user_id: str,
        auth_provider_type: str,
    ) -> None:
        target = self._insert_session(
            db_engine,
            user_id=user_id,
            auth_provider_type=auth_provider_type,
            forked_from_session_id=session_id,
        )
        await blob_service.create_blob(session_id, "source.csv", b"source", "text/csv")

        with pytest.raises(AuditIntegrityError, match="principal does not match"):
            await self._copy(blob_service, session_id, target)

        assert await blob_service.list_blobs(target, limit=None) == []

    @pytest.mark.asyncio
    async def test_repeat_copy_returns_same_deterministic_ids(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
    ) -> None:
        first = await blob_service.create_blob(
            session_id=session_id,
            filename="first.csv",
            content=b"first",
            mime_type="text/csv",
            created_by="user",
        )
        second = await blob_service.create_blob(
            session_id=session_id,
            filename="second.csv",
            content=b"second",
            mime_type="text/csv",
            created_by="user",
        )

        first_result = await self._copy(blob_service, session_id, target_session_id)
        second_result = await self._copy(blob_service, session_id, target_session_id)

        assert set(first_result) == {first.id, second.id}
        assert {source_id: record.id for source_id, record in first_result.items()} == {
            source_id: record.id for source_id, record in second_result.items()
        }
        assert all(record.session_id == target_session_id for record in second_result.values())
        assert len(await blob_service.list_blobs(target_session_id, limit=None)) == 2

    @pytest.mark.asyncio
    async def test_copy_runs_checkpoint_before_plan_or_quota_reads(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        source = await blob_service.create_blob(session_id, "source.csv", b"source", "text/csv")
        plan = await self._plan(blob_service, session_id, target_session_id)

        def _unexpected_custody(*_args, **_kwargs):
            pytest.fail("expired fence must stop before plan/quota reads")

        async def _lost_fence() -> None:
            raise RuntimeError("fence lost")

        monkeypatch.setattr(blob_service_module, "_verify_fork_child_custody", _unexpected_custody)
        with pytest.raises(RuntimeError, match="fence lost"):
            await blob_service.copy_blobs_for_fork(
                session_id,
                target_session_id,
                plan,
                await self._authorize_copy(blob_service, session_id, target_session_id, plan),
                checkpoint=_lost_fence,
            )
        assert source.id not in {blob.id for blob in await blob_service.list_blobs(target_session_id, limit=None)}

    @pytest.mark.asyncio
    async def test_copy_uses_frozen_plan_and_ignores_newly_finalized_parent_blob(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
    ) -> None:
        frozen = await blob_service.create_blob(session_id, "frozen.csv", b"frozen", "text/csv")
        plan = await self._plan(blob_service, session_id, target_session_id)
        late = await blob_service.create_blob(session_id, "late.csv", b"late", "text/csv")

        copied = await blob_service.copy_blobs_for_fork(
            session_id,
            target_session_id,
            plan,
            await self._authorize_copy(blob_service, session_id, target_session_id, plan),
            checkpoint=self._checkpoint,
        )

        assert set(copied) == {frozen.id}
        assert late.id not in copied

    @pytest.mark.asyncio
    @pytest.mark.parametrize("fork_status", ["in_progress", "completed", "failed"])
    async def test_delete_enforces_session_fork_retention_lifecycle(
        self,
        blob_service: BlobServiceImpl,
        db_engine,
        session_id: UUID,
        target_session_id: UUID,
        fork_status: str,
    ) -> None:
        source = await blob_service.create_blob(session_id, "source.csv", b"source", "text/csv")
        now = datetime.now(UTC)
        operation_id = str(uuid4())
        values: dict[str, object] = {
            "session_id": str(session_id),
            "operation_id": operation_id,
            "kind": "session_fork",
            "status": fork_status,
            "request_hash": "a" * 64,
            "attempt": 1,
            "created_at": now,
            "updated_at": now,
        }
        if fork_status == "in_progress":
            values.update(lease_token="lease", lease_expires_at=now + timedelta(hours=1))
        elif fork_status == "completed":
            values.update(
                settled_at=now,
                result_kind="session",
                result_session_id=str(target_session_id),
                response_hash="b" * 64,
            )
        else:
            values.update(settled_at=now, failure_code="operation_failed")
        with db_engine.begin() as conn:
            conn.execute(guided_operations_table.insert().values(**values))

        if fork_status == "in_progress":
            with pytest.raises(BlobInProgressForkError, match=operation_id):
                await blob_service.delete_blob(source.id)

            assert await blob_service.read_blob_content(source.id) == b"source"
        else:
            await blob_service.delete_blob(source.id)
            with pytest.raises(BlobNotFoundError):
                await blob_service.read_blob_content(source.id)

    @pytest.mark.asyncio
    async def test_partial_prior_success_resumes_without_duplicate(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
    ) -> None:
        first = await blob_service.create_blob(session_id, "first.csv", b"first", "text/csv")
        first_pass = await self._copy(blob_service, session_id, target_session_id)
        second = await blob_service.create_blob(session_id, "second.csv", b"second", "text/csv")

        resumed = await self._copy(blob_service, session_id, target_session_id)

        assert resumed[first.id].id == first_pass[first.id].id
        assert set(resumed) == {first.id, second.id}
        assert len(await blob_service.list_blobs(target_session_id, limit=None)) == 2

    @pytest.mark.asyncio
    async def test_quota_preflight_charges_only_missing_expected_copies(
        self,
        blob_service: BlobServiceImpl,
        db_engine,
        session_id: UUID,
        target_session_id: UUID,
        tmp_path: Path,
    ) -> None:
        first = await blob_service.create_blob(session_id, "first.csv", b"first", "text/csv")
        quota_service = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=11)
        first_pass = await self._copy(quota_service, session_id, target_session_id)
        second = await blob_service.create_blob(session_id, "second.csv", b"second", "text/csv")

        resumed = await self._copy(quota_service, session_id, target_session_id)

        assert resumed[first.id].id == first_pass[first.id].id
        assert second.id in resumed
        assert sum(blob.size_bytes for blob in await quota_service.list_blobs(target_session_id, limit=None)) == 11

    @pytest.mark.asyncio
    async def test_zero_write_replay_succeeds_after_quota_is_lowered_below_current_usage(
        self,
        db_engine,
        session_id: UUID,
        target_session_id: UUID,
        tmp_path: Path,
    ) -> None:
        high_quota = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=100)
        source = await high_quota.create_blob(session_id, "source.csv", b"source", "text/csv")
        first = await self._copy(high_quota, session_id, target_session_id)
        low_quota = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=1)

        replay = await self._copy(low_quota, session_id, target_session_id)

        assert replay[source.id].id == first[source.id].id
        assert [blob.id for blob in await low_quota.list_blobs(target_session_id, limit=None)] == [first[source.id].id]

    @pytest.mark.asyncio
    async def test_copy_supports_more_than_fifty_ready_blobs(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
    ) -> None:
        for index in range(51):
            await blob_service.create_blob(session_id, f"item-{index}.csv", str(index).encode(), "text/csv")

        result = await self._copy(blob_service, session_id, target_session_id)

        assert len(result) == 51
        assert len(await blob_service.list_blobs(target_session_id, limit=None)) == 51

    @pytest.mark.asyncio
    async def test_copy_ignores_non_ready_source_blobs(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
    ) -> None:
        ready = await blob_service.create_blob(session_id, "ready.csv", b"ready", "text/csv")
        await blob_service.create_pending_blob(session_id, "pending.csv", "text/csv")

        result = await self._copy(blob_service, session_id, target_session_id)

        assert set(result) == {ready.id}

    @pytest.mark.asyncio
    async def test_copy_preserves_source_content_integrity_gate(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
    ) -> None:
        source = await blob_service.create_blob(session_id, "ready.csv", b"ready", "text/csv")
        Path(source.storage_path).write_bytes(b"tampered")

        with pytest.raises(BlobIntegrityError):
            await self._copy(blob_service, session_id, target_session_id)

        assert await blob_service.list_blobs(target_session_id, limit=None) == []

    @pytest.mark.asyncio
    async def test_empty_source_returns_empty_map(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
    ) -> None:
        """No blobs in source session → empty mapping, no errors."""
        result = await self._copy(blob_service, session_id, target_session_id)
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
        await blob_service.create_blob(
            session_id=session_id,
            filename="big.csv",
            content=b"x" * 100,
            mime_type="text/csv",
            created_by="user",
        )

        small_quota = BlobServiceImpl(db_engine, tmp_path, max_storage_per_session=10)

        with pytest.raises(BlobQuotaExceededError):
            await self._copy(small_quota, session_id, target_session_id)

        target_blobs = await blob_service.list_blobs(target_session_id)
        assert target_blobs == []

    @pytest.mark.asyncio
    async def test_cleanup_is_idempotent_and_typed(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
    ) -> None:
        source = await blob_service.create_blob(session_id, "first.csv", b"first", "text/csv")
        copied = await self._copy(blob_service, session_id, target_session_id)
        operation_id = self._fail_fork(blob_service, session_id, target_session_id)

        first = await blob_service.cleanup_blobs_for_fork(session_id, target_session_id, operation_id)
        second = await blob_service.cleanup_blobs_for_fork(session_id, target_session_id, operation_id)

        assert type(first) is BlobForkCleanupResult
        assert type(first.deleted_ids) is tuple
        assert type(first.errors) is tuple
        assert tuple(first.deleted_ids) == (copied[source.id].id,)
        assert tuple(first.errors) == ()
        assert tuple(second.deleted_ids) == ()
        assert tuple(second.errors) == ()
        assert await blob_service.list_blobs(target_session_id, limit=None) == []

    @pytest.mark.asyncio
    async def test_cleanup_rejects_wrong_parent_and_preserves_child_blobs(
        self,
        blob_service: BlobServiceImpl,
        db_engine,
        session_id: UUID,
        target_session_id: UUID,
    ) -> None:
        wrong_parent = self._insert_session(db_engine)
        source = await blob_service.create_blob(session_id, "source.csv", b"source", "text/csv")
        copied = await self._copy(blob_service, session_id, target_session_id)
        operation_id = self._fail_fork(blob_service, session_id, target_session_id)

        with pytest.raises(AuditIntegrityError, match="not a fork child"):
            await blob_service.cleanup_blobs_for_fork(wrong_parent, target_session_id, operation_id)

        assert [blob.id for blob in await blob_service.list_blobs(target_session_id, limit=None)] == [copied[source.id].id]

    @pytest.mark.asyncio
    async def test_cleanup_rejects_active_completed_child_and_preserves_blobs(
        self,
        blob_service: BlobServiceImpl,
        db_engine,
        session_id: UUID,
        target_session_id: UUID,
    ) -> None:
        await blob_service.create_blob(session_id, "source.csv", b"source", "text/csv")
        await self._copy(blob_service, session_id, target_session_id)
        operation_id = self._fail_fork(blob_service, session_id, target_session_id)
        before = await blob_service.list_blobs(target_session_id, limit=None)
        with db_engine.begin() as conn:
            conn.execute(sessions_table.update().where(sessions_table.c.id == str(target_session_id)).values(archived_at=None))

        with pytest.raises(AuditIntegrityError, match="not an archived staged fork child"):
            await blob_service.cleanup_blobs_for_fork(session_id, target_session_id, operation_id)

        assert await blob_service.list_blobs(target_session_id, limit=None) == before

    @pytest.mark.asyncio
    async def test_cleanup_treats_already_missing_snapshot_row_as_success(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await blob_service.create_blob(session_id, "first.csv", b"first", "text/csv")
        await self._copy(blob_service, session_id, target_session_id)
        stale_snapshot = await blob_service.list_blobs(target_session_id, limit=None)
        await blob_service.delete_blob(stale_snapshot[0].id)
        operation_id = self._fail_fork(blob_service, session_id, target_session_id)
        result = await blob_service.cleanup_blobs_for_fork(session_id, target_session_id, operation_id)

        assert tuple(result.deleted_ids) == ()
        assert tuple(result.errors) == ()

    @pytest.mark.asyncio
    async def test_cleanup_continues_after_item_failure_and_records_residual_metric(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await blob_service.create_blob(session_id, "first.csv", b"first", "text/csv")
        await blob_service.create_blob(session_id, "second.csv", b"second", "text/csv")
        await self._copy(blob_service, session_id, target_session_id)
        operation_id = self._fail_fork(blob_service, session_id, target_session_id)
        target = await blob_service.list_blobs(target_session_id, limit=None)
        failing_id = target[0].id

        orphan_counter = _FakeCounter()
        monkeypatch.setattr(blob_service_module, "_BLOB_COPY_FORK_ORPHAN_ROWS_COUNTER", orphan_counter)
        original_delete = blob_service._delete_blob_row_locked

        def _fail_one(conn, *, row, blob_id_str: str):
            if blob_id_str == str(failing_id):
                raise OSError(5, "cleanup failed")
            return original_delete(conn, row=row, blob_id_str=blob_id_str)

        monkeypatch.setattr(blob_service, "_delete_blob_row_locked", _fail_one)
        result = await blob_service.cleanup_blobs_for_fork(session_id, target_session_id, operation_id)

        assert type(result) is BlobForkCleanupResult
        assert set(result.deleted_ids) == {record.id for record in target if record.id != failing_id}
        assert len(result.errors) == 1
        error = result.errors[0]
        assert type(error) is BlobForkCleanupError
        assert error.blob_id == failing_id
        assert error.exc_type == "OSError"
        assert "cleanup failed" in error.detail
        assert [blob.id for blob in await blob_service.list_blobs(target_session_id, limit=None)] == [failing_id]
        assert len(orphan_counter.calls) == 1
        amount, attrs, context = orphan_counter.calls[0]
        assert amount == 1
        assert attrs == {
            "orphan_blob_id": str(failing_id),
            "target_session_id": str(target_session_id),
            "exc_type": "OSError",
        }
        assert context is None

    @pytest.mark.asyncio
    async def test_mid_copy_failure_retains_partial_rows_for_exact_plan_takeover(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await blob_service.create_blob(session_id, "first.csv", b"first", "text/csv")
        await blob_service.create_blob(session_id, "second.csv", b"second", "text/csv")
        original_persist = blob_service_module._persist_blob_content
        target_copy_calls = 0

        def _fail_second_target_copy(**kwargs):
            nonlocal target_copy_calls
            if str(kwargs["session_id"]) == str(target_session_id) and kwargs["idempotent"] is True:
                target_copy_calls += 1
                if target_copy_calls == 2:
                    raise RuntimeError("mid-copy failure")
            return original_persist(**kwargs)

        monkeypatch.setattr(blob_service_module, "_persist_blob_content", _fail_second_target_copy)

        with pytest.raises(RuntimeError, match="mid-copy failure"):
            await self._copy(blob_service, session_id, target_session_id)

        assert len(await blob_service.list_blobs(target_session_id, limit=None)) == 1
        target_dir = tmp_path.resolve() / "blobs" / str(target_session_id)
        assert len(list(target_dir.iterdir())) == 1

    @pytest.mark.asyncio
    async def test_copy_failure_does_not_invoke_automatic_cleanup(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await blob_service.create_blob(session_id, "first.csv", b"first", "text/csv")
        await blob_service.create_blob(session_id, "second.csv", b"second", "text/csv")
        original_persist = blob_service_module._persist_blob_content
        target_copy_calls = 0

        def _fail_second_target_copy(**kwargs):
            nonlocal target_copy_calls
            if str(kwargs["session_id"]) == str(target_session_id) and kwargs["idempotent"] is True:
                target_copy_calls += 1
                if target_copy_calls == 2:
                    raise RuntimeError("primary copy failure")
            return original_persist(**kwargs)

        async def _unexpected_cleanup(*_args, **_kwargs) -> None:
            pytest.fail("copy service must leave cleanup ownership to the fail-CAS winner")

        monkeypatch.setattr(blob_service_module, "_persist_blob_content", _fail_second_target_copy)
        monkeypatch.setattr(blob_service, "cleanup_blobs_for_fork", _unexpected_cleanup)

        with pytest.raises(RuntimeError, match="primary copy failure") as exc_info:
            await self._copy(blob_service, session_id, target_session_id)

        assert type(exc_info.value) is RuntimeError
        assert getattr(exc_info.value, "__notes__", []) == []
        assert len(await blob_service.list_blobs(target_session_id, limit=None)) == 1

    @pytest.mark.asyncio
    async def test_fail_cleanup_wins_pause_before_persist_and_stale_writer_leaves_no_blob(
        self,
        blob_service: BlobServiceImpl,
        session_id: UUID,
        target_session_id: UUID,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await blob_service.create_blob(session_id, "source.csv", b"source", "text/csv")
        reached_persist = threading.Event()
        release_persist = threading.Event()
        original_persist = blob_service_module._persist_blob_content

        def _pause_before_persist(**kwargs):
            reached_persist.set()
            if not release_persist.wait(timeout=5):
                raise TimeoutError("test did not release paused fork writer")
            return original_persist(**kwargs)

        monkeypatch.setattr(blob_service_module, "_persist_blob_content", _pause_before_persist)
        copy_task = asyncio.create_task(self._copy(blob_service, session_id, target_session_id))
        assert await asyncio.to_thread(reached_persist.wait, 5)
        operation_id = self._fail_fork(blob_service, session_id, target_session_id)
        cleanup = await blob_service.cleanup_blobs_for_fork(session_id, target_session_id, operation_id)
        assert cleanup.errors == ()
        release_persist.set()

        with pytest.raises(BlobForkFenceLostError):
            await copy_task
        assert await blob_service.list_blobs(target_session_id, limit=None) == []


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
                    # Plan §2294: every test-side direct composition_states
                    # insert must supply provenance after Task 3's CHECK
                    # constraint. ``session_seed`` is the broadened-semantics
                    # default for setup-only rows that don't model a real
                    # compose-loop transition.
                    provenance="session_seed",
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
                    # Plan §2294: every test-side direct composition_states
                    # insert must supply provenance after Task 3's CHECK
                    # constraint. ``session_seed`` is the broadened-semantics
                    # default for setup-only rows that don't model a real
                    # compose-loop transition.
                    provenance="session_seed",
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

    @staticmethod
    def _deny_read_bytes(monkeypatch: pytest.MonkeyPatch, denied_path: Path) -> None:
        original_read_bytes = Path.read_bytes

        def _read_bytes_or_permission_error(path: Path) -> bytes:
            if path == denied_path:
                raise PermissionError(f"Permission denied: '{denied_path}'")
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", _read_bytes_or_permission_error)

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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When file read raises OSError, loop continues to next blob."""
        run_id, _ = run_env

        await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b1.csv", b"data1")
        b2 = await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b2.csv", b"data2")

        self._deny_read_bytes(monkeypatch, Path(b2.storage_path))
        result = await blob_service.finalize_run_output_blobs(run_id, success=True)

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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When per-blob catch fires, the failed blob is set to 'error' status."""
        from elspeth.web.sessions.models import blobs_table as bt

        run_id, _ = run_env

        b1 = await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b1.csv", b"data1")
        b2 = await self._create_linked_blob(blob_service, session_id, run_id, db_engine, "b2.csv", b"data2")

        self._deny_read_bytes(monkeypatch, Path(b1.storage_path))
        result = await blob_service.finalize_run_output_blobs(run_id, success=True)

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
                    # Plan §2294: every test-side direct composition_states
                    # insert must supply provenance after Task 3's CHECK
                    # constraint. ``session_seed`` is the broadened-semantics
                    # default for setup-only rows that don't model a real
                    # compose-loop transition.
                    provenance="session_seed",
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
                    # Plan §2294: setup-only row; provenance required.
                    provenance="session_seed",
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

    @pytest.mark.asyncio
    async def test_duplicate_same_direction_link_is_idempotent(self, blob_service, session_id, db_engine) -> None:
        """A source bind and inline-content ref can share the same input blob."""
        run_id = self._make_run(db_engine, session_id)
        blob = await blob_service.create_blob(
            session_id=session_id,
            filename="prompt.csv",
            content=b"prompt",
            mime_type="text/csv",
            created_by="user",
        )

        await blob_service.link_blob_to_run(blob.id, run_id, "input")
        await blob_service.link_blob_to_run(blob.id, run_id, "input")

        links = await blob_service.get_blob_run_links(blob.id)
        assert [(link.run_id, link.direction) for link in links] == [(run_id, "input")]


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
            # Inline-blob provenance defaults (Phase 5a Task 2.5): the
            # synthetic row mirrors a verbatim row produced by the
            # user-upload write path (creation_modality='verbatim',
            # everything else NULL).
            "creation_modality": "verbatim",
            "created_from_message_id": None,
            "creating_model_identifier": None,
            "creating_model_version": None,
            "creating_provider": None,
            "creating_composer_skill_hash": None,
            "creating_arguments_hash": None,
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
