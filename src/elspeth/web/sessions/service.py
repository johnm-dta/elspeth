"""SessionService implementation -- CRUD, state versioning, active run enforcement.

Uses SQLAlchemy Core with a synchronous engine. Database calls run in a
thread pool executor to avoid blocking the async event loop.
"""

from __future__ import annotations

import contextlib
import shutil
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID

from sqlalchemy import ColumnElement, Connection, Engine, delete, desc, func, insert, select, update
from sqlalchemy.exc import IntegrityError

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.freeze import deep_thaw
from elspeth.web.async_workers import run_sync_in_worker
from elspeth.web.sessions.models import (
    chat_messages_table,
    composition_states_table,
    run_events_table,
    runs_table,
    sessions_table,
)
from elspeth.web.sessions.protocol import (
    LEGAL_RUN_TRANSITIONS,
    OPERATOR_COMPLETION_RUN_STATUS_VALUES,
    SESSION_TERMINAL_RUN_STATUS_VALUES,
    ChatMessageRecord,
    CompositionStateData,
    CompositionStateRecord,
    IllegalRunTransitionError,
    RunAlreadyActiveError,
    RunRecord,
    SessionRecord,
    SessionRunStatus,
)


def _assert_state_in_session(
    conn: Connection,
    *,
    state_id: str,
    expected_session_id: str,
    caller: str,
) -> None:
    """Offensive guard: composition state must belong to the expected session.

    Catches cross-session reference bugs at the service boundary, before
    they hit the DB-level composite FK. Produces a diagnostic naming the
    caller, the state, and the session mismatch — something a generic
    ``IntegrityError`` cannot.

    Raises ``RuntimeError`` because a cross-session reference is a bug
    in caller code, not invalid user input. The audit trail records the
    attempted violation through the standard exception path.

    Contrast with ``set_active_state``, which raises ``ValueError`` for
    an equivalent-looking cross-session check on purpose: that method
    receives the state_id from the HTTP body and must map an unknown /
    non-owned state to 404 rather than 500. The exception type is
    load-bearing and encodes whether the caller (RuntimeError) or the
    user (ValueError) is wrong.
    """
    state_session_id = conn.execute(select(composition_states_table.c.session_id).where(composition_states_table.c.id == state_id)).scalar()
    if state_session_id is None:
        raise RuntimeError(f"{caller}: composition_state_id={state_id!r} does not exist (expected in session={expected_session_id!r})")
    if state_session_id != expected_session_id:
        raise RuntimeError(
            f"{caller}: composition_state_id={state_id!r} belongs to session "
            f"{state_session_id!r}, not {expected_session_id!r} — cross-session "
            f"reference is a contract violation"
        )


class SessionServiceImpl:
    """Concrete session service backed by SQLAlchemy Core.

    All public methods are async. Database I/O runs through _run_sync() in a
    bounded worker thread so the async event loop is never blocked.
    """

    def __init__(self, engine: Engine, data_dir: Path | None = None) -> None:
        self._engine = engine
        self._data_dir = data_dir

    async def _run_sync(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Run a synchronous callable in the thread pool executor."""
        return await run_sync_in_worker(func, *args, **kwargs)

    def _now(self) -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        """Restore UTC tzinfo stripped by SQLite round-trip.

        SQLite stores DateTime(timezone=True) as ISO-8601 text and drops
        tzinfo on read.  All timestamps in this service originate from
        _now() which uses UTC, so re-attaching UTC is safe.
        """
        if dt.tzinfo is not None:
            return dt
        return dt.replace(tzinfo=UTC)

    async def create_session(
        self,
        user_id: str,
        title: str,
        auth_provider_type: str,
        forked_from_session_id: UUID | None = None,
        forked_from_message_id: UUID | None = None,
    ) -> SessionRecord:
        """Create a new session and return its record."""
        session_id = uuid.uuid4()
        now = self._now()

        def _sync() -> None:
            with self._engine.begin() as conn:
                conn.execute(
                    insert(sessions_table).values(
                        id=str(session_id),
                        user_id=user_id,
                        auth_provider_type=auth_provider_type,
                        title=title,
                        created_at=now,
                        updated_at=now,
                        forked_from_session_id=str(forked_from_session_id) if forked_from_session_id else None,
                        forked_from_message_id=str(forked_from_message_id) if forked_from_message_id else None,
                    )
                )

        await self._run_sync(_sync)

        return SessionRecord(
            id=session_id,
            user_id=user_id,
            auth_provider_type=auth_provider_type,
            title=title,
            created_at=now,
            updated_at=now,
            forked_from_session_id=forked_from_session_id,
            forked_from_message_id=forked_from_message_id,
        )

    async def get_session(self, session_id: UUID) -> SessionRecord:
        """Fetch a session by ID. Raises ValueError if not found."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(select(sessions_table).where(sessions_table.c.id == str(session_id))).fetchone()

        row = await self._run_sync(_sync)

        if row is None:
            raise ValueError(f"Session not found: {session_id}")

        return SessionRecord(
            id=UUID(row.id),
            user_id=row.user_id,
            auth_provider_type=row.auth_provider_type,
            title=row.title,
            created_at=self._ensure_utc(row.created_at),
            updated_at=self._ensure_utc(row.updated_at),
            forked_from_session_id=UUID(row.forked_from_session_id) if row.forked_from_session_id else None,
            forked_from_message_id=UUID(row.forked_from_message_id) if row.forked_from_message_id else None,
        )

    async def list_sessions(
        self,
        user_id: str,
        auth_provider_type: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[SessionRecord]:
        """List sessions for a user, ordered by updated_at descending."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(
                    select(sessions_table)
                    .where(
                        sessions_table.c.user_id == user_id,
                        sessions_table.c.auth_provider_type == auth_provider_type,
                    )
                    .order_by(desc(sessions_table.c.updated_at))
                    .limit(limit)
                    .offset(offset)
                ).fetchall()

        rows = await self._run_sync(_sync)

        return [
            SessionRecord(
                id=UUID(row.id),
                user_id=row.user_id,
                auth_provider_type=row.auth_provider_type,
                title=row.title,
                created_at=self._ensure_utc(row.created_at),
                updated_at=self._ensure_utc(row.updated_at),
                forked_from_session_id=UUID(row.forked_from_session_id) if row.forked_from_session_id else None,
                forked_from_message_id=UUID(row.forked_from_message_id) if row.forked_from_message_id else None,
            )
            for row in rows
        ]

    async def archive_session(self, session_id: UUID) -> None:
        """Delete a session and cascade to all related records and files.

        Filesystem cleanup is staged before the DB delete so the service can
        restore the original blob directory if the transaction fails. After the
        DB commit, any purge failure leaves the staged directory in a recoverable
        quarantine path instead of raising after the session is already gone.
        """
        sid = str(session_id)

        def _sync() -> None:
            blob_dir: Path | None = None
            staged_blob_dir: Path | None = None
            if self._data_dir is not None:
                blob_dir = self._data_dir / "blobs" / sid
                if blob_dir.is_dir():
                    staged_blob_dir = self._data_dir / ".archive_quarantine" / sid
                    staged_blob_dir.parent.mkdir(parents=True, exist_ok=True)
                    if staged_blob_dir.exists():
                        raise OSError(
                            f"archive_session({sid}): quarantine path {staged_blob_dir} already exists. "
                            "Manual cleanup of the stale staged blob directory is required before archive can proceed."
                        )
                    blob_dir.rename(staged_blob_dir)

            try:
                with self._engine.begin() as conn:
                    # Delete in dependency order (children first for non-CASCADE DBs)
                    # Get run IDs for this session to delete run_events
                    run_ids = [r.id for r in conn.execute(select(runs_table.c.id).where(runs_table.c.session_id == sid)).fetchall()]
                    if run_ids:
                        conn.execute(delete(run_events_table).where(run_events_table.c.run_id.in_(run_ids)))
                    conn.execute(delete(runs_table).where(runs_table.c.session_id == sid))
                    conn.execute(delete(chat_messages_table).where(chat_messages_table.c.session_id == sid))
                    conn.execute(delete(composition_states_table).where(composition_states_table.c.session_id == sid))
                    conn.execute(delete(sessions_table).where(sessions_table.c.id == sid))
            except Exception as primary_exc:
                if blob_dir is not None and staged_blob_dir is not None and staged_blob_dir.exists():
                    try:
                        blob_dir.parent.mkdir(parents=True, exist_ok=True)
                        staged_blob_dir.rename(blob_dir)
                    except OSError as restore_exc:
                        primary_exc.add_note(
                            f"RecoveryFailed[{type(restore_exc).__name__}]: "
                            f"could not restore staged blob directory for session {sid} "
                            f"from {staged_blob_dir} to {blob_dir} after archive rollback "
                            f"({restore_exc}). Manual cleanup required."
                        )
                raise

            # After commit, the delete succeeded from the caller's perspective.
            # If the final purge fails, keep the staged directory in quarantine
            # for manual cleanup rather than raising a false "delete failed"
            # error after the session rows are already gone.
            if staged_blob_dir is not None and staged_blob_dir.exists():
                try:
                    shutil.rmtree(staged_blob_dir)
                except OSError:
                    return

                quarantine_root = staged_blob_dir.parent
                with contextlib.suppress(OSError):
                    quarantine_root.rmdir()

        await self._run_sync(_sync)

    async def add_message(
        self,
        session_id: UUID,
        role: Literal["user", "assistant", "system", "tool"],
        content: str,
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
        composition_state_id: UUID | None = None,
        raw_content: str | None = None,
    ) -> ChatMessageRecord:
        """Add a chat message and update the session's updated_at."""
        msg_id = uuid.uuid4()
        now = self._now()
        sid = str(session_id)
        csid = str(composition_state_id) if composition_state_id else None

        def _sync() -> None:
            with self._engine.begin() as conn:
                if csid is not None:
                    _assert_state_in_session(
                        conn,
                        state_id=csid,
                        expected_session_id=sid,
                        caller="add_message",
                    )
                conn.execute(
                    insert(chat_messages_table).values(
                        id=str(msg_id),
                        session_id=sid,
                        role=role,
                        content=content,
                        raw_content=raw_content,
                        tool_calls=tool_calls,
                        created_at=now,
                        composition_state_id=csid,
                    )
                )
                conn.execute(update(sessions_table).where(sessions_table.c.id == sid).values(updated_at=now))

        await self._run_sync(_sync)

        return ChatMessageRecord(
            id=msg_id,
            session_id=session_id,
            role=role,
            content=content,
            raw_content=raw_content,
            tool_calls=tool_calls,
            created_at=now,
            composition_state_id=composition_state_id,
        )

    async def get_messages(
        self,
        session_id: UUID,
        limit: int | None = 100,
        offset: int = 0,
    ) -> list[ChatMessageRecord]:
        """Get messages for a session, ordered by created_at ascending."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(
                    select(chat_messages_table)
                    .where(chat_messages_table.c.session_id == str(session_id))
                    .order_by(chat_messages_table.c.created_at)
                    .limit(limit)
                    .offset(offset)
                ).fetchall()

        rows = await self._run_sync(_sync)

        return [
            ChatMessageRecord(
                id=UUID(row.id),
                session_id=UUID(row.session_id),
                role=row.role,
                content=row.content,
                raw_content=row.raw_content,
                tool_calls=row.tool_calls,
                created_at=self._ensure_utc(row.created_at),
                composition_state_id=UUID(row.composition_state_id) if row.composition_state_id else None,
            )
            for row in rows
        ]

    async def save_composition_state(
        self,
        session_id: UUID,
        state: CompositionStateData,
    ) -> CompositionStateRecord:
        """Save a new immutable composition state snapshot.

        Version is max(existing versions for session) + 1, starting at 1.
        """
        state_id = uuid.uuid4()
        now = self._now()
        sid = str(session_id)

        # Seam contract A: wrap JSON columns with _version envelope
        # for schema evolution. deep_thaw() handles MappingProxyType→dict
        # and tuple→list from freeze_fields().
        def _enveloped(val: Any) -> Any:
            raw = deep_thaw(val)
            if raw is None:
                return None
            return {"_version": 1, "data": raw}

        def _sync() -> int:
            # Retry loop handles concurrent version increment (TOCTOU).
            # The UniqueConstraint on (session_id, version) is the real guard.
            for _attempt in range(3):
                try:
                    return _try_insert_state()
                except IntegrityError:
                    # The only constraint on this insert is uq_composition_state_version
                    # (PK is UUID4, FK is pre-validated). Retry with next version number.
                    continue
            raise RuntimeError(f"Failed to allocate version for session {sid} after 3 attempts")

        def _try_insert_state() -> int:
            with self._engine.begin() as conn:
                result = conn.execute(
                    select(func.max(composition_states_table.c.version)).where(composition_states_table.c.session_id == sid)
                ).scalar()
                version = (result or 0) + 1

                conn.execute(
                    insert(composition_states_table).values(
                        id=str(state_id),
                        session_id=sid,
                        version=version,
                        source=_enveloped(state.source),
                        nodes=_enveloped(state.nodes),
                        edges=_enveloped(state.edges),
                        outputs=_enveloped(state.outputs),
                        metadata_=_enveloped(state.metadata_),
                        is_valid=state.is_valid,
                        validation_errors=deep_thaw(state.validation_errors),
                        derived_from_state_id=None,
                        created_at=now,
                    )
                )
                return version

        version = await self._run_sync(_sync)

        return CompositionStateRecord(
            id=state_id,
            session_id=session_id,
            version=version,
            source=state.source,
            nodes=state.nodes,
            edges=state.edges,
            outputs=state.outputs,
            metadata_=state.metadata_,
            is_valid=state.is_valid,
            validation_errors=state.validation_errors,
            created_at=now,
            derived_from_state_id=None,
        )

    async def get_current_state(
        self,
        session_id: UUID,
    ) -> CompositionStateRecord | None:
        """Return the highest-version state for a session, or None."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == str(session_id))
                    .order_by(desc(composition_states_table.c.version))
                    .limit(1)
                ).fetchone()

        row = await self._run_sync(_sync)

        if row is None:
            return None

        return self._row_to_state_record(row)

    async def get_state_versions(
        self,
        session_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[CompositionStateRecord]:
        """Return state versions for a session, ascending order."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(
                    select(composition_states_table)
                    .where(composition_states_table.c.session_id == str(session_id))
                    .order_by(composition_states_table.c.version)
                    .limit(limit)
                    .offset(offset)
                ).fetchall()

        rows = await self._run_sync(_sync)

        return [self._row_to_state_record(row) for row in rows]

    @staticmethod
    def _unwrap_envelope(val: Any) -> Any:
        """Unwrap _version envelope from a JSON column value.

        Seam contract A: JSON columns are stored with {"_version": 1, "data": ...}.
        Raises ValueError on unknown versions. Returns None for NULL columns.
        """
        if val is None:
            return None
        if isinstance(val, dict) and "_version" in val:
            if val["_version"] != 1:
                raise ValueError(f"Unknown composition state envelope version: {val['_version']}")
            return val["data"]
        raise ValueError(
            f"Composition state column has no _version envelope: {val!r}. This indicates a bug in the write path or database corruption."
        )

    def _row_to_state_record(self, row: Any) -> CompositionStateRecord:
        """Convert a SQLAlchemy row to a CompositionStateRecord.

        Seam contract A: metadata_ maps DB column metadata_ back to the
        dataclass field. JSON columns are unwrapped from their _version envelope.
        """
        return CompositionStateRecord(
            id=UUID(row.id),
            session_id=UUID(row.session_id),
            version=row.version,
            source=self._unwrap_envelope(row.source),
            nodes=self._unwrap_envelope(row.nodes),
            edges=self._unwrap_envelope(row.edges),
            outputs=self._unwrap_envelope(row.outputs),
            metadata_=self._unwrap_envelope(row.metadata_),
            is_valid=row.is_valid,
            validation_errors=row.validation_errors,
            created_at=self._ensure_utc(row.created_at),
            derived_from_state_id=(UUID(row.derived_from_state_id) if row.derived_from_state_id is not None else None),
        )

    async def create_run(
        self,
        session_id: UUID,
        state_id: UUID,
        pipeline_yaml: str | None = None,
    ) -> RunRecord:
        """Create a new pending run, enforcing one active run per session (B6).

        Enforced by partial unique index uq_runs_one_active_per_session
        (at most one row with status IN ('pending','running') per session_id).
        The SELECT is an early-out optimization; the index is the real guard.
        Raises RunAlreadyActiveError if a pending or running run exists.
        """
        run_id = uuid.uuid4()
        now = self._now()
        sid = str(session_id)
        state_sid = str(state_id)

        def _sync() -> None:
            with self._engine.begin() as conn:
                _assert_state_in_session(
                    conn,
                    state_id=state_sid,
                    expected_session_id=sid,
                    caller="create_run",
                )

                # Early-out: check before INSERT to give a clear error message
                active = conn.execute(
                    select(runs_table.c.id).where(
                        runs_table.c.session_id == sid,
                        runs_table.c.status.in_(["pending", "running"]),
                    )
                ).fetchone()

                if active is not None:
                    raise RunAlreadyActiveError(sid)

                try:
                    conn.execute(
                        insert(runs_table).values(
                            id=str(run_id),
                            session_id=sid,
                            state_id=state_sid,
                            status="pending",
                            started_at=now,
                            rows_processed=0,
                            rows_failed=0,
                            pipeline_yaml=pipeline_yaml,
                        )
                    )
                except IntegrityError as exc:
                    # The pre-check for active runs passed, but a concurrent insert
                    # hit the partial unique index. This is genuinely "run already active."
                    raise RunAlreadyActiveError(sid) from exc

        await self._run_sync(_sync)

        return RunRecord(
            id=run_id,
            session_id=session_id,
            state_id=state_id,
            status="pending",
            started_at=now,
            finished_at=None,
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
            error=None,
            landscape_run_id=None,
            pipeline_yaml=pipeline_yaml,
        )

    async def get_run(self, run_id: UUID) -> RunRecord:
        """Fetch a run by ID. Raises ValueError if not found."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(select(runs_table).where(runs_table.c.id == str(run_id))).fetchone()

        row = await self._run_sync(_sync)

        if row is None:
            raise ValueError(f"Run not found: {run_id}")

        return self._row_to_run_record(row)

    async def list_runs_for_session(self, session_id: UUID) -> list[RunRecord]:
        """List all runs for a session, newest first."""
        sid = str(session_id)

        def _sync() -> Any:
            with self._engine.connect() as conn:
                return conn.execute(
                    select(runs_table).where(runs_table.c.session_id == sid).order_by(runs_table.c.started_at.desc())
                ).fetchall()

        rows = await self._run_sync(_sync)
        return [self._row_to_run_record(row) for row in rows]

    async def update_run_status(
        self,
        run_id: UUID,
        status: SessionRunStatus,
        error: str | None = None,
        landscape_run_id: str | None = None,
        rows_processed: int | None = None,
        rows_succeeded: int | None = None,
        rows_failed: int | None = None,
        rows_routed_success: int | None = None,
        rows_routed_failure: int | None = None,
        rows_quarantined: int | None = None,
    ) -> None:
        """Update a run's status and optional fields.

        Enforces LEGAL_RUN_TRANSITIONS (D3). Enforces landscape_run_id
        write-once semantics (D4). Sets finished_at for terminal states
        (completed, failed, cancelled). Optional parameters only update
        the column when not None. Raises ValueError if run not found or
        transition is illegal.
        """
        now = self._now()
        rid = str(run_id)

        def _sync() -> None:
            with self._engine.begin() as conn:
                # Read current state for transition + write-once validation
                current = conn.execute(
                    select(
                        runs_table.c.status,
                        runs_table.c.landscape_run_id,
                    ).where(runs_table.c.id == rid)
                ).fetchone()

                if current is None:
                    raise ValueError(f"Run not found: {run_id}")

                # D3: Enforce legal transitions — direct access; KeyError = Tier 1 crash.
                # Use the narrow subclass so the cancelled-race recovery in
                # ExecutionService can match identity, not message.  See
                # IllegalRunTransitionError docstring for the full rationale.
                current_status = current.status
                allowed = LEGAL_RUN_TRANSITIONS[current_status]
                if status not in allowed:
                    raise IllegalRunTransitionError(current_status, status, allowed)

                # D4: landscape_run_id is write-once
                if landscape_run_id is not None and current.landscape_run_id is not None:
                    raise ValueError(f"landscape_run_id already set to {current.landscape_run_id!r}; cannot overwrite")
                if status in OPERATOR_COMPLETION_RUN_STATUS_VALUES and not (landscape_run_id or current.landscape_run_id):
                    raise ValueError(f"{status} status requires landscape_run_id")
                if status == "failed" and not error:
                    raise ValueError("failed status requires error")

                values: dict[str, Any] = {"status": status}
                if status in SESSION_TERMINAL_RUN_STATUS_VALUES:
                    values["finished_at"] = now
                if error is not None:
                    values["error"] = error
                if landscape_run_id is not None:
                    values["landscape_run_id"] = landscape_run_id
                if rows_processed is not None:
                    values["rows_processed"] = rows_processed
                if rows_succeeded is not None:
                    values["rows_succeeded"] = rows_succeeded
                if rows_failed is not None:
                    values["rows_failed"] = rows_failed
                if rows_routed_success is not None:
                    values["rows_routed_success"] = rows_routed_success
                if rows_routed_failure is not None:
                    values["rows_routed_failure"] = rows_routed_failure
                if rows_quarantined is not None:
                    values["rows_quarantined"] = rows_quarantined

                conn.execute(update(runs_table).where(runs_table.c.id == rid).values(**values))

        await self._run_sync(_sync)

    async def get_active_run(
        self,
        session_id: UUID,
    ) -> RunRecord | None:
        """Return the pending/running run for a session, or None."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(
                    select(runs_table).where(
                        runs_table.c.session_id == str(session_id),
                        runs_table.c.status.in_(["pending", "running"]),
                    )
                ).fetchone()

        row = await self._run_sync(_sync)

        if row is None:
            return None

        return self._row_to_run_record(row)

    async def get_state(self, state_id: UUID) -> CompositionStateRecord:
        """Fetch a composition state by its primary key. Raises ValueError if not found."""

        def _sync() -> Any:
            with self._engine.begin() as conn:
                return conn.execute(select(composition_states_table).where(composition_states_table.c.id == str(state_id))).fetchone()

        row = await self._run_sync(_sync)

        if row is None:
            raise ValueError(f"State not found: {state_id}")

        return self._row_to_state_record(row)

    async def get_state_in_session(
        self,
        state_id: UUID,
        session_id: UUID,
    ) -> CompositionStateRecord:
        """Scoped read: fetch state and verify it belongs to ``session_id``.

        Runtime defence-in-depth complementing the current-schema
        composite foreign key: persisted data cannot create cross-session
        state references at the schema layer, and this method raises
        ``AuditIntegrityError`` on any mismatch it
        encounters. The exception class is chosen deliberately to match
        the cross-session blob-ref rejection in the
        ``fork_from_message`` route handler (web/sessions/routes.py):
        a Tier 1 anomaly in our own data must surface as corruption,
        not as a soft 404. ``ValueError`` on "state does not exist at
        all" is preserved from ``get_state`` for callers that still
        need to distinguish absence from mismatch.

        The ``AuditIntegrityError`` class is the same signal used by
        the ``fork_from_message`` route handler in
        ``sessions/routes.py`` for the cross-session blob-reference
        rejection — the two guards are the read-side and write-side of
        the same Tier 1 invariant.
        """
        record = await self.get_state(state_id)
        if record.session_id != session_id:
            raise AuditIntegrityError(
                f"Tier 1 audit anomaly: composition_state {state_id} "
                f"belongs to session {record.session_id}, not {session_id}. "
                f"Migration 007 composite FK prevents this for post-007 "
                f"data; pre-007 orphans should have been deleted by "
                f"Variant-A repair. Cross-session state reference rejected."
            )
        return record

    async def set_active_state(
        self,
        session_id: UUID,
        state_id: UUID,
    ) -> CompositionStateRecord:
        """Revert to a prior state by copying it as a new version.

        Creates a new version record that is a copy of the specified prior
        version (looked up by state_id). The new record gets
        version = max(existing) + 1. Raises ValueError if state_id not
        found or does not belong to the session.
        """
        sid = str(session_id)
        new_state_id = uuid.uuid4()
        now = self._now()

        def _sync() -> tuple[Any, int]:
            # Retry loop handles concurrent version increment (TOCTOU).
            # The UniqueConstraint on (session_id, version) is the real guard.
            for _attempt in range(3):
                try:
                    return _try_insert_revert()
                except IntegrityError:
                    # The only constraint on this insert is uq_composition_state_version
                    # (PK is UUID4, FK is pre-validated). Retry with next version number.
                    continue
            raise RuntimeError(f"Failed to allocate version for session {sid} after 3 attempts")

        def _try_insert_revert() -> tuple[Any, int]:
            with self._engine.begin() as conn:
                prior_row = conn.execute(select(composition_states_table).where(composition_states_table.c.id == str(state_id))).fetchone()

                # NOTE: Both branches below raise ValueError (not RuntimeError),
                # and the HTTP handler at routes.py maps ValueError to 404. This
                # is INTENTIONAL and distinct from _assert_state_in_session
                # (module-level) which raises RuntimeError on cross-session
                # references:
                #
                #   * _assert_state_in_session guards internal callers that
                #     supply BOTH session_id and state_id from the same scope
                #     (e.g. add_message, create_run). A mismatch there is a
                #     caller-code contract violation — RuntimeError/500 is
                #     the correct signal because no legitimate user input
                #     can produce it.
                #
                #   * set_active_state receives state_id from the HTTP body
                #     while session_id comes from the authenticated URL path.
                #     A state owned by another user's session is
                #     indistinguishable from "does not exist" to this user —
                #     surfacing a RuntimeError/500 would leak the existence
                #     of that other session's states. Collapsing both cases
                #     to ValueError -> 404 is the correct information-hiding
                #     boundary for user-supplied identifiers.
                #
                # If you find yourself tempted to consolidate these checks,
                # reconsider: the exception type is load-bearing because it
                # encodes WHO is wrong (caller code vs. user) and the HTTP
                # status depends on it.
                if prior_row is None:
                    raise ValueError(f"State not found: {state_id}")
                if prior_row.session_id != sid:
                    raise ValueError(f"State {state_id} does not belong to session {session_id}")

                max_version = conn.execute(
                    select(func.max(composition_states_table.c.version)).where(composition_states_table.c.session_id == sid)
                ).scalar()
                new_version = (max_version or 0) + 1

                conn.execute(
                    insert(composition_states_table).values(
                        id=str(new_state_id),
                        session_id=sid,
                        version=new_version,
                        # prior_row.* values are already enveloped — copy as-is
                        source=prior_row.source,
                        nodes=prior_row.nodes,
                        edges=prior_row.edges,
                        outputs=prior_row.outputs,
                        metadata_=prior_row.metadata_,
                        is_valid=prior_row.is_valid,
                        validation_errors=prior_row.validation_errors,
                        derived_from_state_id=str(state_id),
                        created_at=now,
                    )
                )
                return prior_row, new_version

        prior_row, new_version = await self._run_sync(_sync)

        return CompositionStateRecord(
            id=new_state_id,
            session_id=session_id,
            version=new_version,
            source=self._unwrap_envelope(prior_row.source),
            nodes=self._unwrap_envelope(prior_row.nodes),
            edges=self._unwrap_envelope(prior_row.edges),
            outputs=self._unwrap_envelope(prior_row.outputs),
            metadata_=self._unwrap_envelope(prior_row.metadata_),
            is_valid=prior_row.is_valid,
            validation_errors=prior_row.validation_errors,
            created_at=now,
            derived_from_state_id=state_id,
        )

    async def cancel_orphaned_runs(
        self,
        session_id: UUID,
        max_age_seconds: int = 3600,
    ) -> list[RunRecord]:
        """Force-cancel runs stuck in 'pending' or 'running' beyond max_age_seconds.

        Returns the list of cancelled RunRecords. Called by the execution
        service on startup and periodically to prevent orphaned runs from
        permanently blocking sessions (D5). Includes 'pending' because a
        crash between create_run() and the first update_run_status("running")
        would leave a permanently unblockable session otherwise.
        """
        sid = str(session_id)
        now = self._now()
        cutoff = now - timedelta(seconds=max_age_seconds)

        def _sync() -> list[RunRecord]:
            cancelled: list[RunRecord] = []
            with self._engine.begin() as conn:
                stale_rows = conn.execute(
                    select(runs_table).where(
                        runs_table.c.session_id == sid,
                        runs_table.c.status.in_(["pending", "running"]),
                        runs_table.c.started_at <= cutoff,
                    )
                ).fetchall()

                for row in stale_rows:
                    conn.execute(update(runs_table).where(runs_table.c.id == row.id).values(status="cancelled", finished_at=now))
                    cancelled.append(
                        RunRecord(
                            id=UUID(row.id),
                            session_id=UUID(row.session_id),
                            state_id=UUID(row.state_id),
                            status="cancelled",
                            started_at=self._ensure_utc(row.started_at),
                            finished_at=now,
                            rows_processed=row.rows_processed,
                            rows_succeeded=row.rows_succeeded,
                            rows_failed=row.rows_failed,
                            rows_routed_success=row.rows_routed_success,
                            rows_routed_failure=row.rows_routed_failure,
                            rows_quarantined=row.rows_quarantined,
                            error=row.error,
                            landscape_run_id=row.landscape_run_id,
                            pipeline_yaml=row.pipeline_yaml,
                        )
                    )
            return cancelled

        result: list[RunRecord] = cast(list[RunRecord], await self._run_sync(_sync))
        return result

    async def cancel_all_orphaned_runs(
        self,
        max_age_seconds: int | None = None,
        exclude_run_ids: frozenset[str] = frozenset(),
        reason: str | None = None,
    ) -> int:
        """Force-cancel orphaned runs across all sessions.

        Called on startup to recover sessions blocked by runs orphaned
        during a previous server crash. Returns the count of cancelled runs.

        Args:
            max_age_seconds: Only cancel runs older than this. None cancels
                all non-terminal runs (correct for single-process servers
                where every non-terminal run is orphaned after restart).
            exclude_run_ids: Run IDs known to have active executor threads.
                These are skipped even if they exceed max_age_seconds.
            reason: Written to the error column so operators can distinguish
                orphan-cleanup cancellations from user cancellations.
        """
        now = self._now()

        def _sync() -> int:
            with self._engine.begin() as conn:
                conditions: list[ColumnElement[bool]] = [runs_table.c.status.in_(["pending", "running"])]
                if max_age_seconds is not None:
                    cutoff = now - timedelta(seconds=max_age_seconds)
                    conditions.append(runs_table.c.started_at <= cutoff)
                if exclude_run_ids:
                    conditions.append(runs_table.c.id.not_in(exclude_run_ids))

                stale_rows = conn.execute(select(runs_table.c.id).where(*conditions)).fetchall()

                values: dict[str, Any] = {"status": "cancelled", "finished_at": now}
                if reason is not None:
                    values["error"] = reason

                for row in stale_rows:
                    conn.execute(update(runs_table).where(runs_table.c.id == row.id).values(**values))
                return len(stale_rows)

        return cast(int, await self._run_sync(_sync))

    async def prune_state_versions(
        self,
        session_id: UUID,
        keep_latest: int = 50,
    ) -> int:
        """Delete old composition state versions beyond keep_latest.

        Preserves the most recent `keep_latest` versions and any versions
        referenced by a run (via runs.state_id). Returns the count of
        deleted versions.
        """
        sid = str(session_id)

        def _sync() -> int:
            with self._engine.begin() as conn:
                # Get all version IDs ordered by version DESC
                all_rows = conn.execute(
                    select(
                        composition_states_table.c.id,
                        composition_states_table.c.version,
                    )
                    .where(composition_states_table.c.session_id == sid)
                    .order_by(desc(composition_states_table.c.version))
                ).fetchall()

                if len(all_rows) <= keep_latest:
                    return 0

                # IDs to keep: the top keep_latest versions
                keep_ids = {row.id for row in all_rows[:keep_latest]}

                # IDs referenced by runs
                run_referenced = {
                    row.state_id
                    for row in conn.execute(
                        select(runs_table.c.state_id).where(
                            runs_table.c.session_id == sid,
                        )
                    ).fetchall()
                }

                # IDs referenced by chat messages (provenance tracking)
                message_referenced = {
                    row.composition_state_id
                    for row in conn.execute(
                        select(chat_messages_table.c.composition_state_id).where(
                            chat_messages_table.c.session_id == sid,
                            chat_messages_table.c.composition_state_id.isnot(None),
                        )
                    ).fetchall()
                }

                # IDs referenced via derived_from_state_id (revert lineage).
                # Build transitive closure: if v5→v3→v1 and v5 is kept,
                # both v3 and v1 must be protected.
                derived_from_map: dict[str, str | None] = {
                    row.id: row.derived_from_state_id
                    for row in conn.execute(
                        select(
                            composition_states_table.c.id,
                            composition_states_table.c.derived_from_state_id,
                        ).where(composition_states_table.c.session_id == sid)
                    ).fetchall()
                }
                lineage_protected: set[str] = set()
                seeds = keep_ids | run_referenced | message_referenced
                for seed_id in seeds:
                    parent = derived_from_map.get(seed_id)
                    while parent is not None and parent not in lineage_protected:
                        lineage_protected.add(parent)
                        parent = derived_from_map.get(parent)

                # Candidates for deletion: not kept, not referenced, not in lineage
                protected = keep_ids | run_referenced | message_referenced | lineage_protected
                delete_ids = [row.id for row in all_rows if row.id not in protected]

                if not delete_ids:
                    return 0

                result = conn.execute(delete(composition_states_table).where(composition_states_table.c.id.in_(delete_ids)))
                return result.rowcount

        return cast(int, await self._run_sync(_sync))

    async def fork_session(
        self,
        source_session_id: UUID,
        fork_message_id: UUID,
        new_message_content: str,
        user_id: str,
        auth_provider_type: str,
    ) -> tuple[SessionRecord, list[ChatMessageRecord], CompositionStateRecord | None]:
        """Fork a session from a specific user message.

        All writes happen in a single transaction — if anything fails,
        the entire fork is rolled back with no partial state.

        Creates a new session containing:
        1. Composition state copied from the fork message's pre-send state
        2. All messages BEFORE the fork message (with NULL state provenance)
        3. A synthetic system message noting the fork
        4. The new edited user message (provenance = copied state, not source)

        Returns (new_session, new_messages, copied_state_or_none).
        """
        from elspeth.web.sessions.protocol import InvalidForkTargetError

        # Load source data (read-only, outside the write transaction)
        source_session = await self.get_session(source_session_id)
        source_messages = await self.get_messages(source_session_id, limit=None)

        # Find the fork message — must be a user message
        fork_msg = None
        fork_idx = -1
        for i, msg in enumerate(source_messages):
            if msg.id == fork_message_id:
                fork_msg = msg
                fork_idx = i
                break

        if fork_msg is None:
            raise ValueError(f"Message {fork_message_id} not found in session {source_session_id}")
        if fork_msg.role != "user":
            raise InvalidForkTargetError(str(fork_message_id), fork_msg.role)

        messages_to_copy = source_messages[:fork_idx]
        pre_send_state_id = fork_msg.composition_state_id

        # Load source composition state if it exists (read-only)
        source_state_record: CompositionStateRecord | None = None
        if pre_send_state_id is not None:
            source_state_record = await self.get_state_in_session(
                pre_send_state_id,
                source_session_id,
            )

        # Prepare IDs and timestamps upfront
        new_session_id = uuid.uuid4()
        new_session_id_str = str(new_session_id)
        now = self._now()
        title = f"{source_session.title} (fork)"

        # Prepare state copy if needed
        copied_state_id = uuid.uuid4() if source_state_record is not None else None
        copied_state_id_str = str(copied_state_id) if copied_state_id else None

        # Prepare all message rows upfront — preserve original created_at
        # so get_messages() ordering is deterministic.  Stamping all rows
        # with `now` would make them indistinguishable by timestamp and
        # produce non-deterministic ordering on subsequent reads.
        msg_records_data: list[dict[str, Any]] = []
        for msg in messages_to_copy:
            msg_records_data.append(
                {
                    "id": str(uuid.uuid4()),
                    "session_id": new_session_id_str,
                    "role": msg.role,
                    "content": msg.content,
                    # raw_content preserves audit provenance for intercepted assistant turns
                    "raw_content": msg.raw_content,
                    "tool_calls": deep_thaw(msg.tool_calls) if msg.tool_calls else None,
                    "created_at": msg.created_at,
                    "composition_state_id": None,  # Don't reference source session states
                }
            )
        # System message — no raw_content (synthetic, not from the LLM)
        system_msg_id = str(uuid.uuid4())
        msg_records_data.append(
            {
                "id": system_msg_id,
                "session_id": new_session_id_str,
                "role": "system",
                "content": "Conversation forked from an earlier point.",
                "raw_content": None,
                "tool_calls": None,
                "created_at": now,
                "composition_state_id": None,
            }
        )
        # New edited user message — provenance points to COPIED state, not source.
        # Offset by 1 microsecond so get_messages() ordering is deterministic
        # (system note before user turn).  Without this, SQLite/Postgres can
        # return the two rows in either order since they share created_at.
        # raw_content is None: this is a new user-authored message, not an LLM turn.
        new_user_msg_id = str(uuid.uuid4())
        msg_records_data.append(
            {
                "id": new_user_msg_id,
                "session_id": new_session_id_str,
                "role": "user",
                "content": new_message_content,
                "raw_content": None,
                "tool_calls": None,
                "created_at": now + timedelta(microseconds=1),
                "composition_state_id": copied_state_id_str,
            }
        )

        def _enveloped(val: Any) -> Any:
            raw = deep_thaw(val)
            if raw is None:
                return None
            return {"_version": 1, "data": raw}

        def _sync() -> int | None:
            """Single atomic transaction for the entire fork."""
            with self._engine.begin() as conn:
                # 1. Create session
                conn.execute(
                    insert(sessions_table).values(
                        id=new_session_id_str,
                        user_id=user_id,
                        auth_provider_type=auth_provider_type,
                        title=title,
                        created_at=now,
                        updated_at=now,
                        forked_from_session_id=str(source_session_id),
                        forked_from_message_id=str(fork_message_id),
                    )
                )

                # 2. Copy composition state (before messages, so FK is valid)
                state_version: int | None = None
                if source_state_record is not None and copied_state_id_str is not None:
                    state_version = 1
                    conn.execute(
                        insert(composition_states_table).values(
                            id=copied_state_id_str,
                            session_id=new_session_id_str,
                            version=1,
                            source=_enveloped(source_state_record.source),
                            nodes=_enveloped(source_state_record.nodes),
                            edges=_enveloped(source_state_record.edges),
                            outputs=_enveloped(source_state_record.outputs),
                            metadata_=_enveloped(source_state_record.metadata_),
                            is_valid=source_state_record.is_valid,
                            validation_errors=deep_thaw(source_state_record.validation_errors),
                            derived_from_state_id=None,
                            created_at=now,
                        )
                    )

                # 3. Insert all messages in batch
                if msg_records_data:
                    conn.execute(insert(chat_messages_table), msg_records_data)

                return state_version

        state_version = await self._run_sync(_sync)

        # Build return records from the pre-computed data
        new_session = SessionRecord(
            id=new_session_id,
            user_id=user_id,
            auth_provider_type=auth_provider_type,
            title=title,
            created_at=now,
            updated_at=now,
            forked_from_session_id=source_session_id,
            forked_from_message_id=fork_message_id,
        )

        new_messages = [
            ChatMessageRecord(
                id=UUID(d["id"]),
                session_id=new_session_id,
                role=d["role"],
                content=d["content"],
                raw_content=d["raw_content"],
                tool_calls=d["tool_calls"],
                created_at=d["created_at"],
                composition_state_id=UUID(d["composition_state_id"]) if d["composition_state_id"] else None,
            )
            for d in msg_records_data
        ]

        copied_state: CompositionStateRecord | None = None
        if source_state_record is not None and copied_state_id is not None and state_version is not None:
            copied_state = CompositionStateRecord(
                id=copied_state_id,
                session_id=new_session_id,
                version=state_version,
                source=source_state_record.source,
                nodes=source_state_record.nodes,
                edges=source_state_record.edges,
                outputs=source_state_record.outputs,
                metadata_=source_state_record.metadata_,
                is_valid=source_state_record.is_valid,
                validation_errors=source_state_record.validation_errors,
                created_at=now,
                derived_from_state_id=None,
            )

        return new_session, new_messages, copied_state

    async def update_message_composition_state(
        self,
        message_id: UUID,
        composition_state_id: UUID,
    ) -> None:
        """Re-point a message's composition_state_id to a different state.

        Enforces same-session ownership: the target state must belong to
        the same session as the message. Cross-session re-pointing is a
        caller bug and raises RuntimeError.
        """
        mid = str(message_id)
        csid = str(composition_state_id)

        def _sync() -> None:
            with self._engine.begin() as conn:
                message_session_id = conn.execute(select(chat_messages_table.c.session_id).where(chat_messages_table.c.id == mid)).scalar()
                if message_session_id is None:
                    raise ValueError(f"Message {message_id} not found")

                _assert_state_in_session(
                    conn,
                    state_id=csid,
                    expected_session_id=str(message_session_id),
                    caller="update_message_composition_state",
                )

                conn.execute(update(chat_messages_table).where(chat_messages_table.c.id == mid).values(composition_state_id=csid))

        await self._run_sync(_sync)

    def _row_to_run_record(self, row: Any) -> RunRecord:
        """Convert a SQLAlchemy row to a RunRecord."""
        return RunRecord(
            id=UUID(row.id),
            session_id=UUID(row.session_id),
            state_id=UUID(row.state_id),
            status=row.status,
            started_at=self._ensure_utc(row.started_at),
            finished_at=self._ensure_utc(row.finished_at) if row.finished_at is not None else None,
            rows_processed=row.rows_processed,
            rows_succeeded=row.rows_succeeded,
            rows_failed=row.rows_failed,
            rows_routed_success=row.rows_routed_success,
            rows_routed_failure=row.rows_routed_failure,
            rows_quarantined=row.rows_quarantined,
            error=row.error,
            landscape_run_id=row.landscape_run_id,
            pipeline_yaml=row.pipeline_yaml,
        )
