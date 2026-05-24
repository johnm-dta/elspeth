"""Tests for SessionServiceImpl -- CRUD, state versioning, active run enforcement."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
import structlog
from sqlalchemy import insert, select
from sqlalchemy.pool import StaticPool

from elspeth.web.execution.schemas import (
    RunAccounting,
    RunAccountingIntegrity,
    RunAccountingRouting,
    RunAccountingSource,
    RunAccountingTokens,
    RunStatusResponse,
)
from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import composer_completion_events_table
from elspeth.web.sessions.protocol import (
    ChatMessageRecord,
    CompositionStateData,
    CompositionStateRecord,
    RunAlreadyActiveError,
    RunRecord,
    SessionRecord,
)
from elspeth.web.sessions.schema import initialize_session_schema
from elspeth.web.sessions.service import SessionServiceImpl
from elspeth.web.sessions.telemetry import build_sessions_telemetry


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine with all tables.

    Uses StaticPool so that run_in_executor threads share the same
    in-memory database connection.
    """
    eng = create_session_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    initialize_session_schema(eng)
    return eng


@pytest.fixture
def service(engine):
    """Create a SessionServiceImpl backed by the in-memory engine."""
    return SessionServiceImpl(
        engine,
        telemetry=build_sessions_telemetry(),
        log=structlog.get_logger("test"),
    )


class TestSessionCRUD:
    """Tests for session create, get, list, and archive."""

    @pytest.mark.asyncio
    async def test_create_session(self, service) -> None:
        session = await service.create_session("alice", "My Session", "local")
        assert isinstance(session, SessionRecord)
        assert session.user_id == "alice"
        assert session.auth_provider_type == "local"
        assert session.title == "My Session"
        assert isinstance(session.id, uuid.UUID)
        assert isinstance(session.created_at, datetime)

    @pytest.mark.asyncio
    async def test_get_session(self, service) -> None:
        created = await service.create_session("alice", "Test", "local")
        fetched = await service.get_session(created.id)
        assert fetched.id == created.id
        assert fetched.user_id == "alice"
        assert fetched.title == "Test"

    @pytest.mark.asyncio
    async def test_update_session_title_persists_and_refreshes_timestamp(self, service) -> None:
        created = await service.create_session("alice", "Test", "local")

        updated = await service.update_session_title(created.id, "Renamed pipeline")

        assert updated.id == created.id
        assert updated.title == "Renamed pipeline"
        assert updated.updated_at >= created.updated_at
        fetched = await service.get_session(created.id)
        assert fetched.title == "Renamed pipeline"

    @pytest.mark.asyncio
    async def test_get_session_not_found_raises(self, service) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.get_session(uuid.uuid4())

    @pytest.mark.asyncio
    async def test_list_sessions_user_scoped(self, service) -> None:
        await service.create_session("alice", "Session A", "local")
        await service.create_session("alice", "Session B", "local")
        await service.create_session("bob", "Session C", "local")

        alice_sessions = await service.list_sessions("alice", "local")
        assert len(alice_sessions) == 2
        assert all(s.user_id == "alice" for s in alice_sessions)

        bob_sessions = await service.list_sessions("bob", "local")
        assert len(bob_sessions) == 1

    @pytest.mark.asyncio
    async def test_list_sessions_ordered_by_updated_at_desc(self, service) -> None:
        s1 = await service.create_session("alice", "First", "local")
        await service.create_session("alice", "Second", "local")
        # Add a message to s1 to update its updated_at
        await service.add_message(s1.id, "user", "hello", writer_principal="route_user_message")

        sessions = await service.list_sessions("alice", "local")
        # s1 should be first (most recently updated)
        assert sessions[0].id == s1.id

    @pytest.mark.asyncio
    async def test_archive_session_deletes_unrun_session(self, service) -> None:
        session = await service.create_session("alice", "To Archive", "local")
        await service.add_message(session.id, "user", "hello", writer_principal="route_user_message")
        await service.archive_session(session.id)

        with pytest.raises(ValueError):
            await service.get_session(session.id)

        messages = await service.get_messages(session.id)
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_archive_session_hides_session_with_durable_completion_history(self, engine, service) -> None:
        session = await service.create_session("alice", "To Archive", "local")
        await service.add_message(session.id, "user", "hello", writer_principal="route_user_message")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        with engine.begin() as conn:
            conn.execute(
                insert(composer_completion_events_table).values(
                    id=str(uuid.uuid4()),
                    session_id=str(session.id),
                    composition_state_id=str(state.id),
                    event_type="export_yaml",
                    actor="user:alice",
                    created_at=datetime.now(UTC),
                )
            )

        await service.archive_session(session.id)

        archived = await service.get_session(session.id)
        assert archived.archived_at is not None

        visible_sessions = await service.list_sessions("alice", "local")
        assert [s.id for s in visible_sessions] == []

        archived_sessions = await service.list_sessions("alice", "local", include_archived=True)
        assert [s.id for s in archived_sessions] == [session.id]

        messages = await service.get_messages(session.id)
        assert len(messages) == 1
        with engine.begin() as conn:
            remaining_completion_events = conn.execute(
                select(composer_completion_events_table).where(composer_completion_events_table.c.session_id == str(session.id))
            ).all()
        assert len(remaining_completion_events) == 1

    @pytest.mark.asyncio
    async def test_archive_session_deletes_blob_directory(self, engine, tmp_path) -> None:
        """Archiving a session removes its blob directory from the filesystem."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        service_with_dir = SessionServiceImpl(
            engine,
            data_dir=data_dir,
            telemetry=build_sessions_telemetry(),
            log=structlog.get_logger("test"),
        )

        session = await service_with_dir.create_session("alice", "Blob Session", "local")
        sid = str(session.id)

        # Create blob directory with a file (simulating stored blobs)
        blob_dir = data_dir / "blobs" / sid
        blob_dir.mkdir(parents=True)
        (blob_dir / "some-blob_data.csv").write_text("col1\nval1")
        assert blob_dir.is_dir()

        await service_with_dir.archive_session(session.id)

        # Blob directory should be cleaned up
        assert not blob_dir.exists()

        # Session should be gone
        with pytest.raises(ValueError):
            await service_with_dir.get_session(session.id)

    @pytest.mark.asyncio
    async def test_archive_session_quarantines_blob_dir_when_post_commit_purge_fails(
        self,
        engine,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Post-commit blob purge failure must not surface as a false delete failure.

        The session delete has already committed by the time the filesystem purge
        runs. The service must preserve a recoverable quarantine path instead of
        raising after the session is already gone.
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        service_with_dir = SessionServiceImpl(
            engine,
            data_dir=data_dir,
            telemetry=build_sessions_telemetry(),
            log=structlog.get_logger("test"),
        )

        session = await service_with_dir.create_session("alice", "Blob Session", "local")
        sid = str(session.id)
        blob_dir = data_dir / "blobs" / sid
        blob_dir.mkdir(parents=True)
        blob_file = blob_dir / "some-blob_data.csv"
        blob_file.write_text("col1\nval1")

        quarantine_dir = data_dir / ".archive_quarantine" / sid

        def fail_rmtree(_path: object) -> None:
            raise OSError("permission denied removing staged blob directory")

        monkeypatch.setattr("elspeth.web.sessions.service.shutil.rmtree", fail_rmtree)

        await service_with_dir.archive_session(session.id)

        with pytest.raises(ValueError):
            await service_with_dir.get_session(session.id)

        assert not blob_dir.exists()
        assert quarantine_dir.is_dir()
        assert (quarantine_dir / blob_file.name).read_text() == "col1\nval1"


class TestMessagePersistence:
    """Tests for chat message add and retrieval."""

    @pytest.mark.asyncio
    async def test_add_and_get_messages(self, service) -> None:
        session = await service.create_session("alice", "Chat", "local")
        msg1 = await service.add_message(session.id, "user", "Hello", writer_principal="route_user_message")
        await service.add_message(session.id, "assistant", "Hi there", writer_principal="compose_loop")

        assert isinstance(msg1, ChatMessageRecord)
        assert msg1.role == "user"
        assert msg1.content == "Hello"

        messages = await service.get_messages(session.id)
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_messages_ordered_by_created_at_asc(self, service) -> None:
        session = await service.create_session("alice", "Chat", "local")
        await service.add_message(session.id, "user", "First", writer_principal="route_user_message")
        await service.add_message(session.id, "assistant", "Second", writer_principal="compose_loop")
        await service.add_message(session.id, "user", "Third", writer_principal="route_user_message")

        messages = await service.get_messages(session.id)
        assert [m.content for m in messages] == ["First", "Second", "Third"]

    @pytest.mark.asyncio
    async def test_add_message_with_tool_calls(self, service) -> None:
        session = await service.create_session("alice", "Chat", "local")
        tool_calls_data = [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "set_source",
                    "arguments": '{"type":"csv"}',
                },
            }
        ]
        msg = await service.add_message(
            session.id,
            "assistant",
            "Setting source",
            tool_calls=tool_calls_data,
            writer_principal="compose_loop",
        )
        assert msg.tool_calls is not None

    @pytest.mark.asyncio
    async def test_add_message_updates_session_updated_at(self, service) -> None:
        session = await service.create_session("alice", "Chat", "local")
        original_updated = session.updated_at.replace(tzinfo=None)
        await service.add_message(session.id, "user", "hello", writer_principal="route_user_message")
        refreshed = await service.get_session(session.id)
        # SQLite strips timezone info; compare naive datetimes (both are UTC)
        refreshed_updated = refreshed.updated_at.replace(tzinfo=None)
        assert refreshed_updated >= original_updated


class TestCompositionStateVersioning:
    """Tests for immutable state snapshots with monotonic versioning."""

    @pytest.mark.asyncio
    async def test_first_state_version_is_1(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state_data = CompositionStateData(is_valid=False)
        state = await service.save_composition_state(session.id, state_data, provenance="session_seed")
        assert isinstance(state, CompositionStateRecord)
        assert state.version == 1
        # New states (not reverts) have no lineage (D2/D7)
        assert state.derived_from_state_id is None

    @pytest.mark.asyncio
    async def test_version_increments_monotonically(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        s1 = await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")
        s2 = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        assert s1.version == 1
        assert s2.version == 2

    @pytest.mark.asyncio
    async def test_get_current_state_returns_highest_version(
        self,
        service,
    ) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"type": "csv", "path": "old.csv"},
                is_valid=False,
            ),
            provenance="session_seed",
        )
        await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"type": "csv", "path": "new.csv"},
                is_valid=True,
            ),
            provenance="session_seed",
        )
        current = await service.get_current_state(session.id)
        assert current is not None
        assert current.version == 2
        assert current.is_valid is True

    @pytest.mark.asyncio
    async def test_named_sources_round_trip_through_session_state(self, service) -> None:
        session = await service.create_session("alice", "Multi-source", "local")
        sources = {
            "orders": {
                "plugin": "csv",
                "on_success": "orders_out",
                "options": {"path": "orders.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
            "refunds": {
                "plugin": "csv",
                "on_success": "refunds_out",
                "options": {"path": "refunds.csv", "schema": {"mode": "observed"}},
                "on_validation_failure": "quarantine",
            },
        }

        await service.save_composition_state(
            session.id,
            CompositionStateData(
                sources=sources,
                outputs=[
                    {"name": "orders_out", "plugin": "json", "options": {"path": "orders.jsonl"}, "on_write_failure": "discard"},
                    {"name": "refunds_out", "plugin": "json", "options": {"path": "refunds.jsonl"}, "on_write_failure": "discard"},
                ],
                metadata_={"name": "Multi-source", "description": ""},
                is_valid=True,
            ),
            provenance="session_seed",
        )

        current = await service.get_current_state(session.id)

        assert current is not None
        assert current.sources == sources

    @pytest.mark.asyncio
    async def test_get_current_state_returns_none_when_empty(
        self,
        service,
    ) -> None:
        session = await service.create_session("alice", "Empty", "local")
        current = await service.get_current_state(session.id)
        assert current is None

    @pytest.mark.asyncio
    async def test_composer_meta_roundtrips_through_persistence(
        self,
        service,
    ) -> None:
        """``composer_meta`` survives DB roundtrip and reaches state record.

        Regression for the ``state.composer_meta.repair_turns_used`` surface
        — the convergence-suite eval scorer reads this field via
        ``GET /api/sessions/{id}/state``. If the DB column is dropped or the
        envelope wrap/unwrap is misaligned, scoring silently ambers.
        """
        session = await service.create_session("alice", "Pipeline", "local")
        state_data = CompositionStateData(
            is_valid=True,
            composer_meta={"repair_turns_used": 1},
        )
        saved = await service.save_composition_state(session.id, state_data, provenance="session_seed")
        assert saved.composer_meta is not None
        assert saved.composer_meta["repair_turns_used"] == 1

        # Load via a different code path (get_current_state hits
        # _row_to_state_record / _unwrap_envelope) to prove the value survives
        # the JSON envelope wrap and unwrap.
        loaded = await service.get_current_state(session.id)
        assert loaded is not None
        assert loaded.composer_meta is not None
        assert loaded.composer_meta["repair_turns_used"] == 1

    @pytest.mark.asyncio
    async def test_composer_meta_absent_persists_as_none(
        self,
        service,
    ) -> None:
        """``composer_meta`` defaulting to ``None`` round-trips as ``None``.

        Honest absence: revert/fork paths and historical pre-plumbing rows
        must not synthesise a fake ``repair_turns_used: 0``. The eval scorer
        relies on this distinction (absent => AMBER with explanation, not
        silent pass).
        """
        session = await service.create_session("alice", "Pipeline", "local")
        state_data = CompositionStateData(is_valid=True)
        saved = await service.save_composition_state(session.id, state_data, provenance="session_seed")
        assert saved.composer_meta is None

        loaded = await service.get_current_state(session.id)
        assert loaded is not None
        assert loaded.composer_meta is None

    @pytest.mark.asyncio
    async def test_get_state_versions_returns_all_ascending(
        self,
        service,
    ) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")
        await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")
        await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        versions = await service.get_state_versions(session.id)
        assert len(versions) == 3
        assert [v.version for v in versions] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_state_preserves_pipeline_data(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state_data = CompositionStateData(
            source={"type": "csv", "path": "/data/input.csv"},
            nodes=[{"name": "classify", "type": "transform"}],
            edges=[{"from": "source", "to": "classify"}],
            outputs=[{"name": "results", "type": "csv_sink"}],
            metadata_={"pipeline_name": "Test Pipeline"},
            is_valid=True,
            validation_errors=None,
        )
        state = await service.save_composition_state(session.id, state_data, provenance="session_seed")
        assert state.is_valid is True


class TestOneActiveRunEnforcement:
    """Tests for B6 -- one active run per session."""

    @pytest.mark.asyncio
    async def test_second_pending_run_raises(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        # First run should succeed
        await service.create_run(session.id, state.id)
        # Second run should fail
        with pytest.raises(RunAlreadyActiveError):
            await service.create_run(session.id, state.id)

    @pytest.mark.asyncio
    async def test_create_run_returns_run_record(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        assert isinstance(run, RunRecord)
        assert run.status == "pending"
        assert run.session_id == session.id
        assert run.state_id == state.id
        assert run.pipeline_yaml is None

    @pytest.mark.asyncio
    async def test_create_run_with_pipeline_yaml(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(
            session.id,
            state.id,
            pipeline_yaml="source:\n  type: csv",
        )
        assert run.pipeline_yaml == "source:\n  type: csv"

    @pytest.mark.asyncio
    async def test_completed_run_allows_new_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        # Transition through legal path: pending -> running -> completed
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "completed", landscape_run_id="lscp-complete-1")
        # New run should succeed
        run2 = await service.create_run(session.id, state.id)
        assert run2.status == "pending"

    @pytest.mark.asyncio
    async def test_failed_run_allows_new_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        # Transition through legal path: pending -> running -> failed
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "failed", error="boom")
        run2 = await service.create_run(session.id, state.id)
        assert run2.status == "pending"

    @pytest.mark.asyncio
    async def test_running_run_blocks_new_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        with pytest.raises(RunAlreadyActiveError):
            await service.create_run(session.id, state.id)


class TestGetState:
    """Tests for get_state -- fetch a specific CompositionStateRecord by UUID."""

    @pytest.mark.asyncio
    async def test_get_state_by_id(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        saved = await service.save_composition_state(
            session.id,
            CompositionStateData(
                source={"type": "csv"},
                is_valid=True,
            ),
            provenance="session_seed",
        )
        fetched = await service.get_state(saved.id)
        assert fetched.id == saved.id
        assert fetched.version == saved.version

    @pytest.mark.asyncio
    async def test_get_state_not_found_raises(self, service) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.get_state(uuid.uuid4())


class TestGetStateInSession:
    """Tests for get_state_in_session -- scoped read with Tier 1 invariant check.

    Regression guard (P2f): list_session_runs resolves each run's
    state_id without a session-scope check. Migration 007's composite FK
    prevents future cross-session state refs at the schema layer, but
    pre-007 orphans repaired with Variant-A (delete orphans) have no
    runtime defense-in-depth. ``get_state_in_session`` is that
    defense-in-depth.
    """

    @pytest.mark.asyncio
    async def test_returns_record_when_session_matches(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        saved = await service.save_composition_state(
            session.id, CompositionStateData(source={"type": "csv"}, is_valid=True), provenance="session_seed"
        )
        fetched = await service.get_state_in_session(saved.id, session.id)
        assert fetched.id == saved.id
        assert fetched.session_id == session.id

    @pytest.mark.asyncio
    async def test_raises_audit_integrity_error_on_session_mismatch(self, service) -> None:
        """State belongs to session A, caller says it's in session B — Tier 1."""
        from elspeth.contracts.errors import AuditIntegrityError

        session_a = await service.create_session("alice", "Pipeline A", "local")
        session_b = await service.create_session("alice", "Pipeline B", "local")
        state_in_a = await service.save_composition_state(
            session_a.id, CompositionStateData(source={"type": "csv"}, is_valid=True), provenance="session_seed"
        )
        with pytest.raises(AuditIntegrityError, match="Tier 1 audit anomaly"):
            await service.get_state_in_session(state_in_a.id, session_b.id)

    @pytest.mark.asyncio
    async def test_raises_value_error_when_state_missing(self, service) -> None:
        """Nonexistent state_id must still raise ValueError, not AuditIntegrityError.

        Absence is distinguishable from corruption — callers that map to
        404 rely on the exception class to know which is which.
        """
        session = await service.create_session("alice", "Pipeline", "local")
        with pytest.raises(ValueError, match="not found"):
            await service.get_state_in_session(uuid.uuid4(), session.id)


class TestSetActiveState:
    """Tests for set_active_state -- revert by copying a prior version."""

    @pytest.mark.asyncio
    async def test_revert_creates_new_version(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        v1 = await service.save_composition_state(
            session.id, CompositionStateData(source={"type": "csv"}, is_valid=True), provenance="session_seed"
        )
        await service.save_composition_state(
            session.id, CompositionStateData(source={"type": "api"}, is_valid=True), provenance="session_seed"
        )
        # Revert to v1 -- should create v3 as a copy of v1
        reverted = await service.set_active_state(session.id, v1.id)
        assert reverted.version == 3
        # Content should match v1, not v2
        assert reverted.sources == v1.sources
        # Lineage: reverted state records where it came from (D6)
        assert reverted.derived_from_state_id == v1.id

    @pytest.mark.asyncio
    async def test_revert_preserves_named_sources(self, service) -> None:
        session = await service.create_session("alice", "Multi-source", "local")
        sources = {
            "orders": {"plugin": "csv", "on_success": "orders_rows", "on_validation_failure": "discard", "options": {"path": "orders.csv"}},
            "refunds": {
                "plugin": "csv",
                "on_success": "refunds_rows",
                "on_validation_failure": "discard",
                "options": {"path": "refunds.csv"},
            },
        }
        v1 = await service.save_composition_state(
            session.id,
            CompositionStateData(sources=sources, is_valid=True),
            provenance="session_seed",
        )
        await service.save_composition_state(
            session.id,
            CompositionStateData(source={"plugin": "json", "on_success": "rows", "on_validation_failure": "discard", "options": {}}),
            provenance="session_seed",
        )

        reverted = await service.set_active_state(session.id, v1.id)

        assert reverted.sources == sources

    @pytest.mark.asyncio
    async def test_revert_preserves_history(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")
        v2 = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        await service.set_active_state(session.id, v2.id)
        versions = await service.get_state_versions(session.id)
        # All three versions should exist (v1, v2, v3)
        assert len(versions) == 3
        assert [v.version for v in versions] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_revert_state_not_found_raises(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        with pytest.raises(ValueError, match="not found"):
            await service.set_active_state(session.id, uuid.uuid4())

    @pytest.mark.asyncio
    async def test_revert_state_wrong_session_raises(self, service) -> None:
        s1 = await service.create_session("alice", "Session 1", "local")
        s2 = await service.create_session("alice", "Session 2", "local")
        state = await service.save_composition_state(s1.id, CompositionStateData(is_valid=True), provenance="session_seed")
        with pytest.raises(ValueError, match="does not belong"):
            await service.set_active_state(s2.id, state.id)


class TestGetRun:
    """Tests for get_run -- fetch a RunRecord by UUID."""

    @pytest.mark.asyncio
    async def test_get_run_returns_record(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        created = await service.create_run(session.id, state.id)
        fetched = await service.get_run(created.id)
        assert isinstance(fetched, RunRecord)
        assert fetched.id == created.id
        assert fetched.status == "pending"

    @pytest.mark.asyncio
    async def test_get_run_not_found_raises(self, service) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.get_run(uuid.uuid4())


class TestGetActiveRun:
    """Tests for get_active_run -- pending/running run for a session."""

    @pytest.mark.asyncio
    async def test_returns_active_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        active = await service.get_active_run(session.id)
        assert active is not None
        assert active.id == run.id

    @pytest.mark.asyncio
    async def test_returns_none_when_no_active_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        active = await service.get_active_run(session.id)
        assert active is None

    @pytest.mark.asyncio
    async def test_returns_none_after_completion(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "completed", landscape_run_id="lscp-active-none")
        active = await service.get_active_run(session.id)
        assert active is None


class TestUpdateRunStatusExpanded:
    """Tests for expanded update_run_status signature (R6)."""

    @pytest.mark.asyncio
    async def test_update_with_error(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(
            run.id,
            "failed",
            error="Source file not found",
        )
        fetched = await service.get_run(run.id)
        assert fetched.status == "failed"
        assert fetched.error == "Source file not found"
        assert fetched.finished_at is not None

    @pytest.mark.asyncio
    async def test_update_with_landscape_run_id(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(
            run.id,
            "completed_with_failures",
            landscape_run_id="lscp-abc-123",
            rows_processed=100,
            rows_succeeded=4,
            rows_routed_success=4,
            rows_routed_failure=0,
            rows_failed=3,
        )
        fetched = await service.get_run(run.id)
        assert fetched.status == "completed_with_failures"
        assert fetched.landscape_run_id == "lscp-abc-123"
        assert fetched.rows_processed == 100
        assert fetched.rows_succeeded == 4
        assert fetched.rows_routed_success == 4
        assert fetched.rows_routed_failure == 0
        assert fetched.rows_failed == 3


class TestAdr019LegacyCounterReadCompatibility:
    """Pre-ADR-019 sessions.db rows used disjoint routed/quarantine counters."""

    @staticmethod
    def _accounting_from_run(run: RunRecord) -> RunAccounting:
        return RunAccounting(
            source=RunAccountingSource(rows_processed=run.rows_processed),
            tokens=RunAccountingTokens(
                emitted=run.rows_succeeded + run.rows_failed,
                terminal=run.rows_succeeded + run.rows_failed,
                succeeded=run.rows_succeeded,
                failed=run.rows_failed,
                structural=0,
                pending=0,
            ),
            routing=RunAccountingRouting(
                routed_success=run.rows_routed_success,
                routed_failure=run.rows_routed_failure,
                quarantined=run.rows_quarantined,
                discarded=0,
            ),
            integrity=RunAccountingIntegrity(
                closure="closed",
                missing_terminal_outcomes=0,
                duplicate_terminal_outcomes=0,
            ),
        )

    @staticmethod
    def _status_response_from_run(run: RunRecord) -> RunStatusResponse:
        return RunStatusResponse(
            run_id=str(run.id),
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            accounting=TestAdr019LegacyCounterReadCompatibility._accounting_from_run(run),
            error=run.error,
            landscape_run_id=run.landscape_run_id,
        )

    @pytest.mark.asyncio
    async def test_get_run_normalizes_legacy_gate_routed_success_counter(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(
            run.id,
            "completed",
            landscape_run_id="lscp-legacy-gate",
            rows_processed=4,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=4,
            rows_routed_failure=0,
            rows_quarantined=0,
        )

        fetched = await service.get_run(run.id)

        assert fetched.rows_succeeded == 4
        assert fetched.rows_routed_success == 4
        response = self._status_response_from_run(fetched)
        assert response.status == "completed"

    @pytest.mark.asyncio
    async def test_get_run_normalizes_legacy_quarantine_failure_counter(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(
            run.id,
            "completed_with_failures",
            landscape_run_id="lscp-legacy-quarantine",
            rows_processed=3,
            rows_succeeded=1,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=2,
        )

        fetched = await service.get_run(run.id)

        assert fetched.rows_failed == 2
        assert fetched.rows_quarantined == 2
        response = self._status_response_from_run(fetched)
        assert response.status == "completed_with_failures"

    @pytest.mark.asyncio
    async def test_get_run_leaves_current_subset_counters_unchanged(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(
            run.id,
            "completed",
            landscape_run_id="lscp-current-gate",
            rows_processed=4,
            rows_succeeded=4,
            rows_failed=0,
            rows_routed_success=2,
            rows_routed_failure=0,
            rows_quarantined=0,
        )

        fetched = await service.get_run(run.id)

        assert fetched.rows_succeeded == 4
        assert fetched.rows_routed_success == 2

    @pytest.mark.asyncio
    async def test_update_not_found_raises(self, service) -> None:
        with pytest.raises(ValueError, match="not found"):
            await service.update_run_status(uuid.uuid4(), "completed")

    @pytest.mark.asyncio
    async def test_completed_requires_landscape_run_id(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        with pytest.raises(ValueError, match="landscape_run_id"):
            await service.update_run_status(run.id, "completed")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["completed", "completed_with_failures", "empty"])
    async def test_operator_completion_status_requires_landscape_run_id(self, service, status) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        with pytest.raises(ValueError, match="landscape_run_id"):
            await service.update_run_status(run.id, status)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["completed_with_failures", "empty"])
    async def test_widened_operator_completion_status_stamps_finished_at(self, service, status) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, status, landscape_run_id=f"lscp-{status}")

        fetched = await service.get_run(run.id)
        assert fetched.status == status
        assert fetched.finished_at is not None
        assert fetched.landscape_run_id == f"lscp-{status}"

    @pytest.mark.asyncio
    async def test_failed_requires_error(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        with pytest.raises(ValueError, match="requires error"):
            await service.update_run_status(run.id, "failed")


class TestRunTransitionEnforcement:
    """Tests for D3 -- LEGAL_RUN_TRANSITIONS enforcement."""

    @pytest.mark.asyncio
    async def test_legal_transition_pending_to_running(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        fetched = await service.get_run(run.id)
        assert fetched.status == "running"

    @pytest.mark.asyncio
    async def test_legal_transition_pending_to_cancelled(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "cancelled")
        fetched = await service.get_run(run.id)
        assert fetched.status == "cancelled"
        assert fetched.finished_at is not None

    @pytest.mark.asyncio
    async def test_illegal_transition_pending_to_completed_raises(
        self,
        service,
    ) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        with pytest.raises(ValueError, match=r"Illegal.*transition"):
            await service.update_run_status(run.id, "completed", landscape_run_id="lscp-illegal")

    @pytest.mark.asyncio
    async def test_illegal_transition_completed_to_running_raises(
        self,
        service,
    ) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "completed", landscape_run_id="lscp-finished")
        with pytest.raises(ValueError, match=r"Illegal.*transition"):
            await service.update_run_status(run.id, "running")


class TestLandscapeRunIdWriteOnce:
    """Tests for D4 -- landscape_run_id is write-once."""

    @pytest.mark.asyncio
    async def test_set_landscape_run_id(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(
            run.id,
            "running",
            landscape_run_id="lscp-001",
        )
        fetched = await service.get_run(run.id)
        assert fetched.landscape_run_id == "lscp-001"

    @pytest.mark.asyncio
    async def test_overwrite_landscape_run_id_raises(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(
            run.id,
            "running",
            landscape_run_id="lscp-001",
        )
        with pytest.raises(ValueError, match=r"landscape_run_id.*already set"):
            await service.update_run_status(
                run.id,
                "completed",
                landscape_run_id="lscp-002",
            )

    @pytest.mark.asyncio
    async def test_none_landscape_run_id_does_not_overwrite(
        self,
        service,
    ) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(
            run.id,
            "running",
            landscape_run_id="lscp-001",
        )
        # Passing None (default) should not trigger the write-once guard
        await service.update_run_status(run.id, "completed")
        fetched = await service.get_run(run.id)
        assert fetched.landscape_run_id == "lscp-001"


class TestCancelOrphanedRuns:
    """Tests for D5 -- cancel_orphaned_runs."""

    @pytest.mark.asyncio
    async def test_cancels_stale_running_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        # Cancel with max_age_seconds=0 so ANY running run is considered stale
        cancelled = await service.cancel_orphaned_runs(
            session.id,
            max_age_seconds=0,
        )
        assert len(cancelled) == 1
        assert cancelled[0].id == run.id
        assert cancelled[0].status == "cancelled"

    @pytest.mark.asyncio
    async def test_does_not_cancel_recent_running_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        # max_age_seconds=3600 -- run was just created, so not stale
        cancelled = await service.cancel_orphaned_runs(
            session.id,
            max_age_seconds=3600,
        )
        assert len(cancelled) == 0

    @pytest.mark.asyncio
    async def test_does_not_cancel_completed_runs(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "completed", landscape_run_id="lscp-orphan-1")
        cancelled = await service.cancel_orphaned_runs(
            session.id,
            max_age_seconds=0,
        )
        assert len(cancelled) == 0

    @pytest.mark.asyncio
    async def test_cancel_unblocks_session_for_new_run(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.cancel_orphaned_runs(session.id, max_age_seconds=0)
        # Session should now accept a new run
        run2 = await service.create_run(session.id, state.id)
        assert run2.status == "pending"

    @pytest.mark.asyncio
    async def test_cancel_includes_pending_orphans(self, service) -> None:
        """A run stuck in 'pending' (crash before transition to running) is also cleaned."""
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        # Create run that stays in pending (simulates crash before running transition)
        await service.create_run(session.id, state.id)
        cancelled = await service.cancel_orphaned_runs(session.id, max_age_seconds=0)
        assert len(cancelled) == 1
        assert cancelled[0].status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_does_not_touch_completed_runs(self, service) -> None:
        """Completed runs are never cancelled regardless of age."""
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "completed", landscape_run_id="lscp-orphan-2")
        cancelled = await service.cancel_orphaned_runs(session.id, max_age_seconds=0)
        assert len(cancelled) == 0


class TestCancelAllOrphanedRuns:
    """Tests for cancel_all_orphaned_runs (global startup cleanup)."""

    @pytest.mark.asyncio
    async def test_cancels_all_non_terminal_runs_without_age_filter(self, service) -> None:
        """Default (max_age_seconds=None) cancels ALL pending/running runs,
        not just old ones. Critical for single-process server restarts."""
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        # Create a fresh run (just created, zero age)
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")

        # No age filter — should cancel even a brand-new run
        cancelled = await service.cancel_all_orphaned_runs()
        assert cancelled == 1

        updated = await service.get_run(run.id)
        assert updated.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancels_pending_runs_without_age_filter(self, service) -> None:
        """Pending runs (never transitioned to running) are also cancelled."""
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        await service.create_run(session.id, state.id)

        cancelled = await service.cancel_all_orphaned_runs()
        assert cancelled == 1

    @pytest.mark.asyncio
    async def test_does_not_cancel_terminal_runs(self, service) -> None:
        """Completed/cancelled/failed runs are never touched."""
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")
        await service.update_run_status(run.id, "completed", landscape_run_id="lscp-global-1")

        cancelled = await service.cancel_all_orphaned_runs()
        assert cancelled == 0

    @pytest.mark.asyncio
    async def test_age_filter_still_works_when_provided(self, service) -> None:
        """When max_age_seconds is given, only old runs are cancelled."""
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")

        # Run was just created — 3600s filter should skip it
        cancelled = await service.cancel_all_orphaned_runs(max_age_seconds=3600)
        assert cancelled == 0

    @pytest.mark.asyncio
    async def test_unblocks_session_after_cancellation(self, service) -> None:
        """After cancelling orphaned runs, session can accept new runs."""
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        await service.create_run(session.id, state.id)

        await service.cancel_all_orphaned_runs()

        # Session should now be unblocked
        run2 = await service.create_run(session.id, state.id)
        assert run2.status == "pending"


class TestCancelAllOrphanedRunsExcludeRunIds:
    """Tests for exclude_run_ids — liveness-aware orphan cleanup."""

    @pytest.mark.asyncio
    async def test_excludes_live_run_ids_from_cancellation(self, service) -> None:
        """Runs with IDs in exclude_run_ids are skipped even if they exceed max_age."""
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")

        # Exclude this run's ID — it should NOT be cancelled
        cancelled = await service.cancel_all_orphaned_runs(
            max_age_seconds=0,
            exclude_run_ids=frozenset({str(run.id)}),
        )
        assert cancelled == 0

        # Run should still be running
        fetched = await service.get_run(run.id)
        assert fetched.status == "running"

    @pytest.mark.asyncio
    async def test_cancels_non_excluded_runs(self, service) -> None:
        """Runs NOT in exclude_run_ids are still cancelled normally."""
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")

        # Exclude a different run ID — this run should be cancelled
        cancelled = await service.cancel_all_orphaned_runs(
            max_age_seconds=0,
            exclude_run_ids=frozenset({"not-this-run-id"}),
        )
        assert cancelled == 1

        fetched = await service.get_run(run.id)
        assert fetched.status == "cancelled"

    @pytest.mark.asyncio
    async def test_empty_exclude_set_cancels_all(self, service) -> None:
        """Empty exclude_run_ids (default) does not change behaviour."""
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")

        cancelled = await service.cancel_all_orphaned_runs(
            max_age_seconds=0,
            exclude_run_ids=frozenset(),
        )
        assert cancelled == 1


class TestCancelAllOrphanedRunsReason:
    """Tests for reason parameter — error provenance on orphan cancellation."""

    @pytest.mark.asyncio
    async def test_reason_written_to_error_column(self, service) -> None:
        """When reason is provided, it's stored in the run's error field."""
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")

        await service.cancel_all_orphaned_runs(
            max_age_seconds=0,
            reason="Orphaned by server restart — no active process",
        )

        fetched = await service.get_run(run.id)
        assert fetched.status == "cancelled"
        assert fetched.error == "Orphaned by server restart — no active process"

    @pytest.mark.asyncio
    async def test_no_reason_leaves_error_null(self, service) -> None:
        """When reason is None (default), error field stays unset."""
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "running")

        await service.cancel_all_orphaned_runs(max_age_seconds=0)

        fetched = await service.get_run(run.id)
        assert fetched.status == "cancelled"
        assert fetched.error is None


class TestCancelledTerminalTransitions:
    """Tests for cancelled as a terminal state — no outgoing transitions.

    These transitions are the exact paths triggered when the orphan cleanup
    cancels a run in the DB while the executor thread is still running.
    The executor then tries cancelled→completed or cancelled→failed,
    both of which must be rejected.
    """

    @pytest.mark.asyncio
    async def test_illegal_transition_cancelled_to_completed_raises(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "cancelled")
        with pytest.raises(ValueError, match=r"Illegal.*transition"):
            await service.update_run_status(run.id, "completed", landscape_run_id="lscp-cancelled")

    @pytest.mark.asyncio
    async def test_illegal_transition_cancelled_to_failed_raises(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        run = await service.create_run(session.id, state.id)
        await service.update_run_status(run.id, "cancelled")
        with pytest.raises(ValueError, match=r"Illegal.*transition"):
            await service.update_run_status(run.id, "failed", error="boom")


class TestArchiveSessionWithActiveRun:
    """Tests for archive_session when a run is active."""

    @pytest.mark.asyncio
    async def test_archive_soft_hides_session_with_active_run(self, service) -> None:
        """A session with a durable run is soft-archived, not deleted.

        Commit 4c3e81182 ("Polish RC5 composer UX and archive behavior")
        defined the contract: ``archive_session`` physically deletes
        sessions with no durable history (no runs, no composer
        completion events) and soft-hides sessions that have either.
        An active run counts as durable history — the row remains, an
        ``archived_at`` timestamp is set, and the session is hidden
        from the default list but visible when ``include_archived``
        is requested. Preserving the row keeps the run's audit
        lineage queryable.
        """
        session = await service.create_session("alice", "Pipeline", "local")
        state = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        await service.create_run(session.id, state.id)

        await service.archive_session(session.id)

        archived = await service.get_session(session.id)
        assert archived.archived_at is not None, "Soft-archive should populate archived_at, not delete the row"

        default_listing = await service.list_sessions("alice", "local")
        assert session.id not in [s.id for s in default_listing], "Soft-archived session must be hidden from default listing"

        with_archived = await service.list_sessions("alice", "local", include_archived=True)
        assert session.id in [s.id for s in with_archived], "Soft-archived session must be retrievable via include_archived"


class TestGetMessagesNonexistentSession:
    """Tests for get_messages behavior with a nonexistent session."""

    @pytest.mark.asyncio
    async def test_get_messages_returns_empty_for_nonexistent_session(self, service) -> None:
        """get_messages silently returns [] for a nonexistent session_id.

        This is by design — the WHERE clause filters by session_id and
        returns no rows. Callers (routes) should verify session existence
        via _verify_session_ownership before calling get_messages.
        """
        import uuid

        result = await service.get_messages(uuid.uuid4())
        assert result == []


class TestPagination:
    """Tests for limit/offset pagination on list endpoints."""

    @pytest.mark.asyncio
    async def test_list_sessions_limit(self, service) -> None:
        for i in range(5):
            await service.create_session("alice", f"Session {i}", "local")
        sessions = await service.list_sessions("alice", "local", limit=2)
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_list_sessions_offset(self, service) -> None:
        for i in range(5):
            await service.create_session("alice", f"Session {i}", "local")
        all_sessions = await service.list_sessions("alice", "local")
        offset_sessions = await service.list_sessions("alice", "local", limit=2, offset=2)
        assert len(offset_sessions) == 2
        assert offset_sessions[0].id == all_sessions[2].id
        assert offset_sessions[1].id == all_sessions[3].id

    @pytest.mark.asyncio
    async def test_list_sessions_offset_past_end(self, service) -> None:
        await service.create_session("alice", "Only One", "local")
        sessions = await service.list_sessions("alice", "local", offset=10)
        assert sessions == []

    @pytest.mark.asyncio
    async def test_get_messages_limit(self, service) -> None:
        session = await service.create_session("alice", "Chat", "local")
        for i in range(5):
            await service.add_message(session.id, "user", f"Message {i}", writer_principal="route_user_message")
        messages = await service.get_messages(session.id, limit=3)
        assert len(messages) == 3
        assert messages[0].content == "Message 0"

    @pytest.mark.asyncio
    async def test_get_messages_offset(self, service) -> None:
        session = await service.create_session("alice", "Chat", "local")
        for i in range(5):
            await service.add_message(session.id, "user", f"Message {i}", writer_principal="route_user_message")
        messages = await service.get_messages(session.id, limit=2, offset=3)
        assert len(messages) == 2
        assert messages[0].content == "Message 3"
        assert messages[1].content == "Message 4"

    @pytest.mark.asyncio
    async def test_get_state_versions_limit(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        for _ in range(5):
            await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")
        versions = await service.get_state_versions(session.id, limit=2)
        assert len(versions) == 2
        assert versions[0].version == 1
        assert versions[1].version == 2

    @pytest.mark.asyncio
    async def test_get_state_versions_offset(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        for _ in range(5):
            await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")
        versions = await service.get_state_versions(session.id, limit=2, offset=3)
        assert len(versions) == 2
        assert versions[0].version == 4
        assert versions[1].version == 5


class TestPruneStateVersions:
    """Tests for prune_state_versions -- delete old versions, preserve recent and run-referenced."""

    @pytest.mark.asyncio
    async def test_prune_deletes_old_versions(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        for _ in range(5):
            await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")

        deleted = await service.prune_state_versions(session.id, keep_latest=2)
        assert deleted == 3

        remaining = await service.get_state_versions(session.id)
        assert len(remaining) == 2
        assert [v.version for v in remaining] == [4, 5]

    @pytest.mark.asyncio
    async def test_prune_preserves_run_referenced_versions(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        v1 = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")
        await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")

        # Create a run referencing v1
        await service.create_run(session.id, v1.id)

        # Prune keeping only latest 1 -- v1 should survive (run-referenced), v2 deleted
        deleted = await service.prune_state_versions(session.id, keep_latest=1)
        assert deleted == 1  # only v2 deleted

        remaining = await service.get_state_versions(session.id)
        remaining_versions = [v.version for v in remaining]
        assert 1 in remaining_versions  # preserved by run reference
        assert 2 not in remaining_versions  # deleted
        assert 3 in remaining_versions  # kept as latest

    @pytest.mark.asyncio
    async def test_prune_returns_zero_when_nothing_to_prune(self, service) -> None:
        session = await service.create_session("alice", "Pipeline", "local")
        for _ in range(2):
            await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")

        deleted = await service.prune_state_versions(session.id, keep_latest=5)
        assert deleted == 0

    @pytest.mark.asyncio
    async def test_prune_preserves_derived_from_lineage(self, service) -> None:
        """States referenced via derived_from_state_id must survive pruning.

        Scenario: v1 (normal), v2 (normal), v3 (revert to v1).
        Prune with keep_latest=1 keeps v3 (latest).  v1 must survive
        because v3.derived_from_state_id points at it.  v2 can be deleted.
        """
        session = await service.create_session("alice", "Pipeline", "local")
        v1 = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")
        # Revert to v1 — creates v3 with derived_from_state_id = v1.id
        v3 = await service.set_active_state(session.id, v1.id)
        assert v3.derived_from_state_id == v1.id

        deleted = await service.prune_state_versions(session.id, keep_latest=1)
        assert deleted == 1  # only v2 deleted

        remaining = await service.get_state_versions(session.id)
        remaining_ids = {v.id for v in remaining}
        assert v1.id in remaining_ids, "v1 must survive — referenced by v3.derived_from_state_id"
        assert v3.id in remaining_ids, "v3 must survive — it is the latest version"

    @pytest.mark.asyncio
    async def test_prune_preserves_transitive_derived_lineage(self, service) -> None:
        """Transitive derived_from chains must be fully preserved.

        Scenario: v1, v2, v3 (revert→v1), v4, v5 (revert→v3).
        Prune with keep_latest=1 keeps v5.  v3 must survive (v5 points
        at it), and v1 must survive (v3 points at it).  v2 and v4 can go.
        """
        session = await service.create_session("alice", "Pipeline", "local")
        v1 = await service.save_composition_state(session.id, CompositionStateData(is_valid=True), provenance="session_seed")
        await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")
        # v3: revert to v1
        v3 = await service.set_active_state(session.id, v1.id)
        await service.save_composition_state(session.id, CompositionStateData(is_valid=False), provenance="session_seed")
        # v5: revert to v3
        v5 = await service.set_active_state(session.id, v3.id)

        deleted = await service.prune_state_versions(session.id, keep_latest=1)
        assert deleted == 2  # v2 and v4 deleted

        remaining = await service.get_state_versions(session.id)
        remaining_ids = {v.id for v in remaining}
        assert v1.id in remaining_ids, "v1 must survive — v3.derived_from_state_id"
        assert v3.id in remaining_ids, "v3 must survive — v5.derived_from_state_id"
        assert v5.id in remaining_ids, "v5 must survive — latest version"
