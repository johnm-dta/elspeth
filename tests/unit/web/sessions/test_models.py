"""Tests for SQLAlchemy session table definitions."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import insert, inspect, select, text
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    chat_messages_table,
    composition_states_table,
    run_events_table,
    runs_table,
    sessions_table,
    user_secrets_table,
)
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine migrated to head."""
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    return eng


class TestTableCreation:
    """Verify the session-database tables are created with correct schemas."""

    def test_all_tables_exist(self, engine) -> None:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        assert "sessions" in table_names
        assert "chat_messages" in table_names
        assert "composition_states" in table_names
        assert "runs" in table_names

    def test_sessions_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("sessions")}
        assert columns >= {"id", "user_id", "auth_provider_type", "title", "created_at", "updated_at"}

    def test_chat_messages_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("chat_messages")}
        assert columns >= {
            "id",
            "session_id",
            "role",
            "content",
            "tool_calls",
            "created_at",
        }

    def test_composition_states_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("composition_states")}
        assert columns >= {
            "id",
            "session_id",
            "version",
            "source",
            "nodes",
            "edges",
            "outputs",
            "metadata_",
            "is_valid",
            "validation_errors",
            "created_at",
            "derived_from_state_id",
        }

    def test_runs_columns(self, engine) -> None:
        inspector = inspect(engine)
        columns = {c["name"] for c in inspector.get_columns("runs")}
        assert columns >= {
            "id",
            "session_id",
            "state_id",
            "status",
            "started_at",
            "finished_at",
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "rows_routed_success",
            "rows_routed_failure",
            "rows_quarantined",
            "error",
            "landscape_run_id",
            "pipeline_yaml",
        }


class TestCompositionStateUniqueConstraint:
    """Verify the UNIQUE(session_id, version) constraint."""

    def test_duplicate_version_raises(self, engine) -> None:
        session_id = str(uuid.uuid4())
        state_id_1 = str(uuid.uuid4())
        state_id_2 = str(uuid.uuid4())

        with engine.begin() as conn:
            # Insert a session first (FK constraint)
            conn.execute(
                insert(sessions_table).values(
                    id=session_id,
                    user_id="alice",
                    title="Test",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
            # First state version
            conn.execute(
                insert(composition_states_table).values(
                    id=state_id_1,
                    session_id=session_id,
                    version=1,
                    is_valid=False,
                    # Plan §2294: schema-test direct insert must supply
                    # provenance after Task 3's NOT NULL/CHECK addition.
                    provenance="session_seed",
                    created_at=datetime.now(UTC),
                )
            )
            # Duplicate version should fail
            with pytest.raises(IntegrityError):
                conn.execute(
                    insert(composition_states_table).values(
                        id=state_id_2,
                        session_id=session_id,
                        version=1,
                        is_valid=False,
                        # Plan §2294: provenance required even on the
                        # expected-to-fail row, so the failure exercises
                        # ``uq_composition_state_version`` (the actual
                        # subject under test) rather than the provenance
                        # NOT NULL constraint.
                        provenance="session_seed",
                        created_at=datetime.now(UTC),
                    )
                )


class TestSessionForeignKeys:
    """Verify foreign key relationships."""

    def test_chat_message_requires_valid_session(self, engine) -> None:
        """Inserting a message with a nonexistent session_id should fail
        if FK enforcement is on (SQLite needs PRAGMA foreign_keys=ON)."""
        # SQLite does not enforce FK by default; this test verifies
        # the column exists and accepts valid references.
        session_id = str(uuid.uuid4())
        msg_id = str(uuid.uuid4())

        with engine.begin() as conn:
            # Enable FK enforcement for SQLite
            conn.execute(
                insert(sessions_table).values(
                    id=session_id,
                    user_id="alice",
                    title="Test",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
            conn.execute(
                insert(chat_messages_table).values(
                    id=msg_id,
                    session_id=session_id,
                    role="user",
                    content="Hello",
                    sequence_no=1,
                    writer_principal="route_user_message",
                    created_at=datetime.now(UTC),
                )
            )
            # Verify it was inserted
            result = conn.execute(select(chat_messages_table).where(chat_messages_table.c.id == msg_id)).fetchone()
            assert result is not None

    def test_orphan_message_rejected_with_fk_enforcement(self, engine) -> None:
        """With PRAGMA foreign_keys=ON, orphan messages are rejected."""
        with engine.begin() as conn:
            conn.execute(text("PRAGMA foreign_keys=ON"))
            with pytest.raises(IntegrityError):
                conn.execute(
                    insert(chat_messages_table).values(
                        id=str(uuid.uuid4()),
                        session_id="nonexistent-session",
                        role="user",
                        content="Orphan message",
                        sequence_no=1,
                        writer_principal="route_user_message",
                        created_at=datetime.now(UTC),
                    )
                )

    def test_forked_from_session_id_is_historical_without_live_session_fk(self, engine) -> None:
        """Verify that sessions.forked_from_session_id has no ForeignKey constraint."""
        inspector = inspect(engine)
        foreign_keys = inspector.get_foreign_keys("sessions")
        constrained_columns = {column for foreign_key in foreign_keys for column in foreign_key["constrained_columns"]}
        assert "forked_from_session_id" not in constrained_columns

    def test_forked_from_message_id_has_set_null_chat_message_fk(self, engine) -> None:
        """Fork message provenance is cleared if the source chat message is deleted."""
        inspector = inspect(engine)
        foreign_keys = inspector.get_foreign_keys("sessions")

        matching = [foreign_key for foreign_key in foreign_keys if tuple(foreign_key["constrained_columns"]) == ("forked_from_message_id",)]

        assert len(matching) == 1
        foreign_key = matching[0]
        assert foreign_key["name"] == "fk_sessions_forked_from_message"
        assert foreign_key["referred_table"] == "chat_messages"
        assert tuple(foreign_key["referred_columns"]) == ("id",)
        assert str(foreign_key["options"]["ondelete"]).lower() == "set null"

    def test_orphan_forked_from_message_id_rejected_with_fk_enforcement(self, engine) -> None:
        """Child sessions cannot point fork provenance at a nonexistent message."""
        with engine.begin() as conn:
            conn.execute(text("PRAGMA foreign_keys=ON"))
            with pytest.raises(IntegrityError):
                conn.execute(
                    insert(sessions_table).values(
                        id=str(uuid.uuid4()),
                        user_id="alice",
                        title="Fork with missing message",
                        forked_from_message_id=str(uuid.uuid4()),
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                    )
                )


class TestCheckConstraints:
    """Verify CHECK constraints reject invalid values."""

    def test_auth_provider_type_constraints_exist(self, engine) -> None:
        inspector = inspect(engine)
        session_checks = {check["name"] for check in inspector.get_check_constraints("sessions")}
        user_secret_checks = {check["name"] for check in inspector.get_check_constraints("user_secrets")}

        assert "ck_sessions_auth_provider_type" in session_checks
        assert "ck_user_secrets_auth_provider_type" in user_secret_checks

    def test_invalid_session_auth_provider_type_rejected(self, engine) -> None:
        with engine.begin() as conn, pytest.raises(IntegrityError):
            conn.execute(
                insert(sessions_table).values(
                    id=str(uuid.uuid4()),
                    user_id="alice",
                    auth_provider_type="saml",
                    title="Test",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )

    def test_invalid_user_secret_auth_provider_type_rejected(self, engine) -> None:
        with engine.begin() as conn, pytest.raises(IntegrityError):
            conn.execute(
                insert(user_secrets_table).values(
                    id=str(uuid.uuid4()),
                    name="api-key",
                    user_id="alice",
                    auth_provider_type="saml",
                    encrypted_value=b"ciphertext",
                    salt=b"salt",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )

    def test_invalid_chat_message_role_rejected(self, engine) -> None:
        session_id = str(uuid.uuid4())
        with engine.begin() as conn:
            conn.execute(
                insert(sessions_table).values(
                    id=session_id,
                    user_id="alice",
                    title="Test",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
            with pytest.raises(IntegrityError):
                conn.execute(
                    insert(chat_messages_table).values(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        role="invalid_role",
                        content="Hello",
                        sequence_no=1,
                        writer_principal="route_user_message",
                        created_at=datetime.now(UTC),
                    )
                )

    def test_invalid_run_status_rejected(self, engine) -> None:
        session_id = str(uuid.uuid4())
        state_id = str(uuid.uuid4())
        with engine.begin() as conn:
            conn.execute(
                insert(sessions_table).values(
                    id=session_id,
                    user_id="alice",
                    title="Test",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
            conn.execute(
                insert(composition_states_table).values(
                    id=state_id,
                    session_id=session_id,
                    version=1,
                    is_valid=True,
                    # Plan §2294: schema-test direct insert; provenance
                    # required for the FK-target row that subsequent
                    # runs_table inserts depend on.
                    provenance="session_seed",
                    created_at=datetime.now(UTC),
                )
            )
            with pytest.raises(IntegrityError):
                conn.execute(
                    insert(runs_table).values(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        state_id=state_id,
                        status="invalid_status",
                        started_at=datetime.now(UTC),
                        rows_processed=0,
                        rows_failed=0,
                    )
                )

    def test_invalid_run_event_type_rejected(self, engine) -> None:
        session_id = str(uuid.uuid4())
        state_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        with engine.begin() as conn:
            conn.execute(
                insert(sessions_table).values(
                    id=session_id,
                    user_id="alice",
                    title="Test",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
            )
            conn.execute(
                insert(composition_states_table).values(
                    id=state_id,
                    session_id=session_id,
                    version=1,
                    is_valid=True,
                    # Plan §2294: schema-test direct insert; provenance
                    # required for the FK-target row that subsequent
                    # runs_table inserts depend on.
                    provenance="session_seed",
                    created_at=datetime.now(UTC),
                )
            )
            conn.execute(
                insert(runs_table).values(
                    id=run_id,
                    session_id=session_id,
                    state_id=state_id,
                    status="pending",
                    started_at=datetime.now(UTC),
                    rows_processed=0,
                    rows_failed=0,
                )
            )
            with pytest.raises(IntegrityError):
                conn.execute(
                    insert(run_events_table).values(
                        id=str(uuid.uuid4()),
                        run_id=run_id,
                        sequence=1,
                        timestamp=datetime.now(UTC),
                        event_type="invalid_type",
                        data="{}",
                    )
                )
