"""Schema tests for ``composer_completion_events`` table.

Phase 6A (UX redesign 2026-05) — completion-gesture audit table.

Tests follow the spec at
``docs/composer/ux-redesign-2026-05/19a-phase-6a-backend.md`` (Task 1).

The table is append-only from day 1: both BEFORE UPDATE and BEFORE DELETE
triggers unconditionally ABORT. Unlike ``interpretation_events_table`` —
which permits DELETE on PENDING rows for orphan recovery — completion
events have no recovery path; both triggers are unconditional ABORT,
correcting the Phase 18 omission tracked at filigree elspeth-9aba8da942.

Tests use ``create_session_engine`` + ``initialize_session_schema`` so the
full production bootstrap (PRAGMAs, trigger DDL, schema validator) is
exercised end-to-end, matching ``test_interpretation_events_table.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import insert, inspect, text
from sqlalchemy.exc import IntegrityError, OperationalError

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import (
    SESSION_SCHEMA_EPOCH,
    composer_completion_events_table,
    metadata,
    sessions_table,
)
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture
def engine():
    """In-memory engine wired through the production bootstrap path."""
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    return eng


# ---- schema-presence / structural tests ----


def test_table_exists_in_metadata() -> None:
    assert "composer_completion_events" in metadata.tables


def test_table_present_in_engine(engine) -> None:
    inspector = inspect(engine)
    assert "composer_completion_events" in inspector.get_table_names()


def test_session_schema_epoch_bumped_to_4() -> None:
    """Phase 6 schema-change cohort: SESSION_SCHEMA_EPOCH must bump from 3 to 4.

    Per project_db_migration_policy, bumping the epoch is the mechanical signal
    that triggers the operator's DB-delete action on next deploy. Without the
    bump, live deployments crash mid-request on first INSERT — see filigree
    finding elspeth-c03e9bfcf8 for the analogous Landscape-DB defect.
    """

    assert SESSION_SCHEMA_EPOCH == 4


def test_user_version_stamped_with_new_epoch(engine) -> None:
    """initialize_session_schema must stamp PRAGMA user_version = 4 on fresh DBs."""

    with engine.connect() as conn:
        version = conn.execute(text("PRAGMA user_version")).scalar_one()
    assert version == SESSION_SCHEMA_EPOCH


# ---- CHECK constraint on event_type ----


def _insert_session(conn, session_id: str = "s1", user_id: str = "user1") -> None:
    now = datetime.now(UTC)
    conn.execute(
        insert(sessions_table).values(
            id=session_id,
            user_id=user_id,
            auth_provider_type="local",
            title="t",
            trust_mode="auto_commit",
            density_default="high",
            created_at=now,
            updated_at=now,
            interpretation_review_disabled=False,
        )
    )


def test_check_constraint_rejects_invalid_event_type(engine) -> None:
    with engine.connect() as conn:
        _insert_session(conn)
        with pytest.raises(IntegrityError):
            conn.execute(
                insert(composer_completion_events_table).values(
                    id="e1",
                    session_id="s1",
                    event_type="invalid_type",
                    actor="user1",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()


def test_check_constraint_accepts_mark_ready_for_review(engine) -> None:
    with engine.connect() as conn:
        _insert_session(conn)
        conn.execute(
            insert(composer_completion_events_table).values(
                id="e1",
                session_id="s1",
                event_type="mark_ready_for_review",
                actor="user1",
                created_at=datetime.now(UTC),
                payload_digest="sha256:" + ("ab" * 32),
                expires_at=datetime.now(UTC),
            )
        )
        conn.commit()


def test_check_constraint_accepts_export_yaml(engine) -> None:
    with engine.connect() as conn:
        _insert_session(conn)
        conn.execute(
            insert(composer_completion_events_table).values(
                id="e2",
                session_id="s1",
                event_type="export_yaml",
                actor="user1",
                created_at=datetime.now(UTC),
            )
        )
        conn.commit()


# ---- append-only triggers ----


def test_update_trigger_blocks_mutation(engine) -> None:
    """Audit table is append-only — UPDATE must raise unconditionally."""

    with engine.connect() as conn:
        _insert_session(conn)
        conn.execute(
            insert(composer_completion_events_table).values(
                id="e1",
                session_id="s1",
                event_type="export_yaml",
                actor="user1",
                created_at=datetime.now(UTC),
            )
        )
        conn.commit()
        with pytest.raises((OperationalError, IntegrityError), match="append-only"):
            conn.execute(text("UPDATE composer_completion_events SET actor = 'attacker' WHERE id = 'e1'"))
            conn.commit()


def test_delete_trigger_blocks_removal(engine) -> None:
    """Audit table is append-only — DELETE must raise unconditionally.

    Phase 6 ships both UPDATE and DELETE triggers from day 1, correcting the
    Phase 18 omission tracked at filigree elspeth-9aba8da942.
    """

    with engine.connect() as conn:
        _insert_session(conn)
        conn.execute(
            insert(composer_completion_events_table).values(
                id="e1",
                session_id="s1",
                event_type="export_yaml",
                actor="user1",
                created_at=datetime.now(UTC),
            )
        )
        conn.commit()
        with pytest.raises((OperationalError, IntegrityError), match="append-only"):
            conn.execute(text("DELETE FROM composer_completion_events WHERE id = 'e1'"))
            conn.commit()


def test_session_fk_cascade_is_blocked_by_append_only_trigger(engine) -> None:
    """FK cascade from sessions → composer_completion_events is blocked by the trigger.

    When a parent sessions row is deleted with FK cascades enabled, SQLite attempts to
    cascade-delete the dependent composer_completion_events rows. The BEFORE DELETE
    trigger fires before the cascade can complete and raises ABORT, which rolls back
    the parent DELETE as well. This confirms completion events are permanently
    retained: a session cannot be deleted while it has completion-event children.
    """

    with engine.connect() as conn:
        _insert_session(conn)
        conn.execute(
            insert(composer_completion_events_table).values(
                id="e1",
                session_id="s1",
                event_type="export_yaml",
                actor="user1",
                created_at=datetime.now(UTC),
            )
        )
        conn.commit()
        with pytest.raises((OperationalError, IntegrityError), match="append-only"):
            conn.execute(text("DELETE FROM sessions WHERE id = 's1'"))
            conn.commit()


# ---- triggers registered in _REQUIRED_SQLITE_TRIGGERS ----


def test_triggers_registered_in_required_set() -> None:
    """Startup validator must enforce trigger presence on existing DBs."""
    from elspeth.web.sessions.schema import _REQUIRED_SQLITE_TRIGGERS

    assert "trg_composer_completion_events_no_update" in _REQUIRED_SQLITE_TRIGGERS
    assert "trg_composer_completion_events_no_delete" in _REQUIRED_SQLITE_TRIGGERS


def test_triggers_present_in_live_database(engine) -> None:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'trg_composer_completion_events%'")
        ).all()
    names = {row[0] for row in rows}
    assert names == {
        "trg_composer_completion_events_no_update",
        "trg_composer_completion_events_no_delete",
    }


# ---- expected indexes ----


def test_session_created_index_exists(engine) -> None:
    inspector = inspect(engine)
    indexes = {idx["name"] for idx in inspector.get_indexes("composer_completion_events")}
    assert "ix_composer_completion_events_session_created" in indexes
