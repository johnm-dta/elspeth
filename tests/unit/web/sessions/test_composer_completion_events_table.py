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
    composition_states_table,
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


def test_session_schema_epoch_is_post_phase_6a_completion() -> None:
    """SESSION_SCHEMA_EPOCH must be at least the Phase-6A bump.

    Per project_db_migration_policy, bumping the epoch is the mechanical signal
    that triggers the operator's DB-delete action on next deploy. The Phase-6A
    cohort took the epoch to 4 (table added); a Phase-6A follow-up cohort took
    it to 5 (per-event-type partial CHECKs added — SQLite cannot ALTER
    constraints, so the constraint change is a fresh-schema bump).
    """

    assert SESSION_SCHEMA_EPOCH >= 4


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


def _seed_composition_state(
    conn,
    state_id: str = "cs1",
    session_id: str = "s1",
) -> None:
    """Insert a minimal valid composition_states row.

    Required by the per-event-type CHECK constraints on
    ``composer_completion_events`` — every event row references a
    composition state, and the schema-level NOT NULL is enforced by
    ``ck_composer_completion_events_composition_state_id_required``.
    """
    conn.execute(
        insert(composition_states_table).values(
            id=state_id,
            session_id=session_id,
            version=1,
            is_valid=True,
            created_at=datetime.now(UTC),
            provenance="tool_call",
        )
    )


def test_check_constraint_rejects_invalid_event_type(engine) -> None:
    with engine.connect() as conn:
        _insert_session(conn)
        _seed_composition_state(conn)
        with pytest.raises(IntegrityError):
            conn.execute(
                insert(composer_completion_events_table).values(
                    id="e1",
                    session_id="s1",
                    composition_state_id="cs1",
                    event_type="invalid_type",
                    actor="user1",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()


def test_check_constraint_accepts_mark_ready_for_review(engine) -> None:
    with engine.connect() as conn:
        _insert_session(conn)
        _seed_composition_state(conn)
        conn.execute(
            insert(composer_completion_events_table).values(
                id="e1",
                session_id="s1",
                composition_state_id="cs1",
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
        _seed_composition_state(conn)
        conn.execute(
            insert(composer_completion_events_table).values(
                id="e2",
                session_id="s1",
                composition_state_id="cs1",
                event_type="export_yaml",
                actor="user1",
                created_at=datetime.now(UTC),
            )
        )
        conn.commit()


def test_completion_event_state_must_belong_to_same_session(engine) -> None:
    """The audit row's state/session pair is one composite identity."""

    with engine.connect() as conn:
        _insert_session(conn, session_id="s1")
        _insert_session(conn, session_id="s2")
        _seed_composition_state(conn, state_id="cs1", session_id="s1")
        with pytest.raises(IntegrityError):
            conn.execute(
                insert(composer_completion_events_table).values(
                    id="e-cross-session",
                    session_id="s2",
                    composition_state_id="cs1",
                    event_type="export_yaml",
                    actor="user1",
                    created_at=datetime.now(UTC),
                )
            )
            conn.commit()


# ---- per-event-type partial CHECK constraints ----


def test_mark_ready_for_review_requires_payload_digest(engine) -> None:
    """mark_ready_for_review without payload_digest must fail the partial CHECK."""

    with engine.connect() as conn:
        _insert_session(conn)
        _seed_composition_state(conn)
        with pytest.raises(IntegrityError):
            conn.execute(
                insert(composer_completion_events_table).values(
                    id="e1",
                    session_id="s1",
                    composition_state_id="cs1",
                    event_type="mark_ready_for_review",
                    actor="user1",
                    created_at=datetime.now(UTC),
                    payload_digest=None,
                    expires_at=datetime.now(UTC),
                )
            )
            conn.commit()


def test_mark_ready_for_review_requires_expires_at(engine) -> None:
    """mark_ready_for_review without expires_at must fail the partial CHECK."""

    with engine.connect() as conn:
        _insert_session(conn)
        _seed_composition_state(conn)
        with pytest.raises(IntegrityError):
            conn.execute(
                insert(composer_completion_events_table).values(
                    id="e1",
                    session_id="s1",
                    composition_state_id="cs1",
                    event_type="mark_ready_for_review",
                    actor="user1",
                    created_at=datetime.now(UTC),
                    payload_digest="sha256:" + ("ab" * 32),
                    expires_at=None,
                )
            )
            conn.commit()


def test_export_yaml_rejects_payload_digest(engine) -> None:
    """export_yaml carrying a payload_digest must fail — payload_digest is
    a mark_ready_for_review-specific column."""

    with engine.connect() as conn:
        _insert_session(conn)
        _seed_composition_state(conn)
        with pytest.raises(IntegrityError):
            conn.execute(
                insert(composer_completion_events_table).values(
                    id="e1",
                    session_id="s1",
                    composition_state_id="cs1",
                    event_type="export_yaml",
                    actor="user1",
                    created_at=datetime.now(UTC),
                    payload_digest="sha256:" + ("ab" * 32),
                    expires_at=None,
                )
            )
            conn.commit()


def test_export_yaml_rejects_expires_at(engine) -> None:
    """export_yaml carrying an expires_at must fail — expires_at is a
    mark_ready_for_review-specific column."""

    with engine.connect() as conn:
        _insert_session(conn)
        _seed_composition_state(conn)
        with pytest.raises(IntegrityError):
            conn.execute(
                insert(composer_completion_events_table).values(
                    id="e1",
                    session_id="s1",
                    composition_state_id="cs1",
                    event_type="export_yaml",
                    actor="user1",
                    created_at=datetime.now(UTC),
                    payload_digest=None,
                    expires_at=datetime.now(UTC),
                )
            )
            conn.commit()


def test_composition_state_id_required(engine) -> None:
    """composition_state_id is NOT NULL for both event types."""

    with engine.connect() as conn:
        _insert_session(conn)
        with pytest.raises(IntegrityError):
            conn.execute(
                insert(composer_completion_events_table).values(
                    id="e1",
                    session_id="s1",
                    composition_state_id=None,
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
        _seed_composition_state(conn)
        conn.execute(
            insert(composer_completion_events_table).values(
                id="e1",
                session_id="s1",
                composition_state_id="cs1",
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

    Phase 6 ships both UPDATE and DELETE triggers from day 1.
    """

    with engine.connect() as conn:
        _insert_session(conn)
        _seed_composition_state(conn)
        conn.execute(
            insert(composer_completion_events_table).values(
                id="e1",
                session_id="s1",
                composition_state_id="cs1",
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

    User-facing archive must soft-hide sessions with durable completion
    history. Raw parent-row deletion remains blocked because it would purge
    audit facts along with the session row.
    """

    with engine.connect() as conn:
        _insert_session(conn)
        _seed_composition_state(conn)
        conn.execute(
            insert(composer_completion_events_table).values(
                id="e1",
                session_id="s1",
                composition_state_id="cs1",
                event_type="export_yaml",
                actor="user1",
                created_at=datetime.now(UTC),
            )
        )
        conn.commit()
        with pytest.raises((OperationalError, IntegrityError), match="append-only"):
            conn.execute(text("DELETE FROM sessions WHERE id = 's1'"))
            conn.commit()


# ---- triggers registered in _REQUIRED_AUDIT_TRIGGERS ----


def test_triggers_registered_in_required_set() -> None:
    """Startup validator must enforce trigger presence on existing DBs."""
    from elspeth.web.sessions.schema import _REQUIRED_AUDIT_TRIGGERS

    assert "trg_composer_completion_events_no_update" in _REQUIRED_AUDIT_TRIGGERS
    assert "trg_composer_completion_events_no_delete" in _REQUIRED_AUDIT_TRIGGERS


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
