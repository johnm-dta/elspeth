"""Schema tests for ``interpretation_events`` table and its dependencies.

Phase 5b Task 2 — covers the new table, the
``composition_states.provenance`` closed-enum extension, the
``sessions.interpretation_review_disabled`` column, the two BEFORE UPDATE
triggers (``trg_interpretation_events_immutable_resolved`` and
``trg_chat_messages_immutable_content``), the
``calls.resolved_prompt_template_hash`` column added to the L1 Landscape,
the partial unique index on pending tool calls, and the F-11 lookup index
on ``composition_state_id``.

Tests follow the spec at
``docs/composer/ux-redesign-2026-05/18a-phase-5b-backend.md`` (Task 2,
Step "Test shape"). Test numbering mirrors the spec for traceability.
"""

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
    interpretation_events_table,
    sessions_table,
    skill_markdown_history_table,
)
from elspeth.web.sessions.schema import initialize_session_schema


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine initialised via the production bootstrap path."""
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    return eng


def _insert_session(conn, session_id: str) -> None:
    conn.execute(
        insert(sessions_table).values(
            id=session_id,
            user_id="alice",
            auth_provider_type="local",
            title="Phase 5b Test",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )


def _seed_composition_state(conn, *, state_id: str, session_id: str) -> None:
    conn.execute(
        insert(composition_states_table).values(
            id=state_id,
            session_id=session_id,
            version=1,
            is_valid=False,
            provenance="session_seed",
            created_at=datetime.now(UTC),
        )
    )


def _user_approved_row(
    *,
    row_id: str,
    session_id: str,
    state_id: str,
    tool_call_id: str = "tool_call_1",
    choice: str = "pending",
    resolved_at: datetime | None = None,
    accepted_value: str | None = None,
    hash_domain_version: str | None = None,
    arguments_hash: str | None = None,
    resolved_prompt_template_hash: str | None = None,
    runtime_model_identifier_at_resolve: str | None = None,
    runtime_model_version_at_resolve: str | None = None,
) -> dict:
    return {
        "id": row_id,
        "session_id": session_id,
        "composition_state_id": state_id,
        "affected_node_id": "llm_transform_1",
        "tool_call_id": tool_call_id,
        "user_term": "cool",
        "llm_draft": "A draft definition of cool",
        "accepted_value": accepted_value,
        "choice": choice,
        "created_at": datetime.now(UTC),
        "resolved_at": resolved_at,
        "actor": "alice",
        "model_identifier": "anthropic/claude-opus-4-7",
        "model_version": "2026-05-01",
        "provider": "anthropic",
        "composer_skill_hash": "0" * 64,
        "arguments_hash": arguments_hash,
        "hash_domain_version": hash_domain_version,
        "interpretation_source": "user_approved",
        "runtime_model_identifier_at_resolve": runtime_model_identifier_at_resolve,
        "runtime_model_version_at_resolve": runtime_model_version_at_resolve,
        "resolved_prompt_template_hash": resolved_prompt_template_hash,
    }


def _opt_out_row(*, row_id: str, session_id: str, tool_call_id: str | None = None) -> dict:
    return {
        "id": row_id,
        "session_id": session_id,
        "composition_state_id": None,
        "affected_node_id": None,
        "tool_call_id": tool_call_id,
        "user_term": None,
        "llm_draft": None,
        "accepted_value": None,
        "choice": "opted_out",
        "created_at": datetime.now(UTC),
        "resolved_at": datetime.now(UTC),
        "actor": "alice",
        "model_identifier": None,
        "model_version": None,
        "provider": None,
        "composer_skill_hash": None,
        "arguments_hash": None,
        "hash_domain_version": None,
        "interpretation_source": "auto_interpreted_opt_out",
        "runtime_model_identifier_at_resolve": None,
        "runtime_model_version_at_resolve": None,
        "resolved_prompt_template_hash": None,
    }


def _no_surfaces_row(
    *,
    row_id: str,
    session_id: str,
    tool_call_id: str | None = None,
    user_term: str | None = None,
    model_identifier: str | None = "anthropic/claude-opus-4-7",
) -> dict:
    return {
        "id": row_id,
        "session_id": session_id,
        "composition_state_id": None,
        "affected_node_id": None,
        "tool_call_id": tool_call_id,
        "user_term": user_term,
        "llm_draft": None,
        "accepted_value": None,
        "choice": "opted_out",
        "created_at": datetime.now(UTC),
        "resolved_at": datetime.now(UTC),
        "actor": "alice",
        "model_identifier": model_identifier,
        "model_version": "2026-05-01" if model_identifier is not None else None,
        "provider": "anthropic" if model_identifier is not None else None,
        "composer_skill_hash": ("0" * 64) if model_identifier is not None else None,
        "arguments_hash": None,
        "hash_domain_version": None,
        "interpretation_source": "auto_interpreted_no_surfaces",
        "runtime_model_identifier_at_resolve": None,
        "runtime_model_version_at_resolve": None,
        "resolved_prompt_template_hash": None,
    }


# Test 1 — table exists with all expected columns -----------------------------
class TestSchema:
    def test_interpretation_events_columns(self, engine) -> None:
        inspector = inspect(engine)
        assert "interpretation_events" in inspector.get_table_names()
        columns = {c["name"] for c in inspector.get_columns("interpretation_events")}
        assert columns == {
            "id",
            "session_id",
            "composition_state_id",
            "affected_node_id",
            "tool_call_id",
            "user_term",
            "llm_draft",
            "accepted_value",
            "choice",
            "created_at",
            "resolved_at",
            "actor",
            "model_identifier",
            "model_version",
            "provider",
            "composer_skill_hash",
            "arguments_hash",
            "hash_domain_version",
            "interpretation_source",
            "runtime_model_identifier_at_resolve",
            "runtime_model_version_at_resolve",
            "resolved_prompt_template_hash",
        }

    def test_skill_markdown_history_columns(self, engine) -> None:
        inspector = inspect(engine)
        assert "skill_markdown_history" in inspector.get_table_names()
        columns = {c["name"] for c in inspector.get_columns("skill_markdown_history")}
        assert columns == {"hash", "filename", "content", "first_seen_at"}

    # Test 9 — sessions.interpretation_review_disabled exists
    def test_sessions_has_interpretation_review_disabled(self, engine) -> None:
        inspector = inspect(engine)
        cols = {c["name"]: c for c in inspector.get_columns("sessions")}
        assert "interpretation_review_disabled" in cols
        # Boolean, NOT NULL
        assert cols["interpretation_review_disabled"]["nullable"] is False
        # Default 'false' applied at INSERT time
        session_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            row = conn.execute(select(sessions_table.c.interpretation_review_disabled).where(sessions_table.c.id == session_id)).fetchone()
        assert row is not None
        assert row[0] in (0, False)


# Test 2 — pending row with resolved_at set raises IntegrityError -------------
class TestStatusConsistencyCheck:
    def test_pending_with_resolved_at_raises(self, engine) -> None:
        session_id = str(uuid.uuid4())
        state_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            _seed_composition_state(conn, state_id=state_id, session_id=session_id)
            with pytest.raises(IntegrityError):
                conn.execute(
                    insert(interpretation_events_table).values(
                        _user_approved_row(
                            row_id=str(uuid.uuid4()),
                            session_id=session_id,
                            state_id=state_id,
                            choice="pending",
                            resolved_at=datetime.now(UTC),
                        )
                    )
                )

    # Test 3 — choice=accepted_as_drafted with accepted_value NULL raises
    def test_accepted_as_drafted_with_null_accepted_value_raises(self, engine) -> None:
        session_id = str(uuid.uuid4())
        state_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            _seed_composition_state(conn, state_id=state_id, session_id=session_id)
            with pytest.raises(IntegrityError):
                conn.execute(
                    insert(interpretation_events_table).values(
                        _user_approved_row(
                            row_id=str(uuid.uuid4()),
                            session_id=session_id,
                            state_id=state_id,
                            choice="accepted_as_drafted",
                            resolved_at=datetime.now(UTC),
                            accepted_value=None,
                            hash_domain_version="v1",
                            arguments_hash="a" * 64,
                            resolved_prompt_template_hash="b" * 64,
                            runtime_model_identifier_at_resolve="anthropic/claude-opus-4-7",
                            runtime_model_version_at_resolve="2026-05-01",
                        )
                    )
                )


# Tests 4 / 4a / 4b / 4c — interpretation_source nullability shapes -----------
class TestSourceNullability:
    # Test 4 — opted_out row with all nullable fields NULL succeeds
    def test_opted_out_with_all_null_fields_succeeds(self, engine) -> None:
        session_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            conn.execute(insert(interpretation_events_table).values(_opt_out_row(row_id=str(uuid.uuid4()), session_id=session_id)))

    # Test 4a — opted_out + composition_state_id non-NULL → IntegrityError
    def test_opted_out_with_non_null_composition_state_id_raises(self, engine) -> None:
        session_id = str(uuid.uuid4())
        state_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            _seed_composition_state(conn, state_id=state_id, session_id=session_id)
            row = _opt_out_row(row_id=str(uuid.uuid4()), session_id=session_id)
            row["composition_state_id"] = state_id
            with pytest.raises(IntegrityError):
                conn.execute(insert(interpretation_events_table).values(row))

    # Test 4b — user_approved with user_term NULL → IntegrityError
    def test_user_approved_with_null_user_term_raises(self, engine) -> None:
        session_id = str(uuid.uuid4())
        state_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            _seed_composition_state(conn, state_id=state_id, session_id=session_id)
            row = _user_approved_row(
                row_id=str(uuid.uuid4()),
                session_id=session_id,
                state_id=state_id,
            )
            row["user_term"] = None
            with pytest.raises(IntegrityError):
                conn.execute(insert(interpretation_events_table).values(row))

    # Test 4b inverse — opted_out + non-NULL user_term → IntegrityError
    def test_opted_out_with_non_null_user_term_raises(self, engine) -> None:
        session_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            row = _opt_out_row(row_id=str(uuid.uuid4()), session_id=session_id)
            row["user_term"] = "cool"
            with pytest.raises(IntegrityError):
                conn.execute(insert(interpretation_events_table).values(row))

    # Test 4c — auto_interpreted_no_surfaces shape
    def test_no_surfaces_with_non_null_user_term_raises(self, engine) -> None:
        session_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            row = _no_surfaces_row(
                row_id=str(uuid.uuid4()),
                session_id=session_id,
                user_term="cool",
            )
            with pytest.raises(IntegrityError):
                conn.execute(insert(interpretation_events_table).values(row))

    def test_no_surfaces_with_null_surfaces_and_provenance_succeeds(self, engine) -> None:
        session_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            conn.execute(insert(interpretation_events_table).values(_no_surfaces_row(row_id=str(uuid.uuid4()), session_id=session_id)))

    def test_no_surfaces_with_null_model_identifier_raises(self, engine) -> None:
        session_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            row = _no_surfaces_row(
                row_id=str(uuid.uuid4()),
                session_id=session_id,
                model_identifier=None,
            )
            with pytest.raises(IntegrityError):
                conn.execute(insert(interpretation_events_table).values(row))


# Tests 5 / 5a / 6 — partial unique index on pending tool calls ----------------
class TestPartialUniqueOnPending:
    def test_two_pending_same_tool_call_raises(self, engine) -> None:
        session_id = str(uuid.uuid4())
        state_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            _seed_composition_state(conn, state_id=state_id, session_id=session_id)
            conn.execute(
                insert(interpretation_events_table).values(
                    _user_approved_row(
                        row_id=str(uuid.uuid4()),
                        session_id=session_id,
                        state_id=state_id,
                        tool_call_id="tc_shared",
                        choice="pending",
                    )
                )
            )
            with pytest.raises(IntegrityError):
                conn.execute(
                    insert(interpretation_events_table).values(
                        _user_approved_row(
                            row_id=str(uuid.uuid4()),
                            session_id=session_id,
                            state_id=state_id,
                            tool_call_id="tc_shared",
                            choice="pending",
                        )
                    )
                )

    def test_two_resolved_same_tool_call_succeeds(self, engine) -> None:
        session_id = str(uuid.uuid4())
        state_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            _seed_composition_state(conn, state_id=state_id, session_id=session_id)
            for _ in range(2):
                conn.execute(
                    insert(interpretation_events_table).values(
                        _user_approved_row(
                            row_id=str(uuid.uuid4()),
                            session_id=session_id,
                            state_id=state_id,
                            tool_call_id="tc_shared",
                            choice="accepted_as_drafted",
                            resolved_at=now,
                            accepted_value="ok",
                            hash_domain_version="v1",
                            arguments_hash="a" * 64,
                        )
                    )
                )


# Test 7 — composite FK enforcement on non-opted-out rows ---------------------
class TestCompositeForeignKey:
    def test_user_approved_with_nonexistent_state_id_raises(self, engine) -> None:
        session_id = str(uuid.uuid4())
        with engine.begin() as conn:
            conn.execute(text("PRAGMA foreign_keys=ON"))
            _insert_session(conn, session_id)
            row = _user_approved_row(
                row_id=str(uuid.uuid4()),
                session_id=session_id,
                state_id="does-not-exist",
            )
            with pytest.raises(IntegrityError):
                conn.execute(insert(interpretation_events_table).values(row))


# Test 8 — composition_states.provenance closed enum ---------------------------
class TestCompositionStatesProvenanceEnum:
    def test_invalid_provenance_rejected(self, engine) -> None:
        session_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            with pytest.raises(IntegrityError):
                conn.execute(
                    insert(composition_states_table).values(
                        id=str(uuid.uuid4()),
                        session_id=session_id,
                        version=1,
                        is_valid=False,
                        provenance="not_a_real_value",
                        created_at=datetime.now(UTC),
                    )
                )

    def test_interpretation_resolve_provenance_accepted(self, engine) -> None:
        session_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            conn.execute(
                insert(composition_states_table).values(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    version=1,
                    is_valid=False,
                    provenance="interpretation_resolve",
                    created_at=datetime.now(UTC),
                )
            )


# Test 10 — append-only trigger on interpretation_events ----------------------
class TestImmutabilityTrigger:
    def _insert_resolved_row(self, engine, *, accepted_value: str = "ok") -> tuple[str, str]:
        session_id = str(uuid.uuid4())
        state_id = str(uuid.uuid4())
        row_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            _seed_composition_state(conn, state_id=state_id, session_id=session_id)
            conn.execute(
                insert(interpretation_events_table).values(
                    _user_approved_row(
                        row_id=row_id,
                        session_id=session_id,
                        state_id=state_id,
                        choice="accepted_as_drafted",
                        resolved_at=datetime.now(UTC),
                        accepted_value=accepted_value,
                        hash_domain_version="v1",
                        arguments_hash="a" * 64,
                    )
                )
            )
        return session_id, row_id

    def test_update_accepted_value_on_resolved_raises(self, engine) -> None:
        _, row_id = self._insert_resolved_row(engine)
        with pytest.raises(IntegrityError, match="immutable"), engine.begin() as conn:
            conn.execute(
                text("UPDATE interpretation_events SET accepted_value = 'tampered' WHERE id = :id"),
                {"id": row_id},
            )

    def test_flip_resolved_choice_back_to_pending_raises(self, engine) -> None:
        _, row_id = self._insert_resolved_row(engine)
        with pytest.raises(IntegrityError, match="immutable"), engine.begin() as conn:
            conn.execute(
                text("UPDATE interpretation_events SET choice = 'pending' WHERE id = :id"),
                {"id": row_id},
            )

    def test_update_pending_non_settled_field_does_not_raise(self, engine) -> None:
        session_id = str(uuid.uuid4())
        state_id = str(uuid.uuid4())
        row_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            _seed_composition_state(conn, state_id=state_id, session_id=session_id)
            conn.execute(
                insert(interpretation_events_table).values(
                    _user_approved_row(
                        row_id=row_id,
                        session_id=session_id,
                        state_id=state_id,
                        choice="pending",
                    )
                )
            )
        # Pending row: trigger does not fire because OLD.resolved_at IS NULL.
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE interpretation_events SET model_version = 'updated' WHERE id = :id"),
                {"id": row_id},
            )


# Test 11 — F-8 trigger-existence via production bootstrap path ----------------
class TestTriggerInstalledByBootstrap:
    def test_immutable_resolved_trigger_present(self, engine) -> None:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='trigger' AND name='trg_interpretation_events_immutable_resolved'")
            ).fetchone()
        assert row is not None

    def test_immutable_chat_messages_trigger_present(self, engine) -> None:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='trigger' AND name='trg_chat_messages_immutable_content'")
            ).fetchone()
        assert row is not None

    def test_chat_messages_content_immutable(self, engine) -> None:
        """chat_messages.content cannot be updated once written."""
        session_id = str(uuid.uuid4())
        msg_id = str(uuid.uuid4())
        with engine.begin() as conn:
            _insert_session(conn, session_id)
            conn.execute(
                insert(chat_messages_table).values(
                    id=msg_id,
                    session_id=session_id,
                    role="user",
                    content="original",
                    sequence_no=1,
                    writer_principal="route_user_message",
                    created_at=datetime.now(UTC),
                )
            )
        with pytest.raises(IntegrityError, match="append-only"), engine.begin() as conn:
            conn.execute(
                text("UPDATE chat_messages SET content = 'tampered' WHERE id = :id"),
                {"id": msg_id},
            )


# Test 12 — F-11 composition_state_id index uses index, not scan --------------
class TestCompositionStateIdIndex:
    def test_lookup_by_composition_state_id_uses_index(self, engine) -> None:
        with engine.connect() as conn:
            plan_rows = conn.execute(
                text("EXPLAIN QUERY PLAN SELECT * FROM interpretation_events WHERE composition_state_id = :sid"),
                {"sid": "any-id"},
            ).fetchall()
        plan_text = " ".join(str(row) for row in plan_rows)
        assert "SEARCH" in plan_text and "INDEX" in plan_text, f"Expected SEARCH USING INDEX in plan; got: {plan_text}"


# F-24 — schema validator catches missing triggers ----------------------------
class TestSchemaValidatorCatchesMissingTrigger:
    def test_validator_raises_when_trigger_dropped(self, engine, tmp_path) -> None:
        from elspeth.web.sessions.schema import SessionSchemaError, initialize_session_schema

        db_path = tmp_path / "session_drop_trigger.db"
        eng = create_session_engine(f"sqlite:///{db_path}")
        initialize_session_schema(eng)
        with eng.begin() as conn:
            conn.execute(text("DROP TRIGGER trg_interpretation_events_immutable_resolved"))
        # Second initialize call validates the existing DB and must crash.
        with pytest.raises(SessionSchemaError, match="trigger"):
            initialize_session_schema(eng)


# Cross-DB — Landscape calls.resolved_prompt_template_hash --------------------
class TestLandscapeCallsColumn:
    def test_resolved_prompt_template_hash_column_exists(self) -> None:
        from elspeth.core.landscape.schema import calls_table

        assert "resolved_prompt_template_hash" in calls_table.c
        col = calls_table.c.resolved_prompt_template_hash
        assert col.nullable is True

    def test_index_on_resolved_prompt_template_hash_exists(self) -> None:
        from elspeth.core.landscape.schema import calls_table

        index_names = {idx.name for idx in calls_table.indexes}
        # Sanity check via metadata.indexes too — Index() declared at module
        # scope is attached to the table.
        assert any(name == "ix_calls_resolved_prompt_template_hash" for name in index_names) or _index_exists_in_metadata(
            "ix_calls_resolved_prompt_template_hash"
        )


def _index_exists_in_metadata(name: str) -> bool:
    from elspeth.core.landscape.schema import metadata

    for table in metadata.tables.values():
        for idx in table.indexes:
            if idx.name == name:
                return True
    return False


# skill_markdown_history insert smoke -----------------------------------------
class TestSkillMarkdownHistory:
    def test_insert_skill_row(self, engine) -> None:
        with engine.begin() as conn:
            conn.execute(
                insert(skill_markdown_history_table).values(
                    hash="d" * 64,
                    filename="pipeline_composer.md",
                    content="# composer skill",
                    first_seen_at=datetime.now(UTC),
                )
            )
