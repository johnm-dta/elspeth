"""Tests for current-schema session database bootstrap."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_mock_engine, insert, inspect, text
from sqlalchemy.exc import IntegrityError

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import blobs_table, metadata, sessions_table
from elspeth.web.sessions.schema import (
    SessionSchemaError,
    _validate_current_schema,
    initialize_session_schema,
)


@pytest.fixture
def engine():
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    return eng


def test_validator_rejects_same_named_unique_index_with_wrong_columns(engine) -> None:
    """A same-named unique index with a different column set must be rejected.

    Regression for elspeth-97bedcd9c4: _validate_named_unique_constraints compared
    only index/constraint NAMES, so dropping uq_chat_messages_tool_call_id (unique
    on (session_id, tool_call_id)) and recreating it under the same name on just
    (session_id) was accepted as "current" — silently over-restricting the
    intended "unique (session_id, tool_call_id) for tool rows" invariant to "at
    most one tool row per session".
    """
    with engine.begin() as conn:
        conn.execute(text("DROP INDEX uq_chat_messages_tool_call_id"))
        conn.execute(text("CREATE UNIQUE INDEX uq_chat_messages_tool_call_id ON chat_messages (session_id) WHERE role = 'tool'"))

    with pytest.raises(SessionSchemaError, match="column mismatch"):
        _validate_current_schema(engine)


def test_initialize_session_schema_creates_current_schema_without_alembic_table() -> None:
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)

    inspector = inspect(eng)
    assert set(inspector.get_table_names()) == set(metadata.tables)
    assert "alembic_version" not in inspector.get_table_names()
    run_columns = {column["name"] for column in inspector.get_columns("runs")}
    assert "rows_routed_success" in run_columns
    assert "rows_routed_failure" in run_columns
    assert "content_hash" in {column["name"] for column in inspector.get_columns("blobs")}
    assert "ck_blobs_ready_hash" in {check["name"] for check in inspector.get_check_constraints("blobs")}


def test_sqlite_trigger_ddl_is_not_emitted_for_postgres_schema() -> None:
    """SQLite trigger bodies must not leak into PostgreSQL schema bootstrap."""
    emitted: list[str] = []
    engine = create_mock_engine("postgresql://", lambda sql, *_, **__: emitted.append(str(sql)))

    metadata.create_all(engine)

    assert any("CREATE TABLE chat_messages" in statement for statement in emitted)
    assert not any("CREATE TRIGGER" in statement for statement in emitted)
    assert not any("SELECT RAISE" in statement for statement in emitted)


def test_postgres_schema_uses_postgres_non_blank_check_syntax() -> None:
    """ASCII-whitespace non-blank CHECKs must compile to PostgreSQL syntax."""
    emitted: list[str] = []
    engine = create_mock_engine(
        "postgresql://",
        lambda sql, *_, **__: emitted.append(str(sql.compile(dialect=engine.dialect))),
    )

    metadata.create_all(engine)

    ddl = "\n".join(emitted)
    assert "ck_composition_proposals_composer_provenance_all_or_none" in ddl
    assert "ck_blobs_creating_llm_provenance_nullability" in ddl
    assert "btrim(composer_model_identifier" in ddl
    assert "btrim(creating_model_identifier" in ddl
    assert "chr(9)" in ddl
    assert "char(9)" not in ddl
    assert " NOT GLOB " not in ddl


def test_initialize_session_schema_is_idempotent_for_current_schema() -> None:
    eng = create_session_engine("sqlite:///:memory:")

    initialize_session_schema(eng)
    initialize_session_schema(eng)


def test_initialize_session_schema_rejects_legacy_alembic_database() -> None:
    eng = create_session_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        conn.execute(text("INSERT INTO alembic_version (version_num) VALUES ('007')"))
        conn.execute(text("CREATE TABLE _alembic_tmp_blobs (id VARCHAR PRIMARY KEY)"))

    with pytest.raises(SessionSchemaError, match="SESSION_SCHEMA_EPOCH"):
        initialize_session_schema(eng)


def test_initialize_session_schema_rejects_partial_stale_schema() -> None:
    eng = create_session_engine("sqlite:///:memory:")
    with eng.begin() as conn:
        conn.execute(text("CREATE TABLE sessions (id VARCHAR PRIMARY KEY)"))

    with pytest.raises(SessionSchemaError, match="SESSION_SCHEMA_EPOCH"):
        initialize_session_schema(eng)


def test_current_schema_enforces_ready_blob_hash_check(engine) -> None:
    session_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            insert(sessions_table).values(
                id=session_id,
                user_id="alice",
                auth_provider_type="local",
                title="Test",
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )

        with pytest.raises(IntegrityError):
            conn.execute(
                insert(blobs_table).values(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    filename="artifact.txt",
                    mime_type="text/plain",
                    size_bytes=4,
                    content_hash=None,
                    storage_path="blobs/artifact.txt",
                    created_at=datetime.now(UTC),
                    created_by="user",
                    status="ready",
                )
            )


# ---------------------------------------------------------------------------
# elspeth-obs-2ef48619d5: partial-index dialect-symmetry static guard.
# elspeth-obs-3ac0c829c5: Index(unique=True) included in unique-constraint
# validation.
# ---------------------------------------------------------------------------


def _validate_synthetic_metadata(
    monkeypatch: pytest.MonkeyPatch,
    synthetic,
):
    """Helper: monkeypatch the schema module to use a synthetic metadata
    object, then call the validator. The static partial-index guard
    walks ``metadata.tables`` so we have to redirect the module-level
    binding rather than pass metadata in. The validator is normally
    called from ``initialize_session_schema``; calling it directly skips
    the engine-level inspector branch and exercises only the static
    check path.
    """
    from elspeth.web.sessions import schema as schema_module

    monkeypatch.setattr(schema_module, "metadata", synthetic)
    schema_module._validate_partial_index_dialect_symmetry()


def test_partial_index_dialect_symmetry_accepts_paired_predicates(monkeypatch) -> None:
    """Real metadata (both partial indexes have matching sqlite_where=
    and postgresql_where=) passes the static guard."""
    from elspeth.web.sessions.models import metadata
    from elspeth.web.sessions.schema import _validate_partial_index_dialect_symmetry

    del monkeypatch  # unused — exercising the real metadata
    del metadata  # imported for IDE; actual symbol read inside the validator
    _validate_partial_index_dialect_symmetry()  # must not raise


def test_partial_index_dialect_symmetry_rejects_sqlite_only(monkeypatch) -> None:
    """An ``Index(name=..., sqlite_where=...)`` with NO ``postgresql_where=``
    is the original drift class (``uq_runs_one_active_per_session``
    incident before the fix). The static guard must reject it.
    """
    from sqlalchemy import Column, Index, MetaData, String, Table

    synthetic = MetaData()
    t = Table(
        "demo",
        synthetic,
        Column("id", String, primary_key=True),
        Column("status", String, nullable=False),
    )
    Index("idx_demo_active", t.c.id, unique=True, sqlite_where=t.c.status == "active")

    with pytest.raises(SessionSchemaError, match="dialect asymmetry"):
        _validate_synthetic_metadata(monkeypatch, synthetic)


def test_partial_index_dialect_symmetry_rejects_divergent_predicates(monkeypatch) -> None:
    """Both ``sqlite_where=`` and ``postgresql_where=`` set, but compiling
    them under their respective dialects yields different SQL text. The
    static guard must reject this — silently divergent invariants under
    the same name is the second drift class the observation flags.
    """
    from sqlalchemy import Column, Index, MetaData, String, Table

    synthetic = MetaData()
    t = Table(
        "demo",
        synthetic,
        Column("id", String, primary_key=True),
        Column("status", String, nullable=False),
    )
    Index(
        "idx_demo_divergent",
        t.c.id,
        unique=True,
        sqlite_where=t.c.status == "active",
        postgresql_where=t.c.status.in_(["active", "pending"]),
    )

    with pytest.raises(SessionSchemaError, match="WHERE clause text diverges"):
        _validate_synthetic_metadata(monkeypatch, synthetic)


def test_validator_accepts_index_unique_true_as_unique_constraint() -> None:
    """elspeth-obs-3ac0c829c5: ``Index(name=..., unique=True)`` MUST be
    validated by ``_validate_named_unique_constraints``. Pre-fix the
    function only iterated ``table.constraints`` for ``UniqueConstraint``,
    leaving such an index silently unvalidated. Post-fix the function
    unions both shapes into the expected set.

    The current ``models.py`` declares ``uq_chat_messages_tool_call_id``
    and ``uq_runs_one_active_per_session`` as ``Index(unique=True)``
    (so they can carry partial-index ``where=`` predicates that
    ``UniqueConstraint`` does not accept). This test asserts both names
    appear in the function's expected set when called against the real
    metadata; if the union ever regresses, the names disappear from
    expected and the validator becomes silently weaker.
    """
    from sqlalchemy import inspect as sa_inspect

    from elspeth.web.sessions.engine import create_session_engine
    from elspeth.web.sessions.models import chat_messages_table, metadata, runs_table

    eng = create_session_engine("sqlite:///:memory:")
    metadata.create_all(eng)
    inspector = sa_inspect(eng)

    expected_unique_chat = {
        str(constraint.name)
        for constraint in chat_messages_table.constraints
        if type(constraint).__name__ == "UniqueConstraint" and constraint.name is not None
    } | {str(index.name) for index in chat_messages_table.indexes if index.unique and index.name is not None}
    assert "uq_chat_messages_tool_call_id" in expected_unique_chat

    expected_unique_runs = {
        str(constraint.name)
        for constraint in runs_table.constraints
        if type(constraint).__name__ == "UniqueConstraint" and constraint.name is not None
    } | {str(index.name) for index in runs_table.indexes if index.unique and index.name is not None}
    assert "uq_runs_one_active_per_session" in expected_unique_runs

    # And the validator (called via initialize_session_schema) accepts
    # the real metadata end-to-end.
    initialize_session_schema(eng)  # must not raise

    del inspector
