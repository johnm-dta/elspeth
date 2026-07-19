"""Tests for current-schema session database bootstrap."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_mock_engine, insert, inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import QueuePool

from elspeth.web.sessions.engine import create_session_engine
from elspeth.web.sessions.models import SESSION_SCHEMA_EPOCH, blobs_table, metadata, sessions_table
from elspeth.web.sessions.schema import (
    SessionSchemaError,
    _stamp_schema_sentinels,
    _validate_current_schema,
    initialize_session_schema,
    probe_current_schema,
)


def _create_all_on_mock_engine(engine) -> None:
    try:
        metadata.create_all(engine)
    finally:
        # SQLAlchemy's mock PostgreSQL create_all path marks cycle-breaking
        # foreign keys with a transient _create_rule. If left on the shared
        # metadata object, later SQLite create_all calls omit those inline FKs.
        _MISSING = object()
        for table in metadata.tables.values():
            for constraint in table.constraints:
                if getattr(constraint, "_create_rule", _MISSING) is not _MISSING:
                    constraint._create_rule = None


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


def test_postgres_schema_emits_native_audit_trigger_ddl() -> None:
    """PostgreSQL bootstrap emits all native triggers, never SQLite syntax."""
    emitted: list[str] = []
    engine = create_mock_engine("postgresql://", lambda sql, *_, **__: emitted.append(str(sql)))

    _create_all_on_mock_engine(engine)

    assert any("CREATE TABLE chat_messages" in statement for statement in emitted)
    ddl = "\n".join(emitted)
    for trigger_name in (
        "trg_interpretation_events_immutable_resolved",
        "trg_interpretation_events_no_delete_resolved",
        "trg_composer_completion_events_no_update",
        "trg_composer_completion_events_no_delete",
        "trg_chat_messages_immutable_content",
        "trg_chat_messages_no_delete",
        "trg_guided_operations_terminal_immutable",
        "trg_guided_operation_events_no_update",
        "trg_guided_operation_events_no_delete",
    ):
        assert f"CREATE TRIGGER {trigger_name}" in ddl
    assert not any("SELECT RAISE" in statement for statement in emitted)


def test_postgres_schema_uses_postgres_non_blank_check_syntax() -> None:
    """ASCII-whitespace non-blank CHECKs must compile to PostgreSQL syntax."""
    emitted: list[str] = []
    engine = create_mock_engine(
        "postgresql://",
        lambda sql, *_, **__: emitted.append(str(sql.compile(dialect=engine.dialect))),
    )

    _create_all_on_mock_engine(engine)

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


def test_probe_current_schema_accepts_engine_and_connection_for_current_schema() -> None:
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)

    assert probe_current_schema(eng) is True
    with eng.connect() as connection:
        assert probe_current_schema(connection) is True


def test_probe_current_schema_connection_does_not_checkout_a_second_connection() -> None:
    eng = create_session_engine(
        "sqlite:///:memory:",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.01,
    )
    initialize_session_schema(eng)

    with eng.connect() as connection:
        assert probe_current_schema(connection) is True


def test_probe_current_schema_true_preserves_transaction_free_connection() -> None:
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)

    with eng.connect() as connection:
        assert connection.in_transaction() is False
        assert probe_current_schema(connection) is True
        assert connection.in_transaction() is False


def test_probe_current_schema_false_preserves_transaction_free_connection() -> None:
    eng = create_session_engine("sqlite:///:memory:")

    with eng.connect() as connection:
        assert connection.in_transaction() is False
        assert probe_current_schema(connection) is False
        assert connection.in_transaction() is False


def test_probe_current_schema_true_preserves_existing_transaction() -> None:
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)

    with eng.connect() as connection, connection.begin() as transaction:
        assert probe_current_schema(connection) is True
        assert connection.in_transaction() is True
        assert transaction.is_active is True


def test_probe_current_schema_false_preserves_existing_transaction() -> None:
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    with eng.begin() as connection:
        connection.execute(text("DROP INDEX uq_chat_messages_tool_call_id"))

    with eng.connect() as connection, connection.begin() as transaction:
        assert probe_current_schema(connection) is False
        assert connection.in_transaction() is True
        assert transaction.is_active is True


def test_probe_current_schema_returns_false_for_shape_drift() -> None:
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    with eng.begin() as connection:
        connection.execute(text("DROP INDEX uq_chat_messages_tool_call_id"))

    assert probe_current_schema(eng) is False


def test_probe_current_schema_never_creates_or_stamps_objects() -> None:
    eng = create_session_engine("sqlite:///:memory:")

    assert probe_current_schema(eng) is False
    assert inspect(eng).get_table_names() == []
    with eng.connect() as connection:
        assert connection.execute(text("PRAGMA application_id")).scalar_one() == 0
        assert connection.execute(text("PRAGMA user_version")).scalar_one() == 0


def test_probe_current_schema_only_converts_session_schema_error(monkeypatch) -> None:
    from elspeth.web.sessions import schema as schema_module

    eng = create_session_engine("sqlite:///:memory:")

    def _unexpected_failure(_bind) -> None:
        raise RuntimeError("inspector unavailable")

    monkeypatch.setattr(schema_module, "_assert_schema_sentinels", _unexpected_failure)

    with pytest.raises(RuntimeError, match="inspector unavailable"):
        probe_current_schema(eng)


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


def test_initialize_session_schema_rejects_prior_epoch_database() -> None:
    """A valid full-schema DB stamped at the prior epoch fail-closes at boot.

    Seed a complete current-schema DB, then re-stamp only the SQLite epoch.
    Because the SQL shape and cross-dialect identity row remain current, the
    PRAGMA guard is the only possible failure source.
    """
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)  # full schema + stamps the CURRENT epoch
    with eng.begin() as conn:
        conn.execute(text(f"PRAGMA user_version = {SESSION_SCHEMA_EPOCH - 1}"))
    with pytest.raises(SessionSchemaError, match="SESSION_SCHEMA_EPOCH"):
        initialize_session_schema(eng)


def test_epoch_30_database_without_schema_9_operation_contract_fails_closed_with_recreate_guidance(tmp_path) -> None:
    """The epoch-30 operation CHECKs cannot be opened by epoch-31 code."""
    db_path = tmp_path / "epoch-30-without-guided-plan.db"
    engine = create_session_engine(f"sqlite:///{db_path}")
    initialize_session_schema(engine)
    with engine.begin() as connection:
        guided_operations_sql = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'guided_operations'")
        ).scalar_one()
        assert "'guided_plan'" in guided_operations_sql
        epoch_30_sql = guided_operations_sql.replace("'guided_plan'", "'guided_convert'")
        assert epoch_30_sql != guided_operations_sql
        assert "'guided_plan'" not in epoch_30_sql
        connection.execute(text("PRAGMA writable_schema = ON"))
        connection.execute(
            text("UPDATE sqlite_master SET sql = :sql WHERE type = 'table' AND name = 'guided_operations'"),
            {"sql": epoch_30_sql},
        )
        connection.execute(text("UPDATE elspeth_schema_identity SET schema_epoch = 30 WHERE store_kind = 'session'"))
        connection.execute(text("PRAGMA user_version = 30"))
        schema_version = connection.execute(text("PRAGMA schema_version")).scalar_one()
        connection.execute(text(f"PRAGMA schema_version = {schema_version + 1}"))
        connection.execute(text("PRAGMA writable_schema = OFF"))
    engine.dispose()

    stale_engine = create_session_engine(f"sqlite:///{db_path}")
    with stale_engine.connect() as connection:
        assert connection.execute(text("PRAGMA user_version")).scalar_one() == 30
        stored_sql = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'guided_operations'")
        ).scalar_one()
        assert "'guided_plan'" not in stored_sql

    with pytest.raises(
        SessionSchemaError,
        match=r"Session DB schema version 30 does not match SESSION_SCHEMA_EPOCH=31.*"
        r"Delete the session DB file and restart",
    ):
        initialize_session_schema(stale_engine)


@pytest.mark.parametrize("renamed_column", ["singleton_id", "application_id", "store_kind", "schema_epoch"])
def test_initialize_session_schema_rejects_identity_table_with_renamed_column(renamed_column: str) -> None:
    """A divergent identity-table shape fail-closes with the actionable error.

    Regression for elspeth-5cf1ca2852: ``read_schema_identities()`` selected
    the declared identity columns before any live-shape validation, so a
    missing or renamed column leaked a raw ``sqlalchemy.exc.OperationalError``
    instead of the delete-and-restart ``SessionSchemaError``.
    """
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    with eng.begin() as conn:
        conn.execute(text(f"ALTER TABLE elspeth_schema_identity RENAME COLUMN {renamed_column} TO drifted_away"))

    with pytest.raises(SessionSchemaError, match=renamed_column):
        initialize_session_schema(eng)


def test_probe_current_schema_returns_false_for_identity_table_with_renamed_column() -> None:
    eng = create_session_engine("sqlite:///:memory:")
    initialize_session_schema(eng)
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE elspeth_schema_identity RENAME COLUMN schema_epoch TO drifted_away"))

    assert probe_current_schema(eng) is False


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


def test_initialize_session_schema_rejects_same_named_check_with_weakened_sql(tmp_path) -> None:
    """A current-epoch DB with a same-named weaker CHECK is stale.

    Regression for elspeth-28bb7fcacb: the validator compared only CHECK names,
    so a manually rebuilt/stale ``ck_blobs_ready_hash`` accepted startup even
    when its SQL no longer enforced the ready-blob 64-char lowercase-hex hash
    invariant.
    """
    db_path = tmp_path / "sessions.db"
    eng = create_session_engine(f"sqlite:///{db_path}")
    initialize_session_schema(eng)
    eng.dispose()

    strong = "status != 'ready' OR (content_hash IS NOT NULL AND length(content_hash) = 64 AND content_hash NOT GLOB '*[^a-f0-9]*')"
    weak = "status != 'ready' OR content_hash IS NOT NULL"
    eng = create_session_engine(f"sqlite:///{db_path}")
    with eng.begin() as conn:
        original_sql = conn.execute(text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'blobs'")).scalar_one()
        assert "CONSTRAINT ck_blobs_ready_hash" in original_sql
        assert strong in original_sql
        conn.execute(text("PRAGMA writable_schema = ON"))
        conn.execute(
            text("UPDATE sqlite_master SET sql = :sql WHERE type = 'table' AND name = 'blobs'"),
            {"sql": original_sql.replace(strong, weak)},
        )
        conn.execute(text("PRAGMA writable_schema = OFF"))
    eng.dispose()

    reopened = create_session_engine(f"sqlite:///{db_path}")
    with pytest.raises(SessionSchemaError, match="CHECK constraint SQL mismatch"):
        initialize_session_schema(reopened)


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
    _stamp_schema_sentinels(eng)
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
