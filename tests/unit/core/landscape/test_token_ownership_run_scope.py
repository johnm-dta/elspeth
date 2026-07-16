"""Run-scoped token ownership and epoch-24 migration regressions."""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.schema import CreateTable

from elspeth.contracts import NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import database as landscape_database
from elspeth.core.landscape.database import (
    _REQUIRED_COMPOSITE_FOREIGN_KEYS,
    LandscapeDB,
    LandscapeSchemaShape,
    SchemaCompatibilityError,
    probe_schema_shape,
)
from elspeth.core.landscape.schema import SQLITE_SCHEMA_EPOCH, tokens_table
from tests.fixtures.landscape import make_factory, make_recorder_with_run

_OBSERVED_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})

_LEGACY_TOKENS_DDL = """
CREATE TABLE tokens_epoch_23 (
    token_id VARCHAR(64) NOT NULL,
    row_id VARCHAR(64) NOT NULL,
    run_id VARCHAR(64) NOT NULL,
    fork_group_id VARCHAR(64),
    join_group_id VARCHAR(64),
    expand_group_id VARCHAR(32),
    branch_name VARCHAR(64),
    step_in_pipeline INTEGER,
    token_data_ref VARCHAR(64),
    created_at DATETIME NOT NULL,
    PRIMARY KEY (token_id),
    UNIQUE (token_id, run_id),
    FOREIGN KEY(row_id) REFERENCES rows (row_id),
    FOREIGN KEY(run_id) REFERENCES runs (run_id)
)
"""

_LEGACY_OPERATIONS_DDL = """
CREATE TABLE operations_epoch_25 (
    operation_id VARCHAR(64) NOT NULL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES runs(run_id),
    node_id VARCHAR(64) NOT NULL,
    operation_type VARCHAR(32) NOT NULL,
    started_at DATETIME NOT NULL,
    completed_at DATETIME,
    status VARCHAR(16) NOT NULL,
    input_data_ref VARCHAR(256), input_data_hash VARCHAR(64),
    output_data_ref VARCHAR(256), output_data_hash VARCHAR(64),
    error_message TEXT, duration_ms FLOAT,
    FOREIGN KEY(node_id, run_id) REFERENCES nodes(node_id, run_id)
)
"""

_LEGACY_ARTIFACTS_DDL = """
CREATE TABLE artifacts_epoch_25 (
    artifact_id VARCHAR(64) NOT NULL PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL REFERENCES runs(run_id),
    produced_by_state_id VARCHAR(64) NOT NULL,
    sink_node_id VARCHAR(64) NOT NULL,
    artifact_type VARCHAR(64) NOT NULL,
    path_or_uri VARCHAR(512) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    size_bytes INTEGER NOT NULL,
    idempotency_key VARCHAR(256),
    created_at DATETIME NOT NULL,
    FOREIGN KEY(produced_by_state_id, run_id) REFERENCES node_states(state_id, run_id),
    FOREIGN KEY(sink_node_id, run_id) REFERENCES nodes(node_id, run_id)
)
"""


def _rewrite_tokens_as_epoch_23(engine: Engine) -> None:
    """Rebuild the current schema into the genuine epoch-23 predecessor."""
    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        cursor.execute("PRAGMA foreign_keys = OFF")
        cursor.execute("BEGIN IMMEDIATE")
        cursor.execute(_LEGACY_TOKENS_DDL)
        cursor.execute(
            """
            INSERT INTO tokens_epoch_23 (
                token_id, row_id, run_id, fork_group_id, join_group_id,
                expand_group_id, branch_name, step_in_pipeline,
                token_data_ref, created_at
            )
            SELECT
                token_id, row_id, run_id, fork_group_id, join_group_id,
                expand_group_id, branch_name, step_in_pipeline,
                token_data_ref, created_at
            FROM tokens
            """
        )
        cursor.execute("DROP TABLE tokens")
        cursor.execute("ALTER TABLE tokens_epoch_23 RENAME TO tokens")
        cursor.execute("CREATE INDEX ix_tokens_expand_group_id ON tokens (expand_group_id)")
        cursor.execute("CREATE INDEX ix_tokens_row_id ON tokens (row_id)")
        cursor.execute("CREATE INDEX ix_tokens_run_id ON tokens (run_id)")

        cursor.execute(_LEGACY_OPERATIONS_DDL)
        cursor.execute(
            """
            INSERT INTO operations_epoch_25
            SELECT operation_id, run_id, node_id, operation_type, started_at,
                   completed_at, status, input_data_ref, input_data_hash,
                   output_data_ref, output_data_hash, error_message, duration_ms
            FROM operations
            """
        )
        cursor.execute("DROP TABLE operations")
        cursor.execute("ALTER TABLE operations_epoch_25 RENAME TO operations")
        cursor.execute("CREATE INDEX ix_operations_node_run ON operations(node_id, run_id)")
        cursor.execute("CREATE INDEX ix_operations_run_id ON operations(run_id)")

        cursor.execute(_LEGACY_ARTIFACTS_DDL)
        cursor.execute(
            """
            INSERT INTO artifacts_epoch_25
            SELECT artifact_id, run_id, produced_by_state_id, sink_node_id,
                   artifact_type, path_or_uri, content_hash, size_bytes,
                   idempotency_key, created_at
            FROM artifacts
            """
        )
        cursor.execute("DROP TABLE artifacts")
        cursor.execute("ALTER TABLE artifacts_epoch_25 RENAME TO artifacts")
        cursor.execute("CREATE INDEX ix_artifacts_run ON artifacts(run_id)")

        cursor.execute("DROP INDEX IF EXISTS uq_runs_export_witness")
        cursor.execute("DROP INDEX IF EXISTS uq_tokens_identity_row_run")
        for table_name in (
            "sink_effect_attempts",
            "sink_effect_export_snapshots",
            "sink_effect_members",
            "sink_effects",
            "sink_effect_streams",
            "audit_export_snapshot_chunks",
            "audit_export_snapshots",
        ):
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        cursor.execute("DROP INDEX IF EXISTS uq_artifacts_run_idempotency_key")
        cursor.execute("DROP TABLE elspeth_schema_identity")
        cursor.execute("PRAGMA user_version = 23")
        raw.commit()
        cursor.execute("PRAGMA foreign_keys = ON")
        assert cursor.execute("PRAGMA foreign_keys").fetchone() == (1,)
    except Exception:
        raw.rollback()
        raise
    finally:
        raw.close()


def _forge_token_run(engine: Engine, *, token_id: str, forged_run_id: str) -> None:
    """Simulate a legacy bad writer that disabled FK enforcement."""
    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        cursor.execute("PRAGMA foreign_keys = OFF")
        cursor.execute(
            "UPDATE tokens SET run_id = ? WHERE token_id = ?",
            (forged_run_id, token_id),
        )
        assert cursor.rowcount == 1
        raw.commit()
        cursor.execute("PRAGMA foreign_keys = ON")
        assert cursor.execute("PRAGMA foreign_keys").fetchone() == (1,)
    finally:
        raw.close()


def _populate_two_runs(db: LandscapeDB) -> tuple[Any, str, str]:
    factory = make_factory(db)
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-A")
    factory.data_flow.register_node(
        run_id="run-A",
        plugin_name="source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-A",
        schema_config=_OBSERVED_SCHEMA,
    )
    row = factory.data_flow.create_row(
        "run-A",
        "source-A",
        row_index=0,
        data={"value": 1},
        row_id="row-A",
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row.row_id, token_id="token-A")
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")
    return factory, row.row_id, token.token_id


def _normalized_tokens_shape(engine: Engine) -> dict[str, object]:
    inspector = inspect(engine)

    def _normalize(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[tuple[object, ...]]:
        def _value(row: dict[str, Any], key: str) -> object:
            value = row.get(key)
            return str(value) if key == "type" else value

        return sorted(tuple(_value(row, key) for key in keys) for row in rows)

    return {
        "columns": _normalize(
            inspector.get_columns("tokens"),
            ("name", "type", "nullable", "default", "primary_key"),
        ),
        "foreign_keys": _normalize(
            inspector.get_foreign_keys("tokens"),
            ("constrained_columns", "referred_table", "referred_columns"),
        ),
        "unique_constraints": _normalize(
            inspector.get_unique_constraints("tokens"),
            ("column_names",),
        ),
        "indexes": _normalize(
            inspector.get_indexes("tokens"),
            ("name", "column_names", "unique"),
        ),
        "primary_key": inspector.get_pk_constraint("tokens").get("constrained_columns"),
    }


def _sqlite_file_snapshot(db_path: Path) -> tuple[int, list[tuple[object, ...]]]:
    """Return the exact persisted epoch and declared schema for mutation ratchets."""
    with sqlite3.connect(db_path) as conn:
        epoch = int(conn.execute("PRAGMA user_version").fetchone()[0])
        schema = conn.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_schema
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
    return epoch, schema


def _token_row_run_fk(engine: Engine) -> list[tuple[str, str]]:
    """Return the ordered row/run composite FK declared by SQLite."""
    with engine.connect() as conn:
        rows = conn.exec_driver_sql("PRAGMA foreign_key_list(tokens)").mappings().all()
    grouped: dict[int, list[tuple[int, str, str]]] = {}
    for row in rows:
        if row["table"] == "rows":
            grouped.setdefault(int(row["id"]), []).append((int(row["seq"]), str(row["from"]), str(row["to"])))
    for parts in grouped.values():
        ordered = sorted(parts)
        if [(source, target) for _, source, target in ordered] == [("row_id", "row_id"), ("run_id", "run_id")]:
            return [(source, target) for _, source, target in ordered]
    return []


class _InjectedMigrationCursor:
    """Small DBAPI cursor fake for migration-cleanup exception ordering."""

    def __init__(self, *, locked_epoch: int, body_error: Exception | None, restore_error: Exception | None) -> None:
        self.locked_epoch = locked_epoch
        self.body_error = body_error
        self.restore_error = restore_error
        self.last_statement = ""
        self.foreign_keys_enabled = True

    def execute(self, statement: str, _parameters: object = None) -> _InjectedMigrationCursor:
        self.last_statement = " ".join(statement.split())
        if self.last_statement == "PRAGMA foreign_keys = OFF":
            self.foreign_keys_enabled = False
        if self.last_statement == "PRAGMA foreign_keys = ON" and self.restore_error is not None:
            raise self.restore_error
        if self.last_statement == "PRAGMA foreign_keys = ON":
            self.foreign_keys_enabled = True
        if self.last_statement.startswith("CREATE TABLE tokens_epoch_24") and self.body_error is not None:
            raise self.body_error
        return self

    def fetchone(self) -> tuple[int] | None:
        if self.last_statement == "PRAGMA foreign_keys":
            return (int(self.foreign_keys_enabled),)
        if self.last_statement == "PRAGMA user_version":
            return (self.locked_epoch,)
        if self.last_statement.startswith("SELECT t.token_id"):
            return None
        return None

    def fetchall(self) -> list[tuple[object, ...]]:
        return []


class _InjectedMigrationConnection:
    """Pool-proxy fake that records whether an uncertain connection is invalidated."""

    def __init__(
        self,
        *,
        locked_epoch: int,
        body_error: Exception | None = None,
        rollback_error: Exception | None = None,
        restore_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._cursor = _InjectedMigrationCursor(
            locked_epoch=locked_epoch,
            body_error=body_error,
            restore_error=restore_error,
        )
        self.rollback_error = rollback_error
        self.close_error = close_error
        self.invalidated = False
        self.closed = False

    def cursor(self) -> _InjectedMigrationCursor:
        return self._cursor

    def rollback(self) -> None:
        if self.rollback_error is not None:
            raise self.rollback_error

    def commit(self) -> None:
        return None

    def invalidate(self) -> None:
        self.invalidated = True

    def close(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _InjectedMigrationEngine:
    def __init__(self, connection: _InjectedMigrationConnection) -> None:
        self.connection = connection

    def raw_connection(self) -> _InjectedMigrationConnection:
        return self.connection


class _InjectedInspector:
    def get_table_names(self) -> list[str]:
        return ["tokens"]


def _injected_migration_db(
    monkeypatch: pytest.MonkeyPatch,
    connection: _InjectedMigrationConnection,
) -> LandscapeDB:
    engine = _InjectedMigrationEngine(connection)
    db = LandscapeDB._from_parts("sqlite:///injected.db", engine)  # type: ignore[arg-type]
    monkeypatch.setattr(db, "_get_sqlite_schema_epoch", lambda: 23)
    monkeypatch.setattr(db, "_validate_schema", lambda **_kwargs: None)
    monkeypatch.setattr("sqlalchemy.inspect", lambda _engine: _InjectedInspector())
    return db


def test_current_epoch_preserves_epoch_24_token_row_run_ownership_for_sqlite_and_postgres() -> None:
    assert SQLITE_SCHEMA_EPOCH == 26
    assert (
        "tokens",
        ("row_id", "run_id"),
        "rows",
        ("row_id", "run_id"),
    ) in _REQUIRED_COMPOSITE_FOREIGN_KEYS

    postgres_ddl = str(CreateTable(tokens_table).compile(dialect=postgresql.dialect()))
    assert "FOREIGN KEY(row_id, run_id) REFERENCES rows (row_id, run_id)" in postgres_ddl


def test_fresh_sqlite_rejects_cross_run_token_row_pair() -> None:
    setup = make_recorder_with_run(run_id="run-A", source_node_id="source-A")
    row = setup.data_flow.create_row(
        "run-A",
        "source-A",
        row_index=0,
        data={"value": 1},
        row_id="row-A",
        source_row_index=0,
        ingest_sequence=0,
    )
    setup.factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")

    with pytest.raises(IntegrityError), setup.db.engine.begin() as conn:
        conn.execute(
            tokens_table.insert().values(
                token_id="forged-token",
                row_id=row.row_id,
                run_id="run-B",
                created_at=datetime.now(UTC),
            )
        )


def test_read_path_rejects_legacy_forged_token_run_mismatch() -> None:
    setup = make_recorder_with_run(run_id="run-A", source_node_id="source-A")
    row = setup.data_flow.create_row(
        "run-A",
        "source-A",
        row_index=0,
        data={"value": 1},
        row_id="row-A",
        source_row_index=0,
        ingest_sequence=0,
    )
    token = setup.data_flow.create_token(row.row_id, token_id="token-A")
    setup.factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")
    _forge_token_run(setup.db.engine, token_id=token.token_id, forged_run_id="run-B")

    with pytest.raises(AuditIntegrityError, match=r"token-A.*row-A.*run-A.*run-B"):
        setup.data_flow._resolve_token_ownership(token.token_id)


def test_epoch_23_database_migrates_atomically_and_matches_fresh_schema(tmp_path: Path) -> None:
    migrated_path = tmp_path / "migrated.db"
    migrated_url = f"sqlite:///{migrated_path}"
    predecessor = LandscapeDB(migrated_url)
    _factory, row_id, token_id = _populate_two_runs(predecessor)
    predecessor.close()

    predecessor_engine = create_engine(migrated_url)
    _rewrite_tokens_as_epoch_23(predecessor_engine)
    predecessor_engine.dispose()
    with sqlite3.connect(migrated_path) as connection:
        assert (
            connection.execute("SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'elspeth_schema_identity'").fetchone() is None
        )

    migrated = LandscapeDB.from_url(migrated_url)
    try:
        with migrated.engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA user_version").scalar_one() == 26
            token_row = conn.exec_driver_sql(
                "SELECT row_id, run_id FROM tokens WHERE token_id = ?",
                (token_id,),
            ).one()
            assert tuple(token_row) == (row_id, "run-A")
            assert conn.exec_driver_sql("PRAGMA foreign_key_check").fetchall() == []
            identity_epoch = conn.exec_driver_sql("SELECT schema_epoch FROM elspeth_schema_identity").scalar_one()
            assert identity_epoch == 26
        assert probe_schema_shape(migrated.engine) is LandscapeSchemaShape.MATCHES

        fresh = LandscapeDB(f"sqlite:///{tmp_path / 'fresh.db'}")
        try:
            assert _normalized_tokens_shape(migrated.engine) == _normalized_tokens_shape(fresh.engine)
        finally:
            fresh.close()
    finally:
        migrated.close()


def test_epoch_23_concurrent_openers_handle_predecessor_validation_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opener that observed 23 accepts a later epoch-24 validation race."""
    db_path = tmp_path / "concurrent-open.db"
    url = f"sqlite:///{db_path}"
    predecessor = LandscapeDB(url)
    predecessor.close()
    predecessor_engine = create_engine(url)
    _rewrite_tokens_as_epoch_23(predecessor_engine)
    predecessor_engine.dispose()

    original_get_epoch = LandscapeDB._get_sqlite_schema_epoch
    opener_b_read_epoch_23 = threading.Event()
    opener_a_finished = threading.Event()
    opener_b_waited = False
    wait_guard = threading.Lock()

    def _synchronized_get_epoch(db: LandscapeDB) -> int:
        nonlocal opener_b_waited
        epoch = original_get_epoch(db)
        if threading.current_thread().name == "epoch-24-opener-b" and epoch == 23:
            with wait_guard:
                should_wait = not opener_b_waited
                opener_b_waited = True
            if should_wait:
                opener_b_read_epoch_23.set()
                if not opener_a_finished.wait(timeout=15):
                    raise AssertionError("opener A did not complete the epoch-24 migration")
        return epoch

    monkeypatch.setattr(LandscapeDB, "_get_sqlite_schema_epoch", _synchronized_get_epoch)
    opened: list[LandscapeDB] = []
    errors: list[BaseException] = []
    result_guard = threading.Lock()

    def _open(*, signal_finished: bool) -> None:
        try:
            db = LandscapeDB.from_url(url)
            with result_guard:
                opened.append(db)
        except BaseException as exc:  # pragma: no cover - asserted through errors
            with result_guard:
                errors.append(exc)
        finally:
            if signal_finished:
                opener_a_finished.set()

    opener_b = threading.Thread(target=_open, kwargs={"signal_finished": False}, name="epoch-24-opener-b")
    opener_b.start()
    assert opener_b_read_epoch_23.wait(timeout=15)
    opener_a = threading.Thread(target=_open, kwargs={"signal_finished": True}, name="epoch-24-opener-a")
    opener_a.start()
    opener_a.join(timeout=15)
    opener_b.join(timeout=15)

    try:
        assert not opener_a.is_alive()
        assert not opener_b.is_alive()
        assert errors == []
        assert len(opened) == 2
        with opened[0].engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA user_version").scalar_one() == 26
        assert _token_row_run_fk(opened[0].engine) == [("row_id", "row_id"), ("run_id", "run_id")]
    finally:
        for db in opened:
            db.close()


def test_epoch_23_migration_succeeds_with_single_connection_queue_pool(tmp_path: Path) -> None:
    """Predecessor validation must not request a second pooled connection under the write lock."""
    db_path = tmp_path / "single-connection-pool.db"
    url = f"sqlite:///{db_path}"
    predecessor = LandscapeDB(url)
    predecessor.close()
    predecessor_engine = create_engine(url)
    _rewrite_tokens_as_epoch_23(predecessor_engine)
    predecessor_engine.dispose()

    migrated = LandscapeDB.from_url(
        url,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.25,
    )
    try:
        assert isinstance(migrated.engine.pool, QueuePool)
        with migrated.engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA user_version").scalar_one() == 26
        assert _token_row_run_fk(migrated.engine) == [("row_id", "row_id"), ("run_id", "run_id")]
    finally:
        migrated.close()


def test_epoch_23_predecessor_programming_failure_is_not_swallowed_by_epoch_24_race_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _InjectedMigrationConnection(locked_epoch=23)
    db = _injected_migration_db(monkeypatch, connection)
    observed_epochs = iter((23, 24))
    programming_error = RuntimeError("injected predecessor validator programming failure")
    monkeypatch.setattr(db, "_get_sqlite_schema_epoch", lambda: next(observed_epochs))

    def _raise_programming_error(**_kwargs: object) -> None:
        raise programming_error

    monkeypatch.setattr(db, "_validate_schema", _raise_programming_error)

    with pytest.raises(RuntimeError, match="validator programming failure") as raised:
        db._migrate_sqlite_schema()

    assert raised.value is programming_error
    assert next(observed_epochs) == 24
    assert connection.closed is False


@pytest.mark.parametrize("read_only", [False, True], ids=["inspection", "read-only"])
def test_epoch_23_non_schema_managing_open_is_exactly_non_mutating(
    tmp_path: Path,
    read_only: bool,
) -> None:
    db_path = tmp_path / f"epoch-23-{'ro' if read_only else 'inspect'}.db"
    url = f"sqlite:///{db_path}"
    predecessor = LandscapeDB(url)
    predecessor.close()
    predecessor_engine = create_engine(url)
    _rewrite_tokens_as_epoch_23(predecessor_engine)
    predecessor_engine.dispose()
    before = _sqlite_file_snapshot(db_path)

    with pytest.raises(SchemaCompatibilityError, match=r"epoch.*23|composite foreign key"):
        LandscapeDB.from_url(url, create_tables=False, read_only=read_only)

    assert _sqlite_file_snapshot(db_path) == before
    assert before[0] == 23


def test_epoch_23_mid_rebuild_failure_rolls_back_schema_and_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "mid-rebuild-failure.db"
    url = f"sqlite:///{db_path}"
    predecessor = LandscapeDB(url)
    predecessor.close()
    predecessor_engine = create_engine(url)
    _rewrite_tokens_as_epoch_23(predecessor_engine)
    predecessor_engine.dispose()
    before = _sqlite_file_snapshot(db_path)
    monkeypatch.setattr(
        landscape_database,
        "_SQLITE_EPOCH_24_TOKEN_INDEX_DDL",
        (
            *landscape_database._SQLITE_EPOCH_24_TOKEN_INDEX_DDL,
            ("ix_tokens_injected_failure", "CREATE INDEX ix_tokens_injected_failure ON tokens (missing_column)"),
        ),
    )

    with pytest.raises(sqlite3.OperationalError, match="missing_column"):
        LandscapeDB(url)

    assert _sqlite_file_snapshot(db_path) == before
    assert before[0] == 23
    with sqlite3.connect(db_path) as conn:
        table_names = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")}
    assert "tokens_epoch_24" not in table_names


def test_epoch_23_static_pool_mid_rebuild_failure_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file-backed StaticPool must not let validation roll back the migration transaction."""
    db_path = tmp_path / "static-pool-mid-rebuild-failure.db"
    url = f"sqlite:///{db_path}"
    predecessor = LandscapeDB(url)
    predecessor.close()
    predecessor_engine = create_engine(url)
    _rewrite_tokens_as_epoch_23(predecessor_engine)
    predecessor_engine.dispose()
    before = _sqlite_file_snapshot(db_path)
    monkeypatch.setattr(
        landscape_database,
        "_SQLITE_EPOCH_24_TOKEN_INDEX_DDL",
        (
            *landscape_database._SQLITE_EPOCH_24_TOKEN_INDEX_DDL,
            ("ix_tokens_injected_failure", "CREATE INDEX ix_tokens_injected_failure ON tokens (missing_column)"),
        ),
    )

    with pytest.raises(sqlite3.OperationalError, match="missing_column"):
        LandscapeDB.from_url(
            url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )

    assert _sqlite_file_snapshot(db_path) == before
    with sqlite3.connect(db_path) as conn:
        assert int(conn.execute("PRAGMA user_version").fetchone()[0]) == 23
        table_names = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_schema WHERE type = 'table'")}
        foreign_keys = conn.execute("PRAGMA foreign_key_list(tokens)").fetchall()
    assert "tokens_epoch_24" not in table_names
    assert not any(str(row[2]) == "rows" and str(row[3]) == "run_id" for row in foreign_keys)


def test_epoch_23_migration_cleanup_preserves_primary_error_and_invalidates_uncertain_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body_error = RuntimeError("injected migration body failure")
    connection = _InjectedMigrationConnection(
        locked_epoch=23,
        body_error=body_error,
        rollback_error=RuntimeError("injected rollback failure"),
        restore_error=RuntimeError("injected foreign-key restore failure"),
    )
    db = _injected_migration_db(monkeypatch, connection)

    with pytest.raises(RuntimeError, match="injected migration body failure") as raised:
        db._migrate_sqlite_schema()

    assert raised.value is body_error
    assert any("rollback" in note for note in getattr(raised.value, "__notes__", []))
    assert any("foreign-key" in note for note in getattr(raised.value, "__notes__", []))
    assert connection.invalidated is True
    assert connection.closed is True


def test_epoch_24_restore_failure_after_success_invalidates_and_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _InjectedMigrationConnection(
        locked_epoch=24,
        restore_error=RuntimeError("injected foreign-key restore failure"),
    )
    db = _injected_migration_db(monkeypatch, connection)

    with pytest.raises(AuditIntegrityError, match="restore SQLite foreign-key enforcement") as raised:
        db._migrate_sqlite_schema()

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert connection.invalidated is True
    assert connection.closed is True


def test_epoch_23_close_failure_preserves_primary_error_and_invalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body_error = RuntimeError("injected migration body failure")
    connection = _InjectedMigrationConnection(
        locked_epoch=23,
        body_error=body_error,
        close_error=RuntimeError("injected connection close failure"),
    )
    db = _injected_migration_db(monkeypatch, connection)

    with pytest.raises(RuntimeError, match="injected migration body failure") as raised:
        db._migrate_sqlite_schema()

    assert raised.value is body_error
    assert any("connection close" in note for note in getattr(raised.value, "__notes__", []))
    assert connection.invalidated is True
    assert connection.closed is True


def test_epoch_24_close_failure_after_success_invalidates_and_reports_close_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_error = RuntimeError("injected connection close failure")
    connection = _InjectedMigrationConnection(
        locked_epoch=24,
        close_error=close_error,
    )
    db = _injected_migration_db(monkeypatch, connection)

    with pytest.raises(AuditIntegrityError, match=r"close.*SQLite connection") as raised:
        db._migrate_sqlite_schema()

    assert raised.value.__cause__ is close_error
    assert "foreign-key" not in str(raised.value)
    assert connection.invalidated is True
    assert connection.closed is True


def test_epoch_23_migration_preserves_declared_dependent_indexes_and_triggers(tmp_path: Path) -> None:
    """The table rebuild must not silently discard compatible local objects."""
    db_path = tmp_path / "dependent-schema.db"
    url = f"sqlite:///{db_path}"
    predecessor = LandscapeDB(url)
    predecessor.close()

    predecessor_engine = create_engine(url)
    _rewrite_tokens_as_epoch_23(predecessor_engine)
    with predecessor_engine.begin() as conn:
        conn.exec_driver_sql("CREATE INDEX ix_tokens_branch_custom ON tokens (branch_name)")
        conn.exec_driver_sql("CREATE TRIGGER trg_tokens_noop AFTER INSERT ON tokens BEGIN SELECT 1; END")
    predecessor_engine.dispose()

    migrated = LandscapeDB.from_url(url)
    try:
        assert "ix_tokens_branch_custom" in {str(index["name"]) for index in inspect(migrated.engine).get_indexes("tokens")}
        with migrated.engine.connect() as conn:
            trigger_sql = conn.exec_driver_sql(
                "SELECT sql FROM sqlite_schema WHERE type = 'trigger' AND name = 'trg_tokens_noop'"
            ).scalar_one_or_none()
        assert trigger_sql is not None
    finally:
        migrated.close()


def test_epoch_23_migration_rejects_forged_ownership_without_partial_commit(tmp_path: Path) -> None:
    db_path = tmp_path / "corrupt-predecessor.db"
    url = f"sqlite:///{db_path}"
    predecessor = LandscapeDB(url)
    _factory, _row_id, token_id = _populate_two_runs(predecessor)
    predecessor.close()

    predecessor_engine = create_engine(url)
    _rewrite_tokens_as_epoch_23(predecessor_engine)
    _forge_token_run(predecessor_engine, token_id=token_id, forged_run_id="run-B")
    predecessor_engine.dispose()

    with pytest.raises(AuditIntegrityError, match=r"token-A.*row-A.*run-A.*run-B"):
        LandscapeDB(url)

    check = create_engine(url)
    try:
        with check.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA user_version").scalar_one() == 23
        assert "tokens_epoch_24" not in inspect(check).get_table_names()
    finally:
        check.dispose()


def test_epoch_23_migration_refuses_other_stale_shape_without_bumping_epoch(tmp_path: Path) -> None:
    """The migration allowlist exempts only its new composite FK."""
    db_path = tmp_path / "stale-predecessor.db"
    url = f"sqlite:///{db_path}"
    predecessor = LandscapeDB(url)
    predecessor.close()

    predecessor_engine = create_engine(url)
    _rewrite_tokens_as_epoch_23(predecessor_engine)
    with predecessor_engine.begin() as conn:
        conn.exec_driver_sql("DROP INDEX ix_checkpoints_run_sequence_unique")
    predecessor_engine.dispose()

    with pytest.raises(SchemaCompatibilityError, match=r"checkpoints.*index"):
        LandscapeDB(url)

    check = create_engine(url)
    try:
        with check.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA user_version").scalar_one() == 23
        assert "ix_checkpoints_run_sequence_unique" not in {str(index["name"]) for index in inspect(check).get_indexes("checkpoints")}
        assert "tokens_epoch_24" not in inspect(check).get_table_names()
    finally:
        check.dispose()
