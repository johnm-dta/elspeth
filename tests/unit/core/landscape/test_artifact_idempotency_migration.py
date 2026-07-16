"""Epoch-25 SQLite migration proofs for artifact logical-effect identity."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import Engine, create_engine, inspect
from sqlalchemy.pool import QueuePool, StaticPool

from elspeth.contracts.enums import NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape import database as landscape_database
from elspeth.core.landscape.database import LandscapeDB, LandscapeSchemaShape, SchemaCompatibilityError, probe_schema_shape
from elspeth.core.landscape.factory import RecorderFactory
from tests.unit.core.landscape.test_token_ownership_run_scope import _rewrite_tokens_as_epoch_23

_ARTIFACT_INDEX = "uq_artifacts_run_idempotency_key"
_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})

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


def _rewrite_current_as_epoch_25(db_path: Path) -> None:
    """Create the genuine physical predecessor used by migration tests."""
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(_LEGACY_OPERATIONS_DDL)
        conn.execute(
            """
            INSERT INTO operations_epoch_25
            SELECT operation_id, run_id, node_id, operation_type, started_at,
                   completed_at, status, input_data_ref, input_data_hash,
                   output_data_ref, output_data_hash, error_message, duration_ms
            FROM operations
            """
        )
        conn.execute("DROP TABLE operations")
        conn.execute("ALTER TABLE operations_epoch_25 RENAME TO operations")
        conn.execute("CREATE INDEX ix_operations_node_run ON operations(node_id, run_id)")
        conn.execute("CREATE INDEX ix_operations_run_id ON operations(run_id)")

        conn.execute(_LEGACY_ARTIFACTS_DDL)
        conn.execute(
            """
            INSERT INTO artifacts_epoch_25
            SELECT artifact_id, run_id, produced_by_state_id, sink_node_id,
                   artifact_type, path_or_uri, content_hash, size_bytes,
                   idempotency_key, created_at
            FROM artifacts
            """
        )
        conn.execute("DROP TABLE artifacts")
        conn.execute("ALTER TABLE artifacts_epoch_25 RENAME TO artifacts")
        conn.execute("CREATE INDEX ix_artifacts_run ON artifacts(run_id)")
        conn.execute(
            "CREATE UNIQUE INDEX uq_artifacts_run_idempotency_key ON artifacts(run_id, idempotency_key) WHERE idempotency_key IS NOT NULL"
        )

        conn.execute("DROP INDEX uq_runs_export_witness")
        conn.execute("DROP INDEX uq_tokens_identity_row_run")
        for table_name in (
            "sink_effect_attempts",
            "sink_effect_export_snapshots",
            "sink_effect_members",
            "sink_effects",
            "sink_effect_streams",
            "audit_export_snapshot_chunks",
            "audit_export_snapshots",
        ):
            conn.execute(f"DROP TABLE {table_name}")
        identity_update = conn.execute(
            """
            UPDATE elspeth_schema_identity
            SET schema_epoch = 25
            WHERE singleton_id = 1
              AND application_id = 'elspeth'
              AND store_kind = 'landscape'
              AND schema_epoch = 26
            """
        )
        assert identity_update.rowcount == 1
        conn.execute("PRAGMA user_version = 25")
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")


def _seed_current_database(db_path: Path) -> str:
    url = f"sqlite:///{db_path}"
    db = LandscapeDB(url)
    try:
        factory = RecorderFactory(db)
        run = factory.run_lifecycle.begin_run(
            config={},
            canonical_version="v1",
            run_id="run-artifact-migration",
            openrouter_catalog_sha256="0" * 64,
            openrouter_catalog_source="bundled",
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-artifact-migration",
            schema_config=_SCHEMA,
        )
        factory.data_flow.register_node(
            run_id=run.run_id,
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-artifact-migration",
            schema_config=_SCHEMA,
        )
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id="source-artifact-migration",
            row_index=0,
            data={"value": 1},
            row_id="row-artifact-migration",
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id, token_id="token-artifact-migration")
        state = factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id="sink-artifact-migration",
            run_id=run.run_id,
            step_index=0,
            input_data={"value": 1},
            state_id="state-artifact-migration",
        )
        with db.engine.begin() as connection:
            connection.exec_driver_sql(
                """
                INSERT INTO artifacts (
                    artifact_id, run_id, produced_by_state_id, sink_effect_id,
                    sink_node_id, artifact_type, path_or_uri, content_hash,
                    size_bytes, idempotency_key, publication_performed,
                    publication_evidence_kind, created_at
                ) VALUES (?, ?, ?, NULL, ?, 'csv', '/output/migration.csv',
                          'sha256:migration', 128, ?, 1, 'returned', CURRENT_TIMESTAMP)
                """,
                (
                    "artifact-migration-original",
                    run.run_id,
                    state.state_id,
                    "sink-artifact-migration",
                    "run-artifact-migration:row-artifact-migration:csv_sink",
                ),
            )
    finally:
        db.close()
    _rewrite_current_as_epoch_25(db_path)
    return url


def _rewrite_as_epoch_24(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.exec_driver_sql(f"DROP INDEX {_ARTIFACT_INDEX}")
        result = conn.exec_driver_sql(
            """
            UPDATE elspeth_schema_identity
            SET schema_epoch = 24
            WHERE singleton_id = 1
              AND application_id = 'elspeth'
              AND store_kind = 'landscape'
              AND schema_epoch = 25
            """
        )
        assert result.rowcount == 1
        conn.exec_driver_sql("PRAGMA user_version = 24")


def _identity_epochs(db_path: Path) -> tuple[int, ...] | None:
    """Return the physical identity stamp, or ``None`` before epoch 24."""
    with sqlite3.connect(db_path) as conn:
        table_exists = conn.execute("SELECT 1 FROM sqlite_schema WHERE type = 'table' AND name = 'elspeth_schema_identity'").fetchone()
        if table_exists is None:
            return None
        return tuple(int(row[0]) for row in conn.execute("SELECT schema_epoch FROM elspeth_schema_identity ORDER BY singleton_id"))


def _epoch_and_schema(db_path: Path) -> tuple[int, list[tuple[object, ...]]]:
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


def _artifact_indexes(engine: Engine) -> set[str]:
    return {str(index["name"]) for index in inspect(engine).get_indexes("artifacts")}


def _artifact_index_shapes(engine: Engine) -> set[tuple[str, tuple[str, ...], bool, str]]:
    return {
        (
            str(index["name"]),
            tuple(str(column) for column in index["column_names"]),
            bool(index["unique"]),
            str(index.get("dialect_options", {}).get("sqlite_where", "")),
        )
        for index in inspect(engine).get_indexes("artifacts")
    }


class _Epoch25FailureCursorProxy:
    def __init__(self, cursor: sqlite3.Cursor, statements: list[str]) -> None:
        self._cursor = cursor
        self._statements = statements

    def execute(self, statement: str, parameters: Any = ()) -> _Epoch25FailureCursorProxy:
        normalized = " ".join(statement.split())
        self._cursor.execute(statement, parameters)
        self._statements.append(normalized)
        if normalized == "PRAGMA user_version = 25":
            raise sqlite3.OperationalError("injected failure after epoch-25 index creation")
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class _Epoch25FailureConnectionProxy:
    def __init__(self, connection: sqlite3.Connection, statements: list[str]) -> None:
        self.connection = connection
        self._statements = statements

    @property
    def isolation_level(self) -> str | None:
        return self.connection.isolation_level

    @isolation_level.setter
    def isolation_level(self, value: str | None) -> None:
        self.connection.isolation_level = value

    def cursor(self) -> _Epoch25FailureCursorProxy:
        return _Epoch25FailureCursorProxy(self.connection.cursor(), self._statements)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.connection, name)


def _epoch_25_failing_creator(
    db_path: Path,
    statements: list[str],
    connections: list[_Epoch25FailureConnectionProxy],
) -> _Epoch25FailureConnectionProxy:
    connection = _Epoch25FailureConnectionProxy(sqlite3.connect(db_path, check_same_thread=False), statements)
    connections.append(connection)
    return connection


def test_epoch_24_migrates_to_25_and_preserves_artifact(tmp_path: Path) -> None:
    db_path = tmp_path / "forward.db"
    url = _seed_current_database(db_path)
    predecessor = create_engine(url)
    _rewrite_as_epoch_24(predecessor)
    predecessor.dispose()
    assert _identity_epochs(db_path) == (24,)

    migrated = LandscapeDB(url)
    try:
        with migrated.engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA user_version").scalar_one() == 26
            artifact = conn.exec_driver_sql("SELECT artifact_id, content_hash FROM artifacts WHERE idempotency_key IS NOT NULL").one()
        assert tuple(artifact) == ("artifact-migration-original", "sha256:migration")
        assert _ARTIFACT_INDEX in _artifact_indexes(migrated.engine)
        assert probe_schema_shape(migrated.engine) is LandscapeSchemaShape.MATCHES
        assert _identity_epochs(db_path) == (26,)
    finally:
        migrated.close()


def test_epoch_23_migrates_sequentially_through_24_to_25(tmp_path: Path) -> None:
    db_path = tmp_path / "sequential.db"
    url = _seed_current_database(db_path)
    predecessor = create_engine(url)
    _rewrite_as_epoch_24(predecessor)
    _rewrite_tokens_as_epoch_23(predecessor)
    predecessor.dispose()
    assert _identity_epochs(db_path) is None

    migrated = LandscapeDB(url)
    try:
        with migrated.engine.connect() as conn:
            assert conn.exec_driver_sql("PRAGMA user_version").scalar_one() == 26
            token_fks = conn.exec_driver_sql("PRAGMA foreign_key_list(tokens)").fetchall()
        assert any(str(row[2]) == "rows" and str(row[3]) == "run_id" and str(row[4]) == "run_id" for row in token_fks)
        assert _ARTIFACT_INDEX in _artifact_indexes(migrated.engine)
        assert probe_schema_shape(migrated.engine) is LandscapeSchemaShape.MATCHES
        assert _identity_epochs(db_path) == (26,)
    finally:
        migrated.close()


def test_epoch_24_migrated_schema_matches_fresh_schema(tmp_path: Path) -> None:
    migrated_path = tmp_path / "migrated.db"
    url = _seed_current_database(migrated_path)
    predecessor = create_engine(url)
    _rewrite_as_epoch_24(predecessor)
    predecessor.dispose()

    migrated = LandscapeDB(url)
    fresh = LandscapeDB(f"sqlite:///{tmp_path / 'fresh.db'}")
    try:
        assert probe_schema_shape(migrated.engine) is LandscapeSchemaShape.MATCHES
        assert probe_schema_shape(fresh.engine) is LandscapeSchemaShape.MATCHES
        migrated_index = _artifact_index_shapes(migrated.engine)
        fresh_index = _artifact_index_shapes(fresh.engine)
        assert migrated_index == fresh_index
    finally:
        migrated.close()
        fresh.close()


@pytest.mark.parametrize("divergent", [False, True], ids=["identical", "divergent"])
def test_epoch_24_duplicate_key_refuses_without_mutation(tmp_path: Path, divergent: bool) -> None:
    db_path = tmp_path / f"duplicate-{'divergent' if divergent else 'identical'}.db"
    url = _seed_current_database(db_path)
    predecessor = create_engine(url)
    _rewrite_as_epoch_24(predecessor)
    with predecessor.begin() as conn:
        conn.exec_driver_sql(
            """
            INSERT INTO artifacts (
                artifact_id, run_id, produced_by_state_id, sink_node_id,
                artifact_type, path_or_uri, content_hash, size_bytes,
                idempotency_key, created_at
            )
            SELECT
                'artifact-migration-duplicate', run_id, produced_by_state_id,
                sink_node_id, artifact_type, path_or_uri,
                CASE WHEN ? THEN 'sha256:divergent' ELSE content_hash END,
                size_bytes, idempotency_key, created_at
            FROM artifacts
            WHERE artifact_id = 'artifact-migration-original'
            """,
            (divergent,),
        )
    predecessor.dispose()
    before = _epoch_and_schema(db_path)

    with pytest.raises(AuditIntegrityError, match=r"duplicate.*idempotency"):
        LandscapeDB(url)

    assert _epoch_and_schema(db_path) == before
    assert before[0] == 24
    assert _identity_epochs(db_path) == (24,)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT count(*) FROM artifacts").fetchone() == (2,)


def test_epoch_24_other_shape_divergence_refuses_before_mutation(tmp_path: Path) -> None:
    db_path = tmp_path / "divergent-shape.db"
    url = _seed_current_database(db_path)
    predecessor = create_engine(url)
    _rewrite_as_epoch_24(predecessor)
    with predecessor.begin() as conn:
        conn.exec_driver_sql("DROP INDEX ix_checkpoints_run_sequence_unique")
    predecessor.dispose()
    before = _epoch_and_schema(db_path)

    with pytest.raises(SchemaCompatibilityError, match=r"checkpoints.*index"):
        LandscapeDB(url)

    assert _epoch_and_schema(db_path) == before


def test_epoch_24_concurrent_openers_converge_on_completed_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "concurrent.db"
    url = _seed_current_database(db_path)
    predecessor = create_engine(url)
    _rewrite_as_epoch_24(predecessor)
    predecessor.dispose()

    original_get_epoch = LandscapeDB._get_sqlite_schema_epoch
    opener_b_read_epoch_24 = threading.Event()
    opener_a_finished = threading.Event()
    opener_b_waited = False
    wait_guard = threading.Lock()

    def _synchronized_get_epoch(db: LandscapeDB) -> int:
        nonlocal opener_b_waited
        epoch = original_get_epoch(db)
        if threading.current_thread().name == "epoch-25-opener-b" and epoch == 24:
            with wait_guard:
                should_wait = not opener_b_waited
                opener_b_waited = True
            if should_wait:
                opener_b_read_epoch_24.set()
                if not opener_a_finished.wait(timeout=15):
                    raise AssertionError("opener A did not complete the epoch-25 migration")
        return epoch

    monkeypatch.setattr(LandscapeDB, "_get_sqlite_schema_epoch", _synchronized_get_epoch)
    opened: list[LandscapeDB] = []
    errors: list[BaseException] = []
    result_guard = threading.Lock()

    def _open(*, signal_finished: bool) -> None:
        try:
            db = LandscapeDB(url)
            with result_guard:
                opened.append(db)
        except BaseException as exc:  # pragma: no cover - asserted below
            with result_guard:
                errors.append(exc)
        finally:
            if signal_finished:
                opener_a_finished.set()

    opener_b = threading.Thread(target=_open, kwargs={"signal_finished": False}, name="epoch-25-opener-b")
    opener_b.start()
    assert opener_b_read_epoch_24.wait(timeout=15)
    opener_a = threading.Thread(target=_open, kwargs={"signal_finished": True}, name="epoch-25-opener-a")
    opener_a.start()
    opener_a.join(timeout=15)
    opener_b.join(timeout=15)

    try:
        assert not opener_a.is_alive()
        assert not opener_b.is_alive()
        assert errors == []
        assert len(opened) == 2
        assert _ARTIFACT_INDEX in _artifact_indexes(opened[0].engine)
    finally:
        for db in opened:
            db.close()


def test_epoch_24_migration_succeeds_with_single_connection_queue_pool(tmp_path: Path) -> None:
    db_path = tmp_path / "queue-size-one.db"
    url = _seed_current_database(db_path)
    predecessor = create_engine(url)
    _rewrite_as_epoch_24(predecessor)
    predecessor.dispose()

    migrated = LandscapeDB.from_url(url, pool_size=1, max_overflow=0, pool_timeout=0.25)
    try:
        assert isinstance(migrated.engine.pool, QueuePool)
        assert _ARTIFACT_INDEX in _artifact_indexes(migrated.engine)
    finally:
        migrated.close()


@pytest.mark.parametrize("read_only", [False, True], ids=["inspection", "read-only"])
def test_epoch_24_non_schema_managing_open_is_non_mutating(tmp_path: Path, read_only: bool) -> None:
    db_path = tmp_path / f"nonmutating-{'ro' if read_only else 'inspect'}.db"
    url = _seed_current_database(db_path)
    predecessor = create_engine(url)
    _rewrite_as_epoch_24(predecessor)
    predecessor.dispose()
    before = _epoch_and_schema(db_path)

    with pytest.raises(SchemaCompatibilityError, match=r"epoch.*24|idempotency.*index"):
        LandscapeDB.from_url(url, create_tables=False, read_only=read_only)

    assert _epoch_and_schema(db_path) == before


def test_epoch_24_static_pool_failure_rolls_back_index_and_epoch(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "static-pool-rollback.db"
    url = _seed_current_database(db_path)
    predecessor = create_engine(url)
    _rewrite_as_epoch_24(predecessor)
    predecessor.dispose()
    before = _epoch_and_schema(db_path)
    statements: list[str] = []
    connections: list[_Epoch25FailureConnectionProxy] = []

    def _creator() -> _Epoch25FailureConnectionProxy:
        return _epoch_25_failing_creator(db_path, statements, connections)

    expected_index_ddl = " ".join(landscape_database._SQLITE_EPOCH_25_ARTIFACT_INDEX_DDL.split())
    with pytest.raises(sqlite3.OperationalError, match="after epoch-25 index creation"):
        LandscapeDB.from_url(
            url,
            poolclass=StaticPool,
            creator=_creator,
        )

    assert expected_index_ddl in statements
    assert statements.index(expected_index_ddl) < statements.index("PRAGMA user_version = 25")
    assert _epoch_and_schema(db_path) == before
    assert before[0] == 24
    assert _identity_epochs(db_path) == (24,)


def test_epoch_24_single_connection_queue_pool_failure_rolls_back_and_returns_safe_connection(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "queue-pool-rollback.db"
    url = _seed_current_database(db_path)
    predecessor = create_engine(url)
    _rewrite_as_epoch_24(predecessor)
    predecessor.dispose()
    before = _epoch_and_schema(db_path)
    statements: list[str] = []
    connections: list[_Epoch25FailureConnectionProxy] = []

    def _creator() -> _Epoch25FailureConnectionProxy:
        return _epoch_25_failing_creator(db_path, statements, connections)

    expected_index_ddl = " ".join(landscape_database._SQLITE_EPOCH_25_ARTIFACT_INDEX_DDL.split())
    with pytest.raises(sqlite3.OperationalError, match="after epoch-25 index creation"):
        LandscapeDB.from_url(
            url,
            poolclass=QueuePool,
            pool_size=1,
            max_overflow=0,
            pool_timeout=0.25,
            creator=_creator,
        )

    assert len(connections) == 1
    assert expected_index_ddl in statements
    assert statements.index(expected_index_ddl) < statements.index("PRAGMA user_version = 25")
    assert _epoch_and_schema(db_path) == before
    physical = connections[0].connection
    assert physical.execute("PRAGMA foreign_keys").fetchone() == (1,)
    assert physical.execute("PRAGMA user_version").fetchone() == (24,)
    physical.execute("BEGIN IMMEDIATE")
    assert physical.execute("SELECT 1").fetchone() == (1,)
    physical.rollback()
