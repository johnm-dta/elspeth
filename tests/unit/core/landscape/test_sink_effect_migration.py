"""SQLite epoch-25 to epoch-26 sink-effect migration proofs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from elspeth.core.landscape import database as landscape_database
from elspeth.core.landscape.database import LandscapeDB
from tests.unit.core.landscape.test_artifact_idempotency_migration import _epoch_and_schema, _identity_epochs, _seed_current_database


def _seed_operation_and_call(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO operations (
                operation_id, run_id, node_id, operation_type, started_at,
                completed_at, status, input_data_ref, input_data_hash,
                output_data_ref, output_data_hash, error_message, duration_ms
            ) VALUES (
                'operation-legacy', 'run-artifact-migration',
                'sink-artifact-migration', 'sink_write', CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP, 'completed', 'input-ref', 'input-hash',
                'output-ref', 'output-hash', NULL, 12.5
            )
            """
        )
        connection.execute(
            """
            INSERT INTO calls (
                call_id, state_id, operation_id, call_index, call_type, status,
                request_hash, request_ref, response_hash, response_ref,
                error_json, latency_ms, created_at
            ) VALUES (
                'call-legacy', NULL, 'operation-legacy', 0, 'http', 'success',
                'request-hash', 'request-ref', 'response-hash', 'response-ref',
                NULL, 10.0, CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()


def _schema_and_rows(
    db_path: Path,
) -> tuple[int, list[tuple[object, ...]], list[tuple[object, ...]], list[tuple[object, ...]], tuple[int, ...] | None]:
    epoch, schema = _epoch_and_schema(db_path)
    with sqlite3.connect(db_path) as connection:
        operations = connection.execute("SELECT * FROM operations ORDER BY operation_id").fetchall()
        artifacts = connection.execute("SELECT * FROM artifacts ORDER BY artifact_id").fetchall()
    return epoch, schema, operations, artifacts, _identity_epochs(db_path)


def test_epoch_25_migrates_to_26_and_preserves_legacy_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "epoch25.db"
    url = _seed_current_database(db_path)
    _seed_operation_and_call(db_path)

    db = LandscapeDB(url)
    try:
        with db.engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA user_version").scalar_one() == 26
            artifact = connection.exec_driver_sql(
                """
                SELECT produced_by_state_id, sink_effect_id,
                       publication_performed, publication_evidence_kind
                FROM artifacts WHERE artifact_id = 'artifact-migration-original'
                """
            ).one()
            assert tuple(artifact) == ("state-artifact-migration", None, True, "legacy_returned")
            operation = connection.exec_driver_sql(
                """
                SELECT operation_id, sink_effect_id, input_data_ref,
                       output_data_ref, duration_ms
                FROM operations WHERE operation_id = 'operation-legacy'
                """
            ).one()
            assert tuple(operation) == ("operation-legacy", None, "input-ref", "output-ref", 12.5)
            call = connection.exec_driver_sql("SELECT operation_id, call_index FROM calls WHERE call_id = 'call-legacy'").one()
            assert tuple(call) == ("operation-legacy", 0)
            tables = {
                str(row[0])
                for row in connection.exec_driver_sql("SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            }
            assert {
                "sink_effect_streams",
                "sink_effects",
                "sink_effect_members",
                "sink_effect_attempts",
                "audit_export_snapshots",
                "audit_export_snapshot_chunks",
                "sink_effect_export_snapshots",
            } <= tables
            witness = connection.exec_driver_sql("PRAGMA index_list('runs')").fetchall()
            exact = next(row for row in witness if row[1] == "uq_runs_export_witness")
            assert (exact[2], exact[4]) == (1, 0)
            witness_columns = [
                row[2] for row in connection.exec_driver_sql("PRAGMA index_xinfo('uq_runs_export_witness')").fetchall() if row[5] == 1
            ]
            assert witness_columns == ["run_id", "status", "completed_at"]
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").fetchall() == []
            assert connection.exec_driver_sql("SELECT schema_epoch FROM elspeth_schema_identity").scalar_one() == 26
    finally:
        db.close()


def test_epoch_25_to_26_failure_rolls_back_every_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "rollback.db"
    url = _seed_current_database(db_path)
    _seed_operation_and_call(db_path)
    before = _schema_and_rows(db_path)
    original = landscape_database._compile_sqlite_table_ddl

    def _fail_after_operation_copy(table, *, replacement_name=None):  # type: ignore[no-untyped-def]
        if table.name == "artifacts" and replacement_name == "artifacts_epoch_26":
            raise RuntimeError("injected epoch-26 artifact rebuild failure")
        return original(table, replacement_name=replacement_name)

    monkeypatch.setattr(landscape_database, "_compile_sqlite_table_ddl", _fail_after_operation_copy)
    with pytest.raises(RuntimeError, match="artifact rebuild failure"):
        LandscapeDB(url)

    assert _schema_and_rows(db_path) == before
    assert before[0] == 25


def test_epoch_26_refuses_divergent_new_table_before_stamping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "divergent-new-table.db"
    url = _seed_current_database(db_path)
    before = _schema_and_rows(db_path)
    original = landscape_database._compile_sqlite_table_ddl
    injected = False

    def _inject_extra_column_once(table, *, replacement_name=None):  # type: ignore[no-untyped-def]
        nonlocal injected
        statement = original(table, replacement_name=replacement_name)
        if table.name == "sink_effect_attempts" and replacement_name is None and not injected:
            injected = True
            return statement.replace("(\n", "(\n\tsurprise INTEGER, \n", 1)
        return statement

    monkeypatch.setattr(landscape_database, "_compile_sqlite_table_ddl", _inject_extra_column_once)
    with pytest.raises(Exception, match=r"physical DDL mismatch.*sink_effect_attempts"):
        LandscapeDB(url)

    assert _schema_and_rows(db_path) == before
    assert before[0] == 25


def test_epoch_25_malformed_predecessor_is_refused_without_mutation(tmp_path: Path) -> None:
    db_path = tmp_path / "malformed.db"
    url = _seed_current_database(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP INDEX ix_operations_node_run")
        connection.commit()
    before = _schema_and_rows(db_path)

    with pytest.raises(Exception, match=r"operations|index"):
        LandscapeDB(url)

    assert _schema_and_rows(db_path) == before
