"""Connection-bound, immutable audit-export read snapshot tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import event

from elspeth.contracts import NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.export_read_model import ConnectionBoundExportReadModel, open_export_read_transaction
from elspeth.core.landscape.schema import (
    batches_table,
    operations_table,
    run_attributions_table,
    runs_table,
    secret_resolutions_table,
    token_outcomes_table,
    transform_errors_table,
    validation_errors_table,
)
from tests.fixtures.landscape import make_factory, register_test_node

COMPLETED_AT = datetime(2026, 7, 16, 2, 3, 4, 567890, tzinfo=UTC)


def _insert_run(db: LandscapeDB, *, run_id: str, status: str, completed_at: datetime | None) -> None:
    with db.engine.begin() as connection:
        connection.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=COMPLETED_AT,
                completed_at=completed_at,
                config_hash="0" * 64,
                settings_json="{}",
                canonical_version="v1",
                status=status,
                openrouter_catalog_sha256="1" * 64,
                openrouter_catalog_source="bundled",
            )
        )


def test_terminal_witness_accepts_only_immutable_export_terminal_statuses() -> None:
    db = LandscapeDB.in_memory()
    try:
        _insert_run(db, run_id="completed", status="completed", completed_at=COMPLETED_AT)
        _insert_run(db, run_id="running", status="running", completed_at=None)
        _insert_run(db, run_id="failed", status="failed", completed_at=COMPLETED_AT)
        _insert_run(db, run_id="interrupted", status="interrupted", completed_at=COMPLETED_AT)
        with db.engine.connect() as connection:
            model = ConnectionBoundExportReadModel(connection)
            witness = model.get_export_terminal_witness("completed")
            assert witness.source_run_id == "completed"
            assert witness.source_completed_at == COMPLETED_AT
            for run_id in ("running", "failed", "interrupted"):
                with pytest.raises(AuditIntegrityError, match="export-terminal"):
                    model.get_export_terminal_witness(run_id)
    finally:
        db.close()


def test_all_queries_use_the_exact_bound_connection() -> None:
    db = LandscapeDB.in_memory()
    checkouts = 0

    def observe_checkout(*_args: object) -> None:
        nonlocal checkouts
        checkouts += 1

    event.listen(db.engine, "checkout", observe_checkout)
    try:
        _insert_run(db, run_id="completed", status="completed", completed_at=COMPLETED_AT)
        with db.engine.connect() as connection:
            baseline = checkouts
            model = ConnectionBoundExportReadModel(connection)
            assert model.get_run("completed") is not None
            assert model.get_run_attribution("completed") is None
            assert model.get_nodes("completed") == []
            assert list(model.iter_rows_for_run("completed", batch_size=2)) == []
            assert model.get_artifacts("completed") == []
            assert checkouts == baseline
    finally:
        event.remove(db.engine, "checkout", observe_checkout)
        db.close()


def test_sqlite_open_read_transaction_excludes_later_committed_rows(tmp_path: Path) -> None:
    db = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'audit.db'}")
    try:
        _insert_run(db, run_id="completed", status="completed", completed_at=COMPLETED_AT)
        with open_export_read_transaction(db.engine) as model:
            assert model.dialect_name == "sqlite"
            assert model.get_export_terminal_witness("completed").source_completed_at == COMPLETED_AT
            # Force the SQLite snapshot before the concurrent writer commits.
            assert model.get_run_attribution("completed") is None
            with db.engine.begin() as writer:
                writer.execute(
                    run_attributions_table.insert().values(
                        run_id="completed",
                        recorded_at=COMPLETED_AT,
                        initiated_by_user_id="later-user",
                        auth_provider_type="local",
                    )
                )
            assert model.get_run_attribution("completed") is None
    finally:
        db.close()


def test_read_transaction_rolls_back_and_closes_on_failure(tmp_path: Path) -> None:
    db = LandscapeDB.from_url(f"sqlite:///{tmp_path / 'failure.db'}")
    connection = None
    try:
        with pytest.raises(RuntimeError, match="boom"), open_export_read_transaction(db.engine) as model:
            connection = model.connection
            raise RuntimeError("boom")
        assert connection is not None and connection.closed
    finally:
        db.close()


def test_export_enumerations_break_timestamp_ties_by_primary_key() -> None:
    """Concurrent same-registry-key exporters must materialize byte-identical
    snapshots, so every enumeration whose sort key is a non-unique timestamp
    must end in the primary key. Rows are inserted with identical timestamps
    in DESCENDING primary-key order: an enumeration that leans on database
    natural order (insertion order on SQLite, backend-dependent on
    PostgreSQL) returns them reversed and would trip the spurious
    'same audit-export registry key produced a divergent snapshot candidate'
    alarm downstream."""
    db = LandscapeDB.in_memory()
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source = register_test_node(factory.data_flow, run.run_id, "tie-source", node_type=NodeType.SOURCE, plugin_name="source")
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source,
            row_index=0,
            data={"value": 1},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row.row_id)
        tie = COMPLETED_AT
        with db.engine.begin() as writer:
            writer.execute(
                batches_table.insert().values(
                    batch_id="tie-batch",
                    run_id=run.run_id,
                    aggregation_node_id=source,
                    status="open",
                    created_at=tie,
                )
            )
            for suffix in ("b", "a"):  # descending primary keys, identical timestamps
                writer.execute(
                    secret_resolutions_table.insert().values(
                        resolution_id=f"res-{suffix}",
                        run_id=run.run_id,
                        timestamp=1234.5,
                        env_var_name="TIE_VAR",
                        source="env",
                        fingerprint="f" * 64,
                    )
                )
                writer.execute(
                    operations_table.insert().values(
                        operation_id=f"op-{suffix}",
                        run_id=run.run_id,
                        node_id=source,
                        operation_type="source_load",
                        started_at=tie,
                        status="open",
                    )
                )
                writer.execute(
                    validation_errors_table.insert().values(
                        error_id=f"val-{suffix}",
                        run_id=run.run_id,
                        row_hash="0" * 64,
                        error="tie",
                        schema_mode="strict",
                        destination="quarantine",
                        created_at=tie,
                    )
                )
                writer.execute(
                    transform_errors_table.insert().values(
                        error_id=f"tra-{suffix}",
                        run_id=run.run_id,
                        token_id=token.token_id,
                        transform_id=source,
                        row_hash="0" * 64,
                        destination="discard",
                        created_at=tie,
                    )
                )
                writer.execute(
                    token_outcomes_table.insert().values(
                        outcome_id=f"out-{suffix}",
                        run_id=run.run_id,
                        token_id=token.token_id,
                        outcome=None,
                        path="buffered",
                        completed=0,
                        recorded_at=tie,
                        batch_id="tie-batch",
                    )
                )
        with db.engine.connect() as connection:
            model = ConnectionBoundExportReadModel(connection)
            assert [item.resolution_id for item in model.get_secret_resolutions_for_run(run.run_id)] == ["res-a", "res-b"]
            assert [item.operation_id for item in model.get_operations_for_run(run.run_id)] == ["op-a", "op-b"]
            assert [item.error_id for item in model.get_validation_errors_for_run(run.run_id)] == ["val-a", "val-b"]
            assert [item.error_id for item in model.get_transform_errors_for_run(run.run_id)] == ["tra-a", "tra-b"]
            outcomes = model.get_token_outcomes_for_tokens(run.run_id, [token.token_id])
            assert [item.outcome_id for item in outcomes] == ["out-a", "out-b"]
    finally:
        db.close()
