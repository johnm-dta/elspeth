"""Connection-bound, immutable audit-export read snapshot tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import event

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.export_read_model import ConnectionBoundExportReadModel, open_export_read_transaction
from elspeth.core.landscape.schema import run_attributions_table, runs_table

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
