"""Read-only behavior for the Landscape MCP analyzer."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import insert

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import LandscapeReadRepositories
from elspeth.core.landscape.schema import runs_table
from elspeth.mcp.analyzer import LandscapeAnalyzer

# Root bypasses directory permission bits, so ``chmod(0o555)`` does not actually
# make a directory read-only for uid 0. CI runs as root (the workflow installs
# apt packages without sudo), which means the immutable read-only-directory
# optimization this test exercises legitimately does not engage there: the
# analyzer can still open ``mode=ro`` against a writable directory and SQLite is
# free to create the ``-wal`` / ``-shm`` sidecars. The URL-selection logic is
# covered deterministically, without relying on filesystem permissions, by
# ``test_sqlite_read_only_url_does_not_use_immutable_for_writable_live_directory``.


def _create_file_backed_audit_db(db_path: Path) -> None:
    db = LandscapeDB.from_url(f"sqlite:///{db_path}")
    try:
        with db.write_connection() as conn:
            conn.execute(
                insert(runs_table).values(
                    run_id="run-read-only",
                    started_at=datetime.now(UTC),
                    config_hash="0" * 64,
                    settings_json="{}",
                    canonical_version="v1",
                    status="completed",
                    seeded_from_cache=False,
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )
    finally:
        db.close()


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root bypasses directory permission bits; a 0o555 dir stays writable, "
    "so the immutable read-only-directory optimization asserted here does not engage",
)
def test_landscape_analyzer_reads_sqlite_without_wal_files_in_read_only_directory(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    _create_file_backed_audit_db(db_path)

    tmp_path.chmod(0o555)
    try:
        analyzer = LandscapeAnalyzer(f"sqlite:///{db_path}")
        try:
            runs = analyzer.list_runs()
            assert [run["run_id"] for run in runs] == ["run-read-only"]
            assert not Path(f"{db_path}-wal").exists()
            assert not Path(f"{db_path}-shm").exists()
        finally:
            analyzer.close()
    finally:
        tmp_path.chmod(0o755)


def test_sqlite_read_only_url_does_not_use_immutable_for_writable_live_directory(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    db_path.touch()

    read_only_url = LandscapeDB._sqlite_read_only_url(f"sqlite:///{db_path}")

    assert "mode=ro" in read_only_url
    assert "immutable=1" not in read_only_url


def test_landscape_analyzer_reads_committed_rows_from_live_wal(tmp_path: Path) -> None:
    db_path = tmp_path / "audit.db"
    writer = LandscapeDB.from_url(f"sqlite:///{db_path}")
    try:
        with writer.engine.begin() as conn:
            conn.exec_driver_sql("PRAGMA wal_autocheckpoint=0")
            conn.execute(
                insert(runs_table).values(
                    run_id="run-live-wal",
                    started_at=datetime.now(UTC),
                    config_hash="0" * 64,
                    settings_json="{}",
                    canonical_version="v1",
                    status="completed",
                    seeded_from_cache=False,
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )

        assert Path(f"{db_path}-wal").exists()

        analyzer = LandscapeAnalyzer(f"sqlite:///{db_path}")
        try:
            assert isinstance(analyzer._factory, LandscapeReadRepositories)
            runs = analyzer.list_runs()
            assert [run["run_id"] for run in runs] == ["run-live-wal"]
        finally:
            analyzer.close()
    finally:
        writer.close()
