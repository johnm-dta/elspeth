"""Read-only behavior for the Landscape MCP analyzer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import insert

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import runs_table
from elspeth.mcp.analyzer import LandscapeAnalyzer


def _create_file_backed_audit_db(db_path: Path) -> None:
    db = LandscapeDB.from_url(f"sqlite:///{db_path}")
    try:
        with db.connection() as conn:
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
