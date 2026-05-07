# tests/integration/audit/test_sqlcipher_pipeline.py
"""Integration tests for SQLCipher-encrypted audit databases.

Verifies that a full pipeline run produces a correctly encrypted audit trail,
and that the encrypted file is unreadable without the passphrase.
"""

import sqlite3
from pathlib import Path

import pytest

sqlcipher3 = pytest.importorskip("sqlcipher3", reason="sqlcipher3 not installed (install with: uv pip install 'elspeth[security]')")


class TestPipelineWithSQLCipherLandscape:
    """Full pipeline operations with encrypted audit database."""

    def test_pipeline_with_sqlcipher_landscape(self, tmp_path: Path) -> None:
        """A full CRUD cycle via RecorderFactory works on an encrypted DB."""
        from sqlalchemy import select

        from elspeth.contracts import NodeType, RunStatus
        from elspeth.contracts.schema import SchemaConfig
        from elspeth.core.landscape.database import LandscapeDB
        from elspeth.core.landscape.schema import nodes_table, rows_table, runs_table
        from tests.fixtures.landscape import make_factory

        db_path = tmp_path / "pipeline_audit.db"
        passphrase = "integration-test-passphrase"

        db = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=passphrase)
        try:
            factory = make_factory(db)

            # Begin a run
            run = factory.run_lifecycle.begin_run(
                config={"source": {"plugin": "csv"}},
                canonical_version="1.0.0",
            )
            run_id = run.run_id

            # Register a source node
            source_node = factory.data_flow.register_node(
                run_id=run_id,
                plugin_name="csv",
                node_type=NodeType.SOURCE,
                plugin_version="1.0",
                config={"path": "data.csv"},
                schema_config=SchemaConfig.from_dict({"mode": "observed"}),
            )

            # Create a row
            row = factory.data_flow.create_row(
                run_id=run_id,
                source_node_id=source_node.node_id,
                row_index=0,
                data={"customer_id": "C001", "amount": 42.0},
            )

            # Complete the run
            factory.run_lifecycle.complete_run(run_id, status=RunStatus.COMPLETED)

            # Verify via direct SQL
            with db.connection() as conn:
                run_row = conn.execute(select(runs_table).where(runs_table.c.run_id == run_id)).fetchone()
                assert run_row is not None
                assert run_row.status == RunStatus.COMPLETED

                nodes = conn.execute(select(nodes_table).where(nodes_table.c.run_id == run_id)).fetchall()
                assert len(nodes) == 1
                assert nodes[0].plugin_name == "csv"

                rows = conn.execute(select(rows_table).where(rows_table.c.run_id == run_id)).fetchall()
                assert len(rows) == 1
                assert rows[0].row_id == row.row_id
        finally:
            db.close()

    def test_encrypted_db_unreadable_without_key(self, tmp_path: Path) -> None:
        """Standard sqlite3 cannot open a SQLCipher-encrypted database."""
        from elspeth.core.landscape.database import LandscapeDB

        db_path = tmp_path / "encrypted_only.db"
        passphrase = "secret-key-123"

        # Create an encrypted DB with some data
        db = LandscapeDB.from_url(f"sqlite:///{db_path}", passphrase=passphrase)
        db.close()

        # Standard sqlite3 should not be able to read the file
        conn = sqlite3.connect(str(db_path))
        try:
            with pytest.raises(sqlite3.DatabaseError, match="file is not a database"):
                conn.execute("SELECT * FROM sqlite_master")
        finally:
            conn.close()
