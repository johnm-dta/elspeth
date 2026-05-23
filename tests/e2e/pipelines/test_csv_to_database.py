# tests/e2e/pipelines/test_csv_to_database.py
"""E2E: CSV source -> transform -> database sink.

Verifies real CSV-to-SQLite pipeline with audit trail integrity.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from sqlalchemy import create_engine, func, select, text

from elspeth.contracts import RunStatus
from elspeth.core.config import SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import rows_table, token_outcomes_table
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.sinks.database_sink import DatabaseSink
from elspeth.plugins.sources.csv_source import CSVSource
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.factories import wire_transforms
from tests.fixtures.plugins import PassTransform


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    """Write a CSV file with given headers and rows."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


class TestCSVToDatabase:
    """CSV source -> PassTransform -> DatabaseSink (SQLite)."""

    def test_csv_to_sqlite_database(self, tmp_path: Path) -> None:
        """Create CSV, run through pipeline to DatabaseSink, query output DB."""
        # -- Arrange --
        input_csv = tmp_path / "input.csv"
        output_db_path = tmp_path / "output.db"
        output_db_url = f"sqlite:///{output_db_path}"

        headers = ["id", "name", "score"]
        input_rows = [
            ["1", "Alice", "95"],
            ["2", "Bob", "87"],
            ["3", "Carol", "91"],
            ["4", "Dave", "78"],
            ["5", "Eve", "88"],
        ]
        _write_csv(input_csv, headers, input_rows)

        # DatabaseSink needs ELSPETH_ALLOW_RAW_SECRETS for non-env-var URLs in tests
        env_patch = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS")
        os.environ["ELSPETH_ALLOW_RAW_SECRETS"] = "true"

        try:
            source = CSVSource(
                {
                    "path": str(input_csv),
                    "schema": {"mode": "observed"},
                    "on_validation_failure": "discard",
                }
            )
            source.on_success = "csv_out"
            source_settings = SourceSettings(plugin=source.name, on_success="csv_out", options={})
            transform = PassTransform()
            wired = wire_transforms([transform], source_connection="csv_out", final_sink="default")
            sink = DatabaseSink(
                {
                    "url": output_db_url,
                    "table": "pipeline_output",
                    "schema": {"mode": "observed"},
                    "if_exists": "replace",
                }
            )

            # Build graph via production path (BUG-LINEAGE-01)
            graph = ExecutionGraph.from_plugin_instances(
                sources={"primary": source},
                source_settings_map={"primary": source_settings},
                transforms=wired,
                sinks={"default": sink},
                aggregations={},
                gates=[],
            )

            # -- Act --
            audit_db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
            payload_store = FilesystemPayloadStore(tmp_path / "payloads")
            orchestrator = Orchestrator(audit_db)
            config = PipelineConfig(
                sources={"primary": as_source(source)},
                transforms=[as_transform(transform)],
                sinks={"default": as_sink(sink)},
            )
            result = orchestrator.run(config, graph=graph, payload_store=payload_store)
        finally:
            # Restore environment
            if env_patch is None:
                os.environ.pop("ELSPETH_ALLOW_RAW_SECRETS", None)
            else:
                os.environ["ELSPETH_ALLOW_RAW_SECRETS"] = env_patch

        # -- Assert: pipeline completed --
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 5
        assert result.rows_succeeded == 5

        # -- Assert: data in output database --
        engine = create_engine(output_db_url)
        with engine.connect() as conn:
            db_rows = conn.execute(text("SELECT * FROM pipeline_output")).fetchall()
            assert len(db_rows) == 5

            # CSV source reads as strings; verify all present
            names = {row[1] for row in db_rows}
            assert names == {"Alice", "Bob", "Carol", "Dave", "Eve"}
        engine.dispose()

    def test_database_sink_audit_integrity(self, tmp_path: Path) -> None:
        """Verify audit trail records all rows and sink write operations."""
        # -- Arrange --
        input_csv = tmp_path / "input.csv"
        output_db_path = tmp_path / "output.db"
        output_db_url = f"sqlite:///{output_db_path}"

        headers = ["id", "value"]
        input_rows = [
            ["1", "alpha"],
            ["2", "beta"],
            ["3", "gamma"],
        ]
        _write_csv(input_csv, headers, input_rows)

        env_patch = os.environ.get("ELSPETH_ALLOW_RAW_SECRETS")
        os.environ["ELSPETH_ALLOW_RAW_SECRETS"] = "true"

        try:
            source = CSVSource(
                {
                    "path": str(input_csv),
                    "schema": {"mode": "observed"},
                    "on_validation_failure": "discard",
                }
            )
            source.on_success = "csv_out"
            source_settings = SourceSettings(plugin=source.name, on_success="csv_out", options={})
            transform = PassTransform()
            wired = wire_transforms([transform], source_connection="csv_out", final_sink="default")
            sink = DatabaseSink(
                {
                    "url": output_db_url,
                    "table": "test_output",
                    "schema": {"mode": "observed"},
                    "if_exists": "replace",
                }
            )

            graph = ExecutionGraph.from_plugin_instances(
                sources={"primary": source},
                source_settings_map={"primary": source_settings},
                transforms=wired,
                sinks={"default": sink},
                aggregations={},
                gates=[],
            )

            audit_db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
            payload_store = FilesystemPayloadStore(tmp_path / "payloads")
            orchestrator = Orchestrator(audit_db)
            config = PipelineConfig(
                sources={"primary": as_source(source)},
                transforms=[as_transform(transform)],
                sinks={"default": as_sink(sink)},
            )
            result = orchestrator.run(config, graph=graph, payload_store=payload_store)
        finally:
            if env_patch is None:
                os.environ.pop("ELSPETH_ALLOW_RAW_SECRETS", None)
            else:
                os.environ["ELSPETH_ALLOW_RAW_SECRETS"] = env_patch

        # -- Assert: audit trail --
        assert result.status == RunStatus.COMPLETED

        with audit_db.engine.connect() as conn:
            # All 3 source rows recorded in audit
            row_count = conn.execute(select(func.count()).select_from(rows_table).where(rows_table.c.run_id == result.run_id)).scalar()
            assert row_count == 3

            # All tokens have terminal outcomes
            terminal_count = conn.execute(
                select(func.count())
                .select_from(token_outcomes_table)
                .where(
                    token_outcomes_table.c.run_id == result.run_id,
                    token_outcomes_table.c.completed == 1,
                )
            ).scalar()
            assert terminal_count == 3
