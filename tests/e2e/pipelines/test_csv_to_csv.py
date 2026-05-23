# tests/e2e/pipelines/test_csv_to_csv.py
"""E2E: CSV source -> transform -> CSV sink roundtrip.

Verifies real file I/O through the full pipeline path with
audit trail integrity.
"""

from __future__ import annotations

import csv
from pathlib import Path

from sqlalchemy import func, select

from elspeth.contracts import RunStatus
from elspeth.core.config import SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import rows_table, token_outcomes_table
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.sinks.csv_sink import CSVSink
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and return rows as dicts."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


class TestCSVToCSV:
    """CSV source -> PassTransform -> CSV sink roundtrip."""

    def test_csv_source_to_csv_sink_roundtrip(self, tmp_path: Path) -> None:
        """Create CSV, run through pipeline, verify output matches input."""
        # -- Arrange: write input CSV --
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        headers = ["id", "name", "score"]
        input_rows = [
            ["1", "Alice", "95"],
            ["2", "Bob", "87"],
            ["3", "Carol", "91"],
            ["4", "Dave", "78"],
            ["5", "Eve", "88"],
        ]
        _write_csv(input_csv, headers, input_rows)

        # Build real plugins
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
        sink = CSVSink(
            {
                "path": str(output_csv),
                "schema": {"mode": "observed"},
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

        # -- Act: run pipeline --
        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        orchestrator = Orchestrator(db)
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # -- Assert --
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 5
        assert result.rows_succeeded == 5

        # Verify output CSV exists and contains correct data
        assert output_csv.exists()
        output_rows = _read_csv(output_csv)
        assert len(output_rows) == 5

        # CSV source reads everything as strings; verify roundtrip
        for i, row in enumerate(output_rows):
            assert row["id"] == input_rows[i][0]
            assert row["name"] == input_rows[i][1]
            assert row["score"] == input_rows[i][2]

    def test_csv_pipeline_audit_trail(self, tmp_path: Path) -> None:
        """Verify audit trail records all rows with lineage."""
        # -- Arrange --
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        headers = ["id", "value"]
        input_rows = [
            ["1", "alpha"],
            ["2", "beta"],
            ["3", "gamma"],
            ["4", "delta"],
            ["5", "epsilon"],
        ]
        _write_csv(input_csv, headers, input_rows)

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
        sink = CSVSink(
            {
                "path": str(output_csv),
                "schema": {"mode": "observed"},
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

        db_path = tmp_path / "audit.db"
        db = LandscapeDB(f"sqlite:///{db_path}")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        orchestrator = Orchestrator(db)
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        # -- Act --
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # -- Assert: audit trail completeness --
        assert result.status == RunStatus.COMPLETED

        with db.engine.connect() as conn:
            # All 5 source rows recorded
            row_count = conn.execute(select(func.count()).select_from(rows_table).where(rows_table.c.run_id == result.run_id)).scalar()
            assert row_count == 5

            # All tokens have terminal outcomes
            terminal_count = conn.execute(
                select(func.count())
                .select_from(token_outcomes_table)
                .where(
                    token_outcomes_table.c.run_id == result.run_id,
                    token_outcomes_table.c.completed == 1,
                )
            ).scalar()
            assert terminal_count == 5
