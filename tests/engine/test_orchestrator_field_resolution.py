# tests/engine/test_orchestrator_field_resolution.py
"""Test field resolution recording in audit trail.

This verifies the fix for P2 where field resolution must be recorded
AFTER the source iterator executes (not before) because CSVSource.load()
is a generator that only computes field resolution when iterated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy import select

from elspeth.contracts import ArtifactDescriptor
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.schema import runs_table
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.sources.csv_source import CSVSource
from tests.conftest import _TestSinkBase


class TestFieldResolutionRecording:
    """Tests that field resolution is correctly recorded in audit trail."""

    def test_field_resolution_recorded_with_normalize_fields(self, tmp_path: Path, payload_store) -> None:
        """Field resolution is recorded when normalize_fields=True.

        This is the key test for the P2 fix - field resolution must be captured
        AFTER the source iterator executes, not before.
        """
        # Create CSV with headers that need normalization
        csv_input = tmp_path / "input.csv"
        csv_input.write_text("User ID,Amount $,Email Address\n1,100,a@b.com\n2,200,c@d.com\n")

        # Create in-memory database
        db = LandscapeDB.in_memory()

        # Create source with normalize_fields=True
        source = CSVSource(
            {
                "path": str(csv_input),
                "normalize_fields": True,
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "discard",
            }
        )

        # Create sink using test helper
        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def on_complete(self, ctx: Any) -> None:
                pass

            def close(self) -> None:
                pass

        sink = CollectSink()

        # Build config and graph
        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": sink},
        )

        # Use helper to build graph
        from tests.engine.orchestrator_test_helpers import build_production_graph

        graph = build_production_graph(config)

        # Run pipeline
        orchestrator = Orchestrator(db)
        result = orchestrator.run(config=config, graph=graph, payload_store=payload_store)

        # Query the audit database for field resolution
        with db.engine.connect() as conn:
            row = conn.execute(select(runs_table.c.source_field_resolution_json).where(runs_table.c.run_id == result.run_id)).fetchone()

            resolution_json = row[0]

        # This is the key assertion - resolution must be recorded
        assert resolution_json is not None, "source_field_resolution_json was not recorded in audit trail"

        # Verify the content
        resolution_data = json.loads(resolution_json)
        mapping = resolution_data["resolution_mapping"]
        version = resolution_data["normalization_version"]

        assert mapping["User ID"] == "user_id"
        assert mapping["Amount $"] == "amount"
        assert mapping["Email Address"] == "email_address"
        assert version == "1.0.0"

        # Verify sink received normalized data
        assert len(sink.results) == 2
        assert "user_id" in sink.results[0]
        assert sink.results[0]["user_id"] == "1"

    def test_field_resolution_identity_without_normalization(self, tmp_path: Path, payload_store) -> None:
        """Field resolution records identity mapping when normalize_fields=False.

        Even without normalization, CSVSource records an identity mapping
        (header -> header) with normalization_version=None. This is correct
        for auditability - we always capture the original headers from the CSV.
        """
        # Create CSV with headers
        csv_input = tmp_path / "input.csv"
        csv_input.write_text("user_id,amount\n1,100\n2,200\n")

        # Create in-memory database
        db = LandscapeDB.in_memory()

        # Create source without normalization (default)
        source = CSVSource(
            {
                "path": str(csv_input),
                # normalize_fields defaults to False
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "discard",
            }
        )

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def on_complete(self, ctx: Any) -> None:
                pass

            def close(self) -> None:
                pass

        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": sink},
        )

        from tests.engine.orchestrator_test_helpers import build_production_graph

        graph = build_production_graph(config)

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config=config, graph=graph, payload_store=payload_store)

        # Query the audit database for field resolution
        with db.engine.connect() as conn:
            row = conn.execute(select(runs_table.c.source_field_resolution_json).where(runs_table.c.run_id == result.run_id)).fetchone()

            resolution_json = row[0]

        # Even without normalization, CSVSource records an identity mapping
        # This is correct - we capture what headers the CSV had
        assert resolution_json is not None, "Field resolution should be recorded even without normalization"

        resolution_data = json.loads(resolution_json)
        mapping = resolution_data["resolution_mapping"]
        version = resolution_data["normalization_version"]

        # Identity mapping - headers map to themselves
        assert mapping["user_id"] == "user_id"
        assert mapping["amount"] == "amount"

        # No normalization version since no normalization was applied
        assert version is None

        # Verify sink received data unchanged
        assert len(sink.results) == 2
        assert "user_id" in sink.results[0]

    def test_field_resolution_recorded_for_empty_source(self, tmp_path: Path, payload_store) -> None:
        """Field resolution is recorded even when source yields zero rows.

        This tests the edge case of header-only CSV files - the source plugin
        reads headers (and computes field resolution) but yields no data rows.
        The audit trail must still capture the header normalization mapping.

        Regression test for: P3 review comment about empty source field resolution.
        """
        # Create CSV with headers ONLY - no data rows
        csv_input = tmp_path / "input.csv"
        csv_input.write_text("User ID,Amount $,Email Address\n")  # Header only!

        # Create in-memory database
        db = LandscapeDB.in_memory()

        # Create source with normalize_fields=True
        source = CSVSource(
            {
                "path": str(csv_input),
                "normalize_fields": True,
                "schema": {"fields": "dynamic"},
                "on_validation_failure": "discard",
            }
        )

        class CollectSink(_TestSinkBase):
            name = "collect"

            def __init__(self) -> None:
                self.results: list[dict[str, Any]] = []

            def on_start(self, ctx: Any) -> None:
                pass

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                self.results.extend(rows)
                return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="")

            def on_complete(self, ctx: Any) -> None:
                pass

            def close(self) -> None:
                pass

        sink = CollectSink()

        config = PipelineConfig(
            source=source,
            transforms=[],
            sinks={"default": sink},
        )

        from tests.engine.orchestrator_test_helpers import build_production_graph

        graph = build_production_graph(config)

        orchestrator = Orchestrator(db)
        result = orchestrator.run(config=config, graph=graph, payload_store=payload_store)

        # Query the audit database for field resolution
        with db.engine.connect() as conn:
            row = conn.execute(select(runs_table.c.source_field_resolution_json).where(runs_table.c.run_id == result.run_id)).fetchone()

            resolution_json = row[0]

        # KEY ASSERTION: Resolution must be recorded even for empty sources
        assert resolution_json is not None, (
            "source_field_resolution_json was not recorded for header-only CSV. "
            "Field resolution should be captured after the loop when no rows were processed."
        )

        # Verify the content - normalization still happened on headers
        resolution_data = json.loads(resolution_json)
        mapping = resolution_data["resolution_mapping"]
        version = resolution_data["normalization_version"]

        assert mapping["User ID"] == "user_id"
        assert mapping["Amount $"] == "amount"
        assert mapping["Email Address"] == "email_address"
        assert version == "1.0.0"

        # Verify no rows were processed
        assert len(sink.results) == 0
        assert result.rows_processed == 0
