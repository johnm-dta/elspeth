# tests/e2e/pipelines/test_large_pipeline.py
"""E2E: Large pipeline with 1000 rows.

Verifies pipeline correctness and audit completeness at scale.
Uses direct SQL queries against the audit database for verification.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from sqlalchemy import func, select

from elspeth.contracts import RunStatus, SinkProtocol, SourceProtocol
from elspeth.core.config import SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import rows_table, token_outcomes_table
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.factories import wire_transforms
from tests.fixtures.plugins import CollectSink, ListSource, PassTransform


class TestLargePipeline:
    """Pipeline correctness and audit completeness at scale."""

    def test_1000_row_pipeline_completes(self, tmp_path: Path) -> None:
        """Create 1000 rows, run through pipeline, verify all reach sink."""
        # -- Arrange --
        source_data = [{"id": i, "value": f"item_{i}", "score": i * 0.1} for i in range(1000)]

        source = ListSource(source_data, on_success="source_out")
        source_settings = SourceSettings(plugin=source.name, on_success="source_out", options={})
        transform = PassTransform()
        wired = wire_transforms([transform], source_connection="source_out", final_sink="default")
        sink = CollectSink()

        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": cast(SourceProtocol, source)},
            source_settings_map={"primary": source_settings},
            transforms=wired,
            sinks=cast("dict[str, SinkProtocol]", {"default": sink}),
            aggregations={},
            gates=[],
        )

        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        orchestrator = Orchestrator(db)
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        # -- Act --
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # -- Assert --
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 1000
        assert result.rows_succeeded == 1000
        assert result.rows_failed == 0
        assert len(sink.results) == 1000

        # Spot-check: verify first and last rows
        assert sink.results[0]["id"] == 0
        assert sink.results[999]["id"] == 999

    def test_large_pipeline_audit_completeness(self, tmp_path: Path) -> None:
        """Verify all 1000 rows have audit records via direct SQL."""
        # -- Arrange --
        source_data = [{"id": i, "value": f"record_{i}"} for i in range(1000)]

        source = ListSource(source_data, on_success="source_out")
        source_settings = SourceSettings(plugin=source.name, on_success="source_out", options={})
        transform = PassTransform()
        wired = wire_transforms([transform], source_connection="source_out", final_sink="default")
        sink = CollectSink()

        graph = ExecutionGraph.from_plugin_instances(
            sources={"primary": cast(SourceProtocol, source)},
            source_settings_map={"primary": source_settings},
            transforms=wired,
            sinks=cast("dict[str, SinkProtocol]", {"default": sink}),
            aggregations={},
            gates=[],
        )

        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        orchestrator = Orchestrator(db)
        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        # -- Act --
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # -- Assert: audit trail completeness via direct SQL --
        assert result.status == RunStatus.COMPLETED

        with db.engine.connect() as conn:
            # All 1000 source rows recorded in audit database
            row_count = conn.execute(select(func.count()).select_from(rows_table).where(rows_table.c.run_id == result.run_id)).scalar()
            assert row_count == 1000

            # All 1000 tokens have terminal outcomes
            terminal_count = conn.execute(
                select(func.count())
                .select_from(token_outcomes_table)
                .where(
                    token_outcomes_table.c.run_id == result.run_id,
                    token_outcomes_table.c.completed == 1,
                )
            ).scalar()
            assert terminal_count == 1000

            # No non-terminal tokens remaining (all rows should be fully processed)
            non_terminal_count = conn.execute(
                select(func.count())
                .select_from(token_outcomes_table)
                .where(
                    token_outcomes_table.c.run_id == result.run_id,
                    token_outcomes_table.c.completed == 0,
                )
            ).scalar()
            assert non_terminal_count == 0

            # Verify row indices cover 0..999
            min_idx = conn.execute(select(func.min(rows_table.c.row_index)).where(rows_table.c.run_id == result.run_id)).scalar()
            max_idx = conn.execute(select(func.max(rows_table.c.row_index)).where(rows_table.c.run_id == result.run_id)).scalar()
            assert min_idx == 0
            assert max_idx == 999
