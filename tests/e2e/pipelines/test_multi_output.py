# tests/e2e/pipelines/test_multi_output.py
"""E2E: Config-driven gate routing to multiple sinks.

Verifies that GateSettings correctly directs rows to different sinks
based on expression conditions, with full audit trail integrity.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select

from elspeth.contracts import RunStatus
from elspeth.core.config import GateSettings
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import rows_table, token_outcomes_table
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource


class TestMultiOutput:
    """Config-driven gate routing to multiple sinks."""

    def test_gate_routes_to_multiple_sinks(self, tmp_path: Path) -> None:
        """Create 10 rows with category A/B, verify correct routing via gate."""
        # -- Arrange --
        source_data = [{"id": i, "category": "A" if i % 2 == 0 else "B", "value": i * 10} for i in range(1, 11)]
        # Expected: ids 2,4,6,8,10 -> sink_a (category A), ids 1,3,5,7,9 -> sink_b (category B)

        source = ListSource(source_data, on_success="sink_a")
        sink_a = CollectSink("sink_a")
        sink_b = CollectSink("sink_b")

        # Config-driven gate: route by category field
        category_gate = GateSettings(
            name="category_router",
            input="source_out",
            condition="row['category'] == 'A'",
            routes={"true": "sink_a", "false": "sink_b"},
        )

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[category_gate],
        )
        graph = build_production_graph(config)

        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        orchestrator = Orchestrator(db)

        # -- Act --
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # -- Assert --
        # elspeth-5069612f3c: gate route_to_sink is intentional MOVE
        # (rows_routed_success). Gate-routed-only run -> COMPLETED.
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 10

        # 5 rows with even ids (category A) -> sink_a
        assert len(sink_a.results) == 5
        for row in sink_a.results:
            assert row["category"] == "A"

        # 5 rows with odd ids (category B) -> sink_b
        assert len(sink_b.results) == 5
        for row in sink_b.results:
            assert row["category"] == "B"

    def test_multi_sink_audit_trail(self, tmp_path: Path) -> None:
        """Verify audit trail shows correct routing decisions for each row."""
        # -- Arrange --
        source_data = [
            {"id": 1, "category": "A", "value": 10},
            {"id": 2, "category": "B", "value": 20},
            {"id": 3, "category": "A", "value": 30},
            {"id": 4, "category": "B", "value": 40},
            {"id": 5, "category": "A", "value": 50},
        ]

        source = ListSource(source_data, on_success="sink_a")
        sink_a = CollectSink("sink_a")
        sink_b = CollectSink("sink_b")

        category_gate = GateSettings(
            name="category_router",
            input="source_out",
            condition="row['category'] == 'A'",
            routes={"true": "sink_a", "false": "sink_b"},
        )

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[],
            sinks={"sink_a": as_sink(sink_a), "sink_b": as_sink(sink_b)},
            gates=[category_gate],
        )
        graph = build_production_graph(config)

        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        orchestrator = Orchestrator(db)

        # -- Act --
        result = orchestrator.run(config, graph=graph, payload_store=payload_store)

        # -- Assert: audit trail --
        # elspeth-5069612f3c: gate route_to_sink is intentional MOVE
        # (rows_routed_success). Gate-routed-only run -> COMPLETED.
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

            # Verify routing to correct sinks in outcomes
            routed_to_a = conn.execute(
                select(func.count())
                .select_from(token_outcomes_table)
                .where(
                    token_outcomes_table.c.run_id == result.run_id,
                    token_outcomes_table.c.sink_name == "sink_a",
                    token_outcomes_table.c.completed == 1,
                )
            ).scalar()

            routed_to_b = conn.execute(
                select(func.count())
                .select_from(token_outcomes_table)
                .where(
                    token_outcomes_table.c.run_id == result.run_id,
                    token_outcomes_table.c.sink_name == "sink_b",
                    token_outcomes_table.c.completed == 1,
                )
            ).scalar()

            # 3 category A -> sink_a, 2 category B -> sink_b
            assert routed_to_a == 3
            assert routed_to_b == 2
