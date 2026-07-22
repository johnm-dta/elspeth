"""Regression tests for sink diversion counter accounting."""

from __future__ import annotations

from sqlalchemy import select

from elspeth.contracts import TerminalOutcome, TerminalPath
from elspeth.core.landscape.schema import token_outcomes_table
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource


def test_diverted_sink_rows_do_not_remain_counted_as_success(payload_store, landscape_db) -> None:
    """Durable success counts must exclude rows later diverted during sink write."""
    source = ListSource([{"value": 1}, {"value": 2}], on_success="default")
    sink = CollectSink("default", divert_ordinals=frozenset({1}))
    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[],
        sinks={"default": as_sink(sink)},
        sink_effect_modes={"default": "write"},
    )

    orchestrator = Orchestrator(landscape_db)
    result = orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

    assert result.rows_processed == 2
    assert result.rows_succeeded == 1
    assert result.rows_failed == 1
    assert result.rows_diverted == 1
    assert len(sink.results) == 1

    with landscape_db.engine.connect() as conn:
        diverted = conn.execute(
            select(token_outcomes_table).where(
                (token_outcomes_table.c.run_id == result.run_id)
                & (token_outcomes_table.c.outcome == TerminalOutcome.FAILURE)
                & (token_outcomes_table.c.path == TerminalPath.SINK_DISCARDED)
            )
        ).fetchall()

    assert len(diverted) == 1
