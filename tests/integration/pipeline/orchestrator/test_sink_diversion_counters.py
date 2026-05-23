"""Regression tests for sink diversion counter accounting."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from elspeth.contracts import ArtifactDescriptor, TerminalOutcome, TerminalPath
from elspeth.contracts.diversion import RowDiversion, SinkWriteResult
from elspeth.core.landscape.schema import token_outcomes_table
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import _TestSinkBase, as_sink, as_source
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import ListSource


class DivertSecondRowSink(_TestSinkBase):
    """Sink that diverts the second row and durably writes the rest."""

    def __init__(self, name: str = "default") -> None:
        super().__init__()
        self.name = name
        self.results: list[dict[str, object]] = []
        self._artifact_counter = 0

    def write(self, rows: list[dict[str, object]], ctx: Any) -> SinkWriteResult:
        del ctx
        self._artifact_counter += 1

        primary_rows = [row for index, row in enumerate(rows) if index != 1]
        self.results.extend(primary_rows)

        diversions: tuple[RowDiversion, ...] = ()
        if len(rows) > 1:
            diversions = (
                RowDiversion(
                    row_index=1,
                    reason="sink rejected second row",
                    row_data=rows[1],
                ),
            )

        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_file(
                path=f"memory://{self.name}_{self._artifact_counter}",
                size_bytes=len(str(primary_rows)),
                content_hash=f"hash_{self._artifact_counter}",
            ),
            diversions=diversions,
        )


def test_diverted_sink_rows_do_not_remain_counted_as_success(payload_store, landscape_db) -> None:
    """Durable success counts must exclude rows later diverted during sink write."""
    source = ListSource([{"value": 1}, {"value": 2}], on_success="default")
    sink = DivertSecondRowSink("default")
    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[],
        sinks={"default": as_sink(sink)},
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
