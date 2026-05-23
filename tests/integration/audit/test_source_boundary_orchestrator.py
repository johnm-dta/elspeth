"""Production-path source boundary enforcement integration coverage."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from elspeth.contracts import RunStatus
from elspeth.contracts.enums import NodeStateStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import SourceGuaranteedFieldsViolation
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.schema import node_states_table, runs_table, token_outcomes_table
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source
from tests.fixtures.pipeline import build_linear_pipeline
from tests.fixtures.stores import MockPayloadStore


def test_orchestrator_records_source_boundary_failure_before_reraising() -> None:
    """Source boundary violations must be enforced on the production path."""
    db = LandscapeDB.in_memory()
    payload_store = MockPayloadStore()

    source, _tx_list, sinks, graph = build_linear_pipeline([{"value": 42}], transforms=[])
    source.declared_guaranteed_fields = frozenset({"customer_id"})
    sink = sinks["default"]

    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[],
        sinks={"default": as_sink(sink)},
    )

    with pytest.raises(SourceGuaranteedFieldsViolation):
        Orchestrator(db).run(config, graph=graph, payload_store=payload_store)

    with db.connection() as conn:
        run_rows = conn.execute(select(runs_table)).fetchall()
        assert len(run_rows) == 1
        run_row = run_rows[0]
        outcome_rows = conn.execute(select(token_outcomes_table).where(token_outcomes_table.c.run_id == run_row.run_id)).fetchall()
        state_rows = conn.execute(select(node_states_table).where(node_states_table.c.run_id == run_row.run_id)).fetchall()

    assert run_row.status == RunStatus.FAILED
    assert sink.results == []
    assert len(outcome_rows) == 1
    assert outcome_rows[0].outcome == TerminalOutcome.FAILURE.value
    assert outcome_rows[0].path == TerminalPath.UNROUTED.value

    source_states = [row for row in state_rows if row.node_id == source.node_id]
    assert len(source_states) == 1
    assert source_states[0].status == NodeStateStatus.FAILED
