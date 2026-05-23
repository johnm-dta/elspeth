# tests/integration/pipeline/orchestrator/test_completed_outcome_timing.py
"""Regression tests for COMPLETED outcome timing relative to sink durability.

These tests guard against a previously-fixed bug where COMPLETED token outcomes
were recorded BEFORE sink writes happened. This violated the token outcome
contract (docs/contracts/token-outcomes/00-token-outcome-contract.md):

- Invariant 3: "COMPLETED implies the token has a completed sink node_state"
- Invariant 4: "Completed sink node_state implies a COMPLETED token_outcome with sink_name"

The fix ensured COMPLETED outcomes are only recorded AFTER successful sink writes.
These tests verify the CORRECT behavior to prevent regression.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from elspeth.contracts import (
    NodeStateStatus,
    TerminalOutcome,
    TerminalPath,
)
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.schema import node_states_table, token_outcomes_table
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import (
    as_sink,
    as_source,
    as_transform,
)
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import FailingSink, ListSource, PassTransform


class TestCompletedOutcomeTimingContract:
    """Regression tests for COMPLETED outcome timing contract.

    Per the token outcome contract (docs/contracts/token-outcomes/00-token-outcome-contract.md):
    - Invariant 3: "COMPLETED implies the token has a completed sink node_state"
    - Invariant 4: "Completed sink node_state implies a COMPLETED token_outcome"

    These tests verify the correct behavior and prevent regression.
    """

    def test_no_completed_outcomes_when_sink_write_fails(self, tmp_path: Path, payload_store) -> None:
        """COMPLETED outcomes must NOT exist when sink.write() throws.

        CORRECT BEHAVIOR: COMPLETED should only be recorded AFTER successful
        sink writes. If sink.write() throws, no data reached the sink, so
        no tokens should have COMPLETED outcomes.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")

        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = PassTransform(name="passthrough", on_success="default", on_error="discard")
        sink = FailingSink(error_message="Simulated sink failure - data never written")

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)

        with pytest.raises(RuntimeError, match="Simulated sink failure"):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        with db.engine.connect() as conn:
            completed_outcomes = conn.execute(
                select(token_outcomes_table).where(
                    (token_outcomes_table.c.outcome == TerminalOutcome.SUCCESS) & (token_outcomes_table.c.path == TerminalPath.DEFAULT_FLOW)
                )
            ).fetchall()

            completed_sink_states = conn.execute(
                select(node_states_table).where(
                    (node_states_table.c.node_id == "sink_default") & (node_states_table.c.status == NodeStateStatus.COMPLETED)
                )
            ).fetchall()

        assert len(completed_outcomes) == 0, (
            f"CONTRACT VIOLATION: Found {len(completed_outcomes)} COMPLETED outcomes "
            f"but sink.write() threw an exception - data never reached the sink!\n"
            f"This violates Invariant 3: 'COMPLETED implies token has completed sink node_state'\n"
            f"Completed sink node_states: {len(completed_sink_states)}\n"
            f"The audit trail falsely claims success for data that was never written."
        )

    def test_invariant_3_completed_implies_sink_node_state(self, tmp_path: Path, payload_store) -> None:
        """Every COMPLETED outcome must have a corresponding completed sink node_state.

        Contract Invariant 3: "COMPLETED implies the token has a completed sink node_state"
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")

        source = ListSource([{"value": i} for i in range(5)])
        transform = PassTransform(name="passthrough", on_success="default", on_error="discard")
        sink = FailingSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)

        with pytest.raises(RuntimeError):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        with db.engine.connect() as conn:
            completed_outcomes = conn.execute(
                select(token_outcomes_table).where(
                    (token_outcomes_table.c.outcome == TerminalOutcome.SUCCESS) & (token_outcomes_table.c.path == TerminalPath.DEFAULT_FLOW)
                )
            ).fetchall()

            completed_sink_states = conn.execute(
                select(node_states_table).where(
                    (node_states_table.c.node_id == "sink_default") & (node_states_table.c.status == NodeStateStatus.COMPLETED)
                )
            ).fetchall()

        completed_outcome_token_ids = {o.token_id for o in completed_outcomes}
        sink_state_token_ids = {s.token_id for s in completed_sink_states}

        orphan_completed = completed_outcome_token_ids - sink_state_token_ids

        assert len(orphan_completed) == 0, (
            f"INVARIANT 3 VIOLATION: {len(orphan_completed)} tokens have COMPLETED "
            f"outcomes but NO completed sink node_state.\n"
            f"Orphan token_ids: {orphan_completed}\n"
            f"Total COMPLETED outcomes: {len(completed_outcomes)}\n"
            f"Total completed sink states: {len(completed_sink_states)}\n"
            f"Contract: 'COMPLETED implies the token has a completed sink node_state'"
        )

    def test_audit_trail_does_not_lie_about_success(self, tmp_path: Path, payload_store) -> None:
        """Audit trail must not claim success for data that didn't reach sink.

        An auditor querying token_outcomes should NOT see COMPLETED for rows
        that never made it to the sink.

        CORRECT: After sink failure, zero COMPLETED outcomes.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")

        source = ListSource([{"value": i} for i in range(5)])
        transform = PassTransform(name="passthrough", on_success="default", on_error="discard")
        sink = FailingSink()

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)

        with pytest.raises(RuntimeError):
            orchestrator.run(config, graph=build_production_graph(config), payload_store=payload_store)

        with db.engine.connect() as conn:
            completed_outcomes = conn.execute(
                select(token_outcomes_table).where(
                    (token_outcomes_table.c.outcome == TerminalOutcome.SUCCESS) & (token_outcomes_table.c.path == TerminalPath.DEFAULT_FLOW)
                )
            ).fetchall()
            completed_count = len(completed_outcomes)

        assert completed_count == 0, (
            f"AUDIT INTEGRITY VIOLATION: Audit trail shows {completed_count} COMPLETED "
            f"outcomes for rows that NEVER reached the sink!\n"
            f"An auditor would incorrectly conclude these rows were successfully processed.\n"
            f"This is evidence tampering from an audit perspective."
        )
