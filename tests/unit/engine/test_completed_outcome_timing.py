# tests/unit/engine/test_completed_outcome_timing.py
"""Tests for COMPLETED outcome timing relative to sink durability.

BUG UNDER TEST: COMPLETED token outcomes are recorded BEFORE sink writes happen.
This violates the token outcome contract (docs/contracts/token-outcomes/00-token-outcome-contract.md):

- Invariant 3: "COMPLETED implies the token has a completed sink node_state"
- Invariant 4: "Completed sink node_state implies a COMPLETED token_outcome with sink_name"

The current code (orchestrator.py:1015-1022) records COMPLETED outcomes in the
processing loop, but sink writes happen AFTER the loop (orchestrator.py:1175).
If sink.write() fails or the run crashes between processing and sink write,
the audit trail falsely claims success for data that never reached a sink.

These tests define the CORRECT behavior: COMPLETED outcomes should only exist
after successful sink writes. They currently FAIL because the bug exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from elspeth.contracts import (
    ArtifactDescriptor,
    NodeStateStatus,
    NodeType,
    PipelineRow,
    PluginSchema,
    RoutingMode,
    RowOutcome,
    SinkName,
)
from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.contracts.types import NodeID
from elspeth.core.checkpoint import CheckpointManager
from elspeth.core.config import CheckpointSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.schema import node_states_table, token_outcomes_table
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult
from tests.fixtures.base_classes import (
    _TestSinkBase,
    as_sink,
    as_source,
    as_transform,
)
from tests.fixtures.plugins import ListSource


def _build_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build a simple linear graph for testing."""
    graph = ExecutionGraph()
    schema_config = {"schema": {"mode": "observed"}}

    # Add source
    graph.add_node(
        "source",
        node_type=NodeType.SOURCE,
        plugin_name=config.source.name,
        config=schema_config,
    )

    # Add transforms
    prev = "source"
    for i, t in enumerate(config.transforms):
        node_id = f"transform_{i}"
        graph.add_node(
            node_id,
            node_type=NodeType.TRANSFORM,
            plugin_name=t.name,
            config=schema_config,
        )
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    # Add sink
    sink_node_id = NodeID("sink_default")
    graph.add_node(
        sink_node_id,
        node_type=NodeType.SINK,
        plugin_name="failing_sink",
        config=schema_config,
    )
    graph.add_edge(prev, sink_node_id, label="continue", mode=RoutingMode.MOVE)

    # Set internal mappings
    graph.set_sink_id_map({SinkName("default"): sink_node_id})
    graph.set_transform_id_map({i: NodeID(f"transform_{i}") for i in range(len(config.transforms))})
    graph.set_config_gate_id_map({})
    graph.set_route_resolution_map({})

    return graph


class TestCompletedOutcomeTimingContract:
    """Tests for COMPLETED outcome timing contract.

    Per the token outcome contract (docs/contracts/token-outcomes/00-token-outcome-contract.md):
    - Invariant 3: "COMPLETED implies the token has a completed sink node_state"
    - Invariant 4: "Completed sink node_state implies a COMPLETED token_outcome"

    These tests verify the CORRECT behavior. They currently FAIL because of the bug.
    """

    def test_no_completed_outcomes_when_sink_write_fails(self, tmp_path: Path, payload_store) -> None:
        """COMPLETED outcomes must NOT exist when sink.write() throws.

        CORRECT BEHAVIOR: COMPLETED should only be recorded AFTER successful
        sink writes. If sink.write() throws, no data reached the sink, so
        no tokens should have COMPLETED outcomes.

        CURRENT BUG: COMPLETED is recorded in the processing loop BEFORE
        sink.write() is called, so tokens have COMPLETED outcomes even
        though they never reached the sink.

        This test FAILS with current code (proving the bug) and will PASS
        when the bug is fixed.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_manager = CheckpointManager(db)
        checkpoint_settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(checkpoint_settings)

        class RowSchema(PluginSchema):
            value: int

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = RowSchema
            output_schema = RowSchema
            on_error = "discard"
            on_success = "default"

            def __init__(self) -> None:
                super().__init__({"schema": {"mode": "observed"}})

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                return TransformResult.success(
                    PipelineRow(row.to_dict(), row.contract),
                    success_reason={"action": "passthrough"},
                )

        class FailingSink(_TestSinkBase):
            """Sink that always throws on write.

            This simulates a crash/failure at the sink write boundary.
            If COMPLETED is recorded correctly (after sink write), then
            no COMPLETED outcomes should exist when this sink throws.
            """

            name = "failing_sink"

            def __init__(self) -> None:
                super().__init__()

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                raise RuntimeError("Simulated sink failure - data never written")

        # Create test data: 3 rows
        source = ListSource([{"value": 1}, {"value": 2}, {"value": 3}])
        transform = PassthroughTransform()
        sink = FailingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_manager,
            checkpoint_config=checkpoint_config,
        )

        # Run the pipeline - expect it to fail due to sink exception
        with pytest.raises(RuntimeError, match="Simulated sink failure"):
            orchestrator.run(config, graph=_build_graph(config), payload_store=payload_store)

        # Check the audit trail
        with db.engine.connect() as conn:
            # Query token_outcomes for COMPLETED outcomes
            completed_outcomes = conn.execute(
                select(token_outcomes_table).where(token_outcomes_table.c.outcome == RowOutcome.COMPLETED.value)
            ).fetchall()

            # Query node_states at sink to verify no successful sink processing
            completed_sink_states = conn.execute(
                select(node_states_table).where(
                    (node_states_table.c.node_id == "sink_default") & (node_states_table.c.status == NodeStateStatus.COMPLETED.value)
                )
            ).fetchall()

        # CONTRACT REQUIREMENT: No COMPLETED outcomes when sink.write() fails
        # If sink.write() threw, data never reached the sink, so claiming
        # COMPLETED violates the audit contract.
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

        This test verifies that whenever a COMPLETED outcome exists, there is
        also a node_state at a sink node with status=COMPLETED for that token.

        This test FAILS with current code when sink.write() fails, because
        COMPLETED outcomes exist without corresponding sink node_states.
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")

        class RowSchema(PluginSchema):
            value: int

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = RowSchema
            output_schema = RowSchema
            on_error = "discard"
            on_success = "default"

            def __init__(self) -> None:
                super().__init__({"schema": {"mode": "observed"}})

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                return TransformResult.success(
                    PipelineRow(row.to_dict(), row.contract),
                    success_reason={"action": "passthrough"},
                )

        class FailingSink(_TestSinkBase):
            name = "failing_sink"

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                raise RuntimeError("Sink write failed")

        source = ListSource([{"value": i} for i in range(5)])
        transform = PassthroughTransform()
        sink = FailingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)

        with pytest.raises(RuntimeError):
            orchestrator.run(config, graph=_build_graph(config), payload_store=payload_store)

        # Verify Invariant 3: For every COMPLETED outcome, a sink node_state must exist
        with db.engine.connect() as conn:
            completed_outcomes = conn.execute(
                select(token_outcomes_table).where(token_outcomes_table.c.outcome == RowOutcome.COMPLETED.value)
            ).fetchall()

            # Get all completed sink node_states
            completed_sink_states = conn.execute(
                select(node_states_table).where(
                    (node_states_table.c.node_id == "sink_default") & (node_states_table.c.status == NodeStateStatus.COMPLETED.value)
                )
            ).fetchall()

        completed_outcome_token_ids = {o.token_id for o in completed_outcomes}
        sink_state_token_ids = {s.token_id for s in completed_sink_states}

        # Every COMPLETED outcome must have a corresponding sink node_state
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

        This is the user-facing impact: an auditor querying token_outcomes
        should NOT see COMPLETED for rows that never made it to the sink.

        CORRECT: After sink failure, zero COMPLETED outcomes.
        BUGGY: After sink failure, COMPLETED outcomes exist (false claims).
        """
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")

        class RowSchema(PluginSchema):
            value: int

        class PassthroughTransform(BaseTransform):
            name = "passthrough"
            input_schema = RowSchema
            output_schema = RowSchema
            on_error = "discard"
            on_success = "default"

            def __init__(self) -> None:
                super().__init__({"schema": {"mode": "observed"}})

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                return TransformResult.success(
                    PipelineRow(row.to_dict(), row.contract),
                    success_reason={"action": "passthrough"},
                )

        class FailingSink(_TestSinkBase):
            name = "failing_sink"

            def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
                raise RuntimeError("Sink write failed")

        source = ListSource([{"value": i} for i in range(5)])
        transform = PassthroughTransform()
        sink = FailingSink()

        config = PipelineConfig(
            source=as_source(source),
            transforms=[as_transform(transform)],
            sinks={"default": as_sink(sink)},
        )

        orchestrator = Orchestrator(db)

        with pytest.raises(RuntimeError):
            orchestrator.run(config, graph=_build_graph(config), payload_store=payload_store)

        # Simulate what an auditor would see
        with db.engine.connect() as conn:
            completed_outcomes = conn.execute(
                select(token_outcomes_table).where(token_outcomes_table.c.outcome == RowOutcome.COMPLETED.value)
            ).fetchall()
            completed_count = len(completed_outcomes)

        # AUDIT INTEGRITY: No false claims of success
        assert completed_count == 0, (
            f"AUDIT INTEGRITY VIOLATION: Audit trail shows {completed_count} COMPLETED "
            f"outcomes for rows that NEVER reached the sink!\n"
            f"An auditor would incorrectly conclude these rows were successfully processed.\n"
            f"This is evidence tampering from an audit perspective."
        )
