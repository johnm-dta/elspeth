# tests/unit/core/landscape/repository_integration/test_csv_sink_executor_audit.py
"""End-to-end audit-chain integration test for SinkExecutor + CSVSink.

Relocated from tests/unit/contracts/sink_contracts/test_csv_sink_contract.py
(formerly TestCSVSinkExecutorAuditContract). The unit-tier version drove the
executor with MagicMock-backed execution/data_flow/spans repositories and
asserted call-arg equality on those mocks. That verified only that the
executor *called* the mocks in a particular shape — NOT that the real
Landscape would have accepted and persisted a valid artifact record.

Per CLAUDE.md the Landscape audit trail is the legal record; mock-shape
assertions on the audit chain are inadequate confidence. This integration
test drives the SinkExecutor through its production path with a real
RecorderFactory / LandscapeDB and verifies the persisted state by querying
the real database after the write completes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from tests.fixtures.base_classes import create_observed_contract, inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_recorder_with_run, register_test_node

from elspeth.contracts import NodeType, PendingOutcome, TokenInfo
from elspeth.contracts.enums import NodeStateStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.engine.executors import SinkExecutor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.sinks.csv_sink import CSVSink


class TestCSVSinkExecutorAuditChain:
    """Audit-chain contract: a successful CSV sink write through the SinkExecutor
    must produce a fully-formed artifact record, a COMPLETED sink node_state,
    and a recorded terminal token_outcome — all retrievable from a real
    Landscape database.

    These are not mock-shape assertions. Each assertion queries the real
    Landscape DB after the executor returns and checks that the persisted
    record matches what the executor's contract promises.
    """

    def test_executor_registers_csv_artifact_in_real_landscape(self, tmp_path: Path) -> None:
        """A durable CSV write MUST persist a valid artifact record.

        The executor's contract is that a successful sink.write() / sink.flush()
        produces (a) a COMPLETED node_state at the sink node, (b) a registered
        artifact whose content_hash matches the actual file bytes, and (c) a
        terminal token_outcome on the token. We assert all three by querying
        the real Landscape DB.
        """
        # ── Setup: real Landscape, real run, real source + sink nodes ──
        setup = make_recorder_with_run()
        sink_node_id = "csv-sink-node"
        register_test_node(
            setup.data_flow,
            setup.run_id,
            sink_node_id,
            node_type=NodeType.SINK,
            plugin_name="csv",
        )

        # ── Setup: real row + token in the database ──
        row_data = {"id": 1, "name": "Alice"}
        row = setup.data_flow.create_row(
            run_id=setup.run_id,
            source_node_id=setup.source_node_id,
            row_index=0,
            data=row_data,
        )
        db_token = setup.data_flow.create_token(row_id=row.row_id)

        contract = create_observed_contract(row_data)
        token = TokenInfo(
            row_id=row.row_id,
            token_id=db_token.token_id,
            row_data=PipelineRow(data=row_data, contract=contract),
        )

        # ── Setup: real CSVSink writing to a real file ──
        csv_path = tmp_path / "audited_output.csv"
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(csv_path),
                    "schema": {"mode": "fixed", "fields": ["id: int", "name: str"]},
                }
            )
        )
        sink.node_id = sink_node_id

        ctx = make_context(run_id=setup.run_id, node_id=sink_node_id)

        executor = SinkExecutor(
            execution=setup.execution,
            data_flow=setup.data_flow,
            span_factory=SpanFactory(),
            run_id=setup.run_id,
        )

        # ── Drive the production path ──
        try:
            artifact, diversion_counts = executor.write(
                sink=sink,
                tokens=[token],
                ctx=ctx,
                step_in_pipeline=2,
                sink_name="csv_output",
                pending_outcome=PendingOutcome(
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.DEFAULT_FLOW,
                ),
            )
        finally:
            sink.close()

        # ── Verify return values reflect a successful write ──
        assert diversion_counts.total == 0, "No rows should have been diverted"
        assert artifact is not None, (
            "Executor returned no artifact for a successful write — the audit chain is broken: there is no record of the produced output."
        )

        # ── Verify the artifact persisted to the real Landscape DB ──
        # Query the audit DB to confirm the artifact was actually committed,
        # not just constructed in memory.
        persisted_artifacts = setup.execution.get_artifacts(setup.run_id)
        assert len(persisted_artifacts) == 1, (
            f"Expected exactly one artifact record in the Landscape after a "
            f"single sink write; found {len(persisted_artifacts)}. The audit "
            f"trail is missing the produced output."
        )
        persisted = persisted_artifacts[0]

        expected_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        expected_size = csv_path.stat().st_size
        expected_path = f"file://{csv_path}"

        assert persisted.artifact_id == artifact.artifact_id
        assert persisted.sink_node_id == sink_node_id, (
            f"Persisted artifact references sink_node_id={persisted.sink_node_id!r}, expected {sink_node_id!r}"
        )
        assert persisted.artifact_type == "file"
        assert persisted.path_or_uri == expected_path, (
            f"Persisted artifact path={persisted.path_or_uri!r} does not match the actual file location {expected_path!r}"
        )
        assert persisted.content_hash == expected_hash, (
            f"Persisted artifact content_hash={persisted.content_hash!r} does "
            f"not match SHA-256 of the actual file bytes ({expected_hash!r}). "
            f"The audit hash is fabricated relative to the persisted output."
        )
        assert persisted.size_bytes == expected_size

        # ── Verify the sink node_state was completed in the real DB ──
        states = setup.query.get_node_states_for_token(token.token_id)
        sink_states = [s for s in states if s.node_id == sink_node_id]
        assert len(sink_states) == 1, (
            f"Expected exactly one sink node_state for token {token.token_id!r}; "
            f"found {len(sink_states)}. The audit trail does not record this "
            f"token reaching the sink."
        )
        sink_state = sink_states[0]
        assert sink_state.status == NodeStateStatus.COMPLETED, (
            f"Sink node_state status is {sink_state.status!r}, expected COMPLETED. A sink that wrote durably must record success."
        )
        assert sink_state.step_index == 2

        # ── Verify the terminal token_outcome was recorded ──
        outcomes = setup.query.get_all_token_outcomes_for_run(setup.run_id)
        token_outcomes = [o for o in outcomes if o.token_id == token.token_id]
        assert len(token_outcomes) == 1, (
            f"Expected exactly one terminal token_outcome for token "
            f"{token.token_id!r}; found {len(token_outcomes)}. The audit "
            f"chain is missing a terminal state for this token."
        )
        outcome = token_outcomes[0]
        assert outcome.outcome == TerminalOutcome.SUCCESS
        assert outcome.path == TerminalPath.DEFAULT_FLOW
        assert outcome.sink_name == "csv_output"
