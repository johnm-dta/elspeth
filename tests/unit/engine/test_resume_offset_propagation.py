"""Regression guard: token-carried resume offset reaches sink node_state.

ADDENDUM 4 corrects the design gap introduced in a3ead6692: resume_attempt_offset
and resume_checkpoint_id must live on TokenInfo (not WorkItem) so that SinkExecutor,
which buffers TokenInfos from multiple WorkItems and calls begin_node_state
per-token in a loop, can read the correct per-token offset.

This test is the minimal regression guard for that specific gap:
- Constructs a TokenInfo with resume_attempt_offset=1, resume_checkpoint_id="ck-test"
- Drives it through SinkExecutor.write() with a REAL LandscapeDB (no mock)
- Reads the resulting node_states row back and asserts attempt=1, resume_checkpoint_id="ck-test"

Limitation: this test exercises the per-token read inside the SinkExecutor loop
(the exact gap a3ead6692 left). It does NOT test that upstream buffering paths
preserve the field-carrying TokenInfo end-to-end — that guarantee lands with
Task 7's full RED→GREEN drive of the integration test.
"""

from __future__ import annotations

from sqlalchemy import select

from elspeth.contracts import NodeType, PendingOutcome, PluginSchema, TokenInfo
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.schema_contract import SchemaContract
from elspeth.core.landscape.schema import node_states_table
from elspeth.engine.executors import SinkExecutor
from elspeth.testing import make_field, make_row
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_recorder_with_run, register_test_node


class _PermissiveSchema(PluginSchema):
    """Accept arbitrary sink rows for executor plumbing tests."""


class _SinkDouble:
    def __init__(self, node_id: str = "node-sink-1") -> None:
        self.name = "primary_sink"
        self.node_id = node_id
        self.input_schema = _PermissiveSchema
        self.declared_guaranteed_fields = frozenset()
        self.declared_required_fields = frozenset()
        self._on_write_failure = "discard"

    def write(self, rows: list[dict[str, object]], ctx: object) -> SinkWriteResult:
        del rows, ctx
        return SinkWriteResult(
            artifact=ArtifactDescriptor.for_file(path="/tmp/test-sink", content_hash="a" * 64, size_bytes=10),
            diversions=(),
        )

    def _reset_diversion_log(self) -> None:
        return None

    def flush(self) -> None:
        return None


class _SpanContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _SpanFactoryDouble:
    def sink_span(
        self,
        sink_name: str,
        *,
        node_id: str | None = None,
        token_ids: list[str] | None = None,
    ) -> _SpanContext:
        del sink_name, node_id, token_ids
        return _SpanContext()


def _make_permissive_contract() -> SchemaContract:
    return SchemaContract(
        fields=(make_field("value", python_type=str, original_name="value", required=False, source="observed"),),
        mode="OBSERVED",
        locked=True,
    )


class TestTokenResumeOffsetReachesSinkNodeState:
    """Regression guard: per-token resume offset/provenance reaches node_states via sink loop."""

    def test_token_resume_offset_reaches_sink_node_state(self) -> None:
        """Token-carried resume offset and checkpoint id are written into node_states.

        Constructs a TokenInfo with resume_attempt_offset=1, resume_checkpoint_id="ck-test",
        drives it through SinkExecutor.write() with a real LandscapeDB, reads back the
        resulting node_states row, and asserts:
        - attempt == 1  (offset from the token, not the default 0)
        - resume_checkpoint_id == "ck-test" (provenance from the token)

        This is the direct regression guard for the sink batching gap identified in
        ADDENDUM 4: SinkExecutor buffers TokenInfos across WorkItem boundaries, so the
        offset MUST live on the token, not on a WorkItem-level scalar.

        The test uses a real LandscapeDB + real ExecutionRepository + real DataFlowRepository
        (no mock of begin_node_state) to verify the DB write, not just that the argument
        was passed.
        """
        # ── Setup: real landscape DB + factory ──
        setup = make_recorder_with_run(run_id="resume-test-run", source_node_id="src-node")
        db = setup.db
        factory = setup.factory
        run_id = setup.run_id

        # Register a sink node (separate from the source)
        sink_node_id = "node-sink-1"
        register_test_node(
            factory.data_flow,
            run_id,
            sink_node_id,
            node_type=NodeType.SINK,
            plugin_name="test_sink",
        )

        # Create a row and token in the DB (satisfies FK constraints)
        row = factory.data_flow.create_row(
            run_id,
            setup.source_node_id,
            row_index=0,
            data={"value": "hello"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_db = factory.data_flow.create_token(row.row_id)

        # resume_checkpoint_id is a marker-only id (no FK to checkpoints), so no checkpoint
        # row needs to exist — the provenance string is written/read on node_states alone.
        checkpoint_id = "ck-test"

        # ── The critical TokenInfo: carries nonzero resume offset and provenance ──
        contract = _make_permissive_contract()
        row_data = make_row({"value": "hello"}, contract=contract)
        token = TokenInfo(
            row_id=row.row_id,
            token_id=token_db.token_id,
            row_data=row_data,
            resume_attempt_offset=1,  # Nonzero — this must survive to begin_node_state
            resume_checkpoint_id=checkpoint_id,  # Must appear in node_states.resume_checkpoint_id
        )

        # ── Build a real SinkExecutor using the real repositories ──
        executor = SinkExecutor(
            factory.execution,
            factory.data_flow,
            _SpanFactoryDouble(),
            run_id,
        )

        sink = _SinkDouble(node_id=sink_node_id)
        ctx = make_context(landscape=factory.plugin_audit_writer(), run_id=run_id)
        pending = PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)

        # ── Drive the token through the sink ──
        executor.write(
            sink=sink,
            tokens=[token],
            ctx=ctx,
            step_in_pipeline=1,
            sink_name="primary_sink",
            pending_outcome=pending,
        )

        # ── Read back the node_states row and assert the resume fields were written ──
        with db.engine.connect() as conn:
            rows = conn.execute(
                select(node_states_table).where(
                    node_states_table.c.token_id == token.token_id,
                    node_states_table.c.node_id == sink_node_id,
                )
            ).fetchall()

        assert len(rows) == 1, f"Expected exactly 1 node_state for (token_id={token.token_id}, node_id={sink_node_id}), got {len(rows)}"
        ns_row = rows[0]
        assert ns_row.attempt == 1, (
            f"Expected attempt=1 (from token.resume_attempt_offset), got attempt={ns_row.attempt}. "
            "This means the per-token resume offset is not reaching begin_node_state in SinkExecutor."
        )
        assert ns_row.resume_checkpoint_id == checkpoint_id, (
            f"Expected resume_checkpoint_id={checkpoint_id!r} (from token.resume_checkpoint_id), "
            f"got {ns_row.resume_checkpoint_id!r}. "
            "This means the per-token provenance is not reaching begin_node_state in SinkExecutor."
        )

    def test_token_default_offset_reaches_sink_node_state_as_zero(self) -> None:
        """Non-resume tokens (offset=0, checkpoint=None) produce run-1 node_states.

        Verifies the no-op case: existing behavior is unchanged when a TokenInfo
        carries the default resume_attempt_offset=0 / resume_checkpoint_id=None.
        """
        setup = make_recorder_with_run(run_id="default-test-run", source_node_id="src-node")
        db = setup.db
        factory = setup.factory
        run_id = setup.run_id

        sink_node_id = "node-sink-default"
        register_test_node(
            factory.data_flow,
            run_id,
            sink_node_id,
            node_type=NodeType.SINK,
            plugin_name="test_sink",
        )

        row = factory.data_flow.create_row(
            run_id,
            setup.source_node_id,
            row_index=0,
            data={"value": "world"},
            source_row_index=0,
            ingest_sequence=0,
        )
        token_db = factory.data_flow.create_token(row.row_id)

        contract = _make_permissive_contract()
        row_data = make_row({"value": "world"}, contract=contract)
        # Default fields: resume_attempt_offset=0, resume_checkpoint_id=None
        token = TokenInfo(
            row_id=row.row_id,
            token_id=token_db.token_id,
            row_data=row_data,
        )

        executor = SinkExecutor(
            factory.execution,
            factory.data_flow,
            _SpanFactoryDouble(),
            run_id,
        )

        sink = _SinkDouble(node_id=sink_node_id)
        ctx = make_context(landscape=factory.plugin_audit_writer(), run_id=run_id)
        pending = PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)

        executor.write(
            sink=sink,
            tokens=[token],
            ctx=ctx,
            step_in_pipeline=1,
            sink_name="primary_sink",
            pending_outcome=pending,
        )

        with db.engine.connect() as conn:
            rows = conn.execute(
                select(node_states_table).where(
                    node_states_table.c.token_id == token.token_id,
                    node_states_table.c.node_id == sink_node_id,
                )
            ).fetchall()

        assert len(rows) == 1
        ns_row = rows[0]
        assert ns_row.attempt == 0, f"Default token must produce attempt=0 (run-1), got {ns_row.attempt}"
        assert ns_row.resume_checkpoint_id is None, (
            f"Default token must produce NULL resume_checkpoint_id, got {ns_row.resume_checkpoint_id!r}"
        )
