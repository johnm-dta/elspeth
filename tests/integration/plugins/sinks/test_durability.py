# tests/integration/plugins/sinks/test_durability.py
"""Integration tests for sink durability and checkpoint ordering (Bug #2).

These tests verify that:
1. Checkpoints are only created AFTER sink writes are durable
2. If flush() fails, no checkpoint is created
3. If checkpoint fails after flush(), the sink artifact exists but is logged
"""

from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import Determinism, NodeType, PendingOutcome, TokenInfo
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.checkpoint import CheckpointManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.executors import SinkExecutor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.sinks.csv_sink import CSVSink
from tests.fixtures.base_classes import create_observed_contract, inject_write_failure
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory


class FlushProbe:
    """Callable test fake for monkey-patching a sink flush method."""

    def __init__(self, *, side_effect: OSError | None = None, call_order: list[str] | None = None) -> None:
        self.side_effect = side_effect
        self.call_order = call_order
        self.call_count = 0

    def __call__(self) -> None:
        self.call_count += 1
        if self.call_order is not None:
            self.call_order.append("flush")
        if self.side_effect is not None:
            raise self.side_effect


class TestSinkDurability:
    """Integration tests for sink durability guarantees."""

    def _register_nodes_raw(self, db: LandscapeDB, run_id: str) -> None:
        """Register nodes using raw SQL to avoid schema_config requirement."""
        from datetime import UTC, datetime

        from elspeth.core.landscape.schema import nodes_table

        now = datetime.now(UTC)

        with db.engine.connect() as conn:
            # Source node
            conn.execute(
                nodes_table.insert().values(
                    node_id="source",
                    run_id=run_id,
                    plugin_name="test_source",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Sink node
            conn.execute(
                nodes_table.insert().values(
                    node_id="sink",
                    run_id=run_id,
                    plugin_name="csv",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.IO_WRITE,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            conn.commit()

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment with database and payload store."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        factory = make_factory(db)

        return {
            "db": db,
            "payload_store": payload_store,
            "checkpoint_manager": checkpoint_mgr,
            "factory": factory,
            "tmp_path": tmp_path,
        }

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a minimal mock graph."""
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        return graph

    @pytest.fixture
    def real_sink(self, tmp_path: Path) -> CSVSink:
        """Create a real CSVSink targeting a temp file.

        Uses a real sink instead of a Mock — write() produces actual files
        and artifacts. Tests that need flush failure patch sink.flush directly.
        """
        output_file = tmp_path / "output.csv"
        sink = inject_write_failure(
            CSVSink(
                {
                    "path": str(output_file),
                    "schema": {"mode": "observed"},
                }
            )
        )
        sink.node_id = "sink"
        return sink

    def test_checkpoint_not_created_if_flush_fails(
        self,
        test_env: dict[str, Any],
        mock_graph: ExecutionGraph,
        real_sink: CSVSink,
    ) -> None:
        """Verify checkpoint not created if sink flush() fails.

        Scenario:
        1. Sink write() succeeds
        2. Sink flush() raises IOError (simulated crash)
        3. No checkpoint should be created
        4. Resume should process row again

        This is Bug #2: checkpoint MUST NOT be created if flush fails,
        because the data is not durable.
        """
        factory = test_env["factory"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]

        # Create run and register nodes
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        self._register_nodes_raw(db, run.run_id)

        # Create sink executor with correct run_id
        sink_executor = SinkExecutor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Create row and token in database
        row_data = {"id": 1, "value": "test"}
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data=row_data,
            source_row_index=0,
            ingest_sequence=0,
        )
        db_token = factory.data_flow.create_token(row_id=row.row_id)

        # Create TokenInfo for executor (includes PipelineRow)
        from elspeth.contracts.schema_contract import PipelineRow

        pipeline_row = PipelineRow(data=row_data, contract=create_observed_contract(row_data))
        token = TokenInfo(
            row_id=row.row_id,
            token_id=db_token.token_id,
            row_data=pipeline_row,
        )

        # Create context
        ctx = make_context(run_id=run.run_id, node_id="sink")

        # Create checkpoint callback
        checkpoint_created = False

        def checkpoint_callback(token_info):
            nonlocal checkpoint_created
            checkpoint_mgr.create_checkpoint(
                run_id=run.run_id,
                sequence_number=0,
                barrier_scalars=None,
                graph=mock_graph,
            )
            checkpoint_created = True

        # Patch real sink's flush to fail.
        failing_flush = FlushProbe(side_effect=OSError("Disk full - simulated crash"))
        real_sink.flush = failing_flush  # type: ignore[method-assign]  # intentional monkey-patch for test

        # Execute sink write - should fail on flush
        tokens = [token]
        with pytest.raises(IOError, match="Disk full"):
            sink_executor.write(
                sink=real_sink,
                tokens=tokens,
                ctx=ctx,
                step_in_pipeline=1,
                sink_name="output",
                pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
                on_token_written=checkpoint_callback,
            )

        # Verify: Checkpoint was NOT created
        assert checkpoint_created is False
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run.run_id)
        assert checkpoint is None

        # Verify: flush() was patched and called (crashed as expected)
        assert failing_flush.call_count == 1

    def test_checkpoint_failure_raises_after_successful_flush(
        self,
        test_env: dict[str, Any],
        mock_graph: ExecutionGraph,
        real_sink: CSVSink,
    ) -> None:
        """Verify checkpoint failure after durable write raises AuditIntegrityError.

        Scenario:
        1. Sink write() succeeds
        2. Sink flush() succeeds (data is durable)
        3. Checkpoint creation fails (database error)
        4. AuditIntegrityError is raised — the audit trail is inconsistent

        Checkpoint failure after durable flush means the sink artifact exists
        but no checkpoint record was created. Silently continuing would cause
        duplicate writes on resume — crashing is the correct response.
        """
        factory = test_env["factory"]
        db = test_env["db"]

        # Create run and register nodes
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        self._register_nodes_raw(db, run.run_id)

        # Create sink executor with correct run_id
        sink_executor = SinkExecutor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Create row and token in database
        row_data = {"id": 1, "value": "test"}
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data=row_data,
            source_row_index=0,
            ingest_sequence=0,
        )
        db_token = factory.data_flow.create_token(row_id=row.row_id)

        # Create TokenInfo for executor (includes PipelineRow)
        from elspeth.contracts.schema_contract import PipelineRow

        pipeline_row = PipelineRow(data=row_data, contract=create_observed_contract(row_data))
        token = TokenInfo(
            row_id=row.row_id,
            token_id=db_token.token_id,
            row_data=pipeline_row,
        )

        # Create context
        ctx = make_context(run_id=run.run_id, node_id="sink")

        # Create checkpoint callback that fails
        def failing_checkpoint_callback(token_info):
            raise RuntimeError("Database connection lost - checkpoint failed")

        tokens = [token]

        with pytest.raises(AuditIntegrityError, match="Checkpoint failed after durable sink write"):
            sink_executor.write(
                sink=real_sink,
                tokens=tokens,
                ctx=ctx,
                step_in_pipeline=1,
                sink_name="output",
                pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
                on_token_written=failing_checkpoint_callback,
            )

    def test_flush_called_before_checkpoint_callback(
        self,
        test_env: dict[str, Any],
        mock_graph: ExecutionGraph,
        real_sink: CSVSink,
    ) -> None:
        """Verify flush() is called BEFORE checkpoint callback.

        This is the core fix for Bug #2: ensure ordering is:
        1. write()
        2. flush() - data is now durable
        3. checkpoint callback - safe to checkpoint
        """
        factory = test_env["factory"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]

        # Create run and register nodes
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        self._register_nodes_raw(db, run.run_id)

        # Create sink executor with correct run_id
        sink_executor = SinkExecutor(
            execution=factory.execution,
            data_flow=factory.data_flow,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Create row and token in database
        row_data = {"id": 1, "value": "test"}
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data=row_data,
            source_row_index=0,
            ingest_sequence=0,
        )
        db_token = factory.data_flow.create_token(row_id=row.row_id)

        # Create TokenInfo for executor (includes PipelineRow)
        from elspeth.contracts.schema_contract import PipelineRow

        pipeline_row = PipelineRow(data=row_data, contract=create_observed_contract(row_data))
        token = TokenInfo(
            row_id=row.row_id,
            token_id=db_token.token_id,
            row_data=pipeline_row,
        )

        # Create context
        ctx = make_context(run_id=run.run_id, node_id="sink")

        # Track call order
        call_order = []

        def tracking_checkpoint_callback(token_info):
            call_order.append("checkpoint")
            checkpoint_mgr.create_checkpoint(
                run_id=run.run_id,
                sequence_number=0,
                barrier_scalars=None,
                graph=mock_graph,
            )

        tracking_flush = FlushProbe(call_order=call_order)
        real_sink.flush = tracking_flush  # type: ignore[method-assign]  # intentional monkey-patch for test

        # Execute sink write
        tokens = [token]
        sink_executor.write(
            sink=real_sink,
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=1,
            sink_name="output",
            pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
            on_token_written=tracking_checkpoint_callback,
        )

        # Verify: flush() was called BEFORE checkpoint callback
        assert call_order == ["flush", "checkpoint"]

        # Verify: Checkpoint was created successfully
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run.run_id)
        assert checkpoint is not None
