# tests/integration/plugins/sinks/test_durability.py
"""Integration tests for sink-effect durability and checkpoint ordering.

These tests verify that:
1. Checkpoints are only created after an effect commit is durable.
2. If effect commit fails, no checkpoint is created.
3. If checkpointing fails after commit, execution fails closed.
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
from tests.fixtures.base_classes import create_observed_contract
from tests.fixtures.factories import make_context
from tests.fixtures.landscape import make_factory
from tests.helpers.checkpoint import create_checkpoint


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

        Uses a real effect-capable sink so commit produces actual files and artifacts.
        """
        output_file = tmp_path / "output.csv"
        sink = CSVSink(
            {
                "path": str(output_file),
                "schema": {"mode": "observed"},
            }
        )
        sink.node_id = "sink"
        return sink

    def test_checkpoint_not_created_if_effect_commit_fails(
        self,
        test_env: dict[str, Any],
        mock_graph: ExecutionGraph,
        real_sink: CSVSink,
    ) -> None:
        """Verify checkpoint is not created if the effect cannot commit.

        Scenario:
        1. Sink effect preparation succeeds.
        2. Effect commit raises IOError (simulated crash).
        3. No checkpoint should be created
        4. Resume should process row again

        The callback must remain downstream of durable publication.
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
            factory=factory,
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
            create_checkpoint(
                checkpoint_mgr,
                run_id=run.run_id,
                sequence_number=0,
                barrier_scalars=None,
                graph=mock_graph,
            )
            checkpoint_created = True

        def fail_commit(_plan: object, _ctx: object) -> object:
            raise OSError("Disk full - simulated crash")

        real_sink.commit_effect = fail_commit  # type: ignore[method-assign]

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
                effect_mode="write",
                on_token_written=checkpoint_callback,
            )

        # Verify: Checkpoint was NOT created
        assert checkpoint_created is False
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run.run_id)
        assert checkpoint is None

    def test_checkpoint_failure_raises_after_successful_effect_commit(
        self,
        test_env: dict[str, Any],
        mock_graph: ExecutionGraph,
        real_sink: CSVSink,
    ) -> None:
        """Verify checkpoint failure after durable commit raises AuditIntegrityError.

        Scenario:
        1. Sink effect commits (data is durable).
        3. Checkpoint creation fails (database error)
        4. AuditIntegrityError is raised — the audit trail is inconsistent

        Checkpoint failure after durable commit means the sink artifact exists
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
            factory=factory,
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

        with pytest.raises(AuditIntegrityError, match="Checkpoint failed after durable sink effect"):
            sink_executor.write(
                sink=real_sink,
                tokens=tokens,
                ctx=ctx,
                step_in_pipeline=1,
                sink_name="output",
                pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
                effect_mode="write",
                on_token_written=failing_checkpoint_callback,
            )

    def test_effect_commit_called_before_checkpoint_callback(
        self,
        test_env: dict[str, Any],
        mock_graph: ExecutionGraph,
        real_sink: CSVSink,
    ) -> None:
        """Verify effect commit is called before the checkpoint callback.

        This is the core fix for Bug #2: ensure ordering is:
        1. prepare the exact effect
        2. commit the effect durably
        3. invoke the checkpoint callback
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
            factory=factory,
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
            create_checkpoint(
                checkpoint_mgr,
                run_id=run.run_id,
                sequence_number=0,
                barrier_scalars=None,
                graph=mock_graph,
            )

        original_commit = real_sink.commit_effect

        def tracking_commit(plan: object, effect_ctx: object) -> object:
            result = original_commit(plan, effect_ctx)  # type: ignore[arg-type]
            call_order.append("commit")
            return result

        real_sink.commit_effect = tracking_commit  # type: ignore[method-assign]

        # Execute sink write
        tokens = [token]
        sink_executor.write(
            sink=real_sink,
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=1,
            sink_name="output",
            pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
            effect_mode="write",
            on_token_written=tracking_checkpoint_callback,
        )

        assert call_order == ["commit", "checkpoint"]

        # Verify: Checkpoint was created successfully
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run.run_id)
        assert checkpoint is not None
