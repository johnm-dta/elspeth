# tests/integration/test_sink_durability.py
"""Integration tests for sink durability and checkpoint ordering (Bug #2).

These tests verify that:
1. Checkpoints are only created AFTER sink writes are durable
2. If flush() fails, no checkpoint is created
3. If checkpoint fails after flush(), the sink artifact exists but is logged
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from elspeth.contracts import Determinism, NodeType, PendingOutcome, RowOutcome, TokenInfo
from elspeth.core.checkpoint import CheckpointManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.executors import SinkExecutor
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.context import PluginContext


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
        recorder = LandscapeRecorder(db)

        return {
            "db": db,
            "payload_store": payload_store,
            "checkpoint_manager": checkpoint_mgr,
            "recorder": recorder,
            "tmp_path": tmp_path,
        }

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a minimal mock graph."""
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        return graph

    @pytest.fixture
    def mock_sink(self, tmp_path: Path) -> Mock:
        """Create a mock sink that simulates writes."""
        from elspeth.contracts import ArtifactDescriptor

        sink = Mock()
        sink.name = "csv"
        sink.node_id = "sink"
        sink.plugin_version = "1.0"
        sink.determinism = Determinism.IO_WRITE

        # Mock write() to return artifact descriptor
        def mock_write(rows, ctx):
            return ArtifactDescriptor.for_file(
                path=str(tmp_path / "output.csv"),
                content_hash="abc123",
                size_bytes=100,
            )

        sink.write = Mock(side_effect=mock_write)
        sink.flush = Mock()  # Default: successful flush
        sink.close = Mock()

        return sink

    def test_checkpoint_not_created_if_flush_fails(
        self,
        test_env: dict[str, Any],
        mock_graph: ExecutionGraph,
        mock_sink: Mock,
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
        recorder = test_env["recorder"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]

        # Create run and register nodes
        run = recorder.begin_run(config={}, canonical_version="v1")
        self._register_nodes_raw(db, run.run_id)

        # Create sink executor with correct run_id
        sink_executor = SinkExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Create row and token in database
        row_data = {"id": 1, "value": "test"}
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data=row_data,
        )
        db_token = recorder.create_token(row_id=row.row_id)

        # Create TokenInfo for executor (includes row_data)
        token = TokenInfo(
            row_id=row.row_id,
            token_id=db_token.token_id,
            row_data=row_data,
        )

        # Create context
        ctx = PluginContext(
            run_id=run.run_id,
            config={},
        )
        ctx.node_id = "sink"

        # Create checkpoint callback
        checkpoint_created = False

        def checkpoint_callback(token_info):
            nonlocal checkpoint_created
            checkpoint_mgr.create_checkpoint(
                run_id=run.run_id,
                token_id=token_info.token_id,
                node_id="sink",
                sequence_number=0,
                graph=mock_graph,
            )
            checkpoint_created = True

        # Configure mock sink to fail on flush
        mock_sink.flush.side_effect = OSError("Disk full - simulated crash")

        # Execute sink write - should fail on flush
        tokens = [token]
        with pytest.raises(IOError, match="Disk full"):
            sink_executor.write(
                sink=mock_sink,
                tokens=tokens,
                ctx=ctx,
                step_in_pipeline=1,
                sink_name="output",
                pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
                on_token_written=checkpoint_callback,
            )

        # Verify: Checkpoint was NOT created
        assert checkpoint_created is False
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run.run_id)
        assert checkpoint is None

        # Verify: write() was called but flush() crashed
        mock_sink.write.assert_called_once()
        mock_sink.flush.assert_called_once()

    def test_checkpoint_failure_logged_after_successful_flush(
        self,
        test_env: dict[str, Any],
        mock_graph: ExecutionGraph,
        mock_sink: Mock,
    ) -> None:
        """Verify sink write persists even if checkpoint creation fails.

        Scenario:
        1. Sink write() succeeds
        2. Sink flush() succeeds (data is durable)
        3. Checkpoint creation fails (database error)
        4. Artifact should still be registered
        5. Error should be logged (not raised)

        This is Bug #10: checkpoint failure after durable flush cannot
        be rolled back, so we log the error and continue. Resume will
        replay the row (duplicate write - acceptable for RC-1).
        """
        recorder = test_env["recorder"]
        db = test_env["db"]

        # Create run and register nodes
        run = recorder.begin_run(config={}, canonical_version="v1")
        self._register_nodes_raw(db, run.run_id)

        # Create sink executor with correct run_id
        sink_executor = SinkExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Create row and token in database
        row_data = {"id": 1, "value": "test"}
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data=row_data,
        )
        db_token = recorder.create_token(row_id=row.row_id)

        # Create TokenInfo for executor (includes row_data)
        token = TokenInfo(
            row_id=row.row_id,
            token_id=db_token.token_id,
            row_data=row_data,
        )

        # Create context
        ctx = PluginContext(
            run_id=run.run_id,
            config={},
        )
        ctx.node_id = "sink"

        # Create checkpoint callback that fails
        def failing_checkpoint_callback(token_info):
            raise RuntimeError("Database connection lost - checkpoint failed")

        # Execute sink write with failing checkpoint callback
        # This should NOT raise - error should be logged
        tokens = [token]

        # Patch logger to capture error log
        with patch("elspeth.engine.executors.logger") as mock_logger:
            artifact = sink_executor.write(
                sink=mock_sink,
                tokens=tokens,
                ctx=ctx,
                step_in_pipeline=1,
                sink_name="output",
                pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
                on_token_written=failing_checkpoint_callback,
            )

            # Verify: Sink write completed successfully
            assert artifact is not None
            mock_sink.write.assert_called_once()
            mock_sink.flush.assert_called_once()

            # Verify: Error was logged (not raised)
            assert mock_logger.error.called
            error_call = mock_logger.error.call_args
            assert "Checkpoint failed after durable sink write" in error_call[0][0]
            assert token.token_id in error_call[0]

        # Verify: Artifact was registered in database
        artifacts = recorder.get_artifacts(run.run_id)
        assert len(artifacts) == 1
        # path_or_uri is a file:// URI, so check it ends with the expected path
        assert artifacts[0].path_or_uri.endswith(str(test_env["tmp_path"] / "output.csv"))

    def test_flush_called_before_checkpoint_callback(
        self,
        test_env: dict[str, Any],
        mock_graph: ExecutionGraph,
        mock_sink: Mock,
    ) -> None:
        """Verify flush() is called BEFORE checkpoint callback.

        This is the core fix for Bug #2: ensure ordering is:
        1. write()
        2. flush() - data is now durable
        3. checkpoint callback - safe to checkpoint
        """
        recorder = test_env["recorder"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]

        # Create run and register nodes
        run = recorder.begin_run(config={}, canonical_version="v1")
        self._register_nodes_raw(db, run.run_id)

        # Create sink executor with correct run_id
        sink_executor = SinkExecutor(
            recorder=recorder,
            span_factory=SpanFactory(),
            run_id=run.run_id,
        )

        # Create row and token in database
        row_data = {"id": 1, "value": "test"}
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data=row_data,
        )
        db_token = recorder.create_token(row_id=row.row_id)

        # Create TokenInfo for executor (includes row_data)
        token = TokenInfo(
            row_id=row.row_id,
            token_id=db_token.token_id,
            row_data=row_data,
        )

        # Create context
        ctx = PluginContext(
            run_id=run.run_id,
            config={},
        )
        ctx.node_id = "sink"

        # Track call order
        call_order = []

        def tracking_flush():
            call_order.append("flush")

        def tracking_checkpoint_callback(token_info):
            call_order.append("checkpoint")
            checkpoint_mgr.create_checkpoint(
                run_id=run.run_id,
                token_id=token_info.token_id,
                node_id="sink",
                sequence_number=0,
                graph=mock_graph,
            )

        mock_sink.flush = Mock(side_effect=tracking_flush)

        # Execute sink write
        tokens = [token]
        sink_executor.write(
            sink=mock_sink,
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=1,
            sink_name="output",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            on_token_written=tracking_checkpoint_callback,
        )

        # Verify: flush() was called BEFORE checkpoint callback
        assert call_order == ["flush", "checkpoint"]

        # Verify: Checkpoint was created successfully
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run.run_id)
        assert checkpoint is not None
        assert checkpoint.token_id == token.token_id
