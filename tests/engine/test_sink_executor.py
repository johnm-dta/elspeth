"""Tests for sink executor."""

from typing import Any

import pytest

from elspeth.contracts import NodeType, PendingOutcome, RowOutcome
from elspeth.contracts.schema import SchemaConfig
from tests.conftest import as_sink

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


def make_mock_sink(
    name: str,
    node_id: str,
    *,
    artifact_path: str | None = None,
    artifact_size: int = 1024,
    artifact_hash: str = "mock_hash",
    raises: Exception | None = None,
) -> Any:
    """Factory for creating mock sinks with common patterns.

    Args:
        name: Sink name
        node_id: Node ID to assign
        artifact_path: If provided, returns artifact for this path
        artifact_size: Size for the artifact
        artifact_hash: Hash for the artifact
        raises: If provided, write() raises this exception

    Returns:
        Mock sink instance with write() and flush() methods
    """
    from elspeth.contracts import ArtifactDescriptor
    from elspeth.plugins.context import PluginContext

    class MockSink:
        def __init__(self) -> None:
            self.name = name
            self.node_id = node_id

        def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
            if raises:
                raise raises
            if artifact_path:
                return ArtifactDescriptor.for_file(
                    path=artifact_path,
                    size_bytes=artifact_size,
                    content_hash=artifact_hash,
                )
            raise ValueError("Must provide artifact_path or raises")

        def flush(self) -> None:
            pass  # No-op for tests

    return MockSink()


class TestSinkExecutor:
    """Sink execution with artifact recording."""

    def test_write_records_artifact(self) -> None:
        """Write tokens to sink records artifact in Landscape."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="csv_output",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={"path": "/tmp/output.csv"},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Use factory for simple mock sink
        sink = make_mock_sink(
            name="csv_output",
            node_id=sink_node.node_id,
            artifact_path="/tmp/output.csv",
            artifact_size=1024,
            artifact_hash="abc123",
        )
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        # Create tokens
        tokens = []
        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i * 10},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=sink_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            tokens.append(token)

        # Write tokens to sink
        artifact = executor.write(
            sink=as_sink(sink),
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="mock_sink",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
        )

        # Verify artifact returned with correct info
        assert artifact is not None
        assert artifact.path_or_uri == "file:///tmp/output.csv"
        assert artifact.size_bytes == 1024
        assert artifact.content_hash == "abc123"
        assert artifact.artifact_type == "file"
        assert artifact.sink_node_id == sink_node.node_id

        # Verify artifact recorded in Landscape
        artifacts = recorder.get_artifacts(run.run_id)
        assert len(artifacts) == 1
        assert artifacts[0].artifact_id == artifact.artifact_id

        # Verify node_state created for EACH token (COMPLETED terminal state derivation)
        from elspeth.contracts.audit import NodeStateCompleted

        for token in tokens:
            states = recorder.get_node_states_for_token(token.token_id)
            assert len(states) == 1
            state = states[0]
            assert state.status == "completed"
            assert state.node_id == sink_node.node_id
            # NodeStateCompleted has duration_ms - access directly (no hasattr guards)
            assert isinstance(state, NodeStateCompleted)
            assert state.duration_ms is not None

    def test_write_empty_tokens_returns_none(self) -> None:
        """Write with empty tokens returns None without side effects."""
        from elspeth.contracts import ArtifactDescriptor
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="empty_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class EmptySink:
            name = "empty_sink"
            node_id: str | None = sink_node.node_id

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
                raise AssertionError("Should not be called for empty tokens")

        sink = EmptySink()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        # Write with empty tokens
        artifact = executor.write(
            sink=as_sink(sink),
            tokens=[],
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="empty_sink",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
        )

        assert artifact is None

        # Verify no artifacts recorded
        artifacts = recorder.get_artifacts(run.run_id)
        assert len(artifacts) == 0

    def test_write_exception_records_failure(self) -> None:
        """Sink raising exception still records audit state for all tokens."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="exploding_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Use factory for exception-raising sink
        sink = make_mock_sink(
            name="exploding_sink",
            node_id=sink_node.node_id,
            raises=RuntimeError("disk full!"),
        )
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        # Create tokens
        tokens = []
        for i in range(2):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=sink_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            tokens.append(token)

        # Write should raise
        with pytest.raises(RuntimeError, match="disk full"):
            executor.write(
                sink=as_sink(sink),
                tokens=tokens,
                ctx=ctx,
                step_in_pipeline=5,
                sink_name="failing_sink",
                pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            )

        # Verify failure recorded for ALL tokens
        from elspeth.contracts.audit import NodeStateFailed

        for token in tokens:
            states = recorder.get_node_states_for_token(token.token_id)
            assert len(states) == 1
            state = states[0]
            assert state.status == "failed"
            # NodeStateFailed has duration_ms - access directly (no hasattr guards)
            assert isinstance(state, NodeStateFailed)
            assert state.duration_ms is not None

        # Verify no artifact recorded (write failed)
        artifacts = recorder.get_artifacts(run.run_id)
        assert len(artifacts) == 0

    def test_write_multiple_batches_creates_multiple_artifacts(self) -> None:
        """Multiple sink writes create separate artifacts."""
        from elspeth.contracts import ArtifactDescriptor, TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="batch_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Custom sink with state - can't use factory
        class BatchSink:
            name = "batch_sink"
            node_id: str | None = sink_node.node_id
            _batch_count: int = 0

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
                self._batch_count += 1
                return ArtifactDescriptor.for_file(
                    path=f"/tmp/batch_{self._batch_count}.json",
                    size_bytes=len(rows) * 100,
                    content_hash=f"hash_{self._batch_count}",
                )

            def flush(self) -> None:
                pass  # Mock - no actual flush needed

        sink = BatchSink()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        artifacts = []
        # Write two batches
        for batch_num in range(2):
            tokens = []
            for i in range(2):
                idx = batch_num * 2 + i
                token = TokenInfo(
                    row_id=f"row-{idx}",
                    token_id=f"token-{idx}",
                    row_data={"batch": batch_num, "index": i},
                )
                row = recorder.create_row(
                    run_id=run.run_id,
                    source_node_id=sink_node.node_id,
                    row_index=idx,
                    data=token.row_data,
                    row_id=token.row_id,
                )
                recorder.create_token(row_id=row.row_id, token_id=token.token_id)
                tokens.append(token)

            artifact = executor.write(
                sink=as_sink(sink),
                tokens=tokens,
                ctx=ctx,
                step_in_pipeline=5,
                sink_name="batch_sink",
                pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            )
            artifacts.append(artifact)

        # Verify two distinct artifacts
        assert len(artifacts) == 2
        assert artifacts[0] is not None
        assert artifacts[1] is not None
        assert artifacts[0].artifact_id != artifacts[1].artifact_id
        assert artifacts[0].path_or_uri == "file:///tmp/batch_1.json"
        assert artifacts[1].path_or_uri == "file:///tmp/batch_2.json"

        # Verify both in Landscape
        all_artifacts = recorder.get_artifacts(run.run_id)
        assert len(all_artifacts) == 2

    def test_artifact_linked_to_first_state(self) -> None:
        """Artifact is linked to first token's state_id for audit lineage."""
        from elspeth.contracts import TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="linked_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Use factory for simple mock sink
        sink = make_mock_sink(
            name="linked_sink",
            node_id=sink_node.node_id,
            artifact_path="/tmp/linked.csv",
            artifact_size=512,
            artifact_hash="xyz",
        )
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        # Create multiple tokens
        tokens = []
        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"index": i},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=sink_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            tokens.append(token)

        artifact = executor.write(
            sink=as_sink(sink),
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="linked_sink",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
        )

        # Get first token's state
        first_token_states = recorder.get_node_states_for_token(tokens[0].token_id)
        assert len(first_token_states) == 1
        first_state_id = first_token_states[0].state_id

        # Verify artifact is linked to first state
        assert artifact is not None
        assert artifact.produced_by_state_id == first_state_id

    def test_sink_external_calls_attributed_to_operation(self) -> None:
        """BUG-RECORDER-01: Sink execution sets state_id on context for external call recording."""
        from elspeth.contracts import ArtifactDescriptor, CallStatus, CallType, TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="webhook_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Mock sink that makes external HTTP call during write
        class WebhookSink:
            name = "webhook_sink"
            node_id: str | None = sink_node.node_id

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
                # Sink makes HTTP POST to external webhook
                ctx.record_call(
                    call_type=CallType.HTTP,
                    status=CallStatus.SUCCESS,
                    request_data={"url": "https://webhook.example.com", "rows": rows},
                    response_data={"status": "accepted"},
                    latency_ms=150.0,
                )
                return ArtifactDescriptor.for_file(
                    path="/tmp/webhook_output.json",
                    size_bytes=len(rows) * 100,
                    content_hash="webhook123",
                )

            def flush(self) -> None:
                # Mock sink - no actual flush needed
                pass

        sink = WebhookSink()
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=recorder)
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        # Create multiple tokens
        tokens = []
        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i * 10},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=sink_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            tokens.append(token)

        artifact = executor.write(
            sink=as_sink(sink),
            tokens=tokens,
            ctx=ctx,
            step_in_pipeline=5,
            sink_name="external_call_sink",
            pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
        )

        # Verify sink succeeded
        assert artifact is not None

        # Verify external call was recorded via operation (not token state)
        # With the operations model, sink I/O calls are attributed to the sink_write operation
        from sqlalchemy import select

        from elspeth.core.landscape.schema import operations_table

        # Query operations for this run and sink node
        query = (
            select(operations_table)
            .where(operations_table.c.run_id == run.run_id)
            .where(operations_table.c.node_id == sink_node.node_id)
            .where(operations_table.c.operation_type == "sink_write")
        )
        with db.connection() as conn:
            ops_rows = conn.execute(query).fetchall()
        assert len(ops_rows) == 1
        sink_op = ops_rows[0]

        # Get calls for this operation
        calls = recorder.get_operation_calls(sink_op.operation_id)
        assert len(calls) == 1
        assert calls[0].call_type == CallType.HTTP
        assert calls[0].status == CallStatus.SUCCESS
        assert calls[0].latency_ms == 150.0

        # Verify the operation itself was recorded correctly
        operation = recorder.get_operation(sink_op.operation_id)
        assert operation is not None
        assert operation.operation_type == "sink_write"
        assert operation.status == "completed"
        assert operation.node_id == sink_node.node_id

    def test_flush_exception_records_failure_for_all_tokens(self) -> None:
        """P1-FIX: Sink flush() raising exception still records audit state for all tokens.

        This tests the fix for the audit integrity gap where sink.flush() failures
        left node_states OPEN permanently. The fix wraps flush() in try/except
        and completes all states as FAILED before re-raising.

        Bug: If flush() fails after write() succeeds, node_states were left OPEN
        because complete_node_state() was only called after flush() returned.
        """
        from elspeth.contracts import ArtifactDescriptor, TokenInfo
        from elspeth.contracts.audit import NodeStateFailed
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="flush_exploding_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        # Custom sink where write() succeeds but flush() fails
        class FlushExplodingSink:
            name = "flush_exploding_sink"
            node_id: str | None = sink_node.node_id

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
                # Write succeeds - returns valid artifact
                return ArtifactDescriptor.for_file(
                    path="/tmp/partial_output.csv",
                    size_bytes=len(rows) * 100,
                    content_hash="partial_hash",
                )

            def flush(self) -> None:
                # Flush fails - simulates disk full, network error, etc.
                raise OSError("Flush failed: disk quota exceeded")

        sink = FlushExplodingSink()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        # Create tokens
        tokens = []
        for i in range(3):
            token = TokenInfo(
                row_id=f"row-{i}",
                token_id=f"token-{i}",
                row_data={"value": i * 10},
            )
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id=sink_node.node_id,
                row_index=i,
                data=token.row_data,
                row_id=token.row_id,
            )
            recorder.create_token(row_id=row.row_id, token_id=token.token_id)
            tokens.append(token)

        # Write should raise (flush fails)
        with pytest.raises(IOError, match="disk quota exceeded"):
            executor.write(
                sink=as_sink(sink),
                tokens=tokens,
                ctx=ctx,
                step_in_pipeline=5,
                sink_name="flush_failing_sink",
                pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            )

        # CRITICAL VERIFICATION: All node_states must be FAILED, not OPEN
        # This is the core invariant we're testing
        for token in tokens:
            states = recorder.get_node_states_for_token(token.token_id)
            assert len(states) == 1, f"Expected 1 state for token {token.token_id}"
            state = states[0]
            assert state.status == "failed", (
                f"Expected FAILED status for token {token.token_id}, got {state.status}. "
                "This indicates node_state was left OPEN after flush() failure."
            )
            # Verify it's properly typed as NodeStateFailed
            assert isinstance(state, NodeStateFailed)
            assert state.duration_ms is not None
            # Verify error contains flush phase indicator
            assert state.error_json is not None, "Error JSON should be recorded"
            import json

            error_data = json.loads(state.error_json)
            assert error_data.get("phase") == "flush", (
                f"Expected error phase='flush', got {error_data}. This helps distinguish flush failures from write failures."
            )

        # Verify no artifact recorded (flush failed = not durable)
        artifacts = recorder.get_artifacts(run.run_id)
        assert len(artifacts) == 0, "No artifact should be registered when flush fails. Artifact registration happens after flush succeeds."

    def test_flush_exception_preserves_crash_behavior(self) -> None:
        """P1-FIX: Flush exceptions still propagate (crash) after audit cleanup.

        The original design intent was "if flush fails, we want to crash" (per comment).
        This test verifies that behavior is preserved - we don't silently swallow
        the exception after completing audit states.
        """
        from elspeth.contracts import ArtifactDescriptor, TokenInfo
        from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
        from elspeth.engine.executors import SinkExecutor
        from elspeth.engine.spans import SpanFactory
        from elspeth.plugins.context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        run = recorder.begin_run(config={}, canonical_version="v1")
        sink_node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="crash_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            schema_config=DYNAMIC_SCHEMA,
        )

        class CrashingSink:
            name = "crash_sink"
            node_id: str | None = sink_node.node_id

            def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> ArtifactDescriptor:
                return ArtifactDescriptor.for_file(
                    path="/tmp/crash.csv",
                    size_bytes=100,
                    content_hash="crash_hash",
                )

            def flush(self) -> None:
                raise RuntimeError("CRITICAL: Flush catastrophic failure")

        sink = CrashingSink()
        ctx = PluginContext(run_id=run.run_id, config={})
        executor = SinkExecutor(recorder, SpanFactory(), run.run_id)

        token = TokenInfo(row_id="row-0", token_id="token-0", row_data={"x": 1})
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=sink_node.node_id,
            row_index=0,
            data=token.row_data,
            row_id=token.row_id,
        )
        recorder.create_token(row_id=row.row_id, token_id=token.token_id)

        # Exception MUST propagate (crash behavior preserved)
        with pytest.raises(RuntimeError, match="CRITICAL: Flush catastrophic failure"):
            executor.write(
                sink=as_sink(sink),
                tokens=[token],
                ctx=ctx,
                step_in_pipeline=5,
                sink_name="crash_sink",
                pending_outcome=PendingOutcome(RowOutcome.COMPLETED),
            )

        # Verify audit state was cleaned up BEFORE the crash
        states = recorder.get_node_states_for_token(token.token_id)
        assert len(states) == 1
        assert states[0].status == "failed"
