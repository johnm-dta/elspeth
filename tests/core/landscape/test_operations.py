# tests/core/landscape/test_operations.py
"""Tests for source/sink operations and the track_operation context manager.

These tests verify Phase 9 requirements from the source-sink-audit-design.md:
- Operation lifecycle (begin → complete)
- Operation call recording
- PluginContext routing (state vs operation)
- XOR constraint enforcement
- Double-complete raises FrameworkBugError
- DB failure doesn't mask original exception
"""

import logging
import time
from datetime import UTC
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import (
    BatchPendingError,
    CallStatus,
    CallType,
    Determinism,
    FrameworkBugError,
    NodeType,
)
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.operations import track_operation
from elspeth.plugins.context import PluginContext

# Dynamic schema for tests that don't care about specific fields
DYNAMIC_SCHEMA = SchemaConfig.from_dict({"fields": "dynamic"})


@pytest.fixture
def db() -> LandscapeDB:
    """Create in-memory database for testing."""
    return LandscapeDB.in_memory()


@pytest.fixture
def recorder(db: LandscapeDB) -> LandscapeRecorder:
    """Create recorder with in-memory database."""
    return LandscapeRecorder(db)


@pytest.fixture
def run_id(recorder: LandscapeRecorder) -> str:
    """Create a run and return its ID."""
    run = recorder.begin_run(config={"test": "config"}, canonical_version="1.0.0")
    return run.run_id


@pytest.fixture
def source_node_id(recorder: LandscapeRecorder, run_id: str) -> str:
    """Register a source node and return its ID."""
    node = recorder.register_node(
        run_id=run_id,
        plugin_name="test_source",
        plugin_version="1.0.0",
        node_type=NodeType.SOURCE,
        config={},
        determinism=Determinism.DETERMINISTIC,
        schema_config=DYNAMIC_SCHEMA,
    )
    return node.node_id


@pytest.fixture
def sink_node_id(recorder: LandscapeRecorder, run_id: str) -> str:
    """Register a sink node and return its ID."""
    node = recorder.register_node(
        run_id=run_id,
        plugin_name="test_sink",
        plugin_version="1.0.0",
        node_type=NodeType.SINK,
        config={},
        determinism=Determinism.DETERMINISTIC,
        schema_config=DYNAMIC_SCHEMA,
    )
    return node.node_id


@pytest.fixture
def plugin_context(recorder: LandscapeRecorder, run_id: str) -> PluginContext:
    """Create a PluginContext for testing."""
    return PluginContext(
        run_id=run_id,
        config={},
        node_id="test_node",
        landscape=recorder,
    )


class TestOperationLifecycle:
    """Tests for begin_operation → complete_operation lifecycle."""

    def test_begin_operation_creates_open_operation(self, recorder: LandscapeRecorder, run_id: str, source_node_id: str) -> None:
        """begin_operation creates operation with status='open'."""
        operation = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )

        assert operation.status == "open"
        assert operation.operation_type == "source_load"
        assert operation.run_id == run_id
        assert operation.node_id == source_node_id
        assert operation.started_at is not None
        assert operation.completed_at is None

    def test_begin_operation_stores_input_data(
        self, recorder: LandscapeRecorder, run_id: str, source_node_id: str, tmp_path: Any, payload_store
    ) -> None:
        """begin_operation stores input_data via payload store."""
        from pathlib import Path

        from elspeth.core.payload_store import FilesystemPayloadStore

        payload_store = FilesystemPayloadStore(Path(tmp_path) / "payloads")
        recorder_with_store = LandscapeRecorder(recorder._db, payload_store=payload_store)

        operation = recorder_with_store.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
            input_data={"source_config": "test.csv"},
        )

        assert operation.input_data_ref is not None
        assert payload_store.exists(operation.input_data_ref)

    def test_complete_operation_sets_completed_status(self, recorder: LandscapeRecorder, run_id: str, source_node_id: str) -> None:
        """complete_operation updates status to 'completed'."""
        operation = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )

        recorder.complete_operation(
            operation_id=operation.operation_id,
            status="completed",
            duration_ms=100.0,
        )

        updated = recorder.get_operation(operation.operation_id)
        assert updated is not None
        assert updated.status == "completed"
        assert updated.completed_at is not None
        assert updated.duration_ms == 100.0

    def test_complete_operation_sets_failed_status_with_error(self, recorder: LandscapeRecorder, run_id: str, source_node_id: str) -> None:
        """complete_operation with status='failed' stores error message."""
        operation = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )

        recorder.complete_operation(
            operation_id=operation.operation_id,
            status="failed",
            error="Connection refused",
            duration_ms=50.0,
        )

        updated = recorder.get_operation(operation.operation_id)
        assert updated is not None
        assert updated.status == "failed"
        assert updated.error_message == "Connection refused"

    def test_complete_operation_sets_pending_status(self, recorder: LandscapeRecorder, run_id: str, sink_node_id: str) -> None:
        """complete_operation with status='pending' for BatchPendingError."""
        operation = recorder.begin_operation(
            run_id=run_id,
            node_id=sink_node_id,
            operation_type="sink_write",
        )

        recorder.complete_operation(
            operation_id=operation.operation_id,
            status="pending",
            duration_ms=25.0,
        )

        updated = recorder.get_operation(operation.operation_id)
        assert updated is not None
        assert updated.status == "pending"


class TestOperationDoubleComplete:
    """Tests for status transition validation."""

    def test_double_complete_raises_framework_bug_error(self, recorder: LandscapeRecorder, run_id: str, source_node_id: str) -> None:
        """Completing an already-completed operation raises FrameworkBugError."""
        operation = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )

        recorder.complete_operation(
            operation_id=operation.operation_id,
            status="completed",
        )

        with pytest.raises(FrameworkBugError, match="already-completed operation"):
            recorder.complete_operation(
                operation_id=operation.operation_id,
                status="completed",
            )

    def test_complete_nonexistent_operation_raises_framework_bug_error(self, recorder: LandscapeRecorder) -> None:
        """Completing a non-existent operation raises FrameworkBugError."""
        with pytest.raises(FrameworkBugError, match="non-existent operation"):
            recorder.complete_operation(
                operation_id="nonexistent_op_id",
                status="completed",
            )


class TestOperationCallRecording:
    """Tests for recording calls attributed to operations."""

    def test_record_operation_call_creates_call_with_operation_id(
        self, recorder: LandscapeRecorder, run_id: str, source_node_id: str
    ) -> None:
        """record_operation_call creates call with operation_id, not state_id."""
        operation = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )

        call = recorder.record_operation_call(
            operation_id=operation.operation_id,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"url": "https://example.com"},
            response_data={"status": 200},
            latency_ms=150.0,
            provider="http",
        )

        assert call.operation_id == operation.operation_id
        assert call.state_id is None
        assert call.call_type == CallType.HTTP
        assert call.status == CallStatus.SUCCESS

    def test_get_operation_calls_returns_calls_ordered_by_index(
        self, recorder: LandscapeRecorder, run_id: str, source_node_id: str
    ) -> None:
        """get_operation_calls returns calls in call_index order."""
        operation = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )

        recorder.record_operation_call(
            operation_id=operation.operation_id,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"call": 1},
            latency_ms=100.0,
            provider="http",
        )
        recorder.record_operation_call(
            operation_id=operation.operation_id,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"call": 2},
            latency_ms=200.0,
            provider="http",
        )

        calls = recorder.get_operation_calls(operation.operation_id)
        assert len(calls) == 2
        assert calls[0].call_index == 0
        assert calls[1].call_index == 1

    def test_operation_call_index_allocation_is_sequential(self, recorder: LandscapeRecorder, run_id: str, source_node_id: str) -> None:
        """allocate_operation_call_index returns sequential indices."""
        operation = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )

        idx0 = recorder.allocate_operation_call_index(operation.operation_id)
        idx1 = recorder.allocate_operation_call_index(operation.operation_id)
        idx2 = recorder.allocate_operation_call_index(operation.operation_id)

        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2


class TestPluginContextCallRouting:
    """Tests for PluginContext.record_call() routing between state and operation."""

    def test_record_call_with_operation_id_routes_to_operation_call(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        source_node_id: str,
    ) -> None:
        """record_call with operation_id set routes to record_operation_call."""
        operation = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )

        ctx = PluginContext(
            run_id=run_id,
            config={},
            node_id=source_node_id,
            landscape=recorder,
            operation_id=operation.operation_id,
        )

        call = ctx.record_call(
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"test": "data"},
            latency_ms=50.0,
        )

        assert call is not None
        assert call.operation_id == operation.operation_id
        assert call.state_id is None

    def test_record_call_with_state_id_routes_to_state_call(self, recorder: LandscapeRecorder, run_id: str, source_node_id: str) -> None:
        """record_call with state_id set routes to record_call (transform path)."""
        # Create row and token
        row = recorder.create_row(
            run_id=run_id,
            row_index=0,
            source_node_id=source_node_id,
            data={"x": 1},
        )
        token = recorder.create_token(row_id=row.row_id)

        # Register a transform node
        transform_node = recorder.register_node(
            run_id=run_id,
            plugin_name="test_transform",
            plugin_version="1.0.0",
            node_type=NodeType.TRANSFORM,
            config={},
            determinism=Determinism.DETERMINISTIC,
            schema_config=DYNAMIC_SCHEMA,
        )

        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=transform_node.node_id,
            run_id=run_id,
            step_index=0,
            input_data={"x": 1},
        )

        ctx = PluginContext(
            run_id=run_id,
            config={},
            node_id=transform_node.node_id,
            landscape=recorder,
            state_id=state.state_id,
        )

        call = ctx.record_call(
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "test"},
            response_data={"response": "ok"},
            latency_ms=100.0,
        )

        assert call is not None
        assert call.state_id == state.state_id
        assert call.operation_id is None

    def test_record_call_with_both_ids_raises_framework_bug_error(
        self, recorder: LandscapeRecorder, run_id: str, source_node_id: str
    ) -> None:
        """record_call with BOTH state_id and operation_id raises FrameworkBugError."""
        ctx = PluginContext(
            run_id=run_id,
            config={},
            node_id=source_node_id,
            landscape=recorder,
            state_id="some_state_id",
            operation_id="some_operation_id",
        )

        with pytest.raises(FrameworkBugError, match="BOTH state_id and operation_id"):
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data={"test": "data"},
            )

    def test_record_call_with_neither_id_raises_framework_bug_error(self, recorder: LandscapeRecorder, run_id: str) -> None:
        """record_call without state_id or operation_id raises FrameworkBugError."""
        ctx = PluginContext(
            run_id=run_id,
            config={},
            node_id="some_node",
            landscape=recorder,
        )

        with pytest.raises(FrameworkBugError, match="without state_id or operation_id"):
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data={"test": "data"},
            )


class TestTrackOperationContextManager:
    """Tests for the track_operation context manager."""

    def test_track_operation_creates_and_completes_operation(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        source_node_id: str,
        plugin_context: PluginContext,
    ) -> None:
        """track_operation creates operation on entry and completes on exit."""
        plugin_context.node_id = source_node_id

        with track_operation(
            recorder=recorder,
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
            ctx=plugin_context,
        ) as handle:
            operation_id = handle.operation.operation_id
            op = recorder.get_operation(operation_id)
            assert op is not None
            assert op.status == "open"

        op = recorder.get_operation(operation_id)
        assert op is not None
        assert op.status == "completed"

    def test_track_operation_sets_context_operation_id(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        source_node_id: str,
        plugin_context: PluginContext,
    ) -> None:
        """track_operation sets ctx.operation_id during the block."""
        assert plugin_context.operation_id is None

        with track_operation(
            recorder=recorder,
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
            ctx=plugin_context,
        ) as handle:
            assert plugin_context.operation_id == handle.operation.operation_id

        assert plugin_context.operation_id is None

    def test_track_operation_on_exception_marks_failed(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        source_node_id: str,
        plugin_context: PluginContext,
    ) -> None:
        """track_operation marks operation as failed on exception."""
        operation_id = ""
        with (
            pytest.raises(ValueError, match="Test error"),
            track_operation(
                recorder=recorder,
                run_id=run_id,
                node_id=source_node_id,
                operation_type="source_load",
                ctx=plugin_context,
            ) as handle,
        ):
            operation_id = handle.operation.operation_id
            raise ValueError("Test error")

        op = recorder.get_operation(operation_id)
        assert op is not None
        assert op.status == "failed"
        assert op.error_message == "Test error"

    def test_track_operation_on_batch_pending_marks_pending(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        sink_node_id: str,
        plugin_context: PluginContext,
    ) -> None:
        """track_operation marks operation as pending on BatchPendingError."""
        operation_id = ""
        with (
            pytest.raises(BatchPendingError),
            track_operation(
                recorder=recorder,
                run_id=run_id,
                node_id=sink_node_id,
                operation_type="sink_write",
                ctx=plugin_context,
            ) as handle,
        ):
            operation_id = handle.operation.operation_id
            raise BatchPendingError(
                batch_id="test-batch-123",
                status="submitted",
            )

        op = recorder.get_operation(operation_id)
        assert op is not None
        assert op.status == "pending"

    def test_track_operation_records_duration(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        source_node_id: str,
        plugin_context: PluginContext,
    ) -> None:
        """track_operation records duration_ms on completion."""
        with track_operation(
            recorder=recorder,
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
            ctx=plugin_context,
        ) as handle:
            operation_id = handle.operation.operation_id
            time.sleep(0.01)

        op = recorder.get_operation(operation_id)
        assert op is not None
        assert op.duration_ms is not None
        assert op.duration_ms >= 10.0

    def test_track_operation_records_output_data(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        sink_node_id: str,
        plugin_context: PluginContext,
        tmp_path: Path,
        payload_store,
    ) -> None:
        """track_operation records handle.output_data on completion."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder_with_store = LandscapeRecorder(recorder._db, payload_store=payload_store)

        with track_operation(
            recorder=recorder_with_store,
            run_id=run_id,
            node_id=sink_node_id,
            operation_type="sink_write",
            ctx=plugin_context,
        ) as handle:
            operation_id = handle.operation.operation_id
            handle.output_data = {"rows_written": 100}

        op = recorder_with_store.get_operation(operation_id)
        assert op is not None
        assert op.output_data_ref is not None


class TestTrackOperationExceptionSafety:
    """Tests for exception handling in track_operation."""

    def test_db_failure_in_complete_does_not_mask_original_exception(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        source_node_id: str,
        plugin_context: PluginContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """DB failure during complete_operation doesn't mask the original exception."""

        def failing_complete(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("DB connection lost")

        with (
            pytest.raises(ValueError, match="Original error"),
            track_operation(
                recorder=recorder,
                run_id=run_id,
                node_id=source_node_id,
                operation_type="source_load",
                ctx=plugin_context,
            ),
        ):
            recorder.complete_operation = failing_complete  # type: ignore[method-assign]
            raise ValueError("Original error")

        assert any("Failed to complete operation" in record.message for record in caplog.records if record.levelno >= logging.CRITICAL)

    def test_context_operation_id_cleared_even_on_db_failure(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        source_node_id: str,
        plugin_context: PluginContext,
    ) -> None:
        """Context operation_id is cleared even when complete_operation fails."""

        def failing_complete(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("DB failure")

        try:
            with track_operation(
                recorder=recorder,
                run_id=run_id,
                node_id=source_node_id,
                operation_type="source_load",
                ctx=plugin_context,
            ):
                recorder.complete_operation = failing_complete  # type: ignore[method-assign]
                raise ValueError("Test error")
        except ValueError:
            pass

        assert plugin_context.operation_id is None

    def test_db_failure_on_successful_operation_raises_db_error(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        source_node_id: str,
        plugin_context: PluginContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When operation succeeds but audit write fails, the DB error is raised.

        This enforces Tier-1 audit integrity: a successful operation with a
        missing audit record must fail the run, not silently continue.
        """

        def failing_complete(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("DB connection lost")

        with (
            pytest.raises(RuntimeError, match="DB connection lost"),
            track_operation(
                recorder=recorder,
                run_id=run_id,
                node_id=source_node_id,
                operation_type="source_load",
                ctx=plugin_context,
            ),
        ):
            # Inject failing complete AFTER operation starts
            recorder.complete_operation = failing_complete  # type: ignore[method-assign]
            # Operation "succeeds" - no exception raised here

        # Verify critical log was emitted
        assert any("Failed to complete operation" in record.message for record in caplog.records if record.levelno >= logging.CRITICAL)

    def test_context_cleared_on_db_failure_with_successful_operation(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        source_node_id: str,
        plugin_context: PluginContext,
    ) -> None:
        """Context operation_id is cleared even when complete fails on successful operation."""

        def failing_complete(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("DB failure")

        try:
            with track_operation(
                recorder=recorder,
                run_id=run_id,
                node_id=source_node_id,
                operation_type="source_load",
                ctx=plugin_context,
            ):
                recorder.complete_operation = failing_complete  # type: ignore[method-assign]
                # No exception - operation succeeds
        except RuntimeError:
            pass

        # Context must be cleaned up even when DB error is raised
        assert plugin_context.operation_id is None


class TestGetOperationsForRun:
    """Tests for get_operations_for_run query method."""

    def test_get_operations_for_run_returns_all_operations(
        self, recorder: LandscapeRecorder, run_id: str, source_node_id: str, sink_node_id: str
    ) -> None:
        """get_operations_for_run returns all operations for a run."""
        source_op = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )
        recorder.complete_operation(source_op.operation_id, status="completed")

        sink_op = recorder.begin_operation(
            run_id=run_id,
            node_id=sink_node_id,
            operation_type="sink_write",
        )
        recorder.complete_operation(sink_op.operation_id, status="completed")

        operations = recorder.get_operations_for_run(run_id)
        assert len(operations) == 2
        op_types = {op.operation_type for op in operations}
        assert op_types == {"source_load", "sink_write"}

    def test_get_operations_for_run_orders_by_started_at(self, recorder: LandscapeRecorder, run_id: str, source_node_id: str) -> None:
        """get_operations_for_run returns operations ordered by started_at."""
        op1 = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )
        time.sleep(0.001)
        op2 = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )

        operations = recorder.get_operations_for_run(run_id)
        assert len(operations) == 2
        assert operations[0].operation_id == op1.operation_id
        assert operations[1].operation_id == op2.operation_id

    def test_get_operations_for_run_empty_for_nonexistent_run(self, recorder: LandscapeRecorder) -> None:
        """get_operations_for_run returns empty list for non-existent run."""
        operations = recorder.get_operations_for_run("nonexistent_run_id")
        assert operations == []


class TestXORConstraintAtDatabaseLevel:
    """Tests for XOR constraint on calls table (state_id XOR operation_id)."""

    def test_xor_constraint_exists_in_schema(self, recorder: LandscapeRecorder) -> None:
        """Verify XOR constraint is defined in schema."""
        from sqlalchemy import text

        with recorder._db.connection() as conn:
            result = conn.execute(text("SELECT sql FROM sqlite_master WHERE name='calls'"))
            schema = result.scalar()
            assert schema is not None
            assert "calls_has_parent" in schema

    def test_call_with_both_state_and_operation_id_raises_integrity_error(
        self, recorder: LandscapeRecorder, run_id: str, source_node_id: str
    ) -> None:
        """Calls with both state_id and operation_id should raise IntegrityError.

        The XOR constraint requires exactly one parent: state_id XOR operation_id.
        Setting both violates the constraint at the database level.
        """
        from datetime import datetime

        from sqlalchemy import insert
        from sqlalchemy.exc import IntegrityError

        from elspeth.core.landscape.schema import calls_table

        # Create a valid state and operation to reference
        row = recorder.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            data={"test": "data"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            node_id=source_node_id,
            run_id=run_id,
            token_id=token.token_id,
            step_index=0,
            input_data={"test": "data"},
        )
        operation = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )

        # Attempt to insert a call with BOTH state_id AND operation_id (violates XOR)
        with recorder._db.connection() as conn, pytest.raises(IntegrityError, match="calls_has_parent"):
            conn.execute(
                insert(calls_table).values(
                    call_id="xor_violation_test",
                    state_id=state.state_id,  # Both set!
                    operation_id=operation.operation_id,  # Both set!
                    call_index=0,
                    call_type=CallType.HTTP.value,
                    status=CallStatus.SUCCESS.value,
                    request_hash="abc123",
                    created_at=datetime.now(UTC),
                )
            )

    def test_call_with_neither_state_nor_operation_id_raises_integrity_error(self, recorder: LandscapeRecorder) -> None:
        """Calls with neither state_id nor operation_id should raise IntegrityError.

        The XOR constraint requires exactly one parent: state_id XOR operation_id.
        Setting neither violates the constraint at the database level.
        """
        from datetime import datetime

        from sqlalchemy import insert
        from sqlalchemy.exc import IntegrityError

        from elspeth.core.landscape.schema import calls_table

        # Attempt to insert a call with NEITHER state_id NOR operation_id (violates XOR)
        with recorder._db.connection() as conn, pytest.raises(IntegrityError, match="calls_has_parent"):
            conn.execute(
                insert(calls_table).values(
                    call_id="xor_violation_neither",
                    state_id=None,  # Neither set!
                    operation_id=None,  # Neither set!
                    call_index=0,
                    call_type=CallType.HTTP.value,
                    status=CallStatus.SUCCESS.value,
                    request_hash="abc123",
                    created_at=datetime.now(UTC),
                )
            )


class TestCallIndexUniquenessConstraints:
    """Tests for partial unique indexes on call_index.

    The calls table has two partial unique indexes:
    - (state_id, call_index) WHERE state_id IS NOT NULL
    - (operation_id, call_index) WHERE operation_id IS NOT NULL

    These ensure call ordering is unambiguous for replay/verification.
    """

    def test_partial_unique_indexes_exist_in_schema(self, db: LandscapeDB) -> None:
        """Verify partial unique indexes are created in the schema."""
        from sqlalchemy import text

        with db.connection() as conn:
            result = conn.execute(text("SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name='calls'"))
            indexes = {row[0]: row[1] for row in result.fetchall()}

            # Verify state_id partial unique index
            assert "ix_calls_state_call_index_unique" in indexes
            state_sql = indexes["ix_calls_state_call_index_unique"]
            assert "UNIQUE" in state_sql
            assert "state_id IS NOT NULL" in state_sql

            # Verify operation_id partial unique index
            assert "ix_calls_operation_call_index_unique" in indexes
            op_sql = indexes["ix_calls_operation_call_index_unique"]
            assert "UNIQUE" in op_sql
            assert "operation_id IS NOT NULL" in op_sql

    def test_duplicate_state_call_index_raises_integrity_error(self, recorder: LandscapeRecorder, run_id: str, source_node_id: str) -> None:
        """Duplicate (state_id, call_index) should raise IntegrityError."""
        from sqlalchemy.exc import IntegrityError

        # Create a row and token to get a valid state_id
        row = recorder.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            data={"test": "data"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            node_id=source_node_id,
            run_id=run_id,
            token_id=token.token_id,
            step_index=0,
            input_data={"test": "data"},
        )

        # First call should succeed
        recorder.record_call(
            state_id=state.state_id,
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"url": "http://example.com"},
            response_data={"status": 200},
        )

        # Second call with same call_index should fail
        with pytest.raises(IntegrityError):
            recorder.record_call(
                state_id=state.state_id,
                call_index=0,  # Duplicate!
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data={"url": "http://example2.com"},
                response_data={"status": 200},
            )

    def test_duplicate_operation_call_index_raises_integrity_error(
        self, recorder: LandscapeRecorder, run_id: str, source_node_id: str
    ) -> None:
        """Duplicate (operation_id, call_index) should raise IntegrityError."""
        from datetime import datetime

        from sqlalchemy import insert
        from sqlalchemy.exc import IntegrityError

        from elspeth.core.landscape.schema import calls_table

        # Create an operation
        operation = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )

        # First call should succeed
        recorder.record_operation_call(
            operation_id=operation.operation_id,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"url": "http://example.com"},
            response_data={"status": 200},
        )

        # Manually insert a duplicate call_index (bypassing allocator)
        with recorder._db.connection() as conn, pytest.raises(IntegrityError):
            conn.execute(
                insert(calls_table).values(
                    call_id="dup_call_123",
                    operation_id=operation.operation_id,
                    call_index=0,  # Duplicate!
                    call_type=CallType.HTTP.value,
                    status=CallStatus.SUCCESS.value,
                    request_hash="abc123",
                    created_at=datetime.now(UTC),
                )
            )

    def test_same_call_index_allowed_for_different_parents(self, recorder: LandscapeRecorder, run_id: str, source_node_id: str) -> None:
        """Same call_index is allowed for different state_ids or operation_ids."""
        # Create two states
        row = recorder.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            data={"test": "data"},
        )
        token1 = recorder.create_token(row_id=row.row_id)
        token2 = recorder.create_token(row_id=row.row_id)

        state1 = recorder.begin_node_state(
            node_id=source_node_id,
            run_id=run_id,
            token_id=token1.token_id,
            step_index=0,
            input_data={"test": "data"},
        )
        state2 = recorder.begin_node_state(
            node_id=source_node_id,
            run_id=run_id,
            token_id=token2.token_id,
            step_index=0,
            input_data={"test": "data"},
        )

        # Both can have call_index=0
        recorder.record_call(
            state_id=state1.state_id,
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"url": "http://example.com"},
            response_data={"status": 200},
        )
        recorder.record_call(
            state_id=state2.state_id,
            call_index=0,  # Same index, different state - OK!
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"url": "http://example.com"},
            response_data={"status": 200},
        )

        # Verify both recorded
        calls1 = recorder.get_calls(state1.state_id)
        calls2 = recorder.get_calls(state2.state_id)
        assert len(calls1) == 1
        assert len(calls2) == 1

    def test_plugin_context_record_call_uses_centralized_allocator(
        self, recorder: LandscapeRecorder, run_id: str, source_node_id: str
    ) -> None:
        """PluginContext.record_call() must use recorder.allocate_call_index().

        This verifies the fix for P1-2026-01-31-context-record-call-bypasses-allocator.

        The bug: PluginContext maintained its own _call_index counter instead of
        delegating to LandscapeRecorder.allocate_call_index(). This caused duplicate
        (state_id, call_index) pairs when mixing ctx.record_call() with audited clients.

        This test interleaves:
        1. recorder.allocate_call_index() - simulates what audited clients do
        2. ctx.record_call() - what transforms call directly

        If ctx.record_call() uses its own counter, both will produce call_index=0.
        If ctx.record_call() delegates to the centralized allocator, they coordinate.
        """
        # Create a row/token/state for testing
        row = recorder.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            data={"test": "data"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            node_id=source_node_id,
            run_id=run_id,
            token_id=token.token_id,
            step_index=0,
            input_data={"test": "data"},
        )

        ctx = PluginContext(
            run_id=run_id,
            config={},
            node_id=source_node_id,
            landscape=recorder,
            state_id=state.state_id,
        )

        # Step 1: Audited client allocates index (simulates AuditedLLMClient.query())
        # This is what audited clients do internally via _next_call_index()
        client_index_1 = recorder.allocate_call_index(state.state_id)
        assert client_index_1 == 0

        # Record the call (as if AuditedLLMClient recorded it)
        recorder.record_call(
            state_id=state.state_id,
            call_index=client_index_1,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "test"},
            response_data={"response": "ok"},
        )

        # Step 2: Transform calls ctx.record_call() directly
        # BUG: If ctx uses its own _call_index, this will be 0 (collision!)
        # FIX: If ctx delegates to allocator, this will be 1 (no collision)
        call_2 = ctx.record_call(
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"url": "http://example.com"},
            response_data={"status": 200},
            latency_ms=50.0,
        )

        # Verify ctx.record_call() got a DIFFERENT index than the audited client
        assert call_2 is not None
        assert call_2.call_index == 1, (
            f"Expected call_index=1 but got {call_2.call_index}. PluginContext.record_call() is not using centralized allocator!"
        )

        # Step 3: Another audited client call (simulates second LLM call)
        client_index_2 = recorder.allocate_call_index(state.state_id)
        assert client_index_2 == 2

        # Step 4: Another ctx.record_call()
        call_4 = ctx.record_call(
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"url": "http://example2.com"},
            response_data={"status": 200},
            latency_ms=50.0,
        )
        assert call_4 is not None
        assert call_4.call_index == 3, (
            f"Expected call_index=3 but got {call_4.call_index}. PluginContext.record_call() is not coordinating with allocator!"
        )

        # Verify all calls recorded with unique indices
        calls = recorder.get_calls(state.state_id)
        indices = [c.call_index for c in calls]
        # We only recorded 3 calls (indices 0, 1, 3 - index 2 was allocated but not recorded)
        assert len(calls) == 3
        assert sorted(indices) == [0, 1, 3], f"Expected indices [0, 1, 3] but got {sorted(indices)}"


class TestConcurrentCallIndexAllocation:
    """Tests for thread-safe call index allocation.

    The allocate_call_index and allocate_operation_call_index methods must be
    thread-safe to ensure unique call indices when multiple threads allocate
    concurrently. These tests verify the lock mechanism works correctly.

    Note: These tests verify the allocator logic only, not full DB writes,
    because SQLite in-memory databases don't support multi-threaded access.
    """

    def test_concurrent_operation_call_index_allocation_is_unique(
        self, recorder: LandscapeRecorder, run_id: str, source_node_id: str
    ) -> None:
        """Multiple threads allocating operation call indices get unique values.

        This verifies the thread-safety of allocate_operation_call_index().
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Create an operation
        operation = recorder.begin_operation(
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
        )

        num_threads = 100
        allocated_indices: list[int] = []
        lock = threading.Lock()

        def allocate_task() -> int:
            """Task that allocates an index and returns it."""
            idx = recorder.allocate_operation_call_index(operation.operation_id)
            with lock:
                allocated_indices.append(idx)
            return idx

        # Execute concurrent allocations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(allocate_task) for _ in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # Verify all indices are unique
        assert len(allocated_indices) == num_threads
        assert len(set(allocated_indices)) == num_threads, f"Duplicate indices found: {sorted(allocated_indices)}"

        # Verify indices are 0 through num_threads-1 (no gaps)
        assert set(allocated_indices) == set(range(num_threads))

    def test_concurrent_state_call_index_allocation_is_unique(self, recorder: LandscapeRecorder, run_id: str, source_node_id: str) -> None:
        """Multiple threads allocating state call indices get unique values.

        This verifies the thread-safety of allocate_call_index() for state calls.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Create a state
        row = recorder.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            data={"test": "data"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            node_id=source_node_id,
            run_id=run_id,
            token_id=token.token_id,
            step_index=0,
            input_data={"test": "data"},
        )

        num_threads = 100
        allocated_indices: list[int] = []
        lock = threading.Lock()

        def allocate_task() -> int:
            """Task that allocates an index and returns it."""
            idx = recorder.allocate_call_index(state.state_id)
            with lock:
                allocated_indices.append(idx)
            return idx

        # Execute concurrent allocations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(allocate_task) for _ in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # Verify all indices are unique
        assert len(allocated_indices) == num_threads
        assert len(set(allocated_indices)) == num_threads, f"Duplicate indices found: {sorted(allocated_indices)}"

        # Verify indices are 0 through num_threads-1 (no gaps)
        assert set(allocated_indices) == set(range(num_threads))

    def test_independent_state_allocators_do_not_interfere(self, recorder: LandscapeRecorder, run_id: str, source_node_id: str) -> None:
        """Allocations for different state_ids should be independent.

        Each state_id should have its own counter starting at 0.
        """
        # Create two states
        row = recorder.create_row(
            run_id=run_id,
            source_node_id=source_node_id,
            row_index=0,
            data={"test": "data"},
        )
        token1 = recorder.create_token(row_id=row.row_id)
        token2 = recorder.create_token(row_id=row.row_id)

        state1 = recorder.begin_node_state(
            node_id=source_node_id,
            run_id=run_id,
            token_id=token1.token_id,
            step_index=0,
            input_data={"test": "data"},
        )
        state2 = recorder.begin_node_state(
            node_id=source_node_id,
            run_id=run_id,
            token_id=token2.token_id,
            step_index=0,
            input_data={"test": "data"},
        )

        # Allocate indices for state1
        idx1_0 = recorder.allocate_call_index(state1.state_id)
        idx1_1 = recorder.allocate_call_index(state1.state_id)
        idx1_2 = recorder.allocate_call_index(state1.state_id)

        # Allocate indices for state2 (should start at 0, not 3)
        idx2_0 = recorder.allocate_call_index(state2.state_id)
        idx2_1 = recorder.allocate_call_index(state2.state_id)

        # Each state has independent counter
        assert [idx1_0, idx1_1, idx1_2] == [0, 1, 2]
        assert [idx2_0, idx2_1] == [0, 1]


class TestTrackOperationContextGuards:
    """Tests for context manager guard conditions."""

    def test_track_operation_restores_previous_operation_id(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        source_node_id: str,
        plugin_context: PluginContext,
    ) -> None:
        """track_operation should restore previous operation_id after completion.

        This is important when operations are nested or when the context is
        reused across multiple operations.
        """
        # Set a "previous" operation_id on the context
        plugin_context.operation_id = "previous_operation_123"

        with track_operation(
            recorder=recorder,
            run_id=run_id,
            node_id=source_node_id,
            operation_type="source_load",
            ctx=plugin_context,
        ):
            # During the operation, context has the new operation_id
            assert plugin_context.operation_id is not None
            assert plugin_context.operation_id != "previous_operation_123"

        # After completion, previous operation_id is restored
        assert plugin_context.operation_id == "previous_operation_123"

    def test_track_operation_restores_previous_operation_id_on_exception(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        source_node_id: str,
        plugin_context: PluginContext,
    ) -> None:
        """Previous operation_id is restored even when operation fails."""
        plugin_context.operation_id = "previous_operation_456"

        with (
            pytest.raises(ValueError, match="Test failure"),
            track_operation(
                recorder=recorder,
                run_id=run_id,
                node_id=source_node_id,
                operation_type="source_load",
                ctx=plugin_context,
            ),
        ):
            raise ValueError("Test failure")

        # Previous operation_id restored despite exception
        assert plugin_context.operation_id == "previous_operation_456"

    def test_context_reusable_after_track_operation_exception(
        self,
        recorder: LandscapeRecorder,
        run_id: str,
        source_node_id: str,
        plugin_context: PluginContext,
    ) -> None:
        """PluginContext should be reusable after a track_operation raises.

        This verifies that the context is properly cleaned up and can be
        used in subsequent track_operation calls.
        """
        # First operation fails
        with (
            pytest.raises(ValueError),
            track_operation(
                recorder=recorder,
                run_id=run_id,
                node_id=source_node_id,
                operation_type="source_load",
                ctx=plugin_context,
            ),
        ):
            raise ValueError("First operation failed")

        # Context should be clean (operation_id was None before)
        assert plugin_context.operation_id is None

        # Second operation should work normally
        with track_operation(
            recorder=recorder,
            run_id=run_id,
            node_id=source_node_id,
            operation_type="sink_write",
            ctx=plugin_context,
        ) as handle:
            # Operation succeeds
            handle.output_data = {"result": "success"}

        # Context should be clean again
        assert plugin_context.operation_id is None

        # Verify both operations were recorded
        operations = recorder.get_operations_for_run(run_id)
        assert len(operations) == 2
        assert operations[0].status == "failed"
        assert operations[1].status == "completed"
