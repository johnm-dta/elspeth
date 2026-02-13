from __future__ import annotations

import pytest

from elspeth.contracts import CallStatus, CallType, FrameworkBugError, NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape import LandscapeDB, LandscapeRecorder
from elspeth.core.landscape.schema import operations_table

_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _setup(*, run_id: str = "run-1") -> tuple[LandscapeDB, LandscapeRecorder, str]:
    """Create DB, recorder, run, nodes, row, token, and node_state. Returns (db, recorder, state_id)."""
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    recorder.begin_run(config={}, canonical_version="v1", run_id=run_id)
    recorder.register_node(
        run_id=run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-0",
        schema_config=_DYNAMIC_SCHEMA,
    )
    recorder.register_node(
        run_id=run_id,
        plugin_name="transform",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        node_id="transform-1",
        schema_config=_DYNAMIC_SCHEMA,
    )
    recorder.create_row(run_id, "source-0", 0, {"name": "test"}, row_id="row-1")
    recorder.create_token("row-1", token_id="tok-1")
    state = recorder.begin_node_state("tok-1", "transform-1", run_id, 0, {"name": "test"}, state_id="state-1")
    return db, recorder, state.state_id


def _setup_with_operation(
    *,
    run_id: str = "run-1",
) -> tuple[LandscapeDB, LandscapeRecorder, str, str]:
    """Create DB, recorder, run, source node, and a source_load operation. Returns (db, recorder, state_id, operation_id)."""
    db, recorder, state_id = _setup(run_id=run_id)
    op = recorder.begin_operation(run_id, "source-0", "source_load")
    return db, recorder, state_id, op.operation_id


class TestAllocateCallIndex:
    """Tests for thread-safe call index allocation per node_state."""

    def test_sequential_allocation_starts_at_zero(self):
        _db, recorder, state_id = _setup()

        idx0 = recorder.allocate_call_index(state_id)
        idx1 = recorder.allocate_call_index(state_id)
        idx2 = recorder.allocate_call_index(state_id)

        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

    def test_independent_per_state_id(self):
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        recorder.begin_run(config={}, canonical_version="v1", run_id="run-1")
        recorder.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id="run-1",
            plugin_name="t1",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="transform-1",
            schema_config=_DYNAMIC_SCHEMA,
        )
        recorder.register_node(
            run_id="run-1",
            plugin_name="t2",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="transform-2",
            schema_config=_DYNAMIC_SCHEMA,
        )
        recorder.create_row("run-1", "source-0", 0, {"x": 1}, row_id="row-1")
        recorder.create_token("row-1", token_id="tok-a")
        recorder.create_token("row-1", token_id="tok-b")
        state_a = recorder.begin_node_state("tok-a", "transform-1", "run-1", 0, {"x": 1}, state_id="state-a")
        state_b = recorder.begin_node_state("tok-b", "transform-2", "run-1", 0, {"x": 1}, state_id="state-b")

        assert recorder.allocate_call_index(state_a.state_id) == 0
        assert recorder.allocate_call_index(state_b.state_id) == 0
        assert recorder.allocate_call_index(state_a.state_id) == 1
        assert recorder.allocate_call_index(state_b.state_id) == 1

    def test_single_allocation(self):
        _db, recorder, state_id = _setup()

        idx = recorder.allocate_call_index(state_id)

        assert idx == 0

    def test_seeds_from_database_on_recorder_recreation(self):
        """Simulate resume: new recorder on same DB continues call indices."""
        db, recorder, state_id = _setup()

        # Record 3 calls with the first recorder
        for i in range(3):
            idx = recorder.allocate_call_index(state_id)
            recorder.record_call(
                state_id,
                idx,
                CallType.LLM,
                CallStatus.SUCCESS,
                request_data={"i": i},
                response_data={"r": i},
            )

        # Create a NEW recorder on the same DB (simulates resume)
        recorder2 = LandscapeRecorder(db)

        # New recorder should seed from DB and continue at index 3
        idx = recorder2.allocate_call_index(state_id)
        assert idx == 3

        idx = recorder2.allocate_call_index(state_id)
        assert idx == 4

    def test_seeds_operation_call_index_on_recorder_recreation(self):
        """Simulate resume: new recorder seeds operation call indices from DB."""
        db, recorder, _state_id, operation_id = _setup_with_operation()

        # Record 2 calls via the first recorder
        for _i in range(2):
            recorder.record_operation_call(
                operation_id,
                CallType.HTTP,
                CallStatus.SUCCESS,
                request_data={"url": "https://example.com"},
                response_data={"status": 200},
            )

        # Create a NEW recorder on the same DB (simulates resume)
        recorder2 = LandscapeRecorder(db)

        # New recorder should seed from DB and continue at index 2
        idx = recorder2.allocate_operation_call_index(operation_id)
        assert idx == 2

    def test_fresh_state_id_starts_at_zero_with_db_seeding(self):
        """A state_id with no DB entries still starts at 0."""
        _db, recorder, _state_id = _setup()

        # Allocate for a state_id that has no recorded calls
        idx = recorder.allocate_call_index("brand-new-state-id")
        assert idx == 0


class TestRecordCall:
    """Tests for recording external calls linked to a node_state."""

    def test_creates_call_with_request_hash(self):
        _db, recorder, state_id = _setup()
        idx = recorder.allocate_call_index(state_id)

        call = recorder.record_call(
            state_id,
            idx,
            CallType.LLM,
            CallStatus.SUCCESS,
            request_data={"prompt": "hello"},
            response_data={"text": "world"},
            latency_ms=42,
        )

        assert call.call_id is not None
        assert call.call_index == 0
        assert call.call_type == CallType.LLM
        assert call.status == CallStatus.SUCCESS
        assert call.request_hash is not None
        assert call.state_id == state_id
        assert call.latency_ms == 42

    def test_roundtrip_via_response_hash(self):
        _db, recorder, state_id = _setup()
        idx = recorder.allocate_call_index(state_id)

        call = recorder.record_call(
            state_id,
            idx,
            CallType.HTTP,
            CallStatus.SUCCESS,
            request_data={"url": "https://example.com"},
            response_data={"status": 200},
        )

        assert call.response_hash is not None
        assert call.request_hash is not None
        assert call.call_type == CallType.HTTP

    def test_error_call_has_error_json(self):
        _db, recorder, state_id = _setup()
        idx = recorder.allocate_call_index(state_id)

        call = recorder.record_call(
            state_id,
            idx,
            CallType.LLM,
            CallStatus.ERROR,
            request_data={"prompt": "fail"},
            error={"code": "rate_limit", "message": "Too many requests"},
            latency_ms=100,
        )

        assert call.status == CallStatus.ERROR
        assert call.error_json is not None
        assert "rate_limit" in call.error_json

    def test_call_with_refs(self):
        _db, recorder, state_id = _setup()
        idx = recorder.allocate_call_index(state_id)

        call = recorder.record_call(
            state_id,
            idx,
            CallType.SQL,
            CallStatus.SUCCESS,
            request_data={"query": "SELECT 1"},
            request_ref="req-ref-abc",
            response_ref="resp-ref-xyz",
        )

        assert call.request_ref == "req-ref-abc"
        assert call.response_ref == "resp-ref-xyz"

    def test_call_without_response_data(self):
        _db, recorder, state_id = _setup()
        idx = recorder.allocate_call_index(state_id)

        call = recorder.record_call(
            state_id,
            idx,
            CallType.FILESYSTEM,
            CallStatus.SUCCESS,
            request_data={"path": "/tmp/file.txt"},
        )

        assert call.response_hash is None
        assert call.call_type == CallType.FILESYSTEM

    def test_multiple_calls_on_same_state(self):
        _db, recorder, state_id = _setup()

        calls = []
        for i in range(3):
            idx = recorder.allocate_call_index(state_id)
            call = recorder.record_call(
                state_id,
                idx,
                CallType.LLM,
                CallStatus.SUCCESS,
                request_data={"prompt": f"call-{i}"},
                response_data={"text": f"response-{i}"},
            )
            calls.append(call)

        assert [c.call_index for c in calls] == [0, 1, 2]
        assert len({c.call_id for c in calls}) == 3


class TestBeginOperation:
    """Tests for beginning source/sink operations."""

    def test_creates_operation_with_open_status(self):
        _db, recorder, _state_id = _setup()

        op = recorder.begin_operation("run-1", "source-0", "source_load")

        assert op.operation_id is not None
        assert op.run_id == "run-1"
        assert op.node_id == "source-0"
        assert op.operation_type == "source_load"
        assert op.status == "open"
        assert op.started_at is not None
        assert op.completed_at is None

    def test_generates_unique_ids(self):
        _db, recorder, _state_id = _setup()

        op1 = recorder.begin_operation("run-1", "source-0", "source_load")
        op2 = recorder.begin_operation("run-1", "source-0", "source_load")

        assert op1.operation_id != op2.operation_id

    def test_sink_write_operation(self):
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        recorder.begin_run(config={}, canonical_version="v1", run_id="run-1")
        recorder.register_node(
            run_id="run-1",
            plugin_name="csv_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink-0",
            schema_config=_DYNAMIC_SCHEMA,
        )

        op = recorder.begin_operation("run-1", "sink-0", "sink_write")

        assert op.operation_type == "sink_write"
        assert op.status == "open"

    def test_operation_with_input_data(self):
        _db, recorder, _state_id = _setup()

        op = recorder.begin_operation("run-1", "source-0", "source_load", input_data={"path": "/data/input.csv"})

        assert op.operation_id is not None
        assert op.status == "open"
        assert op.input_data_hash == stable_hash({"path": "/data/input.csv"})

    def test_operation_without_input_data_has_no_hash(self):
        _db, recorder, _state_id = _setup()

        op = recorder.begin_operation("run-1", "source-0", "source_load")

        assert op.input_data_hash is None
        assert op.input_data_ref is None

    def test_input_hash_persisted_without_payload_store(self):
        """Hash must be computed even when no payload store is configured."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)  # No payload_store
        recorder.begin_run(config={}, canonical_version="v1", run_id="run-1")
        recorder.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )

        op = recorder.begin_operation("run-1", "source-0", "source_load", input_data={"file": "data.csv"})

        assert op.input_data_hash == stable_hash({"file": "data.csv"})
        assert op.input_data_ref is None  # No payload store → no ref

        # Verify hash round-trips through the database
        fetched = recorder.get_operation(op.operation_id)
        assert fetched.input_data_hash == op.input_data_hash


class TestCompleteOperation:
    """Tests for completing operations with status, output, and error handling."""

    def test_completes_with_status_and_duration(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()

        recorder.complete_operation(op_id, "completed", duration_ms=150)

        op = recorder.get_operation(op_id)
        assert op.status == "completed"
        assert op.completed_at is not None
        assert op.duration_ms == 150

    def test_raises_on_invalid_status_from_db(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()
        recorder._ops.execute_update(operations_table.update().where(operations_table.c.operation_id == op_id).values(status="corrupt"))

        with pytest.raises(ValueError, match="status"):
            recorder.get_operation(op_id)

    def test_raises_on_invalid_operation_type_from_db(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()
        recorder._ops.execute_update(
            operations_table.update().where(operations_table.c.operation_id == op_id).values(operation_type="corrupt")
        )

        with pytest.raises(ValueError, match="operation_type"):
            recorder.get_operation(op_id)

    def test_completes_with_failure(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()

        recorder.complete_operation(op_id, "failed", error="File not found", duration_ms=5)

        op = recorder.get_operation(op_id)
        assert op.status == "failed"
        assert op.error_message == "File not found"
        assert op.duration_ms == 5

    def test_completes_with_output_data(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()

        recorder.complete_operation(op_id, "completed", output_data={"rows_loaded": 100}, duration_ms=500)

        op = recorder.get_operation(op_id)
        assert op.status == "completed"
        assert op.output_data_hash == stable_hash({"rows_loaded": 100})

    def test_output_hash_none_when_no_output_data(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()

        recorder.complete_operation(op_id, "completed", duration_ms=150)

        op = recorder.get_operation(op_id)
        assert op.output_data_hash is None

    def test_output_hash_persisted_without_payload_store(self):
        """Output hash must be computed even when no payload store is configured."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)  # No payload_store
        recorder.begin_run(config={}, canonical_version="v1", run_id="run-1")
        recorder.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )
        op = recorder.begin_operation("run-1", "source-0", "source_load")

        recorder.complete_operation(op.operation_id, "completed", output_data={"count": 42}, duration_ms=100)

        fetched = recorder.get_operation(op.operation_id)
        assert fetched.output_data_hash == stable_hash({"count": 42})
        assert fetched.output_data_ref is None  # No payload store → no ref

    def test_raises_framework_bug_error_for_nonexistent_operation(self):
        _db, recorder, _state_id = _setup()

        with pytest.raises(FrameworkBugError):
            recorder.complete_operation("nonexistent-op-id", "completed")

    def test_raises_framework_bug_error_for_double_complete(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()
        recorder.complete_operation(op_id, "completed", duration_ms=10)

        with pytest.raises(FrameworkBugError):
            recorder.complete_operation(op_id, "completed", duration_ms=20)

    def test_no_orphaned_payload_on_double_complete(self, tmp_path):
        """Payload must not be stored when operation is already completed."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB.in_memory()
        store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=store)
        recorder.begin_run(config={}, canonical_version="v1", run_id="run-1")
        recorder.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )
        op = recorder.begin_operation("run-1", "source-0", "source_load")

        # First completion succeeds
        recorder.complete_operation(op.operation_id, "completed", duration_ms=10)

        # Count payload files before the duplicate attempt
        blobs_before = list(tmp_path.joinpath("payloads").rglob("*"))
        blobs_before = [p for p in blobs_before if p.is_file()]

        # Second completion with output_data must fail without storing a blob
        with pytest.raises(FrameworkBugError):
            recorder.complete_operation(op.operation_id, "completed", output_data={"leaked": True}, duration_ms=20)

        blobs_after = list(tmp_path.joinpath("payloads").rglob("*"))
        blobs_after = [p for p in blobs_after if p.is_file()]

        assert len(blobs_after) == len(blobs_before), (
            f"Orphaned payload blob created on duplicate completion: {len(blobs_after) - len(blobs_before)} new blob(s)"
        )

    def test_no_orphaned_payload_on_nonexistent_operation(self, tmp_path):
        """Payload must not be stored when operation_id is invalid."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB.in_memory()
        store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=store)
        recorder.begin_run(config={}, canonical_version="v1", run_id="run-1")

        with pytest.raises(FrameworkBugError):
            recorder.complete_operation("nonexistent-op", "completed", output_data={"leaked": True})

        blobs = list(tmp_path.joinpath("payloads").rglob("*"))
        blobs = [p for p in blobs if p.is_file()]
        assert len(blobs) == 0, f"Orphaned payload blob created for nonexistent operation: {len(blobs)} blob(s)"

    def test_output_data_ref_set_with_payload_store(self, tmp_path):
        """When payload store is configured, output_data_ref must be set on success."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB.in_memory()
        store = FilesystemPayloadStore(tmp_path / "payloads")
        recorder = LandscapeRecorder(db, payload_store=store)
        recorder.begin_run(config={}, canonical_version="v1", run_id="run-1")
        recorder.register_node(
            run_id="run-1",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=SchemaConfig.from_dict({"mode": "observed"}),
        )
        op = recorder.begin_operation("run-1", "source-0", "source_load")

        recorder.complete_operation(op.operation_id, "completed", output_data={"rows_loaded": 42}, duration_ms=100)

        completed = recorder.get_operation(op.operation_id)
        assert completed.status == "completed"
        assert completed.output_data_ref is not None, "output_data_ref should be set when payload store is configured"
        assert completed.output_data_hash == stable_hash({"rows_loaded": 42}), "output_data_hash should be set alongside ref"


class TestAllocateOperationCallIndex:
    """Tests for thread-safe operation call index allocation."""

    def test_sequential_allocation_starts_at_zero(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()

        idx0 = recorder.allocate_operation_call_index(op_id)
        idx1 = recorder.allocate_operation_call_index(op_id)
        idx2 = recorder.allocate_operation_call_index(op_id)

        assert idx0 == 0
        assert idx1 == 1
        assert idx2 == 2

    def test_independent_per_operation_id(self):
        _db, recorder, _state_id = _setup()
        op_a = recorder.begin_operation("run-1", "source-0", "source_load")
        op_b = recorder.begin_operation("run-1", "source-0", "source_load")

        assert recorder.allocate_operation_call_index(op_a.operation_id) == 0
        assert recorder.allocate_operation_call_index(op_b.operation_id) == 0
        assert recorder.allocate_operation_call_index(op_a.operation_id) == 1
        assert recorder.allocate_operation_call_index(op_b.operation_id) == 1


class TestRecordOperationCall:
    """Tests for recording calls linked to operations rather than node_states."""

    def test_creates_call_linked_to_operation(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()
        recorder.allocate_operation_call_index(op_id)

        call = recorder.record_operation_call(
            op_id,
            CallType.HTTP,
            CallStatus.SUCCESS,
            request_data={"url": "https://api.example.com/data"},
            response_data={"rows": 50},
            latency_ms=200,
        )

        assert call.call_id is not None
        assert call.call_type == CallType.HTTP
        assert call.status == CallStatus.SUCCESS
        assert call.operation_id == op_id
        assert call.state_id is None
        assert call.request_hash is not None
        assert call.latency_ms == 200

    def test_error_operation_call(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()

        call = recorder.record_operation_call(
            op_id,
            CallType.SQL,
            CallStatus.ERROR,
            request_data={"query": "SELECT * FROM missing"},
            error={"code": "table_not_found", "message": "Table does not exist"},
            latency_ms=3,
        )

        assert call.status == CallStatus.ERROR
        assert call.error_json is not None
        assert "table_not_found" in call.error_json

    def test_operation_call_with_provider(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()

        call = recorder.record_operation_call(
            op_id,
            CallType.LLM,
            CallStatus.SUCCESS,
            request_data={"prompt": "classify"},
            response_data={"label": "A"},
            provider="azure-openai",
        )

        assert call.call_id is not None
        assert call.call_type == CallType.LLM

    def test_operation_call_with_refs(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()

        call = recorder.record_operation_call(
            op_id,
            CallType.FILESYSTEM,
            CallStatus.SUCCESS,
            request_data={"path": "/data/file.csv"},
            request_ref="req-ref-001",
            response_ref="resp-ref-001",
        )

        assert call.request_ref == "req-ref-001"
        assert call.response_ref == "resp-ref-001"

    def test_multiple_operation_calls(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()

        calls = []
        for i in range(3):
            call = recorder.record_operation_call(
                op_id,
                CallType.HTTP,
                CallStatus.SUCCESS,
                request_data={"url": f"https://example.com/{i}"},
            )
            calls.append(call)

        assert len(calls) == 3
        assert len({c.call_id for c in calls}) == 3


class TestGetOperation:
    """Tests for retrieving operations by ID."""

    def test_roundtrip(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()

        op = recorder.get_operation(op_id)

        assert op is not None
        assert op.operation_id == op_id
        assert op.run_id == "run-1"
        assert op.node_id == "source-0"
        assert op.operation_type == "source_load"
        assert op.status == "open"

    def test_returns_none_for_unknown_id(self):
        _db, recorder, _state_id = _setup()

        op = recorder.get_operation("nonexistent-op-id")

        assert op is None

    def test_reflects_completion(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()
        recorder.complete_operation(op_id, "completed", duration_ms=99)

        op = recorder.get_operation(op_id)

        assert op.status == "completed"
        assert op.duration_ms == 99
        assert op.completed_at is not None


class TestGetOperationCalls:
    """Tests for retrieving calls associated with an operation."""

    def test_returns_calls_ordered_by_call_index(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()
        for i in range(3):
            recorder.record_operation_call(
                op_id,
                CallType.HTTP,
                CallStatus.SUCCESS,
                request_data={"index": i},
            )

        calls = recorder.get_operation_calls(op_id)

        assert len(calls) == 3
        indices = [c.call_index for c in calls]
        assert indices == sorted(indices)

    def test_empty_for_operation_with_no_calls(self):
        _db, recorder, _state_id, op_id = _setup_with_operation()

        calls = recorder.get_operation_calls(op_id)

        assert calls == []

    def test_does_not_include_state_linked_calls(self):
        _db, recorder, state_id, op_id = _setup_with_operation()
        idx = recorder.allocate_call_index(state_id)
        recorder.record_call(
            state_id,
            idx,
            CallType.LLM,
            CallStatus.SUCCESS,
            request_data={"prompt": "state-call"},
        )
        recorder.record_operation_call(
            op_id,
            CallType.HTTP,
            CallStatus.SUCCESS,
            request_data={"url": "https://example.com"},
        )

        op_calls = recorder.get_operation_calls(op_id)

        assert len(op_calls) == 1
        assert op_calls[0].call_type == CallType.HTTP


class TestGetOperationsForRun:
    """Tests for retrieving all operations belonging to a run."""

    def test_returns_all_operations_for_run(self):
        _db, recorder, _state_id = _setup()
        recorder.begin_operation("run-1", "source-0", "source_load")
        recorder.begin_operation("run-1", "source-0", "source_load")

        ops = recorder.get_operations_for_run("run-1")

        assert len(ops) == 2
        assert all(o.run_id == "run-1" for o in ops)

    def test_empty_for_run_with_no_operations(self):
        _db, recorder, _state_id = _setup()

        ops = recorder.get_operations_for_run("run-1")

        assert ops == []

    def test_does_not_include_operations_from_other_runs(self):
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)
        recorder.begin_run(config={}, canonical_version="v1", run_id="run-a")
        recorder.register_node(
            run_id="run-a",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        recorder.begin_run(config={}, canonical_version="v1", run_id="run-b")
        recorder.register_node(
            run_id="run-b",
            plugin_name="csv",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source-0",
            schema_config=_DYNAMIC_SCHEMA,
        )
        recorder.begin_operation("run-a", "source-0", "source_load")
        recorder.begin_operation("run-b", "source-0", "source_load")

        ops_a = recorder.get_operations_for_run("run-a")
        ops_b = recorder.get_operations_for_run("run-b")

        assert len(ops_a) == 1
        assert ops_a[0].run_id == "run-a"
        assert len(ops_b) == 1
        assert ops_b[0].run_id == "run-b"


class TestGetAllOperationCallsForRun:
    """Tests for batch retrieval of all operation-parented calls in a run."""

    def test_returns_all_operation_calls(self):
        _db, recorder, _state_id = _setup()
        op1 = recorder.begin_operation("run-1", "source-0", "source_load")
        op2 = recorder.begin_operation("run-1", "source-0", "source_load")
        recorder.record_operation_call(
            op1.operation_id,
            CallType.HTTP,
            CallStatus.SUCCESS,
            request_data={"url": "a"},
        )
        recorder.record_operation_call(
            op2.operation_id,
            CallType.SQL,
            CallStatus.SUCCESS,
            request_data={"query": "b"},
        )

        all_calls = recorder.get_all_operation_calls_for_run("run-1")

        assert len(all_calls) == 2
        call_types = {c.call_type for c in all_calls}
        assert CallType.HTTP in call_types
        assert CallType.SQL in call_types

    def test_empty_when_no_operation_calls(self):
        _db, recorder, _state_id = _setup()

        all_calls = recorder.get_all_operation_calls_for_run("run-1")

        assert all_calls == []

    def test_does_not_include_state_linked_calls(self):
        _db, recorder, state_id = _setup()
        op = recorder.begin_operation("run-1", "source-0", "source_load")
        recorder.record_operation_call(
            op.operation_id,
            CallType.HTTP,
            CallStatus.SUCCESS,
            request_data={"url": "op-call"},
        )
        idx = recorder.allocate_call_index(state_id)
        recorder.record_call(
            state_id,
            idx,
            CallType.LLM,
            CallStatus.SUCCESS,
            request_data={"prompt": "state-call"},
        )

        all_calls = recorder.get_all_operation_calls_for_run("run-1")

        assert len(all_calls) == 1
        assert all_calls[0].call_type == CallType.HTTP


class TestFindCallByRequestHash:
    """Tests for finding calls by their request hash within a run."""

    def test_finds_call_by_hash(self):
        _db, recorder, state_id = _setup()
        idx = recorder.allocate_call_index(state_id)
        original = recorder.record_call(
            state_id,
            idx,
            CallType.LLM,
            CallStatus.SUCCESS,
            request_data={"prompt": "unique-request"},
            response_data={"text": "response"},
        )

        found = recorder.find_call_by_request_hash("run-1", CallType.LLM, original.request_hash)

        assert found is not None
        assert found.call_id == original.call_id

    def test_returns_none_for_unknown_hash(self):
        _db, recorder, _state_id = _setup()

        found = recorder.find_call_by_request_hash("run-1", CallType.LLM, "nonexistent-hash")

        assert found is None

    def test_returns_none_for_wrong_call_type(self):
        _db, recorder, state_id = _setup()
        idx = recorder.allocate_call_index(state_id)
        original = recorder.record_call(
            state_id,
            idx,
            CallType.LLM,
            CallStatus.SUCCESS,
            request_data={"prompt": "typed-request"},
        )

        found = recorder.find_call_by_request_hash("run-1", CallType.HTTP, original.request_hash)

        assert found is None

    def test_sequence_index_for_duplicate_hashes(self):
        _db, recorder, state_id = _setup()
        same_request = {"prompt": "identical"}
        calls = []
        for _ in range(3):
            idx = recorder.allocate_call_index(state_id)
            call = recorder.record_call(
                state_id,
                idx,
                CallType.LLM,
                CallStatus.SUCCESS,
                request_data=same_request,
            )
            calls.append(call)

        found_0 = recorder.find_call_by_request_hash("run-1", CallType.LLM, calls[0].request_hash, sequence_index=0)
        found_1 = recorder.find_call_by_request_hash("run-1", CallType.LLM, calls[0].request_hash, sequence_index=1)

        assert found_0 is not None
        assert found_1 is not None
        assert found_0.call_id != found_1.call_id


class TestGetCallResponseData:
    """Tests for retrieving response data from the payload store."""

    def test_returns_none_without_payload_store(self):
        _db, recorder, state_id = _setup()
        idx = recorder.allocate_call_index(state_id)
        call = recorder.record_call(
            state_id,
            idx,
            CallType.LLM,
            CallStatus.SUCCESS,
            request_data={"prompt": "hello"},
            response_data={"text": "world"},
        )

        result = recorder.get_call_response_data(call.call_id)

        assert result is None

    def test_returns_none_for_call_without_response(self):
        _db, recorder, state_id = _setup()
        idx = recorder.allocate_call_index(state_id)
        call = recorder.record_call(
            state_id,
            idx,
            CallType.LLM,
            CallStatus.ERROR,
            request_data={"prompt": "fail"},
            error={"code": "error"},
        )

        result = recorder.get_call_response_data(call.call_id)

        assert result is None
