# tests/core/landscape/test_recorder_calls.py
"""Tests for external call recording API.

Tests LandscapeRecorder.record_call() and get_calls() methods.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from sqlalchemy.exc import IntegrityError

from elspeth.contracts import CallStatus, CallType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.payload_store import FilesystemPayloadStore


class TestRecordCall:
    """Tests for LandscapeRecorder.record_call()."""

    @pytest.fixture
    def recorder(self) -> LandscapeRecorder:
        """Create recorder with in-memory DB."""
        db = LandscapeDB.in_memory()
        return LandscapeRecorder(db)

    @pytest.fixture
    def state_id(self, recorder: LandscapeRecorder) -> str:
        """Create a node state to attach calls to."""
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"input": "test"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={"input": "test"},
        )
        return state.state_id

    def test_record_successful_llm_call(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test recording a successful LLM call."""
        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"model": "gpt-4", "prompt": "Hello"},
            response_data={"completion": "Hi there!"},
            latency_ms=150.5,
        )

        assert call.call_id is not None
        assert call.state_id == state_id
        assert call.call_index == 0
        assert call.call_type == CallType.LLM
        assert call.status == CallStatus.SUCCESS
        assert call.request_hash is not None
        assert call.response_hash is not None
        assert call.latency_ms == 150.5
        assert call.error_json is None

    def test_record_failed_call_with_error(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test recording a failed call with error details."""
        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.ERROR,
            request_data={"url": "https://api.example.com"},
            error={"code": 500, "message": "Internal Server Error"},
            latency_ms=50.0,
        )

        assert call.status == CallStatus.ERROR
        assert call.response_hash is None
        assert call.error_json is not None
        assert "500" in call.error_json

    def test_multiple_calls_same_state(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test recording multiple calls for the same state."""
        call1 = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "First"},
            response_data={"response": "First response"},
        )
        call2 = recorder.record_call(
            state_id=state_id,
            call_index=1,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "Second"},
            response_data={"response": "Second response"},
        )

        assert call1.call_index == 0
        assert call2.call_index == 1

        # Verify via get_calls
        calls = recorder.get_calls(state_id)
        assert len(calls) == 2
        assert calls[0].call_index == 0
        assert calls[1].call_index == 1

    def test_call_with_payload_refs(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test recording calls with payload store references."""
        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "Large prompt..."},
            response_data={"response": "Large response..."},
            request_ref="sha256:abc123...",
            response_ref="sha256:def456...",
        )

        assert call.request_ref == "sha256:abc123..."
        assert call.response_ref == "sha256:def456..."

    def test_duplicate_call_index_raises_integrity_error(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test that duplicate (state_id, call_index) raises IntegrityError."""
        # First call succeeds
        recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "First"},
            response_data={"response": "First"},
        )

        # Duplicate call_index fails
        with pytest.raises(IntegrityError):
            recorder.record_call(
                state_id=state_id,
                call_index=0,  # Same index!
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data={"prompt": "Second"},
                response_data={"response": "Second"},
            )

    def test_invalid_state_id_raises_integrity_error(self, recorder: LandscapeRecorder) -> None:
        """Test that invalid state_id raises IntegrityError (FK constraint)."""
        with pytest.raises(IntegrityError):
            recorder.record_call(
                state_id="nonexistent_state_id",
                call_index=0,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data={"prompt": "Test"},
                response_data={"response": "Test"},
            )

    def test_record_http_call(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test recording an HTTP call type."""
        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.SUCCESS,
            request_data={"method": "GET", "url": "https://api.example.com/data"},
            response_data={"status": 200, "body": {"data": "result"}},
            latency_ms=250.0,
        )

        assert call.call_type == CallType.HTTP
        assert call.latency_ms == 250.0

    def test_record_sql_call(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test recording a SQL call type."""
        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.SQL,
            status=CallStatus.SUCCESS,
            request_data={"query": "SELECT * FROM users WHERE id = ?", "params": [42]},
            response_data={"rows": [{"id": 42, "name": "Alice"}]},
            latency_ms=15.0,
        )

        assert call.call_type == CallType.SQL

    def test_record_filesystem_call(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test recording a filesystem call type."""
        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.FILESYSTEM,
            status=CallStatus.SUCCESS,
            request_data={"operation": "read", "path": "/data/config.json"},
            response_data={"content": '{"key": "value"}'},
            latency_ms=5.0,
        )

        assert call.call_type == CallType.FILESYSTEM

    def test_call_without_latency(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test recording a call without latency information."""
        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "Test"},
            response_data={"response": "Test response"},
            # No latency_ms provided
        )

        assert call.latency_ms is None

    def test_error_call_without_response(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test recording an error call with no response data."""
        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.ERROR,
            request_data={"url": "https://api.example.com/timeout"},
            # No response_data - the call timed out
            error={"type": "timeout", "message": "Request timed out after 30s"},
            latency_ms=30000.0,
        )

        assert call.status == CallStatus.ERROR
        assert call.response_hash is None
        assert call.error_json is not None
        assert "timeout" in call.error_json

    def test_created_at_is_set(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test that created_at timestamp is automatically set."""
        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "Test"},
            response_data={"response": "Test"},
        )

        assert call.created_at is not None

    def test_request_hash_is_deterministic(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test that the same request data produces the same hash."""
        request_data = {"model": "gpt-4", "prompt": "Hello, world!"}

        call1 = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data=request_data,
            response_data={"response": "Hi!"},
        )

        # Create another state for second call
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"input": "test"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state2 = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={"input": "test"},
        )

        call2 = recorder.record_call(
            state_id=state2.state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data=request_data,  # Same request data
            response_data={"response": "Hello!"},  # Different response
        )

        # Same request should produce same hash
        assert call1.request_hash == call2.request_hash
        # Different response should produce different hash
        assert call1.response_hash != call2.response_hash


class TestCallPayloadPersistence:
    """Tests for auto-persist behavior with payload store.

    When a payload store is configured, record_call should automatically
    persist request/response data and populate the refs. This enables
    replay/verify modes to retrieve the original payloads.
    """

    def _create_state(self, recorder: LandscapeRecorder) -> str:
        """Helper to create a node state for attaching calls."""
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_transform",
            node_type="transform",
            plugin_version="1.0",
            config={},
            schema_config=schema,
        )
        row = recorder.create_row(
            run_id=run.run_id,
            source_node_id=node.node_id,
            row_index=0,
            data={"input": "test"},
        )
        token = recorder.create_token(row_id=row.row_id)
        state = recorder.begin_node_state(
            token_id=token.token_id,
            node_id=node.node_id,
            step_index=0,
            input_data={"input": "test"},
        )
        return state.state_id

    def test_auto_persist_response_when_payload_store_configured(self) -> None:
        """When payload store exists, response_data is auto-persisted and ref populated."""
        with TemporaryDirectory() as tmpdir:
            payload_store = FilesystemPayloadStore(Path(tmpdir) / "payloads")
            db = LandscapeDB.in_memory()
            recorder = LandscapeRecorder(db, payload_store=payload_store)
            state_id = self._create_state(recorder)

            response_data = {"content": "Hello!", "model": "gpt-4"}
            call = recorder.record_call(
                state_id=state_id,
                call_index=0,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data={"prompt": "Hi"},
                response_data=response_data,
            )

            # response_ref should be automatically populated
            assert call.response_ref is not None

            # The response should be retrievable
            retrieved = recorder.get_call_response_data(call.call_id)
            assert retrieved == response_data

    def test_auto_persist_request_when_payload_store_configured(self) -> None:
        """When payload store exists, request_data is auto-persisted and ref populated."""
        with TemporaryDirectory() as tmpdir:
            payload_store = FilesystemPayloadStore(Path(tmpdir) / "payloads")
            db = LandscapeDB.in_memory()
            recorder = LandscapeRecorder(db, payload_store=payload_store)
            state_id = self._create_state(recorder)

            request_data = {"model": "gpt-4", "prompt": "Hello, world!"}
            call = recorder.record_call(
                state_id=state_id,
                call_index=0,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data=request_data,
                response_data={"content": "Hi!"},
            )

            # request_ref should be automatically populated
            assert call.request_ref is not None

    def test_no_auto_persist_without_payload_store(self) -> None:
        """Without payload store, refs remain None."""
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)  # No payload store
        state_id = self._create_state(recorder)

        call = recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "Hi"},
            response_data={"content": "Hello!"},
        )

        # Without payload store, refs should remain None
        assert call.request_ref is None
        assert call.response_ref is None

        # get_call_response_data should return None
        retrieved = recorder.get_call_response_data(call.call_id)
        assert retrieved is None

    def test_explicit_ref_not_overwritten(self) -> None:
        """If caller provides explicit ref, it should not be overwritten."""
        with TemporaryDirectory() as tmpdir:
            payload_store = FilesystemPayloadStore(Path(tmpdir) / "payloads")
            db = LandscapeDB.in_memory()
            recorder = LandscapeRecorder(db, payload_store=payload_store)
            state_id = self._create_state(recorder)

            explicit_ref = "explicit-reference-123"
            call = recorder.record_call(
                state_id=state_id,
                call_index=0,
                call_type=CallType.LLM,
                status=CallStatus.SUCCESS,
                request_data={"prompt": "Hi"},
                response_data={"content": "Hello!"},
                response_ref=explicit_ref,  # Caller provides explicit ref
            )

            # Should use caller's explicit ref, not auto-generate
            assert call.response_ref == explicit_ref

    def test_error_call_without_response_no_ref(self) -> None:
        """Error calls without response_data should not have response_ref."""
        with TemporaryDirectory() as tmpdir:
            payload_store = FilesystemPayloadStore(Path(tmpdir) / "payloads")
            db = LandscapeDB.in_memory()
            recorder = LandscapeRecorder(db, payload_store=payload_store)
            state_id = self._create_state(recorder)

            call = recorder.record_call(
                state_id=state_id,
                call_index=0,
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data={"url": "https://api.example.com"},
                # No response_data - request failed before response
                error={"type": "ConnectionError", "message": "Connection refused"},
            )

            # request_ref should be populated (we have request_data)
            assert call.request_ref is not None
            # response_ref should be None (no response_data)
            assert call.response_ref is None
