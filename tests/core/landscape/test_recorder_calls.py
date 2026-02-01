# tests/core/landscape/test_recorder_calls.py
"""Tests for external call recording API.

Tests LandscapeRecorder.record_call() and get_calls() methods.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from sqlalchemy.exc import IntegrityError

from elspeth.contracts import CallStatus, CallType, NodeType
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
            node_type=NodeType.TRANSFORM,
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
            run_id=run.run_id,
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

    def test_persisted_call_fields_match_expected_values(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Verify all persisted call fields match expected values.

        P1 audit trail verification: Tests must not just check non-null,
        but verify actual values match expected hashes and enums.
        """
        from elspeth.core.canonical import stable_hash

        request_data = {"model": "gpt-4", "prompt": "Hello"}
        response_data = {"completion": "Hi there!"}

        recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data=request_data,
            response_data=response_data,
            latency_ms=150.5,
        )

        # Retrieve persisted call via get_calls
        calls = recorder.get_calls(state_id)
        persisted = calls[0]

        # Verify enums are actual enum types (not strings)
        assert isinstance(persisted.call_type, CallType), f"call_type should be CallType enum, got {type(persisted.call_type)}"
        assert isinstance(persisted.status, CallStatus), f"status should be CallStatus enum, got {type(persisted.status)}"
        assert persisted.call_type == CallType.LLM
        assert persisted.status == CallStatus.SUCCESS

        # Verify hashes match expected stable_hash values
        expected_request_hash = stable_hash(request_data)
        expected_response_hash = stable_hash(response_data)
        assert persisted.request_hash == expected_request_hash, (
            f"request_hash mismatch: expected {expected_request_hash}, got {persisted.request_hash}"
        )
        assert persisted.response_hash == expected_response_hash, (
            f"response_hash mismatch: expected {expected_response_hash}, got {persisted.response_hash}"
        )

        # Verify other fields
        assert persisted.call_index == 0
        assert persisted.latency_ms == 150.5
        assert persisted.error_json is None

    def test_persisted_error_call_fields(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Verify error call fields are correctly persisted."""
        from elspeth.core.canonical import canonical_json, stable_hash

        request_data = {"url": "https://api.example.com"}
        error_data = {"code": 500, "message": "Internal Server Error"}

        recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.HTTP,
            status=CallStatus.ERROR,
            request_data=request_data,
            error=error_data,
            latency_ms=50.0,
        )

        # Retrieve persisted call
        calls = recorder.get_calls(state_id)
        persisted = calls[0]

        # Verify enum types and values
        assert isinstance(persisted.call_type, CallType)
        assert isinstance(persisted.status, CallStatus)
        assert persisted.call_type == CallType.HTTP
        assert persisted.status == CallStatus.ERROR

        # Verify request hash
        expected_request_hash = stable_hash(request_data)
        assert persisted.request_hash == expected_request_hash

        # Verify error_json matches canonical serialization
        expected_error_json = canonical_json(error_data)
        assert persisted.error_json == expected_error_json

        # Verify no response hash for error calls
        assert persisted.response_hash is None

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

    def test_duplicate_call_index_rejected_at_db_level(self, recorder: LandscapeRecorder, state_id: str) -> None:
        """Test that duplicate (state_id, call_index) is rejected at DB level.

        The schema has a partial unique index: UNIQUE(state_id, call_index) WHERE state_id IS NOT NULL.
        This enforces call ordering uniqueness for audit integrity - call_index must be
        unambiguous for replay/verification.

        Callers should use allocate_call_index() to get unique indices, but the DB
        constraint serves as defense-in-depth against bugs that might bypass the allocator.
        """
        # First call succeeds
        recorder.record_call(
            state_id=state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"prompt": "First"},
            response_data={"response": "First"},
        )

        # Duplicate call_index is rejected by DB constraint
        with pytest.raises(IntegrityError):
            recorder.record_call(
                state_id=state_id,
                call_index=0,  # Same index - rejected at DB level
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
            node_type=NodeType.TRANSFORM,
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
            run_id=run.run_id,
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
            node_type=NodeType.TRANSFORM,
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
            run_id=run.run_id,
            step_index=0,
            input_data={"input": "test"},
        )
        return state.state_id

    def test_auto_persist_response_when_payload_store_configured(self, payload_store) -> None:
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

    def test_auto_persist_request_when_payload_store_configured(self, payload_store) -> None:
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

    def test_explicit_ref_not_overwritten(self, payload_store) -> None:
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

    def test_error_call_without_response_no_ref(self, payload_store) -> None:
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


class TestFindCallByRequestHashRunIsolation:
    """Tests for cross-run isolation in find_call_by_request_hash.

    These tests verify that find_call_by_request_hash returns calls from the
    correct run when the same node_id is reused across multiple runs. This is
    critical for replay mode correctness.

    Background: The nodes table has composite PK (node_id, run_id). When the
    same pipeline runs twice, node_ids are reused. Queries must filter by
    run_id to avoid returning calls from the wrong run.
    """

    def _create_run_with_call(
        self,
        recorder: LandscapeRecorder,
        node_id: str,
        request_data: dict[str, str],
        response_data: dict[str, str],
    ) -> tuple[str, str]:
        """Create a run with a transform node and an LLM call.

        Args:
            recorder: LandscapeRecorder instance
            node_id: Node ID to use (for testing reuse across runs)
            request_data: Request data for the call
            response_data: Response data for the call

        Returns:
            Tuple of (run_id, call_id)
        """
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run = recorder.begin_run(config={}, canonical_version="v1")

        # Register node with specified node_id
        node = recorder.register_node(
            run_id=run.run_id,
            plugin_name="llm_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=schema,
            node_id=node_id,  # Explicit node_id for collision testing
        )

        # Create row, token, and state
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
            run_id=run.run_id,
            step_index=0,
            input_data={"input": "test"},
        )

        # Record the call
        call = recorder.record_call(
            state_id=state.state_id,
            call_index=0,
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data=request_data,
            response_data=response_data,
        )

        return run.run_id, call.call_id

    def test_find_call_returns_call_from_correct_run_with_same_node_ids(self) -> None:
        """find_call_by_request_hash returns call from requested run only.

        Scenario:
        - Run A: node_id="llm_1", call with request_hash=H
        - Run B: node_id="llm_1" (SAME), call with request_hash=H (SAME)
        - Query for Run B should return Run B's call, NOT Run A's

        This tests the critical cross-run isolation requirement.
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        # Identical request data = same request_hash
        request_data = {"model": "gpt-4", "prompt": "Hello"}
        node_id = "shared_llm_transform_1"

        # Create Run A with call
        _run_a_id, call_a_id = self._create_run_with_call(
            recorder,
            node_id=node_id,
            request_data=request_data,
            response_data={"content": "Response from Run A"},
        )

        # Create Run B with call (same node_id, same request_hash)
        run_b_id, call_b_id = self._create_run_with_call(
            recorder,
            node_id=node_id,
            request_data=request_data,
            response_data={"content": "Response from Run B"},
        )

        # Compute request hash for lookup
        from elspeth.core.canonical import stable_hash

        request_hash = stable_hash(request_data)

        # Query for Run B's call
        result = recorder.find_call_by_request_hash(
            run_id=run_b_id,
            call_type="llm",
            request_hash=request_hash,
        )

        # CRITICAL: Must return Run B's call, not Run A's
        assert result is not None, "Should find call in Run B"
        assert result.call_id == call_b_id, (
            f"Should return Run B's call ({call_b_id}), but got {result.call_id} (Run A's call is {call_a_id})"
        )

    def test_find_call_returns_none_when_only_other_run_has_call(self) -> None:
        """find_call_by_request_hash returns None if call only exists in other run.

        Scenario:
        - Run A: node_id="llm_1", call with request_hash=H
        - Run B: node_id="llm_1" (SAME), NO calls
        - Query for Run B should return None, not Run A's call

        This ensures run isolation is complete (not just filtering).
        """
        db = LandscapeDB.in_memory()
        recorder = LandscapeRecorder(db)

        request_data = {"model": "gpt-4", "prompt": "Hello"}
        node_id = "shared_llm_transform_1"

        # Create Run A with call
        _run_a_id, _ = self._create_run_with_call(
            recorder,
            node_id=node_id,
            request_data=request_data,
            response_data={"content": "Response from Run A"},
        )

        # Create Run B WITHOUT any calls (just the node)
        schema = SchemaConfig.from_dict({"fields": "dynamic"})
        run_b = recorder.begin_run(config={}, canonical_version="v1")
        recorder.register_node(
            run_id=run_b.run_id,
            plugin_name="llm_transform",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            schema_config=schema,
            node_id=node_id,
        )

        # Compute request hash for lookup
        from elspeth.core.canonical import stable_hash

        request_hash = stable_hash(request_data)

        # Query for Run B's call (should not find Run A's)
        result = recorder.find_call_by_request_hash(
            run_id=run_b.run_id,
            call_type="llm",
            request_hash=request_hash,
        )

        # CRITICAL: Must return None, not Run A's call
        assert result is None, f"Should return None for Run B (no calls), but got call_id={result.call_id if result else 'N/A'}"
