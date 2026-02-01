# tests/plugins/test_context.py
"""Tests for plugin context."""

from contextlib import nullcontext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class TestPluginContext:
    """Context passed to all plugin operations."""

    def test_minimal_context(self) -> None:
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        assert ctx.run_id == "run-001"
        assert ctx.config == {}

    def test_optional_integrations_default_none(self) -> None:
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        # Phase 3 integration points - optional in Phase 2
        assert ctx.landscape is None
        assert ctx.tracer is None
        assert ctx.payload_store is None

    def test_start_span_without_tracer(self) -> None:
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        # Should return nullcontext when no tracer
        span_ctx = ctx.start_span("test_operation")
        assert isinstance(span_ctx, nullcontext)

    def test_get_config_value(self) -> None:
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(
            run_id="run-001",
            config={"threshold": 0.5, "nested": {"key": "value"}},
        )
        assert ctx.get("threshold") == 0.5
        assert ctx.get("nested.key") == "value"
        assert ctx.get("missing", default="default") == "default"


class TestCheckpointAPI:
    """Tests for checkpoint API used by batch transforms."""

    def test_checkpoint_methods_exist(self) -> None:
        """PluginContext has checkpoint methods."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        assert hasattr(ctx, "get_checkpoint")
        assert hasattr(ctx, "update_checkpoint")
        assert hasattr(ctx, "clear_checkpoint")

    def test_get_checkpoint_returns_none_when_empty(self) -> None:
        """Empty checkpoint returns None (not empty dict)."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        assert ctx.get_checkpoint() is None

    def test_update_checkpoint_stores_data(self) -> None:
        """update_checkpoint stores data accessible via get_checkpoint."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        ctx.update_checkpoint({"batch_id": "batch-123", "row_count": 5})

        checkpoint = ctx.get_checkpoint()
        assert checkpoint is not None
        assert checkpoint["batch_id"] == "batch-123"
        assert checkpoint["row_count"] == 5

    def test_update_checkpoint_merges_data(self) -> None:
        """Multiple update_checkpoint calls merge data."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        ctx.update_checkpoint({"batch_id": "batch-123"})
        ctx.update_checkpoint({"status": "submitted"})

        checkpoint = ctx.get_checkpoint()
        assert checkpoint is not None
        assert checkpoint["batch_id"] == "batch-123"
        assert checkpoint["status"] == "submitted"

    def test_clear_checkpoint_removes_all_data(self) -> None:
        """clear_checkpoint removes all checkpoint data."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        ctx.update_checkpoint({"batch_id": "batch-123"})
        assert ctx.get_checkpoint() is not None

        ctx.clear_checkpoint()
        assert ctx.get_checkpoint() is None

    def test_checkpoint_typical_batch_workflow(self) -> None:
        """Checkpoint API supports typical batch transform workflow."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})

        # Phase 1: Submit - no existing checkpoint
        assert ctx.get_checkpoint() is None

        # Save checkpoint after batch submission
        ctx.update_checkpoint(
            {
                "batch_id": "batch-xyz789",
                "input_file_id": "file-abc123",
                "row_mapping": {"row-0": 0, "row-1": 1},
                "submitted_at": "2024-01-01T00:00:00Z",
            }
        )

        # Phase 2: Resume - checkpoint exists
        checkpoint = ctx.get_checkpoint()
        assert checkpoint is not None
        assert checkpoint["batch_id"] == "batch-xyz789"

        # After completion, clear checkpoint
        ctx.clear_checkpoint()
        assert ctx.get_checkpoint() is None


class TestValidationErrorRecording:
    """Tests for recording validation errors from sources."""

    def test_record_validation_error_exists(self) -> None:
        """PluginContext has record_validation_error method."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={})
        assert hasattr(ctx, "record_validation_error")
        assert callable(ctx.record_validation_error)

    def test_record_validation_error_returns_quarantine_token(self) -> None:
        """record_validation_error returns token for tracking quarantined row."""
        from elspeth.plugins.context import PluginContext, ValidationErrorToken

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        token = ctx.record_validation_error(
            row={"id": 42, "invalid": "data"},
            error="validation failed",
            schema_mode="strict",
            destination="discard",
        )

        assert token is not None
        assert isinstance(token, ValidationErrorToken)
        assert token.row_id is not None
        assert token.node_id == "source_node"

    def test_record_validation_error_without_landscape_logs_warning(self, caplog: "pytest.LogCaptureFixture") -> None:
        """record_validation_error logs warning when no landscape configured."""
        import logging

        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        with caplog.at_level(logging.WARNING):
            token = ctx.record_validation_error(
                row={"id": 42, "invalid": "data"},
                error="validation failed",
                schema_mode="strict",
                destination="discard",
            )

        # Should still return a token even without landscape
        assert token is not None
        assert token.error_id is None  # Not recorded to landscape
        # Check that warning was logged
        assert "no landscape" in caplog.text.lower()

    def test_record_validation_error_uses_id_field_as_row_id(self) -> None:
        """record_validation_error uses row's id field if present."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        token = ctx.record_validation_error(
            row={"id": "row-42", "invalid": "data"},
            error="validation failed",
            schema_mode="strict",
            destination="discard",
        )

        assert token.row_id == "row-42"

    def test_record_validation_error_generates_row_id_from_hash(self) -> None:
        """record_validation_error generates row_id from content hash if no id field."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        token = ctx.record_validation_error(
            row={"no_id_field": "data"},
            error="validation failed",
            schema_mode="strict",
            destination="discard",
        )

        # Should generate a row_id from hash (16 chars)
        assert token.row_id is not None
        assert len(token.row_id) == 16

    def test_record_validation_error_with_landscape(self) -> None:
        """record_validation_error records to landscape when configured."""
        from unittest.mock import MagicMock

        from elspeth.plugins.context import PluginContext

        # Create mock landscape recorder
        mock_landscape = MagicMock()
        mock_landscape.record_validation_error.return_value = "verr_abc123"

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=mock_landscape,
        )

        token = ctx.record_validation_error(
            row={"id": 42, "invalid": "data"},
            error="validation failed",
            schema_mode="strict",
            destination="quarantine_sink",
        )

        # Should have called landscape
        mock_landscape.record_validation_error.assert_called_once()
        call_kwargs = mock_landscape.record_validation_error.call_args[1]
        assert call_kwargs["run_id"] == "test-run"
        assert call_kwargs["node_id"] == "source_node"
        assert call_kwargs["row_data"] == {"id": 42, "invalid": "data"}
        assert call_kwargs["error"] == "validation failed"
        assert call_kwargs["schema_mode"] == "strict"
        assert call_kwargs["destination"] == "quarantine_sink"

        # Token should have error_id from landscape
        assert token.error_id == "verr_abc123"
        assert token.destination == "quarantine_sink"


class TestValidationErrorDestination:
    """Tests for validation error destination tracking."""

    def test_validation_error_token_has_destination(self) -> None:
        """ValidationErrorToken includes destination field."""
        from elspeth.plugins.context import ValidationErrorToken

        token = ValidationErrorToken(
            row_id="row_1",
            node_id="source_node",
            destination="quarantine_sink",
        )

        assert token.destination == "quarantine_sink"

    def test_validation_error_token_defaults_to_discard(self) -> None:
        """ValidationErrorToken defaults to 'discard' if not specified."""
        from elspeth.plugins.context import ValidationErrorToken

        token = ValidationErrorToken(row_id="row_1", node_id="source_node")

        assert token.destination == "discard"

    def test_record_validation_error_requires_destination(self) -> None:
        """record_validation_error requires destination parameter."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        # Should work with destination
        token = ctx.record_validation_error(
            row={"id": 1, "bad": "data"},
            error="validation failed",
            schema_mode="strict",
            destination="quarantine_sink",
        )

        assert token.destination == "quarantine_sink"

    def test_record_validation_error_with_discard_destination(self) -> None:
        """record_validation_error accepts 'discard' as destination."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        token = ctx.record_validation_error(
            row={"id": 1, "bad": "data"},
            error="validation failed",
            schema_mode="strict",
            destination="discard",
        )

        assert token.destination == "discard"


class TestRouteToSink:
    """Tests for route_to_sink method."""

    def test_route_to_sink_exists(self) -> None:
        """PluginContext has route_to_sink method."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="test_node")
        assert hasattr(ctx, "route_to_sink")
        assert callable(ctx.route_to_sink)

    def test_route_to_sink_logs_action(self, caplog: "pytest.LogCaptureFixture") -> None:
        """route_to_sink logs the routing action."""
        import logging

        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        with caplog.at_level(logging.INFO):
            ctx.route_to_sink(
                sink_name="quarantine",
                row={"id": 1, "bad": "data"},
                metadata={"reason": "validation failed"},
            )

        assert "route_to_sink" in caplog.text
        assert "quarantine" in caplog.text

    def test_route_to_sink_accepts_none_metadata(self) -> None:
        """route_to_sink works without metadata."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="test_node")

        # Should not raise
        ctx.route_to_sink(sink_name="error_sink", row={"id": 42})


class TestTransformErrorRecording:
    """Tests for transform error recording."""

    def test_transform_error_token_exists(self) -> None:
        """TransformErrorToken can be created."""
        from elspeth.plugins.context import TransformErrorToken

        token = TransformErrorToken(
            token_id="tok_123",
            transform_id="field_mapper",
            destination="error_sink",
        )

        assert token.token_id == "tok_123"
        assert token.transform_id == "field_mapper"
        assert token.destination == "error_sink"
        assert token.error_id is None

    def test_transform_error_token_defaults_to_discard(self) -> None:
        """TransformErrorToken defaults to 'discard' if not specified."""
        from elspeth.plugins.context import TransformErrorToken

        token = TransformErrorToken(
            token_id="tok_123",
            transform_id="field_mapper",
        )

        assert token.destination == "discard"

    def test_record_transform_error_without_landscape(self) -> None:
        """record_transform_error works without landscape (logs warning)."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="transform_node")

        token = ctx.record_transform_error(
            token_id="tok_123",
            transform_id="field_mapper",
            row={"id": 1, "data": "test"},
            error_details={"reason": "validation_failed", "error": "Cannot process"},
            destination="error_sink",
        )

        assert token.token_id == "tok_123"
        assert token.transform_id == "field_mapper"
        assert token.destination == "error_sink"
        assert token.error_id is None  # No landscape

    def test_record_transform_error_without_landscape_logs_warning(self, caplog: "pytest.LogCaptureFixture") -> None:
        """record_transform_error logs warning when no landscape configured."""
        import logging

        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="transform_node")

        with caplog.at_level(logging.WARNING):
            ctx.record_transform_error(
                token_id="tok_123",
                transform_id="field_mapper",
                row={"id": 1, "data": "test"},
                error_details={"reason": "validation_failed", "error": "Cannot process"},
                destination="error_sink",
            )

        # Check that warning was logged
        assert "no landscape" in caplog.text.lower()

    def test_record_transform_error_with_landscape(self) -> None:
        """record_transform_error stores in audit trail via mock."""
        from unittest.mock import MagicMock

        from elspeth.plugins.context import PluginContext

        # Create mock landscape recorder
        mock_landscape = MagicMock()
        mock_landscape.record_transform_error.return_value = "terr_abc123def4"

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="transform_node",
            landscape=mock_landscape,
        )

        token = ctx.record_transform_error(
            token_id="tok_456",
            transform_id="field_mapper",
            row={"id": 42, "value": "bad"},
            error_details={"reason": "validation_failed", "error": "Division by zero"},
            destination="failed_rows",
        )

        # Should have called landscape
        mock_landscape.record_transform_error.assert_called_once()
        call_kwargs = mock_landscape.record_transform_error.call_args[1]
        assert call_kwargs["run_id"] == "test-run"
        assert call_kwargs["token_id"] == "tok_456"
        assert call_kwargs["transform_id"] == "field_mapper"
        assert call_kwargs["row_data"] == {"id": 42, "value": "bad"}
        assert call_kwargs["error_details"] == {"reason": "validation_failed", "error": "Division by zero"}
        assert call_kwargs["destination"] == "failed_rows"

        # Token should have error_id from landscape
        assert token.error_id == "terr_abc123def4"
        assert token.destination == "failed_rows"


class TestTokenField:
    """Tests for token field used by BatchTransformMixin for row-level pipelining."""

    def test_token_field_defaults_to_none(self) -> None:
        """PluginContext.token defaults to None."""
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={})
        assert ctx.token is None

    def test_token_accepts_token_info(self) -> None:
        """PluginContext accepts TokenInfo via token parameter."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.plugins.context import PluginContext

        token = TokenInfo(row_id="row-1", token_id="token-row-1", row_data={"x": 1})
        ctx = PluginContext(run_id="test-run", config={}, token=token)

        assert ctx.token is not None
        assert ctx.token is token  # Same object reference
        assert ctx.token.row_id == "row-1"
        assert ctx.token.token_id == "token-row-1"
        assert ctx.token.row_data == {"x": 1}

    def test_token_identity_preserved_on_access(self) -> None:
        """Token identity is preserved - multiple accesses return same object."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.plugins.context import PluginContext

        token = TokenInfo(row_id="row-42", token_id="token-42", row_data={"value": 100})
        ctx = PluginContext(run_id="test-run", config={}, token=token)

        # Multiple accesses should return the exact same object
        access1 = ctx.token
        access2 = ctx.token
        assert access1 is access2
        assert access1 is token

    def test_token_can_be_mutated_after_construction(self) -> None:
        """Token field can be set after construction (engine sets it per-row)."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.plugins.context import PluginContext

        ctx = PluginContext(run_id="test-run", config={})
        assert ctx.token is None

        # Engine sets token before calling batch transforms
        token = TokenInfo(row_id="row-99", token_id="token-99", row_data={"data": "test"})
        ctx.token = token

        assert ctx.token is token
        assert ctx.token.row_id == "row-99"


class TestRecordCallTelemetryResponseHash:
    """Tests for response hash handling in record_call telemetry.

    Regression test for P3 issue: empty-but-valid responses should still get
    hashed for telemetry/audit correlation.
    """

    def test_empty_dict_response_gets_hashed(self) -> None:
        """Empty dict {} response should emit response_hash in telemetry.

        Bug: Using truthiness check (if response_data) causes empty responses
        to emit response_hash=None, breaking telemetry/audit correlation.
        """
        from unittest.mock import MagicMock

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.plugins.context import PluginContext

        # Set up telemetry callback to capture emitted events
        emitted_events: list = []

        def capture_telemetry(event):
            emitted_events.append(event)

        # Mock landscape to avoid DB setup - we only care about telemetry
        mock_landscape = MagicMock()
        mock_landscape.record_call.return_value = MagicMock(call_id="call-001")

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_landscape,
            state_id="state-001",  # Required for call recording
            telemetry_emit=capture_telemetry,
        )

        # Call with empty dict response (valid but falsy)
        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data={"endpoint": "/empty"},
            response_data={},  # Empty dict - valid response
            latency_ms=50.0,
            status=CallStatus.SUCCESS,
        )

        # Verify telemetry was emitted with response_hash
        assert len(emitted_events) == 1
        event = emitted_events[0]
        # response_hash should NOT be None for empty dict
        assert event.response_hash is not None, (
            "Empty dict response should still get hashed. Got response_hash=None which breaks telemetry/audit correlation."
        )

    def test_empty_list_response_gets_hashed(self) -> None:
        """Empty list [] response should emit response_hash in telemetry."""
        from unittest.mock import MagicMock

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.plugins.context import PluginContext

        emitted_events: list = []

        def capture_telemetry(event):
            emitted_events.append(event)

        mock_landscape = MagicMock()
        mock_landscape.record_call.return_value = MagicMock(call_id="call-001")

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_landscape,
            state_id="state-001",
            telemetry_emit=capture_telemetry,
        )

        # Empty list response
        ctx.record_call(
            call_type=CallType.SQL,
            provider="database",
            request_data={"query": "SELECT * FROM empty_table"},
            response_data=[],  # Empty list - no results
            latency_ms=10.0,
            status=CallStatus.SUCCESS,
        )

        assert len(emitted_events) == 1
        assert emitted_events[0].response_hash is not None

    def test_empty_string_response_gets_hashed(self) -> None:
        """Empty string '' response should emit response_hash in telemetry."""
        from unittest.mock import MagicMock

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.plugins.context import PluginContext

        emitted_events: list = []

        def capture_telemetry(event):
            emitted_events.append(event)

        mock_landscape = MagicMock()
        mock_landscape.record_call.return_value = MagicMock(call_id="call-001")

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_landscape,
            state_id="state-001",
            telemetry_emit=capture_telemetry,
        )

        # Empty string response (e.g., HTTP 204 No Content)
        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data={"method": "DELETE"},
            response_data="",  # Empty string - valid 204 response
            latency_ms=25.0,
            status=CallStatus.SUCCESS,
        )

        assert len(emitted_events) == 1
        assert emitted_events[0].response_hash is not None

    def test_none_response_does_not_get_hashed(self) -> None:
        """None response should emit response_hash=None (no response data)."""
        from unittest.mock import MagicMock

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.plugins.context import PluginContext

        emitted_events: list = []

        def capture_telemetry(event):
            emitted_events.append(event)

        mock_landscape = MagicMock()
        mock_landscape.record_call.return_value = MagicMock(call_id="call-001")

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_landscape,
            state_id="state-001",
            telemetry_emit=capture_telemetry,
        )

        # None response - truly no data
        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data={"method": "HEAD"},
            response_data=None,  # No response data at all
            latency_ms=15.0,
            status=CallStatus.SUCCESS,
        )

        assert len(emitted_events) == 1
        # None is correct for truly missing response
        assert emitted_events[0].response_hash is None
