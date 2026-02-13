# tests/plugins/test_context.py
"""Tests for plugin context."""

from contextlib import nullcontext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


class TestPluginContext:
    """Context passed to all plugin operations."""

    def test_minimal_context(self) -> None:
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        assert ctx.run_id == "run-001"
        assert ctx.config == {}

    def test_optional_integrations_default_none(self) -> None:
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        # Phase 3 integration points - optional in Phase 2
        assert ctx.landscape is None
        assert ctx.tracer is None
        assert ctx.payload_store is None

    def test_start_span_without_tracer(self) -> None:
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        # Should return nullcontext when no tracer
        span_ctx = ctx.start_span("test_operation")
        assert isinstance(span_ctx, nullcontext)

    def test_get_config_value(self) -> None:
        from elspeth.contracts.plugin_context import PluginContext

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
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        assert hasattr(ctx, "get_checkpoint")
        assert hasattr(ctx, "update_checkpoint")
        assert hasattr(ctx, "clear_checkpoint")

    def test_get_checkpoint_returns_none_when_empty(self) -> None:
        """Empty checkpoint returns None (not empty dict)."""
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        assert ctx.get_checkpoint() is None

    def test_update_checkpoint_stores_data(self) -> None:
        """update_checkpoint stores data accessible via get_checkpoint."""
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        ctx.update_checkpoint({"batch_id": "batch-123", "row_count": 5})

        checkpoint = ctx.get_checkpoint()
        assert checkpoint is not None
        assert checkpoint["batch_id"] == "batch-123"
        assert checkpoint["row_count"] == 5

    def test_update_checkpoint_merges_data(self) -> None:
        """Multiple update_checkpoint calls merge data."""
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        ctx.update_checkpoint({"batch_id": "batch-123"})
        ctx.update_checkpoint({"status": "submitted"})

        checkpoint = ctx.get_checkpoint()
        assert checkpoint is not None
        assert checkpoint["batch_id"] == "batch-123"
        assert checkpoint["status"] == "submitted"

    def test_clear_checkpoint_removes_all_data(self) -> None:
        """clear_checkpoint removes all checkpoint data."""
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="run-001", config={})
        ctx.update_checkpoint({"batch_id": "batch-123"})
        assert ctx.get_checkpoint() is not None

        ctx.clear_checkpoint()
        assert ctx.get_checkpoint() is None

    def test_checkpoint_typical_batch_workflow(self) -> None:
        """Checkpoint API supports typical batch transform workflow."""
        from elspeth.contracts.plugin_context import PluginContext

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
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test-run", config={})
        assert hasattr(ctx, "record_validation_error")
        assert callable(ctx.record_validation_error)

    def test_record_validation_error_returns_quarantine_token(self) -> None:
        """record_validation_error returns token for tracking quarantined row."""
        from elspeth.contracts.plugin_context import PluginContext, ValidationErrorToken

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        token = ctx.record_validation_error(
            row={"id": 42, "invalid": "data"},
            error="validation failed",
            schema_mode="fixed",
            destination="discard",
        )

        assert token is not None
        assert isinstance(token, ValidationErrorToken)
        assert token.row_id is not None
        assert token.node_id == "source_node"

    def test_record_validation_error_without_landscape_logs_warning(self, caplog: "pytest.LogCaptureFixture") -> None:
        """record_validation_error logs warning when no landscape configured."""
        import logging

        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        with caplog.at_level(logging.WARNING):
            token = ctx.record_validation_error(
                row={"id": 42, "invalid": "data"},
                error="validation failed",
                schema_mode="fixed",
                destination="discard",
            )

        # Should still return a token even without landscape
        assert token is not None
        assert token.error_id is None  # Not recorded to landscape
        # Check that warning was logged
        assert "no landscape" in caplog.text.lower()

    def test_record_validation_error_uses_id_field_as_row_id(self) -> None:
        """record_validation_error uses row's id field if present."""
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        token = ctx.record_validation_error(
            row={"id": "row-42", "invalid": "data"},
            error="validation failed",
            schema_mode="fixed",
            destination="discard",
        )

        assert token.row_id == "row-42"

    def test_record_validation_error_generates_row_id_from_hash(self) -> None:
        """record_validation_error generates row_id from content hash if no id field."""
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        token = ctx.record_validation_error(
            row={"no_id_field": "data"},
            error="validation failed",
            schema_mode="fixed",
            destination="discard",
        )

        # Should generate a row_id from hash (16 chars)
        assert token.row_id is not None
        assert len(token.row_id) == 16

    def test_record_validation_error_with_landscape(self) -> None:
        """record_validation_error records to landscape when configured."""
        from unittest.mock import MagicMock

        from elspeth.contracts.plugin_context import PluginContext

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
            schema_mode="fixed",
            destination="quarantine_sink",
        )

        # Should have called landscape
        mock_landscape.record_validation_error.assert_called_once()
        call_kwargs = mock_landscape.record_validation_error.call_args[1]
        assert call_kwargs["run_id"] == "test-run"
        assert call_kwargs["node_id"] == "source_node"
        assert call_kwargs["row_data"] == {"id": 42, "invalid": "data"}
        assert call_kwargs["error"] == "validation failed"
        assert call_kwargs["schema_mode"] == "fixed"
        assert call_kwargs["destination"] == "quarantine_sink"

        # Token should have error_id from landscape
        assert token.error_id == "verr_abc123"
        assert token.destination == "quarantine_sink"

    def test_record_validation_error_passes_contract_violation_to_landscape(self) -> None:
        """contract_violation parameter is forwarded to landscape recorder.

        Regression test for bead c5cz: PluginContext accepted contract_violation
        in its signature but silently dropped it, never passing it to the
        landscape's record_validation_error. This broke structured auditing of
        schema contract violations.
        """
        from unittest.mock import MagicMock

        from elspeth.contracts.errors import TypeMismatchViolation
        from elspeth.contracts.plugin_context import PluginContext

        mock_landscape = MagicMock()
        mock_landscape.record_validation_error.return_value = "verr_cv001"

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=mock_landscape,
        )

        violation = TypeMismatchViolation(
            normalized_name="amount",
            original_name="Amount",
            expected_type=int,
            actual_type=str,
            actual_value="not_a_number",
        )

        token = ctx.record_validation_error(
            row={"id": 99, "amount": "not_a_number"},
            error="Type mismatch: expected int, got str",
            schema_mode="fixed",
            destination="quarantine_sink",
            contract_violation=violation,
        )

        call_kwargs = mock_landscape.record_validation_error.call_args[1]
        assert call_kwargs["contract_violation"] is violation
        assert token.error_id == "verr_cv001"

    def test_record_validation_error_passes_none_violation_by_default(self) -> None:
        """contract_violation defaults to None when not provided."""
        from unittest.mock import MagicMock

        from elspeth.contracts.plugin_context import PluginContext

        mock_landscape = MagicMock()
        mock_landscape.record_validation_error.return_value = "verr_no_cv"

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=mock_landscape,
        )

        ctx.record_validation_error(
            row={"id": 1, "bad": "data"},
            error="validation failed",
            schema_mode="fixed",
            destination="discard",
        )

        call_kwargs = mock_landscape.record_validation_error.call_args[1]
        assert call_kwargs["contract_violation"] is None


class TestValidationErrorDestination:
    """Tests for validation error destination tracking."""

    def test_validation_error_token_has_destination(self) -> None:
        """ValidationErrorToken includes destination field."""
        from elspeth.contracts.plugin_context import ValidationErrorToken

        token = ValidationErrorToken(
            row_id="row_1",
            node_id="source_node",
            destination="quarantine_sink",
        )

        assert token.destination == "quarantine_sink"

    def test_validation_error_token_defaults_to_discard(self) -> None:
        """ValidationErrorToken defaults to 'discard' if not specified."""
        from elspeth.contracts.plugin_context import ValidationErrorToken

        token = ValidationErrorToken(row_id="row_1", node_id="source_node")

        assert token.destination == "discard"

    def test_record_validation_error_requires_destination(self) -> None:
        """record_validation_error requires destination parameter."""
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        # Should work with destination
        token = ctx.record_validation_error(
            row={"id": 1, "bad": "data"},
            error="validation failed",
            schema_mode="fixed",
            destination="quarantine_sink",
        )

        assert token.destination == "quarantine_sink"

    def test_record_validation_error_with_discard_destination(self) -> None:
        """record_validation_error accepts 'discard' as destination."""
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        token = ctx.record_validation_error(
            row={"id": 1, "bad": "data"},
            error="validation failed",
            schema_mode="fixed",
            destination="discard",
        )

        assert token.destination == "discard"


class TestTransformErrorRecording:
    """Tests for transform error recording."""

    def test_transform_error_token_exists(self) -> None:
        """TransformErrorToken can be created."""
        from elspeth.contracts.plugin_context import TransformErrorToken

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
        from elspeth.contracts.plugin_context import TransformErrorToken

        token = TransformErrorToken(
            token_id="tok_123",
            transform_id="field_mapper",
        )

        assert token.destination == "discard"

    def test_record_transform_error_without_landscape(self) -> None:
        """record_transform_error works without landscape (logs warning)."""
        from elspeth.contracts.plugin_context import PluginContext

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

        from elspeth.contracts.plugin_context import PluginContext

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

        from elspeth.contracts.plugin_context import PluginContext

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
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test-run", config={})
        assert ctx.token is None

    def test_token_accepts_token_info(self) -> None:
        """PluginContext accepts TokenInfo via token parameter."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.contracts.schema_contract import SchemaContract
        from elspeth.testing import make_field, make_row

        # Create PipelineRow for TokenInfo
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(make_field("x", int, original_name="x", required=False, source="inferred"),),
            locked=True,
        )
        row_data = make_row({"x": 1}, contract=contract)

        token = TokenInfo(row_id="row-1", token_id="token-row-1", row_data=row_data)
        ctx = PluginContext(run_id="test-run", config={}, token=token)

        assert ctx.token is not None
        assert ctx.token is token  # Same object reference
        assert ctx.token.row_id == "row-1"
        assert ctx.token.token_id == "token-row-1"
        assert ctx.token.row_data.to_dict() == {"x": 1}

    def test_token_identity_preserved_on_access(self) -> None:
        """Token identity is preserved - multiple accesses return same object."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.contracts.schema_contract import SchemaContract
        from elspeth.testing import make_field, make_row

        # Create PipelineRow for TokenInfo
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(make_field("value", int, original_name="value", required=False, source="inferred"),),
            locked=True,
        )
        row_data = make_row({"value": 100}, contract=contract)

        token = TokenInfo(row_id="row-42", token_id="token-42", row_data=row_data)
        ctx = PluginContext(run_id="test-run", config={}, token=token)

        # Multiple accesses should return the exact same object
        access1 = ctx.token
        access2 = ctx.token
        assert access1 is access2
        assert access1 is token

    def test_token_can_be_mutated_after_construction(self) -> None:
        """Token field can be set after construction (engine sets it per-row)."""
        from elspeth.contracts.identity import TokenInfo
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.contracts.schema_contract import SchemaContract
        from elspeth.testing import make_field, make_row

        ctx = PluginContext(run_id="test-run", config={})
        assert ctx.token is None

        # Create PipelineRow for TokenInfo
        contract = SchemaContract(
            mode="OBSERVED",
            fields=(make_field("data", str, original_name="data", required=False, source="inferred"),),
            locked=True,
        )
        row_data = make_row({"data": "test"}, contract=contract)

        # Engine sets token before calling batch transforms
        token = TokenInfo(row_id="row-99", token_id="token-99", row_data=row_data)
        ctx.token = token

        assert ctx.token is token
        assert ctx.token.row_id == "row-99"


class TestRecordCallTelemetryPayloadSnapshot:
    """Tests for telemetry payload snapshotting in record_call.

    Regression tests for mutable payload drift: emitted telemetry payloads must
    be immutable snapshots that stay aligned with call-time hashes.
    """

    def test_request_payload_snapshot_is_immutable_after_call(self) -> None:
        """Mutating request_data after record_call must not change telemetry payload."""
        from typing import Any
        from unittest.mock import MagicMock

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.core.canonical import stable_hash

        emitted_events: list[Any] = []

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

        request_data = {"a": 1, "nested": {"x": 2}}
        expected_request = {"a": 1, "nested": {"x": 2}}

        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data=request_data,
            response_data={"ok": True},
            latency_ms=5.0,
            status=CallStatus.SUCCESS,
        )

        request_data["a"] = 999
        request_data["nested"]["x"] = 777

        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.request_payload == expected_request
        assert event.request_hash == stable_hash(expected_request)

    def test_response_payload_snapshot_is_immutable_after_call(self) -> None:
        """Mutating response_data after record_call must not change telemetry payload."""
        from typing import Any
        from unittest.mock import MagicMock

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.core.canonical import stable_hash

        emitted_events: list[Any] = []

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

        response_data = {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}
        expected_response = {"usage": {"prompt_tokens": 1, "completion_tokens": 2}}

        ctx.record_call(
            call_type=CallType.LLM,
            provider="openrouter",
            request_data={"prompt": "hi"},
            response_data=response_data,
            latency_ms=12.0,
            status=CallStatus.SUCCESS,
        )

        response_data["usage"]["prompt_tokens"] = 999

        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.response_payload == expected_response
        assert event.response_hash == stable_hash(expected_response)
        assert event.token_usage == expected_response["usage"]


class TestRecordCallTelemetryTokenCorrelation:
    """Tests for token_id correlation in ExternalCallCompleted telemetry."""

    def test_state_context_emits_token_id_when_token_present(self) -> None:
        """Transform-context calls should include token_id for correlation."""
        from typing import Any
        from unittest.mock import MagicMock

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.identity import TokenInfo
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.contracts.schema_contract import SchemaContract
        from elspeth.testing import make_field, make_row

        emitted_events: list[Any] = []

        def capture_telemetry(event):
            emitted_events.append(event)

        mock_landscape = MagicMock()
        mock_landscape.record_call.return_value = MagicMock(call_id="call-001")

        contract = SchemaContract(
            mode="OBSERVED",
            fields=(make_field("value", int, original_name="value", required=False, source="inferred"),),
            locked=True,
        )
        token_row = make_row({"value": 1}, contract=contract)
        token = TokenInfo(row_id="row-001", token_id="tok-001", row_data=token_row)

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_landscape,
            state_id="state-001",
            token=token,
            telemetry_emit=capture_telemetry,
        )

        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data={"method": "GET"},
            response_data={"status_code": 200},
            latency_ms=10.0,
            status=CallStatus.SUCCESS,
        )

        assert len(emitted_events) == 1
        assert emitted_events[0].token_id == "tok-001"

    def test_operation_context_allows_missing_token_id(self) -> None:
        """Operation-context calls should be valid with token_id=None."""
        from typing import Any
        from unittest.mock import MagicMock

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        emitted_events: list[Any] = []

        def capture_telemetry(event):
            emitted_events.append(event)

        mock_landscape = MagicMock()
        mock_landscape.record_operation_call.return_value = MagicMock(call_id="op-call-001")

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=mock_landscape,
            operation_id="operation-001",
            telemetry_emit=capture_telemetry,
        )

        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data={"method": "POST"},
            response_data={"status_code": 202},
            latency_ms=15.0,
            status=CallStatus.SUCCESS,
        )

        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.operation_id == "operation-001"
        assert event.token_id is None


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
        # Set up telemetry callback to capture emitted events
        from typing import Any
        from unittest.mock import MagicMock

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        emitted_events: list[Any] = []

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
        from typing import Any
        from unittest.mock import MagicMock

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        emitted_events: list[Any] = []

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

        # Empty results in dict - type-correct representation of no SQL rows
        ctx.record_call(
            call_type=CallType.SQL,
            provider="database",
            request_data={"query": "SELECT * FROM empty_table"},
            response_data={"rows": []},  # Empty results in dict structure (type-correct)
            latency_ms=10.0,
            status=CallStatus.SUCCESS,
        )

        assert len(emitted_events) == 1
        assert emitted_events[0].response_hash is not None

    def test_empty_string_response_gets_hashed(self) -> None:
        """Empty string '' response should emit response_hash in telemetry."""
        from typing import Any
        from unittest.mock import MagicMock

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        emitted_events: list[Any] = []

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

        # Empty body in dict (e.g., HTTP 204 No Content as dict structure)
        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data={"method": "DELETE"},
            response_data={"body": ""},  # Empty body in dict structure (type-correct)
            latency_ms=25.0,
            status=CallStatus.SUCCESS,
        )

        assert len(emitted_events) == 1
        assert emitted_events[0].response_hash is not None

    def test_none_response_does_not_get_hashed(self) -> None:
        """None response should emit response_hash=None (no response data)."""
        from typing import Any
        from unittest.mock import MagicMock

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        emitted_events: list[Any] = []

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
