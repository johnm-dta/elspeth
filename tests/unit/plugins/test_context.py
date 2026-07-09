# tests/plugins/test_context.py
"""Tests for plugin context."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from tests.fixtures.factories import make_source_context


def _call_result(
    *,
    call_id: str = "call-001",
    request_hash: str | None = None,
    response_hash: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(call_id=call_id, request_hash=request_hash, response_hash=response_hash)


@dataclass(slots=True)
class _RecordingLandscape:
    validation_error_id: str | None = None
    transform_error_id: str | None = None
    call_result: Any = field(default_factory=_call_result)
    operation_call_result: Any = field(default_factory=lambda: _call_result(call_id="op-call-001"))
    node_state: Any = field(default_factory=lambda: SimpleNamespace(token_id=None))
    validation_error_calls: list[dict[str, Any]] = field(default_factory=list)
    transform_error_calls: list[dict[str, Any]] = field(default_factory=list)
    record_call_calls: list[dict[str, Any]] = field(default_factory=list)
    record_operation_call_calls: list[dict[str, Any]] = field(default_factory=list)
    allocate_call_index_calls: list[str] = field(default_factory=list)
    get_node_state_calls: list[str] = field(default_factory=list)

    def record_validation_error(self, **kwargs: Any) -> str | None:
        self.validation_error_calls.append(kwargs)
        return self.validation_error_id

    def record_transform_error(self, **kwargs: Any) -> str | None:
        self.transform_error_calls.append(kwargs)
        return self.transform_error_id

    def allocate_call_index(self, state_id: str) -> int:
        self.allocate_call_index_calls.append(state_id)
        return len(self.allocate_call_index_calls)

    def record_call(self, **kwargs: Any) -> Any:
        self.record_call_calls.append(kwargs)
        return self.call_result

    def record_operation_call(self, **kwargs: Any) -> Any:
        self.record_operation_call_calls.append(kwargs)
        return self.operation_call_result

    def get_node_state(self, state_id: str) -> Any:
        self.get_node_state_calls.append(state_id)
        return self.node_state


class TestPluginContext:
    """Context passed to all plugin operations."""

    def test_minimal_context(self) -> None:
        from elspeth.contracts.plugin_context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = RecorderFactory(db).plugin_audit_writer()
        ctx = PluginContext(run_id="run-001", config={}, landscape=recorder)
        assert ctx.run_id == "run-001"
        assert ctx.config == {}

    def test_optional_integrations_default_none(self) -> None:
        from elspeth.contracts.plugin_context import PluginContext

        db = LandscapeDB.in_memory()
        recorder = RecorderFactory(db).plugin_audit_writer()
        ctx = PluginContext(run_id="run-001", config={}, landscape=recorder)
        assert ctx.rate_limit_registry is None
        assert ctx.concurrency_config is None

    def test_config_cannot_be_rebound_after_construction(self) -> None:
        import pytest

        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="run-001", config={"source": {"path": "input.csv"}})

        with pytest.raises(AttributeError):
            ctx.config = {"source": {"path": "other.csv"}}  # type: ignore[misc]

        assert ctx.config["source"]["path"] == "input.csv"

    def test_dataclasses_replace_preserves_read_only_config(self) -> None:
        import dataclasses

        from elspeth.contracts.node_state_context import AggregationBatchContext
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="run-001", config={"source": {"path": "input.csv"}})

        replaced = dataclasses.replace(
            ctx,
            aggregation_batch=AggregationBatchContext(
                trigger_type="count",
                batch_id="agg",
                batch_size=1,
                flush_index=1,
                row_start=1,
                row_end=1,
                rows_seen_total=1,
                is_end_of_source=False,
            ),
        )

        assert replaced.config == ctx.config
        assert replaced.config["source"]["path"] == "input.csv"
        assert replaced.aggregation_batch is not None


class TestValidationErrorRecording:
    """Tests for recording validation errors from sources."""

    def test_record_validation_error_returns_quarantine_token(self) -> None:
        """record_validation_error returns token for tracking quarantined row."""
        from elspeth.contracts.plugin_context import ValidationErrorToken

        ctx = make_source_context(node_id="source_node")

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

    def test_record_validation_error_without_landscape_raises(self) -> None:
        """record_validation_error raises FrameworkBugError when no landscape configured."""
        import pytest

        from elspeth.contracts.errors import FrameworkBugError
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="source_node")

        with pytest.raises(FrameworkBugError, match=r"record_validation_error.*without landscape"):
            ctx.record_validation_error(
                row={"id": 42, "invalid": "data"},
                error="validation failed",
                schema_mode="fixed",
                destination="discard",
            )

    def test_record_validation_error_uses_id_field_as_row_id(self) -> None:
        """record_validation_error uses row's id field if present."""
        ctx = make_source_context(node_id="source_node")

        token = ctx.record_validation_error(
            row={"id": "row-42", "invalid": "data"},
            error="validation failed",
            schema_mode="fixed",
            destination="discard",
        )

        assert token.row_id == "row-42"

    def test_record_validation_error_generates_row_id_from_hash(self) -> None:
        """record_validation_error generates row_id from content hash if no id field."""
        ctx = make_source_context(node_id="source_node")

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
        from elspeth.contracts.plugin_context import PluginContext

        landscape = _RecordingLandscape(validation_error_id="verr_abc123")

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=landscape,
        )

        token = ctx.record_validation_error(
            row={"id": 42, "invalid": "data"},
            error="validation failed",
            schema_mode="fixed",
            destination="quarantine_sink",
        )

        # Should have called landscape
        assert len(landscape.validation_error_calls) == 1
        call_kwargs = landscape.validation_error_calls[0]
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
        from elspeth.contracts.errors import TypeMismatchViolation
        from elspeth.contracts.plugin_context import PluginContext

        landscape = _RecordingLandscape(validation_error_id="verr_cv001")

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=landscape,
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

        call_kwargs = landscape.validation_error_calls[0]
        assert call_kwargs["contract_violation"] is violation
        assert token.error_id == "verr_cv001"

    def test_record_validation_error_passes_none_violation_by_default(self) -> None:
        """contract_violation defaults to None when not provided."""
        from elspeth.contracts.plugin_context import PluginContext

        landscape = _RecordingLandscape(validation_error_id="verr_no_cv")

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="source_node",
            landscape=landscape,
        )

        ctx.record_validation_error(
            row={"id": 1, "bad": "data"},
            error="validation failed",
            schema_mode="fixed",
            destination="discard",
        )

        call_kwargs = landscape.validation_error_calls[0]
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
        ctx = make_source_context(node_id="source_node")

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
        ctx = make_source_context(node_id="source_node")

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

    def test_record_transform_error_without_landscape_raises(self) -> None:
        """record_transform_error raises FrameworkBugError when no landscape configured."""
        import pytest

        from elspeth.contracts.errors import FrameworkBugError
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test-run", config={}, node_id="transform_node")

        with pytest.raises(FrameworkBugError, match=r"record_transform_error.*without landscape"):
            ctx.record_transform_error(
                token_id="tok_123",
                transform_id="field_mapper",
                row={"id": 1, "data": "test"},
                error_details={"reason": "validation_failed", "error": "Cannot process"},
                destination="error_sink",
            )

    def test_record_transform_error_with_landscape(self) -> None:
        """record_transform_error stores in audit trail via mock."""
        from elspeth.contracts.plugin_context import PluginContext

        landscape = _RecordingLandscape(transform_error_id="terr_abc123def4")

        ctx = PluginContext(
            run_id="test-run",
            config={},
            node_id="transform_node",
            landscape=landscape,
        )

        token = ctx.record_transform_error(
            token_id="tok_456",
            transform_id="field_mapper",
            row={"id": 42, "value": "bad"},
            error_details={"reason": "validation_failed", "error": "Division by zero"},
            destination="failed_rows",
        )

        # Should have called landscape
        assert len(landscape.transform_error_calls) == 1
        call_kwargs = landscape.transform_error_calls[0]
        assert call_kwargs["ref"].run_id == "test-run"
        assert call_kwargs["ref"].token_id == "tok_456"
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

        db = LandscapeDB.in_memory()
        recorder = RecorderFactory(db).plugin_audit_writer()
        ctx = PluginContext(run_id="test-run", config={}, landscape=recorder)
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
        db = LandscapeDB.in_memory()
        recorder = RecorderFactory(db).plugin_audit_writer()
        ctx = PluginContext(run_id="test-run", config={}, landscape=recorder, token=token)

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
        db = LandscapeDB.in_memory()
        recorder = RecorderFactory(db).plugin_audit_writer()
        ctx = PluginContext(run_id="test-run", config={}, landscape=recorder, token=token)

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

        db = LandscapeDB.in_memory()
        recorder = RecorderFactory(db).plugin_audit_writer()
        ctx = PluginContext(run_id="test-run", config={}, landscape=recorder)
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

        from elspeth.contracts.call_data import RawCallPayload
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.core.canonical import stable_hash

        emitted_events: list[Any] = []

        def capture_telemetry(event):
            emitted_events.append(event)

        request_data: dict[str, Any] = {"a": 1, "nested": {"x": 2}}
        expected_request: dict[str, Any] = {"a": 1, "nested": {"x": 2}}

        landscape = _RecordingLandscape(
            call_result=_call_result(
                call_id="call-001",
                request_hash=stable_hash(expected_request),
                response_hash=stable_hash({"ok": True}),
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
            state_id="state-001",
            telemetry_emit=capture_telemetry,
        )

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
        assert isinstance(event.request_payload, RawCallPayload)
        assert event.request_payload.to_dict() == expected_request
        assert event.request_hash == stable_hash(expected_request)

    def test_response_payload_snapshot_is_immutable_after_call(self) -> None:
        """Mutating response_data after record_call must not change telemetry payload."""
        from typing import Any

        from elspeth.contracts.call_data import RawCallPayload
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.core.canonical import stable_hash

        emitted_events: list[Any] = []

        def capture_telemetry(event):
            emitted_events.append(event)

        landscape = _RecordingLandscape(
            call_result=_call_result(
                call_id="call-001",
                request_hash=stable_hash({"prompt": "hi"}),
                response_hash=stable_hash({"usage": {"prompt_tokens": 1, "completion_tokens": 2}}),
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
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
        assert isinstance(event.response_payload, RawCallPayload)
        assert event.response_payload.to_dict() == expected_response
        assert event.response_hash == stable_hash(expected_response)
        from elspeth.contracts.token_usage import TokenUsage

        assert event.token_usage == TokenUsage(prompt_tokens=1, completion_tokens=2)


class TestRecordCallTelemetryTokenCorrelation:
    """Tests for token_id correlation in ExternalCallCompleted telemetry."""

    def test_state_context_emits_token_id_when_token_present(self) -> None:
        """Transform-context calls should include token_id for correlation."""
        from typing import Any

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.identity import TokenInfo
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.contracts.schema_contract import SchemaContract
        from elspeth.testing import make_field, make_row

        emitted_events: list[Any] = []

        def capture_telemetry(event):
            emitted_events.append(event)

        # get_node_state must return an object with the authoritative token_id
        # (record_call resolves token_id from state_id lookup, not ctx.token)
        landscape = _RecordingLandscape(
            call_result=_call_result(call_id="call-001"),
            node_state=SimpleNamespace(token_id="tok-001"),
        )

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
            landscape=landscape,
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

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        emitted_events: list[Any] = []

        def capture_telemetry(event):
            emitted_events.append(event)

        landscape = _RecordingLandscape(operation_call_result=_call_result(call_id="op-call-001"))

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
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

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        emitted_events: list[Any] = []

        def capture_telemetry(event):
            emitted_events.append(event)

        # Fake landscape recorder is the single source of truth for hashes.
        # Empty dict {} is a valid response with a real hash, not None.
        from elspeth.core.canonical import stable_hash

        landscape = _RecordingLandscape(
            call_result=_call_result(
                call_id="call-001",
                request_hash=stable_hash({"endpoint": "/empty"}),
                response_hash=stable_hash({}),
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
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

        # Verify telemetry was emitted with response_hash from recorder
        assert len(emitted_events) == 1
        event = emitted_events[0]
        # response_hash should NOT be None for empty dict
        assert event.response_hash is not None, (
            "Empty dict response should still get hashed. Got response_hash=None which breaks telemetry/audit correlation."
        )
        assert event.response_hash == stable_hash({})

    def test_empty_list_response_gets_hashed(self) -> None:
        """Empty list [] response should emit response_hash in telemetry."""
        from typing import Any

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        emitted_events: list[Any] = []

        def capture_telemetry(event):
            emitted_events.append(event)

        from elspeth.core.canonical import stable_hash

        landscape = _RecordingLandscape(
            call_result=_call_result(
                call_id="call-001",
                request_hash=stable_hash({"query": "SELECT * FROM empty_table"}),
                response_hash=stable_hash({"rows": []}),
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
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

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.core.canonical import stable_hash

        emitted_events: list[Any] = []

        def capture_telemetry(event):
            emitted_events.append(event)

        landscape = _RecordingLandscape(
            call_result=_call_result(
                call_id="call-001",
                request_hash=stable_hash({"method": "DELETE"}),
                response_hash=stable_hash({"body": ""}),
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
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

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        emitted_events: list[Any] = []

        def capture_telemetry(event):
            emitted_events.append(event)

        landscape = _RecordingLandscape(
            call_result=_call_result(
                call_id="call-001",
                request_hash="abc123",
                response_hash=None,  # No response recorded
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
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


class TestRecordCallRawCallPayloadWrapping:
    """Tests for RawCallPayload wrapping in record_call.

    Verifies that request_data and response_data dicts are wrapped in
    RawCallPayload before being passed to the Landscape recorder (lines 262-265
    of plugin_context.py) and that None response_data is NOT wrapped.
    """

    def test_response_data_none_not_wrapped_in_raw_call_payload(self) -> None:
        """record_call() with response_data=None should pass None, not RawCallPayload(None)."""
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        landscape = _RecordingLandscape(
            call_result=_call_result(
                call_id="call-001",
                request_hash="req-hash",
                response_hash=None,
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
            state_id="state-001",
        )

        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data={"method": "HEAD"},
            response_data=None,
            latency_ms=5.0,
            status=CallStatus.SUCCESS,
        )

        # Verify record_call was invoked with response_data=None (not wrapped)
        call_kwargs = landscape.record_call_calls[0]
        assert call_kwargs["response_data"] is None

    def test_response_data_dict_wrapped_in_raw_call_payload(self) -> None:
        """record_call() with response_data=dict should wrap in RawCallPayload."""
        from elspeth.contracts.call_data import RawCallPayload
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        landscape = _RecordingLandscape(
            call_result=_call_result(
                call_id="call-002",
                request_hash="req-hash",
                response_hash="resp-hash",
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
            state_id="state-001",
        )

        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data={"method": "GET"},
            response_data={"key": "val"},
            latency_ms=10.0,
            status=CallStatus.SUCCESS,
        )

        call_kwargs = landscape.record_call_calls[0]
        # response_data should be RawCallPayload, not a raw dict
        assert isinstance(call_kwargs["response_data"], RawCallPayload)
        assert call_kwargs["response_data"].to_dict() == {"key": "val"}

    def test_request_data_wrapped_in_raw_call_payload(self) -> None:
        """record_call() always wraps request_data in RawCallPayload."""
        from elspeth.contracts.call_data import RawCallPayload
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        landscape = _RecordingLandscape(
            call_result=_call_result(
                call_id="call-003",
                request_hash="req-hash",
                response_hash=None,
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
            state_id="state-001",
        )

        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data={"endpoint": "/test"},
            response_data=None,
            latency_ms=5.0,
            status=CallStatus.SUCCESS,
        )

        call_kwargs = landscape.record_call_calls[0]
        assert isinstance(call_kwargs["request_data"], RawCallPayload)
        assert call_kwargs["request_data"].to_dict() == {"endpoint": "/test"}

    def test_telemetry_event_request_payload_is_raw_call_payload(self) -> None:
        """Telemetry event emitted by record_call has request_payload as RawCallPayload."""
        from typing import Any

        from elspeth.contracts.call_data import RawCallPayload
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        emitted_events: list[Any] = []

        def capture_telemetry(event: Any) -> None:
            emitted_events.append(event)

        landscape = _RecordingLandscape(
            call_result=_call_result(
                call_id="call-004",
                request_hash="req-hash",
                response_hash="resp-hash",
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
            state_id="state-001",
            telemetry_emit=capture_telemetry,
        )

        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data={"method": "POST", "body": "data"},
            response_data={"status": 200},
            latency_ms=12.0,
            status=CallStatus.SUCCESS,
        )

        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert isinstance(event.request_payload, RawCallPayload)
        assert isinstance(event.response_payload, RawCallPayload)

    def test_operation_context_also_wraps_in_raw_call_payload(self) -> None:
        """record_call() via operation_id path also wraps in RawCallPayload."""
        from elspeth.contracts.call_data import RawCallPayload
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext

        landscape = _RecordingLandscape(
            operation_call_result=_call_result(
                call_id="op-call-001",
                request_hash="req-hash",
                response_hash="resp-hash",
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
            operation_id="operation-001",
        )

        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data={"url": "https://example.com"},
            response_data={"body": "response"},
            latency_ms=20.0,
            status=CallStatus.SUCCESS,
        )

        call_kwargs = landscape.record_operation_call_calls[0]
        assert isinstance(call_kwargs["request_data"], RawCallPayload)
        assert isinstance(call_kwargs["response_data"], RawCallPayload)
        assert call_kwargs["request_data"].to_dict() == {"url": "https://example.com"}
        assert call_kwargs["response_data"].to_dict() == {"body": "response"}


class TestRecordCallFrozenData:
    """record_call must work with frozen container data (no thaw-refreeze).

    After removing deep_thaw from record_call, request_data/response_data
    may contain MappingProxyType values. RawCallPayload must receive and
    freeze them correctly, and token usage extraction must use Mapping ABC.
    """

    def test_token_usage_extracted_from_mapping_proxy(self) -> None:
        """Token usage extraction must work when raw_usage is MappingProxyType.

        This is the critical regression guard for the isinstance(Mapping)
        widening. If this check reverts to isinstance(dict), frozen LLM
        responses silently lose token usage data.
        """
        from types import MappingProxyType
        from typing import Any

        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.contracts.token_usage import TokenUsage
        from elspeth.core.canonical import stable_hash

        emitted_events: list[Any] = []

        landscape = _RecordingLandscape(
            call_result=_call_result(
                call_id="call-001",
                request_hash=stable_hash({"prompt": "hi"}),
                response_hash=stable_hash({"usage": {"prompt_tokens": 10, "completion_tokens": 5}}),
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
            state_id="state-001",
            telemetry_emit=emitted_events.append,
        )

        # Response with frozen usage dict (MappingProxyType, not plain dict)
        frozen_response = {
            "content": "hello",
            "usage": MappingProxyType(
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                }
            ),
        }

        ctx.record_call(
            call_type=CallType.LLM,
            provider="openrouter",
            request_data={"prompt": "hi"},
            response_data=frozen_response,
            latency_ms=12.0,
            status=CallStatus.SUCCESS,
        )

        assert len(emitted_events) == 1
        event = emitted_events[0]
        # Token usage must be extracted despite usage being MappingProxyType
        assert event.token_usage is not None
        assert event.token_usage == TokenUsage(prompt_tokens=10, completion_tokens=5, reported_total=15)

    def test_raw_call_payload_freezes_data_without_intermediate_thaw(self) -> None:
        """RawCallPayload must receive frozen data and freeze it correctly.

        Verifies the spec requirement: 'RawCallPayload receives and freezes
        the data correctly without intermediate thaw.'
        """
        from types import MappingProxyType
        from typing import Any

        from elspeth.contracts.call_data import RawCallPayload
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.plugin_context import PluginContext
        from elspeth.core.canonical import stable_hash

        emitted_events: list[Any] = []

        expected_request = {"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]}
        landscape = _RecordingLandscape(
            call_result=_call_result(
                call_id="call-001",
                request_hash=stable_hash(expected_request),
                response_hash=stable_hash({"ok": True}),
            )
        )

        ctx = PluginContext(
            run_id="test-run",
            config={},
            landscape=landscape,
            state_id="state-001",
            telemetry_emit=emitted_events.append,
        )

        # Request data with nested frozen containers
        frozen_request = {
            "model": "gpt-4",
            "messages": (MappingProxyType({"role": "user", "content": "hi"}),),
        }

        ctx.record_call(
            call_type=CallType.HTTP,
            provider="api.example.com",
            request_data=frozen_request,
            response_data={"ok": True},
            latency_ms=5.0,
            status=CallStatus.SUCCESS,
        )

        assert len(emitted_events) == 1
        event = emitted_events[0]
        # RawCallPayload must contain the data (frozen internally by deep_freeze)
        assert isinstance(event.request_payload, RawCallPayload)
        # to_dict() must produce the equivalent unfrozen structure
        assert event.request_payload.to_dict() == expected_request
