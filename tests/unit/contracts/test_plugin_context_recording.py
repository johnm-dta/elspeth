"""Tests for PluginContext audit recording helpers.

Tests the offensive programming guards (FrameworkBugError) and basic delegation
to ExecutionRepository. Uses make_source_context() for real landscape integration
and manual PluginContext construction for guard-clause tests.
"""

import logging
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import Mock

import pytest

from elspeth.contracts import CallStatus, CallType, FrameworkBugError, NodeStateStatus
from elspeth.contracts.audit import Call, NodeStateCompleted, TokenRef
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.events import ExternalCallCompleted
from elspeth.contracts.identity import TokenInfo
from elspeth.contracts.plugin_context import (
    PluginContext,
    TransformErrorToken,
    ValidationErrorToken,
)
from elspeth.testing import make_pipeline_row
from tests.fixtures.factories import make_source_context


def _completed_node_state(*, token_id: str = "token-001", state_id: str = "state-001") -> NodeStateCompleted:
    return NodeStateCompleted(
        state_id=state_id,
        token_id=token_id,
        node_id="transform-1",
        step_index=0,
        attempt=0,
        status=NodeStateStatus.COMPLETED,
        input_hash="input-hash",
        output_hash="output-hash",
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        completed_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        duration_ms=1.0,
    )


class _FakePluginAuditWriter:
    def __init__(self, *, node_state: NodeStateCompleted | None = None) -> None:
        self.node_state = node_state
        self.allocated_state_ids: list[str] = []
        self.node_state_lookups: list[str] = []
        self.state_calls: list[dict[str, Any]] = []
        self.operation_calls: list[dict[str, Any]] = []
        self.transform_error_calls: list[dict[str, Any]] = []

    def allocate_call_index(self, state_id: str) -> int:
        self.allocated_state_ids.append(state_id)
        return len(self.allocated_state_ids) - 1

    def record_call(
        self,
        *,
        state_id: str,
        call_index: int,
        call_type: CallType,
        status: CallStatus,
        request_data: RawCallPayload,
        response_data: RawCallPayload | None = None,
        error: RawCallPayload | None = None,
        latency_ms: float | None = None,
    ) -> Call:
        self.state_calls.append(
            {
                "state_id": state_id,
                "call_index": call_index,
                "call_type": call_type,
                "status": status,
                "request_data": request_data,
                "response_data": response_data,
                "error": error,
                "latency_ms": latency_ms,
            }
        )
        return Call(
            call_id=f"call-{call_index}",
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_hash="request-hash",
            response_hash="response-hash" if response_data is not None else None,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            state_id=state_id,
            latency_ms=latency_ms,
        )

    def record_operation_call(
        self,
        *,
        operation_id: str,
        call_type: CallType,
        status: CallStatus,
        request_data: RawCallPayload,
        response_data: RawCallPayload | None = None,
        error: RawCallPayload | None = None,
        latency_ms: float | None = None,
    ) -> Call:
        call_index = len(self.operation_calls)
        self.operation_calls.append(
            {
                "operation_id": operation_id,
                "call_type": call_type,
                "status": status,
                "request_data": request_data,
                "response_data": response_data,
                "error": error,
                "latency_ms": latency_ms,
            }
        )
        return Call(
            call_id=f"operation-call-{call_index}",
            call_index=call_index,
            call_type=call_type,
            status=status,
            request_hash="operation-request-hash",
            response_hash="operation-response-hash" if response_data is not None else None,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            operation_id=operation_id,
            latency_ms=latency_ms,
        )

    def get_node_state(self, state_id: str) -> NodeStateCompleted | None:
        self.node_state_lookups.append(state_id)
        return self.node_state

    def record_transform_error(
        self,
        *,
        ref: TokenRef,
        transform_id: str,
        row_data: Any,
        error_details: Any,
        destination: str,
    ) -> str:
        error_id = f"terr-{len(self.transform_error_calls)}"
        self.transform_error_calls.append(
            {
                "ref": ref,
                "transform_id": transform_id,
                "row_data": row_data,
                "error_details": error_details,
                "destination": destination,
                "error_id": error_id,
            }
        )
        return error_id


def _token_info(token_id: str) -> TokenInfo:
    return TokenInfo(row_id="row-001", token_id=token_id, row_data=make_pipeline_row({"value": 1}))


class TestRecordValidationErrorGuards:
    """record_validation_error() must crash on missing landscape or node_id."""

    def test_raises_when_landscape_is_none(self) -> None:
        ctx = PluginContext(run_id="run-1", config={}, landscape=None, node_id="source")
        with pytest.raises(FrameworkBugError, match=r"record_validation_error.*without landscape"):
            ctx.record_validation_error(
                row={"name": "test"},
                error="field X is NULL",
                schema_mode="fixed",
                destination="discard",
            )

    def test_raises_when_node_id_is_none(self) -> None:
        ctx = PluginContext(run_id="run-1", config={}, landscape=Mock(), node_id=None)
        with pytest.raises(FrameworkBugError, match=r"record_validation_error.*without node_id"):
            ctx.record_validation_error(
                row={"name": "test"},
                error="field X is NULL",
                schema_mode="fixed",
                destination="discard",
            )


class TestRecordValidationErrorHappyPath:
    """record_validation_error() delegates to landscape and returns token."""

    def test_returns_validation_error_token(self) -> None:
        """Happy path: row with id field -> token with that row_id."""
        ctx = make_source_context()
        token = ctx.record_validation_error(
            row={"id": "row-42", "name": "test"},
            error="field X is NULL",
            schema_mode="fixed",
            destination="discard",
        )
        assert isinstance(token, ValidationErrorToken)
        assert token.row_id == "row-42"
        assert token.node_id == "source"
        assert token.destination == "discard"
        assert token.error_id is not None  # Landscape assigns an error_id

    def test_row_without_id_uses_content_hash(self) -> None:
        """Row without 'id' field -> row_id derived from stable_hash."""
        ctx = make_source_context()
        token = ctx.record_validation_error(
            row={"name": "test"},
            error="missing required field",
            schema_mode="flexible",
            destination="quarantine_sink",
        )
        assert isinstance(token, ValidationErrorToken)
        assert len(token.row_id) == 16  # stable_hash[:16]
        assert token.destination == "quarantine_sink"

    def test_non_dict_row_uses_repr_hash(self) -> None:
        """Non-dict row (e.g., JSON primitive) -> row_id from repr_hash."""
        ctx = make_source_context()
        token = ctx.record_validation_error(
            row="not a dict",
            error="expected dict, got str",
            schema_mode="parse",
            destination="discard",
        )
        assert isinstance(token, ValidationErrorToken)
        assert len(token.row_id) == 16

    def test_non_canonical_row_does_not_leak_row_content_to_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        """elspeth-05a5727489: when stable_hash() fails on non-canonical row data, the
        fallback warning must log only the error TYPE, never str(e). Hashing
        canonicalization errors embed `Got: {obj!r}` (raw row content), which must
        stay in Landscape, not cross a normal logging boundary."""
        ctx = make_source_context()
        secret = "ROW-SECRET-payload-42"
        with caplog.at_level(logging.WARNING, logger="elspeth.contracts.plugin_context"):
            token = ctx.record_validation_error(
                row={"data": frozenset({secret})},  # frozenset -> non-canonical -> repr_hash fallback
                error="non-serializable external data",
                schema_mode="flexible",
                destination="discard",
            )
        assert isinstance(token, ValidationErrorToken)
        assert len(token.row_id) == 16  # repr_hash fallback still produced an id
        log_text = "\n".join(r.getMessage() for r in caplog.records)
        assert secret not in log_text  # row content must not leak through the logger
        assert "TypeError" in log_text  # the diagnostic error type is still logged

    def test_custom_destination_propagated(self) -> None:
        """Destination string flows through to the returned token."""
        ctx = make_source_context()
        token = ctx.record_validation_error(
            row={"id": "row-1"},
            error="bad data",
            schema_mode="fixed",
            destination="error_sink",
        )
        assert token.destination == "error_sink"

    def test_quarantine_destinations_queue_error_for_row_linkage(self) -> None:
        """Non-discard validation errors should be available for quarantine row linking."""
        ctx = make_source_context()
        row = {"name": "test"}

        token = ctx.record_validation_error(
            row=row,
            error="missing required field",
            schema_mode="fixed",
            destination="quarantine_sink",
        )

        assert ctx.pop_pending_quarantine_validation_error_id(row) == token.error_id
        assert ctx.pop_pending_quarantine_validation_error_id(row) is None

    def test_discard_validation_errors_are_not_queued_for_row_linkage(self) -> None:
        """Discarded rows should not leave stale pending linkage entries behind."""
        ctx = make_source_context()
        row = {"name": "discard-me"}

        ctx.record_validation_error(
            row=row,
            error="bad data",
            schema_mode="fixed",
            destination="discard",
        )

        assert ctx.pop_pending_quarantine_validation_error_id(row) is None

    def test_noncanonical_quarantine_linkage_uses_repr_fallback(self) -> None:
        """Pending linkage must still match non-canonical raw rows like NaN payloads."""
        ctx = make_source_context()
        row = {"value": float("nan")}

        token = ctx.record_validation_error(
            row=row,
            error="Row contains NaN",
            schema_mode="observed",
            destination="quarantine_sink",
        )

        assert ctx.pop_pending_quarantine_validation_error_id({"value": float("nan")}) == token.error_id


class TestRecordCallGuards:
    """record_call() must fail closed when audit parentage is invalid."""

    def test_raises_when_landscape_is_none(self) -> None:
        ctx = PluginContext(run_id="run-1", config={}, landscape=None, state_id="state-001")

        with pytest.raises(FrameworkBugError, match=r"record_call\(\) called without landscape"):
            ctx.record_call(
                CallType.HTTP,
                CallStatus.SUCCESS,
                {"url": "https://example.test"},
            )

    def test_raises_when_both_state_and_operation_are_set(self) -> None:
        writer = _FakePluginAuditWriter(node_state=_completed_node_state())
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=cast(Any, writer),
            state_id="state-001",
            operation_id="operation-001",
        )

        with pytest.raises(FrameworkBugError, match="BOTH state_id and operation_id"):
            ctx.record_call(
                CallType.HTTP,
                CallStatus.SUCCESS,
                {"url": "https://example.test"},
            )

        assert writer.state_calls == []
        assert writer.operation_calls == []

    def test_raises_when_no_parent_id_is_set(self) -> None:
        writer = _FakePluginAuditWriter()
        ctx = PluginContext(run_id="run-1", config={}, landscape=cast(Any, writer))

        with pytest.raises(FrameworkBugError, match="without state_id or operation_id"):
            ctx.record_call(
                CallType.HTTP,
                CallStatus.SUCCESS,
                {"url": "https://example.test"},
            )

        assert writer.state_calls == []
        assert writer.operation_calls == []

    def test_raises_when_state_id_cannot_be_resolved_to_node_state(self) -> None:
        writer = _FakePluginAuditWriter(node_state=None)
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=cast(Any, writer),
            state_id="state-001",
        )

        with pytest.raises(FrameworkBugError, match=r"get_node_state\(\) returned None"):
            ctx.record_call(
                CallType.HTTP,
                CallStatus.SUCCESS,
                {"url": "https://example.test"},
            )

        assert writer.allocated_state_ids == ["state-001"]
        assert writer.node_state_lookups == ["state-001"]
        assert len(writer.state_calls) == 1

    def test_raises_when_context_token_disagrees_with_node_state_token(self) -> None:
        writer = _FakePluginAuditWriter(node_state=_completed_node_state(token_id="token-from-state"))
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=cast(Any, writer),
            state_id="state-001",
            token=_token_info("token-from-context"),
        )

        with pytest.raises(FrameworkBugError, match="token mismatch"):
            ctx.record_call(
                CallType.HTTP,
                CallStatus.SUCCESS,
                {"url": "https://example.test"},
            )

        assert writer.allocated_state_ids == ["state-001"]
        assert writer.node_state_lookups == ["state-001"]
        assert len(writer.state_calls) == 1

    def test_propagates_landscape_record_call_failure_without_telemetry(self) -> None:
        writer = Mock()
        writer.allocate_call_index.return_value = 0
        writer.record_call.side_effect = RuntimeError("landscape call write failed")
        emitted_events: list[ExternalCallCompleted] = []
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=writer,
            state_id="state-001",
            telemetry_emit=emitted_events.append,
        )

        with pytest.raises(RuntimeError, match="landscape call write failed"):
            ctx.record_call(
                CallType.HTTP,
                CallStatus.SUCCESS,
                {"url": "https://example.test"},
                response_data={"status_code": 200},
                provider="example",
            )

        writer.allocate_call_index.assert_called_once_with("state-001")
        writer.get_node_state.assert_not_called()
        assert emitted_events == []
        record_kwargs = writer.record_call.call_args.kwargs
        assert record_kwargs["state_id"] == "state-001"
        assert record_kwargs["call_index"] == 0
        assert record_kwargs["call_type"] is CallType.HTTP
        assert record_kwargs["status"] is CallStatus.SUCCESS
        assert record_kwargs["request_data"].to_dict() == {"url": "https://example.test"}
        assert record_kwargs["response_data"].to_dict() == {"status_code": 200}


class TestRecordCallHappyPath:
    """record_call() writes to Landscape before emitting telemetry."""

    def test_state_context_records_call_and_emits_token_correlated_telemetry(self) -> None:
        writer = _FakePluginAuditWriter(node_state=_completed_node_state(token_id="token-001"))
        emitted_events: list[ExternalCallCompleted] = []
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=cast(Any, writer),
            state_id="state-001",
            token=_token_info("token-001"),
            telemetry_emit=emitted_events.append,
        )

        recorded = ctx.record_call(
            CallType.HTTP,
            CallStatus.SUCCESS,
            {"url": "https://example.test"},
            response_data={"status_code": 200},
            latency_ms=12.5,
            provider="example",
        )

        assert recorded is not None
        assert recorded.state_id == "state-001"
        assert recorded.operation_id is None
        assert recorded.call_index == 0
        assert writer.allocated_state_ids == ["state-001"]
        assert writer.node_state_lookups == ["state-001"]

        state_call = writer.state_calls[0]
        assert state_call["request_data"].to_dict() == {"url": "https://example.test"}
        assert state_call["response_data"].to_dict() == {"status_code": 200}
        assert state_call["error"] is None

        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.state_id == "state-001"
        assert event.operation_id is None
        assert event.token_id == "token-001"
        assert event.provider == "example"
        assert event.request_hash == recorded.request_hash
        assert event.response_hash == recorded.response_hash
        assert event.request_payload is not None
        assert event.request_payload.to_dict() == {"url": "https://example.test"}
        assert event.response_payload is not None
        assert event.response_payload.to_dict() == {"status_code": 200}

    def test_operation_context_records_operation_call_without_token_lookup(self) -> None:
        writer = _FakePluginAuditWriter()
        emitted_events: list[ExternalCallCompleted] = []
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=cast(Any, writer),
            operation_id="operation-001",
            telemetry_emit=emitted_events.append,
        )

        recorded = ctx.record_call(
            CallType.FILESYSTEM,
            CallStatus.ERROR,
            {"path": "/tmp/output.csv"},
            error={"type": "OSError", "message": "disk full"},
            provider="filesystem",
        )

        assert recorded is not None
        assert recorded.operation_id == "operation-001"
        assert recorded.state_id is None
        assert writer.allocated_state_ids == []
        assert writer.node_state_lookups == []
        assert writer.operation_calls[0]["request_data"].to_dict() == {"path": "/tmp/output.csv"}
        assert writer.operation_calls[0]["error"].to_dict() == {"type": "OSError", "message": "disk full"}
        assert writer.operation_calls[0]["latency_ms"] is None

        assert len(emitted_events) == 1
        event = emitted_events[0]
        assert event.state_id is None
        assert event.operation_id == "operation-001"
        assert event.token_id is None
        assert event.provider == "filesystem"
        assert event.latency_ms is None
        assert event.to_dict()["latency_ms"] is None
        assert event.request_hash == recorded.request_hash


class TestRecordTransformErrorGuards:
    """record_transform_error() must crash on missing landscape."""

    def test_raises_when_landscape_is_none(self) -> None:
        ctx = PluginContext(run_id="run-1", config={}, landscape=None, node_id="transform-1")
        with pytest.raises(FrameworkBugError, match=r"record_transform_error.*without landscape"):
            ctx.record_transform_error(
                token_id="tok-1",
                transform_id="transform-1",
                row={"data": "test"},
                error_details={"reason": "api_error", "error": "API returned 500"},
                destination="discard",
            )


class TestRecordTransformErrorHappyPath:
    """record_transform_error() delegates to landscape and returns token."""

    def test_returns_transform_error_token(self) -> None:
        """Happy path: landscape.record_transform_error is called and token fields are populated.

        record_transform_error requires a pre-existing token FK in the DB.
        Use a Mock landscape to test the delegation and return-value logic
        without needing to build the full token/row/node FK chain — that
        belongs in integration tests (test_recorder_errors.py).
        """
        writer = _FakePluginAuditWriter()
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=cast(Any, writer),
            node_id="transform-1",
        )
        token = ctx.record_transform_error(
            token_id="tok-1",
            transform_id="transform-1",
            row={"data": "test"},
            error_details={"reason": "api_error", "error": "API returned 500"},
            destination="error_sink",
        )
        assert isinstance(token, TransformErrorToken)
        assert token.token_id == "tok-1"
        assert token.transform_id == "transform-1"
        assert token.destination == "error_sink"
        assert token.error_id == "terr-0"
        assert writer.transform_error_calls == [
            {
                "ref": TokenRef(token_id="tok-1", run_id="run-1"),
                "transform_id": "transform-1",
                "row_data": {"data": "test"},
                "error_details": {"reason": "api_error", "error": "API returned 500"},
                "destination": "error_sink",
                "error_id": "terr-0",
            }
        ]

    def test_propagates_landscape_write_failure(self) -> None:
        mock_landscape = Mock()
        mock_landscape.record_transform_error.side_effect = RuntimeError("transform recorder failed")
        ctx = PluginContext(
            run_id="run-1",
            config={},
            landscape=mock_landscape,
            node_id="transform-1",
        )

        with pytest.raises(RuntimeError, match="transform recorder failed"):
            ctx.record_transform_error(
                token_id="tok-1",
                transform_id="transform-1",
                row={"data": "test"},
                error_details={"reason": "api_error", "error": "API returned 500"},
                destination="discard",
            )

        mock_landscape.record_transform_error.assert_called_once_with(
            ref=TokenRef(token_id="tok-1", run_id="run-1"),
            transform_id="transform-1",
            row_data={"data": "test"},
            error_details={"reason": "api_error", "error": "API returned 500"},
            destination="discard",
        )
