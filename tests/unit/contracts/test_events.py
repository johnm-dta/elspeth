"""Tests for contracts/events.py exports."""

from datetime import UTC, datetime

import pytest


class TestTokenCompletedTwoAxis:
    """ADR-019 Phase 1: TokenCompleted telemetry event carries (outcome, path)."""

    def test_token_completed_carries_outcome_and_path(self) -> None:
        from elspeth.contracts.enums import TerminalOutcome, TerminalPath
        from elspeth.contracts.events import TokenCompleted

        evt = TokenCompleted(
            timestamp=datetime.now(UTC),
            run_id="run_001",
            row_id="row_001",
            token_id="tok_001",
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="primary",
        )
        assert evt.outcome == TerminalOutcome.SUCCESS
        assert evt.path == TerminalPath.DEFAULT_FLOW


def test_transform_completed_allows_none_hashes():
    """TransformCompleted accepts None for input_hash and output_hash.

    Bug: P3-2026-01-31-transform-completed-output-hash-coercion

    Failed transforms may not have produced output, so output_hash should
    be None (not empty string). Semantically:
    - output_hash="" suggests "there was output, but it was empty"
    - output_hash=None correctly conveys "there was no output"
    """
    from elspeth.contracts import TransformCompleted
    from elspeth.contracts.enums import NodeStateStatus

    # Failed transform - no output hash
    event = TransformCompleted(
        timestamp=datetime.now(UTC),
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        node_id="node-1",
        plugin_name="test",
        status=NodeStateStatus.FAILED,
        duration_ms=50.0,
        input_hash="abc",
        output_hash=None,  # No output was produced
    )
    assert event.output_hash is None
    assert event.status == NodeStateStatus.FAILED


def test_transform_completed_allows_none_input_hash():
    """TransformCompleted accepts None for input_hash in edge cases."""
    from elspeth.contracts import TransformCompleted
    from elspeth.contracts.enums import NodeStateStatus

    event = TransformCompleted(
        timestamp=datetime.now(UTC),
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        node_id="node-1",
        plugin_name="test",
        status=NodeStateStatus.FAILED,
        duration_ms=0.0,
        input_hash=None,  # Edge case during error handling
        output_hash=None,
    )
    assert event.input_hash is None


class TestExternalCallCompletedSerialization:
    """Tests for ExternalCallCompleted.to_dict() with typed payloads.

    Verifies that to_dict() delegates to DTO.to_dict() instead of relying
    on generic dataclass field decomposition — which would produce wrong
    shapes for DTOs that omit None fields or spread extra_kwargs.
    """

    def test_to_dict_with_llm_dto_payloads(self) -> None:
        """LLM DTO payloads serialize correctly via to_dict()."""
        from elspeth.contracts.call_data import LLMCallRequest, LLMCallResponse
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.events import ExternalCallCompleted
        from elspeth.contracts.token_usage import TokenUsage

        req = LLMCallRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
            provider="azure",
            max_tokens=100,
            extra_kwargs={"top_p": 0.9},
        )
        resp = LLMCallResponse(
            content="world",
            model="gpt-4",
            usage=TokenUsage(prompt_tokens=5, completion_tokens=3),
            raw_response={"id": "resp-1"},
        )

        event = ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-1",
            call_type=CallType.LLM,
            provider="azure",
            status=CallStatus.SUCCESS,
            latency_ms=50.0,
            state_id="state-1",
            request_payload=req,
            response_payload=resp,
        )

        d = event.to_dict()

        # LLMCallRequest.to_dict() spreads extra_kwargs at top level
        assert d["request_payload"]["top_p"] == 0.9
        assert "extra_kwargs" not in d["request_payload"]
        assert d["request_payload"]["max_tokens"] == 100
        assert d["request_payload"]["model"] == "gpt-4"

        # LLMCallResponse.to_dict() serializes nested TokenUsage
        assert d["response_payload"]["content"] == "world"
        assert d["response_payload"]["usage"] == {"prompt_tokens": 5, "completion_tokens": 3}

    def test_to_dict_with_raw_call_payloads(self) -> None:
        """RawCallPayload.to_dict() returns the wrapped dict unchanged."""
        from elspeth.contracts.call_data import RawCallPayload
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.events import ExternalCallCompleted

        raw_req = RawCallPayload(data={"method": "GET", "url": "http://example.com"})
        raw_resp = RawCallPayload(data={"status_code": 200, "body": "ok"})

        event = ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-1",
            call_type=CallType.HTTP,
            provider="example.com",
            status=CallStatus.SUCCESS,
            latency_ms=10.0,
            state_id="state-1",
            request_payload=raw_req,
            response_payload=raw_resp,
        )

        d = event.to_dict()
        assert d["request_payload"] == {"method": "GET", "url": "http://example.com"}
        assert d["response_payload"] == {"status_code": 200, "body": "ok"}

    def test_to_dict_with_none_payloads(self) -> None:
        """None payloads remain None in to_dict() output."""
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.events import ExternalCallCompleted

        event = ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-1",
            call_type=CallType.HTTP,
            provider="example.com",
            status=CallStatus.ERROR,
            latency_ms=5.0,
            state_id="state-1",
            request_payload=None,
            response_payload=None,
        )

        d = event.to_dict()
        assert d["request_payload"] is None
        assert d["response_payload"] is None

    def test_to_dict_with_partial_token_usage_omits_none(self) -> None:
        """Partial TokenUsage omits unknown fields via DTO serializer.

        Bug: elspeth-rapid-359b31

        When only prompt_tokens is known, to_dict() must emit
        {"prompt_tokens": 42} — NOT {"prompt_tokens": 42, "completion_tokens": null}.
        The generic _event_field_to_serializable path walks all dataclass fields
        and includes None values; the override must call TokenUsage.to_dict().
        """
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.events import ExternalCallCompleted
        from elspeth.contracts.token_usage import TokenUsage

        event = ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-1",
            call_type=CallType.LLM,
            provider="azure",
            status=CallStatus.SUCCESS,
            latency_ms=50.0,
            state_id="state-1",
            token_usage=TokenUsage(prompt_tokens=42, completion_tokens=None),
        )

        d = event.to_dict()
        assert d["token_usage"] == {"prompt_tokens": 42}
        assert "completion_tokens" not in d["token_usage"]

    def test_to_dict_with_fully_unknown_token_usage_emits_empty(self) -> None:
        """Fully-unknown TokenUsage emits {} not {prompt_tokens: null, ...}.

        Bug: elspeth-rapid-359b31
        """
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.events import ExternalCallCompleted
        from elspeth.contracts.token_usage import TokenUsage

        event = ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-1",
            call_type=CallType.LLM,
            provider="azure",
            status=CallStatus.SUCCESS,
            latency_ms=50.0,
            state_id="state-1",
            token_usage=TokenUsage.unknown(),
        )

        d = event.to_dict()
        assert d["token_usage"] == {}

    def test_to_dict_with_full_token_usage(self) -> None:
        """Fully-known TokenUsage serializes both fields."""
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.events import ExternalCallCompleted
        from elspeth.contracts.token_usage import TokenUsage

        event = ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-1",
            call_type=CallType.LLM,
            provider="azure",
            status=CallStatus.SUCCESS,
            latency_ms=50.0,
            state_id="state-1",
            token_usage=TokenUsage.known(10, 5),
        )

        d = event.to_dict()
        assert d["token_usage"] == {"prompt_tokens": 10, "completion_tokens": 5}

    def test_to_dict_with_none_token_usage(self) -> None:
        """None token_usage remains None in to_dict() output."""
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.events import ExternalCallCompleted

        event = ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-1",
            call_type=CallType.HTTP,
            provider="example.com",
            status=CallStatus.SUCCESS,
            latency_ms=10.0,
            state_id="state-1",
            token_usage=None,
        )

        d = event.to_dict()
        assert d["token_usage"] is None

    def test_to_dict_with_http_dto_omits_none_fields(self) -> None:
        """HTTPCallRequest.to_dict() conditionally omits None fields."""
        from elspeth.contracts.call_data import HTTPCallRequest
        from elspeth.contracts.enums import CallStatus, CallType
        from elspeth.contracts.events import ExternalCallCompleted

        req = HTTPCallRequest(
            method="GET",
            url="http://example.com",
            headers={"Accept": "application/json"},
            resolved_ip="93.184.216.34",
        )

        event = ExternalCallCompleted(
            timestamp=datetime.now(UTC),
            run_id="run-1",
            call_type=CallType.HTTP,
            provider="example.com",
            status=CallStatus.SUCCESS,
            latency_ms=10.0,
            state_id="state-1",
            request_payload=req,
        )

        d = event.to_dict()
        # SSRF fields are additive — they never suppress other fields.
        # GET always emits params (even None) for hash stability.
        assert d["request_payload"]["resolved_ip"] == "93.184.216.34"
        assert "json" not in d["request_payload"]
        assert d["request_payload"]["params"] is None


# ---------------------------------------------------------------------------
# require_int validation: RunSummary, RunFinished, FieldResolutionApplied
# ---------------------------------------------------------------------------


class TestRunSummaryIntValidation:
    """RunSummary rejects bool values for int fields."""

    def test_rejects_bool_total_rows(self) -> None:
        from elspeth.contracts.events import RunCompletionStatus, RunSummary

        with pytest.raises(TypeError, match="total_rows must be int"):
            RunSummary(
                run_id="r",
                status=RunCompletionStatus.COMPLETED,
                total_rows=True,
                succeeded=0,
                failed=0,
                quarantined=0,
                duration_seconds=1.0,
                exit_code=0,
            )

    def test_rejects_bool_exit_code(self) -> None:
        from elspeth.contracts.events import RunCompletionStatus, RunSummary

        with pytest.raises(TypeError, match="exit_code must be int"):
            RunSummary(
                run_id="r",
                status=RunCompletionStatus.COMPLETED,
                total_rows=0,
                succeeded=0,
                failed=0,
                quarantined=0,
                duration_seconds=1.0,
                exit_code=False,
            )

    def test_rejects_negative_total_rows(self) -> None:
        from elspeth.contracts.events import RunCompletionStatus, RunSummary

        with pytest.raises(ValueError, match="total_rows must be >= 0"):
            RunSummary(
                run_id="r",
                status=RunCompletionStatus.COMPLETED,
                total_rows=-1,
                succeeded=0,
                failed=0,
                quarantined=0,
                duration_seconds=1.0,
                exit_code=0,
            )

    def test_valid_run_summary_accepted(self) -> None:
        from elspeth.contracts.events import RunCompletionStatus, RunSummary

        summary = RunSummary(
            run_id="r",
            status=RunCompletionStatus.COMPLETED,
            total_rows=10,
            succeeded=9,
            failed=1,
            quarantined=0,
            duration_seconds=2.5,
            exit_code=1,
            routed_success=3,
            routed_failure=0,
        )
        assert summary.total_rows == 10


class TestRunFinishedIntValidation:
    """RunFinished rejects bool values for row_count."""

    def test_rejects_bool_row_count(self) -> None:
        from datetime import UTC, datetime

        from elspeth.contracts.enums import RunStatus
        from elspeth.contracts.events import RunFinished

        with pytest.raises(TypeError, match="row_count must be int"):
            RunFinished(
                timestamp=datetime.now(UTC),
                run_id="r",
                status=RunStatus.COMPLETED,
                row_count=True,
                duration_ms=100.0,
            )

    def test_valid_run_finished_accepted(self) -> None:
        from datetime import UTC, datetime

        from elspeth.contracts.enums import RunStatus
        from elspeth.contracts.events import RunFinished

        event = RunFinished(
            timestamp=datetime.now(UTC),
            run_id="r",
            status=RunStatus.COMPLETED,
            row_count=5,
            duration_ms=100.0,
        )
        assert event.row_count == 5


class TestFieldResolutionAppliedIntValidation:
    """FieldResolutionApplied rejects bool values for field_count."""

    def test_rejects_bool_field_count(self) -> None:
        from datetime import UTC, datetime

        from elspeth.contracts.events import FieldResolutionApplied

        with pytest.raises(TypeError, match="field_count must be int"):
            FieldResolutionApplied(
                timestamp=datetime.now(UTC),
                run_id="r",
                source_plugin="csv",
                field_count=True,
                normalization_version="v1",
                resolution_mapping={"original": "normalized"},
            )

    def test_valid_field_resolution_accepted(self) -> None:
        from datetime import UTC, datetime

        from elspeth.contracts.events import FieldResolutionApplied

        event = FieldResolutionApplied(
            timestamp=datetime.now(UTC),
            run_id="r",
            source_plugin="csv",
            field_count=3,
            normalization_version="v1",
            resolution_mapping={"a": "b"},
        )
        assert event.field_count == 3
