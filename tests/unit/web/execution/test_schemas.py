"""Tests for execution response models."""

from __future__ import annotations

from datetime import UTC, datetime

import pydantic
import pytest

from elspeth.web.execution.schemas import (
    RUN_STATUS_ALL_VALUES,
    RUN_STATUS_NON_TERMINAL_VALUES,
    RUN_STATUS_TERMINAL_VALUES,
    CancelledData,
    CompletedData,
    DiscardSummary,
    ErrorData,
    FailedData,
    ProgressData,
    RunEvent,
    RunResultsResponse,
    RunStatusResponse,
    ValidationCheck,
    ValidationError,
    ValidationResult,
)


class TestDiscardSummary:
    """Virtual discard sink summary is derived Tier 1 response data."""

    def test_accepts_matching_total(self) -> None:
        summary = DiscardSummary(
            total=6,
            validation_errors=1,
            transform_errors=2,
            sink_discards=3,
        )

        assert summary.total == 6

    def test_rejects_mismatched_total(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="Discard summary total mismatch"):
            DiscardSummary(
                total=5,
                validation_errors=1,
                transform_errors=2,
                sink_discards=3,
            )


class TestValidationResult:
    def test_invalid_result_with_attributed_error(self) -> None:
        result = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(name="settings_load", passed=True, detail="OK"),
                ValidationCheck(
                    name="graph_structure",
                    passed=False,
                    detail="Graph validation failed",
                ),
            ],
            errors=[
                ValidationError(
                    component_id="gate_1",
                    component_type="gate",
                    message="Route destination 'nonexistent_sink' not found",
                    suggestion="Check sink names in gate configuration",
                ),
            ],
        )
        assert result.is_valid is False
        assert result.errors[0].component_id == "gate_1"
        assert result.errors[0].component_type == "gate"

    def test_structural_error_has_null_component(self) -> None:
        err = ValidationError(
            component_id=None,
            component_type=None,
            message="Graph contains a cycle",
            suggestion=None,
        )
        assert err.component_id is None
        assert err.component_type is None

    def test_skipped_check_recorded(self) -> None:
        """When settings_load fails, downstream checks are skipped but recorded."""
        result = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(
                    name="settings_load",
                    passed=False,
                    detail="Invalid YAML syntax",
                ),
                ValidationCheck(
                    name="plugin_instantiation",
                    passed=False,
                    detail="Skipped: settings_load failed",
                ),
                ValidationCheck(
                    name="graph_structure",
                    passed=False,
                    detail="Skipped: settings_load failed",
                ),
                ValidationCheck(
                    name="schema_compatibility",
                    passed=False,
                    detail="Skipped: settings_load failed",
                ),
            ],
            errors=[
                ValidationError(
                    component_id=None,
                    component_type=None,
                    message="Invalid YAML syntax",
                    suggestion=None,
                ),
            ],
        )
        assert result.is_valid is False
        skipped = [c for c in result.checks if "Skipped" in c.detail]
        assert len(skipped) == 3


class TestRunEvent:
    def test_invalid_event_type_rejected(self) -> None:
        """event_type is a Literal — Pydantic rejects unknown values."""
        with pytest.raises(pydantic.ValidationError):
            RunEvent(
                run_id="run-123",
                timestamp=datetime.now(tz=UTC),
                event_type="unknown",  # type: ignore[arg-type]  # deliberate bad value for Pydantic to reject
                data=ProgressData(rows_processed=0, rows_failed=0),
            )

    def test_progress_event_valid(self) -> None:
        event = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="progress",
            data=ProgressData(rows_processed=10, rows_failed=2),
        )
        assert isinstance(event.data, ProgressData)
        assert event.data.rows_processed == 10
        assert event.data.rows_failed == 2

    def test_completed_event_valid(self) -> None:
        event = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="completed",
            data=CompletedData(
                rows_processed=100,
                rows_succeeded=95,
                rows_failed=3,
                rows_quarantined=2,
                landscape_run_id="lscape-1",
            ),
        )
        assert isinstance(event.data, CompletedData)
        assert event.data.rows_succeeded == 95
        assert event.data.landscape_run_id == "lscape-1"

    def test_cancelled_event_valid(self) -> None:
        event = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="cancelled",
            data=CancelledData(rows_processed=50, rows_failed=1),
        )
        assert isinstance(event.data, CancelledData)
        assert event.data.rows_processed == 50

    def test_failed_event_valid(self) -> None:
        event = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="failed",
            data=FailedData(detail="Pipeline crashed", node_id=None),
        )
        assert isinstance(event.data, FailedData)
        assert event.data.detail == "Pipeline crashed"

    def test_error_event_valid(self) -> None:
        event = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="error",
            data=ErrorData(message="Row parse failure", node_id="csv_source", row_id="row-42"),
        )
        assert isinstance(event.data, ErrorData)
        assert event.data.message == "Row parse failure"

    def test_mismatched_event_type_and_data_rejected(self) -> None:
        """event_type='progress' with FailedData must crash — offensive programming."""
        with pytest.raises(pydantic.ValidationError, match="requires ProgressData"):
            RunEvent(
                run_id="run-1",
                timestamp=datetime.now(tz=UTC),
                event_type="progress",
                data=FailedData(detail="wrong type", node_id=None),
            )

    def test_empty_dict_data_rejected(self) -> None:
        """Regression: data={} was accepted under the old untyped schema."""
        with pytest.raises(pydantic.ValidationError):
            RunEvent(
                run_id="run-1",
                timestamp=datetime.now(tz=UTC),
                event_type="cancelled",
                data={},  # type: ignore[arg-type]  # deliberate bad value for Pydantic to reject
            )

    def test_cancelled_requires_row_counts(self) -> None:
        """CancelledData must include rows_processed and rows_failed."""
        with pytest.raises(pydantic.ValidationError):
            RunEvent(
                run_id="run-1",
                timestamp=datetime.now(tz=UTC),
                event_type="cancelled",
                data=CancelledData(rows_processed=0),  # type: ignore[call-arg]
            )


class TestRunEventJsonRoundTrip:
    """Verify model_dump(mode='json') → model_validate round-trip.

    The production WebSocket path serializes via model_dump(mode='json')
    and the reconnect path constructs through model_validate. Both must
    produce identical results.
    """

    def _round_trip(self, event: RunEvent) -> RunEvent:
        json_dict = event.model_dump(mode="json")
        return RunEvent.model_validate(json_dict)

    def test_progress_round_trip(self) -> None:
        original = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="progress",
            data=ProgressData(rows_processed=50, rows_failed=3),
        )
        restored = self._round_trip(original)
        assert restored.event_type == "progress"
        assert isinstance(restored.data, ProgressData)
        assert restored.data.rows_processed == 50

    def test_completed_round_trip(self) -> None:
        original = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="completed",
            data=CompletedData(
                rows_processed=100,
                rows_succeeded=95,
                rows_failed=3,
                rows_quarantined=2,
                landscape_run_id="lscape-1",
            ),
        )
        restored = self._round_trip(original)
        assert restored.event_type == "completed"
        assert isinstance(restored.data, CompletedData)
        assert restored.data.rows_succeeded == 95
        assert restored.data.landscape_run_id == "lscape-1"

    def test_cancelled_round_trip(self) -> None:
        """Cancelled has identical shape to Progress — round-trip must
        preserve the correct type via model_validator.
        """
        original = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="cancelled",
            data=CancelledData(rows_processed=10, rows_failed=1),
        )
        restored = self._round_trip(original)
        assert restored.event_type == "cancelled"
        assert isinstance(restored.data, CancelledData)

    def test_failed_round_trip(self) -> None:
        original = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="failed",
            data=FailedData(detail="kaboom", node_id=None),
        )
        restored = self._round_trip(original)
        assert isinstance(restored.data, FailedData)
        assert restored.data.detail == "kaboom"


class TestCompletedDataDecomposition:
    """Enforce rows_processed >= succeeded + failed + routed + quarantined.

    Narrow invariant per elspeth-31d53c7493 — full DAG-aware balance is
    tracked in elspeth-cf84eb1b52 (P0).  Over-counting (sum > processed)
    is still a Tier 1 anomaly; under-counting (sum < processed) is the
    legitimate signature of an aggregation pipeline that absorbed source
    rows into CONSUMED_IN_BATCH.
    """

    def test_consistent_counts_accepted(self) -> None:
        data = CompletedData(
            rows_processed=100,
            rows_succeeded=95,
            rows_failed=3,
            rows_quarantined=2,
            landscape_run_id="lscape-1",
        )
        assert data.rows_processed == 100

    def test_overcounted_terminals_rejected(self) -> None:
        """Sum of terminal counts must not exceed rows_processed (over-counting bug)."""
        with pytest.raises(pydantic.ValidationError, match="decomposition mismatch"):
            CompletedData(
                rows_processed=100,
                rows_succeeded=95,
                rows_failed=5,
                rows_routed=0,
                rows_quarantined=3,  # 95 + 5 + 0 + 3 = 103 > 100
                landscape_run_id="lscape-1",
            )

    def test_aggregation_undercount_accepted(self) -> None:
        """S2 reproducer (run id 44f52421-a379-459b-96a8-6f0656086f16, 2026-05-01 eval).

        csv 6 source rows -> batch_stats(group_by=customer_tier) -> json sink emits
        2 aggregation outputs.  Source rows reach CONSUMED_IN_BATCH (no counter)
        so rows_processed=6 but sum-of-four-buckets=2.  Equality formulation
        rejected this legitimate shape, breaking pipeline_done_callback.
        Inequality formulation accepts it.  Full DAG-aware balance equation
        in elspeth-cf84eb1b52.
        """
        data = CompletedData(
            rows_processed=6,
            rows_succeeded=2,
            rows_failed=0,
            rows_routed=0,
            rows_quarantined=0,
            landscape_run_id="lscape-batchstats",
        )
        assert data.rows_processed == 6
        assert data.rows_succeeded == 2

    def test_zero_counts_accepted(self) -> None:
        data = CompletedData(
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            rows_quarantined=0,
            landscape_run_id="lscape-empty",
        )
        assert data.rows_processed == 0

    def test_routed_rows_participate_in_decomposition(self) -> None:
        data = CompletedData(
            rows_processed=100,
            rows_succeeded=90,
            rows_failed=3,
            rows_routed=5,
            rows_quarantined=2,
            landscape_run_id="lscape-1",
        )
        assert data.rows_routed == 5


class TestRowCountConstraints:
    """Row count fields must be non-negative."""

    def test_negative_rows_processed_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ProgressData(rows_processed=-1, rows_failed=0)

    def test_negative_rows_failed_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ProgressData(rows_processed=0, rows_failed=-1)

    def test_negative_cancelled_rows_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CancelledData(rows_processed=-1, rows_failed=0)

    def test_negative_completed_rows_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CompletedData(
                rows_processed=-1,
                rows_succeeded=0,
                rows_failed=0,
                rows_quarantined=0,
                landscape_run_id="lscape-1",
            )


class TestFailedDataConstraints:
    """FailedData.detail must be non-empty."""

    def test_empty_detail_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="string_too_short"):
            FailedData(detail="", node_id=None)

    def test_nonempty_detail_accepted(self) -> None:
        data = FailedData(detail="Pipeline crashed", node_id=None)
        assert data.detail == "Pipeline crashed"


class TestErrorDataConstraints:
    """ErrorData.message must be non-empty (parity with FailedData.detail)."""

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="string_too_short"):
            ErrorData(message="", node_id=None, row_id=None)

    def test_nonempty_message_accepted(self) -> None:
        data = ErrorData(message="Row parse failure", node_id="src", row_id="r1")
        assert data.message == "Row parse failure"


class TestCompletedDataLandscapeRunId:
    """CompletedData.landscape_run_id must be non-empty."""

    def test_empty_landscape_run_id_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="string_too_short"):
            CompletedData(
                rows_processed=10,
                rows_succeeded=10,
                rows_failed=0,
                rows_quarantined=0,
                landscape_run_id="",
            )


class TestResponseModelConstraints:
    """RunStatusResponse and RunResultsResponse enforce non-negative row counts."""

    def test_status_response_rejects_negative_rows(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            RunStatusResponse(
                run_id="r1",
                status="completed",
                started_at=None,
                finished_at=None,
                rows_processed=-1,
                rows_succeeded=0,
                rows_failed=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            )

    def test_results_response_rejects_negative_rows(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            RunResultsResponse(
                run_id="r1",
                status="completed",
                rows_processed=10,
                rows_succeeded=10,
                rows_failed=-1,
                rows_quarantined=0,
                landscape_run_id=None,
                error=None,
            )


# ── Tier 1 strictness regression tests ───────────────────────────────
#
# All execution response models serialize system-owned data (Tier 1).
# Coercion and extra fields must be rejected — silent normalization
# hides bugs and violates the Data Manifesto.


class TestStrictCoercionRejected:
    """String-to-int and string-to-bool coercion must crash, not silently convert."""

    def test_validation_check_rejects_string_bool(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ValidationCheck(name="test", passed="true", detail="ok")  # type: ignore[arg-type]

    def test_validation_result_rejects_string_bool(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ValidationResult(is_valid="false", checks=[], errors=[])  # type: ignore[arg-type]

    def test_run_status_response_rejects_string_int(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            RunStatusResponse(
                run_id="r1",
                status="completed",
                started_at=None,
                finished_at=None,
                rows_processed="7",  # type: ignore[arg-type]
                rows_succeeded=0,
                rows_failed=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            )

    def test_run_results_response_rejects_string_int(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            RunResultsResponse(
                run_id="r1",
                status="completed",
                rows_processed=10,
                rows_succeeded=10,
                rows_failed="2",  # type: ignore[arg-type]
                rows_quarantined=0,
                landscape_run_id=None,
                error=None,
            )

    def test_progress_data_rejects_string_int(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ProgressData(rows_processed="10", rows_failed=0)  # type: ignore[arg-type]

    def test_completed_data_rejects_string_int(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CompletedData(
                rows_processed="100",  # type: ignore[arg-type]
                rows_succeeded=95,
                rows_failed=3,
                rows_quarantined=2,
                landscape_run_id="lscape-1",
            )

    def test_cancelled_data_rejects_string_int(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CancelledData(rows_processed="50", rows_failed=1)  # type: ignore[arg-type]

    def test_error_data_rejects_int_as_string(self) -> None:
        """node_id is str|None — an int should not be coerced to str."""
        with pytest.raises(pydantic.ValidationError):
            ErrorData(message="fail", node_id=42, row_id=None)  # type: ignore[arg-type]

    def test_failed_data_rejects_int_as_string(self) -> None:
        """node_id is str|None — an int should not be coerced to str."""
        with pytest.raises(pydantic.ValidationError):
            FailedData(detail="crash", node_id=42)  # type: ignore[arg-type]


class TestExtraFieldsRejected:
    """Extra fields must raise, not be silently dropped."""

    def test_validation_check_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            ValidationCheck(name="test", passed=True, detail="ok", severity="high")  # type: ignore[call-arg]

    def test_validation_error_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            ValidationError(
                component_id=None,
                component_type=None,
                message="bad",
                suggestion=None,
                stack_trace="...",  # type: ignore[call-arg]
            )

    def test_validation_result_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            ValidationResult(is_valid=True, checks=[], errors=[], warnings=[])  # type: ignore[call-arg]

    def test_run_status_response_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            RunStatusResponse(
                run_id="r1",
                status="completed",
                started_at=None,
                finished_at=None,
                rows_processed=10,
                rows_succeeded=10,
                rows_failed=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
                extra_field=42,  # type: ignore[call-arg]
            )

    def test_run_results_response_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            RunResultsResponse(
                run_id="r1",
                status="completed",
                rows_processed=10,
                rows_succeeded=10,
                rows_failed=0,
                rows_quarantined=0,
                landscape_run_id=None,
                error=None,
                duration_ms=1234,  # type: ignore[call-arg]
            )

    def test_progress_data_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            ProgressData(rows_processed=10, rows_failed=0, percent=50.0)  # type: ignore[call-arg]

    def test_completed_data_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            CompletedData(
                rows_processed=100,
                rows_succeeded=95,
                rows_failed=3,
                rows_quarantined=2,
                landscape_run_id="lscape-1",
                duration_ms=5000,  # type: ignore[call-arg]
            )

    def test_run_event_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            RunEvent(
                run_id="run-1",
                timestamp=datetime.now(tz=UTC),
                event_type="progress",
                data=ProgressData(rows_processed=10, rows_failed=0),
                session_id="s-1",  # type: ignore[call-arg]
            )

    def test_cancelled_data_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            CancelledData(rows_processed=10, rows_failed=0, reason="timeout")  # type: ignore[call-arg]

    def test_failed_data_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            FailedData(detail="crash", node_id=None, stack_trace="...")  # type: ignore[call-arg]

    def test_error_data_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            ErrorData(message="fail", node_id=None, row_id=None, severity="high")  # type: ignore[call-arg]


class TestRunStatusResponseDatetimeStrict:
    """RunStatusResponse datetime fields reject string coercion (no JSON round-trip path)."""

    def test_started_at_rejects_iso_string(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            RunStatusResponse(
                run_id="r1",
                status="running",
                started_at="2026-04-15T10:00:00+00:00",  # type: ignore[arg-type]
                finished_at=None,
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            )

    def test_finished_at_rejects_iso_string(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            RunStatusResponse(
                run_id="r1",
                status="completed",
                started_at=datetime.now(tz=UTC),
                finished_at="2026-04-15T10:05:00+00:00",  # type: ignore[arg-type]
                rows_processed=10,
                rows_succeeded=10,
                rows_failed=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            )


class TestRunEventTimestampCoercion:
    """RunEvent.timestamp accepts datetime and ISO strings, rejects integers."""

    def test_accepts_datetime_directly(self) -> None:
        event = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="progress",
            data=ProgressData(rows_processed=0, rows_failed=0),
        )
        assert isinstance(event.timestamp, datetime)

    def test_accepts_iso_string_via_model_validate(self) -> None:
        """Production reconnect path: model_dump(mode='json') → model_validate."""
        raw = {
            "run_id": "run-1",
            "timestamp": "2026-04-15T10:00:00+00:00",
            "event_type": "progress",
            "data": {"rows_processed": 0, "rows_failed": 0},
        }
        event = RunEvent.model_validate(raw)
        assert isinstance(event.timestamp, datetime)

    def test_rejects_unix_epoch_integer(self) -> None:
        """Unix epoch integers must NOT be silently coerced to datetime."""
        with pytest.raises(pydantic.ValidationError, match="timestamp"):
            RunEvent(
                run_id="run-1",
                timestamp=1713254400,
                event_type="progress",
                data=ProgressData(rows_processed=0, rows_failed=0),
            )


class TestRunStatusDecomposition:
    """RunStatusResponse enforces row count decomposition ONLY on terminal states.

    Non-terminal states (pending/running) may have transiently inconsistent
    counts while the orchestrator is mid-flight.  Terminal states must
    decompose cleanly — a mismatch is a Tier 1 anomaly.
    """

    def test_running_accepts_inconsistent_counts(self) -> None:
        """In-flight counts can be transiently inconsistent."""
        resp = RunStatusResponse(
            run_id="r1",
            status="running",
            started_at=datetime.now(tz=UTC),
            finished_at=None,
            rows_processed=5,
            rows_succeeded=2,
            rows_failed=1,
            rows_quarantined=0,  # 2 + 1 + 0 = 3 != 5, but OK while running
            error=None,
            landscape_run_id=None,
        )
        assert resp.rows_processed == 5

    def test_pending_accepts_inconsistent_counts(self) -> None:
        resp = RunStatusResponse(
            run_id="r1",
            status="pending",
            started_at=None,
            finished_at=None,
            rows_processed=10,
            rows_succeeded=0,
            rows_failed=0,
            rows_quarantined=0,  # 0 != 10, but pending means nothing resolved
            error=None,
            landscape_run_id=None,
        )
        assert resp.rows_processed == 10

    def test_completed_rejects_overcounted_terminals(self) -> None:
        """Sum of terminal counts > rows_processed is still a Tier 1 anomaly."""
        with pytest.raises(pydantic.ValidationError, match="decomposition mismatch"):
            RunStatusResponse(
                run_id="r1",
                status="completed",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                rows_processed=100,
                rows_succeeded=95,
                rows_failed=5,
                rows_routed=0,
                rows_quarantined=3,  # 95 + 5 + 0 + 3 = 103 > 100
                error=None,
                landscape_run_id="lscape-1",
            )

    def test_completed_aggregation_undercount_accepted(self) -> None:
        """Aggregation pipelines legitimately undercount under the narrow
        invariant (elspeth-31d53c7493).  S2 reproducer shape via the
        terminal-status path.  Full balance in elspeth-cf84eb1b52.
        """
        resp = RunStatusResponse(
            run_id="r1",
            status="completed",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            rows_processed=6,
            rows_succeeded=2,
            rows_failed=0,
            rows_routed=0,
            rows_quarantined=0,
            error=None,
            landscape_run_id="lscape-batchstats",
        )
        assert resp.rows_processed == 6
        assert resp.rows_succeeded == 2

    def test_failed_rejects_overcounted_terminals(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="decomposition mismatch"):
            RunStatusResponse(
                run_id="r1",
                status="failed",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                rows_processed=10,
                rows_succeeded=8,
                rows_failed=5,
                rows_routed=0,
                rows_quarantined=0,  # 8 + 5 = 13 > 10
                error="pipeline crashed",
                landscape_run_id=None,
            )

    def test_cancelled_rejects_overcounted_terminals(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="decomposition mismatch"):
            RunStatusResponse(
                run_id="r1",
                status="cancelled",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                rows_processed=50,
                rows_succeeded=30,
                rows_failed=10,
                rows_routed=20,
                rows_quarantined=5,  # 30 + 10 + 20 + 5 = 65 > 50
                error=None,
                landscape_run_id=None,
            )

    def test_completed_accepts_consistent_counts(self) -> None:
        # Phase 2.2: this shape (rows_failed=3, rows_quarantined=2 alongside
        # rows_succeeded=95) is now `completed_with_failures` per the
        # presence-indicator predicate.  The test still pins the
        # row-count decomposition inequality at the API surface.
        resp = RunStatusResponse(
            run_id="r1",
            status="completed_with_failures",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            rows_processed=100,
            rows_succeeded=95,
            rows_failed=3,
            rows_routed=0,
            rows_quarantined=2,
            error=None,
            landscape_run_id="lscape-1",
        )
        assert resp.rows_processed == 100

    def test_completed_accepts_routed_rows_in_terminal_counts(self) -> None:
        # Phase 2.2: shape with failures => `completed_with_failures`.
        resp = RunStatusResponse(
            run_id="r1",
            status="completed_with_failures",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            rows_processed=100,
            rows_succeeded=90,
            rows_failed=3,
            rows_routed=5,
            rows_quarantined=2,
            error=None,
            landscape_run_id="lscape-1",
        )
        assert resp.rows_routed == 5


class TestRunStatusTerminalInvariants:
    """Terminal run statuses must carry the fields the rest of the web layer assumes."""

    def test_completed_requires_landscape_run_id(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="landscape_run_id"):
            RunStatusResponse(
                run_id="r1",
                status="completed",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                rows_processed=1,
                rows_succeeded=1,
                rows_failed=0,
                rows_routed=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            )

    def test_failed_requires_error(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="error"):
            RunStatusResponse(
                run_id="r1",
                status="failed",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                rows_processed=1,
                rows_succeeded=0,
                rows_failed=1,
                rows_routed=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            )

    def test_terminal_status_requires_finished_at(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="finished_at"):
            RunStatusResponse(
                run_id="r1",
                status="cancelled",
                started_at=datetime.now(tz=UTC),
                finished_at=None,
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_routed=0,
                rows_quarantined=0,
                error=None,
                landscape_run_id=None,
            )


class TestRunResultsDecomposition:
    """RunResultsResponse enforces row count decomposition unconditionally.

    The Literal restricts status to terminal values, so the invariant
    always applies — parity with CompletedData.
    """

    def test_consistent_counts_accepted(self) -> None:
        # Phase 2.2: shape with failures => `completed_with_failures`.
        resp = RunResultsResponse(
            run_id="r1",
            status="completed_with_failures",
            rows_processed=100,
            rows_succeeded=95,
            rows_failed=3,
            rows_routed=0,
            rows_quarantined=2,
            landscape_run_id="lscape-1",
            error=None,
        )
        assert resp.rows_processed == 100

    def test_overcounted_terminals_rejected(self) -> None:
        """Sum of terminal counts > rows_processed is over-counting (Tier 1 bug)."""
        with pytest.raises(pydantic.ValidationError, match="decomposition mismatch"):
            RunResultsResponse(
                run_id="r1",
                status="completed",
                rows_processed=100,
                rows_succeeded=50,
                rows_failed=20,
                rows_routed=40,
                rows_quarantined=10,  # 50 + 20 + 40 + 10 = 120 > 100
                landscape_run_id="lscape-1",
                error=None,
            )

    def test_aggregation_undercount_accepted(self) -> None:
        """S2 reproducer (2026-05-01 eval) via results endpoint.  Aggregation
        pipelines legitimately have rows_processed > sum-of-terminal-counts
        because CONSUMED_IN_BATCH source-row outcomes increment no counter
        (engine/orchestrator/outcomes.py:194).  Full balance in elspeth-cf84eb1b52.
        """
        resp = RunResultsResponse(
            run_id="r1",
            status="completed",
            rows_processed=6,
            rows_succeeded=2,
            rows_failed=0,
            rows_routed=0,
            rows_quarantined=0,
            landscape_run_id="lscape-batchstats",
            error=None,
        )
        assert resp.rows_processed == 6
        assert resp.rows_succeeded == 2

    def test_failed_status_overcounted_rejected(self) -> None:
        """Unconditional check: any terminal status triggers validation."""
        with pytest.raises(pydantic.ValidationError, match="decomposition mismatch"):
            RunResultsResponse(
                run_id="r1",
                status="failed",
                rows_processed=10,
                rows_succeeded=8,
                rows_failed=5,
                rows_routed=0,
                rows_quarantined=0,  # 8 + 5 = 13 > 10
                landscape_run_id=None,
                error="kaboom",
            )

    def test_consistent_counts_accept_routed_rows(self) -> None:
        # Phase 2.2: shape with failures => `completed_with_failures`.
        resp = RunResultsResponse(
            run_id="r1",
            status="completed_with_failures",
            rows_processed=100,
            rows_succeeded=90,
            rows_failed=3,
            rows_routed=5,
            rows_quarantined=2,
            landscape_run_id="lscape-1",
            error=None,
        )
        assert resp.rows_routed == 5


class TestRunResultsTerminalInvariants:
    """RunResultsResponse must enforce terminal-state semantics."""

    def test_completed_requires_landscape_run_id(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="landscape_run_id"):
            RunResultsResponse(
                run_id="r1",
                status="completed",
                rows_processed=1,
                rows_succeeded=1,
                rows_failed=0,
                rows_routed=0,
                rows_quarantined=0,
                landscape_run_id=None,
                error=None,
            )

    def test_failed_requires_error(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="error"):
            RunResultsResponse(
                run_id="r1",
                status="failed",
                rows_processed=1,
                rows_succeeded=0,
                rows_failed=1,
                rows_routed=0,
                rows_quarantined=0,
                landscape_run_id=None,
                error=None,
            )


class TestRunStatusDerivedSets:
    """Sets derived from Literal annotations — guards against drift."""

    def test_terminal_is_subset_of_all(self) -> None:
        assert RUN_STATUS_TERMINAL_VALUES.issubset(RUN_STATUS_ALL_VALUES)

    def test_non_terminal_is_complement(self) -> None:
        assert RUN_STATUS_NON_TERMINAL_VALUES == (RUN_STATUS_ALL_VALUES - RUN_STATUS_TERMINAL_VALUES)

    def test_non_terminal_matches_hardcoded_expected(self) -> None:
        """Pinning the current contract: pending/running are non-terminal.

        If a maintainer adds a new non-terminal status (e.g., "paused"),
        this test fails loudly — forcing a deliberate review of all
        downstream consumers of the /results 409 guard.
        """
        assert frozenset({"pending", "running"}) == RUN_STATUS_NON_TERMINAL_VALUES

    def test_terminal_matches_hardcoded_expected(self) -> None:
        # Phase 2.2 (elspeth-0de989c56d): four-value terminal taxonomy.
        # `completed_with_failures` and `empty` join the previous three so
        # operators reading /api/runs/{rid} can distinguish "ran cleanly"
        # from "ran but no row succeeded" without opening output files.
        assert frozenset({"completed", "completed_with_failures", "failed", "empty", "cancelled"}) == RUN_STATUS_TERMINAL_VALUES

    def test_non_terminal_is_nonempty(self) -> None:
        """The /results 409 guard depends on this set being non-empty."""
        assert RUN_STATUS_NON_TERMINAL_VALUES


class TestRunStatusResponseStatusInvariant:
    """Phase 2.2 (elspeth-0de989c56d) — Pydantic mirror of the L0 biconditional.

    The same presence-indicator predicate that gates ``RunResult.__post_init__``
    must gate the API surface so neither layer can serialize an inconsistent
    status / row-count pair.  Mirrors the Phase 2.3 precedent
    (``SecretInventoryResponse._check_reason_invariant``).
    """

    @staticmethod
    def _build(**overrides: object) -> RunStatusResponse:
        kwargs: dict[str, object] = {
            "run_id": "run-1",
            "status": "completed",
            "started_at": datetime.now(tz=UTC),
            "finished_at": datetime.now(tz=UTC),
            "rows_processed": 10,
            "rows_succeeded": 10,
            "rows_failed": 0,
            "rows_routed": 0,
            "rows_quarantined": 0,
            "error": None,
            "landscape_run_id": "landscape-1",
        }
        kwargs.update(overrides)
        return RunStatusResponse(**kwargs)  # type: ignore[arg-type]

    def test_completed_with_failures_legal(self) -> None:
        response = self._build(
            status="completed_with_failures",
            rows_processed=10,
            rows_succeeded=7,
            rows_failed=3,
        )
        assert response.status == "completed_with_failures"

    def test_empty_legal(self) -> None:
        response = self._build(
            status="empty",
            rows_processed=0,
            rows_succeeded=0,
        )
        assert response.status == "empty"

    def test_failed_with_zero_rows_succeeded_legal(self) -> None:
        """S1B msg2 reproducer at the API layer."""
        response = self._build(
            status="failed",
            rows_processed=6,
            rows_succeeded=0,
            rows_failed=6,
            error="LLM transform raised on every row",
        )
        assert response.status == "failed"

    def test_failed_s1a_reproducer_legal(self) -> None:
        """S1A reproducer at the API layer (rows_routed-only failure shape)."""
        response = self._build(
            status="failed",
            rows_processed=6,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed=6,
            error="LLM transform diverted every row to on_error",
        )
        assert response.status == "failed"

    def test_completed_rejects_zero_succeeded(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"completed.*rows_succeeded > 0"):
            self._build(status="completed", rows_processed=10, rows_succeeded=0, rows_failed=10, error="X")

    def test_completed_rejects_failures(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"completed.*requires no failures"):
            self._build(status="completed", rows_processed=10, rows_succeeded=7, rows_failed=3)

    def test_completed_with_failures_rejects_zero_succeeded(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"completed_with_failures.*rows_succeeded > 0"):
            self._build(status="completed_with_failures", rows_processed=6, rows_succeeded=0, rows_failed=6, error="X")

    def test_completed_with_failures_rejects_no_failures(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"completed_with_failures.*requires.*failure"):
            self._build(status="completed_with_failures", rows_processed=10, rows_succeeded=10, rows_failed=0)

    def test_failed_tolerates_partial_successes_for_exception_bounded_runs(self) -> None:
        # See ``_check_status_row_count_invariant`` — `failed` has two
        # origins (predicate-decided no-success or exception-bounded with
        # partial successes).  The API mirror tolerates either; the
        # predicate picks COMPLETED_WITH_FAILURES on the success path
        # whenever rows_succeeded > 0 alongside failures.
        response = self._build(status="failed", rows_processed=10, rows_succeeded=5, rows_failed=5, error="X")
        assert response.status == "failed"

    def test_empty_rejects_nonzero_processed(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"empty.*rows_processed == 0"):
            self._build(status="empty", rows_processed=5, rows_succeeded=0)

    # NOTE: `empty` + `has_failures` (rows_failed > 0 / rows_quarantined > 0)
    # is unreachable at the API layer because `_validate_row_decomposition`
    # fires first — `rows_processed=0` with any non-zero failure counter
    # violates the inequality `rows_processed >= sum_terminal_counts`.  The
    # L0 contract layer catches this case directly without the inequality
    # in the way; see ``tests/unit/contracts/test_run_result.py::
    # TestRunResultStatusInvariant::test_empty_rejects_failures`` for the
    # primary coverage.


class TestRunResultsResponseStatusInvariant:
    """Same biconditional, but on the terminal-only ``RunResultsResponse``."""

    @staticmethod
    def _build(**overrides: object) -> RunResultsResponse:
        kwargs: dict[str, object] = {
            "run_id": "run-1",
            "status": "completed",
            "rows_processed": 10,
            "rows_succeeded": 10,
            "rows_failed": 0,
            "rows_routed": 0,
            "rows_quarantined": 0,
            "landscape_run_id": "landscape-1",
            "error": None,
        }
        kwargs.update(overrides)
        return RunResultsResponse(**kwargs)  # type: ignore[arg-type]

    def test_completed_with_failures_legal(self) -> None:
        response = self._build(
            status="completed_with_failures",
            rows_processed=10,
            rows_succeeded=7,
            rows_failed=3,
        )
        assert response.status == "completed_with_failures"

    def test_empty_legal(self) -> None:
        assert self._build(status="empty", rows_processed=0, rows_succeeded=0).status == "empty"

    def test_completed_rejects_failures(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"completed.*requires no failures"):
            self._build(status="completed", rows_processed=10, rows_succeeded=7, rows_failed=3)


class TestErrorEventRoundTrip:
    """Round-trip coverage for the error event type (no backend producer yet)."""

    def test_error_round_trip(self) -> None:
        original = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="error",
            data=ErrorData(message="Row parse failure", node_id="csv_source", row_id="row-42"),
        )
        json_dict = original.model_dump(mode="json")
        restored = RunEvent.model_validate(json_dict)
        assert restored.event_type == "error"
        assert isinstance(restored.data, ErrorData)
        assert restored.data.message == "Row parse failure"
        assert restored.data.node_id == "csv_source"
        assert isinstance(restored.timestamp, datetime)


def test_validation_result_accepts_semantic_contracts():
    from elspeth.web.execution.schemas import (
        SemanticEdgeContractResponse,
        ValidationCheck,
        ValidationResult,
    )

    contract = SemanticEdgeContractResponse(
        from_id="scrape",
        to_id="explode",
        consumer_plugin="line_explode",
        producer_plugin="web_scrape",
        producer_field="content",
        consumer_field="content",
        outcome="conflict",
        requirement_code="line_explode.source_field.line_framed_text",
    )
    result = ValidationResult(
        is_valid=False,
        checks=[ValidationCheck(name="semantic_contracts", passed=False, detail="failed")],
        errors=[],
        semantic_contracts=[contract],
    )
    payload = result.model_dump()
    assert payload["semantic_contracts"][0]["outcome"] == "conflict"
    assert payload["semantic_contracts"][0]["consumer_plugin"] == "line_explode"


def test_validation_result_rejects_unknown_field():
    # Confirms extra="forbid" still applies — the new field doesn't
    # accidentally weaken strict-mode enforcement.
    from pydantic import ValidationError as PydanticValidationError

    from elspeth.web.execution.schemas import ValidationResult

    with pytest.raises(PydanticValidationError):
        ValidationResult(
            is_valid=True,
            checks=[],
            errors=[],
            invented_extra_field="nope",  # type: ignore[call-arg]
        )
