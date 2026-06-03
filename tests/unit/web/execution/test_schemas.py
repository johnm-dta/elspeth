"""Tests for execution response models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

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
    RunAccounting,
    RunAccountingIntegrity,
    RunAccountingRouting,
    RunAccountingSource,
    RunAccountingTokens,
    RunEvent,
    RunResultsResponse,
    RunStatusResponse,
    ValidationCheck,
    ValidationError,
    ValidationReadiness,
    ValidationReadinessBlocker,
    ValidationResult,
)


def _accounting(
    *,
    source_rows: int = 10,
    succeeded: int = 10,
    failed: int = 0,
    structural: int = 0,
    pending: int = 0,
    emitted: int | None = None,
    closure: Literal["closed", "open", "unknown"] = "closed",
    missing_terminal_outcomes: int = 0,
    duplicate_terminal_outcomes: int = 0,
    routed_success: int = 0,
    routed_failure: int = 0,
    quarantined: int = 0,
    discarded: int = 0,
) -> RunAccounting:
    terminal = succeeded + failed + structural
    if emitted is None:
        emitted = terminal + pending
    return RunAccounting(
        source=RunAccountingSource(rows_processed=source_rows),
        tokens=RunAccountingTokens(
            emitted=emitted,
            terminal=terminal,
            succeeded=succeeded,
            failed=failed,
            structural=structural,
            pending=pending,
        ),
        routing=RunAccountingRouting(
            routed_success=routed_success,
            routed_failure=routed_failure,
            quarantined=quarantined,
            discarded=discarded,
        ),
        integrity=RunAccountingIntegrity(
            closure=closure,
            missing_terminal_outcomes=missing_terminal_outcomes,
            duplicate_terminal_outcomes=duplicate_terminal_outcomes,
        ),
    )


def _progress_data(
    *,
    source_rows_processed: int = 10,
    tokens_succeeded: int = 10,
    tokens_failed: int = 0,
    tokens_quarantined: int = 0,
    tokens_routed_success: int = 0,
    tokens_routed_failure: int = 0,
) -> ProgressData:
    return ProgressData(
        source_rows_processed=source_rows_processed,
        tokens_succeeded=tokens_succeeded,
        tokens_failed=tokens_failed,
        tokens_quarantined=tokens_quarantined,
        tokens_routed_success=tokens_routed_success,
        tokens_routed_failure=tokens_routed_failure,
    )


def _ready_readiness() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[])


def _cancelled_data(
    *,
    source_rows_processed: int = 10,
    tokens_succeeded: int = 10,
    tokens_failed: int = 0,
    tokens_quarantined: int = 0,
    tokens_routed_success: int = 0,
    tokens_routed_failure: int = 0,
) -> CancelledData:
    return CancelledData(
        source_rows_processed=source_rows_processed,
        tokens_succeeded=tokens_succeeded,
        tokens_failed=tokens_failed,
        tokens_quarantined=tokens_quarantined,
        tokens_routed_success=tokens_routed_success,
        tokens_routed_failure=tokens_routed_failure,
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
    def test_validation_result_requires_readiness(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ValidationResult(is_valid=True, checks=[], errors=[], semantic_contracts=[])

    def test_validation_readiness_accepts_pending_interpretation_blocker(self) -> None:
        result = ValidationResult(
            is_valid=False,
            checks=[],
            errors=[
                ValidationError(
                    component_id="rate_coolness",
                    component_type="transform",
                    message="Interpretation review is pending for 'coolness'.",
                    suggestion="Resolve the pending interpretation review before running.",
                    error_code="interpretation_review_pending",
                )
            ],
            readiness=ValidationReadiness(
                authoring_valid=True,
                execution_ready=False,
                completion_ready=True,
                blockers=[
                    ValidationReadinessBlocker(
                        code="interpretation_review_pending",
                        component_id="rate_coolness",
                        component_type="transform",
                        detail="coolness",
                    )
                ],
            ),
            semantic_contracts=[],
        )

        assert result.readiness.authoring_valid is True
        assert result.readiness.execution_ready is False
        assert result.readiness.blockers[0].code == "interpretation_review_pending"

    def test_secret_refs_check_requires_structured_outcome_code(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="secret_refs checks must carry"):
            ValidationCheck(
                name="secret_refs",
                passed=True,
                detail="No secret references found",
                affected_nodes=(),
                outcome_code=None,
            )

    def test_secret_refs_check_accepts_no_refs_outcome_code(self) -> None:
        check = ValidationCheck(
            name="secret_refs",
            passed=True,
            detail="No secret references found",
            affected_nodes=(),
            outcome_code="secret_refs.no_refs",
        )

        assert check.outcome_code == "secret_refs.no_refs"

    def test_non_secret_check_may_have_null_outcome_code(self) -> None:
        check = ValidationCheck(
            name="settings_load",
            passed=True,
            detail="OK",
            affected_nodes=(),
            outcome_code=None,
        )

        assert check.outcome_code is None

    def test_unknown_check_outcome_code_is_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ValidationCheck(
                name="secret_refs",
                passed=True,
                detail="No secret references found",
                affected_nodes=(),
                outcome_code="secret_refs.maybe",  # type: ignore[arg-type]
            )

    def test_invalid_result_with_attributed_error(self) -> None:
        result = ValidationResult(
            is_valid=False,
            checks=[
                ValidationCheck(name="settings_load", passed=True, detail="OK", affected_nodes=(), outcome_code=None),
                ValidationCheck(
                    name="graph_structure",
                    passed=False,
                    detail="Graph validation failed",
                    affected_nodes=(),
                    outcome_code=None,
                ),
            ],
            errors=[
                ValidationError(
                    component_id="gate_1",
                    component_type="gate",
                    message="Route destination 'nonexistent_sink' not found",
                    suggestion="Check sink names in gate configuration",
                    error_code=None,
                ),
            ],
            readiness=ValidationReadiness(
                authoring_valid=False,
                execution_ready=False,
                completion_ready=False,
                blockers=[
                    ValidationReadinessBlocker(
                        code="graph_structure",
                        component_id="gate_1",
                        component_type="gate",
                        detail="Route destination not found.",
                    )
                ],
            ),
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
            error_code=None,
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
                    affected_nodes=(),
                    outcome_code=None,
                ),
                ValidationCheck(
                    name="plugin_instantiation",
                    passed=False,
                    detail="Skipped: settings_load failed",
                    affected_nodes=(),
                    outcome_code="validation.skipped_after_failure",
                ),
                ValidationCheck(
                    name="graph_structure",
                    passed=False,
                    detail="Skipped: settings_load failed",
                    affected_nodes=(),
                    outcome_code="validation.skipped_after_failure",
                ),
                ValidationCheck(
                    name="schema_compatibility",
                    passed=False,
                    detail="Skipped: settings_load failed",
                    affected_nodes=(),
                    outcome_code="validation.skipped_after_failure",
                ),
            ],
            errors=[
                ValidationError(
                    component_id=None,
                    component_type=None,
                    message="Invalid YAML syntax",
                    suggestion=None,
                    error_code=None,
                ),
            ],
            readiness=ValidationReadiness(
                authoring_valid=False,
                execution_ready=False,
                completion_ready=False,
                blockers=[
                    ValidationReadinessBlocker(
                        code="settings_load",
                        component_id=None,
                        component_type=None,
                        detail="Invalid YAML syntax.",
                    )
                ],
            ),
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
                data=_progress_data(source_rows_processed=0, tokens_succeeded=0),
            )

    def test_progress_event_valid(self) -> None:
        event = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="progress",
            data=_progress_data(
                source_rows_processed=10,
                tokens_succeeded=4,
                tokens_failed=2,
                tokens_routed_success=3,
                tokens_routed_failure=1,
            ),
        )
        assert isinstance(event.data, ProgressData)
        assert event.data.source_rows_processed == 10
        assert event.data.tokens_succeeded == 4
        assert event.data.tokens_failed == 2
        assert event.data.tokens_quarantined == 0
        assert event.data.tokens_routed_success == 3
        assert event.data.tokens_routed_failure == 1

    def test_progress_event_carries_token_routed_split(self) -> None:
        """ProgressData keeps route subsets while naming their token unit."""
        progress_fields = set(ProgressData.model_fields)
        assert "tokens_routed_success" in progress_fields
        assert "tokens_routed_failure" in progress_fields

    def test_completed_event_valid(self) -> None:
        event = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="completed",
            data=CompletedData(
                status="completed_with_failures",
                accounting=_accounting(source_rows=100, succeeded=95, failed=3, quarantined=2),
                landscape_run_id="lscape-1",
            ),
        )
        assert isinstance(event.data, CompletedData)
        assert event.data.accounting.tokens.succeeded == 95
        assert event.data.landscape_run_id == "lscape-1"

    def test_cancelled_event_valid(self) -> None:
        event = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="cancelled",
            data=_cancelled_data(
                source_rows_processed=50,
                tokens_succeeded=45,
                tokens_failed=2,
                tokens_quarantined=2,
                tokens_routed_success=1,
                tokens_routed_failure=1,
            ),
        )
        assert isinstance(event.data, CancelledData)
        assert event.data.source_rows_processed == 50
        assert event.data.tokens_succeeded == 45
        assert event.data.tokens_quarantined == 2

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
        """CancelledData must include all six row counters — fabrication test.

        S-8: defaulting any counter to ``0`` would fabricate "we don't know"
        as "definitely zero".  Constructing with only one counter must crash
        on the five missing required fields.
        """
        with pytest.raises(pydantic.ValidationError):
            RunEvent(
                run_id="run-1",
                timestamp=datetime.now(tz=UTC),
                event_type="cancelled",
                data=CancelledData(source_rows_processed=0),  # type: ignore[call-arg]
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
            data=_progress_data(source_rows_processed=50, tokens_succeeded=47, tokens_failed=3),
        )
        restored = self._round_trip(original)
        assert restored.event_type == "progress"
        assert isinstance(restored.data, ProgressData)
        assert restored.data.source_rows_processed == 50

    def test_completed_round_trip(self) -> None:
        original = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="completed",
            data=CompletedData(
                status="completed_with_failures",
                accounting=_accounting(source_rows=100, succeeded=95, failed=3, quarantined=2),
                landscape_run_id="lscape-1",
            ),
        )
        restored = self._round_trip(original)
        assert restored.event_type == "completed"
        assert isinstance(restored.data, CompletedData)
        assert restored.data.accounting.tokens.succeeded == 95
        assert restored.data.landscape_run_id == "lscape-1"

    def test_cancelled_round_trip(self) -> None:
        """Cancelled has identical shape to Progress — round-trip must
        preserve the correct type via model_validator.
        """
        original = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="cancelled",
            data=_cancelled_data(tokens_succeeded=8, tokens_failed=1, tokens_quarantined=1),
        )
        restored = self._round_trip(original)
        assert restored.event_type == "cancelled"
        assert isinstance(restored.data, CancelledData)
        assert restored.data.tokens_succeeded == 8
        assert restored.data.tokens_quarantined == 1

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


class TestCompletedDataAccounting:
    """Completed terminal events use Landscape-derived token accounting."""

    def test_completed_with_failures_accounting_accepted(self) -> None:
        data = CompletedData(
            status="completed_with_failures",
            accounting=_accounting(source_rows=100, succeeded=95, failed=3, quarantined=2),
            landscape_run_id="lscape-1",
        )
        assert data.accounting.source.rows_processed == 100
        assert data.accounting.tokens.succeeded == 95
        assert data.accounting.tokens.failed == 3

    def test_aggregation_source_rows_and_output_tokens_are_separate(self) -> None:
        data = CompletedData(
            status="completed",
            accounting=_accounting(source_rows=6, succeeded=2),
            landscape_run_id="lscape-batchstats",
        )
        assert data.accounting.source.rows_processed == 6
        assert data.accounting.tokens.succeeded == 2

    def test_zero_counts_accepted(self) -> None:
        data = CompletedData(
            status="empty",
            accounting=_accounting(source_rows=0, succeeded=0, emitted=0),
            landscape_run_id="lscape-empty",
        )
        assert data.accounting.source.rows_processed == 0
        assert data.accounting.tokens.emitted == 0

    def test_routed_rows_report_subset_details(self) -> None:
        data = CompletedData(
            status="completed_with_failures",
            accounting=_accounting(
                source_rows=100,
                succeeded=90,
                failed=3,
                routed_success=5,
                routed_failure=0,
                quarantined=2,
            ),
            landscape_run_id="lscape-1",
        )
        assert data.accounting.routing.routed_success == 5
        assert data.accounting.routing.routed_failure == 0

    def test_one_row_routed_success_completion_accepted(self) -> None:
        data = CompletedData(
            status="completed",
            accounting=_accounting(source_rows=1, succeeded=1, routed_success=1),
            landscape_run_id="lscape-routed",
        )

        assert data.accounting.routing.routed_success == 1
        assert data.accounting.tokens.succeeded == 1

    def test_routed_success_exceeding_succeeded_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"routing\.routed_success must be a subset"):
            _accounting(source_rows=1, succeeded=0, routed_success=1)


class TestRowCountConstraints:
    """Progress/cancelled counter fields must be non-negative."""

    def test_negative_source_rows_processed_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            _progress_data(source_rows_processed=-1)

    def test_negative_tokens_failed_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            _progress_data(source_rows_processed=0, tokens_succeeded=0, tokens_failed=-1)

    def test_negative_cancelled_source_rows_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            _cancelled_data(source_rows_processed=-1, tokens_succeeded=0)

    def test_negative_completed_rows_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CompletedData(
                status="completed",
                accounting=_accounting(source_rows=-1, succeeded=0),
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
                status="completed",
                accounting=_accounting(),
                landscape_run_id="",
            )


class TestResponseModelConstraints:
    """RunStatusResponse and RunResultsResponse enforce non-negative accounting."""

    def test_status_response_rejects_negative_accounting(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            RunStatusResponse(
                run_id="r1",
                status="completed",
                started_at=None,
                finished_at=None,
                accounting=_accounting(source_rows=-1, succeeded=0),
                error=None,
                landscape_run_id="lscape-1",
            )

    def test_results_response_rejects_negative_accounting(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            RunResultsResponse(
                run_id="r1",
                status="completed",
                accounting=_accounting(source_rows=10, succeeded=0, failed=-1),
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
            ValidationCheck(name="test", passed="true", detail="ok", affected_nodes=(), outcome_code=None)  # type: ignore[arg-type]

    def test_validation_result_rejects_string_bool(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ValidationResult(is_valid="false", checks=[], errors=[], readiness=_ready_readiness())  # type: ignore[arg-type]

    def test_run_status_response_rejects_string_int(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            RunStatusResponse(
                run_id="r1",
                status="completed",
                started_at=None,
                finished_at=None,
                accounting=RunAccounting(
                    source=RunAccountingSource(rows_processed="7"),  # type: ignore[arg-type]
                    tokens=RunAccountingTokens(
                        emitted=0,
                        terminal=0,
                        succeeded=0,
                        failed=0,
                        structural=0,
                        pending=0,
                    ),
                    routing=RunAccountingRouting(
                        routed_success=0,
                        routed_failure=0,
                        quarantined=0,
                        discarded=0,
                    ),
                    integrity=RunAccountingIntegrity(
                        closure="closed",
                        missing_terminal_outcomes=0,
                        duplicate_terminal_outcomes=0,
                    ),
                ),
                error=None,
                landscape_run_id="lscape-1",
            )

    def test_run_results_response_rejects_string_int(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            RunResultsResponse(
                run_id="r1",
                status="completed",
                accounting=RunAccounting(
                    source=RunAccountingSource(rows_processed=10),
                    tokens=RunAccountingTokens(
                        emitted=12,
                        terminal=12,
                        succeeded=10,
                        failed="2",  # type: ignore[arg-type]
                        structural=0,
                        pending=0,
                    ),
                    routing=RunAccountingRouting(
                        routed_success=0,
                        routed_failure=0,
                        quarantined=0,
                        discarded=0,
                    ),
                    integrity=RunAccountingIntegrity(
                        closure="closed",
                        missing_terminal_outcomes=0,
                        duplicate_terminal_outcomes=0,
                    ),
                ),
                landscape_run_id="lscape-1",
                error=None,
            )

    def test_progress_data_rejects_string_int(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            ProgressData(
                source_rows_processed="10",  # type: ignore[arg-type]
                tokens_succeeded=0,
                tokens_failed=0,
                tokens_quarantined=0,
                tokens_routed_success=0,
                tokens_routed_failure=0,
            )

    def test_completed_data_rejects_string_int(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CompletedData(
                status="completed_with_failures",
                accounting=RunAccounting(
                    source=RunAccountingSource(rows_processed="100"),  # type: ignore[arg-type]
                    tokens=RunAccountingTokens(
                        emitted=98,
                        terminal=98,
                        succeeded=95,
                        failed=3,
                        structural=0,
                        pending=0,
                    ),
                    routing=RunAccountingRouting(
                        routed_success=0,
                        routed_failure=0,
                        quarantined=2,
                        discarded=0,
                    ),
                    integrity=RunAccountingIntegrity(
                        closure="closed",
                        missing_terminal_outcomes=0,
                        duplicate_terminal_outcomes=0,
                    ),
                ),
                landscape_run_id="lscape-1",
            )

    def test_cancelled_data_rejects_string_int(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CancelledData(
                source_rows_processed="50",  # type: ignore[arg-type]
                tokens_succeeded=45,
                tokens_failed=1,
                tokens_quarantined=2,
                tokens_routed_success=1,
                tokens_routed_failure=1,
            )

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
            ValidationCheck(name="test", passed=True, detail="ok", affected_nodes=(), outcome_code=None, severity="high")  # type: ignore[call-arg]

    def test_validation_error_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            ValidationError(
                component_id=None,
                component_type=None,
                message="bad",
                suggestion=None,
                error_code=None,
                stack_trace="...",  # type: ignore[call-arg]
            )

    def test_validation_result_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            ValidationResult(is_valid=True, checks=[], errors=[], readiness=_ready_readiness(), warnings=[])  # type: ignore[call-arg]

    def test_run_status_response_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            RunStatusResponse(
                run_id="r1",
                status="completed",
                started_at=None,
                finished_at=None,
                accounting=_accounting(),
                error=None,
                landscape_run_id="lscape-1",
                extra_field=42,  # type: ignore[call-arg]
            )

    def test_run_results_response_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            RunResultsResponse(
                run_id="r1",
                status="completed",
                accounting=_accounting(),
                landscape_run_id="lscape-1",
                error=None,
                duration_ms=1234,  # type: ignore[call-arg]
            )

    def test_progress_data_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            ProgressData(
                source_rows_processed=10,
                tokens_succeeded=10,
                tokens_failed=0,
                tokens_quarantined=0,
                tokens_routed_success=0,
                tokens_routed_failure=0,
                percent=50.0,  # type: ignore[call-arg]
            )

    def test_completed_data_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            CompletedData(
                status="completed_with_failures",
                accounting=_accounting(source_rows=100, succeeded=95, failed=3, quarantined=2),
                landscape_run_id="lscape-1",
                duration_ms=5000,  # type: ignore[call-arg]
            )

    def test_run_event_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            RunEvent(
                run_id="run-1",
                timestamp=datetime.now(tz=UTC),
                event_type="progress",
                data=_progress_data(),
                session_id="s-1",  # type: ignore[call-arg]
            )

    def test_cancelled_data_rejects_extra(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="extra"):
            CancelledData(
                source_rows_processed=10,
                tokens_succeeded=8,
                tokens_failed=0,
                tokens_quarantined=2,
                tokens_routed_success=0,
                tokens_routed_failure=0,
                reason="timeout",  # type: ignore[call-arg]
            )

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
                accounting=None,
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
                accounting=_accounting(),
                error=None,
                landscape_run_id="lscape-1",
            )


class TestRunEventTimestampCoercion:
    """RunEvent.timestamp accepts datetime and ISO strings, rejects integers."""

    def test_accepts_datetime_directly(self) -> None:
        event = RunEvent(
            run_id="run-1",
            timestamp=datetime.now(tz=UTC),
            event_type="progress",
            data=_progress_data(source_rows_processed=0, tokens_succeeded=0),
        )
        assert isinstance(event.timestamp, datetime)

    def test_accepts_iso_string_via_model_validate(self) -> None:
        """Production reconnect path: model_dump(mode='json') → model_validate."""
        raw = {
            "run_id": "run-1",
            "timestamp": "2026-04-15T10:00:00+00:00",
            "event_type": "progress",
            "data": {
                "source_rows_processed": 0,
                "tokens_succeeded": 0,
                "tokens_failed": 0,
                "tokens_quarantined": 0,
                "tokens_routed_success": 0,
                "tokens_routed_failure": 0,
            },
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
                data=_progress_data(source_rows_processed=0, tokens_succeeded=0),
            )


class TestRunStatusAccounting:
    """RunStatusResponse validates terminal statuses against token accounting."""

    def test_running_accepts_missing_accounting(self) -> None:
        resp = RunStatusResponse(
            run_id="r1",
            status="running",
            started_at=datetime.now(tz=UTC),
            finished_at=None,
            accounting=None,
            error=None,
            landscape_run_id=None,
        )
        assert resp.accounting is None

    def test_pending_accepts_missing_accounting(self) -> None:
        resp = RunStatusResponse(
            run_id="r1",
            status="pending",
            started_at=None,
            finished_at=None,
            accounting=None,
            error=None,
            landscape_run_id=None,
        )
        assert resp.accounting is None

    def test_completed_accepts_one_source_row_many_tokens(self) -> None:
        resp = RunStatusResponse(
            run_id="r1",
            status="completed",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            accounting=_accounting(source_rows=1, succeeded=9323, structural=1),
            error=None,
            landscape_run_id="lscape-fanout",
        )
        assert resp.accounting is not None
        assert resp.accounting.source.rows_processed == 1
        assert resp.accounting.tokens.emitted == 9324

    def test_completed_rejects_open_accounting(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="requires closed token accounting"):
            RunStatusResponse(
                run_id="r1",
                status="completed",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                accounting=_accounting(
                    source_rows=1,
                    succeeded=1,
                    pending=1,
                    closure="open",
                    missing_terminal_outcomes=1,
                ),
                error=None,
                landscape_run_id="lscape-1",
            )

    def test_completed_with_failures_accepts_success_and_failure_tokens(self) -> None:
        resp = RunStatusResponse(
            run_id="r1",
            status="completed_with_failures",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            accounting=_accounting(source_rows=100, succeeded=95, failed=3, quarantined=2),
            error=None,
            landscape_run_id="lscape-1",
        )
        assert resp.accounting is not None
        assert resp.accounting.tokens.failed == 3

    def test_completed_with_failures_rejects_no_failure_tokens(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"tokens\.failed > 0"):
            RunStatusResponse(
                run_id="r1",
                status="completed_with_failures",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                accounting=_accounting(source_rows=10, succeeded=10, failed=0),
                error=None,
                landscape_run_id="lscape-1",
            )

    def test_empty_accepts_zero_source_rows_and_zero_tokens(self) -> None:
        resp = RunStatusResponse(
            run_id="r1",
            status="empty",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            accounting=_accounting(source_rows=0, succeeded=0, emitted=0),
            error=None,
            landscape_run_id="lscape-empty",
        )
        assert resp.status == "empty"

    def test_empty_rejects_nonzero_source_rows(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"source\.rows_processed == 0"):
            RunStatusResponse(
                run_id="r1",
                status="empty",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                accounting=_accounting(source_rows=1, succeeded=0, emitted=0),
                error=None,
                landscape_run_id="lscape-empty",
            )

    def test_failed_status_may_omit_accounting_for_exception_origin(self) -> None:
        resp = RunStatusResponse(
            run_id="r1",
            status="failed",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            accounting=None,
            error="pipeline crashed",
            landscape_run_id=None,
        )
        assert resp.accounting is None

    def test_cancelled_status_may_omit_accounting(self) -> None:
        resp = RunStatusResponse(
            run_id="r1",
            status="cancelled",
            started_at=datetime.now(tz=UTC),
            finished_at=datetime.now(tz=UTC),
            accounting=None,
            error=None,
            landscape_run_id=None,
        )
        assert resp.accounting is None


class TestRunStatusTerminalInvariants:
    """Terminal run statuses must carry the fields the rest of the web layer assumes."""

    def test_completed_requires_landscape_run_id(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="landscape_run_id"):
            RunStatusResponse(
                run_id="r1",
                status="completed",
                started_at=datetime.now(tz=UTC),
                finished_at=datetime.now(tz=UTC),
                accounting=_accounting(source_rows=1, succeeded=1),
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
                accounting=None,
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
                accounting=None,
                error=None,
                landscape_run_id=None,
            )


class TestRunResultsAccounting:
    """RunResultsResponse uses the same terminal accounting contract."""

    def test_completed_with_failures_accounting_accepted(self) -> None:
        resp = RunResultsResponse(
            run_id="r1",
            status="completed_with_failures",
            accounting=_accounting(source_rows=100, succeeded=95, failed=3, quarantined=2),
            landscape_run_id="lscape-1",
            error=None,
        )
        assert resp.accounting is not None
        assert resp.accounting.source.rows_processed == 100

    def test_completed_accepts_one_source_row_many_tokens(self) -> None:
        resp = RunResultsResponse(
            run_id="r1",
            status="completed",
            accounting=_accounting(source_rows=1, succeeded=9323, structural=1),
            landscape_run_id="lscape-fanout",
            error=None,
        )
        assert resp.accounting is not None
        assert resp.accounting.tokens.emitted == 9324

    def test_completed_rejects_open_accounting(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="requires closed token accounting"):
            RunResultsResponse(
                run_id="r1",
                status="completed",
                accounting=_accounting(
                    source_rows=1,
                    succeeded=1,
                    pending=1,
                    closure="open",
                    missing_terminal_outcomes=1,
                ),
                landscape_run_id="lscape-1",
                error=None,
            )

    def test_failed_status_may_omit_accounting_for_exception_origin(self) -> None:
        resp = RunResultsResponse(
            run_id="r1",
            status="failed",
            accounting=None,
            landscape_run_id=None,
            error="kaboom",
        )
        assert resp.accounting is None


class TestRunResultsTerminalInvariants:
    """RunResultsResponse must enforce terminal-state semantics."""

    def test_completed_requires_landscape_run_id(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="landscape_run_id"):
            RunResultsResponse(
                run_id="r1",
                status="completed",
                accounting=_accounting(source_rows=1, succeeded=1),
                landscape_run_id=None,
                error=None,
            )

    def test_failed_requires_error(self) -> None:
        with pytest.raises(pydantic.ValidationError, match="error"):
            RunResultsResponse(
                run_id="r1",
                status="failed",
                accounting=None,
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
    """Pydantic mirrors the token-accounting status taxonomy."""

    @staticmethod
    def _build(**overrides: object) -> RunStatusResponse:
        kwargs: dict[str, object] = {
            "run_id": "run-1",
            "status": "completed",
            "started_at": datetime.now(tz=UTC),
            "finished_at": datetime.now(tz=UTC),
            "accounting": _accounting(),
            "error": None,
            "landscape_run_id": "landscape-1",
        }
        kwargs.update(overrides)
        return RunStatusResponse(**kwargs)  # type: ignore[arg-type]

    def test_completed_with_failures_legal(self) -> None:
        response = self._build(
            status="completed_with_failures",
            accounting=_accounting(source_rows=10, succeeded=7, failed=3),
        )
        assert response.status == "completed_with_failures"

    def test_empty_legal(self) -> None:
        response = self._build(
            status="empty",
            accounting=_accounting(source_rows=0, succeeded=0, emitted=0),
            landscape_run_id="landscape-empty",
        )
        assert response.status == "empty"

    def test_failed_with_zero_rows_succeeded_legal(self) -> None:
        response = self._build(
            status="failed",
            accounting=None,
            error="LLM transform raised on every row",
            landscape_run_id=None,
        )
        assert response.status == "failed"

    def test_failed_with_failure_accounting_legal(self) -> None:
        response = self._build(
            status="failed",
            accounting=_accounting(source_rows=6, succeeded=0, failed=6, routed_failure=6),
            error="LLM transform diverted every row to on_error",
        )
        assert response.status == "failed"

    def test_completed_rejects_zero_succeeded(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"completed.*tokens.succeeded > 0"):
            self._build(status="completed", accounting=_accounting(source_rows=10, succeeded=0, emitted=0))

    def test_completed_rejects_failures(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"completed.*tokens.failed == 0"):
            self._build(status="completed", accounting=_accounting(source_rows=10, succeeded=7, failed=3))

    def test_completed_with_failures_rejects_zero_succeeded(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"completed_with_failures.*tokens.succeeded > 0"):
            self._build(status="completed_with_failures", accounting=_accounting(source_rows=6, succeeded=0, failed=6))

    def test_completed_with_failures_rejects_no_failures(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"completed_with_failures.*tokens.failed > 0"):
            self._build(status="completed_with_failures", accounting=_accounting(source_rows=10, succeeded=10, failed=0))

    def test_failed_tolerates_partial_successes_for_exception_bounded_runs(self) -> None:
        response = self._build(status="failed", accounting=_accounting(source_rows=10, succeeded=5, failed=5), error="X")
        assert response.status == "failed"

    def test_empty_rejects_nonzero_processed(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"empty.*source.rows_processed == 0"):
            self._build(status="empty", accounting=_accounting(source_rows=5, succeeded=0, emitted=0), landscape_run_id="landscape-empty")


class TestRunResultsResponseStatusInvariant:
    """Same token-accounting taxonomy, but on terminal-only results."""

    @staticmethod
    def _build(**overrides: object) -> RunResultsResponse:
        kwargs: dict[str, object] = {
            "run_id": "run-1",
            "status": "completed",
            "accounting": _accounting(),
            "landscape_run_id": "landscape-1",
            "error": None,
        }
        kwargs.update(overrides)
        return RunResultsResponse(**kwargs)  # type: ignore[arg-type]

    def test_completed_with_failures_legal(self) -> None:
        response = self._build(
            status="completed_with_failures",
            accounting=_accounting(source_rows=10, succeeded=7, failed=3),
        )
        assert response.status == "completed_with_failures"

    def test_empty_legal(self) -> None:
        assert (
            self._build(
                status="empty",
                accounting=_accounting(source_rows=0, succeeded=0, emitted=0),
                landscape_run_id="landscape-empty",
            ).status
            == "empty"
        )

    def test_completed_rejects_failures(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"completed.*tokens.failed == 0"):
            self._build(status="completed", accounting=_accounting(source_rows=10, succeeded=7, failed=3))


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
        ValidationReadiness,
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
        checks=[ValidationCheck(name="semantic_contracts", passed=False, detail="failed", affected_nodes=(), outcome_code=None)],
        errors=[],
        readiness=ValidationReadiness(authoring_valid=False, execution_ready=False, completion_ready=False, blockers=[]),
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
            readiness=_ready_readiness(),
            invented_extra_field="nope",  # type: ignore[call-arg]
        )


class TestRunAccountingPublicApiFieldStability:
    """Terminal API models expose accounting, not legacy mixed-unit rows."""

    @pytest.mark.parametrize(
        "model_cls",
        [CompletedData, RunStatusResponse, RunResultsResponse],
    )
    def test_terminal_models_expose_accounting_not_legacy_rows(self, model_cls) -> None:
        properties = model_cls.model_json_schema()["properties"]
        assert "accounting" in properties, f"{model_cls.__name__} must expose accounting in its JSON schema"
        assert "rows_processed" not in properties, f"{model_cls.__name__} must not expose legacy rows_processed"
        assert "rows_succeeded" not in properties, f"{model_cls.__name__} must not expose legacy rows_succeeded"
        assert "rows_failed" not in properties, f"{model_cls.__name__} must not expose legacy rows_failed"
        assert "rows_routed" not in properties, f"{model_cls.__name__} must not expose legacy rows_routed"

    @pytest.mark.parametrize("model_cls", [ProgressData, CancelledData])
    def test_live_counter_models_still_expose_split_fields_in_json_schema(self, model_cls) -> None:
        properties = model_cls.model_json_schema()["properties"]
        assert "tokens_routed_success" in properties, f"{model_cls.__name__} must expose tokens_routed_success in its JSON schema"
        assert "tokens_routed_failure" in properties, f"{model_cls.__name__} must expose tokens_routed_failure in its JSON schema"
        assert "rows_routed" not in properties, f"{model_cls.__name__} must not expose the legacy rows_routed field"


class TestTerminalEventStatusDiscriminator:
    """Phase 2.2 propagation: SSE terminal payloads carry ``status`` so the
    frontend can render the widened taxonomy without re-implementing the
    backend's ``failure_indicator`` predicate.

    Without an explicit discriminator the frontend would have to redo the
    ``success_indicator``/``failure_indicator`` classification from row counts,
    duplicating the L0 invariant in ``_check_status_row_count_invariant`` and
    creating exactly the dual-source-of-truth drift that
    ``sessions/protocol.py:69-80`` exists to prevent.
    """

    def test_completed_data_accepts_completed_with_failures(self) -> None:
        data = CompletedData(
            status="completed_with_failures",
            accounting=_accounting(source_rows=10, succeeded=7, failed=3),
            landscape_run_id="lscape-1",
        )
        assert data.status == "completed_with_failures"

    def test_completed_data_accepts_empty(self) -> None:
        data = CompletedData(
            status="empty",
            accounting=_accounting(source_rows=0, succeeded=0, emitted=0),
            landscape_run_id="lscape-empty",
        )
        assert data.status == "empty"

    def test_completed_data_rejects_failed_status(self) -> None:
        """``CompletedData`` only accepts the operator-completion subset.

        ``failed`` is a separate event_type with its own payload (FailedData)
        — the ``Literal`` constraint on ``CompletedData.status`` keeps the
        SSE event_type and status from drifting apart.
        """
        with pytest.raises(pydantic.ValidationError):
            CompletedData(
                status="failed",  # type: ignore[arg-type]
                accounting=_accounting(source_rows=0, succeeded=0, emitted=0),
                landscape_run_id="lscape-1",
            )

    def test_completed_data_rejects_running_status(self) -> None:
        """Non-terminal statuses must not appear on a terminal SSE event."""
        with pytest.raises(pydantic.ValidationError):
            CompletedData(
                status="running",  # type: ignore[arg-type]
                accounting=_accounting(source_rows=0, succeeded=0, emitted=0),
                landscape_run_id="lscape-1",
            )

    def test_completed_data_rejects_completed_with_no_success_indicator(self) -> None:
        """status='completed' with zero success tokens is a taxonomy mismatch."""
        with pytest.raises(pydantic.ValidationError, match=r"tokens\.succeeded > 0"):
            CompletedData(
                status="completed",
                accounting=_accounting(source_rows=5, succeeded=0, emitted=0),
                landscape_run_id="lscape-1",
            )

    def test_completed_data_rejects_completed_with_failures_with_no_failure_indicator(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"tokens\.failed > 0"):
            CompletedData(
                status="completed_with_failures",
                accounting=_accounting(source_rows=10, succeeded=10, failed=0),
                landscape_run_id="lscape-1",
            )

    def test_completed_data_rejects_empty_with_nonzero_rows_processed(self) -> None:
        with pytest.raises(pydantic.ValidationError, match=r"source\.rows_processed == 0"):
            CompletedData(
                status="empty",
                accounting=_accounting(source_rows=5, succeeded=0, emitted=0),
                landscape_run_id="lscape-1",
            )

    def test_cancelled_data_status_defaults_to_cancelled(self) -> None:
        """``CancelledData.status`` has a default — existing call sites that
        don't pass it continue to work, and the wire payload is uniform with
        ``CompletedData``/``FailedData``.
        """
        data = _cancelled_data(tokens_succeeded=6, tokens_failed=2, tokens_quarantined=2)
        assert data.status == "cancelled"

    def test_cancelled_data_rejects_other_status(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CancelledData(
                status="completed",  # type: ignore[arg-type]
                source_rows_processed=10,
                tokens_succeeded=10,
                tokens_failed=0,
                tokens_quarantined=0,
                tokens_routed_success=0,
                tokens_routed_failure=0,
            )

    def test_failed_data_status_defaults_to_failed(self) -> None:
        data = FailedData(detail="Pipeline crashed", node_id=None)
        assert data.status == "failed"

    def test_failed_data_rejects_other_status(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            FailedData(status="completed", detail="x", node_id=None)  # type: ignore[arg-type]

    def test_run_event_completed_carries_status_through_round_trip(self) -> None:
        """Frontend consumes the SSE event as JSON — ``data.status`` must
        survive serialization so the React store can read it without redoing
        the row-count classification.
        """
        original = RunEvent(
            run_id="run-cwf",
            timestamp=datetime.now(tz=UTC),
            event_type="completed",
            data=CompletedData(
                status="completed_with_failures",
                accounting=_accounting(source_rows=10, succeeded=7, failed=3),
                landscape_run_id="lscape-cwf",
            ),
        )
        as_json = original.model_dump_json()
        restored = RunEvent.model_validate_json(as_json)
        assert isinstance(restored.data, CompletedData)
        assert restored.data.status == "completed_with_failures"


class TestS8FabricationGuard:
    """S-8: ProgressData and CancelledData require all six counters explicitly.

    Per CLAUDE.md fabrication test, defaulting an absent count to ``0`` makes
    "we don't know" indistinguishable from "definitely zero".  The engine's
    ``ProgressEvent`` (contracts/cli.py) already populates every counter on
    every emission; making the wire schema require them too closes the
    producer-drift door at the consumer layer.
    """

    _PROGRESS_REQUIRED_FIELDS = (
        "source_rows_processed",
        "tokens_succeeded",
        "tokens_failed",
        "tokens_quarantined",
        "tokens_routed_success",
        "tokens_routed_failure",
    )

    _CANCELLED_REQUIRED_FIELDS = _PROGRESS_REQUIRED_FIELDS

    @pytest.mark.parametrize("missing_field", _PROGRESS_REQUIRED_FIELDS)
    def test_progress_data_requires_all_six_counters(self, missing_field: str) -> None:
        """Every counter must be supplied — omitting any one crashes."""
        kwargs: dict[str, int] = dict.fromkeys(self._PROGRESS_REQUIRED_FIELDS, 1)
        del kwargs[missing_field]
        with pytest.raises(pydantic.ValidationError) as exc_info:
            ProgressData(**kwargs)  # type: ignore[arg-type]
        assert missing_field in str(exc_info.value)

    def test_progress_data_no_default_zero(self) -> None:
        """Constructing with only source_rows_processed must crash on five missing fields.

        Pre-fix: the four routed/categorical defaults silently filled in 0.
        Post-fix: only source_rows_processed is supplied; ProgressData crashes.
        """
        with pytest.raises(pydantic.ValidationError):
            ProgressData(source_rows_processed=10)  # type: ignore[call-arg]

    def test_progress_data_relaxed_sum_invariant(self) -> None:
        """Mid-flight transient inconsistency is allowed.

        ProgressData may emit ``source_rows_processed=10`` while every terminal
        bucket reads 0 — this represents a moment between row ingestion and
        categorisation.  The sum-invariant
        (succeeded+failed+quarantined+routed_success+routed_failure
        <= processed) is documented in the docstring as NOT enforced here.
        """
        data = _progress_data(source_rows_processed=10, tokens_succeeded=0)
        assert data.source_rows_processed == 10

    def test_progress_event_json_roundtrip_strict(self) -> None:
        """Reconnect-replay path: dict → RunEvent.model_validate.

        Pins the ``_resolve_data_from_event_type`` before-validator
        (schemas.py).  This path is what WebSocket reconnect deserialisation
        uses; it must accept the full six-counter shape and preserve every
        value.
        """
        payload = {
            "run_id": "r-1",
            "timestamp": "2026-05-03T12:00:00+00:00",
            "event_type": "progress",
            "data": {
                "source_rows_processed": 100,
                "tokens_succeeded": 80,
                "tokens_failed": 5,
                "tokens_quarantined": 10,
                "tokens_routed_success": 3,
                "tokens_routed_failure": 2,
            },
        }
        event = RunEvent.model_validate(payload)
        assert isinstance(event.data, ProgressData)
        assert event.data.tokens_succeeded == 80
        assert event.data.tokens_quarantined == 10
        assert event.data.tokens_routed_success == 3
        assert event.data.tokens_routed_failure == 2

    def test_progress_event_json_roundtrip_partial_rejected(self) -> None:
        """Reconnect-replay path: partial data dict must crash.

        If a buffered or stale event from the pre-fix wire shape arrives
        post-deploy, ``model_validate`` must reject it rather than silently
        substituting 0 for the missing counters.  Pins the regression door.
        """
        payload = {
            "run_id": "r-1",
            "timestamp": "2026-05-03T12:00:00+00:00",
            "event_type": "progress",
            "data": {
                "source_rows_processed": 100,
                "tokens_failed": 5,
                "tokens_routed_success": 3,
                "tokens_routed_failure": 2,
            },
        }
        with pytest.raises(pydantic.ValidationError):
            RunEvent.model_validate(payload)

    @pytest.mark.parametrize("missing_field", _CANCELLED_REQUIRED_FIELDS)
    def test_cancelled_data_requires_all_six_counters(self, missing_field: str) -> None:
        """CancelledData mirrors the ProgressData fabrication contract."""
        kwargs: dict[str, int] = dict.fromkeys(self._CANCELLED_REQUIRED_FIELDS, 1)
        del kwargs[missing_field]
        with pytest.raises(pydantic.ValidationError) as exc_info:
            CancelledData(**kwargs)  # type: ignore[arg-type]
        assert missing_field in str(exc_info.value)
