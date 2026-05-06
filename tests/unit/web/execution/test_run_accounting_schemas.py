from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from elspeth.web.execution.schemas import (
    CompletedData,
    ProgressData,
    RunAccounting,
    RunAccountingIntegrity,
    RunAccountingRouting,
    RunAccountingSource,
    RunAccountingTokens,
    RunResultsResponse,
    RunStatusResponse,
)


def _fanout_accounting() -> RunAccounting:
    return RunAccounting(
        source=RunAccountingSource(rows_processed=1),
        tokens=RunAccountingTokens(
            emitted=9324,
            terminal=9324,
            succeeded=9323,
            failed=0,
            structural=1,
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
    )


def test_run_status_accepts_one_source_row_many_terminal_tokens() -> None:
    response = RunStatusResponse(
        run_id="a2a7354a-5732-475b-a4ac-ed166a9e0f25",
        status="completed",
        started_at=datetime(2026, 5, 6, 14, 30, tzinfo=UTC),
        finished_at=datetime(2026, 5, 6, 14, 31, tzinfo=UTC),
        accounting=_fanout_accounting(),
        error=None,
        landscape_run_id="a2a7354a-5732-475b-a4ac-ed166a9e0f25",
        discard_summary=None,
    )

    assert response.accounting is not None
    assert response.accounting.source.rows_processed == 1
    assert response.accounting.tokens.succeeded == 9323
    assert response.accounting.tokens.structural == 1


def test_run_results_accepts_one_source_row_many_terminal_tokens() -> None:
    response = RunResultsResponse(
        run_id="a2a7354a-5732-475b-a4ac-ed166a9e0f25",
        status="completed",
        accounting=_fanout_accounting(),
        landscape_run_id="a2a7354a-5732-475b-a4ac-ed166a9e0f25",
        error=None,
        discard_summary=None,
    )

    assert response.accounting.tokens.emitted == 9324
    assert response.accounting.tokens.terminal == 9324


def test_completed_event_carries_accounting_instead_of_mixed_rows() -> None:
    event = CompletedData(
        status="completed",
        accounting=_fanout_accounting(),
        landscape_run_id="a2a7354a-5732-475b-a4ac-ed166a9e0f25",
    )

    assert event.accounting.tokens.succeeded == 9323


def test_closed_accounting_requires_all_emitted_tokens_terminal() -> None:
    with pytest.raises(ValidationError, match="closed accounting requires pending == 0"):
        RunAccounting(
            source=RunAccountingSource(rows_processed=1),
            tokens=RunAccountingTokens(
                emitted=3,
                terminal=2,
                succeeded=2,
                failed=0,
                structural=0,
                pending=1,
            ),
            routing=RunAccountingRouting(
                routed_success=0,
                routed_failure=0,
                quarantined=0,
                discarded=0,
            ),
            integrity=RunAccountingIntegrity(
                closure="closed",
                missing_terminal_outcomes=1,
                duplicate_terminal_outcomes=0,
            ),
        )


def test_completed_status_requires_closed_accounting() -> None:
    accounting = _fanout_accounting().model_copy(
        update={
            "integrity": RunAccountingIntegrity(
                closure="open",
                missing_terminal_outcomes=1,
                duplicate_terminal_outcomes=0,
            ),
            "tokens": RunAccountingTokens(
                emitted=9324,
                terminal=9323,
                succeeded=9323,
                failed=0,
                structural=0,
                pending=1,
            ),
        }
    )

    with pytest.raises(ValidationError, match="status='completed' requires closed token accounting"):
        RunStatusResponse(
            run_id="run-1",
            status="completed",
            started_at=datetime(2026, 5, 6, 14, 30, tzinfo=UTC),
            finished_at=datetime(2026, 5, 6, 14, 31, tzinfo=UTC),
            accounting=accounting,
            error=None,
            landscape_run_id="run-1",
            discard_summary=None,
        )


def test_progress_event_uses_explicit_source_and_token_names() -> None:
    progress = ProgressData(
        source_rows_processed=1,
        tokens_succeeded=9323,
        tokens_failed=0,
        tokens_quarantined=0,
        tokens_routed_success=0,
        tokens_routed_failure=0,
    )

    payload = progress.model_dump()

    assert payload["source_rows_processed"] == 1
    assert payload["tokens_succeeded"] == 9323
    assert "rows_processed" not in payload
