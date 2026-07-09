"""Tests for best-effort post-audit ceremony handling."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from elspeth.contracts.errors import (
    AuditIntegrityError,
    FrameworkBugError,
    RunLeadershipLostError,
)
from elspeth.engine._best_effort import best_effort


class _StrictWarningLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def warning(self, event: str, **kwargs: object) -> None:
        self.calls.append((event, kwargs))


def test_best_effort_logs_exception_type_without_raw_message() -> None:
    """Ceremony failures must not leak raw exception detail into structured logs."""
    sensitive_detail = "SQL failed [parameters: {'tenant_internal_id': 'operator-only'}] /srv/private.db"

    with (
        patch("elspeth.engine._best_effort._slog") as slog,
        best_effort("TokenCompleted telemetry", run_id="run-1", token_id="token-1"),
    ):
        raise RuntimeError(sensitive_detail)

    slog.warning.assert_called_once()
    (message,) = slog.warning.call_args.args
    fields = slog.warning.call_args.kwargs

    assert message == "best-effort ceremony failed during error propagation; original event preserved"
    assert fields["operation"] == "TokenCompleted telemetry"
    assert fields["run_id"] == "run-1"
    assert fields["token_id"] == "token-1"
    assert fields["error_type"] == "RuntimeError"
    assert "error" not in fields
    assert sensitive_detail not in str(slog.warning.call_args)


def test_best_effort_suppresses_logger_failures() -> None:
    """Logger failures must not replace the ceremony failure being suppressed."""
    with patch("elspeth.engine._best_effort._slog") as slog:
        slog.warning.side_effect = RuntimeError("logger backend failed")

        with best_effort("TokenCompleted telemetry", run_id="run-1"):
            raise ValueError("primary ceremony failure")

    slog.warning.assert_called_once()


def test_best_effort_propagates_audit_integrity_error() -> None:
    """Tier-1 audit corruption must fail closed, never downgrade to a warning (elspeth-1e7cabb903)."""
    with (
        patch("elspeth.engine._best_effort._slog") as slog,
        pytest.raises(AuditIntegrityError, match="Run not found after UPDATE"),
        best_effort("Generic failure ceremony on run failure", run_id="run-1"),
    ):
        raise AuditIntegrityError("Run not found after UPDATE - database corruption")

    slog.warning.assert_not_called()


def test_best_effort_propagates_registered_tier_1_errors() -> None:
    """Every @tier_1_error-registered exception escapes, not just AuditIntegrityError."""
    with (
        patch("elspeth.engine._best_effort._slog") as slog,
        pytest.raises(FrameworkBugError, match="double-completed operation"),
        best_effort("TokenCompleted telemetry", run_id="run-1"),
    ):
        raise FrameworkBugError("double-completed operation")

    slog.warning.assert_not_called()


def test_best_effort_still_suppresses_run_leadership_lost() -> None:
    """Tier-2 coordination signals stay suppressed: a deposed leader's ceremony
    refusal must NOT crash the exiting worker (the designed semantics the
    finalize call sites rely on)."""
    with (
        patch("elspeth.engine._best_effort._slog") as slog,
        best_effort("FAILED finalize ceremony", run_id="run-1"),
    ):
        raise RunLeadershipLostError(run_id="run-1", worker_id="w-1", leader_epoch=3, verb="finalize_run")

    slog.warning.assert_called_once()
    assert slog.warning.call_args.kwargs["error_type"] == "RunLeadershipLostError"


def test_best_effort_reserved_context_keys_cannot_mask_ceremony_failure() -> None:
    """Caller context cannot collide with reserved structured log fields."""
    with (
        patch("elspeth.engine._best_effort._slog") as slog,
        best_effort("TokenCompleted telemetry", operation="caller-operation", error_type="caller-error"),
    ):
        raise ValueError("primary ceremony failure")

    slog.warning.assert_called_once()
    fields = slog.warning.call_args.kwargs
    assert fields["operation"] == "TokenCompleted telemetry"
    assert fields["error_type"] == "ValueError"


def test_best_effort_remaps_caller_event_context_key() -> None:
    """Caller ``event`` context must not collide with structlog's event argument."""
    logger = _StrictWarningLogger()

    with (
        patch("elspeth.engine._best_effort._slog", new=logger),
        best_effort("TokenCompleted telemetry", event="caller-event", run_id="run-1"),
    ):
        raise ValueError("primary ceremony failure")

    assert logger.calls == [
        (
            "best-effort ceremony failed during error propagation; original event preserved",
            {
                "context_event": "caller-event",
                "error_type": "ValueError",
                "operation": "TokenCompleted telemetry",
                "run_id": "run-1",
            },
        )
    ]
