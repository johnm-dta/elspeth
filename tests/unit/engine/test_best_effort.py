"""Tests for best-effort post-audit ceremony handling."""

from __future__ import annotations

from unittest.mock import patch

from elspeth.engine._best_effort import best_effort


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
