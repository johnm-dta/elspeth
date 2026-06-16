"""Regression tests for _close_orchestrator_resources teardown masking.

elspeth-5310f58a2b: _orchestrator_context() always closed rate_limit_registry
and telemetry_manager in its finally block, unguarded. RateLimiter.close()
crosses pyrate-limiter/SQLite internals and TelemetryManager.close() re-raises
non-transport exporter close errors, so a teardown exception raised from that
finally replaced the primary exception from orchestrator.run()/resume() — the
CLI reported a cleanup failure instead of the real pipeline failure.

The teardown logic is extracted into _close_orchestrator_resources so the
masking-preservation behaviour is unit-testable without building the whole
orchestrator context.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import elspeth.contracts.errors as contract_errors
from elspeth.cli import _close_orchestrator_resources


class _CloseRaises:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.closed = False

    def close(self) -> None:
        self.closed = True
        raise self._exc


class _CloseOk:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _interactive_config() -> SimpleNamespace:
    return SimpleNamespace(
        landscape=SimpleNamespace(
            url="sqlite:///audit.db",
            dump_to_jsonl=False,
            dump_to_jsonl_path=None,
            dump_to_jsonl_fail_on_error=False,
            dump_to_jsonl_include_payloads=False,
            dump_to_jsonl_payload_base_path=None,
        ),
        payload_store=SimpleNamespace(
            backend="filesystem",
            base_path=Path(".elspeth/payloads"),
        ),
    )


@contextmanager
def _patched_interactive_execution(mock_db: MagicMock, mock_run_result: object | BaseException):
    mock_ctx = MagicMock()
    mock_ctx.pipeline_config = MagicMock()
    if isinstance(mock_run_result, BaseException):
        mock_ctx.orchestrator.run.side_effect = mock_run_result
    else:
        mock_ctx.orchestrator.run.return_value = mock_run_result

    with (
        patch("elspeth.core.landscape.LandscapeDB") as mock_db_cls,
        patch("elspeth.core.payload_store.FilesystemPayloadStore"),
        patch("elspeth.cli._orchestrator_context") as mock_orch_ctx,
        patch("elspeth.plugins.infrastructure.runtime_factory.make_sink_factory", return_value=MagicMock()),
        patch(
            "elspeth.plugins.transforms.llm.model_catalog.read_openrouter_catalog_snapshot_id",
            return_value=("sha256", "test"),
        ),
    ):
        mock_db_cls.from_url.return_value = mock_db
        mock_orch_ctx.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_orch_ctx.return_value.__exit__ = MagicMock(return_value=False)
        yield


def test_close_failure_does_not_mask_pending_pipeline_exception():
    """With a pipeline exception pending, close failures are suppressed (logged), not raised."""
    rate_limit_registry = _CloseRaises(RuntimeError("rate limiter teardown boom"))
    telemetry_manager = _CloseRaises(RuntimeError("telemetry close boom"))

    # Must not raise — the pending pipeline exception (param) outranks teardown errors.
    _close_orchestrator_resources(
        rate_limit_registry,
        telemetry_manager,
        pending_exc=ValueError("real pipeline failure"),
    )

    # Both resources attempted, even though the first close() raised.
    assert rate_limit_registry.closed and telemetry_manager.closed


def test_close_failure_surfaces_when_no_pending_exception():
    """With no pipeline exception pending, a teardown close failure must surface."""
    rate_limit_registry = _CloseOk()
    telemetry_manager = _CloseRaises(RuntimeError("telemetry close boom"))

    with pytest.raises(RuntimeError, match="telemetry close boom"):
        _close_orchestrator_resources(rate_limit_registry, telemetry_manager, pending_exc=None)

    # Both still attempted before the surfaced error.
    assert rate_limit_registry.closed and telemetry_manager.closed


def test_tier1_error_propagates_even_when_pipeline_exception_pending():
    """Audit-integrity errors during teardown outrank even the pending pipeline exception."""
    rate_limit_registry = _CloseRaises(contract_errors.AuditIntegrityError("audit corruption during close"))
    telemetry_manager = _CloseOk()

    with pytest.raises(contract_errors.AuditIntegrityError):
        _close_orchestrator_resources(
            rate_limit_registry,
            telemetry_manager,
            pending_exc=ValueError("real pipeline failure"),
        )


def test_none_resources_are_skipped():
    """A None resource (construction failed before assignment) is simply skipped."""
    _close_orchestrator_resources(None, None, pending_exc=None)  # must not raise


def test_interactive_pipeline_failure_is_not_masked_by_db_close_failure():
    """The interactive run wrapper must preserve the primary orchestrator error."""
    from elspeth.cli import _execute_pipeline_with_instances

    mock_db = MagicMock()
    mock_db.close.side_effect = RuntimeError("close failed")
    with _patched_interactive_execution(mock_db, ValueError("pipeline failed")), pytest.raises(ValueError, match="pipeline failed"):
        _execute_pipeline_with_instances(
            _interactive_config(),
            graph=MagicMock(),
            plugins=MagicMock(),
        )


def test_interactive_success_propagates_db_close_failure():
    """A clean interactive run must not report success when db.close() failed."""
    from elspeth.cli import _execute_pipeline_with_instances

    mock_db = MagicMock()
    mock_db.close.side_effect = RuntimeError("close failed")
    run_result = SimpleNamespace(run_id="run-1", status="completed", rows_processed=1)
    with _patched_interactive_execution(mock_db, run_result), pytest.raises(RuntimeError, match="close failed"):
        _execute_pipeline_with_instances(
            _interactive_config(),
            graph=MagicMock(),
            plugins=MagicMock(),
        )
