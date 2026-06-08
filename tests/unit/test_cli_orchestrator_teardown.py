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
