"""Regression tests for RunCeremony.safe_flush_telemetry teardown masking.

elspeth-1e4ca5b1db: TelemetryManager.flush() re-raises ANY stored background
exception (manager.py stores non-transport programming errors from the export
thread and re-raises them on flush()), but safe_flush_telemetry() previously
caught only TelemetryExporterError. A stored programming error therefore
escaped the finally block and replaced the in-flight run exception, despite the
method's contract that telemetry failures must not mask run errors.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from elspeth.contracts.errors import AuditIntegrityError, TelemetryExporterError
from elspeth.engine.orchestrator.ceremony import RunCeremony


class _FlushRaises:
    """Minimal TelemetryManagerProtocol stand-in whose flush() raises."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.flush_calls = 0

    def flush(self) -> None:
        self.flush_calls += 1
        raise self._exc

    def handle_event(self, event: object) -> None:  # pragma: no cover - unused
        pass


def _ceremony(telemetry: object) -> RunCeremony:
    return RunCeremony(events=MagicMock(), telemetry=telemetry)


def test_programming_error_does_not_mask_pending_run_exception():
    """A stored programming error from the export thread must not replace the run error."""
    telem = _FlushRaises(RuntimeError("export-thread programming error"))
    ceremony = _ceremony(telem)

    with pytest.raises(ValueError, match="original run failure"):
        try:
            raise ValueError("original run failure")
        finally:
            ceremony.safe_flush_telemetry()

    assert telem.flush_calls == 1  # flush was attempted, not skipped


def test_programming_error_surfaces_when_no_pending_exception():
    """With no run error in flight, a telemetry programming error must surface, not vanish."""
    telem = _FlushRaises(RuntimeError("export-thread programming error"))
    ceremony = _ceremony(telem)

    with pytest.raises(RuntimeError, match="export-thread programming error"):
        ceremony.safe_flush_telemetry()


def test_tier1_audit_error_propagates_even_when_run_exception_pending():
    """A Tier-1 audit-integrity error stored by the export thread must PROPAGATE
    even during teardown with a run exception pending — audit corruption outranks
    even the primary run failure (matching the cli.py _close_orchestrator_resources
    guard). best_effort suppression must not swallow it.
    """
    telem = _FlushRaises(AuditIntegrityError("checkpoint hash mismatch"))
    ceremony = _ceremony(telem)

    with pytest.raises(AuditIntegrityError, match="checkpoint hash mismatch"):
        try:
            raise ValueError("original run failure")
        finally:
            ceremony.safe_flush_telemetry()

    assert telem.flush_calls == 1  # flush was attempted


def test_tier1_audit_error_surfaces_when_no_pending_exception():
    """A Tier-1 error on an otherwise-clean run surfaces (raw path)."""
    telem = _FlushRaises(AuditIntegrityError("checkpoint hash mismatch"))
    ceremony = _ceremony(telem)

    with pytest.raises(AuditIntegrityError, match="checkpoint hash mismatch"):
        ceremony.safe_flush_telemetry()


def test_exporter_error_does_not_mask_pending_run_exception():
    """Preserved behaviour: a TelemetryExporterError must not mask the run error either."""
    telem = _FlushRaises(TelemetryExporterError("otlp", "all exporters failed"))
    ceremony = _ceremony(telem)

    with pytest.raises(ValueError, match="original run failure"):
        try:
            raise ValueError("original run failure")
        finally:
            ceremony.safe_flush_telemetry()


def test_exporter_error_surfaces_when_no_pending_exception():
    """Preserved behaviour: a TelemetryExporterError surfaces on an otherwise-clean run."""
    telem = _FlushRaises(TelemetryExporterError("otlp", "all exporters failed"))
    ceremony = _ceremony(telem)

    with pytest.raises(TelemetryExporterError):
        ceremony.safe_flush_telemetry()
