"""Regression tests for Tier-1 escape from best_effort-wrapped failure ceremony.

elspeth-1e7cabb903: every failure-ceremony call site in orchestrator/core.py
wraps ``emit_failed_ceremony`` in ``best_effort``, whose broad catch used to
swallow ALL exceptions — including ``AuditIntegrityError`` raised by
``finalize_run`` on audit corruption. Audit corruption during the failure
ceremony was downgraded to a warning. Tier-1 errors must escape best_effort;
Tier-2 coordination refusals (deposed leader) must stay suppressed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from elspeth.contracts.coordination import CoordinationToken
from elspeth.contracts.errors import AuditIntegrityError, RunLeadershipLostError
from elspeth.core.events import EventBusProtocol
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine._best_effort import best_effort
from elspeth.engine.orchestrator.ceremony import RunCeremony
from elspeth.engine.orchestrator.ports import TelemetryManagerProtocol


def _raising_factory(exc: Exception) -> MagicMock:
    factory = MagicMock(spec=RecorderFactory)
    factory.run_lifecycle.finalize_run.side_effect = exc
    return factory


def test_finalize_audit_corruption_escapes_best_effort_ceremony() -> None:
    """AuditIntegrityError from finalize_run must reach the caller, mirroring core.py's wrap."""
    ceremony = RunCeremony(events=MagicMock(spec=EventBusProtocol), telemetry=MagicMock(spec=TelemetryManagerProtocol))
    factory = _raising_factory(AuditIntegrityError("Run not found after UPDATE - database corruption"))

    with (
        pytest.raises(AuditIntegrityError, match="Run not found after UPDATE"),
        best_effort("Generic failure ceremony on run failure", run_id="run-1"),
    ):
        ceremony.emit_failed_ceremony("run-1", factory, 0.0, token=MagicMock(spec=CoordinationToken))


def test_deposed_leader_finalize_refusal_stays_suppressed() -> None:
    """RunLeadershipLostError (Tier-2) keeps the documented swallow: the deposed
    leader exits without stamping FAILED over the new leader's progress."""
    ceremony = RunCeremony(events=MagicMock(spec=EventBusProtocol), telemetry=MagicMock(spec=TelemetryManagerProtocol))
    factory = _raising_factory(RunLeadershipLostError(run_id="run-1", worker_id="w-1", leader_epoch=3, verb="finalize_run"))

    with best_effort("Generic failure ceremony on run failure", run_id="run-1"):
        ceremony.emit_failed_ceremony("run-1", factory, 0.0, token=MagicMock(spec=CoordinationToken))
