"""RunCeremony: telemetry emission and run lifecycle event helpers.

Extracted from Orchestrator (core.py) — these methods touch only
``self._telemetry`` and ``self._events`` and have no dependencies on
the rest of the Orchestrator state.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import structlog

from elspeth.contracts.enums import RunStatus
from elspeth.contracts.errors import GracefulShutdownError
from elspeth.contracts.events import (
    PhaseError,
    PipelinePhase,
    RunCompletionStatus,
    RunFinished,
    RunSummary,
)
from elspeth.contracts.run_result import RunResult
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine._best_effort import best_effort

if TYPE_CHECKING:
    from elspeth.contracts.coordination import CoordinationToken
    from elspeth.contracts.events import TelemetryEvent
    from elspeth.core.events import EventBusProtocol
    from elspeth.engine.orchestrator.types import TelemetryManagerProtocol

slog = structlog.get_logger(__name__)


class RunCeremony:
    def __init__(
        self,
        *,
        events: EventBusProtocol,
        telemetry: TelemetryManagerProtocol | None,
    ) -> None:
        self._events = events
        self._telemetry = telemetry

    def emit_telemetry(self, event: TelemetryEvent) -> None:
        """Emit telemetry event if manager is configured.

        Telemetry is emitted AFTER Landscape recording succeeds. Landscape is
        the legal record; telemetry is operational visibility.

        Args:
            event: The telemetry event to emit
        """
        if self._telemetry is not None:
            self._telemetry.handle_event(event)

    def flush_telemetry(self) -> None:
        """Flush telemetry events if manager is configured.

        Ensures queued telemetry is exported before returning control to caller.
        """
        if self._telemetry is not None:
            self._telemetry.flush()

    def emit_run_finished(
        self,
        *,
        run_id: str,
        status: RunStatus,
        row_count: int,
        duration_seconds: float,
    ) -> None:
        """Emit the terminal RunFinished telemetry event."""
        self.emit_telemetry(
            RunFinished(
                timestamp=datetime.now(UTC),
                run_id=run_id,
                status=status,
                row_count=row_count,
                duration_ms=duration_seconds * 1000,
            )
        )

    def emit_run_summary(
        self,
        *,
        run_id: str,
        status: RunCompletionStatus,
        rows_processed: int,
        rows_succeeded: int,
        rows_failed: int,
        rows_quarantined: int,
        duration_seconds: float,
        exit_code: int,
        rows_routed_success: int = 0,
        rows_routed_failure: int = 0,
        routed_destinations: Mapping[str, int] | tuple[tuple[str, int], ...] = (),
    ) -> None:
        """Emit the terminal operator-facing RunSummary event."""
        if isinstance(routed_destinations, Mapping):
            destination_items = tuple(routed_destinations.items())
        else:
            destination_items = tuple(routed_destinations)
        self._events.emit(
            RunSummary(
                run_id=run_id,
                status=status,
                total_rows=rows_processed,
                succeeded=rows_succeeded,
                failed=rows_failed,
                quarantined=rows_quarantined,
                duration_seconds=duration_seconds,
                exit_code=exit_code,
                routed_success=rows_routed_success,
                routed_failure=rows_routed_failure,
                routed_destinations=destination_items,
            )
        )

    def emit_partial_summary(
        self,
        *,
        run_id: str,
        result: RunResult,
        start_time: float,
    ) -> None:
        """Emit a PARTIAL summary for post-completion failures such as export."""
        total_duration = time.perf_counter() - start_time
        self.emit_run_summary(
            run_id=run_id,
            status=RunCompletionStatus.PARTIAL,
            rows_processed=result.rows_processed,
            rows_succeeded=result.rows_succeeded,
            rows_failed=result.rows_failed,
            rows_quarantined=result.rows_quarantined,
            duration_seconds=total_duration,
            exit_code=1,
            rows_routed_success=result.rows_routed_success,
            rows_routed_failure=result.rows_routed_failure,
            routed_destinations=result.routed_destinations,
        )

    def emit_phase_error(
        self,
        phase: PipelinePhase,
        error: BaseException,
        target: str | None = None,
    ) -> None:
        """Best-effort PhaseError emission that never masks the original exception.

        Called from except blocks before re-raise. If PhaseError construction
        or EventBus.emit() fails (e.g., handler bug), the original exception
        must take precedence — observable telemetry is secondary to preserving
        the actual error.
        """
        with best_effort(
            "PhaseError emission",
            phase=phase.value,
            original_error=type(error).__name__,
            target=target,
        ):
            self._events.emit(PhaseError.from_exception(phase=phase, error=error, target=target))

    def safe_flush_telemetry(self) -> None:
        """Flush telemetry in a finally block, preserving any pending exception.

        flush_telemetry() can raise more than TelemetryExporterError: the
        TelemetryManager re-raises ANY exception stored by its export thread on
        flush() (manager.py), including non-transport programming errors. When a
        run exception is already in flight, none of these may replace it —
        telemetry failures must not mask run errors — so the flush runs inside
        best_effort, which logs and suppresses any Exception. When no run
        exception is pending, a telemetry failure should surface rather than
        vanish, so the flush runs raw and propagates.

        EXCEPTION: Tier-1 / audit-integrity errors always propagate, even with a
        run exception pending — audit corruption outranks even the primary
        failure. The export thread stores FrameworkBugError/AuditIntegrityError
        (manager.py types _stored_exception as BaseException for exactly this),
        and these are not-yet-surfaced audit signals; best_effort's broad
        suppression would silently discard them, so they are re-raised before it.
        Mirrors the cli.py _close_orchestrator_resources teardown guard.

        (Previously this caught only TelemetryExporterError, so a stored
        programming error escaped the finally and replaced the run error —
        elspeth-1e4ca5b1db.)
        """
        import sys

        # Live attribute access of the lazily materialized TIER_1_ERRORS tuple —
        # never a from-import snapshot (which would capture an empty/stale tuple
        # and let Tier-1 errors fall through to best_effort's broad suppression).
        import elspeth.contracts.errors as contract_errors

        if sys.exc_info()[0] is not None:
            try:
                self.flush_telemetry()
            except contract_errors.TIER_1_ERRORS:
                raise
            except Exception:
                # Re-raise inside best_effort so its safe, class-only logging and
                # suppression apply to non-Tier-1 telemetry failures.
                with best_effort("Telemetry flush during exception cleanup"):
                    raise
        else:
            self.flush_telemetry()

    def emit_interrupted_ceremony(
        self,
        run_id: str,
        factory: RecorderFactory,
        shutdown_exc: GracefulShutdownError,
        start_time: float,
        *,
        token: CoordinationToken | None = None,
    ) -> None:
        """Emit telemetry and EventBus events for a gracefully interrupted run.

        Shared between run() and resume() — the interrupted ceremony is identical
        in both paths: finalize as INTERRUPTED, emit RunFinished, emit RunSummary.

        ``token`` (ADR-030) is threaded into the finalize, whose epoch fence
        refuses a deposed leader's ceremony — "the run is no longer its to
        fail" (§C.4 row 4). Every ceremony call site wraps this in
        ``best_effort``, which logs-and-suppresses: a deposed leader's
        ``RunLeadershipLostError`` is swallowed and the process exits WITHOUT
        stamping INTERRUPTED over the new leader's progress. That is the
        designed semantics; the ``fence_refusal`` event (written by the
        fenced verb before the raise propagates into ``best_effort``) is the
        audit trace.
        """

        total_duration = time.perf_counter() - start_time
        factory.run_lifecycle.finalize_run(run_id, status=RunStatus.INTERRUPTED, token=token)

        self.emit_run_finished(
            run_id=run_id,
            status=RunStatus.INTERRUPTED,
            row_count=shutdown_exc.rows_processed,
            duration_seconds=total_duration,
        )
        self.emit_run_summary(
            run_id=run_id,
            status=RunCompletionStatus.INTERRUPTED,
            rows_processed=shutdown_exc.rows_processed,
            rows_succeeded=shutdown_exc.rows_succeeded,
            rows_failed=shutdown_exc.rows_failed,
            rows_quarantined=shutdown_exc.rows_quarantined,
            duration_seconds=total_duration,
            exit_code=3,
            rows_routed_success=shutdown_exc.rows_routed_success,
            rows_routed_failure=shutdown_exc.rows_routed_failure,
            routed_destinations=shutdown_exc.routed_destinations,
        )

    def emit_failed_ceremony(
        self,
        run_id: str,
        factory: RecorderFactory,
        start_time: float,
        result: RunResult | None = None,
        *,
        token: CoordinationToken | None = None,
    ) -> None:
        """Emit telemetry and EventBus events for a failed run.

        Finalizes the run as FAILED, emits RunFinished telemetry and RunSummary
        with the best available metrics. Shared between run() (when
        run_completed=False) and resume().

        ``token`` (ADR-030) is threaded into the finalize, whose epoch fence
        refuses a deposed leader's ceremony — "the run is no longer its to
        fail" (§C.4 row 4). Every ceremony call site wraps this in
        ``best_effort``, which logs-and-suppresses: a deposed leader's
        ``RunLeadershipLostError`` is swallowed and the process exits WITHOUT
        stamping FAILED over the new leader's progress. That is the designed
        semantics; the ``fence_refusal`` event (written by the fenced verb
        before the raise propagates into ``best_effort``) is the audit trace.
        """

        failed_result = result or RunResult(
            run_id=run_id,
            status=RunStatus.FAILED,
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
            rows_forked=0,
            rows_coalesced=0,
            rows_coalesce_failed=0,
            rows_expanded=0,
            rows_buffered=0,
            rows_diverted=0,
            routed_destinations={},
        )
        total_duration = time.perf_counter() - start_time
        factory.run_lifecycle.finalize_run(run_id, status=RunStatus.FAILED, token=token)

        self.emit_run_finished(
            run_id=run_id,
            status=RunStatus.FAILED,
            row_count=failed_result.rows_processed,
            duration_seconds=total_duration,
        )
        self.emit_run_summary(
            run_id=run_id,
            status=RunCompletionStatus.FAILED,
            rows_processed=failed_result.rows_processed,
            rows_succeeded=failed_result.rows_succeeded,
            rows_failed=failed_result.rows_failed,
            rows_quarantined=failed_result.rows_quarantined,
            duration_seconds=total_duration,
            exit_code=2,  # exit_code: 0=success, 1=partial, 2=total failure
            rows_routed_success=failed_result.rows_routed_success,
            rows_routed_failure=failed_result.rows_routed_failure,
            routed_destinations=failed_result.routed_destinations,
        )
