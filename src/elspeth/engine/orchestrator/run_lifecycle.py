"""RunLifecycleCoordinator: fresh-run lifecycle ownership for the orchestrator.

Extracted from ``Orchestrator.run`` / ``Orchestrator._initialize_database_phase``
/ ``Orchestrator._execute_export_phase`` (filigree elspeth-9e71ae82a4). The
facade keeps thin delegators; this coordinator owns the ordering.

CRASH-BEHAVIOUR CONTRACT (do not reorder):
- The heartbeat thread starts AFTER the seat is minted + token bound, BEFORE
  the run body's try block (ADR-030 §A.3).
- ``_heartbeat.stop()`` is the FIRST statement of every except arm that
  releases the seat, and runs just before ``release_seat`` on the success
  path; the ``finally`` stop is the idempotent safety net.
- Seat release happens only AFTER the terminal finalize for that arm
  succeeded (inside the same ``best_effort`` block), so a finalize failure
  leaves the seat to lapse rather than vacating an unfinalized run.
- Terminal status is derived from the audit trail on ALL paths (ADR-030 §D);
  live loop counters are demoted to a parity cross-check.

The two facade-owned seams (``initialize_database_phase``, ``execute_run``)
are injected per call as bound methods so tests that monkeypatch
``Orchestrator._initialize_database_phase`` (test_adr_019_sweep_durability)
or stub instance methods keep intercepting. Tests that previously patched
``…orchestrator.core.export_landscape`` patch this module instead.
"""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from elspeth.contracts import (
    ExportStatus,
    SecretResolutionInput,
    SinkProtocol,
)
from elspeth.contracts.coordination import CoordinationToken, mint_worker_id
from elspeth.contracts.errors import (
    GracefulShutdownError,
    OrchestrationInvariantError,
)
from elspeth.contracts.events import (
    PhaseAction,
    PhaseChanged,
    PhaseCompleted,
    PhaseStarted,
    PipelinePhase,
    RunStarted,
)
from elspeth.core.landscape._helpers import generate_id
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine._best_effort import best_effort
from elspeth.engine.orchestrator.bootstrap import prepare_for_run
from elspeth.engine.orchestrator.export import (
    export_landscape,
)
from elspeth.engine.orchestrator.heartbeat import RunHeartbeatThread
from elspeth.engine.orchestrator.run_status import (
    assert_terminal_counter_parity,
    cli_completion_for,
    derive_terminal_status_from_audit,
)
from elspeth.engine.orchestrator.shutdown import shutdown_handler_context
from elspeth.engine.orchestrator.types import _RunFailedWithPartialResultError

if TYPE_CHECKING:
    import threading
    from collections.abc import Callable

    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.config import ElspethSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.dependency_config import PreflightResult
    from elspeth.core.events import EventBusProtocol
    from elspeth.core.landscape import LandscapeDB
    from elspeth.engine.orchestrator.ceremony import RunCeremony
    from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator
    from elspeth.engine.orchestrator.types import PipelineConfig, RunResult
    from elspeth.engine.spans import SpanFactory


class InitializeDatabasePhase(Protocol):
    """Facade-bound DATABASE-phase seam (``Orchestrator._initialize_database_phase``)."""

    def __call__(
        self,
        config: PipelineConfig,
        payload_store: PayloadStore,
        secret_resolutions: list[SecretResolutionInput] | None,
        *,
        run_id: str | None = None,
        initiated_by_user_id: str | None = None,
        auth_provider_type: str | None = None,
        openrouter_catalog_sha256: str,
        openrouter_catalog_source: str,
    ) -> tuple[RecorderFactory, Any, CoordinationToken]: ...


class ExecuteRun(Protocol):
    """Facade-bound run-body seam (``Orchestrator._execute_run``)."""

    def __call__(
        self,
        factory: RecorderFactory,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
        settings: ElspethSettings | None = None,
        *,
        payload_store: PayloadStore,
        shutdown_event: threading.Event | None = None,
        coordination_token: CoordinationToken | None = None,
        check_coordination_latch: Callable[[], None] | None = None,
    ) -> RunResult: ...


class RunLifecycleCoordinator:
    """Owns fresh-run bootstrap, heartbeat lifecycle, finalize ordering and ceremonies."""

    def __init__(
        self,
        *,
        db: LandscapeDB,
        events: EventBusProtocol,
        ceremony: RunCeremony,
        checkpoints: CheckpointCoordinator,
        span_factory: SpanFactory,
        canonical_version: str,
    ) -> None:
        self._db = db
        self._events = events
        self._ceremony = ceremony
        self._checkpoints = checkpoints
        self._span_factory = span_factory
        self._canonical_version = canonical_version

    def initialize_database_phase(
        self,
        config: PipelineConfig,
        payload_store: PayloadStore,
        secret_resolutions: list[SecretResolutionInput] | None,
        *,
        run_id: str | None = None,
        initiated_by_user_id: str | None = None,
        auth_provider_type: str | None = None,
        openrouter_catalog_sha256: str,
        openrouter_catalog_source: str,
    ) -> tuple[RecorderFactory, Any, CoordinationToken]:
        """Execute the DATABASE phase: create factory, begin run, record secrets.

        Args:
            config: Pipeline configuration.
            payload_store: PayloadStore for audit compliance.
            secret_resolutions: Optional secret resolution records.
            run_id: Optional caller-supplied run ID for audit correlation.
            initiated_by_user_id: Optional authenticated web user that initiated the run.
            auth_provider_type: Optional auth provider namespace for the initiating user.

        Returns:
            Tuple of (factory, run, coordination_token) where run has run_id
            and config_hash attributes. The token is the epoch-1 leader seat
            minted atomically with the runs row (ADR-030 uniformity rule:
            N=1 = leader-of-its-own-run); epoch 1 is a constant on the fresh
            path so no read-back is needed.

        Raises:
            Exception: Re-raises any database connection or initialization failure.
        """

        phase_start = time.perf_counter()
        try:
            self._events.emit(PhaseStarted(phase=PipelinePhase.DATABASE, action=PhaseAction.CONNECTING))

            # Serialize the first source's schema for resume type restoration.
            # This enables proper type coercion (datetime/Decimal) when resuming
            # from JSON payloads. Per-source schema contracts live exclusively in
            # ``run_sources`` (one row per declared source), populated by
            # ``run_main_processing_loop`` as each source iterates — this is the
            # G6 (elspeth-2e2f2184ab) contract. The run-level ``schema_contract``
            # singleton was deleted because writers and the integrity verifier
            # disagreed about which surface was authoritative.
            # ``runs.source_schema_json`` remains as the first source's typed-
            # resume header (single-source legacy shape) and is scheduled for
            # deletion when G6 (elspeth-2e2f2184ab) lands; multi-source resume
            # already reads per-source schemas from ``run_sources.schema_json``.
            first_source_name = next(iter(config.sources))
            first_source = config.sources[first_source_name]
            source_schema_json = json.dumps(first_source.output_schema.model_json_schema())

            factory = RecorderFactory(self._db, payload_store=payload_store)

            # Epoch 21 (ADR-030 §A.1/§B.4): hoist run-id generation so the
            # leader worker identity can embed it, then let begin_run mint
            # the run_coordination seat (epoch 1) atomically with the runs
            # row. The token is constructed locally — epoch 1 is a constant
            # on the fresh path, no read-back.
            run_id = run_id or generate_id()
            worker_id = mint_worker_id(run_id)
            run = factory.run_lifecycle.begin_run(
                config=config.config,
                canonical_version=self._canonical_version,
                source_schema_json=source_schema_json,
                run_id=run_id,
                initiated_by_user_id=initiated_by_user_id,
                auth_provider_type=auth_provider_type,
                openrouter_catalog_sha256=openrouter_catalog_sha256,
                openrouter_catalog_source=openrouter_catalog_source,
                leader_worker_id=worker_id,
            )
            coordination_token = CoordinationToken(run_id=run.run_id, worker_id=worker_id, leader_epoch=1)

            # Record secret resolutions in audit trail (deferred from pre-run loading)
            # Resolutions already contain pre-computed fingerprints (no plaintext values)
            if secret_resolutions:
                factory.run_lifecycle.record_secret_resolutions(
                    run_id=run.run_id,
                    resolutions=secret_resolutions,
                )

            # Emit telemetry AFTER Landscape succeeds - Landscape is the legal record
            self._ceremony.emit_telemetry(
                RunStarted(
                    timestamp=datetime.now(UTC),
                    run_id=run.run_id,
                    config_hash=run.config_hash,
                    source_plugin=first_source.name,
                )
            )

            self._events.emit(PhaseCompleted(phase=PipelinePhase.DATABASE, duration_seconds=time.perf_counter() - phase_start))
        except Exception as e:
            self._ceremony.emit_phase_error(PipelinePhase.DATABASE, e)
            raise  # CRITICAL: Always re-raise - database connection failure is fatal

        return factory, run, coordination_token

    def execute_export_phase(
        self,
        factory: RecorderFactory,
        run_id: str,
        settings: ElspethSettings,
        sink_factory: Callable[[str], SinkProtocol],
    ) -> None:
        """Execute the EXPORT phase: export Landscape data to configured sink.

        Args:
            factory: RecorderFactory for status tracking.
            run_id: Run identifier.
            settings: Full settings (export config accessed from settings.landscape.export).
            sink_factory: Creates a fresh sink instance by name for export.

        Raises:
            Exception: Re-raises any export failure (run is still "completed" in Landscape).
        """

        export_config = settings.landscape.export
        factory.run_lifecycle.set_export_status(
            run_id,
            status=ExportStatus.PENDING,
            export_format=export_config.format,
            export_sink=export_config.sink,
        )

        phase_start = time.perf_counter()
        try:
            self._events.emit(PhaseStarted(phase=PipelinePhase.EXPORT, action=PhaseAction.EXPORTING, target=export_config.sink))

            # Emit telemetry PhaseChanged for EXPORT
            self._ceremony.emit_telemetry(
                PhaseChanged(
                    timestamp=datetime.now(UTC),
                    run_id=run_id,
                    phase=PipelinePhase.EXPORT,
                    action=PhaseAction.EXPORTING,
                )
            )

            export_landscape(self._db, run_id, settings, sink_factory)

            factory.run_lifecycle.set_export_status(run_id, status=ExportStatus.COMPLETED)
            self._events.emit(PhaseCompleted(phase=PipelinePhase.EXPORT, duration_seconds=time.perf_counter() - phase_start))
        except Exception as export_error:
            self._ceremony.emit_phase_error(PipelinePhase.EXPORT, export_error, target=export_config.sink)
            with best_effort(
                "Export status FAILED recording",
                run_id=run_id,
                original_error=type(export_error).__name__,
            ):
                factory.run_lifecycle.set_export_status(
                    run_id,
                    status=ExportStatus.FAILED,
                    error=str(export_error),
                )
            # Re-raise so caller knows export failed
            # (run is still "completed" in Landscape)
            raise

    def run(
        self,
        config: PipelineConfig,
        graph: ExecutionGraph | None = None,
        settings: ElspethSettings | None = None,
        *,
        payload_store: PayloadStore,
        secret_resolutions: list[SecretResolutionInput] | None = None,
        preflight_results: PreflightResult | None = None,
        shutdown_event: threading.Event | None = None,
        sink_factory: Callable[[str], SinkProtocol] | None = None,
        run_id: str | None = None,
        initiated_by_user_id: str | None = None,
        auth_provider_type: str | None = None,
        openrouter_catalog_sha256: str | None = None,
        openrouter_catalog_source: str | None = None,
        initialize_database_phase: InitializeDatabasePhase,
        execute_run: ExecuteRun,
    ) -> RunResult:
        """Execute a pipeline run (see ``Orchestrator.run`` for the public contract).

        The two seams (``initialize_database_phase``, ``execute_run``) are the
        facade's bound delegator methods, resolved at each ``Orchestrator.run``
        call so class- and instance-level test patches are honoured.

        Raises:
            OrchestrationInvariantError: If graph or payload_store is not provided
        """
        if graph is None:
            raise OrchestrationInvariantError("ExecutionGraph is required. Build with ExecutionGraph.from_plugin_instances()")
        if payload_store is None:
            raise OrchestrationInvariantError("PayloadStore is required for audit compliance.")

        # ADR-010 §Decision 3: assert registry non-empty and freeze both
        # registries before any row is processed. prepare_for_run() is
        # idempotent when the registry is already frozen (short-circuits on
        # repeat calls from multi-run test suites). The non-empty assertion
        # only fires on the first call per process lifetime; subsequent calls
        # trust the earlier freeze was performed after a passing assertion.
        prepare_for_run()

        # Schema validation now happens in ExecutionGraph.validate() during graph construction
        self._checkpoints.reset_sequence()

        # OpenRouter catalog snapshot is mandatory for the audit trail —
        # every run records which model catalog blessed its decisions.
        # Resolution happens at the L3 entry point (web lifespan, CLI
        # bootstrap) so the engine (L2) doesn't import from plugins (L3);
        # arrival here with ``None`` is a programmer bug in the caller
        # and crashes loudly rather than writing NULL into the audit row.
        if openrouter_catalog_sha256 is None or openrouter_catalog_source is None:
            raise OrchestrationInvariantError(
                "openrouter_catalog_sha256 and openrouter_catalog_source are required. "
                "Resolve via plugins.transforms.llm.model_catalog.read_openrouter_catalog_snapshot_id() "
                "at the L3 entry point (web lifespan or CLI bootstrap) and pass through."
            )

        # DATABASE phase - create factory and begin run (mints the epoch-1
        # leader seat — ADR-030 uniformity rule)
        factory, run, coordination_token = initialize_database_phase(
            config,
            payload_store,
            secret_resolutions,
            run_id=run_id,
            initiated_by_user_id=initiated_by_user_id,
            auth_provider_type=auth_provider_type,
            openrouter_catalog_sha256=openrouter_catalog_sha256,
            openrouter_catalog_source=openrouter_catalog_source,
        )

        # Record pre-flight results (deferred from bootstrap_and_run)
        if preflight_results is not None:
            factory.run_lifecycle.record_preflight_results(
                run_id=run.run_id,
                preflight=preflight_results,
            )

        # Thread the coordination token to the collaborators that step 4 of
        # slice 2 fences (checkpoint writes, finalize, ceremonies): the
        # token is carried by value, never re-read mid-run.
        self._checkpoints.bind_coordination(coordination_token)

        # ADR-030 §A.3 (slice 4): start the dedicated heartbeat thread AFTER
        # the seat is minted and the token is bound, BEFORE the run body's
        # try/except block.  The thread beats both the run_workers row and the
        # run_coordination seat in ONE BEGIN IMMEDIATE transaction so the two
        # liveness clocks can never skew.
        #
        # Sequencing invariant (design §A.3 "joined before release_seat"): the
        # thread must NOT beat the seat after release_seat vacates it — a beat
        # on a vacant seat would re-set leader_heartbeat_expires_at and fool
        # the entry guard's liveness check.  So stop() is called as the FIRST
        # statement of every except arm that calls release_seat and in the
        # success path just before release_seat; the finally block calls stop()
        # again as an idempotent safety net for any path that exits without an
        # explicit stop.
        _heartbeat = RunHeartbeatThread(
            factory.run_coordination,
            token=coordination_token,
        )
        _heartbeat.start()

        run_completed = False
        run_start_time = time.perf_counter()
        try:
            # When shutdown_event is provided (testing), skip signal handler
            # installation and use the caller's event directly.
            shutdown_ctx = nullcontext(shutdown_event) if shutdown_event is not None else shutdown_handler_context()
            with self._span_factory.run_span(run.run_id), shutdown_ctx as active_event:
                result = execute_run(
                    factory,
                    run.run_id,
                    config,
                    graph,
                    settings,
                    payload_store=payload_store,
                    shutdown_event=active_event,
                    coordination_token=coordination_token,
                    # ADR-030 §A.3 / §C.2: wire the heartbeat latch into the
                    # per-row drain boundary so a deposed leader raises
                    # RunWorkerEvictedError without waiting for the next fenced
                    # write to refuse.  The latch is an optimization on top of the
                    # epoch/membership fences — both independently refuse the same
                    # writes — but the latch surfaces the condition proactively.
                    check_coordination_latch=_heartbeat.check_and_raise,
                )

            # ADR-030 §D (audit-derived terminal status on ALL paths — bug
            # elspeth-ff6d48c180): the normal completion arm now derives its
            # terminal status AND counters from the audit trail, exactly like
            # both resume branches. Sequencing is sound here: _execute_run
            # returned only after the end-of-source flushes, the sink writes
            # and sweep_deferred_invariants_or_crash committed, so every
            # outcome is visible to the derive. The live loop counters are
            # demoted to a parity cross-check (loud on unexplained mismatch;
            # the two documented rows_coalesce_failed divergences are
            # tolerated — see assert_terminal_counter_parity).
            terminal_status, audit_counters = derive_terminal_status_from_audit(factory, run.run_id)
            assert_terminal_counter_parity(live=result, audit=audit_counters, run_id=run.run_id)

            # Complete run with reproducibility grade computation
            factory.run_lifecycle.finalize_run(run.run_id, status=terminal_status, token=coordination_token)
            result = audit_counters.to_run_result(run.run_id, terminal_status)
            run_completed = True

            # Delete checkpoints on successful completion (checkpoints are
            # for recovery, not needed after success). LEADER WORK: the
            # delete is epoch-fenced (ADR-030 §C.4 row 5), so it must run
            # BEFORE the seat release vacates the fence's CAS target.
            self._checkpoints.delete_checkpoints(run.run_id)

            # ADR-030 §A.3: stop the heartbeat thread BEFORE releasing the
            # seat — the thread must not beat the seat after it is vacated.
            _heartbeat.stop()

            # Seat hygiene (ADR-030 §D): the leader releases its seat AFTER
            # the terminal finalize succeeds. Best-effort — a failed release
            # leaves the seat to lapse on its liveness window; it must never
            # un-complete a completed run.
            with best_effort("Seat release after finalize", run_id=run.run_id):
                factory.run_coordination.release_seat(token=coordination_token, now=datetime.now(UTC))

            # Emit telemetry AFTER Landscape finalize succeeds
            run_duration = time.perf_counter() - run_start_time
            self._ceremony.emit_run_finished(
                run_id=run.run_id,
                status=terminal_status,
                row_count=result.rows_processed,
                duration_seconds=run_duration,
            )

            # EXPORT phase - post-run landscape export (if enabled)
            if settings is not None and settings.landscape.export.enabled:
                if sink_factory is None:
                    raise ValueError(
                        "Export is enabled but no sink_factory was provided to orchestrator.run(). "
                        "The caller must supply a sink_factory so the export phase can create "
                        "a fresh sink instance (the pipeline's sinks are already closed)."
                    )
                self.execute_export_phase(factory, run.run_id, settings, sink_factory)

            # Emit RunSummary event with final metrics.  Map the new
            # terminal status onto the CLI exit-code taxonomy via
            # ``cli_completion_for`` so the operator-facing CLI summary
            # remains coherent with /api/runs/{rid}.
            cli_status, exit_code = cli_completion_for(terminal_status)
            total_duration = time.perf_counter() - run_start_time
            self._ceremony.emit_run_summary(
                run_id=run.run_id,
                status=cli_status,
                rows_processed=result.rows_processed,
                rows_succeeded=result.rows_succeeded,
                rows_failed=result.rows_failed,
                rows_quarantined=result.rows_quarantined,
                duration_seconds=total_duration,
                exit_code=exit_code,
                rows_routed_success=result.rows_routed_success,
                rows_routed_failure=result.rows_routed_failure,
                routed_destinations=result.routed_destinations,
            )

            return result

        except GracefulShutdownError as shutdown_exc:
            # ADR-030 §A.3: stop the heartbeat thread before the seat is released.
            _heartbeat.stop()
            with best_effort("Interrupted ceremony on graceful shutdown", run_id=run.run_id):
                self._ceremony.emit_interrupted_ceremony(run.run_id, factory, shutdown_exc, run_start_time, token=coordination_token)
                # Seat hygiene: released only AFTER the INTERRUPTED finalize
                # succeeded (same best_effort block), so a finalize failure
                # leaves the seat to lapse rather than vacating a run whose
                # terminal status was never recorded.
                factory.run_coordination.release_seat(token=coordination_token, now=datetime.now(UTC))
            raise  # Propagate to CLI
        except _RunFailedWithPartialResultError as failed_exc:
            # ADR-030 §A.3: stop the heartbeat thread before the seat is released.
            _heartbeat.stop()
            with best_effort(
                "Failed/partial-result ceremony on run failure",
                run_id=run.run_id,
                run_completed=run_completed,
            ):
                if run_completed:
                    # Export failed after successful run — emit PARTIAL status.
                    # RunFinished was already emitted before the export attempt,
                    # so only emit the EventBus RunSummary here.
                    self._ceremony.emit_partial_summary(run_id=run.run_id, result=result, start_time=run_start_time)
                else:
                    self._ceremony.emit_failed_ceremony(
                        run.run_id,
                        factory,
                        run_start_time,
                        failed_exc.partial_result,
                        token=coordination_token,
                    )
                    # Seat hygiene: after the FAILED finalize succeeded.
                    factory.run_coordination.release_seat(token=coordination_token, now=datetime.now(UTC))
            raise failed_exc.original_error.with_traceback(failed_exc.original_traceback) from None
        except Exception:
            # Outer broad-except: any unhandled exception type is a run failure
            # requiring a RunSummary. The inner ceremony is best-effort and must
            # not mask the original; the outer catch re-raises after.
            # ADR-030 §A.3: stop the heartbeat thread before the seat is released.
            _heartbeat.stop()
            with best_effort(
                "Generic failure ceremony on run failure",
                run_id=run.run_id,
                run_completed=run_completed,
            ):
                if run_completed:
                    # Export failed after successful run — emit PARTIAL status.
                    # RunFinished was already emitted before the export attempt,
                    # so only emit the EventBus RunSummary here.
                    self._ceremony.emit_partial_summary(run_id=run.run_id, result=result, start_time=run_start_time)
                else:
                    self._ceremony.emit_failed_ceremony(run.run_id, factory, run_start_time, token=coordination_token)
                    # Seat hygiene: after the FAILED finalize succeeded.
                    factory.run_coordination.release_seat(token=coordination_token, now=datetime.now(UTC))
            raise  # CRITICAL: Always re-raise - observability doesn't suppress errors
        finally:
            # ADR-030 §A.3: safety-net stop (idempotent) — covers any exit
            # path that did not already stop the thread (e.g. an exception
            # raised before any except handler ran release_seat).
            _heartbeat.stop()
            self._ceremony.safe_flush_telemetry()
