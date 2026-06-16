"""Core Orchestrator class for pipeline execution.

Coordinates:
- Run initialization
- Source loading
- Row processing
- Sink writing
- Run completion
- Post-run audit export (when configured)

The Orchestrator is the main entry point for running ELSPETH pipelines.
It delegates to focused helper modules for:
- Validation: Route and sink validation (validation.py)
- Export: Landscape export functionality (export.py)
- Aggregation: Timeout and flush handling (aggregation.py)
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.events import EventBusProtocol
    from elspeth.engine.orchestrator.types import TelemetryManagerProtocol

import elspeth.engine.executors.declaration_contract_bootstrap  # noqa: F401
from elspeth.contracts import (
    ExportStatus,
    RunStatus,
    SecretResolutionInput,
    SinkProtocol,
)
from elspeth.contracts.cli import ProgressEvent
from elspeth.contracts.config import RuntimeRetryConfig
from elspeth.contracts.coordination import (
    DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
    CoordinationToken,
    mint_worker_id,
)
from elspeth.contracts.declaration_contracts import (
    EXPECTED_CONTRACT_SITES,
    contract_sites,
    declaration_registry_is_frozen,
    freeze_declaration_registry,
    registered_declaration_contracts,
)
from elspeth.contracts.errors import (
    GracefulShutdownError,
    JoinRefusedError,
    OrchestrationInvariantError,
)
from elspeth.contracts.events import (
    PhaseAction,
    PhaseChanged,
    PhaseCompleted,
    PhaseStarted,
    PipelinePhase,
    RunCompletionStatus,
    RunFinished,
    RunStarted,
    RunSummary,
)
from elspeth.contracts.tier_registry import freeze_tier_registry
from elspeth.contracts.types import (
    AggregationName,
    CoalesceName,
    GateName,
    NodeID,
    SinkName,
)
from elspeth.core.canonical import stable_hash
from elspeth.core.config import resolve_config
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape._helpers import generate_id
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine._best_effort import best_effort

# Import module functions from orchestrator submodules
from elspeth.engine.orchestrator.ceremony import RunCeremony
from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator
from elspeth.engine.orchestrator.cleanup import cleanup_plugins
from elspeth.engine.orchestrator.export import (
    export_landscape,
)
from elspeth.engine.orchestrator.heartbeat import RunHeartbeatThread
from elspeth.engine.orchestrator.landscape_registration import (
    register_nodes_with_landscape,
)
from elspeth.engine.orchestrator.outcomes import accumulate_row_outcomes
from elspeth.engine.orchestrator.resume import ResumeCoordinator
from elspeth.engine.orchestrator.run_core import RunExecutionCore
from elspeth.engine.orchestrator.run_status import (
    assert_terminal_counter_parity,
    cli_completion_for,
    derive_terminal_status_from_audit,
)
from elspeth.engine.orchestrator.runtime_preflight import run_transform_runtime_preflights
from elspeth.engine.orchestrator.shutdown import shutdown_handler_context
from elspeth.engine.orchestrator.source_iteration import SourceIterationDriver
from elspeth.engine.orchestrator.types import (
    ExecutionCounters,
    GraphArtifacts,
    LoopContext,
    LoopResult,
    PipelineConfig,
    RunResult,
)
from elspeth.engine.orchestrator.validation import (
    validate_route_destinations,
    validate_sink_failsink_destinations,
    validate_source_quarantine_destination,
    validate_transform_error_sinks,
)
from elspeth.engine.retry import RetryManager
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from elspeth.contracts import ResumePoint
    from elspeth.contracts.config.runtime import RuntimeCheckpointConfig, RuntimeConcurrencyConfig
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.config import ElspethSettings
    from elspeth.core.dependency_config import PreflightResult
    from elspeth.core.rate_limit import RateLimitRegistry
    from elspeth.engine.clock import Clock

slog = structlog.get_logger(__name__)


class _RunFailedWithPartialResultError(Exception):
    """Internal wrapper used to move partial counters to outer failure handlers."""

    def __init__(self, original_error: Exception, partial_result: RunResult) -> None:
        super().__init__(str(original_error))
        self.original_error = original_error
        self.partial_result = partial_result
        self.original_traceback = original_error.__traceback__


def prepare_for_run() -> None:
    """Assert framework invariants and freeze both registries at bootstrap.

    This is the canonical bootstrap entry point (ADR-010 §Decision 3). It must
    be called AFTER all plugin modules have been imported (so module-level
    side-effects like ``register_declaration_contract`` have fired) and BEFORE
    any DAG node begins execution.

    The normal import chain guarantees this ordering in production:
    - ``orchestrator/core.py`` imports
      ``engine.executors.declaration_contract_bootstrap`` at module level.
    - ``declaration_contract_bootstrap.py`` imports every production contract
      module with a module-level ``register_declaration_contract(...)`` call.
    - Each imported contract module registers its contract as a module-level
      side-effect.

    If the declaration registry is empty at this point, the import chain is
    broken — this is an import-order bug, not a runtime configuration error.
    Crashing here prevents the framework from running silently without any
    runtime VAL checks active (the exact failure mode ADR-010 was designed to
    prevent).

    Raises:
        RuntimeError: every registered ``(contract_name, dispatch_site)``
            pair does not exactly equal the pairs in
            ``EXPECTED_CONTRACT_SITES`` (N1 per-site manifest extension,
            ADR-010 §Semantics amendment 2026-04-20). The message names
            every missing and every extra contract or site so the failure
            is self-diagnosing. Indicates one of: an import-order bug
            (contract module not imported), manifest drift (contract
            registered without a manifest entry), or site drift
            (contract's ``@implements_dispatch_site`` markers disagree
            with the manifest).
    """
    # Short-circuit if the registry is already frozen — bootstrap already ran.
    # Idempotency is required because Orchestrator.run() can be called multiple
    # times in a single process (e.g. test suites). The manifest-equality
    # assertion only needs to fire ONCE, on the first call; subsequent calls
    # trust that the previous freeze was performed after a successful
    # assertion.
    #
    # The ``_clear_registry_for_tests()`` helper resets ``_FROZEN = False``, so
    # test isolation that clears and repopulates the registry will still trigger
    # the assertion on the next call.
    if declaration_registry_is_frozen():
        return

    # ADR-010 §Decision 3 manifest gate, extended by §H2 landing scope N1:
    # Assert SET EQUALITY between every registered (contract_name,
    # dispatch_site) pair and every pair in ``EXPECTED_CONTRACT_SITES``
    # BEFORE freezing. The original C2 closure checked contract-name
    # equality; N1 tightens this to per-(name, site) equality so a
    # contract registering for the wrong site (or silently no-opping at a
    # site it claims to cover) is detected at bootstrap, not masked until
    # first row.
    #
    # Every plugin behaviour recorded as "compliant" (no violation raised)
    # must be evidence of every applicable contract's method having been
    # invoked — under audit-complete semantics (ADR-010 §Semantics) absence of violation
    # means "checked and passed," which is only true if the dispatcher
    # actually dispatched to the contract for its claimed sites. The N1
    # manifest closes the (name, site) drift vector the C2 set-of-names
    # manifest missed.
    contracts = registered_declaration_contracts()
    registered_sites: dict[str, frozenset[str]] = {c.name: frozenset(contract_sites(c)) for c in contracts}
    expected_sites: dict[str, frozenset[str]] = {name: frozenset(sites) for name, sites in EXPECTED_CONTRACT_SITES.items()}
    if registered_sites != expected_sites:
        # Compose a self-diagnosing message naming every drifted (name, site)
        # pair. Five mutually exclusive drift classes are surfaced:
        #   * name missing (manifest claims, nothing registered)
        #   * name extra (registered, manifest absent)
        #   * per-name: sites missing (contract registered with fewer sites
        #     than manifest declares)
        #   * per-name: sites extra (contract registered with more sites
        #     than manifest declares)
        #   * per-name: site-set mismatch (disjoint)
        expected_names = frozenset(expected_sites.keys())
        registered_names = frozenset(registered_sites.keys())
        missing_names = expected_names - registered_names
        extra_names = registered_names - expected_names
        site_drift_lines: list[str] = []
        for name in sorted(expected_names & registered_names):
            expected_for_name = expected_sites[name]
            registered_for_name = registered_sites[name]
            if expected_for_name == registered_for_name:
                continue
            missing_sites = expected_for_name - registered_for_name
            extra_sites = registered_for_name - expected_for_name
            site_drift_lines.append(
                f"  {name!r}: expected_sites={sorted(expected_for_name)!r}, "
                f"registered_sites={sorted(registered_for_name)!r}, "
                f"missing={sorted(missing_sites)!r}, extra={sorted(extra_sites)!r}"
            )
        raise RuntimeError(
            "Declaration contract registry mismatch at orchestrator bootstrap "
            "(ADR-010 §Decision 3 manifest gate + §H2 landing scope N1).\n"
            f"  Expected (manifest):  {sorted((n, sorted(s)) for n, s in expected_sites.items())!r}\n"
            f"  Registered:           {sorted((n, sorted(s)) for n, s in registered_sites.items())!r}\n"
            f"  Missing names (not registered but in manifest): {sorted(missing_names)!r}\n"
            f"  Extra names  (registered but not in manifest): {sorted(extra_names)!r}\n"
            "  Per-name site drift:\n" + ("\n".join(site_drift_lines) if site_drift_lines else "    (none)") + "\n"
            "\n"
            "If a name is missing: the contract's module-level "
            "register_declaration_contract(...) call did not fire. Check for "
            "a conditional import that skipped it, or an import-order bug "
            "where the module was not imported before prepare_for_run().\n"
            "If a name is extra: a contract was registered without being "
            "added to EXPECTED_CONTRACT_SITES. Update the manifest in the "
            "same commit as the registration.\n"
            "If per-name sites drift: the contract's "
            "@implements_dispatch_site(...) markers disagree with "
            "EXPECTED_CONTRACT_SITES. Either fix the markers or update the "
            "manifest (and run scripts/cicd/enforce_contract_manifest.py to "
            "confirm MC3a/b/c are clean).\n"
            "\n"
            "A silent runtime VAL disable is exactly the failure mode ADR-010 "
            "was designed to prevent — extended to per-site coverage under "
            "the §Semantics amendment (2026-04-20)."
        )
    freeze_declaration_registry()
    freeze_tier_registry()


class Orchestrator:
    """Orchestrates full pipeline runs.

    Manages the complete lifecycle:
    1. Begin run in Landscape
    2. Register all nodes (and set node_id on each plugin instance)
    3. Load rows from source
    4. Process rows through transforms
    5. Write to sinks
    6. Complete run

    The Orchestrator sets node_id on each plugin instance AFTER registering
    it with Landscape. This is part of the plugin protocol contract - all
    plugins define node_id: str | None and the orchestrator populates it.
    """

    def __init__(
        self,
        db: LandscapeDB,
        *,
        event_bus: EventBusProtocol | None = None,
        canonical_version: str = "sha256-rfc8785-v1",
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_config: RuntimeCheckpointConfig | None = None,
        clock: Clock | None = None,
        rate_limit_registry: RateLimitRegistry | None = None,
        concurrency_config: RuntimeConcurrencyConfig | None = None,
        telemetry_manager: TelemetryManagerProtocol | None = None,
        coalesce_completed_keys_limit: int = 10000,
    ) -> None:
        from elspeth.core.events import NullEventBus
        from elspeth.engine.clock import DEFAULT_CLOCK

        self._db = db
        self._events = event_bus if event_bus is not None else NullEventBus()
        self._canonical_version = canonical_version
        self._span_factory = SpanFactory()
        self._checkpoint_manager = checkpoint_manager
        self._clock = clock if clock is not None else DEFAULT_CLOCK
        self._rate_limit_registry = rate_limit_registry
        self._concurrency_config = concurrency_config
        self._coalesce_completed_keys_limit = coalesce_completed_keys_limit
        self._telemetry = telemetry_manager  # Optional, disabled by default
        self._ceremony = RunCeremony(events=self._events, telemetry=self._telemetry)
        self._checkpoints = CheckpointCoordinator(checkpoint_manager=checkpoint_manager, checkpoint_config=checkpoint_config)
        self._source_driver = SourceIterationDriver(events=self._events, span_factory=self._span_factory, ceremony=self._ceremony)
        self._run_core = RunExecutionCore(
            ceremony=self._ceremony,
            checkpoints=self._checkpoints,
            span_factory=self._span_factory,
            clock=self._clock,
            concurrency_config=self._concurrency_config,
            rate_limit_registry=self._rate_limit_registry,
            coalesce_completed_keys_limit=self._coalesce_completed_keys_limit,
            telemetry=self._telemetry,
        )
        self._resume_coordinator = ResumeCoordinator(
            db=self._db,
            events=self._events,
            ceremony=self._ceremony,
            checkpoints=self._checkpoints,
            run_core=self._run_core,
            checkpoint_manager=self._checkpoint_manager,
        )

    def _initialize_database_phase(
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

    def _execute_export_phase(
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
    ) -> RunResult:
        """Execute a pipeline run.

        Args:
            config: Pipeline configuration with plugins
            graph: Pre-validated execution graph (required)
            settings: Full settings (for post-run hooks like export)
            payload_store: PayloadStore for persisting source row payloads.
            secret_resolutions: Optional secret resolution records from
                load_secrets_from_config(). Recorded in audit trail after run creation.
            preflight_results: Optional pre-flight results (dependency runs and
                commencement gates) from bootstrap_and_run(). Recorded in audit
                trail after run creation.
            shutdown_event: Optional pre-created shutdown event for testing.
                Skips signal handler installation when provided.
            sink_factory: Creates a fresh sink instance by name. Required when
                landscape export is enabled (the pipeline's sinks are already
                closed by the time export runs).
            run_id: Optional caller-supplied Landscape run ID. When omitted,
                Landscape generates a run ID.
            initiated_by_user_id: Optional authenticated web user that initiated the run.
            auth_provider_type: Optional auth provider namespace for the initiating user.

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
        factory, run, coordination_token = self._initialize_database_phase(
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
                result = self._execute_run(
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
            run_duration_ms = (time.perf_counter() - run_start_time) * 1000
            self._ceremony.emit_telemetry(
                RunFinished(
                    timestamp=datetime.now(UTC),
                    run_id=run.run_id,
                    status=terminal_status,
                    row_count=result.rows_processed,
                    duration_ms=run_duration_ms,
                )
            )

            # EXPORT phase - post-run landscape export (if enabled)
            if settings is not None and settings.landscape.export.enabled:
                if sink_factory is None:
                    raise ValueError(
                        "Export is enabled but no sink_factory was provided to orchestrator.run(). "
                        "The caller must supply a sink_factory so the export phase can create "
                        "a fresh sink instance (the pipeline's sinks are already closed)."
                    )
                self._execute_export_phase(factory, run.run_id, settings, sink_factory)

            # Emit RunSummary event with final metrics.  Map the new
            # terminal status onto the CLI exit-code taxonomy via
            # ``cli_completion_for`` so the operator-facing CLI summary
            # remains coherent with /api/runs/{rid}.
            cli_status, exit_code = cli_completion_for(terminal_status)
            total_duration = time.perf_counter() - run_start_time
            self._events.emit(
                RunSummary(
                    run_id=run.run_id,
                    status=cli_status,
                    total_rows=result.rows_processed,
                    succeeded=result.rows_succeeded,
                    failed=result.rows_failed,
                    quarantined=result.rows_quarantined,
                    duration_seconds=total_duration,
                    exit_code=exit_code,
                    routed_success=result.rows_routed_success,
                    routed_failure=result.rows_routed_failure,
                    routed_destinations=tuple(result.routed_destinations.items()),
                )
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
                    total_duration = time.perf_counter() - run_start_time
                    self._events.emit(
                        RunSummary(
                            run_id=run.run_id,
                            status=RunCompletionStatus.PARTIAL,
                            total_rows=result.rows_processed,
                            succeeded=result.rows_succeeded,
                            failed=result.rows_failed,
                            quarantined=result.rows_quarantined,
                            duration_seconds=total_duration,
                            exit_code=1,
                            routed_success=result.rows_routed_success,
                            routed_failure=result.rows_routed_failure,
                            routed_destinations=tuple(result.routed_destinations.items()),
                        )
                    )
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
                    total_duration = time.perf_counter() - run_start_time
                    self._events.emit(
                        RunSummary(
                            run_id=run.run_id,
                            status=RunCompletionStatus.PARTIAL,
                            total_rows=result.rows_processed,
                            succeeded=result.rows_succeeded,
                            failed=result.rows_failed,
                            quarantined=result.rows_quarantined,
                            duration_seconds=total_duration,
                            exit_code=1,
                            routed_success=result.rows_routed_success,
                            routed_failure=result.rows_routed_failure,
                            routed_destinations=tuple(result.routed_destinations.items()),
                        )
                    )
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

    def _register_graph_nodes_and_edges(
        self,
        factory: RecorderFactory,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
    ) -> GraphArtifacts:
        """Register all graph nodes and edges in Landscape. Returns artifacts for subsequent phases.

        Performs the GRAPH phase:
        1. Build node_to_plugin mapping from config
        2. Register each node with Landscape (metadata, determinism, schema)
        3. Register edges and build edge_map
        4. Validate route destinations, error sinks, quarantine destinations

        Args:
            factory: RecorderFactory for audit trail
            run_id: Run identifier
            config: Pipeline configuration
            graph: Execution graph

        Returns:
            GraphArtifacts with edge_map, source_id, and all ID mappings
        """

        # Get execution order from graph
        execution_order = graph.topological_order()

        # Build node_id -> plugin instance mapping for metadata extraction.
        source_id_map: dict[str, NodeID] = {}
        for candidate_source_id in graph.get_sources():
            source_info = graph.get_node_info(candidate_source_id)
            # Per ADR-025 §2, the DAG builder unconditionally sets
            # ``source_name`` on every source node. A missing key would
            # collide ``source_id_map["source"] = ...`` across multiple
            # sources, silently overwriting earlier entries.
            if "source_name" not in source_info.config:
                raise OrchestrationInvariantError(
                    f"DAG source node {candidate_source_id!r} is missing 'source_name' in its config. "
                    f"Per ADR-025 §2 the DAG builder MUST set source_name on every source node. "
                    f"This is a graph-construction bug — node config keys: {sorted(source_info.config.keys())}."
                )
            source_name = str(source_info.config["source_name"])
            source_id_map[source_name] = candidate_source_id
        source_id = next(iter(source_id_map.values()))
        transform_id_map: dict[int, NodeID] = graph.get_transform_id_map()
        sink_id_map: dict[SinkName, NodeID] = graph.get_sink_id_map()
        config_gate_id_map: dict[GateName, NodeID] = graph.get_config_gate_id_map()
        aggregation_id_map: dict[AggregationName, NodeID] = graph.get_aggregation_id_map()

        # Build node ID sets for special node types
        config_gate_node_ids: set[NodeID] = set(config_gate_id_map.values())
        aggregation_node_ids: set[NodeID] = set(aggregation_id_map.values())

        # Map plugin instances to their node IDs for metadata extraction
        # Config gates and coalesce nodes don't have plugin instances (they're structural)
        # Aggregation transforms DO have instances - they're in config.transforms with node_id set
        node_to_plugin: dict[NodeID, Any] = {}
        for source_name, source_node_id in source_id_map.items():
            node_to_plugin[source_node_id] = config.sources[source_name]
        for seq, transform in enumerate(config.transforms):
            if seq in transform_id_map:
                # Regular transform - mapped by sequence number
                node_to_plugin[transform_id_map[seq]] = transform
            elif transform.node_id is not None and NodeID(transform.node_id) in aggregation_node_ids:
                # Aggregation transform - has node_id set by CLI, not in transform_id_map
                node_to_plugin[NodeID(transform.node_id)] = transform
        for sink_name, sink in config.sinks.items():
            if SinkName(sink_name) in sink_id_map:
                node_to_plugin[sink_id_map[SinkName(sink_name)]] = sink
        coalesce_id_map: dict[CoalesceName, NodeID] = graph.get_coalesce_id_map()
        coalesce_node_ids: set[NodeID] = set(coalesce_id_map.values())

        # GRAPH phase - register nodes and edges in Landscape
        phase_start = time.perf_counter()
        try:
            self._events.emit(PhaseStarted(phase=PipelinePhase.GRAPH, action=PhaseAction.BUILDING))

            # Emit telemetry PhaseChanged - we now have run_id from begin_run
            self._ceremony.emit_telemetry(
                PhaseChanged(
                    timestamp=datetime.now(UTC),
                    run_id=run_id,
                    phase=PipelinePhase.GRAPH,
                    action=PhaseAction.BUILDING,
                )
            )

            # Register nodes with Landscape using graph's node IDs and actual plugin metadata
            register_nodes_with_landscape(
                factory,
                run_id,
                config,
                graph,
                execution_order,
                node_to_plugin,
                source_id,
                config_gate_node_ids,
                coalesce_node_ids,
            )
            self._record_declared_sources_ready(
                factory=factory,
                run_id=run_id,
                config=config,
                source_id_map=source_id_map,
            )

            # Register edges from graph - key by (from_node, label) for lookup
            # Gates return route labels, so edge_map is keyed by label
            edge_map: dict[tuple[NodeID, str], str] = {}

            for edge_info in graph.get_edges():
                edge = factory.data_flow.register_edge(
                    run_id=run_id,
                    from_node_id=edge_info.from_node,
                    to_node_id=edge_info.to_node,
                    label=edge_info.label,
                    mode=edge_info.mode,
                )
                # Key by edge label - gates return route labels, transforms use "continue"
                edge_map[(NodeID(edge_info.from_node), edge_info.label)] = edge.edge_id

            # Get route resolution map - maps (gate_node, label) -> "continue" | sink_name
            route_resolution_map = graph.get_route_resolution_map()

            # NOTE — value-source compliance is enforced at the entry-point
            # boundary, NOT here. The walker
            # (``engine/orchestrator/preflight.validate_value_source_compliance``)
            # runs inside ``runtime_factory.instantiate_plugins_from_config`` and
            # the composer/web-execution validate paths
            # (``web/execution/validation.validate_pipeline``,
            # ``web/execution/service._run_pipeline``). Every legitimate caller
            # that builds a ``PipelineConfig`` passes through one of those
            # surfaces, so by the time we reach ``Orchestrator.run`` the bundle
            # has already been gated. If you add a new entry point that
            # constructs a ``PipelineConfig`` directly (test harness,
            # programmatic API, etc.), call ``validate_value_source_compliance``
            # at that boundary too — the orchestrator does NOT re-validate
            # value-source declarations per run, and a bypassing entry point
            # would silently skip the check otherwise.
            #
            # Validate all route destinations BEFORE processing any rows
            # This catches config errors early instead of after partial processing
            # Note: config gates also add to route_resolution_map, validated the same way
            # Call module function directly (no wrapper method)
            validate_route_destinations(
                route_resolution_map=route_resolution_map,
                available_sinks=set(config.sinks.keys()),
                transform_id_map=transform_id_map,
                transforms=config.transforms,
                config_gate_id_map=config_gate_id_map,
                config_gates=config.gates,
            )

            # Validate transform error sink destinations
            # Call module function directly (no wrapper method)
            validate_transform_error_sinks(
                transforms=config.transforms,
                available_sinks=set(config.sinks.keys()),
            )

            # Validate source quarantine destination
            # Call module function directly (no wrapper method)
            for source in config.sources.values():
                validate_source_quarantine_destination(
                    source=source,
                    available_sinks=set(config.sinks.keys()),
                )

            # Validate sink failsink destinations

            sink_validation_stubs = {name: SimpleNamespace(on_write_failure=sink._on_write_failure) for name, sink in config.sinks.items()}
            sink_plugins = {name: sink.name for name, sink in config.sinks.items()}
            validate_sink_failsink_destinations(
                sink_configs=sink_validation_stubs,
                available_sinks=set(config.sinks.keys()),
                sink_plugins=sink_plugins,
            )

            self._events.emit(PhaseCompleted(phase=PipelinePhase.GRAPH, duration_seconds=time.perf_counter() - phase_start))
        except Exception as e:
            self._ceremony.emit_phase_error(PipelinePhase.GRAPH, e)
            raise  # CRITICAL: Always re-raise - graph validation failure is fatal

        return GraphArtifacts(
            edge_map=edge_map,
            source_id=source_id,
            source_id_map=source_id_map,
            sink_id_map=sink_id_map,
            transform_id_map=transform_id_map,
            config_gate_id_map=config_gate_id_map,
            coalesce_id_map=coalesce_id_map,
        )

    def _record_declared_sources_ready(
        self,
        *,
        factory: RecorderFactory,
        run_id: str,
        config: PipelineConfig,
        source_id_map: Mapping[str, NodeID],
    ) -> None:
        """Seed run_sources for every declared source before iteration starts.

        A hard kill between source lifecycles must leave audit evidence that
        later sources were declared but never exhausted. The source-specific
        loop updates the active source from ready -> loading -> exhausted ->
        loaded, or interrupted; unstarted later sources remain ready and resume
        refuses rather than fabricating source exhaustion.
        """
        for source_name, source_node_id in source_id_map.items():
            source = config.sources[source_name]
            factory.run_lifecycle.record_run_source(
                run_id=run_id,
                source_node_id=source_node_id,
                source_name=source_name,
                plugin_name=source.name,
                config_hash=stable_hash(source.config),
                source_schema_json=json.dumps(source.output_schema.model_json_schema()),
                schema_contract=source.get_schema_contract(),
                lifecycle_state="ready",
            )

    def _execute_run(
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
    ) -> RunResult:
        """Execute the run using the execution graph.

        Orchestrates the four phases: graph registration, context initialization,
        source+process loop, sink writes. Returns RunStatus.RUNNING — the public
        run() wrapper transitions to COMPLETED after finalize_run().

        Parameters
        ----------
        check_coordination_latch:
            Optional callable forwarded to the per-source processing loop.
            Pass ``RunHeartbeatThread.check_and_raise`` from the enclosing
            ``run()`` method so the drain loop surfaces
            :class:`~elspeth.contracts.errors.RunWorkerEvictedError` at each
            row boundary when the heartbeat thread detects seat deposition.
            ``None`` disables latch polling (non-coordinated runs).
        """
        self._checkpoints.set_active_graph(graph)

        # F1 design D4: sequence-0 run-start checkpoint. Written before any
        # source iteration so every checkpointing-enabled run carries a
        # topology baseline; a run with NO checkpoint row then genuinely
        # predates run-start checkpointing or ran with checkpointing
        # disabled (can_resume's missing-baseline refusal, Task 3.2).
        # The resume path does NOT write this — it rebases onto the
        # persisted sequence (ResumeCoordinator.resume -> rebase_sequence).
        # Failures propagate: no baseline means the run cannot checkpoint.
        self._checkpoints.checkpoint_run_start(run_id)

        # 1. Register graph nodes and edges
        artifacts = self._register_graph_nodes_and_edges(factory, run_id, config, graph)

        # 2. Initialize context + processor
        run_ctx = self._run_core.initialize_run_context(
            factory,
            run_id,
            config,
            graph,
            settings,
            artifacts,
            payload_store,
            shutdown_event=shutdown_event,
            coordination_token=coordination_token,
        )
        preflight_retry_manager = RetryManager(RuntimeRetryConfig.from_settings(settings.retry)) if settings is not None else None
        run_transform_runtime_preflights(
            factory,
            run_id,
            config,
            run_ctx.ctx,
            retry_manager=preflight_retry_manager,
            shutdown_event=shutdown_event,
        )

        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={name: [] for name in config.sinks},
            processor=run_ctx.processor,
            ctx=run_ctx.ctx,
            config=config,
            agg_transform_lookup=run_ctx.agg_transform_lookup,
            coalesce_executor=run_ctx.coalesce_executor,
            coalesce_node_map=run_ctx.coalesce_node_map,
        )

        try:
            # 3. Source + Process phase. This is sequential multi-source ingest:
            # each declared source is iterated in turn with the active
            # ``SourceProtocol`` passed explicitly into the loop.
            # YAML declaration order is the determinism anchor for cross-source
            # ``ingest_sequence`` assignment. The scheduler's concurrency
            # contract is worker-token concurrency, not concurrent source iteration.
            # Per ADR-025 §1, the prior synthetic
            # ``replace(config, source=..., sources={...})`` per-iteration
            # config-mutation pattern is deleted.
            loop_result: LoopResult | None = None
            source_items = tuple(artifacts.source_id_map.items())
            for source_ordinal, (source_name, source_id) in enumerate(source_items):
                active_source = config.sources[source_name]
                source_loop_ctx = LoopContext(
                    counters=loop_ctx.counters,
                    pending_tokens=loop_ctx.pending_tokens,
                    processor=loop_ctx.processor,
                    ctx=loop_ctx.ctx,
                    config=config,
                    agg_transform_lookup=loop_ctx.agg_transform_lookup,
                    coalesce_executor=loop_ctx.coalesce_executor,
                    coalesce_node_map=loop_ctx.coalesce_node_map,
                )
                loop_result = self._source_driver.run_main_processing_loop(
                    source_loop_ctx,
                    factory,
                    run_id,
                    source_id,
                    artifacts.edge_map,
                    active_source_name=source_name,
                    active_source=active_source,
                    shutdown_event=shutdown_event,
                    flush_end_of_input=source_ordinal == len(source_items) - 1,
                    check_coordination_latch=check_coordination_latch,
                )
                if loop_result.interrupted:
                    break

            if loop_result is None:
                raise OrchestrationInvariantError("Pipeline has no sources to process")

            # 4b-pre. ADR-030 multi-worker: BEFORE checking for unresolved scheduler
            # work, wait for peer followers to finish any in-flight LEASED items.
            # A follower that claimed an item just before the leader's source loop
            # exited will still hold a LEASED row (pending_sink_name IS NULL) that
            # has_unresolved_scheduler_work() counts as unresolved.  Waiting here
            # ensures followers complete their claims (LEASED → PENDING_SINK) before
            # the invariant check fires.  In the single-worker case,
            # has_peer_active_leases() returns False immediately and the loop is
            # skipped.
            #
            # The wait is BOUNDED by the active item lease plus the stall budget:
            # a live peer gets the full lease window to finish legitimate work,
            # while a wedged-but-alive peer that keeps its lease refreshed must
            # not hang a deposed/interrupted leader forever.
            # Each iteration also (a) honours the in-scope shutdown_event (SIGINT)
            # and check_coordination_latch (epoch deposition) so the leader can break
            # out, and (b) drives lease maintenance so a peer that DIED mid-lease is
            # actively reaped to READY within the liveness window rather than waiting
            # out the full item TTL.  On timeout we fall through to the existing
            # has_unresolved_scheduler_work raise, which names the still-leased peers.
            def _shutdown_during_wait() -> GracefulShutdownError:
                """Build the canonical INTERRUPTED signal from the live counters.

                A SIGINT observed while waiting on / draining peer work must surface
                the same resumable GracefulShutdownError the source loop and sink
                flush raise (counter-bearing, run_id-scoped), not a bare message.
                """
                _c = loop_ctx.counters
                return GracefulShutdownError(
                    rows_processed=_c.rows_processed,
                    run_id=run_id,
                    rows_succeeded=_c.rows_succeeded,
                    rows_failed=_c.rows_failed,
                    rows_quarantined=_c.rows_quarantined,
                    rows_routed_success=_c.rows_routed_success,
                    rows_routed_failure=_c.rows_routed_failure,
                    routed_destinations=dict(_c.routed_destinations),
                )

            if not loop_result.interrupted:
                peer_wait_seconds = loop_ctx.processor.peer_lease_wait_budget_seconds()
                peer_wait_deadline = time.monotonic() + peer_wait_seconds
                while loop_ctx.processor.has_peer_active_leases():
                    # SIGINT during the wait: surface the graceful-shutdown path
                    # rather than spinning.
                    if shutdown_event is not None and shutdown_event.is_set():
                        raise _shutdown_during_wait()
                    # Epoch deposition during the wait: check_and_raise surfaces
                    # RunWorkerEvictedError so the deposed leader runs its
                    # INTERRUPTED ceremony instead of spinning.
                    if check_coordination_latch is not None:
                        check_coordination_latch()
                    # Actively reap a dead peer's expired lease (recovers it to
                    # READY within the liveness window).
                    loop_ctx.processor.reap_expired_peer_leases()
                    if not loop_ctx.processor.has_peer_active_leases():
                        break
                    if time.monotonic() >= peer_wait_deadline:
                        still_leased = loop_ctx.processor.peer_active_lease_owners()
                        slog.warning(
                            "Bounded peer-lease wait timed out; falling through to the unresolved-work invariant",
                            run_id=run_id,
                            still_leased_peers=list(still_leased),
                            waited_seconds=peer_wait_seconds,
                        )
                        break
                    time.sleep(0.5)

            if not loop_result.interrupted and loop_ctx.processor.has_unresolved_scheduler_work():
                active_work = "; ".join(loop_ctx.processor.summarize_unresolved_scheduler_work()) or "<unknown>"
                raise OrchestrationInvariantError(
                    f"Run '{run_ctx.processor.run_id}' left non-terminal scheduler work after final source flush. "
                    "Blocked or READY scheduler state must be resolved before run completion. "
                    f"Active scheduler work: {active_work}."
                )

            # 4. Sink writes — outside source_load track_operation context.
            # Each sink write has its own track_operation (sink_write) in SinkExecutor.
            self._run_core.flush_and_write_sinks(
                factory,
                run_id,
                loop_ctx,
                artifacts.sink_id_map,
                artifacts.edge_map,
                loop_result.interrupted,
                on_token_written_factory=self._checkpoints.make_checkpoint_after_sink_factory(run_id, run_ctx.processor),
            )

            # 4b. ADR-030 multi-worker: after the leader's own sink writes are done
            # (all leader PENDING_SINK rows are now TERMINAL), drain the PENDING_SINK
            # rows produced by follower workers and write those to sinks.  In the
            # single-worker case, has_scheduled_work() returns False immediately
            # (no follower PENDING_SINK rows exist) and the loop body never runs.
            #
            # LOOP (not single-pass): a follower that transitions LEASED→PENDING_SINK
            # AFTER drain_scheduled_work's claim loop returns would leave a late
            # PENDING_SINK row.  We re-drain until BOTH has_peer_active_leases() (no
            # peer still holds an in-flight lease that could become PENDING_SINK) and
            # has_scheduled_work() (no undrained PENDING_SINK row) are false.  The
            # wait is bounded by the same liveness-multiple deadline as 4b-pre and
            # honours shutdown/deposition each iteration; on timeout we stop draining
            # and rely on complete_run's quiescence arm as the backstop (a residual
            # PENDING_SINK row makes the run FAIL loudly — the correct, resumable
            # exactly-once fail-direction — never a silent lost row).
            if not loop_result.interrupted:
                drain_deadline = time.monotonic() + (3.0 * DEFAULT_RUN_LIVENESS_WINDOW_SECONDS)
                while loop_ctx.processor.has_peer_active_leases() or loop_ctx.processor.has_scheduled_work():
                    if shutdown_event is not None and shutdown_event.is_set():
                        raise _shutdown_during_wait()
                    if check_coordination_latch is not None:
                        check_coordination_latch()

                    if loop_ctx.processor.has_scheduled_work():
                        follower_results = loop_ctx.processor.drain_scheduled_work(loop_ctx.ctx)
                        if follower_results:
                            # Clear pending_tokens before re-flush: write_pending_to_sinks
                            # does NOT consume entries (it iterates without clearing), so the
                            # leader's already-written tokens remain.  Accumulating follower
                            # results on top of them would cause the next flush to re-write
                            # every leader token → UNIQUE constraint on node_states.
                            for _sink_list in loop_ctx.pending_tokens.values():
                                _sink_list.clear()
                            accumulate_row_outcomes(follower_results, loop_ctx.counters, loop_ctx.pending_tokens)
                            self._run_core.flush_and_write_sinks(
                                factory,
                                run_id,
                                loop_ctx,
                                artifacts.sink_id_map,
                                artifacts.edge_map,
                                interrupted_by_shutdown=False,
                                on_token_written_factory=self._checkpoints.make_checkpoint_after_sink_factory(run_id, run_ctx.processor),
                            )
                            # Made progress this iteration; re-check immediately.
                            continue

                    # No drainable work this pass but a peer still holds a lease that
                    # could yet produce a PENDING_SINK row.  Actively reap a dead
                    # peer, then wait briefly — bounded.
                    if not (loop_ctx.processor.has_peer_active_leases() or loop_ctx.processor.has_scheduled_work()):
                        break
                    loop_ctx.processor.reap_expired_peer_leases()
                    if not (loop_ctx.processor.has_peer_active_leases() or loop_ctx.processor.has_scheduled_work()):
                        break
                    if time.monotonic() >= drain_deadline:
                        slog.warning(
                            "Bounded follower-drain wait timed out; relying on complete_run quiescence backstop",
                            run_id=run_id,
                            still_leased_peers=list(loop_ctx.processor.peer_active_lease_owners()),
                            has_scheduled_work=loop_ctx.processor.has_scheduled_work(),
                            waited_seconds=3.0 * DEFAULT_RUN_LIVENESS_WINDOW_SECONDS,
                        )
                        break
                    time.sleep(0.5)

            # ADR-019 Phase 4: deferred cross-table invariant sweep.
            #
            # AUDIT-TRAIL DURABILITY CONTRACT:
            # 1. The run is still RUNNING here; successful terminal finalization
            #    has not executed.
            # 2. If this raises AuditIntegrityError, the exception propagates to
            #    the public run() failure ceremony, which finalizes the run as
            #    FAILED and re-raises the original exception.
            # 3. The offending token_outcomes/batch rows are evidence and are
            #    not deleted by the sweep.
            # 4. GracefulShutdownError skips this naturally because sink flush
            #    raises before this post-sink call site.
            factory.data_flow.sweep_deferred_invariants_or_crash(run_id)

            # 5. Final progress + PROCESS phase completion — AFTER sink writes
            # so these events reflect concrete, durable results. On shutdown,
            # _flush_and_write_sinks raises GracefulShutdownError before we
            # reach here — matching the pre-extraction behavior where the
            # shutdown raise prevented progress/PhaseCompleted emission.
            progress_interval = 100
            current_time = time.perf_counter()
            time_since_last_progress = current_time - loop_result.last_progress_time
            if loop_ctx.counters.rows_processed % progress_interval != 0 or time_since_last_progress >= 1.0:
                elapsed = current_time - loop_result.start_time
                self._events.emit(
                    ProgressEvent(
                        rows_processed=loop_ctx.counters.rows_processed,
                        # elspeth-5069612f3c — rows_routed split. See the
                        # earlier emitter in this file for the full rationale;
                        # this final-progress emission must match so the last
                        # streaming snapshot before terminal events agrees
                        # with the forthcoming CompletedData payload.
                        rows_succeeded=loop_ctx.counters.rows_succeeded,
                        rows_failed=loop_ctx.counters.rows_failed,
                        rows_quarantined=loop_ctx.counters.rows_quarantined,
                        rows_routed_success=loop_ctx.counters.rows_routed_success,
                        rows_routed_failure=loop_ctx.counters.rows_routed_failure,
                        elapsed_seconds=elapsed,
                    )
                )

            self._events.emit(PhaseCompleted(phase=PipelinePhase.PROCESS, duration_seconds=current_time - loop_result.phase_start))
        except GracefulShutdownError:
            raise
        except Exception as exc:
            raise _RunFailedWithPartialResultError(
                original_error=exc,
                partial_result=loop_ctx.counters.to_run_result(run_id, status=RunStatus.FAILED),
            ) from exc

        finally:
            cleanup_plugins(config, run_ctx.ctx, include_source=True)

        self._checkpoints.set_active_graph(None)
        return loop_ctx.counters.to_run_result(run_id, status=RunStatus.RUNNING)

    def resume(
        self,
        resume_point: ResumePoint,
        config: PipelineConfig,
        graph: ExecutionGraph,
        *,
        payload_store: PayloadStore,
        settings: ElspethSettings | None = None,
        shutdown_event: threading.Event | None = None,
    ) -> RunResult:
        """Resume a failed run from a checkpoint.

        Delegates to :class:`ResumeCoordinator`, which owns the resume-path
        orchestration extracted from this class. The public signature is the
        stable contract; the implementation lives in resume.py.
        """
        return self._resume_coordinator.resume(
            resume_point,
            config,
            graph,
            payload_store=payload_store,
            settings=settings,
            shutdown_event=shutdown_event,
        )

    @staticmethod
    def _join_preflight(db: LandscapeDB, run_id: str) -> None:
        """§B.1 step 0: filesystem write-access preflight for the join path.

        Verifies that the joining process can write to the DB file, its
        containing directory, and any existing ``-wal`` / ``-shm`` sidecars
        BEFORE touching the registry.  Raises :class:`JoinRefusedError` with
        an actionable path and description on any failure so the operator
        knows exactly what permission is missing.

        SQLCipher: passphrase is verified implicitly by LandscapeDB's PRAGMA
        probe at open time (the engine construction that precedes this call
        fails with an opaque SQLite error if the passphrase is wrong).  This
        preflight only handles filesystem-level write access.

        Non-SQLite (Postgres): no filesystem sidecars exist; skip silently.
        """
        url = db.connection_string
        if not url.startswith("sqlite"):
            # Postgres / other backends: no filesystem sidecars to check.
            return

        # Use SQLAlchemy's make_url to reliably extract the database path
        # (handles sqlite:///path, sqlite:////abs/path, and :memory: forms).
        from sqlalchemy.engine.url import make_url as _make_url

        parsed_url = _make_url(url)
        db_file = parsed_url.database
        if db_file is None or db_file in ("", ":memory:"):
            return  # in-memory: no filesystem checks needed

        db_path = Path(db_file)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path

        checks: list[tuple[Path, str]] = [
            (db_path, "write access to the audit DB file"),
            (db_path.parent, "write access to the audit DB directory"),
        ]
        for sidecar_suffix in ("-wal", "-shm"):
            sidecar = db_path.parent / (db_path.name + sidecar_suffix)
            if sidecar.exists():
                checks.append((sidecar, f"write access to {sidecar_suffix} sidecar"))

        for path, description in checks:
            if not os.access(path, os.W_OK):
                raise JoinRefusedError(
                    run_id,
                    f"filesystem preflight failed: {description} at {path} — "
                    "ensure the joining process has write permission "
                    "(shared group + group-writable state directory required for "
                    "cross-uid joins; see ADR-030 §B.1 step 0 and the slice-6 runbook)",
                )

    def join_run(
        self,
        run_id: str,
        settings: ElspethSettings,
        *,
        now: datetime | None = None,
        window_seconds: float | None = None,
    ) -> str:
        """§B.1: atomic follower admission — new public entry point (ADR-030).

        NOT a ``resume()`` variant.  ``resume()`` keeps refusing
        RUNNING-with-live-leader; ``join_run`` is the cooperative follower
        attach path.

        Steps performed (design §B.1):

        0. Filesystem preflight — write access to DB file + dir + any
           existing ``-wal``/``-shm`` sidecars.  Raises :class:`JoinRefusedError`
           naming the path if the joining process cannot write.

        1. DB is already open through ``self._db`` (the caller's
           ``LandscapeDB`` carries the PRAGMA probe + epoch check — G28
           cross-process uniformity by construction).

        2. Atomic admission (one ``BEGIN IMMEDIATE`` transaction via
           :meth:`RunCoordinationRepository.admit_follower`):

           - ``SELECT runs.status, runs.config_hash`` — status must be
             ``RUNNING``, else :class:`JoinRefusedError`;
           - joiner's resolved settings hash must equal ``config_hash``,
             else refused (different pipeline ⇒ different graph + barrier
             keys);
           - ``run_coordination`` seat must be live
             (``leader_heartbeat_expires_at > now``), else refused
             ("no live leader — use ``elspeth resume``");
           - ``INSERT run_workers`` (role='follower', status='active') +
             ``worker_register`` event.  COMMIT.

        Args:
            run_id: Landscape run ID to join.
            settings: The joining process's resolved ``ElspethSettings``.
                Its ``stable_hash(resolve_config(settings))`` is compared
                to ``runs.config_hash``; they must be equal.
            now: Clock injection for tests (defaults to ``datetime.now(UTC)``).
            window_seconds: Heartbeat liveness window (defaults to
                :data:`~elspeth.contracts.coordination.DEFAULT_RUN_LIVENESS_WINDOW_SECONDS`).

        Returns:
            The minted ``worker_id`` string (``worker:{run_id}:{uuid4().hex}``)
            so the caller can construct a follower-mode ``RowProcessor`` with
            ``lease_owner=worker_id``.

        Raises:
            JoinRefusedError: Filesystem preflight failed, run is not RUNNING,
                config hash mismatch, or no live leader seat.
        """

        _now = now if now is not None else datetime.now(UTC)
        _window = window_seconds if window_seconds is not None else DEFAULT_RUN_LIVENESS_WINDOW_SECONDS

        # Step 0: filesystem preflight BEFORE touching the registry.
        self._join_preflight(self._db, run_id)

        # Step 1: DB already open through self._db (PRAGMA probe + epoch
        # check inherited at LandscapeDB construction — G28 uniformity).

        # Step 2: atomic admission via the slice-5 RunCoordinationRepository
        # surface.  The joiner computes its own config_hash from its settings
        # using the same stable_hash(resolve_config(settings)) recipe that
        # begin_run stored.
        joiner_config_hash = stable_hash(resolve_config(settings))
        worker_id = mint_worker_id(run_id)

        factory = RecorderFactory(self._db, payload_store=None)
        factory.run_coordination.admit_follower(
            run_id=run_id,
            worker_id=worker_id,
            config_hash=joiner_config_hash,
            now=_now,
            window_seconds=_window,
        )

        return worker_id
