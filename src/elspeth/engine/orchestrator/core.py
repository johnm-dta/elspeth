"""Core Orchestrator facade for pipeline execution.

The Orchestrator is the main entry point for running ELSPETH pipelines. Since
filigree elspeth-9e71ae82a4 it is a composition facade: ``__init__`` wires the
collaborating services and every public entry point delegates to exactly one
of them.

Phase ownership:
- Run lifecycle (bootstrap, DATABASE phase, heartbeat, finalize ordering,
  export, ceremonies): RunLifecycleCoordinator (run_lifecycle.py)
- GRAPH phase (node/edge registration + routing validation):
  GraphRegistrationService (graph_registration.py)
- Run-body phase sequencing (source loop, peer-lease wait, sink flush,
  follower drain): LeaderDrainCoordinator (leader_drain.py)
- Resume-path orchestration: ResumeCoordinator (resume.py)
- Follower admission (ADR-030 §B.1): JoinAdmissionService (join_admission.py)

The thin private delegators (``_initialize_database_phase``, ``_execute_run``,
``_register_graph_nodes_and_edges``) are deliberate test seams: they are passed
into the coordinators as bound methods AT CALL TIME, so class-level
monkeypatches and instance-level stubs on the Orchestrator keep intercepting
the corresponding phase.

Re-exports for the stable legacy import path ``…orchestrator.core``:
- prepare_for_run (bootstrap.py)
- _RunFailedWithPartialResultError (types.py)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import elspeth.engine.executors.declaration_contract_bootstrap  # noqa: F401
from elspeth.engine.orchestrator.bootstrap import prepare_for_run as prepare_for_run
from elspeth.engine.orchestrator.ceremony import RunCeremony
from elspeth.engine.orchestrator.checkpointing import CheckpointCoordinator
from elspeth.engine.orchestrator.graph_registration import GraphRegistrationService
from elspeth.engine.orchestrator.join_admission import JoinAdmissionService
from elspeth.engine.orchestrator.leader_drain import LeaderDrainCoordinator
from elspeth.engine.orchestrator.resume import ResumeCoordinator
from elspeth.engine.orchestrator.run_core import RunExecutionCore
from elspeth.engine.orchestrator.run_lifecycle import RunLifecycleCoordinator
from elspeth.engine.orchestrator.source_iteration import SourceIterationDriver
from elspeth.engine.orchestrator.types import (
    _RunFailedWithPartialResultError as _RunFailedWithPartialResultError,
)
from elspeth.engine.spans import SpanFactory

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    from elspeth.contracts import (
        ResumePoint,
        SecretResolutionInput,
        SinkProtocol,
    )
    from elspeth.contracts.config.runtime import RuntimeCheckpointConfig, RuntimeConcurrencyConfig
    from elspeth.contracts.coordination import CoordinationToken
    from elspeth.contracts.payload_store import PayloadStore
    from elspeth.core.checkpoint import CheckpointManager
    from elspeth.core.config import ElspethSettings
    from elspeth.core.dag import ExecutionGraph
    from elspeth.core.dependency_config import PreflightResult
    from elspeth.core.events import EventBusProtocol
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.landscape.factory import RecorderFactory
    from elspeth.core.rate_limit import RateLimitRegistry
    from elspeth.engine.clock import Clock
    from elspeth.engine.orchestrator.types import (
        GraphArtifacts,
        PipelineConfig,
        RunResult,
        TelemetryManagerProtocol,
    )


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
        self._graph_registration = GraphRegistrationService(
            events=self._events,
            ceremony=self._ceremony,
        )
        self._leader_drain = LeaderDrainCoordinator(
            events=self._events,
            checkpoints=self._checkpoints,
            run_core=self._run_core,
            source_driver=self._source_driver,
        )
        self._run_lifecycle = RunLifecycleCoordinator(
            db=self._db,
            events=self._events,
            ceremony=self._ceremony,
            checkpoints=self._checkpoints,
            span_factory=self._span_factory,
            canonical_version=self._canonical_version,
        )
        self._join_admission = JoinAdmissionService(db=self._db)

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
        """DATABASE-phase delegator (test seam — see RunLifecycleCoordinator)."""
        return self._run_lifecycle.initialize_database_phase(
            config,
            payload_store,
            secret_resolutions,
            run_id=run_id,
            initiated_by_user_id=initiated_by_user_id,
            auth_provider_type=auth_provider_type,
            openrouter_catalog_sha256=openrouter_catalog_sha256,
            openrouter_catalog_source=openrouter_catalog_source,
        )

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

        Delegates to :class:`RunLifecycleCoordinator`, which owns the
        fresh-run lifecycle ordering extracted from this class. The public
        signature is the stable contract; the implementation lives in
        run_lifecycle.py.

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
        return self._run_lifecycle.run(
            config,
            graph,
            settings,
            payload_store=payload_store,
            secret_resolutions=secret_resolutions,
            preflight_results=preflight_results,
            shutdown_event=shutdown_event,
            sink_factory=sink_factory,
            run_id=run_id,
            initiated_by_user_id=initiated_by_user_id,
            auth_provider_type=auth_provider_type,
            openrouter_catalog_sha256=openrouter_catalog_sha256,
            openrouter_catalog_source=openrouter_catalog_source,
            # Bound AT CALL TIME (not construction) so monkeypatch.setattr on
            # the class and patch.object on this instance keep intercepting.
            initialize_database_phase=self._initialize_database_phase,
            execute_run=self._execute_run,
        )

    def _register_graph_nodes_and_edges(
        self,
        factory: RecorderFactory,
        run_id: str,
        config: PipelineConfig,
        graph: ExecutionGraph,
    ) -> GraphArtifacts:
        """GRAPH-phase delegator (test seam — see GraphRegistrationService)."""
        return self._graph_registration.register_graph_nodes_and_edges(factory, run_id, config, graph)

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
        """Run-body delegator (test seam — see LeaderDrainCoordinator).

        Returns RunStatus.RUNNING — the public run() wrapper transitions to
        COMPLETED after finalize_run().
        """
        return self._leader_drain.execute_run(
            factory,
            run_id,
            config,
            graph,
            settings,
            payload_store=payload_store,
            shutdown_event=shutdown_event,
            coordination_token=coordination_token,
            check_coordination_latch=check_coordination_latch,
            register_graph_nodes_and_edges=self._register_graph_nodes_and_edges,
        )

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

    def join_run(
        self,
        run_id: str,
        settings: ElspethSettings,
        *,
        now: datetime | None = None,
        window_seconds: float | None = None,
    ) -> str:
        """§B.1: atomic follower admission — public entry point (ADR-030).

        Delegates to :class:`JoinAdmissionService`, which owns the follower
        attach path (filesystem preflight + atomic admission) extracted from
        this class. The public signature is the stable contract; the
        implementation lives in join_admission.py.
        """
        return self._join_admission.join_run(
            run_id,
            settings,
            now=now,
            window_seconds=window_seconds,
        )
