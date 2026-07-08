"""Regression tests for Phase 0 orchestrator fixes.

#6: Resume leaves RUNNING — when _process_resumed_rows raises a non-shutdown
    exception, the run must be finalized as FAILED (not left as RUNNING).

#7: Plugin cleanup skipped — when _build_processor raises after on_start
    completes, _cleanup_plugins must still be called.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from elspeth.contracts import Checkpoint, NodeID, ResumedRow, ResumePoint, RoutingMode, RunStatus
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenOutcome
from elspeth.contracts.checkpoint import ResumeCheck
from elspeth.contracts.coordination import CoordinationSnapshot, CoordinationToken
from elspeth.contracts.enums import NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.events import RunSummary
from elspeth.contracts.payload_store import PayloadStore
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.plugin_protocols import SinkProtocol, SourceProtocol, TransformProtocol
from elspeth.contracts.run_result import RunResult
from elspeth.contracts.runtime_val_manifest import build_runtime_val_manifest
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.types import SinkName
from elspeth.core.canonical import canonical_json
from elspeth.core.checkpoint.manager import CheckpointManager
from elspeth.core.checkpoint.recovery import NonResumableRunError, RecoveryManager, ResumeWorkSet
from elspeth.core.config import AggregationSettings, ElspethSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.data_flow_repository import DataFlowRepository
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.run_lifecycle_repository import RunLifecycleRepository
from elspeth.engine.orchestrator import PipelineConfig, prepare_for_run
from elspeth.engine.orchestrator.cleanup import cleanup_plugins
from elspeth.engine.orchestrator.core import Orchestrator
from elspeth.engine.orchestrator.resume import run_resume_processing_loop, setup_resume_context
from elspeth.engine.orchestrator.run_state import LoopContext, LoopResult, ResumeState, _RunFailedWithPartialResultError
from elspeth.engine.orchestrator.types import ExecutionCounters
from elspeth.engine.processor import RowProcessor
from elspeth.testing import make_row_result, make_source_row
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.stores import MockPayloadStore


def _make_heartbeat_safe_token(run_id: str, mock_factory: MagicMock) -> CoordinationToken:
    """Return a valid CoordinationToken and configure mock_factory.run_coordination so
    the RunHeartbeatThread that resume() starts does not trip the latch.

    ADR-030 §A.3 (slice 4): resume() now always starts a RunHeartbeatThread.
    Unit tests that mock ``reconstruct_resume_state`` must supply a non-None
    coordination_token in the returned ResumeState AND configure the mock repo
    to return a healthy CoordinationSnapshot so the background thread's latch
    is never set.
    """
    worker_id = f"worker-test:{run_id}"
    token = CoordinationToken(run_id=run_id, worker_id=worker_id, leader_epoch=1)
    healthy_snapshot = CoordinationSnapshot(
        leader_worker_id=worker_id,
        leader_epoch=1,
        seat_live=True,
        worker_active=True,
    )
    mock_factory.run_coordination.worker_heartbeat.return_value = healthy_snapshot
    return token


def _make_orchestrator(db: LandscapeDB | None = None) -> Orchestrator:
    """Create an Orchestrator with minimal dependencies."""
    if db is None:
        db = make_landscape_db()
    return Orchestrator(db)


def _insert_failed_run(db: LandscapeDB, run_id: str) -> None:
    """Insert the FAILED ``runs`` row the resume-under-test claims to resume.

    The resume() entry guard (elspeth-2f23292372) re-checks the run's status
    in the real audit DB before any mutation, so these mock-heavy resume
    tests must persist the run they resume — a missing or non-resumable run
    is now refused at entry with NonResumableRunError.
    """
    from elspeth.core.landscape.schema import runs_table

    with db.connection() as conn:
        conn.execute(
            runs_table.insert().values(
                run_id=run_id,
                started_at=datetime.now(UTC),
                config_hash="cfg",
                settings_json="{}",
                canonical_version="sha256-rfc8785-v1",
                status=RunStatus.FAILED,
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )


def _admit_resume_point(orch: Orchestrator, resume_point: ResumePoint) -> Any:
    """Satisfy the resume() entry guard's read-only checkpoint checks.

    elspeth-5129406607: resume() refuses a resume point that is not the run's
    LATEST checkpoint or whose recorded topology diverges from the supplied
    graph. These tests pin DOWNSTREAM finalization behavior (the guard has
    its own pins in test_resume_entry_guard.py), so the guard is satisfied,
    not exercised: the stub manager serves the test's own checkpoint as the
    latest, and the returned context manager patches the compatibility
    validator to report compatible (the tests' graphs are mocks with no
    hashable topology).
    """
    manager = MagicMock(spec=CheckpointManager)
    manager.get_latest_checkpoint.return_value = resume_point.checkpoint
    orch._checkpoint_manager = manager
    orch._resume_coordinator._checkpoint_manager = manager
    return patch(
        "elspeth.engine.orchestrator.resume.CheckpointCompatibilityValidator.validate",
        return_value=ResumeCheck(can_resume=True),
    )


# Protocol-specced plugin doubles. ``SourceProtocol``/``SinkProtocol``/
# ``TransformProtocol`` declare ``name``/``node_id``/``on_success``/``config``
# only by annotation, so they are absent from ``dir()`` and a bare
# ``MagicMock(spec=...)`` rejects reads of them — and ``name`` cannot be seeded
# via the ``MagicMock`` constructor (it is the reserved repr-name kwarg). These
# factories spec against the real protocol (so ``isinstance`` holds and method
# typos are caught) and seed the ``name`` and ``node_id`` the orchestrator's
# cleanup hooks read back (lifecycle attribution requires ``node_id``).
def _specced_source(*, name: str = "source", **attrs: object) -> MagicMock:
    mock = MagicMock(spec=SourceProtocol)
    mock.name = name
    mock.node_id = name
    for key, value in attrs.items():
        setattr(mock, key, value)
    return mock


def _specced_sink(*, name: str = "sink", **attrs: object) -> MagicMock:
    mock = MagicMock(spec=SinkProtocol)
    mock.name = name
    mock.node_id = name
    for key, value in attrs.items():
        setattr(mock, key, value)
    return mock


def _specced_transform(*, name: str = "transform", **attrs: object) -> MagicMock:
    mock = MagicMock(spec=TransformProtocol)
    mock.name = name
    mock.node_id = name
    for key, value in attrs.items():
        setattr(mock, key, value)
    return mock


def _make_token_outcome(
    *,
    run_id: str,
    token_id: str,
    outcome: TerminalOutcome | None,
    path: TerminalPath,
    completed: bool = True,
    sink_name: str | None = None,
    batch_id: str | None = None,
    fork_group_id: str | None = None,
    join_group_id: str | None = None,
    expand_group_id: str | None = None,
    error_hash: str | None = None,
) -> TokenOutcome:
    return TokenOutcome(
        outcome_id=f"outcome-{token_id}-{path.value}",
        run_id=run_id,
        token_id=token_id,
        outcome=outcome,
        path=path,
        completed=completed,
        recorded_at=datetime.now(UTC),
        sink_name=sink_name,
        batch_id=batch_id,
        fork_group_id=fork_group_id,
        join_group_id=join_group_id,
        expand_group_id=expand_group_id,
        error_hash=error_hash,
    )


def _observed_contract(field_name: str, python_type: type) -> SchemaContract:
    """Create a locked first-row-inferred source contract for orchestrator tests."""
    return SchemaContract(
        mode="OBSERVED",
        fields=(
            FieldContract(
                normalized_name=field_name,
                original_name=field_name,
                python_type=python_type,
                required=True,
                source="inferred",
            ),
        ),
        locked=True,
    )


class TestResumeFinalizesAsFailed:
    """Regression test for Phase 0 fix #6: Resume leaves RUNNING.

    Bug: If _process_resumed_rows raised a non-GracefulShutdownError
    exception during resume, the run status stayed as RUNNING permanently.
    This blocked future resume attempts since recovery rejects RUNNING status.

    Fix: Added `except Exception` handler in resume() that calls
    recorder.finalize_run(run_id, status=RunStatus.FAILED).
    """

    def test_reconstruct_resume_state_refuses_incompatible_checkpoint_before_snapshot_work(self) -> None:
        db = make_landscape_db()
        orch = _make_orchestrator(db)
        run_id = "run-direct-reconstruct-incompatible-format"
        checkpoint = Checkpoint(
            checkpoint_id="cp-direct-reconstruct-incompatible-format",
            run_id=run_id,
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            format_version=None,
        )
        resume_point = ResumePoint(checkpoint=checkpoint, sequence_number=checkpoint.sequence_number)

        with (
            patch("elspeth.engine.orchestrator.resume.RecorderFactory") as recorder_factory,
            pytest.raises(NonResumableRunError, match="missing format_version"),
        ):
            orch._resume_coordinator.reconstruct_resume_state(resume_point, MagicMock(spec=PayloadStore))

        recorder_factory.assert_not_called()

    def test_resume_failure_finalizes_run_as_failed(self) -> None:
        """When _process_resumed_rows raises, run status becomes FAILED."""
        db = make_landscape_db()
        orch = _make_orchestrator(db)
        _insert_failed_run(db, "test-run-123")

        # Real resume point anchored to the run's (stubbed) latest checkpoint
        checkpoint = Checkpoint(
            checkpoint_id="cp-test-run-123",
            run_id="test-run-123",
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )
        resume_point = ResumePoint(checkpoint=checkpoint, sequence_number=checkpoint.sequence_number)
        admit_guard = _admit_resume_point(orch, resume_point)

        # Create mock config and graph
        config = MagicMock(spec=PipelineConfig)
        graph = MagicMock(spec=ExecutionGraph)
        payload_store = MagicMock(spec=PayloadStore)
        settings = MagicMock(spec=ElspethSettings)

        # Mock factory to capture finalize_run calls
        mock_factory = MagicMock(spec=RecorderFactory)
        mock_factory.run_lifecycle.get_source_schema.return_value = '{"mode": "observed"}'
        # ADR-025 §3 Decision 5 (G6): the run-level singleton contract was
        # deleted; ``_reconstruct_resume_state`` now reads per-source records
        # exclusively, and an empty map raises ``EmptyResumeStateError`` before
        # we reach the failure path under test. Supply at least one
        # ``run_sources`` record so the resume gets far enough to execute
        # ``_process_resumed_rows`` and trip the injected RuntimeError.
        mock_factory.run_lifecycle.get_run_source_resume_records.return_value = {
            NodeID("source-node"): SimpleNamespace(
                source_name="source",
                lifecycle_state="loaded",
                source_schema_json='{"properties": {}, "required": []}',
                schema_contract=MagicMock(spec=SchemaContract, name="contract"),
            ),
        }
        mock_factory.run_lifecycle.get_run_source_lifecycle_records.return_value = {
            NodeID("source-node"): SimpleNamespace(source_name="source", lifecycle_state="loaded"),
        }
        mock_factory.execution.get_incomplete_batches.return_value = []
        # FAILED finalization derives its counter baseline from the audit
        # projections; stub them with real zeros (mirroring the sibling tests
        # below) — the derive no longer swallows arbitrary mock errors, so an
        # unstubbed projection would propagate instead of degrading to None.
        mock_factory.query.get_all_token_outcomes_for_run.return_value = []
        mock_factory.run_status_projection.count_distinct_source_rows_with_terminal_outcome.return_value = 0
        mock_factory.run_status_projection.count_failed_coalesce_barrier_rows.return_value = 0
        prepare_for_run()
        mock_factory.run_lifecycle.get_runtime_val_manifest.return_value = canonical_json(build_runtime_val_manifest())

        # Mock RecoveryManager. The unprocessed row keeps the resume on the
        # processing path (non-quiescent), where the injected RuntimeError fires.
        mock_recovery = MagicMock(spec=RecoveryManager)
        mock_recovery.get_resume_workset.return_value = ResumeWorkSet(
            row_ids=("row-1",),
            incomplete_by_row={},
            buffered_token_ids=frozenset(),
        )
        mock_recovery.count_blocked_barrier_items.return_value = 0
        mock_recovery.get_unprocessed_row_data_by_source.return_value = [
            ResumedRow(
                row_id="row-1",
                row_index=0,
                source_node_id=NodeID("source-node"),
                row_data={"field": "value"},
            ),
        ]

        # Make _process_resumed_rows raise a RuntimeError (non-shutdown)
        with (
            admit_guard,
            patch.object(orch._resume_coordinator, "process_resumed_rows", side_effect=RuntimeError("test failure")),
            patch("elspeth.engine.orchestrator.resume.RecorderFactory", return_value=mock_factory),
            patch("elspeth.engine.orchestrator.resume.reconstruct_schema_from_json", return_value=MagicMock(spec=SchemaContract)),
            patch("elspeth.core.checkpoint.RecoveryManager", return_value=mock_recovery),
            patch.object(orch._ceremony, "emit_telemetry"),
            pytest.raises(RuntimeError, match="test failure"),
        ):
            orch.resume(
                resume_point,
                config,
                graph,
                payload_store=payload_store,
                settings=settings,
            )

        # Verify finalize_run was called with FAILED status
        # finalize_run(run_id, status) — status can be positional or keyword
        finalize_calls = mock_factory.run_lifecycle.finalize_run.call_args_list
        found_failed = False
        for call in finalize_calls:
            args, kwargs = call
            status = kwargs.get("status", args[1] if len(args) > 1 else None)
            if status == RunStatus.FAILED:
                found_failed = True
                break
        assert found_failed, (
            f"Run should be finalized as FAILED when resume fails with non-shutdown exception. finalize_run calls: {finalize_calls}"
        )

    def test_resume_partial_failure_ceremony_reports_cumulative_audit_counters(self) -> None:
        """Partial-result resume failures must not emit resume-local-only counters."""
        db = make_landscape_db()
        event_bus = MagicMock(spec_set=["emit"])
        emitted_events: list[object] = []
        event_bus.emit.side_effect = emitted_events.append
        orch = Orchestrator(db, event_bus=event_bus)
        run_id = "run-resume-partial-audit-counters"
        _insert_failed_run(db, run_id)

        mock_factory = MagicMock(spec=RecorderFactory)
        coordination_token = _make_heartbeat_safe_token(run_id, mock_factory)
        checkpoint = Checkpoint(
            checkpoint_id="cp-resume-partial-audit-counters",
            run_id=run_id,
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )
        resume_point = ResumePoint(checkpoint=checkpoint, sequence_number=checkpoint.sequence_number)
        resume_state = ResumeState(
            factory=mock_factory,
            run_id=run_id,
            unprocessed_rows=(
                ResumedRow(
                    row_id="row-resumed",
                    row_index=1,
                    source_node_id=NodeID("source"),
                    row_data={"value": "resumed"},
                ),
            ),
            incomplete_by_row={},
            recovery_manager=MagicMock(spec=RecoveryManager),
            schema_contracts_by_source={NodeID("source"): MagicMock(spec=SchemaContract)},
            source_names_by_source={NodeID("source"): "source"},
            source_lifecycle_by_source={NodeID("source"): "exhausted"},
            has_restored_barrier_work=False,
            coordination_token=coordination_token,
        )
        resume_only_result = RunResult(
            run_id=run_id,
            status=RunStatus.FAILED,
            rows_processed=1,
            rows_succeeded=1,
            rows_failed=0,
            rows_routed_success=1,
            rows_routed_failure=0,
            rows_quarantined=0,
            routed_destinations={"default": 1},
        )
        baseline_counters = ExecutionCounters(
            rows_processed=3,
            rows_succeeded=2,
            rows_failed=1,
            rows_routed_success=2,
            rows_routed_failure=1,
        )
        baseline_counters.routed_destinations.update({"default": 2, "failsink": 1})
        failure = RuntimeError("resume sweep failed")

        admit_guard = _admit_resume_point(orch, resume_point)
        with (
            admit_guard,
            patch.object(orch._resume_coordinator, "reconstruct_resume_state", return_value=resume_state),
            patch.object(
                orch._resume_coordinator,
                "process_resumed_rows",
                side_effect=_RunFailedWithPartialResultError(failure, resume_only_result),
            ),
            patch(
                "elspeth.engine.orchestrator.resume.derive_resume_terminal_status_from_audit",
                return_value=(RunStatus.COMPLETED_WITH_FAILURES, baseline_counters),
            ),
            pytest.raises(RuntimeError, match="resume sweep failed"),
        ):
            orch.resume(
                resume_point,
                MagicMock(spec=PipelineConfig),
                MagicMock(spec=ExecutionGraph),
                payload_store=MockPayloadStore(),
            )

        summaries = [event for event in emitted_events if isinstance(event, RunSummary)]
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.status.value == "failed"
        assert summary.total_rows == 4
        assert summary.succeeded == 3
        assert summary.failed == 1
        assert summary.routed_success == 3
        assert summary.routed_failure == 1
        assert dict(summary.routed_destinations) == {"default": 3, "failsink": 1}

    def test_resume_loop_drains_scheduler_work_before_replaying_rows(self) -> None:
        """Persisted scheduler work supersedes the old unprocessed-row replay path."""
        processor = MagicMock(spec=RowProcessor)
        processor.has_scheduled_work.return_value = True
        processor.has_unresolved_scheduler_work.return_value = False
        processor.active_scheduled_row_ids.return_value = frozenset({"row-should-not-replay"})
        processor.drain_scheduled_work.return_value = [make_row_result({"value": 1}, sink_name="default")]
        processor.process_existing_row.side_effect = AssertionError("source row replay must not run while scheduler work exists")
        config = PipelineConfig(
            sources={"primary": _specced_source()},
            transforms=(),
            sinks={"default": _specced_sink()},
        )
        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"default": []},
            processor=processor,
            ctx=MagicMock(spec=PluginContext),
            config=config,
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )

        interrupted = run_resume_processing_loop(
            loop_ctx,
            unprocessed_rows=(
                ResumedRow(
                    row_id="row-should-not-replay",
                    row_index=0,
                    source_node_id=NodeID("source"),
                    row_data={"value": 1},
                ),
            ),
            incomplete_by_row={},
            recovery_manager=MagicMock(spec=RecoveryManager),
            payload_store=MockPayloadStore(),
            run_id="run-scheduler-drain",
            resume_checkpoint_id="checkpoint-scheduler-drain",
            schema_contracts_by_source={NodeID("source"): MagicMock(spec=SchemaContract)},
        )

        assert interrupted is False
        processor.drain_scheduled_work.assert_called_once_with(loop_ctx.ctx)
        processor.process_existing_row.assert_not_called()
        assert loop_ctx.counters.rows_processed == 1
        assert loop_ctx.counters.rows_succeeded == 1
        assert len(loop_ctx.pending_tokens["default"]) == 1

    def test_resume_loop_fails_closed_when_scheduler_does_not_cover_all_recovered_rows(self) -> None:
        """Run-level scheduler presence must not suppress uncovered recovery rows."""
        from elspeth.contracts.errors import AuditIntegrityError

        processor = MagicMock(spec=RowProcessor)
        processor.has_scheduled_work.return_value = True
        processor.active_scheduled_row_ids.return_value = frozenset({"row-scheduled"})
        processor.drain_scheduled_work.return_value = [make_row_result({"value": 1}, sink_name="default")]
        processor.process_existing_row.side_effect = AssertionError("mixed scheduler coverage must fail before row replay policy")
        config = PipelineConfig(
            sources={"primary": _specced_source()},
            transforms=(),
            sinks={"default": _specced_sink()},
        )
        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"default": []},
            processor=processor,
            ctx=MagicMock(spec=PluginContext),
            config=config,
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )

        with pytest.raises(AuditIntegrityError, match="row-uncovered"):
            run_resume_processing_loop(
                loop_ctx,
                unprocessed_rows=(
                    ResumedRow(
                        row_id="row-scheduled",
                        row_index=0,
                        source_node_id=NodeID("source"),
                        row_data={"value": 1},
                    ),
                    ResumedRow(
                        row_id="row-uncovered",
                        row_index=1,
                        source_node_id=NodeID("source"),
                        row_data={"value": 2},
                    ),
                ),
                incomplete_by_row={},
                recovery_manager=MagicMock(spec=RecoveryManager),
                payload_store=MockPayloadStore(),
                run_id="run-scheduler-coverage",
                resume_checkpoint_id="checkpoint-scheduler-coverage",
                schema_contracts_by_source={NodeID("source"): MagicMock(spec=SchemaContract)},
            )

        processor.drain_scheduled_work.assert_not_called()
        processor.process_existing_row.assert_not_called()

    def test_resume_loop_refuses_completion_when_scheduler_work_remains_blocked(self) -> None:
        """Blocked/future scheduler work must survive resume instead of falling back to source replay."""
        orch = _make_orchestrator(make_landscape_db())
        source = _specced_source()
        source.on_success = "default"
        processor = MagicMock(spec=RowProcessor)
        processor.run_id = "run-with-blocked-work"
        processor.has_scheduled_work.return_value = True
        processor.has_unresolved_scheduler_work.return_value = True
        processor.active_scheduled_row_ids.return_value = frozenset({"row-should-not-replay"})
        processor.drain_scheduled_work.return_value = []
        processor.process_existing_row.side_effect = AssertionError("source row replay must not run while scheduler work remains")
        processor.summarize_unresolved_scheduler_work.return_value = ("BLOCKED count=1 node=join-results",)
        config = PipelineConfig(
            sources={"source": source},
            transforms=(),
            sinks={"default": _specced_sink()},
        )
        artifacts = SimpleNamespace(
            source_id_map={"source": NodeID("source")},
            edge_map={},
            sink_id_map={},
            source_id=NodeID("source"),
        )
        run_ctx = SimpleNamespace(
            processor=processor,
            ctx=MagicMock(spec=PluginContext),
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )

        with (
            patch("elspeth.engine.orchestrator.resume.setup_resume_context", return_value=artifacts),
            patch.object(orch._context_factory, "initialize_run_context", return_value=run_ctx),
            patch("elspeth.engine.orchestrator.resume.run_transform_runtime_preflights"),
            patch.object(orch._sink_flush, "flush_and_write_sinks") as flush_sinks,
            pytest.raises(Exception, match="left non-terminal scheduler work after end-of-source flush") as exc_info,
        ):
            orch._resume_coordinator.process_resumed_rows(
                MagicMock(spec=RecorderFactory),
                "run-with-blocked-work",
                config,
                MagicMock(spec=ExecutionGraph),
                unprocessed_rows=(
                    ResumedRow(
                        row_id="row-should-not-replay",
                        row_index=0,
                        source_node_id=NodeID("source"),
                        row_data={"value": 1},
                    ),
                ),
                barrier_restore=None,
                payload_store=MagicMock(spec=PayloadStore),
                incomplete_by_row={},
                recovery_manager=MagicMock(spec=RecoveryManager),
                resume_checkpoint_id="checkpoint-blocked-work",
                schema_contracts_by_source={NodeID("source"): MagicMock(spec=SchemaContract)},
            )

        assert isinstance(exc_info.value.__cause__, OrchestrationInvariantError)
        processor.drain_scheduled_work.assert_called_once_with(run_ctx.ctx)
        processor.process_existing_row.assert_not_called()
        flush_sinks.assert_not_called()

    def test_resume_runtime_preflight_failure_cleans_initialized_plugins(self) -> None:
        """Resume preflight failure must tear down transform/sink resources opened by on_start."""
        orch = _make_orchestrator(make_landscape_db())
        source = _specced_source()
        source.on_success = "default"
        transform = _specced_transform()
        sink = _specced_sink()
        config = PipelineConfig(
            sources={"source": source},
            transforms=(transform,),
            sinks={"default": sink},
        )
        processor = MagicMock(spec=RowProcessor)
        processor.run_id = "run-resume-runtime-preflight-fails"
        artifacts = SimpleNamespace(
            source_id_map={"source": NodeID("source")},
            edge_map={},
            sink_id_map={"default": NodeID("sink")},
            source_id=NodeID("source"),
        )
        run_ctx = SimpleNamespace(
            processor=processor,
            ctx=MagicMock(spec=PluginContext),
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )
        graph = MagicMock(spec=ExecutionGraph)

        with (
            patch("elspeth.engine.orchestrator.resume.setup_resume_context", return_value=artifacts),
            patch.object(orch._context_factory, "initialize_run_context", return_value=run_ctx),
            patch(
                "elspeth.engine.orchestrator.resume.run_transform_runtime_preflights",
                side_effect=RuntimeError("resume runtime preflight exploded"),
            ),
            patch.object(orch._sink_flush, "flush_and_write_sinks") as flush_sinks,
            pytest.raises(RuntimeError, match="resume runtime preflight exploded"),
        ):
            orch._resume_coordinator.process_resumed_rows(
                MagicMock(spec=RecorderFactory),
                "run-resume-runtime-preflight-fails",
                config,
                graph,
                unprocessed_rows=(),
                barrier_restore=None,
                payload_store=MagicMock(spec=PayloadStore),
                incomplete_by_row={},
                recovery_manager=MagicMock(spec=RecoveryManager),
                resume_checkpoint_id="checkpoint-runtime-preflight-fails",
                schema_contracts_by_source={NodeID("source"): MagicMock(spec=SchemaContract)},
            )

        source.on_complete.assert_not_called()
        source.close.assert_not_called()
        assert orch._checkpoints._active_graph is None
        transform.on_complete.assert_called_once_with(run_ctx.ctx)
        transform.close.assert_called_once_with()
        sink.on_complete.assert_called_once_with(run_ctx.ctx)
        sink.close.assert_called_once_with()
        flush_sinks.assert_not_called()

    def test_setup_resume_context_uses_all_source_roots(self) -> None:
        """Multi-source resume must build a full source map instead of calling graph.get_sources()[0]."""
        graph = ExecutionGraph()
        graph.add_node(
            "source-orders",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"source_name": "orders", "schema": {"mode": "observed"}},
        )
        graph.add_node(
            "source-refunds",
            node_type=NodeType.SOURCE,
            plugin_name="csv",
            config={"source_name": "refunds", "schema": {"mode": "observed"}},
        )
        graph.add_node("sink-output", node_type=NodeType.SINK, plugin_name="json", config={"schema": {"mode": "observed"}})
        graph.add_edge("source-orders", "sink-output", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("source-refunds", "sink-output", label="continue", mode=RoutingMode.MOVE)
        graph.set_sink_id_map({SinkName("output"): NodeID("sink-output")})

        orders_source = _specced_source()
        orders_source.name = "csv"
        orders_source._on_validation_failure = "discard"
        refunds_source = _specced_source()
        refunds_source.name = "csv"
        refunds_source._on_validation_failure = "discard"
        sink = _specced_sink()
        sink.name = "json"
        sink._on_write_failure = "discard"
        config = PipelineConfig(
            sources={"orders": orders_source, "refunds": refunds_source},
            transforms=(),
            sinks={"output": sink},
        )
        factory = MagicMock(spec=RecorderFactory)
        factory.data_flow.get_edge_map.return_value = {}

        artifacts = setup_resume_context(factory, "run-multi-source-resume", config, graph)

        assert artifacts.source_id_map == {
            "orders": NodeID("source-orders"),
            "refunds": NodeID("source-refunds"),
        }
        assert artifacts.source_id == NodeID("source-orders")

    def test_resume_loop_uses_source_scoped_contract_for_each_replayed_row(self) -> None:
        """Replayed rows from different sources must keep their source-specific schema contract."""
        processor = MagicMock(spec=RowProcessor)
        processor.has_scheduled_work.return_value = False
        processor.has_unresolved_scheduler_work.return_value = False
        processor.process_existing_row.return_value = []
        config = PipelineConfig(
            sources={"orders": _specced_source(), "refunds": _specced_source()},
            transforms=(),
            sinks={"default": _specced_sink()},
        )
        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"default": []},
            processor=processor,
            ctx=MagicMock(spec=PluginContext),
            config=config,
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )
        orders_contract = MagicMock(spec=SchemaContract, name="orders-contract")
        refunds_contract = MagicMock(spec=SchemaContract, name="refunds-contract")

        interrupted = run_resume_processing_loop(
            loop_ctx,
            unprocessed_rows=(
                ResumedRow(
                    row_id="row-orders",
                    row_index=0,
                    source_node_id=NodeID("source-orders"),
                    row_data={"order_id": 1},
                ),
                ResumedRow(
                    row_id="row-refunds",
                    row_index=1,
                    source_node_id=NodeID("source-refunds"),
                    row_data={"refund_id": "r1"},
                ),
            ),
            incomplete_by_row={},
            recovery_manager=MagicMock(spec=RecoveryManager),
            payload_store=MockPayloadStore(),
            run_id="run-source-scoped-contracts",
            resume_checkpoint_id="checkpoint-source-scoped-contracts",
            schema_contracts_by_source={
                NodeID("source-orders"): orders_contract,
                NodeID("source-refunds"): refunds_contract,
            },
            source_on_success_by_source={
                NodeID("source-orders"): "orders_sink",
                NodeID("source-refunds"): "refunds_sink",
            },
        )

        assert interrupted is False
        contracts = [call.kwargs["row_data"].contract for call in processor.process_existing_row.call_args_list]
        assert contracts == [orders_contract, refunds_contract]
        source_node_ids = [call.kwargs["source_node_id"] for call in processor.process_existing_row.call_args_list]
        assert source_node_ids == [NodeID("source-orders"), NodeID("source-refunds")]
        source_on_success = [call.kwargs["source_on_success"] for call in processor.process_existing_row.call_args_list]
        assert source_on_success == ["orders_sink", "refunds_sink"]

    def test_source_resume_metadata_is_recorded_before_first_row_processing(self) -> None:
        """A mid-source crash after row creation must not leave run_sources absent."""

        @contextmanager
        def _source_operation(*args, **kwargs):
            yield SimpleNamespace(operation=SimpleNamespace(operation_id="source-op-1"))

        orch = _make_orchestrator(make_landscape_db())
        source = _specced_source(output_schema=MagicMock(spec=BaseModel))
        source.name = "csv"
        source.config = {"path": "refunds.csv"}
        source.on_success = "refunds_sink"
        source.output_schema.model_json_schema.return_value = {"type": "object", "properties": {"refund_id": {"type": "string"}}}
        source.get_schema_contract.return_value = MagicMock(spec=SchemaContract, name="refunds-contract")
        source.get_field_resolution.return_value = None
        config = PipelineConfig(
            sources={"refunds": source},
            transforms=(),
            sinks={"refunds_sink": _specced_sink()},
        )
        factory = MagicMock(spec=RecorderFactory)
        events: list[str] = []

        def _record_run_source(**kwargs):
            events.append(f"record:{kwargs['lifecycle_state']}")

        factory.run_lifecycle.record_run_source.side_effect = _record_run_source
        processor = MagicMock(spec=RowProcessor)

        def _process_row(**kwargs):
            events.append("process")
            raise RuntimeError("boom after row persisted")

        processor.process_row.side_effect = _process_row
        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"refunds_sink": []},
            processor=processor,
            ctx=MagicMock(spec=PluginContext),
            config=config,
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )

        def _load_source(run_id_arg, ctx_arg, *, active_source):
            events.append("load")
            assert ctx_arg.node_id == NodeID("source-refunds")
            assert ctx_arg.operation_id == "source-op-1"
            return iter((make_source_row({"refund_id": "r1"}),))

        with (
            patch("elspeth.engine.orchestrator.source_iteration.track_operation", _source_operation),
            patch.object(orch._source_driver, "load_source_with_events", side_effect=_load_source),
            pytest.raises(RuntimeError, match="boom after row persisted"),
        ):
            orch._source_driver.run_main_processing_loop(
                loop_ctx,
                factory,
                run_id="run-1",
                source_id=NodeID("source-refunds"),
                edge_map={},
                active_source_name="refunds",
                active_source=source,
                flush_end_of_input=True,
            )

        assert events == ["record:loading", "load", "process"]
        factory.run_lifecycle.record_run_source.assert_called_once()

    def test_source_exhaustion_is_recorded_before_eof_flush_failure(self) -> None:
        """A crash in EOF engine work must not look like an incomplete source load."""

        @contextmanager
        def _source_operation(*args, **kwargs):
            yield SimpleNamespace(operation=SimpleNamespace(operation_id="source-op-1"))

        orch = _make_orchestrator(make_landscape_db())
        source_contract = _observed_contract("refund_id", str)
        source = _specced_source(output_schema=MagicMock(spec=BaseModel))
        source.name = "csv"
        source.config = {"path": "refunds.csv"}
        source.on_success = "refunds_sink"
        source.output_schema.model_json_schema.return_value = {"type": "object", "properties": {"refund_id": {"type": "string"}}}
        source.get_schema_contract.return_value = source_contract
        source.get_field_resolution.return_value = None
        config = PipelineConfig(
            sources={"refunds": source},
            transforms=(),
            sinks={"refunds_sink": _specced_sink()},
            aggregation_settings={
                NodeID("agg-refunds"): AggregationSettings(
                    name="agg-refunds",
                    plugin="refund_batcher",
                    input="refunds_sink",
                    on_error="discard",
                )
            },
        )
        factory = MagicMock(spec=RecorderFactory)
        events: list[str] = []

        def _record_run_source(**kwargs):
            events.append(f"record:{kwargs['lifecycle_state']}")

        factory.run_lifecycle.record_run_source.side_effect = _record_run_source
        processor = MagicMock(spec=RowProcessor)
        processor.check_aggregation_timeout.return_value = (False, None)
        processor.process_row.side_effect = lambda **kwargs: events.append("process") or []
        # Slice 3 (ADR-030 §D): the EOF flush helper gates on journal
        # quiescence and runs a journal-first intake pass first.
        processor.count_unquiesced_scheduler_work.return_value = 0
        processor.run_barrier_intake.return_value = []
        processor.has_blocked_barrier_work.return_value = False

        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"refunds_sink": []},
            processor=processor,
            ctx=MagicMock(spec=PluginContext),
            config=config,
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )

        def _load_source(run_id_arg, ctx_arg, *, active_source):
            events.append("load")
            return iter((make_source_row({"refund_id": "r1"}, contract=source_contract),))

        def _flush_eof_buffers(**kwargs):
            events.append("eof_flush")
            raise RuntimeError("boom during EOF flush")

        with (
            patch("elspeth.engine.orchestrator.source_iteration.track_operation", _source_operation),
            patch.object(orch._source_driver, "load_source_with_events", side_effect=_load_source),
            # Slice 3 re-pin: the EOF flush seam moved into the
            # run_end_of_input_barrier_flush helper (orchestrator/leader_drain.py).
            patch("elspeth.engine.orchestrator.leader_drain.flush_remaining_aggregation_buffers", side_effect=_flush_eof_buffers),
            pytest.raises(RuntimeError, match="boom during EOF flush"),
        ):
            orch._source_driver.run_main_processing_loop(
                loop_ctx,
                factory,
                run_id="run-1",
                source_id=NodeID("source-refunds"),
                edge_map={},
                active_source_name="refunds",
                active_source=source,
                flush_end_of_input=True,
            )

        assert events == ["record:loading", "load", "process", "record:exhausted", "eof_flush"]

    def test_second_observed_source_persists_resume_contract_before_row_failure(self) -> None:
        """A later source crash must leave source-scoped resume metadata behind."""

        @contextmanager
        def _source_operation(*args, **kwargs):
            yield SimpleNamespace(operation=SimpleNamespace(operation_id="source-op-1"))

        orders_contract = _observed_contract("order_id", int)
        refunds_contract = _observed_contract("refund_id", str)
        orch = _make_orchestrator(make_landscape_db())
        source = _specced_source(output_schema=MagicMock(spec=BaseModel))
        source.name = "csv"
        source.config = {"path": "refunds.csv"}
        source.on_success = "refunds_sink"
        source.output_schema.model_json_schema.return_value = {"type": "object", "properties": {"refund_id": {"type": "string"}}}
        source.get_field_resolution.return_value = None
        # Two calls per source (multi-source-token-scheduler Fix 2):
        # 1) ``record_run_source`` BEFORE first row (contract not yet locked),
        # 2) ``_record_schema_contract`` AFTER first valid row.
        source.get_schema_contract.side_effect = [None, refunds_contract]
        config = PipelineConfig(
            sources={"refunds": source},
            transforms=(),
            sinks={"refunds_sink": _specced_sink()},
        )
        factory = MagicMock(spec=RecorderFactory)
        events: list[str] = []
        factory.run_lifecycle.update_run_source_contract.side_effect = lambda **kwargs: events.append("source_contract")
        factory.data_flow.update_node_output_contract.side_effect = lambda *args, **kwargs: events.append("node_contract")
        processor = MagicMock(spec=RowProcessor)

        def _process_row(**kwargs):
            events.append("process")
            assert kwargs["ctx"].contract is refunds_contract
            raise RuntimeError("boom after source contract persisted")

        processor.process_row.side_effect = _process_row
        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"refunds_sink": []},
            processor=processor,
            ctx=MagicMock(spec=PluginContext, contract=orders_contract),
            config=config,
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )

        def _load_source(run_id_arg, ctx_arg, *, active_source):
            return iter((make_source_row({"refund_id": "r1"}, contract=refunds_contract),))

        with (
            patch("elspeth.engine.orchestrator.source_iteration.track_operation", _source_operation),
            patch.object(orch._source_driver, "load_source_with_events", side_effect=_load_source),
            pytest.raises(RuntimeError, match="boom after source contract persisted"),
        ):
            orch._source_driver.run_main_processing_loop(
                loop_ctx,
                factory,
                run_id="run-1",
                source_id=NodeID("source-refunds"),
                edge_map={},
                active_source_name="refunds",
                active_source=source,
                flush_end_of_input=True,
            )

        assert events == ["source_contract", "node_contract", "process"]

    def test_fresh_run_refuses_success_when_scheduler_work_remains(self) -> None:
        """A fresh multi-source run must not complete with durable scheduler work still active."""
        orch = _make_orchestrator(make_landscape_db())
        source = _specced_source()
        source.name = "csv"
        source.on_success = "sink"
        sink = _specced_sink()
        config = PipelineConfig(
            sources={"orders": source},
            transforms=(),
            sinks={"sink": sink},
        )
        processor = MagicMock(spec=RowProcessor)
        processor.run_id = "run-stuck-scheduler"
        processor.has_peer_active_leases.return_value = False
        processor.peer_lease_wait_budget_seconds.return_value = 0.0
        processor.has_unresolved_scheduler_work.return_value = True
        processor.summarize_unresolved_scheduler_work.return_value = ("READY count=1 node=transform-normalize",)
        run_ctx = SimpleNamespace(
            processor=processor,
            ctx=MagicMock(spec=PluginContext),
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )
        artifacts = SimpleNamespace(
            source_id_map={"orders": NodeID("source-orders")},
            edge_map={},
            sink_id_map={},
            source_id=NodeID("source-orders"),
        )

        with (
            patch.object(orch, "_register_graph_nodes_and_edges", return_value=artifacts),
            patch.object(orch._context_factory, "initialize_run_context", return_value=run_ctx),
            patch("elspeth.engine.orchestrator.leader_drain.run_transform_runtime_preflights"),
            patch.object(
                orch._source_driver,
                "run_main_processing_loop",
                return_value=LoopResult(interrupted=False, start_time=0.0, phase_start=0.0, last_progress_time=0.0),
            ),
            patch.object(orch._sink_flush, "flush_and_write_sinks") as flush_sinks,
            pytest.raises(Exception, match="left non-terminal scheduler work after final source flush") as exc_info,
        ):
            orch._execute_run(
                MagicMock(spec=RecorderFactory),
                "run-stuck-scheduler",
                config,
                MagicMock(spec=ExecutionGraph),
                payload_store=MagicMock(spec=PayloadStore),
            )

        assert isinstance(exc_info.value.__cause__, OrchestrationInvariantError)
        flush_sinks.assert_not_called()

    def test_reconstruct_resume_state_uses_run_sources_records_for_multi_source_rows(self) -> None:
        """Resume reconstruction must restore schema classes and contracts per source node."""
        db = make_landscape_db()
        orch = _make_orchestrator(db)
        orch._checkpoint_manager = MagicMock(spec=CheckpointManager)
        orch._resume_coordinator._checkpoint_manager = orch._checkpoint_manager
        run_id = "run-multi-source-reconstruct"
        checkpoint = Checkpoint(
            checkpoint_id="cp-multi-source-reconstruct",
            run_id=run_id,
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )
        resume_point = ResumePoint(
            checkpoint=checkpoint,
            sequence_number=checkpoint.sequence_number,
        )
        orders_contract = MagicMock(spec=SchemaContract, name="orders-contract")
        refunds_contract = MagicMock(spec=SchemaContract, name="refunds-contract")
        mock_factory = MagicMock(spec=RecorderFactory)
        mock_factory.run_lifecycle.get_source_schema.side_effect = AssertionError("run-level source schema must not be used")
        prepare_for_run()
        mock_factory.run_lifecycle.get_runtime_val_manifest.return_value = canonical_json(build_runtime_val_manifest())
        mock_factory.run_lifecycle.get_run_source_resume_records.return_value = {
            NodeID("source-orders"): SimpleNamespace(
                source_name="orders",
                lifecycle_state="loaded",
                source_schema_json='{"title":"Orders"}',
                schema_contract=orders_contract,
            ),
            NodeID("source-refunds"): SimpleNamespace(
                source_name="refunds",
                lifecycle_state="loaded",
                source_schema_json='{"title":"Refunds"}',
                schema_contract=refunds_contract,
            ),
        }
        mock_recovery = MagicMock(spec=RecoveryManager)
        mock_recovery.get_resume_workset.return_value = ResumeWorkSet(
            row_ids=("row-orders", "row-refunds"),
            incomplete_by_row={},
            buffered_token_ids=frozenset(),
        )
        mock_recovery.count_blocked_barrier_items.return_value = 0
        mock_recovery.get_unprocessed_row_data_by_source.return_value = (
            ResumedRow(
                row_id="row-orders",
                row_index=0,
                source_node_id=NodeID("source-orders"),
                row_data={"order_id": 1},
            ),
            ResumedRow(
                row_id="row-refunds",
                row_index=1,
                source_node_id=NodeID("source-refunds"),
                row_data={"refund_id": "r1"},
            ),
        )
        orders_schema = MagicMock(spec=SchemaContract, name="OrdersSchema")
        refunds_schema = MagicMock(spec=SchemaContract, name="RefundsSchema")

        with (
            patch("elspeth.engine.orchestrator.resume.RecorderFactory", return_value=mock_factory),
            patch("elspeth.core.checkpoint.RecoveryManager", return_value=mock_recovery),
            patch("elspeth.engine.orchestrator.resume.reconstruct_schema_from_json", side_effect=[orders_schema, refunds_schema]),
        ):
            state = orch._resume_coordinator.reconstruct_resume_state(resume_point, MockPayloadStore())

        assert state.unprocessed_rows == (
            ResumedRow(
                row_id="row-orders",
                row_index=0,
                source_node_id=NodeID("source-orders"),
                row_data={"order_id": 1},
            ),
            ResumedRow(
                row_id="row-refunds",
                row_index=1,
                source_node_id=NodeID("source-refunds"),
                row_data={"refund_id": "r1"},
            ),
        )
        assert state.schema_contracts_by_source == {
            NodeID("source-orders"): orders_contract,
            NodeID("source-refunds"): refunds_contract,
        }
        mock_recovery.get_unprocessed_row_data_by_source.assert_called_once()
        assert mock_recovery.get_unprocessed_row_data_by_source.call_args.kwargs["source_schema_classes"] == {
            NodeID("source-orders"): orders_schema,
            NodeID("source-refunds"): refunds_schema,
        }
        assert mock_recovery.get_unprocessed_row_data_by_source.call_args.kwargs["row_ids"] == (
            "row-orders",
            "row-refunds",
        )

    def test_reconstruct_resume_state_accepts_exhausted_source_lifecycle(self) -> None:
        """Exhausted sources are complete enough for engine-only EOF work resume."""
        db = make_landscape_db()
        orch = _make_orchestrator(db)
        orch._checkpoint_manager = MagicMock(spec=CheckpointManager)
        orch._resume_coordinator._checkpoint_manager = orch._checkpoint_manager
        run_id = "run-exhausted-source-reconstruct"
        checkpoint = Checkpoint(
            checkpoint_id="cp-exhausted-source-reconstruct",
            run_id=run_id,
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )
        resume_point = ResumePoint(
            checkpoint=checkpoint,
            sequence_number=checkpoint.sequence_number,
        )
        source_contract = MagicMock(spec=SchemaContract, name="source-contract")
        mock_factory = MagicMock(spec=RecorderFactory)
        prepare_for_run()
        mock_factory.run_lifecycle.get_runtime_val_manifest.return_value = canonical_json(build_runtime_val_manifest())
        mock_factory.run_lifecycle.get_run_source_lifecycle_records.return_value = {
            NodeID("source-primary"): SimpleNamespace(source_name="primary", lifecycle_state="exhausted")
        }
        mock_factory.run_lifecycle.get_run_source_resume_records.return_value = {
            NodeID("source-primary"): SimpleNamespace(
                source_name="primary",
                lifecycle_state="exhausted",
                source_schema_json='{"title":"Primary"}',
                schema_contract=source_contract,
            )
        }
        mock_recovery = MagicMock(spec=RecoveryManager)
        mock_recovery.get_resume_workset.return_value = ResumeWorkSet(
            row_ids=(),
            incomplete_by_row={},
            buffered_token_ids=frozenset(),
        )
        # F1: a non-zero journal BLOCKED barrier count must surface on the
        # ResumeState as has_restored_barrier_work=True (quiescence-gate input).
        mock_recovery.count_blocked_barrier_items.return_value = 3
        mock_recovery.get_unprocessed_row_data_by_source.return_value = ()
        source_schema = MagicMock(spec=SchemaContract, name="PrimarySchema")

        with (
            patch("elspeth.engine.orchestrator.resume.RecorderFactory", return_value=mock_factory),
            patch("elspeth.core.checkpoint.RecoveryManager", return_value=mock_recovery),
            patch("elspeth.engine.orchestrator.resume.reconstruct_schema_from_json", return_value=source_schema),
        ):
            state = orch._resume_coordinator.reconstruct_resume_state(resume_point, MockPayloadStore())

        assert state.source_lifecycle_by_source == {NodeID("source-primary"): "exhausted"}
        assert state.schema_contracts_by_source == {NodeID("source-primary"): source_contract}
        mock_recovery.count_blocked_barrier_items.assert_called_once_with(run_id)
        assert state.has_restored_barrier_work is True

    def test_resume_treats_empty_journal_as_all_rows_processed(self) -> None:
        """No unprocessed rows + no journal BLOCKED barrier work -> early completion.

        F1 journal semantics (rewritten from the blob-era "empty restored
        coalesce checkpoint" test): the quiescence gate reads the scheduler
        journal. An empty journal (no BLOCKED barrier rows) with zero
        unprocessed rows means the run genuinely finished all work before
        crashing — resume must finalize from audit truth without forcing a
        processing pass.
        """
        db = make_landscape_db()
        orch = _make_orchestrator(db)
        run_id = "run-empty-journal"
        _insert_failed_run(db, run_id)
        mock_factory = MagicMock(spec=RecorderFactory)
        mock_factory.data_flow.sweep_deferred_invariants_or_crash = MagicMock(spec=object)
        mock_factory.run_lifecycle.finalize_run = MagicMock(spec=object)
        # ADR-030 §A.3 (slice 4): resume() always starts a RunHeartbeatThread.
        # Provide a valid token and configure the mock repo to return a healthy
        # snapshot so the thread's latch is never set during this test.
        coordination_token = _make_heartbeat_safe_token(run_id, mock_factory)

        checkpoint = Checkpoint(
            checkpoint_id="cp-empty-journal",
            run_id=run_id,
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )
        resume_point = ResumePoint(
            checkpoint=checkpoint,
            sequence_number=checkpoint.sequence_number,
        )
        resume_state = ResumeState(
            factory=mock_factory,
            run_id=run_id,
            unprocessed_rows=(),
            incomplete_by_row={},
            recovery_manager=MagicMock(spec=RecoveryManager),
            schema_contracts_by_source={NodeID("source"): MagicMock(spec=SchemaContract)},
            source_names_by_source={NodeID("source"): "source"},
            source_lifecycle_by_source={NodeID("source"): "loaded"},
            has_restored_barrier_work=False,
            coordination_token=coordination_token,
        )

        admit_guard = _admit_resume_point(orch, resume_point)
        with (
            admit_guard,
            patch.object(orch._resume_coordinator, "reconstruct_resume_state", return_value=resume_state),
            patch.object(orch._resume_coordinator, "process_resumed_rows", side_effect=AssertionError("empty journal should early-exit")),
            patch(
                "elspeth.engine.orchestrator.resume.derive_resume_terminal_status_from_audit",
                return_value=(RunStatus.COMPLETED, ExecutionCounters(rows_processed=3, rows_succeeded=3)),
            ),
            patch.object(orch._ceremony, "emit_telemetry"),
            patch.object(orch._checkpoints, "delete_checkpoints"),
        ):
            result = orch.resume(
                resume_point,
                MagicMock(spec=object),
                MagicMock(spec=object),
                payload_store=MockPayloadStore(),
            )

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 3
        mock_factory.run_lifecycle.finalize_run.assert_called_once_with(run_id, status=RunStatus.COMPLETED, token=coordination_token)

    def test_resume_with_only_journal_barrier_work_does_not_early_complete(self) -> None:
        """THE F1 TASK 3.2 TRAP: fully-buffered crashed run must not early-complete.

        Journal BLOCKED barrier rows are EXCLUDED from unprocessed_rows
        (restored, not re-driven), so a run whose entire remaining work sits
        at barriers presents zero unprocessed rows. If the quiescence gate
        ignored the journal, resume would finalize the run and delete its
        checkpoints WITHOUT building the BarrierJournalRestoreContext — the
        buffered batch would never flush. The gate must route through the
        processing path, handing the restore context (carrying the
        checkpoint's barrier scalars) to processor construction.
        """
        from elspeth.contracts.barrier_scalars import AggregationNodeScalars, BarrierScalars

        db = make_landscape_db()
        orch = _make_orchestrator(db)
        run_id = "run-buffered-only"
        _insert_failed_run(db, run_id)
        mock_factory = MagicMock(spec=RecorderFactory)
        mock_factory.run_lifecycle.finalize_run = MagicMock(spec=object)
        mock_factory.run_status_projection.count_distinct_source_rows_with_terminal_outcome.return_value = 0
        mock_factory.run_status_projection.count_failed_coalesce_barrier_rows.return_value = 0
        # ADR-030 §A.3 (slice 4): provide a valid token + healthy heartbeat snapshot.
        coordination_token = _make_heartbeat_safe_token(run_id, mock_factory)
        scalars = BarrierScalars(
            aggregation={"agg-node": AggregationNodeScalars(count_fire_offset=1.0, condition_fire_offset=None)},
            coalesce={},
        )

        checkpoint = Checkpoint(
            checkpoint_id="cp-buffered-only",
            run_id=run_id,
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )
        resume_point = ResumePoint(
            checkpoint=checkpoint,
            sequence_number=checkpoint.sequence_number,
            barrier_scalars=scalars,
        )
        resume_state = ResumeState(
            factory=mock_factory,
            run_id=run_id,
            unprocessed_rows=(),
            incomplete_by_row={},
            recovery_manager=MagicMock(spec=RecoveryManager),
            schema_contracts_by_source={NodeID("source"): MagicMock(spec=SchemaContract)},
            source_names_by_source={NodeID("source"): "source"},
            source_lifecycle_by_source={NodeID("source"): "exhausted"},
            has_restored_barrier_work=True,
            batch_id_remap={"batch-dead": "batch-retry"},
            coordination_token=coordination_token,
        )
        resumed_result = RunResult(
            run_id=run_id,
            status=RunStatus.RUNNING,
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
        )

        admit_guard = _admit_resume_point(orch, resume_point)
        with (
            admit_guard,
            patch.object(orch._resume_coordinator, "reconstruct_resume_state", return_value=resume_state),
            patch.object(orch._resume_coordinator, "process_resumed_rows", return_value=resumed_result) as process_resumed,
            patch.object(orch._ceremony, "emit_telemetry"),
            patch.object(orch._checkpoints, "delete_checkpoints") as delete_checkpoints,
        ):
            result = orch.resume(
                resume_point,
                MagicMock(spec=PipelineConfig),
                MagicMock(spec=ExecutionGraph),
                payload_store=MockPayloadStore(),
            )

        # The processing path ran (no early-complete) and the journal-restore
        # context was built from the resume point + reconstructed state.
        process_resumed.assert_called_once()
        barrier_restore = process_resumed.call_args.kwargs["barrier_restore"]
        assert barrier_restore is not None
        assert barrier_restore.resume_checkpoint_id == "cp-buffered-only"
        assert barrier_restore.barrier_scalars is scalars
        assert dict(barrier_restore.batch_id_remap) == {"batch-dead": "batch-retry"}
        # Checkpoints are deleted only AFTER the processing path completed.
        delete_checkpoints.assert_called_once_with(run_id)
        assert result.status == RunStatus.EMPTY

    def test_all_rows_processed_resume_replays_structural_counters_from_audit(self) -> None:
        """All-terminal resume must not fabricate structural counters as zero."""
        db = make_landscape_db()
        orch = _make_orchestrator(db)
        run_id = "run-structural-counter-resume"
        _insert_failed_run(db, run_id)
        mock_factory = MagicMock(spec=RecorderFactory)
        mock_factory.data_flow.sweep_deferred_invariants_or_crash = MagicMock(spec=DataFlowRepository.sweep_deferred_invariants_or_crash)
        mock_factory.run_lifecycle.finalize_run = MagicMock(spec=RunLifecycleRepository.finalize_run)
        # F2 (resume-fork-reemit): rows_processed is now sourced from a dedicated
        # distinct-source-row query, not a per-leaf tally over the outcome list.
        # This synthetic scenario represents 3 source rows reaching a terminal
        # outcome (the success / coalesced / sink-discarded predicate rows); the
        # buffered/batch-consumed/fork-parent/expand-parent records are structural
        # and do not add new source rows.  Mock the query to return that count —
        # the real QueryRepository computes it via COUNT(DISTINCT row_id) over a
        # tokens-table JOIN, which a pure-outcome-list mock cannot reproduce.
        mock_factory.run_status_projection.count_distinct_source_rows_with_terminal_outcome.return_value = 3
        # rows_coalesce_failed likewise derives from a dedicated query (DISTINCT
        # failed-barrier pairs over node_states); no coalesce failures here.
        mock_factory.run_status_projection.count_failed_coalesce_barrier_rows.return_value = 0
        # ADR-030 §A.3 (slice 4): provide a valid token + healthy heartbeat snapshot.
        coordination_token = _make_heartbeat_safe_token(run_id, mock_factory)
        mock_factory.query.get_all_token_outcomes_for_run.return_value = [
            _make_token_outcome(
                run_id=run_id,
                token_id="tok-buffered",
                outcome=None,
                path=TerminalPath.BUFFERED,
                completed=False,
                batch_id="batch-1",
            ),
            _make_token_outcome(
                run_id=run_id,
                token_id="tok-buffered",
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.BATCH_CONSUMED,
                batch_id="batch-1",
            ),
            _make_token_outcome(
                run_id=run_id,
                token_id="tok-fork-parent",
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.FORK_PARENT,
                fork_group_id="fork-1",
            ),
            _make_token_outcome(
                run_id=run_id,
                token_id="tok-expand-parent",
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.EXPAND_PARENT,
                expand_group_id="expand-1",
            ),
            _make_token_outcome(
                run_id=run_id,
                token_id="tok-success",
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="default",
            ),
            _make_token_outcome(
                run_id=run_id,
                token_id="tok-coalesced",
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.COALESCED,
                sink_name="default",
                join_group_id="join-1",
            ),
            _make_token_outcome(
                run_id=run_id,
                token_id="tok-discarded",
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.SINK_DISCARDED,
                sink_name=DISCARD_SINK_NAME,
                error_hash="sinkdiscard0001",
            ),
        ]

        checkpoint = Checkpoint(
            checkpoint_id="cp-structural-counter-resume",
            run_id=run_id,
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )
        resume_point = ResumePoint(
            checkpoint=checkpoint,
            sequence_number=checkpoint.sequence_number,
        )
        resume_state = ResumeState(
            factory=mock_factory,
            run_id=run_id,
            unprocessed_rows=(),
            incomplete_by_row={},
            recovery_manager=MagicMock(spec=RecoveryManager),
            schema_contracts_by_source={NodeID("source"): MagicMock(spec=SchemaContract)},
            source_names_by_source={NodeID("source"): "source"},
            source_lifecycle_by_source={NodeID("source"): "exhausted"},
            has_restored_barrier_work=False,
            coordination_token=coordination_token,
        )

        admit_guard = _admit_resume_point(orch, resume_point)
        with (
            admit_guard,
            patch.object(orch._resume_coordinator, "reconstruct_resume_state", return_value=resume_state),
            patch.object(
                orch._resume_coordinator, "process_resumed_rows", side_effect=AssertionError("all-terminal resume should early-exit")
            ),
            patch.object(orch._ceremony, "emit_telemetry"),
            patch.object(orch._checkpoints, "delete_checkpoints"),
        ):
            result = orch.resume(
                resume_point,
                MagicMock(spec=PipelineConfig),
                MagicMock(spec=ExecutionGraph),
                payload_store=MockPayloadStore(),
            )

        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_processed == 3
        assert result.rows_succeeded == 2
        assert result.rows_failed == 1
        assert result.rows_forked == 1
        assert result.rows_expanded == 1
        assert result.rows_buffered == 1
        assert result.rows_coalesced == 1
        assert result.rows_diverted == 1

    def test_resume_allows_exhausted_source_with_restored_engine_work(self) -> None:
        """Resume must not reject exhausted sources before engine-only work drains.

        F1: "restored engine work" is journal BLOCKED barrier rows
        (has_restored_barrier_work), no longer blob aggregation state.
        """
        db = make_landscape_db()
        orch = _make_orchestrator(db)
        run_id = "run-exhausted-source-engine-work"
        _insert_failed_run(db, run_id)
        mock_factory = MagicMock(spec=RecorderFactory)
        mock_factory.run_status_projection.count_distinct_source_rows_with_terminal_outcome.return_value = 0
        mock_factory.run_status_projection.count_failed_coalesce_barrier_rows.return_value = 0
        # ADR-030 §A.3 (slice 4): provide a valid token + healthy heartbeat snapshot.
        coordination_token = _make_heartbeat_safe_token(run_id, mock_factory)

        checkpoint = Checkpoint(
            checkpoint_id="cp-exhausted-source-engine-work",
            run_id=run_id,
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )
        resume_point = ResumePoint(
            checkpoint=checkpoint,
            sequence_number=checkpoint.sequence_number,
        )
        resume_state = ResumeState(
            factory=mock_factory,
            run_id=run_id,
            unprocessed_rows=(),
            incomplete_by_row={},
            recovery_manager=MagicMock(spec=RecoveryManager),
            schema_contracts_by_source={NodeID("source"): MagicMock(spec=SchemaContract)},
            source_names_by_source={NodeID("source"): "source"},
            source_lifecycle_by_source={NodeID("source"): "exhausted"},
            has_restored_barrier_work=True,
            coordination_token=coordination_token,
        )
        resumed_result = RunResult(
            run_id=run_id,
            status=RunStatus.RUNNING,
            rows_processed=0,
            rows_succeeded=0,
            rows_failed=0,
            rows_routed_success=0,
            rows_routed_failure=0,
            rows_quarantined=0,
        )

        admit_guard = _admit_resume_point(orch, resume_point)
        with (
            admit_guard,
            patch.object(orch._resume_coordinator, "reconstruct_resume_state", return_value=resume_state),
            patch.object(orch._resume_coordinator, "process_resumed_rows", return_value=resumed_result) as process_resumed,
            patch.object(orch._ceremony, "emit_telemetry"),
            patch.object(orch._checkpoints, "delete_checkpoints"),
        ):
            result = orch.resume(
                resume_point,
                MagicMock(spec=PipelineConfig),
                MagicMock(spec=ExecutionGraph),
                payload_store=MockPayloadStore(),
            )

        process_resumed.assert_called_once()
        assert result.status == RunStatus.EMPTY


class TestBuildProcessorCallsCleanupOnFailure:
    """Regression test for Phase 0 fix #7: Plugin cleanup skipped.

    Bug: When _build_processor raised after on_start completed for all
    plugins, _cleanup_plugins was never called. This leaked resources
    (DB connections, file handles, thread pools).

    Fix: Wrapped _build_processor in try/except that calls
    _cleanup_plugins(config, ctx, include_source=True) on failure.
    """

    def test_cleanup_plugins_runs_full_teardown(self) -> None:
        """Verify _cleanup_plugins cleans up all plugin types:
        transforms get on_complete + close, sinks get close, source gets close.
        """
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test", config={}, landscape=None)

        config = MagicMock(spec=PipelineConfig)
        tracked_transform = _specced_transform(node_id="transform-1")
        tracked_transform.name = "tracked"
        config.transforms = [tracked_transform]
        config.sinks = {}
        primary_source = _specced_source(node_id="source-1")
        config.sources = {"primary": primary_source}

        cleanup_plugins(config, ctx)

        tracked_transform.on_complete.assert_called_once()
        tracked_transform.close.assert_called_once()
        primary_source.close.assert_called_once()

    def test_build_processor_failure_path_cleans_up_with_source(self) -> None:
        """When build_processor raises inside initialize_run_context,
        cleanup_plugins must be called with include_source matching the
        run path.

        This test exercises the actual except handler in
        RunContextFactory.initialize_run_context, not just cleanup_plugins
        in isolation. The original
        bug leaked already-started plugins — especially the source — because
        the except block didn't exist. A regression to include_source=False
        or removal of the except block will cause this test to fail.
        """
        from elspeth.engine.orchestrator.run_state import GraphArtifacts

        db = make_landscape_db()
        orch = _make_orchestrator(db)

        # Minimal config with trackable plugins
        config = MagicMock(spec=PipelineConfig)
        tracked_source = _specced_source()
        tracked_transform = _specced_transform()
        tracked_transform.name = "tracked"
        tracked_transform.node_id = None
        config.sources["primary"] = tracked_source
        config.sources = {"source": tracked_source}
        config.transforms = [tracked_transform]
        config.sinks = {}
        config.config = {}

        graph = MagicMock(spec=ExecutionGraph)
        graph.get_route_resolution_map.return_value = {}
        settings = MagicMock(spec=ElspethSettings)
        payload_store = MagicMock(spec=PayloadStore)
        mock_factory = MagicMock(spec=RecorderFactory)

        artifacts = GraphArtifacts(
            edge_map={},
            source_id=NodeID("source-1"),
            source_id_map={"source": NodeID("source-1")},
            sink_id_map={},
            transform_id_map={0: NodeID("transform-1")},
            config_gate_id_map={},
            coalesce_id_map={},
        )

        # build_processor fails after on_start has been called on all plugins
        with (
            patch.object(orch._processor_factory, "build_processor", side_effect=RuntimeError("processor build failed")),
            # cleanup_plugins is now a module function; patch it where
            # run_context_factory.py looks it up (the imported name in that
            # module's namespace), not on the instance.
            patch("elspeth.engine.orchestrator.run_context_factory.cleanup_plugins", wraps=cleanup_plugins) as spy_cleanup,
            pytest.raises(RuntimeError, match="processor build failed"),
        ):
            orch._context_factory.initialize_run_context(
                mock_factory,
                "test-run",
                config,
                graph,
                settings,
                artifacts,
                payload_store,
                include_source_on_start=True,
            )

        # Verify _cleanup_plugins was called with include_source=True.
        # This is the key assertion: if someone changes the except handler
        # to pass include_source=False, or removes it, this fails.
        spy_cleanup.assert_called_once()
        call_kwargs = spy_cleanup.call_args
        assert call_kwargs.kwargs.get("include_source") is True, (
            f"cleanup_plugins must be called with include_source=True when source was started. Got: {call_kwargs}"
        )
        # The config passed must be the same config object
        assert call_kwargs.args[0] is config

    def test_transform_start_failure_cleans_only_successfully_started_plugins(self) -> None:
        """A transform whose on_start fails must not receive teardown hooks."""
        from elspeth.engine.orchestrator.run_state import GraphArtifacts

        db = make_landscape_db()
        orch = _make_orchestrator(db)

        config = MagicMock(spec=PipelineConfig)
        started_source = _specced_source(name="source")
        started_transform = _specced_transform(name="started-transform", node_id=None)
        failing_transform = _specced_transform(name="failing-transform", node_id=None)
        failing_transform.on_start.side_effect = RuntimeError("transform startup failed")
        unstarted_sink = _specced_sink(name="unstarted-sink")
        config.sources = {"source": started_source}
        config.transforms = [started_transform, failing_transform]
        config.sinks = {"sink": unstarted_sink}
        config.config = {}

        graph = MagicMock(spec=ExecutionGraph)
        graph.get_route_resolution_map.return_value = {}
        graph.get_aggregation_id_map.return_value = {}
        artifacts = GraphArtifacts(
            edge_map={},
            source_id=NodeID("source-1"),
            source_id_map={"source": NodeID("source-1")},
            sink_id_map={"sink": NodeID("sink-1")},
            transform_id_map={0: NodeID("transform-1"), 1: NodeID("transform-2")},
            config_gate_id_map={},
            coalesce_id_map={},
        )

        with (
            patch.object(orch._processor_factory, "build_processor") as build_processor,
            pytest.raises(RuntimeError, match="transform startup failed"),
        ):
            orch._context_factory.initialize_run_context(
                MagicMock(spec=RecorderFactory),
                "test-run",
                config,
                graph,
                MagicMock(spec=ElspethSettings),
                artifacts,
                MagicMock(spec=PayloadStore),
                include_source_on_start=True,
            )

        build_processor.assert_not_called()
        started_source.on_complete.assert_called_once()
        started_source.close.assert_called_once()
        started_transform.on_complete.assert_called_once()
        started_transform.close.assert_called_once()
        failing_transform.on_complete.assert_not_called()
        failing_transform.close.assert_not_called()
        unstarted_sink.on_start.assert_not_called()
        unstarted_sink.on_complete.assert_not_called()
        unstarted_sink.close.assert_not_called()

    def test_source_start_failure_skips_failing_source_and_unstarted_plugins(self) -> None:
        """A source whose on_start fails is not considered started for cleanup."""
        from elspeth.engine.orchestrator.run_state import GraphArtifacts

        db = make_landscape_db()
        orch = _make_orchestrator(db)

        config = MagicMock(spec=PipelineConfig)
        failing_source = _specced_source(name="failing-source")
        failing_source.on_start.side_effect = RuntimeError("source startup failed")
        unstarted_transform = _specced_transform(name="unstarted-transform", node_id=None)
        unstarted_sink = _specced_sink(name="unstarted-sink")
        config.sources = {"source": failing_source}
        config.transforms = [unstarted_transform]
        config.sinks = {"sink": unstarted_sink}
        config.config = {}

        graph = MagicMock(spec=ExecutionGraph)
        graph.get_route_resolution_map.return_value = {}
        graph.get_aggregation_id_map.return_value = {}
        artifacts = GraphArtifacts(
            edge_map={},
            source_id=NodeID("source-1"),
            source_id_map={"source": NodeID("source-1")},
            sink_id_map={"sink": NodeID("sink-1")},
            transform_id_map={0: NodeID("transform-1")},
            config_gate_id_map={},
            coalesce_id_map={},
        )

        with (
            patch.object(orch._processor_factory, "build_processor") as build_processor,
            pytest.raises(RuntimeError, match="source startup failed"),
        ):
            orch._context_factory.initialize_run_context(
                MagicMock(spec=RecorderFactory),
                "test-run",
                config,
                graph,
                MagicMock(spec=ElspethSettings),
                artifacts,
                MagicMock(spec=PayloadStore),
                include_source_on_start=True,
            )

        build_processor.assert_not_called()
        failing_source.on_complete.assert_not_called()
        failing_source.close.assert_not_called()
        unstarted_transform.on_start.assert_not_called()
        unstarted_transform.on_complete.assert_not_called()
        unstarted_transform.close.assert_not_called()
        unstarted_sink.on_start.assert_not_called()
        unstarted_sink.on_complete.assert_not_called()
        unstarted_sink.close.assert_not_called()

    def test_sink_start_failure_cleans_only_successfully_started_sinks(self) -> None:
        """A sink whose on_start fails must not receive teardown hooks."""
        from elspeth.engine.orchestrator.run_state import GraphArtifacts

        db = make_landscape_db()
        orch = _make_orchestrator(db)

        config = MagicMock(spec=PipelineConfig)
        started_source = _specced_source(name="source")
        started_transform = _specced_transform(name="started-transform", node_id=None)
        started_sink = _specced_sink(name="started-sink")
        failing_sink = _specced_sink(name="failing-sink")
        failing_sink.on_start.side_effect = RuntimeError("sink startup failed")
        config.sources = {"source": started_source}
        config.transforms = [started_transform]
        config.sinks = {"started": started_sink, "failing": failing_sink}
        config.config = {}

        graph = MagicMock(spec=ExecutionGraph)
        graph.get_route_resolution_map.return_value = {}
        graph.get_aggregation_id_map.return_value = {}
        artifacts = GraphArtifacts(
            edge_map={},
            source_id=NodeID("source-1"),
            source_id_map={"source": NodeID("source-1")},
            sink_id_map={"started": NodeID("sink-1"), "failing": NodeID("sink-2")},
            transform_id_map={0: NodeID("transform-1")},
            config_gate_id_map={},
            coalesce_id_map={},
        )

        with (
            patch.object(orch._processor_factory, "build_processor") as build_processor,
            pytest.raises(RuntimeError, match="sink startup failed"),
        ):
            orch._context_factory.initialize_run_context(
                MagicMock(spec=RecorderFactory),
                "test-run",
                config,
                graph,
                MagicMock(spec=ElspethSettings),
                artifacts,
                MagicMock(spec=PayloadStore),
                include_source_on_start=True,
            )

        build_processor.assert_not_called()
        started_source.on_complete.assert_called_once()
        started_source.close.assert_called_once()
        started_transform.on_complete.assert_called_once()
        started_transform.close.assert_called_once()
        started_sink.on_complete.assert_called_once()
        started_sink.close.assert_called_once()
        failing_sink.on_complete.assert_not_called()
        failing_sink.close.assert_not_called()

    def test_startup_hooks_receive_their_plugin_node_id(self) -> None:
        """Each plugin on_start hook must see its own orchestrator-assigned node id."""
        from elspeth.engine.orchestrator.run_state import GraphArtifacts

        db = make_landscape_db()
        orch = _make_orchestrator(db)
        observed: list[tuple[str, str | None]] = []

        config = MagicMock(spec=PipelineConfig)
        source = _specced_source(name="source")
        transform = _specced_transform(name="transform", node_id=None)
        sink = _specced_sink(name="sink")
        source.on_start.side_effect = lambda ctx: observed.append(("source", ctx.node_id))
        transform.on_start.side_effect = lambda ctx: observed.append(("transform", ctx.node_id))
        sink.on_start.side_effect = lambda ctx: observed.append(("sink", ctx.node_id))
        config.sources = {"source": source}
        config.transforms = [transform]
        config.sinks = {"sink": sink}
        config.config = {}
        config.aggregation_settings = None

        graph = MagicMock(spec=ExecutionGraph)
        graph.get_route_resolution_map.return_value = {}
        graph.get_aggregation_id_map.return_value = {}
        artifacts = GraphArtifacts(
            edge_map={},
            source_id=NodeID("source-1"),
            source_id_map={"source": NodeID("source-1")},
            sink_id_map={"sink": NodeID("sink-1")},
            transform_id_map={0: NodeID("transform-1")},
            config_gate_id_map={},
            coalesce_id_map={},
        )
        processor = MagicMock(spec=RowProcessor)

        with patch.object(orch._processor_factory, "build_processor", return_value=(processor, {}, None)):
            run_ctx = orch._context_factory.initialize_run_context(
                MagicMock(spec=RecorderFactory),
                "test-run",
                config,
                graph,
                MagicMock(spec=ElspethSettings),
                artifacts,
                MagicMock(spec=PayloadStore),
                include_source_on_start=True,
            )

        assert observed == [
            ("source", "source-1"),
            ("transform", "transform-1"),
            ("sink", "sink-1"),
        ]
        assert run_ctx.ctx.node_id == "source-1"

    def test_cleanup_hooks_receive_their_plugin_node_id(self) -> None:
        """Each plugin on_complete hook must run under its own node id."""
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test", config={}, landscape=None)
        ctx.node_id = "source-1"
        observed: list[tuple[str, str | None]] = []

        source = _specced_source(name="source", node_id="source-1")
        transform = _specced_transform(name="transform", node_id="transform-1")
        sink = _specced_sink(name="sink", node_id="sink-1")
        source.on_complete.side_effect = lambda hook_ctx: observed.append(("source", hook_ctx.node_id))
        transform.on_complete.side_effect = lambda hook_ctx: observed.append(("transform", hook_ctx.node_id))
        sink.on_complete.side_effect = lambda hook_ctx: observed.append(("sink", hook_ctx.node_id))

        config = MagicMock(spec=PipelineConfig)
        config.sources = {"source": source}
        config.transforms = [transform]
        config.sinks = {"sink": sink}

        cleanup_plugins(config, ctx)

        assert observed == [
            ("transform", "transform-1"),
            ("sink", "sink-1"),
            ("source", "source-1"),
        ]
        assert ctx.node_id == "source-1"

    def test_cleanup_requires_protocol_node_id_attribute_for_lifecycle_scope(self) -> None:
        """Cleanup must fail loudly when a plugin lacks the protocol node_id attribute."""
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test", config={}, landscape=None)
        sink_without_node_id = _specced_sink(name="sink-without-node-id")
        # The factory seeds node_id (lifecycle attribution requires it); this
        # test pins the missing-attribute invariant, so remove it explicitly.
        del sink_without_node_id.node_id
        config = MagicMock(spec=PipelineConfig)
        config.sources = {}
        config.transforms = []
        config.sinks = {"sink": sink_without_node_id}

        with pytest.raises(OrchestrationInvariantError, match="node_id"):
            cleanup_plugins(config, ctx, include_source=False)


class TestCleanupPluginsReRaisesSystemExceptions:
    """Regression test: _cleanup_plugins must re-raise FrameworkBugError/AuditIntegrityError.

    Bug: All 6 except handlers in _cleanup_plugins caught Exception broadly
    and downgraded every error to a cleanup warning. FrameworkBugError and
    AuditIntegrityError indicate system-level corruption (Tier 1 violations)
    and must crash immediately, not be silently downgraded.

    Fix: run_hook() catches TIER_1_ERRORS in a dedicated except clause that
    re-raises before the broad-catch clause can downgrade them — the canonical
    ``except TIER_1_ERRORS: raise`` form documented in contracts/errors.py.
    """

    def test_source_code_has_reraise_guard(self) -> None:
        """Verify cleanup re-raises Tier 1 errors via a TIER_1_ERRORS except clause.

        Structural test: inspect the source to confirm a dedicated
        ``except ...TIER_1_ERRORS:`` handler whose body is a bare ``raise``
        precedes the broad-catch clause, so system-level corruption crashes
        immediately instead of being collected as a cleanup warning.
        """
        import ast
        import inspect
        import textwrap

        source = inspect.getsource(cleanup_plugins)
        # Dedent for consistency (module-level function is already unindented).
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Look for TIER_1_ERRORS usage in the function
        assert "TIER_1_ERRORS" in source, "cleanup_plugins must guard on TIER_1_ERRORS"

        # Find an `except ...TIER_1_ERRORS:` handler whose body re-raises.
        found_reraise = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                handler_type_src = ast.dump(node.type) if node.type is not None else ""
                if "TIER_1_ERRORS" not in handler_type_src:
                    continue
                if any(isinstance(stmt, ast.Raise) for stmt in node.body):
                    found_reraise = True
                    break

        assert found_reraise, "Expected `except TIER_1_ERRORS: raise` handler in cleanup_plugins"

    def test_framework_bug_error_propagates_through_cleanup(self) -> None:
        """FrameworkBugError from plugin.on_complete() must propagate, not be swallowed."""
        from elspeth.contracts import FrameworkBugError
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test", config={}, landscape=None)

        # Create a mock config with a transform that raises FrameworkBugError
        config = MagicMock(spec=PipelineConfig)
        bad_transform = _specced_transform(node_id="transform-1")
        bad_transform.on_complete.side_effect = FrameworkBugError("internal corruption")
        bad_transform.name = "bad_transform"
        config.transforms = [bad_transform]
        config.sinks = {}
        config.sources["primary"] = _specced_source(node_id="source-1")

        with pytest.raises(FrameworkBugError, match="internal corruption"):
            cleanup_plugins(config, ctx)

    def test_audit_integrity_error_propagates_through_cleanup(self) -> None:
        """AuditIntegrityError from sink.close() must propagate, not be swallowed."""
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test", config={}, landscape=None)

        # Create a mock config with a sink that raises AuditIntegrityError on close
        config = MagicMock(spec=PipelineConfig)
        config.transforms = []
        bad_sink = _specced_sink(node_id="sink-1")
        bad_sink.close.side_effect = AuditIntegrityError("audit DB corrupted")
        bad_sink.name = "bad_sink"
        config.sinks = {"output": bad_sink}
        config.sources["primary"] = _specced_source(node_id="source-1")

        with pytest.raises(AuditIntegrityError, match="audit DB corrupted"):
            cleanup_plugins(config, ctx)

    def test_regular_exceptions_still_collected_as_cleanup_errors(self) -> None:
        """Non-system exceptions are still collected and reported as RuntimeError."""
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test", config={}, landscape=None)

        # Create a mock config with a transform that raises a regular error
        config = MagicMock(spec=PipelineConfig)
        bad_transform = _specced_transform(node_id="transform-1")
        bad_transform.on_complete.side_effect = RuntimeError("connection refused")
        bad_transform.name = "flaky_transform"
        config.transforms = [bad_transform]
        config.sinks = {}
        config.sources["primary"] = _specced_source(node_id="source-1")

        with pytest.raises(RuntimeError, match="Plugin cleanup failed"):
            cleanup_plugins(config, ctx)


class TestResumeLoopCoordinationLatch:
    """ADR-030 §A.3 (slice 4): check_coordination_latch is polled per-row in
    run_resume_processing_loop, matching the run() drain-loop pattern in
    source_iteration.py.

    These tests drive the loop directly with a stub processor and a
    configurable latch callable — no real DB, no real heartbeat thread.
    Wall-clock sleeps: zero (latch is a synchronous callable).
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_loop_ctx(sink_name: str = "default") -> LoopContext:
        """Minimal LoopContext with a MagicMock processor that succeeds."""
        processor = MagicMock(spec=RowProcessor)
        processor.has_scheduled_work.return_value = False
        processor.has_unresolved_scheduler_work.return_value = False
        # Return a single sink-bound row result for each process_existing_row call.
        processor.process_existing_row.side_effect = lambda **kwargs: [make_row_result({"v": 1}, sink_name=sink_name)]
        config = PipelineConfig(
            sources={"primary": _specced_source()},
            transforms=(),
            sinks={sink_name: _specced_sink()},
        )
        return LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={sink_name: []},
            processor=processor,
            ctx=MagicMock(spec=PluginContext),
            config=config,
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )

    @staticmethod
    def _one_row(source_node_id: str = "source") -> tuple[ResumedRow, ...]:
        return (
            ResumedRow(
                row_id="row-latch-test",
                row_index=0,
                source_node_id=NodeID(source_node_id),
                row_data={"v": 1},
            ),
        )

    @staticmethod
    def _schema_contracts(source_node_id: str = "source") -> dict[NodeID, MagicMock]:
        return {NodeID(source_node_id): MagicMock(spec=SchemaContract)}

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_latch_callable_is_polled_once_per_row(self) -> None:
        """check_coordination_latch is called exactly once after each row is processed.

        With one unprocessed row the latch must be called exactly once.  With
        zero rows it must not be called at all (the loop body never executes).
        """

        latch_calls: list[int] = []

        def _counting_latch() -> None:
            latch_calls.append(1)

        # Zero rows — latch must NOT be called.
        loop_ctx_0 = self._make_loop_ctx()
        run_resume_processing_loop(
            loop_ctx_0,
            (),
            incomplete_by_row={},
            recovery_manager=MagicMock(spec=RecoveryManager),
            payload_store=MockPayloadStore(),
            run_id="run-latch-zero",
            resume_checkpoint_id="ckpt-0",
            schema_contracts_by_source=self._schema_contracts(),
            source_on_success_by_source={NodeID("source"): "default"},
            check_coordination_latch=_counting_latch,
        )
        assert latch_calls == [], "latch must not be called when there are no rows"

        # One row — latch must be called exactly once.
        loop_ctx_1 = self._make_loop_ctx()
        run_resume_processing_loop(
            loop_ctx_1,
            self._one_row(),
            incomplete_by_row={},
            recovery_manager=MagicMock(spec=RecoveryManager),
            payload_store=MockPayloadStore(),
            run_id="run-latch-one",
            resume_checkpoint_id="ckpt-1",
            schema_contracts_by_source=self._schema_contracts(),
            source_on_success_by_source={NodeID("source"): "default"},
            check_coordination_latch=_counting_latch,
        )
        assert len(latch_calls) == 1, f"latch must be called once per row; got {len(latch_calls)} calls"

    def test_latch_raising_propagates_runworkerevictederror(self) -> None:
        """If the latch callable raises RunWorkerEvictedError it propagates out of the loop.

        This is the proactive-surfacing deliverable (b): a deposed resume-takeover-leader
        raises RunWorkerEvictedError at the per-row boundary without waiting for the
        next fenced write to refuse.
        """
        from elspeth.contracts.errors import RunWorkerEvictedError

        def _evicting_latch() -> None:
            raise RunWorkerEvictedError(worker_id="worker-B", run_id="run-latch-evict")

        loop_ctx = self._make_loop_ctx()
        with pytest.raises(RunWorkerEvictedError) as exc_info:
            run_resume_processing_loop(
                loop_ctx,
                self._one_row(),
                incomplete_by_row={},
                recovery_manager=MagicMock(spec=RecoveryManager),
                payload_store=MockPayloadStore(),
                run_id="run-latch-evict",
                resume_checkpoint_id="ckpt-evict",
                schema_contracts_by_source=self._schema_contracts(),
                source_on_success_by_source={NodeID("source"): "default"},
                check_coordination_latch=_evicting_latch,
            )
        assert exc_info.value.worker_id == "worker-B"
        assert exc_info.value.run_id == "run-latch-evict"

    def test_none_latch_is_safe_no_poll(self) -> None:
        """check_coordination_latch=None (the default) runs without polling — no AttributeError."""
        loop_ctx = self._make_loop_ctx()
        interrupted = run_resume_processing_loop(
            loop_ctx,
            self._one_row(),
            incomplete_by_row={},
            recovery_manager=MagicMock(spec=RecoveryManager),
            payload_store=MockPayloadStore(),
            run_id="run-latch-none",
            resume_checkpoint_id="ckpt-none",
            schema_contracts_by_source=self._schema_contracts(),
            source_on_success_by_source={NodeID("source"): "default"},
            check_coordination_latch=None,
        )
        assert interrupted is False

    def test_latch_polled_after_row_results_accumulated(self) -> None:
        """The latch fires AFTER the row is fully processed and outcomes accumulated.

        This matches the source_iteration.py pattern: the latch is the LAST
        step in the per-row boundary sequence, after coalesce timeouts and
        before the shutdown check.  We verify ordering by asserting the
        processor's process_existing_row was already called when the latch fires.
        """
        from elspeth.contracts.errors import RunWorkerEvictedError

        processing_order: list[str] = []

        # Capture when process_existing_row is called.
        processor = MagicMock(spec=RowProcessor)
        processor.has_scheduled_work.return_value = False
        processor.has_unresolved_scheduler_work.return_value = False

        def _process(**kwargs: object) -> list[object]:
            processing_order.append("process_existing_row")
            return [make_row_result({"v": 1}, sink_name="default")]

        processor.process_existing_row.side_effect = _process

        config = PipelineConfig(
            sources={"primary": _specced_source()},
            transforms=(),
            sinks={"default": _specced_sink()},
        )
        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"default": []},
            processor=processor,
            ctx=MagicMock(spec=PluginContext),
            config=config,
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )

        def _ordering_latch() -> None:
            processing_order.append("latch")
            raise RunWorkerEvictedError(worker_id="w", run_id="run-latch-order")

        with pytest.raises(RunWorkerEvictedError):
            run_resume_processing_loop(
                loop_ctx,
                self._one_row(),
                incomplete_by_row={},
                recovery_manager=MagicMock(spec=RecoveryManager),
                payload_store=MockPayloadStore(),
                run_id="run-latch-order",
                resume_checkpoint_id="ckpt-order",
                schema_contracts_by_source=self._schema_contracts(),
                source_on_success_by_source={NodeID("source"): "default"},
                check_coordination_latch=_ordering_latch,
            )

        assert processing_order == ["process_existing_row", "latch"], (
            f"process_existing_row must complete before latch fires; got: {processing_order}"
        )
