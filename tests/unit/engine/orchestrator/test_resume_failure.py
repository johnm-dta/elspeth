"""Regression tests for Phase 0 orchestrator fixes.

#6: Resume leaves RUNNING — when _process_resumed_rows raises a non-shutdown
    exception, the run must be finalized as FAILED (not left as RUNNING).

#7: Plugin cleanup skipped — when _build_processor raises after on_start
    completes, _cleanup_plugins must still be called.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts import Checkpoint, NodeID, ResumePoint, RunStatus
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenOutcome
from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.runtime_val_manifest import build_runtime_val_manifest
from elspeth.core.canonical import canonical_json
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.orchestrator import prepare_for_run
from elspeth.engine.orchestrator.cleanup import cleanup_plugins
from elspeth.engine.orchestrator.core import Orchestrator
from elspeth.engine.orchestrator.types import ExecutionCounters, ResumeState
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.stores import MockPayloadStore


def _make_orchestrator(db: LandscapeDB | None = None) -> Orchestrator:
    """Create an Orchestrator with minimal dependencies."""
    if db is None:
        db = make_landscape_db()
    return Orchestrator(db)


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


class TestResumeFinalizesAsFailed:
    """Regression test for Phase 0 fix #6: Resume leaves RUNNING.

    Bug: If _process_resumed_rows raised a non-GracefulShutdownError
    exception during resume, the run status stayed as RUNNING permanently.
    This blocked future resume attempts since recovery rejects RUNNING status.

    Fix: Added `except Exception` handler in resume() that calls
    recorder.finalize_run(run_id, status=RunStatus.FAILED).
    """

    def test_resume_failure_finalizes_run_as_failed(self) -> None:
        """When _process_resumed_rows raises, run status becomes FAILED."""
        db = make_landscape_db()
        orch = _make_orchestrator(db)

        # Mock the checkpoint manager requirement
        orch._checkpoint_manager = MagicMock()

        # Create a mock resume_point
        resume_point = MagicMock()
        resume_point.checkpoint.run_id = "test-run-123"
        resume_point.aggregation_state = None
        resume_point.node_id = "node-1"

        # Create mock config and graph
        config = MagicMock()
        graph = MagicMock()
        payload_store = MagicMock()
        settings = MagicMock()

        # Mock factory to capture finalize_run calls
        mock_factory = MagicMock(spec=RecorderFactory)
        mock_factory.run_lifecycle.get_source_schema.return_value = '{"mode": "observed"}'
        mock_factory.run_lifecycle.get_run_contract.return_value = MagicMock()
        prepare_for_run()
        mock_factory.run_lifecycle.get_runtime_val_manifest.return_value = canonical_json(build_runtime_val_manifest())

        # Mock RecoveryManager
        mock_recovery = MagicMock()
        mock_recovery.get_unprocessed_row_data.return_value = [
            ("row-1", 0, {"field": "value"}),
        ]

        # Make _process_resumed_rows raise a RuntimeError (non-shutdown)
        with (
            patch.object(orch, "_process_resumed_rows", side_effect=RuntimeError("test failure")),
            patch("elspeth.engine.orchestrator.core.RecorderFactory", return_value=mock_factory),
            patch("elspeth.engine.orchestrator.core.reconstruct_schema_from_json", return_value=MagicMock()),
            patch("elspeth.core.checkpoint.RecoveryManager", return_value=mock_recovery),
            patch.object(orch, "_emit_telemetry"),
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

    def test_resume_treats_empty_coalesce_checkpoint_as_all_rows_processed(self) -> None:
        """Empty restored coalesce state must not force a resume processing pass."""
        db = make_landscape_db()
        orch = _make_orchestrator(db)
        run_id = "run-empty-coalesce-state"
        empty_coalesce_state = CoalesceCheckpointState(version="4.0", pending=(), completed_keys=())
        mock_factory = MagicMock(spec=RecorderFactory)
        mock_factory.data_flow.sweep_deferred_invariants_or_crash = MagicMock(spec=object)
        mock_factory.run_lifecycle.finalize_run = MagicMock(spec=object)

        checkpoint = Checkpoint(
            checkpoint_id="cp-empty-coalesce-state",
            run_id=run_id,
            token_id="tok-empty-coalesce-state",
            node_id="node-empty-coalesce-state",
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            checkpoint_node_config_hash="b" * 64,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )
        resume_point = ResumePoint(
            checkpoint=checkpoint,
            token_id=checkpoint.token_id,
            node_id=checkpoint.node_id,
            sequence_number=checkpoint.sequence_number,
            coalesce_state=empty_coalesce_state,
        )
        resume_state = ResumeState(
            factory=mock_factory,
            run_id=run_id,
            restored_aggregation_state={},
            restored_coalesce_state=empty_coalesce_state,
            unprocessed_rows=(),
            schema_contract=MagicMock(spec=object),
            incomplete_by_row={},
            recovery_manager=MagicMock(),
        )

        with (
            patch.object(orch, "_reconstruct_resume_state", return_value=resume_state),
            patch.object(orch, "_process_resumed_rows", side_effect=AssertionError("empty coalesce state should early-exit")),
            patch(
                "elspeth.engine.orchestrator.core.derive_resume_terminal_status_from_audit",
                return_value=(RunStatus.COMPLETED, ExecutionCounters(rows_processed=3, rows_succeeded=3)),
            ),
            patch.object(orch, "_emit_telemetry"),
            patch.object(orch, "_delete_checkpoints"),
        ):
            result = orch.resume(
                resume_point,
                MagicMock(spec=object),
                MagicMock(spec=object),
                payload_store=MockPayloadStore(),
            )

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 3
        mock_factory.run_lifecycle.finalize_run.assert_called_once_with(run_id, status=RunStatus.COMPLETED)

    def test_all_rows_processed_resume_replays_structural_counters_from_audit(self) -> None:
        """All-terminal resume must not fabricate structural counters as zero."""
        db = make_landscape_db()
        orch = _make_orchestrator(db)
        run_id = "run-structural-counter-resume"
        mock_factory = MagicMock(spec=RecorderFactory)
        mock_factory.data_flow.sweep_deferred_invariants_or_crash = MagicMock()
        mock_factory.run_lifecycle.finalize_run = MagicMock()
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
            token_id="tok-structural-counter-resume",
            node_id="node-structural-counter-resume",
            sequence_number=1,
            created_at=datetime.now(UTC),
            upstream_topology_hash="a" * 64,
            checkpoint_node_config_hash="b" * 64,
            format_version=Checkpoint.CURRENT_FORMAT_VERSION,
        )
        resume_point = ResumePoint(
            checkpoint=checkpoint,
            token_id=checkpoint.token_id,
            node_id=checkpoint.node_id,
            sequence_number=checkpoint.sequence_number,
        )
        resume_state = ResumeState(
            factory=mock_factory,
            run_id=run_id,
            restored_aggregation_state={},
            restored_coalesce_state=None,
            unprocessed_rows=(),
            schema_contract=MagicMock(),
            incomplete_by_row={},
            recovery_manager=MagicMock(),
        )

        with (
            patch.object(orch, "_reconstruct_resume_state", return_value=resume_state),
            patch.object(orch, "_process_resumed_rows", side_effect=AssertionError("all-terminal resume should early-exit")),
            patch.object(orch, "_emit_telemetry"),
            patch.object(orch, "_delete_checkpoints"),
        ):
            result = orch.resume(
                resume_point,
                MagicMock(),
                MagicMock(),
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

        config = MagicMock()
        tracked_transform = MagicMock()
        tracked_transform.name = "tracked"
        config.transforms = [tracked_transform]
        config.sinks = {}
        config.source = MagicMock()

        cleanup_plugins(config, ctx)

        tracked_transform.on_complete.assert_called_once()
        tracked_transform.close.assert_called_once()
        config.source.close.assert_called_once()

    def test_build_processor_failure_path_cleans_up_with_source(self) -> None:
        """When _build_processor raises inside _initialize_run_context,
        _cleanup_plugins must be called with include_source matching the
        run path.

        This test exercises the actual except handler in _initialize_run_context
        (line 1665-1667), not just _cleanup_plugins in isolation. The original
        bug leaked already-started plugins — especially the source — because
        the except block didn't exist. A regression to include_source=False
        or removal of the except block will cause this test to fail.
        """
        from elspeth.engine.orchestrator.types import GraphArtifacts

        db = make_landscape_db()
        orch = _make_orchestrator(db)

        # Minimal config with trackable plugins
        config = MagicMock()
        tracked_source = MagicMock()
        tracked_transform = MagicMock()
        tracked_transform.name = "tracked"
        tracked_transform.node_id = None
        config.source = tracked_source
        config.transforms = [tracked_transform]
        config.sinks = {}
        config.config = {}

        graph = MagicMock()
        graph.get_route_resolution_map.return_value = {}
        settings = MagicMock()
        payload_store = MagicMock()
        mock_factory = MagicMock(spec=RecorderFactory)

        artifacts = GraphArtifacts(
            edge_map={},
            source_id=NodeID("source-1"),
            sink_id_map={},
            transform_id_map={0: NodeID("transform-1")},
            config_gate_id_map={},
            coalesce_id_map={},
        )

        # _build_processor fails after on_start has been called on all plugins
        with (
            patch.object(orch, "_build_processor", side_effect=RuntimeError("processor build failed")),
            # cleanup_plugins is now a module function; patch it where core.py looks
            # it up (the imported name in core's namespace), not on the instance.
            patch("elspeth.engine.orchestrator.core.cleanup_plugins", wraps=cleanup_plugins) as spy_cleanup,
            pytest.raises(RuntimeError, match="processor build failed"),
        ):
            orch._initialize_run_context(
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


class TestCleanupPluginsReRaisesSystemExceptions:
    """Regression test: _cleanup_plugins must re-raise FrameworkBugError/AuditIntegrityError.

    Bug: All 6 except handlers in _cleanup_plugins caught Exception broadly
    and downgraded every error to a cleanup warning. FrameworkBugError and
    AuditIntegrityError indicate system-level corruption (Tier 1 violations)
    and must crash immediately, not be silently downgraded.

    Fix: record_cleanup_error() checks isinstance before logging and re-raises
    system-level exceptions.
    """

    def test_source_code_has_reraise_guard(self) -> None:
        """Verify record_cleanup_error re-raises Tier 1 errors via TIER_1_ERRORS.

        Structural test: inspect the source to confirm the isinstance check
        with TIER_1_ERRORS exists inside record_cleanup_error.
        """
        import ast
        import inspect
        import textwrap

        source = inspect.getsource(cleanup_plugins)
        # Dedent for consistency (module-level function is already unindented).
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Look for TIER_1_ERRORS usage in the function
        assert "TIER_1_ERRORS" in source, "cleanup_plugins must use TIER_1_ERRORS guard in record_cleanup_error"

        # Find a Raise inside an If that checks isinstance with TIER_1_ERRORS
        found_reraise = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if_source = ast.dump(node)
                if "isinstance" in if_source and "TIER_1_ERRORS" in if_source:
                    # Check that the if body contains a raise
                    for child in ast.walk(node):
                        if isinstance(child, ast.Raise):
                            found_reraise = True
                            break

        assert found_reraise, "Expected isinstance(error, TIER_1_ERRORS) guard with raise inside record_cleanup_error"

    def test_framework_bug_error_propagates_through_cleanup(self) -> None:
        """FrameworkBugError from plugin.on_complete() must propagate, not be swallowed."""
        from elspeth.contracts import FrameworkBugError
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test", config={}, landscape=None)

        # Create a mock config with a transform that raises FrameworkBugError
        config = MagicMock()
        bad_transform = MagicMock()
        bad_transform.on_complete.side_effect = FrameworkBugError("internal corruption")
        bad_transform.name = "bad_transform"
        config.transforms = [bad_transform]
        config.sinks = {}
        config.source = MagicMock()

        with pytest.raises(FrameworkBugError, match="internal corruption"):
            cleanup_plugins(config, ctx)

    def test_audit_integrity_error_propagates_through_cleanup(self) -> None:
        """AuditIntegrityError from sink.close() must propagate, not be swallowed."""
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test", config={}, landscape=None)

        # Create a mock config with a sink that raises AuditIntegrityError on close
        config = MagicMock()
        config.transforms = []
        bad_sink = MagicMock()
        bad_sink.close.side_effect = AuditIntegrityError("audit DB corrupted")
        bad_sink.name = "bad_sink"
        config.sinks = {"output": bad_sink}
        config.source = MagicMock()

        with pytest.raises(AuditIntegrityError, match="audit DB corrupted"):
            cleanup_plugins(config, ctx)

    def test_regular_exceptions_still_collected_as_cleanup_errors(self) -> None:
        """Non-system exceptions are still collected and reported as RuntimeError."""
        from elspeth.contracts.plugin_context import PluginContext

        ctx = PluginContext(run_id="test", config={}, landscape=None)

        # Create a mock config with a transform that raises a regular error
        config = MagicMock()
        bad_transform = MagicMock()
        bad_transform.on_complete.side_effect = RuntimeError("connection refused")
        bad_transform.name = "flaky_transform"
        config.transforms = [bad_transform]
        config.sinks = {}
        config.source = MagicMock()

        with pytest.raises(RuntimeError, match="Plugin cleanup failed"):
            cleanup_plugins(config, ctx)
