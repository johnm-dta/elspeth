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
from unittest.mock import MagicMock, patch

import pytest

from elspeth.contracts import Checkpoint, NodeID, ResumedRow, ResumePoint, RoutingMode, RunStatus
from elspeth.contracts.audit import DISCARD_SINK_NAME, TokenOutcome
from elspeth.contracts.coalesce_checkpoint import CoalesceCheckpointState
from elspeth.contracts.enums import NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import OrchestrationInvariantError
from elspeth.contracts.runtime_val_manifest import build_runtime_val_manifest
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.types import SinkName
from elspeth.core.canonical import canonical_json
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.orchestrator import PipelineConfig, prepare_for_run
from elspeth.engine.orchestrator.core import Orchestrator
from elspeth.engine.orchestrator.types import ExecutionCounters, LoopContext, LoopResult, ResumeState
from elspeth.testing import make_row_result, make_source_row
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
                schema_contract=MagicMock(name="contract"),
            ),
        }
        prepare_for_run()
        mock_factory.run_lifecycle.get_runtime_val_manifest.return_value = canonical_json(build_runtime_val_manifest())

        # Mock RecoveryManager
        mock_recovery = MagicMock()
        mock_recovery.get_unprocessed_row_data.return_value = [
            ResumedRow(
                row_id="row-1",
                row_index=0,
                source_node_id=NodeID("source-node"),
                row_data={"field": "value"},
            ),
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

    def test_resume_loop_drains_scheduler_work_before_replaying_rows(self) -> None:
        """Persisted scheduler work supersedes the old unprocessed-row replay path."""
        orch = _make_orchestrator(make_landscape_db())
        processor = MagicMock()
        processor.has_scheduled_work.side_effect = [True, False]
        processor.active_scheduled_row_ids.return_value = frozenset({"row-should-not-replay"})
        processor.drain_scheduled_work.return_value = [make_row_result({"value": 1}, sink_name="default")]
        processor.process_existing_row.side_effect = AssertionError("source row replay must not run while scheduler work exists")
        config = PipelineConfig(
            sources={"primary": MagicMock()},
            transforms=(),
            sinks={"default": MagicMock()},
        )
        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"default": []},
            processor=processor,
            ctx=MagicMock(),
            config=config,
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )

        interrupted = orch._run_resume_processing_loop(
            loop_ctx,
            unprocessed_rows=(
                ResumedRow(
                    row_id="row-should-not-replay",
                    row_index=0,
                    source_node_id=NodeID("source"),
                    row_data={"value": 1},
                ),
            ),
            schema_contracts_by_source={NodeID("source"): MagicMock()},
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

        orch = _make_orchestrator(make_landscape_db())
        processor = MagicMock()
        processor.has_scheduled_work.return_value = True
        processor.active_scheduled_row_ids.return_value = frozenset({"row-scheduled"})
        processor.drain_scheduled_work.return_value = [make_row_result({"value": 1}, sink_name="default")]
        processor.process_existing_row.side_effect = AssertionError("mixed scheduler coverage must fail before row replay policy")
        config = PipelineConfig(
            sources={"primary": MagicMock()},
            transforms=(),
            sinks={"default": MagicMock()},
        )
        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"default": []},
            processor=processor,
            ctx=MagicMock(),
            config=config,
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )

        with pytest.raises(AuditIntegrityError, match="row-uncovered"):
            orch._run_resume_processing_loop(
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
                schema_contracts_by_source={NodeID("source"): MagicMock()},
            )

        processor.drain_scheduled_work.assert_not_called()
        processor.process_existing_row.assert_not_called()

    def test_resume_loop_refuses_completion_when_scheduler_work_remains_blocked(self) -> None:
        """Blocked/future scheduler work must survive resume instead of falling back to source replay."""
        orch = _make_orchestrator(make_landscape_db())
        source = MagicMock()
        source.on_success = "default"
        processor = MagicMock()
        processor.run_id = "run-with-blocked-work"
        processor.has_scheduled_work.side_effect = [True, True]
        processor.active_scheduled_row_ids.return_value = frozenset({"row-should-not-replay"})
        processor.drain_scheduled_work.return_value = []
        processor.process_existing_row.side_effect = AssertionError("source row replay must not run while scheduler work remains")
        processor.summarize_scheduled_work.return_value = ("BLOCKED count=1 node=join-results",)
        config = PipelineConfig(
            sources={"source": source},
            transforms=(),
            sinks={"default": MagicMock()},
        )
        artifacts = SimpleNamespace(
            source_id_map={"source": NodeID("source")},
            edge_map={},
            sink_id_map={},
            source_id=NodeID("source"),
        )
        run_ctx = SimpleNamespace(
            processor=processor,
            ctx=MagicMock(),
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )

        with (
            patch.object(orch, "_setup_resume_context", return_value=artifacts),
            patch.object(orch, "_initialize_run_context", return_value=run_ctx),
            patch.object(orch, "_run_transform_runtime_preflights"),
            patch.object(orch, "_flush_and_write_sinks") as flush_sinks,
            pytest.raises(Exception, match="left non-terminal scheduler work after sink durability") as exc_info,
        ):
            orch._process_resumed_rows(
                MagicMock(spec=RecorderFactory),
                "run-with-blocked-work",
                config,
                MagicMock(),
                unprocessed_rows=(
                    ResumedRow(
                        row_id="row-should-not-replay",
                        row_index=0,
                        source_node_id=NodeID("source"),
                        row_data={"value": 1},
                    ),
                ),
                restored_aggregation_state={},
                restored_coalesce_state=None,
                payload_store=MagicMock(),
                schema_contracts_by_source={NodeID("source"): MagicMock()},
            )

        assert isinstance(exc_info.value.__cause__, OrchestrationInvariantError)
        processor.drain_scheduled_work.assert_called_once_with(run_ctx.ctx)
        processor.process_existing_row.assert_not_called()
        flush_sinks.assert_called_once()

    def test_setup_resume_context_uses_all_source_roots(self) -> None:
        """Multi-source resume must build a full source map instead of calling graph.get_sources()[0]."""
        orch = _make_orchestrator(make_landscape_db())
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

        orders_source = MagicMock()
        orders_source.name = "csv"
        orders_source._on_validation_failure = "discard"
        refunds_source = MagicMock()
        refunds_source.name = "csv"
        refunds_source._on_validation_failure = "discard"
        sink = MagicMock()
        sink.name = "json"
        sink._on_write_failure = "discard"
        config = PipelineConfig(
            sources={"orders": orders_source, "refunds": refunds_source},
            transforms=(),
            sinks={"output": sink},
        )
        factory = MagicMock(spec=RecorderFactory)
        factory.data_flow.get_edge_map.return_value = {}

        artifacts = orch._setup_resume_context(factory, "run-multi-source-resume", config, graph)

        assert artifacts.source_id_map == {
            "orders": NodeID("source-orders"),
            "refunds": NodeID("source-refunds"),
        }
        assert artifacts.source_id == NodeID("source-orders")

    def test_resume_loop_uses_source_scoped_contract_for_each_replayed_row(self) -> None:
        """Replayed rows from different sources must keep their source-specific schema contract."""
        orch = _make_orchestrator(make_landscape_db())
        processor = MagicMock()
        processor.has_scheduled_work.return_value = False
        processor.process_existing_row.return_value = []
        config = PipelineConfig(
            sources={"orders": MagicMock(), "refunds": MagicMock()},
            transforms=(),
            sinks={"default": MagicMock()},
        )
        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"default": []},
            processor=processor,
            ctx=MagicMock(),
            config=config,
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )
        orders_contract = MagicMock(name="orders-contract")
        refunds_contract = MagicMock(name="refunds-contract")

        interrupted = orch._run_resume_processing_loop(
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
        source = MagicMock()
        source.name = "csv"
        source.config = {"path": "refunds.csv"}
        source.on_success = "refunds_sink"
        source.output_schema.model_json_schema.return_value = {"type": "object", "properties": {"refund_id": {"type": "string"}}}
        source.get_schema_contract.return_value = MagicMock(name="refunds-contract")
        source.get_field_resolution.return_value = None
        config = PipelineConfig(
            sources={"refunds": source},
            transforms=(),
            sinks={"refunds_sink": MagicMock()},
        )
        factory = MagicMock(spec=RecorderFactory)
        events: list[str] = []

        def _record_run_source(**kwargs):
            events.append(f"record:{kwargs['lifecycle_state']}")

        factory.run_lifecycle.record_run_source.side_effect = _record_run_source
        processor = MagicMock()

        def _process_row(**kwargs):
            events.append("process")
            raise RuntimeError("boom after row persisted")

        processor.process_row.side_effect = _process_row
        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"refunds_sink": []},
            processor=processor,
            ctx=MagicMock(),
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
            patch("elspeth.engine.orchestrator.core.track_operation", _source_operation),
            patch.object(orch, "_load_source_with_events", side_effect=_load_source),
            pytest.raises(RuntimeError, match="boom after row persisted"),
        ):
            orch._run_main_processing_loop(
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

    def test_second_observed_source_persists_resume_contract_before_row_failure(self) -> None:
        """A later source crash must leave source-scoped resume metadata behind."""

        @contextmanager
        def _source_operation(*args, **kwargs):
            yield SimpleNamespace(operation=SimpleNamespace(operation_id="source-op-1"))

        orders_contract = _observed_contract("order_id", int)
        refunds_contract = _observed_contract("refund_id", str)
        orch = _make_orchestrator(make_landscape_db())
        source = MagicMock()
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
            sinks={"refunds_sink": MagicMock()},
        )
        factory = MagicMock(spec=RecorderFactory)
        events: list[str] = []
        factory.run_lifecycle.update_run_source_contract.side_effect = lambda **kwargs: events.append("source_contract")
        factory.data_flow.update_node_output_contract.side_effect = lambda *args, **kwargs: events.append("node_contract")
        processor = MagicMock()

        def _process_row(**kwargs):
            events.append("process")
            assert kwargs["ctx"].contract is refunds_contract
            raise RuntimeError("boom after source contract persisted")

        processor.process_row.side_effect = _process_row
        loop_ctx = LoopContext(
            counters=ExecutionCounters(),
            pending_tokens={"refunds_sink": []},
            processor=processor,
            ctx=MagicMock(contract=orders_contract),
            config=config,
            agg_transform_lookup={},
            coalesce_executor=None,
            coalesce_node_map={},
        )

        def _load_source(run_id_arg, ctx_arg, *, active_source):
            return iter((make_source_row({"refund_id": "r1"}, contract=refunds_contract),))

        with (
            patch("elspeth.engine.orchestrator.core.track_operation", _source_operation),
            patch.object(orch, "_load_source_with_events", side_effect=_load_source),
            pytest.raises(RuntimeError, match="boom after source contract persisted"),
        ):
            orch._run_main_processing_loop(
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
        source = MagicMock()
        source.name = "csv"
        source.on_success = "sink"
        sink = MagicMock()
        config = PipelineConfig(
            sources={"orders": source},
            transforms=(),
            sinks={"sink": sink},
        )
        processor = MagicMock()
        processor.run_id = "run-stuck-scheduler"
        processor.has_scheduled_work.return_value = True
        processor.summarize_scheduled_work.return_value = ("READY count=1 node=transform-normalize",)
        run_ctx = SimpleNamespace(
            processor=processor,
            ctx=MagicMock(),
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
            patch.object(orch, "_initialize_run_context", return_value=run_ctx),
            patch.object(orch, "_run_transform_runtime_preflights"),
            patch.object(
                orch,
                "_run_main_processing_loop",
                return_value=LoopResult(interrupted=False, start_time=0.0, phase_start=0.0, last_progress_time=0.0),
            ),
            patch.object(orch, "_flush_and_write_sinks") as flush_sinks,
            pytest.raises(Exception, match="left non-terminal scheduler work after sink durability") as exc_info,
        ):
            orch._execute_run(
                MagicMock(spec=RecorderFactory),
                "run-stuck-scheduler",
                config,
                MagicMock(),
                payload_store=MagicMock(),
            )

        assert isinstance(exc_info.value.__cause__, OrchestrationInvariantError)
        flush_sinks.assert_called_once()

    def test_reconstruct_resume_state_uses_run_sources_records_for_multi_source_rows(self) -> None:
        """Resume reconstruction must restore schema classes and contracts per source node."""
        db = make_landscape_db()
        orch = _make_orchestrator(db)
        orch._checkpoint_manager = MagicMock()
        run_id = "run-multi-source-reconstruct"
        checkpoint = Checkpoint(
            checkpoint_id="cp-multi-source-reconstruct",
            run_id=run_id,
            token_id="tok-multi-source-reconstruct",
            node_id="node-multi-source-reconstruct",
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
        orders_contract = MagicMock(name="orders-contract")
        refunds_contract = MagicMock(name="refunds-contract")
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
        mock_recovery = MagicMock()
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
        orders_schema = MagicMock(name="OrdersSchema")
        refunds_schema = MagicMock(name="RefundsSchema")

        with (
            patch("elspeth.engine.orchestrator.core.RecorderFactory", return_value=mock_factory),
            patch("elspeth.core.checkpoint.RecoveryManager", return_value=mock_recovery),
            patch("elspeth.engine.orchestrator.core.reconstruct_schema_from_json", side_effect=[orders_schema, refunds_schema]),
        ):
            state = orch._reconstruct_resume_state(resume_point, MockPayloadStore())

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
            schema_contracts_by_source={NodeID("source"): MagicMock()},
            source_names_by_source={NodeID("source"): "source"},
            source_lifecycle_by_source={NodeID("source"): "loaded"},
        )

        with (
            patch.object(orch, "_reconstruct_resume_state", return_value=resume_state),
            patch.object(orch, "_process_resumed_rows", side_effect=AssertionError("empty coalesce state should early-exit")),
            patch.object(
                orch,
                "_derive_resume_terminal_status_from_audit",
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
            schema_contracts_by_source={NodeID("source"): MagicMock()},
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

        db = make_landscape_db()
        orch = _make_orchestrator(db)
        ctx = PluginContext(run_id="test", config={}, landscape=None)

        config = MagicMock()
        tracked_transform = MagicMock()
        tracked_transform.name = "tracked"
        config.transforms = [tracked_transform]
        config.sinks = {}
        primary_source = MagicMock()
        config.sources = {"primary": primary_source}

        orch._cleanup_plugins(config, ctx)

        tracked_transform.on_complete.assert_called_once()
        tracked_transform.close.assert_called_once()
        primary_source.close.assert_called_once()

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
        config.sources["primary"] = tracked_source
        config.sources = {"source": tracked_source}
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
            source_id_map={"source": NodeID("source-1")},
            sink_id_map={},
            transform_id_map={0: NodeID("transform-1")},
            config_gate_id_map={},
            coalesce_id_map={},
        )

        # _build_processor fails after on_start has been called on all plugins
        with (
            patch.object(orch, "_build_processor", side_effect=RuntimeError("processor build failed")),
            patch.object(orch, "_cleanup_plugins", wraps=orch._cleanup_plugins) as spy_cleanup,
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
            f"_cleanup_plugins must be called with include_source=True when source was started. Got: {call_kwargs}"
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

        source = inspect.getsource(Orchestrator._cleanup_plugins)
        # Dedent because getsource preserves indentation from the class
        source = textwrap.dedent(source)
        tree = ast.parse(source)

        # Look for TIER_1_ERRORS usage in the function
        assert "TIER_1_ERRORS" in source, "_cleanup_plugins must use TIER_1_ERRORS guard in record_cleanup_error"

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

        db = make_landscape_db()
        orch = _make_orchestrator(db)
        ctx = PluginContext(run_id="test", config={}, landscape=None)

        # Create a mock config with a transform that raises FrameworkBugError
        config = MagicMock()
        bad_transform = MagicMock()
        bad_transform.on_complete.side_effect = FrameworkBugError("internal corruption")
        bad_transform.name = "bad_transform"
        config.transforms = [bad_transform]
        config.sinks = {}
        config.sources["primary"] = MagicMock()

        with pytest.raises(FrameworkBugError, match="internal corruption"):
            orch._cleanup_plugins(config, ctx)

    def test_audit_integrity_error_propagates_through_cleanup(self) -> None:
        """AuditIntegrityError from sink.close() must propagate, not be swallowed."""
        from elspeth.contracts.errors import AuditIntegrityError
        from elspeth.contracts.plugin_context import PluginContext

        db = make_landscape_db()
        orch = _make_orchestrator(db)
        ctx = PluginContext(run_id="test", config={}, landscape=None)

        # Create a mock config with a sink that raises AuditIntegrityError on close
        config = MagicMock()
        config.transforms = []
        bad_sink = MagicMock()
        bad_sink.close.side_effect = AuditIntegrityError("audit DB corrupted")
        bad_sink.name = "bad_sink"
        config.sinks = {"output": bad_sink}
        config.sources["primary"] = MagicMock()

        with pytest.raises(AuditIntegrityError, match="audit DB corrupted"):
            orch._cleanup_plugins(config, ctx)

    def test_regular_exceptions_still_collected_as_cleanup_errors(self) -> None:
        """Non-system exceptions are still collected and reported as RuntimeError."""
        from elspeth.contracts.plugin_context import PluginContext

        db = make_landscape_db()
        orch = _make_orchestrator(db)
        ctx = PluginContext(run_id="test", config={}, landscape=None)

        # Create a mock config with a transform that raises a regular error
        config = MagicMock()
        bad_transform = MagicMock()
        bad_transform.on_complete.side_effect = RuntimeError("connection refused")
        bad_transform.name = "flaky_transform"
        config.transforms = [bad_transform]
        config.sinks = {}
        config.sources["primary"] = MagicMock()

        with pytest.raises(RuntimeError, match="Plugin cleanup failed"):
            orch._cleanup_plugins(config, ctx)
