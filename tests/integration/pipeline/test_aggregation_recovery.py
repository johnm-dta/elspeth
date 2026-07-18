# tests/integration/pipeline/test_aggregation_recovery.py
"""Integration tests for aggregation crash recovery — journal era (F1).

The recovery SCENARIOS predate F1 and are preserved; the SURFACES are the
scheduler journal (token_work_items BLOCKED/PENDING_SINK rows) instead of the
deleted checkpoint aggregation-state blob:

1. A flush output crash window (sink write fails AFTER the flush consumed its
   inputs) leaves the output journal-durable — the F1/D6 atomic
   ``complete_barrier`` guarantee — and resume delivers it WITHOUT re-running
   the batch transform or the source.
2. Crash-during-flush machinery: incomplete batches are found, EXECUTING
   batches fail-then-retry, member ordinals survive retry, and only failed
   batches may be retried.
3. Trigger-timeout SLA preservation (Bug #6): the journal restore derives
   batch age from ``barrier_blocked_at`` of the oldest BLOCKED row, so a
   60s-timeout batch that crashed 30s in fires after 30 more seconds, not 60.

NOTE: Manual node registration (raw SQL) is intentional in the synthetic
batch-machinery tests. The real-path tests use ``from_plugin_instances``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select, update

from elspeth.contracts import (
    Determinism,
    PipelineRow,
    RestrictedSinkEffectContext,
    RunStatus,
    SinkEffectCommitResult,
    SinkEffectPlan,
)
from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.contracts.enums import BatchStatus, NodeType
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.results import SourceRow
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.types import AggregationName, NodeID
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.config import AggregationSettings, CheckpointSettings, SourceSettings, TriggerConfig
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import batch_members_table, batches_table, sink_effects_table, token_work_items_table
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestSchema, _TestSourceBase, as_sink, as_source, as_transform
from tests.fixtures.factories import wire_transforms
from tests.fixtures.landscape import make_factory
from tests.fixtures.plugins import CollectSink, ListSource
from tests.helpers.checkpoint import create_checkpoint


def _create_test_schema_contract() -> SchemaContract:
    """Create a minimal schema contract for test run creation."""
    field_contracts = (
        FieldContract(
            normalized_name="test_field",
            original_name="test_field",
            python_type=str,
            required=True,
            source="declared",
        ),
    )
    return SchemaContract(fields=field_contracts, mode="FIXED", locked=True)


# =============================================================================
# Real-path fixtures (crash-window proof)
# =============================================================================


class _LoadCountingSource(_TestSourceBase):
    """Source that counts load() invocations — the source-replay tripwire."""

    name = "load_counting_source"
    output_schema = ListSource.output_schema

    def __init__(self, rows: list[dict[str, int]], *, on_success: str) -> None:
        super().__init__()
        self._rows = rows
        self.on_success = on_success
        self.load_invocations = 0

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        self.load_invocations += 1
        for source_row_index, row in enumerate(self._rows):
            fields = tuple(
                FieldContract(
                    normalized_name=key,
                    original_name=key,
                    python_type=object,
                    required=False,
                    source="inferred",
                )
                for key in row
            )
            contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
            self._schema_contract = contract
            yield SourceRow.valid(row, contract=contract, source_row_index=source_row_index)


class _SumBatchTransform(BaseTransform):
    """Batch transform that sums its EOF batch; counts batch invocations."""

    name = "sum_batch"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True
    on_success = "output"
    on_error = "discard"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.batch_calls = 0

    def process(self, row: PipelineRow | list[PipelineRow], ctx: Any) -> TransformResult:
        if isinstance(row, list):
            self.batch_calls += 1
            total = sum(r.get("value", 0) for r in row)
            contract = SchemaContract(
                mode="OBSERVED",
                fields=(
                    FieldContract(normalized_name="value", original_name="value", python_type=int, required=False, source="inferred"),
                    FieldContract(normalized_name="count", original_name="count", python_type=int, required=False, source="inferred"),
                ),
                locked=True,
            )
            return TransformResult.success(
                PipelineRow({"value": total, "count": len(row)}, contract),
                success_reason={"action": "batch_sum"},
            )
        return TransformResult.success(row, success_reason={"action": "buffer"})


class _FailOnceSink(CollectSink):
    """Sink whose first effect commit crashes in the post-flush window.

    The crash is injected in the SINK (not a repository mock): the aggregation
    flush has fully completed and consumed its inputs by the time write() runs,
    so the raise lands exactly in the out-of-claim flush crash window (NOT the
    accept-then-crash-before-mark_blocked window — D7 guard).
    """

    def __init__(self, name: str = "output") -> None:
        super().__init__(name)
        self._fail_next_write = True

    def write(self, rows: Any, ctx: Any) -> Any:
        if self._fail_next_write:
            self._fail_next_write = False
            raise RuntimeError("injected sink write crash")
        return super().write(rows, ctx)

    def commit_effect(
        self,
        plan: SinkEffectPlan,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectCommitResult:
        if self._fail_next_write:
            self._fail_next_write = False
            raise RuntimeError("injected sink write crash")
        return super().commit_effect(plan, ctx)


class _CountingPassTransform(BaseTransform):
    """Downstream transform whose call count proves continuation exactness."""

    name = "counting_pass"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema
    on_success = "output"
    on_error = "discard"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.process_calls = 0

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        self.process_calls += 1
        return TransformResult.success(row, success_reason={"action": "counted_passthrough"})


def _build_eof_aggregation_pipeline(
    source: Any,
    transform: _SumBatchTransform,
    output_sink: CollectSink,
    downstream: _CountingPassTransform | None = None,
) -> tuple[PipelineConfig, ExecutionGraph]:
    """Count-triggered transform-mode aggregation whose flush only fires at EOF."""
    aggregation_output = "aggregate_ready" if downstream is not None else "output"
    transform.on_success = aggregation_output
    agg_settings = AggregationSettings(
        name="eof_sum",
        plugin=transform.name,
        input="batch_in",
        on_success=aggregation_output,
        on_error="discard",
        trigger=TriggerConfig(count=100, timeout_seconds=3600),
        output_mode="transform",
    )
    wired_transforms = (
        []
        if downstream is None
        else wire_transforms(
            [as_transform(downstream)],
            source_connection=aggregation_output,
            final_sink="output",
            names=[downstream.name],
        )
    )

    graph = ExecutionGraph.from_plugin_instances(
        sources={"primary": as_source(source)},
        source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="batch_in", options={})},
        transforms=wired_transforms,
        sinks={"output": as_sink(output_sink)},
        aggregations={"eof_sum": (as_transform(transform), agg_settings)},
        gates=[],
    )
    agg_node_id = graph.get_aggregation_id_map()[AggregationName("eof_sum")]
    transform.node_id = agg_node_id

    config = PipelineConfig(
        sources={"primary": as_source(source)},
        # Regular transforms must retain their graph sequence before the
        # graph-confirmed aggregation plugin, whose node_id is pre-assigned.
        transforms=([as_transform(downstream)] if downstream is not None else []) + [as_transform(transform)],
        sinks={"output": as_sink(output_sink)},
        aggregation_settings={agg_node_id: agg_settings},
    )
    return config, graph


@pytest.mark.timeout(120)
class TestFlushOutputJournalDurability:
    """F1/D6: the out-of-claim flush output is journal-durable before the sink write."""

    def test_timeout_flush_output_is_journal_durable_before_sink_write(self, tmp_path: Any) -> None:
        """A timeout/EOF flush output survives a sink-write crash via its PENDING_SINK row.

        Drives the out-of-claim flush arm (``handle_timeout_flush``, trigger
        END_OF_SOURCE — the same arm TIMEOUT flushes take) through a sink
        write that crashes. The atomic ``complete_barrier`` transition must
        have left, in ONE journal transaction:

        - every consumed input token's BLOCKED row TERMINAL, and
        - the emitted aggregate token's PENDING_SINK row on the node_id-NULL
          terminal lane, under the emitted token's REAL token_id —

        so there is no observable state where the inputs are consumed and the
        output is absent. Resume then delivers the output from the journal
        WITHOUT re-running the batch transform or re-invoking the source.
        """
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))

        source = _LoadCountingSource([{"value": 10}, {"value": 20}, {"value": 30}], on_success="batch_in")
        transform = _SumBatchTransform()
        output_sink = _FailOnceSink("output")
        config, graph = _build_eof_aggregation_pipeline(source, transform, output_sink)

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        with pytest.raises(RuntimeError, match="injected sink write crash"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)

        # The flush ran exactly once and nothing reached the sink.
        assert transform.batch_calls == 1
        assert output_sink.results == []

        with db.connection() as conn:
            run_id = str(conn.execute(select(batches_table.c.run_id)).scalars().first())
            consumed_token_ids = set(
                conn.execute(
                    select(batch_members_table.c.token_id)
                    .join(batches_table, batch_members_table.c.batch_id == batches_table.c.batch_id)
                    .where(batches_table.c.run_id == run_id)
                )
                .scalars()
                .all()
            )
            journal = conn.execute(
                select(
                    token_work_items_table.c.token_id,
                    token_work_items_table.c.status,
                    token_work_items_table.c.node_id,
                    token_work_items_table.c.pending_sink_name,
                ).where(token_work_items_table.c.run_id == run_id)
            ).all()

        assert len(consumed_token_ids) == 3
        statuses_by_token = {row.token_id: row.status for row in journal}

        # Consumed inputs: TERMINAL — and emphatically NOT still blocked.
        for token_id in consumed_token_ids:
            assert statuses_by_token[token_id] == "terminal", f"consumed input {token_id} is {statuses_by_token[token_id]!r}"

        # The emitted aggregate: exactly one PENDING_SINK row, node_id-NULL
        # terminal lane, real (new) token_id, sink-handoff metadata present.
        pending_rows = [row for row in journal if row.status == "pending_sink"]
        assert len(pending_rows) == 1, f"expected exactly one durable flush output, got {pending_rows!r}"
        emitted = pending_rows[0]
        assert emitted.node_id is None
        assert emitted.pending_sink_name == "output"
        assert emitted.token_id not in consumed_token_ids

        # No state where inputs are consumed and the output is absent:
        # nothing remains blocked at the barrier.
        assert not [row for row in journal if row.status == "blocked"]

        # A real replacement process may take over an external-effect lease
        # only after the crashed worker's lease expires.  The injected
        # RuntimeError unwinds in-process, so advance that crash-recovery
        # boundary explicitly instead of making resume steal a live lease.
        expired_at = datetime.now(UTC) - timedelta(seconds=1)
        with db.write_connection() as conn:
            expired = conn.execute(
                update(sink_effects_table)
                .where(
                    sink_effects_table.c.run_id == run_id,
                    sink_effects_table.c.state == "in_flight",
                )
                .values(
                    lease_heartbeat_at=expired_at - timedelta(seconds=1),
                    lease_expires_at=expired_at,
                )
            )
        assert expired.rowcount == 1

        # ── Resume: the journal-durable output reaches the sink without
        # re-running the transform or the source.
        recovery = RecoveryManager(db, checkpoint_mgr)
        check = recovery.can_resume(run_id, graph)
        assert check.can_resume, f"Expected resumable run, got: {check.reason}"
        resume_point = recovery.get_resume_point(run_id, graph)
        assert resume_point is not None

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        assert result.status == RunStatus.COMPLETED
        assert output_sink.results == [{"value": 60, "count": 3}]
        assert transform.batch_calls == 1, "resume must deliver the journal-durable output, not re-run the flush"
        assert source.load_invocations == 1, "resume must not re-invoke the source plugin"

        with db.connection() as conn:
            work_statuses = (
                conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == run_id)).scalars().all()
            )
        assert work_statuses
        assert set(work_statuses) <= {"terminal"}

    def test_non_sink_flush_atomic_ready_child_reconciles_in_process(self, tmp_path: Any) -> None:
        """The live continuation loop reconciles the READY row instead of duplicating it."""
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
        source = _LoadCountingSource([{"value": 10}, {"value": 20}, {"value": 30}], on_success="batch_in")
        batch_transform = _SumBatchTransform()
        downstream = _CountingPassTransform()
        output_sink = CollectSink("output")
        config, graph = _build_eof_aggregation_pipeline(source, batch_transform, output_sink, downstream)
        downstream_node_id = graph.get_transform_id_map()[0]

        result = Orchestrator(db).run(
            config,
            graph=graph,
            payload_store=FilesystemPayloadStore(tmp_path / "payloads"),
        )

        assert result.status == RunStatus.COMPLETED
        assert output_sink.results == [{"value": 60, "count": 3}]
        assert downstream.process_calls == 1
        assert batch_transform.batch_calls == 1
        assert source.load_invocations == 1

        with db.connection() as conn:
            journal = conn.execute(
                select(
                    token_work_items_table.c.node_id,
                    token_work_items_table.c.status,
                ).where(token_work_items_table.c.run_id == result.run_id)
            ).all()
        assert len([row for row in journal if row.node_id == downstream_node_id]) == 1
        assert {row.status for row in journal} <= {"terminal"}

    def test_non_sink_flush_crash_after_barrier_completion_resumes_child_exactly_once(
        self,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A crash after TS-15 cannot lose a non-sink aggregation continuation."""
        import elspeth.engine.orchestrator.aggregation as aggregation_runtime
        from elspeth.core.payload_store import FilesystemPayloadStore

        db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))

        source = _LoadCountingSource([{"value": 10}, {"value": 20}, {"value": 30}], on_success="batch_in")
        batch_transform = _SumBatchTransform()
        downstream = _CountingPassTransform()
        output_sink = CollectSink("output")
        config, graph = _build_eof_aggregation_pipeline(source, batch_transform, output_sink, downstream)
        downstream_node_id = graph.get_transform_id_map()[0]

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        real_process_flush_results = aggregation_runtime._process_flush_results

        def crash_before_continuation(*args: Any, **kwargs: Any) -> None:
            work_items = args[1]
            assert len(work_items) == 1
            assert work_items[0].current_node_id == downstream_node_id
            raise RuntimeError("injected crash after aggregation barrier completion")

        monkeypatch.setattr(aggregation_runtime, "_process_flush_results", crash_before_continuation)
        with pytest.raises(RuntimeError, match="injected crash after aggregation barrier completion"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)
        monkeypatch.setattr(aggregation_runtime, "_process_flush_results", real_process_flush_results)

        assert source.load_invocations == 1
        assert batch_transform.batch_calls == 1
        assert downstream.process_calls == 0
        assert output_sink.results == []

        with db.connection() as conn:
            run_id = str(conn.execute(select(batches_table.c.run_id)).scalars().first())
            journal = conn.execute(
                select(
                    token_work_items_table.c.token_id,
                    token_work_items_table.c.node_id,
                    token_work_items_table.c.status,
                ).where(token_work_items_table.c.run_id == run_id)
            ).all()

        ready_rows = [row for row in journal if row.status == "ready"]
        assert len(ready_rows) == 1
        assert ready_rows[0].node_id == downstream_node_id
        assert len([row for row in journal if row.status == "terminal"]) == 3
        assert not [row for row in journal if row.status == "blocked"]

        recovery = RecoveryManager(db, checkpoint_mgr)
        check = recovery.can_resume(run_id, graph)
        assert check.can_resume, f"Expected resumable run, got: {check.reason}"
        resume_point = recovery.get_resume_point(run_id, graph)
        assert resume_point is not None

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        assert result.status == RunStatus.COMPLETED
        assert output_sink.results == [{"value": 60, "count": 3}]
        assert downstream.process_calls == 1
        assert batch_transform.batch_calls == 1, "resume must not re-run the completed aggregation flush"
        assert source.load_invocations == 1, "resume must not re-invoke the source plugin"

        with db.connection() as conn:
            work_statuses = (
                conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == run_id)).scalars().all()
            )
        assert work_statuses
        assert set(work_statuses) <= {"terminal"}


class _FailBatchTransform(BaseTransform):
    """Batch transform that FAILS its EOF flush by returning an error-status result.

    Returning (not raising) an error TransformResult drives the failure arm of
    ``handle_timeout_flush`` (processor.py: ``result.status != "success"``),
    which records terminal FAILURE/UNROUTED token_outcomes via
    ``_handle_flush_error`` and then releases the BLOCKED scheduler rows via
    ``_mark_buffered_scheduler_work_terminal`` — the two-transaction split this
    test crashes between.
    """

    name = "fail_batch"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True
    on_success = "output"
    on_error = "discard"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self.batch_calls = 0

    def process(self, row: PipelineRow | list[PipelineRow], ctx: Any) -> TransformResult:
        if isinstance(row, list):
            self.batch_calls += 1
            return TransformResult.error({"reason": "injected batch flush failure"})
        return TransformResult.success(row, success_reason={"action": "buffer"})


@pytest.mark.timeout(120)
class TestFailedFlushReconcile:
    """ADR-030 §E.3a (aggregation mirror): a FAILED out-of-claim flush that
    crashes between the terminal-outcome write and the BLOCKED-row release must
    not brick resume (elspeth-55546a6fd6)."""

    def test_failed_flush_crash_between_terminal_write_and_release_resumes(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Crash in the FAILED-flush two-transaction window is reconciled at restore.

        Reproduces the brick: ``_handle_flush_error`` commits terminal
        FAILURE/UNROUTED token_outcomes (completed=1) for every buffered token,
        then a crash strikes before ``_mark_buffered_scheduler_work_terminal``
        releases the durable BLOCKED scheduler rows. On resume the orphaned
        BLOCKED rows partition to the aggregation node, but
        ``list_live_buffered_outcomes`` excludes completed-witness tokens, so
        ``_derive_restored_batch_id`` historically raised
        ``AuditIntegrityError('...no matching BUFFERED token_outcome...')`` on
        EVERY attempt — the run was permanently unresumable.

        The restore-side aggregation reconcile (mirror of the coalesce §E.3a
        holdless path) must instead journal-release the orphaned BLOCKED rows
        (their tokens are already terminal) and let the run complete.
        """
        from elspeth.core.landscape.schema import token_outcomes_table
        from elspeth.core.payload_store import FilesystemPayloadStore
        from elspeth.engine.processor import RowProcessor

        db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))

        source = _LoadCountingSource([{"value": 10}, {"value": 20}, {"value": 30}], on_success="batch_in")
        transform = _FailBatchTransform()
        output_sink = CollectSink("output")
        config, graph = _build_eof_aggregation_pipeline(source, transform, output_sink)

        # Crash injection: replace the BLOCKED-row release with a raise. By the
        # time it runs, _handle_flush_error has already committed the terminal
        # FAILURE outcomes — landing the crash squarely in the two-transaction
        # window.
        def _crash_before_release(self: RowProcessor, node_id: Any, tokens: Any) -> None:
            raise RuntimeError("injected crash before BLOCKED-row release")

        monkeypatch.setattr(RowProcessor, "_mark_buffered_scheduler_work_terminal", _crash_before_release)

        orchestrator = Orchestrator(
            db=db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        with pytest.raises(RuntimeError, match="injected crash before BLOCKED-row release"):
            orchestrator.run(config, graph=graph, payload_store=payload_store)

        assert transform.batch_calls == 1

        # ── Confirm the reproduction: terminal FAILURE outcomes are durable AND
        # the BLOCKED scheduler rows leaked (the crash-window signature).
        with db.connection() as conn:
            run_id = str(conn.execute(select(token_outcomes_table.c.run_id)).scalars().first())
            terminal_failures = (
                conn.execute(
                    select(token_outcomes_table.c.token_id)
                    .where(token_outcomes_table.c.run_id == run_id)
                    .where(token_outcomes_table.c.completed == 1)
                    .where(token_outcomes_table.c.path == "unrouted")
                )
                .scalars()
                .all()
            )
            blocked_tokens = (
                conn.execute(
                    select(token_work_items_table.c.token_id)
                    .where(token_work_items_table.c.run_id == run_id)
                    .where(token_work_items_table.c.status == "blocked")
                )
                .scalars()
                .all()
            )

        assert len(terminal_failures) == 3, "all three buffered tokens must be terminally FAILED before the crash"
        assert set(blocked_tokens) == set(terminal_failures), "the FAILED tokens' BLOCKED scheduler rows must have leaked"

        # ── Resume must NOT brick. The reconcile journal-releases the orphaned
        # BLOCKED rows and the run completes.
        recovery = RecoveryManager(db, checkpoint_mgr)
        check = recovery.can_resume(run_id, graph)
        assert check.can_resume, f"Expected resumable run, got: {check.reason}"
        resume_point = recovery.get_resume_point(run_id, graph)
        assert resume_point is not None

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        # The run is no longer bricked: resume finalizes to its truthful,
        # audit-derived terminal status. All three rows genuinely FAILED in the
        # flush ((FAILURE, UNROUTED) → rows_failed), so the honest status is
        # FAILED — the point is that resume COMPLETES instead of raising
        # AuditIntegrityError on every attempt.
        assert result.status == RunStatus.FAILED
        assert result.rows_failed == 3
        # The failed flush produced no output; the sink stays empty.
        assert output_sink.results == []
        # No BLOCKED rows survive the resume — every work item is terminal.
        with db.connection() as conn:
            work_statuses = (
                conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == run_id)).scalars().all()
            )
        assert work_statuses
        assert set(work_statuses) <= {"terminal"}, f"expected all-terminal journal, got {set(work_statuses)!r}"


# =============================================================================
# Batch-machinery recovery (synthetic, journal-era signatures)
# =============================================================================


class TestAggregationRecoveryIntegration:
    """Crash-recovery batch machinery: find, fail, retry, preserve order."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up complete test environment."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        factory = make_factory(db)

        return {
            "db": db,
            "checkpoint_manager": checkpoint_mgr,
            "recovery_manager": recovery_mgr,
            "factory": factory,
        }

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a minimal mock graph for aggregation recovery tests."""
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        agg_config = {
            "trigger": {"count": 1},
            "output_mode": "transform",
            "options": {"schema": {"mode": "observed"}},
            "schema": {"mode": "observed"},
        }
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test", config=schema_config)
        graph.add_node("sum_aggregator", node_type=NodeType.AGGREGATION, plugin_name="test", config=agg_config)
        graph.add_node("count_aggregator", node_type=NodeType.AGGREGATION, plugin_name="count_agg", config=agg_config)
        return graph

    def test_full_recovery_cycle(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Simulate crash during flush and verify recovery works.

        Journal era: the checkpoint row carries NO aggregation blob — buffered
        token payloads live in journal BLOCKED rows and batch membership in
        the audit tables. This test pins the batch-machinery half of recovery:
        the interrupted EXECUTING batch is found, failed, and retried with its
        members intact, and the run is resumable from the (blob-free)
        checkpoint row.
        """
        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        factory: RecorderFactory = test_env["factory"]

        # === PHASE 1: Normal execution until crash ===

        run = factory.run_lifecycle.begin_run(
            config={"aggregation": {"trigger": {"count": 3}}},
            canonical_version="sha256-rfc8785-v1",
        )

        # Register nodes using raw SQL to avoid schema_config requirement
        self._register_nodes_raw(db, run.run_id)

        # Record source rows and create tokens
        tokens = []
        for i in range(3):
            row = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"id": i, "value": i * 100},
                source_row_index=i,
                ingest_sequence=i,
            )
            token = factory.data_flow.create_token(row_id=row.row_id)
            tokens.append(token)

        # Create batch and add members
        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        for i, token in enumerate(tokens):
            factory.execution.add_batch_member(batch.batch_id, token.token_id, ordinal=i)

        # Checkpoint before flush — journal era: scalar-only checkpoint row
        # (buffered payloads live in journal BLOCKED rows, not in a blob).
        create_checkpoint(
            checkpoint_mgr,
            run_id=run.run_id,
            sequence_number=2,
            barrier_scalars=None,
            graph=mock_graph,
        )

        # Simulate crash during flush
        factory.execution.update_batch_status(batch.batch_id, BatchStatus.EXECUTING)
        factory.run_lifecycle.complete_run(run.run_id, status=RunStatus.FAILED)

        # === PHASE 2: Verify recovery is possible ===

        check = recovery_mgr.can_resume(run.run_id, mock_graph)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        resume_point = recovery_mgr.get_resume_point(run.run_id, mock_graph)
        assert resume_point is not None
        assert resume_point.checkpoint.sequence_number == 2

        # === PHASE 3: Execute recovery steps ===

        # Find incomplete batches
        incomplete = factory.execution.get_incomplete_batches(run.run_id)
        assert len(incomplete) == 1
        assert incomplete[0].batch_id == batch.batch_id
        assert incomplete[0].status == BatchStatus.EXECUTING

        # Mark executing as failed (crash interrupted)
        factory.execution.complete_batch(batch.batch_id, BatchStatus.FAILED)

        # Retry the batch
        retry_batch = factory.execution.retry_batch(batch.batch_id)
        assert retry_batch.attempt == 1
        assert retry_batch.status == BatchStatus.DRAFT

        # Verify members were copied
        retry_members = factory.execution.get_batch_members(retry_batch.batch_id)
        assert len(retry_members) == 3

        # === PHASE 4: Verify final state ===

        # Original batch is failed
        original_batch = factory.execution.get_batch(batch.batch_id)
        assert original_batch is not None
        assert original_batch.status == BatchStatus.FAILED

        # Retry batch exists
        all_batches = factory.execution.get_batches(run.run_id, node_id="sum_aggregator")
        assert len(all_batches) == 2  # Original + retry

        # Verify attempt progression
        attempts = sorted([b.attempt for b in all_batches])
        assert attempts == [0, 1]

    def test_recovery_with_multiple_aggregations(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Verify recovery handles multiple aggregation nodes independently."""
        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        recovery_mgr = test_env["recovery_manager"]
        factory: RecorderFactory = test_env["factory"]

        run = factory.run_lifecycle.begin_run(
            config={"aggregations": ["sum", "count"]},
            canonical_version="sha256-rfc8785-v1",
        )

        # Register multiple aggregation nodes
        self._register_nodes_raw(
            db,
            run.run_id,
            extra_nodes=[
                ("count_aggregator", "count_agg", NodeType.AGGREGATION),
            ],
        )

        # Create rows and tokens
        tokens = []
        for i in range(4):
            row = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"id": i, "value": i * 10},
                source_row_index=i,
                ingest_sequence=i,
            )
            token = factory.data_flow.create_token(row_id=row.row_id)
            tokens.append(token)

        # Create batch for sum_aggregator (completed successfully)
        sum_batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        for i, token in enumerate(tokens[:2]):
            factory.execution.add_batch_member(sum_batch.batch_id, token.token_id, ordinal=i)
        factory.execution.complete_batch(sum_batch.batch_id, BatchStatus.COMPLETED)

        # Create batch for count_aggregator (crashed during execution)
        count_batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="count_aggregator",
        )
        for i, token in enumerate(tokens[2:]):
            factory.execution.add_batch_member(count_batch.batch_id, token.token_id, ordinal=i)
        factory.execution.update_batch_status(count_batch.batch_id, BatchStatus.EXECUTING)

        # Checkpoint at last processed token — journal era: scalar-only row.
        create_checkpoint(
            checkpoint_mgr,
            run_id=run.run_id,
            sequence_number=3,
            barrier_scalars=None,
            graph=mock_graph,
        )

        factory.run_lifecycle.complete_run(run.run_id, status=RunStatus.FAILED)

        # Verify recovery
        check = recovery_mgr.can_resume(run.run_id, mock_graph)
        assert check.can_resume is True

        # Only count_aggregator batch should be incomplete
        incomplete = factory.execution.get_incomplete_batches(run.run_id)
        assert len(incomplete) == 1
        assert incomplete[0].aggregation_node_id == "count_aggregator"
        assert incomplete[0].status == BatchStatus.EXECUTING

    def test_recovery_preserves_batch_member_order(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Verify batch member ordinals are preserved through retry."""
        db = test_env["db"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        factory: RecorderFactory = test_env["factory"]

        run = factory.run_lifecycle.begin_run(
            config={"test": "order_preservation"},
            canonical_version="sha256-rfc8785-v1",
        )

        self._register_nodes_raw(db, run.run_id)

        # Create 5 rows with specific order
        tokens = []
        for i in range(5):
            row = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"seq": i, "data": f"item_{i}"},
                source_row_index=i,
                ingest_sequence=i,
            )
            token = factory.data_flow.create_token(row_id=row.row_id)
            tokens.append(token)

        # Create batch with specific member ordering
        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        # Add in reverse order to test ordinal preservation
        for i, token in enumerate(reversed(tokens)):
            factory.execution.add_batch_member(batch.batch_id, token.token_id, ordinal=i)

        # Mark as failed for retry
        factory.execution.complete_batch(batch.batch_id, BatchStatus.FAILED)

        # Checkpoint (journal era: scalar-only row)
        create_checkpoint(
            checkpoint_mgr,
            run_id=run.run_id,
            sequence_number=4,
            barrier_scalars=None,
            graph=mock_graph,
        )

        # Retry
        retry_batch = factory.execution.retry_batch(batch.batch_id)

        # Verify member order is preserved
        original_members = factory.execution.get_batch_members(batch.batch_id)
        retry_members = factory.execution.get_batch_members(retry_batch.batch_id)

        assert len(retry_members) == len(original_members)
        for orig, retry in zip(original_members, retry_members, strict=False):
            assert orig.token_id == retry.token_id
            assert orig.ordinal == retry.ordinal

    def test_recovery_cannot_retry_non_failed_batch(self, test_env: dict[str, Any]) -> None:
        """Verify retry_batch only works on failed batches."""
        db = test_env["db"]
        factory: RecorderFactory = test_env["factory"]

        run = factory.run_lifecycle.begin_run(
            config={"test": "retry_validation"},
            canonical_version="sha256-rfc8785-v1",
        )

        self._register_nodes_raw(db, run.run_id)

        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id="source",
            row_index=0,
            data={"id": 0},
            source_row_index=0,
            ingest_sequence=0,
        )
        token = factory.data_flow.create_token(row_id=row.row_id)

        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        factory.execution.add_batch_member(batch.batch_id, token.token_id, ordinal=0)

        # Test with draft status
        with pytest.raises(AuditIntegrityError, match="can only retry failed batches"):
            factory.execution.retry_batch(batch.batch_id)

        # Test with executing status
        factory.execution.update_batch_status(batch.batch_id, BatchStatus.EXECUTING)
        with pytest.raises(AuditIntegrityError, match="can only retry failed batches"):
            factory.execution.retry_batch(batch.batch_id)

        # Test with completed status
        factory.execution.complete_batch(batch.batch_id, BatchStatus.COMPLETED)
        with pytest.raises(AuditIntegrityError, match="can only retry failed batches"):
            factory.execution.retry_batch(batch.batch_id)

    def _register_nodes_raw(
        self,
        db: LandscapeDB,
        run_id: str,
        *,
        extra_nodes: list[tuple[str, str, NodeType]] | None = None,
    ) -> None:
        """Register nodes using raw SQL to avoid schema_config requirement.

        Args:
            db: LandscapeDB instance
            run_id: Run to register nodes for
            extra_nodes: Optional list of (node_id, plugin_name, node_type) tuples
        """
        from elspeth.contracts.contract_records import ContractAuditRecord
        from elspeth.core.landscape.schema import nodes_table, run_sources_table

        now = datetime.now(UTC)

        with db.engine.connect() as conn:
            # Source node
            conn.execute(
                nodes_table.insert().values(
                    node_id="source",
                    run_id=run_id,
                    plugin_name="test_source",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # ADR-025 §3 Decision 5: verify_contract_integrity reads from
            # run_sources, so each test source must carry its contract there.
            contract = _create_test_schema_contract()
            audit_record = ContractAuditRecord.from_contract(contract)
            conn.execute(
                run_sources_table.insert().values(
                    run_id=run_id,
                    source_node_id="source",
                    source_name="source",
                    plugin_name="test_source",
                    lifecycle_state="loaded",
                    config_hash="test",
                    schema_json="{}",
                    schema_contract_json=audit_record.to_json(),
                    schema_contract_hash=contract.version_hash(),
                    field_resolution_json=None,
                    recorded_at=now,
                )
            )

            # Sum aggregator node
            conn.execute(
                nodes_table.insert().values(
                    node_id="sum_aggregator",
                    run_id=run_id,
                    plugin_name="sum_agg",
                    node_type=NodeType.AGGREGATION,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Extra nodes
            if extra_nodes:
                for node_id, plugin_name, node_type in extra_nodes:
                    conn.execute(
                        nodes_table.insert().values(
                            node_id=node_id,
                            run_id=run_id,
                            plugin_name=plugin_name,
                            node_type=node_type,
                            plugin_version="1.0",
                            determinism=Determinism.DETERMINISTIC,
                            config_hash="test",
                            config_json="{}",
                            registered_at=now,
                        )
                    )

            conn.commit()

    def test_timeout_preservation_on_resume(self, test_env: dict[str, Any]) -> None:
        """Verify aggregation timeout window doesn't reset on resume (Bug #6).

        Journal era: the restore derives batch age from the absolute
        ``barrier_blocked_at`` stamp of the OLDEST BLOCKED journal row —
        ``restore_from_journal(now=...)`` computes ``now - min(blocked_at)``.

        Scenario (SLA preserved across resume):
        1. 60s-timeout aggregation; 3 rows blocked at the barrier, the oldest
           30 seconds before the crash.
        2. Crash and resume: journal restore rebuilds the buffer.
        3. The restored trigger thinks ~30s have already elapsed — it must NOT
           fire immediately, and it must fire after 30 MORE seconds (60s
           total), not after a fresh 60s window.
        """
        import time

        from elspeth.contracts.barrier_scalars import AggregationNodeScalars
        from elspeth.core.config import TriggerConfig
        from elspeth.engine.executors import AggregationExecutor
        from elspeth.engine.spans import SpanFactory

        db = test_env["db"]
        factory: RecorderFactory = test_env["factory"]

        # === PHASE 1: Original run with timeout trigger ===

        run = factory.run_lifecycle.begin_run(
            config={"aggregation": {"trigger": {"timeout_seconds": 60}}},
            canonical_version="sha256-rfc8785-v1",
            leader_worker_id="seeder",
        )

        self._register_nodes_raw(db, run.run_id)

        now = datetime.now(UTC)
        first_blocked_at = now - timedelta(seconds=30.0)

        # 3 rows accepted into one in-progress batch; their tokens are
        # BLOCKED at the barrier through the production scheduler verbs
        # (mark_blocked stamps barrier_blocked_at — the restore's age source).
        tokens = []
        batch = factory.execution.create_batch(
            run_id=run.run_id,
            aggregation_node_id="sum_aggregator",
        )
        for i in range(3):
            row_obj = factory.data_flow.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"id": i, "value": i * 100},
                source_row_index=i,
                ingest_sequence=i,
            )
            token = factory.data_flow.create_token(row_id=row_obj.row_id)
            tokens.append(token)
            factory.execution.add_batch_member(batch.batch_id, token.token_id, ordinal=i)
            payload = PipelineRow({"id": i, "value": i * 100}, _create_test_schema_contract())
            factory.scheduler.enqueue_ready(
                run_id=run.run_id,
                token_id=token.token_id,
                row_id=token.row_id,
                node_id="sum_aggregator",
                step_index=1,
                ingest_sequence=i,
                row_payload_json=factory.scheduler.serialize_row_payload(payload),
                available_at=now,
            )
            claimed = factory.scheduler.claim_ready(run_id=run.run_id, lease_owner="seeder", lease_seconds=60, now=now)
            assert claimed is not None and claimed.token_id == token.token_id
            blocked_at = first_blocked_at + timedelta(seconds=i)  # oldest row anchors the age
            factory.scheduler.mark_blocked(
                work_item_id=claimed.work_item_id,
                queue_key=None,
                barrier_key="sum_aggregator",
                now=blocked_at,
                expected_lease_owner="seeder",
            )
        factory.run_lifecycle.complete_run(run.run_id, status=RunStatus.FAILED)

        # === PHASE 2: Resume and verify timeout preservation ===

        trigger_config = TriggerConfig(timeout_seconds=60.0)
        span_factory = SpanFactory()  # No tracer = no-op spans

        agg_settings = {
            NodeID("sum_aggregator"): AggregationSettings(
                name="sum_aggregator",
                plugin="test_aggregation",
                input="source_out",
                on_error="discard",
                trigger=trigger_config,
                output_mode="transform",
                options={},
            )
        }

        executor = AggregationExecutor(
            execution=factory.execution,
            span_factory=span_factory,
            step_resolver=lambda node_id: 1,
            run_id=run.run_id,
            aggregation_settings=agg_settings,
        )

        items = factory.scheduler.list_blocked_barrier_items(run_id=run.run_id)
        assert len(items) == 3
        executor.restore_from_journal(
            node_id=NodeID("sum_aggregator"),
            items=items,
            member_order=[t.token_id for t in tokens],
            batch_id=batch.batch_id,
            accepted_count_total=3,
            completed_flush_count=0,
            scalars=AggregationNodeScalars(None, None),
            attempt_offsets={t.token_id: 0 for t in tokens},
            resume_checkpoint_id="ckpt-resume-1",
            now=now,
        )

        # Verify buffer was restored
        assert executor.get_buffer_count(NodeID("sum_aggregator")) == 3

        # Get the restored evaluator
        node_state = executor._nodes.get(NodeID("sum_aggregator"))
        assert node_state is not None
        restored_evaluator = node_state.trigger

        # Verify batch count was restored
        assert restored_evaluator.batch_count == 3

        # Verify timeout age was restored from barrier_blocked_at (Bug #6 fix):
        # the restored evaluator should think ~30s have already elapsed.
        restored_age = restored_evaluator.batch_age_seconds
        assert 29.0 <= restored_age <= 31.0, f"Expected ~30s, got {restored_age}s"

        # Should NOT trigger immediately (need 30 more seconds)
        assert restored_evaluator.should_trigger() is False

        # Simulate 30 more seconds passing (total 60s)
        restored_evaluator._first_accept_time = time.monotonic() - 60.0

        # NOW it should trigger (60s total elapsed)
        assert restored_evaluator.should_trigger() is True
        assert restored_evaluator.which_triggered() == "timeout"

        # Verify it triggers at ~60s, not at ~90s (which would be 30s stored + 60s new timeout)
        final_age = restored_evaluator.batch_age_seconds
        assert 59.0 <= final_age <= 61.0, f"Timeout should trigger at ~60s, got {final_age}s"
