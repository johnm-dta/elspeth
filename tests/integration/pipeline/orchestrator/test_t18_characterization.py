# tests/integration/pipeline/orchestrator/test_t18_characterization.py
"""Characterization tests for T18 orchestrator decomposition.

These tests exercise the full _execute_run() and _process_resumed_rows() paths
with multi-feature pipelines. They serve as regression oracles for the 15-commit
extraction sequence — if any extraction breaks behavior, these tests catch it.

IMPORTANT: Do NOT modify these tests during the extraction. If a test fails
after an extraction commit, the extraction introduced a regression.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

import pytest
from sqlalchemy import text

from elspeth.contracts import (
    Determinism,
    PipelineRow,
    ResumedRow,
    RunStatus,
    SourceProtocol,
    TransformProtocol,
)
from elspeth.contracts.results import SourceRow
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.types import NodeID
from elspeth.core.config import AggregationSettings, SourceSettings, TriggerConfig
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.engine.orchestrator.core import _RunFailedWithPartialResultError
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.testing import make_pipeline_row, make_source_row
from tests.fixtures.base_classes import (
    _TestSchema,
    _TestSourceBase,
    _TestTransformBase,
    as_sink,
    as_source,
    as_transform,
)
from tests.fixtures.factories import wire_transforms
from tests.fixtures.landscape import make_landscape_db
from tests.fixtures.pipeline import build_production_graph
from tests.fixtures.plugins import CollectSink, ListSource
from tests.fixtures.stores import MockPayloadStore

# ---------------------------------------------------------------------------
# Test fixtures: Quarantine-capable source
# ---------------------------------------------------------------------------


class QuarantiningSource(_TestSourceBase):
    """Source that quarantines rows based on a 'valid' field.

    Rows with valid=False are quarantined. This exercises:
    - Field resolution ordering (must be from first VALID row)
    - Quarantine routing (direct to configured sink)
    - Schema contract recording (must skip quarantined rows)
    """

    name = "quarantining_source"
    output_schema = _TestSchema

    def __init__(self, rows: list[dict[str, Any]], quarantine_sink: str = "errors") -> None:
        super().__init__()
        self._rows = rows
        self._on_validation_failure = quarantine_sink

    def load(self, ctx: Any) -> Any:
        for source_row_index, row in enumerate(self._rows):
            if not row.get("valid", True):
                yield SourceRow.quarantined(
                    row=row,
                    error="validation_failed:valid=False",
                    destination=self._on_validation_failure,
                    source_row_index=source_row_index,
                )
            else:
                yield make_source_row(row, source_row_index=source_row_index)

    def get_field_resolution(self) -> tuple[Mapping[str, str], str] | None:
        return ({"value": "value", "valid": "valid"}, "identity")


class DoubleValueTransform(_TestTransformBase):
    """Transform that doubles the 'value' field."""

    name = "double_value"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        data = row.to_dict()
        data["doubled"] = data.get("value", 0) * 2
        return TransformResult.success(
            make_pipeline_row(data),
            success_reason={"action": "doubled"},
        )


# ---------------------------------------------------------------------------
# Helper: set up factory + run_id for direct _execute_run() calls
# ---------------------------------------------------------------------------


def _begin_test_run(db: LandscapeDB) -> tuple[RecorderFactory, str, MockPayloadStore]:
    """Create a factory and begin a run for testing.

    Returns (factory, run_id, payload_store).
    """
    payload_store = MockPayloadStore()
    factory = RecorderFactory(db, payload_store=payload_store)
    run = factory.run_lifecycle.begin_run(
        config={},
        canonical_version="sha256-rfc8785-v1",
    )
    return factory, run.run_id, payload_store


# ---------------------------------------------------------------------------
# Characterization test: Full _execute_run() path
# ---------------------------------------------------------------------------


class TestT18CharacterizationExecuteRun:
    """Regression oracle for the T18 extraction sequence.

    Pipeline: QuarantiningSource → DoubleValueTransform → CollectSink("output") + CollectSink("errors")
    Input: 5 rows — first 2 quarantined, next 3 valid.

    This exercises:
    - Quarantine routing (rows 0-1 → errors sink)
    - Field resolution from first VALID row (row 2, not row 0)
    - Transform processing (rows 2-4 → output sink)
    - Counter arithmetic with quarantine + success
    - operation_id attribution (transforms see None)
    - Landscape audit records (nodes, node_states, routing_events)
    """

    def _build_pipeline(
        self,
    ) -> tuple[
        SourceProtocol,
        TransformProtocol,
        Any,
        Any,
        PipelineConfig,
    ]:
        rows = [
            {"value": 10, "valid": False},  # quarantined (row 0)
            {"value": 20, "valid": False},  # quarantined (row 1)
            {"value": 30, "valid": True},  # valid (row 2) — first valid row
            {"value": 40, "valid": True},  # valid (row 3)
            {"value": 50, "valid": True},  # valid (row 4)
        ]
        output_collect = CollectSink("output")
        error_collect = CollectSink("errors")
        source = as_source(QuarantiningSource(rows, quarantine_sink="errors"))
        transform = as_transform(DoubleValueTransform())
        output_sink = as_sink(output_collect)
        error_sink = as_sink(error_collect)

        config = PipelineConfig(
            sources={"primary": source},
            transforms=[transform],
            sinks={"output": output_sink, "errors": error_sink},
        )
        return source, transform, output_collect, error_collect, config

    def test_counter_values_exact(self) -> None:
        """Assert exact counter values for the characterization pipeline.

        Note: _execute_run() returns RUNNING status (the public run() wrapper
        sets COMPLETED after finalize_run()). This is the current behavior.
        """
        _source, _transform, _output_sink, _error_sink, config = self._build_pipeline()
        db = make_landscape_db()
        orchestrator = Orchestrator(db)

        graph = build_production_graph(config)
        factory, run_id, payload_store = _begin_test_run(db)

        result = orchestrator._execute_run(
            factory=factory,
            run_id=run_id,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        # _execute_run returns RUNNING — run() wrapper sets COMPLETED
        assert result.status == RunStatus.RUNNING
        assert result.rows_processed == 5
        assert result.rows_quarantined == 2
        assert result.rows_succeeded == 3
        assert result.rows_failed == 2
        assert result.rows_routed_success == 0
        assert result.rows_routed_failure == 0
        assert result.rows_forked == 0

    def test_sink_contents(self) -> None:
        """Assert sink contents match expected routing."""
        _source, _transform, output_sink, error_sink, config = self._build_pipeline()
        db = make_landscape_db()
        orchestrator = Orchestrator(db)

        graph = build_production_graph(config)
        factory, run_id, payload_store = _begin_test_run(db)

        orchestrator._execute_run(
            factory=factory,
            run_id=run_id,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        # Output sink gets 3 valid rows (doubled)
        assert len(output_sink.results) == 3
        # Error sink gets 2 quarantined rows
        assert len(error_sink.results) == 2

    def test_operation_id_not_leaked_to_transforms(self) -> None:
        """Assert transforms never see a non-None operation_id.

        Uses patch.object spy pattern from the design doc. The source_load
        operation sets operation_id, but it must be cleared before transforms
        execute. If extraction breaks the operation_id lifecycle, this catches it.
        """
        _source, transform, _output_sink, _error_sink, config = self._build_pipeline()
        db = make_landscape_db()
        orchestrator = Orchestrator(db)

        graph = build_production_graph(config)
        factory, run_id, payload_store = _begin_test_run(db)

        captured_operation_ids: list[str | None] = []
        original_process = transform.process

        def spy_process(row: PipelineRow, ctx: Any) -> TransformResult:
            captured_operation_ids.append(ctx.operation_id)
            return original_process(row, ctx)

        with patch.object(transform, "process", side_effect=spy_process):
            orchestrator._execute_run(
                factory=factory,
                run_id=run_id,
                config=config,
                graph=graph,
                payload_store=payload_store,
            )

        # All 3 valid rows should have been processed
        assert len(captured_operation_ids) == 3
        # None of them should have seen a non-None operation_id
        assert all(op_id is None for op_id in captured_operation_ids), (
            f"operation_id leaked into transform execution: {captured_operation_ids}"
        )

    def test_field_resolution_recorded_despite_first_quarantine(self) -> None:
        """Assert field resolution is recorded even when first row is quarantined.

        Field resolution must come from the first VALID row, not be skipped
        when the first row is quarantined. Stored as source_field_resolution_json
        on the runs table.
        """
        _source, _transform, _output_sink, _error_sink, config = self._build_pipeline()
        db = make_landscape_db()
        orchestrator = Orchestrator(db)

        graph = build_production_graph(config)
        factory, run_id, payload_store = _begin_test_run(db)

        orchestrator._execute_run(
            factory=factory,
            run_id=run_id,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        # Check that field resolution was recorded on the runs table
        assert db._engine is not None
        with db._engine.connect() as conn:
            result = conn.execute(
                text("SELECT source_field_resolution_json FROM runs WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            row = result.fetchone()
        assert row is not None, "Run not found in DB"
        assert row[0] is not None, "source_field_resolution_json should be set (first valid row provides field resolution)"

    def test_audit_record_counts(self) -> None:
        """Assert Landscape audit records are complete.

        After the run:
        - nodes table should have entries for source, transform, output sink, errors sink
        - routing_events should have DIVERT entries for quarantined rows
        """
        _source, _transform, _output_sink, _error_sink, config = self._build_pipeline()
        db = make_landscape_db()
        orchestrator = Orchestrator(db)

        graph = build_production_graph(config)
        factory, run_id, payload_store = _begin_test_run(db)

        orchestrator._execute_run(
            factory=factory,
            run_id=run_id,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        assert db._engine is not None
        with db._engine.connect() as conn:
            # Nodes registered
            node_count = conn.execute(
                text("SELECT COUNT(*) FROM nodes WHERE run_id = :run_id"),
                {"run_id": run_id},
            ).scalar()
            assert node_count is not None
            # At minimum: source + transform + 2 sinks = 4
            assert node_count >= 4, f"Expected >= 4 nodes, got {node_count}"

            # Routing events for quarantined rows (DIVERT mode)
            divert_count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM routing_events re "
                    "JOIN node_states ns ON re.state_id = ns.state_id "
                    "WHERE ns.run_id = :run_id AND re.mode = :mode"
                ),
                {"run_id": run_id, "mode": "divert"},
            ).scalar()
            assert divert_count == 2, f"Expected 2 DIVERT routing events, got {divert_count}"

    def test_current_graph_not_cleared_on_error(self) -> None:
        """Assert _current_graph is NOT cleared when _execute_run() raises.

        Currently, _current_graph = None is OUTSIDE the finally block in
        _execute_run(), so errors leave it set. This characterizes the current
        behavior — if a future extraction moves cleanup into the finally block,
        this test should be updated to expect None.
        """
        db = make_landscape_db()
        orchestrator = Orchestrator(db)

        class ErrorTransform(_TestTransformBase):
            name = "error_transform"
            determinism = Determinism.DETERMINISTIC
            input_schema = _TestSchema
            output_schema = _TestSchema

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                raise RuntimeError("deliberate error for characterization test")

        source = as_source(ListSource([{"value": 1}]))
        transform = as_transform(ErrorTransform())
        sink = as_sink(CollectSink("output"))

        config = PipelineConfig(
            sources={"primary": source},
            transforms=[transform],
            sinks={"output": sink},
        )

        graph = build_production_graph(config)
        factory, run_id, payload_store = _begin_test_run(db)

        with pytest.raises(_RunFailedWithPartialResultError, match="deliberate error") as exc_info:
            orchestrator._execute_run(
                factory=factory,
                run_id=run_id,
                config=config,
                graph=graph,
                payload_store=payload_store,
            )
        assert isinstance(exc_info.value.original_error, RuntimeError)
        assert str(exc_info.value.original_error) == "deliberate error for characterization test"

        # Current behavior: _current_graph is NOT cleared on error
        # (the assignment is after the finally block, not inside it)
        assert orchestrator._current_graph is not None


# ---------------------------------------------------------------------------
# Characterization test: Resume path
# ---------------------------------------------------------------------------


class TestT18CharacterizationResumePath:
    """Resume-specific characterization tests.

    These verify the behavioral divergences between _execute_run() and
    _process_resumed_rows() documented in the design.
    """

    def test_schema_contract_set_before_transform_execution(self) -> None:
        """Assert ctx.contract is set to the resume schema_contract BEFORE transforms execute.

        The resume path sets run_ctx.ctx.contract = schema_contract between
        _initialize_run_context() and LoopContext construction. If a future
        refactor reorders these steps, transforms would see contract=None.

        Strategy: Run the original pipeline to populate DB with rows, then
        resume with one of those rows and spy on the transform to capture
        ctx.contract during process(). Verify it matches the passed contract.
        """
        db = make_landscape_db()
        orchestrator = Orchestrator(db)

        captured_contracts: list[SchemaContract | None] = []

        class ContractCapturingTransform(_TestTransformBase):
            name = "contract_capture_transform"
            determinism = Determinism.DETERMINISTIC
            input_schema = _TestSchema
            output_schema = _TestSchema

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                captured_contracts.append(ctx.contract)
                return TransformResult.success(
                    make_pipeline_row(row.to_dict()),
                    success_reason={"action": "identity"},
                )

        source = as_source(ListSource([{"value": 1}]))
        transform = as_transform(ContractCapturingTransform())
        output_sink = as_sink(CollectSink("output"))

        config = PipelineConfig(
            sources={"primary": source},
            transforms=[transform],
            sinks={"output": output_sink},
        )

        graph = build_production_graph(config)
        factory, run_id, payload_store = _begin_test_run(db)

        # First: run the full pipeline to populate DB with rows, nodes, edges
        orchestrator._execute_run(
            factory=factory,
            run_id=run_id,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        # Retrieve the actual row_id and source_node_id created during the original run.
        # ADR-025 §3 + §4: resume rows are ResumedRow instances carrying
        # source_node_id; the per-source contract map is keyed by that identity.
        assert db._engine is not None
        with db._engine.connect() as conn:
            row_record = conn.execute(
                text("SELECT row_id, source_node_id FROM rows WHERE run_id = :run_id LIMIT 1"),
                {"run_id": run_id},
            ).first()
        assert row_record is not None, "Original run should have created at least one row"
        row_id, source_node_id_str = row_record
        source_node_id = NodeID(source_node_id_str)

        # Clear captures from original run
        captured_contracts.clear()

        # Create a distinct schema contract for resume (different from what
        # the original run would have set, so we can verify identity)
        resume_contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(
                    normalized_name="value",
                    original_name="value",
                    python_type=int,
                    required=False,
                    source="inferred",
                ),
            ),
            locked=True,
        )

        # Resume with the real row_id from the original run
        orchestrator._process_resumed_rows(
            factory=factory,
            run_id=run_id,
            config=config,
            graph=graph,
            unprocessed_rows=[
                ResumedRow(
                    row_id=row_id,
                    row_index=0,
                    source_node_id=source_node_id,
                    row_data={"value": 1},
                ),
            ],
            restored_aggregation_state={},
            restored_coalesce_state=None,
            payload_store=payload_store,
            schema_contracts_by_source={source_node_id: resume_contract},
        )

        # The transform must have seen the contract during process()
        assert len(captured_contracts) == 1, f"Expected 1 transform call, got {len(captured_contracts)}"
        assert captured_contracts[0] is resume_contract, (
            f"ctx.contract during resume must be the passed schema_contract, got {captured_contracts[0]!r}"
        )

    def test_source_on_start_not_called_during_resume(self) -> None:
        """Assert source.on_start() is NOT called during resume.

        The resume path uses include_source_on_start=False because the source
        was fully consumed in the original run. Transform/sink on_start MUST
        still fire.

        Strategy: First do a full _execute_run() to populate DB with nodes/edges,
        then call _process_resumed_rows() with empty rows on the same run.
        """
        db = make_landscape_db()
        orchestrator = Orchestrator(db)

        on_start_calls: dict[str, int] = {"source": 0, "transform": 0, "sink": 0}

        class TrackingSource(_TestSourceBase):
            name = "tracking_source"
            output_schema = _TestSchema

            def on_start(self, ctx: Any) -> None:
                on_start_calls["source"] += 1

            def load(self, ctx: Any) -> Any:
                # Yield one row so the original run has data to process
                yield make_source_row({"value": 1})

        class TrackingTransform(_TestTransformBase):
            name = "tracking_transform"
            determinism = Determinism.DETERMINISTIC
            input_schema = _TestSchema
            output_schema = _TestSchema

            def on_start(self, ctx: Any) -> None:
                super().on_start(ctx)
                on_start_calls["transform"] += 1

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                return TransformResult.success(
                    make_pipeline_row(row.to_dict()),
                    success_reason={"action": "identity"},
                )

        source = as_source(TrackingSource())
        transform = as_transform(TrackingTransform())
        output_sink = as_sink(CollectSink("output"))

        config = PipelineConfig(
            sources={"primary": source},
            transforms=[transform],
            sinks={"output": output_sink},
        )

        graph = build_production_graph(config)
        factory, run_id, payload_store = _begin_test_run(db)

        # First: run the full pipeline to populate DB with nodes, edges, etc.
        orchestrator._execute_run(
            factory=factory,
            run_id=run_id,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        # Reset counters — we only care about resume behavior
        on_start_calls["source"] = 0
        on_start_calls["transform"] = 0
        on_start_calls["sink"] = 0

        # Create a minimal schema contract for resume
        schema_contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(
                    normalized_name="value",
                    original_name="value",
                    python_type=int,
                    required=False,
                    source="inferred",
                ),
            ),
            locked=True,
        )

        # Spy on sink on_start using patch.object
        original_sink_on_start = output_sink.on_start

        def tracking_sink_on_start(ctx: Any) -> None:
            on_start_calls["sink"] += 1
            original_sink_on_start(ctx)

        # Call _process_resumed_rows directly with empty rows.
        # ADR-025 §3: schema_contracts_by_source is non-empty by contract;
        # derive the source NodeID from the graph rather than fabricating one,
        # so the test exercises the production reconstruction path identity.
        sources_list = list(graph.get_sources())
        assert len(sources_list) == 1, "single-source pipeline expected for this test"
        source_node_id = sources_list[0]
        with patch.object(output_sink, "on_start", side_effect=tracking_sink_on_start):
            result = orchestrator._process_resumed_rows(
                factory=factory,
                run_id=run_id,
                config=config,
                graph=graph,
                unprocessed_rows=[],
                restored_aggregation_state={},
                restored_coalesce_state=None,
                payload_store=payload_store,
                schema_contracts_by_source={source_node_id: schema_contract},
            )

        # _process_resumed_rows also returns RUNNING (same as _execute_run)
        assert result.status == RunStatus.RUNNING
        assert on_start_calls["source"] == 0, "Source on_start should NOT be called during resume"
        assert on_start_calls["transform"] == 1, "Transform on_start should be called during resume"
        assert on_start_calls["sink"] == 1, "Sink on_start should be called during resume"


# ---------------------------------------------------------------------------
# Test fixtures: Batch-aware transform for aggregation characterization
# ---------------------------------------------------------------------------


class SumBatchTransform(BaseTransform):
    """Batch-aware transform that sums the 'value' field across buffered rows.

    When flushed (receives list[PipelineRow]), produces one aggregated row.
    This exercises the output_mode="transform" deaggregation path where
    N input tokens → CONSUMED_IN_BATCH and 1 output token is expanded.
    """

    name = "sum_batch"
    determinism = Determinism.DETERMINISTIC
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True
    on_success: str | None = "output"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow | list[PipelineRow], ctx: Any) -> TransformResult:
        if isinstance(row, list):
            total = sum(r.get("value", 0) for r in row)
            output = {"value": total, "count": len(row)}
            contract = SchemaContract(
                mode="OBSERVED",
                fields=(
                    FieldContract(normalized_name="value", original_name="value", python_type=int, required=False, source="inferred"),
                    FieldContract(normalized_name="count", original_name="count", python_type=int, required=False, source="inferred"),
                ),
                locked=True,
            )
            return TransformResult.success(
                PipelineRow(output, contract),
                success_reason={"action": "batch_sum"},
            )
        return TransformResult.success(make_pipeline_row(row.to_dict()), success_reason={"action": "buffer"})


def _build_aggregation_pipeline() -> tuple[SourceProtocol, TransformProtocol, Any, PipelineConfig, ExecutionGraph]:
    """Build a pipeline with batch-aware aggregation for characterization.

    Pipeline: ListSource → SumBatchTransform → CollectSink("output")
    Trigger: count=100 (won't trigger mid-stream — only end-of-source flush)
    Input: 3 rows with values [10, 20, 30]
    Expected: 1 aggregated output row with value=60, count=3
    """
    output_collect = CollectSink("output")
    source = as_source(ListSource([{"value": 10}, {"value": 20}, {"value": 30}], name="agg_source", on_success="source_out"))
    transform = as_transform(SumBatchTransform())
    output_sink = as_sink(output_collect)

    # Build graph via production path
    graph = ExecutionGraph.from_plugin_instances(
        sources={"primary": source},
        source_settings_map={"primary": SourceSettings(plugin=source.name, on_success="source_out", options={})},
        transforms=wire_transforms([transform], source_connection="source_out", final_sink="output"),
        sinks={"output": output_sink},
        aggregations={},
        gates=[],
        coalesce_settings=None,
    )

    # Map the transform's node ID to aggregation settings
    transform_id_map = graph.get_transform_id_map()
    transform_node_id = transform_id_map[0]

    agg_settings = AggregationSettings(
        name="sum_agg",
        plugin="sum_batch",
        input="source_out",
        on_success="output",
        on_error="discard",
        trigger=TriggerConfig(count=100, timeout_seconds=3600),
        output_mode="transform",
    )

    config = PipelineConfig(
        sources={"primary": source},
        transforms=[transform],
        sinks={"output": output_sink},
        aggregation_settings={transform_node_id: agg_settings},
    )

    return source, transform, output_collect, config, graph


# ---------------------------------------------------------------------------
# Characterization test: Aggregation (deaggregation) path
# ---------------------------------------------------------------------------


class TestT18CharacterizationAggregation:
    """Regression oracle for aggregation/deaggregation through _execute_run().

    Exercises end-of-source aggregation flush: rows buffer during processing
    (count trigger is unreachable) and flush at source completion. This path
    exercises:
    - Aggregation buffer setup via aggregation_settings in PipelineConfig
    - End-of-source flush_remaining_aggregation_buffers()
    - CONSUMED_IN_BATCH terminal state for input tokens
    - Expanded output tokens for deaggregation
    """

    def test_aggregation_counter_values(self) -> None:
        """Assert exact counter values for end-of-source aggregation flush.

        3 input rows buffer during processing, flush at end-of-source produces
        1 aggregated row that reaches the output sink.
        """
        _source, _transform, output_sink, config, graph = _build_aggregation_pipeline()
        db = make_landscape_db()
        orchestrator = Orchestrator(db)
        factory, run_id, payload_store = _begin_test_run(db)

        result = orchestrator._execute_run(
            factory=factory,
            run_id=run_id,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        assert result.status == RunStatus.RUNNING
        assert result.rows_processed == 3
        assert result.rows_buffered == 3, "All 3 input rows buffered in aggregation"
        assert result.rows_succeeded == 1, "Aggregated output row counts as succeeded"
        assert result.rows_failed == 0
        assert len(output_sink.results) == 1, "One aggregated row in output"

    def test_aggregation_output_content(self) -> None:
        """Assert the aggregated output row has correct content."""
        _source, _transform, output_sink, config, graph = _build_aggregation_pipeline()
        db = make_landscape_db()
        orchestrator = Orchestrator(db)
        factory, run_id, payload_store = _begin_test_run(db)

        orchestrator._execute_run(
            factory=factory,
            run_id=run_id,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        # The single output row should have value=60 (10+20+30) and count=3
        output_row = output_sink.results[0]
        assert output_row["value"] == 60
        assert output_row["count"] == 3

    def test_aggregation_flush_inside_source_load_operation(self) -> None:
        """Assert aggregation flush happens INSIDE track_operation(source_load).

        The source_load operation must encompass the end-of-source flush.
        If the flush happened OUTSIDE the operation context, the aggregated
        output would not reach the sink (it would be lost). The fact that
        output_sink has results AND the source_load operation completed
        successfully proves the boundary is correct.

        Additionally, verify the operations table structure:
        - Exactly one source_load operation
        - At least one sink_write operation (for the aggregated output)
        """
        _source, _transform, output_sink, config, graph = _build_aggregation_pipeline()
        db = make_landscape_db()
        orchestrator = Orchestrator(db)
        factory, run_id, payload_store = _begin_test_run(db)

        orchestrator._execute_run(
            factory=factory,
            run_id=run_id,
            config=config,
            graph=graph,
            payload_store=payload_store,
        )

        # Aggregated output reached the sink (proves flush was inside the loop)
        assert len(output_sink.results) == 1, "Flush must produce output before sink writes"

        # Verify operation attribution in Landscape DB
        assert db._engine is not None
        with db._engine.connect() as conn:
            source_ops = conn.execute(
                text("SELECT COUNT(*) FROM operations WHERE run_id = :run_id AND operation_type = 'source_load'"),
                {"run_id": run_id},
            ).scalar()
            assert source_ops == 1, f"Expected exactly 1 source_load operation, got {source_ops}"

            sink_ops = conn.execute(
                text("SELECT COUNT(*) FROM operations WHERE run_id = :run_id AND operation_type = 'sink_write'"),
                {"run_id": run_id},
            ).scalar()
            assert sink_ops is not None
            assert sink_ops >= 1, f"Expected >= 1 sink_write operation, got {sink_ops}"
