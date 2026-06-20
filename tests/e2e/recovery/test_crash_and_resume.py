"""E2E tests for crash recovery and resume scenarios.

Tests verify that ELSPETH can recover from crashes and resume
processing, producing the same results as uninterrupted runs.

Uses file-based SQLite and real payload stores. No mocks except
external services.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import ConfigDict
from sqlalchemy import select

from elspeth.contracts import (
    ArtifactDescriptor,
    Determinism,
    NodeType,
    PipelineRow,
    PluginSchema,
    RoutingMode,
    RunStatus,
    SourceRow,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.barrier_scalars import AggregationNodeScalars, BarrierScalars
from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.contracts.contract_records import ContractAuditRecord
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.enums import TerminalOutcome, TerminalPath
from elspeth.contracts.errors import GracefulShutdownError, IncompleteSourceResumeError
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.types import NodeID, SinkName
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.config import CheckpointSettings, QueueSettings, SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    run_coordination_events_table,
    run_coordination_table,
    run_sources_table,
    run_workers_table,
    runs_table,
    token_outcomes_table,
    tokens_table,
    transform_errors_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from elspeth.plugins.sinks.json_sink import JSONSink
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import (
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
    as_transform,
)
from tests.fixtures.factories import wire_transforms
from tests.fixtures.landscape import make_factory

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _RowSchema(PluginSchema):
    """Schema for recovery test rows."""

    id: int
    value: int


def _create_test_schema_contract() -> tuple[str, str]:
    """Create a minimal schema contract for test runs.

    Returns:
        Tuple of (schema_contract_json, schema_contract_hash)
    """
    field_contracts = (
        FieldContract(
            normalized_name="id",
            original_name="id",
            python_type=int,
            required=True,
            source="declared",
        ),
        FieldContract(
            normalized_name="value",
            original_name="value",
            python_type=int,
            required=True,
            source="declared",
        ),
    )
    contract = SchemaContract(fields=field_contracts, mode="FIXED", locked=True)
    audit_record = ContractAuditRecord.from_contract(contract)
    return audit_record.to_json(), contract.version_hash()


def _build_linear_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build a simple linear graph for checkpoint tests.

    NOTE: Manual ExecutionGraph construction is required here because
    checkpoint tests need deterministic node IDs that match stored
    checkpoint records (which reference specific node IDs like
    "source", "transform_0", "sink_default"). Using
    from_plugin_instances() would generate hash-based IDs that don't
    match the checkpoint data, causing resume validation to fail.
    """
    graph = ExecutionGraph()

    schema_config = {"schema": {"mode": "observed"}}
    graph.add_node(
        "source",
        node_type=NodeType.SOURCE,
        plugin_name=config.sources["primary"].name,
        config={**schema_config, "source_name": "primary"},
    )

    transform_ids: dict[int, NodeID] = {}
    prev = "source"
    for i, t in enumerate(config.transforms):
        node_id = NodeID(f"transform_{i}")
        transform_ids[i] = node_id
        graph.add_node(
            node_id,
            node_type=NodeType.TRANSFORM,
            plugin_name=t.name,
            config=schema_config,
        )
        graph.add_edge(prev, node_id, label="continue", mode=RoutingMode.MOVE)
        prev = node_id

    sink_ids: dict[SinkName, NodeID] = {}
    for sink_name, sink in config.sinks.items():
        node_id = NodeID(f"sink_{sink_name}")
        sink_ids[SinkName(sink_name)] = node_id
        graph.add_node(
            node_id,
            node_type=NodeType.SINK,
            plugin_name=sink.name,
            config=schema_config,
        )

    if SinkName("default") in sink_ids:
        graph.add_edge(
            prev,
            sink_ids[SinkName("default")],
            label="continue",
            mode=RoutingMode.MOVE,
        )

    graph.set_sink_id_map(sink_ids)
    graph.set_transform_id_map(transform_ids)
    graph.set_route_resolution_map({})
    graph.set_config_gate_id_map({})

    return graph


# ---------------------------------------------------------------------------
# Test plugins for resume tests
# ---------------------------------------------------------------------------


class _DoublerTransform(BaseTransform):
    """Transform that doubles the value field."""

    name = "doubler"
    input_schema = _RowSchema
    output_schema = _RowSchema
    determinism = Determinism.DETERMINISTIC
    on_error = "discard"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        return TransformResult.success(
            make_pipeline_row({**row, "value": row["value"] * 2}),
            success_reason={"action": "doubler"},
        )


class _ResumeSink(_TestSinkBase):
    """Sink that collects rows into a class-level list for resume verification."""

    name = "collect_sink"
    results: ClassVar[list[dict[str, Any]]] = []

    def __init__(self) -> None:
        super().__init__()
        _ResumeSink.results = []

    def on_start(self, ctx: Any) -> None:
        pass

    def on_complete(self, ctx: Any) -> None:
        pass

    def write(self, rows: Any, ctx: Any) -> SinkWriteResult:
        _ResumeSink.results.extend(rows)
        return SinkWriteResult(artifact=ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="abc"))

    def close(self) -> None:
        pass


class _ResumeSource(_TestSourceBase):
    """Source that yields rows from a list with proper schema."""

    name = "list_source"
    output_schema = _RowSchema

    def __init__(self, data: list[dict[str, Any]]) -> None:
        super().__init__()
        self._data = data

    def on_start(self, ctx: Any) -> None:
        pass

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        yield from self.wrap_rows(self._data)

    def close(self) -> None:
        pass


class _MultiSourceAnySchema(PluginSchema):
    """Broad schema for mixed-contract multi-source resume tests."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


class _OrderResumeSchema(PluginSchema):
    """Order source schema for multi-source resume tests."""

    order_id: int
    amount: int


class _RefundResumeSchema(PluginSchema):
    """Refund source schema for multi-source resume tests."""

    refund_id: str
    amount: int


class _TypedResumeSource(_TestSourceBase):
    """Source with a fixed per-source schema contract and optional interrupt."""

    determinism = Determinism.IO_READ

    def __init__(
        self,
        *,
        name: str,
        rows: list[dict[str, Any]],
        output_schema: type[PluginSchema],
        field_types: dict[str, type[Any]],
        on_success: str,
        interrupt_after: int | None = None,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.output_schema = output_schema
        self.on_success = on_success
        self._rows = rows
        self._interrupt_after = interrupt_after
        self._shutdown_event = shutdown_event
        self._schema_contract = SchemaContract(
            mode="FIXED",
            fields=tuple(
                FieldContract(
                    normalized_name=field_name,
                    original_name=field_name,
                    python_type=field_type,
                    required=True,
                    source="declared",
                )
                for field_name, field_type in field_types.items()
            ),
            locked=True,
        )

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        contract = self._schema_contract
        assert contract is not None
        for source_row_index, row in enumerate(self._rows):
            if self._shutdown_event is not None and self._interrupt_after is not None and source_row_index + 1 >= self._interrupt_after:
                self._shutdown_event.set()
            yield SourceRow.valid(row, contract=contract, source_row_index=source_row_index)


class _FailingBeforeRowsSource(_TypedResumeSource):
    """Source that fails before yielding any row."""

    def load(self, ctx: Any) -> Iterator[SourceRow]:
        raise RuntimeError("orders source exploded before yielding rows")
        yield  # pragma: no cover - keeps this function a generator


class _SourceContractAnnotatingTransform(BaseTransform):
    """Annotate rows with the source contract visible to transform execution."""

    name = "source_contract_annotator"
    determinism = Determinism.DETERMINISTIC
    input_schema = _MultiSourceAnySchema
    output_schema = _MultiSourceAnySchema
    on_error = "discard"

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        data = row.to_dict()
        contract_fields = ",".join(sorted(field.normalized_name for field in row.contract.fields))
        if "order_id" in row:
            source_kind = "orders"
            business_id = f"order:{row['order_id']}"
        elif "refund_id" in row:
            source_kind = "refunds"
            business_id = f"refund:{row['refund_id']}"
        else:
            raise AssertionError(f"Unexpected multi-source row shape: {sorted(data)}")

        return TransformResult.success(
            make_pipeline_row(
                {
                    "source_kind": source_kind,
                    "business_id": business_id,
                    "amount": data["amount"],
                    "schema_fields": contract_fields,
                }
            ),
            success_reason={"action": "annotated_source_contract"},
        )


@dataclass(frozen=True)
class _MultiSourceResumeContext:
    """Live run state for production-path multi-source resume tests."""

    db: LandscapeDB
    checkpoint_mgr: CheckpointManager
    recovery_mgr: RecoveryManager
    checkpoint_config: RuntimeCheckpointConfig
    payload_store: FilesystemPayloadStore
    run_id: str
    graph: ExecutionGraph
    output_path: Path
    refunds_source_node_id: NodeID


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_multi_source_resume_pipeline(
    *,
    output_path: Path,
    refund_shutdown_event: threading.Event | None = None,
) -> tuple[PipelineConfig, ExecutionGraph]:
    orders = _TypedResumeSource(
        name="orders_source",
        rows=[
            {"order_id": 1, "amount": 100},
            {"order_id": 2, "amount": 125},
        ],
        output_schema=_OrderResumeSchema,
        field_types={"order_id": int, "amount": int},
        on_success="inbound",
    )
    refunds = _TypedResumeSource(
        name="refunds_source",
        rows=[{"refund_id": "R-1", "amount": 30}],
        output_schema=_RefundResumeSchema,
        field_types={"refund_id": str, "amount": int},
        on_success="inbound",
        interrupt_after=1 if refund_shutdown_event is not None else None,
        shutdown_event=refund_shutdown_event,
    )
    transform = _SourceContractAnnotatingTransform()
    sink = JSONSink(
        {
            "path": str(output_path),
            "format": "jsonl",
            "mode": "append",
            "schema": {"mode": "observed"},
        }
    )
    sources = {
        "orders": as_source(orders),
        "refunds": as_source(refunds),
    }
    sinks = {"output": as_sink(sink)}
    wired_transforms = wire_transforms(
        [as_transform(transform)],
        source_connection="inbound",
        final_sink="output",
        names=["annotate_source_contract"],
    )
    graph = ExecutionGraph.from_plugin_instances(
        sources=sources,
        source_settings_map={
            "orders": SourceSettings(plugin=orders.name, on_success="inbound", options={}),
            "refunds": SourceSettings(plugin=refunds.name, on_success="inbound", options={}),
        },
        transforms=wired_transforms,
        sinks=sinks,
        aggregations={},
        gates=[],
        queues={"inbound": QueueSettings(description="multi-source resume fan-in")},
    )
    config = PipelineConfig(
        sources=sources,
        transforms=[as_transform(transform)],
        sinks=sinks,
    )
    return config, graph


def _start_interrupted_multi_source_run(tmp_path: Path) -> _MultiSourceResumeContext:
    db = LandscapeDB(f"sqlite:///{tmp_path}/multi_source_resume.db")
    checkpoint_mgr = CheckpointManager(db)
    recovery_mgr = RecoveryManager(db, checkpoint_mgr)
    checkpoint_config = RuntimeCheckpointConfig.from_settings(CheckpointSettings(enabled=True, frequency="every_row"))
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    output_path = tmp_path / "multi_source_resume.jsonl"
    shutdown_event = threading.Event()
    config, graph = _build_multi_source_resume_pipeline(
        output_path=output_path,
        refund_shutdown_event=shutdown_event,
    )

    orchestrator = Orchestrator(
        db,
        checkpoint_manager=checkpoint_mgr,
        checkpoint_config=checkpoint_config,
    )
    with pytest.raises(GracefulShutdownError) as exc_info:
        orchestrator.run(
            config,
            graph=graph,
            payload_store=payload_store,
            shutdown_event=shutdown_event,
        )

    run_id = exc_info.value.run_id
    assert run_id is not None
    assert checkpoint_mgr.get_latest_checkpoint(run_id) is not None
    check = recovery_mgr.can_resume(run_id, graph)
    assert check.can_resume, f"Expected interrupted multi-source run to be resumable: {check.reason}"
    with db.engine.connect() as conn:
        refunds_source_node_id = conn.execute(
            select(run_sources_table.c.source_node_id).where(
                run_sources_table.c.run_id == run_id,
                run_sources_table.c.source_name == "refunds",
            )
        ).scalar_one()

    return _MultiSourceResumeContext(
        db=db,
        checkpoint_mgr=checkpoint_mgr,
        recovery_mgr=recovery_mgr,
        checkpoint_config=checkpoint_config,
        payload_store=payload_store,
        run_id=run_id,
        graph=graph,
        output_path=output_path,
        refunds_source_node_id=NodeID(refunds_source_node_id),
    )


def _append_crashed_refund_row(ctx: _MultiSourceResumeContext) -> str:
    """Simulate hard kill after source row persistence but before token creation."""
    factory = RecorderFactory(ctx.db, payload_store=ctx.payload_store)
    row = factory.data_flow.create_row(
        run_id=ctx.run_id,
        source_node_id=ctx.refunds_source_node_id,
        row_index=3,
        data={"refund_id": "R-2", "amount": 45},
        source_row_index=1,
        ingest_sequence=3,
    )
    return row.row_id


def _resume_multi_source_run(ctx: _MultiSourceResumeContext) -> Any:
    resume_config, resume_graph = _build_multi_source_resume_pipeline(output_path=ctx.output_path)
    resume_point = ctx.recovery_mgr.get_resume_point(ctx.run_id, resume_graph)
    assert resume_point is not None
    return Orchestrator(
        ctx.db,
        checkpoint_manager=ctx.checkpoint_mgr,
        checkpoint_config=ctx.checkpoint_config,
    ).resume(
        resume_point,
        resume_config,
        resume_graph,
        payload_store=ctx.payload_store,
    )


class TestMultiSourceCrashResume:
    """Production-path crash/resume coverage for mixed-source pipelines."""

    def test_declared_sources_are_recorded_ready_before_source_iteration(self, tmp_path: Path) -> None:
        """Later declared sources must be visible to resume even if never started."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/multi_source_ready.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        output_path = tmp_path / "multi_source_ready.jsonl"
        orders = _FailingBeforeRowsSource(
            name="orders_source",
            rows=[],
            output_schema=_OrderResumeSchema,
            field_types={"order_id": int, "amount": int},
            on_success="inbound",
        )
        refunds = _TypedResumeSource(
            name="refunds_source",
            rows=[{"refund_id": "R-1", "amount": 30}],
            output_schema=_RefundResumeSchema,
            field_types={"refund_id": str, "amount": int},
            on_success="inbound",
        )
        transform = _SourceContractAnnotatingTransform()
        sink = JSONSink(
            {
                "path": str(output_path),
                "format": "jsonl",
                "mode": "append",
                "schema": {"mode": "observed"},
            }
        )
        sources = {
            "orders": as_source(orders),
            "refunds": as_source(refunds),
        }
        sinks = {"output": as_sink(sink)}
        wired_transforms = wire_transforms(
            [as_transform(transform)],
            source_connection="inbound",
            final_sink="output",
            names=["annotate_source_contract"],
        )
        graph = ExecutionGraph.from_plugin_instances(
            sources=sources,
            source_settings_map={
                "orders": SourceSettings(plugin=orders.name, on_success="inbound", options={}),
                "refunds": SourceSettings(plugin=refunds.name, on_success="inbound", options={}),
            },
            transforms=wired_transforms,
            sinks=sinks,
            aggregations={},
            gates=[],
            queues={"inbound": QueueSettings(description="multi-source resume fan-in")},
        )
        config = PipelineConfig(
            sources=sources,
            transforms=[as_transform(transform)],
            sinks=sinks,
        )
        run_id = "run-multi-source-ready-before-iteration"

        with pytest.raises(RuntimeError, match="orders source exploded"):
            Orchestrator(db).run(
                config,
                graph=graph,
                payload_store=payload_store,
                run_id=run_id,
            )

        with db.engine.connect() as conn:
            lifecycle_rows = conn.execute(
                select(run_sources_table.c.source_name, run_sources_table.c.lifecycle_state)
                .where(run_sources_table.c.run_id == run_id)
                .order_by(run_sources_table.c.source_name)
            ).fetchall()

        assert {str(row.source_name): str(row.lifecycle_state) for row in lifecycle_rows} == {
            "orders": "loading",
            "refunds": "ready",
        }
        db.close()

    def test_interrupted_source_with_unprocessed_rows_refuses_resume(self, tmp_path: Path) -> None:
        ctx = _start_interrupted_multi_source_run(tmp_path)
        _append_crashed_refund_row(ctx)

        with pytest.raises(IncompleteSourceResumeError, match=r"source.*refunds.*interrupted"):
            _resume_multi_source_run(ctx)
        ctx.db.close()

    def test_interrupted_source_refusal_preserves_persisted_unprocessed_row(self, tmp_path: Path) -> None:
        ctx = _start_interrupted_multi_source_run(tmp_path)
        crashed_row_id = _append_crashed_refund_row(ctx)

        assert ctx.recovery_mgr.get_unprocessed_rows(ctx.run_id) == [crashed_row_id]
        with pytest.raises(IncompleteSourceResumeError, match=r"source.*refunds.*interrupted"):
            _resume_multi_source_run(ctx)
        assert ctx.recovery_mgr.get_unprocessed_rows(ctx.run_id) == [crashed_row_id]
        ctx.db.close()

    def test_interrupted_source_refusal_does_not_append_replayed_rows(self, tmp_path: Path) -> None:
        ctx = _start_interrupted_multi_source_run(tmp_path)
        _append_crashed_refund_row(ctx)

        with pytest.raises(IncompleteSourceResumeError, match=r"source.*refunds.*interrupted"):
            _resume_multi_source_run(ctx)

        assert [row["business_id"] for row in _jsonl_rows(ctx.output_path)] == [
            "order:1",
            "order:2",
            "refund:R-1",
        ]
        ctx.db.close()

    def test_resume_after_crash_during_source_iteration_records_partial_source_lifecycle(self, tmp_path: Path) -> None:
        ctx = _start_interrupted_multi_source_run(tmp_path)

        with ctx.db.engine.connect() as conn:
            lifecycle_rows = conn.execute(
                select(run_sources_table.c.source_name, run_sources_table.c.lifecycle_state)
                .where(run_sources_table.c.run_id == ctx.run_id)
                .order_by(run_sources_table.c.source_name)
            ).fetchall()

        lifecycle_by_source = {str(row.source_name): str(row.lifecycle_state) for row in lifecycle_rows}
        assert lifecycle_by_source == {
            "orders": "exhausted",
            "refunds": "interrupted",
        }
        with pytest.raises(IncompleteSourceResumeError, match=r"source.*refunds.*interrupted"):
            _resume_multi_source_run(ctx)
        ctx.db.close()


class TestResumeIdempotence:
    """Tests for resume idempotence -- same results whether interrupted or not."""

    def test_resume_produces_same_result_as_uninterrupted(self, tmp_path: Path) -> None:
        """Resume after interruption produces same final output.

        This test verifies the recovery idempotence property:
        1. Run pipeline A completely (baseline)
        2. Run pipeline B with checkpoint, simulate crash after 3 rows processed
        3. Resume pipeline B
        4. Verify: pre-crash output + resumed output == baseline output
        """
        source_data = [{"id": i, "value": (i + 1) * 10} for i in range(5)]
        # Expected after doubler: values = [20, 40, 60, 80, 100]

        # ===== Pipeline A: Run completely (baseline) =====
        db_a = LandscapeDB(f"sqlite:///{tmp_path}/baseline.db")
        payload_store_a = FilesystemPayloadStore(tmp_path / "payloads_a")
        source_a = _ResumeSource(source_data)
        transform_a = _DoublerTransform()
        sink_a = _ResumeSink()

        config_a = PipelineConfig(
            sources={"primary": as_source(source_a)},
            transforms=[transform_a],  # type: ignore[list-item]
            sinks={"default": as_sink(sink_a)},
        )

        # NOTE: Manual graph construction is required because the resume
        # portion of this test needs deterministic node IDs that match
        # stored checkpoint records. See _build_linear_graph docstring.
        orchestrator_a = Orchestrator(db_a)
        result_a = orchestrator_a.run(
            config_a,
            graph=_build_linear_graph(config_a),
            payload_store=payload_store_a,
        )

        assert result_a.status == RunStatus.COMPLETED
        assert result_a.rows_processed == 5
        baseline_output = list(_ResumeSink.results)
        assert len(baseline_output) == 5
        db_a.close()

        # ===== Pipeline B: Simulate crash after 3 rows, then resume =====
        db_b = LandscapeDB(f"sqlite:///{tmp_path}/resume_test.db")
        checkpoint_mgr = CheckpointManager(db_b)
        checkpoint_settings = CheckpointSettings(enabled=True, frequency="every_row")
        checkpoint_config = RuntimeCheckpointConfig.from_settings(checkpoint_settings)
        payload_store_b = FilesystemPayloadStore(tmp_path / "payloads_b")
        factory = RecorderFactory(db_b, payload_store=payload_store_b)

        # Create the source schema contract needed for resume
        source_contract = SchemaContract(
            mode="OBSERVED",
            fields=(
                FieldContract(
                    normalized_name="id",
                    original_name="id",
                    python_type=object,
                    required=False,
                    source="inferred",
                ),
                FieldContract(
                    normalized_name="value",
                    original_name="value",
                    python_type=object,
                    required=False,
                    source="inferred",
                ),
            ),
            locked=True,
        )

        # Phase 1: Create a "crashed" run with first 3 rows processed
        run = factory.run_lifecycle.begin_run(
            config={"test": "resume"},
            canonical_version="sha256-rfc8785-v1",
        )
        run_id = run.run_id

        # Store source schema for resume type fidelity
        with db_b.engine.connect() as conn:
            conn.execute(
                runs_table.update()
                .where(runs_table.c.run_id == run_id)
                .values(
                    source_schema_json=json.dumps(
                        {
                            "properties": {
                                "id": {"type": "integer"},
                                "value": {"type": "integer"},
                            },
                            "required": ["id", "value"],
                        }
                    )
                )
            )
            conn.commit()

        # Register nodes
        factory.data_flow.register_node(
            run_id=run_id,
            plugin_name="list_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source",
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig(mode="observed", fields=None),
        )
        factory.data_flow.register_node(
            run_id=run_id,
            plugin_name="doubler",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="transform_0",
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig(mode="observed", fields=None),
        )
        factory.data_flow.register_node(
            run_id=run_id,
            plugin_name="collect_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink_default",
            determinism=Determinism.IO_WRITE,
            schema_config=SchemaConfig(mode="observed", fields=None),
        )
        factory.data_flow.register_edge(
            run_id=run_id,
            from_node_id="source",
            to_node_id="transform_0",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        factory.data_flow.register_edge(
            run_id=run_id,
            from_node_id="transform_0",
            to_node_id="sink_default",
            label="continue",
            mode=RoutingMode.MOVE,
        )

        # ADR-025 §3 Decision 5 (G6): schema contracts live exclusively in
        # ``run_sources``. The legacy ``begin_run(schema_contract=...)``
        # path was deleted; resume reconstruction now reads the contract
        # from the per-source ``run_sources`` record. Mirror what the
        # production orchestrator writes via ``_emit_source_loading``.
        factory.run_lifecycle.record_run_source(
            run_id=run_id,
            source_node_id="source",
            source_name="source",
            plugin_name="list_source",
            config_hash="crash-and-resume",
            lifecycle_state="loaded",
            source_schema_json=json.dumps(
                {
                    "properties": {
                        "id": {"type": "integer"},
                        "value": {"type": "integer"},
                    },
                    "required": ["id", "value"],
                }
            ),
            schema_contract=source_contract,
        )
        # Record the source node's output contract for resume.
        factory.data_flow.update_node_output_contract(run_id, "source", source_contract)

        # Create all 5 rows with payloads (create_row auto-stores via payload_store_b)
        row_ids = []
        token_ids = []
        for i, row_data in enumerate(source_data):
            row = factory.data_flow.create_row(
                run_id=run_id,
                source_node_id="source",
                row_index=i,
                data=row_data,
                source_row_index=i,
                ingest_sequence=i,
            )
            row_ids.append(row.row_id)
            token = factory.data_flow.create_token(row_id=row.row_id)
            token_ids.append(token.token_id)

        # Build graph for checkpoint -- manual construction required because
        # checkpoint node IDs must match what we registered above.
        graph_b = ExecutionGraph()
        schema_config_dict: dict[str, Any] = {"schema": {"mode": "observed"}}
        graph_b.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="list_source",
            config={**schema_config_dict, "source_name": "primary"},
        )
        graph_b.add_node(
            "transform_0",
            node_type=NodeType.TRANSFORM,
            plugin_name="doubler",
            config=schema_config_dict,
        )
        graph_b.add_node(
            "sink_default",
            node_type=NodeType.SINK,
            plugin_name="collect_sink",
            config=schema_config_dict,
        )
        graph_b.add_edge("source", "transform_0", label="continue", mode=RoutingMode.MOVE)
        graph_b.add_edge("transform_0", "sink_default", label="continue", mode=RoutingMode.MOVE)
        graph_b.set_sink_id_map({SinkName("default"): NodeID("sink_default")})
        graph_b.set_transform_id_map({0: NodeID("transform_0")})
        graph_b.set_route_resolution_map({})
        graph_b.set_config_gate_id_map({})

        # Simulate that first 3 rows were processed (doubled)
        pre_crash_output = [{"id": i, "value": (i + 1) * 10 * 2} for i in range(3)]

        # Record terminal outcomes for first 3 rows
        for i in range(3):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=token_ids[i], run_id=run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="default",
            )

        # Create checkpoint at row 2 (0-indexed, so rows 0-2 processed)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            sequence_number=3,
            barrier_scalars=None,
            graph=graph_b,
        )

        # Mark run as failed (simulating crash)
        factory.run_lifecycle.complete_run(run_id, status=RunStatus.FAILED)

        # Epoch 21 (ADR-030 §B.4): begin_run minted this run's leader seat
        # (uniformity rule), and a hard-killed leader never releases it — the
        # seat stays HELD until its liveness window lapses. Resume's takeover
        # CAS requires vacant-or-expired, so craft the post-window image
        # deterministically (the in-DB picture an operator sees ~80s after
        # the crash) instead of sleeping out the window.
        with db_b.engine.begin() as conn:
            conn.execute(
                run_coordination_table.update()
                .where(run_coordination_table.c.run_id == run_id)
                .values(leader_heartbeat_expires_at=datetime.now(UTC) - timedelta(seconds=1))
            )

        # Phase 2: Resume and process remaining rows
        recovery_mgr = RecoveryManager(db_b, checkpoint_mgr)

        check = recovery_mgr.can_resume(run_id, graph_b)
        assert check.can_resume, f"Cannot resume: {check.reason}"

        resume_point = recovery_mgr.get_resume_point(run_id, graph_b)
        assert resume_point is not None

        # Create fresh plugins for resume
        _ResumeSink.results = []
        source_b = _ResumeSource(source_data)
        transform_b = _DoublerTransform()
        sink_b = _ResumeSink()

        config_b = PipelineConfig(
            sources={"primary": as_source(source_b)},
            transforms=[transform_b],  # type: ignore[list-item]
            sinks={"default": as_sink(sink_b)},
        )

        # NOTE: Manual graph construction for resume because node IDs
        # must match the checkpoint data from the pre-crash run.
        resume_graph = _build_linear_graph(config_b)

        orchestrator_b = Orchestrator(
            db_b,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        result_b = orchestrator_b.resume(
            resume_point,
            config_b,
            resume_graph,
            payload_store=payload_store_b,
        )

        assert result_b.status == RunStatus.COMPLETED
        # F2 (resume-fork-reemit): the resume RunResult now reports CUMULATIVE
        # counters reconstructed from the audit trail (both resume branches
        # finalize via derive_resume_terminal_status_from_audit), so a resumed
        # run matches an uninterrupted run — this test's whole premise. All 5
        # source rows reached a terminal outcome (3 recorded pre-crash + 2
        # re-driven), so rows_processed is the cumulative 5, matching run A
        # (line ~276), NOT the pre-F2 resume-only count of 2 (rows 3 and 4).
        assert result_b.rows_processed == 5

        resumed_output = list(_ResumeSink.results)
        assert len(resumed_output) == 2

        # Verify: pre-crash output + resumed output == baseline output
        combined_output = pre_crash_output + resumed_output
        assert len(combined_output) == 5
        assert combined_output == baseline_output, (
            f"Resume did not produce same result as uninterrupted run.\nExpected: {baseline_output}\nGot: {combined_output}"
        )

        # Epoch 21 (ADR-030 §B.4) — the resume-side coordination pin: the
        # takeover CAS bumped the seat to epoch 2, identity-evicted the
        # crashed begin_run leader, and the successful finalize released the
        # seat (vacant, epoch retained).
        with db_b.engine.connect() as conn:
            seat = conn.execute(select(run_coordination_table).where(run_coordination_table.c.run_id == run_id)).one()
            workers = conn.execute(
                select(run_workers_table).where(run_workers_table.c.run_id == run_id).order_by(run_workers_table.c.registered_at)
            ).all()
            event_types = (
                conn.execute(
                    select(run_coordination_events_table.c.event_type)
                    .where(run_coordination_events_table.c.run_id == run_id)
                    .order_by(run_coordination_events_table.c.seq)
                )
                .scalars()
                .all()
            )
        assert seat.leader_epoch == 2, "resume's takeover CAS must bump the seat epoch"
        assert seat.leader_worker_id is None, "graceful completion must release the seat"
        statuses_by_entry = {w.entry_point: w.status for w in workers}
        assert statuses_by_entry == {"run": "evicted", "resume": "departed"}, (
            "the crashed begin_run leader must be identity-evicted by the takeover; the resume leader departs on release"
        )
        assert "worker_evict" in event_types
        assert event_types[-1] == "leader_release"

        db_b.close()


class TestRetryBehavior:
    """Tests for retry behavior during processing."""

    def test_pipeline_with_failed_transform_records_failure(self, tmp_path: Path) -> None:
        """A pipeline that has a failing transform records the failure.

        When a transform returns TransformResult.error():
        1. Pipeline completes with status "completed"
        2. Only non-failing rows reach the sink
        3. Error is recorded in transform_errors table
        4. Error details match what the transform produced
        """
        from sqlalchemy import select

        # Custom transform that fails on specific row IDs
        class _ErroringTransform(BaseTransform):
            """Transform that returns error for specific row IDs."""

            name = "erroring_transform"
            input_schema = _RowSchema
            output_schema = _RowSchema
            determinism = Determinism.DETERMINISTIC
            on_error = "discard"

            def __init__(self, fail_ids: set[int]) -> None:
                super().__init__({"schema": {"mode": "observed"}})
                self._fail_ids = fail_ids

            def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
                if row["id"] in self._fail_ids:
                    return TransformResult.error(
                        {
                            "reason": "validation_failed",
                            "error": f"Row {row['id']} failed validation",
                        }
                    )
                return TransformResult.success(
                    make_pipeline_row(row.to_dict()),
                    success_reason={"action": "test"},
                )

        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")

        source_data = [
            {"id": 1, "value": 100},
            {"id": 2, "value": 200},
            {"id": 3, "value": 300},
        ]
        source = _ResumeSource(source_data)
        transform = _ErroringTransform(fail_ids={2})
        sink = _ResumeSink()
        _ResumeSink.results = []

        config = PipelineConfig(
            sources={"primary": as_source(source)},
            transforms=[transform],  # type: ignore[list-item]
            sinks={"default": as_sink(sink)},
        )

        # NOTE: Manual graph construction for consistency with checkpoint
        # tests that also use this helper; node IDs must be predictable.
        orchestrator = Orchestrator(db)
        result = orchestrator.run(
            config,
            graph=_build_linear_graph(config),
            payload_store=payload_store,
        )

        # Pipeline completes (errors are handled via routing, not as failures)
        assert result.status == RunStatus.COMPLETED_WITH_FAILURES
        assert result.rows_processed == 3

        # Only 2 rows make it to the sink (id=2 was discarded)
        assert len(_ResumeSink.results) == 2
        sink_ids = {r["id"] for r in _ResumeSink.results}
        assert sink_ids == {1, 3}

        # Verify error recorded in transform_errors table
        with db.engine.connect() as conn:
            errors = conn.execute(select(transform_errors_table).where(transform_errors_table.c.run_id == result.run_id)).fetchall()

        assert len(errors) == 1, f"Expected 1 error, got {len(errors)}"

        error = errors[0]
        assert error.destination == "discard"

        error_details = json.loads(error.error_details_json)
        assert error_details["reason"] == "validation_failed"
        assert error_details["error"] == "Row 2 failed validation"

        db.close()


class TestCheckpointRecovery:
    """Tests for checkpoint-based recovery."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment with database and checkpoint manager."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)

        return {
            "db": db,
            "checkpoint_manager": checkpoint_mgr,
            "recovery_manager": recovery_mgr,
            "tmp_path": tmp_path,
        }

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a minimal graph for checkpoint recovery tests.

        NOTE: Manual ExecutionGraph construction is required here because
        checkpoint tests need deterministic node IDs that match stored
        checkpoint records. The checkpoint table has FK constraints
        referencing specific node IDs, so we must use the exact IDs
        that were registered in the audit trail.
        """
        graph = ExecutionGraph()
        schema_config: dict[str, Any] = {"schema": {"mode": "observed"}}
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="test",
            config={**schema_config, "source_name": "source"},
        )
        graph.add_node(
            "transform",
            node_type=NodeType.TRANSFORM,
            plugin_name="test",
            config=schema_config,
        )
        return graph

    def test_checkpoint_preserves_partial_progress(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Create run with 5 rows, checkpoint at row 2, mark as failed.

        Verify get_unprocessed_rows() returns only rows 3-4.
        """
        db: LandscapeDB = test_env["db"]
        checkpoint_mgr: CheckpointManager = test_env["checkpoint_manager"]
        recovery_mgr: RecoveryManager = test_env["recovery_manager"]

        run_id = "checkpoint-partial-progress-test"
        now = datetime.now(UTC)
        contract_json, contract_hash = _create_test_schema_contract()
        source_schema_json = json.dumps(
            {"properties": {"id": {"type": "integer"}, "value": {"type": "integer"}}, "required": ["id", "value"]}
        )

        with db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )

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

            conn.execute(
                run_sources_table.insert().values(
                    run_id=run_id,
                    source_node_id="source",
                    source_name="source",
                    plugin_name="test_source",
                    lifecycle_state="loaded",
                    config_hash="test",
                    schema_json=source_schema_json,
                    schema_contract_json=contract_json,
                    schema_contract_hash=contract_hash,
                    field_resolution_json=None,
                    recorded_at=now,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="transform",
                    run_id=run_id,
                    plugin_name="test_transform",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create 5 rows with tokens
            for i in range(5):
                row_id = f"row-{i:03d}"
                token_id = f"tok-{i:03d}"
                conn.execute(
                    rows_table.insert().values(
                        row_id=row_id,
                        run_id=run_id,
                        source_node_id="source",
                        row_index=i,
                        source_row_index=i,
                        ingest_sequence=i,
                        source_data_hash=f"hash-{i}",
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=token_id,
                        row_id=row_id,
                        run_id=run_id,
                        created_at=now,
                    )
                )
                # Mark rows 0, 1, 2 as COMPLETED
                if i < 3:
                    conn.execute(
                        token_outcomes_table.insert().values(
                            outcome_id=f"outcome-{i:03d}",
                            run_id=run_id,
                            token_id=token_id,
                            outcome=TerminalOutcome.SUCCESS.value,
                            path=TerminalPath.DEFAULT_FLOW.value,
                            completed=1,
                            recorded_at=now,
                            sink_name="default",
                        )
                    )
            conn.commit()

        # Checkpoint at row 2 (token tok-002)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            sequence_number=2,
            barrier_scalars=None,
            graph=mock_graph,
        )

        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None
        assert checkpoint.sequence_number == 2

        check = recovery_mgr.can_resume(run_id, mock_graph)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        unprocessed = recovery_mgr.get_unprocessed_rows(run_id)
        assert len(unprocessed) == 2
        assert unprocessed == ["row-003", "row-004"]

    def test_checkpoint_survives_process_restart(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Create run + checkpoint with file-based DB, close DB, reopen.

        Verify checkpoint data is intact after simulated process restart.
        """
        tmp_path: Path = test_env["tmp_path"]
        db_path = tmp_path / "restart_test.db"

        # PHASE 1: Create database, run, and checkpoint
        db1 = LandscapeDB(f"sqlite:///{db_path}")
        checkpoint_mgr1 = CheckpointManager(db1)

        run_id = "checkpoint-restart-test"
        now = datetime.now(UTC)
        contract_json, contract_hash = _create_test_schema_contract()
        source_schema_json = json.dumps(
            {"properties": {"id": {"type": "integer"}, "value": {"type": "integer"}}, "required": ["id", "value"]}
        )

        with db1.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )

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

            conn.execute(
                run_sources_table.insert().values(
                    run_id=run_id,
                    source_node_id="source",
                    source_name="source",
                    plugin_name="test_source",
                    lifecycle_state="loaded",
                    config_hash="test",
                    schema_json=source_schema_json,
                    schema_contract_json=contract_json,
                    schema_contract_hash=contract_hash,
                    field_resolution_json=None,
                    recorded_at=now,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="transform",
                    run_id=run_id,
                    plugin_name="test_transform",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            conn.execute(
                rows_table.insert().values(
                    row_id="row-000",
                    run_id=run_id,
                    source_node_id="source",
                    row_index=0,
                    source_row_index=0,
                    ingest_sequence=0,
                    source_data_hash="hash-0",
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000",
                    row_id="row-000",
                    run_id=run_id,
                    created_at=now,
                )
            )
            conn.commit()

        # F1: the checkpoint carries only scalar barrier metadata; buffered
        # tokens live in journal BLOCKED rows.
        _scalars = BarrierScalars(
            aggregation={"test_agg": AggregationNodeScalars(count_fire_offset=2.0, condition_fire_offset=None)},
            coalesce={},
        )
        original_checkpoint = checkpoint_mgr1.create_checkpoint(
            run_id=run_id,
            sequence_number=0,
            barrier_scalars=_scalars,
            graph=mock_graph,
        )

        # Close database (simulate process exit)
        db1.close()

        # PHASE 2: Reopen database (simulate process restart)
        db2 = LandscapeDB(f"sqlite:///{db_path}")
        checkpoint_mgr2 = CheckpointManager(db2)
        recovery_mgr2 = RecoveryManager(db2, checkpoint_mgr2)

        restored_checkpoint = checkpoint_mgr2.get_latest_checkpoint(run_id)
        assert restored_checkpoint is not None

        # Verify all checkpoint fields match
        assert restored_checkpoint.checkpoint_id == original_checkpoint.checkpoint_id
        assert restored_checkpoint.run_id == original_checkpoint.run_id
        assert restored_checkpoint.sequence_number == original_checkpoint.sequence_number
        assert restored_checkpoint.upstream_topology_hash == original_checkpoint.upstream_topology_hash

        # Verify can_resume works with restored checkpoint
        check = recovery_mgr2.can_resume(run_id, mock_graph)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        # Verify resume point with barrier scalars — typed DTO round-trip
        resume_point = recovery_mgr2.get_resume_point(run_id, mock_graph)
        assert resume_point is not None
        assert resume_point.barrier_scalars is not None
        assert "test_agg" in resume_point.barrier_scalars.aggregation
        node_scalars = resume_point.barrier_scalars.aggregation["test_agg"]
        assert node_scalars.count_fire_offset == 2.0
        assert node_scalars.condition_fire_offset is None

        db2.close()


class TestAggregationRecovery:
    """Tests for recovery of aggregation-in-progress."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment with database and checkpoint manager."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        factory = make_factory(db)

        return {
            "db": db,
            "checkpoint_manager": checkpoint_mgr,
            "recovery_manager": recovery_mgr,
            "factory": factory,
            "tmp_path": tmp_path,
        }

    @pytest.fixture
    def mock_graph(self) -> ExecutionGraph:
        """Create a minimal graph for aggregation recovery tests.

        NOTE: Manual ExecutionGraph construction is required here because
        checkpoint tests need deterministic node IDs that match stored
        checkpoint records. The aggregation node ID "aggregator" must
        match what we register in the audit trail.
        """
        graph = ExecutionGraph()
        schema_config: dict[str, Any] = {"schema": {"mode": "observed"}}
        agg_config: dict[str, Any] = {
            "trigger": {"count": 1},
            "output_mode": "transform",
            "options": {"schema": {"mode": "observed"}},
            "schema": {"mode": "observed"},
        }
        graph.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="test",
            config={**schema_config, "source_name": "source"},
        )
        graph.add_node(
            "aggregator",
            node_type=NodeType.AGGREGATION,
            plugin_name="sum_agg",
            config=agg_config,
        )
        return graph

    def test_aggregation_barrier_scalars_recover_after_crash(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Checkpoint with barrier scalars, simulate crash, verify recovery.

        The recovery must restore the exact scalar trigger metadata. F1: the
        buffered tokens themselves persist as journal BLOCKED rows (proven by
        the journal restore suites); the checkpoint round-trips only the
        underivable trigger latches, so this pin needs no fabricated
        rows/tokens at all.
        """
        db: LandscapeDB = test_env["db"]
        checkpoint_mgr: CheckpointManager = test_env["checkpoint_manager"]
        recovery_mgr: RecoveryManager = test_env["recovery_manager"]
        factory: RecorderFactory = test_env["factory"]

        test_contract = SchemaContract(
            fields=(
                FieldContract(
                    normalized_name="test_field",
                    original_name="test_field",
                    python_type=str,
                    required=True,
                    source="declared",
                ),
            ),
            mode="FIXED",
            locked=True,
        )
        run = factory.run_lifecycle.begin_run(
            config={"aggregation": {"trigger": {"count": 5}}},
            canonical_version="sha256-rfc8785-v1",
        )

        # Register nodes using raw SQL
        now = datetime.now(UTC)
        with db.engine.connect() as conn:
            conn.execute(
                nodes_table.insert().values(
                    node_id="source",
                    run_id=run.run_id,
                    plugin_name="test_source",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="aggregator",
                    run_id=run.run_id,
                    plugin_name="sum_agg",
                    node_type=NodeType.AGGREGATION,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )
            conn.commit()

        # ADR-025 §3 Decision 5 (G6): record per-source contract via
        # ``run_sources`` (the legacy ``begin_run(schema_contract=...)``
        # path was deleted; readers and writers are now symmetric on
        # ``run_sources``).
        factory.run_lifecycle.record_run_source(
            run_id=run.run_id,
            source_node_id="source",
            source_name="source",
            plugin_name="test_source",
            config_hash="test",
            lifecycle_state="loaded",
            source_schema_json='{"properties": {"test_field": {"type": "string"}}, "required": ["test_field"]}',
            schema_contract=test_contract,
        )

        # Create the scalar barrier metadata for the in-flight aggregation —
        # the only checkpoint-borne barrier state post-F1.
        scalars = BarrierScalars(
            aggregation={"aggregator": AggregationNodeScalars(count_fire_offset=3.0, condition_fire_offset=1.0)},
            coalesce={},
        )

        # Create checkpoint with barrier scalars
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            sequence_number=2,
            barrier_scalars=scalars,
            graph=mock_graph,
        )

        # Simulate crash
        factory.run_lifecycle.complete_run(run.run_id, status=RunStatus.FAILED)

        # Verify can resume
        check = recovery_mgr.can_resume(run.run_id, mock_graph)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        # Get resume point with barrier scalars
        resume_point = recovery_mgr.get_resume_point(run.run_id, mock_graph)
        assert resume_point is not None

        # Verify the scalar trigger metadata is restored exactly — typed DTO
        assert resume_point.barrier_scalars is not None
        restored_scalars = resume_point.barrier_scalars

        assert "aggregator" in restored_scalars.aggregation
        node_scalars = restored_scalars.aggregation["aggregator"]
        assert node_scalars.count_fire_offset == 3.0
        assert node_scalars.condition_fire_offset == 1.0
