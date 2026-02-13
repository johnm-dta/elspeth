"""E2E tests for crash recovery and resume scenarios.

Tests verify that ELSPETH can recover from crashes and resume
processing, producing the same results as uninterrupted runs.

Uses file-based SQLite and real payload stores. No mocks except
external services.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest

from elspeth.contracts import (
    ArtifactDescriptor,
    Determinism,
    NodeType,
    PipelineRow,
    PluginSchema,
    RoutingMode,
    RowOutcome,
    RunStatus,
    SourceRow,
)
from elspeth.contracts.config.runtime import RuntimeCheckpointConfig
from elspeth.contracts.contract_records import ContractAuditRecord
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.contracts.types import NodeID, SinkName
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.config import CheckpointSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    runs_table,
    token_outcomes_table,
    tokens_table,
    transform_errors_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.results import TransformResult
from elspeth.testing import make_pipeline_row
from tests.fixtures.base_classes import (
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
)

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
        plugin_name=config.source.name,
        config=schema_config,
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

    def write(self, rows: Any, ctx: Any) -> ArtifactDescriptor:
        _ResumeSink.results.extend(rows)
        return ArtifactDescriptor.for_file(path="memory", size_bytes=0, content_hash="abc")

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
            source=as_source(source_a),
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
        recorder = LandscapeRecorder(db_b, payload_store=payload_store_b)

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
        run = recorder.begin_run(
            config={"test": "resume"},
            canonical_version="sha256-rfc8785-v1",
            schema_contract=source_contract,
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
        recorder.register_node(
            run_id=run_id,
            plugin_name="list_source",
            node_type=NodeType.SOURCE,
            plugin_version="1.0",
            config={},
            node_id="source",
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig(mode="observed", fields=None),
        )
        recorder.register_node(
            run_id=run_id,
            plugin_name="doubler",
            node_type=NodeType.TRANSFORM,
            plugin_version="1.0",
            config={},
            node_id="transform_0",
            determinism=Determinism.DETERMINISTIC,
            schema_config=SchemaConfig(mode="observed", fields=None),
        )
        recorder.register_node(
            run_id=run_id,
            plugin_name="collect_sink",
            node_type=NodeType.SINK,
            plugin_version="1.0",
            config={},
            node_id="sink_default",
            determinism=Determinism.IO_WRITE,
            schema_config=SchemaConfig(mode="observed", fields=None),
        )
        recorder.register_edge(
            run_id=run_id,
            from_node_id="source",
            to_node_id="transform_0",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        recorder.register_edge(
            run_id=run_id,
            from_node_id="transform_0",
            to_node_id="sink_default",
            label="continue",
            mode=RoutingMode.MOVE,
        )

        # Record the contract on the source node for resume
        recorder.update_run_contract(run_id, source_contract)
        recorder.update_node_output_contract(run_id, "source", source_contract)

        # Create all 5 rows with payloads
        row_ids = []
        token_ids = []
        for i, row_data in enumerate(source_data):
            payload_ref = payload_store_b.store(json.dumps(row_data).encode("utf-8"))
            row = recorder.create_row(
                run_id=run_id,
                source_node_id="source",
                row_index=i,
                data=row_data,
                payload_ref=payload_ref,
            )
            row_ids.append(row.row_id)
            token = recorder.create_token(row_id=row.row_id)
            token_ids.append(token.token_id)

        # Build graph for checkpoint -- manual construction required because
        # checkpoint node IDs must match what we registered above.
        graph_b = ExecutionGraph()
        schema_config_dict: dict[str, Any] = {"schema": {"mode": "observed"}}
        graph_b.add_node(
            "source",
            node_type=NodeType.SOURCE,
            plugin_name="list_source",
            config=schema_config_dict,
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
            recorder.record_token_outcome(
                token_id=token_ids[i],
                run_id=run_id,
                outcome=RowOutcome.COMPLETED,
                sink_name="default",
            )

        # Create checkpoint at row 2 (0-indexed, so rows 0-2 processed)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id=token_ids[2],
            node_id="sink_default",
            sequence_number=3,
            graph=graph_b,
        )

        # Mark run as failed (simulating crash)
        recorder.complete_run(run_id, status=RunStatus.FAILED)

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
            source=as_source(source_b),
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
        assert result_b.rows_processed == 2  # Only rows 3 and 4

        resumed_output = list(_ResumeSink.results)
        assert len(resumed_output) == 2

        # Verify: pre-crash output + resumed output == baseline output
        combined_output = pre_crash_output + resumed_output
        assert len(combined_output) == 5
        assert combined_output == baseline_output, (
            f"Resume did not produce same result as uninterrupted run.\nExpected: {baseline_output}\nGot: {combined_output}"
        )

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

            def __init__(self, fail_ids: set[str]) -> None:
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
            {"id": "row_1", "value": 100},
            {"id": "row_2", "value": 200},
            {"id": "row_3", "value": 300},
        ]
        source = _ResumeSource(source_data)
        transform = _ErroringTransform(fail_ids={"row_2"})
        sink = _ResumeSink()
        _ResumeSink.results = []

        config = PipelineConfig(
            source=as_source(source),
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
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 3

        # Only 2 rows make it to the sink (row_2 was discarded)
        assert len(_ResumeSink.results) == 2
        sink_ids = {r["id"] for r in _ResumeSink.results}
        assert sink_ids == {"row_1", "row_3"}

        # Verify error recorded in transform_errors table
        with db.engine.connect() as conn:
            errors = conn.execute(select(transform_errors_table).where(transform_errors_table.c.run_id == result.run_id)).fetchall()

        assert len(errors) == 1, f"Expected 1 error, got {len(errors)}"

        error = errors[0]
        assert error.destination == "discard"

        error_details = json.loads(error.error_details_json)
        assert error_details["reason"] == "validation_failed"
        assert error_details["error"] == "Row row_2 failed validation"

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
            config=schema_config,
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

        with db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                    schema_contract_json=contract_json,
                    schema_contract_hash=contract_hash,
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
                        source_data_hash=f"hash-{i}",
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=token_id,
                        row_id=row_id,
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
                            outcome=RowOutcome.COMPLETED.value,
                            is_terminal=1,
                            recorded_at=now,
                            sink_name="default",
                        )
                    )
            conn.commit()

        # Checkpoint at row 2 (token tok-002)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="tok-002",
            node_id="transform",
            sequence_number=2,
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

        with db1.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                    schema_contract_json=contract_json,
                    schema_contract_hash=contract_hash,
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
                    source_data_hash="hash-0",
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-000",
                    row_id="row-000",
                    created_at=now,
                )
            )
            conn.commit()

        original_checkpoint = checkpoint_mgr1.create_checkpoint(
            run_id=run_id,
            token_id="tok-000",
            node_id="transform",
            sequence_number=0,
            graph=mock_graph,
            aggregation_state={"buffer": [1, 2, 3], "sum": 6},
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
        assert restored_checkpoint.token_id == original_checkpoint.token_id
        assert restored_checkpoint.node_id == original_checkpoint.node_id
        assert restored_checkpoint.sequence_number == original_checkpoint.sequence_number
        assert restored_checkpoint.upstream_topology_hash == original_checkpoint.upstream_topology_hash
        assert restored_checkpoint.checkpoint_node_config_hash == original_checkpoint.checkpoint_node_config_hash

        # Verify can_resume works with restored checkpoint
        check = recovery_mgr2.can_resume(run_id, mock_graph)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        # Verify resume point with aggregation state
        resume_point = recovery_mgr2.get_resume_point(run_id, mock_graph)
        assert resume_point is not None
        assert resume_point.aggregation_state is not None
        assert resume_point.aggregation_state["buffer"] == [1, 2, 3]
        assert resume_point.aggregation_state["sum"] == 6

        db2.close()


class TestAggregationRecovery:
    """Tests for recovery of aggregation-in-progress."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment with database and checkpoint manager."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        checkpoint_mgr = CheckpointManager(db)
        recovery_mgr = RecoveryManager(db, checkpoint_mgr)
        recorder = LandscapeRecorder(db)

        return {
            "db": db,
            "checkpoint_manager": checkpoint_mgr,
            "recovery_manager": recovery_mgr,
            "recorder": recorder,
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
            config=schema_config,
        )
        graph.add_node(
            "aggregator",
            node_type=NodeType.AGGREGATION,
            plugin_name="sum_agg",
            config=agg_config,
        )
        return graph

    def test_aggregation_state_recovers_after_crash(self, test_env: dict[str, Any], mock_graph: ExecutionGraph) -> None:
        """Create run with partial aggregation (3 rows buffered, trigger at 5).

        Checkpoint with aggregation state, simulate crash, verify recovery
        restores exact aggregation state (buffer, count, sum).
        """
        db: LandscapeDB = test_env["db"]
        checkpoint_mgr: CheckpointManager = test_env["checkpoint_manager"]
        recovery_mgr: RecoveryManager = test_env["recovery_manager"]
        recorder: LandscapeRecorder = test_env["recorder"]

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
        run = recorder.begin_run(
            config={"aggregation": {"trigger": {"count": 5}}},
            canonical_version="sha256-rfc8785-v1",
            schema_contract=test_contract,
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

        # Create 3 rows (partial aggregation -- trigger is at 5)
        tokens = []
        for i in range(3):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data={"id": i, "value": (i + 1) * 100},
            )
            token = recorder.create_token(row_id=row.row_id)
            tokens.append(token)

        # Create aggregation state (buffer of 3 rows, sum=600)
        agg_state = {
            "buffer": [
                {"id": 0, "value": 100},
                {"id": 1, "value": 200},
                {"id": 2, "value": 300},
            ],
            "count": 3,
            "sum": 600,
            "expected_trigger": 5,
        }

        # Create checkpoint with aggregation state
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=tokens[-1].token_id,
            node_id="aggregator",
            sequence_number=2,
            aggregation_state=agg_state,
            graph=mock_graph,
        )

        # Simulate crash
        recorder.complete_run(run.run_id, status=RunStatus.FAILED)

        # Verify can resume
        check = recovery_mgr.can_resume(run.run_id, mock_graph)
        assert check.can_resume is True, f"Cannot resume: {check.reason}"

        # Get resume point with aggregation state
        resume_point = recovery_mgr.get_resume_point(run.run_id, mock_graph)
        assert resume_point is not None

        # Verify aggregation state is restored exactly
        assert resume_point.aggregation_state is not None
        restored_state = resume_point.aggregation_state

        assert restored_state["count"] == 3
        assert restored_state["sum"] == 600
        assert restored_state["expected_trigger"] == 5
        assert len(restored_state["buffer"]) == 3

        # Verify buffer contents
        assert restored_state["buffer"][0] == {"id": 0, "value": 100}
        assert restored_state["buffer"][1] == {"id": 1, "value": 200}
        assert restored_state["buffer"][2] == {"id": 2, "value": 300}
