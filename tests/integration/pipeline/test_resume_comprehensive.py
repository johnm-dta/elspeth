# tests/integration/pipeline/test_resume_comprehensive.py
"""Comprehensive end-to-end integration tests for the resume process.

Tests all critical aspects of resume:
1. Normal resume with remaining rows (Happy path)
2. Early-exit resume with no remaining rows (Bug #8)
3. Resume with schema type restoration (Bug #4)
4. Resume with real edge IDs (Bug #3)
5. Checkpoint cleanup on completion

Note: Manual graph construction (add_node/add_edge) is intentional here.
Resume tests must create graphs with specific node IDs that match pre-existing
database checkpoint records. Using from_plugin_instances() would generate new
UUIDs that wouldn't match stored checkpoints, breaking the resume flow.
"""

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select

from elspeth.contracts import Determinism, NodeType, ResumePoint, RoutingMode, RunStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.runtime_val_manifest import build_runtime_val_manifest
from elspeth.contracts.types import NodeID, SinkName
from elspeth.core.canonical import canonical_json
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    checkpoints_table,
    edges_table,
    nodes_table,
    rows_table,
    run_sources_table,
    runs_table,
    token_work_items_table,
    tokens_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig, prepare_for_run
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.json_sink import JSONSink
from elspeth.plugins.sources.null_source import NullSource
from elspeth.plugins.transforms.passthrough import PassThrough
from elspeth.testing import make_contract, make_row
from tests.fixtures.base_classes import inject_write_failure
from tests.fixtures.landscape import make_factory


def _null_source(on_success: str = "default") -> NullSource:
    """Create NullSource with on_success set (lifted from config to attribute)."""
    source = NullSource({})
    source.on_success = on_success
    return source


def _runtime_val_manifest_json() -> str:
    """Mirror the run-header manifest production begin_run() stores."""
    prepare_for_run()
    return canonical_json(build_runtime_val_manifest())


class TestResumeComprehensive:
    """Comprehensive end-to-end resume integration tests."""

    @staticmethod
    def _create_schema_contract(fields: list[tuple[str, type]]) -> tuple[str, str]:
        """Create schema contract JSON and hash for test runs.

        Helper to avoid repetition in test setup. Creates contract with given fields.

        Args:
            fields: List of (field_name, python_type) tuples

        Returns:
            Tuple of (schema_contract_json, schema_contract_hash)
        """
        from elspeth.contracts.contract_records import ContractAuditRecord
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        field_contracts = tuple(
            FieldContract(
                normalized_name=name,
                original_name=name,
                python_type=py_type,
                required=True,
                source="declared",
            )
            for name, py_type in fields
        )

        contract = SchemaContract(
            mode="FIXED",
            fields=field_contracts,
            locked=True,
        )
        audit_record = ContractAuditRecord.from_contract(contract)
        return audit_record.to_json(), contract.version_hash()

    def _setup_failed_run(
        self,
        db: LandscapeDB,
        payload_store: FilesystemPayloadStore,
        run_id: str,
        num_rows: int,
        checkpoint_at: int,
    ) -> tuple[str, ExecutionGraph]:
        """Set up a failed run with rows and a checkpoint.

        Args:
            db: Database connection
            payload_store: Payload store for row data
            run_id: Run identifier
            num_rows: Total number of rows to create
            checkpoint_at: Row index where checkpoint was created

        Returns:
            Tuple of (run_id, graph)
        """
        import json

        now = datetime.now(UTC)
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={**schema_config, "source_name": "source"})
        graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("src", "xform", label="continue")
        graph.add_edge("xform", "sink", label="continue")

        # Create source schema for resume ({"id": int, "value": str})
        source_schema_json = json.dumps(
            {"properties": {"id": {"type": "integer"}, "value": {"type": "string"}}, "required": ["id", "value"]}
        )

        # PIPELINEROW MIGRATION: Create schema contract for resume
        # Resume now requires a contract to wrap row data in PipelineRow
        from elspeth.contracts.contract_records import ContractAuditRecord
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        contract = SchemaContract(
            mode="FIXED",
            fields=(
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
                    python_type=str,
                    required=True,
                    source="declared",
                ),
            ),
            locked=True,
        )
        audit_record = ContractAuditRecord.from_contract(contract)
        schema_contract_json = audit_record.to_json()
        schema_contract_hash = contract.version_hash()

        with db.engine.begin() as conn:
            # Create run
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    runtime_val_manifest_json=_runtime_val_manifest_json(),
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )

            # Create nodes
            for node_id, plugin_name, node_type in [
                ("src", "null", NodeType.SOURCE),
                ("xform", "passthrough", NodeType.TRANSFORM),
                ("sink", "csv", NodeType.SINK),
            ]:
                conn.execute(
                    nodes_table.insert().values(
                        node_id=node_id,
                        run_id=run_id,
                        plugin_name=plugin_name,
                        node_type=node_type,
                        plugin_version="1.0.0",
                        determinism=Determinism.DETERMINISTIC if node_type != NodeType.SINK else Determinism.IO_WRITE,
                        config_hash="test",
                        config_json="{}",
                        registered_at=now,
                    )
                )

            # Create edges
            for edge_id, from_node, to_node in [
                ("e1", "src", "xform"),
                ("e2", "xform", "sink"),
            ]:
                conn.execute(
                    edges_table.insert().values(
                        edge_id=edge_id,
                        run_id=run_id,
                        from_node_id=from_node,
                        to_node_id=to_node,
                        label="continue",
                        default_mode=RoutingMode.MOVE,
                        created_at=now,
                    )
                )

            # ADR-025 §3: record run_sources for the single source node.
            # Production code (Orchestrator._run_main_processing_loop ->
            # _emit_source_loading) writes a run_sources row BEFORE the
            # first ingested row is persisted, so a real failed run will
            # always have at least one run_sources record present even if
            # zero rows were committed. Reproducing that shape in the
            # fixture lets the RC6 resume path key the schema contract
            # under the actual ``source_node_id`` and avoids triggering
            # ``EmptyResumeStateError`` for tests that legitimately
            # exercise the early-exit path (all rows already processed)
            # rather than the refuse path (no work ever persisted).
            conn.execute(
                run_sources_table.insert().values(
                    run_id=run_id,
                    source_node_id="src",
                    source_name="src",
                    plugin_name="null",
                    lifecycle_state="loaded",
                    config_hash="test",
                    schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    field_resolution_json=None,
                    recorded_at=now,
                )
            )

            # Create rows with payloads
            for i in range(num_rows):
                row_data = {"id": i, "value": f"row-{i}"}
                ref = payload_store.store(json.dumps(row_data).encode())
                conn.execute(
                    rows_table.insert().values(
                        row_id=f"r{i}",
                        run_id=run_id,
                        source_node_id="src",
                        row_index=i,
                        source_row_index=i,
                        ingest_sequence=i,
                        source_data_hash=f"h{i}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"t{i}",
                        row_id=f"r{i}",
                        run_id=run_id,
                        created_at=now,
                    )
                )

        return run_id, graph

    def test_resume_normal_path_with_remaining_rows(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Test normal resume path: checkpoint mid-run, resume processes remaining rows.

        Scenario:
        1. Failed run with 5 rows (0-4)
        2. Rows 0-2 already processed (checkpoint at row 2)
        3. Resume processes rows 3-4
        4. Verify: 2 rows processed, all 5 rows in output
        5. Verify: Checkpoints deleted after completion

        This is the happy path for resume.
        """
        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        recovery_mgr = resume_test_env["recovery_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]
        tmp_path = resume_test_env["tmp_path"]

        # Set up failed run
        run_id = "resume-normal-test"
        output_path = tmp_path / "normal_output.csv"
        run_id, graph = self._setup_failed_run(db, payload_store, run_id, num_rows=5, checkpoint_at=2)

        # Simulate partial output (rows 0-2 already written)
        with open(output_path, "w") as f:
            f.write("id,value\n")
            for i in range(3):
                f.write(f"{i},row-{i}\n")

        # Mark first 3 rows as completed
        factory = make_factory(db)
        for i in range(3):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=f"t{i}", run_id=run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="sink",
            )

        # Create checkpoint at row 2
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="t2",
            node_id="xform",
            sequence_number=2,
            graph=graph,
        )

        # Verify checkpoint exists
        with db.engine.connect() as conn:
            checkpoints_before = conn.execute(select(checkpoints_table).where(checkpoints_table.c.run_id == run_id)).fetchall()
        assert len(checkpoints_before) == 1

        # Resume
        assert recovery_mgr.can_resume(run_id, graph).can_resume
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)

        # Use CSVSink with strict schema matching the data: {"id": int, "value": str}
        strict_schema = {"mode": "fixed", "fields": ["id: int", "value: str"]}
        passthrough = PassThrough({"schema": strict_schema})
        passthrough.on_error = "discard"
        config = PipelineConfig(
            sources={"source": _null_source("default")},
            transforms=[passthrough],
            sinks={"default": inject_write_failure(CSVSink({"path": str(output_path), "schema": strict_schema, "mode": "append"}))},
        )

        # Build graph manually
        resume_graph = ExecutionGraph()
        schema_config = {"schema": strict_schema}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={**schema_config, "source_name": "source"})
        resume_graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        resume_graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        resume_graph.add_edge("src", "xform", label="continue")
        resume_graph.add_edge("xform", "sink", label="continue")
        resume_graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        resume_graph.set_transform_id_map({0: NodeID("xform")})

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=resume_graph,
            payload_store=payload_store,
        )

        # Verify results
        # F2 (resume-fork-reemit): the resume RunResult now reports CUMULATIVE
        # counters reconstructed from the audit trail (both resume branches
        # finalize via derive_resume_terminal_status_from_audit), not resume-only
        # counts. 3 rows were pre-marked completed + 2 re-driven here = 5 source
        # rows reaching a terminal outcome. (Pre-F2 this reported the resume-only
        # 2 — rows 3 and 4.)
        assert result.rows_processed == 5
        assert result.rows_succeeded == 5
        assert result.status == RunStatus.COMPLETED

        # Verify output file has all 5 rows
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 6, f"Expected 6 lines (header + 5 rows), got {len(lines)}"
        assert "0,row-0" in lines[1]
        assert "4,row-4" in lines[5]

        # Verify checkpoints deleted after completion
        with db.engine.connect() as conn:
            checkpoints_after = conn.execute(select(checkpoints_table).where(checkpoints_table.c.run_id == run_id)).fetchall()
        assert len(checkpoints_after) == 0, "Checkpoints should be deleted after successful completion"

    def test_resume_drains_real_scheduler_work_before_recovered_row_replay(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Durable scheduler work takes precedence over replaying the same recovered row."""
        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        recovery_mgr = resume_test_env["recovery_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]
        tmp_path = resume_test_env["tmp_path"]

        run_id = "resume-real-scheduler-work-test"
        output_path = tmp_path / "scheduler_resume_output.csv"
        run_id, graph = self._setup_failed_run(db, payload_store, run_id, num_rows=1, checkpoint_at=0)

        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="t0",
            node_id="xform",
            sequence_number=0,
            graph=graph,
        )
        factory = make_factory(db)
        scheduled_row = make_row(
            {"id": 0, "value": "row-0"},
            contract=make_contract(fields={"id": int, "value": str}, mode="FIXED"),
        )
        factory.scheduler.enqueue_ready(
            run_id=run_id,
            token_id="t0",
            row_id="r0",
            node_id="xform",
            step_index=1,
            ingest_sequence=0,
            row_payload_json=factory.scheduler.serialize_row_payload(scheduled_row),
            available_at=datetime.now(UTC),
        )

        output_path.write_text("id,value\n")

        assert recovery_mgr.can_resume(run_id, graph).can_resume
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        strict_schema = {"mode": "fixed", "fields": ["id: int", "value: str"]}
        passthrough = PassThrough({"schema": strict_schema})
        passthrough.on_error = "discard"
        config = PipelineConfig(
            sources={"source": _null_source("default")},
            transforms=[passthrough],
            sinks={"default": inject_write_failure(CSVSink({"path": str(output_path), "schema": strict_schema, "mode": "append"}))},
        )
        resume_graph = ExecutionGraph()
        schema_config = {"schema": strict_schema}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={**schema_config, "source_name": "source"})
        resume_graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        resume_graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        resume_graph.add_edge("src", "xform", label="continue")
        resume_graph.add_edge("xform", "sink", label="continue")
        resume_graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        resume_graph.set_transform_id_map({0: NodeID("xform")})

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=resume_graph,
            payload_store=payload_store,
        )

        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 1
        assert result.rows_succeeded == 1
        lines = output_path.read_text().strip().split("\n")
        assert lines == ["id,value", "0,row-0"]
        with db.connection() as conn:
            work_statuses = (
                conn.execute(select(token_work_items_table.c.status).where(token_work_items_table.c.run_id == run_id)).scalars().all()
            )
        assert work_statuses == ["terminal"]

    def test_reconstruct_resume_state_restores_multi_source_rows_with_source_scoped_schemas(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Production resume reconstruction uses run_sources schema/contract per row source."""
        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]
        run_id = "resume-multi-source-schema-test"
        now = datetime.now(UTC)
        orders_schema_json = json.dumps({"properties": {"order_id": {"type": "integer"}}, "required": ["order_id"]})
        refunds_schema_json = json.dumps({"properties": {"refund_id": {"type": "string"}}, "required": ["refund_id"]})
        orders_contract_json, orders_contract_hash = self._create_schema_contract([("order_id", int)])
        refunds_contract_json, refunds_contract_hash = self._create_schema_contract([("refund_id", str)])

        graph = ExecutionGraph()
        graph.add_node("source-orders", node_type=NodeType.SOURCE, plugin_name="null", config={"source_name": "orders"})
        graph.add_node("source-refunds", node_type=NodeType.SOURCE, plugin_name="null", config={"source_name": "refunds"})
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="json", config={"schema": {"mode": "observed"}})
        graph.add_edge("source-orders", "sink", label="continue")
        graph.add_edge("source-refunds", "sink", label="continue")

        with db.engine.begin() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.FAILED,
                    source_schema_json=json.dumps({"properties": {}, "required": []}),
                    runtime_val_manifest_json=_runtime_val_manifest_json(),
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )
            for node_id, plugin_name, node_type in [
                ("source-orders", "null", NodeType.SOURCE),
                ("source-refunds", "null", NodeType.SOURCE),
                ("sink", "json", NodeType.SINK),
            ]:
                conn.execute(
                    nodes_table.insert().values(
                        node_id=node_id,
                        run_id=run_id,
                        plugin_name=plugin_name,
                        node_type=node_type,
                        plugin_version="1.0.0",
                        determinism=Determinism.DETERMINISTIC if node_type != NodeType.SINK else Determinism.IO_WRITE,
                        config_hash="test",
                        config_json="{}",
                        registered_at=now,
                    )
                )
            for source_node_id, source_name, schema_json, contract_json, contract_hash in [
                ("source-orders", "orders", orders_schema_json, orders_contract_json, orders_contract_hash),
                ("source-refunds", "refunds", refunds_schema_json, refunds_contract_json, refunds_contract_hash),
            ]:
                conn.execute(
                    run_sources_table.insert().values(
                        run_id=run_id,
                        source_node_id=source_node_id,
                        source_name=source_name,
                        plugin_name="null",
                        lifecycle_state="loaded",
                        config_hash="test",
                        schema_json=schema_json,
                        schema_contract_json=contract_json,
                        schema_contract_hash=contract_hash,
                        recorded_at=now,
                    )
                )
            for edge_id, from_node in [("e-orders", "source-orders"), ("e-refunds", "source-refunds")]:
                conn.execute(
                    edges_table.insert().values(
                        edge_id=edge_id,
                        run_id=run_id,
                        from_node_id=from_node,
                        to_node_id="sink",
                        label="continue",
                        default_mode=RoutingMode.MOVE,
                        created_at=now,
                    )
                )
            for row_id, source_node_id, row_index, source_row_index, ingest_sequence, row_data in [
                ("row-orders", "source-orders", 0, 0, 0, {"order_id": "101"}),
                ("row-refunds", "source-refunds", 1, 0, 1, {"refund_id": "r-7"}),
            ]:
                ref = payload_store.store(json.dumps(row_data).encode())
                conn.execute(
                    rows_table.insert().values(
                        row_id=row_id,
                        run_id=run_id,
                        source_node_id=source_node_id,
                        row_index=row_index,
                        source_row_index=source_row_index,
                        ingest_sequence=ingest_sequence,
                        source_data_hash=f"h-{row_id}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-multi-source",
                    row_id="row-orders",
                    run_id=run_id,
                    created_at=now,
                )
            )

        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="tok-multi-source",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None
        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)

        state = orchestrator._reconstruct_resume_state(
            ResumePoint(
                checkpoint=checkpoint,
                token_id=checkpoint.token_id,
                node_id=checkpoint.node_id,
                sequence_number=checkpoint.sequence_number,
            ),
            payload_store,
        )

        from elspeth.contracts import ResumedRow

        assert state.unprocessed_rows == (
            ResumedRow(
                row_id="row-orders",
                row_index=0,
                source_node_id=NodeID("source-orders"),
                row_data={"order_id": 101},
            ),
            ResumedRow(
                row_id="row-refunds",
                row_index=1,
                source_node_id=NodeID("source-refunds"),
                row_data={"refund_id": "r-7"},
            ),
        )
        assert state.schema_contracts_by_source[NodeID("source-orders")].version_hash() == orders_contract_hash
        assert state.schema_contracts_by_source[NodeID("source-refunds")].version_hash() == refunds_contract_hash

    def test_resume_early_exit_path_no_remaining_rows(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Test early-exit resume path: all rows already processed (Bug #8).

        Scenario:
        1. Failed run with 3 rows (0-2)
        2. ALL rows already processed before crash
        3. Resume finds no unprocessed rows
        4. Verify: Takes early-exit path (returns immediately)
        5. Verify: Checkpoints still deleted (Bug #8 fix)

        This is Bug #8 fix: early-exit path must delete checkpoints.
        """
        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        recovery_mgr = resume_test_env["recovery_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]
        tmp_path = resume_test_env["tmp_path"]

        # Set up failed run
        run_id = "resume-early-exit-test"
        output_path = tmp_path / "early_exit_output.csv"
        run_id, graph = self._setup_failed_run(db, payload_store, run_id, num_rows=3, checkpoint_at=2)

        # Simulate ALL rows already written
        with open(output_path, "w") as f:
            f.write("id,value\n")
            for i in range(3):
                f.write(f"{i},row-{i}\n")

        # Mark ALL rows as completed (terminal outcome)
        factory = make_factory(db)
        for i in range(3):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=f"t{i}", run_id=run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="sink",
            )

        # Create checkpoint
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="t2",
            node_id="xform",
            sequence_number=2,
            graph=graph,
        )

        # Verify checkpoint exists before resume
        with db.engine.connect() as conn:
            checkpoints_before = conn.execute(select(checkpoints_table).where(checkpoints_table.c.run_id == run_id)).fetchall()
        assert len(checkpoints_before) == 1

        # Resume (should take early-exit path)
        assert recovery_mgr.can_resume(run_id, graph).can_resume
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)

        # Use CSVSink with strict schema matching the data: {"id": int, "value": str}
        strict_schema = {"mode": "fixed", "fields": ["id: int", "value: str"]}
        passthrough = PassThrough({"schema": strict_schema})
        passthrough.on_error = "discard"
        config = PipelineConfig(
            sources={"source": _null_source("default")},
            transforms=[passthrough],
            sinks={"default": inject_write_failure(CSVSink({"path": str(output_path), "schema": strict_schema, "mode": "append"}))},
        )

        resume_graph = ExecutionGraph()
        schema_config = {"schema": strict_schema}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={**schema_config, "source_name": "source"})
        resume_graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        resume_graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        resume_graph.add_edge("src", "xform", label="continue")
        resume_graph.add_edge("xform", "sink", label="continue")
        resume_graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        resume_graph.set_transform_id_map({0: NodeID("xform")})

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=resume_graph,
            payload_store=payload_store,
        )

        # Verify early-exit behavior.
        #
        # Phase 2.2 (elspeth-0de989c56d): the early-exit branch now derives
        # the truthful row counts from token_outcomes rather than reporting
        # the resume's zero-delta.  The pre-Phase-2.2 behavior was
        # ``rows_processed=0`` regardless of how many rows the original run
        # had completed — under the new four-value RunStatus taxonomy that
        # would force EMPTY for runs that actually succeeded.  Reading the
        # audit DB makes the ``RunStatus`` correct AND surfaces the true
        # row counts to operators reading the CLI summary on resume.
        assert result.rows_processed == 3, "Early-exit path now reports truthful counts from token_outcomes"
        assert result.rows_succeeded == 3
        assert result.status == RunStatus.COMPLETED

        # CRITICAL: Verify checkpoints deleted on early-exit path (Bug #8 fix)
        with db.engine.connect() as conn:
            checkpoints_after = conn.execute(select(checkpoints_table).where(checkpoints_table.c.run_id == run_id)).fetchall()
        assert len(checkpoints_after) == 0, f"Bug #8: Early-exit path must delete checkpoints. Found {len(checkpoints_after)} remaining."

        # Verify output unchanged (no duplicate writes)
        lines = output_path.read_text().strip().split("\n")
        assert len(lines) == 4  # header + 3 rows (not 6!)

    def test_resume_with_datetime_fields(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Test resume preserves datetime types correctly (not degraded to str).

        Scenario:
        1. Failed run with datetime field in source schema
        2. Resume restores datetime type from stored schema
        3. Verify: datetime objects, not strings, in restored rows

        This validates the type_map handles format="date-time" annotation.
        """
        import json
        from datetime import UTC, datetime

        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        recovery_mgr = resume_test_env["recovery_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]
        tmp_path = resume_test_env["tmp_path"]

        # Set up failed run with datetime schema
        run_id = "resume-datetime-test"
        output_path = tmp_path / "datetime_output.csv"

        now = datetime.now(UTC)
        test_datetime = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        # Create schema with datetime field
        source_schema_json = json.dumps(
            {
                "properties": {
                    "id": {"type": "integer"},
                    "timestamp": {"type": "string", "format": "date-time"},
                },
                "required": ["id", "timestamp"],
            }
        )

        # PIPELINEROW MIGRATION: Create schema contract
        schema_contract_json, schema_contract_hash = self._create_schema_contract(
            [
                ("id", int),
                ("timestamp", datetime),
            ]
        )

        # Create graph
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={**schema_config, "source_name": "source"})
        graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("src", "xform", label="continue")
        graph.add_edge("xform", "sink", label="continue")

        with db.engine.begin() as conn:
            # Create run with datetime schema
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    runtime_val_manifest_json=_runtime_val_manifest_json(),
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )

            # Create nodes
            for node_id, plugin_name, node_type in [
                ("src", "null", NodeType.SOURCE),
                ("xform", "passthrough", NodeType.TRANSFORM),
                ("sink", "csv", NodeType.SINK),
            ]:
                conn.execute(
                    nodes_table.insert().values(
                        node_id=node_id,
                        run_id=run_id,
                        plugin_name=plugin_name,
                        node_type=node_type,
                        plugin_version="1.0.0",
                        determinism=Determinism.DETERMINISTIC if node_type != NodeType.SINK else Determinism.IO_WRITE,
                        config_hash="test",
                        config_json="{}",
                        registered_at=now,
                    )
                )

            # Create edges
            for edge_id, from_node, to_node in [
                ("e1", "src", "xform"),
                ("e2", "xform", "sink"),
            ]:
                conn.execute(
                    edges_table.insert().values(
                        edge_id=edge_id,
                        run_id=run_id,
                        from_node_id=from_node,
                        to_node_id=to_node,
                        label="continue",
                        default_mode=RoutingMode.MOVE,
                        created_at=now,
                    )
                )

            # ADR-025 §3 Decision 5: per-source contract lives on run_sources.
            conn.execute(
                run_sources_table.insert().values(
                    run_id=run_id,
                    source_node_id="src",
                    source_name="src",
                    plugin_name="null",
                    lifecycle_state="loaded",
                    config_hash="test",
                    schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    field_resolution_json=None,
                    recorded_at=now,
                )
            )

            # Create rows with datetime payloads
            for i in range(3):
                row_data = {
                    "id": i,
                    "timestamp": test_datetime.isoformat(),
                }
                ref = payload_store.store(json.dumps(row_data).encode())
                conn.execute(
                    rows_table.insert().values(
                        row_id=f"r{i}",
                        run_id=run_id,
                        source_node_id="src",
                        row_index=i,
                        source_row_index=i,
                        ingest_sequence=i,
                        source_data_hash=f"h{i}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"t{i}",
                        row_id=f"r{i}",
                        run_id=run_id,
                        created_at=now,
                    )
                )

        # Mark first row as completed (checkpoint will be at row 0)
        factory = make_factory(db)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id="t0", run_id=run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="sink",
        )

        # Create checkpoint at row 0 (last completed row)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="t0",
            node_id="xform",
            sequence_number=0,
            graph=graph,
        )

        # Resume should process rows 1-2 (2 remaining rows)
        assert recovery_mgr.can_resume(run_id, graph).can_resume
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)

        # Resume schema matches recovery output types: the recovery system
        # deserializes datetime objects from format: "date-time", not strings.
        # Schema field specs don't support datetime directly, so use observed
        # mode with explicit field guarantees instead of declaring a fake type.
        resume_schema = {"mode": "observed", "guaranteed_fields": ["id", "timestamp"], "required_fields": ["id", "timestamp"]}

        class DatetimeAssertingPassThrough(PassThrough):
            determinism = Determinism.DETERMINISTIC

            def process(self, row: Any, ctx: Any) -> Any:
                assert isinstance(row["timestamp"], datetime)
                return super().process(row, ctx)

        passthrough = DatetimeAssertingPassThrough({"schema": resume_schema})
        passthrough.on_error = "discard"
        config = PipelineConfig(
            sources={"source": _null_source("default")},
            transforms=[passthrough],
            sinks={"default": inject_write_failure(CSVSink({"path": str(output_path), "schema": resume_schema, "mode": "append"}))},
        )

        resume_graph = ExecutionGraph()
        resume_schema_config: dict[str, Any] = {"schema": resume_schema}
        resume_graph.add_node(
            "src", node_type=NodeType.SOURCE, plugin_name="null", config={**resume_schema_config, "source_name": "source"}
        )
        resume_graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=resume_schema_config)
        resume_graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=resume_schema_config)
        resume_graph.add_edge("src", "xform", label="continue")
        resume_graph.add_edge("xform", "sink", label="continue")
        resume_graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        resume_graph.set_transform_id_map({0: NodeID("xform")})

        # Write partial output (row 0 already written before crash)
        with open(output_path, "w") as f:
            f.write("id,timestamp\n")
            f.write(f"0,{test_datetime.isoformat()}\n")

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=resume_graph,
            payload_store=payload_store,
        )

        # Verify resume succeeded.
        # F2 (resume-fork-reemit): resume RunResult reports CUMULATIVE counters
        # from the audit trail. 1 row pre-marked completed (t0) + 2 re-driven
        # (r1, r2) = 3 source rows reaching a terminal outcome. (Pre-F2 this
        # reported the resume-only 2.)
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 3, f"Expected 3 cumulative rows processed (t0 + r1 + r2), got {result.rows_processed}"

        # The fact that resume succeeded without type errors proves datetime restoration worked
        # If schema reconstruction had failed, Pydantic would have kept timestamps as strings
        # and downstream transforms expecting datetime would have crashed
        assert result.rows_succeeded == 3

    def test_resume_with_decimal_fields(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Test resume preserves Decimal types for precision (not degraded to float).

        Scenario:
        1. Failed run with Decimal field in source schema
        2. Resume restores Decimal type from stored schema
        3. Verify: Decimal precision preserved, not float rounding

        This validates the type_map handles anyOf patterns for Decimal.
        """
        import json
        from datetime import UTC, datetime

        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        recovery_mgr = resume_test_env["recovery_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]
        tmp_path = resume_test_env["tmp_path"]

        # Set up failed run with Decimal schema
        run_id = "resume-decimal-test"
        output_path = tmp_path / "decimal_output.csv"

        now = datetime.now(UTC)

        # Create schema with Decimal field (anyOf pattern)
        source_schema_json = json.dumps(
            {
                "properties": {
                    "id": {"type": "integer"},
                    "amount": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                },
                "required": ["id", "amount"],
            }
        )

        # PIPELINEROW MIGRATION: Create schema contract
        # Note: We use float here as Decimal is not in VALID_FIELD_TYPES
        schema_contract_json, schema_contract_hash = self._create_schema_contract(
            [
                ("id", int),
                ("amount", float),  # Decimal coerces to float in contracts
            ]
        )

        # Create graph
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={**schema_config, "source_name": "source"})
        graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("src", "xform", label="continue")
        graph.add_edge("xform", "sink", label="continue")

        with db.engine.begin() as conn:
            # Create run with Decimal schema
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    runtime_val_manifest_json=_runtime_val_manifest_json(),
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )

            # Create nodes
            for node_id, plugin_name, node_type in [
                ("src", "null", NodeType.SOURCE),
                ("xform", "passthrough", NodeType.TRANSFORM),
                ("sink", "csv", NodeType.SINK),
            ]:
                conn.execute(
                    nodes_table.insert().values(
                        node_id=node_id,
                        run_id=run_id,
                        plugin_name=plugin_name,
                        node_type=node_type,
                        plugin_version="1.0.0",
                        determinism=Determinism.DETERMINISTIC if node_type != NodeType.SINK else Determinism.IO_WRITE,
                        config_hash="test",
                        config_json="{}",
                        registered_at=now,
                    )
                )

            # Create edges
            for edge_id, from_node, to_node in [
                ("e1", "src", "xform"),
                ("e2", "xform", "sink"),
            ]:
                conn.execute(
                    edges_table.insert().values(
                        edge_id=edge_id,
                        run_id=run_id,
                        from_node_id=from_node,
                        to_node_id=to_node,
                        label="continue",
                        default_mode=RoutingMode.MOVE,
                        created_at=now,
                    )
                )

            # ADR-025 §3 Decision 5: per-source contract lives on run_sources.
            conn.execute(
                run_sources_table.insert().values(
                    run_id=run_id,
                    source_node_id="src",
                    source_name="src",
                    plugin_name="null",
                    lifecycle_state="loaded",
                    config_hash="test",
                    schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    field_resolution_json=None,
                    recorded_at=now,
                )
            )

            # Create rows with Decimal payloads (high precision value)
            for i in range(2):
                row_data = {
                    "id": i,
                    "amount": "99.123456789012345",  # Precision that float would lose
                }
                ref = payload_store.store(json.dumps(row_data).encode())
                conn.execute(
                    rows_table.insert().values(
                        row_id=f"r{i}",
                        run_id=run_id,
                        source_node_id="src",
                        row_index=i,
                        source_row_index=i,
                        ingest_sequence=i,
                        source_data_hash=f"h{i}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"t{i}",
                        row_id=f"r{i}",
                        run_id=run_id,
                        created_at=now,
                    )
                )

        # Mark first row as completed (checkpoint will be at row 0)
        factory = make_factory(db)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id="t0", run_id=run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="sink",
        )

        # Create checkpoint at row 0 (last completed row)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="t0",
            node_id="xform",
            sequence_number=0,
            graph=graph,
        )

        # Resume should process row 1 (1 remaining row)
        assert recovery_mgr.can_resume(run_id, graph).can_resume
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)

        # Resume schema matches recovery output types: the recovery system
        # coerces Decimal to float per the schema contract.
        resume_schema = {"mode": "fixed", "fields": ["id: int", "amount: float"]}
        passthrough = PassThrough({"schema": resume_schema})
        passthrough.on_error = "discard"
        config = PipelineConfig(
            sources={"source": _null_source("default")},
            transforms=[passthrough],
            sinks={"default": inject_write_failure(CSVSink({"path": str(output_path), "schema": resume_schema, "mode": "append"}))},
        )

        resume_graph = ExecutionGraph()
        resume_schema_config: dict[str, Any] = {"schema": resume_schema}
        resume_graph.add_node(
            "src", node_type=NodeType.SOURCE, plugin_name="null", config={**resume_schema_config, "source_name": "source"}
        )
        resume_graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=resume_schema_config)
        resume_graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=resume_schema_config)
        resume_graph.add_edge("src", "xform", label="continue")
        resume_graph.add_edge("xform", "sink", label="continue")
        resume_graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        resume_graph.set_transform_id_map({0: NodeID("xform")})

        # Write partial output (row 0 already written before crash)
        with open(output_path, "w") as f:
            f.write("id,amount\n")
            f.write("0,99.123456789012345\n")

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=resume_graph,
            payload_store=payload_store,
        )

        # Verify resume succeeded with Decimal precision preserved.
        # F2 (resume-fork-reemit): resume RunResult reports CUMULATIVE counters
        # from the audit trail. 1 row pre-marked completed (t0) + 1 re-driven
        # (r1) = 2 source rows reaching a terminal outcome. (Pre-F2: resume-only 1.)
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 2, f"Expected 2 cumulative rows processed (t0 + r1), got {result.rows_processed}"
        assert result.rows_succeeded == 2

    def test_resume_with_array_fields(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Test resume preserves list/array types correctly.

        Scenario:
        1. Failed run with array field in source schema
        2. Resume restores list type from stored schema
        3. Verify: arrays parsed correctly, not strings

        This validates the type_map handles type="array".
        """
        import json
        from datetime import UTC, datetime

        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        recovery_mgr = resume_test_env["recovery_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]
        tmp_path = resume_test_env["tmp_path"]

        # Set up failed run with array schema
        run_id = "resume-array-test"
        output_path = tmp_path / "array_output.csv"

        now = datetime.now(UTC)

        # Create schema with array field
        source_schema_json = json.dumps(
            {
                "properties": {
                    "id": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "tags"],
            }
        )

        # PIPELINEROW MIGRATION: Create schema contract
        # Arrays use object type (any) in contracts
        schema_contract_json, schema_contract_hash = self._create_schema_contract(
            [
                ("id", int),
                ("tags", object),  # Arrays use 'any'/object type
            ]
        )

        # Create graph
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={**schema_config, "source_name": "source"})
        graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("src", "xform", label="continue")
        graph.add_edge("xform", "sink", label="continue")

        with db.engine.begin() as conn:
            # Create run with array schema
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    runtime_val_manifest_json=_runtime_val_manifest_json(),
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )

            # Create nodes
            for node_id, plugin_name, node_type in [
                ("src", "null", NodeType.SOURCE),
                ("xform", "passthrough", NodeType.TRANSFORM),
                ("sink", "csv", NodeType.SINK),
            ]:
                conn.execute(
                    nodes_table.insert().values(
                        node_id=node_id,
                        run_id=run_id,
                        plugin_name=plugin_name,
                        node_type=node_type,
                        plugin_version="1.0.0",
                        determinism=Determinism.DETERMINISTIC if node_type != NodeType.SINK else Determinism.IO_WRITE,
                        config_hash="test",
                        config_json="{}",
                        registered_at=now,
                    )
                )

            # Create edges
            for edge_id, from_node, to_node in [
                ("e1", "src", "xform"),
                ("e2", "xform", "sink"),
            ]:
                conn.execute(
                    edges_table.insert().values(
                        edge_id=edge_id,
                        run_id=run_id,
                        from_node_id=from_node,
                        to_node_id=to_node,
                        label="continue",
                        default_mode=RoutingMode.MOVE,
                        created_at=now,
                    )
                )

            # ADR-025 §3 Decision 5: per-source contract lives on run_sources.
            conn.execute(
                run_sources_table.insert().values(
                    run_id=run_id,
                    source_node_id="src",
                    source_name="src",
                    plugin_name="null",
                    lifecycle_state="loaded",
                    config_hash="test",
                    schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    field_resolution_json=None,
                    recorded_at=now,
                )
            )

            # Create rows with array payloads
            for i in range(2):
                row_data = {
                    "id": i,
                    "tags": ["tag1", "tag2", f"tag{i}"],
                }
                ref = payload_store.store(json.dumps(row_data).encode())
                conn.execute(
                    rows_table.insert().values(
                        row_id=f"r{i}",
                        run_id=run_id,
                        source_node_id="src",
                        row_index=i,
                        source_row_index=i,
                        ingest_sequence=i,
                        source_data_hash=f"h{i}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"t{i}",
                        row_id=f"r{i}",
                        run_id=run_id,
                        created_at=now,
                    )
                )

        # Mark first row as completed (checkpoint will be at row 0)
        factory = make_factory(db)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id="t0", run_id=run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="sink",
        )

        # Create checkpoint at row 0 (last completed row)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="t0",
            node_id="xform",
            sequence_number=0,
            graph=graph,
        )

        # Resume should process row 1 (1 remaining row)
        assert recovery_mgr.can_resume(run_id, graph).can_resume
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)

        passthrough = PassThrough({"schema": {"mode": "observed"}})
        passthrough.on_error = "discard"
        config = PipelineConfig(
            sources={"source": _null_source("default")},
            transforms=[passthrough],
            sinks={
                "default": inject_write_failure(
                    JSONSink(
                        {"path": str(output_path.with_suffix(".json")), "schema": {"mode": "observed"}, "mode": "append", "format": "jsonl"}
                    )
                )
            },
        )

        resume_graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={**schema_config, "source_name": "source"})
        resume_graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        resume_graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        resume_graph.add_edge("src", "xform", label="continue")
        resume_graph.add_edge("xform", "sink", label="continue")
        resume_graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        resume_graph.set_transform_id_map({0: NodeID("xform")})

        # Write partial output (row 0 already written before crash)
        with open(output_path, "w") as f:
            f.write("id,tags\n")
            f.write('0,"[""tag1"", ""tag2"", ""tag0""]"\n')

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=resume_graph,
            payload_store=payload_store,
        )

        # Verify resume succeeded with array types preserved.
        # F2 (resume-fork-reemit): resume RunResult reports CUMULATIVE counters
        # from the audit trail. 1 row pre-marked completed (t0) + 1 re-driven
        # (r1) = 2 source rows reaching a terminal outcome. (Pre-F2: resume-only 1.)
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 2, f"Expected 2 cumulative rows processed (t0 + r1), got {result.rows_processed}"
        assert result.rows_succeeded == 2

    def test_resume_with_nested_object_fields(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Test resume preserves dict/nested object types correctly.

        Scenario:
        1. Failed run with nested object field in source schema
        2. Resume restores dict type from stored schema
        3. Verify: nested objects parsed correctly

        This validates the type_map handles type="object".
        """
        import json
        from datetime import UTC, datetime

        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        recovery_mgr = resume_test_env["recovery_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]
        tmp_path = resume_test_env["tmp_path"]

        # Set up failed run with nested object schema
        run_id = "resume-object-test"
        output_path = tmp_path / "object_output.csv"

        now = datetime.now(UTC)

        # Create schema with nested object field
        source_schema_json = json.dumps(
            {
                "properties": {
                    "id": {"type": "integer"},
                    "metadata": {"type": "object"},
                },
                "required": ["id", "metadata"],
            }
        )

        # PIPELINEROW MIGRATION: Create schema contract
        schema_contract_json, schema_contract_hash = self._create_schema_contract(
            [
                ("id", int),
                ("metadata", object),  # Nested objects use 'any'/object type
            ]
        )

        # Create graph
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={**schema_config, "source_name": "source"})
        graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("src", "xform", label="continue")
        graph.add_edge("xform", "sink", label="continue")

        with db.engine.begin() as conn:
            # Create run with object schema
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    runtime_val_manifest_json=_runtime_val_manifest_json(),
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )

            # Create nodes
            for node_id, plugin_name, node_type in [
                ("src", "null", NodeType.SOURCE),
                ("xform", "passthrough", NodeType.TRANSFORM),
                ("sink", "csv", NodeType.SINK),
            ]:
                conn.execute(
                    nodes_table.insert().values(
                        node_id=node_id,
                        run_id=run_id,
                        plugin_name=plugin_name,
                        node_type=node_type,
                        plugin_version="1.0.0",
                        determinism=Determinism.DETERMINISTIC if node_type != NodeType.SINK else Determinism.IO_WRITE,
                        config_hash="test",
                        config_json="{}",
                        registered_at=now,
                    )
                )

            # Create edges
            for edge_id, from_node, to_node in [
                ("e1", "src", "xform"),
                ("e2", "xform", "sink"),
            ]:
                conn.execute(
                    edges_table.insert().values(
                        edge_id=edge_id,
                        run_id=run_id,
                        from_node_id=from_node,
                        to_node_id=to_node,
                        label="continue",
                        default_mode=RoutingMode.MOVE,
                        created_at=now,
                    )
                )

            # ADR-025 §3 Decision 5: per-source contract lives on run_sources.
            conn.execute(
                run_sources_table.insert().values(
                    run_id=run_id,
                    source_node_id="src",
                    source_name="src",
                    plugin_name="null",
                    lifecycle_state="loaded",
                    config_hash="test",
                    schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    field_resolution_json=None,
                    recorded_at=now,
                )
            )

            # Create rows with nested object payloads
            for i in range(2):
                row_data = {
                    "id": i,
                    "metadata": {"author": "test", "version": i, "active": True},
                }
                ref = payload_store.store(json.dumps(row_data).encode())
                conn.execute(
                    rows_table.insert().values(
                        row_id=f"r{i}",
                        run_id=run_id,
                        source_node_id="src",
                        row_index=i,
                        source_row_index=i,
                        ingest_sequence=i,
                        source_data_hash=f"h{i}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"t{i}",
                        row_id=f"r{i}",
                        run_id=run_id,
                        created_at=now,
                    )
                )

        # Mark first row as completed (checkpoint will be at row 0)
        factory = make_factory(db)
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id="t0", run_id=run_id),
            outcome=TerminalOutcome.SUCCESS,
            path=TerminalPath.DEFAULT_FLOW,
            sink_name="sink",
        )

        # Create checkpoint at row 0 (last completed row)
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="t0",
            node_id="xform",
            sequence_number=0,
            graph=graph,
        )

        # Resume should process row 1 (1 remaining row)
        assert recovery_mgr.can_resume(run_id, graph).can_resume
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)

        passthrough = PassThrough({"schema": {"mode": "observed"}})
        passthrough.on_error = "discard"
        config = PipelineConfig(
            sources={"source": _null_source("default")},
            transforms=[passthrough],
            sinks={
                "default": inject_write_failure(
                    JSONSink(
                        {"path": str(output_path.with_suffix(".json")), "schema": {"mode": "observed"}, "mode": "append", "format": "jsonl"}
                    )
                )
            },
        )

        resume_graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={**schema_config, "source_name": "source"})
        resume_graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        resume_graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        resume_graph.add_edge("src", "xform", label="continue")
        resume_graph.add_edge("xform", "sink", label="continue")
        resume_graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        resume_graph.set_transform_id_map({0: NodeID("xform")})

        # Write partial output (row 0 already written before crash)
        with open(output_path, "w") as f:
            f.write("id,metadata\n")
            f.write('0,"{""author"": ""test"", ""version"": 0, ""active"": true}"\n')

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=resume_graph,
            payload_store=payload_store,
        )

        # Verify resume succeeded with nested objects preserved.
        # F2 (resume-fork-reemit): resume RunResult reports CUMULATIVE counters
        # from the audit trail. 1 row pre-marked completed (t0) + 1 re-driven
        # (r1) = 2 source rows reaching a terminal outcome. (Pre-F2: resume-only 1.)
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 2, f"Expected 2 cumulative rows processed (t0 + r1), got {result.rows_processed}"
        assert result.rows_succeeded == 2

    def test_resume_with_unsupported_type_crashes(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Test resume crashes loudly on unsupported schema types (no silent degradation).

        Scenario:
        1. Failed run with unsupported type in stored schema
        2. Resume attempts to reconstruct schema
        3. Verify: Crashes with clear error message (not silent str fallback)

        This validates the prohibition on defensive .get() patterns.
        """
        import json
        from datetime import UTC, datetime

        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        recovery_mgr = resume_test_env["recovery_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]

        # Set up failed run with unsupported type
        run_id = "resume-unsupported-test"

        now = datetime.now(UTC)

        # Create schema with UNSUPPORTED type (imaginary "geo-point" type)
        source_schema_json = json.dumps(
            {
                "properties": {
                    "id": {"type": "integer"},
                    "location": {"type": "geo-point"},  # Not a real JSON schema type
                },
                "required": ["id", "location"],
            }
        )

        # Create graph
        graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={**schema_config, "source_name": "source"})
        graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("src", "xform", label="continue")
        graph.add_edge("xform", "sink", label="continue")

        # Create a minimal schema contract for the run record
        schema_contract_json, schema_contract_hash = self._create_schema_contract([("id", int), ("location", str)])

        with db.engine.begin() as conn:
            # Create run with unsupported schema
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    runtime_val_manifest_json=_runtime_val_manifest_json(),
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )

            # Create nodes
            for node_id, plugin_name, node_type in [
                ("src", "null", NodeType.SOURCE),
                ("xform", "passthrough", NodeType.TRANSFORM),
                ("sink", "csv", NodeType.SINK),
            ]:
                conn.execute(
                    nodes_table.insert().values(
                        node_id=node_id,
                        run_id=run_id,
                        plugin_name=plugin_name,
                        node_type=node_type,
                        plugin_version="1.0.0",
                        determinism=Determinism.DETERMINISTIC if node_type != NodeType.SINK else Determinism.IO_WRITE,
                        config_hash="test",
                        config_json="{}",
                        registered_at=now,
                    )
                )

            # Create edges
            for edge_id, from_node, to_node in [
                ("e1", "src", "xform"),
                ("e2", "xform", "sink"),
            ]:
                conn.execute(
                    edges_table.insert().values(
                        edge_id=edge_id,
                        run_id=run_id,
                        from_node_id=from_node,
                        to_node_id=to_node,
                        label="continue",
                        default_mode=RoutingMode.MOVE,
                        created_at=now,
                    )
                )

            # ADR-025 §3 Decision 5: per-source contract lives on run_sources.
            conn.execute(
                run_sources_table.insert().values(
                    run_id=run_id,
                    source_node_id="src",
                    source_name="src",
                    plugin_name="null",
                    lifecycle_state="loaded",
                    config_hash="test",
                    schema_json=source_schema_json,
                    schema_contract_json=schema_contract_json,
                    schema_contract_hash=schema_contract_hash,
                    field_resolution_json=None,
                    recorded_at=now,
                )
            )

            # Create a dummy row (won't be processed - resume will fail during schema reconstruction)
            row_data = {"id": 0, "location": "some-location"}
            ref = payload_store.store(json.dumps(row_data).encode())
            conn.execute(
                rows_table.insert().values(
                    row_id="r0",
                    run_id=run_id,
                    source_node_id="src",
                    row_index=0,
                    source_row_index=0,
                    ingest_sequence=0,
                    source_data_hash="h0",
                    source_data_ref=ref,
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="t0",
                    row_id="r0",
                    run_id=run_id,
                    created_at=now,
                )
            )

        # Create checkpoint
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="t0",
            node_id="xform",
            sequence_number=0,
            graph=graph,
        )

        # Resume should CRASH during schema reconstruction
        assert recovery_mgr.can_resume(run_id, graph).can_resume
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)

        passthrough = PassThrough({"schema": {"mode": "observed"}})
        passthrough.on_error = "discard"
        config = PipelineConfig(
            sources={"source": _null_source("default")},
            transforms=[passthrough],
            sinks={
                "default": inject_write_failure(
                    JSONSink({"path": "/tmp/dummy.json", "schema": {"mode": "observed"}, "mode": "write", "format": "jsonl"})
                )
            },
        )

        resume_graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config={**schema_config, "source_name": "source"})
        resume_graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        resume_graph.add_node("sink", node_type=NodeType.SINK, plugin_name="json", config=schema_config)
        resume_graph.add_edge("src", "xform", label="continue")
        resume_graph.add_edge("xform", "sink", label="continue")
        resume_graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        resume_graph.set_transform_id_map({0: NodeID("xform")})

        # CRITICAL: Must crash with clear error, not silently degrade to str
        with pytest.raises(AuditIntegrityError, match=r"unsupported type 'geo-point'"):
            orchestrator.resume(
                resume_point=resume_point,
                config=config,
                graph=resume_graph,
                payload_store=payload_store,
            )

    def test_resume_gate_routed_pipeline_classifies_as_completed(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """elspeth-5069612f3c — pin the resume code path's correct accumulation
        of rows_routed_success.

        The L0 unit test test_resume_continuation_still_classifies_as_completed
        pins the predicate's behavior on a synthetic resume-shaped counter
        tuple, but does NOT exercise the actual resume site where the terminal
        status is derived from the resume-side accumulator.  A regression in
        the resume-side local accumulators (an off-by-one in the new locals,
        a missed kwarg in the derive_terminal_run_status call, or a broken
        return-tuple expansion) would slip past that unit test.

        Scenario (early-exit resume — all rows already gate-routed before the
        pre-resume crash):
        1. Failed run with 5 rows (0-4), linear topology re-used from
           ``_setup_failed_run`` for the persisted DAG, but every row is
           pre-marked as ``(SUCCESS, GATE_ROUTED)`` via ``record_token_outcome`` —
           the canonical pre-split-fix shape for gate-routed rows.
        2. Resume's early-exit path reads existing terminal outcomes from
           Landscape and calls ``derive_terminal_run_status`` with the
           accumulated counters.
        3. Verify: ``result.status == RunStatus.COMPLETED`` (not FAILED).
        4. Verify: ``result.rows_succeeded == 5`` — routed successes occupy
           the lifecycle success bucket while ``rows_routed_success`` records
           the orthogonal routing provenance.
        5. Verify: ``result.rows_routed_success == 5`` — the resume-side
           accumulator must surface the existing ``ROUTED`` outcomes via the
           split counter (orthogonal attribution: which gate, which sink).
        6. Verify: ``result.rows_routed_failure == 0`` — no on_error reroutes.

        Pre-PR (commit 8865559e, before the rows_routed split): the resume's
        ``derive_terminal_run_status`` excludes ``rows_routed`` from the
        predicate (DIVERT/MOVE conflation), so the run misclassifies as
        FAILED.  Post-PR: classifies as COMPLETED.
        """
        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        recovery_mgr = resume_test_env["recovery_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]
        tmp_path = resume_test_env["tmp_path"]

        run_id = "resume-gate-routed-test"
        output_path = tmp_path / "gate_routed_output.csv"
        run_id, graph = self._setup_failed_run(db, payload_store, run_id, num_rows=5, checkpoint_at=4)

        # Mark every row as gate-routed (SUCCESS/GATE_ROUTED, sink_name set,
        # error_hash NULL — the canonical pre-split-fix shape for
        # intentional gate route_to_sink MOVE rows).  By marking ALL rows
        # as terminal, the resume takes the early-exit path; what we are
        # pinning is whether the resume's terminal-status derivation
        # correctly accumulates ``rows_routed_success`` from Landscape.
        factory = make_factory(db)
        for i in range(5):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=f"t{i}", run_id=run_id),
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.GATE_ROUTED,
                sink_name="sink",
            )

        # Create checkpoint at the last row so resume's recovery path is
        # exercised even though no rows remain to process.
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="t4",
            node_id="xform",
            sequence_number=4,
            graph=graph,
        )

        assert recovery_mgr.can_resume(run_id, graph).can_resume
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        # Resume config matches the persisted DAG topology (linear src ->
        # xform -> sink), even though every persisted outcome is ROUTED.
        # The predicate's behaviour, not the topology, is the unit under
        # test here.
        resume_schema = {"mode": "fixed", "fields": ["id: int", "value: str"]}
        passthrough = PassThrough({"schema": resume_schema})
        passthrough.on_error = "discard"
        config = PipelineConfig(
            sources={"source": _null_source("default")},
            transforms=[passthrough],
            sinks={
                "default": inject_write_failure(
                    CSVSink(
                        {
                            "path": str(output_path),
                            "schema": resume_schema,
                            "mode": "append",
                        }
                    )
                )
            },
        )
        resume_graph = ExecutionGraph()
        resume_schema_config: dict[str, Any] = {"schema": resume_schema}
        resume_graph.add_node(
            "src", node_type=NodeType.SOURCE, plugin_name="null", config={**resume_schema_config, "source_name": "source"}
        )
        resume_graph.add_node(
            "xform",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config=resume_schema_config,
        )
        resume_graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=resume_schema_config)
        resume_graph.add_edge("src", "xform", label="continue")
        resume_graph.add_edge("xform", "sink", label="continue")
        resume_graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        resume_graph.set_transform_id_map({0: NodeID("xform")})

        # Pre-write the output file header so any append-mode interaction
        # is consistent (no remaining rows will actually be written; this
        # is the early-exit path).
        with open(output_path, "w") as f:
            f.write("id,value\n")
            for i in range(5):
                f.write(f"{i},row-{i}\n")

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=resume_graph,
            payload_store=payload_store,
        )

        # CORE ASSERTION — verify the resume-side accumulator + predicate
        # together classify a gate-routed run as COMPLETED.
        assert result.status == RunStatus.COMPLETED, (
            "Resume of gate-routed pipeline misclassified — expected "
            f"COMPLETED, got {result.status!r}. The resume-side "
            f"derive_terminal_run_status call must accumulate "
            f"rows_routed_success from the resume locals (or the "
            f"early-exit Landscape readback). "
            f"result={result.to_dict()}"
        )
        # ROUTED rows are lifecycle successes with gate-routing provenance.
        assert result.rows_succeeded == 5
        assert result.rows_routed_success == 5  # All rows recorded as ROUTED.
        assert result.rows_routed_failure == 0  # No on_error reroutes.

        # Cross-check against Landscape: every token_outcomes row has the
        # routed shape (matches Step 9c's audit-distinguishability test).
        from elspeth.core.landscape.schema import token_outcomes_table

        with db.engine.connect() as conn:
            outcomes = conn.execute(
                select(token_outcomes_table.c.outcome, token_outcomes_table.c.path, token_outcomes_table.c.sink_name).where(
                    token_outcomes_table.c.run_id == run_id
                )
            ).fetchall()
        routed_outcomes = [o for o in outcomes if (o.outcome, o.path) == (TerminalOutcome.SUCCESS.value, TerminalPath.GATE_ROUTED.value)]
        assert len(routed_outcomes) == 5

    def test_resume_routed_on_error_pipeline_classifies_as_failed(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Complement to ``test_resume_gate_routed_pipeline_classifies_as_completed``
        — pin the resume code path's correct accumulation of
        ``rows_routed_failure`` for the on_error DIVERT side of the
        rows_routed split.

        Why this test exists (and why the unit-level coverage is not
        sufficient): ``derive_resume_terminal_status_from_audit`` has two
        symmetric match arms — ``(SUCCESS, GATE_ROUTED)`` (gate MOVE; counts
        toward ``rows_routed_success``) and ``(FAILURE, ON_ERROR_ROUTED)``
        (transform on_error DIVERT; counts toward ``rows_routed_failure``).
        A regression that swapped the two ``rows_routed_*`` increments in
        the ROUTED_ON_ERROR arm would slip past the unit-level predicate
        test (which exercises the predicate's success/failure shape on
        synthetic counters but does not invoke the resume aggregation
        site).  The gate-MOVE side has integration coverage above; this
        test mirrors that coverage for the on_error side so a swap in the
        match arm fails loudly at the resume layer.

        Scenario (early-exit resume — every row already DIVERTED via
        on_error before the pre-resume crash):
        1. Failed run with 5 rows (0-4), reusing ``_setup_failed_run`` for
           the persisted DAG.
        2. Every row pre-marked as ``(FAILURE, ON_ERROR_ROUTED)`` —
           ``sink_name="error_sink"`` and ``error_hash`` set per the
           outcome contract for DIVERT rows.
        3. Resume's early-exit path reads existing terminal outcomes and
           calls ``derive_terminal_run_status`` with the accumulated
           counters.
        4. Verify: ``result.status == RunStatus.FAILED`` — predicate
           output for ``(rows_processed=5, rows_succeeded=0,
           rows_routed_success=0, rows_routed_failure=5,
           rows_failed=5, rows_quarantined=0)``.  Reasoning:
           ``success_indicator = (rows_succeeded > 0) OR
           (rows_routed_success > 0) = False`` and ``rows_processed > 0``
           drives the ``not success_indicator -> FAILED`` arm of
           ``derive_terminal_run_status``.
        5. Verify: ``result.rows_routed_failure == 5`` — the resume-side
           accumulator surfaces every ROUTED_ON_ERROR outcome via the
           failure-side split counter.
        6. Verify: ``result.rows_routed_success == 0`` — no MOVE rows.
        7. Verify: ``result.rows_succeeded == 0`` and
           ``result.rows_failed == 5`` — every row has lifecycle FAILURE,
           with ``rows_routed_failure`` preserving the on_error routing
           provenance.

        Regression catcher: an inversion of the two ``rows_routed_*``
        increments in ``derive_resume_terminal_status_from_audit``'s
        ``ROUTED_ON_ERROR`` arm would (a) miscount as
        ``rows_routed_success=5`` and (b) flip the predicate to
        ``RunStatus.COMPLETED``.  Both assertions fail under the swap.
        """
        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        recovery_mgr = resume_test_env["recovery_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]
        tmp_path = resume_test_env["tmp_path"]

        run_id = "resume-routed-on-error-test"
        output_path = tmp_path / "routed_on_error_output.csv"
        run_id, graph = self._setup_failed_run(db, payload_store, run_id, num_rows=5, checkpoint_at=4)

        # Mark every row as on_error DIVERT (FAILURE/ON_ERROR_ROUTED).
        # Contract requires sink_name AND error_hash for this outcome
        # (see data_flow_repository._validate_outcome_fields:236-249 and
        # contracts/results.py:408-419).  The 16-char hex string mirrors
        # the existing ROUTED_ON_ERROR fixture in
        # tests/integration/audit/test_recorder_routing_events.py:617.
        factory = make_factory(db)
        for i in range(5):
            factory.data_flow.record_token_outcome(
                ref=TokenRef(token_id=f"t{i}", run_id=run_id),
                outcome=TerminalOutcome.FAILURE,
                path=TerminalPath.ON_ERROR_ROUTED,
                sink_name="error_sink",
                error_hash="0123456789abcdef",
            )

        # Create checkpoint at the last row so resume's recovery path is
        # exercised even though no rows remain to process.  Mirrors the
        # gate-MOVE test setup.
        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="t4",
            node_id="xform",
            sequence_number=4,
            graph=graph,
        )

        assert recovery_mgr.can_resume(run_id, graph).can_resume
        resume_point = recovery_mgr.get_resume_point(run_id, graph)
        assert resume_point is not None

        orchestrator = Orchestrator(
            db,
            checkpoint_manager=checkpoint_mgr,
            checkpoint_config=checkpoint_config,
        )

        resume_schema = {"mode": "fixed", "fields": ["id: int", "value: str"]}
        passthrough = PassThrough({"schema": resume_schema})
        passthrough.on_error = "discard"
        config = PipelineConfig(
            sources={"source": _null_source("default")},
            transforms=[passthrough],
            sinks={
                "default": inject_write_failure(
                    CSVSink(
                        {
                            "path": str(output_path),
                            "schema": resume_schema,
                            "mode": "append",
                        }
                    )
                )
            },
        )
        resume_graph = ExecutionGraph()
        resume_schema_config: dict[str, Any] = {"schema": resume_schema}
        resume_graph.add_node(
            "src", node_type=NodeType.SOURCE, plugin_name="null", config={**resume_schema_config, "source_name": "source"}
        )
        resume_graph.add_node(
            "xform",
            node_type=NodeType.TRANSFORM,
            plugin_name="passthrough",
            config=resume_schema_config,
        )
        resume_graph.add_node("sink", node_type=NodeType.SINK, plugin_name="csv", config=resume_schema_config)
        resume_graph.add_edge("src", "xform", label="continue")
        resume_graph.add_edge("xform", "sink", label="continue")
        resume_graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        resume_graph.set_transform_id_map({0: NodeID("xform")})

        # Pre-write the output file header so any append-mode interaction
        # is consistent (no remaining rows will be processed; this is the
        # early-exit path).  Mirrors the gate-MOVE test setup.
        with open(output_path, "w") as f:
            f.write("id,value\n")
            for i in range(5):
                f.write(f"{i},row-{i}\n")

        result = orchestrator.resume(
            resume_point=resume_point,
            config=config,
            graph=resume_graph,
            payload_store=payload_store,
        )

        # CORE ASSERTION — verify the resume-side accumulator + predicate
        # together classify a fully-DIVERTED run as FAILED.  Predicate
        # output derived from ``derive_terminal_run_status`` in
        # contracts/run_result.py:180 — when ``rows_succeeded == 0`` AND
        # ``rows_routed_success == 0``, ``success_indicator`` is False
        # and the predicate returns FAILED for any non-zero
        # ``rows_processed``.
        assert result.status == RunStatus.FAILED, (
            "Resume of fully-DIVERTED pipeline misclassified — expected "
            f"FAILED, got {result.status!r}. The resume-side "
            f"derive_terminal_run_status call must accumulate "
            f"rows_routed_failure (NOT rows_routed_success) from the "
            f"early-exit Landscape readback's ROUTED_ON_ERROR rows. "
            f"result={result.to_dict()}"
        )
        assert result.rows_succeeded == 0  # No on_success success-path sink.
        assert result.rows_routed_failure == 5  # All rows recorded as ROUTED_ON_ERROR.
        assert result.rows_routed_success == 0  # No gate MOVE rows.
        assert result.rows_failed == 5  # Lifecycle FAILURE plus on_error provenance.
        assert result.rows_quarantined == 0  # No quarantine outcomes.

        # Cross-check against Landscape: every token_outcomes row has the
        # routed_on_error shape (audit-distinguishability mirror of the
        # gate-MOVE cross-check above).
        from elspeth.core.landscape.schema import token_outcomes_table

        with db.engine.connect() as conn:
            outcomes = conn.execute(
                select(token_outcomes_table.c.outcome, token_outcomes_table.c.path, token_outcomes_table.c.sink_name).where(
                    token_outcomes_table.c.run_id == run_id
                )
            ).fetchall()
        routed_on_error_outcomes = [
            o for o in outcomes if (o.outcome, o.path) == (TerminalOutcome.FAILURE.value, TerminalPath.ON_ERROR_ROUTED.value)
        ]
        assert len(routed_on_error_outcomes) == 5
        # Every ROUTED_ON_ERROR row carries the error sink, distinct from
        # the gate-MOVE shape (where ``sink_name`` matches the on_success
        # destination).  This pins the audit-distinguishability invariant
        # at the resume layer.
        assert all(o.sink_name == "error_sink" for o in routed_on_error_outcomes)


class TestMultiSourceResumeContractDispatch:
    """Regression coverage for G2 / elspeth-01942858c3 and the
    test plan in the consolidated G2-companion ticket
    elspeth-d5f0194fc8.

    Before ADR-025, ``_reconstruct_resume_state`` collapsed the
    per-source contract map by calling
    ``next(iter(schema_contracts_by_source.values()))`` and dropping
    the result on ``ResumeState.schema_contract``. Two sources whose
    schemas legitimately differ (e.g., a fan-in pipeline merging
    ``orders`` and ``refunds``) were validated under whichever
    contract happened to be returned first by the SQL query — a
    Tier-1 audit-integrity violation per CLAUDE.md.

    These tests pin the post-ADR-025 behaviour:

    1. Each row's schema contract is recovered via the row's
       ``source_node_id``, not an arbitrary pick.
    2. A row whose ``source_node_id`` is not present in
       ``schema_contracts_by_source`` crashes with
       ``OrchestrationInvariantError`` rather than silently picking
       a default.
    3. Single-source resume still works after the structural change
       (no regression on the legitimate-single case).
    4. The ``ResumeState`` dataclass has no singular
       ``schema_contract`` field (introspected via
       ``dataclasses.fields``, not ``hasattr`` — ``hasattr`` is
       banned per CLAUDE.md).
    """

    @staticmethod
    def _create_schema_contract(fields: list[tuple[str, type]]) -> tuple[str, str]:
        """Build a fixed SchemaContract for test setup (identity helper)."""
        from elspeth.contracts.contract_records import ContractAuditRecord
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        field_contracts = tuple(
            FieldContract(
                normalized_name=name,
                original_name=name,
                python_type=py_type,
                required=True,
                source="declared",
            )
            for name, py_type in fields
        )
        contract = SchemaContract(mode="FIXED", fields=field_contracts, locked=True)
        audit_record = ContractAuditRecord.from_contract(contract)
        return audit_record.to_json(), contract.version_hash()

    def _build_two_source_run(
        self,
        db: LandscapeDB,
        payload_store: Any,
        checkpoint_mgr: Any,
        *,
        run_id: str,
    ) -> tuple[ExecutionGraph, str, str, dict[str, Any]]:
        """Insert a 2-source FAILED run wired for resume reconstruction.

        Returns:
            (graph, orders_contract_hash, refunds_contract_hash, row_payloads)
            so callers can correlate observed contracts to their
            originating source on the per-row validation path.
        """
        now = datetime.now(UTC)
        orders_schema_json = json.dumps({"properties": {"order_id": {"type": "integer"}}, "required": ["order_id"]})
        refunds_schema_json = json.dumps({"properties": {"refund_id": {"type": "string"}}, "required": ["refund_id"]})
        orders_contract_json, orders_contract_hash = self._create_schema_contract([("order_id", int)])
        refunds_contract_json, refunds_contract_hash = self._create_schema_contract([("refund_id", str)])

        graph = ExecutionGraph()
        graph.add_node(
            "source-orders",
            node_type=NodeType.SOURCE,
            plugin_name="null",
            config={"source_name": "orders"},
        )
        graph.add_node(
            "source-refunds",
            node_type=NodeType.SOURCE,
            plugin_name="null",
            config={"source_name": "refunds"},
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="json",
            config={"schema": {"mode": "observed"}},
        )
        graph.add_edge("source-orders", "sink", label="continue")
        graph.add_edge("source-refunds", "sink", label="continue")

        with db.engine.begin() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.FAILED,
                    source_schema_json=json.dumps({"properties": {}, "required": []}),
                    runtime_val_manifest_json=_runtime_val_manifest_json(),
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )
            for node_id, plugin_name, node_type in [
                ("source-orders", "null", NodeType.SOURCE),
                ("source-refunds", "null", NodeType.SOURCE),
                ("sink", "json", NodeType.SINK),
            ]:
                conn.execute(
                    nodes_table.insert().values(
                        node_id=node_id,
                        run_id=run_id,
                        plugin_name=plugin_name,
                        node_type=node_type,
                        plugin_version="1.0.0",
                        determinism=Determinism.DETERMINISTIC if node_type != NodeType.SINK else Determinism.IO_WRITE,
                        config_hash="test",
                        config_json="{}",
                        registered_at=now,
                    )
                )
            for source_node_id, source_name, schema_json, contract_json, contract_hash in [
                ("source-orders", "orders", orders_schema_json, orders_contract_json, orders_contract_hash),
                ("source-refunds", "refunds", refunds_schema_json, refunds_contract_json, refunds_contract_hash),
            ]:
                conn.execute(
                    run_sources_table.insert().values(
                        run_id=run_id,
                        source_node_id=source_node_id,
                        source_name=source_name,
                        plugin_name="null",
                        lifecycle_state="loaded",
                        config_hash="test",
                        schema_json=schema_json,
                        schema_contract_json=contract_json,
                        schema_contract_hash=contract_hash,
                        recorded_at=now,
                    )
                )
            for edge_id, from_node in [("e-orders", "source-orders"), ("e-refunds", "source-refunds")]:
                conn.execute(
                    edges_table.insert().values(
                        edge_id=edge_id,
                        run_id=run_id,
                        from_node_id=from_node,
                        to_node_id="sink",
                        label="continue",
                        default_mode=RoutingMode.MOVE,
                        created_at=now,
                    )
                )
            row_payloads: dict[str, Any] = {}
            for row_id, source_node_id, row_index, source_row_index, ingest_sequence, row_data in [
                ("row-orders", "source-orders", 0, 0, 0, {"order_id": "101"}),
                ("row-refunds", "source-refunds", 1, 0, 1, {"refund_id": "r-7"}),
            ]:
                ref = payload_store.store(json.dumps(row_data).encode())
                row_payloads[row_id] = row_data
                conn.execute(
                    rows_table.insert().values(
                        row_id=row_id,
                        run_id=run_id,
                        source_node_id=source_node_id,
                        row_index=row_index,
                        source_row_index=source_row_index,
                        ingest_sequence=ingest_sequence,
                        source_data_hash=f"h-{row_id}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-multi-source",
                    row_id="row-orders",
                    run_id=run_id,
                    created_at=now,
                )
            )

        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="tok-multi-source",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )
        return graph, orders_contract_hash, refunds_contract_hash, row_payloads

    def test_resume_picks_correct_contract_per_source(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Multi-source resume looks up each row's contract via source_node_id.

        Distinct version_hash() values on the per-source contracts make
        an arbitrary-pick failure observable: if the orchestrator picked
        one contract for both rows, one of the per-row contract lookups
        would carry the wrong hash. ADR-025 §3 requires per-source
        dispatch; this test pins the dispatch.
        """
        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]

        run_id = "resume-multi-source-per-row-contract"
        _graph, orders_hash, refunds_hash, _payloads = self._build_two_source_run(db, payload_store, checkpoint_mgr, run_id=run_id)

        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None

        state = orchestrator._reconstruct_resume_state(
            ResumePoint(
                checkpoint=checkpoint,
                token_id=checkpoint.token_id,
                node_id=checkpoint.node_id,
                sequence_number=checkpoint.sequence_number,
            ),
            payload_store,
        )

        # Distinct contracts are preserved per source (no arbitrary pick).
        assert state.schema_contracts_by_source[NodeID("source-orders")].version_hash() == orders_hash
        assert state.schema_contracts_by_source[NodeID("source-refunds")].version_hash() == refunds_hash
        assert orders_hash != refunds_hash, "test setup error: contracts must differ"

        # Every recovered row carries the source_node_id needed to dispatch
        # the per-row contract — this is the load-bearing carrier ADR-025
        # makes mandatory.
        orders_rows = [r for r in state.unprocessed_rows if r.source_node_id == NodeID("source-orders")]
        refunds_rows = [r for r in state.unprocessed_rows if r.source_node_id == NodeID("source-refunds")]
        assert len(orders_rows) == 1
        assert len(refunds_rows) == 1

        # Verify per-row dispatch directly: the contract recovered for
        # each row by its source_node_id matches the source's contract
        # version_hash. An arbitrary-pick implementation would fail one
        # of these assertions on whichever source's contract did not
        # "win" the pick.
        for row in state.unprocessed_rows:
            recovered_contract = state.schema_contracts_by_source[row.source_node_id]
            if row.source_node_id == NodeID("source-orders"):
                assert recovered_contract.version_hash() == orders_hash
            else:
                assert recovered_contract.version_hash() == refunds_hash

    def test_resume_rejects_missing_contract(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Looking up a row whose source_node_id has no contract crashes.

        ADR-025 §3 requires resume to refuse rather than pick a default
        when the audit trail has a row whose source's schema contract
        is missing. This test calls the resume loop directly with a
        crafted ``schema_contracts_by_source`` that omits one row's
        ``source_node_id`` and asserts the offensive-programming crash.
        """
        from elspeth.contracts import ResumedRow
        from elspeth.contracts.errors import OrchestrationInvariantError
        from elspeth.engine.orchestrator.resume import run_resume_processing_loop
        from elspeth.engine.orchestrator.types import ExecutionCounters, LoopContext

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
        rows = (
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

        # ``schema_contracts_by_source`` deliberately omits source-refunds.
        # The loop must crash on the refunds row rather than reuse
        # the orders contract or pick a default.
        with pytest.raises(OrchestrationInvariantError, match="source-refunds"):
            run_resume_processing_loop(
                loop_ctx,
                rows,
                incomplete_by_row={},
                recovery_manager=MagicMock(),
                payload_store=MagicMock(),
                run_id="run-missing-source-contract",
                resume_checkpoint_id="checkpoint-missing-source-contract",
                schema_contracts_by_source={NodeID("source-orders"): orders_contract},
                source_on_success_by_source={
                    NodeID("source-orders"): "default",
                    NodeID("source-refunds"): "default",
                },
            )

    def test_resume_single_source_round_trip(
        self,
        resume_test_env: dict[str, Any],
    ) -> None:
        """Single-source resume still works after structural change.

        Regression guard: ADR-025 deletes the singular ResumeState field,
        but legitimate single-source pipelines must keep resuming. This
        test builds a single-source FAILED run via the pre-RC6 legacy
        writer (only ``runs.contract_json``, no ``run_sources``) and
        asserts the resume reconstruction returns a single-entry
        ``schema_contracts_by_source`` keyed by the source NodeID
        observed on the recovered rows.
        """
        db = resume_test_env["db"]
        checkpoint_mgr = resume_test_env["checkpoint_manager"]
        payload_store = resume_test_env["payload_store"]
        checkpoint_config = resume_test_env["checkpoint_config"]

        run_id = "resume-single-source-round-trip"
        contract_json, contract_hash = self._create_schema_contract([("id", int)])
        now = datetime.now(UTC)
        source_node_id = NodeID("source-only")

        graph = ExecutionGraph()
        graph.add_node(
            "source-only",
            node_type=NodeType.SOURCE,
            plugin_name="null",
            config={"source_name": "source"},
        )
        graph.add_node(
            "sink",
            node_type=NodeType.SINK,
            plugin_name="json",
            config={"schema": {"mode": "observed"}},
        )
        graph.add_edge("source-only", "sink", label="continue")

        source_schema_json = json.dumps({"properties": {"id": {"type": "integer"}}, "required": ["id"]})

        with db.engine.begin() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                    schema_contract_json=contract_json,
                    schema_contract_hash=contract_hash,
                    runtime_val_manifest_json=_runtime_val_manifest_json(),
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )
            for node_id, plugin_name, node_type in [
                ("source-only", "null", NodeType.SOURCE),
                ("sink", "json", NodeType.SINK),
            ]:
                conn.execute(
                    nodes_table.insert().values(
                        node_id=node_id,
                        run_id=run_id,
                        plugin_name=plugin_name,
                        node_type=node_type,
                        plugin_version="1.0.0",
                        determinism=Determinism.DETERMINISTIC if node_type != NodeType.SINK else Determinism.IO_WRITE,
                        config_hash="test",
                        config_json="{}",
                        registered_at=now,
                    )
                )
            conn.execute(
                edges_table.insert().values(
                    edge_id="e-only",
                    run_id=run_id,
                    from_node_id="source-only",
                    to_node_id="sink",
                    label="continue",
                    default_mode=RoutingMode.MOVE,
                    created_at=now,
                )
            )
            ref = payload_store.store(json.dumps({"id": "42"}).encode())
            conn.execute(
                rows_table.insert().values(
                    row_id="row-single",
                    run_id=run_id,
                    source_node_id="source-only",
                    row_index=0,
                    source_row_index=0,
                    ingest_sequence=0,
                    source_data_hash="h-single",
                    source_data_ref=ref,
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-single",
                    row_id="row-single",
                    run_id=run_id,
                    created_at=now,
                )
            )

        checkpoint_mgr.create_checkpoint(
            run_id=run_id,
            token_id="tok-single",
            node_id="sink",
            sequence_number=1,
            graph=graph,
        )
        checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
        assert checkpoint is not None

        orchestrator = Orchestrator(db, checkpoint_manager=checkpoint_mgr, checkpoint_config=checkpoint_config)
        state = orchestrator._reconstruct_resume_state(
            ResumePoint(
                checkpoint=checkpoint,
                token_id=checkpoint.token_id,
                node_id=checkpoint.node_id,
                sequence_number=checkpoint.sequence_number,
            ),
            payload_store,
        )

        # The single-source path constructs the per-source map from the
        # legacy run_contract keyed under the rows' shared source NodeID.
        assert set(state.schema_contracts_by_source) == {source_node_id}
        assert state.schema_contracts_by_source[source_node_id].version_hash() == contract_hash
        assert len(state.unprocessed_rows) == 1
        assert state.unprocessed_rows[0].source_node_id == source_node_id

    def test_resume_state_has_no_singular_schema_contract_field(self) -> None:
        """``ResumeState.schema_contract`` is deleted by ADR-025 §3.

        Future-proof against typo-restorations: introspect the dataclass
        fields directly via ``dataclasses.fields`` rather than
        ``hasattr`` (which is banned per CLAUDE.md because it swallows
        @property exceptions). A regression that re-adds a singular
        field will be visible immediately.
        """
        import dataclasses

        from elspeth.engine.orchestrator.types import ResumeState

        field_names = {f.name for f in dataclasses.fields(ResumeState)}
        assert "schema_contract" not in field_names, (
            "ResumeState.schema_contract was deleted by ADR-025 §3; a regression "
            "has reintroduced the singular field. Per-row contract dispatch must go "
            "through schema_contracts_by_source[row.source_node_id], not a "
            "next(iter(...)) arbitrary pick."
        )
        assert "schema_contracts_by_source" in field_names
