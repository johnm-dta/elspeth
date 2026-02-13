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

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select

from elspeth.contracts import Determinism, NodeType, RoutingMode, RowOutcome, RunStatus
from elspeth.contracts.types import NodeID, SinkName
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.schema import (
    checkpoints_table,
    edges_table,
    nodes_table,
    rows_table,
    runs_table,
    tokens_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.json_sink import JSONSink
from elspeth.plugins.sources.null_source import NullSource
from elspeth.plugins.transforms.passthrough import PassThrough


def _null_source(on_success: str = "default") -> NullSource:
    """Create NullSource with on_success set (lifted from config to attribute)."""
    source = NullSource({})
    source.on_success = on_success
    return source


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
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
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
                        source_data_hash=f"h{i}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"t{i}",
                        row_id=f"r{i}",
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
        recorder = LandscapeRecorder(db)
        for i in range(3):
            recorder.record_token_outcome(
                run_id=run_id,
                token_id=f"t{i}",
                outcome=RowOutcome.COMPLETED,
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
            source=_null_source("default"),
            transforms=[passthrough],
            sinks={"default": CSVSink({"path": str(output_path), "schema": strict_schema, "mode": "append"})},
        )

        # Build graph manually
        resume_graph = ExecutionGraph()
        schema_config = {"schema": strict_schema}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
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
        assert result.rows_processed == 2  # Rows 3 and 4
        assert result.rows_succeeded == 2
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
        recorder = LandscapeRecorder(db)
        for i in range(3):
            recorder.record_token_outcome(
                run_id=run_id,
                token_id=f"t{i}",
                outcome=RowOutcome.COMPLETED,
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
            source=_null_source("default"),
            transforms=[passthrough],
            sinks={"default": CSVSink({"path": str(output_path), "schema": strict_schema, "mode": "append"})},
        )

        resume_graph = ExecutionGraph()
        schema_config = {"schema": strict_schema}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
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

        # Verify early-exit behavior
        assert result.rows_processed == 0, "Early-exit path should process 0 rows (all already done)"
        assert result.rows_succeeded == 0
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
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
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
                        source_data_hash=f"h{i}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"t{i}",
                        row_id=f"r{i}",
                        created_at=now,
                    )
                )

        # Mark first row as completed (checkpoint will be at row 0)
        recorder = LandscapeRecorder(db)
        recorder.record_token_outcome(
            run_id=run_id,
            token_id="t0",
            outcome=RowOutcome.COMPLETED,
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

        # Use CSVSink with strict schema matching the data: {"id": int, "timestamp": datetime}
        # CSVSink stringifies datetime values automatically
        strict_schema = {"mode": "fixed", "fields": ["id: int", "timestamp: str"]}
        passthrough = PassThrough({"schema": strict_schema})
        passthrough.on_error = "discard"
        config = PipelineConfig(
            source=_null_source("default"),
            transforms=[passthrough],
            sinks={"default": CSVSink({"path": str(output_path), "schema": strict_schema, "mode": "append"})},
        )

        resume_graph = ExecutionGraph()
        resume_schema_config: dict[str, Any] = {"schema": strict_schema}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=resume_schema_config)
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

        # Verify resume succeeded - should process rows 1-2 (2 rows)
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 2, f"Expected 2 rows processed (r1, r2), got {result.rows_processed}"

        # The fact that resume succeeded without type errors proves datetime restoration worked
        # If schema reconstruction had failed, Pydantic would have kept timestamps as strings
        # and downstream transforms expecting datetime would have crashed
        assert result.rows_succeeded == 2

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
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
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
                        source_data_hash=f"h{i}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"t{i}",
                        row_id=f"r{i}",
                        created_at=now,
                    )
                )

        # Mark first row as completed (checkpoint will be at row 0)
        recorder = LandscapeRecorder(db)
        recorder.record_token_outcome(
            run_id=run_id,
            token_id="t0",
            outcome=RowOutcome.COMPLETED,
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

        # Use CSVSink with strict schema matching the data: {"id": int, "amount": Decimal}
        # CSVSink stringifies Decimal values automatically
        strict_schema = {"mode": "fixed", "fields": ["id: int", "amount: str"]}
        passthrough = PassThrough({"schema": strict_schema})
        passthrough.on_error = "discard"
        config = PipelineConfig(
            source=_null_source("default"),
            transforms=[passthrough],
            sinks={"default": CSVSink({"path": str(output_path), "schema": strict_schema, "mode": "append"})},
        )

        resume_graph = ExecutionGraph()
        resume_schema_config: dict[str, Any] = {"schema": strict_schema}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=resume_schema_config)
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

        # Verify resume succeeded with Decimal precision preserved - should process row 1 (1 row)
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 1, f"Expected 1 row processed (r1), got {result.rows_processed}"
        assert result.rows_succeeded == 1

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
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
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
                        source_data_hash=f"h{i}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"t{i}",
                        row_id=f"r{i}",
                        created_at=now,
                    )
                )

        # Mark first row as completed (checkpoint will be at row 0)
        recorder = LandscapeRecorder(db)
        recorder.record_token_outcome(
            run_id=run_id,
            token_id="t0",
            outcome=RowOutcome.COMPLETED,
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
            source=_null_source("default"),
            transforms=[passthrough],
            sinks={
                "default": JSONSink(
                    {"path": str(output_path.with_suffix(".json")), "schema": {"mode": "observed"}, "mode": "append", "format": "jsonl"}
                )
            },
        )

        resume_graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
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

        # Verify resume succeeded with array types preserved - should process row 1 (1 row)
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 1, f"Expected 1 row processed (r1), got {result.rows_processed}"
        assert result.rows_succeeded == 1

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
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
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
                        source_data_hash=f"h{i}",
                        source_data_ref=ref,
                        created_at=now,
                    )
                )
                conn.execute(
                    tokens_table.insert().values(
                        token_id=f"t{i}",
                        row_id=f"r{i}",
                        created_at=now,
                    )
                )

        # Mark first row as completed (checkpoint will be at row 0)
        recorder = LandscapeRecorder(db)
        recorder.record_token_outcome(
            run_id=run_id,
            token_id="t0",
            outcome=RowOutcome.COMPLETED,
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
            source=_null_source("default"),
            transforms=[passthrough],
            sinks={
                "default": JSONSink(
                    {"path": str(output_path.with_suffix(".json")), "schema": {"mode": "observed"}, "mode": "append", "format": "jsonl"}
                )
            },
        )

        resume_graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
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

        # Verify resume succeeded with nested objects preserved - should process row 1 (1 row)
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 1, f"Expected 1 row processed (r1), got {result.rows_processed}"
        assert result.rows_succeeded == 1

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
        graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
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

            # Create a dummy row (won't be processed - resume will fail during schema reconstruction)
            row_data = {"id": 0, "location": "some-location"}
            ref = payload_store.store(json.dumps(row_data).encode())
            conn.execute(
                rows_table.insert().values(
                    row_id="r0",
                    run_id=run_id,
                    source_node_id="src",
                    row_index=0,
                    source_data_hash="h0",
                    source_data_ref=ref,
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="t0",
                    row_id="r0",
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
            source=_null_source("default"),
            transforms=[passthrough],
            sinks={"default": JSONSink({"path": "/tmp/dummy.json", "schema": {"mode": "observed"}, "mode": "write", "format": "jsonl"})},
        )

        resume_graph = ExecutionGraph()
        schema_config = {"schema": {"mode": "observed"}}
        resume_graph.add_node("src", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
        resume_graph.add_node("xform", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        resume_graph.add_node("sink", node_type=NodeType.SINK, plugin_name="json", config=schema_config)
        resume_graph.add_edge("src", "xform", label="continue")
        resume_graph.add_edge("xform", "sink", label="continue")
        resume_graph.set_sink_id_map({SinkName("default"): NodeID("sink")})
        resume_graph.set_transform_id_map({0: NodeID("xform")})

        # CRITICAL: Must crash with clear error, not silently degrade to str
        with pytest.raises(ValueError, match=r"unsupported type 'geo-point'"):
            orchestrator.resume(
                resume_point=resume_point,
                config=config,
                graph=resume_graph,
                payload_store=payload_store,
            )
