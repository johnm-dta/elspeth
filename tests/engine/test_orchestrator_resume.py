# tests/engine/test_orchestrator_resume.py
"""Tests for orchestrator resume with row processing.

Covers the full resume workflow:
1. Create failed run with checkpoint and payload data
2. Call orchestrator.resume() with payload_store
3. Verify rows after checkpoint are processed
4. P1 Fix: Verify audit trail (node_states, token_outcomes, artifacts)
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import Determinism, NodeID, NodeType, RoutingMode, RowOutcome, RunStatus, SinkName
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.schema import (
    edges_table,
    nodes_table,
    rows_table,
    runs_table,
    token_outcomes_table,
    tokens_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig, RunResult
from elspeth.plugins.sinks.json_sink import JSONSink
from elspeth.plugins.sources.null_source import NullSource
from elspeth.plugins.transforms.passthrough import PassThrough


class TestOrchestratorResumeRowProcessing:
    """Tests for Orchestrator.resume() row processing."""

    @pytest.fixture
    def landscape_db(self, tmp_path: Path) -> LandscapeDB:
        """Create test database."""
        return LandscapeDB(f"sqlite:///{tmp_path}/test.db")

    @pytest.fixture
    def payload_store(self, tmp_path: Path) -> FilesystemPayloadStore:
        """Create test payload store."""
        return FilesystemPayloadStore(tmp_path / "payloads")

    @pytest.fixture
    def checkpoint_manager(self, landscape_db: LandscapeDB) -> CheckpointManager:
        """Create checkpoint manager."""
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        """Create recovery manager."""
        return RecoveryManager(landscape_db, checkpoint_manager)

    @pytest.fixture
    def orchestrator(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> Orchestrator:
        """Create orchestrator with checkpoint manager."""
        return Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_manager,
        )

    @pytest.fixture
    def failed_run_with_payloads(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        payload_store: FilesystemPayloadStore,
    ) -> dict[str, Any]:
        """Create a failed run with 5 rows, checkpoint at row 2.

        Rows 0, 1, 2 are processed (checkpoint at row 2).
        Rows 3, 4 are unprocessed and need resume.

        Returns dict with run_id, row data, and output file info.
        """
        run_id = "test-resume-rows"
        now = datetime.now(UTC)

        # Create source schema for resume ({"id": int, "value": str})
        source_schema_json = json.dumps(
            {"properties": {"id": {"type": "integer"}, "value": {"type": "string"}}, "required": ["id", "value"]}
        )

        with landscape_db.engine.connect() as conn:
            # Create run (failed status)
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                )
            )

            # Create source node
            conn.execute(
                nodes_table.insert().values(
                    node_id="source-node",
                    run_id=run_id,
                    plugin_name="null",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create transform node
            conn.execute(
                nodes_table.insert().values(
                    node_id="transform-node",
                    run_id=run_id,
                    plugin_name="passthrough",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create sink node
            conn.execute(
                nodes_table.insert().values(
                    node_id="sink-node",
                    run_id=run_id,
                    plugin_name="csv",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.IO_WRITE,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create edges (required for resume FK integrity)
            conn.execute(
                edges_table.insert().values(
                    edge_id="e1",
                    run_id=run_id,
                    from_node_id="source-node",
                    to_node_id="transform-node",
                    label="continue",
                    default_mode=RoutingMode.MOVE,
                    created_at=now,
                )
            )
            conn.execute(
                edges_table.insert().values(
                    edge_id="e2",
                    run_id=run_id,
                    from_node_id="transform-node",
                    to_node_id="sink-node",
                    label="continue",
                    default_mode=RoutingMode.MOVE,
                    created_at=now,
                )
            )

            # Create 5 rows with payload data
            all_row_data = []
            token_ids = []
            for i in range(5):
                row_id = f"row-{i:03d}"
                token_id = f"tok-{i:03d}"
                row_data = {"id": i, "value": f"data-{i}"}
                all_row_data.append(row_data)
                token_ids.append(token_id)

                # Store payload
                payload_bytes = json.dumps(row_data).encode("utf-8")
                payload_ref = payload_store.store(payload_bytes)

                conn.execute(
                    rows_table.insert().values(
                        row_id=row_id,
                        run_id=run_id,
                        source_node_id="source-node",
                        row_index=i,
                        source_data_hash=f"hash{i}",
                        source_data_ref=payload_ref,
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

                # Mark rows 0, 1, 2 as COMPLETED (processed before checkpoint)
                # Rows 3, 4 have no terminal outcomes and will be returned as unprocessed
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

        # Build graph matching the nodes created above
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph.add_node("source-node", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
        graph.add_node("transform-node", node_type=NodeType.TRANSFORM, plugin_name="passthrough", config=schema_config)
        graph.add_node("sink-node", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("source-node", "transform-node", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform-node", "sink-node", label="continue", mode=RoutingMode.MOVE)

        # Set internal ID maps needed by orchestrator (normally done by from_plugin_instances)
        # These are required for the orchestrator to resolve nodes during execution
        graph._sink_id_map = {SinkName("default"): NodeID("sink-node")}
        graph._transform_id_map = {0: NodeID("transform-node")}
        graph._config_gate_id_map = {}
        graph._route_resolution_map = {}
        graph._default_sink = "default"

        # Create checkpoint at row 2 (rows 3-4 are unprocessed)
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-002",
            node_id="transform-node",
            sequence_number=2,
            graph=graph,
        )

        return {
            "run_id": run_id,
            "all_row_data": all_row_data,
            "unprocessed_indices": [3, 4],
            "unprocessed_token_ids": token_ids[3:5],
            "graph": graph,
        }

    def _create_test_config(self, tmp_path: Path) -> tuple[PipelineConfig, Path]:
        """Create test pipeline config with real plugins.

        Returns:
            Tuple of (PipelineConfig, output_json_path)
        """
        output_path = tmp_path / "output.json"

        # Use NullSource - resume gets data from payload store
        source = NullSource({})

        # Use PassThrough - simple transform
        transform = PassThrough({"schema": {"fields": "dynamic"}})

        # Use JSONSink in append mode (JSONSink accepts dynamic schemas)
        sink = JSONSink(
            {
                "path": str(output_path),
                "schema": {"fields": "dynamic"},
                "mode": "append",
                "format": "jsonl",
            }
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        return config, output_path

    def test_resume_processes_unprocessed_rows(
        self,
        orchestrator: Orchestrator,
        recovery_manager: RecoveryManager,
        payload_store: FilesystemPayloadStore,
        failed_run_with_payloads: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """resume() processes rows after the checkpoint.

        Given: Failed run with 5 rows, checkpoint at row 2
        When: resume() is called with payload_store
        Then: rows_processed == 2 (rows 3 and 4)
        """
        run_id = failed_run_with_payloads["run_id"]
        fixture_graph = failed_run_with_payloads["graph"]

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id, graph=fixture_graph)
        assert resume_point is not None

        # Create config (uses real plugins) but reuse fixture_graph
        # The fixture_graph has node IDs that match the database entries
        config, _output_path = self._create_test_config(tmp_path)

        # Act: Resume with payload store, using fixture graph to match DB nodes
        result = orchestrator.resume(
            resume_point,
            config,
            fixture_graph,  # Must use fixture graph to match DB node IDs
            payload_store=payload_store,
        )

        # Assert: 2 rows were processed (rows 3 and 4)
        assert result.rows_processed == 2
        assert result.rows_succeeded == 2
        assert result.rows_failed == 0

    def test_resume_writes_to_sink(
        self,
        orchestrator: Orchestrator,
        recovery_manager: RecoveryManager,
        payload_store: FilesystemPayloadStore,
        failed_run_with_payloads: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """resume() writes processed rows to the sink.

        Given: Failed run with rows 3 and 4 unprocessed
        When: resume() is called
        Then: Output CSV contains 2 rows
        """
        run_id = failed_run_with_payloads["run_id"]
        fixture_graph = failed_run_with_payloads["graph"]

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id, graph=fixture_graph)
        assert resume_point is not None

        # Create config (uses real plugins) but reuse fixture_graph
        config, output_path = self._create_test_config(tmp_path)

        # Act: Resume with payload store, using fixture graph to match DB nodes
        orchestrator.resume(
            resume_point,
            config,
            fixture_graph,
            payload_store=payload_store,
        )

        # Assert: Output file was created with rows 3 and 4
        assert output_path.exists()
        content = output_path.read_text()
        lines = content.strip().split("\n")
        # JSONL has no header - just 2 data rows
        assert len(lines) == 2, f"Expected 2 lines (JSONL rows), got {len(lines)}"

        # Verify data content (rows 3 and 4)
        assert "data-3" in content
        assert "data-4" in content

    def test_resume_requires_payload_store(
        self,
        orchestrator: Orchestrator,
        recovery_manager: RecoveryManager,
        failed_run_with_payloads: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """resume() raises error when payload_store is not provided.

        Row data comes from payload store during resume, so it's required.
        """
        run_id = failed_run_with_payloads["run_id"]
        fixture_graph = failed_run_with_payloads["graph"]

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id, graph=fixture_graph)
        assert resume_point is not None

        # Create config (uses real plugins) but reuse fixture_graph
        config, _output_path = self._create_test_config(tmp_path)

        # Act & Assert: Should raise without payload_store
        # Python's type system enforces this as a required keyword-only argument,
        # so we get TypeError before any runtime validation
        with pytest.raises(TypeError, match=r"payload_store"):
            orchestrator.resume(resume_point, config, fixture_graph)

    def test_resume_returns_run_result_with_status(
        self,
        orchestrator: Orchestrator,
        landscape_db: LandscapeDB,
        recovery_manager: RecoveryManager,
        payload_store: FilesystemPayloadStore,
        failed_run_with_payloads: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """resume() returns RunResult with correct status and counts.

        P2 Fix: Assert exact expected values instead of vacuous >= 0.
        """
        run_id = failed_run_with_payloads["run_id"]
        fixture_graph = failed_run_with_payloads["graph"]

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id, graph=fixture_graph)
        assert resume_point is not None

        # Create config (uses real plugins) but reuse fixture_graph
        config, _output_path = self._create_test_config(tmp_path)

        # Act: Resume with payload store, using fixture graph to match DB nodes
        result = orchestrator.resume(
            resume_point,
            config,
            fixture_graph,
            payload_store=payload_store,
        )

        # P2 Fix: Assert exact expected values
        assert isinstance(result, RunResult)
        assert result.run_id == run_id
        # Resume should complete successfully
        assert result.status == RunStatus.COMPLETED, f"Expected COMPLETED, got {result.status}"
        # Exactly 2 rows should be processed (rows 3 and 4)
        assert result.rows_processed == 2, f"Expected 2 rows_processed, got {result.rows_processed}"
        assert result.rows_succeeded == 2, f"Expected 2 rows_succeeded, got {result.rows_succeeded}"
        assert result.rows_failed == 0, f"Expected 0 rows_failed, got {result.rows_failed}"

    def test_resume_creates_audit_trail_for_resumed_tokens(
        self,
        orchestrator: Orchestrator,
        landscape_db: LandscapeDB,
        recovery_manager: RecoveryManager,
        payload_store: FilesystemPayloadStore,
        failed_run_with_payloads: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """P1 Fix: resume() creates proper audit trail entries.

        Verifies that node_states and token_outcomes are created for
        the resumed rows (rows 3 and 4). Note: resume creates NEW tokens
        for the rows, so we query by row_id to find the processed tokens.
        """
        run_id = failed_run_with_payloads["run_id"]
        fixture_graph = failed_run_with_payloads["graph"]

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id, graph=fixture_graph)
        assert resume_point is not None

        # Create config (uses real plugins) but reuse fixture_graph
        config, _output_path = self._create_test_config(tmp_path)

        # Act: Resume with payload store, using fixture graph to match DB nodes
        result = orchestrator.resume(
            resume_point,
            config,
            fixture_graph,
            payload_store=payload_store,
        )

        assert result.status == RunStatus.COMPLETED

        # P1 Fix: Verify audit trail entries for resumed rows
        recorder = LandscapeRecorder(landscape_db)

        # The unprocessed rows (indices 3 and 4) should have tokens with audit records
        # Resume may create new tokens for these rows, so query by row_id
        unprocessed_row_ids = ["row-003", "row-004"]

        for row_id in unprocessed_row_ids:
            # Get tokens for this row
            tokens = recorder.get_tokens(row_id)
            assert len(tokens) >= 1, f"Row {row_id} should have at least one token after resume"

            # Check that at least one token has node_states (processing records)
            has_node_states = False
            has_outcome = False
            for token in tokens:
                states = recorder.get_node_states_for_token(token.token_id)
                if len(states) > 0:
                    has_node_states = True
                outcome = recorder.get_token_outcome(token.token_id)
                if outcome is not None:
                    has_outcome = True

            assert has_node_states, f"Row {row_id} should have tokens with node_states after resume"
            assert has_outcome, f"Row {row_id} should have tokens with token_outcome after resume"


class TestOrchestratorResumeCleanup:
    """Tests for resource cleanup during Orchestrator.resume().

    P3-2026-01-28: Bug fix ensures transform.close() is called during resume.
    """

    @pytest.fixture
    def landscape_db(self, tmp_path: Path) -> LandscapeDB:
        """Create test database."""
        return LandscapeDB(f"sqlite:///{tmp_path}/test.db")

    @pytest.fixture
    def payload_store(self, tmp_path: Path) -> FilesystemPayloadStore:
        """Create test payload store."""
        return FilesystemPayloadStore(tmp_path / "payloads")

    @pytest.fixture
    def checkpoint_manager(self, landscape_db: LandscapeDB) -> CheckpointManager:
        """Create checkpoint manager."""
        return CheckpointManager(landscape_db)

    @pytest.fixture
    def recovery_manager(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> RecoveryManager:
        """Create recovery manager."""
        return RecoveryManager(landscape_db, checkpoint_manager)

    @pytest.fixture
    def orchestrator(self, landscape_db: LandscapeDB, checkpoint_manager: CheckpointManager) -> Orchestrator:
        """Create orchestrator with checkpoint manager."""
        return Orchestrator(
            db=landscape_db,
            checkpoint_manager=checkpoint_manager,
        )

    def test_transform_close_called_during_resume(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        payload_store: FilesystemPayloadStore,
        orchestrator: Orchestrator,
        recovery_manager: RecoveryManager,
        tmp_path: Path,
    ) -> None:
        """transform.close() should be called during resume cleanup.

        P3-2026-01-28: Bug fix ensures _process_resumed_rows() finally block
        calls transform.close() in addition to on_complete().
        """
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.results import TransformResult

        # Create tracking transform
        class TrackingTransform(BaseTransform):
            name = "tracking"
            input_schema = None  # type: ignore[assignment]
            output_schema = None  # type: ignore[assignment]

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.close_called = False
                self.on_complete_called = False

            def process(self, row: dict, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

            def on_complete(self, ctx: Any) -> None:
                self.on_complete_called = True

            def close(self) -> None:
                self.close_called = True

        # Set up fixture run with checkpoint
        run_id = "test-close-resume"
        now = datetime.now(UTC)

        source_schema_json = json.dumps({"properties": {"id": {"type": "integer"}}, "required": ["id"]})

        with landscape_db.engine.connect() as conn:
            # Create run
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                )
            )

            # Create nodes
            conn.execute(
                nodes_table.insert().values(
                    node_id="source-node",
                    run_id=run_id,
                    plugin_name="null",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="transform-node",
                    run_id=run_id,
                    plugin_name="tracking",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="sink-node",
                    run_id=run_id,
                    plugin_name="csv",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.IO_WRITE,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create edges
            conn.execute(
                edges_table.insert().values(
                    edge_id="e1-close",
                    run_id=run_id,
                    from_node_id="source-node",
                    to_node_id="transform-node",
                    label="continue",
                    default_mode=RoutingMode.MOVE,
                    created_at=now,
                )
            )
            conn.execute(
                edges_table.insert().values(
                    edge_id="e2-close",
                    run_id=run_id,
                    from_node_id="transform-node",
                    to_node_id="sink-node",
                    label="continue",
                    default_mode=RoutingMode.MOVE,
                    created_at=now,
                )
            )

            # Create one row that needs processing
            row_data = {"id": 0}
            payload_bytes = json.dumps(row_data).encode("utf-8")
            payload_ref = payload_store.store(payload_bytes)

            conn.execute(
                rows_table.insert().values(
                    row_id="row-close-000",
                    run_id=run_id,
                    source_node_id="source-node",
                    row_index=0,
                    source_data_hash="hash0",
                    source_data_ref=payload_ref,
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-close-000",
                    row_id="row-close-000",
                    created_at=now,
                )
            )
            # Row not marked as completed, so will be processed on resume

            conn.commit()

        # Build graph
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph.add_node("source-node", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
        graph.add_node("transform-node", node_type=NodeType.TRANSFORM, plugin_name="tracking", config=schema_config)
        graph.add_node("sink-node", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("source-node", "transform-node", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform-node", "sink-node", label="continue", mode=RoutingMode.MOVE)
        graph._sink_id_map = {SinkName("default"): NodeID("sink-node")}
        graph._transform_id_map = {0: NodeID("transform-node")}
        graph._config_gate_id_map = {}
        graph._route_resolution_map = {}
        graph._default_sink = "default"

        # Create checkpoint - use the existing token for FK integrity
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-close-000",  # Use actual token for FK constraint
            node_id="source-node",
            sequence_number=0,
            graph=graph,
        )

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id, graph=graph)
        assert resume_point is not None

        # Create config with tracking transform
        from elspeth.plugins.sinks.json_sink import JSONSink
        from elspeth.plugins.sources.null_source import NullSource

        output_path = tmp_path / "output.csv"
        tracking_transform = TrackingTransform()

        config = PipelineConfig(
            source=NullSource({}),
            transforms=[tracking_transform],
            sinks={
                "default": JSONSink(
                    {"path": str(output_path.with_suffix(".json")), "schema": {"fields": "dynamic"}, "mode": "append", "format": "jsonl"}
                )
            },
        )

        # Act: Resume the run
        orchestrator.resume(
            resume_point,
            config,
            graph,
            payload_store=payload_store,
        )

        # Assert: Both on_complete AND close were called
        assert tracking_transform.on_complete_called, "transform.on_complete() was not called during resume"
        assert tracking_transform.close_called, "transform.close() was not called during resume (P3 bug fix)"

    def test_transform_close_called_when_on_complete_fails(
        self,
        landscape_db: LandscapeDB,
        checkpoint_manager: CheckpointManager,
        payload_store: FilesystemPayloadStore,
        orchestrator: Orchestrator,
        recovery_manager: RecoveryManager,
        tmp_path: Path,
    ) -> None:
        """transform.close() should be called even if on_complete() raises.

        Verifies that the suppress(Exception) wrapper in the finally block
        allows close() to be called even when on_complete() fails.
        """
        from elspeth.plugins.base import BaseTransform
        from elspeth.plugins.results import TransformResult

        class FailingOnCompleteTransform(BaseTransform):
            """Transform where on_complete() raises but close() should still be called."""

            name = "failing_on_complete"
            input_schema = None  # type: ignore[assignment]
            output_schema = None  # type: ignore[assignment]

            def __init__(self) -> None:
                super().__init__({"schema": {"fields": "dynamic"}})
                self.close_called = False
                self.on_complete_called = False

            def process(self, row: dict, ctx: Any) -> TransformResult:
                return TransformResult.success(row, success_reason={"action": "test"})

            def on_complete(self, ctx: Any) -> None:
                self.on_complete_called = True
                raise RuntimeError("on_complete intentionally failed")

            def close(self) -> None:
                self.close_called = True

        # Set up fixture run with checkpoint (reuse pattern from above test)
        run_id = "test-close-on-failure"
        now = datetime.now(UTC)

        source_schema_json = json.dumps({"properties": {"id": {"type": "integer"}}, "required": ["id"]})

        with landscape_db.engine.connect() as conn:
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status=RunStatus.FAILED,
                    source_schema_json=source_schema_json,
                )
            )

            conn.execute(
                nodes_table.insert().values(
                    node_id="source-node-fail",
                    run_id=run_id,
                    plugin_name="null",
                    node_type=NodeType.SOURCE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="transform-node-fail",
                    run_id=run_id,
                    plugin_name="failing_on_complete",
                    node_type=NodeType.TRANSFORM,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )
            conn.execute(
                nodes_table.insert().values(
                    node_id="sink-node-fail",
                    run_id=run_id,
                    plugin_name="csv",
                    node_type=NodeType.SINK,
                    plugin_version="1.0",
                    determinism=Determinism.IO_WRITE,
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            conn.execute(
                edges_table.insert().values(
                    edge_id="e1-fail",
                    run_id=run_id,
                    from_node_id="source-node-fail",
                    to_node_id="transform-node-fail",
                    label="continue",
                    default_mode=RoutingMode.MOVE,
                    created_at=now,
                )
            )
            conn.execute(
                edges_table.insert().values(
                    edge_id="e2-fail",
                    run_id=run_id,
                    from_node_id="transform-node-fail",
                    to_node_id="sink-node-fail",
                    label="continue",
                    default_mode=RoutingMode.MOVE,
                    created_at=now,
                )
            )

            row_data = {"id": 0}
            payload_bytes = json.dumps(row_data).encode("utf-8")
            payload_ref = payload_store.store(payload_bytes)

            conn.execute(
                rows_table.insert().values(
                    row_id="row-fail-000",
                    run_id=run_id,
                    source_node_id="source-node-fail",
                    row_index=0,
                    source_data_hash="hash0",
                    source_data_ref=payload_ref,
                    created_at=now,
                )
            )
            conn.execute(
                tokens_table.insert().values(
                    token_id="tok-fail-000",
                    row_id="row-fail-000",
                    created_at=now,
                )
            )

            conn.commit()

        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph.add_node("source-node-fail", node_type=NodeType.SOURCE, plugin_name="null", config=schema_config)
        graph.add_node("transform-node-fail", node_type=NodeType.TRANSFORM, plugin_name="failing_on_complete", config=schema_config)
        graph.add_node("sink-node-fail", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_edge("source-node-fail", "transform-node-fail", label="continue", mode=RoutingMode.MOVE)
        graph.add_edge("transform-node-fail", "sink-node-fail", label="continue", mode=RoutingMode.MOVE)
        graph._sink_id_map = {SinkName("default"): NodeID("sink-node-fail")}
        graph._transform_id_map = {0: NodeID("transform-node-fail")}
        graph._config_gate_id_map = {}
        graph._route_resolution_map = {}
        graph._default_sink = "default"

        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-fail-000",
            node_id="source-node-fail",
            sequence_number=0,
            graph=graph,
        )

        resume_point = recovery_manager.get_resume_point(run_id, graph=graph)
        assert resume_point is not None

        from elspeth.plugins.sinks.json_sink import JSONSink
        from elspeth.plugins.sources.null_source import NullSource

        output_path = tmp_path / "output_fail.json"
        failing_transform = FailingOnCompleteTransform()

        config = PipelineConfig(
            source=NullSource({}),
            transforms=[failing_transform],
            sinks={
                "default": JSONSink(
                    {"path": str(output_path.with_suffix(".json")), "schema": {"fields": "dynamic"}, "mode": "append", "format": "jsonl"}
                )
            },
        )

        # Resume should raise because on_complete failed (cleanup errors are fatal)
        with pytest.raises(RuntimeError, match="Plugin cleanup failed"):
            orchestrator.resume(
                resume_point,
                config,
                graph,
                payload_store=payload_store,
            )

        # Assert: on_complete was called (and raised), but close() was STILL called
        assert failing_transform.on_complete_called, "on_complete() was not called"
        assert failing_transform.close_called, "close() must be called even when on_complete() raises"
