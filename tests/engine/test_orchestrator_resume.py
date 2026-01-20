# tests/engine/test_orchestrator_resume.py
"""Tests for orchestrator resume with row processing.

Covers the full resume workflow:
1. Create failed run with checkpoint and payload data
2. Call orchestrator.resume() with payload_store
3. Verify rows after checkpoint are processed
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from elspeth.contracts import RoutingMode
from elspeth.core.checkpoint import CheckpointManager, RecoveryManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.schema import (
    nodes_table,
    rows_table,
    runs_table,
    tokens_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig, RunResult
from elspeth.plugins.sinks.csv_sink import CSVSink
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

        with landscape_db.engine.connect() as conn:
            # Create run (failed status)
            conn.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="test",
                    settings_json="{}",
                    canonical_version="sha256-rfc8785-v1",
                    status="failed",
                )
            )

            # Create source node
            conn.execute(
                nodes_table.insert().values(
                    node_id="source-node",
                    run_id=run_id,
                    plugin_name="null",
                    node_type="source",
                    plugin_version="1.0",
                    determinism="deterministic",
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
                    node_type="transform",
                    plugin_version="1.0",
                    determinism="deterministic",
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
                    node_type="sink",
                    plugin_version="1.0",
                    determinism="io_write",
                    config_hash="x",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Create 5 rows with payload data
            all_row_data = []
            for i in range(5):
                row_id = f"row-{i:03d}"
                row_data = {"id": i, "value": f"data-{i}"}
                all_row_data.append(row_data)

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
                        token_id=f"tok-{i:03d}",
                        row_id=row_id,
                        created_at=now,
                    )
                )

            conn.commit()

        # Create checkpoint at row 2 (rows 3-4 are unprocessed)
        checkpoint_manager.create_checkpoint(
            run_id=run_id,
            token_id="tok-002",
            node_id="transform-node",
            sequence_number=2,
        )

        return {
            "run_id": run_id,
            "all_row_data": all_row_data,
            "unprocessed_indices": [3, 4],
        }

    def _create_test_config(self, tmp_path: Path) -> tuple[PipelineConfig, Path]:
        """Create test pipeline config with real plugins.

        Returns:
            Tuple of (PipelineConfig, output_csv_path)
        """
        output_path = tmp_path / "output.csv"

        # Use NullSource - resume gets data from payload store
        source = NullSource({})

        # Use PassThrough - simple transform
        transform = PassThrough({"schema": {"fields": "dynamic"}})

        # Use CSVSink in append mode
        sink = CSVSink(
            {
                "path": str(output_path),
                "schema": {"fields": "dynamic"},
                "mode": "append",
            }
        )

        config = PipelineConfig(
            source=source,
            transforms=[transform],
            sinks={"default": sink},
        )

        return config, output_path

    def _create_test_graph(self) -> ExecutionGraph:
        """Create execution graph matching the test config.

        Graph: source-node -> transform-node -> sink-node
        """
        graph = ExecutionGraph()

        # Add nodes
        graph.add_node(
            "source-node",
            node_type="source",
            plugin_name="null",
            config={},
        )
        graph.add_node(
            "transform-node",
            node_type="transform",
            plugin_name="passthrough",
            config={"schema": {"fields": "dynamic"}},
        )
        graph.add_node(
            "sink-node",
            node_type="sink",
            plugin_name="csv",
            config={"schema": {"fields": "dynamic"}},
        )

        # Add edges
        graph.add_edge(
            "source-node",
            "transform-node",
            label="continue",
            mode=RoutingMode.MOVE,
        )
        graph.add_edge(
            "transform-node",
            "sink-node",
            label="continue",
            mode=RoutingMode.MOVE,
        )

        # Set up internal maps
        graph._sink_id_map = {"default": "sink-node"}
        graph._transform_id_map = {0: "transform-node"}
        graph._config_gate_id_map = {}
        graph._output_sink = "default"
        graph._route_resolution_map = {}

        return graph

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

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id)
        assert resume_point is not None

        # Create config and graph
        config, _output_path = self._create_test_config(tmp_path)
        graph = self._create_test_graph()

        # Act: Resume with payload store
        result = orchestrator.resume(
            resume_point,
            config,
            graph,
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

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id)
        assert resume_point is not None

        # Create config and graph
        config, output_path = self._create_test_config(tmp_path)
        graph = self._create_test_graph()

        # Act: Resume with payload store
        orchestrator.resume(
            resume_point,
            config,
            graph,
            payload_store=payload_store,
        )

        # Assert: Output file was created with rows 3 and 4
        assert output_path.exists()
        content = output_path.read_text()
        lines = content.strip().split("\n")
        # Header + 2 data rows
        assert len(lines) == 3, f"Expected 3 lines (header + 2 rows), got {len(lines)}"

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

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id)
        assert resume_point is not None

        # Create config and graph
        config, _output_path = self._create_test_config(tmp_path)
        graph = self._create_test_graph()

        # Act & Assert: Should raise without payload_store
        with pytest.raises(ValueError, match=r"payload_store.*required"):
            orchestrator.resume(resume_point, config, graph)

    def test_resume_returns_run_result_with_status(
        self,
        orchestrator: Orchestrator,
        recovery_manager: RecoveryManager,
        payload_store: FilesystemPayloadStore,
        failed_run_with_payloads: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        """resume() returns RunResult with correct status and counts."""
        run_id = failed_run_with_payloads["run_id"]

        # Get resume point
        resume_point = recovery_manager.get_resume_point(run_id)
        assert resume_point is not None

        # Create config and graph
        config, _output_path = self._create_test_config(tmp_path)
        graph = self._create_test_graph()

        # Act
        result = orchestrator.resume(
            resume_point,
            config,
            graph,
            payload_store=payload_store,
        )

        # Assert
        assert isinstance(result, RunResult)
        assert result.run_id == run_id
        # Status should be set by completion
        assert result.rows_processed >= 0
        assert result.rows_succeeded >= 0
        assert result.rows_failed >= 0
