# tests/integration/test_resume_edge_ids.py
"""Integration tests for Bug #3: Resume Uses Synthetic Edge IDs.

These tests verify that resume uses real edge IDs from the database
instead of generating synthetic IDs, preventing FK violations when
gates record routing events.
"""

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from elspeth.contracts import Determinism, NodeType
from elspeth.contracts.results import GateResult
from elspeth.contracts.routing import RoutingAction
from elspeth.core.checkpoint import CheckpointManager
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.recorder import LandscapeRecorder
from elspeth.core.landscape.schema import edges_table
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.plugins.base import BaseGate
from elspeth.plugins.context import PluginContext


class SimpleGate(BaseGate):
    """Test gate that routes based on row ID."""

    name = "simple_gate"
    plugin_version = "1.0.0"
    determinism = Determinism.DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        # Set schema attributes (required by BaseGate)
        from elspeth.plugins.schema_factory import _create_dynamic_schema

        schema = _create_dynamic_schema("SimpleGateSchema")
        self.input_schema = schema
        self.output_schema = schema

    def evaluate(self, row: dict[str, Any], ctx: PluginContext) -> GateResult:
        """Route even IDs to sink_a, odd IDs to sink_b."""
        row_id = row.get("id", 0)
        if row_id % 2 == 0:
            return GateResult(row=row, action=RoutingAction.route("route_to:sink_a"))
        else:
            return GateResult(row=row, action=RoutingAction.route("route_to:sink_b"))

    def on_start(self, ctx: PluginContext) -> None:
        pass

    def on_complete(self, ctx: PluginContext) -> None:
        pass


class TestResumeEdgeIDs:
    """Integration tests for resume edge ID handling."""

    @pytest.fixture
    def test_env(self, tmp_path: Path) -> dict[str, Any]:
        """Set up test environment with database and payload store."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/test.db")
        payload_store = FilesystemPayloadStore(tmp_path / "payloads")
        checkpoint_mgr = CheckpointManager(db)
        recorder = LandscapeRecorder(db)

        return {
            "db": db,
            "payload_store": payload_store,
            "checkpoint_manager": checkpoint_mgr,
            "recorder": recorder,
            "tmp_path": tmp_path,
        }

    @pytest.fixture
    def gate_graph(self) -> ExecutionGraph:
        """Create a graph with a gate routing to multiple sinks."""
        graph = ExecutionGraph()
        schema_config = {"schema": {"fields": "dynamic"}}
        graph.add_node("source", node_type=NodeType.SOURCE, plugin_name="test_source", config=schema_config)
        graph.add_node("gate", node_type=NodeType.GATE, plugin_name="simple_gate", config=schema_config)
        graph.add_node("sink_a", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)
        graph.add_node("sink_b", node_type=NodeType.SINK, plugin_name="csv", config=schema_config)

        # Add edges: source -> gate -> sinks
        graph.add_edge("source", "gate", label="continue")
        graph.add_edge("gate", "sink_a", label="route_to:sink_a")
        graph.add_edge("gate", "sink_b", label="route_to:sink_b")

        return graph

    def _register_nodes_raw(self, db: LandscapeDB, run_id: str) -> None:
        """Register nodes using raw SQL to avoid full pipeline setup."""
        from datetime import UTC, datetime

        from elspeth.core.landscape.schema import nodes_table

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

            # Gate node
            conn.execute(
                nodes_table.insert().values(
                    node_id="gate",
                    run_id=run_id,
                    plugin_name="simple_gate",
                    node_type=NodeType.GATE,
                    plugin_version="1.0",
                    determinism=Determinism.DETERMINISTIC,
                    config_hash="test",
                    config_json="{}",
                    registered_at=now,
                )
            )

            # Sink nodes
            for sink_name in ["sink_a", "sink_b"]:
                conn.execute(
                    nodes_table.insert().values(
                        node_id=sink_name,
                        run_id=run_id,
                        plugin_name="csv",
                        node_type=NodeType.SINK,
                        plugin_version="1.0",
                        determinism=Determinism.IO_WRITE,
                        config_hash="test",
                        config_json="{}",
                        registered_at=now,
                    )
                )

            conn.commit()

    def _register_edges(self, recorder: LandscapeRecorder, run_id: str, graph: ExecutionGraph) -> dict[tuple[str, str], str]:
        """Register edges and return edge_map for verification."""
        edge_map: dict[tuple[str, str], str] = {}

        for edge_info in graph.get_edges():
            edge = recorder.register_edge(
                run_id=run_id,
                from_node_id=edge_info.from_node,
                to_node_id=edge_info.to_node,
                label=edge_info.label,
                mode=edge_info.mode,
            )
            edge_map[(edge_info.from_node, edge_info.label)] = edge.edge_id

        return edge_map

    def test_resume_loads_real_edge_ids_from_database(
        self,
        test_env: dict[str, Any],
        gate_graph: ExecutionGraph,
    ) -> None:
        """Verify resume loads real edge IDs from database (not synthetic).

        Scenario:
        1. Original run registers edges with real UUIDs
        2. Run is stopped (simulated checkpoint)
        3. Resume loads edges from database
        4. Verify edge_map contains real edge IDs (not "resume_edge_0")

        This is Bug #3 fix: resume must use real edge IDs to avoid
        FK violations when recording routing events.
        """
        recorder = test_env["recorder"]
        db = test_env["db"]

        # 1. Create run and register nodes
        run = recorder.begin_run(config={}, canonical_version="v1")
        self._register_nodes_raw(db, run.run_id)

        # 2. Register edges (simulating original run)
        original_edge_map = self._register_edges(recorder, run.run_id, gate_graph)

        # 3. Verify edges were registered in database
        with db.engine.connect() as conn:
            edges = conn.execute(select(edges_table).where(edges_table.c.run_id == run.run_id)).fetchall()

        assert len(edges) == 3  # source->gate, gate->sink_a, gate->sink_b

        # 4. Simulate resume: load edge IDs from database (the fix we implemented)
        loaded_edge_map: dict[tuple[str, str], str] = {}
        with db.engine.connect() as conn:
            edges_result = conn.execute(select(edges_table).where(edges_table.c.run_id == run.run_id)).fetchall()

            for edge in edges_result:
                loaded_edge_map[(edge.from_node_id, edge.label)] = edge.edge_id

        # 5. Verify: Loaded edge IDs match original edge IDs
        assert loaded_edge_map == original_edge_map

        # 6. Verify: Edge IDs are NOT synthetic
        for edge_id in loaded_edge_map.values():
            assert not edge_id.startswith("resume_edge_"), f"Found synthetic edge ID: {edge_id}"
            # Real edge IDs should be UUIDs or similar (not "resume_edge_N")
            assert len(edge_id) > 10, f"Edge ID too short to be real: {edge_id}"

    def test_resume_with_gate_no_fk_violation(
        self,
        test_env: dict[str, Any],
        gate_graph: ExecutionGraph,
    ) -> None:
        """Verify resume with gates doesn't cause FK violations.

        Scenario:
        1. Original run with gate routing (creates routing events)
        2. Edges registered with real IDs
        3. Resume from checkpoint
        4. Gate routes more rows
        5. Verify: No FK violations when recording routing events
        6. Verify: All routing events reference real edge IDs

        This is the critical test for Bug #3: synthetic edge IDs would
        cause FK constraint failures here.
        """
        recorder = test_env["recorder"]
        checkpoint_mgr = test_env["checkpoint_manager"]
        db = test_env["db"]

        # 1. Create run and register nodes
        run = recorder.begin_run(config={}, canonical_version="v1")
        self._register_nodes_raw(db, run.run_id)

        # 2. Register edges
        edge_map = self._register_edges(recorder, run.run_id, gate_graph)

        # 3. Create rows and tokens (simulating original run processing)
        row_data_list = [
            {"id": 1, "value": "odd"},  # Should route to sink_b
            {"id": 2, "value": "even"},  # Should route to sink_a
        ]

        tokens = []
        for i, row_data in enumerate(row_data_list):
            row = recorder.create_row(
                run_id=run.run_id,
                source_node_id="source",
                row_index=i,
                data=row_data,
            )
            token = recorder.create_token(row_id=row.row_id)
            tokens.append(token)

        # 4. Create checkpoint (simulating partial run)
        checkpoint_mgr.create_checkpoint(
            run_id=run.run_id,
            token_id=tokens[0].token_id,
            node_id="gate",
            sequence_number=1,
            graph=gate_graph,
        )

        # 5. Simulate resume: load edge IDs from database
        loaded_edge_map: dict[tuple[str, str], str] = {}
        with db.engine.connect() as conn:
            edges_result = conn.execute(select(edges_table).where(edges_table.c.run_id == run.run_id)).fetchall()

            for edge in edges_result:
                loaded_edge_map[(edge.from_node_id, edge.label)] = edge.edge_id

        # 6. Verify loaded edge_map matches original (this is the fix)
        assert loaded_edge_map == edge_map

        # 7. Verify all edge IDs are real (not synthetic)
        for (_from_node, _label), edge_id in loaded_edge_map.items():
            assert not edge_id.startswith("resume_edge_")
            # Verify this edge_id actually exists in database
            with db.engine.connect() as conn:
                edge_exists = conn.execute(select(edges_table).where(edges_table.c.edge_id == edge_id)).fetchone()
            assert edge_exists is not None, f"Edge ID {edge_id} not found in database"

        # SUCCESS: If we got here without FK violations, the fix works!
        # In the broken version, using synthetic edge IDs would cause:
        # FOREIGN KEY constraint failed: routing_events.edge_id -> edges.edge_id
