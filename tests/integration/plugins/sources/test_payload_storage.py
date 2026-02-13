# tests/integration/plugins/sources/test_payload_storage.py
"""Integration test for P0 bug: Source row payloads never persisted.

This test verifies that source row payloads are stored in the PayloadStore
during normal pipeline runs, as required by CLAUDE.md's non-negotiable
audit requirement: "Source entry - Raw data stored before any processing"
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from elspeth.contracts import NodeType, SourceRow
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.row_data import RowDataState
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import (
    _TestSchema,
    _TestSinkBase,
    _TestSourceBase,
    as_sink,
    as_source,
)

if TYPE_CHECKING:
    from elspeth.core.dag import ExecutionGraph


def _make_contract(data: dict[str, Any]) -> SchemaContract:
    """Create a simple schema contract for test data."""
    fields = tuple(
        FieldContract(
            normalized_name=k,
            original_name=k,
            python_type=object,
            required=False,
            source="inferred",
        )
        for k in data
    )
    return SchemaContract(mode="OBSERVED", fields=fields, locked=True)


def _build_simple_graph(config: PipelineConfig) -> ExecutionGraph:
    """Build minimal graph: source -> sink."""
    from elspeth.contracts import NodeID, RoutingMode, SinkName
    from elspeth.core.dag import ExecutionGraph

    graph = ExecutionGraph()
    schema_config = {"schema": {"mode": "observed"}}
    graph.add_node("source", node_type=NodeType.SOURCE, plugin_name=config.source.name, config=schema_config)

    # Build sink nodes and ID mapping
    sink_ids: dict[SinkName, NodeID] = {}
    for sink_name, sink in config.sinks.items():
        node_id = f"sink_{sink_name}"
        sink_ids[SinkName(sink_name)] = NodeID(node_id)
        graph.add_node(node_id, node_type=NodeType.SINK, plugin_name=sink.name, config=schema_config)
        graph.add_edge("source", node_id, label="continue", mode=RoutingMode.MOVE)

    # Set the internal mappings that orchestrator expects
    graph.set_sink_id_map(sink_ids)
    graph.set_transform_id_map({})  # No transforms in this simple test

    return graph


def test_source_row_payloads_are_stored_during_run(tmp_path: Path, payload_store) -> None:
    """Test that source row payloads are persisted to PayloadStore during normal runs.

    This is a P0 audit requirement from CLAUDE.md:
    "Source entry - Raw data stored before any processing" (non-negotiable)

    Bug: Currently source_data_ref stays NULL and get_row_data returns NEVER_STORED
    Expected: source_data_ref populated with payload reference after run
    """
    # Arrange: Setup database and payload store
    db_path = tmp_path / "test.db"
    payload_path = tmp_path / "payloads"
    payload_path.mkdir()

    db = LandscapeDB.from_url(f"sqlite:///{db_path}")
    payload_store = FilesystemPayloadStore(payload_path)

    # Create simple source with test data
    source_data = [
        {"id": 1, "value": "first row"},
        {"id": 2, "value": "second row"},
    ]

    class _PayloadTestSource(_TestSourceBase):
        name = "test_source"
        output_schema = _TestSchema
        on_success = "output"

        def __init__(self, data: list[dict[str, Any]]) -> None:
            super().__init__()
            self._data = data

        def on_start(self, ctx: Any) -> None:
            pass

        def load(self, ctx: Any) -> Any:
            for row in self._data:
                yield SourceRow.valid(row, contract=_make_contract(row))

        def close(self) -> None:
            pass

    # Create simple sink that collects rows
    class _PayloadTestSink(_TestSinkBase):
        name = "test_sink"

        def __init__(self) -> None:
            super().__init__()
            self.received_rows: list[dict[str, Any]] = []

        def on_start(self, ctx: Any) -> None:
            pass

        def on_complete(self, ctx: Any) -> None:
            pass

        def write(self, rows: Any, ctx: Any) -> Any:
            from elspeth.contracts import ArtifactDescriptor

            self.received_rows.extend(rows)
            return ArtifactDescriptor.for_file(path="memory://test", size_bytes=len(rows), content_hash="test_hash")

    test_source = _PayloadTestSource(source_data)
    test_sink = _PayloadTestSink()

    # Build pipeline config
    config = PipelineConfig(
        source=as_source(test_source),
        transforms=[],
        sinks={"output": as_sink(test_sink)},
        config={},
    )

    graph = _build_simple_graph(config)

    # Act: Run pipeline with payload_store
    orchestrator = Orchestrator(db)
    result = orchestrator.run(
        config,
        graph=graph,
        payload_store=payload_store,  # This parameter doesn't exist yet!
    )

    # Assert: Verify source payloads were stored
    assert result.rows_processed == 2, "Should process 2 rows"

    # Get the recorded rows from landscape
    from elspeth.core.landscape import LandscapeRecorder

    recorder = LandscapeRecorder(db, payload_store=payload_store)
    rows = recorder.get_rows(result.run_id)

    assert len(rows) == 2, "Should have 2 rows recorded"

    # CRITICAL: Verify source_data_ref is NOT NULL
    for row in rows:
        assert row.source_data_ref is not None, (
            f"Row {row.row_id} source_data_ref should be set, but is NULL. "
            "This violates CLAUDE.md's non-negotiable audit requirement: "
            "'Source entry - Raw data stored before any processing'"
        )

        # Verify payload can be retrieved
        row_data_result = recorder.get_row_data(row.row_id)
        assert row_data_result.state == RowDataState.AVAILABLE, (
            f"Row {row.row_id} payload should be AVAILABLE, got {row_data_result.state}. "
            "PayloadStore should have stored the source row data."
        )

        # Verify payload content matches original
        assert row_data_result.data is not None, "Payload data should not be None"
        original_data = source_data[row.row_index]
        assert row_data_result.data == original_data, (
            f"Retrieved payload should match original source data. Expected {original_data}, got {row_data_result.data}"
        )

    # Verify payloads actually exist in filesystem
    for row in rows:
        source_data_ref = row.source_data_ref
        assert source_data_ref is not None, f"Row {row.row_id} source_data_ref should be set"
        assert payload_store.exists(source_data_ref), f"Payload file {source_data_ref} should exist in {payload_path}"

    db.close()
