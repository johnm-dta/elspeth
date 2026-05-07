# tests/integration/plugins/sources/test_payload_storage.py
"""Integration test for P0 bug: Source row payloads never persisted.

This test verifies that source row payloads are stored in the PayloadStore
during normal pipeline runs, as required by CLAUDE.md's non-negotiable
audit requirement: "Source entry - Raw data stored before any processing"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from elspeth.contracts import SourceRow
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
    create_observed_contract,
)
from tests.fixtures.pipeline import build_production_graph


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
                yield SourceRow.valid(row, contract=create_observed_contract(row))

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
            from elspeth.contracts.diversion import SinkWriteResult

            self.received_rows.extend(rows)
            artifact = ArtifactDescriptor.for_file(path="memory://test", size_bytes=len(rows), content_hash="test_hash")
            return SinkWriteResult(artifact=artifact)

    test_source = _PayloadTestSource(source_data)
    test_sink = _PayloadTestSink()

    # Build pipeline config
    config = PipelineConfig(
        source=as_source(test_source),
        transforms=[],
        sinks={"output": as_sink(test_sink)},
        config={},
    )

    graph = build_production_graph(config)

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
    from elspeth.core.landscape.factory import RecorderFactory

    factory = RecorderFactory(db, payload_store=payload_store)
    rows = factory.query.get_rows(result.run_id)

    assert len(rows) == 2, "Should have 2 rows recorded"

    # CRITICAL: Verify source_data_ref is NOT NULL
    for row in rows:
        assert row.source_data_ref is not None, (
            f"Row {row.row_id} source_data_ref should be set, but is NULL. "
            "This violates CLAUDE.md's non-negotiable audit requirement: "
            "'Source entry - Raw data stored before any processing'"
        )

        # Verify payload can be retrieved
        row_data_result = factory.query.get_row_data(row.row_id)
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
