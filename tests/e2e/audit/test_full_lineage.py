# tests/e2e/audit/test_full_lineage.py
"""E2E tests for audit lineage completeness.

Migrated from tests/system/audit_verification/test_lineage_completeness.py.
Uses file-based SQLite, real payload stores, and production assembly path
(ExecutionGraph.from_plugin_instances() via build_linear_pipeline).

Per ELSPETH's guiding principle: "I don't know what happened" is never
an acceptable answer for any output.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from elspeth.contracts import NodeStateStatus, PipelineRow, RunStatus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.lineage import explain
from elspeth.core.landscape.row_data import RowDataState
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.core.retention.purge import PurgeManager
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_linear_pipeline
from tests.fixtures.plugins import CollectSink, PassTransform


class _EnrichTransform(BaseTransform):
    """Transform that adds enrichment fields to the data."""

    name = "enricher"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        enriched = {**row.to_dict(), "enriched": True, "processed_by": self.name}
        return TransformResult.success(
            PipelineRow(enriched, row.contract),
            success_reason={"action": "enrich"},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_pipeline(
    tmp_path: Path,
    source_data: list[dict[str, Any]],
    transforms: list[Any] | None = None,
) -> tuple[str, LandscapeDB, FilesystemPayloadStore, CollectSink]:
    """Run a linear pipeline and return (run_id, db, payload_store, sink)."""
    db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")

    source, tx_list, sinks, graph = build_linear_pipeline(source_data, transforms=transforms)
    sink = sinks["default"]

    config = PipelineConfig(
        source=as_source(source),
        transforms=[as_transform(t) for t in tx_list],
        sinks={"default": as_sink(sink)},
    )

    orchestrator = Orchestrator(db)
    result = orchestrator.run(config, graph=graph, payload_store=payload_store)
    assert result.status == RunStatus.COMPLETED
    return result.run_id, db, payload_store, sink


# ---------------------------------------------------------------------------
# TestLineageCompleteness
# ---------------------------------------------------------------------------


class TestLineageCompleteness:
    """Tests for verifying lineage is complete for all processed rows."""

    def test_simple_pipeline_has_complete_lineage(self, tmp_path: Path) -> None:
        """Run 3 rows through source->transform->sink.

        Verify all rows completed and sink received all rows.
        """
        source_data = [
            {"id": "row_1", "value": 100},
            {"id": "row_2", "value": 200},
            {"id": "row_3", "value": 300},
        ]
        run_id, db, payload_store, sink = _run_pipeline(tmp_path, source_data, transforms=[PassTransform()])

        # Verify sink received all 3 rows
        assert len(sink.results) == 3

        # Verify all rows are recorded in audit trail
        factory = RecorderFactory(db, payload_store=payload_store)
        rows = factory.query.get_rows(run_id)
        assert len(rows) == 3

        # Verify each row has a lineage
        for row in rows:
            lineage = explain(factory.query, factory.data_flow, run_id=run_id, row_id=row.row_id)
            assert lineage is not None, f"Row {row.row_id} has no lineage"
            assert lineage.source_row is not None
            assert len(lineage.node_states) > 0

        db.close()

    def test_multi_transform_pipeline_has_complete_lineage(self, tmp_path: Path) -> None:
        """Run 2 rows through source->transform1->transform2->sink.

        Verify enrichment happened and all rows completed.
        """
        source_data = [
            {"id": "doc_1", "value": 10},
            {"id": "doc_2", "value": 20},
        ]
        run_id, db, payload_store, sink = _run_pipeline(
            tmp_path,
            source_data,
            transforms=[PassTransform(), _EnrichTransform()],
        )

        # Verify sink received enriched data
        assert len(sink.results) == 2
        for row_data in sink.results:
            assert row_data.get("enriched") is True
            assert row_data.get("processed_by") == "enricher"

        # Verify each row has lineage with at least 2 transform states
        factory = RecorderFactory(db, payload_store=payload_store)
        rows = factory.query.get_rows(run_id)
        assert len(rows) == 2

        for row in rows:
            lineage = explain(factory.query, factory.data_flow, run_id=run_id, row_id=row.row_id)
            assert lineage is not None
            assert len(lineage.node_states) >= 2, f"Row {row.row_id} expected >= 2 node_states, got {len(lineage.node_states)}"

        db.close()


# ---------------------------------------------------------------------------
# TestLineageAfterRetention
# ---------------------------------------------------------------------------


class TestLineageAfterRetention:
    """Tests for lineage availability after payload retention purge."""

    def test_lineage_hashes_survive_payload_purge(self, tmp_path: Path) -> None:
        """Run 1 row, purge payloads, verify hash remains but payload is gone.

        ELSPETH design: Hashes survive payload deletion -- integrity is
        always verifiable even after payloads are purged.
        """
        source_data = [{"id": "row_1", "value": 100}]
        run_id, db, payload_store, _sink = _run_pipeline(tmp_path, source_data, transforms=[PassTransform()])

        factory = RecorderFactory(db, payload_store=payload_store)
        rows = factory.query.get_rows(run_id)
        assert len(rows) == 1
        row = rows[0]
        assert row.source_data_ref is not None
        assert row.source_data_hash is not None

        # Verify payload exists before purge
        assert payload_store.exists(row.source_data_ref)
        before = factory.query.get_row_data(row.row_id)
        assert before.state == RowDataState.AVAILABLE

        # Purge payloads (treat run as expired by using retention_days=0 and future as_of)
        purge_manager = PurgeManager(db, payload_store)
        as_of = datetime.now(UTC) + timedelta(minutes=1)
        refs = purge_manager.find_expired_payload_refs(retention_days=0, as_of=as_of)
        assert row.source_data_ref in refs
        purge_manager.purge_payloads(refs)

        # Payload should now be purged, but hash remains
        after = factory.query.get_row_data(row.row_id)
        assert after.state == RowDataState.PURGED

        lineage = factory.query.explain_row(run_id, row.row_id)
        assert lineage is not None
        assert lineage.source_data_hash == row.source_data_hash
        assert lineage.payload_available is False
        assert lineage.source_data is None

        db.close()


# ---------------------------------------------------------------------------
# TestExplainQueryFunctionality
# ---------------------------------------------------------------------------


class TestExplainQueryFunctionality:
    """Tests for explain query functionality."""

    def test_explain_returns_source_data(self, tmp_path: Path) -> None:
        """Run 1 row, use explain() to get lineage, verify source_row data matches."""
        source_data = {"id": "trace_me", "value": 42}
        run_id, db, payload_store, _sink = _run_pipeline(tmp_path, [source_data], transforms=[PassTransform()])

        factory = RecorderFactory(db, payload_store=payload_store)
        rows = factory.query.get_rows(run_id)
        assert len(rows) == 1

        row = rows[0]
        lineage = explain(factory.query, factory.data_flow, run_id=run_id, row_id=row.row_id)

        assert lineage is not None, "Explain must return lineage for processed row"
        assert lineage.source_row is not None
        assert lineage.source_row.payload_available is True
        assert lineage.source_row.source_data == source_data

    def test_explain_returns_transform_history(self, tmp_path: Path) -> None:
        """Run 1 row through 2 transforms, verify explain() returns
        node_states with correct ordering by step_index.
        """
        source_data = [{"id": "history_row", "value": 100}]
        run_id, db, payload_store, _sink = _run_pipeline(
            tmp_path,
            source_data,
            transforms=[PassTransform(), _EnrichTransform()],
        )

        factory = RecorderFactory(db, payload_store=payload_store)
        rows = factory.query.get_rows(run_id)
        assert len(rows) == 1

        row = rows[0]
        lineage = explain(factory.query, factory.data_flow, run_id=run_id, row_id=row.row_id)

        assert lineage is not None
        assert len(lineage.node_states) >= 2, f"Expected >= 2 node_states (2 transforms), got {len(lineage.node_states)}"

        # Verify node_states are ordered by step_index
        step_indices = [state.step_index for state in lineage.node_states]
        assert step_indices == sorted(step_indices), f"Node states should be ordered by step_index: {step_indices}"

        # Verify all states completed successfully
        for state in lineage.node_states:
            assert state.status == NodeStateStatus.COMPLETED, f"Node state at step {state.step_index} has status {state.status}"

        db.close()
