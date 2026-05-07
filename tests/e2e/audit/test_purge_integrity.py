# tests/e2e/audit/test_purge_integrity.py
"""E2E tests verifying payload purge does not break audit integrity.

ELSPETH design: Hashes survive payload deletion -- integrity is
always verifiable even after payloads are purged for storage.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from elspeth.contracts import RunStatus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.lineage import explain
from elspeth.core.landscape.row_data import RowDataState
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.core.retention.purge import PurgeManager
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_linear_pipeline
from tests.fixtures.plugins import CollectSink, PassTransform

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_pipeline(
    tmp_path: Path,
    source_data: list[dict[str, Any]],
) -> tuple[str, LandscapeDB, FilesystemPayloadStore, CollectSink]:
    """Run a linear pipeline with a passthrough transform."""
    db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")

    source, tx_list, sinks, graph = build_linear_pipeline(source_data, transforms=[PassTransform()])
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


def _purge_all(
    db: LandscapeDB,
    payload_store: FilesystemPayloadStore,
) -> None:
    """Purge all payloads by treating everything as expired."""
    purge_manager = PurgeManager(db, payload_store)
    as_of = datetime.now(UTC) + timedelta(minutes=1)
    refs = purge_manager.find_expired_payload_refs(retention_days=0, as_of=as_of)
    if refs:
        purge_manager.purge_payloads(refs)


# ---------------------------------------------------------------------------
# TestPurgeIntegrity
# ---------------------------------------------------------------------------


class TestPurgeIntegrity:
    """Verify payload purge preserves audit integrity."""

    def test_purge_preserves_all_hashes(self, tmp_path: Path) -> None:
        """Run 5 rows, purge all payloads, verify every row's hash is still
        present in the database.
        """
        source_data = [{"id": f"row_{i}", "value": i * 10} for i in range(5)]
        run_id, db, payload_store, _sink = _run_pipeline(tmp_path, source_data)

        factory = RecorderFactory(db, payload_store=payload_store)
        rows = factory.query.get_rows(run_id)
        assert len(rows) == 5

        # Capture hashes before purge
        hashes_before = {row.row_id: row.source_data_hash for row in rows}
        for h in hashes_before.values():
            assert h is not None

        # Purge all payloads
        _purge_all(db, payload_store)

        # Re-query rows and verify hashes are preserved
        rows_after = factory.query.get_rows(run_id)
        for row in rows_after:
            assert row.source_data_hash == hashes_before[row.row_id], f"Hash for {row.row_id} changed after purge"

        db.close()

    def test_purge_marks_rows_as_purged(self, tmp_path: Path) -> None:
        """Run 3 rows, purge, verify get_row_data() returns PURGED for each."""
        source_data = [{"id": f"row_{i}", "value": i} for i in range(3)]
        run_id, db, payload_store, _sink = _run_pipeline(tmp_path, source_data)

        factory = RecorderFactory(db, payload_store=payload_store)
        rows = factory.query.get_rows(run_id)
        assert len(rows) == 3

        # Verify data is available before purge
        for row in rows:
            result = factory.query.get_row_data(row.row_id)
            assert result.state == RowDataState.AVAILABLE

        # Purge all payloads
        _purge_all(db, payload_store)

        # Verify data is now marked as purged
        for row in rows:
            result = factory.query.get_row_data(row.row_id)
            assert result.state == RowDataState.PURGED, f"Row {row.row_id}: expected PURGED, got {result.state}"

        db.close()

    def test_explain_works_after_purge(self, tmp_path: Path) -> None:
        """Run rows, purge, verify explain() still returns lineage
        with payload_available=False.
        """
        source_data = [{"id": "trace_me", "value": 42}]
        run_id, db, payload_store, _sink = _run_pipeline(tmp_path, source_data)

        factory = RecorderFactory(db, payload_store=payload_store)
        rows = factory.query.get_rows(run_id)
        assert len(rows) == 1
        row = rows[0]

        # Verify explain works before purge with payload
        lineage_before = explain(factory.query, factory.data_flow, run_id=run_id, row_id=row.row_id)
        assert lineage_before is not None
        assert lineage_before.source_row.payload_available is True

        # Purge all payloads
        _purge_all(db, payload_store)

        # Verify explain still works after purge
        lineage_after = explain(factory.query, factory.data_flow, run_id=run_id, row_id=row.row_id)
        assert lineage_after is not None, "explain() must work after purge"
        assert lineage_after.source_row is not None
        assert lineage_after.source_row.payload_available is False
        assert lineage_after.source_row.source_data is None

        # Node states should still be intact
        assert len(lineage_after.node_states) > 0
        # Hash should be preserved
        assert lineage_after.source_row.source_data_hash == row.source_data_hash

        db.close()

    def test_partial_purge_only_affects_targeted_refs(self, tmp_path: Path) -> None:
        """Run 5 rows, purge only the first 2 rows' payloads,
        verify last 3 still have payload available.
        """
        source_data = [{"id": f"row_{i}", "value": i * 100} for i in range(5)]
        run_id, db, payload_store, _sink = _run_pipeline(tmp_path, source_data)

        factory = RecorderFactory(db, payload_store=payload_store)
        rows = factory.query.get_rows(run_id)
        assert len(rows) == 5

        # Purge only the first 2 rows' payload refs
        refs_to_purge = []
        for row in rows[:2]:
            assert row.source_data_ref is not None
            refs_to_purge.append(row.source_data_ref)

        purge_manager = PurgeManager(db, payload_store)
        purge_result = purge_manager.purge_payloads(refs_to_purge)
        assert purge_result.deleted_count == len(set(refs_to_purge))

        # Verify first 2 rows are purged
        for row in rows[:2]:
            result = factory.query.get_row_data(row.row_id)
            assert result.state == RowDataState.PURGED, f"Row {row.row_id} should be PURGED"

        # Verify last 3 rows still have payload available
        for row in rows[2:]:
            result = factory.query.get_row_data(row.row_id)
            assert result.state == RowDataState.AVAILABLE, f"Row {row.row_id} should still be AVAILABLE, got {result.state}"

        db.close()
