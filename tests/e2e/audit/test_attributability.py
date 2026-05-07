# tests/e2e/audit/test_attributability.py
"""E2E tests implementing CLAUDE.md's Attributability Test.

From CLAUDE.md:
    For any output, the system must prove complete lineage:

        lineage = landscape.explain(run_id, token_id=token_id, field=field)
        assert lineage.source_row is not None
        assert len(lineage.node_states) > 0

This is the most important E2E test for audit integrity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from elspeth.contracts import PipelineRow, RunStatus
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.lineage import explain
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.results import TransformResult
from tests.fixtures.base_classes import _TestSchema, as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_linear_pipeline
from tests.fixtures.plugins import CollectSink, PassTransform


class _AddFieldTransform(BaseTransform):
    """Transform that adds a 'stage' field to track processing."""

    name = "add_field"
    input_schema = _TestSchema
    output_schema = _TestSchema

    def __init__(self, stage_name: str = "stage_1") -> None:
        super().__init__({"schema": {"mode": "observed"}})
        self._stage_name = stage_name

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        data = row.to_dict()
        stages = data.get("stages", [])
        stages = [*stages, self._stage_name]
        enriched = {**data, "stages": stages, f"processed_{self._stage_name}": True}
        return TransformResult.success(
            PipelineRow(enriched, row.contract),
            success_reason={"action": "add_field", "metadata": {"stage": self._stage_name}},
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
# TestAttributability
# ---------------------------------------------------------------------------


class TestAttributability:
    """The Attributability Test from CLAUDE.md.

    For EVERY output row, the system must prove complete lineage.
    """

    def test_every_output_row_has_complete_lineage(self, tmp_path: Path) -> None:
        """Run 10 rows through source->transform->sink.

        For EVERY row, query lineage and verify:
        1. lineage.source_row is not None
        2. len(lineage.node_states) > 0
        3. lineage.source_row.source_data is not None (payload available)
        """
        source_data = [{"id": f"row_{i}", "value": i * 10} for i in range(10)]
        run_id, db, payload_store, sink = _run_pipeline(tmp_path, source_data, transforms=[PassTransform()])

        assert len(sink.results) == 10

        factory = RecorderFactory(db, payload_store=payload_store)
        rows = factory.query.get_rows(run_id)
        assert len(rows) == 10

        for row in rows:
            lineage = explain(factory.query, factory.data_flow, run_id=run_id, row_id=row.row_id)
            assert lineage is not None, f"Row {row.row_id} (index={row.row_index}) has no lineage"

            # Attributability Test from CLAUDE.md
            assert lineage.source_row is not None, f"Row {row.row_id}: lineage.source_row is None"
            assert len(lineage.node_states) > 0, f"Row {row.row_id}: no node_states in lineage"
            assert lineage.source_row.source_data is not None, f"Row {row.row_id}: source_data is None (payload unavailable)"

        db.close()

    def test_lineage_after_multi_transform(self, tmp_path: Path) -> None:
        """Run 5 rows through source->transform1->transform2->sink.

        For every row, verify lineage shows both transforms in correct order.
        """
        source_data = [{"id": f"doc_{i}", "value": i} for i in range(5)]
        transforms = [
            _AddFieldTransform("stage_a"),
            _AddFieldTransform("stage_b"),
        ]
        run_id, db, payload_store, sink = _run_pipeline(tmp_path, source_data, transforms=transforms)

        assert len(sink.results) == 5

        factory = RecorderFactory(db, payload_store=payload_store)
        rows = factory.query.get_rows(run_id)
        assert len(rows) == 5

        for row in rows:
            lineage = explain(factory.query, factory.data_flow, run_id=run_id, row_id=row.row_id)
            assert lineage is not None
            assert lineage.source_row is not None

            # Must have at least 2 node_states for 2 transforms
            assert len(lineage.node_states) >= 2, f"Row {row.row_id}: expected >= 2 node_states, got {len(lineage.node_states)}"

            # Verify ordering by step_index
            step_indices = [s.step_index for s in lineage.node_states]
            assert step_indices == sorted(step_indices), f"Row {row.row_id}: step_indices not ordered: {step_indices}"

        db.close()

    def test_lineage_includes_hash_integrity(self, tmp_path: Path) -> None:
        """Run rows, verify each row's source_data_hash is not None
        and consistent with stored payload.
        """
        source_data = [{"id": f"hash_{i}", "value": i * 100} for i in range(5)]
        run_id, db, payload_store, _sink = _run_pipeline(tmp_path, source_data, transforms=[PassTransform()])

        factory = RecorderFactory(db, payload_store=payload_store)
        rows = factory.query.get_rows(run_id)
        assert len(rows) == 5

        for row in rows:
            # Hash must always be present
            assert row.source_data_hash is not None, f"Row {row.row_id}: source_data_hash is None"
            assert len(row.source_data_hash) > 0, f"Row {row.row_id}: source_data_hash is empty"

            # Payload ref must be present (we used a real payload store)
            assert row.source_data_ref is not None, f"Row {row.row_id}: source_data_ref is None"

            # Payload must exist in the store
            assert payload_store.exists(row.source_data_ref), f"Row {row.row_id}: payload {row.source_data_ref} not in store"

            # Lineage must also reflect the hash
            lineage = explain(factory.query, factory.data_flow, run_id=run_id, row_id=row.row_id)
            assert lineage is not None
            assert lineage.source_row.source_data_hash == row.source_data_hash

        db.close()
