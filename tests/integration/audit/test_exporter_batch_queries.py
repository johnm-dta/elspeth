"""Integration tests verifying exporter batch query correctness.

The LandscapeExporter uses 6 bulk queries to pre-load tokens, parents,
states, routing events, calls, and outcomes into lookup dicts (Bug 76r fix).
These tests run real pipelines against real SQLite and verify the exported
records are relationally consistent — not just present, but correctly
cross-referenced.

The existing E2E tests (test_export_reimport.py) verify record-type
completeness and field presence. These tests verify relational integrity:
every token_parent references a real token, every routing_event references
a real node_state, and record counts match direct DB queries.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from elspeth.contracts import RunStatus
from elspeth.core.config import GateSettings
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.exporter import LandscapeExporter
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.pipeline import build_fork_pipeline, build_linear_pipeline
from tests.fixtures.plugins import CollectSink, PassTransform

# ── Helpers ───────────────────────────────────────────────────────────


def _run_linear(
    tmp_path: Path,
    source_data: list[dict[str, Any]],
) -> tuple[str, LandscapeDB]:
    """Run a linear pipeline and return (run_id, db)."""
    db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")

    source, tx_list, sinks, graph = build_linear_pipeline(source_data, transforms=[PassTransform()])

    config = PipelineConfig(
        source=as_source(source),
        transforms=[as_transform(t) for t in tx_list],
        sinks={"default": as_sink(sinks["default"])},
    )

    result = Orchestrator(db).run(config, graph=graph, payload_store=payload_store)
    assert result.status == RunStatus.COMPLETED
    return result.run_id, db


def _run_fork(
    tmp_path: Path,
    source_data: list[dict[str, Any]],
) -> tuple[str, LandscapeDB]:
    """Run a fork pipeline (gate routes to two sinks) and return (run_id, db)."""
    db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")

    gate = GateSettings(
        name="router",
        input="list_source_out",
        condition="row['value'] > 50",
        routes={"true": "high_sink", "false": "low_sink"},
    )
    sinks = {
        "high_sink": CollectSink("high_sink"),
        "low_sink": CollectSink("low_sink"),
    }

    source, tx_list, all_sinks, graph = build_fork_pipeline(
        source_data,
        gate=gate,
        branch_transforms={},
        sinks=sinks,
    )

    config = PipelineConfig(
        source=as_source(source),
        transforms=[as_transform(t) for t in tx_list],
        sinks={name: as_sink(s) for name, s in all_sinks.items()},
        gates=[gate],
    )

    result = Orchestrator(db).run(config, graph=graph, payload_store=payload_store)
    # elspeth-5069612f3c: gate route_to_sink is intentional MOVE
    # (rows_routed_success). Gate-routed-only run -> COMPLETED.
    assert result.status == RunStatus.COMPLETED
    return result.run_id, db


def _group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group exported records by record_type."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[r["record_type"]].append(r)
    return grouped


# ── Tests ─────────────────────────────────────────────────────────────


class TestExporterBatchQueryIntegrity:
    """Verify batch-queried export records are relationally consistent."""

    def test_token_parent_references_existing_tokens(self, tmp_path: Path) -> None:
        """Every token_parent.token_id and parent_token_id must exist in tokens."""
        source_data = [{"id": f"r{i}", "value": i * 20} for i in range(5)]
        run_id, db = _run_linear(tmp_path, source_data)

        try:
            records = list(LandscapeExporter(db).export_run(run_id))
            grouped = _group_records(records)

            token_ids = {t["token_id"] for t in grouped["token"]}

            for parent_rec in grouped.get("token_parent", []):
                assert parent_rec["token_id"] in token_ids, f"token_parent references unknown token_id: {parent_rec['token_id']}"
                assert parent_rec["parent_token_id"] in token_ids, (
                    f"token_parent references unknown parent_token_id: {parent_rec['parent_token_id']}"
                )
        finally:
            db.close()

    def test_node_state_references_existing_tokens(self, tmp_path: Path) -> None:
        """Every node_state.token_id must exist in tokens."""
        source_data = [{"id": f"r{i}", "value": i * 10} for i in range(5)]
        run_id, db = _run_linear(tmp_path, source_data)

        try:
            records = list(LandscapeExporter(db).export_run(run_id))
            grouped = _group_records(records)

            token_ids = {t["token_id"] for t in grouped["token"]}

            for state in grouped.get("node_state", []):
                assert state["token_id"] in token_ids, f"node_state references unknown token_id: {state['token_id']}"
        finally:
            db.close()

    def test_routing_events_reference_existing_states(self, tmp_path: Path) -> None:
        """Every routing_event.state_id must exist in node_state records."""
        source_data = [{"id": f"r{i}", "value": i * 30} for i in range(5)]
        run_id, db = _run_fork(tmp_path, source_data)

        try:
            records = list(LandscapeExporter(db).export_run(run_id))
            grouped = _group_records(records)

            state_ids = {s["state_id"] for s in grouped.get("node_state", [])}

            for event in grouped.get("routing_event", []):
                assert event["state_id"] in state_ids, f"routing_event references unknown state_id: {event['state_id']}"
        finally:
            db.close()

    def test_token_outcome_references_existing_tokens(self, tmp_path: Path) -> None:
        """Every token_outcome.token_id must exist in tokens."""
        source_data = [{"id": f"r{i}", "value": i * 10} for i in range(5)]
        run_id, db = _run_linear(tmp_path, source_data)

        try:
            records = list(LandscapeExporter(db).export_run(run_id))
            grouped = _group_records(records)

            token_ids = {t["token_id"] for t in grouped["token"]}

            for outcome in grouped.get("token_outcome", []):
                assert outcome["token_id"] in token_ids, f"token_outcome references unknown token_id: {outcome['token_id']}"
        finally:
            db.close()

    def test_export_deterministic(self, tmp_path: Path) -> None:
        """Exporting the same run twice produces identical record sequences."""
        source_data = [{"id": f"r{i}", "value": i} for i in range(10)]
        run_id, db = _run_linear(tmp_path, source_data)

        try:
            exporter = LandscapeExporter(db)
            first = list(exporter.export_run(run_id))
            second = list(exporter.export_run(run_id))

            assert len(first) == len(second), "Record count differs between exports"
            for i, (a, b) in enumerate(zip(first, second, strict=True)):
                assert a == b, f"Record {i} differs between exports: {a} != {b}"
        finally:
            db.close()

    def test_fork_pipeline_produces_routing_events(self, tmp_path: Path) -> None:
        """A gate pipeline must produce routing_event records in the export."""
        source_data = [
            {"id": "high", "value": 80},
            {"id": "low", "value": 20},
        ]
        run_id, db = _run_fork(tmp_path, source_data)

        try:
            records = list(LandscapeExporter(db).export_run(run_id))
            grouped = _group_records(records)

            assert len(grouped.get("routing_event", [])) > 0, "Fork pipeline should produce routing_event records"
        finally:
            db.close()

    def test_record_counts_match_direct_queries(self, tmp_path: Path) -> None:
        """Exported record counts must match direct query repository queries."""
        source_data = [{"id": f"r{i}", "value": i * 10} for i in range(10)]
        run_id, db = _run_linear(tmp_path, source_data)

        try:
            from tests.fixtures.landscape import make_factory

            factory = make_factory(db)
            direct_rows = factory.query.get_rows(run_id)
            direct_tokens = factory.query.get_all_tokens_for_run(run_id)
            direct_states = factory.query.get_all_node_states_for_run(run_id)
            direct_outcomes = factory.query.get_all_token_outcomes_for_run(run_id)

            grouped = _group_records(list(LandscapeExporter(db).export_run(run_id)))

            assert len(grouped["row"]) == len(direct_rows), f"row count: export={len(grouped['row'])} vs db={len(direct_rows)}"
            assert len(grouped["token"]) == len(direct_tokens), f"token count: export={len(grouped['token'])} vs db={len(direct_tokens)}"
            assert len(grouped.get("node_state", [])) == len(direct_states), (
                f"node_state count: export={len(grouped.get('node_state', []))} vs db={len(direct_states)}"
            )
            assert len(grouped.get("token_outcome", [])) == len(direct_outcomes), (
                f"token_outcome count: export={len(grouped.get('token_outcome', []))} vs db={len(direct_outcomes)}"
            )
        finally:
            db.close()
