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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from elspeth.contracts import (
    BatchStatus,
    CallStatus,
    CallType,
    NodeStateStatus,
    NodeType,
    RunStatus,
    TriggerType,
)
from elspeth.contracts.audit import TokenRef
from elspeth.contracts.call_data import RawCallPayload
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.config import GateSettings
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.exporter import LandscapeExporter
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.landscape import make_factory, register_test_node
from tests.fixtures.pipeline import build_fork_pipeline, build_linear_pipeline
from tests.fixtures.plugins import CollectSink, PassTransform

# ── Helpers ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SeededAuditIds:
    """IDs inserted by the exporter-isolation regression through real repositories."""

    node_id: str
    row_id: str
    token_ids: frozenset[str]
    state_ids: frozenset[str]
    operation_id: str
    call_ids: frozenset[str]
    batch_id: str
    artifact_id: str


def _run_linear_on_db(
    db: LandscapeDB,
    payload_root: Path,
    source_data: list[dict[str, Any]],
) -> str:
    """Run a linear pipeline against an existing LandscapeDB and return run_id."""
    payload_store = FilesystemPayloadStore(payload_root)
    source, tx_list, sinks, graph = build_linear_pipeline(source_data, transforms=[PassTransform()])

    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[as_transform(t) for t in tx_list],
        sinks={"default": as_sink(sinks["default"])},
    )

    result = Orchestrator(db).run(config, graph=graph, payload_store=payload_store)
    assert result.status == RunStatus.COMPLETED
    return result.run_id


def _run_linear(
    tmp_path: Path,
    source_data: list[dict[str, Any]],
) -> tuple[str, LandscapeDB]:
    """Run a linear pipeline and return (run_id, db)."""
    db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
    return _run_linear_on_db(db, tmp_path / "payloads", source_data), db


def _run_fork_on_db(
    db: LandscapeDB,
    payload_root: Path,
    source_data: list[dict[str, Any]],
) -> str:
    """Run a fork pipeline against an existing LandscapeDB and return run_id."""
    payload_store = FilesystemPayloadStore(payload_root)
    gate = GateSettings(
        name="router",
        input="primary_out",
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
        sources={"primary": as_source(source)},
        transforms=[as_transform(t) for t in tx_list],
        sinks={name: as_sink(s) for name, s in all_sinks.items()},
        gates=[gate],
    )

    result = Orchestrator(db).run(config, graph=graph, payload_store=payload_store)
    # elspeth-5069612f3c: gate route_to_sink is intentional MOVE
    # (rows_routed_success). Gate-routed-only run -> COMPLETED.
    assert result.status == RunStatus.COMPLETED
    return result.run_id


def _run_fork(
    tmp_path: Path,
    source_data: list[dict[str, Any]],
) -> tuple[str, LandscapeDB]:
    """Run a fork pipeline (gate routes to two sinks) and return (run_id, db)."""
    db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
    return _run_fork_on_db(db, tmp_path / "payloads", source_data), db


def _group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group exported records by record_type."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[r["record_type"]].append(r)
    return grouped


def _one_node_id(factory: RecorderFactory, run_id: str, node_type: NodeType) -> str:
    matches = [node.node_id for node in factory.data_flow.get_nodes(run_id) if node.node_type == node_type]
    assert matches, f"Run {run_id} should have a {node_type.value} node"
    return matches[0]


def _seed_exporter_isolation_records(db: LandscapeDB, run_id: str, label: str) -> _SeededAuditIds:
    """Add exporter record families to a completed run without mocking repositories."""
    factory = make_factory(db)

    source_node_id = _one_node_id(factory, run_id, NodeType.SOURCE)
    sink_node_id = _one_node_id(factory, run_id, NodeType.SINK)
    aggregation_node_id = f"audit-extra-aggregation-{label}"
    register_test_node(
        factory.data_flow,
        run_id,
        aggregation_node_id,
        node_type=NodeType.AGGREGATION,
        plugin_name="audit_extra_aggregation",
    )

    row = factory.data_flow.create_row(
        run_id,
        source_node_id,
        row_index=10_000,
        data={"seed": label},
        source_row_index=10_000,
        ingest_sequence=10_000,
        row_id=f"audit-extra-row-{label}",
    )
    parent_token = factory.data_flow.create_token(row.row_id, token_id=f"audit-extra-token-{label}")
    child_tokens, _fork_group_id = factory.data_flow.fork_token(
        TokenRef(token_id=parent_token.token_id, run_id=run_id),
        row.row_id,
        [f"{label}-left", f"{label}-right"],
        step_in_pipeline=90,
    )
    token_ids = frozenset({parent_token.token_id, *(token.token_id for token in child_tokens)})
    child_token_id = child_tokens[0].token_id

    aggregation_state = factory.execution.begin_node_state(
        child_token_id,
        aggregation_node_id,
        run_id,
        91,
        {"seed": label, "stage": "aggregation"},
    )
    factory.execution.complete_node_state(
        aggregation_state.state_id,
        NodeStateStatus.COMPLETED,
        output_data={"seed": label, "stage": "aggregation"},
        duration_ms=1.0,
    )
    sink_state = factory.execution.begin_node_state(
        child_token_id,
        sink_node_id,
        run_id,
        92,
        {"seed": label, "stage": "sink"},
    )
    state_call = factory.execution.record_call(
        sink_state.state_id,
        factory.execution.allocate_call_index(sink_state.state_id),
        CallType.LLM,
        CallStatus.SUCCESS,
        request_data=RawCallPayload({"seed": label, "scope": "state"}),
        response_data=RawCallPayload({"ok": True}),
        latency_ms=1.0,
    )
    factory.execution.complete_node_state(
        sink_state.state_id,
        NodeStateStatus.COMPLETED,
        output_data={"seed": label, "stage": "sink"},
        duration_ms=1.0,
    )

    operation = factory.execution.begin_operation(
        run_id,
        source_node_id,
        "source_load",
        input_data={"seed": label, "scope": "operation"},
    )
    operation_call = factory.execution.record_operation_call(
        operation.operation_id,
        CallType.HTTP,
        CallStatus.SUCCESS,
        request_data=RawCallPayload({"seed": label, "scope": "operation"}),
        response_data=RawCallPayload({"ok": True}),
        latency_ms=1.0,
    )
    factory.execution.complete_operation(
        operation.operation_id,
        "completed",
        output_data={"seed": label, "scope": "operation"},
        duration_ms=1.0,
    )

    batch = factory.execution.create_batch(run_id, aggregation_node_id, batch_id=f"audit-extra-batch-{label}")
    factory.execution.add_batch_member(batch.batch_id, child_token_id, ordinal=0)
    factory.execution.complete_batch(
        batch.batch_id,
        BatchStatus.COMPLETED,
        trigger_type=TriggerType.COUNT,
        trigger_reason=f"exporter isolation seed {label}",
        state_id=aggregation_state.state_id,
    )

    artifact = factory.execution.register_artifact(
        run_id,
        sink_state.state_id,
        sink_node_id,
        artifact_type="test",
        path=f"file:///tmp/elspeth-exporter-isolation-{label}.json",
        content_hash="0" * 64,
        size_bytes=0,
        artifact_id=f"audit-extra-artifact-{label}",
        idempotency_key=f"audit-extra-artifact-{label}",
    )

    return _SeededAuditIds(
        node_id=aggregation_node_id,
        row_id=row.row_id,
        token_ids=token_ids,
        state_ids=frozenset({aggregation_state.state_id, sink_state.state_id}),
        operation_id=operation.operation_id,
        call_ids=frozenset({state_call.call_id, operation_call.call_id}),
        batch_id=batch.batch_id,
        artifact_id=artifact.artifact_id,
    )


def _exported_ids(grouped: dict[str, list[dict[str, Any]]], record_type: str, id_field: str) -> set[str]:
    return {record[id_field] for record in grouped.get(record_type, [])}


def _assert_seed_present(grouped: dict[str, list[dict[str, Any]]], seed: _SeededAuditIds) -> None:
    assert seed.node_id in _exported_ids(grouped, "node", "node_id")
    assert seed.row_id in _exported_ids(grouped, "row", "row_id")
    assert seed.token_ids <= _exported_ids(grouped, "token", "token_id")
    assert seed.state_ids <= _exported_ids(grouped, "node_state", "state_id")
    assert seed.operation_id in _exported_ids(grouped, "operation", "operation_id")
    assert seed.call_ids <= _exported_ids(grouped, "call", "call_id")
    assert seed.batch_id in _exported_ids(grouped, "batch", "batch_id")
    assert seed.artifact_id in _exported_ids(grouped, "artifact", "artifact_id")


def _assert_seed_absent(grouped: dict[str, list[dict[str, Any]]], seed: _SeededAuditIds) -> None:
    assert seed.node_id not in _exported_ids(grouped, "node", "node_id")
    assert seed.row_id not in _exported_ids(grouped, "row", "row_id")
    assert seed.token_ids.isdisjoint(_exported_ids(grouped, "token", "token_id"))
    assert seed.state_ids.isdisjoint(_exported_ids(grouped, "node_state", "state_id"))
    assert seed.operation_id not in _exported_ids(grouped, "operation", "operation_id")
    assert seed.call_ids.isdisjoint(_exported_ids(grouped, "call", "call_id"))
    assert seed.batch_id not in _exported_ids(grouped, "batch", "batch_id")
    assert seed.artifact_id not in _exported_ids(grouped, "artifact", "artifact_id")


def _assert_export_relationships_are_closed(grouped: dict[str, list[dict[str, Any]]]) -> None:
    node_ids = _exported_ids(grouped, "node", "node_id")
    edge_ids = _exported_ids(grouped, "edge", "edge_id")
    row_ids = _exported_ids(grouped, "row", "row_id")
    token_ids = _exported_ids(grouped, "token", "token_id")
    state_ids = _exported_ids(grouped, "node_state", "state_id")
    operation_ids = _exported_ids(grouped, "operation", "operation_id")
    batch_ids = _exported_ids(grouped, "batch", "batch_id")

    for record_type, records in grouped.items():
        for record in records:
            assert record["run_id"] == grouped["run"][0]["run_id"], f"{record_type} has sibling run_id: {record}"

    for edge in grouped.get("edge", []):
        assert edge["from_node_id"] in node_ids
        assert edge["to_node_id"] in node_ids
    for operation in grouped.get("operation", []):
        assert operation["node_id"] in node_ids
    for row in grouped.get("row", []):
        assert row["source_node_id"] in node_ids
    for token in grouped.get("token", []):
        assert token["row_id"] in row_ids
    for parent in grouped.get("token_parent", []):
        assert parent["token_id"] in token_ids
        assert parent["parent_token_id"] in token_ids
    for outcome in grouped.get("token_outcome", []):
        assert outcome["token_id"] in token_ids
        if outcome["batch_id"] is not None:
            assert outcome["batch_id"] in batch_ids
    for state in grouped.get("node_state", []):
        assert state["token_id"] in token_ids
        assert state["node_id"] in node_ids
    for event in grouped.get("routing_event", []):
        assert event["state_id"] in state_ids
        assert event["edge_id"] in edge_ids
    for call in grouped.get("call", []):
        state_id = call["state_id"]
        operation_id = call["operation_id"]
        assert (state_id is None) != (operation_id is None)
        if state_id is not None:
            assert state_id in state_ids
        if operation_id is not None:
            assert operation_id in operation_ids
    for batch in grouped.get("batch", []):
        assert batch["aggregation_node_id"] in node_ids
    for member in grouped.get("batch_member", []):
        assert member["batch_id"] in batch_ids
        assert member["token_id"] in token_ids
    for artifact in grouped.get("artifact", []):
        assert artifact["sink_node_id"] in node_ids
        assert artifact["produced_by_state_id"] in state_ids


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

    def test_export_run_is_isolated_from_sibling_run_records(self, tmp_path: Path) -> None:
        """A multi-run export must contain only records owned by the requested run."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        try:
            target_run_id = _run_fork_on_db(
                db,
                tmp_path / "payloads-target",
                [
                    {"id": "target-high", "value": 80},
                    {"id": "target-low", "value": 20},
                ],
            )
            sibling_run_id = _run_fork_on_db(
                db,
                tmp_path / "payloads-sibling",
                [
                    {"id": "sibling-high", "value": 90},
                    {"id": "sibling-low", "value": 10},
                ],
            )
            assert target_run_id != sibling_run_id

            target_seed = _seed_exporter_isolation_records(db, target_run_id, "target")
            sibling_seed = _seed_exporter_isolation_records(db, sibling_run_id, "sibling")

            grouped = _group_records(list(LandscapeExporter(db).export_run(target_run_id)))

            expected_record_types = {
                "artifact",
                "batch",
                "batch_member",
                "call",
                "edge",
                "node",
                "node_state",
                "operation",
                "routing_event",
                "row",
                "run",
                "token",
                "token_outcome",
                "token_parent",
            }
            missing_record_types = sorted(record_type for record_type in expected_record_types if not grouped.get(record_type))
            assert not missing_record_types, f"Export regression did not exercise record types: {missing_record_types}"
            assert any(call["operation_id"] is not None for call in grouped["call"])
            assert any(call["state_id"] is not None for call in grouped["call"])

            _assert_seed_present(grouped, target_seed)
            _assert_seed_absent(grouped, sibling_seed)
            _assert_export_relationships_are_closed(grouped)
        finally:
            db.close()


class TestExporterRowBatchStreaming:
    """elspeth-3ae79a4775: the exporter streams the row family in bounded batches.

    The export stream must be IDENTICAL regardless of row_batch_size — record
    order feeds the per-record HMAC signatures and the manifest hash chain, so
    any reordering under chunking is a compliance regression, not a cosmetic one.
    """

    _ROW_FAMILY_RECORD_TYPES = (
        "row",
        "token",
        "token_parent",
        "token_outcome",
        "scheduler_event",
        "node_state",
        "routing_event",
        "call",
    )

    def _build_seeded_fork_run(self, db: LandscapeDB, tmp_path: Path) -> str:
        """Fork run + seeded extras so every row-family record type is exercised."""
        run_id = _run_fork_on_db(
            db,
            tmp_path / "payloads",
            [
                {"id": "r0", "value": 80},
                {"id": "r1", "value": 20},
                {"id": "r2", "value": 60},
                {"id": "r3", "value": 40},
            ],
        )
        seed = _seed_exporter_isolation_records(db, run_id, "stream")
        # The seeded families cover forks/batches/artifacts but not scheduler
        # events; enqueue one so batching covers that export path too.
        factory = make_factory(db)
        payload = factory.scheduler.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))
        factory.scheduler.enqueue_ready(
            run_id=run_id,
            token_id=sorted(seed.token_ids)[0],
            row_id=seed.row_id,
            node_id=seed.node_id,
            step_index=1,
            ingest_sequence=10_000,
            available_at=datetime.now(UTC),
            row_payload_json=payload,
        )
        return run_id

    def test_export_identical_across_row_batch_sizes(self, tmp_path: Path) -> None:
        """Chunked exports must be record-for-record identical to the default."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        try:
            run_id = self._build_seeded_fork_run(db, tmp_path)

            baseline = list(LandscapeExporter(db).export_run(run_id))
            grouped = _group_records(baseline)
            for record_type in self._ROW_FAMILY_RECORD_TYPES:
                assert grouped.get(record_type), f"equivalence fixture must exercise record type: {record_type}"
            assert len(grouped["row"]) >= 4, "fixture must span multiple row batches at small batch sizes"

            for batch_size in (1, 2, 3):
                chunked = list(LandscapeExporter(db, row_batch_size=batch_size).export_run(run_id))
                assert chunked == baseline, f"row_batch_size={batch_size} changed the export stream"
        finally:
            db.close()

    def test_signed_export_hash_chain_identical_across_batch_sizes(self, tmp_path: Path) -> None:
        """Signatures and the manifest hash chain must survive row batching."""
        db = LandscapeDB(f"sqlite:///{tmp_path}/audit.db")
        try:
            run_id = self._build_seeded_fork_run(db, tmp_path)
            key = b"row-batch-equivalence-key"

            signed_default = list(LandscapeExporter(db, signing_key=key).export_run(run_id, sign=True))
            signed_batched = list(LandscapeExporter(db, signing_key=key, row_batch_size=1).export_run(run_id, sign=True))

            # Every data record — signature included — must match; only the
            # trailing manifest legitimately differs (exported_at timestamp).
            assert signed_batched[:-1] == signed_default[:-1]
            manifest_default, manifest_batched = signed_default[-1], signed_batched[-1]
            assert manifest_default["record_type"] == "manifest"
            assert manifest_batched["record_type"] == "manifest"
            assert manifest_batched["final_hash"] == manifest_default["final_hash"]
            assert manifest_batched["record_count"] == manifest_default["record_count"]
        finally:
            db.close()
