"""Multi-source isolation regression coverage.

These tests exercise the production pipeline assembly path for source-boundary
and transform-boundary isolation, then hit the durable scheduler repository
directly for claim-ordering isolation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import and_, select

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.contracts import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.core.config import load_settings_from_yaml_string
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.schema import (
    edges_table,
    node_states_table,
    routing_events_table,
    rows_table,
    run_sources_table,
    token_outcomes_table,
    token_work_items_table,
    tokens_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config
from tests.fixtures.landscape import make_factory, make_landscape_db


@dataclass(frozen=True, slots=True)
class _RunFixture:
    db: LandscapeDB
    run_id: str
    output_path: Path
    quarantine_path: Path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run_yaml_pipeline(tmp_path: Path, yaml_text: str, *, db_name: str = "audit.db") -> _RunFixture:
    settings = load_settings_from_yaml_string(yaml_text)
    bundle = instantiate_plugins_from_config(settings)
    graph = ExecutionGraph.from_plugin_instances(
        sources=bundle.sources,
        source_settings_map=bundle.source_settings_map,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        queues=settings.queues,
    )
    config = assemble_and_validate_pipeline_config(
        sources=bundle.sources,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    db = LandscapeDB(f"sqlite:///{tmp_path / db_name}")
    result = Orchestrator(db).run(
        config,
        graph=graph,
        settings=settings,
        payload_store=FilesystemPayloadStore(tmp_path / f"{db_name}-payloads"),
    )
    return _RunFixture(
        db=db,
        run_id=result.run_id,
        output_path=tmp_path / "out.jsonl",
        quarantine_path=tmp_path / "quarantine.jsonl",
    )


def _type_coerce_pipeline_yaml(
    *,
    source_a_path: Path,
    source_b_path: Path,
    output_path: Path,
    quarantine_path: Path,
) -> str:
    return f"""
sources:
  source_a:
    plugin: csv
    on_success: inbound
    options:
      path: {source_a_path}
      on_validation_failure: discard
      schema:
        mode: observed
  source_b:
    plugin: csv
    on_success: inbound
    options:
      path: {source_b_path}
      on_validation_failure: discard
      schema:
        mode: observed
queues:
  inbound: {{}}
transforms:
  - name: coerce_total
    plugin: type_coerce
    input: inbound
    on_success: output
    on_error: quarantine
    options:
      schema:
        mode: observed
      conversions:
        - field: total
          to: int
sinks:
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: {output_path}
      format: jsonl
      schema:
        mode: observed
  quarantine:
    plugin: json
    on_write_failure: discard
    options:
      path: {quarantine_path}
      format: jsonl
      schema:
        mode: observed
"""


def _outcomes_by_source(db: LandscapeDB, run_id: str) -> list[tuple[str, str, str, str | None]]:
    stmt = (
        select(
            run_sources_table.c.source_name,
            token_outcomes_table.c.outcome,
            token_outcomes_table.c.path,
            token_outcomes_table.c.sink_name,
        )
        .join(
            tokens_table,
            and_(
                tokens_table.c.run_id == token_outcomes_table.c.run_id,
                tokens_table.c.token_id == token_outcomes_table.c.token_id,
            ),
        )
        .join(
            rows_table,
            and_(
                rows_table.c.run_id == tokens_table.c.run_id,
                rows_table.c.row_id == tokens_table.c.row_id,
            ),
        )
        .join(
            run_sources_table,
            and_(
                run_sources_table.c.run_id == rows_table.c.run_id,
                run_sources_table.c.source_node_id == rows_table.c.source_node_id,
            ),
        )
        .where(token_outcomes_table.c.run_id == run_id)
        .order_by(run_sources_table.c.source_name, token_outcomes_table.c.path)
    )
    with db.connection() as conn:
        return [(row.source_name, row.outcome, row.path, row.sink_name) for row in conn.execute(stmt)]


def test_source_a_transform_failure_does_not_quarantine_source_b_rows(tmp_path: Path) -> None:
    source_a_path = tmp_path / "source-a.csv"
    source_b_path = tmp_path / "source-b.csv"
    output_path = tmp_path / "out.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    source_a_path.write_text("id,total\n1,not-an-int\n")
    source_b_path.write_text("id,total\n2,20\n")

    fixture = _run_yaml_pipeline(
        tmp_path,
        _type_coerce_pipeline_yaml(
            source_a_path=source_a_path,
            source_b_path=source_b_path,
            output_path=output_path,
            quarantine_path=quarantine_path,
        ),
    )

    output_rows = _read_jsonl(fixture.output_path)
    quarantine_rows = _read_jsonl(fixture.quarantine_path)
    assert len(output_rows) == 1
    assert output_rows[0]["total"] == 20
    assert len(quarantine_rows) == 1
    assert quarantine_rows[0]["total"] == "not-an-int"
    assert _outcomes_by_source(fixture.db, fixture.run_id) == [
        ("source_a", "failure", "on_error_routed", "quarantine"),
        ("source_b", "success", "default_flow", "output"),
    ]


def test_source_a_credentials_not_present_in_source_b_audit_payload(tmp_path: Path) -> None:
    secret_marker = "SOURCE_A_SECRET_TOKEN_DO_NOT_LEAK"
    source_a_path = tmp_path / f"source-a-{secret_marker}.csv"
    source_b_path = tmp_path / "source-b.csv"
    output_path = tmp_path / "out.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    source_a_path.write_text("id,total\n1,not-an-int\n")
    source_b_path.write_text("id,total\n2,20\n")

    fixture = _run_yaml_pipeline(
        tmp_path,
        _type_coerce_pipeline_yaml(
            source_a_path=source_a_path,
            source_b_path=source_b_path,
            output_path=output_path,
            quarantine_path=quarantine_path,
        ),
    )

    stmt = (
        select(run_sources_table.c.source_name, token_work_items_table.c.row_payload_json)
        .join(
            rows_table,
            and_(
                rows_table.c.run_id == token_work_items_table.c.run_id,
                rows_table.c.row_id == token_work_items_table.c.row_id,
            ),
        )
        .join(
            run_sources_table,
            and_(
                run_sources_table.c.run_id == rows_table.c.run_id,
                run_sources_table.c.source_node_id == rows_table.c.source_node_id,
            ),
        )
        .where(token_work_items_table.c.run_id == fixture.run_id)
        .where(run_sources_table.c.source_name == "source_b")
    )
    with fixture.db.connection() as conn:
        source_b_payloads = [row.row_payload_json for row in conn.execute(stmt)]

    assert source_b_payloads
    assert all(secret_marker not in payload for payload in source_b_payloads)


def test_source_a_oversize_row_does_not_starve_source_b_claim_ordering() -> None:
    db = make_landscape_db()
    factory = make_factory(db)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-isolation")
    schema_config = SchemaConfig.from_dict({"mode": "observed"})
    source_a = factory.data_flow.register_node(
        run_id=run.run_id,
        node_id="source-a",
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=schema_config,
    )
    source_b = factory.data_flow.register_node(
        run_id=run.run_id,
        node_id="source-b",
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        schema_config=schema_config,
    )
    transform = factory.data_flow.register_node(
        run_id=run.run_id,
        node_id="normalize",
        plugin_name="type_coerce",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        schema_config=schema_config,
    )
    row_a = factory.data_flow.create_row(
        run.run_id,
        source_a.node_id,
        0,
        {"payload": "x" * 8192},
        row_id="row-a",
        source_row_index=0,
        ingest_sequence=0,
    )
    row_b = factory.data_flow.create_row(
        run.run_id,
        source_b.node_id,
        1,
        {"payload": "ok"},
        row_id="row-b",
        source_row_index=0,
        ingest_sequence=1,
    )
    token_a = factory.data_flow.create_token(row_a.row_id, token_id="token-a")
    token_b = factory.data_flow.create_token(row_b.row_id, token_id="token-b")
    contract = SchemaContract(mode="OBSERVED", fields=(), locked=True)
    now = datetime.now(UTC)
    assert row_a.ingest_sequence is not None
    assert row_b.ingest_sequence is not None
    item_a = factory.scheduler.enqueue_ready(
        run_id=run.run_id,
        token_id=token_a.token_id,
        row_id=row_a.row_id,
        node_id=transform.node_id,
        step_index=1,
        ingest_sequence=row_a.ingest_sequence,
        available_at=now,
        row_payload_json=factory.scheduler.serialize_row_payload(PipelineRow({"payload": "x" * 8192}, contract)),
    )
    item_b = factory.scheduler.enqueue_ready(
        run_id=run.run_id,
        token_id=token_b.token_id,
        row_id=row_b.row_id,
        node_id=transform.node_id,
        step_index=1,
        ingest_sequence=row_b.ingest_sequence,
        available_at=now,
        row_payload_json=factory.scheduler.serialize_row_payload(PipelineRow({"payload": "ok"}, contract)),
    )

    claimed_a = factory.scheduler.claim_ready(run_id=run.run_id, lease_owner="worker-a", lease_seconds=30, now=now)
    claimed_b = factory.scheduler.claim_ready(run_id=run.run_id, lease_owner="worker-b", lease_seconds=30, now=now)

    assert claimed_a is not None
    assert claimed_b is not None
    assert claimed_a.work_item_id == item_a.work_item_id
    assert claimed_b.work_item_id == item_b.work_item_id
    assert claimed_a.row_id == "row-a"
    assert claimed_b.row_id == "row-b"


def test_source_a_quarantine_edge_routes_only_source_a_rows(tmp_path: Path) -> None:
    source_a_path = tmp_path / "source-a.csv"
    source_b_path = tmp_path / "source-b.csv"
    output_path = tmp_path / "out.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    source_a_path.write_text("id,total\nnot-an-int,10\n")
    source_b_path.write_text("id,total\n2,20\n")
    settings = f"""
sources:
  source_a:
    plugin: csv
    on_success: inbound
    options:
      path: {source_a_path}
      on_validation_failure: quarantine
      schema:
        mode: fixed
        fields: ["id: int", "total: int"]
  source_b:
    plugin: csv
    on_success: inbound
    options:
      path: {source_b_path}
      on_validation_failure: quarantine
      schema:
        mode: fixed
        fields: ["id: int", "total: int"]
queues:
  inbound: {{}}
transforms:
  - name: normalize_rows
    plugin: passthrough
    input: inbound
    on_success: output
    on_error: quarantine
    options:
      schema:
        mode: observed
sinks:
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: {output_path}
      format: jsonl
      schema:
        mode: observed
  quarantine:
    plugin: json
    on_write_failure: discard
    options:
      path: {quarantine_path}
      format: jsonl
      schema:
        mode: observed
"""

    fixture = _run_yaml_pipeline(tmp_path, settings)

    assert _read_jsonl(fixture.output_path) == [{"id": 2, "total": 20}]
    assert len(_read_jsonl(fixture.quarantine_path)) == 1

    stmt = (
        select(run_sources_table.c.source_name)
        .join(
            rows_table,
            and_(
                rows_table.c.run_id == run_sources_table.c.run_id,
                rows_table.c.source_node_id == run_sources_table.c.source_node_id,
            ),
        )
        .join(
            tokens_table,
            and_(
                tokens_table.c.run_id == rows_table.c.run_id,
                tokens_table.c.row_id == rows_table.c.row_id,
            ),
        )
        .join(
            node_states_table,
            and_(
                node_states_table.c.run_id == tokens_table.c.run_id,
                node_states_table.c.token_id == tokens_table.c.token_id,
            ),
        )
        .join(routing_events_table, routing_events_table.c.state_id == node_states_table.c.state_id)
        .join(edges_table, edges_table.c.edge_id == routing_events_table.c.edge_id)
        .where(node_states_table.c.run_id == fixture.run_id)
        .where(edges_table.c.label == "__quarantine__")
    )
    with fixture.db.connection() as conn:
        quarantine_sources = [row.source_name for row in conn.execute(stmt)]

    assert quarantine_sources == ["source_a"]
    assert _outcomes_by_source(fixture.db, fixture.run_id) == [
        ("source_a", "failure", "quarantined_at_source", "quarantine"),
        ("source_b", "success", "default_flow", "output"),
    ]
