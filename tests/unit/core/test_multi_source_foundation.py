"""Schema-first foundation tests for multi-source roots and token scheduling."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Literal

import pytest
from sqlalchemy import create_engine, event, insert, select
from sqlalchemy.exc import IntegrityError

from elspeth.contracts import NodeType, RoutingMode
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.core.canonical import canonical_json
from elspeth.core.config import SourceSettings, load_settings_from_yaml_string
from elspeth.core.dag import ExecutionGraph, GraphValidationError
from elspeth.core.landscape.schema import (
    metadata,
    nodes_table,
    rows_table,
    run_sources_table,
    runs_table,
    token_work_items_table,
    tokens_table,
)

_SchedulerTransition = Literal["waiting", "blocked", "terminal", "failed"]


def test_plural_sources_are_canonical_and_stable_named() -> None:
    settings = load_settings_from_yaml_string(
        """
sources:
  primary_csv:
    plugin: csv
    on_success: normalize
    options:
      path: data/a.csv
  enrichment_json:
    plugin: json
    on_success: normalize
    options:
      path: data/b.json
sinks:
  output:
    plugin: json
    on_write_failure: discard
transforms:
  - name: normalize_rows
    plugin: identity
    input: normalize
    on_success: output
    on_error: discard
"""
    )

    assert list(settings.sources) == ["primary_csv", "enrichment_json"]
    assert settings.sources["primary_csv"].plugin == "csv"
    assert settings.sources["enrichment_json"].on_success == "normalize"


def test_legacy_source_is_normalized_into_single_named_source() -> None:
    settings = load_settings_from_yaml_string(
        """
source:
  plugin: csv
  on_success: output
sinks:
  output:
    plugin: json
    on_write_failure: discard
"""
    )

    assert list(settings.sources) == ["source"]
    assert settings.source == settings.sources["source"]


def test_legacy_single_source_graph_preserves_rc52_node_identity() -> None:
    from tests.fixtures.plugins import CollectSink, ListSource

    source = ListSource([{"id": 1}], name="list_source", on_success="output")
    graph = ExecutionGraph.from_plugin_instances(
        source=source,
        source_settings=SourceSettings(plugin="list_source", on_success="output"),
        sinks={"output": CollectSink("output")},
    )

    source_id = graph.get_sources()[0]
    expected_hash = hashlib.sha256(canonical_json(source.config).encode()).hexdigest()[:12]

    assert str(source_id) == f"source_list_source_{expected_hash}"
    assert "source_name" not in graph.get_node_info(source_id).config


def test_explicit_named_sources_keep_source_name_in_identity_and_audit_config() -> None:
    from tests.fixtures.plugins import CollectSink, ListSource

    orders = ListSource([{"id": 1}], name="list_source", on_success="output")
    refunds = ListSource([{"id": 2}], name="list_source", on_success="output")
    graph = ExecutionGraph.from_plugin_instances(
        sources={"orders": orders, "refunds": refunds},
        source_settings_map={
            "orders": SourceSettings(plugin="list_source", on_success="output"),
            "refunds": SourceSettings(plugin="list_source", on_success="output"),
        },
        sinks={"output": CollectSink("output")},
    )

    source_ids = {str(source_id): graph.get_node_info(source_id).config for source_id in graph.get_sources()}

    assert len(source_ids) == 2
    assert all(source_id.startswith("source_") for source_id in source_ids)
    assert {config["source_name"] for config in source_ids.values()} == {"orders", "refunds"}


def test_plugin_bundle_instantiates_named_sources_via_production_path(tmp_path) -> None:
    orders_path = tmp_path / "orders.csv"
    refunds_path = tmp_path / "refunds.csv"
    output_path = tmp_path / "out.jsonl"
    orders_path.write_text("id,total\n1,10\n")
    refunds_path.write_text("id,total\n2,-3\n")

    settings = load_settings_from_yaml_string(
        f"""
sources:
  orders:
    plugin: csv
    on_success: inbound
    options:
      path: {orders_path}
      on_validation_failure: discard
      schema:
        mode: observed
  refunds:
    plugin: csv
    on_success: inbound
    options:
      path: {refunds_path}
      on_validation_failure: discard
      schema:
        mode: observed
queues:
  inbound: {{}}
transforms:
  - name: normalize_rows
    plugin: passthrough
    input: inbound
    on_success: output
    on_error: discard
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
"""
    )

    from elspeth.cli_helpers import instantiate_plugins_from_config

    bundle = instantiate_plugins_from_config(settings, preflight_mode=True)

    assert list(bundle.sources) == ["orders", "refunds"]
    assert list(bundle.source_settings_map) == ["orders", "refunds"]
    assert bundle.source is bundle.sources["orders"]
    assert bundle.source_settings == bundle.source_settings_map["orders"]
    with pytest.raises(TypeError):
        bundle.sources["late"] = bundle.sources["orders"]  # type: ignore[index]


def test_from_plugin_instances_builds_declared_queue_fan_in_via_production_path(tmp_path) -> None:
    orders_path = tmp_path / "orders.csv"
    refunds_path = tmp_path / "refunds.csv"
    output_path = tmp_path / "out.jsonl"
    orders_path.write_text("id,total\n1,10\n")
    refunds_path.write_text("id,total\n2,-3\n")

    settings = load_settings_from_yaml_string(
        f"""
sources:
  orders:
    plugin: csv
    on_success: inbound
    options:
      path: {orders_path}
      on_validation_failure: discard
      schema:
        mode: observed
  refunds:
    plugin: csv
    on_success: inbound
    options:
      path: {refunds_path}
      on_validation_failure: discard
      schema:
        mode: observed
queues:
  inbound:
    description: fan-in for scheduling only
transforms:
  - name: normalize_rows
    plugin: passthrough
    input: inbound
    on_success: output
    on_error: discard
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
"""
    )

    from elspeth.cli_helpers import instantiate_plugins_from_config

    bundle = instantiate_plugins_from_config(settings, preflight_mode=True)
    graph = ExecutionGraph.from_plugin_instances(
        sources=bundle.sources,
        source_settings_map=bundle.source_settings_map,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        queues=settings.queues,
    )

    graph.validate()
    queue_nodes = [node for node in graph.get_nodes() if node.node_type == NodeType.QUEUE]
    assert [node.plugin_name for node in queue_nodes] == ["queue:inbound"]

    queue_id = queue_nodes[0].node_id
    incoming_sources = {
        graph.get_node_info(edge.from_node).plugin_name
        for edge in graph.get_incoming_edges(queue_id)
        if graph.get_node_info(edge.from_node).node_type == NodeType.SOURCE
    }
    assert incoming_sources == {"csv"}
    assert len(graph.get_incoming_edges(queue_id)) == 2

    step_map = graph.build_step_map()
    assert {step_map[source_id] for source_id in graph.get_sources()} == {0}
    assert step_map[queue_id] == 1
    assert graph.get_node_info(graph.get_pipeline_node_sequence()[1]).plugin_name == "passthrough"


def test_pipeline_config_assembly_preserves_named_sources(tmp_path) -> None:
    orders_path = tmp_path / "orders.csv"
    refunds_path = tmp_path / "refunds.csv"
    output_path = tmp_path / "out.jsonl"
    orders_path.write_text("id,total\n1,10\n")
    refunds_path.write_text("id,total\n2,-3\n")

    settings = load_settings_from_yaml_string(
        f"""
sources:
  orders:
    plugin: csv
    on_success: inbound
    options:
      path: {orders_path}
      on_validation_failure: discard
      schema:
        mode: observed
  refunds:
    plugin: csv
    on_success: inbound
    options:
      path: {refunds_path}
      on_validation_failure: discard
      schema:
        mode: observed
queues:
  inbound: {{}}
transforms:
  - name: normalize_rows
    plugin: passthrough
    input: inbound
    on_success: output
    on_error: discard
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
"""
    )
    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config

    bundle = instantiate_plugins_from_config(settings, preflight_mode=True)
    graph = ExecutionGraph.from_plugin_instances(
        sources=bundle.sources,
        source_settings_map=bundle.source_settings_map,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        gates=list(settings.gates),
        queues=settings.queues,
    )

    pipeline_config = assemble_and_validate_pipeline_config(
        sources=bundle.sources,
        source=bundle.source,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )

    assert list(pipeline_config.sources) == ["orders", "refunds"]
    assert pipeline_config.source is pipeline_config.sources["orders"]


def test_graph_allows_multiple_source_roots_when_reachable() -> None:
    graph = ExecutionGraph()
    graph.add_node("source_a", node_type=NodeType.SOURCE, plugin_name="csv")
    graph.add_node("source_b", node_type=NodeType.SOURCE, plugin_name="json")
    graph.add_node("queue_ingest", node_type=NodeType.QUEUE, plugin_name="queue")
    graph.add_node("normalize", node_type=NodeType.TRANSFORM, plugin_name="identity")
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="json")
    graph.add_edge("source_a", "queue_ingest", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("source_b", "queue_ingest", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("queue_ingest", "normalize", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("normalize", "sink", label="on_success", mode=RoutingMode.MOVE)

    graph.validate()
    step_map = graph.build_step_map()

    assert step_map["source_a"] == 0
    assert step_map["source_b"] == 0
    assert step_map["queue_ingest"] == 1
    assert step_map["normalize"] == 2


def test_graph_rejects_fan_in_without_queue() -> None:
    graph = ExecutionGraph()
    graph.add_node("source_a", node_type=NodeType.SOURCE, plugin_name="csv")
    graph.add_node("source_b", node_type=NodeType.SOURCE, plugin_name="json")
    graph.add_node("normalize", node_type=NodeType.TRANSFORM, plugin_name="identity")
    graph.add_node("sink", node_type=NodeType.SINK, plugin_name="json")
    graph.add_edge("source_a", "normalize", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("source_b", "normalize", label="continue", mode=RoutingMode.MOVE)
    graph.add_edge("normalize", "sink", label="on_success", mode=RoutingMode.MOVE)

    with pytest.raises(GraphValidationError, match=r"fan-in.*queue"):
        graph.validate()


def test_run_sources_and_source_scoped_rows_are_schema_enforced() -> None:
    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    now = datetime.now(UTC)

    with engine.begin() as conn:
        conn.execute(
            insert(runs_table).values(
                run_id="run-1",
                started_at=now,
                config_hash="cfg",
                settings_json="{}",
                canonical_version="test",
                status="running",
                openrouter_catalog_sha256="0" * 64,
                openrouter_catalog_source="bundled",
            )
        )
        for source_node_id in ("source_a", "source_b"):
            conn.execute(
                insert(nodes_table).values(
                    node_id=source_node_id,
                    run_id="run-1",
                    plugin_name="csv",
                    node_type="source",
                    plugin_version="test",
                    determinism="deterministic",
                    config_hash=source_node_id,
                    config_json="{}",
                    registered_at=now,
                )
            )
            conn.execute(
                insert(run_sources_table).values(
                    run_id="run-1",
                    source_node_id=source_node_id,
                    source_name=source_node_id,
                    plugin_name="csv",
                    config_hash=source_node_id,
                    lifecycle_state="ready",
                    recorded_at=now,
                )
            )

        conn.execute(
            insert(rows_table).values(
                row_id="row-a-0",
                run_id="run-1",
                source_node_id="source_a",
                source_row_index=0,
                ingest_sequence=0,
                source_data_hash="hash-a",
                created_at=now,
            )
        )
        conn.execute(
            insert(rows_table).values(
                row_id="row-b-0",
                run_id="run-1",
                source_node_id="source_b",
                source_row_index=0,
                ingest_sequence=1,
                source_data_hash="hash-b",
                created_at=now,
            )
        )

        rows = conn.execute(select(rows_table.c.source_node_id, rows_table.c.source_row_index)).all()

    assert rows == [("source_a", 0), ("source_b", 0)]


def test_record_run_source_rejects_missing_source_node() -> None:
    from elspeth.core.landscape import LandscapeDB, RecorderFactory

    db = LandscapeDB.in_memory()
    factory = RecorderFactory(db)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="test", run_id="run-missing-source-node")

    with pytest.raises(AuditIntegrityError, match=r"run_sources.*source_node_id='missing-source'.*does not exist"):
        factory.run_lifecycle.record_run_source(
            run_id=run.run_id,
            source_node_id="missing-source",
            source_name="missing",
            plugin_name="csv",
            config_hash="cfg",
            lifecycle_state="ready",
        )


def test_record_run_source_rejects_non_source_node() -> None:
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.core.landscape import LandscapeDB, RecorderFactory

    db = LandscapeDB.in_memory()
    factory = RecorderFactory(db)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="test", run_id="run-non-source-node")
    factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="passthrough",
        node_type=NodeType.TRANSFORM,
        plugin_version="test",
        config={},
        node_id="transform-node",
        schema_config=SchemaConfig.from_dict({"mode": "observed"}),
    )

    with pytest.raises(
        AuditIntegrityError, match=r"run_sources.*source_node_id='transform-node'.*node_type='transform'.*expected 'source'"
    ):
        factory.run_lifecycle.record_run_source(
            run_id=run.run_id,
            source_node_id="transform-node",
            source_name="not_a_source",
            plugin_name="csv",
            config_hash="cfg",
            lifecycle_state="ready",
        )


def test_record_run_source_rejects_unknown_lifecycle_state() -> None:
    from elspeth.contracts.schema import SchemaConfig
    from elspeth.core.landscape import LandscapeDB, RecorderFactory

    db = LandscapeDB.in_memory()
    factory = RecorderFactory(db)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="test", run_id="run-invalid-source-lifecycle")
    factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="test",
        config={},
        node_id="source-node",
        schema_config=SchemaConfig.from_dict({"mode": "observed"}),
    )

    with pytest.raises(AuditIntegrityError, match=r"Invalid run source lifecycle_state='done'"):
        factory.run_lifecycle.record_run_source(
            run_id=run.run_id,
            source_node_id="source-node",
            source_name="orders",
            plugin_name="csv",
            config_hash="cfg",
            lifecycle_state="done",
        )


def test_run_sources_foreign_key_rejects_node_from_different_run() -> None:
    from sqlalchemy.exc import IntegrityError

    from elspeth.core.landscape import LandscapeDB

    db = LandscapeDB.in_memory()
    now = datetime.now(UTC)
    with db.engine.begin() as conn:
        for run_id in ("run-1", "run-2"):
            conn.execute(
                insert(runs_table).values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="cfg",
                    settings_json="{}",
                    canonical_version="test",
                    status="running",
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )
        conn.execute(
            insert(nodes_table).values(
                node_id="shared-source-id",
                run_id="run-2",
                plugin_name="csv",
                node_type="source",
                plugin_version="test",
                determinism="deterministic",
                config_hash="cfg",
                config_json="{}",
                registered_at=now,
            )
        )

        with pytest.raises(IntegrityError):
            conn.execute(
                insert(run_sources_table).values(
                    run_id="run-1",
                    source_node_id="shared-source-id",
                    source_name="source",
                    plugin_name="csv",
                    config_hash="cfg",
                    lifecycle_state="ready",
                    recorded_at=now,
                )
            )


def test_run_lifecycle_records_per_source_contract_and_resolution() -> None:
    from elspeth.contracts.schema_contract import FieldContract, SchemaContract
    from elspeth.core.landscape import LandscapeDB, RecorderFactory

    db = LandscapeDB("sqlite:///:memory:")
    factory = RecorderFactory(db)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="test")
    now = datetime.now(UTC)

    with db.connection() as conn:
        for source_node_id in ("source_orders", "source_refunds"):
            conn.execute(
                insert(nodes_table).values(
                    node_id=source_node_id,
                    run_id=run.run_id,
                    plugin_name="csv",
                    node_type="source",
                    plugin_version="test",
                    determinism="deterministic",
                    config_hash=source_node_id,
                    config_json="{}",
                    registered_at=now,
                )
            )

    contract = SchemaContract(
        mode="OBSERVED",
        fields=(
            FieldContract(
                normalized_name="id",
                original_name="Order ID",
                python_type=str,
                source="declared",
                nullable=False,
                required=True,
            ),
        ),
        locked=True,
    )

    factory.run_lifecycle.record_run_source(
        run_id=run.run_id,
        source_node_id="source_orders",
        source_name="orders",
        plugin_name="csv",
        config_hash="orders-hash",
        source_schema_json='{"title":"Orders"}',
        schema_contract=contract,
        field_resolution_mapping={"Order ID": "id"},
        normalization_version="v1",
        lifecycle_state="loaded",
    )
    factory.run_lifecycle.record_run_source(
        run_id=run.run_id,
        source_node_id="source_refunds",
        source_name="refunds",
        plugin_name="csv",
        config_hash="refunds-hash",
        source_schema_json='{"title":"Refunds"}',
        field_resolution_mapping={"Refund ID": "id"},
        normalization_version="v1",
        lifecycle_state="loaded",
    )

    with db.connection() as conn:
        rows = conn.execute(
            select(
                run_sources_table.c.source_name,
                run_sources_table.c.schema_contract_hash,
                run_sources_table.c.field_resolution_json,
            )
            .where(run_sources_table.c.run_id == run.run_id)
            .order_by(run_sources_table.c.source_name)
        ).all()

    assert [row.source_name for row in rows] == ["orders", "refunds"]
    assert rows[0].schema_contract_hash == contract.version_hash()
    assert '"Order ID":"id"' in rows[0].field_resolution_json
    assert '"Refund ID":"id"' in rows[1].field_resolution_json


def test_data_flow_create_row_accepts_source_row_index_and_ingest_sequence() -> None:
    from elspeth.core.landscape import LandscapeDB, RecorderFactory

    db = LandscapeDB("sqlite:///:memory:")
    factory = RecorderFactory(db)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="test")
    now = datetime.now(UTC)

    with db.connection() as conn:
        for source_node_id in ("source_orders", "source_refunds"):
            conn.execute(
                insert(nodes_table).values(
                    node_id=source_node_id,
                    run_id=run.run_id,
                    plugin_name="csv",
                    node_type="source",
                    plugin_version="test",
                    determinism="deterministic",
                    config_hash=source_node_id,
                    config_json="{}",
                    registered_at=now,
                )
            )

    factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id="source_orders",
        row_index=0,
        source_row_index=0,
        ingest_sequence=0,
        data={"id": "order-1"},
    )
    factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id="source_refunds",
        row_index=1,
        source_row_index=0,
        ingest_sequence=1,
        data={"id": "refund-1"},
    )

    with db.connection() as conn:
        rows = conn.execute(
            select(rows_table.c.source_node_id, rows_table.c.source_row_index, rows_table.c.ingest_sequence)
            .where(rows_table.c.run_id == run.run_id)
            .order_by(rows_table.c.ingest_sequence)
        ).all()

    assert rows == [("source_orders", 0, 0), ("source_refunds", 0, 1)]


def test_two_sources_feed_queue_end_to_end_with_source_scoped_rows(tmp_path) -> None:
    orders_path = tmp_path / "orders.csv"
    refunds_path = tmp_path / "refunds.csv"
    output_path = tmp_path / "out.jsonl"
    orders_path.write_text("id,total\n1,10\n2,20\n")
    refunds_path.write_text("id,total\nr1,-3\n")

    settings = load_settings_from_yaml_string(
        f"""
sources:
  orders:
    plugin: csv
    on_success: inbound
    options:
      path: {orders_path}
      on_validation_failure: discard
      schema:
        mode: observed
  refunds:
    plugin: csv
    on_success: inbound
    options:
      path: {refunds_path}
      on_validation_failure: discard
      schema:
        mode: observed
queues:
  inbound: {{}}
transforms:
  - name: normalize_rows
    plugin: passthrough
    input: inbound
    on_success: output
    on_error: discard
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
"""
    )

    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.payload_store import FilesystemPayloadStore
    from elspeth.engine.orchestrator import Orchestrator
    from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config

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
        source=bundle.source,
        sources=bundle.sources,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    result = Orchestrator(db).run(config, graph=graph, settings=settings, payload_store=FilesystemPayloadStore(tmp_path / "payloads"))

    assert result.rows_processed == 3
    with db.connection() as conn:
        rows = conn.execute(
            select(rows_table.c.source_node_id, rows_table.c.source_row_index, rows_table.c.ingest_sequence)
            .where(rows_table.c.run_id == result.run_id)
            .order_by(rows_table.c.ingest_sequence)
        ).all()
        source_records = conn.execute(
            select(run_sources_table.c.source_name, run_sources_table.c.lifecycle_state)
            .where(run_sources_table.c.run_id == result.run_id)
            .order_by(run_sources_table.c.source_name)
        ).all()
        scheduled_work = conn.execute(
            select(
                token_work_items_table.c.status,
                token_work_items_table.c.ingest_sequence,
                token_work_items_table.c.row_payload_json,
            )
            .where(token_work_items_table.c.run_id == result.run_id)
            .order_by(token_work_items_table.c.ingest_sequence)
        ).all()

    assert [row.source_row_index for row in rows] == [0, 1, 0]
    assert [row.ingest_sequence for row in rows] == [0, 1, 2]
    assert rows[0].source_node_id == rows[1].source_node_id
    assert rows[2].source_node_id != rows[0].source_node_id
    assert source_records == [("orders", "loaded"), ("refunds", "loaded")]
    assert [work.status for work in scheduled_work] == ["terminal", "terminal", "terminal"]
    assert [work.ingest_sequence for work in scheduled_work] == [0, 1, 2]
    scrubbed_payloads = [json.loads(work.row_payload_json) for work in scheduled_work]
    assert scrubbed_payloads == [
        {"payload_hash": scrubbed_payloads[0]["payload_hash"], "row_payload": "purged"},
        {"payload_hash": scrubbed_payloads[1]["payload_hash"], "row_payload": "purged"},
        {"payload_hash": scrubbed_payloads[2]["payload_hash"], "row_payload": "purged"},
    ]
    assert all("id" not in work.row_payload_json for work in scheduled_work)


def test_source_validation_failure_isolated_to_one_source(tmp_path) -> None:
    good_path = tmp_path / "good.csv"
    bad_path = tmp_path / "bad.csv"
    output_path = tmp_path / "out.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    good_path.write_text("id,total\n1,10\n")
    bad_path.write_text("id\nnot-an-int\n")

    settings = load_settings_from_yaml_string(
        f"""
sources:
  good:
    plugin: csv
    on_success: inbound
    options:
      path: {good_path}
      on_validation_failure: discard
      schema:
        mode: observed
  bad:
    plugin: csv
    on_success: inbound
    options:
      path: {bad_path}
      on_validation_failure: quarantine
      schema:
        mode: fixed
        fields: ["id: int"]
queues:
  inbound: {{}}
transforms:
  - name: normalize_rows
    plugin: passthrough
    input: inbound
    on_success: output
    on_error: discard
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
    )

    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.payload_store import FilesystemPayloadStore
    from elspeth.engine.orchestrator import Orchestrator
    from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config

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
        source=bundle.source,
        sources=bundle.sources,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    result = Orchestrator(db).run(config, graph=graph, settings=settings, payload_store=FilesystemPayloadStore(tmp_path / "payloads"))

    assert result.rows_processed == 2
    assert result.rows_succeeded == 1
    assert result.rows_quarantined == 1
    assert output_path.read_text().count("\n") == 1
    assert quarantine_path.read_text().count("\n") == 1
    with db.connection() as conn:
        source_rows = conn.execute(
            select(run_sources_table.c.source_name, rows_table.c.source_row_index)
            .join(
                rows_table,
                (rows_table.c.run_id == run_sources_table.c.run_id) & (rows_table.c.source_node_id == run_sources_table.c.source_node_id),
            )
            .where(rows_table.c.run_id == result.run_id)
            .order_by(rows_table.c.ingest_sequence)
        ).all()

    assert source_rows == [("good", 0), ("bad", 0)]


def test_two_independent_source_branches_end_to_end(tmp_path) -> None:
    orders_path = tmp_path / "orders.csv"
    refunds_path = tmp_path / "refunds.csv"
    orders_output = tmp_path / "orders.jsonl"
    refunds_output = tmp_path / "refunds.jsonl"
    orders_path.write_text("id,total\n1,10\n2,20\n")
    refunds_path.write_text("id,total\nr1,-3\n")

    settings = load_settings_from_yaml_string(
        f"""
sources:
  orders:
    plugin: csv
    on_success: orders_in
    options:
      path: {orders_path}
      on_validation_failure: discard
      schema:
        mode: observed
  refunds:
    plugin: csv
    on_success: refunds_in
    options:
      path: {refunds_path}
      on_validation_failure: discard
      schema:
        mode: observed
transforms:
  - name: normalize_orders
    plugin: passthrough
    input: orders_in
    on_success: orders_out
    on_error: discard
    options:
      schema:
        mode: observed
  - name: normalize_refunds
    plugin: passthrough
    input: refunds_in
    on_success: refunds_out
    on_error: discard
    options:
      schema:
        mode: observed
sinks:
  orders_out:
    plugin: json
    on_write_failure: discard
    options:
      path: {orders_output}
      format: jsonl
      schema:
        mode: observed
  refunds_out:
    plugin: json
    on_write_failure: discard
    options:
      path: {refunds_output}
      format: jsonl
      schema:
        mode: observed
"""
    )

    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.payload_store import FilesystemPayloadStore
    from elspeth.engine.orchestrator import Orchestrator
    from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config

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
        source=bundle.source,
        sources=bundle.sources,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")

    result = Orchestrator(db).run(config, graph=graph, settings=settings, payload_store=FilesystemPayloadStore(tmp_path / "payloads"))

    assert result.rows_processed == 3
    assert result.rows_succeeded == 3
    assert orders_output.read_text().count("\n") == 2
    assert refunds_output.read_text().count("\n") == 1
    with db.connection() as conn:
        source_to_sink = conn.execute(
            select(run_sources_table.c.source_name, rows_table.c.source_row_index)
            .join(
                rows_table,
                (rows_table.c.run_id == run_sources_table.c.run_id) & (rows_table.c.source_node_id == run_sources_table.c.source_node_id),
            )
            .where(rows_table.c.run_id == result.run_id)
            .order_by(rows_table.c.ingest_sequence)
        ).all()
    assert source_to_sink == [("orders", 0), ("orders", 1), ("refunds", 0)]


def test_source_only_multi_source_routes_each_source_to_its_own_sink(tmp_path) -> None:
    orders_path = tmp_path / "orders.csv"
    refunds_path = tmp_path / "refunds.csv"
    orders_output = tmp_path / "orders.jsonl"
    refunds_output = tmp_path / "refunds.jsonl"
    orders_path.write_text("id,total\n1,10\n2,20\n")
    refunds_path.write_text("id,total\nr1,-3\n")

    settings = load_settings_from_yaml_string(
        f"""
sources:
  orders:
    plugin: csv
    on_success: orders_out
    options:
      path: {orders_path}
      on_validation_failure: discard
      schema:
        mode: observed
  refunds:
    plugin: csv
    on_success: refunds_out
    options:
      path: {refunds_path}
      on_validation_failure: discard
      schema:
        mode: observed
sinks:
  orders_out:
    plugin: json
    on_write_failure: discard
    options:
      path: {orders_output}
      format: jsonl
      schema:
        mode: observed
  refunds_out:
    plugin: json
    on_write_failure: discard
    options:
      path: {refunds_output}
      format: jsonl
      schema:
        mode: observed
"""
    )

    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.payload_store import FilesystemPayloadStore
    from elspeth.engine.orchestrator import Orchestrator
    from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config

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
        source=bundle.source,
        sources=bundle.sources,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")

    result = Orchestrator(db).run(config, graph=graph, settings=settings, payload_store=FilesystemPayloadStore(tmp_path / "payloads"))

    assert result.rows_processed == 3
    assert result.rows_succeeded == 3
    orders_rows = [json.loads(line) for line in orders_output.read_text().splitlines()]
    refunds_rows = [json.loads(line) for line in refunds_output.read_text().splitlines()]
    assert orders_rows == [{"id": "1", "total": "10"}, {"id": "2", "total": "20"}]
    assert refunds_rows == [{"id": "r1", "total": "-3"}]


def test_mixed_direct_and_transformed_sources_route_to_their_own_sinks(tmp_path) -> None:
    direct_path = tmp_path / "direct.csv"
    transformed_path = tmp_path / "transformed.csv"
    direct_output = tmp_path / "direct.jsonl"
    transformed_output = tmp_path / "transformed.jsonl"
    direct_path.write_text("id,total\n1,10\n")
    transformed_path.write_text("id,total\nr1,-3\n")

    settings = load_settings_from_yaml_string(
        f"""
sources:
  direct_orders:
    plugin: csv
    on_success: direct_out
    options:
      path: {direct_path}
      on_validation_failure: discard
      schema:
        mode: observed
  transformed_refunds:
    plugin: csv
    on_success: refund_inbound
    options:
      path: {transformed_path}
      on_validation_failure: discard
      schema:
        mode: observed
queues:
  refund_inbound: {{}}
transforms:
  - name: normalize_refunds
    plugin: passthrough
    input: refund_inbound
    on_success: transformed_out
    on_error: discard
    options:
      schema:
        mode: observed
sinks:
  direct_out:
    plugin: json
    on_write_failure: discard
    options:
      path: {direct_output}
      format: jsonl
      schema:
        mode: observed
  transformed_out:
    plugin: json
    on_write_failure: discard
    options:
      path: {transformed_output}
      format: jsonl
      schema:
        mode: observed
"""
    )

    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.payload_store import FilesystemPayloadStore
    from elspeth.engine.orchestrator import Orchestrator
    from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config

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
        source=bundle.source,
        sources=bundle.sources,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")

    result = Orchestrator(db).run(config, graph=graph, settings=settings, payload_store=FilesystemPayloadStore(tmp_path / "payloads"))

    assert result.rows_processed == 2
    assert result.rows_succeeded == 2
    assert [json.loads(line) for line in direct_output.read_text().splitlines()] == [{"id": "1", "total": "10"}]
    assert [json.loads(line) for line in transformed_output.read_text().splitlines()] == [{"id": "r1", "total": "-3"}]


def test_multi_source_aggregation_flushes_once_after_all_sources_exhaust(tmp_path) -> None:
    orders_path = tmp_path / "orders.csv"
    refunds_path = tmp_path / "refunds.csv"
    output_path = tmp_path / "totals.jsonl"
    orders_path.write_text("id,amount\n1,10\n")
    refunds_path.write_text("id,amount\nr1,5\n")

    settings = load_settings_from_yaml_string(
        f"""
sources:
  orders:
    plugin: csv
    on_success: batch_in
    options:
      path: {orders_path}
      on_validation_failure: discard
      schema:
        mode: fixed
        fields:
          - "id: str"
          - "amount: int"
  refunds:
    plugin: csv
    on_success: batch_in
    options:
      path: {refunds_path}
      on_validation_failure: discard
      schema:
        mode: fixed
        fields:
          - "id: str"
          - "amount: int"
queues:
  batch_in: {{}}
aggregations:
  - name: total_amounts
    plugin: batch_stats
    input: batch_in
    on_success: output
    on_error: discard
    trigger:
      count: 3
    output_mode: transform
    options:
      schema:
        mode: observed
      value_field: amount
sinks:
  output:
    plugin: json
    on_write_failure: discard
    options:
      path: {output_path}
      format: jsonl
      schema:
        mode: observed
"""
    )

    from elspeth.cli_helpers import instantiate_plugins_from_config
    from elspeth.core.landscape import LandscapeDB
    from elspeth.core.payload_store import FilesystemPayloadStore
    from elspeth.engine.orchestrator import Orchestrator
    from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config

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
        source=bundle.source,
        sources=bundle.sources,
        transforms=bundle.transforms,
        sinks=bundle.sinks,
        aggregations=bundle.aggregations,
        settings=settings,
        graph=graph,
    )
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")

    result = Orchestrator(db).run(config, graph=graph, settings=settings, payload_store=FilesystemPayloadStore(tmp_path / "payloads"))

    assert result.rows_processed == 2
    assert result.rows_succeeded == 1
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [
        {
            "batch_size": 2,
            "count": 2,
            "mean": 7.5,
            "sum": 15,
        }
    ]


def test_scheduler_claims_ready_work_and_recovers_expired_leases() -> None:
    from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    contract = SchemaContract(
        mode="OBSERVED",
        fields=(
            FieldContract(
                normalized_name="id",
                original_name="id",
                python_type=int,
                required=True,
                source="declared",
            ),
            FieldContract(
                normalized_name="total",
                original_name="total",
                python_type=int,
                required=True,
                source="declared",
            ),
        ),
        locked=True,
    )
    payload = repo.serialize_row_payload(
        PipelineRow(
            {"id": 1, "total": 10},
            contract,
        )
    )
    _insert_scheduler_owner_records(engine, token_specs=(("token-1", "row-1", 0),), node_ids=("normalize",))

    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
        on_success_sink="default",
    )

    claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now)
    assert claimed is not None
    assert claimed.status is TokenWorkStatus.LEASED
    assert claimed.lease_owner == "worker-a"
    assert claimed.row_payload_json == item.row_payload_json
    assert claimed.on_success_sink == "default"
    restored = repo.deserialize_row_payload(claimed.row_payload_json)
    assert restored.to_dict() == {"id": 1, "total": 10}
    assert restored.contract.version_hash() == contract.version_hash()

    assert repo.claim_ready(run_id="run-1", lease_owner="worker-b", lease_seconds=30, now=now) is None

    recovered = repo.recover_expired_leases(run_id="run-1", now=now + timedelta(seconds=31))
    assert recovered == 1

    reclaimed = repo.claim_ready(run_id="run-1", lease_owner="worker-b", lease_seconds=30, now=now + timedelta(seconds=32))
    assert reclaimed is not None
    assert reclaimed.status is TokenWorkStatus.LEASED
    assert reclaimed.lease_owner == "worker-b"
    assert reclaimed.on_success_sink == "default"
    assert reclaimed.attempt == item.attempt + 1
    assert reclaimed.work_item_id != item.work_item_id


@pytest.mark.parametrize("transition", ["waiting", "blocked", "terminal", "failed"])
def test_scheduler_claimed_transition_rejects_stale_lease_owner_after_reclaim(transition: _SchedulerTransition) -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))
    _insert_scheduler_owner_records(engine, token_specs=(("token-1", "row-1", 0),), node_ids=("normalize",))

    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    first_claim = repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now)
    assert first_claim is not None
    assert first_claim.work_item_id == item.work_item_id

    recovered = repo.recover_expired_leases(run_id="run-1", now=now + timedelta(seconds=31))
    assert recovered == 1
    second_claim = repo.claim_ready(run_id="run-1", lease_owner="worker-b", lease_seconds=30, now=now + timedelta(seconds=32))
    assert second_claim is not None
    assert second_claim.work_item_id != item.work_item_id
    assert second_claim.attempt == item.attempt + 1

    with pytest.raises(AuditIntegrityError, match=r"expected lease_owner 'worker-a'.*actual lease_owner 'worker-b'"):
        _apply_scheduler_transition(
            repo,
            transition,
            work_item_id=second_claim.work_item_id,
            now=now + timedelta(seconds=33),
            expected_lease_owner="worker-a",
        )

    with engine.connect() as conn:
        row = conn.execute(
            select(token_work_items_table.c.status, token_work_items_table.c.lease_owner).where(
                token_work_items_table.c.work_item_id == second_claim.work_item_id
            )
        ).one()
    assert row == (TokenWorkStatus.LEASED.value, "worker-b")


@pytest.mark.parametrize(
    ("row_payload_json", "message"),
    [
        ("not json", "Corrupt scheduler row payload JSON"),
        ('"a string"', "expected object, got str"),
        ("[1, 2, 3]", "expected object, got list"),
        ("null", "expected object, got NoneType"),
        ('{"contract": {}}', "missing 'row'"),
        ('{"row": {}}', "missing 'contract'"),
        ('{"row": [], "contract": {}}', "row must be object, got list"),
        ('{"row": {}, "contract": "not-an-object"}', "contract must be object, got str"),
    ],
)
def test_scheduler_deserialize_row_payload_rejects_corrupt_payload_branches(row_payload_json: str, message: str) -> None:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    with pytest.raises(AuditIntegrityError, match=message):
        TokenSchedulerRepository.deserialize_row_payload(row_payload_json)


def test_scheduler_claim_ready_raises_when_selected_row_was_already_claimed() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))
    _insert_scheduler_owner_records(engine, token_specs=(("token-1", "row-1", 0),), node_ids=("normalize",))
    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    raced = False

    @event.listens_for(engine, "before_cursor_execute")
    def lease_selected_item_before_claim_update(conn, cursor, statement, parameters, context, executemany) -> None:  # type: ignore[no-untyped-def]
        nonlocal raced
        if raced or not statement.startswith("UPDATE token_work_items SET status="):
            return
        raced = True
        cursor.execute(
            "UPDATE token_work_items SET status = ?, lease_owner = ?, lease_expires_at = ?, updated_at = ? WHERE work_item_id = ?",
            (
                TokenWorkStatus.LEASED.value,
                "worker-racer",
                (now + timedelta(seconds=30)).isoformat(sep=" "),
                now.isoformat(sep=" "),
                item.work_item_id,
            ),
        )

    with pytest.raises(AuditIntegrityError, match=f"run_id='run-1'.*work_item_id='{item.work_item_id}'"):
        repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now)

    assert raced is True


def test_scheduler_claim_ready_two_workers_claim_distinct_items() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))
    _insert_scheduler_owner_records(
        engine,
        token_specs=(("token-1", "row-1", 0), ("token-2", "row-2", 1)),
        node_ids=("normalize",),
    )

    first = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    second = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-2",
        row_id="row-2",
        node_id="normalize",
        step_index=1,
        ingest_sequence=1,
        available_at=now,
        row_payload_json=payload,
    )

    claimed_a = repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now)
    claimed_b = repo.claim_ready(run_id="run-1", lease_owner="worker-b", lease_seconds=30, now=now)

    assert claimed_a is not None
    assert claimed_b is not None
    assert {claimed_a.work_item_id, claimed_b.work_item_id} == {first.work_item_id, second.work_item_id}
    assert claimed_a.work_item_id != claimed_b.work_item_id
    assert claimed_a.status is TokenWorkStatus.LEASED
    assert claimed_b.status is TokenWorkStatus.LEASED
    assert {claimed_a.lease_owner, claimed_b.lease_owner} == {"worker-a", "worker-b"}


def _apply_scheduler_transition(
    repo,
    transition: _SchedulerTransition,
    *,
    work_item_id: str,
    now: datetime,
    expected_lease_owner: str = "worker-a",
):
    if transition == "waiting":
        return repo.mark_waiting(
            work_item_id=work_item_id,
            available_at=now + timedelta(seconds=10),
            now=now,
            expected_lease_owner=expected_lease_owner,
        )
    if transition == "blocked":
        return repo.mark_blocked(
            work_item_id=work_item_id,
            queue_key="queue:inbound",
            barrier_key="barrier:row-1",
            now=now,
            expected_lease_owner=expected_lease_owner,
        )
    if transition == "terminal":
        return repo.mark_terminal(work_item_id=work_item_id, now=now, expected_lease_owner=expected_lease_owner)
    if transition == "failed":
        return repo.mark_failed(work_item_id=work_item_id, now=now, expected_lease_owner=expected_lease_owner)
    raise AssertionError(f"Unhandled scheduler transition {transition!r}")


def _insert_scheduler_owner_records(
    engine,
    *,
    run_id: str = "run-1",
    token_specs: tuple[tuple[str, str, int], ...] = (("token-1", "row-1", 0),),
    node_ids: tuple[str, ...] = ("normalize",),
    source_node_id: str = "source-0",
) -> None:
    now = datetime.now(UTC)
    with engine.begin() as conn:
        if conn.execute(select(runs_table.c.run_id).where(runs_table.c.run_id == run_id)).scalar_one_or_none() is None:
            conn.execute(
                insert(runs_table).values(
                    run_id=run_id,
                    started_at=now,
                    config_hash="config",
                    settings_json="{}",
                    canonical_version="v1",
                    status="running",
                    openrouter_catalog_sha256="0" * 64,
                    openrouter_catalog_source="bundled",
                )
            )
        registered_nodes = conn.execute(
            select(nodes_table.c.node_id).where(nodes_table.c.run_id == run_id, nodes_table.c.node_id.in_((source_node_id, *node_ids)))
        ).scalars()
        missing_nodes = {source_node_id, *node_ids} - set(registered_nodes)
        for node_id in sorted(missing_nodes):
            conn.execute(
                insert(nodes_table).values(
                    run_id=run_id,
                    node_id=node_id,
                    plugin_name="csv" if node_id == source_node_id else "identity",
                    node_type=NodeType.SOURCE.value if node_id == source_node_id else NodeType.TRANSFORM.value,
                    plugin_version="1.0",
                    determinism="deterministic",
                    config_hash="config",
                    config_json="{}",
                    registered_at=now,
                )
            )
        for token_id, row_id, ingest_sequence in token_specs:
            if conn.execute(select(rows_table.c.row_id).where(rows_table.c.row_id == row_id)).scalar_one_or_none() is None:
                conn.execute(
                    insert(rows_table).values(
                        row_id=row_id,
                        run_id=run_id,
                        source_node_id=source_node_id,
                        row_index=ingest_sequence,
                        source_row_index=ingest_sequence,
                        ingest_sequence=ingest_sequence,
                        source_data_hash=f"hash-{row_id}",
                        created_at=now,
                    )
                )
            if conn.execute(select(tokens_table.c.token_id).where(tokens_table.c.token_id == token_id)).scalar_one_or_none() is None:
                conn.execute(
                    insert(tokens_table).values(
                        token_id=token_id,
                        row_id=row_id,
                        run_id=run_id,
                        created_at=now,
                    )
                )


def _enqueue_scheduler_test_item(repo, *, engine, now: datetime, token_id: str = "token-1", ingest_sequence: int = 0):
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract

    metadata.create_all(engine)
    _insert_scheduler_owner_records(
        engine,
        token_specs=((token_id, f"row-{ingest_sequence + 1}", ingest_sequence),),
    )
    payload = repo.serialize_row_payload(PipelineRow({"id": ingest_sequence + 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))
    return repo.enqueue_ready(
        run_id="run-1",
        token_id=token_id,
        row_id=f"row-{ingest_sequence + 1}",
        node_id="normalize",
        step_index=1,
        ingest_sequence=ingest_sequence,
        available_at=now,
        row_payload_json=payload,
    )


def test_scheduler_repository_rejects_token_from_other_run() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    _insert_scheduler_owner_records(engine, run_id="run-A", token_specs=(), node_ids=("normalize",))
    _insert_scheduler_owner_records(engine, run_id="run-B", token_specs=(("token-cross", "row-cross", 0),), node_ids=())
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))

    with pytest.raises(AuditIntegrityError, match=r"token_id='token-cross'.*run_id='run-A'"):
        repo.enqueue_ready(
            run_id="run-A",
            token_id="token-cross",
            row_id="row-cross",
            node_id="normalize",
            step_index=1,
            ingest_sequence=0,
            available_at=now,
            row_payload_json=payload,
        )


def test_scheduler_repository_rejects_node_from_other_run() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    _insert_scheduler_owner_records(engine, run_id="run-A", token_specs=(("token-1", "row-1", 0),), node_ids=())
    _insert_scheduler_owner_records(engine, run_id="run-B", token_specs=(), node_ids=("normalize",))
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))

    with pytest.raises(AuditIntegrityError, match=r"node_id='normalize'.*run_id='run-A'"):
        repo.enqueue_ready(
            run_id="run-A",
            token_id="token-1",
            row_id="row-1",
            node_id="normalize",
            step_index=1,
            ingest_sequence=0,
            available_at=now,
            row_payload_json=payload,
        )


def test_scheduler_repository_rejects_ready_work_with_wrong_ingest_sequence() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    _insert_scheduler_owner_records(engine, token_specs=(("token-1", "row-1", 7),), node_ids=("normalize",))
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))

    with pytest.raises(AuditIntegrityError, match=r"row_id='row-1'.*ingest_sequence=7.*not scheduled ingest_sequence=8"):
        repo.enqueue_ready(
            run_id="run-1",
            token_id="token-1",
            row_id="row-1",
            node_id="normalize",
            step_index=1,
            ingest_sequence=8,
            available_at=now,
            row_payload_json=payload,
        )


def test_scheduler_repository_rejects_checkpoint_block_with_wrong_ingest_sequence() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    _insert_scheduler_owner_records(engine, token_specs=(("token-1", "row-1", 3),), node_ids=("coalesce_node",))
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))

    with pytest.raises(AuditIntegrityError, match=r"row_id='row-1'.*ingest_sequence=3.*not scheduled ingest_sequence=4"):
        repo.ensure_blocked_barrier_work_item(
            run_id="run-1",
            token_id="token-1",
            row_id="row-1",
            node_id="coalesce_node",
            step_index=1,
            ingest_sequence=4,
            available_at=now,
            row_payload_json=payload,
            barrier_key="coalesce:row-1",
            coalesce_node_id="coalesce_node",
            coalesce_name="join_orders",
        )


def test_scheduler_repository_allows_terminal_cursor_without_fake_node() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    _insert_scheduler_owner_records(engine, token_specs=(("token-terminal", "row-terminal", 0),), node_ids=())
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))

    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-terminal",
        row_id="row-terminal",
        node_id=None,
        step_index=99,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )

    assert item.node_id is None
    claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-terminal", lease_seconds=30, now=now)
    assert claimed is not None
    assert claimed.status is TokenWorkStatus.LEASED
    assert claimed.node_id is None


def test_scheduler_repository_idempotently_accepts_duplicate_enqueue_with_identical_cursor() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    _insert_scheduler_owner_records(engine, token_specs=(("token-1", "row-1", 0),), node_ids=("normalize",))
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(
        PipelineRow({"id": 1, "secret": "do-not-leak"}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
    )

    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )

    assert item.run_id == "run-1"
    assert item.token_id == "token-1"
    duplicate = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )

    assert duplicate == item


def test_scheduler_repository_rejects_duplicate_enqueue_with_incompatible_cursor() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.errors import LandscapeRecordError
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    _insert_scheduler_owner_records(engine, token_specs=(("token-1", "row-1", 0),), node_ids=("normalize",))
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(
        PipelineRow({"id": 1, "secret": "do-not-leak"}, SchemaContract(mode="OBSERVED", fields=(), locked=True))
    )
    repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )

    with pytest.raises(LandscapeRecordError) as exc_info:
        repo.enqueue_ready(
            run_id="run-1",
            token_id="token-1",
            row_id="row-1",
            node_id="normalize",
            step_index=2,
            ingest_sequence=0,
            available_at=now,
            row_payload_json=payload,
        )

    message = str(exc_info.value)
    assert "run_id='run-1'" in message
    assert "token_id='token-1'" in message
    assert "row_id='row-1'" in message
    assert "node_id='normalize'" in message
    assert "attempt=1" in message
    assert "do-not-leak" not in message


@pytest.mark.parametrize("transition", ["waiting", "blocked", "terminal", "failed"])
def test_scheduler_transitions_raise_for_missing_work_item(transition: _SchedulerTransition) -> None:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = create_engine("sqlite:///:memory:")
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    _enqueue_scheduler_test_item(repo, engine=engine, now=now)

    with pytest.raises(AuditIntegrityError, match=f"transition to .*work_item_id='missing-{transition}'"):
        _apply_scheduler_transition(repo, transition, work_item_id=f"missing-{transition}", now=now)


@pytest.mark.parametrize("transition", ["waiting", "blocked", "terminal", "failed"])
def test_scheduler_transitions_raise_when_work_item_already_in_target_status(transition: _SchedulerTransition) -> None:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = create_engine("sqlite:///:memory:")
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    item = _enqueue_scheduler_test_item(repo, engine=engine, now=now)
    claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now)
    assert claimed is not None

    _apply_scheduler_transition(repo, transition, work_item_id=item.work_item_id, now=now)

    with pytest.raises(AuditIntegrityError, match=f"transition to '{transition.upper()}'.*work_item_id='{item.work_item_id}'"):
        _apply_scheduler_transition(repo, transition, work_item_id=item.work_item_id, now=now + timedelta(seconds=1))


@pytest.mark.parametrize("transition", ["waiting", "blocked", "terminal"])
def test_scheduler_transitions_raise_when_work_item_is_not_leased(transition: _SchedulerTransition) -> None:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = create_engine("sqlite:///:memory:")
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    item = _enqueue_scheduler_test_item(repo, engine=engine, now=now)

    with pytest.raises(AuditIntegrityError, match=f"transition to '{transition.upper()}'.*expected status LEASED"):
        _apply_scheduler_transition(repo, transition, work_item_id=item.work_item_id, now=now)


def test_scheduler_marks_failed_clears_lease_and_blocks_reclaim() -> None:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    item = _enqueue_scheduler_test_item(repo, engine=engine, now=now)

    claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now)
    assert claimed is not None
    assert claimed.lease_owner == "worker-a"
    assert claimed.lease_expires_at is not None

    failed = repo.mark_failed(
        work_item_id=item.work_item_id,
        now=now + timedelta(seconds=1),
        expected_lease_owner="worker-a",
    )

    assert failed.status is TokenWorkStatus.FAILED
    assert failed.lease_owner is None
    assert failed.lease_expires_at is None
    assert repo.count_active_work(run_id="run-1") == 0
    assert repo.active_row_ids(run_id="run-1") == frozenset()
    assert repo.claim_ready(run_id="run-1", lease_owner="worker-b", lease_seconds=30, now=now + timedelta(seconds=2)) is None


def test_scheduler_marks_failed_from_ready() -> None:
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    item = _enqueue_scheduler_test_item(repo, engine=engine, now=now)

    failed = repo.mark_failed(work_item_id=item.work_item_id, now=now)

    assert failed.status is TokenWorkStatus.FAILED
    assert failed.lease_owner is None
    assert failed.lease_expires_at is None
    assert repo.count_active_work(run_id="run-1") == 0


def test_scheduler_requeues_waits_blocks_and_marks_terminal_with_leased_ownership() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))
    _insert_scheduler_owner_records(
        engine,
        token_specs=(("token-1", "row-1", 0), ("token-2", "row-2", 1)),
        node_ids=("normalize",),
    )

    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now)
    assert claimed is not None

    retry_at = now + timedelta(seconds=10)
    waiting = repo.mark_waiting(
        work_item_id=item.work_item_id,
        available_at=retry_at,
        now=now,
        expected_lease_owner="worker-a",
    )
    assert waiting.status is TokenWorkStatus.WAITING
    assert waiting.available_at == retry_at

    released = repo.release_waiting(run_id="run-1", now=retry_at)
    assert released == 1
    reclaimed = repo.claim_ready(run_id="run-1", lease_owner="worker-b", lease_seconds=30, now=retry_at)
    assert reclaimed is not None
    assert reclaimed.status is TokenWorkStatus.LEASED

    blocked = repo.mark_blocked(
        work_item_id=item.work_item_id,
        queue_key="queue:inbound",
        barrier_key="barrier:row-1",
        now=retry_at,
        expected_lease_owner="worker-b",
    )
    assert blocked.status is TokenWorkStatus.BLOCKED
    assert blocked.queue_key == "queue:inbound"
    assert blocked.barrier_key == "barrier:row-1"
    restarted_repo = TokenSchedulerRepository(engine)
    assert restarted_repo.count_active_work(run_id="run-1") == 1
    assert restarted_repo.active_row_ids(run_id="run-1") == frozenset({"row-1"})
    assert restarted_repo.claim_ready(run_id="run-1", lease_owner="worker-c", lease_seconds=30, now=retry_at) is None

    completed = repo.mark_blocked_barrier_terminal(
        run_id="run-1",
        barrier_key="barrier:row-1",
        token_ids=("token-1",),
        now=retry_at,
    )
    assert completed == 1
    assert restarted_repo.count_active_work(run_id="run-1") == 0
    assert restarted_repo.active_row_ids(run_id="run-1") == frozenset()

    second = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-2",
        row_id="row-2",
        node_id="normalize",
        step_index=1,
        ingest_sequence=1,
        available_at=retry_at,
        row_payload_json=payload,
    )
    claimed_second = repo.claim_ready(run_id="run-1", lease_owner="worker-d", lease_seconds=30, now=retry_at)
    assert claimed_second is not None
    assert claimed_second.work_item_id == second.work_item_id

    terminal = repo.mark_terminal(work_item_id=second.work_item_id, now=retry_at, expected_lease_owner="worker-d")
    assert terminal.status is TokenWorkStatus.TERMINAL
    with pytest.raises(AuditIntegrityError, match=f"work_item_id='{second.work_item_id}'"):
        repo.mark_terminal(
            work_item_id=second.work_item_id,
            now=retry_at + timedelta(seconds=1),
            expected_lease_owner="worker-d",
        )
    assert repo.claim_ready(run_id="run-1", lease_owner="worker-c", lease_seconds=30, now=retry_at) is None


def test_scheduler_barrier_completion_only_terminalizes_consumed_tokens() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))
    _insert_scheduler_owner_records(
        engine,
        token_specs=(("token-row-1-branch-a", "row-1", 0), ("token-row-2-branch-a", "row-2", 1)),
        node_ids=("coalesce_merge",),
    )
    first = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-row-1-branch-a",
        row_id="row-1",
        node_id="coalesce_merge",
        step_index=3,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    second = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-row-2-branch-a",
        row_id="row-2",
        node_id="coalesce_merge",
        step_index=3,
        ingest_sequence=1,
        available_at=now,
        row_payload_json=payload,
    )
    first_claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now)
    assert first_claimed is not None
    assert first_claimed.work_item_id == first.work_item_id
    repo.mark_blocked(
        work_item_id=first.work_item_id,
        queue_key=None,
        barrier_key="merge",
        now=now,
        expected_lease_owner="worker-a",
    )

    second_claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-b", lease_seconds=30, now=now)
    assert second_claimed is not None
    assert second_claimed.work_item_id == second.work_item_id
    repo.mark_blocked(
        work_item_id=second.work_item_id,
        queue_key=None,
        barrier_key="merge",
        now=now,
        expected_lease_owner="worker-b",
    )

    completed = repo.mark_blocked_barrier_terminal(
        run_id="run-1",
        barrier_key="merge",
        token_ids=("token-row-1-branch-a",),
        now=now,
    )

    assert completed == 1
    assert repo.count_active_work(run_id="run-1") == 1
    with engine.connect() as conn:
        rows = conn.execute(
            select(token_work_items_table.c.token_id, token_work_items_table.c.status)
            .where(token_work_items_table.c.run_id == "run-1")
            .order_by(token_work_items_table.c.ingest_sequence)
        ).all()
    assert rows == [
        ("token-row-1-branch-a", TokenWorkStatus.TERMINAL.value),
        ("token-row-2-branch-a", TokenWorkStatus.BLOCKED.value),
    ]


def test_scheduler_mark_blocked_rejects_missing_release_keys() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))
    _insert_scheduler_owner_records(engine, token_specs=(("token-1", "row-1", 0),), node_ids=("normalize",))
    item = repo.enqueue_ready(
        run_id="run-1",
        token_id="token-1",
        row_id="row-1",
        node_id="normalize",
        step_index=1,
        ingest_sequence=0,
        available_at=now,
        row_payload_json=payload,
    )
    claimed = repo.claim_ready(run_id="run-1", lease_owner="worker-a", lease_seconds=30, now=now)
    assert claimed is not None

    with pytest.raises(AuditIntegrityError, match=rf"work_item_id='{item.work_item_id}'.*queue_key.*barrier_key"):
        repo.mark_blocked(
            work_item_id=item.work_item_id,
            queue_key=None,
            barrier_key=None,
            now=now,
            expected_lease_owner="worker-a",
        )


def _scheduler_work_values(
    *,
    work_item_id: str,
    run_id: str,
    token_id: str,
    row_id: str,
    node_id: str | None,
    now: datetime,
    coalesce_node_id: str | None = None,
) -> dict[str, object]:
    return {
        "work_item_id": work_item_id,
        "run_id": run_id,
        "token_id": token_id,
        "row_id": row_id,
        "node_id": node_id,
        "step_index": 1,
        "ingest_sequence": 0,
        "row_payload_json": "{}",
        "status": "ready",
        "attempt": 1,
        "available_at": now,
        "created_at": now,
        "updated_at": now,
        "coalesce_node_id": coalesce_node_id,
    }


def test_scheduler_schema_rejects_cross_run_token_reference() -> None:
    from sqlalchemy.exc import IntegrityError

    from elspeth.contracts import NodeType
    from elspeth.contracts.schema import SchemaConfig
    from tests.fixtures.landscape import make_factory, make_landscape_db

    db = make_landscape_db()
    factory = make_factory(db)
    schema_config = SchemaConfig.from_dict({"mode": "observed"})
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-A")
    factory.data_flow.register_node(
        run_id="run-A",
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-0",
        schema_config=schema_config,
    )
    row_a = factory.data_flow.create_row(
        run_id="run-A",
        source_node_id="source-0",
        row_index=0,
        source_row_index=0,
        ingest_sequence=0,
        data={"value": 1},
    )
    token_a = factory.data_flow.create_token(row_a.row_id)

    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")
    factory.data_flow.register_node(
        run_id="run-B",
        plugin_name="normalize",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        node_id="normalize",
        schema_config=schema_config,
    )
    now = datetime.now(UTC)

    with pytest.raises(IntegrityError), db.connection() as conn:
        conn.execute(
            token_work_items_table.insert().values(
                **_scheduler_work_values(
                    work_item_id="cross-run-token",
                    run_id="run-B",
                    token_id=token_a.token_id,
                    row_id=row_a.row_id,
                    node_id="normalize",
                    now=now,
                )
            )
        )


def test_scheduler_schema_rejects_cross_run_node_reference() -> None:
    from sqlalchemy.exc import IntegrityError

    from elspeth.contracts import NodeType
    from elspeth.contracts.schema import SchemaConfig
    from tests.fixtures.landscape import make_factory, make_landscape_db

    db = make_landscape_db()
    factory = make_factory(db)
    schema_config = SchemaConfig.from_dict({"mode": "observed"})
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-A")
    factory.data_flow.register_node(
        run_id="run-A",
        plugin_name="normalize",
        node_type=NodeType.TRANSFORM,
        plugin_version="1.0",
        config={},
        node_id="normalize",
        schema_config=schema_config,
    )
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-B")
    factory.data_flow.register_node(
        run_id="run-B",
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-0",
        schema_config=schema_config,
    )
    row_b = factory.data_flow.create_row(
        run_id="run-B",
        source_node_id="source-0",
        row_index=0,
        source_row_index=0,
        ingest_sequence=0,
        data={"value": 1},
    )
    token_b = factory.data_flow.create_token(row_b.row_id)
    now = datetime.now(UTC)

    with pytest.raises(IntegrityError), db.connection() as conn:
        conn.execute(
            token_work_items_table.insert().values(
                **_scheduler_work_values(
                    work_item_id="cross-run-node",
                    run_id="run-B",
                    token_id=token_b.token_id,
                    row_id=row_b.row_id,
                    node_id="normalize",
                    now=now,
                )
            )
        )


def test_scheduler_schema_allows_null_node_for_terminal_cursor() -> None:
    from elspeth.contracts import NodeType
    from elspeth.contracts.schema import SchemaConfig
    from tests.fixtures.landscape import make_factory, make_landscape_db

    db = make_landscape_db()
    factory = make_factory(db)
    schema_config = SchemaConfig.from_dict({"mode": "observed"})
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
    factory.data_flow.register_node(
        run_id="run-1",
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-0",
        schema_config=schema_config,
    )
    row = factory.data_flow.create_row(
        run_id="run-1",
        source_node_id="source-0",
        row_index=0,
        source_row_index=0,
        ingest_sequence=0,
        data={"value": 1},
    )
    token = factory.data_flow.create_token(row.row_id)
    now = datetime.now(UTC)

    with db.connection() as conn:
        conn.execute(
            token_work_items_table.insert().values(
                **_scheduler_work_values(
                    work_item_id="terminal-cursor",
                    run_id="run-1",
                    token_id=token.token_id,
                    row_id=row.row_id,
                    node_id=None,
                    now=now,
                )
            )
        )


def test_scheduler_schema_rejects_duplicate_null_node_terminal_identity() -> None:
    from elspeth.contracts import NodeType
    from elspeth.contracts.schema import SchemaConfig
    from tests.fixtures.landscape import make_factory, make_landscape_db

    db = make_landscape_db()
    factory = make_factory(db)
    schema_config = SchemaConfig.from_dict({"mode": "observed"})
    factory.run_lifecycle.begin_run(config={}, canonical_version="v1", run_id="run-1")
    factory.data_flow.register_node(
        run_id="run-1",
        plugin_name="csv",
        node_type=NodeType.SOURCE,
        plugin_version="1.0",
        config={},
        node_id="source-0",
        schema_config=schema_config,
    )
    row = factory.data_flow.create_row(
        run_id="run-1",
        source_node_id="source-0",
        row_index=0,
        source_row_index=0,
        ingest_sequence=0,
        data={"value": 1},
    )
    token = factory.data_flow.create_token(row.row_id)
    now = datetime.now(UTC)

    with db.connection() as conn:
        conn.execute(
            token_work_items_table.insert().values(
                **_scheduler_work_values(
                    work_item_id="terminal-cursor-a",
                    run_id="run-1",
                    token_id=token.token_id,
                    row_id=row.row_id,
                    node_id=None,
                    now=now,
                )
            )
        )
        with pytest.raises(IntegrityError):
            conn.execute(
                token_work_items_table.insert().values(
                    **_scheduler_work_values(
                        work_item_id="terminal-cursor-b",
                        run_id="run-1",
                        token_id=token.token_id,
                        row_id=row.row_id,
                        node_id=None,
                        now=now,
                    )
                )
            )


def test_scheduler_barrier_terminal_raises_when_live_tokens_missing_from_durable_blocked_rows() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))
    _insert_scheduler_owner_records(
        engine,
        token_specs=(("token-a", "row-0", 0), ("token-b", "row-1", 1)),
        node_ids=("coalesce_merge",),
    )

    for index, token_id in enumerate(("token-a", "token-b")):
        item = repo.enqueue_ready(
            run_id="run-1",
            token_id=token_id,
            row_id=f"row-{index}",
            node_id="coalesce_merge",
            step_index=3,
            ingest_sequence=index,
            available_at=now,
            row_payload_json=payload,
        )
        claimed = repo.claim_ready(run_id="run-1", lease_owner=f"worker-{index}", lease_seconds=30, now=now)
        assert claimed is not None
        assert claimed.work_item_id == item.work_item_id
        repo.mark_blocked(
            work_item_id=item.work_item_id,
            queue_key=None,
            barrier_key="merge",
            now=now,
            expected_lease_owner=f"worker-{index}",
        )

    with pytest.raises(AuditIntegrityError, match=r"live consumed 3 token.*durable BLOCKED rows.*2 matching.*missing token_ids.*token-c"):
        repo.mark_blocked_barrier_terminal(
            run_id="run-1",
            barrier_key="merge",
            token_ids=("token-a", "token-b", "token-c"),
            now=now,
        )

    with engine.connect() as conn:
        statuses = conn.execute(
            select(token_work_items_table.c.token_id, token_work_items_table.c.status)
            .where(token_work_items_table.c.run_id == "run-1")
            .order_by(token_work_items_table.c.token_id)
        ).all()
    assert statuses == [
        ("token-a", TokenWorkStatus.BLOCKED.value),
        ("token-b", TokenWorkStatus.BLOCKED.value),
    ]


def test_scheduler_barrier_terminal_raises_when_durable_blocked_token_set_is_disjoint() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))
    _insert_scheduler_owner_records(
        engine,
        token_specs=(("durable-a", "row-0", 0), ("durable-b", "row-1", 1), ("durable-c", "row-2", 2)),
        node_ids=("coalesce_merge",),
    )

    for index, token_id in enumerate(("durable-a", "durable-b", "durable-c")):
        item = repo.enqueue_ready(
            run_id="run-1",
            token_id=token_id,
            row_id=f"row-{index}",
            node_id="coalesce_merge",
            step_index=3,
            ingest_sequence=index,
            available_at=now,
            row_payload_json=payload,
        )
        claimed = repo.claim_ready(run_id="run-1", lease_owner=f"worker-{index}", lease_seconds=30, now=now)
        assert claimed is not None
        assert claimed.work_item_id == item.work_item_id
        repo.mark_blocked(
            work_item_id=item.work_item_id,
            queue_key=None,
            barrier_key="merge",
            now=now,
            expected_lease_owner=f"worker-{index}",
        )

    with pytest.raises(AuditIntegrityError, match=r"live consumed 3 token.*0 matching.*missing token_ids.*live-a"):
        repo.mark_blocked_barrier_terminal(
            run_id="run-1",
            barrier_key="merge",
            token_ids=("live-a", "live-b", "live-c"),
            now=now,
        )

    with engine.connect() as conn:
        statuses = conn.execute(
            select(token_work_items_table.c.token_id, token_work_items_table.c.status)
            .where(token_work_items_table.c.run_id == "run-1")
            .order_by(token_work_items_table.c.token_id)
        ).all()
    assert statuses == [
        ("durable-a", TokenWorkStatus.BLOCKED.value),
        ("durable-b", TokenWorkStatus.BLOCKED.value),
        ("durable-c", TokenWorkStatus.BLOCKED.value),
    ]


def test_scheduler_barrier_terminal_rejects_empty_live_token_set() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))
    _insert_scheduler_owner_records(
        engine,
        token_specs=(("token-a", "row-0", 0), ("token-b", "row-1", 1)),
        node_ids=("coalesce_merge",),
    )

    for index, token_id in enumerate(("token-a", "token-b")):
        item = repo.enqueue_ready(
            run_id="run-1",
            token_id=token_id,
            row_id=f"row-{index}",
            node_id="coalesce_merge",
            step_index=3,
            ingest_sequence=index,
            available_at=now,
            row_payload_json=payload,
        )
        claimed = repo.claim_ready(run_id="run-1", lease_owner=f"worker-{index}", lease_seconds=30, now=now)
        assert claimed is not None
        assert claimed.work_item_id == item.work_item_id
        repo.mark_blocked(
            work_item_id=item.work_item_id,
            queue_key=None,
            barrier_key="merge",
            now=now,
            expected_lease_owner=f"worker-{index}",
        )

    with pytest.raises(AuditIntegrityError, match="requires at least one live token_id"):
        repo.mark_blocked_barrier_terminal(
            run_id="run-1",
            barrier_key="merge",
            token_ids=(),
            now=now,
        )

    with engine.connect() as conn:
        statuses = conn.execute(
            select(token_work_items_table.c.token_id, token_work_items_table.c.status)
            .where(token_work_items_table.c.run_id == "run-1")
            .order_by(token_work_items_table.c.token_id)
        ).all()
    assert statuses == [
        ("token-a", TokenWorkStatus.BLOCKED.value),
        ("token-b", TokenWorkStatus.BLOCKED.value),
    ]


def test_checkpoint_restore_rejects_existing_blocked_work_with_stale_resume_payload() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    _insert_scheduler_owner_records(
        engine,
        token_specs=(("token-a", "row-0", 0),),
        node_ids=("aggregate", "coalesce_node"),
    )
    contract = SchemaContract(mode="OBSERVED", fields=(), locked=True)
    original_payload = repo.serialize_row_payload(PipelineRow({"id": 1}, contract))
    restored_payload = repo.serialize_row_payload(PipelineRow({"id": 2}, contract))

    repo.ensure_blocked_barrier_work_item(
        run_id="run-1",
        token_id="token-a",
        row_id="row-0",
        node_id="aggregate",
        step_index=4,
        ingest_sequence=0,
        row_payload_json=original_payload,
        barrier_key="aggregate",
        available_at=now,
        on_success_sink="first_sink",
        branch_name="branch-a",
        fork_group_id="fork-a",
        join_group_id="join-a",
        expand_group_id="expand-a",
        coalesce_node_id="coalesce_node",
        coalesce_name="merge-a",
    )

    with pytest.raises(
        AuditIntegrityError,
        match=r"stale existing work_item_id=.*row_payload_json.*on_success_sink.*coalesce_name",
    ):
        repo.ensure_blocked_barrier_work_item(
            run_id="run-1",
            token_id="token-a",
            row_id="row-0",
            node_id="aggregate",
            step_index=5,
            ingest_sequence=0,
            row_payload_json=restored_payload,
            barrier_key="aggregate",
            available_at=now + timedelta(seconds=1),
            on_success_sink="second_sink",
            branch_name="branch-b",
            fork_group_id="fork-b",
            join_group_id="join-b",
            expand_group_id="expand-b",
            coalesce_node_id="coalesce_node",
            coalesce_name="merge-b",
        )
