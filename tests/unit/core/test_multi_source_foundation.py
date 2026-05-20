"""Schema-first foundation tests for multi-source roots and token scheduling."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, insert, select

from elspeth.contracts import NodeType, RoutingMode
from elspeth.core.config import load_settings_from_yaml_string
from elspeth.core.dag import ExecutionGraph, GraphValidationError
from elspeth.core.landscape.schema import metadata, nodes_table, rows_table, run_sources_table, runs_table, token_work_items_table


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

    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository

    assert [row.source_row_index for row in rows] == [0, 1, 0]
    assert [row.ingest_sequence for row in rows] == [0, 1, 2]
    assert rows[0].source_node_id == rows[1].source_node_id
    assert rows[2].source_node_id != rows[0].source_node_id
    assert source_records == [("orders", "loaded"), ("refunds", "loaded")]
    assert [work.status for work in scheduled_work] == ["terminal", "terminal", "terminal"]
    assert [work.ingest_sequence for work in scheduled_work] == [0, 1, 2]
    assert [TokenSchedulerRepository.deserialize_row_payload(work.row_payload_json).to_dict()["id"] for work in scheduled_work] == [
        "1",
        "2",
        "r1",
    ]


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
    assert claimed.status is TokenWorkStatus.LEASED
    assert claimed.lease_owner == "worker-a"
    assert claimed.row_payload_json == item.row_payload_json
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


def test_scheduler_requeues_waits_blocks_and_marks_terminal_idempotently() -> None:
    from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
    from elspeth.core.landscape.scheduler_repository import TokenSchedulerRepository, TokenWorkStatus

    engine = create_engine("sqlite:///:memory:")
    metadata.create_all(engine)
    repo = TokenSchedulerRepository(engine)
    now = datetime.now(UTC)
    payload = repo.serialize_row_payload(PipelineRow({"id": 1}, SchemaContract(mode="OBSERVED", fields=(), locked=True)))

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
    )
    assert blocked.status is TokenWorkStatus.BLOCKED
    assert blocked.queue_key == "queue:inbound"
    assert blocked.barrier_key == "barrier:row-1"
    restarted_repo = TokenSchedulerRepository(engine)
    assert restarted_repo.count_active_work(run_id="run-1") == 1
    assert restarted_repo.claim_ready(run_id="run-1", lease_owner="worker-c", lease_seconds=30, now=retry_at) is None

    terminal = repo.mark_terminal(work_item_id=item.work_item_id, now=retry_at)
    terminal_again = repo.mark_terminal(work_item_id=item.work_item_id, now=retry_at + timedelta(seconds=1))
    assert terminal.status is TokenWorkStatus.TERMINAL
    assert terminal_again.status is TokenWorkStatus.TERMINAL
    assert repo.claim_ready(run_id="run-1", lease_owner="worker-c", lease_seconds=30, now=retry_at) is None
