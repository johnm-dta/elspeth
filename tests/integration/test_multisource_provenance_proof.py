"""C1 proofs: multi-source entry with durable per-source provenance.

Drives the REAL orchestrator (real SQLite LandscapeDB, real scheduler, real
plugin instantiation from YAML settings) and asserts ADR-025 semantics:

1. Multi-entry: two registered sources feed one pipeline end-to-end and both
   sources' rows arrive in the sink with matching counts.
2. Provenance durability: every row/token/node_state in the audit database
   attributes to its originating source node; ``source_row_index`` is
   per-source (restarts at 0 per source) while ``ingest_sequence`` is the
   globally-unique cross-source ordering primitive (ADR-026); ``run_sources``
   lifecycle rows reach their terminal state for both sources.
3. Per-source contracts: each source's fixed schema contract validates only
   its own rows — distinct contracts on two sources do not bleed across
   source boundaries (ADR-025 mixed-contract pipelines are first-class).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select

from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import ElspethSettings, load_settings_from_yaml_string
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape import LandscapeDB
from elspeth.core.landscape.schema import (
    node_states_table,
    nodes_table,
    rows_table,
    run_sources_table,
    token_work_items_table,
    tokens_table,
)
from elspeth.core.payload_store import FilesystemPayloadStore
from elspeth.engine.orchestrator import Orchestrator
from elspeth.engine.orchestrator.preflight import assemble_and_validate_pipeline_config


def _run_pipeline(settings: ElspethSettings, tmp_path: Path):
    """Instantiate plugins from settings and execute via the production path."""
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
    db = LandscapeDB(f"sqlite:///{tmp_path / 'audit.db'}")
    result = Orchestrator(db).run(
        config,
        graph=graph,
        settings=settings,
        payload_store=FilesystemPayloadStore(tmp_path / "payloads"),
    )
    return result, db


def _two_source_settings(tmp_path: Path) -> tuple[ElspethSettings, Path]:
    """3-row orders source + 2-row refunds source fanning into one queue."""
    orders_path = tmp_path / "orders.csv"
    refunds_path = tmp_path / "refunds.csv"
    output_path = tmp_path / "out.jsonl"
    orders_path.write_text("id,origin\no1,orders\no2,orders\no3,orders\n")
    refunds_path.write_text("id,origin\nr1,refunds\nr2,refunds\n")

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
    return settings, output_path


@pytest.mark.timeout(60)
def test_two_registered_sources_both_deliver_rows_to_sink(tmp_path: Path) -> None:
    """C1 multi-entry: rows from both registered sources reach the sink, counts match."""
    settings, output_path = _two_source_settings(tmp_path)
    result, _db = _run_pipeline(settings, tmp_path)

    assert result.rows_processed == 5
    assert result.rows_succeeded == 5

    sink_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(sink_rows) == 5
    by_origin: dict[str, set[str]] = {}
    for row in sink_rows:
        by_origin.setdefault(row["origin"], set()).add(row["id"])
    assert by_origin == {
        "orders": {"o1", "o2", "o3"},
        "refunds": {"r1", "r2"},
    }


@pytest.mark.timeout(60)
def test_per_source_provenance_is_durable_in_audit_database(tmp_path: Path) -> None:
    """C1 provenance: every row/token/node_state attributes to its source node;
    source_row_index restarts per source, ingest_sequence is globally monotone,
    and run_sources lifecycle reaches its terminal state for both sources."""
    settings, _output_path = _two_source_settings(tmp_path)
    result, db = _run_pipeline(settings, tmp_path)
    run_id = result.run_id

    with db.connection() as conn:
        source_nodes = conn.execute(
            select(nodes_table.c.node_id).where(nodes_table.c.run_id == run_id).where(nodes_table.c.node_type == "source")
        ).all()
        run_sources = conn.execute(
            select(
                run_sources_table.c.source_name,
                run_sources_table.c.source_node_id,
                run_sources_table.c.lifecycle_state,
                run_sources_table.c.schema_contract_json,
            )
            .where(run_sources_table.c.run_id == run_id)
            .order_by(run_sources_table.c.source_name)
        ).all()
        rows = conn.execute(
            select(
                rows_table.c.row_id,
                rows_table.c.source_node_id,
                rows_table.c.source_row_index,
                rows_table.c.ingest_sequence,
            )
            .where(rows_table.c.run_id == run_id)
            .order_by(rows_table.c.ingest_sequence)
        ).all()
        tokens = conn.execute(select(tokens_table.c.token_id, tokens_table.c.row_id).where(tokens_table.c.run_id == run_id)).all()
        node_state_tokens = conn.execute(select(node_states_table.c.token_id).where(node_states_table.c.run_id == run_id)).all()
        work_items = conn.execute(
            select(
                token_work_items_table.c.status,
                token_work_items_table.c.ingest_sequence,
            )
            .where(token_work_items_table.c.run_id == run_id)
            .order_by(token_work_items_table.c.ingest_sequence)
        ).all()

    # Two distinct source nodes registered, and run_sources covers exactly those.
    source_node_ids = {node.node_id for node in source_nodes}
    assert len(source_node_ids) == 2
    assert {rec.source_node_id for rec in run_sources} == source_node_ids
    assert [rec.source_name for rec in run_sources] == ["orders", "refunds"]

    # run_sources lifecycle reached its terminal state for BOTH sources, and
    # each source carries its own persisted schema contract.
    assert [rec.lifecycle_state for rec in run_sources] == ["exhausted", "exhausted"]
    assert all(rec.schema_contract_json for rec in run_sources)

    # Every row attributes to a registered source node (NOT NULL is structural;
    # this proves the values are the real source nodes, not placeholders).
    assert len(rows) == 5
    assert all(row.source_node_id in source_node_ids for row in rows)

    # ingest_sequence is the global cross-source ordering primitive (ADR-026):
    # gapless, monotone, declaration-ordered (orders rows before refunds rows).
    assert [row.ingest_sequence for row in rows] == [0, 1, 2, 3, 4]
    orders_node_id = run_sources[0].source_node_id
    refunds_node_id = run_sources[1].source_node_id
    assert [row.source_node_id for row in rows] == [orders_node_id] * 3 + [refunds_node_id] * 2

    # source_row_index is per-source: restarts at 0 for each source (ADR-025).
    per_source_indices: dict[str, list[int]] = {}
    for row in rows:
        per_source_indices.setdefault(row.source_node_id, []).append(row.source_row_index)
    assert per_source_indices == {
        orders_node_id: [0, 1, 2],
        refunds_node_id: [0, 1],
    }

    # Every token joins back to a provenance-bearing row (token -> row ->
    # source_node_id is the durable attribution path; tokens carry no
    # source column of their own).
    row_source_by_id = {row.row_id: row.source_node_id for row in rows}
    assert len(tokens) >= len(rows)
    for token in tokens:
        assert token.row_id in row_source_by_id
        assert row_source_by_id[token.row_id] in source_node_ids

    # Every node_state joins back through its token to an attributed row.
    token_row_by_id = {token.token_id: token.row_id for token in tokens}
    assert node_state_tokens
    for state in node_state_tokens:
        assert state.token_id in token_row_by_id
        assert token_row_by_id[state.token_id] in row_source_by_id

    # Scheduler work items carry ingest_sequence and all reached terminal.
    assert [item.status for item in work_items] == ["terminal"] * 5
    assert [item.ingest_sequence for item in work_items] == [0, 1, 2, 3, 4]


@pytest.mark.timeout(60)
def test_per_source_fixed_contracts_do_not_bleed_across_sources(tmp_path: Path) -> None:
    """C1 per-source contract: two sources with disjoint fixed contracts both
    validate cleanly — if either source's rows were validated under the other
    source's contract, the missing required field would discard them."""
    orders_path = tmp_path / "orders.csv"
    refunds_path = tmp_path / "refunds.csv"
    orders_output = tmp_path / "orders.jsonl"
    refunds_output = tmp_path / "refunds.jsonl"
    orders_path.write_text("id,total\n1,10\n2,20\n")
    refunds_path.write_text("ref,amount\nr1,-3.5\n")

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
        mode: fixed
        fields:
          - "id: int"
          - "total: int"
  refunds:
    plugin: csv
    on_success: refunds_out
    options:
      path: {refunds_path}
      on_validation_failure: discard
      schema:
        mode: fixed
        fields:
          - "ref: str"
          - "amount: float"
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
    result, db = _run_pipeline(settings, tmp_path)

    # No row was rejected: each source validated under ITS OWN contract.
    assert result.rows_processed == 3
    assert result.rows_succeeded == 3

    orders_rows = [json.loads(line) for line in orders_output.read_text().splitlines()]
    refunds_rows = [json.loads(line) for line in refunds_output.read_text().splitlines()]
    assert len(orders_rows) == 2
    assert len(refunds_rows) == 1
    assert all(set(row) == {"id", "total"} for row in orders_rows)
    assert all(set(row) == {"ref", "amount"} for row in refunds_rows)

    # The persisted per-source contracts are distinct and each names only its
    # own fields (run_sources is the single per-source contract writer).
    with db.connection() as conn:
        contracts = conn.execute(
            select(
                run_sources_table.c.source_name,
                run_sources_table.c.schema_contract_json,
                run_sources_table.c.schema_contract_hash,
            )
            .where(run_sources_table.c.run_id == result.run_id)
            .order_by(run_sources_table.c.source_name)
        ).all()

    assert [rec.source_name for rec in contracts] == ["orders", "refunds"]
    orders_contract, refunds_contract = contracts
    assert orders_contract.schema_contract_hash != refunds_contract.schema_contract_hash
    assert "total" in orders_contract.schema_contract_json
    assert "amount" not in orders_contract.schema_contract_json
    assert "amount" in refunds_contract.schema_contract_json
    assert "total" not in refunds_contract.schema_contract_json
