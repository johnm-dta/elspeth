"""Characterization tests for durable row-identity edges under quarantine and multi-source.

Drives the REAL orchestrator (real SQLite LandscapeDB, real scheduler, real
plugin instantiation from YAML settings) and pins the CURRENT ingest-identity
semantics of the source iteration loop (source_iteration.py):

1. ``ingest_sequence`` is assigned from ``counters.rows_processed`` BEFORE the
   quarantine-vs-valid branch, so quarantined rows consume a sequence number:
   the GLOBAL ordering primitive is gapless over ALL ingested rows (valid +
   quarantined) while the valid-row subsequence is gapped around quarantines.
   Quarantined rows are not identity-less: each gets a durable rows-table row
   (source_node_id, source_row_index, ingest_sequence, payload) plus a token
   with a completed terminal outcome, so resume's unprocessed-rows filter
   excludes them and ordering cannot misattribute (filigree elspeth-1869c9ba64).
2. The same invariants hold across sources: interleaved quarantines preserve
   ``ingest_sequence`` as a strict gapless global ordering in declaration
   order, and per-source ``source_row_index`` stays contiguous INCLUDING
   quarantined rows.
3. Per-source identity is passed explicitly on every production call site:
   in a 2-source fan-in run, zero rows originating from source #2's data
   attribute to source #1's node — neither in the rows table nor in any
   step-0 (source) node_state. This guards the RowProcessor run-level
   "first source" legacy default (run_core.py RowProcessor construction;
   ``record_source_node_state``'s ``source_node_id or self._source_node_id``
   fallback) against silent misattribution. Strengthens
   tests/integration/test_multisource_provenance_proof.py::
   test_per_source_provenance_is_durable_in_audit_database, which proves
   positional attribution but not content->node binding.
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
    token_outcomes_table,
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
    payload_store = FilesystemPayloadStore(tmp_path / "payloads")
    result = Orchestrator(db).run(
        config,
        graph=graph,
        settings=settings,
        payload_store=payload_store,
    )
    return result, db, payload_store


def _select_rows(db: LandscapeDB, run_id: str):
    with db.connection() as conn:
        return conn.execute(
            select(
                rows_table.c.row_id,
                rows_table.c.source_node_id,
                rows_table.c.source_row_index,
                rows_table.c.ingest_sequence,
                rows_table.c.source_data_hash,
                rows_table.c.source_data_ref,
            )
            .where(rows_table.c.run_id == run_id)
            .order_by(rows_table.c.ingest_sequence)
        ).all()


def _row_payload(payload_store: FilesystemPayloadStore, source_data_ref: str) -> dict[str, object]:
    payload = json.loads(payload_store.retrieve(source_data_ref).decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def _outcomes_by_row(db: LandscapeDB, run_id: str) -> dict[str, tuple[str, str, str | None, int]]:
    """row_id -> (outcome, path, sink_name, completed) via the token join."""
    with db.connection() as conn:
        records = conn.execute(
            select(
                tokens_table.c.row_id,
                token_outcomes_table.c.outcome,
                token_outcomes_table.c.path,
                token_outcomes_table.c.sink_name,
                token_outcomes_table.c.completed,
            )
            .join(
                tokens_table,
                (tokens_table.c.run_id == token_outcomes_table.c.run_id) & (tokens_table.c.token_id == token_outcomes_table.c.token_id),
            )
            .where(token_outcomes_table.c.run_id == run_id)
        ).all()
    return {rec.row_id: (rec.outcome, rec.path, rec.sink_name, rec.completed) for rec in records}


def _quarantine_pipeline_yaml(
    *,
    sources: dict[str, Path],
    output_path: Path,
    quarantine_path: Path,
) -> str:
    source_blocks = "".join(
        f"""
  {name}:
    plugin: csv
    on_success: inbound
    options:
      path: {path}
      on_validation_failure: quarantine
      schema:
        mode: fixed
        fields: ["id: int", "origin: str"]"""
        for name, path in sources.items()
    )
    return f"""
sources:{source_blocks}
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


@pytest.mark.timeout(60)
def test_quarantined_row_consumes_ingest_sequence_with_durable_identity(tmp_path: Path) -> None:
    """A [valid, quarantined, valid] source yields a gapless GLOBAL
    ingest_sequence [0, 1, 2] where the quarantined row owns sequence 1 with
    full durable identity (rows-table row, payload, completed terminal
    outcome), and the valid rows' subsequence [0, 2] is gapped around it."""
    source_path = tmp_path / "orders.csv"
    source_path.write_text("id,origin\n1,valid-a\nnot-an-int,quarantined-row\n3,valid-b\n")
    output_path = tmp_path / "out.jsonl"
    quarantine_path = tmp_path / "quarantine.jsonl"
    settings = load_settings_from_yaml_string(
        _quarantine_pipeline_yaml(
            sources={"orders": source_path},
            output_path=output_path,
            quarantine_path=quarantine_path,
        )
    )
    result, db, payload_store = _run_pipeline(settings, tmp_path)

    # All three source rows were processed: 2 valid + 1 quarantined.
    assert result.rows_processed == 3
    assert result.rows_quarantined == 1

    rows = _select_rows(db, result.run_id)
    assert len(rows) == 3

    # The global ordering primitive is gapless over ALL ingested rows and
    # per-source identity is contiguous — the quarantined row consumed both.
    assert [row.ingest_sequence for row in rows] == [0, 1, 2]
    assert [row.source_row_index for row in rows] == [0, 1, 2]
    assert len({row.source_node_id for row in rows}) == 1

    outcomes = _outcomes_by_row(db, result.run_id)
    assert set(outcomes) == {row.row_id for row in rows}

    # The quarantined row is exactly the middle one, with durable content
    # identity: its payload is retrievable and carries the rejected value.
    payload_by_sequence = {row.ingest_sequence: _row_payload(payload_store, row.source_data_ref) for row in rows}
    quarantined = [row for row in rows if outcomes[row.row_id][1] == "quarantined_at_source"]
    assert len(quarantined) == 1
    assert quarantined[0].ingest_sequence == 1
    assert quarantined[0].source_data_hash
    assert outcomes[quarantined[0].row_id] == ("failure", "quarantined_at_source", "quarantine", 1)
    assert payload_by_sequence[1]["id"] == "not-an-int"

    # The valid rows' subsequence is GAPPED around the quarantine — consumers
    # of ingest order must tolerate gaps in the valid-row numbering.
    valid_sequences = sorted(row.ingest_sequence for row in rows if outcomes[row.row_id][1] != "quarantined_at_source")
    assert valid_sequences == [0, 2]
    assert payload_by_sequence[0]["origin"] == "valid-a"
    assert payload_by_sequence[2]["origin"] == "valid-b"

    # Every row reached a completed terminal outcome, including the
    # quarantined one — resume's unprocessed-rows filter (rows lacking a
    # terminal outcome) therefore never replays it despite the gap.
    assert all(completed == 1 for (_, _, _, completed) in outcomes.values())

    # The quarantined row landed in the quarantine sink, the valid rows in output.
    assert len(quarantine_path.read_text().splitlines()) == 1
    assert len(output_path.read_text().splitlines()) == 2


@pytest.mark.timeout(60)
def test_interleaved_quarantines_preserve_strict_global_ordering_across_sources(tmp_path: Path) -> None:
    """With a quarantined row inside EACH of two sources, ingest_sequence is a
    strict gapless global ordering over all rows in source declaration order,
    and per-source source_row_index stays contiguous including quarantines."""
    source_a_path = tmp_path / "alpha.csv"
    source_b_path = tmp_path / "beta.csv"
    source_a_path.write_text("id,origin\n1,alpha\nnot-an-int,alpha\n")
    source_b_path.write_text("id,origin\nnot-an-int,beta\n4,beta\n")
    settings = load_settings_from_yaml_string(
        _quarantine_pipeline_yaml(
            sources={"alpha": source_a_path, "beta": source_b_path},
            output_path=tmp_path / "out.jsonl",
            quarantine_path=tmp_path / "quarantine.jsonl",
        )
    )
    result, db, _payload_store = _run_pipeline(settings, tmp_path)

    assert result.rows_processed == 4
    assert result.rows_quarantined == 2

    with db.connection() as conn:
        node_by_source_name = {
            rec.source_name: rec.source_node_id
            for rec in conn.execute(
                select(run_sources_table.c.source_name, run_sources_table.c.source_node_id).where(
                    run_sources_table.c.run_id == result.run_id
                )
            )
        }
    rows = _select_rows(db, result.run_id)
    outcomes = _outcomes_by_row(db, result.run_id)

    # Strict gapless global ordering in declaration order: alpha's two rows
    # (including its quarantine) before beta's two rows (including its
    # quarantine) — quarantined rows participate, they are not skipped.
    assert [row.ingest_sequence for row in rows] == [0, 1, 2, 3]
    assert [row.source_node_id for row in rows] == [
        node_by_source_name["alpha"],
        node_by_source_name["alpha"],
        node_by_source_name["beta"],
        node_by_source_name["beta"],
    ]
    quarantined_sequences = sorted(row.ingest_sequence for row in rows if outcomes[row.row_id][1] == "quarantined_at_source")
    assert quarantined_sequences == [1, 2]

    # Per-source identity is contiguous including the quarantined rows.
    per_source_indices: dict[str, list[int]] = {}
    for row in rows:
        per_source_indices.setdefault(row.source_node_id, []).append(row.source_row_index)
    assert per_source_indices == {
        node_by_source_name["alpha"]: [0, 1],
        node_by_source_name["beta"]: [0, 1],
    }


@pytest.mark.timeout(60)
def test_second_source_rows_never_attribute_to_first_source_node(tmp_path: Path) -> None:
    """Every production call site passes explicit per-source identity: in a
    2-source fan-in run, each row's durable source_node_id matches the source
    that actually authored its content (payload-verified), and every step-0
    (source) node_state carries that same node — zero rows originating from
    the second source attribute to the first source's node, so the
    RowProcessor first-source legacy default is never what gets recorded."""
    orders_path = tmp_path / "orders.csv"
    refunds_path = tmp_path / "refunds.csv"
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
      path: {tmp_path / "out.jsonl"}
      format: jsonl
      schema:
        mode: observed
"""
    )
    result, db, payload_store = _run_pipeline(settings, tmp_path)
    run_id = result.run_id
    assert result.rows_processed == 5

    with db.connection() as conn:
        node_by_source_name = {
            rec.source_name: rec.source_node_id
            for rec in conn.execute(
                select(run_sources_table.c.source_name, run_sources_table.c.source_node_id).where(run_sources_table.c.run_id == run_id)
            )
        }
        source_node_ids = {
            rec.node_id
            for rec in conn.execute(
                select(nodes_table.c.node_id).where(nodes_table.c.run_id == run_id).where(nodes_table.c.node_type == "source")
            )
        }
        tokens = conn.execute(select(tokens_table.c.token_id, tokens_table.c.row_id).where(tokens_table.c.run_id == run_id)).all()
        source_states = conn.execute(
            select(node_states_table.c.token_id, node_states_table.c.node_id)
            .where(node_states_table.c.run_id == run_id)
            .where(node_states_table.c.step_index == 0)
        ).all()
    rows = _select_rows(db, run_id)

    assert set(node_by_source_name) == {"orders", "refunds"}
    assert set(node_by_source_name.values()) == source_node_ids

    # Content -> node binding: each row's durable attribution matches the
    # source that authored its payload. In particular, ZERO rows whose content
    # came from refunds.csv attribute to the orders node.
    assert len(rows) == 5
    payload_origins: dict[str, str] = {}
    for row in rows:
        payload = _row_payload(payload_store, row.source_data_ref)
        origin = payload["origin"]
        assert isinstance(origin, str)
        payload_origins[row.row_id] = origin
        assert row.source_node_id == node_by_source_name[origin]
    assert sorted(payload_origins.values()) == ["orders"] * 3 + ["refunds"] * 2

    # Step-0 node_states are the source-node audit records: every one must
    # carry its row's own source node, not the run-level first-source default
    # (record_source_node_state falls back to RowProcessor._source_node_id
    # when a call site omits source_node_id — that fallback must never be
    # what reaches the audit trail for source #2's rows).
    row_node_by_id = {row.row_id: row.source_node_id for row in rows}
    token_row_by_id = {token.token_id: token.row_id for token in tokens}
    source_step_states = [state for state in source_states if state.node_id in source_node_ids]
    assert len(source_step_states) == 5
    for state in source_step_states:
        assert state.node_id == row_node_by_id[token_row_by_id[state.token_id]]
