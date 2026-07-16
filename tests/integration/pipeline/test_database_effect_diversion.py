"""Pipeline-boundary attribution proofs for database effect constraint diversions.

Regression coverage for elspeth-e0342d547f: a constraint-diverted member of a
database effect must durably record reason/error attribution (result evidence,
ledger marker, live diversion log) so executor finalization completes instead
of raising AuditIntegrityError after the accepted rows and marker committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import Column, Integer, MetaData, Table, Text, create_engine, func, insert, select

from elspeth.contracts import NodeType, PendingOutcome, TerminalOutcome, TerminalPath, TokenInfo
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.contracts.sink_effects import SinkEffectRole
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.executors.sink import SinkExecutor
from elspeth.engine.executors.sink_effects import SinkEffectExecutionSeam, SinkEffectInjectedFault
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.sinks.database_sink import DatabaseSink, database_effect_ledger_table
from tests.fixtures.base_classes import create_observed_contract, inject_write_failure
from tests.fixtures.landscape import make_factory, register_test_node

_SCHEMA = {"mode": "fixed", "fields": ["id: int", "name: str"]}
_LEDGER = {
    "table": "_elspeth_sink_effects",
    "schema_version": 1,
    "permissions": ["select", "insert"],
}


def _provision_target(url: str) -> None:
    engine = create_engine(url)
    metadata = MetaData()
    Table(
        "output",
        metadata,
        Column("id", Integer, nullable=False, unique=True),
        Column("name", Text, nullable=False),
    )
    database_effect_ledger_table(metadata, "_elspeth_sink_effects")
    metadata.create_all(engine)
    with engine.begin() as conn:
        target = Table("output", MetaData(), autoload_with=conn)
        conn.execute(insert(target), [{"id": 2, "name": "existing"}])
    engine.dispose()


def _sink_config(url: str) -> dict[str, object]:
    return {"url": url, "table": "output", "schema": _SCHEMA, "if_exists": "append", "effect_ledger": _LEDGER}


def _make_tokens(factory, run_id: str, source_id: str, rows: list[dict[str, object]]) -> list[TokenInfo]:
    tokens: list[TokenInfo] = []
    for index, row_data in enumerate(rows):
        row = factory.data_flow.create_row(
            run_id=run_id,
            source_node_id=source_id,
            row_index=index,
            data=row_data,
            source_row_index=index,
            ingest_sequence=index,
        )
        durable_token = factory.data_flow.create_token(row.row_id)
        tokens.append(
            TokenInfo(
                row_id=row.row_id,
                token_id=durable_token.token_id,
                row_data=PipelineRow(row_data, create_observed_contract(row_data)),
            )
        )
    return tokens


_ROWS: list[dict[str, object]] = [
    {"id": 1, "name": "one"},
    {"id": 2, "name": "duplicate"},
    {"id": 3, "name": "three"},
]


def test_constraint_diversion_discards_durably_without_audit_error(tmp_path: Path) -> None:
    """Live path: a UNIQUE-constraint diversion must finalize with durable attribution."""
    target_url = f"sqlite:///{tmp_path / 'target.db'}"
    _provision_target(target_url)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'landscape.db'}")
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        sink_id = register_test_node(factory.data_flow, run.run_id, "database-sink", node_type=NodeType.SINK, plugin_name="database")
        tokens = _make_tokens(factory, run.run_id, source_id, _ROWS)

        sink = inject_write_failure(DatabaseSink(_sink_config(target_url)))
        sink.node_id = sink_id
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=sink_id)

        artifact, counts = SinkExecutor(
            factory.execution,
            factory.data_flow,
            SpanFactory(),
            run.run_id,
            factory=factory,
            worker_id="worker-a",
        ).write(
            sink,  # type: ignore[arg-type]
            tokens,
            ctx,
            1,
            sink_name="output",
            pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
            effect_mode="append",
        )

        assert artifact is not None
        assert counts.discard_mode == 1
        assert counts.failsink_mode == 0

        # Accepted rows landed exactly once; the diverted row never did.
        engine = create_engine(target_url)
        target = Table("output", MetaData(), autoload_with=engine)
        with engine.connect() as conn:
            assert conn.scalar(select(func.count()).select_from(target)) == 3
            names = {row.name for row in conn.execute(select(target)).fetchall()}
            marker = conn.execute(select(Table("_elspeth_sink_effects", MetaData(), autoload_with=conn))).mappings().one()
        engine.dispose()
        assert "duplicate" not in names

        # Ledger marker durably binds full attribution for the diverted ordinal.
        hashes = json.loads(marker["diversion_hashes_json"])
        assert [item["ordinal"] for item in hashes] == [1]
        assert set(hashes[0]) == {"error_hash", "ordinal", "reason_hash"}
        evidence = json.loads(marker["evidence_json"])
        assert evidence["diversion_attribution"] == hashes

        # Durable member partition records the diversion.
        durable = factory.execution.sink_effects.get_members_for_tokens(
            run_id=run.run_id,
            sink_node_id=sink_id,
            role=SinkEffectRole.PRIMARY,
            token_ids=[token.token_id for token in tokens],
        )
        dispositions = {member.ordinal: member.prepared_disposition for member in durable}
        assert dispositions == {0: "accepted", 1: "diverted", 2: "accepted"}
    finally:
        db.close()


def test_constraint_diversion_recovers_attribution_after_crash_before_finalize(tmp_path: Path) -> None:
    """Recovery path: attribution must be recoverable from durable effect evidence."""
    target_url = f"sqlite:///{tmp_path / 'target.db'}"
    _provision_target(target_url)
    db = LandscapeDB(f"sqlite:///{tmp_path / 'landscape.db'}")
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        sink_id = register_test_node(factory.data_flow, run.run_id, "database-sink", node_type=NodeType.SINK, plugin_name="database")
        tokens = _make_tokens(factory, run.run_id, source_id, _ROWS)
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=sink_id)

        calls = 0

        def fail_once(observed: SinkEffectExecutionSeam) -> None:
            nonlocal calls
            if observed is SinkEffectExecutionSeam.AFTER_RETURN_BEFORE_FINALIZE and calls == 0:
                calls += 1
                raise SinkEffectInjectedFault(observed)

        first_sink = inject_write_failure(DatabaseSink(_sink_config(target_url)))
        first_sink.node_id = sink_id
        with pytest.raises(SinkEffectInjectedFault):
            SinkExecutor(
                factory.execution,
                factory.data_flow,
                SpanFactory(),
                run.run_id,
                factory=factory,
                worker_id="worker-a",
                sink_effect_fault_hook=fail_once,
            ).write(
                first_sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
                effect_mode="append",
            )

        # Fresh process: a new sink instance has an empty live diversion log,
        # so the executor must recover attribution from durable evidence.
        recovered_factory = make_factory(db)
        recovered_sink = inject_write_failure(DatabaseSink(_sink_config(target_url)))
        recovered_sink.node_id = sink_id
        artifact, counts = SinkExecutor(
            recovered_factory.execution,
            recovered_factory.data_flow,
            SpanFactory(),
            run.run_id,
            factory=recovered_factory,
            worker_id="worker-a",
        ).write(
            recovered_sink,  # type: ignore[arg-type]
            tokens,
            ctx,
            1,
            sink_name="output",
            pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
            effect_mode="append",
        )

        assert artifact is not None
        assert counts.discard_mode == 1

        # Accepted rows landed exactly once despite the replay.
        engine = create_engine(target_url)
        target = Table("output", MetaData(), autoload_with=engine)
        with engine.connect() as conn:
            assert conn.scalar(select(func.count()).select_from(target)) == 3
        engine.dispose()
    finally:
        db.close()
