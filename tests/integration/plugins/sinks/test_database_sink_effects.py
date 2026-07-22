"""SQLite recovery proofs for the Database sink target-side effect ledger."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pytest
from sqlalchemy import CheckConstraint, Column, Integer, MetaData, Table, Text, create_engine, func, insert, inspect, select
from sqlalchemy.exc import IntegrityError

from elspeth.contracts.hashing import canonical_json
from elspeth.contracts.sink_effects import (
    RestrictedSinkEffectContext,
    SinkEffectDescriptorMode,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPrepareRequest,
    SinkEffectReconcileKind,
)
from elspeth.plugins.sinks.database_sink import DatabaseEffectLedgerError, DatabaseSink, database_effect_ledger_table
from tests.fixtures.base_classes import inject_write_failure

_SCHEMA = {"mode": "fixed", "fields": ["id: int", "name: str"]}
_LEDGER = {
    "table": "_elspeth_sink_effects",
    "schema_version": 1,
    "permissions": ["select", "insert"],
}
_CTX = RestrictedSinkEffectContext(
    run_id="run-database-effect",
    run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
    operation_id="operation-database-effect",
    sink_node_id="sink-database-effect",
)


def _member(ordinal: int, row: dict[str, object]) -> SinkEffectMember:
    payload = canonical_json(row).encode()
    return SinkEffectMember(
        ordinal=ordinal,
        token_id=f"token-{ordinal}",
        row_id=f"row-{ordinal}",
        ingest_sequence=ordinal,
        lineage_json="[]",
        lineage_hash=sha256(b"[]").hexdigest(),
        payload_hash=sha256(payload).hexdigest(),
        row=row,
        member_effect_id=sha256(f"member-{ordinal}".encode()).hexdigest(),
    )


def _config(url: str, *, ledger: dict[str, object] | None = _LEDGER) -> dict[str, object]:
    config: dict[str, object] = {"url": url, "table": "output", "schema": _SCHEMA, "if_exists": "append"}
    if ledger is not None:
        config["effect_ledger"] = ledger
    return config


def _provision(url: str, *, ledger: bool = True) -> None:
    engine = create_engine(url)
    metadata = MetaData()
    Table(
        "output",
        metadata,
        Column("id", Integer, nullable=False, unique=True),
        Column("name", Text, nullable=False),
    )
    if ledger:
        database_effect_ledger_table(metadata, "_elspeth_sink_effects")
    metadata.create_all(engine)
    engine.dispose()


def _prepare(sink: DatabaseSink, rows: tuple[dict[str, object], ...], *, effect_id: str = "a" * 64):
    members = tuple(_member(ordinal, row) for ordinal, row in enumerate(rows))
    inspection = sink.inspect_effect(
        SinkEffectInspectionRequest(effect_id=effect_id, target="{}", predecessor_descriptor=None),
        _CTX,
    )
    return sink.prepare_effect(
        SinkEffectPrepareRequest(
            effect_id=effect_id,
            effect_input=SinkEffectPipelineMembersInput(members=members, target_snapshot_members=members),
            inspection=inspection,
        ),
        _CTX,
    )


def test_inspection_is_read_only_and_refuses_missing_operator_ledger(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'missing-ledger.db'}"
    _provision(url, ledger=False)
    sink = inject_write_failure(DatabaseSink(_config(url)))
    engine = create_engine(url)
    before = set(inspect(engine).get_table_names())

    with pytest.raises(DatabaseEffectLedgerError, match="provision"):
        sink.inspect_effect(
            SinkEffectInspectionRequest(effect_id="a" * 64, target="{}", predecessor_descriptor=None),
            _CTX,
        )

    assert set(inspect(engine).get_table_names()) == before
    engine.dispose()


def test_marker_and_accepted_rows_commit_once_with_constraint_diversion(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'effect.db'}"
    _provision(url)
    engine = create_engine(url)
    target = Table("output", MetaData(), autoload_with=engine)
    with engine.begin() as conn:
        conn.execute(insert(target), [{"id": 2, "name": "existing"}])

    sink = inject_write_failure(DatabaseSink(_config(url)))
    plan = _prepare(sink, ({"id": 1, "name": "one"}, {"id": 2, "name": "duplicate"}, {"id": 3, "name": "three"}))
    assert plan.descriptor_mode is SinkEffectDescriptorMode.RESULT_DERIVED
    assert plan.expected_descriptor is None

    first = sink.commit_effect(plan, _CTX)
    replay = inject_write_failure(DatabaseSink(_config(url))).commit_effect(plan, _CTX)
    reconciled = inject_write_failure(DatabaseSink(_config(url))).reconcile_effect(plan, _CTX)

    assert replay == first
    assert first.accepted_ordinals == (0, 2)
    assert first.diverted_ordinals == (1,)
    assert first.descriptor.metadata == {"table": "output", "row_count": 2}
    assert reconciled.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR
    assert reconciled.descriptor == first.descriptor
    assert reconciled.accepted_ordinals == (0, 2)
    assert reconciled.diverted_ordinals == (1,)

    ledger = Table("_elspeth_sink_effects", MetaData(), autoload_with=engine)
    with engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(target)) == 3
        marker = conn.execute(select(ledger).where(ledger.c.effect_id == plan.effect_id)).mappings().one()
    assert json.loads(marker["accepted_ordinals_json"]) == [0, 2]
    assert json.loads(marker["diverted_ordinals_json"]) == [1]
    assert marker["accepted_payload_hash"] == first.descriptor.content_hash
    assert "duplicate" not in marker["diversion_hashes_json"]
    # Full attribution is durable in the marker and bound into the evidence
    # before finalization (elspeth-e0342d547f).
    marker_attribution = json.loads(marker["diversion_hashes_json"])
    assert [set(item) for item in marker_attribution] == [{"error_hash", "ordinal", "reason_hash"}]
    assert marker_attribution[0]["ordinal"] == 1
    assert json.loads(marker["evidence_json"])["diversion_attribution"] == marker_attribution
    # The live diversion log carries the real constraint reason for routing.
    live = sink._get_diversions()
    assert [item.row_index for item in live] == [1]
    assert live[0].reason.startswith("Constraint violation:")
    assert [dict(item) for item in first.evidence["diversion_attribution"]] == marker_attribution
    engine.dispose()


def test_response_loss_recovers_exact_zero_row_result_from_marker(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'zero.db'}"
    _provision(url)
    engine = create_engine(url)
    target = Table("output", MetaData(), autoload_with=engine)
    with engine.begin() as conn:
        conn.execute(insert(target), [{"id": 7, "name": "existing"}])

    plan = _prepare(
        inject_write_failure(DatabaseSink(_config(url))),
        ({"id": 7, "name": "duplicate-a"}, {"id": 7, "name": "duplicate-b"}),
        effect_id="b" * 64,
    )
    lost_result = inject_write_failure(DatabaseSink(_config(url))).commit_effect(plan, _CTX)
    recovered = inject_write_failure(DatabaseSink(_config(url))).reconcile_effect(plan, _CTX)

    assert lost_result.accepted_ordinals == ()
    assert lost_result.diverted_ordinals == (0, 1)
    assert lost_result.descriptor.metadata == {"table": "output", "row_count": 0}
    assert recovered.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR
    assert recovered.descriptor == lost_result.descriptor
    assert recovered.accepted_ordinals == ()
    assert recovered.diverted_ordinals == (0, 1)
    with engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(target)) == 1
    engine.dispose()


def test_reconcile_missing_is_not_applied_and_divergent_marker_is_unknown(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'reconcile.db'}"
    _provision(url)
    sink = inject_write_failure(DatabaseSink(_config(url)))
    plan = _prepare(sink, ({"id": 1, "name": "one"},), effect_id="c" * 64)

    missing = sink.reconcile_effect(plan, _CTX)
    assert missing.kind is SinkEffectReconcileKind.NOT_APPLIED

    sink.commit_effect(plan, _CTX)
    engine = create_engine(url)
    ledger = Table("_elspeth_sink_effects", MetaData(), autoload_with=engine)
    with engine.begin() as conn:
        conn.execute(ledger.update().where(ledger.c.effect_id == plan.effect_id).values(plan_hash="f" * 64))
    divergent = inject_write_failure(DatabaseSink(_config(url))).reconcile_effect(plan, _CTX)
    assert divergent.kind is SinkEffectReconcileKind.UNKNOWN
    engine.dispose()


def test_marker_insert_failure_rolls_back_accepted_rows_in_same_transaction(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'atomic-rollback.db'}"
    engine = create_engine(url)
    metadata = MetaData()
    target = Table(
        "output",
        metadata,
        Column("id", Integer, nullable=False, unique=True),
        Column("name", Text, nullable=False),
    )
    ledger = database_effect_ledger_table(metadata, "_elspeth_sink_effects")
    ledger.append_constraint(CheckConstraint("plan_hash = '" + ("f" * 64) + "'", name="force_marker_rejection"))
    metadata.create_all(engine)
    sink = inject_write_failure(DatabaseSink(_config(url)))
    plan = _prepare(sink, ({"id": 1, "name": "one"},), effect_id="e" * 64)

    with pytest.raises(IntegrityError):
        sink.commit_effect(plan, _CTX)

    with engine.connect() as conn:
        assert conn.scalar(select(func.count()).select_from(target)) == 0
        assert conn.scalar(select(func.count()).select_from(ledger)) == 0
    engine.dispose()
