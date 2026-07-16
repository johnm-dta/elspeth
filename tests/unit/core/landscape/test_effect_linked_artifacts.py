"""Epoch-26 artifact and operation producer-link contracts."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from elspeth.contracts import Artifact
from elspeth.contracts.enums import NodeType
from elspeth.contracts.schema import SchemaConfig
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import sink_effect_members_table, sink_effects_table

_HASH = "a" * 64
_DYNAMIC_SCHEMA = SchemaConfig.from_dict({"mode": "observed"})


def _artifact(**overrides: object) -> Artifact:
    values: dict[str, object] = {
        "artifact_id": "artifact-1",
        "run_id": "run-1",
        "sink_node_id": "sink-1",
        "artifact_type": "file",
        "path_or_uri": "/output/result.csv",
        "content_hash": _HASH,
        "size_bytes": 1,
        "created_at": datetime(2026, 7, 16, tzinfo=UTC),
        "produced_by_state_id": "state-1",
        "sink_effect_id": None,
        "publication_performed": True,
        "publication_evidence_kind": "legacy_returned",
    }
    values.update(overrides)
    return Artifact(**values)  # type: ignore[arg-type]


def test_artifact_requires_exactly_one_producer_link() -> None:
    with pytest.raises(ValueError, match="exactly one producer"):
        _artifact(produced_by_state_id=None, sink_effect_id=None)
    with pytest.raises(ValueError, match="exactly one producer"):
        _artifact(produced_by_state_id="state-1", sink_effect_id="effect-1")


@pytest.mark.parametrize(
    ("produced_by_state_id", "sink_effect_id", "publication_performed", "evidence"),
    [
        ("state-1", None, True, "legacy_returned"),
        (None, "effect-1", True, "returned"),
        (None, "effect-1", True, "reconciled"),
        (None, "effect-1", False, "inherited"),
        (None, "effect-1", False, "virtual"),
    ],
)
def test_artifact_accepts_only_producer_appropriate_publication_evidence(
    produced_by_state_id: str | None,
    sink_effect_id: str | None,
    publication_performed: bool,
    evidence: str,
) -> None:
    artifact = _artifact(
        produced_by_state_id=produced_by_state_id,
        sink_effect_id=sink_effect_id,
        publication_performed=publication_performed,
        publication_evidence_kind=evidence,
    )

    assert artifact.producer_kind == ("node_state" if produced_by_state_id is not None else "sink_effect")


@pytest.mark.parametrize(
    ("produced_by_state_id", "sink_effect_id", "publication_performed", "evidence"),
    [
        ("state-1", None, True, "returned"),
        ("state-1", None, False, "legacy_returned"),
        (None, "effect-1", False, "returned"),
        (None, "effect-1", True, "inherited"),
        (None, "effect-1", False, "legacy_returned"),
        (None, "effect-1", True, "not-closed"),
    ],
)
def test_artifact_rejects_ambiguous_publication_evidence(
    produced_by_state_id: str | None,
    sink_effect_id: str | None,
    publication_performed: bool,
    evidence: str,
) -> None:
    with pytest.raises(ValueError, match="publication"):
        _artifact(
            produced_by_state_id=produced_by_state_id,
            sink_effect_id=sink_effect_id,
            publication_performed=publication_performed,
            publication_evidence_kind=evidence,
        )


def _factory_with_sink_and_state() -> tuple[LandscapeDB, RecorderFactory, str, str, str]:
    db = LandscapeDB.in_memory()
    factory = RecorderFactory(db)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    sink = factory.data_flow.register_node(
        run_id=run.run_id,
        plugin_name="csv_sink",
        node_type=NodeType.SINK,
        plugin_version="1.0",
        config={},
        schema_config=_DYNAMIC_SCHEMA,
    )
    row = factory.data_flow.create_row(
        run_id=run.run_id,
        source_node_id=sink.node_id,
        row_index=0,
        data={"value": 1},
        source_row_index=0,
        ingest_sequence=0,
    )
    token = factory.data_flow.create_token(row_id=row.row_id)
    state = factory.execution.begin_node_state(
        token_id=token.token_id,
        node_id=sink.node_id,
        run_id=run.run_id,
        step_index=0,
        input_data={},
    )
    return db, factory, run.run_id, sink.node_id, state.state_id


def _insert_reserved_effect(db: LandscapeDB, *, run_id: str, sink_node_id: str, state_id: str) -> str:
    del state_id
    effect_id = "b" * 64
    now = datetime(2026, 7, 16, tzinfo=UTC)
    with db.write_connection() as conn:
        token_row = conn.exec_driver_sql(
            "SELECT ns.token_id, t.row_id FROM node_states AS ns "
            "JOIN tokens AS t ON t.token_id = ns.token_id "
            "WHERE ns.run_id = ? ORDER BY ns.started_at LIMIT 1",
            (run_id,),
        ).one()
        conn.execute(
            sink_effects_table.insert().values(
                effect_id=effect_id,
                run_id=run_id,
                sink_node_id=sink_node_id,
                role="primary",
                state="reserved",
                protocol_version="sink-effect-v1",
                input_kind="pipeline_members",
                required_member_ordinal=0,
                required_snapshot_slot=None,
                config_hash="c" * 64,
                membership_or_manifest_hash="d" * 64,
                group_payload_hash="e" * 64,
                artifact_id="f" * 64,
                artifact_idempotency_key="effect-artifact-key",
                target_json='{"target":"test"}',
                generation=0,
                created_at=now,
                updated_at=now,
            )
        )
        conn.execute(
            sink_effect_members_table.insert().values(
                effect_id=effect_id,
                input_kind="pipeline_members",
                ordinal=0,
                run_id=run_id,
                sink_node_id=sink_node_id,
                role="primary",
                token_id=token_row.token_id,
                row_id=token_row.row_id,
                ingest_sequence=0,
                lineage_json="[]",
                lineage_hash="1" * 64,
                payload_hash="2" * 64,
            )
        )
    return effect_id


def test_repository_round_trips_legacy_and_effect_producer_links() -> None:
    db, factory, run_id, sink_node_id, state_id = _factory_with_sink_and_state()
    effect_id = _insert_reserved_effect(db, run_id=run_id, sink_node_id=sink_node_id, state_id=state_id)

    legacy = factory.execution.register_artifact(
        run_id=run_id,
        state_id=state_id,
        sink_effect_id=None,
        sink_node_id=sink_node_id,
        artifact_type="file",
        path="/output/legacy.csv",
        content_hash="3" * 64,
        size_bytes=3,
    )
    effect = factory.execution.register_artifact(
        run_id=run_id,
        state_id=None,
        sink_effect_id=effect_id,
        sink_node_id=sink_node_id,
        artifact_type="file",
        path="/output/effect.csv",
        content_hash="4" * 64,
        size_bytes=4,
        idempotency_key="effect-key",
        publication_performed=False,
        publication_evidence_kind="inherited",
    )

    assert legacy.producer_kind == "node_state"
    assert legacy.sink_effect_id is None
    assert legacy.publication_evidence_kind == "legacy_returned"
    assert effect.producer_kind == "sink_effect"
    assert effect.produced_by_state_id is None
    assert effect.sink_effect_id == effect_id
    assert effect.publication_performed is False
    assert factory.execution.get_artifacts(run_id) == [legacy, effect]


def test_idempotent_artifact_rejects_divergent_effect_linkage_and_publication_evidence() -> None:
    db, factory, run_id, sink_node_id, state_id = _factory_with_sink_and_state()
    effect_id = _insert_reserved_effect(db, run_id=run_id, sink_node_id=sink_node_id, state_id=state_id)
    values = {
        "run_id": run_id,
        "state_id": None,
        "sink_effect_id": effect_id,
        "sink_node_id": sink_node_id,
        "artifact_type": "file",
        "path": "/output/effect.csv",
        "content_hash": "5" * 64,
        "size_bytes": 5,
        "idempotency_key": "same-effect-key",
        "publication_performed": True,
        "publication_evidence_kind": "returned",
    }
    factory.execution.register_artifact(**values)

    with pytest.raises(LandscapeRecordError, match="publication_evidence_kind"):
        factory.execution.register_artifact(**(values | {"publication_evidence_kind": "reconciled"}))


def test_effect_linked_sink_operation_round_trips_and_rejects_non_sink_write() -> None:
    db, factory, run_id, sink_node_id, state_id = _factory_with_sink_and_state()
    effect_id = _insert_reserved_effect(db, run_id=run_id, sink_node_id=sink_node_id, state_id=state_id)

    operation = factory.execution.begin_operation(
        run_id,
        sink_node_id,
        "sink_write",
        sink_effect_id=effect_id,
    )
    assert operation.sink_effect_id == effect_id
    loaded = factory.execution.get_operation(operation.operation_id)
    assert loaded is not None
    assert loaded.operation_id == operation.operation_id
    assert loaded.sink_effect_id == effect_id

    with pytest.raises(ValueError, match="sink_write"):
        factory.execution.begin_operation(
            run_id,
            sink_node_id,
            "source_load",
            sink_effect_id=effect_id,
        )
