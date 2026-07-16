"""Reservation contract for durable, ordered sink effects."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256

import pytest
from sqlalchemy import func, select, update

from elspeth.contracts import NodeType
from elspeth.contracts.sink_effects import SinkEffectInputKind, SinkEffectMember, SinkEffectMemberCandidate, SinkEffectRole
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.sink_effect_identity import compute_pipeline_effect_identity, resolve_sink_effect_members
from elspeth.core.landscape.execution.sink_effect_reservation import (
    SinkEffectReservationRequest,
    SinkEffectReservationResult,
)
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import (
    audit_export_snapshot_chunks_table,
    audit_export_snapshots_table,
    operations_table,
    sink_effect_export_snapshots_table,
    sink_effect_members_table,
    sink_effects_table,
)
from tests.fixtures.landscape import make_factory, make_landscape_db, register_test_node


@pytest.fixture
def db_factory() -> Iterator[tuple[LandscapeDB, RecorderFactory]]:
    db = make_landscape_db()
    try:
        yield db, make_factory(db)
    finally:
        db.close()


def _pipeline_members(factory: RecorderFactory, count: int = 3) -> tuple[str, str, tuple[SinkEffectMember, ...]]:
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
    sink_id = register_test_node(factory.data_flow, run.run_id, "sink", node_type=NodeType.SINK, plugin_name="sink")
    candidates: list[SinkEffectMemberCandidate] = []
    for ordinal in range(count):
        payload = {"ordinal": ordinal}
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source_id,
            row_index=ordinal,
            data=payload,
            source_row_index=ordinal,
            ingest_sequence=ordinal,
        )
        token = factory.data_flow.create_token(row.row_id)
        factory.execution.begin_node_state(
            token_id=token.token_id,
            node_id=sink_id,
            run_id=run.run_id,
            step_index=0,
            input_data=payload,
        )
        candidates.append(SinkEffectMemberCandidate(token_id=token.token_id, row=payload))
    return run.run_id, sink_id, resolve_sink_effect_members(factory, candidates)


def _pipeline_request(
    run_id: str,
    sink_id: str,
    members: Sequence[SinkEffectMember],
    *,
    replacing_target: bool = False,
) -> SinkEffectReservationRequest:
    canonical_members = tuple(
        replace(member, ordinal=ordinal, member_effect_id=None)
        for ordinal, member in enumerate(sorted(members, key=lambda member: member.ordinal))
    )
    identity = compute_pipeline_effect_identity(
        run_id=run_id,
        sink_node_id=sink_id,
        role=SinkEffectRole.PRIMARY,
        sink_config={"name": "sink"},
        target_config={"path": "out.jsonl"},
        members=canonical_members,
    )
    return SinkEffectReservationRequest(
        run_id=run_id,
        sink_node_id=sink_id,
        role=SinkEffectRole.PRIMARY,
        input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        requested_target_hash=identity.requested_target_hash,
        members=members,
        audit_export_snapshot_id=None,
        config_hash=identity.config_hash,
        replacing_target=replacing_target,
        primary_effect_id=None,
    )


def test_pipeline_reservation_is_idempotent_under_reverse_arrival(db_factory: tuple[LandscapeDB, RecorderFactory]) -> None:
    db, factory = db_factory
    run_id, sink_id, members = _pipeline_members(factory)

    first = factory.execution.sink_effects.reserve(_pipeline_request(run_id, sink_id, tuple(reversed(members))))
    second = factory.execution.sink_effects.reserve(_pipeline_request(run_id, sink_id, members))

    assert isinstance(first, SinkEffectReservationResult)
    assert first.new_effect is not None
    expected_identity = compute_pipeline_effect_identity(
        run_id=run_id,
        sink_node_id=sink_id,
        role=SinkEffectRole.PRIMARY,
        sink_config={"name": "sink"},
        target_config={"path": "out.jsonl"},
        members=members,
    )
    assert first.new_effect.effect_id == expected_identity.effect_id
    assert first.finalized_effect_ids == ()
    assert first.open_effect_ids == ()
    assert second.new_effect is None
    assert second.open_effect_ids == (first.new_effect.effect_id,)
    with db.read_only_connection() as conn:
        assert conn.scalar(select(func.count()).select_from(sink_effects_table)) == 1
        assert conn.scalar(select(func.count()).select_from(sink_effect_members_table)) == 3
        assert conn.scalar(select(func.count()).select_from(operations_table)) == 1


def test_overlap_partitions_finalized_open_and_unbound_members(db_factory: tuple[LandscapeDB, RecorderFactory]) -> None:
    db, factory = db_factory
    run_id, sink_id, members = _pipeline_members(factory, 4)
    finalized = factory.execution.sink_effects.reserve(_pipeline_request(run_id, sink_id, members[:1])).new_effect
    opened = factory.execution.sink_effects.reserve(_pipeline_request(run_id, sink_id, members[1:3])).new_effect
    assert finalized is not None and opened is not None
    now = datetime.now(UTC)
    with db.engine.begin() as conn:
        conn.execute(
            update(sink_effects_table)
            .where(sink_effects_table.c.effect_id == finalized.effect_id)
            .values(
                state="finalized",
                plan_hash="a" * 64,
                result_descriptor_hash="b" * 64,
                publication_performed=True,
                publication_evidence_kind="test",
                finalized_at=now,
                updated_at=now,
            )
        )

    result = factory.execution.sink_effects.reserve(_pipeline_request(run_id, sink_id, members))

    assert result.finalized_effect_ids == (finalized.effect_id,)
    assert result.open_effect_ids == (opened.effect_id,)
    assert result.new_effect is not None
    assert [member.token_id for member in factory.execution.sink_effects.get_members(result.new_effect.effect_id)] == [members[3].token_id]


def test_existing_member_lineage_divergence_fails_closed(db_factory: tuple[LandscapeDB, RecorderFactory]) -> None:
    _db, factory = db_factory
    run_id, sink_id, members = _pipeline_members(factory, 1)
    factory.execution.sink_effects.reserve(_pipeline_request(run_id, sink_id, members))
    divergent_lineage = "[[0,[]]]"
    divergent = replace(
        members[0],
        lineage_json=divergent_lineage,
        lineage_hash=sha256(divergent_lineage.encode()).hexdigest(),
    )

    with pytest.raises(ValueError, match="divergent"):
        factory.execution.sink_effects.reserve(_pipeline_request(run_id, sink_id, (divergent,)))


def test_replacing_target_allocates_monotonic_stream_predecessors(db_factory: tuple[LandscapeDB, RecorderFactory]) -> None:
    _db, factory = db_factory
    run_id, sink_id, members = _pipeline_members(factory, 2)
    first = factory.execution.sink_effects.reserve(_pipeline_request(run_id, sink_id, members[:1], replacing_target=True)).new_effect
    second = factory.execution.sink_effects.reserve(_pipeline_request(run_id, sink_id, members[1:], replacing_target=True)).new_effect

    assert first is not None and second is not None
    assert (first.stream_sequence, first.predecessor_effect_id) == (0, None)
    assert (second.stream_sequence, second.predecessor_effect_id) == (1, first.effect_id)
    stream = factory.execution.sink_effects.get_stream(first.stream_id)
    assert stream is not None
    assert (stream.next_sequence, stream.tail_effect_id) == (2, second.effect_id)


def test_invalid_pipeline_export_xor_is_rejected_before_sql(db_factory: tuple[LandscapeDB, RecorderFactory]) -> None:
    db, factory = db_factory
    run_id, sink_id, members = _pipeline_members(factory, 1)
    valid = _pipeline_request(run_id, sink_id, members)
    with pytest.raises(ValueError, match="snapshot"):
        replace(valid, audit_export_snapshot_id="9" * 64)
    with db.read_only_connection() as conn:
        assert conn.scalar(select(func.count()).select_from(sink_effects_table)) == 0


def _insert_snapshot(db: LandscapeDB, run_id: str) -> str:
    snapshot_id = "2" * 64
    completed = datetime(2026, 7, 16, 1, 2, 3, 456789, tzinfo=UTC)
    with db.engine.begin() as conn:
        from elspeth.core.landscape.schema import runs_table

        conn.execute(update(runs_table).where(runs_table.c.run_id == run_id).values(status="completed", completed_at=completed))
        conn.execute(
            audit_export_snapshot_chunks_table.insert().values(
                snapshot_id=snapshot_id,
                ordinal=0,
                content_ref=f"sha256:{'3' * 64}",
                content_hash="3" * 64,
                size_bytes=10,
                record_count=1,
                predecessor_seal_hash=None,
                cumulative_records=1,
                cumulative_bytes=10,
                chunk_seal_hash="4" * 64,
            )
        )
        conn.execute(
            audit_export_snapshots_table.insert().values(
                snapshot_id=snapshot_id,
                source_run_id=run_id,
                source_status="completed",
                source_completed_at=completed,
                exported_at=completed,
                registry_key_hash="5" * 64,
                exporter_version="v2",
                serialization_version="audit-export-v2",
                export_format="json",
                signing_mode="unsigned",
                signer_key_id="UNSIGNED",
                derivation_version="audit-export-derivation-v1",
                public_export_config_hash="6" * 64,
                chunking_algorithm_version="chunk-v1",
                per_chunk_record_limit=100,
                per_chunk_byte_limit=1024,
                record_count=1,
                total_bytes=10,
                chunk_count=1,
                terminal_chunk_ordinal=0,
                content_store_id="durable-store",
                manifest_hash="7" * 64,
                last_chunk_seal_hash="4" * 64,
                snapshot_hash="8" * 64,
                snapshot_seal_hash="9" * 64,
                signature_hex=None,
                record_chain_algorithm="sha256_concat_record_sha256_v1",
                final_hash="a" * 64,
                signed_manifest_schema="elspeth.audit-export-manifest.v2",
                signed_manifest_hash="b" * 64,
                signed_manifest_ref=f"sha256:{'b' * 64}",
                signed_manifest_size_bytes=128,
            )
        )
    return snapshot_id


def test_export_reservation_is_zero_member_and_idempotent(db_factory: tuple[LandscapeDB, RecorderFactory]) -> None:
    db, factory = db_factory
    run_id, sink_id, _members = _pipeline_members(factory, 1)
    snapshot_id = _insert_snapshot(db, run_id)
    request = SinkEffectReservationRequest(
        run_id=run_id,
        sink_node_id=sink_id,
        role=SinkEffectRole.PRIMARY,
        input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
        requested_target_hash="c" * 64,
        members=(),
        audit_export_snapshot_id=snapshot_id,
        config_hash="c" * 64,
        replacing_target=False,
        primary_effect_id=None,
    )

    first = factory.execution.sink_effects.reserve(request)
    second = factory.execution.sink_effects.reserve(request)

    assert first.new_effect is not None
    assert second.new_effect is None
    assert second.open_effect_ids == (first.new_effect.effect_id,)
    assert factory.execution.sink_effects.get_members(first.new_effect.effect_id) == ()
    with db.read_only_connection() as conn:
        assert conn.scalar(select(func.count()).select_from(sink_effect_export_snapshots_table)) == 1


def test_concurrent_export_reservation_reuses_one_effect_and_association(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    db, factory = db_factory
    run_id, sink_id, _members = _pipeline_members(factory, 1)
    snapshot_id = _insert_snapshot(db, run_id)
    request = SinkEffectReservationRequest(
        run_id=run_id,
        sink_node_id=sink_id,
        role=SinkEffectRole.PRIMARY,
        input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
        requested_target_hash="c" * 64,
        members=(),
        audit_export_snapshot_id=snapshot_id,
        config_hash="c" * 64,
        replacing_target=False,
        primary_effect_id=None,
    )
    second = make_factory(db)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (
            pool.submit(factory.execution.sink_effects.reserve, request),
            pool.submit(second.execution.sink_effects.reserve, request),
        )
        results = tuple(future.result(timeout=5) for future in futures)

    effect_ids = {result.new_effect.effect_id if result.new_effect is not None else result.open_effect_ids[0] for result in results}
    assert len(effect_ids) == 1
    with db.read_only_connection() as conn:
        assert conn.scalar(select(func.count()).select_from(sink_effects_table)) == 1
        assert conn.scalar(select(func.count()).select_from(sink_effect_members_table)) == 0
        assert conn.scalar(select(func.count()).select_from(sink_effect_export_snapshots_table)) == 1


@pytest.mark.parametrize(
    ("input_kind", "members_present", "snapshot_present", "match"),
    [
        (SinkEffectInputKind.PIPELINE_MEMBERS, False, False, "at least one member"),
        (SinkEffectInputKind.PIPELINE_MEMBERS, True, True, "cannot carry"),
        (SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT, True, True, "cannot carry"),
        (SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT, False, False, "snapshot"),
    ],
)
def test_input_kind_membership_xor_fails_before_repository_sql(
    db_factory: tuple[LandscapeDB, RecorderFactory],
    input_kind: SinkEffectInputKind,
    members_present: bool,
    snapshot_present: bool,
    match: str,
) -> None:
    db, factory = db_factory
    run_id, sink_id, members = _pipeline_members(factory, 1)
    with pytest.raises(ValueError, match=match):
        SinkEffectReservationRequest(
            run_id=run_id,
            sink_node_id=sink_id,
            role=SinkEffectRole.PRIMARY,
            input_kind=input_kind,
            requested_target_hash="c" * 64,
            members=members if members_present else (),
            audit_export_snapshot_id="d" * 64 if snapshot_present else None,
            config_hash="c" * 64,
            replacing_target=False,
            primary_effect_id=None,
        )
    with db.read_only_connection() as conn:
        assert conn.scalar(select(func.count()).select_from(sink_effects_table)) == 0
