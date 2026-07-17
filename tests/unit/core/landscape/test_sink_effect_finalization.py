"""Atomic sink-effect finalization and retry convergence."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlalchemy import delete, func, select, update

from elspeth.contracts import CallType, NodeStateStatus, NodeType, TerminalOutcome, TerminalPath
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    SinkEffectAttemptAction,
    SinkEffectCommitResult,
    SinkEffectDescriptorMode,
    SinkEffectInputKind,
    SinkEffectInspectionMode,
    SinkEffectPlan,
    SinkEffectReconcileKind,
    SinkEffectReconcileResult,
    SinkEffectRole,
    SinkEffectState,
)
from elspeth.core.canonical import canonical_json, stable_hash
from elspeth.core.landscape._helpers import now
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution.sink_effect_attempt_results import encode_sink_effect_returned_result
from elspeth.core.landscape.execution.sink_effect_finalization import (
    SinkEffectFinalizationMember,
    SinkEffectFinalizeRequest,
)
from elspeth.core.landscape.execution.sink_effect_identity import compute_pipeline_effect_identity
from elspeth.core.landscape.execution.sink_effect_lifecycle import SinkEffectAttemptRequest, SinkEffectAttemptResult
from elspeth.core.landscape.execution.sink_effect_reservation import SinkEffectReservationRequest
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import (
    artifacts_table,
    node_states_table,
    operations_table,
    sink_effect_attempts_table,
    sink_effect_members_table,
    sink_effects_table,
    token_outcomes_table,
)
from tests.fixtures.landscape import make_factory, make_landscape_db, register_test_node
from tests.unit.core.landscape.test_sink_effect_reservation import _pipeline_members, _pipeline_request


@pytest.fixture
def db_factory() -> Iterator[tuple[LandscapeDB, RecorderFactory]]:
    db = make_landscape_db()
    try:
        yield db, make_factory(db)
    finally:
        db.close()


def _descriptor(*, content_hash: str = "d" * 64) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        artifact_type="file",
        path_or_uri="file:///tmp/result.jsonl",
        content_hash=content_hash,
        size_bytes=12,
    )


def _prepared(
    factory: RecorderFactory,
    *,
    count: int = 2,
    replacing_target: bool = False,
    descriptor_mode: SinkEffectDescriptorMode = SinkEffectDescriptorMode.PRECOMPUTED,
) -> tuple[object, tuple[object, ...], object]:
    run_id, sink_id, members = _pipeline_members(factory, count)
    effect = factory.execution.sink_effects.reserve(
        _pipeline_request(run_id, sink_id, members, replacing_target=replacing_target)
    ).new_effect
    assert effect is not None
    descriptor = _descriptor()
    claim = factory.execution.sink_effects.claim_preparation(
        effect.effect_id,
        owner="worker-a",
        ttl=timedelta(seconds=30),
    )
    factory.execution.sink_effects.complete_plan(
        effect.effect_id,
        SinkEffectPlan(
            effect_id=effect.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=descriptor_mode,
            inspection_mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
            target=descriptor.path_or_uri,
            plan_hash="a" * 64,
            payload_hash="b" * 64,
            expected_descriptor=descriptor if descriptor_mode is SinkEffectDescriptorMode.PRECOMPUTED else None,
            safe_evidence={"inspection_reference": "no-inspection-required:v1"},
        ),
        claim=claim,
    )
    lease = factory.execution.sink_effects.acquire_lease(
        effect.effect_id,
        owner="worker-a",
        ttl=timedelta(seconds=30),
    )
    return effect, members, lease


def _request(
    factory: RecorderFactory,
    effect: object,
    members: tuple[object, ...],
    lease: object,
    *,
    evidence: dict[str, object] | None = None,
    attempt_evidence: dict[str, object] | None = None,
) -> SinkEffectFinalizeRequest:
    exact_evidence = {"result": "exact"} if evidence is None else evidence
    attempt = factory.execution.sink_effects.begin_attempt(
        SinkEffectAttemptRequest(
            effect_id=effect.effect_id,
            member_ordinal=None,
            generation=lease.generation,
            action=SinkEffectAttemptAction.COMMIT,
            call_kind=CallType.FILESYSTEM,
            request_hash="a" * 64,
        )
    )
    recorded_evidence = (
        encode_sink_effect_returned_result(
            SinkEffectCommitResult(
                descriptor=_descriptor(),
                evidence=exact_evidence,
                accepted_ordinals=tuple(range(len(members))),
                diverted_ordinals=(),
            )
        )
        if attempt_evidence is None
        else attempt_evidence
    )
    factory.execution.sink_effects.record_attempt_result(
        SinkEffectAttemptResult(attempt_id=attempt.attempt_id, evidence=recorded_evidence, latency_ms=1.0)
    )
    return SinkEffectFinalizeRequest(
        effect_id=effect.effect_id,
        lease_owner=lease.owner,
        generation=lease.generation,
        descriptor=_descriptor(),
        publication_performed=True,
        publication_evidence_kind="returned",
        accepted_ordinals=tuple(range(len(members))),
        diverted_ordinals=(),
        evidence=exact_evidence,
        members=tuple(
            SinkEffectFinalizationMember(
                ordinal=ordinal,
                output_data={"row": {"ordinal": ordinal}},
                duration_ms=1.0,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="sink",
            )
            for ordinal in range(len(members))
        ),
        attempt_id=attempt.attempt_id,
    )


def test_finalization_is_one_transaction_and_retry_returns_winner(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    db, factory = db_factory
    effect, members, lease = _prepared(factory)
    request = _request(factory, effect, members, lease)

    first = factory.execution.sink_effects.finalize(request)
    second = factory.execution.sink_effects.finalize(request)

    assert first == second
    assert first.effect.state is SinkEffectState.FINALIZED
    assert first.artifact.sink_effect_id == effect.effect_id
    with db.read_only_connection() as conn:
        assert (
            conn.scalar(select(func.count()).select_from(artifacts_table).where(artifacts_table.c.sink_effect_id == effect.effect_id)) == 1
        )
        assert conn.scalar(select(func.count()).select_from(node_states_table).where(node_states_table.c.status == "completed")) == 2
        assert conn.scalar(select(func.count()).select_from(token_outcomes_table).where(token_outcomes_table.c.completed == 1)) == 2
        assert conn.scalar(select(operations_table.c.status).where(operations_table.c.sink_effect_id == effect.effect_id)) == "completed"


def test_new_attempt_state_witness_is_resolved_not_part_of_identity(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    db, factory = db_factory
    effect, members, lease = _prepared(factory, count=1)
    with db.engine.begin() as conn:
        current = conn.execute(select(node_states_table).where(node_states_table.c.token_id == members[0].token_id)).one()
        conn.execute(
            update(node_states_table)
            .where(node_states_table.c.state_id == current.state_id)
            .values(status=NodeStateStatus.FAILED.value, error_json='{"phase":"resume"}', duration_ms=0.0, completed_at=current.started_at)
        )
    factory.execution.begin_node_state(
        token_id=members[0].token_id,
        node_id=effect.sink_node_id,
        run_id=effect.run_id,
        step_index=0,
        input_data={"ordinal": 0},
        attempt=1,
    )

    result = factory.execution.sink_effects.finalize(_request(factory, effect, members, lease))

    assert result.effect.effect_id == effect.effect_id
    assert result.artifact.sink_effect_id == effect.effect_id


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lease_owner", "worker-b", "owner"),
        ("generation", 999, "generation"),
        ("descriptor", _descriptor(content_hash="e" * 64), "descriptor"),
    ],
)
def test_finalization_refuses_stale_or_divergent_authority(
    db_factory: tuple[LandscapeDB, RecorderFactory],
    field: str,
    value: object,
    message: str,
) -> None:
    _db, factory = db_factory
    effect, members, lease = _prepared(factory, count=1)
    request = replace(_request(factory, effect, members, lease), **{field: value})

    with pytest.raises(LandscapeRecordError, match=message):
        factory.execution.sink_effects.finalize(request)


def test_stream_head_cas_advances_exact_predecessor(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    _db, factory = db_factory
    effect, members, lease = _prepared(factory, count=1, replacing_target=True)

    result = factory.execution.sink_effects.finalize(_request(factory, effect, members, lease))
    stream = factory.execution.sink_effects.get_stream(effect.stream_id)

    assert result.effect.state is SinkEffectState.FINALIZED
    assert stream is not None
    assert stream.head_effect_id == effect.effect_id
    assert stream.head_descriptor_hash == result.effect.result_descriptor_hash


def test_finalization_response_loss_retry_observes_committed_winner(
    db_factory: tuple[LandscapeDB, RecorderFactory],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _db, factory = db_factory
    effect, members, lease = _prepared(factory, count=1)
    request = _request(factory, effect, members, lease)
    finalizer = factory.execution.sink_effects._finalization
    original = finalizer._after_commit
    calls = 0

    def lose_once(effect_id: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError(effect_id)
        original(effect_id)

    monkeypatch.setattr(finalizer, "_after_commit", lose_once)
    with pytest.raises(ConnectionError):
        factory.execution.sink_effects.finalize(request)

    assert factory.execution.sink_effects.finalize(request).effect.state is SinkEffectState.FINALIZED


def test_failed_finalization_rolls_back_every_audit_write(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    db, factory = db_factory
    effect, members, lease = _prepared(factory, count=1)
    divergent = replace(_request(factory, effect, members, lease), descriptor=_descriptor(content_hash="f" * 64))

    with pytest.raises(LandscapeRecordError):
        factory.execution.sink_effects.finalize(divergent)

    with db.read_only_connection() as conn:
        assert (
            conn.scalar(select(func.count()).select_from(artifacts_table).where(artifacts_table.c.sink_effect_id == effect.effect_id)) == 0
        )
        assert conn.scalar(select(func.count()).select_from(token_outcomes_table)) == 0
        assert conn.scalar(select(sink_effects_table.c.state).where(sink_effects_table.c.effect_id == effect.effect_id)) == "in_flight"


def test_result_derived_descriptor_requires_exact_authoritative_evidence(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    _db, factory = db_factory
    effect, members, lease = _prepared(factory, count=1, descriptor_mode=SinkEffectDescriptorMode.RESULT_DERIVED)
    exact_evidence = {
        "accepted_ordinals": [0],
        "descriptor": {
            "artifact_type": _descriptor().artifact_type,
            "content_hash": _descriptor().content_hash,
            "metadata": None,
            "path_or_uri": _descriptor().path_or_uri,
            "size_bytes": _descriptor().size_bytes,
        },
        "diverted_ordinals": [],
    }
    request = _request(factory, effect, members, lease, evidence=exact_evidence)

    result = factory.execution.sink_effects.finalize(request)

    assert result.artifact.content_hash == request.descriptor.content_hash


def test_result_derived_descriptor_refuses_non_authoritative_evidence(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    _db, factory = db_factory
    effect, members, lease = _prepared(factory, count=1, descriptor_mode=SinkEffectDescriptorMode.RESULT_DERIVED)

    with pytest.raises(LandscapeRecordError, match="result-derived evidence"):
        factory.execution.sink_effects.finalize(_request(factory, effect, members, lease))


def test_result_derived_reconciled_retry_preserves_ordinals_and_returns_winner(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    """A repeated RESULT_DERIVED reconciled finalization must reconstruct the
    accepted/diverted ordinal partition byte-for-byte and return the committed
    winner instead of raising (elspeth-1ce83f7249)."""
    _db, factory = db_factory
    effect, _members, lease = _prepared(factory, count=2, descriptor_mode=SinkEffectDescriptorMode.RESULT_DERIVED)
    descriptor = _descriptor()
    evidence = {
        "accepted_ordinals": [0],
        "descriptor": {
            "artifact_type": descriptor.artifact_type,
            "content_hash": descriptor.content_hash,
            "metadata": None,
            "path_or_uri": descriptor.path_or_uri,
            "size_bytes": descriptor.size_bytes,
        },
        "diverted_ordinals": [1],
    }
    reconciliation = SinkEffectReconcileResult.applied(
        descriptor,
        evidence=evidence,
        accepted_ordinals=(0,),
        diverted_ordinals=(1,),
    )
    attempt = factory.execution.sink_effects.begin_attempt(
        SinkEffectAttemptRequest(
            effect_id=effect.effect_id,
            member_ordinal=None,
            generation=lease.generation,
            action=SinkEffectAttemptAction.RECONCILE,
            call_kind=CallType.FILESYSTEM,
            request_hash="a" * 64,
        )
    )
    factory.execution.sink_effects.record_attempt_result(
        SinkEffectAttemptResult(
            attempt_id=attempt.attempt_id,
            evidence=encode_sink_effect_returned_result(reconciliation),
            latency_ms=1.0,
        )
    )
    request = SinkEffectFinalizeRequest(
        effect_id=effect.effect_id,
        lease_owner=lease.owner,
        generation=lease.generation,
        descriptor=descriptor,
        publication_performed=True,
        publication_evidence_kind="reconciled",
        accepted_ordinals=(0,),
        diverted_ordinals=(1,),
        evidence=evidence,
        members=(
            SinkEffectFinalizationMember(
                ordinal=0,
                output_data={"row": {"ordinal": 0}},
                duration_ms=1.0,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="sink",
            ),
        ),
        attempt_id=attempt.attempt_id,
        reconcile_kind=SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR,
    )

    first = factory.execution.sink_effects.finalize(request)
    second = factory.execution.sink_effects.finalize(request)

    assert first.effect.state is SinkEffectState.FINALIZED
    assert second.effect == first.effect
    assert second.artifact == first.artifact
    assert second.state_ids == first.state_ids
    assert second.outcome_ids == first.outcome_ids


def test_no_publication_finalization_registers_virtual_artifact_without_lease(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    db, factory = db_factory
    run_id, sink_id, members = _pipeline_members(factory, 1)
    effect = factory.execution.sink_effects.reserve(_pipeline_request(run_id, sink_id, members)).new_effect
    assert effect is not None
    descriptor = ArtifactDescriptor(
        artifact_type="file",
        path_or_uri="/virtual/empty",
        content_hash="0" * 64,
        size_bytes=0,
    )
    descriptor_payload = {
        "artifact_type": descriptor.artifact_type,
        "content_hash": descriptor.content_hash,
        "metadata": None,
        "path_or_uri": descriptor.path_or_uri,
        "size_bytes": descriptor.size_bytes,
    }
    timestamp = now()
    plan_json = canonical_json(
        {
            "descriptor_mode": SinkEffectDescriptorMode.NO_PUBLICATION.value,
            "effect_id": effect.effect_id,
            "expected_descriptor": descriptor_payload,
            "input_kind": SinkEffectInputKind.PIPELINE_MEMBERS.value,
            "inspection_mode": SinkEffectInspectionMode.NO_INSPECTION_REQUIRED.value,
            "payload_hash": "b" * 64,
            "plan_hash": "a" * 64,
            "protocol_version": SINK_EFFECT_PROTOCOL_VERSION,
            "safe_evidence": {"inspection_reference": "no-inspection-required:v1"},
            "target": descriptor.path_or_uri,
        }
    )
    with db.engine.begin() as conn:
        conn.execute(
            update(sink_effects_table)
            .where(sink_effects_table.c.effect_id == effect.effect_id)
            .values(
                state=SinkEffectState.PREPARED.value,
                inspection_mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED.value,
                inspection_attempt_id=stable_hash(
                    {
                        "effect_id": effect.effect_id,
                        "reference": "no-inspection-required:v1",
                        "schema": "sink-effect-inspection-sentinel-v1",
                    }
                ),
                plan_json=plan_json,
                plan_hash="a" * 64,
                descriptor_mode=SinkEffectDescriptorMode.NO_PUBLICATION.value,
                expected_descriptor_hash=stable_hash(descriptor_payload),
                precondition_hash="c" * 64,
                prepared_at=timestamp,
                updated_at=timestamp,
            )
        )
        conn.execute(
            update(node_states_table)
            .where(node_states_table.c.token_id == members[0].token_id)
            .values(status=NodeStateStatus.FAILED.value, error_json='{"phase":"diverted"}', duration_ms=0.0, completed_at=timestamp)
        )
    request = SinkEffectFinalizeRequest(
        effect_id=effect.effect_id,
        lease_owner=None,
        generation=0,
        descriptor=descriptor,
        publication_performed=False,
        publication_evidence_kind="virtual",
        accepted_ordinals=(),
        diverted_ordinals=(0,),
        evidence={"kind": "virtual_empty"},
        members=(),
    )

    result = factory.execution.sink_effects.finalize(request)

    assert result.artifact.publication_performed is False
    assert result.artifact.publication_evidence_kind == "virtual"


def test_missing_current_state_witness_refuses_before_artifact_or_outcome(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    db, factory = db_factory
    effect, members, lease = _prepared(factory, count=1)
    with db.engine.begin() as conn:
        conn.execute(delete(node_states_table).where(node_states_table.c.token_id == members[0].token_id))

    with pytest.raises(LandscapeRecordError, match="current open state witness"):
        factory.execution.sink_effects.finalize(_request(factory, effect, members, lease))

    with db.read_only_connection() as conn:
        assert (
            conn.scalar(select(func.count()).select_from(artifacts_table).where(artifacts_table.c.sink_effect_id == effect.effect_id)) == 0
        )


def test_finalized_retry_refuses_a_divergent_descriptor(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    _db, factory = db_factory
    effect, members, lease = _prepared(factory, count=1)
    request = _request(factory, effect, members, lease)
    factory.execution.sink_effects.finalize(request)

    with pytest.raises(LandscapeRecordError, match="descriptor"):
        factory.execution.sink_effects.finalize(replace(request, descriptor=_descriptor(content_hash="f" * 64)))


def test_finalization_refuses_raw_evidence_attempt_encoding(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    _db, factory = db_factory
    effect, members, lease = _prepared(factory, count=1)

    with pytest.raises(LandscapeRecordError, match="exact evidence"):
        _request(factory, effect, members, lease, attempt_evidence={"result": "exact"})

    divergent_envelope = encode_sink_effect_returned_result(
        SinkEffectCommitResult(
            descriptor=_descriptor(),
            evidence={"result": "divergent"},
            accepted_ordinals=tuple(range(len(members))),
            diverted_ordinals=(),
        )
    )
    request = _request(factory, effect, members, lease, attempt_evidence=divergent_envelope)
    with pytest.raises(LandscapeRecordError, match="finalization evidence differs"):
        factory.execution.sink_effects.finalize(request)


def test_finalized_retry_refuses_raw_evidence_attempt_encoding(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    db, factory = db_factory
    effect, members, lease = _prepared(factory, count=1)
    request = _request(factory, effect, members, lease)
    factory.execution.sink_effects.finalize(request)
    with db.engine.begin() as conn:
        conn.execute(
            update(sink_effect_attempts_table)
            .where(sink_effect_attempts_table.c.attempt_id == request.attempt_id)
            .values(evidence_json=canonical_json({"result": "exact"}))
        )

    with pytest.raises(LandscapeRecordError, match="finalized retry attempt/evidence differs"):
        factory.execution.sink_effects.finalize(request)


@pytest.mark.parametrize("primary_member_corruption", [None, "accepted", "not_finalized"])
def test_failsink_finalization_requires_and_uses_exact_primary_linkage(
    db_factory: tuple[LandscapeDB, RecorderFactory],
    primary_member_corruption: str | None,
) -> None:
    db, factory = db_factory
    primary, members, primary_lease = _prepared(factory, count=1)
    primary_attempt = factory.execution.sink_effects.begin_attempt(
        SinkEffectAttemptRequest(
            effect_id=primary.effect_id,
            member_ordinal=None,
            generation=primary_lease.generation,
            action=SinkEffectAttemptAction.COMMIT,
            call_kind=CallType.FILESYSTEM,
            request_hash="a" * 64,
        )
    )
    factory.execution.sink_effects.record_attempt_result(
        SinkEffectAttemptResult(
            attempt_id=primary_attempt.attempt_id,
            evidence=encode_sink_effect_returned_result(
                SinkEffectCommitResult(
                    descriptor=_descriptor(),
                    evidence={"result": "primary-exact"},
                    accepted_ordinals=(),
                    diverted_ordinals=(0,),
                )
            ),
            latency_ms=1.0,
        )
    )
    factory.execution.sink_effects.finalize(
        SinkEffectFinalizeRequest(
            effect_id=primary.effect_id,
            lease_owner=primary_lease.owner,
            generation=primary_lease.generation,
            descriptor=_descriptor(),
            publication_performed=True,
            publication_evidence_kind="returned",
            accepted_ordinals=(),
            diverted_ordinals=(0,),
            evidence={"result": "primary-exact"},
            members=(),
            attempt_id=primary_attempt.attempt_id,
        )
    )
    if primary_member_corruption is not None:
        column_values = (
            {"prepared_disposition": "accepted"}
            if primary_member_corruption == "accepted"
            else {"member_state": SinkEffectState.PREPARED.value}
        )
        with db.engine.begin() as conn:
            conn.execute(
                update(sink_effect_members_table).where(sink_effect_members_table.c.effect_id == primary.effect_id).values(**column_values)
            )

    failsink_id = register_test_node(
        factory.data_flow,
        primary.run_id,
        "failsink",
        node_type=NodeType.SINK,
        plugin_name="failsink",
    )
    factory.execution.begin_node_state(
        token_id=members[0].token_id,
        node_id=failsink_id,
        run_id=primary.run_id,
        step_index=1,
        input_data={"ordinal": 0},
    )
    identity = compute_pipeline_effect_identity(
        run_id=primary.run_id,
        sink_node_id=failsink_id,
        role=SinkEffectRole.FAILSINK,
        sink_config={"name": "failsink"},
        target_config={"path": "failsink.jsonl"},
        members=members,
    )
    failsink = factory.execution.sink_effects.reserve(
        SinkEffectReservationRequest(
            run_id=primary.run_id,
            sink_node_id=failsink_id,
            role=SinkEffectRole.FAILSINK,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            requested_target_hash=identity.requested_target_hash,
            members=members,
            audit_export_snapshot_id=None,
            config_hash=identity.config_hash,
            replacing_target=False,
            primary_effect_id=primary.effect_id,
        )
    ).new_effect
    assert failsink is not None
    descriptor = ArtifactDescriptor(
        artifact_type="file",
        path_or_uri="file:///tmp/failsink.jsonl",
        content_hash="e" * 64,
        size_bytes=9,
    )
    failsink_claim = factory.execution.sink_effects.claim_preparation(
        failsink.effect_id,
        owner="worker-a",
        ttl=timedelta(seconds=30),
    )
    factory.execution.sink_effects.complete_plan(
        failsink.effect_id,
        SinkEffectPlan(
            effect_id=failsink.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
            target=descriptor.path_or_uri,
            plan_hash="1" * 64,
            payload_hash="2" * 64,
            expected_descriptor=descriptor,
            safe_evidence={"inspection_reference": "no-inspection-required:v1"},
        ),
        claim=failsink_claim,
    )
    lease = factory.execution.sink_effects.acquire_lease(failsink.effect_id, owner="worker-f", ttl=timedelta(seconds=30))
    attempt = factory.execution.sink_effects.begin_attempt(
        SinkEffectAttemptRequest(
            effect_id=failsink.effect_id,
            member_ordinal=None,
            generation=lease.generation,
            action=SinkEffectAttemptAction.COMMIT,
            call_kind=CallType.FILESYSTEM,
            request_hash="1" * 64,
        )
    )
    factory.execution.sink_effects.record_attempt_result(
        SinkEffectAttemptResult(
            attempt_id=attempt.attempt_id,
            evidence=encode_sink_effect_returned_result(
                SinkEffectCommitResult(
                    descriptor=descriptor,
                    evidence={"result": "failsink-exact"},
                    accepted_ordinals=(0,),
                    diverted_ordinals=(),
                )
            ),
            latency_ms=1.0,
        )
    )

    request = SinkEffectFinalizeRequest(
        effect_id=failsink.effect_id,
        lease_owner=lease.owner,
        generation=lease.generation,
        descriptor=descriptor,
        publication_performed=True,
        publication_evidence_kind="returned",
        accepted_ordinals=(0,),
        diverted_ordinals=(),
        evidence={"result": "failsink-exact"},
        members=(
            SinkEffectFinalizationMember(
                ordinal=0,
                output_data={"row": {"ordinal": 0}},
                duration_ms=1.0,
                outcome=TerminalOutcome.TRANSIENT,
                path=TerminalPath.SINK_FALLBACK_TO_FAILSINK,
                sink_name="failsink",
                error_hash="f" * 64,
            ),
        ),
        attempt_id=attempt.attempt_id,
    )

    if primary_member_corruption is not None:
        with pytest.raises(LandscapeRecordError, match="diverted finalized primary member"):
            factory.execution.sink_effects.finalize(request)
        return

    result = factory.execution.sink_effects.finalize(request)

    assert result.effect.primary_effect_id == primary.effect_id
    assert result.artifact.sink_effect_id == failsink.effect_id
    assert len(result.outcome_ids) == 1
