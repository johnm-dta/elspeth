"""Durable sink-effect inspection, plan, lease, and attempt lifecycle."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import timedelta

import pytest
from sqlalchemy import select

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    SinkEffectAttemptAction,
    SinkEffectAttemptState,
    SinkEffectDescriptorMode,
    SinkEffectInputKind,
    SinkEffectInspectionMode,
    SinkEffectPlan,
    SinkEffectState,
)
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.errors import LandscapeRecordError
from elspeth.core.landscape.execution.sink_effect_lifecycle import (
    SinkEffectAttemptRequest,
    SinkEffectAttemptResult,
)
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import calls_table, operations_table
from tests.fixtures.landscape import make_factory, make_landscape_db
from tests.unit.core.landscape.test_sink_effect_reservation import _pipeline_members, _pipeline_request


@pytest.fixture
def db_factory() -> Iterator[tuple[LandscapeDB, RecorderFactory]]:
    db = make_landscape_db()
    try:
        yield db, make_factory(db)
    finally:
        db.close()


def _reserved(factory: RecorderFactory):
    run_id, sink_id, members = _pipeline_members(factory, 1)
    effect = factory.execution.sink_effects.reserve(_pipeline_request(run_id, sink_id, members)).new_effect
    assert effect is not None
    return effect


def _plan(effect_id: str, *, plan_hash: str = "a" * 64) -> SinkEffectPlan:
    return SinkEffectPlan(
        effect_id=effect_id,
        protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
        input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
        descriptor_mode=SinkEffectDescriptorMode.RESULT_DERIVED,
        inspection_mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
        target="file:///tmp/result.jsonl",
        plan_hash=plan_hash,
        payload_hash="b" * 64,
        expected_descriptor=None,
        safe_evidence={"inspection_reference": "no-inspection-required:v1"},
    )


def test_reserved_effect_cannot_lease_before_complete_plan(db_factory: tuple[LandscapeDB, RecorderFactory]) -> None:
    _db, factory = db_factory
    effect = _reserved(factory)

    with pytest.raises(LandscapeRecordError, match="prepared"):
        factory.execution.sink_effects.acquire_lease(effect.effect_id, owner="worker-a", ttl=timedelta(seconds=30))


def test_concurrent_plan_cas_accepts_equal_and_rejects_divergent(db_factory: tuple[LandscapeDB, RecorderFactory]) -> None:
    _db, factory = db_factory
    effect = _reserved(factory)
    repo = factory.execution.sink_effects

    first = repo.complete_plan(effect.effect_id, _plan(effect.effect_id))
    second = repo.complete_plan(effect.effect_id, _plan(effect.effect_id))

    assert first == second
    assert first.state is SinkEffectState.PREPARED
    with pytest.raises(AuditIntegrityError, match="divergent plan"):
        repo.complete_plan(effect.effect_id, _plan(effect.effect_id, plan_hash="c" * 64))


def test_inspected_plan_requires_returned_inspection_attempt(db_factory: tuple[LandscapeDB, RecorderFactory]) -> None:
    _db, factory = db_factory
    effect = _reserved(factory)
    inspected = replace(_plan(effect.effect_id), inspection_mode=SinkEffectInspectionMode.INSPECTED)

    with pytest.raises(LandscapeRecordError, match="returned inspect"):
        factory.execution.sink_effects.complete_plan(effect.effect_id, inspected)


def test_lease_takeover_increments_generation_and_fences_stale_results(db_factory: tuple[LandscapeDB, RecorderFactory]) -> None:
    _db, factory = db_factory
    effect = _reserved(factory)
    repo = factory.execution.sink_effects
    repo.complete_plan(effect.effect_id, _plan(effect.effect_id))
    first = repo.acquire_lease(effect.effect_id, owner="worker-a", ttl=timedelta(microseconds=1))
    second = repo.takeover_expired(effect.effect_id, owner="worker-b", ttl=timedelta(seconds=30))
    assert second.generation == first.generation + 1

    with pytest.raises(LandscapeRecordError, match="stale generation"):
        repo.begin_attempt(
            SinkEffectAttemptRequest(
                effect_id=effect.effect_id,
                member_ordinal=None,
                generation=first.generation,
                action=SinkEffectAttemptAction.COMMIT,
                request_hash="d" * 64,
            )
        )


def test_abandoned_commit_intent_becomes_response_lost_with_stable_call_index(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    db, factory = db_factory
    effect = _reserved(factory)
    repo = factory.execution.sink_effects
    repo.complete_plan(effect.effect_id, _plan(effect.effect_id))
    lease = repo.acquire_lease(effect.effect_id, owner="worker-a", ttl=timedelta(seconds=30))
    attempt = repo.begin_attempt(
        SinkEffectAttemptRequest(
            effect_id=effect.effect_id,
            member_ordinal=None,
            generation=lease.generation,
            action=SinkEffectAttemptAction.COMMIT,
            request_hash="e" * 64,
        )
    )

    recovered = repo.mark_response_lost(attempt.attempt_id)

    assert recovered.state is SinkEffectAttemptState.RESPONSE_LOST
    with db.read_only_connection() as conn:
        operation_id = conn.scalar(select(operations_table.c.operation_id).where(operations_table.c.sink_effect_id == effect.effect_id))
        calls = conn.execute(
            select(calls_table.c.call_index, calls_table.c.status, calls_table.c.error_json)
            .where(calls_table.c.operation_id == operation_id)
            .order_by(calls_table.c.call_index)
        ).fetchall()
    assert [(row.call_index, row.status) for row in calls] == [(0, "error")]
    assert "response_lost" in (calls[0].error_json or "")


def test_attempt_return_records_one_success_call_and_is_idempotent(db_factory: tuple[LandscapeDB, RecorderFactory]) -> None:
    db, factory = db_factory
    effect = _reserved(factory)
    repo = factory.execution.sink_effects
    repo.complete_plan(effect.effect_id, _plan(effect.effect_id))
    lease = repo.acquire_lease(effect.effect_id, owner="worker-a", ttl=timedelta(seconds=30))
    attempt = repo.begin_attempt(
        SinkEffectAttemptRequest(
            effect_id=effect.effect_id,
            member_ordinal=None,
            generation=lease.generation,
            action=SinkEffectAttemptAction.COMMIT,
            request_hash="f" * 64,
        )
    )
    result = SinkEffectAttemptResult(attempt_id=attempt.attempt_id, evidence={"descriptor": "exact"}, latency_ms=2.5)

    first = repo.record_attempt_result(result)
    second = repo.record_attempt_result(result)

    assert first == second
    assert first.state is SinkEffectAttemptState.RETURNED
    with db.read_only_connection() as conn:
        operation_id = conn.scalar(select(operations_table.c.operation_id).where(operations_table.c.sink_effect_id == effect.effect_id))
        assert conn.scalar(select(calls_table.c.call_index).where(calls_table.c.operation_id == operation_id)) == 0
