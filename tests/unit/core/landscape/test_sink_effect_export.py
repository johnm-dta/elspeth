"""Complete, credential-safe export of durable sink-effect history."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from elspeth.contracts.sink_effects import SinkEffectAttemptAction, SinkEffectAttemptRequest
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.exporter import LandscapeExporter
from elspeth.core.landscape.factory import RecorderFactory
from tests.fixtures.landscape import make_factory, make_landscape_db
from tests.unit.core.landscape.test_sink_effect_finalization import _prepared


@pytest.fixture
def db_factory() -> Iterator[tuple[LandscapeDB, RecorderFactory]]:
    db = make_landscape_db()
    try:
        yield db, make_factory(db)
    finally:
        db.close()


def test_export_preserves_abandoned_intent_and_safe_effect_history(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    db, factory = db_factory
    effect, _members, lease = _prepared(factory, count=1, replacing_target=True)
    attempt = factory.execution.sink_effects.begin_attempt(
        SinkEffectAttemptRequest(
            effect_id=effect.effect_id,
            member_ordinal=None,
            generation=lease.generation,
            action=SinkEffectAttemptAction.COMMIT,
            request_hash="c" * 64,
        )
    )

    records = list(LandscapeExporter(db)._iter_records(effect.run_id))
    stream_records = [record for record in records if record["record_type"] == "sink_effect_stream"]
    effect_records = [record for record in records if record["record_type"] == "sink_effect"]
    member_records = [record for record in records if record["record_type"] == "sink_effect_member"]
    attempt_records = [record for record in records if record["record_type"] == "sink_effect_attempt"]

    assert [record["stream_id"] for record in stream_records] == [effect.stream_id]
    assert [record["effect_id"] for record in effect_records] == [effect.effect_id]
    assert [record["ordinal"] for record in member_records] == [0]
    assert attempt_records == [
        {
            "record_type": "sink_effect_attempt",
            "run_id": effect.run_id,
            "attempt_id": attempt.attempt_id,
            "effect_id": effect.effect_id,
            "attempt_index": 0,
            "member_ordinal": None,
            "generation": lease.generation,
            "action": "commit",
            "call_kind": "commit",
            "request_hash": "c" * 64,
            "state": "intent",
            "evidence_hash": None,
            "started_at": attempt.started_at.isoformat(),
            "completed_at": None,
            "latency_ms": None,
        }
    ]
    assert "target_json" not in effect_records[0]
    assert "plan_json" not in effect_records[0]
    assert "evidence_json" not in attempt_records[0]


def test_attempt_export_uses_stable_per_effect_call_indexes(
    db_factory: tuple[LandscapeDB, RecorderFactory],
) -> None:
    db, factory = db_factory
    effect, _members, lease = _prepared(factory, count=1)
    first = factory.execution.sink_effects.begin_attempt(
        SinkEffectAttemptRequest(
            effect_id=effect.effect_id,
            member_ordinal=None,
            generation=lease.generation,
            action=SinkEffectAttemptAction.COMMIT,
            request_hash="d" * 64,
        )
    )
    second = factory.execution.sink_effects.begin_attempt(
        SinkEffectAttemptRequest(
            effect_id=effect.effect_id,
            member_ordinal=0,
            generation=lease.generation,
            action=SinkEffectAttemptAction.COMMIT,
            request_hash="e" * 64,
        )
    )

    attempts = [record for record in LandscapeExporter(db)._iter_records(effect.run_id) if record["record_type"] == "sink_effect_attempt"]

    assert [(record["attempt_id"], record["attempt_index"]) for record in attempts] == [
        (first.attempt_id, 0),
        (second.attempt_id, 1),
    ]
