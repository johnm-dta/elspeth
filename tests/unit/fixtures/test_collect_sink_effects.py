from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

import pytest
from tests.fixtures.plugins import CollectSink

from elspeth.contracts import (
    RestrictedSinkEffectContext,
    SinkEffectInputKind,
    SinkEffectInspectionRequest,
    SinkEffectMember,
    SinkEffectPipelineMembersInput,
    SinkEffectPrepareRequest,
    SinkEffectReconcileKind,
)
from elspeth.contracts.hashing import canonical_json
from elspeth.engine.orchestrator.preflight import SinkEffectCapabilityError, validate_sink_effect_capability


def _member(row: dict[str, object]) -> SinkEffectMember:
    return SinkEffectMember(
        ordinal=0,
        token_id="token-1",
        row_id="row-1",
        ingest_sequence=0,
        lineage_json="[]",
        lineage_hash=sha256(b"[]").hexdigest(),
        payload_hash=sha256(canonical_json(row).encode()).hexdigest(),
        row=row,
    )


def test_collect_sink_declares_only_pipeline_write_effect_capability() -> None:
    sink = CollectSink()

    validate_sink_effect_capability(
        sink,
        mode="write",
        required_input_kind=SinkEffectInputKind.PIPELINE_MEMBERS,
    )

    with pytest.raises(SinkEffectCapabilityError, match="audit_export_snapshot"):
        validate_sink_effect_capability(
            sink,
            mode="write",
            required_input_kind=SinkEffectInputKind.AUDIT_EXPORT_SNAPSHOT,
        )


def test_collect_sink_effect_commit_is_idempotent_and_reconcilable() -> None:
    sink = CollectSink()
    ctx = RestrictedSinkEffectContext(
        run_id="run-1",
        run_started_at=datetime(2026, 7, 16, tzinfo=UTC),
        operation_id="operation-1",
        sink_node_id="sink-1",
    )
    inspection = sink.inspect_effect(
        SinkEffectInspectionRequest(
            effect_id="effect-1",
            target="memory://collect/effect-1",
            predecessor_descriptor=None,
        ),
        ctx,
    )
    plan = sink.prepare_effect(
        SinkEffectPrepareRequest(
            effect_id="effect-1",
            effect_input=SinkEffectPipelineMembersInput(
                members=(_member({"value": 1}),),
                target_snapshot_members=(),
            ),
            inspection=inspection,
        ),
        ctx,
    )

    first = sink.commit_effect(plan, ctx)
    second = sink.commit_effect(plan, ctx)
    reconciled = sink.reconcile_effect(plan, ctx)

    assert sink.results == [{"value": 1}]
    assert first == second
    assert first.accepted_ordinals == (0,)
    assert first.diverted_ordinals == ()
    assert reconciled.kind is SinkEffectReconcileKind.APPLIED_WITH_EXACT_DESCRIPTOR
    assert reconciled.descriptor == first.descriptor
