"""Caller-level contract for durable sink-effect execution."""

from __future__ import annotations

import ast
from datetime import timedelta
from pathlib import Path

import pytest

from elspeth.contracts import TerminalOutcome, TerminalPath
from elspeth.contracts.sink_effects import SinkEffectPipelineMembersInput
from elspeth.core.landscape.execution.sink_effect_finalization import SinkEffectFinalizationMember
from elspeth.core.landscape.execution.sink_effect_identity import compute_pipeline_effect_identity
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.executors.sink_effects import (
    SinkEffectCoordinator,
    SinkEffectExecutionRequest,
    SinkEffectExecutionSeam,
    SinkEffectInjectedFault,
)
from tests.fixtures.landscape import make_factory, make_landscape_db
from tests.fixtures.sink_effects import DuplicateObservableSink, DuplicateObservableTarget
from tests.unit.core.landscape.test_sink_effect_reservation import _pipeline_members, _pipeline_request

_ROOT = Path(__file__).parents[3]


def _production_calls(path: str, method: str) -> list[int]:
    source_path = _ROOT / path
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in {"sink", "failsink"}
    ]


def test_pipeline_executor_has_no_legacy_write_or_flush_publication_boundary() -> None:
    assert _production_calls("src/elspeth/engine/executors/sink.py", "write") == []
    assert _production_calls("src/elspeth/engine/executors/sink.py", "flush") == []


def test_audit_export_has_no_legacy_write_or_flush_publication_boundary() -> None:
    assert _production_calls("src/elspeth/engine/orchestrator/export.py", "write") == []
    assert _production_calls("src/elspeth/engine/orchestrator/export.py", "flush") == []


@pytest.mark.parametrize("seam", list(SinkEffectExecutionSeam))
def test_fresh_executor_retry_publishes_once(seam: SinkEffectExecutionSeam) -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run_id, sink_id, members = _pipeline_members(factory, 1)
        identity = compute_pipeline_effect_identity(
            run_id=run_id,
            sink_node_id=sink_id,
            role=_pipeline_request(run_id, sink_id, members).role,
            sink_config={"name": "duplicate-observable"},
            target_config={"path": "duplicate-observable.jsonl"},
            members=members,
        )
        # Reservation and input identity are independently constructed from the
        # same public configuration, so the coordinator must exact-check them.
        reservation = _pipeline_request(run_id, sink_id, identity.members)
        effect_input = SinkEffectPipelineMembersInput(
            members=identity.members,
            target_snapshot_members=identity.members,
        )
        request = SinkEffectExecutionRequest(
            reservation=reservation,
            effect_input=effect_input,
            finalization_members=(
                SinkEffectFinalizationMember(
                    ordinal=0,
                    output_data={"ordinal": 0},
                    duration_ms=0.0,
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.DEFAULT_FLOW,
                    sink_name="duplicate-observable",
                ),
            ),
        )
        target = DuplicateObservableTarget()
        calls = 0

        def fail_once(observed: SinkEffectExecutionSeam) -> None:
            nonlocal calls
            if observed is seam and calls == 0:
                calls += 1
                raise SinkEffectInjectedFault(seam)

        first = SinkEffectCoordinator(
            factory=factory,
            worker_id="worker-a",
            lease_ttl=timedelta(seconds=30),
            fault_hook=fail_once,
        )
        with pytest.raises(SinkEffectInjectedFault):
            first.execute(request, DuplicateObservableSink(target))

        recovered = SinkEffectCoordinator(
            factory=make_factory(db),
            worker_id="worker-a",
            lease_ttl=timedelta(seconds=30),
        ).execute(request, DuplicateObservableSink(target))

        assert target.publication_count == 1
        assert recovered.effect.effect_id == target.effect_id
        assert recovered.artifact.content_hash == target.descriptor.content_hash  # type: ignore[union-attr]
    finally:
        db.close()


def test_unknown_reconciliation_never_commits() -> None:
    """A divergent external target is a hard stop, never permission to publish."""
    db = make_landscape_db()
    try:
        factory: RecorderFactory = make_factory(db)
        run_id, sink_id, members = _pipeline_members(factory, 1)
        identity = compute_pipeline_effect_identity(
            run_id=run_id,
            sink_node_id=sink_id,
            role=_pipeline_request(run_id, sink_id, members).role,
            sink_config={"name": "duplicate-observable"},
            target_config={"path": "duplicate-observable.jsonl"},
            members=members,
        )
        request = SinkEffectExecutionRequest(
            reservation=_pipeline_request(run_id, sink_id, identity.members),
            effect_input=SinkEffectPipelineMembersInput(identity.members, identity.members),
            finalization_members=(
                SinkEffectFinalizationMember(
                    ordinal=0,
                    output_data={"ordinal": 0},
                    duration_ms=0.0,
                    outcome=TerminalOutcome.SUCCESS,
                    path=TerminalPath.DEFAULT_FLOW,
                    sink_name="duplicate-observable",
                ),
            ),
        )
        target = DuplicateObservableTarget(publication_count=1, effect_id="f" * 64)
        with pytest.raises(Exception, match=r"UNKNOWN|unknown|divergent"):
            SinkEffectCoordinator(factory=factory, worker_id="worker-a").execute(request, DuplicateObservableSink(target))
        assert target.publication_count == 1
    finally:
        db.close()
