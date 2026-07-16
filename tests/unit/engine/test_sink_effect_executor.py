"""Caller-level contract for durable sink-effect execution."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import update

from elspeth.contracts import NodeStateStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.hashing import stable_hash
from elspeth.contracts.results import ArtifactDescriptor
from elspeth.contracts.sink_effects import (
    SINK_EFFECT_PROTOCOL_VERSION,
    RestrictedSinkEffectContext,
    SinkEffectAttemptAction,
    SinkEffectCommitResult,
    SinkEffectDescriptorMode,
    SinkEffectInspection,
    SinkEffectInspectionMode,
    SinkEffectInspectionRequest,
    SinkEffectPipelineMembersInput,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileResult,
)
from elspeth.core.landscape.execution.sink_effect_attempt_results import encode_sink_effect_returned_result
from elspeth.core.landscape.execution.sink_effect_finalization import SinkEffectFinalizationMember
from elspeth.core.landscape.execution.sink_effect_identity import compute_pipeline_effect_identity
from elspeth.core.landscape.execution.sink_effect_lifecycle import SinkEffectAttemptRequest, SinkEffectAttemptResult
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import node_states_table, sink_effect_attempts_table
from elspeth.engine.executors.sink_effects import (
    SinkEffectCoordinator,
    SinkEffectExecutionRequest,
    SinkEffectExecutionSeam,
    SinkEffectInjectedFault,
    SinkEffectPredecessorPending,
)
from tests.fixtures.landscape import make_factory, make_landscape_db
from tests.fixtures.sink_effects import DuplicateObservableSink, DuplicateObservableTarget
from tests.fixtures.stores import MockPayloadStore
from tests.unit.core.landscape.test_sink_effect_reservation import _pipeline_members, _pipeline_request

_ROOT = Path(__file__).parents[3]


@dataclass(slots=True)
class _CumulativeTarget:
    effect_id: str | None = None
    descriptor: ArtifactDescriptor | None = None
    published_rows: list[list[dict[str, object]]] = field(default_factory=list)


class _CumulativeObservableSink:
    def __init__(self, target: _CumulativeTarget) -> None:
        self._target = target
        self._rows_by_effect: dict[str, list[dict[str, object]]] = {}
        self._accepted_by_effect: dict[str, tuple[int, ...]] = {}
        self.inspect_calls = 0
        self.prepare_calls = 0
        self.reconcile_calls = 0
        self.commit_calls = 0

    def inspect_effect(
        self,
        request: SinkEffectInspectionRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectInspection:
        del request, ctx
        self.inspect_calls += 1
        return SinkEffectInspection(
            mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
            reference="no-inspection-required:v1",
            evidence={},
        )

    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        del ctx
        self.prepare_calls += 1
        assert isinstance(request.effect_input, SinkEffectPipelineMembersInput)
        rows = [deep_thaw(member.row) for member in request.effect_input.target_snapshot_members]
        assert all(isinstance(row, dict) for row in rows)
        payload_hash = stable_hash(rows)
        descriptor = ArtifactDescriptor.for_file(
            path="file:///tmp/cumulative-observable.jsonl",
            content_hash=payload_hash,
            size_bytes=len(rows),
        )
        self._rows_by_effect[request.effect_id] = rows
        self._accepted_by_effect[request.effect_id] = tuple(member.ordinal for member in request.effect_input.members)
        return SinkEffectPlan(
            effect_id=request.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=request.input_kind,
            descriptor_mode=SinkEffectDescriptorMode.PRECOMPUTED,
            inspection_mode=request.inspection.mode,
            target=descriptor.path_or_uri,
            plan_hash=stable_hash(
                {
                    "descriptor": descriptor.content_hash,
                    "effect_id": request.effect_id,
                    "schema": "cumulative-observable-plan-v1",
                }
            ),
            payload_hash=payload_hash,
            expected_descriptor=descriptor,
            safe_evidence={"inspection_reference": request.inspection.reference},
        )

    def commit_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectCommitResult:
        del ctx
        self.commit_calls += 1
        assert plan.expected_descriptor is not None
        self._target.effect_id = plan.effect_id
        self._target.descriptor = plan.expected_descriptor
        self._target.published_rows.append(self._rows_by_effect[plan.effect_id])
        return SinkEffectCommitResult(
            descriptor=plan.expected_descriptor,
            evidence={"effect_id": plan.effect_id},
            accepted_ordinals=self._accepted_by_effect[plan.effect_id],
            diverted_ordinals=(),
        )

    def reconcile_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectReconcileResult:
        del ctx
        self.reconcile_calls += 1
        if self._target.effect_id == plan.effect_id and self._target.descriptor == plan.expected_descriptor:
            assert self._target.descriptor is not None
            return SinkEffectReconcileResult.applied(self._target.descriptor, evidence={"effect_id": plan.effect_id})
        return SinkEffectReconcileResult.not_applied(evidence={"target": "not_applied"})


class _PrepareFailsOnceSink(_CumulativeObservableSink):
    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        if self.prepare_calls == 0:
            self.prepare_calls += 1
            raise RuntimeError("injected prepare failure after durable inspection")
        return super().prepare_effect(request, ctx)


class _ResultDerivedReconciledSink(_CumulativeObservableSink):
    def prepare_effect(
        self,
        request: SinkEffectPrepareRequest,
        ctx: RestrictedSinkEffectContext,
    ) -> SinkEffectPlan:
        del ctx
        self.prepare_calls += 1
        assert isinstance(request.effect_input, SinkEffectPipelineMembersInput)
        payload_hash = stable_hash([deep_thaw(member.row) for member in request.effect_input.members])
        descriptor = ArtifactDescriptor(
            artifact_type="database",
            path_or_uri="database-result:sha256:" + "a" * 64,
            content_hash=payload_hash,
            size_bytes=1,
            metadata={"table": "output", "row_count": 1},
        )
        self._target.effect_id = request.effect_id
        self._target.descriptor = descriptor
        return SinkEffectPlan(
            effect_id=request.effect_id,
            protocol_version=SINK_EFFECT_PROTOCOL_VERSION,
            input_kind=request.input_kind,
            descriptor_mode=SinkEffectDescriptorMode.RESULT_DERIVED,
            inspection_mode=request.inspection.mode,
            target=descriptor.path_or_uri,
            plan_hash=stable_hash({"effect_id": request.effect_id, "payload_hash": payload_hash}),
            payload_hash=payload_hash,
            expected_descriptor=None,
            safe_evidence={"inspection_reference": request.inspection.reference},
        )

    def reconcile_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectReconcileResult:
        del ctx
        self.reconcile_calls += 1
        assert self._target.descriptor is not None
        descriptor = self._target.descriptor
        return SinkEffectReconcileResult.applied(
            descriptor,
            evidence={
                "accepted_ordinals": [0],
                "descriptor": {
                    "artifact_type": descriptor.artifact_type,
                    "content_hash": descriptor.content_hash,
                    "metadata": deep_thaw(descriptor.metadata),
                    "path_or_uri": descriptor.path_or_uri,
                    "size_bytes": descriptor.size_bytes,
                },
                "diverted_ordinals": [1],
            },
            accepted_ordinals=(0,),
            diverted_ordinals=(1,),
        )

    def commit_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectCommitResult:
        del plan, ctx
        raise AssertionError("exact result-derived reconciliation must not commit again")


def _execution_request(run_id: str, sink_id: str, members: tuple[object, ...]) -> SinkEffectExecutionRequest:
    typed_members = tuple(members)
    reservation = _pipeline_request(run_id, sink_id, typed_members, replacing_target=True)  # type: ignore[arg-type]
    identity = compute_pipeline_effect_identity(
        run_id=run_id,
        sink_node_id=sink_id,
        role=reservation.role,
        sink_config={"name": "sink"},
        target_config={"path": "out.jsonl"},
        members=reservation.members,
    )
    return SinkEffectExecutionRequest(
        reservation=reservation,
        effect_input=SinkEffectPipelineMembersInput(identity.members, identity.members),
        finalization_members=tuple(
            SinkEffectFinalizationMember(
                ordinal=member.ordinal,
                output_data={"row": dict(member.row)},
                duration_ms=0.0,
                outcome=TerminalOutcome.SUCCESS,
                path=TerminalPath.DEFAULT_FLOW,
                sink_name="cumulative-observable",
            )
            for member in identity.members
        ),
    )


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


def test_result_derived_reconciliation_finalizes_exact_marker_partition() -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run_id, sink_id, members = _pipeline_members(factory, 2)
        sink = _ResultDerivedReconciledSink(_CumulativeTarget())

        result = SinkEffectCoordinator(factory=factory, worker_id="worker-a").execute(
            _execution_request(run_id, sink_id, members),
            sink,
        )

        assert len(result.state_ids) == 1
        assert len(result.outcome_ids) == 1
        assert sink.commit_calls == 0
        assert sink.reconcile_calls == 1
    finally:
        db.close()


def test_replacing_successor_prepares_cumulative_predecessor_and_current_members() -> None:
    db = make_landscape_db()
    try:
        payload_store = MockPayloadStore()
        factory = make_factory(db, payload_store=payload_store)
        run_id, sink_id, members = _pipeline_members(factory, 2)
        target = _CumulativeTarget()
        sink = _CumulativeObservableSink(target)

        SinkEffectCoordinator(factory=factory, worker_id="worker-a").execute(_execution_request(run_id, sink_id, members[:1]), sink)
        recovered_factory = make_factory(db, payload_store=payload_store)
        successor = SinkEffectCoordinator(factory=recovered_factory, worker_id="worker-b").execute(
            _execution_request(run_id, sink_id, members[1:]), sink
        )

        assert successor.effect.stream_sequence == 1
        assert target.published_rows == [[{"ordinal": 0}], [{"ordinal": 0}, {"ordinal": 1}]]
    finally:
        db.close()


def test_predecessor_snapshot_excludes_diverted_members() -> None:
    """Cumulative successors must not replay members the predecessor diverted
    away from the target (elspeth-0278416cc5)."""
    db = make_landscape_db()
    try:
        payload_store = MockPayloadStore()
        factory = make_factory(db, payload_store=payload_store)
        run_id, sink_id, members = _pipeline_members(factory, 3)
        # Predecessor accepts ordinal 0 and diverts ordinal 1.
        first_sink = _ResultDerivedReconciledSink(_CumulativeTarget())
        SinkEffectCoordinator(factory=factory, worker_id="worker-a").execute(
            _execution_request(run_id, sink_id, members[:2]),
            first_sink,
        )

        target = _CumulativeTarget()
        successor_sink = _CumulativeObservableSink(target)
        successor_factory = make_factory(db, payload_store=payload_store)
        SinkEffectCoordinator(factory=successor_factory, worker_id="worker-b").execute(
            _execution_request(run_id, sink_id, members[2:]),
            successor_sink,
        )

        # The diverted row {"ordinal": 1} never reached the target, so the
        # cumulative successor must publish only the accepted predecessor
        # member plus its own current member.
        assert target.published_rows == [[{"ordinal": 0}, {"ordinal": 2}]]
    finally:
        db.close()


def test_mixed_overlap_recovers_open_effect_and_executes_new_partition() -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run_id, sink_id, members = _pipeline_members(factory, 4)
        target = _CumulativeTarget()
        sink = _CumulativeObservableSink(target)
        coordinator = SinkEffectCoordinator(factory=factory, worker_id="worker-a")

        first = coordinator.execute(_execution_request(run_id, sink_id, members[:1]), sink)
        opened = factory.execution.sink_effects.reserve(_pipeline_request(run_id, sink_id, members[1:3], replacing_target=True)).new_effect
        assert opened is not None and opened.predecessor_effect_id == first.effect.effect_id

        result = coordinator.execute(_execution_request(run_id, sink_id, members), sink)

        assert result.effect.stream_sequence == 2
        assert target.published_rows == [
            [{"ordinal": 0}],
            [{"ordinal": 0}, {"ordinal": 1}, {"ordinal": 2}],
            [{"ordinal": 0}, {"ordinal": 1}, {"ordinal": 2}, {"ordinal": 3}],
        ]
    finally:
        db.close()


def test_second_preparer_refuses_while_preparation_claim_is_live() -> None:
    """A rival worker must never run side-effecting preparation while another
    worker's durable preparation claim is live (elspeth-3f87c0c055)."""
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run_id, sink_id, members = _pipeline_members(factory, 1)
        target = _CumulativeTarget()
        request = _execution_request(run_id, sink_id, members)
        rival_factory = make_factory(db)
        rival_sink = _CumulativeObservableSink(target)

        class _RacingSink(_CumulativeObservableSink):
            def prepare_effect(
                self,
                inner_request: SinkEffectPrepareRequest,
                ctx: RestrictedSinkEffectContext,
            ) -> SinkEffectPlan:
                if self.prepare_calls == 0:
                    # Simulate a concurrent worker arriving mid-preparation:
                    # it must refuse before invoking any adapter method.
                    with pytest.raises(SinkEffectPredecessorPending, match="preparation"):
                        SinkEffectCoordinator(factory=rival_factory, worker_id="worker-b").execute(request, rival_sink)
                return super().prepare_effect(inner_request, ctx)

        sink = _RacingSink(target)
        result = SinkEffectCoordinator(factory=factory, worker_id="worker-a").execute(request, sink)

        assert result.effect.state.value == "finalized"
        assert sink.prepare_calls == 1
        # The rival never mutated staging: no inspect, prepare, or commit calls.
        assert (rival_sink.inspect_calls, rival_sink.prepare_calls, rival_sink.commit_calls) == (0, 0, 0)
        assert target.published_rows == [[{"ordinal": 0}]]
    finally:
        db.close()


def test_mixed_overlap_waits_for_live_open_partition_before_executing_new() -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run_id, sink_id, members = _pipeline_members(factory, 2)
        sink = _CumulativeObservableSink(_CumulativeTarget())

        def fail_before_effect(seam: SinkEffectExecutionSeam) -> None:
            if seam is SinkEffectExecutionSeam.BEFORE_EFFECT:
                raise SinkEffectInjectedFault(seam)

        with pytest.raises(SinkEffectInjectedFault):
            SinkEffectCoordinator(
                factory=factory,
                worker_id="worker-a",
                fault_hook=fail_before_effect,
            ).execute(_execution_request(run_id, sink_id, members[:1]), sink)
        calls_before_wait = (sink.inspect_calls, sink.prepare_calls, sink.reconcile_calls, sink.commit_calls)

        with pytest.raises(SinkEffectPredecessorPending, match="live lease"):
            SinkEffectCoordinator(factory=factory, worker_id="worker-b").execute(_execution_request(run_id, sink_id, members), sink)

        assert (sink.inspect_calls, sink.prepare_calls, sink.reconcile_calls, sink.commit_calls) == calls_before_wait
        reserved_new = factory.execution.sink_effects.reserve(
            _pipeline_request(run_id, sink_id, members, replacing_target=True)
        ).open_effect_ids
        assert len(reserved_new) == 2
    finally:
        db.close()


@pytest.mark.parametrize("terminal_status", (NodeStateStatus.COMPLETED, NodeStateStatus.FAILED))
def test_non_open_latest_state_refuses_before_sink_io(terminal_status: NodeStateStatus) -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run_id, sink_id, members = _pipeline_members(factory, 1)
        with db.engine.begin() as conn:
            conn.execute(
                update(node_states_table)
                .where(
                    node_states_table.c.run_id == run_id,
                    node_states_table.c.node_id == sink_id,
                    node_states_table.c.token_id == members[0].token_id,
                )
                .values(status=terminal_status.value)
            )
        sink = _CumulativeObservableSink(_CumulativeTarget())

        with pytest.raises(ValueError, match="latest sink-node state must be open"):
            SinkEffectCoordinator(factory=factory, worker_id="worker-a").execute(
                _execution_request(run_id, sink_id, members),
                sink,
            )

        assert (sink.inspect_calls, sink.prepare_calls, sink.reconcile_calls, sink.commit_calls) == (0, 0, 0, 0)
    finally:
        db.close()


def test_retry_reuses_durable_returned_inspection_without_second_provider_call() -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run_id, sink_id, members = _pipeline_members(factory, 1)
        target = _CumulativeTarget()
        sink = _PrepareFailsOnceSink(target)
        request = _execution_request(run_id, sink_id, members)

        with pytest.raises(RuntimeError, match="prepare failure"):
            SinkEffectCoordinator(factory=factory, worker_id="worker-a").execute(request, sink)
        result = SinkEffectCoordinator(factory=factory, worker_id="worker-a").execute(request, sink)

        assert result.effect.state.value == "finalized"
        assert sink.inspect_calls == 1
        assert sink.prepare_calls == 2
    finally:
        db.close()


def test_same_generation_retry_closes_abandoned_commit_intent_before_new_call() -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run_id, sink_id, members = _pipeline_members(factory, 1)
        target = _CumulativeTarget()
        sink = _CumulativeObservableSink(target)
        request = _execution_request(run_id, sink_id, members)

        def fail_before_effect(seam: SinkEffectExecutionSeam) -> None:
            if seam is SinkEffectExecutionSeam.BEFORE_EFFECT:
                raise SinkEffectInjectedFault(seam)

        with pytest.raises(SinkEffectInjectedFault):
            SinkEffectCoordinator(
                factory=factory,
                worker_id="worker-a",
                fault_hook=fail_before_effect,
            ).execute(request, sink)
        SinkEffectCoordinator(factory=factory, worker_id="worker-a").execute(request, sink)

        with db.read_only_connection() as conn:
            commits = conn.execute(
                sink_effect_attempts_table.select()
                .where(sink_effect_attempts_table.c.action == "commit")
                .order_by(sink_effect_attempts_table.c.started_at, sink_effect_attempts_table.c.attempt_id)
            ).fetchall()
        # Generation 1 is consumed by the preparation claim; the execution
        # lease (and thus both commit attempts) run at generation 2.
        assert [(row.generation, row.state) for row in commits] == [(2, "response_lost"), (2, "returned")]
        assert target.published_rows == [[{"ordinal": 0}]]
    finally:
        db.close()


def test_takeover_closes_stale_abandoned_intent_before_new_generation_call() -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run_id, sink_id, members = _pipeline_members(factory, 1)
        target = _CumulativeTarget()
        sink = _CumulativeObservableSink(target)
        request = _execution_request(run_id, sink_id, members)

        def fail_before_effect(seam: SinkEffectExecutionSeam) -> None:
            if seam is SinkEffectExecutionSeam.BEFORE_EFFECT:
                raise SinkEffectInjectedFault(seam)

        with pytest.raises(SinkEffectInjectedFault):
            SinkEffectCoordinator(
                factory=factory,
                worker_id="worker-a",
                lease_ttl=timedelta(microseconds=1),
                fault_hook=fail_before_effect,
            ).execute(request, sink)
        SinkEffectCoordinator(factory=factory, worker_id="worker-b").execute(request, sink)

        with db.read_only_connection() as conn:
            commits = conn.execute(
                sink_effect_attempts_table.select()
                .where(sink_effect_attempts_table.c.action == "commit")
                .order_by(sink_effect_attempts_table.c.generation)
            ).fetchall()
        # Generation 1 is the preparation claim, 2 the abandoned execution
        # lease, 3 the takeover under which the retry returns.
        assert [(row.generation, row.state) for row in commits] == [(2, "response_lost"), (3, "returned")]
        assert target.published_rows == [[{"ordinal": 0}]]
    finally:
        db.close()


@pytest.mark.parametrize("takeover", (False, True))
def test_retry_consumes_returned_commit_without_another_reconcile(takeover: bool) -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run_id, sink_id, members = _pipeline_members(factory, 1)
        target = _CumulativeTarget()
        sink = _CumulativeObservableSink(target)
        request = _execution_request(run_id, sink_id, members)

        def fail_after_return(seam: SinkEffectExecutionSeam) -> None:
            if seam is SinkEffectExecutionSeam.AFTER_RETURN_BEFORE_FINALIZE:
                raise SinkEffectInjectedFault(seam)

        with pytest.raises(SinkEffectInjectedFault):
            SinkEffectCoordinator(
                factory=factory,
                worker_id="worker-a",
                lease_ttl=timedelta(microseconds=1) if takeover else timedelta(seconds=30),
                fault_hook=fail_after_return,
            ).execute(request, sink)
        SinkEffectCoordinator(
            factory=factory,
            worker_id="worker-b" if takeover else "worker-a",
        ).execute(request, sink)

        assert sink.commit_calls == 1
        assert sink.reconcile_calls == 1
        assert target.published_rows == [[{"ordinal": 0}]]
    finally:
        db.close()


@pytest.mark.parametrize("takeover", (False, True))
def test_retry_consumes_returned_reconcile_before_commit(takeover: bool) -> None:
    db = make_landscape_db()
    try:
        factory = make_factory(db)
        run_id, sink_id, members = _pipeline_members(factory, 1)
        target = _CumulativeTarget()
        sink = _CumulativeObservableSink(target)
        request = _execution_request(run_id, sink_id, members)
        reserved = factory.execution.sink_effects.reserve(request.reservation).new_effect
        assert reserved is not None
        inspection = SinkEffectInspection(
            mode=SinkEffectInspectionMode.NO_INSPECTION_REQUIRED,
            reference="no-inspection-required:v1",
            evidence={},
        )
        plan = sink.prepare_effect(
            SinkEffectPrepareRequest(
                effect_id=reserved.effect_id,
                effect_input=request.effect_input,  # type: ignore[arg-type]
                inspection=inspection,
            ),
            RestrictedSinkEffectContext(
                run_id=run_id,
                run_started_at=factory.run_lifecycle.get_run(run_id).started_at,  # type: ignore[union-attr]
                operation_id=next(
                    operation.operation_id
                    for operation in factory.execution.get_operations_for_run(run_id)
                    if operation.sink_effect_id == reserved.effect_id
                ),
                sink_node_id=sink_id,
            ),
        )
        claim = factory.execution.sink_effects.claim_preparation(
            reserved.effect_id,
            owner="worker-a",
            ttl=timedelta(seconds=30),
        )
        factory.execution.sink_effects.complete_plan(reserved.effect_id, plan, claim=claim)
        lease = factory.execution.sink_effects.acquire_lease(
            reserved.effect_id,
            owner="worker-a",
            ttl=timedelta(microseconds=1) if takeover else timedelta(seconds=30),
        )
        reconciliation = SinkEffectReconcileResult.not_applied(evidence={"target": "not_applied"})
        attempt = factory.execution.sink_effects.begin_attempt(
            SinkEffectAttemptRequest(
                effect_id=reserved.effect_id,
                member_ordinal=None,
                generation=lease.generation,
                action=SinkEffectAttemptAction.RECONCILE,
                request_hash=SinkEffectCoordinator._reconcile_request_hash(plan),
            )
        )
        factory.execution.sink_effects.record_attempt_result(
            SinkEffectAttemptResult(
                attempt_id=attempt.attempt_id,
                evidence=encode_sink_effect_returned_result(reconciliation),
                latency_ms=0.0,
            )
        )

        SinkEffectCoordinator(
            factory=factory,
            worker_id="worker-b" if takeover else "worker-a",
        ).execute(request, sink)

        assert sink.reconcile_calls == 0
        assert sink.commit_calls == 1
        assert target.published_rows == [[{"ordinal": 0}]]
    finally:
        db.close()


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
