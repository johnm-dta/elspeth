"""Production pipeline-boundary recovery for durable sink effects."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from elspeth.contracts import NodeStateStatus, NodeType, PendingOutcome, RoutingMode, TerminalOutcome, TerminalPath, TokenInfo
from elspeth.contracts.audit import SinkEffect, SinkEffectMemberRecord, TokenRef
from elspeth.contracts.diversion import RowDiversion
from elspeth.contracts.errors import (
    AuditIntegrityError,
    FrameworkBugError,
    OrchestrationInvariantError,
    SinkTransactionalInvariantError,
)
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.contracts.sink_effects import (
    RestrictedSinkEffectContext,
    SinkEffectCommitResult,
    SinkEffectPlan,
    SinkEffectPrepareRequest,
    SinkEffectReconcileResult,
    SinkEffectRole,
)
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.sink_effect_reservation import SinkEffectReservation
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.core.landscape.schema import node_states_table, routing_events_table, sink_effect_members_table
from elspeth.engine.executors.sink import SinkExecutor
from elspeth.engine.executors.sink_effects import SinkEffectExecutionSeam, SinkEffectInjectedFault
from elspeth.engine.spans import SpanFactory
from tests.fixtures.base_classes import create_observed_contract
from tests.fixtures.landscape import make_factory, register_test_node
from tests.fixtures.sink_effects import DuplicateObservableSink, DuplicateObservableTarget, PartitioningObservableSink


@pytest.mark.parametrize(
    "seam",
    (
        SinkEffectExecutionSeam.BEFORE_EFFECT,
        SinkEffectExecutionSeam.AFTER_EFFECT_BEFORE_RETURN,
        SinkEffectExecutionSeam.AFTER_RETURN_BEFORE_FINALIZE,
    ),
)
def test_fresh_pipeline_executor_reuses_interrupted_open_state_and_publishes_once(
    tmp_path: Path,
    seam: SinkEffectExecutionSeam,
) -> None:
    db = LandscapeDB(f"sqlite:///{tmp_path / 'landscape.db'}")
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        sink_id = register_test_node(
            factory.data_flow,
            run.run_id,
            "duplicate-sink",
            node_type=NodeType.SINK,
            plugin_name="duplicate-observable",
        )
        row_data = {"ordinal": 0}
        row = factory.data_flow.create_row(
            run_id=run.run_id,
            source_node_id=source_id,
            row_index=0,
            data=row_data,
            source_row_index=0,
            ingest_sequence=0,
        )
        durable_token = factory.data_flow.create_token(row.row_id)
        token = TokenInfo(
            row_id=row.row_id,
            token_id=durable_token.token_id,
            row_data=PipelineRow(row_data, create_observed_contract(row_data)),
        )
        target = DuplicateObservableTarget()
        calls = 0

        def fail_once(observed: SinkEffectExecutionSeam) -> None:
            nonlocal calls
            if observed is seam and calls == 0:
                calls += 1
                raise SinkEffectInjectedFault(seam)

        first_sink = DuplicateObservableSink(target)
        first_sink.node_id = sink_id
        first = SinkExecutor(
            factory.execution,
            factory.data_flow,
            SpanFactory(),
            run.run_id,
            factory=factory,
            worker_id="worker-a",
            sink_effect_fault_hook=fail_once,
        )
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=sink_id)
        with pytest.raises(SinkEffectInjectedFault):
            first.write(
                first_sink,  # type: ignore[arg-type]
                [token],
                ctx,
                1,
                sink_name="output",
                pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
                effect_mode="write",
            )

        recovered_factory = make_factory(db)
        recovered_sink = DuplicateObservableSink(target)
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
            [token],
            ctx,
            1,
            sink_name="output",
            pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
            effect_mode="write",
        )

        assert target.publication_count == 1
        assert artifact is not None and artifact.sink_effect_id == target.effect_id
        assert counts.total == 0
        assert recovered_factory.data_flow.get_token_outcome(token.token_id) is not None
    finally:
        db.close()


def _effect_tokens(factory, *, run_id: str, source_id: str, rows: list[dict[str, object]]) -> list[TokenInfo]:  # type: ignore[no-untyped-def]
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
        durable = factory.data_flow.create_token(row.row_id)
        tokens.append(
            TokenInfo(
                row_id=row.row_id,
                token_id=durable.token_id,
                row_data=PipelineRow(row_data, create_observed_contract(row_data)),
            )
        )
    return tokens


def test_primary_finalizes_once_while_diverted_token_waits_for_linked_failsink(tmp_path: Path) -> None:
    db = LandscapeDB(f"sqlite:///{tmp_path / 'linked.db'}")
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        primary_id = register_test_node(
            factory.data_flow,
            run.run_id,
            "primary",
            node_type=NodeType.SINK,
            plugin_name="partitioning-primary",
        )
        failsink_id = register_test_node(
            factory.data_flow,
            run.run_id,
            "failsink",
            node_type=NodeType.SINK,
            plugin_name="partitioning-failsink",
        )
        edge = factory.data_flow.register_edge(
            run.run_id,
            primary_id,
            failsink_id,
            "__failsink__",
            RoutingMode.DIVERT,
        )
        accepted, diverted = _effect_tokens(
            factory,
            run_id=run.run_id,
            source_id=source_id,
            rows=[{"value": 1}, {"value": 2, "divert": True}],
        )
        primary_target = DuplicateObservableTarget()
        failsink_target = DuplicateObservableTarget()
        primary = PartitioningObservableSink(primary_target, name="primary")
        primary.node_id = primary_id
        failsink = PartitioningObservableSink(
            failsink_target,
            name="failsink",
            fail_prepare_once=True,
            divert_rows=False,
        )
        failsink.node_id = failsink_id
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=primary_id)
        executor = SinkExecutor(
            factory.execution,
            factory.data_flow,
            SpanFactory(),
            run.run_id,
            factory=factory,
            worker_id="worker-a",
        )

        with pytest.raises(RuntimeError, match="between primary and failsink"):
            executor.write(
                primary,  # type: ignore[arg-type]
                [accepted, diverted],
                ctx,
                1,
                sink_name="output",
                pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
                effect_mode="write",
                failsink=failsink,  # type: ignore[arg-type]
                failsink_name="failsink",
                failsink_effect_mode="write",
                failsink_edge_id=edge.edge_id,
            )

        effects = factory.execution.sink_effects.get_effects_for_run(run.run_id)
        primary_effect = next(effect for effect in effects if effect.role.value == "primary")
        failsink_effect = next(effect for effect in effects if effect.role.value == "failsink")
        assert primary_effect.state.value == "finalized"
        assert failsink_effect.state.value == "reserved"
        assert failsink_effect.primary_effect_id == primary_effect.effect_id
        assert factory.data_flow.get_token_outcome(diverted.token_id) is None

        recovered_factory = make_factory(db)
        recovered_primary = PartitioningObservableSink(primary_target, name="primary")
        recovered_primary.node_id = primary_id
        recovered_failsink = PartitioningObservableSink(failsink_target, name="failsink", divert_rows=False)
        recovered_failsink.node_id = failsink_id
        artifact, counts = SinkExecutor(
            recovered_factory.execution,
            recovered_factory.data_flow,
            SpanFactory(),
            run.run_id,
            factory=recovered_factory,
            worker_id="worker-a",
        ).write(
            recovered_primary,  # type: ignore[arg-type]
            [accepted, diverted],
            ctx,
            1,
            sink_name="output",
            pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
            effect_mode="write",
            failsink=recovered_failsink,  # type: ignore[arg-type]
            failsink_name="failsink",
            failsink_effect_mode="write",
            failsink_edge_id=edge.edge_id,
        )

        assert artifact is not None and artifact.sink_effect_id == primary_effect.effect_id
        assert primary_target.publication_count == 1
        assert failsink_target.publication_count == 1
        assert counts.failsink_mode == 1
        recovered_effects = recovered_factory.execution.sink_effects.get_effects_for_run(run.run_id)
        recovered_failsink_effect = next(effect for effect in recovered_effects if effect.role.value == "failsink")
        outcome = recovered_factory.data_flow.get_token_outcome(diverted.token_id)
        assert recovered_failsink_effect.artifact_id is not None
        assert outcome is not None
        assert outcome.sink_name == "failsink"
        assert outcome.path is TerminalPath.SINK_FALLBACK_TO_FAILSINK
        primary_state = next(
            state for state in recovered_factory.query.get_node_states_for_token(diverted.token_id) if state.node_id == primary_id
        )
        routing_events = recovered_factory.query.get_routing_events(primary_state.state_id)
        assert len(routing_events) == 1
        assert routing_events[0].edge_id == edge.edge_id
        assert routing_events[0].mode is RoutingMode.DIVERT
    finally:
        db.close()


def test_recovered_two_primary_batch_preserves_per_member_failsink_provenance(tmp_path: Path) -> None:
    db = LandscapeDB(f"sqlite:///{tmp_path / 'two-primary-failsink.db'}")
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        primary_id = register_test_node(
            factory.data_flow,
            run.run_id,
            "primary",
            node_type=NodeType.SINK,
            plugin_name="partitioning-primary",
        )
        failsink_id = register_test_node(
            factory.data_flow,
            run.run_id,
            "failsink",
            node_type=NodeType.SINK,
            plugin_name="partitioning-failsink",
        )
        edge = factory.data_flow.register_edge(
            run.run_id,
            primary_id,
            failsink_id,
            "__failsink__",
            RoutingMode.DIVERT,
        )
        first_token, second_token = _effect_tokens(
            factory,
            run_id=run.run_id,
            source_id=source_id,
            rows=[{"value": 1, "divert": True}, {"value": 2, "divert": True}],
        )
        primary_target = DuplicateObservableTarget()
        failsink_target = DuplicateObservableTarget()
        primary = PartitioningObservableSink(primary_target, name="primary")
        primary.node_id = primary_id
        failsink = PartitioningObservableSink(failsink_target, name="failsink", divert_rows=False)
        failsink.node_id = failsink_id
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=primary_id)

        def stop_after_first_primary(seam: SinkEffectExecutionSeam) -> None:
            if seam is SinkEffectExecutionSeam.AFTER_FINALIZE_BEFORE_RESPONSE:
                raise SinkEffectInjectedFault(seam)

        with pytest.raises(SinkEffectInjectedFault):
            SinkExecutor(
                factory.execution,
                factory.data_flow,
                SpanFactory(),
                run.run_id,
                factory=factory,
                worker_id="worker-a",
                sink_effect_fault_hook=stop_after_first_primary,
            ).write(
                primary,  # type: ignore[arg-type]
                [first_token],
                ctx,
                1,
                sink_name="output",
                pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
                effect_mode="write",
                failsink=failsink,  # type: ignore[arg-type]
                failsink_name="failsink",
                failsink_effect_mode="write",
                failsink_edge_id=edge.edge_id,
            )

        recovered_factory = make_factory(db)
        recovered_primary = PartitioningObservableSink(primary_target, name="primary")
        recovered_primary.node_id = primary_id
        recovered_failsink = PartitioningObservableSink(failsink_target, name="failsink", divert_rows=False)
        recovered_failsink.node_id = failsink_id
        _artifact, counts = SinkExecutor(
            recovered_factory.execution,
            recovered_factory.data_flow,
            SpanFactory(),
            run.run_id,
            factory=recovered_factory,
            worker_id="worker-b",
        ).write(
            recovered_primary,  # type: ignore[arg-type]
            [first_token, second_token],
            ctx,
            1,
            sink_name="output",
            pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
            effect_mode="write",
            failsink=recovered_failsink,  # type: ignore[arg-type]
            failsink_name="failsink",
            failsink_effect_mode="write",
            failsink_edge_id=edge.edge_id,
        )

        effects = recovered_factory.execution.sink_effects.get_effects_for_run(run.run_id)
        primary_effects = tuple(effect for effect in effects if effect.role is SinkEffectRole.PRIMARY)
        failsink_effects = tuple(effect for effect in effects if effect.role is SinkEffectRole.FAILSINK)
        assert len(primary_effects) == 2
        assert len(failsink_effects) == 1
        primary_by_token = {
            member.token_id: effect.effect_id
            for effect in primary_effects
            for member in recovered_factory.execution.sink_effects.get_members(effect.effect_id)
        }
        failsink_effect = failsink_effects[0]
        failsink_members = recovered_factory.execution.sink_effects.get_members(failsink_effect.effect_id)
        observed_primary_by_token = {member.token_id: member.primary_effect_id for member in failsink_members}

        assert counts.failsink_mode == 2
        assert failsink_effect.primary_effect_id is None
        assert observed_primary_by_token == primary_by_token
    finally:
        db.close()


class _RejectingFailsink(PartitioningObservableSink):
    """Failsink whose transactional backstop rejects every enriched row."""

    declared_required_fields = frozenset({"must_exist"})


def test_failsink_validation_rejection_terminalizes_states_and_outcomes(tmp_path: Path) -> None:
    """When the linked failsink rejects the enriched rows at the validation
    boundary, the diverted token must receive a failure outcome and every
    opened primary/quarantine node state must be terminalized
    (elspeth-2a75af7f8f)."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'reject.db'}")
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        primary_id = register_test_node(factory.data_flow, run.run_id, "primary", node_type=NodeType.SINK, plugin_name="partitioning")
        failsink_id = register_test_node(factory.data_flow, run.run_id, "failsink", node_type=NodeType.SINK, plugin_name="rejecting")
        edge = factory.data_flow.register_edge(run.run_id, primary_id, failsink_id, "__failsink__", RoutingMode.DIVERT)
        accepted, diverted = _effect_tokens(
            factory,
            run_id=run.run_id,
            source_id=source_id,
            rows=[{"value": 1}, {"value": 2, "divert": True}],
        )
        primary_target = DuplicateObservableTarget()
        failsink_target = DuplicateObservableTarget()
        primary = PartitioningObservableSink(primary_target, name="primary")
        primary.node_id = primary_id
        failsink = _RejectingFailsink(failsink_target, name="failsink", divert_rows=False)
        failsink.node_id = failsink_id
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=primary_id)

        with pytest.raises(SinkTransactionalInvariantError, match="must_exist"):
            SinkExecutor(
                factory.execution,
                factory.data_flow,
                SpanFactory(),
                run.run_id,
                factory=factory,
                worker_id="worker-a",
            ).write(
                primary,  # type: ignore[arg-type]
                [accepted, diverted],
                ctx,
                1,
                sink_name="output",
                pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
                effect_mode="write",
                failsink=failsink,  # type: ignore[arg-type]
                failsink_name="failsink",
                failsink_effect_mode="write",
                failsink_edge_id=edge.edge_id,
            )

        assert failsink_target.publication_count == 0
        # The diverted token must not be left without a terminal outcome.
        outcome = factory.data_flow.get_token_outcome(diverted.token_id)
        assert outcome is not None
        assert outcome.outcome is TerminalOutcome.FAILURE
        assert outcome.path is TerminalPath.UNROUTED
        # Every opened node state for the diverted token (primary anchor and
        # failsink quarantine state) must be terminalized.
        diverted_states = factory.query.get_node_states_for_token(diverted.token_id)
        assert {state.node_id for state in diverted_states} >= {primary_id, failsink_id}
        assert all(state.status is not NodeStateStatus.OPEN for state in diverted_states)
        accepted_outcome = factory.data_flow.get_token_outcome(accepted.token_id)
        assert accepted_outcome is not None
        assert accepted_outcome.outcome is TerminalOutcome.SUCCESS
    finally:
        db.close()


class _StreamingObservableSink(PartitioningObservableSink):
    """Partitioning double whose target accepts successive stream effects."""

    def reconcile_effect(self, plan: object, ctx: object) -> SinkEffectReconcileResult:
        del ctx
        if self.external_target.effect_id == plan.effect_id and self.external_target.descriptor == plan.expected_descriptor:  # type: ignore[attr-defined]
            assert self.external_target.descriptor is not None
            return SinkEffectReconcileResult.applied(
                self.external_target.descriptor,
                evidence={"effect_id": plan.effect_id},  # type: ignore[attr-defined]
            )
        return SinkEffectReconcileResult.not_applied(evidence={"target": "not_applied"})


def test_retry_with_mixed_interrupted_and_fresh_members_progresses(tmp_path: Path) -> None:
    """A retry batch mixing one interrupted durable member with one fresh
    member must partition and publish instead of wedging on the durable
    witness-set guard (elspeth-d1a1399381)."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'mixed.db'}")
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        sink_id = register_test_node(factory.data_flow, run.run_id, "primary", node_type=NodeType.SINK, plugin_name="streaming")
        tokens = _effect_tokens(
            factory,
            run_id=run.run_id,
            source_id=source_id,
            rows=[{"value": 1}, {"value": 2}],
        )
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=sink_id)
        pending = PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)
        target = DuplicateObservableTarget()
        calls = 0

        def fail_once(observed: SinkEffectExecutionSeam) -> None:
            nonlocal calls
            if observed is SinkEffectExecutionSeam.BEFORE_EFFECT and calls == 0:
                calls += 1
                raise SinkEffectInjectedFault(observed)

        first_sink = _StreamingObservableSink(target, name="primary")
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
                tokens[:1],
                ctx,
                1,
                sink_name="output",
                pending_outcome=pending,
                effect_mode="write",
            )

        recovered_factory = make_factory(db)
        recovered_sink = _StreamingObservableSink(target, name="primary")
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
            pending_outcome=pending,
            effect_mode="write",
        )

        assert artifact is not None
        assert counts.total == 0
        # The interrupted member and the fresh member each publish exactly
        # once, through the recovered effect and a newly reserved successor.
        assert target.publication_count == 2
        effects = recovered_factory.execution.sink_effects.get_effects_for_run(run.run_id)
        assert [effect.state.value for effect in effects] == ["finalized", "finalized"]
        for token in tokens:
            outcome = recovered_factory.data_flow.get_token_outcome(token.token_id)
            assert outcome is not None
            assert outcome.outcome is TerminalOutcome.SUCCESS
        # The interrupted member reused its open state; the fresh member got
        # exactly one new state at the sink node.
        for token in tokens:
            states = [state for state in recovered_factory.query.get_node_states_for_token(token.token_id) if state.node_id == sink_id]
            assert len(states) == 1
    finally:
        db.close()


def test_redrive_after_crash_before_reservation_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after node states are opened but BEFORE reservation writes any
    durable sink_effect/member rows must not wedge the batch: the re-drive
    reuses the open states and publishes exactly once instead of raising
    AuditIntegrityError from the deleted durable witness-set guard
    (elspeth-0c38e49a74, fixed by 38c20dc52)."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'prereserve.db'}")
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        sink_id = register_test_node(factory.data_flow, run.run_id, "primary", node_type=NodeType.SINK, plugin_name="partitioning")
        tokens = _effect_tokens(
            factory,
            run_id=run.run_id,
            source_id=source_id,
            rows=[{"value": 1}, {"value": 2}],
        )
        token_ids = tuple(token.token_id for token in tokens)
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=sink_id)
        pending = PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)
        target = DuplicateObservableTarget()

        # Crash the FIRST traversal of the reservation entry point. This is
        # the narrowest durable-write seam the executor traverses:
        # SinkExecutor._write_primary_effect -> SinkEffectCoordinator.execute
        # -> SinkEffectRepository.reserve -> SinkEffectReservation.reserve.
        # Raising here fails the write AFTER node states are opened but
        # BEFORE any sink_effects/sink_effect_members rows exist.
        original_reserve = SinkEffectReservation.reserve
        reserve_calls = 0

        def crash_once(self: SinkEffectReservation, request: object) -> object:
            nonlocal reserve_calls
            reserve_calls += 1
            if reserve_calls == 1:
                raise RuntimeError("injected crash before sink-effect reservation")
            return original_reserve(self, request)  # type: ignore[arg-type]

        monkeypatch.setattr(SinkEffectReservation, "reserve", crash_once)

        first_sink = PartitioningObservableSink(target, name="primary")
        first_sink.node_id = sink_id
        with pytest.raises(RuntimeError, match="injected crash before sink-effect reservation"):
            SinkExecutor(
                factory.execution,
                factory.data_flow,
                SpanFactory(),
                run.run_id,
                factory=factory,
                worker_id="worker-a",
            ).write(
                first_sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=pending,
                effect_mode="write",
            )

        # Wedge-state precondition: every token's node state is still OPEN
        # while ZERO durable effect/member rows exist. This zero-durable
        # window is what distinguishes the pre-reservation wedge from the
        # mixed interrupted/fresh retry, whose interrupted member is durable.
        open_state_ids = factory.execution.get_open_node_state_ids(
            run.run_id,
            node_ids=(sink_id,),
            token_ids=token_ids,
        )
        assert set(open_state_ids) == set(token_ids)
        assert (
            factory.execution.sink_effects.get_members_for_tokens(
                run_id=run.run_id,
                sink_node_id=sink_id,
                role=SinkEffectRole.PRIMARY,
                token_ids=token_ids,
            )
            == ()
        )
        assert not factory.execution.sink_effects.get_effects_for_run(run.run_id)
        assert target.publication_count == 0

        recovered_factory = make_factory(db)
        recovered_sink = PartitioningObservableSink(target, name="primary")
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
            pending_outcome=pending,
            effect_mode="write",
        )

        assert artifact is not None
        assert counts.total == 0
        assert target.publication_count == 1
        effects = recovered_factory.execution.sink_effects.get_effects_for_run(run.run_id)
        assert [effect.state.value for effect in effects] == ["finalized"]
        assert artifact.sink_effect_id == effects[0].effect_id
        for token in tokens:
            outcome = recovered_factory.data_flow.get_token_outcome(token.token_id)
            assert outcome is not None
            assert outcome.outcome is TerminalOutcome.SUCCESS
            # The re-drive reused the exact node state the crash left OPEN —
            # no replacement state was opened, and it is now terminal.
            states = [state for state in recovered_factory.query.get_node_states_for_token(token.token_id) if state.node_id == sink_id]
            assert len(states) == 1
            assert states[0].state_id == open_state_ids[token.token_id]
            assert states[0].status is not NodeStateStatus.OPEN
    finally:
        db.close()


def test_recovery_batch_spanning_effects_keys_dispositions_by_effect_and_ordinal(tmp_path: Path) -> None:
    """A recovered batch spanning two effects whose member ordinals both start
    at zero must attribute each diversion to the right effect member instead of
    collapsing into ordinal-only maps (elspeth-a6eba4b4e2)."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'span.db'}")
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        sink_id = register_test_node(factory.data_flow, run.run_id, "primary", node_type=NodeType.SINK, plugin_name="partitioning")
        tokens = _effect_tokens(
            factory,
            run_id=run.run_id,
            source_id=source_id,
            rows=[
                {"value": 1, "divert": True},
                {"value": 2},
                {"value": 3, "divert": True},
                {"value": 4},
            ],
        )
        ctx = PluginContext(run_id=run.run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=sink_id)
        pending = PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)

        # Effect 1 (member ordinals 0, 1): diverts token 0, accepts token 1.
        first_target = DuplicateObservableTarget()
        first_sink = PartitioningObservableSink(first_target, name="primary")
        first_sink.node_id = sink_id
        _artifact, first_counts = SinkExecutor(
            factory.execution,
            factory.data_flow,
            SpanFactory(),
            run.run_id,
            factory=factory,
            worker_id="worker-a",
        ).write(
            first_sink,  # type: ignore[arg-type]
            tokens[:2],
            ctx,
            1,
            sink_name="output",
            pending_outcome=pending,
            effect_mode="write",
        )
        assert first_counts.discard_mode == 1

        # Effect 2 (member ordinals restart at 0): diverts token 2, accepts
        # token 3, and is interrupted before its external commit.
        second_target = DuplicateObservableTarget()
        second_sink = PartitioningObservableSink(second_target, name="primary")
        second_sink.node_id = sink_id
        calls = 0

        def fail_once(observed: SinkEffectExecutionSeam) -> None:
            nonlocal calls
            if observed is SinkEffectExecutionSeam.BEFORE_EFFECT and calls == 0:
                calls += 1
                raise SinkEffectInjectedFault(observed)

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
                second_sink,  # type: ignore[arg-type]
                tokens[2:],
                ctx,
                1,
                sink_name="output",
                pending_outcome=pending,
                effect_mode="write",
            )

        # Recover the union batch: both effects' dispositions must be applied.
        recovered_factory = make_factory(db)
        recovered_sink = PartitioningObservableSink(second_target, name="primary")
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
            pending_outcome=pending,
            effect_mode="write",
            on_token_written=lambda token: None,
        )

        assert artifact is not None
        assert second_target.publication_count == 1
        assert counts.discard_mode == 2
        expected_error_hash = sha256(b"injected diversion").hexdigest()[:16]
        for diverted_token in (tokens[0], tokens[2]):
            outcome = recovered_factory.data_flow.get_token_outcome(diverted_token.token_id)
            assert outcome is not None
            assert outcome.outcome is TerminalOutcome.FAILURE
            assert outcome.path is TerminalPath.SINK_DISCARDED
            assert outcome.error_hash == expected_error_hash
        for accepted_token in (tokens[1], tokens[3]):
            outcome = recovered_factory.data_flow.get_token_outcome(accepted_token.token_id)
            assert outcome is not None
            assert outcome.outcome is TerminalOutcome.SUCCESS
    finally:
        db.close()


def test_all_diverted_primary_finalizes_virtual_no_publication_before_discard(tmp_path: Path) -> None:
    db = LandscapeDB(f"sqlite:///{tmp_path / 'no-publication.db'}")
    try:
        factory = make_factory(db)
        run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
        source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
        sink_id = register_test_node(factory.data_flow, run.run_id, "primary", node_type=NodeType.SINK, plugin_name="partitioning")
        token = _effect_tokens(
            factory,
            run_id=run.run_id,
            source_id=source_id,
            rows=[{"value": 1, "divert": True}],
        )[0]
        target = DuplicateObservableTarget()
        sink = PartitioningObservableSink(target, name="primary")
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
            [token],
            ctx,
            1,
            sink_name="output",
            pending_outcome=PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW),
            effect_mode="write",
        )

        assert artifact is not None
        assert artifact.publication_performed is False
        assert artifact.publication_evidence_kind == "virtual"
        assert target.publication_count == 0
        assert counts.discard_mode == 1
        outcome = factory.data_flow.get_token_outcome(token.token_id)
        assert outcome is not None
        assert outcome.outcome is TerminalOutcome.FAILURE
        assert outcome.path is TerminalPath.SINK_DISCARDED
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Effect-safety guards, the diverted-token checkpoint contract, and the
# primary-sink Layer 2 backstop (elspeth-c98e2afa53). Each guard test
# constructs the exact corrupted or divergent durable precondition its guard
# defends against and pins the fail-closed error contract.
# ---------------------------------------------------------------------------

_PENDING_SUCCESS = PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)


def _fail_once(seam: SinkEffectExecutionSeam) -> Callable[[SinkEffectExecutionSeam], None]:
    """Fault hook that crashes at ``seam`` on its first traversal only."""
    calls = 0

    def hook(observed: SinkEffectExecutionSeam) -> None:
        nonlocal calls
        if observed is seam and calls == 0:
            calls += 1
            raise SinkEffectInjectedFault(observed)

    return hook


def _executor_for(
    factory: RecorderFactory,
    run_id: str,
    *,
    fault_hook: Callable[[SinkEffectExecutionSeam], None] | None = None,
) -> SinkExecutor:
    return SinkExecutor(
        factory.execution,
        factory.data_flow,
        SpanFactory(),
        run_id,
        factory=factory,
        worker_id="worker-a",
        sink_effect_fault_hook=fault_hook,
    )


def _primary_setup(
    db: LandscapeDB,
    rows: list[dict[str, object]],
) -> tuple[RecorderFactory, str, str, list[TokenInfo], PluginContext]:
    """One run with a source, one primary sink node, and one token per row."""
    factory = make_factory(db)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
    sink_id = register_test_node(factory.data_flow, run.run_id, "primary", node_type=NodeType.SINK, plugin_name="partitioning")
    tokens = _effect_tokens(factory, run_id=run.run_id, source_id=source_id, rows=rows)
    ctx = PluginContext(run_id=run.run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=sink_id)
    return factory, run.run_id, sink_id, tokens, ctx


def _failsink_setup(
    db: LandscapeDB,
    rows: list[dict[str, object]],
) -> tuple[RecorderFactory, str, str, str, str, list[TokenInfo], PluginContext]:
    """Like ``_primary_setup`` plus a linked failsink node and DIVERT edge."""
    factory = make_factory(db)
    run = factory.run_lifecycle.begin_run(config={}, canonical_version="v1")
    source_id = register_test_node(factory.data_flow, run.run_id, "source", node_type=NodeType.SOURCE, plugin_name="source")
    primary_id = register_test_node(factory.data_flow, run.run_id, "primary", node_type=NodeType.SINK, plugin_name="partitioning")
    failsink_id = register_test_node(factory.data_flow, run.run_id, "failsink", node_type=NodeType.SINK, plugin_name="failsink")
    edge = factory.data_flow.register_edge(run.run_id, primary_id, failsink_id, "__failsink__", RoutingMode.DIVERT)
    tokens = _effect_tokens(factory, run_id=run.run_id, source_id=source_id, rows=rows)
    ctx = PluginContext(run_id=run.run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=primary_id)
    return factory, run.run_id, primary_id, failsink_id, edge.edge_id, tokens, ctx


def test_redrive_with_missing_node_state_witness_fails_closed(tmp_path: Path) -> None:
    """A durable member whose current node-state witness has disappeared must
    fail closed on re-drive instead of fabricating a replacement state
    (``_open_or_reuse_effect_states`` recovery witness guard)."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'witness.db'}")
    try:
        factory, run_id, sink_id, tokens, ctx = _primary_setup(db, [{"value": 1}])
        target = DuplicateObservableTarget()
        first_sink = PartitioningObservableSink(target, name="primary")
        first_sink.node_id = sink_id
        with pytest.raises(SinkEffectInjectedFault):
            _executor_for(factory, run_id, fault_hook=_fail_once(SinkEffectExecutionSeam.BEFORE_EFFECT)).write(
                first_sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )

        # Corrupt the durable image: the interrupted member row survives while
        # its node state vanishes (a partially restored audit database). The
        # opened state has no dependent rows yet, so FK enforcement allows it.
        with db.engine.begin() as conn:
            conn.execute(node_states_table.delete().where(node_states_table.c.token_id == tokens[0].token_id))

        recovered_factory = make_factory(db)
        recovered_sink = PartitioningObservableSink(target, name="primary")
        recovered_sink.node_id = sink_id
        with pytest.raises(AuditIntegrityError, match="current node-state witness is divergent"):
            _executor_for(recovered_factory, run_id).write(
                recovered_sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )
        assert target.publication_count == 0
    finally:
        db.close()


def test_durable_partition_dropping_a_requested_token_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """INVARIANT: once effect execution returns, every requested primary token
    has a durable member row; finalizing from a partition that lost one would
    leak an unaccounted token. The coordinator persists that partition inside
    the same ``write()`` call, so the divergence is unreachable through the
    public API — the corrupted read is simulated by shearing the post-execution
    member query, armed only after the last coordinator seam."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'partition.db'}")
    try:
        factory, run_id, sink_id, tokens, ctx = _primary_setup(db, [{"value": 1}, {"value": 2}])
        armed = False

        def arm(observed: SinkEffectExecutionSeam) -> None:
            nonlocal armed
            if observed is SinkEffectExecutionSeam.AFTER_FINALIZE_BEFORE_RESPONSE:
                armed = True

        original_get_members = factory.execution.sink_effects.get_members_for_tokens

        def shear(
            *,
            run_id: str,
            sink_node_id: str,
            role: SinkEffectRole,
            token_ids: tuple[str, ...],
        ) -> tuple[SinkEffectMemberRecord, ...]:
            members = original_get_members(run_id=run_id, sink_node_id=sink_node_id, role=role, token_ids=token_ids)
            return members[:-1] if armed else members

        monkeypatch.setattr(factory.execution.sink_effects, "get_members_for_tokens", shear)
        sink = PartitioningObservableSink(DuplicateObservableTarget(), name="primary")
        sink.node_id = sink_id

        with pytest.raises(AuditIntegrityError, match="does not cover every requested primary token"):
            _executor_for(factory, run_id, fault_hook=arm).write(
                sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )
    finally:
        db.close()


def test_finalized_member_stripped_of_disposition_fails_accepted_checkpoint(tmp_path: Path) -> None:
    """A durable member whose accepted disposition was lost must stop the
    Phase 2 checkpoint: checkpointing a token the durable partition no longer
    vouches for would permit a duplicate publication on resume."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'disposition.db'}")
    try:
        factory, run_id, sink_id, tokens, ctx = _primary_setup(db, [{"value": 1}, {"value": 2}])
        target = DuplicateObservableTarget()
        first_sink = PartitioningObservableSink(target, name="primary")
        first_sink.node_id = sink_id
        artifact, _counts = _executor_for(factory, run_id).write(
            first_sink,  # type: ignore[arg-type]
            tokens,
            ctx,
            1,
            sink_name="output",
            pending_outcome=_PENDING_SUCCESS,
            effect_mode="write",
        )
        assert artifact is not None

        # Strip ordinal 0's accepted disposition. NULL passes the schema CHECK
        # (unlike any non-vocabulary string), modelling a lost disposition
        # write rather than a corrupted one.
        effect = factory.execution.sink_effects.get_effects_for_run(run_id)[0]
        with db.engine.begin() as conn:
            conn.execute(
                sink_effect_members_table.update()
                .where(
                    sink_effect_members_table.c.effect_id == effect.effect_id,
                    sink_effect_members_table.c.ordinal == 0,
                )
                .values(prepared_disposition=None)
            )

        recovered_factory = make_factory(db)
        recovered_sink = PartitioningObservableSink(target, name="primary")
        recovered_sink.node_id = sink_id
        with pytest.raises(AuditIntegrityError, match="disagrees with accepted primary tokens"):
            _executor_for(recovered_factory, run_id).write(
                recovered_sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
                on_token_written=lambda token: None,
            )
    finally:
        db.close()


def test_durable_partition_referencing_missing_effect_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """INVARIANT: every effect id carried by durable members resolves to a
    ledger row when diversion evidence is recovered. A member row referencing
    a missing effect requires ledger corruption (FK enforcement forbids
    producing it through any repository write), so the guard is pinned by
    blanking the effect lookup after the last coordinator seam — the
    coordinator itself must keep seeing the real row."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'missing-effect.db'}")
    try:
        factory, run_id, sink_id, tokens, ctx = _primary_setup(db, [{"value": 1}, {"value": 2}])
        armed = False

        def arm(observed: SinkEffectExecutionSeam) -> None:
            nonlocal armed
            if observed is SinkEffectExecutionSeam.AFTER_FINALIZE_BEFORE_RESPONSE:
                armed = True

        original_get_effect = factory.execution.sink_effects.get_effect

        def vanish(effect_id: str) -> SinkEffect | None:
            return None if armed else original_get_effect(effect_id)

        monkeypatch.setattr(factory.execution.sink_effects, "get_effect", vanish)
        sink = PartitioningObservableSink(DuplicateObservableTarget(), name="primary")
        sink.node_id = sink_id

        with pytest.raises(AuditIntegrityError, match="references a missing effect"):
            _executor_for(factory, run_id, fault_hook=arm).write(
                sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )
    finally:
        db.close()


class _TaintedCommitEvidenceSink(PartitioningObservableSink):
    """Partitioning double whose commit result carries caller-chosen
    diversion-attribution evidence (the durable evidence the executor
    recovers commit-time diversions from)."""

    def __init__(self, target: DuplicateObservableTarget, *, name: str, attribution: tuple[object, ...]) -> None:
        super().__init__(target, name=name)
        self._attribution = attribution

    def commit_effect(self, plan: SinkEffectPlan, ctx: RestrictedSinkEffectContext) -> SinkEffectCommitResult:
        result = super().commit_effect(plan, ctx)
        return SinkEffectCommitResult(
            descriptor=result.descriptor,
            evidence={"diversion_attribution": list(self._attribution), "effect_id": plan.effect_id},
            accepted_ordinals=result.accepted_ordinals,
            diverted_ordinals=result.diverted_ordinals,
        )


@pytest.mark.parametrize(
    ("attribution", "match"),
    (
        pytest.param(
            ("not-a-mapping",),
            "effect diversion attribution is not a mapping",
            id="non-mapping-entry",
        ),
        pytest.param(
            ({"ordinal": "0", "reason_hash": "a" * 64, "error_hash": "a" * 16},),
            "effect diversion attribution is incomplete",
            id="mistyped-ordinal",
        ),
        pytest.param(
            ({"ordinal": 0, "reason_hash": "b" * 64, "error_hash": "b" * 16},),
            "effect diversion attribution sources diverge for one durable member",
            id="divergent-sources",
        ),
    ),
)
def test_malformed_commit_diversion_attribution_fails_closed(
    tmp_path: Path,
    attribution: tuple[object, ...],
    match: str,
) -> None:
    """Commit-returned diversion attribution is durable audit evidence: a
    non-mapping entry, an incompletely typed entry, or an entry disagreeing
    with the plan's attribution for the same durable member must fail closed
    rather than silently re-attribute a diversion."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'attribution.db'}")
    try:
        factory, run_id, sink_id, tokens, ctx = _primary_setup(db, [{"value": 1, "divert": True}, {"value": 2}])
        sink = _TaintedCommitEvidenceSink(DuplicateObservableTarget(), name="primary", attribution=attribution)
        sink.node_id = sink_id

        with pytest.raises(AuditIntegrityError, match=match):
            _executor_for(factory, run_id).write(
                sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )
    finally:
        db.close()


class _PhantomDiversionLogSink(PartitioningObservableSink):
    """Durably accepts every member while its in-memory log claims a diversion."""

    def prepare_effect(self, request: SinkEffectPrepareRequest, ctx: RestrictedSinkEffectContext) -> SinkEffectPlan:
        plan = super().prepare_effect(request, ctx)
        self._diversions = (RowDiversion(row_index=0, reason="phantom diversion", row_data={"value": 1}),)
        return plan


def test_in_memory_diversion_log_disagreeing_with_durable_partition_fails_closed(tmp_path: Path) -> None:
    """The sink's returned in-memory diversion log must exactly match the
    durable member partition; a phantom diversion the ledger never recorded
    must fail closed instead of failing a token the effect accepted."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'phantom.db'}")
    try:
        factory, run_id, sink_id, tokens, ctx = _primary_setup(db, [{"value": 1}, {"value": 2}])
        sink = _PhantomDiversionLogSink(DuplicateObservableTarget(), name="primary")
        sink.node_id = sink_id

        with pytest.raises(AuditIntegrityError, match="does not match the durable member partition"):
            _executor_for(factory, run_id).write(
                sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )
    finally:
        db.close()


class _UnattributedDivertingSink(PartitioningObservableSink):
    """Diverting double whose plan omits its durable diversion attribution."""

    def prepare_effect(self, request: SinkEffectPrepareRequest, ctx: RestrictedSinkEffectContext) -> SinkEffectPlan:
        plan = super().prepare_effect(request, ctx)
        evidence = {key: value for key, value in deep_thaw(plan.safe_evidence).items() if key != "diversion_attribution"}
        return replace(plan, safe_evidence=evidence)


def test_recovered_diversion_without_durable_attribution_fails_closed(tmp_path: Path) -> None:
    """A recovered effect (no in-memory diversion log — its plan was durably
    bound before the crash, so ``prepare_effect`` never re-runs) must find its
    diversion attribution in the durable plan or attempt evidence; when the
    plan was bound without it, the re-drive must fail closed instead of
    inventing diversion reasons."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'unattributed.db'}")
    try:
        factory, run_id, sink_id, tokens, ctx = _primary_setup(db, [{"value": 1}, {"value": 2, "divert": True}])
        target = DuplicateObservableTarget()
        first_sink = _UnattributedDivertingSink(target, name="primary")
        first_sink.node_id = sink_id
        with pytest.raises(SinkEffectInjectedFault):
            _executor_for(factory, run_id, fault_hook=_fail_once(SinkEffectExecutionSeam.BEFORE_EFFECT)).write(
                first_sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )

        recovered_factory = make_factory(db)
        recovered_sink = _UnattributedDivertingSink(target, name="primary")
        recovered_sink.node_id = sink_id
        with pytest.raises(AuditIntegrityError, match="recovered effect is missing durable diversion attribution"):
            _executor_for(recovered_factory, run_id).write(
                recovered_sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )
    finally:
        db.close()


def test_failsink_diverting_a_linked_member_is_a_framework_bug(tmp_path: Path) -> None:
    """A linked failsink is the terminal quarantine for already-diverted rows:
    it must accept every member. One that diverts again is a framework bug,
    not a routable diversion."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'requarantine.db'}")
    try:
        factory, run_id, primary_id, failsink_id, edge_id, tokens, ctx = _failsink_setup(
            db,
            [{"value": 1}, {"value": 2, "divert": True}],
        )
        primary = PartitioningObservableSink(DuplicateObservableTarget(), name="primary")
        primary.node_id = primary_id
        # divert_rows stays True: the enriched quarantine row still carries the
        # original ``divert`` marker, so the failsink re-diverts its member.
        failsink = PartitioningObservableSink(DuplicateObservableTarget(), name="failsink")
        failsink.node_id = failsink_id

        with pytest.raises(FrameworkBugError, match="diverted a linked failsink member"):
            _executor_for(factory, run_id).write(
                primary,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
                failsink=failsink,  # type: ignore[arg-type]
                failsink_name="failsink",
                failsink_effect_mode="write",
                failsink_edge_id=edge_id,
            )
    finally:
        db.close()


def test_redrive_with_missing_divert_routing_event_fails_closed(tmp_path: Path) -> None:
    """A finalized diverted anchor must still carry exactly its one DIVERT
    routing event on re-drive; missing or divergent routing evidence means the
    audit trail no longer proves where the row went, and must fail closed."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'routing.db'}")
    try:
        factory, run_id, primary_id, failsink_id, edge_id, tokens, ctx = _failsink_setup(
            db,
            [{"value": 1}, {"value": 2, "divert": True}],
        )
        diverted = tokens[1]
        primary_target = DuplicateObservableTarget()
        failsink_target = DuplicateObservableTarget()
        primary = PartitioningObservableSink(primary_target, name="primary")
        primary.node_id = primary_id
        failsink = PartitioningObservableSink(failsink_target, name="failsink", divert_rows=False)
        failsink.node_id = failsink_id
        _artifact, counts = _executor_for(factory, run_id).write(
            primary,  # type: ignore[arg-type]
            tokens,
            ctx,
            1,
            sink_name="output",
            pending_outcome=_PENDING_SUCCESS,
            effect_mode="write",
            failsink=failsink,  # type: ignore[arg-type]
            failsink_name="failsink",
            failsink_effect_mode="write",
            failsink_edge_id=edge_id,
        )
        assert counts.failsink_mode == 1

        primary_state = next(state for state in factory.query.get_node_states_for_token(diverted.token_id) if state.node_id == primary_id)
        assert len(factory.query.get_routing_events(primary_state.state_id)) == 1
        # Corrupt the durable image: erase the DIVERT routing event while the
        # FAILED anchor and the finalized failsink effect survive.
        with db.engine.begin() as conn:
            conn.execute(routing_events_table.delete().where(routing_events_table.c.state_id == primary_state.state_id))

        recovered_factory = make_factory(db)
        recovered_primary = PartitioningObservableSink(primary_target, name="primary")
        recovered_primary.node_id = primary_id
        recovered_failsink = PartitioningObservableSink(failsink_target, name="failsink", divert_rows=False)
        recovered_failsink.node_id = failsink_id
        with pytest.raises(AuditIntegrityError, match="divergent primary routing evidence"):
            _executor_for(recovered_factory, run_id).write(
                recovered_primary,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
                failsink=recovered_failsink,  # type: ignore[arg-type]
                failsink_name="failsink",
                failsink_effect_mode="write",
                failsink_edge_id=edge_id,
            )
    finally:
        db.close()


def test_redrive_with_completed_failsink_primary_anchor_fails_closed(tmp_path: Path) -> None:
    """A diverted token's primary anchor may only be OPEN (first drive) or
    FAILED (already finalized); a COMPLETED anchor claims the row reached the
    primary sink and re-driving its diversion would double-account the row."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'anchor-failsink.db'}")
    try:
        factory, run_id, primary_id, failsink_id, edge_id, tokens, ctx = _failsink_setup(
            db,
            [{"value": 1}, {"value": 2, "divert": True}],
        )
        diverted = tokens[1]
        primary_target = DuplicateObservableTarget()
        failsink_target = DuplicateObservableTarget()
        primary = PartitioningObservableSink(primary_target, name="primary")
        primary.node_id = primary_id
        failsink = PartitioningObservableSink(failsink_target, name="failsink", fail_prepare_once=True, divert_rows=False)
        failsink.node_id = failsink_id
        with pytest.raises(RuntimeError, match="between primary and failsink"):
            _executor_for(factory, run_id).write(
                primary,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
                failsink=failsink,  # type: ignore[arg-type]
                failsink_name="failsink",
                failsink_effect_mode="write",
                failsink_edge_id=edge_id,
            )

        # Corrupt the interrupted image through the repository itself: close
        # the still-open divert anchor as COMPLETED, as a buggy rival caller
        # could have done.
        open_ids = factory.execution.get_open_node_state_ids(run_id, node_ids=(primary_id,), token_ids=(diverted.token_id,))
        factory.execution.complete_node_state(
            state_id=open_ids[diverted.token_id],
            status=NodeStateStatus.COMPLETED,
            output_data={"rival": True},
            duration_ms=0.0,
        )

        recovered_factory = make_factory(db)
        recovered_primary = PartitioningObservableSink(primary_target, name="primary")
        recovered_primary.node_id = primary_id
        recovered_failsink = PartitioningObservableSink(failsink_target, name="failsink", divert_rows=False)
        recovered_failsink.node_id = failsink_id
        with pytest.raises(AuditIntegrityError, match="primary anchor is not OPEN or FAILED"):
            _executor_for(recovered_factory, run_id).write(
                recovered_primary,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
                failsink=recovered_failsink,  # type: ignore[arg-type]
                failsink_name="failsink",
                failsink_effect_mode="write",
                failsink_edge_id=edge_id,
            )
    finally:
        db.close()


def test_redrive_with_completed_discard_primary_anchor_fails_closed(tmp_path: Path) -> None:
    """Discard-mode mirror of the failsink anchor guard: a COMPLETED anchor
    for a durably diverted token contradicts the discard evidence."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'anchor-discard.db'}")
    try:
        factory, run_id, sink_id, tokens, ctx = _primary_setup(db, [{"value": 1}, {"value": 2, "divert": True}])
        diverted = tokens[1]
        target = DuplicateObservableTarget()
        first_sink = PartitioningObservableSink(target, name="primary")
        first_sink.node_id = sink_id
        # Crash after the primary effect finalizes but before Phase 3 records
        # the discard, leaving the diverted token's anchor OPEN.
        with pytest.raises(SinkEffectInjectedFault):
            _executor_for(
                factory,
                run_id,
                fault_hook=_fail_once(SinkEffectExecutionSeam.AFTER_FINALIZE_BEFORE_RESPONSE),
            ).write(
                first_sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )

        open_ids = factory.execution.get_open_node_state_ids(run_id, node_ids=(sink_id,), token_ids=(diverted.token_id,))
        factory.execution.complete_node_state(
            state_id=open_ids[diverted.token_id],
            status=NodeStateStatus.COMPLETED,
            output_data={"rival": True},
            duration_ms=0.0,
        )

        recovered_factory = make_factory(db)
        recovered_sink = PartitioningObservableSink(target, name="primary")
        recovered_sink.node_id = sink_id
        with pytest.raises(AuditIntegrityError, match="discard primary anchor is not OPEN or FAILED"):
            _executor_for(recovered_factory, run_id).write(
                recovered_sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )
    finally:
        db.close()


def test_redrive_with_divergent_discard_outcome_fails_closed(tmp_path: Path) -> None:
    """Discard recovery must verify an existing terminal outcome is the exact
    discard it would have recorded; any divergence (here the error hash) means
    two irreconcilable accounts of the same token and must fail closed."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'discard-outcome.db'}")
    try:
        factory, run_id, sink_id, tokens, ctx = _primary_setup(db, [{"value": 1}, {"value": 2, "divert": True}])
        diverted = tokens[1]
        target = DuplicateObservableTarget()
        first_sink = PartitioningObservableSink(target, name="primary")
        first_sink.node_id = sink_id
        with pytest.raises(SinkEffectInjectedFault):
            _executor_for(
                factory,
                run_id,
                fault_hook=_fail_once(SinkEffectExecutionSeam.AFTER_FINALIZE_BEFORE_RESPONSE),
            ).write(
                first_sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )

        # A rival recorded a discard outcome that disagrees with the durable
        # diversion attribution (wrong error hash).
        factory.data_flow.record_token_outcome(
            ref=TokenRef(token_id=diverted.token_id, run_id=run_id),
            outcome=TerminalOutcome.FAILURE,
            path=TerminalPath.SINK_DISCARDED,
            error_hash="0" * 16,
            sink_name="__discard__",
        )

        recovered_factory = make_factory(db)
        recovered_sink = PartitioningObservableSink(target, name="primary")
        recovered_sink.node_id = sink_id
        with pytest.raises(AuditIntegrityError, match="recovered sink discard outcome is divergent"):
            _executor_for(recovered_factory, run_id).write(
                recovered_sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )
    finally:
        db.close()


def test_failsink_execution_requires_name_mode_and_routing_edge(tmp_path: Path) -> None:
    """A configured failsink without its name, mode, and routing edge is an
    orchestrator wiring bug the diversion phase must refuse to route through."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'failsink-wiring.db'}")
    try:
        factory, run_id, primary_id, failsink_id, edge_id, tokens, ctx = _failsink_setup(
            db,
            [{"value": 1}, {"value": 2, "divert": True}],
        )
        primary = PartitioningObservableSink(DuplicateObservableTarget(), name="primary")
        primary.node_id = primary_id
        failsink = PartitioningObservableSink(DuplicateObservableTarget(), name="failsink", divert_rows=False)
        failsink.node_id = failsink_id

        with pytest.raises(OrchestrationInvariantError, match="requires its name, mode, and routing edge"):
            _executor_for(factory, run_id).write(
                primary,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
                failsink=failsink,  # type: ignore[arg-type]
                failsink_name=None,
                failsink_effect_mode="write",
                failsink_edge_id=edge_id,
            )
    finally:
        db.close()


# --- Diverted-token checkpoint contract (SinkExecutor.write docstring):
# primary tokens are checkpointed after Phase 2, diverted tokens after
# Phase 3, and every token exactly once — a diverted token missing its
# checkpoint would be re-driven (duplicate write) on resume.


def test_on_token_written_checkpoints_every_token_exactly_once_discard_mode(tmp_path: Path) -> None:
    db = LandscapeDB(f"sqlite:///{tmp_path / 'checkpoint-discard.db'}")
    try:
        factory, run_id, sink_id, tokens, ctx = _primary_setup(db, [{"value": 1}, {"value": 2, "divert": True}])
        sink = PartitioningObservableSink(DuplicateObservableTarget(), name="primary")
        sink.node_id = sink_id
        checkpointed: list[str] = []

        artifact, counts = _executor_for(factory, run_id).write(
            sink,  # type: ignore[arg-type]
            tokens,
            ctx,
            1,
            sink_name="output",
            pending_outcome=_PENDING_SUCCESS,
            effect_mode="write",
            on_token_written=lambda token: checkpointed.append(token.token_id),
        )

        assert artifact is not None
        assert counts.discard_mode == 1
        # Accepted AND diverted tokens each checkpoint exactly once.
        assert sorted(checkpointed) == sorted(token.token_id for token in tokens)
    finally:
        db.close()


def test_on_token_written_checkpoints_every_token_exactly_once_failsink_mode(tmp_path: Path) -> None:
    db = LandscapeDB(f"sqlite:///{tmp_path / 'checkpoint-failsink.db'}")
    try:
        factory, run_id, primary_id, failsink_id, edge_id, tokens, ctx = _failsink_setup(
            db,
            [{"value": 1}, {"value": 2, "divert": True}],
        )
        primary = PartitioningObservableSink(DuplicateObservableTarget(), name="primary")
        primary.node_id = primary_id
        failsink = PartitioningObservableSink(DuplicateObservableTarget(), name="failsink", divert_rows=False)
        failsink.node_id = failsink_id
        checkpointed: list[str] = []

        artifact, counts = _executor_for(factory, run_id).write(
            primary,  # type: ignore[arg-type]
            tokens,
            ctx,
            1,
            sink_name="output",
            pending_outcome=_PENDING_SUCCESS,
            effect_mode="write",
            failsink=failsink,  # type: ignore[arg-type]
            failsink_name="failsink",
            failsink_effect_mode="write",
            failsink_edge_id=edge_id,
            on_token_written=lambda token: checkpointed.append(token.token_id),
        )

        assert artifact is not None
        assert counts.failsink_mode == 1
        assert sorted(checkpointed) == sorted(token.token_id for token in tokens)
    finally:
        db.close()


def test_primary_tokens_checkpointed_before_diverted_tokens(tmp_path: Path) -> None:
    """All accepted tokens checkpoint (Phase 2) before any diverted token
    (Phase 3): a checkpoint order inversion would let a resume skip a diverted
    token whose terminal path is not durable yet."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'checkpoint-order.db'}")
    try:
        factory, run_id, sink_id, tokens, ctx = _primary_setup(
            db,
            [
                {"value": 1, "divert": True},
                {"value": 2},
                {"value": 3, "divert": True},
                {"value": 4},
            ],
        )
        sink = PartitioningObservableSink(DuplicateObservableTarget(), name="primary")
        sink.node_id = sink_id
        checkpointed: list[str] = []

        _artifact, counts = _executor_for(factory, run_id).write(
            sink,  # type: ignore[arg-type]
            tokens,
            ctx,
            1,
            sink_name="output",
            pending_outcome=_PENDING_SUCCESS,
            effect_mode="write",
            on_token_written=lambda token: checkpointed.append(token.token_id),
        )

        assert counts.discard_mode == 2
        accepted_ids = [tokens[1].token_id, tokens[3].token_id]
        diverted_ids = [tokens[0].token_id, tokens[2].token_id]
        assert checkpointed == accepted_ids + diverted_ids
    finally:
        db.close()


class _RequiredFieldsPrimarySink(PartitioningObservableSink):
    """Primary sink whose transactional backstop requires ``must_exist``."""

    declared_required_fields = frozenset({"must_exist"})


def test_primary_validation_rejection_terminalizes_states_and_outcomes(tmp_path: Path) -> None:
    """PRIMARY-sink mirror of the failsink Layer 2 backstop test above
    (ADR-010 F3): a row missing a declared required field at the transactional
    commit boundary must raise SinkTransactionalInvariantError (Tier 1 — never
    absorbed into diversion), terminalize every opened node state as FAILED,
    and record FAILED token outcomes for the whole batch."""
    db = LandscapeDB(f"sqlite:///{tmp_path / 'primary-reject.db'}")
    try:
        factory, run_id, sink_id, tokens, ctx = _primary_setup(db, [{"value": 1, "must_exist": True}, {"value": 2}])
        target = DuplicateObservableTarget()
        sink = _RequiredFieldsPrimarySink(target, name="primary")
        sink.node_id = sink_id

        with pytest.raises(SinkTransactionalInvariantError, match="must_exist"):
            _executor_for(factory, run_id).write(
                sink,  # type: ignore[arg-type]
                tokens,
                ctx,
                1,
                sink_name="output",
                pending_outcome=_PENDING_SUCCESS,
                effect_mode="write",
            )

        # The rejection happened before any effect was reserved or published.
        assert target.publication_count == 0
        assert not factory.execution.sink_effects.get_effects_for_run(run_id)
        for token in tokens:
            # No opened state is leaked OPEN — all are terminalized FAILED.
            states = [state for state in factory.query.get_node_states_for_token(token.token_id) if state.node_id == sink_id]
            assert states
            assert all(state.status is NodeStateStatus.FAILED for state in states)
            outcome = factory.data_flow.get_token_outcome(token.token_id)
            assert outcome is not None
            assert outcome.outcome is TerminalOutcome.FAILURE
            assert outcome.path is TerminalPath.UNROUTED
    finally:
        db.close()
