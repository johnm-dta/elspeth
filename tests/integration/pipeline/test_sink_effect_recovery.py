"""Production pipeline-boundary recovery for durable sink effects."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from elspeth.contracts import NodeStateStatus, NodeType, PendingOutcome, RoutingMode, TerminalOutcome, TerminalPath, TokenInfo
from elspeth.contracts.errors import SinkTransactionalInvariantError
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.contracts.sink_effects import SinkEffectReconcileResult, SinkEffectRole
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.execution.sink_effect_reservation import SinkEffectReservation
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
