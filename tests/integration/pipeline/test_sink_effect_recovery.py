"""Production pipeline-boundary recovery for durable sink effects."""

from __future__ import annotations

from pathlib import Path

import pytest

from elspeth.contracts import NodeType, PendingOutcome, RoutingMode, TerminalOutcome, TerminalPath, TokenInfo
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.core.landscape.database import LandscapeDB
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
