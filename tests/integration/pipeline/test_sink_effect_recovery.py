"""Production pipeline-boundary recovery for durable sink effects."""

from __future__ import annotations

from pathlib import Path

import pytest

from elspeth.contracts import NodeType, PendingOutcome, TerminalOutcome, TerminalPath, TokenInfo
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.core.landscape.database import LandscapeDB
from elspeth.engine.executors.sink import SinkExecutor
from elspeth.engine.executors.sink_effects import SinkEffectExecutionSeam, SinkEffectInjectedFault
from elspeth.engine.spans import SpanFactory
from tests.fixtures.base_classes import create_observed_contract
from tests.fixtures.landscape import make_factory, register_test_node
from tests.fixtures.sink_effects import DuplicateObservableSink, DuplicateObservableTarget


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
