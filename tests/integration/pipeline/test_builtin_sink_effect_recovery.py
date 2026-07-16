"""Recovery proofs for the real local-file sink effect adapters."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest

from elspeth.contracts import Artifact, NodeType, PendingOutcome, RoutingMode, TerminalOutcome, TerminalPath, TokenInfo
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.contracts.sink_effects import SinkEffectRole, SinkEffectState
from elspeth.core.landscape.database import LandscapeDB
from elspeth.core.landscape.factory import RecorderFactory
from elspeth.engine.executors.sink import DiversionCounts, SinkExecutor
from elspeth.engine.executors.sink_effects import SinkEffectExecutionSeam, SinkEffectInjectedFault
from elspeth.engine.spans import SpanFactory
from elspeth.plugins.infrastructure.base import BaseSink
from elspeth.plugins.sinks.csv_sink import CSVSink
from elspeth.plugins.sinks.json_sink import JSONSink
from tests.fixtures.base_classes import create_observed_contract, inject_write_failure
from tests.fixtures.landscape import make_factory, register_test_node


def _tokens(
    factory: RecorderFactory,
    *,
    run_id: str,
    source_id: str,
    rows: list[dict[str, object]],
) -> list[TokenInfo]:
    result: list[TokenInfo] = []
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
        result.append(
            TokenInfo(
                row_id=row.row_id,
                token_id=durable.token_id,
                row_data=PipelineRow(row_data, create_observed_contract(row_data)),
            )
        )
    return result


def _begin(db: LandscapeDB) -> tuple[RecorderFactory, str, str]:
    factory = make_factory(db)
    run = factory.run_lifecycle.begin_run(
        config={},
        canonical_version="v1",
        openrouter_catalog_sha256="0" * 64,
        openrouter_catalog_source="bundled",
    )
    source_id = register_test_node(
        factory.data_flow,
        run.run_id,
        "source",
        node_type=NodeType.SOURCE,
        plugin_name="source",
    )
    return factory, run.run_id, source_id


def _register_sink(factory: RecorderFactory, run_id: str, *, name: str, plugin_name: str) -> str:
    return register_test_node(
        factory.data_flow,
        run_id,
        name,
        node_type=NodeType.SINK,
        plugin_name=plugin_name,
    )


def _csv_sink(path: Path, *, encoding: str = "utf-8", on_write_failure: str = "discard") -> CSVSink:
    return inject_write_failure(
        CSVSink(
            {
                "path": str(path),
                "encoding": encoding,
                "mode": "write",
                "schema": {"mode": "observed"},
            }
        ),
        on_write_failure,
    )


def _json_sink(path: Path, *, on_write_failure: str = "discard") -> JSONSink:
    return inject_write_failure(
        JSONSink(
            {
                "path": str(path),
                "format": "jsonl",
                "mode": "write",
                "schema": {"mode": "observed"},
            }
        ),
        on_write_failure,
    )


def _pending_success() -> PendingOutcome:
    return PendingOutcome(outcome=TerminalOutcome.SUCCESS, path=TerminalPath.DEFAULT_FLOW)


def _write(
    *,
    factory: RecorderFactory,
    run_id: str,
    sink: BaseSink,
    tokens: list[TokenInfo],
    sink_name: str = "output",
    fault_hook: Callable[[SinkEffectExecutionSeam], None] | None = None,
) -> tuple[Artifact | None, DiversionCounts]:
    assert sink.node_id is not None
    ctx = PluginContext(run_id=run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=sink.node_id)
    return SinkExecutor(
        factory.execution,
        factory.data_flow,
        SpanFactory(),
        run_id,
        factory=factory,
        worker_id="worker-a",
        sink_effect_fault_hook=fault_hook,
    ).write(
        sink,
        tokens,
        ctx,
        1,
        sink_name=sink_name,
        pending_outcome=_pending_success(),
        effect_mode="write",
    )


def test_csv_mixed_primary_publishes_accepted_row_and_discards_exact_diversion(tmp_path: Path) -> None:
    output = tmp_path / "mixed.csv"
    db = LandscapeDB(f"sqlite:///{tmp_path / 'mixed.db'}")
    try:
        factory, run_id, source_id = _begin(db)
        sink_id = _register_sink(factory, run_id, name="output", plugin_name="csv")
        accepted, diverted = _tokens(
            factory,
            run_id=run_id,
            source_id=source_id,
            rows=[{"id": 1}, {"id": 2, "unexpected": "quarantine me"}],
        )
        sink = _csv_sink(output)
        sink.node_id = sink_id

        artifact, counts = _write(factory=factory, run_id=run_id, sink=sink, tokens=[accepted, diverted])

        assert artifact is not None
        assert artifact.publication_performed is True
        assert counts == DiversionCounts(discard_mode=1)
        with output.open(newline="", encoding="utf-8") as stream:
            assert list(csv.DictReader(stream)) == [{"id": "1"}]

        accepted_outcome = factory.data_flow.get_token_outcome(accepted.token_id)
        diverted_outcome = factory.data_flow.get_token_outcome(diverted.token_id)
        assert accepted_outcome is not None
        assert accepted_outcome.outcome is TerminalOutcome.SUCCESS
        assert accepted_outcome.path is TerminalPath.DEFAULT_FLOW
        assert accepted_outcome.sink_name == "output"
        assert diverted_outcome is not None
        assert diverted_outcome.outcome is TerminalOutcome.FAILURE
        assert diverted_outcome.path is TerminalPath.SINK_DISCARDED
        assert diverted_outcome.sink_name == "__discard__"

        effects = factory.execution.sink_effects.get_effects_for_run(run_id)
        assert len(effects) == 1
        assert effects[0].state is SinkEffectState.FINALIZED
        members = factory.execution.sink_effects.get_members(effects[0].effect_id)
        assert [(member.ordinal, member.prepared_disposition) for member in members] == [(0, "accepted"), (1, "diverted")]
    finally:
        db.close()


def test_csv_primary_routes_one_diversion_through_linked_json_failsink(tmp_path: Path) -> None:
    primary_output = tmp_path / "primary.csv"
    failsink_output = tmp_path / "failsink.jsonl"
    db = LandscapeDB(f"sqlite:///{tmp_path / 'linked.db'}")
    try:
        factory, run_id, source_id = _begin(db)
        primary_id = _register_sink(factory, run_id, name="output", plugin_name="csv")
        failsink_id = _register_sink(factory, run_id, name="quarantine", plugin_name="json")
        edge = factory.data_flow.register_edge(run_id, primary_id, failsink_id, "__failsink__", RoutingMode.DIVERT)
        accepted, diverted = _tokens(
            factory,
            run_id=run_id,
            source_id=source_id,
            rows=[{"id": 1}, {"id": 2, "unexpected": "preserve me"}],
        )
        primary = _csv_sink(primary_output, on_write_failure="quarantine")
        primary.node_id = primary_id
        failsink = _json_sink(failsink_output)
        failsink.node_id = failsink_id
        ctx = PluginContext(run_id=run_id, config={}, landscape=factory.plugin_audit_writer(), node_id=primary_id)
        returned_commits = 0

        def lose_failsink_response(seam: SinkEffectExecutionSeam) -> None:
            nonlocal returned_commits
            if seam is not SinkEffectExecutionSeam.AFTER_EFFECT_BEFORE_RETURN:
                return
            returned_commits += 1
            if returned_commits == 2:
                raise SinkEffectInjectedFault(seam)

        with pytest.raises(SinkEffectInjectedFault):
            SinkExecutor(
                factory.execution,
                factory.data_flow,
                SpanFactory(),
                run_id,
                factory=factory,
                worker_id="worker-a",
                sink_effect_fault_hook=lose_failsink_response,
            ).write(
                primary,
                [accepted, diverted],
                ctx,
                1,
                sink_name="output",
                pending_outcome=_pending_success(),
                effect_mode="write",
                failsink=failsink,
                failsink_name="quarantine",
                failsink_effect_mode="write",
                failsink_edge_id=edge.edge_id,
            )

        first_effects = factory.execution.sink_effects.get_effects_for_run(run_id)
        first_primary = next(effect for effect in first_effects if effect.role is SinkEffectRole.PRIMARY)
        first_failsink = next(effect for effect in first_effects if effect.role is SinkEffectRole.FAILSINK)
        assert first_primary.state is SinkEffectState.FINALIZED
        assert first_failsink.state is SinkEffectState.IN_FLIGHT
        assert factory.data_flow.get_token_outcome(diverted.token_id) is None

        recovered_factory = make_factory(db)
        recovered_primary = _csv_sink(primary_output, on_write_failure="quarantine")
        recovered_primary.node_id = primary_id
        recovered_failsink = _json_sink(failsink_output)
        recovered_failsink.node_id = failsink_id
        recovered_ctx = PluginContext(
            run_id=run_id,
            config={},
            landscape=recovered_factory.plugin_audit_writer(),
            node_id=primary_id,
        )
        artifact, counts = SinkExecutor(
            recovered_factory.execution,
            recovered_factory.data_flow,
            SpanFactory(),
            run_id,
            factory=recovered_factory,
            worker_id="worker-a",
        ).write(
            recovered_primary,
            [accepted, diverted],
            recovered_ctx,
            1,
            sink_name="output",
            pending_outcome=_pending_success(),
            effect_mode="write",
            failsink=recovered_failsink,
            failsink_name="quarantine",
            failsink_effect_mode="write",
            failsink_edge_id=edge.edge_id,
        )

        assert artifact is not None
        assert counts == DiversionCounts(failsink_mode=1)
        with primary_output.open(newline="", encoding="utf-8") as stream:
            assert list(csv.DictReader(stream)) == [{"id": "1"}]
        quarantined = [json.loads(line) for line in failsink_output.read_text(encoding="utf-8").splitlines()]
        assert len(quarantined) == 1
        assert quarantined[0]["id"] == 2
        assert quarantined[0]["unexpected"] == "preserve me"
        assert quarantined[0]["__diverted_from"] == "output"
        assert quarantined[0]["__diversion_reason"].startswith("effect-diversion:")

        accepted_outcome = recovered_factory.data_flow.get_token_outcome(accepted.token_id)
        diverted_outcome = recovered_factory.data_flow.get_token_outcome(diverted.token_id)
        assert accepted_outcome is not None and accepted_outcome.outcome is TerminalOutcome.SUCCESS
        assert diverted_outcome is not None
        assert diverted_outcome.outcome is TerminalOutcome.TRANSIENT
        assert diverted_outcome.path is TerminalPath.SINK_FALLBACK_TO_FAILSINK
        assert diverted_outcome.sink_name == "quarantine"

        effects = recovered_factory.execution.sink_effects.get_effects_for_run(run_id)
        primary_effect = next(effect for effect in effects if effect.role is SinkEffectRole.PRIMARY)
        failsink_effect = next(effect for effect in effects if effect.role is SinkEffectRole.FAILSINK)
        assert primary_effect.state is SinkEffectState.FINALIZED
        assert failsink_effect.state is SinkEffectState.FINALIZED
        assert failsink_effect.primary_effect_id == primary_effect.effect_id
        primary_state = next(
            state for state in recovered_factory.query.get_node_states_for_token(diverted.token_id) if state.node_id == primary_id
        )
        routing_events = recovered_factory.query.get_routing_events(primary_state.state_id)
        assert len(routing_events) == 1
        assert routing_events[0].edge_id == edge.edge_id
        assert routing_events[0].mode is RoutingMode.DIVERT
    finally:
        db.close()


@pytest.mark.parametrize("kind", ("csv", "json"))
def test_builtin_local_sink_response_loss_reconciles_without_duplicate_publication(
    tmp_path: Path,
    kind: Literal["csv", "json"],
) -> None:
    suffix = ".csv" if kind == "csv" else ".jsonl"
    output = tmp_path / f"response-loss{suffix}"
    db = LandscapeDB(f"sqlite:///{tmp_path / f'{kind}-response-loss.db'}")
    try:
        factory, run_id, source_id = _begin(db)
        sink_id = _register_sink(factory, run_id, name="output", plugin_name=kind)
        token = _tokens(factory, run_id=run_id, source_id=source_id, rows=[{"id": 1, "value": "once"}])[0]

        fired = False

        def lose_first_response(seam: SinkEffectExecutionSeam) -> None:
            nonlocal fired
            if seam is SinkEffectExecutionSeam.AFTER_EFFECT_BEFORE_RETURN and not fired:
                fired = True
                raise SinkEffectInjectedFault(seam)

        first = _csv_sink(output) if kind == "csv" else _json_sink(output)
        first.node_id = sink_id
        with pytest.raises(SinkEffectInjectedFault):
            _write(factory=factory, run_id=run_id, sink=first, tokens=[token], fault_hook=lose_first_response)

        recovered_factory = make_factory(db)
        recovered = _csv_sink(output) if kind == "csv" else _json_sink(output)
        recovered.node_id = sink_id
        artifact, counts = _write(factory=recovered_factory, run_id=run_id, sink=recovered, tokens=[token])

        assert artifact is not None
        assert counts == DiversionCounts()
        if kind == "csv":
            with output.open(newline="", encoding="utf-8") as stream:
                assert list(csv.DictReader(stream)) == [{"id": "1", "value": "once"}]
        else:
            assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == [{"id": 1, "value": "once"}]
        effects = recovered_factory.execution.sink_effects.get_effects_for_run(run_id)
        assert len(effects) == 1
        assert effects[0].state is SinkEffectState.FINALIZED
        assert effects[0].artifact_id == artifact.artifact_id
        outcome = recovered_factory.data_flow.get_token_outcome(token.token_id)
        assert outcome is not None
        assert outcome.outcome is TerminalOutcome.SUCCESS
        assert outcome.path is TerminalPath.DEFAULT_FLOW
    finally:
        db.close()


def test_json_effect_thaws_nested_pipeline_rows_before_serialization(tmp_path: Path) -> None:
    output = tmp_path / "nested.jsonl"
    db = LandscapeDB(f"sqlite:///{tmp_path / 'nested.db'}")
    try:
        factory, run_id, source_id = _begin(db)
        sink_id = _register_sink(factory, run_id, name="output", plugin_name="json")
        nested_row: dict[str, object] = {
            "id": 1,
            "payload": {
                "flags": [True, False],
                "items": [{"code": "A"}, {"code": "B"}],
            },
        }
        token = _tokens(factory, run_id=run_id, source_id=source_id, rows=[nested_row])[0]
        sink = _json_sink(output)
        sink.node_id = sink_id

        artifact, counts = _write(factory=factory, run_id=run_id, sink=sink, tokens=[token])

        assert artifact is not None
        assert counts == DiversionCounts()
        assert [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()] == [nested_row]
        outcome = factory.data_flow.get_token_outcome(token.token_id)
        assert outcome is not None
        assert outcome.outcome is TerminalOutcome.SUCCESS
        assert outcome.path is TerminalPath.DEFAULT_FLOW
    finally:
        db.close()


def test_all_diverted_csv_primary_is_virtual_no_publication_and_discards_deterministically(tmp_path: Path) -> None:
    output = tmp_path / "all-diverted.csv"
    db = LandscapeDB(f"sqlite:///{tmp_path / 'all-diverted.db'}")
    try:
        factory, run_id, source_id = _begin(db)
        sink_id = _register_sink(factory, run_id, name="output", plugin_name="csv")
        token = _tokens(factory, run_id=run_id, source_id=source_id, rows=[{"value": "café"}])[0]
        sink = _csv_sink(output, encoding="ascii")
        sink.node_id = sink_id

        artifact, counts = _write(factory=factory, run_id=run_id, sink=sink, tokens=[token])

        assert artifact is not None
        assert artifact.publication_performed is False
        assert artifact.publication_evidence_kind == "virtual"
        assert artifact.size_bytes == 0
        assert not output.exists()
        assert counts == DiversionCounts(discard_mode=1)
        outcome = factory.data_flow.get_token_outcome(token.token_id)
        assert outcome is not None
        assert outcome.outcome is TerminalOutcome.FAILURE
        assert outcome.path is TerminalPath.SINK_DISCARDED
        effects = factory.execution.sink_effects.get_effects_for_run(run_id)
        assert len(effects) == 1
        assert effects[0].state is SinkEffectState.FINALIZED
        assert effects[0].publication_performed is False
        assert effects[0].publication_evidence_kind == "virtual"
    finally:
        db.close()
