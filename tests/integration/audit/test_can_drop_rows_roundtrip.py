"""CanDropRowsContract -> dispatcher -> Landscape -> queryable terminal state."""

from __future__ import annotations

import json
from typing import Any, TypedDict

from sqlalchemy import text

from elspeth.contracts import Determinism, NodeStateFailed, NodeType, PluginSchema, TransformResult
from elspeth.contracts.declaration_contracts import (
    AggregateDeclarationContractViolation,
    BatchFlushInputs,
    BatchFlushOutputs,
    DeclarationContract,
    DeclarationContractViolation,
    DispatchSite,
    ExampleBundle,
    PostEmissionInputs,
    PostEmissionOutputs,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.contracts.enums import NodeStateStatus, TerminalOutcome, TerminalPath
from elspeth.contracts.errors import ExecutionError, UnexpectedEmptyEmissionViolation
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.core.config import SourceSettings
from elspeth.core.dag import ExecutionGraph
from elspeth.core.landscape.lineage import explain
from elspeth.engine.executors.can_drop_rows import CanDropRowsContract
from elspeth.engine.executors.declaration_dispatch import (
    run_batch_flush_checks,
    run_post_emission_checks,
)
from elspeth.engine.executors.pass_through import PassThroughDeclarationContract
from elspeth.engine.orchestrator import Orchestrator, PipelineConfig
from elspeth.plugins.infrastructure.base import BaseTransform
from tests.fixtures.base_classes import as_sink, as_source, as_transform
from tests.fixtures.factories import wire_transforms
from tests.fixtures.landscape import make_landscape_db, make_recorder_with_run, register_test_node
from tests.fixtures.plugins import CollectSink, ListSource
from tests.fixtures.stores import MockPayloadStore


def _setup_landscape(*, run_id: str, row_id: str, token_id: str, node_id: str):
    setup = make_recorder_with_run(
        run_id=run_id,
        source_node_id="source-0",
        source_plugin_name="test-source",
    )
    register_test_node(
        setup.factory.data_flow,
        run_id=run_id,
        node_id=node_id,
        node_type=NodeType.TRANSFORM,
        plugin_name="CanDropRowsTransform",
    )
    row = setup.factory.data_flow.create_row(
        run_id=run_id,
        source_node_id="source-0",
        row_index=0,
        data={"source": "v"},
        row_id=row_id,
        source_row_index=0,
        ingest_sequence=0,
    )
    setup.factory.data_flow.create_token(row_id=row.row_id, token_id=token_id)
    return setup


def _record_failure(setup, *, token_id: str, node_id: str, run_id: str, error: ExecutionError) -> dict[str, Any]:
    state = setup.factory.execution.begin_node_state(
        token_id=token_id,
        node_id=node_id,
        run_id=run_id,
        step_index=1,
        input_data={"source": "v"},
    )
    setup.factory.execution.complete_node_state(
        state.state_id,
        NodeStateStatus.FAILED,
        duration_ms=1.0,
        error=error,
    )

    lineage = explain(
        query=setup.factory.query,
        data_flow=setup.factory.data_flow,
        run_id=run_id,
        token_id=token_id,
    )
    assert lineage is not None
    failed_states = [ns for ns in lineage.node_states if isinstance(ns, NodeStateFailed)]
    assert len(failed_states) == 1
    assert failed_states[0].error_json is not None
    return json.loads(failed_states[0].error_json)["context"]


def _contract(fields: tuple[str, ...]) -> SchemaContract:
    return SchemaContract(
        mode="OBSERVED",
        fields=tuple(
            FieldContract(
                normalized_name=name,
                original_name=name,
                python_type=str,
                required=True,
                source="inferred",
                nullable=False,
            )
            for name in fields
        ),
        locked=True,
    )


def _row(fields: tuple[str, ...]) -> PipelineRow:
    return PipelineRow(dict.fromkeys(fields, "v"), _contract(fields))


def _plugin(
    *,
    name: str = "CanDropRowsTransform",
    node_id: str = "can-drop-rows-node",
    passes_through_input: bool = True,
    can_drop_rows: bool = False,
    is_batch_aware: bool = False,
) -> Any:
    plugin = type("CanDropRowsPlugin", (), {})()
    plugin.name = name
    plugin.node_id = node_id
    plugin.passes_through_input = passes_through_input
    plugin.can_drop_rows = can_drop_rows
    plugin.declared_output_fields = frozenset()
    plugin.declared_input_fields = frozenset()
    plugin.is_batch_aware = is_batch_aware
    plugin._output_schema_config = None
    return plugin


def _build_production_graph(config: PipelineConfig) -> ExecutionGraph:
    primary_name = next(iter(config.sources))
    primary = config.sources[primary_name]
    return ExecutionGraph.from_plugin_instances(
        sources={primary_name: primary},
        source_settings_map={primary_name: SourceSettings(plugin=primary.name, on_success="source_out", options={})},
        transforms=wire_transforms(
            [as_transform(transform) for transform in config.transforms],
            source_connection="source_out",
            final_sink="default",
        ),
        sinks=config.sinks,
        aggregations={},
        gates=[],
        coalesce_settings=None,
    )


class _TestSchema(PluginSchema):
    pass


class _SecretPayload(TypedDict):
    marker: str


class _SecretRoundTripViolation(DeclarationContractViolation):
    payload_schema = _SecretPayload


class _SecretRoundTripContract(DeclarationContract):
    name = "can_drop_rows_secret_roundtrip"
    payload_schema: type = _SecretPayload
    violation_class: type[_SecretRoundTripViolation] = _SecretRoundTripViolation

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(self, inputs: PostEmissionInputs, outputs: PostEmissionOutputs) -> None:
        raise _SecretRoundTripViolation(
            plugin=inputs.plugin.name,
            node_id=inputs.node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
            payload={"marker": "sk-abcdef1234567890abcdef1234567890"},
            message="secret round-trip test",
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        inputs = PostEmissionInputs(
            plugin=_plugin(),
            node_id="can-drop-secret-neg-node",
            run_id="can-drop-secret-neg-run",
            row_id="can-drop-secret-neg-row",
            token_id="can-drop-secret-neg-token",
            input_row=_row(("source",)),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=(_row(("source",)),))
        return ExampleBundle(site=DispatchSite.POST_EMISSION, args=(inputs, outputs))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return cls.negative_example()


class _HonestFilterTransform(BaseTransform):
    name = "honest_filter_transform"
    determinism = Determinism.DETERMINISTIC
    input_schema: type[PluginSchema] = _TestSchema
    output_schema: type[PluginSchema] = _TestSchema
    passes_through_input = True
    can_drop_rows = True

    def __init__(self) -> None:
        super().__init__({"schema": {"mode": "observed"}})

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        return TransformResult.success_empty(success_reason={"action": "filtered"})


class TestCanDropRowsRoundTrip:
    def setup_method(self) -> None:
        self._snapshot = _snapshot_registry_for_tests()
        _clear_registry_for_tests()

    def teardown_method(self) -> None:
        _restore_registry_snapshot_for_tests(self._snapshot)

    def test_post_emission_violation_survives_landscape_round_trip(self) -> None:
        register_declaration_contract(CanDropRowsContract())

        run_id = "run-can-drop-rows"
        row_id = "row-can-drop-rows"
        token_id = "token-can-drop-rows"
        node_id = "node-can-drop-rows"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        plugin = _plugin(node_id=node_id, passes_through_input=True, can_drop_rows=False)
        inputs = PostEmissionInputs(
            plugin=plugin,
            node_id=node_id,
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            input_row=_row(("source",)),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"source"}),
        )

        try:
            run_post_emission_checks(inputs=inputs, outputs=PostEmissionOutputs(emitted_rows=()))
        except UnexpectedEmptyEmissionViolation as violation:
            error = ExecutionError(
                exception=str(violation),
                exception_type=type(violation).__name__,
                phase="executor_post_process",
                context=violation.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected UnexpectedEmptyEmissionViolation")

        context = _record_failure(setup, token_id=token_id, node_id=node_id, run_id=run_id, error=error)
        assert context["exception_type"] == "UnexpectedEmptyEmissionViolation"
        assert context["contract_name"] == "can_drop_rows"
        assert context["plugin"] == "CanDropRowsTransform"
        assert context["payload"] == {
            "passes_through_input": True,
            "can_drop_rows": False,
            "emitted_count": 0,
        }

    def test_batch_flush_violation_survives_landscape_round_trip(self) -> None:
        register_declaration_contract(CanDropRowsContract())

        run_id = "run-can-drop-rows-batch"
        row_id = "row-can-drop-rows-batch"
        token_id = "token-can-drop-rows-batch"
        node_id = "node-can-drop-rows-batch"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        plugin = _plugin(node_id=node_id, passes_through_input=True, can_drop_rows=False, is_batch_aware=True)
        token_row = _row(("source",))
        inputs = BatchFlushInputs(
            plugin=plugin,
            node_id=node_id,
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            buffered_tokens=(token_row,),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"source"}),
        )

        try:
            run_batch_flush_checks(inputs=inputs, outputs=BatchFlushOutputs(emitted_rows=()))
        except UnexpectedEmptyEmissionViolation as violation:
            error = ExecutionError(
                exception=str(violation),
                exception_type=type(violation).__name__,
                phase="batch_flush_dispatch",
                context=violation.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected UnexpectedEmptyEmissionViolation on batch-flush path")

        context = _record_failure(setup, token_id=token_id, node_id=node_id, run_id=run_id, error=error)
        assert context["exception_type"] == "UnexpectedEmptyEmissionViolation"
        assert context["contract_name"] == "can_drop_rows"
        assert context["payload"]["emitted_count"] == 0

    def test_secret_like_payload_value_is_scrubbed_before_landscape_round_trip(self) -> None:
        register_declaration_contract(_SecretRoundTripContract())

        run_id = "run-can-drop-rows-secret"
        row_id = "row-can-drop-rows-secret"
        token_id = "token-can-drop-rows-secret"
        node_id = "node-can-drop-rows"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        inputs = PostEmissionInputs(
            plugin=_plugin(node_id=node_id),
            node_id=node_id,
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            input_row=_row(("source",)),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=(_row(("source",)),))

        try:
            run_post_emission_checks(inputs=inputs, outputs=outputs)
        except _SecretRoundTripViolation as violation:
            error = ExecutionError(
                exception=str(violation),
                exception_type=type(violation).__name__,
                phase="executor_post_process",
                context=violation.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected _SecretRoundTripViolation")

        context = _record_failure(setup, token_id=token_id, node_id=node_id, run_id=run_id, error=error)
        assert "sk-abcdef" not in json.dumps(context)
        assert context["contract_name"] == "can_drop_rows_secret_roundtrip"
        assert context["payload"]["marker"] == "<redacted-secret>"

    def test_aggregate_round_trip_with_pass_through(self) -> None:
        register_declaration_contract(PassThroughDeclarationContract())
        register_declaration_contract(CanDropRowsContract())

        run_id = "run-can-drop-rows-aggregate"
        row_id = "row-can-drop-rows-aggregate"
        token_id = "token-can-drop-rows-aggregate"
        node_id = "node-can-drop-rows-aggregate"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        plugin = _plugin(node_id=node_id, passes_through_input=True, can_drop_rows=False)
        inputs = PostEmissionInputs(
            plugin=plugin,
            node_id=node_id,
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            input_row=_row(("source", "carry")),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"source", "carry"}),
        )

        try:
            run_post_emission_checks(inputs=inputs, outputs=PostEmissionOutputs(emitted_rows=()))
        except AggregateDeclarationContractViolation as aggregate:
            error = ExecutionError(
                exception=str(aggregate),
                exception_type=type(aggregate).__name__,
                phase="executor_post_process",
                context=aggregate.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected AggregateDeclarationContractViolation")

        context = _record_failure(setup, token_id=token_id, node_id=node_id, run_id=run_id, error=error)
        assert context["exception_type"] == "AggregateDeclarationContractViolation"
        assert context["is_aggregate"] is True
        child_types = {entry["exception_type"] for entry in context["violations"]}
        assert child_types == {"UnexpectedEmptyEmissionViolation", "PassThroughContractViolation"}

        empty_child = next(entry for entry in context["violations"] if entry["exception_type"] == "UnexpectedEmptyEmissionViolation")
        assert empty_child["contract_name"] == "can_drop_rows"
        assert empty_child["payload"]["emitted_count"] == 0

        pass_through_child = next(entry for entry in context["violations"] if entry["exception_type"] == "PassThroughContractViolation")
        assert pass_through_child["divergence_set"] == ["carry", "source"]


def test_legitimate_filter_records_queryable_dropped_by_filter_terminal_state() -> None:
    db = make_landscape_db()
    payload_store = MockPayloadStore()
    source = ListSource([{"value": 1}])
    transform = _HonestFilterTransform()
    sink = CollectSink("default")

    config = PipelineConfig(
        sources={"primary": as_source(source)},
        transforms=[as_transform(transform)],
        sinks={"default": as_sink(sink)},
    )

    orchestrator = Orchestrator(db)
    run = orchestrator.run(config, graph=_build_production_graph(config), payload_store=payload_store)

    assert sink.results == []

    with db.connection() as conn:
        outcomes = conn.execute(
            text("""
                SELECT o.outcome, o.path, o.completed, o.error_hash
                FROM token_outcomes o
                JOIN tokens t ON t.token_id = o.token_id
                JOIN rows r ON r.row_id = t.row_id
                WHERE r.run_id = :run_id
                ORDER BY o.recorded_at
            """),
            {"run_id": run.run_id},
        ).fetchall()

    assert len(outcomes) == 1
    assert outcomes[0].outcome == TerminalOutcome.SUCCESS.value
    assert outcomes[0].path == TerminalPath.FILTER_DROPPED.value
    assert outcomes[0].completed == 1
    assert outcomes[0].error_hash is None
