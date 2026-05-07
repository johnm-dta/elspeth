"""SchemaConfigModeContract -> dispatcher -> Landscape -> explain() round-trip."""

from __future__ import annotations

import json
from typing import Any, TypedDict

from elspeth.contracts import NodeStateFailed, NodeType
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
from elspeth.contracts.enums import NodeStateStatus
from elspeth.contracts.errors import ExecutionError, SchemaConfigModeViolation
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.core.landscape.lineage import explain
from elspeth.engine.executors.declaration_dispatch import (
    run_batch_flush_checks,
    run_post_emission_checks,
)
from elspeth.engine.executors.pass_through import PassThroughDeclarationContract
from elspeth.engine.executors.schema_config_mode import SchemaConfigModeContract
from tests.fixtures.landscape import make_recorder_with_run, register_test_node


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
        plugin_name="SchemaConfigModeTransform",
    )
    row = setup.factory.data_flow.create_row(
        run_id=run_id,
        source_node_id="source-0",
        row_index=0,
        data={"source": "v", "carry": "v"},
        row_id=row_id,
    )
    setup.factory.data_flow.create_token(row_id=row.row_id, token_id=token_id)
    return setup


def _record_failure(setup, *, token_id: str, node_id: str, run_id: str, error: ExecutionError) -> dict[str, Any]:
    state = setup.factory.execution.begin_node_state(
        token_id=token_id,
        node_id=node_id,
        run_id=run_id,
        step_index=1,
        input_data={"source": "v", "carry": "v"},
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


def _contract(
    fields: tuple[str, ...],
    *,
    mode: str = "OBSERVED",
    locked: bool = True,
) -> SchemaContract:
    return SchemaContract(
        mode=mode,  # type: ignore[arg-type]  # test helper uses closed-set literals
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
        locked=locked,
    )


def _row(
    fields: tuple[str, ...],
    *,
    mode: str = "OBSERVED",
    locked: bool = True,
) -> PipelineRow:
    return PipelineRow(dict.fromkeys(fields, "v"), _contract(fields, mode=mode, locked=locked))


def _schema_config(
    *,
    mode: str,
    fields: tuple[str, ...] | None = None,
) -> SchemaConfig:
    config: dict[str, object] = {"mode": mode}
    if fields is not None:
        config["fields"] = [{"name": name, "type": "str", "required": True, "nullable": False} for name in fields]
    return SchemaConfig.from_dict(config)


def _plugin(
    *,
    name: str = "SchemaConfigModeTransform",
    node_id: str = "schema-config-mode-node",
    output_schema_config: SchemaConfig | None = None,
    passes_through_input: bool = False,
    can_drop_rows: bool = False,
) -> Any:
    plugin = type("SchemaConfigModePlugin", (), {})()
    plugin.name = name
    plugin.node_id = node_id
    plugin._output_schema_config = output_schema_config
    plugin.passes_through_input = passes_through_input
    plugin.can_drop_rows = can_drop_rows
    plugin.declared_output_fields = frozenset()
    plugin.declared_input_fields = frozenset()
    plugin.is_batch_aware = False
    return plugin


class _SecretPayload(TypedDict):
    marker: str


class _SecretRoundTripViolation(DeclarationContractViolation):
    payload_schema = _SecretPayload


class _SecretRoundTripContract(DeclarationContract):
    name = "schema_config_mode_secret_roundtrip"
    payload_schema: type = _SecretPayload

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
            node_id="schema-config-secret-neg-node",
            run_id="schema-config-secret-neg-run",
            row_id="schema-config-secret-neg-row",
            token_id="schema-config-secret-neg-token",
            input_row=_row(("source",), mode="FIXED"),
            static_contract=frozenset({"source"}),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=(_row(("source",), mode="OBSERVED"),))
        return ExampleBundle(site=DispatchSite.POST_EMISSION, args=(inputs, outputs))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return cls.negative_example()


class TestSchemaConfigModeRoundTrip:
    def setup_method(self) -> None:
        self._snapshot = _snapshot_registry_for_tests()
        _clear_registry_for_tests()

    def teardown_method(self) -> None:
        _restore_registry_snapshot_for_tests(self._snapshot)

    def test_post_emission_violation_survives_landscape_round_trip(self) -> None:
        register_declaration_contract(SchemaConfigModeContract())

        run_id = "run-schema-config-mode"
        row_id = "row-schema-config-mode"
        token_id = "token-schema-config-mode"
        node_id = "node-schema-config-mode"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        plugin = _plugin(
            node_id=node_id,
            output_schema_config=_schema_config(mode="fixed", fields=("source",)),
        )
        inputs = PostEmissionInputs(
            plugin=plugin,
            node_id=node_id,
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            input_row=_row(("source",), mode="FIXED"),
            static_contract=frozenset({"source"}),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=(_row(("source",), mode="OBSERVED"),))

        try:
            run_post_emission_checks(inputs=inputs, outputs=outputs)
        except SchemaConfigModeViolation as violation:
            error = ExecutionError(
                exception=str(violation),
                exception_type=type(violation).__name__,
                phase="executor_post_process",
                context=violation.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected SchemaConfigModeViolation")

        context = _record_failure(setup, token_id=token_id, node_id=node_id, run_id=run_id, error=error)
        assert context["exception_type"] == "SchemaConfigModeViolation"
        assert context["contract_name"] == "schema_config_mode"
        assert context["payload"] == {
            "declared_mode": "fixed",
            "observed_mode": "observed",
            "declared_locked": True,
            "observed_locked": True,
        }

    def test_batch_flush_aggregate_round_trip_with_pass_through(self) -> None:
        register_declaration_contract(PassThroughDeclarationContract())
        register_declaration_contract(SchemaConfigModeContract())

        run_id = "run-schema-config-mode-aggregate"
        row_id = "row-schema-config-mode-aggregate"
        token_id = "token-schema-config-mode-aggregate"
        node_id = "node-schema-config-mode-aggregate"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        plugin = _plugin(
            node_id=node_id,
            output_schema_config=_schema_config(mode="fixed", fields=("source", "carry")),
            passes_through_input=True,
        )
        token_row = _row(("source", "carry"), mode="FIXED")
        inputs = BatchFlushInputs(
            plugin=plugin,
            node_id=node_id,
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            buffered_tokens=(token_row,),
            static_contract=frozenset({"source", "carry"}),
            effective_input_fields=frozenset({"source", "carry"}),
        )
        outputs = BatchFlushOutputs(emitted_rows=(_row(("source",), mode="OBSERVED"),))

        try:
            run_batch_flush_checks(inputs=inputs, outputs=outputs)
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
        assert child_types == {"PassThroughContractViolation", "SchemaConfigModeViolation"}

        schema_child = next(entry for entry in context["violations"] if entry["exception_type"] == "SchemaConfigModeViolation")
        assert schema_child["contract_name"] == "schema_config_mode"
        assert schema_child["payload"]["observed_mode"] == "observed"

        pass_through_child = next(entry for entry in context["violations"] if entry["exception_type"] == "PassThroughContractViolation")
        assert pass_through_child["divergence_set"] == ["carry"]

    def test_secret_like_payload_value_is_scrubbed_before_landscape_round_trip(self) -> None:
        register_declaration_contract(_SecretRoundTripContract())

        run_id = "run-schema-config-mode-secret"
        row_id = "row-schema-config-mode-secret"
        token_id = "token-schema-config-mode-secret"
        node_id = "node-schema-config-mode"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        inputs = PostEmissionInputs(
            plugin=_plugin(node_id=node_id, output_schema_config=_schema_config(mode="fixed", fields=("source",))),
            node_id=node_id,
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            input_row=_row(("source",), mode="FIXED"),
            static_contract=frozenset({"source"}),
            effective_input_fields=frozenset({"source"}),
        )
        outputs = PostEmissionOutputs(emitted_rows=(_row(("source",), mode="OBSERVED"),))

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
        assert context["contract_name"] == "schema_config_mode_secret_roundtrip"
        assert context["payload"]["marker"] == "<redacted-secret>"
