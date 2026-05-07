"""DeclaredRequiredFieldsContract -> dispatcher -> Landscape -> explain() round-trip."""

from __future__ import annotations

import json
from typing import Any, TypedDict

from elspeth.contracts import NodeStateFailed, NodeType
from elspeth.contracts.declaration_contracts import (
    AggregateDeclarationContractViolation,
    DeclarationContract,
    DeclarationContractViolation,
    DispatchSite,
    ExampleBundle,
    PreEmissionInputs,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.contracts.enums import NodeStateStatus
from elspeth.contracts.errors import (
    DeclaredRequiredInputFieldsViolation,
    ExecutionError,
)
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.core.landscape.lineage import explain
from elspeth.engine.executors.declaration_dispatch import run_pre_emission_checks
from elspeth.engine.executors.declared_required_fields import DeclaredRequiredFieldsContract
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
        plugin_name="DeclaredRequiredFieldsTransform",
    )
    row = setup.factory.data_flow.create_row(
        run_id=run_id,
        source_node_id="source-0",
        row_index=0,
        data={"account_id": "v"},
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
        input_data={"account_id": "v"},
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
    name: str = "DeclaredRequiredFieldsTransform",
    node_id: str = "declared-required-fields-node",
    declared_input_fields: frozenset[str] = frozenset({"customer_id", "account_id"}),
    passes_through_input: bool = False,
    can_drop_rows: bool = False,
) -> Any:
    plugin = type("DeclaredRequiredFieldsPlugin", (), {})()
    plugin.name = name
    plugin.node_id = node_id
    plugin.passes_through_input = passes_through_input
    plugin.can_drop_rows = can_drop_rows
    plugin.declared_output_fields = frozenset()
    plugin.declared_input_fields = declared_input_fields
    plugin._output_schema_config = None
    plugin.is_batch_aware = False
    return plugin


class _SecondaryPayload(TypedDict):
    note: str


class _SecondaryPreEmissionViolation(DeclarationContractViolation):
    payload_schema = _SecondaryPayload


class _SecretPayload(TypedDict):
    marker: str


class _SecretRoundTripViolation(DeclarationContractViolation):
    payload_schema = _SecretPayload


class _SecondaryPreEmissionContract(DeclarationContract):
    name = "secondary_declared_required_test"
    payload_schema: type = _SecondaryPayload

    def applies_to(self, plugin: Any) -> bool:
        return bool(plugin.declared_input_fields)

    @implements_dispatch_site("pre_emission_check")
    def pre_emission_check(self, inputs: PreEmissionInputs) -> None:
        raise _SecondaryPreEmissionViolation(
            plugin=inputs.plugin.name,
            node_id=inputs.node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
            payload={"note": "second"},
            message="secondary pre-emission violation",
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        inputs = PreEmissionInputs(
            plugin=_plugin(),
            node_id="declared-required-secondary-neg-node",
            run_id="declared-required-secondary-neg-run",
            row_id="declared-required-secondary-neg-row",
            token_id="declared-required-secondary-neg-token",
            input_row=_row(("account_id",)),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"account_id"}),
        )
        return ExampleBundle(site=DispatchSite.PRE_EMISSION, args=(inputs,))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        inputs = PreEmissionInputs(
            plugin=_plugin(declared_input_fields=frozenset()),
            node_id="declared-required-secondary-non-app-node",
            run_id="declared-required-secondary-non-app-run",
            row_id="declared-required-secondary-non-app-row",
            token_id="declared-required-secondary-non-app-token",
            input_row=_row(("account_id",)),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"account_id"}),
        )
        return ExampleBundle(site=DispatchSite.PRE_EMISSION, args=(inputs,))


class _SecretPreEmissionContract(DeclarationContract):
    name = "declared_required_fields_secret_roundtrip"
    payload_schema: type = _SecretPayload

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("pre_emission_check")
    def pre_emission_check(self, inputs: PreEmissionInputs) -> None:
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
        inputs = PreEmissionInputs(
            plugin=_plugin(),
            node_id="declared-required-secret-neg-node",
            run_id="declared-required-secret-neg-run",
            row_id="declared-required-secret-neg-row",
            token_id="declared-required-secret-neg-token",
            input_row=_row(("account_id",)),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"account_id"}),
        )
        return ExampleBundle(site=DispatchSite.PRE_EMISSION, args=(inputs,))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return cls.negative_example()


class TestDeclaredRequiredFieldsRoundTrip:
    def setup_method(self) -> None:
        self._snapshot = _snapshot_registry_for_tests()
        _clear_registry_for_tests()

    def teardown_method(self) -> None:
        _restore_registry_snapshot_for_tests(self._snapshot)

    def test_pre_emission_violation_survives_landscape_round_trip(self) -> None:
        register_declaration_contract(DeclaredRequiredFieldsContract())

        run_id = "run-declared-required-fields"
        row_id = "row-declared-required-fields"
        token_id = "token-declared-required-fields"
        node_id = "node-declared-required-fields"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        plugin = _plugin(node_id=node_id)
        inputs = PreEmissionInputs(
            plugin=plugin,
            node_id=node_id,
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            input_row=_row(("account_id",)),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"account_id"}),
        )

        try:
            run_pre_emission_checks(inputs=inputs)
        except DeclaredRequiredInputFieldsViolation as violation:
            error = ExecutionError(
                exception=str(violation),
                exception_type=type(violation).__name__,
                phase="executor_pre_process",
                context=violation.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected DeclaredRequiredInputFieldsViolation")

        context = _record_failure(setup, token_id=token_id, node_id=node_id, run_id=run_id, error=error)
        assert context["exception_type"] == "DeclaredRequiredInputFieldsViolation"
        assert context["contract_name"] == "declared_required_fields"
        assert context["plugin"] == "DeclaredRequiredFieldsTransform"
        assert context["node_id"] == node_id
        assert context["run_id"] == run_id
        assert context["row_id"] == row_id
        assert context["token_id"] == token_id
        assert context["payload"]["declared"] == ["account_id", "customer_id"]
        assert context["payload"]["effective_input_fields"] == ["account_id"]
        assert context["payload"]["missing"] == ["customer_id"]

    def test_aggregate_round_trip_with_second_pre_emission_contract(self) -> None:
        register_declaration_contract(DeclaredRequiredFieldsContract())
        register_declaration_contract(_SecondaryPreEmissionContract())

        run_id = "run-declared-required-fields-aggregate"
        row_id = "row-declared-required-fields-aggregate"
        token_id = "token-declared-required-fields-aggregate"
        node_id = "node-declared-required-fields-aggregate"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        plugin = _plugin(node_id=node_id)
        inputs = PreEmissionInputs(
            plugin=plugin,
            node_id=node_id,
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            input_row=_row(("account_id",)),
            static_contract=frozenset(),
            effective_input_fields=frozenset({"account_id"}),
        )

        try:
            run_pre_emission_checks(inputs=inputs)
        except AggregateDeclarationContractViolation as aggregate:
            error = ExecutionError(
                exception=str(aggregate),
                exception_type=type(aggregate).__name__,
                phase="executor_pre_process",
                context=aggregate.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected AggregateDeclarationContractViolation")

        context = _record_failure(setup, token_id=token_id, node_id=node_id, run_id=run_id, error=error)
        assert context["exception_type"] == "AggregateDeclarationContractViolation"
        assert context["is_aggregate"] is True
        child_types = {entry["exception_type"] for entry in context["violations"]}
        assert child_types == {"DeclaredRequiredInputFieldsViolation", "_SecondaryPreEmissionViolation"}

        declared_child = next(entry for entry in context["violations"] if entry["exception_type"] == "DeclaredRequiredInputFieldsViolation")
        assert declared_child["contract_name"] == "declared_required_fields"
        assert declared_child["payload"]["missing"] == ["customer_id"]

    def test_secret_like_payload_value_is_scrubbed_before_landscape_round_trip(self) -> None:
        register_declaration_contract(_SecretPreEmissionContract())

        run_id = "run-declared-required-fields-secret"
        row_id = "row-declared-required-fields-secret"
        token_id = "token-declared-required-fields-secret"
        node_id = "node-declared-required-fields"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        try:
            run_pre_emission_checks(
                inputs=PreEmissionInputs(
                    plugin=_plugin(node_id=node_id),
                    node_id=node_id,
                    run_id=run_id,
                    row_id=row_id,
                    token_id=token_id,
                    input_row=_row(("account_id",)),
                    static_contract=frozenset(),
                    effective_input_fields=frozenset({"account_id"}),
                )
            )
        except _SecretRoundTripViolation as violation:
            error = ExecutionError(
                exception=str(violation),
                exception_type=type(violation).__name__,
                phase="executor_pre_process",
                context=violation.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected _SecretRoundTripViolation")

        context = _record_failure(setup, token_id=token_id, node_id=node_id, run_id=run_id, error=error)
        assert "sk-abcdef" not in json.dumps(context)
        assert context["contract_name"] == "declared_required_fields_secret_roundtrip"
        assert context["payload"]["marker"] == "<redacted-secret>"
