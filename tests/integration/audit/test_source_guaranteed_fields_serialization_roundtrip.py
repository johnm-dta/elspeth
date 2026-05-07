"""SourceGuaranteedFieldsContract -> dispatcher -> Landscape -> explain() round-trip."""

from __future__ import annotations

import json
from typing import Any, ClassVar, TypedDict

from elspeth.contracts import NodeStateFailed
from elspeth.contracts.declaration_contracts import (
    AggregateDeclarationContractViolation,
    BoundaryInputs,
    BoundaryOutputs,
    DeclarationContract,
    DeclarationContractViolation,
    DispatchSite,
    ExampleBundle,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.contracts.enums import NodeStateStatus
from elspeth.contracts.errors import ExecutionError, SourceGuaranteedFieldsViolation
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.results import SourceRow
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.core.landscape.lineage import explain
from elspeth.engine.executors.declaration_dispatch import run_boundary_checks
from elspeth.engine.executors.source_guaranteed_fields import SourceGuaranteedFieldsContract
from elspeth.plugins.infrastructure.base import BaseSource
from tests.fixtures.landscape import make_recorder_with_run


def _setup_landscape(*, run_id: str, row_id: str, token_id: str, node_id: str):
    setup = make_recorder_with_run(
        run_id=run_id,
        source_node_id=node_id,
        source_plugin_name="SourceGuaranteedFieldsSource",
    )
    row = setup.factory.data_flow.create_row(
        run_id=run_id,
        source_node_id=node_id,
        row_index=0,
        data={"customer_id": "v", "account_id": "v"},
        row_id=row_id,
    )
    setup.factory.data_flow.create_token(row_id=row.row_id, token_id=token_id)
    return setup


def _record_failure(setup, *, token_id: str, node_id: str, run_id: str, error: ExecutionError) -> dict[str, Any]:
    state = setup.factory.execution.begin_node_state(
        token_id=token_id,
        node_id=node_id,
        run_id=run_id,
        step_index=0,
        input_data={"customer_id": "v", "account_id": "v"},
    )
    setup.factory.execution.complete_node_state(
        state.state_id,
        NodeStateStatus.FAILED,
        duration_ms=0.0,
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


class _TestSourcePlugin(BaseSource):
    name = "SourceGuaranteedFieldsSource"
    output_schema = object

    def __init__(
        self,
        *,
        name: str,
        node_id: str,
        declared_guaranteed_fields: frozenset[str],
    ) -> None:
        super().__init__({})
        self.name = name
        self.node_id = node_id
        self.declared_guaranteed_fields = declared_guaranteed_fields

    def load(self, ctx: PluginContext):
        yield SourceRow.valid({"customer_id": "v"}, contract=_contract(("customer_id",)))

    def close(self) -> None:
        pass


def _plugin(
    *,
    name: str = "SourceGuaranteedFieldsSource",
    node_id: str = "source-guaranteed-fields-node",
    declared_guaranteed_fields: frozenset[str] = frozenset({"customer_id", "account_id"}),
) -> Any:
    return _TestSourcePlugin(
        name=name,
        node_id=node_id,
        declared_guaranteed_fields=declared_guaranteed_fields,
    )


class _SecondaryPayload(TypedDict):
    note: str


class _SecondaryBoundaryViolation(DeclarationContractViolation):
    payload_schema = _SecondaryPayload


class _SecretPayload(TypedDict):
    marker: str


class _SecretRoundTripViolation(DeclarationContractViolation):
    payload_schema = _SecretPayload


class _SecondaryBoundaryContract(DeclarationContract):
    name: ClassVar[str] = "secondary_source_boundary_test"
    payload_schema: ClassVar[type] = _SecondaryPayload

    def applies_to(self, plugin: Any) -> bool:
        return bool(plugin.declared_guaranteed_fields)

    @implements_dispatch_site("boundary_check")
    def boundary_check(self, inputs: BoundaryInputs, outputs: BoundaryOutputs) -> None:
        raise _SecondaryBoundaryViolation(
            plugin=inputs.plugin.name,
            node_id=inputs.node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
            payload={"note": "second"},
            message="secondary source boundary violation",
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        inputs = BoundaryInputs(
            plugin=_plugin(),
            node_id="source-secondary-neg-node",
            run_id="source-secondary-neg-run",
            row_id="source-secondary-neg-row",
            token_id="source-secondary-neg-token",
            static_contract=frozenset({"customer_id", "account_id"}),
            row_data={"account_id": "v"},
            row_contract=_contract(("account_id",)),
        )
        return ExampleBundle(site=DispatchSite.BOUNDARY, args=(inputs, BoundaryOutputs()))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        inputs = BoundaryInputs(
            plugin=_plugin(declared_guaranteed_fields=frozenset()),
            node_id="source-secondary-non-app-node",
            run_id="source-secondary-non-app-run",
            row_id="source-secondary-non-app-row",
            token_id="source-secondary-non-app-token",
            static_contract=frozenset(),
            row_data={"customer_id": "v"},
            row_contract=_contract(("customer_id",)),
        )
        return ExampleBundle(site=DispatchSite.BOUNDARY, args=(inputs, BoundaryOutputs()))


class _SecretBoundaryContract(DeclarationContract):
    name: ClassVar[str] = "source_guaranteed_fields_secret_roundtrip"
    payload_schema: ClassVar[type] = _SecretPayload

    def applies_to(self, plugin: Any) -> bool:
        return True

    @implements_dispatch_site("boundary_check")
    def boundary_check(self, inputs: BoundaryInputs, outputs: BoundaryOutputs) -> None:
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
        inputs = BoundaryInputs(
            plugin=_plugin(),
            node_id="source-secret-neg-node",
            run_id="source-secret-neg-run",
            row_id="source-secret-neg-row",
            token_id="source-secret-neg-token",
            static_contract=frozenset({"customer_id"}),
            row_data={"customer_id": "v"},
            row_contract=_contract(("customer_id",)),
        )
        return ExampleBundle(site=DispatchSite.BOUNDARY, args=(inputs, BoundaryOutputs()))

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return cls.negative_example()


class TestSourceGuaranteedFieldsRoundTrip:
    def setup_method(self) -> None:
        self._snapshot = _snapshot_registry_for_tests()
        _clear_registry_for_tests()

    def teardown_method(self) -> None:
        _restore_registry_snapshot_for_tests(self._snapshot)

    def test_boundary_violation_survives_landscape_round_trip(self) -> None:
        register_declaration_contract(SourceGuaranteedFieldsContract())

        run_id = "run-source-guaranteed-fields"
        row_id = "row-source-guaranteed-fields"
        token_id = "token-source-guaranteed-fields"
        node_id = "source-guaranteed-fields-node"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        try:
            run_boundary_checks(
                inputs=BoundaryInputs(
                    plugin=_plugin(node_id=node_id),
                    node_id=node_id,
                    run_id=run_id,
                    row_id=row_id,
                    token_id=token_id,
                    static_contract=frozenset({"customer_id", "account_id"}),
                    row_data={"customer_id": "v", "account_id": "v"},
                    row_contract=_contract(("account_id",)),
                ),
                outputs=BoundaryOutputs(),
            )
        except SourceGuaranteedFieldsViolation as violation:
            error = ExecutionError(
                exception=str(violation),
                exception_type=type(violation).__name__,
                phase="source_boundary_check",
                context=violation.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected SourceGuaranteedFieldsViolation")

        context = _record_failure(setup, token_id=token_id, node_id=node_id, run_id=run_id, error=error)
        assert context["exception_type"] == "SourceGuaranteedFieldsViolation"
        assert context["contract_name"] == "source_guaranteed_fields"
        assert context["plugin"] == "SourceGuaranteedFieldsSource"
        assert context["node_id"] == node_id
        assert context["run_id"] == run_id
        assert context["row_id"] == row_id
        assert context["token_id"] == token_id
        assert context["payload"]["declared"] == ["account_id", "customer_id"]
        assert context["payload"]["runtime_observed"] == ["account_id"]
        assert context["payload"]["missing"] == ["customer_id"]

    def test_aggregate_round_trip_with_second_boundary_contract(self) -> None:
        register_declaration_contract(SourceGuaranteedFieldsContract())
        register_declaration_contract(_SecondaryBoundaryContract())

        run_id = "run-source-guaranteed-fields-aggregate"
        row_id = "row-source-guaranteed-fields-aggregate"
        token_id = "token-source-guaranteed-fields-aggregate"
        node_id = "source-guaranteed-fields-node"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        try:
            run_boundary_checks(
                inputs=BoundaryInputs(
                    plugin=_plugin(node_id=node_id),
                    node_id=node_id,
                    run_id=run_id,
                    row_id=row_id,
                    token_id=token_id,
                    static_contract=frozenset({"customer_id", "account_id"}),
                    row_data={"account_id": "v"},
                    row_contract=_contract(("account_id",)),
                ),
                outputs=BoundaryOutputs(),
            )
        except AggregateDeclarationContractViolation as aggregate:
            error = ExecutionError(
                exception=str(aggregate),
                exception_type=type(aggregate).__name__,
                phase="source_boundary_check",
                context=aggregate.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected AggregateDeclarationContractViolation")

        context = _record_failure(setup, token_id=token_id, node_id=node_id, run_id=run_id, error=error)
        assert context["exception_type"] == "AggregateDeclarationContractViolation"
        assert context["is_aggregate"] is True
        child_types = {entry["exception_type"] for entry in context["violations"]}
        assert child_types == {"SourceGuaranteedFieldsViolation", "_SecondaryBoundaryViolation"}

    def test_secret_like_payload_value_is_scrubbed_before_landscape_round_trip(self) -> None:
        register_declaration_contract(_SecretBoundaryContract())

        run_id = "run-source-guaranteed-fields-secret"
        row_id = "row-source-guaranteed-fields-secret"
        token_id = "token-source-guaranteed-fields-secret"
        node_id = "source-guaranteed-fields-node"
        setup = _setup_landscape(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id)

        try:
            run_boundary_checks(
                inputs=BoundaryInputs(
                    plugin=_plugin(node_id=node_id),
                    node_id=node_id,
                    run_id=run_id,
                    row_id=row_id,
                    token_id=token_id,
                    static_contract=frozenset({"customer_id"}),
                    row_data={"customer_id": "v"},
                    row_contract=_contract(("customer_id",)),
                ),
                outputs=BoundaryOutputs(),
            )
        except _SecretRoundTripViolation as violation:
            error = ExecutionError(
                exception=str(violation),
                exception_type=type(violation).__name__,
                phase="source_boundary_check",
                context=violation.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected _SecretRoundTripViolation")

        context = _record_failure(setup, token_id=token_id, node_id=node_id, run_id=run_id, error=error)
        assert "sk-abcdef" not in json.dumps(context)
        assert context["contract_name"] == "source_guaranteed_fields_secret_roundtrip"
        assert context["payload"]["marker"] == "<redacted-secret>"
