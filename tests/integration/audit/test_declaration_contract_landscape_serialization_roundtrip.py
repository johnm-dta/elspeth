"""DeclarationContractViolation -> dispatcher -> Landscape -> explain() round-trip.

CLAUDE.md attributability mandate: for any output, explain(recorder, run_id,
token_id) must prove complete lineage. These tests exercise declaration
violations end-to-end through the dispatcher before they are recorded to the
Landscape, so contract-name attribution is proven on the same path production
uses at runtime.
"""

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
    PostEmissionInputs,
    PostEmissionOutputs,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.contracts.enums import NodeStateStatus
from elspeth.contracts.errors import ExecutionError
from elspeth.core.landscape.lineage import explain
from elspeth.engine.executors.declaration_dispatch import run_post_emission_checks
from tests.fixtures.landscape import make_recorder_with_run, register_test_node


class _RoundTripPayload(TypedDict):
    field_name: str
    expected_type: str
    observed_type: str
    nested: dict[str, Any]


class _RoundTripViolation(DeclarationContractViolation):
    payload_schema = _RoundTripPayload


class _SecretPayload(TypedDict):
    api_key: str


class _SecretViolation(DeclarationContractViolation):
    payload_schema = _SecretPayload


class _SecretMessagePayload(TypedDict):
    note: str


class _SecretMessageViolation(DeclarationContractViolation):
    payload_schema = _SecretMessagePayload


class _AggregateChildPayload(TypedDict):
    reason: str


class _AggregateChildViolationA(DeclarationContractViolation):
    payload_schema = _AggregateChildPayload


class _AggregateChildViolationB(DeclarationContractViolation):
    payload_schema = _AggregateChildPayload


def _setup_landscape(*, run_id: str, row_id: str, token_id: str, node_id: str):
    """Create a landscape with one run, source node, row, token, and transform node."""
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
        plugin_name="FakeTransform",
    )
    row = setup.factory.data_flow.create_row(
        run_id=run_id,
        source_node_id="source-0",
        row_index=0,
        data={"amount": "not_an_int"},
        row_id=row_id,
    )
    setup.factory.data_flow.create_token(row_id=row.row_id, token_id=token_id)
    return setup


def _plugin():
    plugin = type("RoundTripPlugin", (), {})()
    plugin.name = "FakeTransform"
    return plugin


def _post_emission_inputs(*, run_id: str, row_id: str, token_id: str, node_id: str) -> PostEmissionInputs:
    return PostEmissionInputs(
        plugin=_plugin(),
        node_id=node_id,
        run_id=run_id,
        row_id=row_id,
        token_id=token_id,
        input_row={"amount": "not_an_int"},
        static_contract=frozenset(),
        effective_input_fields=frozenset(),
    )


def _post_emission_outputs() -> PostEmissionOutputs:
    return PostEmissionOutputs(emitted_rows=({"amount": "not_an_int"},))


def _record_failure(
    setup,
    *,
    token_id: str,
    node_id: str,
    run_id: str,
    input_data: dict[str, Any],
    error: ExecutionError,
) -> dict[str, Any]:
    state = setup.factory.execution.begin_node_state(
        token_id=token_id,
        node_id=node_id,
        run_id=run_id,
        step_index=1,
        input_data=input_data,
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


class _RoundTripContract(DeclarationContract):
    name = "test_roundtrip"
    payload_schema: type = _RoundTripPayload

    def applies_to(self, plugin: object) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(self, inputs: PostEmissionInputs, outputs: PostEmissionOutputs) -> None:
        raise _RoundTripViolation(
            plugin=inputs.plugin.name,
            node_id=inputs.node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
            payload={
                "field_name": "amount",
                "expected_type": "int",
                "observed_type": "str",
                "nested": {"inner": [1, 2, 3]},
            },
            message="round-trip test violation",
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return ExampleBundle(
            site=DispatchSite.POST_EMISSION,
            args=(
                _post_emission_inputs(run_id="neg-run", row_id="neg-row", token_id="neg-token", node_id="neg-node"),
                _post_emission_outputs(),
            ),
        )

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return cls.negative_example()


class _SecretContract(DeclarationContract):
    name = "secret_test"
    payload_schema: type = _SecretPayload

    def applies_to(self, plugin: object) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(self, inputs: PostEmissionInputs, outputs: PostEmissionOutputs) -> None:
        raise _SecretViolation(
            plugin=inputs.plugin.name,
            node_id=inputs.node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
            payload={"api_key": "sk-abcdef1234567890abcdef1234567890"},
            message="secret test",
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return ExampleBundle(
            site=DispatchSite.POST_EMISSION,
            args=(
                _post_emission_inputs(
                    run_id="secret-neg-run", row_id="secret-neg-row", token_id="secret-neg-token", node_id="secret-neg-node"
                ),
                _post_emission_outputs(),
            ),
        )

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return cls.negative_example()


class _SecretMessageContract(DeclarationContract):
    name = "secret_message_test"
    payload_schema: type = _SecretMessagePayload

    def applies_to(self, plugin: object) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(self, inputs: PostEmissionInputs, outputs: PostEmissionOutputs) -> None:
        raise _SecretMessageViolation(
            plugin=inputs.plugin.name,
            node_id=inputs.node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
            payload={"note": "safe"},
            message="secret sk-abcdef1234567890abcdef1234567890 leaked",
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return ExampleBundle(
            site=DispatchSite.POST_EMISSION,
            args=(
                _post_emission_inputs(
                    run_id="secret-msg-neg-run",
                    row_id="secret-msg-neg-row",
                    token_id="secret-msg-neg-token",
                    node_id="secret-msg-neg-node",
                ),
                _post_emission_outputs(),
            ),
        )

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return cls.negative_example()


class _AggregateChildContractA(DeclarationContract):
    name = "contract_a"
    payload_schema: type = _AggregateChildPayload

    def applies_to(self, plugin: object) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(self, inputs: PostEmissionInputs, outputs: PostEmissionOutputs) -> None:
        raise _AggregateChildViolationA(
            plugin=inputs.plugin.name,
            node_id=inputs.node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
            payload={"reason": "child-a-triggered"},
            message="first child violation",
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return ExampleBundle(
            site=DispatchSite.POST_EMISSION,
            args=(
                _post_emission_inputs(run_id="agg-a-neg-run", row_id="agg-a-neg-row", token_id="agg-a-neg-token", node_id="agg-a-neg-node"),
                _post_emission_outputs(),
            ),
        )

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return cls.negative_example()


class _AggregateChildContractB(DeclarationContract):
    name = "contract_b"
    payload_schema: type = _AggregateChildPayload

    def applies_to(self, plugin: object) -> bool:
        return True

    @implements_dispatch_site("post_emission_check")
    def post_emission_check(self, inputs: PostEmissionInputs, outputs: PostEmissionOutputs) -> None:
        raise _AggregateChildViolationB(
            plugin=inputs.plugin.name,
            node_id=inputs.node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
            payload={"reason": "child-b-triggered"},
            message="second child violation",
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        return ExampleBundle(
            site=DispatchSite.POST_EMISSION,
            args=(
                _post_emission_inputs(run_id="agg-b-neg-run", row_id="agg-b-neg-row", token_id="agg-b-neg-token", node_id="agg-b-neg-node"),
                _post_emission_outputs(),
            ),
        )

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return cls.negative_example()


class TestDeclarationContractViolationRoundTrip:
    """Verify declaration violations serialize losslessly through the Landscape."""

    def setup_method(self) -> None:
        self._snapshot = _snapshot_registry_for_tests()
        _clear_registry_for_tests()

    def teardown_method(self) -> None:
        _restore_registry_snapshot_for_tests(self._snapshot)

    def test_violation_payload_survives_landscape_round_trip(self) -> None:
        run_id = "run-roundtrip"
        row_id = "row-roundtrip"
        token_id = "tok-roundtrip"
        node_id = "n-roundtrip"

        register_declaration_contract(_RoundTripContract())

        setup = _setup_landscape(
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            node_id=node_id,
        )

        try:
            run_post_emission_checks(
                inputs=_post_emission_inputs(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id),
                outputs=_post_emission_outputs(),
            )
        except _RoundTripViolation as violation:
            error = ExecutionError(
                exception=str(violation),
                exception_type=type(violation).__name__,
                phase="executor_post_process",
                context=violation.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected _RoundTripViolation")

        context = _record_failure(
            setup,
            token_id=token_id,
            node_id=node_id,
            run_id=run_id,
            input_data={"amount": "not_an_int"},
            error=error,
        )

        assert context["exception_type"] == "_RoundTripViolation"
        assert context["contract_name"] == "test_roundtrip"
        assert context["plugin"] == "FakeTransform"
        assert context["node_id"] == node_id
        assert context["run_id"] == run_id
        assert context["row_id"] == row_id
        assert context["token_id"] == token_id
        assert context["message"] == "round-trip test violation"

        payload = context["payload"]
        assert payload["field_name"] == "amount"
        assert payload["expected_type"] == "int"
        assert payload["observed_type"] == "str"
        assert payload["nested"]["inner"] == [1, 2, 3]

    def test_secrets_in_payload_are_scrubbed_before_landscape_write(self) -> None:
        run_id = "run-secret"
        row_id = "row-secret"
        token_id = "tok-secret"
        node_id = "n-secret"

        register_declaration_contract(_SecretContract())

        setup = _setup_landscape(
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            node_id=node_id,
        )

        try:
            run_post_emission_checks(
                inputs=_post_emission_inputs(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id),
                outputs=_post_emission_outputs(),
            )
        except _SecretViolation as violation:
            error = ExecutionError(
                exception=str(violation),
                exception_type=type(violation).__name__,
                phase="executor_post_process",
                context=violation.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected _SecretViolation")

        context = _record_failure(
            setup,
            token_id=token_id,
            node_id=node_id,
            run_id=run_id,
            input_data={"api_key": "[redacted-in-input]"},
            error=error,
        )

        assert "sk-abcdef" not in json.dumps(context)
        assert context["contract_name"] == "secret_test"
        assert context["payload"]["api_key"] == "<redacted-secret>"

    def test_secrets_in_message_are_scrubbed_before_landscape_write(self) -> None:
        run_id = "run-secret-message"
        row_id = "row-secret-message"
        token_id = "tok-secret-message"
        node_id = "n-secret-message"

        register_declaration_contract(_SecretMessageContract())

        setup = _setup_landscape(
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            node_id=node_id,
        )

        try:
            run_post_emission_checks(
                inputs=_post_emission_inputs(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id),
                outputs=_post_emission_outputs(),
            )
        except _SecretMessageViolation as violation:
            error = ExecutionError(
                exception=str(violation),
                exception_type=type(violation).__name__,
                phase="executor_post_process",
                context=violation.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected _SecretMessageViolation")

        context = _record_failure(
            setup,
            token_id=token_id,
            node_id=node_id,
            run_id=run_id,
            input_data={"note": "safe"},
            error=error,
        )

        assert "sk-abcdef" not in json.dumps(context)
        assert context["contract_name"] == "secret_message_test"
        assert context["message"] == "<redacted-secret>"
        assert context["payload"]["note"] == "safe"


class TestAggregateDeclarationContractViolationRoundTrip:
    """Verify aggregate declaration violations survive Landscape serialisation."""

    def setup_method(self) -> None:
        self._snapshot = _snapshot_registry_for_tests()
        _clear_registry_for_tests()

    def teardown_method(self) -> None:
        _restore_registry_snapshot_for_tests(self._snapshot)

    def test_aggregate_payload_survives_landscape_round_trip(self) -> None:
        run_id = "run-aggregate"
        row_id = "row-aggregate"
        token_id = "tok-aggregate"
        node_id = "n-aggregate"

        register_declaration_contract(_AggregateChildContractA())
        register_declaration_contract(_AggregateChildContractB())

        setup = _setup_landscape(
            run_id=run_id,
            row_id=row_id,
            token_id=token_id,
            node_id=node_id,
        )

        try:
            run_post_emission_checks(
                inputs=_post_emission_inputs(run_id=run_id, row_id=row_id, token_id=token_id, node_id=node_id),
                outputs=_post_emission_outputs(),
            )
        except AggregateDeclarationContractViolation as aggregate:
            error = ExecutionError(
                exception=str(aggregate),
                exception_type=type(aggregate).__name__,
                phase="declaration_dispatch",
                context=aggregate.to_audit_dict(),
            )
        else:
            raise AssertionError("Expected AggregateDeclarationContractViolation")

        context = _record_failure(
            setup,
            token_id=token_id,
            node_id=node_id,
            run_id=run_id,
            input_data={"amount": "not_an_int"},
            error=error,
        )

        assert context["exception_type"] == "AggregateDeclarationContractViolation"
        assert context["is_aggregate"] is True
        assert "contract_name" not in context, "aggregate must NOT emit contract_name (C5/S2-001)"

        violations = context["violations"]
        assert len(violations) == 2
        child_types = {v["exception_type"] for v in violations}
        assert child_types == {"_AggregateChildViolationA", "_AggregateChildViolationB"}

        child_names = {v["contract_name"] for v in violations}
        assert child_names == {"contract_a", "contract_b"}

        for child in violations:
            assert child["payload"]["reason"].startswith("child-")
