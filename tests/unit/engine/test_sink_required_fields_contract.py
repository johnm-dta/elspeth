"""SinkRequiredFieldsContract — ADR-017 behaviour and dispatcher coverage."""

from __future__ import annotations

from typing import Any, ClassVar, TypedDict

import pytest

from elspeth.contracts.declaration_contracts import (
    AggregateDeclarationContractViolation,
    BoundaryInputs,
    BoundaryOutputs,
    DeclarationContract,
    DeclarationContractViolation,
    ExampleBundle,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    contract_sites,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.contracts.diversion import SinkWriteResult
from elspeth.contracts.errors import (
    OrchestrationInvariantError,
    SinkRequiredFieldsViolation,
)
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.engine.executors.declaration_dispatch import run_boundary_checks
from elspeth.engine.executors.sink_required_fields import SinkRequiredFieldsContract
from elspeth.plugins.infrastructure.base import BaseSink


def _contract(
    *,
    required_fields: tuple[str, ...],
    optional_fields: tuple[str, ...] = (),
) -> SchemaContract:
    fields = tuple(
        FieldContract(
            normalized_name=name,
            original_name=name,
            python_type=str,
            required=True,
            source="inferred",
            nullable=False,
        )
        for name in required_fields
    ) + tuple(
        FieldContract(
            normalized_name=name,
            original_name=name,
            python_type=str,
            required=False,
            source="inferred",
            nullable=False,
        )
        for name in optional_fields
    )
    return SchemaContract(mode="OBSERVED", fields=fields, locked=True)


class _TestSinkPlugin(BaseSink):
    name = "SinkRequiredFieldsSink"
    input_schema = object

    def __init__(
        self,
        *,
        name: str,
        node_id: str | None,
        declared_required_fields: frozenset[str],
    ) -> None:
        super().__init__({})
        self.name = name
        self.node_id = node_id
        self.declared_required_fields = declared_required_fields

    def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> SinkWriteResult:
        raise NotImplementedError

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _plugin(
    *,
    name: str = "SinkRequiredFieldsSink",
    node_id: str | None = "sink-required-fields-node",
    declared_required_fields: frozenset[str] = frozenset({"customer_id", "amount"}),
) -> Any:
    return _TestSinkPlugin(
        name=name,
        node_id=node_id,
        declared_required_fields=declared_required_fields,
    )


class _SecondaryPayload(TypedDict):
    note: str


class _SecondaryBoundaryViolation(DeclarationContractViolation):
    payload_schema = _SecondaryPayload


class _SecondaryBoundaryContract(DeclarationContract):
    name: ClassVar[str] = "secondary_sink_boundary_test"
    payload_schema: ClassVar[type] = _SecondaryPayload

    def applies_to(self, plugin: Any) -> bool:
        return bool(plugin.declared_required_fields)

    @implements_dispatch_site("boundary_check")
    def boundary_check(self, inputs: BoundaryInputs, outputs: BoundaryOutputs) -> None:
        raise _SecondaryBoundaryViolation(
            plugin=inputs.plugin.name,
            node_id=inputs.node_id,
            run_id=inputs.run_id,
            row_id=inputs.row_id,
            token_id=inputs.token_id,
            payload={"note": "second"},
            message="secondary sink boundary violation",
        )

    @classmethod
    def negative_example(cls) -> ExampleBundle:
        raise NotImplementedError

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        raise NotImplementedError


@pytest.fixture()
def _isolated_registry():
    snapshot = _snapshot_registry_for_tests()
    _clear_registry_for_tests()
    yield
    _restore_registry_snapshot_for_tests(snapshot)


def test_applies_to_uses_direct_attribute() -> None:
    contract = SinkRequiredFieldsContract()
    plugin = _plugin()
    assert contract.applies_to(plugin) is True
    plugin.declared_required_fields = frozenset()
    assert contract.applies_to(plugin) is False


def test_applies_to_on_plugin_missing_attribute_returns_false() -> None:
    contract = SinkRequiredFieldsContract()

    class _NoAttr:
        pass

    assert contract.applies_to(_NoAttr()) is False


def test_applies_to_true_for_inherited_declared_required_fields() -> None:
    contract = SinkRequiredFieldsContract()

    class _DeclaredSinkBase(BaseSink):
        name = "declared-sink-base"
        input_schema = object
        declared_required_fields = frozenset({"customer_id"})

        def __init__(self) -> None:
            self.config = {}
            self.node_id = None

        def write(self, rows: list[dict[str, Any]], ctx: PluginContext) -> SinkWriteResult:
            raise NotImplementedError

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    class _InheritedDeclaredSink(_DeclaredSinkBase):
        pass

    assert contract.applies_to(_InheritedDeclaredSink()) is True


def test_applies_to_false_for_wrong_plugin_role_even_when_attr_present() -> None:
    contract = SinkRequiredFieldsContract()

    class _NotASink:
        declared_required_fields = frozenset({"customer_id"})

    assert contract.applies_to(_NotASink()) is False


def test_contract_claims_boundary_dispatch_site() -> None:
    assert contract_sites(SinkRequiredFieldsContract()) == frozenset({"boundary_check"})


def test_boundary_check_raises_on_missing_required_field() -> None:
    contract = SinkRequiredFieldsContract()
    inputs = BoundaryInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        static_contract=frozenset({"customer_id", "amount"}),
        row_data={"customer_id": "v"},
        row_contract=_contract(required_fields=("customer_id",), optional_fields=("amount",)),
    )

    with pytest.raises(SinkRequiredFieldsViolation) as exc_info:
        contract.boundary_check(inputs, BoundaryOutputs())

    assert tuple(exc_info.value.payload["missing"]) == ("amount",)
    assert tuple(exc_info.value.payload["runtime_observed"]) == ("customer_id",)
    assert "coalesce merge" in str(exc_info.value)


def test_boundary_check_returns_none_when_required_fields_present() -> None:
    contract = SinkRequiredFieldsContract()
    inputs = BoundaryInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        static_contract=frozenset({"customer_id", "amount"}),
        row_data={"customer_id": "v", "amount": "1"},
        row_contract=_contract(required_fields=("customer_id", "amount")),
    )

    assert contract.boundary_check(inputs, BoundaryOutputs()) is None


def test_boundary_check_skips_contract_annotation_when_row_contract_absent() -> None:
    contract = SinkRequiredFieldsContract()
    inputs = BoundaryInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        static_contract=frozenset({"customer_id", "amount"}),
        row_data={"customer_id": "v"},
        row_contract=None,
    )

    with pytest.raises(SinkRequiredFieldsViolation) as exc_info:
        contract.boundary_check(inputs, BoundaryOutputs())

    assert "coalesce merge" not in str(exc_info.value)


def test_boundary_check_preserves_orchestration_invariant_on_missing_node_id() -> None:
    contract = SinkRequiredFieldsContract()
    inputs = BoundaryInputs(
        plugin=_plugin(node_id=None),
        node_id="",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        static_contract=frozenset({"customer_id"}),
        row_data={"customer_id": "v"},
        row_contract=None,
    )

    with pytest.raises(OrchestrationInvariantError, match="has no node_id set"):
        contract.boundary_check(inputs, BoundaryOutputs())


def test_dispatcher_raises_identity_preserving_violation(_isolated_registry) -> None:
    register_declaration_contract(SinkRequiredFieldsContract())

    with pytest.raises(SinkRequiredFieldsViolation) as exc_info:
        run_boundary_checks(
            inputs=BoundaryInputs(
                plugin=_plugin(node_id="node-1"),
                node_id="node-1",
                run_id="run-1",
                row_id="row-1",
                token_id="token-1",
                static_contract=frozenset({"customer_id", "amount"}),
                row_data={"customer_id": "v"},
                row_contract=None,
            ),
            outputs=BoundaryOutputs(),
        )

    assert exc_info.value.contract_name == "sink_required_fields"


def test_dispatcher_aggregates_multiple_boundary_violations(_isolated_registry) -> None:
    register_declaration_contract(SinkRequiredFieldsContract())
    register_declaration_contract(_SecondaryBoundaryContract())

    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_boundary_checks(
            inputs=BoundaryInputs(
                plugin=_plugin(node_id="node-1"),
                node_id="node-1",
                run_id="run-1",
                row_id="row-1",
                token_id="token-1",
                static_contract=frozenset({"customer_id", "amount"}),
                row_data={"customer_id": "v"},
                row_contract=None,
            ),
            outputs=BoundaryOutputs(),
        )

    child_types = {entry["exception_type"] for entry in exc_info.value.to_audit_dict()["violations"]}
    assert child_types == {"SinkRequiredFieldsViolation", "_SecondaryBoundaryViolation"}
