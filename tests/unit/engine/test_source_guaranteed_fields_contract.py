"""SourceGuaranteedFieldsContract — ADR-016 behaviour and dispatcher coverage."""

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
from elspeth.contracts.errors import (
    FrameworkBugError,
    OrchestrationInvariantError,
    SourceGuaranteedFieldsViolation,
)
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.results import SourceRow
from elspeth.contracts.schema_contract import FieldContract, SchemaContract
from elspeth.engine.executors.declaration_dispatch import run_boundary_checks
from elspeth.engine.executors.source_guaranteed_fields import SourceGuaranteedFieldsContract
from elspeth.plugins.infrastructure.base import BaseSource


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
        node_id: str | None,
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
    node_id: str | None = "source-guaranteed-fields-node",
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
    contract = SourceGuaranteedFieldsContract()
    plugin = _plugin()
    assert contract.applies_to(plugin) is True
    plugin.declared_guaranteed_fields = frozenset()
    assert contract.applies_to(plugin) is False


def test_applies_to_on_plugin_missing_attribute_returns_false() -> None:
    contract = SourceGuaranteedFieldsContract()

    class _NoAttr:
        pass

    assert contract.applies_to(_NoAttr()) is False


def test_applies_to_true_for_inherited_declared_guaranteed_fields() -> None:
    contract = SourceGuaranteedFieldsContract()

    class _DeclaredSourceBase(BaseSource):
        name = "declared-source-base"
        output_schema = object
        declared_guaranteed_fields = frozenset({"customer_id"})

        def __init__(self) -> None:
            # Unit test for contract role + MRO lookup only. We intentionally
            # skip BaseSource.__init__ so the inherited class attribute remains
            # visible and the test isolates applies_to() semantics.
            self.config = {}
            self.node_id = None

        def load(self, ctx: PluginContext):
            yield SourceRow.valid({"customer_id": "v"}, contract=_contract(("customer_id",)))

        def close(self) -> None:
            pass

    class _InheritedDeclaredSource(_DeclaredSourceBase):
        pass

    assert contract.applies_to(_InheritedDeclaredSource()) is True


def test_applies_to_false_for_wrong_plugin_role_even_when_attr_present() -> None:
    contract = SourceGuaranteedFieldsContract()

    class _NotASource:
        declared_guaranteed_fields = frozenset({"customer_id"})

    assert contract.applies_to(_NotASource()) is False


def test_contract_claims_boundary_dispatch_site() -> None:
    assert contract_sites(SourceGuaranteedFieldsContract()) == frozenset({"boundary_check"})


def test_boundary_check_raises_on_missing_declared_guaranteed_field() -> None:
    contract = SourceGuaranteedFieldsContract()
    inputs = BoundaryInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        static_contract=frozenset({"customer_id", "account_id"}),
        row_data={"customer_id": "v", "account_id": "v"},
        row_contract=_contract(("account_id",)),
    )

    with pytest.raises(SourceGuaranteedFieldsViolation) as exc_info:
        contract.boundary_check(inputs, BoundaryOutputs())

    assert tuple(exc_info.value.payload["missing"]) == ("customer_id",)
    assert tuple(exc_info.value.payload["runtime_observed"]) == ("account_id",)


def test_boundary_check_raises_when_row_contract_missing() -> None:
    contract = SourceGuaranteedFieldsContract()
    inputs = BoundaryInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        static_contract=frozenset({"customer_id"}),
        row_data={"customer_id": "v"},
        row_contract=None,
    )

    with pytest.raises(FrameworkBugError, match="without a schema contract"):
        contract.boundary_check(inputs, BoundaryOutputs())


def test_boundary_check_returns_none_when_guarantees_present() -> None:
    contract = SourceGuaranteedFieldsContract()
    inputs = BoundaryInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        static_contract=frozenset({"customer_id", "account_id"}),
        row_data={"customer_id": "v", "account_id": "v"},
        row_contract=_contract(("customer_id", "account_id")),
    )

    assert contract.boundary_check(inputs, BoundaryOutputs()) is None


def test_boundary_check_preserves_orchestration_invariant_on_missing_node_id() -> None:
    contract = SourceGuaranteedFieldsContract()
    inputs = BoundaryInputs(
        plugin=_plugin(node_id=None),
        node_id="",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        static_contract=frozenset({"customer_id"}),
        row_data={"customer_id": "v"},
        row_contract=_contract(("customer_id",)),
    )

    with pytest.raises(OrchestrationInvariantError, match="has no node_id set"):
        contract.boundary_check(inputs, BoundaryOutputs())


def test_dispatcher_raises_identity_preserving_violation(_isolated_registry) -> None:
    register_declaration_contract(SourceGuaranteedFieldsContract())

    with pytest.raises(SourceGuaranteedFieldsViolation) as exc_info:
        run_boundary_checks(
            inputs=BoundaryInputs(
                plugin=_plugin(node_id="node-1"),
                node_id="node-1",
                run_id="run-1",
                row_id="row-1",
                token_id="token-1",
                static_contract=frozenset({"customer_id", "account_id"}),
                row_data={"customer_id": "v"},
                row_contract=_contract(("customer_id",)),
            ),
            outputs=BoundaryOutputs(),
        )

    assert exc_info.value.contract_name == "source_guaranteed_fields"


def test_dispatcher_aggregates_multiple_boundary_violations(_isolated_registry) -> None:
    register_declaration_contract(SourceGuaranteedFieldsContract())
    register_declaration_contract(_SecondaryBoundaryContract())

    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_boundary_checks(
            inputs=BoundaryInputs(
                plugin=_plugin(node_id="node-1"),
                node_id="node-1",
                run_id="run-1",
                row_id="row-1",
                token_id="token-1",
                static_contract=frozenset({"customer_id", "account_id"}),
                row_data={"account_id": "v"},
                row_contract=_contract(("account_id",)),
            ),
            outputs=BoundaryOutputs(),
        )

    child_types = {entry["exception_type"] for entry in exc_info.value.to_audit_dict()["violations"]}
    assert child_types == {"SourceGuaranteedFieldsViolation", "_SecondaryBoundaryViolation"}
