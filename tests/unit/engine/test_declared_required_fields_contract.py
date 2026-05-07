"""DeclaredRequiredFieldsContract — ADR-013 behaviour and dispatcher coverage."""

from __future__ import annotations

from typing import Any, TypedDict

import pytest

from elspeth.contracts import TransformResult
from elspeth.contracts.data import PluginSchema as _PermissiveSchema
from elspeth.contracts.declaration_contracts import (
    AggregateDeclarationContractViolation,
    DeclarationContract,
    DeclarationContractViolation,
    ExampleBundle,
    PreEmissionInputs,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    contract_sites,
    implements_dispatch_site,
    register_declaration_contract,
)
from elspeth.contracts.errors import (
    DeclaredRequiredInputFieldsViolation,
    FrameworkBugError,
    OrchestrationInvariantError,
)
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.engine.executors.declaration_dispatch import run_pre_emission_checks
from elspeth.engine.executors.declared_required_fields import DeclaredRequiredFieldsContract
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig


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
    node_id: str | None = "declared-required-fields-node",
    declared_input_fields: frozenset[str] = frozenset({"customer_id", "account_id"}),
    passes_through_input: bool = False,
    can_drop_rows: bool = False,
    is_batch_aware: bool = False,
) -> Any:
    plugin = type("DeclaredRequiredFieldsPlugin", (), {})()
    plugin.name = name
    plugin.node_id = node_id
    plugin.passes_through_input = passes_through_input
    plugin.can_drop_rows = can_drop_rows
    plugin.declared_output_fields = frozenset()
    plugin.declared_input_fields = declared_input_fields
    plugin._output_schema_config = None
    plugin.is_batch_aware = is_batch_aware
    return plugin


class _DummyTransformConfig(TransformDataConfig):
    pass


class _DummyTransform(BaseTransform):
    name = "dummy_declared_required"
    config_model = _DummyTransformConfig
    input_schema = _PermissiveSchema
    output_schema = _PermissiveSchema

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = _DummyTransformConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(cfg)

    def process(self, row: PipelineRow, ctx: Any) -> TransformResult:
        raise NotImplementedError


class _DummyBatchTransform(_DummyTransform):
    is_batch_aware = True


class _SecondaryPayload(TypedDict):
    note: str


class _SecondaryPreEmissionViolation(DeclarationContractViolation):
    payload_schema = _SecondaryPayload


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
        return DeclaredRequiredFieldsContract.negative_example()

    @classmethod
    def positive_example_does_not_apply(cls) -> ExampleBundle:
        return DeclaredRequiredFieldsContract.positive_example_does_not_apply()


@pytest.fixture()
def _isolated_registry():
    snapshot = _snapshot_registry_for_tests()
    _clear_registry_for_tests()
    yield
    _restore_registry_snapshot_for_tests(snapshot)


def test_applies_to_uses_direct_attribute() -> None:
    contract = DeclaredRequiredFieldsContract()
    plugin = _plugin()
    assert contract.applies_to(plugin) is True
    plugin.declared_input_fields = frozenset()
    assert contract.applies_to(plugin) is False


def test_applies_to_on_plugin_missing_attribute_crashes() -> None:
    contract = DeclaredRequiredFieldsContract()

    class _NoAttr:
        pass

    with pytest.raises(AttributeError):
        contract.applies_to(_NoAttr())


def test_applies_to_empty_declaration_still_requires_batch_flag_shape() -> None:
    contract = DeclaredRequiredFieldsContract()

    class _NoBatchFlag:
        name = "no-batch-flag"
        node_id = "node-1"
        declared_input_fields = frozenset()

    with pytest.raises(AttributeError):
        contract.applies_to(_NoBatchFlag())


def test_applies_to_fails_closed_for_batch_aware_transform() -> None:
    contract = DeclaredRequiredFieldsContract()

    with pytest.raises(FrameworkBugError, match="batch-pre-execution dispatch site exists"):
        contract.applies_to(_plugin(is_batch_aware=True))


def test_contract_claims_pre_emission_dispatch_site() -> None:
    assert contract_sites(DeclaredRequiredFieldsContract()) == frozenset({"pre_emission_check"})


def test_pre_emission_check_raises_on_missing_declared_input_field() -> None:
    contract = DeclaredRequiredFieldsContract()
    inputs = PreEmissionInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("account_id",)),
        static_contract=frozenset(),
        effective_input_fields=frozenset({"account_id"}),
    )

    with pytest.raises(DeclaredRequiredInputFieldsViolation) as exc_info:
        contract.pre_emission_check(inputs)

    assert tuple(exc_info.value.payload["missing"]) == ("customer_id",)
    assert tuple(exc_info.value.payload["effective_input_fields"]) == ("account_id",)


def test_pre_emission_check_returns_none_when_declared_fields_present() -> None:
    contract = DeclaredRequiredFieldsContract()
    inputs = PreEmissionInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("customer_id", "account_id")),
        static_contract=frozenset(),
        effective_input_fields=frozenset({"customer_id", "account_id"}),
    )

    assert contract.pre_emission_check(inputs) is None


def test_pre_emission_check_preserves_orchestration_invariant_on_missing_node_id() -> None:
    contract = DeclaredRequiredFieldsContract()
    plugin = _plugin(node_id=None)
    inputs = PreEmissionInputs(
        plugin=plugin,
        node_id="",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("account_id",)),
        static_contract=frozenset(),
        effective_input_fields=frozenset({"account_id"}),
    )

    with pytest.raises(OrchestrationInvariantError):
        contract.pre_emission_check(inputs)


def test_dispatcher_raises_single_violation_for_declared_required_fields_only(_isolated_registry) -> None:
    register_declaration_contract(DeclaredRequiredFieldsContract())
    plugin = _plugin()
    inputs = PreEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-single",
        row_id="row-single",
        token_id="token-single",
        input_row=_row(("account_id",)),
        static_contract=frozenset(),
        effective_input_fields=frozenset({"account_id"}),
    )

    with pytest.raises(DeclaredRequiredInputFieldsViolation) as exc_info:
        run_pre_emission_checks(inputs=inputs)

    assert exc_info.value.contract_name == "declared_required_fields"
    assert tuple(exc_info.value.payload["declared"]) == ("account_id", "customer_id")


def test_dispatcher_aggregates_with_second_pre_emission_contract(_isolated_registry) -> None:
    register_declaration_contract(DeclaredRequiredFieldsContract())
    register_declaration_contract(_SecondaryPreEmissionContract())

    plugin = _plugin()
    inputs = PreEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-aggregate",
        row_id="row-aggregate",
        token_id="token-aggregate",
        input_row=_row(("account_id",)),
        static_contract=frozenset(),
        effective_input_fields=frozenset({"account_id"}),
    )

    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_pre_emission_checks(inputs=inputs)

    aggregate = exc_info.value
    assert not isinstance(aggregate, DeclarationContractViolation)
    child_types = {type(child).__name__ for child in aggregate.violations}
    assert child_types == {"DeclaredRequiredInputFieldsViolation", "_SecondaryPreEmissionViolation"}

    declared_child = next(child for child in aggregate.violations if isinstance(child, DeclaredRequiredInputFieldsViolation))
    assert declared_child.contract_name == "declared_required_fields"
    assert tuple(declared_child.payload["missing"]) == ("customer_id",)

    audit = aggregate.to_audit_dict()
    assert audit["is_aggregate"] is True
    child_exception_types = {entry["exception_type"] for entry in audit["violations"]}
    assert child_exception_types == {"DeclaredRequiredInputFieldsViolation", "_SecondaryPreEmissionViolation"}


def test_declared_input_fields_normalized_from_config() -> None:
    transform = _DummyTransform(
        {
            "schema": {"mode": "observed"},
            "required_input_fields": ["customer_id", "account_id"],
        }
    )

    assert transform.declared_input_fields == frozenset({"customer_id", "account_id"})


def test_declared_input_fields_empty_when_not_configured() -> None:
    transform = _DummyTransform({"schema": {"mode": "observed"}})
    assert transform.declared_input_fields == frozenset()


def test_batch_aware_declared_input_fields_fail_closed_at_construction() -> None:
    with pytest.raises(FrameworkBugError, match="batch-pre-execution dispatch site exists"):
        _DummyBatchTransform(
            {
                "schema": {"mode": "observed"},
                "required_input_fields": ["customer_id"],
            }
        )
