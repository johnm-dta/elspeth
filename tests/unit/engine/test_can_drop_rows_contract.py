"""CanDropRowsContract — ADR-012 behaviour and dispatcher coverage."""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts.declaration_contracts import (
    EXPECTED_CONTRACT_SITES,
    AggregateDeclarationContractViolation,
    BatchFlushInputs,
    BatchFlushOutputs,
    PostEmissionInputs,
    PostEmissionOutputs,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    contract_sites,
    register_declaration_contract,
)
from elspeth.contracts.errors import (
    PassThroughContractViolation,
    UnexpectedEmptyEmissionViolation,
)
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.engine.executors.can_drop_rows import (
    CanDropRowsContract,
    verify_can_drop_rows,
    verify_zero_emission_declaration_path,
)
from elspeth.engine.executors.declaration_dispatch import (
    run_batch_flush_checks,
    run_post_emission_checks,
)
from elspeth.engine.executors.pass_through import PassThroughDeclarationContract


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
    node_id: str | None = "can-drop-rows-node",
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


@pytest.fixture()
def _isolated_registry():
    snapshot = _snapshot_registry_for_tests()
    _clear_registry_for_tests()
    yield
    _restore_registry_snapshot_for_tests(snapshot)


def test_applies_to_uses_direct_attributes() -> None:
    contract = CanDropRowsContract()
    plugin = _plugin()
    assert contract.applies_to(plugin) is True
    plugin.can_drop_rows = True
    assert contract.applies_to(plugin) is False
    plugin.can_drop_rows = False
    plugin.passes_through_input = False
    assert contract.applies_to(plugin) is False


def test_applies_to_on_plugin_missing_attribute_crashes() -> None:
    contract = CanDropRowsContract()

    class _NoAttr:
        pass

    with pytest.raises(AttributeError):
        contract.applies_to(_NoAttr())


@pytest.mark.parametrize(
    ("attr_name", "bad_value"),
    [
        ("passes_through_input", "false"),
        ("can_drop_rows", 0),
    ],
)
def test_applies_to_rejects_non_boolean_flags(attr_name: str, bad_value: object) -> None:
    contract = CanDropRowsContract()
    plugin = _plugin()
    setattr(plugin, attr_name, bad_value)

    with pytest.raises(TypeError, match=rf"{attr_name} must be bool"):
        contract.applies_to(plugin)


def test_contract_claims_both_dispatch_sites() -> None:
    assert contract_sites(CanDropRowsContract()) == frozenset({"post_emission_check", "batch_flush_check"})


def test_manifest_entry_matches_claimed_sites() -> None:
    assert EXPECTED_CONTRACT_SITES["can_drop_rows"] == frozenset({"post_emission_check", "batch_flush_check"})


def test_post_emission_check_raises_on_zero_rows() -> None:
    contract = CanDropRowsContract()
    plugin = _plugin(passes_through_input=True, can_drop_rows=False)
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("source",)),
        static_contract=frozenset(),
        effective_input_fields=frozenset({"source"}),
    )

    with pytest.raises(UnexpectedEmptyEmissionViolation) as exc_info:
        contract.post_emission_check(inputs, PostEmissionOutputs(emitted_rows=()))

    assert exc_info.value.payload == {
        "passes_through_input": True,
        "can_drop_rows": False,
        "emitted_count": 0,
    }


def test_post_emission_check_noops_when_can_drop_rows_true() -> None:
    contract = CanDropRowsContract()
    plugin = _plugin(passes_through_input=True, can_drop_rows=True)
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("source",)),
        static_contract=frozenset(),
        effective_input_fields=frozenset({"source"}),
    )

    contract.post_emission_check(inputs, PostEmissionOutputs(emitted_rows=()))


def test_post_emission_check_scopes_out_when_not_pass_through() -> None:
    contract = CanDropRowsContract()
    plugin = _plugin(passes_through_input=False, can_drop_rows=False)
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("source",)),
        static_contract=frozenset(),
        effective_input_fields=frozenset({"source"}),
    )

    contract.post_emission_check(inputs, PostEmissionOutputs(emitted_rows=()))


def test_batch_flush_check_raises_on_zero_rows() -> None:
    contract = CanDropRowsContract()
    plugin = _plugin(passes_through_input=True, can_drop_rows=False, is_batch_aware=True)
    token_row = _row(("source",))
    inputs = BatchFlushInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-batch",
        row_id="row-batch",
        token_id="token-batch",
        buffered_tokens=(token_row,),
        static_contract=frozenset(),
        effective_input_fields=frozenset({"source"}),
    )

    with pytest.raises(UnexpectedEmptyEmissionViolation) as exc_info:
        contract.batch_flush_check(inputs, BatchFlushOutputs(emitted_rows=()))

    assert exc_info.value.payload["emitted_count"] == 0


def test_verify_can_drop_rows_rejects_non_boolean_flags_before_building_payload() -> None:
    plugin = _plugin()
    plugin.passes_through_input = 1
    plugin.can_drop_rows = 0

    with pytest.raises(TypeError, match=r"passes_through_input must be bool"):
        verify_can_drop_rows(
            plugin=plugin,
            plugin_name=plugin.name,
            node_id=plugin.node_id or "",
            run_id="run-1",
            row_id="row-1",
            token_id="token-1",
            emitted_count=0,
        )


def test_verify_zero_emission_declaration_path_rejects_non_boolean_flags() -> None:
    plugin = _plugin()
    plugin.passes_through_input = "false"
    plugin.can_drop_rows = "false"

    with pytest.raises(TypeError, match=r"passes_through_input must be bool"):
        verify_zero_emission_declaration_path(
            plugin=plugin,
            plugin_name=plugin.name,
            node_id=plugin.node_id or "",
            run_id="run-1",
            row_id="row-1",
            token_id="token-1",
            emitted_count=0,
            used_success_empty=True,
        )


def test_dispatcher_aggregates_with_pass_through_on_zero_rows(_isolated_registry) -> None:
    register_declaration_contract(PassThroughDeclarationContract())
    register_declaration_contract(CanDropRowsContract())

    plugin = _plugin(passes_through_input=True, can_drop_rows=False)
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-aggregate",
        row_id="row-aggregate",
        token_id="token-aggregate",
        input_row=_row(("source", "carry")),
        static_contract=frozenset(),
        effective_input_fields=frozenset({"source", "carry"}),
    )

    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_post_emission_checks(inputs=inputs, outputs=PostEmissionOutputs(emitted_rows=()))

    child_types = {type(child).__name__ for child in exc_info.value.violations}
    assert child_types == {"UnexpectedEmptyEmissionViolation", "PassThroughContractViolation"}

    empty_child = next(child for child in exc_info.value.violations if isinstance(child, UnexpectedEmptyEmissionViolation))
    assert empty_child.contract_name == "can_drop_rows"
    assert empty_child.payload["emitted_count"] == 0

    pass_through_child = next(child for child in exc_info.value.violations if isinstance(child, PassThroughContractViolation))
    assert pass_through_child.divergence_set == frozenset({"carry", "source"})


def test_batch_flush_dispatcher_aggregates_with_pass_through_on_zero_rows(_isolated_registry) -> None:
    register_declaration_contract(PassThroughDeclarationContract())
    register_declaration_contract(CanDropRowsContract())

    plugin = _plugin(passes_through_input=True, can_drop_rows=False, is_batch_aware=True)
    token_row = _row(("source", "carry"))
    inputs = BatchFlushInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-batch-aggregate",
        row_id="row-batch-aggregate",
        token_id="token-batch-aggregate",
        buffered_tokens=(token_row,),
        static_contract=frozenset(),
        effective_input_fields=frozenset({"source", "carry"}),
    )

    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_batch_flush_checks(inputs=inputs, outputs=BatchFlushOutputs(emitted_rows=()))

    child_types = {type(child).__name__ for child in exc_info.value.violations}
    assert child_types == {"UnexpectedEmptyEmissionViolation", "PassThroughContractViolation"}
