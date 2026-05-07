"""DeclaredOutputFieldsContract — ADR-011 behaviour and dispatcher coverage."""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts.declaration_contracts import (
    EXPECTED_CONTRACT_SITES,
    AggregateDeclarationContractViolation,
    BatchFlushInputs,
    BatchFlushOutputs,
    DeclarationContractViolation,
    PostEmissionInputs,
    PostEmissionOutputs,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    contract_sites,
    register_declaration_contract,
)
from elspeth.contracts.errors import DeclaredOutputFieldsViolation
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.engine.executors.declaration_dispatch import (
    run_batch_flush_checks,
    run_post_emission_checks,
)
from elspeth.engine.executors.declared_output_fields import (
    DeclaredOutputFieldsContract,
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
    name: str = "DeclaredOutputFieldsTransform",
    node_id: str | None = "declared-output-fields-node",
    declared_output_fields: frozenset[str] = frozenset({"new_a", "new_b"}),
    passes_through_input: bool = False,
    can_drop_rows: bool = False,
    declared_input_fields: frozenset[str] = frozenset(),
    is_batch_aware: bool = False,
) -> Any:
    plugin = type("DeclaredOutputFieldsPlugin", (), {})()
    plugin.name = name
    plugin.node_id = node_id
    plugin.declared_output_fields = declared_output_fields
    plugin.passes_through_input = passes_through_input
    plugin.can_drop_rows = can_drop_rows
    plugin.declared_input_fields = declared_input_fields
    plugin.is_batch_aware = is_batch_aware
    plugin._output_schema_config = None
    return plugin


@pytest.fixture()
def _isolated_registry():
    snapshot = _snapshot_registry_for_tests()
    _clear_registry_for_tests()
    yield
    _restore_registry_snapshot_for_tests(snapshot)


def test_applies_to_uses_direct_attribute() -> None:
    contract = DeclaredOutputFieldsContract()
    plugin = _plugin()
    assert contract.applies_to(plugin) is True
    plugin.declared_output_fields = frozenset()
    assert contract.applies_to(plugin) is False


def test_applies_to_on_plugin_missing_attribute_crashes() -> None:
    contract = DeclaredOutputFieldsContract()

    class _NoAttr:
        pass

    with pytest.raises(AttributeError):
        contract.applies_to(_NoAttr())


def test_contract_claims_both_dispatch_sites() -> None:
    assert contract_sites(DeclaredOutputFieldsContract()) == frozenset({"post_emission_check", "batch_flush_check"})


def test_post_emission_check_raises_on_missing_declared_output_field() -> None:
    contract = DeclaredOutputFieldsContract()
    inputs = PostEmissionInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("source",)),
        static_contract=frozenset({"new_a", "new_b"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = PostEmissionOutputs(emitted_rows=(_row(("source", "new_a")),))

    with pytest.raises(DeclaredOutputFieldsViolation) as exc_info:
        contract.post_emission_check(inputs, outputs)

    assert tuple(exc_info.value.payload["declared"]) == ("new_a", "new_b")
    assert tuple(exc_info.value.payload["violations"][0]["missing"]) == ("new_b",)
    assert tuple(exc_info.value.payload["violations"][0]["runtime_observed"]) == ("new_a", "source")


def test_batch_flush_check_raises_on_missing_declared_output_field() -> None:
    contract = DeclaredOutputFieldsContract()
    token_row = _row(("source",))
    inputs = BatchFlushInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        buffered_tokens=(token_row,),
        static_contract=frozenset({"new_a", "new_b"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = BatchFlushOutputs(emitted_rows=(_row(("source", "new_a")),))

    with pytest.raises(DeclaredOutputFieldsViolation) as exc_info:
        contract.batch_flush_check(inputs, outputs)

    assert tuple(exc_info.value.payload["violations"][0]["missing"]) == ("new_b",)


def test_post_emission_check_records_all_failing_emitted_rows() -> None:
    contract = DeclaredOutputFieldsContract()
    inputs = PostEmissionInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("source",)),
        static_contract=frozenset({"new_a", "new_b"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = PostEmissionOutputs(emitted_rows=(_row(("source", "new_a")), _row(("source", "new_b"))))

    with pytest.raises(DeclaredOutputFieldsViolation) as exc_info:
        contract.post_emission_check(inputs, outputs)

    violations = exc_info.value.payload["violations"]
    assert tuple(entry["emitted_index"] for entry in violations) == (0, 1)
    assert tuple(tuple(entry["runtime_observed"]) for entry in violations) == (
        ("new_a", "source"),
        ("new_b", "source"),
    )
    assert tuple(tuple(entry["missing"]) for entry in violations) == (("new_b",), ("new_a",))


def test_batch_flush_check_records_all_failing_emitted_rows() -> None:
    contract = DeclaredOutputFieldsContract()
    token_row = _row(("source",))
    inputs = BatchFlushInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        buffered_tokens=(token_row,),
        static_contract=frozenset({"new_a", "new_b"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = BatchFlushOutputs(emitted_rows=(_row(("source", "new_a")), _row(("source", "new_b"))))

    with pytest.raises(DeclaredOutputFieldsViolation) as exc_info:
        contract.batch_flush_check(inputs, outputs)

    violations = exc_info.value.payload["violations"]
    assert tuple(entry["emitted_index"] for entry in violations) == (0, 1)
    assert tuple(tuple(entry["missing"]) for entry in violations) == (("new_b",), ("new_a",))


def test_empty_emission_is_noop() -> None:
    contract = DeclaredOutputFieldsContract()
    inputs = PostEmissionInputs(
        plugin=_plugin(),
        node_id="n-1",
        run_id="run-1",
        row_id="row-1",
        token_id="token-1",
        input_row=_row(("source",)),
        static_contract=frozenset({"new_a", "new_b"}),
        effective_input_fields=frozenset({"source"}),
    )

    contract.post_emission_check(inputs, PostEmissionOutputs(emitted_rows=()))


def test_dispatcher_raises_single_violation_for_declared_output_fields_only(_isolated_registry) -> None:
    register_declaration_contract(DeclaredOutputFieldsContract())
    plugin = _plugin()
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-single",
        row_id="row-single",
        token_id="token-single",
        input_row=_row(("source",)),
        static_contract=frozenset({"new_a", "new_b"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = PostEmissionOutputs(emitted_rows=(_row(("source", "new_a")),))

    with pytest.raises(DeclaredOutputFieldsViolation) as exc_info:
        run_post_emission_checks(inputs=inputs, outputs=outputs)

    assert exc_info.value.contract_name == "declared_output_fields"
    assert tuple(exc_info.value.payload["declared"]) == ("new_a", "new_b")


def test_dispatcher_aggregates_with_pass_through(_isolated_registry) -> None:
    register_declaration_contract(PassThroughDeclarationContract())
    register_declaration_contract(DeclaredOutputFieldsContract())

    plugin = _plugin(
        declared_output_fields=frozenset({"new_a", "new_b"}),
        passes_through_input=True,
    )
    input_row = _row(("source", "carry"))
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-aggregate",
        row_id="row-aggregate",
        token_id="token-aggregate",
        input_row=input_row,
        static_contract=frozenset({"new_a", "new_b"}),
        effective_input_fields=frozenset({"source", "carry"}),
    )
    outputs = PostEmissionOutputs(emitted_rows=(_row(("source", "new_a")),))

    with pytest.raises(AggregateDeclarationContractViolation) as exc_info:
        run_post_emission_checks(inputs=inputs, outputs=outputs)

    aggregate = exc_info.value
    assert not isinstance(aggregate, DeclarationContractViolation)
    child_types = {type(child).__name__ for child in aggregate.violations}
    assert child_types == {"DeclaredOutputFieldsViolation", "PassThroughContractViolation"}

    declared_child = next(child for child in aggregate.violations if isinstance(child, DeclaredOutputFieldsViolation))
    assert declared_child.contract_name == "declared_output_fields"
    assert tuple(declared_child.payload["violations"][0]["missing"]) == ("new_b",)

    pass_through_child = next(child for child in aggregate.violations if type(child).__name__ == "PassThroughContractViolation")
    assert tuple(pass_through_child.divergence_set) == ("carry",)

    audit = aggregate.to_audit_dict()
    assert audit["is_aggregate"] is True
    child_exception_types = {entry["exception_type"] for entry in audit["violations"]}
    assert child_exception_types == {"DeclaredOutputFieldsViolation", "PassThroughContractViolation"}


def test_batch_flush_dispatcher_raises_declared_output_fields_violation(_isolated_registry) -> None:
    register_declaration_contract(DeclaredOutputFieldsContract())
    plugin = _plugin()
    token_row = _row(("source",))
    inputs = BatchFlushInputs(
        plugin=plugin,
        node_id=plugin.node_id or "",
        run_id="run-batch",
        row_id="row-batch",
        token_id="token-batch",
        buffered_tokens=(token_row,),
        static_contract=frozenset({"new_a", "new_b"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = BatchFlushOutputs(emitted_rows=(_row(("source", "new_a")),))

    with pytest.raises(DeclaredOutputFieldsViolation) as exc_info:
        run_batch_flush_checks(inputs=inputs, outputs=outputs)

    assert exc_info.value.contract_name == "declared_output_fields"


def test_phase_2c_manifest_contains_all_production_contracts() -> None:
    assert len(EXPECTED_CONTRACT_SITES) == 7
    assert frozenset(EXPECTED_CONTRACT_SITES.keys()) == frozenset(
        {
            "passes_through_input",
            "declared_output_fields",
            "declared_required_fields",
            "schema_config_mode",
            "can_drop_rows",
            "source_guaranteed_fields",
            "sink_required_fields",
        }
    )
