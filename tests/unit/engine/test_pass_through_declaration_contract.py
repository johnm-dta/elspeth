"""PassThroughDeclarationContract — behaviour parity with direct verify_pass_through.

Post-H2 (ADR-010 §Semantics amendment 2026-04-20): the contract inherits the
nominal ``DeclarationContract`` ABC and claims ``post_emission_check`` +
``batch_flush_check`` via ``@implements_dispatch_site`` markers. Dispatch
bundles are ``PostEmissionInputs`` / ``PostEmissionOutputs``;
``effective_input_fields`` is caller-derived (no ``override_input_fields``
sentinel).
"""

from __future__ import annotations

import pytest

from elspeth.contracts.declaration_contracts import (
    BatchFlushInputs,
    BatchFlushOutputs,
    PostEmissionInputs,
    PostEmissionOutputs,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    derive_effective_input_fields,
    register_declaration_contract,
)
from elspeth.contracts.errors import (
    FrameworkBugError,
    OrchestrationInvariantError,
    PassThroughContractViolation,
    UnexpectedEmptyEmissionViolation,
)
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.engine.executors.can_drop_rows import CanDropRowsContract
from elspeth.engine.executors.declaration_dispatch import (
    run_batch_flush_checks,
    run_post_emission_checks,
)
from elspeth.engine.executors.pass_through import (
    PassThroughDeclarationContract,
    verify_pass_through,
)


def _contract(fields: tuple[str, ...]) -> SchemaContract:
    return SchemaContract(
        mode="OBSERVED",
        fields=tuple(
            FieldContract(
                normalized_name=n,
                original_name=n,
                python_type=str,
                required=True,
                source="inferred",
                nullable=False,
            )
            for n in fields
        ),
        locked=True,
    )


def _row(fields: tuple[str, ...]) -> PipelineRow:
    return PipelineRow(dict.fromkeys(fields, "v"), _contract(fields))


def _empty_row() -> PipelineRow:
    return PipelineRow({}, _contract(()))


class _FakeTransform:
    name = "Fake"
    node_id = "n-1"
    passes_through_input = True
    can_drop_rows = False
    declared_input_fields = frozenset()
    is_batch_aware = False
    _output_schema_config = None


@pytest.fixture()
def _isolated_registry():
    snapshot = _snapshot_registry_for_tests()
    _clear_registry_for_tests()
    yield
    _restore_registry_snapshot_for_tests(snapshot)


def test_applies_to_uses_direct_attribute() -> None:
    c = PassThroughDeclarationContract()
    t = _FakeTransform()
    assert c.applies_to(t) is True
    t.passes_through_input = False
    assert c.applies_to(t) is False


def test_applies_to_on_plugin_missing_attribute_crashes() -> None:
    """CLAUDE.md offensive programming: plugin missing passes_through_input is
    a framework bug; must crash loudly, not silently return False."""
    c = PassThroughDeclarationContract()

    class _NoAttr:
        pass

    with pytest.raises(AttributeError):
        c.applies_to(_NoAttr())


def test_post_emission_check_raises_on_divergence() -> None:
    c = PassThroughDeclarationContract()
    input_row = _row(("a", "b", "c"))
    inputs = PostEmissionInputs(
        plugin=_FakeTransform(),
        node_id="n-1",
        run_id="r",
        row_id="rw",
        token_id="t",
        input_row=input_row,
        static_contract=frozenset({"a", "b", "c"}),
        effective_input_fields=derive_effective_input_fields(input_row),
    )
    outputs = PostEmissionOutputs(emitted_rows=(_row(("a", "c")),))
    with pytest.raises(PassThroughContractViolation) as exc_info:
        c.post_emission_check(inputs, outputs)
    assert exc_info.value.divergence_set == frozenset({"b"})


def test_post_emission_check_empty_emission_raises_when_can_drop_rows_false() -> None:
    c = PassThroughDeclarationContract()
    input_row = _row(("a",))
    inputs = PostEmissionInputs(
        plugin=_FakeTransform(),
        node_id="n-1",
        run_id="r",
        row_id="rw",
        token_id="t",
        input_row=input_row,
        static_contract=frozenset(),
        effective_input_fields=derive_effective_input_fields(input_row),
    )
    with pytest.raises(PassThroughContractViolation):
        c.post_emission_check(inputs, PostEmissionOutputs(emitted_rows=()))


def test_post_emission_check_empty_emission_is_noop_when_can_drop_rows_true() -> None:
    c = PassThroughDeclarationContract()
    input_row = _row(("a",))
    plugin = _FakeTransform()
    plugin.can_drop_rows = True
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id="n-1",
        run_id="r",
        row_id="rw",
        token_id="t",
        input_row=input_row,
        static_contract=frozenset(),
        effective_input_fields=derive_effective_input_fields(input_row),
    )
    c.post_emission_check(inputs, PostEmissionOutputs(emitted_rows=()))


def test_verify_pass_through_empty_emission_is_noop_when_effective_input_fields_empty() -> None:
    verify_pass_through(
        input_fields=frozenset(),
        emitted_rows=(),
        can_drop_rows=False,
        static_contract=frozenset(),
        transform_name="Fake",
        transform_node_id="n-1",
        run_id="r",
        row_id="rw",
        token_id="t",
    )


def test_derive_effective_input_fields_crashes_on_missing_contract() -> None:
    """Post-H2 (panel F1): the CALLER derives ``effective_input_fields``
    once and passes it in the bundle. A PipelineRow without a contract is
    a framework bug that surfaces at ``derive_effective_input_fields`` —
    not inside the contract's method body.

    The B2 regression coverage moved to the caller-side helper, which
    raises FrameworkBugError (Tier-1) with the same "input row has no
    contract" message the inline check used pre-H2.
    """

    class _BadRow:
        contract = None

    with pytest.raises(FrameworkBugError, match="input row has no contract"):
        derive_effective_input_fields(_BadRow())


def test_derive_effective_input_fields_excludes_optional_contract_fields_missing_from_payload() -> None:
    """Effective input fields are the row's exposed keys, not all contract fields.

    Coalesce/fan-in paths can carry optional branch fields in the contract while
    omitting them from a particular row's payload. The declaration-contract
    layer must treat those fields as absent at runtime.
    """
    contract = SchemaContract(
        mode="OBSERVED",
        fields=(
            FieldContract(
                normalized_name="required_field",
                original_name="required_field",
                python_type=str,
                required=True,
                source="declared",
                nullable=False,
            ),
            FieldContract(
                normalized_name="optional_branch_field",
                original_name="optional_branch_field",
                python_type=str,
                required=False,
                source="declared",
                nullable=False,
            ),
        ),
        locked=True,
    )
    input_row = PipelineRow({"required_field": "v"}, contract)

    assert derive_effective_input_fields(input_row) == frozenset({"required_field"})


def test_post_emission_check_preserves_orchestrationinvarianterror_on_missing_node_id() -> None:
    """B2 regression: transform.node_id=None must raise OrchestrationInvariantError."""
    c = PassThroughDeclarationContract()
    plugin = _FakeTransform()
    plugin.node_id = None
    input_row = _row(("a",))
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id="",
        run_id="r",
        row_id="rw",
        token_id="t",
        input_row=input_row,
        static_contract=frozenset(),
        effective_input_fields=derive_effective_input_fields(input_row),
    )
    with pytest.raises(OrchestrationInvariantError):
        c.post_emission_check(inputs, PostEmissionOutputs(emitted_rows=(_row(("a",)),)))


def test_negative_example_fires_violation() -> None:
    c = PassThroughDeclarationContract()
    bundle = c.negative_example()
    method = getattr(c, bundle.site.value)
    with pytest.raises(PassThroughContractViolation):
        method(*bundle.args)


def test_contract_claims_both_dispatch_sites() -> None:
    """H2 regression: PassThroughDeclarationContract claims post_emission_check
    AND batch_flush_check. Registered in EXPECTED_CONTRACT_SITES under both."""
    from elspeth.contracts.declaration_contracts import contract_sites

    c = PassThroughDeclarationContract()
    sites = contract_sites(c)
    assert sites == frozenset({"post_emission_check", "batch_flush_check"})


def test_batch_flush_check_raises_on_divergence() -> None:
    """The batch-flush site uses BatchFlushInputs; contract's logic parallels
    post_emission_check but reads ``effective_input_fields`` as the caller-
    computed intersection of every buffered token's contract."""
    from elspeth.contracts.declaration_contracts import BatchFlushInputs, BatchFlushOutputs

    c = PassThroughDeclarationContract()
    token_row = _row(("a", "b", "c"))
    inputs = BatchFlushInputs(
        plugin=_FakeTransform(),
        node_id="n-1",
        run_id="r",
        row_id="rw",
        token_id="t",
        buffered_tokens=(token_row,),
        static_contract=frozenset({"a", "b", "c"}),
        effective_input_fields=frozenset({"a", "b", "c"}),
    )
    outputs = BatchFlushOutputs(emitted_rows=(_row(("a", "c")),))
    with pytest.raises(PassThroughContractViolation) as exc_info:
        c.batch_flush_check(inputs, outputs)
    assert exc_info.value.divergence_set == frozenset({"b"})


def test_dispatcher_zero_emission_with_empty_effective_input_fields_raises_only_can_drop_rows(
    _isolated_registry,
) -> None:
    register_declaration_contract(PassThroughDeclarationContract())
    register_declaration_contract(CanDropRowsContract())

    plugin = _FakeTransform()
    input_row = _empty_row()
    inputs = PostEmissionInputs(
        plugin=plugin,
        node_id=plugin.node_id,
        run_id="r",
        row_id="rw",
        token_id="t",
        input_row=input_row,
        static_contract=frozenset(),
        effective_input_fields=frozenset(),
    )

    with pytest.raises(UnexpectedEmptyEmissionViolation) as exc_info:
        run_post_emission_checks(inputs=inputs, outputs=PostEmissionOutputs(emitted_rows=()))

    assert exc_info.value.payload["emitted_count"] == 0


def test_batch_flush_dispatcher_zero_emission_with_empty_effective_input_fields_raises_only_can_drop_rows(
    _isolated_registry,
) -> None:
    register_declaration_contract(PassThroughDeclarationContract())
    register_declaration_contract(CanDropRowsContract())

    plugin = _FakeTransform()
    input_row = _empty_row()
    inputs = BatchFlushInputs(
        plugin=plugin,
        node_id=plugin.node_id,
        run_id="r-batch",
        row_id="rw-batch",
        token_id="t-batch",
        buffered_tokens=(input_row,),
        static_contract=frozenset(),
        effective_input_fields=frozenset(),
    )

    with pytest.raises(UnexpectedEmptyEmissionViolation) as exc_info:
        run_batch_flush_checks(inputs=inputs, outputs=BatchFlushOutputs(emitted_rows=()))

    assert exc_info.value.payload["emitted_count"] == 0
