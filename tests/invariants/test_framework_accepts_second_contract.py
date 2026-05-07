"""Framework-extensibility proof: a real production adopter fits the registry.

ADR-015 rejects ``creates_tokens`` as a production declaration contract. This
file therefore re-points the old "second-shape contract" proof at the first
real Phase 2B production adopter: ``DeclaredOutputFieldsContract``.

The goal is structural evidence, not duplicate behaviour coverage:

- the registry admits a second production contract shape alongside pass-through,
- the dispatcher invokes it on a compliant plugin without raising,
- its violation type remains ``AuditEvidenceBase``-compatible.

Dedicated contract behaviour and round-trip coverage lives in
``tests/unit/engine/test_declared_output_fields_contract.py`` and
``tests/integration/audit/test_declared_output_fields_roundtrip.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.contracts.audit_evidence import AuditEvidenceBase
from elspeth.contracts.declaration_contracts import (
    PostEmissionInputs,
    PostEmissionOutputs,
    _attach_contract_name_from_dispatcher,
    _clear_registry_for_tests,
    _restore_registry_snapshot_for_tests,
    _snapshot_registry_for_tests,
    register_declaration_contract,
    registered_declaration_contracts,
)
from elspeth.contracts.errors import DeclaredOutputFieldsViolation
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.engine.executors.declaration_dispatch import run_post_emission_checks
from elspeth.engine.executors.declared_output_fields import DeclaredOutputFieldsContract
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


@pytest.fixture()
def proof_contract_registered() -> Any:
    snapshot = _snapshot_registry_for_tests()
    _clear_registry_for_tests()
    register_declaration_contract(PassThroughDeclarationContract())
    contract = DeclaredOutputFieldsContract()
    register_declaration_contract(contract)
    yield contract
    _restore_registry_snapshot_for_tests(snapshot)


def test_registry_admits_declared_output_fields_proof(proof_contract_registered: Any) -> None:
    names = {c.name for c in registered_declaration_contracts()}
    assert names == {"passes_through_input", "declared_output_fields"}
    assert proof_contract_registered.name == "declared_output_fields"


def test_dispatcher_invokes_declared_output_fields_proof(proof_contract_registered: Any) -> None:
    class _CompliantTransform:
        name = "CompliantDeclaredOutputFields"
        node_id = "dof-ok-1"
        passes_through_input = False
        declared_output_fields = frozenset({"new_a", "new_b"})
        declared_input_fields = frozenset()
        is_batch_aware = False
        _output_schema_config = None

    inputs = PostEmissionInputs(
        plugin=_CompliantTransform(),
        node_id="dof-ok-1",
        run_id="disp-run",
        row_id="disp-row",
        token_id="disp-token",
        input_row=_row(("source",)),
        static_contract=frozenset({"new_a", "new_b"}),
        effective_input_fields=frozenset({"source"}),
    )
    outputs = PostEmissionOutputs(emitted_rows=(_row(("source", "new_a", "new_b")),))

    run_post_emission_checks(inputs=inputs, outputs=outputs)


def test_declared_output_fields_violation_is_audit_evidence() -> None:
    violation = DeclaredOutputFieldsViolation(
        plugin="DeclaredOutputFieldsTransform",
        node_id="n",
        run_id="r",
        row_id="rw",
        token_id="tk",
        payload={
            "declared": ["new_a", "new_b"],
            "violations": [
                {
                    "emitted_index": 0,
                    "runtime_observed": ["new_a"],
                    "missing": ["new_b"],
                }
            ],
        },
        message="declared output fields missing at runtime",
    )

    assert isinstance(violation, AuditEvidenceBase)

    _attach_contract_name_from_dispatcher(violation, "declared_output_fields")
    audit = violation.to_audit_dict()

    assert audit["exception_type"] == "DeclaredOutputFieldsViolation"
    assert audit["contract_name"] == "declared_output_fields"
    assert audit["payload"]["violations"][0]["missing"] == ["new_b"]


def test_negative_example_for_real_production_adopter_fires() -> None:
    contract = DeclaredOutputFieldsContract()
    bundle = contract.negative_example()
    method = getattr(contract, bundle.site.value)
    with pytest.raises(DeclaredOutputFieldsViolation):
        method(*bundle.args)
