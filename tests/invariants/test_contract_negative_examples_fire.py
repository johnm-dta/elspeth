"""For every registered DeclarationContract, runtime_check(negative_example())
MUST raise the contract's violation AND the raised violation's payload MUST
cover every required key the contract's payload_schema declares.

- Layer 1 — dormant-runtime_check invariant (reviewer B6/F-7).
- Layer 2 (N2 Layer B, issue elspeth-50509ed2bc) — payload-representativeness:
  a minimal-payload negative_example can pass H5 Layer 1 construction-time
  validation for an empty schema, but if the contract declares required
  keys the negative_example MUST exercise them so the harness proves the
  audit-record shape matches what production would persist.
"""

from __future__ import annotations

import pytest

import elspeth.engine.executors.declaration_contract_bootstrap  # noqa: F401
from elspeth.contracts.declaration_contracts import (
    DeclarationContractViolation,
    ExampleBundle,
    negative_example_bundles,
    registered_declaration_contracts,
    resolve_payload_schema_key_sets,
)
from elspeth.contracts.errors import PluginContractViolation

_CONTRACT_VIOLATION_TYPES = (DeclarationContractViolation, PluginContractViolation)


def _assert_exception_passed_through_dispatch_method(exc: BaseException, *, contract_name: str, bundle: ExampleBundle) -> None:
    method_name = bundle.site.value
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_name == method_name:
            return
        tb = tb.tb_next
    raise AssertionError(
        f"Contract {contract_name!r}'s negative_example raised {type(exc).__name__}, "
        f"but the traceback never entered dispatch method {method_name!r}. "
        "The invariant must prove the contract body fired, not just accept a setup-time exception."
    )


@pytest.mark.parametrize(
    "contract",
    list(registered_declaration_contracts()),
    ids=lambda c: c.name,
)
def test_negative_example_fires_violation(contract) -> None:
    """Dormant-dispatch-method invariant (ADR-010 §Decision 3 / reviewer B6/F-7).

    If a registered contract's decorated dispatch method silently returns
    None on its own negative_example, the framework's runtime VAL is
    disabled for that contract without any loud failure. This test is the
    last line of defence.

    Post-H2 (ADR-010 §Semantics amendment): the harness reads the
    site-tagged ``ExampleBundle`` and invokes the method named by
    ``bundle.site.value``. It accepts only the DCV hierarchy and the
    pre-ADR-010 PluginContractViolation hierarchy (PassThroughContractViolation);
    generic RuntimeError must not satisfy the invariant.
    """
    for bundle in negative_example_bundles(contract):
        method = getattr(contract, bundle.site.value)
        with pytest.raises(_CONTRACT_VIOLATION_TYPES) as exc_info:
            method(*bundle.args)
        assert exc_info.value is not None, (
            f"Contract {contract.name!r}'s runtime_check did not raise on its own negative_example — VAL is dormant for this contract."
        )
        violation = exc_info.value
        _assert_exception_passed_through_dispatch_method(violation, contract_name=contract.name, bundle=bundle)
        if isinstance(violation, DeclarationContractViolation):
            # C1 (plan-review B1-amended): reject bare-base raises for DCV-family
            # contracts so triage SQL can filter by exception_type. The isinstance
            # guard skips this check for pre-ADR-010 hierarchies (e.g.
            # PassThroughContractViolation via PluginContractViolation) which have
            # their own purpose-built subclass and carry no bare-base risk.
            assert type(violation) is not DeclarationContractViolation, (
                f"Contract {contract.name!r} raised the bare base "
                f"DeclarationContractViolation rather than its declared subclass. "
                f"Each contract MUST raise a purpose-built subclass of "
                f"DeclarationContractViolation so triage SQL can filter by "
                f"exception_type. Bare-base raises serialize to "
                f"exception_type = 'DeclarationContractViolation', which is "
                f"unfiltered in triage SQL and masks attribution at the audit "
                f"boundary."
            )


@pytest.mark.parametrize(
    "contract",
    list(registered_declaration_contracts()),
    ids=lambda c: c.name,
)
def test_negative_example_payload_covers_required_schema_keys(contract) -> None:
    """Payload-representativeness invariant (issue elspeth-50509ed2bc / N2 Layer B).

    Same site-tagged-bundle pattern as the dormant harness above; asserts
    in addition that the raised DCV subclass's payload covers every
    ``Required`` key declared on its ``payload_schema``.
    """
    for bundle in negative_example_bundles(contract):
        method = getattr(contract, bundle.site.value)
        with pytest.raises(_CONTRACT_VIOLATION_TYPES) as exc_info:
            method(*bundle.args)

        violation = exc_info.value
        _assert_exception_passed_through_dispatch_method(violation, contract_name=contract.name, bundle=bundle)
        if not isinstance(violation, DeclarationContractViolation):
            # PassThroughContractViolation predates the ADR-010 DeclarationContract
            # Violation hierarchy (kept as a dedicated subclass for triage-SQL
            # continuity per ADR-010 §Consequences line 86). It does not carry the
            # payload_schema + __init__-time Layer 1 validation, so this layer of
            # representativeness cannot be asserted against it today. Phase 2B
            # transform adopters register native DeclarationContractViolation
            # subclasses and will be covered here automatically.
            pytest.skip(
                f"Contract {contract.name!r} raises {type(violation).__name__} — a "
                f"pre-ADR-010-framework exception class (PluginContractViolation "
                f"subclass). N2 Layer B payload-representativeness applies to "
                f"DeclarationContractViolation subclasses only; the transitional "
                f"PassThroughContractViolation does not have a payload_schema "
                f"enforced at __init__ time, so the invariant is vacuously "
                f"satisfied here."
            )

        # Violations raised outside the dispatcher have no contract_name
        # attached (C4 read-only property). The payload itself is populated at
        # __init__ time independent of dispatcher attribution, so we inspect it
        # directly rather than through to_audit_dict() (which requires the
        # attribute).
        required_keys, _optional_keys = resolve_payload_schema_key_sets(type(violation).payload_schema)
        actual_keys = frozenset(violation.payload.keys())
        missing = required_keys - actual_keys
        assert not missing, (
            f"Contract {contract.name!r}'s negative_example raised a "
            f"{type(violation).__name__} whose payload is missing required "
            f"keys declared on {type(violation).payload_schema.__name__}: "
            f"{sorted(missing)!r}. The payload actually carries: "
            f"{sorted(actual_keys)!r}. H5 Layer 1 SHOULD have rejected this at "
            f"construction — if the violation was raised, H5 Layer 1 has a "
            f"regression. Otherwise the negative_example is too narrow: it "
            f"triggers the violation on a payload that happens to satisfy "
            f"__init__ but does not cover the shape an auditor expects to see "
            f"in the Landscape legal record."
        )
