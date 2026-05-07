"""N2 Layer A harness — every registered DeclarationContract MUST have a
``positive_example_does_not_apply`` scenario; ``applies_to`` MUST return False
on it; and (belt-and-suspenders) ``runtime_check`` MUST NOT raise the
contract's violation on it.

Closes the "negative_example coverage illusion" identified by the H2 panel
(systems-thinker R2 loop): the existing ``negative_example`` harness tests
the positive case (contract fires when it should). Before N2 there was no
coverage for the complementary case (contract does NOT fire when
preconditions are absent). A contract with a buggy ``applies_to`` that
returned True for the wrong plugin kind would fire in the wrong context and
mis-attribute the violation — the Landscape legal record would carry a
violation from contract X against a plugin X was never designed to cover.

See issue elspeth-50509ed2bc and the panel synthesis comment on
elspeth-425047a599.
"""

from __future__ import annotations

import pytest

import elspeth.engine.executors.declaration_contract_bootstrap  # noqa: F401
from elspeth.contracts.declaration_contracts import (
    DeclarationContractViolation,
    positive_example_does_not_apply_bundles,
    registered_declaration_contracts,
)


@pytest.mark.parametrize(
    "contract",
    list(registered_declaration_contracts()),
    ids=lambda c: c.name,
)
def test_positive_example_does_not_apply_returns_non_applying_scenario(contract) -> None:
    """applies_to MUST return False on the contract's non-fire scenario."""
    for bundle in positive_example_does_not_apply_bundles(contract):
        # First positional arg of every bundle is the "inputs" carrying .plugin.
        inputs = bundle.args[0]
        assert not contract.applies_to(inputs.plugin), (
            f"Contract {contract.name!r}'s positive_example_does_not_apply() "
            f"returned a scenario where applies_to() is True. The scenario MUST "
            f"be one where applies_to is False — the whole point of the harness "
            f"is to prove the contract correctly excludes plugins outside its "
            f"scope. A contract claiming every plugin is in-scope (applies_to "
            f"always True) is a design smell at best and an audit-integrity "
            f"Tier-1 bug at worst."
        )


@pytest.mark.parametrize(
    "contract",
    list(registered_declaration_contracts()),
    ids=lambda c: c.name,
)
def test_runtime_check_does_not_fire_on_non_apply_scenario(contract) -> None:
    """Belt-and-suspenders: if ``runtime_check`` is invoked despite
    ``applies_to=False`` (e.g. a future dispatcher refactor bypasses the
    short-circuit), the contract MUST NOT raise its own violation on the
    declared non-fire scenario.

    The dispatcher short-circuits on ``applies_to=False`` today, so this
    assertion is defence-in-depth — but the audit trail's claim "no
    violation raised = plugin behaviour was compliant" is only as strong as
    every path that could raise it. The per-contract declaration of a known
    non-fire scenario is the explicit contract between contract author and
    harness that "this scenario should never cause me to fire."
    """
    for bundle in positive_example_does_not_apply_bundles(contract):
        method = getattr(contract, bundle.site.value)
        try:
            method(*bundle.args)
        except DeclarationContractViolation as exc:
            pytest.fail(
                f"Contract {contract.name!r}'s runtime_check raised "
                f"{type(exc).__name__} on its own declared non-fire scenario. "
                f"Either positive_example_does_not_apply is mis-specified (the "
                f"scenario is actually a fire case), or the contract's "
                f"runtime_check lacks the pre-filter that applies_to encodes. "
                f"Contracts with a false-positive applies_to branch silently "
                f"mis-attribute violations to the wrong plugin kind — a Tier-1 "
                f"audit-integrity failure."
            )
