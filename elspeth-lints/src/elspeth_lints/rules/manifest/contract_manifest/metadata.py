"""Metadata for the declaration-contract manifest rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "manifest.contract_manifest"
RULE_MC1 = "MC1"
RULE_MC2 = "MC2"
RULE_MC3A = "MC3a"
RULE_MC3B = "MC3b"
RULE_MC3C = "MC3c"
SUGGESTION = "Keep EXPECTED_CONTRACT_SITES, register_declaration_contract(...) calls, and @implements_dispatch_site markers in sync."

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Declaration contract manifest parity",
    description=(
        "Prevents drift between EXPECTED_CONTRACT_SITES, declaration contract registrations, and @implements_dispatch_site markers."
    ),
    severity=Severity.ERROR,
    category=Category.MANIFEST,
    cwe=("CWE-693",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r"^src/elspeth/.*\.py$|^contracts/.*\.py$",
    examples_violation_count=4,
    examples_clean_count=2,
)
