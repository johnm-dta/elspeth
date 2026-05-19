"""Metadata for the source-symbol inventory rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "manifest.symbol_inventory"

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Source symbol inventory",
    description="Inventories stale source symbols and brittle outcome/path string checks.",
    severity=Severity.ERROR,
    category=Category.MANIFEST,
    cwe=("CWE-710",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r"^src/elspeth/.*\.py$",
    examples_violation_count=1,
    examples_clean_count=1,
)
