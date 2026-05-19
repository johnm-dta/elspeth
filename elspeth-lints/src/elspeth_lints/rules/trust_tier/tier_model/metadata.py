"""Metadata for the trust-tier model rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "trust_tier.tier_model"

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Trust-tier model",
    description="Detects defensive patterns and upward imports that hide trust-boundary or layer-contract bugs.",
    severity=Severity.ERROR,
    category=Category.TRUST_TIER,
    cwe=("CWE-20", "CWE-1188"),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r".*\.py$",
    examples_violation_count=1,
    examples_clean_count=1,
)
