"""Metadata for the tier-1 decoration rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "audit_evidence.tier_1_decoration"
RULE_TDE1 = "TDE1"
RULE_TDE2 = "TDE2"
SUGGESTION_TDE1 = "Add @tier_1_error(reason=...) or a non-empty # TIER-2: justification comment"
SUGGESTION_TDE2 = "Pass caller_module=__name__ literally at every tier_1_error call site"

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Tier-1 decoration",
    description="Exception classes must be marked Tier-1 or explicitly justified as Tier-2.",
    severity=Severity.ERROR,
    category=Category.AUDIT_EVIDENCE,
    cwe=("CWE-117", "CWE-778"),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r".*errors\.py$",
    examples_violation_count=3,
    examples_clean_count=2,
)
