"""Metadata for the GraphValidationError attribution rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "audit_evidence.gve_attribution"
LEGACY_RULE_ID = "GA1"
SUGGESTION = "Add component_id=<node_id> to the raise site, or allowlist genuinely structural graph errors with no single node at fault"

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="GraphValidationError attribution",
    description="GraphValidationError raise sites must include component_id for UI attribution unless structural.",
    severity=Severity.ERROR,
    category=Category.AUDIT_EVIDENCE,
    cwe=("CWE-754",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r".*\.py$",
    examples_violation_count=3,
    examples_clean_count=1,
)
