"""Metadata for the audit-evidence nominal inheritance rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "audit_evidence.nominal_base"
LEGACY_RULE_ID = "AEN1"
SUGGESTION = "Inherit AuditEvidenceBase explicitly, or remove the accidental to_audit_dict method"

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Audit evidence nominal base",
    description="Classes that define to_audit_dict must explicitly inherit AuditEvidenceBase.",
    severity=Severity.ERROR,
    category=Category.AUDIT_EVIDENCE,
    cwe=("CWE-117", "CWE-778"),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r".*\.py$",
    examples_violation_count=1,
    examples_clean_count=1,
)
