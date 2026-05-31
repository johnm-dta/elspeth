"""Metadata for the guard-symmetry rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "audit_evidence.guard_symmetry"
LEGACY_RULE_ID = "GS1"
SUGGESTION = "Add AuditIntegrityError validation to the loader's load() method for Tier 1 read-side integrity"

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Guard symmetry",
    description="Dataclasses with post-init validation must have matching read-side AuditIntegrityError loader guards.",
    severity=Severity.ERROR,
    category=Category.AUDIT_EVIDENCE,
    cwe=("CWE-754",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r".*\.py$",
    examples_violation_count=1,
    examples_clean_count=1,
)
