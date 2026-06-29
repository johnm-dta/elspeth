"""Metadata for the no-new-bespoke-CI-enforcers meta rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "meta.no-new-bespoke-cicd-enforcer"
MANIFEST_PATH = "config/cicd/lint_migration_status.yaml"
RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="No new bespoke CI enforcers",
    description=(
        "New ELSPETH-specific CI checks must be elspeth-lints rules, not new "
        "scripts/cicd/enforce_*.py files, workflow-invoked scripts/cicd/*.py "
        "policy scripts, or untracked legacy inventory scripts."
    ),
    severity=Severity.ERROR,
    category=Category.MANIFEST,
    cwe=("CWE-1059",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r"^(scripts/cicd/.*\.py|\.github/workflows/.*\.ya?ml)$",
    examples_violation_count=1,
    examples_clean_count=1,
)
