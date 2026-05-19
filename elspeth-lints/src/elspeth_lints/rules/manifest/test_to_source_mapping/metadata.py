"""Metadata for the test-to-source mapping inventory rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "manifest.test_to_source_mapping"

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Test-to-source mapping inventory",
    description="Inventories tests that still assert obsolete RowOutcome or token_outcomes outcome-only behavior.",
    severity=Severity.ERROR,
    category=Category.MANIFEST,
    cwe=("CWE-710",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r"^tests/.*\.py$",
    examples_violation_count=1,
    examples_clean_count=1,
)
