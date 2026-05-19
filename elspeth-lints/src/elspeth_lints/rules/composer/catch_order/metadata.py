"""Metadata for the composer catch-order rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "composer.catch_order"
LEGACY_RULE_ID = "CCO1"
SUGGESTION = (
    "Move the subclass except-handler above the supertype handler. Python evaluates except clauses top-to-bottom; catch narrow first."
)

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Composer catch order",
    description="ComposerServiceError subclass handlers must appear before their supertypes in the same try block.",
    severity=Severity.ERROR,
    category=Category.COMPOSER,
    cwe=("CWE-754", "CWE-396"),
    scope=RuleScope.INCREMENTAL,
    path_filter=r"^web/.*\.py$",
    examples_violation_count=1,
    examples_clean_count=1,
)
