"""Metadata for the frozen-annotations immutability rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "immutability.frozen_annotations"
RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Frozen annotations",
    description="Frozen dataclasses must expose immutable container annotations when post-init freezing changes runtime mutability.",
    severity=Severity.ERROR,
    category=Category.IMMUTABILITY,
    cwe=("CWE-471",),
    scope=RuleScope.INCREMENTAL,
    path_filter=r".*\.py$",
    examples_violation_count=3,
    examples_clean_count=3,
)
