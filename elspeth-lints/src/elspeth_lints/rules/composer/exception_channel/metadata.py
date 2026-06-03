"""Metadata for the composer exception-channel rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "composer.exception_channel"
LEGACY_RULE_ID = "CEC1"
SUGGESTION = "raise ToolArgumentError(argument=..., expected=..., actual_type=...) from exc, or catch locally and return _failure_result"

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Composer exception channel",
    description="Composer tool handlers must use ToolArgumentError for LLM-argument failures.",
    severity=Severity.ERROR,
    category=Category.COMPOSER,
    cwe=("CWE-754", "CWE-396"),
    scope=RuleScope.INCREMENTAL,
    path_filter=r"^web/composer/tools\.py$",
    examples_violation_count=1,
    examples_clean_count=1,
)
