"""Metadata for the plugin hash declaration rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "plugin_contract.plugin_hashes"
RULE_PH1 = "PH1"
RULE_PH2 = "PH2"
RULE_PH3 = "PH3"
SUGGESTION_PH1 = "Declare a non-placeholder plugin_version on every plugin class."
SUGGESTION_PH2 = "Declare source_file_hash on every plugin class and keep it synchronized with the source file."
SUGGESTION_PH3 = "Run the plugin hash fixer to refresh the stale source_file_hash declaration."

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Plugin hash declarations",
    description="Plugin classes must declare plugin_version and matching source_file_hash values.",
    severity=Severity.ERROR,
    category=Category.PLUGIN_CONTRACT,
    cwe=("CWE-1078",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r"plugins/.+\.py$",
    examples_violation_count=3,
    examples_clean_count=1,
)
