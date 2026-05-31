"""Metadata for the plugin options metadata rule."""

from __future__ import annotations

from pathlib import Path

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "plugin_contract.options_metadata"
ALLOWLIST_PATH = Path("config/cicd/enforce_options_metadata/allowlist.yaml")
RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Plugin options metadata",
    description="Plugin configuration fields must expose title and description metadata for catalog and composer surfaces.",
    severity=Severity.ERROR,
    category=Category.PLUGIN_CONTRACT,
    cwe=("CWE-1059",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r"^src/elspeth/plugins/.*\.py$",
    examples_violation_count=3,
    examples_clean_count=3,
)
