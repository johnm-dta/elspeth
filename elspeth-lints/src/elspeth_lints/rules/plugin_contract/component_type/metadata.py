"""Metadata for the plugin component-type rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "plugin_contract.component_type"
LEGACY_RULE_ID = "CT1"
SUGGESTION = (
    "Set _plugin_component_type: ClassVar[str | None] = 'source' | 'sink' | 'transform' "
    "on the class, or inherit from SourceDataConfig / SinkPathConfig / TransformDataConfig."
)

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Plugin component type",
    description="DataPluginConfig descendants must declare or inherit _plugin_component_type.",
    severity=Severity.ERROR,
    category=Category.PLUGIN_CONTRACT,
    cwe=("CWE-1078",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r".*\.py$",
    examples_violation_count=3,
    examples_clean_count=1,
)
