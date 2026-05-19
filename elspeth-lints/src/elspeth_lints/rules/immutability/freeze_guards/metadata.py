"""Metadata for the freeze-guards immutability rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "immutability.freeze_guards"
RULES: dict[str, dict[str, str]] = {
    "FG1": {
        "name": "bare-mapping-proxy",
        "description": "Bare MappingProxyType wrap in __post_init__ -- shallow freeze misses nested mutables",
        "remediation": "Use deep_freeze() instead of MappingProxyType(dict(self.x)) for recursive immutability",
    },
    "FG2": {
        "name": "isinstance-freeze-guard",
        "description": "isinstance() type guard used to conditionally skip freezing in __post_init__",
        "remediation": "Use deep_freeze() which is idempotent on already-frozen values -- no guard needed",
    },
    "FG3": {
        "name": "missing-freeze-guard",
        "description": "Frozen dataclass with container-typed fields lacks freeze_fields/deep_freeze in __post_init__",
        "remediation": "Add __post_init__ with freeze_fields(self, 'field1', 'field2', ...) for container fields",
    },
}
RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Freeze guards",
    description="Frozen dataclasses with container fields must use recursive freeze guards, not shallow or conditional patterns.",
    severity=Severity.ERROR,
    category=Category.IMMUTABILITY,
    cwe=("CWE-471",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r".*\.py$",
    examples_violation_count=3,
    examples_clean_count=3,
)
