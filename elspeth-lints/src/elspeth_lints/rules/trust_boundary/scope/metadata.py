"""Metadata for the ``trust_boundary.scope`` honesty-gate rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "trust_boundary.scope"
RULE_NOPARAM = "TBS1"
RULE_DEAD = "TBS2"

SUGGESTION_NOPARAM = (
    "@trust_boundary(source_param=...) must name an actual parameter of the "
    "decorated function. Update source_param to a real parameter name, or "
    "remove the decorator if no external-data parameter exists."
)

SUGGESTION_DEAD = (
    "@trust_boundary(source_param=...) names a parameter the function body "
    "never reads. The decorator is structurally inert — its suppression scope "
    "would cover nothing. Either drop the decorator, change source_param to "
    "the parameter the function actually treats as external data, or extend "
    "the function body to read from the declared boundary."
)

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Trust-boundary scope",
    description=(
        "@trust_boundary source_param must name a function parameter that "
        "the body actually reads from. Stops drive-by decorators on functions "
        "that don't take external data."
    ),
    severity=Severity.ERROR,
    category=Category.TRUST_TIER,
    cwe=("CWE-754",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r".*\.py$",
    examples_violation_count=1,
    examples_clean_count=1,
)
