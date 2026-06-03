"""Metadata for the ``trust_boundary.scope`` honesty-gate rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "trust_boundary.scope"
RULE_NOPARAM = "TBS1"
RULE_DEAD = "TBS2"
# TBS3 (NONLITERAL) is the self-enforcement code emitted when this rule
# sees a non-literal kwarg on the decorator. It deliberately overlaps with
# ``trust_tier.tier_model``'s R_TB_NONLITERAL: each honesty rule must
# self-enforce literal-only kwargs so that suppressing tier_model on a
# file cannot grant honesty-gate immunity here. See epic
# elspeth-2ed3bb0f7d, ticket elspeth-1f4634235a (C6-4).
RULE_NONLITERAL = "TBS3"

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

SUGGESTION_NONLITERAL = (
    "@trust_boundary kwargs must be static literals. The scope honesty gate "
    "cannot verify a name reference, call, or comprehension because the "
    "static analyzer cannot prove the value at decoration time. Replace the "
    "kwarg value with a string / tuple / int literal, or remove the decorator."
)

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Trust-boundary scope",
    description=(
        "@trust_boundary source_param must name a function parameter that "
        "the body actually reads from. Stops drive-by decorators on functions "
        "that don't take a locally scoped boundary parameter; the source "
        "description remains reviewer-facing documentation, not an external "
        "call-graph proof."
    ),
    severity=Severity.ERROR,
    category=Category.TRUST_TIER,
    cwe=("CWE-754",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r".*\.py$",
    examples_violation_count=2,
    examples_clean_count=1,
)
