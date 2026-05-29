"""Metadata for the ``trust_boundary.tier`` honesty-gate rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "trust_boundary.tier"
RULE_INVALID = "TBT1"
# TBT2 (NONLITERAL) is the self-enforcement code emitted when this rule
# sees a non-literal kwarg on the decorator. It deliberately overlaps with
# ``trust_tier.tier_model``'s R_TB_NONLITERAL: each honesty rule must
# self-enforce literal-only kwargs so that suppressing tier_model on a
# file cannot grant honesty-gate immunity here. See epic
# elspeth-2ed3bb0f7d, ticket elspeth-1f4634235a (C6-4).
RULE_NONLITERAL = "TBT2"

SUGGESTION_INVALID = (
    "@trust_boundary only applies to tier=3 (external-data) boundaries. "
    "Tier-1 (audit/landscape) and Tier-2 (post-source pipeline) invariants must "
    "crash on anomaly per the data manifesto, not suppress lint findings. "
    "Either change tier=3 or remove the decorator."
)

SUGGESTION_NONLITERAL = (
    "@trust_boundary kwargs must be static literals. The tier honesty gate "
    "cannot verify a name reference, call, or comprehension because the "
    "static analyzer cannot prove the value at decoration time. Replace the "
    "kwarg value with a literal integer / string / tuple, or remove the "
    "decorator."
)

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Trust-boundary tier",
    description=(
        "@trust_boundary tier kwarg must be the literal integer 3. The decorator "
        "is a Tier-3 external-data boundary marker; Tier-1 and Tier-2 must crash."
    ),
    severity=Severity.ERROR,
    category=Category.TRUST_TIER,
    cwe=("CWE-754",),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r".*\.py$",
    examples_violation_count=1,
    examples_clean_count=1,
)
