"""Metadata for the ``trust_boundary.tier`` honesty-gate rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "trust_boundary.tier"
RULE_INVALID = "TBT1"

SUGGESTION_INVALID = (
    "@trust_boundary only applies to tier=3 (external-data) boundaries. "
    "Tier-1 (audit/landscape) and Tier-2 (post-source pipeline) invariants must "
    "crash on anomaly per the data manifesto, not suppress lint findings. "
    "Either change tier=3 or remove the decorator."
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
