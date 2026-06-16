"""Metadata for the validation-theatre rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "contract_invariants.validation_theatre"
SUGGESTION = (
    "Do not return success from a validation/check/verify function when the comment says the check was skipped or deferred. "
    "Return a distinct deferred/skipped result, fail closed, or move the success return after the validation actually runs."
)

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Validation theatre",
    description="Validation-like functions must not return success from branches documented as skipped or deferred validation.",
    severity=Severity.ERROR,
    category=Category.CONTRACT_INVARIANTS,
    cwe=("CWE-754",),
    scope=RuleScope.INCREMENTAL,
    path_filter=r".*\.py$",
    examples_violation_count=1,
    examples_clean_count=2,
)
