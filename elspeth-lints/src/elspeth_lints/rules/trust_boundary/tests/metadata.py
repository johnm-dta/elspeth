"""Metadata for the ``trust_boundary.tests`` honesty-gate rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "trust_boundary.tests"
RULE_MISSING = "TBE1"
RULE_NOTFOUND = "TBE2"
RULE_WEAK = "TBE3"
# TBE4 (NONLITERAL) is the self-enforcement code emitted when this rule
# sees a non-literal kwarg on the decorator. It deliberately overlaps with
# ``trust_tier.tier_model``'s R_TB_NONLITERAL: each honesty rule must
# self-enforce literal-only kwargs so that suppressing tier_model on a
# file cannot grant honesty-gate immunity here. See epic
# elspeth-2ed3bb0f7d, ticket elspeth-1f4634235a (C6-4).
RULE_NONLITERAL = "TBE4"

SUGGESTION_MISSING = (
    "Every @trust_boundary must carry a test_ref pointing to a pytest node "
    "that exercises the documented invariant on malformed input. Add "
    "test_ref='tests/.../test_module.py::test_function'."
)

SUGGESTION_NOTFOUND = (
    "test_ref nodeid does not resolve. Verify the file path (relative to the "
    "repository root) and the test function name. Format: "
    "'tests/path/to/test_file.py::test_function_name'."
)

SUGGESTION_WEAK = (
    "The referenced test does not assert on raising behaviour. The honesty "
    "gate requires the test body to contain at least one pytest.raises(...) "
    "context manager, pytest.raises(...) call, or unittest assertRaises(...) "
    "call. Add an assertion that the invariant rejects malformed input."
)

SUGGESTION_NONLITERAL = (
    "@trust_boundary kwargs must be static literals. The tests honesty gate "
    "cannot verify a test_ref nodeid that is a name reference, call, or "
    "comprehension — the analyzer cannot resolve the value to a real pytest "
    "node. Replace the kwarg value with a string literal "
    "('tests/.../test_file.py::test_function'), or remove the decorator."
)

RULE_METADATA = RuleMetadata(
    id=RULE_ID,
    name="Trust-boundary tests",
    description=(
        "@trust_boundary must carry a test_ref pointing to a real pytest "
        "node whose body asserts on raising behaviour. The decorator's "
        "invariant claim is unverifiable without a behavioural test."
    ),
    severity=Severity.ERROR,
    category=Category.TRUST_TIER,
    cwe=("CWE-1059", "CWE-754"),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r".*\.py$",
    examples_violation_count=1,
    examples_clean_count=1,
)
