"""Metadata for the ``trust_boundary.tests`` honesty-gate rule."""

from __future__ import annotations

from elspeth_lints.core.protocols import Category, RuleMetadata, RuleScope, Severity

RULE_ID = "trust_boundary.tests"
RULE_MISSING = "TBE1"
RULE_NOTFOUND = "TBE2"
RULE_WEAK = "TBE3"

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
