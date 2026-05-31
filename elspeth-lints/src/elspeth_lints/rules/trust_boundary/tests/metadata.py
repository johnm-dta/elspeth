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
RULE_FILE_MISSING = "R_TB_TESTS_FILE_MISSING"
RULE_PARSE_ERROR = "R_TB_TESTS_PARSE_ERROR"
RULE_FUNCTION_MISSING = "R_TB_TESTS_FUNCTION_MISSING"
RULE_TOO_LARGE = "R_TB_TESTS_FILE_TOO_LARGE"
RULE_INVARIANT_MISMATCH = "R_TB_TESTS_INVARIANT_MISMATCH"
RULE_INPUT_IRRELEVANT = "R_TB_TESTS_IRRELEVANT_INPUT"
RULE_FINGERPRINT_MISSING = "R_TB_TESTS_FINGERPRINT_MISSING"
RULE_FINGERPRINT_MISMATCH = "R_TB_TESTS_FINGERPRINT_MISMATCH"

SUGGESTION_MISSING = (
    "Every @trust_boundary must carry a test_ref pointing to a pytest node "
    "whose own body contains a raising assertion for malformed-input "
    "rejection and calls the decorated function through source_param. Add "
    "test_ref='tests/.../test_module.py::test_function' plus the current "
    "test_fingerprint."
)

SUGGESTION_NOTFOUND = (
    "test_ref nodeid does not resolve. Verify the file path (relative to the "
    "repository root) and the test function name. Format: "
    "'tests/path/to/test_file.py::test_function_name'."
)

SUGGESTION_FILE_MISSING = (
    "test_ref file path does not resolve inside the repository root. Verify "
    "the file path is relative to the repository root and does not use path "
    "traversal."
)

SUGGESTION_PARSE_ERROR = (
    "test_ref points to a file that cannot be parsed or read by the analyzer. "
    "Fix the test file syntax/readability before relying on it as a "
    "trust-boundary honesty gate."
)

SUGGESTION_FUNCTION_MISSING = (
    "test_ref file exists, but the named pytest function or method does not. "
    "Verify the nodeid suffix, including class segments and parametrized test "
    "ids."
)

SUGGESTION_TOO_LARGE = (
    "test_ref points to an oversized test file. Keep trust-boundary honesty "
    "references narrow enough for the rule to read bounded source files."
)

SUGGESTION_WEAK = (
    "The referenced test does not assert on raising behaviour. The honesty "
    "gate requires the test body to contain at least one pytest.raises(...) "
    "context manager, pytest.raises(...) call, or unittest assertRaises(...) call."
)

SUGGESTION_INVARIANT_MISMATCH = (
    "Make the decorator invariant and referenced test agree on the exception "
    "type. Either update invariant='raises <ExceptionType> ...' to match the "
    "tested failure mode, or update the referenced test to assert the documented "
    "exception."
)

SUGGESTION_INPUT_IRRELEVANT = (
    "The referenced test must directly call the decorated function inside the "
    "raising assertion and supply the declared source_param by keyword or "
    "signature position. Do not rely on helper indirection or a test that raises "
    "without exercising the boundary subject."
)

SUGGESTION_FINGERPRINT_MISSING = (
    "Add test_fingerprint with the current canonical AST fingerprint emitted "
    "by this rule. The fingerprint binds test_ref to the specific test body the "
    "decorator was reviewed against."
)

SUGGESTION_FINGERPRINT_MISMATCH = (
    "Refresh test_fingerprint only after reviewing the changed test body still "
    "exercises the decorated trust-boundary invariant. A mismatch means the "
    "nodeid still resolves, but the test content drifted."
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
        "node whose own body asserts direct raising behaviour, calls the "
        "decorated symbol through source_param, and matches the exception "
        "type declared in invariant when one is present."
    ),
    severity=Severity.ERROR,
    category=Category.TRUST_TIER,
    cwe=("CWE-1059", "CWE-754"),
    scope=RuleScope.WHOLE_REPO,
    path_filter=r".*\.py$",
    examples_violation_count=3,
    examples_clean_count=1,
)
