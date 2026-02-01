# Test Quality Review: test_retry_policy.py

## Summary
This test file contains only 3 trivial smoke tests for RetryPolicy TypedDict and RetryConfig.from_policy(). It is severely incomplete - missing validation tests, edge cases, type safety verification, and any tests that verify the core retry semantics documented in CLAUDE.md ("(run_id, row_id, transform_seq, attempt) is unique" and "Each attempt recorded separately"). The file name suggests it should test retry policy behavior, but it only tests TypedDict importability.

## Poorly Constructed Tests

### Test: test_retry_policy_importable (line 8)
**Issue**: Trivial smoke test that provides zero value - TypedDict imports are validated by mypy
**Evidence**:
```python
def test_retry_policy_importable(self) -> None:
    """RetryPolicy should be importable from contracts."""
    from elspeth.contracts import RetryPolicy

    policy: RetryPolicy = {
        "max_attempts": 3,
        "base_delay": 1.0,
    }
    assert policy["max_attempts"] == 3
```
This test verifies that a dict can be created and that dict keys can be accessed - both guaranteed by Python itself.

**Fix**: Delete this test entirely. Import validity is checked by mypy. If you want to test TypedDict behavior, test the actual type checking (e.g., with reveal_type or runtime validation via typeguard).

**Priority**: P3 (delete on cleanup pass)

### Test: test_retry_config_from_policy_with_typed_dict (line 18)
**Issue**: Incomplete happy-path test - doesn't verify all fields map correctly
**Evidence**: Test creates a policy with 4 fields but the RetryPolicy TypedDict has exactly 4 fields. This is just testing that the factory method works at all, not that it handles edge cases.

**Fix**: Expand to property-based test using Hypothesis to verify from_policy() handles all valid combinations of optional fields. Test with missing fields, partial specifications, invalid values.

**Priority**: P2 (missing edge case coverage)

### Test: test_retry_policy_partial (line 36)
**Issue**: Hardcoded default values make test brittle and fail to document source of truth
**Evidence**:
```python
# Defaults for unspecified fields
assert config.base_delay == 1.0
```
This embeds RetryConfig defaults in the test. If defaults change in the implementation, test breaks even though behavior is correct.

**Fix**: Either:
1. Read defaults from RetryConfig class definition and assert against those, OR
2. Only test that unspecified fields get *some* default, not what that default is

**Priority**: P2 (brittle, will break on valid changes)

## Misclassified Tests

### All tests in this file
**Issue**: File name is `test_retry_policy.py` but tests are for RetryConfig factory methods
**Evidence**: Only 3 tests, all focused on TypedDict importability and RetryConfig.from_policy(). No tests of retry policy *behavior* (backoff, attempt counting, retryability filtering).

**Fix**:
1. Rename file to `test_retry_config_factories.py` since it only tests RetryConfig construction, OR
2. Expand file to actually test retry policy behavior (preferred)

**Priority**: P1 (misleading file organization)

## Infrastructure Gaps

### Gap: No integration with test_retry.py
**Issue**: test_retry.py exists and tests RetryManager behavior, but there's no clear division of responsibility
**Evidence**:
- `test_retry.py` has tests like `test_from_policy_none_returns_no_retry` and `test_from_policy_handles_malformed` (lines 87-107 in test_retry.py)
- `test_retry_policy.py` has `test_retry_policy_partial` which also tests from_policy()
- Duplicate coverage with no clear boundary

**Fix**: Consolidate all RetryConfig tests into one file. Either:
1. Merge test_retry_policy.py into test_retry.py as a nested class `TestRetryConfigFactories`, OR
2. Move all from_policy() tests to test_retry_policy.py and keep test_retry.py focused on RetryManager execution

**Priority**: P1 (organizational debt, confusing test discovery)

### Gap: Missing validation tests
**Issue**: No tests verify that RetryPolicy TypedDict fields have correct types at runtime
**Evidence**: TypedDict with `total=False` means all fields are optional, but from_policy() uses .get() with defaults. No test verifies what happens if someone passes `{"max_attempts": "not an int"}`.

**Fix**: Add tests:
```python
def test_from_policy_wrong_types_handled():
    """from_policy() should handle type coercion or raise clear error."""
    # Currently uses .get() which will pass through wrong types!
    policy = {"max_attempts": "five"}  # str instead of int
    # What should happen? Type error? Coercion? Document and test it.
```

**Priority**: P0 (trust boundary violation per CLAUDE.md - external config is Tier 3)

### Gap: No verification of CLAUDE.md retry semantics
**Issue**: CLAUDE.md specifies "(run_id, row_id, transform_seq, attempt) is unique" and "Each attempt recorded separately" but no tests verify this
**Evidence**: These tests only check RetryConfig construction, not that the retry system actually records attempts with proper keys.

**Fix**: This is likely tested at integration level (test_processor.py or test_orchestrator.py), but if not, add integration tests that:
1. Execute a transform that fails twice then succeeds
2. Query Landscape for attempt records
3. Verify (run_id, row_id, transform_seq, attempt) tuples exist and are unique

**Priority**: P0 (core audit requirement from CLAUDE.md)

### Gap: No property-based testing for from_policy()
**Issue**: from_policy() clamps invalid values (negative max_attempts → 1) but only one test case exists
**Evidence**: Only test_retry_policy_partial tests partial specification. Need to verify:
- Empty dict
- Dict with only max_attempts
- Dict with only base_delay
- Dict with negative values
- Dict with zero values
- Dict with None values
- Dict with float instead of int for max_attempts

**Fix**: Use Hypothesis to generate random RetryPolicy dicts and verify from_policy() never crashes and always produces valid RetryConfig.

**Priority**: P1 (factory method is a trust boundary)

### Gap: No test for from_policy() with all fields None
**Issue**: TypedDict with total=False allows `{"max_attempts": None}` - what happens?
**Evidence**: Code uses `policy.get("max_attempts", 3)` which returns None if key exists with None value, not the default.

**Fix**: Add test:
```python
def test_from_policy_none_values():
    config = RetryConfig.from_policy({"max_attempts": None})
    # Does this crash? Return 3? Return 1? Test and document.
```

**Priority**: P0 (likely a bug - .get() doesn't work how the code assumes)

## Missing Test Categories

### Category: Type safety at boundaries
**Why missing**: No tests verify that from_policy() handles malformed external config correctly per CLAUDE.md Tier 3 (external data is zero trust)

**What's needed**:
- Wrong types (str instead of int)
- Missing required conversions
- Overflow values (sys.maxsize for delays)
- Special float values (inf, -inf, nan should be rejected per CLAUDE.md canonical JSON rules)

**Priority**: P0 (trust boundary per Data Manifesto)

### Category: Round-trip serialization
**Why missing**: RetryPolicy is a TypedDict that might be serialized to/from YAML/JSON in settings files

**What's needed**:
- Verify RetryPolicy can be dumped to JSON and reloaded
- Verify canonical JSON representation (if used in hashing)
- Verify no information loss in serialization

**Priority**: P2 (if retry policies are part of audit trail)

### Category: Integration with RetrySettings
**Why missing**: RetryConfig has from_settings() factory but no tests in this file verify the two config paths (from_policy vs from_settings) produce equivalent configs

**What's needed**:
```python
def test_policy_and_settings_equivalent():
    """Verify from_policy() and from_settings() produce same config for same values."""
    policy: RetryPolicy = {"max_attempts": 5, "base_delay": 2.0}
    settings = RetrySettings(max_attempts=5, initial_delay_seconds=2.0)

    from_policy_config = RetryConfig.from_policy(policy)
    from_settings_config = RetryConfig.from_settings(settings)

    assert from_policy_config.max_attempts == from_settings_config.max_attempts
    # etc for all fields
```

**Priority**: P2 (prevents config drift between two config paths)

## Positive Observations
- Tests use type annotations correctly (`-> None`)
- Test names are descriptive
- Tests are isolated (no shared state)
- Docstrings explain what each test verifies

## Recommended Actions

### Immediate (P0)
1. Add validation tests for from_policy() with wrong types (trust boundary violation)
2. Test from_policy() behavior when dict contains `None` values (likely a bug)
3. Verify retry attempt uniqueness semantics from CLAUDE.md are tested somewhere (if not here, document where)

### Short-term (P1)
1. Consolidate retry tests - merge test_retry_policy.py into test_retry.py or vice versa
2. Add property-based tests for from_policy() edge cases

### Medium-term (P2)
1. Fix brittle default value assertions
2. Add round-trip serialization tests if retry policies are audited
3. Test equivalence between from_policy() and from_settings()

### Cleanup (P3)
1. Delete test_retry_policy_importable - provides no value
