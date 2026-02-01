# Test Quality Review: test_triggers.py

## Summary

Tests are mostly well-constructed but exhibit critical timing vulnerabilities (sleepy assertions), incomplete mutation testing, missing edge cases, and gaps in state invariant verification. The test suite covers happy paths adequately but lacks property-based testing for a security-critical component that parses user expressions.

## Poorly Constructed Tests

### Test: test_timeout_trigger_reached (line 63)
**Issue**: Sleepy assertion with fixed time delay
**Evidence**:
```python
config = TriggerConfig(timeout_seconds=0.01)
evaluator = TriggerEvaluator(config)
evaluator.record_accept()
time.sleep(0.02)  # 💥 SLEEPY ASSERTION - flaky on slow CI
assert evaluator.should_trigger() is True
```
**Fix**: Use condition-based polling with timeout instead of fixed sleep. However, for unit testing time-based logic, consider mocking `time.monotonic()` to make tests deterministic.
**Priority**: P1

### Test: test_condition_trigger_with_age (line 102)
**Issue**: Sleepy assertion with fixed time delay
**Evidence**:
```python
for _ in range(15):
    evaluator.record_accept()
time.sleep(0.02)  # 💥 SLEEPY ASSERTION
assert evaluator.should_trigger() is True
```
**Fix**: Mock `time.monotonic()` to control timing deterministically.
**Priority**: P1

### Test: test_combined_count_and_timeout_timeout_wins (line 133)
**Issue**: Sleepy assertion with fixed time delay
**Evidence**:
```python
for _ in range(5):
    evaluator.record_accept()
time.sleep(0.02)  # 💥 SLEEPY ASSERTION
result = evaluator.should_trigger()
```
**Fix**: Mock `time.monotonic()` to avoid timing dependency.
**Priority**: P1

### Test: test_batch_age_seconds_property (line 218)
**Issue**: Sleepy assertion with vague inequality check
**Evidence**:
```python
evaluator.record_accept()
time.sleep(0.01)
assert evaluator.batch_age_seconds > 0.0  # Vague - how much greater?
```
**Fix**: Mock `time.monotonic()` to return controlled values, then assert exact batch age. Current test doesn't validate computation correctness.
**Priority**: P1

### Test: test_reset_clears_state (line 185)
**Issue**: Incomplete state verification after reset
**Evidence**:
```python
evaluator.reset()
assert evaluator.should_trigger() is False
assert evaluator.batch_count == 0
# MISSING: assert evaluator.batch_age_seconds == 0.0
# MISSING: assert evaluator.which_triggered() is None
# MISSING: verify _first_accept_time is None (internal but critical)
```
**Fix**: Add assertions for all state fields that reset() should clear. Test doesn't verify complete state cleanup.
**Priority**: P2

## Missing Test Coverage

### Edge Case: Zero timeout (line N/A)
**Issue**: No test for `timeout_seconds=0.0`
**Evidence**: Missing test case
**Fix**: Add test verifying that `timeout_seconds=0.0` triggers immediately after first accept (or validate config rejects it).
**Priority**: P2

### Edge Case: Negative timeout (line N/A)
**Issue**: No test for invalid configuration values
**Evidence**: Missing test case
**Fix**: Add test verifying that negative `timeout_seconds` raises validation error (or triggers immediately).
**Priority**: P2

### Edge Case: Zero count (line N/A)
**Issue**: No test for `count=0`
**Evidence**: Missing test case
**Fix**: Add test verifying behavior when `count=0` (should trigger immediately? or reject config?).
**Priority**: P2

### Edge Case: Condition evaluation exceptions (line N/A)
**Issue**: No test for condition expressions that raise exceptions during evaluation
**Evidence**: Missing test case
**Fix**: Add test where `ExpressionParser.evaluate()` raises exception (e.g., `row['missing_field']` when field doesn't exist in context). Per "No Bug-Hiding Patterns" rule, this should crash, not return False.
**Priority**: P1

### Edge Case: Condition with division by zero (line N/A)
**Issue**: No test for arithmetic errors in user-provided expressions
**Evidence**: Missing test case
**Fix**: Add test for `condition="row['batch_count'] / 0 > 10"`. Should this crash (their data is our code's problem) or quarantine?
**Priority**: P2

### Invariant: should_trigger() called multiple times (line N/A)
**Issue**: No test verifying idempotence of `should_trigger()`
**Evidence**: Missing test case
**Fix**: Add test calling `should_trigger()` multiple times without state change, verifying it returns same result. Critical because implementation has side effect (`self._last_triggered = None` at start).
**Priority**: P1

### State Mutation: Verify reset() doesn't affect which_triggered() before should_trigger() (line N/A)
**Issue**: No test for interaction between reset() and which_triggered()
**Evidence**: Missing test case
**Fix**: Add test: trigger fires → check which_triggered() → reset() → check which_triggered() again. Should return None after reset but before next should_trigger().
**Priority**: P2

### State Mutation: Multiple record_accept() after trigger fires (line N/A)
**Issue**: No test verifying behavior when accumulating rows after trigger has fired but before reset
**Evidence**: Missing test case
**Fix**: Add test: reach count threshold → should_trigger() returns True → continue calling record_accept() → verify batch_count continues incrementing. Tests whether trigger is one-shot or retriggerable.
**Priority**: P1

### Integration: get_trigger_type() method (line N/A)
**Issue**: `get_trigger_type()` method is never tested
**Evidence**: Method exists in implementation (line 144) but no test exercises it
**Fix**: Add tests verifying `get_trigger_type()` returns correct `TriggerType` enum for each trigger type, and `None` when no trigger fired.
**Priority**: P2

### Integration: get_age_seconds() method (line N/A)
**Issue**: `get_age_seconds()` method is never tested
**Evidence**: Method exists in implementation (line 74) but no test exercises it
**Fix**: Add test verifying `get_age_seconds()` returns same value as `batch_age_seconds` property. This method exists for checkpoint serialization clarity.
**Priority**: P3

### Security: Expression injection protection (line N/A)
**Issue**: No test verifying that condition trigger rejects malicious expressions
**Evidence**: Tests use trusted expressions like `"row['batch_count'] >= 50"` but don't verify security boundaries
**Fix**: Add tests for forbidden constructs (per ExpressionParser): lambda expressions, function calls, attribute access beyond row.get(), imports. These should raise `ExpressionSecurityError` at `TriggerConfig` construction (or `TriggerEvaluator.__init__`).
**Priority**: P1

## Infrastructure Gaps

### Repeated Setup: Import statements in every test (line 13-30)
**Issue**: Every test method imports `TriggerConfig` and `TriggerEvaluator` locally
**Evidence**: Lines 15-16, 28-29, 42-43, 54-55, etc.
**Fix**: Add module-level imports or pytest fixture for common dependencies. Reduces 4 lines per test to 0.
**Priority**: P3

### Missing Fixture: Time mocking infrastructure (line N/A)
**Issue**: No shared fixture for mocking `time.monotonic()` to make timing tests deterministic
**Evidence**: Tests use `time.sleep()` which is slow and flaky
**Fix**: Create `@pytest.fixture` that patches `time.monotonic()` with controllable mock. Tests can advance time explicitly.
**Priority**: P1

### Missing Fixture: Standard TriggerConfig instances (line N/A)
**Issue**: Tests construct TriggerConfig inline, making it verbose to test common configurations
**Evidence**: `TriggerConfig(count=100)` appears 5 times
**Fix**: Add fixtures like `@pytest.fixture def count_trigger_100()` returning pre-configured instances. Reduces duplication and makes intent clearer.
**Priority**: P3

### Missing Fixture: TriggerEvaluator with pre-seeded state (line N/A)
**Issue**: Tests that need "batch with 50 accepts" must loop 50 times in every test
**Evidence**: `for _ in range(50): evaluator.record_accept()` appears multiple times
**Fix**: Add parameterized fixture `evaluator_with_accepts(n)` that returns evaluator with N accepts already recorded.
**Priority**: P3

### Test Isolation: No verification of clean slate (line N/A)
**Issue**: Tests don't verify that TriggerEvaluator starts in expected initial state
**Evidence**: No test explicitly asserts `batch_count == 0` and `batch_age_seconds == 0.0` on fresh instance
**Fix**: Add test `test_initial_state()` verifying all properties on newly constructed evaluator.
**Priority**: P2

## Misclassified Tests

### All tests: Should be unit tests (CORRECT)
**Assessment**: Tests are correctly classified as unit tests. They test `TriggerEvaluator` in isolation without external dependencies (no database, no filesystem, no network).
**No action needed**

## Property Testing Gaps

### Property: Trigger order independence (line N/A)
**Issue**: No property test verifying that trigger evaluation order doesn't affect which trigger wins
**Evidence**: Implementation checks count → timeout → condition (lines 108-128). If multiple triggers fire simultaneously, first one wins. Tests don't verify this is deterministic.
**Fix**: Add property test with Hypothesis: generate random (count, timeout, condition) configs and random accept sequences, verify same trigger always wins for same state.
**Priority**: P2

### Property: Monotonic batch_count (line N/A)
**Issue**: No property test verifying batch_count never decreases except on reset
**Evidence**: `record_accept()` should only increase batch_count
**Fix**: Add property test: generate random sequence of record_accept() calls, verify batch_count monotonically increases.
**Priority**: P3

### Property: Monotonic batch_age_seconds (line N/A)
**Issue**: No property test verifying batch_age_seconds never decreases except on reset
**Evidence**: Time should only move forward
**Fix**: Add property test verifying age never decreases between should_trigger() calls.
**Priority**: P3

## Positive Observations

- **Clear test names**: Every test method clearly describes what it validates
- **Comprehensive trigger type coverage**: Tests cover all three trigger types (count, timeout, condition)
- **Combined trigger testing**: Tests verify OR logic for multiple simultaneous triggers
- **Boundary testing**: Tests check exact threshold (test_count_trigger_reached), below threshold, and above threshold
- **State inspection**: Tests verify internal state via `batch_count` and `which_triggered()` properties
- **Security awareness**: Tests use `row['field']` syntax per ExpressionParser security model (line 81, 94, etc.)
- **Documentation**: Tests include docstrings explaining what they validate

## Recommendations Summary

**P0 - Critical (Fix Before RC-1)**
- None

**P1 - High (Fix Soon)**
1. Replace all `time.sleep()` calls with `time.monotonic()` mocking for deterministic tests
2. Add test for condition evaluation exceptions (missing field access)
3. Add test for `should_trigger()` idempotence and side effects
4. Add test for continued accepts after trigger fires (retriggerable vs one-shot)
5. Add tests for malicious expression rejection (security boundary)

**P2 - Medium (Fix This Sprint)**
6. Add edge case tests for zero/negative timeout and count values
7. Complete state verification in `test_reset_clears_state`
8. Add test for `get_trigger_type()` method coverage
9. Add test for initial state verification
10. Add property tests for trigger order independence

**P3 - Low (Nice to Have)**
11. Extract common imports and configs to fixtures
12. Add property tests for monotonic invariants
13. Add fixture for pre-seeded evaluator states

**Test Infrastructure Priority**
- **Immediate**: Mock `time.monotonic()` to eliminate flakiness
- **Soon**: Add fixtures for common configs and seeded states
- **Later**: Property testing with Hypothesis
