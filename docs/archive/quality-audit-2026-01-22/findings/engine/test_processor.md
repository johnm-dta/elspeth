# Test Quality Review: test_processor.py

## Summary

Reviewed 3,892 lines of RowProcessor tests. Found critical infrastructure gaps including missing fixtures for repeated setup, no property testing for state transitions, incomplete audit trail verification, and shared mutable state risks. Several tests verify outcomes without validating Landscape recording completeness, violating the auditability standard.

## Poorly Constructed Tests

### Test: test_nested_forks_all_children_executed (line 908)
**Issue**: Fragile assertion checking count field presence
**Evidence**:
```python
# Line 1059
assert result.final_data.get("count") == 1
```
**Fix**: Use direct access `result.final_data["count"]` since MarkerTransform always sets this field. The `.get()` call suggests uncertainty about whether the field exists, which is a bug-hiding pattern. If the transform doesn't set count=1, that's a transform bug that should crash.
**Priority**: P2

### Test: test_fork_then_coalesce_require_all (line 1487)
**Issue**: Missing verification of consumed token state transitions
**Evidence**: Test verifies merged data exists but doesn't confirm that consumed tokens have terminal COALESCED outcome recorded in Landscape. This violates "every row reaches exactly one terminal state".
**Fix**: Add explicit checks:
```python
# Verify consumed tokens reached COALESCED state in Landscape
for child_token_id in child_token_ids:
    token_record = recorder.get_token(child_token_id)
    assert token_record.terminal_outcome == "COALESCED"  # If this field exists
```
**Priority**: P1

### Test: test_coalesced_token_audit_trail_complete (line 1669)
**Issue**: Incomplete audit trail verification
**Evidence**: Test checks node_states exist but doesn't verify input_data_hash and output_data_hash are recorded at each step. Lines 1878-1880 check for "at least one state" but don't validate hash completeness.
**Fix**: Add hash verification loop:
```python
for state in states:
    assert state.input_data_hash is not None, f"Step {state.step_index} missing input hash"
    assert state.output_data_hash is not None, f"Step {state.step_index} missing output hash"
```
**Priority**: P0 (auditability critical)

### Test: test_coalesce_best_effort_with_quarantined_child (line 1898)
**Issue**: Uses time.sleep for timeout testing
**Evidence**:
```python
# Line 1997
time.sleep(0.15)
```
**Fix**: Mock the timeout mechanism instead of sleeping. This is a classic "sleepy assertion" anti-pattern that makes tests slow and flaky.
```python
# Replace with:
with patch.object(coalesce_executor, '_get_current_time', side_effect=[0, 0.2]):
    timed_out = coalesce_executor.check_timeouts("merger", step_in_pipeline=3)
```
**Priority**: P1

### Test: test_retries_transient_transform_exception (line 2490)
**Issue**: Relies on call_count mutation
**Evidence**:
```python
# Lines 2498-2505
call_count = 0
def flaky_execute(*args, **kwargs):
    nonlocal call_count
    call_count += 1
```
**Fix**: Use mock.call_count instead of maintaining separate state. This pattern is vulnerable to shared mutable state bugs if tests run in parallel.
**Priority**: P2

### Test: test_max_retries_exceeded_returns_failed_outcome (line 2587)
**Issue**: Vague error message assertion
**Evidence**:
```python
# Line 2661
assert "MaxRetriesExceeded" in str(result.error) or "attempts" in str(result.error)
```
**Fix**: Test exact error structure. The `or` makes this assertion nearly meaningless - it will pass if the error just contains the word "attempts" anywhere.
```python
# Should be:
assert "MaxRetriesExceeded" in str(result.error)
assert result.error_metadata["attempts"] == 2
```
**Priority**: P2

### Test: test_processor_buffers_rows_for_aggregation_node (line 2702)
**Issue**: Appending to results list destroys first-class assertion clarity
**Evidence**:
```python
# Lines 2774-2783
results = []
for i in range(3):
    result_list = processor.process_row(...)
    results.append(result_list[0])
```
**Fix**: Make assertions inline in the loop or use separate named variables:
```python
result0 = processor.process_row(row_index=0, row_data={"value": 1}, transforms=[transform], ctx=ctx)[0]
result1 = processor.process_row(row_index=1, row_data={"value": 2}, transforms=[transform], ctx=ctx)[0]
result2 = processor.process_row(row_index=2, row_data={"value": 3}, transforms=[transform], ctx=ctx)[0]

assert result0.outcome == RowOutcome.CONSUMED_IN_BATCH
assert result1.outcome == RowOutcome.CONSUMED_IN_BATCH
assert result2.outcome == RowOutcome.COMPLETED
```
This makes test failures much clearer about which row failed.
**Priority**: P2

### Test: test_aggregation_passthrough_mode (line 3154)
**Issue**: Fragile token ID tracking with append pattern
**Evidence**:
```python
# Lines 3234-3249
buffered_token_ids = []
for i in range(2):
    if i < 2:
        buffered_token_ids.append(result_list[0].token_id)
```
**Fix**: The `if i < 2` check inside a `range(2)` loop is redundant and suggests copy-paste confusion. Extract to explicit assertions.
**Priority**: P3

## Misclassified Tests

### Test: test_work_queue_iteration_guard_prevents_infinite_loop (line 1065)
**Issue**: Not actually testing the guard - just mutating module state
**Evidence**:
```python
# Lines 1097-1114
original_max = proc_module.MAX_WORK_QUEUE_ITERATIONS
proc_module.MAX_WORK_QUEUE_ITERATIONS = 5
try:
    # This doesn't actually trigger the guard!
    results = processor.process_row(...)
    assert len(results) == 1  # Passes normally
finally:
    proc_module.MAX_WORK_QUEUE_ITERATIONS = original_max
```
**Fix**: This should be a negative test that creates a malformed transform that would actually trigger infinite work. As written, it's testing nothing - the guard never fires. Either delete this test or make it actually trigger the guard with a pathological transform.
**Priority**: P1

### Test: test_processor_accepts_coalesce_executor (line 1448)
**Issue**: Pure smoke test - provides zero value
**Evidence**:
```python
# Lines 1477-1485
processor = RowProcessor(..., coalesce_executor=coalesce_executor)
assert processor._coalesce_executor is coalesce_executor
```
**Fix**: Delete this test. It's testing Python assignment, not RowProcessor behavior. The real test is whether the executor is *used correctly*, which is tested elsewhere.
**Priority**: P3

### Test: test_processor_accepts_retry_manager (line 2470)
**Issue**: Another pure smoke test
**Evidence**:
```python
# Lines 2480-2488
processor = RowProcessor(..., retry_manager=retry_manager)
assert processor._retry_manager is retry_manager
```
**Fix**: Delete. Same issue - testing assignment, not behavior.
**Priority**: P3

### Test: test_processor_accepts_restored_aggregation_state (line 2667)
**Issue**: Smoke test that doesn't verify restoration behavior
**Evidence**:
```python
# Lines 2682-2696
processor = RowProcessor(..., restored_aggregation_state=restored_state)
assert processor._aggregation_executor.get_restored_state("agg_node") == {...}
```
**Fix**: Either delete or promote to integration test that processes a row and verifies the restored state is actually *used*.
**Priority**: P2

## Infrastructure Gaps

### Gap: Missing ProcessorFixture
**Issue**: Every test manually creates LandscapeDB, LandscapeRecorder, run, source node, SpanFactory, PluginContext
**Evidence**: Lines 44-105, 126-176, 190-214, 234-280 all repeat identical setup
**Fix**: Create `@pytest.fixture` for standard processor setup:
```python
@dataclass
class ProcessorFixture:
    db: LandscapeDB
    recorder: LandscapeRecorder
    run: RunInfo
    source_node: NodeInfo
    span_factory: SpanFactory
    ctx: PluginContext

    def create_processor(self, **kwargs) -> RowProcessor:
        return RowProcessor(
            recorder=self.recorder,
            span_factory=self.span_factory,
            run_id=self.run.run_id,
            source_node_id=self.source_node.node_id,
            **kwargs
        )

@pytest.fixture
def processor_fixture() -> ProcessorFixture:
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    run = recorder.begin_run(config={}, canonical_version="v1")
    source = recorder.register_node(...)
    return ProcessorFixture(db, recorder, run, source, SpanFactory(), PluginContext(...))
```
**Priority**: P1

### Gap: No property testing for state transition invariants
**Issue**: Terminal row states are critical but only tested with hand-written examples
**Evidence**: Tests check specific outcomes (COMPLETED, ROUTED, FORKED) but don't verify invariants like "every row reaches exactly one terminal state" across arbitrary transform combinations.
**Fix**: Add Hypothesis-based property test:
```python
from hypothesis import given, strategies as st

@given(
    row_data=st.dictionaries(st.text(), st.integers()),
    num_transforms=st.integers(min_value=0, max_value=5)
)
def test_all_rows_reach_terminal_state(row_data, num_transforms):
    # Generate random transform chain
    # Process row
    # Assert: exactly one result per fork path reaches terminal state
    # Assert: all tokens recorded in Landscape have terminal outcomes
```
**Priority**: P1

### Gap: Missing audit completeness validator
**Issue**: Tests verify data correctness but don't systematically check Landscape recording completeness
**Evidence**: Tests like `test_process_through_transforms` (line 39) check `result.final_data == {"value": 21}` but don't verify that input/output hashes, node_states, and token relationships are all recorded.
**Fix**: Create helper function:
```python
def assert_audit_trail_complete(recorder, token_id, expected_steps):
    """Verify complete audit trail for a token."""
    token = recorder.get_token(token_id)
    assert token is not None

    states = recorder.get_node_states_for_token(token_id)
    assert len(states) == expected_steps

    for state in states:
        assert state.input_data_hash is not None
        assert state.output_data_hash is not None
        assert state.step_index >= 0
        # Could retrieve payload via hash and verify content
```
Then use in every test that processes rows.
**Priority**: P0 (auditability critical)

### Gap: No shared TestTransform base classes
**Issue**: Every test class defines its own DoubleTransform, AddOneTransform, EnricherTransform with identical patterns
**Evidence**: Lines 74-96, 86-96, 151-161, 319-334, 387-402, etc.
**Fix**: Create test_helpers.py with standard transforms:
```python
# tests/engine/test_helpers.py
def make_identity_transform(name: str, node_id: str) -> BaseTransform:
    """Returns a transform that passes data through unchanged."""
    ...

def make_math_transform(name: str, node_id: str, op: Callable[[int], int]) -> BaseTransform:
    """Returns a transform that applies op to row["value"]."""
    ...

def make_error_transform(name: str, node_id: str, error_msg: str, on_error: str) -> BaseTransform:
    """Returns a transform that always errors with on_error policy."""
    ...
```
**Priority**: P2

### Gap: Missing integration tests for multi-step audit queries
**Issue**: Tests verify that data is recorded but don't test the explain() query path
**Evidence**: No tests call `landscape.explain()` or `recorder.get_lineage()` to verify the audit trail is queryable
**Fix**: Add integration tests that process rows through complex DAGs then query explain() to verify all steps are traceable.
**Priority**: P1

### Gap: No test isolation verification
**Issue**: Tests share module-level imports and could have hidden interdependencies
**Evidence**: Line 1097 mutates `proc_module.MAX_WORK_QUEUE_ITERATIONS` globally. If this test fails to restore the value, subsequent tests would behave differently.
**Fix**: Use pytest markers to detect shared state:
```python
@pytest.fixture(autouse=True)
def verify_no_processor_globals():
    """Ensure tests don't pollute global state."""
    import elspeth.engine.processor as proc
    original_max = proc.MAX_WORK_QUEUE_ITERATIONS
    yield
    assert proc.MAX_WORK_QUEUE_ITERATIONS == original_max, "Test mutated global state"
```
**Priority**: P2

### Gap: Missing negative tests for audit trail tampering
**Issue**: No tests verify that corrupted Landscape data causes crashes (as required by Three-Tier Trust Model)
**Evidence**: CLAUDE.md states "Bad data in the audit trail = crash immediately" but no tests verify this.
**Fix**: Add tests that corrupt node_states, tokens, or rows and verify the processor crashes with clear error messages:
```python
def test_processor_crashes_on_corrupted_node_state():
    # Create valid state
    # Corrupt node_state.status to invalid enum value
    # Attempt to process row
    # Assert: raises with message about corrupted audit data
```
**Priority**: P1

## Positive Observations

- **Comprehensive outcome coverage**: Tests cover all terminal states (COMPLETED, ROUTED, FORKED, QUARANTINED, FAILED, CONSUMED_IN_BATCH, COALESCED)
- **Real Landscape integration**: Tests use real LandscapeDB/LandscapeRecorder, not mocks, which validates actual database behavior
- **Clear test class organization**: Logical grouping (TestRowProcessor, TestRowProcessorGates, TestQuarantineIntegration, etc.)
- **Docstrings explain intent**: Most tests have clear docstrings explaining what behavior is being verified
- **No assertion-free tests**: All tests contain meaningful assertions (no execute-only smoke tests beyond the few flagged above)
- **Uses real plugin inheritance**: Test transforms inherit from BaseTransform, exercising the actual plugin type detection logic (isinstance checks)

## Confidence Assessment

**Confidence: Medium-High (70%)**

I have read the full test file (3,892 lines) and CLAUDE.md standards. However:

- **Information Gap**: Haven't reviewed the actual RowProcessor implementation to verify test coverage completeness
- **Information Gap**: Haven't checked if there are companion tests in other files (test_orchestrator.py, test_executors.py) that might test audit trail verification
- **Information Gap**: Haven't verified if pytest plugins or custom assertions exist that handle audit verification automatically

## Risk Assessment

**Highest Risk**: Missing audit trail completeness checks (P0). Tests verify business logic but don't systematically validate that every transform boundary records input_hash, output_hash, and node_state. This directly violates the auditability standard: "if it's not recorded, it didn't happen."

**Medium Risk**: Sleepy assertions with time.sleep (P1). Makes tests slow and creates false negatives in CI environments.

**Low Risk**: Duplicate setup code (P2). Creates maintenance burden but doesn't affect correctness.

## Caveats

1. **Scope**: This review focuses on test architecture anti-patterns, not the correctness of RowProcessor itself
2. **Property testing recommendation**: May be overkill if the transform space is small and well-covered by existing tests - would need to see actual bugs to justify the investment
3. **Audit trail validation**: The priority of P0 assumes no centralized audit validation exists elsewhere in the test suite
4. **Smoke test deletion**: Recommendations to delete smoke tests assume those parameters are tested through integration tests elsewhere

## Information Gaps

1. What does `recorder.get_node_states_for_token()` return when a token has no states? Does it return `[]` or raise? This affects whether tests should check `len(states) == 0` vs. `assert states is not None`.
2. Is there a centralized audit trail validator in `tests/conftest.py` or a pytest plugin that validates Landscape completeness automatically?
3. Are there performance benchmarks that would justify keeping time.sleep tests as-is rather than mocking?
4. Does the codebase have a policy on test fixture complexity? Some teams prefer inline setup for clarity over DRY fixtures.
