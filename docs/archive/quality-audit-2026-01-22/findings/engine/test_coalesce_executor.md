# Test Quality Review: test_coalesce_executor.py

## Summary

This test suite covers CoalesceExecutor functionality comprehensively but contains critical flakiness issues from sleepy assertions (line 482, 566), incomplete lineage verification for the COALESCED terminal state, and missing property-based tests for merge strategy correctness. Infrastructure has significant duplication in node registration boilerplate.

---

## Poorly Constructed Tests

### Test: test_quorum_does_not_merge_on_timeout_if_quorum_not_met (line 407)
**Issue**: Sleepy assertion using fixed `time.sleep(0.15)` instead of condition-based wait
**Evidence**:
```python
# Wait for timeout
time.sleep(0.15)

# check_timeouts should return empty list (quorum not met)
timed_out = executor.check_timeouts("quorum_merge", step_in_pipeline=2)
```
**Fix**: Replace with deterministic timeout injection via clock mocking or executor method that accepts a `now` parameter. Example:
```python
# Option 1: Mock time
with patch('time.monotonic', return_value=start_time + 0.2):
    timed_out = executor.check_timeouts("quorum_merge", step_in_pipeline=2)

# Option 2: Add _check_timeouts_at(timestamp) testing method
timed_out = executor._check_timeouts_at(
    "quorum_merge",
    step_in_pipeline=2,
    now=start_time + 0.2
)
```
**Priority**: P0 (causes CI flakiness, violates sleep prohibition)

---

### Test: test_best_effort_merges_on_timeout (line 492)
**Issue**: Sleepy assertion using fixed `time.sleep(0.15)` instead of condition-based wait
**Evidence**:
```python
# Wait for timeout
time.sleep(0.15)

# Check timeout and force merge
timed_out = executor.check_timeouts("best_effort_merge", step_in_pipeline=2)
```
**Fix**: Same as above - replace with deterministic time injection via mocking or testing API
**Priority**: P0 (causes CI flakiness, violates sleep prohibition)

---

### Test: test_executor_initializes (line 54)
**Issue**: Assertion-free test - verifies construction but not behavior
**Evidence**:
```python
executor = CoalesceExecutor(
    recorder=recorder,
    span_factory=span_factory,
    token_manager=token_manager,
    run_id=run_id,
)

assert executor is not None  # Only checks construction
```
**Fix**: Either delete this test (construction failure would raise anyway) or verify executor state:
```python
# Verify executor properly initialized internal structures
assert len(executor._coalesces) == 0
assert executor._recorder is recorder
assert executor._run_id == run.run_id
```
**Priority**: P2 (provides minimal value)

---

### Test: test_flush_pending_quorum_met_merges (line 1058)
**Issue**: Test comment admits test is conceptually broken - quorum met means already merged, so flushing has nothing to do
**Evidence**:
```python
# The 2nd accept should trigger merge immediately
# So there's nothing to flush with quorum met (it already merged)

# The useful test is: ensure flush_pending doesn't break when
# there's nothing pending (empty case)
```
Lines 1092-1106 are a commented-out thought process that should have been deleted.
**Fix**: Either:
1. Rename test to `test_flush_pending_empty_after_quorum_merge` to reflect what it actually tests
2. Delete test entirely (duplicates `test_flush_pending_empty_when_no_pending`)
**Priority**: P1 (confusing, wastes reviewer time)

---

## Misclassified Tests

### Test: TestCoalesceIntegration.test_fork_process_coalesce_full_flow (line 709)
**Issue**: Classified as "integration" but doesn't integrate subsystems - it's a unit test using mocks
**Evidence**:
```python
# Simulates transforms but doesn't actually execute them:
sentiment_token = TokenInfo(
    row_id=children[0].row_id,
    token_id=children[0].token_id,
    row_data={"sentiment": "positive", "confidence": 0.92},
    branch_name="sentiment",
)
```
No actual Transform plugins are executed. This is unit-level testing of CoalesceExecutor in isolation.
**Fix**: Move to `TestCoalesceExecutorRequireAll` class or rename class to `TestCoalesceExecutorFlows` (drop "Integration")
**Priority**: P3 (misleading organization)

---

## Infrastructure Gaps

### Gap: Repeated node registration boilerplate across all tests
**Issue**: Every test manually registers source_node and coalesce_node with identical parameters except `node_id` and `plugin_name`
**Evidence**: Lines 87-104, 162-179, 254-271, 333-350, 507-524, 616-633, 718-736, 819-836, 909-926, 990-1007, 1071-1088
**Fix**: Create parametrized fixture:
```python
@pytest.fixture
def nodes_for_coalesce(recorder: LandscapeRecorder, run: Run) -> dict[str, NodeRegistration]:
    """Register standard source and coalesce nodes for testing."""
    source = recorder.register_node(
        run_id=run.run_id,
        node_id="test_source",
        plugin_name="test_source",
        node_type=NodeType.SOURCE,
        plugin_version="1.0.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )
    coalesce = recorder.register_node(
        run_id=run.run_id,
        node_id="test_coalesce",
        plugin_name="test_coalesce",
        node_type=NodeType.COALESCE,
        plugin_version="1.0.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )
    return {"source": source, "coalesce": coalesce}
```
**Priority**: P2 (reduces duplication, makes tests easier to read)

---

### Gap: No verification of COALESCED terminal state in Landscape
**Issue**: Tests verify merged token data but never query Landscape to confirm COALESCED state was recorded for consumed tokens
**Evidence**: Per CLAUDE.md line 348: "COALESCED - Merged in join" is a terminal state. No test verifies this.
**Fix**: Add assertion after merge:
```python
# After successful coalesce
outcome2 = executor.accept(token_b, "merge_results", step_in_pipeline=2)
assert outcome2.merged_token is not None

# Verify consumed tokens recorded as COALESCED
for consumed in outcome2.consumed_tokens:
    node_state = recorder.get_node_state(
        run_id=run.run_id,
        token_id=consumed.token_id,
        node_id=coalesce_node.node_id
    )
    assert node_state.final_state == FinalState.COALESCED
```
**Priority**: P0 (violates auditability standard - "if it's not recorded, it didn't happen")

---

### Gap: No verification of parent_token_id lineage for merged tokens
**Issue**: DAG execution model (CLAUDE.md line 232) requires `parent_token_id` tracking for joins, but tests never verify merged token has correct parent references
**Evidence**: Tests check `row_data` but not lineage metadata
**Fix**: Add lineage verification:
```python
merged_token = outcome2.merged_token
assert merged_token.parent_token_id is not None

# Verify all consumed tokens are parents
consumed_ids = {t.token_id for t in outcome2.consumed_tokens}
# Query Landscape for parent relationships
lineage = recorder.get_token_lineage(run_id=run.run_id, token_id=merged_token.token_id)
assert set(lineage.parent_token_ids) == consumed_ids
```
**Priority**: P0 (violates auditability - can't trace merged results back to source branches)

---

### Gap: No property-based tests for merge strategy correctness
**Issue**: Merge strategies (union, nested, select) are tested with single examples but not systematically
**Evidence**: CLAUDE.md line 310 endorses Hypothesis for property testing. Current tests use hardcoded examples.
**Fix**: Add property tests:
```python
from hypothesis import given, strategies as st

@given(
    branch_count=st.integers(min_value=2, max_value=5),
    fields_per_branch=st.integers(min_value=1, max_value=3),
)
def test_union_merge_includes_all_fields(branch_count, fields_per_branch):
    """Union merge should preserve all fields from all branches."""
    # Generate branch data
    branches = [f"branch_{i}" for i in range(branch_count)]
    tokens = [
        create_token_with_fields(branch, fields_per_branch)
        for branch in branches
    ]

    # Merge with union strategy
    merged = merge_union(tokens)

    # All fields from all branches must be present
    expected_field_count = branch_count * fields_per_branch
    assert len(merged.row_data) == expected_field_count
```
**Priority**: P1 (proves merge correctness across edge cases)

---

### Gap: No test coverage for concurrent token arrival (race conditions)
**Issue**: CoalesceExecutor likely has mutable state (`_pending_by_row_id`). Tests are sequential - no verification of thread safety.
**Evidence**: Real pipelines may process multiple rows concurrently. If two rows fork and tokens arrive interleaved, does state management work correctly?
**Fix**: Add concurrency test:
```python
def test_concurrent_rows_dont_interfere():
    """Multiple rows coalescing simultaneously should not interfere."""
    # Create 2 source rows
    row1_token = create_initial_token(row_index=0)
    row2_token = create_initial_token(row_index=1)

    # Fork both
    row1_children = fork_token(row1_token, ["a", "b"])
    row2_children = fork_token(row2_token, ["a", "b"])

    # Interleave accepts: row1_a, row2_a, row1_b, row2_b
    executor.accept(row1_children[0])
    executor.accept(row2_children[0])
    executor.accept(row1_children[1])  # Should complete row1
    executor.accept(row2_children[1])  # Should complete row2

    # Verify both rows merged correctly without interference
    assert row1_merged.row_data != row2_merged.row_data
```
**Priority**: P1 (if executor is used in parallel contexts, this is critical)

---

### Gap: No test for invalid branch names (branch arrives that wasn't expected)
**Issue**: What if a token arrives with `branch_name="unexpected"` when settings specified `branches=["path_a", "path_b"]`?
**Evidence**: No defensive test for configuration mismatch
**Fix**: Add error case test:
```python
def test_accept_rejects_unexpected_branch():
    """Accepting token from unexpected branch should raise ValueError."""
    settings = CoalesceSettings(
        name="merge",
        branches=["path_a", "path_b"],
        policy="require_all",
        merge="union",
    )
    executor.register_coalesce(settings, coalesce_node.node_id)

    # Create token from branch NOT in settings
    bad_token = TokenInfo(
        row_id=uuid4(),
        token_id=uuid4(),
        row_data={},
        branch_name="path_c",  # NOT in settings.branches
    )

    with pytest.raises(ValueError, match="unexpected branch"):
        executor.accept(bad_token, "merge", step_in_pipeline=2)
```
**Priority**: P1 (catches configuration bugs early)

---

### Gap: No test for merge strategies with field conflicts
**Issue**: Union merge - what if both branches produce `{"score": 0.9}` and `{"score": 0.8}`?
**Evidence**: Test line 231-234 shows union with disjoint fields, but not overlapping fields
**Fix**: Add conflict test:
```python
def test_union_merge_with_field_conflict_raises():
    """Union merge should detect and reject field conflicts."""
    token_a = TokenInfo(..., row_data={"score": 0.9})
    token_b = TokenInfo(..., row_data={"score": 0.8})  # Conflict!

    # Should raise or have defined conflict resolution
    with pytest.raises(ValueError, match="field conflict.*score"):
        executor.accept(token_b, "merge", step_in_pipeline=2)
```
**Priority**: P0 (silent data loss would violate audit integrity)

---

## Positive Observations

- **Comprehensive policy coverage**: All four coalesce policies (require_all, first, quorum, best_effort) have dedicated test classes
- **Clear test organization**: Test classes grouped by policy type makes navigation easy
- **Good use of fixtures**: `executor_setup` fixture reduces boilerplate in simple tests
- **Timeout testing**: Tests verify both timeout expiry and policy-specific timeout behavior (though implementation is flaky)
- **Metadata verification**: `test_coalesce_records_audit_metadata` checks audit trail completeness (line 602)
- **Flush semantics**: Comprehensive coverage of `flush_pending()` across different policies (lines 803-1182)

---

## Confidence Assessment

**Confidence Level**: Medium-High (75%)

**Rationale**:
- Read CLAUDE.md thoroughly - understand coalesce semantics, terminal states, auditability requirements
- Read test file completely - identified patterns, anti-patterns, and gaps
- Familiar with test anti-patterns from quality engineering domain knowledge
- Some uncertainty around CoalesceExecutor internal implementation (e.g., thread safety, field conflict handling) - would need to read source to be 100% certain

---

## Risk Assessment

**High Risks**:
1. **Sleepy assertions** (P0): Tests will flake in CI, especially under load or slow machines
2. **Missing COALESCED state verification** (P0): Audit trail may be incomplete - violates core requirement
3. **Missing parent_token_id lineage** (P0): Cannot trace merged results back to source branches
4. **Merge field conflicts** (P0): Silent data loss would destroy audit integrity

**Medium Risks**:
5. **No property-based testing**: Merge strategies may have edge-case bugs
6. **No concurrency testing**: If executor is used in parallel, state corruption possible
7. **No invalid branch testing**: Configuration bugs slip through

**Low Risks**:
8. **Infrastructure duplication**: Makes tests harder to maintain but doesn't affect correctness

---

## Information Gaps

1. **CoalesceExecutor source code**: Haven't read implementation to verify thread safety, field conflict handling, or internal state management
2. **TokenManager behavior**: Assume it handles parent_token_id correctly, but haven't verified
3. **LandscapeRecorder API**: Assume methods like `get_node_state()` and `get_token_lineage()` exist, but haven't checked
4. **Merge strategy specifications**: Don't know if field conflicts are allowed, forbidden, or have defined resolution rules

---

## Caveats

1. **Recommended fixes assume API existence**: Suggestions for lineage verification assume `get_token_lineage()` method exists - may need alternative implementation
2. **Concurrency recommendations conditional**: Only relevant if CoalesceExecutor is used in multi-threaded contexts (check orchestrator usage)
3. **Property test complexity**: Hypothesis tests require careful design - bad generators produce useless tests
4. **Time mocking alternatives**: If mocking `time.monotonic()` breaks executor internals, may need to add `_now` parameter to methods for testing

---

## Recommended Action Plan

**Immediate (Pre-RC1 Release)**:
1. Fix sleepy assertions in lines 482 and 566 (P0)
2. Add COALESCED state verification to merge tests (P0)
3. Add parent_token_id lineage verification to merge tests (P0)
4. Add field conflict test for union merge (P0)
5. Add invalid branch rejection test (P1)

**Post-RC1**:
6. Refactor node registration into fixture (P2)
7. Add property-based tests for merge strategies (P1)
8. Add concurrency test if executor used in parallel (P1)
9. Delete or rename `test_flush_pending_quorum_met_merges` (P1)
10. Delete or fix `test_executor_initializes` (P2)
