# Test Quality Review: test_executors.py

## Summary
4073 lines testing TransformExecutor, GateExecutor, ConfigGateExecutor, SinkExecutor, and AggregationExecutor. Tests are generally well-structured with strong isolation and comprehensive coverage. However, there are significant infrastructure gaps (repeated setup, missing fixtures, no property-based testing), mutation vulnerabilities in state management tests, and several poorly-constructed edge case tests that rely on fragile mocks or internal implementation details.

## Poorly Constructed Tests

### Test: test_execute_transform_records_attempt_number (line 492)
**Issue**: Uses unittest.mock.patch to spy on internal method calls instead of verifying behavior through observable state
**Evidence**:
```python
with patch.object(recorder, "begin_node_state", wraps=recorder.begin_node_state) as mock:
    executor.execute_transform(...)
mock.assert_called_once()
call_kwargs = mock.call_args.kwargs
assert call_kwargs.get("attempt") == 2
```
**Fix**: Query landscape for node state and verify attempt number from persisted data: `state = recorder.get_node_states_for_token(token.token_id)[0]; assert state.attempt == 2`
**Priority**: P2 - Test works but couples to implementation, brittle against refactoring

### Test: test_checkpoint_size_warning_at_1mb_threshold (line 2554)
**Issue**: Mocks logging to verify warning messages instead of treating warnings as a behavioral contract
**Evidence**: `with patch.object(logging, "getLogger") as mock_get_logger: mock_logger.warning.assert_called`
**Fix**: Either (1) make checkpoint size limits part of the contract and test via RuntimeError at hard limit, or (2) remove size validation entirely from unit tests and test in production observability
**Priority**: P3 - Testing logging is low-value; size limits matter but warnings don't affect correctness

### Test: test_checkpoint_size_error_at_10mb_limit (line 3629)
**Issue**: Creates 6000 tokens with 2KB data each to exceed 10MB, extremely slow and memory-intensive
**Evidence**: Loop creates `very_large_row_data = {"data": "x" * 2000}` 6000 times
**Fix**: Mock `sys.getsizeof` or JSON serializer to return large size without actually allocating memory
**Priority**: P1 - Performance killer, likely skipped in practice

### Test: test_checkpoint_size_warning_but_no_error_between_thresholds (line 3814)
**Issue**: Same as above - creates 2500 tokens with 2KB each (~5MB), slow and wasteful
**Evidence**: Loop creates `large_row_data = {"data": "x" * 2000}` 2500 times
**Fix**: Same - mock size calculation instead of allocating real memory
**Priority**: P1 - Performance killer

### Test: test_checkpoint_size_no_warning_under_1mb (line 3743)
**Issue**: Same pattern - creates 900 tokens with 1KB each, slower than necessary
**Evidence**: Loop creates `medium_row_data = {"data": "x" * 1000}` 900 times
**Fix**: Same - mock size calculation
**Priority**: P2 - Less severe but still wasteful

### Test: test_execute_flush_detects_incomplete_restoration (line 3493)
**Issue**: Directly manipulates private state (`executor._buffers`, `executor._buffer_tokens`) to simulate corruption instead of reaching that state through valid operations
**Evidence**:
```python
executor._buffers[agg_node.node_id] = [{"value": 10}, {"value": 20}]
executor._buffer_tokens[agg_node.node_id] = []  # EMPTY - the bug state!
```
**Fix**: If this state is impossible to reach through public API, delete the test. If it IS reachable, reproduce via that path. If it's a defensive guard, document why corruption is possible (e.g., checkpoint format migration).
**Priority**: P2 - Either useless or testing wrong thing

### Test: test_restore_from_checkpoint_crashes_on_missing_tokens_key (line 3926)
**Issue**: Tests validation of checkpoint format, but format errors should crash at restore time anyway - this is testing error messages, not behavior
**Evidence**: `with pytest.raises(ValueError, match="missing 'tokens' key")`
**Fix**: Consolidate all checkpoint format validation tests into one test with parametrized inputs
**Priority**: P3 - Excessive coverage of error paths

### Test: test_restore_from_checkpoint_crashes_on_invalid_tokens_type (line 3974)
**Issue**: Same as above - testing error message text for invalid checkpoint
**Evidence**: `with pytest.raises(ValueError, match="'tokens' must be a list")`
**Fix**: Same - consolidate into parametrized test
**Priority**: P3 - Excessive

### Test: test_restore_from_checkpoint_crashes_on_missing_token_fields (line 4021)
**Issue**: Same as above - third variation of "malformed checkpoint crashes"
**Evidence**: `with pytest.raises(ValueError, match=r"missing required fields.*row_data")`
**Fix**: Same - consolidate into parametrized test
**Priority**: P3 - Excessive

## Misclassified Tests

### All tests in TestAggregationExecutorCheckpoint (lines 3034-4074)
**Issue**: These are integration tests (test serialization + deserialization + state restoration) but run as unit tests without markers
**Evidence**: `test_checkpoint_roundtrip` creates executor1, serializes to JSON, creates executor2, deserializes - crosses multiple component boundaries
**Fix**: Add `@pytest.mark.integration` to entire class or move to `tests/integration/test_aggregation_checkpoint.py`
**Priority**: P2 - Mislabeling makes it unclear what breaks in fast vs slow test runs

### Test: test_write_multiple_batches_creates_multiple_artifacts (line 2289)
**Issue**: Tests orchestration of multiple write calls with shared sink state - this is closer to integration than unit
**Evidence**: Creates custom `BatchSink` with `_batch_count` state, calls `write()` twice, verifies both artifacts exist in landscape
**Fix**: Move to integration tests or extract sink state management into separate unit test
**Priority**: P3 - Borderline, but sink state across writes is orchestrator concern

## Infrastructure Gaps

### Gap: Massive test fixture duplication
**Issue**: Every test manually creates `db`, `recorder`, `run`, `node`, `token`, `row` - 400+ lines of identical setup code
**Evidence**: Lines 27-65 (test_execute_transform_success), 89-125 (test_execute_transform_error), etc. - pattern repeats 50+ times
**Fix**: Create pytest fixtures:
```python
@pytest.fixture
def executor_setup():
    db = LandscapeDB.in_memory()
    recorder = LandscapeRecorder(db)
    run = recorder.begin_run(config={}, canonical_version="v1")
    return db, recorder, run

@pytest.fixture
def mock_transform_node(executor_setup):
    _, recorder, run = executor_setup
    return recorder.register_node(
        run_id=run.run_id,
        plugin_name="test_transform",
        node_type="transform",
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )
```
**Priority**: P0 - 1000+ lines could be 200 lines, massive maintainability win

### Gap: No property-based tests for aggregation buffering
**Issue**: Aggregation tests use hard-coded counts (2, 3, 5, 10), missing edge cases around boundary conditions
**Evidence**: `test_buffer_updates_trigger_evaluator` tests count=3 with exactly 3 rows, misses off-by-one errors
**Fix**: Use Hypothesis to test "trigger fires IFF buffered_count >= trigger_count" with random counts 0-1000
**Priority**: P1 - Buffering is complex stateful logic, prime candidate for property testing

### Gap: No concurrency/race condition tests
**Issue**: Executors manage mutable state (`_buffers`, `_batch_ids`, `_trigger_evaluators`) but no tests verify thread safety
**Evidence**: Zero `threading` or `concurrent.futures` imports, no race condition coverage
**Fix**: Either (1) document that executors are single-threaded per design, or (2) add property tests with concurrent buffer_row() calls
**Priority**: P2 - Important if executors used across threads, but may be architectural constraint

### Gap: No mutation testing for state isolation
**Issue**: Tests verify buffered data matches expected values but don't verify mutations to returned data don't affect internal state
**Evidence**: `test_get_buffered_data_does_not_clear_buffer` verifies no clearing but doesn't verify defensive copying
**Fix**: After `buffered_rows = executor.get_buffered_rows(node_id)`, mutate `buffered_rows[0]["key"] = "MUTATED"` and verify internal buffer unchanged
**Priority**: P1 - Critical for audit integrity - mutating returned data must not corrupt audit trail

### Gap: No factory functions for test data builders
**Issue**: `make_mock_sink` factory exists (line 2045) but no equivalent for transforms, gates, tokens
**Evidence**: Every test creates tokens with `TokenInfo(row_id="row-1", token_id="token-1", row_data={...})` manually
**Fix**: Create builder functions:
```python
def make_token(index=0, **row_data):
    return TokenInfo(
        row_id=f"row-{index}",
        token_id=f"token-{index}",
        row_data=row_data or {"value": index},
    )
```
**Priority**: P2 - Reduces noise, improves readability

### Gap: No test coverage for terminal state transitions
**Issue**: CLAUDE.md defines 7 terminal states (COMPLETED, ROUTED, FORKED, CONSUMED_IN_BATCH, COALESCED, QUARANTINED, FAILED) but tests don't verify EVERY row reaches EXACTLY ONE terminal state
**Evidence**: Tests check individual states (e.g., `assert state.status == "completed"`) but no test verifies "no row is in two terminal states" or "no row has zero states"
**Fix**: Add invariant test: after any executor operation, `assert len(get_terminal_states(token_id)) == 1`
**Priority**: P0 - Core audit requirement from CLAUDE.md ("every row reaches exactly one terminal state - no silent drops")

### Gap: No tests for NaN/Infinity in row data
**Issue**: CLAUDE.md mandates "NaN and Infinity are strictly rejected" but tests don't verify executors reject these in row_data
**Evidence**: Zero tests with `{"value": float('nan')}` or `{"value": float('inf')}`
**Fix**: Add test that passes NaN/Inf in token.row_data and verifies executor crashes or quarantines
**Priority**: P1 - Explicit requirement from CLAUDE.md's canonical JSON section

### Gap: Test names don't follow RFC 2119 "MUST/SHOULD" convention
**Issue**: Tests describe what happens ("returns_none", "creates_batch") but don't state whether this is required or optional behavior
**Evidence**: `test_write_empty_tokens_returns_none` - is None return required for correctness, or implementation detail?
**Fix**: Rename to indicate requirement level: `test_write_empty_tokens_MUST_return_none_for_idempotency` or mark as `test_write_empty_tokens_implementation_returns_none`
**Priority**: P3 - Nice-to-have for documentation clarity

## Positive Observations

**Strong isolation**: Every test creates independent in-memory database, no shared state pollution
**Comprehensive error path coverage**: Tests verify both success and failure paths (exceptions record audit state)
**Audit-first design**: Tests consistently verify landscape recording (node states, routing events, artifacts) not just return values
**Regression test documentation**: `TestTransformErrorIdRegression` class includes bug ticket reference and clear before/after explanation
**Deleted code markers**: `TestAggregationExecutorOldInterfaceDeleted` explicitly verifies old methods removed, prevents resurrection
**Factory pattern emergence**: `make_mock_sink` shows awareness of duplication, just needs expansion
**Type narrowing assertions**: Uses `hasattr(state, "duration_ms")` to satisfy type checker after status check, good defensive practice

## Recommendations

**Immediate (P0)**:
1. Create pytest fixtures for common setup (db, recorder, run, node) - eliminate 1000+ lines of duplication
2. Add terminal state invariant test: verify every token reaches exactly one terminal state across all executor tests

**High Priority (P1)**:
1. Replace memory-intensive checkpoint size tests (lines 3629, 3814) with mocked size calculations
2. Add property-based tests for aggregation buffering (trigger conditions, count accuracy)
3. Add mutation tests for buffered data isolation (verify defensive copying)
4. Add NaN/Infinity rejection tests per CLAUDE.md requirements

**Medium Priority (P2)**:
1. Mark checkpoint roundtrip tests as `@pytest.mark.integration` or move to integration/
2. Replace mock-based `test_execute_transform_records_attempt_number` with landscape query
3. Create builder functions for TokenInfo, mock transforms, mock gates
4. Document thread-safety expectations for executor state management
5. Fix `test_execute_flush_detects_incomplete_restoration` - either reach corruption state validly or delete

**Low Priority (P3)**:
1. Consolidate checkpoint format validation tests into single parametrized test
2. Remove logging mock tests (size warnings) unless logging is contractual
3. Consider RFC 2119 naming convention for clarity
