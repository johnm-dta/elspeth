# Test Quality Review: test_checkpoint_recovery.py

## Summary
Integration test suite demonstrates strong architecture understanding but suffers from multiple critical quality issues: sleepy assertions (implicit database waits), mutable shared state via direct database manipulation, missing failure scenario coverage, and test interdependence through hardcoded IDs and shared fixtures.

## Poorly Constructed Tests

### Test: test_full_checkpoint_recovery_cycle (line 53)
**Issue**: Sleepy assertion - implicit database transaction wait without verification
**Evidence**: Lines 59-65 create checkpoint via `_setup_partial_run()`, then immediately query `get_latest_checkpoint()` assuming write has completed. No explicit transaction verification or retry logic.
```python
run_id = self._setup_partial_run(db, checkpoint_mgr, mock_graph)
# Implicit: assumes database write completed
checkpoint = checkpoint_mgr.get_latest_checkpoint(run_id)
```
**Fix**: Add explicit database flush/commit verification or use condition-based wait to confirm checkpoint persistence before querying.
**Priority**: P2

### Test: test_full_checkpoint_recovery_cycle (line 53)
**Issue**: Assertion-free code path - verifies can_resume but never verifies what happens if you DON'T resume
**Evidence**: Lines 68-77 verify happy path only. No test for "what if we ignore the checkpoint and try to restart fresh?"
**Fix**: Add negative test case verifying fresh restart behavior when checkpoint exists.
**Priority**: P3

### Test: test_checkpoint_sequence_ordering (line 79)
**Issue**: Weak assertion on ordering - uses `>=` allowing duplicate sequences
**Evidence**: Line 106 `assert len(checkpoints) >= 3` allows 4, 5, 6... checkpoints from prior test pollution
**Fix**: Use exact equality `assert len(checkpoints) == 3` with proper test isolation
**Priority**: P2

### Test: test_recovery_with_aggregation_state (line 117)
**Issue**: Missing validation of aggregation state type contract
**Evidence**: Lines 127-131 create aggregation state with arbitrary dict structure. No schema validation ensuring this matches expected AggregationState contract.
**Fix**: Use typed dataclass or validate against actual AggregationState schema. Test should fail if we pass wrong structure.
**Priority**: P1

### Test: test_checkpoint_cleanup_after_completion (line 149)
**Issue**: Test name claims "after completion" but actually tests "manual deletion"
**Evidence**: Line 161 `checkpoint_mgr.delete_checkpoints(run_id)` is manual cleanup, not automatic cleanup triggered by run completion signal.
**Fix**: Test actual completion path - call `orchestrator.complete_run()` or equivalent and verify checkpoints auto-delete via hook/cleanup logic.
**Priority**: P1

### Test: test_recovery_respects_checkpoint_boundary (line 168)
**Issue**: Tests equivalence but not correctness - verifies resume_point matches checkpoint, but never verifies checkpoint is at the CORRECT boundary
**Evidence**: Lines 176-184 compare `resume_point` to `checkpoint` fields, but never validate that sequence_number=2 is actually where processing stopped (could be wrong and test would pass)
**Fix**: Verify checkpoint boundary against actual processing state (e.g., rows_processed count, last completed token)
**Priority**: P2

### Test: test_multiple_runs_independent_checkpoints (line 186)
**Issue**: Incomplete isolation test - deletes run1 checkpoints and checks run2 still exists, but never verifies run2's checkpoint DATA is unchanged
**Evidence**: Lines 202-205 only check counts, not checkpoint content integrity
**Fix**: Before deletion, capture run2 checkpoint state (token_id, sequence_number). After deletion, verify run2 checkpoint content identical.
**Priority**: P3

### Test: _setup_partial_run (line 207)
**Issue**: Helper creates mutable shared state via direct SQL inserts, violating "our data" trust tier
**Evidence**: Lines 231-290 bypass Landscape API and directly insert rows via SQLAlchemy. If Landscape schema changes (e.g., add NOT NULL column), these tests silently break or create invalid data.
**Fix**: Use LandscapeRecorder API exclusively. Tests should break LOUDLY when schema contracts change.
**Priority**: P0 - violates CLAUDE.md "Our Data (Audit Database / Landscape) - FULL TRUST" principle

### Test: test_full_resume_processes_remaining_rows (line 293)
**Issue**: Output file mutation vulnerability - writes to output_path without cleanup, creates test interdependence
**Evidence**: Lines 435-439 write CSV header + 3 rows to `tmp_path / "resume_output.csv"`. If test fails before completion, file persists and pollutes next test run.
**Fix**: Use context manager or fixture with explicit cleanup. Better: use StringIO for in-memory CSV testing.
**Priority**: P2

### Test: test_full_resume_processes_remaining_rows (line 293)
**Issue**: Magic numbers without explanation - why 5 rows? Why checkpoint at row 2?
**Evidence**: Lines 411 `for i in range(5)`, line 442 `sequence_number=2`, expected outcomes hardcoded to these values
**Fix**: Add constants with explanatory names: `TOTAL_ROWS = 5`, `CHECKPOINT_AFTER_ROW = 2`, `EXPECTED_RESUMED_ROWS = TOTAL_ROWS - CHECKPOINT_AFTER_ROW - 1`
**Priority**: P3

### Test: test_full_resume_processes_remaining_rows (line 293)
**Issue**: Manual graph construction duplicates config - violates DRY
**Evidence**: Lines 318-322 create `test_graph`, then lines 479-490 recreate identical graph structure with different variable name `graph`
**Fix**: Extract graph construction to fixture or reuse `test_graph` variable
**Priority**: P3

### Test: test_full_resume_processes_remaining_rows (line 293)
**Issue**: Backdoor graph state mutation - directly sets private attributes
**Evidence**: Lines 488-490 `graph._sink_id_map = ...`, `graph._transform_id_map = ...` bypasses public API
**Fix**: Either expose public setters or construct graph via proper config-driven builder. Direct attribute access to `_private` fields violates encapsulation and breaks if internals change.
**Priority**: P1

### Test: test_checkpoint_hash_matches_graph_at_creation_time (line 526)
**Issue**: Conditional assertion reduces test determinism
**Evidence**: Lines 611-613 `if expected_hash != modified_hash:` - test behavior changes based on whether graph modification affects hash
**Fix**: Make graph modification ALWAYS affect hash (e.g., add edge, change config) so test path is deterministic
**Priority**: P2

### Test: test_checkpoint_hash_matches_graph_at_creation_time (line 526)
**Issue**: Test verifies Bug #1 fix but doesn't demonstrate the actual race condition
**Evidence**: Lines 599-600 comment "simulating what could happen in a race condition" but test is single-threaded - race condition never actually occurs
**Fix**: Use threading to create ACTUAL race: Thread 1 calls create_checkpoint(), Thread 2 modifies graph mid-transaction. Current test only proves hash is captured, not that it's captured ATOMICALLY.
**Priority**: P2

### Test: test_full_checkpoint_recovery_cycle (line 53)
**Issue**: No audit trail verification - tests API success but not Landscape data correctness
**Evidence**: Test verifies `can_resume()` returns True and `get_unprocessed_rows()` returns 2 rows, but never queries Landscape tables to verify checkpoint record structure, foreign keys, or data integrity.
**Fix**: After checkpoint creation, query `checkpoints_table` directly and verify: run_id matches, token_id exists in tokens_table, node_id exists in nodes_table, upstream_topology_hash is not NULL.
**Priority**: P1

## Missing Failure Scenarios

### Scenario: Database corruption in checkpoints table
**Evidence**: No tests verify behavior when checkpoint has NULL required field, wrong type, or orphaned foreign key
**Expected**: Test should verify checkpoint read fails fast with clear error (Tier 1 trust violation)
**Priority**: P0 - CLAUDE.md mandates "Bad data in audit trail = crash immediately"

### Scenario: Checkpoint version mismatch during resume
**Evidence**: No test creates checkpoint with old schema version, then attempts resume with new code version
**Expected**: Test should verify graceful rejection with clear error message
**Priority**: P1

### Scenario: Partial checkpoint write (transaction interrupted)
**Evidence**: No test simulates checkpoint write that commits run/row records but fails to write checkpoint record
**Expected**: Test should verify recovery detects incomplete state and either: (1) crashes with clear error, or (2) restarts from beginning
**Priority**: P1

### Scenario: Unprocessed rows have missing/deleted payloads
**Evidence**: `get_unprocessed_rows()` assumes payload refs are valid. No test for row with `source_data_ref` pointing to deleted/missing payload file
**Expected**: Test should verify resume either: (1) crashes with clear error, or (2) quarantines rows with missing payloads
**Priority**: P1

### Scenario: Checkpoint at sequence N, but rows only go to N-1
**Evidence**: No test for checkpoint claiming sequence_number=10 when only 9 rows exist in database
**Expected**: Test should verify resume detects inconsistency and crashes (data integrity violation)
**Priority**: P0 - inconsistent audit trail

### Scenario: Multiple checkpoints with same sequence_number
**Evidence**: No test creates duplicate checkpoint sequences (e.g., two checkpoints both claim sequence=5)
**Expected**: Test should verify either: (1) database rejects duplicate via UNIQUE constraint, or (2) get_latest_checkpoint() deterministically chooses winner
**Priority**: P1

### Scenario: Resume after aggregation timeout (partial batch)
**Evidence**: `test_recovery_with_aggregation_state` tests aggregation buffer recovery, but not timeout-triggered batches
**Expected**: Test should create checkpoint with aggregation buffer + timeout metadata, resume, and verify batch completes on time-based trigger
**Priority**: P2

### Scenario: Graph topology mismatch between checkpoint and resume
**Evidence**: Tests verify hash consistency, but never test what happens when hashes DON'T match (e.g., user changes pipeline config between crash and resume)
**Expected**: Test should create checkpoint with graph A, attempt resume with graph B, verify rejection with clear error
**Priority**: P1

### Scenario: Checkpoint references token_id that doesn't exist
**Evidence**: No test for orphaned checkpoint (token deleted but checkpoint remains)
**Expected**: Test should verify resume crashes with foreign key violation or clear error
**Priority**: P2

### Scenario: Concurrent resume attempts on same run_id
**Evidence**: No test for two orchestrators simultaneously calling `resume()` on same failed run
**Expected**: Test should verify either: (1) database lock prevents concurrent resume, or (2) second resume fails with clear error
**Priority**: P2

## Infrastructure Gaps

### Gap: No fixture for common graph topologies
**Evidence**: Lines 47-51, 318-322, 479-490, 543-545, 672 - each test manually constructs ExecutionGraph
**Fix**: Create `@pytest.fixture` for standard topologies: `linear_graph` (source→transform→sink), `fork_graph`, `aggregation_graph`
**Priority**: P2

### Gap: Database state leaks between test classes
**Evidence**: `tmp_path` fixture provides isolation, but if tests fail mid-execution, database state remains. No explicit cleanup in teardown.
**Fix**: Add `@pytest.fixture(autouse=True)` that ensures database cleanup even on test failure
**Priority**: P2

### Gap: No parametrized tests for checkpoint frequencies
**Evidence**: Only tests `frequency="every_row"` (line 35). ELSPETH supports multiple checkpoint strategies but tests hardcode one.
**Fix**: Use `@pytest.mark.parametrize("checkpoint_freq", ["every_row", "every_n_rows", "time_based"])` to verify all strategies
**Priority**: P3

### Gap: Repeated graph construction boilerplate
**Evidence**: Lines 318-322, 479-490, 543-545 duplicate identical graph structure
**Fix**: Extract to helper function `build_simple_pipeline_graph()` or fixture
**Priority**: P3

### Gap: No shared constants for test data values
**Evidence**: Hardcoded strings scattered: `"test-run-{run_suffix}"`, `"node-{run_suffix}"`, `"tok-{run_suffix}-{i:03d}"`
**Fix**: Define constants module for test IDs, makes pattern matching easier
**Priority**: P4

### Gap: No verification helpers for audit trail completeness
**Evidence**: Tests manually query `rows_table`, `tokens_table`, etc. No helper to verify "run_id X has complete audit trail"
**Fix**: Create `assert_audit_trail_complete(run_id)` helper that verifies all required tables populated
**Priority**: P3

### Gap: Manual SQL inserts bypass schema validation
**Evidence**: `_setup_partial_run()` uses raw `table.insert().values()` - if Landscape schema adds validation logic, tests bypass it
**Fix**: All test data creation should use LandscapeRecorder API
**Priority**: P0 - violates architecture trust model

## Misclassified Tests

### Test: test_checkpoint_cleanup_after_completion (line 149)
**Issue**: Misclassified as integration test - actually a unit test for CheckpointManager.delete_checkpoints()
**Evidence**: Only tests single component (CheckpointManager), no interaction with Orchestrator or full pipeline
**Fix**: Move to `tests/core/checkpoint/test_manager.py` as unit test
**Priority**: P3

### Test: test_checkpoint_validates_graph_parameter (line 615)
**Issue**: Misclassified as integration test - actually validates input contract (unit test)
**Evidence**: Tests CheckpointManager parameter validation in isolation, no integration with other components
**Fix**: Move to `tests/core/checkpoint/test_manager.py` under "input validation" section
**Priority**: P3

### Test: test_checkpoint_validates_node_exists_in_graph (line 660)
**Issue**: Misclassified as integration test - validates input contract (unit test)
**Evidence**: Tests CheckpointManager parameter validation in isolation
**Fix**: Move to `tests/core/checkpoint/test_manager.py` under "input validation" section
**Priority**: P3

## Test Interdependence Issues

### Issue: Shared mock_graph fixture is mutable
**Evidence**: Line 47-51 returns single ExecutionGraph instance. If any test mutates graph (e.g., adds node), affects all subsequent tests using same fixture in session.
**Fix**: Change fixture scope or return fresh instance: `@pytest.fixture` (without scope) creates new instance per test
**Priority**: P1

### Issue: Hardcoded run_id patterns create collision risk
**Evidence**: `_setup_partial_run()` uses `run_id = f"test-run-{run_suffix}"` - if two tests use same run_suffix, database collision
**Fix**: Use `uuid.uuid4()` or `f"test-run-{run_suffix}-{uuid.uuid4()}"` for guaranteed uniqueness
**Priority**: P2

### Issue: Tests assume database starts empty
**Evidence**: `test_checkpoint_sequence_ordering` line 106 `assert len(checkpoints) >= 3` - breaks if prior test left checkpoints
**Fix**: Either use strict equality checks, or add cleanup to ensure test isolation
**Priority**: P2

### Issue: test_full_resume_processes_remaining_rows creates specific file path
**Evidence**: Line 325 `output_path = tmp_path / "resume_output.csv"` - if test fails, file persists
**Fix**: Use `tmp_path / f"resume_output_{uuid.uuid4()}.csv"` or cleanup in finally block
**Priority**: P3

## Positive Observations

- **Strong Bug #1 regression test**: `test_checkpoint_hash_matches_graph_at_creation_time()` clearly documents race condition and verifies fix with explicit before/after hash comparison
- **Good use of helper methods**: `_setup_partial_run()` reduces boilerplate, though implementation needs improvement
- **Comprehensive coverage of resume path**: `test_full_resume_processes_remaining_rows()` is end-to-end test that verifies actual rows processed, not just API calls
- **Proper use of tmp_path fixture**: Isolates file system state per test
- **Clear test names**: Most tests have descriptive names that explain intent (e.g., `test_multiple_runs_independent_checkpoints`)

## Risk Assessment

**CRITICAL RISKS (P0):**
1. Tests bypass Landscape API via direct SQL inserts - violates Tier 1 trust model
2. Missing tests for corrupted checkpoint data - contradicts "crash immediately on audit trail anomaly"
3. Missing tests for checkpoint/row count inconsistencies - could allow silent data loss

**HIGH RISKS (P1):**
1. No version mismatch testing - production resume could fail unpredictably
2. No missing payload testing - resume could crash on valid checkpoint
3. Aggregation state lacks schema validation - could persist invalid state
4. Backdoor mutation of graph internals - tests break if implementation changes
5. No audit trail verification - tests verify API but not data correctness

**RECOMMENDATIONS:**
1. Rewrite `_setup_partial_run()` to use LandscapeRecorder API exclusively
2. Add comprehensive failure scenario tests (10+ missing cases identified above)
3. Extract graph construction to fixtures to reduce duplication
4. Move input validation tests to unit test suite
5. Add concurrent resume testing for production readiness
