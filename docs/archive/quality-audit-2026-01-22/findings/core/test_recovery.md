# Test Quality Review: test_recovery.py

## Summary

The recovery test suite has good structural coverage of basic resume scenarios but suffers from **critical gaps in corruption detection, mutation testing, and integration-level validation**. Tests verify happy paths effectively but fail to verify that the recovery system correctly handles database corruption (Tier 1 "our data" must crash), checkpoint tampering, or race conditions. Several tests have **incomplete assertions** that pass without verifying critical behavior. Infrastructure is duplicated across test classes instead of leveraged from conftest. No property-based testing despite ideal suitability for invariants.

## Poorly Constructed Tests

### Test: test_get_resume_point (line 197)
**Issue**: Incomplete assertions - tests presence but not correctness
**Evidence**:
```python
assert resume_point.token_id is not None
assert resume_point.node_id is not None
assert resume_point.sequence_number > 0
```
These assertions pass for **ANY** non-null values. Doesn't verify that `token_id == "tok-001"`, `node_id == "node-001"`, or `sequence_number == 1` as created by the fixture.
**Fix**: Assert exact values from fixture:
```python
assert resume_point.token_id == "tok-001"
assert resume_point.node_id == "node-001"
assert resume_point.sequence_number == 1
```
**Priority**: P1

### Test: test_can_resume_returns_true_for_failed_run_with_checkpoint (line 160)
**Issue**: Missing graph topology validation test
**Evidence**: Tests that `can_resume=True` but doesn't verify that the returned `ResumeCheck` validated the checkpoint's upstream topology against the provided graph. Recovery code calls `CheckpointCompatibilityValidator().validate(checkpoint, graph)` but test doesn't confirm this happened.
**Fix**: Create a test with intentionally incompatible graph topology (different upstream nodes) and verify `can_resume=False` with reason containing "incompatible" or "topology".
**Priority**: P1

### Test: test_get_resume_point_with_aggregation_state (line 218)
**Issue**: Creates its own fixtures instead of using fixture composition; duplicates setup code
**Evidence**: Manually creates run_id, inserts tables, creates checkpoint_manager - all logic already in fixtures. This is 50+ lines of duplicated setup.
**Fix**: Refactor to use parameterized fixture or helper function from conftest. Setup logic should live in ONE place.
**Priority**: P2

### Test: test_frozen (line 459)
**Issue**: Generic immutability test belongs in contract tests, not recovery tests
**Evidence**: Tests that `ResumeCheck` is frozen (immutable dataclass). This is a **contract test** for the dataclass, not a recovery protocol test. Doesn't test recovery behavior.
**Fix**: Move to `tests/contracts/test_audit.py` where other contract dataclass tests live.
**Priority**: P3

## Critical Missing Tests (Corruption Detection)

### Missing: Checkpoint references deleted token
**Issue**: No test for checkpoint pointing to non-existent token_id
**Evidence**: `get_unprocessed_rows()` has this code:
```python
if checkpointed_row_result is None:
    raise RuntimeError(
        f"Checkpoint references non-existent token: {checkpoint.token_id}. "
        "This indicates database corruption or a bug in checkpoint creation."
    )
```
But no test validates this crash path. Per CLAUDE.md Tier 1 Trust Model: "Bad data in the audit trail = crash immediately."
**Fix**: Test that creates checkpoint with `token_id="ghost-token"` (doesn't exist in tokens table), then calls `get_unprocessed_rows()` and asserts `RuntimeError` is raised with correct message.
**Priority**: P0

### Missing: Checkpoint with NULL critical fields
**Issue**: No test for corrupted checkpoint with NULL token_id, node_id, or sequence_number
**Evidence**: Per CLAUDE.md: "NULL where unexpected = crash". Checkpoints are Tier 1 data. No test validates that NULL in critical fields causes immediate crash.
**Fix**: Manually insert checkpoint row with `token_id=NULL`, attempt `can_resume()`, verify crash (not coercion, not silent return False).
**Priority**: P0

### Missing: Run status corruption
**Issue**: No test for invalid run status enum value in database
**Evidence**: `_get_run()` reads `runs_table` but no test validates crash on invalid enum. Per CLAUDE.md: "invalid enum value = crash". If DB contains `status="garbage"`, system must crash.
**Fix**: Manually insert run with `status="invalid_status"`, attempt `can_resume()`, verify crash with enum validation error.
**Priority**: P0

### Missing: Checkpoint topology_json corruption
**Issue**: No test for malformed JSON in checkpoint.topology_json
**Evidence**: Checkpoints store graph topology as JSON. If JSON is corrupted (`topology_json="{{malformed"`), deserialization will fail. No test validates this crash path.
**Fix**: Manually corrupt `topology_json` field, attempt `can_resume()`, verify crash during validation.
**Priority**: P1

### Missing: Row index out-of-order
**Issue**: No test for rows with non-sequential row_index (gaps or duplicates)
**Evidence**: `get_unprocessed_rows()` relies on `row_index` ordering. If database has rows [0, 1, 1, 3] (duplicate index 1) or [0, 2, 4] (gaps), behavior is undefined. Should this crash? Return duplicates? Tests don't specify.
**Fix**: Create scenario with duplicate row_index values, verify behavior matches Tier 1 trust model (likely should crash on corruption).
**Priority**: P1

## Missing Tests (Edge Cases)

### Missing: Multiple checkpoints for same run
**Issue**: No test validates that `get_latest_checkpoint()` correctly returns the LATEST checkpoint when multiple exist
**Evidence**: Method name implies "latest" selection but no test validates ordering. If run has checkpoints at sequence [1, 5, 3], does it return sequence=5? Tests don't verify.
**Fix**: Create run with 3 checkpoints at different sequences, verify `get_latest_checkpoint()` returns highest sequence_number.
**Priority**: P1

### Missing: Checkpoint with zero sequence_number
**Issue**: No test for edge case where checkpoint has `sequence_number=0` (first row)
**Evidence**: Current tests use sequence ≥1. If checkpoint is at sequence=0, does `get_unprocessed_rows()` return all rows or rows with index>0? Boundary condition untested.
**Fix**: Create checkpoint with `sequence_number=0`, verify correct row boundary.
**Priority**: P2

### Missing: Run with only checkpointed row (no unprocessed rows)
**Issue**: Test exists (line 400) but creates checkpoint at row 4, not at the LAST row
**Evidence**: `test_returns_empty_list_when_all_rows_processed` creates 5 rows (0-4) and checkpoints at row 4. This is correct, but doesn't test the case where run has ONLY 1 row and it's checkpointed (edge case of single-row run).
**Fix**: Add test with 1 row, checkpoint at that row, verify empty unprocessed list.
**Priority**: P3

### Missing: Resume point with empty aggregation_state
**Issue**: No test for checkpoint with `aggregation_state_json=""` (empty string vs NULL)
**Evidence**: `get_resume_point()` checks `if checkpoint.aggregation_state_json:` but doesn't distinguish empty string from NULL. Different semantics?
**Fix**: Create checkpoint with `aggregation_state_json=""`, verify it's treated as None (not deserialized as empty object).
**Priority**: P3

## Misclassified Tests

### Test Class: TestResumeCheck (line 426)
**Issue**: Unit tests for dataclass validation belong in contracts, not recovery tests
**Evidence**: Tests `ResumeCheck` dataclass invariants (can_resume=True must not have reason, etc.). These are **contract tests** for the dataclass structure, not recovery protocol tests.
**Fix**: Move entire class to `tests/contracts/test_audit.py` where `Run`, `Node`, `Edge` dataclass tests should live.
**Priority**: P2

### Test: test_handles_nonexistent_run_id_gracefully (line 419)
**Issue**: Method name says "gracefully" but returns empty list - is this graceful or a bug?
**Evidence**: `get_unprocessed_rows("ghost-run")` returns `[]` (empty list). Per Tier 1 trust model, should this crash? If run_id doesn't exist, that's either (a) caller bug (should crash) or (b) valid check (should return explicit "not found"). Returning `[]` conflates "no unprocessed rows" with "run doesn't exist".
**Fix**: Either (1) make it crash on non-existent run (Tier 1 strict), or (2) change return type to `list[str] | None` where `None` means "run not found". Current behavior is ambiguous.
**Priority**: P1

## Infrastructure Gaps

### Gap: Duplicated database setup across test classes
**Issue**: Every test class duplicates `landscape_db`, `checkpoint_manager`, `recovery_manager` fixtures
**Evidence**: Lines 17-27, 286-296, 475-485, 581-591 all define identical fixtures. Violates DRY.
**Fix**: Define these fixtures ONCE in conftest.py at module level. Test classes should use session/module-scoped fixtures.
**Priority**: P2

### Gap: Helper methods (_setup_run_with_rows, _setup_fork_scenario, _setup_failure_scenario) should be fixtures
**Issue**: Helper methods create test data but aren't reusable as fixtures; requires manual call in each test
**Evidence**: `_setup_run_with_rows()` (line 298), `_setup_fork_scenario()` (line 487), `_setup_failure_scenario()` (line 593) are all helper methods that should be parameterized fixtures.
**Fix**: Convert to `@pytest.fixture` with parameters, or create fixture factory pattern. Example:
```python
@pytest.fixture
def run_with_rows(landscape_db, checkpoint_manager, request):
    return _setup_run_with_rows(landscape_db, checkpoint_manager, create_checkpoint=request.param)
```
**Priority**: P2

### Gap: No fixture for corrupted checkpoints
**Issue**: Testing corruption scenarios requires manual SQL injection in each test; no reusable corruption fixture
**Evidence**: To test corruption, tests must manually insert malformed data. No `corrupted_checkpoint` fixture.
**Fix**: Create fixture factory for corrupted checkpoints:
```python
@pytest.fixture
def corrupt_checkpoint(checkpoint_manager):
    """Factory for creating checkpoints with specific corruptions."""
    def _factory(run_id, corruption_type):
        # corruption_type: "null_token_id", "invalid_json", "missing_topology", etc.
        ...
    return _factory
```
**Priority**: P2

### Gap: No property-based tests for invariants
**Issue**: Recovery protocol has clear invariants but no Hypothesis tests
**Evidence**: CLAUDE.md lists Hypothesis in technology stack. Clear invariants exist:
- `len(get_unprocessed_rows(run_id)) + checkpointed_rows == total_rows`
- For any checkpoint at row N, unprocessed rows must all have `row_index > N`
- `get_resume_point(run_id).sequence_number` must match `checkpoint.sequence_number`

Property testing would catch boundary bugs (off-by-one in row index comparison).
**Fix**: Create `tests/property/checkpoint/test_recovery_invariants.py`:
```python
@given(total_rows=st.integers(min_value=1, max_value=100),
       checkpoint_at=st.integers(min_value=0, max_value=99))
def test_unprocessed_rows_partition_invariant(total_rows, checkpoint_at):
    assume(checkpoint_at < total_rows)
    # Setup run with total_rows, checkpoint at checkpoint_at
    # Assert: unprocessed = [checkpoint_at+1 .. total_rows-1]
```
**Priority**: P1

### Gap: No marker for integration tests
**Issue**: Some tests are actually integration tests (write to real DB, use real CheckpointManager) but not marked
**Evidence**: All tests in file use real SQLite database (via `tmp_path`), real CheckpointManager, real LandscapeDB. These are integration tests, not unit tests.
**Fix**: Add `@pytest.mark.integration` to all test classes. Create pure unit tests that mock LandscapeDB.
**Priority**: P3

## Positive Observations

- **Good scenario coverage for fork and failure cases**: `TestGetUnprocessedRowsForkScenarios` and `TestGetUnprocessedRowsFailureScenarios` test non-trivial row boundary calculations (fork where sequence != row_index).
- **Clear test organization**: Test classes are well-named and focused on specific aspects (recovery protocol, unprocessed rows, resume checks).
- **Fixture composition in conftest**: `conftest.py` provides reusable `_create_test_graph()` helper and shared fixtures.
- **Realistic test data**: Fixtures create realistic run/node/row/token structures that mirror production usage.
- **Correct test isolation**: Each test uses `tmp_path` for independent database, preventing cross-test pollution.

## Summary Statistics

| Category | Count | Priority Breakdown |
|----------|-------|-------------------|
| Poorly Constructed Tests | 4 | P1: 2, P2: 1, P3: 1 |
| Critical Missing Tests (Corruption) | 5 | P0: 3, P1: 2 |
| Missing Tests (Edge Cases) | 4 | P1: 1, P2: 1, P3: 2 |
| Misclassified Tests | 2 | P1: 1, P2: 1 |
| Infrastructure Gaps | 5 | P1: 1, P2: 3, P3: 1 |

**Total Issues**: 20 (P0: 3, P1: 5, P2: 5, P3: 5)

## Recommended Actions (Priority Order)

1. **P0 - Add corruption detection tests** (missing tests for NULL fields, invalid enums, phantom token references) - these are **audit integrity violations** per CLAUDE.md
2. **P1 - Fix incomplete assertions** (test_get_resume_point, topology validation)
3. **P1 - Add property-based invariant tests** (using Hypothesis as specified in tech stack)
4. **P1 - Clarify graceful vs crash semantics** for non-existent run_id
5. **P2 - Consolidate fixtures** to conftest, eliminate duplication
6. **P2 - Create corruption fixture factory** for reusable corruption testing
7. **P3 - Move contract tests** to appropriate test module
