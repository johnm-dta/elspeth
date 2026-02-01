# Test Quality Review: test_recorder.py

## Summary

The test file contains 3,766 lines of unit tests for LandscapeRecorder, the audit trail backbone. Tests are generally well-structured with good coverage of edge cases. However, there are critical gaps in audit trail integrity verification, pervasive fixture duplication creating maintenance burden, and several tests that make unverifiable claims about auditability.

## Poorly Constructed Tests

### Test: test_begin_run (line 23)
**Issue**: Incomplete audit verification - only checks presence of fields, not integrity
**Evidence**:
```python
assert run.run_id is not None
assert run.status == RunStatus.RUNNING
assert run.started_at is not None
```
**Fix**: Add assertions for:
- `config_hash` is non-NULL and matches canonical hash of config
- `canonical_version` stored correctly
- `started_at` is reasonable (not far in past/future)
**Priority**: P1 (core auditability principle: "Hashes survive payload deletion - integrity is always verifiable")

### Test: test_create_row (line 375)
**Issue**: Does not verify source data hash integrity
**Evidence**: `assert row.source_data_hash is not None` - checks presence, not correctness
**Fix**:
```python
from elspeth.core.canonical import stable_hash
expected_hash = stable_hash({"value": 42})
assert row.source_data_hash == expected_hash
```
**Priority**: P1 (auditability standard: hash integrity is foundational)

### Test: test_complete_node_state_success (line 668)
**Issue**: Does not verify hash of output data
**Evidence**: `assert completed.output_hash is not None` - presence check only
**Fix**: Compare output_hash against stable_hash(output_data)
**Priority**: P1 (every transform boundary must have verifiable hashes)

### Test: test_record_routing_event (line 929)
**Issue**: Does not verify routing reason was recorded correctly
**Evidence**: Creates routing event with reason dict, never verifies it's persisted
**Fix**: Retrieve the event and assert `event.reason_json` deserializes to expected dict
**Priority**: P2 (gate decisions must be fully auditable)

### Test: test_explain_row_with_corrupted_payload (line 2183)
**Issue**: Test claims payload is "corrupted" but only tests invalid JSON
**Evidence**: `corrupted_data = b"this is not valid json {{{{"`
**Fix**: Add separate tests for:
- Hash mismatch (payload doesn't match stored hash)
- File exists but is unreadable (permissions)
- Partial file corruption (truncated JSON)
**Priority**: P2 (payload integrity is critical for replay)

### Test: test_retry_increments_attempt (line 881)
**Issue**: Does not verify both attempts are recorded in audit trail
**Evidence**: Creates two node states with attempt 0 and 1, only checks second one's attempt number
**Fix**: Query all states for the token and verify both attempts exist with correct ordering
**Priority**: P2 (retry semantics: "Each attempt recorded separately")

### Test: test_batch_lifecycle (line 1180)
**Issue**: Does not verify state transitions are recorded with timestamps
**Evidence**: Updates batch status multiple times, never checks transition audit trail
**Fix**: Verify `completed_at` is NULL for draft/executing, non-NULL for completed
**Priority**: P2 (terminal states must have completion timestamps)

### Test: test_get_artifacts_for_run (line 1320)
**Issue**: Does not verify content_hash integrity or size_bytes accuracy
**Evidence**: Registers artifacts with arbitrary hash/size values, never validates them
**Fix**: Compute actual hash of dummy content, verify it matches registered hash
**Priority**: P1 (artifact integrity: "Final artifacts with content hashes")

### Test: test_fork_token (line 430)
**Issue**: Does not verify fork_group_id is non-NULL and shared
**Evidence**: Asserts children share fork_group_id but doesn't check it's not None first
**Fix**:
```python
assert child_tokens[0].fork_group_id is not None
assert child_tokens[1].fork_group_id is not None
assert child_tokens[0].fork_group_id == child_tokens[1].fork_group_id
```
**Priority**: P2 (token lineage must be complete)

### Test: test_coalesce_tokens (line 466)
**Issue**: Does not verify parent relationships are recorded in token_parents table
**Evidence**: Asserts merged token has join_group_id but doesn't check parent links
**Fix**: Call `recorder.get_token_parents(merged.token_id)` and verify both children are listed
**Priority**: P2 (DAG lineage must be queryable)

### Test: test_pure_pipeline_gets_full_reproducible (line 2284)
**Issue**: Does not verify determinism values are actually stored
**Evidence**: Registers nodes with Determinism.DETERMINISTIC, assumes they're stored correctly
**Fix**: Retrieve nodes via `get_node()` and assert `node.determinism == "deterministic"`
**Priority**: P2 (reproducibility grade depends on stored determinism values)

### Test: test_explain_with_missing_row_payload (line 1950)
**Issue**: Test name implies "missing payload" but actually tests "purged payload"
**Evidence**: Creates payload, then explicitly deletes it - that's purged, not missing
**Fix**: Rename to `test_explain_with_purged_row_payload` for accuracy
**Priority**: P3 (clarity)

### Test: test_explain_row_rejects_run_id_mismatch (line 2227)
**Issue**: Does not test token-level mismatch (token from different run)
**Evidence**: Only tests row belonging to wrong run, not tokens
**Fix**: Add test where token belongs to different run than queried run_id
**Priority**: P2 (prevents audit trail cross-contamination)

## Misclassified Tests

### Test Class: TestReproducibilityGradeComputation (line 2281)
**Issue**: Should be integration tests, not unit tests
**Evidence**: Tests compute grade from multiple nodes across database, involves complex state
**Fix**: Move to `tests/integration/test_reproducibility_grade.py`
**Priority**: P2 (unit tests should test single method in isolation)

### Test Class: TestExplainGracefulDegradation (line 1947)
**Issue**: These are integration tests with filesystem payload store
**Evidence**: Uses `tmp_path` fixture, tests interaction between recorder and payload store
**Fix**: Move to `tests/integration/test_explain_with_payloads.py`
**Priority**: P2 (filesystem operations = integration test)

### Test: test_record_transform_error_stores_in_database (line 2808)
**Issue**: Direct SQL query against database - should be in database layer tests
**Evidence**: `conn.execute(select(transform_errors_table)...)`
**Fix**: Use recorder's query methods, or move to `test_database.py` if testing raw SQL
**Priority**: P3 (violates abstraction boundary)

## Infrastructure Gaps

### Gap: No fixtures for common setup patterns
**Issue**: Every test creates db, recorder, run, nodes manually
**Evidence**: Lines 28-29, 45-46, 68-69, etc. - identical 2-line setup repeated 100+ times
**Fix**: Create pytest fixtures:
```python
@pytest.fixture
def recorder() -> LandscapeRecorder:
    db = LandscapeDB.in_memory()
    return LandscapeRecorder(db)

@pytest.fixture
def run(recorder: LandscapeRecorder) -> Run:
    return recorder.begin_run(config={}, canonical_version="v1")

@pytest.fixture
def source_node(recorder: LandscapeRecorder, run: Run) -> Node:
    return recorder.register_node(
        run_id=run.run_id,
        plugin_name="test_source",
        node_type="source",
        plugin_version="1.0",
        config={},
        schema_config=DYNAMIC_SCHEMA,
    )
```
**Priority**: P1 (reduces 3,766 lines to ~2,500 lines, eliminates mutation risk)

### Gap: No helper for creating token lineage chains
**Issue**: Fork/coalesce setup repeated in multiple tests with subtle variations
**Evidence**: Lines 430-465, 819-860, 1820-1860 all recreate fork/join scenarios
**Fix**: Create helper method:
```python
def create_fork_join_scenario(recorder, run_id, branches=["a", "b"]):
    """Creates row -> parent token -> fork to N branches -> coalesce."""
    # ... implementation ...
    return (parent_token, child_tokens, merged_token)
```
**Priority**: P2 (reduces duplication, ensures correct setup)

### Gap: No property-based tests for hash stability
**Issue**: Hash integrity is critical but only tested with hand-picked examples
**Evidence**: Manual assertions like `assert row.source_data_hash is not None`
**Fix**: Add Hypothesis property tests:
```python
@given(st.dictionaries(st.text(), st.integers()))
def test_row_hash_deterministic(data):
    # Create row twice with same data, verify hash matches
```
**Priority**: P1 (hashes are foundational to audit integrity)

### Gap: No tests for concurrent recorder access
**Issue**: Multiple processes may write to landscape simultaneously
**Evidence**: No tests simulate concurrent `begin_run`, `create_row`, etc.
**Fix**: Add concurrency tests (probably in integration suite):
```python
def test_concurrent_row_creation():
    # Spawn threads creating rows for same run
    # Verify all rows recorded with unique IDs
```
**Priority**: P2 (production will have concurrent writes)

### Gap: No tests for database constraint violations
**Issue**: FK constraints exist but not tested (e.g., token without valid row_id)
**Evidence**: Test at line 2730 manually creates dependencies for FK constraints but doesn't test what happens when violated
**Fix**: Add tests that attempt to violate constraints and verify they're rejected
**Priority**: P2 (constraints are defense-in-depth for audit integrity)

### Gap: Helper method `_create_token_with_dependencies` is test-specific infrastructure
**Issue**: Method at line 2730 belongs in test utilities, not inline in test class
**Evidence**: `@staticmethod` decorator suggests it's utility code
**Fix**: Move to `tests/helpers/landscape_helpers.py` for reuse
**Priority**: P3 (test organization)

### Gap: No parametrized tests for enum coercion
**Issue**: Enum coercion tested separately for each enum (RunStatus, NodeType, etc.)
**Evidence**: Lines 84-227 test RunStatus coercion, 254-305 test NodeType, 3404-3524 test ExportStatus
**Fix**: Parametrize over enums:
```python
@pytest.mark.parametrize("method,enum_type,valid_values", [
    ("begin_run", RunStatus, ["running", "completed"]),
    ("register_node", NodeType, ["source", "transform"]),
    # ...
])
def test_enum_coercion(method, enum_type, valid_values):
    # Test enum vs string for each method
```
**Priority**: P2 (reduces 200+ lines of duplication)

### Gap: No tests for recorder state isolation between runs
**Issue**: Tests assume in-memory database is fresh, don't verify isolation
**Evidence**: No test creates run1, run2, and verifies data doesn't leak
**Fix**: Add test:
```python
def test_runs_isolated():
    run1 = recorder.begin_run(...)
    run2 = recorder.begin_run(...)
    row1 = recorder.create_row(run_id=run1.run_id, ...)
    # Verify get_rows(run2.run_id) doesn't return row1
```
**Priority**: P2 (prevents audit trail cross-contamination)

## Audit Trail Completeness Gaps

### Gap: No verification that hashes persist after payload deletion
**Issue**: CLAUDE.md states "Hashes survive payload deletion - integrity is always verifiable"
**Evidence**: Tests delete payloads but only check `payload_available=False`, don't verify hash remains
**Fix**: In test_explain_with_missing_row_payload (line 1950):
```python
# After deleting payload
lineage = recorder.explain_row(...)
assert lineage.source_data_hash is not None  # Hash preserved!
# Verify hash matches what was originally stored
original_row = recorder.get_row(row.row_id)
assert lineage.source_data_hash == original_row.source_data_hash
```
**Priority**: P0 (foundational auditability principle)

### Gap: No tests verify all terminal row states are actually terminal
**Issue**: CLAUDE.md lists 7 terminal states, no test verifies they're the only ones
**Evidence**: No test queries all rows in a completed run and verifies every row is in a terminal state
**Fix**: Add integration test that processes multiple rows and verifies no rows left in intermediate state
**Priority**: P1 ("Every row reaches exactly one terminal state - no silent drops")

### Gap: No tests for "no inference" principle
**Issue**: CLAUDE.md: "No inference - if it's not recorded, it didn't happen"
**Evidence**: No test verifies that missing data causes failures rather than defaults
**Fix**: Add tests that corrupt audit trail (set field to NULL) and verify reads crash
**Priority**: P1 (already partially covered by TestNodeStateIntegrityValidation but needs expansion)

### Gap: No tests verify config_hash covers complete configuration
**Issue**: Runs store config and config_hash but no test verifies all config changes produce different hashes
**Evidence**: Most tests pass empty config `{}`
**Fix**: Add test with nested config changes:
```python
config1 = {"source": {"path": "a.csv", "encoding": "utf-8"}}
config2 = {"source": {"path": "a.csv", "encoding": "latin1"}}
run1 = recorder.begin_run(config=config1, ...)
run2 = recorder.begin_run(config=config2, ...)
assert run1.config_hash != run2.config_hash  # Different encoding = different hash
```
**Priority**: P1 (config traceability is foundational)

### Gap: No tests for transform boundary recording completeness
**Issue**: CLAUDE.md: "Transform boundaries - Input AND output captured at every transform"
**Evidence**: Tests verify input_hash and output_hash exist but don't verify BOTH are required
**Fix**: Add test that attempts to complete node state with only input or only output, verify it fails
**Priority**: P1 (transform boundary completeness is mandatory)

### Gap: No tests verify external call recording
**Issue**: CLAUDE.md: "External calls - Full request AND response recorded"
**Evidence**: No tests verify LLM/API call recording captures both request and response
**Fix**: This may be outside recorder scope (plugin responsibility), but verify the schema supports it
**Priority**: P2 (may belong in plugin tests)

### Gap: No tests for sink output verification
**Issue**: CLAUDE.md: "Sink output - Final artifacts with content hashes"
**Evidence**: Tests create artifacts but don't verify the hash matches actual content
**Fix**: In test_register_artifact (line 1277):
```python
content = b"test,data\n1,2\n"
content_hash = hashlib.sha256(content).hexdigest()
artifact = recorder.register_artifact(..., content_hash=content_hash, ...)
# Later: verify hash matches actual file if path exists
```
**Priority**: P1 (artifact integrity is final audit proof)

## Mutation Vulnerability Tests

### Test: test_complete_node_state_success (line 668)
**Issue**: Mutates `input_data` dict after passing to begin_node_state
**Evidence**: Test doesn't verify immutability - could mutate and affect recorded hash
**Fix**: Add assertion:
```python
input_data = {"x": 1}
state = recorder.begin_node_state(..., input_data=input_data)
input_data["x"] = 999  # Mutate after recording
# Verify recorded hash didn't change
retrieved = recorder.get_node_states_for_token(token.token_id)[0]
expected_hash = stable_hash({"x": 1})  # Original value
assert retrieved.input_hash == expected_hash
```
**Priority**: P2 (defense against mutation bugs)

### Test: test_record_routing_event (line 929)
**Issue**: Passes mutable dict as `reason`, doesn't verify it's deep-copied
**Evidence**: `reason={"rule": "value > 1000", "result": True}` could be mutated
**Fix**: Similar to above - mutate reason after recording, verify recorded JSON unchanged
**Priority**: P2 (routing decisions must be immutable)

## Tests That Make Unverifiable Claims

### Test: test_explain_row_not_found (line 2092)
**Issue**: Asserts `lineage is None` but doesn't verify WHY (row doesn't exist vs. run mismatch vs. other error)
**Evidence**: Only checks None return, could be masking errors
**Fix**: Add negative tests with specific error cases and verify error messages
**Priority**: P3 (error clarity)

### Test: test_get_row_not_found (line 1771)
**Issue**: Same as above - None could mean many things
**Evidence**: `assert result is None` with no context
**Fix**: Check database state before and after to verify row truly doesn't exist
**Priority**: P3

### Test: test_update_grade_after_purge_nonexistent_run (line 2508)
**Issue**: "Silently handles nonexistent run" is a claim, not a verification
**Evidence**: Just calls function, doesn't verify it's a no-op (could have side effects)
**Fix**: Query database before and after, verify no changes occurred
**Priority**: P3 (defensive correctness)

## Positive Observations

**Excellent bug regression coverage**: Tests like `test_complete_node_state_with_empty_output` (line 751) and `test_fork_token_rejects_empty_branches` (line 545) show thorough edge case thinking and reference specific bug tickets.

**Good use of docstrings for context**: Tests reference CLAUDE.md principles and bug tickets directly in docstrings, making audit trail requirements explicit.

**Comprehensive enum validation**: TestLandscapeRecorderRunStatusValidation class thoroughly tests enum coercion, though it could be parametrized for DRY.

**Corruption testing**: TestNodeStateIntegrityValidation (line 3526) correctly tests Tier 1 audit integrity by corrupting database and expecting crashes.

**Good separation of concerns**: Tests focus on recorder interface without testing SQL directly (mostly - see misclassification above).

**Temporal ordering tests**: test_get_node_states_orders_by_step_index_and_attempt (line 3666) explicitly tests deterministic ordering for retries - critical for signed exports.
