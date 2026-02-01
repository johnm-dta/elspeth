# Test Quality Review: test_orchestrator_resume.py

## Summary
This test suite covers basic resume functionality but has critical gaps in verifying resume correctness, idempotence, and audit trail integrity. The tests verify happy-path row processing but lack tests for failure scenarios, stateful aggregation resume, audit trail completeness, and potential data corruption during resume. Several tests have weak assertions that don't verify correctness deeply enough.

## Poorly Constructed Tests

### Test: test_resume_processes_unprocessed_rows (line 312)
**Issue**: Weak assertions - counts processed rows but doesn't verify WHICH rows were processed
**Evidence**:
```python
assert result.rows_processed == 2
assert result.rows_succeeded == 2
```
The test verifies that 2 rows were processed but doesn't verify that rows 3 and 4 (the unprocessed ones) were actually processed. What if resume accidentally reprocessed rows 0 and 1 instead? The count would be correct but the behavior wrong.
**Fix**: Add assertions to verify the audit trail shows rows 3 and 4 were processed, not rows 0-2. Query the `tokens_table` and verify which row_ids have terminal states after resume.
**Priority**: P1

### Test: test_resume_writes_to_sink (line 350)
**Issue**: Extremely weak assertion - checks for substrings "data-3" and "data-4" in CSV content without verifying correct structure or absence of duplicates
**Evidence**:
```python
assert "data-3" in content
assert "data-4" in content
```
This would pass even if the sink wrote "garbage-data-3-data-4-garbage" to a single line. It doesn't verify CSV structure, doesn't check that rows 0-2 are absent, and doesn't verify correct column structure.
**Fix**: Parse the CSV properly and verify:
1. Exactly 2 data rows (plus header)
2. Row IDs are 3 and 4
3. Values match expected structure
4. Rows 0-2 are NOT present
5. No duplicate rows
**Priority**: P0

### Test: test_resume_returns_run_result_with_status (line 420)
**Issue**: Assertion-free test - verifies only type and non-negative counts, not correctness
**Evidence**:
```python
assert result.rows_processed >= 0  # Vacuous assertion
assert result.rows_succeeded >= 0  # Would pass even if all failed
assert result.rows_failed >= 0     # Would pass even if all succeeded
```
These assertions are nearly worthless. `>= 0` would pass for ANY resume result including complete failure.
**Fix**: Assert exact expected values based on fixture setup:
```python
assert result.rows_processed == 2
assert result.rows_succeeded == 2
assert result.rows_failed == 0
assert result.status == RunStatus.COMPLETED
```
**Priority**: P0

### Test: failed_run_with_payloads fixture (line 67)
**Issue**: Hidden mutation vulnerability - fixture builds graph with internal state assignments that could leak between tests
**Evidence**:
```python
# Line 304-308 in _create_test_graph()
graph._sink_id_map = {"default": "sink-node"}
graph._transform_id_map = {0: "transform-node"}
graph._config_gate_id_map = {}
graph._output_sink = "default"
graph._route_resolution_map = {}
```
Direct mutation of private `_` prefixed attributes bypasses graph's API. If ExecutionGraph changes implementation, these tests silently break. Worse, this fixture creates mutable state that could leak if tests fail mid-execution.
**Fix**: Use ExecutionGraph's public API to build the graph. If the API doesn't support necessary setup, that's a design smell - fix the API, don't hack around it.
**Priority**: P2

## Missing Critical Test Coverage

### Missing: Resume idempotence verification
**Issue**: No test verifies that calling resume() twice doesn't duplicate data or corrupt audit trail
**Evidence**: No test with pattern `resume(); resume();`
**Scenario**: If a resume operation appears to fail but actually wrote data, operator re-runs resume. Does it:
1. Detect already-processed rows and skip them?
2. Write duplicate data to sinks?
3. Create duplicate audit trail entries?
**Fix**: Add test:
```python
def test_resume_is_idempotent():
    # Resume once
    result1 = orchestrator.resume(...)
    # Resume again with same checkpoint
    result2 = orchestrator.resume(...)
    # Should detect no unprocessed rows
    assert result2.rows_processed == 0
    # Verify sink has no duplicates
    # Verify audit trail has no duplicate entries
```
**Priority**: P0 - Production correctness requirement per CLAUDE.md

### Missing: Resume with stateful aggregation
**Issue**: Resume restores aggregation state but no test verifies state is used correctly
**Evidence**: `restored_state` is set in resume() but never tested
**Scenario**: Run fails mid-batch (e.g., collected 3 of 5 rows). Resume should restore the partial batch and continue. Does it?
**Fix**: Add test with aggregation transform:
```python
def test_resume_restores_aggregation_state():
    # Create checkpoint with partial batch (3 rows buffered)
    # Resume should restore batch and process remaining 2 rows
    # Verify batch emits when complete
    # Verify audit trail shows correct batch composition
```
**Priority**: P1 - Phase 5 requirement (Production: Checkpointing)

### Missing: Resume failure scenarios
**Issue**: Only happy-path resume tested, no error handling verification
**Evidence**: No tests for:
- Row failure during resume (should quarantine, not crash)
- Sink failure during resume (retry? fail gracefully?)
- Checkpoint corruption during resume
**Fix**: Add tests for failure paths:
```python
def test_resume_quarantines_failed_rows()
def test_resume_handles_sink_failure()
def test_resume_with_corrupted_checkpoint_crashes()
```
**Priority**: P1

### Missing: Audit trail completeness after resume
**Issue**: No test verifies audit trail integrity after resume
**Evidence**: No assertions query `nodes_table`, `edges_table`, or verify terminal states
**Scenario**: Does resume correctly update row terminal states? Are transform boundaries recorded? Can we explain() a resumed row's lineage?
**Fix**: Add test:
```python
def test_resume_maintains_audit_trail_integrity():
    result = orchestrator.resume(...)
    # Verify all resumed rows have terminal states
    # Verify transform boundaries recorded
    # Verify explain() works for resumed rows
    lineage = landscape.explain(run_id, row_id="row-003")
    assert lineage.source_row is not None
    assert len(lineage.node_states) > 0
```
**Priority**: P0 - Auditability Standard is core requirement per CLAUDE.md

### Missing: Resume without payload_store edge cases
**Issue**: Test verifies ValueError is raised but doesn't verify error message quality
**Evidence**: `test_resume_requires_payload_store` uses weak regex match
```python
with pytest.raises(ValueError, match=r"payload_store.*required"):
```
**Fix**: Assert exact error message to verify user gets actionable guidance. Error messages are part of the API.
**Priority**: P3

### Missing: Resume with modified graph detection
**Issue**: No test verifies resume fails safely when graph changed between runs
**Evidence**: No test modifies graph between checkpoint and resume
**Scenario**: Original run had 3 transforms, resume config has 2 transforms. Should crash with clear error, not silently produce wrong results.
**Fix**: Add test:
```python
def test_resume_detects_graph_topology_mismatch():
    # Create checkpoint with 3-node pipeline
    # Attempt resume with 2-node pipeline
    # Should raise with clear error about topology mismatch
```
**Priority**: P1 - Data corruption prevention

## Misclassified Tests

### Test Suite: TestOrchestratorResumeRowProcessing (line 35)
**Issue**: Tests are integration tests masquerading as unit tests
**Evidence**:
- Creates real database (`LandscapeDB`)
- Creates real filesystem payload store (`FilesystemPayloadStore`)
- Uses real plugins (`CSVSink`, `PassThrough`, `NullSource`)
- Reads/writes actual files
- Tests end-to-end behavior across multiple subsystems

**Current location**: `tests/engine/test_orchestrator_resume.py`
**Should be**: `tests/integration/test_orchestrator_resume.py`

**Rationale**: These tests exercise the full resume pipeline including database, filesystem, plugins, and orchestrator. Unit tests for orchestrator would mock dependencies and test decision logic in isolation. Integration tests verify subsystem composition works correctly.

Per the test pyramid principle, these are integration tests (slower, broader scope, test multiple components together). They belong with other integration tests like `test_resume_comprehensive.py`.

**Fix**: Move entire file to `tests/integration/` and rename to avoid confusion with existing `test_resume_comprehensive.py`. Suggest `test_orchestrator_resume_integration.py` or merge relevant scenarios into `test_resume_comprehensive.py`.
**Priority**: P2 - Organizational clarity

## Infrastructure Gaps

### Gap: Excessive fixture duplication with test_resume_comprehensive.py
**Issue**: Same fixture patterns duplicated across multiple test files
**Evidence**: Compare `failed_run_with_payloads` in this file with `_setup_failed_run` in test_resume_comprehensive.py - nearly identical database setup logic.
**Impact**: Changes to resume setup requirements must be duplicated across files, increasing maintenance burden and risk of divergence.
**Fix**: Extract shared fixtures to `tests/conftest.py` or create `tests/fixtures/resume_fixtures.py`:
```python
# tests/fixtures/resume_fixtures.py
@pytest.fixture
def failed_run_factory():
    def _make_failed_run(db, payload_store, num_rows, checkpoint_at):
        # Shared setup logic
        ...
    return _make_failed_run
```
**Priority**: P2

### Gap: Hardcoded node IDs prevent reusability
**Issue**: Tests use hardcoded node IDs ("source-node", "transform-node", "sink-node") that are brittle
**Evidence**: Fixture has hardcoded node ID setup, test helper methods also hardcode same IDs, creating coupling
**Impact**: If node ID generation logic changes, all tests break. Tests are coupled to implementation details.
**Fix**: Use helper to generate consistent node IDs based on plugin names/types, or parameterize node IDs in fixtures.
**Priority**: P3

### Gap: No fixture for ExecutionGraph creation
**Issue**: `_create_test_graph()` method duplicates graph creation logic
**Evidence**: Lines 262-310 build ExecutionGraph manually in each test class. Same logic exists in other resume tests.
**Impact**: Graph topology changes require updating multiple test files.
**Fix**: Create reusable fixture:
```python
@pytest.fixture
def simple_pipeline_graph():
    graph = ExecutionGraph()
    # Standard source -> transform -> sink
    graph.add_node("source-node", ...)
    graph.add_node("transform-node", ...)
    graph.add_node("sink-node", ...)
    # ... edges ...
    return graph
```
**Priority**: P2

### Gap: CSV parsing uses weak string operations instead of proper CSV parsing
**Issue**: `test_resume_writes_to_sink` reads CSV with `.read_text()` and `.split("\n")` instead of using csv module
**Evidence**: Line 385-388
```python
content = output_path.read_text()
lines = content.strip().split("\n")
assert len(lines) == 3
```
**Impact**: Fragile to CSV formatting changes (CRLF vs LF, quoting, etc.)
**Fix**: Use csv.DictReader or pandas to verify CSV structure:
```python
import csv
with open(output_path) as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 2
assert rows[0]["id"] == "3"
assert rows[1]["id"] == "4"
```
**Priority**: P1

## Positive Observations

**Good: Fixture isolation** - Each test gets fresh database and payload store via tmp_path, preventing test pollution.

**Good: Real plugin usage** - Tests use actual plugins (CSVSink, PassThrough, NullSource) rather than mocks, catching integration issues.

**Good: Docstrings** - Each test has clear Given/When/Then documentation explaining purpose.

**Good: Error case coverage** - Test for missing payload_store requirement exists (though weak).

## Recommendations

**Immediate (P0)**:
1. Fix `test_resume_writes_to_sink` to properly parse CSV and verify content structure
2. Fix `test_resume_returns_run_result_with_status` to assert exact expected values
3. Add audit trail completeness verification after resume
4. Add idempotence test to prevent duplicate data corruption

**Short-term (P1)**:
1. Fix weak assertions in `test_resume_processes_unprocessed_rows` to verify WHICH rows processed
2. Add resume failure scenario tests (quarantine, sink failure)
3. Add graph topology mismatch detection test
4. Replace string operations with proper CSV parsing
5. Add aggregation state restoration test

**Long-term (P2)**:
1. Move tests to integration/ directory where they belong
2. Extract shared fixtures to reduce duplication
3. Create reusable graph construction fixtures
4. Fix ExecutionGraph private attribute mutation antipattern
