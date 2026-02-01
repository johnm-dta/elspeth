# Test Quality Review: test_orchestrator.py

## Summary
The test file contains 5106 lines testing the Orchestrator subsystem. While comprehensive in scope, it exhibits significant test construction issues including excessive mocking that obscures integration value, repeated test fixture boilerplate creating 700+ lines of duplication, and missing critical audit trail verification that undermines the project's high-stakes accountability standard.

## Poorly Constructed Tests

### Test: test_run_simple_pipeline (line 214)
**Issue**: Inline class definitions create massive duplication
**Evidence**: Lines 230-280 define ListSource, DoubleTransform, and CollectSink classes that appear in 30+ tests with minor variations
**Fix**: Extract to shared fixtures in conftest.py or test-level fixtures. The pattern `class ListSource(_TestSourceBase)` with identical `__init__`, `on_start`, `load`, `close` appears in nearly every test.
**Priority**: P1

### Test: test_run_records_landscape_entries (line 378)
**Issue**: No verification of audit trail content beyond count
**Evidence**: Lines 451-467 only assert node count and names exist, but never verify node metadata, edge relationships, or token lineage. Critical for auditability standard.
**Fix**: Add assertions for node.plugin_version, node.determinism, edge labels/modes, and token parent relationships. Query `recorder.get_node_states(run_id, row_id)` to verify full lineage.
**Priority**: P0

### Test: test_run_marks_failed_on_transform_exception (line 473)
**Issue**: Doesn't verify failed row is recorded in landscape
**Evidence**: Lines 547-556 verify run status=FAILED, but never check if the failed row has a node_state record with terminal state FAILED and error details
**Fix**: Query `recorder.get_node_states(run_id, row_id)` and assert final state is FAILED with captured exception
**Priority**: P1

### Test: test_checkpoint_preserved_on_failure (line 2472)
**Issue**: Non-deterministic test due to dict iteration order
**Evidence**: Lines 2624-2651 test comments acknowledge "sink iteration order in dict may vary" - makes assertions conditional on which sink ran first
**Fix**: Use OrderedDict or explicit sink execution order in config. Test must be deterministic.
**Priority**: P1

### Test: test_maybe_checkpoint_creates_on_every_row (line 2210)
**Issue**: No verification of checkpoint creation
**Evidence**: Lines 2294-2297 comment says "we can't check the checkpoint count here - it's cleaned up" then doesn't verify checkpoints were created
**Fix**: Either pause before cleanup and verify count, or mock checkpoint_manager.create_checkpoint and assert call count
**Priority**: P2

### Test: test_orchestrator_uses_graph_node_ids (line 883)
**Issue**: Excessive mocking obscures integration value
**Evidence**: Lines 924-944 mock source and sink entirely, including `load.return_value = iter([])` - test runs zero rows and verifies nothing about actual orchestration
**Fix**: Use real minimal plugins (NullSource, CollectSink) to test actual node_id assignment during real execution
**Priority**: P2

### Test: test_orchestrator_exports_landscape_when_configured (line 1443)
**Issue**: No verification of export content structure
**Evidence**: Lines 1570-1574 only assert `len(export_sink.captured_rows) > 0` and one record has `record_type="run"`. Doesn't verify required fields, signature structure, or manifest presence.
**Fix**: Assert export contains all record types (run, node, node_state, artifact), verify each has required fields, check signature format when signing enabled
**Priority**: P1

### Test: test_invalid_source_quarantine_destination_fails_at_init (line 3672)
**Issue**: Test data contradicts test purpose
**Evidence**: Lines 3700-3717 define QuarantiningSource that yields quarantined rows in `load()`, but line 3763 expects validation to fail BEFORE load() is called. The `_on_validation_failure` attribute (line 3703) is never consumed by the orchestrator.
**Fix**: Either remove the yielded quarantined row (test should fail in config validation, not runtime), or change to test runtime validation if that's the actual behavior
**Priority**: P1

### Test: test_progress_callback_called_every_100_rows (line 4815)
**Issue**: Magic numbers without explanation
**Evidence**: Lines 4877-4884 assert 4 progress events at rows 1, 100, 200, 250 but the 100-row interval is hardcoded in orchestrator with no configuration
**Fix**: Document why 100-row interval, or make configurable and test at different intervals
**Priority**: P3

### Test: test_orchestrator_handles_list_results_from_processor (line 3568)
**Issue**: Comment says test is blocked but test runs anyway
**Evidence**: Lines 3562-3565 say "Full fork testing at orchestrator level is blocked by ExecutionGraph using DiGraph instead of MultiDiGraph" but test still asserts fork behavior
**Fix**: Either remove obsolete comment if fork works, or move test to test_processor.py as comment suggests
**Priority**: P3

## Misclassified Tests

### Test: test_orchestrator_accepts_checkpoint_manager (line 2182)
**Issue**: Unit test masquerading as integration test
**Evidence**: Lines 2182-2194 only verify constructor accepts parameter and stores it in `_checkpoint_manager` attribute. No orchestration occurs.
**Fix**: Move to separate `TestOrchestratorConstruction` unit test class, or merge with actual checkpoint integration test
**Priority**: P2

### Test: test_orchestrator_run_accepts_graph (line 963)
**Issue**: Signature inspection test, not behavior test
**Evidence**: Lines 963-983 use `inspect.signature()` to verify parameter exists. Tests interface contract, not runtime behavior.
**Fix**: Remove - parameter existence is verified by actual integration tests that pass graph
**Priority**: P3

### Test: Multiple "creates X from settings" tests (lines 2182, 3878, 4105)
**Issue**: Configuration parsing tests in orchestrator suite
**Evidence**: Tests like `test_orchestrator_creates_retry_manager_from_settings` verify settings → object construction, not orchestration
**Fix**: Move to test_config.py or separate test_orchestrator_configuration.py file
**Priority**: P2

## Infrastructure Gaps

### Gap: Repeated plugin fixture boilerplate
**Issue**: 700+ lines of duplicated class definitions
**Evidence**: Every test defines its own `ListSource`, `CollectSink`, `IdentityTransform`, etc. with 95% identical implementations
**Fix**: Create pytest fixtures in conftest.py:
```python
@pytest.fixture
def list_source_factory():
    def _factory(data, schema=ValueSchema):
        class ListSource(_TestSourceBase):
            name = "list_source"
            output_schema = schema
            def __init__(self) -> None:
                self._data = data
            # ... standard methods
        return ListSource()
    return _factory
```
Use as `source = list_source_factory([{"value": 1}])`
**Priority**: P0

### Gap: No audit trail verification utilities
**Issue**: Every test that should verify audit trail either skips it or writes custom queries
**Evidence**: Only 2 tests query landscape (test_run_records_landscape_entries, test_node_metadata_records_plugin_version). Others verify outcomes but not audit trail.
**Fix**: Create helper functions:
```python
def assert_complete_audit_trail(db, run_id, row_id, expected_terminal_state):
    """Verify audit trail from source to terminal state."""
    recorder = LandscapeRecorder(db)
    # Assert source entry exists
    # Assert transform node_states recorded
    # Assert terminal state matches expected
    # Assert token lineage is unbroken
```
**Priority**: P0

### Gap: No graph construction utilities for common patterns
**Issue**: `_build_test_graph()` and `_build_fork_test_graph()` at module level, but tests still manually build graphs
**Evidence**: Lines 1255-1263, 1338-1346, 1422-1429, 2588-2615 all manually construct graphs with repeated boilerplate
**Fix**: Create fixture:
```python
@pytest.fixture
def graph_builder():
    """Returns callable that builds graph from config."""
    return lambda config, **overrides: _build_test_graph(config, **overrides)
```
**Priority**: P1

### Gap: Missing terminal state verification
**Issue**: Tests verify counts (rows_succeeded, rows_failed) but not that every row reached exactly one terminal state
**Evidence**: No test queries `recorder.get_node_states(run_id, row_id)` and asserts terminal state enum
**Fix**: Add test:
```python
def test_every_row_reaches_terminal_state():
    # Run pipeline with 100 rows (mix of success, quarantine, failure, routing)
    # For each row_id, query node_states
    # Assert final state in {COMPLETED, ROUTED, QUARANTINED, FAILED, FORKED, CONSUMED_IN_BATCH, COALESCED}
    # Assert no rows have "in-flight" states at end
```
**Priority**: P0

### Gap: No tests for token identity tracking through DAG
**Issue**: DAG execution model requires `row_id` vs `token_id` vs `parent_token_id` but no tests verify lineage
**Evidence**: No test asserts token.parent_token_id or token.branch_name for forked rows
**Fix**: Add tests for fork scenarios verifying:
- Parent token has terminal state FORKED
- Child tokens have parent_token_id pointing to parent
- Child tokens have distinct token_ids
- row_id is preserved across fork
**Priority**: P0

### Gap: No checkpoint version compatibility tests
**Issue**: Checkpointing is critical for recovery but no tests verify checkpoint schema migrations
**Evidence**: Tests verify checkpoint creation/deletion but never test loading checkpoint from different version
**Fix**: Add test that creates checkpoint with v1 schema, simulates upgrade, resumes from checkpoint. Verify compatibility validator catches breaking changes.
**Priority**: P1

### Gap: Mock-heavy tests provide false confidence
**Issue**: 15+ tests use `MagicMock()` for source/sink/transform, removing all real plugin behavior
**Evidence**: Lines 1129-1178 (test_gate_routes_to_named_sink) mock everything except routing logic
**Fix**: Replace with real minimal plugins (NullSource yields 1 row, CollectSink stores results). Mocks are acceptable for external systems (Azure API) but not internal plugins.
**Priority**: P1

## Positive Observations

- Comprehensive feature coverage: checkpointing, retries, routing, gates, landscape export, progress callbacks
- Good test organization with descriptive class names (TestOrchestratorAuditTrail, TestOrchestratorErrorHandling)
- Lifecycle hook testing is thorough (on_start, on_complete, on_error)
- Error message validation tests exist (test_error_message_includes_route_label)
- Route validation testing follows "fail fast" principle (validation before processing)
- Progress callback edge cases tested (quarantined rows, routed rows)
