# Test Defect Report

## Summary

- Using using-quality-engineering (flaky-test-prevention) to audit test quality; audit trail test only checks run status and nodes, missing node_states/token_outcomes/artifacts/hash/lineage verification.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/engine/test_orchestrator.py:451` only asserts run status and node registration:
```python
recorder = LandscapeRecorder(db)
run = recorder.get_run(run_result.run_id)

assert run is not None
assert run.status == RunStatus.COMPLETED

# Verify nodes were registered
nodes = recorder.get_nodes(run_result.run_id)
assert len(nodes) == 3  # source, transform, sink

node_names = [n.plugin_name for n in nodes]
assert "test_source" in node_names
assert "identity" in node_names
assert "test_sink" in node_names
```
- `tests/engine/test_orchestrator.py:460` ends the audit trail assertions at node registration; there are no assertions on node_states, token_outcomes, artifacts, hashes, or lineage in this test.

## Impact

- Audit trail regressions (missing node_states entries, incorrect token_outcomes terminal_state, missing artifacts/content_hash, broken lineage) can ship without test failures, undermining auditability guarantees.

## Root Cause Hypothesis

- The audit trail test was implemented as a minimal smoke check and never expanded to validate the required Landscape tables and hash/lineage integrity.

## Recommended Fix

- Extend the audit trail test in `tests/engine/test_orchestrator.py` to query recorder methods like `get_node_states_for_token`, `get_token_outcome`, `get_artifacts`, and `get_token_parents`, assert state/input_hash/output_hash/error fields, terminal_state and sink_name, content_hash/payload_id/artifact_type, and parent_token_id/row_id lineage with deterministic hash comparisons.
---
# Test Defect Report

## Summary

- Progress callback tests assert exact event counts even though the orchestrator emits based on elapsed time; this can be flaky in slow CI.

## Severity

- Severity: major
- Priority: P1

## Category

- Sleepy Assertions

## Evidence

- `tests/engine/test_orchestrator.py:4877` hard-codes the expected event count:
```python
# Should be called at 1 (first row), 100, 200, and 250 (final)
assert len(progress_events) == 4
```
- `tests/engine/test_orchestrator.py:5009` and `tests/engine/test_orchestrator.py:5095` also assume fixed event counts:
```python
# Progress should fire at row 1 (first), 100, and final 150
assert len(progress_events) == 3  # At 1 (first row), 100, and final 150
```
- `src/elspeth/engine/orchestrator.py:939` emits progress on a time interval as well as row count:
```python
time_since_last_progress = current_time - last_progress_time
should_emit = (
    rows_processed == 1
    or rows_processed % progress_interval == 0
    or time_since_last_progress >= progress_time_interval  # Every 5 seconds
)
```

## Impact

- Tests can fail intermittently when processing exceeds the time threshold, causing extra progress events and false negatives.

## Root Cause Hypothesis

- Tests assume row-count-only emission and do not control or mock the clock.

## Recommended Fix

- Mock `time.perf_counter` to deterministic values in these tests, or relax assertions to verify required row checkpoints while allowing extra time-triggered events.
---
# Test Defect Report

## Summary

- Checkpoint frequency and disabled tests do not assert checkpoint creation or absence; they only assert run completion.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/engine/test_orchestrator.py:2291` documents the expected behavior but never checks it:
```python
assert result.status == "completed"
assert result.rows_processed == 3

# Checkpoints should have been created during processing
# After completion, they should be deleted
# So we can't check the checkpoint count here - it's cleaned up
# Instead, we verify the run completed successfully with checkpointing enabled
```
- `tests/engine/test_orchestrator.py:2382` and `tests/engine/test_orchestrator.py:2734` likewise only assert status and stop:
```python
assert result.status == "completed"
```

## Impact

- Regressions in checkpoint frequency (every_row/every_n) or disabled checkpointing can slip through without failing tests.

## Root Cause Hypothesis

- Tests were left as placeholders due to checkpoint cleanup on successful completion.

## Recommended Fix

- Spy on `CheckpointManager.create_checkpoint` to assert call counts, or force a controlled failure run that preserves checkpoints and assert on `checkpoint_mgr.get_checkpoints`, including zero calls when checkpointing is disabled.
---
# Test Defect Report

## Summary

- Node ID assignment test uses `hasattr` and MagicMock equality, so it can pass even if node_id is never set.

## Severity

- Severity: minor
- Priority: P2

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- `tests/engine/test_orchestrator.py:923` uses MagicMock and then checks `hasattr`/equality:
```python
mock_source = MagicMock()
...
orchestrator.run(pipeline_config, graph=graph)

# Source should have node_id set from graph
assert hasattr(mock_source, "node_id")
assert mock_source.node_id == graph.get_source()
...
assert hasattr(mock_sink, "node_id")
assert mock_sink.node_id == sink_id_map["output"]
```

## Impact

- The test can return a false positive if node_id assignment regresses, weakening coverage for a critical plugin contract.

## Root Cause Hypothesis

- MagicMock was used for convenience without freezing attributes, and defensive `hasattr` checks were added to avoid attribute errors.

## Recommended Fix

- Replace MagicMock with a small fake class that defines `node_id = None`, or explicitly set `mock_source.node_id = None` and `mock_sink.node_id = None` before the run, remove `hasattr`, and assert direct equality on the resulting string values.
