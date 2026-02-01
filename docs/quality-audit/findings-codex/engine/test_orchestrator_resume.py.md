# Test Defect Report

## Summary

- Resume tests validate counts/CSV output but never verify audit trail tables (node_states, token_outcomes, artifacts, hashes, lineage) for resumed rows.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/engine/test_orchestrator_resume.py:341` Only checks row counts after resume, no audit DB verification.
```python
assert result.rows_processed == 2
assert result.rows_succeeded == 2
assert result.rows_failed == 0
```
- `tests/engine/test_orchestrator_resume.py:379` Only checks output CSV content, no audit trail assertions.
```python
assert output_path.exists()
content = output_path.read_text()
...
assert "data-3" in content
```
- `src/elspeth/engine/orchestrator.py:1759` Resume records terminal outcomes that are never asserted in tests.
```python
recorder.record_token_outcome(
    run_id=run_id,
    token_id=result.token.token_id,
    outcome=RowOutcome.COMPLETED,
    sink_name=sink_name,
)
```
- `src/elspeth/engine/executors.py:1453` Sink execution creates node_states for each token; tests do not query these.
```python
state = self._recorder.begin_node_state(
    token_id=token.token_id,
    node_id=sink_node_id,
    step_index=step_in_pipeline,
    input_data=token.row_data,
)
```
- `src/elspeth/engine/executors.py:1512` Sink execution registers artifacts; tests do not verify artifact records.
```python
artifact = self._recorder.register_artifact(
    run_id=self._run_id,
    state_id=first_state.state_id,
    sink_node_id=sink_node_id,
    artifact_type=artifact_info.artifact_type,
    path=artifact_info.path_or_uri,
    content_hash=artifact_info.content_hash,
    size_bytes=artifact_info.size_bytes,
)
```

## Impact

- Audit trail regressions in resume (missing node_states, wrong hashes, missing token_outcomes/artifacts, broken lineage) would go undetected.
- Creates false confidence in compliance-critical auditability even if resume path fails to record required evidence.

## Root Cause Hypothesis

- Tests focus on functional outcomes (counts and CSV output) and omit audit verification despite auditability being a core requirement.

## Recommended Fix

- Add post-resume assertions against Landscape tables:
  - Query `node_states_table` for resumed tokens and assert `status="completed"`, `input_hash`/`output_hash` match `stable_hash` of row data and sink output, and `error_json` is null.
  - Query `token_outcomes_table` for resumed tokens and assert terminal `outcome="COMPLETED"`, `is_terminal=1`, `sink_name="default"`.
  - Query `artifacts_table` for run_id and assert `content_hash`, `artifact_type`, `path_or_uri`, and `produced_by_state_id` link to node_states.
  - Validate lineage fields (`parent_token_id`/row_id relationships) where applicable.
- Use SQLAlchemy `select(...)` with `LandscapeDB.connection()` to keep checks deterministic and audit-focused.
---
# Test Defect Report

## Summary

- `test_resume_returns_run_result_with_status` uses vacuous `>= 0` assertions and never checks `status`, so it can pass even with incorrect resume results.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/engine/test_orchestrator_resume.py:445` Assertions only ensure non-negative counts, which are always true.
```python
assert isinstance(result, RunResult)
assert result.run_id == run_id
# Status should be set by completion
assert result.rows_processed >= 0
assert result.rows_succeeded >= 0
assert result.rows_failed >= 0
```

## Impact

- Incorrect status or incorrect row counts during resume would not fail the test, masking regressions in resume logic.

## Root Cause Hypothesis

- Placeholder assertions left in place to avoid brittleness, not updated to validate expected outcomes.

## Recommended Fix

- Assert exact expected values using known fixture data:
  - `assert result.status == RunStatus.COMPLETED`
  - `assert result.rows_processed == len(failed_run_with_payloads["unprocessed_indices"])`
  - `assert result.rows_succeeded == 2`
  - `assert result.rows_failed == 0`
- Import `RunStatus` in the test file and align with resume behavior.
---
# Test Defect Report

## Summary

- Tests mutate private `ExecutionGraph` fields directly, bypassing the public graph construction API and risking brittle, non-representative graph setups.

## Severity

- Severity: minor
- Priority: P3

## Category

- Infrastructure Gaps

## Evidence

- `tests/engine/test_orchestrator_resume.py:299` Direct writes to private graph internals.
```python
graph._sink_id_map = {"default": "sink-node"}
graph._transform_id_map = {0: "transform-node"}
graph._config_gate_id_map = {}
graph._output_sink = "default"
graph._route_resolution_map = {}
```
- `src/elspeth/core/dag.py:298` Public API advertises `from_plugin_instances` as the correct construction path.
```python
"""Build ExecutionGraph from plugin instances.

CORRECT method for graph construction - enables schema validation.
Schemas extracted directly from instance attributes.
"""
```

## Impact

- Tests may pass even if real graph construction changes or invariants break, masking resume failures caused by incorrect graph maps or node IDs.

## Root Cause Hypothesis

- Convenience setup to align hard-coded node IDs with seeded DB state, at the expense of using the supported API.

## Recommended Fix

- Build the graph via `ExecutionGraph.from_plugin_instances` using the same plugin instances as the test config.
- Use the resulting node IDs when seeding `nodes_table`, `edges_table`, and checkpoint data so the DB and graph remain consistent.
- Avoid assigning to `_sink_id_map`/`_transform_id_map` directly; keep graph construction aligned with production behavior.
