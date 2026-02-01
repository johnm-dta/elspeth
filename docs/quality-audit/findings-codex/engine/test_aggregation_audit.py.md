# Test Defect Report

## Summary

- Missing audit trail verification for aggregation flush: tests assert only status/results and omit required audit fields (input_hash/output_hash/error_json) and batch linkage/trigger_type in success and failure paths.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/engine/test_aggregation_audit.py:247` only checks status on the node_state; no hash assertions:
  ```python
  assert result.status == "success"
  assert result.row == {"sum": 30, "count": 2}
  ...
  assert agg_state.status == "completed"
  ```
- `tests/engine/test_aggregation_audit.py:332` and `tests/engine/test_aggregation_audit.py:385` verify only failure status, not `error_json` or missing `output_hash`:
  ```python
  assert batch.status == BatchStatus.FAILED.value
  ...
  assert agg_state.status == "failed"
  ```
- `tests/engine/test_aggregation_audit.py:279` only verifies batch status and ignores `aggregation_state_id`/`trigger_type`:
  ```python
  assert batch.status == BatchStatus.DRAFT.value
  ...
  assert batch.status == BatchStatus.COMPLETED.value
  ```
- `src/elspeth/core/landscape/schema.py:169` requires `input_hash` (non-null) and defines `output_hash` and `error_json` fields that are never asserted in these tests.
- `src/elspeth/engine/executors.py:913` computes `input_hash` and records `output_hash`/`error` via `complete_node_state`, so these audit writes are expected but unverified.

## Impact

- Regressions that drop or corrupt audit hashes, error details, or batch linkage would still pass, undermining ELSPETH's auditability guarantees.
- Creates false confidence on the aggregation audit trail, especially for failure scenarios where error recording is legally significant.

## Root Cause Hypothesis

- Tests were written to validate functional outcomes (statuses/results) but not updated to enforce audit field invariants introduced in the recorder schema and executor logic.

## Recommended Fix

- Add explicit assertions in `tests/engine/test_aggregation_audit.py` for node_state `input_hash`, `output_hash`, and `error_json`, plus batch `aggregation_state_id` and `trigger_type` on completed/failed flushes.
- Example success-path assertions:
  ```python
  from elspeth.core.canonical import stable_hash

  batch_input = {"batch_rows": [{"x": 10}, {"x": 20}]}
  expected_input_hash = stable_hash(batch_input)
  expected_output_hash = stable_hash({"sum": 30, "count": 2})

  states = recorder.get_node_states_for_token(token1.token_id)
  agg_state = next(s for s in states if s.node_id == aggregation_node_id)
  assert agg_state.input_hash == expected_input_hash
  assert agg_state.output_hash == expected_output_hash

  batch = recorder.get_batch(batch_id)
  assert batch.aggregation_state_id == agg_state.state_id
  assert batch.trigger_type == TriggerType.COUNT.value
  ```
- Example failure-path assertions:
  ```python
  import json

  states = recorder.get_node_states_for_token(token.token_id)
  agg_state = next(s for s in states if s.node_id == aggregation_node_id)
  assert agg_state.output_hash is None
  assert agg_state.error_json is not None
  error = json.loads(agg_state.error_json)
  assert error["type"] == "RuntimeError"
  assert error["exception"] == "intentional failure"
  ```
- Priority justification: These checks enforce non-negotiable audit invariants; without them, core audit trail integrity regressions can slip through unnoticed.
