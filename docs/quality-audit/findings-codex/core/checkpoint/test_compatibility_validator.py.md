# Test Defect Report

## Summary

- Missing tests for core compatibility checks (checkpoint node missing and checkpoint node config change) in the compatibility validator suite

## Severity

- Severity: major
- Priority: P1

## Category

- Incomplete Contract Coverage

## Evidence

- `src/elspeth/core/checkpoint/compatibility.py:54` and `src/elspeth/core/checkpoint/compatibility.py:66` implement explicit rejection paths for missing checkpoint node and config mismatch:
  ```python
  if not current_graph.has_node(checkpoint.node_id):
      return ResumeCheck(...)
  ...
  if checkpoint.checkpoint_node_config_hash != current_config_hash:
      return ResumeCheck(...)
  ```
- `tests/core/checkpoint/test_compatibility_validator.py:41` and `tests/core/checkpoint/test_compatibility_validator.py:61` show tests always keep the checkpoint node present with unchanged config/hash, so those branches are never exercised:
  ```python
  original_config_hash = stable_hash({"prompt": "test"})
  ...
  modified_graph.add_node("checkpoint_node", node_type="transform", plugin_name="llm", config={"prompt": "test"})
  ```

## Impact

- Regressions in node-existence or config-hash validation could slip through (e.g., allowing resume after checkpoint node removal or config change), risking inconsistent processing and audit integrity.
- The test suite gives false confidence that all compatibility checks are enforced when two explicit validation branches are untested.

## Root Cause Hypothesis

- Tests were added to cover topology-change gaps and did not include coverage for the validator’s basic node-existence and config-mismatch checks.

## Recommended Fix

- Add two focused tests in `tests/core/checkpoint/test_compatibility_validator.py`:
  1) `test_resume_rejects_missing_checkpoint_node`: build a checkpoint from an original graph, validate against a modified graph that omits `checkpoint_node`; assert `not result.can_resume` and reason mentions missing node.
  2) `test_resume_rejects_checkpoint_config_change`: keep topology identical but change `checkpoint_node` config (e.g., `{"prompt": "changed"}`) so `checkpoint_node_config_hash` mismatches; assert `not result.can_resume` and reason mentions configuration change.
- Example pattern:
  ```python
  modified_graph.add_node("checkpoint_node", node_type="transform", plugin_name="llm", config={"prompt": "changed"})
  result = validator.validate(checkpoint, modified_graph)
  assert not result.can_resume
  assert "configuration has changed" in result.reason.lower()
  ```
- Priority P1 because these checks are core to resume safety and auditability.
