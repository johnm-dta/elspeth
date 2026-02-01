# Test Defect Report

## Summary

- Config gate integration tests validate routing counters and sink outputs but never verify Landscape audit records (node_states, token_outcomes, artifacts/hashes), leaving auditability untested for config-driven gates.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/engine/test_config_gates.py:163` runs the pipeline and only asserts run counters/sink length; no audit table checks follow.
```python
result = orchestrator.run(config, graph=_build_test_graph_with_config_gates(config))

assert result.status == "completed"
assert result.rows_processed == 2
assert result.rows_succeeded == 2
assert len(sink.results) == 2
```
- `tests/engine/test_config_gates.py:548` is the only Landscape query in this file and it checks only the `nodes` table, not `node_states`, `token_outcomes`, or `artifacts`.
```python
nodes = conn.execute(
    text("SELECT plugin_name, node_type FROM nodes WHERE run_id = :run_id"),
    {"run_id": result.run_id},
).fetchall()
```
- `tests/engine/test_config_gates.py:544` shows the audit verification is limited to node registration; there are no assertions for input_hash/output_hash, error fields, terminal outcomes, or artifacts.

## Impact

- Audit trail regressions for config gates (missing node_states, incorrect hashes, missing token_outcomes) can ship undetected.
- False confidence: tests pass even if routing events are recorded but terminal outcomes or artifacts are missing or incorrect.
- Violates ELSPETH auditability standard expectations for end-to-end traceability.

## Root Cause Hypothesis

- Tests prioritize routing behavior over audit integrity, and there is no shared audit assertion helper for Landscape records.
- Audit requirements are documented but not encoded into the test patterns used for engine integration tests.

## Recommended Fix

- Add audit assertions to each run-based test to validate `node_states` (status, input_hash, output_hash, error_json), `token_outcomes` (terminal outcome and sink_name), and `artifacts` (artifact_type, content_hash, path_or_uri). Use `LandscapeRecorder` helpers or direct SQL, and compute hashes via `stable_hash` to verify determinism.
```python
from elspeth.core.canonical import stable_hash
from elspeth.core.landscape.recorder import LandscapeRecorder

recorder = LandscapeRecorder(db)
rows = recorder.get_rows(result.run_id)
tokens = recorder.get_tokens(rows[0].row_id)
states = recorder.get_node_states_for_token(tokens[0].token_id)
assert states[-1].status == "completed"
assert states[-1].error is None
assert states[-1].input_hash == stable_hash(tokens[0].row_data)
```
- For routed rows, assert a terminal `token_outcomes` entry with `outcome == "ROUTED"` and correct `sink_name`; for continue paths, assert `COMPLETED` outcomes.
- Priority justification: audit trail correctness is core to ELSPETH’s trust model, so missing verification is a high-risk gap.
