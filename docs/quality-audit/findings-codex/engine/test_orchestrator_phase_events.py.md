# Test Defect Report

## Summary

- Phase failure tests assert only emitted events and exception strings, but never verify Landscape audit records (run status, node_states/error), leaving auditability untested on failure paths.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/engine/test_orchestrator_phase_events.py:84` creates `LandscapeDB.in_memory()` but the test only inspects `phase_errors` and never queries audit tables.
- `tests/engine/test_orchestrator_phase_events.py:161` asserts only PhaseError count/phase/message for PROCESS failure; no Landscape verification.
- `tests/engine/test_orchestrator_phase_events.py:226` repeats the same pattern for SOURCE failure; no Landscape verification.
- `src/elspeth/engine/executors.py:161` shows transform exceptions are recorded to `node_states` (failed with error), which the test should assert but does not.
- `src/elspeth/engine/orchestrator.py:585` records failed run status in Landscape on exceptions, but the test never checks it.

```python
# tests/engine/test_orchestrator_phase_events.py
db = LandscapeDB.in_memory()
...
with pytest.raises(RuntimeError, match="Transform exploded"):
    orchestrator.run(config=config, graph=_build_test_graph(config))
assert len(phase_errors) == 1
assert phase_errors[0].phase == PipelinePhase.PROCESS
```

## Impact

- A regression that stops recording failed node_states or run failure status would still pass these tests, yielding false confidence.
- Audit trail completeness on failure paths (a core ELSPETH requirement) can silently break without detection.
- Compliance and incident analysis would be compromised because failures would not be traceable in Landscape.

## Root Cause Hypothesis

- Tests were scoped narrowly to event emission and did not include audit trail validation; no shared helper in this file to make audit assertions easy.

## Recommended Fix

- After the failure, instantiate `LandscapeRecorder` and assert audit trail state:
  - Verify the run status is `FAILED`.
  - For PROCESS failure, fetch rows/tokens and assert `node_states` contains a failed state with `error_json` containing the exception message.
  - For SOURCE failure, assert the run is `FAILED` and zero rows/tokens were created.
- Example pattern:

```python
from elspeth.core.landscape import LandscapeRecorder
from elspeth.contracts import RunStatus

recorder = LandscapeRecorder(db)
run = recorder.list_runs()[0]
assert run.status == RunStatus.FAILED

rows = recorder.get_rows(run.run_id)
tokens = recorder.get_tokens(rows[0].row_id)
states = recorder.get_node_states_for_token(tokens[0].token_id)
assert states[-1].status == "failed"
assert "Transform exploded" in (states[-1].error_json or "")
```
---
# Test Defect Report

## Summary

- `_build_test_graph` mutates private `ExecutionGraph` fields, bypassing the public graph builder and schema validation, coupling tests to internal implementation details.

## Severity

- Severity: minor
- Priority: P2

## Category

- Infrastructure Gaps

## Evidence

- `tests/engine/test_orchestrator_phase_events.py:69` directly sets private fields (`_sink_id_map`, `_transform_id_map`, `_route_resolution_map`, `_output_sink`).
- `src/elspeth/core/dag.py:298` documents `ExecutionGraph.from_plugin_instances(...)` as the correct public constructor that performs validation.

```python
# tests/engine/test_orchestrator_phase_events.py
graph._sink_id_map = sink_ids
graph._transform_id_map = transform_ids
graph._config_gate_id_map = {}
graph._route_resolution_map = route_resolution_map
graph._output_sink = output_sink
```

## Impact

- Tests bypass schema validation and graph construction logic, so regressions in `ExecutionGraph.from_plugin_instances` or mapping logic can slip through.
- Any refactor of internal fields will break tests even if public API stays stable.
- Reduces confidence in graph-building behavior used in production runs.

## Root Cause Hypothesis

- Helper copied from other tests for convenience; no shared fixture or utility using the public builder was available.

## Recommended Fix

- Replace `_build_test_graph` with the public constructor to avoid private field mutation and to exercise validation:
  - Use `ExecutionGraph.from_plugin_instances(...)` with empty `aggregations`/`gates` for this test.
  - Preserve the output sink selection logic via `output_sink`.

```python
graph = ExecutionGraph.from_plugin_instances(
    source=config.source,
    transforms=config.transforms,
    sinks=config.sinks,
    aggregations={},
    gates=[],
    output_sink="default" if "default" in config.sinks else next(iter(config.sinks), ""),
)
```
