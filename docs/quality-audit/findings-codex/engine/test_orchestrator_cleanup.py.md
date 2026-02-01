# Test Defect Report

## Summary

- `_build_test_graph` detects gates with `hasattr(..., "evaluate")`, a defensive pattern that diverges from production’s `isinstance(BaseGate)` checks and can misclassify plugins.

## Severity

- Severity: minor
- Priority: P2

## Category

- [Bug-Hiding Defensive Patterns]

## Evidence

- `tests/engine/test_orchestrator_cleanup.py:37`, `tests/engine/test_orchestrator_cleanup.py:56`, `tests/engine/test_orchestrator_cleanup.py:68`:
```python
is_gate = hasattr(t, "evaluate")
...
if hasattr(t, "evaluate"):
```
- `src/elspeth/engine/processor.py:657`:
```python
if isinstance(transform, BaseGate):
```

## Impact

- Tests can pass even if a transform accidentally defines an `evaluate` attribute or a gate is mis-typed, masking plugin protocol bugs and producing graphs that don’t match runtime detection.
- Cleanup behavior for gates vs transforms is validated using a different classification rule than production, so regressions in gate handling could slip.

## Root Cause Hypothesis

- The helper was implemented for convenience without importing `BaseGate`, using attribute presence to avoid strict type checks despite the repo’s anti-defensive rules.

## Recommended Fix

- Replace `hasattr(t, "evaluate")` with type-safe gate detection, e.g.:
```python
from elspeth.plugins.base import BaseGate

is_gate = isinstance(t, BaseGate)
```
- Apply the same change in all gate checks inside `_build_test_graph` to align with production semantics.
- Priority justification: removes a banned defensive pattern and keeps test graphs aligned with runtime behavior.
---
# Test Defect Report

## Summary

- `_build_test_graph` mutates private `ExecutionGraph` internals (`_sink_id_map`, `_transform_id_map`, `_route_resolution_map`, `_output_sink`) instead of using `ExecutionGraph.from_plugin_instances`, bypassing normal graph construction and validation.

## Severity

- Severity: minor
- Priority: P2

## Category

- [Infrastructure Gaps]

## Evidence

- `tests/engine/test_orchestrator_cleanup.py:62`, `tests/engine/test_orchestrator_cleanup.py:63`, `tests/engine/test_orchestrator_cleanup.py:72`, `tests/engine/test_orchestrator_cleanup.py:76`:
```python
graph._sink_id_map = sink_ids
graph._transform_id_map = transform_ids
graph._route_resolution_map = route_resolution_map
graph._output_sink = "default"
```
- `src/elspeth/engine/orchestrator.py:459`:
```python
raise ValueError("ExecutionGraph is required. Build with ExecutionGraph.from_plugin_instances()")
```
- `src/elspeth/core/dag.py:297`:
```python
def from_plugin_instances(...):
    """Build ExecutionGraph from plugin instances.

    CORRECT method for graph construction - enables schema validation.
```

## Impact

- Tests can pass with graphs that would be invalid or inconsistent in production, reducing confidence in cleanup behavior under real graph construction.
- Bugs in ID mapping, route resolution, or schema validation could slip because the test bypasses the sanctioned builder.

## Root Cause Hypothesis

- The inline test fixtures lack the `config` attribute expected by `ExecutionGraph.from_plugin_instances`, leading to a custom builder that writes private fields directly.

## Recommended Fix

- Use `ExecutionGraph.from_plugin_instances` in these tests by updating fixtures to provide `config` (e.g., subclass `_TestSourceBase`/`_TestSinkBase` from `tests/conftest.py` or add `self.config = {}`), then build the graph via:
```python
graph = ExecutionGraph.from_plugin_instances(
    source=as_source(source),
    transforms=[transform_1, transform_2],
    sinks={"default": as_sink(sink)},
    aggregations={},
    gates=[],
    output_sink="default",
)
```
- Remove all direct assignments to `graph._*` fields.
- Priority justification: aligns test graphs with production construction, preventing false positives from invalid graph state.
