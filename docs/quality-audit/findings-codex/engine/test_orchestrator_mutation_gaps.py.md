# Test Defect Report

## Summary

- Node type metadata tests are tautological; they never exercise orchestrator node registration or inspect recorded nodes, so determinism/plugin_version mutations in `Orchestrator` are untested.

## Severity

- Severity: major
- Priority: P1

## Category

- [Weak Assertions]

## Evidence

- `tests/engine/test_orchestrator_mutation_gaps.py:442` only asserts enum constants and never calls orchestrator or inspects audit records:
```python
def test_config_gate_has_deterministic_flag(self) -> None:
    assert Determinism.DETERMINISTIC.value == "deterministic"

def test_coalesce_is_deterministic(self) -> None:
    determinism = Determinism.DETERMINISTIC
    assert determinism == Determinism.DETERMINISTIC
```
- `src/elspeth/engine/orchestrator.py:670` contains the actual metadata assignment and `register_node` call that the tests never hit:
```python
if node_id in config_gate_node_ids:
    plugin_version = "1.0.0"
    determinism = Determinism.DETERMINISTIC
...
recorder.register_node(
    run_id=run_id,
    node_id=node_id,
    plugin_version=plugin_version,
    determinism=determinism,
)
```

## Impact

- Changes to determinism or plugin_version for config gates, aggregations, or coalesce nodes can slip through undetected.
- Audit trail integrity is at risk because incorrect metadata could be recorded without test coverage.
- Mutation survivors in this area will persist, giving false confidence.

## Root Cause Hypothesis

- Tests were added to “kill mutants” quickly by asserting enum constants rather than running the orchestrator and validating recorded metadata.

## Recommended Fix

- Build a minimal pipeline/graph that includes a config gate, aggregation node, and coalesce node; run `Orchestrator.run` (or `_execute_run`) with `LandscapeDB.in_memory()` and then query `LandscapeRecorder.get_nodes(run_id)` to assert `determinism` and `plugin_version` per node.
- Add assertions on recorded node rows (not enums) so mutations in registration logic are caught.
- Priority justification: these fields are part of the audit trail and are core to ELSPETH’s determinism guarantees.
---
# Test Defect Report

## Summary

- `test_rows_routed_defaults_to_zero` does not test a default value; it passes `rows_routed=0` explicitly even though `rows_routed` has no default, so the test is vacuous.

## Severity

- Severity: minor
- Priority: P3

## Category

- [Weak Assertions]

## Evidence

- `tests/engine/test_orchestrator_mutation_gaps.py:45` explicitly supplies `rows_routed`, so no default is exercised:
```python
result = RunResult(
    run_id="test-run",
    status="completed",
    rows_processed=10,
    rows_succeeded=10,
    rows_failed=0,
    rows_routed=0,
)
```
- `src/elspeth/engine/orchestrator.py:87` shows `rows_routed` is required (no default):
```python
rows_failed: int
rows_routed: int
rows_quarantined: int = 0
```

## Impact

- A mutation that adds or changes a default for `rows_routed` would not be detected.
- The test provides false confidence about default behavior that does not exist.

## Root Cause Hypothesis

- Misinterpretation of `RunResult` fields during mutation-gap triage led to a test that checks assignment rather than defaulting semantics.

## Recommended Fix

- Replace the test with an assertion that omitting `rows_routed` raises `TypeError`, or revise it to check a real defaulted field only.
- Optional: use `RunStatus.COMPLETED` to align with the enum contract in these tests.
- Priority justification: minor correctness issue but easy to fix and reduces misleading coverage.
