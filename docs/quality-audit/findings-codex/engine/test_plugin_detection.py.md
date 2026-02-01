# Test Defect Report

## Summary

- Tests labeled as processor plugin detection only assert Python `isinstance`/`hasattr` results and never execute `RowProcessor`, so a regression in detection logic would go unnoticed.

## Severity

- Severity: major
- Priority: P1

## Category

- Weak Assertions

## Evidence

- The file claims to test processor detection but only asserts type relationships and duck-typing, with no engine invocation (`tests/engine/test_plugin_detection.py:2`, `tests/engine/test_plugin_detection.py:25`, `tests/engine/test_plugin_detection.py:61`, `tests/engine/test_plugin_detection.py:81`).
- The real detection and failure behavior is in `RowProcessor._process_single_token`, which raises a `TypeError` for unknown types, but this path is not exercised here (`src/elspeth/engine/processor.py:656`, `src/elspeth/engine/processor.py:728`, `src/elspeth/engine/processor.py:862`).

```python
# tests/engine/test_plugin_detection.py
duck = DuckTypedGate()
assert hasattr(duck, "evaluate")
assert not isinstance(duck, BaseGate)
```

```python
# src/elspeth/engine/processor.py
if isinstance(transform, BaseGate):
    ...
elif isinstance(transform, BaseTransform):
    ...
else:
    raise TypeError(...)
```

## Impact

- A regression to duck-typed detection (e.g., `hasattr` on `evaluate`) would still pass these tests, allowing non-plugin objects to be treated as gates/transforms.
- The suite can report "green" without exercising the actual processor contract, creating false confidence for a core safety boundary.
- Errors in the processor’s unknown-type handling could slip through to runtime, leading to misrouting or audit trail inconsistencies.

## Root Cause Hypothesis

- The tests were added as a quick guardrail after refactoring to `isinstance` checks, but they stopped at Python semantics instead of exercising `RowProcessor`.
- Engine setup cost likely led to a shortcut that bypassed actual detection logic.

## Recommended Fix

- Replace or augment the `DuckTyped*` tests to call `RowProcessor.process_row` with duck-typed gate/transform objects and assert a `TypeError` is raised, using the minimal in-memory setup from existing processor tests.
- Ensure the assertion checks the error message includes "BaseTransform" and "BaseGate" to lock in the expected contract.

```python
# Example pattern to add in tests/engine/test_plugin_detection.py
with pytest.raises(TypeError):
    processor.process_row(
        row_index=0,
        row_data={"value": 1},
        transforms=[DuckTypedGate()],
        ctx=PluginContext(run_id=run.run_id, config={}),
    )
```
