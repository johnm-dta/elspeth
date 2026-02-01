# Test Defect Report

## Summary

- The test that claims to verify `__all__` exports only checks a hard-coded subset, so many public exports can break without failing tests.

## Severity

- Severity: minor
- Priority: P2

## Category

- Incomplete Contract Coverage

## Evidence

- `tests/core/landscape/test_exports.py:49` defines `test_can_import_all_exports` with a manual import list and only checks non-None, which does not track `__all__` as it changes.
```python
def test_can_import_all_exports(self) -> None:
    """Verify __all__ exports are importable."""
    from elspeth.core.landscape import (
        Artifact,
        Batch,
        BatchMember,
        BatchOutput,
        Call,
        Edge,
        LandscapeDB,
        LandscapeExporter,
        LandscapeRecorder,
        Node,
        NodeState,
        RoutingEvent,
        Row,
        Run,
        Token,
        TokenParent,
        # Tables
        artifacts_table,
        batch_members_table,
        batch_outputs_table,
        batches_table,
        calls_table,
        edges_table,
        metadata,
        node_states_table,
        nodes_table,
        routing_events_table,
        rows_table,
        runs_table,
        token_parents_table,
        tokens_table,
    )
```
- `src/elspeth/core/landscape/__init__.py:78` shows `__all__` includes additional exports not covered by the test, such as `CSVFormatter`, `JSONFormatter`, `CallStatus`, `CallType`, `Checkpoint`, `LineageResult`, `RowDataResult`, `RowDataState`, `RunStatus`, `SchemaCompatibilityError`, `compute_grade`, `set_run_grade`, `update_grade_after_purge`, and `explain`.
```python
__all__ = [
    "Artifact",
    "Batch",
    "BatchMember",
    "BatchOutput",
    "CSVFormatter",
    "Call",
    "CallStatus",
    "CallType",
    "Checkpoint",
    ...
    "SchemaCompatibilityError",
    ...
    "compute_grade",
    "explain",
    ...
    "set_run_grade",
    ...
    "update_grade_after_purge",
]
```

## Impact

- Public API regressions in the omitted exports will not be caught by tests.
- The test provides false confidence that `__all__` is fully importable while only validating a subset.

## Root Cause Hypothesis

- The test was written before new exports were added and was not updated as `__all__` grew, resulting in drift between the test list and the actual export contract.

## Recommended Fix

- Derive the test directly from `elspeth.core.landscape.__all__` and assert every entry resolves via `getattr`, or assert the hard-coded list matches `__all__` to prevent drift.
```python
import elspeth.core.landscape as landscape

def test_all_exports_importable() -> None:
    for name in landscape.__all__:
        assert getattr(landscape, name) is not None
```
- Priority justification: P2 because it safeguards the public API contract; without it, missing exports can ship undetected.
