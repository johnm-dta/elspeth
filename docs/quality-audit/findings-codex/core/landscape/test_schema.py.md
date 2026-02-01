# Test Defect Report

## Summary

- Checkpoint schema tests omit required topology validation columns (`upstream_topology_hash`, `checkpoint_node_config_hash`), so regressions can drop them without failing tests.

## Severity

- Severity: major
- Priority: P1

## Category

- [Incomplete Contract Coverage]

## Evidence

- `tests/core/landscape/test_schema.py:200-217` only checks a subset of checkpoint columns and never asserts the topology hash fields:
```python
assert "checkpoint_id" in columns
assert "run_id" in columns
assert "token_id" in columns
assert "node_id" in columns
assert "created_at" in columns
...
assert "sequence_number" in columns
assert "aggregation_state_json" in columns
```
- `src/elspeth/core/landscape/schema.py:346-358` defines the required topology validation columns that are not asserted in the test:
```python
Column("upstream_topology_hash", String(64), nullable=False),
Column("checkpoint_node_config_hash", String(64), nullable=False),
```

## Impact

- A schema regression that removes/renames these columns will pass the tests, breaking checkpoint compatibility validation and undermining recovery integrity without detection.

## Root Cause Hypothesis

- The checkpoint tests were written before topology validation fields were added and never updated to cover the new required columns.

## Recommended Fix

- Extend the checkpoint schema tests to assert `upstream_topology_hash` and `checkpoint_node_config_hash` exist (and are non-null) in `checkpoints_table`.
- Example:
```python
columns = {c.name: c for c in checkpoints_table.columns}
assert "upstream_topology_hash" in columns
assert columns["upstream_topology_hash"].nullable is False
assert "checkpoint_node_config_hash" in columns
assert columns["checkpoint_node_config_hash"].nullable is False
```
---
# Test Defect Report

## Summary

- Model tests instantiate `RoutingEvent` and `Batch` with `created_at=None` (with `type: ignore`) and only assert a single field, allowing invalid audit timestamps to pass unnoticed.

## Severity

- Severity: minor
- Priority: P2

## Category

- [Weak Assertions]

## Evidence

- `tests/core/landscape/test_schema.py:123-149` uses `created_at=None` for required datetime fields:
```python
event = RoutingEvent(
    ...,
    created_at=None,  # type: ignore[arg-type]
)
...
batch = Batch(
    ...,
    created_at=None,  # type: ignore[arg-type]
)
```
- `src/elspeth/core/landscape/models.py:277-301` defines `created_at` as a required `datetime` for both models:
```python
class RoutingEvent:
    ...
    created_at: datetime

class Batch:
    ...
    created_at: datetime
```

## Impact

- Tests normalize invalid timestamps and would still pass if production code accidentally allowed `created_at` to be missing, which would compromise audit ordering and traceability.

## Root Cause Hypothesis

- The tests use `None` as a shortcut to avoid constructing timestamps and were not aligned with the non-null audit requirements.

## Recommended Fix

- Use real `datetime` values and assert `created_at` is preserved; remove `type: ignore`.
- Example:
```python
event = RoutingEvent(..., created_at=datetime.now(UTC))
assert event.created_at is not None
```
