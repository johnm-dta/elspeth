# Test Defect Report

## Summary

- Coalesce audit trail test only checks node state count/status and skips required audit tables and hash/lineage validation, leaving audit integrity unverified.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/engine/test_coalesce_integration.py:598` and `tests/engine/test_coalesce_integration.py:609` show the test only queries `node_states_table` and asserts status/count.
```python
from elspeth.core.landscape.schema import node_states_table, nodes_table
...
states_result = conn.execute(
    node_states_table.select().where(node_states_table.c.node_id == coalesce_node.node_id)
).fetchall()
assert len(states_result) == 2
for state in states_result:
    assert state.status == "completed"
```
- `src/elspeth/core/landscape/schema.py:178`, `src/elspeth/core/landscape/schema.py:179`, and `src/elspeth/core/landscape/schema.py:183` define `input_hash`, `output_hash`, and `error_json` fields that are not asserted.
```python
Column("input_hash", String(64), nullable=False),
Column("output_hash", String(64)),
Column("error_json", Text),
```
- `src/elspeth/core/landscape/schema.py:120`, `src/elspeth/core/landscape/schema.py:153`, and `src/elspeth/core/landscape/schema.py:212` define `token_outcomes_table`, `token_parents_table`, and `artifacts_table`, but `tests/engine/test_coalesce_integration.py` never queries them for the coalesce run.

## Impact

- Audit regressions (missing/incorrect token outcomes, lineage, or artifact hashes) can slip through while tests still pass.
- Creates false confidence that coalesce operations are fully auditable.

## Root Cause Hypothesis

- Integration test focuses on functional completion rather than audit trail invariants mandated by the project standard.

## Recommended Fix

- Extend `tests/engine/test_coalesce_integration.py:543` to assert audit trail completeness:
  - Query `token_outcomes_table` for expected outcomes (FORKED for parent token, COALESCED for merged token, and sink terminal outcome if recorded) and verify `is_terminal`, `join_group_id`, `sink_name`.
  - Query `token_parents_table` to ensure the merged token has two parents with correct ordinals.
  - Assert `node_states_table.input_hash` and `output_hash` are non-null and `error_json` is null for coalesce node states.
  - Query `artifacts_table` to verify `content_hash`, `artifact_type`, and `sink_node_id` match the sink output.
```python
from elspeth.core.landscape.schema import (
    token_outcomes_table,
    token_parents_table,
    artifacts_table,
    node_states_table,
)

outcomes = conn.execute(token_outcomes_table.select()).fetchall()
assert any(o.outcome == "coalesced" for o in outcomes)
parents = conn.execute(token_parents_table.select()).fetchall()
assert len(parents) == 2
states = conn.execute(node_states_table.select()).fetchall()
assert all(s.input_hash and s.output_hash and s.error_json is None for s in states)
artifacts = conn.execute(artifacts_table.select()).fetchall()
assert len(artifacts) == 1 and artifacts[0].content_hash == "test"
```
- Priority justification: audit trail integrity is core to ELSPETH; missing verification risks shipping non-compliant recording.
---
# Test Defect Report

## Summary

- Output assertions allow extra or duplicated sink rows because they only require `len(sink.rows) >= 1` instead of the exact expected count.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/engine/test_coalesce_integration.py:318` uses a non-strict count.
```python
# Sink should have received merged output
assert len(sink.rows) >= 1
merged = sink.rows[0]
```
- `tests/engine/test_coalesce_integration.py:466` repeats the non-strict count.
```python
# Merged output should have enriched=True from transform
assert len(sink.rows) >= 1
merged = sink.rows[0]
```

## Impact

- Duplicate outputs or incorrect coalesce behavior (e.g., emitting both branch rows plus a merged row) would still pass.
- Reduces confidence that coalesce produces exactly one merged row per input.

## Root Cause Hypothesis

- Assertions were loosened to avoid order/count flakiness but now mask correctness issues.

## Recommended Fix

- Tighten assertions to exact counts for single-row tests and verify no extra rows are emitted.
```python
assert len(sink.rows) == 1
merged = sink.rows[0]
```
- If ordering is nondeterministic, assert set-based equality on row content rather than using `>=`.
- Priority justification: prevents silent regressions in coalesce output cardinality.
