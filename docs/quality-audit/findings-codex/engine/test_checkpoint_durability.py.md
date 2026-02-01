# Test Defect Report

## Summary

- Checkpoint durability tests validate checkpoint counts but never assert audit trail records (node_states/token_outcomes/artifacts) for sink writes.

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/engine/test_checkpoint_durability.py:199` `tests/engine/test_checkpoint_durability.py:203` `tests/engine/test_checkpoint_durability.py:212` show the test ends after sink output + checkpoint counts without any audit-table assertions:
```python
result = orchestrator.run(config, graph=_build_test_graph(config))
assert result.status == "completed"
assert result.rows_processed == 5

# Verify: 5 rows written
assert len(sink.results) == 5
...
final_checkpoints = checkpoint_manager.get_checkpoints(run_id)
assert len(final_checkpoints) == 0, "Checkpoints should be deleted on success"
```
- `tests/engine/test_checkpoint_durability.py:417` `tests/engine/test_checkpoint_durability.py:421` only assert checkpoint count and unprocessed rows during recovery setup; there are no checks against `node_states`, `token_outcomes`, or `artifacts` tables:
```python
checkpoints = checkpoint_manager.get_checkpoints(run_id)
assert len(checkpoints) == 2, f"Expected 2 checkpoints, got {len(checkpoints)}"

recovery = RecoveryManager(db, checkpoint_manager)
unprocessed_row_ids = recovery.get_unprocessed_rows(run_id)
assert len(unprocessed_row_ids) == 3, f"Expected 3 unprocessed rows, got {len(unprocessed_row_ids)}"
```

## Impact

- Audit trail regressions (missing node_states, wrong hashes, absent token_outcomes, missing artifacts) could pass these tests, undermining durability guarantees.
- Creates false confidence that “durable output” is correctly recorded because only checkpoint rows are validated.

## Root Cause Hypothesis

- Tests focus narrowly on checkpoint mechanics and omit the auditability standard checks for Landscape tables.

## Recommended Fix

- After each run/resume, query `node_states`, `token_outcomes`, and `artifacts` to assert recorded state, hashes, and artifact metadata align with sink writes.
- Example pattern (adapt per test):
```python
from sqlalchemy import select
from elspeth.core.landscape.schema import node_states_table, token_outcomes_table, artifacts_table

with db.engine.connect() as conn:
    states = conn.execute(select(node_states_table).where(node_states_table.c.node_id == sink_node_id)).fetchall()
    assert len(states) == expected_rows
    assert all(s.status == "completed" for s in states)

    outcomes = conn.execute(select(token_outcomes_table).where(token_outcomes_table.c.run_id == run_id)).fetchall()
    assert {o.terminal_state for o in outcomes} == {"COMPLETED"}

    artifacts = conn.execute(select(artifacts_table).where(artifacts_table.c.run_id == run_id)).fetchall()
    assert all(a.content_hash for a in artifacts)
```
- Priority justification: audit trail integrity is a core ELSPETH requirement; missing verification risks silent audit corruption.
---
# Test Defect Report

## Summary

- `test_every_n_checkpointing_respects_sink_writes` asserts only the count of checkpoints, not which sequence numbers or tokens were checkpointed.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/engine/test_checkpoint_durability.py:878` `tests/engine/test_checkpoint_durability.py:880` show the test claims checkpoints should be at sequence 2 and 4 but only checks the count:
```python
# With 5 rows and checkpoint_interval=2:
# Checkpoints at sequence 2 and 4 (sequence 1, 3, 5 are not boundaries)
assert len(captured_checkpoints) == 2, (
    f"Expected 2 checkpoints (at sequence 2 and 4 with interval=2), "
    f"got {len(captured_checkpoints)}. "
    "every_n checkpointing should create checkpoints at interval boundaries."
)
```

## Impact

- A regression that checkpoints the wrong rows (e.g., sequence 1 and 3) would still pass, masking errors in every_n checkpointing logic.

## Root Cause Hypothesis

- Assertion was written to validate presence rather than correctness of the checkpoint boundaries.

## Recommended Fix

- Assert actual checkpoint sequence numbers (and/or token_ids/row values) match expected interval boundaries.
- Example:
```python
seqs = sorted(cp.sequence_number for cp in captured_checkpoints)
assert seqs == [2, 4]
```
- Priority justification: this protects core checkpoint-frequency behavior with minimal test changes.
---
# Test Defect Report

## Summary

- Tests mutate and read `ExecutionGraph` private fields (`_sink_id_map`, `_transform_id_map`, etc.) instead of using public graph construction/accessors.

## Severity

- Severity: minor
- Priority: P3

## Category

- Infrastructure Gaps

## Evidence

- `tests/engine/test_checkpoint_durability.py:79` `tests/engine/test_checkpoint_durability.py:81` directly assign private fields:
```python
graph._sink_id_map = sink_ids
graph._transform_id_map = transform_ids
```
- `tests/engine/test_checkpoint_durability.py:656` `tests/engine/test_checkpoint_durability.py:658` read private fields for assertions:
```python
sink_node_id = graph._sink_id_map["default"]
transform_node_id = graph._transform_id_map[0]
```

## Impact

- Tests are brittle to internal refactors of `ExecutionGraph` and bypass deterministic node-id generation, potentially hiding graph-construction issues.

## Root Cause Hypothesis

- Convenience helper avoids the public `ExecutionGraph.from_plugin_instances` builder and getters.

## Recommended Fix

- Build graphs via `ExecutionGraph.from_plugin_instances` and access maps via `get_sink_id_map()` / `get_transform_id_map()` in assertions.
- Example:
```python
graph = ExecutionGraph.from_plugin_instances(
    source=source,
    transforms=[transform],
    sinks={"default": sink},
    aggregations={},
    gates=[],
    output_sink="default",
)
sink_node_id = graph.get_sink_id_map()["default"]
```
- Priority justification: improves test stability and alignment with production graph behavior.
