# Test Defect Report

## Summary

- RowProcessor tests assert RowResult outputs but skip mandatory Landscape audit trail verification (node_states/token_outcomes/hashes/lineage) on core paths (success, routing, fork, quarantine).

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/engine/test_processor.py:121` checks output only; no audit DB assertions (node_states/token_outcomes/hashes).
```python
assert result.final_data == {"value": 21}
assert result.outcome == RowOutcome.COMPLETED
```
- `tests/engine/test_processor.py:355` ends the quarantine test with RowResult assertions only; no token_outcome or error_hash verification.
```python
assert result.outcome == RowOutcome.QUARANTINED
assert result.final_data == {"value": -5}
```
- `tests/engine/test_processor.py:601` asserts routed outcome/sink name but never validates routing events or token_outcome records.
```python
assert result.outcome == RowOutcome.ROUTED
assert result.sink_name == "high_values"
```
- `tests/engine/test_processor.py:699` and `tests/engine/test_processor.py:712` validate fork counts/branch_name but do not assert parent_token_id or fork_group_id lineage in audit tables.
```python
forked_results = [r for r in results if r.outcome == RowOutcome.FORKED]
...
assert child.token.branch_name in ("path_a", "path_b")
```
- `src/elspeth/engine/processor.py:709` and `src/elspeth/engine/processor.py:775` show RowProcessor records audit outcomes that these tests never validate.
```python
self._recorder.record_token_outcome(... outcome=RowOutcome.FORKED, ...)
...
self._recorder.record_token_outcome(... outcome=RowOutcome.QUARANTINED, ...)
```

## Impact

- Audit trail regressions (missing token_outcomes or node_states) can slip through undetected.
- Hash integrity (input_hash/output_hash) could break without test failures, undermining auditability guarantees.
- Lineage/routing errors (fork_group_id, parent_token_id, sink_name) would pass despite corrupting traceability.
- Tests give false confidence by validating only in-memory RowResult fields.

## Root Cause Hypothesis

- Tests were written to validate RowResult behavior only, before auditability checks became mandatory.
- Audit assertions were added only for one quarantine case, leaving other critical paths unverified.

## Recommended Fix

- Add audit trail assertions in each path that triggers recording: use `recorder.get_node_states_for_token` and `recorder.get_token_outcome` for success, quarantined, routed, and forked tokens.
- Verify hash integrity via `stable_hash` against `NodeState.input_hash`/`output_hash` and check `TokenOutcome.error_hash` for QUARANTINED/FAILED.
- For forks, assert lineage via `recorder.get_token_parents(child.token_id)` and validate parent `fork_group_id`.
- Example pattern to add:
```python
from elspeth.core.canonical import stable_hash

states = recorder.get_node_states_for_token(result.token_id)
assert states[0].input_hash == stable_hash({"value": -5})

outcome = recorder.get_token_outcome(result.token_id)
assert outcome.outcome == RowOutcome.QUARANTINED
assert outcome.error_hash is not None
```
- Priority P1 because audit trail integrity is a core ELSPETH requirement.
---
# Test Defect Report

## Summary

- Work queue guard test never exercises the iteration limit; it only asserts normal completion, so the guard could be removed without failing tests.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/engine/test_processor.py:1096` sets a low iteration cap, but the test does not create any extra work items.
```python
proc_module.MAX_WORK_QUEUE_ITERATIONS = 5
```
- `tests/engine/test_processor.py:1104` and `tests/engine/test_processor.py:1111` show the test only asserts a normal COMPLETED result.
```python
results = processor.process_row(..., transforms=[], ...)
assert len(results) == 1
assert results[0].outcome == RowOutcome.COMPLETED
```
- `src/elspeth/engine/processor.py:523` raises RuntimeError when the guard is exceeded, but this path is never exercised.
```python
if iterations > MAX_WORK_QUEUE_ITERATIONS:
    raise RuntimeError(...)
```

## Impact

- The guard can regress or be removed without test failures.
- Infinite loop bugs in work-queue handling would surface only in production.
- The test provides a false sense of protection against runaway iteration.

## Root Cause Hypothesis

- The test avoided constructing an actual infinite loop, leaving the guard untested.
- No harness was added to simulate a work-queue overflow scenario.

## Recommended Fix

- Use `pytest` monkeypatch to force `_process_single_token` to re-enqueue a child work item repeatedly and assert `RuntimeError`.
- Keep `MAX_WORK_QUEUE_ITERATIONS` low to make the test fast and deterministic.
- Example pattern:
```python
import elspeth.engine.processor as proc_module

def loop_process(self, token, transforms, ctx, start_step, coalesce_at_step=None, coalesce_name=None):
    return (None, [proc_module._WorkItem(token=token, start_step=0)])

monkeypatch.setattr(RowProcessor, "_process_single_token", loop_process)
proc_module.MAX_WORK_QUEUE_ITERATIONS = 5
with pytest.raises(RuntimeError):
    processor.process_row(row_index=0, row_data={"value": 1}, transforms=[], ctx=ctx)
```
- Priority P2 because this is a guardrail test that should fail when the guard regresses.
