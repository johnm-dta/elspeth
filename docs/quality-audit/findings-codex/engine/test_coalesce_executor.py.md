# Test Defect Report

## Summary

- Timeout-related tests rely on real `time.sleep`, making them timing-dependent and flaky.

## Severity

- Severity: major
- Priority: P1

## Category

- [Sleepy Assertions]

## Evidence

- `tests/engine/test_coalesce_executor.py:481-486`
```python
        # Wait for timeout
        time.sleep(0.15)

        # check_timeouts should return empty list (quorum not met)
        timed_out = executor.check_timeouts("quorum_merge", step_in_pipeline=2)
```
- `tests/engine/test_coalesce_executor.py:565-569`
```python
        # Wait for timeout
        time.sleep(0.15)

        # Check timeout and force merge
        timed_out = executor.check_timeouts("best_effort_merge", step_in_pipeline=2)
```

## Impact

- Test outcomes depend on scheduler/CI timing and can intermittently fail or slow down.
- Flaky timeout tests reduce confidence in timeout behavior and slow feedback loops.

## Root Cause Hypothesis

- Tests simulate timeouts with wall-clock sleeps instead of controlling `time.monotonic`, so timing is non-deterministic.

## Recommended Fix

- Replace sleeps with a deterministic clock using `monkeypatch` on `elspeth.engine.coalesce_executor.time.monotonic`.
- Example pattern:
```python
import elspeth.engine.coalesce_executor as coalesce_mod

t = 100.0
def fake_monotonic() -> float:
    return t

monkeypatch.setattr(coalesce_mod.time, "monotonic", fake_monotonic)

executor.accept(token_a, "best_effort_merge", step_in_pipeline=2)
t += 0.2  # advance past timeout
timed_out = executor.check_timeouts("best_effort_merge", step_in_pipeline=2)
```
- Priority justification: eliminating real sleeps removes flakiness and speeds tests (P1 per flaky-test guideline).
---
# Test Defect Report

## Summary

- Coalesce tests assert only in-memory outcomes/metadata and never verify Landscape audit records (node_states hashes, lineage).

## Severity

- Severity: major
- Priority: P1

## Category

- [Missing Audit Trail Verification]

## Evidence

- `tests/engine/test_coalesce_executor.py:686-703` checks `coalesce_metadata` only and ends without querying the audit DB:
```python
        # Verify coalesce_metadata is populated
        assert outcome2.coalesce_metadata is not None
        metadata = outcome2.coalesce_metadata

        # Check required fields
        assert metadata["policy"] == "require_all"
        assert metadata["merge_strategy"] == "union"
        assert metadata["expected_branches"] == ["path_a", "path_b"]
        assert set(metadata["branches_arrived"]) == {"path_a", "path_b"}
        assert metadata["wait_duration_ms"] >= 0
```
- `src/elspeth/engine/coalesce_executor.py:236-249` writes node_states, which are not verified by any test in this file:
```python
        for token in consumed_tokens:
            state = self._recorder.begin_node_state(
                token_id=token.token_id,
                node_id=node_id,
                step_index=step_in_pipeline,
                input_data=token.row_data,
            )
            self._recorder.complete_node_state(
                state_id=state.state_id,
                status="completed",
                output_data={"merged_into": merged_token.token_id},
                duration_ms=0,
            )
```

## Impact

- Audit trail regressions (missing node_states, incorrect hashes, broken lineage) can ship undetected.
- Undermines the Auditability Standard and Three-Tier Trust Model requirements for verifiable records.

## Root Cause Hypothesis

- Tests focus on functional merge behavior and metadata, with audit validation deferred or overlooked.

## Recommended Fix

- Extend `test_coalesce_records_audit_metadata` (or add a new test) to query the recorder and verify audit records.
- Example checks:
```python
from elspeth.core.canonical import stable_hash
from elspeth.contracts import NodeStateStatus

states = recorder.get_node_states_for_token(token_a.token_id)
assert len(states) == 1
state = states[0]
assert state.status == NodeStateStatus.COMPLETED
assert state.input_hash == stable_hash(token_a.row_data)
assert state.output_hash == stable_hash({"merged_into": outcome2.merged_token.token_id})

parents = recorder.get_token_parents(outcome2.merged_token.token_id)
assert [p.parent_token_id for p in parents] == [token_a.token_id, token_b.token_id]
```
- Priority justification: audit trail verification is critical for ELSPETH’s compliance guarantees (P1).
