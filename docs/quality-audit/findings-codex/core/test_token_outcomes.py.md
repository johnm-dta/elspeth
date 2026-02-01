# Test Defect Report

## Summary

- record_token_outcome tests do not verify any audit trail writes or field correctness, only that an outcome_id was returned

## Severity

- Severity: major
- Priority: P1

## Category

- Missing Audit Trail Verification

## Evidence

- `tests/core/test_token_outcomes.py:247` only checks the return value, not the token_outcomes table or stored fields:
```python
outcome_id = recorder.record_token_outcome(
    run_id=run.run_id,
    token_id=token.token_id,
    outcome=RowOutcome.COMPLETED,
    sink_name="output",
)

assert outcome_id is not None
assert outcome_id.startswith("out_")
```
- `tests/core/test_token_outcomes.py:262` and `tests/core/test_token_outcomes.py:276` repeat the same pattern (no audit table validation) for ROUTED and BUFFERED→terminal:
```python
outcome_id = recorder.record_token_outcome(
    run_id=run.run_id,
    token_id=token.token_id,
    outcome=RowOutcome.ROUTED,
    sink_name="errors",
)
assert outcome_id is not None
```
- Nowhere in this class is `token_outcomes_table` queried or `get_token_outcome` used to assert stored values (run_id, token_id, outcome, is_terminal, sink_name, batch_id, error_hash, context_json), despite AUD-001 requirements.

## Impact

- A bug that writes incorrect outcome values, wrong run_id/token_id, incorrect is_terminal flags, or omits optional fields would still pass these tests.
- Audit trail integrity regressions could slip through because the tests never inspect the persisted record.

## Root Cause Hypothesis

- Tests were written as API smoke checks (return value only) rather than audit-integrity checks, which conflicts with ELSPETH’s auditability standard.

## Recommended Fix

- After each `record_token_outcome` call in `TestRecordTokenOutcome`, query `token_outcomes_table` (or use `recorder.get_token_outcome`) and assert all relevant fields:
  - `run_id`, `token_id`, `outcome`, `is_terminal`, `recorded_at` not null
  - outcome-specific fields (`sink_name`, `batch_id`, `fork_group_id`, `join_group_id`, `expand_group_id`, `error_hash`, `context_json`)
- For BUFFERED→terminal, assert two rows exist (one non-terminal, one terminal) and verify correct `is_terminal` values.
- Priority justification: this is core audit-trail recording logic; missing verification undermines AUD-001 guarantees.
---
# Test Defect Report

## Summary

- “returns terminal over buffered” test never records a buffered outcome, so it doesn’t exercise the intended behavior

## Severity

- Severity: minor
- Priority: P2

## Category

- Missing Edge Cases

## Evidence

- Fixture only records a terminal outcome (`RowOutcome.COMPLETED`) and no buffered outcome (`RowOutcome.BUFFERED`): `tests/core/test_token_outcomes.py:340-359`.
- The test asserts terminal status without any competing buffered outcome: `tests/core/test_token_outcomes.py:371-376`.
```python
# fixture
outcome_id = recorder.record_token_outcome(run.run_id, token.token_id, RowOutcome.COMPLETED, sink_name="out")

# test
result = recorder.get_token_outcome(token.token_id)
assert result.is_terminal is True
```

## Impact

- A regression where `get_token_outcome` returns a buffered outcome even when a terminal exists would go undetected.
- Explain/lineage queries could surface the wrong outcome state without test coverage.

## Root Cause Hypothesis

- The test name suggests an intended buffered-vs-terminal scenario, but setup was copied from the simpler fixture and never updated to include a buffered record.

## Recommended Fix

- In `test_get_token_outcome_returns_terminal_over_buffered`, explicitly record a BUFFERED outcome first (or in fixture), then record a terminal outcome for the same token.
- Assert the returned outcome is the terminal one (e.g., `result.outcome == RowOutcome.COMPLETED` and `result.is_terminal is True`) to validate selection ordering.
