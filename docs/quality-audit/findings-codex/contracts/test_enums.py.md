# Test Defect Report

## Summary

- Uses `hasattr` to check enum members instead of direct access, violating the no-defensive-programming rule and weakening contract checks

## Severity

- Severity: minor
- Priority: P3

## Category

- Bug-Hiding Defensive Patterns

## Evidence

- Defensive presence checks via `hasattr` in `tests/contracts/test_enums.py:13` and `tests/contracts/test_enums.py:59`
- These checks only assert presence, not correct member/value/type, so they can pass with wrong attributes (`tests/contracts/test_enums.py:13`)
- Snippet:
```python
assert hasattr(Determinism, "DETERMINISTIC")
assert hasattr(RowOutcome, "COMPLETED")
```

## Impact

- Allows contract tests to pass even if enum members are replaced with mis-typed placeholders
- Conflicts with CLAUDE.md’s prohibition on defensive patterns, reducing test signal
- False confidence that enum contracts are strictly enforced

## Root Cause Hypothesis

- Presence-check pattern copied from generic Python checks instead of strict contract assertions
- Test file not aligned with repository’s no-defensive-programming rule

## Recommended Fix

- Replace `hasattr` checks with direct member access and explicit set/value assertions in `tests/contracts/test_enums.py`
- Example:
```python
expected = {"DETERMINISTIC", "SEEDED", "IO_READ", "IO_WRITE", "EXTERNAL_CALL", "NON_DETERMINISTIC"}
assert {e.name for e in Determinism} == expected
```
- Priority justification: low-risk, test-only change that restores contract strictness
---
# Test Defect Report

## Summary

- RowOutcome contract tests do not assert value strings or `is_terminal` for all outcomes; one test is assertion-free

## Severity

- Severity: minor
- Priority: P2

## Category

- Incomplete Contract Coverage

## Evidence

- The “all outcomes” test only touches the property without asserting expected values (`tests/contracts/test_enums.py:107`)
- Only COMPLETED/EXPANDED/BUFFERED value strings are asserted in `tests/contracts/test_enums.py:51`, `tests/contracts/test_enums.py:71`, and `tests/contracts/test_enums.py:77`
- Snippet:
```python
for outcome in RowOutcome:
    _ = outcome.is_terminal
```

## Impact

- ROUTED/FORKED/FAILED/QUARANTINED/CONSUMED_IN_BATCH/COALESCED value changes could slip without detection
- `is_terminal` regressions for unasserted outcomes could alter terminal-state accounting in audit data
- False confidence that RowOutcome coverage matches architecture and audit expectations

## Root Cause Hypothesis

- Tests were added incrementally for new outcomes (EXPANDED/BUFFERED) without expanding full coverage
- Contract tests rely on enum definitions as truth instead of asserting the full expected set

## Recommended Fix

- Add explicit expected mappings for all RowOutcome values and terminal flags in `tests/contracts/test_enums.py`
- Example:
```python
expected_values = {
    RowOutcome.COMPLETED: "completed",
    RowOutcome.ROUTED: "routed",
    RowOutcome.FORKED: "forked",
    RowOutcome.FAILED: "failed",
    RowOutcome.QUARANTINED: "quarantined",
    RowOutcome.CONSUMED_IN_BATCH: "consumed_in_batch",
    RowOutcome.COALESCED: "coalesced",
    RowOutcome.EXPANDED: "expanded",
    RowOutcome.BUFFERED: "buffered",
}
assert {o: o.value for o in RowOutcome} == expected_values
assert {o for o in RowOutcome if o.is_terminal} == {k for k in expected_values if k is not RowOutcome.BUFFERED}
```
- Priority justification: RowOutcome is stored in audit tables, so missing coverage risks silent contract drift
