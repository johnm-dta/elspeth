# Test Defect Report

## Summary

- Core `explain` tests validate only token/source_row and ignore audit trail fields (node states, routing events, calls, error records, outcome) that `LineageResult` is designed to surface.

## Severity

- Severity: major
- Priority: P1

## Category

- [Missing Audit Trail Verification]

## Evidence

- `LineageResult` includes audit trail fields that should be validated by lineage tests (`src/elspeth/core/landscape/lineage.py:41`, `src/elspeth/core/landscape/lineage.py:44`, `src/elspeth/core/landscape/lineage.py:47`, `src/elspeth/core/landscape/lineage.py:50`, `src/elspeth/core/landscape/lineage.py:53`, `src/elspeth/core/landscape/lineage.py:56`, `src/elspeth/core/landscape/lineage.py:59`).
```python
node_states: list[NodeState]
routing_events: list[RoutingEvent]
calls: list[Call]
parent_tokens: list[Token]
validation_errors: list[ValidationErrorRecord] = field(default_factory=list)
transform_errors: list[TransformErrorRecord] = field(default_factory=list)
outcome: TokenOutcome | None = None
```
- Core tests only assert token/source_row and never assert any audit trail fields (`tests/core/landscape/test_lineage.py:92`, `tests/core/landscape/test_lineage.py:93`, `tests/core/landscape/test_lineage.py:94`).
```python
result = explain(recorder, run_id=run.run_id, token_id=token.token_id)

assert isinstance(result, LineageResult)
assert result.token.token_id == token.token_id
assert result.source_row.row_id == row.row_id
```
- Other `explain` assertions in this file similarly skip audit fields (e.g., `tests/core/landscape/test_lineage.py:136`, `tests/core/landscape/test_lineage.py:203`, `tests/core/landscape/test_lineage.py:253`).

## Impact

- Regressions that drop or misorder `node_states`, omit errors, or lose `outcome` can pass the core lineage tests, reducing audit trail integrity and giving false confidence in explainability.

## Root Cause Hypothesis

- The tests focus on disambiguation control flow (row_id vs token_id) and basic existence checks, leaving audit completeness to other suites.

## Recommended Fix

- Add at least one test in this file that records node states, routing events, calls, and an outcome via `LandscapeRecorder`, then asserts that `explain()` returns them with correct values and ordering (e.g., `node_states` ordered by `step_index`, `outcome.outcome` matches recorded `RowOutcome`, and error lists reflect stored records).
- Extend existing tests (e.g., `test_explain_returns_lineage_result`) with explicit assertions for `node_states`, `routing_events`, `calls`, `parent_tokens`, `validation_errors`, `transform_errors`, and `outcome` to enforce audit trail completeness at the core test level.
---
# Test Defect Report

## Summary

- No test covers `explain`'s required-parameter guard (token_id and row_id both missing), leaving the ValueError contract unverified.

## Severity

- Severity: minor
- Priority: P2

## Category

- [Missing Edge Cases]

## Evidence

- `explain` explicitly raises when both identifiers are absent (`src/elspeth/core/landscape/lineage.py:90`, `src/elspeth/core/landscape/lineage.py:91`).
```python
if token_id is None and row_id is None:
    raise ValueError("Must provide either token_id or row_id")
```
- All `explain` calls in the test file pass `token_id` or `row_id`; there is no `pytest.raises` coverage for this guard (e.g., `tests/core/landscape/test_lineage.py:90`, `tests/core/landscape/test_lineage.py:134`, `tests/core/landscape/test_lineage.py:201`).
```python
result = explain(recorder, run_id=run.run_id, token_id=token.token_id)
```

## Impact

- If the guard is removed or changed, callers could pass invalid inputs and get misleading `None` results or undefined behavior without any test catching the regression.

## Root Cause Hypothesis

- Parameter validation was overlooked while focusing on happy-path and disambiguation scenarios.

## Recommended Fix

- Add a negative test that asserts the guard behavior, for example:
```python
with pytest.raises(ValueError, match="Must provide either token_id or row_id"):
    explain(recorder, run_id=run.run_id)
```
- Place it near other `explain` error-path tests in `tests/core/landscape/test_lineage.py` so the API contract remains enforced in core coverage.
