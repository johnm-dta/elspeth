# Test Defect Report

## Summary

- The mutability test does not verify TokenInfo is unfrozen; it only mutates the inner dict, which would still succeed even if the dataclass were frozen.

## Severity

- Severity: minor
- Priority: P2

## Category

- Weak Assertions

## Evidence

- `tests/contracts/test_identity.py:35` claims to verify “not frozen,” but the test only mutates `token.row_data` (see `tests/contracts/test_identity.py:39` and `tests/contracts/test_identity.py:40`), which is allowed even on frozen dataclasses.
```python
token = TokenInfo(row_id="r", token_id="t", row_data={"a": 1})
token.row_data["b"] = 2
```
- `src/elspeth/contracts/identity.py:19` and `src/elspeth/contracts/identity.py:20` explicitly state TokenInfo is NOT frozen because executors update tokens, but this contract requirement is not asserted in tests.

## Impact

- A regression to `@dataclass(frozen=True)` would not be caught, despite the contract requiring mutability at the object level.
- Tests provide false confidence about TokenInfo mutability, risking runtime failures if field reassignment is required in executors or future code.

## Root Cause Hypothesis

- Assumed that mutating a nested dict proves the dataclass is unfrozen; overlooked that frozen dataclasses still allow mutation of mutable field contents.

## Recommended Fix

- Add an explicit assertion that field assignment works (e.g., set `branch_name` or reassign `row_data`), or assert `token.__dataclass_params__.frozen is False`.
```python
token.branch_name = "sentiment"
assert token.branch_name == "sentiment"
```
- Priority justification: P2 because TokenInfo is a core identity contract and a frozen regression would break token updates without test coverage.
