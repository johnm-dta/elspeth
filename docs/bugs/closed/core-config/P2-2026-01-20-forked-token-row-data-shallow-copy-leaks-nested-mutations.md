# Bug Report: Forked TokenInfo row_data uses shallow copy (nested structures can leak across branches)

## Summary

- `TokenInfo.row_data` is a mutable dict by contract, and forked tokens are created with `data.copy()` (a shallow copy).
- If a row contains nested dict/list values (common for JSON sources), siblings created by fork share nested objects. A branch-local in-place mutation can affect other branches' token data, causing cross-branch contamination and audit confusion.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-20
- Related run/issue ID: N/A

## Environment

- Commit/branch: `8cfebea78be241825dd7487fed3773d89f2d7079` (main)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A

## Steps To Reproduce

1. Demonstrate shallow-copy leakage:

```python
data = {"payload": {"x": 1}}
child_a = data.copy()
child_b = data.copy()
child_a["payload"]["x"] = 999
assert child_b["payload"]["x"] == 999  # shared nested dict
```

2. In ELSPETH runtime, fork a token whose `row_data` contains nested objects (e.g., from a JSON source).
3. If any transform/gate mutates nested structures in-place on one branch, observe sibling branches' `row_data` also changing.

## Expected Behavior

- Forked tokens have independent row_data snapshots so branches cannot affect each other via shared object references.

## Actual Behavior

- Forking uses shallow copies, so nested objects can be shared across tokens.

## Evidence

- TokenInfo explicitly allows mutable dict data:
  - `src/elspeth/contracts/identity.py:10-26`
- Forking uses `data.copy()` (shallow):
  - `src/elspeth/engine/tokens.py:109-126`

## Impact

- User-facing impact: subtle, hard-to-debug incorrect results (branch outputs can influence each other).
- Data integrity / security impact: medium/high. Audit trail and determinism assumptions break if token data changes unexpectedly across branches.
- Performance or cost impact: N/A (fix may add CPU/memory if deep copying).

## Root Cause Hypothesis

- Fork implementation optimizes for speed using shallow copies, assuming rows are flat primitives, but contracts allow nested structures.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/engine/tokens.py`: use `copy.deepcopy(data)` when constructing forked child token `row_data`, OR
  - enforce/validate that row_data is JSON-primitive-only (no nested mutables) before forking (and crash if violated), OR
  - adopt an immutable row representation and require transforms to return new dicts (structural sharing is possible with persistent data structures, if desired).
- Tests to add/update:
  - Add a test that forks a token with nested row_data, mutates nested data in one child, and asserts siblings are unaffected.

## Architectural Deviations

- Spec or doc reference: CLAUDE.md invariants around auditability and "no inference"
- Observed divergence: shared mutable nested state across branches can make the audit record misleading
- Alignment plan or decision needed: decide between deep-copying vs enforcing immutable/flat row structures

## Acceptance Criteria

- Forked child tokens do not share nested mutable objects.
- A regression test demonstrates branch isolation for nested row_data.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/engine -k fork`
- New tests required: yes

---

## Resolution

**Status:** CLOSED
**Date:** 2026-01-21
**Resolved by:** Claude Opus 4.5

### Root Cause Confirmed

The `fork_token` method in `TokenManager` used `data.copy()` (shallow copy) at line 148. When row_data contains nested dicts or lists, all forked children share those nested objects, causing cross-branch mutation leakage.

### Fix Applied

Modified `src/elspeth/engine/tokens.py`:

1. Added `import copy` at module level
2. Changed `row_data=data.copy()` to `row_data=copy.deepcopy(data)` in the fork_token method
3. Added critical comment explaining why deepcopy is required for audit trail integrity

### Tests Added

Two new tests in `tests/engine/test_tokens.py` under `TestTokenManagerForkIsolation`:

1. `test_fork_nested_data_isolation` - Forks a token with nested dict and list, mutates one child's nested data, verifies siblings are unaffected
2. `test_fork_with_custom_nested_data_isolation` - Same test but with custom row_data provided to fork

### Verification

- All 14 token tests pass
- All 467 engine tests pass
- Type checking (mypy) passes
- Linting (ruff) passes

### Performance Note

`copy.deepcopy()` is slower than `dict.copy()` but is necessary for correctness. For ELSPETH's use case (audit trail integrity), correctness must take priority over performance. If performance becomes an issue with deeply nested structures, consider:
- Enforcing flat row structures at the source level
- Using structural sharing (e.g., immutable data structures)
