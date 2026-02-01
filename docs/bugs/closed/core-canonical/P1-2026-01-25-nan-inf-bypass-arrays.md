# Bug Report: NaN/Infinity Rejection Bypassed in Multi-Dimensional NumPy Arrays

## Summary

- Canonical JSON validation checks scalars for NaN/Infinity but doesn't check multi-dimensional NumPy arrays, allowing prohibited values to enter audit trail.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Branch Bug Scan (fix/rc1-bug-burndown-session-4)
- Date: 2026-01-25
- Related run/issue ID: BUG-CANON-01

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: NumPy array with NaN/Inf values

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of canonical.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Create row data with NumPy array containing NaN:
   ```python
   row = {"matrix": np.array([[1.0, float('nan')], [2.0, 3.0]])}
   ```
2. Hash row data via `stable_hash()`.
3. Observe NaN passes validation.

## Expected Behavior

- `stable_hash()` should reject NaN/Infinity in arrays with ValueError.

## Actual Behavior

- Scalar check passes arrays without element-wise validation.
- NaN/Inf enters audit trail.

## Evidence

```python
# Current check (canonical.py)
if isinstance(obj, (float, np.floating)):
    if math.isnan(obj) or math.isinf(obj):
        raise ValueError("NaN/Infinity not allowed")

# But multi-dimensional arrays not checked:
np.array([[1.0, float('nan')], [2.0, 3.0]])  # Passes!
```

## Impact

- User-facing impact: Silent acceptance of invalid data.
- Data integrity / security impact: NaN/Infinity in audit trail violates canonicalization guarantees, breaks hash determinism.
- Performance or cost impact: Cannot trust audit trail hashes for data integrity verification.

## Root Cause Hypothesis

- Array validation missing element-wise NaN/Inf checks.

## Proposed Fix

```python
if isinstance(obj, np.ndarray):
    if np.any(np.isnan(obj)) or np.any(np.isinf(obj)):
        raise ValueError(
            "NaN/Infinity found in NumPy array. "
            "Audit trail requires finite values only."
        )
    return obj.tolist()  # Then convert to list
```

- Config or schema changes: None.
- Tests to add/update:
  - `test_numpy_array_with_nan_rejected()` - 2D array with NaN
  - `test_numpy_array_with_inf_rejected()` - 3D array with Inf
  - `test_numpy_array_all_finite_accepted()` - Valid array passes

- Risks or migration steps: Existing pipelines with NaN/Inf arrays will now fail (acceptable per CLAUDE.md).

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` - Canonical JSON with NaN/Infinity rejection
- Observed divergence: Arrays bypass validation.
- Reason (if known): Scalar-only validation logic.
- Alignment plan or decision needed: Add array element-wise validation.

## Acceptance Criteria

- Multi-dimensional arrays with NaN/Inf rejected.
- Valid arrays pass validation.

## Tests

- Suggested tests to run: `pytest tests/core/test_canonical.py`
- New tests required: yes (3 tests above)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs:
  - `docs/bugs/BRANCH_BUG_TRIAGE_2026-01-25.md`
  - `CLAUDE.md` - Canonical JSON
