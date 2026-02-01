# Bug Report: Decimal NaN/Infinity bypass non-finite rejection

## Summary

`canonical_json` converts `Decimal("NaN")`/`Decimal("Infinity")` to JSON strings instead of raising, violating the stated "reject NaN/Infinity" policy and allowing non-finite numeric values into audit hashes.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (GPT-5)
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: main (d8df733)
- OS: Linux
- Python version: 3.12+
- Config profile / env vars: Any
- Data set or fixture: Data containing Decimal("NaN") or Decimal("Infinity")

## Agent Context (if relevant)

- Goal or task prompt: Static analysis deep bug audit for canonical.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox; approval_policy=never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: Ran `python` to evaluate `canonical_json(Decimal("NaN"))`

## Steps To Reproduce

1. `from decimal import Decimal`
2. `from elspeth.core.canonical import canonical_json, stable_hash`
3. Call `canonical_json(Decimal("NaN"))` or `canonical_json(Decimal("Infinity"))`

## Expected Behavior

- A `ValueError` is raised for non-finite Decimal values, consistent with the "reject NaN/Infinity" policy

## Actual Behavior

- `canonical_json(Decimal("NaN"))` returns the JSON string `"NaN"`
- `canonical_json(Decimal("Infinity"))` returns `"Infinity"`
- `stable_hash` succeeds

## Evidence

- Logs or stack traces: Local run shows `canonical_json(Decimal("NaN"))` -> `"NaN"` and `canonical_json(Decimal("Infinity"))` -> `"Infinity"`
- Artifacts (paths, IDs, screenshots):
  - `src/elspeth/core/canonical.py:47` - non-finite check only covers float/np.floating
  - `src/elspeth/core/canonical.py:87` - Decimal converted to string without non-finite validation
- Minimal repro input (attach or link): `Decimal("NaN")`, `Decimal("Infinity")`

## Impact

- User-facing impact: Non-finite numeric values can be accepted and hashed without error
- Data integrity / security impact: Violates audit integrity policy by silently converting non-finite numbers into strings, masking invalid data states
- Performance or cost impact: Negligible

## Root Cause Hypothesis

`_normalize_value` only rejects non-finite `float`/`np.floating` and converts `Decimal` to `str` without checking `Decimal.is_finite()`.

## Proposed Fix

- Code changes (modules/files): Add a non-finite guard for `Decimal` in `src/elspeth/core/canonical.py` before converting to string (e.g., `if not obj.is_finite(): raise ValueError(...)`)
- Config or schema changes: None
- Tests to add/update: Add tests in `tests/core/test_canonical.py` or `tests/property/canonical/test_nan_rejection.py` asserting `canonical_json` and `stable_hash` raise `ValueError` for `Decimal("NaN")`, `Decimal("Infinity")`, and `Decimal("-Infinity")`
- Risks or migration steps: Existing datasets with Decimal non-finite values will start failing fast (desired per policy)

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:324` and `src/elspeth/core/canonical.py:33`
- Observed divergence: Policy requires rejecting NaN/Infinity, but Decimal non-finite values are accepted and converted to strings
- Reason (if known): Missing Decimal non-finite validation
- Alignment plan or decision needed: Enforce non-finite checks for Decimal to match canonicalization policy

## Acceptance Criteria

- `canonical_json(Decimal("NaN"))`, `canonical_json(Decimal("Infinity"))`, and `stable_hash` on those values raise `ValueError`
- Added tests pass

## Tests

- Suggested tests to run: `python -m pytest tests/core/test_canonical.py tests/property/canonical/test_nan_rejection.py`
- New tests required: Yes, Decimal non-finite rejection cases

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:324`

## Verification Status

- [x] Bug confirmed via reproduction
- [x] Root cause verified
- [x] Fix implemented
- [x] Tests added
- [x] Fix verified

## Resolution

**Fixed by:** Claude Opus 4.5 (2026-01-23)

**Changes:**
- `src/elspeth/core/canonical.py`: Added `is_finite()` check for `Decimal` in `_normalize_value()`, updated docstring
- `tests/core/test_canonical.py`: Added `TestDecimalNonFiniteRejection` class with parameterized tests for NaN, sNaN, Infinity, -Infinity

**Verification:**
- All 5 new tests pass
- Full test suite (3,160 tests) passes with no regressions
- Golden hash stability test confirms existing hashes unchanged
