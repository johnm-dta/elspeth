# Bug Report: Decimal NaN/Infinity bypass non-finite rejection

## Summary

- `canonical_json` converts `Decimal("NaN")`/`Decimal("Infinity")` to JSON strings instead of raising, violating the stated "reject NaN/Infinity" policy and allowing non-finite numeric values into audit hashes.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Codex (GPT-5)
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: 81a0925d7d6de0d0e16fdd2d535f63d096a7d052 (fix/rc1-bug-burndown-session-2)
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic #91-Ubuntu SMP PREEMPT_DYNAMIC Tue Nov 18 14:14:30 UTC 2025 x86_64
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: You are a static analysis agent doing a deep bug audit.
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox; approval_policy=never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: ran `python` to evaluate `canonical_json(Decimal("NaN"))`.

## Steps To Reproduce

1. `from decimal import Decimal`
2. `from elspeth.core.canonical import canonical_json, stable_hash`
3. Call `canonical_json(Decimal("NaN"))` or `canonical_json(Decimal("Infinity"))`

## Expected Behavior

- A `ValueError` is raised for non-finite Decimal values, consistent with the "reject NaN/Infinity" policy.

## Actual Behavior

- `canonical_json(Decimal("NaN"))` returns the JSON string `"NaN"`; `canonical_json(Decimal("Infinity"))` returns `"Infinity"`, and `stable_hash` succeeds.

## Evidence

- Logs or stack traces: local run shows `canonical_json(Decimal("NaN"))` -> `"NaN"` and `canonical_json(Decimal("Infinity"))` -> `"Infinity"`.
- Artifacts (paths, IDs, screenshots): `src/elspeth/core/canonical.py:47` (non-finite check only covers float/np.floating); `src/elspeth/core/canonical.py:87` (Decimal converted to string without non-finite validation).
- Minimal repro input (attach or link): `Decimal("NaN")`, `Decimal("Infinity")`.

## Impact

- User-facing impact: non-finite numeric values can be accepted and hashed without error.
- Data integrity / security impact: violates audit integrity policy by silently converting non-finite numbers into strings, masking invalid data states.
- Performance or cost impact: negligible.

## Root Cause Hypothesis

- `_normalize_value` only rejects non-finite `float`/`np.floating` and converts `Decimal` to `str` without checking `Decimal.is_finite()`.

## Proposed Fix

- Code changes (modules/files): add a non-finite guard for `Decimal` in `src/elspeth/core/canonical.py` before converting to string (e.g., `if not obj.is_finite(): raise ValueError(...)`).
- Config or schema changes: None.
- Tests to add/update: add tests in `tests/core/test_canonical.py` or `tests/property/canonical/test_nan_rejection.py` asserting `canonical_json` and `stable_hash` raise `ValueError` for `Decimal("NaN")`, `Decimal("Infinity")`, and `Decimal("-Infinity")`.
- Risks or migration steps: existing datasets with Decimal non-finite values will start failing fast (desired per policy).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:324` and `src/elspeth/core/canonical.py:33`.
- Observed divergence: policy requires rejecting NaN/Infinity, but Decimal non-finite values are accepted and converted to strings.
- Reason (if known): missing Decimal non-finite validation.
- Alignment plan or decision needed: enforce non-finite checks for Decimal to match canonicalization policy.

## Acceptance Criteria

- `canonical_json(Decimal("NaN"))`, `canonical_json(Decimal("Infinity"))`, and `stable_hash` on those values raise `ValueError`.
- Added tests pass.

## Tests

- Suggested tests to run: `python -m pytest tests/core/test_canonical.py tests/property/canonical/test_nan_rejection.py`
- New tests required: yes, Decimal non-finite rejection cases.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: `CLAUDE.md:324`
---
# Bug Report: Nested numpy arrays skip recursive normalization

## Summary

- Multi-dimensional `np.ndarray` values are only shallowly normalized, leaving nested lists unprocessed; arrays containing pandas/numpy objects (e.g., `pd.Timestamp`) trigger `CanonicalizationError` instead of producing canonical JSON.

## Severity

- Severity: minor
- Priority: P2

## Reporter

- Name or handle: Codex (GPT-5)
- Date: 2026-01-22
- Related run/issue ID: Unknown

## Environment

- Commit/branch: 81a0925d7d6de0d0e16fdd2d535f63d096a7d052 (fix/rc1-bug-burndown-session-2)
- OS: Linux nyx.foundryside.dev 6.8.0-90-generic #91-Ubuntu SMP PREEMPT_DYNAMIC Tue Nov 18 14:14:30 UTC 2025 x86_64
- Python version: 3.13.1
- Config profile / env vars: Unknown
- Data set or fixture: Unknown

## Agent Context (if relevant)

- Goal or task prompt: You are a static analysis agent doing a deep bug audit.
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox; approval_policy=never
- Determinism details (seed, run ID): Unknown
- Notable tool calls or steps: ran `python` to evaluate `canonical_json` on a 2D numpy array with `pd.Timestamp`.

## Steps To Reproduce

1. `import numpy as np; import pandas as pd`
2. `from elspeth.core.canonical import canonical_json`
3. `arr = np.array([[pd.Timestamp("2026-01-12 10:30:00")]], dtype=object); canonical_json({"array": arr})`

## Expected Behavior

- The nested array is fully normalized and serialized (timestamp converted to UTC ISO string).

## Actual Behavior

- `canonical_json` raises `CanonicalizationError` (`unsupported type: <class 'pandas._libs.tslibs.timestamps.Timestamp'>`).

## Evidence

- Logs or stack traces: local run raises `CanonicalizationError unsupported type: <class 'pandas._libs.tslibs.timestamps.Timestamp'>` for the repro above.
- Artifacts (paths, IDs, screenshots): `src/elspeth/core/canonical.py:64` (np.ndarray normalization only applies `_normalize_value` to top-level elements), `src/elspeth/core/canonical.py:93` (recursive normalization handles lists/tuples but is bypassed by the ndarray path).
- Minimal repro input (attach or link): `np.array([[pd.Timestamp("2026-01-12 10:30:00")]], dtype=object)`.

## Impact

- User-facing impact: canonicalization fails for valid array-shaped data containing pandas/numpy types, causing pipeline crashes when such structures appear.
- Data integrity / security impact: none directly; failure prevents hashing rather than corrupting data.
- Performance or cost impact: negligible.

## Root Cause Hypothesis

- `np.ndarray` handling uses `_normalize_value` rather than recursive normalization, so nested lists from multi-dimensional arrays are not walked.

## Proposed Fix

- Code changes (modules/files): in `src/elspeth/core/canonical.py`, change ndarray normalization to recurse (e.g., `return _normalize_for_canonical(obj.tolist())` or apply `_normalize_for_canonical` to each element).
- Config or schema changes: None.
- Tests to add/update: add tests for 2D `np.ndarray` containing `pd.Timestamp` (and/or nested numpy scalars) to ensure canonicalization succeeds.
- Risks or migration steps: none; behavior becomes more permissive for nested arrays.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `src/elspeth/core/canonical.py:5` and `src/elspeth/core/canonical.py:94`.
- Observed divergence: normalization claims to convert pandas/numpy types to JSON-safe primitives but fails for nested ndarray structures.
- Reason (if known): ndarray branch skips recursive normalization.
- Alignment plan or decision needed: normalize ndarray contents recursively to honor canonicalization contract.

## Acceptance Criteria

- `canonical_json` succeeds for a 2D `np.ndarray` containing `pd.Timestamp` (and similar nested pandas/numpy values), producing JSON with normalized values.
- Added tests pass.

## Tests

- Suggested tests to run: `python -m pytest tests/core/test_canonical.py`
- New tests required: yes, nested ndarray normalization cases.

## Notes / Links

- Related issues/PRs: Unknown
- Related design docs: Unknown
