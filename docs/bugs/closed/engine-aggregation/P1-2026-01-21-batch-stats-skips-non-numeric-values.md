# Bug Report: BatchStats skips non-numeric values instead of surfacing upstream type bugs

## Summary

- BatchStats converts values with float() and silently skips non-convertible inputs, which violates the transform contract (no coercion at transform boundary) and yields incorrect aggregates without surfacing upstream schema violations.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: not checked
- OS: not checked (workspace sandbox)
- Python version: not checked
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/transforms for bugs
- Model/version: GPT-5 Codex
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: reviewed BatchStats process logic

## Steps To Reproduce

1. Configure batch_stats with value_field: "amount" and a strict schema expecting numeric types.
2. Provide a batch with one row where amount is "not_a_number".
3. Observe the transform silently skips the row and returns count/sum based only on numeric values.

## Expected Behavior

- Wrong types in pipeline data should crash or surface as transform errors (upstream bug), not be skipped.

## Actual Behavior

- Non-numeric values are ignored, producing aggregates that do not reflect the actual batch and masking upstream bugs.

## Evidence

- Coercion and skip logic: src/elspeth/plugins/transforms/batch_stats.py:100-132
- Contract prohibits coercion at transform boundary: docs/contracts/plugin-protocol.md:151-176

## Impact

- User-facing impact: aggregates are silently wrong when data types are invalid.
- Data integrity / security impact: audit trail records incorrect stats without surfacing upstream bug.
- Performance or cost impact: downstream decisions based on incorrect aggregates.

## Root Cause Hypothesis

- BatchStats treats value_field as untrusted and attempts to coerce/skip, contradicting the pipeline trust model for transforms.

## Proposed Fix

- Code changes (modules/files): src/elspeth/plugins/transforms/batch_stats.py
- Config or schema changes: validate input schema and fail on non-numeric types; remove float() coercion or treat conversion failures as TransformResult.error.
- Tests to add/update: update tests to assert failures on wrong types (currently expect skipping).
- Risks or migration steps: existing pipelines with invalid types will now fail fast (intended per trust model).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/contracts/plugin-protocol.md:151-176 (Transforms MUST NOT coerce types)
- Observed divergence: transform converts/ignores wrong types instead of failing.
- Reason (if known): defensive handling added for convenience.
- Alignment plan or decision needed: enforce strict typing and fail on wrong types.

## Acceptance Criteria

- Non-numeric value_field inputs cause a crash or TransformResult.error (no silent skipping).
- Aggregates reflect the actual batch or the row is quarantined via on_error.

## Tests

- Suggested tests to run: pytest tests/plugins/transforms/test_batch_stats.py
- New tests required: yes, wrong-type handling and error routing.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/contracts/plugin-protocol.md

---

## Verification (2026-01-24)

**Status: STILL VALID**

### Current Code Analysis

Examined `/home/john/elspeth-rapid/src/elspeth/plugins/transforms/batch_stats.py`:

- Lines 100-106: `_try_convert_to_float()` method converts values to float, returns None on failure
- Lines 124-132: `process()` method silently skips non-convertible values with comment: "Their data can have non-numeric values - this is expected, not a bug"
- Line 134: `count = len(values)` only counts successfully converted values, not total rows

### Test Evidence

Test file `/home/john/elspeth-rapid/tests/plugins/transforms/test_batch_stats.py`:

- Lines 116-141: `test_skips_non_numeric_values()` explicitly tests and **expects** the silent skipping behavior
- Test provides `amount: "not_a_number"` and verifies it's skipped (count=2 instead of 3)
- Test confirms `batch_size=3` (total rows) vs `count=2` (numeric values only)

### Architectural Conflict

From `/home/john/elspeth-rapid/docs/contracts/plugin-protocol.md` lines 151-176:

| Zone | On Error | Plugin Action | ELSPETH Action |
|------|----------|---------------|----------------|
| **Their Data Types** | Wrong type at Transform/Sink | — | This is an upstream bug, should crash |

The contract explicitly states: "Transforms/Sinks MUST NOT coerce types" and "Wrong type at Transform/Sink... This is an upstream bug, should crash"

### Git History

- Original code added in commit c786410 (RC1, 2026-01-22)
- Comment justifying behavior was added in the initial commit
- No subsequent changes to the type conversion logic
- Commit cc0d364 (2026-01-21) only changed schema setup, not type handling

### Root Cause Confirmed

BatchStats treats aggregation as a special case where "missing or non-numeric values are expected" (per inline comment line 125). This conflicts with the Three-Tier Trust Model in CLAUDE.md which states:

> **Tier 2: Pipeline Data (Post-Source) - ELEVATED TRUST**
> - Transforms/sinks **expect conformance** - if types are wrong, that's an upstream plugin bug
> - **No coercion** at transform/sink level

The current behavior masks upstream schema violations. If a source or upstream transform outputs non-numeric values in a field expected to be numeric, BatchStats silently hides this bug instead of surfacing it.

### Impact Assessment

1. **Audit Integrity**: Statistics recorded in audit trail (`count`, `sum`, `mean`) are computed only from valid values, potentially misrepresenting the data
2. **Bug Masking**: If an upstream plugin has a type conversion bug, BatchStats will silently skip affected rows rather than failing fast
3. **Silent Data Loss**: Rows with conversion failures are excluded from aggregates without being quarantined or logged as errors

### Recommendation

Bug is **STILL VALID** and should be prioritized for fix:

1. Remove `_try_convert_to_float()` coercion helper
2. Expect numeric types directly (let ValueError/TypeError crash)
3. Update test `test_skips_non_numeric_values()` to verify crash behavior instead
4. If "partial aggregation despite bad rows" is desired, implement via explicit error handling with `TransformResult.error()` and quarantine routing, not silent skipping
