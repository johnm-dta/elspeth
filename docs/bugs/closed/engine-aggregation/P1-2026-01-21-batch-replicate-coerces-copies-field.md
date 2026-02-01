# Bug Report: BatchReplicate coerces copies_field and masks upstream type bugs

## Summary

- BatchReplicate uses int() coercion and defaults for copies_field values, which violates the transform contract (no coercion at transform boundary) and can silently produce incorrect replication counts.

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
- Notable tool calls or steps: reviewed BatchReplicate process logic

## Steps To Reproduce

1. Configure batch_replicate with copies_field: "copies" and default_copies: 1.
2. Provide a row where copies is "3" (string) or "abc" or None.
3. Observe the transform coerces or defaults instead of failing.

## Expected Behavior

- Wrong types in pipeline data should crash or surface as transform errors (upstream bug), not be coerced or defaulted.

## Actual Behavior

- Non-int values are coerced via int() or replaced by default_copies, silently changing behavior.

## Evidence

- Coercion and defaulting logic: src/elspeth/plugins/transforms/batch_replicate.py:122-130
- Contract prohibits coercion at transform boundary: docs/contracts/plugin-protocol.md:151-176

## Impact

- User-facing impact: replication counts are silently wrong on bad data.
- Data integrity / security impact: audit trail records incorrect outputs without surfacing upstream bug.
- Performance or cost impact: extra or missing rows change downstream workload.

## Root Cause Hypothesis

- BatchReplicate treats copies_field as untrusted and attempts to coerce, contradicting pipeline trust model for transforms.

## Proposed Fix

- Code changes (modules/files): src/elspeth/plugins/transforms/batch_replicate.py
- Config or schema changes: require copies_field type in schema and access directly; remove int() coercion and default fallback for type errors.
- Tests to add/update: add tests asserting wrong-type copies_field raises or returns TransformResult.error.
- Risks or migration steps: pipelines with invalid types will now fail fast (intended per trust model).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/contracts/plugin-protocol.md:151-176 (Transforms MUST NOT coerce types)
- Observed divergence: transform converts/normalizes wrong types instead of failing.
- Reason (if known): defensive handling added for convenience.
- Alignment plan or decision needed: enforce strict typing and fail on wrong types.

## Acceptance Criteria

- copies_field type violations cause a crash or TransformResult.error (no coercion/defaulting).
- Valid int values replicate exactly; invalid types surface upstream bugs.

## Tests

- Suggested tests to run: pytest tests/plugins/transforms/test_batch_replicate.py
- New tests required: yes, wrong-type copies_field behavior.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/contracts/plugin-protocol.md

---

## VERIFICATION: 2026-01-24

**Status:** STILL VALID

**Verified By:** Claude Code verification task

**Current Code Analysis:**

The bug is **still present** in the current codebase. Examination of `/home/john/elspeth-rapid/src/elspeth/plugins/transforms/batch_replicate.py` confirms:

1. **Coercion still active (lines 122-130):**
   ```python
   copies = self._default_copies
   if self._copies_field in row:
       raw_copies = row[self._copies_field]
       try:
           copies = int(raw_copies)  # COERCION - violates transform contract
           if copies < 1:
               copies = self._default_copies
       except (TypeError, ValueError):
           copies = self._default_copies  # DEFAULTING - hides upstream bug
   ```

2. **Contract violation confirmed:**
   - Transform uses `int()` coercion on pipeline data (row values)
   - Falls back to `default_copies` on type/value errors
   - Both behaviors violate docs/contracts/plugin-protocol.md:151-176 ("Transforms MUST NOT coerce types")

**Git History:**

- File last modified: commit `c786410` (2026-01-22) - "ELSPETH - Release Candidate 1"
- No changes to type handling since bug was filed (2026-01-21)
- No uncommitted changes in working tree

**Related Evidence:**

A similar bug in `JSONExplode` (P2-2026-01-19-json-explode-iterable-nonstrict-types.md) was **CLOSED** on 2026-01-23 with the correct fix pattern:

```python
# From json_explode.py lines 138-142
if not isinstance(array_value, list):
    raise TypeError(
        f"Field '{self._array_field}' must be a list, got {type(array_value).__name__}. "
        f"This indicates an upstream validation bug - check source schema or prior transforms."
    )
```

This demonstrates the **correct approach**: explicit type check that crashes on wrong types instead of coercing/defaulting.

**Sibling Bug:**

`P1-2026-01-21-batch-stats-skips-non-numeric-values.md` - same pattern (float() coercion + silent skipping) in a different transform. Both need the same fix approach.

**Impact Assessment:**

- **Audit integrity risk:** Transform can silently use wrong replication counts without recording the type violation
- **Downstream effects:** Incorrect row counts propagate through pipeline, potentially causing wrong batch sizes, aggregations, or sink outputs
- **No test coverage:** `tests/plugins/transforms/test_batch_replicate.py` does not exist to catch this behavior

**Recommended Fix Pattern:**

Follow the `JSONExplode` precedent:
1. Add explicit `isinstance(copies, int)` check after extracting from row
2. Raise `TypeError` with clear message if type is wrong
3. Remove `int()` coercion and exception handling
4. Let `default_copies` only apply when field is missing (not when type is wrong)
5. Add tests for type enforcement

**Next Steps:**

- Create test file `tests/plugins/transforms/test_batch_replicate.py` with type violation test cases
- Implement explicit type check (no coercion)
- Consider fixing both `batch_replicate` and `batch_stats` in same changeset (same root cause)
