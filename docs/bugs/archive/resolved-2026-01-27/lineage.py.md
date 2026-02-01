# Bug Report: explain() silently drops missing parent tokens, masking audit integrity violations

## RESOLUTION (2026-01-27)

**Status: FIXED**

This was a **real bug** that violated CLAUDE.md's Tier 1 trust model. Fixed by making `explain()` crash on missing parent tokens instead of silently returning incomplete lineage.

### Fix Details:
- **File:** `src/elspeth/core/landscape/lineage.py:167-182`
- **Change:** Replaced silent `if parent_token is not None` skip with explicit `ValueError` raise
- **Test:** Added `test_explain_crashes_on_missing_parent_token` to verify behavior

### Defense in Depth:
- **Primary defense:** FK constraints prevent deletion of parent tokens (this was already working)
- **Secondary defense:** Code now crashes if corruption is detected (e.g., from external data import)

### Verification:
```bash
.venv/bin/python -m pytest tests/core/landscape/test_lineage.py -v  # 14 passed
```

---

## Summary (Original Report)

- Lineage assembly ignores missing parent_token_id references, returning incomplete parent_tokens without surfacing the audit DB anomaly.

## Severity

- Severity: ~~major~~ **FIXED**
- Priority: ~~P1~~ **CLOSED**

## Reporter

- Name or handle: Codex
- Date: 2026-01-25
- Related run/issue ID: N/A

## Environment

- Commit/branch: 8635789
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: SQLite in-memory with a token_parents row referencing a non-existent parent token

## Agent Context (if relevant)

- Goal or task prompt: Static analysis bug audit for src/elspeth/core/landscape/lineage.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Create a run, row, and token using LandscapeRecorder.
2. Disable FK checks on the SQLite connection and insert a token_parents row where parent_token_id does not exist.
3. Call `explain(recorder, run_id=<run_id>, token_id=<child_token_id>)`.

## Expected Behavior

- explain() should raise an error indicating an audit integrity violation when a parent token is missing.

## Actual Behavior

- explain() silently skips the missing parent and returns LineageResult with incomplete parent_tokens.

## Evidence

- `src/elspeth/core/landscape/lineage.py:168-173` drops missing parents with `if parent_token is not None`, which hides integrity violations.
- `CLAUDE.md:40-43` and `CLAUDE.md:125-127` require crashing on any audit DB anomaly instead of silently recovering.

## Impact

- User-facing impact: explain() can return incomplete lineage for fork/coalesce paths without warning.
- Data integrity / security impact: audit trail integrity violations are masked, undermining traceability guarantees.
- Performance or cost impact: None.

## Root Cause Hypothesis

- Defensive handling in lineage assembly treats missing parent tokens as optional, violating the Tier 1 trust model for audit data.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/landscape/lineage.py`: raise a ValueError (or similar) when a parent_token_id cannot be resolved, including token_id and parent_token_id in the message.
- Config or schema changes: None.
- Tests to add/update:
  - Add a test that inserts a token_parents row with a missing parent and asserts explain() raises.
- Risks or migration steps:
  - None; this is a strictness increase aligned with auditability requirements.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `CLAUDE.md:40-43` and `CLAUDE.md:125-127`
- Observed divergence: explain() silently ignores missing audit DB references instead of crashing.
- Reason (if known): Defensive check to avoid None, likely added for convenience.
- Alignment plan or decision needed: Enforce strict failure on missing parent tokens in explain().

## Acceptance Criteria

- explain() raises an error when any token_parents entry references a missing parent token.
- Added test covers the missing-parent case and passes.

## Tests

- Suggested tests to run: `.venv/bin/python -m pytest tests/core/landscape/test_lineage.py -k missing_parent`
- New tests required: yes, missing-parent integrity test

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `CLAUDE.md`
