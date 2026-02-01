# Bug Report: Aggregation output defaults to empty row on contract violation

## Summary

- In aggregation `output_mode` single/transform, the processor substitutes `{}` when `result.row` is missing. This masks plugin contract violations and can emit empty rows instead of crashing, violating the repository’s “no defensive programming” rule.

## Severity

- Severity: minor
- Priority: P3

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 / fix/rc1-bug-burndown-session-2
- OS: Linux
- Python version: Python 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive src/elspeth/engine/processor.py for bugs; create reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Implement a batch-aware transform that incorrectly returns `TransformResult.success_multi(...)` while aggregation is configured as `output_mode: single`, or returns an invalid `TransformResult` with `row=None`.
2. Trigger a batch flush.

## Expected Behavior

- The processor should crash on the contract violation (single-mode expects a single row), not fabricate output.

## Actual Behavior

- The processor substitutes an empty dict and proceeds.

## Evidence

- Single mode fallback: `src/elspeth/engine/processor.py:224-227` uses `{}` if `result.row` is None.
- Transform mode fallback: `src/elspeth/engine/processor.py:323-325` uses `{}` if `result.row` is None.
- CLAUDE.md prohibits defensive patterns that mask plugin bugs.

## Impact

- User-facing impact: Silent emission of empty rows.
- Data integrity / security impact: Audit trail contains fabricated output.
- Performance or cost impact: Downstream steps process meaningless data.

## Root Cause Hypothesis

- Defensive defaults in aggregation output handling mask invalid TransformResult payloads.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/engine/processor.py`
- Config or schema changes: None
- Tests to add/update: Add tests that assert contract violations raise errors.
- Risks or migration steps: None; aligns with crash-on-bug policy.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): CLAUDE.md “no defensive programming”.
- Observed divergence: Processor substitutes `{}` instead of raising.
- Reason (if known): Convenience fallback added during aggregation implementation.
- Alignment plan or decision needed: Decide whether to enforce strict output-mode contracts.

## Acceptance Criteria

- Aggregation output-mode violations raise errors instead of emitting empty rows.

## Tests

- Suggested tests to run: `pytest tests/engine/test_processor.py -k aggregation_contract`
- New tests required: Yes (contract violation handling).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md

## Verification (2026-01-25)

**Status: STILL VALID**

### Current State

The defensive patterns identified in the original bug report are still present in the codebase:

1. **Line 229** (single mode): `final_data = result.row if result.row is not None else {}`
2. **Line 319** (transform mode): `output_rows = [result.row] if result.row is not None else [{}]`

### Evidence

Both patterns were introduced in commit `c786410` (ELSPETH - Release Candidate 1, 2026-01-22) and have persisted through the current codebase state.

**Git blame verification:**
```
c786410 (John Morrissey 2026-01-22 12:22:17 +1100 229) final_data = result.row if result.row is not None else {}
c786410 (John Morrissey 2026-01-22 12:22:17 +1100 319) output_rows = [result.row] if result.row is not None else [{}]
```

### Policy Violation Analysis

This pattern directly violates **CLAUDE.md § Plugin Ownership: System Code, Not User Code**:

- **Line 166**: "Plugin returns wrong type → **CRASH** - bug in our code"
- **Line 174**: "A defective plugin that silently produces wrong results is **worse than a crash**"

And **CLAUDE.md § PROHIBITION ON "DEFENSIVE PROGRAMMING" PATTERNS**:

- **Line 494**: "If code would fail without a defensive pattern, that failure is a bug to fix, not a symptom to suppress."

### Impact Assessment

If a batch-aware transform plugin returns `TransformResult.success(row=None)`:
- **Current behavior**: Processor silently substitutes `{}` and continues, creating audit trail entries with empty row data
- **Expected behavior**: Processor should crash immediately with a clear error message indicating plugin contract violation

This is particularly problematic because:
1. Plugins are system-owned code (CLAUDE.md §129-204), not user extensions
2. Empty row fabrication violates audit integrity - the audit trail records something that never happened
3. Silent failures hide plugin bugs that should be fixed in the codebase

### Test Coverage Gap

No tests exist that verify contract violations raise errors:
- Search for test files containing "aggregation.*contract" or "contract.*violation": **No results**
- Search for tests verifying empty row defensive behavior: **No results**

The bug report correctly identified this gap in "Tests to add/update: Add tests that assert contract violations raise errors."

### Recommendation

This bug should remain **OPEN** and be prioritized for fixing:

1. **Remove defensive patterns** at lines 229 and 319
2. **Add explicit validation** that crashes when `result.row is None` in single/transform modes
3. **Add test coverage** for plugin contract violations (as suggested in original report)
4. **Error message** should clearly indicate this is a plugin bug, not a data issue

The fix aligns with the codebase's core principle: "Plugin bugs are system bugs - they get fixed in the codebase" (CLAUDE.md §201).

---

## Resolution

**Fixed in:** 2026-01-28
**Fixed by:** Claude Code (Opus 4.5)

**Fix:** Replaced all 5 defensive `{}` substitution patterns with contract assertions that raise RuntimeError:

**Code changes:**
- `src/elspeth/engine/processor.py`:
  - Line 417-423 (single mode, `_process_aggregation_node`): Now raises RuntimeError if `result.row is None`
  - Line 525-532 (transform mode, `_process_aggregation_node`): Now raises RuntimeError if `result.row is None`
  - Line 708-715 (single mode, `_process_batch_aggregation_node`): Now raises RuntimeError if `result.row is None`
  - Line 846-853 (transform mode, `_process_batch_aggregation_node`): Now raises RuntimeError if `result.row is None`
- `src/elspeth/engine/executors.py`:
  - Line 1165-1177 (`execute_flush`): Replaced `assert result.rows is not None` with proper RuntimeError message

**Error message format:**
```
Aggregation transform '{transform.name}' returned None for result.row in '{output_mode}' mode.
Batch-aware transforms must return a row via TransformResult.success(row). This is a plugin bug.
```

**Tests added:**
- `tests/engine/test_processor_batch.py`: Added `test_aggregation_transform_returns_none_raises_contract_error`

**Commits:**
- fix(processor): replace defensive {} with contract assertions for aggregation output (P3-2026-01-28)
