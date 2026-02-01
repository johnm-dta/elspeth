# Bug Report: Select merge strategy falls back to wrong branch when select_branch is missing

## Summary

- For `merge: select`, if the selected branch has not arrived, the executor silently falls back to the first arrived branch instead of failing or waiting for the selected branch.
- This violates the contract that `select` takes output from a specific branch only.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-22
- Related run/issue ID: N/A

## Environment

- Commit/branch: `ae2c0e6f088f467276582fa8016f91b4d3bb26c7` (fix/rc1-bug-burndown-session-2)
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into coalesce_executor, identify bugs, create bug docs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of coalesce executor

## Steps To Reproduce

1. Configure coalesce with `merge: select` and `select_branch: preferred`.
2. Use a policy that allows partial arrival (e.g., `best_effort` timeout) and ensure `preferred` never arrives.
3. Observe the merged output.

## Expected Behavior

- If `select_branch` is missing, the merge should fail or wait (per policy), not substitute another branch.

## Actual Behavior

- The executor silently returns the first arrived branch output.

## Evidence

- Fallback to first arrival when select branch missing: `src/elspeth/engine/coalesce_executor.py:297`
- `select` is defined as “take output from specific branch only”: `docs/contracts/plugin-protocol.md#L1105`

## Impact

- User-facing impact: merged outputs can contain data from the wrong branch.
- Data integrity / security impact: audit trail cannot prove the intended branch was used.
- Performance or cost impact: none.

## Root Cause Hypothesis

- A convenience fallback was added to avoid missing-branch errors, but it violates the `select` contract.

## Proposed Fix

- Code changes (modules/files):
  - Remove the fallback and treat missing `select_branch` as a failure (or hold until arrival if policy permits).
  - Record failure or timeout appropriately when `select_branch` never arrives.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that `merge: select` fails when `select_branch` is missing.
- Risks or migration steps:
  - Existing pipelines relying on fallback (if any) will fail fast; document behavior change.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md#L1105`
- Observed divergence: select branch can be replaced by another branch.
- Reason (if known): fallback implemented for convenience.
- Alignment plan or decision needed: enforce strict select semantics.

## Acceptance Criteria

- Coalesce with `merge: select` never emits data from a non-selected branch.
- Missing selected branch results in a recorded failure or timeout handling.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_coalesce_executor.py -k select`
- New tests required: yes (select branch missing)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 2

**Current Code Analysis:**

The bug is **still present** in the current codebase. The problematic code is at `/home/john/elspeth-rapid/src/elspeth/engine/coalesce_executor.py:295-301`:

```python
# settings.merge == "select":
# Take specific branch output
assert settings.select_branch is not None
if settings.select_branch in arrived:
    return arrived[settings.select_branch].row_data.copy()
# Fallback to first arrived if select branch not present
return next(iter(arrived.values())).row_data.copy()
```

The fallback on line 301 violates the contract for `merge: select`, which per `docs/contracts/plugin-protocol.md:1105` is defined as "Take output from specific branch only". When the selected branch has not arrived (even though the configuration validator confirms `select_branch` is one of the expected branches), the code silently substitutes the first-arrived branch.

**Git History:**

- Examined all commits to `coalesce_executor.py` since the bug report date (2026-01-22)
- The file has not been modified since RC1 commit `c786410` (2026-01-22)
- The fallback logic was present in the original implementation and has never been changed
- No commits addressed this issue

**Root Cause Confirmed:**

Yes, the root cause is exactly as described:
1. The `_merge_data()` method implements a "convenience fallback" when `select_branch` is not in the `arrived` dict
2. This can occur when:
   - Policy is `best_effort` with timeout - selected branch never arrives, timeout fires, merge happens with whatever did arrive
   - Policy is `quorum` - selected branch is missing but quorum is met with other branches
   - Policy is `first` - a non-selected branch arrives first and triggers immediate merge
3. The existing test at `tests/engine/test_coalesce_executor.py:273-314` actually **relies on this bug**:
   - Configured with `merge="select"` and `select_branch="fast"`
   - Uses `policy="first"`
   - Test sends the "slow" branch first
   - Expects to get data from "slow" branch (line 314: `assert outcome.merged_token.row_data == {"result": "from_slow"}`)
   - This test documents that the fallback behavior is intentional, but it violates the contract

**Recommendation:**

**Keep open** - This is a valid bug with contract-violating behavior. The fix requires:

1. **Code change:** Remove the fallback in `_merge_data()` - raise an error or return a failure outcome when `select_branch not in arrived`
2. **Policy interaction decision:** Determine how to handle missing select branch for each policy:
   - `require_all`: Natural - if select branch missing, not all branches arrived, so merge doesn't happen
   - `quorum`: Should fail if select branch missing, even if quorum met
   - `best_effort`: Should record failure if select branch never arrived
   - `first`: Contradictory semantics - "first" suggests any branch, "select" suggests specific branch
3. **Test updates:** The test at line 273 needs to be updated or removed - it validates broken behavior
4. **Audit trail:** When merge fails due to missing select branch, record appropriate failure reason in coalesce metadata

---

## Verification (2026-02-01)

**Status: FIXED**

- `_merge_data()` now raises if `select_branch` is missing; no fallback exists. (`src/elspeth/engine/coalesce_executor.py:452-467`)

## Closure Report (2026-02-01)

**Status:** CLOSED (FIXED)

### Closure Notes

- Select-merge no longer falls back to an arbitrary branch; missing selected branch now raises.
