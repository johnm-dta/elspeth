# Bug Report: Duplicate branch arrivals overwrite earlier tokens without error

## Summary

- When a second token arrives for the same `(row_id, branch_name)`, the executor overwrites the first token in `pending.arrived` without error.
- This silently drops data and can merge the wrong token if duplicates occur due to retries or bugs.

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

1. Configure a coalesce with branches `A` and `B`.
2. Cause branch `A` to emit two tokens for the same `row_id` (e.g., via retry bug or duplicate processing).
3. Observe that only the last token is used in the merge.

## Expected Behavior

- Duplicate arrivals for the same branch should raise an error or be explicitly rejected with a recorded failure.

## Actual Behavior

- The later token silently overwrites the earlier one, losing data and masking upstream bugs.

## Evidence

- Arrival overwrites per-branch token without validation: `src/elspeth/engine/coalesce_executor.py:172`

## Impact

- User-facing impact: merged output can be inconsistent or derived from the wrong token.
- Data integrity / security impact: silent data loss and audit gaps.
- Performance or cost impact: none directly, but debugging is harder.

## Root Cause Hypothesis

- `pending.arrived` is treated as a simple map with no duplicate detection, so later arrivals replace earlier ones.

## Proposed Fix

- Code changes (modules/files):
  - Detect duplicate arrivals for the same branch and raise a hard error (bug in engine) or record a failure outcome.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that duplicate branch arrivals raise or are recorded as failures.
- Risks or migration steps:
  - If duplicates can occur legitimately, define explicit de-duplication semantics and document them.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md#L1109`
- Observed divergence: duplicate branch arrivals are silently overwritten.
- Reason (if known): no explicit duplicate handling.
- Alignment plan or decision needed: enforce one token per branch per row_id.

## Acceptance Criteria

- Duplicate arrivals for the same branch are detected and handled deterministically (error or explicit failure record).
- The merge uses exactly one token per branch per row_id.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_coalesce_executor.py -k duplicate`
- New tests required: yes (duplicate branch arrival detection)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 2

**Current Code Analysis:**

Examined `/home/john/elspeth-rapid/src/elspeth/engine/coalesce_executor.py` at the reported location (lines 172-174):

```python
# Record arrival
pending.arrived[token.branch_name] = token
pending.arrival_times[token.branch_name] = now
```

The bug is confirmed. The code performs a simple dictionary assignment with no validation:

1. **No duplicate detection**: The code uses `pending.arrived[token.branch_name] = token` which silently overwrites any existing token for that branch
2. **No error raised**: There is no check like `if token.branch_name in pending.arrived: raise ValueError(...)`
3. **Silent data loss**: If two tokens arrive for the same `(row_id, branch_name)` pair, the first token is completely lost from the pending state
4. **Audit trail gap**: The overwritten token's arrival is not recorded anywhere - it disappears without trace

**Test Coverage:**

Examined `/home/john/elspeth-rapid/tests/engine/test_coalesce_executor.py` - confirmed there are NO tests for duplicate branch arrival scenarios. The test file contains 1182 lines but no test with "duplicate" in the name or testing this edge case.

**Git History:**

- File was created in commit `c786410` (ELSPETH - Release Candidate 1) on 2026-01-22
- No subsequent changes to the `accept()` method or duplicate handling logic since bug was reported
- The code structure remains identical to the reported commit `ae2c0e6`

**Root Cause Confirmed:**

YES. The `_PendingCoalesce.arrived` dictionary (line 43) is a simple `dict[str, TokenInfo]` with no safeguards. When the same branch_name is used as a key twice, Python's dict semantics silently overwrite the previous value.

This is particularly problematic because:
- Upstream retries could send the same branch token twice
- Engine bugs could route tokens incorrectly
- The silent overwrite masks these bugs instead of surfacing them
- Per ELSPETH's audit integrity principles, "silent wrong result is worse than a crash"

**Related Issues:**

This bug is architecturally related to P1-2026-01-22-coalesce-late-arrivals-duplicate-merge.md, which handles the case where tokens arrive AFTER a merge completes. This bug handles the case where duplicate tokens arrive for the same branch BEFORE the merge completes. Both stem from insufficient state tracking in the coalesce executor.

**Recommendation:**

**Keep open** - This is a valid P2 bug that violates ELSPETH's audit integrity principles. The fix should:

1. Add validation: `if token.branch_name in pending.arrived: raise ValueError(f"Duplicate arrival for branch {token.branch_name} on row {token.row_id}")`
2. Add test coverage for the duplicate branch arrival scenario
3. Document whether duplicates are considered an engine bug (crash) or an expected failure mode (quarantine)

Per CLAUDE.md "Plugin Ownership" section, this should crash immediately since duplicates indicate a bug in the engine's token routing logic, not a problem with user data.

---

## Verification (2026-02-01)

**Status: FIXED**

- Duplicate arrivals are now detected and raise immediately before overwriting. (`src/elspeth/engine/coalesce_executor.py:241-249`)

## Closure Report (2026-02-01)

**Status:** CLOSED (FIXED)

### Closure Notes

- Duplicate branch arrivals now raise a hard error, preventing silent overwrite.
