# Bug Report: Coalesce merge metadata is computed but never recorded

## Summary

- Coalesce builds a rich `coalesce_metadata` payload (policy, branches, timing) but never persists it to the audit trail.
- Node state output only records `merged_into` and does not capture the merged row data or merge details.

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

1. Run a pipeline with a coalesce step.
2. Inspect node state output for the coalesce node and any audit tables for merge metadata.

## Expected Behavior

- Audit trail should include coalesce event details: input token IDs, policy used, wait duration, branches arrived, and merge strategy, along with merged output data.

## Actual Behavior

- `coalesce_metadata` is computed but discarded; output data only includes `merged_into`.

## Evidence

- Coalesce metadata is computed but only returned, not persisted: `src/elspeth/engine/coalesce_executor.py:252`
- Node state output only records `merged_into`: `src/elspeth/engine/coalesce_executor.py:244`
- Audit contract requires merge timing and strategy details: `docs/contracts/plugin-protocol.md#L1137`

## Impact

- User-facing impact: explain/replay cannot show how/when branches merged or which policy fired.
- Data integrity / security impact: missing audit data violates “input/output captured at every transform”.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Coalesce executor returns metadata to the caller, but the caller ignores it and no audit persistence exists.

## Proposed Fix

- Code changes (modules/files):
  - Persist `coalesce_metadata` and merged output data in node state output or a dedicated coalesce audit table.
  - Update processor/orchestrator to store metadata returned in `CoalesceOutcome`.
- Config or schema changes: none (unless a new audit table is needed).
- Tests to add/update:
  - Add a test asserting coalesce metadata is present in audit records.
- Risks or migration steps:
  - Ensure storage size is acceptable; consider storing hashes if payload is large.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md#L1137`, `docs/design/subsystems/00-overview.md#L349`
- Observed divergence: coalesce audit details are not recorded.
- Reason (if known): metadata is computed but never written.
- Alignment plan or decision needed: decide audit storage location for coalesce event details.

## Acceptance Criteria

- Coalesce audit records include policy, branches arrived, wait duration, and merge strategy.
- Merged output data (or its hash) is recorded as the coalesce node output.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_coalesce_executor.py -k audit`
- New tests required: yes (coalesce audit metadata recording)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`, `docs/design/subsystems/00-overview.md`

---

## Verification (2026-01-24)

**Status: STILL VALID**

### Investigation Summary

Verified the bug still exists on commit `36e17f2` (fix/rc1-bug-burndown-session-4):

1. **Metadata is computed but not persisted** - CONFIRMED
   - `CoalesceExecutor._execute_merge()` builds rich `coalesce_metadata` dict (lines 252-265)
   - Metadata includes: policy, merge_strategy, expected_branches, branches_arrived, arrival_order, wait_duration_ms
   - Metadata is returned in `CoalesceOutcome` object but never persisted to database

2. **Two persistence mechanisms available but unused:**

   a. **Node state `context_after` parameter** (exists but not used)
      - `LandscapeRecorder.complete_node_state()` accepts `context_after: dict[str, Any] | None` parameter (recorder.py:1123)
      - Original design doc showed metadata should be stored via `context_after` parameter
      - Current implementation at coalesce_executor.py:244-249 does NOT pass `context_after`
      - Only stores `{"merged_into": merged_token.token_id}` in output_data

   b. **Token outcome `context` parameter** (exists but not used)
      - `LandscapeRecorder.record_token_outcome()` accepts `context: dict[str, Any] | None` parameter (recorder.py:2220)
      - Processor calls `record_token_outcome()` for COALESCED outcome at processor.py:973-978
      - Does NOT pass `coalesce_metadata` from `CoalesceOutcome.coalesce_metadata`

3. **Original design intent was to persist metadata:**
   - Design doc `docs/plans/completed/plugin-refactor/2026-01-18-wp08-coalesce-executor.md:1435` shows:
     ```python
     context_after=coalesce_metadata,  # Store metadata in context_after
     ```
   - This line was in the plan but never implemented in the actual code

4. **Current test coverage:**
   - Tests verify metadata exists in `CoalesceOutcome` object (test_coalesce_executor.py:686-695)
   - Tests verify node_states exist for consumed tokens (test_coalesce_integration.py:609-614)
   - NO tests verify metadata is persisted to `node_states.context_after_json` or `token_outcomes.context_json`

5. **Git history check:**
   - No commits since RC-1 (2026-01-22) have addressed this bug
   - No branches reference this bug ticket ID
   - Bug report date matches RC-1 commit (c786410)

### Impact Verification

- **Audit trail gap confirmed:** Cannot reconstruct coalesce merge decisions from database
- **explain() feature limitation:** Cannot show which policy triggered merge or timing details
- **Data loss:** Rich metadata (policy, branches_arrived, arrival_order, wait_duration_ms) is computed then discarded

### Fix Locations Identified

**Option 1: Store in node_states (recommended)**
- File: `src/elspeth/engine/coalesce_executor.py`
- Line: 244-249 (complete_node_state call)
- Change: Add `context_after=coalesce_metadata` parameter

**Option 2: Store in token_outcomes**
- File: `src/elspeth/engine/processor.py`
- Line: 973-978 (record_token_outcome call)
- Change: Add `context=coalesce_outcome.coalesce_metadata` parameter

**Option 3: Both** (belt and suspenders for auditability)
- Implement both Option 1 and Option 2
- Node states capture per-token consumption details
- Token outcome captures merge event metadata

### Verification Conclusion

Bug is **STILL VALID** and straightforward to fix. The infrastructure exists (both parameters are already defined), implementation simply needs to pass the metadata that's already being computed.

---

## VERIFICATION: 2026-01-25

**Status:** STILL VALID

**Verified By:** Claude Code P2 verification wave 2

**Current Code Analysis:**

Verified on commit `0e2f6da` (fix/rc1-bug-burndown-session-4):

1. **Bug still present in coalesce_executor.py (lines 244-265):**
   - `complete_node_state()` called on line 244-249 WITHOUT `context_after` parameter
   - `coalesce_metadata` built on lines 252-265 (AFTER node state completion)
   - Metadata includes: policy, merge_strategy, expected_branches, branches_arrived, arrival_order, wait_duration_ms
   - Metadata is returned in `CoalesceOutcome` but never persisted

2. **Bug still present in processor.py (lines 973-978):**
   - `record_token_outcome()` called on line 973-978 WITHOUT `context` parameter
   - `coalesce_outcome.coalesce_metadata` is available but not passed to recorder
   - Only stores `join_group_id`, missing all merge details

3. **Infrastructure verified (recorder.py):**
   - `complete_node_state()` accepts `context_after: dict[str, Any] | None` parameter (line 1088, 1100, 1112, 1123)
   - `record_token_outcome()` accepts `context: dict[str, Any] | None` parameter (line 2220)
   - Both parameters exist and are ready to receive metadata

**Git History:**

- Commit `26f8eb1` (2026-01-18): "feat(coalesce): record audit metadata for coalesce events"

---

## Verification (2026-02-01)

**Status: FIXED**

- Coalesce metadata is now recorded via `context_after` on the pending node state completion. (`src/elspeth/engine/coalesce_executor.py:374-405`)

## Closure Report (2026-02-01)

**Status:** CLOSED (FIXED)

### Closure Notes

- Coalesce metadata is persisted in node state `context_after`, satisfying the audit trail requirement for merge context.
  - This commit CREATED the metadata computation but did NOT persist it
  - Added `coalesce_metadata` dict to `CoalesceOutcome` object
  - Metadata is built and returned but never written to database
  - This is the ROOT CAUSE: partial implementation that stopped before persistence

- Commit `0a9cf2a` (2026-01-24): "fix(audit): record COMPLETED outcomes with sink_name"
  - Fixed different bug (COMPLETED outcomes missing sink_name)
  - Modified processor.py and orchestrator.py but did NOT address coalesce metadata
  - Not related to this bug

- Commit `935ee6b` (2026-01-24): "cleanup: delete ExecutionGraph.from_config()"
  - Test cleanup, no impact on coalesce metadata bug

- No commits since 2026-01-24 have addressed coalesce metadata persistence

**Root Cause Confirmed:**

YES - bug is still present exactly as described in original report and 2026-01-24 verification:

1. Metadata computation exists (lines 252-265 in coalesce_executor.py)
2. Infrastructure exists (context_after and context parameters in recorder)
3. Implementation gap: metadata is computed but not passed to persistence layer
4. Architectural flaw: metadata is built AFTER `complete_node_state()` call, making it impossible to pass via `context_after` without reordering code

**Recommendation:**

**Keep open** - bug remains valid and unfixed.

**Fix complexity:** LOW - requires two simple changes:
1. Move `coalesce_metadata` computation BEFORE `complete_node_state()` call in coalesce_executor.py
2. Add `context_after=coalesce_metadata` parameter to `complete_node_state()` call (line 244)
3. Add `context=coalesce_outcome.coalesce_metadata` parameter to `record_token_outcome()` call in processor.py (line 973)

**Priority justification:** P2 is appropriate - this is an audit trail gap but not a functional failure. The system works correctly, but explain() cannot show merge details.
