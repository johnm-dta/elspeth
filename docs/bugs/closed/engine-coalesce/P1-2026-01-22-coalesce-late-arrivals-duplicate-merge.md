# Bug Report: Coalesce allows late arrivals to start a second merge (duplicate outputs)

## Summary

- After a merge completes, the executor deletes pending state but never marks the `(coalesce_name, row_id)` as closed, so late-arriving branch tokens create a new pending entry and can trigger a second merge.
- The `first` policy effectively merges every branch arrival because it never discards later branches.

## Severity

- Severity: major
- Priority: P1

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

1. Configure a coalesce with `policy: first` and branches `A`, `B`.
2. Send branch `A` token first; coalesce merges immediately.
3. Send branch `B` token later for the same `row_id`.

## Expected Behavior

- After the first merge, later arrivals for the same `(coalesce_name, row_id)` should be discarded or explicitly recorded as late/ignored, not merged again.

## Actual Behavior

- A new pending entry is created and a second merge can occur, producing duplicate outputs for the same source row.

## Evidence

- Pending state is created whenever the key is missing: `src/elspeth/engine/coalesce_executor.py:159`
- Pending state is deleted after merge, with no closed-set tracking: `src/elspeth/engine/coalesce_executor.py:267`
- `first` policy merges on any single arrival: `src/elspeth/engine/coalesce_executor.py:201`
- Policy contract requires `first` to discard later arrivals: `docs/contracts/plugin-protocol.md#L1097`

## Impact

- User-facing impact: duplicate outputs and inconsistent downstream results for a single input row.
- Data integrity / security impact: audit trail can show multiple merged tokens for the same branch group.
- Performance or cost impact: extra merges and sink writes.

## Root Cause Hypothesis

- Coalesce executor tracks only pending merges and does not track completed merges by key.

## Proposed Fix

- Code changes (modules/files):
  - Track completed `(coalesce_name, row_id)` keys and reject or quarantine late arrivals.
  - For `first` policy, explicitly discard later arrivals after the first merge.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that sends a late arrival after a merge and asserts it is ignored or flagged.
- Risks or migration steps:
  - Define behavior for late arrivals (drop, error, or quarantine) and document it.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md#L1097`
- Observed divergence: late arrivals are merged again instead of discarded.
- Reason (if known): no completed-set tracking.
- Alignment plan or decision needed: define and enforce late-arrival handling.

## Acceptance Criteria

- A `(coalesce_name, row_id)` can only be merged once.
- Late arrivals do not create additional merged tokens.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_coalesce_executor.py -k first`
- New tests required: yes (late arrival suppression)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`

---

## Verification (2026-01-24)

**Status: STILL VALID**

### Code Inspection

Examined `/home/john/elspeth-rapid/src/elspeth/engine/coalesce_executor.py`:

1. **No closed-set tracking exists** (lines 91-96):
   - Only tracks pending coalesces: `self._pending: dict[tuple[str, str], _PendingCoalesce] = {}`
   - No data structure for completed merges

2. **Late arrivals create new pending entries** (lines 163-168):
   ```python
   if key not in self._pending:
       self._pending[key] = _PendingCoalesce(
           arrived={},
           arrival_times={},
           first_arrival=now,
       )
   ```
   After merge completion, pending state is deleted (line 268), so the key is no longer present. A late arrival for the same `(coalesce_name, row_id)` will pass the `if key not in self._pending` check and create a fresh pending entry.

3. **First policy merges on single arrival** (lines 201-202):
   ```python
   elif settings.policy == "first":
       return arrived_count >= 1
   ```
   Any arrival triggers merge, so both the first arrival AND any late arrival will each trigger a separate merge.

4. **Pending state deleted after merge** (line 268):
   ```python
   del self._pending[key]
   ```
   No completed-merge tracking replaces it.

### Test Evidence

Found existing test documenting this behavior in `/home/john/elspeth-rapid/tests/engine/test_processor.py` (commit `7bd254a`, 2026-01-18):

```python
# === Late arrival behavior ===
# The slow branch arrives after merge is complete.
# Since pending state was deleted, this creates a NEW pending entry.
# This is by design - the row processing would have already continued
# with the merged token, so this late arrival is effectively orphaned.
slow_token = TokenInfo(...)
outcome3 = coalesce_executor.accept(slow_token, "merger", step_in_pipeline=3)

# Late arrival creates new pending state (waiting for more branches)
# This is the expected behavior - in real pipelines, the orchestrator
# would track that this row already coalesced and not submit the late token.
assert outcome3.held is True
assert outcome3.merged_token is None
```

**Problem:** The test comment claims "the orchestrator would track that this row already coalesced" but:
- No such tracking exists in the orchestrator or processor
- The comment says late arrival is "held" (waiting for more branches), which means if another late arrival comes, it could trigger a second merge
- For `first` policy specifically, the first late arrival would trigger an immediate second merge

### Contract Violation

`/home/john/elspeth-rapid/docs/contracts/plugin-protocol.md:1097`:
```
| first | Take first arrival, discard others |
```

**Violation:** "discard others" is not implemented. Later arrivals create new pending entries and can trigger additional merges.

### Git History

No fixes found since bug report date (2026-01-22):
- `git log --all --since="2026-01-22" -- src/elspeth/engine/coalesce_executor.py` returned no commits
- No commits mention "late arrival" suppression or closed-set tracking
- Most recent coalesce_executor commit: `c786410` (Release Candidate 1)

### Impact Confirmation

**Data corruption scenarios:**

1. **First policy:**
   - First branch arrives → merge 1 created → sent to sink
   - Second branch arrives late → merge 2 created → sent to sink
   - Result: Same source row produces two outputs

2. **Quorum policy:**
   - Branches A, B arrive → quorum met → merge 1 created
   - Branch C arrives late → creates new pending entry (held)
   - If pipeline restarts and branches A, B arrive again → merge 2 created
   - Result: Duplicate outputs

3. **Best-effort with timeout:**
   - Branch A arrives → timeout expires → merge 1 created
   - Branch B arrives late → creates new pending entry (held)
   - At flush_pending: merge 2 created (line 400-408)
   - Result: Duplicate outputs

### Conclusion

Bug is **STILL VALID** and **HIGH SEVERITY**:
- Code structure unchanged since bug report
- No closed-set tracking implemented
- Contract explicitly requires "discard others" for first policy
- Test acknowledges behavior but incorrectly claims orchestrator handles it
- Multiple merge scenarios can produce duplicate outputs for single source row
