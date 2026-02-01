# Bug Report: Coalesce never records COALESCED outcomes for parent tokens

## Summary

- Coalesce merges branch tokens but never records a terminal `RowOutcome.COALESCED` for the consumed parent tokens; only the merged token gets an outcome record.
- This violates the audit contract that every token reaches exactly one terminal state and that child tokens are marked COALESCED.

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
- Notable tool calls or steps: code inspection of coalesce executor and processor

## Steps To Reproduce

1. Configure a pipeline with a fork followed by a coalesce (any policy).
2. Run a row through both branches so the coalesce merges.
3. Inspect `token_outcomes` for the branch tokens.

## Expected Behavior

- Each branch token should be recorded with terminal outcome `COALESCED` (and join group metadata) when merged.

## Actual Behavior

- Branch tokens have no terminal outcome recorded; only the merged token is recorded as `COALESCED`.

## Evidence

- Coalesce merge only records node states, not token outcomes, for consumed tokens: `src/elspeth/engine/coalesce_executor.py:236`
- Processor records `COALESCED` outcome for the merged token instead: `src/elspeth/engine/processor.py:983`
- Contract requires child tokens to be marked `COALESCED`: `docs/contracts/plugin-protocol.md#L1111`

## Impact

- User-facing impact: explain/replay shows branch tokens with missing terminal state; audit trail is incomplete.
- Data integrity / security impact: violates AUD-001 terminal state guarantee for every token.
- Performance or cost impact: none.

## Root Cause Hypothesis

- Coalesce implementation focuses on node states and merged token creation, but does not emit `record_token_outcome()` for parent tokens.

## Proposed Fix

- Code changes (modules/files):
  - In `CoalesceExecutor._execute_merge()`, record `RowOutcome.COALESCED` for each consumed token.
  - Expose or retrieve the `join_group_id` created by `LandscapeRecorder.coalesce_tokens()` so it can be stored with the outcome.
  - Consider whether the merged token should have a different outcome (e.g., `COMPLETED` or `ROUTED`) to avoid double-terminal labeling.
- Config or schema changes: none.
- Tests to add/update:
  - Add a coalesce integration test asserting `token_outcomes` contains COALESCED for each branch token.
- Risks or migration steps:
  - Ensure outcome uniqueness constraints are respected (terminal outcomes are unique per token).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/contracts/plugin-protocol.md#L1111`
- Observed divergence: consumed child tokens are not marked as COALESCED.
- Reason (if known): outcome recording exists in processor for merged token, but not for parents.
- Alignment plan or decision needed: decide correct outcome for merged token vs parents and update recording logic.

## Acceptance Criteria

- After a merge, every consumed branch token has a COALESCED outcome recorded with join group metadata.
- Merged token continues and receives the appropriate terminal outcome later in the pipeline.

## Tests

- Suggested tests to run:
  - `pytest tests/engine/test_processor_outcomes.py -k coalesce`
- New tests required: yes (coalesce parent outcome recording)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/contracts/plugin-protocol.md`

---

## Verification (2026-01-24)

**Verified by:** Claude Sonnet 4.5
**Current commit:** `36e17f2` (fix/rc1-bug-burndown-session-4)
**Status:** **STILL VALID**

### Verification Process

1. **Code Review** - Examined current state of outcome recording in coalesce flow:
   - `/home/john/elspeth-rapid/src/elspeth/engine/coalesce_executor.py` (lines 236-249)
   - `/home/john/elspeth-rapid/src/elspeth/engine/processor.py` (lines 973-978)

2. **Git History** - Searched for fixes since bug report date (2026-01-22):
   - Checked commits mentioning "coalesce", "outcome", "COALESCED", "parent", "consumed"
   - Found outcome recording improvements in commit `e93e56c` (2026-01-21, before bug report)
   - Found sink_name fix in commit `0a9cf2a` (2026-01-24) - does NOT address this bug

3. **Test Coverage** - Examined test suite:
   - `/home/john/elspeth-rapid/tests/engine/test_coalesce_integration.py` - verifies node states, NOT token outcomes
   - No tests found that verify consumed/parent tokens receive COALESCED outcomes

### Current Behavior Confirmed

**In `CoalesceExecutor._execute_merge()` (lines 236-249):**
```python
# Record node states for consumed tokens
for token in consumed_tokens:
    state = self._recorder.begin_node_state(
        token_id=token.token_id,
        node_id=node_id,
        step_index=step_in_pipeline,
        input_data=token.row_data,
    )
    self._recorder.complete_node_state(
        state_id=state.state_id,
        status="completed",
        output_data={"merged_into": merged_token.token_id},
        duration_ms=0,
    )
```

**Observation:** Node states are recorded, but NO call to `record_token_outcome()` for consumed tokens.

**In `RowProcessor.process()` (lines 973-978):**
```python
join_group_id = f"{coalesce_name}_{uuid.uuid4().hex[:8]}"
self._recorder.record_token_outcome(
    run_id=self._run_id,
    token_id=coalesce_outcome.merged_token.token_id,  # ONLY merged token
    outcome=RowOutcome.COALESCED,
    join_group_id=join_group_id,
)
```

**Observation:** Only the merged token receives a COALESCED outcome, not the consumed parent/branch tokens.

### Contract Violation Confirmed

From `/home/john/elspeth-rapid/docs/contracts/plugin-protocol.md` (line 1113):

> 3. **Child tokens marked with terminal state `COALESCED`**

From `/home/john/elspeth-rapid/src/elspeth/contracts/enums.py` (line 154):

> - COALESCED: Merged in join from parallel paths

**Interpretation Issue Identified:** The contract and enum documentation are ambiguous about WHO gets marked COALESCED:
- Bug report interprets: consumed branch tokens (children) should be marked COALESCED
- Current code implements: merged token (parent) is marked COALESCED

### Additional Evidence

**`CoalesceOutcome` dataclass includes consumed tokens:**
```python
consumed_tokens: list[TokenInfo] = field(default_factory=list)  # Line 34
```

This field is populated (line 227) but **never consumed** by the processor. No code path accesses `coalesce_outcome.consumed_tokens`.

### Conclusion

**BUG STATUS: STILL VALID**

The bug remains unfixed. Consumed branch tokens receive:
- ✅ Node state records (lines 236-249 in coalesce_executor.py)
- ❌ No token outcome records

This violates the AUD-001 requirement that every token reaches exactly one terminal state. Branch tokens disappear from the audit trail without terminal outcomes.

### Recommended Action

1. **Add outcome recording in `CoalesceExecutor._execute_merge()`:**
   - Loop through `consumed_tokens`
   - Call `self._recorder.record_token_outcome()` with `RowOutcome.COALESCED` for each
   - Use same `join_group_id` as merged token for lineage

2. **Clarify contract documentation:**
   - Update plugin-protocol.md line 1113 to explicitly state which tokens receive COALESCED outcome
   - Consider: should BOTH consumed and merged tokens be marked COALESCED, or only consumed?

3. **Add integration test:**
   - Extend `test_coalesce_integration.py` to verify token_outcomes table
   - Assert all consumed tokens have COALESCED outcome with join_group_id
