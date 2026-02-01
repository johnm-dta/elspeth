# Bug Report: Batch Mode Uses Synthetic state_id, Breaking Call Audit FK

## Summary

- Batch mode generates synthetic `state_id` values for call audit records, but these IDs never exist in `node_states` table, causing foreign key violations on audit queries.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Branch Bug Scan (fix/rc1-bug-burndown-session-4)
- Date: 2026-01-25
- Related run/issue ID: BUG-AZURE-02

## Environment

- Commit/branch: fix/rc1-bug-burndown-session-4
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: Azure batch LLM transform

## Agent Context (if relevant)

- Goal or task prompt: Static analysis agent doing a deep bug audit of azure_batch.py
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): read-only sandbox
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Run Azure batch LLM transform.
2. Query `call_audit` table joined with `node_states`.
3. Observe FK constraint violations or missing joins.

## Expected Behavior

- Every `call_audit.state_id` should reference valid `node_states.state_id`.

## Actual Behavior

- Batch calls create synthetic state_ids that don't exist in node_states.
- FK queries fail or return empty results.

## Evidence

- `src/elspeth/plugins/llm/azure_batch.py` - Generates synthetic state_ids
- `node_states` table never receives these synthetic IDs
- Call audit integrity check fails

## Impact

- User-facing impact: Cannot query call audit via FK joins.
- Data integrity / security impact: Audit referential integrity broken.
- Performance or cost impact: Queries fail, forcing manual audit reconstruction.

## Root Cause Hypothesis

- Batch calls don't create real node_states, use synthetic IDs instead.

## Proposed Fix

**Option A: Create real node_states for batch calls**
```python
for call in batch_calls:
    state_id = recorder.record_node_state(
        run_id=run_id,
        token_id=call.token_id,
        node_id=node_id,
        ...
    )
    call.state_id = state_id
```

**Option B: Make state_id nullable in call_audit**
```python
# Schema change
state_id TEXT NULL  # Was: NOT NULL

# Document: Batch calls may not have node states
```

- Config or schema changes: Option B requires Alembic migration.
- Tests to add/update:
  - `test_batch_call_state_ids_valid()` - Verify FK integrity

- Risks or migration steps: Option A preferred (no schema change).

## Architectural Deviations

- Spec or doc reference: Database schema FK constraints
- Observed divergence: FK violations
- Reason (if known): Batch optimization skips node state creation
- Alignment plan or decision needed: Either create real states or make FK nullable

## Acceptance Criteria

- No FK violations in call_audit queries.
- Batch calls either have real state_ids or NULL (documented).

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_batch.py`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/bugs/BRANCH_BUG_TRIAGE_2026-01-25.md`

---

## CLOSURE: 2026-01-29

**Status:** FIXED

**Fixed By:** Claude (with 4-specialist review board)

**Resolution:**

LLM calls now use real state_ids (the batch's `aggregation_state_id`) rather than synthetic ones:

1. `AggregationExecutor` creates ONE node_state per batch at line 1094: `ctx.state_id = state.state_id`
2. When `ctx.record_call()` is invoked, it uses this real `state_id`
3. Multiple LLM calls per batch are distinguished by `call_index` (auto-incremented)
4. No synthetic state_ids are generated - all FK constraints are satisfied
5. The `calls` table's `UniqueConstraint("state_id", "call_index")` supports this pattern

**Architecture Note:**
The key insight is that the batch already has a node_state created by `AggregationExecutor`. We record N calls against that ONE state using different `call_index` values, rather than creating N new states (which would violate uniqueness constraints and audit semantics).

**Option A was chosen** (use real state_ids) rather than Option B (make FK nullable).

**Tests Added:**
- Integration tests verify `state_id` is valid and FK joins work
- `test_llm_calls_visible_in_explain` confirms calls link to real states

**Verified By:** 4-specialist review board (2026-01-29)
