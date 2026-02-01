# Bug Report: Azure Batch Audit Calls Omit JSONL Payloads

## Summary

- Azure batch mode creates JSONL files with full request/response data but never records payload hashes or stores payloads in payload store, breaking audit trail completeness.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: Branch Bug Scan (fix/rc1-bug-burndown-session-4)
- Date: 2026-01-25
- Related run/issue ID: BUG-AZURE-01

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
2. Check call_audit table for batch calls.
3. Query explain() for batch call details.

## Expected Behavior

- Call audit records should have `request_hash` and `response_hash`.
- Payload store should contain full request/response data.
- explain() should be able to retrieve full call details.

## Actual Behavior

- Call audit records created with `state_id` but without payload hashes.
- Payloads never stored in payload store.
- explain() cannot retrieve call details (audit trail incomplete).

## Evidence

- `src/elspeth/plugins/llm/azure_batch.py` - Creates JSONL batch files but doesn't call `payload_store.store()`
- Call audit records reference non-existent payload hashes

## Impact

- User-facing impact: Cannot explain batch LLM decisions via audit trail.
- Data integrity / security impact: Auditability broken for batch operations, violates CLAUDE.md standard.
- Performance or cost impact: Cannot debug batch issues without full audit trail.

## Root Cause Hypothesis

- Batch processing path omits payload storage steps that single-call path includes.

## Proposed Fix

```python
# After creating batch JSONL
for call in batch_calls:
    request_hash = stable_hash(call.request_data)
    response_hash = stable_hash(call.response_data)

    payload_store.store(request_hash, call.request_data)
    payload_store.store(response_hash, call.response_data)

    # Then record call with hashes
    recorder.record_call(
        state_id=call.state_id,
        request_hash=request_hash,
        response_hash=response_hash,
        ...
    )
```

- Config or schema changes: None.
- Tests to add/update:
  - `test_batch_calls_have_payload_hashes()` - Verify hashes recorded
  - `test_batch_call_payloads_retrievable()` - Verify explain() works

- Risks or migration steps: None.

## Architectural Deviations

- Spec or doc reference: `CLAUDE.md` - Auditability Standard
- Observed divergence: Audit trail incomplete for batch operations.
- Reason (if known): Batch path missing payload storage.
- Alignment plan or decision needed: Add payload storage to batch path.

## Acceptance Criteria

- Batch calls have request_hash and response_hash in call_audit.
- Payloads retrievable from payload store.
- explain() works for batch calls.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_batch.py`
- New tests required: yes (2 tests above)

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/bugs/BRANCH_BUG_TRIAGE_2026-01-25.md`

---

## CLOSURE: 2026-01-29

**Status:** FIXED

**Fixed By:** Claude (with 4-specialist review board)

**Resolution:**

The Azure batch transform now records per-row LLM calls against the batch's existing node_state:

1. Original requests stored in checkpoint during submit phase (`requests_by_id`)
2. When results return, each LLM prompt/response is recorded as a call via `ctx.record_call()`:
   - `call_type=CallType.LLM`
   - Unique `call_index` per row (auto-incremented)
   - `custom_id` in request_data for token mapping
   - Full request/response data stored
3. Uses existing `calls` table with `call_index` - no new node_states needed
4. Tier 3 boundary validation added at parse time (Azure API response structure)
5. `explain()` queries now return full LLM interaction details

**Architecture Note:**
Original plan proposed creating N new node_states per batch. This was rejected by review board as it violated audit model semantics (one state per batch, multiple calls against it via `call_index`).

**Tests Added:**
- `test_checkpoint_includes_original_requests`
- `test_download_results_records_llm_calls`
- `test_download_results_records_failed_llm_calls_correctly`
- `test_llm_calls_visible_in_explain` (integration)
- `test_multiple_llm_calls_recorded_per_batch` (integration)
- `test_failed_llm_call_recorded` (integration)

**Verified By:** 4-specialist review board (2026-01-29)
