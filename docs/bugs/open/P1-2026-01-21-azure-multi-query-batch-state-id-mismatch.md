# Bug Report: Azure multi-query batch uses synthetic state_id not tied to node_state

## Summary

- Batch processing appends "_row{i}" to ctx.state_id and records external calls under those synthetic IDs, but only the original ctx.state_id exists in node_states. This violates the calls -> node_states FK and breaks audit traceability for batch runs.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 (fix/rc1-bug-burndown-session-2)
- OS: unknown
- Python version: unknown
- Config profile / env vars: N/A
- Data set or fixture: any batch run using azure_multi_query_llm

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/llm for bugs
- Model/version: Codex (GPT-5)
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code review only

## Steps To Reproduce

1. Configure an aggregation node that flushes to `azure_multi_query_llm` with landscape enabled.
2. Run a batch that triggers aggregation flush.
3. Observe external call recording or DB insert behavior for LLM calls.

## Expected Behavior

- All external LLM calls are recorded under the batch node's actual `state_id` so they link to a real `node_states` row.

## Actual Behavior

- LLM calls are recorded under synthetic IDs like `<state_id>_row0`, which do not exist in `node_states`, causing FK violations or orphaned call records.

## Evidence

- Per-row state IDs are constructed in `src/elspeth/plugins/llm/azure_multi_query.py:492`.
- External calls require a valid `node_states.state_id` per FK in `src/elspeth/core/landscape/schema.py:191`.

## Impact

- User-facing impact: batch runs can crash when recording calls.
- Data integrity / security impact: audit trail loses linkage between calls and node state.
- Performance or cost impact: failed batches may require reruns.

## Root Cause Hypothesis

- Batch path uses synthetic state IDs for per-row isolation without creating matching node_states entries.

## Proposed Fix

- Code changes (modules/files):
  - Use `ctx.state_id` for all per-row LLM calls in batch mode, or create per-row node_states before calling the LLM and use those IDs.
- Config or schema changes: N/A
- Tests to add/update:
  - Add a batch-mode test that asserts calls are recorded against an existing state_id.
- Risks or migration steps:
  - If switching to shared state_id, verify call_index uniqueness and audit ordering.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): CLAUDE.md "External calls - Full request AND response recorded" and calls FK to node_states.
- Observed divergence: external calls recorded under nonexistent state_id.
- Reason (if known): attempt to isolate per-row calls in batch without audit model support.
- Alignment plan or decision needed: confirm whether batch uses shared state_id or supports per-row node_states.

## Acceptance Criteria

- Batch-mode multi-query runs record all calls under valid node_states IDs with no FK errors.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_azure_multi_query.py -k batch -v`
- New tests required: yes, audit linkage test for calls -> node_states.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: CLAUDE.md auditability standard
