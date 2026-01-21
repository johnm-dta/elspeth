# Bug Report: Azure content safety and prompt shield use a global call_index, not per state_id

## Summary

- AzureContentSafety and AzurePromptShield record external calls using a single instance-wide call_index counter, so call_index does not reset per state_id, breaking the intended "index within state" semantics and call replay by index.

## Severity

- Severity: critical
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: not checked
- OS: not checked (workspace sandbox)
- Python version: not checked
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into src/elspeth/plugins/transforms for bugs
- Model/version: GPT-5 Codex
- Tooling and permissions (sandbox/approvals): workspace-write sandbox, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: reviewed Azure content_safety and prompt_shield implementations

## Steps To Reproduce

1. Run a pipeline with AzureContentSafety or AzurePromptShield on two rows.
2. Inspect external_calls records for each row's state_id.
3. Observe that the first call for the second row has call_index > 0 (continuing from the prior row).

## Expected Behavior

- call_index should be a 0-based index within each state_id (per-state ordering and replay by index).

## Actual Behavior

- call_index increments globally across rows/states, so per-state call indices do not start at 0 and may have gaps.

## Evidence

- AzureContentSafety global counter: src/elspeth/plugins/transforms/azure/content_safety.py:177-180, 456-489
- AzurePromptShield global counter: src/elspeth/plugins/transforms/azure/prompt_shield.py:148-150, 425-460
- Spec: call_index is "Index of call within state": docs/plans/completed/2026-01-12-phase6-external-calls.md:956-966
- Ordering invariant uses (state_id, call_index): docs/design/architecture.md:271-275

## Impact

- User-facing impact: call replay by index can fail (call_index=0 may not exist for a state).
- Data integrity / security impact: audit ordering semantics are violated for external calls.
- Performance or cost impact: none directly, but audit debugging becomes unreliable.

## Root Cause Hypothesis

- call_index is tracked as a single instance counter instead of per state_id or using PluginContext.record_call.

## Proposed Fix

- Code changes (modules/files): src/elspeth/plugins/transforms/azure/content_safety.py, src/elspeth/plugins/transforms/azure/prompt_shield.py
- Config or schema changes: N/A
- Tests to add/update: add tests to ensure call_index resets per state_id.
- Risks or migration steps: existing runs already stored; fix applies to new runs.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): docs/plans/completed/2026-01-12-phase6-external-calls.md:956-966
- Observed divergence: call_index does not reflect per-state ordering.
- Reason (if known): global counter used for thread-safety but not keyed by state_id.
- Alignment plan or decision needed: use per-state counters or PluginContext.record_call.

## Acceptance Criteria

- For each state_id, the first recorded call has call_index 0 and increments by 1 for each subsequent call in that state.
- No cross-row leakage of call_index values.

## Tests

- Suggested tests to run: pytest tests/plugins/transforms/azure/test_content_safety.py pytest tests/plugins/transforms/azure/test_prompt_shield.py
- New tests required: yes, per-state call_index reset behavior.

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: docs/design/architecture.md
