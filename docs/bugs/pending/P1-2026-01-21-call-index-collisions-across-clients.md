# Bug Report: Call indices collide across audited clients sharing a state

## Summary

- `AuditedClientBase` tracks call indices per client instance; using multiple audited clients (HTTP + LLM) in the same node state yields duplicate `(state_id, call_index)` and triggers DB integrity errors.

## Severity

- Severity: major
- Priority: P1

## Reporter

- Name or handle: codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-2` @ `ae2c0e6f088f467276582fa8016f91b4d3bb26c7`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive into `src/elspeth/plugins/clients` and file bugs
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection of audited client base and landscape schema

## Steps To Reproduce

1. Create a transform that instantiates both `AuditedHTTPClient` and `AuditedLLMClient` with the same `state_id`.
2. Make one call with each client in the same node state.
3. Observe the second `record_call` fails with `IntegrityError` on the `calls(state_id, call_index)` unique constraint.

## Expected Behavior

- Call indices are unique per `state_id` across all external calls in a node state, regardless of which client type makes the call.

## Actual Behavior

- Each audited client starts its own counter at 0, so multiple clients in the same state generate duplicate `call_index` values.

## Evidence

- Per-client call index counter: `src/elspeth/plugins/clients/base.py:41-58`
- Unique constraint on `(state_id, call_index)`: `src/elspeth/core/landscape/schema.py:188-204`

## Impact

- User-facing impact: pipelines that use multiple audited clients in a single transform crash when recording calls.
- Data integrity / security impact: calls may be missing from the audit trail if collisions occur.
- Performance or cost impact: N/A

## Root Cause Hypothesis

- Call indices are scoped to individual client instances instead of the node state, but the database enforces uniqueness per state.

## Proposed Fix

- Code changes (modules/files):
  - Centralize call index allocation by `state_id` (e.g., in `LandscapeRecorder` or `PluginContext`) and have audited clients request the next index.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that uses two audited clients with the same `state_id` and asserts both calls record successfully with distinct indices.
- Risks or migration steps:
  - Ensure any cached clients preserve monotonic indices across retries.

## Architectural Deviations

- Spec or doc reference: `src/elspeth/core/landscape/schema.py:203` (unique constraint on call indices)
- Observed divergence: audited clients do not coordinate indices across client types.
- Reason (if known): per-client counter was simpler to implement.
- Alignment plan or decision needed: adopt a shared per-state counter.

## Acceptance Criteria

- Multiple audited clients can record calls under the same `state_id` without integrity errors.

## Tests

- Suggested tests to run: `pytest tests/plugins/clients/ -k call_index`
- New tests required: yes

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
