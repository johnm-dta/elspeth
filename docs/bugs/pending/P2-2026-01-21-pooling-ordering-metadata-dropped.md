# Bug Report: ReorderBuffer ordering metadata is dropped before audit

## Summary

- ReorderBuffer captures submit/complete indices and timing metadata, but PooledExecutor discards those fields and returns only `TransformResult`. The ordering metadata never reaches the audit trail, violating the design requirement to record submit/complete indices.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: Codex
- Date: 2026-01-21
- Related run/issue ID: N/A

## Environment

- Commit/branch: ae2c0e6 / fix/rc1-bug-burndown-session-2
- OS: Linux
- Python version: Python 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: Deep dive src/elspeth/plugins/pooling for bugs; create bug reports.
- Model/version: GPT-5 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, no escalations
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: Code inspection only

## Steps To Reproduce

1. Use `PooledExecutor.execute_batch` with any batch size > 1.
2. Inspect the returned results and any recorded audit context for submit/complete indices.

## Expected Behavior

- Each row should have ordering metadata (`submit_index`, `complete_index`, timestamps) recorded for audit/observability.

## Actual Behavior

- `execute_batch` extracts only `entry.result`, so submit/complete indices and timing data are never surfaced or recorded.

## Evidence

- Code: `src/elspeth/plugins/pooling/executor.py:199-201` appends `entry.result` only.
- Source metadata: `src/elspeth/plugins/pooling/reorder_buffer.py:12-34` defines submit/complete indices and timing.
- Spec: `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md:170-175` requires ordering metadata in audit context.

## Impact

- User-facing impact: Harder to diagnose out-of-order behavior or missing rows in pooled batches.
- Data integrity / security impact: Audit trail lacks required ordering proof.
- Performance or cost impact: None directly.

## Root Cause Hypothesis

- PooledExecutor returns only `TransformResult` and drops `BufferEntry` metadata without recording it.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/plugins/pooling/executor.py`, potentially recorder integration
- Config or schema changes: Add fields for ordering metadata in node state context or call records.
- Tests to add/update: Add a test that verifies submit/complete indices are captured in audit context.
- Risks or migration steps: Ensure metadata is recorded deterministically without inflating per-row payloads excessively.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md:170-175`
- Observed divergence: Ordering metadata is computed but discarded before audit recording.
- Reason (if known): Missing propagation path from PooledExecutor to recorder.
- Alignment plan or decision needed: Decide where to persist ordering metadata (node_state context vs call records).

## Acceptance Criteria

- Audit context includes `submit_index` and `complete_index` per pooled row.
- Tests verify metadata is present and consistent with reorder buffer ordering.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_reorder_buffer.py -k timing`
- New tests required: Yes (audit metadata propagation).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md`
