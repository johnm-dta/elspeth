# Bug Report: Pool stats missing required concurrency and delay fields

## Summary

- `PooledExecutor.get_stats()` omits `max_concurrent_reached` and `dispatch_delay_at_completion_ms`, which are specified for node state context in the pooled LLM design. The audit context cannot report peak concurrency or the delay in effect at completion.

## Severity

- Severity: minor
- Priority: P3

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

1. Instantiate `PooledExecutor` and call `get_stats()`.
2. Inspect the returned `pool_stats` and `pool_config` dictionaries.

## Expected Behavior

- Stats include `max_concurrent_reached` and `dispatch_delay_at_completion_ms` as specified in the pooling design for audit context.

## Actual Behavior

- Only capacity/success counters and current/peak delay are returned; concurrency and dispatch-at-completion metrics are missing.

## Evidence

- Code: `src/elspeth/plugins/pooling/executor.py:122-140` returns limited stats.
- Spec: `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md:152-165` lists required fields.

## Impact

- User-facing impact: Reduced observability for pooled execution behavior.
- Data integrity / security impact: Audit context lacks required concurrency and delay metadata.
- Performance or cost impact: None directly.

## Root Cause Hypothesis

- PooledExecutor does not track active concurrency or last dispatch delay at completion.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/plugins/pooling/executor.py`
- Config or schema changes: Add fields in stats output and ensure recorder persists them.
- Tests to add/update: Add tests verifying `max_concurrent_reached` and `dispatch_delay_at_completion_ms` are present.
- Risks or migration steps: None (additive metadata).

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md:152-165`
- Observed divergence: Required pool stats fields are missing from `get_stats()` output.
- Reason (if known): Not implemented when pooling code moved to `plugins/pooling`.
- Alignment plan or decision needed: Implement concurrency counters and delay snapshotting.

## Acceptance Criteria

- `get_stats()` includes `max_concurrent_reached` and `dispatch_delay_at_completion_ms`.
- Tests demonstrate the metrics are populated under pooled execution.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_pooled_executor.py -k stats`
- New tests required: Yes (stats fields).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: `docs/plans/completed/2026-01-20-pooled-llm-queries-design.md`
