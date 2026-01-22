# Bug Report: Concurrent execute_batch calls can mix results across batches

## Summary

- PooledExecutor uses a single ReorderBuffer for all batches. If `execute_batch()` is called concurrently on the same executor, results from different batches can be interleaved and returned to the wrong caller because both calls drain the shared buffer.

## Severity

- Severity: major
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

1. Create a single `PooledExecutor` instance.
2. Spawn two threads that both call `execute_batch()` with different marker rows.
3. Use `process_fn` delays so completions interleave across batches.
4. Observe each thread's results list contains rows from the other batch or blocks waiting on the other batch's indices.

## Expected Behavior

- Each `execute_batch()` call should be isolated, or concurrent calls should be rejected with a clear error.

## Actual Behavior

- Results can be drained from a shared reorder buffer, causing cross-batch contamination or ordering stalls.

## Evidence

- Code: single buffer shared per executor instance (`src/elspeth/plugins/pooling/executor.py:98-101`).
- `execute_batch()` drains the shared buffer without scoping results to a batch (`src/elspeth/plugins/pooling/executor.py:192-210`).

## Impact

- User-facing impact: Wrong rows returned to callers in multi-threaded usage.
- Data integrity / security impact: Audit trail can associate results with the wrong row set.
- Performance or cost impact: Potential stalls if batches block each other on ordering.

## Root Cause Hypothesis

- Reorder buffer and counters are shared across batches; `execute_batch()` is not re-entrant or guarded.

## Proposed Fix

- Code changes (modules/files): `src/elspeth/plugins/pooling/executor.py`
- Config or schema changes: None
- Tests to add/update: Add a concurrency test that runs two simultaneous batches and asserts isolation.
- Risks or migration steps: If concurrent calls are unsupported, enforce a single-flight lock and document it.

## Architectural Deviations

- Spec or doc reference (e.g., docs/design/architecture.md#L...): N/A
- Observed divergence: No batch scoping or mutual exclusion for concurrent calls.
- Reason (if known): Executor assumes single caller.
- Alignment plan or decision needed: Decide whether to support concurrency or reject it explicitly.

## Acceptance Criteria

- Concurrent execute_batch calls either isolate buffers or raise a deterministic error.
- Tests demonstrate no cross-batch result mixing.

## Tests

- Suggested tests to run: `pytest tests/plugins/llm/test_pooled_executor.py -k concurrent`
- New tests required: Yes (concurrent batch isolation).

## Notes / Links

- Related issues/PRs: N/A
- Related design docs: N/A
