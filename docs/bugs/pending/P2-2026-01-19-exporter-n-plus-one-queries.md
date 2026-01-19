# Bug Report: LandscapeExporter uses N+1 query pattern (likely very slow for large runs)

## Summary

- `LandscapeExporter._iter_records()` performs nested per-entity queries:
  - for each row → query tokens
  - for each token → query token_parents + node_states
  - for each state → query routing_events + calls
- This can lead to extremely high query counts and poor performance on large runs, even on local SQLite.

## Severity

- Severity: major
- Priority: P2

## Reporter

- Name or handle: codex
- Date: 2026-01-19
- Related run/issue ID: N/A

## Environment

- Commit/branch: `main` @ `8ca061c9293db459c9a900f2f74b19b59a364a42`
- OS: Linux (Ubuntu kernel 6.8.0-90-generic)
- Python version: 3.13.1
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Agent Context (if relevant)

- Goal or task prompt: deep dive subsystem 4 (Landscape) and create bug tickets
- Model/version: GPT-5.2 (Codex CLI)
- Tooling and permissions (sandbox/approvals): workspace-write, network restricted, approvals on-request
- Determinism details (seed, run ID): N/A
- Notable tool calls or steps: code inspection (complexity analysis)

## Steps To Reproduce

1. Run a pipeline with thousands of rows and multiple node states per token.
2. Enable post-run export (`landscape.export.enabled: true`).
3. Observe export time grows superlinearly due to many small DB transactions/queries.

## Expected Behavior

- Export should execute a bounded number of queries per run (ideally one per table/record type), using joins or batch loads keyed by `run_id` and related IDs.

## Actual Behavior

- Export performs nested queries across row/token/state hierarchies.

## Evidence

- Nested query loops:
  - `src/elspeth/core/landscape/exporter.py:199-332`
- Each `LandscapeRecorder.get_*` call opens its own transaction/connection:
  - `src/elspeth/core/landscape/recorder.py` (multiple `with self._db.connection()` blocks)

## Impact

- User-facing impact: slow exports for realistic datasets; can make “post-run export” unusable at scale.
- Data integrity / security impact: low.
- Performance or cost impact: high DB overhead (many transactions) and potential lock contention on SQLite.

## Root Cause Hypothesis

- Exporter was implemented for correctness first and composes existing per-entity query methods without batching.

## Proposed Fix

- Code changes (modules/files):
  - Implement batched export queries:
    - prefetch all rows/tokens/states/events/calls with a small number of queries
    - assemble records in Python using maps keyed by IDs
  - Alternatively, add dedicated export query methods on recorder that return grouped results efficiently.
- Config or schema changes: none.
- Tests to add/update:
  - Add a performance/regression test (bounded query count) if feasible, or at least a benchmark harness.
- Risks or migration steps:
  - Ensure deterministic output ordering is preserved when switching to batch queries (important for signing).

## Architectural Deviations

- Spec or doc reference: `docs/design/requirements.md` (export feature expectations)
- Observed divergence: exporter likely scales poorly.
- Reason (if known): N+1 composition pattern.
- Alignment plan or decision needed: decide acceptable export performance targets (rows/sec, max run size).

## Acceptance Criteria

- Export for large runs performs a bounded number of queries and completes in reasonable time (define target).
- Output ordering remains deterministic for signed exports.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_exporter.py`
- New tests required: optional (benchmark/harness)

## Notes / Links

- Related issues/PRs: N/A
