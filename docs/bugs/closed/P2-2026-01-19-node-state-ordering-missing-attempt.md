# Bug Report: `get_node_states_for_token()` orders only by `step_index` (retry attempts can be nondeterministic)

## Summary

- `node_states` supports retries via `attempt` and enforces uniqueness on `(token_id, step_index, attempt)`.
- `LandscapeRecorder.get_node_states_for_token()` orders only by `step_index`. When multiple attempts exist for the same `step_index`, relative ordering is not guaranteed across database backends.
- This can:
  - produce confusing `explain()` output ordering for retries
  - break determinism expectations for signed exports (record order changes -> signature chain changes)

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
- Notable tool calls or steps: code inspection

## Steps To Reproduce

1. Create two node states for the same `token_id` with the same `step_index` but different `attempt` values (0 then 1).
2. Call `get_node_states_for_token(token_id)` on different database backends (or after different query plans/vacuum/reindex).
3. Observe that ordering between attempts is not guaranteed because the query orders only by `step_index`.

## Expected Behavior

- Node state ordering is explicit and deterministic:
  - `ORDER BY step_index, attempt` (and optionally `state_id`) so retries appear in ascending attempt order.

## Actual Behavior

- Query orders only by `step_index`, leaving attempt ordering undefined.

## Evidence

- Query ordering omits attempt:
  - `src/elspeth/core/landscape/recorder.py:1750-1763`
- Schema indicates retry attempts are a first-class dimension:
  - `src/elspeth/core/landscape/schema.py` `node_states` has `attempt` and unique constraints on `(token_id, step_index, attempt)`

## Impact

- User-facing impact: confusing explain output for retries (“attempt 1” could appear before “attempt 0”).
- Data integrity / security impact: low.
- Performance or cost impact: can invalidate deterministic export assumptions (signed export record ordering changes).

## Root Cause Hypothesis

- Query ordering was written for the “no retries” happy path and not updated when `attempt` was introduced.

## Proposed Fix

- Code changes (modules/files):
  - `src/elspeth/core/landscape/recorder.py`:
    - Change ordering to `.order_by(node_states_table.c.step_index, node_states_table.c.attempt)`
    - Optionally add a stable tie-breaker (e.g., `state_id`) if needed for multi-backend determinism.
- Config or schema changes: none.
- Tests to add/update:
  - Add a test that creates two attempts for same step_index and asserts returned ordering is attempt-ascending.
- Risks or migration steps:
  - None.

## Architectural Deviations

- Spec or doc reference: `docs/design/requirements.md` (signed export determinism expectations)
- Observed divergence: retry attempt ordering not deterministic.
- Reason (if known): missing ordering clause.
- Alignment plan or decision needed: none.

## Acceptance Criteria

- `get_node_states_for_token()` returns states ordered by `(step_index, attempt)` consistently.

## Tests

- Suggested tests to run: `pytest tests/core/landscape/test_recorder.py -k retry`
- New tests required: yes (ordering assertion)

## Notes / Links

- Related issues/PRs: N/A

## Resolution

**Status:** CLOSED (2026-01-21)
**Resolved by:** Claude Opus 4.5

### Changes Made

**Code fix (`src/elspeth/core/landscape/recorder.py`):**

Changed `get_node_states_for_token()` to order by both `step_index` and `attempt`:

```python
# Before (Bug):
query = select(node_states_table).where(node_states_table.c.token_id == token_id).order_by(node_states_table.c.step_index)

# After (Fix):
query = (
    select(node_states_table)
    .where(node_states_table.c.token_id == token_id)
    .order_by(node_states_table.c.step_index, node_states_table.c.attempt)
)
```

Added comment explaining the fix:
```python
# Order by (step_index, attempt) for deterministic ordering across retries
# Bug fix: P2-2026-01-19-node-state-ordering-missing-attempt
```

Updated docstring to reflect the ordering:
```python
Returns:
    List of NodeState models (discriminated union), ordered by (step_index, attempt)
```

**Tests added (`tests/core/landscape/test_recorder.py`):**
- `TestNodeStateOrderingWithRetries` class with 1 regression test:
  - `test_get_node_states_orders_by_step_index_and_attempt` - Creates node states with multiple attempts inserted out of order, verifies they're returned in (step_index, attempt) order

### Verification

```bash
.venv/bin/python -m pytest tests/core/landscape/test_recorder.py -v
# 100 passed (99 existing + 1 new)
```

### Notes

Without explicit attempt ordering, results were non-deterministic across database backends because SQLite/Postgres may return rows in different orders when only partial ordering is specified. This could cause:
1. Confusing `explain()` output (attempt 1 appearing before attempt 0)
2. Non-deterministic signed exports (record order affects signature chain)
