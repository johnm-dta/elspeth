# Bug Report: get_outcome_analysis counts fork/join operations across all runs

## Summary

- Fork/join count queries in `get_outcome_analysis` have no `run_id` filter, counting operations across all runs in the database.

## Severity

- Severity: moderate
- Priority: P2

## Reporter

- Name or handle: Codex (triaged by Claude)
- Date: 2026-01-31

## Evidence

- `src/elspeth/mcp/server.py:1106-1121` - fork_count/join_count queries have no run_id filter
- `tokens_table` lacks `run_id` column, would need join through `rows_table`

## Impact

- User-facing impact: Incorrect counts in multi-run databases
- Data integrity: None (MCP is debugging tool)

## Proposed Fix

- Add run_id filter via join to rows_table or token_outcomes_table

## Acceptance Criteria

- Fork/join counts scoped to requested run_id

## Verification (2026-02-01)

**Status: STILL VALID**

- `fork_count`/`join_count` still query `tokens_table` without scoping to `run_id`. (`src/elspeth/mcp/server.py:1106-1121`)

## Resolution (2026-02-02)

**Status: FIXED**

- Changed queries to use `token_outcomes_table` instead of `tokens_table`
- `token_outcomes_table` already has `run_id`, `fork_group_id`, and `join_group_id` columns
- Added `run_id` filter to both fork_count and join_count queries
- Removed unused `tokens_table` import from function
- Fix at `src/elspeth/mcp/server.py:1107-1130`
