# BUG: Batched Example Fails with Node ID Collision

**Reported:** 2026-01-26
**Severity:** Medium (blocks batched aggregation examples)
**Status:** RESOLVED

## Summary

Running `examples/openrouter_sentiment/settings_batched.yaml` fails with a UNIQUE constraint violation on `nodes.node_id` when run twice against the same database.

## Root Cause

The `nodes` table had `PRIMARY KEY (node_id)` but `node_id` is derived from a deterministic config hash. This means:

1. Same pipeline configuration → same `node_id`
2. Second run with same config → `UNIQUE constraint failed`

The design conflict: node IDs should be unique per run, not globally unique across all runs.

## Fix

Changed schema to use composite primary key `(node_id, run_id)`:

1. **schema.py**: Changed `nodes_table` to `PrimaryKeyConstraint("node_id", "run_id")`
2. **schema.py**: Updated all tables with FK to nodes to use `ForeignKeyConstraint` for composite FK
3. **recorder.py**: Added `run_id` parameter to `begin_node_state()`
4. **executors.py / coalesce_executor.py**: Pass `run_id` to `begin_node_state()` calls

This allows the same node configuration to exist in multiple runs while preserving checkpoint/resume compatibility (same run_id = same node_id).

## Files Changed

- `src/elspeth/core/landscape/schema.py` - Composite PK and FKs
- `src/elspeth/core/landscape/recorder.py` - Added run_id to begin_node_state
- `src/elspeth/engine/executors.py` - Pass run_id to recorder
- `src/elspeth/engine/coalesce_executor.py` - Pass run_id to recorder
- Test files updated to pass run_id parameter

## Verification

Both commands now succeed:

```bash
rm -f examples/openrouter_sentiment/runs/audit_batched.db*
.venv/bin/elspeth run -s examples/openrouter_sentiment/settings_batched.yaml --execute
.venv/bin/elspeth run -s examples/openrouter_sentiment/settings_batched.yaml --execute
# ✓ Run COMPLETED: 5 rows processed (both runs succeed)
```
