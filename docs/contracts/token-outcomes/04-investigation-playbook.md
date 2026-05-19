# Token Outcome Gap Investigation Playbook

Current as of 2026-05-20.

Use this playbook when token outcomes are missing, illegal, duplicated, or
inconsistent with node/batch/artifact evidence.

## Inputs

- Landscape database path.
- Run ID.
- Settings file or exported run settings when available.
- `token_id` or `row_id` for at least one affected token.
- Output from the relevant [audit sweep](02-audit-sweep.md) query.

## Step 1: Confirm Run State

```sql
SELECT run_id, status, started_at, completed_at
FROM runs
WHERE run_id = :run_id;
```

Only enforce terminal-outcome completeness after the run is no longer actively
processing.

## Step 2: Classify By Pair

Group failures by `completed`, `outcome`, and `path`:

```sql
SELECT completed, outcome, path, COUNT(*) AS records
FROM token_outcomes
WHERE run_id = :run_id
GROUP BY completed, outcome, path
ORDER BY completed, outcome, path;
```

Classify the gap as:

- missing completed outcome
- duplicate completed outcome
- illegal `(outcome, path, completed)` pair
- missing required discriminator field
- forbidden discriminator field present
- sink/node-state mismatch
- parent/child or batch evidence gap

## Step 3: Explain A Representative Token

```bash
elspeth explain --run <RUN_ID> --token <TOKEN_ID> --database <DB> --json
```

If only a row is known:

```bash
elspeth explain --run <RUN_ID> --row <ROW_ID> --database <DB> --json
```

Use [Investigate Routing](../../runbooks/investigate-routing.md) for operator
lineage queries.

## Step 4: Map To Producer

Use [Outcome Path Map](01-outcome-path-map.md) to locate the producer. Verify the
current source before editing; do not rely on old line-number references.

## Step 5: Reproduce Minimally

Create the smallest reproduction that exercises the producer path:

- one source row for source quarantine or default sink success
- one gate for gate-route and gate-discard paths
- one transform returning failure for `on_error_routed`
- one filter/drop transform for `filter_dropped`
- one fork plus coalesce for parent/child lineage
- one batch-aware transform for `buffered` and `batch_consumed`

Prefer repository-level tests for field-constraint bugs and integration tests
for config-to-runtime path bugs.

## Step 6: Fix And Re-Verify

1. Add the failing regression first.
2. Fix the earliest producer or repository boundary that can enforce the
   invariant mechanically.
3. Re-run the focused regression.
4. Re-run the audit sweep query that exposed the gap.
5. Update this contract set if the legal model changed.

## Escalate Immediately

Escalate when:

- completed runs have tokens without completed outcomes
- duplicate completed outcomes appear
- illegal `(outcome, path)` pairs are persisted
- sink node states and token outcomes disagree
- audit read paths require coercion to interpret Tier-1 data
