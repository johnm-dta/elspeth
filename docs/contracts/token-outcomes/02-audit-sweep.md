# Audit Sweep: Token Outcome Gaps

This sweep is the primary gap detector. Run it after a completed run to find
missing outcomes, mismatched sink states, and missing required fields.

## Preconditions

- Run status is COMPLETED or FAILED (not RUNNING).
- Aggregation buffers have been flushed at end-of-source.

## Safety rule for joins

The nodes table uses a composite key `(node_id, run_id)`.
When joining node_states to nodes, join on BOTH keys or filter by
`node_states.run_id` directly.

## Core queries

### 1) Tokens missing terminal outcome

```sql
SELECT t.token_id, t.row_id
FROM tokens t
JOIN rows r ON r.row_id = t.row_id
LEFT JOIN token_outcomes o
  ON o.token_id = t.token_id AND o.is_terminal = 1
WHERE r.run_id = :run_id
  AND o.token_id IS NULL;
```

### 2) Duplicate terminal outcomes (should be empty)

```sql
SELECT token_id, COUNT(*) AS terminal_count
FROM token_outcomes
WHERE is_terminal = 1
GROUP BY token_id
HAVING COUNT(*) > 1;
```

### 3) Required fields missing for outcomes

```sql
SELECT outcome_id, token_id, outcome
FROM token_outcomes
WHERE
  (outcome IN ('completed','routed') AND sink_name IS NULL)
  OR (outcome IN ('failed','quarantined') AND error_hash IS NULL)
  OR (outcome = 'forked' AND fork_group_id IS NULL)
  OR (outcome = 'coalesced' AND join_group_id IS NULL)
  OR (outcome = 'expanded' AND expand_group_id IS NULL)
  OR (outcome IN ('buffered','consumed_in_batch') AND batch_id IS NULL);
```

### 4) COMPLETED outcome without completed sink node_state

```sql
SELECT o.token_id
FROM token_outcomes o
JOIN rows r ON r.run_id = o.run_id
LEFT JOIN node_states ns
  ON ns.token_id = o.token_id AND ns.run_id = o.run_id
LEFT JOIN nodes n
  ON n.node_id = ns.node_id AND n.run_id = ns.run_id
WHERE o.run_id = :run_id
  AND o.outcome = 'completed'
  AND NOT (n.node_type = 'sink' AND ns.status = 'completed');
```

### 5) Completed sink node_state without COMPLETED outcome

```sql
SELECT DISTINCT ns.token_id
FROM node_states ns
JOIN nodes n
  ON n.node_id = ns.node_id AND n.run_id = ns.run_id
LEFT JOIN token_outcomes o
  ON o.token_id = ns.token_id AND o.is_terminal = 1 AND o.outcome = 'completed'
WHERE ns.run_id = :run_id
  AND n.node_type = 'sink'
  AND ns.status = 'completed'
  AND o.token_id IS NULL;
```

### 6) Fork children missing parent links

```sql
SELECT t.token_id
FROM tokens t
LEFT JOIN token_parents p ON p.token_id = t.token_id
WHERE t.fork_group_id IS NOT NULL
  AND p.token_id IS NULL;
```

### 7) Expand children missing parent links

```sql
SELECT t.token_id
FROM tokens t
LEFT JOIN token_parents p ON p.token_id = t.token_id
WHERE t.expand_group_id IS NOT NULL
  AND p.token_id IS NULL;
```

## Interpretation guide

- Missing terminal outcomes (Query 1) are always bugs after run completion.
- Duplicate terminal outcomes (Query 2) should be impossible unless the unique
  index was bypassed or the DB is corrupted.
- Missing required fields (Query 3) indicate incorrect recorder call sites.
- Sink mismatches (Queries 4 and 5) indicate a broken sink path or outcome
  recording gap.

## What to do with results

1. Group failures by outcome type.
2. Map each group to the outcome path map (docs/contracts/token-outcomes/01-outcome-path-map.md).
3. Reproduce with a minimal pipeline test.
4. Add regression test that fails the audit sweep.
