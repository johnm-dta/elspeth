# Audit Sweep: Token Outcome Gaps

Current as of 2026-05-20.

Run these read-only checks after a run reaches a terminal run status. The checks
target the ADR-019 two-axis token outcome model.

## Preconditions

- The run is no longer actively processing.
- End-of-source aggregation and coalesce flushes have run.
- Use `completed`, not the retired `is_terminal` column.
- Join `nodes` with both `node_id` and `run_id`.

## 1. Tokens Missing A Completed Outcome

```sql
SELECT t.token_id, t.row_id
FROM tokens t
LEFT JOIN token_outcomes o
  ON o.run_id = t.run_id
 AND o.token_id = t.token_id
 AND o.completed = 1
WHERE t.run_id = :run_id
  AND o.token_id IS NULL;
```

This should be empty after a run is terminal.

## 2. Duplicate Completed Outcomes

```sql
SELECT run_id, token_id, COUNT(*) AS completed_count
FROM token_outcomes
WHERE run_id = :run_id
  AND completed = 1
GROUP BY run_id, token_id
HAVING COUNT(*) > 1;
```

This should be impossible unless the partial unique index was bypassed or the
database is corrupt.

## 3. Illegal Non-Terminal Rows

```sql
SELECT outcome_id, token_id, outcome, path, completed
FROM token_outcomes
WHERE run_id = :run_id
  AND completed = 0
  AND NOT (outcome IS NULL AND path = 'buffered');
```

## 4. Illegal Completed Rows

```sql
SELECT outcome_id, token_id, outcome, path, completed
FROM token_outcomes
WHERE run_id = :run_id
  AND completed = 1
  AND (
    outcome IS NULL
    OR (outcome, path) NOT IN (
      ('success', 'default_flow'),
      ('success', 'gate_routed'),
      ('success', 'gate_discarded'),
      ('failure', 'on_error_routed'),
      ('success', 'filter_dropped'),
      ('success', 'coalesced'),
      ('failure', 'unrouted'),
      ('failure', 'quarantined_at_source'),
      ('transient', 'sink_fallback_to_failsink'),
      ('failure', 'sink_discarded'),
      ('transient', 'fork_parent'),
      ('transient', 'expand_parent'),
      ('transient', 'batch_consumed')
    )
  );
```

## 5. Required Discriminator Fields Missing

```sql
SELECT outcome_id, token_id, outcome, path
FROM token_outcomes
WHERE run_id = :run_id
  AND (
    (path IN ('default_flow', 'gate_routed') AND sink_name IS NULL)
    OR (path = 'on_error_routed' AND (sink_name IS NULL OR error_hash IS NULL))
    OR (path = 'coalesced' AND join_group_id IS NULL)
    OR (path IN ('unrouted', 'quarantined_at_source') AND error_hash IS NULL)
    OR (path = 'sink_fallback_to_failsink' AND (sink_name IS NULL OR error_hash IS NULL))
    OR (path = 'sink_discarded' AND (sink_name IS NULL OR error_hash IS NULL OR sink_name <> '__discard__'))
    OR (path = 'fork_parent' AND fork_group_id IS NULL)
    OR (path = 'expand_parent' AND expand_group_id IS NULL)
    OR (path IN ('batch_consumed', 'buffered') AND batch_id IS NULL)
  );
```

## 6. Sink Success Without Completed Sink State

```sql
SELECT o.token_id, o.path, o.sink_name
FROM token_outcomes o
LEFT JOIN node_states ns
  ON ns.run_id = o.run_id
 AND ns.token_id = o.token_id
LEFT JOIN nodes n
  ON n.run_id = ns.run_id
 AND n.node_id = ns.node_id
 AND n.node_type = 'sink'
 AND ns.status = 'completed'
WHERE o.run_id = :run_id
  AND o.completed = 1
  AND o.outcome = 'success'
  AND o.sink_name IS NOT NULL
  AND n.node_id IS NULL;
```

## 7. Completed Sink State Without Success Outcome

```sql
SELECT DISTINCT ns.token_id
FROM node_states ns
JOIN nodes n
  ON n.run_id = ns.run_id
 AND n.node_id = ns.node_id
LEFT JOIN token_outcomes o
  ON o.run_id = ns.run_id
 AND o.token_id = ns.token_id
 AND o.completed = 1
 AND o.outcome = 'success'
 AND o.sink_name IS NOT NULL
WHERE ns.run_id = :run_id
  AND n.node_type = 'sink'
  AND ns.status = 'completed'
  AND o.token_id IS NULL;
```

## 8. Fork Or Expand Children Missing Parent Links

```sql
SELECT t.token_id, t.row_id, t.fork_group_id, t.expand_group_id
FROM tokens t
LEFT JOIN token_parents p
  ON p.token_id = t.token_id
WHERE t.run_id = :run_id
  AND (t.fork_group_id IS NOT NULL OR t.expand_group_id IS NOT NULL)
  AND p.token_id IS NULL;
```

## What To Do With Results

1. Group failures by `(outcome, path)`.
2. Use [Outcome Path Map](01-outcome-path-map.md) to find the producer.
3. Reproduce the gap with the smallest pipeline or repository-level test.
4. Add a regression that fails the relevant sweep query.
5. Fix the producer path and re-run the sweep.
