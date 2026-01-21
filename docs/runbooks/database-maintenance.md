# Runbook: Database Maintenance

Maintain the audit database and payload store.

---

## Symptoms

- Audit database growing large
- Slow query performance
- Disk space warnings
- Old runs no longer needed for compliance

---

## Prerequisites

- Database access (SQLite file or PostgreSQL credentials)
- Understanding of data retention requirements
- Backup before any destructive operations

---

## Procedure

### Step 1: Assess Current State

**SQLite:**

```bash
# Database file size
ls -lh runs/audit.db

# Row counts
sqlite3 runs/audit.db "
  SELECT 'runs' as table_name, COUNT(*) as row_count FROM runs
  UNION ALL
  SELECT 'rows', COUNT(*) FROM rows
  UNION ALL
  SELECT 'tokens', COUNT(*) FROM tokens
  UNION ALL
  SELECT 'node_states', COUNT(*) FROM node_states
  UNION ALL
  SELECT 'checkpoints', COUNT(*) FROM checkpoints
  UNION ALL
  SELECT 'artifacts', COUNT(*) FROM artifacts;
"

# Runs by date
sqlite3 runs/audit.db "
  SELECT DATE(started_at) as run_date, COUNT(*) as run_count
  FROM runs
  GROUP BY DATE(started_at)
  ORDER BY run_date DESC
  LIMIT 30;
"
```

**PostgreSQL:**

```bash
psql -d elspeth -c "
  SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as size
  FROM pg_tables
  WHERE schemaname = 'public'
  ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;
"
```

### Step 2: Identify Retention Candidates

Find runs older than retention period (e.g., 90 days):

```bash
sqlite3 runs/audit.db "
  SELECT run_id, started_at, status,
         (SELECT COUNT(*) FROM rows WHERE rows.run_id = runs.run_id) as row_count
  FROM runs
  WHERE started_at < datetime('now', '-90 days')
  ORDER BY started_at;
"
```

### Step 3: Export Before Deletion (Optional)

If compliance requires archives:

```bash
# Export run data to JSON
sqlite3 runs/audit.db "
  SELECT json_object(
    'run_id', run_id,
    'config', config,
    'status', status,
    'started_at', started_at,
    'completed_at', completed_at
  )
  FROM runs
  WHERE run_id = '<RUN_ID>';
" > run_archive.json

# Export all processing states for a run
sqlite3 -header -csv runs/audit.db "
  SELECT ns.*, t.row_id, t.branch_name
  FROM node_states ns
  JOIN tokens t ON ns.token_id = t.token_id
  JOIN rows r ON t.row_id = r.row_id
  WHERE r.run_id = '<RUN_ID>';
" > node_states_archive.csv
```

### Step 4: Delete Old Data

**⚠️ CAUTION: DESTRUCTIVE OPERATION**

This procedure permanently deletes audit data. Before proceeding:
- [ ] Verify you have a recent backup
- [ ] Confirm retention period meets compliance requirements
- [ ] Test the deletion query with `SELECT COUNT(*)` first
- [ ] Ensure no pipelines are currently running

```bash
# Backup first
cp runs/audit.db runs/audit.db.backup.$(date +%Y%m%d)

# Delete runs older than 90 days (cascades to related tables)
# Execute statements one at a time due to foreign key constraints
sqlite3 runs/audit.db "DELETE FROM checkpoints WHERE run_id IN (SELECT run_id FROM runs WHERE started_at < datetime('now', '-90 days'));"
sqlite3 runs/audit.db "DELETE FROM routing_events WHERE state_id IN (SELECT state_id FROM node_states WHERE token_id IN (SELECT token_id FROM tokens WHERE row_id IN (SELECT row_id FROM rows WHERE run_id IN (SELECT run_id FROM runs WHERE started_at < datetime('now', '-90 days')))));"
sqlite3 runs/audit.db "DELETE FROM token_outcomes WHERE run_id IN (SELECT run_id FROM runs WHERE started_at < datetime('now', '-90 days'));"
sqlite3 runs/audit.db "DELETE FROM node_states WHERE token_id IN (SELECT token_id FROM tokens WHERE row_id IN (SELECT row_id FROM rows WHERE run_id IN (SELECT run_id FROM runs WHERE started_at < datetime('now', '-90 days'))));"
sqlite3 runs/audit.db "DELETE FROM tokens WHERE row_id IN (SELECT row_id FROM rows WHERE run_id IN (SELECT run_id FROM runs WHERE started_at < datetime('now', '-90 days')));"
sqlite3 runs/audit.db "DELETE FROM artifacts WHERE run_id IN (SELECT run_id FROM runs WHERE started_at < datetime('now', '-90 days'));"
sqlite3 runs/audit.db "DELETE FROM rows WHERE run_id IN (SELECT run_id FROM runs WHERE started_at < datetime('now', '-90 days'));"
sqlite3 runs/audit.db "DELETE FROM nodes WHERE run_id IN (SELECT run_id FROM runs WHERE started_at < datetime('now', '-90 days'));"
sqlite3 runs/audit.db "DELETE FROM edges WHERE run_id IN (SELECT run_id FROM runs WHERE started_at < datetime('now', '-90 days'));"
sqlite3 runs/audit.db "DELETE FROM runs WHERE started_at < datetime('now', '-90 days');"
```

### Step 5: Vacuum the Database

Reclaim disk space after deletion:

**SQLite:**

```bash
sqlite3 runs/audit.db "VACUUM;"
```

**PostgreSQL:**

```bash
psql -d elspeth -c "VACUUM ANALYZE;"
```

### Step 6: Clean Payload Store

Remove orphaned payloads:

```bash
# Find payload retention setting
cat pipeline.yaml | grep -A3 "payload_store:"

# Delete payloads older than retention
find .elspeth/payloads -type f -mtime +90 -delete
```

---

## Maintenance Schedule

| Task | Frequency | Command |
|------|-----------|---------|
| Check database size | Weekly | `ls -lh runs/audit.db` |
| Delete old runs | Monthly | See Step 4 |
| Vacuum database | After deletions | `sqlite3 runs/audit.db "VACUUM;"` |
| Clean payload store | Monthly | `find .elspeth/payloads -mtime +90 -delete` |

---

## Performance Optimization

### Add Indexes (if missing)

```sql
-- Index for querying rows by run
CREATE INDEX IF NOT EXISTS idx_rows_run_id ON rows(run_id);

-- Index for querying node states by token
CREATE INDEX IF NOT EXISTS idx_node_states_token ON node_states(token_id);

-- Index for date-based queries
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
```

### Analyze Tables

```bash
sqlite3 runs/audit.db "ANALYZE;"
```

---

## PostgreSQL-Specific Tasks

### Check for Bloat

```sql
SELECT
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) as total_size,
  pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) as table_size
FROM pg_tables
WHERE schemaname = 'public';
```

### Reindex

```bash
psql -d elspeth -c "REINDEX DATABASE elspeth;"
```

### Connection Pool Monitoring

```sql
SELECT count(*) FROM pg_stat_activity WHERE datname = 'elspeth';
```

---

## Troubleshooting

### Database Locked (SQLite)

```bash
# Check for active connections
fuser runs/audit.db

# Wait for lock to release or kill process
```

### Slow Queries

1. Check for missing indexes
2. Run ANALYZE
3. Consider partitioning large tables (PostgreSQL)

### Disk Full

1. Stop running pipelines
2. Delete old runs (Step 4)
3. Vacuum (Step 5)
4. Consider moving to larger storage

---

## See Also

- [Backup and Recovery](backup-and-recovery.md)
- [Configuration Reference](../reference/configuration.md#landscape-settings-audit-trail)
