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
  SELECT 'row_events', COUNT(*) FROM row_events
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
         (SELECT COUNT(*) FROM row_events WHERE row_events.run_id = runs.run_id) as event_count
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

# Export all row events for a run
sqlite3 -header -csv runs/audit.db "
  SELECT * FROM row_events WHERE run_id = '<RUN_ID>';
" > row_events_archive.csv
```

### Step 4: Delete Old Data

**CAUTION:** Always backup before deletion.

```bash
# Backup first
cp runs/audit.db runs/audit.db.backup.$(date +%Y%m%d)

# Delete runs older than 90 days (cascades to related tables)
sqlite3 runs/audit.db "
  -- Delete checkpoints
  DELETE FROM checkpoints WHERE run_id IN (
    SELECT run_id FROM runs WHERE started_at < datetime('now', '-90 days')
  );

  -- Delete row events
  DELETE FROM row_events WHERE run_id IN (
    SELECT run_id FROM runs WHERE started_at < datetime('now', '-90 days')
  );

  -- Delete artifacts
  DELETE FROM artifacts WHERE run_id IN (
    SELECT run_id FROM runs WHERE started_at < datetime('now', '-90 days')
  );

  -- Delete runs
  DELETE FROM runs WHERE started_at < datetime('now', '-90 days');
"
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
-- Index for querying by run
CREATE INDEX IF NOT EXISTS idx_row_events_run_id ON row_events(run_id);

-- Index for querying by row within run
CREATE INDEX IF NOT EXISTS idx_row_events_run_row ON row_events(run_id, row_id);

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
