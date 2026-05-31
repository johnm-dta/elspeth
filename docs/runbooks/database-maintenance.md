# Runbook: Database Maintenance

Current as of 2026-05-20.

Maintain the Landscape audit database and payload store without damaging
ELSPETH's audit evidence.

## When To Use

- The Landscape database is growing quickly.
- Query performance has degraded.
- Payload storage is approaching retention or capacity limits.
- A deployment needs pre-maintenance backup and verification.
- Historical audit evidence needs a retention decision.

## Before Any Maintenance

1. Stop active writers or confirm the target database is a restored copy.
2. Back up the database and payload store.
3. Confirm legal/compliance retention requirements.
4. Record any accepted audit limitation before deleting evidence.

See [Backup and Recovery](backup-and-recovery.md) for backup commands.

## Assess Current State

SQLite database size:

```bash
ls -lh runs/audit.db
```

Core row counts:

```bash
sqlite3 runs/audit.db "
  SELECT 'runs' AS table_name, COUNT(*) AS row_count FROM runs
  UNION ALL SELECT 'rows', COUNT(*) FROM rows
  UNION ALL SELECT 'tokens', COUNT(*) FROM tokens
  UNION ALL SELECT 'token_outcomes', COUNT(*) FROM token_outcomes
  UNION ALL SELECT 'node_states', COUNT(*) FROM node_states
  UNION ALL SELECT 'calls', COUNT(*) FROM calls
  UNION ALL SELECT 'artifacts', COUNT(*) FROM artifacts;
"
```

Runs by date:

```bash
sqlite3 runs/audit.db "
  SELECT DATE(started_at) AS run_date, COUNT(*) AS run_count
  FROM runs
  GROUP BY DATE(started_at)
  ORDER BY run_date DESC
  LIMIT 30;
"
```

PostgreSQL table sizes:

```bash
psql -d elspeth -c "
  SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS size
  FROM pg_tables
  WHERE schemaname = 'public'
  ORDER BY pg_total_relation_size(schemaname || '.' || tablename) DESC;
"
```

## Payload Retention

Use the maintained `elspeth purge` command for payload retention. Do not delete
payload files with `find -delete`; the purge command discovers references from
Landscape and updates reproducibility grades where applicable.

Dry-run:

```bash
elspeth purge --dry-run --database runs/audit.db --payload-dir .elspeth/payloads
```

Execute:

```bash
elspeth purge \
  --database runs/audit.db \
  --payload-dir .elspeth/payloads \
  --retention-days 90 \
  --yes
```

If the payload store path is configured in `settings.yaml`, the command can read
it from settings when run from the pipeline directory.

## Audit Database Retention

Do not hand-delete rows from Landscape tables as routine maintenance. Landscape
uses composite keys, partial indexes, cross-table evidence, and Tier-1 integrity
assumptions; ad hoc deletes can produce a plausible but false audit story.

If an environment must retire audit database records, create a dedicated
retention procedure that:

- states the legal retention basis
- snapshots the full database before deletion
- runs against a restored copy first
- proves foreign-key integrity after deletion
- proves token-outcome completeness after deletion
- records any accepted audit limitation in release or incident evidence

Use [Token Outcome Assurance](../contracts/token-outcomes/README.md) to verify
token lifecycle integrity after any retention migration.

## Vacuum And Analyze

Run these after supported maintenance operations.

SQLite:

```bash
sqlite3 runs/audit.db "PRAGMA integrity_check;"
sqlite3 runs/audit.db "VACUUM;"
sqlite3 runs/audit.db "ANALYZE;"
```

PostgreSQL:

```bash
psql -d elspeth -c "VACUUM ANALYZE;"
```

## PostgreSQL-Specific Checks

Check connection count:

```sql
SELECT count(*) FROM pg_stat_activity WHERE datname = 'elspeth';
```

Check bloat/table size:

```sql
SELECT
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size,
  pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) AS table_size
FROM pg_tables
WHERE schemaname = 'public';
```

Reindex only during a planned maintenance window:

```bash
psql -d elspeth -c "REINDEX DATABASE elspeth;"
```

## Historical Semantic Boundary

Runs created before the rows-routed counter split and before ADR-019's two-axis
terminal model require date/commit-context qualification. Do not interpret old
single-axis `routed` evidence as if it used the current
`TerminalOutcome`/`TerminalPath` model.

If old Landscape data must be preserved for audit, retain the database snapshot
and document the limitation rather than migrating it silently.

## Maintenance Schedule

| Task | Frequency | Command |
|------|-----------|---------|
| Check database size | Weekly | `ls -lh runs/audit.db` |
| Check row counts | Weekly | Core row-count SQL above |
| Payload purge dry-run | Monthly | `elspeth purge --dry-run --database runs/audit.db` |
| Payload purge | Per retention policy | `elspeth purge --database runs/audit.db --retention-days 90 --yes` |
| SQLite integrity check | Before/after maintenance | `sqlite3 runs/audit.db "PRAGMA integrity_check;"` |
| SQLite vacuum/analyze | After supported deletion | `sqlite3 runs/audit.db "VACUUM; ANALYZE;"` |

## Troubleshooting

### Database Locked

```bash
fuser runs/audit.db
```

Wait for the active writer to finish. Kill a process only under incident
response procedure.

### Slow Queries

1. Run `ANALYZE`.
2. Check query plans.
3. For PostgreSQL, inspect table size and indexes.
4. For SQLite, consider whether the workload should move to PostgreSQL.

### Disk Full

1. Stop active writers.
2. Take a backup if possible.
3. Run payload purge dry-run and then purge if policy permits.
4. Move backups/payloads to larger storage.
5. Vacuum only after supported deletion.

## See Also

- [Backup and Recovery](backup-and-recovery.md)
- [Incident Response](incident-response.md)
- [Configuration Reference](../reference/configuration.md#landscape-settings-audit-trail)
