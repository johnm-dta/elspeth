# Runbook: Backup and Recovery

Backup audit trail and recover from data loss.

---

## What to Backup

| Component | Location | Priority |
|-----------|----------|----------|
| Audit database | `runs/audit.db` or PostgreSQL | **Critical** |
| Payload store | `.elspeth/payloads/` | High |
| Configuration | `pipeline.yaml` | High |
| Output files | `output/` | Medium (can regenerate) |

---

## Backup Procedures

### SQLite Backup

**Online backup (while pipeline may be running):**

```bash
sqlite3 runs/audit.db ".backup 'runs/audit.db.backup'"
```

**Scheduled backup script:**

```bash
#!/bin/bash
# backup-elspeth.sh

BACKUP_DIR="/backups/elspeth"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup audit database
sqlite3 runs/audit.db ".backup '$BACKUP_DIR/audit_$DATE.db'"

# Backup payload store
tar -czf "$BACKUP_DIR/payloads_$DATE.tar.gz" .elspeth/payloads/

# Backup configuration
cp pipeline.yaml "$BACKUP_DIR/pipeline_$DATE.yaml"

# Keep only last 30 days
find "$BACKUP_DIR" -type f -mtime +30 -delete

echo "Backup completed: $DATE"
```

**Cron schedule (daily at 2 AM):**

```bash
0 2 * * * /opt/elspeth/backup-elspeth.sh >> /var/log/elspeth-backup.log 2>&1
```

### PostgreSQL Backup

**Logical backup (pg_dump):**

```bash
pg_dump -Fc -d elspeth -f /backups/elspeth_$(date +%Y%m%d).dump
```

**Point-in-time recovery setup:**

```bash
# Enable WAL archiving in postgresql.conf
archive_mode = on
archive_command = 'cp %p /backups/wal/%f'

# Base backup
pg_basebackup -D /backups/base -Fp -Xs -P
```

### Docker Backup

When running in Docker, backup the mounted volumes:

```bash
# Stop container for consistent backup
docker compose stop elspeth

# Backup volumes
tar -czf elspeth_backup_$(date +%Y%m%d).tar.gz \
  ./config \
  ./state \
  ./output

# Restart container
docker compose start elspeth
```

---

## Recovery Procedures

### SQLite Recovery

**Restore from backup:**

```bash
# Stop any running pipelines
pkill -f elspeth

# Restore database
cp /backups/elspeth/audit_20240115.db runs/audit.db

# Verify integrity
sqlite3 runs/audit.db "PRAGMA integrity_check;"

# Restore payloads
tar -xzf /backups/elspeth/payloads_20240115.tar.gz
```

**Recover from corruption:**

```bash
# Attempt recovery
sqlite3 runs/audit.db ".recover" | sqlite3 runs/audit_recovered.db

# Verify recovered database
sqlite3 runs/audit_recovered.db "SELECT COUNT(*) FROM runs;"

# Replace if valid
mv runs/audit_recovered.db runs/audit.db
```

### PostgreSQL Recovery

**Restore from pg_dump:**

```bash
# Create fresh database
dropdb elspeth
createdb elspeth

# Restore
pg_restore -d elspeth /backups/elspeth_20240115.dump
```

**Point-in-time recovery:**

```bash
# Stop PostgreSQL
pg_ctl stop -D /var/lib/postgresql/data

# Restore base backup
rm -rf /var/lib/postgresql/data/*
tar -xf /backups/base.tar -C /var/lib/postgresql/data/

# Create recovery.conf
cat > /var/lib/postgresql/data/recovery.conf << EOF
restore_command = 'cp /backups/wal/%f %p'
recovery_target_time = '2024-01-15 14:30:00'
EOF

# Start PostgreSQL
pg_ctl start -D /var/lib/postgresql/data
```

### Docker Recovery

```bash
# Stop container
docker compose stop elspeth

# Restore volumes
tar -xzf elspeth_backup_20240115.tar.gz

# Restart
docker compose start elspeth

# Verify
docker compose run --rm elspeth health --verbose
```

---

## Disaster Recovery

### Complete Environment Loss

1. **Provision new infrastructure**
2. **Restore configuration:**
   ```bash
   cp /backups/pipeline_latest.yaml pipeline.yaml
   ```
3. **Restore audit database:**
   ```bash
   cp /backups/audit_latest.db runs/audit.db
   ```
4. **Restore payload store:**
   ```bash
   tar -xzf /backups/payloads_latest.tar.gz
   ```
5. **Verify system:**
   ```bash
   elspeth health --verbose
   elspeth validate --settings pipeline.yaml
   ```
6. **Resume any incomplete runs:**
   ```bash
   sqlite3 runs/audit.db "SELECT run_id FROM runs WHERE status = 'running';"
   elspeth resume <RUN_ID>
   ```

### Partial Data Loss

**Missing payloads:**
- Audit hashes remain valid (hashes survive payload deletion)
- Re-run affected data if payload content needed

**Corrupted rows:**
- Export valid rows
- Create new database
- Import valid data
- Re-run affected source data

---

## Backup Verification

### Weekly Verification Checklist

```bash
#!/bin/bash
# verify-backup.sh

BACKUP="/backups/elspeth/audit_latest.db"

# Check file exists
if [ ! -f "$BACKUP" ]; then
  echo "FAIL: Backup file not found"
  exit 1
fi

# Check integrity
INTEGRITY=$(sqlite3 "$BACKUP" "PRAGMA integrity_check;" 2>&1)
if [ "$INTEGRITY" != "ok" ]; then
  echo "FAIL: Integrity check failed: $INTEGRITY"
  exit 1
fi

# Check recent data
RECENT=$(sqlite3 "$BACKUP" "SELECT COUNT(*) FROM runs WHERE started_at > datetime('now', '-7 days');")
if [ "$RECENT" -eq 0 ]; then
  echo "WARN: No runs in last 7 days"
fi

# Check table counts
echo "Runs: $(sqlite3 "$BACKUP" "SELECT COUNT(*) FROM runs;")"
echo "Tokens: $(sqlite3 "$BACKUP" "SELECT COUNT(*) FROM tokens;")"
echo "States: $(sqlite3 "$BACKUP" "SELECT COUNT(*) FROM node_states;")"

echo "OK: Backup verification passed"
```

### Monthly Recovery Test

1. Restore backup to test environment
2. Run `elspeth health --verbose`
3. Run `elspeth explain --run <recent_run> --row 1`
4. Verify data integrity
5. Document results

---

## Retention Policy

| Data Type | Retention | Rationale |
|-----------|-----------|-----------|
| Audit database | 7 years | Compliance requirement |
| Payload store | 90 days | Storage cost |
| Daily backups | 30 days | Operational recovery |
| Weekly backups | 90 days | Extended recovery |
| Monthly backups | 7 years | Compliance archives |

---

## Monitoring Alerts

Configure alerts for:

- [ ] Backup job failure
- [ ] Backup size anomaly (sudden growth/shrink)
- [ ] Backup age > 24 hours
- [ ] Backup integrity check failure
- [ ] Disk space < 20% on backup storage

---

## See Also

- [Database Maintenance](database-maintenance.md)
- [Incident Response](incident-response.md)
- [Configuration Reference](../reference/configuration.md)
