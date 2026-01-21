# Runbooks

Operational procedures for ELSPETH pipeline management.

---

## Quick Reference

| Runbook | When to Use |
|---------|-------------|
| [Resume Failed Run](resume-failed-run.md) | Pipeline crashed or was interrupted |
| [Investigate Routing](investigate-routing.md) | Need to explain why a row was routed |
| [Database Maintenance](database-maintenance.md) | Audit DB growing large, need cleanup |
| [Incident Response](incident-response.md) | Production issue needs investigation |
| [Backup and Recovery](backup-and-recovery.md) | Backup audit trail, restore from backup |

---

## Common Tasks

### Check Pipeline Status

```bash
# Validate configuration
elspeth validate --settings pipeline.yaml

# List recent runs
sqlite3 runs/audit.db "SELECT run_id, status, started_at FROM runs ORDER BY started_at DESC LIMIT 10;"
```

### Quick Health Check

```bash
elspeth health --verbose
```

### View Available Plugins

```bash
elspeth plugins list
```

---

## Emergency Contacts

> **⚠️ Customize This Section:** Replace these generic contacts with your organization's actual contacts before deploying these runbooks.

| Issue | Contact |
|-------|---------|
| Pipeline failures | On-call engineer (e.g., PagerDuty, Slack #oncall) |
| Data integrity concerns | Data team lead |
| Audit trail questions | Compliance team |

---

## See Also

- [Configuration Reference](../reference/configuration.md)
- [Docker Guide](../guides/docker.md)
- [Your First Pipeline](../guides/your-first-pipeline.md)
