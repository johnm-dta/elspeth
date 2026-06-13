# Runbooks

Operational procedures for ELSPETH pipeline management.

---

## Quick Reference

| Runbook | When to Use |
|---------|-------------|
| [Resume Failed Run](resume-failed-run.md) | Pipeline crashed or was interrupted |
| [Investigate Routing](investigate-routing.md) | Need to explain why a row was routed |
| [Scheduler Lease Recovery](scheduler-lease-recovery.md) | Token work items stuck `leased`, SCREAM invariant fired, `attempt` churn from lease expiries, or (N>1) dead-leader takeover, a wedged lock-holder, or follower recovery |
| [Database Maintenance](database-maintenance.md) | Audit DB growing large, need cleanup |
| [Incident Response](incident-response.md) | Production issue needs investigation |
| [Backup and Recovery](backup-and-recovery.md) | Backup audit trail, restore from backup |
| [Ansible Ubuntu Deployment](ansible-ubuntu-deployment.md) | Automate Ubuntu 24.04/22.04 VM, Azure Front Door, and Azure container deployments |
| [Audit Tier-1 Violation](audit-tier1-violation.md) | Compose-loop audit counters or audit-grade transcript logging fail |

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
