# Runbook: Backup And Recovery

Current as of 2026-05-20.

Back up ELSPETH's audit evidence, payloads, configuration, session state, and
outputs before maintenance or deployment. Landscape audit data is the legal
record; losing it is not equivalent to losing cache.

## What To Back Up

| Component | Typical location | Priority |
|-----------|------------------|----------|
| Landscape audit database | `runs/audit.db`, `data/runs/audit.db`, or configured PostgreSQL URL | Critical |
| Payload store | `.elspeth/payloads/`, `data/payloads/`, or `payload_store.base_path` | High |
| Web session database | `data/sessions.db` or `ELSPETH_WEB__SESSION_DB_URL` | High for Web UI deployments |
| Settings/configuration | The settings file used for each run; deployment env files | High |
| Outputs/artifacts | Example or deployment output directories; object stores | Medium to critical, depending on retention policy |

For staging-specific service layout, see
[staging session database recreation](staging-session-db-recreation.md).

## Locate The Databases

CLI runs usually take their Landscape URL from the `landscape.url` setting:

```bash
python - <<'PY'
from pathlib import Path
from elspeth.core.config import load_settings
settings = load_settings(Path("settings.yaml"))
print(settings.landscape.url)
print(settings.payload_store.base_path)
PY
```

Web deployments default to `data/runs/audit.db`, `data/payloads/`, and
`data/sessions.db` under `ELSPETH_WEB__DATA_DIR` unless explicit URLs/paths are
configured.

## SQLite Backup

Use SQLite's online backup command for a running database:

```bash
mkdir -p /backups/elspeth
sqlite3 runs/audit.db ".backup '/backups/elspeth/audit-$(date +%Y%m%dT%H%M%S).db'"
```

For Web UI deployments, also back up the session database:

```bash
sqlite3 data/sessions.db ".backup '/backups/elspeth/sessions-$(date +%Y%m%dT%H%M%S).db'"
```

Back up payloads and settings:

```bash
tar -czf "/backups/elspeth/payloads-$(date +%Y%m%dT%H%M%S).tar.gz" data/payloads
cp settings.yaml "/backups/elspeth/settings-$(date +%Y%m%dT%H%M%S).yaml"
```

Do not print or archive secret values into broad-access locations. Deployment
environment files may contain secrets; store those backups in a restricted
secret-management location.

## PostgreSQL Backup

Use logical dumps for routine recovery:

```bash
pg_dump -Fc -d "$ELSPETH_DATABASE_URL" -f "/backups/elspeth/landscape-$(date +%Y%m%dT%H%M%S).dump"
```

Use your PostgreSQL platform's WAL/PITR process for point-in-time recovery. Do
not rely on a generic `recovery.conf` snippet; PostgreSQL recovery configuration
varies by version and hosting platform.

## Restore SQLite

1. Stop the application or pipeline writers.
2. Archive the current possibly-bad database before overwriting it.
3. Restore the selected backup.
4. Verify integrity.
5. Restart the application or resume the affected run only after verification.

```bash
cp runs/audit.db "runs/audit.db.before-restore-$(date +%Y%m%dT%H%M%S)"
cp /backups/elspeth/audit-20260520T020000.db runs/audit.db
sqlite3 runs/audit.db "PRAGMA integrity_check;"
```

Restore payloads:

```bash
tar -xzf /backups/elspeth/payloads-20260520T020000.tar.gz -C /
```

If the restore is for an interrupted CLI run, dry-run resume first:

```bash
elspeth resume <RUN_ID> --database runs/audit.db
elspeth resume <RUN_ID> --database runs/audit.db --execute
```

## Verify A Backup

```bash
BACKUP=/backups/elspeth/audit-latest.db

test -f "$BACKUP"
sqlite3 "$BACKUP" "PRAGMA integrity_check;"
sqlite3 "$BACKUP" "SELECT COUNT(*) FROM runs;"
sqlite3 "$BACKUP" "SELECT COUNT(*) FROM tokens;"
sqlite3 "$BACKUP" "SELECT COUNT(*) FROM token_outcomes;"
```

For a recent run, verify explain still works:

```bash
elspeth explain --run latest --database "$BACKUP" --json
```

## Partial Data Loss

Missing payload files do not invalidate stored hashes, but they do limit what
operators can rehydrate from the audit trail. Record the limitation in incident
notes and rerun affected source data if payload content is required for review.

If audit rows are corrupt, prefer restoring the whole database from a known-good
backup. Do not manually coerce or patch Tier-1 audit rows to make read paths
green.

## Retention Policy

Set retention by deployment policy, legal requirements, and storage cost. The
default payload retention setting is 90 days, but audit database retention is a
compliance decision rather than a framework constant.

## See Also

- [Database Maintenance](database-maintenance.md)
- [Incident Response](incident-response.md)
- [Resume Failed Run](resume-failed-run.md)
- [Configuration Reference](../reference/configuration.md)
