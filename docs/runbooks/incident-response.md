# Runbook: Incident Response

Investigate and resolve production pipeline issues.

---

## Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| **P1** | Pipeline down, no data processing | Immediate |
| **P2** | Data quality issues, incorrect routing | < 1 hour |
| **P3** | Performance degradation, slow processing | < 4 hours |
| **P4** | Minor issues, cosmetic problems | Next business day |

---

## P1: Pipeline Down

### Symptoms
- Pipeline process not running
- No new rows being processed
- Health check failing

### Immediate Actions

1. **Check process status:**
   ```bash
   # Find elspeth processes
   ps aux | grep elspeth

   # Check Docker container
   docker ps -a | grep elspeth
   ```

2. **Check logs:**
   ```bash
   # Recent logs
   tail -100 /var/log/elspeth/pipeline.log

   # Docker logs
   docker logs elspeth --tail 100
   ```

3. **Check system resources:**
   ```bash
   # Disk space
   df -h

   # Memory
   free -h

   # CPU
   top -bn1 | head -20
   ```

4. **Check database connectivity:**
   ```bash
   sqlite3 runs/audit.db "SELECT 1;"
   # or
   psql -d elspeth -c "SELECT 1;"
   ```

### Recovery Steps

1. **If process crashed:**
   ```bash
   # Resume from last checkpoint
   elspeth resume <RUN_ID>
   ```

2. **If disk full:**
   ```bash
   # Emergency cleanup
   find /tmp -type f -mtime +1 -delete
   # Then see database-maintenance.md for proper cleanup
   ```

3. **If database locked:**
   ```bash
   # Find locking process
   fuser runs/audit.db
   # Restart or wait
   ```

### Post-Incident

1. Document the incident
2. Identify root cause
3. Implement preventive measures

---

## P2: Data Quality Issues

### Symptoms
- Rows appearing in wrong sink
- Unexpected quarantine volume
- Transform producing wrong values

### Investigation Steps

1. **Identify affected rows:**
   ```bash
   sqlite3 runs/audit.db "
     SELECT r.row_id, r.row_index, to.sink_name
     FROM token_outcomes to
     JOIN tokens t ON to.token_id = t.token_id
     JOIN rows r ON t.row_id = r.row_id
     WHERE to.run_id = '<RUN_ID>'
       AND to.sink_name = '<WRONG_SINK>'
     LIMIT 20;
   "
   ```

2. **Trace a sample row:**
   ```bash
   elspeth explain --run <RUN_ID> --row <ROW_ID>
   ```

3. **Check configuration:**
   ```bash
   sqlite3 runs/audit.db "
     SELECT config FROM runs WHERE run_id = '<RUN_ID>';
   " | python -m json.tool | grep -A10 "gates"
   ```

4. **Compare with expected:**
   - Review gate conditions
   - Check source data format
   - Verify transform logic

### Common Causes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| All rows to one sink | Gate condition always true/false | Check comparison operators |
| Wrong field values | Transform bug | Review transform config |
| High quarantine rate | Schema mismatch | Check source data format |
| Type coercion issues | Missing schema field types | Add explicit types in source schema |

### Remediation

1. Fix configuration
2. Re-run affected data subset
3. Verify with `explain` command

---

## P3: Performance Degradation

### Symptoms
- Processing slower than normal
- High memory usage
- Database queries timing out

### Investigation Steps

1. **Check processing rate:**
   ```bash
   sqlite3 runs/audit.db "
     SELECT
       strftime('%H:%M', created_at) as minute,
       COUNT(*) as rows_per_minute
     FROM rows
     WHERE run_id = '<RUN_ID>'
     GROUP BY minute
     ORDER BY minute DESC
     LIMIT 30;
   "
   ```

2. **Check for bottlenecks:**
   ```bash
   # External API response times (if logged)
   grep "API_CALL" pipeline.log | tail -100

   # Database query times
   sqlite3 runs/audit.db "EXPLAIN QUERY PLAN SELECT * FROM node_states ns JOIN tokens t ON ns.token_id = t.token_id WHERE t.row_id = 'xxx';"
   ```

3. **Check rate limiting:**
   ```bash
   # See if rate limit is throttling
   grep "rate_limit" pipeline.log
   ```

### Common Causes

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Steady slowdown | Database bloat | Run VACUUM, add indexes |
| Sudden slowdown | External API throttling | Check rate limits |
| Memory growth | Large payloads | Check payload sizes |
| CPU spike | Complex transforms | Profile transform code |

### Quick Fixes

1. **Increase workers** (if CPU-bound):
   ```yaml
   concurrency:
     max_workers: 16
   ```

2. **Adjust rate limits** (if API-throttled):
   ```yaml
   rate_limit:
     services:
       external_api:
         requests_per_second: 5
   ```

3. **Vacuum database**:
   ```bash
   sqlite3 runs/audit.db "VACUUM;"
   ```

---

## P4: Minor Issues

### Common Issues

**Logs too verbose:**
- Adjust logging level in config

**Disk usage growing:**
- Review payload retention
- See [Database Maintenance](database-maintenance.md)

**Cosmetic TUI issues:**
- Check terminal compatibility
- Use direct database queries for non-interactive environments

---

## Communication Templates

### Initial Response

```
INCIDENT: [P1/P2/P3] [Brief Description]
TIME: [Timestamp]
STATUS: Investigating

Impact: [What's affected]
Current Action: [What you're doing]
ETA: [When you expect update]
```

### Resolution

```
INCIDENT RESOLVED: [Brief Description]
TIME: [Timestamp]

Root Cause: [What happened]
Resolution: [What fixed it]
Prevention: [What changes will prevent recurrence]
```

---

## Escalation Path

| Condition | Escalate To |
|-----------|-------------|
| P1 not resolved in 30 min | Engineering lead |
| Data integrity concerns | Data team lead + compliance |
| Security incident | Security team |
| External API issues | Vendor support |

---

## Post-Incident Checklist

- [ ] Document timeline of events
- [ ] Identify root cause
- [ ] Document resolution steps
- [ ] Create tickets for follow-up work
- [ ] Update runbooks if needed
- [ ] Schedule post-mortem (for P1/P2)
- [ ] Communicate resolution to stakeholders

---

## See Also

- [Resume Failed Run](resume-failed-run.md)
- [Investigate Routing](investigate-routing.md)
- [Database Maintenance](database-maintenance.md)
