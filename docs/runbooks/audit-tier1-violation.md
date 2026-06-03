# Audit Tier-1 Violation Runbook

Use this runbook when any compose-loop audit counter that protects the chat
transcript or audit-grade transcript access becomes non-zero.

## Signals

| Counter | Severity | Meaning |
|---------|----------|---------|
| `composer.audit.tool_row_tier1_violation_total` | Critical | A compose-loop tool call succeeded, but the audit write failed. The system did work it cannot prove in the transcript. |
| `composer.audit.tool_row_integrity_violation_total` | Critical | A schema or writer invariant rejected an audit row, such as a duplicate tool-call id, invalid writer principal, or sequence collision. |
| `composer.audit.audit_access_log_write_failed_total` | High | A request for audit-grade transcript rows could not be recorded in `audit_access_log`; the API must fail closed and return no transcript rows. |
| `composer.audit.tool_row_persist_failed_during_unwind_total` | High | A plugin crashed and the best-effort failure-unwind audit write also failed. The plugin error remains primary, but audit evidence is incomplete. |
| `composer.audit.audit_grade_view_total` | Informational | Count of successful audit-grade transcript reads, currently `GET /api/sessions/{id}/messages?include_tool_rows=true`. |
| `composer.tool_call_cap_exceeded_total` | Informational | The composer emitted more tool calls in one turn than the configured cap and the turn was rejected before tool execution. |

The SLO for the two critical counters is zero. Treat any non-zero increment as
an incident until the affected session and database state are understood.

## Immediate Response

1. Stop retry loops for the affected session. Do not re-run compose turns to
   "fill in" missing rows; duplicate tool-call ids can create additional
   integrity failures.
2. Capture the deployment version, request id, session id, and counter name from
   the alert or logs. Do not copy tool payloads or row-level transcript content
   into the incident channel.
3. Check the API response contract:
   - `AuditIntegrityError` returns HTTP 500 with `error_type:
     audit_integrity_error` and no `messages` or `tool_rows` body fields.
   - `StaleComposeStateError` returns HTTP 409 with `error_type:
     stale_compose_state`.
   - `AuditAccessLogWriteError` returns HTTP 500 with `error_type:
     audit_access_log_write_failed` and no `messages` body field.
4. If `audit_access_log_write_failed_total` moved, verify that the transcript
   endpoint failed closed. Tool rows must not be returned unless the
   audit-grade view write succeeded.
5. Preserve the session database before manual repair attempts:

```bash
sqlite3 data/sessions.db ".backup '/tmp/elspeth-sessions-incident-$(date +%Y%m%dT%H%M%S).db'"
```

Adjust the database path for the deployment. For staging, inspect
`deploy/elspeth-web.env` without printing secret values.

## Triage Queries

Run these against the session database copy or under a maintenance window.

```sql
-- Tool rows without a same-session assistant parent.
SELECT tool.id, tool.session_id, tool.tool_call_id, tool.parent_assistant_id
  FROM chat_messages tool
  LEFT JOIN chat_messages assistant
    ON assistant.id = tool.parent_assistant_id
   AND assistant.session_id = tool.session_id
   AND assistant.role = 'assistant'
 WHERE tool.role = 'tool'
   AND assistant.id IS NULL;

-- Tool-call state rows without a matching tool transcript row.
SELECT cs.id, cs.session_id, cs.version
  FROM composition_states cs
  LEFT JOIN chat_messages cm
    ON cm.composition_state_id = cs.id
   AND cm.role = 'tool'
 WHERE cs.provenance = 'tool_call'
   AND cs.version > 0
   AND cm.id IS NULL;

-- Duplicate tool-call ids in one session.
SELECT session_id, tool_call_id, COUNT(*) AS count
  FROM chat_messages
 WHERE role = 'tool'
 GROUP BY session_id, tool_call_id
HAVING COUNT(*) > 1;

-- Audit-grade transcript reads that were recorded.
SELECT session_id, actor, route_name, created_at
  FROM audit_access_log
 ORDER BY created_at DESC
 LIMIT 20;
```

## Expected Recovery Shape

- For `tool_row_tier1_violation_total`, keep the compose result failed. Do not
  synthesize assistant/tool rows after the fact. Preserve the database and open a
  bug with the chained DB exception class and the failing helper path.
- For `tool_row_integrity_violation_total`, identify the invariant that rejected
  the row. Common causes are duplicate `tool_call_id`, stale compose state, or a
  writer outside `compose_loop` using the compose-turn primitive.
- For `audit_access_log_write_failed_total`, repair the `audit_access_log` write
  path first. The transcript endpoint may be retried only after the logging write
  succeeds.
- For `tool_row_persist_failed_during_unwind_total`, investigate both the
  original plugin crash and the audit-write failure. The plugin crash is the
  primary user-facing failure, but the audit failure is still operationally
  urgent.

## Production-Deploy Follow-Ups

This repository currently has telemetry documentation and runbooks, but no
deployable Alertmanager/Grafana dashboard configuration under `config/` or
`infra/` for this PR to patch. These Filigree tasks must be resolved before
Phase 3 ships to production:

- `elspeth-6e55a05547` — alert route for compose-loop Tier-1 audit failures.
- `elspeth-2255661995` — dashboard visibility for compose-loop audit counters.
