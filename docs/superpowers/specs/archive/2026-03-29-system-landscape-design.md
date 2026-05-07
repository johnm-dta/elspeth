# System Landscape — Platform-Level Audit Trail

**Status:** Draft
**Date:** 2026-03-29
**Relates to:** `elspeth-d12948c164` (auth events not recorded)
**Depends on:** Web UX Composer MVP (must land first)
**Branch:** TBD (post web-ux)

---

## Overview

ELSPETH has a Pipeline Landscape — a per-run audit trail that records every
row-level decision during pipeline execution. It answers "what happened to
this data?"

The System Landscape is a new, logically separate audit trail that records
platform-level events: authentication, session lifecycle, run initiation,
and system health. It answers "who did what and when?"

These are two distinct audit domains with different write patterns, retention
policies, and query surfaces. They cross-reference via `landscape_run_id`.

---

## Why Two Landscapes

| Concern | Pipeline Landscape | System Landscape |
|---------|-------------------|-----------------|
| Purpose | Execution audit trail | Platform event log |
| Scope | Per-run | Per-deployment |
| Lifecycle | Created per pipeline run | Always running |
| Data | Rows, tokens, routing, node states | Auth, sessions, run attribution, system lifecycle |
| Write pattern | High-volume burst during execution | Low-volume continuous |
| Retention | Tied to run retention policy | Independent (compliance-driven) |
| Query pattern | "Explain row 42 in run X" | "Who authenticated at 3am?" / "Who triggered run X?" |
| Existing? | Yes (`core/landscape/`) | **New** |

The Pipeline Landscape remains unchanged. The System Landscape is additive.

---

## Relationship to Semi-Autonomous Platform

The semi-autonomous design (`docs/architecture/semi-autonomous/design.md`)
defines a typed task event log for governance events: `PromptSubmitted`,
`ArtifactCreated`, `PreviewExecuted`, `ApprovalGranted`, etc.

The System Landscape is that event log — it starts with auth and run
attribution events in v1, and grows into the full governance event log when
approval workflows arrive in v2.

| Semi-Autonomous Event | System Landscape Event (v1) | Added In |
|-----------------------|-----------------------------|----------|
| — | `auth.login` | v1 |
| — | `auth.login_failed` | v1 |
| — | `auth.token_issued` | v1 |
| — | `auth.token_expired` | v1 |
| — | `run.initiated` | v1 |
| — | `run.completed` | v1 |
| — | `session.created` | v1 |
| — | `session.archived` | v1 |
| `PromptSubmitted` | `composition.message_sent` | v2 |
| `ArtifactCreated` | `composition.state_sealed` | v2 |
| `PreviewExecuted` | `run.preview_executed` | v2 |
| `ApprovalGranted` | `governance.approval_granted` | v2 |

---

## Event Schema

All system events share a common envelope:

| Field | Type | Purpose |
|-------|------|---------|
| event_id | UUID | Primary key |
| event_type | str | Dot-namespaced type (e.g., `auth.login`) |
| timestamp | datetime | UTC, server clock |
| actor | str | User ID or system identifier |
| actor_ip | str or null | Client IP (for auth events) |
| session_id | UUID or null | Web session (when applicable) |
| details | JSON | Event-type-specific payload |
| correlation_id | str or null | Request correlation ID |

### v1 Event Types

**Auth events:**

| Event Type | Details Payload | When Recorded |
|------------|----------------|---------------|
| `auth.login` | `{provider: str, method: "password" \| "oidc" \| "entra"}` | Successful login |
| `auth.login_failed` | `{provider: str, reason: str}` | Failed login (no password in payload) |
| `auth.token_issued` | `{provider: str, expires_at: datetime}` | JWT issued or refreshed |
| `auth.token_expired` | `{token_hash: str}` | Token validation fails due to expiry |

**Run attribution events:**

| Event Type | Details Payload | When Recorded |
|------------|----------------|---------------|
| `run.initiated` | `{session_id: UUID, state_id: UUID, state_version: int, landscape_run_id: str}` | User clicks Execute |
| `run.completed` | `{landscape_run_id: str, status: str, rows_processed: int, rows_failed: int, duration_seconds: float}` | Pipeline finishes |

**Session events:**

| Event Type | Details Payload | When Recorded |
|------------|----------------|---------------|
| `session.created` | `{title: str}` | New session |
| `session.archived` | `{}` | Session archived |

---

## Architecture

### SystemLandscape Class

A new `SystemLandscape` class, separate from the existing `LandscapeDB` /
`LandscapeRecorder`. Different responsibilities, different schema, different
lifecycle.

| Concern | Decision |
|---------|----------|
| Database | Separate SQLite/Postgres DB, configured via `WebSettings.system_landscape_url` |
| Schema | Single `system_events` table with the envelope above |
| Schema management | `metadata.create_all()` on startup; Alembic for production |
| Write API | `record(event_type, actor, details, **kwargs)` — synchronous, crash-on-failure (Tier 1) |
| Read API | Query by event_type, actor, time range, session_id, correlation_id |
| Layer | L1 (`core/`) — alongside the existing Landscape |
| Thread safety | Connection-per-call (same as existing Landscape) |

### Configuration

Add to `WebSettings`:

| Field | Type | Default |
|-------|------|---------|
| `system_landscape_url` | `str \| None` | `None` → derives `sqlite:///{data_dir}/system_audit.db` |

Add `get_system_landscape_url()` derived accessor, same pattern as
`get_landscape_url()` and `get_session_db_url()`.

### Integration Points

| Module | What It Records | How |
|--------|----------------|-----|
| `web/auth/middleware.py` | `auth.token_expired` on 401 | After `AuthenticationError` is raised |
| `web/auth/routes.py` | `auth.login`, `auth.login_failed`, `auth.token_issued` | In login/token route handlers |
| `web/execution/service.py` | `run.initiated`, `run.completed` | In `execute()` and `_run_pipeline()` |
| `web/sessions/routes.py` | `session.created`, `session.archived` | In create/archive route handlers |
| `web/app.py` (lifespan) | SystemLandscape constructed and stored on `app.state` | During lifespan startup |

### Tier Model Compliance

The System Landscape is Tier 1 (our data, full trust):

- Written by our code, read by our code
- Bad data = crash immediately
- No coercion, no defaults, no silent recovery
- The `details` JSON is structured by our code, not user input
- `actor_ip` comes from the HTTP request (Tier 3 at the boundary, but
  validated to a string and stored as Tier 1 — we record what we saw)

---

## Cross-Reference Model

The two Landscapes cross-reference through `landscape_run_id`:

```
System Landscape                     Pipeline Landscape
─────────────────                    ──────────────────
run.initiated                        runs table
  landscape_run_id: "abc-123" ─────► run_id: "abc-123"
  actor: "alice"                       status: COMPLETED
  session_id: "sess-456"              rows_processed: 10000
                                       node_states, token_outcomes, etc.

run.completed
  landscape_run_id: "abc-123"
  status: "completed"
  rows_processed: 10000
```

An auditor asking "who ran pipeline abc-123?" queries the System Landscape.
An auditor asking "what did pipeline abc-123 do?" queries the Pipeline Landscape.

---

## What This Spec Does NOT Cover

- MCP analysis server integration (read-only access to system events)
- TUI `explain` command for system events
- Governance events (approval workflows — v2)
- System health events (startup, shutdown, errors — future)
- Event retention policies (separate from pipeline retention)
- Event export to external systems (SIEM, log aggregators)

These are natural follow-ons but not in scope for the initial implementation.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/core/landscape/system_landscape.py` | SystemLandscape class, schema, record method |
| Modify | `src/elspeth/web/config.py` | Add `system_landscape_url`, `get_system_landscape_url()` |
| Modify | `src/elspeth/web/app.py` | Construct SystemLandscape in lifespan |
| Modify | `src/elspeth/web/auth/middleware.py` | Record `auth.token_expired` |
| Modify | `src/elspeth/web/auth/routes.py` | Record `auth.login`, `auth.login_failed`, `auth.token_issued` |
| Modify | `src/elspeth/web/execution/service.py` | Record `run.initiated`, `run.completed` |
| Modify | `src/elspeth/web/sessions/routes.py` | Record `session.created`, `session.archived` |
| Create | `tests/unit/core/landscape/test_system_landscape.py` | SystemLandscape unit tests |
| Create | `tests/integration/web/test_system_audit.py` | End-to-end: login → create session → execute → verify all events recorded |

---

## Acceptance Criteria

1. `SystemLandscape.record()` persists an event with all envelope fields
2. All 8 v1 event types are recorded at the specified integration points
3. `auth.login_failed` does not contain passwords or tokens
4. `run.initiated` contains `landscape_run_id` linking to the Pipeline Landscape
5. `run.completed` contains final row counts matching the Pipeline Landscape
6. System Landscape DB is separate from Pipeline Landscape and Session DB
7. `WebSettings.system_landscape_url` defaults to `{data_dir}/system_audit.db`
8. SystemLandscape is constructed in the app lifespan, not `create_app()`
9. Bad data written to the System Landscape crashes immediately (Tier 1)
10. Integration test verifies the full auth → execute → audit trail
