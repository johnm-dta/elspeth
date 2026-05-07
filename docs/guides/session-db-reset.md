# Session DB Reset Runbook

Use this runbook when a web session schema-bootstrap change requires deleting or archiving a stale `sessions.db`. The session database is separate from the Landscape audit database; resetting the session database must not touch Landscape, payload storage, blobs, or Filigree tracker data.

## Stop/Go Gates

Before any staging reset, verify that the current Landscape schema and Landscape write/read code do not reference web-session identifiers:

```bash
rg -n "session_id|chat_message_id|composition_state_id" src/elspeth/core/landscape
```

Expected for the current architecture: no output.

If this command finds a reference, stop. Inspect the table/column and decide whether deleting `sessions.db` would orphan Landscape audit rows. Do not reset the session database until the owning issue has explicit orphaning analysis and a preservation plan.

For SQLite deployments, also inspect the live Landscape database schema after resolving the active Landscape URL:

```bash
sqlite3 /path/to/audit.db ".schema" | grep -E "session_id|chat_message_id|composition_state_id"
```

Expected for the current architecture: no output.

If this command prints any table definition, stop and preserve both databases until the relationship is understood.

## Resolve Database Paths

Resolve the active session DB from `WebSettings.get_session_db_url()` semantics:

- If `ELSPETH_WEB__SESSION_DB_URL` is set, that is the session database URL.
- Otherwise, if `ELSPETH_WEB__DATA_DIR` is set, the default session DB is `${ELSPETH_WEB__DATA_DIR}/sessions.db`.
- Otherwise, the code default is `data/sessions.db` relative to the process working directory.

Resolve the active Landscape DB separately:

- If `ELSPETH_WEB__LANDSCAPE_URL` is set, that is the Landscape URL.
- Otherwise, the default Landscape DB is `${ELSPETH_WEB__DATA_DIR}/runs/audit.db`, or `data/runs/audit.db` if `ELSPETH_WEB__DATA_DIR` is unset.

Never print secret values from `deploy/elspeth-web.env`. It is acceptable to print only the derived file paths after redacting credentials and confirming they are SQLite paths.

## Local Or Dev Reset

1. Stop the local web process or ensure no process is using the session DB.
2. Confirm the path is the session DB, not Landscape:

   ```bash
   sqlite3 /path/to/sessions.db ".tables"
   ```

   Expected session tables include `sessions`, `chat_messages`, `composition_states`, `runs`, `run_events`, `blobs`, `blob_run_links`, and `user_secrets`.

3. Archive the session DB before deleting it:

   ```bash
   cp /path/to/sessions.db /path/to/sessions.db.before-reset
   ```

4. Delete or move only the confirmed session DB file.
5. Start the web process. `initialize_session_schema()` recreates the DB with current metadata.
6. Create a new session through the web API or UI. A startup `SessionSchemaError` means a stale session DB is still being used.

## Staging Reset For `elspeth.foundryside.dev`

The staging site is a source-checkout systemd/Caddy deployment from `/home/john/elspeth`, not the generic VM/Docker flow.

1. Use a host shell with systemd access. The Codex sandbox may be unable to access `systemctl` or `sudo`.
2. Inspect `deploy/elspeth-web.env` directly for `ELSPETH_WEB__SESSION_DB_URL`, `ELSPETH_WEB__DATA_DIR`, and `ELSPETH_WEB__LANDSCAPE_URL` without printing secret values.
3. Run both Stop/Go Gates above against the current checkout and live Landscape DB.
4. Stop or quiesce `elspeth-web.service`.
5. Archive/delete only the confirmed session DB. Do not delete Landscape, payloads, blobs, or `.filigree`.
6. Restart with the approved host-side restart mechanism.
7. Verify health:

   ```bash
   curl --unix-socket /run/elspeth/uvicorn.sock -fsS http://localhost/api/health
   curl -fsS https://elspeth.foundryside.dev/api/health
   ```

8. Create a new session and confirm no `SessionSchemaError` appears in the service journal.

## Failure Handling

- If the stop/go grep finds Landscape references to session identifiers, do not reset. Preserve both databases and create or update the rollout issue with the reference evidence.
- If startup still fails with `SessionSchemaError`, the running process is pointing at a different stale `sessions.db` than the one reset. Re-resolve `WebSettings.get_session_db_url()` from the live environment.
- If `systemctl` or `sudo` fails from the sandbox, report the exact blocker and have the operator run the host-side restart. Do not claim staging was restarted or live-verified from inside the sandbox.
