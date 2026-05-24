# Session DB Reset Runbook

Use this runbook when a web session schema-bootstrap change requires deleting or archiving a stale `sessions.db`. Historically the session database was reset in isolation from the Landscape audit database, payload storage, blobs, and Filigree tracker data. **From the Phase 4 hello-world tutorial schema cutover onward, any deploy that changes both `SESSION_SCHEMA_EPOCH` and `SQLITE_SCHEMA_EPOCH` must reset the session DB and Landscape audit DB together.** Phase 4 adds tutorial run/audit-story columns on both sides of the web/Landscape boundary; Phase 5b (commit `2e390fc0b`) adds the later cross-DB invariant where `interpretation_events.resolved_prompt_template_hash` is byte-equal to the matching Landscape `calls_table.resolved_prompt_template_hash`. See [Phase 5b: Two-DB Reset](#phase-5b-two-db-reset) below. Payload storage, blobs outside the session DB, and Filigree tracker data are still out of scope for this runbook.

## Current Session-Only Schema Cutover

P3 of `elspeth-fdebcaa79a` adds `blob_inline_resolutions` to the web
session DB and bumps `SESSION_SCHEMA_EPOCH` to 12. This is a
session-only schema cutover: delete/recreate the session DB with this
runbook before deploying the build, but do not reset the Landscape DB
unless the same deploy also changes `SQLITE_SCHEMA_EPOCH`.

## Deployment Scope (Schedule 1A)

Schedule 1A's per-session write discipline (`_session_write_lock`,
`_reserve_sequence_range`, `_insert_chat_message`,
`_insert_composition_state`) is proven against **SQLite single-worker
deployments only**. The proof rests on:

- SQLite `_session_write_lock` is implemented as a process-wide
  per-session `threading.RLock` (`_sqlite_lock_for_session`). It
  serialises same-session writers within one process; it does NOT
  cross process boundaries.
- The deployed staging service (`elspeth-web.service`) runs single-worker:
  no `--workers` flag, `WEB_CONCURRENCY` unset or `1`, and the startup
  multi-worker guard in `src/elspeth/web/app.py` enforces this.
- The PostgreSQL branch of `_session_write_lock` uses
  `pg_advisory_xact_lock(...)` and is correct in principle, but is NOT
  exercised by the Schedule 1A test surface and is NOT claimed here.

**What 1A does NOT prove:**

- Multi-worker SQLite deployments (the per-session RLock cannot
  serialise across processes).
- PostgreSQL schema correctness against a live Postgres instance
  (`tests/unit/web/sessions/` runs against in-memory and file-backed
  SQLite only; the model definitions are dialect-agnostic but the
  migration path, CHECK-constraint translations, and partial-index
  semantics are not exercised against `pg_*`).
- Cross-process advisory-lock concurrency under real load.

**Where the gaps land:** Schedule 1C is the explicit home for
PostgreSQL DDL parity, advisory-lock concurrency proofs, and the
multi-worker deployment path. Operators planning to deploy
Schedule 1A on anything other than single-worker SQLite must wait
for Schedule 1C — the schema cutover is internally consistent on
Postgres in principle, but the proof obligation has not been
discharged on this branch.

The Phase 1A merge gate is therefore SQLite-only:

```bash
.venv/bin/python -m pytest tests/unit/web/ tests/integration/web/
```

A passing run of that command demonstrates SQLite-current
deployability. It does not demonstrate PostgreSQL deployability or
multi-worker safety.

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

   Expected session tables include `sessions`, `chat_messages`, `composition_states`, `runs`, `run_events`, `blobs`, `blob_run_links`, `user_secrets`, and `audit_access_log`. `initialize_session_schema()` validates the metadata table set exactly; a guide that lists fewer tables than the live metadata is stale and cannot be used as cutover evidence.

3. Archive the session DB artifact set before deleting it. For SQLite, `sessions.db`, `sessions.db-wal`, `sessions.db-shm`, and `sessions.db-journal` are one rollback/recreate unit — never archive or delete only the main file:

   ```bash
   for artifact in /path/to/sessions.db /path/to/sessions.db-wal /path/to/sessions.db-shm /path/to/sessions.db-journal; do
       if [ -e "$artifact" ]; then
           cp -a "$artifact" "${artifact}.before-reset"
       fi
   done
   ```

4. Delete or move the confirmed session DB artifact set (main file plus `-wal`, `-shm`, and `-journal` sidecars together). Mixing a new main file with stale sidecars, or restoring a main file without its sidecars, is not a valid reset.
5. Start the web process. `initialize_session_schema()` recreates the DB with current metadata.
6. Create a new session through the web API or UI. A startup `SessionSchemaError` means a stale session DB is still being used.

## Phase 5b: Two-DB Reset

This procedure applies to any staging deploy that changes both the web session DB schema and the Landscape audit DB schema in the same cutover. Phase 4 hello-world tutorial work is in scope because it changes `SESSION_SCHEMA_EPOCH` and `SQLITE_SCHEMA_EPOCH` together for tutorial completion and run/audit-story replay. Phase 5b is also in scope: skipping the Landscape delete after a Phase 5b deploy leaves stale `calls_table` rows whose `resolved_prompt_template_hash` is absent or stale; the first composer run after deploy will diverge from the session DB's `interpretation_events.resolved_prompt_template_hash` and the cross-DB byte-equality invariant (asserted by `tests/integration/web/composer/test_interpretation_runtime_handoff.py`) will fire.

Authority: `docs/composer/ux-redesign-2026-05/18a-phase-5b-backend.md` §"Migration runner ownership", lines 160–177.

### Two-DB preconditions

1. The Stop/Go Gates above have been run for the session DB.
2. The deploy changes both `SESSION_SCHEMA_EPOCH` and `SQLITE_SCHEMA_EPOCH`; this includes the Phase 4 hello-world tutorial dual-schema cutover and commit `2e390fc0b` or later (Phase 5b session/Landscape schema changes).
3. The operator has resolved the active Landscape DB path per "Resolve Database Paths" above:
   - If `ELSPETH_WEB__LANDSCAPE_URL` is set, that is the Landscape URL.
   - Otherwise the default is `${ELSPETH_WEB__DATA_DIR}/runs/audit.db`, or `data/runs/audit.db` if `ELSPETH_WEB__DATA_DIR` is unset.
4. The operator has explicitly signed off on losing the Landscape audit history in staging. The two-DB reset destroys staging audit data alongside session data; this is acceptable for staging only.

### Two-DB procedure (in addition to the staging session-DB reset below)

Run after `sudo systemctl stop "$SERVICE"` and before `sudo systemctl start "$SERVICE"` in the staging procedure. Both DBs are reset under the same service-stop window.

```bash
# Continuing from the staging procedure: $PROJECT_ROOT, $ENV_FILE, $SERVICE,
# $PROJECT_ROOT_CANON already resolved; $SERVICE is stopped.

LANDSCAPE_URL="$(grep -E '^ELSPETH_WEB__LANDSCAPE_URL=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"
DATA_DIR="$(grep -E '^ELSPETH_WEB__DATA_DIR=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"

if [ -n "$LANDSCAPE_URL" ]; then
    case "$LANDSCAPE_URL" in
        sqlite:///*) LANDSCAPE_PATH="${LANDSCAPE_URL#sqlite:///}" ;;
        sqlite:////*) LANDSCAPE_PATH="/${LANDSCAPE_URL#sqlite:////}" ;;
        *) echo "REFUSING: non-sqlite Landscape URL requires a migration plan." >&2; exit 1 ;;
    esac
elif [ -n "$DATA_DIR" ]; then
    LANDSCAPE_PATH="$DATA_DIR/runs/audit.db"
else
    LANDSCAPE_PATH="$PROJECT_ROOT/data/runs/audit.db"
fi

case "$LANDSCAPE_PATH" in
    /*) ;;
    *) LANDSCAPE_PATH="$PROJECT_ROOT/$LANDSCAPE_PATH" ;;
esac
LANDSCAPE_PATH="$(realpath -m "$LANDSCAPE_PATH")"

case "$LANDSCAPE_PATH" in
    "$PROJECT_ROOT_CANON"/*) ;;
    *) echo "REFUSING: LANDSCAPE_PATH is outside $PROJECT_ROOT_CANON: $LANDSCAPE_PATH" >&2; exit 1 ;;
esac

LANDSCAPE_ARTIFACTS=(
    "$LANDSCAPE_PATH"
    "$LANDSCAPE_PATH-wal"
    "$LANDSCAPE_PATH-shm"
    "$LANDSCAPE_PATH-journal"
)

if command -v fuser >/dev/null 2>&1; then
    for artifact in "${LANDSCAPE_ARTIFACTS[@]}"; do
        if [ -e "$artifact" ] && sudo fuser "$artifact" >/dev/null 2>&1; then
            echo "REFUSING: $artifact is still open after $SERVICE stopped." >&2
            sudo fuser -v "$artifact" >&2 || true
            exit 1
        fi
    done
fi

FOUND_LANDSCAPE_ARTIFACT=0
for artifact in "${LANDSCAPE_ARTIFACTS[@]}"; do
    if [ -e "$artifact" ]; then
        FOUND_LANDSCAPE_ARTIFACT=1
    fi
done

if [ "$FOUND_LANDSCAPE_ARTIFACT" -eq 1 ]; then
    LANDSCAPE_SNAPSHOT_DIR="$LANDSCAPE_PATH.pre-two-db-reset.$(date -u +%Y%m%dT%H%M%SZ)"
    sudo mkdir -p "$LANDSCAPE_SNAPSHOT_DIR"
    for artifact in "${LANDSCAPE_ARTIFACTS[@]}"; do
        if [ -e "$artifact" ]; then
            sudo cp -a "$artifact" "$LANDSCAPE_SNAPSHOT_DIR/$(basename "$artifact")"
        fi
    done
    echo "Archived existing Landscape DB artifact set to $LANDSCAPE_SNAPSHOT_DIR"
fi

for artifact in "${LANDSCAPE_ARTIFACTS[@]}"; do
    sudo rm -f "$artifact"
done
```

`LandscapeDB.from_url(...)` recreates the audit DB on the first pipeline execution after restart.

### Two-DB verification

After both DBs are reset, both health checks pass, and a new session is created through the UI:

1. Run the canonical composer test case end-to-end (`create a list of 5 government web pages and use an LLM to rate how cool they are`) so a composer pipeline executes at least one LLM call.
2. Confirm the service journal contains no `AssertionError` or invariant-violation traceback referencing `resolved_prompt_template_hash` or `test_interpretation_runtime_handoff`.
3. Confirm both `sessions.db` and `audit.db` were recreated by the service (file mtime is later than the service start time).

If the invariant fires, do not retry. Stop the service, preserve both DB snapshots, and inspect the diverging hashes against the deployed commit.

## Skill Changes Require Service Restart, Not Reload

The composer LLM system prompt is loaded from `src/elspeth/web/composer/skills/pipeline_composer.md` (and the guided variants under `src/elspeth/web/composer/guided/prompts.py`) through module-level `@lru_cache` decorators (`functools.lru_cache` on `build_system_prompt` and the guided prompt loaders). Cache entries live for the process lifetime and are not invalidated by file mtime, `SIGHUP`, or `systemctl reload`.

When deploying skill-content changes such as Phase 5a.8 (`34d272360` — inline_blob preference for chat-typed source data) and Phase 5b.8 (`d6219faa2` — teaching the LLM when to call `request_interpretation_review`), or any future edit to `pipeline_composer.md` / guided prompt fragments:

```bash
sudo systemctl restart elspeth-web.service
```

`systemctl reload` is not sufficient and will silently serve the previous skill text.

After restart, verify by composing a new session and confirming the new guidance is reflected in the assistant's behaviour (for Phase 5b.8: the LLM offers `request_interpretation_review` when the composition includes a non-trivial interpretive choice; for Phase 5a.8: simple chat-typed source data prefers `inline_blob` over CSV upload).

## Staging Reset For `elspeth.foundryside.dev`

The staging site is a source-checkout systemd/Caddy deployment from `/home/john/elspeth`, not the generic VM/Docker flow. When a pre-release plan changes the session DB schema (e.g. composer-progress-persistence Phase 1A and later schema-changing phases), the schema validator at startup will refuse a stale DB; the only accepted cutover path is archive + delete + recreate. Row-level `DELETE FROM chat_messages` / `DELETE FROM composition_states` is incorrect: it leaves the old table shape behind and startup rejects the stale DB.

This procedure destroys staging session rows, chat history, composition states, audit access log rows, runs, run events, blob/blob-link database records, and encrypted `user_secrets` stored in the web session DB. It does not delete blob payload files under the data directory, payload storage, Filigree state, or source files. **If the deploy changes only the session DB schema, do not touch the Landscape audit DB. If the deploy changes both `SESSION_SCHEMA_EPOCH` and `SQLITE_SCHEMA_EPOCH`, run the additional [Phase 5b: Two-DB Reset](#phase-5b-two-db-reset) procedure inside the same service-stop window. Phase 4 hello-world tutorial is a dual-schema cutover; Phase 5b and later dual-schema cutovers also require it, and the cross-DB hash invariant will fire on the first run otherwise.** **Do not run any of this outside staging.**

For SQLite, `sessions.db`, `sessions.db-wal`, `sessions.db-shm`, and `sessions.db-journal` are handled as one matched artifact set for archive, deletion, and rollback.

### Preconditions

1. The host is the staging host for `elspeth.foundryside.dev`.
2. No human operator is mid-session.
3. The source checkout at `/home/john/elspeth` is on the commit being deployed.
4. `deploy/elspeth-web.env` has been inspected directly for session DB settings without printing secret values.
5. The Stop/Go Gates above have been run: Landscape code/schema must not reference web-session identifiers.
6. The pre-cutover source ref compatible with the archived DB has been recorded. If rollback is needed, restore that ref and the archived DB together; never run the old DB under the new schema code.
7. The live SQLite deployment is single-worker: `deploy/elspeth-web.service` has no `--workers` flag, `WEB_CONCURRENCY` is unset or `1`, and the startup multi-worker guard in `src/elspeth/web/app.py` remains enabled.
8. The operator has explicitly signed off on the `user_secrets` blast radius. Either the archived DB is the accepted recovery point, or staging secrets have a documented re-entry/reseed procedure before users resume composer work.
9. No other host-side process is writing the SQLite DB. The procedure stops `elspeth-web.service` and checks open handles before copying; if another process still has the main DB or a sidecar open, stop and identify it before continuing.

### Procedure

Use a host shell with `systemctl` and `sudo` access. The Codex sandbox cannot run this procedure end-to-end.

```bash
set -euo pipefail

PROJECT_ROOT="/home/john/elspeth"
ENV_FILE="$PROJECT_ROOT/deploy/elspeth-web.env"
SERVICE="elspeth-web.service"
PROJECT_ROOT_CANON="$(realpath -m "$PROJECT_ROOT")"

if [ "$(pwd)" != "$PROJECT_ROOT" ]; then
    echo "REFUSING: run from $PROJECT_ROOT so relative defaults are unambiguous." >&2
    exit 1
fi

if ! systemctl show "$SERVICE" --property=FragmentPath --value | grep -q "elspeth-web.service"; then
    echo "REFUSING: $SERVICE is not the active staging service." >&2
    exit 1
fi

# Resolve the DB path without echoing the environment file. Secrets may
# live in the same file. Precedence:
#   1. ELSPETH_WEB__SESSION_DB_URL=file-or-sqlite-url
#   2. ELSPETH_WEB__DATA_DIR/sessions.db
#   3. /home/john/elspeth/data/sessions.db
SESSION_DB_URL="$(grep -E '^ELSPETH_WEB__SESSION_DB_URL=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"
DATA_DIR="$(grep -E '^ELSPETH_WEB__DATA_DIR=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"

if [ -n "$SESSION_DB_URL" ]; then
    case "$SESSION_DB_URL" in
        sqlite:///*) DB_PATH="${SESSION_DB_URL#sqlite:///}" ;;
        sqlite:////*) DB_PATH="/${SESSION_DB_URL#sqlite:////}" ;;
        *) echo "REFUSING: non-sqlite session DB URL requires a migration plan." >&2; exit 1 ;;
    esac
elif [ -n "$DATA_DIR" ]; then
    DB_PATH="$DATA_DIR/sessions.db"
else
    DB_PATH="$PROJECT_ROOT/data/sessions.db"
fi

case "$DB_PATH" in
    /*) ;;
    *) DB_PATH="$PROJECT_ROOT/$DB_PATH" ;;
esac
DB_PATH="$(realpath -m "$DB_PATH")"

case "$DB_PATH" in
    "$PROJECT_ROOT_CANON"/*) ;;
    *) echo "REFUSING: DB_PATH is outside $PROJECT_ROOT_CANON: $DB_PATH" >&2; exit 1 ;;
esac

DB_ARTIFACTS=(
    "$DB_PATH"
    "$DB_PATH-wal"
    "$DB_PATH-shm"
    "$DB_PATH-journal"
)

echo "Resolved staging session DB path: $DB_PATH"
read -r -p "Archive and recreate this staging DB? Type RECREATE to continue: " CONFIRM
if [ "$CONFIRM" != "RECREATE" ]; then
    echo "Aborted."
    exit 1
fi

sudo systemctl stop "$SERVICE"

if command -v fuser >/dev/null 2>&1; then
    for artifact in "${DB_ARTIFACTS[@]}"; do
        if [ -e "$artifact" ] && sudo fuser "$artifact" >/dev/null 2>&1; then
            echo "REFUSING: $artifact is still open after $SERVICE stopped." >&2
            sudo fuser -v "$artifact" >&2 || true
            exit 1
        fi
    done
fi

FOUND_DB_ARTIFACT=0
for artifact in "${DB_ARTIFACTS[@]}"; do
    if [ -e "$artifact" ]; then
        FOUND_DB_ARTIFACT=1
    fi
done

if [ "$FOUND_DB_ARTIFACT" -eq 1 ]; then
    SNAPSHOT_DIR="$DB_PATH.pre-phase1.$(date -u +%Y%m%dT%H%M%SZ)"
    sudo mkdir -p "$SNAPSHOT_DIR"
    for artifact in "${DB_ARTIFACTS[@]}"; do
        if [ -e "$artifact" ]; then
            sudo cp -a "$artifact" "$SNAPSHOT_DIR/$(basename "$artifact")"
        fi
    done
    echo "Archived existing DB artifact set to $SNAPSHOT_DIR"
fi

for artifact in "${DB_ARTIFACTS[@]}"; do
    sudo rm -f "$artifact"
done
sudo systemctl start "$SERVICE"

curl --unix-socket /run/elspeth/uvicorn.sock -fsS http://localhost/api/health
curl -fsS https://elspeth.foundryside.dev/api/health
sudo systemctl status "$SERVICE" --no-pager --lines=20
```

`initialize_session_schema()` recreates the file on service startup. If either health check fails, inspect `journalctl -u elspeth-web.service --no-pager -n 80` before retrying.

After health checks pass, create a new session through the API or UI and confirm no `SessionSchemaError` appears in the service journal.

Before handing staging back to users, verify the `user_secrets` outcome the operator chose in the preconditions. If secrets were intentionally cleared, confirm the affected composer/provider flow reports the expected missing-secret state and that the operator has re-entered or reseeded any required staging secrets. If rollback depends on the archive, confirm the archived DB is retained because it contains the pre-reset encrypted secret rows as well as chat/session data.

### Rollback

Rollback is allowed only before users resume work on the recreated DB, or after explicitly preserving both the failed new DB and the archived old DB for operator review. If the new service has accepted user traffic, do not overwrite the new DB with the archive without a data-preservation decision.

Rollback must restore source and data as a compatible pair. Because the session DB also stores encrypted `user_secrets`, restoring the archive also restores the pre-cutover staging secrets; never mix a new-source checkout with the old secret/schema archive.

```bash
set -euo pipefail

PROJECT_ROOT="/home/john/elspeth"
SERVICE="elspeth-web.service"
DB_PATH="/absolute/path/resolved/by/the/procedure"
SNAPSHOT_DIR="/absolute/path/to/sessions.db.pre-phase1.YYYYMMDDTHHMMSSZ"
PRE_CUTOVER_REF="<recorded commit/ref compatible with SNAPSHOT_DIR>"
DB_ARTIFACTS=(
    "$DB_PATH"
    "$DB_PATH-wal"
    "$DB_PATH-shm"
    "$DB_PATH-journal"
)

sudo systemctl stop "$SERVICE"

FOUND_NEW_ARTIFACT=0
for artifact in "${DB_ARTIFACTS[@]}"; do
    if [ -e "$artifact" ]; then
        FOUND_NEW_ARTIFACT=1
    fi
done

if [ "$FOUND_NEW_ARTIFACT" -eq 1 ]; then
    FAILED_NEW_DB_DIR="$DB_PATH.failed-phase1.$(date -u +%Y%m%dT%H%M%SZ)"
    sudo mkdir -p "$FAILED_NEW_DB_DIR"
    for artifact in "${DB_ARTIFACTS[@]}"; do
        if [ -e "$artifact" ]; then
            sudo cp -a "$artifact" "$FAILED_NEW_DB_DIR/$(basename "$artifact")"
        fi
    done
    echo "Preserved failed new DB artifact set at $FAILED_NEW_DB_DIR"
fi

if [ -n "$(git -C "$PROJECT_ROOT" status --porcelain)" ]; then
    echo "REFUSING: $PROJECT_ROOT has uncommitted changes; preserve or commit them before rollback." >&2
    git -C "$PROJECT_ROOT" status --short >&2
    exit 1
fi

# Use the approved source-checkout rollback mechanism for staging. The
# important invariant is compatibility: the restored process must run the
# pre-cutover code that understands SNAPSHOT_DIR's schema. Do not use
# `git reset --hard` from automation; the dirty-tree guard above makes
# this fail closed before switching refs.
git -C "$PROJECT_ROOT" switch --detach "$PRE_CUTOVER_REF"

for artifact in "${DB_ARTIFACTS[@]}"; do
    sudo rm -f "$artifact"
done
for artifact in "${DB_ARTIFACTS[@]}"; do
    archived="$SNAPSHOT_DIR/$(basename "$artifact")"
    if [ -e "$archived" ]; then
        sudo cp -a "$archived" "$artifact"
    fi
done
sudo systemctl start "$SERVICE"

curl --unix-socket /run/elspeth/uvicorn.sock -fsS http://localhost/api/health
curl -fsS https://elspeth.foundryside.dev/api/health
sudo journalctl -u "$SERVICE" --no-pager -n 80
```

If rollback still fails, keep the service stopped, preserve both DB files, and inspect the journal before trying another reset. Retrying the new schema is appropriate when the failure was operational (wrong path, permission, stale process). Rolling back is appropriate when the new code or metadata is defective and the old DB snapshot must be served again.

### Why DELETE Is Forbidden For Schema-Changing Phases

Row deletion does not add columns, CHECK constraints, FKs, or indexes. For schema-changing Phase plans (Phase 1A introduces NOT NULL `chat_messages.sequence_no`, NOT NULL `chat_messages.writer_principal`, NOT NULL `composition_states.provenance`, and the new `audit_access_log` table), row deletion leaves a stale DB shape that the startup schema validator correctly rejects. The validator is primarily a name/shape guard for expected tables, columns, CHECK names, and index names; it is not a compatibility migration engine and must not be treated as proof that stale CHECK expressions, partial-index predicates, or old table layouts are safe. Archive/delete/recreate is the only accepted cutover path for schema-changing phases. Archive, delete, and rollback must handle the SQLite main DB plus `-wal`, `-shm`, and `-journal` sidecars as a single artifact set; mixing a new main file with stale sidecars, or restoring a main file without its sidecars, is not a valid reset.

## Failure Handling

- If the stop/go grep finds Landscape references to session identifiers, do not reset. Preserve both databases and create or update the rollout issue with the reference evidence.
- If startup still fails with `SessionSchemaError`, the running process is pointing at a different stale `sessions.db` than the one reset. Re-resolve `WebSettings.get_session_db_url()` from the live environment.
- If `systemctl` or `sudo` fails from the sandbox, report the exact blocker and have the operator run the host-side restart. Do not claim staging was restarted or live-verified from inside the sandbox.
