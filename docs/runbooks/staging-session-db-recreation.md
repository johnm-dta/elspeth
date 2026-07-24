# Session DB Reset Runbook

Use this runbook when a pre-1.0 schema change requires deleting or archiving stale `sessions.db` and Landscape databases. Any deploy that changes both `SESSION_SCHEMA_EPOCH` and `SQLITE_SCHEMA_EPOCH` must coordinate both databases in one service-stop window. Before 1.0, the supported upgrade is uninstall, archive/export when required, recreate, and reinstall; ELSPETH does not migrate either database in place. Phase 4 adds tutorial run/audit-story columns on both sides of the web/Landscape boundary; Phase 5b (commit `2e390fc0b`) adds the later cross-DB invariant where `interpretation_events.resolved_prompt_template_hash` is byte-equal to the matching Landscape `calls_table.resolved_prompt_template_hash`. See [Phase 5b: Two-DB Reset](#phase-5b-two-db-reset) below. Payload storage, blobs outside the session DB, and Filigree tracker data are still out of scope for this runbook.

## Current Cutover: 0.7.2 blob deletion cleanup (session epoch 36 and Landscape epoch 29)

0.7.2 advances `SESSION_SCHEMA_EPOCH` from 35 to 36 so a committed blob
deletion whose tombstone unlink or directory fsync fails remains retryable
after restart. An epoch-35 database cannot represent that durable blob-deletion cleanup state
and must be recreated.

0.7.1 advances the session store from epoch 26 through epoch 35. Epoch 27 lets
`user_preferences.freeform_intro_dismissed_at` persist the account-wide
freeform-primer preference, then to 28 so SQLite and PostgreSQL session stores
carry the same application/store/epoch identity proof. Composer parity then
advances the session store to epoch 29 for guided schema 8 and durable fenced
guided operations, and to epoch 30 because the closed
`guided_operations.failure_code` CHECK gains `quota_exceeded`. That final
boundary makes a fork quota failure settle and replay as a stable HTTP 413.
Later hard cuts add guided pipeline-proposal replay (31), exact failed-operation
audit cohorts (32), guided-start negative admission (33), guided schema 10 (34),
and exclusive guided-confirmation proposal admission (35). The universal web
plugin-policy work in 0.7.1 also advances
`SQLITE_SCHEMA_EPOCH` from 22 to 23 and adds `run_web_plugin_policy`. This
table is optional per run but required in the schema: web runs receive one
policy-evidence row atomically with the run, attribution, and leader records;
CLI runs receive none. Database hardening then advances Landscape from epoch 23
to 24 and adds `tokens(row_id, run_id) -> rows(row_id, run_id)`, then advances
to 25 with a partial unique index over non-null
`artifacts(run_id, idempotency_key)` pairs. Durable sink effects advance the
Landscape to epoch 26, durable coalesce effects advance it to epoch 27, and
per-member failsink-to-primary provenance advances it to epoch 28. Epoch 29
adds canonical node output-contract hashes, run-scoped token ancestry and
validation-error links, durable batch-expansion claims, and the
transaction-owned sidecar-journal outbox.

Archive and recreate the session database, its sidecars, and every stale
Landscape database under the service-stop procedure below. Every predecessor
session epoch is a recreate boundary, including epoch 35. Landscape epoch 29
is the current release boundary; recreate a Landscape database only when
its own sentinel is stale. Any stale PostgreSQL session shape is recreated by
the schema owner; the runtime role remains DML-only.

Validate-only startup and doctor must leave stale databases unchanged. Do not
use `create_all`, `--init-schema`, an old migrator, or code rollback as an improvised repair
mechanism. `auth.db` is a separate file and is not reset.

The release/schema compatibility record for every candidate using this shape
must state: candidate git SHA and immutable image/task-definition identity;
session and Landscape epochs; presence of `run_web_plugin_policy`; structural
and semantics-only changes; archive/export decision and approver; destructive
reset requirement and database-operator approval; previous release identity
and epochs; forward and backward compatibility decisions; and an explicit
`rollback_permitted` decision with evidence. Older code is not compatible with
the freshly recreated current databases. Rollback across this boundary is
unsupported: keep the service drained, repair the epoch-36 release forward,
recreate fresh state, and retry. The release acceptance record must cite the
session-epoch-36/Landscape-epoch-29 record when binding candidate and rollback
decisions.

Deployments crossing the 0.7.0 boundary from an older release must also account
for the historical epoch-21 to epoch-22 Landscape reset described below.

## Historical Cutover: 0.7.0 (two-DB reset)

0.7.0 advances **both** schema epochs: `SESSION_SCHEMA_EPOCH` is now 26 and
the Landscape audit DB `SQLITE_SCHEMA_EPOCH` is now 22. This is a
**two-DB reset**: follow the [Staging Reset](#staging-reset-for-elspethfoundrysidedev)
procedure and the [Phase 5b: Two-DB Reset](#phase-5b-two-db-reset) procedure
inside the same service-stop window. Do not run 0.7.0 against a stale
Landscape audit DB from epoch 21.

The session epoch changes in this release are:

- **23→24 / `GUIDED_SESSION_SCHEMA_VERSION` 5→6** added guided metadata
  (`profile`, `advisor_checkpoint_passes_used`,
  `advisor_signoff_escape_offered`) inside the
  `composition_states.composer_meta` JSON blob.
- **24→25 / `GUIDED_SESSION_SCHEMA_VERSION` 6→7** dropped the vestigial
  `profile.entry_seed` key. Without the lockstep epoch bump, a stale
  `entry_seed`-bearing blob would slip past both version gates and lazy-500
  inside `WorkflowProfile.from_dict`'s closed-key-set check.
- **25→26** adds first-run tutorial resume columns to `user_preferences`
  (`tutorial_stage`, `tutorial_session_id`, `tutorial_run_id`,
  `tutorial_source_data_hash`).

The Landscape epoch change is **21→22**: `routing_events` now carries `run_id`
and composite state/edge foreign keys so routing decisions cannot cross
audit-run boundaries.

Each session epoch bump converts what would otherwise be a lazy per-row failure
into a loud boot fail-close over the **whole** session DB. 0.7.0 boot fails
closed on a stale session DB via the `_assert_schema_sentinels` guard with an
error like `SessionSchemaError: Session DB schema version 25 does not match
SESSION_SCHEMA_EPOCH=26. Pre-release ELSPETH does not migrate session
databases. Delete the session DB file and restart.` `auth.db` is a separate
file and is NOT reset — local user accounts survive this cutover.

## Historical Cutover: 0.6.0 (two-DB reset)

0.6.0 advances **both** epochs: `SESSION_SCHEMA_EPOCH` to 19 and the
Landscape audit DB `SQLITE_SCHEMA_EPOCH` to 21. Per the rule above, this
is a **two-DB reset** — run the staging procedure below *and* the
[Phase 5b: Two-DB Reset](#phase-5b-two-db-reset) procedure inside the same
service-stop window. 0.6.0 boot fails closed on a stale session DB with
`SessionSchemaError: Session DB schema version 18 does not match
SESSION_SCHEMA_EPOCH=19. Pre-release ELSPETH does not migrate session
databases. Delete the session DB file and restart.` `auth.db` is a
separate file and is NOT reset — local user accounts survive.

### Earlier cutover (historical)

P3 of `elspeth-fdebcaa79a` added `blob_inline_resolutions` to the web
session DB and bumped `SESSION_SCHEMA_EPOCH` to 12. That was a
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

This destructive procedure applies whenever the configured Landscape store is
older than the running pre-1.0 release. There is no supported in-place path for
SQLite or PostgreSQL predecessor schemas. Archive or export required evidence,
delete/recreate the store, and reinstall/initialize the current release.
Read-only and `create_tables=False` inspection opens report incompatibility
without mutation. PostgreSQL recreation is performed by the schema owner; the
runtime role remains DML-only.

Historical rationale remains unchanged: Phase 4 changed both epochs for
tutorial completion and run/audit-story replay. Phase 5b also required deleting
the stale Landscape database because old `calls_table` rows lacked the current
`resolved_prompt_template_hash`; keeping them would violate the cross-DB
byte-equality invariant asserted by
`tests/integration/web/composer/test_interpretation_runtime_handoff.py` on the
first composer run.

Authority: current source epoch constants and schema tests:
`src/elspeth/web/sessions/models.py:SESSION_SCHEMA_EPOCH`,
`src/elspeth/core/landscape/schema.py:SQLITE_SCHEMA_EPOCH`, and
`tests/unit/core/landscape/test_schema_epoch_and_required_columns.py`.

### Two-DB preconditions

1. The Stop/Go Gates above have been run for the session DB.
2. The approved cutover record explicitly requires destructive Landscape
   recreation. Historical examples include the Phase 4 tutorial dual-schema
   cutover, Phase 5b schema changes, and the 0.7.0 epoch-26 / epoch-22 release
   cutover.
3. The operator has resolved the active Landscape DB path per "Resolve Database Paths" above:
   - If `ELSPETH_WEB__LANDSCAPE_URL` is set, that is the Landscape URL.
   - Otherwise the default is `${ELSPETH_WEB__DATA_DIR}/runs/audit.db`, or `data/runs/audit.db` if `ELSPETH_WEB__DATA_DIR` is unset.
4. The operator has explicitly signed off on losing the Landscape audit history
   in staging. The two-DB reset destroys staging audit data alongside session
   data; this is acceptable for staging only.

### Two-DB procedure (in addition to the staging session-DB reset below)

When the destructive preconditions apply, run after
`sudo systemctl stop "$SERVICE"` and before
`sudo systemctl start "$SERVICE"` in the staging procedure. Both DBs are reset
under the same service-stop window. Archive the matched Landscape artifact set
when evidence retention requires it, then delete every predecessor artifact.
The archive is evidence only, not an input to the current release.

```bash
# Continuing from the staging procedure: $PROJECT_ROOT, $ENV_FILE, $SERVICE,
# $PROJECT_ROOT_CANON already resolved; $SERVICE is stopped.

LANDSCAPE_URL="$(grep -E '^ELSPETH_WEB__LANDSCAPE_URL=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"
DATA_DIR="$(grep -E '^ELSPETH_WEB__DATA_DIR=' "$ENV_FILE" | tail -n1 | cut -d= -f2- || true)"

if [ -n "$LANDSCAPE_URL" ]; then
    case "$LANDSCAPE_URL" in
        sqlite:///*) LANDSCAPE_PATH="${LANDSCAPE_URL#sqlite:///}" ;;
        sqlite:////*) LANDSCAPE_PATH="/${LANDSCAPE_URL#sqlite:////}" ;;
        *) echo "REFUSING: this SQLite reset procedure cannot recreate a non-sqlite Landscape store." >&2; exit 1 ;;
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

This procedure destroys staging session rows, chat history, composition states, audit access log rows, runs, run events, blob/blob-link database records, and encrypted `user_secrets` stored in the web session DB. It does not delete blob payload files under the data directory, payload storage, Filigree state, or source files. **If the deploy changes only the session DB schema, do not touch a current Landscape audit DB. If the Landscape schema is stale, archive/export required evidence and recreate it with the current release; no predecessor schema is transformed in place.** **Do not run any of this outside staging.**

For SQLite, `sessions.db`, `sessions.db-wal`, `sessions.db-shm`, and `sessions.db-journal` are handled as one matched artifact set for archive, deletion, and recreation.

### Preconditions

1. The host is the staging host for `elspeth.foundryside.dev`.
2. No human operator is mid-session.
3. The source checkout at `/home/john/elspeth` is on the commit being deployed.
4. `deploy/elspeth-web.env` has been inspected directly for session DB settings without printing secret values.
5. The Stop/Go Gates above have been run: Landscape code/schema must not reference web-session identifiers.
6. The candidate source ref has been recorded. Archived predecessor databases
   are evidence only and are never opened by the current release.
7. The live SQLite deployment is single-worker: `deploy/elspeth-web.service` has no `--workers` flag, `WEB_CONCURRENCY` is unset or `1`, and the startup multi-worker guard in `src/elspeth/web/app.py` remains enabled.
8. The operator has explicitly signed off on the `user_secrets` blast radius and
   has a documented re-entry/reseed procedure before users resume Composer work.
   **Any archive is a long-lived copy of live encrypted secret material and is
   retained as evidence, not as a recovery database.** The `user_secrets` rows
   are encrypted with `settings.secret_key`; if the key is reused, the archive
   remains decryptable. Decide the key-rotation and archive-retention outcomes
   up front and record them in the deploy plan.
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
        *) echo "REFUSING: this SQLite reset procedure cannot recreate a non-sqlite session store." >&2; exit 1 ;;
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

# Checkpoint the WAL into the main DB FIRST so the archive captures a
# self-contained copy. The session DB also stores the encrypted
# UserSecretStore (constructed on the session engine in app.py:
# `UserSecretStore(session_engine, settings.secret_key)`), so the -wal
# sidecar can hold uncheckpointed encrypted secret material. Folding the
# WAL into the main file means the long-lived archive does not depend on a
# matched -wal/-shm sidecar set to be readable, and no secret rows are
# stranded in a sidecar at archive time. Guard on existence: a missing DB
# is the first-deploy case, and running sqlite3 against an absent path
# would create a junk 0-byte file that the archive loop would then copy.
if [ -e "$DB_PATH" ]; then
    sqlite3 "$DB_PATH" 'PRAGMA wal_checkpoint(TRUNCATE);'
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

After health checks pass, prove the recreated session store carries the current
hard-cut sentinel before creating any session:

```bash
sqlite3 "$DB_PATH" 'PRAGMA user_version;'  # expect 36 (== SESSION_SCHEMA_EPOCH)
```

An epoch-35 result is not repairable in place: keep the service drained,
recreate the session database with the current release, and rerun the probe.
Then create a new session through the API or UI and confirm no
`SessionSchemaError` appears in the service journal.

#### 0.7.0 epoch + smoke verification

Confirm the recreated session DB and Landscape audit DB carry the new epoch
sentinels, then drive a fresh guided session to completion to prove the 0.7.0
build is serving the recreated schemas cleanly:

```bash
# Confirm the recreated session DB carries the new epoch sentinel.
sqlite3 "$DB_PATH" 'PRAGMA user_version;'         # expect 26 (== SESSION_SCHEMA_EPOCH)

# If LANDSCAPE_PATH is not already set from the two-DB reset procedure, resolve
# it with that procedure's Landscape path block before running this check.
sqlite3 "$LANDSCAPE_PATH" 'PRAGMA user_version;'  # expect 22 (== SQLITE_SCHEMA_EPOCH)
```

If either `PRAGMA user_version` is not the expected value, the running process
is serving a stale or non-recreated DB; stop and re-resolve the paths per
"Resolve Database Paths" before continuing.

Then run a fresh-session smoke through the UI:

1. Create a new session.
2. Start the tutorial on the `TUTORIAL` profile.
3. Drive the staged guided walk through to a `terminal=completed` state.
4. Run the resulting pipeline.

Confirm the service journal shows **no** `SessionSchemaError` (the boot
guard passed), **no** per-row HTTP 500 from `GuidedSession.from_dict`
(the guided-schema bump landed in the recreated DB, not lazily on a stale
row), and **no** `UnresolvedInterpretationPlaceholderError` (proving the
B1 interpretation-surfacing fix is in the deployed build). Any of these in
the journal means the deploy is not clean — stop and inspect before
handing staging back to users.

Before handing staging back to users, verify the `user_secrets` outcome the operator chose in the preconditions. Confirm the affected composer/provider flow reports the expected missing-secret state and that the operator has re-entered or reseeded any required staging secrets. Never reopen the predecessor archive in the current release.

At the **end of the deploy window**, destroy or secure the archive directories created above. Each is a long-lived copy of live encrypted secret material. It is only inert if `settings.secret_key` was **rotated** during this deploy; if the key was reused, the archive is decryptable with the running app's key, so an unattended snapshot directory is equivalent to leaving a readable copy of every staging secret on disk. If evidence retention requires keeping an archive, rotate `settings.secret_key` or move the archive to access-controlled storage.

### Failed Cutover

If the fresh current release fails, keep the service stopped, preserve the
failed fresh stores and logs for diagnosis, fix the current release, then repeat
the uninstall/recreate/reinstall procedure. Do not restore predecessor source
or databases as the repair path.

### Why DELETE Is Forbidden For Schema-Changing Phases

Row deletion does not add columns, CHECK constraints, FKs, or indexes. For schema-changing Phase plans (Phase 1A introduces NOT NULL `chat_messages.sequence_no`, NOT NULL `chat_messages.writer_principal`, NOT NULL `composition_states.provenance`, and the new `audit_access_log` table), row deletion leaves a stale DB shape that the startup schema validator correctly rejects. The validator is primarily a name/shape guard for expected tables, columns, CHECK names, and index names; it is not a compatibility migration engine and must not be treated as proof that stale CHECK expressions, partial-index predicates, or old table layouts are safe. Archive/delete/recreate is the only accepted cutover path for schema-changing phases. Archive and deletion must handle the SQLite main DB plus `-wal`, `-shm`, and `-journal` sidecars as a single artifact set; mixing a new main file with stale sidecars is not a valid reset.

## Failure Handling

- If the stop/go grep finds Landscape references to session identifiers, do not reset. Preserve both databases and create or update the rollout issue with the reference evidence.
- If startup still fails with `SessionSchemaError`, the running process is pointing at a different stale `sessions.db` than the one reset. Re-resolve `WebSettings.get_session_db_url()` from the live environment.
- If `systemctl` or `sudo` fails from the sandbox, report the exact blocker and have the operator run the host-side restart. Do not claim staging was restarted or live-verified from inside the sandbox.
