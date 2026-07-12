# AWS ECS Doctor CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `elspeth doctor aws-ecs [--init-schema] --json`, a non-persisting-by-default preflight that reports settings, database, schema, filesystem, dependency, and plugin readiness; only `--init-schema` may create schema objects.

**Architecture:** `src/elspeth/web/doctor.py` consumes Plan 01's pure deployment checks and Plan 02's target/probe/init APIs. Collection has four ordered phases: pure settings and logical-target validation; ephemeral filesystem and dependency/plugin checks; read-only connectivity/schema inspection of both databases; and, only for an eligible all-green preflight, initialization of repairable schema states followed by final verification. The target gate runs before any engine exists, both databases are inspected before either is mutated, and every one-shot engine is disposed in `finally`. A thin Typer subcommand emits the complete ordered `ContractCheck` report without opening `auth.db`.

**Read-only terminology:** The design spec requires doctor to prove filesystem permissions by creating, reading, flushing/fsyncing, and deleting a probe file as the runtime user. Therefore “read-only by default” means no persistent application, schema, configuration, or directory state is created or changed. The active probe runs in both modes, uses an unpredictable exclusive temporary file, unlinks it immediately while its descriptor remains open, and leaves no named artifact. `--init-schema` is the only mode that may persist changes, and those changes are limited to Plan 02's schema initializers.

**Tech Stack:** Python 3.13, Typer, SQLAlchemy 2.0, psycopg v3, pytest, `testcontainers[postgres]`.

**Depends on:**

- Tasks 1-2 / Filigree `elspeth-dffe064287`: Plans 01 and 02.
- Tasks 3-4 / Filigree `elspeth-397ac915b8`: Plans 06, 07, 09, 14, and the complete 15A→15B→15C chain must also be integrated so `aws_s3`, boto3, Jinja2-backed S3 key templates, `bedrock`, the operator-owned AWS OTLP posture, the effective web plugin policy, and both Bedrock Guardrail transforms can pass. Task 4 is the full Plan 03 handoff for this integrated slice. The integrated tree must already include Plan 08's load-bearing web-authorship gate before Plan 06/07 registrations and Plan 15B before Plan 15C registration, as required by the program DAG.
- Plan 12 owns the final integrated, zero-skip Docker evidence. Plan 03 records the exact command it must repeat.

**Plan 02 interfaces consumed:** `SchemaState`, `DatabaseTargetConflictError`, `SchemaInitBusyError`, `SchemaLockCleanupError`, `postgres_engine_kwargs`, `require_distinct_postgres_targets`, `probe_session_schema`, `probe_landscape_schema`, `init_session_schema`, and `init_landscape_schema`. Do not add another URL/search-path parser.

**Global Constraints:**

- Read raw `settings.session_db_url`, `settings.landscape_url`, and `settings.payload_store_path`; never use getters that silently fall back to local SQLite or derived paths.
- A failed/absent/non-PostgreSQL Plan 01 URL check or failed Plan 02 target check prevents all engine construction, probes, and initialization. Same-database targets pass only when Plan 02 proves distinct explicit single-schema search paths.
- The filesystem probe never calls `mkdir`. Missing `data_dir`, payload, or blob directories are provisioning/EFS failures in both modes; `--init-schema` does not create them or their parents. Before probing payload writability, apply the same existing-directory contract as `FilesystemPayloadStore`: no symlink and no group/world-write bits.
- `MISSING` and Landscape `PARTIAL` are repairable only under `--init-schema`. Session partial presence classifies `STALE`. `STALE` is never mutated and receives static drop/recreate instructions.
- Both database states must be known before either initializer runs. Any settings, target, filesystem, dependency, plugin, connectivity, or `STALE` failure blocks both initializers. A race after that preflight can still leave one target initialized and the other failed; initialization is idempotent, so the remedy is to resolve the reported failure and rerun.
- Errors expose static context plus exception class only. Never emit URLs, credentials, SQL, driver text, paths, exception messages, or `repr()` values.
- JSON output is a bare ordered list of `{name, ok, detail}` objects. Check names are unique and deterministic. Exit 1 when any check fails; text mode prints one line per check.
- Doctor never opens, creates, stats, or migrates `auth.db`.

---

### Task 1: Add the safe report shell and CLI

**Files:**

- Create: `src/elspeth/web/doctor.py`
- Modify: `src/elspeth/cli.py` (add a `doctor` sub-Typer beside the existing sub-app registrations)
- Create: `tests/unit/web/test_doctor.py`
- Create: `tests/unit/cli/test_doctor_command.py`

**Interfaces — Produces:**

```python
def sanitize_error(context: str, exc: BaseException) -> str: ...

def probe_directory_writable(label: str, path: Path | None) -> ContractCheck: ...

def plugin_and_dependency_checks() -> list[ContractCheck]: ...

def collect_checks(settings: WebSettings, *, init_schema: bool = False) -> list[ContractCheck]: ...
```

`probe_directory_writable` has one implementation for both modes:

1. Return a static failure when `path is None` or is not an existing directory. Never create it.
2. Call `tempfile.mkstemp(prefix=".doctor-probe-", dir=path)` so creation is unpredictable, exclusive (`O_EXCL`), and mode-restricted by the standard library.
3. Immediately `os.unlink(probe_name)` while the file descriptor is still open; wrap it with `os.fdopen(fd, "w+b")`, write a sentinel, flush, `os.fsync`, seek, and read the sentinel through the same descriptor. Do not reopen by pathname.
4. In `finally`, close any still-owned descriptor and unlink the name if the early unlink did not complete. Cleanup failure is a failing sanitized check, not success.

Use raw `settings.payload_store_path`. Derive the blob directory with `allowed_source_directories(str(settings.data_dir))[0]`, but only inspect it; no descendant creation is permitted. Before the payload active probe, use `lstat` to reject symlinks, non-directories, and `stat.S_IWGRP | stat.S_IWOTH`, then require `resolve(strict=True)`; convert every validation/stat/resolve failure to static label-only detail. This must match the constructor contract startup later enforces, so doctor cannot approve a path `FilesystemPayloadStore` rejects. Keep the three names `data_dir_writable`, `payload_store_writable`, and `blob_writable`.

`plugin_and_dependency_checks` isolates each capability so one failure preserves the rest of the report:

- `aws_s3_plugin`: call `get_shared_plugin_manager()` inside its own `try`, require `aws_s3` in both source and sink registrations, and catch sanitized `Exception`.
- `bedrock_provider`: lazily import `_PROVIDERS` and require `bedrock`; this is an explicit narrow private-registry read until the provider registry publishes a public capability accessor. Catch sanitized `Exception`, not only `ImportError`.
- `aws_operator_telemetry`: require the AWS ECS effective telemetry policy to resolve to one enabled generic `otlp` exporter at the fixed task-local endpoint, empty headers, explicit bounded resource identity, and `lifecycle` or `rows` granularity. Never print the endpoint or effective config.
- `bedrock_guardrail_plugins`: require both `aws_bedrock_prompt_shield` and `aws_bedrock_content_safety` plus the provider-neutral prompt-shield/content-safety capability mapping. This is registration/config-shape proof only; Plan 12 owns the live task-role call.
- `psycopg_dependency`, `boto3_dependency`, `ijson_dependency`, and
  `jinja2_dependency`: independently call `importlib.import_module` and
  independently catch sanitized `Exception`. The latter three prove the final
  source+sink `aws` extra, rather than letting plugin discovery's lazy imports
  mask an incomplete production install.

The CLI command lazily imports `_settings_from_env`, `ContractCheck`, `collect_checks`, and `sanitize_error`. Settings construction failure becomes the sole `settings_load` check. `collect_checks` retains a last-resort `doctor_internal_error` conversion, but ordinary filesystem, plugin, dependency, target, engine, probe, and init failures are caught at their own check boundary so successful checks are not discarded. Render JSON with `dataclasses.asdict`; text uses `OK`/`FAIL` lines.

- [ ] Write failing filesystem and capability tests first:

  - existing directory succeeds and no `.doctor-probe-*` name remains;
  - missing directory and `None` fail without creating anything;
  - payload symlink and group/world-writable directories fail before the active probe, matching `FilesystemPayloadStore`, with no secret path in detail;
  - a spy around `tempfile.mkstemp` proves unpredictable exclusive creation is used;
  - write/read/fsync/unlink/cleanup failures are sanitized and leave no named file when cleanup is possible;
  - two concurrent probes of the same directory do not collide;
  - raw exception messages containing a credential URL and secret path never appear;
  - missing/broken S3 manager, Bedrock registry import, psycopg, boto3, ijson,
    and Jinja2 each yield their named failure while all other checks remain
    present;
  - `collect_checks` uses raw `payload_store_path=None` rather than the fallback from `get_payload_store_path()`;
  - a pre-existing `auth.db` has identical bytes and stat metadata after collection and no check name contains `auth`.

- [ ] Implement `sanitize_error`, the active filesystem probe, isolated capability checks, and the initial ordered collector. At this task boundary, database checks may be static `not implemented until Task 2` failures; do not construct a temporary SQLite path or duplicate Plan 02 behavior.
- [ ] Add CLI tests for bare-list JSON, deterministic text output, exit 0/1, `--no-dotenv`, sanitized settings-load failure, and propagation of `--init-schema` to `collect_checks`.
- [ ] Add `doctor_app` and `doctor aws-ecs` to `cli.py`. Pin the exact ordered, unique output names in a unit test so later phases replace check values in place rather than reorder them.
- [ ] Run:

  ```bash
  uv run pytest tests/unit/web/test_doctor.py tests/unit/cli/test_doctor_command.py -q
  ```

  Expected: all Task 1 tests pass.

- [ ] Commit only Task 1 files:

  ```bash
  git add src/elspeth/web/doctor.py src/elspeth/cli.py tests/unit/web/test_doctor.py tests/unit/cli/test_doctor_command.py
  git commit -m "feat(cli): add aws-ecs doctor report shell"
  ```

---

### Task 2: Add fail-closed two-phase database inspection and initialization

**Files:**

- Modify: `src/elspeth/web/doctor.py`
- Modify: `tests/unit/web/test_doctor.py`
- Modify: `tests/unit/cli/test_doctor_command.py`

**Interfaces — Produces:**

```python
def database_target_check(session_url: str | None, landscape_url: str | None) -> ContractCheck: ...

def schema_check(label: str, state: SchemaState) -> ContractCheck: ...

def _inspect_database(
    label: str,
    raw_url: str,
    probe_fn: Callable[[Engine | Connection], SchemaState],
) -> tuple[SchemaState | None, ContractCheck]: ...

def _initialize_database(
    label: str,
    raw_url: str,
    probe_fn: Callable[[Engine | Connection], SchemaState],
    init_fn: Callable[[Engine], None],
) -> ContractCheck: ...
```

`database_target_check` calls only `require_distinct_postgres_targets`. `DatabaseTargetConflictError` becomes `ContractCheck("separate_db_targets", False, <static remedy>)`. If either Plan 01 database URL check failed, do not call it with bad input; return a static failure explaining that comparison was not attempted because the database URL contract failed.

Both engine helpers:

- identify PostgreSQL with `make_url(raw_url).drivername.split("+")[0] == "postgresql"`;
- add `connect_args={"connect_timeout": 10}` only for PostgreSQL;
- create session engines with `create_session_engine(raw_url, connect_args=..., **postgres_engine_kwargs(raw_url))`;
- create Landscape engines with `create_engine(raw_url, connect_args=..., **postgres_engine_kwargs(raw_url))` (do not pass a second `pool_pre_ping`);
- in `_inspect_database`, execute `SELECT 1` and call the Plan 02 probe through the same open `Connection`;
- in `_initialize_database`, call the Plan 02 initializer with the helper-owned engine while no outer connection is held. The initializer owns one checkout and performs DDL plus its postcondition probe on that lock-owning connection; after it returns, open one new connection and run `SELECT 1` plus the final doctor probe together on that connection;
- dispose every successfully constructed engine in `finally`, on success and on connect, probe, init, busy-lock, cleanup-uncertainty, compatibility, and unexpected-error paths.

`collect_checks` implements this exact orchestration:

1. Run Plan 01 checks and index them by name. URL eligibility requires passing `session_db_url` and `landscape_url`; deployment eligibility also requires `deployment_target`.
2. Run the Plan 02 target check before any engine construction. On conflict or URL-contract failure, emit static blocked schema checks and skip all database work.
3. Run filesystem, dependency, and plugin checks. They remain visible even if a database prerequisite failed.
4. If database prerequisites pass, inspect both targets read-only and retain both states. `CURRENT` is ready; `MISSING` and `PARTIAL` are repairable; `STALE` and operational errors are blockers.
5. In default mode, render both states with `schema_check`: `CURRENT` passes; `MISSING/PARTIAL` says rerun with `--init-schema`; `STALE` gives static drop/recreate instructions. Remove the unprovable claim that dropping any partially initialized database “loses nothing.”
6. In init mode, compute one eligibility decision after both probes. Every non-schema check must pass, and each database must be `CURRENT`, `MISSING`, or allowed `PARTIAL`. If not eligible, call neither initializer and retain/replace repairable schema checks with a static “not initialized because the complete preflight failed” detail.
7. Initialize only repairable targets. `_initialize_database` calls the Plan 02 initializer, whose return contract already means its lock-owning same-connection postcondition probe reached `CURRENT`; after return, doctor performs an independent `SELECT 1` and final probe on one new connection. Success requires that after-return probe to be `CURRENT` and reports `current; initialization completed or was already completed`. An already-current target is not initialized.
8. Catch in this order:
   - `SchemaInitBusyError`: static “another schema initialization is in progress; wait for it to finish and rerun”;
   - `SchemaLockCleanupError`: static “initialization may have completed but lock cleanup was not verified; investigate the database connection and rerun”;
   - `SessionSchemaError` / `SchemaCompatibilityError`: render `STALE` drop/recreate instructions;
   - other exceptions: a sanitized operational failure.

- [ ] Write failing target-gate tests for identical URLs, same database with absent/ambiguous search paths, and same database with two explicitly distinct parseable schemas. On every failing case, engine/probe/init spies must remain untouched. Assert the detail contains no supplied URL, user, password, host, database, or options text.
- [ ] Write failing engine-lifecycle tests that assert Plan 02 pool kwargs and PostgreSQL `connect_timeout=10` reach both factories, `SELECT 1` and the schema probe share one connection, and `dispose()` runs after success, connect failure, probe failure, initializer failure, busy lock, and cleanup uncertainty. Assert non-PostgreSQL input is blocked by Plan 01 before these helpers are reached through `collect_checks`.
- [ ] Write failing state tests for session `MISSING/CURRENT/STALE`, Landscape `MISSING/PARTIAL/CURRENT/STALE`, static and redacted details, and exact unique check ordering.
- [ ] Write failing two-phase tests:

  - session `MISSING` plus Landscape `STALE` initializes neither;
  - session `MISSING` plus Landscape connection failure initializes neither;
  - any filesystem, dependency, or plugin failure initializes neither;
  - both repairable states initialize and finish `CURRENT`;
  - one current and one repairable target calls only the repairable initializer;
  - a compatibility race becomes `STALE` without leaking its message;
  - busy and cleanup exceptions remain their named schema checks and never collapse to `doctor_internal_error`;
  - after the first initializer succeeds and the second fails, the report remains truthful and a rerun is safe/idempotent.

- [ ] Implement the target gate, engine helpers, two read-only probes before mutation, one eligibility decision, and final init verification. Delete the old `_target_key`/advisory comparison design entirely.
- [ ] Extend CLI tests so both lock-domain exceptions produce exit 1, valid complete JSON, their static remedy, and no raw cause text.
- [ ] Run:

  ```bash
  uv run pytest tests/unit/web/test_doctor.py tests/unit/cli/test_doctor_command.py -q
  uv run pytest tests/unit/web/test_deployment_contract.py tests/unit/web/test_schema_probe.py tests/unit/web/test_doctor.py tests/unit/cli/test_doctor_command.py -q
  ```

  Expected: all tests pass.

- [ ] Commit only the shared Task 2 files:

  ```bash
  git add src/elspeth/web/doctor.py tests/unit/web/test_doctor.py tests/unit/cli/test_doctor_command.py
  git commit -m "feat(cli): add safe schema initialization to aws-ecs doctor"
  ```

---

### Task 3: Prove the real CLI and concurrent initialization on PostgreSQL

**Files:**

- Create: `tests/testcontainer/web/test_doctor_aws_ecs_postgres.py`

Keep this file outside `tests/integration/`: that directory's `conftest.py` auto-adds the unrelated `integration` marker. Mark the new tests `pytest.mark.testcontainer`; the normal unit lane continues to exclude that marker.

Use one module-scoped `PostgresContainer("postgres:16-alpine", driver="psycopg")`. For each test, create two uniquely named logical databases from an autocommit admin connection, then derive `postgresql+psycopg` URLs with `URL.set(database=...).render_as_string(hide_password=False)`. Do not use `str(URL)`, which masks the password as `***` and produces an unusable subprocess environment URL. Drop the databases in fixture cleanup. This gives session and Landscape distinct Plan 02 targets without paying for two containers per test.

Create the `data_dir`, explicit `payload_store_path`, and derived blob directory before invoking doctor; their absence is an EFS provisioning failure and doctor never creates them. Supply complete production-shaped environment values, including `ELSPETH_WEB__DEPLOYMENT_TARGET=aws-ecs`, both DB URLs, host, secret key, shareable-link signing key, and explicit paths.

- [ ] Add `test_doctor_init_schema_cli_succeeds_against_fresh_postgres`: invoke `runner.invoke(app, ["--no-dotenv", "doctor", "aws-ecs", "--init-schema", "--json"], env=...)`; require exit 0, parse a bare list, require unique names and `all(item["ok"] for item in report)`, then independently construct engines and assert both Plan 02 probes return `SchemaState.CURRENT`.
- [ ] Add `test_concurrent_doctor_init_schema_cli_runs_are_safe`: launch two real subprocesses with the same environment and fresh database pair using

  ```python
  [sys.executable, "-m", "elspeth.cli", "--no-dotenv", "doctor", "aws-ecs", "--init-schema", "--json"]
  ```

  Start both before waiting. Use `try/finally`: on a timeout or assertion failure, terminate each live child, then kill and bounded-wait for any child that does not terminate, so fixture teardown can never race leaked doctor processes. Require both processes to exit 0 within a bounded timeout, both stdout payloads to be valid all-green JSON lists, credentials and URLs to be absent from both stdout and stderr, and independent final probes to report both schemas `CURRENT`. This proves concurrent-safe CLI composition. Plan 02 Task 6's `pg_locks` contention test remains the evidence that the advisory lock actually serialized the overlapping critical sections; both tests are required.
- [ ] Verify Docker first, then run the file with no accepted skips:

  ```bash
  docker info
  uv run pytest tests/testcontainer/web/test_doctor_aws_ecs_postgres.py -m testcontainer -q
  ```

  Expected: both commands exit 0 and pytest reports zero skips. If Docker is unavailable, status is `BLOCKED`, not pass/skip.

- [ ] Commit:

  ```bash
  git add tests/testcontainer/web/test_doctor_aws_ecs_postgres.py
  git commit -m "test: prove aws-ecs doctor CLI against PostgreSQL"
  ```

---

### Task 4: Run Plan 03 handoff gates

**Files:** Verify only files touched by Tasks 1-3. No new implementation files are expected.

- [ ] Run focused and subsystem tests:

  ```bash
  uv run pytest tests/unit/web/test_deployment_contract.py tests/unit/web/test_schema_probe.py tests/unit/web/test_doctor.py tests/unit/cli/test_doctor_command.py -q
  uv run pytest tests/unit/web/ tests/unit/cli/ -q
  docker info
  uv run pytest tests/testcontainer/web/test_doctor_aws_ecs_postgres.py -m testcontainer -q
  ```

  Expected: every command exits 0 and the Docker-backed file reports zero skips.

- [ ] Run repository static gates exactly:

  ```bash
  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  git diff --check
  ```

  Expected: every command exits 0.

- [ ] Because this slice handles external URLs, database state, and filesystem paths, use the `wardline-gate` skill and run:

  ```bash
  wardline scan . --fail-on ERROR
  ```

  Expected: exit 0. On a finding, follow scan → explain → boundary fix → rescan. Do not add a waiver or signed allowlist entry in this plan.

- [ ] Record the downstream handoff in the Plan 03 closeout comment:

  - Plan 10 provisions and mounts the `data_dir`, explicit payload, and blob directories before doctor; doctor never creates directories.
  - Plan 12 runs `docker info` followed by

    ```bash
    uv run pytest tests/testcontainer/web/test_schema_probe_postgres.py tests/testcontainer/web/test_doctor_aws_ecs_postgres.py -m testcontainer -q
    ```

    and requires both files to execute with zero skips on the integrated candidate.
  - The complete report and both CLI exits remain redacted; neither a successful unit lane nor Plan 02's helper-level concurrency proof substitutes for the real CLI evidence.

- [ ] A final commit is needed only if verification required a scoped correction. Stage exact paths; never use `git add -A` in a shared worktree and do not create an empty verification commit.
