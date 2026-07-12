# AWS ECS Validate-Only Web Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In `ELSPETH_WEB__DEPLOYMENT_TARGET=aws-ecs`, make `create_app()` fail closed before any persistent startup mutation unless settings, logical database targets, mounted runtime directories, connectivity, and both schemas are valid and current; preserve local/default create-if-missing behavior.

**Architecture:** A new `aws_ecs_startup.py` owns the pure Plan 01/02 preflight, static redacted errors, bounded connection retry, and connection-bound schema probes. `create_app()` runs that AWS-only preflight before catalog/auth construction, so a rejected deployment cannot create `auth.db`, schema, or overlay-backed runtime directories. The existing local/default order and mutation behavior stay unchanged. Orphan reconciliation gains an explicit `create_tables` policy so later lifespan/background work cannot reintroduce DDL after the startup gate.

**Tech Stack:** Python 3.13, SQLAlchemy 2.0, psycopg v3, Typer/WebSettings contracts, pytest, PostgreSQL 16, `testcontainers[postgres]`.

**Depends on:**

- Plan 01: `ContractCheck`, `DEPLOYMENT_TARGET_AWS_ECS`, `validate_aws_ecs_settings`.
- Plan 02: `SchemaState`, `DatabaseTargetConflictError`, `require_distinct_postgres_targets`, `postgres_engine_kwargs`, connection-bound probes.
- Plan 03/10 handoff: doctor validates the same paths without creating directories; Plan 10 provisions `data_dir`, explicit payload, and blob directories before doctor/startup.
- Filigree implementation step: `elspeth-03cf981d4a`. This planning repair does not claim it.

**Required downstream handoffs:** Plan 05 wraps the lifespan orphan-finalization call with the same redacted recoverable-error posture established here; Plan 11 preserves this plan's explicit `create_tables` argument and seals every other web-layer Landscape writer; Plan 12 owns the exact zero-skip Docker command recorded in Task 5.

**Global Constraints:**

- Plan 01 URL checks and Plan 02 target separation run before directory inspection, auth/catalog construction, engine construction, or a probe. A failed/invalid URL is never passed to the target helper.
- Read raw `settings.session_db_url`, `settings.landscape_url`, and `settings.payload_store_path`; never invoke fallback getters in the AWS preflight.
- Same database targets pass only when Plan 02 proves two distinct explicit single-schema search paths. Identical, absent, or ambiguous paths fail closed without a connection.
- AWS startup never calls a schema initializer, `metadata.create_all`, `mkdir`, or a constructor that can create `auth.db` before the full validation gate succeeds.
- `data_dir`, explicit payload store, and derived blob directory must already exist as directories. The payload path must also satisfy `FilesystemPayloadStore`'s existing no-symlink and no-group/world-write rules before auth construction. Errors name only logical labels/environment variables, never configured paths.
- Probe session first, then Landscape, matching the spec. Any `MISSING`, `PARTIAL`, or `STALE` state fails immediately with static doctor instructions; startup never repairs it.
- Each dependency has a 45-second connection-retry window and 10-second connection timeout. Backoff/retry starts are deadline-aware; successful schema-inspection time is the spec's separate “plus probe time.”
- All database, path, and target errors are redacted. Public exceptions and logs never include URLs, credentials, SQL, driver messages, paths, exception messages, or raw causes.
- The Landscape probe engine is always disposed. A session engine that fails validation is explicitly disposed under `BaseException`; only a successfully validated long-lived session engine reaches app state/finalization.
- Local/default mode keeps today's directory creation and schema initialization byte-for-byte in its branch.

---

### Task 1: Build the fail-closed startup contract and retry module

**Files:**

- Create: `src/elspeth/web/aws_ecs_startup.py`
- Create: `tests/unit/web/test_aws_ecs_startup.py`

**Interfaces — Produces:**

```python
class AwsEcsStartupContractError(RuntimeError): ...

class AwsEcsSchemaNotReadyError(RuntimeError): ...

def enforce_aws_ecs_contract(settings: WebSettings) -> None: ...

def require_runtime_directories_mounted(settings: WebSettings) -> None: ...

def _probe_with_connection_budget(
    engine: Engine,
    probe: Callable[[Engine | Connection], SchemaState],
    *,
    label: str,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> SchemaState: ...

def validate_only_schema_or_raise(settings: WebSettings, session_engine: Engine) -> None: ...
```

`enforce_aws_ecs_contract` implements one pure ordered gate:

1. Call `validate_aws_ecs_settings` and index the ordered checks without discarding duplicates.
2. If any check fails, raise `AwsEcsStartupContractError` containing only failing check names, a static environment-remedy sentence, and `Run 'elspeth doctor aws-ecs' for full diagnostics.` Plan 01 details are already static, but startup does not need to echo them.
3. Only after both URL checks pass, assert the two raw URLs are non-`None` and call `require_distinct_postgres_targets`.
4. Translate `DatabaseTargetConflictError` to a static `AwsEcsStartupContractError` using `raise ... from None`; never include either URL or the original exception text.

`require_runtime_directories_mounted` checks, without writing:

- `settings.data_dir`;
- raw `settings.payload_store_path` (not `get_payload_store_path()`);
- `allowed_source_directories(str(settings.data_dir))[0]` for the blob root.

All three must be existing directories. For the payload path, mirror the existing non-creating checks in `FilesystemPayloadStore`: use `lstat`, require a directory, reject a symlink, reject `stat.S_IWGRP | stat.S_IWOTH`, and require `resolve(strict=True)` to succeed. Wrap blob-path derivation plus all path stat/resolve work in a boundary catch for `OSError`, `RuntimeError`, and validation `ValueError`; translate it to a static `AwsEcsStartupContractError` from `None`. Report missing/invalid labels and known environment variable names only. Never interpolate a path, call `mkdir`, or inspect `auth.db`.

`_probe_with_connection_budget` opens one `Connection` per attempt and calls `probe(conn)` on that same connection. It retries only `OperationalError`; a returned state is never retried. The first attempt starts immediately. After a failure:

- compute `remaining = deadline - monotonic()` and stop at `remaining <= 0`;
- reserve `_CONNECT_TIMEOUT_SECONDS` before another attempt; if `remaining < 10`, stop rather than launching a connection that can cross the retry deadline;
- sleep `min(backoff, remaining - 10)` (zero is allowed), with capped sequence 1, 2, 4, 8, 8…;
- recompute the remaining time after sleep and refuse the next attempt if the deadline/reserved timeout condition no longer holds;
- emit `aws_ecs_schema_probe_retry` with only `label`, one-indexed `attempt`, `elapsed_seconds`, and `exc_class`;
- on exhaustion, raise a static `AwsEcsSchemaNotReadyError` from `None` pointing to doctor.

Non-operational `SQLAlchemyError`, `SessionSchemaError`, and `SchemaCompatibilityError` are translated immediately to the same static error from `None`; these schema exceptions may contain object/SQL detail and must not escape. Unexpected programming exceptions such as `TypeError` propagate immediately and are not retried. The 45-second value budgets connection failures/backoff; an already-successful connection's structural inspection is the spec's separately acknowledged probe time. Define and use `_CONNECT_TIMEOUT_SECONDS = 10` and `_CONNECT_RETRY_BUDGET_SECONDS = 45` everywhere instead of restating literals in code.

`validate_only_schema_or_raise`:

1. Probe the passed session engine. Any non-`CURRENT` state raises a static `AwsEcsSchemaNotReadyError` naming `session_schema` and doctor; Landscape is not contacted after a known-failing session state.
2. Assert raw `settings.landscape_url` is present, then create a throwaway engine with `connect_args={"connect_timeout": _CONNECT_TIMEOUT_SECONDS}` and `**postgres_engine_kwargs(raw_url)`.
3. Probe Landscape through `_probe_with_connection_budget`; any non-`CURRENT` state raises the equivalent static error.
4. Dispose the Landscape engine in `finally`, including engine/probe/error paths where construction succeeded.

- [ ] Write the failing Plan 01/02 gate tests first:

  - failed `session_db_url`/`landscape_url` checks never call `require_distinct_postgres_targets`;
  - identical targets and same-database absent/ambiguous search paths raise a static contract error;
  - different databases and explicitly distinct parseable schemas pass;
  - credential-bearing URLs and target-helper causes are absent from `str(exc)`, `repr(exc)`, and captured logs.

- [ ] Write directory tests for each missing path independently. Assert the error names only `data_dir`, `payload_store`, or `blob`, the secret configured path is absent, no path is created, and no `auth.db` is touched. Pin raw `payload_store_path=None` as failure rather than accepting the fallback getter. Add payload symlink and group/world-writable cases, plus injected secret-bearing `lstat`/`resolve` failures; the startup gate and a direct `FilesystemPayloadStore` construction must agree that each invalid shape is rejected.
- [ ] Write retry tests with an injectable fake clock:

  - two `OperationalError`s then `CURRENT` yields sleeps 1 and 2;
  - `MISSING` returns once without retry;
  - `TypeError` propagates once without retry;
  - backoff caps at 8;
  - sleep is clipped so ten seconds remain for the next connection attempt;
  - no new attempt starts when remaining time is below ten seconds;
  - equality with the deadline (`>=`) is exhausted;
  - terminal operational/non-operational SQLAlchemy, session-schema, and Landscape-compatibility failures become static `AwsEcsSchemaNotReadyError` with no original cause text;
  - retry logs contain only the allowed fields and omit a credential/SQL/path sentinel embedded in the DBAPI exception.

- [ ] Write state/order/lifecycle tests for `validate_only_schema_or_raise`: session is probed before Landscape; session non-current prevents Landscape engine construction; Landscape uses the raw URL, timeout, and Plan 02 pool kwargs; all states other than `CURRENT` fail; Landscape engine disposal occurs after success, non-current state, operational exhaustion, and unexpected `BaseException` from the probe.
- [ ] Implement the module and run:

  ```bash
  uv run pytest tests/unit/web/test_aws_ecs_startup.py -q
  ```

  Expected: all tests pass.

- [ ] Commit:

  ```bash
  git add src/elspeth/web/aws_ecs_startup.py tests/unit/web/test_aws_ecs_startup.py
  git commit -m "feat(web): add fail-closed aws-ecs startup preflight"
  ```

---

### Task 2: Run AWS validation before any auth or service mutation

**Files:**

- Modify: `src/elspeth/web/app.py` (`create_app` settings/directory/session setup)
- Modify: `tests/unit/web/test_app.py` (new `TestAwsEcsValidateOnlyStartup`)

**AWS-only ordering in `create_app`:** Immediately after `app.state.settings = settings`, and before catalog/auth setup:

1. `enforce_aws_ecs_contract(settings)`;
2. `require_runtime_directories_mounted(settings)`;
3. build the session engine from raw `settings.session_db_url` with `connect_args={"connect_timeout": _CONNECT_TIMEOUT_SECONDS}` and `**postgres_engine_kwargs(raw_url)`;
4. call `validate_only_schema_or_raise(settings, session_engine)`;
5. on any `BaseException`, dispose the session engine and re-raise;
6. after success, retain that prevalidated engine and register its app lifetime finalizer before proceeding to catalog/auth construction.

This ordering is load-bearing: `LocalAuthProvider.__init__` eagerly creates/alters `auth.db`, and lifespan's `FilesystemPayloadStore` calls `mkdir(parents=True)`. Neither may run before the AWS contract, directory, target, and schema gate succeeds. The directory gate ensures the later payload-store constructor sees an already-provisioned path and performs no creation.

Replace the current unconditional directory block with two branches:

- AWS: run the preflight above and call no `mkdir`/initializer.
- default: retain `settings.data_dir.mkdir(parents=True, exist_ok=True)` and `(settings.data_dir / "runs").mkdir(exist_ok=True)` exactly.

At the existing session setup block, AWS reuses the already-validated engine without calling `initialize_session_schema`; default retains `get_session_db_url()`, `create_session_engine`, and `initialize_session_schema` exactly. Do not install a duplicate finalizer for the AWS engine.

- [ ] Add contract/target ordering tests with spies proving a failed Plan 01 check or Plan 02 target conflict constructs neither session nor Landscape engine and never calls any probe.
- [ ] Add one missing-directory test for each of data, payload, and blob, plus payload-symlink and unsafe-mode tests. Assert no engine is built, no path is created, no auth database exists, and a secret path string is absent from the public exception.
- [ ] Add `test_aws_ecs_schema_failure_precedes_local_auth_mutation`: valid mounted paths, `auth_provider="local"`, absent `auth.db`, session probe `MISSING`; `create_app` raises doctor guidance and `auth.db` still does not exist.
- [ ] Parametrize session and Landscape state failures across `MISSING`, `PARTIAL`, and `STALE`; use distinct valid PostgreSQL URLs. Assert the correct label and no initializer call.
- [ ] Add session-engine disposal tests for session state failure, Landscape state failure, exhausted connectivity, `SchemaCompatibilityError`/unexpected probe exception, and `KeyboardInterrupt`. Every failure disposes exactly once; success retains the engine and installs its finalizer.
- [ ] Add a successful AWS test proving session-before-Landscape order, no directory creation, no `initialize_session_schema`, and app construction proceeds only after both states are `CURRENT`.
- [ ] Preserve default behavior with tests proving omitted directories are created, session initialization is called, the AWS helpers are not called, and current default tests remain unchanged.
- [ ] Run:

  ```bash
  uv run pytest tests/unit/web/test_app.py -k AwsEcsValidateOnlyStartup -q
  uv run pytest tests/unit/web/test_app.py -q
  ```

  Expected: all tests pass.

- [ ] Commit only the Task 2 files:

  ```bash
  git add src/elspeth/web/app.py tests/unit/web/test_app.py
  git commit -m "feat(web): validate aws-ecs state before startup mutation"
  ```

---

### Task 3: Close the orphan-reconciliation DDL gap

**Files:**

- Modify: `src/elspeth/web/app.py` (`_finalize_orphaned_landscape_runs`, `_periodic_orphan_cleanup`, lifespan and background call sites)
- Modify: `tests/unit/web/test_app.py` (`TestOrphanLandscapeReconciliation`, `TestPeriodicOrphanCleanup`)

**Interfaces:**

- `_finalize_orphaned_landscape_runs(landscape_url, cancelled_runs, *, create_tables: bool = True)` passes the flag explicitly to `LandscapeDB.from_url`.
- `_periodic_orphan_cleanup(..., *, landscape_url: str | None = None, create_tables: bool = True)` forwards it.
- Both AWS call sites pass `create_tables=False`; default passes `True`.
- Add `SchemaCompatibilityError` to the existing periodic recoverable tuple and keep logging class-only (`periodic_orphan_cleanup_failed`, `exc_class`). Do not log URL, path, SQL, exception text, or traceback.

Plan 05 later wraps the lifespan one-shot call with the same `(SQLAlchemyError, OSError, SchemaCompatibilityError)` posture. This is cleanup after the real fail-closed gate, so graceful retry/readiness degradation is correct; it does not weaken schema validation.

- [ ] Add a direct fresh-database test with a cancelled run carrying non-`None` `landscape_run_id`: `create_tables=False` raises `SchemaCompatibilityError` and creates no tables. Avoid a vacuous record that returns before opening Landscape.
- [ ] Add a fail-once periodic test proving `SchemaCompatibilityError` logs only its class, leaks no sentinel, and the loop survives to a second iteration.
- [ ] Add a real-call-site lifespan test recording both `create_tables` kwargs: AWS passes `False`, default passes `True`, and the pending periodic coroutine is cancelled/awaited cleanly.
- [ ] Keep the existing default orphan-finalization test green to prove `create_tables=True` behavior is unchanged.
- [ ] Run:

  ```bash
  uv run pytest tests/unit/web/test_app.py -k "OrphanLandscape or PeriodicOrphanCleanup" -q
  ```

  Expected: all tests pass.

- [ ] Commit:

  ```bash
  git add src/elspeth/web/app.py tests/unit/web/test_app.py
  git commit -m "fix(web): prevent aws-ecs orphan cleanup schema creation"
  ```

---

### Task 4: Prove validate-only startup with DDL-denied PostgreSQL roles

**Files:**

- Create: `tests/testcontainer/web/test_aws_ecs_validate_only_startup.py`

Keep this outside `tests/integration/`, whose `conftest.py` auto-adds the unrelated `integration` marker. Mark the module only `pytest.mark.testcontainer`.

Use one module-scoped `PostgresContainer("postgres:16-alpine", driver="psycopg")`. For each test, create fresh distinct session and Landscape databases from an autocommit admin connection and derive credential-preserving URLs with `URL.set(database=...).render_as_string(hide_password=False)`. Generate database and role identifiers from lowercase UUID hex, validate them against `^[a-z0-9_]+$`, and still quote them with `psycopg.sql.Identifier`; never interpolate identifiers into SQL. Initialize schemas only through Plan 02 owner/admin engines, then create a per-test runtime login role. In each database, `REVOKE CREATE ON SCHEMA public FROM PUBLIC`, grant that role CONNECT, schema USAGE, and read access, and grant no CREATE privilege. Prove the role cannot execute `CREATE TABLE` before using its URLs for startup. Dispose every owner/runtime/app engine before dropping databases/roles in fixture `finally`.

Create `data_dir`, explicit payload store, and blob root with safe modes before every CLI/app invocation. Supply the full Plan 01 contract: target, distinct raw URLs, container host, production-shaped secrets, and explicit paths.

- [ ] `test_aws_ecs_startup_fails_closed_on_missing_session_schema`: both databases fresh/empty; session fails first with `AwsEcsSchemaNotReadyError`, no schema objects appear.
- [ ] `test_aws_ecs_startup_fails_closed_when_only_landscape_schema_missing`: initialize session only; startup reaches and rejects Landscape; session remains unchanged.
- [ ] `test_aws_ecs_startup_succeeds_with_current_schema_under_ddl_denied_roles`: initialize both as owner, establish that each runtime role's harmless rolled-back `CREATE TABLE` raises SQLAlchemy `ProgrammingError` whose psycopg `.orig.sqlstate == "42501"` (`InsufficientPrivilege`), snapshot table/column/index/constraint catalog identities, run `create_app`, then require identical catalogs and both probes `CURRENT`. Explicitly dispose `app.state.session_engine`.
- [ ] `test_aws_ecs_startup_rejects_same_or_ambiguous_target_before_connecting`: unit tests are authoritative for spy ordering; this Docker case uses an invalid same-target pair and asserts the static target error rather than a DBAPI error.
- [ ] Verify Docker and execute with zero accepted skips:

  ```bash
  docker info
  uv run pytest tests/testcontainer/web/test_aws_ecs_validate_only_startup.py -m testcontainer -q
  ```

  Expected: both commands exit 0 and pytest reports zero skips. Docker unavailable is `BLOCKED`, never pass/skip.

- [ ] Commit:

  ```bash
  git add tests/testcontainer/web/test_aws_ecs_validate_only_startup.py
  git commit -m "test(web): prove ddl-free aws-ecs startup on PostgreSQL"
  ```

---

### Task 5: Run Plan 04 handoff gates

**Files:** Verify files touched by Tasks 1-4. No new implementation files are expected.

- [ ] Run focused and subsystem tests:

  ```bash
  uv run pytest tests/unit/web/test_aws_ecs_startup.py tests/unit/web/test_app.py tests/unit/web/test_deployment_contract.py tests/unit/web/test_schema_probe.py -q
  uv run pytest tests/unit/web/ -q
  docker info
  uv run pytest tests/testcontainer/web/test_aws_ecs_validate_only_startup.py -m testcontainer -q
  ```

  Expected: all commands exit 0 and the Docker-backed file reports zero skips.

- [ ] Run repository static gates exactly:

  ```bash
  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  git diff --check
  ```

  Expected: every command exits 0.

- [ ] Because this slice handles external URLs, filesystem paths, database state, retry logging, and startup trust boundaries, use the `wardline-gate` skill and run:

  ```bash
  wardline scan . --fail-on ERROR
  ```

  Expected: exit 0. On a finding, follow scan → explain → boundary fix → rescan. Do not add a waiver or signed allowlist entry in this plan.

- [ ] Record downstream handoffs in the Plan 04 closeout comment:

  - Plan 05 preserves class-only orphan-cleanup logging and uses the exact three-exception recoverable tuple without weakening the pre-bind schema gate.
  - Plan 11 retains explicit `create_tables` at both `app.py` call sites and its AST seal must pass after this plan.
  - Plan 12 runs `docker info` and then:

    ```bash
    uv run pytest tests/testcontainer/web/test_schema_probe_postgres.py tests/testcontainer/web/test_doctor_aws_ecs_postgres.py tests/testcontainer/web/test_aws_ecs_validate_only_startup.py -m testcontainer -q
    ```

    All three files must execute with zero skips on the integrated candidate.

- [ ] Create a final commit only if verification required a scoped correction. Stage exact files; never use `git add -A` and do not create an empty verification commit.
