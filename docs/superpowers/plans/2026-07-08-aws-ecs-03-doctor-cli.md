# AWS ECS Doctor CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add `elspeth doctor aws-ecs [--init-schema] --json`, a read-only-by-default preflight (mutating only under `--init-schema`) that reports settings, DB, schema, filesystem, dependency, and plugin readiness for the AWS ECS deployment target.

**Architecture:** A new orchestration module `src/elspeth/web/doctor.py` builds `list[ContractCheck]` by consuming plan 01's pure `validate_aws_ecs_settings` and plan 02's `probe_*`/`init_*`, plus three novel local pieces: an exception sanitizer, a create/read/fsync/delete writability probe, and engine construction that avoids `LandscapeDB(url)`/`LandscapeDB.from_url()` (the bare constructor eagerly creates/migrates schema; `from_url(create_tables=False)` still eagerly calls `_validate_schema()`, which raises on STALE instead of returning a classification — both conflict with plan 02's non-raising `probe_landscape_schema` contract this plan needs). The connectivity/schema probe reads `settings.session_db_url`/`settings.landscape_url` directly, never `get_session_db_url()`/`get_landscape_url()` — those getters silently fall back to a local SQLite path, and doctor must never construct an engine, let alone mutate schema under `--init-schema`, against a database the operator never configured. A thin `@doctor_app.command("aws-ecs")` in `cli.py` loads settings, calls the module, and emits a bare JSON list + exit code, mirroring the existing `plugins_app` sub-Typer pattern rather than `health`'s inline dict-of-checks shape.

**Tech Stack:** Python, Typer, SQLAlchemy, pytest, CliRunner, `testcontainers[postgres]` (Task 3).

**Depends on:** Tasks 1–2: `2026-07-08-aws-ecs-01-deployment-contract.md` (`ContractCheck`, `validate_aws_ecs_settings`, `DEPLOYMENT_TARGET_AWS_ECS`), `2026-07-08-aws-ecs-02-postgres-schema-support.md` (`SchemaState`, `probe_session_schema`, `probe_landscape_schema`, `init_session_schema`, `init_landscape_schema` — advisory locking lives inside the `init_*` functions, not here; `init_landscape_schema` creates on MISSING **and** PARTIAL via `create_all(checkfirst=True)`, `init_session_schema` creates on MISSING **only** — session PARTIAL is unreachable-defensive and raises like STALE, per plan 02's probe semantics). The plugin/dependency checks in Tasks 1–2 degrade gracefully (they *report* not-registered/not-importable; unit tests monkeypatch them), so Tasks 1–2 do **not** need the plugin plans. **Task 3 additionally depends on** `…-06-s3-source.md` + `…-07-s3-sink.md` (its `all(c.ok)` assertion requires `aws_s3` registered as source *and* sink, plus `boto3` via the `aws` extra) and `…-09-bedrock-provider.md` (`bedrock` in `_PROVIDERS`) — it is a Wave 3 task even though Tasks 1–2 are Wave 2.

**Global Constraints:** "The command is read-only by default... `--init-schema` is the only mutating mode. It may create missing schema objects in fresh Aurora databases. It must not drop, truncate, or auto-repair existing data." "If a database contains stale or incompatible schema, doctor must fail with explicit pre-1.0 operator instructions: drop or recreate the affected Aurora database/schema..." "`elspeth doctor aws-ecs` must not open `auth.db`" — this plan's checks touch only session/landscape DBs and data/payload/blob dirs, never `auth.db`. **PARTIAL semantics (binding, cross-plan):** PARTIAL is a correct-shape-but-incomplete schema (e.g. interrupted init), additively completable by `init_*`; it is never STALE and never gets the destructive drop/recreate message. `--init-schema` creates/completes on MISSING **and** PARTIAL; the drop/recreate message fires on STALE only. Read-only mode fails closed on MISSING/PARTIAL/STALE alike but the message differs: MISSING/PARTIAL → "rerun with --init-schema"; STALE → drop/recreate. **PARTIAL is landscape-only in practice:** `probe_session_schema` never returns it — an incomplete session table set is shape-unverifiable (`_validate_current_schema` raises on any table-set mismatch before checking shapes) and classifies STALE, so the session DB surfaces only MISSING/CURRENT/STALE here (plan 02). **Note (accepted gap, not fixed here):** the spec's 45s bounded-retry-with-backoff for Aurora cold starts (spec "Aurora Serverless v2 Considerations") is scoped to "the startup validate path" (plan 04's `_probe_with_connection_budget`/`enforce_aws_ecs_contract`), not to doctor. Doctor's one-shot connectivity probe uses only the 10s `connect_timeout`; a future plan can extend plan 04's retry helper to doctor if cold-start doctor failures prove disruptive in practice — not duplicated here to avoid cross-plan scope creep and word budget.

### Task 1: Read-only `web/doctor.py` module + `doctor aws-ecs` command

**Files:**
- Create: `src/elspeth/web/doctor.py`
- Modify: `src/elspeth/cli.py:60` (mirror `app.add_typer(composer_app, name="composer")` — add `doctor_app = typer.Typer(help=...)` + `app.add_typer(doctor_app, name="doctor")`; command body follows the `health` command pattern at `cli.py:2937` for lazy in-function imports, but NOT its `{status, value}` JSON shape, and NOT its `str(e)` leaks at `cli.py:3057-3061,3123-3127,3186-3191` — doctor's new code avoids repeating that anti-pattern via `sanitize_error`, it does not retroactively fix `health`)
- Test: `tests/unit/web/test_doctor.py`, `tests/unit/cli/test_doctor_command.py`

**Interfaces:**
- Consumes: `validate_aws_ecs_settings(settings) -> list[ContractCheck]` (plan 01); `probe_session_schema(engine) -> SchemaState`, `probe_landscape_schema(engine) -> SchemaState` (plan 02); `create_session_engine(url, **kwargs) -> Engine` (`src/elspeth/web/sessions/engine.py:34` — required by the `contract_invariants.session_engine_factory` lint; note its WAL/FK PRAGMA probe raises a bare `RuntimeError` embedding `engine.url!r`, so it must be caught alongside `SQLAlchemyError`, never left to propagate); `WebSettings.session_db_url`/`.landscape_url` (raw `Optional[str]` fields, config.py — read directly, **not** `get_session_db_url()`/`get_landscape_url()`, which silently fall back to a local `sqlite:///{data_dir}/...` path); `WebSettings.get_payload_store_path()` (config.py:653-657 — no silent-fallback hazard for a writability probe); `allowed_source_directories(str(settings.data_dir))[0]` for the blob root (`src/elspeth/web/paths.py:43-46`; signature is `(data_dir: str)`, cast `Path` with `str()` per the convention at `execution/service.py:538`); `_settings_from_env() -> WebSettings` (`src/elspeth/web/app.py:547`); `get_shared_plugin_manager()` (`src/elspeth/plugins/infrastructure/manager.py:297`) and `_PROVIDERS` (`src/elspeth/plugins/transforms/llm/transform.py:251`, imported lazily inside its check, wrapped in `try/except ImportError` so a missing `llm`/`aws` extra reports itself instead of crashing); `make_url` (`sqlalchemy.engine.url` — same safe-unguarded justification as plan 01's `_check_postgres_url`: `WebSettings._validate_db_url`, config.py:334-342, guarantees any non-`None` `session_db_url`/`landscape_url` is already parseable).
- Produces: `sanitize_error(context, exc) -> str` = `f"{context}: {type(exc).__name__}"`; `probe_writable(label, path) -> ContractCheck`; `schema_check(label, state) -> ContractCheck`; `_probe_and_check_schema(label, raw_url, probe_fn) -> ContractCheck` (private); `_check_separate_db_targets(session_url, landscape_url) -> ContractCheck` (private, advisory-only — always `ok=True`); `collect_checks(settings: WebSettings) -> list[ContractCheck]`.

```python
def sanitize_error(context: str, exc: Exception) -> str:
    return f"{context}: {type(exc).__name__}"

def probe_writable(
    label: str,
    path: Path,
    *,
    allow_create: bool = False,
    missing_remedy: str = "rerun with --init-schema to create it, or verify the EFS mount",
) -> ContractCheck:
    # Read-only-by-default discipline applies to the FILESYSTEM too: never
    # mkdir unless the caller is in the explicit mutating mode. Creating a
    # missing directory here would mask an unmounted EFS volume by silently
    # writing into the container's overlay filesystem — the probe would pass
    # while production state lands on ephemeral storage.
    if not path.is_dir():
        if not allow_create:
            return ContractCheck(f"{label}_writable", False, f"directory does not exist; {missing_remedy}")
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return ContractCheck(f"{label}_writable", False, sanitize_error(f"{label} not creatable", exc))
    # PID-suffixed filename: two doctor invocations racing the same EFS dir
    # (spec's deploy-retry scenario) never contend for one probe path.
    probe = path / f".doctor-probe-{os.getpid()}"
    try:
        with probe.open("wb") as fh:
            fh.write(b"ok")
            fh.flush()
            os.fsync(fh.fileno())
        probe.read_bytes()
        probe.unlink()
    except OSError as exc:
        return ContractCheck(f"{label}_writable", False, sanitize_error(f"{label} not writable", exc))
    return ContractCheck(f"{label}_writable", True, "writable")

def schema_check(label: str, state: SchemaState) -> ContractCheck:
    # Single source of truth for classification-detail text in BOTH modes.
    if state is SchemaState.CURRENT:
        return ContractCheck(f"{label}_schema", True, "current")
    if state in (SchemaState.MISSING, SchemaState.PARTIAL):
        return ContractCheck(f"{label}_schema", False, "schema missing or incomplete; rerun with --init-schema to create/complete it")
    return ContractCheck(  # STALE only: the destructive drop/recreate message
        f"{label}_schema", False,
        f"STALE: drop or recreate the {label} database via your environment's normal procedures, then rerun 'elspeth doctor aws-ecs --init-schema'",
    )

def _probe_and_check_schema(label: str, raw_url: str | None, probe_fn: Callable[[Engine], SchemaState]) -> ContractCheck:
    if raw_url is None:
        return ContractCheck(f"{label}_schema", False, "cannot probe: URL not configured")
    connect_args = {"connect_timeout": 10} if raw_url.startswith("postgresql") else {}
    build = create_session_engine if label == "session" else (lambda url, **kw: create_engine(url, pool_pre_ping=True, **kw))
    try:
        engine = build(raw_url, connect_args=connect_args)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        state = probe_fn(engine)
    except Exception as exc:  # bare RuntimeError (session PRAGMA probe) and SQLAlchemyError both land here
        return ContractCheck(f"{label}_schema", False, sanitize_error(f"{label} connection failed", exc))
    return schema_check(label, state)

def _target_key(raw_url: str) -> tuple[str, str, str]:
    # host/database never carry credentials in a parsed URL (those live in
    # separate .username/.password attributes, untouched here); "options" is
    # the conventional carrier for an explicit Postgres schema (e.g.
    # "-c search_path=landscape") and is compared opaquely -- classifying
    # sameness doesn't need to parse search_path itself.
    url = make_url(raw_url)
    return (url.host or "", url.database or "", url.query.get("options", ""))

def _check_separate_db_targets(session_url: str | None, landscape_url: str | None) -> ContractCheck:
    # Advisory only (spec's "should", not "must") -- ok=True on every branch,
    # per plan 01's framing: ContractCheck.ok is binary with no severity
    # tier, so an ok=False here would over-enforce a soft recommendation.
    # This IS the doctor-owned advisory output plan 01 deferred to, not a
    # new gate.
    if session_url is None or landscape_url is None:
        return ContractCheck("separate_db_targets", True, "cannot compare: one or both URLs not configured")
    if _target_key(session_url) == _target_key(landscape_url):
        return ContractCheck(
            "separate_db_targets", True,
            "session and landscape URLs resolve to the same logical target (host+database+schema); "
            "the spec recommends separate databases or schemas even when sharing one Aurora cluster",
        )
    return ContractCheck("separate_db_targets", True, "separate targets")

def collect_checks(settings: WebSettings) -> list[ContractCheck]:
    checks = list(validate_aws_ecs_settings(settings))
    # data_dir root: allow_create NEVER passed — a missing root IS the
    # unmounted-EFS signal, and --init-schema must not create it either
    # (Task 2 threads allow_create into the two SUBDIR probes only).
    checks.append(probe_writable(
        "data_dir", settings.data_dir,
        missing_remedy="verify the EFS volume is mounted at ELSPETH_WEB__DATA_DIR (doctor never creates the data_dir root)",
    ))
    checks.append(probe_writable("payload_store", settings.get_payload_store_path()))
    checks.append(probe_writable("blob", allowed_source_directories(str(settings.data_dir))[0]))
    checks.append(_probe_and_check_schema("session", settings.session_db_url, probe_session_schema))
    checks.append(_probe_and_check_schema("landscape", settings.landscape_url, probe_landscape_schema))
    checks.append(_check_separate_db_targets(settings.session_db_url, settings.landscape_url))
    checks.extend(_plugin_and_dependency_checks())
    return checks

def _plugin_and_dependency_checks() -> list[ContractCheck]:
    checks: list[ContractCheck] = []
    try:
        manager = get_shared_plugin_manager()
        has_s3 = "aws_s3" in {s.name for s in manager.get_sources()} and "aws_s3" in {s.name for s in manager.get_sinks()}
        checks.append(ContractCheck("aws_s3_plugin", has_s3, "registered" if has_s3 else "aws_s3 source/sink not registered"))
    except Exception as exc:
        checks.append(ContractCheck("aws_s3_plugin", False, sanitize_error("aws_s3 plugin check failed", exc)))
    try:
        from elspeth.plugins.transforms.llm.transform import _PROVIDERS
        checks.append(ContractCheck("bedrock_provider", "bedrock" in _PROVIDERS, "registered" if "bedrock" in _PROVIDERS else "bedrock provider not registered"))
    except ImportError as exc:
        checks.append(ContractCheck("bedrock_provider", False, sanitize_error("bedrock provider check failed", exc)))
    for mod in ("psycopg", "boto3"):
        try:
            importlib.import_module(mod)
            checks.append(ContractCheck(f"{mod}_dependency", True, "importable"))
        except ImportError as exc:
            checks.append(ContractCheck(f"{mod}_dependency", False, sanitize_error(f"{mod} dependency missing", exc)))
    return checks
```

CLI wiring wraps settings load AND `collect_checks()` separately — a defect in one check must never crash the whole command with a raw traceback:
```python
try:
    settings = _settings_from_env()
except Exception as exc:
    checks = [ContractCheck("settings_load", False, sanitize_error("settings load failed", exc))]
else:
    try:
        checks = collect_checks(settings)
    except Exception as exc:
        checks = [ContractCheck("doctor_internal_error", False, sanitize_error("doctor check collection failed", exc))]
```
JSON output: `json.dumps([dataclasses.asdict(c) for c in checks])`. Text mode: one line per check, `f"{'OK' if c.ok else 'FAIL'} {c.name}: {c.detail}"`. `raise typer.Exit(1)` if any `not ok`.

- [ ] Write failing tests in `tests/unit/web/test_doctor.py`: `test_probe_writable_reports_sanitized_detail_on_permission_error` (monkeypatch `Path.open` to raise `OSError` against an EXISTING `tmp_path` dir; `path` string absent from `detail`); `test_probe_writable_missing_dir_fails_closed_without_creating` (`probe_writable("data_dir", tmp_path / "nope")` → `ok is False`, `"does not exist"` in detail, and `(tmp_path / "nope").exists()` is still `False` afterward — the read-only probe must never mkdir, that's the unmounted-EFS mask); `test_probe_writable_allow_create_creates_missing_dir_and_probes` (same missing path with `allow_create=True` → `ok is True`, `detail == "writable"`, dir now exists); `test_schema_check_stale_reports_recreate_instructions` (`schema_check("session", SchemaState.STALE)` → `ok is False`, `"rerun"` in detail, `"drop"` in detail); `test_schema_check_partial_reports_init_schema_instructions_not_recreate` (`SchemaState.PARTIAL` → `ok is False`, `"--init-schema"` in detail, `"drop"` **not** in detail — closes the PARTIAL/STALE conflation); `test_collect_checks_includes_deployment_contract_and_writability_checks` (`WebSettings(session_db_url=None, ...)` → a check named `"session_db_url"` with `ok is False` is present, plus `"data_dir_writable"`/`"payload_store_writable"`/`"blob_writable"` all present); `test_probe_and_check_schema_skips_engine_construction_when_url_unset` (monkeypatch `create_session_engine` to raise if called; `_probe_and_check_schema("session", None, probe_session_schema)` → `ok is False`, `"not configured"` in detail, spy never invoked); `test_probe_and_check_schema_catches_engine_construction_runtime_error` (monkeypatch `create_session_engine` to raise `RuntimeError("... /secret/path ...")`; resulting `detail` omits the path); `test_collect_checks_never_touches_auth_db` (`tmp_path` with a pre-created `auth.db`; `session_db_url=landscape_url=None`; call `collect_checks`; assert `auth.db`'s bytes/mtime unchanged and no check name contains `"auth"`); `test_collect_checks_all_pass_returns_all_ok` (monkeypatch `elspeth.web.doctor.validate_aws_ecs_settings` to return all-`ok=True`, `probe_session_schema`/`probe_landscape_schema` to `SchemaState.CURRENT`, `_plugin_and_dependency_checks` to an all-`ok=True` list; settings use `session_db_url=landscape_url="sqlite:///:memory:"` — `create_session_engine` skips its WAL assertion for `:memory:`, so no real Postgres is needed; `all(c.ok for c in checks)`); `test_collect_checks_warns_when_session_and_landscape_share_same_target` (`session_db_url=landscape_url="postgresql://u:p@host/samedb"` → the `"separate_db_targets"` check has `ok is True` and `"same logical target"` in detail — advisory, not fail-closed); `test_collect_checks_reports_separate_targets_for_distinct_urls` (`session_db_url="postgresql://u:p@host/sessiondb"`, `landscape_url="postgresql://u:p@host/landscapedb"` → `ok is True`, detail `== "separate targets"`). Run `pytest tests/unit/web/test_doctor.py -x` — expect `ModuleNotFoundError: elspeth.web.doctor`.
- [ ] Implement `doctor.py` per above. Run again — expect PASS.
- [ ] Write `tests/unit/cli/test_doctor_command.py::test_doctor_aws_ecs_json_reports_checks_and_exits_nonzero_on_failure` — invoke `runner.invoke(app, ["--no-dotenv", "doctor", "aws-ecs", "--json"])` (repo has a real `.env`; mirror `test_web_command.py:97`'s `--no-dotenv` guard) with `ELSPETH_WEB__SESSION_DB_URL` unset via `monkeypatch.delenv`; assert `exit_code == 1`, `json.loads(result.stdout)` is a list of `{"name","ok","detail"}` dicts; `test_doctor_aws_ecs_exits_zero_and_prints_ok_lines_when_all_pass` (monkeypatch `doctor.collect_checks` to an all-`ok=True` list; `exit_code == 0`, text-mode stdout contains `"OK "`); `test_doctor_aws_ecs_reports_sanitized_error_on_settings_load_failure` (`--no-dotenv`, `ELSPETH_WEB__PORT=notanumber`; `exit_code == 1`, a `"settings_load"` entry present, raw exception text absent from stdout).
- [ ] Wire `doctor_app`/`doctor_aws_ecs` in `cli.py` per the CLI wiring block above. Run — expect PASS.
- [ ] `git add src/elspeth/web/doctor.py src/elspeth/cli.py tests/unit/web/test_doctor.py tests/unit/cli/test_doctor_command.py && git commit -m "feat(cli): add read-only elspeth doctor aws-ecs command"`

### Task 2: `--init-schema` mutating mode

**Files:** Modify `src/elspeth/web/doctor.py` (Task 1 anchor), `src/elspeth/cli.py` (Task 1 anchor); Test: `tests/unit/web/test_doctor.py`, `tests/unit/cli/test_doctor_command.py`.

**Interfaces:** Consumes `init_session_schema(engine) -> None`, `init_landscape_schema(engine) -> None` (plan 02); `SessionSchemaError` (`src/elspeth/web/sessions/schema.py:68`, a `RuntimeError` subclass, not `SQLAlchemyError`); `SchemaCompatibilityError` (`src/elspeth/core/landscape/database.py:130`, a bare `Exception` subclass, not `SQLAlchemyError`); `sqlalchemy.exc.SQLAlchemyError`. Produces: `_probe_and_check_schema` gains `init_fn` and `init_schema` params; `collect_checks(settings, *, init_schema: bool = False)`. `collect_checks` also threads `allow_create=init_schema` into the `payload_store` and `blob` `probe_writable` calls — creating those subdirectories is a legitimate act of the explicit mutating mode — but **never** into the `data_dir` root probe, whose absence must keep reading as an unmounted EFS volume even under `--init-schema`.

`_probe_and_check_schema` extends Task 1's version — same connect/probe try/except, then, before falling through to `schema_check`:
```python
def _probe_and_check_schema(label, raw_url, probe_fn, init_fn, *, init_schema: bool) -> ContractCheck:
    ...  # unchanged connect + probe_fn(engine) -> state, same except Exception branch
    if init_schema and state in (SchemaState.MISSING, SchemaState.PARTIAL):
        try:
            init_fn(engine)
            return ContractCheck(f"{label}_schema", True, "created")
        except (SessionSchemaError, SchemaCompatibilityError):
            # init_fn raises exactly these two types on its fail-closed
            # paths (plan 02 Task 3): its advisory-locked re-probe found
            # STALE — a race this function's outer probe_fn(engine) couldn't
            # have seen — or, session-side only, the defensive PARTIAL arm
            # (unreachable from probe_session_schema, which never returns
            # PARTIAL). Surface schema_check's STALE detail (the
            # drop/recreate operator instructions) rather than a sanitized
            # class name that would discard them.
            return schema_check(label, SchemaState.STALE)
        except SQLAlchemyError as exc:
            return ContractCheck(f"{label}_schema", False, sanitize_error(f"{label} schema init failed", exc))
    return schema_check(label, state)
```
`collect_checks` threads `init_fn=init_session_schema`/`init_landscape_schema` and `init_schema=init_schema` into both calls. STALE is never overridden (falls through to `schema_check`'s drop/recreate branch) regardless of `init_schema`.

- [ ] Write failing tests: `test_collect_checks_init_schema_creates_missing_subdirs_never_data_dir_root` (settings with `data_dir=tmp_path` existing but `payload_store_path`/blob subdirs absent; `collect_checks(settings, init_schema=True)` → `payload_store_writable`/`blob_writable` both `ok is True` and the subdirs now exist; then a second settings whose `data_dir=tmp_path / "unmounted"` does NOT exist → `collect_checks(settings, init_schema=True)` leaves it uncreated and `data_dir_writable.ok is False` with `"EFS"` in detail); `test_collect_checks_init_schema_creates_on_missing` (probe monkeypatched to `MISSING`, spy `init_session_schema`; `ok is True`, `detail == "created"`, spy called once); `test_collect_checks_init_schema_creates_on_partial` (same, `PARTIAL` — closes the high-severity MISSING-only gap); `test_collect_checks_init_schema_still_fails_closed_on_stale` (`STALE`; `ok is False`, spy **not** called, `"drop"` in detail); `test_collect_checks_init_schema_reports_stale_recreate_detail_on_race` (`MISSING`, spy `init_session_schema` raises `SessionSchemaError("stale race")`; `collect_checks` returns normally with `ok is False` and `"drop"` in detail — plan 02's drop/recreate operator instructions surface here, not a sanitized class name, and not a propagated exception); `test_collect_checks_init_schema_reports_sanitized_error_on_other_init_failure` (`MISSING`, spy `init_session_schema` raises `SQLAlchemyError("connection lost")`; `ok is False`, a sanitized detail, `"drop"` **not** in detail — distinguishes a genuine init-time DB error from the STALE-race path above). Run `pytest tests/unit/web/test_doctor.py -k init_schema` — expect `TypeError: collect_checks() got an unexpected keyword argument 'init_schema'`.
- [ ] Extend `_probe_and_check_schema`/`collect_checks` per above.
- [ ] Add `--init-schema` flag to `doctor_aws_ecs`, threaded to `collect_checks(settings, init_schema=init_schema)`. Add `tests/unit/cli/test_doctor_command.py::test_doctor_init_schema_flag_threads_through` (`--no-dotenv`). Run full file — expect PASS.
- [ ] `git add -A && git commit -m "feat(cli): add --init-schema mutating mode to doctor aws-ecs"`

### Task 3: PostgreSQL integration proof

**Files:** Create `tests/integration/web/test_doctor_aws_ecs_postgres.py`.

**Interfaces:** Consumes `PostgresContainer` (fixture pattern: `tests/integration/web/test_blobs_ready_hash_postgres.py:59-73`), `elspeth.web.doctor.collect_checks`. Closes the spec's Testing Strategy line naming this exact CLI invocation, otherwise untested by any subplan. **Note (accepted deviation):** the spec's Testing Strategy also names two *concurrent* `elspeth doctor aws-ecs --init-schema` CLI runs; that scenario is proxied, not re-run, here — it's covered at the `init_*` level by plan 02 Task 5's `test_concurrent_init_session_schema_serializes_via_advisory_lock`, since the advisory lock lives inside `init_session_schema`/`init_landscape_schema`, not in this task's single `collect_checks` invocation.

- [ ] Write `test_doctor_init_schema_succeeds_against_fresh_postgres` (`pytest.mark.testcontainer`): two fresh `PostgresContainer("postgres:16-alpine", driver="psycopg")`; build `WebSettings` with both URLs pointed at the containers, `data_dir`/`payload_store_path` under `tmp_path` (mkdir the `data_dir` root in the test's arrange step — doctor never creates it, even under `--init-schema`; the payload/blob subdirs are left absent so `init_schema=True` exercises their `allow_create` path), production-shaped `secret_key`/`shareable_link_signing_key`, `host="0.0.0.0"`; `collect_checks(settings, init_schema=True)` → `all(c.ok for c in checks)`.
- [ ] Run `pytest tests/integration/web/test_doctor_aws_ecs_postgres.py -m testcontainer` — expect PASS (clean skip without Docker).
- [ ] `git add tests/integration/web/test_doctor_aws_ecs_postgres.py && git commit -m "test: verify elspeth doctor aws-ecs --init-schema against real PostgreSQL"`
