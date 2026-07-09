# AWS ECS PostgreSQL Schema Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Give ELSPETH a `postgres` packaging extra and a `schema_probe` module that classifies and create-if-missing's the session and landscape schemas against PostgreSQL/Aurora, serialized via `pg_advisory_lock`, with explicit pool sizing.

**Architecture:** `schema_probe.py` orchestrates: `probe_*` classify state via one new non-mutating helper per existing schema module (`sessions/schema.py`, `core/landscape/database.py`); `init_*` wrap `metadata.create_all(bind=conn, checkfirst=True)` (additive-only) in a session-scoped `pg_advisory_lock`, raising on STALE. Pool kwargs pass through existing engine factories when the URL dialect is `postgresql`.

**Tech Stack:** SQLAlchemy 2.0 `postgresql+psycopg` (psycopg v3), `testcontainers[postgres]`.

**Depends on:** `2026-07-08-aws-ecs-01-deployment-contract.md` (not required here — this module is dialect-driven, not `deployment_target`-gated — but its startup wiring calls these functions).

**Global Constraints:** `postgres` extra ships psycopg v3; spec URLs use `postgresql+psycopg`. `init_*` acquire `pg_advisory_lock` (deterministic key per target) on PostgreSQL, create missing objects only, never drop/truncate/repair. aws-ecs pool: `pool_size=5, max_overflow=5, pool_pre_ping=True`. Integration tests: create+validate session/landscape DBs against real PostgreSQL; two concurrent `init_*` serialize via the lock without corruption.

### Task 1: `postgres` packaging extra
**Files:** Modify `pyproject.toml:94-122` (`dev`), `:181-235` (`all`).
- [ ] Add `postgres = ["psycopg[binary]>=3.2,<4"]` (mirror the `security` single-line extra at `:147-150`, or `mcp` at `:142-145`).
- [ ] Add the same line to `dev` (next to `testcontainers[postgres]`/`psycopg2-binary` at `:120-121`) and `all` (`:204-205`) — the new integration tests use `driver="psycopg"`, needing v3 installed.
- [ ] Run `uv sync --extra dev`; verify `python -c "import psycopg"` succeeds.
- [ ] Commit: `git add pyproject.toml uv.lock && git commit -m "chore: add postgres packaging extra (psycopg v3)"`.

### Task 2: Non-mutating schema-shape probes on the existing modules
**Files:** Modify `web/sessions/schema.py` (after `initialize_session_schema`, `:76`); `core/landscape/database.py` (after `_validate_schema`, `:971-1180`). Test: `tests/unit/web/sessions/test_schema.py`, `tests/unit/core/landscape/test_database_schema_probe.py` (new).
- [ ] Write failing test: in-memory sqlite + `metadata.create_all` minus one table → `probe_current_schema(engine)` is `False`; full schema → `True`. Run `pytest tests/unit/web/sessions/test_schema.py -k probe_current_schema`; expect `AttributeError`.
- [ ] Add `probe_current_schema(engine: Engine) -> bool` to `sessions/schema.py`: calls `_assert_schema_sentinels(engine)` then `_validate_current_schema(engine)` in `try/except SessionSchemaError: return False`, else `return True`. Reuses only same-file helpers.
- [ ] Add `LandscapeSchemaShape(Enum)` (`EMPTY, INCOMPLETE, DIVERGENT, MATCHES`; `EMPTY` = "no recognized ELSPETH tables", not "zero tables in the DB") and module-level `probe_schema_shape(engine: Engine) -> LandscapeSchemaShape` to `core/landscape/database.py`. First extract `_validate_schema`'s five FK/check/index divergence loops (`missing_fks, missing_composite_fks, forbidden_fks, missing_checks, missing_indexes`, inlined at `:1029-1104`, guarded on local `existing_tables`) into module-level `_collect_structural_divergences(inspector: Inspector, existing: set[str]) -> tuple[list, list, list, list, list]` — same five loops, `existing` replacing `existing_tables` — and update `_validate_schema` to call it and unpack the five lists (behavior-preserving rename). Then:
  ```python
  def probe_schema_shape(engine):
      from sqlalchemy import inspect
      inspector = inspect(engine)
      existing, expected = set(inspector.get_table_names()), set(metadata.tables)
      if not (existing & expected):
          return LandscapeSchemaShape.EMPTY
      divergent = bool(
          _collect_missing_required_columns(inspector)
          or _collect_token_outcomes_shape_errors(
              inspector, engine=engine, inspect_sqlite_indexes=(engine.dialect.name == "sqlite")
          )
          or any(_collect_structural_divergences(inspector, existing))
      )
      if divergent:
          return LandscapeSchemaShape.DIVERGENT
      if (expected - existing) - _ADDITIVE_TABLE_NAMES:
          return LandscapeSchemaShape.INCOMPLETE
      return LandscapeSchemaShape.MATCHES
  ```
  Divergence is checked *before* the missing-table check, not after: `_collect_missing_required_columns`/`_collect_token_outcomes_shape_errors`/`_collect_structural_divergences` all individually skip tables absent from `existing` (confirmed at `:534-574`,`:1029-1104`), so they safely report on whatever tables ARE present even when others are missing. A DB missing a table AND carrying a wrong-shaped existing table must classify `DIVERGENT`, not `INCOMPLETE` — the binding SchemaState semantics define PARTIAL as "correct-shape objects present but incomplete," which a divergent existing table isn't, and `checkfirst=True` would otherwise silently add the missing table while leaving the divergent one unrepaired and unreported. Never calls `create_all`/epoch-sync — read-only sibling of `_validate_schema`. `inspect_sqlite_indexes` is now dialect-aware (matches `_validate_schema`'s `startswith("sqlite")`) — a literal `False` would blind this probe to the ADR-019 stale-index drift class on SQLite.

  **Known limitation (round 3, stated so it is not over-claimed):** on PostgreSQL there is **no schema-epoch backstop** — the epoch mechanism rides SQLite's `PRAGMA user_version` (`database.py:1004` hardcodes `schema_epoch = 0` for non-sqlite URLs; `:1106` treats `0` as compatible), so Aurora drift detection is *structural only* (tables/columns/FKs/checks/indexes). A future semantics-only schema revision — an epoch bump with no structural change — would validate `CURRENT` against Aurora. Every epoch bump to date has also been structural, so this is acceptable for 0.7.0; a PostgreSQL schema-version table is future work. The spec's "stale schema fails closed → drop/recreate" acceptance criterion is therefore materially weaker on Aurora than on SQLite, and doctor/startup docs must not imply otherwise.
- [ ] Run both tests; expect PASS. Add: drop `run_attributions` (an `_ADDITIVE_TABLE_NAMES` member) from an otherwise-complete DB → `MATCHES`; an otherwise-complete DB missing one `_REQUIRED_INDEXES` entry → `DIVERGENT`; a DB missing one non-additive table AND carrying a required-column gap on a table that IS present → `DIVERGENT`, not `INCOMPLETE` (guards the reordering above).
- [ ] Before committing, run `pytest tests/unit/core/landscape/test_database_compatibility_guards.py tests/unit/core/landscape/test_schema_epoch_and_required_columns.py` — these exercise `_validate_schema`'s divergence branches directly; the `_collect_structural_divergences` extraction must not change any of their `SchemaCompatibilityError` assertions.
- [ ] Commit: `git add src/elspeth/web/sessions/schema.py src/elspeth/core/landscape/database.py tests/unit/web/sessions/test_schema.py tests/unit/core/landscape/test_database_schema_probe.py && git commit -m "feat: add non-mutating schema-shape probes for session and landscape DBs"`.

### Task 3: `schema_probe.py` — SchemaState, probe/init, advisory lock
**Files:** Create `src/elspeth/web/schema_probe.py`. Modify `src/elspeth/contracts/advisory_locks.py:53` (append new classid). Test: `tests/unit/web/test_schema_probe.py` (new).
**Interfaces — Produces:** `SchemaState(Enum)`, `probe_session_schema`, `probe_landscape_schema`, `init_session_schema`, `init_landscape_schema`, `AWS_ECS_POOL_KWARGS`, `postgres_engine_kwargs(url)`. Aliased imports: `from elspeth.web.sessions.models import metadata as session_metadata`, `from elspeth.core.landscape.schema import metadata as landscape_metadata` (both modules name their `MetaData` object plain `metadata`; aliasing avoids a collision since this module imports both).
- [ ] Write failing test: sqlite `:memory:`, no tables → `probe_session_schema(engine) is SchemaState.MISSING`; `create_all` minus a table → `STALE` (session **never** returns PARTIAL — partial presence is unverifiable, see the probe rationale below); full+correct → `CURRENT`; full-but-wrong-shape → `STALE`. For `probe_landscape_schema`: the same MISSING/CURRENT/STALE cases **plus** `create_all` minus a table → `PARTIAL` (landscape's shape checks verify the present tables first, so its PARTIAL is genuinely additively completable). Run `pytest tests/unit/web/test_schema_probe.py`; expect `ModuleNotFoundError`.
- [ ] Add to `contracts/advisory_locks.py`, matching `ELSPETH_SESSIONS_LOCK_CLASSID`'s (`:53`) comment style — correcting its "cluster-wide" claim opportunistically: advisory locks scope to the *connecting database*, not the cluster; classids only prevent collision within one database (the spec allows session/landscape to share one Aurora database via separate schemas, so `target` — not classid — is what keeps the two locks apart there):
  ```python
  # 0x53434845 = ASCII "SCHE". schema_probe.py's init_session_schema/
  # init_landscape_schema locks. Scoped to the connecting database
  # (advisory locks are per-database, not cluster-wide, despite this
  # file's framing above); target strings keep session vs landscape
  # apart even when both share one database via separate schemas.
  ELSPETH_SCHEMA_INIT_LOCK_CLASSID: int = 0x53434845
  ```
- [ ] Implement `schema_probe.py` with module-level `logger = structlog.get_logger()` (mirrors `web/app.py:213`), used by `_run_locked`'s unlock-failure branch below.
  ```python
  class SchemaState(Enum):
      MISSING = "missing"
      PARTIAL = "partial"
      CURRENT = "current"
      STALE = "stale"

  def _run_locked(engine: Engine, *, target: str, body: Callable[[Connection], None]) -> None:
      # Holds pg_advisory_lock(classid, hashtext(target)) on ONE connection;
      # create_all binds to this same conn (a separate pool checkout would
      # not share the lock). No-op lock on non-PostgreSQL.
      with engine.connect() as conn:
          locked = engine.dialect.name == "postgresql"
          if locked:
              # Bound the lock wait (round 3): a crashed holder self-heals
              # (postgres releases session advisory locks when it detects the
              # dead backend), but a hung-but-alive holder would otherwise
              # block this waiter forever — e.g. a second concurrent
              # `doctor --init-schema` from a deploy retry hanging with no
              # task stop-timeout. On timeout postgres raises
              # lock_not_available (an OperationalError); callers surface it
              # as "another schema init appears to be in progress; wait for
              # it or investigate the holder, then rerun".
              conn.execute(text(f"SET lock_timeout = '{_LOCK_WAIT_TIMEOUT}'"))
              conn.execute(text("SELECT pg_advisory_lock(:c, hashtext(:t))"),
                           {"c": ELSPETH_SCHEMA_INIT_LOCK_CLASSID, "t": target})
          try:
              body(conn)
              conn.commit()
          except Exception:
              if locked:
                  # body() may leave the tx aborted (real DDL error);
                  # rollback first so unlock runs clean, not masking it.
                  conn.rollback()
                  try:
                      conn.execute(text("SELECT pg_advisory_unlock(:c, hashtext(:t))"),
                                   {"c": ELSPETH_SCHEMA_INIT_LOCK_CLASSID, "t": target})
                      conn.commit()
                  except Exception:
                      logger.error("schema_probe.unlock_failed", target=target)
              raise
          else:
              if locked:
                  conn.execute(text("SELECT pg_advisory_unlock(:c, hashtext(:t))"),
                               {"c": ELSPETH_SCHEMA_INIT_LOCK_CLASSID, "t": target})
                  conn.commit()
  ```
  (`pg_advisory_xact_lock` considered instead — spec names `pg_advisory_lock` explicitly, and Task 5's lock-release test below only makes sense against a manual unlock path, so kept as-is.) `_LOCK_WAIT_TIMEOUT = "60s"` is a module constant (monkeypatchable for the bounded-wait test; `SET lock_timeout` scopes to the session, and the connection is closed by the `with` block, so no reset is needed).
  `init_session_schema(engine)`: `_body(conn)` closes over `engine` (probing is Engine-typed — `probe_session_schema` opens its own connection; passing the locked `Connection` in would `AttributeError`, SQLAlchemy 2.0's `Connection` has no `.connect()`) and reserves `conn` for `create_all` only: `state = probe_session_schema(engine)`; `STALE` **or** `PARTIAL` → `raise SessionSchemaError` with the spec's drop/recreate text verbatim ("drop or recreate the affected Aurora database/schema..., then rerun `elspeth doctor aws-ecs --init-schema`" — not just "run `--init-schema`"); `CURRENT` → return; `MISSING` (and only MISSING) → `session_metadata.create_all(bind=conn, checkfirst=True)`. The PARTIAL arm is defensive dead code — `probe_session_schema` never returns it (see the probe rationale below) — but if a future caller hands one in, `create_all` against an unverified partial schema is exactly the act this plan forbids, so it fails closed rather than falling through to create. Call `_run_locked(engine, target="session_schema", body=_body)`, then `if engine.dialect.name == "sqlite": _stamp_schema_sentinels(engine)` — after `_run_locked` commits, since `_stamp_schema_sentinels` opens its own connection and SQLite allows only one writer (idempotent; unreached on STALE). `init_landscape_schema` mirrors this shape (target=`"landscape_schema"`, `landscape_metadata.create_all`, raise `SchemaCompatibilityError` with the same spec wording, no stamp) with one deliberate divergence: its create branch runs on `MISSING` **and** `PARTIAL` — landscape PARTIAL is shape-verified by `probe_schema_shape`'s divergence-first ordering (Task 2), so it is genuinely additively completable, unlike session's. Its create branch omits `_create_additive_indexes()`: sole member `ix_tokens_run_id` is a normally-registered `Index` (`landscape/schema.py:1078`), so `create_all` already creates it here — the additive-indexes method is for backfilling a pre-existing table, out of scope (no SQLite-to-Aurora migration).
- [ ] Add `AWS_ECS_POOL_KWARGS = {"pool_size": 5, "max_overflow": 5, "pool_pre_ping": True}` and `postgres_engine_kwargs(url)` (returns it if `make_url(url).drivername.split("+")[0] == "postgresql"` else `{}`; mirrors `web/config.py:593`'s driver-split idiom).
- [ ] `probe_session_schema`/`probe_landscape_schema`:
  ```python
  def probe_session_schema(engine: Engine) -> SchemaState:
      with engine.connect() as conn:
          existing = set(inspect(conn).get_table_names())
      expected = set(session_metadata.tables)
      if not (existing & expected):
          return SchemaState.MISSING
      if expected - existing:
          # Unverifiable, not additively completable — see rationale below.
          return SchemaState.STALE
      return SchemaState.CURRENT if probe_current_schema(engine) else SchemaState.STALE

  def probe_landscape_schema(engine: Engine) -> SchemaState:
      return {
          LandscapeSchemaShape.EMPTY: SchemaState.MISSING,
          LandscapeSchemaShape.INCOMPLETE: SchemaState.PARTIAL,
          LandscapeSchemaShape.DIVERGENT: SchemaState.STALE,
          LandscapeSchemaShape.MATCHES: SchemaState.CURRENT,
      }[probe_schema_shape(engine)]
  ```
  `probe_landscape_schema` calls `probe_schema_shape` exactly once and maps its four states directly — no separate table-set pre-check (that would make `probe_schema_shape`'s own `EMPTY`/`INCOMPLETE` branches dead code on this call path, or risk disagreeing with it). `probe_session_schema` needs its own table-set split because `probe_current_schema` is boolean-only, with no landscape-side equivalent to delegate to — and, unlike landscape, it genuinely cannot check divergence-before-incompleteness: `_validate_current_schema` (`sessions/schema.py:189`) raises on any table-set mismatch before checking column shapes at all, so when some expected tables are missing the shape of the tables that ARE present is **unverifiable**. That is why partial presence maps to `STALE`, never `PARTIAL`: the binding cross-plan semantics define PARTIAL as *verified-correct-shape but incomplete, safe for `create_all(checkfirst=True)` to complete* — a claim the session probe cannot make, and acting on it would add tables around a possibly-divergent survivor set. Fail-closed STALE hands the operator the pre-1.0 drop/recreate instruction, which is also the only repair available when shape can't be proven. **`probe_session_schema` therefore never returns PARTIAL** (doctor/startup/readiness consumers see session as MISSING, CURRENT, or STALE only); making session genuinely PARTIAL-aware needs `_validate_current_schema` itself to tolerate a partial table set, out of this task's scope.
- [ ] Run tests; expect PASS.
- [ ] Commit: `git add src/elspeth/web/schema_probe.py src/elspeth/contracts/advisory_locks.py tests/unit/web/test_schema_probe.py && git commit -m "feat: add PostgreSQL-aware schema probe/init for session and landscape DBs"`.

### Task 4: Pool-kwargs passthrough on `LandscapeDB` + call-site wiring
**Files:** Modify `core/landscape/database.py:580,629,1295,1352` (`__init__`/`_setup_engine`/`from_url`, two `create_engine` sites); `web/app.py:874`; `web/execution/service.py:1100`. Test: `tests/unit/core/landscape/test_database_compatibility_guards.py` (existing file — no `tests/unit/core/landscape/test_database.py` exists).
- [ ] Write failing test `test_from_url_forwards_engine_kwargs_to_create_engine` in `test_database_compatibility_guards.py`, mirroring `test_from_url_dump_to_jsonl_requires_explicit_path_for_non_sqlite` (`:1494-1502`)'s `_CreateEngineFake` pattern — a live `from_url(pg_url, pool_size=3, ...)` round trip can't pass standalone, since `from_url` calls `instance._validate_schema()` (`:1385`), a real DB round trip, before returning. Monkeypatch `database_module.create_engine` with `_CreateEngineFake()`; call `LandscapeDB.from_url("postgresql://user@host/db", pool_size=3, max_overflow=2, pool_pre_ping=True)` inside `pytest.raises(...)` (the fake's `object()` return breaks `_validate_schema`'s `inspect(self.engine)`, mirroring the dump_to_jsonl tests' early exit); assert `create_engine_fake.assert_called_once_with("postgresql://user@host/db", echo=False, pool_size=3, max_overflow=2, pool_pre_ping=True)`.
- [ ] Add `**engine_kwargs: Any` to `__init__`/`_setup_engine`/`from_url`. TWO separate `create_engine(...)` sites need it, not one: `_setup_engine`'s at `:635-638` (used by `__init__`, hence by `execution/service.py:1100`'s raw `LandscapeDB(...)`) AND `from_url`'s own `create_engine(engine_url, echo=False)` at `:1352` (`from_url` never calls `_setup_engine` — it builds its engine directly via `_from_parts`/`cls.__new__`, bypassing `__init__`). Skip sqlcipher (aws-ecs never pairs `landscape_passphrase` with PostgreSQL). `_setup_engine`'s path has no dedicated test here — the `_CreateEngineFake` early-exit above needs `dump_to_jsonl`'s pre-`_validate_schema` `ValueError`, which `__init__` lacks — verify manually that `:635-638` also receives `**engine_kwargs`.
- [ ] Wire `postgres_engine_kwargs()` into `app.py:874`'s `create_session_engine(...)` and `execution/service.py:1100`'s `LandscapeDB(...)`. Scope note: only these two sites are wired here — `LandscapeDB.from_url(...)` is also constructed at ~15 other sites (`web/app.py:175`, four in `web/auth/audit.py`, one each in `web/execution/{accounting,outputs,diagnostics,discard_summary}.py`, `web/composer/tutorial_service.py`, `web/sessions/routes/runs.py`), left on SQLAlchemy's default pool — an owed follow-up, not swept here.
  Known gap, not fixed here: `execution/service.py:1100`'s raw `LandscapeDB(...)` still `create_all`s on first pipeline run against a fresh/empty aws-ecs landscape DB (or an additive-table gap), bypassing `doctor --init-schema` as the sole init path (`_validate_schema` still fails closed on genuine drift). Plan 02 is dialect-driven, not `deployment_target`-gated, so gating this needs plan 01's `deployment_contract`. **Owned by plan 11** (`2026-07-08-aws-ecs-11-landscape-write-gate.md`), which replaces this site's construction with a `create_tables`-gated factory call — plan 11 runs after this plan and explicitly preserves the `**postgres_engine_kwargs(url)` wiring this task adds here, so land this task as written; the factory migration subsumes it.
- [ ] Run test; expect PASS. Run `pytest tests/unit/web/ tests/unit/core/landscape/`; no regressions.
- [ ] Commit: `git add src/elspeth/core/landscape/database.py src/elspeth/web/app.py src/elspeth/web/execution/service.py tests/unit/core/landscape/test_database_compatibility_guards.py && git commit -m "feat: size PostgreSQL connection pools explicitly for aws-ecs"`.

### Task 5: PostgreSQL integration tests
**Files:** Create `tests/integration/web/test_schema_probe_postgres.py`.
- [ ] Mirror the `pg_engine` fixture at `test_blobs_ready_hash_postgres.py:59-73`, but `PostgresContainer("postgres:16-alpine", driver="psycopg")` (not default `psycopg2`); skip `initialize_session_schema` in the fixture — tests drive `init_session_schema`/`init_landscape_schema` themselves.
- [ ] `test_init_session_schema_creates_and_validates`: fresh container → `probe_session_schema == MISSING` → `init_session_schema(engine)` → `CURRENT`.
- [ ] `test_init_landscape_schema_creates_and_validates`: same shape against `core.landscape.schema.metadata`.
- [ ] `test_concurrent_init_session_schema_serializes_via_advisory_lock`: two threads call `init_session_schema(engine)` on the SAME container, synchronized with `threading.Barrier(2)` so both enter together (mirror `test_compose_loop_concurrent_sessions.py:443-488` — not an unsynchronized `Thread.start()/.join()` pair, which would pass even with the lock deleted). Stall one thread's `create_all` briefly (monkeypatch to sleep) while it holds the lock; from a third connection assert a `pg_locks` row with `locktype='advisory', granted=true, classid=ELSPETH_SCHEMA_INIT_LOCK_CLASSID` (mirror `test_advisory_lock_actually_acquired_on_postgres:220-283`) — a dialect-guard typo would otherwise silently no-op the lock without failing this test. After both threads join: no exception, `probe_session_schema == CURRENT`.
- [ ] `test_run_locked_releases_lock_after_body_failure`: call `schema_probe._run_locked(engine, target="probe_release", body=lambda conn: conn.execute(text("SELECT 1/0")))` inside `pytest.raises(DBAPIError)` — assert the error is the original divide-by-zero, not `InFailedSqlTransaction` (proves rollback-before-unlock). From a second connection, `SELECT pg_try_advisory_lock(:c, hashtext(:t))` on the same classid/target — assert `True` (released, not leaked), then `pg_advisory_unlock` to clean up. Also assert `init_session_schema(engine)` immediately after completes under `@pytest.mark.timeout(10)` — a leaked lock would hang with no timeout, silently, not fail visibly.
- [ ] `test_run_locked_lock_wait_is_bounded` (round 3): from a second connection take `pg_advisory_lock` on the same classid/target and hold it; `monkeypatch.setattr(schema_probe, "_LOCK_WAIT_TIMEOUT", "500ms")`; `_run_locked(engine, target=<same>, body=...)` raises `OperationalError` (postgres `lock_not_available`) within `@pytest.mark.timeout(10)` rather than blocking forever behind the hung-but-alive holder; release the second connection's lock in cleanup.
- [ ] Mark all `pytest.mark.testcontainer` (already registered, deselected by default per `pyproject.toml:424`).
- [ ] Run `pytest tests/integration/web/test_schema_probe_postgres.py -m testcontainer`; expect PASS (skips cleanly without Docker).
- [ ] Commit: `git add tests/integration/web/test_schema_probe_postgres.py && git commit -m "test: verify PostgreSQL schema create/validate and advisory-lock serialization"`.
