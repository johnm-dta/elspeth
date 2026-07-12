# AWS ECS PostgreSQL Schema Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship psycopg v3 support and a fail-closed PostgreSQL schema classifier/initializer for the session and Landscape databases, with full structural validation, explicitly bounded pooling, and cleanup-safe advisory-lock serialization.

**Architecture:** A new dialect-aware `core/schema_shape.py` compares every present table against SQLAlchemy metadata; it becomes the shared source of structural truth for the existing session validator and the new Landscape probe. `web/schema_probe.py` maps those facts into `MISSING/PARTIAL/CURRENT/STALE`, and its initializers mutate only a truly empty target or the explicitly additive Landscape objects while holding one per-database PostgreSQL session advisory lock. Probes, DDL, postcondition verification, and lock cleanup all use the same checked-out `Connection` so a size-one pool cannot deadlock.

**Tech Stack:** Python 3.13, SQLAlchemy 2.0, `postgresql+psycopg` (psycopg v3), PostgreSQL 16, `testcontainers[postgres]`, pytest.

**Depends on:** `2026-07-08-aws-ecs-01-deployment-contract.md` for the `aws-ecs` URL/settings contract. The implementation in this plan remains dialect-driven rather than `deployment_target`-gated.

**Required downstream handoffs:** This plan owns one pure PostgreSQL logical-target normalizer/check in `schema_probe.py`; Plans 03/04 consume it to hard-fail identical or unprovably-distinct session/Landscape targets before constructing engines or running probes. Plan 03 must surface `SchemaInitBusyError` and `SchemaLockCleanupError` separately; Plan 11 must preserve this plan's `PARTIAL`/`STALE` boundary; and Plan 12 owns the zero-skip Docker-backed PostgreSQL gate. Session and Landscape metadata both own incompatible `runs` tables, so same-database support means distinct, explicitly parseable PostgreSQL schemas/search paths, never the same or ambiguous logical schema.

**Global Constraints:**

- `postgres` ships psycopg v3; production URLs use `postgresql+psycopg`.
- `MISSING` means zero user tables. A non-empty foreign target is `STALE` and is never mutated.
- Session schemas are all-or-nothing: any partial table set is `STALE`.
- Landscape `PARTIAL` is limited to missing `_ADDITIVE_TABLE_NAMES` (`auth_events`, `run_attributions`) and/or `_ADDITIVE_INDEX_NAMES` (`ix_tokens_run_id`) after every surviving object passes full shape validation. A missing core table is `STALE`, not auto-recreated.
- Structural validation covers table sets, column order/names/types/nullability/PK/defaults, FKs, checks, unique constraints, and indexes including uniqueness, columns, and dialect predicate. PostgreSQL still has no semantics-only epoch backstop; that explicitly accepted limitation does not permit structural blind spots.
- Both initializers use the same advisory-lock key inside each connecting database. This safely over-serializes session and Landscape initialization when distinct schemas share one database and removes the unsafe same-schema cross-kind race.
- Lock timeout state is transaction-local; lock cleanup is `BaseException`-safe; a failed/unproven unlock invalidates the physical connection instead of returning it to the pool.
- Long-lived PostgreSQL pools use `pool_size=5`, `max_overflow=5`, `pool_pre_ping=True`.
- Docker-backed PostgreSQL tests must execute and pass. Unavailable Docker is `BLOCKED`, not a clean acceptance skip.

---

### Task 1: Add the `postgres` packaging extra

**Files:**

- Modify: `pyproject.toml:94-122` (`dev`)
- Modify: `pyproject.toml:181-235` (`all`)
- Modify: `uv.lock`

- [ ] Add `postgres = ["psycopg[binary]>=3.2,<4"]`, mirroring the existing single-purpose extras.
- [ ] Add `psycopg[binary]>=3.2,<4` to `dev` next to `testcontainers[postgres]`/`psycopg2-binary`, and to `all`. Keep psycopg2 for the existing testcontainer modules that still use testcontainers' default driver.
- [ ] Regenerate the lock and verify both the development and standalone production extra:

  ```bash
  uv sync --extra dev
  uv lock --check
  uv run --isolated --no-dev --frozen --extra postgres python -c "import psycopg; assert psycopg.__version__.split('.')[0] == '3'"
  ```

  Expected: every command exits 0; the import resolves psycopg v3 from the frozen lock.

- [ ] Commit:

  ```bash
  git add pyproject.toml uv.lock
  git commit -m "chore: add postgres packaging extra (psycopg v3)"
  ```

---

### Task 2: Make metadata structurally verifiable across SQLite and PostgreSQL

**Files:**

- Create: `src/elspeth/core/schema_shape.py`
- Modify: `src/elspeth/core/landscape/schema.py:192`
- Modify: `src/elspeth/web/sessions/schema.py:176-540`
- Create: `tests/unit/core/test_schema_shape.py`
- Modify: `tests/unit/web/sessions/test_schema.py`

**Interfaces — Produces:**

```python
@dataclass(frozen=True, slots=True)
class SchemaShapeIssue:
    subject: str
    expected: object
    actual: object


def collect_metadata_shape_issues(
    inspector: Inspector,
    metadata: MetaData,
    *,
    dialect: Dialect,
    present_tables: AbstractSet[str],
    allowed_missing_index_names: AbstractSet[str] = frozenset(),
) -> tuple[SchemaShapeIssue, ...]: ...


def probe_current_schema(bind: Engine | Connection) -> bool: ...
```

- [ ] Write a PostgreSQL DDL compilation regression for `runs.seeded_from_cache`. Its current `Boolean(..., server_default=text("0"))` compiles as invalid PostgreSQL `BOOLEAN DEFAULT 0`. Assert PostgreSQL DDL contains a dialect-native false default and not `BOOLEAN DEFAULT 0`.
- [ ] Change the model to import SQLAlchemy `false` and use `server_default=false()`. Assert SQLite compilation remains valid and Task 6's real PostgreSQL round trip inserts a row without explicitly supplying `seeded_from_cache`, then reads `False`.
- [ ] Before implementing the collector, add failing parameterized tests using fresh SQLite metadata plus one deliberately rebuilt minimal table/index per case. Pin every dimension below, including a same-named index on wrong columns or reversed column order, a wrong partial-index predicate, `String(64)` changed to `Text`, nullability drift, default drift, FK `ondelete` drift, the session `proposal_events` FK losing `deferrable=True`/`initially="DEFERRED"`, an unnamed Landscape composite unique constraint changing columns, and `ck_runs_openrouter_catalog_source` changing the persisted literal `'live'` to `'LIVE'`. Each mutation is isolated so an earlier mismatch cannot mask a later comparison bug. Also add green control cases asserting freshly created full session and Landscape schemas produce no issues once the collector exists.
- [ ] Move the reusable comparison mechanics currently embedded in `web/sessions/schema.py` into `core/schema_shape.py`; `core` must not import `web`. Preserve the session validator's static partial-index dialect-symmetry guard and SQLite trigger check in their current module.
- [ ] Make `collect_metadata_shape_issues` collect, rather than raise on, every mismatch for the tables named by `present_tables`. For each table it must compare:

  1. exact ordered column names;
  2. primary-key membership and nullability;
  3. column type SQL compiled through the live dialect after case/whitespace normalization;
  4. server-default presence and normalized SQL value, including PostgreSQL's reflected outer parentheses and type casts;
  5. the complete FK set: constrained columns, referred schema/table/columns, `onupdate`, `ondelete`, `deferrable`, `initially`, and `match`, normalizing default-schema/`NO ACTION` reflection only where semantically equivalent;
  6. named CHECK constraint names and normalized compiled/reflected SQL;
  7. every `UniqueConstraint` by semantic column signature, including unnamed composites, plus unique-`Index` name, ordered columns, and predicate;
  8. non-unique index names, uniqueness, ordered columns, and dialect-specific `WHERE` predicate.

  Unexpected integrity-bearing objects or same-named/different-shaped objects are issues. Additional ordinary non-unique indexes are the sole tolerated extra-object class. An index in `allowed_missing_index_names` may be absent, but if present its full shape must still match. Do not use the curated Landscape `_REQUIRED_*` lists as a substitute for this comparison: they cover historical high-value drift classes, not the complete 320-column/58-index metadata surface.
- [ ] Keep normalization deliberately narrow and tested. Collapse whitespace and remove redundant balanced outer parentheses/table qualification. Case-normalize type SQL and only unquoted SQL keywords/identifiers in expressions; preserve single-quoted, double-quoted, escaped, and dollar-quoted literal bytes exactly so `'live'` can never compare equal to `'LIVE'`. Normalize PostgreSQL reflection-only casts for literal defaults. Honor SQLAlchemy `ddl_if` dialect filtering when constructing the expected constraint/index surface. Do not erase operators, column names, lengths, index predicates, deferrability, initial mode, or referential actions. An unfamiliar difference fails closed as an issue.
- [ ] Pin the known dialect equivalences explicitly rather than discovering them in production: PostgreSQL `FLOAT`/`DOUBLE PRECISION`; SQLite's loss of `DateTime(timezone=True)` metadata; PostgreSQL's implicit sequence/default for an autoincrement integer PK (`run_coordination_events.seq`); absent FK action versus `NO ACTION`; referred schema `None` versus PostgreSQL `public`; PostgreSQL literal casts such as `'{}'::text`; PostgreSQL's `IN (...)` to `ANY (ARRAY[...])` CHECK rewrite; predicate casts/qualification; and PostgreSQL's duplicate reflection of unique constraints as indexes. A fresh SQLite schema and a fresh PostgreSQL 16 schema must canonicalize with zero issues before any drift test is trusted.
- [ ] Treat an additional ordinary non-unique index as compatible: it changes performance, not accepted writes. Additional columns, PK members, FKs, CHECKs, unique constraints, or unique indexes are divergent because they can reject or reinterpret runtime writes.
- [ ] Refactor `_validate_current_schema` to call the shared collector after its exact table-set check. Convert the first `SchemaShapeIssue` back through `_schema_error(subject, expected=..., actual=...)` so existing actionable exception formatting remains intact.
- [ ] Add public non-mutating `probe_current_schema(bind) -> bool` beside `initialize_session_schema`. It runs the sentinel and full current-schema validators on the supplied `Engine | Connection`, returns `False` only for `SessionSchemaError`, and never creates or stamps objects.
- [ ] Run the parameterized comparator tests and verify the previously failing cases now pass.
- [ ] Run:

  ```bash
  uv run pytest tests/unit/core/test_schema_shape.py tests/unit/web/sessions/test_schema.py -q
  ```

  Expected: PASS.

- [ ] Commit:

  ```bash
  git add src/elspeth/core/schema_shape.py src/elspeth/core/landscape/schema.py src/elspeth/web/sessions/schema.py tests/unit/core/test_schema_shape.py tests/unit/web/sessions/test_schema.py
  git commit -m "feat: add dialect-aware metadata shape validation"
  ```

---

### Task 3: Add fail-closed Landscape shape classification

**Files:**

- Modify: `src/elspeth/core/landscape/database.py:530-1165`
- Create: `tests/unit/core/landscape/test_database_schema_probe.py`
- Modify: `tests/unit/core/landscape/test_database_compatibility_guards.py`

**Interfaces — Produces:**

```python
class LandscapeSchemaShape(Enum):
    EMPTY = "empty"
    FOREIGN = "foreign"
    INCOMPLETE = "incomplete"
    DIVERGENT = "divergent"
    MATCHES = "matches"


def probe_schema_shape(bind: Engine | Connection) -> LandscapeSchemaShape: ...


def create_additive_indexes(bind: Engine | Connection) -> None: ...
```

- [ ] Write failing state-lattice tests for all branches:

  - zero user tables → `EMPTY`;
  - zero user tables with an incompatible non-zero SQLite epoch → `DIVERGENT`;
  - unrelated table only, or full Landscape tables plus any unrelated table → `FOREIGN`;
  - full correct metadata → `MATCHES`;
  - missing only `auth_events`, `run_attributions`, or `ix_tokens_run_id` → `INCOMPLETE`;
  - missing any non-additive/core table → `DIVERGENT`;
  - an additive gap plus any surviving shape error → `DIVERGENT`;
  - incompatible non-zero SQLite `PRAGMA user_version` → `DIVERGENT`.

- [ ] Implement the classifier in this exact order:

  ```python
  inspector = inspect(bind)
  existing = set(inspector.get_table_names())
  expected = set(metadata.tables)

  if _sqlite_epoch_is_incompatible(bind):
      return LandscapeSchemaShape.DIVERGENT
  if not existing:
      return LandscapeSchemaShape.EMPTY
  if existing - expected:
      return LandscapeSchemaShape.FOREIGN

  present = existing & expected
  if not present:
      return LandscapeSchemaShape.FOREIGN

  issues = collect_metadata_shape_issues(
      inspector,
      metadata,
      dialect=bind.dialect,
      present_tables=present,
      allowed_missing_index_names=_ADDITIVE_INDEX_NAMES,
  )
  if issues:
      return LandscapeSchemaShape.DIVERGENT

  missing_tables = expected - existing
  if missing_tables - _ADDITIVE_TABLE_NAMES:
      return LandscapeSchemaShape.DIVERGENT

  if missing_tables or _missing_additive_indexes(inspector, present):
      return LandscapeSchemaShape.INCOMPLETE
  return LandscapeSchemaShape.MATCHES
  ```

  `Engine` and `Connection` both expose `.dialect`; `inspect()` accepts either. `_sqlite_epoch_is_incompatible` must read through the supplied connection when one is supplied rather than checking out a second connection. It checks the non-zero epoch before the empty-table branch so a tableless file carrying an incompatible Landscape epoch is not mistaken for a fresh target.
- [ ] Extract the body of `LandscapeDB._create_additive_indexes()` into module-level `create_additive_indexes(bind)`. The method delegates to it. This is required because `MetaData.create_all(checkfirst=True)` does **not** create a missing index on an already-existing table.
- [ ] Align `LandscapeDB._validate_schema()` with the new truth source while preserving its existing high-signal error details. It must reject an incompatible non-zero SQLite epoch even when no tables exist, and explicitly reject `existing_tables - expected_tables` before calling the shared collector, so direct `LandscapeDB` opens and `probe_schema_shape` cannot disagree about foreign/extra tables. Add the full-shape issues to its aggregated incompatibility error. When `_require_existing_schema=True` (`from_url(create_tables=False)`), every expected table—including additive tables—must exist; when table creation is permitted, only `_ADDITIVE_TABLE_NAMES` may be absent from an otherwise-existing schema. Missing core tables always raise.
- [ ] Update the existing `create_tables=False` tests: missing `auth_events` or `run_attributions` must now raise `SchemaCompatibilityError` without mutation. Keep the missing `ix_tokens_run_id` policy explicit: validate-only opens fail on `INCOMPLETE` through the probe/startup gate, while direct legacy read-only `LandscapeDB` opens may tolerate this non-integrity performance gap until doctor repairs it.
- [ ] Run:

  ```bash
  uv run pytest tests/unit/core/landscape/test_database_schema_probe.py tests/unit/core/landscape/test_database_compatibility_guards.py tests/unit/core/landscape/test_schema_epoch_and_required_columns.py -q
  ```

  Expected: PASS; existing compatibility error assertions remain recognizable.

- [ ] Commit:

  ```bash
  git add src/elspeth/core/landscape/database.py tests/unit/core/landscape/test_database_schema_probe.py tests/unit/core/landscape/test_database_compatibility_guards.py
  git commit -m "feat: classify Landscape schema state without mutation"
  ```

---

### Task 4: Implement connection-bound schema probes and cleanup-safe initialization

**Files:**

- Create: `src/elspeth/web/schema_probe.py`
- Modify: `src/elspeth/web/sessions/schema.py:76-176`
- Modify: `src/elspeth/contracts/advisory_locks.py:1-60`
- Create: `tests/unit/web/test_schema_probe.py`

**Interfaces — Produces:**

```python
class SchemaState(Enum):
    MISSING = "missing"
    PARTIAL = "partial"
    CURRENT = "current"
    STALE = "stale"


class SchemaInitBusyError(RuntimeError): ...
class SchemaLockCleanupError(RuntimeError): ...
class DatabaseTargetConflictError(ValueError): ...


@dataclass(frozen=True, slots=True)
class PostgresLogicalTarget:
    host: str
    port: int
    database: str
    explicit_schema: str | None


AWS_ECS_POOL_KWARGS = {
    "pool_size": 5,
    "max_overflow": 5,
    "pool_pre_ping": True,
}


def postgres_engine_kwargs(url: str | URL) -> dict[str, object]: ...
def postgres_logical_target_key(url: str | URL) -> PostgresLogicalTarget: ...
def require_distinct_postgres_targets(session_url: str | URL, landscape_url: str | URL) -> None: ...
def probe_session_schema(bind: Engine | Connection) -> SchemaState: ...
def probe_landscape_schema(bind: Engine | Connection) -> SchemaState: ...
def init_session_schema(engine: Engine) -> None: ...
def init_landscape_schema(engine: Engine) -> None: ...
```

- [ ] Before implementing the module, write failing unit tests for every state, a foreign non-empty target, a tableless SQLite target carrying a foreign session application ID or incompatible session/Landscape epoch, missing core/additive Landscape objects, no-op `create_all`, timeout SQLSTATE mapping, body `KeyboardInterrupt`, false unlock, unlock exception, original-error preservation, connection invalidation, same-connection body/verify, size-one-pool behavior, immutable pool-kwargs behavior, and log/public-message redaction. Inject cleanup/acquisition failures containing a sentinel credential and raw SQL fragment; capture structlog and assert neither appears in exception text or any logged field.
- [ ] Add failing pure target tests: separate host/port/database tuples pass without requiring search-path options; the same host/port/database with both paths absent or one absent/one explicit (including explicit `public`) fails closed; the same database with two distinct explicitly parsed single-schema search paths passes; `search_path=Foo` and `search_path=foo` conflict after PostgreSQL unquoted-identifier normalization; a multi-schema, duplicated, conflicting, quoted, `$user`, or otherwise unparseable `options` value fails closed; missing host/database and non-PostgreSQL driver families fail with the same static redacted error; credentials never participate in the key or appear in errors; PostgreSQL driver variants normalize together; absent port normalizes to 5432 and host case normalizes to lower case.
- [ ] Extend the session sentinel and validation helpers to accept `Engine | Connection`. When passed a `Connection`, they execute on it directly, never call `.connect()`, and never commit or roll back—the `_run_locked` caller owns that transaction. The existing `Engine` path retains `initialize_session_schema(engine)` behavior; refactor `_stamp_schema_sentinels` so only its self-owned Engine transaction commits.
- [ ] Implement session classification as: no user tables first run `_assert_schema_sentinels` on the same bind—zero/current sentinels → `MISSING`, but a foreign application ID or incompatible non-zero epoch → `STALE`; non-empty with no session-table overlap → `STALE`; any partial/extra table set → `STALE`; exact table set plus successful full validation → `CURRENT`; any validation error → `STALE`. Session never returns `PARTIAL`.
- [ ] Map Landscape shapes exactly: `EMPTY → MISSING`, `INCOMPLETE → PARTIAL`, `MATCHES → CURRENT`, and `FOREIGN/DIVERGENT → STALE`.
- [ ] Add `ELSPETH_SCHEMA_INIT_LOCK_CLASSID = 0x53434845` with a corrected file-level comment: PostgreSQL advisory locks are scoped to the connecting database, not cluster-wide. Both initializers use the same target string, `"elspeth_schema_init"`, so they serialize even when session and Landscape schemas share one database.
- [ ] Implement `_run_locked(engine, *, target, body, verify)` with these invariants:

  - `body` and `verify` receive the same checked-out `Connection`;
  - PostgreSQL calls parameterized `SELECT set_config('lock_timeout', :timeout, true)` before `pg_advisory_lock`; the third argument makes the setting transaction-local;
  - SQLSTATE `55P03` raises static, redacted `SchemaInitBusyError` chained from the `OperationalError`;
  - acquisition ambiguity under any other `BaseException` invalidates the physical connection;
  - after `body(conn)`, commit DDL, then call `verify(conn)` while the session lock remains held;
  - cleanup runs for every `BaseException`, rolls back any active transaction before unlocking, and requires `pg_advisory_unlock(...)` to return literal `True`;
  - unlock false/error invalidates the physical connection; it raises `SchemaLockCleanupError` only when there is no earlier body/commit/verify exception, otherwise it logs exception classes only and preserves the original exception;
  - no raw URL, SQL, credentials, or DBAPI message enters logs or public exception text.

- [ ] Implement the initializers:

  - `init_session_schema`: under the lock, `CURRENT` returns, `MISSING` runs `session_metadata.create_all(bind=conn, checkfirst=True)` and stamps SQLite sentinels through that connection, and `STALE/PARTIAL` raises `SessionSchemaError` with the drop/recreate instruction.
  - `init_landscape_schema`: under the lock, `CURRENT` returns, `MISSING/PARTIAL` runs `landscape_metadata.create_all(bind=conn, checkfirst=True)` followed by `create_additive_indexes(conn)`, and `STALE` raises `SchemaCompatibilityError` with the same operator instruction.
  - Each `verify(conn)` re-probes after the DDL commit and requires `CURRENT`; a no-op or incomplete `create_all` must never be reported as success.

- [ ] `postgres_engine_kwargs` uses `make_url`, accepts `postgresql` driver variants, returns a fresh `dict(AWS_ECS_POOL_KWARGS)` for PostgreSQL, and `{}` otherwise.
- [ ] Implement `postgres_logical_target_key` without connecting: parse only psycopg's canonical `options=-csearch_path=<identifier>` / `options=-c search_path=<identifier>` single-schema forms. Absent options produce `explicit_schema=None`, not `public`, because role/database settings and PostgreSQL's default `"$user", public` path cannot be proven without connecting. Lowercase accepted unquoted identifiers exactly as PostgreSQL does; reject quoted identifiers, comma-separated/multiple schemas, repeated/conflicting `search_path` settings, `$user`, or any options string whose effective single schema cannot be proven. `require_distinct_postgres_targets` returns immediately for different normalized host/port/database tuples; for the same tuple it requires two non-`None`, distinct normalized schemas, otherwise raising static `DatabaseTargetConflictError` without interpolating either URL.
- [ ] Run the unit tests and verify all previously failing state, lock, pool, and redaction cases pass.
- [ ] Run:

  ```bash
  uv run pytest tests/unit/web/test_schema_probe.py tests/unit/web/sessions/test_schema.py -q
  ```

  Expected: PASS.

- [ ] Record the Plan 03 consumer contract in the implementation handoff: catch `SchemaInitBusyError` and `SchemaLockCleanupError` before generic `SQLAlchemyError`/`RuntimeError` handling. Busy returns the static “another schema initialization is in progress” remedy; cleanup uncertainty returns a static “initialization may have completed but lock cleanup was not verified; investigate and rerun” remedy. Neither may collapse to a generic exception class.
- [ ] Commit:

  ```bash
  git add src/elspeth/web/schema_probe.py src/elspeth/web/sessions/schema.py src/elspeth/contracts/advisory_locks.py tests/unit/web/test_schema_probe.py
  git commit -m "feat: add locked PostgreSQL schema probe and initialization"
  ```

---

### Task 5: Thread explicit engine kwargs through both Landscape construction paths

**Files:**

- Modify: `src/elspeth/core/landscape/database.py:580-640,1295-1391`
- Modify: `src/elspeth/web/app.py:874-876`
- Modify: `src/elspeth/web/execution/service.py:1095-1104`
- Modify: `tests/unit/core/landscape/test_database_compatibility_guards.py`

- [ ] Add `**engine_kwargs: Any` to `LandscapeDB.__init__`, `_setup_engine`, and `from_url`. `__init__` must call `self._setup_engine(**engine_kwargs)`; both distinct `create_engine` sites forward the kwargs. Reject engine kwargs on the SQLCipher branch rather than silently ignoring them.
- [ ] Add automated tests for both paths:

  - `LandscapeDB.from_url(pg_url, create_tables=False, pool_size=3, max_overflow=2, pool_pre_ping=True)` reaches its `create_engine` call with those values while `_validate_schema` is monkeypatched to no-op; `create_tables=False` prevents the classmethod's direct `metadata.create_all(engine)` branch;
  - the raw `LandscapeDB(pg_url, pool_size=3, ...)` constructor reaches `_setup_engine` and its separate `create_engine` call with the same values. Monkeypatch `_validate_schema`, `_create_tables`, `_create_additive_indexes`, and epoch sync to no-op so the test stops at engine construction without a database.

- [ ] Wire `**postgres_engine_kwargs(url)` into `app.py`'s long-lived session engine and `execution/service.py`'s Landscape writer. Preserve the kwargs when Plan 11 later replaces the raw constructor with its create-tables-gated factory.
- [ ] Document the remaining short-lived context-managed Landscape opens as deliberately disposable engines rather than an undefined follow-up. Plan 03 doctor and Plan 04 startup must apply the helper while retaining their `connect_args`; Plan 03 disposes every one-shot engine in `finally`; Plan 05 and Plan 11 already consume/preserve the helper.
- [ ] Run:

  ```bash
  uv run pytest tests/unit/core/landscape/test_database_compatibility_guards.py tests/unit/web/ -q
  ```

  Expected: PASS.

- [ ] Commit:

  ```bash
  git add src/elspeth/core/landscape/database.py src/elspeth/web/app.py src/elspeth/web/execution/service.py tests/unit/core/landscape/test_database_compatibility_guards.py
  git commit -m "feat: size PostgreSQL connection pools explicitly"
  ```

---

### Task 6: Prove classification, initialization, and lock behavior on PostgreSQL

**Files:**

- Create: `tests/testcontainer/web/test_schema_probe_postgres.py`

- [ ] Keep only the `PostgresContainer("postgres:16-alpine", driver="psycopg")` container module-scoped. Create and forcibly drop a unique PostgreSQL database for every test using an internally generated `elspeth_schema_<uuid.hex>` identifier and an AUTOCOMMIT admin connection. Assert the generated identifier matches `^[a-z0-9_]+$` before quoting it; no user input reaches identifier SQL. Each yielded engine therefore begins with zero tables and has its own advisory-lock namespace. Dispose the test engine before `DROP DATABASE ... WITH (FORCE)`.
- [ ] Mark the module with `pytest.mark.testcontainer` only. Keep it under `tests/testcontainer/`, outside `tests/integration/`, so the integration conftest cannot auto-add the `integration` marker and the existing hosted integration selector/coverage remains unchanged. Plan 12 is the explicit Docker-capable owner of this new file.
- [ ] Add fresh-create tests for both schemas: `MISSING → init → CURRENT`. The Landscape case must execute a raw SQL `INSERT` supplying every other required `runs` value but omitting `seeded_from_cache`, then read back `False`. Do not use SQLAlchemy Core's model insert for this assertion because its Python-side `default=False` could mask a broken server default.
- [ ] Add parameterized Landscape state tests on real reflection. Every mutation case receives its own fresh per-test database so the first `STALE` signal cannot mask later comparator defects:

  - drop `auth_events`, `run_attributions`, and `ix_tokens_run_id` independently → `PARTIAL`; init restores each and reaches `CURRENT`;
  - drop a core table → `STALE`; init raises and does not recreate it;
  - create an unrelated table in an otherwise-empty target → `STALE`; init creates no ELSPETH tables;
  - alter a column type/nullability/default, recreate a same-named index with wrong columns/predicate, remove/change a CHECK, unique, and FK → `STALE`; init performs no repair.

- [ ] Add the corresponding PostgreSQL session drift cases for foreign, partial, and wrong-shaped tables; all must be `STALE` and non-mutating.
- [ ] Parameterize the concurrency proof over session and Landscape initialization. Use a `Barrier` to start both workers, an `Event` set inside the first patched `create_all` after the advisory lock is held, and a second event to release it. Before releasing, poll `pg_locks` with a bounded deadline until the exact classid/database/target has one granted and one waiting row. Put `release_holder.set()` and bounded worker joins in `finally` so a failed assertion cannot strand the suite. Assert both workers finish, no exceptions escape, exactly one `create_all` runs, and the final state is `CURRENT`.
- [ ] Prove release after success, a SQLSTATE `22012` body failure, and a custom `BaseException`. Keep a `NullPool` observer connection open or compare `pg_backend_pid()` values so the verifier cannot be the reentrant lock-owning backend. `pg_try_advisory_lock` must return true, then the observer releases its test lock.
- [ ] Hold the lock from a distinct backend inside `try/finally`, monkeypatch the timeout to 250 ms, and assert `SchemaInitBusyError` with an `OperationalError` cause carrying SQLSTATE `55P03`; the body is never called and the public message contains no URL, credentials, SQL, or raw driver text. The `finally` branch always releases the holder lock.
- [ ] Construct PostgreSQL engines with `pool_size=1, max_overflow=0` and run both initializers under a bounded test timeout. Each must reach `CURRENT`, proving the same-connection design rather than merely asserting it with fakes.
- [ ] Require Docker and execute the file:

  ```bash
  docker info
  uv run pytest tests/testcontainer/web/test_schema_probe_postgres.py -m testcontainer -q
  ```

  Expected: Docker is available; every test executes and passes; zero tests skip. If Docker is unavailable, stop as `BLOCKED` and do not close the implementation issue.

- [ ] Commit:

  ```bash
  git add tests/testcontainer/web/test_schema_probe_postgres.py
  git commit -m "test: verify PostgreSQL schema state and advisory locking"
  ```

---

### Task 7: Run the Plan 02 handoff gates

**Files:** Verify all files touched by Tasks 1-6. No new implementation files are expected.

- [ ] Re-run the focused and subsystem tests:

  ```bash
  uv lock --check
  uv run pytest tests/unit/core/test_schema_shape.py tests/unit/core/landscape/test_database_schema_probe.py tests/unit/core/landscape/test_database_compatibility_guards.py tests/unit/core/landscape/test_schema_epoch_and_required_columns.py tests/unit/web/sessions/test_schema.py tests/unit/web/test_schema_probe.py -q
  uv run pytest tests/unit/web/ tests/unit/core/landscape/ -q
  docker info
  uv run pytest tests/testcontainer/web/test_schema_probe_postgres.py -m testcontainer -q
  ```

  Expected: all commands exit 0 and the PostgreSQL file reports zero skips.

- [ ] Run the repository's static gates exactly:

  ```bash
  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  git diff --check
  ```

  Expected: every command exits 0. Do not narrow mypy to one file; the CI configuration is strict for both source trees.

- [ ] Because this slice handles external URLs and database state, use the `wardline-gate` skill and run the mandatory project gate:

  ```bash
  wardline scan . --fail-on ERROR
  ```

  Expected: exit 0. On a finding, run the scan → explain → boundary fix → rescan loop; do not add a waiver or signed allowlist entry as part of this plan.

- [ ] Record these downstream handoffs in the Plan 02 closeout comment; they gate the named consumer plans, not closure of the already-verified Plan 02 implementation issue:

  - Plan 03 consumes `schema_probe.require_distinct_postgres_targets` and stops before engine construction or any `init_*` call; for same host/port/database, absent or ambiguous search-path options fail closed rather than assuming `public`. It catches `DatabaseTargetConflictError`, `SchemaInitBusyError`, and `SchemaLockCleanupError` separately and disposes one-shot engines in `finally`;
  - Plan 04 consumes the same `schema_probe.require_distinct_postgres_targets` check before startup probes rather than implementing a second target parser;
  - Plan 11 pins additive-table gaps as doctor-owned `PARTIAL`, core gaps as `STALE`, and request paths as non-mutating;
  - Plan 12 runs `docker info` followed by `uv run pytest tests/testcontainer/web/test_schema_probe_postgres.py -m testcontainer -q`, requires zero skips, and records that proof on the integrated candidate. The ordinary unit lane excludes `testcontainer`, and the existing hosted integration lane remains unchanged, so absence of this Plan 12 evidence is a runtime `NO-GO`.

- [ ] Final commit is needed only if the gates required a scoped correction. Do not create an empty verification commit.
