# AWS ECS Landscape Write Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** No web request path can create or mutate Landscape schema in `aws-ecs` mode: every web-layer writer receives one fail-closed `create_tables` policy derived from `WebSettings.deployment_target`, and an import-aware AST regression guard seals every direct web-layer `LandscapeDB` construction form.

**Architecture:** A new `src/elspeth/web/landscape_access.py` owns the writer policy exactly once: exact `"aws-ecs"` denies schema creation, exact `"default"` preserves local create-if-missing behavior, and any unknown future vocabulary raises before a database is opened. The two settings-holding writer sites (`execution/service.py`, `composer/tutorial_service.py`) call `open_landscape_db(settings)`; the four `auth/audit.py` writer sites carry a **required**, non-defaulted `create_tables` field computed by `from_settings` because the frozen recorder retains URL/passphrase rather than settings. An import-aware AST guard resolves direct, aliased, and module-qualified `LandscapeDB` imports, bans raw constructors, `in_memory()`, and stored `from_url` aliases, rejects literal `create_tables=True`, and requires an explicit policy on every direct `from_url` call. The existing read-only loaders continue to pass `create_tables=False, read_only=True` directly. Unit seams plus a DDL-denied PostgreSQL test prove the real policy; the AST guard is a regression seal over direct web-layer syntax, not a claim that arbitrary external wrappers are statically impossible.

**Why this is a subplan, not a follow-up:** the spec's "In ECS mode, web startup validates existing schema and does not create or mutate it" is falsified by any later request path that can emit DDL. `LandscapeDB.__init__` unconditionally runs `_create_tables()` + `_create_additive_indexes()` (`core/landscape/database.py:624-626`); `from_url(..., create_tables=True)` (the default) runs `_create_additive_indexes()` (`:1385-1389`), which issues `CREATE INDEX` for any `_ADDITIVE_INDEX_NAMES` member missing under version skew. On a CURRENT schema both are no-ops — the residual is version skew, a `get_landscape_url()` SQLite fallback, or mid-life drift, each of which today can emit real DDL from a pipeline run, tutorial projection, or auth-audit event and would hard-fail under the least-privilege DDL-denied **runtime database role** used by the web task.

**Tech Stack:** Python 3.13, SQLAlchemy, pytest, `ast` (stdlib) for the guard.

**Depends on:** `…-01-deployment-contract.md` (`WebSettings.deployment_target`, `DEPLOYMENT_TARGET_AWS_ECS`); `…-02-postgres-schema-support.md` Task 4 (`postgres_engine_kwargs`) **and Task 5** (`**engine_kwargs` support on `LandscapeDB.from_url` plus the `execution/service.py` wiring this plan later subsumes); `…-04-validate-only-startup.md` (its orphan-reconciliation work makes the Plan-04-owned `app.py` writer pass `create_tables` explicitly). Start only when all three Filigree blockers are closed. Wave 3.

**Global Constraints (verbatim from spec):** "In ECS mode, web startup validates existing schema and does not create or mutate it. Schema initialization is explicit and operator-controlled via the doctor command." Local/default mode keeps create-if-missing behavior — this plan changes **aws-ecs** behavior only.

**Auth failure posture and non-goal:** all four `AuthAuditRecorder` methods are reachable beyond local login. `record_auth_failure` is used by protected-route middleware for every configured provider and by `/api/auth/me`; OIDC/Entra and the Cognito-shaped OIDC production path therefore share the fail-closed response posture. A Landscape audit failure propagates instead of allowing the request to return its ordinary success/401/503 response. This is **response fail-closed**, not cross-database atomicity: local registration and email verification can already commit `auth.db` state before token-audit persistence, and login writes success/token events in separate Landscape transactions. Plan 11 must not claim those state changes roll back. That pre-existing integrity defect is tracked separately as `elspeth-57c4e276a4`; it is not hidden inside this schema-write gate.

### Task 1: `landscape_access.py` factory module

**Files:**
- Create: `src/elspeth/web/landscape_access.py`
- Test: `tests/unit/web/test_landscape_access.py`

**Interfaces:**
- Consumes: `LandscapeDB.from_url` (`core/landscape/database.py:1295`), `DEPLOYMENT_TARGET_AWS_ECS` (plan 01), `postgres_engine_kwargs` (plan 02, `web/schema_probe.py`), `WebSettings.get_landscape_url()`/`.landscape_passphrase`/`.deployment_target`.
- Produces: `landscape_create_tables_allowed(settings: WebSettings) -> bool`; `open_landscape_db(settings: WebSettings) -> LandscapeDB`.

```python
from __future__ import annotations

from typing import TYPE_CHECKING

from elspeth.core.landscape.database import LandscapeDB
from elspeth.web.deployment_contract import DEPLOYMENT_TARGET_AWS_ECS
from elspeth.web.schema_probe import postgres_engine_kwargs

if TYPE_CHECKING:
    from elspeth.web.config import WebSettings


def landscape_create_tables_allowed(settings: WebSettings) -> bool:
    """Whether this deployment may lazily create/complete landscape schema.

    aws-ecs is validate-only: `elspeth doctor aws-ecs --init-schema` is the
    sole schema-mutation path. Local/default keeps create-if-missing.
    """
    if settings.deployment_target == DEPLOYMENT_TARGET_AWS_ECS:
        return False
    if settings.deployment_target == "default":
        return True
    raise ValueError("unsupported deployment_target for Landscape schema policy")


def open_landscape_db(settings: WebSettings) -> LandscapeDB:
    """The one web-layer way to open a WRITER LandscapeDB.

    Read-only loaders keep calling from_url(create_tables=False,
    read_only=True) directly; the guard test pins both patterns.
    """
    create_tables = landscape_create_tables_allowed(settings)
    url = settings.get_landscape_url()
    return LandscapeDB.from_url(
        url,
        passphrase=settings.landscape_passphrase,
        create_tables=create_tables,
        **postgres_engine_kwargs(url),
    )
```

- [ ] Write failing tests in `tests/unit/web/test_landscape_access.py`. Build settings with a `_settings(deployment_target)` helper (a `SimpleNamespace` with `deployment_target`, `landscape_passphrase=None`, and `get_landscape_url=lambda: url` is sufficient — the factory duck-types). Monkeypatch the class attribute so the fake sees every route in: `monkeypatch.setattr("elspeth.web.landscape_access.LandscapeDB", _FakeLandscapeDB)` where `_FakeLandscapeDB.from_url` is a `classmethod` recording `(url, kwargs)` and returning a sentinel. Tests: `test_aws_ecs_disables_create_tables` (`deployment_target=DEPLOYMENT_TARGET_AWS_ECS` → recorded `create_tables is False`); `test_local_default_keeps_create_tables` (`"default"` → `True` — **vocabulary caution:** `deployment_target` is `Literal["default", "aws-ecs"]` per plan 01; `"local"` is `auth_provider` vocabulary and `WebSettings(deployment_target="local")` raises `ValidationError` — do not cross the two fields); `test_unknown_deployment_target_fails_before_url_or_db_open` (a future/invalid literal raises `ValueError`, and neither `get_landscape_url` nor the fake is called); `test_forwards_url_and_passphrase` (sqlite URL + passphrase sentinel appear verbatim); `test_postgres_url_gets_pool_kwargs` (`postgresql+psycopg://u@h/db` → recorded kwargs ⊇ `AWS_ECS_POOL_KWARGS`); `test_sqlite_url_gets_no_pool_kwargs` (`sqlite:///x.db` → no `pool_size` key). Run `uv run pytest tests/unit/web/test_landscape_access.py -v`; expect `ModuleNotFoundError: No module named 'elspeth.web.landscape_access'`.
- [ ] Implement the module per the code block above. Run again; expect all PASS.
- [ ] `git add src/elspeth/web/landscape_access.py tests/unit/web/test_landscape_access.py && git commit -m "feat(web): add create_tables-gated landscape DB factory for aws-ecs"`

### Task 2: Migrate the two settings-holding writer sites

**Files:**
- Modify: `src/elspeth/web/execution/service.py:1100` (bare `LandscapeDB(...)` inside `_run_pipeline`, `:1016`)
- Modify: `src/elspeth/web/composer/tutorial_service.py:204` (`from_url` inside `_project_live_tutorial_output`, `:195`)
- Test: `tests/unit/web/execution/test_service.py` (all 37 current `service.LandscapeDB` patch seams, including `TestB3Construction`)
- Test: `tests/integration/pipeline/test_composer_runtime_agreement.py` (the two current `service.LandscapeDB` patch seams)
- Test: `tests/unit/web/composer/test_tutorial_service.py`

**Interfaces:** Consumes `open_landscape_db` (Task 1). Behavioral delta at `service.py:1100`, deliberate: bare-ctor → `from_url` means `_validate_schema()` runs and then creation is **gated** instead of unconditional; in local mode (`create_tables=True`) `from_url` validates-then-creates exactly like the ctor did (`database.py:1385-1389`), so local behavior is unchanged.

- [ ] Rewrite `TestB3Construction::test_landscape_db_constructed_from_settings` FIRST: replace its patch target with `elspeth.web.execution.service.open_landscape_db` and replace the constructor assertion with `mock_open_landscape.assert_called_once_with(service._settings)`; keep the payload-store assertion. Run `uv run pytest tests/unit/web/execution/test_service.py::TestB3Construction -x`; expect FAIL (`AttributeError: ... has no attribute 'open_landscape_db'`).
- [ ] Migrate the **entire current mock seam**, not only `TestB3Construction`: replace all 37 exact `@patch("elspeth.web.execution.service.LandscapeDB")` targets in `tests/unit/web/execution/test_service.py` and both exact targets in `tests/integration/pipeline/test_composer_runtime_agreement.py` with `open_landscape_db`. Preserve existing `return_value`, `side_effect`, and `assert_not_called` behavior; rename mock parameters only where clarity requires it. After the sweep, `rg -n 'elspeth\.web\.execution\.service\.LandscapeDB' tests/unit/web/execution/test_service.py tests/integration/pipeline/test_composer_runtime_agreement.py` must return no matches. This count is a current-tree precondition, not a magic future constant: if upstream dependency work changes it, migrate every live match and record the new census.
- [ ] In `service.py`: add `from elspeth.web.landscape_access import open_landscape_db`, replace the constructor block with `landscape_db = open_landscape_db(self._settings)`, and move the `LandscapeDB` import under `TYPE_CHECKING` because it remains a type annotation at `_run_pipeline`/accounting boundaries but is no longer a runtime construction seam. Keep Plan 02 Task 5's `postgres_engine_kwargs` import removal in this sweep if the factory is now its only consumer. Run the complete unit and affected integration files; the migrated mocks must prevent any real/default database open.
- [ ] Write `test_projection_opens_landscape_via_gated_factory` in `test_tutorial_service.py` using a **file-backed** SQLite Landscape under `tmp_path`; do not reuse `make_landscape_db()` because it returns `sqlite:///:memory:` and reopening that URL creates a different empty database. Seed the run/source row required by `_project_live_tutorial_output`; stub `_rows_from_artifacts` to a bounded known result because artifact decoding is outside this seam test (or seed a complete valid audited artifact). Build real `WebSettings` with `deployment_target="default"` and the file-backed URL. Spy on `open_landscape_db`, delegate to the real factory, call the projection, assert exactly one call with the same settings object, and assert the intended run-row projection update persisted. Run first before importing the production seam and expect `AttributeError`; after the import/replacement expect PASS.
- [ ] In `tutorial_service.py`: add the import, replace the `LandscapeDB.from_url(settings.get_landscape_url(), passphrase=settings.landscape_passphrase)` context-manager head at `:204` with `open_landscape_db(settings)` (the `db.write_connection()` wrapping is unchanged). Run; expect PASS.
- [ ] Run `uv run pytest tests/unit/web/execution/test_service.py tests/unit/web/composer/test_tutorial_service.py tests/integration/pipeline/test_composer_runtime_agreement.py -q`; no regressions and no unmocked database opens.
- [ ] Stage only `src/elspeth/web/execution/service.py`, `src/elspeth/web/composer/tutorial_service.py`, `tests/unit/web/execution/test_service.py`, `tests/unit/web/composer/test_tutorial_service.py`, and `tests/integration/pipeline/test_composer_runtime_agreement.py`; verify `git diff --cached --name-only` is exactly that allowlist before `git commit -m "fix(web): gate per-run and tutorial landscape writers behind aws-ecs create_tables policy"`. Never use `git add -A` in this shared repository.

### Task 3: Gate the four auth-audit writer sites

**Files:**
- Modify: `src/elspeth/web/auth/audit.py` (`AuthAuditRecorder` dataclass `:153-164`; `from_url` calls at `:174,197,226,246`)
- Test: Create `tests/unit/web/auth/test_audit.py`
- Test: Modify `tests/unit/web/auth/test_middleware.py`, `tests/unit/web/auth/test_routes.py`

**Interfaces:** Produces required `AuthAuditRecorder.create_tables: bool` with **no default**; `from_settings` computes it via `landscape_create_tables_allowed(settings)`. Production currently constructs the recorder only through `from_settings`, so a permissive default would create a new future bypass rather than preserve a required compatibility surface. The recorder holds URL/passphrase rather than settings, so the field—not the factory—carries the policy. Route its four public write methods through one private context-managed opener that calls `LandscapeDB.from_url(..., create_tables=self.create_tables)`. On `SchemaCompatibilityError`, `LandscapeRecordError`, `SQLAlchemyError`, or `OSError`, log only the fixed event name, operation enum, and exception class, then re-raise; never log URL, SQL, exception text, request metadata, or credentials. No pool kwargs here — these remain deliberately short-lived per-event engines; Plan 12 records connection-churn capacity as an operational constraint rather than pretending these are bounded by the long-lived pool.

**Failure posture (decided):** the audit writers remain **response fail-closed**. A Landscape outage or post-boot incompatible schema failure propagates after emitting only the redacted class-only operator signal; no audit method converts it into success. This affects local login/register/verify/refresh **and** protected-route authentication/profile failures for OIDC, Entra, and Cognito-shaped OIDC. The asymmetry with Plans 04/05's recoverable orphan cleanup is intentional: cleanup protects availability, while auth audit is integrity evidence. Do not overclaim transactionality, however: this plan does not roll back prior `auth.db` transitions or make the two login audit events atomic; `elspeth-57c4e276a4` owns that larger defect. With `create_tables=False`, missing additive tables fail without repair; a missing non-integrity additive index may remain tolerated by direct `LandscapeDB` validation after Plan 02, but it must never be recreated by a request. Startup's stricter schema probe should already block that PARTIAL state before traffic.

- [ ] Write failing configuration tests in `tests/unit/web/auth/test_audit.py`: AWS settings map to `False`; default settings map to `True`; an invalid/future deployment target propagates the factory's `ValueError`; and direct `AuthAuditRecorder(landscape_url=..., landscape_passphrase=...)` without `create_tables` raises `TypeError`. This pins the settings-to-runtime contract and prevents a permissive constructor default.
- [ ] Add a parameterized writer test covering `record_login_success`, `record_token_issued`, `record_auth_failure`, and `record_login_failure`. Use a kwargs-recording context-manager fake for `LandscapeDB.from_url`, a mocked `RecorderFactory`, method-specific valid arguments, and a syntactically valid bounded JWT for token issuance. Construct the recorder with `create_tables=False`; every case must capture exactly one open with `create_tables is False`. A single-method test is insufficient because the other three methods could otherwise hard-code `True` and still satisfy the AST keyword-presence rule.
- [ ] Add failure/observability tests parameterized over opener-side `SchemaCompatibilityError("drift")` and repository-side `LandscapeRecordError("sentinel")`: each propagates; the captured structured event contains only `auth_audit_write_failed`, the closed operation name, and the exact exception class; URL, exception message, request metadata, token, credentials, and SQL are absent. Add raising-recorder tests in `test_middleware.py` and `test_routes.py` proving that an OIDC/Entra protected-request auth failure and `/api/auth/me` profile failure use the recorder and preserve the selected response-fail-closed propagation. These tests document provider blast radius; they do not claim cross-store rollback.
- [ ] Implement the required field, `from_settings` mapping, private opener, class-only logging, and shared use by all four public methods. Run `uv run pytest tests/unit/web/auth/ tests/unit/web/test_app.py -q`; no regressions (`app.py`'s `from_settings` call site needs no edit).
- [ ] Stage only `src/elspeth/web/auth/audit.py`, `tests/unit/web/auth/test_audit.py`, `tests/unit/web/auth/test_middleware.py`, and `tests/unit/web/auth/test_routes.py`; verify the staged path allowlist, then commit `fix(web): gate auth-audit landscape writers behind aws-ecs create_tables policy`.

### Task 4: AST regression guard

**Files:**
- Create: `tests/unit/web/test_landscape_access_guard.py`

**Interfaces:** private test helpers parse one module, conservatively identify any direct callee rooted at the literal identifier/attribute `LandscapeDB`, resolve import aliases, and return stable `path:line:reason` offenders:

- every `from ... import LandscapeDB` and alias such as `as LDB`, regardless of whether it comes from `elspeth.core.landscape.database`, the public `elspeth.core.landscape` re-export, or the current `elspeth.web.sessions.routes._helpers` re-export;
- both `Import` and `ImportFrom` forms for the known provider-module set `{elspeth.core.landscape, elspeth.core.landscape.database, elspeth.web.sessions.routes._helpers}`, including module aliases, `from elspeth.core.landscape import database as ...`, and unaliased fully-qualified attribute chains ending in `.LandscapeDB`; conservatively treat any direct receiver whose identifier/attribute terminal is literally `LandscapeDB` as a candidate even when its import provenance is unfamiliar;
- direct constructor calls and `LandscapeDB.in_memory()` (both always create schema);
- `LandscapeDB.from_url` captured into a variable/attribute for a later indirect call (forbid the alias assignment rather than pretending to follow data flow);
- direct `from_url(...)` without `create_tables=`, and direct literal `create_tables=True`.

An explicit non-literal policy (`create_tables=create_tables`, `self.create_tables`, or the factory's computed local) and literal `False` pass. `landscape_access.py`, Plan 04's `app.py` helper, the recorder opener, and existing read-only loaders therefore require no filename allowlist. Keep this claim honest: the guard seals **direct web-layer syntax and import aliases**. It cannot prove that an arbitrary helper imported from outside `src/elspeth/web/` never wraps a schema-creating call; the factory tests, PostgreSQL proof, core API review, and mandatory code review remain necessary controls.

- [ ] Implement the collector helpers first and table-drive them with synthetic source strings parsed in memory. Negative fixtures must catch exact direct ctor; `from elspeth.core.landscape import LandscapeDB`; aliased and **unaliased** `import elspeth.core.landscape`; aliased and unaliased database-module imports; `from ...landscape import database as ...`; direct and module-aliased imports of current `_helpers`; unfamiliar-import literal `LandscapeDB`; `in_memory()`; captured `from_url` alias; missing keyword; and literal `create_tables=True`. Positive fixtures must accept each supported direct/aliased/qualified `from_url` shape with `False` or a non-literal policy. Assert stable line numbers and reason codes. These synthetic fixtures replace the unsafe instruction to edit production source temporarily; do not mutate `auth/audit.py`, use `git stash`, or require the whole worktree diff to be empty.
- [ ] Add the real-tree test over every `*.py` under `src/elspeth/web/`. Require no offenders after Tasks 2–3 and Plan 04, and pin the expected direct-call map rather than merely asserting non-zero: one policy call each in `landscape_access.py`, `auth/audit.py`, and Plan-04-owned `app.py`, plus one explicit read-only call each in `execution/accounting.py`, `execution/diagnostics.py`, `execution/discard_summary.py`, `execution/outputs.py`, and `sessions/routes/runs.py` — eight direct calls total under the repaired shared-opener design. A changed map requires explicit review and guard update. This both catches a broken root/path and ensures the `_helpers` re-exported `runs.py` site cannot disappear from the census. The guard must name every offender, never stop on the first file.
- [ ] Run `uv run pytest tests/unit/web/test_landscape_access_guard.py -v`; then run `uv run pytest tests/unit/web/ -q`.
- [ ] Stage only `tests/unit/web/test_landscape_access_guard.py`, verify the staged allowlist, and commit `test(web): ban ungated LandscapeDB construction in the web layer`.

### Task 5: Prove request-time writer behavior with a DDL-denied PostgreSQL role

**Files:**
- Create: `tests/testcontainer/web/test_landscape_write_gate_postgres.py`

**Interfaces:** consumes Plan 02's real PostgreSQL classifier/initializer and Plan 04's safe DDL-denied-role fixture pattern; produces no runtime API. Mark this file `pytest.mark.testcontainer` only. Plan 12 owns mandatory Docker execution with zero skips.

- [ ] Use module-scoped `PostgresContainer("postgres:16-alpine", driver="psycopg")` and per-test unique database/role identifiers derived from lowercase UUID hex, regex-validated and quoted with `psycopg.sql.Identifier`. Initialize the Landscape schema only through the owner path. Create a runtime login with CONNECT/schema USAGE and the exact SELECT/INSERT/UPDATE/sequence privileges needed by the writer proof, revoke CREATE from PUBLIC and the runtime role, and prove a harmless rolled-back `CREATE TABLE` fails with SQLSTATE `42501`. Dispose every engine before forced cleanup in `finally`; never interpolate identifiers, URLs, SQL, or credentials into logs/assertion messages.
- [ ] `test_aws_ecs_factory_and_auth_writer_succeed_without_ddl`: snapshot sorted canonical table/index/constraint tuples, construct a complete Plan-01-valid `WebSettings` (`deployment_target="aws-ecs"`, explicit distinct credential-bearing PostgreSQL session/runtime Landscape URLs supplied only by the fixture, `landscape_passphrase=None` because passphrases are SQLite/SQLCipher-only, explicit state paths, production-shaped `secret_key` and `shareable_link_signing_key`, and single-worker settings), open through `open_landscape_db`, then construct the production recorder through `AuthAuditRecorder.from_settings(settings)` and persist one bounded auth-audit event through its real repository path. Require the DML row exists while the catalog tuples remain identical. This proves the settings-to-factory-to-recorder path can operate with no DDL privilege.
- [ ] `test_request_open_does_not_repair_missing_additive_index`: as owner, remove only Plan 02's explicitly tolerated non-integrity additive index after the initial snapshot; record that the stricter startup probe would classify this `PARTIAL`; then open through the AWS factory under the runtime role, persist bounded DML, and require the index remains absent. Plan 02's direct `LandscapeDB.from_url` contract tolerates this performance-only gap, so this test has one exact expected branch: request-time open succeeds but never issues `CREATE INDEX` or repairs the catalog.
- [ ] `test_request_open_rejects_missing_additive_table_without_repair`: remove one additive table as owner, call the AWS factory/recorder under the runtime role, require `SchemaCompatibilityError`, and require the missing table remains absent. No request path may turn Plan 02 `PARTIAL` into an implicit repair.
- [ ] Run `docker info` and `uv run pytest tests/testcontainer/web/test_landscape_write_gate_postgres.py -m testcontainer -q`; Docker unavailable, any skip, or any catalog mutation is BLOCKED/NO-GO.
- [ ] Stage only the new test file, verify the staged allowlist, and commit `test(web): prove request landscape writers need no ddl privilege`.

### Task 6: Handoff verification and downstream evidence

- [ ] Run the complete scoped regression set:

  ```bash
  uv run pytest tests/unit/web/test_landscape_access.py tests/unit/web/test_landscape_access_guard.py tests/unit/web/auth/ tests/unit/web/execution/test_service.py tests/unit/web/composer/test_tutorial_service.py tests/integration/pipeline/test_composer_runtime_agreement.py -q
  docker info
  uv run pytest tests/testcontainer/web/test_landscape_write_gate_postgres.py -m testcontainer -q
  uv run pytest tests/unit/web/ -q
  ```

  Every test executes and passes; Docker/testcontainer absence is BLOCKED rather than skipped.
- [ ] Run repository static and manifest gates exactly:

  ```bash
  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules manifest.symbol_inventory,manifest.test_to_source_mapping --root .
  git diff --check
  ```

  If the manifest gate names the new production/test files, update only its canonical declarative inventory in this task, add that exact path to the staged allowlist, and rerun; do not add a bespoke checker. Signed whole-repository trust-tier judgement remains Plan 12's integration-owner gate.
- [ ] Before every task commit, stage only that task's explicit files, verify the staged path set, then run `git diff --cached --name-only -z | xargs -0 uv run pre-commit run --files`. Do not use `--all-files`, `git add -A`, `git stash`, `--no-verify`, or hand-write signed allowlist metadata.
- [ ] Because this slice carries deployment configuration into database mutation policy and handles externally triggered auth/request paths, use the `wardline-gate` skill and run `wardline scan . --fail-on ERROR`. Exit 0 is required. On exit 1, explain every active taint and fix validation/rejection at the boundary; on exit 2, stop and surface the Wardline/tool error. Do not baseline or waive a finding to make the plan pass.
- [ ] Add a Filigree closeout comment to `elspeth-25286192ee` with the integrated commit, exact commands/exits, the 39-patch migration census (or recorded current-head replacement count), DDL-denied PostgreSQL evidence, AST synthetic/real-tree evidence, and the separate non-atomic-auth bug `elspeth-57c4e276a4`. Plan 12 must run this testcontainer file alongside the other load-bearing PostgreSQL files; no Plan 11 status transition occurs until all gates pass.

**Definition of Done:** exact `"aws-ecs"` settings fail closed to `create_tables=False`; unknown deployment-target vocabulary cannot silently enable DDL; every current writer receives that policy; all direct web-layer constructor/import forms are sealed; all 39 current service mock seams (or the recorded current-head census) are migrated; every auth writer and non-local failure path is covered; real PostgreSQL request-time DML succeeds with CREATE denied and never repairs schema; scoped/full tests, manifest/static/pre-commit, and Wardline gates pass; Plan 11's issue remains dependency-controlled until implementation evidence is attached.
