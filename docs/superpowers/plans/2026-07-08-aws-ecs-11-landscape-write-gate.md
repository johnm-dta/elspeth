# AWS ECS Landscape Write Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** No web request path can create or mutate landscape schema in `aws-ecs` mode: every web-layer `LandscapeDB` construction routes through one `create_tables`-gated factory (or passes `create_tables` explicitly), enforced by an AST regression guard.

**Architecture:** A new `src/elspeth/web/landscape_access.py` owns the policy (`create_tables = deployment_target != aws-ecs`) exactly once. The two settings-holding writer sites (`execution/service.py:1100`, `composer/tutorial_service.py:204`) call its `open_landscape_db(settings)`; the four `auth/audit.py` writer sites thread a `create_tables` field computed at `from_settings` time (the frozen recorder holds url+passphrase, not settings). An AST guard test bans bare `LandscapeDB(...)` construction and defaulted `from_url(...)` calls under `src/elspeth/web/` forever. The read-only loaders (`accounting`/`outputs`/`diagnostics`/`discard_summary`/`sessions/routes/runs`) already pass `create_tables=False, read_only=True` explicitly and need no change — the guard simply pins them.

**Why this is a subplan, not a follow-up:** the spec's "In ECS mode, web startup validates existing schema and does not create or mutate it" is falsified by any request path that can emit DDL. `LandscapeDB.__init__` unconditionally runs `_create_tables()` + `_create_additive_indexes()` (`core/landscape/database.py:624-626`); `from_url(..., create_tables=True)` (the default) runs `_create_additive_indexes()` (`:1385-1389`), which issues `CREATE INDEX` for any `_ADDITIVE_INDEX_NAMES` member missing under version skew. On a CURRENT schema both are no-ops — the residual is version skew, a `get_landscape_url()` SQLite fallback, or a mid-life drift, each of which today emits real DDL from a pipeline run, a tutorial projection, or a **login event**, and would hard-fail under a least-privilege DDL-denied task role.

**Tech Stack:** Python 3.13, SQLAlchemy, pytest, `ast` (stdlib) for the guard.

**Depends on:** `…-01-deployment-contract.md` (`WebSettings.deployment_target`, `DEPLOYMENT_TARGET_AWS_ECS`); `…-02-postgres-schema-support.md` (`postgres_engine_kwargs`, and Task 4's `**engine_kwargs` support on `from_url` — this plan **preserves** the `postgres_engine_kwargs` wiring plan 02 Task 4 adds at `execution/service.py:1100`, then subsumes the call site); `…-04-validate-only-startup.md` (its orphan-reconciliation work makes `app.py:176`'s `from_url` pass `create_tables` explicitly — without it, Task 4's guard here fails on a site plan 04 owns). Wave 3.

**Global Constraints (verbatim from spec):** "In ECS mode, web startup validates existing schema and does not create or mutate it. Schema initialization is explicit and operator-controlled via the doctor command." Local/default mode keeps create-if-missing behavior — this plan changes **aws-ecs** behavior only.

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
    return settings.deployment_target != DEPLOYMENT_TARGET_AWS_ECS


def open_landscape_db(settings: WebSettings) -> LandscapeDB:
    """The one web-layer way to open a WRITER LandscapeDB.

    Read-only loaders keep calling from_url(create_tables=False,
    read_only=True) directly; the guard test pins both patterns.
    """
    url = settings.get_landscape_url()
    return LandscapeDB.from_url(
        url,
        passphrase=settings.landscape_passphrase,
        create_tables=landscape_create_tables_allowed(settings),
        **postgres_engine_kwargs(url),
    )
```

- [ ] Write failing tests in `tests/unit/web/test_landscape_access.py`. Build settings with a `_settings(deployment_target)` helper (a `SimpleNamespace` with `deployment_target`, `landscape_passphrase=None`, and `get_landscape_url=lambda: url` is sufficient — the factory duck-types). Monkeypatch the class attribute so the fake sees every route in: `monkeypatch.setattr("elspeth.web.landscape_access.LandscapeDB", _FakeLandscapeDB)` where `_FakeLandscapeDB.from_url` is a `classmethod` recording `(url, kwargs)` and returning a sentinel. Tests: `test_aws_ecs_disables_create_tables` (`deployment_target=DEPLOYMENT_TARGET_AWS_ECS` → recorded `create_tables is False`); `test_local_default_keeps_create_tables` (`"default"` → `True` — **vocabulary caution:** `deployment_target` is `Literal["default", "aws-ecs"]` per plan 01; `"local"` is `auth_provider` vocabulary and `WebSettings(deployment_target="local")` raises `ValidationError` — do not cross the two fields); `test_forwards_url_and_passphrase` (sqlite URL + passphrase sentinel appear verbatim); `test_postgres_url_gets_pool_kwargs` (`postgresql+psycopg://u@h/db` → recorded kwargs ⊇ `AWS_ECS_POOL_KWARGS`); `test_sqlite_url_gets_no_pool_kwargs` (`sqlite:///x.db` → no `pool_size` key). Run `pytest tests/unit/web/test_landscape_access.py -v`; expect `ModuleNotFoundError: No module named 'elspeth.web.landscape_access'`.
- [ ] Implement the module per the code block above. Run again; expect all PASS.
- [ ] `git add src/elspeth/web/landscape_access.py tests/unit/web/test_landscape_access.py && git commit -m "feat(web): add create_tables-gated landscape DB factory for aws-ecs"`

### Task 2: Migrate the two settings-holding writer sites

**Files:**
- Modify: `src/elspeth/web/execution/service.py:1100` (bare `LandscapeDB(...)` inside `_run_pipeline`, `:1016`)
- Modify: `src/elspeth/web/composer/tutorial_service.py:204` (`from_url` inside `_project_live_tutorial_output`, `:195`)
- Test: `tests/unit/web/execution/test_service.py` (existing `TestB3Construction`, `:1229`), `tests/unit/web/composer/test_tutorial_service.py`

**Interfaces:** Consumes `open_landscape_db` (Task 1). Behavioral delta at `service.py:1100`, deliberate: bare-ctor → `from_url` means `_validate_schema()` runs and then creation is **gated** instead of unconditional; in local mode (`create_tables=True`) `from_url` validates-then-creates exactly like the ctor did (`database.py:1385-1389`), so local behavior is unchanged.

- [ ] Rewrite `TestB3Construction::test_landscape_db_constructed_from_settings` FIRST: replace `@patch("elspeth.web.execution.service.LandscapeDB")` with `@patch("elspeth.web.execution.service.open_landscape_db")` (keep the other patches) and replace the final assertion pair with `mock_open_landscape.assert_called_once_with(service._settings)` (the payload-store assertion stays). Run `pytest tests/unit/web/execution/test_service.py::TestB3Construction -x`; expect FAIL (`AttributeError: ... has no attribute 'open_landscape_db'` — the module doesn't import it yet).
- [ ] In `service.py`: add `from elspeth.web.landscape_access import open_landscape_db`, replace the `landscape_db = LandscapeDB(connection_string=..., passphrase=..., ...)` block with `landscape_db = open_landscape_db(self._settings)`, and delete the now-unused `LandscapeDB` import if nothing else in the module uses it (ruff F401 will tell you). Keep plan 02's `postgres_engine_kwargs` import removal in the same sweep if this was its only consumer — the factory now owns that wiring. Run the test; expect PASS.
- [ ] Write `test_projection_opens_landscape_via_gated_factory` in `test_tutorial_service.py`: monkeypatch `elspeth.web.composer.tutorial_service.open_landscape_db` with a spy that records the `settings` argument and delegates to the real factory against a seeded sqlite landscape DB — mirror the arrange block of `test_count_calls_for_run_counts_only_llm_calls` (`:156`), which already builds a landscape DB with a `runs_table` row for this exact projection path, and point the spy's settings at that DB's URL with `deployment_target="default"` (plan 01's non-ECS literal — not `"local"`, which is `auth_provider` vocabulary and fails `WebSettings` validation). Call `_project_live_tutorial_output(settings, run_id=..., landscape_run_id=<seeded>, session_id=...)`; assert the spy recorded exactly one call with `settings`. Run; expect FAIL (`AttributeError` on the module attribute).
- [ ] In `tutorial_service.py`: add the import, replace the `LandscapeDB.from_url(settings.get_landscape_url(), passphrase=settings.landscape_passphrase)` context-manager head at `:204` with `open_landscape_db(settings)` (the `db.write_connection()` wrapping is unchanged). Run; expect PASS.
- [ ] Run `pytest tests/unit/web/execution/test_service.py tests/unit/web/composer/test_tutorial_service.py -q`; no regressions.
- [ ] `git add -A && git commit -m "fix(web): gate per-run and tutorial landscape writers behind aws-ecs create_tables policy"`

### Task 3: Gate the four auth-audit writer sites

**Files:**
- Modify: `src/elspeth/web/auth/audit.py` (`AuthAuditRecorder` dataclass `:153-164`; `from_url` calls at `:174,197,226,246`)
- Test: Create `tests/unit/web/auth/test_audit.py`

**Interfaces:** Produces `AuthAuditRecorder.create_tables: bool = True` (new frozen-dataclass field, defaulted so existing direct constructions keep today's behavior); `from_settings` computes it via `landscape_create_tables_allowed(settings)`. The recorder holds url+passphrase rather than settings, so the field—not the factory—carries the policy; each of the four `LandscapeDB.from_url(self.landscape_url, passphrase=self.landscape_passphrase)` calls gains `create_tables=self.create_tables`. No pool kwargs here — these are short-lived per-event engines where pooling is meaningless; deliberate.

**Failure posture (decided, round 3):** the audit writers stay **fail-closed** — no try/except is added around these four `from_url` calls (their `routes.py:235,244` callers are equally unwrapped), so a landscape outage or post-boot schema drift fails the login request rather than skipping the audit record. This is deliberate and now explicit: a login that cannot be audited must not succeed — the audit trail is an integrity guarantee, unlike plans 04/05's orphan-cleanup paths, which degrade gracefully because they protect availability, not integrity. The asymmetry with 04/05 is a decision, not an oversight. Blast radius: aws-ecs + `auth_provider=local` only (`routes.py:227` 404s non-local providers; Cognito/OIDC — the recommended production path — never reaches these writers). What `create_tables=False` changes on this path: a MISSING or additive-index-gap landscape that today silently emitted DDL from a login now raises instead; STALE already raised. Converting the resulting raw 500 into a structured, operator-actionable error is deferred until the posture proves noisy in practice.

- [ ] Write failing tests in `tests/unit/web/auth/test_audit.py`: `test_from_settings_disables_create_tables_for_aws_ecs` (settings stub with `deployment_target=DEPLOYMENT_TARGET_AWS_ECS`, `get_landscape_url`, `landscape_passphrase=None` → `AuthAuditRecorder.from_settings(settings).create_tables is False`); `test_from_settings_keeps_create_tables_for_default` (settings stub with `deployment_target="default"` → `True`); `test_record_login_success_threads_create_tables` (monkeypatch `elspeth.web.auth.audit.LandscapeDB.from_url` with a kwargs-recording `@contextmanager` fake yielding `MagicMock()`, and `elspeth.web.auth.audit.RecorderFactory` with `MagicMock()`; construct `AuthAuditRecorder(landscape_url="sqlite:///x.db", landscape_passphrase=None, create_tables=False)`; call `record_login_success(MagicMock(), provider=<any valid AuthProviderType>, user_id="u", username="n")`; assert recorded kwargs include `create_tables=False`); `test_audit_write_failure_propagates` (monkeypatch the `from_url` fake to raise `SchemaCompatibilityError("drift")`; `record_login_success(...)` propagates it — pins the fail-closed posture above as decided behavior, not an accident a later "helpful" try/except may silently reverse). Run `pytest tests/unit/web/auth/test_audit.py -v`; expect FAIL (`TypeError: unexpected keyword argument 'create_tables'`).
- [ ] Add the field, the `from_settings` computation (import `landscape_create_tables_allowed` from `landscape_access`), and `create_tables=self.create_tables` at all four `from_url` sites. Run; expect PASS.
- [ ] Run `pytest tests/unit/web/auth/ tests/unit/web/test_app.py -q`; no regressions (`app.py:862`'s `from_settings` call site needs no edit — the classmethod signature is unchanged).
- [ ] `git add src/elspeth/web/auth/audit.py tests/unit/web/auth/test_audit.py && git commit -m "fix(web): gate auth-audit landscape writers behind aws-ecs create_tables policy"`

### Task 4: AST regression guard

**Files:**
- Create: `tests/unit/web/test_landscape_access_guard.py`

**Interfaces:** none produced — this is the structural seal. Two rules over every `*.py` under `src/elspeth/web/`: (1) no bare `LandscapeDB(...)` `ast.Call` whose `func` is the `Name` `LandscapeDB`; (2) every `LandscapeDB.from_url(...)` call carries an explicit `create_tables=` keyword. `landscape_access.py` itself passes rule 2 (it passes the kwarg it computes), so there is **no allowlist**. **Known boundary (round 3):** the guard is deliberately syntactic — an aliased import (`from ... import LandscapeDB as LDB`), a via-variable `from_url` call, or a wrapper constructed outside `web/` would evade both rules. That is the accepted trade: it is a regression seal against naive reintroduction (the round-2 failure class, and all eleven current web-layer sites match by name today), not a proof; the `open_landscape_db` factory seam and review remain the real control.

```python
import ast
from pathlib import Path

import elspeth.web

WEB_ROOT = Path(elspeth.web.__file__).parent


def _calls(tree: ast.AST):
    return (n for n in ast.walk(tree) if isinstance(n, ast.Call))


def test_no_bare_landscape_db_constructor_in_web_layer() -> None:
    offenders = [
        f"{path.relative_to(WEB_ROOT)}:{node.lineno}"
        for path in WEB_ROOT.rglob("*.py")
        for node in _calls(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node.func, ast.Name) and node.func.id == "LandscapeDB"
    ]
    assert not offenders, (
        "Bare LandscapeDB(...) unconditionally creates schema on construction "
        "(database.py:624-626) — use elspeth.web.landscape_access.open_landscape_db "
        f"or from_url(create_tables=...). Offenders: {offenders}"
    )


def test_every_web_from_url_passes_create_tables_explicitly() -> None:
    offenders = [
        f"{path.relative_to(WEB_ROOT)}:{node.lineno}"
        for path in WEB_ROOT.rglob("*.py")
        for node in _calls(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_url"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "LandscapeDB"
        and not any(kw.arg == "create_tables" for kw in node.keywords)
    ]
    assert not offenders, (
        "from_url defaults create_tables=True, which can emit DDL from a web "
        "request path in aws-ecs mode — pass create_tables explicitly (or use "
        f"open_landscape_db). Offenders: {offenders}"
    )
```

- [ ] Write the file exactly as above and run `pytest tests/unit/web/test_landscape_access_guard.py -v`; expect PASS — Tasks 2–3 (and plan 04's `app.py` work) already migrated every site, so this test lands green *as a seal*, not red-first. Sanity-check its teeth before trusting it (do NOT use `git stash` — a repo hook blocks it): temporarily delete the `create_tables=self.create_tables` kwarg from ONE `from_url` call in `auth/audit.py`, rerun — expect FAIL naming `auth/audit.py` with that line number — then restore the kwarg (`git diff` must be empty again) and rerun to green.
- [ ] Run `pytest tests/unit/web/ -q`; no regressions.
- [ ] `git add tests/unit/web/test_landscape_access_guard.py && git commit -m "test(web): ban ungated LandscapeDB construction in the web layer"`
