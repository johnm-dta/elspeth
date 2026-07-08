# AWS ECS Readiness Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add a dependency-aware, unauthenticated `GET /api/ready` endpoint with budgeted, redacted, cached checks, while proving `GET /api/health` and process startup both stay independent of Aurora/EFS availability.

**Architecture:** A new module `src/elspeth/web/readiness.py` exposes `readiness_report()`, an async orchestrator running five grouped dependency probes (session DB, landscape DB, three writable-directory probes) concurrently, each capped at 2s, the whole computation capped at 5s, plus a pure in-process auth-mode check — two deliberate deviations from a literal spec reading, below. `app.py` wires a single-flight `ReadinessCache` (2s TTL) around it at `GET /api/ready`, unauthenticated, alongside the untouched `GET /api/health` (`app.py:1157-1159`). The 5s ceiling is a backstop, not an SLA: five 2s-capped checks running concurrently make it unreachable barring a bug in `bounded()`; on that timeout the response discards any already-completed results for one synthetic `overall` check — accepted, not a defect.

**Tech Stack:** Python 3.13, `asyncio`, SQLAlchemy 2.x sync `Engine`, FastAPI/Starlette, pytest, `pytest-asyncio` (strict mode — every async test needs an explicit `@pytest.mark.asyncio`, per existing usage at `tests/unit/web/test_app.py:334-370`).

**Depends on:** `2026-07-08-aws-ecs-01-deployment-contract.md` and `2026-07-08-aws-ecs-02-postgres-schema-support.md` (for `SchemaState`, `probe_session_schema(engine)`, `probe_landscape_schema(engine)`, `postgres_engine_kwargs(url)` in `elspeth.web.schema_probe`) — must land first per wave ordering; not otherwise used here.

**Global Constraints (verbatim spec):** 2s per-check timeout; 5s whole-endpoint ceiling; 2s TTL result cache is the stampede control, not authentication — `/api/ready` stays unauthenticated. Checks: session DB connectivity, session schema validity, landscape DB connectivity, landscape schema validity, `data_dir`/payload-store/blob-dir writability (create/read/fsync-or-close/delete probe file, no path or secret leakage), configured auth-mode requirements. `/api/health` "must not depend on Aurora, EFS, or remote providers."

**Deviations:** (1) Session/landscape checks share one 2s connectivity+schema budget per DB instead of four independently-budgeted checks — recorded per panel review rather than argued away in prose only. (2) Readiness probes dispatch onto a dedicated 4-worker pool inside `readiness.py`, never the shared pool `run_sync_in_worker` (`async_workers.py:82`) uses elsewhere — a wedged Aurora/EFS connect on the shared pool starves session/execution/composer/auth routes app-wide (panel finding, HIGH); nothing can kill a wedged thread, only abandon it, so isolating the pool is the available mitigation.

**Residual limitation:** A hard, sustained wedge (e.g. a dropped-SYN security-group misconfiguration) still exhausts readiness's own 4-worker pool over repeated cache-refresh cycles, degrading `/api/ready` further without affecting the rest of the app. The ops doc (Task 2) tells operators that persistent `/api/ready` failures beyond a few polling cycles may need manual task recycling, since `/api/health` alone will not detect an internally wedged pool. A driver-level `connect_timeout` remains a follow-up.

### Task 1: `readiness.py` — checks, cache, orchestrator

**Files:**
- Create: `src/elspeth/web/readiness.py`
- Test: `tests/unit/web/test_readiness.py`

**Interfaces:**
- Consumes: `SchemaState`, `probe_session_schema(engine)`, `probe_landscape_schema(engine)`, `postgres_engine_kwargs(url)` (`elspeth.web.schema_probe`); `WebSettings.get_landscape_url()/get_payload_store_path()` (`config.py:646-657`); SQLAlchemy `create_engine`/`text`; `create_session_engine`/`initialize_session_schema` (mirrors `app.py:874-875`, test-fixture use only). Deliberately NOT `LandscapeDB.from_url()` for the live probe path — see Task 1 prose below.
- Produces (Task 2 consumes): `ReadinessCheck(name, ok, detail)`, `ReadinessReport(ready, checks)`, `ReadinessCache`, `async def readiness_report(settings, session_engine) -> ReadinessReport`. Not `elspeth.web.audit_readiness.service.ReadinessService` (per-run audit snapshot — unrelated, name collision only).

```python
from __future__ import annotations
import functools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from sqlalchemy import create_engine, text
from elspeth.web.schema_probe import SchemaState, probe_session_schema, probe_landscape_schema, postgres_engine_kwargs

_PROBE_TIMEOUT_SECONDS = 2.0
_TOTAL_TIMEOUT_SECONDS = 5.0
_READINESS_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="readiness-worker")

@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    ok: bool
    detail: str

@dataclass(frozen=True)
class ReadinessReport:
    ready: bool
    checks: tuple[ReadinessCheck, ...]

def _probe_directory_writable(name: str, directory: Path) -> ReadinessCheck:
    probe_path = directory / f".readiness-probe-{uuid4().hex}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with probe_path.open("wb") as fh:
            fh.write(b"ok"); fh.flush(); os.fsync(fh.fileno())
        if probe_path.read_bytes() != b"ok":
            return ReadinessCheck(name, False, f"{name} probe file content mismatch")
    except OSError as exc:
        return ReadinessCheck(name, False, f"{name} is not writable ({type(exc).__name__})")
    finally:
        probe_path.unlink(missing_ok=True)
    return ReadinessCheck(name, True, f"{name} is writable")

def _dir_check(name: str, directory: Path) -> tuple[ReadinessCheck, ...]:
    return (_probe_directory_writable(name, directory),)

def _check_session(engine: Engine) -> tuple[ReadinessCheck, ...]:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    state = probe_session_schema(engine)
    return (
        ReadinessCheck("session_db", True, "connected"),
        ReadinessCheck("session_schema", state is SchemaState.CURRENT, f"schema state: {state.name}"),
    )

def _check_landscape(settings: WebSettings) -> tuple[ReadinessCheck, ...]:
    engine = create_engine(settings.get_landscape_url(), **postgres_engine_kwargs(settings.get_landscape_url()))
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        state = probe_landscape_schema(engine)
    finally:
        engine.dispose()
    return (
        ReadinessCheck("landscape_db", True, "connected"),
        ReadinessCheck("landscape_schema", state is SchemaState.CURRENT, f"schema state: {state.name}"),
    )

class ReadinessCache:
    def __init__(self, ttl_seconds: float = 2.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl, self._clock, self._lock = ttl_seconds, clock, asyncio.Lock()
        self._report: ReadinessReport | None = None
        self._computed_at = float("-inf")

    async def get(self, compute: Callable[[], Awaitable[ReadinessReport]]) -> ReadinessReport:
        async with self._lock:
            if self._report is not None and self._clock() - self._computed_at < self._ttl:
                return self._report
            self._report = await compute()
            self._computed_at = self._clock()
            return self._report

def _drain(future: asyncio.Future[object]) -> None:
    if not future.cancelled():
        future.exception()  # retrieve so an abandoned wedged probe never logs "exception never retrieved"

async def bounded(label: str, fn: Callable[..., tuple[ReadinessCheck, ...]], *args: object) -> tuple[ReadinessCheck, ...]:
    # Polls in 0.1s ticks like run_sync_in_worker (async_workers.py:82-118), not a
    # bare wait_for(future) — an executor future is exactly the case that loop
    # guards against (selector wake can lag in this repo's sandboxed CI);
    # readiness_report's wait_for below wraps coroutines instead, a different case.
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(_READINESS_EXECUTOR, functools.partial(fn, *args))
    elapsed = 0.0
    try:
        while elapsed < _PROBE_TIMEOUT_SECONDS:
            done, _pending = await asyncio.wait({future}, timeout=0.1)
            if done:
                return future.result()
            elapsed += 0.1
        return (ReadinessCheck(label, False, f"probe timed out after {_PROBE_TIMEOUT_SECONDS}s"),)
    except Exception as exc:  # a probe must degrade, never crash /api/ready (BLE not in this repo's ruff select — no noqa needed)
        return (ReadinessCheck(label, False, f"probe failed ({type(exc).__name__})"),)
    finally:
        if not future.done():
            future.add_done_callback(_drain)

async def readiness_report(settings: WebSettings, session_engine: Engine) -> ReadinessReport:
    try:
        grouped = await asyncio.wait_for(asyncio.gather(
            bounded("session_db", _check_session, session_engine),
            bounded("landscape_db", _check_landscape, settings),
            bounded("data_dir", _dir_check, "data_dir", settings.data_dir),
            bounded("payload_store", _dir_check, "payload_store", settings.get_payload_store_path()),
            bounded("blob_dir", _dir_check, "blob_dir", settings.data_dir / "blobs"),
        ), timeout=_TOTAL_TIMEOUT_SECONDS)
    except TimeoutError:
        return ReadinessReport(False, (ReadinessCheck("overall", False, f"exceeded {_TOTAL_TIMEOUT_SECONDS}s ceiling"),))
    try:
        checks = [_check_auth_mode(settings)]
    except Exception as exc:  # WebSettings._validate_auth_fields makes this unreachable today; guards a future auth-mode literal added without updating this mirror
        checks = [ReadinessCheck("auth_mode", False, f"probe failed ({type(exc).__name__})")]
    for group in grouped:
        checks.extend(group)
    return ReadinessReport(ready=all(c.ok for c in checks), checks=tuple(checks))
```
`_check_session`/`_check_landscape` mirror only the connectivity shape of the CLI DB check (`cli.py:3044-3067`: `with engine.connect() as conn: conn.execute(text("SELECT 1"))`) — never its `except Exception as e: ...str(e)` branch (`cli.py:3057-3060`, a known leak pattern); both let every exception propagate to `bounded()`'s single, already-redacted (`type(exc).__name__`-only) catch. `_check_landscape` opens a short-lived bare `Engine` via `create_engine`, not `LandscapeDB.from_url()`: `from_url()` eagerly calls `_validate_schema()` (`database.py:971,1385`), which raises `SchemaCompatibilityError` (a bare `Exception` subclass) for any non-`CURRENT` schema — this repo's own `data/runs/audit.db` is in exactly that state today — making `probe_landscape_schema` unreachable and collapsing two named checks into one. `_check_auth_mode(settings) -> ReadinessCheck` mirrors `settings.auth_provider` at `app.py:830-857`; `WebSettings._validate_auth_fields` (`config.py:478-548`) already makes its failure branches unreachable through normal construction, so `ok=False` here is defense-in-depth against a future auth-mode literal, not a live signal — note this in the ops doc (Task 2).

**Steps:**
- [ ] Write `test_probe_directory_writable_creates_missing_dir_and_cleans_up`, `test_probe_directory_writable_reports_exception_class_only_no_path` (blocked parent), `test_probe_directory_writable_reports_exception_class_only_when_unlink_fails` (monkeypatch `Path.unlink` to raise `OSError("EIO")`; the failure propagates through `_dir_check` to `bounded()`'s catch, which must still redact to `type(exc).__name__` only) as plain sync tests. Mark every `TestReadinessCache`/`TestReadinessReport` test `@pytest.mark.asyncio` (strict-mode `pytest-asyncio`, mirrors `test_app.py:334`). `TestReadinessCache::{test_returns_cached_within_ttl, test_recomputes_after_ttl, test_concurrent_callers_collapse_to_one_compute}` (fake clock + counting async `compute`).
- [ ] `TestReadinessReport`: pre-seed a landscape DB via `LandscapeDB.from_url(f"sqlite:///{tmp_path}/landscape.db", create_tables=True)` (test-fixture only) and pass that explicit `landscape_url` to `_settings(...)` (sidesteps `create_app`'s `runs/`-dir creation at `app.py:818`, which a direct `readiness_report()` call never runs); build `session_engine` via `create_session_engine` + `initialize_session_schema` (mirrors `app.py:874-875`). `test_ready_true_when_all_checks_pass`: assert `ready is True` and exactly 8 named checks, all `ok`. `test_ready_false_and_no_leakage_when_session_db_unreachable` / `..._landscape_unreachable`: `monkeypatch.setattr(readiness, "_check_session"/"_check_landscape", ...)` with a stub raising `OperationalError("connect", {}, Exception("...sentinel-credential-9f3a..."))`; assert `ready is False`, exactly 7 checks (the paired `*_schema` check is absent because connectivity itself failed), the surviving `*_db` check's `detail == "probe failed (OperationalError)"`, and `"sentinel-credential-9f3a"` in no detail. This is driver-independent, unlike a live bad DSN: psycopg v3 isn't installed in this repo, so `postgresql+psycopg://...` raises `ModuleNotFoundError` before any socket attempt, letting a leaking `str(exc)` implementation pass a live-DSN test vacuously. `test_ready_false_when_landscape_schema_stale`: `monkeypatch.setattr(readiness, "probe_landscape_schema", lambda engine: SchemaState.STALE)` against the reachable pre-seeded landscape DB; assert 8 checks, `landscape_db.ok is True`, `landscape_schema.ok is False` and `detail == "schema state: STALE"`. `test_overall_check_on_5s_ceiling`: `monkeypatch.setattr(readiness, "_TOTAL_TIMEOUT_SECONDS", 0.001)`; assert `detail == "exceeded 0.001s ceiling"`.
- [ ] `TestCheckAuthMode` (direct, bypassing `WebSettings`'s validator so both branches are reachable): call `_check_auth_mode` with `types.SimpleNamespace(auth_provider=..., oidc_issuer=..., oidc_audience=..., entra_tenant_id=...)` stubs covering local / oidc-complete / oidc-missing-issuer / entra-complete; assert `ok` matches field presence.
- [ ] Run `pytest tests/unit/web/test_readiness.py -v` → `ModuleNotFoundError: No module named 'elspeth.web.readiness'`.
- [ ] Implement `readiness.py` as above.
- [ ] Run again → all pass.
- [ ] `git add src/elspeth/web/readiness.py tests/unit/web/test_readiness.py && git commit -m "feat(web): add budgeted, cached, redacted readiness probes on a dedicated worker pool"`

### Task 2: Wire `GET /api/ready`, guard startup orphan-finalization, prove liveness independence, document probe wiring

**Files:**
- Modify: `src/elspeth/web/app.py:896` (insert `app.state.readiness_cache = ReadinessCache()`); `app.py:1159` (insert the `/api/ready` route); `app.py:293` (wrap `_finalize_orphaned_landscape_runs(...)` — **note:** by the time this task executes, Plan 04 Task 3 has already added a `create_tables` kwarg to this same call and widened `_periodic_orphan_cleanup`'s except tuple at `app.py:230-269`/`:237` from `(SQLAlchemyError, OSError)` to `except (SQLAlchemyError, OSError, SchemaCompatibilityError)`; wrap the already-modified `:293` call in that identical 3-tuple, not the narrower pair — `:293` is startup *cleanup*, not the schema *validation gate* (Plan 04's `validate_only_schema_or_raise` is the real fail-closed gate), so a landscape schema lost or drifted after boot raising `SchemaCompatibilityError` under Plan 04's `create_tables=False` belongs in the same recoverable-failure class as the other two here too, letting `/api/health`/`/api/ready` come up during a landscape outage. `SchemaCompatibilityError` is already imported at `app.py:41` by Plan 04 Task 3 — no new import needed here. `exc_class`-only logging — see Steps for why). Add imports (`dataclasses.asdict`, `elspeth.web.readiness.{ReadinessCache, readiness_report}`).
- Modify: `tests/unit/web/test_app.py` (extend `TestHealthEndpoint`; add `TestReadyEndpoint`, `TestLifespanLandscapeOutage`).
- Create: `docs/operator/aws-ecs-health-and-readiness.md`.

**Interfaces:** Produces `GET /api/ready` returning `{"ready": bool, "checks": [{"name", "ok", "detail"}, ...]}`, status 200 if ready else 503, via `request.app.state.readiness_cache.get(lambda: readiness_report(request.app.state.settings, request.app.state.session_engine))`.

```python
@app.get("/api/ready")
async def ready(request: Request) -> JSONResponse:
    cache: ReadinessCache = request.app.state.readiness_cache
    report = await cache.get(lambda: readiness_report(request.app.state.settings, request.app.state.session_engine))
    return JSONResponse(status_code=200 if report.ready else 503,
                         content={"ready": report.ready, "checks": [asdict(c) for c in report.checks]})
```

**Steps:**
- [ ] In `test_app.py`, add `TestReadyEndpoint::test_ready_returns_200_when_all_checks_pass` (Task 1's pass-case fixture; 8 named checks all `ok`) and `test_ready_returns_503_with_redacted_body_when_landscape_unreachable` (monkeypatch `_check_landscape` per Task 1's technique; 503, `body["ready"] is False`, no sentinel in the response text). Extend `TestHealthEndpoint` with `test_health_independent_of_ready_when_landscape_unreachable` (same monkeypatch: `/api/health`→200, `/api/ready`→503 on one app instance) — docstring-note this proves only the route handler is dependency-free, since `_sync_asgi_client.py`'s `ASGITransport` never runs lifespan.
- [ ] Add `TestLifespanLandscapeOutage::test_lifespan_survives_landscape_unreachable_during_orphan_finalization`, mirroring `test_lifespan_startup_orphan_cleanup_terminalizes_landscape_run` (`test_app.py:739-776`) but simpler: seed a `running` web run with a `landscape_run_id` first (session-side only, no landscape row needed), then `monkeypatch.setattr(LandscapeDB, "from_url", staticmethod(lambda *a, **kw: (_ for _ in ()).throw(OperationalError("connect", {}, Exception("boom")))))` (deterministic, driver-independent). `async with lifespan(app): pass` — before the `app.py:293` fix this raises uncaught (the liveness-independence gap this plan's Goal claims to close); after, it must not raise, and `(await session_service.get_run(web_run.id)).status == "cancelled"` (session-side cancellation lands even though landscape finalization is skipped and logged).
- [ ] Run `pytest tests/unit/web/test_app.py -k "TestReadyEndpoint or test_health_independent or TestLifespanLandscapeOutage" -v` → route tests fail `AttributeError: 'State' object has no attribute 'readiness_cache'`; the lifespan test fails by raising `OperationalError` uncaught.
- [ ] Wire `app.state.readiness_cache` and the route as above; wrap the `app.py:293` call per Files.
- [ ] Run again → all pass. Run `pytest tests/unit/web/test_app.py -v` → no regressions (confirms `/api/health` unchanged).
- [ ] Create `docs/operator/aws-ecs-health-and-readiness.md` stating the four-way probe contract: ECS container `healthCheck` → `GET /api/health` (liveness only, never restarts on dependency failure; set `healthCheck.startPeriod` to ~90s, absorbing Plan 04's true two-cold-cluster validate-only-startup worst case — see Plan 04 Deviations — so ECS doesn't kill the task before its first readiness); ALB target-group health check → `GET /api/ready` (200/503, 2s/5s/2s budgets, unauthenticated since ALB presents no credentials); `elspeth doctor aws-ecs [--init-schema]` runs once as a pre-traffic task before the service update shifts traffic; `elspeth health` is explicitly **not** wired to any ECS probe. Also state this plan's Residual limitation (dedicated pool isolates but doesn't prevent exhaustion under a sustained wedge; manual task recycling may be needed) and that `auth_mode` reflects startup-validated settings, not a live probe.
- [ ] `git add src/elspeth/web/app.py tests/unit/web/test_app.py docs/operator/aws-ecs-health-and-readiness.md && git commit -m "feat(web): wire GET /api/ready, guard startup orphan-finalization against landscape outages, document ECS probe contract"`
