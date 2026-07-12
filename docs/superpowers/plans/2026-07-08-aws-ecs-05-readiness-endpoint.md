# AWS ECS Readiness Endpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an unauthenticated, dependency-aware `GET /api/ready` with deterministic redacted checks, hard request/per-check budgets, cancellation-safe single-flight caching, and bounded probe admission, while keeping `GET /api/health` shallow after Plan 04's pre-bind startup gate succeeds.

**Architecture:** `src/elspeth/web/readiness.py` owns immutable report types, a five-label readiness probe runner, a cancellation-safe 2-second cache, and the dependency checks. The runner admits at most one unresolved future per closed label, uses a dedicated five-worker executor, and refuses duplicate/saturated work rather than feeding `ThreadPoolExecutor`'s unbounded queue. PostgreSQL checks use dedicated, tightly bounded one-shot engines and connection-bound Plan 02 probes; filesystem checks use the same exclusive/unlinked descriptor algorithm as doctor. `app.py` wires the runner/cache/route and degrades the complete post-gate orphan-cleanup unit without weakening Plan 04's startup validation.

**Tech Stack:** Python 3.13, asyncio, `concurrent.futures`, SQLAlchemy 2.0, psycopg v3, FastAPI/Starlette, pytest, pytest-asyncio, PostgreSQL 16 testcontainers.

**Depends on:**

- Plan 04 (`elspeth-03cf981d4a`) is the direct hard dependency and already carries Plans 01/02 transitively. Task 3 consumes Plan 04's pre-auth ordering, orphan-cleanup `create_tables` argument, and exception imports.
- Plan 02 provides `SchemaState`, `postgres_engine_kwargs`, and connection-bound probes.
- Plan 03 provides the filesystem-probe security contract this plan mirrors exactly; no runtime import from doctor is introduced.
- Filigree implementation step: `elspeth-1a1c31bcce`. This planning repair does not claim it.

**Required downstream handoffs:** Plan 10/12 must mechanically verify the ALB target group is configured to probe exactly `/api/ready`, not merely curl the route manually. Plan 12 also owns the zero-skip PostgreSQL readiness file from Task 4.

**Global Constraints:**

- `/api/ready` is unauthenticated because ALB cannot present credentials. Cache single-flight plus bounded admission is the DoS/stampede control.
- Every response has exactly eight unique, ordered checks: `auth_mode`, `session_db`, `session_schema`, `landscape_db`, `landscape_schema`, `data_dir`, `payload_store`, `blob_dir`. A dependency failure never deletes its paired schema check; it reports static “not checked”/failure detail.
- Each dependency group returns within 2 seconds of wall-clock time. The entire route, including cache waiting, returns within 5 seconds. Deadlines use `loop.time()`, never synthetic polling counters.
- A timed-out/cancelled queued future is cancelled. A running future remains registered under its label until its completion callback drains it; later refreshes return a static in-flight/saturated failure without submitting duplicate work.
- Cancellation of the leading request never cancels the shared computation. Followers await the same shielded task; no request owns probe lifetime.
- PostgreSQL readiness engines use raw AWS URLs, connection/pool/statement timeouts below the 2-second async ceiling, small isolated pools, connection-bound probes, and `BaseException`-safe disposal. Default-mode file-backed SQLite may use the live session engine in the worker. A `sqlite:///:memory:` session engine is never checked out from the readiness worker: SQLAlchemy's live `SingletonThreadPool` would expose a different empty database per thread and may violate SQLite thread affinity, so readiness returns a deterministic paired not-ready result instructing the developer/test operator to use file-backed SQLite.
- No filesystem path derivation, stat, create/read/fsync/delete operation, database connect, or schema inspection runs on the event-loop thread.
- Filesystem probes never call `mkdir` and never reopen a probe by pathname. Payload checks reject the same symlink/unsafe-mode shapes as Plans 03/04 and `FilesystemPayloadStore`.
- Response bodies and logs contain only closed check names, schema enum names, static remedies, and exception class names. Never expose URLs, credentials, SQL, DBAPI messages, paths, exception messages, or tracebacks.
- `GET /api/health` remains the existing constant shallow response. Plan 04 intentionally remains dependency-sensitive before socket bind; this plan covers liveness after successful startup and post-gate outages.

---

### Task 1: Implement bounded probe admission and cancellation-safe caching

**Files:**

- Create: `src/elspeth/web/readiness.py`
- Create: `tests/unit/web/test_readiness.py`

**Interfaces — Produces:**

```python
@dataclass(frozen=True, slots=True)
class ReadinessCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    ready: bool
    checks: tuple[ReadinessCheck, ...]


class ReadinessProbeRunner:
    async def run(
        self,
        label: ReadinessProbeLabel,
        check_names: tuple[str, ...],
        fn: Callable[..., tuple[ReadinessCheck, ...]],
        *args: object,
    ) -> tuple[ReadinessCheck, ...]: ...

    def close(self) -> None: ...


class ReadinessCache:
    async def get(
        self,
        compute: Callable[[], Awaitable[ReadinessReport]],
    ) -> ReadinessReport: ...
```

`ReadinessProbeLabel` is the closed literal `"session" | "landscape" | "data_dir" | "payload_store" | "blob_dir"`. The runner owns `ThreadPoolExecutor(max_workers=5, thread_name_prefix="readiness-worker")`, a `threading.Lock`, and `dict[label, concurrent.futures.Future[tuple[ReadinessCheck, ...]]]`.

Submission/lifecycle rules:

1. Under the lock, if the same label has an unresolved future, return one static failed check for each requested `check_names`; submit nothing.
2. Otherwise call `executor.submit` and store the future before releasing the lock. There are only five labels and at most one future per label, so submitted+running work is bounded at five and the executor queue cannot grow across refreshes.
3. Attach the completion callback only **after releasing the lock**. A concurrent future may already be complete, in which case `add_done_callback` invokes synchronously; attaching under the registry lock would deadlock. The callback reacquires the lock, removes only the identical registered future, releases the lock, and then calls `future.exception()` unless cancelled, so late failures are always retrieved without logging their text.
4. Wrap the concurrent future once with `asyncio.wrap_future`, immediately attach an asyncio-side done callback that calls `wrapped.exception()` unless cancelled, then poll it with `asyncio.wait({wrapped}, timeout=min(0.1, deadline - loop.time()))` against `deadline = loop.time() + 2.0`. Do not use an unshielded short `asyncio.wait_for`, which would cancel the shared wrapper on its first polling timeout. Event-loop delay therefore consumes the real budget.
5. On per-check timeout, whole-report cancellation, caller cancellation, or runner close, cancel/drain **both** layers: call `wrapped.cancel()` and `future.cancel()`, retrieving an already-completed wrapper exception without rendering or logging it. If the source future was already running, retain it in the registry until the source callback completes; never remove early and resubmit. The asyncio done callback consumes any late source exception so the event loop cannot emit a credential/path-bearing `Future exception was never retrieved` message.
6. `close()` is idempotent. Under the lock it marks the runner closed and snapshots the registered futures, then releases the lock before calling `future.cancel()` or `executor.shutdown(wait=False, cancel_futures=True)`: cancelling a queued future may invoke its callback synchronously, so neither cancellation nor shutdown may occur while holding the registry lock. No new submission is accepted afterward.

Failure rendering is deterministic: runner timeout/capacity/exception paths emit all names in `check_names`, using class-only/static details. For a DB exception, `<kind>_db` says `probe failed (ClassName)` and `<kind>_schema` says `not checked: connectivity probe failed`; for timeout/in-flight saturation both use static label-only details.

`ReadinessCache` stores a shared `asyncio.Task[ReadinessReport]`, not a coroutine awaited while holding the lock:

- the lock protects only freshness/task selection and result publication;
- every caller awaits `asyncio.shield(shared_task)` outside the lock;
- cancelling the leader leaves the task alive and visible to followers;
- a completed task is harvested under the lock before any replacement is created, even if its original requester disappeared;
- successful reports receive a completion timestamp and 2-second TTL; failed task exceptions clear the task without poisoning the prior cached report;
- cache waiting is ultimately bounded by the route's separate 5-second absolute deadline.

- [ ] Write runner tests with `threading.Event`-blocked functions. Prove one submission per label, repeated refreshes do not increase submitted/running counts, a different free label can still run, queued cancellation removes only after completion/cancel callback, and `close()` rejects new work. Add immediate-completion-before-callback-registration and concurrent-`close()` regression tests with a short hard test timeout to prove neither callback path deadlocks. Release every event in `finally` so pytest cannot retain non-daemon worker threads.
- [ ] Add a wall-clock test using an event-loop heartbeat and a blocking worker: `run` returns at 2 seconds within a narrow monotonic tolerance while the loop continues ticking. Add a deliberate event-loop stall and prove deadline calculation uses `loop.time()` rather than 20 synthetic 0.1-second iterations.
- [ ] Add cancellation tests: cancel an awaiting runner caller and prove the running future stays registered; after releasing it, its exception/result is drained and the label becomes admissible again. A queued future must be cancelled immediately. Install a capturing event-loop exception handler, make an abandoned worker fail late with credential/path sentinels, force callback/GC turns, and assert the handler receives neither an unretrieved-future event nor any sentinel.
- [ ] Write cache tests for fresh-hit TTL, recompute after TTL, 50 concurrent callers collapsing to one task, cancelled leader plus surviving follower, completed-task harvesting with no original waiter, compute exception recovery, and a follower never waiting for two sequential computations.
- [ ] Implement the types, runner, and cache. Use correctly parameterized `concurrent.futures.Future[...]`; do not pass `asyncio.Future[object]` to a callback expecting a narrower result type.
- [ ] Run:

  ```bash
  uv run pytest tests/unit/web/test_readiness.py -k "ProbeRunner or ReadinessCache" -q
  ```

  Expected: all Task 1 tests pass.

- [ ] Commit:

  ```bash
  git add src/elspeth/web/readiness.py tests/unit/web/test_readiness.py
  git commit -m "feat(web): add bounded readiness probe runner and cache"
  ```

---

### Task 2: Add redacted database, filesystem, auth, and report checks

**Files:**

- Modify: `src/elspeth/web/readiness.py`
- Modify: `tests/unit/web/test_readiness.py`

**Database check contract:**

- In AWS mode use raw `settings.session_db_url` / `settings.landscape_url`, asserted present after Plan 04 startup. Never call fallback getters.
- For PostgreSQL create an owned one-shot engine per DB check. Session uses `create_session_engine`; Landscape uses `create_engine`. Start from `postgres_engine_kwargs(url)` but override to `pool_size=1`, `max_overflow=0`, and `pool_timeout=0.5`; add `connect_args={"connect_timeout": 1}`.
- Inside one `with engine.connect() as conn`, execute the static `SET LOCAL statement_timeout = '1000ms'`, then `SELECT 1`, then `probe_session_schema(conn)` or `probe_landscape_schema(conn)`. Connectivity and shape therefore use the same checkout.
- Dispose every owned engine in `finally` for success, `Exception`, and `BaseException` paths.
- In default mode, reuse `app.state.session_engine` only for file-backed SQLite. Detect `sqlite:///:memory:` inside the session worker and return deterministic `session_db`/`session_schema` not-ready checks with the static remedy `in-memory SQLite is not readiness-probeable; use a file-backed session database`; never checkout that engine on the readiness thread or claim schema currency. Landscape/default uses its resolved configured/fallback URL and an owned engine. No PostgreSQL check may borrow the live session pool.

Each DB function returns exactly two checks. `CURRENT` passes schema; any other `SchemaState` produces `ok=False` with `schema state: <NAME>`. Exceptions escape to the runner, which emits the deterministic paired failures.

**Filesystem check contract:**

- Resolve/check every directory inside its worker function. `blob_dir` calls `allowed_source_directories(str(settings.data_dir))[0]` inside the runner boundary; path-resolution failure becomes a static check, never route 500 or event-loop EFS I/O.
- For payload, use raw AWS `payload_store_path`; default mode may use its fallback. Before the active probe, use `lstat`, reject symlink/non-directory/`stat.S_IWGRP | stat.S_IWOTH`, and require strict resolution.
- Active probe uses `tempfile.mkstemp(prefix=".readiness-probe-", dir=directory)`, immediately unlinks the name while the FD is open, wraps with `os.fdopen(fd, "w+b")`, writes, flushes, fsyncs, seeks, and reads the sentinel through that descriptor. Cleanup closes any owned FD and removes a still-named file in `finally`; cleanup uncertainty is failure.

**Auth check contract:** `_check_auth_mode` is pure and total. Local passes. OIDC requires `oidc_issuer`, `oidc_audience`, and `oidc_client_id`. Entra requires `entra_tenant_id`, `oidc_audience`, and `oidc_client_id`. An unknown provider fails. It does not perform discovery; Plan 04/lifespan already validates/resolves that startup contract.

`readiness_report(settings, session_engine, runner)` gathers the five runner calls concurrently, prepends `auth_mode`, and returns the exact eight-name order. It has its own 5-second ceiling, but route-level Task 3 timing also includes cache wait. Any unexpected path/auth/gather error becomes one deterministic eight-check not-ready report via a final static boundary; it never escapes as HTTP 500. `_finalize` logs one `readiness_check_not_ready` event per failing check with `check` and already-redacted `detail` only.

- [ ] Add DB tests proving raw AWS URLs, isolated pool/connect/statement timeout kwargs, session factory usage, same-Connection `SELECT 1`+probe, no live PostgreSQL pool checkout, file-backed default SQLite identity reuse, deterministic not-ready/no-checkout behavior for `sqlite:///:memory:`, all schema states, paired deterministic failures, and owned-engine disposal under success/exception/`KeyboardInterrupt`.
- [ ] Inject credential-bearing URL/SQL/DBAPI sentinels into construction, connection, statement, and probe failures. Assert no response detail or captured log contains them; only exception classes appear.
- [ ] Add filesystem tests for missing/non-directory, payload symlink/unsafe mode, secret-bearing resolve/stat failure, concurrent probes, exclusive creation, immediate unlink, same-FD readback, fsync failure, cleanup failure, and no leftover named file. A blocking blob resolver must time out through the runner while an event-loop heartbeat continues.
- [ ] Add auth tests for every required field independently, including missing `oidc_client_id` for both OIDC and Entra, and unknown provider.
- [ ] Add report tests pinning exact ordered unique names, all-green readiness, each individual failure, 2-second grouped timeout, 5-second overall timeout, class-only logging, and no exception/path/URL/SQL leakage.
- [ ] Run:

  ```bash
  uv run pytest tests/unit/web/test_readiness.py -q
  ```

  Expected: all tests pass.

- [ ] Commit:

  ```bash
  git add src/elspeth/web/readiness.py tests/unit/web/test_readiness.py
  git commit -m "feat(web): add bounded dependency readiness checks"
  ```

---

### Task 3: Wire the route, complete post-gate cleanup degradation, and document probes

**Files:**

- Modify: `src/elspeth/web/app.py` (readiness state/route, lifespan cleanup and shutdown)
- Modify: `src/elspeth/web/sessions/protocol.py` (durable reconciliation-candidate methods)
- Modify: `src/elspeth/web/sessions/service.py` (pending/complete orphan-reconciliation marker queries)
- Modify: `tests/unit/web/test_app.py`
- Modify: `tests/unit/web/sessions/test_service.py`
- Create: `docs/operator/aws-ecs-health-and-readiness.md`

After the session engine is established in either runtime mode, create one `ReadinessProbeRunner` and `ReadinessCache`, store both on `app.state`, and register an idempotent app finalizer for the runner. In AWS mode this construction occurs only after Plan 04's pre-bind gate succeeds; in default mode the route receives the same state and remains usable. Lifespan shutdown calls `runner.close()` after cancelling periodic cleanup and before final engine disposal.

Route contract:

```python
@app.get("/api/ready")
async def ready(request: Request) -> JSONResponse:
    async def compute() -> ReadinessReport:
        return await readiness_report(
            request.app.state.settings,
            request.app.state.session_engine,
            request.app.state.readiness_probe_runner,
        )

    try:
        async with asyncio.timeout(5.0):
            report = await request.app.state.readiness_cache.get(compute)
    except TimeoutError:
        report = overall_timeout_report()
    return JSONResponse(
        status_code=200 if report.ready else 503,
        content={"ready": report.ready, "checks": [asdict(check) for check in report.checks]},
    )
```

The route timeout covers lock/task selection, follower waiting, and computation. `ReadinessCache.get` shields the shared task, so route timeout/caller cancellation does not cancel the batch. `overall_timeout_report()` returns the same exact eight names with static timeout details and passes through the normal class-free `_finalize` logging path.

Make Landscape orphan reconciliation durable without a schema migration by using exact closed markers in the existing session-run `error` reason. Define startup and periodic reason constants that retain the human-readable reason and end in `[landscape-reconciliation:pending]`. Add protocol/service methods that (a) list every `cancelled` row with that exact pending suffix, including rows whose `landscape_run_id` is null, and (b) atomically replace only that suffix with an outcome-specific closed suffix for specified run IDs: `[landscape-reconciliation:complete]` for a null anchor or an existing Landscape row, and `[landscape-reconciliation:absent]` when a non-null anchor has no Landscape row. No substring/prefix inference, arbitrary error rewriting, or new database column is allowed.

The live execution order persists `status=running` plus `landscape_run_id` before constructing Landscape and before `begin_run`, so SIGKILL can legitimately leave a non-null anchor with no Landscape run row. The same observed absence could also mean later audit-row loss; it must never be silently blessed as ordinary completion. Reconciliation therefore has three idempotent outcomes: a running Landscape row becomes `INTERRUPTED`; an existing terminal row is already complete; and an absent row needs no possible Landscape mutation but receives the distinct durable `absent` marker. For absence, emit one static `orphan_landscape_run_absent` error event with an outcome/count and `operator_action="investigate audit-row absence"` only — no run ID, URL, path, or exception text. The distinct marker closes automatic retry while preserving the audit exception for operator investigation. A null `landscape_run_id` is a normal no-op candidate and receives the ordinary complete marker rather than remaining pending forever.

Wrap the complete startup orphan-cleanup/reconciliation unit after Plan 04's validation:

1. initialize `cancelled_runs = []`;
2. inside one try, await `cancel_all_orphaned_run_records` using the exact pending-marker startup reason;
3. list **all** durable pending reconciliation candidates, including rows committed by an earlier failed process;
4. partition null anchors as no-op, then call idempotent `_finalize_orphaned_landscape_runs(..., create_tables=<Plan 04 policy>)` for anchored candidates — running rows become interrupted, already-terminal rows are not rewritten, and absent pre-begin rows return the static absent outcome rather than raising `AuditIntegrityError`;
5. only after the whole Landscape pass succeeds, atomically mark null/existing-row candidates complete and missing-row candidates absent in the session database; if Landscape fails partway or either marker update fails, pending markers remain and the entire idempotent set is retried;
6. catch `(SQLAlchemyError, OSError, SchemaCompatibilityError)` and log only `lifespan_orphan_cleanup_failed`, `exc_class`;
7. still emit cancellation telemetry for any new session cancellations that completed before Landscape failure.

Apply the same list → idempotent Landscape finalize → complete-marker sequence in Plan 04's periodic loop after cancelling new orphans with the periodic pending-marker reason. Thus periodic cleanup and the next process startup can recover a session row already committed as `cancelled`; neither relies on `cancel_all_orphaned_run_records` returning that row again.

This does not weaken the pre-bind gate: it handles a dependency lost after validation but before/during lifespan. `/api/health` remains the existing constant 200 route.

- [ ] Add route tests for 200/all-green and 503/every dependency failure, exact eight-check JSON, no sentinel leakage, and the route's true 5-second ceiling including a waiter behind a shared compute.
- [ ] Add cancellation/single-flight ASGI tests: cancel the leading readiness request, keep a follower, and prove one batch; after route timeout, a later request sees the same in-flight labels rather than submitting duplicates.
- [ ] Extend `TestHealthEndpoint`: with readiness DB/EFS checks blocked or failing, `/api/health` still returns its unchanged 200 body while `/api/ready` returns bounded 503 on the same app. Verify an event-loop heartbeat/health call remains responsive while blob resolution is blocked in the readiness worker.
- [ ] Add session-service tests pinning exact pending-marker selection (including null anchors), exclusion of unrelated/user errors and every closed marker, outcome-specific compare-and-update behavior, and preservation of the human-readable reason. Add startup/periodic lifespan tests that fail after the session cancellation commit but before Landscape completion, then retry and prove the same durable candidate becomes `INTERRUPTED` and receives the complete marker. Also cover failure after Landscape completion but before session marker update; the next pass must idempotently mark complete. Reproduce the real SIGKILL window with a non-null session anchor but no Landscape row and prove startup emits only the static operator-action event, writes the distinct durable absent marker, and does not reselect it; prove a null-anchor candidate receives the ordinary complete marker without Landscape access. Use `composer_boot_probe_enabled=False`, a controllable short periodic interval where required (otherwise `3600`), and `patch("httpx.AsyncClient", return_value=_StaticAsyncClient([]))` so unrelated composer/OIDC/catalog work cannot escape. Assert class-only logs, no sentinel, session telemetry only for newly cancelled rows, and clean runner shutdown.
- [ ] Create `docs/operator/aws-ecs-health-and-readiness.md` with the four exact surfaces: container health check → `/api/health`; ALB target group → `/api/ready`; doctor as one-shot pre-traffic task; `elspeth health` never wired. Document 2s/5s/2s budgets, bounded admission/saturation remedy, unauthenticated rationale, Plan 04's pre-bind connection-refused window, and both ~150s ECS startup/grace knobs. Also document that `[landscape-reconciliation:absent]` is a durable audit exception (possible pre-begin crash or later audit loss), name the static log event, and require operator investigation rather than treating it as successful audit reconciliation.
- [ ] Run:

  ```bash
  uv run pytest tests/unit/web/test_readiness.py tests/unit/web/test_app.py tests/unit/web/sessions/test_service.py -q
  ```

  Expected: all tests pass.

- [ ] Commit:

  ```bash
  git add src/elspeth/web/app.py src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/service.py tests/unit/web/test_app.py tests/unit/web/sessions/test_service.py docs/operator/aws-ecs-health-and-readiness.md
  git commit -m "feat(web): wire bounded readiness and resilient post-gate cleanup"
  ```

---

### Task 4: Prove PostgreSQL readiness and bounded redacted failure

**Files:**

- Create: `tests/testcontainer/web/test_aws_ecs_readiness_postgres.py`

Mark only `pytest.mark.testcontainer`; keep the file outside auto-`integration` marking. Reuse the Plan 04 pattern: one module Postgres 16 container, fresh distinct databases per test, safe UUID-derived/quoted identifiers, complete AWS settings, and pre-created data/payload/blob directories. Initialize schemas with owner engines, render credential-preserving runtime URLs, and dispose all engines/apps in fixture `finally`.

- [ ] `test_ready_returns_200_for_current_postgres`: construct the Plan 04-validated app, call `/api/ready`, require status 200, exact eight unique names, all `ok`, and independently probe both schemas `CURRENT`.
- [ ] `test_ready_returns_bounded_redacted_503_after_connect_revocation`: after app construction, use the admin connection to `ALTER ROLE <safely quoted runtime role> NOLOGIN` for one runtime role and terminate that role's existing backends so the next dedicated readiness connection must fail. (`REVOKE CONNECT` from the role alone is insufficient because a new PostgreSQL database grants `CONNECT` to `PUBLIC`.) Before exercising HTTP, assert that a fresh direct connection with the runtime URL actually fails; this proves the fault injection is effective. Then call `/api/ready`; require 503 in under 5 seconds, paired DB/schema failures, and absence of username/password/URL/SQL/driver text from body and captured logs. Restore `LOGIN`, release blocked work, and dispose connections in `finally`.
- [ ] `test_repeated_failed_refreshes_do_not_grow_probe_work`: hold one database probe until after repeated cache expiries/requests; require at most one unresolved future for its label and at most five total runner futures, then release and prove recovery.
- [ ] Verify Docker and zero skips:

  ```bash
  docker info
  uv run pytest tests/testcontainer/web/test_aws_ecs_readiness_postgres.py -m testcontainer -q
  ```

  Expected: both commands exit 0 and the file reports zero skips. Docker unavailable is `BLOCKED`.

- [ ] Commit:

  ```bash
  git add tests/testcontainer/web/test_aws_ecs_readiness_postgres.py
  git commit -m "test(web): prove bounded PostgreSQL readiness"
  ```

---

### Task 5: Run Plan 05 handoff gates

**Files:** Verify only files touched by Tasks 1-4. No new implementation files are expected.

- [ ] Run focused and subsystem tests:

  ```bash
  uv run pytest tests/unit/web/test_readiness.py tests/unit/web/test_app.py tests/unit/web/sessions/test_service.py -q
  uv run pytest tests/unit/web/ -q
  docker info
  uv run pytest tests/testcontainer/web/test_aws_ecs_readiness_postgres.py -m testcontainer -q
  ```

  Expected: every command exits 0 and the Docker file reports zero skips.

- [ ] Run repository static gates:

  ```bash
  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  git diff --check
  ```

  Expected: every command exits 0.

- [ ] Because this is an unauthenticated external-input and filesystem/database boundary, use the `wardline-gate` skill and run:

  ```bash
  wardline scan . --fail-on ERROR
  ```

  Expected: exit 0. On a finding, follow scan → explain → boundary fix → rescan; add no waiver or signed allowlist entry.

- [ ] Record Plan 10/12 handoffs: the target group must have `HealthCheckEnabled == true`, exact `HealthCheckPath == "/api/ready"`, exact success matcher `HttpCode == "200"`, and `HealthCheckTimeoutSeconds >= 6` (strictly beyond the 5-second endpoint ceiling). Plan 12 runs:

  ```bash
  docker info
  uv run pytest tests/testcontainer/web/test_schema_probe_postgres.py tests/testcontainer/web/test_doctor_aws_ecs_postgres.py tests/testcontainer/web/test_aws_ecs_validate_only_startup.py tests/testcontainer/web/test_aws_ecs_readiness_postgres.py -m testcontainer -q
  ```

  All four files must execute with zero skips.

- [ ] Create a final commit only if verification required a scoped correction. Stage exact files; never use `git add -A` or create an empty verification commit.
