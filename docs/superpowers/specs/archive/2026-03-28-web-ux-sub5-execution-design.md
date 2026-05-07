# Web UX Sub-Spec 5: Execution

**Status:** Draft
**Date:** 2026-03-28
**Parent Spec:** `docs/superpowers/specs/2026-03-28-web-ux-composer-mvp-design.md`
**Phase:** 5
**Depends On:** Sub-Specs 2 (Auth & Sessions), 3 (Catalog), 4 (Composer)
**Blocks:** Sub-Spec 6 (Frontend)

---

## Scope

**In scope:**

- ExecutionService protocol and implementation (validate, execute, get_status, cancel)
- Dry-run validation (Stage 2) using real engine code paths
- Background thread execution model with ThreadPoolExecutor
- ProgressBroadcaster for thread-safe async event delivery
- Cancel mechanism via threading.Event per run
- WebSocket endpoint for live progress streaming
- REST endpoints for validation, execution, status, cancel, and results
- ValidationResult and RunEvent response models
- Integration test: CSV source through passthrough transform to CSV sink, end-to-end through the web layer

**Out of scope:**

- Stage 1 composition-time validation (Phase 4, ComposerService)
- Frontend progress UI (Phase 6)
- Redis Streams event distribution (later increment)
- Worker pool isolation or multi-process execution
- Governance tiers or approval workflows
- Live preview execution with sample rows

---

## ExecutionService Protocol

The ExecutionService protocol defines four operations. `execute`, `get_status`,
and `cancel` are `async def` — they call async SessionService methods directly
(they run in the event loop thread). `validate` is synchronous (called via
`run_in_executor` from the route handler).

**validate(state: CompositionState, settings: WebSettings) -> ValidationResult** -- Synchronous dry-run validation. Generates YAML from the composition state, runs it through the real engine validation pipeline (settings loading, plugin instantiation, graph construction, graph validation), and returns a structured result with per-component error attribution. This is Stage 2 validation; Stage 1 happens inside the ComposerService tool-use loop.

**execute(session_id: UUID, state_id: UUID) -> UUID** -- Starts a background pipeline run. Returns the run ID immediately. Enforces the one-active-run-per-session constraint (B6) before submission. Creates the Run record in pending status, stores a shutdown Event for cancellation, submits the pipeline function to the thread pool, and attaches a done callback as a safety net.

**get_status(run_id: UUID) -> RunStatus** -- Returns the current run status, including rows_processed, rows_failed, error message (if failed), and landscape_run_id (if available). Reads from the Run record via SessionService.

**cancel(run_id: UUID) -> None** -- Cancels a run. If the run is actively executing (a shutdown Event exists for it), sets the Event, which the Orchestrator checks during row processing. If the run is pending (submitted to the thread pool but not yet started), updates the Run record status to cancelled directly.

The protocol is defined in `execution/protocol.py`. The implementation class is `ExecutionServiceImpl` in `execution/service.py`. Route handlers receive the service via FastAPI's dependency injection system (`dependencies.py`).

---

## Dry-Run Validation

Dry-run validation calls the real engine validation code. There is no parallel validation logic -- the web layer uses the same code paths as `elspeth run`.

### Validation Pipeline

The validation function `validate_pipeline(state: CompositionState, settings: WebSettings) -> ValidationResult` executes six steps in sequence:

1. **Source path allowlist check (security fix S2).** If the source options contain a `path` or `file` key, verify the resolved path is under `{settings.data_dir}/uploads/`. Use `Path.resolve()` to canonicalise the path (defeating `../` traversal), then check `resolved.is_relative_to(settings.data_dir / "uploads")`. If the check fails, return a `ValidationResult` with `is_valid=False` and an error attributed to the source component: `"Source file path must be within the uploads directory. Path '{path}' is not allowed."` This is defense in depth — the same check runs in Sub-Spec 4's `set_source` tool — but the execution boundary is the authoritative enforcement point because it cannot be bypassed by prompt injection.

2. **YAML generation.** Call `yaml_generator.generate_yaml(state)` to produce ELSPETH pipeline YAML from the CompositionState.

3. **Settings loading.** Write the generated YAML to a temporary file and call `load_settings(tmp_path)` to produce an `ElspethSettings` instance. The temporary file is cleaned up in a finally block.

4. **Plugin instantiation.** Call `instantiate_plugins_from_config(settings)` to create live plugin instances. This catches unknown plugin names, invalid config options, and batch-awareness violations on aggregations.

5. **Graph construction.** Call `ExecutionGraph.from_plugin_instances()` with the source, transforms, sinks, aggregations, gates, and coalesce settings from the PluginBundle. This catches route destination violations, missing on_success declarations, and unknown sink references.

6. **Graph validation.** Call `graph.validate()` to verify structural integrity (acyclicity, single source, reachable nodes, unique edge labels, route label completeness). Then call `graph.validate_edge_compatibility()` to verify schema compatibility across edges.

### Exception Handling

The validation function catches only typed exceptions. Bare `except Exception` is forbidden (W18 fix). The specific exception types caught are:

- `pydantic.ValidationError` -- from `load_settings()` when config fields fail Pydantic schema validation. Attributed to the component whose config is invalid.
- `FileNotFoundError` -- from `load_settings()` if the temp file write fails (should not happen, but the function declares it).
- `ValueError` -- from `instantiate_plugins_from_config()` when a plugin name is unknown or an aggregation uses a non-batch-aware transform. Attributed to the specific plugin reference.
- `GraphValidationError` -- from `ExecutionGraph.from_plugin_instances()` or `graph.validate()` when the graph structure is invalid. The error message contains node IDs that map to CompositionState node IDs for attribution.

Any exception not in this list propagates uncaught. If the engine adds new exception types, the validation function must be updated to handle them -- silent swallowing via a catch-all would hide bugs in our code (plugin ownership principle).

### Per-Component Attribution

When a validation error references a node ID (e.g., "gate_1"), the validation function maps that ID back to the corresponding component in the CompositionState. Each error in the ValidationResult carries a `component_id` field (nullable) that the frontend uses to highlight the failing component in the spec view.

Attribution is best-effort: some errors (e.g., "graph contains a cycle") are structural and cannot be attributed to a single component. These have `component_id = None`.

---

## ValidationResult Model

The ValidationResult model is defined in `execution/schemas.py` as a Pydantic model.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `is_valid` | `bool` | True if all checks passed |
| `checks` | `list[ValidationCheck]` | Individual checks that were run |
| `errors` | `list[ValidationError]` | Errors with per-component attribution |

**ValidationCheck fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Check name (e.g., "settings_load", "plugin_instantiation", "graph_structure", "schema_compatibility") |
| `passed` | `bool` | Whether this check passed |
| `detail` | `str` | Human-readable description of what was checked |

**ValidationError fields:**

| Field | Type | Description |
|-------|------|-------------|
| `component_id` | `str or None` | Node ID in CompositionState, or None for structural errors |
| `component_type` | `str or None` | "source", "transform", "gate", "sink", "aggregation", or None |
| `message` | `str` | Error description |
| `suggestion` | `str or None` | Suggested fix, if determinable |

The validation pipeline short-circuits: if settings loading fails, plugin instantiation is not attempted. Each check is recorded in `checks` regardless of whether it was reached, with `passed=False` and appropriate detail for skipped checks.

---

## Background Execution

### Thread Pool Model

Pipeline execution runs in a `concurrent.futures.ThreadPoolExecutor` with `max_workers=1`. One worker is sufficient because the one-active-run-per-session constraint (B6) limits concurrency at the application level, and multiple sessions running simultaneously is acceptable but not a high-throughput requirement for the MVP.

The thread pool is created during `ExecutionServiceImpl` construction and shut down during application shutdown (registered via FastAPI's lifespan context manager).

### Execution Flow

When `execute()` is called:

1. Call `session_service.get_active_run(session_id)` to check for any active run on this session (status in pending, running). If one exists, raise `RunAlreadyActiveError`. The ExecutionService does not have direct DB access -- all Run CRUD goes through SessionService (sub-2).
2. Load the CompositionState by `state_id` from the SessionService.
3. Generate pipeline YAML via `yaml_generator.generate_yaml(state)`.
4. Call `session_service.create_run(session_id, state_id)` to create a Run record with status=pending, binding it to the specific CompositionState version.
5. Create a `threading.Event` instance for shutdown signalling.
6. Store the Event in `self._shutdown_events[run_id]`.
7. Submit `_run_pipeline(run_id, pipeline_yaml, shutdown_event)` to the thread pool.
8. Attach `_on_pipeline_done` as a `future.add_done_callback()`.
9. Return the run_id.

### _run_pipeline() -- Background Thread Entry Point

This function runs in the background thread. It is the only function in the execution service that runs outside the asyncio event loop.

**Async/sync bridging (B8 fix):** All SessionService methods are `async def` (Sub-Spec 2 uses an async-compatible SQLAlchemy engine). `_run_pipeline()` runs in a synchronous `ThreadPoolExecutor` worker thread and cannot `await` directly. Every call from `_run_pipeline()` to an async SessionService method **must** use `asyncio.run_coroutine_threadsafe(coro, self._loop).result()` to bridge the sync/async boundary. The `self._loop` reference is the same event loop captured by the ProgressBroadcaster (obtained via `asyncio.get_running_loop()` inside the FastAPI lifespan). It is passed to `ExecutionServiceImpl` at construction time alongside the ProgressBroadcaster.

A private helper method encapsulates this pattern:

```python
def _call_async(self, coro: Coroutine[Any, Any, T]) -> T:
    """Bridge an async call from the background thread to the main event loop."""
    future = asyncio.run_coroutine_threadsafe(coro, self._loop)
    return future.result()  # Blocks the background thread until complete
```

All `session_service` calls in `_run_pipeline()` use `self._call_async(session_service.method(...))`.

**Lifecycle:**

1. Call `self._call_async(session_service.update_run_status(run_id, "running"))` to transition the Run record.
2. Construct LandscapeDB from `WebSettings.get_landscape_url()` (B3 fix).
3. Construct FilesystemPayloadStore from `WebSettings.get_payload_store_path()` (B3 fix).
4. Write the pipeline YAML to a temporary file and load settings via `load_settings(tmp_path)`.
5. Instantiate plugins via `instantiate_plugins_from_config(settings)`.
6. Build the ExecutionGraph via `ExecutionGraph.from_plugin_instances()` with the PluginBundle fields.
7. Construct the Orchestrator with the LandscapeDB instance.
8. Call `orchestrator.run()` with the graph, payload_store, shutdown_event, and a progress callback that broadcasts RunEvents through the ProgressBroadcaster.
9. On successful completion, call `self._call_async(session_service.update_run_status(run_id, "completed", landscape_run_id=result.run_id))` to record the terminal state and landscape_run_id from the RunResult.

**Exception and cleanup contract:** The entire body of `_run_pipeline()` is wrapped in `try/except BaseException/finally` (B7 fix). The except clause catches `BaseException`, not `Exception`, to handle `KeyboardInterrupt`, `SystemExit`, and OOM-triggered exceptions. On any exception, `self._call_async(session_service.update_run_status(run_id, "failed", error=str(exc)))` is called to record the failure before the exception is re-raised. The finally clause removes the shutdown Event from `self._shutdown_events`. Re-raising ensures the `future.add_done_callback()` safety net can also observe the failure.

**Invariant:** `_run_pipeline()` never calls `await` directly — it always uses `self._call_async()`. Direct `await` in a synchronous thread context would raise `SyntaxError` (the function is not `async def`) or, if manually constructed, would fail because there is no running event loop on the worker thread.

### _on_pipeline_done() -- Safety Net Callback

Registered via `future.add_done_callback()` on every submitted pipeline run. Its purpose is to catch exceptions that somehow bypassed the try/finally in `_run_pipeline()` -- a scenario that should not happen but would leave the Run record stuck in running status if it did. The callback calls `future.exception()` and, if non-None, logs the exception via structlog. It does not attempt to update the Run record via SessionService because `_run_pipeline()` should have already done so; this is purely a diagnostic backstop.

---

## Thread Safety

This section documents the four blocking review fixes (B1, B2, B7, B8) and the infrastructure fix (B3) that address the async/thread boundary in the execution service.

### B1 Fix: ProgressBroadcaster Uses loop.call_soon_threadsafe()

**Problem:** The `_run_pipeline()` function executes in a background thread managed by ThreadPoolExecutor. The ProgressBroadcaster's subscribers are `asyncio.Queue` instances owned by the asyncio event loop running in the main thread. Calling `queue.put_nowait()` directly from a background thread would corrupt the event loop's internal state, because asyncio primitives are not thread-safe.

**Solution:** The ProgressBroadcaster captures a reference to the asyncio event loop at construction time (when it is created in the app factory on the main thread). Its `broadcast()` method wraps every `queue.put_nowait()` call in `self._loop.call_soon_threadsafe()`, which schedules the put operation on the event loop's thread. This is the standard Python pattern for pushing data from a synchronous thread into an asyncio context.

**Construction timing:** The ProgressBroadcaster is constructed inside the FastAPI lifespan async context manager (defined in sub-1), NOT in the synchronous `create_app()` factory. The loop reference is obtained via `asyncio.get_running_loop()` inside the lifespan, which guarantees a running event loop exists and is Python 3.12+ compatible (`asyncio.get_event_loop()` emits a deprecation warning when no running loop exists). The broadcaster instance is then stored as application state and injected into the ExecutionServiceImpl constructor and into the WebSocket route handler via FastAPI's dependency injection.

### B2 Fix: Always Pass shutdown_event to Orchestrator.run()

**Problem:** `Orchestrator.run()` installs a SIGTERM/SIGINT signal handler when no `shutdown_event` is provided. Python's `signal.signal()` can only be called from the main thread. When `_run_pipeline()` runs in a ThreadPoolExecutor worker thread and does not pass a `shutdown_event`, the Orchestrator attempts `signal.signal()` from the worker thread, raising a `ValueError: signal only works in main thread` that kills the pipeline run.

**Solution:** `_run_pipeline()` always passes the pre-created `threading.Event` as the `shutdown_event` parameter to `orchestrator.run()`. The Orchestrator's existing logic already handles this: when `shutdown_event is not None`, it uses `nullcontext(shutdown_event)` instead of `self._shutdown_handler_context()`, skipping signal handler installation entirely. The shutdown Event is the same one stored in `self._shutdown_events[run_id]` for cancel support.

**Invariant:** Every call to `orchestrator.run()` from the web execution service must pass `shutdown_event`. There is no scenario where it should be omitted. If a future refactor changes the Orchestrator's signal handling, this invariant must be preserved.

### B3 Fix: LandscapeDB and PayloadStore Construction

**Problem:** The Orchestrator requires a LandscapeDB instance and a PayloadStore instance. In the CLI path, these are constructed from `ElspethSettings` fields. The web execution service does not use the CLI path -- it constructs the Orchestrator directly. Without explicit construction, the Orchestrator would have no database or payload store.

**Solution:** `WebSettings` provides two resolver methods: `get_landscape_url()` returns a SQLAlchemy connection string (defaulting to `sqlite:///{data_dir}/runs/audit.db`), and `get_payload_store_path()` returns a `Path` (defaulting to `{data_dir}/payloads/`). The `_run_pipeline()` function constructs `LandscapeDB(connection_string=self._settings.get_landscape_url())` and `FilesystemPayloadStore(base_path=self._settings.get_payload_store_path())` at the start of each run.

**Per-run construction:** LandscapeDB and PayloadStore are constructed fresh for each pipeline run, not shared across runs. This avoids SQLAlchemy session management complexity across threads and ensures each run has an independent database connection. The audit database file is shared (SQLite supports concurrent readers), but each run's LandscapeDB instance manages its own SQLAlchemy engine and connection pool.

### B7 Fix: BaseException Handling and Done Callback

**Problem:** If `_run_pipeline()` uses `except Exception`, it misses `KeyboardInterrupt`, `SystemExit`, and other `BaseException` subclasses. If the background thread is killed by any of these, the Run record remains in running status permanently -- a stuck ghost run that blocks the session from executing new pipelines (B6 constraint).

**Solution:** Two layers of protection:

**Layer 1 -- try/except BaseException/finally.** The `_run_pipeline()` body is wrapped in `try/except BaseException as exc`. The except clause calls `session_service.update_run_status(run_id, "failed", error=str(exc))` to record the failure, then re-raises. The finally clause unconditionally removes the shutdown Event from `self._shutdown_events`, preventing a resource leak.

**Layer 2 -- future.add_done_callback().** After submitting `_run_pipeline()` to the thread pool, `execute()` attaches `_on_pipeline_done` via `future.add_done_callback()`. This callback fires when the Future completes, regardless of how it completed. It calls `future.exception()` and logs any non-None result. This is purely diagnostic -- the try/finally in Layer 1 should have already handled status updates -- but it provides a safety net against scenarios where the try/finally itself fails (e.g., the `_update_run_status()` call in the except clause raises).

**Ordering guarantee:** `add_done_callback()` runs in the thread that completed the Future (the worker thread), or in the thread that called `add_done_callback()` if the Future was already completed. Since `execute()` calls `add_done_callback()` immediately after `submit()`, and the pipeline takes non-trivial time to start, the callback will consistently run in the worker thread.

### B8 Fix: Async SessionService Calls From Background Thread

**Problem:** `SessionServiceImpl` uses an async-compatible SQLAlchemy engine (aiosqlite for SQLite dev, asyncpg for Postgres prod). All `SessionServiceProtocol` methods are `async def`. `_run_pipeline()` runs in a synchronous `ThreadPoolExecutor` worker thread with no running event loop. Calling an async method directly from this thread would either: (a) create a coroutine that is never awaited (producing a `RuntimeWarning` and silently doing nothing), or (b) raise `RuntimeError: no running event loop` if the code attempts `asyncio.get_event_loop().run_until_complete()`.

**Consequence if unfixed:** If `session_service.update_run_status()` silently no-ops, the Run record stays in `"running"` permanently after any pipeline failure. The B6 constraint blocks all future executions on that session. The WebSocket never gets a terminal event. The ProgressBroadcaster leaks a subscriber queue. Recovery requires direct database manipulation.

**Solution:** `ExecutionServiceImpl` captures the main event loop reference (`asyncio.AbstractEventLoop`) at construction time — the same loop used by the ProgressBroadcaster (obtained via `asyncio.get_running_loop()` inside the FastAPI lifespan). A private `_call_async(coro)` helper bridges all async calls using `asyncio.run_coroutine_threadsafe(coro, self._loop).result()`. This blocks the background thread until the coroutine completes on the main event loop. All SessionService calls in `_run_pipeline()` use this helper.

**Invariant:** `_run_pipeline()` never calls an async method without `_call_async()`. Direct coroutine creation (calling an `async def` without `await` or `run_coroutine_threadsafe`) is a silent correctness failure — Python emits only a `RuntimeWarning`, not an exception.

---

## ProgressBroadcaster

The ProgressBroadcaster is the in-process event distribution mechanism. It bridges the synchronous background thread (where the Orchestrator runs) to the async WebSocket handlers (where clients receive events).

### Data Structure

The broadcaster maintains a `dict[str, set[asyncio.Queue[RunEvent]]]` mapping run IDs to sets of subscriber queues. Each WebSocket connection creates one queue via `subscribe()` and removes it via `unsubscribe()` on disconnect.

### Lifecycle

**Construction:** Created inside the FastAPI lifespan async context manager with `asyncio.get_running_loop()`. Stored as application state and injected into both the ExecutionServiceImpl and the WebSocket route handler.

**subscribe(run_id: str) -> asyncio.Queue[RunEvent]:** Creates a new `asyncio.Queue`, adds it to the subscriber set for the given run_id, and returns it. Called from the WebSocket handler when a client connects.

**unsubscribe(run_id: str, queue: asyncio.Queue[RunEvent]) -> None:** Removes the queue from the subscriber set. Called from the WebSocket handler on disconnect (in a finally block to ensure cleanup).

**broadcast(run_id: str, event: RunEvent) -> None:** Iterates over all subscriber queues for the run_id and schedules `queue.put_nowait(event)` via `self._loop.call_soon_threadsafe()`. This is the only method called from the background thread. It is safe to call with no subscribers -- the iteration is over an empty set.

### Cleanup

When a run completes, fails, or is cancelled, the execution service broadcasts a terminal RunEvent (event_type "completed", "error", or "cancelled"). The WebSocket handler receives this event, forwards it to the client, and disconnects. The `unsubscribe()` call in the handler's finally block removes the queue. If no clients are connected, the subscriber set for that run_id becomes empty and is eventually cleaned up.

---

## Cancel Mechanism

### Active Run Cancellation

When `cancel(run_id)` is called and the run is actively executing:

1. Look up the `threading.Event` in `self._shutdown_events[run_id]`.
2. Call `event.set()`.
3. The Orchestrator checks `shutdown_event.is_set()` during row processing. When it detects the set Event, it stops processing and returns a RunResult with appropriate status.
4. The `_run_pipeline()` function receives the result, calls `session_service.update_run_status(run_id, "cancelled")`, and broadcasts a terminal `cancelled` RunEvent.

### Pending Run Cancellation

When `cancel(run_id)` is called and the run is pending (submitted to the thread pool but the worker has not started `_run_pipeline()` yet):

1. The shutdown Event exists in `self._shutdown_events` but `_run_pipeline()` has not been entered.
2. `cancel()` sets the Event.
3. When `_run_pipeline()` eventually starts, it updates status to running, then the Orchestrator immediately detects the set Event and terminates.
4. Alternatively, `cancel()` can update the Run status to cancelled directly. When `_run_pipeline()` starts and checks the status, it sees cancelled and exits without constructing the Orchestrator.

The implementation uses the Event-based approach for both cases (set the Event, let the pipeline detect it) because it avoids a race condition between the cancel and the pipeline start. The Event is checked before row processing begins, so a pre-set Event results in immediate termination.

### Cancel Idempotency

Calling `cancel()` on an already-completed, already-failed, or already-cancelled run is a no-op. The function checks the current Run status before attempting cancellation. Setting an already-set Event is also safe (it remains set).

---

## WebSocket Progress

### Endpoint

`WS /ws/runs/{run_id}?token=<jwt>` -- authenticated WebSocket endpoint that streams RunEvent payloads for a specific run. Authentication is via query parameter: the client passes the JWT as the `token` query parameter when opening the connection.

### Connection Lifecycle

1. Client opens WebSocket connection to `/ws/runs/{run_id}?token=<jwt>`.
2. The handler extracts the token from the query string and validates it via the same AuthProvider protocol used by REST endpoints. If the token is missing or invalid, the handler closes the WebSocket with code 4001 (custom close code for auth failure). The client MUST NOT auto-reconnect on 4001 close codes -- the token must be refreshed or the user must re-authenticate.
3. The handler verifies the run exists and belongs to the authenticated user's session (IDOR prevention -- returns 404-equivalent close if not found or foreign-owned, W5 related).
4. The handler calls `broadcaster.subscribe(run_id)` to get an asyncio.Queue.
5. The handler enters a read loop: `event = await queue.get()`, serialize event to JSON, `await websocket.send_json(event_dict)`.
6. On terminal event (event_type "completed", "error", or "cancelled"), the handler sends the event and closes the connection.
7. On client disconnect (WebSocketDisconnect exception), the handler exits the loop.
8. In a finally block, the handler calls `broadcaster.unsubscribe(run_id, queue)`.

### Late Joins

If a client connects after a run has already started, they receive only events from the point of connection onward. Prior events are not replayed. This is a known v1 limitation (W14). The client can call `GET /api/runs/{run_id}` to get the current cumulative status (rows_processed, rows_failed) and then subscribe for incremental updates.

### Disconnection Handling

If the WebSocket connection drops, the queue remains subscribed until the finally block runs. Events accumulate in the queue but are never consumed. The queue is bounded only by memory. For the MVP (single-user, single run), this is acceptable. The subscriber set cleanup after run completion prevents unbounded growth across runs.

---

## REST API

### POST /api/sessions/{session_id}/validate

**Request:** No body. Uses the current (latest version) CompositionState for the session.

**Response (200):** `ValidationResult` -- `is_valid`, `checks`, `errors` as defined in the ValidationResult Model section.

**Errors:** 404 if session not found, no CompositionState exists, or session belongs to another user (IDOR prevention -- returning 404 rather than 403 avoids leaking the existence of other users' sessions).

**Behaviour:** The route handler calls `validate_pipeline()` via `await asyncio.get_running_loop().run_in_executor(None, validate_pipeline, state)` because the validation is synchronous and takes 1-5 seconds depending on plugin count. Running it directly would block the FastAPI event loop for the duration. The response is not cached -- each call re-validates against current state.

### POST /api/sessions/{session_id}/execute

**Request:** Optional body with `state_id` (UUID). If omitted, uses the latest CompositionState version for the session.

**Response (202):** `{ "run_id": "<uuid>" }` -- the run has been accepted and queued.

**Errors:** 404 if session not found, state_id not found, or session belongs to another user (IDOR prevention). 409 if an active run already exists on this session (RunAlreadyActiveError).

**Behaviour:** Calls `execution_service.execute()`. The run starts in pending status and transitions to running when the thread pool worker picks it up.

### GET /api/runs/{run_id}

**Request:** No body.

**Response (200):** RunStatus model -- `run_id`, `status` (pending/running/completed/failed/cancelled), `started_at`, `finished_at`, `rows_processed`, `rows_failed`, `error`, `landscape_run_id`, `pipeline_yaml`.

**Errors:** 404 if run not found or run belongs to a session owned by another user (IDOR prevention).

### POST /api/runs/{run_id}/cancel

**Request:** No body.

**Response (200):** `{ "status": "cancelled" }` or `{ "status": "<current>" }` if the run was already in a terminal state.

**Errors:** 404 if run not found or run belongs to a session owned by another user (IDOR prevention).

**Behaviour:** Calls `execution_service.cancel()`. Returns immediately -- cancellation is asynchronous (the Orchestrator detects the set Event during the next row-processing check).

### GET /api/runs/{run_id}/results

**Request:** No body.

**Response (200):** Run summary including rows_processed, rows_succeeded, rows_failed, rows_quarantined, rows_routed, rows_forked from the RunResult. Also includes landscape_run_id for cross-referencing with the ELSPETH audit trail.

**Errors:** 404 if run not found or run belongs to a session owned by another user (IDOR prevention). 409 if run is not yet in a terminal state (pending or running).

---

## RunEvent Model

RunEvent is the WebSocket payload model. It has the same shape whether forwarded in-process (v1) or via Redis Streams (later increment). Defined in `execution/schemas.py` as a Pydantic model.

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | `str` | UUID of the pipeline run |
| `timestamp` | `datetime` | When the event was generated (UTC) |
| `event_type` | `str` | One of: "progress", "error", "completed", "cancelled" |
| `data` | `dict` | Event-type-specific payload |

**Event type payloads:**

- **progress:** `{ "rows_processed": int, "rows_failed": int }` -- cumulative counts at the time of the event. Note: `estimated_total` is not available from the engine in v1. The frontend progress bar should use indeterminate mode (no `aria-valuemax`), showing only `rows_processed` and `rows_failed` counters without a percentage. If the source reports a row count (some sources know their total), it can be included in a future `start` event type.
- **error:** `{ "message": str, "node_id": str or null, "row_id": str or null }` -- an exception during processing. Non-terminal: the pipeline continues processing other rows.
- **completed:** `{ "rows_processed": int, "rows_succeeded": int, "rows_failed": int, "rows_quarantined": int, "landscape_run_id": str }` -- terminal event. The pipeline has finished.
- **cancelled:** `{ "rows_processed": int, "rows_failed": int }` -- terminal event. The pipeline was cancelled via the cancel mechanism. When `_run_pipeline()` detects the shutdown event and the orchestrator returns, it broadcasts this event. The WebSocket handler closes the connection after sending this terminal event.

The execution service constructs RunEvent instances from the Orchestrator's internal event types. The `_to_run_event()` method on ExecutionServiceImpl handles this translation. The translation is explicit -- it maps known Orchestrator event types to RunEvent fields. Unknown event types are not silently dropped; they raise a ValueError (offensive programming).

---

## Known Constraints

### W10: Multi-Worker Limitation

The ProgressBroadcaster holds subscriber queues in process memory. If uvicorn is started with `--workers N` where N > 1, each worker process has its own ProgressBroadcaster instance. A WebSocket connection to worker A will not receive events from a pipeline running in worker B.

**Mitigation for v1:** The application factory emits a startup warning via structlog if the `WEB_CONCURRENCY` environment variable is set to a value greater than 1. The warning states that WebSocket progress will not work correctly with multiple workers and recommends `WEB_CONCURRENCY=1`.

**Future fix:** Replace the in-process ProgressBroadcaster with Redis Streams. The RunEvent model is already designed for this -- same shape, different transport.

### W14: WebSocket Reconnect

If a WebSocket connection drops and the client reconnects, prior events are not replayed. The client must call `GET /api/runs/{run_id}` to get the current cumulative status. This is documented as a v1 limitation.

### W18: No Bare except Exception in Validation

The `validate_pipeline()` function catches only typed exceptions (pydantic.ValidationError, ValueError, GraphValidationError). If the engine introduces a new exception type that is not caught, validation will propagate the exception as a 500 Internal Server Error. This is correct behaviour -- it signals that the validation function needs updating, rather than silently swallowing an unexpected error.

### Temporary File for Settings Loading

`load_settings()` accepts a file path, not YAML content. Both `validate_pipeline()` and `_run_pipeline()` must write the generated YAML to a temporary file, call `load_settings(tmp_path)`, and clean up the file. This is handled with `tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)` in a try/finally block. The temp file is written, the path is passed to `load_settings()`, and the file is deleted in the finally clause.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/elspeth/web/execution/__init__.py` | Module init |
| Create | `src/elspeth/web/execution/protocol.py` | ExecutionService protocol (validate, execute, get_status, cancel) |
| Create | `src/elspeth/web/execution/schemas.py` | ValidationResult, ValidationCheck, ValidationError, RunStatus, RunEvent Pydantic models |
| Create | `src/elspeth/web/execution/validation.py` | `validate_pipeline()` -- dry-run using real engine code |
| Create | `src/elspeth/web/execution/service.py` | ExecutionServiceImpl -- thread pool, _run_pipeline, cancel, done callback |
| Create | `src/elspeth/web/execution/progress.py` | ProgressBroadcaster -- subscriber queues, loop.call_soon_threadsafe |
| Create | `src/elspeth/web/execution/routes.py` | REST endpoints + WS /ws/runs/{id} |
| Create | `tests/unit/web/execution/__init__.py` | Test package init |
| Create | `tests/unit/web/execution/test_validation.py` | Dry-run validation: valid pipeline, invalid schema, unknown plugin, structural error |
| Create | `tests/unit/web/execution/test_service.py` | Background execution: status transitions, cancel, done callback, active-run constraint |
| Create | `tests/unit/web/execution/test_progress.py` | ProgressBroadcaster: subscribe, broadcast from thread, unsubscribe cleanup |
| Create | `tests/unit/web/execution/test_routes.py` | Route tests: validate, execute, status, cancel, results |
| Create | `tests/integration/web/__init__.py` | Integration test package init |
| Create | `tests/integration/web/test_execute_pipeline.py` | End-to-end: CSV source, passthrough transform, CSV sink through web layer |

---

## Acceptance Criteria

1. **Dry-run validation uses real engine code.** `validate_pipeline()` calls `load_settings()`, `instantiate_plugins_from_config()`, `ExecutionGraph.from_plugin_instances()`, and `graph.validate()`. No parallel validation logic exists.

2. **Validation catches only typed exceptions.** No bare `except Exception` in `validate_pipeline()`. Only `pydantic.ValidationError`, `ValueError`, and `GraphValidationError` are caught and translated to ValidationResult errors. All other exceptions propagate.

3. **ValidationResult provides per-component attribution.** Each error carries a `component_id` and `component_type` when the error can be traced to a specific node in the CompositionState.

4. **Background execution does not block the event loop.** `execute()` returns a run_id immediately. The pipeline runs in a ThreadPoolExecutor worker thread.

5. **ProgressBroadcaster is thread-safe.** `broadcast()` uses `loop.call_soon_threadsafe()` to schedule queue puts from the background thread. Direct `queue.put_nowait()` from a non-asyncio thread never occurs.

6. **shutdown_event is always passed to Orchestrator.run().** No code path in `_run_pipeline()` calls `orchestrator.run()` without the shutdown_event parameter.

7. **_run_pipeline() catches BaseException, not Exception.** The except clause handles `KeyboardInterrupt`, `SystemExit`, and all other `BaseException` subclasses. The Run record reaches a terminal state on any failure.

8. **future.add_done_callback() is registered on every submitted pipeline.** The safety net callback fires even if the try/finally in `_run_pipeline()` fails.

9. **LandscapeDB and PayloadStore are constructed from WebSettings.** `_run_pipeline()` uses `self._settings.get_landscape_url()` and `self._settings.get_payload_store_path()` -- not hardcoded paths.

10. **Cancel sets the shutdown Event.** `cancel()` on an active run sets the `threading.Event`, which the Orchestrator detects during row processing. Cancel on a pending run results in immediate termination when the pipeline starts.

11. **One active run per session.** `execute()` raises `RunAlreadyActiveError` if a pending or running Run exists for the session.

12. **WebSocket streams RunEvent JSON.** Clients connecting to `WS /ws/runs/{run_id}?token=<jwt>` receive RunEvent payloads with event_type "progress", "error", "completed", or "cancelled". WebSocket auth failure closes the connection with code 4001.

13. **REST endpoints enforce ownership.** All run and session endpoints verify that the resource belongs to the authenticated user. Foreign session/run access returns 404 (not 403) to avoid leaking the existence of other users' resources.

14. **Integration test passes end-to-end.** A test creates a session, saves a CompositionState for a CSV-passthrough-CSV pipeline, validates it (is_valid=True), executes it, polls to completion, and verifies rows_processed > 0 and rows_failed == 0. The landscape_run_id links to a real audit trail entry.

15. **Multi-worker warning emitted.** If `WEB_CONCURRENCY > 1`, the application logs a startup warning about WebSocket progress limitations.

16. **Validation does not block the event loop.** The validate route handler calls `validate_pipeline()` via `await asyncio.get_running_loop().run_in_executor(None, validate_pipeline, state)`. Direct synchronous invocation from the route handler is forbidden.

17. **ExecutionService delegates Run CRUD to SessionService.** All Run record creation and status updates go through `session_service.create_run()`, `session_service.update_run_status()`, and `session_service.get_active_run()`. The ExecutionService has no direct database access for Run records.

17a. **Async SessionService calls from background thread use `_call_async()` (B8).** Every call from `_run_pipeline()` to an async SessionService method uses `asyncio.run_coroutine_threadsafe(coro, self._loop).result()` via the `_call_async()` helper. Direct coroutine creation without `_call_async()` is forbidden — it silently no-ops. A unit test verifies that `_run_pipeline()` successfully transitions Run status from `"pending"` through `"running"` to `"completed"` (or `"failed"`) when using the real async SessionService with an aiosqlite backend.

18. **Progress bar uses indeterminate mode.** `estimated_total` is not available from the engine in v1. Progress events report only `rows_processed` and `rows_failed` counters without a percentage or total.

19. **Cancelled runs broadcast a terminal event.** When a run is cancelled, `_run_pipeline()` broadcasts a `cancelled` RunEvent with `{rows_processed, rows_failed}` data. The WebSocket handler closes the connection after sending this terminal event.

20. **Cross-module integration test.** An integration test exercises the full flow: create session, save composition state, validate, execute, poll status, verify completion. This test uses real SessionService, real CatalogService, and real ExecutionService wired together.
