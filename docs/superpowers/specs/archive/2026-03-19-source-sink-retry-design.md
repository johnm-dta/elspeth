# Source and Sink I/O Retry — Design Spec

**Date:** 2026-03-19
**Status:** Reviewed (R2 — 4-agent peer review findings resolved: never-retry guard ordering made non-overridable, lifecycle failure semantics defined, null-byte separator replaced with length-prefixed encoding, aggregation/LLM DAG warnings added, Retry-After header forwarding and rate-limit circuit breaker added, processed_keys memory bound added, retry_pass audit traceability added, integration tests restructured to use production assembly path, property tests added, RetryableError protocol scoped to elspeth-owned exceptions)
**Scope:** Composition-based retry wrappers for source and sink I/O operations, with content-based key tracking for source retry skip

## Overview

Add transient failure retry to source and sink I/O operations. Currently only transforms have retry support (per-row, via `RetryManager`). Sources and sinks propagate all exceptions to the orchestrator, killing the run on any transient failure — token expiry, rate limiting, network blips.

Two composition-based wrappers (`SourceWrap`, `SinkWrap`) intercept source `load()` and sink `write()` calls, retrying on transient failures with exponential backoff. The orchestrator sees the same `SourceProtocol` / `SinkProtocol` interfaces — retry is invisible to it.

For source retry, content-based key tracking avoids re-processing rows that already entered the pipeline. The source declares which field(s) uniquely identify a row (e.g., `contactid` for Dataverse). On retry, the wrapper re-fetches all pages from the external API but skips rows whose key was already yielded to the orchestrator. This is safe for eventually-consistent APIs because it matches on content identity, not positional index.

## Motivation

Any source or sink that calls an external API faces transient failures:
- OAuth2 token expiry mid-pagination (Azure credential TTL: 60-75 minutes)
- Rate limiting (429 with `Retry-After`)
- Transient server errors (500-599)
- Network timeouts and connection resets

Without retry, a pipeline processing 50,000 Dataverse records that hits a 401 on page 47 loses all progress. The operator must manually re-run from scratch. With retry, the wrapper reconstructs the credential, re-paginates, skips the 46 pages of already-processed rows by content key, and resumes from page 47's data.

### Why Not Positional Skip?

An earlier design considered "skip to row N" — count how many rows were yielded, re-paginate, and discard the first N rows from the retry pass. Expert review rejected this for eventually-consistent APIs (Dataverse, REST endpoints):

- **Insert between passes:** A record inserted before row 200 shifts all subsequent positions. Row 488 in the retry pass is a different entity than row 488 in the original pass. Silent data loss.
- **Delete between passes:** A deleted record shifts positions backward. The engine skips past rows it never processed. Silent data loss.
- **No snapshot isolation:** OData pagination with `@odata.nextLink` does not provide snapshot isolation. Each page request is a new query. `$orderby` on an immutable key doesn't help — inserts and deletes still shift page boundaries.

Content-based key matching is immune to all three scenarios because it matches on what the row IS, not where it appears.

Positional skip IS safe for static file sources (CSV, JSON) where the file doesn't change between reads. The wrapper degrades gracefully: sources that don't declare key fields get full re-run retry (safe but slower).

## Architecture

### Wrapper Pattern

```text
Orchestrator
    │
    ├── source_iterator = source.load(ctx)     ← unchanged
    │        │
    │        └── SourceWrap.load(ctx)
    │              │
    │              ├── inner_source.load(ctx)  → yields rows
    │              ├── tracks processed keys per row (up to max_processed_keys)
    │              ├── on transient error:
    │              │     ├── record retry decision (with retry_pass counter)
    │              │     ├── backoff (Retry-After override if 429, circuit breaker if sustained)
    │              │     ├── inner_source.close()       — failure swallowed
    │              │     ├── inner_source.on_start()    — failure consumes attempt
    │              │     ├── inner_source.load(ctx)     → fresh generator
    │              │     ├── skip rows with known keys (or yield all if non-keyed)
    │              │     └── resume yielding new rows
    │              └── from orchestrator's view: iterator just continues
    │
    └── sink.write(rows, ctx)                  ← unchanged
              │
              └── SinkWrap.write(rows, ctx)
                    │
                    ├── inner_sink.write(rows, ctx)
                    ├── on transient error:
                    │     ├── Stage 1 guard: never-retry types → re-raise immediately
                    │     ├── guard: inner_sink.idempotent must be True
                    │     ├── backoff (Retry-After override if 429)
                    │     └── inner_sink.write(rows, ctx)  → retry same batch
                    └── return ArtifactDescriptor
```

### Layer Placement

Both wrappers live at L2 (engine) in a new `engine/retryable_io.py`. They compose L3 plugins but don't depend on any specific plugin. They satisfy `SourceProtocol` / `SinkProtocol` via delegation.

### Instantiation

The orchestrator wraps sources and sinks during run setup, after `instantiate_plugins_from_config()` returns. Wrapping is conditional:

- Source wrapped if `source.supports_retry is True` and `io_retry.enabled is True`
- Sink wrapped if `sink.supports_retry is True` and `io_retry.enabled is True`

Sources that declare `supports_retry = False` (e.g., a future destructive streaming source) are never wrapped. This is an explicit opt-out — the default is `True`.

## Wrapper Lifecycle Delegation

Both wrappers satisfy `SourceProtocol` / `SinkProtocol` by delegating all methods and properties to the inner plugin. Only `load()` (SourceWrap) and `write()` (SinkWrap) contain retry logic — everything else is pass-through.

### SourceWrap Delegation

| Member | Delegation |
|---|---|
| `name` | `self._inner.name` |
| `determinism` | `self._inner.determinism` |
| `output_schema` | `self._inner.output_schema` |
| `node_id` | `self._inner.node_id` (set by orchestrator) |
| `plugin_version` | `self._inner.plugin_version` |
| `supports_resume` | `self._inner.supports_resume` |
| `on_start(ctx)` | Delegates to `self._inner.on_start(ctx)` AND captures `ctx` for retry reconstruction |
| `on_complete(ctx)` | Delegates to `self._inner.on_complete(ctx)` |
| `close()` | Delegates to `self._inner.close()` |
| `get_field_resolution()` | `self._inner.get_field_resolution()` |
| `get_schema_contract()` | `self._inner.get_schema_contract()` |
| `set_schema_contract(c)` | `self._inner.set_schema_contract(c)` |
| `load(ctx)` | **Intercepted** — retry + key tracking logic |

### SinkWrap Delegation

| Member | Delegation |
|---|---|
| `name` | `self._inner.name` |
| `determinism` | `self._inner.determinism` |
| `input_schema` | `self._inner.input_schema` |
| `node_id` | `self._inner.node_id` |
| `plugin_version` | `self._inner.plugin_version` |
| `idempotent` | `self._inner.idempotent` (queried at retry decision time) |
| `supports_resume` | `self._inner.supports_resume` |
| `on_start(ctx)` | Delegates to `self._inner.on_start(ctx)` AND captures `ctx` |
| `on_complete(ctx)` | Delegates to `self._inner.on_complete(ctx)` |
| `close()` | Delegates to `self._inner.close()` |
| `flush()` | Delegates to `self._inner.flush()` |
| `write(rows, ctx)` | **Intercepted** — retry + idempotency guard |

The delegation is mechanical — a `__getattr__` fallback to `self._inner` could replace explicit delegation, but explicit delegation is preferred for protocol compliance visibility and mypy satisfaction.

## SourceWrap

### Normal Flow

```python
def load(self, ctx: SourceContext) -> Iterator[SourceRow]:
    gen = self._inner.load(ctx)
    for source_row in gen:
        if not source_row.is_quarantined and self._key_fields is not None:
            key = self._extract_key(source_row.row)
            self._processed_keys.add(key)
        yield source_row
```

### Retry Flow

When the generator raises a retryable exception:

1. Record retry event via `ctx.record_call()` (audit: the retry decision is recorded)
2. Backoff via `RetryManager` (exponential with jitter, or `Retry-After` override — see below)
3. Close the inner source (`inner.close()`) — see lifecycle failure handling below
4. Re-initialize the inner source (`inner.on_start(lifecycle_ctx)`) — reconstructs client, credential, etc.
5. Get fresh generator (`inner.load(ctx)`)
6. Consume rows from fresh generator:
   - If `key_fields` declared: extract key, skip if key in `processed_keys`, yield if new
   - If `key_fields` is `None`: yield all rows (full re-run)
7. Continue until generator exhausts or another transient failure triggers another retry

### Lifecycle Failure During Retry

The `close()` + `on_start()` + `load()` cycle on retry can itself fail. Each lifecycle step has defined failure semantics:

| Step | Failure Behavior | Rationale |
|---|---|---|
| `close()` | **Swallowed** — logged at WARNING, retry continues to `on_start()` | `close()` is cleanup. If the source is already in a broken state (which triggered the retry), `close()` may fail too. Propagating would prevent the retry from even attempting reconstruction. Resource leaks are acceptable for the duration of a single run — the process exits and the OS reclaims. |
| `on_start()` | **Consumes one retry attempt** — recorded as a failed attempt via `record_call()`, backoff applied, then retry loop continues to the next attempt | `on_start()` failure means reconstruction failed (e.g., credential service is also down). This is a genuine transient failure that merits backoff and retry. It consumes an attempt because each attempt should represent one full reconstruction cycle. |
| `load()` (getting fresh generator) | **Consumes one retry attempt** — same as `on_start()` failure | If the source can reconstruct but can't start iterating, that's a transient issue too. |

**The wrapper's retry loop is therefore:** for each attempt, try the full `close()` → `on_start()` → `load()` → consume sequence. If any step after `close()` fails with a retryable error, that attempt is spent. If it fails with a never-retry error, propagate immediately (Stage 1 guard still applies).

```python
# Pseudocode for lifecycle failure handling
for attempt in range(remaining_attempts):
    backoff()
    try:
        self._inner.close()
    except Exception:
        logger.warning("close() failed during retry, continuing reconstruction")
    try:
        self._inner.on_start(self._lifecycle_ctx)
        gen = self._inner.load(ctx)
    except NEVER_RETRY_TYPES:
        raise  # Stage 1: unconditional
    except Exception as exc:
        if not self._is_retryable(exc):
            raise
        record_call(attempt=attempt, reason="reconstruction_failed", error=str(exc))
        continue  # Spend this attempt, try again
    # Success — consume from gen with key skip logic
    ...
```

### Python Generator Limitation

Python generators close permanently after raising an exception — `next()` returns `StopIteration` forever after. Source retry therefore requires calling `source.load()` again to get a fresh generator. This is why the wrapper calls `close()` + `on_start()` + `load()` on retry — the inner source must be fully reconstructed.

### Key Tracking

The wrapper maintains a `set[str]` of processed keys in memory during the run. Key extraction:

- **Single-field key:** `repr(row[field])` — uses `repr()` not `str()` to distinguish `None` from `"None"`, `42` from `"42"`, etc.
- **Composite key:** `"\x00".join(f"{len(repr(v))}:{repr(v)}" for v in (row[f] for f in fields))` — length-prefixed encoding makes collisions impossible regardless of field content, including values containing null bytes. The length prefix prevents ambiguous splits: `("a\x00b", "c")` produces `"6:'a\\x00b'\x005:'c'"` while `("a", "b\x00c")` produces `"3:'a'\x006:'b\\x00c'"` — distinct keys.
- **Quarantined rows:** Skipped for key tracking (may lack key fields). Always yielded regardless of retry state — they need to reach the quarantine sink.
- **Size:** For 100K rows with UUID keys (36 chars + repr overhead ~2 chars), the set is ~4MB. Acceptable for in-memory storage during a single run.
- **Memory bound:** The wrapper enforces a `max_processed_keys` limit (default: 1,000,000). When exceeded, the wrapper clears the key set and falls back to full re-run semantics for the remainder of the run. A WARNING-level log is emitted (last-resort channel — not an audit event, not a telemetry event, because this is a resource pressure signal with no probative value). The fallback is safe: full re-run produces duplicates but no data loss. Operators processing datasets larger than 1M rows should declare `source_key_fields` for content-based skip efficiency, or accept the duplicate-on-retry cost.

Keys are NOT checkpointed to disk. Source retry is within a single run attempt. If all retries exhaust, the run fails and `elspeth resume` handles recovery through the existing checkpoint/resume mechanism (which bypasses the source entirely, restoring rows from payload store).

### Non-Keyed Retry (Full Re-Run) Semantics

When `source_key_fields` is `None`, the wrapper retries by yielding ALL rows from the fresh generator — including rows that were already processed in the first pass. This produces duplicate rows entering the orchestrator. The implications:

1. **Static file sources** (CSV, JSON, AzureBlob) return identical data on re-read. The same rows enter the pipeline with the same content. Transforms re-execute and sinks re-write. For idempotent sinks, this is safe. For non-idempotent sinks, the SinkWrap refuses to retry — so the run fails and the operator re-runs manually.
2. **Row identity:** Each duplicate row gets a NEW `row_id` (UUID generated by the engine at token creation). The audit trail correctly shows two separate processing events for the same source content. This is auditably honest — both events happened. A `retry_pass` field in the retry decision `record_call` allows auditors to correlate which rows were yielded during which pass (see Audit Trail Recording).
3. **Transform cost:** Re-executing transforms (especially LLM transforms) on already-processed rows costs real money. Content-based key tracking (by declaring `source_key_fields`) is the solution for sources where this cost matters.
4. **Aggregation correctness hazard:** Duplicate rows entering an aggregation transform inflate counts, trigger thresholds early, and produce incorrect aggregate results. **This is a correctness concern, not just a performance concern.** The DAG validator emits a structured WARNING at construction time when all three conditions are met: (a) source declares `source_key_fields = None`, (b) DAG contains an aggregation node, (c) `io_retry.enabled = True`. The warning message: `"Non-keyed source retry with aggregation transforms may produce incorrect aggregate results on retry. Declare source_key_fields to enable content-based skip."` The pipeline still runs — the operator accepts the risk — but the warning is recorded in the run's audit trail via `record_call()` with `provider="dag_validation"`.
5. **LLM cost warning:** Similarly, the DAG validator emits a WARNING when a non-keyed source feeds an LLM transform with retry enabled: `"Non-keyed source retry with LLM transforms will re-execute LLM calls on retry, incurring duplicate API costs."`

### Lifecycle Context Capture

The wrapper captures the `LifecycleContext` from the first `on_start()` call. On retry, it passes the same context to `inner.on_start()`. The context is run-scoped (`run_id`, `rate_limit_registry`, `telemetry_emit`) — it doesn't change between retry attempts within a run.

## SinkWrap

### Normal Flow

```python
def write(self, rows: list[dict[str, Any]], ctx: SinkContext) -> ArtifactDescriptor:
    return self._inner.write(rows, ctx)
```

### Retry Flow

When `write()` raises a retryable exception:

1. **Idempotency guard:** If `inner.idempotent is False`, raise immediately. Non-idempotent sinks cannot safely re-send the same batch. The error is recorded with `"reason": "retryable_but_non_idempotent"`.
2. Record retry event via `ctx.record_call()`
3. Backoff via `RetryManager`
4. Re-call `inner.write(rows, ctx)` with the same batch

No `close()` / `on_start()` cycle needed — the sink's client connection pool handles reconnection internally. The same `rows` list reference is passed on retry.

**Executor interaction:** The `SinkExecutor` calls `sink_wrap.write(rows, ctx)`. The wrapper retries internally. If all retries fail, the wrapper raises, and THEN the executor marks states as FAILED. The executor never sees intermediate failures — only the final outcome.

### Mode-Aware Idempotency

The DataverseSink currently declares `idempotent = False` at the class level (conservative — future create/update modes are not idempotent). The `idempotent` attribute should become a property that checks the configured write mode:

```python
@property
def idempotent(self) -> bool:
    return self._mode == "upsert"
```

This allows the SinkWrap to query actual runtime idempotency. Other sinks (CSV, JSON, Database) remain `False`.

## Retry Capability Model

**Default stance: all sources and sinks are retryable.** Plugins that cannot support retry must explicitly opt out.

### Source Properties

| Property | Type | Default | Purpose |
|---|---|---|---|
| `supports_retry` | `bool` | `True` | Whether retry is possible at all. `False` for destructive/streaming sources. |
| `source_key_fields` | `tuple[str, ...] \| None` | `None` | Fields for content-based skip. `None` = full re-run on retry. |
| `classify_retryable` | method | Returns `None` | Plugin-specific error classification override. |

### Sink Properties

| Property | Type | Default | Purpose |
|---|---|---|---|
| `supports_retry` | `bool` | `True` | Whether retry is possible at all. |
| `idempotent` | `bool` or property | `False` | Whether batch re-send is safe. SinkWrap refuses to retry if `False`. |
| `classify_retryable` | method | Returns `None` | Plugin-specific error classification override. |

### Retry Behavior Matrix

| Source | `supports_retry` | `source_key_fields` | Retry Behavior |
|---|---|---|---|
| DataverseSource (structured) | `True` (default) | `("<entity>id",)` inferred | Content-based skip on retry |
| DataverseSource (FetchXML) | `True` (default) | From config or `None` | Content-based skip or full re-run |
| CSVSource | `True` (default) | `None` | Full re-run (file is static) |
| AzureBlobSource | `True` (default) | `None` | Full re-run |
| JSONSource | `True` (default) | `None` | Full re-run |
| Future streaming source | `False` (explicit) | N/A | No retry |

| Sink | `supports_retry` | `idempotent` | Retry Behavior |
|---|---|---|---|
| DataverseSink (upsert) | `True` (default) | `True` (mode-aware) | Batch re-send |
| DataverseSink (future create) | `True` (default) | `False` (mode-aware) | Retry refused |
| CSVSink | `True` (default) | `False` | Retry refused |
| JSONSink | `True` (default) | `False` | Retry refused |
| DatabaseSink | `True` (default) | `False` | Retry refused |

## Error Classification

### Classification Ordering

Error classification follows a strict three-stage pipeline. The ordering is non-negotiable — the never-retry guard MUST execute first, unconditionally, before any plugin override or default predicate:

```text
Exception raised
    │
    ├─ Stage 1: NEVER-RETRY GUARD (non-overridable)
    │   Is exc an instance of a never-retry type?
    │   Yes → re-raise immediately, no retry, no classify_retryable consultation
    │
    ├─ Stage 2: PLUGIN OVERRIDE
    │   inner.classify_retryable(exc) returns True/False/None?
    │   True → retryable    False → not retryable    None → continue to Stage 3
    │
    └─ Stage 3: DEFAULT PREDICATE
        RetryableError protocol check, then type-based fallback
```

**Why Stage 1 is non-overridable:** A plugin `classify_retryable()` returning `True` for `FrameworkBugError` would retry on a framework integrity violation. This must be structurally impossible, not just documented. The never-retry guard is a `try/except` that catches specific types before the general `Exception` handler — `classify_retryable()` is never called for these types.

### Never Retried (Stage 1)

- `FrameworkBugError`, `AuditIntegrityError` — system integrity, always re-raised immediately
- `KeyError`, `TypeError`, `AttributeError`, `NameError` — programming bugs
- `PluginConfigError` — configuration problem

These are caught in a dedicated `except` clause that re-raises before the general `except Exception` handler. No predicate, no override, no retry — unconditionally.

### Default Predicate (Stage 3)

```python
from elspeth.contracts.errors import RetryableError  # New protocol

@runtime_checkable
class RetryableError(Protocol):
    """Protocol for exceptions that carry retryability classification."""
    @property
    def retryable(self) -> bool: ...

def _default_is_retryable(exc: BaseException) -> bool:
    # Protocol: system-owned exceptions declare retryability via RetryableError
    if isinstance(exc, RetryableError):
        return exc.retryable

    # Type-based fallback for third-party exceptions without the protocol
    return isinstance(exc, (
        httpx.TimeoutException,
        httpx.ConnectError,
        ConnectionError,
        TimeoutError,
    ))
```

**Why `isinstance(exc, RetryableError)` not `getattr`:** Per CLAUDE.md, `getattr` with defaults is banned. System-owned exceptions (`DataverseClientError`, `LLMClientError`) satisfy the `RetryableError` protocol structurally — `isinstance` with `@runtime_checkable` checks protocol conformance without `getattr`. Third-party exceptions (e.g., `azure.identity` errors) that don't satisfy the protocol fall through to the type-based branch.

**Protocol scope:** The `RetryableError` protocol match is restricted to `elspeth`-owned exceptions. The wrapper checks `type(exc).__module__.startswith("elspeth.")` before trusting the protocol match. Third-party exceptions that accidentally satisfy the protocol structurally (e.g., an `azure.identity` exception with a `.retryable` attribute) are not trusted — they fall through to the type-based branch. This prevents accidental structural matches from forcing retries on non-transient errors.

### Plugin Override (Stage 2)

Sources and sinks can optionally provide their own classification:

```python
def classify_retryable(self, exc: BaseException) -> bool | None:
    """Returns True/False for definitive classification, None for default."""
    return None
```

The wrapper checks `inner.classify_retryable(exc)` after the never-retry guard but before the default predicate. If it returns `None`, the default predicate applies. The plugin override **cannot override the never-retry guard** — Stage 1 exceptions never reach Stage 2.

## Configuration

### Settings Model

```python
class IORetrySettings(BaseModel):
    """Retry configuration for source and sink I/O operations."""
    max_attempts: int = Field(default=3, gt=0)
    initial_delay_seconds: float = Field(default=1.0, gt=0)
    max_delay_seconds: float = Field(default=60.0, gt=0)
    exponential_base: float = Field(default=2.0, gt=1.0)
    max_processed_keys: int = Field(default=1_000_000, gt=0)
    retry_after_consecutive_429s: int = Field(default=2, gt=0)
    enabled: bool = Field(default=True)
```

### Pipeline YAML

```yaml
io_retry:
  max_attempts: 3
  initial_delay_seconds: 1.0
  max_delay_seconds: 60.0
  max_processed_keys: 1000000       # Memory bound for key tracking; fallback to full re-run when exceeded
  retry_after_consecutive_429s: 2   # Switch to Retry-After-only mode after N consecutive 429s

# OR disabled:
io_retry:
  enabled: false
```

Single config section applies to all sources and sinks. Per-plugin overrides deferred to a future iteration.

`IORetrySettings` nests under the top-level `PipelineSettings` alongside the existing `RetrySettings` (which governs transform retry). In `core/config.py`:

```python
class PipelineSettings(BaseModel):
    retry: RetrySettings = Field(default_factory=RetrySettings)       # transforms
    io_retry: IORetrySettings = Field(default_factory=IORetrySettings) # sources/sinks
    # ... existing fields
```

### Runtime Mapping

```python
@dataclass(frozen=True, slots=True)
class RuntimeIORetryConfig:
    max_attempts: int
    base_delay: float
    max_delay: float
    jitter: float  # Required by RuntimeRetryProtocol
    exponential_base: float
    max_processed_keys: int
    retry_after_consecutive_429s: int
    enabled: bool

    @classmethod
    def from_settings(cls, settings: IORetrySettings) -> RuntimeIORetryConfig:
        return cls(
            max_attempts=settings.max_attempts,
            base_delay=settings.initial_delay_seconds,
            max_delay=settings.max_delay_seconds,
            jitter=INTERNAL_DEFAULTS["retry"]["jitter"],  # 1.0 — not user-configurable
            exponential_base=settings.exponential_base,
            max_processed_keys=settings.max_processed_keys,
            retry_after_consecutive_429s=settings.retry_after_consecutive_429s,
            enabled=settings.enabled,
        )
```

This satisfies `RuntimeRetryProtocol` (including the required `jitter` property) — reuses `RetryManager` internally. The `jitter` field uses the same hardcoded internal default as `RuntimeRetryConfig`.

### Config Flow

```text
YAML → IORetrySettings (Pydantic) → RuntimeIORetryConfig (frozen dataclass)
  → stored in PipelineConfig
  → passed to SourceWrap/SinkWrap constructors
  → wrappers create RetryManager(config) internally
```

## Audit Trail Recording

### Source Retry Audit

The wrapper maintains a `retry_pass` counter (0 = first pass, 1 = first retry, etc.). This counter is included in audit records to allow auditors to distinguish original rows from retry-pass rows.

Each retry attempt produces:

1. **Retry decision record** via `ctx.record_call()`:
   - `call_type`: `CallType.HTTP` with `provider="io_retry"`
   - `status`: `CallStatus.ERROR`
   - `error`: `{"reason": "transient_auth_failure", "attempt": 2, "max_attempts": 3, "retry_pass": 1, "processed_keys_count": 100, "strategy": "content_key_skip"}`

2. **Inner source HTTP calls** recorded normally by the inner source's `record_call()` calls. On retry pass, pages 1-N appear again in the audit trail — each HTTP call actually happened.

3. **Skipped rows** are NOT recorded in `node_states`. They were already recorded during the first pass. The wrapper consumes them from the generator but doesn't yield them to the orchestrator, so no token is created. The skip decision is recorded in aggregate in the retry decision record (`processed_keys_count` field).

4. **Row index continuity.** The orchestrator's `row_index` (from `enumerate(source_iterator)`) continues monotonically. The wrapper yielded 100 rows before failure, then yields rows 100+ after retry. No gaps.

5. **Retry pass propagation for non-keyed retry.** When `source_key_fields` is `None` and retry produces duplicate rows, the `retry_pass` counter is included in each yielded `SourceRow`'s metadata (via a new `retry_pass` field on `SourceRow`, default 0). This allows auditors to query "show me all rows from retry pass 1" and identify duplicates. For keyed retry, only genuinely new rows are yielded, so `retry_pass > 0` rows are always new data (not duplicates).

### Sink Retry Audit

Each retry attempt produces:

1. **Retry decision record** via `ctx.record_call()`
2. **Re-sent rows** produce duplicate `record_call` entries — attempt 1 row A SUCCESS + attempt 2 row A SUCCESS. Both calls actually happened. Auditor can see retry and confirm idempotent behavior.

## Dataverse Integration

### 401 Credential Reconstruction

The tactical fix (move `reconstruct_credential()` inside `DataverseClient._execute_request()`) is now part of the strategic design. The client handles 401 retry internally as a protocol-level concern. The `SourceWrap` handles higher-level transient failures (429, 5xx, network errors).

This is two levels of retry:
- **Client level:** 401 → reconstruct credential → retry immediately (once, no backoff)
- **Wrapper level:** 429/5xx/network → backoff → close/on_start/load cycle → content-based skip

### Retry-After Header Forwarding

When the external API returns a 429 with `Retry-After`, the delay signal should propagate from the client exception to the wrapper's backoff logic. Without this, the wrapper uses its own exponential backoff (max 60s) which may be shorter than the API's actual rate-limit window — causing the wrapper to retry into a system that hasn't recovered yet, exhausting all attempts for nothing.

**Mechanism:** Retryable exceptions that carry rate-limit metadata expose it via an optional `retry_after_seconds` attribute:

```python
class DataverseClientError(Exception):
    def __init__(self, message: str, *, retryable: bool, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
```

The wrapper's backoff logic checks for `retry_after_seconds` on the caught exception:

```python
def _compute_delay(self, exc: BaseException, attempt: int) -> float:
    # If the exception carries a server-specified delay, use it as the floor
    server_delay = getattr(exc, "retry_after_seconds", None)  # Note: getattr OK here —
    # this is a Tier 3 trust boundary (external signal on exception), not defensive access
    # on a typed internal object. The exception may or may not carry this attribute.
    exponential_delay = min(
        self._config.base_delay * (self._config.exponential_base ** attempt),
        self._config.max_delay,
    )
    if server_delay is not None and server_delay > 0:
        return max(server_delay, exponential_delay)
    return exponential_delay
```

**Why `max(server_delay, exponential_delay)` not just `server_delay`:** The server's Retry-After may be unreasonably short (0.1s) as a politeness hint. Exponential backoff provides a minimum floor. Conversely, if the server says 120s and our backoff says 4s, we respect the server's signal — it knows its own recovery timeline.

**No `max_delay` cap on server-specified delays.** If the server says `Retry-After: 300`, we wait 300 seconds. The `max_delay_seconds` config only caps our own exponential calculation. Rationale: the server's signal is authoritative for its own state. If 300s is too long for the operator, `max_attempts` will exhaust and the run will fail — which is the correct outcome (the operator can then resume later).

### Rate-Limit Circuit Breaker

For sustained rate-limit scenarios, the wrapper applies a simple circuit breaker heuristic: if the last N consecutive errors were all 429s (where N is configurable, default 2), the wrapper switches to **Retry-After-only mode** — it ignores exponential backoff entirely and waits exactly what the server specifies. If no `Retry-After` header is present, it waits `max_delay_seconds`.

This prevents the "Fixes that Fail" archetype identified in systems review: without the circuit breaker, the wrapper's shorter-than-recovery-window delays cause it to exhaust all attempts before the rate limit clears.

### Source Key Fields

Dataverse entities always have a GUID primary key named `<entitylogicalname>id`:

```python
# DataverseSource (structured mode)
@property
def source_key_fields(self) -> tuple[str, ...] | None:
    if self._entity is not None:
        return (f"{self._entity}id",)
    # FetchXML: use config-specified key field, or None for full re-run
    return (self._source_key_field,) if self._source_key_field else None
```

Config for FetchXML mode:

```yaml
source:
  plugin: dataverse
  options:
    fetch_xml: |
      <fetch>...</fetch>
    source_key_field: contactid   # explicit when inference isn't possible
```

### Sink Idempotency

DataverseSink changes `idempotent` from a class attribute to a mode-aware property:

```python
@property
def idempotent(self) -> bool:
    return self._mode == "upsert"
```

## File Inventory

### New Files

| File | Purpose |
|---|---|
| `engine/retryable_io.py` | `SourceWrap`, `SinkWrap`, default retryable predicate |

### Config Pipeline (5 files)

| File | Change |
|---|---|
| `core/config.py` | Add `IORetrySettings` |
| `contracts/config/protocols.py` | Add `RuntimeIORetryProtocol` |
| `contracts/config/runtime.py` | Add `RuntimeIORetryConfig` with `from_settings()` |
| `contracts/config/alignment.py` | Add field mapping documentation |
| `contracts/config/defaults.py` | Add `POLICY_DEFAULTS` for io_retry |

### Plugin Protocol + Base (3 files)

| File | Change |
|---|---|
| `contracts/plugin_protocols.py` | Add `source_key_fields`, `supports_retry`, `classify_retryable` to both protocols |
| `plugins/infrastructure/base.py` | Add default implementations to `BaseSource` and `BaseSink` |
| `plugins/sinks/dataverse.py` | Override `idempotent` as mode-aware property |

### Dataverse (3 files)

| File | Change |
|---|---|
| `plugins/sources/dataverse.py` | Declare `source_key_fields` property; add `source_key_field` to `DataverseSourceConfig` for FetchXML mode |
| `plugins/infrastructure/clients/dataverse.py` | Move 401 retry into `_execute_request`; delete public `reconstruct_credential()` |
| `contracts/errors.py` | Add `RetryableError` runtime-checkable protocol |

### Engine (2 files)

| File | Change |
|---|---|
| `engine/orchestrator/core.py` | Wrap source/sinks in `SourceWrap`/`SinkWrap` during run setup (~10 lines) |
| `engine/orchestrator/types.py` | Add `io_retry_config` to `PipelineConfig` |

### CI/Enforcement (3 files)

| File | Change |
|---|---|
| `config/cicd/enforce_tier_model/` | Allowlist entries for `retryable_io.py` |
| `config/cicd/contracts-whitelist.yaml` | Dict pattern entries for wrapper methods |
| `tests/unit/core/test_config_alignment.py` | Alignment test for new settings→runtime mapping |

### Contracts (1 file)

| File | Change |
|---|---|
| `contracts/types.py` or `contracts/source_types.py` | Add `retry_pass: int = 0` field to `SourceRow` (or equivalent metadata carrier) |

### Tests (4 files)

| File | Coverage |
|---|---|
| `tests/unit/engine/test_retryable_io.py` | SourceWrap + SinkWrap unit tests (core flows, edge cases, lifecycle failures, classification ordering, Retry-After, circuit breaker, memory bound) |
| `tests/integration/engine/test_retryable_io.py` | Production assembly path tests through orchestrator, Landscape audit queries, DAG validation warnings |
| `tests/property/engine/test_retryable_io_properties.py` | Key extraction collision resistance, retry invariants, error classification properties |
| Update existing Dataverse test files | Key fields, idempotency property, removed reconstruct_credential |

### Unchanged

`RetryManager`, `RowProcessor`, `SinkExecutor`, `TransformExecutor`, DAG construction, Landscape schema, checkpoint schema, Alembic migrations, all existing source plugins except Dataverse.

## Testing Strategy

### Unit Tests (`test_retryable_io.py`)

**Core flows:**
- SourceWrap: normal flow (no failure, all rows yielded)
- SourceWrap: single retry with content-based skip (key match)
- SourceWrap: single retry without key fields (full re-run)
- SourceWrap: max retries exhausted → exception propagates
- SourceWrap: non-retryable exception → immediate propagation
- SourceWrap: quarantined rows always yielded (not key-tracked)
- SourceWrap: `supports_retry=False` source → not wrapped
- SourceWrap: `FrameworkBugError` never caught (Stage 1 guard)
- SinkWrap: normal flow
- SinkWrap: retry with idempotent sink
- SinkWrap: retry refused for non-idempotent sink
- SinkWrap: max retries exhausted
- SinkWrap: `classify_retryable` override respected

**Generator edge cases (from QA review):**
- SourceWrap: empty generator (zero rows, no failure) → yields nothing, key set empty
- SourceWrap: failure on first row (zero rows yielded before exception) → retry with empty key set, all rows are new
- SourceWrap: failure on last row (all-but-one yielded) → retry skips all but last
- SourceWrap: failure during retry pass (cascaded failure) → attempt count tracks correctly across cascades
- SourceWrap: quarantined row in first pass, same row quarantined again in retry pass → yielded both times (not key-tracked)
- SinkWrap: empty batch (`rows=[]`) → delegates correctly, retries on failure

**Lifecycle failure handling (from architecture + QA review):**
- SourceWrap: `close()` raises during retry → swallowed, retry continues to `on_start()`
- SourceWrap: `on_start()` raises retryable error during retry → consumes one attempt, continues
- SourceWrap: `on_start()` raises never-retry error during retry → propagates immediately (Stage 1)
- SourceWrap: `load()` raises during retry → consumes one attempt

**Error classification ordering (from security review):**
- Stage 1: `FrameworkBugError` never reaches `classify_retryable()` even if plugin would return `True`
- Stage 1: `AuditIntegrityError` never reaches `classify_retryable()`
- Stage 2: `classify_retryable()` returning `True` overrides default predicate for non-Stage-1 exceptions
- Stage 3: `RetryableError` protocol only trusted for `elspeth.*` module exceptions
- Stage 3: Third-party exception with `.retryable` attribute → falls through to type-based check

**Retry-After and circuit breaker:**
- Wrapper respects `retry_after_seconds` from exception as delay floor
- Wrapper uses `max(server_delay, exponential_delay)` — server can extend but not shorten backoff
- After N consecutive 429s, wrapper switches to Retry-After-only mode
- No `max_delay` cap applied to server-specified delays

**Memory bound:**
- Key set exceeding `max_processed_keys` → set cleared, fallback to full re-run, warning logged
- Key set at limit-1 → still tracking, no fallback

**Default predicate:**
- `DataverseClientError(retryable=True)` → retryable
- `DataverseClientError(retryable=False)` → not retryable
- `httpx.TimeoutException` → retryable
- `KeyError` → not retryable (Stage 1 guard)

### Integration Tests (`tests/integration/engine/test_retryable_io.py`)

**IMPORTANT:** Integration tests MUST exercise the production assembly path. The orchestrator's wrapping logic (~10 lines in `core.py`) is the code most likely to have bugs (wrong config passed, wrapping condition inverted). Tests that manually construct `SourceWrap` are unit tests, not integration tests. See BUG-LINEAGE-01 precedent.

**Production path tests (through orchestrator run setup):**
- Orchestrator wraps source when `io_retry.enabled=True` and `supports_retry=True` → verify wrapping via type check or behavior
- Orchestrator does NOT wrap source when `io_retry.enabled=False` → verify no wrapping
- Orchestrator does NOT wrap source when `supports_retry=False` → verify no wrapping
- Full pipeline with flaky source (fails on page N), retry succeeds → verify via Landscape queries:
  - `record_call` entries with `provider="io_retry"` and correct `attempt`, `retry_pass`, `processed_keys_count`, `strategy` fields
  - Row index monotonicity (no gaps in `node_states` row indices after retry)
  - Absence of `node_states` records for skipped rows (key-based skip)
  - `retry_pass` field on `SourceRow` metadata for non-keyed retry duplicates
- Full pipeline with flaky sink, retry succeeds → verify duplicate `record_call` entries for re-sent batch
- Full pipeline with non-idempotent sink failure → verify retry refused, `record_call` with `"reason": "retryable_but_non_idempotent"`

**DAG validation warnings:**
- Non-keyed source + aggregation + `io_retry.enabled` → warning emitted and recorded
- Non-keyed source + LLM transform + `io_retry.enabled` → warning emitted
- Keyed source + aggregation → no warning

### Property Tests (`tests/property/engine/test_retryable_io_properties.py`)

**Key extraction properties (Hypothesis):**
- For any two distinct rows (where at least one key field differs), composite keys are distinct
- `repr(None)` and `repr("None")` produce distinct single-field keys
- Key extraction is deterministic (same row → same key on every call)
- Quarantined rows never appear in the processed_keys set

**Retry invariant properties:**
- `max_attempts` is always respected regardless of error type mix (retryable, non-retryable, lifecycle failures)
- Idempotent sink always receives the exact same `rows` list reference on retry
- Non-idempotent sink never gets a second `write()` call after failure
- Total rows yielded by SourceWrap with key tracking == unique rows from source (no duplicates, no omissions)

**Error classification properties:**
- Any exception satisfying `RetryableError` protocol from `elspeth.*` module with `retryable=True` → is retried
- Any Stage 1 exception is never retried regardless of any other condition
- `classify_retryable` override takes precedence over default predicate for non-Stage-1 exceptions

## Security Considerations

- **Credential reconstruction on retry:** The SourceWrap calls `inner.on_start()` which reconstructs the credential. The old credential is discarded. Token material is never stored in the retry state.
- **Processed keys in memory:** The `set[str]` of entity IDs is held in process memory only. Not persisted to disk, not logged, not included in audit trail payloads. Entity IDs are not PII but may be considered sensitive in some contexts — the set is garbage-collected when the wrapper is collected. Memory bounded by `max_processed_keys` (default 1M entries, ~40MB for UUID keys).
- **Retry amplification:** The wrapper enforces `max_attempts` strictly. A source that returns `retryable=True` on every attempt will exhaust retries and fail, not loop infinitely. Rate limiting (`_acquire_rate_limit`) applies on every HTTP call including retries. The rate-limit circuit breaker (Retry-After-only mode after consecutive 429s) prevents the wrapper from overwhelming a rate-limited endpoint.
- **Key collision resistance:** Length-prefixed `repr()` encoding ensures composite keys are injective — distinct rows always produce distinct keys, even if field values contain null bytes or the string `"None"`. The encoding is deterministic (same row → same key).
- **Error classification scoping:** The `RetryableError` protocol match is restricted to `elspeth.*` module exceptions via `type(exc).__module__` check. Third-party exceptions that accidentally satisfy the protocol structurally are not trusted — they fall through to the type-based predicate. This prevents dependency updates from silently changing retry behavior.
- **Never-retry non-overridability:** The Stage 1 exception guard (`FrameworkBugError`, `AuditIntegrityError`, programming bugs) is structurally unreachable from `classify_retryable()` plugin overrides. A plugin cannot force retries on framework integrity violations.

## Dependencies

No new dependencies. `RetryManager` (tenacity-based) is reused. `SourceWrap` and `SinkWrap` are pure Python composition with no external library requirements.
